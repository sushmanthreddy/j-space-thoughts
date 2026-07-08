from __future__ import annotations

import torch
from torch import nn

from src.interventions import (
    ablate_direction,
    clamp_swapped_coordinates,
    residual_edit_hooks,
    swap_coordinates,
)


def test_ablation_zeros_selected_coordinate_only() -> None:
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(2, 5, 7, generator=generator)
    direction = torch.randn(7, generator=generator)
    direction = direction / direction.norm()

    edited = ablate_direction(hidden, direction, positions=[1, -1])

    assert torch.allclose(edited[:, [1, 4]] @ direction, torch.zeros(2, 2), atol=1e-6)
    assert torch.equal(edited[:, [0, 2, 3]], hidden[:, [0, 2, 3]])
    assert edited.dtype == hidden.dtype


def test_scaled_ablation_interpolates_and_supports_negative_steering() -> None:
    hidden = torch.tensor([[[2.0, 3.0, 4.0]]])
    direction = torch.tensor([1.0, 0.0, 0.0])
    half = ablate_direction(hidden, direction, strength=0.5)
    doubled = ablate_direction(hidden, direction, strength=2.0)
    negative = ablate_direction(hidden, direction, strength=-1.0)
    assert torch.equal(half, torch.tensor([[[1.0, 3.0, 4.0]]]))
    assert torch.equal(doubled, torch.tensor([[[-2.0, 3.0, 4.0]]]))
    assert torch.equal(negative, torch.tensor([[[4.0, 3.0, 4.0]]]))


def test_nonorthogonal_swap_is_exact_and_preserves_orthogonal_part() -> None:
    concept = torch.tensor([1.0, 0.0, 0.0])
    foil = torch.tensor([0.5, 0.5, 0.0])
    foil = foil / foil.norm()
    hidden = torch.tensor([[[2.0, -1.0, 4.0]]])
    basis = torch.stack([concept, foil])
    original_projection = hidden.float() @ basis.T

    edited = swap_coordinates(hidden, concept, foil)
    edited_projection = edited.float() @ basis.T

    assert torch.allclose(edited_projection, original_projection.flip(-1), atol=1e-6)
    assert torch.equal(edited[..., 2], hidden[..., 2])


def test_swap_matches_paper_pseudoinverse_reference() -> None:
    generator = torch.Generator().manual_seed(37)
    hidden = torch.randn(2, 4, 9, generator=generator)
    source = torch.randn(9, generator=generator)
    target = torch.randn(9, generator=generator)
    source = source / source.norm()
    target = target / target.norm()
    # Column-basis form used in the paper: c = V^+ h and h' = h + V(Pc-c).
    columns = torch.stack([source, target], dim=1)
    coefficients = torch.einsum("kd,bsd->bsk", torch.linalg.pinv(columns), hidden)
    reference = hidden + torch.einsum(
        "dk,bsk->bsd", columns, coefficients.flip(-1) - coefficients
    )

    actual = swap_coordinates(hidden, source, target)

    assert torch.allclose(actual, reference, atol=2e-6, rtol=2e-6)


def test_clamped_swap_strength_matches_paper_coefficients() -> None:
    generator = torch.Generator().manual_seed(41)
    clean = torch.randn(1, 5, 7, generator=generator)
    current = torch.randn(1, 5, 7, generator=generator)
    source = torch.randn(7, generator=generator)
    target = torch.randn(7, generator=generator)
    source = source / source.norm()
    target = target / target.norm()
    basis = torch.stack([source, target])
    inverse_gram = torch.linalg.inv(basis @ basis.T)
    clean_coeff = (clean @ basis.T) @ inverse_gram
    current_coeff = (current @ basis.T) @ inverse_gram
    current_orthogonal = current - current_coeff @ basis

    for strength in (0.0, 0.5, 1.0, 2.0):
        edited = clamp_swapped_coordinates(
            current, clean, source, target, strength=strength
        )
        edited_coeff = (edited @ basis.T) @ inverse_gram
        desired = clean_coeff + strength * (clean_coeff.flip(-1) - clean_coeff)
        edited_orthogonal = edited - edited_coeff @ basis
        assert torch.allclose(edited_coeff, desired, atol=2e-6, rtol=2e-6)
        assert torch.allclose(
            edited_orthogonal, current_orthogonal, atol=2e-6, rtol=2e-6
        )


def test_swap_rejects_singular_pair() -> None:
    hidden = torch.randn(1, 2, 3)
    direction = torch.tensor([1.0, 0.0, 0.0])
    try:
        swap_coordinates(hidden, direction, direction)
    except ValueError as error:
        assert "ill-conditioned" in str(error)
    else:
        raise AssertionError("Singular coordinate pair was not rejected")


def test_clamped_swap_uses_clean_coefficients_and_current_orthogonal_part() -> None:
    concept = torch.tensor([1.0, 0.0, 0.0])
    foil = torch.tensor([0.5, 0.5, 0.0])
    foil = foil / foil.norm()
    clean = torch.tensor([[[2.0, -1.0, 4.0]]])
    current = clean + torch.tensor([[[0.0, 0.0, 3.0]]])
    basis = torch.stack([concept, foil])
    clean_coefficients = (clean @ basis.T) @ torch.linalg.inv(basis @ basis.T)

    edited = clamp_swapped_coordinates(current, clean, concept, foil)
    edited_coefficients = (edited @ basis.T) @ torch.linalg.inv(basis @ basis.T)

    assert torch.allclose(edited_coefficients, clean_coefficients.flip(-1), atol=1e-6)
    assert edited[..., 2].item() == current[..., 2].item()


class _TupleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extra = object()

    def forward(self, hidden: torch.Tensor):
        return hidden + 1, self.extra


def test_tuple_hook_preserves_extras_and_is_removed() -> None:
    block = _TupleBlock()
    hidden = torch.zeros(1, 1, 2)
    clean, clean_extra = block(hidden)
    with residual_edit_hooks([block], {0: lambda value: value * 3}):
        edited, edited_extra = block(hidden)
    restored, restored_extra = block(hidden)

    assert torch.equal(edited, clean * 3)
    assert edited_extra is clean_extra
    assert torch.equal(restored, clean)
    assert restored_extra is clean_extra


def test_hook_cleanup_after_exception() -> None:
    block = nn.Identity()
    try:
        with residual_edit_hooks([block], {0: lambda value: value + 5}):
            assert torch.equal(block(torch.zeros(1)), torch.tensor([5.0]))
            raise RuntimeError("intentional")
    except RuntimeError:
        pass
    assert torch.equal(block(torch.zeros(1)), torch.zeros(1))

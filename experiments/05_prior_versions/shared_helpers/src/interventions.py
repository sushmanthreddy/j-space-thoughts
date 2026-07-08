"""Real residual-stream interventions and the output-suppression control."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn


TensorEdit = Callable[[torch.Tensor], torch.Tensor]


def _resolved_positions(sequence_length: int, positions: Sequence[int] | None) -> list[int]:
    if positions is None:
        return list(range(sequence_length))
    resolved: list[int] = []
    for position in positions:
        index = int(position)
        if index < 0:
            index += sequence_length
        if not 0 <= index < sequence_length:
            raise IndexError(
                f"Position {position} outside sequence of length {sequence_length}"
            )
        resolved.append(index)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate intervention positions after resolution: {resolved}")
    return resolved


def ablate_direction(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    *,
    positions: Sequence[int] | None = None,
    strength: float = 1.0,
) -> torch.Tensor:
    """Remove a scaled unit-direction coordinate from selected positions."""

    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden shape [B, S, D], got {tuple(hidden.shape)}")
    vector = direction.detach().to(device=hidden.device, dtype=torch.float32)
    norm = vector.norm()
    if not torch.isfinite(norm) or not torch.isclose(
        norm, torch.ones((), device=norm.device), atol=1e-4, rtol=1e-4
    ):
        raise ValueError(f"Ablation direction must be unit norm, got {float(norm)}")
    if not torch.isfinite(torch.tensor(strength)):
        raise ValueError(f"Ablation strength must be finite, got {strength}")

    indices = _resolved_positions(hidden.shape[1], positions)
    edited = hidden.clone()
    selected = hidden[:, indices, :].float()
    projection = torch.einsum("bsd,d->bs", selected, vector)
    replacement = selected - float(strength) * projection.unsqueeze(-1) * vector
    edited[:, indices, :] = replacement.to(hidden.dtype)
    return edited


def swap_coordinates(
    hidden: torch.Tensor,
    concept_direction: torch.Tensor,
    foil_direction: torch.Tensor,
    *,
    positions: Sequence[int] | None = None,
    max_condition: float = 1e4,
    strength: float = 1.0,
) -> torch.Tensor:
    """Exactly swap two possibly nonorthogonal dot-product coordinates.

    If ``D=[v_concept; v_foil]`` and ``G=D D.T``, the minimum-subspace edit is
    ``h' = h + (swap(h D.T) - h D.T) G^-1 D``. The component orthogonal to
    both directions is unchanged. Ill-conditioned pairs are rejected because
    regularization would change the intervention's meaning.
    """

    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden shape [B, S, D], got {tuple(hidden.shape)}")
    concept = concept_direction.detach().to(hidden.device, torch.float32)
    foil = foil_direction.detach().to(hidden.device, torch.float32)
    for name, vector in (("concept", concept), ("foil", foil)):
        norm = vector.norm()
        if not torch.isfinite(norm) or not torch.isclose(
            norm, torch.ones((), device=norm.device), atol=1e-4, rtol=1e-4
        ):
            raise ValueError(f"{name} direction must be unit norm, got {float(norm)}")

    basis = torch.stack([concept, foil], dim=0)
    gram = basis @ basis.T
    condition = torch.linalg.cond(gram)
    if not torch.isfinite(condition) or float(condition) > max_condition:
        raise ValueError(
            f"Concept/foil Gram matrix is ill-conditioned: cond={float(condition):.4g}"
        )
    inverse_gram = torch.linalg.inv(gram)

    indices = _resolved_positions(hidden.shape[1], positions)
    edited = hidden.clone()
    selected = hidden[:, indices, :].float()
    projections = selected @ basis.T
    swapped = projections.flip(dims=(-1,))
    correction = (swapped - projections) @ inverse_gram @ basis
    if not torch.isfinite(torch.tensor(strength)):
        raise ValueError(f"Swap strength must be finite, got {strength}")
    edited[:, indices, :] = (selected + float(strength) * correction).to(hidden.dtype)
    return edited


def clamp_swapped_coordinates(
    hidden: torch.Tensor,
    clean_hidden: torch.Tensor,
    concept_direction: torch.Tensor,
    foil_direction: torch.Tensor,
    *,
    positions: Sequence[int] | None = None,
    max_condition: float = 1e4,
    strength: float = 1.0,
) -> torch.Tensor:
    """Clamp current concept/foil coefficients to clean counterfactual values.

    Multi-layer swaps must not repeatedly re-swap already-corrupted coordinates.
    At each layer this operation keeps the current orthogonal complement but
    sets its two basis coefficients to the swapped coefficients from the clean
    pass. ``strength=0`` clamps to clean; ``strength=1`` is the exact swap.
    """

    if hidden.shape != clean_hidden.shape or hidden.ndim != 3:
        raise ValueError(
            "Current and clean hidden states must share shape [B, S, D], got "
            f"{tuple(hidden.shape)} and {tuple(clean_hidden.shape)}"
        )
    concept = concept_direction.detach().to(hidden.device, torch.float32)
    foil = foil_direction.detach().to(hidden.device, torch.float32)
    basis = torch.stack([concept, foil], dim=0)
    norms = basis.norm(dim=-1)
    if not torch.isfinite(norms).all() or not torch.allclose(
        norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4
    ):
        raise ValueError(f"Clamp directions must be unit norm, got {norms.tolist()}")
    gram = basis @ basis.T
    condition = torch.linalg.cond(gram)
    if not torch.isfinite(condition) or float(condition) > max_condition:
        raise ValueError(
            f"Concept/foil Gram matrix is ill-conditioned: cond={float(condition):.4g}"
        )
    if not torch.isfinite(torch.tensor(strength)):
        raise ValueError(f"Swap strength must be finite, got {strength}")
    inverse_gram = torch.linalg.inv(gram)

    indices = _resolved_positions(hidden.shape[1], positions)
    current = hidden[:, indices, :].float()
    clean = clean_hidden.detach().to(hidden.device)[:, indices, :].float()
    current_coefficients = (current @ basis.T) @ inverse_gram
    clean_coefficients = (clean @ basis.T) @ inverse_gram
    desired_coefficients = clean_coefficients + float(strength) * (
        clean_coefficients.flip(dims=(-1,)) - clean_coefficients
    )
    correction = (desired_coefficients - current_coefficients) @ basis
    edited = hidden.clone()
    edited[:, indices, :] = (current + correction).to(hidden.dtype)
    return edited


def _replace_hidden(output: Any, edit: TensorEdit) -> Any:
    """Apply an edit to tensor/tuple decoder-block outputs without losing extras."""

    if torch.is_tensor(output):
        return edit(output)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (edit(output[0]), *output[1:])
    raise TypeError(f"Unsupported decoder-block output type: {type(output).__name__}")


@contextmanager
def residual_edit_hooks(
    blocks: Sequence[nn.Module],
    edits: Mapping[int, TensorEdit],
) -> Iterator[None]:
    """Install post-block edits and guarantee handle cleanup after exceptions."""

    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for layer, edit in sorted(edits.items()):
            if not 0 <= int(layer) < len(blocks):
                raise IndexError(f"Layer {layer} outside range [0, {len(blocks)})")

            def hook(module, inputs, output, *, _edit=edit):
                del module, inputs
                return _replace_hidden(output, _edit)

            handles.append(blocks[int(layer)].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def ablation_edits(
    directions: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int] | None = None,
    strength: float = 1.0,
) -> dict[int, TensorEdit]:
    """Create one scaled post-block ablation closure per layer."""

    return {
        layer: (
            lambda hidden, vector=direction: ablate_direction(
                hidden, vector, positions=positions, strength=strength
            )
        )
        for layer, direction in directions.items()
    }


def swap_edits(
    concept_directions: Mapping[int, torch.Tensor],
    foil_directions: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int] | None = None,
    max_condition: float = 1e4,
    strength: float = 1.0,
) -> dict[int, TensorEdit]:
    """Create exact concept/foil coordinate-swap closures for shared layers."""

    if set(concept_directions) != set(foil_directions):
        raise ValueError("Concept and foil directions must cover identical layers")
    return {
        layer: (
            lambda hidden, concept=concept_directions[layer], foil=foil_directions[layer]: (
                swap_coordinates(
                    hidden,
                    concept,
                    foil,
                    positions=positions,
                    max_condition=max_condition,
                    strength=strength,
                )
            )
        )
        for layer in concept_directions
    }


def clamped_swap_edits(
    clean_residuals: Mapping[int, torch.Tensor],
    concept_directions: Mapping[int, torch.Tensor],
    foil_directions: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int] | None = None,
    max_condition: float = 1e4,
    strength: float = 1.0,
) -> dict[int, TensorEdit]:
    """Create reference-defined clean-coordinate clamps for multiple layers."""

    layers = set(concept_directions)
    if layers != set(foil_directions) or layers != set(clean_residuals):
        raise ValueError(
            "Clean residuals, concept directions, and foil directions must "
            "cover identical layers"
        )
    return {
        layer: (
            lambda hidden,
            clean=clean_residuals[layer],
            concept=concept_directions[layer],
            foil=foil_directions[layer]: clamp_swapped_coordinates(
                hidden,
                clean,
                concept,
                foil,
                positions=positions,
                max_condition=max_condition,
                strength=strength,
            )
        )
        for layer in sorted(layers)
    }


@torch.no_grad()
def forward_logits(
    hf_model: nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    blocks: Sequence[nn.Module] | None = None,
    edits: Mapping[int, TensorEdit] | None = None,
) -> torch.Tensor:
    """Run an exact HF forward with optional residual edits; return fp32 logits."""

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if edits:
        if blocks is None:
            raise ValueError("blocks must be supplied when edits are requested")
        with residual_edit_hooks(blocks, edits):
            return hf_model(**kwargs).logits.float()
    return hf_model(**kwargs).logits.float()


def suppress_output_token(
    logits: torch.Tensor,
    token_id: int,
    *,
    value: float | None = None,
) -> torch.Tensor:
    """Clamp one output vocabulary logit without touching internal layers."""

    if not 0 <= int(token_id) < logits.shape[-1]:
        raise IndexError(f"token_id={token_id} outside vocabulary {logits.shape[-1]}")
    edited = logits.clone()
    clamp_value = torch.finfo(edited.dtype).min if value is None else float(value)
    edited[..., int(token_id)] = clamp_value
    return edited

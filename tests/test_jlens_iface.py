from __future__ import annotations

from types import SimpleNamespace

import torch

from src.jlens_iface import jlens_direction, write_by_position


def test_direction_orientation_is_j_transpose_times_unembedding() -> None:
    jacobian = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lm_head = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
        )
    lens = SimpleNamespace(
        d_model=2,
        source_layers=[0],
        jacobians={0: jacobian},
    )
    model = SimpleNamespace(d_model=2, n_layers=1, _lm_head=lm_head)

    direction = jlens_direction(lens, model, token_id=2, layer=0)
    expected = jacobian.T @ lm_head.weight[2]
    expected = expected / expected.norm()

    assert torch.allclose(direction, expected)
    assert not torch.allclose(direction, (jacobian @ lm_head.weight[2]) / (jacobian @ lm_head.weight[2]).norm())


def test_write_projection_keeps_layer_and_position_axes_explicit() -> None:
    residuals = {2: torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])}
    directions = {2: torch.tensor([1.0, 0.0])}
    result = write_by_position(residuals, directions, positions=[-1])
    assert result[2].shape == (1, 1)
    assert result[2].item() == 3.0


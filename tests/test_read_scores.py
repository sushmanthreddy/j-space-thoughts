from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.interventions import ablation_edits, forward_logits, residual_edit_hooks
from src.read_scores import (
    attribution_read,
    behavior_specific_read,
    exact_path_patch_scores,
)
from src.v2_read import _layer_aligned_weight_read


def _tiny_qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(3)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_attribution_read_matches_directional_finite_difference() -> None:
    model = _tiny_qwen()
    input_ids = torch.tensor([[1, 7, 9, 4]])
    direction = F.normalize(torch.arange(1, 17, dtype=torch.float32), dim=0)
    layer = 1
    result = attribution_read(
        model,
        model.model.layers,
        input_ids,
        {layer: direction},
        target_token_id=5,
        foil_token_id=6,
        intervention_positions=[-1],
    )

    epsilon = 1e-3

    def add_scaled(hidden: torch.Tensor, scale: float) -> torch.Tensor:
        edited = hidden.clone()
        edited[:, -1, :] = edited[:, -1, :] + scale * direction.to(hidden.dtype)
        return edited

    clean = forward_logits(model, input_ids)
    with residual_edit_hooks(
        model.model.layers, {layer: lambda hidden: add_scaled(hidden, epsilon)}
    ):
        plus = model(input_ids=input_ids, use_cache=False).logits.float()
    with residual_edit_hooks(
        model.model.layers, {layer: lambda hidden: add_scaled(hidden, -epsilon)}
    ):
        minus = model(input_ids=input_ids, use_cache=False).logits.float()
    plus_metric = plus[0, -1, 5] - plus[0, -1, 6]
    minus_metric = minus[0, -1, 5] - minus[0, -1, 6]
    finite_difference = float((plus_metric - minus_metric) / (2 * epsilon))

    assert abs(finite_difference - float(result.read[layer][0])) < 2e-3
    restored = forward_logits(model, input_ids)
    assert torch.equal(clean, restored)


def test_exact_path_patching_is_componentwise_and_restores_hooks() -> None:
    model = _tiny_qwen()
    blocks = model.model.layers
    input_ids = torch.tensor([[1, 7, 9, 4]])
    direction = F.normalize(torch.arange(1, 17, dtype=torch.float32), dim=0)
    def metric_fn(logits):
        return logits[0, -1, 5] - logits[0, -1, 6]
    original_mlp_hooks = [len(block.mlp._forward_hooks) for block in blocks]
    original_attention_hooks = [
        len(block.self_attn.o_proj._forward_pre_hooks) for block in blocks
    ]

    result = exact_path_patch_scores(
        model,
        blocks,
        input_ids,
        ablation_edits({0: direction}, positions=[-1]),
        metric_fn,
        component_layers=[1, 2],
    )

    assert len(result["mlps"]) == 2
    assert len(result["attention_heads"]) == 8
    assert result["estimator"]["gradient_used"] is False
    assert all(
        torch.isfinite(torch.tensor(row["patched_contribution"]))
        for row in (*result["mlps"], *result["attention_heads"])
    )
    assert [len(block.mlp._forward_hooks) for block in blocks] == original_mlp_hooks
    assert [
        len(block.self_attn.o_proj._forward_pre_hooks) for block in blocks
    ] == original_attention_hooks

    no_op = exact_path_patch_scores(
        model,
        blocks,
        input_ids,
        {},
        metric_fn,
        component_layers=[1, 2],
    )
    assert no_op["actual_delta"] == pytest.approx(0.0, abs=1e-8)
    assert max(
        row["abs_patched_contribution"]
        for row in (*no_op["mlps"], *no_op["attention_heads"])
    ) == pytest.approx(0.0, abs=1e-8)


def test_behavior_specific_read_matches_old_normalization_for_same_set() -> None:
    model = _tiny_qwen()
    direction = F.normalize(torch.arange(1, 17, dtype=torch.float32), dim=0)
    directions = {0: direction, 1: direction}
    path_scores = {
        "mlps": [
            {
                "component": "L1.MLP",
                "layer": 1,
                "abs_patched_contribution": 0.1,
                "patched_contribution": -0.1,
            }
        ],
        "attention_heads": [
            {
                "component": "L1.H0",
                "layer": 1,
                "head": 0,
                "abs_patched_contribution": 0.2,
                "patched_contribution": -0.2,
            }
        ],
    }
    seed = 41
    result = behavior_specific_read(
        model.model.layers,
        directions,
        path_scores,
        path_threshold=0.05,
        n_random=32,
        seed=seed,
    )
    old = _layer_aligned_weight_read(
        type("Bundle", (), {"lens_model": type("Lens", (), {"layers": model.model.layers})()})(),
        directions,
        {
            "mlps": [{"component": "L1.MLP", "layer": 1}],
            "attention_heads": [{"component": "L1.H0", "layer": 1, "head": 0}],
        },
        seed=seed,
    )

    assert result["s_m"]["component_ids"] == ["L1.MLP", "L1.H0"]
    assert result["mlp_primary"] == pytest.approx(old["mlp_primary"])
    assert result["attention_primary"] == pytest.approx(old["attention_primary"])
    assert result["equal_family_composite"] == pytest.approx(
        old["equal_family_composite"]
    )

    empty = behavior_specific_read(
        model.model.layers,
        directions,
        {"mlps": [], "attention_heads": []},
        path_threshold=0.05,
        seed=seed,
    )
    assert empty["status"] == "NO_COMPONENT_ABOVE_THRESHOLD"
    assert empty["equal_family_composite"] == 0.0
    assert set(empty["family_status"].values()) == {"NO_PATH_COMPONENTS_ZERO"}

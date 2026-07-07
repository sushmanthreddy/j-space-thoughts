from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.localization_phase import (
    NON_ADDITIVE_WARNING,
    capture_qwen_components,
    choose_f4_candidates,
    choose_source_layers,
    component_grad_delta_scores,
    flag_top_components,
    localize_source_direction,
    plot_f4_localization,
    qwen_attention_weight_read_with_null,
    select_localization_subset,
    spearman_rank_agreement,
    weight_read_for_flagged_components,
)
from src.read_scores import qwen_head_ov_read
from src.twohop_phase import PRIMARY_DIRECTION_METHOD


def _selection_row(name: str, write: float, read: float, damage: float) -> dict:
    return {
        "name": name,
        "measurement_status": "OK",
        "direction_method": PRIMARY_DIRECTION_METHOD,
        "aggregate": {
            "write_abs_mean": write,
            "read_abs_mean": read,
        },
        "ablation": {"positive_damage": damage},
    }


def test_quantile_subset_is_deterministic_spans_four_cells_and_assigns_no_class() -> None:
    rows = [
        _selection_row("hh-extreme", 10.0, 10.0, 4.0),
        _selection_row("hh-inner", 9.0, 9.0, 3.0),
        _selection_row("hl-extreme", 10.0, 1.0, 0.2),
        _selection_row("hl-inner", 9.0, 2.0, 0.3),
        _selection_row("lh-extreme", 1.0, 10.0, 5.0),
        _selection_row("lh-inner", 2.0, 9.0, 4.0),
        _selection_row("ll-extreme", 1.0, 1.0, 0.1),
        _selection_row("ll-inner", 2.0, 2.0, 0.2),
        _selection_row("ignored-md", 100.0, 100.0, 100.0)
        | {"direction_method": "mean_difference"},
        _selection_row("ignored-error", 100.0, 100.0, 100.0)
        | {"measurement_status": "ERROR"},
    ]

    first = select_localization_subset(rows)
    repeated = select_localization_subset(list(reversed(rows)))

    assert [row["name"] for row in first["selected"]] == [
        "hh-extreme",
        "hl-extreme",
        "lh-extreme",
        "ll-extreme",
    ]
    assert [row["name"] for row in repeated["selected"]] == [
        row["name"] for row in first["selected"]
    ]
    assert {row["cell"] for row in first["selected"]} == {
        "high_write_high_read",
        "high_write_low_read",
        "low_write_high_read",
        "low_write_low_read",
    }
    assert all(row["strict_threshold_match"] for row in first["selected"])
    assert first["provenance"]["n_eligible_raw_rows"] == 8
    assert "No item is assigned" in first["provenance"]["class_guardrail"]

    candidates = choose_f4_candidates(first)
    assert candidates["driver_candidate"]["name"] == "lh-extreme"
    assert candidates["low_read_candidate"]["name"] == "hl-extreme"
    assert "not declared narration" in candidates["guardrail"]


def test_source_layer_selection_recomputes_exact_signed_products() -> None:
    row = {
        "name": "source-test",
        "raw_arrays": {
            "write_by_layer_position": {
                "3": [2.0, -1.0],
                "4": [3.0, 2.0],
                "5": [100.0, 100.0],
            },
            "attribution_read_by_layer_position": {
                "3": [0.5, 2.0],
                "4": [1.0, 1.0],
                "5": [1.0, 1.0],
            },
        },
    }

    result = choose_source_layers(row, n_source_layers=2, max_source_layer=4)

    assert result["selected_layers"] == [4, 3]
    assert result["selected"][0]["signed_first_order_positive_damage"] == 5.0
    assert result["selected"][1]["signed_first_order_positive_damage"] == -1.0
    assert result["selected"][1]["selection_score"] == 1.0
    assert "targeting only" in result["role"]


def test_component_grad_delta_formula_separates_heads_and_rejects_bad_shapes() -> None:
    clean_mlp = torch.zeros(1, 2, 4)
    perturbed_mlp = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]]]
    )
    gradient_mlp = torch.full((1, 2, 4), 2.0)
    clean_attention = torch.zeros(1, 2, 4)
    perturbed_attention = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )
    gradient_attention = torch.ones(1, 2, 4)
    clean = {
        "mlp": {1: clean_mlp},
        "attention_pre_o_proj": {1: clean_attention},
    }
    perturbed = {
        "mlp": {1: perturbed_mlp},
        "attention_pre_o_proj": {1: perturbed_attention},
    }
    gradients = {
        "mlp": {1: gradient_mlp},
        "attention_pre_o_proj": {1: gradient_attention},
    }

    result = component_grad_delta_scores(
        clean,
        perturbed,
        gradients,
        head_geometry={1: (2, 2)},
    )

    assert result["mlps"][0]["score"] == pytest.approx(
        float((perturbed_mlp * gradient_mlp).sum())
    )
    assert [row["score"] for row in result["attention_heads"]] == pytest.approx(
        [1.0 + 2.0 + 5.0 + 6.0, 3.0 + 4.0 + 7.0 + 8.0]
    )
    assert result["attention_heads"][0]["score_by_position"] == pytest.approx(
        [3.0, 11.0]
    )

    bad = copy.deepcopy(perturbed)
    bad["mlp"][1] = torch.zeros(1, 2, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        component_grad_delta_scores(
            clean,
            bad,
            gradients,
            head_geometry={1: (2, 2)},
        )


def _tiny_qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(3)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_tiny_qwen_localization_weight_nulls_cleanup_and_f4(tmp_path) -> None:
    model = _tiny_qwen()
    blocks = model.model.layers
    input_ids = torch.tensor([[1, 7, 9, 4]])
    direction = F.normalize(torch.arange(1, 17, dtype=torch.float32), dim=0)
    original_mlp_hooks = [len(block.mlp._forward_hooks) for block in blocks]
    original_attention_hooks = [
        len(block.self_attn.o_proj._forward_pre_hooks) for block in blocks
    ]

    with pytest.raises(RuntimeError, match="deliberate"):
        with capture_qwen_components(blocks, [1, 2], start_graph=False):
            raise RuntimeError("deliberate hook-scope failure")
    assert [len(block.mlp._forward_hooks) for block in blocks] == original_mlp_hooks
    assert [
        len(block.self_attn.o_proj._forward_pre_hooks) for block in blocks
    ] == original_attention_hooks

    localization = localize_source_direction(
        model,
        blocks,
        input_ids,
        direction,
        source_layer=0,
        target_token_id=5,
        foil_token_id=6,
    )

    assert len(localization["mlps"]) == 3
    assert len(localization["attention_heads"]) == 12
    assert localization["component_layers"] == [1, 2, 3]
    assert np.isfinite(localization["actual_delta"])
    assert localization["positive_damage"] == pytest.approx(
        -localization["actual_delta"]
    )
    assert NON_ADDITIVE_WARNING in localization["localization_estimator"]["warning"]
    assert [len(block.mlp._forward_hooks) for block in blocks] == original_mlp_hooks
    assert [
        len(block.self_attn.o_proj._forward_pre_hooks) for block in blocks
    ] == original_attention_hooks

    flagged = flag_top_components(
        localization,
        top_k_mlps=3,
        top_k_heads=4,
    )
    weight = weight_read_for_flagged_components(
        blocks,
        direction,
        flagged,
        label_direction=direction,
        n_random=5,
        seed=19,
    )
    assert len(weight["mlps"]) == 3
    assert len(weight["attention_heads"]) == 4
    assert weight["metadata"]["raw_random_nulls_retained"] is True
    for row in weight["mlps"]:
        assert len(row["random_gains"]) == 5
        assert row["normalized_gain"] == pytest.approx(
            row["gain"] / row["random_median"]
        )
    for row in weight["attention_heads"]:
        assert len(row["random_ov_norms"]) == 5
        assert len(row["random_label_cosines"]) == 5
        assert row["normalized_ov_norm"] == pytest.approx(
            row["ov_norm"] / row["random_median_ov_norm"]
        )
        assert -1.0 <= row["label_cosine"] <= 1.0

    null_seed = 23
    direct_null = qwen_attention_weight_read_with_null(
        blocks[1].self_attn,
        direction,
        label_direction=direction,
        n_random=3,
        seed=null_seed,
    )
    generator = torch.Generator(device="cpu").manual_seed(null_seed)
    first_random = F.normalize(
        torch.randn(3, direction.numel(), generator=generator)[0], dim=0
    )
    first_helper = qwen_head_ov_read(
        blocks[1].self_attn,
        first_random,
        label_direction=direction,
    )
    for vectorized, scalar in zip(direct_null, first_helper, strict=True):
        assert vectorized["random_ov_norms"][0] == pytest.approx(
            scalar["ov_norm"], rel=1e-5, abs=1e-7
        )
        assert vectorized["random_label_cosines"][0] == pytest.approx(
            scalar["label_cosine"], rel=1e-5, abs=1e-7
        )

    second = copy.deepcopy(localization)
    for row in second["mlps"]:
        row["score"] *= -0.5
    for row in second["attention_heads"]:
        row["score"] *= -0.5
    records = [
        {
            "name": "driver",
            "source_layer": 0,
            "source_selection_rank": 1,
            "localization": localization,
        },
        {
            "name": "low-read",
            "source_layer": 0,
            "source_selection_rank": 1,
            "localization": second,
        },
    ]
    candidates = {
        "driver_candidate": {"name": "driver"},
        "low_read_candidate": {"name": "low-read"},
    }
    path = plot_f4_localization(records, candidates, tmp_path / "f4.png")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_spearman_rank_agreement_reports_exact_monotonic_order_and_null_case() -> None:
    increasing = [
        {"component": f"c{index}", "abs_score": float(index), "weight": index**2}
        for index in range(1, 6)
    ]
    result = spearman_rank_agreement(increasing, weight_key="weight")
    assert result["status"] == "ESTIMATED"
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n"] == 5
    assert result["paired_ranks"][0]["component_id"] == "c5"

    constant = [
        {"component": f"c{index}", "abs_score": float(index), "weight": 1.0}
        for index in range(1, 6)
    ]
    not_estimable = spearman_rank_agreement(constant, weight_key="weight")
    assert not_estimable["status"] == "NOT_ESTIMABLE"
    assert "constant" in not_estimable["reason"]

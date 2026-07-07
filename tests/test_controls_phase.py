from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pytest
import torch

from src.controls_phase import (
    adapt_core_control_rows,
    assert_output_suppression_complete,
    behavior_effect_record,
    core_output_suppression_comparison,
    identity_jacobian_direction,
    known_narration_reproduction_summary,
    plot_capability_controls,
    plot_internal_vs_output_suppression,
    plot_known_narration_controls,
    plot_random_null_controls,
    seeded_random_direction_bank,
    select_absent_concept,
    select_offtarget_capability_items,
    teacher_forced_nll,
)


def test_random_null_bank_is_seeded_unit_norm_and_all_layer() -> None:
    references = {
        2: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        3: torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    first, first_seeds = seeded_random_direction_bank(
        references, item_name="alpha", draw_index=4, seed=17
    )
    repeated, repeated_seeds = seeded_random_direction_bank(
        references, item_name="alpha", draw_index=4, seed=17
    )
    different, _ = seeded_random_direction_bank(
        references, item_name="alpha", draw_index=5, seed=17
    )

    assert set(first) == set(references)
    assert first_seeds == repeated_seeds
    assert all(torch.equal(first[layer], repeated[layer]) for layer in first)
    assert all(torch.isclose(vector.norm(), torch.tensor(1.0)) for vector in first.values())
    assert any(not torch.equal(first[layer], different[layer]) for layer in first)
    assert first_seeds[2] != first_seeds[3]


def test_absent_selection_uses_fixed_order_rank_rule_and_exclusions() -> None:
    selection = select_absent_concept(
        [11, 12, 13, 14],
        {
            11: {2: [5_000]},
            12: {2: [4_000, 1_200], 3: [3_000]},
            13: {2: [7_000]},
            14: {2: [9_000]},
        },
        excluded_token_ids=[11],
        min_rank=2_000,
    )

    assert selection["token_id"] == 13
    statuses = [row["status"] for row in selection["candidate_audit_until_selection"]]
    assert statuses == [
        "excluded_target_or_foil",
        "rejected_too_present",
        "qualified",
    ]
    assert "behavior" not in selection["selection_rule"]


def test_absent_selection_does_not_relax_failed_preregistered_rule() -> None:
    with pytest.raises(ValueError, match="must not be changed"):
        select_absent_concept(
            [1, 2],
            {1: [10, 20], 2: [30, 40]},
            min_rank=100,
        )


def test_teacher_forced_nll_matches_manual_targets_and_masks_padding() -> None:
    logits = torch.zeros(2, 4, 5)
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 4, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    logits[0, 0, 1] = 2.0
    logits[0, 1, 2] = 1.0
    logits[0, 2, 3] = 3.0
    logits[1, 0, 4] = 4.0
    logits[1, 1:, :] = float("nan")  # masked positions must not enter the mean

    observed = teacher_forced_nll(logits, input_ids, attention)
    expected_first = torch.stack(
        [
            -logits[0, 0].log_softmax(-1)[1],
            -logits[0, 1].log_softmax(-1)[2],
            -logits[0, 2].log_softmax(-1)[3],
        ]
    ).mean()
    expected_second = -logits[1, 0].log_softmax(-1)[4]

    assert torch.allclose(observed[0], expected_first)
    assert torch.allclose(observed[1], expected_second)


def test_teacher_forced_nll_excludes_first_token_after_left_padding() -> None:
    logits = torch.zeros(1, 4, 3)
    input_ids = torch.tensor([[0, 0, 1, 2]])
    attention = torch.tensor([[0, 0, 1, 1]])
    logits[0, 2, 2] = 5.0

    observed = teacher_forced_nll(logits, input_ids, attention)
    expected = -logits[0, 2].log_softmax(-1)[2]

    assert torch.allclose(observed[0], expected)


def test_core_suppression_adapter_requires_finite_explicit_values() -> None:
    rows = [
        {"name": "a", "suppression_delta": 0.0, "actual_delta": -2.0},
        {
            "item": {"name": "b"},
            "output_suppression_delta": 0.25,
            "actual_delta": -1.0,
            "predicted_delta": -1.0,
        },
    ]
    adapted = assert_output_suppression_complete(
        rows, expected_item_names=["a", "b"]
    )
    assert [row["suppression_delta"] for row in adapted] == [0.0, 0.25]
    assert adapted[1]["name"] == "b"

    with pytest.raises(ValueError, match="lacks"):
        adapt_core_control_rows([{"name": "missing"}])
    with pytest.raises(ValueError, match="nonfinite"):
        adapt_core_control_rows([{"name": "bad", "suppression_delta": math.nan}])
    with pytest.raises(ValueError, match="nonfinite actual_delta"):
        adapt_core_control_rows(
            [{"name": "bad", "suppression_delta": 0.0, "actual_delta": math.inf}]
        )
    with pytest.raises(ValueError, match="coverage"):
        assert_output_suppression_complete(rows, expected_item_names=["a", "c"])

    comparison = core_output_suppression_comparison(adapted)
    assert comparison["n"] == 2
    assert comparison["rows"][0]["internal_minus_output_delta"] == -2.0


def test_identity_jacobian_is_normalized_unembedding_row() -> None:
    weight = torch.tensor([[3.0, 4.0], [-1.0, 0.0]])
    direction = identity_jacobian_direction(weight, 0)
    assert torch.allclose(direction, torch.tensor([0.6, 0.8]))
    assert torch.isclose(direction.norm(), torch.tensor(1.0))


def test_behavior_effect_record_uses_edited_minus_clean_sign() -> None:
    record = behavior_effect_record(clean_metric=3.5, edited_metric=-0.5)
    assert record == {"clean_metric": 3.5, "edited_metric": -0.5, "delta": -4.0}


def test_offtarget_capability_selection_is_cyclic_and_excludes_overlap() -> None:
    def item(name: str, base: int, prompt: str) -> dict:
        return {
            "name": name,
            "prompt": prompt,
            "intermediate": f"concept-{base}",
            "swap_to": f"foil-{base}",
            "concept_token_id": base,
            "foil_concept_token_id": base + 1,
            "target_token_id": base + 2,
            "foil_token_id": base + 3,
        }

    source = item("source", 10, "source prompt")
    shared = item("shared", 11, "overlapping role")
    first = item("first", 30, "unrelated first")
    second = item("second", 40, "unrelated second")
    selected = select_offtarget_capability_items(
        source,
        [source, shared, first, second],
        n_tasks=2,
    )
    assert [row["name"] for row in selected] == ["first", "second"]


def test_control_plot_helpers_smoke() -> None:
    random_figure, _ = plot_random_null_controls(
        {
            "rows": [
                {
                    "observed_concept_delta": -2.0,
                    "draws": [{"delta": -0.2}, {"delta": 0.1}],
                }
            ]
        }
    )
    capability_figure, _ = plot_capability_controls(
        {
            "general_language": {"rows": [{"delta_nll": 0.02}]},
            "twohop": {"clean_accuracy": 0.8, "edited_accuracy": 0.7},
        }
    )
    narration_figure, _ = plot_known_narration_controls(
        {
            "rows": [
                {
                    "passage_key": "fr1",
                    "attribution_predicted_delta": -0.5,
                    "internal_ablation_delta": -0.7,
                    "output_suppression_delta": 0.0,
                }
            ]
        }
    )
    suppression_figure, suppression_axis = plot_internal_vs_output_suppression(
        {
            "rows": [
                {
                    "internal_ablation_delta": -1.2,
                    "output_suppression_delta": 0.0,
                },
                {
                    "internal_ablation_delta": -0.4,
                    "output_suppression_delta": 0.1,
                },
            ]
        }
    )

    assert len(random_figure.axes) == 1
    assert len(capability_figure.axes) == 2
    assert len(narration_figure.axes) == 1
    assert len(suppression_figure.axes) == 1
    assert suppression_axis.get_xlabel().startswith("internal")
    plt.close("all")


def test_known_narration_reproduction_gate_requires_six_joint_passages() -> None:
    rows = [
        {
            "high_write": index < 7,
            "low_causal": index != 6,
            "clean_capable": True,
            "reproduces_known_narration": index < 6,
        }
        for index in range(8)
    ]
    passing = known_narration_reproduction_summary(rows)
    assert passing["status"] == "PASS"
    assert passing["n_reproduced"] == 6
    assert passing["n_high_write"] == 7
    assert passing["n_low_causal"] == 7
    assert passing["n_clean_capable"] == 8

    rows[5]["high_write"] = False
    rows[5]["reproduces_known_narration"] = False
    failing = known_narration_reproduction_summary(rows)
    assert failing["status"] == "FAIL"
    assert failing["n_reproduced"] == 5

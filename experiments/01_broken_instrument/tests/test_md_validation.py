"""Focused tests for cached mean-difference validation metrics."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.md_validation import (
    fit_score_calibration,
    heldout_calibrated_retrieval,
    heldout_matched_baseline_deltas,
    leave_one_train_slot_out_stability,
)


CONCEPTS = ("alpha", "beta", "delta", "gamma")
GRAPH = {
    "alpha": {"beta"},
    "beta": {"alpha"},
    "gamma": {"delta"},
    "delta": {"gamma"},
}
EPSILONS = {
    "alpha": torch.tensor([-0.3, -0.1, 0.1, 0.3]),
    "beta": torch.tensor([-0.1, 0.3, -0.3, 0.1]),
    "delta": torch.tensor([0.1, -0.3, 0.3, -0.1]),
    "gamma": torch.tensor([0.3, 0.1, -0.1, -0.3]),
}


def _eligible(concept: str) -> list[str]:
    return [
        other
        for other in CONCEPTS
        if other != concept and other not in GRAPH[concept]
    ]


def _synthetic_inputs():
    basis = {
        concept: torch.nn.functional.one_hot(
            torch.tensor(index), num_classes=5
        ).double()
        for index, concept in enumerate(CONCEPTS)
    }
    train: dict[str, dict[int, torch.Tensor]] = {}
    heldout: dict[str, dict[int, torch.Tensor]] = {}
    for concept in CONCEPTS:
        train[concept] = {}
        heldout[concept] = {}
        for layer, factor in ((0, 1.0), (3, 2.0)):
            train[concept][layer] = (
                factor
                * (1.0 + EPSILONS[concept]).unsqueeze(1)
                * basis[concept].unsqueeze(0)
            )
            heldout[concept][layer] = (
                factor
                * torch.tensor([0.9, 1.1]).unsqueeze(1)
                * basis[concept].unsqueeze(0)
            )
    directions: dict[str, dict[int, torch.Tensor]] = {}
    for concept in CONCEPTS:
        directions[concept] = {}
        for layer in (0, 3):
            baseline = torch.stack(
                [train[other][layer] for other in _eligible(concept)]
            ).mean(0)
            difference = (train[concept][layer] - baseline).mean(0)
            directions[concept][layer] = torch.nn.functional.normalize(
                difference, dim=0
            )
    return train, heldout, directions


def test_train_calibration_uses_only_matched_eligible_negatives() -> None:
    train, _, directions = _synthetic_inputs()
    calibration = fit_score_calibration(train, directions, GRAPH)

    fitted = calibration["alpha"][0]
    eligible = _eligible("alpha")
    scores = torch.stack(
        [train[other][0] @ directions["alpha"][0] for other in eligible]
    )
    assert fitted["eligible_concepts"] == eligible
    assert fitted["n_negative_scores"] == len(eligible) * 4
    assert fitted["center"] == pytest.approx(float(scores.mean()))
    assert fitted["scale"] == pytest.approx(float(scores.std(correction=0)))
    assert fitted["slot_centers"] == pytest.approx(
        [float(value) for value in scores.mean(0)]
    )


def test_training_validation_rejects_flipped_direction_and_wrong_orientation() -> None:
    train, _, directions = _synthetic_inputs()
    flipped = copy.deepcopy(directions)
    flipped["alpha"][0] = -flipped["alpha"][0]
    with pytest.raises(ValueError, match="orientation/formula mismatch"):
        fit_score_calibration(train, flipped, GRAPH)

    transposed = copy.deepcopy(train)
    transposed["alpha"][0] = transposed["alpha"][0].T
    with pytest.raises(ValueError, match=r"oriented \[prompt_slot, d_model\]"):
        fit_score_calibration(transposed, directions, GRAPH)


def test_calibrated_retrieval_rows_metrics_auroc_and_permutation_are_deterministic() -> None:
    train, heldout, directions = _synthetic_inputs()
    calibration = fit_score_calibration(train, directions, GRAPH)

    first = heldout_calibrated_retrieval(
        heldout,
        directions,
        calibration,
        n_permutations=2000,
        permutation_seed=41,
    )
    second = heldout_calibrated_retrieval(
        heldout,
        directions,
        calibration,
        n_permutations=2000,
        permutation_seed=41,
    )

    assert first == second
    assert first["concept_order"] == list(CONCEPTS)
    assert len(first["rows"]) == 2 * len(CONCEPTS) * 2
    for layer in (0, 3):
        summary = first["by_layer"][layer]
        assert summary["n_way"] == 4
        assert summary["n_rows"] == 8
        assert summary["top1_accuracy"] == 1.0
        assert summary["mean_reciprocal_rank"] == 1.0
        assert summary["macro_ovr_auroc"] == 1.0
        assert summary["top1_fixed_label_permutation"]["p_value"] < 0.01
        assert summary["top1_fixed_label_permutation"]["seed"] == 41
    assert all(row["rank"] == 1 and row["top1"] == 1 for row in first["rows"])


def test_heldout_delta_uses_same_slot_and_omits_paired_foil() -> None:
    _, heldout, directions = _synthetic_inputs()
    rows = heldout_matched_baseline_deltas(heldout, directions, GRAPH)
    row = next(
        value
        for value in rows
        if value["concept"] == "alpha"
        and value["layer"] == 0
        and value["heldout_slot"] == 0
    )
    eligible = _eligible("alpha")
    direction = directions["alpha"][0]
    expected_own = float(heldout["alpha"][0][0] @ direction)
    expected_baseline = float(
        np.mean([float(heldout[name][0][0] @ direction) for name in eligible])
    )
    assert row["eligible_concepts"] == eligible
    assert "beta" not in row["eligible_concepts"]
    assert row["own_score"] == pytest.approx(expected_own)
    assert row["matched_baseline_score"] == pytest.approx(expected_baseline)
    assert row["delta"] == pytest.approx(expected_own - expected_baseline)

    altered = copy.deepcopy(heldout)
    altered["beta"][0] *= 1000
    altered_row = next(
        value
        for value in heldout_matched_baseline_deltas(altered, directions, GRAPH)
        if value["concept"] == "alpha"
        and value["layer"] == 0
        and value["heldout_slot"] == 0
    )
    assert altered_row["delta"] == pytest.approx(row["delta"])


def test_leave_one_slot_out_reuses_matched_leave_exclusion_out_formula() -> None:
    train, _, directions = _synthetic_inputs()
    rows = leave_one_train_slot_out_stability(train, directions, GRAPH)
    assert len(rows) == 2 * len(CONCEPTS) * 4

    row = next(
        value
        for value in rows
        if value["concept"] == "alpha"
        and value["layer"] == 0
        and value["left_out_slot"] == 0
    )
    kept = torch.tensor([1, 2, 3])
    baseline = torch.stack(
        [train[other][0].index_select(0, kept) for other in _eligible("alpha")]
    ).mean(0)
    difference = (
        train["alpha"][0].index_select(0, kept) - baseline
    ).mean(0)
    loo = torch.nn.functional.normalize(difference, dim=0)
    expected = float(torch.dot(loo, directions["alpha"][0]))
    assert row["eligible_concepts"] == _eligible("alpha")
    assert row["n_slots_retained"] == 3
    assert row["cosine_to_full_direction"] == pytest.approx(expected)


def test_exclusion_graph_must_be_symmetric_and_complete() -> None:
    train, _, directions = _synthetic_inputs()
    asymmetric = copy.deepcopy(GRAPH)
    asymmetric["alpha"] = set()
    with pytest.raises(ValueError, match="must be symmetric"):
        fit_score_calibration(train, directions, asymmetric)

    incomplete = copy.deepcopy(GRAPH)
    del incomplete["alpha"]
    with pytest.raises(ValueError, match="keys must exactly match"):
        fit_score_calibration(train, directions, incomplete)

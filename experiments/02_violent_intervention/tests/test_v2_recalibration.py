import torch

from src.v2_recalibration import (
    _gram_matched_random_pair,
    _stage2_gate_criteria,
    language_mass_metric,
    subtract_output_logit,
    subtract_output_token_set,
)


def test_output_suppression_arm_fires_on_pair_metric():
    logits = torch.tensor([[[1.0, 4.0, 2.0]]])
    clean = float(logits[0, -1, 1] - logits[0, -1, 2])
    source = subtract_output_logit(logits, 1)
    foil = subtract_output_logit(logits, 2)
    source_delta = float(source[0, -1, 1] - source[0, -1, 2]) - clean
    foil_delta = float(foil[0, -1, 1] - foil[0, -1, 2]) - clean
    assert source_delta == -1.0
    assert foil_delta == 1.0
    assert 0.5 * (source_delta + foil_delta) == 0.0


def test_output_suppression_rejects_nonpositive_amount():
    logits = torch.zeros(1, 1, 2)
    try:
        subtract_output_logit(logits, 0, amount=0.0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected nonpositive suppression to fail")


def test_token_family_suppression_arms_fire_with_unequal_set_sizes():
    logits = torch.tensor(
        [[[0.1, 0.2, 0.3, -0.1, 0.5], [0.4, -0.2, 0.6, 0.3, 0.1]]]
    )
    positions = [0, 1]
    source_ids = [0, 1]
    english_ids = [2, 3, 4]
    clean = float(
        language_mass_metric(logits, positions, source_ids, english_ids)
    )
    source = subtract_output_token_set(logits, source_ids, positions)
    english = subtract_output_token_set(logits, english_ids, positions)
    source_effect = float(
        language_mass_metric(source, positions, source_ids, english_ids)
    ) - clean
    english_effect = float(
        language_mass_metric(english, positions, source_ids, english_ids)
    ) - clean
    assert abs(source_effect + 1.0) < 1e-6
    assert abs(english_effect - 1.0) < 1e-6


def test_random_pair_null_preserves_real_gram_geometry():
    source = {
        3: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        4: torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    target = {
        3: torch.tensor([0.8, 0.6, 0.0, 0.0]),
        4: torch.tensor([0.0, -0.5, 0.0, 3**0.5 / 2]),
    }
    random_source, random_target, _, _ = _gram_matched_random_pair(
        source, target, item_name="fixture", draw_index=0
    )
    for layer in source:
        observed = float(torch.dot(random_source[layer], random_target[layer]))
        expected = float(torch.dot(source[layer], target[layer]))
        assert abs(observed - expected) < 2e-5


def test_stage2_gate_includes_specificity_controls():
    passing = {"status": "PASS"}
    criteria = _stage2_gate_criteria(
        gswap=passing,
        controls_fire=passing,
        random_null=passing,
        absent=passing,
        capability=passing,
        gpos=passing,
    )
    assert all(criteria.values())
    criteria = _stage2_gate_criteria(
        gswap=passing,
        controls_fire=passing,
        random_null={"status": "FAIL"},
        absent={"status": "FAIL"},
        capability=passing,
        gpos=passing,
    )
    assert not criteria["random_pair_specific"]
    assert not criteria["absent_coordinate_specific"]
    assert not all(criteria.values())

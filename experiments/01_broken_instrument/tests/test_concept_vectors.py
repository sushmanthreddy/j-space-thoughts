"""Unit tests for leakage-safe mean-difference construction."""

from __future__ import annotations

import pytest
import torch

import src.concept_vectors as concept_vectors


def test_mean_difference_bank_excludes_predeclared_foil() -> None:
    means = {
        "alpha": torch.tensor([2.0, 0.0]),
        "foil": torch.tensor([1.0, 0.0]),
        "other": torch.tensor([0.0, 2.0]),
    }
    directions, _ = concept_vectors.mean_difference_bank_from_matrices(
        {name: {0: value.unsqueeze(0)} for name, value in means.items()},
        baseline_exclusions={"alpha": ["foil"]},
    )

    expected = torch.nn.functional.normalize(means["alpha"] - means["other"], dim=0)
    assert torch.allclose(directions["alpha"][0], expected)


def test_mean_difference_bank_rejects_unknown_exclusion() -> None:
    with pytest.raises(ValueError, match="unknown concepts"):
        concept_vectors.mean_difference_bank_from_matrices(
            {
                "alpha": {0: torch.tensor([[1.0, 0.0]])},
                "beta": {0: torch.tensor([[0.0, 1.0]])},
            },
            baseline_exclusions={"alpha": ["missing"]},
        )


def test_mean_difference_bank_rejects_empty_baseline() -> None:
    with pytest.raises(ValueError, match="No eligible baseline"):
        concept_vectors.mean_difference_bank_from_matrices(
            {
                "alpha": {0: torch.tensor([[1.0, 0.0]])},
                "beta": {0: torch.tensor([[0.0, 1.0]])},
            },
            baseline_exclusions={"alpha": ["beta"]},
        )


def test_matched_prompt_slots_remove_carrier_offsets() -> None:
    matrices = {
        "alpha": torch.tensor([[11.0, 0.0], [0.0, 11.0]]),
        "beta": torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
        "gamma": torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
    }

    directions, _ = concept_vectors.mean_difference_bank_from_matrices(
        {name: {0: matrix} for name, matrix in matrices.items()},
        matched_prompt_slots=True,
    )

    expected = torch.tensor([2**-0.5, 2**-0.5])
    assert torch.allclose(directions["alpha"][0], expected)

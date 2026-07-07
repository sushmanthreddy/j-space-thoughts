from __future__ import annotations

import numpy as np
import pytest
import torch

from src.twohop_phase import (
    MD_DIRECTION_METHOD,
    PRIMARY_DIRECTION_METHOD,
    aggregate_write_read,
    analyze_measurements,
    clean_eligibility_from_logits,
    load_mean_difference_artifact,
)


def test_aggregate_write_read_retains_signs_and_exact_prediction() -> None:
    write = {
        3: np.array([2.0, -1.0]),
        4: np.array([3.0, 0.5]),
    }
    read = {
        3: np.array([0.5, 2.0]),
        4: np.array([-1.0, 0.25]),
    }

    result = aggregate_write_read(write, read)

    expected_product_sum = 1.0 - 2.0 - 3.0 + 0.125
    assert result["n_coordinates"] == 4
    assert result["write_signed_sum"] == pytest.approx(4.5)
    assert result["write_abs_sum"] == pytest.approx(6.5)
    assert result["read_signed_sum"] == pytest.approx(1.75)
    assert result["write_read_signed_sum"] == pytest.approx(expected_product_sum)
    assert result["first_order_predicted_delta"] == pytest.approx(-expected_product_sum)
    assert result["first_order_predicted_positive_damage"] == pytest.approx(
        expected_product_sum
    )
    assert result["support_oriented_read"] == pytest.approx(expected_product_sum / 6.5)
    assert result["write_abs_sum"] * result["support_oriented_read"] == pytest.approx(
        result["first_order_predicted_positive_damage"]
    )
    assert result["by_layer"]["3"]["write_read_signed_sum"] == pytest.approx(-1.0)


def _synthetic_rows(method: str, *, seed: int, n: int = 80) -> list[dict]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    read = np.abs(latent) + 0.25
    write = 1.5 + 0.35 * read + rng.normal(scale=0.25, size=n)
    causal = 2.2 * read + 0.08 * write + rng.normal(scale=0.18, size=n)
    predicted = 2.0 * read + rng.normal(scale=0.2, size=n)
    rows = []
    for index in range(n):
        rows.append(
            {
                "name": f"{method}-{index}",
                "direction_method": method,
                "measurement_status": "OK",
                "aggregate": {
                    "write_abs_mean": float(write[index]),
                    "write_signed_mean": float(write[index]),
                    "read_abs_mean": float(read[index]),
                    "read_signed_mean": float(latent[index]),
                    "support_oriented_read": float(latent[index]),
                    "first_order_predicted_positive_damage": float(predicted[index]),
                },
                "ablation": {
                    "positive_damage": float(causal[index]),
                    "delta": float(-causal[index]),
                },
                "clean_clamped_swap": {
                    "positive_damage": float(1.2 * causal[index]),
                    "delta": float(-1.2 * causal[index]),
                },
                "output_suppression": {
                    "concept": {"positive_damage": 0.0, "delta": 0.0}
                },
            }
        )
    return rows


def test_synthetic_analysis_groups_methods_and_finds_independent_read_signal() -> None:
    rows = [
        *_synthetic_rows(PRIMARY_DIRECTION_METHOD, seed=2),
        *_synthetic_rows(MD_DIRECTION_METHOD, seed=3),
    ]

    result = analyze_measurements(rows, n_bootstrap=200, seed=11)

    assert result["outcome"] == "ablation"
    assert set(result["by_method"]) == {
        PRIMARY_DIRECTION_METHOD,
        MD_DIRECTION_METHOD,
    }
    for method in (PRIMARY_DIRECTION_METHOD, MD_DIRECTION_METHOD):
        stats = result["by_method"][method]
        assert stats["n"] == 80
        assert stats["variables"]["read"] == "aggregate.read_abs_mean"
        assert stats["pearson"]["causal_vs_read"]["status"] == "ESTIMATED"
        assert (
            stats["partial_correlations"]["causal_read_given_write"]["estimate"] > 0.95
        )
        additive = stats["regressions"]["causal_on_write_plus_read"]
        assert additive["status"] == "ESTIMATED"
        assert abs(additive["coefficients"]["read"]) > abs(
            additive["coefficients"]["write"]
        )
        assert additive["coefficient_intervals"]["read"]["ci_low"] > 0.8
        assert "excluded from P1" in stats["product_based_diagnostic"]["warning"]
        assert (
            stats["signed_supplement"]["pearson"]["causal_delta_vs_signed_read"][
                "status"
            ]
            == "ESTIMATED"
        )


def test_clean_eligibility_is_target_top1_and_latent_tokens_absent_top10() -> None:
    item = {
        "name": "example",
        "target_token_id": 3,
        "foil_token_id": 2,
        "concept_token_id": 4,
        "foil_concept_token_id": 5,
    }
    logits = torch.arange(20, dtype=torch.float32)
    logits[3] = 100.0
    logits[4] = -100.0
    logits[5] = -101.0

    accepted = clean_eligibility_from_logits(item, logits)
    assert accepted["eligible"] is True
    assert accepted["target_rank"] == 1
    assert accepted["rejection_reasons"] == []

    logits[4] = 99.0
    rejected = clean_eligibility_from_logits(item, logits)
    assert rejected["eligible"] is False
    assert rejected["rejection_reasons"] == ["concept_in_clean_top10"]


def test_md_artifact_loader_validates_schema_layers_model_and_half_norms(
    tmp_path,
) -> None:
    first = torch.tensor([1.0, 0.0], dtype=torch.float16)
    second = torch.tensor([0.0, 1.0], dtype=torch.float16)
    path = tmp_path / "directions.pt"
    torch.save(
        {
            "metadata": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": "revision",
                "workspace_layers": [2, 3],
                "baseline_exclusions": {"Spain": {"Canada"}},
            },
            "mean_difference": {
                "Spain": {2: first, 3: second},
                "Canada": {2: second, 3: first},
            },
        },
        path,
    )

    artifact = load_mean_difference_artifact(
        path,
        expected_layers=[2, 3],
        expected_model_id="Qwen/Qwen2.5-7B-Instruct",
        expected_model_revision="revision",
    )

    assert artifact["n_concepts"] == 2
    assert artifact["canonical_lookup"]["spain"] == "Spain"
    assert artifact["source_dtypes"] == ["torch.float16"]
    assert torch.linalg.vector_norm(
        artifact["mean_difference"]["Spain"][2]
    ).item() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="model revision"):
        load_mean_difference_artifact(
            path,
            expected_layers=[2, 3],
            expected_model_revision="different",
        )

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.metrics import partial_correlation_with_ci, pearson_with_ci
from src.model_utils import MODEL_REVISIONS
from src.scale_phase import (
    MODEL_14B,
    PRIMARY_DIRECTION_METHOD,
    MD_DIRECTION_METHOD,
    WIKITEXT_REVISION,
    WIKITEXT_SELECTION,
    compare_scale_runs,
    plot_f7_scale_comparison,
    validate_qwen14b_lens_provenance,
)


def _prompt_hash(index: int) -> str:
    return hashlib.sha256(f"prompt-{index}".encode()).hexdigest()


def test_validate_qwen14b_lens_provenance_checks_revision_dimensions_and_fit(
    tmp_path,
) -> None:
    lens_path = tmp_path / "qwen2.5-14b_jlens_100prompts.pt"
    lens_path.write_bytes(b"synthetic lens bytes")
    metadata_path = tmp_path / "qwen2.5-14b_jlens_100prompts.json"
    metadata = {
        "seed": 1729,
        "model_id": MODEL_14B,
        "model_revision": MODEL_REVISIONS[MODEL_14B],
        "wikitext_revision": WIKITEXT_REVISION,
        "selection": WIKITEXT_SELECTION,
        "prompt_sha256": [_prompt_hash(index) for index in range(100)],
        "n_prompts_requested": 100,
        "n_prompts_fitted": 100,
        "source_layers": [2, 3],
        "target_layer": 4,
        "dim_batch": 128,
        "max_seq_len": 128,
        "checkpoint_every": 10,
        "lens_path": str(lens_path),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    lens_model = SimpleNamespace(n_layers=5, d_model=3)
    bundle = SimpleNamespace(
        model_id=MODEL_14B,
        revision=MODEL_REVISIONS[MODEL_14B],
        lens_model=lens_model,
    )
    lens = SimpleNamespace(
        d_model=3,
        source_layers=[2, 3],
        n_prompts=100,
        jacobians={
            2: torch.eye(3, dtype=torch.float32),
            3: torch.eye(3, dtype=torch.float32),
        },
    )

    result = validate_qwen14b_lens_provenance(
        bundle,
        lens,
        metadata,
        lens_path=lens_path,
        metadata_path=metadata_path,
    )

    assert result["status"] == "PASS"
    assert result["d_model"] == 3
    assert result["n_layers"] == 5
    assert result["workspace_layers"] == [2, 3]
    assert result["lens_sha256"] == hashlib.sha256(lens_path.read_bytes()).hexdigest()

    bad_revision = dict(metadata, model_revision="wrong")
    with pytest.raises(ValueError, match="model revision"):
        validate_qwen14b_lens_provenance(
            bundle,
            lens,
            bad_revision,
            lens_path=lens_path,
            metadata_path=None,
        )

    bad_lens = SimpleNamespace(
        d_model=3,
        source_layers=[2, 3],
        n_prompts=100,
        jacobians={2: torch.eye(3), 3: torch.zeros(3, 4)},
    )
    with pytest.raises(ValueError, match="shape"):
        validate_qwen14b_lens_provenance(
            bundle,
            bad_lens,
            metadata,
            lens_path=lens_path,
            metadata_path=metadata_path,
        )


def _statistic_payload(
    names: list[str],
    write: np.ndarray,
    read: np.ndarray,
    causal: np.ndarray,
    predicted: np.ndarray,
    *,
    seed: int,
) -> dict:
    return {
        "n": len(names),
        "variables": {
            "write": "aggregate.write_abs_mean",
            "read": "aggregate.read_abs_mean",
            "causal": "ablation.positive_damage",
        },
        "partial_correlations": {
            "causal_read_given_write": {
                "status": "ESTIMATED",
                **partial_correlation_with_ci(
                    causal,
                    read,
                    write,
                    n_bootstrap=200,
                    seed=seed,
                ),
            },
            "causal_write_given_read": {
                "status": "ESTIMATED",
                **partial_correlation_with_ci(
                    causal,
                    write,
                    read,
                    n_bootstrap=200,
                    seed=seed + 1,
                ),
            },
        },
        "pearson": {
            "predicted_vs_real": {
                "status": "ESTIMATED",
                **pearson_with_ci(
                    predicted,
                    causal,
                    n_bootstrap=200,
                    seed=seed + 2,
                ),
            }
        },
        "raw_analysis_vectors": {
            "item_names": names,
            "write_strength": write.tolist(),
            "read_strength": read.tolist(),
            "causal_positive_damage": causal.tolist(),
            "predicted_positive_damage": predicted.tolist(),
        },
    }


def _scale_run(tag: str, *, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 60
    names = [f"item-{index:03d}" for index in range(n)]
    read = np.abs(rng.normal(size=n)) + 0.25
    write = 0.9 * read + rng.normal(scale=0.35, size=n) + 1.0
    if tag == "7B":
        causal = 0.25 * read + 0.65 * write + rng.normal(scale=0.35, size=n)
        model = "Qwen/Qwen2.5-7B-Instruct"
        layers = [11, 12]
        g2_status = "FAIL"
    else:
        causal = 2.0 * read + 0.05 * write + rng.normal(scale=0.20, size=n)
        model = MODEL_14B
        layers = [19, 20]
        g2_status = "PASS"
    predicted = causal + rng.normal(scale=0.25, size=n)
    methods = {
        PRIMARY_DIRECTION_METHOD: _statistic_payload(
            names,
            write,
            read,
            causal,
            predicted,
            seed=seed + 10,
        ),
        MD_DIRECTION_METHOD: _statistic_payload(
            names,
            write * 1.03,
            read * 0.97,
            causal * 0.95,
            predicted * 0.96,
            seed=seed + 20,
        ),
    }
    revision = MODEL_REVISIONS[model]
    return {
        "gates": {
            "metadata": {
                "model_id": model,
                "model_revision": revision,
                "workspace_layers": layers,
            },
            "gates": {
                "g1": {
                    "status": "PASS",
                    "n": 20,
                    "threshold_mean_kl": 1e-3,
                    "max_prompt_mean_kl": 1e-8,
                },
                "g2": {
                    "status": g2_status,
                    "directional_subgate": "PASS",
                    "strict_criterion": "synthetic",
                    "clean_metric": 5.0,
                    "min_spider_jlens_rank": 1,
                },
                "g3": {
                    "status": "PASS",
                    "n": n,
                    "attribution_reliable": True,
                    "correlation": methods[PRIMARY_DIRECTION_METHOD]["pearson"][
                        "predicted_vs_real"
                    ],
                    "meaning": "synthetic",
                },
            },
        },
        "md_validation": {
            "status": "PASS",
            "criteria": {"synthetic": True},
            "n_concepts": 40,
            "metadata": {
                "model_id": model,
                "fixed_validation_layer": layers[-1],
            },
        },
        "twohop": {
            "status": "COMPUTED",
            "metadata": {
                "model_id": model,
                "model_revision": revision,
                "workspace_layers": layers,
            },
            "corpus_criterion": {
                "status": "PASS",
                "n_clean_eligible": n,
            },
            "sample_counts": {
                "n_combined": n,
                "n_clean_eligible": n,
                "n_by_method": {
                    method: {"assigned": n, "successful": n} for method in methods
                },
            },
            "analyses": {"ablation": {"by_method": methods}},
        },
        "lens_provenance": {"kind": "synthetic"},
    }


@pytest.fixture(scope="module")
def synthetic_comparison() -> dict:
    return compare_scale_runs(
        {
            "7B": _scale_run("7B", seed=101),
            "14B": _scale_run("14B", seed=102),
        },
        n_bootstrap=200,
        seed=55,
    )


def test_compare_scale_runs_reports_p1_ablation_attribution_gates_and_ns(
    synthetic_comparison,
) -> None:
    comparison = synthetic_comparison

    assert comparison["status"] == "COMPUTED"
    assert comparison["models"]["7B"]["gates"]["strict_workspace_usable"] is False
    assert comparison["models"]["14B"]["gates"]["strict_workspace_usable"] is True
    for tag in ("7B", "14B"):
        for method in (PRIMARY_DIRECTION_METHOD, MD_DIRECTION_METHOD):
            summary = comparison["models"][tag]["methods"][method]
            assert summary["n"] == 60
            assert summary["mean_ablation_positive_damage"]["status"] == "ESTIMATED"
            assert summary["attribution_predicted_vs_real"]["status"] == "ESTIMATED"
    paired = comparison["paired_14b_minus_7b"][PRIMARY_DIRECTION_METHOD]
    assert paired["n_common"] == 60
    assert (
        paired["delta_14b_minus_7b"]["partial_causal_read_given_write"]["estimate"]
        > 0.0
    )
    assert comparison["p1_interpretation"]["status"] == "DESCRIPTIVE_ESTIMATES_ONLY"


def test_plot_f7_scale_comparison_writes_bootstrap_ci_figure(
    tmp_path,
    synthetic_comparison,
) -> None:
    path = tmp_path / "f7.png"

    result = plot_f7_scale_comparison(synthetic_comparison, path)

    assert result == path.resolve()
    assert path.is_file()
    assert path.stat().st_size > 10_000

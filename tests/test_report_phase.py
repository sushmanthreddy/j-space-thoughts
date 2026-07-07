from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.report_phase import (
    MODEL_7B_KEY,
    ReportCompletenessError,
    build_report_summary,
    render_report,
    validate_report_inputs,
)


def _stat(
    estimate: float = 0.6,
    low: float = 0.2,
    high: float = 0.8,
    *,
    n: int = 8,
    status: str = "ESTIMATED",
) -> dict:
    return {
        "status": status,
        "n": n,
        "estimate": estimate,
        "ci_level": 0.95,
        "ci_low": low,
        "ci_high": high,
        "n_bootstrap": 500,
        "seed": 1729,
    }


def _regression(*, n: int = 8, interaction: bool = False) -> dict:
    coefficients = {"intercept": 0.0, "write": 0.02, "read": 0.65}
    intervals = {
        "intercept": {"ci_low": -0.1, "ci_high": 0.1},
        "write": {"ci_low": -0.2, "ci_high": 0.2},
        "read": {"ci_low": 0.2, "ci_high": 0.9},
    }
    if interaction:
        coefficients["write_x_read"] = 0.1
        intervals["write_x_read"] = {"ci_low": -0.1, "ci_high": 0.3}
    return {
        "status": "ESTIMATED",
        "n": n,
        "coefficients": coefficients,
        "coefficient_intervals": intervals,
        "interaction": interaction,
        "r_squared": 0.4,
        "r_squared_interval": {"ci_low": 0.2, "ci_high": 0.6},
        "n_bootstrap": 500,
        "seed": 1729,
    }


def _method_result(*, n: int = 8) -> dict:
    return {
        "n": n,
        "partial_correlations": {
            "causal_read_given_write": _stat(n=n),
            "causal_write_given_read": _stat(0.01, -0.2, 0.2, n=n),
        },
        "regressions": {
            "causal_on_write_plus_read": _regression(n=n),
            "causal_on_write_times_read": _regression(n=n, interaction=True),
        },
        "pearson": {"predicted_vs_real": _stat(0.7, 0.4, 0.9, n=n)},
        "raw_analysis_vectors": {
            "item_names": [f"item-{index}" for index in range(n)],
            "write_strength": [float(index + 1) for index in range(n)],
            "read_strength": [float(index + 2) for index in range(n)],
            "causal_positive_damage": [float(index) for index in range(n)],
            "predicted_positive_damage": [float(index) for index in range(n)],
        },
    }


def _twohop_row(
    name: str,
    write: float,
    read: float,
    causal: float,
) -> dict:
    return {
        "name": name,
        "measurement_status": "OK",
        "direction_method": "jlens_raw_wu_j",
        "aggregate": {"write_abs_mean": write, "read_abs_mean": read},
        "ablation": {"positive_damage": causal},
        "output_suppression": {"concept": {"positive_damage": 0.0}},
    }


def _weight_family(*, n: int = 8) -> dict:
    additive = _regression(n=n)
    interaction = _regression(n=n, interaction=True)
    return {
        "n": n,
        "partial_correlations": {
            "causal_weight_read_given_write": _stat(n=n),
            "causal_write_given_weight_read": _stat(0.0, -0.2, 0.2, n=n),
        },
        "pearson": {"write_vs_weight_read": _stat(0.2, -0.1, 0.4, n=n)},
        "regressions": {
            "causal_on_write_plus_weight_read": additive,
            "causal_on_write_times_weight_read": interaction,
        },
        "standardized_regression": additive,
    }


def _create_figure(root: Path, name: str) -> str:
    path = root / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic image")
    return str(path.relative_to(root))


def _complete_metrics(root: Path) -> dict:
    figure = {
        name: _create_figure(root, f"{name}.png")
        for name in (
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "random",
            "capability",
            "narration",
            "alignment",
        )
    }
    rows = [
        _twohop_row("hh", 10.0, 10.0, 10.0),
        _twohop_row("hh2", 9.0, 9.0, 9.0),
        _twohop_row("posthoc-candidate", 10.0, 1.0, 0.1),
        _twohop_row("hl2", 9.0, 2.0, 3.0),
        _twohop_row("lh", 1.0, 10.0, 10.0),
        _twohop_row("lh2", 2.0, 9.0, 9.0),
        _twohop_row("ll", 1.0, 1.0, 1.0),
        _twohop_row("ll2", 2.0, 2.0, 2.0),
    ]
    methods = {
        "jlens_raw_wu_j": _method_result(),
        "mean_difference": _method_result(),
    }
    weight_methods = {
        method: {
            "n": 8,
            "weight_families": {
                "mlp": _weight_family(),
                "attention": _weight_family(),
            },
        }
        for method in ("jlens_raw_wu_j", "mean_difference")
    }
    ambiguity_stats = {
        "swap_flip_rate": _stat(0.5, 0.2, 0.8, n=2),
        "counterbalance_robust_swap_flip_rate": _stat(0.5, 0.1, 0.9, n=2),
        "internal_ablation_positive_damage": _stat(1.2, 0.5, 1.8, n=2),
        "internal_minus_suppression_damage": _stat(1.2, 0.5, 1.8, n=2),
        "output_suppression_positive_damage": _stat(0.0, 0.0, 0.0, n=2),
        "ablation_exceeds_suppression_rate": _stat(1.0, 1.0, 1.0, n=2),
        "counterbalance_agreement_rate": _stat(1.0, 1.0, 1.0, n=2),
    }
    scale_methods = {
        method: {
            "status": "ESTIMATED",
            "n": 8,
            "partial_correlations": {
                "causal_read_given_write": _stat(),
                "causal_write_given_read": _stat(0.0, -0.2, 0.2),
            },
            "mean_ablation_positive_damage": _stat(1.0, 0.5, 1.5),
            "attribution_predicted_vs_real": _stat(0.7, 0.4, 0.9),
        }
        for method in ("jlens_raw_wu_j", "mean_difference")
    }
    return {
        "metadata": {"model_id": "Qwen/Qwen2.5-7B-Instruct"},
        "gates": {
            "g1": {
                "status": "PASS",
                "n": 20,
                "max_prompt_mean_kl": 1e-8,
            },
            "g2": {
                "status": "PASS",
                "directional_subgate": "PASS",
                "min_spider_jlens_rank": 1,
            },
            "g3": {
                "status": "PASS",
                "n": 20,
                "attribution_reliable": True,
                "correlation": _stat(0.7, 0.4, 0.9, n=20),
                "figure": figure["f5"],
            },
        },
        "concept_vectors": {
            "status": "PASS",
            "figure": figure["alignment"],
        },
        "twohop": {
            MODEL_7B_KEY: {
                "schema_version": "twohop-phase-v1",
                "status": "COMPUTED",
                "metadata": {
                    "model_id": "Qwen/Qwen2.5-7B-Instruct",
                    "model_revision": "revision",
                },
                "rows": rows,
                "analyses": {"ablation": {"by_method": methods}},
                "figures": {
                    "f1": figure["f1"],
                    "f2": figure["f2"],
                    "f6": figure["f6"],
                },
            }
        },
        "controls": {
            MODEL_7B_KEY: {
                "random_direction_null": {
                    "aggregate": {"mean_abs_random_delta": _stat(0.1, 0.05, 0.2)}
                },
                "absent_coordinate_null": {
                    "status": "PASS",
                    "aggregate": {"mean_abs_delta": _stat(0.05, 0.0, 0.1)},
                },
                "capability": {
                    "n_fixed_texts": 8,
                    "general_language": {
                        "mean_clean_nll": 2.0,
                        "mean_edited_nll": 2.1,
                        "mean_delta_nll": 0.1,
                    },
                    "twohop": {"clean_accuracy": 0.9, "edited_accuracy": 0.85},
                },
                "known_narration": {
                    "status": "PASS",
                    "reproduction_gate": {
                        "status": "PASS",
                        "n_reproduced": 7,
                        "n_passages": 8,
                    },
                },
                "logit_lens_identity_jacobian": {
                    "predictor": _stat(0.5, 0.2, 0.8, status="COMPUTED")
                },
                "core_output_suppression_assertion": {
                    "status": "PASS",
                    "n_rows": len(rows),
                },
                "figures": {
                    "f3_internal_vs_output_suppression": figure["f3"],
                    "random_null": figure["random"],
                    "capability": figure["capability"],
                    "known_narration": figure["narration"],
                },
                "limitations": ["Synthetic control limitation."],
            }
        },
        "localization": {
            MODEL_7B_KEY: {
                "schema_version": "localization-phase-v2",
                "status": "COMPUTED",
                "metadata": {"model_id": "Qwen/Qwen2.5-7B-Instruct"},
                "population_weight_read": {
                    "status": "COMPUTED",
                    "analysis": {
                        "status": "COMPUTED",
                        "by_method": weight_methods,
                    },
                },
                "figures": {"f4": figure["f4"], "f6": figure["f6"]},
            }
        },
        "ambiguity": {
            MODEL_7B_KEY: {
                "schema_version": "ambiguity-phase-v1",
                "direction_method": "jlens_raw_wu_j",
                "rows": [
                    {
                        "measurement_status": "OK",
                        "counterbalanced": {
                            "clean_committed_margin_by_variant": [2.0, 2.0],
                            "clean_clamped_swap": {
                                "variant_positive_damage": [3.0, 1.0]
                            },
                        },
                    },
                    {
                        "measurement_status": "OK",
                        "counterbalanced": {
                            "clean_committed_margin_by_variant": [2.0, 2.0],
                            "clean_clamped_swap": {
                                "variant_positive_damage": [3.0, 3.0]
                            },
                        },
                    },
                ],
                "p3": {
                    "verdict": "supported",
                    "direction_method": "jlens_raw_wu_j",
                    "overall": {"n": 2, "statistics": ambiguity_stats},
                },
                "figures": {"f8": figure["f8"]},
            }
        },
        "scale_comparison": {
            "schema_version": "scale-phase-v1",
            "status": "COMPUTED",
            "qwen14b_gates": {
                "g1": {"status": "PASS", "n": 20, "max_prompt_mean_kl": 1e-8},
                "g2": {"status": "PASS", "directional_subgate": "PASS"},
                "g3_attribution_validation": {
                    "status": "PASS",
                    "n": 8,
                    "attribution_reliable": True,
                    "correlation": _stat(0.7, 0.4, 0.9),
                },
                "strict_workspace_usable": True,
            },
            "comparison": {
                "models": {
                    "7B": {
                        "gates": {"strict_workspace_usable": True},
                        "twohop_status": "COMPUTED",
                        "methods": scale_methods,
                    },
                    "14B": {
                        "gates": {"strict_workspace_usable": True},
                        "twohop_status": "COMPUTED",
                        "methods": scale_methods,
                    },
                },
                "paired_14b_minus_7b": {},
            },
            "f7": figure["f7"],
        },
    }


def test_completeness_requires_00_through_06_and_f1_through_f8(tmp_path) -> None:
    metrics = _complete_metrics(tmp_path)

    complete = validate_report_inputs(metrics, root=tmp_path)

    assert complete["status"] == "PASS"
    assert complete["missing_metrics"] == []
    assert complete["missing_figures"] == []
    assert complete["required_phase_notebooks"] == [
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
    ]
    assert metrics.get("blackmail") is None
    assert metrics["scale_comparison"].get("qwen32b") is None

    broken = copy.deepcopy(metrics)
    del broken["scale_comparison"]
    with pytest.raises(ReportCompletenessError) as error:
        validate_report_inputs(broken, root=tmp_path)
    assert "scale_comparison (notebook 06)" in str(error.value)
    assert "required figure F7" in str(error.value)


def test_p1_requires_direction_weight_and_strict_workspace_robustness(tmp_path) -> None:
    metrics = _complete_metrics(tmp_path)
    summary = build_report_summary(metrics)
    seven = next(
        row for row in summary["p1"]["model_assessments"] if row["model"] == "7B"
    )
    assert seven["status"] == "SUPPORTED_PATTERN"
    assert summary["p1"]["status"] == "SUPPORTED_PATTERN"

    no_weight = copy.deepcopy(metrics)
    del no_weight["localization"][MODEL_7B_KEY]["population_weight_read"]
    unsupported = build_report_summary(no_weight)["p1"]
    assert unsupported["status"] == "UNSUPPORTED"
    seven = next(
        row for row in unsupported["model_assessments"] if row["model"] == "7B"
    )
    assert seven["weight_robustness_available"] is False
    assert any("weight READ robustness" in reason for reason in seven["reasons"])

    failed_gate = copy.deepcopy(metrics)
    failed_gate["gates"]["g2"]["status"] = "FAIL"
    assert build_report_summary(failed_gate)["p1"]["status"] == "UNSUPPORTED"


def test_p2_is_posthoc_and_unestablished_when_controls_fail(tmp_path) -> None:
    metrics = _complete_metrics(tmp_path)
    metrics["controls"][MODEL_7B_KEY]["known_narration"]["status"] = "FAIL"
    metrics["controls"][MODEL_7B_KEY]["absent_coordinate_null"]["status"] = "FAIL"

    p2 = build_report_summary(metrics)["p2"]

    assert p2["status"] == "UNESTABLISHED"
    assert p2["analysis_role"].startswith("posthoc descriptive")
    assert [row["name"] for row in p2["candidates"]] == ["posthoc-candidate"]
    assert p2["thresholds"]["write_q75"] == pytest.approx(9.25)
    assert all(
        row["analysis_role"] == "posthoc_quantile_screen_candidate"
        for row in p2["candidates"]
    )


def test_p3_reports_mean_variant_and_both_flip_rates_with_structural_caveat(
    tmp_path,
) -> None:
    p3 = build_report_summary(_complete_metrics(tmp_path))["p3"]

    assert p3["mean_margin_flip_rate"]["estimate"] == pytest.approx(0.5)
    assert p3["variant_1_flip_rate"]["estimate"] == pytest.approx(1.0)
    assert p3["variant_2_flip_rate"]["estimate"] == pytest.approx(0.5)
    assert p3["both_variants_flip_rate"]["estimate"] == pytest.approx(0.5)
    assert p3["both_variants_flip_rate"]["ci_low"] == pytest.approx(0.1)
    assert "structural" in p3["structural_zero_suppression_caveat"]
    assert p3["g2_context"]["strict_status"] == "PASS"


def test_report_is_deterministic_includes_cis_controls_scale_and_limitations(
    tmp_path,
) -> None:
    metrics = _complete_metrics(tmp_path)

    first = render_report(metrics, root=tmp_path)
    second = render_report(copy.deepcopy(metrics), root=tmp_path)

    assert first == second
    markdown = first["markdown"]
    assert "Preregistered hypothesis" in markdown
    assert "95% CI" in markdown
    assert "N=8" in markdown
    assert "β WRITE×READ" in markdown
    assert "both variants flip | 0.500 (95% CI 0.100, 0.900; N=2)" in markdown
    assert "Matched random-direction" not in markdown  # table uses concise label
    assert "random-direction null" in markdown
    assert r"mean \|Δ\|=0.100 (95% CI 0.050, 0.200; N=8)" in markdown
    assert (
        r"absent-coordinate null | PASS; mean \|Δ\|=0.050 "
        "(95% CI 0.000, 0.100; N=8)"
    ) in markdown
    assert "Scale comparison" in markdown
    assert "Flash/SDPA" in markdown
    assert "legacy first-10-prompt checkpoint" in markdown
    assert first["summary"]["p4"]["status"] == "NOT_RUN_OPTIONAL"
    assert first["summary"]["scale"]["qwen32b"]["status"] == "NOT_AVAILABLE_OPTIONAL"


def test_notebook08_persists_guarded_stage4_fallback_and_is_executed() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks/08_report.ipynb"
    notebook = json.loads(path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [cell.get("execution_count") for cell in code_cells] == list(
        range(1, len(code_cells) + 1)
    )
    assert all(
        output.get("output_type") != "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert all(cell.get("id") for cell in notebook["cells"])
    assert "persist_stage4" in source
    assert "results/RESULTS.md" in source
    assert "stage3_notebooks" in source
    assert "SKIPPED_PREREQUISITE" in source
    assert "science_executed" in source
    assert "model_inference_run" in source
    assert "P1 | **NOT_TESTED**" in source
    assert "hypothesis" in source.lower()
    assert "load_model(" not in source
    assert "AutoModel" not in source

    live_metrics = json.loads(
        (Path(__file__).resolve().parents[1] / "results/metrics.json").read_text()
    )["repair_v2"]
    assert live_metrics["gate_ledger"]["stage4_report"] == "COMPLETE"
    assert live_metrics["gate_ledger"]["stage3_science"] == (
        "SKIPPED_PREREQUISITE"
    )
    assert live_metrics["stage4_report"]["predictions"] == {
        "P1": "NOT_TESTED",
        "P2": "NOT_TESTED",
        "P3": "NOT_TESTED",
    }

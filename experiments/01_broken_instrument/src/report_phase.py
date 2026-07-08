"""Deterministic, schema-validated final report over persisted metrics.

This module is deliberately model-free.  It separates *completeness* (were the
required phases and figures produced?) from *support* (did the preregistered
predictions survive all required robustness checks?).  Failed gates and failed
controls are reportable results; missing phase artifacts are completeness
errors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics import binomial_rate_with_ci


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "report-phase-v1"
MODEL_7B_KEY = "qwen2.5-7b-instruct"
MODEL_14B_KEY = "qwen2.5-14b-instruct"
PRIMARY_DIRECTION = "jlens_raw_wu_j"
MD_DIRECTION = "mean_difference"
PREREGISTERED_HYPOTHESIS = (
    "A concept's causal influence on behavior is governed by whether "
    "behavior-relevant downstream circuits READ its residual-stream direction, "
    "not merely by how strongly that direction is WRITTEN into the residual "
    "stream. The preregistered residual prediction is that CAUSAL tracks READ "
    "conditional on WRITE, while WRITE contributes approximately zero once READ "
    "is controlled. This pattern must survive both raw J-Lens and independent "
    "mean-difference directions and both attribution- and weight-based READ."
)


class ReportCompletenessError(ValueError):
    """Required report inputs are missing or malformed."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(str(issue) for issue in issues)
        message = "Report completeness failed:\n- " + "\n- ".join(self.issues)
        super().__init__(message)


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load one metrics JSON object without mutating it."""

    target = Path(path)
    try:
        with target.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read metrics JSON at {target}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object")
    return payload


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _finite(value: Any, *, path: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _model_label(key: str, payload: Mapping[str, Any] | None = None) -> str:
    model_id = ""
    if payload is not None:
        metadata = payload.get("metadata", {})
        if isinstance(metadata, Mapping):
            model_id = str(metadata.get("model_id", ""))
    text = f"{key} {model_id}".casefold()
    if "14b" in text:
        return "14B"
    if "7b" in text:
        return "7B"
    return key


def _compact_statistic(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "NOT_AVAILABLE"}
    source_status = str(
        value.get("status", "ESTIMATED" if "estimate" in value else "UNKNOWN")
    )
    has_interval = all(key in value for key in ("estimate", "ci_low", "ci_high"))
    status = (
        "ESTIMATED"
        if has_interval and source_status in {"ESTIMATED", "COMPUTED", "PASS"}
        else source_status
    )
    result: dict[str, Any] = {"status": status}
    if source_status != status:
        result["source_status"] = source_status
    for key in (
        "n",
        "estimate",
        "ci_level",
        "ci_low",
        "ci_high",
        "n_bootstrap",
        "seed",
        "error",
        "error_type",
    ):
        if key in value:
            result[key] = value[key]
    return result


def _compact_regression(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "NOT_AVAILABLE"}
    result: dict[str, Any] = {
        "status": str(
            value.get("status", "ESTIMATED" if "coefficients" in value else "UNKNOWN")
        )
    }
    for key in (
        "n",
        "coefficients",
        "coefficient_intervals",
        "r_squared",
        "r_squared_interval",
        "interaction",
        "n_bootstrap",
        "seed",
        "error",
    ):
        if key in value:
            result[key] = value[key]
    return result


def _stat_estimated(statistic: Mapping[str, Any]) -> bool:
    return statistic.get("status") == "ESTIMATED" and all(
        key in statistic and statistic[key] is not None
        for key in ("estimate", "ci_low", "ci_high")
    )


def _positive_ci(statistic: Mapping[str, Any]) -> bool:
    return _stat_estimated(statistic) and float(statistic["ci_low"]) > 0.0


def _zero_compatible_ci(statistic: Mapping[str, Any]) -> bool:
    return _stat_estimated(statistic) and (
        float(statistic["ci_low"]) <= 0.0 <= float(statistic["ci_high"])
    )


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def collect_figure_catalog(
    metrics: Mapping[str, Any],
    *,
    root: str | Path = ROOT,
) -> list[dict[str, Any]]:
    """Collect numbered and control figures in a frozen display order."""

    project_root = Path(root)
    twohop = metrics.get("twohop", {}).get(MODEL_7B_KEY, {})
    twohop_14b = metrics.get("twohop", {}).get(MODEL_14B_KEY, {})
    controls = metrics.get("controls", {}).get(MODEL_7B_KEY, {})
    localization = metrics.get("localization", {}).get(MODEL_7B_KEY, {})
    ambiguity = metrics.get("ambiguity", {}).get(MODEL_7B_KEY, {})
    gates = metrics.get("gates", {})
    scale = metrics.get("scale_comparison", {})
    concepts = metrics.get("concept_vectors", {})

    candidates: list[tuple[str, str, Any, bool]] = [
        ("F1", "CAUSAL versus READ/WRITE", twohop.get("figures", {}).get("f1"), True),
        (
            "F1_14B",
            "CAUSAL versus READ/WRITE (14B)",
            twohop_14b.get("figures", {}).get("f1"),
            False,
        ),
        ("F2", "Conditional WRITE and READ", twohop.get("figures", {}).get("f2"), True),
        (
            "F2_14B",
            "Conditional WRITE and READ (14B)",
            twohop_14b.get("figures", {}).get("f2"),
            False,
        ),
        (
            "F3",
            "Internal ablation versus output suppression",
            controls.get("figures", {}).get("f3_internal_vs_output_suppression"),
            True,
        ),
        ("F4", "READ localization", localization.get("figures", {}).get("f4"), True),
        (
            "F5",
            "Attribution versus real ablation",
            gates.get("g3", {}).get("figure") if isinstance(gates, Mapping) else None,
            True,
        ),
        (
            "F6_14B",
            "Direction robustness (14B)",
            twohop_14b.get("figures", {}).get("f6"),
            False,
        ),
        (
            "F6",
            "Direction/READ robustness",
            localization.get("figures", {}).get("f6")
            or twohop.get("figures", {}).get("f6"),
            True,
        ),
        ("F7", "Scale comparison", scale.get("f7"), True),
        ("F8", "Ambiguity WRITE/READ", ambiguity.get("figures", {}).get("f8"), True),
        (
            "control_random",
            "Matched random-direction null",
            controls.get("figures", {}).get("random_null"),
            True,
        ),
        (
            "control_capability",
            "Capability controls",
            controls.get("figures", {}).get("capability"),
            True,
        ),
        (
            "control_narration",
            "Known-narration control",
            controls.get("figures", {}).get("known_narration"),
            True,
        ),
        (
            "concept_alignment",
            "MD/J-Lens concept alignment",
            concepts.get("figure"),
            False,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for figure_id, title, raw_path, required in candidates:
        resolved = (
            _resolve_path(project_root, raw_path)
            if isinstance(raw_path, (str, Path)) and str(raw_path)
            else None
        )
        rows.append(
            {
                "id": figure_id,
                "title": title,
                "required": required,
                "path": _display_path(project_root, resolved) if resolved else None,
                "absolute_path": str(resolved) if resolved else None,
                "exists": bool(resolved is not None and resolved.is_file()),
                "bytes": (
                    int(resolved.stat().st_size)
                    if resolved is not None and resolved.is_file()
                    else 0
                ),
            }
        )
    return rows


def validate_report_inputs(
    metrics: Mapping[str, Any],
    *,
    root: str | Path = ROOT,
    require_files: bool = True,
) -> dict[str, Any]:
    """Validate phases 00–06 and F1–F8/control figure availability."""

    issues: list[str] = []

    def require_mapping(path: str, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            issues.append(f"missing mapping {path}")
            return {}
        return value

    gates = require_mapping("gates (notebook 00)", metrics.get("gates"))
    for gate in ("g1", "g2", "g3"):
        if not isinstance(gates.get(gate), Mapping):
            issues.append(f"missing gates.{gate} (notebook 00)")
    concepts = require_mapping(
        "concept_vectors (notebook 01)", metrics.get("concept_vectors")
    )
    if "status" not in concepts:
        issues.append("concept_vectors.status is missing")
    twohop_root = require_mapping("twohop (notebook 02)", metrics.get("twohop"))
    twohop = require_mapping(f"twohop.{MODEL_7B_KEY}", twohop_root.get(MODEL_7B_KEY))
    if twohop and twohop.get("schema_version") != "twohop-phase-v1":
        issues.append("twohop 7B schema_version must be twohop-phase-v1")
    controls_root = require_mapping("controls (notebook 03)", metrics.get("controls"))
    controls = require_mapping(
        f"controls.{MODEL_7B_KEY}", controls_root.get(MODEL_7B_KEY)
    )
    for key in (
        "random_direction_null",
        "absent_coordinate_null",
        "capability",
        "known_narration",
        "logit_lens_identity_jacobian",
        "core_output_suppression_assertion",
    ):
        if controls and not isinstance(controls.get(key), Mapping):
            issues.append(f"controls 7B missing {key}")
    if controls:
        raw_twohop_n = sum(
            1
            for row in twohop.get("rows", [])
            if isinstance(row, Mapping)
            and row.get("measurement_status") == "OK"
            and row.get("direction_method") == PRIMARY_DIRECTION
        )
        suppression_n = controls.get("core_output_suppression_assertion", {}).get(
            "n_rows"
        )
        if suppression_n != raw_twohop_n:
            issues.append(
                "core output suppression coverage does not match successful raw two-hop rows"
            )
    localization_root = require_mapping(
        "localization (notebook 04)", metrics.get("localization")
    )
    localization = require_mapping(
        f"localization.{MODEL_7B_KEY}", localization_root.get(MODEL_7B_KEY)
    )
    if localization and localization.get("schema_version") != "localization-phase-v2":
        issues.append("localization 7B schema_version must be localization-phase-v2")
    population_analysis = (
        localization.get("population_weight_read", {}).get("analysis", {})
        if localization
        else {}
    )
    weight_methods = (
        population_analysis.get("by_method", {})
        if isinstance(population_analysis, Mapping)
        else {}
    )
    for method in (PRIMARY_DIRECTION, MD_DIRECTION):
        families = (
            weight_methods.get(method, {}).get("weight_families", {})
            if isinstance(weight_methods, Mapping)
            else {}
        )
        for family in ("mlp", "attention"):
            interaction = (
                families.get(family, {})
                .get("regressions", {})
                .get("causal_on_write_times_weight_read")
                if isinstance(families, Mapping)
                else None
            )
            if (
                not isinstance(interaction, Mapping)
                or interaction.get("status") != "ESTIMATED"
            ):
                issues.append(
                    f"localization missing {method}/{family} weight READ interaction regression"
                )
    ambiguity_root = require_mapping(
        "ambiguity (notebook 05)", metrics.get("ambiguity")
    )
    ambiguity = require_mapping(
        f"ambiguity.{MODEL_7B_KEY}", ambiguity_root.get(MODEL_7B_KEY)
    )
    if ambiguity and ambiguity.get("schema_version") != "ambiguity-phase-v1":
        issues.append("ambiguity 7B schema_version must be ambiguity-phase-v1")
    if ambiguity:
        for row in ambiguity.get("rows", []):
            if not isinstance(row, Mapping) or row.get("measurement_status") != "OK":
                continue
            for key, record in row.get("meta_counterbalanced", {}).items():
                if not isinstance(record, Mapping) or not isinstance(
                    record.get("output_suppression"), Mapping
                ):
                    issues.append(
                        f"ambiguity row {row.get('id')} meta token {key} lacks output suppression"
                    )
                    break
    scale = require_mapping(
        "scale_comparison (notebook 06)", metrics.get("scale_comparison")
    )
    if scale and scale.get("schema_version") != "scale-phase-v1":
        issues.append("scale_comparison schema_version must be scale-phase-v1")

    figures = collect_figure_catalog(metrics, root=root)
    if require_files:
        for figure in figures:
            if figure["required"] and not figure["exists"]:
                issues.append(
                    f"required figure {figure['id']} missing/nonempty path: "
                    f"{figure['path']!r}"
                )
    if issues:
        raise ReportCompletenessError(issues)
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "missing_metrics": [],
        "missing_figures": [],
        "required_phase_notebooks": ["00", "01", "02", "03", "04", "05", "06"],
        "optional_phase_notebooks": ["07 (P4 blackmail)"],
        "required_figure_ids": [f"F{index}" for index in range(1, 9)],
        "required_control_figure_ids": [
            "control_random",
            "control_capability",
            "control_narration",
        ],
        "figures": figures,
    }


def _gate_summary(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gates_7b = metrics.get("gates", {})
    if isinstance(gates_7b, Mapping):
        g1 = gates_7b.get("g1", {})
        g2 = gates_7b.get("g2", {})
        g3 = gates_7b.get("g3", {})
        rows.append(
            {
                "model": "7B",
                "g1_status": g1.get("status"),
                "g1_n": g1.get("n"),
                "g1_max_mean_kl": g1.get("max_prompt_mean_kl"),
                "g2_strict_status": g2.get("status"),
                "g2_directional_status": g2.get("directional_subgate"),
                "g2_min_jlens_rank": g2.get("min_spider_jlens_rank"),
                "g3_status": g3.get("status"),
                "g3_n": g3.get("n"),
                "g3_attribution_reliable": g3.get("attribution_reliable"),
                "g3_correlation": _compact_statistic(g3.get("correlation")),
                "strict_workspace_usable": (
                    g1.get("status") == "PASS" and g2.get("status") == "PASS"
                ),
                "context": (
                    "confirmatory"
                    if g1.get("status") == "PASS" and g2.get("status") == "PASS"
                    else "diagnostic because strict workspace gate failed"
                ),
            }
        )
    scale = metrics.get("scale_comparison", {})
    if isinstance(scale, Mapping):
        gates_14b = scale.get("qwen14b_gates")
        if isinstance(gates_14b, Mapping):
            g1 = gates_14b.get("g1", {})
            g2 = gates_14b.get("g2", {})
            g3 = gates_14b.get("g3_attribution_validation", {})
            strict = bool(gates_14b.get("strict_workspace_usable"))
            rows.append(
                {
                    "model": "14B",
                    "g1_status": g1.get("status"),
                    "g1_n": g1.get("n"),
                    "g1_max_mean_kl": g1.get("max_prompt_mean_kl"),
                    "g2_strict_status": g2.get("status"),
                    "g2_directional_status": g2.get("directional_subgate"),
                    "g2_min_jlens_rank": g2.get("min_spider_jlens_rank"),
                    "g3_status": g3.get("status"),
                    "g3_n": g3.get("n"),
                    "g3_attribution_reliable": g3.get("attribution_reliable"),
                    "g3_correlation": _compact_statistic(g3.get("correlation")),
                    "strict_workspace_usable": strict,
                    "context": "confirmatory" if strict else "diagnostic",
                }
            )
    return rows


def _p1_summary(
    metrics: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    gate_by_model = {str(row["model"]): row for row in gates}
    attribution_rows: list[dict[str, Any]] = []
    model_keys: dict[str, str] = {}
    twohop_root = metrics.get("twohop", {})
    if isinstance(twohop_root, Mapping):
        for key in sorted(twohop_root):
            payload = twohop_root[key]
            if not isinstance(payload, Mapping):
                continue
            model = _model_label(str(key), payload)
            model_keys[model] = str(key)
            methods = (
                payload.get("analyses", {}).get("ablation", {}).get("by_method", {})
            )
            if not isinstance(methods, Mapping):
                continue
            for method in (PRIMARY_DIRECTION, MD_DIRECTION):
                result = methods.get(method)
                if not isinstance(result, Mapping):
                    attribution_rows.append(
                        {
                            "model": model,
                            "method": method,
                            "status": "NOT_AVAILABLE",
                            "n": 0,
                        }
                    )
                    continue
                partials = result.get("partial_correlations", {})
                pearson = result.get("pearson", {})
                regressions = result.get("regressions", {})
                attribution_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "status": "ESTIMATED",
                        "n": int(result.get("n", 0)),
                        "strict_workspace_usable": bool(
                            gate_by_model.get(model, {}).get("strict_workspace_usable")
                        ),
                        "causal_read_given_write": _compact_statistic(
                            partials.get("causal_read_given_write")
                        ),
                        "causal_write_given_read": _compact_statistic(
                            partials.get("causal_write_given_read")
                        ),
                        "write_read_correlation": _compact_statistic(
                            pearson.get("write_vs_read")
                        ),
                        "regression_additive": _compact_regression(
                            regressions.get("causal_on_write_plus_read")
                        ),
                        "regression_interaction": _compact_regression(
                            regressions.get("causal_on_write_times_read")
                        ),
                    }
                )

    weight_rows: list[dict[str, Any]] = []
    localization_root = metrics.get("localization", {})
    if isinstance(localization_root, Mapping):
        for key in sorted(localization_root):
            payload = localization_root[key]
            if not isinstance(payload, Mapping):
                continue
            population = payload.get("population_weight_read")
            if not isinstance(population, Mapping):
                continue
            analysis = population.get("analysis", {})
            by_method = (
                analysis.get("by_method", {}) if isinstance(analysis, Mapping) else {}
            )
            model = _model_label(str(key), payload)
            for method in (PRIMARY_DIRECTION, MD_DIRECTION):
                method_result = (
                    by_method.get(method, {}) if isinstance(by_method, Mapping) else {}
                )
                families = (
                    method_result.get("weight_families", {})
                    if isinstance(method_result, Mapping)
                    else {}
                )
                for family in ("mlp", "attention"):
                    result = families.get(family)
                    if not isinstance(result, Mapping):
                        weight_rows.append(
                            {
                                "model": model,
                                "method": method,
                                "family": family,
                                "status": "NOT_AVAILABLE",
                                "n": 0,
                            }
                        )
                        continue
                    partials = result.get("partial_correlations", {})
                    pearson = result.get("pearson", {})
                    regressions = result.get("regressions", {})
                    additive = regressions.get(
                        "causal_on_write_plus_weight_read",
                        result.get("standardized_regression"),
                    )
                    weight_rows.append(
                        {
                            "model": model,
                            "method": method,
                            "family": family,
                            "status": "ESTIMATED",
                            "n": int(result.get("n", 0)),
                            "selection_conditioned": True,
                            "causal_read_given_write": _compact_statistic(
                                partials.get("causal_weight_read_given_write")
                            ),
                            "causal_write_given_read": _compact_statistic(
                                partials.get("causal_write_given_weight_read")
                            ),
                            "write_read_correlation": _compact_statistic(
                                pearson.get("write_vs_weight_read")
                            ),
                            "regression_additive": _compact_regression(additive),
                            "regression_interaction": _compact_regression(
                                regressions.get("causal_on_write_times_weight_read")
                            ),
                        }
                    )

    model_assessments: list[dict[str, Any]] = []
    models = sorted(set(model_keys) | {str(row["model"]) for row in weight_rows})
    for model in models:
        attr = [row for row in attribution_rows if row["model"] == model]
        weights = [row for row in weight_rows if row["model"] == model]
        required_attr = {
            row["method"]: row for row in attr if row.get("status") == "ESTIMATED"
        }
        required_weight = {
            (row["method"], row["family"]): row
            for row in weights
            if row.get("status") == "ESTIMATED"
        }
        attr_complete = set(required_attr) == {PRIMARY_DIRECTION, MD_DIRECTION}
        weight_complete = set(required_weight) == {
            (method, family)
            for method in (PRIMARY_DIRECTION, MD_DIRECTION)
            for family in ("mlp", "attention")
        }
        read_stats = [
            row["causal_read_given_write"] for row in required_attr.values()
        ] + [row["causal_read_given_write"] for row in required_weight.values()]
        write_stats = [
            row["causal_write_given_read"] for row in required_attr.values()
        ] + [row["causal_write_given_read"] for row in required_weight.values()]
        all_estimated = bool(read_stats) and all(
            _stat_estimated(statistic) for statistic in [*read_stats, *write_stats]
        )
        positive_read_pattern = all_estimated and all(
            _positive_ci(statistic) for statistic in read_stats
        )
        zero_compatible_write_pattern = all_estimated and all(
            _zero_compatible_ci(statistic) for statistic in write_stats
        )
        strict = bool(gate_by_model.get(model, {}).get("strict_workspace_usable"))
        supported = (
            strict
            and attr_complete
            and weight_complete
            and positive_read_pattern
            and zero_compatible_write_pattern
        )
        reasons: list[str] = []
        if not strict:
            reasons.append("strict G1+G2 workspace context is not usable")
        if not attr_complete:
            reasons.append("raw+MD attribution READ robustness is incomplete")
        if not weight_complete:
            reasons.append("raw+MD MLP/attention weight READ robustness is incomplete")
        if attr_complete and weight_complete and not positive_read_pattern:
            reasons.append("not every READ partial-correlation 95% CI is above zero")
        if attr_complete and weight_complete and not zero_compatible_write_pattern:
            reasons.append("not every WRITE partial-correlation 95% CI includes zero")
        model_assessments.append(
            {
                "model": model,
                "status": "SUPPORTED_PATTERN" if supported else "UNSUPPORTED",
                "strict_workspace_usable": strict,
                "attribution_robustness_available": attr_complete,
                "weight_robustness_available": weight_complete,
                "positive_read_ci_pattern": positive_read_pattern,
                "zero_compatible_write_ci_pattern": zero_compatible_write_pattern,
                "reasons": reasons,
            }
        )
    supported_models = [
        row["model"]
        for row in model_assessments
        if row["status"] == "SUPPORTED_PATTERN"
    ]
    return {
        "status": "SUPPORTED_PATTERN" if supported_models else "UNSUPPORTED",
        "decision_rule": (
            "A model is labelled supported only when strict G1+G2 passes, raw and "
            "MD attribution plus MLP/attention weight READ are all available, every "
            "READ partial-correlation 95% CI lies above zero, and every WRITE "
            "partial-correlation 95% CI includes zero. This CI-sign rule does not "
            "supply the missing preregistered numeric definition of 'large'."
        ),
        "attribution_rows": attribution_rows,
        "weight_rows": weight_rows,
        "model_assessments": model_assessments,
        "supported_models": supported_models,
    }


def _p2_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    twohop = metrics.get("twohop", {}).get(MODEL_7B_KEY, {})
    rows = twohop.get("rows", []) if isinstance(twohop, Mapping) else []
    selected: list[Mapping[str, Any]] = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("measurement_status") == "OK"
        and row.get("direction_method") == PRIMARY_DIRECTION
    ]
    if not selected:
        return {
            "status": "UNESTABLISHED",
            "reason": "No successful raw J-Lens two-hop rows",
            "candidates": [],
        }
    write = np.asarray(
        [
            _finite(row["aggregate"]["write_abs_mean"], path="P2 WRITE")
            for row in selected
        ]
    )
    read = np.asarray(
        [_finite(row["aggregate"]["read_abs_mean"], path="P2 READ") for row in selected]
    )
    causal_abs = np.asarray(
        [
            abs(_finite(row["ablation"]["positive_damage"], path="P2 CAUSAL"))
            for row in selected
        ]
    )
    write_q75 = float(np.quantile(write, 0.75, method="linear"))
    read_q25 = float(np.quantile(read, 0.25, method="linear"))
    causal_q25 = float(np.quantile(causal_abs, 0.25, method="linear"))
    candidates: list[dict[str, Any]] = []
    for row, write_value, read_value, causal_value in zip(
        selected, write, read, causal_abs, strict=True
    ):
        if (
            write_value >= write_q75
            and read_value <= read_q25
            and causal_value <= causal_q25
        ):
            suppression = row.get("output_suppression", {}).get("concept", {})
            candidates.append(
                {
                    "name": str(row["name"]),
                    "write_strength": float(write_value),
                    "read_strength": float(read_value),
                    "causal_abs_damage": float(causal_value),
                    "causal_positive_damage": float(row["ablation"]["positive_damage"]),
                    "output_suppression_abs_damage": abs(
                        float(suppression.get("positive_damage", 0.0))
                    ),
                    "selection": "WRITE>=Q75, READ<=Q25, |CAUSAL|<=Q25",
                    "analysis_role": "posthoc_quantile_screen_candidate",
                }
            )
    candidates.sort(key=lambda row: row["name"])
    full_narration_candidates = [
        row
        for row in candidates
        if abs(row["causal_abs_damage"] - row["output_suppression_abs_damage"]) <= 0.5
    ]

    controls = metrics.get("controls", {}).get(MODEL_7B_KEY, {})
    narration = (
        controls.get("known_narration", {}) if isinstance(controls, Mapping) else {}
    )
    narration_status = str(
        narration.get(
            "status", narration.get("reproduction_gate", {}).get("status", "MISSING")
        )
    )
    absent = (
        controls.get("absent_coordinate_null")
        if isinstance(controls, Mapping)
        else None
    )
    if isinstance(absent, Mapping):
        explicit_status = absent.get("status")
        absent_status = (
            str(explicit_status)
            if explicit_status is not None
            else "DESCRIPTIVE_NO_EQUIVALENCE_THRESHOLD"
        )
        absent_summary = {
            "status": absent_status,
            "rank_feasibility": absent.get("rank_feasibility"),
            "aggregate": absent.get("aggregate"),
        }
    elif isinstance(controls, Mapping) and isinstance(
        controls.get("absent_concept_swap"), Mapping
    ):
        absent_status = "LEGACY_CONCEPT_TO_ABSENT_STRESS_TEST_NOT_A_NULL"
        absent_summary = {"status": absent_status}
    else:
        absent_status = "MISSING"
        absent_summary = {"status": absent_status}
    established = (
        bool(full_narration_candidates)
        and narration_status == "PASS"
        and absent_status == "PASS"
    )
    return {
        "status": "ESTABLISHED" if established else "UNESTABLISHED",
        "analysis_role": "posthoc descriptive; not a preregistered class assignment",
        "population": PRIMARY_DIRECTION,
        "n_population": len(selected),
        "quantile_method": "numpy linear",
        "thresholds": {
            "write_q75": write_q75,
            "read_q25": read_q25,
            "abs_causal_q25": causal_q25,
        },
        "n_candidates": len(candidates),
        "candidates": candidates,
        "full_narration_candidates": [row["name"] for row in full_narration_candidates],
        "n_full_narration_candidates": len(full_narration_candidates),
        "ablation_suppression_gap_screen": 0.5,
        "known_narration_control_status": narration_status,
        "absent_null_control": absent_summary,
        "establishment_rule": (
            "A quantile-screen candidate must also have |CAUSAL|-|suppression| "
            "within 0.5, and the known-narration reproduction plus an explicitly "
            "passing absent-coordinate null are required."
        ),
        "reason": (
            None
            if established
            else "Narration remains unestablished: no candidate and control set met the full operational definition."
        ),
    }


def _rate(values: Sequence[bool]) -> dict[str, Any]:
    return (
        binomial_rate_with_ci(values)
        if values
        else {"status": "NOT_AVAILABLE", "n": 0, "estimate": None}
    )


def _p3_summary(
    metrics: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ambiguity = metrics.get("ambiguity", {}).get(MODEL_7B_KEY, {})
    p3 = ambiguity.get("p3", {}) if isinstance(ambiguity, Mapping) else {}
    overall = p3.get("overall", {}) if isinstance(p3, Mapping) else {}
    statistics = overall.get("statistics", {}) if isinstance(overall, Mapping) else {}
    rows = ambiguity.get("rows", []) if isinstance(ambiguity, Mapping) else []
    variant_flips: list[list[bool]] = [[], []]
    both_flips: list[bool] = []
    meta_suppression_values: list[float] = []
    n_meta_expected = 0
    for row in rows:
        if not isinstance(row, Mapping) or row.get("measurement_status") != "OK":
            continue
        counterbalanced = row.get("counterbalanced", {})
        clean = counterbalanced.get("clean_committed_margin_by_variant", [])
        damage = counterbalanced.get("clean_clamped_swap", {}).get(
            "variant_positive_damage", []
        )
        if len(clean) != 2 or len(damage) != 2:
            continue
        flips = [float(clean[index]) - float(damage[index]) < 0.0 for index in range(2)]
        for index in range(2):
            variant_flips[index].append(flips[index])
        both_flips.append(all(flips))
        meta_records = row.get("meta_counterbalanced", {})
        if isinstance(meta_records, Mapping):
            n_meta_expected += len(meta_records)
            for record in meta_records.values():
                if not isinstance(record, Mapping):
                    continue
                suppression_record = record.get("output_suppression")
                if (
                    isinstance(suppression_record, Mapping)
                    and suppression_record.get("positive_damage") is not None
                ):
                    meta_suppression_values.append(
                        float(suppression_record["positive_damage"])
                    )
    gate_7b = next((row for row in gates if row.get("model") == "7B"), {})
    suppression = _compact_statistic(
        statistics.get("output_suppression_positive_damage")
    )
    persisted_both_flips = _compact_statistic(
        statistics.get("counterbalance_robust_swap_flip_rate")
    )
    both_variant_rate = (
        persisted_both_flips
        if _stat_estimated(persisted_both_flips)
        else _rate(both_flips)
    )
    meta = (
        ambiguity.get("meta_token_diagnostics", {})
        if isinstance(ambiguity, Mapping)
        else {}
    )
    across_meta = (
        meta.get("across_candidate_means", {}) if isinstance(meta, Mapping) else {}
    )
    pooled_meta = (
        meta.get("pooled_item_candidate_diagnostic", {})
        if isinstance(meta, Mapping)
        else {}
    )
    return {
        "status": str(p3.get("verdict", "NOT_AVAILABLE")).upper(),
        "n": int(overall.get("n", p3.get("n_frozen_items", 0)) or 0),
        "mean_margin_flip_rate": _compact_statistic(statistics.get("swap_flip_rate")),
        "variant_1_flip_rate": _rate(variant_flips[0]),
        "variant_2_flip_rate": _rate(variant_flips[1]),
        "both_variants_flip_rate": both_variant_rate,
        "committed_concept_write": _compact_statistic(
            statistics.get("committed_concept_write_abs_mean")
        ),
        "alternate_concept_write": _compact_statistic(
            statistics.get("alternate_concept_write_abs_mean")
        ),
        "committed_concept_read": _compact_statistic(
            statistics.get("committed_concept_read_abs_mean")
        ),
        "alternate_concept_read": _compact_statistic(
            statistics.get("alternate_concept_read_abs_mean")
        ),
        "internal_ablation_positive_damage": _compact_statistic(
            statistics.get("internal_ablation_positive_damage")
        ),
        "internal_minus_suppression_damage": _compact_statistic(
            statistics.get("internal_minus_suppression_damage")
        ),
        "output_suppression_positive_damage": suppression,
        "ablation_exceeds_suppression_rate": _compact_statistic(
            statistics.get("ablation_exceeds_suppression_rate")
        ),
        "counterbalance_agreement_rate": _compact_statistic(
            statistics.get("counterbalance_agreement_rate")
        ),
        "meta_token_diagnostics": {
            "analysis_role": meta.get("analysis_role")
            if isinstance(meta, Mapping)
            else None,
            "n_candidates": across_meta.get("n_candidates")
            if isinstance(across_meta, Mapping)
            else None,
            "candidate_mean_read_vs_damage": _compact_statistic(
                across_meta.get("read_vs_ablation_damage")
                if isinstance(across_meta, Mapping)
                else None
            ),
            "candidate_mean_partial_read_given_write": _compact_statistic(
                across_meta.get("partial_causal_read_given_write")
                if isinstance(across_meta, Mapping)
                else None
            ),
            "pooled_read_vs_damage": _compact_statistic(
                pooled_meta.get("read_vs_ablation_damage")
                if isinstance(pooled_meta, Mapping)
                else None
            ),
            "pooled_partial_read_given_write": _compact_statistic(
                pooled_meta.get("partial_causal_read_given_write")
                if isinstance(pooled_meta, Mapping)
                else None
            ),
            "pooled_warning": pooled_meta.get("warning")
            if isinstance(pooled_meta, Mapping)
            else None,
            "interpretation_warning": meta.get("interpretation_warning")
            if isinstance(meta, Mapping)
            else None,
            "output_suppression_coverage": {
                "n_expected": n_meta_expected,
                "n_present": len(meta_suppression_values),
                "all_exact_zero": bool(
                    meta_suppression_values
                    and all(value == 0.0 for value in meta_suppression_values)
                ),
            },
        },
        "direction_method": p3.get("direction_method"),
        "g2_context": {
            "strict_status": gate_7b.get("g2_strict_status"),
            "directional_status": gate_7b.get("g2_directional_status"),
            "strict_workspace_usable": gate_7b.get("strict_workspace_usable"),
        },
        "structural_zero_suppression_caveat": (
            "The ambiguity behavior metric is a reading-answer logit difference, "
            "while output suppression clamps a separate concept-token logit. Its "
            "exact zero is therefore structural under this metric, so ablation > "
            "suppression is not an independent output-steering test."
        ),
        "limitation": p3.get("limitation"),
    }


def _controls_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    controls = metrics.get("controls", {}).get(MODEL_7B_KEY, {})
    if not isinstance(controls, Mapping):
        return {"status": "MISSING"}
    random_null = controls.get("random_direction_null", {})
    random_aggregate = (
        random_null.get("aggregate") if isinstance(random_null, Mapping) else None
    )
    if not isinstance(random_aggregate, Mapping):
        random_rows = (
            random_null.get("rows", []) if isinstance(random_null, Mapping) else []
        )
        null_abs = [
            abs(float(row["null_summary"]["mean"]))
            for row in random_rows
            if isinstance(row, Mapping) and isinstance(row.get("null_summary"), Mapping)
        ]
        random_aggregate = {
            "status": "LEGACY_DESCRIPTIVE",
            "n_items": len(random_rows),
            "mean_abs_item_null_mean": float(np.mean(null_abs)) if null_abs else None,
        }
    else:
        random_aggregate = {
            "status": "COMPUTED",
            **{
                key: random_aggregate[key]
                for key in ("bootstrap_unit", "n_items", "n_draws_total")
                if key in random_aggregate
            },
            **{
                key: _compact_statistic(random_aggregate.get(key))
                for key in (
                    "mean_random_delta",
                    "mean_abs_random_delta",
                    "mean_observed_delta",
                    "mean_abs_observed_delta",
                    "paired_mean_abs_observed_minus_random",
                )
            },
        }
    absent = controls.get("absent_coordinate_null")
    if isinstance(absent, Mapping):
        absent_aggregate = absent.get("aggregate")
        if isinstance(absent_aggregate, Mapping):
            absent_aggregate = {
                **{
                    key: absent_aggregate[key]
                    for key in (
                        "n_items",
                        "median_delta",
                        "median_abs_delta",
                        "max_abs_delta",
                    )
                    if key in absent_aggregate
                },
                "mean_delta": _compact_statistic(absent_aggregate.get("mean_delta")),
                "mean_abs_delta": _compact_statistic(
                    absent_aggregate.get("mean_abs_delta")
                ),
            }
        absent_summary = {
            "status": absent.get("status", "DESCRIPTIVE_NO_EQUIVALENCE_THRESHOLD"),
            "rank_feasibility": absent.get("rank_feasibility"),
            "aggregate": absent_aggregate,
        }
    elif isinstance(controls.get("absent_concept_swap"), Mapping):
        absent_summary = {
            "status": "LEGACY_CONCEPT_TO_ABSENT_STRESS_TEST_NOT_A_NULL",
            "n": len(controls["absent_concept_swap"].get("rows", [])),
        }
    else:
        absent_summary = {"status": "MISSING"}
    capability = controls.get("capability", {})
    general = (
        capability.get("general_language", {})
        if isinstance(capability, Mapping)
        else {}
    )
    task = capability.get("twohop", {}) if isinstance(capability, Mapping) else {}
    narration = controls.get("known_narration", {})
    logit_lens = controls.get("logit_lens_identity_jacobian", {})
    predictor_comparison = controls.get("causal_predictor_comparison", {})
    shared_predictors = (
        predictor_comparison.get("shared_core_causal_target", {})
        if isinstance(predictor_comparison, Mapping)
        else {}
    )
    suppression = controls.get("core_output_suppression_assertion", {})
    return {
        "status": "COMPUTED",
        "random_direction_null": random_aggregate,
        "absent_coordinate_null": absent_summary,
        "capability": {
            "n_fixed_texts": capability.get("n_fixed_texts")
            if isinstance(capability, Mapping)
            else None,
            "n_intervention_banks": capability.get("n_intervention_banks")
            if isinstance(capability, Mapping)
            else None,
            "general_language_n_rows": general.get(
                "n_rows", len(general.get("rows", []))
            ),
            "mean_clean_nll": general.get("mean_clean_nll"),
            "mean_edited_nll": general.get("mean_edited_nll"),
            "mean_delta_nll": general.get("mean_delta_nll"),
            "mean_delta_nll_ci": _compact_statistic(general.get("mean_delta_nll_ci")),
            "twohop_n_rows": task.get("n_rows", len(task.get("rows", []))),
            "twohop_clean_accuracy": task.get("clean_accuracy"),
            "twohop_edited_accuracy": task.get("edited_accuracy"),
            "twohop_clean_accuracy_ci": _compact_statistic(
                task.get("clean_accuracy_ci")
            ),
            "twohop_edited_accuracy_ci": _compact_statistic(
                task.get("edited_accuracy_ci")
            ),
            "twohop_accuracy_delta_ci": _compact_statistic(
                task.get("edited_minus_clean_accuracy_ci")
            ),
        },
        "known_narration": {
            "status": narration.get("status")
            if isinstance(narration, Mapping)
            else "MISSING",
            "reproduction_gate": narration.get("reproduction_gate")
            if isinstance(narration, Mapping)
            else None,
            "write": _compact_statistic(
                narration.get("aggregate_write_read_magnitudes", {}).get(
                    "write_abs_mean_across_passages"
                )
                if isinstance(narration, Mapping)
                else None
            ),
            "read": _compact_statistic(
                narration.get("aggregate_write_read_magnitudes", {}).get(
                    "read_abs_mean_across_passages"
                )
                if isinstance(narration, Mapping)
                else None
            ),
        },
        "logit_lens": {
            "predictor": _compact_statistic(
                logit_lens.get("predictor") if isinstance(logit_lens, Mapping) else None
            ),
            "shared_outcome_core_predictor": _compact_statistic(
                shared_predictors.get("core_first_order_predictor")
                if isinstance(shared_predictors, Mapping)
                else None
            ),
            "shared_outcome_identity_j_predictor": _compact_statistic(
                shared_predictors.get("identity_j_first_order_association")
                if isinstance(shared_predictors, Mapping)
                else None
            ),
        },
        "output_suppression": {
            "status": suppression.get("status")
            if isinstance(suppression, Mapping)
            else None,
            "classification": suppression.get("classification")
            if isinstance(suppression, Mapping)
            else None,
            "n_rows": suppression.get("n_rows")
            if isinstance(suppression, Mapping)
            else None,
            "comparison": suppression.get("comparison")
            if isinstance(suppression, Mapping)
            else None,
        },
        "limitations": list(controls.get("limitations", [])),
    }


def _scale_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    scale = metrics.get("scale_comparison")
    if not isinstance(scale, Mapping):
        return {
            "status": "NOT_AVAILABLE",
            "qwen32b": {"status": "NOT_AVAILABLE_OPTIONAL"},
        }
    comparison = scale.get("comparison", {})
    models = comparison.get("models", {}) if isinstance(comparison, Mapping) else {}
    model_rows: list[dict[str, Any]] = []
    if isinstance(models, Mapping):
        for tag in sorted(models):
            model = models[tag]
            if not isinstance(model, Mapping):
                continue
            methods: list[dict[str, Any]] = []
            for method, result in sorted(model.get("methods", {}).items()):
                if not isinstance(result, Mapping):
                    continue
                partials = result.get("partial_correlations", {})
                methods.append(
                    {
                        "method": method,
                        "status": result.get("status"),
                        "n": result.get("n"),
                        "causal_read_given_write": _compact_statistic(
                            partials.get("causal_read_given_write")
                        ),
                        "causal_write_given_read": _compact_statistic(
                            partials.get("causal_write_given_read")
                        ),
                        "mean_ablation_positive_damage": _compact_statistic(
                            result.get("mean_ablation_positive_damage")
                        ),
                        "attribution_predicted_vs_real": _compact_statistic(
                            result.get("attribution_predicted_vs_real")
                        ),
                    }
                )
            model_rows.append(
                {
                    "model": str(tag),
                    "strict_workspace_usable": model.get("gates", {}).get(
                        "strict_workspace_usable"
                    ),
                    "twohop_status": model.get("twohop_status"),
                    "sample_counts": model.get("sample_counts"),
                    "methods": methods,
                }
            )
    paired_rows: list[dict[str, Any]] = []
    paired = (
        comparison.get("paired_14b_minus_7b", {})
        if isinstance(comparison, Mapping)
        else {}
    )
    if isinstance(paired, Mapping):
        for method in (PRIMARY_DIRECTION, MD_DIRECTION):
            result = paired.get(method)
            if not isinstance(result, Mapping):
                continue
            deltas = result.get("delta_14b_minus_7b", {})
            paired_rows.append(
                {
                    "method": method,
                    "n": result.get("n_common"),
                    "pairing_rule": result.get("pairing_rule"),
                    "read_partial_delta": _compact_statistic(
                        deltas.get("partial_causal_read_given_write")
                        if isinstance(deltas, Mapping)
                        else None
                    ),
                    "write_partial_delta": _compact_statistic(
                        deltas.get("partial_causal_write_given_read")
                        if isinstance(deltas, Mapping)
                        else None
                    ),
                    "ablation_damage_delta": _compact_statistic(
                        deltas.get("mean_ablation_positive_damage")
                        if isinstance(deltas, Mapping)
                        else None
                    ),
                    "attribution_r_delta": _compact_statistic(
                        deltas.get("attribution_predicted_vs_real_r")
                        if isinstance(deltas, Mapping)
                        else None
                    ),
                }
            )
    qwen32b = scale.get("qwen32b")
    return {
        "status": scale.get("status", "COMPUTED"),
        "models": model_rows,
        "paired_14b_minus_7b": paired_rows,
        "qwen14b_lens_provenance": scale.get("qwen14b_lens_provenance"),
        "qwen32b": (
            dict(qwen32b)
            if isinstance(qwen32b, Mapping)
            else {"status": "NOT_AVAILABLE_OPTIONAL"}
        ),
    }


def _p4_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("blackmail", "p4"):
        payload = metrics.get(key)
        if isinstance(payload, Mapping):
            return {
                "status": payload.get("status", "COMPUTED_OPTIONAL"),
                "source_key": key,
                "summary": payload.get("summary", payload.get("verdict")),
            }
    return {
        "status": "NOT_RUN_OPTIONAL",
        "reason": "The optional blackmail phase was not run; this does not fail report completeness.",
    }


def _method_validation_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    concept = metrics.get("concept_vectors", {})
    scale = metrics.get("scale_comparison", {})
    localization = metrics.get("localization", {}).get(MODEL_7B_KEY, {})
    agreement = (
        localization.get("attribution_weight_rank_agreement", {})
        if isinstance(localization, Mapping)
        else {}
    )
    criteria_7b = concept.get("criteria", {}) if isinstance(concept, Mapping) else {}
    md_14b = (
        scale.get("qwen14b_md_validation", {}) if isinstance(scale, Mapping) else {}
    )
    criteria_14b = md_14b.get("criteria", {}) if isinstance(md_14b, Mapping) else {}
    return {
        "md_7b": {
            "status": concept.get("status") if isinstance(concept, Mapping) else None,
            "cosine_raw": _compact_statistic(
                concept.get("cosine_alignment", {}).get("raw_WU_J")
                if isinstance(concept, Mapping)
                else None
            ),
            "retrieval_top1": _compact_statistic(
                concept.get("heldout_retrieval", {}).get("top1_at_fixed_layer")
                if isinstance(concept, Mapping)
                else None
            ),
            "explicit_top5": _compact_statistic(
                concept.get("explicit_known_answer", {}).get("top5")
                if isinstance(concept, Mapping)
                else None
            ),
            "failed_criteria": [
                key for key, passed in criteria_7b.items() if passed is False
            ]
            if isinstance(criteria_7b, Mapping)
            else [],
        },
        "md_14b": {
            "status": md_14b.get("status") if isinstance(md_14b, Mapping) else None,
            "n_concepts": md_14b.get("n_concepts")
            if isinstance(md_14b, Mapping)
            else None,
            "failed_criteria": [
                key for key, passed in criteria_14b.items() if passed is False
            ]
            if isinstance(criteria_14b, Mapping)
            else [],
        },
        "attribution_weight_agreement": {
            "attention_label_weighted_ov_spearman": agreement.get(
                "head_attribution_vs_label_weighted_ov", {}
            ).get("spearman_rho")
            if isinstance(agreement, Mapping)
            else None,
            "attention_normalized_ov_spearman": agreement.get(
                "head_attribution_vs_normalized_ov", {}
            ).get("spearman_rho")
            if isinstance(agreement, Mapping)
            else None,
            "mlp_normalized_gain_spearman": agreement.get(
                "mlp_attribution_vs_normalized_gain", {}
            ).get("spearman_rho")
            if isinstance(agreement, Mapping)
            else None,
            "scope": agreement.get("scope") if isinstance(agreement, Mapping) else None,
            "guardrail": agreement.get("guardrail")
            if isinstance(agreement, Mapping)
            else None,
        },
    }


def _limitations(
    metrics: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    p1: Mapping[str, Any],
    p2: Mapping[str, Any],
    p3: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> list[str]:
    limitations = [
        (
            "Flash/SDPA attention kernels were not proven bitwise deterministic; "
            "seeded reruns can retain low-level nondeterminism despite fixed seeds."
        ),
        (
            "The 14B lens resumed from a legacy first-10-prompt checkpoint that "
            "predated the prompt-hash provenance sidecar; those first ten "
            "contributions cannot be cryptographically bound to the declared prompt list."
        ),
        (
            "The preregistration supplied no numerical cutoff for 'large' READ, "
            "'approximately zero' WRITE, or scale sharpening; estimates and 95% "
            "CIs are primary, and the report's CI-sign rule is conservative bookkeeping."
        ),
        (
            "Weight READ is activation-independent only after direction choice and "
            "is selection-conditioned because components were first flagged by "
            "activation localization."
        ),
        (
            "Population weight READ was run for the 7B two-hop analysis, not for "
            "the 14B scale run or the ambiguity committed/alternate/meta-token "
            "directions; those analyses therefore do not satisfy a two-READ "
            "estimator claim."
        ),
        (
            "F4 contrasts measured driver and low-READ candidates, not a validated "
            "driver-versus-narration class, because the known-narration positive "
            "control did not reproduce."
        ),
        (
            "Concepts are restricted to exact single-token proxies; multi-token "
            "concepts are outside the fitted vocabulary-direction analysis."
        ),
        (
            "Random-direction, absent-coordinate, general capability, known-"
            "narration, and identity-J controls were run for 7B two-hop only, "
            "not repeated at 14B."
        ),
        (
            "The known-answer directional G2 sensitivity pass used RMS-gain-folded "
            "directions; the raw direction used downstream did not change top-1 "
            "to the swapped answer at either scale."
        ),
        str(p3.get("structural_zero_suppression_caveat")),
    ]
    for gate in gates:
        if not gate.get("strict_workspace_usable"):
            limitations.append(
                f"Qwen-{gate['model']} failed the strict usable-workspace context; "
                "its downstream results are diagnostic."
            )
    concept_status = metrics.get("concept_vectors", {}).get("status")
    if concept_status != "PASS":
        limitations.append(
            f"Independent MD direction validation status was {concept_status}; "
            "MD robustness is correspondingly limited."
        )
    if p1.get("status") != "SUPPORTED_PATTERN":
        limitations.append(
            "P1 did not survive every required direction/READ robustness check."
        )
    if p2.get("status") != "ESTABLISHED":
        limitations.append(
            "P2 candidates are posthoc and do not establish a narration class."
        )
    if controls.get("absent_coordinate_null", {}).get("status") != "PASS":
        limitations.append(
            "The absent-coordinate null has no preregistered equivalence margin or did not pass; near-zero specificity is not established."
        )
    for value in controls.get("limitations", []):
        text = str(value)
        if text == (
            "Identity-J and core predictors are validated against different "
            "ablation directions."
        ):
            text = (
                "Within-direction identity-J and J-Lens validity checks use their "
                "own intervention directions; the headline baseline table instead "
                "joins both predictors to the same core-ablation outcome."
            )
        limitations.append(text)
    deduplicated: list[str] = []
    for limitation in limitations:
        if limitation and limitation not in deduplicated:
            deduplicated.append(limitation)
    return deduplicated


def build_report_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compact JSON-ready report result without checking files."""

    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    gates = _gate_summary(metrics)
    p1 = _p1_summary(metrics, gates)
    p2 = _p2_summary(metrics)
    p3 = _p3_summary(metrics, gates)
    p4 = _p4_summary(metrics)
    controls = _controls_summary(metrics)
    scale = _scale_summary(metrics)
    method_validation = _method_validation_summary(metrics)
    limitations = _limitations(metrics, gates, p1, p2, p3, controls)
    p3_refuted = str(p3.get("status", "")).casefold() == "refuted"
    if p3_refuted:
        overall = "NOT SUPPORTED"
        overall_reason = (
            "P3 was diagnostically refuted, P1 is unsupported (including a "
            "significantly negative main-scale raw READ partial), and P2 "
            "narration remains unestablished."
        )
    elif (
        p1.get("status") == "SUPPORTED_PATTERN"
        and p2.get("status") == "ESTABLISHED"
        and str(p3.get("status", "")).casefold() in {"supported", "pass"}
    ):
        overall = "SUPPORTED"
        overall_reason = "P1–P3 all met their report decision rules."
    else:
        overall = "MIXED / INCONCLUSIVE"
        overall_reason = (
            "At least one required prediction is unavailable or unsupported, but no "
            "available required prediction is explicitly refuted."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "preregistered_hypothesis": PREREGISTERED_HYPOTHESIS,
        "overall_verdict": overall,
        "overall_reason": overall_reason,
        "verdicts": {
            "P1": p1["status"],
            "P2": p2["status"],
            "P3": p3["status"],
            "P4": p4["status"],
        },
        "gates": gates,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "controls": controls,
        "scale": scale,
        "method_validation": method_validation,
        "limitations": limitations,
    }


def _fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _fmt_scientific(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}e}"


def _fmt_statistic(statistic: Mapping[str, Any]) -> str:
    if statistic.get("status") != "ESTIMATED":
        return str(statistic.get("status", "NOT_AVAILABLE"))
    estimate = _fmt_number(statistic.get("estimate"))
    lower = _fmt_number(statistic.get("ci_low"))
    upper = _fmt_number(statistic.get("ci_high"))
    n = statistic.get("n", "NA")
    return f"{estimate} (95% CI {lower}, {upper}; N={n})"


def _fmt_rate(statistic: Mapping[str, Any]) -> str:
    if statistic.get("status") == "ESTIMATED":
        return _fmt_statistic(statistic)
    if statistic.get("estimate") is not None:
        return (
            f"{_fmt_number(statistic.get('estimate'))} "
            f"(N={statistic.get('n', 'NA')}; CI not precomputed)"
        )
    return str(statistic.get("status", "NOT_AVAILABLE"))


def _fmt_coefficient(
    coefficients: Mapping[str, Any],
    intervals: Mapping[str, Any],
    key: str,
) -> str:
    interval = intervals.get(key)
    estimate = _fmt_number(coefficients.get(key))
    if not isinstance(interval, Mapping):
        return estimate
    return (
        f"{estimate} ({_fmt_number(interval.get('ci_low'))}, "
        f"{_fmt_number(interval.get('ci_high'))})"
    )


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(map(_escape_table, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_table(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def render_results_markdown(summary: Mapping[str, Any]) -> str:
    """Render a deterministic RESULTS.md body from a compact summary."""

    lines = [
        "# WRITE / READ / CAUSAL results",
        "",
        "## Overall verdict",
        "",
        f"**{summary['overall_verdict']}** — {summary['overall_reason']}",
        "",
        "## Preregistered hypothesis",
        "",
        str(summary["preregistered_hypothesis"]),
        "",
        "## Correctness gates and usable-workspace context",
        "",
    ]
    gate_rows = [
        [
            row["model"],
            row.get("g1_status"),
            row.get("g1_n"),
            _fmt_scientific(row.get("g1_max_mean_kl"), 3),
            row.get("g2_strict_status"),
            row.get("g2_directional_status"),
            row.get("g3_status"),
            "YES" if row.get("g3_attribution_reliable") else "NO",
            _fmt_statistic(row.get("g3_correlation", {})),
            row.get("context"),
        ]
        for row in summary["gates"]
    ]
    lines.extend(
        [
            _markdown_table(
                [
                    "model",
                    "G1",
                    "G1 N",
                    "max mean KL",
                    "G2 strict",
                    "G2 directional",
                    "G3 computed",
                    "attribution reliable",
                    "attribution vs real",
                    "context",
                ],
                gate_rows,
            ),
            "",
            (
                "G3 is a held-out validation gate (N=20 at 7B); the scale table "
                "separately reports full-core attribution correlations. The "
                "directional G2 PASS is the RMS-gain-folded sensitivity variant. "
                "The raw direction used downstream did not move either scale's "
                "known-case answer top-1 to 6, so strict G2 failed."
            ),
            "",
            "## P1 — conditional READ versus WRITE",
            "",
        ]
    )
    p1 = summary["p1"]
    attr_rows = []
    for row in p1["attribution_rows"]:
        additive = row.get("regression_additive", {})
        interaction = row.get("regression_interaction", {})
        coefficients = additive.get("coefficients", {})
        intervals = additive.get("coefficient_intervals", {})
        interaction_coefficients = interaction.get("coefficients", {})
        interaction_intervals = interaction.get("coefficient_intervals", {})
        attr_rows.append(
            [
                row["model"],
                row["method"],
                row.get("n"),
                _fmt_statistic(row.get("write_read_correlation", {})),
                _fmt_statistic(row.get("causal_read_given_write", {})),
                _fmt_statistic(row.get("causal_write_given_read", {})),
                _fmt_coefficient(coefficients, intervals, "read"),
                _fmt_coefficient(coefficients, intervals, "write"),
                _fmt_coefficient(
                    interaction_coefficients,
                    interaction_intervals,
                    "write_x_read",
                ),
                _fmt_number(additive.get("r_squared")),
            ]
        )
    lines.extend(
        [
            (
                "For these regressions, CAUSAL is positive ablation damage "
                "`M_clean - M_edited`; the repository-wide intervention delta "
                "stored per item is the opposite sign, `M_edited - M_clean`."
            ),
            "",
            "Attribution READ:",
            "",
            _markdown_table(
                [
                    "model",
                    "direction",
                    "N",
                    "corr(WRITE, READ)",
                    "partial CAUSAL–READ | WRITE",
                    "partial CAUSAL–WRITE | READ",
                    "β READ (95% CI)",
                    "β WRITE (95% CI)",
                    "β WRITE×READ (95% CI)",
                    "R²",
                ],
                attr_rows,
            ),
            "",
            "Weight READ (activation-independent after direction choice, but localization-selection-conditioned):",
            "",
        ]
    )
    weight_rows = []
    for row in p1["weight_rows"]:
        additive = row.get("regression_additive", {})
        additive_coefficients = additive.get("coefficients", {})
        additive_intervals = additive.get("coefficient_intervals", {})
        interaction = row.get("regression_interaction", {})
        interaction_coefficients = interaction.get("coefficients", {})
        interaction_intervals = interaction.get("coefficient_intervals", {})
        weight_rows.append(
            [
                row["model"],
                row["method"],
                row["family"],
                row.get("n"),
                _fmt_statistic(row.get("write_read_correlation", {})),
                _fmt_statistic(row.get("causal_read_given_write", {})),
                _fmt_statistic(row.get("causal_write_given_read", {})),
                _fmt_coefficient(additive_coefficients, additive_intervals, "read"),
                _fmt_coefficient(additive_coefficients, additive_intervals, "write"),
                _fmt_coefficient(
                    interaction_coefficients,
                    interaction_intervals,
                    "write_x_read",
                ),
                _fmt_number(additive.get("r_squared")),
            ]
        )
    lines.extend(
        [
            _markdown_table(
                [
                    "model",
                    "direction",
                    "weight family",
                    "N",
                    "corr(WRITE, weight READ)",
                    "CAUSAL–READ | WRITE",
                    "CAUSAL–WRITE | READ",
                    "β weight READ (95% CI)",
                    "β WRITE (95% CI)",
                    "β WRITE×READ (95% CI)",
                    "R²",
                ],
                weight_rows,
            ),
            "",
            f"**P1 verdict: {p1['status']}**. {p1['decision_rule']}",
            (
                "At the main 14B scale, the raw-direction READ partial is "
                "significantly negative, so this is contrary evidence rather "
                "than merely missing robustness."
            ),
            "",
            "### Direction and READ-estimator validation",
            "",
            _markdown_table(
                ["check", "result"],
                [
                    [
                        "7B MD validation",
                        str(summary["method_validation"]["md_7b"].get("status"))
                        + "; raw/MD cosine="
                        + _fmt_statistic(
                            summary["method_validation"]["md_7b"]["cosine_raw"]
                        )
                        + "; held-out top-1="
                        + _fmt_statistic(
                            summary["method_validation"]["md_7b"]["retrieval_top1"]
                        )
                        + "; explicit top-5="
                        + _fmt_statistic(
                            summary["method_validation"]["md_7b"]["explicit_top5"]
                        ),
                    ],
                    [
                        "14B MD validation",
                        str(summary["method_validation"]["md_14b"].get("status"))
                        + "; failed criteria="
                        + ", ".join(
                            summary["method_validation"]["md_14b"].get(
                                "failed_criteria", []
                            )
                        ),
                    ],
                    [
                        "selected-head attribution/weight rank agreement",
                        "label-weighted OV ρ="
                        + _fmt_number(
                            summary["method_validation"][
                                "attribution_weight_agreement"
                            ].get("attention_label_weighted_ov_spearman")
                        )
                        + "; OV norm ρ="
                        + _fmt_number(
                            summary["method_validation"][
                                "attribution_weight_agreement"
                            ].get("attention_normalized_ov_spearman")
                        ),
                    ],
                    [
                        "selected-MLP attribution/weight rank agreement",
                        "normalized gain ρ="
                        + _fmt_number(
                            summary["method_validation"][
                                "attribution_weight_agreement"
                            ].get("mlp_normalized_gain_spearman")
                        ),
                    ],
                ],
            ),
            "",
            (
                "Agreement is mixed and selection-conditioned: attention ranks "
                "agree, while selected-MLP ranks are negatively associated."
            ),
            "",
            "## P2 — posthoc high-WRITE/low-READ quantile screen",
            "",
        ]
    )
    p2 = summary["p2"]
    thresholds = p2.get("thresholds", {})
    lines.append(
        "Descriptive selection only: WRITE ≥ Q75 "
        f"({_fmt_number(thresholds.get('write_q75'))}), READ ≤ Q25 "
        f"({_fmt_number(thresholds.get('read_q25'))}), and |CAUSAL| ≤ Q25 "
        f"({_fmt_number(thresholds.get('abs_causal_q25'))})."
    )
    lines.append(
        "These rows pass only the WRITE/READ/CAUSAL quantile screen. A full "
        "narration case must additionally have ablation approximately equal to "
        "suppression; "
        f"{p2.get('n_full_narration_candidates', 0)} screened candidates meet "
        "that additional numerical screen."
    )
    lines.extend(
        [
            "",
            _markdown_table(
                ["item", "WRITE", "READ", "|CAUSAL|", "|suppression|", "role"],
                [
                    [
                        row["name"],
                        _fmt_number(row["write_strength"]),
                        _fmt_number(row["read_strength"]),
                        _fmt_number(row["causal_abs_damage"]),
                        _fmt_number(row["output_suppression_abs_damage"]),
                        row["analysis_role"],
                    ]
                    for row in p2.get("candidates", [])
                ],
            ),
            "",
        ]
    )
    lines.extend(
        [
            f"**P2 verdict: {p2['status']}**. {p2.get('reason') or p2['establishment_rule']}",
            f"Known-narration control: {p2.get('known_narration_control_status')}; "
            f"absent-null control: {p2.get('absent_null_control', {}).get('status')}.",
            "",
            "## P3 — ambiguity commitment",
            "",
        ]
    )
    p3 = summary["p3"]
    lines.extend(
        [
            _markdown_table(
                ["quantity", "estimate"],
                [
                    [
                        "committed concept WRITE",
                        _fmt_statistic(p3["committed_concept_write"]),
                    ],
                    [
                        "alternate concept WRITE",
                        _fmt_statistic(p3["alternate_concept_write"]),
                    ],
                    [
                        "committed concept attribution READ",
                        _fmt_statistic(p3["committed_concept_read"]),
                    ],
                    [
                        "alternate concept attribution READ",
                        _fmt_statistic(p3["alternate_concept_read"]),
                    ],
                    [
                        "mean-margin swap flip rate",
                        _fmt_statistic(p3["mean_margin_flip_rate"]),
                    ],
                    ["variant 1 flip rate", _fmt_rate(p3["variant_1_flip_rate"])],
                    ["variant 2 flip rate", _fmt_rate(p3["variant_2_flip_rate"])],
                    ["both variants flip", _fmt_rate(p3["both_variants_flip_rate"])],
                    [
                        "internal ablation damage",
                        _fmt_statistic(p3["internal_ablation_positive_damage"]),
                    ],
                    [
                        "internal minus suppression",
                        _fmt_statistic(p3["internal_minus_suppression_damage"]),
                    ],
                    [
                        "output suppression damage",
                        _fmt_statistic(p3["output_suppression_positive_damage"]),
                    ],
                ],
            ),
            "",
            f"**P3 diagnostic verdict: {p3['status']}**. G2 context: strict={p3['g2_context'].get('strict_status')}, "
            f"directional={p3['g2_context'].get('directional_status')}.",
            (
                "Committed, alternate, and meta-token directions use attribution "
                "READ only; weight READ and independent mean-difference direction "
                "robustness were not run for this diagnostic phase."
            ),
            "",
            f"Caveat: {p3['structural_zero_suppression_caveat']}",
            "",
            "Interpretive meta-token diagnostics (nonconfirmatory):",
            "",
            _markdown_table(
                ["association", "estimate"],
                [
                    [
                        "candidate-mean READ vs damage",
                        _fmt_statistic(
                            p3["meta_token_diagnostics"][
                                "candidate_mean_read_vs_damage"
                            ]
                        ),
                    ],
                    [
                        "candidate-mean partial READ | WRITE",
                        _fmt_statistic(
                            p3["meta_token_diagnostics"][
                                "candidate_mean_partial_read_given_write"
                            ]
                        ),
                    ],
                    [
                        "pooled item×candidate READ vs damage",
                        _fmt_statistic(
                            p3["meta_token_diagnostics"]["pooled_read_vs_damage"]
                        ),
                    ],
                    [
                        "pooled item×candidate partial READ | WRITE",
                        _fmt_statistic(
                            p3["meta_token_diagnostics"][
                                "pooled_partial_read_given_write"
                            ]
                        ),
                    ],
                ],
            ),
            "",
            str(p3["meta_token_diagnostics"].get("pooled_warning") or ""),
            (
                "Meta-token output suppression coverage: "
                f"{p3['meta_token_diagnostics']['output_suppression_coverage'].get('n_present')}/"
                f"{p3['meta_token_diagnostics']['output_suppression_coverage'].get('n_expected')}; "
                "all exact structural zeros="
                f"{p3['meta_token_diagnostics']['output_suppression_coverage'].get('all_exact_zero')}."
            ),
            "",
            "## P4 — optional blackmail task",
            "",
            f"**{summary['p4']['status']}** — {summary['p4'].get('reason', summary['p4'].get('summary', ''))}",
            "",
            "## Mandatory controls",
            "",
        ]
    )
    controls = summary["controls"]
    capability = controls.get("capability", {})
    narration = controls.get("known_narration", {})
    random_null = controls.get("random_direction_null", {})
    absent_null = controls.get("absent_coordinate_null", {})
    random_result = (
        "mean |Δ|=" + _fmt_statistic(random_null.get("mean_abs_random_delta", {}))
        if isinstance(random_null.get("mean_abs_random_delta"), Mapping)
        else str(random_null.get("status", "COMPUTED"))
    )
    paired_random = random_null.get("paired_mean_abs_observed_minus_random")
    if isinstance(paired_random, Mapping):
        random_result += "; observed−random |Δ|=" + _fmt_statistic(paired_random)
    absent_aggregate = absent_null.get("aggregate", {})
    absent_result = str(absent_null.get("status"))
    if isinstance(absent_aggregate, Mapping) and isinstance(
        absent_aggregate.get("mean_abs_delta"), Mapping
    ):
        absent_result += "; mean |Δ|=" + _fmt_statistic(
            absent_aggregate["mean_abs_delta"]
        )
    narration_gate = narration.get("reproduction_gate")
    narration_result = str(narration.get("status"))
    if isinstance(narration_gate, Mapping):
        narration_result += (
            f"; reproduced {narration_gate.get('n_reproduced')}/"
            f"{narration_gate.get('n_passages')}"
            f"; high-WRITE {narration_gate.get('n_high_write')}/"
            f"{narration_gate.get('n_passages')}"
            f"; low-causal {narration_gate.get('n_low_causal')}/"
            f"{narration_gate.get('n_passages')}"
            f"; clean-capable {narration_gate.get('n_clean_capable')}/"
            f"{narration_gate.get('n_passages')}"
        )
    narration_result += (
        "; mean WRITE="
        + _fmt_statistic(narration.get("write", {}))
        + "; mean READ="
        + _fmt_statistic(narration.get("read", {}))
    )
    capability_nll = capability.get("mean_delta_nll_ci", {})
    capability_nll_result = (
        _fmt_statistic(capability_nll)
        if _stat_estimated(capability_nll)
        else _fmt_number(capability.get("mean_delta_nll"))
    )
    clean_accuracy = capability.get("twohop_clean_accuracy_ci", {})
    edited_accuracy = capability.get("twohop_edited_accuracy_ci", {})
    accuracy_result = (
        f"{_fmt_statistic(clean_accuracy)} → {_fmt_statistic(edited_accuracy)}"
        if _stat_estimated(clean_accuracy) and _stat_estimated(edited_accuracy)
        else f"{_fmt_number(capability.get('twohop_clean_accuracy'))} → "
        f"{_fmt_number(capability.get('twohop_edited_accuracy'))}"
    )
    logit_lens = controls.get("logit_lens", {})
    identity_result = (
        "shared outcome: identity-J="
        + _fmt_statistic(logit_lens.get("shared_outcome_identity_j_predictor", {}))
        + "; J-Lens="
        + _fmt_statistic(logit_lens.get("shared_outcome_core_predictor", {}))
    )
    suppression = controls.get("output_suppression", {})
    suppression_result = (
        f"{suppression.get('status')}; {suppression.get('n_rows')} / "
        f"{suppression.get('n_rows')} exact structural zeros; instrumentation only"
    )
    lines.extend(
        [
            _markdown_table(
                ["control", "result"],
                [
                    ["random-direction null", random_result],
                    ["absent-coordinate null", absent_result],
                    [
                        "capability ΔNLL",
                        capability_nll_result
                        + f"; {capability.get('general_language_n_rows')} text×intervention rows",
                    ],
                    [
                        "two-hop accuracy clean → edited",
                        accuracy_result
                        + f"; {capability.get('twohop_n_rows')} off-target evaluations",
                    ],
                    ["known narration", narration_result],
                    ["identity-J baseline", identity_result],
                    ["output-suppression completeness", suppression_result],
                ],
            ),
            "",
            (
                "Random, absent-coordinate, capability, narration, and identity-J "
                "controls were run for the 7B two-hop phase; they were not repeated "
                "for 14B. Ambiguity has its own structural concept/meta-token "
                "suppression records."
            ),
            "",
            "## Scale comparison",
            "",
            f"Scale phase status: **{summary['scale']['status']}**. Qwen-32B: "
            f"**{summary['scale']['qwen32b'].get('status')}**.",
            "",
        ]
    )
    scale_rows: list[list[Any]] = []
    for model in summary["scale"].get("models", []):
        for method in model.get("methods", []):
            scale_rows.append(
                [
                    model["model"],
                    method["method"],
                    method.get("n"),
                    model.get("strict_workspace_usable"),
                    _fmt_statistic(method.get("causal_read_given_write", {})),
                    _fmt_statistic(method.get("causal_write_given_read", {})),
                    _fmt_statistic(method.get("mean_ablation_positive_damage", {})),
                    _fmt_statistic(method.get("attribution_predicted_vs_real", {})),
                ]
            )
    paired_rows = [
        [
            row["method"],
            row.get("n"),
            _fmt_statistic(row.get("read_partial_delta", {})),
            _fmt_statistic(row.get("write_partial_delta", {})),
            _fmt_statistic(row.get("ablation_damage_delta", {})),
            _fmt_statistic(row.get("attribution_r_delta", {})),
        ]
        for row in summary["scale"].get("paired_14b_minus_7b", [])
    ]
    lines.extend(
        [
            _markdown_table(
                [
                    "scale",
                    "direction",
                    "N",
                    "strict usable",
                    "CAUSAL–READ | WRITE",
                    "CAUSAL–WRITE | READ",
                    "mean ablation damage",
                    "attribution r",
                ],
                scale_rows,
            ),
            "",
            "Paired 14B − 7B differences on common frozen items:",
            "",
            _markdown_table(
                [
                    "direction",
                    "N common",
                    "Δ READ partial",
                    "Δ WRITE partial",
                    "Δ mean damage",
                    "Δ attribution r",
                ],
                paired_rows,
            ),
            "",
            (
                "Negative Δ READ partial means the conditional READ association "
                "weakened at 14B; the raw-direction CI excludes zero, so the "
                "preregistered pattern reversed rather than sharpened."
            ),
            "",
            (
                "32B was not downloaded: published weight sizes project all three "
                "models to 102.8 GiB on the measured 100 GiB quota, before lens, "
                "checkpoint, activation, or temporary-file headroom."
            ),
            "",
            "## Figures",
            "",
            "- [F1 — 7B CAUSAL versus READ/WRITE](figures/f1_twohop_qwen2.5-7b.png)",
            "- [F1 — 14B CAUSAL versus READ/WRITE](figures/f1_twohop_qwen2.5-14b.png)",
            "- [F2 — 7B conditional coefficients](figures/f2_twohop_qwen2.5-7b.png)",
            "- [F2 — 14B conditional coefficients](figures/f2_twohop_qwen2.5-14b.png)",
            "- [F3 — internal ablation versus output suppression](figures/f3_internal_vs_output_suppression.png)",
            "- [F4 — READ localization](figures/f4_read_localization_qwen2.5-7b.png)",
            "- [F5 — attribution versus real ablation](figures/f5_attribution_vs_ablation_qwen7b.png)",
            "- [F6 — direction and weight-READ robustness](figures/f6_direction_robustness_qwen2.5-7b.png)",
            "- [F6 — 14B direction robustness](figures/f6_direction_robustness_qwen2.5-14b.png)",
            "- [F7 — scale comparison](figures/f7_scale_comparison.png)",
            "- [F8 — ambiguity and meta-token diagnostics](figures/f8_ambiguity_write_read.png)",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    lines.extend(
        [
            "",
            "## Verdicts",
            "",
            _markdown_table(
                ["prediction", "verdict"],
                [[key, value] for key, value in summary["verdicts"].items()],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_report(
    metrics: Mapping[str, Any],
    *,
    root: str | Path = ROOT,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate, summarize, and render the final report deterministically."""

    completeness = (
        validate_report_inputs(metrics, root=root, require_files=True)
        if require_complete
        else {
            "status": "NOT_CHECKED",
            "figures": collect_figure_catalog(metrics, root=root),
        }
    )
    summary = build_report_summary(metrics)
    markdown = render_results_markdown(summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "completeness": completeness,
        "summary": summary,
        "markdown": markdown,
        "figures": completeness["figures"],
    }


__all__ = [
    "PREREGISTERED_HYPOTHESIS",
    "ReportCompletenessError",
    "build_report_summary",
    "collect_figure_catalog",
    "load_metrics",
    "render_report",
    "render_results_markdown",
    "validate_report_inputs",
]

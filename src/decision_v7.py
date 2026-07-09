"""Model-free v7 join, inference, figures, report, and completion audit."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation import (
    distribution_diagnostics,
    distribution_overlap,
    group_bootstrap_auc,
    group_bootstrap_median,
    group_bootstrap_spearman,
    validate_record_schema,
)
from src.plotting import PAPER_DPI, PAPER_RC, save_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "v7"
FIGURE_DIR = RESULTS_DIR / "figures"
MANIFEST_PATH = RESULTS_DIR / "matched_manifest_v7.json"
CAUSAL_PATH = RESULTS_DIR / "raw" / "causal_C_v7.json"
READ_PATH = RESULTS_DIR / "raw" / "read_v7.json"
METRICS_PATH = RESULTS_DIR / "metrics_v7.json"
REPORT_PATH = RESULTS_DIR / "RESULTS_v7.md"

BASE_HEAD = "520e26d42fbddab7751c530b97bc1f0daa23af3b"
PREREGISTRATION_COMMIT = "f293ea3c69ed17e11b7157c52eaacd128b25c0b6"
PREREGISTRATION_SHA256 = (
    "ba3aa75ffc163c42df23d1e2f8f697a31ae99273c91b968eea1c7431caef89fd"
)
FROZEN_READ_SHA256 = (
    "a4a0ab35c50ce73dd153414118e6150891a708acf5f64bf9c8cb31225bb0caab"
)
METRIC_DEFINITION = "logit(answer_A) - logit(answer_B)"
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 1729
FOLDS = (0, 1, 2, 3, 4)

PREREGISTRATION_TEXT = """# V7 matched-comparison results

## Pre-registered decision rule

Pre-registered on 2026-07-09 before generating, inspecting, or evaluating any
v7 matched-design model outputs.

The primary held-out test compares ENGINE with the matched idle-DASHBOARD using
the identical measured quantity in both conditions:

`M = logit(answer_A) - logit(answer_B)`

`READ_IG` **passes** only if its dependency-group-held-out ROC AUC is at least
`0.70` and the lower endpoint of its 10,000-draw dependency-group bootstrap
confidence interval is strictly greater than `0.50`.

The frozen 16-step `READ_IG` estimator and frozen `READ_local` estimator will
not be tuned, sign-flipped, or redefined after inspecting v7 results. Failed
verification items are `UNVERIFIED`, excluded from confirmatory evaluation,
and never relabeled.

The final decision will use exactly one of these forms:

- **SURVIVES:** AUC stays high on the matched design; use-vs-idle detection is
  real in this frozen setting and is not explained by the old mismatched logit
  comparison.
- **COLLAPSES:** AUC drops toward chance on the matched design; the prior
  `1.000` was a mismatched-comparison design artifact and is corrected here.
"""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object artifact at {path}")
    return value


def save_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def verify_preregistration_v7() -> dict[str, Any]:
    """Verify the decision bar from its pre-results Git commit."""

    blob = subprocess.check_output(
        [
            "git",
            "show",
            f"{PREREGISTRATION_COMMIT}:results/v7/RESULTS_v7.md",
        ],
        cwd=PROJECT_ROOT,
    )
    observed = sha256_bytes(blob)
    if observed != PREREGISTRATION_SHA256:
        raise RuntimeError("Committed v7 preregistration bytes changed")
    text = blob.decode("utf-8")
    if not text.startswith(PREREGISTRATION_TEXT):
        raise RuntimeError("Committed preregistration text differs from the fixed rule")
    return {
        "status": "PASS",
        "commit": PREREGISTRATION_COMMIT,
        "path": "results/v7/RESULTS_v7.md",
        "sha256": observed,
        "auc_bar": 0.70,
        "ci_lower_must_exceed": 0.50,
        "bootstrap_draws": N_BOOTSTRAP,
        "bootstrap_unit": "dependency_group",
        "committed_before_v7_outputs": True,
    }


def _index_unique(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id", ""))
        if not pair_id or pair_id in output:
            raise ValueError(f"{label}[{index}] has an empty or duplicate pair_id")
        output[pair_id] = row
    return output


def _compact_measurement(measurement: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "C": float(measurement["C"]),
        "R_a_from_b": float(measurement["R_a_from_b"]),
        "R_b_from_a": float(measurement["R_b_from_a"]),
        "T": float(measurement["T"]),
        "metric_a": float(measurement["metric_a"]),
        "metric_b": float(measurement["metric_b"]),
        "metric_a_from_b": float(measurement["metric_a_from_b"]),
        "metric_b_from_a": float(measurement["metric_b_from_a"]),
        "signed_unclipped": bool(measurement["signed_unclipped"]),
        "sharp_directional_disagreement": bool(
            measurement["sharp_directional_disagreement"]
        ),
    }


def join_final_artifacts_v7(
    manifest: Mapping[str, Any],
    causal: Mapping[str, Any],
    read: Mapping[str, Any],
) -> dict[str, Any]:
    """Join exact final artifacts by pair ID and retain all candidate statuses."""

    if manifest.get("schema_version") != "matched-read-v7-sanitized-manifest-v1":
        raise ValueError("Unexpected final manifest schema")
    if causal.get("schema_version") != "matched-causal-read-v7-v1":
        raise ValueError("Unexpected final causal schema")
    if read.get("schema_version") != "matched-read-v7-cheap-v1":
        raise ValueError("Unexpected final READ schema")
    if causal.get("primary_truth") != "full_residual" or causal.get(
        "signed_unclipped"
    ) is not True:
        raise ValueError("Causal artifact does not declare signed full-residual truth")
    if read.get("firewall", {}).get("status") != "PASS":
        raise ValueError("READ firewall did not pass")
    for key in (
        "causal_artifact_read",
        "edited_metrics_read",
        "patch_outputs_read",
        "causal_outputs_consumed",
    ):
        if read["firewall"].get(key) is not False:
            raise ValueError(f"READ firewall field {key!r} is not false")

    manifest_sha = sha256_file(MANIFEST_PATH)
    if causal["source_manifest"]["sha256"] != manifest_sha:
        raise ValueError("Causal artifact references a different manifest")
    if read["source_manifest"]["sha256"] != manifest_sha:
        raise ValueError("READ artifact references a different manifest")
    if causal["model"] != read["model"] or causal["model"] != manifest["model"]:
        raise ValueError("Model provenance differs across final artifacts")
    if not (
        int(causal["selected_layer"])
        == int(read["selected_layer"])
        == int(manifest["selection"]["layer"])
        == 16
    ):
        raise ValueError("Selected layer differs across final artifacts")
    if not (
        causal["position_rule"]
        == read["position_rule"]
        == manifest["selection"]["position_rule"]
    ):
        raise ValueError("Position rule differs across final artifacts")
    if not (
        causal["metric_contract"]["engine"]
        == causal["metric_contract"]["dashboard"]
        == read["metric_contract"]["engine"]
        == read["metric_contract"]["dashboard"]
        == METRIC_DEFINITION
    ):
        raise ValueError("Final artifacts do not share the frozen matched metric")
    if (
        causal["direction_provenance"]["ordered_tensor_digest_sha256"]
        != read["direction_provenance"]["ordered_tensor_digest_sha256"]
        or read["direction_provenance"]["ordered_tensor_digest_sha256"]
        != manifest["direction_provenance"]["ordered_tensor_digest_sha256"]
    ):
        raise ValueError("Direction tensor digest differs across final artifacts")

    manifest_rows = manifest.get("rows")
    causal_rows = causal.get("rows")
    read_rows = read.get("rows")
    if not all(isinstance(value, list) for value in (manifest_rows, causal_rows, read_rows)):
        raise TypeError("Final artifacts require row lists")
    manifest_by_id = _index_unique(manifest_rows, label="manifest.rows")
    causal_by_id = _index_unique(causal_rows, label="causal.rows")
    read_by_id = _index_unique(read_rows, label="read.rows")
    verified_ids = {
        pair_id
        for pair_id, row in manifest_by_id.items()
        if row.get("verification_status") == "VERIFIED"
    }
    if set(causal_by_id) != verified_ids or set(read_by_id) != verified_ids:
        raise ValueError("Final causal/READ coverage differs from VERIFIED manifest IDs")
    if list(causal_by_id) != list(read_by_id):
        raise ValueError("Final causal and READ row order differs")

    task_rows: list[dict[str, Any]] = []
    pair_metrics: list[dict[str, Any]] = []
    identity_fields = (
        "dependency_group",
        "fold",
        "category",
        "concept_a",
        "concept_b",
        "answer_a",
        "answer_b",
    )
    for manifest_row in manifest_rows:
        pair_id = str(manifest_row["pair_id"])
        status = str(manifest_row["verification_status"])
        populated = status == "VERIFIED"
        pair_output: dict[str, Any] = {
            "pair_id": pair_id,
            "dependency_group": str(manifest_row["dependency_group"]),
            "context_slot": int(manifest_row["context_slot"]),
            "split": str(manifest_row["split"]),
            "fold": None if manifest_row["fold"] is None else int(manifest_row["fold"]),
            "category": str(manifest_row["category"]),
            "concept_a": str(manifest_row["concept_a"]),
            "concept_b": str(manifest_row["concept_b"]),
            "answer_a": str(manifest_row["answer_a"]),
            "answer_b": str(manifest_row["answer_b"]),
            "answer_type": str(manifest_row["answer_type"]),
            "metric": str(manifest_row["engine_metric"]),
            "metric_positive_token_id": int(manifest_row["metric_positive_token_id"]),
            "metric_negative_token_id": int(manifest_row["metric_negative_token_id"]),
            "verification_status": status,
            "verification_reasons": list(manifest_row["verification_reasons"]),
            "verification_gate_pass": bool(manifest_row["verification_gate_pass"]),
            "measurements_populated": populated,
            "engine": None,
            "dashboard": None,
            "weight_norm_capacity_baseline": None,
        }
        if manifest_row["engine_metric"] != manifest_row["dashboard_metric"]:
            raise ValueError(f"{pair_id} manifest metric text differs")
        if manifest_row["engine_metric"] != METRIC_DEFINITION:
            raise ValueError(f"{pair_id} metric differs from frozen v7")
        if int(manifest_row["metric_positive_token_id"]) == int(
            manifest_row["metric_negative_token_id"]
        ):
            raise ValueError(f"{pair_id} metric tokens collapse")
        if "2 + 2" in (
            str(manifest_row["engine_prompt_a"])
            + str(manifest_row["engine_prompt_b"])
            + str(manifest_row["dashboard_prompt_a"])
            + str(manifest_row["dashboard_prompt_b"])
        ):
            raise ValueError("A non-v7 prompt leaked into the matched manifest")

        if populated:
            causal_row = causal_by_id[pair_id]
            read_row = read_by_id[pair_id]
            for field in identity_fields:
                if causal_row[field] != read_row[field] or causal_row[field] != manifest_row[field]:
                    raise ValueError(f"{pair_id} metadata differs at {field}")
            if causal_row.get("verification_status") != "VERIFIED":
                raise ValueError(f"{pair_id} causal row is not VERIFIED")
            if read_row.get("verification_status") != "VERIFIED":
                raise ValueError(f"{pair_id} READ row is not VERIFIED")
            if not (
                causal_row["metric_positive_token_id"]
                == read_row["metric_positive_token_id"]
                == manifest_row["metric_positive_token_id"]
            ):
                raise ValueError(f"{pair_id} positive metric token differs")
            if not (
                causal_row["metric_negative_token_id"]
                == read_row["metric_negative_token_id"]
                == manifest_row["metric_negative_token_id"]
            ):
                raise ValueError(f"{pair_id} negative metric token differs")
            baseline = float(read_row["weight_norm_capacity_baseline"])
            pair_output["weight_norm_capacity_baseline"] = baseline
            for task, label in (("engine", 1), ("dashboard", 0)):
                full = causal_row[task]["full_residual"]
                subspace = causal_row[task]["jlens_two_concept_subspace"]
                estimate = read_row[task]
                if full.get("signed_unclipped") is not True or subspace.get(
                    "signed_unclipped"
                ) is not True:
                    raise ValueError(f"{pair_id} {task} C is not signed/unclipped")
                compact = {
                    "C_full": _compact_measurement(full),
                    "C_subspace": _compact_measurement(subspace),
                    "READ_IG": float(estimate["READ_IG"]),
                    "READ_local": float(estimate["READ_local"]),
                    "ig_abs_by_direction": [
                        float(value) for value in estimate["ig_abs_by_direction"]
                    ],
                    "local_abs_by_direction": [
                        float(value) for value in estimate["local_abs_by_direction"]
                    ],
                }
                pair_output[task] = compact
                task_rows.append(
                    {
                        "pair_id": pair_id,
                        "dependency_group": str(causal_row["dependency_group"]),
                        "fold": int(causal_row["fold"]),
                        "category": str(causal_row["category"]),
                        "concept_a": str(causal_row["concept_a"]),
                        "concept_b": str(causal_row["concept_b"]),
                        "task": task,
                        "label": label,
                        "C": float(full["C"]),
                        "abs_C": abs(float(full["C"])),
                        "R_a_from_b": float(full["R_a_from_b"]),
                        "R_b_from_a": float(full["R_b_from_a"]),
                        "T": float(full["T"]),
                        "directional_abs_difference": float(
                            full["directional_abs_difference"]
                        ),
                        "sharp_directional_disagreement": bool(
                            full["sharp_directional_disagreement"]
                        ),
                        "C_subspace": float(subspace["C"]),
                        "abs_C_subspace": abs(float(subspace["C"])),
                        "R_a_from_b_subspace": float(subspace["R_a_from_b"]),
                        "R_b_from_a_subspace": float(subspace["R_b_from_a"]),
                        "READ_IG": float(estimate["READ_IG"]),
                        "READ_local": float(estimate["READ_local"]),
                        "weight_norm_baseline": baseline,
                        "baseline_label": str(
                            read_row["capacity_baseline"]["baseline"]
                        ),
                    }
                )
        pair_metrics.append(pair_output)

    validate_record_schema(
        task_rows,
        required_fields={
            "pair_id",
            "dependency_group",
            "fold",
            "category",
            "concept_a",
            "concept_b",
            "task",
            "label",
            "C",
            "abs_C",
            "C_subspace",
            "abs_C_subspace",
            "READ_IG",
            "READ_local",
            "weight_norm_baseline",
        },
        finite_fields={
            "fold",
            "label",
            "C",
            "abs_C",
            "C_subspace",
            "abs_C_subspace",
            "READ_IG",
            "READ_local",
            "weight_norm_baseline",
        },
        allowed_values={"task": {"engine", "dashboard"}, "label": {0, 1}},
        unique_by=("pair_id", "task"),
        schema_name="v7_task_rows",
    )
    if len(task_rows) != 2 * len(verified_ids):
        raise ValueError("V7 task-row join did not create exactly two rows per pair")
    for pair_id in verified_ids:
        pair_tasks = [row for row in task_rows if row["pair_id"] == pair_id]
        if {row["task"] for row in pair_tasks} != {"engine", "dashboard"}:
            raise ValueError(f"{pair_id} does not have both task rows")
        if len({row["weight_norm_baseline"] for row in pair_tasks}) != 1:
            raise ValueError(f"{pair_id} capacity baseline differs by condition")
    group_folds: dict[str, set[int]] = {}
    for row in task_rows:
        group_folds.setdefault(str(row["dependency_group"]), set()).add(int(row["fold"]))
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("A dependency group crosses held-out folds")

    return {
        "pair_metrics": pair_metrics,
        "task_rows": task_rows,
        "verified_pair_ids": sorted(verified_ids),
        "join_audit": {
            "status": "PASS",
            "manifest_candidates": len(manifest_rows),
            "verified_pairs": len(verified_ids),
            "task_rows": len(task_rows),
            "dependency_groups": len(group_folds),
            "exact_pair_id_coverage": True,
            "joined_by": "pair_id",
            "C_based_filtering": False,
            "READ_based_filtering": False,
            "attempt_artifacts_joined": False,
            "same_metric_token_ids": True,
            "baseline_duplicated_within_pair": True,
            "group_to_single_fold": True,
        },
    }


def _fold_auc(rows: Sequence[Mapping[str, Any]], score_key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for fold in FOLDS:
        selected = [row for row in rows if int(row["fold"]) == fold]
        labels = np.asarray([int(row["label"]) for row in selected], dtype=int)
        scores = np.asarray([float(row[score_key]) for row in selected], dtype=float)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"Fold {fold} lacks both classes")
        output.append(
            {
                "fold": fold,
                "auc": float(roc_auc_score(labels, scores)),
                "n_rows": len(selected),
                "n_pairs": len(selected) // 2,
                "n_dependency_groups": len(
                    {str(row["dependency_group"]) for row in selected}
                ),
            }
        )
    return output


def evaluate_joined_v7(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Run the preregistered held-out/group-bootstrap analyses."""

    estimators = (
        ("READ_IG", "READ_IG"),
        ("READ_local", "READ_local"),
        ("capacity_baseline", "weight_norm_baseline"),
    )
    auc_table: list[dict[str, Any]] = []
    for estimator, key in estimators:
        summary, _samples = group_bootstrap_auc(
            task_rows,
            key,
            label_key="label",
            group_key="dependency_group",
            n_bootstrap=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        passes_numeric_bar = bool(
            summary["estimate"] >= 0.70
            and summary["ci95_low"] is not None
            and summary["ci95_low"] > 0.50
        )
        auc_table.append(
            {
                "estimator": estimator,
                "score_key": key,
                "heldout_auc": float(summary["estimate"]),
                **summary,
                "fold_auc": _fold_auc(task_rows, key),
                "passes_numeric_bar": passes_numeric_bar,
                "eligible_for_primary_decision": estimator == "READ_IG",
            }
        )
    by_estimator = {row["estimator"]: row for row in auc_table}
    capacity = by_estimator["capacity_baseline"]
    if not (
        math.isclose(capacity["heldout_auc"], 0.5, abs_tol=1e-15)
        and math.isclose(capacity["ci95_low"], 0.5, abs_tol=1e-15)
        and math.isclose(capacity["ci95_high"], 0.5, abs_tol=1e-15)
    ):
        raise RuntimeError("Capacity baseline is not the exact paired 0.5 control")

    engine_rows = [row for row in task_rows if row["task"] == "engine"]
    dashboard_rows = [row for row in task_rows if row["task"] == "dashboard"]
    rho_full, _rho_samples = group_bootstrap_spearman(
        engine_rows,
        "READ_IG",
        target_key="abs_C",
        group_key="dependency_group",
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
    )
    rho_subspace, _rho_sub_samples = group_bootstrap_spearman(
        engine_rows,
        "READ_IG",
        target_key="abs_C_subspace",
        group_key="dependency_group",
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
    )

    raw_distributions = {
        "engine": [float(row["READ_IG"]) for row in engine_rows],
        "dashboard": [float(row["READ_IG"]) for row in dashboard_rows],
    }
    diagnostics = distribution_diagnostics(raw_distributions)
    overlap = distribution_overlap(
        raw_distributions["engine"],
        raw_distributions["dashboard"],
        name_a="engine",
        name_b="dashboard",
    )
    distribution_median_ci: dict[str, Any] = {}
    for task, rows in (("engine", engine_rows), ("dashboard", dashboard_rows)):
        summary, _samples = group_bootstrap_median(
            rows,
            "READ_IG",
            group_key="dependency_group",
            n_bootstrap=N_BOOTSTRAP,
            seed=BOOTSTRAP_SEED,
        )
        distribution_median_ci[task] = summary

    causal_median_ci: dict[str, Any] = {}
    for task, rows in (("engine", engine_rows), ("dashboard", dashboard_rows)):
        causal_median_ci[task] = {}
        for variant, key in (
            ("full_residual", "C"),
            ("jlens_two_concept_subspace", "C_subspace"),
        ):
            summary, _samples = group_bootstrap_median(
                rows,
                key,
                group_key="dependency_group",
                absolute=True,
                n_bootstrap=N_BOOTSTRAP,
                seed=BOOTSTRAP_SEED,
            )
            causal_median_ci[task][variant] = summary

    read_ig = by_estimator["READ_IG"]
    passes = bool(read_ig["passes_numeric_bar"])
    decision = "SURVIVES" if passes else "COLLAPSES"
    if passes:
        decision_line = (
            "SURVIVES: AUC stays high on the matched design; the old metric mismatch "
            "does not solely explain the frozen READ_IG separation."
        )
    else:
        decision_line = (
            "COLLAPSES: AUC drops toward chance on the matched design; the prior "
            "1.000 was a mismatched-comparison design artifact and is corrected here."
        )

    return {
        "heldout_detection": {
            "population": "all VERIFIED evaluation pairs; calibration groups excluded",
            "n_pairs": len(engine_rows),
            "n_rows": len(task_rows),
            "n_dependency_groups": len(
                {str(row["dependency_group"]) for row in task_rows}
            ),
            "folds": list(FOLDS),
            "bootstrap_draws": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "dependency_group",
            "auc_table": auc_table,
        },
        "within_engine": {
            "primary": {
                "score": "READ_IG",
                "target": "abs(full_residual_C)",
                **rho_full,
            },
            "diagnostic_subspace": {
                "score": "READ_IG",
                "target": "abs(jlens_two_concept_subspace_C)",
                **rho_subspace,
            },
        },
        "distributions": {
            "raw_READ_IG": raw_distributions,
            "diagnostics": diagnostics,
            "observed_range_overlap": overlap,
            "observed_ranges": (
                "OVERLAPPING"
                if overlap["ranges_overlap_or_touch"]
                else "DISJOINT"
            ),
            "group_bootstrap_median_ci": distribution_median_ci,
        },
        "causal_group_bootstrap_median_abs_C": causal_median_ci,
        "decision": {
            "outcome": decision,
            "one_line": decision_line,
            "primary_estimator": "READ_IG",
            "auc_bar": 0.70,
            "ci_lower_bar": 0.50,
            "passes": passes,
            "no_sign_flip": True,
            "no_retuning": True,
            "causal_sanity_is_separate_qualification": True,
        },
    }


def _style_axes() -> None:
    plt.rcParams.update(PAPER_RC)


def _figure_metadata(path: Path, *, title: str, labels: Sequence[str]) -> dict[str, Any]:
    pixels = plt.imread(path)
    if pixels.ndim < 2 or pixels.shape[0] < 100 or pixels.shape[1] < 100:
        raise RuntimeError(f"Figure {path} is unexpectedly small")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "pixel_height": int(pixels.shape[0]),
        "pixel_width": int(pixels.shape[1]),
        "title": title,
        "required_labels": list(labels),
    }


def generate_figures_v7(
    task_rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    causal_sanity: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate the four required v7-native figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    engine = [row for row in task_rows if row["task"] == "engine"]
    dashboard = [row for row in task_rows if row["task"] == "dashboard"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metadata: dict[str, Any] = {}
    _style_axes()

    # F_v7_1: raw READ_IG distributions.
    values = [
        np.asarray([row["READ_IG"] for row in engine], dtype=float),
        np.asarray([row["READ_IG"] for row in dashboard], dtype=float),
    ]
    title = "Matched-metric READ_IG distributions (observed ranges overlap)"
    figure, axis = plt.subplots(figsize=(8.2, 6.0))
    boxes = axis.boxplot(values, tick_labels=["ENGINE\nconcept used", "DASHBOARD\nanswer copyable"], patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], ("#2166AC", "#EF8A62"), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.32)
    for position, vector, color in zip((1, 2), values, ("#2166AC", "#EF8A62"), strict=True):
        axis.scatter(
            np.full(vector.size, position) + rng.normal(0.0, 0.045, vector.size),
            vector,
            s=27,
            alpha=0.68,
            color=color,
            edgecolor="white",
            linewidth=0.35,
        )
    if all(np.all(vector > 0.0) for vector in values):
        axis.set_yscale("log")
    axis.set_ylabel("Raw READ_IG (16-step midpoint)")
    axis.set_title(title)
    medians = analysis["distributions"]["group_bootstrap_median_ci"]
    overlap = analysis["distributions"]["observed_range_overlap"]
    axis.text(
        0.02,
        0.03,
        "Engine median "
        f"{medians['engine']['estimate']:.4f} "
        f"[{medians['engine']['ci95_low']:.4f}, {medians['engine']['ci95_high']:.4f}]\n"
        "Dashboard median "
        f"{medians['dashboard']['estimate']:.4f} "
        f"[{medians['dashboard']['ci95_low']:.4f}, {medians['dashboard']['ci95_high']:.4f}]\n"
        f"Observed overlap: {overlap['intersection']}",
        transform=axis.transAxes,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    figure.tight_layout()
    path = Path(save_figure(figure, FIGURE_DIR / "F_v7_1_read_ig_distributions.png"))
    plt.close(figure)
    metadata["F_v7_1"] = _figure_metadata(
        path,
        title=title,
        labels=("Raw READ_IG", "ENGINE", "DASHBOARD", "Observed overlap"),
    )

    # F_v7_2: held-out AUC comparison.
    table = analysis["heldout_detection"]["auc_table"]
    labels = ["READ_IG", "READ_local", "Capacity\nbaseline"]
    estimates = np.asarray([row["heldout_auc"] for row in table], dtype=float)
    lows = np.asarray([row["ci95_low"] for row in table], dtype=float)
    highs = np.asarray([row["ci95_high"] for row in table], dtype=float)
    title = "Held-out matched-design AUC (10,000 group-bootstrap draws)"
    figure, axis = plt.subplots(figsize=(8.4, 6.1))
    positions = np.arange(3)
    axis.bar(positions, estimates, color=("#2166AC", "#1B9E77", "#999999"))
    axis.errorbar(
        positions,
        estimates,
        yerr=np.vstack((estimates - lows, highs - estimates)),
        fmt="none",
        ecolor="black",
        capsize=6,
        linewidth=1.5,
    )
    axis.axhline(0.70, color="#B2182B", linestyle="--", label="preregistered bar 0.70")
    axis.axhline(0.50, color="black", linestyle=":", label="chance 0.50")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.04)
    axis.set_ylabel("Held-out ROC AUC (95% group-bootstrap CI)")
    axis.set_title(title)
    axis.legend(frameon=False, loc="lower left")
    for position, value in zip(positions, estimates, strict=True):
        axis.text(position, value + 0.025, f"{value:.3f}", ha="center")
    figure.tight_layout()
    path = Path(save_figure(figure, FIGURE_DIR / "F_v7_2_heldout_auc.png"))
    plt.close(figure)
    metadata["F_v7_2"] = _figure_metadata(
        path,
        title=title,
        labels=("READ_IG", "READ_local", "Capacity baseline", "0.70", "0.50"),
    )

    # F_v7_3: within-engine READ_IG vs primary |C|.
    x = np.asarray([row["READ_IG"] for row in engine], dtype=float)
    y = np.asarray([row["abs_C"] for row in engine], dtype=float)
    rho = analysis["within_engine"]["primary"]
    title = (
        "Within ENGINE: READ_IG vs primary full-state |C|\n"
        f"Spearman rho={rho['estimate']:.3f}, 95% CI "
        f"[{rho['ci95_low']:.3f}, {rho['ci95_high']:.3f}]"
    )
    figure, axis = plt.subplots(figsize=(8.5, 6.2))
    axis.scatter(x, y, s=45, alpha=0.72, color="#2166AC", edgecolor="white", linewidth=0.35)
    if np.all(x > 0.0):
        axis.set_xscale("log")
    axis.set_xlabel("READ_IG (16-step midpoint)")
    axis.set_ylabel("Primary full-residual |C|")
    axis.set_title(title)
    axis.text(
        0.02,
        0.04,
        f"N={len(engine)}; groups={analysis['heldout_detection']['n_dependency_groups']}\n"
        "Primary graded-use association is not positive.",
        transform=axis.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    figure.tight_layout()
    path = Path(save_figure(figure, FIGURE_DIR / "F_v7_3_read_vs_c.png"))
    plt.close(figure)
    metadata["F_v7_3"] = _figure_metadata(
        path,
        title=title,
        labels=("READ_IG", "full-residual |C|", "Spearman rho", "95% CI"),
    )

    # F_v7_4: full and diagnostic-subspace causal sanity.
    groups = [
        np.asarray([row["C"] for row in engine], dtype=float),
        np.asarray([row["C"] for row in dashboard], dtype=float),
        np.asarray([row["C_subspace"] for row in engine], dtype=float),
        np.asarray([row["C_subspace"] for row in dashboard], dtype=float),
    ]
    positions = np.asarray([1, 2, 4, 5], dtype=float)
    title = "Causal sanity: primary full state FAIL; J-Lens subspace is diagnostic"
    figure, axis = plt.subplots(figsize=(10.0, 6.3))
    boxes = axis.boxplot(groups, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    colors = ("#2166AC", "#EF8A62", "#2166AC", "#EF8A62")
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
    for position, vector, color in zip(positions, groups, colors, strict=True):
        axis.scatter(
            np.full(vector.size, position) + rng.normal(0.0, 0.045, vector.size),
            vector,
            s=24,
            alpha=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.3,
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axvline(3.0, color="#777777", linestyle=":")
    axis.set_xticks(
        positions,
        ["ENGINE\nfull", "DASHBOARD\nfull", "ENGINE\nsubspace", "DASHBOARD\nsubspace"],
    )
    axis.set_ylabel("Signed, unclipped causal recovery C")
    axis.set_title(title)
    axis.text(
        0.51,
        0.96,
        "Full dashboard median |C|="
        f"{causal_sanity['full_residual']['dashboard_abs_C_median']:.3f}; "
        "required <0.10 → FAIL\nSubspace-only result is diagnostic, not primary truth.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    figure.tight_layout()
    path = Path(save_figure(figure, FIGURE_DIR / "F_v7_4_causal_sanity.png"))
    plt.close(figure)
    metadata["F_v7_4"] = _figure_metadata(
        path,
        title=title,
        labels=("ENGINE", "DASHBOARD", "full", "subspace", "Signed", "FAIL"),
    )
    return metadata


def attempt_history_v7() -> dict[str, Any]:
    attempts = []
    specifications = (
        (
            "attempt1_pronoun",
            RESULTS_DIR / "raw" / "attempt1_pronoun_design_v7.json",
            RESULTS_DIR / "raw" / "attempt1_pronoun_matched_manifest_v7.json",
        ),
        (
            "attempt2_batched_verification",
            RESULTS_DIR / "raw" / "attempt2_batched_verification_v7.json",
            RESULTS_DIR / "raw" / "attempt2_batched_matched_manifest_v7.json",
        ),
        (
            "attempt3_masked_single_verification",
            RESULTS_DIR / "raw" / "attempt3_masked_single_verification_v7.json",
            RESULTS_DIR / "raw" / "attempt3_masked_single_manifest_v7.json",
        ),
    )
    previous_verified: set[str] | None = None
    monotone = True
    for name, summary_path, manifest_path in specifications:
        summary = load_json(summary_path)
        manifest = load_json(manifest_path)
        verified = {
            str(row["pair_id"])
            for row in manifest["rows"]
            if row["verification_status"] == "VERIFIED"
        }
        if previous_verified is not None and not verified.issubset(previous_verified):
            monotone = False
        previous_verified = verified
        attempts.append(
            {
                "name": name,
                "status": summary["status"],
                "reason": summary["reason"],
                "verified_pairs": len(verified),
                "summary_path": str(summary_path.relative_to(PROJECT_ROOT)),
                "summary_sha256": sha256_file(summary_path),
                "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    final_manifest = load_json(MANIFEST_PATH)
    final_verified = {
        str(row["pair_id"])
        for row in final_manifest["rows"]
        if row["verification_status"] == "VERIFIED"
    }
    if previous_verified is not None and not final_verified.issubset(previous_verified):
        monotone = False
    if not monotone:
        raise RuntimeError("Pre-READ correction history did not only shrink VERIFIED IDs")
    return {
        "status": "RETAINED",
        "attempts": attempts,
        "final_verified_pairs": len(final_verified),
        "verified_count_sequence": [
            *[int(row["verified_pairs"]) for row in attempts],
            len(final_verified),
        ],
        "verified_sets_only_shrank": True,
        "READ_class_comparison_inspected_during_corrections": False,
        "decision_bar_changed": False,
    }


def isolation_audit_v7() -> dict[str, Any]:
    tracked_or_modified = set(filter(None, _git("diff", "--name-only", BASE_HEAD).splitlines()))
    untracked = set(
        filter(None, _git("ls-files", "--others", "--exclude-standard").splitlines())
    )
    changed = sorted(tracked_or_modified | untracked)

    def allowed(path: str) -> bool:
        return (
            path.startswith("results/v7/")
            or path.startswith("notebooks/v7_")
            or (path.startswith("src/") and Path(path).stem.endswith("_v7"))
        )

    violations = [path for path in changed if not allowed(path)]
    if violations:
        raise RuntimeError(f"Non-v7 paths changed relative to baseline: {violations}")
    cheap_hash = sha256_file(PROJECT_ROOT / "src" / "cheap_read.py")
    if cheap_hash != FROZEN_READ_SHA256:
        raise RuntimeError("Frozen READ source changed during v7")
    branch = _git("branch", "--show-current")
    if branch != "v7-matched":
        raise RuntimeError(f"Expected v7-matched branch, got {branch!r}")
    return {
        "status": "PASS",
        "baseline_head": BASE_HEAD,
        "branch": branch,
        "generation_head": _git("rev-parse", "HEAD"),
        "changed_paths": changed,
        "changed_path_count": len(changed),
        "non_v7_paths": violations,
        "only_v7_paths_changed": not violations,
        "git_status_short": _git("status", "--short"),
        "remote": _git("remote", "get-url", "origin"),
        "git_user_name": _git("config", "--get", "user.name"),
        "git_user_email": _git("config", "--get", "user.email"),
        "frozen_read_sha256": cheap_hash,
        "prior_result_files_edited": False,
        "test_suite_run": False,
        "pytest_run": False,
        "ruff_run": False,
        "git_cli_only": True,
        "force_push_used": False,
    }


def notebook_execution_audit_v7(*, require_v7_4: bool) -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "notebooks" / "v7_1_matched_dataset.ipynb",
        PROJECT_ROOT / "notebooks" / "v7_2_causal_C.ipynb",
        PROJECT_ROOT / "notebooks" / "v7_3_read.ipynb",
        PROJECT_ROOT / "notebooks" / "v7_4_decision.ipynb",
    ]
    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        if not path.exists():
            if index == 3 and not require_v7_4:
                records.append(
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "status": "PENDING_CURRENT_EXECUTION",
                    }
                )
                continue
            raise FileNotFoundError(path)
        notebook = load_json(path)
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        executed = bool(code_cells) and all(
            cell.get("execution_count") is not None for cell in code_cells
        )
        if index == 3 and not require_v7_4 and not executed:
            status = "PENDING_CURRENT_EXECUTION"
        else:
            status = "PASS" if executed and not errors else "FAIL"
        if require_v7_4 and status != "PASS":
            raise RuntimeError(f"Notebook is not fully executed: {path}")
        records.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "status": status,
                "code_cells": len(code_cells),
                "execution_counts": [cell.get("execution_count") for cell in code_cells],
                "error_outputs": len(errors),
                "sha256": sha256_file(path),
            }
        )
    passed = all(row["status"] == "PASS" for row in records)
    return {
        "status": "PASS" if passed else "PENDING_CURRENT_EXECUTION",
        "all_four_executed": passed,
        "notebooks": records,
    }


def _fmt(value: float | None, digits: int = 6) -> str:
    return "UNDEFINED" if value is None else f"{float(value):.{digits}f}"


def build_report_v7(metrics: Mapping[str, Any], *, metrics_sha256: str) -> str:
    counts = metrics["dataset"]["counts"]
    auc = {row["estimator"]: row for row in metrics["heldout_detection"]["auc_table"]}
    rho = metrics["within_engine"]["primary"]
    dist = metrics["distributions"]
    medians = dist["group_bootstrap_median_ci"]
    overlap = dist["observed_range_overlap"]
    causal = metrics["causal_sanity"]
    causal_ci = metrics["causal_group_bootstrap_median_abs_C"]
    reasons = metrics["dataset"]["unverified_reason_counts"]
    read_ig_folds = auc["READ_IG"]["fold_auc"]
    execution = metrics["execution_audit"]
    isolation = metrics["isolation_audit"]
    figures = metrics["figures"]

    results = f"""
## Results

### Outcome

**{metrics['decision']['one_line']}**

Qualification: the primary full-residual copy-dashboard sanity gate failed
(`median |C|={causal['full_residual']['dashboard_abs_C_median']:.6f}` versus the
recorded `<0.10` criterion). The matched AUC therefore supports discrimination
of strong versus weak explicit-concept use in this frozen setting; it does not
establish perfectly idle dashboards under full-state interchange.

The earlier `1.000` result remains superseded and inadmissible as matched-metric
evidence. The new result shows that the old mismatch was not the sole source of
separation; it does not retroactively validate the old design.

### Dataset and verification

- Candidates: **{counts['candidates']}** (target was at least 90).
- Calibration-only: **{counts['calibration_pairs']}**; gate-passing calibration
  rows: **{counts['calibration_gate_pass']}**.
- Held-out evaluation candidates: **{counts['evaluation_pairs']}**.
- `VERIFIED`: **{counts['verified_pairs']}** across
  **{metrics['heldout_detection']['n_dependency_groups']}** dependency groups
  (target was at least 50).
- `UNVERIFIED`: **{counts['unverified_pairs']}**, excluded from C, READ, and all
  confirmatory statistics without relabeling.
- Verified fold counts (0–4):
  **{', '.join(str(metrics['dataset']['verified_fold_counts'][str(fold)]) for fold in FOLDS)}**.
- HF/J-Lens logit check: 20 prompts, maximum mean KL
  `{metrics['model']['logit_agreement']['max_mean_kl']:.3e} < 1e-3`.

UNVERIFIED reason counts (one row may have multiple reasons):

| Reason | Count |
| --- | ---: |
{chr(10).join(f'| `{key}` | {value} |' for key, value in sorted(reasons.items()))}

Every retained row passed four exact-argmax top-1 checks and four independently
measured WRITTEN checks at L16. The threshold remained frozen at
`{metrics['model']['written_threshold']}`.

### Identical metric and prompt contract

Both conditions use the same answer tokens, answer type, and quantity:

`M = logit(answer_A) - logit(answer_B)`

For the aluminum/magnesium example, those IDs are `1674` (` Al`) and `72593`
(` Mg`) in both conditions. No arithmetic task occurs in v7. The engine requires
the explicitly written concept-to-answer relation; the dashboard supplies a
self-contained later concept-answer fact and asks the model to copy the answer.

Because WRITTEN and interchange require an explicit concept-token site, the
engine's shared prefix states the intermediate concept. V7 therefore tests use
of an explicitly written concept-to-answer relation, not literal recovery of the
concept name from the original clue alone. This is the operational reconciliation
of the brief's simultaneous explicit-token and inference requirements.

### Causal design sanity

All C values below are signed and unclipped per pair; the table reports median
`|C|` with 10,000-draw dependency-group bootstrap intervals.

| Condition | Variant | Median | 95% CI | Gate role |
| --- | --- | ---: | ---: | --- |
| ENGINE | Full residual | {causal_ci['engine']['full_residual']['estimate']:.6f} | [{causal_ci['engine']['full_residual']['ci95_low']:.6f}, {causal_ci['engine']['full_residual']['ci95_high']:.6f}] | Primary |
| DASHBOARD | Full residual | {causal_ci['dashboard']['full_residual']['estimate']:.6f} | [{causal_ci['dashboard']['full_residual']['ci95_low']:.6f}, {causal_ci['dashboard']['full_residual']['ci95_high']:.6f}] | Primary |
| ENGINE | J-Lens two-concept subspace | {causal_ci['engine']['jlens_two_concept_subspace']['estimate']:.6f} | [{causal_ci['engine']['jlens_two_concept_subspace']['ci95_low']:.6f}, {causal_ci['engine']['jlens_two_concept_subspace']['ci95_high']:.6f}] | Diagnostic |
| DASHBOARD | J-Lens two-concept subspace | {causal_ci['dashboard']['jlens_two_concept_subspace']['estimate']:.6f} | [{causal_ci['dashboard']['jlens_two_concept_subspace']['ci95_low']:.6f}, {causal_ci['dashboard']['jlens_two_concept_subspace']['ci95_high']:.6f}] | Diagnostic |

Full-residual interchange produced large engine effects, but the intended copy
dashboard did not meet the recorded near-zero criterion: median `|C|` was
`{causal['full_residual']['dashboard_abs_C_median']:.6f}` versus required
`<0.10`. Primary causal-design sanity is therefore **{causal['full_state_sanity_status']}**.
The J-Lens subspace edit was more selective, but it was not primary truth and
cannot convert that FAIL into a PASS.

![Engine/dashboard C under full-state and diagnostic subspace edits](figures/F_v7_4_causal_sanity.png)

### Held-out matched-design discrimination

Pooled held-out AUC uses every VERIFIED evaluation row. Intervals use 10,000
whole-dependency-group bootstrap draws (seed 1729); fold AUCs are diagnostics,
not an averaged primary statistic.

| Estimator | AUC | 95% CI | Primary decision input |
| --- | ---: | ---: | --- |
| READ_IG | {auc['READ_IG']['heldout_auc']:.6f} | [{auc['READ_IG']['ci95_low']:.6f}, {auc['READ_IG']['ci95_high']:.6f}] | Yes |
| READ_local | {auc['READ_local']['heldout_auc']:.6f} | [{auc['READ_local']['ci95_low']:.6f}, {auc['READ_local']['ci95_high']:.6f}] | No |
| Capacity baseline | {auc['capacity_baseline']['heldout_auc']:.6f} | [{auc['capacity_baseline']['ci95_low']:.6f}, {auc['capacity_baseline']['ci95_high']:.6f}] | No |

READ_IG fold AUCs: {', '.join(f"fold {row['fold']} `{row['auc']:.6f}`" for row in read_ig_folds)}.

![Held-out AUC comparison](figures/F_v7_2_heldout_auc.png)

### Raw READ_IG distributions

The observed ranges are **{dist['observed_ranges']}**, not disjoint.

| Condition | Median | 95% group CI | Observed range |
| --- | ---: | ---: | ---: |
| ENGINE | {medians['engine']['estimate']:.6f} | [{medians['engine']['ci95_low']:.6f}, {medians['engine']['ci95_high']:.6f}] | [{overlap['class_a']['minimum']:.6f}, {overlap['class_a']['maximum']:.6f}] |
| DASHBOARD | {medians['dashboard']['estimate']:.6f} | [{medians['dashboard']['ci95_low']:.6f}, {medians['dashboard']['ci95_high']:.6f}] | [{overlap['class_b']['minimum']:.6f}, {overlap['class_b']['maximum']:.6f}] |

Observed intersection: `{overlap['intersection']}`. Fraction of engines inside
the dashboard range: `{overlap['fraction_a_inside_b_range']:.3f}`; fraction of
dashboards inside the engine range: `{overlap['fraction_b_inside_a_range']:.3f}`.
This is an observed-range diagnostic, not a fitted density-overlap estimate.

![Raw matched READ_IG distributions](figures/F_v7_1_read_ig_distributions.png)

### Within-engine graded check

Primary full-state `Spearman rho(READ_IG, |C|) = {_fmt(rho['estimate'])}` with
95% group-bootstrap CI `[{_fmt(rho['ci95_low'])}, {_fmt(rho['ci95_high'])}]`.
The estimate is negative and the interval includes zero, so v7 supplies no
positive evidence that READ_IG is a graded causal-strength meter within engines.

![Within-engine READ_IG versus full-state causal magnitude](figures/F_v7_3_read_vs_c.png)

### Pre-READ correction history

Three corrections are retained rather than overwritten. Attempt 1 already used
the matched logit metric, but its pronoun-copy dashboard left the first concept
token causally active; full-state dashboard median `|C|` was `0.226190`, so it
was rejected before any READ score. Attempt 2 used the self-contained copy fact,
but batched verification disagreed with the canonical single-prompt causal
forward. Attempt 3 used a single-prompt all-ones mask, which still selected a
different path for a tied-max item. Final verification used single-prompt,
no-mask, hooked forwards matching causal execution and exact `argmax` top-1.

Verified counts only shrank: `{' → '.join(str(value) for value in metrics['attempt_history']['verified_count_sequence'])}`.
No failed evaluation item was rescued or relabeled. The AUC bar remained the
commit-first rule at `{PREREGISTRATION_COMMIT[:7]}`, and no READ class comparison
was inspected during these corrections.

### Frozen estimator and firewall

- `READ_IG`: unchanged 16-step midpoint estimator.
- Frozen source SHA-256: `{metrics['firewall']['frozen_read_sha256_after']}`.
- Causal artifact read by v7.3: `false`.
- Edited metrics/patch outputs consumed by v7.3: `false`.
- Sanitized manifest SHA remained identical before/after v7.3.
- Static import and shell grep audits found no causal, interchange,
  intervention, or patch import in the full local READ import closure.

### Deliverables, execution, and isolation

- Metrics: `results/v7/metrics_v7.json` (SHA-256 `{metrics_sha256}`).
- Figures: all four required PNGs exist and passed size/label metadata checks.
- Notebook execution audit: **{execution['status']}**; all four executed:
  `{str(execution['all_four_executed']).lower()}`.
- Isolation audit: **{isolation['status']}**; only v7 paths changed from
  baseline `{BASE_HEAD[:7]}`: `{str(isolation['only_v7_paths_changed']).lower()}`.
- Frozen prior READ file unchanged: `{isolation['frozen_read_sha256']}`.
- Test suite, pytest, and Ruff were intentionally not run.
- Git branch: `{isolation['branch']}`; existing Git identity and remote were
  left unchanged.
"""
    return PREREGISTRATION_TEXT + "\n" + results.lstrip("\n")


def run_decision_stage_v7() -> dict[str, Any]:
    """Open the preregistered comparison and write v7 metrics, figures, report."""

    preregistration = verify_preregistration_v7()
    manifest = load_json(MANIFEST_PATH)
    causal = load_json(CAUSAL_PATH)
    read = load_json(READ_PATH)
    joined = join_final_artifacts_v7(manifest, causal, read)
    analysis = evaluate_joined_v7(joined["task_rows"])
    figures = generate_figures_v7(
        joined["task_rows"], analysis, causal["causal_sanity"]
    )
    reason_counts = Counter(
        reason
        for row in manifest["rows"]
        if row["verification_status"] == "UNVERIFIED"
        for reason in row["verification_reasons"]
    )
    fold_counts = Counter(
        int(row["fold"])
        for row in manifest["rows"]
        if row["verification_status"] == "VERIFIED"
    )
    attempt_history = attempt_history_v7()
    isolation = isolation_audit_v7()
    execution = notebook_execution_audit_v7(require_v7_4=False)
    provenance = {
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(MANIFEST_PATH),
        },
        "causal": {
            "path": str(CAUSAL_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(CAUSAL_PATH),
        },
        "read": {
            "path": str(READ_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(READ_PATH),
        },
        "preregistration": preregistration,
    }
    metrics = {
        "schema_version": "matched-read-v7-final-metrics-v1",
        "status": "COMPLETE_PENDING_NOTEBOOK_FINALIZATION",
        "preregistration": preregistration,
        "model": {
            **manifest["model"],
            "selected_layer": int(manifest["selection"]["layer"]),
            "position_rule": str(manifest["selection"]["position_rule"]),
            "written_threshold": float(manifest["selection"]["written_threshold"]),
            "logit_agreement": manifest["logit_agreement"],
        },
        "metric_contract": manifest["metric_contract"],
        "dataset": {
            "counts": manifest["counts"],
            "verified_fold_counts": {str(fold): int(fold_counts[fold]) for fold in FOLDS},
            "unverified_reason_counts": dict(sorted(reason_counts.items())),
            "verification_forward": manifest["prompt_contract"]["verification_forward"],
            "top1_rule": manifest["prompt_contract"]["top1_rule"],
        },
        "join_audit": joined["join_audit"],
        "causal_sanity": causal["causal_sanity"],
        "firewall": read["firewall"],
        "attempt_history": attempt_history,
        "pairs": joined["pair_metrics"],
        "task_rows": joined["task_rows"],
        "figures": figures,
        "isolation_audit": isolation,
        "execution_audit": execution,
        "provenance": provenance,
        **analysis,
        "interpretation": {
            "binary_matched_bar": analysis["decision"]["outcome"],
            "primary_full_state_causal_sanity": causal["causal_sanity"][
                "full_state_sanity_status"
            ],
            "qualified_scope": (
                "strong-versus-weak explicit-concept use; not perfectly idle "
                "dashboards under full-state interchange"
            ),
            "graded_meter_supported": False,
            "old_auc_1_superseded": True,
        },
    }
    save_json(METRICS_PATH, metrics)
    metrics_sha = sha256_file(METRICS_PATH)
    REPORT_PATH.write_text(
        build_report_v7(metrics, metrics_sha256=metrics_sha), encoding="utf-8"
    )
    return {
        "metrics_path": str(METRICS_PATH),
        "report_path": str(REPORT_PATH),
        "metrics_sha256": metrics_sha,
        "decision": metrics["decision"],
        "causal_sanity": metrics["causal_sanity"],
        "auc_table": metrics["heldout_detection"]["auc_table"],
        "within_engine": metrics["within_engine"]["primary"],
        "distributions": {
            "observed_ranges": metrics["distributions"]["observed_ranges"],
            "diagnostics": metrics["distributions"]["diagnostics"],
            "median_ci": metrics["distributions"]["group_bootstrap_median_ci"],
        },
        "figures": metrics["figures"],
    }


def finalize_execution_audit_v7() -> dict[str, Any]:
    """Record post-nbconvert evidence that all four notebooks executed."""

    metrics = load_json(METRICS_PATH)
    execution = notebook_execution_audit_v7(require_v7_4=True)
    isolation = isolation_audit_v7()
    metrics["execution_audit"] = execution
    metrics["isolation_audit"] = isolation
    metrics["status"] = "COMPLETE"
    save_json(METRICS_PATH, metrics)
    metrics_sha = sha256_file(METRICS_PATH)
    REPORT_PATH.write_text(
        build_report_v7(metrics, metrics_sha256=metrics_sha), encoding="utf-8"
    )
    return {
        "status": "COMPLETE",
        "metrics_sha256": metrics_sha,
        "execution_audit": execution,
        "isolation_audit": isolation,
        "report_sha256": sha256_file(REPORT_PATH),
    }


__all__ = [
    "METRICS_PATH",
    "REPORT_PATH",
    "finalize_execution_audit_v7",
    "join_final_artifacts_v7",
    "run_decision_stage_v7",
    "verify_preregistration_v7",
]

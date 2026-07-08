"""Execute isolated v6 CHECK 1: graded READ signal within engines only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.stress_v6 import group_bootstrap_spearman, validate_record_schema


ROOT = Path("/home/jovyan/j-space-thoughts")
LEGACY_METRICS_PATH = ROOT / "results/metrics.json"
V6_DIR = ROOT / "results/v6"
RAW_DIR = V6_DIR / "raw"
FIGURE_DIR = V6_DIR / "figures"
METRICS_PATH = V6_DIR / "metrics_v6.json"
RAW_PATH = RAW_DIR / "v6_1_within_engine.json"
FIGURE_PATH = FIGURE_DIR / "F_v6_1_engine_only_read_vs_c.png"
BASELINE_HEAD = "eb9e44144de7d05d4a8e93f975d1af1351b0d87d"
SEED = 1729
N_BOOTSTRAP = 10_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


current_head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if current_head != BASELINE_HEAD:
    raise RuntimeError(
        f"CHECK 1 must begin from frozen baseline {BASELINE_HEAD}, got {current_head}"
    )

legacy_metrics = json.loads(LEGACY_METRICS_PATH.read_text())
legacy_run = legacy_metrics["symmetric_causal_read_v6"]
if legacy_run["status"] != "COMPLETE" or legacy_run["decision"] != "GO":
    raise RuntimeError("Frozen source run is not the completed GO artifact")

engine_rows = [
    row for row in legacy_run["per_task_rows"] if row["task"] == "engine"
]
schema_audit = validate_record_schema(
    engine_rows,
    required_fields={
        "pair_id",
        "dependency_group",
        "fold",
        "task",
        "C",
        "abs_C",
        "READ_IG",
        "READ_local",
        "weight_norm_baseline",
    },
    finite_fields={
        "C",
        "abs_C",
        "READ_IG",
        "READ_local",
        "weight_norm_baseline",
    },
    allowed_values={"task": {"engine"}},
    unique_by=("pair_id", "task"),
    schema_name="frozen_engine_rows",
)
if len(engine_rows) != 77:
    raise RuntimeError(f"Expected 77 frozen engines, got {len(engine_rows)}")
if len({row["dependency_group"] for row in engine_rows}) != 24:
    raise RuntimeError("Frozen engine dependency-group coverage changed")
if {int(row["fold"]) for row in engine_rows} != set(range(5)):
    raise RuntimeError("Frozen five-fold assignments changed")

score_keys = ("READ_IG", "READ_local", "weight_norm_baseline")
correlations: dict[str, dict] = {}
bootstrap_samples: dict[str, list[float]] = {}
for score_key in score_keys:
    summary, samples = group_bootstrap_spearman(
        engine_rows,
        score_key,
        target_key="abs_C",
        group_key="dependency_group",
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )
    if summary["valid_bootstrap_draws"] != N_BOOTSTRAP:
        raise RuntimeError(f"Undefined bootstrap draws for {score_key}")
    correlations[score_key] = summary
    bootstrap_samples[score_key] = [float(value) for value in samples]

abs_c = np.asarray([row["abs_C"] for row in engine_rows], dtype=np.float64)
sorted_c = np.sort(abs_c)
adjacent_gaps = np.diff(sorted_c)
largest_gap = float(np.max(adjacent_gaps))
largest_gap_index = int(np.argmax(adjacent_gaps))
within_engine_auc = {
    "status": "NOT_RUN_NO_PREREGISTERED_OR_NATURAL_SPLIT",
    "reason": (
        "All frozen engines are strongly causal and the protocol defines no "
        "weak/strong cutoff; introducing a held-out threshold now would be post-hoc."
    ),
    "abs_C_min": float(np.min(abs_c)),
    "abs_C_median": float(np.median(abs_c)),
    "abs_C_max": float(np.max(abs_c)),
    "largest_adjacent_gap": largest_gap,
    "rows_below_largest_gap": largest_gap_index + 1,
}

primary = correlations["READ_IG"]
check1_supports_graded_use = bool(primary["ci_lower_above_zero"])
interpretation = (
    "SUPPORTS_GRADED_USE"
    if check1_supports_graded_use
    else "ARTIFACT_SIDE_CI_SPANS_ZERO"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8.6, 6.3))
x = np.asarray([row["READ_IG"] for row in engine_rows], dtype=np.float64)
ax.scatter(x, abs_c, s=52, alpha=0.72, color="#2166ac", edgecolor="white", linewidth=0.4)
ax.set_xscale("log")
ax.set_xlabel("frozen READ_IG (16-step midpoint)")
ax.set_ylabel("|C| from frozen symmetric full-residual interchange")
ax.set_title(
    "F_v6_1 — engine-only graded-use check\n"
    f"Spearman rho={primary['estimate']:.3f}, group-bootstrap CI95 "
    f"[{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}]"
)
ax.text(
    0.02,
    0.04,
    "N=77 engines; 24 dependency groups\nCI spans zero: no positive graded-use evidence",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=10,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
)
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=180)
plt.close(fig)

raw_artifact = {
    "schema_version": "read-stress-v6-check1-v1",
    "baseline_head": BASELINE_HEAD,
    "source_metrics": {
        "path": str(LEGACY_METRICS_PATH),
        "sha256": sha256(LEGACY_METRICS_PATH),
    },
    "reuse_contract": {
        "causal_truth_recomputed": False,
        "read_estimators_recomputed": False,
        "folds_reassigned": False,
        "layer_reselected": False,
        "source_layer": 16,
        "ig_steps": 16,
    },
    "schema_audit": schema_audit,
    "engine_rows": engine_rows,
    "correlations": correlations,
    "bootstrap_samples": bootstrap_samples,
    "within_engine_auc": within_engine_auc,
    "interpretation": interpretation,
}
save_json(RAW_PATH, raw_artifact)

metrics_v6 = {
    "schema_version": "read-stress-v6-isolated-v1",
    "status": "CHECK_1_COMPLETE",
    "baseline": {
        "head": BASELINE_HEAD,
        "source_metrics_path": str(LEGACY_METRICS_PATH),
        "source_metrics_sha256": sha256(LEGACY_METRICS_PATH),
        "source_run_status": legacy_run["status"],
        "source_decision": legacy_run["decision"],
    },
    "frozen_protocol": {
        "model": legacy_run["protocol"]["model"],
        "source_layer": legacy_run["stage30"]["selection"]["layer"],
        "position_rule": legacy_run["stage30"]["selection"]["position_rule"],
        "ig_steps": legacy_run["stage32"]["ig_steps"],
        "folds": 5,
        "bootstrap_draws": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "bootstrap_unit": "unordered concept dependency group",
        "estimator_retuned": False,
    },
    "check1": {
        "status": "COMPLETE",
        "n_engines": len(engine_rows),
        "n_dependency_groups": len({row["dependency_group"] for row in engine_rows}),
        "correlations": correlations,
        "within_engine_auc": within_engine_auc,
        "interpretation": interpretation,
        "supports_positive_graded_use": check1_supports_graded_use,
        "figure": str(FIGURE_PATH.relative_to(ROOT)),
        "raw_artifact": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "bytes": RAW_PATH.stat().st_size,
            "sha256": sha256(RAW_PATH),
        },
    },
}
save_json(METRICS_PATH, metrics_v6)

print("CHECK 1 complete")
for score_key in score_keys:
    result = correlations[score_key]
    print(
        f"{score_key}: rho={result['estimate']:.6f}, "
        f"CI95=[{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]"
    )
print("within-engine AUC", within_engine_auc["status"])
print("interpretation", interpretation)
print("metrics", METRICS_PATH)
print("figure", FIGURE_PATH)

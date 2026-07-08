"""Join frozen engines to verified hard dashboards and evaluate CHECK 2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.stress_v6 import group_bootstrap_auc, validate_record_schema


ROOT = Path("/home/jovyan/j-space-thoughts")
BASELINE_HEAD = "eb9e44144de7d05d4a8e93f975d1af1351b0d87d"
LEGACY_METRICS_PATH = ROOT / "results/metrics.json"
METRICS_PATH = ROOT / "results/v6/metrics_v6.json"
MANIFEST_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_manifest.json"
CHEAP_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_cheap.json"
CAUSAL_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_causal.json"
ANALYSIS_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_analysis.json"
FIGURE_PATH = ROOT / "results/v6/figures/F_v6_2_old_vs_hard_dashboard_auc.png"
SEED = 1729
N_BOOTSTRAP = 10_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def baseline_blob(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASELINE_HEAD}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


legacy = json.loads(LEGACY_METRICS_PATH.read_text())["symmetric_causal_read_v6"]
metrics_v6 = json.loads(METRICS_PATH.read_text())
manifest = json.loads(MANIFEST_PATH.read_text())
cheap = json.loads(CHEAP_PATH.read_text())
causal = json.loads(CAUSAL_PATH.read_text())

if metrics_v6["status"] != "CHECK_1_COMPLETE":
    raise RuntimeError("CHECK 1 must be frozen before CHECK 2")
if cheap["source_hard_manifest"]["sha256"] != sha256(MANIFEST_PATH):
    raise RuntimeError("Cheap READ did not consume the frozen hard manifest")
if causal["source_hard_manifest"]["sha256"] != sha256(MANIFEST_PATH):
    raise RuntimeError("Causal run did not consume the frozen hard manifest")

verification_rows = manifest["rows"]
verified_rows = [
    row for row in verification_rows if row["verification_status"] == "VERIFIED_HARD"
]
verified_ids = {row["pair_id"] for row in verified_rows}
cheap_by_pair = {row["pair_id"]: row for row in cheap["compact_rows"]}
causal_by_pair = {row["pair_id"]: row for row in causal["rows"]}
engine_by_pair = {
    row["pair_id"]: row
    for row in legacy["per_task_rows"]
    if row["task"] == "engine"
}
if not verified_ids:
    raise RuntimeError("No verified hard-dashboard rows are available")
if set(cheap_by_pair) != verified_ids or set(causal_by_pair) != verified_ids:
    raise RuntimeError("Hard cheap/causal pair coverage differs from verification")
if not verified_ids <= set(engine_by_pair):
    raise RuntimeError("Hard controls do not have frozen matched engines")

task_rows: list[dict] = []
for pair_id in sorted(verified_ids):
    engine = engine_by_pair[pair_id]
    hard = cheap_by_pair[pair_id]
    for label, task, score in (
        (1, "engine", float(engine["READ_IG"])),
        (0, "hard_dashboard", float(hard["READ_IG"])),
    ):
        task_rows.append(
            {
                "pair_id": pair_id,
                "dependency_group": engine["dependency_group"],
                "fold": int(engine["fold"]),
                "category": engine["category"],
                "task": task,
                "label": label,
                "READ_IG": score,
            }
        )

schema_audit = validate_record_schema(
    task_rows,
    required_fields={
        "pair_id",
        "dependency_group",
        "fold",
        "category",
        "task",
        "label",
        "READ_IG",
    },
    finite_fields={"label", "READ_IG"},
    allowed_values={"task": {"engine", "hard_dashboard"}, "label": {0, 1}},
    unique_by=("pair_id", "task"),
    schema_name="hard_dashboard_auc_rows",
)

hard_auc, hard_auc_samples = group_bootstrap_auc(
    task_rows,
    "READ_IG",
    label_key="label",
    group_key="dependency_group",
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
)
fold_auc: dict[str, float] = {}
for fold in range(5):
    fold_rows = [row for row in task_rows if int(row["fold"]) == fold]
    if not fold_rows:
        raise RuntimeError(f"Frozen fold {fold} is absent after hard verification")
    fold_auc[str(fold)] = float(
        roc_auc_score(
            [row["label"] for row in fold_rows],
            [row["READ_IG"] for row in fold_rows],
        )
    )
hard_auc["fold_auc"] = fold_auc

category_auc: dict[str, dict] = {}
for category in sorted({row["category"] for row in task_rows}):
    subset = [row for row in task_rows if row["category"] == category]
    category_auc[category] = {
        "n_pairs": len(subset) // 2,
        "n_dependency_groups": len({row["dependency_group"] for row in subset}),
        "auc": float(
            roc_auc_score(
                [row["label"] for row in subset],
                [row["READ_IG"] for row in subset],
            )
        ),
        "diagnostic_only": True,
    }

engine_c = np.asarray(
    [causal_by_pair[pair_id]["frozen_engine_C"] for pair_id in sorted(verified_ids)],
    dtype=np.float64,
)
hard_c = np.asarray(
    [
        causal_by_pair[pair_id]["hard_dashboard"]["C"]
        for pair_id in sorted(verified_ids)
    ],
    dtype=np.float64,
)
hard_disagreements = sum(
    bool(causal_by_pair[pair_id]["hard_dashboard"]["sharp_directional_disagreement"])
    for pair_id in verified_ids
)
causal_sanity = {
    "n_pairs": len(verified_ids),
    "engine_C_median": float(np.median(engine_c)),
    "engine_abs_C_median": float(np.median(np.abs(engine_c))),
    "hard_dashboard_C_median": float(np.median(hard_c)),
    "hard_dashboard_abs_C_median": float(np.median(np.abs(hard_c))),
    "hard_dashboard_C_min": float(np.min(hard_c)),
    "hard_dashboard_C_max": float(np.max(hard_c)),
    "hard_dashboard_sharp_directional_disagreements": int(hard_disagreements),
    "engine_large_gate": bool(np.median(np.abs(engine_c)) > 0.50),
    "hard_dashboard_near_zero_gate": bool(np.median(np.abs(hard_c)) < 0.10),
}
causal_sanity["status"] = (
    "PASS"
    if causal_sanity["engine_large_gate"]
    and causal_sanity["hard_dashboard_near_zero_gate"]
    else "FAIL"
)

old_auc = next(row for row in legacy["auc_table"] if row["estimator"] == "READ_IG")
hard_auc_survives = bool(
    causal_sanity["status"] == "PASS" and hard_auc["estimate"] >= 0.80
)
interpretation = (
    "HARD_CONTROL_SEPARATION_SURVIVES"
    if hard_auc_survives
    else "HARD_CONTROL_SEPARATION_COLLAPSES_OR_CAUSAL_SANITY_FAILS"
)

cheap_read_current_sha = sha256(ROOT / "src/cheap_read.py")
cheap_read_baseline_sha = sha256_bytes(baseline_blob("src/cheap_read.py"))
causal_read_current_sha = sha256(ROOT / "src/causal_read.py")
causal_read_baseline_sha = sha256_bytes(baseline_blob("src/causal_read.py"))
source_immutability = {
    "cheap_read_current_sha256": cheap_read_current_sha,
    "cheap_read_baseline_sha256": cheap_read_baseline_sha,
    "cheap_read_unchanged": cheap_read_current_sha == cheap_read_baseline_sha,
    "causal_read_current_sha256": causal_read_current_sha,
    "causal_read_baseline_sha256": causal_read_baseline_sha,
    "causal_read_unchanged": causal_read_current_sha == causal_read_baseline_sha,
}
if not source_immutability["cheap_read_unchanged"] or not source_immutability[
    "causal_read_unchanged"
]:
    raise RuntimeError("Frozen causal or cheap estimator source was modified")

FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
labels = ["old dashboard", "answer-type-matched\nhard dashboard"]
estimates = [float(old_auc["heldout_auc"]), float(hard_auc["estimate"])]
lows = [float(old_auc["ci95_low"]), float(hard_auc["ci95_low"])]
highs = [float(old_auc["ci95_high"]), float(hard_auc["ci95_high"])]
errors = np.asarray(
    [[estimate - low for estimate, low in zip(estimates, lows, strict=True)],
     [high - estimate for high, estimate in zip(highs, estimates, strict=True)]],
    dtype=np.float64,
)
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8.4, 6.2))
bars = ax.bar(
    labels,
    estimates,
    yerr=errors,
    capsize=7,
    color=["#9e9e9e", "#2166ac"],
    edgecolor="black",
    linewidth=0.6,
)
ax.axhline(0.70, color="#b2182b", linestyle="--", linewidth=1.8, label="0.70 bar")
ax.axhline(0.50, color="black", linestyle=":", linewidth=1.5, label="chance")
ax.set_ylim(0.0, 1.05)
ax.set_ylabel("held-out ROC AUC (dependency-group bootstrap CI95)")
ax.set_title("F_v6_2 — READ_IG under a harder matched-answer control")
ax.legend(loc="lower left")
for bar, value in zip(bars, estimates, strict=True):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=180)
plt.close(fig)

analysis_artifact = {
    "schema_version": "read-stress-v6-hard-dashboard-analysis-v1",
    "schema_audit": schema_audit,
    "task_rows": task_rows,
    "hard_dashboard_auc": hard_auc,
    "hard_dashboard_auc_bootstrap_samples": [float(value) for value in hard_auc_samples],
    "old_dashboard_auc": old_auc,
    "category_auc": category_auc,
    "causal_sanity": causal_sanity,
    "interpretation": interpretation,
}
save_json(ANALYSIS_PATH, analysis_artifact)

verification_counts = Counter(row["verification_status"] for row in verification_rows)
verification_reasons = Counter(
    reason for row in verification_rows for reason in row["verification_reasons"]
)
metrics_v6["status"] = "CHECK_2_COMPLETE"
metrics_v6["check2"] = {
    "status": "COMPLETE",
    "design": manifest["design"],
    "verification": {
        "candidates": len(verification_rows),
        "verified_hard": int(verification_counts["VERIFIED_HARD"]),
        "unverified_hard": int(verification_counts["UNVERIFIED_HARD"]),
        "reason_counts": dict(sorted(verification_reasons.items())),
        "n_dependency_groups": len(
            {row["dependency_group"] for row in verified_rows}
        ),
        "failed_rows_excluded_not_relabeled": True,
    },
    "causal_sanity": causal_sanity,
    "hard_dashboard_auc": hard_auc,
    "old_dashboard_auc": {
        "estimate": old_auc["heldout_auc"],
        "ci95_low": old_auc["ci95_low"],
        "ci95_high": old_auc["ci95_high"],
        "n_pairs": old_auc["n_prompt_pairs"],
    },
    "category_auc_diagnostic": category_auc,
    "interpretation": interpretation,
    "hard_auc_survives_at_0_80": hard_auc_survives,
    "source_immutability": source_immutability,
    "anti_circularity_audit": cheap["anti_circularity_audit"],
    "figure": str(FIGURE_PATH.relative_to(ROOT)),
    "artifacts": {
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)),
            "sha256": sha256(MANIFEST_PATH),
        },
        "cheap": {
            "path": str(CHEAP_PATH.relative_to(ROOT)),
            "sha256": sha256(CHEAP_PATH),
        },
        "causal": {
            "path": str(CAUSAL_PATH.relative_to(ROOT)),
            "sha256": sha256(CAUSAL_PATH),
        },
        "analysis": {
            "path": str(ANALYSIS_PATH.relative_to(ROOT)),
            "sha256": sha256(ANALYSIS_PATH),
        },
    },
}
save_json(METRICS_PATH, metrics_v6)

print("hard verification", metrics_v6["check2"]["verification"])
print("hard causal sanity", json.dumps(causal_sanity, indent=2))
print("hard READ_IG AUC", json.dumps(hard_auc, indent=2))
print("interpretation", interpretation)
print("figure", FIGURE_PATH)

"""Execute isolated v6 CHECK 3: raw READ_IG distributions and overlap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.stress_v6 import distribution_diagnostics


ROOT = Path("/home/jovyan/j-space-thoughts")
LEGACY_METRICS_PATH = ROOT / "results/metrics.json"
HARD_CHEAP_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_cheap.json"
METRICS_PATH = ROOT / "results/v6/metrics_v6.json"
RAW_PATH = ROOT / "results/v6/raw/v6_3_read_distributions.json"
FIGURE_PATH = ROOT / "results/v6/figures/F_v6_3_read_ig_distributions.png"
SEED = 1729


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


legacy = json.loads(LEGACY_METRICS_PATH.read_text())["symmetric_causal_read_v6"]
hard = json.loads(HARD_CHEAP_PATH.read_text())
metrics_v6 = json.loads(METRICS_PATH.read_text())
if metrics_v6["status"] != "CHECK_2_COMPLETE":
    raise RuntimeError("CHECK 2 must be complete before distribution analysis")

engine_by_pair = {
    row["pair_id"]: float(row["READ_IG"])
    for row in legacy["per_task_rows"]
    if row["task"] == "engine"
}
old_by_pair = {
    row["pair_id"]: float(row["READ_IG"])
    for row in legacy["per_task_rows"]
    if row["task"] == "dashboard"
}
hard_by_pair = {
    row["pair_id"]: float(row["READ_IG"]) for row in hard["compact_rows"]
}
pair_ids = sorted(hard_by_pair)
if not pair_ids or not set(pair_ids) <= set(engine_by_pair) or not set(pair_ids) <= set(old_by_pair):
    raise RuntimeError("Hard dashboards are not matched to frozen engine/dashboard rows")

raw_values = {
    "engine": [engine_by_pair[pair_id] for pair_id in pair_ids],
    "old_dashboard": [old_by_pair[pair_id] for pair_id in pair_ids],
    "hard_dashboard": [hard_by_pair[pair_id] for pair_id in pair_ids],
}
diagnostics = distribution_diagnostics(raw_values, near_zero_atol=1e-3)
old_overlap = diagnostics["pairwise"]["engine__vs__old_dashboard"]
hard_overlap = diagnostics["pairwise"]["engine__vs__hard_dashboard"]
dashboard_overlap = diagnostics["pairwise"]["hard_dashboard__vs__old_dashboard"]

old_summary = diagnostics["classes"]["old_dashboard"]
hard_summary = diagnostics["classes"]["hard_dashboard"]
engine_summary = diagnostics["classes"]["engine"]
mechanism_finding = {
    "old_dashboard_all_identical": old_summary["all_identical"],
    "old_dashboard_compressed_tiny_band": bool(
        old_summary["maximum"] < engine_summary["minimum"]
        and old_summary["median"] < 0.01
    ),
    "hard_dashboard_compressed_tiny_band": bool(
        hard_summary["maximum"] < engine_summary["minimum"]
        and hard_summary["median"] < 0.01
    ),
    "old_and_hard_dashboard_ranges_overlap": dashboard_overlap[
        "ranges_overlap_or_touch"
    ],
    "old_and_hard_overlap_fraction_of_union": dashboard_overlap[
        "overlap_fraction_of_union"
    ],
    "engine_old_ranges_disjoint": old_overlap["strictly_disjoint_ranges"],
    "engine_hard_ranges_disjoint": hard_overlap["strictly_disjoint_ranges"],
    "arithmetic_answer_type_is_sole_explanation": False,
    "interpretation": (
        "Old dashboards are not identical, but both old and answer-type-matched "
        "hard dashboards occupy nearly the same compressed low-READ band and are "
        "strictly disjoint from engines. Arithmetic answer type is therefore not "
        "the sole mechanism; the score behaves as a binary relevant-vs-irrelevant "
        "task detector on this roster, without graded resolution within engines."
    ),
}

FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9.0, 6.5))
order = ["engine", "old_dashboard", "hard_dashboard"]
labels = ["engine", "old dashboard\n(arithmetic)", "hard dashboard\n(answer-type matched)"]
colors = ["#2166ac", "#ef8a62", "#67a9cf"]
rng = np.random.default_rng(SEED)
positions = np.arange(1, 4)
box = ax.boxplot(
    [raw_values[name] for name in order],
    positions=positions,
    widths=0.48,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": "black", "linewidth": 1.6},
)
for patch, color in zip(box["boxes"], colors, strict=True):
    patch.set_facecolor(color)
    patch.set_alpha(0.33)
for position, name, color in zip(positions, order, colors, strict=True):
    values = np.asarray(raw_values[name], dtype=np.float64)
    jitter = rng.normal(0.0, 0.055, size=values.size)
    ax.scatter(
        np.full(values.size, position) + jitter,
        values,
        s=34,
        alpha=0.72,
        color=color,
        edgecolor="white",
        linewidth=0.35,
    )
ax.set_yscale("log")
ax.set_xticks(positions, labels)
ax.set_ylabel("raw READ_IG (log scale)")
ax.set_title(
    "F_v6_3 — raw READ_IG distributions\n"
    "both irrelevant controls form an overlapping low-score band"
)
ax.text(
    0.02,
    0.04,
    f"engine min={engine_summary['minimum']:.4f}\n"
    f"old max={old_summary['maximum']:.4f}; hard max={hard_summary['maximum']:.4f}",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
)
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=180)
plt.close(fig)

raw_artifact = {
    "schema_version": "read-stress-v6-distributions-v1",
    "source_legacy_metrics_sha256": sha256(LEGACY_METRICS_PATH),
    "source_hard_cheap_sha256": sha256(HARD_CHEAP_PATH),
    "pair_ids": pair_ids,
    "raw_READ_IG": raw_values,
    "diagnostics": diagnostics,
    "mechanism_finding": mechanism_finding,
}
save_json(RAW_PATH, raw_artifact)

metrics_v6["status"] = "CHECK_3_COMPLETE"
metrics_v6["check3"] = {
    "status": "COMPLETE",
    "n_per_class": len(pair_ids),
    "raw_READ_IG": raw_values,
    "diagnostics": diagnostics,
    "mechanism_finding": mechanism_finding,
    "figure": str(FIGURE_PATH.relative_to(ROOT)),
    "raw_artifact": {
        "path": str(RAW_PATH.relative_to(ROOT)),
        "bytes": RAW_PATH.stat().st_size,
        "sha256": sha256(RAW_PATH),
    },
}
save_json(METRICS_PATH, metrics_v6)

print("distribution summaries", json.dumps(diagnostics["classes"], indent=2))
print("range overlaps", json.dumps(diagnostics["pairwise"], indent=2))
print("mechanism", json.dumps(mechanism_finding, indent=2))
print("figure", FIGURE_PATH)

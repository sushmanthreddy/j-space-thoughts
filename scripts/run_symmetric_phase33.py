"""Notebook 33 driver: held-out trust check and GO/NO-GO decision."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.metrics import save_json


ROOT = Path("/home/jovyan/j-space-thoughts")
RAW_DIR = ROOT / "data/raw/v6"
FIGURE_DIR = ROOT / "results/figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = ROOT / "results/metrics.json"
RESULTS_PATH = ROOT / "results/RESULTS.md"
VERIFY_PATH = RAW_DIR / "30_dataset_and_verification.json"
CAUSAL_PATH = RAW_DIR / "31_causal_ground_truth.json"
CHEAP_PATH = RAW_DIR / "32_cheap_read.json"
SEED = 1729
N_BOOTSTRAP = 10_000


verification = json.loads(VERIFY_PATH.read_text())
causal = json.loads(CAUSAL_PATH.read_text())
cheap = json.loads(CHEAP_PATH.read_text())
protocol_sha = verification["protocol_sha256"]
if causal["protocol_sha256"] != protocol_sha or cheap["protocol_sha256"] != protocol_sha:
    raise RuntimeError("Phase artifacts do not share one frozen protocol")
if cheap["anti_circularity_audit"]["status"] != "PASS":
    raise RuntimeError("Cheap READ anti-circularity audit did not pass")

causal_by_pair = {row["pair_id"]: row for row in causal["rows"]}
cheap_by_pair = {row["pair_id"]: row for row in cheap["rows"]}
if set(causal_by_pair) != set(cheap_by_pair):
    raise RuntimeError("Causal and cheap estimator pair coverage differs")

task_rows = []
for pair_id in sorted(causal_by_pair):
    causal_pair = causal_by_pair[pair_id]
    cheap_pair = cheap_by_pair[pair_id]
    for task, label in (("engine", 1), ("dashboard", 0)):
        truth = causal_pair[task]["full_residual"]
        subspace = causal_pair[task]["jlens_two_concept_subspace"]
        estimate = cheap_pair[task]
        task_rows.append(
            {
                "pair_id": pair_id,
                "dependency_group": causal_pair["dependency_group"],
                "fold": int(causal_pair["fold"]),
                "category": causal_pair["category"],
                "concept_a": causal_pair["concept_a"],
                "concept_b": causal_pair["concept_b"],
                "task": task,
                "label": label,
                "C": float(truth["C"]),
                "abs_C": abs(float(truth["C"])),
                "R_a_from_b": float(truth["R_a_from_b"]),
                "R_b_from_a": float(truth["R_b_from_a"]),
                "T": float(truth["T"]),
                "directional_abs_difference": float(
                    truth["directional_abs_difference"]
                ),
                "sharp_directional_disagreement": bool(
                    truth["sharp_directional_disagreement"]
                ),
                "C_subspace": float(subspace["C"]),
                "R_a_from_b_subspace": float(subspace["R_a_from_b"]),
                "R_b_from_a_subspace": float(subspace["R_b_from_a"]),
                "READ_IG": float(estimate["READ_IG"]),
                "READ_local": float(estimate["READ_local"]),
                "weight_norm_baseline": float(
                    cheap_pair["weight_norm_capacity_baseline"][
                        "weight_norm_baseline"
                    ]
                ),
                "baseline_label": cheap_pair["weight_norm_capacity_baseline"][
                    "baseline"
                ],
            }
        )

score_keys = ["READ_IG", "READ_local", "weight_norm_baseline"]


def group_bootstrap_auc(score_key: str) -> tuple[dict, list[float]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in task_rows:
        by_group[row["dependency_group"]].append(row)
    group_names = sorted(by_group)
    labels = np.asarray([row["label"] for row in task_rows], dtype=int)
    scores = np.asarray([row[score_key] for row in task_rows], dtype=float)
    estimate = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(N_BOOTSTRAP):
        selected = rng.choice(group_names, size=len(group_names), replace=True)
        sampled_rows = [row for group in selected for row in by_group[str(group)]]
        sampled_labels = [row["label"] for row in sampled_rows]
        sampled_scores = [row[score_key] for row in sampled_rows]
        samples.append(float(roc_auc_score(sampled_labels, sampled_scores)))
    ci_low, ci_high = np.quantile(np.asarray(samples), [0.025, 0.975])
    fold_auc = {}
    for fold in range(5):
        fold_rows = [row for row in task_rows if row["fold"] == fold]
        fold_auc[str(fold)] = float(
            roc_auc_score(
                [row["label"] for row in fold_rows],
                [row[score_key] for row in fold_rows],
            )
        )
    return (
        {
            "estimator": score_key,
            "heldout_auc": estimate,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "bootstrap_draws": N_BOOTSTRAP,
            "bootstrap_seed": SEED,
            "bootstrap_unit": "unordered concept dependency group",
            "n_dependency_groups": len(group_names),
            "n_prompt_pairs": len(task_rows) // 2,
            "fold_auc": fold_auc,
            "passes_numeric_bar": estimate >= 0.70 and float(ci_low) > 0.50,
        },
        samples,
    )


auc_table = []
bootstrap_samples = {}
for score_key in score_keys:
    record, samples = group_bootstrap_auc(score_key)
    values = [row[score_key] for row in task_rows]
    causal_values = [row["abs_C"] for row in task_rows]
    correlation = spearmanr(values, causal_values)
    record["spearman_rho_with_abs_C"] = float(correlation.statistic)
    record["spearman_p_value_descriptive"] = float(correlation.pvalue)
    record["eligible_to_trigger_go"] = score_key == "READ_IG"
    auc_table.append(record)
    bootstrap_samples[score_key] = samples

primary = next(row for row in auc_table if row["estimator"] == "READ_IG")
decision = "GO" if primary["passes_numeric_bar"] else "NO-GO"
if decision == "GO":
    decision_one_line = (
        "GO: READ_IG predicts causal use on held-out Qwen2.5-7B concepts "
        f"(AUC={primary['heldout_auc']:.3f}, 95% CI "
        f"[{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}])."
    )
else:
    decision_one_line = (
        "NO-GO: on Qwen2.5-7B, gradient READ_IG does not clear the "
        "pre-registered held-out causal-use bar "
        f"(AUC={primary['heldout_auc']:.3f}, 95% CI "
        f"[{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}])."
    )

engine_rows = [row for row in task_rows if row["task"] == "engine"]
dashboard_rows = [row for row in task_rows if row["task"] == "dashboard"]
sanity = {
    "engine_C_median": float(np.median([row["C"] for row in engine_rows])),
    "engine_abs_C_median": float(np.median([row["abs_C"] for row in engine_rows])),
    "dashboard_C_median": float(np.median([row["C"] for row in dashboard_rows])),
    "dashboard_abs_C_median": float(
        np.median([row["abs_C"] for row in dashboard_rows])
    ),
    "engine_sharp_directional_disagreement": sum(
        row["sharp_directional_disagreement"] for row in engine_rows
    ),
    "dashboard_sharp_directional_disagreement": sum(
        row["sharp_directional_disagreement"] for row in dashboard_rows
    ),
}

plt.style.use("seaborn-v0_8-whitegrid")
rng = np.random.default_rng(SEED)

# F1: engine versus dashboard signed C.
fig, ax = plt.subplots(figsize=(8.0, 5.8))
values = [[row["C"] for row in engine_rows], [row["C"] for row in dashboard_rows]]
ax.boxplot(values, tick_labels=["engine", "dashboard"], showfliers=False)
for position_index, group_values in enumerate(values, start=1):
    jitter = rng.normal(0.0, 0.045, size=len(group_values))
    ax.scatter(
        np.full(len(group_values), position_index) + jitter,
        group_values,
        alpha=0.55,
        s=20,
    )
ax.axhline(0.0, color="black", linewidth=1)
ax.set_ylabel("signed full-residual C (unclipped)")
ax.set_title(
    "F1 — task-matched causal sanity\n"
    f"median C: engine={sanity['engine_C_median']:.3f}, "
    f"dashboard={sanity['dashboard_C_median']:.3f}"
)
f1_path = FIGURE_DIR / "f1_symmetric_engine_dashboard_c.png"
fig.tight_layout()
fig.savefig(f1_path, dpi=180)
plt.close(fig)

# F2: primary READ versus absolute causal truth.
fig, ax = plt.subplots(figsize=(8.0, 6.0))
for task, color in (("engine", "#1565C0"), ("dashboard", "#EF6C00")):
    subset = [row for row in task_rows if row["task"] == task]
    ax.scatter(
        [row["READ_IG"] for row in subset],
        [row["abs_C"] for row in subset],
        label=task,
        alpha=0.65,
        s=28,
        color=color,
    )
ax.set_xscale("symlog", linthresh=1e-3)
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_xlabel("READ_IG (16-step midpoint, symmetric mean absolute)")
ax.set_ylabel("|C| from full-residual symmetric interchange")
ax.set_title(
    "F2 — cheap READ versus causal truth\n"
    f"held-out AUC={primary['heldout_auc']:.3f} "
    f"CI95 [{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}]"
)
ax.legend()
f2_path = FIGURE_DIR / "f2_read_ig_vs_c.png"
fig.tight_layout()
fig.savefig(f2_path, dpi=180)
plt.close(fig)

# F3: held-out AUC comparison.
fig, ax = plt.subplots(figsize=(8.0, 5.8))
labels = ["READ_IG", "READ_local", "weight-norm\n(broken baseline)"]
centers = np.asarray([row["heldout_auc"] for row in auc_table])
lower = centers - np.asarray([row["ci95_low"] for row in auc_table])
upper = np.asarray([row["ci95_high"] for row in auc_table]) - centers
ax.bar(labels, centers, color=["#1565C0", "#00897B", "#9E9E9E"])
ax.errorbar(
    np.arange(len(labels)),
    centers,
    yerr=np.vstack([lower, upper]),
    fmt="none",
    ecolor="black",
    capsize=5,
)
ax.axhline(0.70, color="#B71C1C", linestyle="--", label="pre-registered AUC bar")
ax.axhline(0.50, color="black", linestyle=":", label="chance")
ax.set_ylim(0.0, 1.02)
ax.set_ylabel("held-out ROC AUC (group-bootstrap CI95)")
ax.set_title("F3 — cheap estimators versus known-broken capacity baseline")
ax.legend(loc="lower right")
f3_path = FIGURE_DIR / "f3_symmetric_auc_comparison.png"
fig.tight_layout()
fig.savefig(f3_path, dpi=180)
plt.close(fig)

# F4: both-direction agreement.
fig, ax = plt.subplots(figsize=(7.2, 6.4))
for task, color in (("engine", "#1565C0"), ("dashboard", "#EF6C00")):
    subset = [row for row in task_rows if row["task"] == task]
    ax.scatter(
        [row["R_a_from_b"] for row in subset],
        [row["R_b_from_a"] for row in subset],
        label=task,
        alpha=0.6,
        s=25,
        color=color,
    )
all_r = [row[key] for row in task_rows for key in ("R_a_from_b", "R_b_from_a")]
low, high = np.quantile(all_r, [0.01, 0.99])
padding = max(0.05, 0.08 * (high - low))
ax.plot([low - padding, high + padding], [low - padding, high + padding], "k--")
ax.set_xlim(low - padding, high + padding)
ax.set_ylim(low - padding, high + padding)
ax.set_xlabel("R_A<-B")
ax.set_ylabel("R_B<-A")
ax.set_title("F4 — signed directional agreement of causal C")
ax.legend()
f4_path = FIGURE_DIR / "f4_symmetric_direction_agreement.png"
fig.tight_layout()
fig.savefig(f4_path, dpi=180)
plt.close(fig)

failure_reasons = Counter(
    reason
    for row in verification["rows"]
    if row["verification_status"] == "UNVERIFIED"
    for reason in row["verification_reasons"]
)
verification_counts = verification["counts"]
auc_markdown = "\n".join(
    "| {name} | {auc:.3f} | [{low:.3f}, {high:.3f}] | {rho:.3f} | {go} |".format(
        name=row["estimator"],
        auc=row["heldout_auc"],
        low=row["ci95_low"],
        high=row["ci95_high"],
        rho=row["spearman_rho_with_abs_C"],
        go="YES" if row["eligible_to_trigger_go"] and row["passes_numeric_bar"] else "NO",
    )
    for row in auc_table
)
new_status = f"""## Dataset verification

- Candidates: {verification_counts['candidates']} distinct matched prompt pairs in
  {verification['candidate_manifest']['n_dependency_groups']} unordered concept groups.
- Calibration-only: {verification_counts['calibration_pairs']} prompt pairs; held-out
  evaluation: {verification_counts['evaluation_pairs']}.
- **VERIFIED: {verification_counts['verified_pairs']}**; **UNVERIFIED:
  {verification_counts['unverified_pairs']}**. The target was at least 60 verified
  pairs. Failures remain logged and excluded; reason counts are
  `{dict(sorted(failure_reasons.items()))}`.
- Frozen layer/position: L{verification['selection']['layer']} at final prompt token;
  WRITTEN threshold `{verification['selection']['written_threshold']:.6f}`.

## Engine-vs-dashboard causal sanity

- Engine median signed C `{sanity['engine_C_median']:.4f}`; median |C|
  `{sanity['engine_abs_C_median']:.4f}`.
- Dashboard median signed C `{sanity['dashboard_C_median']:.4f}`; median |C|
  `{sanity['dashboard_abs_C_median']:.4f}`.
- Sharp directional-disagreement flags: engine
  {sanity['engine_sharp_directional_disagreement']}/{len(engine_rows)}, dashboard
  {sanity['dashboard_sharp_directional_disagreement']}/{len(dashboard_rows)}.
- Full-residual C is primary. The two-concept J-Lens-subspace variant is retained
  per pair as a diagnostic and never substitutes for primary truth.

![F1](figures/{f1_path.name})

![F4](figures/{f4_path.name})

## Held-out trust check

All AUCs are pooled out-of-fold on held-out concept groups. CIs resample entire
unordered concept dependency groups and retain their repeated contexts and paired
engine/dashboard tasks.

| estimator | held-out AUC | group-bootstrap 95% CI | Spearman rho vs |C| | GO trigger |
| --- | ---: | --- | ---: | --- |
{auc_markdown}

![F2](figures/{f2_path.name})

![F3](figures/{f3_path.name})

## DECISION

**{decision_one_line}**
"""
report = RESULTS_PATH.read_text()
start_marker = "## New-run status"
archive_marker = "\n---\n\n# Prior READ Go/No-Go validation"
start = report.index(start_marker)
end = report.index(archive_marker, start)
report = report[:start] + new_status.rstrip() + "\n" + report[end:]
RESULTS_PATH.write_text(report, encoding="utf-8")

figures = [str(path.relative_to(ROOT)) for path in (f1_path, f2_path, f3_path, f4_path)]
raw_artifact = {
    "schema_version": "symmetric-trust-check-v1",
    "protocol_sha256": protocol_sha,
    "upstream": {
        "verification_sha256": hashlib.sha256(VERIFY_PATH.read_bytes()).hexdigest(),
        "causal_sha256": hashlib.sha256(CAUSAL_PATH.read_bytes()).hexdigest(),
        "cheap_sha256": hashlib.sha256(CHEAP_PATH.read_bytes()).hexdigest(),
    },
    "verification_counts": verification_counts,
    "sanity": sanity,
    "auc_table": auc_table,
    "bootstrap_auc_samples": bootstrap_samples,
    "task_rows": task_rows,
    "decision": decision,
    "decision_one_line": decision_one_line,
    "figures": figures,
}
raw_path = RAW_DIR / "33_trust_check.json"
save_json(raw_path, raw_artifact)
raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

metrics = json.loads(METRICS_PATH.read_text())
run = metrics["symmetric_causal_read_v6"]
if run["protocol_sha256"] != protocol_sha:
    raise RuntimeError("Metrics protocol changed before decision")
run["status"] = "TRUST_CHECK_COMPLETE"
run["decision"] = decision
run["decision_one_line"] = decision_one_line
run["per_task_rows"] = task_rows
run["auc_table"] = auc_table
run["stage33"] = {
    "status": "COMPLETE",
    "verification_counts": verification_counts,
    "sanity": sanity,
    "auc_table": auc_table,
    "decision": decision,
    "decision_one_line": decision_one_line,
    "figures": figures,
    "raw_artifact": {
        "path": str(raw_path),
        "bytes": raw_path.stat().st_size,
        "sha256": raw_sha,
    },
}
save_json(METRICS_PATH, metrics)
print(json.dumps({"decision": decision, "auc_table": auc_table}, indent=2))
print(decision_one_line)

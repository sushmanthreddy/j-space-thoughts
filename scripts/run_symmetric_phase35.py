"""Notebook 35 driver: final figures/report and completion audit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import nbformat

from src.metrics import save_json


ROOT = Path("/home/jovyan/j-space-thoughts")
METRICS_PATH = ROOT / "results/metrics.json"
RESULTS_PATH = ROOT / "results/RESULTS.md"
metrics = json.loads(METRICS_PATH.read_text())
run = metrics["symmetric_causal_read_v6"]

required_stage_statuses = {
    "stage30": "COMPLETE",
    "stage31": "COMPLETE",
    "stage32": "COMPLETE",
    "stage33": "COMPLETE",
}
stage_checks = {
    stage: run.get(stage, {}).get("status") == expected
    for stage, expected in required_stage_statuses.items()
}
stage34_status = run.get("stage34", {}).get("status")
phase4_check = (
    stage34_status == "COMPLETE"
    if run["decision"] == "GO"
    else stage34_status == "SKIPPED_PREREQUISITE"
)

required_figures = [
    ROOT / "results/figures/f1_symmetric_engine_dashboard_c.png",
    ROOT / "results/figures/f2_read_ig_vs_c.png",
    ROOT / "results/figures/f3_symmetric_auc_comparison.png",
    ROOT / "results/figures/f4_symmetric_direction_agreement.png",
]
if run["decision"] == "GO":
    required_figures.append(
        ROOT / "results/figures/f5_signed_mediation_faithfulness.png"
    )
figure_checks = {
    str(path.relative_to(ROOT)): path.is_file() and path.stat().st_size > 0
    for path in required_figures
}

cheap_tree = ast.parse((ROOT / "src/cheap_read.py").read_text())
cheap_imports = []
for node in ast.walk(cheap_tree):
    if isinstance(node, ast.Import):
        cheap_imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        cheap_imports.append(node.module)
forbidden = ("src.causal_read", "src.interventions", "src.read_scores")
anti_circularity_check = not any(
    name.startswith(prefix) for name in cheap_imports for prefix in forbidden
)

verification_rows = run["stage30"]["verification_rows"]
per_task_rows = run["per_task_rows"]
verified_ids = {
    row["pair_id"]
    for row in verification_rows
    if row["verification_status"] == "VERIFIED"
}
metric_ids = {row["pair_id"] for row in per_task_rows}
per_pair_coverage_check = verified_ids == metric_ids
per_task_schema_check = all(
    {
        "C",
        "R_a_from_b",
        "R_b_from_a",
        "T",
        "READ_IG",
        "READ_local",
        "weight_norm_baseline",
    }.issubset(row)
    for row in per_task_rows
)
unverified_exclusion_check = all(
    row["pair_id"] in verified_ids for row in per_task_rows
)
signed_unclipped_check = run["stage31"]["signed_unclipped"] is True
auc_check = all(
    row["heldout_auc"] is not None
    and row["ci95_low"] is not None
    and row["ci95_high"] is not None
    for row in run["auc_table"]
)
decision_check = run["decision"] in {"GO", "NO-GO"} and bool(
    run["decision_one_line"]
)
report_text = RESULTS_PATH.read_text()
report_order = [
    "## Preflight",
    "## Pre-registered trust bar",
    "## Dataset verification",
    "## Engine-vs-dashboard causal sanity",
    "## Held-out trust check",
    "## DECISION",
]
report_order_check = all(
    report_text.index(left) < report_text.index(right)
    for left, right in zip(report_order, report_order[1:], strict=True)
)

notebook_checks = {}
for number, name in (
    (30, "dataset_and_verification"),
    (31, "causal_ground_truth"),
    (32, "cheap_read"),
    (33, "trust_check"),
    (34, "localization"),
):
    path = ROOT / f"notebooks/{number}_{name}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    notebook_checks[str(path.relative_to(ROOT))] = bool(code_cells) and all(
        cell.execution_count is not None for cell in code_cells
    )

audit = {
    "stages_complete": stage_checks,
    "phase4_conditional_status_correct": phase4_check,
    "required_figures_present": figure_checks,
    "cheap_read_import_isolation": anti_circularity_check,
    "verified_pair_metric_coverage_exact": per_pair_coverage_check,
    "per_task_metric_schema_complete": per_task_schema_check,
    "unverified_pairs_excluded": unverified_exclusion_check,
    "causal_C_signed_unclipped": signed_unclipped_check,
    "heldout_auc_and_ci_complete": auc_check,
    "decision_present": decision_check,
    "results_section_order": report_order_check,
    "prior_notebooks_executed": notebook_checks,
    "test_suite_run": False,
    "ruff_run": False,
    "pytest_run": False,
}


def all_true(value):
    if isinstance(value, dict):
        return all(all_true(item) for item in value.values())
    return bool(value)


if not all_true(audit):
    raise RuntimeError(f"Final completion audit failed: {json.dumps(audit, indent=2)}")
run["status"] = "COMPLETE"
run["stage35"] = {
    "status": "COMPLETE",
    "audit": audit,
    "results_sha256": hashlib.sha256(RESULTS_PATH.read_bytes()).hexdigest(),
}
save_json(METRICS_PATH, metrics)
print(json.dumps(audit, indent=2))
print(run["decision_one_line"])

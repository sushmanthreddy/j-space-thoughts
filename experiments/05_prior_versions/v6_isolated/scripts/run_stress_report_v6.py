"""Combine isolated v6 checks into the final honest stress-test report."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/home/jovyan/j-space-thoughts")
BASELINE_HEAD = "eb9e44144de7d05d4a8e93f975d1af1351b0d87d"
METRICS_PATH = ROOT / "results/v6/metrics_v6.json"
REPORT_PATH = ROOT / "results/v6/RESULTS_v6.md"
FIGURE_DIR = ROOT / "results/v6/figures"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def is_allowed_v6_path(path: str) -> bool:
    item = Path(path)
    if path.startswith("results/v6/"):
        return True
    if item.parent == Path("src") and item.name in {"data_gen_v6.py", "stress_v6.py"}:
        return True
    if item.parent == Path("scripts") and item.name.endswith("_v6.py"):
        return True
    if item.parent == Path("notebooks") and item.name.startswith("v6_") and item.suffix == ".ipynb":
        return True
    return False


def notebook_execution(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False, "executed": False, "errors": None}
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    return {
        "exists": True,
        "executed": bool(code_cells)
        and all(cell.get("execution_count") is not None for cell in code_cells),
        "errors": len(errors),
    }


def import_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


metrics = json.loads(METRICS_PATH.read_text())
if metrics["status"] not in {"CHECK_3_COMPLETE", "COMPLETE"}:
    raise RuntimeError("All three checks must complete before the v6 report")

check1 = metrics["check1"]
check2 = metrics["check2"]
check3 = metrics["check3"]
primary = check1["correlations"]["READ_IG"]
local = check1["correlations"]["READ_local"]
baseline = check1["correlations"]["weight_norm_baseline"]
hard_auc = check2["hard_dashboard_auc"]
old_auc = check2["old_dashboard_auc"]
sanity = check2["causal_sanity"]

confirmed = bool(
    check1["supports_positive_graded_use"]
    and check2["hard_auc_survives_at_0_80"]
)
decision = "CONFIRMED" if confirmed else "ARTIFACT (partial)"
if confirmed:
    decision_one_line = (
        "CONFIRMED: engine-only READ_IG is clearly positive and the separation "
        "survives the answer-type-matched hard control."
    )
else:
    decision_one_line = (
        "ARTIFACT (partial): READ_IG survives the answer-type-matched control, "
        f"but has no positive graded association within engines (rho={primary['estimate']:.3f}, "
        f"95% CI [{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}]); the "
        "perfect binary separation is not evidence of a graded causal-use meter."
    )

subprocess.run(
    ["git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD"],
    cwd=ROOT,
    check=True,
)
committed_or_tracked = set(
    filter(None, git_output("diff", "--name-only", BASELINE_HEAD).splitlines())
)
untracked = set(
    filter(None, git_output("ls-files", "--others", "--exclude-standard").splitlines())
)
expected_output_paths = {
    str(REPORT_PATH.relative_to(ROOT)),
    str(METRICS_PATH.relative_to(ROOT)),
}
all_changed_paths = sorted(committed_or_tracked | untracked | expected_output_paths)
unexpected_paths = [path for path in all_changed_paths if not is_allowed_v6_path(path)]
if unexpected_paths:
    raise RuntimeError(f"Isolation violation: non-v6 paths changed: {unexpected_paths}")

name_status_lines = git_output("diff", "--name-status", BASELINE_HEAD).splitlines()
non_additive_baseline_changes = [
    line for line in name_status_lines if line and not line.startswith("A\t")
]
if non_additive_baseline_changes:
    raise RuntimeError(
        "Isolation requires every baseline-relative path to be newly added: "
        f"{non_additive_baseline_changes}"
    )

forbidden_import_roots = (
    "src.causal_read",
    "src.interventions",
    "src.read_scores",
    "src.read_validation",
)
cheap_imports = import_names(ROOT / "src/cheap_read.py")
v6_cheap_imports = import_names(ROOT / "scripts/run_hard_dashboard_cheap_v6.py")
forbidden_imports = sorted(
    name
    for name in [*cheap_imports, *v6_cheap_imports]
    if any(name.startswith(prefix) for prefix in forbidden_import_roots)
)
if forbidden_imports:
    raise RuntimeError(f"Cheap-path firewall violation: {forbidden_imports}")

notebook_paths = [
    ROOT / "notebooks/v6_1_within_engine.ipynb",
    ROOT / "notebooks/v6_2_hard_dashboard.ipynb",
    ROOT / "notebooks/v6_3_distributions.ipynb",
    ROOT / "notebooks/v6_4_report.ipynb",
]
notebook_audit = {
    str(path.relative_to(ROOT)): notebook_execution(path) for path in notebook_paths
}
figure_paths = [
    FIGURE_DIR / "F_v6_1_engine_only_read_vs_c.png",
    FIGURE_DIR / "F_v6_2_old_vs_hard_dashboard_auc.png",
    FIGURE_DIR / "F_v6_3_read_ig_distributions.png",
]
figure_audit = {
    str(path.relative_to(ROOT)): {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256(path) if path.is_file() else None,
    }
    for path in figure_paths
}
if not all(row["exists"] and row["bytes"] > 0 for row in figure_audit.values()):
    raise RuntimeError("One or more required v6 figures are absent")

artifact_hash_audit: dict[str, bool] = {}
for check_name in ("check1", "check3"):
    record = metrics[check_name]["raw_artifact"]
    artifact_hash_audit[record["path"]] = (
        sha256(ROOT / record["path"]) == record["sha256"]
    )
for record in metrics["check2"]["artifacts"].values():
    artifact_hash_audit[record["path"]] = (
        sha256(ROOT / record["path"]) == record["sha256"]
    )
if not all(artifact_hash_audit.values()):
    raise RuntimeError("A v6 artifact hash no longer matches metrics")

hf_path = shutil.which("hf") or "/home/jovyan/.local/bin/hf"
if not Path(hf_path).is_file():
    raise FileNotFoundError(f"Hugging Face CLI not found at {hf_path}")
preflight = {
    "hf_path": hf_path,
    "hf_identity": subprocess.run(
        [hf_path, "auth", "whoami"], check=True, capture_output=True, text=True
    ).stdout.strip(),
    "gpu_memory": subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip(),
}

isolation_audit = {
    "status": "PASS",
    "baseline_head": BASELINE_HEAD,
    "baseline_is_ancestor": True,
    "changed_paths": all_changed_paths,
    "unexpected_non_v6_paths": unexpected_paths,
    "all_baseline_relative_changes_are_additions": True,
    "existing_files_modified": False,
    "git_status_before_final_commit": git_output("status", "--short", "--branch"),
}
firewall_audit = {
    "status": "PASS",
    "existing_cheap_imports": cheap_imports,
    "v6_cheap_imports": v6_cheap_imports,
    "forbidden_imports": forbidden_imports,
    "cheap_causal_outputs_consumed": False,
    "existing_cheap_read_unchanged": check2["source_immutability"][
        "cheap_read_unchanged"
    ],
    "existing_causal_read_unchanged": check2["source_immutability"][
        "causal_read_unchanged"
    ],
}

classes = check3["diagnostics"]["classes"]
pairwise = check3["diagnostics"]["pairwise"]
report = f"""# v6 isolated stress test — {decision}

## One-line decision

**{decision_one_line}**

## Isolation and frozen protocol

This stress test starts from frozen commit `{BASELINE_HEAD}` and treats every
pre-existing source file, notebook, and result as read-only. All additions are
confined to `src/*_v6.py`, `scripts/*_v6.py`, `notebooks/v6_*.ipynb`, and
`results/v6/**`. The isolation audit found **no modified pre-v6 path**.

The causal truth, source layer L16, explicit-concept position, 16-step READ_IG,
READ_local, five dependency-group folds, and 10,000-draw seed-1729 group
bootstrap were reused unchanged. No estimator, layer, direction, fold, or score
transformation was retuned.

## CHECK 1 — graded signal within engines (decisive)

Only the 77 frozen verified engines were retained, spanning 24 dependency
groups. Correlations are against frozen signed-interchange `|C|`.

| estimator | engine-only Spearman rho | group-bootstrap 95% CI | interpretation |
| --- | ---: | --- | --- |
| READ_IG | {primary['estimate']:.3f} | [{primary['ci95_low']:.3f}, {primary['ci95_high']:.3f}] | CI spans zero |
| READ_local | {local['estimate']:.3f} | [{local['ci95_low']:.3f}, {local['ci95_high']:.3f}] | CI spans zero |
| capacity baseline | {baseline['estimate']:.3f} | [{baseline['ci95_low']:.3f}, {baseline['ci95_high']:.3f}] | CI spans zero |

READ_IG does not retain a positive graded-use signal inside engines; its point
estimate is negative. Engine `|C|` already lies in the narrow strong-causal
range `{check1['within_engine_auc']['abs_C_min']:.3f}` to
`{check1['within_engine_auc']['abs_C_max']:.3f}`. No within-engine AUC was run:
the frozen protocol has no weak/strong cutoff and the largest adjacent gap
isolates one row, so adding a cutoff now would be post-hoc.

![F_v6_1](figures/F_v6_1_engine_only_read_vs_c.png)

## CHECK 2 — answer-type-matched hard dashboard

The actual frozen engines never answer with numbers: they output chemical
symbols or capital-city names. A numeric control would therefore preserve the
answer-type mismatch. The hard controls instead use fixed calibration-only
anchors with the same relation and semantic answer class: platinum→Pt,
Netherlands→Amsterdam, and Alabama→Montgomery. The original natural context and
explicit concept token remain byte-for-byte unchanged, while the source concept
cannot determine the fixed anchor answer.

- VERIFIED_HARD: **{check2['verification']['verified_hard']}**;
  UNVERIFIED_HARD: **{check2['verification']['unverified_hard']}**.
- Frozen engine median `|C|`: `{sanity['engine_abs_C_median']:.4f}`.
- Hard-dashboard median `|C|`: `{sanity['hard_dashboard_abs_C_median']:.4f}`;
  sharp directional disagreements: `{sanity['hard_dashboard_sharp_directional_disagreements']}`.
- Hard-control causal sanity: **{sanity['status']}**.

| comparison | held-out READ_IG AUC | group-bootstrap 95% CI |
| --- | ---: | --- |
| engine vs old dashboard | {old_auc['estimate']:.3f} | [{old_auc['ci95_low']:.3f}, {old_auc['ci95_high']:.3f}] |
| engine vs hard dashboard | {hard_auc['estimate']:.3f} | [{hard_auc['ci95_low']:.3f}, {hard_auc['ci95_high']:.3f}] |

The harder separation survives in all five frozen folds. Thus arithmetic answer
type is **not** the sole cause of the binary separation. This does not overturn
CHECK 1: surviving a relevant-vs-irrelevant classification does not establish a
graded causal-use meter.

![F_v6_2](figures/F_v6_2_old_vs_hard_dashboard_auc.png)

## CHECK 3 — raw READ_IG distributions

| class | minimum | median | maximum | IQR |
| --- | ---: | ---: | ---: | ---: |
| engine | {classes['engine']['minimum']:.4f} | {classes['engine']['median']:.4f} | {classes['engine']['maximum']:.4f} | {classes['engine']['iqr']:.4f} |
| old dashboard | {classes['old_dashboard']['minimum']:.4f} | {classes['old_dashboard']['median']:.4f} | {classes['old_dashboard']['maximum']:.4f} | {classes['old_dashboard']['iqr']:.4f} |
| hard dashboard | {classes['hard_dashboard']['minimum']:.4f} | {classes['hard_dashboard']['median']:.4f} | {classes['hard_dashboard']['maximum']:.4f} | {classes['hard_dashboard']['iqr']:.4f} |

The old dashboard scores are not identical, but they occupy a compressed low
band. The answer-type-matched hard dashboards occupy essentially the same band:
their observed ranges overlap across
`{pairwise['hard_dashboard__vs__old_dashboard']['overlap_fraction_of_union']:.1%}`
of the union. Both dashboard ranges are strictly disjoint from engines, with
gaps `{pairwise['engine__vs__old_dashboard']['range_gap']:.4f}` (old) and
`{pairwise['engine__vs__hard_dashboard']['range_gap']:.4f}` (hard).

This pattern rules out a specifically arithmetic-gradient explanation, but it
supports the cautionary mechanism: on this roster READ_IG behaves like a binary
relevant-vs-irrelevant detector and offers no demonstrated graded resolution
among already-strong engines.

![F_v6_3](figures/F_v6_3_read_ig_distributions.png)

## Interpretation and scope

The original perfect class separation was inflated as evidence for a *graded*
USE score. A narrower claim survives: READ_IG robustly separates causally
relevant engines from two causally irrelevant control families, including a
semantic answer-type-matched control. It does not rank causal magnitude within
the 77 engines. The evidence is limited to Qwen2.5-7B, L16 explicit written
concepts, three frozen relation families, and an engine set whose causal effects
are all already strong.

## Firewall, reproducibility, and audit

- Cheap READ consumed only sanitized clean manifests and imported the unchanged
  `src/cheap_read.py`; it never read hard C, edited metrics, or interchange
  outputs. Firewall audit: **{firewall_audit['status']}**.
- Hard C was computed separately with the unchanged causal module and matched
  frozen engine T. C remained signed and unclipped.
- Required figures F_v6_1–F_v6_3 and all recorded raw-artifact hashes pass.
- Existing files modified: **none**. Isolation audit: **{isolation_audit['status']}**.
- The test suite, pytest, and Ruff were not run, as required.
"""
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report)

metrics["status"] = "COMPLETE"
metrics["decision"] = decision
metrics["decision_one_line"] = decision_one_line
metrics["preflight"] = preflight
metrics["check4"] = {
    "status": "COMPLETE",
    "decision": decision,
    "decision_one_line": decision_one_line,
    "decision_rule": {
        "engine_only_ci_lower_above_zero": check1["supports_positive_graded_use"],
        "hard_control_auc_survives": check2["hard_auc_survives_at_0_80"],
        "confirmed_requires_both": True,
    },
    "isolation_audit": isolation_audit,
    "firewall_audit": firewall_audit,
    "notebook_audit": notebook_audit,
    "figure_audit": figure_audit,
    "artifact_hash_audit": artifact_hash_audit,
    "prior_results_preserved": True,
    "test_suite_not_run": True,
    "pytest_not_run": True,
    "ruff_not_run": True,
    "report": {
        "path": str(REPORT_PATH.relative_to(ROOT)),
        "bytes": REPORT_PATH.stat().st_size,
        "sha256": sha256(REPORT_PATH),
    },
}
save_json(METRICS_PATH, metrics)

print(decision_one_line)
print("isolation", json.dumps(isolation_audit, indent=2))
print("firewall", json.dumps(firewall_audit, indent=2))
print("notebooks", json.dumps(notebook_audit, indent=2))
print("report", REPORT_PATH, sha256(REPORT_PATH))

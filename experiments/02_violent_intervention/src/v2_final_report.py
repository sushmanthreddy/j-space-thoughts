"""Model-free Stage-4 fallback report for the repair-first v2 workflow.

This module is intentionally limited to the replication-failure path.  It
cannot render or persist a Stage-4 report unless the persisted gate ledger says
that Stage 3 was skipped for a failed prerequisite and Stage 2 requires the
fallback.  The legacy predictor comparison is retained only as the explicitly
requested fallback observation; it is never promoted to a P1--P3 result.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.metrics import save_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "stage4-replication-failure-v2"
STAGE4_HEADING = "## Stage 4 — replication-failure fallback"

LEGACY_SOURCE_COMMIT = "6666385cff42fe4053412e7230ec9f55b0259f79"
LEGACY_PREDICTOR_COMPARISON = {
    "status": "NO_OBSERVED_JLENS_ADVANTAGE",
    "outcome": "legacy v1 measured all-band ablation delta",
    "n": 155,
    "jlens": {
        "pearson_r": 0.6083677,
        "ci_low": 0.5167659,
        "ci_high": 0.6927411,
    },
    "identity_j_logit_lens": {
        "pearson_r": 0.6394085,
        "ci_low": 0.5519587,
        "ci_high": 0.7188587,
    },
    "provenance_commit": LEGACY_SOURCE_COMMIT,
    "interpretation": (
        "The J-Lens point estimate did not exceed the identity-J/logit-lens "
        "baseline; this fallback comparison shows no observed J-Lens advantage."
    ),
    "scope_guardrail": (
        "This legacy comparison is a requested fallback observation only.  It "
        "does not test P1, P2, or P3 and does not rescue the invalidated v1 "
        "science conclusions."
    ),
}


class Stage4PrerequisiteError(ValueError):
    """The persisted run state does not license a Stage-4 fallback report."""


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Read one metrics JSON object without changing it."""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read metrics JSON at {target}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object")
    return payload


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage4PrerequisiteError(f"{path} must be a mapping")
    return value


def _require_stage4_state(metrics: Mapping[str, Any]) -> tuple[
    Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]
]:
    repair = _mapping(metrics.get("repair_v2"), path="repair_v2")
    ledger = _mapping(repair.get("gate_ledger"), path="repair_v2.gate_ledger")
    stage2 = _mapping(
        repair.get("stage2_recalibration"),
        path="repair_v2.stage2_recalibration",
    )

    issues: list[str] = []
    if ledger.get("stage3_science") != "SKIPPED_PREREQUISITE":
        issues.append(
            "gate_ledger.stage3_science must equal SKIPPED_PREREQUISITE"
        )
    if stage2.get("stage4_required") is not True:
        issues.append("stage2_recalibration.stage4_required must be true")
    if stage2.get("stage3_allowed") is not False:
        issues.append("stage2_recalibration.stage3_allowed must be false")
    if stage2.get("status") != "FAIL":
        issues.append("stage2_recalibration.status must equal FAIL")
    if issues:
        raise Stage4PrerequisiteError("; ".join(issues))
    return repair, ledger, stage2


def _status(value: Any, *, path: str) -> str:
    payload = _mapping(value, path=path)
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise Stage4PrerequisiteError(f"{path}.status must be a nonempty string")
    return status


def _custom_swap_summary(
    repair: Mapping[str, Any], stage2: Mapping[str, Any]
) -> dict[str, Any]:
    stage1 = _mapping(repair.get("stage1"), path="repair_v2.stage1")
    g_swap = _mapping(stage1.get("g_swap"), path="repair_v2.stage1.g_swap")
    n_pass = int(g_swap.get("n_pass", -1))
    n_required = int(g_swap.get("n_required", -1))
    reverification = _status(
        stage2.get("g_swap_reverification"),
        path="repair_v2.stage2_recalibration.g_swap_reverification",
    )
    if (
        g_swap.get("status") != "PASS"
        or n_pass != 3
        or n_required != 3
        or reverification != "PASS"
    ):
        raise Stage4PrerequisiteError(
            "Stage 4 expects the persisted custom G-SWAP result to be PASS "
            "(3/3) and its Stage-2 reverification to pass"
        )
    return {
        "status": "PASS",
        "n_pass": n_pass,
        "n_required": n_required,
        "stage2_reverification": reverification,
        "configuration": stage2.get("configuration"),
    }


def _upstream_omission_summary(
    repair: Mapping[str, Any], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    stage0 = _mapping(repair.get("stage0"), path="repair_v2.stage0")
    expected = "UPSTREAM_CAUSAL_SWAP_NOT_RUNNABLE_RELEASE_OMISSION"
    if stage0.get("decision") != expected:
        raise Stage4PrerequisiteError(
            f"repair_v2.stage0.decision must equal {expected}"
        )
    if ledger.get("upstream_causal_swap") != "NOT_RUNNABLE_RELEASE_OMISSION":
        raise Stage4PrerequisiteError(
            "gate_ledger.upstream_causal_swap must record the release omission"
        )
    audit = stage0.get("upstream_release_audit", {})
    code_available = (
        audit.get("causal_swap_code_available")
        if isinstance(audit, Mapping)
        else None
    )
    return {
        "status": "NOT_RUNNABLE_RELEASE_OMISSION",
        "decision": expected,
        "causal_swap_code_available": code_available,
        "interpretation": (
            "The released upstream walkthrough provided readout but no executable "
            "causal-swap implementation; this is a release-scope omission, not "
            "evidence that the hypothesis is false."
        ),
    }


def _blockers(stage2: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: Sequence[tuple[str, str, str]] = (
        ("G-SWAP reverification", "g_swap_reverification", "known-answer swap"),
        ("firing controls", "controls_fire", "non-structural output controls"),
        ("matched random-pair null", "random_pair_null", "specificity null"),
        ("absent-coordinate null", "absent_coordinate_null", "specificity null"),
        ("capability preservation", "capability", "unrelated-text delta NLL"),
        ("G-POS", "g_pos", "known-narration positive control"),
    )
    blockers: list[dict[str, Any]] = []
    for label, key, role in checks:
        payload = _mapping(
            stage2.get(key), path=f"repair_v2.stage2_recalibration.{key}"
        )
        status = _status(payload, path=f"repair_v2.stage2_recalibration.{key}")
        if status == "PASS":
            continue
        evidence: dict[str, Any] = {}
        if key == "capability":
            evidence = {
                field: payload.get(field)
                for field in (
                    "mean_delta_nll",
                    "mean_abs_delta_nll",
                    "threshold",
                    "criterion",
                )
                if field in payload
            }
        elif key == "g_pos":
            evidence = {
                field: payload.get(field)
                for field in (
                    "n_reproduced",
                    "n_passages",
                    "categories_reproduced",
                    "criterion",
                )
                if field in payload
            }
        else:
            evidence = {
                field: payload.get(field)
                for field in ("criterion", "n_eligible", "n_draws_per_item")
                if field in payload
            }
        blockers.append(
            {
                "gate": label,
                "key": key,
                "role": role,
                "status": status,
                "evidence": evidence,
            }
        )
    if not blockers:
        raise Stage4PrerequisiteError(
            "Stage 2 requires Stage 4 but no failed calibration blocker was found"
        )
    return blockers


def _skipped_notebooks(repair: Mapping[str, Any]) -> dict[str, str]:
    records = _mapping(
        repair.get("stage3_notebooks"), path="repair_v2.stage3_notebooks"
    )
    expected = {
        "05": "05_science_twohop.ipynb",
        "06": "06_science_ambiguity.ipynb",
        "07": "07_scale.ipynb",
    }
    skipped: dict[str, str] = {}
    for number, filename in expected.items():
        row = _mapping(
            records.get(number), path=f"repair_v2.stage3_notebooks.{number}"
        )
        if row.get("status") != "SKIPPED_PREREQUISITE":
            raise Stage4PrerequisiteError(
                f"Stage-3 notebook {number} lacks its prerequisite-skip record"
            )
        if row.get("science_executed") is not False:
            raise Stage4PrerequisiteError(
                f"Stage-3 notebook {number} does not prove science_executed=false"
            )
        if row.get("model_inference_run") is not False:
            raise Stage4PrerequisiteError(
                f"Stage-3 notebook {number} does not prove model_inference_run=false"
            )
        skipped[filename] = "SKIPPED_PREREQUISITE"
    return skipped


def _resolve_figure(root: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else root / path).resolve()
    return resolved if resolved.is_file() else None


def _report_relative_path(root: Path, path: Path) -> str:
    report_dir = (root / "results").resolve()
    try:
        return path.resolve().relative_to(report_dir).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())


def _valid_figures(
    repair: Mapping[str, Any], stage2: Mapping[str, Any], *, root: Path
) -> list[dict[str, Any]]:
    stage0 = _mapping(repair.get("stage0"), path="repair_v2.stage0")
    stage1 = _mapping(repair.get("stage1"), path="repair_v2.stage1")
    stage1c = _mapping(
        repair.get("stage1c_concept_finder"),
        path="repair_v2.stage1c_concept_finder",
    )
    stage1d = _mapping(
        repair.get("stage1d_read_validation"),
        path="repair_v2.stage1d_read_validation",
    )
    candidates: Sequence[tuple[str, str, Any, bool]] = (
        ("F0", "Stage-0 upstream audit", stage0.get("figure"), True),
        (
            "G-SWAP",
            "Repaired three-case swap calibration",
            stage1.get("figure"),
            _status(stage1.get("g_swap"), path="repair_v2.stage1.g_swap")
            == "PASS",
        ),
        (
            "G-DIR",
            "Independent direction validation",
            stage1c.get("figure"),
            stage1c.get("status") == "PASS",
        ),
        (
            "F5",
            "Repaired READ validation",
            stage1d.get("figure"),
            stage1d.get("status") == "PASS",
        ),
        (
            "F3",
            "Firing output-suppression control",
            stage2.get("figure_f3"),
            _status(
                stage2.get("controls_fire"),
                path="repair_v2.stage2_recalibration.controls_fire",
            )
            == "PASS",
        ),
    )
    figures: list[dict[str, Any]] = []
    for figure_id, title, raw_path, prerequisite_passed in candidates:
        if not prerequisite_passed:
            continue
        resolved = _resolve_figure(root, raw_path)
        if resolved is None:
            continue
        figures.append(
            {
                "id": figure_id,
                "title": title,
                "path": _report_relative_path(root, resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    return figures


def build_stage4_report(
    metrics: Mapping[str, Any], *, root: str | Path = ROOT
) -> dict[str, Any]:
    """Build the fallback payload after proving the science path was skipped."""

    repair, ledger, stage2 = _require_stage4_state(metrics)
    blockers = _blockers(stage2)
    custom_swap = _custom_swap_summary(repair, stage2)
    upstream = _upstream_omission_summary(repair, ledger)
    predictions = {key: "NOT_TESTED" for key in ("P1", "P2", "P3")}
    skipped = _skipped_notebooks(repair)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "classification": "OPEN_MODEL_INSTRUMENT_REPLICATION_FAILURE",
        "stage3_science": "SKIPPED_PREREQUISITE",
        "stage4_required": True,
        "upstream_release": upstream,
        "custom_swap": custom_swap,
        "calibration_blockers": blockers,
        "skipped_notebooks": skipped,
        "predictions": predictions,
        "legacy_fallback_comparison": json.loads(
            json.dumps(LEGACY_PREDICTOR_COMPARISON)
        ),
        "valid_figures": _valid_figures(repair, stage2, root=Path(root)),
        "claim_boundary": {
            "hypothesis_false_established": False,
            "hypothesis_status": "NOT_TESTED",
            "allowed_claim": (
                "The released upstream causal intervention could not be run "
                "unchanged, and the repaired open-Qwen instrument failed Stage-2 "
                "calibration; therefore P1--P3 were not tested."
            ),
            "forbidden_claim": (
                "This result must not be described as showing that the "
                "WRITE-versus-READ hypothesis is false."
            ),
        },
    }


def _format_blocker_evidence(blocker: Mapping[str, Any]) -> str:
    evidence = blocker.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return "Persisted Stage-2 gate failure."
    if blocker.get("key") == "capability":
        mean = float(evidence.get("mean_delta_nll", float("nan")))
        mean_abs = float(evidence.get("mean_abs_delta_nll", float("nan")))
        threshold = float(evidence.get("threshold", float("nan")))
        return (
            f"mean delta NLL={mean:.3f}; mean absolute delta NLL={mean_abs:.3f}; "
            f"threshold={threshold:.2f}"
        )
    if blocker.get("key") == "g_pos":
        categories = evidence.get("categories_reproduced")
        languages = (
            ", ".join(str(value) for value in categories)
            if isinstance(categories, Sequence) and not isinstance(categories, str)
            else str(categories)
        )
        return (
            f"{evidence.get('n_reproduced')}/{evidence.get('n_passages')} "
            f"passages; languages={languages}"
        )
    return str(evidence.get("criterion") or "Persisted Stage-2 gate failure.")


def render_stage4_section(stage4: Mapping[str, Any]) -> str:
    """Render the final chronological section from one validated payload."""

    if stage4.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected Stage-4 report schema")
    blockers = stage4.get("calibration_blockers", [])
    blocker_rows = "\n".join(
        f"| {row['gate']} | {row['status']} | {_format_blocker_evidence(row)} |"
        for row in blockers
    )
    skipped = stage4.get("skipped_notebooks", {})
    skipped_rows = "\n".join(
        f"| `{name}` | {status} |"
        for name, status in skipped.items()
    )
    predictions = stage4.get("predictions", {})
    prediction_rows = "\n".join(
        f"| {name} | **{status}** |" for name, status in predictions.items()
    )
    legacy = stage4["legacy_fallback_comparison"]
    jlens = legacy["jlens"]
    identity = legacy["identity_j_logit_lens"]
    figures = stage4.get("valid_figures", [])
    if figures:
        figure_lines = "\n".join(
            f"- [{row['id']}: {row['title']}]({row['path']})" for row in figures
        )
    else:
        figure_lines = "- No prerequisite-valid figure file was found at render time."
    custom = stage4["custom_swap"]

    return f"""

{STAGE4_HEADING}

### Final classification

**OPEN-MODEL INSTRUMENT REPLICATION FAILURE; HYPOTHESIS NOT TESTED.** The
custom repaired swap passed {custom['n_pass']}/{custom['n_required']} known-answer
cases and passed Stage-2 reverification, but the complete calibration chain did
not pass. Stage 3 was therefore skipped exactly as preregistered.

The released upstream walkthrough supplied J-Lens readout but omitted executable
causal-swap code (`NOT_RUNNABLE_RELEASE_OMISSION`), so the upstream intervention
could not be run unchanged. The successful 3/3 custom swap is a local repair,
not evidence that all required controls calibrated.

### Calibration blockers

| gate | status | persisted evidence |
| --- | --- | --- |
{blocker_rows}

### Science path deliberately skipped

| notebook | disposition |
| --- | --- |
{skipped_rows}

| preregistered prediction | result |
| --- | --- |
{prediction_rows}

No Stage-3 result was computed or inferred from the failed calibration.

### Requested legacy fallback comparison

From commit `{legacy['provenance_commit']}`, over N={legacy['n']} legacy items:

| predictor | Pearson r | 95% CI |
| --- | ---: | ---: |
| J-Lens | {jlens['pearson_r']:.7f} | [{jlens['ci_low']:.7f}, {jlens['ci_high']:.7f}] |
| identity-J / logit lens | {identity['pearson_r']:.7f} | [{identity['ci_low']:.7f}, {identity['ci_high']:.7f}] |

**No observed J-Lens advantage:** its point estimate (0.6083677) was below the
identity-J/logit-lens estimate (0.6394085), with overlapping intervals. This is
the requested fallback observation only; it does not test P1, P2, or P3 and
does not reinstate any withdrawn v1 science conclusion.

### Prerequisite-valid figures

{figure_lines}

### Claim boundary

**This report does not establish that the WRITE-versus-READ hypothesis is
false.** It establishes only that the causal instrument could not be validated
through the full required calibration chain on the open Qwen setup used here.
"""


def render_results_with_stage4(existing_report: str, stage4: Mapping[str, Any]) -> str:
    """Preserve chronological Stage 0--2 text and replace any Stage-4 tail."""

    required = (
        "## Stage 0",
        "## Stage 1",
        "## Stage 2",
    )
    missing = [heading for heading in required if heading not in existing_report]
    if missing:
        raise ValueError(
            "Existing RESULTS.md lacks chronological prerequisite sections: "
            + ", ".join(missing)
        )
    base = existing_report
    marker = f"\n{STAGE4_HEADING}"
    if marker in base:
        base = base.split(marker, 1)[0]
    return base.rstrip() + render_stage4_section(stage4).rstrip() + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def persist_stage4(
    *,
    metrics_path: str | Path = ROOT / "results" / "metrics.json",
    report_path: str | Path = ROOT / "results" / "RESULTS.md",
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Validate, build, and atomically persist the Stage-4 fallback artifacts."""

    metrics_target = Path(metrics_path)
    report_target = Path(report_path)
    metrics = load_metrics(metrics_target)
    stage4 = build_stage4_report(metrics, root=root)
    try:
        existing_report = report_target.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Cannot read existing RESULTS.md at {report_target}") from error
    rendered_report = render_results_with_stage4(existing_report, stage4)

    repair = metrics["repair_v2"]
    repair["stage4_report"] = stage4
    repair["gate_ledger"]["stage4_report"] = "COMPLETE"
    repair["gate_ledger"]["stage3_science"] = "SKIPPED_PREREQUISITE"
    repair["current_allowed_conclusion"] = (
        "STAGE4_REPLICATION_FAILURE_NO_HYPOTHESIS_INFERENCE"
    )
    save_json(metrics_target, metrics)
    _atomic_write_text(report_target, rendered_report)
    return metrics

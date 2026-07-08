"""Model-free v3 prerequisite skips and Stage-4 fallback reporting."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.metrics import save_json
from src.v3_alpha_sweep import _validate_alpha_sweep
from src.v3_reverify import _repair_v2_sha256


ROOT = Path(__file__).resolve().parents[1]
STAGE3_NOTEBOOKS = {
    "05_science_twohop.ipynb": {
        "prediction_scope": "P1 and P2",
        "title": "two-hop and narration science",
    },
    "06_science_ambiguity.ipynb": {
        "prediction_scope": "P3",
        "title": "ambiguity science",
    },
    "07_scale.ipynb": {
        "prediction_scope": "P1 across model scale",
        "title": "scale science",
    },
}
STAGE3_ORDER = tuple(STAGE3_NOTEBOOKS)


def _load_validated_metrics() -> dict[str, Any]:
    metrics = json.loads(
        (ROOT / "results" / "metrics.json").read_text(encoding="utf-8")
    )
    v3 = metrics["calibration_v3"]
    if (
        _repair_v2_sha256(metrics["repair_v2"])
        != v3["provenance"]["repair_v2_sha256"]
    ):
        raise RuntimeError("Immutable repair_v2 provenance changed during v3")
    _validate_alpha_sweep(v3["stage1_5_alpha_sweep"])
    if (
        v3["gate_ledger"]["stage0_reverify"] != "PASS"
        or v3["gate_ledger"]["g_swap"] != "PASS"
        or v3["gate_ledger"]["g_alpha"] != "FAIL"
        or v3["selected_intervention"] is not None
    ):
        raise RuntimeError("V3 fallback requires PASS/PASS/FAIL and no alpha*")
    return metrics


def _stage2_section(stage2: Mapping[str, Any]) -> str:
    gates = "\n".join(
        f"- {name}: **{status}**"
        for name, status in stage2["alpha_star_gates"].items()
    )
    return f"""

## Stage 2 — recalibration at alpha*

**{stage2['status']}.** {stage2['reason']} No model forward was run in notebook
04. The following alpha*-specific checks therefore remain unmeasured:

{gates}

Stage-0 G-DIR re-verification is retained as an instrument sentinel, but it is
not relabeled as a Stage-2 result at a nonexistent alpha*.
"""


def _stage3_section(stage3: Mapping[str, Mapping[str, Any]]) -> str:
    rows = "\n".join(
        "| {notebook} | {scope} | {status} |".format(
            notebook=name,
            scope=stage3[name]["prediction_scope"],
            status=stage3[name]["status"],
        )
        for name in STAGE3_ORDER
        if name in stage3
    )
    return f"""

## Stage 3 — science prerequisite records

| notebook | preregistered scope | result |
| --- | --- | --- |
{rows}

These notebooks are executed model-free guards. They do not import historical
science values or treat missing measurements as negative effects.
"""


def _stage4_section(stage4: Mapping[str, Any]) -> str:
    observation = stage4["observations"]
    legacy = stage4["legacy_fallback_comparison"]
    read_ratios = ", ".join(
        f"{key}={value:.3f}"
        for key, value in observation["masked_alpha_1_5"][
            "weight_read_ratios"
        ].items()
    )
    return f"""

## Stage 4 — calibration-limitation result

**Classification: {stage4['classification']}.** The working v2 intervention
was reproducible, and v3 again confirmed all three alpha-2 sentinel swaps.
However, the frozen source-capped surgical policy reached at most
**{observation['primary_policy']['max_swaps']}/3** known-answer flips over the
full alpha grid, so it never met G-SWAP.

The strongest exploratory alternative, a carrying-position fractional swap at
alpha=1.50, flipped **3/3** cases and passed the random and absent-coordinate
checks. Its narration changes were small on **8/8** passages, but G-POS was
**0/8** because low primary weight-READ was **0/8**; the mask-specific ratios
were {read_ratios}, all above the <=0.50 criterion. One passage (`es2`) also
lacked clean continuation capability. These ratios are fixed-mask properties,
so tuning alpha cannot repair that subgate.

Capability delta NLL was exactly zero for the masked policies only because
**{observation['capability']['empty_masks']}/{observation['capability']['total_masks']}**
unrelated-text masks were empty.
This is evidence that the detector did not fire on that fixed bank, not an
active-edit capability stress test. The all-position reference did actively
edit those texts; at alpha=2 its signed mean delta NLL was
**{observation['all_position_alpha_2']['mean_delta_nll']:+.3f}** and mean
absolute delta NLL was
**{observation['all_position_alpha_2']['mean_abs_delta_nll']:.3f}**.

### Claim boundary

- P1, P2, and P3 are **NOT TESTED**.
- This run does **not** show that the Written-vs-Read hypothesis is false.
- It shows that the frozen intervention plus primary weight-READ positive
  control could not be jointly calibrated on open Qwen2.5-7B.
- Stage-2 independent weight-READ validation was never licensed and remains
  outstanding.

The requested legacy comparison is descriptive only: invalidated v1 reported
J-Lens `r={legacy['jlens']['pearson_r']:.3f}` versus identity-J/logit-lens
`r={legacy['identity_j_logit_lens']['pearson_r']:.3f}` at `N={legacy['n']}`.
Those values come from commit `{legacy['provenance_commit']}` and cannot be
used as evidence for P1-P3 because that instrument failed its gates.

The complete alpha sweep is in `results/metrics.json`; the full raw draw-level
artifact is `{stage4['raw_artifact']['path']}` with SHA-256
`{stage4['raw_artifact']['sha256']}`. F-ALPHA is the only new figure licensed by
the v3 gate chain.
"""


def _render_downstream_report(metrics: Mapping[str, Any]) -> None:
    v3 = metrics["calibration_v3"]
    report_path = ROOT / "results" / "RESULTS.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## Stage 2 — recalibration at alpha*"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    if "stage2_recalibration" in v3:
        report = report.rstrip() + _stage2_section(v3["stage2_recalibration"])
    stage3 = v3.get("stage3_notebooks", {})
    if stage3:
        report = report.rstrip() + _stage3_section(stage3)
    if "stage4_fallback" in v3:
        report = report.rstrip() + _stage4_section(v3["stage4_fallback"])

    verdict_start = report.index("## Current verdict")
    environment_start = report.index("## Environment")
    if v3["gate_ledger"]["stage4_report"] == "PASS":
        verdict = (
            "## Current verdict\n\n"
            "**V3 COMPLETE — CALIBRATION/READ-POSITIVE-CONTROL LIMITATION; "
            "NO HYPOTHESIS VERDICT.** G-SWAP passed, but no frozen alpha "
            "satisfied G-ALPHA. Stage 2 and Stage 3 were skipped by "
            "prerequisite, and P1-P3 remain untested.\n\n"
        )
    else:
        verdict = (
            "## Current verdict\n\n"
            "**G-ALPHA FAILED; STAGE 2 AND STAGE 3 SKIPPED.** Stage 4 is "
            "required. This is a calibration limitation, not a hypothesis "
            "verdict.\n\n"
        )
    report = report[:verdict_start] + verdict + report[environment_start:]
    report_path.write_text(report.rstrip() + "\n", encoding="utf-8")


def record_stage2_skip() -> dict[str, Any]:
    """Persist the no-alpha Stage-2 prerequisite record."""

    metrics = _load_validated_metrics()
    v3 = metrics["calibration_v3"]
    v3["stage2_recalibration"] = {
        "schema_version": "stage2-skip-v3",
        "status": "SKIPPED_PREREQUISITE",
        "prerequisite": "G-ALPHA PASS with selected alpha*",
        "observed": "G-ALPHA FAIL; selected_intervention=None",
        "reason": (
            "No alpha* exists, so no alpha*-specific recalibration is defined."
        ),
        "model_forward_run": False,
        "alpha_star_gates": {
            "G-SWAP at alpha*": "NOT_EVALUATED_NO_ALPHA",
            "G-DIR at alpha*": "NOT_EVALUATED_NO_ALPHA",
            "capability at alpha*": "NOT_EVALUATED_NO_ALPHA",
            "G-POS at alpha*": "NOT_EVALUATED_NO_ALPHA",
            "weight-READ validation at alpha*": "NOT_EVALUATED_NO_ALPHA",
        },
    }
    v3["gate_ledger"]["stage2_recalibration"] = "SKIPPED_PREREQUISITE"
    v3["gate_ledger"]["stage3_science"] = "SKIPPED_PREREQUISITE"
    v3["current_allowed_conclusion"] = (
        "NO_VALID_ALPHA_CALIBRATION_LIMITATION_NO_HYPOTHESIS_INFERENCE"
    )
    save_json(ROOT / "results" / "metrics.json", metrics)
    _render_downstream_report(metrics)
    return metrics


def record_stage3_skip(notebook: str) -> dict[str, Any]:
    """Persist one ordered Stage-3 prerequisite skip without model inference."""

    if notebook not in STAGE3_NOTEBOOKS:
        raise ValueError(f"Unknown Stage-3 notebook: {notebook}")
    metrics = _load_validated_metrics()
    v3 = metrics["calibration_v3"]
    if v3.get("stage2_recalibration", {}).get("status") != (
        "SKIPPED_PREREQUISITE"
    ):
        raise RuntimeError("Stage-3 skip requires the executed Stage-2 skip")
    index = STAGE3_ORDER.index(notebook)
    prior = STAGE3_ORDER[:index]
    recorded = v3.setdefault("stage3_notebooks", {})
    if any(name not in recorded for name in prior):
        raise RuntimeError("Stage-3 prerequisite notebooks must run in order")
    recorded[notebook] = {
        "status": "SKIPPED_PREREQUISITE",
        "prediction_scope": STAGE3_NOTEBOOKS[notebook]["prediction_scope"],
        "title": STAGE3_NOTEBOOKS[notebook]["title"],
        "reason": "G-ALPHA failed and no alpha* exists",
        "model_forward_run": False,
        "science_values_loaded": False,
    }
    v3["gate_ledger"]["stage3_science"] = "SKIPPED_PREREQUISITE"
    save_json(ROOT / "results" / "metrics.json", metrics)
    _render_downstream_report(metrics)
    return metrics


def build_stage4_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Build the final factual payload from completed v3 calibration metrics."""

    v3 = metrics["calibration_v3"]
    sweep = v3["stage1_5_alpha_sweep"]
    primary_rows = [
        row for row in sweep["rows"] if row["policy"] == "project_out_transfer"
    ]
    masked = next(
        row
        for row in sweep["rows"]
        if row["policy"] == "fractional_swap_carrying_positions"
        and row["alpha"] == 1.5
    )
    all_position_2 = next(
        row
        for row in sweep["rows"]
        if row["policy"] == "fractional_swap_all_positions_reference"
        and row["alpha"] == 2.0
    )
    capability_masks = sweep["mask_manifest"]["capability"]
    weight_read_ratios = {
        row["key"]: float(row["primary_weight_read_ratio"])
        for row in masked["g_pos"]["rows"]
    }
    legacy = copy.deepcopy(
        metrics["repair_v2"]["stage4_report"]["legacy_fallback_comparison"]
    )
    return {
        "schema_version": "stage4-calibration-limitation-v3",
        "status": "COMPLETE",
        "classification": "CALIBRATION_READ_POSITIVE_CONTROL_LIMITATION",
        "failed_gate": "G-ALPHA",
        "predictions": {"P1": "NOT_TESTED", "P2": "NOT_TESTED", "P3": "NOT_TESTED"},
        "observations": {
            "primary_policy": {
                "policy": "project_out_transfer",
                "max_swaps": max(
                    row["known_swaps"]["n_pass"] for row in primary_rows
                ),
            },
            "masked_alpha_1_5": {
                "status": "EXPLORATORY_NONSELCTABLE",
                "swaps": masked["known_swaps"]["n_pass"],
                "g_pos": masked["g_pos"]["n_reproduced"],
                "low_causal_abs": sum(
                    row["checks"]["low_causal_abs_delta"]
                    for row in masked["g_pos"]["rows"]
                ),
                "low_weight_read": sum(
                    row["checks"]["low_primary_weight_read_ratio"]
                    for row in masked["g_pos"]["rows"]
                ),
                "random_null": masked["random_null"]["status"],
                "absent_null": masked["absent_null"]["status"],
                "weight_read_ratios": weight_read_ratios,
            },
            "capability": {
                "empty_masks": sum(
                    not row["positions"] for row in capability_masks.values()
                ),
                "total_masks": len(capability_masks),
                "interpretation": "NO_EDIT_OPPORTUNITY",
            },
            "all_position_alpha_2": {
                "mean_delta_nll": all_position_2["capability"][
                    "mean_delta_nll"
                ],
                "mean_abs_delta_nll": all_position_2["capability"][
                    "mean_abs_delta_nll"
                ],
            },
        },
        "raw_artifact": {
            "path": sweep["raw_artifact"],
            "bytes": sweep["raw_artifact_bytes"],
            "sha256": sweep["raw_artifact_sha256"],
        },
        "claim_boundary": {
            "hypothesis_status": "NOT_TESTED",
            "hypothesis_false_established": False,
            "allowed_claim": (
                "No frozen intervention strength passed G-ALPHA; the current "
                "intervention/weight-READ positive-control pair was not fully "
                "calibrated on Qwen2.5-7B."
            ),
            "forbidden_claim": (
                "The result must not be described as showing that the "
                "Written-vs-Read hypothesis is false."
            ),
        },
        "legacy_fallback_comparison": legacy,
        "valid_figures": [
            {
                "id": "F-ALPHA",
                "path": sweep["figure"],
                "title": "Surgical intervention alpha sweep",
            }
        ],
    }


def record_stage4_fallback() -> dict[str, Any]:
    """Persist the final v3 calibration-limitation claim boundary."""

    metrics = _load_validated_metrics()
    v3 = metrics["calibration_v3"]
    stage3 = v3.get("stage3_notebooks", {})
    if any(
        stage3.get(name, {}).get("status") != "SKIPPED_PREREQUISITE"
        for name in STAGE3_ORDER
    ):
        raise RuntimeError("Stage 4 requires all three executed Stage-3 skips")
    v3["stage4_fallback"] = build_stage4_payload(metrics)
    v3["gate_ledger"]["stage4_report"] = "PASS"
    v3["current_allowed_conclusion"] = (
        "CALIBRATION_READ_POSITIVE_CONTROL_LIMITATION_NO_HYPOTHESIS_VERDICT"
    )
    save_json(ROOT / "results" / "metrics.json", metrics)
    _render_downstream_report(metrics)
    return metrics

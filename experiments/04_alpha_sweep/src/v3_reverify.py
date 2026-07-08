"""V3 preflight and bounded re-verification of the working v2 instrument."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.data_gen import G1_PROMPTS
from src.interventions import clamped_swap_edits, forward_logits
from src.jlens_iface import jlens_direction_bank
from src.metrics import logit_difference, save_json
from src.model_utils import ModelBundle, capture_residuals, hf_wrapper_logit_kl
from src.v2_recalibration import _direct_concept_controls
from src.v2_repair import MODEL_ID, load_calibration_items
from src.v2_stage0 import collect_preflight


ROOT = Path(__file__).resolve().parents[1]
MD_ARTIFACT = ROOT / "data" / "directions" / "qwen2.5-7b_md_v2.pt"
V2_SOURCE_COMMIT = "7219877f6ce1589191ee74c79508ed080d7ba8fc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repair_v2_sha256(repair_v2: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        repair_v2, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _metric(logits: torch.Tensor, item: Mapping[str, Any]) -> float:
    return float(
        logit_difference(
            logits,
            int(item["clean_answer_token_id"]),
            int(item["counterfactual_answer_token_id"]),
        )[0].cpu()
    )


def build_calibration_bank(
    bundle: ModelBundle,
    lens: Any,
    items: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
) -> dict[int, dict[int, torch.Tensor]]:
    token_ids = {
        int(token_id)
        for item in items
        for token_id in (
            item["source_concept_token_id"],
            item["target_concept_token_id"],
        )
    }
    return jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        layers,
        fold_rms_gain=False,
    )


def run_known_swaps(
    bundle: ModelBundle,
    items: Sequence[Mapping[str, Any]],
    bank: Mapping[int, Mapping[int, torch.Tensor]],
    layers: Sequence[int],
    *,
    strength: float = 2.0,
    repeats: int = 3,
) -> dict[str, Any]:
    """Re-run the three frozen known-answer swaps with deterministic repeats."""

    rows: list[dict[str, Any]] = []
    for item in items:
        input_ids = bundle.lens_model.encode(str(item["prompt"]))
        clean = forward_logits(bundle.hf_model, input_ids)
        residuals = capture_residuals(bundle.lens_model, input_ids, layers)
        source = {
            int(layer): bank[int(item["source_concept_token_id"])][int(layer)]
            for layer in layers
        }
        target = {
            int(layer): bank[int(item["target_concept_token_id"])][int(layer)]
            for layer in layers
        }
        outputs = [
            forward_logits(
                bundle.hf_model,
                input_ids,
                blocks=bundle.lens_model.layers,
                edits=clamped_swap_edits(
                    residuals,
                    source,
                    target,
                    strength=strength,
                ),
            )
            for _ in range(repeats)
        ]
        edited = outputs[0]
        clean_top_id = int(clean[0, -1].argmax())
        edited_top_ids = [int(output[0, -1].argmax()) for output in outputs]
        clean_metric = _metric(clean, item)
        edited_metric = _metric(edited, item)
        repeat_error = (
            max(
                float((output - edited).abs().max().cpu())
                for output in outputs[1:]
            )
            if repeats > 1
            else 0.0
        )
        passed = bool(
            clean_top_id == int(item["clean_answer_token_id"])
            and all(
                token_id == int(item["counterfactual_answer_token_id"])
                for token_id in edited_top_ids
            )
            and repeat_error <= 1e-6
        )
        rows.append(
            {
                "name": item["name"],
                "source": item["source_concept_surface"],
                "target": item["target_concept_surface"],
                "strength": strength,
                "layers": [int(layer) for layer in layers],
                "positions": "all_prompt_positions",
                "clean_top_id": clean_top_id,
                "clean_top": bundle.tokenizer.decode([clean_top_id]),
                "edited_top_ids": edited_top_ids,
                "edited_top": bundle.tokenizer.decode([edited_top_ids[0]]),
                "clean_metric": clean_metric,
                "edited_metric": edited_metric,
                "delta_metric": edited_metric - clean_metric,
                "repeat_max_abs_logit_error": repeat_error,
                "pass": passed,
            }
        )
    return {
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "n_pass": sum(row["pass"] for row in rows),
        "n_required": len(rows),
        "rows": rows,
    }


def verify_gdir_artifact(
    v2_metrics: Mapping[str, Any],
    *,
    artifact_path: Path = MD_ARTIFACT,
) -> dict[str, Any]:
    """Validate the persisted held-out G-DIR evidence and its cached directions."""

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing v2 MD artifact: {artifact_path}")
    stage1c = v2_metrics["stage1c_concept_finder"]
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    metadata = artifact["metadata"]
    directions = artifact["mean_difference"]
    norms = [
        float(vector.float().norm())
        for concept in directions.values()
        for vector in concept.values()
    ]
    storage_dtypes = sorted(
        {
            str(vector.dtype)
            for concept in directions.values()
            for vector in concept.values()
        }
    )
    serialized_norm_tolerance = 1e-4
    retrieval = stage1c["retrieval"]["top1_at_fixed_layer"]
    explicit = stage1c["explicit_known_answer"]["heldout_top5"]
    checks = {
        "v2_gate_passed": stage1c["status"] == "PASS",
        "artifact_schema": metadata.get("schema_version") == "md-repair-v2",
        "model_matches": metadata.get("model_id") == MODEL_ID,
        "forty_concepts": len(directions) == 40,
        "unit_norm": (
            max(abs(value - 1.0) for value in norms)
            <= serialized_norm_tolerance
        ),
        "retrieval_above_chance": (
            float(retrieval["estimate"]) > float(stage1c["chance_retrieval"])
        ),
        "known_answer_top5": float(explicit["estimate"]) >= 0.80,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "artifact_path": (
            str(artifact_path.relative_to(ROOT))
            if artifact_path.is_relative_to(ROOT)
            else str(artifact_path.resolve())
        ),
        "artifact_sha256": _sha256(artifact_path),
        "n_concepts": len(directions),
        "n_direction_vectors": len(norms),
        "max_unit_norm_error": max(abs(value - 1.0) for value in norms),
        "serialized_norm_tolerance": serialized_norm_tolerance,
        "storage_dtypes": storage_dtypes,
        "heldout_retrieval_top1": float(retrieval["estimate"]),
        "chance": float(stage1c["chance_retrieval"]),
        "known_answer_top5": float(explicit["estimate"]),
        "verification_scope": (
            "cached v2 held-out evidence plus artifact integrity; no refitting"
        ),
    }


def _g1(bundle: ModelBundle) -> dict[str, Any]:
    rows = hf_wrapper_logit_kl(bundle, G1_PROMPTS)
    maximum = max(float(row["mean_kl"]) for row in rows)
    return {
        "status": "PASS" if maximum < 1e-3 else "FAIL",
        "threshold": 1e-3,
        "n": len(rows),
        "max_prompt_mean_kl": maximum,
        "rows": rows,
    }


def run_stage0_reverify(
    bundle: ModelBundle,
    lens: Any,
    *,
    v2_metrics: Mapping[str, Any],
    workspace_layers: Sequence[int],
    preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the v3 environment and instrument-still-works gate."""

    preflight_result = dict(preflight) if preflight is not None else collect_preflight()
    items = load_calibration_items(bundle.tokenizer)
    bank = build_calibration_bank(bundle, lens, items, workspace_layers)
    swaps = run_known_swaps(bundle, items, bank, workspace_layers)
    controls = _direct_concept_controls(bundle, items, bank, list(workspace_layers))
    gdir = verify_gdir_artifact(v2_metrics)
    g1 = _g1(bundle)
    checks = {
        "preflight": preflight_result["status"] == "PASS",
        "hf_jlens_logits": g1["status"] == "PASS",
        "known_swaps": swaps["status"] == "PASS",
        "gdir_artifact": gdir["status"] == "PASS",
        "controls_fire": controls["status"] == "PASS",
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "preflight": preflight_result,
        "model": {
            "id": bundle.model_id,
            "revision": bundle.revision,
            "dtype": str(next(bundle.hf_model.parameters()).dtype),
        },
        "workspace_layers": [int(layer) for layer in workspace_layers],
        "g1": g1,
        "known_swaps": swaps,
        "gdir": gdir,
        "controls_fire": controls,
        "v2_source_commit": V2_SOURCE_COMMIT,
    }


def _stage0_report(stage0: Mapping[str, Any]) -> str:
    gpu = stage0["preflight"]["gpu"]
    disk = stage0["preflight"]["disk"]
    swaps = stage0["known_swaps"]
    return f"""# Surgical intervention calibration report (v3)

## Current verdict

**V3 CALIBRATION IN PROGRESS; SCIENCE PROHIBITED.** V2 established a working
three-case swap but failed capability and G-POS at alpha=2. V3 will sweep alpha
and carrying-position edits before any hypothesis test.

## Environment

- GPU: {gpu.get('name')}; {gpu.get('memory_total_mib')} MiB total; {gpu.get('memory_free_mib')} MiB free at preflight.
- Home/HF-cache filesystem: {disk.get('total_gib', float('nan')):.1f} GiB total; {disk.get('free_gib', float('nan')):.1f} GiB free.
- Required tool/auth preflight: **{stage0['preflight']['status']}**.
- Model: `{stage0['model']['id']}` at `{stage0['model']['revision']}` in `{stage0['model']['dtype']}`.

## Stage 0 — v2 instrument re-verification

- HF/J-Lens logit gate: **{stage0['g1']['status']}**; max mean KL={stage0['g1']['max_prompt_mean_kl']:.3e}, N={stage0['g1']['n']}.
- Known-answer alpha-2 swaps: **{swaps['status']}** ({swaps['n_pass']}/{swaps['n_required']}).
- Cached held-out G-DIR artifact: **{stage0['gdir']['status']}**; retrieval top-1={stage0['gdir']['heldout_retrieval_top1']:.3f}, known-answer top-5={stage0['gdir']['known_answer_top5']:.4f}.
- Non-structural direct suppression controls: **{stage0['controls_fire']['status']}**.

Stage-0 decision: **{stage0['status']}**. This licenses only G-SWAP confirmation
and the alpha sweep; it does not license Stage-2 recalibration or Stage-3 science.
"""


def persist_stage0(stage0: Mapping[str, Any]) -> dict[str, Any]:
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    repair_v2_sha256 = _repair_v2_sha256(metrics["repair_v2"])
    metrics["calibration_v3"] = {
        "schema_version": "calibrate-intervention-v3",
        "provenance": {
            "started_from_commit": V2_SOURCE_COMMIT,
            "repair_v2_sha256": repair_v2_sha256,
            "frozen_v2_gate_ledger": dict(
                metrics["repair_v2"]["gate_ledger"]
            ),
            "reused_artifacts": {
                "published_lens": "neuronpedia/jacobian-lens pinned in jlens_iface",
                "md_directions": stage0["gdir"]["artifact_path"],
                "md_sha256": stage0["gdir"]["artifact_sha256"],
            },
        },
        "protocol": {
            "model_id": MODEL_ID,
            "model_revision": stage0["model"]["revision"],
            "workspace_layers": stage0["workspace_layers"],
            "alpha_grid": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "alpha_extension_rule": (
                "if alpha=0.25 passes every gate, test 0.125 then 0.0625"
            ),
            "position_rule": (
                "original prompt position selected iff source-label J-Lens rank "
                "is <=10 at any workspace layer; empty masks remain no-op"
            ),
            "primary_edit": "project_out_transfer",
            "diagnostic_edit": "fractional_clamped_swap_all_positions",
            "thresholds": {
                "known_swap_top1": "3/3",
                "capability_abs_grand_and_bank_mean_delta_nll": 0.25,
                "g_pos_min_passages": 6,
                "g_pos_min_languages": 3,
                "g_pos_low_causal_abs_delta": 0.5,
                "g_pos_max_weight_read_ratio": 0.5,
                "random_empirical_p": 0.05,
                "absent_abs_null_over_real": 0.25,
            },
            "selection_rule": (
                "smallest alpha for the primary edit passing all four composite "
                "requirements; no threshold movement"
            ),
        },
        "stage0_reverify": stage0,
        "gate_ledger": {
            "stage0_reverify": stage0["status"],
            "g_swap": "NOT_RUN_V3",
            "g_alpha": "NOT_RUN_V3",
            "stage2_recalibration": "PROHIBITED",
            "stage3_science": "PROHIBITED",
            "stage4_report": "NOT_RUN_V3",
        },
        "current_allowed_conclusion": "V3_CALIBRATION_IN_PROGRESS_NO_SCIENCE",
    }
    save_json(metrics_path, metrics)
    (ROOT / "results" / "RESULTS.md").write_text(
        _stage0_report(stage0), encoding="utf-8"
    )
    return metrics


def run_stage1_confirm(
    bundle: ModelBundle,
    lens: Any,
    *,
    workspace_layers: Sequence[int],
) -> dict[str, Any]:
    items = load_calibration_items(bundle.tokenizer)
    bank = build_calibration_bank(bundle, lens, items, workspace_layers)
    swaps = run_known_swaps(bundle, items, bank, workspace_layers)
    return {
        "status": swaps["status"],
        "configuration": {
            "direction": "raw normalize(J.T @ W_U[token])",
            "strength": 2.0,
            "layers": [int(layer) for layer in workspace_layers],
            "positions": "all_prompt_positions",
            "token_resolution": "exact_label_first",
        },
        "g_swap": swaps,
    }


def persist_stage1(stage1: Mapping[str, Any]) -> dict[str, Any]:
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    v3 = metrics["calibration_v3"]
    if (
        _repair_v2_sha256(metrics["repair_v2"])
        != v3["provenance"]["repair_v2_sha256"]
    ):
        raise RuntimeError("Immutable repair_v2 provenance changed during v3")
    if v3["gate_ledger"]["stage0_reverify"] != "PASS":
        raise RuntimeError("V3 G-SWAP requires Stage-0 re-verification PASS")
    v3["stage1_confirm_swap"] = stage1
    v3["gate_ledger"]["g_swap"] = stage1["status"]
    v3["gate_ledger"]["g_alpha"] = (
        "PENDING" if stage1["status"] == "PASS" else "SKIPPED_PREREQUISITE"
    )
    if stage1["status"] != "PASS":
        v3["gate_ledger"]["stage4_report"] = "REQUIRED"
        v3["current_allowed_conclusion"] = "V3_GSWAP_FAILURE_NO_SCIENCE"
    save_json(metrics_path, metrics)
    rows = "\n".join(
        "| {name} | `{source}` -> `{target}` | `{clean}` | `{edited}` | "
        "{clean_m:.3f} | {edited_m:.3f} | {status} |".format(
            name=row["name"],
            source=row["source"],
            target=row["target"],
            clean=row["clean_top"],
            edited=row["edited_top"],
            clean_m=row["clean_metric"],
            edited_m=row["edited_metric"],
            status="PASS" if row["pass"] else "FAIL",
        )
        for row in stage1["g_swap"]["rows"]
    )
    report_path = ROOT / "results" / "RESULTS.md"
    report = report_path.read_text(encoding="utf-8")
    report += f"""

## Stage 1 — G-SWAP confirmation

| item | concept swap | clean top-1 | edited top-1 | clean M | edited M | gate |
| --- | --- | --- | --- | ---: | ---: | --- |
{rows}

**G-SWAP {stage1['status']} ({stage1['g_swap']['n_pass']}/{stage1['g_swap']['n_required']}).**
The next permitted step is the surgical alpha sweep. Science remains prohibited.
"""
    report_path.write_text(report, encoding="utf-8")
    return metrics

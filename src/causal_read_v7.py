"""Matched-metric causal ground truth for v7.

The primary full-residual measurement delegates to the frozen low-level
interchange in :mod:`src.causal_read`.  Both engine and dashboard receive the
same final-position ``logit(answer_A) - logit(answer_B)`` metric.  This module
also restores the historical two-J-Lens-direction subspace clamp as an
explicitly diagnostic variant, without changing the frozen causal module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import nn

from src.causal_read import (
    clean_state_and_logits,
    symmetric_interchange,
    token_difference_metric,
)
from src.data_gen_v7 import (
    MANIFEST_PATH,
    METRIC_DEFINITION,
    POSITION_RULE,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    SELECTED_LAYER,
    build_direction_bank_v7,
    save_json,
    sha256_file,
)
from src.datasets import validate_sanitized_manifest
from src.jlens_interface import MODEL_ID, MODEL_REVISION, load_model, release_model, set_seed


CAUSAL_PATH = RESULTS_DIR / "raw" / "causal_C_v7.json"
PRIMARY_VARIANT = "full_residual"
SUBSPACE_VARIANT = "jlens_two_concept_subspace"
SHARP_DISAGREEMENT_THRESHOLD = 0.50
FULL_ENGINE_LARGE_THRESHOLD = 0.50
FULL_DASHBOARD_NEAR_ZERO_THRESHOLD = 0.10
MAX_SUBSPACE_CONDITION = 1e4

MetricFn = Callable[[torch.Tensor], torch.Tensor]
TensorEdit = Callable[[torch.Tensor], torch.Tensor]
ProgressFn = Callable[[int, int, Mapping[str, Any]], None]


def _position_index(sequence_length: int, position: int) -> int:
    index = int(position)
    if index < 0:
        index += sequence_length
    if not 0 <= index < sequence_length:
        raise IndexError(f"Position {position} outside sequence length {sequence_length}")
    return index


def _hidden_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported decoder output type {type(output).__name__}")


def _replace_hidden(output: Any, edit: TensorEdit) -> Any:
    if torch.is_tensor(output):
        return edit(output)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (edit(output[0]), *output[1:])
    raise TypeError(f"Unsupported decoder output type {type(output).__name__}")


@contextmanager
def _residual_edit_hooks(
    blocks: Sequence[nn.Module], edits: Mapping[int, TensorEdit]
) -> Iterator[None]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for raw_layer, edit in sorted(edits.items()):
            layer = int(raw_layer)
            if not 0 <= layer < len(blocks):
                raise IndexError(f"Layer {layer} outside decoder range")

            def hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                _edit: TensorEdit = edit,
            ) -> Any:
                return _replace_hidden(output, _edit)

            handles.append(blocks[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def _forward_logits(
    hf_model: nn.Module,
    input_ids: torch.Tensor,
    *,
    blocks: Sequence[nn.Module],
    edits: Mapping[int, TensorEdit],
) -> torch.Tensor:
    with _residual_edit_hooks(blocks, edits):
        return hf_model(input_ids=input_ids, use_cache=False).logits.float()


def _validated_basis(
    donor_state: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    *,
    max_condition: float = MAX_SUBSPACE_CONDITION,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    donor = donor_state.detach().float()
    basis = torch.stack(
        [
            direction_a.detach().to(donor.device, torch.float32),
            direction_b.detach().to(donor.device, torch.float32),
        ],
        dim=0,
    )
    if donor.ndim != 1 or basis.shape != (2, donor.numel()):
        raise ValueError("Donor and two-concept basis have incompatible shapes")
    norms = basis.norm(dim=-1)
    if not torch.isfinite(basis).all() or not torch.allclose(
        norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4
    ):
        raise ValueError(f"Concept directions must be finite unit vectors: {norms}")
    gram = basis @ basis.T
    condition = float(torch.linalg.cond(gram).cpu())
    if not math.isfinite(condition) or condition > float(max_condition):
        raise ValueError(f"Concept subspace is ill-conditioned: {condition:.6g}")
    return basis, torch.linalg.inv(gram), condition


def concept_subspace_interchange_edit_v7(
    donor_state: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    *,
    position: int,
    max_condition: float = MAX_SUBSPACE_CONDITION,
) -> TensorEdit:
    """Clamp donor coordinates only in ``span(v_A, v_B)``."""

    donor = donor_state.detach().float()
    basis, inverse_gram, _condition = _validated_basis(
        donor,
        direction_a,
        direction_b,
        max_condition=max_condition,
    )
    donor_dot_products = donor @ basis.T

    def edit(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != donor.numel():
            raise ValueError("Subspace interchange expects hidden [1,S,D]")
        index = _position_index(hidden.shape[1], position)
        live_basis = basis.to(hidden.device, torch.float32)
        live_inverse = inverse_gram.to(hidden.device, torch.float32)
        current = hidden[0, index].float()
        current_dot_products = current @ live_basis.T
        desired = donor_dot_products.to(hidden.device, torch.float32)
        correction = (desired - current_dot_products) @ live_inverse @ live_basis
        output = hidden.clone()
        output[0, index] = (current + correction).to(hidden.dtype)
        return output

    return edit


def _validated_clean_record(
    record: Mapping[str, Any],
    input_ids: torch.Tensor,
    *,
    side: str,
    layer: int,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if record.get("input_token_ids") != input_ids.detach().cpu().tolist():
        raise ValueError(f"Clean {side} record belongs to different input tokens")
    if int(record.get("layer", -1)) != int(layer):
        raise ValueError(f"Clean {side} record belongs to a different layer")
    if int(record.get("sequence_length", -1)) != int(input_ids.shape[1]):
        raise ValueError(f"Clean {side} record belongs to a different sequence length")
    expected_position = _position_index(input_ids.shape[1], position)
    if int(record.get("resolved_position", -1)) != expected_position:
        raise ValueError(f"Clean {side} record belongs to a different position")
    state = record.get("state")
    logits = record.get("logits")
    if not torch.is_tensor(state) or state.ndim != 1 or not torch.isfinite(state).all():
        raise ValueError(f"Clean {side} record has an invalid state")
    if (
        not torch.is_tensor(logits)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or not torch.isfinite(logits).all()
    ):
        raise ValueError(f"Clean {side} record has invalid logits")
    return state, logits.float()


@torch.no_grad()
def symmetric_subspace_interchange_v7(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    clean_record_a: Mapping[str, Any],
    clean_record_b: Mapping[str, Any],
    metric_fn: MetricFn,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    *,
    pair_id: str,
    task_kind: str,
    layer: int,
    position_a: int,
    position_b: int,
    normalization_t: float | None = None,
    sharp_disagreement_threshold: float = SHARP_DISAGREEMENT_THRESHOLD,
) -> dict[str, Any]:
    """Measure signed, unclipped symmetric C for the two-concept subspace."""

    if hf_model.training:
        raise ValueError("Causal interchange requires eval mode")
    if not pair_id or task_kind not in {"engine", "dashboard"}:
        raise ValueError("Valid pair_id and task_kind are required")
    if not 0 <= int(layer) < len(blocks):
        raise ValueError("Invalid interchange layer")
    state_a, clean_a = _validated_clean_record(
        clean_record_a,
        input_ids_a,
        side="A",
        layer=layer,
        position=position_a,
    )
    state_b, clean_b = _validated_clean_record(
        clean_record_b,
        input_ids_b,
        side="B",
        layer=layer,
        position=position_b,
    )
    _basis, _inverse, condition = _validated_basis(
        state_a, direction_a, direction_b
    )
    if state_b.numel() != state_a.numel():
        raise ValueError("A/B clean state dimensions differ")

    metric_a = float(metric_fn(clean_a).cpu())
    metric_b = float(metric_fn(clean_b).cpu())
    if task_kind == "engine":
        if normalization_t is not None:
            raise ValueError("Engine T must come from its own clean metrics")
        scale = metric_a - metric_b
        normalization_source = "engine_clean_metric_a_minus_clean_metric_b"
    else:
        if normalization_t is None:
            raise ValueError("Dashboard C requires matched engine T")
        scale = float(normalization_t)
        normalization_source = "matched_engine_clean_T"
    if not math.isfinite(scale) or abs(scale) <= 1e-8:
        raise ValueError(f"Causal normalization T must be finite and nonzero: {scale}")

    logits_a_from_b = _forward_logits(
        hf_model,
        input_ids_a,
        blocks=blocks,
        edits={
            int(layer): concept_subspace_interchange_edit_v7(
                state_b,
                direction_a,
                direction_b,
                position=position_a,
            )
        },
    )
    logits_b_from_a = _forward_logits(
        hf_model,
        input_ids_b,
        blocks=blocks,
        edits={
            int(layer): concept_subspace_interchange_edit_v7(
                state_a,
                direction_a,
                direction_b,
                position=position_b,
            )
        },
    )
    metric_a_from_b = float(metric_fn(logits_a_from_b).cpu())
    metric_b_from_a = float(metric_fn(logits_b_from_a).cpu())
    r_a_from_b = (metric_a - metric_a_from_b) / scale
    r_b_from_a = (metric_b_from_a - metric_b) / scale
    causal_c = 0.5 * (r_a_from_b + r_b_from_a)
    disagreement = abs(r_a_from_b - r_b_from_a)
    finite_values = (
        metric_a,
        metric_b,
        metric_a_from_b,
        metric_b_from_a,
        scale,
        r_a_from_b,
        r_b_from_a,
        causal_c,
        disagreement,
        condition,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("Subspace interchange produced non-finite output")
    return {
        "status": "OK",
        "pair_id": str(pair_id),
        "task_kind": task_kind,
        "variant": SUBSPACE_VARIANT,
        "eligible_as_primary_truth": False,
        "layer": int(layer),
        "position_rule": POSITION_RULE,
        "position_a": int(position_a),
        "position_b": int(position_b),
        "metric_a": metric_a,
        "metric_b": metric_b,
        "metric_a_from_b": metric_a_from_b,
        "metric_b_from_a": metric_b_from_a,
        "T": scale,
        "normalization_source": normalization_source,
        "R_a_from_b": r_a_from_b,
        "R_b_from_a": r_b_from_a,
        "C": causal_c,
        "directional_abs_difference": disagreement,
        "sharp_disagreement_threshold": float(sharp_disagreement_threshold),
        "sharp_directional_disagreement": disagreement > sharp_disagreement_threshold,
        "signed_unclipped": True,
        "subspace_definition": "span(v_A, v_B); donor coordinate clamp",
        "subspace_gram_condition": condition,
        "subspace_max_condition": MAX_SUBSPACE_CONDITION,
        "clean_top_token_id_a": int(clean_a[0, -1].argmax().cpu()),
        "clean_top_token_id_b": int(clean_b[0, -1].argmax().cpu()),
        "edited_top_token_id_a_from_b": int(logits_a_from_b[0, -1].argmax().cpu()),
        "edited_top_token_id_b_from_a": int(logits_b_from_a[0, -1].argmax().cpu()),
    }


def _encode(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    values = tokenizer.encode(
        str(prompt), add_special_tokens=False, return_tensors="pt"
    )
    if not torch.is_tensor(values) or values.ndim != 2 or values.shape[0] != 1:
        raise TypeError("Tokenizer did not return one prompt row")
    return values.to(device)


def _assert_clean_targets(
    measurement: Mapping[str, Any],
    *,
    expected_a: int,
    expected_b: int,
    label: str,
) -> None:
    if int(measurement["clean_top_token_id_a"]) != int(expected_a):
        raise RuntimeError(f"{label} clean A target drifted")
    if int(measurement["clean_top_token_id_b"]) != int(expected_b):
        raise RuntimeError(f"{label} clean B target drifted")


@torch.no_grad()
def compute_causal_rows_v7(
    bundle: Any,
    rows: Sequence[Mapping[str, Any]],
    directions: Mapping[int, torch.Tensor],
    *,
    layer: int = SELECTED_LAYER,
    progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Compute both variants for both tasks with one identical metric per pair."""

    if not rows:
        raise ValueError("Causal v7 requires verified rows")
    blocks = bundle.lens_model.layers
    device = next(bundle.hf_model.parameters()).device
    output: list[dict[str, Any]] = []
    for completed, row in enumerate(rows, start=1):
        if row.get("verification_status") != "VERIFIED":
            raise ValueError(f"{row.get('pair_id')} is not VERIFIED")
        pair_id = str(row["pair_id"])
        positive_id = int(row["answer_a_token_id"])
        negative_id = int(row["answer_b_token_id"])
        if positive_id != int(row["metric_positive_token_id"]):
            raise ValueError(f"{pair_id} positive metric ID drifted")
        if negative_id != int(row["metric_negative_token_id"]):
            raise ValueError(f"{pair_id} negative metric ID drifted")
        if row["engine_metric"] != row["dashboard_metric"]:
            raise ValueError(f"{pair_id} metric definitions differ")
        metric_fn = token_difference_metric(positive_id, negative_id)
        direction_a = directions[int(row["concept_a_token_id"])]
        direction_b = directions[int(row["concept_b_token_id"])]

        engine_ids_a = _encode(bundle.tokenizer, str(row["engine_prompt_a"]), device)
        engine_ids_b = _encode(bundle.tokenizer, str(row["engine_prompt_b"]), device)
        engine_position_a = int(row["engine_position_a"])
        engine_position_b = int(row["engine_position_b"])
        engine_clean_a = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            engine_ids_a,
            layer,
            position=engine_position_a,
        )
        engine_clean_b = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            engine_ids_b,
            layer,
            position=engine_position_b,
        )
        engine_full = symmetric_interchange(
            bundle.hf_model,
            blocks,
            engine_ids_a,
            engine_ids_b,
            engine_clean_a,
            engine_clean_b,
            metric_fn,
            pair_id=pair_id,
            task_kind="engine",
            layer=layer,
            position_a=engine_position_a,
            position_b=engine_position_b,
            sharp_disagreement_threshold=SHARP_DISAGREEMENT_THRESHOLD,
        )
        engine_subspace = symmetric_subspace_interchange_v7(
            bundle.hf_model,
            blocks,
            engine_ids_a,
            engine_ids_b,
            engine_clean_a,
            engine_clean_b,
            metric_fn,
            direction_a,
            direction_b,
            pair_id=pair_id,
            task_kind="engine",
            layer=layer,
            position_a=engine_position_a,
            position_b=engine_position_b,
        )
        if not math.isclose(
            float(engine_full["T"]),
            float(engine_subspace["T"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise AssertionError(f"{pair_id} engine T differs across variants")
        _assert_clean_targets(
            engine_full,
            expected_a=positive_id,
            expected_b=negative_id,
            label=f"{pair_id} engine full",
        )
        _assert_clean_targets(
            engine_subspace,
            expected_a=positive_id,
            expected_b=negative_id,
            label=f"{pair_id} engine subspace",
        )

        dashboard_ids_a = _encode(
            bundle.tokenizer, str(row["dashboard_prompt_a"]), device
        )
        dashboard_ids_b = _encode(
            bundle.tokenizer, str(row["dashboard_prompt_b"]), device
        )
        dashboard_position_a = int(row["dashboard_position_a"])
        dashboard_position_b = int(row["dashboard_position_b"])
        dashboard_clean_a = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            dashboard_ids_a,
            layer,
            position=dashboard_position_a,
        )
        dashboard_clean_b = clean_state_and_logits(
            bundle.hf_model,
            blocks,
            dashboard_ids_b,
            layer,
            position=dashboard_position_b,
        )
        dashboard_full = symmetric_interchange(
            bundle.hf_model,
            blocks,
            dashboard_ids_a,
            dashboard_ids_b,
            dashboard_clean_a,
            dashboard_clean_b,
            metric_fn,
            pair_id=pair_id,
            task_kind="dashboard",
            layer=layer,
            position_a=dashboard_position_a,
            position_b=dashboard_position_b,
            normalization_t=float(engine_full["T"]),
            sharp_disagreement_threshold=SHARP_DISAGREEMENT_THRESHOLD,
        )
        dashboard_subspace = symmetric_subspace_interchange_v7(
            bundle.hf_model,
            blocks,
            dashboard_ids_a,
            dashboard_ids_b,
            dashboard_clean_a,
            dashboard_clean_b,
            metric_fn,
            direction_a,
            direction_b,
            pair_id=pair_id,
            task_kind="dashboard",
            layer=layer,
            position_a=dashboard_position_a,
            position_b=dashboard_position_b,
            normalization_t=float(engine_subspace["T"]),
        )
        for label, measurement in (
            ("dashboard full", dashboard_full),
            ("dashboard subspace", dashboard_subspace),
        ):
            _assert_clean_targets(
                measurement,
                expected_a=positive_id,
                expected_b=negative_id,
                label=f"{pair_id} {label}",
            )

        result = {
            "pair_id": pair_id,
            "dependency_group": str(row["dependency_group"]),
            "fold": int(row["fold"]),
            "category": str(row["category"]),
            "concept_a": str(row["concept_a"]),
            "concept_b": str(row["concept_b"]),
            "answer_a": str(row["answer_a"]),
            "answer_b": str(row["answer_b"]),
            "verification_status": "VERIFIED",
            "metric": METRIC_DEFINITION,
            "metric_positive_token_id": positive_id,
            "metric_negative_token_id": negative_id,
            "same_metric_in_both_conditions": True,
            "engine": {
                PRIMARY_VARIANT: engine_full,
                SUBSPACE_VARIANT: engine_subspace,
            },
            "dashboard": {
                PRIMARY_VARIANT: dashboard_full,
                SUBSPACE_VARIANT: dashboard_subspace,
            },
        }
        output.append(result)
        if progress is not None:
            progress(
                completed,
                len(rows),
                {
                    "pair_id": pair_id,
                    "engine_full_C": float(engine_full["C"]),
                    "dashboard_full_C": float(dashboard_full["C"]),
                    "engine_subspace_C": float(engine_subspace["C"]),
                    "dashboard_subspace_C": float(dashboard_subspace["C"]),
                },
            )
    return output


def summarize_causal_v7(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize signed and absolute C without clipping any row."""

    if not rows:
        raise ValueError("Causal summary requires rows")
    summary: dict[str, Any] = {
        "n_pairs": len(rows),
        "primary_truth": PRIMARY_VARIANT,
        "signed_unclipped": True,
        "full_state_gate_thresholds": {
            "engine_abs_C_median_gt": FULL_ENGINE_LARGE_THRESHOLD,
            "dashboard_abs_C_median_lt": FULL_DASHBOARD_NEAR_ZERO_THRESHOLD,
        },
    }
    for variant in (PRIMARY_VARIANT, SUBSPACE_VARIANT):
        engine = [float(row["engine"][variant]["C"]) for row in rows]
        dashboard = [float(row["dashboard"][variant]["C"]) for row in rows]
        summary[variant] = {
            "engine_C_median": float(median(engine)),
            "engine_abs_C_median": float(median(abs(value) for value in engine)),
            "engine_C_min": min(engine),
            "engine_C_max": max(engine),
            "dashboard_C_median": float(median(dashboard)),
            "dashboard_abs_C_median": float(
                median(abs(value) for value in dashboard)
            ),
            "dashboard_C_min": min(dashboard),
            "dashboard_C_max": max(dashboard),
            "engine_sharp_directional_disagreements": sum(
                bool(row["engine"][variant]["sharp_directional_disagreement"])
                for row in rows
            ),
            "dashboard_sharp_directional_disagreements": sum(
                bool(row["dashboard"][variant]["sharp_directional_disagreement"])
                for row in rows
            ),
        }
    full = summary[PRIMARY_VARIANT]
    engine_large = full["engine_abs_C_median"] > FULL_ENGINE_LARGE_THRESHOLD
    dashboard_near_zero = (
        full["dashboard_abs_C_median"] < FULL_DASHBOARD_NEAR_ZERO_THRESHOLD
    )
    summary["full_state_engine_large"] = engine_large
    summary["full_state_dashboard_near_zero"] = dashboard_near_zero
    summary["full_state_sanity_status"] = (
        "PASS" if engine_large and dashboard_near_zero else "FAIL"
    )
    return summary


def run_causal_stage_v7(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    output_path: str | Path = CAUSAL_PATH,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Execute v7_2 from the clean sanitized manifest only."""

    set_seed(SEED)
    source_path = Path(manifest_path)
    manifest = validate_sanitized_manifest(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    verified = [
        row for row in manifest["rows"] if row["verification_status"] == "VERIFIED"
    ]
    if len(verified) < 50:
        raise RuntimeError("Fewer than 50 verified matched pairs reached causal v7")
    if manifest["metric_contract"]["engine"] != manifest["metric_contract"]["dashboard"]:
        raise ValueError("Manifest conditions do not share a metric")
    selection = manifest["selection"]
    if int(selection["layer"]) != SELECTED_LAYER:
        raise ValueError("Manifest layer differs from frozen L16")
    if str(selection["position_rule"]) != POSITION_RULE:
        raise ValueError("Manifest position rule differs from the frozen rule")
    if not math.isclose(
        float(selection["written_threshold"]),
        2.482430934906006,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Manifest WRITTEN threshold differs from frozen calibration")

    bundle = load_model(local_files_only=True)
    try:
        token_ids = [
            int(value) for value in manifest["direction_provenance"]["token_ids"]
        ]
        directions, direction_provenance = build_direction_bank_v7(
            bundle, token_ids, layer=SELECTED_LAYER
        )
        expected_digest = str(
            manifest["direction_provenance"]["ordered_tensor_digest_sha256"]
        )
        observed_digest = str(direction_provenance["ordered_tensor_digest_sha256"])
        if observed_digest != expected_digest:
            raise RuntimeError("Reconstructed J-Lens directions differ from v7_1")
        rows = compute_causal_rows_v7(
            bundle,
            verified,
            directions,
            layer=SELECTED_LAYER,
            progress=progress,
        )
        sanity = summarize_causal_v7(rows)
        artifact = {
            "schema_version": "matched-causal-read-v7-v1",
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "dtype": "torch.bfloat16",
            },
            "selected_layer": SELECTED_LAYER,
            "position_rule": POSITION_RULE,
            "metric_contract": {
                "engine": METRIC_DEFINITION,
                "dashboard": METRIC_DEFINITION,
                "identical_token_ids_enforced": True,
            },
            "primary_truth": PRIMARY_VARIANT,
            "subspace_variant": SUBSPACE_VARIANT,
            "signed_unclipped": True,
            "source_manifest": {
                "path": str(source_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(source_path),
            },
            "frozen_causal_module": {
                "path": "src/causal_read.py",
                "sha256": sha256_file(PROJECT_ROOT / "src" / "causal_read.py"),
                "full_state_function": "src.causal_read.symmetric_interchange",
                "modified": False,
            },
            "direction_provenance": direction_provenance,
            "causal_sanity": sanity,
            "rows": rows,
        }
        destination = save_json(output_path, artifact)
        return {
            "causal_path": str(destination),
            "causal_sha256": sha256_file(destination),
            "n_pairs": len(rows),
            "causal_sanity": sanity,
        }
    finally:
        release_model(bundle)


__all__ = [
    "CAUSAL_PATH",
    "PRIMARY_VARIANT",
    "SUBSPACE_VARIANT",
    "compute_causal_rows_v7",
    "concept_subspace_interchange_edit_v7",
    "run_causal_stage_v7",
    "summarize_causal_v7",
    "symmetric_subspace_interchange_v7",
]

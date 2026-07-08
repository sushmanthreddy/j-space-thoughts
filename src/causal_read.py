"""Expensive causal truth and post-GO signed mediation for symmetric READ.

This module owns donor-state interventions.  The cheap estimator intentionally
lives in :mod:`src.cheap_read` and must not import this module or consume any
value produced here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from src.interventions import forward_logits, residual_edit_hooks


MetricFn = Callable[[torch.Tensor], torch.Tensor]


def token_difference_metric(
    positive_token_id: int,
    negative_token_id: int,
) -> MetricFn:
    """Return the final-position logit difference used by one matched task."""

    positive = int(positive_token_id)
    negative = int(negative_token_id)
    if positive == negative:
        raise ValueError("Behavior metric tokens must differ")

    def metric(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("Symmetric causal metrics require logits shaped [1,S,V]")
        return logits[0, -1, positive].float() - logits[0, -1, negative].float()

    return metric


@torch.no_grad()
def clean_state_and_logits(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids: torch.Tensor,
    layer: int,
    *,
    position: int = -1,
) -> dict[str, torch.Tensor]:
    """Capture one clean post-block state and logits in the same HF forward."""

    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        hidden = output if torch.is_tensor(output) else output[0]
        if not torch.is_tensor(hidden):
            raise TypeError("Decoder block did not return a tensor hidden state")
        captured["hidden"] = hidden.detach()
        return output

    handle = blocks[int(layer)].register_forward_hook(hook)
    try:
        logits = hf_model(input_ids=input_ids, use_cache=False).logits.float()
    finally:
        handle.remove()
    residual = captured.get("hidden")
    if residual is None:
        raise RuntimeError("Selected-layer clean state was not captured")
    index = _position_index(residual.shape[1], position)
    if residual.shape[0] != 1:
        raise ValueError("Clean donor capture requires batch size one")
    state = residual[0, index].detach().float()
    if not torch.isfinite(state).all():
        raise ValueError("Clean residual state contains non-finite values")
    return {
        "state": state,
        "logits": logits,
        "layer": int(layer),
        "requested_position": int(position),
        "resolved_position": int(index),
        "sequence_length": int(input_ids.shape[1]),
        "input_token_ids": input_ids.detach().cpu().tolist(),
    }


def _position_index(sequence_length: int, position: int) -> int:
    index = int(position)
    if index < 0:
        index += sequence_length
    if not 0 <= index < sequence_length:
        raise IndexError(f"Position {position} outside sequence length {sequence_length}")
    return index


def full_residual_interchange_edit(
    donor_state: torch.Tensor,
    *,
    position: int = -1,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a hook edit that copies a clean donor's complete residual state."""

    donor = donor_state.detach().float()
    if donor.ndim != 1 or not torch.isfinite(donor).all():
        raise ValueError("donor_state must be one finite vector")

    def edit(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != donor.numel():
            raise ValueError("Full interchange expects hidden [1,S,D] matching the donor")
        index = _position_index(hidden.shape[1], position)
        output = hidden.clone()
        output[0, index] = donor.to(hidden.device, hidden.dtype)
        return output

    return edit


def concept_subspace_interchange_edit(
    donor_state: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    *,
    position: int = -1,
    max_condition: float = 1e4,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Copy only donor dot-products in the two-direction J-Lens subspace."""

    donor = donor_state.detach().float()
    basis = torch.stack(
        [
            direction_a.detach().to(donor.device, torch.float32),
            direction_b.detach().to(donor.device, torch.float32),
        ],
        dim=0,
    )
    if donor.ndim != 1 or basis.ndim != 2 or basis.shape[1] != donor.numel():
        raise ValueError("Donor and concept directions have incompatible shapes")
    norms = basis.norm(dim=-1)
    if not torch.isfinite(basis).all() or not torch.allclose(
        norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4
    ):
        raise ValueError(f"Concept directions must be finite unit vectors: {norms}")
    gram = basis @ basis.T
    condition = float(torch.linalg.cond(gram))
    if not math.isfinite(condition) or condition > max_condition:
        raise ValueError(f"Concept subspace is ill-conditioned: {condition:.6g}")
    inverse_gram = torch.linalg.inv(gram)
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


@torch.no_grad()
def symmetric_interchange(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    clean_record_a: dict[str, Any],
    clean_record_b: dict[str, Any],
    metric_fn: MetricFn,
    *,
    pair_id: str,
    task_kind: str,
    layer: int,
    normalization_t: float | None = None,
    variant: str = "full_residual",
    direction_a: torch.Tensor | None = None,
    direction_b: torch.Tensor | None = None,
    position_a: int,
    position_b: int,
    sharp_disagreement_threshold: float = 0.50,
) -> dict[str, Any]:
    """Measure signed two-direction interchange and return unclipped C."""

    if variant not in {"full_residual", "jlens_two_concept_subspace"}:
        raise ValueError(f"Unknown interchange variant {variant!r}")
    if hf_model.training:
        raise ValueError("Causal interchange requires eval mode")
    if not pair_id:
        raise ValueError("pair_id is required for causal provenance")
    if task_kind not in {"engine", "dashboard"}:
        raise ValueError("task_kind must be 'engine' or 'dashboard'")
    if not isinstance(layer, int) or not 0 <= layer < len(blocks):
        raise ValueError(f"Invalid interchange layer {layer!r}")
    if not math.isfinite(sharp_disagreement_threshold) or sharp_disagreement_threshold < 0:
        raise ValueError("Directional-disagreement threshold must be finite and nonnegative")

    def validate_clean_record(
        record: dict[str, Any], input_ids: torch.Tensor, side: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_ids = input_ids.detach().cpu().tolist()
        if record.get("input_token_ids") != expected_ids:
            raise ValueError(f"Clean {side} record belongs to different input tokens")
        if int(record.get("layer", -1)) != layer:
            raise ValueError(f"Clean {side} record belongs to a different layer")
        requested_position = position_a if side == "A" else position_b
        expected_position = _position_index(input_ids.shape[1], requested_position)
        if int(record.get("resolved_position", -1)) != expected_position:
            raise ValueError(f"Clean {side} record belongs to a different position")
        state = record.get("state")
        logits = record.get("logits")
        if not torch.is_tensor(state) or state.ndim != 1 or not torch.isfinite(state).all():
            raise ValueError(f"Clean {side} record has an invalid residual state")
        if not torch.is_tensor(logits) or logits.ndim != 3 or not torch.isfinite(logits).all():
            raise ValueError(f"Clean {side} record has invalid logits")
        return state, logits.float()

    state_a, clean_a = validate_clean_record(clean_record_a, input_ids_a, "A")
    state_b, clean_b = validate_clean_record(clean_record_b, input_ids_b, "B")
    metric_a = float(metric_fn(clean_a).cpu())
    metric_b = float(metric_fn(clean_b).cpu())
    if task_kind == "engine":
        if normalization_t is not None:
            raise ValueError("Engine T must be derived from its own two clean metrics")
        scale = metric_a - metric_b
        normalization_source = "engine_clean_metric_a_minus_clean_metric_b"
    else:
        if normalization_t is None:
            raise ValueError("Dashboard C requires the matched engine T")
        scale = float(normalization_t)
        normalization_source = "matched_engine_clean_T"
    if not math.isfinite(scale) or abs(scale) <= 1e-8:
        raise ValueError(f"Symmetric normalization T must be finite and nonzero: {scale}")

    if variant == "full_residual":
        edit_a_from_b = full_residual_interchange_edit(state_b, position=position_a)
        edit_b_from_a = full_residual_interchange_edit(state_a, position=position_b)
    else:
        if direction_a is None or direction_b is None:
            raise ValueError("Subspace interchange requires both concept directions")
        edit_a_from_b = concept_subspace_interchange_edit(
            state_b, direction_a, direction_b, position=position_a
        )
        edit_b_from_a = concept_subspace_interchange_edit(
            state_a, direction_a, direction_b, position=position_b
        )

    logits_a_from_b = forward_logits(
        hf_model,
        input_ids_a,
        blocks=blocks,
        edits={int(layer): edit_a_from_b},
    )
    logits_b_from_a = forward_logits(
        hf_model,
        input_ids_b,
        blocks=blocks,
        edits={int(layer): edit_b_from_a},
    )
    metric_a_from_b = float(metric_fn(logits_a_from_b).cpu())
    metric_b_from_a = float(metric_fn(logits_b_from_a).cpu())
    r_a_from_b = (metric_a - metric_a_from_b) / scale
    r_b_from_a = (metric_b_from_a - metric_b) / scale
    causal_c = 0.5 * (r_a_from_b + r_b_from_a)
    disagreement = abs(r_a_from_b - r_b_from_a)
    finite_values = [
        metric_a,
        metric_b,
        metric_a_from_b,
        metric_b_from_a,
        scale,
        r_a_from_b,
        r_b_from_a,
        causal_c,
        disagreement,
    ]
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("Symmetric interchange produced a non-finite scalar")
    return {
        "status": "OK",
        "pair_id": pair_id,
        "task_kind": task_kind,
        "variant": variant,
        "eligible_as_primary_truth": variant == "full_residual",
        "layer": int(layer),
        "position_rule": "explicit_concept_token_in_shared_context",
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
        "sharp_disagreement_threshold": sharp_disagreement_threshold,
        "sharp_directional_disagreement": disagreement > sharp_disagreement_threshold,
        "signed_unclipped": True,
        "clean_top_token_id_a": int(clean_a[0, -1].argmax().cpu()),
        "clean_top_token_id_b": int(clean_b[0, -1].argmax().cpu()),
        "edited_top_token_id_a_from_b": int(logits_a_from_b[0, -1].argmax().cpu()),
        "edited_top_token_id_b_from_a": int(logits_b_from_a[0, -1].argmax().cpu()),
    }


def signed_component_mediation(
    clean_metric: float,
    edited_metric: float,
    restored_component_metric: float,
) -> dict[str, float]:
    """Compute signed component mediation after a validated concept-level GO."""

    denominator = float(clean_metric) - float(edited_metric)
    if not math.isfinite(denominator) or abs(denominator) <= 1e-8:
        raise ValueError("Signed mediation denominator is zero or non-finite")
    mediation_effect = float(restored_component_metric) - float(edited_metric)
    return {
        "ME_k": mediation_effect,
        "READ_k": mediation_effect / denominator,
        "denominator_clean_minus_edited": denominator,
    }


def circuit_faithfulness(
    clean_metric: float,
    edited_metric: float,
    outside_ablated_clean_metric: float,
    outside_ablated_edited_metric: float,
) -> dict[str, float]:
    """Normalize the source effect surviving after everything outside a circuit is ablated."""

    full_effect = float(clean_metric) - float(edited_metric)
    if not math.isfinite(full_effect) or abs(full_effect) <= 1e-8:
        raise ValueError("Faithfulness denominator is zero or non-finite")
    circuit_only_effect = (
        float(outside_ablated_clean_metric) - float(outside_ablated_edited_metric)
    )
    return {
        "full_effect": full_effect,
        "circuit_only_effect": circuit_only_effect,
        "faithfulness_fraction": circuit_only_effect / full_effect,
    }


def _component_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported component output type {type(output).__name__}")


def _replace_component_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (hidden, *output[1:])
    raise TypeError(f"Unsupported component output type {type(output).__name__}")


def _component_modules(
    blocks: Sequence[nn.Module],
    component_layers: Sequence[int],
) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {}
    for layer in sorted(set(int(value) for value in component_layers)):
        if not 0 <= layer < len(blocks):
            raise IndexError(f"Component layer {layer} outside model")
        modules[f"L{layer}.ATTN"] = blocks[layer].self_attn
        modules[f"L{layer}.MLP"] = blocks[layer].mlp
    if not modules:
        raise ValueError("At least one downstream component layer is required")
    return modules


@contextmanager
def _restore_component_outputs(
    modules: dict[str, nn.Module],
    clean_outputs: dict[str, torch.Tensor],
    component_ids: Sequence[str],
):
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for component_id in component_ids:
            module = modules[component_id]
            clean = clean_outputs[component_id]

            def hook(
                hooked_module: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                *,
                _clean=clean,
            ) -> Any:
                del hooked_module, inputs
                current = _component_hidden(output)
                if current.shape != _clean.shape:
                    raise ValueError("Clean restoration component shape changed")
                restored = _clean.to(current.device, current.dtype)
                return _replace_component_hidden(output, restored)

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _zero_component_outputs(
    modules: dict[str, nn.Module],
    component_ids: Sequence[str],
):
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for component_id in component_ids:
            module = modules[component_id]

            def hook(
                hooked_module: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
            ) -> Any:
                del hooked_module, inputs
                current = _component_hidden(output)
                return _replace_component_hidden(output, torch.zeros_like(current))

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def localize_signed_mediation(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids: torch.Tensor,
    metric_fn: MetricFn,
    source_edits: dict[int, Callable[[torch.Tensor], torch.Tensor]],
    *,
    component_layers: Sequence[int],
    circuit_size: int = 8,
    go_authorized: bool,
) -> dict[str, Any]:
    """Restore downstream components individually, then test top-k faithfulness."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Signed mediation expects one prompt")
    if not go_authorized:
        raise ValueError("Phase-4 mediation is forbidden without a validated GO")
    source_layers = sorted(int(layer) for layer in source_edits)
    if not source_layers:
        raise ValueError("Signed mediation requires a source edit")
    expected_layers = list(range(max(source_layers) + 1, len(blocks)))
    if sorted(set(int(layer) for layer in component_layers)) != expected_layers:
        raise ValueError(
            "Faithfulness requires the complete strictly-downstream component universe"
        )
    modules = _component_modules(blocks, component_layers)
    if not 1 <= circuit_size <= len(modules):
        raise ValueError("circuit_size must lie within the component roster")
    clean_outputs: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for component_id, module in modules.items():

            def capture(
                hooked_module: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                *,
                _component_id=component_id,
            ) -> Any:
                del hooked_module, inputs
                clean_outputs[_component_id] = _component_hidden(output).detach()
                return output

            handles.append(module.register_forward_hook(capture))
        clean_logits = hf_model(input_ids=input_ids, use_cache=False).logits.float()
    finally:
        for handle in handles:
            handle.remove()
    if set(clean_outputs) != set(modules):
        raise RuntimeError("Did not capture every downstream clean component")
    clean_metric = float(metric_fn(clean_logits).cpu())
    edited_logits = forward_logits(
        hf_model,
        input_ids,
        blocks=blocks,
        edits=source_edits,
    )
    edited_metric = float(metric_fn(edited_logits).cpu())

    component_rows: list[dict[str, Any]] = []
    for component_id in modules:
        with (
            residual_edit_hooks(blocks, source_edits),
            _restore_component_outputs(
                modules, clean_outputs, [component_id]
            ),
        ):
            restored_logits = hf_model(input_ids=input_ids, use_cache=False).logits.float()
        restored_metric = float(metric_fn(restored_logits).cpu())
        mediation = signed_component_mediation(
            clean_metric, edited_metric, restored_metric
        )
        component_rows.append(
            {
                "component": component_id,
                "restored_metric": restored_metric,
                **mediation,
            }
        )
    ranked = sorted(
        component_rows,
        key=lambda row: (-abs(float(row["READ_k"])), str(row["component"])),
    )
    circuit = [str(row["component"]) for row in ranked[:circuit_size]]
    outside = [component for component in modules if component not in set(circuit)]
    with _zero_component_outputs(modules, outside):
        clean_audit_logits = hf_model(input_ids=input_ids, use_cache=False).logits.float()
    with (
        residual_edit_hooks(blocks, source_edits),
        _zero_component_outputs(modules, outside),
    ):
        circuit_only_logits = hf_model(input_ids=input_ids, use_cache=False).logits.float()
    outside_clean_metric = float(metric_fn(clean_audit_logits).cpu())
    circuit_only_metric = float(metric_fn(circuit_only_logits).cpu())
    faithfulness = circuit_faithfulness(
        clean_metric,
        edited_metric,
        outside_clean_metric,
        circuit_only_metric,
    )
    return {
        "status": "OK",
        "clean_metric": clean_metric,
        "edited_metric": edited_metric,
        "component_rows": component_rows,
        "circuit_selection": "top absolute signed READ_k; lexical component tie-break",
        "circuit_size": circuit_size,
        "circuit_components": circuit,
        "outside_components_zero_ablated": outside,
        "outside_ablated_clean_metric": outside_clean_metric,
        "circuit_only_edited_metric": circuit_only_metric,
        "faithfulness": faithfulness,
        "individual_component_overlap_warning": (
            "Individual restoration effects can overlap and are not additive"
        ),
    }

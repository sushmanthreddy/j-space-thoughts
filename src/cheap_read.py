"""Clean-forward gradient-only READ estimators.

The module is deliberately self-contained: it neither imports donor-state
interchange code nor accepts causal outcomes.  Its only model operations are
clean activation capture and direction-defined activation paths with gradients.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn


BatchMetricFn = Callable[[torch.Tensor], torch.Tensor]


def batch_token_difference_metric(
    positive_token_id: int,
    negative_token_id: int,
) -> BatchMetricFn:
    """Return one final-position logit difference per batch row."""

    positive = int(positive_token_id)
    negative = int(negative_token_id)
    if positive == negative:
        raise ValueError("Behavior metric tokens must differ")

    def metric(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError("Expected logits shaped [B,S,V]")
        return logits[:, -1, positive].float() - logits[:, -1, negative].float()

    return metric


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


def _replace_output_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (hidden, *output[1:])
    raise TypeError(f"Unsupported decoder output type {type(output).__name__}")


def _validate_directions(
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = direction_a.detach().float()
    b = direction_b.detach().float()
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError("Concept directions must be same-shaped vectors")
    norms = torch.stack([a.norm(), b.norm()])
    if not torch.isfinite(torch.stack([a, b])).all() or not torch.allclose(
        norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4
    ):
        raise ValueError(f"Concept directions must be finite unit vectors: {norms}")
    return a, b


@torch.no_grad()
def clean_batch_states(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids: torch.Tensor,
    metric_fn: BatchMetricFn,
    *,
    layer: int,
    position: int = -1,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Capture clean selected-layer states, task metrics, and top tokens."""

    if hf_model.training:
        raise ValueError("Cheap READ requires deterministic eval mode")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,S]")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if position == -1 and not bool(torch.all(attention_mask[:, -1] == 1)):
            raise ValueError("Position -1 requires left padding with a real final token")
    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        captured["hidden"] = _hidden_from_output(output).detach()
        return output

    handle = blocks[int(layer)].register_forward_hook(hook)
    try:
        logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
    finally:
        handle.remove()
    hidden = captured.get("hidden")
    if hidden is None:
        raise RuntimeError("Selected-layer clean activation was not captured")
    index = _position_index(hidden.shape[1], position)
    metrics = metric_fn(logits)
    if metrics.shape != (input_ids.shape[0],) or not torch.isfinite(metrics).all():
        raise ValueError("Batch behavior metric must return one finite value per row")
    return {
        "states": hidden[:, index].float(),
        "metrics": metrics.detach().float(),
        "top_token_ids": logits[:, -1].argmax(dim=-1).detach(),
        "activation_dtype": str(hidden.dtype),
    }


def activation_metric_gradients(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids: torch.Tensor,
    metric_fn: BatchMetricFn,
    offsets: torch.Tensor,
    *,
    layer: int,
    position: int = -1,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiate batch metrics from a direction-defined residual offset."""

    if any(parameter.requires_grad for parameter in hf_model.parameters()):
        raise ValueError("Model parameters must be frozen; READ differentiates activations only")
    if input_ids.ndim != 2 or offsets.ndim != 2:
        raise ValueError("Expected input_ids [B,S] and offsets [B,D]")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if position == -1 and not bool(torch.all(attention_mask[:, -1] == 1)):
            raise ValueError("Position -1 requires left padding with a real final token")
    if offsets.shape[0] != input_ids.shape[0] or not torch.isfinite(offsets).all():
        raise ValueError("Offsets must be finite and align with the batch")
    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        hidden = _hidden_from_output(output)
        if hidden.shape[0] != offsets.shape[0] or hidden.shape[-1] != offsets.shape[1]:
            raise ValueError("Offset shape does not match selected-layer activation")
        index = _position_index(hidden.shape[1], position)
        captured["base_states"] = hidden[:, index].detach().float()
        leaf = hidden.detach().clone()
        replacement = leaf[:, index].float() + offsets.to(hidden.device, torch.float32)
        leaf[:, index] = replacement.to(hidden.dtype)
        leaf.requires_grad_(True)
        captured["leaf"] = leaf
        return _replace_output_hidden(output, leaf)

    handle = blocks[int(layer)].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits.float()
            metrics = metric_fn(logits)
            if metrics.shape != (input_ids.shape[0],) or not torch.isfinite(metrics).all():
                raise ValueError("Batch behavior metric must return finite row values")
            leaf = captured.get("leaf")
            if leaf is None:
                raise RuntimeError("Selected-layer gradient leaf was not captured")
            gradient = torch.autograd.grad(
                metrics.sum(),
                leaf,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
    finally:
        handle.remove()
    index = _position_index(gradient.shape[1], position)
    return {
        "metrics": metrics.detach().float(),
        "gradients": gradient[:, index].detach().float(),
        "base_states": captured["base_states"],
    }


def symmetric_gradient_read(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    input_ids: torch.Tensor,
    metric_fn: BatchMetricFn,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    *,
    layer: int,
    position: int = -1,
    ig_steps: int = 16,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute symmetric midpoint IG and local presence-times-sensitivity READ."""

    if input_ids.shape[0] != 2:
        raise ValueError("Symmetric READ expects exactly the A and B prompt rows")
    if not isinstance(ig_steps, int) or not 16 <= ig_steps <= 32:
        raise ValueError("Integrated gradients requires an integer 16--32 steps")
    direction_a, direction_b = _validate_directions(direction_a, direction_b)
    clean = clean_batch_states(
        hf_model,
        blocks,
        input_ids,
        metric_fn,
        layer=layer,
        position=position,
        attention_mask=attention_mask,
    )
    states = clean["states"]
    live_a = direction_a.to(states.device)
    live_b = direction_b.to(states.device)
    amount_a = torch.dot(states[0], live_a)
    amount_b = torch.dot(states[1], live_b)
    delta_a = amount_a * (live_b - live_a)
    delta_b = amount_b * (live_a - live_b)
    deltas = torch.stack([delta_a, delta_b])

    zero_offsets = torch.zeros_like(deltas)
    local = activation_metric_gradients(
        hf_model,
        blocks,
        input_ids,
        metric_fn,
        zero_offsets,
        layer=layer,
        position=position,
        attention_mask=attention_mask,
    )
    if not torch.allclose(local["metrics"], clean["metrics"], atol=2e-2, rtol=2e-3):
        raise ValueError("Clean metric changed in the zero-offset gradient pass")
    own_directions = torch.stack([live_a, live_b])
    amounts = torch.stack([amount_a, amount_b])
    local_signed = amounts * (local["gradients"] * own_directions).sum(dim=-1)

    integrands: list[torch.Tensor] = []
    path_metrics: list[torch.Tensor] = []
    for step in range(ig_steps):
        alpha = (step + 0.5) / ig_steps
        point = activation_metric_gradients(
            hf_model,
            blocks,
            input_ids,
            metric_fn,
            deltas * alpha,
            layer=layer,
            position=position,
            attention_mask=attention_mask,
        )
        if not torch.allclose(
            point["base_states"], states, atol=2e-2, rtol=2e-3
        ):
            raise ValueError("Clean base activation changed across the IG path")
        integrands.append((point["gradients"] * deltas).sum(dim=-1))
        path_metrics.append(point["metrics"])
    integrand_tensor = torch.stack(integrands)
    path_metric_tensor = torch.stack(path_metrics)
    ig_signed = integrand_tensor.mean(dim=0)
    endpoint = activation_metric_gradients(
        hf_model,
        blocks,
        input_ids,
        metric_fn,
        deltas,
        layer=layer,
        position=position,
        attention_mask=attention_mask,
    )
    endpoint_delta = endpoint["metrics"] - clean["metrics"]
    completeness_error = ig_signed - endpoint_delta
    finite_tensors = [
        states,
        amounts,
        deltas,
        local_signed,
        integrand_tensor,
        path_metric_tensor,
        ig_signed,
        endpoint_delta,
        completeness_error,
    ]
    if not all(torch.isfinite(tensor).all() for tensor in finite_tensors):
        raise ValueError("Cheap READ produced a non-finite value")
    return {
        "status": "OK",
        "layer": int(layer),
        "position": int(position),
        "position_contract": "left padded; every row has a real token at final column",
        "ig_steps": int(ig_steps),
        "ig_rule": "midpoint Riemann",
        "activation_dtype": clean["activation_dtype"],
        "delta_definition": (
            "Delta_A=(h_A dot v_A)(v_B-v_A), "
            "Delta_B=(h_B dot v_B)(v_A-v_B)"
        ),
        "clean_metrics": [float(value) for value in clean["metrics"].cpu()],
        "clean_top_token_ids": [
            int(value) for value in clean["top_token_ids"].cpu()
        ],
        "concept_amounts": [float(value) for value in amounts.cpu()],
        "delta_norms": [float(value) for value in deltas.norm(dim=-1).cpu()],
        "ig_signed_by_direction": [float(value) for value in ig_signed.cpu()],
        "ig_abs_by_direction": [float(value) for value in ig_signed.abs().cpu()],
        "READ_IG": float(ig_signed.abs().mean().cpu()),
        "direction_path_endpoint_metric_delta": [
            float(value) for value in endpoint_delta.cpu()
        ],
        "ig_completeness_error": [
            float(value) for value in completeness_error.cpu()
        ],
        "local_signed_by_direction": [
            float(value) for value in local_signed.cpu()
        ],
        "local_abs_by_direction": [
            float(value) for value in local_signed.abs().cpu()
        ],
        "READ_local": float(local_signed.abs().mean().cpu()),
        "ig_integrands": integrand_tensor.detach().float().cpu().tolist(),
        "ig_path_metrics": path_metric_tensor.detach().float().cpu().tolist(),
        "causal_outputs_consumed": False,
    }


@torch.no_grad()
def weight_norm_capacity_baseline(
    block: nn.Module,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
) -> dict[str, Any]:
    """Known-broken static MLP gain baseline, independent of the behavior metric."""

    direction_a, direction_b = _validate_directions(direction_a, direction_b)
    parameter = next(block.mlp.parameters())
    directions = torch.stack([direction_a, direction_b]).to(
        parameter.device, parameter.dtype
    )
    output = block.mlp(directions[:, None, :])[:, 0].float()
    gains = output.norm(dim=-1) / directions.float().norm(dim=-1)
    if not torch.isfinite(gains).all():
        raise ValueError("Static MLP capacity baseline is non-finite")
    return {
        "status": "OK",
        "baseline": "KNOWN_BROKEN_MLP_DIRECTION_RESPONSE_NORM_CAPACITY",
        "gain_by_concept": [float(value) for value in gains.cpu()],
        "weight_norm_baseline": float(gains.mean().cpu()),
        "behavior_metric_used": False,
        "eligible_for_go": False,
    }

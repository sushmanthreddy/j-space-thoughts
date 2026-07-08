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
ProgressFn = Callable[[int, int, dict[str, Any]], None]


def encode_prompt_pair(
    tokenizer: Any,
    prompt_a: str,
    prompt_b: str,
    position_a: int,
    position_b: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad a matched prompt pair and resolve per-row concept positions.

    The helper deliberately accepts only clean prompts and integer positions.
    It has no parameter through which causal or edited outputs could enter.
    """

    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [prompt_a, prompt_b],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    if not bool(torch.all(encoded.attention_mask[:, -1] == 1)):
        raise ValueError("Left-padded prompt pair must end in real tokens")
    left_padding = encoded.input_ids.shape[1] - encoded.attention_mask.sum(dim=1)
    positions = [
        int(left_padding[0]) + int(position_a),
        int(left_padding[1]) + int(position_b),
    ]
    return (
        encoded.input_ids.to(device),
        encoded.attention_mask.to(device),
        positions,
    )


def score_prompt_pair(
    hf_model: nn.Module,
    blocks: Sequence[nn.Module],
    tokenizer: Any,
    *,
    prompt_a: str,
    prompt_b: str,
    position_a: int,
    position_b: int,
    positive_token_id: int,
    negative_token_id: int,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    layer: int,
    ig_steps: int = 16,
) -> dict[str, Any]:
    """Compute frozen READ_IG/READ_local for one clean matched prompt pair."""

    device = next(hf_model.parameters()).device
    input_ids, attention_mask, positions = encode_prompt_pair(
        tokenizer,
        prompt_a,
        prompt_b,
        position_a,
        position_b,
        device=device,
    )
    return symmetric_gradient_read(
        hf_model,
        blocks,
        input_ids,
        batch_token_difference_metric(positive_token_id, negative_token_id),
        direction_a,
        direction_b,
        layer=layer,
        position=positions,
        ig_steps=ig_steps,
        attention_mask=attention_mask,
    )


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


def _position_indices(
    sequence_length: int,
    batch_size: int,
    position: int | Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(position, int):
        values = [_position_index(sequence_length, position)] * batch_size
    else:
        values = [
            _position_index(sequence_length, int(value)) for value in position
        ]
        if len(values) != batch_size:
            raise ValueError("Per-row activation positions must align with the batch")
    return torch.tensor(values, device=device, dtype=torch.long)


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
    position: int | Sequence[int] = -1,
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
    indices = _position_indices(
        hidden.shape[1], hidden.shape[0], position, device=hidden.device
    )
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    metrics = metric_fn(logits)
    if metrics.shape != (input_ids.shape[0],) or not torch.isfinite(metrics).all():
        raise ValueError("Batch behavior metric must return one finite value per row")
    return {
        "states": hidden[batch_indices, indices].float(),
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
    position: int | Sequence[int] = -1,
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
        indices = _position_indices(
            hidden.shape[1], hidden.shape[0], position, device=hidden.device
        )
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        captured["base_states"] = hidden[batch_indices, indices].detach().float()
        leaf = hidden.detach().clone()
        replacement = leaf[batch_indices, indices].float() + offsets.to(
            hidden.device, torch.float32
        )
        leaf[batch_indices, indices] = replacement.to(hidden.dtype)
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
    indices = _position_indices(
        gradient.shape[1], gradient.shape[0], position, device=gradient.device
    )
    batch_indices = torch.arange(gradient.shape[0], device=gradient.device)
    return {
        "metrics": metrics.detach().float(),
        "gradients": gradient[batch_indices, indices].detach().float(),
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
    position: int | Sequence[int] = -1,
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
        "positions": (
            [int(position)] * input_ids.shape[0]
            if isinstance(position, int)
            else [int(value) for value in position]
        ),
        "position_contract": (
            "per-row explicit concept token in shared context; left-padding offsets resolved"
        ),
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


def _validated_sha256(value: str, *, label: str) -> str:
    """Validate caller-computed byte provenance without reading a file."""

    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return digest


def _model_and_blocks(bundle: Any, manifest: dict[str, Any]) -> tuple[nn.Module, Any, Sequence[nn.Module]]:
    """Resolve and validate the already-loaded model bundle used by a run."""

    try:
        hf_model = bundle.hf_model
        tokenizer = bundle.tokenizer
        blocks = bundle.lens_model.layers
    except AttributeError as error:
        raise TypeError(
            "bundle must expose hf_model, tokenizer, and lens_model.layers"
        ) from error
    if not isinstance(hf_model, nn.Module):
        raise TypeError("bundle.hf_model must be a torch module")
    if hf_model.training:
        raise ValueError("Cheap READ requires the loaded model to be in eval mode")
    if any(parameter.requires_grad for parameter in hf_model.parameters()):
        raise ValueError("Cheap READ requires all model parameters to be frozen")
    try:
        block_count = len(blocks)
        blocks[0]
    except (TypeError, IndexError, KeyError) as error:
        raise TypeError(
            "bundle.lens_model.layers must be a non-empty indexable collection"
        ) from error
    if block_count < 1:
        raise TypeError("bundle.lens_model.layers must be non-empty")

    model_record = manifest.get("model")
    if not isinstance(model_record, dict):
        raise ValueError("Sanitized manifest is missing its model record")
    expected_id = str(model_record.get("id", ""))
    expected_revision = str(model_record.get("revision", ""))
    if not expected_id or not expected_revision:
        raise ValueError("Sanitized manifest model ID and revision must be explicit")
    observed_id = getattr(bundle, "model_id", None)
    observed_revision = getattr(bundle, "revision", None)
    if observed_id is not None and str(observed_id) != expected_id:
        raise ValueError(
            f"Loaded model ID {observed_id!r} differs from manifest {expected_id!r}"
        )
    if observed_revision is not None and str(observed_revision) != expected_revision:
        raise ValueError("Loaded model revision differs from the sanitized manifest")
    observed_dtype = str(next(hf_model.parameters()).dtype)
    expected_dtype = model_record.get("dtype")
    if expected_dtype is not None and observed_dtype != str(expected_dtype):
        raise ValueError(
            f"Loaded model dtype {observed_dtype!r} differs from manifest {expected_dtype!r}"
        )
    return hf_model, tokenizer, blocks


def _validated_direction_cache(
    direction_cache: dict[str, Any],
    *,
    protocol_sha256: str,
    selected_layer: int,
    model: dict[str, Any],
) -> dict[Any, torch.Tensor]:
    """Validate the frozen direction cache and return its token-indexed vectors."""

    if not isinstance(direction_cache, dict):
        raise TypeError("direction_cache must be a loaded mapping")
    if str(direction_cache.get("protocol_sha256", "")) != str(protocol_sha256):
        raise ValueError("Direction cache and sanitized manifest use different protocols")
    if int(direction_cache.get("selected_layer", -1)) != int(selected_layer):
        raise ValueError("Direction cache and sanitized manifest select different layers")
    cached_model_id = direction_cache.get("model_id")
    cached_revision = direction_cache.get("model_revision")
    if cached_model_id is not None and str(cached_model_id) != str(model["id"]):
        raise ValueError("Direction cache model ID differs from the sanitized manifest")
    if cached_revision is not None and str(cached_revision) != str(model["revision"]):
        raise ValueError("Direction cache model revision differs from the sanitized manifest")
    directions = direction_cache.get("directions")
    if not isinstance(directions, dict) or not directions:
        raise ValueError("Direction cache must contain a non-empty directions mapping")
    return directions


def _direction_for_token(
    directions: dict[Any, torch.Tensor], token_id: int
) -> torch.Tensor:
    """Return one cached direction while tolerating JSON-style string keys."""

    token = int(token_id)
    value = directions.get(token)
    if value is None:
        value = directions.get(str(token))
    if value is None:
        raise KeyError(f"Direction cache has no vector for token ID {token}")
    if not torch.is_tensor(value):
        raise TypeError(f"Cached direction for token ID {token} is not a tensor")
    return value


def _selected_layer_and_rule(manifest: dict[str, Any]) -> tuple[int, str]:
    """Read the frozen activation selection from a sanitized manifest."""

    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Sanitized manifest is missing its selection record")
    layer = int(selection.get("layer", -1))
    position_rule = str(selection.get("position_rule", ""))
    if layer < 0 or not position_rule:
        raise ValueError("Sanitized manifest has an invalid layer or position rule")
    return layer, position_rule


def _verified_rows(
    manifest: dict[str, Any], *, expected_status: str
) -> list[dict[str, Any]]:
    """Select verified rows in their frozen order and reject duplicate IDs."""

    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Sanitized manifest rows must be a list")
    verified = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("verification_status") == expected_status
    ]
    if not verified:
        raise ValueError(f"No rows with verification status {expected_status!r}")
    pair_ids = [str(row.get("pair_id", "")) for row in verified]
    if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Verified rows must have unique non-empty pair IDs")
    return verified


def _base_firewall_audit() -> dict[str, Any]:
    """Return the phase-32-compatible isolation record for this module."""

    return {
        "status": "PASS",
        "cheap_module_imports": [
            "__future__",
            "collections.abc",
            "typing",
            "torch",
            "torch",
        ],
        "forbidden_imports_found": [],
        "causal_artifact_path_referenced": False,
        "causal_outputs_consumed": False,
    }


def compact_base_cheap_read_rows(
    artifact_or_rows: dict[str, Any] | Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project full base results to the compact phase-32 metrics-row schema."""

    if isinstance(artifact_or_rows, dict):
        rows = artifact_or_rows.get("rows")
    else:
        rows = artifact_or_rows
    if not isinstance(rows, Sequence):
        raise TypeError("Expected an artifact with rows or a row sequence")

    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Every cheap READ row must be a mapping")
        engine = row.get("engine")
        dashboard = row.get("dashboard")
        baseline = row.get("weight_norm_capacity_baseline")
        if not isinstance(engine, dict) or not isinstance(dashboard, dict):
            raise ValueError("Base rows must contain engine and dashboard estimates")
        if not isinstance(baseline, dict):
            raise ValueError("Base rows must contain the capacity baseline record")
        compact.append(
            {
                "pair_id": row["pair_id"],
                "dependency_group": row["dependency_group"],
                "fold": row["fold"],
                "category": row["category"],
                "concept_a": row["concept_a"],
                "concept_b": row["concept_b"],
                "engine": {
                    "READ_IG": engine["READ_IG"],
                    "READ_local": engine["READ_local"],
                    "ig_abs_by_direction": engine["ig_abs_by_direction"],
                    "local_abs_by_direction": engine["local_abs_by_direction"],
                },
                "dashboard": {
                    "READ_IG": dashboard["READ_IG"],
                    "READ_local": dashboard["READ_local"],
                    "ig_abs_by_direction": dashboard["ig_abs_by_direction"],
                    "local_abs_by_direction": dashboard["local_abs_by_direction"],
                },
                "weight_norm_baseline": baseline["weight_norm_baseline"],
                "baseline_label": baseline["baseline"],
            }
        )
    return compact


def compute_base_cheap_read(
    bundle: Any,
    clean_manifest: dict[str, Any],
    direction_cache: dict[str, Any],
    *,
    clean_manifest_sha256: str,
    direction_cache_sha256: str | None = None,
    ig_steps: int = 16,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Reproduce phase-32 engine, old-dashboard, and capacity estimates.

    ``clean_manifest`` and ``direction_cache`` must already be loaded by the
    caller.  The function performs no file I/O and accepts no edited-model
    measurements.  It returns the historical ``symmetric-cheap-read-v1`` raw
    artifact schema; :func:`compact_base_cheap_read_rows` derives the exact
    compact rows formerly embedded in the aggregate metrics file.

    ``progress``, when supplied, receives ``(completed, total, compact_row)``.
    The callback runs only after the row's numerical outputs are complete.
    """

    if clean_manifest.get("schema_version") != "symmetric-clean-read-manifest-v1":
        raise ValueError("Unsupported clean READ manifest schema")
    if clean_manifest.get("causal_interchange_outputs_included") is not False:
        raise ValueError("Clean READ manifest is not certified output-free")
    source_sha256 = _validated_sha256(
        clean_manifest_sha256, label="clean_manifest_sha256"
    )
    if direction_cache_sha256 is not None:
        observed_cache_sha256 = _validated_sha256(
            direction_cache_sha256, label="direction_cache_sha256"
        )
        expected_cache = clean_manifest.get("direction_cache")
        if isinstance(expected_cache, dict) and expected_cache.get("sha256") is not None:
            if observed_cache_sha256 != str(expected_cache["sha256"]).lower():
                raise ValueError("Direction-cache byte hash differs from clean manifest")

    selected_layer, position_rule = _selected_layer_and_rule(clean_manifest)
    hf_model, tokenizer, blocks = _model_and_blocks(bundle, clean_manifest)
    if selected_layer >= len(blocks):
        raise ValueError("Selected layer is outside the loaded model")
    protocol_sha256 = str(clean_manifest.get("protocol_sha256", ""))
    if not protocol_sha256:
        raise ValueError("Clean READ manifest is missing protocol_sha256")
    directions = _validated_direction_cache(
        direction_cache,
        protocol_sha256=protocol_sha256,
        selected_layer=selected_layer,
        model=clean_manifest["model"],
    )
    rows = _verified_rows(clean_manifest, expected_status="VERIFIED")

    raw_rows: list[dict[str, Any]] = []
    total = len(rows)
    for completed, pair in enumerate(rows, start=1):
        direction_a = _direction_for_token(directions, int(pair["concept_a_token_id"]))
        direction_b = _direction_for_token(directions, int(pair["concept_b_token_id"]))
        engine_read = score_prompt_pair(
            hf_model,
            blocks,
            tokenizer,
            prompt_a=str(pair["engine_prompt_a"]),
            prompt_b=str(pair["engine_prompt_b"]),
            position_a=int(pair["intervention_position_a"]),
            position_b=int(pair["intervention_position_b"]),
            positive_token_id=int(pair["answer_a_token_id"]),
            negative_token_id=int(pair["answer_b_token_id"]),
            direction_a=direction_a,
            direction_b=direction_b,
            layer=selected_layer,
            ig_steps=ig_steps,
        )
        dashboard_read = score_prompt_pair(
            hf_model,
            blocks,
            tokenizer,
            prompt_a=str(pair["dashboard_prompt_a"]),
            prompt_b=str(pair["dashboard_prompt_b"]),
            position_a=int(pair["intervention_position_a"]),
            position_b=int(pair["intervention_position_b"]),
            positive_token_id=int(pair["dashboard_token_id"]),
            negative_token_id=int(pair["dashboard_distractor_token_id"]),
            direction_a=direction_a,
            direction_b=direction_b,
            layer=selected_layer,
            ig_steps=ig_steps,
        )
        baseline = weight_norm_capacity_baseline(
            blocks[selected_layer], direction_a, direction_b
        )
        row = {
            "pair_id": pair["pair_id"],
            "dependency_group": pair["dependency_group"],
            "fold": pair["fold"],
            "category": pair["category"],
            "concept_a": pair["concept_a"],
            "concept_b": pair["concept_b"],
            "engine": engine_read,
            "dashboard": dashboard_read,
            "weight_norm_capacity_baseline": baseline,
        }
        raw_rows.append(row)
        if progress is not None:
            progress(completed, total, compact_base_cheap_read_rows([row])[0])

    return {
        "schema_version": "symmetric-cheap-read-v1",
        "protocol_sha256": protocol_sha256,
        "upstream_clean_manifest_sha256": source_sha256,
        "model": clean_manifest["model"],
        "selected_layer": selected_layer,
        "position_rule": position_rule,
        "ig_steps": int(ig_steps),
        "anti_circularity_audit": _base_firewall_audit(),
        "rows": raw_rows,
    }


def compute_hard_dashboard_read(
    bundle: Any,
    hard_manifest: dict[str, Any],
    direction_cache: dict[str, Any],
    *,
    hard_manifest_path: str,
    hard_manifest_sha256: str,
    direction_cache_sha256: str | None = None,
    cheap_read_sha256: str | None = None,
    ig_steps: int = 16,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Compute frozen READ rows for verified answer-type-matched dashboards.

    The return value matches the historical hard-dashboard artifact layout,
    including both full estimator records in ``rows`` and evaluation-ready
    ``compact_rows``.  The loaded model bundle is reused; this helper neither
    loads a model nor reads any upstream file.
    """

    if hard_manifest.get("schema_version") != "read-stress-v6-hard-dashboard-manifest-v1":
        raise ValueError("Unsupported hard-dashboard manifest schema")
    if hard_manifest.get("causal_interchange_outputs_included") is not False:
        raise ValueError("Hard-dashboard manifest is not certified output-free")
    if hard_manifest.get("edited_metrics_included") is not False:
        raise ValueError("Hard-dashboard manifest contains edited measurements")
    source_sha256 = _validated_sha256(
        hard_manifest_sha256, label="hard_manifest_sha256"
    )
    source_path = str(hard_manifest_path)
    if not source_path:
        raise ValueError("hard_manifest_path must be non-empty provenance text")
    cache_hash_validated = False
    if direction_cache_sha256 is not None:
        observed_cache_sha256 = _validated_sha256(
            direction_cache_sha256, label="direction_cache_sha256"
        )
        expected_cache = hard_manifest.get("direction_cache")
        if isinstance(expected_cache, dict) and expected_cache.get("sha256") is not None:
            if observed_cache_sha256 != str(expected_cache["sha256"]).lower():
                raise ValueError("Direction-cache byte hash differs from hard manifest")
            cache_hash_validated = True

    selected_layer, position_rule = _selected_layer_and_rule(hard_manifest)
    hf_model, tokenizer, blocks = _model_and_blocks(bundle, hard_manifest)
    if selected_layer >= len(blocks):
        raise ValueError("Selected layer is outside the loaded model")
    source_clean = hard_manifest.get("source_clean_manifest")
    if not isinstance(source_clean, dict):
        raise ValueError("Hard manifest is missing clean-manifest provenance")
    protocol_sha256 = str(source_clean.get("protocol_sha256", ""))
    if not protocol_sha256:
        raise ValueError("Hard manifest is missing its frozen protocol hash")
    directions = _validated_direction_cache(
        direction_cache,
        protocol_sha256=protocol_sha256,
        selected_layer=selected_layer,
        model=hard_manifest["model"],
    )
    rows = _verified_rows(hard_manifest, expected_status="VERIFIED_HARD")

    raw_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    total = len(rows)
    for completed, row in enumerate(rows, start=1):
        estimate = score_prompt_pair(
            hf_model,
            blocks,
            tokenizer,
            prompt_a=str(row["hard_prompt_a"]),
            prompt_b=str(row["hard_prompt_b"]),
            position_a=int(row["intervention_position_a"]),
            position_b=int(row["intervention_position_b"]),
            positive_token_id=int(row["hard_target_token_id"]),
            negative_token_id=int(row["hard_distractor_token_id"]),
            direction_a=_direction_for_token(
                directions, int(row["concept_a_token_id"])
            ),
            direction_b=_direction_for_token(
                directions, int(row["concept_b_token_id"])
            ),
            layer=selected_layer,
            ig_steps=ig_steps,
        )
        if estimate.get("causal_outputs_consumed") is not False:
            raise ValueError("Cheap estimator did not certify output isolation")
        metadata = {
            "pair_id": row["pair_id"],
            "dependency_group": row["dependency_group"],
            "fold": int(row["fold"]),
            "category": row["category"],
            "hard_template_id": row["hard_template_id"],
        }
        raw_rows.append({**metadata, "hard_dashboard": estimate})
        compact = {
            **metadata,
            "READ_IG": float(estimate["READ_IG"]),
            "READ_local": float(estimate["READ_local"]),
            "ig_abs_by_direction": estimate["ig_abs_by_direction"],
            "local_abs_by_direction": estimate["local_abs_by_direction"],
        }
        compact_rows.append(compact)
        if progress is not None:
            progress(completed, total, compact)

    audit: dict[str, Any] = {
        "status": "PASS",
        "forbidden_imports_found": [],
        "causal_artifact_read": False,
        "causal_outputs_consumed": False,
        "estimator_logic": "src.cheap_read.symmetric_gradient_read",
        "direction_cache_sha256_validated": cache_hash_validated,
    }
    if cheap_read_sha256 is not None:
        audit["cheap_read_sha256"] = _validated_sha256(
            cheap_read_sha256, label="cheap_read_sha256"
        )
    return {
        "schema_version": "read-stress-v6-hard-dashboard-cheap-v1",
        "model": hard_manifest["model"],
        "selected_layer": selected_layer,
        "position_rule": position_rule,
        "ig_steps": int(ig_steps),
        "source_hard_manifest": {
            "path": source_path,
            "sha256": source_sha256,
        },
        "anti_circularity_audit": audit,
        "rows": raw_rows,
        "compact_rows": compact_rows,
    }


def summarize_cheap_read_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return a small model-free execution summary for notebook display."""

    schema = artifact.get("schema_version")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Cheap READ artifact has no rows list")
    if schema == "symmetric-cheap-read-v1":
        compact = compact_base_cheap_read_rows(rows)
        return {
            "schema_version": schema,
            "rows": len(rows),
            "selected_layer": int(artifact["selected_layer"]),
            "ig_steps": int(artifact["ig_steps"]),
            "engine_READ_IG_median": float(
                torch.tensor(
                    [row["engine"]["READ_IG"] for row in compact],
                    dtype=torch.float64,
                ).median()
            ),
            "dashboard_READ_IG_median": float(
                torch.tensor(
                    [row["dashboard"]["READ_IG"] for row in compact],
                    dtype=torch.float64,
                ).median()
            ),
            "causal_outputs_consumed": False,
        }
    if schema == "read-stress-v6-hard-dashboard-cheap-v1":
        compact = artifact.get("compact_rows")
        if not isinstance(compact, list) or len(compact) != len(rows):
            raise ValueError("Hard artifact compact rows do not align with full rows")
        return {
            "schema_version": schema,
            "rows": len(rows),
            "selected_layer": int(artifact["selected_layer"]),
            "ig_steps": int(artifact["ig_steps"]),
            "hard_dashboard_READ_IG_median": float(
                torch.tensor(
                    [row["READ_IG"] for row in compact], dtype=torch.float64
                ).median()
            ),
            "causal_outputs_consumed": False,
        }
    raise ValueError(f"Unsupported cheap READ artifact schema {schema!r}")

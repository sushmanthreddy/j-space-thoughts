"""Attribution-based and activation-independent weight-based READ estimators."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from jlens.hooks import ActivationRecorder

from src.interventions import residual_edit_hooks


@dataclass
class AttributionReadResult:
    """Per-position decomposition for one behavior logit difference."""

    metric: float
    write: dict[int, np.ndarray]
    read: dict[int, np.ndarray]
    predicted_delta_by_layer: dict[int, float]
    predicted_delta: float


def _resolve_position(length: int, position: int) -> int:
    resolved = int(position)
    if resolved < 0:
        resolved += length
    if not 0 <= resolved < length:
        raise IndexError(f"Position {position} outside sequence of length {length}")
    return resolved


def attribution_read(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    directions: Mapping[int, torch.Tensor],
    *,
    target_token_id: int,
    foil_token_id: int,
    behavior_position: int = -1,
    intervention_positions: Sequence[int] | None = None,
    attention_mask: torch.Tensor | None = None,
) -> AttributionReadResult:
    """Compute WRITE, ``grad(M)·v``, and the ablation linear prediction.

    One forward and one backward cover all requested layers. The earliest
    requested post-block residual becomes the autograd root; model parameters
    must already be frozen (the official HF J-Lens adapter does this).
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("attribution_read currently requires one unpadded item")
    if target_token_id == foil_token_id:
        raise ValueError("Behavior target and foil token IDs must differ")
    layers = sorted(int(layer) for layer in directions)
    if not layers:
        raise ValueError("At least one direction/layer is required")
    if any(parameter.requires_grad for parameter in hf_model.parameters()):
        raise ValueError("Freeze model parameters before activation attribution")

    sequence_length = int(input_ids.shape[1])
    behavior_index = _resolve_position(sequence_length, behavior_position)
    if intervention_positions is None:
        selected_positions = list(range(sequence_length))
    else:
        selected_positions = [
            _resolve_position(sequence_length, position)
            for position in intervention_positions
        ]

    with torch.enable_grad(), ActivationRecorder(
        blocks,
        at=layers,
        start_graph_at=layers[0],
    ) as recorder:
        logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        metric_tensor = (
            logits[0, behavior_index, int(target_token_id)]
            - logits[0, behavior_index, int(foil_token_id)]
        )
        activation_tuple = tuple(recorder.activations[layer] for layer in layers)
        gradients = torch.autograd.grad(
            metric_tensor,
            activation_tuple,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

    write: dict[int, np.ndarray] = {}
    read: dict[int, np.ndarray] = {}
    predicted_by_layer: dict[int, float] = {}
    for layer, activation, gradient in zip(
        layers, activation_tuple, gradients, strict=True
    ):
        vector = directions[layer].detach().to(activation.device, torch.float32)
        selected_activation = activation[0, selected_positions].detach().float()
        selected_gradient = gradient[0, selected_positions].detach().float()
        layer_write = selected_activation @ vector
        layer_read = selected_gradient @ vector
        write[layer] = layer_write.cpu().numpy()
        read[layer] = layer_read.cpu().numpy()
        predicted_by_layer[layer] = float((-(layer_write * layer_read).sum()).cpu())

    return AttributionReadResult(
        metric=float(metric_tensor.detach().cpu()),
        write=write,
        read=read,
        predicted_delta_by_layer=predicted_by_layer,
        predicted_delta=float(sum(predicted_by_layer.values())),
    )


@torch.no_grad()
def qwen_mlp_gain(
    block: torch.nn.Module,
    direction: torch.Tensor,
    *,
    n_random: int = 128,
    seed: int = 1729,
) -> dict[str, float | list[float]]:
    """Normalized Qwen MLP response norm for a residual direction.

    The same post-attention RMSNorm and gated MLP weights are applied to the
    concept and seeded random unit directions. This estimator uses weights but
    not the current prompt activation.
    """

    if not hasattr(block, "mlp") or not hasattr(block, "post_attention_layernorm"):
        raise TypeError("Expected a Qwen-like block with post-attention norm and MLP")
    device = next(block.parameters()).device
    dtype = next(block.parameters()).dtype
    vector = F.normalize(direction.detach().float(), dim=0).to(device=device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_vectors = torch.randn(
        n_random, vector.numel(), generator=generator, dtype=torch.float32
    )
    random_vectors = F.normalize(random_vectors, dim=-1).to(device=device)
    inputs = torch.cat([vector.unsqueeze(0), random_vectors], dim=0).to(dtype=dtype)
    outputs = block.mlp(block.post_attention_layernorm(inputs)).float()
    norms = outputs.norm(dim=-1).cpu()
    observed = float(norms[0])
    null = norms[1:]
    median = float(null.median())
    return {
        "gain": observed,
        "random_median": median,
        "normalized_gain": observed / median if median > 0 else float("nan"),
        "random_gains": [float(value) for value in null],
    }


@torch.no_grad()
def qwen_head_ov_read(
    attention: torch.nn.Module,
    direction: torch.Tensor,
    *,
    label_direction: torch.Tensor | None = None,
) -> list[dict[str, float | int]]:
    """Per-query-head ``||W_O^h W_V^kv v||`` and label preservation.

    Qwen grouped-query attention shares each KV head across a contiguous group
    of query heads. The output projection slice remains query-head specific.
    """

    required = ("v_proj", "o_proj", "config", "head_dim")
    if not all(hasattr(attention, name) for name in required):
        raise TypeError("Expected a Qwen-like attention module")
    num_heads = int(attention.config.num_attention_heads)
    num_kv_heads = int(attention.config.num_key_value_heads)
    head_dim = int(attention.head_dim)
    if num_heads % num_kv_heads:
        raise ValueError("Query-head count must be divisible by KV-head count")
    group_size = num_heads // num_kv_heads

    vector = F.normalize(direction.detach().float(), dim=0).to(
        attention.v_proj.weight.device
    )
    label = F.normalize(
        (label_direction if label_direction is not None else direction)
        .detach()
        .float(),
        dim=0,
    ).to(attention.o_proj.weight.device)
    v_weight = attention.v_proj.weight.detach().float()
    o_weight = attention.o_proj.weight.detach().float()
    rows: list[dict[str, float | int]] = []
    for head in range(num_heads):
        kv_head = head // group_size
        value_slice = slice(kv_head * head_dim, (kv_head + 1) * head_dim)
        output_slice = slice(head * head_dim, (head + 1) * head_dim)
        value = v_weight[value_slice] @ vector
        ov_output = o_weight[:, output_slice] @ value
        norm = ov_output.norm()
        preservation = (
            F.cosine_similarity(ov_output, label, dim=0)
            if float(norm) > 0
            else torch.tensor(float("nan"), device=norm.device)
        )
        rows.append(
            {
                "head": head,
                "kv_head": kv_head,
                "ov_norm": float(norm.cpu()),
                "label_cosine": float(preservation.cpu()),
            }
        )
    return rows


@contextmanager
def _capture_qwen_path_components(
    blocks: Sequence[torch.nn.Module],
    layers: Sequence[int],
) -> Iterator[dict[str, dict[int, torch.Tensor]]]:
    """Capture Qwen MLP outputs and per-head streams immediately before W_O."""

    selected = sorted(set(int(layer) for layer in layers))
    if not selected or any(layer < 0 or layer >= len(blocks) for layer in selected):
        raise ValueError("Path-component layers must be nonempty and in range")
    captured: dict[str, dict[int, torch.Tensor]] = {
        "mlps": {},
        "attention_pre_o_proj": {},
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for layer in selected:
            block = blocks[layer]

            def mlp_hook(module, inputs, output, *, _layer=layer):
                del module, inputs
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    raise TypeError("Qwen MLP output must be a [B,S,D] tensor")
                captured["mlps"][_layer] = output.detach().clone()

            def attention_pre_hook(module, inputs, *, _layer=layer):
                del module
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise TypeError("Qwen o_proj must receive a tensor")
                if inputs[0].ndim != 3:
                    raise ValueError("Qwen pre-o_proj stream must have shape [B,S,D]")
                captured["attention_pre_o_proj"][_layer] = (
                    inputs[0].detach().clone()
                )

            handles.append(block.mlp.register_forward_hook(mlp_hook))
            handles.append(
                block.self_attn.o_proj.register_forward_pre_hook(
                    attention_pre_hook
                )
            )
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _metric_scalar(
    metric_fn: Callable[[torch.Tensor], torch.Tensor], logits: torch.Tensor
) -> float:
    value = metric_fn(logits)
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError("Behavior metric must return one finite scalar")
    return float(value.detach().cpu())


def _metric_batch(
    metric_fn: Callable[[torch.Tensor], torch.Tensor], logits: torch.Tensor
) -> list[float]:
    return [_metric_scalar(metric_fn, logits[index : index + 1]) for index in range(logits.shape[0])]


@torch.no_grad()
def exact_path_patch_scores(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    edits: Mapping[int, Callable[[torch.Tensor], torch.Tensor]],
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    component_layers: Sequence[int],
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure exact edited-into-clean component patch contributions to ``M``.

    The clean and fixed surgical edited runs are captured once. Each MLP output
    and each individual attention-head stream is then patched from the edited
    run into an otherwise clean forward pass. Attention heads are vectorized by
    layer (one batch element per head); all MLPs are vectorized in one batch.
    Contributions are exact one-component patch effects, not gradient proxies.
    They remain non-additive because downstream paths overlap.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Exact path patching requires one unpadded item")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    layers = sorted(set(int(layer) for layer in component_layers))
    if not layers or any(layer < 0 or layer >= len(blocks) for layer in layers):
        raise ValueError("Component layers must be nonempty and in range")

    clean_logits = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits.float()
    with residual_edit_hooks(blocks, edits):
        edited_logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
    clean_metric = _metric_scalar(metric_fn, clean_logits)
    edited_metric = _metric_scalar(metric_fn, edited_logits)

    mlp_batch_size = len(layers) + 1
    repeated_ids = input_ids.repeat(mlp_batch_size, 1)
    repeated_mask = (
        attention_mask.repeat(mlp_batch_size, 1)
        if attention_mask is not None
        else None
    )
    with (
        residual_edit_hooks(blocks, edits),
        _capture_qwen_path_components(blocks, layers) as edited_mlp_components,
    ):
        hf_model(
            input_ids=repeated_ids,
            attention_mask=repeated_mask,
            use_cache=False,
        )
    mlp_handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for batch_index, layer in enumerate(layers):
            replacement = edited_mlp_components["mlps"][layer]

            def mlp_patch_hook(
                module,
                inputs,
                output,
                *,
                _batch_index=batch_index,
                _replacement=replacement,
            ):
                del module, inputs
                patched = output.clone()
                patched[_batch_index] = _replacement[_batch_index].to(
                    device=patched.device, dtype=patched.dtype
                )
                return patched

            mlp_handles.append(
                blocks[layer].mlp.register_forward_hook(mlp_patch_hook)
            )
        mlp_logits = hf_model(
            input_ids=repeated_ids,
            attention_mask=repeated_mask,
            use_cache=False,
        ).logits.float()
    finally:
        for handle in mlp_handles:
            handle.remove()
    mlp_metrics = _metric_batch(metric_fn, mlp_logits)
    mlp_clean_reference = mlp_metrics[-1]
    mlp_rows = [
        {
            "component": f"L{layer}.MLP",
            "layer": layer,
            "patched_metric": metric,
            "batch_matched_clean_metric": mlp_clean_reference,
            "patched_contribution": metric - mlp_clean_reference,
            "abs_patched_contribution": abs(metric - mlp_clean_reference),
            "patch": "edited MLP output into otherwise clean run",
        }
        for layer, metric in zip(layers, mlp_metrics[:-1], strict=True)
    ]

    head_rows: list[dict[str, Any]] = []
    for layer in layers:
        attention = blocks[layer].self_attn
        num_heads = int(attention.config.num_attention_heads)
        head_dim = int(attention.head_dim)
        head_batch_size = num_heads + 1
        head_ids = input_ids.repeat(head_batch_size, 1)
        head_mask = (
            attention_mask.repeat(head_batch_size, 1)
            if attention_mask is not None
            else None
        )
        with (
            residual_edit_hooks(blocks, edits),
            _capture_qwen_path_components(
                blocks, [layer]
            ) as edited_attention_components,
        ):
            hf_model(
                input_ids=head_ids,
                attention_mask=head_mask,
                use_cache=False,
            )
        replacement = edited_attention_components["attention_pre_o_proj"][layer]
        expected_width = num_heads * head_dim
        if replacement.shape[-1] != expected_width:
            raise ValueError("Captured attention width disagrees with head geometry")

        def attention_patch_hook(module, inputs):
            del module
            stream = inputs[0].clone()
            for head in range(num_heads):
                start = head * head_dim
                stop = start + head_dim
                stream[head, :, start:stop] = replacement[
                    head, :, start:stop
                ].to(
                    device=stream.device, dtype=stream.dtype
                )
            return (stream, *inputs[1:])

        handle = attention.o_proj.register_forward_pre_hook(attention_patch_hook)
        try:
            head_logits = hf_model(
                input_ids=head_ids,
                attention_mask=head_mask,
                use_cache=False,
            ).logits.float()
        finally:
            handle.remove()
        head_metrics = _metric_batch(metric_fn, head_logits)
        head_clean_reference = head_metrics[-1]
        for head, metric in enumerate(head_metrics[:-1]):
            head_rows.append(
                {
                    "component": f"L{layer}.H{head}",
                    "layer": layer,
                    "head": head,
                    "patched_metric": metric,
                    "batch_matched_clean_metric": head_clean_reference,
                    "patched_contribution": metric - head_clean_reference,
                    "abs_patched_contribution": abs(
                        metric - head_clean_reference
                    ),
                    "patch": (
                        "edited pre-o_proj head stream into otherwise clean run"
                    ),
                }
            )

    return {
        "clean_metric": clean_metric,
        "edited_metric": edited_metric,
        "actual_delta": edited_metric - clean_metric,
        "positive_damage": clean_metric - edited_metric,
        "component_layers": layers,
        "mlps": mlp_rows,
        "attention_heads": head_rows,
        "estimator": {
            "type": "exact edited-into-clean component path patching",
            "gradient_used": False,
            "one_component_per_patched_run": True,
            "attention_vectorized_one_batch_element_per_head": True,
            "batch_matched_clean_control": True,
            "batch_matched_edited_activations": True,
            "warning": (
                "Single-component patch effects overlap and must not be summed "
                "as an additive decomposition."
            ),
        },
    }


@torch.no_grad()
def _normalized_single_head_read(
    attention: torch.nn.Module,
    direction: torch.Tensor,
    label_direction: torch.Tensor,
    *,
    head: int,
    n_random: int,
    seed: int,
) -> dict[str, Any]:
    observed_rows = qwen_head_ov_read(
        attention,
        direction,
        label_direction=label_direction,
    )
    if not 0 <= int(head) < len(observed_rows):
        raise IndexError(f"Attention head {head} outside layer")
    observed = observed_rows[int(head)]
    width = int(direction.numel())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_vectors = F.normalize(
        torch.randn(n_random, width, generator=generator, dtype=torch.float32),
        dim=-1,
    )
    num_heads = int(attention.config.num_attention_heads)
    num_kv_heads = int(attention.config.num_key_value_heads)
    head_dim = int(attention.head_dim)
    if num_heads % num_kv_heads:
        raise ValueError("Query-head count must be divisible by KV-head count")
    kv_head = int(head) // (num_heads // num_kv_heads)
    value_slice = slice(kv_head * head_dim, (kv_head + 1) * head_dim)
    output_slice = slice(int(head) * head_dim, (int(head) + 1) * head_dim)
    v_weight = attention.v_proj.weight.detach().float()[value_slice]
    o_weight = attention.o_proj.weight.detach().float()[:, output_slice]
    random_values = random_vectors.to(v_weight.device) @ v_weight.T
    random_outputs = random_values.to(o_weight.device) @ o_weight.T
    random_norms = random_outputs.norm(dim=-1).cpu().numpy()
    median = float(np.median(random_norms))
    observed_norm = float(observed["ov_norm"])
    observed_cosine = float(observed["label_cosine"])
    return {
        **observed,
        "normalized_ov_norm": (
            observed_norm / median if median > 0.0 else float("nan")
        ),
        "random_median_ov_norm": median,
        "random_ov_norms": [float(value) for value in random_norms],
        "ov_norm_random_percentile": float(
            np.mean(random_norms <= observed_norm)
        ),
        "label_weighted_normalized_ov": (
            observed_norm / median * abs(observed_cosine)
            if median > 0.0 and math.isfinite(observed_cosine)
            else float("nan")
        ),
        "n_random": n_random,
        "seed": int(seed),
    }


def behavior_specific_read(
    blocks: Sequence[torch.nn.Module],
    directions: Mapping[int, torch.Tensor],
    path_patch_scores: Mapping[str, Any],
    *,
    path_threshold: float,
    n_random: int = 32,
    seed: int = 1729,
) -> dict[str, Any]:
    """One behavior-specific READ: old weight READ restricted to exact paths.

    ``S_M`` contains every MLP/head whose absolute exact edited-into-clean patch
    contribution to the behavior metric is at least ``path_threshold``. The
    component weight formulas, layer alignment, random normalization, and seed
    schedule exactly match the repaired v2/v3 global weight READ. A family with
    no selected component is explicitly zero because non-path components are
    masked out; it is never silently top-k-filled.
    """

    if not math.isfinite(float(path_threshold)) or path_threshold <= 0:
        raise ValueError("Path threshold must be finite and positive")
    if n_random < 1:
        raise ValueError("n_random must be positive")
    selected_mlps = sorted(
        [
            dict(row)
            for row in path_patch_scores.get("mlps", [])
            if float(row["abs_patched_contribution"]) >= path_threshold
        ],
        key=lambda row: int(row["layer"]),
    )
    selected_heads = sorted(
        [
            dict(row)
            for row in path_patch_scores.get("attention_heads", [])
            if float(row["abs_patched_contribution"]) >= path_threshold
        ],
        key=lambda row: (int(row["layer"]), int(row["head"])),
    )
    identifiers = [row["component"] for row in (*selected_mlps, *selected_heads)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Path component IDs must be unique")

    mlp_rows: list[dict[str, Any]] = []
    for path_row in selected_mlps:
        layer = int(path_row["layer"])
        if layer - 1 not in directions or layer not in directions:
            raise KeyError(f"Directions missing layer alignment for MLP L{layer}")
        weight = qwen_mlp_gain(
            blocks[layer],
            directions[layer - 1],
            n_random=n_random,
            seed=seed + 10_007 * layer,
        )
        block = blocks[layer]
        with torch.no_grad():
            vector = directions[layer - 1].to(
                next(block.parameters()).device,
                next(block.parameters()).dtype,
            )
            output = block.mlp(block.post_attention_layernorm(vector)).float()
            label = directions[layer].to(output.device, torch.float32)
            label_cosine = float(F.cosine_similarity(output, label, dim=0).cpu())
        null = np.asarray(weight["random_gains"], dtype=float)
        mlp_rows.append(
            {
                **path_row,
                **weight,
                "input_direction_layer": layer - 1,
                "label_direction_layer": layer,
                "label_cosine": label_cosine,
                "oriented_normalized_gain": (
                    float(weight["normalized_gain"]) * label_cosine
                ),
                "gain_random_percentile": float(
                    np.mean(null <= float(weight["gain"]))
                ),
            }
        )

    head_rows: list[dict[str, Any]] = []
    for path_row in selected_heads:
        layer = int(path_row["layer"])
        head = int(path_row["head"])
        if layer - 1 not in directions or layer not in directions:
            raise KeyError(f"Directions missing layer alignment for head L{layer}")
        weight = _normalized_single_head_read(
            blocks[layer].self_attn,
            directions[layer - 1],
            directions[layer],
            head=head,
            n_random=n_random,
            seed=seed + 1_009 + 10_007 * layer + 101 * head,
        )
        head_rows.append(
            {
                **path_row,
                **weight,
                "input_direction_layer": layer - 1,
                "label_direction_layer": layer,
                "oriented_normalized_ov": (
                    float(weight["normalized_ov_norm"])
                    * float(weight["label_cosine"])
                ),
            }
        )

    mlp_primary = (
        float(np.mean([row["normalized_gain"] for row in mlp_rows]))
        if mlp_rows
        else 0.0
    )
    attention_primary = (
        float(
            np.mean(
                [row["label_weighted_normalized_ov"] for row in head_rows]
            )
        )
        if head_rows
        else 0.0
    )
    if not math.isfinite(mlp_primary) or not math.isfinite(attention_primary):
        raise ValueError("Behavior-specific READ is non-finite")
    return {
        "status": (
            "OK" if mlp_rows or head_rows else "NO_COMPONENT_ABOVE_THRESHOLD"
        ),
        "path_threshold": float(path_threshold),
        "s_m": {
            "component_ids": identifiers,
            "n_components": len(identifiers),
            "n_mlps": len(mlp_rows),
            "n_attention_heads": len(head_rows),
            "selection_rule": (
                "abs(exact edited-into-clean patched contribution to M) >= "
                f"{float(path_threshold):.6g}"
            ),
        },
        "mlps": mlp_rows,
        "attention_heads": head_rows,
        "mlp_primary": mlp_primary,
        "attention_primary": attention_primary,
        "equal_family_composite": 0.5 * mlp_primary + 0.5 * attention_primary,
        "family_status": {
            "mlp": "MEASURED" if mlp_rows else "NO_PATH_COMPONENTS_ZERO",
            "attention": (
                "MEASURED" if head_rows else "NO_PATH_COMPONENTS_ZERO"
            ),
        },
        "metadata": {
            "estimator_name": "behavior-specific path-restricted weight READ",
            "new_read_estimator_count": 1,
            "component_selection": "exact path patching threshold, not top-k",
            "component_weight_normalization": "identical to repaired v2/v3",
            "input_direction": "v[layer-1]",
            "label_direction": "v[layer]",
            "n_random": int(n_random),
            "base_seed": int(seed),
            "empty_family_policy": (
                "zero with explicit NO_PATH_COMPONENTS_ZERO label"
            ),
            "path_scores_not_used_as_numeric_READ_weights": True,
        },
    }

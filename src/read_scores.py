"""Attribution-based and activation-independent weight-based READ estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from jlens.hooks import ActivationRecorder


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

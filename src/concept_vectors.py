"""Concept-direction construction independent of behavioral interventions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from jlens.hooks import ActivationRecorder


@torch.no_grad()
def mean_residuals(
    lens_model: Any,
    prompts: Sequence[str],
    layers: Iterable[int],
    *,
    position: int = -1,
    max_length: int = 128,
) -> dict[int, torch.Tensor]:
    """Mean post-block residual at one position across a prompt collection."""

    if not prompts:
        raise ValueError("At least one prompt is required")
    layer_list = sorted(set(int(layer) for layer in layers))
    totals = {
        layer: torch.zeros(lens_model.d_model, dtype=torch.float64)
        for layer in layer_list
    }
    for prompt in prompts:
        input_ids = lens_model.encode(prompt, max_length=max_length)
        sequence_length = int(input_ids.shape[1])
        resolved = position if position >= 0 else sequence_length + position
        if not 0 <= resolved < sequence_length:
            raise IndexError(
                f"Position {position} invalid for {sequence_length}-token prompt {prompt!r}"
            )
        with ActivationRecorder(lens_model.layers, at=layer_list) as recorder:
            lens_model.forward(input_ids)
        for layer in layer_list:
            totals[layer] += (
                recorder.activations[layer][0, resolved].detach().double().cpu()
            )
    return {
        layer: (total / len(prompts)).float() for layer, total in totals.items()
    }


def mean_difference_directions(
    lens_model: Any,
    positive_prompts: Sequence[str],
    baseline_prompts: Sequence[str],
    layers: Iterable[int],
    *,
    position: int = -1,
    max_length: int = 128,
    device: str | torch.device = "cpu",
) -> dict[int, torch.Tensor]:
    """Build unit mean-difference directions from positive and baseline cues."""

    layer_list = sorted(set(int(layer) for layer in layers))
    positive = mean_residuals(
        lens_model,
        positive_prompts,
        layer_list,
        position=position,
        max_length=max_length,
    )
    baseline = mean_residuals(
        lens_model,
        baseline_prompts,
        layer_list,
        position=position,
        max_length=max_length,
    )
    directions: dict[int, torch.Tensor] = {}
    for layer in layer_list:
        difference = positive[layer] - baseline[layer]
        if not torch.isfinite(difference).all() or float(difference.norm()) == 0.0:
            raise ValueError(f"Degenerate mean-difference vector at layer {layer}")
        directions[layer] = F.normalize(difference, dim=0).to(device)
    return directions


def cosine_alignment(
    first: Mapping[int, torch.Tensor],
    second: Mapping[int, torch.Tensor],
) -> dict[int, float]:
    """Per-layer cosine similarity for two direction constructions."""

    if set(first) != set(second):
        raise ValueError("Direction dictionaries must contain identical layers")
    return {
        layer: float(
            F.cosine_similarity(
                first[layer].detach().float().cpu(),
                second[layer].detach().float().cpu(),
                dim=0,
            )
        )
        for layer in sorted(first)
    }


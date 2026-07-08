"""Concept-direction construction independent of behavioral interventions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from jlens.hooks import ActivationRecorder


@torch.no_grad()
def residual_prompt_matrices(
    lens_model: Any,
    prompts: Sequence[str],
    layers: Iterable[int],
    *,
    position: int = -1,
    max_length: int = 128,
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Capture one post-block residual per prompt as ``[prompt, d_model]``.

    Batches respect the attention mask and each row's real token positions, so
    padding can never become the measured activation.
    """

    if not prompts:
        raise ValueError("At least one prompt is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    layer_list = sorted(set(int(layer) for layer in layers))
    if not layer_list:
        raise ValueError("At least one layer is required")
    tokenizer = lens_model.tokenizer
    hf_model = getattr(lens_model, "_hf_model", None)
    if hf_model is None:
        raise TypeError("Batched capture requires the official HF J-Lens adapter")
    rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_list}
    for start in range(0, len(prompts), batch_size):
        prompt_batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.to(lens_model.input_device)
        attention_mask = encoded.attention_mask.to(lens_model.input_device)
        selected: list[int] = []
        for row, prompt in enumerate(prompt_batch):
            real_positions = attention_mask[row].nonzero(as_tuple=False).flatten()
            offset = position if position >= 0 else len(real_positions) + position
            if not 0 <= offset < len(real_positions):
                raise IndexError(
                    f"Position {position} invalid for prompt {prompt!r} with "
                    f"{len(real_positions)} tokens"
                )
            selected.append(int(real_positions[offset]))
        with ActivationRecorder(lens_model.layers, at=layer_list) as recorder:
            hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        batch_indices = torch.arange(len(prompt_batch), device=input_ids.device)
        position_indices = torch.tensor(selected, device=input_ids.device)
        for layer in layer_list:
            rows[layer].append(
                recorder.activations[layer][batch_indices, position_indices]
                .detach()
                .float()
                .cpu()
            )
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in rows.items()}


@torch.no_grad()
def mean_residuals(
    lens_model: Any,
    prompts: Sequence[str],
    layers: Iterable[int],
    *,
    position: int = -1,
    max_length: int = 128,
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Mean post-block residual at one position across a prompt collection."""

    matrices = residual_prompt_matrices(
        lens_model,
        prompts,
        layers,
        position=position,
        max_length=max_length,
        batch_size=batch_size,
    )
    return {
        layer: matrix.double().mean(0).float() for layer, matrix in matrices.items()
    }


def prompt_matrix_bank(
    lens_model: Any,
    prompt_sets: Mapping[str, Sequence[str]],
    layers: Iterable[int],
    *,
    position: int = -1,
    max_length: int = 128,
    batch_size: int = 8,
) -> dict[str, dict[int, torch.Tensor]]:
    """Capture a concept-keyed prompt bank in one shared batched sweep."""

    if not prompt_sets or any(not prompts for prompts in prompt_sets.values()):
        raise ValueError("Every concept must contain at least one prompt")
    names = sorted(prompt_sets)
    flat_prompts = [prompt for name in names for prompt in prompt_sets[name]]
    flat = residual_prompt_matrices(
        lens_model,
        flat_prompts,
        layers,
        position=position,
        max_length=max_length,
        batch_size=batch_size,
    )
    output: dict[str, dict[int, torch.Tensor]] = {name: {} for name in names}
    offset = 0
    for name in names:
        count = len(prompt_sets[name])
        for layer, matrix in flat.items():
            output[name][layer] = matrix[offset : offset + count]
        offset += count
    return output


def mean_difference_bank_from_matrices(
    matrices: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    baseline_exclusions: Mapping[str, Iterable[str]] | None = None,
    matched_prompt_slots: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, dict[int, torch.Tensor]]]:
    """Construct a direction bank from cached per-prompt residual matrices."""

    if len(matrices) < 2:
        raise ValueError(
            "At least two concepts are required for one-vs-other baselines"
        )
    concepts = sorted(matrices)
    exclusions = {
        concept: {str(other) for other in excluded}
        for concept, excluded in (baseline_exclusions or {}).items()
    }
    unknown_keys = set(exclusions) - set(concepts)
    unknown_values = (
        set().union(*exclusions.values()) - set(concepts) if exclusions else set()
    )
    if unknown_keys or unknown_values:
        raise ValueError(
            "Baseline exclusions reference unknown concepts: "
            f"keys={sorted(unknown_keys)}, values={sorted(unknown_values)}"
        )
    layers = sorted(matrices[concepts[0]])
    if not layers or any(sorted(matrices[name]) != layers for name in concepts):
        raise ValueError("Every concept matrix must cover identical nonempty layers")
    if (
        matched_prompt_slots
        and len({matrices[name][layers[0]].shape[0] for name in concepts}) != 1
    ):
        raise ValueError("Matched prompt slots require equal prompt counts per concept")
    for name in concepts:
        for layer in layers:
            matrix = matrices[name][layer]
            if matrix.ndim != 2 or matrix.shape[0] < 1:
                raise ValueError(f"Invalid prompt matrix for {name!r}, layer {layer}")

    means = {
        concept: {
            layer: matrix.double().mean(0).float()
            for layer, matrix in concept_matrices.items()
        }
        for concept, concept_matrices in matrices.items()
    }
    directions: dict[str, dict[int, torch.Tensor]] = {}
    for concept in concepts:
        others = [
            name
            for name in concepts
            if name != concept and name not in exclusions.get(concept, set())
        ]
        if not others:
            raise ValueError(f"No eligible baseline concepts remain for {concept!r}")
        directions[concept] = {}
        for layer in layers:
            if matched_prompt_slots:
                baseline = torch.stack([matrices[name][layer] for name in others]).mean(
                    0
                )
                difference = (
                    (matrices[concept][layer] - baseline).double().mean(0).float()
                )
            else:
                baseline = torch.stack([means[name][layer] for name in others]).mean(0)
                difference = means[concept][layer] - baseline
            if not torch.isfinite(difference).all() or float(difference.norm()) == 0.0:
                raise ValueError(
                    f"Degenerate one-vs-other mean difference for {concept!r}, "
                    f"layer {layer}"
                )
            directions[concept][layer] = F.normalize(difference, dim=0).to(device)
    return directions, means


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


def mean_difference_bank(
    lens_model: Any,
    prompt_sets: Mapping[str, Sequence[str]],
    layers: Iterable[int],
    *,
    baseline_exclusions: Mapping[str, Iterable[str]] | None = None,
    matched_prompt_slots: bool = False,
    position: int = -1,
    max_length: int = 128,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, dict[int, torch.Tensor]]]:
    """Build one-vs-other mean-difference directions for many concepts.

    Each concept mean is computed once. Its baseline is the equal-weighted mean
    of eligible *other* concept means, preventing large prompt sets from
    dominating. ``baseline_exclusions`` removes paired foils, aliases, or other
    predeclared leakage risks in addition to the concept itself. With
    ``matched_prompt_slots=True``, each positive cue is contrasted only with
    the same predeclared carrier/template slot for eligible concepts before
    averaging. Returns ``(directions, concept_means)`` for auditability.
    """

    layer_list = sorted(set(int(layer) for layer in layers))
    matrices = prompt_matrix_bank(
        lens_model,
        prompt_sets,
        layer_list,
        position=position,
        max_length=max_length,
        batch_size=batch_size,
    )
    return mean_difference_bank_from_matrices(
        matrices,
        baseline_exclusions=baseline_exclusions,
        matched_prompt_slots=matched_prompt_slots,
        device=device,
    )


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


def direction_retrieval_metrics(
    directions: Mapping[str, Mapping[int, torch.Tensor]],
    heldout_means: Mapping[str, Mapping[int, torch.Tensor]],
) -> dict[int, dict]:
    """Known-answer retrieval control on held-out prompt templates.

    For each layer, every held-out concept mean is scored against every unit
    direction. The true concept's rank and top-1 accuracy are reported without
    fitting a threshold on the held-out set.
    """

    concepts = sorted(directions)
    if concepts != sorted(heldout_means):
        raise ValueError("Directions and held-out means must cover identical concepts")
    layers = sorted(directions[concepts[0]])
    if any(sorted(directions[name]) != layers for name in concepts):
        raise ValueError("Every direction must cover identical layers")
    if any(sorted(heldout_means[name]) != layers for name in concepts):
        raise ValueError("Every held-out mean must cover identical layers")

    results: dict[int, dict] = {}
    for layer in layers:
        residual_matrix = torch.stack(
            [heldout_means[name][layer].detach().float().cpu() for name in concepts]
        )
        direction_matrix = torch.stack(
            [directions[name][layer].detach().float().cpu() for name in concepts]
        )
        scores = residual_matrix @ direction_matrix.T
        true_ranks: list[int] = []
        per_concept: dict[str, dict[str, float | int | str]] = {}
        for index, name in enumerate(concepts):
            true_score = scores[index, index]
            rank = int((scores[index] > true_score).sum() + 1)
            predicted_index = int(scores[index].argmax())
            off_diagonal = torch.cat(
                [scores[index, :index], scores[index, index + 1 :]]
            )
            true_ranks.append(rank)
            per_concept[name] = {
                "rank": rank,
                "predicted": concepts[predicted_index],
                "true_score": float(true_score),
                "margin_vs_best_other": float(true_score - off_diagonal.max()),
            }
        results[layer] = {
            "n_concepts": len(concepts),
            "top1_accuracy": sum(rank == 1 for rank in true_ranks) / len(true_ranks),
            "mean_true_rank": sum(true_ranks) / len(true_ranks),
            "median_true_rank": float(torch.tensor(true_ranks).median()),
            "per_concept": per_concept,
        }
    return results

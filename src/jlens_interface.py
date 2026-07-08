"""Thin, explicit interface around the official Jacobian Lens package."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PUBLISHED_LENSES = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "repo_id": "neuronpedia/jacobian-lens",
        "revision": "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a",
        "filename": (
            "qwen2.5-7b-it/jlens/Salesforce-wikitext/"
            "Qwen2.5-7B-Instruct_jacobian_lens.pt"
        ),
    }
}


def load_published_lens(model_id: str, *, local_files_only: bool = True) -> Any:
    """Load a pinned published lens through the official deserializer."""

    import jlens
    from huggingface_hub import hf_hub_download

    try:
        spec = PUBLISHED_LENSES[model_id]
    except KeyError as exc:
        raise ValueError(f"No published lens registered for {model_id!r}") from exc
    path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        revision=spec["revision"],
        local_files_only=local_files_only,
    )
    return jlens.JacobianLens.load(path)


def load_local_lens(path: str | Path) -> Any:
    """Load a locally fitted lens through :class:`jlens.JacobianLens`."""

    import jlens

    return jlens.JacobianLens.load(str(path))


def validate_lens(lens: Any, lens_model: Any) -> None:
    """Fail if a lens cannot act on the supplied wrapped model."""

    if int(lens.d_model) != int(lens_model.d_model):
        raise ValueError(
            f"Lens d_model={lens.d_model} != model d_model={lens_model.d_model}"
        )
    bad = [layer for layer in lens.source_layers if not 0 <= layer < lens_model.n_layers]
    if bad:
        raise ValueError(f"Lens contains out-of-range source layers: {bad}")


def workspace_layers(
    n_layers: int,
    source_layers: Iterable[int],
    *,
    lower_fraction: float = 0.40,
    upper_fraction: float = 0.90,
) -> list[int]:
    """Return the preregistered middle 40–90% block-output band."""

    if not 0 <= lower_fraction < upper_fraction <= 1:
        raise ValueError("Expected 0 <= lower_fraction < upper_fraction <= 1")
    lower = math.floor(n_layers * lower_fraction)
    upper = math.ceil(n_layers * upper_fraction)
    available = set(int(layer) for layer in source_layers)
    layers = [layer for layer in range(lower, upper) if layer in available]
    if not layers:
        raise ValueError("No fitted lens layers overlap the requested workspace band")
    return layers


def unembedding_weight(lens_model: Any) -> torch.Tensor:
    """Return the model's actual LM-head matrix (Qwen embeddings are untied)."""

    if not hasattr(lens_model, "_lm_head"):
        raise TypeError("The supplied lens model does not expose an HF LM head")
    return lens_model._lm_head.weight  # noqa: SLF001 - official adapter reference


def jlens_direction(
    lens: Any,
    lens_model: Any,
    token_id: int,
    layer: int,
    *,
    fold_rms_gain: bool = False,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Construct the unit source direction ``normalize(J_l.T @ W_U[c])``.

    ``fold_rms_gain=False`` is the preregistered raw ``W_U J`` convention.
    The optional gain-folded variant is a separately labelled sensitivity
    analysis; it must not replace the preregistered direction silently.
    """

    validate_lens(lens, lens_model)
    if layer not in lens.jacobians:
        raise ValueError(f"Layer {layer} was not fitted; available={lens.source_layers}")
    weight = unembedding_weight(lens_model)
    if not 0 <= int(token_id) < weight.shape[0]:
        raise ValueError(f"token_id={token_id} outside vocabulary size {weight.shape[0]}")

    token_row = weight[int(token_id)].detach().float().cpu()
    if fold_rms_gain:
        final_norm = getattr(lens_model, "_final_norm", None)
        gain = getattr(final_norm, "weight", None)
        if gain is None:
            raise TypeError("Final norm has no weight to fold into the direction")
        token_row = token_row * gain.detach().float().cpu()
    jacobian = lens.jacobians[int(layer)].float().cpu()
    direction = jacobian.T @ token_row
    if not torch.isfinite(direction).all() or float(direction.norm()) == 0.0:
        raise ValueError(f"Degenerate J-Lens direction for token={token_id}, layer={layer}")
    return F.normalize(direction, dim=0).to(device)


def jlens_directions(
    lens: Any,
    lens_model: Any,
    token_id: int,
    layers: Iterable[int],
    *,
    fold_rms_gain: bool = False,
    device: str | torch.device = "cpu",
) -> dict[int, torch.Tensor]:
    """Construct one unit concept direction for every requested source layer."""

    return {
        int(layer): jlens_direction(
            lens,
            lens_model,
            token_id,
            int(layer),
            fold_rms_gain=fold_rms_gain,
            device=device,
        )
        for layer in layers
    }


def jlens_direction_bank(
    lens: Any,
    lens_model: Any,
    token_ids: Iterable[int],
    layers: Iterable[int],
    *,
    fold_rms_gain: bool = False,
    compute_device: str | torch.device = "cuda",
    output_device: str | torch.device = "cuda",
) -> dict[int, dict[int, torch.Tensor]]:
    """Batch direction construction for many concepts and layers.

    Moving one Jacobian at a time to the GPU avoids hundreds of slow CPU
    matrix-vector products while keeping peak memory bounded.
    """

    validate_lens(lens, lens_model)
    unique_tokens = sorted(set(int(token_id) for token_id in token_ids))
    layer_list = sorted(set(int(layer) for layer in layers))
    if not unique_tokens or not layer_list:
        raise ValueError("At least one token and layer are required")
    weight = unembedding_weight(lens_model).detach().float().cpu()
    if unique_tokens[0] < 0 or unique_tokens[-1] >= weight.shape[0]:
        raise ValueError("A requested token ID is outside the vocabulary")
    rows = weight[unique_tokens]
    if fold_rms_gain:
        gain = getattr(getattr(lens_model, "_final_norm", None), "weight", None)
        if gain is None:
            raise TypeError("Final norm has no weight to fold into directions")
        rows = rows * gain.detach().float().cpu().unsqueeze(0)
    rows = rows.to(compute_device)
    output: dict[int, dict[int, torch.Tensor]] = {
        token_id: {} for token_id in unique_tokens
    }
    for layer in layer_list:
        if layer not in lens.jacobians:
            raise ValueError(f"Layer {layer} was not fitted")
        jacobian = lens.jacobians[layer].to(compute_device, torch.float32)
        directions = F.normalize(rows @ jacobian, dim=-1)
        if not torch.isfinite(directions).all():
            raise ValueError(f"Non-finite batched directions at layer {layer}")
        for row_index, token_id in enumerate(unique_tokens):
            output[token_id][layer] = directions[row_index].to(output_device)
        del jacobian, directions
    return output


def write_by_position(
    residuals: Mapping[int, torch.Tensor],
    directions: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Project clean post-block residuals onto unit directions in fp32.

    Returned tensors have shape ``[batch, n_positions]``. Negative positions
    follow normal Python indexing. No aggregation is hidden in this function.
    """

    output: dict[int, torch.Tensor] = {}
    for layer, direction in directions.items():
        if layer not in residuals:
            raise KeyError(f"Missing residual activation for layer {layer}")
        hidden = residuals[layer].float()
        if hidden.ndim != 3:
            raise ValueError(f"Expected [batch, seq, d], got {tuple(hidden.shape)}")
        if positions is not None:
            hidden = hidden[:, list(positions), :]
        vector = direction.to(device=hidden.device, dtype=torch.float32)
        output[layer] = torch.einsum("bsd,d->bs", hidden, vector)
    return output


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    """One-indexed rank of ``token_id`` in a one-dimensional logit vector."""

    if logits.ndim != 1:
        raise ValueError(f"Expected one-dimensional logits, got {tuple(logits.shape)}")
    value = logits[int(token_id)]
    return int((logits > value).sum().item() + 1)

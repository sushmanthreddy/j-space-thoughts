"""Reproducible interface to Anthropic's official Jacobian Lens package.

This module owns the final pipeline's model/lens boundary: it loads the pinned
Qwen model in bf16, checks the zero-copy J-Lens wrapper, constructs published
J-Lens directions, and measures clean residual projections (``WRITTEN``).
Behavioral interventions deliberately live elsewhere.
"""

from __future__ import annotations

import gc
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_REVISIONS = {MODEL_ID: MODEL_REVISION}

PUBLISHED_LENSES = {
    MODEL_ID: {
        "repo_id": "neuronpedia/jacobian-lens",
        "revision": "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a",
        "filename": (
            "qwen2.5-7b-it/jlens/Salesforce-wikitext/"
            "Qwen2.5-7B-Instruct_jacobian_lens.pt"
        ),
    }
}


@dataclass(frozen=True)
class ModelBundle:
    """Pinned HF model and zero-copy official J-Lens adapter.

    The HF parameters are frozen by :func:`jlens.from_hf`; downstream READ
    gradients are taken only with respect to residual activations.
    """

    model_id: str
    revision: str
    hf_model: torch.nn.Module
    tokenizer: Any
    lens_model: Any


def set_seed(seed: int = 1729, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without changing model precision."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(
    model_id: str = MODEL_ID,
    *,
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    attn_implementation: str | None = None,
    local_files_only: bool = True,
) -> ModelBundle:
    """Load a pinned causal LM and wrap the same object with official J-Lens.

    ``local_files_only=True`` makes a missing explicit model download fail
    rather than silently resolving a different remote revision. The final
    experiment uses the defaults: pinned Qwen2.5-7B-Instruct in bf16.
    """

    import jlens
    import transformers

    resolved_revision = revision or MODEL_REVISIONS.get(model_id)
    if resolved_revision is None:
        raise ValueError(f"No pinned revision registered for {model_id!r}")
    kwargs: dict[str, Any] = {
        "revision": resolved_revision,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    hf_model.to(device)
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        revision=resolved_revision,
        local_files_only=local_files_only,
    )
    lens_model = jlens.from_hf(hf_model, tokenizer)
    return ModelBundle(
        model_id=model_id,
        revision=resolved_revision,
        hf_model=hf_model,
        tokenizer=tokenizer,
        lens_model=lens_model,
    )


def release_model(bundle: ModelBundle | None = None) -> None:
    """Drop a local bundle reference and release unused CUDA allocations."""

    if bundle is not None:
        del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def single_token_id(tokenizer: Any, text: str) -> int:
    """Return the sole vocabulary ID for ``text`` or fail with diagnostics."""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        pieces = [tokenizer.decode([token_id]) for token_id in token_ids]
        raise ValueError(
            f"Expected one token for {text!r}, got {len(token_ids)}: "
            f"ids={token_ids}, pieces={pieces!r}"
        )
    return int(token_ids[0])


def concept_token_id(tokenizer: Any, concept: str) -> tuple[int, str]:
    """Resolve a word-like concept to one token, preferring a word boundary."""

    candidates = [concept] if concept.startswith(" ") else [f" {concept}", concept]
    diagnostics: list[str] = []
    for surface in candidates:
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        diagnostics.append(f"{surface!r}->{token_ids}")
        if len(token_ids) == 1:
            return int(token_ids[0]), surface
    raise ValueError(
        f"No single-token surface for concept {concept!r}: " + "; ".join(diagnostics)
    )


def decode_topk(tokenizer: Any, logits: torch.Tensor, k: int = 10) -> list[dict]:
    """Decode a one-dimensional logit vector into JSON-friendly records."""

    values, indices = logits.float().topk(k)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(value),
        }
        for value, token_id in zip(values.cpu(), indices.cpu(), strict=True)
    ]


@torch.no_grad()
def batched_next_token_records(
    hf_model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    expected_token_ids: Sequence[int],
    *,
    batch_size: int = 8,
    top_k: int = 10,
    max_length: int = 128,
) -> list[dict[str, Any]]:
    """Rank one predeclared next token for every prompt in padded batches."""

    if len(prompts) != len(expected_token_ids) or not prompts:
        raise ValueError("Prompts and expected token IDs must align and be nonempty")
    if batch_size < 1 or top_k < 1:
        raise ValueError("batch_size and top_k must be positive")
    device = next(hf_model.parameters()).device
    rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = list(prompts[start : start + batch_size])
        token_batch = [
            int(value) for value in expected_token_ids[start : start + batch_size]
        ]
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        output = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        for batch_index, (prompt, expected_id) in enumerate(
            zip(prompt_batch, token_batch, strict=True)
        ):
            positions = attention_mask[batch_index].nonzero(as_tuple=False).flatten()
            position = int(positions[-1])
            logits = output[batch_index, position].float()
            rank = int((logits > logits[expected_id]).sum().item() + 1)
            rows.append(
                {
                    "index": start + batch_index,
                    "prompt": prompt,
                    "n_tokens": int(len(positions)),
                    "expected_token_id": expected_id,
                    "expected_token": tokenizer.decode([expected_id]),
                    "expected_logit": float(logits[expected_id].cpu()),
                    "rank": rank,
                    "top1_correct": int(rank == 1),
                    "top5_correct": int(rank <= 5),
                    "top10_correct": int(rank <= 10),
                    "top_tokens": decode_topk(
                        tokenizer, logits, min(top_k, logits.numel())
                    ),
                }
            )
        del output
    return rows


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


@torch.no_grad()
def capture_residuals(
    lens_model: Any,
    input_ids: torch.Tensor,
    layers: Iterable[int],
    *,
    detach: bool = True,
) -> dict[int, torch.Tensor]:
    """Capture post-block, pre-final-norm residuals with official hooks."""

    from jlens.hooks import ActivationRecorder

    layer_list = sorted(set(int(layer) for layer in layers))
    if not layer_list:
        return {}
    with ActivationRecorder(lens_model.layers, at=layer_list) as recorder:
        lens_model.forward(input_ids)
    return {
        layer: (value.detach() if detach else value)
        for layer, value in recorder.activations.items()
    }


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
    """Capture one residual per prompt as a ``[prompt, d_model]`` matrix.

    The selected position is resolved against each row's attention mask, so
    padding never becomes the measured concept activation. Returned matrices
    are detached fp32 CPU tensors, matching the published experiment.
    """

    from jlens.hooks import ActivationRecorder

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


def written_scores(
    residual_matrix: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Return fp32 ``hᵀv`` WRITTEN scores for clean residual rows.

    ``residual_matrix`` must have shape ``[n, d_model]``. ``directions`` may
    be one shared vector ``[d_model]`` or one matched vector per row
    ``[n, d_model]``. No thresholding or aggregation is hidden here.
    """

    hidden = residual_matrix.float()
    vectors = directions.to(device=hidden.device, dtype=torch.float32)
    if hidden.ndim != 2:
        raise ValueError(
            f"Expected residual matrix [n, d_model], got {tuple(hidden.shape)}"
        )
    if vectors.ndim == 1:
        if vectors.shape[0] != hidden.shape[1]:
            raise ValueError("WRITTEN direction dimension does not match residuals")
        return hidden @ vectors
    if vectors.ndim == 2 and vectors.shape == hidden.shape:
        return torch.einsum("bd,bd->b", hidden, vectors)
    raise ValueError(
        "Directions must be [d_model] or match residual shape [n, d_model]; "
        f"got {tuple(vectors.shape)}"
    )


@torch.no_grad()
def hf_wrapper_logit_kl(
    bundle: ModelBundle,
    prompts: Iterable[str],
    *,
    max_length: int = 128,
) -> list[dict[str, Any]]:
    """Compare exact HF logits with logits reconstructed by J-Lens.

    Every token position is checked. The wrapper path records the final decoder
    block and applies the adapter's final RMSNorm and LM head.
    """

    from jlens.hooks import ActivationRecorder

    results: list[dict[str, Any]] = []
    final_layer = bundle.lens_model.n_layers - 1
    for index, prompt in enumerate(prompts):
        input_ids = bundle.lens_model.encode(prompt, max_length=max_length)
        hf_logits = bundle.hf_model(
            input_ids=input_ids, use_cache=False
        ).logits.float()
        with ActivationRecorder(
            bundle.lens_model.layers, at=[final_layer]
        ) as recorder:
            bundle.lens_model.forward(input_ids)
        wrapper_logits = bundle.lens_model.unembed(
            recorder.activations[final_layer]
        ).float()
        reference_probs = hf_logits.softmax(dim=-1)
        kl_per_position = F.kl_div(
            wrapper_logits.log_softmax(dim=-1),
            reference_probs,
            reduction="none",
        ).sum(dim=-1)
        results.append(
            {
                "index": index,
                "prompt": prompt,
                "n_tokens": int(input_ids.shape[1]),
                "mean_kl": float(kl_per_position.mean().cpu()),
                "max_kl": float(kl_per_position.max().cpu()),
                "max_abs_logit_error": float(
                    (wrapper_logits - hf_logits).abs().max().cpu()
                ),
            }
        )
    return results


def enforce_kl_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 1e-3,
) -> dict[str, Any]:
    """Enforce the preregistered HF/J-Lens mean-KL agreement gate."""

    if not records:
        raise ValueError("KL gate requires at least one prompt record")
    if threshold <= 0:
        raise ValueError("KL threshold must be positive")
    max_mean_kl = max(float(record["mean_kl"]) for record in records)
    if not math.isfinite(max_mean_kl):
        raise ValueError("KL gate received a non-finite mean KL")
    if max_mean_kl > threshold:
        raise RuntimeError(
            f"HF/J-Lens logit gate failed: max mean KL {max_mean_kl:.6g} "
            f"> {threshold:.6g}"
        )
    return {
        "status": "PASS",
        "threshold": float(threshold),
        "n": len(records),
        "max_mean_kl": max_mean_kl,
        "rows": list(records),
    }

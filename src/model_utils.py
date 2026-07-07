"""Model loading, token checks, reproducibility, and the G1 logit gate."""

from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from jlens.hooks import ActivationRecorder


MODEL_REVISIONS = {
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen/Qwen2.5-14B-Instruct": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
}


@dataclass
class ModelBundle:
    """The HF model and the zero-copy J-Lens adapter used by experiments."""

    model_id: str
    revision: str
    hf_model: torch.nn.Module
    tokenizer: Any
    lens_model: Any


def set_seed(seed: int = 1729, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without silently changing precision."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    attn_implementation: str | None = None,
    local_files_only: bool = True,
) -> ModelBundle:
    """Load a pinned causal LM and wrap the same object with official J-Lens.

    Parameters are frozen by :func:`jlens.from_hf`; gradients are taken only
    with respect to residual activations. ``local_files_only=True`` makes a
    missing explicit ``hf download`` fail early rather than changing revisions.
    """

    import jlens
    import transformers

    revision = revision or MODEL_REVISIONS.get(model_id)
    if revision is None:
        raise ValueError(f"No pinned revision registered for {model_id!r}")

    kwargs: dict[str, Any] = {
        "revision": revision,
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
        revision=revision,
        local_files_only=local_files_only,
    )
    lens_model = jlens.from_hf(hf_model, tokenizer)
    return ModelBundle(model_id, revision, hf_model, tokenizer, lens_model)


def single_token_id(tokenizer: Any, text: str) -> int:
    """Return the sole vocabulary ID for ``text`` or fail with diagnostics."""

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        pieces = [tokenizer.decode([token_id]) for token_id in ids]
        raise ValueError(
            f"Expected one token for {text!r}, got {len(ids)}: "
            f"ids={ids}, pieces={pieces!r}"
        )
    return int(ids[0])


def concept_token_id(tokenizer: Any, concept: str) -> tuple[int, str]:
    """Resolve a word-like concept to a single token, preferring word boundary.

    Qwen's vocabulary usually represents latent word labels with a leading
    space (for example, ``" spider"`` is one token while ``"spider"`` is
    two). The returned surface form is persisted with every measurement.
    """

    candidates = [concept] if concept.startswith(" ") else [f" {concept}", concept]
    diagnostics: list[str] = []
    for surface in candidates:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        diagnostics.append(f"{surface!r}->{ids}")
        if len(ids) == 1:
            return int(ids[0]), surface
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
    """Rank one predeclared token after each prompt using padded batches."""

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


@torch.no_grad()
def capture_residuals(
    lens_model: Any,
    input_ids: torch.Tensor,
    layers: Iterable[int],
    *,
    detach: bool = True,
) -> dict[int, torch.Tensor]:
    """Capture post-block, pre-final-norm residuals using J-Lens hooks."""

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
def hf_wrapper_logit_kl(
    bundle: ModelBundle,
    prompts: Iterable[str],
    *,
    max_length: int = 128,
) -> list[dict]:
    """Compare exact HF logits with logits reconstructed through J-Lens hooks.

    The wrapper path records the last decoder block and applies the adapter's
    final RMSNorm plus LM head. Results cover every prompt token, not only the
    last position, making this a stronger G1 check than a next-token spot test.
    """

    results: list[dict] = []
    final_layer = bundle.lens_model.n_layers - 1
    for index, prompt in enumerate(prompts):
        input_ids = bundle.lens_model.encode(prompt, max_length=max_length)
        hf_logits = bundle.hf_model(input_ids=input_ids, use_cache=False).logits.float()
        with ActivationRecorder(bundle.lens_model.layers, at=[final_layer]) as recorder:
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


def release_model(bundle: ModelBundle | None = None) -> None:
    """Release local references and return unused CUDA allocations to PyTorch."""

    if bundle is not None:
        del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

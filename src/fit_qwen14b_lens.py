"""Resumable, pinned 14B Jacobian-lens fit used by notebook 06."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import jlens
from datasets import load_dataset

from src.jlens_iface import workspace_layers
from src.model_utils import load_model, set_seed


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SEED = 1729


def pinned_wikitext_prompts(
    n_prompts: int,
    *,
    min_chars: int = 600,
) -> list[str]:
    """Match the official J-Lens WikiText selection at a pinned revision."""

    if n_prompts < 1:
        raise ValueError("n_prompts must be positive")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        revision=WIKITEXT_REVISION,
        streaming=True,
    )
    prompts: list[str] = []
    for record in dataset:
        text = record["text"]
        if len(text.strip()) >= min_chars:
            prompts.append(text)
            if len(prompts) == n_prompts:
                return prompts
    raise RuntimeError(f"Pinned WikiText stream yielded only {len(prompts)} prompts")


def fit_qwen14b_lens(
    *,
    n_prompts: int = 100,
    dim_batch: int = 128,
    max_seq_len: int = 128,
    checkpoint_every: int = 10,
) -> dict[str, Any]:
    """Fit the preregistered workspace band and persist full provenance."""

    set_seed(SEED)
    output_dir = ROOT / "data" / "lenses"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "qwen2.5-14b_fit_ckpt.pt"
    lens_path = output_dir / f"qwen2.5-14b_jlens_{n_prompts}prompts.pt"
    metadata_path = output_dir / f"qwen2.5-14b_jlens_{n_prompts}prompts.json"

    prompts = pinned_wikitext_prompts(n_prompts)
    prompt_hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]
    bundle = load_model(MODEL_ID)
    layers = workspace_layers(
        bundle.lens_model.n_layers,
        range(bundle.lens_model.n_layers - 1),
    )
    started = time.time()
    lens = jlens.fit(
        bundle.lens_model,
        prompts,
        source_layers=layers,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=checkpoint_every,
        resume=True,
    )
    lens.save(str(lens_path))
    metadata: dict[str, Any] = {
        "seed": SEED,
        "model_id": bundle.model_id,
        "model_revision": bundle.revision,
        "wikitext_revision": WIKITEXT_REVISION,
        "selection": f"first {n_prompts} train records with >=600 characters",
        "prompt_sha256": prompt_hashes,
        "n_prompts_requested": n_prompts,
        "n_prompts_fitted": lens.n_prompts,
        "source_layers": layers,
        "target_layer": bundle.lens_model.n_layers - 1,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "checkpoint_every": checkpoint_every,
        "elapsed_seconds_this_run": time.time() - started,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "lens_path": str(lens_path.relative_to(ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"14B LENS FIT PASS: n={lens.n_prompts}, layers={layers}, "
        f"elapsed={metadata['elapsed_seconds_this_run']:.1f}s, path={lens_path}"
    )
    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    fit_qwen14b_lens()

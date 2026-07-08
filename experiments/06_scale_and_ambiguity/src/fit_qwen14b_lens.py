"""Resumable, pinned 14B Jacobian-lens fit used by notebook 06."""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jlens
import torch
from datasets import load_dataset

from src.jlens_iface import workspace_layers
from src.model_utils import MODEL_REVISIONS, load_model, release_model, set_seed


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SEED = 1729
CHECKPOINT_PROVENANCE_SCHEMA = "qwen14b-jlens-checkpoint-provenance-v1"
SHA256_HEX_LENGTH = 64


def _atomic_json_write(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object through a same-directory atomic replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    try:
        with target.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} JSON at {target}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} at {target} must be a JSON object")
    return payload


@contextmanager
def exclusive_fit_lock(path: str | Path) -> Iterator[Path]:
    """Hold a non-blocking process lock for the entire checkpointed fit.

    The lock file intentionally remains after release: kernel ``flock`` state,
    rather than file existence, determines ownership.  Its JSON content is only
    diagnostic and is replaced after the lock is acquired.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "owner metadata unavailable"
            raise RuntimeError(
                f"Another Qwen-14B lens fit holds {target}: {owner}"
            ) from error
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "acquired_unix_seconds": time.time(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield target
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def checkpoint_provenance_payload(
    *,
    model_id: str,
    model_revision: str,
    prompt_sha256: Sequence[str],
    source_layers: Sequence[int],
    target_layer: int,
    n_prompts: int,
    max_seq_len: int,
    dim_batch: int,
    checkpoint_every: int,
    wikitext_revision: str = WIKITEXT_REVISION,
) -> dict[str, Any]:
    """Build the exact, stable contract bound to a resumable checkpoint."""

    integer_fields = {
        "n_prompts": n_prompts,
        "max_seq_len": max_seq_len,
        "dim_batch": dim_batch,
        "checkpoint_every": checkpoint_every,
    }
    for name, value in integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    hashes = [str(value) for value in prompt_sha256]
    if len(hashes) != n_prompts:
        raise ValueError("Prompt hash count must equal n_prompts")
    if len(set(hashes)) != len(hashes):
        raise ValueError("Prompt hashes must be unique and retain dataset order")
    if any(
        len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise ValueError("Prompt hashes must be lowercase SHA-256 hex digests")
    layers = [int(layer) for layer in source_layers]
    if not layers or layers != sorted(set(layers)):
        raise ValueError("source_layers must be nonempty, sorted, and unique")
    if layers[-1] >= int(target_layer):
        raise ValueError("Every source layer must precede target_layer")
    for name, value in (
        ("model_id", model_id),
        ("model_revision", model_revision),
        ("wikitext_revision", wikitext_revision),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string")
    return {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA,
        "model_id": model_id,
        "model_revision": model_revision,
        "wikitext_revision": wikitext_revision,
        "prompt_sha256": hashes,
        "source_layers": layers,
        "target_layer": int(target_layer),
        "n_prompts": n_prompts,
        "max_seq_len": max_seq_len,
        "dim_batch": dim_batch,
        "checkpoint_every": checkpoint_every,
    }


def prepare_checkpoint_provenance(
    checkpoint_path: str | Path,
    sidecar_path: str | Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a new sidecar or fail closed before resuming a checkpoint."""

    checkpoint = Path(checkpoint_path)
    sidecar = Path(sidecar_path)
    expected_payload = dict(expected)
    if checkpoint.exists():
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError(f"Existing checkpoint is not a nonempty file: {checkpoint}")
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            raise RuntimeError(
                "Refusing to resume an existing checkpoint without a nonempty "
                f"matching provenance sidecar: checkpoint={checkpoint}, sidecar={sidecar}"
            )
        actual = _read_json_object(sidecar, label="checkpoint provenance")
        if actual != expected_payload:
            fields = sorted(
                key
                for key in set(actual) | set(expected_payload)
                if actual.get(key) != expected_payload.get(key)
            )
            raise ValueError(
                "Checkpoint provenance mismatch; refusing a mixed fit. "
                f"Differing fields: {fields}"
            )
        return {
            "mode": "resume_validated",
            "checkpoint_bytes": checkpoint.stat().st_size,
            "sidecar_path": str(sidecar),
        }

    sidecar_replaced = sidecar.exists()
    _atomic_json_write(sidecar, expected_payload)
    return {
        "mode": "new_fit",
        "checkpoint_bytes": 0,
        "sidecar_path": str(sidecar),
        "stale_sidecar_replaced": sidecar_replaced,
    }


def _atomic_lens_save(lens: Any, path: str | Path) -> None:
    """Save a lens completely before exposing the final filename."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        lens.save(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("Lens serializer did not create a nonempty file")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def validate_completed_fit_artifacts(
    lens_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_n_prompts: int = 100,
    expected_source_layers: Sequence[int] | None = None,
    expected_target_layer: int = 47,
    expected_max_seq_len: int = 128,
    expected_dim_batch: int = 128,
    expected_checkpoint_every: int = 10,
) -> dict[str, Any]:
    """Validate the cheap completion contract before notebook 06 skips fitting."""

    lens_file = Path(lens_path)
    metadata_file = Path(metadata_path)
    if not lens_file.is_file() or lens_file.stat().st_size == 0:
        raise ValueError(f"Final lens is missing or empty: {lens_file}")
    if not metadata_file.is_file() or metadata_file.stat().st_size == 0:
        raise ValueError(f"Final lens metadata is missing or empty: {metadata_file}")
    metadata = _read_json_object(metadata_file, label="final lens metadata")
    expected_revision = MODEL_REVISIONS[MODEL_ID]
    layers = (
        list(expected_source_layers)
        if expected_source_layers is not None
        else workspace_layers(48, range(47))
    )
    checks = {
        "model_id": (metadata.get("model_id"), MODEL_ID),
        "model_revision": (metadata.get("model_revision"), expected_revision),
        "wikitext_revision": (
            metadata.get("wikitext_revision"),
            WIKITEXT_REVISION,
        ),
        "n_prompts_requested": (
            metadata.get("n_prompts_requested"),
            expected_n_prompts,
        ),
        "n_prompts_fitted": (
            metadata.get("n_prompts_fitted"),
            expected_n_prompts,
        ),
        "selection": (
            metadata.get("selection"),
            f"first {expected_n_prompts} train records with >=600 characters",
        ),
        "source_layers": (metadata.get("source_layers"), layers),
        "target_layer": (metadata.get("target_layer"), expected_target_layer),
        "max_seq_len": (metadata.get("max_seq_len"), expected_max_seq_len),
        "dim_batch": (metadata.get("dim_batch"), expected_dim_batch),
        "checkpoint_every": (
            metadata.get("checkpoint_every"),
            expected_checkpoint_every,
        ),
    }
    wrong = [name for name, (actual, expected) in checks.items() if actual != expected]
    if wrong:
        raise ValueError(f"Final lens metadata mismatch in fields: {wrong}")
    declared = metadata.get("lens_path")
    if not isinstance(declared, str):
        raise ValueError("Final lens metadata has no string lens_path")
    declared_path = Path(declared)
    if not declared_path.is_absolute():
        declared_path = ROOT / declared_path
    if declared_path.resolve() != lens_file.resolve():
        raise ValueError("Final lens metadata points to a different lens file")
    hashes = metadata.get("prompt_sha256")
    if not isinstance(hashes, list) or len(hashes) != expected_n_prompts:
        raise ValueError("Final lens metadata has the wrong prompt-hash count")
    if len(set(hashes)) != len(hashes) or any(
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise ValueError("Final lens metadata has invalid or duplicate prompt hashes")
    return metadata


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

    for name, value in (
        ("n_prompts", n_prompts),
        ("dim_batch", dim_batch),
        ("max_seq_len", max_seq_len),
        ("checkpoint_every", checkpoint_every),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    output_dir = ROOT / "data" / "lenses"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "qwen2.5-14b_fit_ckpt.pt"
    checkpoint_provenance_path = output_dir / (
        "qwen2.5-14b_fit_ckpt.provenance.json"
    )
    lock_path = output_dir / "qwen2.5-14b_fit_ckpt.lock"
    lens_path = output_dir / f"qwen2.5-14b_jlens_{n_prompts}prompts.pt"
    metadata_path = output_dir / f"qwen2.5-14b_jlens_{n_prompts}prompts.json"

    with exclusive_fit_lock(lock_path):
        # Fail before a dataset fetch or model allocation when a legacy/stale
        # checkpoint cannot possibly satisfy the new resume contract.
        if checkpoint_path.exists() and (
            not checkpoint_provenance_path.is_file()
            or checkpoint_provenance_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Refusing to resume the existing 14B checkpoint without its "
                f"provenance sidecar: {checkpoint_provenance_path}"
            )

        set_seed(SEED)
        prompts = pinned_wikitext_prompts(n_prompts)
        prompt_hashes = [
            hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts
        ]
        bundle: Any | None = None
        try:
            bundle = load_model(MODEL_ID)
            layers = workspace_layers(
                bundle.lens_model.n_layers,
                range(bundle.lens_model.n_layers - 1),
            )
            target_layer = int(bundle.lens_model.n_layers - 1)
            expected_checkpoint_provenance = checkpoint_provenance_payload(
                model_id=bundle.model_id,
                model_revision=bundle.revision,
                prompt_sha256=prompt_hashes,
                source_layers=layers,
                target_layer=target_layer,
                n_prompts=n_prompts,
                max_seq_len=max_seq_len,
                dim_batch=dim_batch,
                checkpoint_every=checkpoint_every,
            )
            checkpoint_state = prepare_checkpoint_provenance(
                checkpoint_path,
                checkpoint_provenance_path,
                expected_checkpoint_provenance,
            )
            started = time.time()
            lens = jlens.fit(
                bundle.lens_model,
                prompts,
                source_layers=layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                checkpoint_path=str(checkpoint_path),
                checkpoint_every=checkpoint_every,
                resume=True,
            )
            if int(lens.n_prompts) != n_prompts:
                raise RuntimeError(
                    f"J-Lens fitted {lens.n_prompts} prompts, expected {n_prompts}; "
                    "refusing to publish a partial final lens"
                )
            _atomic_lens_save(lens, lens_path)
            sidecar_sha256 = hashlib.sha256(
                checkpoint_provenance_path.read_bytes()
            ).hexdigest()
            metadata: dict[str, Any] = {
                "seed": SEED,
                "model_id": bundle.model_id,
                "model_revision": bundle.revision,
                "wikitext_revision": WIKITEXT_REVISION,
                "selection": (
                    f"first {n_prompts} train records with >=600 characters"
                ),
                "prompt_sha256": prompt_hashes,
                "n_prompts_requested": n_prompts,
                "n_prompts_fitted": int(lens.n_prompts),
                "source_layers": layers,
                "target_layer": target_layer,
                "dim_batch": dim_batch,
                "max_seq_len": max_seq_len,
                "checkpoint_every": checkpoint_every,
                "elapsed_seconds_this_run": time.time() - started,
                "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_provenance_path": str(
                    checkpoint_provenance_path.relative_to(ROOT)
                ),
                "checkpoint_provenance_sha256": sidecar_sha256,
                "checkpoint_resume_mode": checkpoint_state["mode"],
                "exclusive_lock_path": str(lock_path.relative_to(ROOT)),
                "atomic_final_artifacts": True,
                "lens_path": str(lens_path.relative_to(ROOT)),
            }
            _atomic_json_write(metadata_path, metadata)
            print(
                f"14B LENS FIT PASS: n={lens.n_prompts}, layers={layers}, "
                f"elapsed={metadata['elapsed_seconds_this_run']:.1f}s, "
                f"path={lens_path}"
            )
            return metadata
        finally:
            # Dropping the caller's reference before the second collection is
            # essential in IPython, where failed-cell tracebacks are retained.
            if bundle is not None:
                release_model(bundle)
            bundle = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    fit_qwen14b_lens()

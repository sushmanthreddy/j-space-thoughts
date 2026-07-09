"""Matched v7 dataset construction and clean verification.

V7 keeps the frozen reciprocal roster, split, L16 J-Lens directions, and
WRITTEN threshold.  The engine and idle-dashboard tasks use the same answer
tokens and the same ``logit(answer_A) - logit(answer_B)`` metric.  The dashboard
extends the engine's exact concept-bearing prefix by stating the answer and
asking the model to copy it.

This module is clean-data only.  It imports no intervention, patching, causal,
or READ implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.datasets import (
    G1_PROMPTS,
    build_symmetric_causal_candidates,
    continuation_token_id,
    validate_sanitized_manifest,
)
from src.jlens_interface import (
    MODEL_ID,
    MODEL_REVISION,
    concept_token_id,
    decode_topk,
    enforce_kl_gate,
    hf_wrapper_logit_kl,
    jlens_direction_bank,
    load_model,
    load_published_lens,
    release_model,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "v7"
RAW_DIR = RESULTS_DIR / "raw"
DATASET_PATH = RAW_DIR / "matched_dataset_v7.json"
MANIFEST_PATH = RESULTS_DIR / "matched_manifest_v7.json"

SEED = 1729
SELECTED_LAYER = 16
FROZEN_WRITTEN_THRESHOLD = 2.482430934906006
FROZEN_THRESHOLD_BALANCED_ACCURACY = 0.97
FROZEN_THRESHOLD_OWN_RECALL = 0.98
FROZEN_THRESHOLD_FOIL_SPECIFICITY = 0.96
N_FOLDS = 5
MIN_CANDIDATES = 90
TARGET_VERIFIED_PAIRS = 50
KL_THRESHOLD = 1e-3
POSITION_RULE = "explicit_concept_token_in_shared_context"
METRIC_DEFINITION = "logit(answer_A) - logit(answer_B)"

FROZEN_PROTOCOL_SHA256 = (
    "0b1858e3b5c52181b0dd57551dc7a9917ad16c0b505b2fbb3e954551546d51a0"
)
FROZEN_DIRECTION_ARTIFACT_SHA256 = (
    "7a00eac9247d2c7160ae0c9ac49e6043201b4e86b493c1cb934b37a27a1e0b12"
)

CATEGORY_CONTRACTS: dict[str, dict[str, str]] = {
    "atomic-number-element-symbol": {
        "answer_type": "chemical element symbol",
        "copy_object": "chemical symbol",
        "relation": "element-to-chemical-symbol",
    },
    "city-country-capital": {
        "answer_type": "national capital city",
        "copy_object": "capital",
        "relation": "country-to-national-capital",
    },
    "us-city-state-capital": {
        "answer_type": "US state capital city",
        "copy_object": "capital",
        "relation": "US-state-to-state-capital",
    },
}

ProgressFn = Callable[[str, Mapping[str, Any]], None]


def _emit(progress: ProgressFn | None, event: str, payload: Mapping[str, Any]) -> None:
    """Emit a compact progress record."""

    if progress is not None:
        progress(event, dict(payload))


def sha256_file(path: str | Path) -> str:
    """Return the byte SHA-256 of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, value: Any) -> Path:
    """Persist a finite, deterministic JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.write_text(serialized, encoding="utf-8")
    return destination


def direction_digest(
    directions: Mapping[int, torch.Tensor],
    *,
    layer: int = SELECTED_LAYER,
) -> str:
    """Hash ordered fp32 direction bytes and their token/layer identities."""

    digest = hashlib.sha256()
    digest.update(f"v7-jlens-directions-layer-{int(layer)}\n".encode())
    for token_id in sorted(int(value) for value in directions):
        vector = directions[token_id].detach().float().contiguous().cpu()
        if vector.ndim != 1 or not torch.isfinite(vector).all():
            raise ValueError(f"Direction {token_id} is not one finite vector")
        if not torch.allclose(vector.norm(), torch.tensor(1.0), atol=1e-4, rtol=1e-4):
            raise ValueError(f"Direction {token_id} is not unit norm")
        digest.update(f"{token_id}:{vector.numel()}:".encode())
        digest.update(vector.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_direction_bank_v7(
    bundle: Any,
    token_ids: Sequence[int],
    *,
    layer: int = SELECTED_LAYER,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Recreate the frozen raw J-Lens directions from the pinned published lens."""

    published_lens = load_published_lens(bundle.model_id, local_files_only=True)
    bank = jlens_direction_bank(
        published_lens,
        bundle.lens_model,
        token_ids,
        [int(layer)],
        fold_rms_gain=False,
        compute_device="cuda",
        output_device="cpu",
    )
    directions = {int(token_id): layers[int(layer)] for token_id, layers in bank.items()}
    provenance = {
        "convention": "normalize(J_layer.T @ W_U[token]); fold_rms_gain=False",
        "layer": int(layer),
        "n_directions": len(directions),
        "token_ids": sorted(directions),
        "ordered_tensor_digest_sha256": direction_digest(directions, layer=layer),
        "frozen_local_artifact_reference": {
            "path": "artifacts/final/01_directions.pt",
            "sha256": FROZEN_DIRECTION_ARTIFACT_SHA256,
            "required_at_runtime": False,
        },
    }
    del published_lens
    return directions, provenance


def _answer_contract(category: str) -> Mapping[str, str]:
    try:
        return CATEGORY_CONTRACTS[category]
    except KeyError as error:
        raise ValueError(f"Unsupported v7 category {category!r}") from error


def _copy_dashboard_prompt(
    shared_context: str,
    concept: str,
    answer_surface: str,
    *,
    copy_object: str,
) -> str:
    """Turn one answer-requiring prompt into its answer-stated copy counterpart."""

    if not shared_context or not shared_context.endswith("."):
        raise ValueError(f"Shared context must be a complete fact: {shared_context!r}")
    if not concept.strip():
        raise ValueError("Dashboard concept must be nonempty")
    if not answer_surface.startswith(" "):
        raise ValueError("Contextual answer surface must start with one word-boundary space")
    return (
        f"{shared_context} The {copy_object} of {concept} is{answer_surface}. "
        f"Copy the {copy_object} exactly:"
    )


def _token_ids(tokenizer: Any, prompt: str) -> list[int]:
    values = tokenizer.encode(prompt, add_special_tokens=False)
    return [int(value) for value in values]


def build_matched_candidates_v7(tokenizer: Any) -> dict[str, Any]:
    """Build the preregistered matched task from the tracked reciprocal roster."""

    source = build_symmetric_causal_candidates(
        seed=SEED,
        calibration_min_pairs=24,
        n_folds=N_FOLDS,
    )
    if int(source["n_candidates"]) < MIN_CANDIDATES:
        raise RuntimeError(
            f"V7 requires at least {MIN_CANDIDATES} candidates; got {source['n_candidates']}"
        )

    rows: list[dict[str, Any]] = []
    for pair in source["pairs"]:
        category = str(pair["category"])
        contract = _answer_contract(category)
        concept_a_id, concept_a_surface = concept_token_id(
            tokenizer, str(pair["concept_a"])
        )
        concept_b_id, concept_b_surface = concept_token_id(
            tokenizer, str(pair["concept_b"])
        )
        answer_a_id, answer_a_surface = continuation_token_id(
            tokenizer, str(pair["engine_prompt_a"]), str(pair["answer_a"])
        )
        answer_b_id, answer_b_surface = continuation_token_id(
            tokenizer, str(pair["engine_prompt_b"]), str(pair["answer_b"])
        )
        if concept_a_id == concept_b_id or answer_a_id == answer_b_id:
            raise ValueError(f"Collapsed concept/answer tokens for {pair['pair_id']}")

        dashboard_a = _copy_dashboard_prompt(
            str(pair["context_a"]),
            str(pair["concept_a"]),
            answer_a_surface,
            copy_object=contract["copy_object"],
        )
        dashboard_b = _copy_dashboard_prompt(
            str(pair["context_b"]),
            str(pair["concept_b"]),
            answer_b_surface,
            copy_object=contract["copy_object"],
        )
        dashboard_a_id, dashboard_a_surface = continuation_token_id(
            tokenizer, dashboard_a, str(pair["answer_a"])
        )
        dashboard_b_id, dashboard_b_surface = continuation_token_id(
            tokenizer, dashboard_b, str(pair["answer_b"])
        )
        if dashboard_a_id != answer_a_id or dashboard_b_id != answer_b_id:
            raise ValueError(f"Metric tokenization changed across conditions: {pair['pair_id']}")

        concept_prefix_a = _token_ids(tokenizer, str(pair["concept_prefix_a"]))
        concept_prefix_b = _token_ids(tokenizer, str(pair["concept_prefix_b"]))
        engine_ids_a = _token_ids(tokenizer, str(pair["engine_prompt_a"]))
        engine_ids_b = _token_ids(tokenizer, str(pair["engine_prompt_b"]))
        dashboard_ids_a = _token_ids(tokenizer, dashboard_a)
        dashboard_ids_b = _token_ids(tokenizer, dashboard_b)
        if not concept_prefix_a or concept_prefix_a[-1] != concept_a_id:
            raise ValueError(f"Concept A is not one explicit final prefix token: {pair['pair_id']}")
        if not concept_prefix_b or concept_prefix_b[-1] != concept_b_id:
            raise ValueError(f"Concept B is not one explicit final prefix token: {pair['pair_id']}")
        position_a = len(concept_prefix_a) - 1
        position_b = len(concept_prefix_b) - 1
        for label, ids, prefix, position, concept_id in (
            ("engine A", engine_ids_a, concept_prefix_a, position_a, concept_a_id),
            ("engine B", engine_ids_b, concept_prefix_b, position_b, concept_b_id),
            ("dashboard A", dashboard_ids_a, concept_prefix_a, position_a, concept_a_id),
            ("dashboard B", dashboard_ids_b, concept_prefix_b, position_b, concept_b_id),
        ):
            if ids[: len(prefix)] != prefix or ids[position] != concept_id:
                raise ValueError(
                    f"{label} changed the frozen explicit-concept prefix for {pair['pair_id']}"
                )

        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "dependency_group": str(pair["dependency_group"]),
                "context_slot": int(pair["context_slot"]),
                "split": str(pair["split"]),
                "fold": None if pair["fold"] is None else int(pair["fold"]),
                "category": category,
                "relation": contract["relation"],
                "answer_type": contract["answer_type"],
                "concept_a": str(pair["concept_a"]),
                "concept_b": str(pair["concept_b"]),
                "concept_a_token_id": concept_a_id,
                "concept_b_token_id": concept_b_id,
                "concept_a_surface": concept_a_surface,
                "concept_b_surface": concept_b_surface,
                "answer_a": str(pair["answer_a"]),
                "answer_b": str(pair["answer_b"]),
                "answer_a_token_id": answer_a_id,
                "answer_b_token_id": answer_b_id,
                "answer_a_surface": answer_a_surface,
                "answer_b_surface": answer_b_surface,
                "engine_prompt_a": str(pair["engine_prompt_a"]),
                "engine_prompt_b": str(pair["engine_prompt_b"]),
                "dashboard_prompt_a": dashboard_a,
                "dashboard_prompt_b": dashboard_b,
                "dashboard_answer_a_surface": dashboard_a_surface,
                "dashboard_answer_b_surface": dashboard_b_surface,
                "context_a": str(pair["context_a"]),
                "context_b": str(pair["context_b"]),
                "concept_prefix_a": str(pair["concept_prefix_a"]),
                "concept_prefix_b": str(pair["concept_prefix_b"]),
                "engine_position_a": position_a,
                "engine_position_b": position_b,
                "dashboard_position_a": position_a,
                "dashboard_position_b": position_b,
                "metric_positive_token_id": answer_a_id,
                "metric_negative_token_id": answer_b_id,
                "engine_metric": METRIC_DEFINITION,
                "dashboard_metric": METRIC_DEFINITION,
                "same_metric_token_ids": True,
                "same_answer_type": True,
                "engine_answer_stated": False,
                "dashboard_answer_stated_and_copyable": True,
                "dashboard_copy_fact_self_contained": True,
                "dashboard_repeats_concept_downstream": True,
                "shared_prefix_token_identical_through_concept": True,
                "source_row_a": str(pair["source_row_a"]),
                "source_row_b": str(pair["source_row_b"]),
            }
        )

    if len(rows) != int(source["n_candidates"]):
        raise AssertionError("V7 candidate construction dropped rows")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("V7 pair IDs are not unique")
    return {
        "rows": rows,
        "candidate_count": len(rows),
        "dependency_group_count": len({row["dependency_group"] for row in rows}),
        "calibration_count": sum(row["split"] == "calibration" for row in rows),
        "evaluation_count": sum(row["split"] == "evaluation" for row in rows),
        "source": {
            "path": "data/specs/twohop_supplement.json",
            "schema_version": source["source_schema_version"],
            "seed": SEED,
            "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "calibration_groups": source["calibration_groups"],
            "fold_by_group": source["fold_by_group"],
        },
    }


def _hidden_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported decoder output type {type(output).__name__}")


@torch.no_grad()
def capture_prompt_positions(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    tokenizer: Any,
    prompts: Sequence[str],
    positions: Sequence[int],
    *,
    layer: int = SELECTED_LAYER,
    batch_size: int = 16,
) -> torch.Tensor:
    """Capture clean post-block residuals at explicit per-prompt positions."""

    if len(prompts) != len(positions) or not prompts:
        raise ValueError("Prompts and positions must align and be nonempty")
    if hf_model.training:
        raise ValueError("WRITTEN capture requires eval mode")
    if batch_size != 1:
        raise ValueError("V7 WRITTEN verification is frozen to single-prompt forwards")
    if not 0 <= int(layer) < len(blocks):
        raise ValueError("WRITTEN layer is outside the model")
    device = next(hf_model.parameters()).device
    chunks: list[torch.Tensor] = []
    for prompt, raw_position in zip(prompts, positions, strict=True):
        position = int(raw_position)
        input_ids = tokenizer.encode(
            str(prompt), add_special_tokens=False, return_tensors="pt"
        ).to(device)
        if not 0 <= position < int(input_ids.shape[1]):
            raise IndexError(
                f"Position {position} outside prompt length {input_ids.shape[1]}: "
                f"{prompt!r}"
            )
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            captured["hidden"] = _hidden_from_output(output).detach()
            return output

        handle = blocks[int(layer)].register_forward_hook(hook)
        try:
            hf_model(
                input_ids=input_ids,
                use_cache=False,
            )
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("Selected WRITTEN activation was not captured")
        chunks.append(hidden[0, position].detach().float().cpu().unsqueeze(0))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def single_prompt_next_token_records_v7(
    hf_model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    expected_token_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank targets with the exact no-mask clean forward used by causal v7."""

    if len(prompts) != len(expected_token_ids) or not prompts:
        raise ValueError("Prompts and expected token IDs must align")
    if hf_model.training:
        raise ValueError("Clean target verification requires eval mode")
    device = next(hf_model.parameters()).device
    records: list[dict[str, Any]] = []
    for index, (prompt, raw_expected_id) in enumerate(
        zip(prompts, expected_token_ids, strict=True)
    ):
        expected_id = int(raw_expected_id)
        input_ids = tokenizer.encode(
            str(prompt), add_special_tokens=False, return_tensors="pt"
        ).to(device)
        logits = hf_model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
        rank = int((logits > logits[expected_id]).sum().cpu().item() + 1)
        top_token_id = int(logits.argmax().cpu())
        records.append(
            {
                "index": index,
                "prompt": str(prompt),
                "n_tokens": int(input_ids.shape[1]),
                "expected_token_id": expected_id,
                "expected_token": tokenizer.decode([expected_id]),
                "expected_logit": float(logits[expected_id].cpu()),
                "rank": rank,
                "top_token_id": top_token_id,
                "top1_correct": int(top_token_id == expected_id),
                "top5_correct": int(rank <= 5),
                "top_tokens": decode_topk(tokenizer, logits, top_k),
                "forward_contract": (
                    "single prompt; add_special_tokens=False; no attention mask"
                ),
            }
        )
    return records


@torch.no_grad()
def single_prompt_state_and_token_records_v7(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    tokenizer: Any,
    prompts: Sequence[str],
    positions: Sequence[int],
    expected_token_ids: Sequence[int],
    *,
    layer: int = SELECTED_LAYER,
    top_k: int = 5,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Capture state and rank from the same hooked forward used by causal v7."""

    if not (
        len(prompts) == len(positions) == len(expected_token_ids)
    ) or not prompts:
        raise ValueError("Prompts, positions, and expected IDs must align")
    if hf_model.training or not 0 <= int(layer) < len(blocks):
        raise ValueError("Exact clean verification requires eval mode and a valid layer")
    device = next(hf_model.parameters()).device
    states: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for index, (prompt, raw_position, raw_expected_id) in enumerate(
        zip(prompts, positions, expected_token_ids, strict=True)
    ):
        position = int(raw_position)
        expected_id = int(raw_expected_id)
        input_ids = tokenizer.encode(
            str(prompt), add_special_tokens=False, return_tensors="pt"
        ).to(device)
        if not 0 <= position < int(input_ids.shape[1]):
            raise IndexError(f"Invalid explicit-concept position for {prompt!r}")
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            captured["hidden"] = _hidden_from_output(output).detach()
            return output

        handle = blocks[int(layer)].register_forward_hook(hook)
        try:
            logits = hf_model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("Exact clean verification did not capture its state")
        states.append(hidden[0, position].detach().float().cpu())
        rank = int((logits > logits[expected_id]).sum().cpu().item() + 1)
        top_token_id = int(logits.argmax().cpu())
        records.append(
            {
                "index": index,
                "prompt": str(prompt),
                "n_tokens": int(input_ids.shape[1]),
                "expected_token_id": expected_id,
                "expected_token": tokenizer.decode([expected_id]),
                "expected_logit": float(logits[expected_id].cpu()),
                "rank": rank,
                "top_token_id": top_token_id,
                "top1_correct": int(top_token_id == expected_id),
                "top5_correct": int(rank <= 5),
                "top_tokens": decode_topk(tokenizer, logits, top_k),
                "forward_contract": (
                    "single prompt; add_special_tokens=False; no attention mask; "
                    "selected-layer capture hook; same state/logit forward"
                ),
            }
        )
    return torch.stack(states), records


def _record_by_pair_side(
    records: Sequence[Mapping[str, Any]],
    *,
    n_pairs: int,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if len(records) != 2 * n_pairs:
        raise ValueError("Flattened clean records do not align with candidate pairs")
    return [(records[2 * i], records[2 * i + 1]) for i in range(n_pairs)]


@torch.no_grad()
def verify_candidates_v7(
    bundle: Any,
    candidates: Sequence[Mapping[str, Any]],
    directions: Mapping[int, torch.Tensor],
    *,
    layer: int = SELECTED_LAYER,
    written_threshold: float = FROZEN_WRITTEN_THRESHOLD,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Apply four top-1 checks and four independent WRITTEN checks per pair."""

    if not candidates:
        raise ValueError("V7 verification requires candidates")
    if bundle.hf_model.training:
        raise ValueError("V7 verification requires eval mode")

    engine_prompts: list[str] = []
    dashboard_prompts: list[str] = []
    expected_ids: list[int] = []
    engine_positions: list[int] = []
    dashboard_positions: list[int] = []
    concept_ids: list[int] = []
    for row in candidates:
        engine_prompts.extend([str(row["engine_prompt_a"]), str(row["engine_prompt_b"])])
        dashboard_prompts.extend(
            [str(row["dashboard_prompt_a"]), str(row["dashboard_prompt_b"])]
        )
        expected_ids.extend(
            [int(row["answer_a_token_id"]), int(row["answer_b_token_id"])]
        )
        engine_positions.extend(
            [int(row["engine_position_a"]), int(row["engine_position_b"])]
        )
        dashboard_positions.extend(
            [int(row["dashboard_position_a"]), int(row["dashboard_position_b"])]
        )
        concept_ids.extend(
            [int(row["concept_a_token_id"]), int(row["concept_b_token_id"])]
        )

    if batch_size != 1:
        raise ValueError("V7 verification is frozen to batch_size=1")
    engine_states, engine_records = single_prompt_state_and_token_records_v7(
        bundle.hf_model,
        bundle.lens_model.layers,
        bundle.tokenizer,
        engine_prompts,
        engine_positions,
        expected_ids,
        layer=layer,
        top_k=5,
    )
    dashboard_states, dashboard_records = single_prompt_state_and_token_records_v7(
        bundle.hf_model,
        bundle.lens_model.layers,
        bundle.tokenizer,
        dashboard_prompts,
        dashboard_positions,
        expected_ids,
        layer=layer,
        top_k=5,
    )
    direction_matrix = torch.stack(
        [directions[int(token_id)].detach().float().cpu() for token_id in concept_ids]
    )
    engine_z = torch.einsum("bd,bd->b", engine_states, direction_matrix)
    dashboard_z = torch.einsum("bd,bd->b", dashboard_states, direction_matrix)
    if not torch.isfinite(engine_z).all() or not torch.isfinite(dashboard_z).all():
        raise ValueError("WRITTEN verification produced non-finite scores")

    paired_engine = _record_by_pair_side(engine_records, n_pairs=len(candidates))
    paired_dashboard = _record_by_pair_side(dashboard_records, n_pairs=len(candidates))
    verified: list[dict[str, Any]] = []
    for index, source in enumerate(candidates):
        row = dict(source)
        engine_a, engine_b = paired_engine[index]
        dashboard_a, dashboard_b = paired_dashboard[index]
        values = {
            "engine_top1_a": bool(engine_a["top1_correct"]),
            "engine_top1_b": bool(engine_b["top1_correct"]),
            "dashboard_top1_a": bool(dashboard_a["top1_correct"]),
            "dashboard_top1_b": bool(dashboard_b["top1_correct"]),
            "engine_written_a": float(engine_z[2 * index]) >= written_threshold,
            "engine_written_b": float(engine_z[2 * index + 1]) >= written_threshold,
            "dashboard_written_a": float(dashboard_z[2 * index]) >= written_threshold,
            "dashboard_written_b": float(dashboard_z[2 * index + 1]) >= written_threshold,
        }
        checks = {
            "ENGINE_A_TARGET_NOT_TOP1": values["engine_top1_a"],
            "ENGINE_B_TARGET_NOT_TOP1": values["engine_top1_b"],
            "DASHBOARD_A_TARGET_NOT_TOP1": values["dashboard_top1_a"],
            "DASHBOARD_B_TARGET_NOT_TOP1": values["dashboard_top1_b"],
            "ENGINE_A_CONCEPT_NOT_WRITTEN": values["engine_written_a"],
            "ENGINE_B_CONCEPT_NOT_WRITTEN": values["engine_written_b"],
            "DASHBOARD_A_CONCEPT_NOT_WRITTEN": values["dashboard_written_a"],
            "DASHBOARD_B_CONCEPT_NOT_WRITTEN": values["dashboard_written_b"],
            "METRIC_TOKEN_IDS_NOT_IDENTICAL": bool(row["same_metric_token_ids"]),
            "ANSWER_TYPE_NOT_IDENTICAL": bool(row["same_answer_type"]),
        }
        reasons = [name for name, passed in checks.items() if not passed]
        gate_pass = not reasons
        if row["split"] == "calibration":
            status = "CALIBRATION_ONLY"
        else:
            status = "VERIFIED" if gate_pass else "UNVERIFIED"
        row.update(values)
        row.update(
            {
                "engine_z_a": float(engine_z[2 * index]),
                "engine_z_b": float(engine_z[2 * index + 1]),
                "dashboard_z_a": float(dashboard_z[2 * index]),
                "dashboard_z_b": float(dashboard_z[2 * index + 1]),
                "written_threshold": float(written_threshold),
                "verification_gate_pass": gate_pass,
                "verification_status": status,
                "verification_reasons": reasons,
                "engine_target_rank_a": int(engine_a["rank"]),
                "engine_target_rank_b": int(engine_b["rank"]),
                "dashboard_target_rank_a": int(dashboard_a["rank"]),
                "dashboard_target_rank_b": int(dashboard_b["rank"]),
                "engine_top_token_id_a": int(engine_a["top_token_id"]),
                "engine_top_token_id_b": int(engine_b["top_token_id"]),
                "dashboard_top_token_id_a": int(dashboard_a["top_token_id"]),
                "dashboard_top_token_id_b": int(dashboard_b["top_token_id"]),
                "four_written_checks_measured_independently": True,
            }
        )
        verified.append(row)
    return verified


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "candidates": len(rows),
        "calibration_pairs": sum(row["split"] == "calibration" for row in rows),
        "calibration_gate_pass": sum(
            row["split"] == "calibration" and bool(row["verification_gate_pass"])
            for row in rows
        ),
        "evaluation_pairs": sum(row["split"] == "evaluation" for row in rows),
        "verified_pairs": sum(row["verification_status"] == "VERIFIED" for row in rows),
        "unverified_pairs": sum(
            row["verification_status"] == "UNVERIFIED" for row in rows
        ),
    }


def build_sanitized_manifest_v7(
    rows: Sequence[Mapping[str, Any]],
    *,
    direction_provenance: Mapping[str, Any],
    logit_agreement: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Create and recursively audit the only artifact cheap READ may consume."""

    manifest = {
        "schema_version": "matched-read-v7-sanitized-manifest-v1",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "torch.bfloat16",
        },
        "selection": {
            "layer": SELECTED_LAYER,
            "position_rule": POSITION_RULE,
            "written_threshold": FROZEN_WRITTEN_THRESHOLD,
            "threshold_frozen_from_calibration_split": True,
            "threshold_balanced_accuracy": FROZEN_THRESHOLD_BALANCED_ACCURACY,
            "threshold_own_recall": FROZEN_THRESHOLD_OWN_RECALL,
            "threshold_foil_specificity": FROZEN_THRESHOLD_FOIL_SPECIFICITY,
        },
        "metric_contract": {
            "engine": METRIC_DEFINITION,
            "dashboard": METRIC_DEFINITION,
            "identical_token_ids_required": True,
            "same_answer_and_answer_type": True,
        },
        "prompt_contract": {
            "engine": "answer absent; concept-to-answer relation must produce it",
            "dashboard": (
                "self-contained downstream concept-answer fact states the answer, "
                "which must only be copied"
            ),
            "shared_prefix_through_explicit_concept_token": True,
            "dashboard_repeats_concept_after_measured_site": True,
            "arithmetic_tasks_present": False,
            "verification_forward": (
                "single prompt; add_special_tokens=False; no attention mask; "
                "selected-layer capture hook; state and logits from the same forward; "
                "matches frozen causal clean forward"
            ),
            "top1_rule": "expected token must equal argmax; tied-max rank is insufficient",
        },
        "source": dict(source),
        "direction_provenance": dict(direction_provenance),
        "logit_agreement": {
            "status": logit_agreement["status"],
            "threshold": float(logit_agreement["threshold"]),
            "n": int(logit_agreement["n"]),
            "max_mean_kl": float(logit_agreement["max_mean_kl"]),
        },
        "counts": _counts(rows),
        "causal_interchange_outputs_included": False,
        "edited_metrics_included": False,
        "rows": [dict(row) for row in rows],
    }
    audited = validate_sanitized_manifest(manifest)
    for row in audited["rows"]:
        if row["engine_metric"] != row["dashboard_metric"]:
            raise ValueError(f"Metric text differs for {row['pair_id']}")
        if int(row["metric_positive_token_id"]) != int(row["answer_a_token_id"]):
            raise ValueError(f"Positive metric token drifted for {row['pair_id']}")
        if int(row["metric_negative_token_id"]) != int(row["answer_b_token_id"]):
            raise ValueError(f"Negative metric token drifted for {row['pair_id']}")
        if "2 + 2" in row["engine_prompt_a"] or "2 + 2" in row["dashboard_prompt_a"]:
            raise ValueError("Arithmetic prompt leaked into v7")
    return audited


def run_dataset_stage_v7(
    *,
    output_dir: str | Path = RESULTS_DIR,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Execute v7_1 and persist the full clean artifact plus sanitized manifest."""

    set_seed(SEED)
    output = Path(output_dir)
    raw_dir = output / "raw"
    dataset_path = raw_dir / DATASET_PATH.name
    manifest_path = output / MANIFEST_PATH.name
    _emit(progress, "load_model", {"model": MODEL_ID, "revision": MODEL_REVISION})
    bundle = load_model(local_files_only=True)
    try:
        dtype = str(next(bundle.hf_model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Pinned model did not load in bf16: {dtype}")
        _emit(progress, "build_candidates", {"minimum": MIN_CANDIDATES})
        candidate_artifact = build_matched_candidates_v7(bundle.tokenizer)
        candidates = candidate_artifact["rows"]

        _emit(progress, "hf_jlens_kl", {"n_prompts": len(G1_PROMPTS)})
        if len(G1_PROMPTS) != 20:
            raise RuntimeError("The frozen HF/J-Lens gate must contain exactly 20 prompts")
        kl_rows = hf_wrapper_logit_kl(bundle, G1_PROMPTS)
        logit_agreement = enforce_kl_gate(kl_rows, threshold=KL_THRESHOLD)
        if not float(logit_agreement["max_mean_kl"]) < KL_THRESHOLD:
            raise RuntimeError("HF/J-Lens KL must be strictly below 1e-3")

        concept_token_ids = sorted(
            {
                int(row[key])
                for row in candidates
                for key in ("concept_a_token_id", "concept_b_token_id")
            }
        )
        _emit(
            progress,
            "directions",
            {"layer": SELECTED_LAYER, "n_tokens": len(concept_token_ids)},
        )
        directions, direction_provenance = build_direction_bank_v7(
            bundle, concept_token_ids, layer=SELECTED_LAYER
        )
        _emit(progress, "verify", {"candidate_pairs": len(candidates)})
        rows = verify_candidates_v7(
            bundle,
            candidates,
            directions,
            layer=SELECTED_LAYER,
            written_threshold=FROZEN_WRITTEN_THRESHOLD,
        )
        counts = _counts(rows)
        manifest = build_sanitized_manifest_v7(
            rows,
            direction_provenance=direction_provenance,
            logit_agreement=logit_agreement,
            source=candidate_artifact["source"],
        )
        manifest_file = save_json(manifest_path, manifest)
        manifest_sha256 = sha256_file(manifest_file)
        artifact = {
            "schema_version": "matched-read-v7-dataset-v1",
            "model": manifest["model"],
            "selection": manifest["selection"],
            "metric_contract": manifest["metric_contract"],
            "prompt_contract": manifest["prompt_contract"],
            "source": manifest["source"],
            "direction_provenance": direction_provenance,
            "logit_agreement": logit_agreement,
            "counts": counts,
            "target_verified_pairs": TARGET_VERIFIED_PAIRS,
            "target_verified_pairs_met": counts["verified_pairs"] >= TARGET_VERIFIED_PAIRS,
            "sanitized_manifest": {
                "path": str(manifest_file.relative_to(PROJECT_ROOT)),
                "sha256": manifest_sha256,
                "causal_interchange_outputs_included": False,
                "edited_metrics_included": False,
            },
            "rows": rows,
        }
        dataset_file = save_json(dataset_path, artifact)
        if counts["verified_pairs"] < TARGET_VERIFIED_PAIRS:
            raise RuntimeError(
                f"Only {counts['verified_pairs']} v7 evaluation pairs verified; "
                f"target is {TARGET_VERIFIED_PAIRS}. Artifact preserved at {dataset_file}."
            )
        _emit(
            progress,
            "complete",
            {
                **counts,
                "manifest_sha256": manifest_sha256,
                "dataset_path": str(dataset_file),
            },
        )
        return {
            "dataset_path": str(dataset_file),
            "manifest_path": str(manifest_file),
            "manifest_sha256": manifest_sha256,
            "counts": counts,
            "logit_agreement": {
                "n": int(logit_agreement["n"]),
                "threshold": float(logit_agreement["threshold"]),
                "max_mean_kl": float(logit_agreement["max_mean_kl"]),
            },
            "direction_digest_sha256": direction_provenance[
                "ordered_tensor_digest_sha256"
            ],
            "target_verified_pairs_met": True,
        }
    finally:
        release_model(bundle)


__all__ = [
    "DATASET_PATH",
    "FROZEN_WRITTEN_THRESHOLD",
    "MANIFEST_PATH",
    "METRIC_DEFINITION",
    "POSITION_RULE",
    "RESULTS_DIR",
    "SEED",
    "SELECTED_LAYER",
    "TARGET_VERIFIED_PAIRS",
    "build_direction_bank_v7",
    "build_matched_candidates_v7",
    "build_sanitized_manifest_v7",
    "capture_prompt_positions",
    "direction_digest",
    "run_dataset_stage_v7",
    "save_json",
    "sha256_file",
    "verify_candidates_v7",
]

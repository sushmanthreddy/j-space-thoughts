"""Reusable two-hop WRITE/READ/CAUSAL measurement and analysis pipeline.

The preregistered primary direction in this module is the raw Jacobian-lens
``normalize(W_U[c] @ J_l)`` direction.  RMS-gain-folded directions are not
constructed here, so they cannot silently replace the primary convention.
Mean-difference (MD) directions are loaded from the independently fitted
notebook-01 artifact and are measured only when *both* members of an item's
concept/foil pair are present.

Every causal sign is persisted twice: ``delta = M_edited - M_clean`` and
``positive_damage = M_clean - M_edited``.  The latter is the outcome used by
the headline analysis because a positive value means that removing the clean
concept support hurt the correct-vs-foil behavior metric.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data_gen import (
    DEFAULT_JLENS_ROOT,
    DEFAULT_TWOHOP_SUPPLEMENT,
    load_combined_twohop_collection,
    tokenize_twohop_item,
)
from src.interventions import (
    ablation_edits,
    clamped_swap_edits,
    forward_logits,
    suppress_output_token,
)
from src.jlens_iface import (
    jlens_direction_bank,
    load_local_lens,
    load_published_lens,
    workspace_layers,
)
from src.metrics import (
    logit_difference,
    partial_correlation_with_ci,
    pearson_with_ci,
    save_json,
    signed_causal_delta,
    standardized_regression_with_ci,
    support_damage,
)
from src.model_utils import capture_residuals, load_model, release_model, set_seed
from src.read_scores import attribution_read


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD_ARTIFACT = ROOT / "data/directions/qwen2.5-7b_concept_vectors.pt"
PRIMARY_DIRECTION_METHOD = "jlens_raw_wu_j"
MD_DIRECTION_METHOD = "mean_difference"
SCHEMA_VERSION = "twohop-phase-v1"
SEED = 1729


def _canonical_concept(value: str) -> str:
    """Canonicalize a concept label without changing its human-readable form."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Concept labels must be nonempty strings")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _json_ready(value: Any) -> Any:
    """Convert nested scientific values to strict, finite JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_ready(value.detach().cpu().item())
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_tokenized_twohop_collection(
    tokenizer: Any,
    *,
    jlens_root: str | Path = DEFAULT_JLENS_ROOT,
    supplement_path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
) -> dict[str, Any]:
    """Load the audited combined set and retain every tokenization rejection."""

    collection = load_combined_twohop_collection(jlens_root, supplement_path)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in collection["items"]:
        try:
            accepted.append(tokenize_twohop_item(tokenizer, item))
        except (ValueError, IndexError) as error:
            rejected.append(
                {
                    "name": item.get("name"),
                    "source": item.get("source"),
                    "reason": str(error),
                }
            )
    return {
        "schema_version": collection["schema_version"],
        "seed": collection["seed"],
        "source": collection["source"],
        "provenance": collection["provenance"],
        "n_combined": len(collection["items"]),
        "n_tokenizable": len(accepted),
        "n_tokenization_rejected": len(rejected),
        "items": accepted,
        "tokenization_rejections": rejected,
    }


def clean_eligibility_from_logits(
    item: Mapping[str, Any],
    logits: torch.Tensor,
    *,
    tokenizer: Any | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Apply the frozen clean filter to one next-token logit vector.

    Eligibility requires (1) the target token is rank one and (2) neither the
    latent concept nor its paired foil concept occurs in the clean top ten.
    The rule is intentionally independent of direction method and is applied
    exactly once before any intervention.
    """

    if logits.ndim != 1:
        raise ValueError(f"Expected one-dimensional logits, got {tuple(logits.shape)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    required = (
        "target_token_id",
        "foil_token_id",
        "concept_token_id",
        "foil_concept_token_id",
    )
    missing = [field for field in required if field not in item]
    if missing:
        raise ValueError(f"Tokenized item missing fields: {missing}")

    values, indices = logits.detach().float().topk(min(top_k, logits.numel()))
    top_ids = [int(token_id) for token_id in indices.cpu()]
    target_id = int(item["target_token_id"])
    foil_id = int(item["foil_token_id"])
    concept_id = int(item["concept_token_id"])
    foil_concept_id = int(item["foil_concept_token_id"])
    target_rank = int((logits > logits[target_id]).sum().item() + 1)

    reasons: list[str] = []
    if target_rank != 1:
        reasons.append("target_not_top1")
    if concept_id in top_ids:
        reasons.append("concept_in_clean_top10")
    if foil_concept_id in top_ids:
        reasons.append("foil_concept_in_clean_top10")

    top_tokens: list[dict[str, Any]] = []
    for value, token_id in zip(values.cpu(), top_ids, strict=True):
        top_tokens.append(
            {
                "token_id": token_id,
                "token": (
                    tokenizer.decode([token_id]) if tokenizer is not None else None
                ),
                "logit": float(value),
            }
        )
    return {
        "name": item.get("name"),
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "rule": "target rank == 1 and concept+foil concept absent from clean top10",
        "target_rank": target_rank,
        "top1_token_id": top_ids[0],
        "top10_token_ids": top_ids,
        "top10": top_tokens,
        "clean_metric": float(logits[target_id] - logits[foil_id]),
        "target_logit": float(logits[target_id]),
        "foil_logit": float(logits[foil_id]),
        "concept_rank": int((logits > logits[concept_id]).sum().item() + 1),
        "foil_concept_rank": int((logits > logits[foil_concept_id]).sum().item() + 1),
    }


@torch.no_grad()
def screen_clean_eligibility(
    hf_model: torch.nn.Module,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    max_length: int = 128,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Run the frozen clean eligibility screen in deterministic padded batches."""

    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive")
    if not items:
        return []
    device = next(hf_model.parameters()).device
    records: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        batch = list(items[start : start + batch_size])
        prompts = [str(item["prompt"]) for item in batch]
        lengths = [
            len(tokenizer.encode(prompt, add_special_tokens=True)) for prompt in prompts
        ]
        too_long = [length for length in lengths if length > max_length]
        if too_long:
            raise ValueError(
                "Refusing to screen truncated two-hop prompts: "
                f"max observed length={max(too_long)}, max_length={max_length}"
            )
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        output = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        for batch_index, item in enumerate(batch):
            real_positions = (
                attention_mask[batch_index].nonzero(as_tuple=False).flatten()
            )
            position = int(real_positions[-1])
            record = clean_eligibility_from_logits(
                item,
                output[batch_index, position],
                tokenizer=tokenizer,
                top_k=top_k,
            )
            record.update(
                {
                    "screen_index": start + batch_index,
                    "n_prompt_tokens": int(len(real_positions)),
                    "behavior_position": position,
                }
            )
            records.append(record)
        del output
    return records


def load_mean_difference_artifact(
    path: str | Path,
    *,
    expected_layers: Sequence[int] | None = None,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    norm_tolerance: float = 5e-3,
) -> dict[str, Any]:
    """Load and validate notebook-01's ``metadata + mean_difference`` artifact."""

    artifact_path = Path(path)
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:  # pragma: no cover - compatibility with older torch
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    else:
        # Notebook 01 stores the frozen exclusion graph as built-in sets. Keep
        # weights-only loading and allowlist only that required primitive.
        with safe_globals([set]):
            payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("MD artifact must be a mapping")
    if not isinstance(payload.get("metadata"), Mapping):
        raise ValueError("MD artifact is missing mapping field 'metadata'")
    bank = payload.get("mean_difference")
    if not isinstance(bank, Mapping) or not bank:
        raise ValueError("MD artifact is missing nonempty 'mean_difference' bank")

    metadata = dict(payload["metadata"])
    if expected_model_id is not None and metadata.get("model_id") != expected_model_id:
        raise ValueError(
            f"MD artifact model {metadata.get('model_id')!r} != {expected_model_id!r}"
        )
    if (
        expected_model_revision is not None
        and metadata.get("model_revision") != expected_model_revision
    ):
        raise ValueError("MD artifact model revision differs from the loaded model")

    requested_layers = (
        tuple(sorted(set(int(layer) for layer in expected_layers)))
        if expected_layers is not None
        else None
    )
    converted: dict[str, dict[int, torch.Tensor]] = {}
    lookup: dict[str, str] = {}
    norm_errors: list[float] = []
    source_dtypes: set[str] = set()
    dimensions: dict[int, int] = {}
    for raw_concept, raw_layers in bank.items():
        if not isinstance(raw_concept, str) or not isinstance(raw_layers, Mapping):
            raise ValueError("MD concepts must map string labels to layer mappings")
        canonical = _canonical_concept(raw_concept)
        if canonical in lookup:
            raise ValueError(f"Canonical MD concept collision for {raw_concept!r}")
        lookup[canonical] = raw_concept
        layer_bank: dict[int, torch.Tensor] = {}
        for raw_layer, raw_direction in raw_layers.items():
            layer = int(raw_layer)
            if not isinstance(raw_direction, torch.Tensor):
                raise TypeError(
                    f"MD direction {raw_concept!r}, layer {layer} is not a tensor"
                )
            if raw_direction.ndim != 1 or not raw_direction.dtype.is_floating_point:
                raise ValueError(
                    f"MD direction {raw_concept!r}, layer {layer} must be floating [d]"
                )
            direction = raw_direction.detach().float().cpu()
            if not torch.isfinite(direction).all():
                raise ValueError(
                    f"Non-finite MD direction for {raw_concept!r}, layer {layer}"
                )
            norm = float(direction.norm())
            error = abs(norm - 1.0)
            if norm == 0.0 or error > norm_tolerance:
                raise ValueError(
                    f"MD direction norm drift for {raw_concept!r}, layer {layer}: "
                    f"norm={norm:.8g}, tolerance={norm_tolerance}"
                )
            source_dtypes.add(str(raw_direction.dtype))
            dimension = int(direction.numel())
            if layer in dimensions and dimensions[layer] != dimension:
                raise ValueError(f"MD d_model mismatch at layer {layer}")
            dimensions[layer] = dimension
            norm_errors.append(error)
            layer_bank[layer] = F.normalize(direction, dim=0)
        if (
            requested_layers is not None
            and tuple(sorted(layer_bank)) != requested_layers
        ):
            raise ValueError(
                f"MD layers for {raw_concept!r} are {sorted(layer_bank)}, "
                f"expected {list(requested_layers)}"
            )
        converted[raw_concept] = layer_bank

    if requested_layers is None:
        layer_sets = {tuple(sorted(layer_bank)) for layer_bank in converted.values()}
        if len(layer_sets) != 1:
            raise ValueError("Every MD concept must cover identical layers")
        requested_layers = next(iter(layer_sets))
    metadata_layers = metadata.get("workspace_layers")
    if (
        metadata_layers is not None
        and tuple(map(int, metadata_layers)) != requested_layers
    ):
        raise ValueError(
            "MD artifact metadata workspace layers disagree with its tensors"
        )
    return {
        "path": str(artifact_path.resolve()),
        "metadata": metadata,
        "mean_difference": converted,
        "canonical_lookup": lookup,
        "layers": list(requested_layers),
        "d_model_by_layer": dimensions,
        "source_dtypes": sorted(source_dtypes),
        "max_source_unit_norm_error": max(norm_errors, default=0.0),
        "n_concepts": len(converted),
    }


def prepare_direction_assignments(
    lens: Any,
    lens_model: Any,
    eligible_items: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    *,
    md_artifact_path: str | Path,
    model_id: str,
    model_revision: str,
    compute_device: str | torch.device = "cuda",
    output_device: str | torch.device = "cuda",
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Construct raw J-Lens pairs and select complete MD artifact pairs."""

    layer_list = sorted(set(int(layer) for layer in layers))
    item_names = [str(item["name"]) for item in eligible_items]
    if len(item_names) != len(set(item_names)):
        raise ValueError("Eligible two-hop item names must be unique")

    token_ids = {
        int(token_id)
        for item in eligible_items
        for token_id in (item["concept_token_id"], item["foil_concept_token_id"])
    }
    raw_bank = jlens_direction_bank(
        lens,
        lens_model,
        token_ids,
        layer_list,
        fold_rms_gain=False,
        compute_device=compute_device,
        output_device=output_device,
    )
    assignments: dict[str, dict[str, dict[str, Any]]] = {
        PRIMARY_DIRECTION_METHOD: {},
        MD_DIRECTION_METHOD: {},
    }
    for item in eligible_items:
        assignments[PRIMARY_DIRECTION_METHOD][str(item["name"])] = {
            "concept": raw_bank[int(item["concept_token_id"])],
            "foil": raw_bank[int(item["foil_concept_token_id"])],
            "direction_convention": "normalize(W_U[token] @ J_layer)",
            "analysis_role": "preregistered_primary",
        }

    artifact = load_mean_difference_artifact(
        md_artifact_path,
        expected_layers=layer_list,
        expected_model_id=model_id,
        expected_model_revision=model_revision,
    )
    md_bank = artifact["mean_difference"]
    lookup = artifact["canonical_lookup"]
    md_skipped: list[dict[str, Any]] = []
    for item in eligible_items:
        concept_key = lookup.get(_canonical_concept(str(item["intermediate"])))
        foil_key = lookup.get(_canonical_concept(str(item["swap_to"])))
        missing = []
        if concept_key is None:
            missing.append(str(item["intermediate"]))
        if foil_key is None:
            missing.append(str(item["swap_to"]))
        if missing:
            md_skipped.append({"name": item["name"], "missing_concepts": missing})
            continue
        concept_directions = {
            layer: md_bank[concept_key][layer].to(output_device) for layer in layer_list
        }
        foil_directions = {
            layer: md_bank[foil_key][layer].to(output_device) for layer in layer_list
        }
        raw_assignment = assignments[PRIMARY_DIRECTION_METHOD][str(item["name"])]
        assignments[MD_DIRECTION_METHOD][str(item["name"])] = {
            "concept": concept_directions,
            "foil": foil_directions,
            "artifact_concept_key": concept_key,
            "artifact_foil_key": foil_key,
            "direction_convention": (
                "matched-slot one-vs-other residual mean difference; paired foil "
                "excluded from baseline"
            ),
            "analysis_role": "required_non_jlens_robustness",
            "cosine_to_raw_jlens": {
                "concept_by_layer": {
                    layer: float(
                        torch.dot(
                            concept_directions[layer].detach().float().cpu(),
                            raw_assignment["concept"][layer].detach().float().cpu(),
                        )
                    )
                    for layer in layer_list
                },
                "foil_by_layer": {
                    layer: float(
                        torch.dot(
                            foil_directions[layer].detach().float().cpu(),
                            raw_assignment["foil"][layer].detach().float().cpu(),
                        )
                    )
                    for layer in layer_list
                },
            },
        }

    coverage = {
        "jlens_raw_wu_j": {
            "n_items": len(assignments[PRIMARY_DIRECTION_METHOD]),
            "coverage_rule": "all frozen-clean-eligible tokenized items",
            "fold_rms_gain": False,
        },
        "mean_difference": {
            "n_items": len(assignments[MD_DIRECTION_METHOD]),
            "coverage_rule": "both concept and paired foil present in notebook-01 artifact",
            "n_skipped": len(md_skipped),
            "skipped": md_skipped,
            "artifact": {
                key: value
                for key, value in artifact.items()
                if key not in {"mean_difference", "canonical_lookup"}
            },
        },
    }
    return assignments, coverage


def _array_summary(write: np.ndarray, read: np.ndarray) -> dict[str, Any]:
    products = write * read
    denominator = float(np.abs(write).sum())
    return {
        "n_coordinates": int(write.size),
        "write_signed_sum": float(write.sum()),
        "write_signed_mean": float(write.mean()),
        "write_abs_sum": float(np.abs(write).sum()),
        "write_abs_mean": float(np.abs(write).mean()),
        "write_strength": float(np.abs(write).mean()),
        "read_signed_sum": float(read.sum()),
        "read_signed_mean": float(read.mean()),
        "read_abs_sum": float(np.abs(read).sum()),
        "read_abs_mean": float(np.abs(read).mean()),
        "read_strength": float(np.abs(read).mean()),
        "write_read_signed_sum": float(products.sum()),
        "write_read_signed_mean": float(products.mean()),
        "write_read_abs_sum": float(np.abs(products).sum()),
        "first_order_predicted_delta": float(-products.sum()),
        "first_order_predicted_positive_damage": float(products.sum()),
        "support_oriented_read": (
            float(products.sum() / denominator) if denominator > 0.0 else None
        ),
    }


def aggregate_write_read(
    write_by_layer: Mapping[int, Sequence[float] | np.ndarray],
    read_by_layer: Mapping[int, Sequence[float] | np.ndarray],
) -> dict[str, Any]:
    """Aggregate aligned layer×position arrays without discarding signed values.

    ``support_oriented_read`` is exactly
    ``sum(WRITE * READ) / sum(abs(WRITE))``.  Multiplying it by the stored
    ``write_abs_sum`` recovers the first-order predicted positive damage.
    """

    if set(write_by_layer) != set(read_by_layer) or not write_by_layer:
        raise ValueError("WRITE and READ must cover identical nonempty layers")
    layers = sorted(int(layer) for layer in write_by_layer)
    write_chunks: list[np.ndarray] = []
    read_chunks: list[np.ndarray] = []
    by_layer: dict[str, Any] = {}
    n_positions: dict[str, int] = {}
    for layer in layers:
        write = np.asarray(write_by_layer[layer], dtype=np.float64).reshape(-1)
        read = np.asarray(read_by_layer[layer], dtype=np.float64).reshape(-1)
        if write.size == 0 or write.shape != read.shape:
            raise ValueError(f"WRITE/READ shape mismatch at layer {layer}")
        if not np.isfinite(write).all() or not np.isfinite(read).all():
            raise ValueError(f"Non-finite WRITE/READ value at layer {layer}")
        write_chunks.append(write)
        read_chunks.append(read)
        by_layer[str(layer)] = _array_summary(write, read)
        n_positions[str(layer)] = int(write.size)
    result = _array_summary(np.concatenate(write_chunks), np.concatenate(read_chunks))
    result.update(
        {
            "n_layers": len(layers),
            "layers": layers,
            "n_positions_by_layer": n_positions,
            "by_layer": by_layer,
            "aggregation_formula": {
                "write_strength": "mean(abs(WRITE)) over all layer-position coordinates",
                "read_strength": "mean(abs(attribution READ)) over all coordinates",
                "support_oriented_read": (
                    "sum(WRITE * attribution READ) / sum(abs(WRITE))"
                ),
                "predicted_delta": "-sum(WRITE * attribution READ)",
            },
            "analysis_roles": {
                "write_strength": "headline WRITE variable",
                "read_strength": (
                    "headline READ variable; independent of the current activation magnitude"
                ),
                "support_oriented_read": (
                    "product-based first-order diagnostic only; never a headline READ variable"
                ),
            },
        }
    )
    return result


def _behavior_metric(logits: torch.Tensor, item: Mapping[str, Any]) -> float:
    return float(
        logit_difference(
            logits,
            int(item["target_token_id"]),
            int(item["foil_token_id"]),
        )[0]
    )


def _effect_record(clean_metric: float, edited_metric: float) -> dict[str, float]:
    return {
        "edited_metric": float(edited_metric),
        "delta": signed_causal_delta(clean_metric, edited_metric),
        "positive_damage": support_damage(clean_metric, edited_metric),
    }


def _output_suppression_record(
    clean_logits: torch.Tensor,
    clean_metric: float,
    item: Mapping[str, Any],
    token_id: int,
    token_role: str,
) -> dict[str, Any]:
    edited = suppress_output_token(clean_logits, token_id)
    edited_metric = _behavior_metric(edited, item)
    return {
        "token_role": token_role,
        "token_id": int(token_id),
        "clean_output_logit": float(clean_logits[0, -1, int(token_id)]),
        "suppressed_output_logit": float(edited[0, -1, int(token_id)]),
        **_effect_record(clean_metric, edited_metric),
    }


def _direction_pair_diagnostics(
    concept: Mapping[int, torch.Tensor],
    foil: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    if set(concept) != set(foil):
        raise ValueError("Concept and foil directions must cover identical layers")
    cosines: dict[str, float] = {}
    conditions: dict[str, float] = {}
    for layer in sorted(concept):
        first = concept[layer].detach().float().cpu()
        second = foil[layer].detach().float().cpu()
        cosine = float(torch.dot(first, second))
        basis = torch.stack([first, second])
        cosines[str(layer)] = cosine
        conditions[str(layer)] = float(torch.linalg.cond(basis @ basis.T))
    return {
        "cosine_by_layer": cosines,
        "gram_condition_by_layer": conditions,
        "max_gram_condition": max(conditions.values()),
    }


def measure_twohop_rows(
    bundle: Any,
    eligible_items: Sequence[Mapping[str, Any]],
    eligibility_records: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Mapping[str, Any]]],
    layers: Sequence[int],
    *,
    max_length: int = 128,
    max_swap_condition: float = 1e4,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Measure raw arrays and real interventions for every assigned item/method."""

    layer_list = sorted(set(int(layer) for layer in layers))
    methods = [PRIMARY_DIRECTION_METHOD, MD_DIRECTION_METHOD]
    unknown_methods = set(assignments) - set(methods)
    if unknown_methods:
        raise ValueError(f"Unknown direction methods: {sorted(unknown_methods)}")
    device = next(bundle.hf_model.parameters()).device
    rows: list[dict[str, Any]] = []
    for item_index, item in enumerate(eligible_items):
        item_name = str(item["name"])
        encoded = bundle.tokenizer(
            str(item["prompt"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        if int(attention_mask.sum()) >= max_length:
            untruncated = len(
                bundle.tokenizer.encode(str(item["prompt"]), add_special_tokens=True)
            )
            if untruncated > max_length:
                raise ValueError(f"Refusing truncated measurement for {item_name!r}")
        clean_logits = forward_logits(
            bundle.hf_model,
            input_ids,
            attention_mask=attention_mask,
        )
        clean_metric = _behavior_metric(clean_logits, item)
        clean_residuals = capture_residuals(
            bundle.lens_model,
            input_ids,
            layer_list,
        )
        prompt_token_ids = [int(value) for value in input_ids[0].detach().cpu()]
        prompt_tokens = [
            bundle.tokenizer.decode([token_id]) for token_id in prompt_token_ids
        ]
        concept_suppression = _output_suppression_record(
            clean_logits,
            clean_metric,
            item,
            int(item["concept_token_id"]),
            "concept",
        )
        foil_suppression = _output_suppression_record(
            clean_logits,
            clean_metric,
            item,
            int(item["foil_concept_token_id"]),
            "foil_concept",
        )

        for method in methods:
            assignment = assignments.get(method, {}).get(item_name)
            if assignment is None:
                continue
            base: dict[str, Any] = {
                "row_index": len(rows),
                "eligible_item_index": item_index,
                "name": item_name,
                "source": item.get("source"),
                "category": item.get("category"),
                "prompt": item["prompt"],
                "intermediate": item["intermediate"],
                "swap_to": item["swap_to"],
                "answer": item["answer"],
                "swap_answer": item["swap_answer"],
                "token_ids": {
                    "concept": int(item["concept_token_id"]),
                    "foil_concept": int(item["foil_concept_token_id"]),
                    "target": int(item["target_token_id"]),
                    "foil": int(item["foil_token_id"]),
                },
                "token_surfaces": {
                    "concept": item["concept_surface"],
                    "foil_concept": item["foil_concept_surface"],
                    "target": item["target_surface"],
                    "foil": item["foil_surface"],
                },
                "prompt_token_ids": prompt_token_ids,
                "prompt_tokens": prompt_tokens,
                "n_prompt_tokens": len(prompt_token_ids),
                "intervention_positions": list(range(len(prompt_token_ids))),
                "workspace_layers": layer_list,
                "direction_method": method,
                "direction_convention": assignment["direction_convention"],
                "analysis_role": assignment["analysis_role"],
                "clean_eligibility": dict(eligibility_records[item_name]),
                "clean_metric": clean_metric,
                "clean_metric_definition": "logit(target) - logit(foil)",
                "output_suppression": {
                    "concept": concept_suppression,
                    "foil_concept": foil_suppression,
                    "note": (
                        "Only the named final-vocabulary logit is clamped; no "
                        "intermediate activation is changed. Recorded on every row."
                    ),
                },
            }
            if "cosine_to_raw_jlens" in assignment:
                base["cosine_to_raw_jlens"] = assignment["cosine_to_raw_jlens"]
            try:
                concept_directions = assignment["concept"]
                foil_directions = assignment["foil"]
                pair_diagnostics = _direction_pair_diagnostics(
                    concept_directions, foil_directions
                )
                if pair_diagnostics["max_gram_condition"] > max_swap_condition:
                    raise ValueError(
                        "Direction pair is too ill-conditioned for an exact swap: "
                        f"{pair_diagnostics['max_gram_condition']:.5g}"
                    )

                attribution = attribution_read(
                    bundle.hf_model,
                    bundle.lens_model.layers,
                    input_ids,
                    concept_directions,
                    target_token_id=int(item["target_token_id"]),
                    foil_token_id=int(item["foil_token_id"]),
                    attention_mask=attention_mask,
                )
                aggregate = aggregate_write_read(
                    attribution.write,
                    attribution.read,
                )
                helper_prediction_error = float(
                    aggregate["first_order_predicted_delta"]
                    - attribution.predicted_delta
                )

                ablated_logits = forward_logits(
                    bundle.hf_model,
                    input_ids,
                    attention_mask=attention_mask,
                    blocks=bundle.lens_model.layers,
                    edits=ablation_edits(concept_directions),
                )
                ablated_metric = _behavior_metric(ablated_logits, item)
                swapped_logits = forward_logits(
                    bundle.hf_model,
                    input_ids,
                    attention_mask=attention_mask,
                    blocks=bundle.lens_model.layers,
                    edits=clamped_swap_edits(
                        clean_residuals,
                        concept_directions,
                        foil_directions,
                        max_condition=max_swap_condition,
                    ),
                )
                swapped_metric = _behavior_metric(swapped_logits, item)
                base.update(
                    {
                        "measurement_status": "OK",
                        "direction_pair": pair_diagnostics,
                        "raw_arrays": {
                            "write_by_layer_position": {
                                str(layer): attribution.write[layer].tolist()
                                for layer in layer_list
                            },
                            "attribution_read_by_layer_position": {
                                str(layer): attribution.read[layer].tolist()
                                for layer in layer_list
                            },
                            "attribution_predicted_delta_by_layer": {
                                str(layer): attribution.predicted_delta_by_layer[layer]
                                for layer in layer_list
                            },
                        },
                        "aggregate": aggregate,
                        "attribution_clean_metric": attribution.metric,
                        "attribution_clean_metric_error": float(
                            attribution.metric - clean_metric
                        ),
                        "attribution_helper_predicted_delta": (
                            attribution.predicted_delta
                        ),
                        "attribution_helper_prediction_sum_error": (
                            helper_prediction_error
                        ),
                        "ablation": {
                            "scope": "all workspace layers and all prompt positions",
                            **_effect_record(clean_metric, ablated_metric),
                        },
                        "clean_clamped_swap": {
                            "scope": "all workspace layers and all prompt positions",
                            "strength": 1.0,
                            **_effect_record(clean_metric, swapped_metric),
                        },
                    }
                )
            except Exception as error:  # retain failed rows instead of cherry-picking
                base.update(
                    {
                        "measurement_status": "ERROR",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                rows.append(base)
                if fail_fast:
                    raise
                continue
            rows.append(base)
    return rows


def _complete_rows(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    outcome: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("direction_method") == method
        and row.get("measurement_status") == "OK"
        and isinstance(row.get(outcome), Mapping)
        and row[outcome].get("positive_damage") is not None
        and row.get("aggregate", {}).get("write_abs_mean") is not None
        and row.get("aggregate", {}).get("read_abs_mean") is not None
    ]


def analysis_vectors(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    *,
    outcome: str = "ablation",
) -> dict[str, Any]:
    """Extract the transparent preregistered analysis variables for one method."""

    selected = _complete_rows(rows, method, outcome)
    return {
        "item_names": [str(row["name"]) for row in selected],
        "write_strength": [
            float(row["aggregate"]["write_abs_mean"]) for row in selected
        ],
        "write_signed_mean": [
            float(row["aggregate"]["write_signed_mean"]) for row in selected
        ],
        "support_oriented_read": [
            (
                float(row["aggregate"]["support_oriented_read"])
                if row["aggregate"]["support_oriented_read"] is not None
                else None
            )
            for row in selected
        ],
        "read_strength": [float(row["aggregate"]["read_abs_mean"]) for row in selected],
        "read_signed_mean": [
            float(row["aggregate"]["read_signed_mean"]) for row in selected
        ],
        "causal_positive_damage": [
            float(row[outcome]["positive_damage"]) for row in selected
        ],
        "causal_delta": [float(row[outcome]["delta"]) for row in selected],
        "predicted_positive_damage": [
            float(row["aggregate"]["first_order_predicted_positive_damage"])
            for row in selected
        ],
        "output_suppression_positive_damage": [
            float(row["output_suppression"]["concept"]["positive_damage"])
            for row in selected
        ],
    }


def _safe_statistic(function: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        result = function(*args, **kwargs)
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            "status": "NOT_ESTIMABLE",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    if any(
        isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
        for value in _walk_values(result)
    ):
        return {
            "status": "NOT_ESTIMABLE",
            "error_type": "NonFiniteStatistic",
            "error": "Statistic returned a non-finite value",
        }
    return {"status": "ESTIMATED", **result}


def _walk_values(value: Any):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def analyze_measurements(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome: str = "ablation",
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run correlations, partial correlations, and standardized regressions."""

    methods = sorted(
        {
            str(row["direction_method"])
            for row in rows
            if row.get("measurement_status") == "OK"
        }
    )
    by_method: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        vectors = analysis_vectors(rows, method, outcome=outcome)
        write = vectors["write_strength"]
        read = vectors["read_strength"]
        causal = vectors["causal_positive_damage"]
        predicted = vectors["predicted_positive_damage"]
        signed_write = vectors["write_signed_mean"]
        signed_read = vectors["read_signed_mean"]
        signed_causal = vectors["causal_delta"]
        support_read = [
            float(value) if value is not None else float("nan")
            for value in vectors["support_oriented_read"]
        ]
        method_seed = seed + 1000 * method_index
        common = {
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
        }
        by_method[method] = {
            "n": len(causal),
            "item_names": vectors["item_names"],
            "outcome": f"{outcome}.positive_damage",
            "variables": {
                "write": "aggregate.write_abs_mean",
                "read": "aggregate.read_abs_mean",
                "causal": f"{outcome}.positive_damage",
                "causal_delta_also_stored": f"{outcome}.delta",
                "independence_note": (
                    "READ contains no WRITE or activation-magnitude factor"
                ),
            },
            "pearson": {
                "causal_vs_read": _safe_statistic(
                    pearson_with_ci,
                    causal,
                    read,
                    seed=method_seed + 1,
                    **common,
                ),
                "causal_vs_write": _safe_statistic(
                    pearson_with_ci,
                    causal,
                    write,
                    seed=method_seed + 2,
                    **common,
                ),
                "write_vs_read": _safe_statistic(
                    pearson_with_ci,
                    write,
                    read,
                    seed=method_seed + 3,
                    **common,
                ),
                "predicted_vs_real": _safe_statistic(
                    pearson_with_ci,
                    predicted,
                    causal,
                    seed=method_seed + 4,
                    **common,
                ),
            },
            "partial_correlations": {
                "causal_read_given_write": _safe_statistic(
                    partial_correlation_with_ci,
                    causal,
                    read,
                    write,
                    seed=method_seed + 5,
                    **common,
                ),
                "causal_write_given_read": _safe_statistic(
                    partial_correlation_with_ci,
                    causal,
                    write,
                    read,
                    seed=method_seed + 6,
                    **common,
                ),
            },
            "regressions": {
                "causal_on_write_plus_read": _safe_statistic(
                    standardized_regression_with_ci,
                    causal,
                    write,
                    read,
                    interaction=False,
                    seed=method_seed + 7,
                    **common,
                ),
                "causal_on_write_times_read": _safe_statistic(
                    standardized_regression_with_ci,
                    causal,
                    write,
                    read,
                    interaction=True,
                    seed=method_seed + 8,
                    **common,
                ),
            },
            "signed_supplement": {
                "variables": {
                    "write": "aggregate.write_signed_mean",
                    "read": "aggregate.read_signed_mean",
                    "causal": f"{outcome}.delta",
                },
                "pearson": {
                    "causal_delta_vs_signed_read": _safe_statistic(
                        pearson_with_ci,
                        signed_causal,
                        signed_read,
                        seed=method_seed + 21,
                        **common,
                    ),
                    "causal_delta_vs_signed_write": _safe_statistic(
                        pearson_with_ci,
                        signed_causal,
                        signed_write,
                        seed=method_seed + 22,
                        **common,
                    ),
                    "signed_write_vs_signed_read": _safe_statistic(
                        pearson_with_ci,
                        signed_write,
                        signed_read,
                        seed=method_seed + 23,
                        **common,
                    ),
                },
                "partial_correlations": {
                    "causal_read_given_write": _safe_statistic(
                        partial_correlation_with_ci,
                        signed_causal,
                        signed_read,
                        signed_write,
                        seed=method_seed + 24,
                        **common,
                    ),
                    "causal_write_given_read": _safe_statistic(
                        partial_correlation_with_ci,
                        signed_causal,
                        signed_write,
                        signed_read,
                        seed=method_seed + 25,
                        **common,
                    ),
                },
                "regressions": {
                    "causal_delta_on_signed_write_plus_read": _safe_statistic(
                        standardized_regression_with_ci,
                        signed_causal,
                        signed_write,
                        signed_read,
                        interaction=False,
                        seed=method_seed + 26,
                        **common,
                    ),
                    "causal_delta_on_signed_write_times_read": _safe_statistic(
                        standardized_regression_with_ci,
                        signed_causal,
                        signed_write,
                        signed_read,
                        interaction=True,
                        seed=method_seed + 27,
                        **common,
                    ),
                },
            },
            "product_based_diagnostic": {
                "variable": "aggregate.support_oriented_read",
                "formula": "sum(WRITE*READ)/sum(abs(WRITE))",
                "warning": (
                    "This diagnostic contains WRITE and is excluded from P1, "
                    "partial correlations, regressions, F1, F2, and F6."
                ),
                "causal_damage_vs_support_oriented_read": _safe_statistic(
                    pearson_with_ci,
                    causal,
                    support_read,
                    seed=method_seed + 31,
                    **common,
                ),
            },
            "raw_analysis_vectors": vectors,
        }
    return {
        "outcome": outcome,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "by_method": by_method,
        "interpretation_guardrail": (
            "These are descriptive estimates and CIs. No PASS is assigned because "
            "'large' and 'approximately zero' had no frozen numeric thresholds."
        ),
    }


def _draw_regression_panel(
    axis: Any,
    x: Sequence[float],
    y: Sequence[float],
    statistic: Mapping[str, Any],
    *,
    xlabel: str,
    title: str,
    color: str,
) -> None:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[finite], y_values[finite]
    axis.set(xlabel=xlabel, ylabel="causal positive damage", title=title)
    if len(x_values) < 3:
        axis.text(
            0.5, 0.5, f"Not estimable\nN = {len(x_values)}", ha="center", va="center"
        )
        return
    axis.scatter(x_values, y_values, s=30, alpha=0.75, color=color)
    if float(np.std(x_values)) > 0:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        grid = np.linspace(float(x_values.min()), float(x_values.max()), 100)
        axis.plot(grid, slope * grid + intercept, color="black", linewidth=1.5)
    if statistic.get("status") == "ESTIMATED":
        annotation = (
            f"r = {statistic['estimate']:.3f}\n"
            f"95% CI [{statistic['ci_low']:.3f}, {statistic['ci_high']:.3f}]\n"
            f"N = {len(x_values)}"
        )
    else:
        annotation = f"r not estimable\nN = {len(x_values)}"
    axis.text(
        0.04,
        0.96,
        annotation,
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )


def plot_f1(
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    path: str | Path,
    *,
    method: str = PRIMARY_DIRECTION_METHOD,
) -> Path:
    """Save F1: causal damage vs attribution READ and vs WRITE."""

    vectors = analysis_vectors(rows, method, outcome=str(analysis["outcome"]))
    method_stats = analysis["by_method"].get(method, {})
    pearson = method_stats.get("pearson", {})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _draw_regression_panel(
        axes[0],
        vectors["read_strength"],
        vectors["causal_positive_damage"],
        pearson.get("causal_vs_read", {}),
        xlabel="mean absolute attribution READ",
        title="CAUSAL vs READ",
        color="#3366A3",
    )
    _draw_regression_panel(
        axes[1],
        vectors["write_strength"],
        vectors["causal_positive_damage"],
        pearson.get("causal_vs_write", {}),
        xlabel="mean absolute WRITE",
        title="CAUSAL vs WRITE",
        color="#C15A2A",
    )
    figure.suptitle(f"F1 — two-hop load-bearing signal ({method})")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return target.resolve()


def plot_f2(
    analysis: Mapping[str, Any],
    path: str | Path,
    *,
    method: str = PRIMARY_DIRECTION_METHOD,
) -> Path:
    """Save F2: standardized coefficients and partial correlations with CIs."""

    stats = analysis["by_method"].get(method, {})
    regression = stats.get("regressions", {}).get("causal_on_write_plus_read", {})
    partial = stats.get("partial_correlations", {})
    labels: list[str] = []
    estimates: list[float] = []
    low: list[float] = []
    high: list[float] = []
    if regression.get("status") == "ESTIMATED":
        for variable, label in (("write", "β WRITE"), ("read", "β READ")):
            interval = regression["coefficient_intervals"][variable]
            labels.append(label)
            estimates.append(float(regression["coefficients"][variable]))
            low.append(float(interval["ci_low"]))
            high.append(float(interval["ci_high"]))
    for key, label in (
        ("causal_write_given_read", "partial r WRITE | READ"),
        ("causal_read_given_write", "partial r READ | WRITE"),
    ):
        value = partial.get(key, {})
        if value.get("status") == "ESTIMATED":
            labels.append(label)
            estimates.append(float(value["estimate"]))
            low.append(float(value["ci_low"]))
            high.append(float(value["ci_high"]))

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if estimates:
        positions = np.arange(len(estimates))
        error = np.maximum(
            0.0,
            np.vstack(
                [
                    np.asarray(estimates) - np.asarray(low),
                    np.asarray(high) - np.asarray(estimates),
                ]
            ),
        )
        axis.bar(positions, estimates, color=["#C15A2A", "#3366A3"] * 2)
        axis.errorbar(
            positions,
            estimates,
            yerr=error,
            fmt="none",
            color="black",
            capsize=4,
        )
        axis.set_xticks(positions, labels, rotation=12, ha="right")
        axis.axhline(0.0, color="black", linewidth=1)
    else:
        axis.text(0.5, 0.5, "Not estimable", ha="center", va="center")
    axis.set(
        ylabel="standardized estimate / partial correlation",
        title=f"F2 — conditional WRITE and READ effects ({method}, N={stats.get('n', 0)})",
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return target.resolve()


def plot_f6(
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Save F6: F1 recomputed under raw J-Lens and independent MD directions."""

    methods = (PRIMARY_DIRECTION_METHOD, MD_DIRECTION_METHOD)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for row_index, method in enumerate(methods):
        vectors = analysis_vectors(rows, method, outcome=str(analysis["outcome"]))
        pearson = analysis["by_method"].get(method, {}).get("pearson", {})
        _draw_regression_panel(
            axes[row_index, 0],
            vectors["read_strength"],
            vectors["causal_positive_damage"],
            pearson.get("causal_vs_read", {}),
            xlabel="mean absolute attribution READ",
            title=f"{method}: CAUSAL vs READ",
            color="#3366A3",
        )
        _draw_regression_panel(
            axes[row_index, 1],
            vectors["write_strength"],
            vectors["causal_positive_damage"],
            pearson.get("causal_vs_write", {}),
            xlabel="mean absolute WRITE",
            title=f"{method}: CAUSAL vs WRITE",
            color="#C15A2A",
        )
    figure.suptitle(
        "F6 — direction robustness (attribution READ; weight READ is localized in nb04)"
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return target.resolve()


def run_twohop_phase(
    bundle: Any,
    lens: Any,
    *,
    md_artifact_path: str | Path = DEFAULT_MD_ARTIFACT,
    jlens_root: str | Path = DEFAULT_JLENS_ROOT,
    supplement_path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
    output_path: str | Path | None = None,
    figures_dir: str | Path = ROOT / "results/figures",
    layers: Sequence[int] | None = None,
    screen_batch_size: int = 8,
    max_length: int = 128,
    n_bootstrap: int = 5000,
    seed: int = SEED,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run dataset loading, frozen filtering, measurements, stats, and F1/F2/F6."""

    set_seed(seed)
    collection = load_tokenized_twohop_collection(
        bundle.tokenizer,
        jlens_root=jlens_root,
        supplement_path=supplement_path,
    )
    layer_list = (
        sorted(set(int(layer) for layer in layers))
        if layers is not None
        else workspace_layers(bundle.lens_model.n_layers, lens.source_layers)
    )
    eligibility = screen_clean_eligibility(
        bundle.hf_model,
        bundle.tokenizer,
        collection["items"],
        batch_size=screen_batch_size,
        max_length=max_length,
        top_k=10,
    )
    eligibility_by_name = {str(record["name"]): record for record in eligibility}
    eligible_items = [
        item
        for item in collection["items"]
        if eligibility_by_name[str(item["name"])]["eligible"]
    ]
    assignments, direction_coverage = prepare_direction_assignments(
        lens,
        bundle.lens_model,
        eligible_items,
        layer_list,
        md_artifact_path=md_artifact_path,
        model_id=bundle.model_id,
        model_revision=bundle.revision,
    )
    rows = measure_twohop_rows(
        bundle,
        eligible_items,
        eligibility_by_name,
        assignments,
        layer_list,
        max_length=max_length,
        fail_fast=fail_fast,
    )
    analyses = {
        "ablation": analyze_measurements(
            rows,
            outcome="ablation",
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "clean_clamped_swap": analyze_measurements(
            rows,
            outcome="clean_clamped_swap",
            n_bootstrap=n_bootstrap,
            seed=seed + 100_000,
        ),
    }

    model_slug = bundle.model_id.split("/")[-1].lower().replace("-instruct", "")
    figures_root = Path(figures_dir)
    figure_paths = {
        "f1": plot_f1(
            rows,
            analyses["ablation"],
            figures_root / f"f1_twohop_{model_slug}.png",
        ),
        "f2": plot_f2(
            analyses["ablation"],
            figures_root / f"f2_twohop_{model_slug}.png",
        ),
        "f6": plot_f6(
            rows,
            analyses["ablation"],
            figures_root / f"f6_direction_robustness_{model_slug}.png",
        ),
    }
    n_expected = sum(len(method_items) for method_items in assignments.values())
    n_success = sum(row.get("measurement_status") == "OK" for row in rows)
    corpus_criterion = {
        "criterion": "n_clean_eligible >= 150",
        "threshold": 150,
        "n_clean_eligible": len(eligible_items),
        "status": "PASS" if len(eligible_items) >= 150 else "FAIL",
        "raises_on_failure": False,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPUTED" if n_success == n_expected else "PARTIAL",
        "hypothesis_status": "DESCRIPTIVE_ESTIMATES_ONLY",
        "hypothesis_status_reason": (
            "The preregistration did not assign numerical thresholds to 'large' "
            "or 'approximately zero'; estimates and bootstrap CIs are reported "
            "without manufacturing a PASS."
        ),
        "metadata": {
            "model_id": bundle.model_id,
            "model_revision": bundle.revision,
            "seed": seed,
            "workspace_layers": layer_list,
            "workspace_intervention_scope": "all layers and all prompt positions",
            "behavior_metric": "logit(target) - logit(foil)",
            "causal_delta_sign": "M_edited - M_clean",
            "positive_damage_sign": "M_clean - M_edited",
            "primary_direction": PRIMARY_DIRECTION_METHOD,
            "primary_direction_formula": "normalize(W_U[token] @ J_layer)",
            "rms_gain_folded_included": False,
            "md_direction_role": "required independent robustness subset",
            "eligibility_rule": (
                "target top1 and concept+foil concept both absent from clean top10"
            ),
            "headline_variables": {
                "write": "mean(abs(WRITE)) across layer-position coordinates",
                "read": (
                    "mean(abs(attribution READ)) across layer-position coordinates; "
                    "does not contain WRITE"
                ),
                "causal": "positive all-band ablation damage",
            },
            "signed_supplement_variables": {
                "write": "mean(WRITE)",
                "read": "mean(attribution READ)",
                "causal": "M_edited - M_clean",
            },
            "product_based_diagnostic_only": (
                "support_oriented_read=sum(WRITE*READ)/sum(abs(WRITE)); excluded "
                "from headline and conditional analyses because it contains WRITE"
            ),
            "weight_based_read": (
                "not estimated here; component-local weight READ is a separate nb04 phase"
            ),
            "n_bootstrap": n_bootstrap,
            "max_length": max_length,
        },
        "corpus_criterion": corpus_criterion,
        "sample_counts": {
            "n_combined": collection["n_combined"],
            "n_tokenizable": collection["n_tokenizable"],
            "n_tokenization_rejected": collection["n_tokenization_rejected"],
            "n_clean_eligible": len(eligible_items),
            "n_clean_ineligible": len(collection["items"]) - len(eligible_items),
            "n_measurement_rows_expected": n_expected,
            "n_measurement_rows": len(rows),
            "n_measurement_success": n_success,
            "n_measurement_error": len(rows) - n_success,
            "n_by_method": {
                method: {
                    "assigned": len(method_items),
                    "successful": sum(
                        row.get("direction_method") == method
                        and row.get("measurement_status") == "OK"
                        for row in rows
                    ),
                }
                for method, method_items in assignments.items()
            },
        },
        "dataset": {
            key: value for key, value in collection.items() if key not in {"items"}
        },
        "eligibility": eligibility,
        "direction_coverage": direction_coverage,
        "rows": rows,
        "analyses": analyses,
        "figures": {
            key: _relative_or_absolute(path) for key, path in figure_paths.items()
        },
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else (ROOT / "data/raw" / f"02_twohop_{model_slug}.json")
    )
    strict_payload = _json_ready(payload)
    save_json(destination, strict_payload)
    print(
        f"TWO-HOP {strict_payload['status']}: "
        f"eligible={len(eligible_items)}, rows={n_success}/{n_expected}, "
        f"MD={len(assignments[MD_DIRECTION_METHOD])}"
    )
    print(
        f"TWO-HOP CORPUS {corpus_criterion['status']}: "
        f"N_clean_eligible={len(eligible_items)} (required >=150); continuing either way."
    )
    print("Hypothesis verdict not auto-assigned; inspect reported estimates and CIs.")
    return strict_payload


def run_qwen_twohop_phase(
    *,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    lens_path: str | Path | None = None,
    md_artifact_path: str | Path = DEFAULT_MD_ARTIFACT,
    **phase_kwargs: Any,
) -> dict[str, Any]:
    """Model-loading entry point for notebook 02.

    The published pinned lens is used only for Qwen2.5-7B.  Other Qwen scales
    must supply an explicitly fitted ``lens_path``; no scale is silently paired
    with the wrong Jacobian lens.
    """

    if not model_id.startswith("Qwen/Qwen2.5-"):
        raise ValueError(
            "This entry point is restricted to the preregistered Qwen2.5 family"
        )
    bundle = load_model(model_id)
    try:
        if lens_path is None:
            if model_id != "Qwen/Qwen2.5-7B-Instruct":
                raise ValueError(f"lens_path is required for {model_id}")
            lens = load_published_lens(model_id)
        else:
            lens = load_local_lens(lens_path)
        return run_twohop_phase(
            bundle,
            lens,
            md_artifact_path=md_artifact_path,
            **phase_kwargs,
        )
    finally:
        release_model(bundle)

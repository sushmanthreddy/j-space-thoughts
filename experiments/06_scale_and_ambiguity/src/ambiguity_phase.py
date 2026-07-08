"""Flagship ambiguity WRITE/READ/CAUSAL orchestration.

The preregistered primary directions are raw Jacobian-lens ``W_U J`` token
directions. RMS-gain-folded directions are constructed only as a separately
labelled direction-sensitivity analysis and never replace the primary causal
interventions. This ambiguity phase is intentionally J-Lens-only: the required
mean-difference robustness analysis for P1 belongs to the two-hop phase.

Commitment is inferred from the mean of the original and mirrored r1-vs-r2
logit margins. No reading is marked as gold, and no intervention outcome is
consulted when choosing the committed reading.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jlens.hooks import ActivationRecorder

from src.ambiguity_data import (
    DATASET_SEED,
    DEFAULT_AMBIGUITY_SPEC,
    load_tokenized_ambiguity_items,
)
from src.interventions import (
    ablation_edits,
    clamped_swap_edits,
    forward_logits,
    suppress_output_token,
)
from src.jlens_iface import (
    PUBLISHED_LENSES,
    jlens_direction_bank,
    load_published_lens,
    token_rank,
    workspace_layers,
)
from src.metrics import (
    binomial_rate_with_ci,
    bootstrap_statistic,
    partial_correlation_with_ci,
    pearson_with_ci,
    save_json,
    signed_causal_delta,
    support_damage,
)
from src.model_utils import (
    MODEL_REVISIONS,
    decode_topk,
    load_model,
    release_model,
    set_seed,
)
from src.plotting import save_figure, set_style
from src.twohop_phase import aggregate_write_read


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = MODEL_REVISIONS[MODEL_ID]
PRIMARY_DIRECTION_METHOD = "jlens_raw_wu_j"
RMS_SENSITIVITY_METHOD = "jlens_rms_gain_folded"
SCHEMA_VERSION = "ambiguity-phase-v1"
SEED = DATASET_SEED
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "05_ambiguity_qwen7b.json"
DEFAULT_FIGURE = ROOT / "results" / "figures" / "f8_ambiguity_write_read.png"
J_LENS_ONLY_LIMITATION = (
    "Ambiguity interventions use only raw J-Lens token directions. This phase "
    "does not establish mean-difference direction robustness; the independent "
    "MD robustness requirement for P1 is evaluated in the two-hop phase."
)
UPSTREAM_GATE_CONTEXT = {
    "source": "data/raw/00_gates_qwen7b.json",
    "model": MODEL_ID,
    "strict_g2_status": "FAIL",
    "strict_workspace_usable": False,
    "raw_primary_known_case_directional_success": False,
    "note": (
        "The RMS-gain-folded sensitivity direction passed only the directional "
        "subgate, while this phase uses raw W_U J directions for causal edits. "
        "All ambiguity results are therefore diagnostic rather than confirmatory."
    ),
}
OUTPUT_SUPPRESSION_INTERPRETATION = {
    "status": "STRUCTURAL_ZERO_NEGATIVE_CONTROL",
    "expected_exact_zero": True,
    "reason": (
        "Ambiguity tokenization requires both concept-token IDs to be disjoint "
        "from the A/B behavior-token IDs. Suppressing only a concept vocabulary "
        "logit cannot change the A-minus-B logit metric, so this control is "
        "exactly zero by construction."
    ),
    "evidential_role": (
        "Schema/instrumentation check only; the zero does not provide additional "
        "causal evidence beyond the internal intervention itself."
    ),
}
DIRECTION_POLICY = {
    "primary": PRIMARY_DIRECTION_METHOD,
    "primary_formula": "normalize(W_U[token] @ J_layer)",
    "fold_rms_gain_primary": False,
    "sensitivity": RMS_SENSITIVITY_METHOD,
    "sensitivity_role": "direction cosine diagnostic only; no headline causal use",
    "independent_read": "mean(abs(grad(M) dot direction)) over layer-position coordinates",
    "diagnostic_only_quantities": (
        "support_oriented_read and sum(WRITE*READ) first-order prediction"
    ),
    "phase_analysis_role": "diagnostic_upstream_strict_g2_failed",
    "limitation": J_LENS_ONLY_LIMITATION,
}

# Frozen before outcome inspection. Tokenization alone determines inclusion.
META_TOKEN_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "key": "interpretation_en",
        "label": "interpretation",
        "language": "en",
        "gloss": "interpretation",
    },
    {
        "key": "meaning_en",
        "label": "meaning",
        "language": "en",
        "gloss": "meaning",
    },
    {
        "key": "ambiguous_en",
        "label": "ambiguous",
        "language": "en",
        "gloss": "ambiguous",
    },
    {
        "key": "ambiguity_en",
        "label": "ambiguity",
        "language": "en",
        "gloss": "ambiguity",
    },
    {
        "key": "explain_zh",
        "label": "解释",
        "language": "zh",
        "gloss": "interpret or explain",
    },
    {
        "key": "meaning_formal_zh",
        "label": "含义",
        "language": "zh",
        "gloss": "meaning",
    },
    {
        "key": "meaning_colloquial_zh",
        "label": "意思",
        "language": "zh",
        "gloss": "meaning or sense",
    },
    {
        "key": "ambiguity_zh",
        "label": "歧义",
        "language": "zh",
        "gloss": "ambiguity",
    },
    {
        "key": "vague_zh",
        "label": "模糊",
        "language": "zh",
        "gloss": "ambiguous or vague",
    },
    {
        "key": "interpret_zh",
        "label": "解读",
        "language": "zh",
        "gloss": "interpretation or reading",
    },
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_ready(value: Any) -> Any:
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return [int(token_id) for token_id in encoded]
    except Exception as error:
        raise ValueError(f"Tokenizer failed on {text!r}: {error}") from error


def resolve_meta_token_candidates(
    tokenizer: Any,
    candidates: Sequence[Mapping[str, str]] = META_TOKEN_CANDIDATES,
) -> dict[str, Any]:
    """Apply the frozen, outcome-independent exact-single-token selection rule."""

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_token_ids: dict[int, str] = {}
    for index, raw_candidate in enumerate(candidates):
        required = {"key", "label", "language", "gloss"}
        _require(
            set(raw_candidate) == required,
            f"Meta candidate {index} must have keys {sorted(required)}",
        )
        candidate = {key: str(raw_candidate[key]) for key in required}
        _require(all(candidate.values()), f"Meta candidate {index} has an empty field")
        _require(
            candidate["key"] not in seen_keys,
            f"Duplicate meta candidate key {candidate['key']!r}",
        )
        seen_keys.add(candidate["key"])
        attempts: list[dict[str, Any]] = []
        resolved: tuple[str, int] | None = None
        for surface in (f" {candidate['label']}", candidate["label"]):
            token_ids = _encode_ids(tokenizer, surface)
            attempts.append({"surface": surface, "token_ids": token_ids})
            if len(token_ids) == 1:
                resolved = (surface, token_ids[0])
                break
        if resolved is None:
            rejected.append(
                {
                    **candidate,
                    "candidate_index": index,
                    "reason": "no_exact_single_token_surface",
                    "attempts": attempts,
                }
            )
            continue
        surface, token_id = resolved
        if token_id in seen_token_ids:
            rejected.append(
                {
                    **candidate,
                    "candidate_index": index,
                    "reason": "token_id_collision",
                    "collides_with": seen_token_ids[token_id],
                    "token_id": token_id,
                    "surface": surface,
                    "attempts": attempts,
                }
            )
            continue
        seen_token_ids[token_id] = candidate["key"]
        selected.append(
            {
                **candidate,
                "candidate_index": index,
                "token_id": token_id,
                "surface": surface,
                "attempts": attempts,
            }
        )
    return {
        "bank_name": "preregistered_abstract_interpretation_meta_tokens_v1",
        "selection_rule": (
            "First exact one-token surface among leading-space label and bare "
            "label; token-ID collisions keep the earlier preregistered candidate."
        ),
        "selection_uses_model_outputs": False,
        "analysis_role": "diagnostic_nonconfirmatory",
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "n_rejected": len(rejected),
        "selected": selected,
        "rejected": rejected,
    }


def _final_logit_vector(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.float()
    if logits.ndim == 3 and logits.shape[0] == 1:
        return logits[0, -1].float()
    raise ValueError(
        f"Expected logits [vocab] or [1, sequence, vocab], got {tuple(logits.shape)}"
    )


def probe_margin_from_logits(
    logits: torch.Tensor,
    prepared_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Orient an A-vs-B margin to the invariant semantic r1-vs-r2 axis."""

    vector = _final_logit_vector(logits)
    try:
        counterbalance = prepared_probe["counterbalance"]
        continuation_a = prepared_probe["continuations"]["A"]
        continuation_b = prepared_probe["continuations"]["B"]
        token_a = int(continuation_a["token_id"])
        token_b = int(continuation_b["token_id"])
        sign = int(counterbalance["fixed_ab_margin_sign"])
        reading_to_label = counterbalance["reading_to_label"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed prepared probe: {error}") from error
    _require(token_a != token_b, "A/B continuation token IDs must differ")
    _require(sign in {-1, 1}, "fixed_ab_margin_sign must be -1 or +1")
    expected_sign = 1 if reading_to_label == {"r1": "A", "r2": "B"} else -1
    _require(
        reading_to_label in ({"r1": "A", "r2": "B"}, {"r1": "B", "r2": "A"}),
        "reading_to_label must be one of the two counterbalanced assignments",
    )
    _require(sign == expected_sign, "counterbalance sign disagrees with assignment")
    fixed_ab = float(vector[token_a] - vector[token_b])
    return {
        "group_id": str(counterbalance["group_id"]),
        "variant": str(counterbalance["variant"]),
        "variant_index": int(counterbalance["variant_index"]),
        "fixed_ab_margin": fixed_ab,
        "r1_minus_r2_margin": float(sign * fixed_ab),
        "fixed_ab_margin_sign": sign,
        "A_token_id": token_a,
        "B_token_id": token_b,
        "A_logit": float(vector[token_a]),
        "B_logit": float(vector[token_b]),
    }


def infer_counterbalanced_commitment(
    clean_variant_margins: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Infer commitment from the mean semantic margin, without a gold reading."""

    _require(len(clean_variant_margins) == 2, "Exactly two clean variants are required")
    _require(
        math.isfinite(tie_tolerance) and tie_tolerance >= 0.0,
        "tie_tolerance must be finite and nonnegative",
    )
    variants = [str(record["variant"]) for record in clean_variant_margins]
    _require(
        variants == ["original", "mirrored"],
        "Clean variants must be ordered original then mirrored",
    )
    group_ids = {str(record["group_id"]) for record in clean_variant_margins}
    _require(len(group_ids) == 1, "Counterbalanced variants must share one group_id")
    margins = np.asarray(
        [float(record["r1_minus_r2_margin"]) for record in clean_variant_margins],
        dtype=float,
    )
    _require(np.isfinite(margins).all(), "Clean semantic margins must be finite")
    mean_margin = float(margins.mean())
    if abs(mean_margin) <= tie_tolerance:
        return {
            "status": "TIE",
            "selection_rule": "sign(mean(original, mirrored) r1-vs-r2 margin)",
            "uses_gold_reading": False,
            "group_id": next(iter(group_ids)),
            "r1_minus_r2_by_variant": margins.tolist(),
            "mean_r1_minus_r2_margin": mean_margin,
            "tie_tolerance": float(tie_tolerance),
            "committed_reading": None,
            "alternate_reading": None,
            "committed_sign": None,
            "committed_mean_margin": None,
            "counterbalance_agreement": False,
        }
    committed = "r1" if mean_margin > 0.0 else "r2"
    alternate = "r2" if committed == "r1" else "r1"
    committed_sign = 1 if committed == "r1" else -1
    committed_margins = committed_sign * margins
    return {
        "status": "COMMITTED",
        "selection_rule": "sign(mean(original, mirrored) r1-vs-r2 margin)",
        "uses_gold_reading": False,
        "group_id": next(iter(group_ids)),
        "r1_minus_r2_by_variant": margins.tolist(),
        "mean_r1_minus_r2_margin": mean_margin,
        "tie_tolerance": float(tie_tolerance),
        "committed_reading": committed,
        "alternate_reading": alternate,
        "committed_sign": committed_sign,
        "committed_margin_by_variant": committed_margins.tolist(),
        "committed_mean_margin": float(committed_margins.mean()),
        "counterbalance_agreement": bool(np.all(committed_margins > 0.0)),
        "counterbalance_gap_abs": float(abs(margins[0] - margins[1])),
    }


def _effect_record(clean_metric: float, edited_metric: float) -> dict[str, float]:
    return {
        "edited_committed_margin": float(edited_metric),
        "delta": signed_causal_delta(clean_metric, edited_metric),
        "positive_damage": support_damage(clean_metric, edited_metric),
    }


def _mean_effect(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    _require(len(records) == 2, "Effect aggregation requires two variants")
    fields = ("edited_committed_margin", "delta", "positive_damage")
    values = {
        field: np.asarray([float(record[field]) for record in records], dtype=float)
        for field in fields
    }
    _require(
        all(np.isfinite(array).all() for array in values.values()),
        "Effect records must be finite",
    )
    return {
        **{field: float(array.mean()) for field, array in values.items()},
        "variant_positive_damage": values["positive_damage"].tolist(),
        "variant_damage_sign_agreement": bool(
            np.all(values["positive_damage"] > 0.0)
            or np.all(values["positive_damage"] < 0.0)
        ),
    }


def aggregate_counterbalanced_variants(
    variant_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average already commitment-oriented intervention results across variants."""

    _require(len(variant_records) == 2, "Exactly two variant records are required")
    variants = [str(record["counterbalance"]["variant"]) for record in variant_records]
    _require(
        variants == ["original", "mirrored"],
        "Variant records must be ordered original then mirrored",
    )
    clean = np.asarray(
        [float(record["clean_committed_margin"]) for record in variant_records],
        dtype=float,
    )
    _require(np.isfinite(clean).all(), "Clean committed margins must be finite")
    ablation = _mean_effect([record["ablation"] for record in variant_records])
    swap = _mean_effect([record["clean_clamped_swap"] for record in variant_records])
    committed_suppression = _mean_effect(
        [
            record["output_suppression"]["committed_concept"]
            for record in variant_records
        ]
    )
    alternate_suppression = _mean_effect(
        [
            record["output_suppression"]["alternate_concept"]
            for record in variant_records
        ]
    )

    aggregate_keys = (
        "write_abs_mean",
        "read_abs_mean",
        "support_oriented_read",
        "first_order_predicted_delta",
        "first_order_predicted_positive_damage",
    )

    def mean_attribution(role: str) -> dict[str, float]:
        aggregates = [
            (
                record["attribution"]["aggregate"]
                if role == "committed_concept"
                else record["attribution"]["alternate_concept"]["aggregate"]
            )
            for record in variant_records
        ]
        result: dict[str, float] = {}
        for key in aggregate_keys:
            values = np.asarray(
                [float(aggregate[key]) for aggregate in aggregates],
                dtype=float,
            )
            _require(
                np.isfinite(values).all(),
                f"Non-finite {role} attribution aggregate {key}",
            )
            result[key] = float(values.mean())
        return result

    committed_attribution = mean_attribution("committed_concept")
    alternate_attribution = mean_attribution("alternate_concept")
    swap_variant_flips = [
        float(record["clean_clamped_swap"]["edited_committed_margin"]) < 0.0
        for record in variant_records
    ]
    damage_gap = ablation["positive_damage"] - committed_suppression["positive_damage"]
    return {
        "n_variants": 2,
        "clean_committed_margin": float(clean.mean()),
        "clean_committed_margin_by_variant": clean.tolist(),
        "attribution": {
            # Flat committed fields are retained for compatibility with the
            # existing analysis/metrics schema.
            **committed_attribution,
            "committed_concept": committed_attribution,
            "alternate_concept": alternate_attribution,
            "comparison": {
                "committed_minus_alternate_write_abs_mean": float(
                    committed_attribution["write_abs_mean"]
                    - alternate_attribution["write_abs_mean"]
                ),
                "committed_minus_alternate_read_abs_mean": float(
                    committed_attribution["read_abs_mean"]
                    - alternate_attribution["read_abs_mean"]
                ),
            },
        },
        "ablation": ablation,
        "clean_clamped_swap": {
            **swap,
            "flips_committed_mean_margin": bool(swap["edited_committed_margin"] < 0.0),
            "variant_flips_committed_margin": swap_variant_flips,
            "counterbalance_robust_flip": bool(all(swap_variant_flips)),
            "counterbalance_robust_flip_rule": (
                "edited committed margin < 0 in both original and mirrored probes"
            ),
        },
        "output_suppression": {
            "committed_concept": committed_suppression,
            "alternate_concept": alternate_suppression,
            "interpretation": OUTPUT_SUPPRESSION_INTERPRETATION,
        },
        "internal_minus_suppression_positive_damage": float(damage_gap),
        "ablation_damage_exceeds_suppression": bool(damage_gap > 0.0),
    }


def _resolved_positions(
    sequence_length: int,
    positions: Sequence[int] | None,
) -> list[int]:
    if positions is None:
        return list(range(sequence_length))
    resolved: list[int] = []
    for raw_position in positions:
        position = int(raw_position)
        if position < 0:
            position += sequence_length
        _require(
            0 <= position < sequence_length,
            f"Position {raw_position} outside sequence length {sequence_length}",
        )
        resolved.append(position)
    _require(len(resolved) == len(set(resolved)), "Attribution positions repeat")
    return resolved


def shared_direction_attribution(
    hf_model: torch.nn.Module,
    blocks: Sequence[torch.nn.Module],
    input_ids: torch.Tensor,
    directions_by_name: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    target_token_id: int,
    foil_token_id: int,
    behavior_position: int = -1,
    intervention_positions: Sequence[int] | None = None,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Project one shared behavior gradient onto every requested direction bank.

    Exactly one forward/backward is used regardless of the number of committed
    or meta-token directions. The returned ``read_abs_mean`` is the independent
    READ magnitude. Support-oriented READ and ``WRITE*READ`` remain explicitly
    diagnostic first-order quantities.
    """

    _require(
        input_ids.ndim == 2 and input_ids.shape[0] == 1,
        "Shared attribution requires one unpadded item",
    )
    _require(target_token_id != foil_token_id, "Behavior token IDs must differ")
    _require(bool(directions_by_name), "At least one direction bank is required")
    _require(
        not any(parameter.requires_grad for parameter in hf_model.parameters()),
        "Freeze model parameters before shared activation attribution",
    )
    names = list(directions_by_name)
    _require(len(names) == len(set(names)), "Direction names must be unique")
    layer_sets = {
        tuple(sorted(int(layer) for layer in directions))
        for directions in directions_by_name.values()
    }
    _require(
        len(layer_sets) == 1 and bool(next(iter(layer_sets))),
        "All direction banks must cover identical nonempty layers",
    )
    layers = list(next(iter(layer_sets)))
    sequence_length = int(input_ids.shape[1])
    behavior_index = behavior_position
    if behavior_index < 0:
        behavior_index += sequence_length
    _require(
        0 <= behavior_index < sequence_length,
        f"Behavior position {behavior_position} is out of range",
    )
    selected_positions = _resolved_positions(
        sequence_length,
        intervention_positions,
    )

    with (
        torch.enable_grad(),
        ActivationRecorder(
            blocks,
            at=layers,
            start_graph_at=layers[0],
        ) as recorder,
    ):
        logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        metric_tensor = (
            logits[0, behavior_index, int(target_token_id)]
            - logits[0, behavior_index, int(foil_token_id)]
        )
        activations = tuple(recorder.activations[layer] for layer in layers)
        gradients = torch.autograd.grad(
            metric_tensor,
            activations,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

    direction_results: dict[str, Any] = {}
    for name, direction_bank in directions_by_name.items():
        writes: dict[int, np.ndarray] = {}
        reads: dict[int, np.ndarray] = {}
        predicted_by_layer: dict[int, float] = {}
        for layer, activation, gradient in zip(
            layers,
            activations,
            gradients,
            strict=True,
        ):
            direction = (
                direction_bank[layer]
                .detach()
                .to(
                    activation.device,
                    torch.float32,
                )
            )
            norm = direction.norm()
            _require(
                bool(torch.isfinite(norm))
                and bool(
                    torch.isclose(
                        norm,
                        torch.ones((), device=norm.device),
                        atol=1e-4,
                        rtol=1e-4,
                    )
                ),
                f"Direction {name!r}, layer {layer} is not unit norm",
            )
            selected_activation = activation[0, selected_positions].detach().float()
            selected_gradient = gradient[0, selected_positions].detach().float()
            layer_write = selected_activation @ direction
            layer_read = selected_gradient @ direction
            writes[layer] = layer_write.cpu().numpy()
            reads[layer] = layer_read.cpu().numpy()
            predicted_by_layer[layer] = float((-(layer_write * layer_read).sum()).cpu())
        aggregate = aggregate_write_read(writes, reads)
        exact_predicted_delta = float(sum(predicted_by_layer.values()))
        _require(
            math.isclose(
                exact_predicted_delta,
                float(aggregate["first_order_predicted_delta"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ),
            f"Prediction aggregation drift for direction {name!r}",
        )
        direction_results[name] = {
            "write": writes,
            "read": reads,
            "predicted_delta_by_layer": predicted_by_layer,
            "predicted_delta": exact_predicted_delta,
            "aggregate": aggregate,
            "independent_read_field": "aggregate.read_abs_mean",
            "diagnostic_only_fields": [
                "aggregate.support_oriented_read",
                "aggregate.first_order_predicted_delta",
                "aggregate.first_order_predicted_positive_damage",
            ],
        }
    return {
        "metric": float(metric_tensor.detach().cpu()),
        "n_forward_backward": 1,
        "layers": layers,
        "positions": selected_positions,
        "direction_results": direction_results,
    }


def _direction_pair_diagnostics(
    concept: Mapping[int, torch.Tensor],
    alternate: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    _require(
        set(concept) == set(alternate) and bool(concept),
        "Direction pairs must cover identical nonempty layers",
    )
    cosines: dict[str, float] = {}
    conditions: dict[str, float] = {}
    for layer in sorted(concept):
        first = concept[layer].detach().float().cpu()
        second = alternate[layer].detach().float().cpu()
        cosine = float(torch.dot(first, second))
        basis = torch.stack([first, second])
        condition = float(torch.linalg.cond(basis @ basis.T))
        cosines[str(layer)] = cosine
        conditions[str(layer)] = condition
    return {
        "cosine_by_layer": cosines,
        "gram_condition_by_layer": conditions,
        "max_gram_condition": max(conditions.values()),
    }


def _rms_sensitivity_record(
    raw_directions: Mapping[int, torch.Tensor],
    folded_directions: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    _require(
        set(raw_directions) == set(folded_directions) and bool(raw_directions),
        "Raw and RMS-folded directions must cover identical layers",
    )
    cosines = {
        str(layer): float(
            torch.dot(
                raw_directions[layer].detach().float().cpu(),
                folded_directions[layer].detach().float().cpu(),
            )
        )
        for layer in sorted(raw_directions)
    }
    return {
        "analysis_role": "sensitivity_only",
        "raw_method": PRIMARY_DIRECTION_METHOD,
        "comparison_method": RMS_SENSITIVITY_METHOD,
        "cosine_by_layer": cosines,
        "mean_cosine": float(np.mean(list(cosines.values()))),
        "used_for_commitment": False,
        "used_for_causal_interventions": False,
    }


def _summarize_write(
    write_by_layer: Mapping[int, torch.Tensor | np.ndarray | Sequence[float]],
) -> dict[str, Any]:
    _require(bool(write_by_layer), "WRITE summary needs at least one layer")
    chunks: list[np.ndarray] = []
    by_layer: dict[str, Any] = {}
    for layer in sorted(write_by_layer):
        raw_values = write_by_layer[layer]
        if isinstance(raw_values, torch.Tensor):
            values = raw_values.detach().float().cpu().numpy().reshape(-1)
        else:
            values = np.asarray(raw_values, dtype=float).reshape(-1)
        _require(values.size > 0, f"Layer {layer} has no WRITE coordinates")
        _require(np.isfinite(values).all(), f"Layer {layer} WRITE is non-finite")
        chunks.append(values)
        by_layer[str(layer)] = {
            "n_positions": int(values.size),
            "signed_sum": float(values.sum()),
            "signed_mean": float(values.mean()),
            "abs_sum": float(np.abs(values).sum()),
            "abs_mean": float(np.abs(values).mean()),
        }
    combined = np.concatenate(chunks)
    return {
        "n_layers": len(chunks),
        "n_coordinates": int(combined.size),
        "signed_sum": float(combined.sum()),
        "signed_mean": float(combined.mean()),
        "abs_sum": float(np.abs(combined).sum()),
        "abs_mean": float(np.abs(combined).mean()),
        "by_layer": by_layer,
    }


def _rank_record(logits: torch.Tensor, token_id: int) -> dict[str, Any]:
    vector = _final_logit_vector(logits)
    return {
        "token_id": int(token_id),
        "logit": float(vector[int(token_id)]),
        "rank": token_rank(vector, int(token_id)),
    }


@torch.no_grad()
def _lens_readout_records(
    lens: Any,
    lens_model: Any,
    tokenizer: Any,
    residuals: Mapping[int, torch.Tensor],
    tracked_tokens: Mapping[str, int],
    layers: Sequence[int],
    *,
    top_k: int,
) -> dict[str, Any]:
    transported_rows: list[torch.Tensor] = []
    for layer in layers:
        hidden = residuals[int(layer)][0, -1].detach().float()
        transported_rows.append(lens.transport(hidden, int(layer)))
    all_logits = lens_model.unembed(torch.stack(transported_rows)).float().cpu()
    records: dict[str, Any] = {}
    for layer_index, layer in enumerate(layers):
        logits = all_logits[layer_index]
        records[str(layer)] = {
            "tracked": {
                key: _rank_record(logits, token_id)
                for key, token_id in tracked_tokens.items()
            },
            "top_tokens": decode_topk(
                tokenizer,
                logits,
                min(int(top_k), int(logits.numel())),
            ),
        }
    return records


def _prompt_tensors(
    bundle: Any,
    prompt: str,
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = len(bundle.tokenizer.encode(prompt, add_special_tokens=True))
    _require(
        token_count <= max_length,
        f"Refusing truncated ambiguity prompt with {token_count} tokens",
    )
    encoded = bundle.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )
    device = next(bundle.hf_model.parameters()).device
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def _clean_variant_context(
    bundle: Any,
    lens: Any,
    item: Mapping[str, Any],
    prepared_probe: Mapping[str, Any],
    meta_tokens: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    *,
    max_length: int,
    output_top_k: int,
    lens_top_k: int,
) -> dict[str, Any]:
    prompt = str(prepared_probe["prompt"])
    input_ids, attention_mask = _prompt_tensors(
        bundle,
        prompt,
        max_length=max_length,
    )
    with ActivationRecorder(bundle.lens_model.layers, at=list(layers)) as recorder:
        clean_logits = forward_logits(
            bundle.hf_model,
            input_ids,
            attention_mask=attention_mask,
        )
    residuals = {
        int(layer): recorder.activations[int(layer)].detach() for layer in layers
    }
    margin = probe_margin_from_logits(clean_logits, prepared_probe)
    final_logits = clean_logits[0, -1]
    concept_ids = {
        "r1_concept": int(item["concepts"]["r1"]["token_id"]),
        "r2_concept": int(item["concepts"]["r2"]["token_id"]),
    }
    meta_ids = {
        str(candidate["key"]): int(candidate["token_id"]) for candidate in meta_tokens
    }
    tracked_ids = {**concept_ids, **meta_ids}
    output_tracked = {
        key: _rank_record(final_logits, token_id)
        for key, token_id in tracked_ids.items()
    }
    behavior_choices = {
        label: _rank_record(
            final_logits,
            int(prepared_probe["continuations"][label]["token_id"]),
        )
        for label in ("A", "B")
    }
    reading_to_label = prepared_probe["counterbalance"]["reading_to_label"]
    lens_readout = _lens_readout_records(
        lens,
        bundle.lens_model,
        bundle.tokenizer,
        residuals,
        tracked_ids,
        layers,
        top_k=lens_top_k,
    )
    prompt_token_ids = [int(value) for value in input_ids[0].detach().cpu()]
    record = {
        "counterbalance": dict(prepared_probe["counterbalance"]),
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "prompt_tokens": [
            bundle.tokenizer.decode([token_id]) for token_id in prompt_token_ids
        ],
        "n_prompt_tokens": len(prompt_token_ids),
        "clean": {
            **margin,
            "behavior_choice_ranks": behavior_choices,
            "reading_choice_ranks": {
                reading: behavior_choices[label]
                for reading, label in reading_to_label.items()
            },
            "output_tracked": output_tracked,
            "top_tokens": decode_topk(
                bundle.tokenizer,
                final_logits,
                min(output_top_k, int(final_logits.numel())),
            ),
        },
        "lens_readout": lens_readout,
    }
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "clean_logits": clean_logits,
        "clean_residuals": residuals,
        "prepared_probe": prepared_probe,
        "record": record,
    }


def _committed_metric(
    logits: torch.Tensor,
    prepared_probe: Mapping[str, Any],
    committed_reading: str,
) -> float:
    margin = probe_margin_from_logits(logits, prepared_probe)["r1_minus_r2_margin"]
    sign = 1.0 if committed_reading == "r1" else -1.0
    _require(committed_reading in {"r1", "r2"}, "Unknown committed reading")
    return float(sign * float(margin))


def _meta_variant_measurements(
    bundle: Any,
    context: Mapping[str, Any],
    meta_tokens: Sequence[Mapping[str, Any]],
    raw_direction_bank: Mapping[int, Mapping[int, torch.Tensor]],
    committed_reading: str,
    layers: Sequence[int],
    *,
    output_top_k: int,
) -> dict[str, Any]:
    record = context["record"]
    clean_metric = float(record["clean"]["committed_margin"])
    shared = context["shared_attribution"]
    output: dict[str, Any] = {}
    for candidate in meta_tokens:
        key = str(candidate["key"])
        token_id = int(candidate["token_id"])
        directions = raw_direction_bank[token_id]
        shared_result = shared["direction_results"][f"meta::{key}"]
        writes = shared_result["write"]
        reads = shared_result["read"]
        aggregate = shared_result["aggregate"]
        write_arrays = {str(layer): writes[layer].tolist() for layer in layers}
        read_arrays = {str(layer): reads[layer].tolist() for layer in layers}
        lens_ranks = {
            str(layer): int(record["lens_readout"][str(layer)]["tracked"][key]["rank"])
            for layer in layers
        }
        final_rank = int(record["clean"]["output_tracked"][key]["rank"])
        base: dict[str, Any] = {
            "key": key,
            "label": candidate["label"],
            "language": candidate["language"],
            "gloss": candidate["gloss"],
            "token_id": token_id,
            "surface": candidate["surface"],
            "analysis_role": "diagnostic_nonconfirmatory",
            "raw_write_by_layer_position": write_arrays,
            "raw_read_by_layer_position": read_arrays,
            "write": _summarize_write(writes),
            "read": {
                "independent_read_abs_mean": float(aggregate["read_abs_mean"]),
                "read_abs_sum": float(aggregate["read_abs_sum"]),
                "read_signed_mean": float(aggregate["read_signed_mean"]),
                "by_layer": {
                    str(layer): {
                        "read_abs_mean": float(
                            aggregate["by_layer"][str(layer)]["read_abs_mean"]
                        ),
                        "read_signed_mean": float(
                            aggregate["by_layer"][str(layer)]["read_signed_mean"]
                        ),
                    }
                    for layer in layers
                },
            },
            "first_order_diagnostic": {
                "exact_negative_sum_write_read": float(
                    aggregate["first_order_predicted_delta"]
                ),
                "predicted_positive_damage": float(
                    aggregate["first_order_predicted_positive_damage"]
                ),
                "support_oriented_read": aggregate["support_oriented_read"],
                "analysis_role": "diagnostic_only",
            },
            "lens_rank_by_layer": lens_ranks,
            "mean_lens_rank": float(np.mean(list(lens_ranks.values()))),
            "final_output_rank": final_rank,
        }
        suppressed_logits = suppress_output_token(context["clean_logits"], token_id)
        suppressed_metric = _committed_metric(
            suppressed_logits,
            context["prepared_probe"],
            committed_reading,
        )
        base["output_suppression"] = {
            "status": "OK",
            "token_id": token_id,
            "clean_output_logit": float(context["clean_logits"][0, -1, token_id]),
            "suppressed_output_logit": float(suppressed_logits[0, -1, token_id]),
            **_effect_record(clean_metric, suppressed_metric),
            "interpretation": OUTPUT_SUPPRESSION_INTERPRETATION,
        }
        if final_rank <= output_top_k:
            base["ablation"] = {
                "status": "CONTROL_REJECTED",
                "reason": f"meta token appears in clean output top {output_top_k}",
            }
            output[key] = base
            continue
        try:
            ablated_logits = forward_logits(
                bundle.hf_model,
                context["input_ids"],
                attention_mask=context["attention_mask"],
                blocks=bundle.lens_model.layers,
                edits=ablation_edits(directions),
            )
            edited_metric = _committed_metric(
                ablated_logits,
                context["prepared_probe"],
                committed_reading,
            )
            base["ablation"] = {
                "status": "OK",
                "scope": "all workspace layers and all prompt positions",
                **_effect_record(clean_metric, edited_metric),
            }
        except Exception as error:  # diagnostic rows are retained independently
            base["ablation"] = {
                "status": "ERROR",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        output[key] = base
    return output


def _aggregate_meta_variants(
    variant_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(len(variant_records) == 2, "Meta aggregation requires two variants")
    candidate_keys = set(variant_records[0]["meta_tokens"])
    _require(
        candidate_keys == set(variant_records[1]["meta_tokens"]),
        "Meta candidates differ across counterbalanced variants",
    )
    output: dict[str, Any] = {}
    for key in sorted(candidate_keys):
        records = [record["meta_tokens"][key] for record in variant_records]
        write_abs = [float(record["write"]["abs_mean"]) for record in records]
        read_abs = [
            float(record["read"]["independent_read_abs_mean"]) for record in records
        ]
        predicted_damage = [
            float(record["first_order_diagnostic"]["predicted_positive_damage"])
            for record in records
        ]
        support_read = [
            record["first_order_diagnostic"]["support_oriented_read"]
            for record in records
        ]
        lens_ranks = [float(record["mean_lens_rank"]) for record in records]
        final_ranks = [float(record["final_output_rank"]) for record in records]
        base = {
            "key": key,
            "label": records[0]["label"],
            "language": records[0]["language"],
            "token_id": int(records[0]["token_id"]),
            "analysis_role": "diagnostic_nonconfirmatory",
            "mean_write_abs": float(np.mean(write_abs)),
            "mean_independent_read_abs": float(np.mean(read_abs)),
            "mean_first_order_predicted_positive_damage": float(
                np.mean(predicted_damage)
            ),
            "mean_support_oriented_read": (
                float(np.mean([float(value) for value in support_read]))
                if all(value is not None for value in support_read)
                else None
            ),
            "product_quantities_role": "diagnostic_only",
            "mean_lens_rank": float(np.mean(lens_ranks)),
            "mean_final_output_rank": float(np.mean(final_ranks)),
            "output_suppression": _mean_effect(
                [record["output_suppression"] for record in records]
            ),
        }
        if all(record["ablation"]["status"] == "OK" for record in records):
            base.update(
                {
                    "status": "OK",
                    "ablation": _mean_effect(
                        [record["ablation"] for record in records]
                    ),
                }
            )
        else:
            base.update(
                {
                    "status": "INCOMPLETE",
                    "variant_statuses": [
                        record["ablation"]["status"] for record in records
                    ],
                }
            )
        output[key] = base
    return output


def _primary_control_violations(
    contexts: Sequence[Mapping[str, Any]],
    *,
    output_top_k: int,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for context in contexts:
        record = context["record"]
        variant = record["counterbalance"]["variant"]
        for reading in ("r1", "r2"):
            rank = int(record["clean"]["output_tracked"][f"{reading}_concept"]["rank"])
            if rank <= output_top_k:
                violations.append(
                    {
                        "variant": variant,
                        "reading": reading,
                        "rank": rank,
                        "reason": f"concept token appears in clean output top {output_top_k}",
                    }
                )
    return violations


def _measure_primary_variant(
    bundle: Any,
    context: Mapping[str, Any],
    committed_reading: str,
    alternate_reading: str,
    committed_directions: Mapping[int, torch.Tensor],
    alternate_directions: Mapping[int, torch.Tensor],
    committed_concept_token_id: int,
    alternate_concept_token_id: int,
    layers: Sequence[int],
    *,
    max_swap_condition: float,
) -> dict[str, Any]:
    probe = context["prepared_probe"]
    counterbalance = probe["counterbalance"]
    reading_to_label = counterbalance["reading_to_label"]
    target_label = reading_to_label[committed_reading]
    foil_label = reading_to_label[alternate_reading]
    target_token_id = int(probe["continuations"][target_label]["token_id"])
    foil_token_id = int(probe["continuations"][foil_label]["token_id"])
    clean_metric = _committed_metric(
        context["clean_logits"],
        probe,
        committed_reading,
    )

    shared_attribution = context["shared_attribution"]
    attribution = shared_attribution["direction_results"]["committed_concept"]
    alternate_attribution = shared_attribution["direction_results"]["alternate_concept"]
    aggregate = attribution["aggregate"]
    ablated_logits = forward_logits(
        bundle.hf_model,
        context["input_ids"],
        attention_mask=context["attention_mask"],
        blocks=bundle.lens_model.layers,
        edits=ablation_edits(committed_directions),
    )
    swapped_logits = forward_logits(
        bundle.hf_model,
        context["input_ids"],
        attention_mask=context["attention_mask"],
        blocks=bundle.lens_model.layers,
        edits=clamped_swap_edits(
            context["clean_residuals"],
            committed_directions,
            alternate_directions,
            max_condition=max_swap_condition,
        ),
    )
    ablated_metric = _committed_metric(ablated_logits, probe, committed_reading)
    swapped_metric = _committed_metric(swapped_logits, probe, committed_reading)

    suppression_records: dict[str, Any] = {}
    for role, token_id in (
        ("committed_concept", committed_concept_token_id),
        ("alternate_concept", alternate_concept_token_id),
    ):
        suppressed_logits = suppress_output_token(context["clean_logits"], token_id)
        suppressed_metric = _committed_metric(
            suppressed_logits,
            probe,
            committed_reading,
        )
        suppression_records[role] = {
            "token_id": int(token_id),
            "clean_output_logit": float(context["clean_logits"][0, -1, token_id]),
            "suppressed_output_logit": float(suppressed_logits[0, -1, token_id]),
            **_effect_record(clean_metric, suppressed_metric),
        }

    return {
        "counterbalance": dict(counterbalance),
        "clean_committed_margin": clean_metric,
        "behavior_token_ids": {
            "committed_target": target_token_id,
            "alternate_foil": foil_token_id,
            "committed_target_label": target_label,
            "alternate_foil_label": foil_label,
        },
        "attribution": {
            "shared_forward_backward_count": int(
                shared_attribution["n_forward_backward"]
            ),
            "shared_direction_names": sorted(shared_attribution["direction_results"]),
            "clean_metric": float(shared_attribution["metric"]),
            "clean_metric_error": float(shared_attribution["metric"] - clean_metric),
            "raw_write_by_layer_position": {
                str(layer): attribution["write"][layer].tolist() for layer in layers
            },
            "raw_read_by_layer_position": {
                str(layer): attribution["read"][layer].tolist() for layer in layers
            },
            "predicted_delta_by_layer": {
                str(layer): attribution["predicted_delta_by_layer"][layer]
                for layer in layers
            },
            "helper_predicted_delta": attribution["predicted_delta"],
            "independent_read_field": "aggregate.read_abs_mean",
            "diagnostic_only_fields": [
                "aggregate.support_oriented_read",
                "aggregate.first_order_predicted_delta",
                "aggregate.first_order_predicted_positive_damage",
            ],
            "aggregate": aggregate,
            "alternate_concept": {
                "raw_write_by_layer_position": {
                    str(layer): alternate_attribution["write"][layer].tolist()
                    for layer in layers
                },
                "raw_read_by_layer_position": {
                    str(layer): alternate_attribution["read"][layer].tolist()
                    for layer in layers
                },
                "predicted_delta_by_layer": {
                    str(layer): alternate_attribution["predicted_delta_by_layer"][layer]
                    for layer in layers
                },
                "helper_predicted_delta": alternate_attribution["predicted_delta"],
                "independent_read_field": "aggregate.read_abs_mean",
                "diagnostic_only_fields": [
                    "aggregate.support_oriented_read",
                    "aggregate.first_order_predicted_delta",
                    "aggregate.first_order_predicted_positive_damage",
                ],
                "aggregate": alternate_attribution["aggregate"],
            },
        },
        "ablation": {
            "scope": "all workspace layers and all prompt positions",
            **_effect_record(clean_metric, ablated_metric),
        },
        "clean_clamped_swap": {
            "scope": "all workspace layers and all prompt positions",
            "strength": 1.0,
            "max_condition": float(max_swap_condition),
            **_effect_record(clean_metric, swapped_metric),
        },
        "output_suppression": {
            **suppression_records,
            "scope": "final vocabulary logits only; no internal activation edited",
            "interpretation": OUTPUT_SUPPRESSION_INTERPRETATION,
        },
    }


def measure_ambiguity_items(
    bundle: Any,
    lens: Any,
    items: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    raw_direction_bank: Mapping[int, Mapping[int, torch.Tensor]],
    rms_folded_bank: Mapping[int, Mapping[int, torch.Tensor]],
    meta_resolution: Mapping[str, Any],
    *,
    max_length: int = 256,
    output_top_k: int = 10,
    lens_top_k: int = 10,
    max_swap_condition: float = 1e4,
    tie_tolerance: float = 0.0,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Measure every frozen item while retaining ties, controls, and failures."""

    _require(len(items) == 120, "Ambiguity phase requires all 120 frozen items")
    layer_list = sorted(set(int(layer) for layer in layers))
    _require(bool(layer_list), "At least one workspace layer is required")
    _require(output_top_k > 0 and lens_top_k > 0, "top_k values must be positive")
    meta_tokens = list(meta_resolution["selected"])
    rows: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        base: dict[str, Any] = {
            "row_index": item_index,
            "id": item["id"],
            "category": item["category"],
            "sentence": item["sentence"],
            "readings": item["readings"],
            "concepts": item["concepts"],
            "workspace_layers": layer_list,
            "direction_method": PRIMARY_DIRECTION_METHOD,
            "direction_policy": DIRECTION_POLICY,
        }
        contexts: list[dict[str, Any]] = []
        try:
            contexts = [
                _clean_variant_context(
                    bundle,
                    lens,
                    item,
                    probe,
                    meta_tokens,
                    layer_list,
                    max_length=max_length,
                    output_top_k=output_top_k,
                    lens_top_k=lens_top_k,
                )
                for probe in item["probe_variants"]
            ]
            margin_records = [context["record"]["clean"] for context in contexts]
            commitment = infer_counterbalanced_commitment(
                margin_records,
                tie_tolerance=tie_tolerance,
            )
            base["commitment"] = commitment
            base["clean_variants"] = [context["record"] for context in contexts]
            if commitment["status"] != "COMMITTED":
                base["measurement_status"] = "TIE"
                rows.append(base)
                continue

            committed_reading = str(commitment["committed_reading"])
            alternate_reading = str(commitment["alternate_reading"])
            for context in contexts:
                r1_margin = float(context["record"]["clean"]["r1_minus_r2_margin"])
                context["record"]["clean"]["committed_reading"] = committed_reading
                context["record"]["clean"]["committed_margin"] = (
                    r1_margin if committed_reading == "r1" else -r1_margin
                )

            committed_token_id = int(item["concepts"][committed_reading]["token_id"])
            alternate_token_id = int(item["concepts"][alternate_reading]["token_id"])
            committed_directions = raw_direction_bank[committed_token_id]
            alternate_directions = raw_direction_bank[alternate_token_id]
            pair_diagnostics = _direction_pair_diagnostics(
                committed_directions,
                alternate_directions,
            )
            base["direction_pair"] = pair_diagnostics
            base["rms_gain_fold_sensitivity"] = {
                "committed": _rms_sensitivity_record(
                    committed_directions,
                    rms_folded_bank[committed_token_id],
                ),
                "alternate": _rms_sensitivity_record(
                    alternate_directions,
                    rms_folded_bank[alternate_token_id],
                ),
            }

            meta_variant_records: list[dict[str, Any]] = []
            for context in contexts:
                reading_to_label = context["prepared_probe"]["counterbalance"][
                    "reading_to_label"
                ]
                target_label = reading_to_label[committed_reading]
                foil_label = reading_to_label[alternate_reading]
                target_token_id = int(
                    context["prepared_probe"]["continuations"][target_label]["token_id"]
                )
                foil_token_id = int(
                    context["prepared_probe"]["continuations"][foil_label]["token_id"]
                )
                shared_directions = {
                    "committed_concept": committed_directions,
                    "alternate_concept": alternate_directions,
                    **{
                        f"meta::{candidate['key']}": raw_direction_bank[
                            int(candidate["token_id"])
                        ]
                        for candidate in meta_tokens
                    },
                }
                context["shared_attribution"] = shared_direction_attribution(
                    bundle.hf_model,
                    bundle.lens_model.layers,
                    context["input_ids"],
                    shared_directions,
                    target_token_id=target_token_id,
                    foil_token_id=foil_token_id,
                    attention_mask=context["attention_mask"],
                )
                context["record"]["shared_attribution"] = {
                    "n_forward_backward": int(
                        context["shared_attribution"]["n_forward_backward"]
                    ),
                    "metric": float(context["shared_attribution"]["metric"]),
                    "clean_metric_error": float(
                        context["shared_attribution"]["metric"]
                        - context["record"]["clean"]["committed_margin"]
                    ),
                    "layers": list(context["shared_attribution"]["layers"]),
                    "positions": list(context["shared_attribution"]["positions"]),
                    "direction_names": sorted(
                        context["shared_attribution"]["direction_results"]
                    ),
                    "note": (
                        "One shared behavior-gradient backward projected onto "
                        "the committed direction and all preregistered meta directions."
                    ),
                }
                meta_record = _meta_variant_measurements(
                    bundle,
                    context,
                    meta_tokens,
                    raw_direction_bank,
                    committed_reading,
                    layer_list,
                    output_top_k=output_top_k,
                )
                context["record"]["meta_tokens"] = meta_record
                meta_variant_records.append(context["record"])
            base["meta_counterbalanced"] = _aggregate_meta_variants(
                meta_variant_records
            )

            violations = _primary_control_violations(
                contexts,
                output_top_k=output_top_k,
            )
            if violations:
                base.update(
                    {
                        "measurement_status": "CONTROL_REJECTED",
                        "control_rejections": violations,
                        "clean_variants": [context["record"] for context in contexts],
                    }
                )
                rows.append(base)
                continue
            if pair_diagnostics["max_gram_condition"] > max_swap_condition:
                raise ValueError(
                    "Direction pair is too ill-conditioned for clean-clamped swap: "
                    f"{pair_diagnostics['max_gram_condition']:.6g}"
                )

            primary_variants: list[dict[str, Any]] = []
            for context in contexts:
                primary = _measure_primary_variant(
                    bundle,
                    context,
                    committed_reading,
                    alternate_reading,
                    committed_directions,
                    alternate_directions,
                    committed_token_id,
                    alternate_token_id,
                    layer_list,
                    max_swap_condition=max_swap_condition,
                )
                context["record"].update(primary)
                primary_variants.append(context["record"])
            base.update(
                {
                    "measurement_status": "OK",
                    "raw_variants": primary_variants,
                    "counterbalanced": aggregate_counterbalanced_variants(
                        primary_variants
                    ),
                }
            )
            base.pop("clean_variants", None)
        except Exception as error:  # all frozen cases remain visible
            base.update(
                {
                    "measurement_status": "ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if contexts:
                base["clean_variants"] = [
                    context.get("record", {}) for context in contexts
                ]
            rows.append(base)
            if fail_fast:
                raise
            continue
        rows.append(base)
    return rows


def _safe_bootstrap(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    try:
        result = bootstrap_statistic(
            [values],
            lambda array: float(np.mean(array)),
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            "status": "NOT_ESTIMABLE",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {"status": "ESTIMATED", **result}


def _safe_pearson(
    first: Sequence[float],
    second: Sequence[float],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    try:
        result = pearson_with_ci(
            first,
            second,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            "status": "NOT_ESTIMABLE",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {"status": "ESTIMATED", **result}


def _safe_partial_correlation(
    outcome: Sequence[float],
    predictor: Sequence[float],
    control: Sequence[float],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    try:
        result = partial_correlation_with_ci(
            outcome,
            predictor,
            control,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            "status": "NOT_ESTIMABLE",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {"status": "ESTIMATED", **result}


def _p3_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    measured = [row for row in rows if row.get("measurement_status") == "OK"]
    clean = [
        float(row["counterbalanced"]["clean_committed_margin"]) for row in measured
    ]
    swap_edited = [
        float(row["counterbalanced"]["clean_clamped_swap"]["edited_committed_margin"])
        for row in measured
    ]
    swap_flips = [
        float(
            row["counterbalanced"]["clean_clamped_swap"]["flips_committed_mean_margin"]
        )
        for row in measured
    ]
    robust_swap_flips = [
        float(
            row["counterbalanced"]["clean_clamped_swap"]["counterbalance_robust_flip"]
        )
        for row in measured
    ]
    ablation_damage = [
        float(row["counterbalanced"]["ablation"]["positive_damage"]) for row in measured
    ]
    suppression_damage = [
        float(
            row["counterbalanced"]["output_suppression"]["committed_concept"][
                "positive_damage"
            ]
        )
        for row in measured
    ]
    damage_gap = [
        float(row["counterbalanced"]["internal_minus_suppression_positive_damage"])
        for row in measured
    ]
    ablation_exceeds = [
        float(row["counterbalanced"]["ablation_damage_exceeds_suppression"])
        for row in measured
    ]
    agreement = [
        float(row["commitment"]["counterbalance_agreement"]) for row in measured
    ]
    committed_write = [
        float(row["counterbalanced"]["attribution"]["write_abs_mean"])
        for row in measured
    ]
    committed_read = [
        float(row["counterbalanced"]["attribution"]["read_abs_mean"])
        for row in measured
    ]
    alternate_write = [
        float(
            row["counterbalanced"]["attribution"]["alternate_concept"]["write_abs_mean"]
        )
        for row in measured
    ]
    alternate_read = [
        float(
            row["counterbalanced"]["attribution"]["alternate_concept"]["read_abs_mean"]
        )
        for row in measured
    ]
    values = {
        "committed_concept_write_abs_mean": committed_write,
        "alternate_concept_write_abs_mean": alternate_write,
        "committed_concept_read_abs_mean": committed_read,
        "alternate_concept_read_abs_mean": alternate_read,
        "clean_committed_margin": clean,
        "swap_edited_committed_margin": swap_edited,
        "swap_flip_rate": swap_flips,
        "counterbalance_robust_swap_flip_rate": robust_swap_flips,
        "internal_ablation_positive_damage": ablation_damage,
        "output_suppression_positive_damage": suppression_damage,
        "internal_minus_suppression_damage": damage_gap,
        "ablation_exceeds_suppression_rate": ablation_exceeds,
        "counterbalance_agreement_rate": agreement,
    }
    binary_statistics = {
        "swap_flip_rate",
        "counterbalance_robust_swap_flip_rate",
        "ablation_exceeds_suppression_rate",
        "counterbalance_agreement_rate",
    }
    summaries = {}
    for index, (name, vector) in enumerate(values.items()):
        summaries[name] = (
            binomial_rate_with_ci(vector, confidence=confidence)
            if name in binary_statistics
            else _safe_bootstrap(
                vector,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed + index,
            )
        )
    return {
        "n_total_rows": len(rows),
        "n": len(measured),
        "status_counts": dict(
            Counter(str(row.get("measurement_status")) for row in rows)
        ),
        "item_ids": [str(row["id"]) for row in measured],
        "statistics": summaries,
        "raw_item_values": {name: vector for name, vector in values.items()},
    }


def analyze_p3(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = SEED,
) -> dict[str, Any]:
    """Test swap flipping and internal-ablation-vs-suppression P3 criteria."""

    overall = _p3_summary(
        rows,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    categories = sorted({str(row["category"]) for row in rows})
    by_category = {
        category: _p3_summary(
            [row for row in rows if row.get("category") == category],
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + 100 * (index + 1),
        )
        for index, category in enumerate(categories)
    }
    swap = overall["statistics"]["swap_flip_rate"]
    gap = overall["statistics"]["internal_minus_suppression_damage"]
    if swap.get("status") == "ESTIMATED" and gap.get("status") == "ESTIMATED":
        supported = swap["ci_low"] > 0.5 and gap["ci_low"] > 0.0
        point_supported = swap["estimate"] > 0.5 and gap["estimate"] > 0.0
        verdict = (
            "supported" if supported else ("mixed" if point_supported else "refuted")
        )
    else:
        verdict = "not_estimable"
    status_counts = Counter(str(row.get("measurement_status")) for row in rows)
    return {
        "prediction": "P3",
        "analysis_role": "diagnostic_upstream_strict_g2_failed",
        "upstream_gate_context": UPSTREAM_GATE_CONTEXT,
        "direction_method": PRIMARY_DIRECTION_METHOD,
        "commitment_rule": (
            "sign of mean original+mirrored clean r1-vs-r2 margin; no gold selection"
        ),
        "swap_success_rule": "counterbalanced committed margin after swap < 0",
        "counterbalance_robust_swap_rule": (
            "edited committed margin after swap < 0 in both original and mirrored probes"
        ),
        "confound_control_rule": (
            "internal ablation positive damage minus final concept-token suppression "
            "positive damage > 0"
        ),
        "output_suppression_interpretation": {
            **OUTPUT_SUPPRESSION_INTERPRETATION,
            "observed_all_exact_zero": bool(
                overall["n"] > 0
                and all(
                    value == 0.0
                    for value in overall["raw_item_values"][
                        "output_suppression_positive_damage"
                    ]
                )
            ),
            "observed_damage_gap_equals_ablation": bool(
                overall["n"] > 0
                and all(
                    gap == ablation
                    for gap, ablation in zip(
                        overall["raw_item_values"]["internal_minus_suppression_damage"],
                        overall["raw_item_values"]["internal_ablation_positive_damage"],
                        strict=True,
                    )
                )
            ),
        },
        "verdict_rule": (
            "supported iff the 95% CI lower bound for swap-flip rate exceeds "
            "0.5 and the CI lower bound for the damage gap exceeds 0; mixed "
            "iff both point criteria hold; otherwise refuted"
        ),
        "verdict": verdict,
        "n_frozen_items": len(rows),
        "status_counts": dict(status_counts),
        "overall": overall,
        "by_category": by_category,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
        "limitation": J_LENS_ONLY_LIMITATION,
    }


def analyze_meta_tokens(
    rows: Sequence[Mapping[str, Any]],
    meta_resolution: Mapping[str, Any],
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = SEED,
) -> dict[str, Any]:
    """Describe preregistered meta-token WRITE, ranks, and real ablation effects."""

    by_candidate: dict[str, Any] = {}
    candidate_mean_write: list[float] = []
    candidate_mean_read: list[float] = []
    candidate_mean_damage: list[float] = []
    pooled_write: list[float] = []
    pooled_read: list[float] = []
    pooled_damage: list[float] = []
    for candidate_index, candidate in enumerate(meta_resolution["selected"]):
        key = str(candidate["key"])
        candidate_rows = [
            row["meta_counterbalanced"][key]
            for row in rows
            if key in row.get("meta_counterbalanced", {})
            and row["meta_counterbalanced"][key].get("status") == "OK"
        ]
        write = [float(record["mean_write_abs"]) for record in candidate_rows]
        read = [float(record["mean_independent_read_abs"]) for record in candidate_rows]
        predicted_damage = [
            float(record["mean_first_order_predicted_positive_damage"])
            for record in candidate_rows
        ]
        lens_rank = [float(record["mean_lens_rank"]) for record in candidate_rows]
        final_rank = [
            float(record["mean_final_output_rank"]) for record in candidate_rows
        ]
        damage = [
            float(record["ablation"]["positive_damage"]) for record in candidate_rows
        ]
        if candidate_rows:
            candidate_mean_write.append(float(np.mean(write)))
            candidate_mean_read.append(float(np.mean(read)))
            candidate_mean_damage.append(float(np.mean(damage)))
            pooled_write.extend(write)
            pooled_read.extend(read)
            pooled_damage.extend(damage)
        base_seed = seed + 1000 * candidate_index
        by_candidate[key] = {
            "candidate": dict(candidate),
            "n_complete_items": len(candidate_rows),
            "mean_abs_write": _safe_bootstrap(
                write,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 1,
            ),
            "mean_independent_read_abs": _safe_bootstrap(
                read,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 2,
            ),
            "mean_lens_rank": _safe_bootstrap(
                lens_rank,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 3,
            ),
            "mean_final_output_rank": _safe_bootstrap(
                final_rank,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 4,
            ),
            "mean_ablation_positive_damage": _safe_bootstrap(
                damage,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 5,
            ),
            "read_vs_ablation_damage": _safe_pearson(
                read,
                damage,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 6,
            ),
            "partial_causal_read_given_write": _safe_partial_correlation(
                damage,
                read,
                write,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 7,
            ),
            "write_vs_ablation_damage": _safe_pearson(
                write,
                damage,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 8,
            ),
            "first_order_prediction_vs_ablation_damage": _safe_pearson(
                predicted_damage,
                damage,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=base_seed + 9,
            ),
            "diagnostic_only_fields": [
                "write_vs_ablation_damage",
                "first_order_prediction_vs_ablation_damage",
            ],
        }
    across_candidates = {
        "n_candidates": len(candidate_mean_read),
        "read_vs_ablation_damage": _safe_pearson(
            candidate_mean_read,
            candidate_mean_damage,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + 90_001,
        ),
        "partial_causal_read_given_write": _safe_partial_correlation(
            candidate_mean_damage,
            candidate_mean_read,
            candidate_mean_write,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + 90_002,
        ),
        "unit_of_analysis": "preregistered meta-token candidate mean across items",
    }
    pooled = {
        "n_item_candidate_observations": len(pooled_read),
        "read_vs_ablation_damage": _safe_pearson(
            pooled_read,
            pooled_damage,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + 91_001,
        ),
        "partial_causal_read_given_write": _safe_partial_correlation(
            pooled_damage,
            pooled_read,
            pooled_write,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + 91_002,
        ),
        "unit_of_analysis": "item × candidate observation",
        "warning": (
            "Observation-level bootstrap is descriptive and does not model "
            "within-item or within-candidate dependence."
        ),
    }
    return {
        "analysis_role": "diagnostic_nonconfirmatory",
        "candidate_selection_uses_outcomes": False,
        "primary_diagnostic_question": (
            "Does independent meta-token READ (mean abs grad(M) dot v) predict "
            "real meta-token ablation damage?"
        ),
        "primary_association": "by_candidate.<key>.read_vs_ablation_damage",
        "adjusted_association": ("by_candidate.<key>.partial_causal_read_given_write"),
        "selection": dict(meta_resolution),
        "interpretation_warning": (
            "These predeclared meta-token results are descriptive diagnostics. "
            "No candidate is selected or promoted based on WRITE, rank, or "
            "ablation outcomes, and they are not part of the confirmatory P3 verdict."
        ),
        "by_candidate": by_candidate,
        "across_candidate_means": across_candidates,
        "pooled_item_candidate_diagnostic": pooled,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
    }


def plot_f8(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[plt.Figure, np.ndarray]:
    """F8: committed-vs-alternate and meta-token WRITE×READ diagnostics."""

    measured = [row for row in rows if row.get("measurement_status") == "OK"]
    _require(len(measured) >= 2, "F8 needs at least two completed ambiguity items")
    set_style()
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    palette = {
        "lexical_ambiguity": "#4477AA",
        "pp_attachment": "#EE6677",
        "garden_path": "#228833",
        "ambiguous_pronoun": "#CCBB44",
    }
    categories = sorted({str(row["category"]) for row in measured})
    for category in categories:
        selected = [row for row in measured if row["category"] == category]
        write = [
            float(row["counterbalanced"]["attribution"]["write_abs_mean"])
            for row in selected
        ]
        read = [
            float(row["counterbalanced"]["attribution"]["read_abs_mean"])
            for row in selected
        ]
        alternate_write = [
            float(
                row["counterbalanced"]["attribution"]["alternate_concept"][
                    "write_abs_mean"
                ]
            )
            for row in selected
        ]
        alternate_read = [
            float(
                row["counterbalanced"]["attribution"]["alternate_concept"][
                    "read_abs_mean"
                ]
            )
            for row in selected
        ]
        predicted = [
            float(
                row["counterbalanced"]["attribution"][
                    "first_order_predicted_positive_damage"
                ]
            )
            for row in selected
        ]
        measured_damage = [
            float(row["counterbalanced"]["ablation"]["positive_damage"])
            for row in selected
        ]
        color = palette.get(category, "0.4")
        for committed_x, committed_y, alternate_x, alternate_y in zip(
            write,
            read,
            alternate_write,
            alternate_read,
            strict=True,
        ):
            axes[0].plot(
                [alternate_x, committed_x],
                [alternate_y, committed_y],
                color=color,
                alpha=0.12,
                linewidth=0.8,
                zorder=1,
            )
        axes[0].scatter(
            write,
            read,
            s=34,
            alpha=0.78,
            label=f"{category} — committed",
            color=color,
            marker="o",
            zorder=3,
        )
        axes[0].scatter(
            alternate_write,
            alternate_read,
            s=30,
            alpha=0.65,
            label=f"{category} — alternate",
            color=color,
            marker="x",
            zorder=2,
        )
        axes[2].scatter(
            predicted,
            measured_damage,
            s=34,
            alpha=0.75,
            label=category,
            color=color,
        )
    axes[0].set(
        xlabel="Concept WRITE strength (mean |projection|)",
        ylabel="Independent attribution READ (mean |grad(M) · direction|)",
        title="Committed circles vs alternate crosses",
    )

    candidate_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in measured:
        for key, record in row.get("meta_counterbalanced", {}).items():
            if record.get("status") == "OK":
                candidate_rows[str(key)].append(record)
    meta_write: list[float] = []
    meta_read: list[float] = []
    meta_damage: list[float] = []
    meta_labels: list[str] = []
    for key, records in sorted(candidate_rows.items()):
        meta_labels.append(key)
        meta_write.append(
            float(np.mean([record["mean_write_abs"] for record in records]))
        )
        meta_read.append(
            float(np.mean([record["mean_independent_read_abs"] for record in records]))
        )
        meta_damage.append(
            float(
                np.mean([record["ablation"]["positive_damage"] for record in records])
            )
        )
    if meta_labels:
        meta_scatter = axes[1].scatter(
            meta_write,
            meta_read,
            c=meta_damage,
            cmap="coolwarm",
            s=75,
            edgecolor="0.25",
        )
        for label, x_value, y_value in zip(
            meta_labels,
            meta_write,
            meta_read,
            strict=True,
        ):
            axes[1].annotate(
                label,
                (x_value, y_value),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
        colorbar = figure.colorbar(meta_scatter, ax=axes[1], pad=0.02)
        colorbar.set_label("mean real ablation damage")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No complete meta-token rows",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    axes[1].set(
        xlabel="Meta-token WRITE strength",
        ylabel="Meta-token independent READ",
        title="Preregistered interpretive meta-tokens",
    )
    all_predicted = np.asarray(
        [
            float(
                row["counterbalanced"]["attribution"][
                    "first_order_predicted_positive_damage"
                ]
            )
            for row in measured
        ]
    )
    all_measured = np.asarray(
        [
            float(row["counterbalanced"]["ablation"]["positive_damage"])
            for row in measured
        ]
    )
    lower = float(min(all_predicted.min(), all_measured.min()))
    upper = float(max(all_predicted.max(), all_measured.max()))
    axes[2].plot([lower, upper], [lower, upper], "--", color="0.45", linewidth=1)
    axes[2].set(
        xlabel=r"Diagnostic first-order $\sum WRITE\,READ$ predicted damage",
        ylabel="Real all-band ablation damage",
        title="Diagnostic product vs. real intervention",
    )
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    figure.suptitle(
        "F8 — diagnostic ambiguity results (upstream strict Qwen-7B G2 failed)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure, axes


def _direction_sensitivity_table(
    raw_bank: Mapping[int, Mapping[int, torch.Tensor]],
    folded_bank: Mapping[int, Mapping[int, torch.Tensor]],
    token_labels: Mapping[int, Sequence[str]],
) -> dict[str, Any]:
    _require(
        set(raw_bank) == set(folded_bank), "Direction banks cover different tokens"
    )
    return {
        str(token_id): {
            "token_id": int(token_id),
            "labels": sorted(set(token_labels.get(token_id, []))),
            **_rms_sensitivity_record(raw_bank[token_id], folded_bank[token_id]),
        }
        for token_id in sorted(raw_bank)
    }


def run_qwen_ambiguity_phase(
    *,
    spec_path: str | Path = DEFAULT_AMBIGUITY_SPEC,
    output_path: str | Path = DEFAULT_OUTPUT,
    figure_path: str | Path = DEFAULT_FIGURE,
    device: str | torch.device = "cuda",
    direction_compute_device: str | torch.device | None = None,
    max_length: int = 256,
    output_top_k: int = 10,
    lens_top_k: int = 10,
    max_swap_condition: float = 1e4,
    tie_tolerance: float = 0.0,
    n_bootstrap: int = 5000,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run and persist the complete 120-item Qwen ambiguity flagship phase."""

    set_seed(SEED)
    compute_device = direction_compute_device or device
    bundle = load_model(
        MODEL_ID,
        revision=MODEL_REVISION,
        device=device,
    )
    try:
        _require(bundle.revision == MODEL_REVISION, "Loaded model revision drifted")
        lens = load_published_lens(MODEL_ID)
        layers = workspace_layers(bundle.lens_model.n_layers, lens.source_layers)
        items = load_tokenized_ambiguity_items(bundle.tokenizer, spec_path)
        _require(len(items) == 120, "Frozen ambiguity item count changed")
        meta_resolution = resolve_meta_token_candidates(bundle.tokenizer)
        _require(
            meta_resolution["n_selected"] > 0,
            "No preregistered meta-token candidate is exactly one token",
        )

        token_labels: dict[int, list[str]] = defaultdict(list)
        for item in items:
            for reading_id in ("r1", "r2"):
                concept = item["concepts"][reading_id]
                token_labels[int(concept["token_id"])].append(str(concept["label"]))
        for candidate in meta_resolution["selected"]:
            token_labels[int(candidate["token_id"])].append(str(candidate["key"]))
        token_ids = sorted(token_labels)
        raw_bank = jlens_direction_bank(
            lens,
            bundle.lens_model,
            token_ids,
            layers,
            fold_rms_gain=False,
            compute_device=compute_device,
            output_device=device,
        )
        folded_bank = jlens_direction_bank(
            lens,
            bundle.lens_model,
            token_ids,
            layers,
            fold_rms_gain=True,
            compute_device=compute_device,
            output_device=device,
        )
        rows = measure_ambiguity_items(
            bundle,
            lens,
            items,
            layers,
            raw_bank,
            folded_bank,
            meta_resolution,
            max_length=max_length,
            output_top_k=output_top_k,
            lens_top_k=lens_top_k,
            max_swap_condition=max_swap_condition,
            tie_tolerance=tie_tolerance,
            fail_fast=fail_fast,
        )
        p3 = analyze_p3(
            rows,
            n_bootstrap=n_bootstrap,
            seed=SEED,
        )
        meta_analysis = analyze_meta_tokens(
            rows,
            meta_resolution,
            n_bootstrap=n_bootstrap,
            seed=SEED,
        )
        figure, _ = plot_f8(rows)
        saved_figure = save_figure(figure, figure_path)
        plt.close(figure)
        result = {
            "schema_version": SCHEMA_VERSION,
            "analysis_role": "diagnostic_upstream_strict_g2_failed",
            "upstream_gate_context": UPSTREAM_GATE_CONTEXT,
            "output_suppression_control": OUTPUT_SUPPRESSION_INTERPRETATION,
            "seed": SEED,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
            },
            "lens": PUBLISHED_LENSES[MODEL_ID],
            "spec_path": str(Path(spec_path).resolve()),
            "n_frozen_items": len(items),
            "workspace_layers": layers,
            "direction_policy": DIRECTION_POLICY,
            "direction_sensitivity": _direction_sensitivity_table(
                raw_bank,
                folded_bank,
                token_labels,
            ),
            "meta_token_preregistration": meta_resolution,
            "rows": rows,
            "p3": p3,
            "meta_token_diagnostics": meta_analysis,
            "figures": {"f8": str(saved_figure)},
            "run_configuration": {
                "max_length": max_length,
                "output_top_k": output_top_k,
                "lens_top_k": lens_top_k,
                "max_swap_condition": max_swap_condition,
                "tie_tolerance": tie_tolerance,
                "n_bootstrap": n_bootstrap,
                "fail_fast": fail_fast,
            },
        }
        json_result = _json_ready(result)
        save_json(output_path, json_result)
        return json_result
    finally:
        release_model(bundle)

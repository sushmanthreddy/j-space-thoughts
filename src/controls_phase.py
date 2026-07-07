"""Mandatory specificity, capability, narration, and identity-J controls.

This module is deliberately split into pure, CPU-testable control definitions and
an orchestration entry point that executes them on an already loaded model.  The
definitions are frozen in code before any behavioral effects are observed:

* random null directions are keyed only by the project seed, item, draw, and layer;
* absent controls use the first preregistered tokens meeting a J-Lens-rank rule;
* capability texts and narration continuation-token margins are fixed constants;
* every effect uses ``delta = edited - clean``.

No generated continuation is inspected or selected in this phase.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data_gen import continuation_token_id
from src.interventions import (
    ablation_edits,
    clamped_swap_edits,
    forward_logits,
    suppress_output_token,
)
from src.jlens_iface import (
    jlens_direction_bank,
    token_rank,
    unembedding_weight,
)
from src.metrics import (
    bootstrap_statistic,
    logit_difference,
    pearson_with_ci,
    save_json,
    signed_causal_delta,
)
from src.model_utils import capture_residuals, single_token_id
from src.plotting import save_figure, set_style
from src.read_scores import attribution_read


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
DEFAULT_RANDOM_DRAWS = 16
DEFAULT_ABSENT_MIN_RANK = 1_000
KNOWN_NARRATION_HIGH_WRITE_MAX_RANK = 10
KNOWN_NARRATION_LOW_CAUSAL_MAX_GAP = 0.5
KNOWN_NARRATION_CLEAN_MARGIN_MIN = 0.0
KNOWN_NARRATION_MIN_REPRODUCED = 6
DEFAULT_NARRATION_SOURCE = (
    Path.home() / "deps" / "jacobian-lens" / "data" / "experiments"
    / "selectivity-language.json"
)

# Order is part of the selection rule.  Every surface is one token for the pinned
# Qwen2.5 tokenizer.  A candidate that is a target/foil or has strong prompt WRITE
# is skipped; the first remaining weak-rank candidate is selected.
ABSENT_CONCEPT_SURFACES_V1 = (
    " accordion",
    " volcano",
    " telescope",
    " bicycle",
    " submarine",
    " glacier",
    " pineapple",
    " lantern",
)

# These short texts are not present in the MD cue manifest or two-hop prompts.  The
# metric is teacher-forced next-token NLL over all tokens after the first, not a
# selectively reported generated span.
CAPABILITY_TEXTS_V1 = (
    {
        "id": "capability_weather",
        "text": "A cold front crossed the coast overnight and brought steady rain.",
    },
    {
        "id": "capability_cooking",
        "text": "The cook rinsed the rice, measured the water, and covered the pot.",
    },
    {
        "id": "capability_arithmetic",
        "text": "Seven boxes with four pencils in each box contain twenty-eight pencils.",
    },
    {
        "id": "capability_science",
        "text": "Liquid water expands when it freezes into ice.",
    },
    {
        "id": "capability_history",
        "text": "The archive preserved letters, maps, and photographs from the expedition.",
    },
    {
        "id": "capability_dialogue",
        "text": "Mira asked whether the train was late, and Jonah checked the clock.",
    },
    {
        "id": "capability_instruction",
        "text": "Before replacing the filter, switch off the machine and unplug it.",
    },
    {
        "id": "capability_nature",
        "text": "At dusk, small waves reflected the orange light across the lake.",
    },
)

# Fixed first-token margins for the upstream automatic-continuation prompts.  The
# target is a grammatical start in the passage language and the foil is its fixed
# English counterpart.  These were tokenizer-audited before any intervention run.
KNOWN_NARRATION_MARGINS_V1: dict[str, dict[str, str]] = {
    "fr1": {"target": "Elle", "foil": "She"},
    "fr2": {"target": "Les", "foil": "The"},
    "de1": {"target": "Er", "foil": "He"},
    "de2": {"target": "Die", "foil": "The"},
    "es1": {"target": "Ella", "foil": "She"},
    "es2": {"target": "Los", "foil": "The"},
    "it1": {"target": "Lei", "foil": "She"},
    "it2": {"target": "La", "foil": "The"},
}


DirectionBank = Mapping[int, torch.Tensor]
Item = Mapping[str, Any]


def _item_name(item: Item) -> str:
    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Every control item requires a nonempty string name")
    return name


def _row_item_name(row: Mapping[str, Any], index: int) -> str:
    name = row.get("name")
    if not isinstance(name, str):
        nested = row.get("item")
        if isinstance(nested, Mapping):
            name = nested.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Core row {index} has no item name")
    return name


def adapt_core_control_rows(
    core_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt notebook-02 rows to the small schema used by this phase.

    ``suppression_delta`` and ``output_suppression_delta`` are the only accepted
    suppression keys.  Requiring an explicit finite value prevents a missing
    negative control from being silently treated as zero.
    """

    adapted: list[dict[str, Any]] = []

    def optional_finite(row: Mapping[str, Any], field: str, name: str) -> float | None:
        if row.get(field) is None:
            return None
        try:
            value = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Core row {name!r} has nonnumeric {field}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Core row {name!r} has nonfinite {field}")
        return value

    for index, row in enumerate(core_rows):
        name = _row_item_name(row, index)
        if "output_suppression_delta" in row:
            suppression = row["output_suppression_delta"]
            suppression_key = "output_suppression_delta"
        elif "suppression_delta" in row:
            suppression = row["suppression_delta"]
            suppression_key = "suppression_delta"
        else:
            raise ValueError(
                f"Core row {name!r} lacks suppression_delta or "
                "output_suppression_delta"
            )
        try:
            suppression_value = float(suppression)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Core row {name!r} has nonnumeric {suppression_key}"
            ) from exc
        if not math.isfinite(suppression_value):
            raise ValueError(
                f"Core row {name!r} has nonfinite {suppression_key}"
            )
        adapted.append(
            {
                "row_index": index,
                "name": name,
                "suppression_delta": suppression_value,
                "suppression_source_key": suppression_key,
                "actual_delta": optional_finite(row, "actual_delta", name),
                "predicted_delta": optional_finite(row, "predicted_delta", name),
                "write_strength": optional_finite(row, "write_strength", name),
            }
        )
    if not adapted:
        raise ValueError("At least one core row is required")
    return adapted


def assert_output_suppression_complete(
    core_rows: Sequence[Mapping[str, Any]],
    *,
    expected_item_names: Sequence[str] | None = None,
    require_actual_delta: bool = False,
) -> list[dict[str, Any]]:
    """Assert a finite output-suppression effect for every core row and item."""

    adapted = adapt_core_control_rows(core_rows)
    if expected_item_names is not None:
        expected = set(expected_item_names)
        observed = {row["name"] for row in adapted}
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing or unexpected:
            raise ValueError(
                "Core/output-suppression coverage differs from eligible items: "
                f"missing={missing}, unexpected={unexpected}"
            )
    if require_actual_delta:
        missing_actual = sorted(
            row["name"] for row in adapted if row["actual_delta"] is None
        )
        if missing_actual:
            raise ValueError(
                "Core rows require finite actual_delta for suppression comparison: "
                f"{missing_actual}"
            )
    return adapted


def assert_structural_concept_output_zero(
    items: Sequence[Item],
    adapted_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assert the exact-zero final concept-logit clamp for the two-hop metric.

    The behavior metric is ``target_logit - foil_logit``.  Clamping a distinct
    concept vocabulary logit therefore leaves the metric exactly unchanged by
    construction.  This validates instrumentation/direct-logit steering only;
    it is not independent evidence about the internal causal intervention.
    """

    row_by_name = {str(row["name"]): row for row in adapted_rows}
    if len(row_by_name) != len(adapted_rows):
        raise ValueError("Structural-zero assertion requires unique core row names")
    audit_rows: list[dict[str, Any]] = []
    for item in items:
        name = _item_name(item)
        if name not in row_by_name:
            raise ValueError(f"Missing output-suppression row for {name!r}")
        concept_id = int(item["concept_token_id"])
        target_id = int(item["target_token_id"])
        foil_id = int(item["foil_token_id"])
        disjoint = concept_id not in {target_id, foil_id}
        if not disjoint:
            raise ValueError(
                "Final concept-token suppression is not structurally separate from "
                f"the target/foil metric for {name!r}: concept={concept_id}, "
                f"target={target_id}, foil={foil_id}"
            )
        delta = float(row_by_name[name]["suppression_delta"])
        exact_zero = delta == 0.0
        if not exact_zero:
            raise ValueError(
                "Final concept-token suppression must be exactly zero when token IDs "
                f"are disjoint for {name!r}; observed delta={delta!r}"
            )
        audit_rows.append(
            {
                "name": name,
                "concept_token_id": concept_id,
                "target_token_id": target_id,
                "foil_token_id": foil_id,
                "concept_disjoint_from_target_and_foil": disjoint,
                "output_suppression_delta": delta,
                "exact_zero": exact_zero,
            }
        )
    return {
        "status": "PASS",
        "classification": (
            "structural-zero instrumentation/direct-logit-steering check; "
            "not additional causal evidence"
        ),
        "operation": "clamp only the final concept-token vocabulary logit",
        "behavior_metric": "target-token logit minus foil-token logit",
        "structural_reason": (
            "concept token IDs are disjoint from both metric token IDs, so the "
            "clamped logit is not an operand of the metric"
        ),
        "n_rows": len(audit_rows),
        "all_token_ids_disjoint": True,
        "all_deltas_exact_zero": True,
        "rows": audit_rows,
    }


def core_output_suppression_comparison(
    adapted_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe internal effects beside the structural final-logit zero check."""

    rows: list[dict[str, Any]] = []
    for row in adapted_rows:
        if row.get("actual_delta") is None:
            raise ValueError("Suppression comparison requires finite actual_delta")
        actual = float(row["actual_delta"])
        suppression = float(row["suppression_delta"])
        rows.append(
            {
                "name": row["name"],
                "internal_ablation_delta": actual,
                "output_suppression_delta": suppression,
                "internal_minus_output_delta": actual - suppression,
                "abs_internal_effect": abs(actual),
                "abs_output_effect": abs(suppression),
                "abs_output_over_internal": (
                    abs(suppression) / abs(actual) if actual != 0 else None
                ),
            }
        )
    return {
        "n": len(rows),
        "median_abs_internal_effect": float(
            np.median([row["abs_internal_effect"] for row in rows])
        ),
        "median_abs_output_effect": float(
            np.median([row["abs_output_effect"] for row in rows])
        ),
        "rows": rows,
    }


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def seeded_random_direction_bank(
    reference_directions: DirectionBank,
    *,
    item_name: str,
    draw_index: int,
    seed: int = SEED,
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    """Create a reproducible independent unit null direction at every layer.

    References must themselves be unit vectors.  Random vectors are generated in
    fp32 on CPU, normalized, and then moved to the reference device.  Thus nulls
    match the norm and all-band layer count without depending on model behavior.
    """

    if not reference_directions:
        raise ValueError("Reference direction bank is empty")
    if draw_index < 0:
        raise ValueError("draw_index must be nonnegative")
    random_bank: dict[int, torch.Tensor] = {}
    layer_seeds: dict[int, int] = {}
    for layer, reference in sorted(reference_directions.items()):
        vector = reference.detach().float()
        if vector.ndim != 1 or not torch.isfinite(vector).all():
            raise ValueError(f"Invalid reference direction at layer {layer}")
        norm = vector.norm()
        if not torch.isclose(norm, torch.ones_like(norm), atol=1e-4, rtol=1e-4):
            raise ValueError(
                f"Reference direction at layer {layer} is not unit norm: {float(norm)}"
            )
        layer_seed = _stable_seed(seed, item_name, draw_index, int(layer))
        generator = torch.Generator(device="cpu").manual_seed(layer_seed)
        random_vector = torch.randn(
            vector.numel(), generator=generator, dtype=torch.float32
        )
        random_vector = F.normalize(random_vector, dim=0)
        random_bank[int(layer)] = random_vector.to(reference.device)
        layer_seeds[int(layer)] = layer_seed
    return random_bank, layer_seeds


def _direction_sha256(direction: torch.Tensor) -> str:
    contiguous = direction.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _rank_values(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        values: list[int] = []
        for nested in value.values():
            values.extend(_rank_values(nested))
        return values
    if isinstance(value, np.ndarray):
        return [int(rank) for rank in value.reshape(-1)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = []
        for nested in value:
            values.extend(_rank_values(nested))
        return values
    return [int(value)]


def select_absent_concept(
    preregistered_token_ids: Sequence[int],
    ranks_by_token: Mapping[int, Any],
    *,
    excluded_token_ids: Sequence[int] = (),
    min_rank: int = DEFAULT_ABSENT_MIN_RANK,
) -> dict[str, Any]:
    """Select the first preregistered token weak at every layer/position.

    Selection sees only token identity, the fixed order, exclusions, and J-Lens
    ranks.  It intentionally has no behavioral-delta argument.  ``min_rank`` is
    one-indexed: larger ranks mean weaker/absent label evidence.
    """

    if min_rank < 2:
        raise ValueError("Absent-concept min_rank must be at least 2")
    if not preregistered_token_ids:
        raise ValueError("Absent-concept candidate list is empty")
    excluded = {int(token_id) for token_id in excluded_token_ids}
    audits: list[dict[str, Any]] = []
    selected: int | None = None
    for order, raw_token_id in enumerate(preregistered_token_ids):
        token_id = int(raw_token_id)
        if token_id in excluded:
            audits.append(
                {
                    "order": order,
                    "token_id": token_id,
                    "status": "excluded_target_or_foil",
                }
            )
            continue
        if token_id not in ranks_by_token:
            raise KeyError(f"Missing preregistered rank data for token {token_id}")
        ranks = _rank_values(ranks_by_token[token_id])
        if not ranks or min(ranks) < 1:
            raise ValueError(f"Invalid J-Lens ranks for token {token_id}: {ranks}")
        strongest_rank = min(ranks)
        qualifies = strongest_rank >= min_rank
        audits.append(
            {
                "order": order,
                "token_id": token_id,
                "status": "qualified" if qualifies else "rejected_too_present",
                "strongest_rank": strongest_rank,
                "weakest_rank": max(ranks),
                "n_rank_observations": len(ranks),
            }
        )
        if qualifies:
            selected = token_id
            break
    if selected is None:
        raise ValueError(
            "No preregistered absent token met the J-Lens rank threshold; "
            "the threshold must not be changed after inspecting behavior deltas"
        )
    return {
        "token_id": selected,
        "excluded_token_ids": sorted(excluded),
        "min_rank_rule": int(min_rank),
        "selection_rule": "first preregistered nonexcluded token with min rank >= threshold",
        "candidate_audit_until_selection": audits,
    }


def audit_absent_pair_selection(
    preregistered_token_ids: Sequence[int],
    ranks_by_token: Mapping[int, Any],
    *,
    excluded_token_ids: Sequence[int] = (),
    min_rank: int = DEFAULT_ABSENT_MIN_RANK,
) -> dict[str, Any]:
    """Audit whether the frozen rank rule supplies one or two absent labels."""

    first = select_absent_concept(
        preregistered_token_ids,
        ranks_by_token,
        excluded_token_ids=excluded_token_ids,
        min_rank=min_rank,
    )
    first_id = int(first["token_id"])
    try:
        second = select_absent_concept(
            preregistered_token_ids,
            ranks_by_token,
            excluded_token_ids=(*excluded_token_ids, first_id),
            min_rank=min_rank,
        )
    except ValueError as error:
        if not str(error).startswith("No preregistered absent token met"):
            raise
        return {
            "status": "ineligible_fewer_than_two_candidates",
            "token_ids": [first_id],
            "first": first,
            "second": None,
            "second_selection_error": str(error),
            "min_rank_rule": int(min_rank),
            "selection_rule": (
                "first two distinct preregistered nonexcluded tokens whose minimum "
                "J-Lens rank across workspace layers and positions meets the threshold"
            ),
            "selection_uses_behavior_outcomes": False,
        }
    return _absent_pair_record(first, second, min_rank=min_rank)


def select_absent_pair(
    preregistered_token_ids: Sequence[int],
    ranks_by_token: Mapping[int, Any],
    *,
    excluded_token_ids: Sequence[int] = (),
    min_rank: int = DEFAULT_ABSENT_MIN_RANK,
) -> dict[str, Any]:
    """Select the first two distinct weak-rank labels without behavior outcomes."""

    audit = audit_absent_pair_selection(
        preregistered_token_ids,
        ranks_by_token,
        excluded_token_ids=excluded_token_ids,
        min_rank=min_rank,
    )
    if audit["status"] != "eligible":
        raise ValueError(str(audit["second_selection_error"]))
    return audit


def _absent_pair_record(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    min_rank: int,
) -> dict[str, Any]:
    """Build the persisted record for a successfully selected absent pair."""

    first_id = int(first["token_id"])
    second_id = int(second["token_id"])
    if first_id == second_id:
        raise RuntimeError("Absent-pair selection returned the same token twice")
    return {
        "status": "eligible",
        "token_ids": [first_id, second_id],
        "first": dict(first),
        "second": dict(second),
        "second_selection_error": None,
        "min_rank_rule": int(min_rank),
        "selection_rule": (
            "first two distinct preregistered nonexcluded tokens whose minimum "
            "J-Lens rank across workspace layers and positions meets the threshold"
        ),
        "selection_uses_behavior_outcomes": False,
    }


def bootstrap_effect_summary(
    deltas: Sequence[float],
    *,
    n_bootstrap: int = 5_000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Item-bootstrap signed and absolute intervention-effect summaries."""

    values = np.asarray(deltas, dtype=float).reshape(-1)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Effect summary requires at least three finite item effects")
    return {
        "n_items": len(values),
        "mean_delta": bootstrap_statistic(
            [values],
            lambda sample: float(np.mean(sample)),
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed,
        ),
        "mean_abs_delta": bootstrap_statistic(
            [values],
            lambda sample: float(np.mean(np.abs(sample))),
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed + 1,
        ),
        "median_delta": float(np.median(values)),
        "median_abs_delta": float(np.median(np.abs(values))),
        "max_abs_delta": float(np.max(np.abs(values))),
    }


def summarize_random_direction_null(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int = 5_000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Cluster random draws by item before bootstrapping null/observed effects."""

    if len(rows) < 3:
        raise ValueError("Random-null aggregate requires at least three items")
    null_mean: list[float] = []
    null_abs_mean: list[float] = []
    observed: list[float] = []
    observed_abs: list[float] = []
    for row in rows:
        draws = np.asarray([draw["delta"] for draw in row["draws"]], dtype=float)
        if not len(draws) or not np.isfinite(draws).all():
            raise ValueError("Every random-null item needs finite retained draws")
        null_mean.append(float(draws.mean()))
        null_abs_mean.append(float(np.abs(draws).mean()))
        if row.get("observed_concept_delta") is None:
            raise ValueError("Paired random-null summary requires every observed effect")
        value = float(row["observed_concept_delta"])
        if not math.isfinite(value):
            raise ValueError("Observed concept effects must be finite")
        observed.append(value)
        observed_abs.append(abs(value))

    return {
        "bootstrap_unit": "item; random draws are averaged within item",
        "n_items": len(rows),
        "n_draws_total": sum(len(row["draws"]) for row in rows),
        "mean_random_delta": bootstrap_statistic(
            [null_mean],
            lambda sample: float(np.mean(sample)),
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed,
        ),
        "mean_abs_random_delta": bootstrap_statistic(
            [null_abs_mean],
            lambda sample: float(np.mean(sample)),
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed + 1,
        ),
        "mean_observed_delta": bootstrap_statistic(
            [observed],
            lambda sample: float(np.mean(sample)),
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed + 2,
        ),
        "mean_abs_observed_delta": bootstrap_statistic(
            [observed_abs],
            lambda sample: float(np.mean(sample)),
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed + 3,
        ),
        "paired_mean_abs_observed_minus_random": bootstrap_statistic(
            [observed_abs, null_abs_mean],
            lambda observed_values, null_values: float(
                np.mean(observed_values - null_values)
            ),
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed + 4,
        ),
    }


def teacher_forced_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-sequence mean next-token NLL, excluding padding positions."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("Expected logits [B,S,V] and input_ids [B,S]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape[1] < 2:
        raise ValueError("Logits/input shapes must align and contain at least two tokens")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    labels = input_ids[:, 1:].to(logits.device)
    mask = attention_mask.to(logits.device).bool()
    # Require both a real context token and a real target token.  The conjunction
    # matters for left-padded tokenizers: their first nonpad token has no modeled
    # predecessor and must not enter teacher-forced NLL.
    valid = mask[:, :-1] & mask[:, 1:]
    counts = valid.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("Every sequence needs at least one unmasked next-token target")
    token_nll = -logits[:, :-1].float().log_softmax(dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    masked_nll = torch.where(valid, token_nll, torch.zeros_like(token_nll))
    return masked_nll.sum(dim=-1) / counts


def identity_jacobian_direction(
    unembedding: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    """Normalized unembedding row: the identity-Jacobian/logit-lens baseline."""

    if unembedding.ndim != 2:
        raise ValueError("Unembedding matrix must have shape [vocab, d_model]")
    if not 0 <= int(token_id) < unembedding.shape[0]:
        raise IndexError(f"token_id={token_id} outside vocabulary")
    row = unembedding[int(token_id)].detach().float()
    if not torch.isfinite(row).all() or float(row.norm()) == 0.0:
        raise ValueError(f"Degenerate unembedding row for token {token_id}")
    return F.normalize(row, dim=0)


def _behavior_metric(logits: torch.Tensor, item: Item) -> float:
    return float(
        logit_difference(
            logits,
            int(item["target_token_id"]),
            int(item["foil_token_id"]),
        )[0].cpu()
    )


def behavior_effect_record(
    clean_metric: float,
    edited_metric: float,
) -> dict[str, float]:
    """JSON-ready behavior record with the project's canonical effect sign."""

    return {
        "clean_metric": float(clean_metric),
        "edited_metric": float(edited_metric),
        "delta": signed_causal_delta(clean_metric, edited_metric),
    }


def _validate_item_direction_banks(
    items: Sequence[Item],
    direction_banks: Mapping[str, DirectionBank],
    layers: Sequence[int],
) -> None:
    expected_layers = {int(layer) for layer in layers}
    if not items or not expected_layers:
        raise ValueError("Controls require nonempty items and layers")
    names = [_item_name(item) for item in items]
    if len(names) != len(set(names)):
        raise ValueError("Eligible item names must be unique")
    missing = sorted(set(names) - set(direction_banks))
    if missing:
        raise KeyError(f"Missing direction banks for items: {missing}")
    for name in names:
        observed_layers = {int(layer) for layer in direction_banks[name]}
        if observed_layers != expected_layers:
            raise ValueError(
                f"Direction layers for {name!r} differ from workspace: "
                f"expected={sorted(expected_layers)}, observed={sorted(observed_layers)}"
            )


def run_random_ablation_controls(
    bundle: Any,
    items: Sequence[Item],
    direction_banks: Mapping[str, DirectionBank],
    *,
    n_draws: int = DEFAULT_RANDOM_DRAWS,
    n_bootstrap: int = 5_000,
    seed: int = SEED,
    observed_deltas: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run all-band matched-unit random ablation nulls and retain every draw."""

    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    rows: list[dict[str, Any]] = []
    for item in items:
        name = _item_name(item)
        input_ids = bundle.lens_model.encode(item["prompt"])
        clean_logits = forward_logits(bundle.hf_model, input_ids)
        clean_metric = _behavior_metric(clean_logits, item)
        draws: list[dict[str, Any]] = []
        for draw_index in range(n_draws):
            random_bank, layer_seeds = seeded_random_direction_bank(
                direction_banks[name],
                item_name=name,
                draw_index=draw_index,
                seed=seed,
            )
            edited_logits = forward_logits(
                bundle.hf_model,
                input_ids,
                blocks=bundle.lens_model.layers,
                edits=ablation_edits(random_bank),
            )
            edited_metric = _behavior_metric(edited_logits, item)
            effect = behavior_effect_record(clean_metric, edited_metric)
            draws.append(
                {
                    "draw_index": draw_index,
                    "layer_seeds": {
                        str(layer): layer_seed
                        for layer, layer_seed in layer_seeds.items()
                    },
                    "direction_sha256": {
                        str(layer): _direction_sha256(direction)
                        for layer, direction in random_bank.items()
                    },
                    "direction_norms": {
                        str(layer): float(direction.float().norm().cpu())
                        for layer, direction in random_bank.items()
                    },
                    **effect,
                }
            )
        null_deltas = np.asarray([draw["delta"] for draw in draws], dtype=float)
        observed_delta = (
            float(observed_deltas[name])
            if observed_deltas is not None and name in observed_deltas
            else None
        )
        empirical_comparison = None
        if observed_delta is not None:
            empirical_comparison = {
                "observed_delta": observed_delta,
                "two_sided_empirical_p": float(
                    (1 + np.count_nonzero(np.abs(null_deltas) >= abs(observed_delta)))
                    / (n_draws + 1)
                ),
                "observed_percentile_in_null": float(
                    100.0 * np.mean(null_deltas <= observed_delta)
                ),
            }
        rows.append(
            {
                "name": name,
                "n_draws": n_draws,
                "observed_concept_delta": observed_delta,
                "null_summary": {
                    "mean": float(null_deltas.mean()),
                    "std": float(null_deltas.std(ddof=0)),
                    "q025": float(np.quantile(null_deltas, 0.025)),
                    "median": float(np.median(null_deltas)),
                    "q975": float(np.quantile(null_deltas, 0.975)),
                },
                "empirical_comparison": empirical_comparison,
                "draws": draws,
            }
        )
    result = {
        "seed": seed,
        "n_draws_per_item": n_draws,
        "direction_rule": "independent seeded fp32 random unit vector per layer",
        "intervention": "all workspace layers; all prompt positions",
        "rows": rows,
    }
    result["aggregate"] = summarize_random_direction_null(
        rows,
        n_bootstrap=n_bootstrap,
        seed=seed + 50_000,
    )
    return result


def _candidate_rank_map(
    lens_logits: Mapping[int, torch.Tensor],
    candidate_token_ids: Sequence[int],
) -> dict[int, dict[int, list[int]]]:
    return {
        int(token_id): {
            int(layer): [
                token_rank(layer_logits[position], int(token_id))
                for position in range(layer_logits.shape[0])
            ]
            for layer, layer_logits in lens_logits.items()
        }
        for token_id in candidate_token_ids
    }


def run_absent_swap_controls(
    bundle: Any,
    lens: Any,
    items: Sequence[Item],
    concept_direction_banks: Mapping[str, DirectionBank],
    candidate_direction_banks: Mapping[int, DirectionBank],
    preregistered_token_ids: Sequence[int],
    *,
    min_rank: int = DEFAULT_ABSENT_MIN_RANK,
    n_bootstrap: int = 5_000,
    seed: int = SEED,
    token_surfaces: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Run an absent-pair null plus the legacy concept-to-absent stress test."""

    null_rows: list[dict[str, Any]] = []
    null_rank_infeasible_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    for item in items:
        name = _item_name(item)
        input_ids = bundle.lens_model.encode(item["prompt"])
        prompt_token_ids = [int(value) for value in input_ids[0].detach().cpu()]
        lens_logits, _, _ = lens.apply(
            bundle.lens_model,
            item["prompt"],
            layers=sorted(concept_direction_banks[name]),
            positions=None,
        )
        rank_map = _candidate_rank_map(lens_logits, preregistered_token_ids)
        # This selection occurs before the clean/edited behavior forward below.
        exclusions = (
            int(item["concept_token_id"]),
            int(item["foil_concept_token_id"]),
            int(item["target_token_id"]),
            int(item["foil_token_id"]),
            *prompt_token_ids,
        )
        pair_audit = audit_absent_pair_selection(
            preregistered_token_ids,
            rank_map,
            excluded_token_ids=exclusions,
            min_rank=min_rank,
        )
        first_selection = pair_audit["first"]
        first_id = int(first_selection["token_id"])
        second_selection = pair_audit["second"]
        pair_selection = pair_audit if pair_audit["status"] == "eligible" else None
        second_id = (
            int(second_selection["token_id"])
            if second_selection is not None
            else None
        )
        selected_ids = [first_id]
        if second_id is not None:
            selected_ids.append(second_id)
        for token_id in selected_ids:
            if token_id not in candidate_direction_banks:
                raise KeyError(f"No direction bank for selected absent token {token_id}")

        clean_logits = forward_logits(bundle.hf_model, input_ids)
        clean_metric = _behavior_metric(clean_logits, item)
        layers = sorted(concept_direction_banks[name])
        clean_residuals = capture_residuals(bundle.lens_model, input_ids, layers)
        stress_logits = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=clamped_swap_edits(
                clean_residuals,
                concept_direction_banks[name],
                candidate_direction_banks[first_id],
            ),
        )
        stress_metric = _behavior_metric(stress_logits, item)
        shared = {
            "name": name,
            "selection_exclusions": {
                "concept_token_id": int(item["concept_token_id"]),
                "foil_concept_token_id": int(item["foil_concept_token_id"]),
                "target_token_id": int(item["target_token_id"]),
                "foil_token_id": int(item["foil_token_id"]),
                "prompt_token_ids": prompt_token_ids,
            },
            "all_candidate_ranks": {
                str(token_id): {
                    str(layer): ranks for layer, ranks in per_layer.items()
                }
                for token_id, per_layer in rank_map.items()
            },
        }
        if pair_selection is not None and second_id is not None:
            null_logits = forward_logits(
                bundle.hf_model,
                input_ids,
                blocks=bundle.lens_model.layers,
                edits=clamped_swap_edits(
                    clean_residuals,
                    candidate_direction_banks[first_id],
                    candidate_direction_banks[second_id],
                ),
            )
            null_metric = _behavior_metric(null_logits, item)
            null_rows.append(
                {
                    **shared,
                    "selected_absent_token_ids": {
                        "first": first_id,
                        "second": second_id,
                    },
                    "selected_absent_surfaces": {
                        "first": (
                            token_surfaces.get(first_id)
                            if token_surfaces is not None
                            else None
                        ),
                        "second": (
                            token_surfaces.get(second_id)
                            if token_surfaces is not None
                            else None
                        ),
                    },
                    "pair_selection": pair_selection,
                    **behavior_effect_record(clean_metric, null_metric),
                }
            )
        else:
            null_rank_infeasible_rows.append(
                {
                    **shared,
                    "status": "excluded_before_behavior_fewer_than_two_candidates",
                    "selected_first_absent_token_id": first_id,
                    "selected_first_absent_surface": (
                        token_surfaces.get(first_id)
                        if token_surfaces is not None
                        else None
                    ),
                    "first_selection": first_selection,
                    "pair_selection_audit": pair_audit,
                }
            )
        stress_rows.append(
            {
                **shared,
                "selected_absent_token_id": first_id,
                "selected_absent_surface": (
                    token_surfaces.get(first_id)
                    if token_surfaces is not None
                    else None
                ),
                "selection": first_selection,
                **behavior_effect_record(clean_metric, stress_metric),
            }
        )

    common = {
        "selection_timing": "rank-only pair selection precedes behavior evaluation",
        "min_rank_threshold": min_rank,
        "candidate_token_order": [int(value) for value in preregistered_token_ids],
        "n_bootstrap": n_bootstrap,
        "confidence": 0.95,
    }
    return {
        "absent_coordinate_null": {
            **common,
            "control_role": "primary absent-coordinate null",
            "intervention": (
                "clean-clamped swap of two rank-qualified absent coordinates at "
                "all workspace layers and prompt positions; task concept untouched"
            ),
            "expected_effect": "near-zero delta; no post-hoc equivalence threshold",
            "rank_feasibility": {
                "criterion": (
                    "at least two preregistered nonexcluded candidates meet the "
                    "frozen rank threshold before behavior evaluation"
                ),
                "n_items_total": len(items),
                "n_items_included": len(null_rows),
                "n_items_excluded": len(null_rank_infeasible_rows),
                "excluded_item_names": [
                    row["name"] for row in null_rank_infeasible_rows
                ],
            },
            "aggregate": bootstrap_effect_summary(
                [row["delta"] for row in null_rows],
                n_bootstrap=n_bootstrap,
                seed=seed + 60_000,
            ),
            "rows": null_rows,
            "rank_infeasible_rows": null_rank_infeasible_rows,
        },
        "concept_to_absent_stress_test": {
            **common,
            "control_role": "non-null specificity stress test",
            "intervention": (
                "clean-clamped task-concept-to-absent coordinate swap; this removes "
                "and replaces the active task-concept coordinate"
            ),
            "expected_effect": (
                "not expected to be zero and must not be interpreted as an absent null"
            ),
            "n_items": len(stress_rows),
            "aggregate": bootstrap_effect_summary(
                [row["delta"] for row in stress_rows],
                n_bootstrap=n_bootstrap,
                seed=seed + 70_000,
            ),
            "rows": stress_rows,
        },
    }


def _perplexities(nll: torch.Tensor) -> list[float]:
    values = nll.detach().float().cpu().numpy()
    perplexities = [math.exp(float(value)) for value in values]
    if not all(math.isfinite(value) for value in perplexities):
        raise ValueError("Nonfinite perplexity in capability control")
    return perplexities


def _top1_correct(logits: torch.Tensor, target_token_id: int) -> int:
    return int(int(logits[0, -1].argmax().cpu()) == int(target_token_id))


def select_offtarget_capability_items(
    intervention_item: Item,
    eligible_items: Sequence[Item],
    *,
    n_tasks: int,
    prompt_token_ids: Mapping[str, Sequence[int]] | None = None,
) -> list[Item]:
    """Choose frozen-order unrelated tasks without using any model outcomes."""

    if n_tasks < 1:
        raise ValueError("n_tasks must be positive")
    items = list(eligible_items)
    source_name = _item_name(intervention_item)
    names = [_item_name(item) for item in items]
    if source_name not in names:
        raise ValueError(f"Intervention item {source_name!r} is not eligible")
    source_index = names.index(source_name)
    source_token_roles = {
        int(intervention_item[field])
        for field in (
            "concept_token_id",
            "foil_concept_token_id",
            "target_token_id",
            "foil_token_id",
        )
    }
    source_terms = {
        str(intervention_item.get(field, "")).strip().casefold()
        for field in ("intermediate", "swap_to")
        if str(intervention_item.get(field, "")).strip()
    }
    ordered = items[source_index + 1 :] + items[:source_index]
    selected: list[Item] = []
    for candidate in ordered:
        candidate_name = _item_name(candidate)
        candidate_roles = {
            int(candidate[field])
            for field in (
                "concept_token_id",
                "foil_concept_token_id",
                "target_token_id",
                "foil_token_id",
            )
        }
        if source_token_roles & candidate_roles:
            continue
        prompt = str(candidate["prompt"]).casefold()
        if any(term in prompt for term in source_terms):
            continue
        if prompt_token_ids is not None and source_token_roles & set(
            int(value) for value in prompt_token_ids[candidate_name]
        ):
            continue
        selected.append(candidate)
        if len(selected) == n_tasks:
            return selected
    raise ValueError(
        f"Only {len(selected)} off-target tasks available for {source_name!r}; "
        f"required {n_tasks}. Do not relax exclusions after inspecting outcomes."
    )


def run_capability_controls(
    bundle: Any,
    items: Sequence[Item],
    direction_banks: Mapping[str, DirectionBank],
    *,
    texts: Sequence[Mapping[str, str]] = CAPABILITY_TEXTS_V1,
    max_interventions: int | None = None,
    twohop_tasks_per_intervention: int = 4,
) -> dict[str, Any]:
    """Measure fixed-text NLL/perplexity and two-hop accuracy under ablation."""

    if not texts:
        raise ValueError("Capability text list must be nonempty")
    selected_items = list(items)
    if max_interventions is not None:
        if max_interventions < 1:
            raise ValueError("max_interventions must be positive or None")
        selected_items = selected_items[:max_interventions]
    prompt_set = {str(item["prompt"]).strip() for item in items}
    for text in texts:
        if text["text"].strip() in prompt_set:
            raise ValueError(f"Capability text {text['id']!r} overlaps a core prompt")

    device = next(bundle.hf_model.parameters()).device
    encoded = bundle.tokenizer(
        [text["text"] for text in texts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    clean_logits = forward_logits(
        bundle.hf_model, input_ids, attention_mask=attention_mask
    )
    clean_nll = teacher_forced_nll(clean_logits, input_ids, attention_mask).cpu()
    clean_ppl = _perplexities(clean_nll)

    twohop_ids_by_name = {
        _item_name(item): bundle.lens_model.encode(item["prompt"]) for item in items
    }
    prompt_token_ids = {
        name: [int(value) for value in ids[0].detach().cpu()]
        for name, ids in twohop_ids_by_name.items()
    }
    clean_twohop: dict[str, dict[str, float | int]] = {}
    for item in items:
        item_name = _item_name(item)
        logits = forward_logits(bundle.hf_model, twohop_ids_by_name[item_name])
        clean_twohop[item_name] = {
            "metric": _behavior_metric(logits, item),
            "top1_correct": _top1_correct(logits, int(item["target_token_id"])),
        }

    language_rows: list[dict[str, Any]] = []
    twohop_rows: list[dict[str, Any]] = []
    for item in selected_items:
        name = _item_name(item)
        edited_logits = forward_logits(
            bundle.hf_model,
            input_ids,
            attention_mask=attention_mask,
            blocks=bundle.lens_model.layers,
            edits=ablation_edits(direction_banks[name]),
        )
        edited_nll = teacher_forced_nll(
            edited_logits, input_ids, attention_mask
        ).cpu()
        edited_ppl = _perplexities(edited_nll)
        for text_index, text in enumerate(texts):
            language_rows.append(
                {
                    "intervention_item": name,
                    "text_id": text["id"],
                    "text": text["text"],
                    "clean_nll": float(clean_nll[text_index]),
                    "edited_nll": float(edited_nll[text_index]),
                    "delta_nll": float(edited_nll[text_index] - clean_nll[text_index]),
                    "clean_perplexity": clean_ppl[text_index],
                    "edited_perplexity": edited_ppl[text_index],
                    "delta_perplexity": edited_ppl[text_index]
                    - clean_ppl[text_index],
                }
            )

        off_target_items = select_offtarget_capability_items(
            item,
            items,
            n_tasks=twohop_tasks_per_intervention,
            prompt_token_ids=prompt_token_ids,
        )
        for evaluated_item in off_target_items:
            evaluated_name = _item_name(evaluated_item)
            twohop_edited = forward_logits(
                bundle.hf_model,
                twohop_ids_by_name[evaluated_name],
                blocks=bundle.lens_model.layers,
                edits=ablation_edits(direction_banks[name]),
            )
            clean_metric = float(clean_twohop[evaluated_name]["metric"])
            edited_metric = _behavior_metric(twohop_edited, evaluated_item)
            twohop_rows.append(
                {
                    "intervention_item": name,
                    "evaluated_item": evaluated_name,
                    "selection": "cyclic frozen order with token/semantic overlap exclusions",
                    "clean_top1_correct": int(
                        clean_twohop[evaluated_name]["top1_correct"]
                    ),
                    "edited_top1_correct": _top1_correct(
                        twohop_edited, int(evaluated_item["target_token_id"])
                    ),
                    **behavior_effect_record(clean_metric, edited_metric),
                }
            )

    return {
        "text_set": "CAPABILITY_TEXTS_V1",
        "n_fixed_texts": len(texts),
        "n_intervention_banks": len(selected_items),
        "intervention_selection": "first items in frozen eligible input order",
        "general_language": {
            "metric": "mean teacher-forced next-token NLL; perplexity=exp(NLL)",
            "rows": language_rows,
            "mean_clean_nll": float(
                np.mean([row["clean_nll"] for row in language_rows])
            ),
            "mean_edited_nll": float(
                np.mean([row["edited_nll"] for row in language_rows])
            ),
            "mean_delta_nll": float(
                np.mean([row["delta_nll"] for row in language_rows])
            ),
        },
        "twohop": {
            "metric": (
                "off-target exact target-token top-1; each intervention is evaluated "
                "on deterministic unrelated held-out tasks"
            ),
            "tasks_per_intervention": twohop_tasks_per_intervention,
            "selection_rule": (
                "cyclic frozen input order; exclude shared concept/foil/answer token "
                "roles, literal source concept terms, and literal source token IDs"
            ),
            "rows": twohop_rows,
            "clean_accuracy": float(
                np.mean([row["clean_top1_correct"] for row in twohop_rows])
            ),
            "edited_accuracy": float(
                np.mean([row["edited_top1_correct"] for row in twohop_rows])
            ),
        },
    }


def load_known_narration_source(
    path: str | Path = DEFAULT_NARRATION_SOURCE,
) -> dict[str, Any]:
    """Load and strictly validate the pinned upstream language experiment."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("task"), dict):
        raise ValueError(f"Malformed narration source: {source}")
    if not isinstance(payload.get("passages"), list) or not payload["passages"]:
        raise ValueError(f"Narration source has no passages: {source}")
    keys = [passage.get("key") for passage in payload["passages"]]
    if len(keys) != len(set(keys)) or set(keys) != set(KNOWN_NARRATION_MARGINS_V1):
        raise ValueError(
            "Narration passage keys differ from the preregistered margins: "
            f"source={sorted(keys)}, margins={sorted(KNOWN_NARRATION_MARGINS_V1)}"
        )
    categories = {passage.get("category") for passage in payload["passages"]}
    if categories != {"French", "German", "Spanish", "Italian"}:
        raise ValueError(f"Unexpected narration categories: {sorted(categories)}")
    payload["source_path"] = str(source.resolve())
    payload["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return payload


def _known_narration_prompt(task: Mapping[str, str], text: str) -> str:
    """Render the upstream automatic condition without adding a local marker."""

    return task["automatic_q"].format(text=text)


def _magnitude_summary(values_by_layer: Mapping[int, np.ndarray]) -> dict[str, Any]:
    if not values_by_layer:
        raise ValueError("WRITE/READ magnitude summary requires at least one layer")
    flattened = np.concatenate(
        [np.asarray(values, dtype=float).reshape(-1) for values in values_by_layer.values()]
    )
    if not flattened.size or not np.isfinite(flattened).all():
        raise ValueError("WRITE/READ arrays must be finite and nonempty")
    return {
        "n_layer_position_coordinates": int(flattened.size),
        "signed_sum": float(flattened.sum()),
        "abs_sum": float(np.abs(flattened).sum()),
        "abs_mean": float(np.abs(flattened).mean()),
        "rms": float(np.sqrt(np.square(flattened).mean())),
    }


def known_narration_reproduction_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen high-WRITE/low-causal reproduction criteria."""

    if len(rows) != len(KNOWN_NARRATION_MARGINS_V1):
        raise ValueError(
            f"Expected {len(KNOWN_NARRATION_MARGINS_V1)} narration rows, got {len(rows)}"
        )
    joint = [
        bool(row["high_write"])
        and bool(row["low_causal"])
        and bool(row["clean_capable"])
        for row in rows
    ]
    for row, expected in zip(rows, joint, strict=True):
        if "reproduces_known_narration" in row and bool(
            row["reproduces_known_narration"]
        ) != expected:
            raise ValueError("Inconsistent known-narration joint criterion flag")
    reproduced = sum(joint)
    high_write = sum(bool(row["high_write"]) for row in rows)
    low_causal = sum(bool(row["low_causal"]) for row in rows)
    clean_capable = sum(bool(row["clean_capable"]) for row in rows)
    return {
        "status": (
            "PASS" if reproduced >= KNOWN_NARRATION_MIN_REPRODUCED else "FAIL"
        ),
        "n_passages": len(rows),
        "n_high_write": high_write,
        "n_low_causal": low_causal,
        "n_clean_capable": clean_capable,
        "n_reproduced": reproduced,
        "criterion": (
            f">={KNOWN_NARRATION_MIN_REPRODUCED}/{len(rows)} passages jointly meet "
            f"min J-Lens rank<={KNOWN_NARRATION_HIGH_WRITE_MAX_RANK} and "
            "|internal ablation delta - output suppression delta|"
            f"<={KNOWN_NARRATION_LOW_CAUSAL_MAX_GAP}, and clean continuation "
            f"margin>{KNOWN_NARRATION_CLEAN_MARGIN_MIN}"
        ),
        "thresholds": {
            "high_write_max_min_jlens_rank": KNOWN_NARRATION_HIGH_WRITE_MAX_RANK,
            "low_causal_max_abs_delta_gap": KNOWN_NARRATION_LOW_CAUSAL_MAX_GAP,
            "clean_capability_min_exclusive_margin": (
                KNOWN_NARRATION_CLEAN_MARGIN_MIN
            ),
            "minimum_joint_passages": KNOWN_NARRATION_MIN_REPRODUCED,
        },
    }


def run_known_narration_controls(
    bundle: Any,
    lens: Any,
    language_direction_banks: Mapping[str, DirectionBank],
    *,
    source_path: str | Path = DEFAULT_NARRATION_SOURCE,
) -> dict[str, Any]:
    """Run language-label WRITE/READ and fixed automatic-continuation controls."""

    payload = load_known_narration_source(source_path)
    rows: list[dict[str, Any]] = []
    for passage in payload["passages"]:
        key = passage["key"]
        category = passage["category"]
        if category not in language_direction_banks:
            raise KeyError(f"Missing language-label directions for {category}")
        prompt = _known_narration_prompt(payload["task"], passage["text"])
        margin = KNOWN_NARRATION_MARGINS_V1[key]
        target_id, target_surface = continuation_token_id(
            bundle.tokenizer, prompt, margin["target"]
        )
        foil_id, foil_surface = continuation_token_id(
            bundle.tokenizer, prompt, margin["foil"]
        )
        label_id = single_token_id(bundle.tokenizer, f" {category}")
        if label_id in {target_id, foil_id}:
            raise ValueError(
                f"Narration label token aliases the metric token for passage {key}"
            )
        input_ids = bundle.lens_model.encode(prompt)
        directions = language_direction_banks[category]
        lens_logits, _, _ = lens.apply(
            bundle.lens_model,
            prompt,
            layers=sorted(directions),
            positions=None,
        )
        label_ranks = {
            int(layer): [
                token_rank(layer_logits[position], label_id)
                for position in range(layer_logits.shape[0])
            ]
            for layer, layer_logits in lens_logits.items()
        }
        min_label_rank = min(
            rank for ranks in label_ranks.values() for rank in ranks
        )
        explicit_prompt = payload["task"]["explicit_q"].format(text=passage["text"])
        explicit_lens_logits, _, _ = lens.apply(
            bundle.lens_model,
            explicit_prompt,
            layers=sorted(directions),
            positions=None,
        )
        explicit_label_ranks = {
            int(layer): [
                token_rank(layer_logits[position], label_id)
                for position in range(layer_logits.shape[0])
            ]
            for layer, layer_logits in explicit_lens_logits.items()
        }
        explicit_min_label_rank = min(
            rank for ranks in explicit_label_ranks.values() for rank in ranks
        )
        attribution = attribution_read(
            bundle.hf_model,
            bundle.lens_model.layers,
            input_ids,
            directions,
            target_token_id=target_id,
            foil_token_id=foil_id,
        )
        clean_logits = forward_logits(bundle.hf_model, input_ids)
        ablated_logits = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=ablation_edits(directions),
        )
        suppressed_logits = suppress_output_token(clean_logits, label_id)
        clean_metric = _behavior_metric(
            clean_logits,
            {"target_token_id": target_id, "foil_token_id": foil_id},
        )
        ablated_metric = _behavior_metric(
            ablated_logits,
            {"target_token_id": target_id, "foil_token_id": foil_id},
        )
        suppressed_metric = _behavior_metric(
            suppressed_logits,
            {"target_token_id": target_id, "foil_token_id": foil_id},
        )
        internal_delta = signed_causal_delta(clean_metric, ablated_metric)
        suppression_delta = signed_causal_delta(clean_metric, suppressed_metric)
        high_write = min_label_rank <= KNOWN_NARRATION_HIGH_WRITE_MAX_RANK
        causal_gap = abs(internal_delta - suppression_delta)
        low_causal = causal_gap <= KNOWN_NARRATION_LOW_CAUSAL_MAX_GAP
        clean_capable = clean_metric > KNOWN_NARRATION_CLEAN_MARGIN_MIN
        rows.append(
            {
                "passage_key": key,
                "language_label": category,
                "language_label_token_id": label_id,
                "prompt": prompt,
                "prompt_condition": "exact upstream automatic_q",
                "explicit_prompt": explicit_prompt,
                "metric_definition": (
                    f"logit({target_surface!r}) - logit({foil_surface!r}) at "
                    "the first fixed continuation token"
                ),
                "target_token_id": target_id,
                "target_surface": target_surface,
                "foil_token_id": foil_id,
                "foil_surface": foil_surface,
                "language_label_jlens_rank_by_layer_position": {
                    str(layer): ranks for layer, ranks in label_ranks.items()
                },
                "min_language_label_jlens_rank": min_label_rank,
                "high_write": high_write,
                "explicit_language_label_jlens_rank_by_layer_position": {
                    str(layer): ranks
                    for layer, ranks in explicit_label_ranks.items()
                },
                "explicit_min_language_label_jlens_rank": explicit_min_label_rank,
                "clean_metric": clean_metric,
                "clean_capable": clean_capable,
                "attribution_predicted_delta": attribution.predicted_delta,
                "attribution_predicted_delta_by_layer": {
                    str(layer): value
                    for layer, value in attribution.predicted_delta_by_layer.items()
                },
                "write_by_layer_position": {
                    str(layer): [float(value) for value in values]
                    for layer, values in attribution.write.items()
                },
                "read_by_layer_position": {
                    str(layer): [float(value) for value in values]
                    for layer, values in attribution.read.items()
                },
                "write_magnitude": _magnitude_summary(attribution.write),
                "read_magnitude": _magnitude_summary(attribution.read),
                "internal_ablation_metric": ablated_metric,
                "internal_ablation_delta": internal_delta,
                "output_suppression_metric": suppressed_metric,
                "output_suppression_delta": suppression_delta,
                "internal_vs_output_abs_delta_gap": causal_gap,
                "low_causal": low_causal,
                "reproduces_known_narration": (
                    high_write and low_causal and clean_capable
                ),
            }
        )
    reproduction = known_narration_reproduction_summary(rows)
    write_abs_means = [float(row["write_magnitude"]["abs_mean"]) for row in rows]
    read_abs_means = [float(row["read_magnitude"]["abs_mean"]) for row in rows]
    return {
        "status": reproduction["status"],
        "reproduction_gate": reproduction,
        "aggregate_write_read_magnitudes": {
            "write_abs_mean_across_passages": bootstrap_statistic(
                [write_abs_means],
                lambda values: float(np.mean(values)),
                n_bootstrap=5_000,
                confidence=0.95,
                seed=SEED,
            ),
            "read_abs_mean_across_passages": bootstrap_statistic(
                [read_abs_means],
                lambda values: float(np.mean(values)),
                n_bootstrap=5_000,
                confidence=0.95,
                seed=SEED + 1,
            ),
            "write_abs_mean_median": float(np.median(write_abs_means)),
            "read_abs_mean_median": float(np.median(read_abs_means)),
        },
        "source_path": payload["source_path"],
        "source_sha256": payload["source_sha256"],
        "metric": (
            "preregistered fixed next-token margin after exact upstream automatic_q; "
            "no generation"
        ),
        "margin_registration": KNOWN_NARRATION_MARGINS_V1,
        "rows": rows,
    }


def run_logit_lens_baseline(
    bundle: Any,
    items: Sequence[Item],
    layers: Sequence[int],
    *,
    n_bootstrap: int = 2_000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run identity-Jacobian directions with WRITE, READ, and real ablation."""

    weight = unembedding_weight(bundle.lens_model)
    rows: list[dict[str, Any]] = []
    layer_list = sorted(int(layer) for layer in layers)
    for item in items:
        name = _item_name(item)
        direction = identity_jacobian_direction(
            weight, int(item["concept_token_id"])
        )
        directions = {layer: direction for layer in layer_list}
        input_ids = bundle.lens_model.encode(item["prompt"])
        attribution = attribution_read(
            bundle.hf_model,
            bundle.lens_model.layers,
            input_ids,
            directions,
            target_token_id=int(item["target_token_id"]),
            foil_token_id=int(item["foil_token_id"]),
        )
        clean_logits = forward_logits(bundle.hf_model, input_ids)
        ablated_logits = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=ablation_edits(directions),
        )
        clean_metric = _behavior_metric(clean_logits, item)
        ablated_metric = _behavior_metric(ablated_logits, item)
        write_values = np.concatenate(
            [np.asarray(values, dtype=float).reshape(-1) for values in attribution.write.values()]
        )
        rows.append(
            {
                "name": name,
                "direction": "normalize(unembedding[concept_token_id]); identity J",
                "write_sum": float(
                    sum(float(values.sum()) for values in attribution.write.values())
                ),
                "write_strength": float(np.abs(write_values).mean()),
                "write_by_layer_position": {
                    str(layer): [float(value) for value in values]
                    for layer, values in attribution.write.items()
                },
                "read_by_layer_position": {
                    str(layer): [float(value) for value in values]
                    for layer, values in attribution.read.items()
                },
                "predicted_delta": attribution.predicted_delta,
                "predicted_delta_by_layer": {
                    str(layer): value
                    for layer, value in attribution.predicted_delta_by_layer.items()
                },
                "actual_delta": signed_causal_delta(clean_metric, ablated_metric),
                "clean_metric": clean_metric,
                "ablated_metric": ablated_metric,
            }
        )
    return {
        "direction_definition": "normalized token unembedding row (identity Jacobian)",
        "predictor": _predictor_summary(rows, n_bootstrap=n_bootstrap, seed=seed),
        "rows": rows,
    }


def _predictor_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    finite_rows = [
        row
        for row in rows
        if row.get("predicted_delta") is not None
        and row.get("actual_delta") is not None
        and math.isfinite(float(row["predicted_delta"]))
        and math.isfinite(float(row["actual_delta"]))
    ]
    predicted = [float(row["predicted_delta"]) for row in finite_rows]
    actual = [float(row["actual_delta"]) for row in finite_rows]
    if len(finite_rows) < 3 or np.std(predicted) == 0 or np.std(actual) == 0:
        return {
            "status": "UNAVAILABLE",
            "reason": "need >=3 finite, nonconstant prediction/effect pairs",
            "n": len(finite_rows),
        }
    return {
        "status": "COMPUTED",
        **pearson_with_ci(
            predicted,
            actual,
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed,
        ),
    }


def compare_causal_predictors(
    core_rows: Sequence[Mapping[str, Any]],
    logit_lens_rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int = 2_000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Report causal-prediction validity for the core and identity-J directions."""

    adapted = adapt_core_control_rows(core_rows)
    core_by_name = {row["name"]: row for row in adapted}
    identity_by_name = {str(row["name"]): row for row in logit_lens_rows}
    common = sorted(set(core_by_name) & set(identity_by_name))
    identity_vs_core = [
        {
            "name": name,
            "predicted_delta": identity_by_name[name]["predicted_delta"],
            "actual_delta": core_by_name[name]["actual_delta"],
        }
        for name in common
    ]
    identity_write_vs_core = [
        {
            "name": name,
            "predicted_delta": identity_by_name[name].get("write_strength"),
            "actual_delta": core_by_name[name]["actual_delta"],
        }
        for name in common
    ]
    core_write_vs_core = [
        {
            "name": name,
            "predicted_delta": core_by_name[name].get("write_strength"),
            "actual_delta": core_by_name[name]["actual_delta"],
        }
        for name in common
    ]
    return {
        "shared_core_causal_target": {
            "outcome": "core direction's measured all-band ablation delta",
            "core_first_order_predictor": _predictor_summary(
                adapted, n_bootstrap=n_bootstrap, seed=seed
            ),
            "identity_j_first_order_association": _predictor_summary(
                identity_vs_core, n_bootstrap=n_bootstrap, seed=seed
            ),
            "core_write_association": _predictor_summary(
                core_write_vs_core, n_bootstrap=n_bootstrap, seed=seed
            ),
            "identity_j_write_association": _predictor_summary(
                identity_write_vs_core, n_bootstrap=n_bootstrap, seed=seed
            ),
            "n_name_aligned": len(common),
        },
        "core_direction_within_intervention": _predictor_summary(
            adapted, n_bootstrap=n_bootstrap, seed=seed
        ),
        "identity_jacobian_within_intervention": _predictor_summary(
            logit_lens_rows, n_bootstrap=n_bootstrap, seed=seed
        ),
        "comparison_note": (
            "The headline baseline comparison joins both estimators to the same core "
            "ablation outcome. Within-direction validity is retained as a diagnostic."
        ),
    }


def plot_random_null_controls(
    random_results: Mapping[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot every retained random-null effect against observed concept effects."""

    set_style()
    null_deltas = [
        float(draw["delta"])
        for row in random_results["rows"]
        for draw in row["draws"]
    ]
    observed = [
        float(row["observed_concept_delta"])
        for row in random_results["rows"]
        if row.get("observed_concept_delta") is not None
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    axis.hist(
        null_deltas,
        bins=min(50, max(10, int(np.sqrt(len(null_deltas))))),
        color="#4C78A8",
        alpha=0.72,
        label=f"random unit directions (N={len(null_deltas)})",
    )
    if observed:
        for value in observed:
            axis.axvline(value, color="#B33A3A", alpha=0.12, lw=1)
        axis.axvline(
            float(np.median(observed)),
            color="#8C1D1D",
            lw=2,
            label=f"concept median (N={len(observed)})",
        )
    axis.axvline(0, color="0.25", ls="--", lw=1)
    axis.set(
        xlabel=r"signed causal effect $\Delta M=M_{edited}-M_{clean}$",
        ylabel="retained draws",
        title="Matched-norm all-band random-direction ablation null",
    )
    axis.legend(frameon=False)
    return figure, axis


def plot_internal_vs_output_suppression(
    comparison: Mapping[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    """F3: internal ablation beside the structural final-logit zero check."""

    set_style()
    rows = comparison["rows"]
    if not rows:
        raise ValueError("F3 requires at least one suppression-comparison row")
    internal = np.asarray(
        [row["internal_ablation_delta"] for row in rows], dtype=float
    )
    output = np.asarray([row["output_suppression_delta"] for row in rows], dtype=float)
    if not np.isfinite(internal).all() or not np.isfinite(output).all():
        raise ValueError("F3 inputs must be finite")
    lower = float(min(internal.min(), output.min()))
    upper = float(max(internal.max(), output.max()))
    if lower == upper:
        padding = max(1.0, abs(lower) * 0.1)
        lower -= padding
        upper += padding
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.scatter(internal, output, s=38, alpha=0.78, color="#4C78A8", edgecolors="none")
    axis.plot([lower, upper], [lower, upper], "--", color="0.35", lw=1.2, label="identity")
    axis.axhline(0, color="0.7", lw=0.8)
    axis.axvline(0, color="0.7", lw=0.8)
    axis.text(
        0.04,
        0.96,
        f"N = {len(rows)}",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.set(
        xlabel=r"internal all-band ablation $\Delta M$",
        ylabel=r"final concept-logit clamp $\Delta M$ (structural zero)",
        title="F3 — internal effect vs structural output-logit check",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    axis.legend(frameon=False)
    return figure, axis


def plot_capability_controls(
    capability: Mapping[str, Any],
) -> tuple[plt.Figure, np.ndarray]:
    """Plot fixed-text NLL damage and clean/edited two-hop accuracy."""

    set_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    nll_deltas = [
        float(row["delta_nll"])
        for row in capability["general_language"]["rows"]
    ]
    axes[0].hist(
        nll_deltas,
        bins=min(40, max(10, int(np.sqrt(len(nll_deltas))))),
        color="#59A14F",
        alpha=0.78,
    )
    axes[0].axvline(0, color="0.25", ls="--", lw=1)
    axes[0].set(
        xlabel="edited NLL - clean NLL",
        ylabel="item × fixed-text observations",
        title="General next-token capability",
    )
    accuracies = [
        capability["twohop"]["clean_accuracy"],
        capability["twohop"]["edited_accuracy"],
    ]
    axes[1].bar(["clean", "intervened"], accuracies, color=["#4C78A8", "#E45756"])
    axes[1].set(ylim=(0, 1), ylabel="exact target top-1 accuracy", title="Two-hop capability")
    return figure, axes


def plot_known_narration_controls(
    narration: Mapping[str, Any],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot attribution, internal-ablation, and output-only effects by passage."""

    set_style()
    rows = narration["rows"]
    labels = [row["passage_key"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    axis.bar(
        x - width,
        [row["attribution_predicted_delta"] for row in rows],
        width,
        label="attribution prediction",
        color="#4C78A8",
    )
    axis.bar(
        x,
        [row["internal_ablation_delta"] for row in rows],
        width,
        label="internal ablation",
        color="#E45756",
    )
    axis.bar(
        x + width,
        [row["output_suppression_delta"] for row in rows],
        width,
        label="language-label output suppression",
        color="#72B7B2",
    )
    axis.axhline(0, color="0.25", ls="--", lw=1)
    axis.set_xticks(x, labels)
    axis.set(
        xlabel="upstream passage key",
        ylabel=r"signed continuation-margin effect $\Delta M$",
        title="Known narration: internal language label vs output-only control",
    )
    axis.legend(frameon=False, ncol=3)
    return figure, axis


def _core_rows_by_name(
    adapted_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in adapted_rows:
        name = str(row["name"])
        if name in result:
            raise ValueError(
                f"The controls runner needs one core row per direction bank; duplicate {name!r}"
            )
        result[name] = row
    return result


def run_controls_phase(
    bundle: Any,
    lens: Any,
    items: Sequence[Item],
    direction_banks: Mapping[str, DirectionBank],
    core_rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    *,
    n_random_draws: int = DEFAULT_RANDOM_DRAWS,
    absent_min_rank: int = DEFAULT_ABSENT_MIN_RANK,
    control_n_bootstrap: int = 5_000,
    capability_item_limit: int | None = None,
    capability_tasks_per_intervention: int = 4,
    seed: int = SEED,
    narration_source: str | Path = DEFAULT_NARRATION_SOURCE,
    fold_rms_gain_for_control_labels: bool = False,
    output_path: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute all notebook-03 controls on explicit eligible items/directions.

    The caller owns model/lens loading and chooses the core direction family.  No
    notebook-02 artifact schema is assumed beyond the explicit item and row fields.
    """

    item_list = list(items)
    layer_list = sorted(int(layer) for layer in layers)
    _validate_item_direction_banks(item_list, direction_banks, layer_list)
    names = [_item_name(item) for item in item_list]
    adapted_core = assert_output_suppression_complete(
        core_rows,
        expected_item_names=names,
        require_actual_delta=True,
    )
    structural_suppression = assert_structural_concept_output_zero(
        item_list,
        adapted_core,
    )
    core_by_name = _core_rows_by_name(adapted_core)
    observed_deltas = {
        name: float(row["actual_delta"])
        for name, row in core_by_name.items()
        if row.get("actual_delta") is not None
        and math.isfinite(float(row["actual_delta"]))
    }

    candidate_ids = [
        single_token_id(bundle.tokenizer, surface)
        for surface in ABSENT_CONCEPT_SURFACES_V1
    ]
    candidate_surfaces = dict(zip(candidate_ids, ABSENT_CONCEPT_SURFACES_V1, strict=True))
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Preregistered absent surfaces do not map to unique tokens")
    candidate_directions = jlens_direction_bank(
        lens,
        bundle.lens_model,
        candidate_ids,
        layer_list,
        fold_rms_gain=fold_rms_gain_for_control_labels,
    )

    language_labels = ("French", "German", "Spanish", "Italian")
    language_ids = {
        label: single_token_id(bundle.tokenizer, f" {label}")
        for label in language_labels
    }
    language_by_token = jlens_direction_bank(
        lens,
        bundle.lens_model,
        language_ids.values(),
        layer_list,
        fold_rms_gain=fold_rms_gain_for_control_labels,
    )
    language_directions = {
        label: language_by_token[token_id]
        for label, token_id in language_ids.items()
    }

    random_results = run_random_ablation_controls(
        bundle,
        item_list,
        direction_banks,
        n_draws=n_random_draws,
        n_bootstrap=control_n_bootstrap,
        seed=seed,
        observed_deltas=observed_deltas,
    )
    absent_controls = run_absent_swap_controls(
        bundle,
        lens,
        item_list,
        direction_banks,
        candidate_directions,
        candidate_ids,
        min_rank=absent_min_rank,
        n_bootstrap=control_n_bootstrap,
        seed=seed,
        token_surfaces=candidate_surfaces,
    )
    capability_results = run_capability_controls(
        bundle,
        item_list,
        direction_banks,
        max_interventions=capability_item_limit,
        twohop_tasks_per_intervention=capability_tasks_per_intervention,
    )
    narration_results = run_known_narration_controls(
        bundle,
        lens,
        language_directions,
        source_path=narration_source,
    )
    logit_lens_results = run_logit_lens_baseline(
        bundle,
        item_list,
        layer_list,
        seed=seed,
    )
    predictor_comparison = compare_causal_predictors(
        core_rows,
        logit_lens_results["rows"],
        seed=seed,
    )
    suppression_comparison = core_output_suppression_comparison(adapted_core)

    figure_root = Path(figures_dir or ROOT / "results" / "figures")
    suppression_figure, _ = plot_internal_vs_output_suppression(
        suppression_comparison
    )
    suppression_path = save_figure(
        suppression_figure,
        figure_root / "f3_internal_vs_output_suppression.png",
    )
    plt.close(suppression_figure)
    random_figure, _ = plot_random_null_controls(random_results)
    random_path = save_figure(
        random_figure, figure_root / "controls_random_direction_null.png"
    )
    plt.close(random_figure)
    capability_figure, _ = plot_capability_controls(capability_results)
    capability_path = save_figure(
        capability_figure, figure_root / "controls_capability.png"
    )
    plt.close(capability_figure)
    narration_figure, _ = plot_known_narration_controls(narration_results)
    narration_path = save_figure(
        narration_figure, figure_root / "controls_known_narration.png"
    )
    plt.close(narration_figure)

    summary: dict[str, Any] = {
        "seed": seed,
        "effect_sign": "delta = edited - clean",
        "n_items": len(item_list),
        "workspace_layers": layer_list,
        "core_output_suppression_assertion": {
            "status": "PASS",
            "n_rows": len(adapted_core),
            "classification": structural_suppression["classification"],
            "structural_zero_assertion": structural_suppression,
            "comparison": suppression_comparison,
            "rows": adapted_core,
        },
        "random_direction_null": random_results,
        "absent_coordinate_null": absent_controls["absent_coordinate_null"],
        "concept_to_absent_stress_test": absent_controls[
            "concept_to_absent_stress_test"
        ],
        "capability": capability_results,
        "known_narration": narration_results,
        "logit_lens_identity_jacobian": logit_lens_results,
        "causal_predictor_comparison": predictor_comparison,
        "direction_convention_for_absent_and_language_labels": (
            "rms_gain_folded" if fold_rms_gain_for_control_labels else "raw_WU_J"
        ),
        "figures": {
            "f3_internal_vs_output_suppression": str(suppression_path),
            "random_null": str(random_path),
            "capability": str(capability_path),
            "known_narration": str(narration_path),
        },
        "limitations": [
            (
                "Final concept-token suppression is structurally zero for this "
                "target-minus-foil metric and checks instrumentation/direct-logit "
                "steering; it is not additional causal evidence."
            ),
            "Random directions are norm- and layer-count matched, not geometry matched.",
            "The absent-coordinate null is conditional on a fixed rank threshold and candidate list.",
            (
                "The retained concept-to-absent intervention removes the active concept "
                "coordinate and is a stress test, not a null."
            ),
            "Capability NLL uses a small authored fixed text set, not a benchmark corpus.",
            "Known narration tests a preregistered first-token margin, not free-form generation quality.",
            "Identity-J and core predictors are validated against different ablation directions.",
        ],
    }
    if output_path is not None:
        save_json(output_path, summary)
    return summary

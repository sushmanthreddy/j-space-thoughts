"""Final matched datasets, verification gates, and sanitized manifests.

The working pipeline uses only the audited reciprocal supplement in
``data/specs/twohop_supplement.json``. It constructs matched engine/idle
prompts, applies clean-answer plus WRITTEN verification, and builds the v6
answer-type-matched hard dashboards. This module never imports causal patching
or cheap READ estimators.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.jlens_interface import (
    MODEL_ID,
    MODEL_REVISION,
    batched_next_token_records,
    concept_token_id,
    enforce_kl_gate,
    hf_wrapper_logit_kl,
    jlens_direction_bank,
    residual_prompt_matrices,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TWOHOP_SUPPLEMENT = PROJECT_ROOT / "data" / "specs" / "twohop_supplement.json"
SUPPLEMENT_TWOHOP_SOURCE = "j-space-thoughts:data/specs/twohop_supplement.json"

SYMMETRIC_READ_SEED = 1729
SYMMETRIC_CALIBRATION_MIN_PAIRS = 24
SYMMETRIC_N_FOLDS = 5
SYMMETRIC_LAYER_CANDIDATES = tuple(range(13, 27))
SYMMETRIC_POSITION_RULE = "explicit_concept_token_in_shared_context"

ProgressCallback = Callable[[str, Mapping[str, Any]], None]

TWOHOP_REQUIRED_FIELDS = {
    "name",
    "category",
    "prompt",
    "intermediate",
    "answer",
    "swap_to",
    "swap_answer",
}

G1_PROMPTS = [
    "Fact: The number of legs on the animal that spins webs is ",
    "Fact: The capital of the country where champagne originated is ",
    "Fact: The language spoken in the country where the Amazon River ends is ",
    "Fact: The atomic number of the gas plants release during photosynthesis is ",
    "Fact: The currency used in the country shaped like a boot is ",
    "A quiet river crossed the valley beneath a cloudless sky.",
    "Write one word that follows naturally: The musician tuned the violin and then",
    "Question: Which planet is known for its prominent rings? Answer:",
    "Translate to Spanish: The library closes at six.",
    "The old bank stood beside the river after the storm.",
    "A botanist catalogued twelve unfamiliar flowers in the greenhouse.",
    "If a dozen eggs are split equally among three baskets, each basket contains",
    "The telescope on the mountain was calibrated before sunset.",
    "Fact: The continent containing the city with the Eiffel Tower is ",
    "Complete the sequence: red, orange, yellow, green, blue,",
    "The committee postponed its decision until the evidence was reviewed.",
    "Question: What metal is liquid near room temperature? Answer:",
    "Fact: The author of Hamlet was born in the country whose capital is ",
    "A chef sliced the bread, warmed the soup, and set the table.",
    "The pronoun in 'Alex thanked Jordan because they helped' may be ambiguous.",
]

_SYMMETRIC_TASK_TEMPLATES: dict[str, dict[str, str]] = {
    "atomic-number-element-symbol": {
        "clue_prefix": "The chemical symbol of ",
        "completion": "Its chemical symbol is",
    },
    "us-city-state-capital": {
        "clue_prefix": "The capital of ",
        "completion": "Its capital is",
    },
    "city-country-capital": {
        "clue_prefix": "The capital of ",
        "completion": "Its capital is",
    },
}

HARD_DASHBOARD_TEMPLATES: dict[str, dict[str, str]] = {
    "atomic-number-element-symbol": {
        "template_id": "fixed-calibration-platinum-symbol",
        "suffix": " For comparison, the chemical symbol of platinum is",
        "target": "Pt",
        "target_surface": " Pt",
        "distractor": "Sn",
        "distractor_surface": " Sn",
        "answer_type": "chemical element symbol",
        "relation": "element-to-chemical-symbol",
        "anchor_dependency_group": "atomic-number-element-symbol:platinum<->tin",
        "irrelevance": "The source element cannot determine platinum's symbol.",
    },
    "city-country-capital": {
        "template_id": "fixed-calibration-netherlands-capital",
        "suffix": " For comparison, the capital of the Netherlands is",
        "target": "Amsterdam",
        "target_surface": " Amsterdam",
        "distractor": "Lima",
        "distractor_surface": " Lima",
        "answer_type": "national capital city",
        "relation": "country-to-national-capital",
        "anchor_dependency_group": "city-country-capital:netherlands<->peru",
        "irrelevance": "The source country cannot determine the Netherlands' capital.",
    },
    "us-city-state-capital": {
        "template_id": "fixed-calibration-alabama-capital",
        "suffix": " For comparison, the capital of the US state of Alabama is",
        "target": "Montgomery",
        "target_surface": " Montgomery",
        "distractor": "Atlanta",
        "distractor_surface": " Atlanta",
        "answer_type": "US state capital city",
        "relation": "US-state-to-state-capital",
        "anchor_dependency_group": "us-city-state-capital:alabama<->georgia",
        "irrelevance": "The source state cannot determine Alabama's capital.",
    },
}


def _validated_twohop_items(items: Any, *, source: Path) -> list[dict[str, Any]]:
    """Validate the schema used by reciprocal prompt construction."""

    if not isinstance(items, list) or not items:
        raise ValueError(f"Malformed two-hop dataset at {source}: non-empty items required")
    validated: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"Item {index} at {source} is not an object")
        missing = TWOHOP_REQUIRED_FIELDS - set(raw_item)
        if missing:
            raise ValueError(f"Item {index} at {source} missing fields {sorted(missing)}")
        for field in TWOHOP_REQUIRED_FIELDS:
            value = raw_item[field]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"Item {index} field {field!r} at {source} must be a non-empty string"
                )
        name = str(raw_item["name"])
        if name in names:
            raise ValueError(f"Duplicate item name {name!r} at {source}")
        names.add(name)
        validated.append(dict(raw_item))
    return validated


def load_twohop_supplement(
    path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
) -> dict[str, Any]:
    """Load and audit the project-owned reciprocal two-hop specification."""

    spec_path = Path(path)
    with spec_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed supplemental two-hop spec at {spec_path}")
    for field in ("schema_version", "seed", "source", "provenance", "items"):
        if field not in payload:
            raise ValueError(f"Supplemental two-hop spec missing {field!r}: {spec_path}")
    if not isinstance(payload["provenance"], dict):
        raise ValueError(f"Supplemental provenance must be an object: {spec_path}")
    items = _validated_twohop_items(payload["items"], source=spec_path)
    expected_count = payload["provenance"].get("supplement_item_count")
    if expected_count != len(items):
        raise ValueError(
            f"Supplemental count mismatch at {spec_path}: "
            f"declared={expected_count!r}, actual={len(items)}"
        )
    return {**payload, "items": items}


def _supplement_source_id(path: Path) -> str:
    """Return a portable project-relative source identifier when possible."""

    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return str(path.resolve())
    return f"j-space-thoughts:{relative.as_posix()}"


def continuation_token_id(
    tokenizer: Any,
    prompt: str,
    continuation: str,
) -> tuple[int, str]:
    """Resolve the exact single token appended after a fixed prompt."""

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokenization is empty")
    if prompt[-1:].isspace():
        candidates = [continuation]
    else:
        candidates = (
            [continuation]
            if continuation.startswith(" ")
            else [f" {continuation}", continuation]
        )
    diagnostics: list[str] = []
    for surface in candidates:
        combined = tokenizer.encode(prompt + surface, add_special_tokens=False)
        appended = (
            combined[len(prompt_ids) :]
            if combined[: len(prompt_ids)] == prompt_ids
            else []
        )
        diagnostics.append(f"{surface!r}: full_tail={combined[-4:]}, appended={appended}")
        if len(appended) == 1:
            return int(appended[0]), surface
    raise ValueError(
        f"Continuation {continuation!r} is not one exact next token after prompt: "
        + "; ".join(diagnostics)
    )


def _concept_answer_key(item: Mapping[str, Any]) -> tuple[str, str]:
    return (str(item["intermediate"]).casefold(), str(item["answer"]).casefold())


def _reciprocal_group_key(
    item: Mapping[str, Any],
) -> tuple[str, tuple[str, str], tuple[str, str]]:
    source = _concept_answer_key(item)
    target = (str(item["swap_to"]).casefold(), str(item["swap_answer"]).casefold())
    low, high = sorted((source, target))
    return (str(item["category"]), low, high)


def _symmetric_prompt(
    category: str,
    concept: str,
    source_prompt: str,
    *,
    dashboard: bool,
) -> tuple[str, str, str]:
    """Construct the exact shared context and task-specific continuation."""

    try:
        template = _SYMMETRIC_TASK_TEMPLATES[category]
    except KeyError as error:
        raise ValueError(f"No symmetric task template for category {category!r}") from error
    clue = " ".join(str(source_prompt).split())
    if clue.startswith("Fact: "):
        clue = clue[len("Fact: ") :]
    if clue.endswith(" is"):
        clue = clue[:-3]
    prefix = template["clue_prefix"]
    if not clue.startswith(prefix):
        raise ValueError(
            f"Source prompt does not match {category!r} clue prefix {prefix!r}: {clue!r}"
        )
    description = clue[len(prefix) :]
    if not concept.strip():
        raise ValueError("Concept label must be non-empty")
    concept_prefix = f"Fact: {description} is {concept}"
    context = f"{concept_prefix}."
    if dashboard:
        prompt = f"{context} 2 + 2 = "
    else:
        prompt = f"{context} {template['completion']}"
    return context, concept_prefix, prompt


def build_symmetric_causal_candidates(
    supplement_path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
    *,
    seed: int = SYMMETRIC_READ_SEED,
    calibration_min_pairs: int = SYMMETRIC_CALIBRATION_MIN_PAIRS,
    n_folds: int = SYMMETRIC_N_FOLDS,
) -> dict[str, Any]:
    """Build reciprocal matched prompts and leakage-safe group assignments.

    Repeated natural contexts remain separate prompt pairs but share one
    dependency group. Whole dependency groups are assigned to calibration or
    one held-out fold; rows are never split independently.
    """

    if calibration_min_pairs < 1 or n_folds < 2:
        raise ValueError("calibration_min_pairs must be positive and n_folds >= 2")
    payload = load_twohop_supplement(supplement_path)
    grouped: dict[
        tuple[str, tuple[str, str], tuple[str, str]], dict[int, list[dict[str, Any]]]
    ] = {}
    for item in payload["items"]:
        key = _reciprocal_group_key(item)
        side = 0 if _concept_answer_key(item) == key[1] else 1
        grouped.setdefault(key, {0: [], 1: []})[side].append(item)

    pairs: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for key in sorted(grouped):
        category, _low, _high = key
        left = sorted(grouped[key][0], key=lambda row: str(row["name"]))
        right = sorted(grouped[key][1], key=lambda row: str(row["name"]))
        if not left or len(left) != len(right):
            raise ValueError(
                f"Reciprocal group {key!r} is unbalanced: {len(left)} versus {len(right)}"
            )
        concept_a, answer_a = left[0]["intermediate"], left[0]["answer"]
        concept_b, answer_b = right[0]["intermediate"], right[0]["answer"]
        dependency_group = (
            f"{category}:{str(concept_a).casefold()}<->{str(concept_b).casefold()}"
        )
        group_counts[dependency_group] = len(left)
        for slot, (row_a, row_b) in enumerate(zip(left, right, strict=True)):
            if (
                str(row_a["intermediate"]).casefold() != str(concept_a).casefold()
                or str(row_b["intermediate"]).casefold() != str(concept_b).casefold()
                or str(row_a["answer"]).casefold() != str(answer_a).casefold()
                or str(row_b["answer"]).casefold() != str(answer_b).casefold()
            ):
                raise ValueError(f"Inconsistent reciprocal group {dependency_group!r}")
            context_a, concept_prefix_a, engine_a = _symmetric_prompt(
                category, str(concept_a), str(row_a["prompt"]), dashboard=False
            )
            context_b, concept_prefix_b, engine_b = _symmetric_prompt(
                category, str(concept_b), str(row_b["prompt"]), dashboard=False
            )
            dash_context_a, dash_prefix_a, dashboard_a = _symmetric_prompt(
                category, str(concept_a), str(row_a["prompt"]), dashboard=True
            )
            dash_context_b, dash_prefix_b, dashboard_b = _symmetric_prompt(
                category, str(concept_b), str(row_b["prompt"]), dashboard=True
            )
            if context_a != dash_context_a or context_b != dash_context_b:
                raise AssertionError("Engine/dashboard contexts must be byte-identical")
            if concept_prefix_a != dash_prefix_a or concept_prefix_b != dash_prefix_b:
                raise AssertionError("Engine/dashboard measurement prefixes must match")
            pairs.append(
                {
                    "pair_id": f"symmetric-{len(pairs):03d}",
                    "dependency_group": dependency_group,
                    "context_slot": slot,
                    "category": category,
                    "concept_a": str(concept_a),
                    "concept_b": str(concept_b),
                    "answer_a": str(answer_a),
                    "answer_b": str(answer_b),
                    "context_a": context_a,
                    "context_b": context_b,
                    "concept_prefix_a": concept_prefix_a,
                    "concept_prefix_b": concept_prefix_b,
                    "engine_prompt_a": engine_a,
                    "engine_prompt_b": engine_b,
                    "dashboard_prompt_a": dashboard_a,
                    "dashboard_prompt_b": dashboard_b,
                    "dashboard_answer": "4",
                    "dashboard_distractor": "5",
                    "source_row_a": str(row_a["name"]),
                    "source_row_b": str(row_b["name"]),
                    "source_prompt_a": str(row_a["prompt"]),
                    "source_prompt_b": str(row_b["prompt"]),
                }
            )

    if len(pairs) < 100:
        raise ValueError(f"Symmetric candidate pool has only {len(pairs)} prompt pairs")
    if len({(row["engine_prompt_a"], row["engine_prompt_b"]) for row in pairs}) != len(
        pairs
    ):
        raise ValueError("Symmetric engine prompt pairs are not unique")
    if len(
        {(row["dashboard_prompt_a"], row["dashboard_prompt_b"]) for row in pairs}
    ) != len(pairs):
        raise ValueError("Symmetric dashboard prompt pairs are not unique")

    group_order = sorted(group_counts)
    rng = random.Random(seed)
    rng.shuffle(group_order)
    calibration_groups: list[str] = []
    calibration_count = 0
    for group in group_order:
        if calibration_count >= calibration_min_pairs:
            break
        calibration_groups.append(group)
        calibration_count += group_counts[group]
    calibration_set = set(calibration_groups)
    evaluation_groups = [group for group in group_order if group not in calibration_set]
    if len(evaluation_groups) < n_folds:
        raise ValueError("Too few held-out dependency groups for the requested folds")
    fold_by_group = {
        group: index % n_folds for index, group in enumerate(evaluation_groups)
    }
    for pair in pairs:
        group = str(pair["dependency_group"])
        pair["split"] = "calibration" if group in calibration_set else "evaluation"
        pair["fold"] = None if group in calibration_set else fold_by_group[group]

    return {
        "schema_version": "symmetric-causal-read-candidates-v1",
        "seed": seed,
        "source_schema_version": payload["schema_version"],
        "source": _supplement_source_id(Path(supplement_path)),
        "n_candidates": len(pairs),
        "n_dependency_groups": len(group_counts),
        "calibration_min_pairs": calibration_min_pairs,
        "n_calibration_pairs": sum(
            pair["split"] == "calibration" for pair in pairs
        ),
        "n_evaluation_pairs": sum(pair["split"] == "evaluation" for pair in pairs),
        "calibration_groups": calibration_groups,
        "evaluation_groups": evaluation_groups,
        "group_counts": group_counts,
        "n_folds": n_folds,
        "fold_by_group": fold_by_group,
        "pairs": pairs,
    }


def tokenize_symmetric_candidate(
    tokenizer: Any,
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach exact concept and answer token IDs to one reciprocal pair."""

    concept_a_id, concept_a_surface = concept_token_id(tokenizer, str(pair["concept_a"]))
    concept_b_id, concept_b_surface = concept_token_id(tokenizer, str(pair["concept_b"]))
    answer_a_id, answer_a_surface = continuation_token_id(
        tokenizer, str(pair["engine_prompt_a"]), str(pair["answer_a"])
    )
    answer_b_id, answer_b_surface = continuation_token_id(
        tokenizer, str(pair["engine_prompt_b"]), str(pair["answer_b"])
    )
    dashboard_id_a, dashboard_surface_a = continuation_token_id(
        tokenizer, str(pair["dashboard_prompt_a"]), str(pair["dashboard_answer"])
    )
    dashboard_id_b, dashboard_surface_b = continuation_token_id(
        tokenizer, str(pair["dashboard_prompt_b"]), str(pair["dashboard_answer"])
    )
    distractor_id_a, distractor_surface_a = continuation_token_id(
        tokenizer,
        str(pair["dashboard_prompt_a"]),
        str(pair["dashboard_distractor"]),
    )
    distractor_id_b, distractor_surface_b = continuation_token_id(
        tokenizer,
        str(pair["dashboard_prompt_b"]),
        str(pair["dashboard_distractor"]),
    )
    if concept_a_id == concept_b_id or answer_a_id == answer_b_id:
        raise ValueError(f"Collapsed concept or answer tokens for {pair['pair_id']}")
    if dashboard_id_a != dashboard_id_b or distractor_id_a != distractor_id_b:
        raise ValueError(f"Dashboard tokenization differs across {pair['pair_id']}")
    if dashboard_id_a == distractor_id_a:
        raise ValueError(f"Dashboard target and distractor collapse for {pair['pair_id']}")

    engine_ids_a = tokenizer.encode(str(pair["engine_prompt_a"]), add_special_tokens=False)
    engine_ids_b = tokenizer.encode(str(pair["engine_prompt_b"]), add_special_tokens=False)
    dashboard_ids_a = tokenizer.encode(
        str(pair["dashboard_prompt_a"]), add_special_tokens=False
    )
    dashboard_ids_b = tokenizer.encode(
        str(pair["dashboard_prompt_b"]), add_special_tokens=False
    )
    context_ids_a = tokenizer.encode(str(pair["context_a"]), add_special_tokens=False)
    context_ids_b = tokenizer.encode(str(pair["context_b"]), add_special_tokens=False)
    concept_prefix_ids_a = tokenizer.encode(
        str(pair["concept_prefix_a"]), add_special_tokens=False
    )
    concept_prefix_ids_b = tokenizer.encode(
        str(pair["concept_prefix_b"]), add_special_tokens=False
    )
    if not context_ids_a or not context_ids_b:
        raise ValueError(f"Empty shared context for {pair['pair_id']}")
    if (
        not concept_prefix_ids_a
        or not concept_prefix_ids_b
        or concept_prefix_ids_a[-1] != concept_a_id
        or concept_prefix_ids_b[-1] != concept_b_id
    ):
        raise ValueError(
            f"Appended concept is not one stable final prefix token: {pair['pair_id']}"
        )
    if (
        engine_ids_a[: len(context_ids_a)] != context_ids_a
        or dashboard_ids_a[: len(context_ids_a)] != context_ids_a
        or engine_ids_b[: len(context_ids_b)] != context_ids_b
        or dashboard_ids_b[: len(context_ids_b)] != context_ids_b
    ):
        raise ValueError(f"Shared context is not a stable token prefix for {pair['pair_id']}")
    if (
        engine_ids_a[: len(concept_prefix_ids_a)] != concept_prefix_ids_a
        or dashboard_ids_a[: len(concept_prefix_ids_a)] != concept_prefix_ids_a
        or engine_ids_b[: len(concept_prefix_ids_b)] != concept_prefix_ids_b
        or dashboard_ids_b[: len(concept_prefix_ids_b)] != concept_prefix_ids_b
    ):
        raise ValueError(f"Concept prefix is not stable in both tasks: {pair['pair_id']}")
    return {
        **pair,
        "concept_a_token_id": concept_a_id,
        "concept_b_token_id": concept_b_id,
        "concept_a_surface": concept_a_surface,
        "concept_b_surface": concept_b_surface,
        "answer_a_token_id": answer_a_id,
        "answer_b_token_id": answer_b_id,
        "answer_a_surface": answer_a_surface,
        "answer_b_surface": answer_b_surface,
        "dashboard_token_id": dashboard_id_a,
        "dashboard_surface_a": dashboard_surface_a,
        "dashboard_surface_b": dashboard_surface_b,
        "dashboard_distractor_token_id": distractor_id_a,
        "dashboard_distractor_surface_a": distractor_surface_a,
        "dashboard_distractor_surface_b": distractor_surface_b,
        "engine_n_tokens_a": len(engine_ids_a),
        "engine_n_tokens_b": len(engine_ids_b),
        "dashboard_n_tokens_a": len(dashboard_ids_a),
        "dashboard_n_tokens_b": len(dashboard_ids_b),
        "context_n_tokens_a": len(context_ids_a),
        "context_n_tokens_b": len(context_ids_b),
        "context_position_a": len(context_ids_a) - 1,
        "context_position_b": len(context_ids_b) - 1,
        "intervention_position_a": len(concept_prefix_ids_a) - 1,
        "intervention_position_b": len(concept_prefix_ids_b) - 1,
    }


def choose_written_threshold(
    own_scores: Sequence[float],
    foil_scores: Sequence[float],
    *,
    minimum_recall: float = 0.80,
) -> dict[str, float]:
    """Choose the frozen calibration-only WRITTEN threshold.

    Candidate thresholds are the unique observed scores. The winner maximizes
    balanced accuracy, then own-concept recall, then prefers the lower threshold,
    exactly matching the final experiment's preregistered tie-break.
    """

    if len(own_scores) != len(foil_scores) or not own_scores:
        raise ValueError("Own and foil WRITTEN scores must align and be nonempty")
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must lie in [0, 1]")
    own_array = np.asarray(own_scores, dtype=float)
    foil_array = np.asarray(foil_scores, dtype=float)
    if not np.isfinite(own_array).all() or not np.isfinite(foil_array).all():
        raise ValueError("WRITTEN calibration scores must be finite")
    threshold_rows: list[dict[str, float]] = []
    for threshold in np.unique(np.concatenate([own_array, foil_array])):
        recall = float(np.mean(own_array >= threshold))
        specificity = float(np.mean(foil_array < threshold))
        if recall < minimum_recall:
            continue
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "own_recall": recall,
                "foil_specificity": specificity,
                "balanced_accuracy": 0.5 * (recall + specificity),
            }
        )
    if not threshold_rows:
        raise RuntimeError(
            "No calibration WRITTEN threshold attains "
            f"recall >= {minimum_recall:.2f}"
        )
    return max(
        threshold_rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["own_recall"],
            -row["threshold"],
        ),
    )


def select_layer_record(
    layer_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the calibration layer using the frozen causal-sanity tie-break."""

    if not layer_records:
        raise ValueError("Layer selection requires at least one calibration record")
    required = {
        "layer",
        "causal_separation",
        "engine_abs_C_median",
        "own_greater_rate",
    }
    for record in layer_records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Layer selection record missing {sorted(missing)}")
        values = [
            float(record["causal_separation"]),
            float(record["engine_abs_C_median"]),
            float(record["own_greater_rate"]),
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Layer selection metrics must be finite")
    selected = max(
        layer_records,
        key=lambda row: (
            float(row["causal_separation"]),
            float(row["engine_abs_C_median"]),
            float(row["own_greater_rate"]),
            -int(row["layer"]),
        ),
    )
    return dict(selected)


def apply_symmetric_verification_gate(
    pairs: Sequence[Mapping[str, Any]],
    engine_records: Sequence[Mapping[str, Any]],
    dashboard_records: Sequence[Mapping[str, Any]],
    written_scores_by_pair: Sequence[tuple[float, float]],
    *,
    written_threshold: float,
) -> list[dict[str, Any]]:
    """Apply clean top-1 and WRITTEN gates without causal quantities."""

    expected_records = 2 * len(pairs)
    if (
        len(engine_records) != expected_records
        or len(dashboard_records) != expected_records
        or len(written_scores_by_pair) != len(pairs)
    ):
        raise ValueError("Pairs, flattened next-token records, and WRITTEN scores misalign")
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        row_a, row_b = 2 * index, 2 * index + 1
        z_a, z_b = (float(value) for value in written_scores_by_pair[index])
        engine_top1_a = int(engine_records[row_a]["rank"]) == 1
        engine_top1_b = int(engine_records[row_b]["rank"]) == 1
        dashboard_top1_a = int(dashboard_records[row_a]["rank"]) == 1
        dashboard_top1_b = int(dashboard_records[row_b]["rank"]) == 1
        engine_written_a = z_a >= written_threshold
        engine_written_b = z_b >= written_threshold
        engine_verified = all(
            (engine_top1_a, engine_top1_b, engine_written_a, engine_written_b)
        )
        control_verified = all(
            (engine_verified, dashboard_top1_a, dashboard_top1_b)
        )
        checks = {
            "ENGINE_A_TARGET_NOT_TOP1": engine_top1_a,
            "ENGINE_B_TARGET_NOT_TOP1": engine_top1_b,
            "ENGINE_A_CONCEPT_NOT_WRITTEN": engine_written_a,
            "ENGINE_B_CONCEPT_NOT_WRITTEN": engine_written_b,
            "DASHBOARD_A_TARGET_NOT_TOP1": dashboard_top1_a,
            "DASHBOARD_B_TARGET_NOT_TOP1": dashboard_top1_b,
            "DASHBOARD_A_CONCEPT_NOT_WRITTEN": engine_written_a,
            "DASHBOARD_B_CONCEPT_NOT_WRITTEN": engine_written_b,
        }
        reasons = [name for name, passed in checks.items() if not passed]
        if pair["split"] == "calibration":
            status = "CALIBRATION_ONLY"
        else:
            status = "VERIFIED" if control_verified else "UNVERIFIED"
        rows.append(
            {
                **pair,
                "verification_status": status,
                "verification_reasons": reasons,
                "engine_verified": engine_verified,
                "control_verified": control_verified,
                "engine_top1_a": engine_top1_a,
                "engine_top1_b": engine_top1_b,
                "dashboard_top1_a": dashboard_top1_a,
                "dashboard_top1_b": dashboard_top1_b,
                "engine_z_a": z_a,
                "engine_z_b": z_b,
                "dashboard_z_a": z_a,
                "dashboard_z_b": z_b,
                "written_threshold": float(written_threshold),
                "engine_top_token_id_a": int(
                    engine_records[row_a]["top_tokens"][0]["token_id"]
                ),
                "engine_top_token_id_b": int(
                    engine_records[row_b]["top_tokens"][0]["token_id"]
                ),
                "dashboard_top_token_id_a": int(
                    dashboard_records[row_a]["top_tokens"][0]["token_id"]
                ),
                "dashboard_top_token_id_b": int(
                    dashboard_records[row_b]["top_tokens"][0]["token_id"]
                ),
            }
        )
    return rows


def _single_token_id(tokenizer: Any, surface: str) -> int:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected one token for {surface!r}, got {token_ids}")
    return int(token_ids[0])


def _assert_calibration_anchors(rows: Sequence[Mapping[str, Any]]) -> None:
    calibration_groups = {
        str(row["dependency_group"])
        for row in rows
        if row.get("split") == "calibration"
    }
    missing = {
        template["anchor_dependency_group"]
        for template in HARD_DASHBOARD_TEMPLATES.values()
    } - calibration_groups
    if missing:
        raise ValueError(f"Hard-dashboard calibration anchors are absent: {sorted(missing)}")


def build_hard_dashboard_candidates(
    source_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Build fixed semantic controls from the frozen VERIFIED roster.

    Every hard prompt preserves the natural source context and explicit concept
    token byte-for-byte, but asks for an answer tied to a fixed calibration-only
    anchor. No causal or edited quantity is accepted by this constructor.
    """

    _assert_calibration_anchors(source_rows)
    verified = [
        row for row in source_rows if row.get("verification_status") == "VERIFIED"
    ]
    if not verified:
        raise ValueError("No frozen VERIFIED pairs are available")
    candidates: list[dict[str, Any]] = []
    for source in verified:
        category = str(source["category"])
        if category not in HARD_DASHBOARD_TEMPLATES:
            raise ValueError(f"No frozen hard-dashboard template for {category!r}")
        template = HARD_DASHBOARD_TEMPLATES[category]
        target_id = _single_token_id(tokenizer, template["target_surface"])
        distractor_id = _single_token_id(tokenizer, template["distractor_surface"])
        if target_id == distractor_id:
            raise ValueError("Hard-dashboard target and distractor must differ")

        hard_prompts: dict[str, str] = {}
        n_tokens: dict[str, int] = {}
        prefix_audits: dict[str, dict[str, Any]] = {}
        for side in ("a", "b"):
            context = str(source[f"context_{side}"])
            prompt = context + template["suffix"]
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            concept_prefix_ids = tokenizer.encode(
                str(source[f"concept_prefix_{side}"]), add_special_tokens=False
            )
            position = int(source[f"intervention_position_{side}"])
            concept_id = int(source[f"concept_{side}_token_id"])
            context_prefix_preserved = prompt_ids[: len(context_ids)] == context_ids
            concept_prefix_preserved = (
                prompt_ids[: len(concept_prefix_ids)] == concept_prefix_ids
            )
            concept_token_preserved = (
                0 <= position < len(prompt_ids) and int(prompt_ids[position]) == concept_id
            )
            if not (
                context_prefix_preserved
                and concept_prefix_preserved
                and concept_token_preserved
            ):
                raise ValueError(
                    f"Hard prompt changed frozen prefix/position for "
                    f"{source['pair_id']} {side}"
                )
            hard_prompts[side] = prompt
            n_tokens[side] = len(prompt_ids)
            prefix_audits[side] = {
                "context_prefix_preserved": context_prefix_preserved,
                "concept_prefix_preserved": concept_prefix_preserved,
                "concept_token_preserved": concept_token_preserved,
                "intervention_position": position,
                "concept_token_id": concept_id,
                "context_n_tokens": len(context_ids),
                "hard_prompt_n_tokens": len(prompt_ids),
            }

        candidates.append(
            {
                "pair_id": str(source["pair_id"]),
                "dependency_group": str(source["dependency_group"]),
                "fold": int(source["fold"]),
                "category": category,
                "concept_a": str(source["concept_a"]),
                "concept_b": str(source["concept_b"]),
                "concept_a_token_id": int(source["concept_a_token_id"]),
                "concept_b_token_id": int(source["concept_b_token_id"]),
                "concept_a_surface": str(source["concept_a_surface"]),
                "concept_b_surface": str(source["concept_b_surface"]),
                "intervention_position_a": int(source["intervention_position_a"]),
                "intervention_position_b": int(source["intervention_position_b"]),
                "hard_prompt_a": hard_prompts["a"],
                "hard_prompt_b": hard_prompts["b"],
                "hard_prompt_n_tokens_a": n_tokens["a"],
                "hard_prompt_n_tokens_b": n_tokens["b"],
                "hard_target": template["target"],
                "hard_target_surface": template["target_surface"],
                "hard_target_token_id": target_id,
                "hard_distractor": template["distractor"],
                "hard_distractor_surface": template["distractor_surface"],
                "hard_distractor_token_id": distractor_id,
                "hard_answer_type": template["answer_type"],
                "engine_answer_type_matched": True,
                "hard_relation": template["relation"],
                "hard_template_id": template["template_id"],
                "anchor_dependency_group": template["anchor_dependency_group"],
                "anchor_selected_from_calibration_only": True,
                "concept_irrelevance_contract": template["irrelevance"],
                "prefix_audit_a": prefix_audits["a"],
                "prefix_audit_b": prefix_audits["b"],
                "hard_z_a": float(source["engine_z_a"]),
                "hard_z_b": float(source["engine_z_b"]),
                "written_threshold": float(source["written_threshold"]),
                "written_provenance": (
                    "Frozen engine z reused because the hard prompt preserves the exact "
                    "causal prefix through the explicit concept token."
                ),
            }
        )
    return candidates


@torch.no_grad()
def verify_hard_dashboard_candidates(
    hf_model: torch.nn.Module,
    tokenizer: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Apply the frozen correctness and WRITTEN gates to hard dashboards."""

    if hf_model.training:
        raise ValueError("Hard-dashboard verification requires eval mode")
    if batch_size < 1 or top_k < 1:
        raise ValueError("batch_size and top_k must be positive")
    flattened: list[tuple[int, str, str, int]] = []
    for index, row in enumerate(candidates):
        for side in ("a", "b"):
            flattened.append(
                (
                    index,
                    side,
                    str(row[f"hard_prompt_{side}"]),
                    int(row["hard_target_token_id"]),
                )
            )

    device = next(hf_model.parameters()).device
    clean_records: dict[tuple[int, str], dict[str, Any]] = {}
    for start in range(0, len(flattened), batch_size):
        batch = flattened[start : start + batch_size]
        prompts = [entry[2] for entry in batch]
        tokenizer.padding_side = "right"
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        logits = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        for batch_index, (row_index, side, _prompt, expected_id) in enumerate(batch):
            positions = attention_mask[batch_index].nonzero(as_tuple=False).flatten()
            final_position = int(positions[-1])
            next_logits = logits[batch_index, final_position]
            top_values, top_ids = next_logits.topk(top_k)
            top_token_id = int(top_ids[0].cpu())
            expected_rank = int(
                (next_logits > next_logits[int(expected_id)]).sum().cpu().item() + 1
            )
            clean_records[(row_index, side)] = {
                "hard_top1": top_token_id == int(expected_id),
                "hard_target_rank": expected_rank,
                "hard_top_token_id": top_token_id,
                "hard_top_token": tokenizer.decode([top_token_id]),
                "hard_top_tokens": [
                    {
                        "token_id": int(token_id),
                        "token": tokenizer.decode([int(token_id)]),
                        "logit": float(value),
                    }
                    for value, token_id in zip(
                        top_values.detach().cpu(),
                        top_ids.detach().cpu(),
                        strict=True,
                    )
                ],
            }
        del logits, input_ids, attention_mask

    verified_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        row = dict(candidate)
        reasons: list[str] = []
        for side in ("a", "b"):
            row.update(
                {
                    f"{key}_{side}": value
                    for key, value in clean_records[(index, side)].items()
                }
            )
            written = float(row[f"hard_z_{side}"]) >= float(row["written_threshold"])
            row[f"hard_written_{side}"] = written
            if not row[f"hard_top1_{side}"]:
                reasons.append(f"HARD_{side.upper()}_TARGET_NOT_TOP1")
            if not written:
                reasons.append(f"HARD_{side.upper()}_CONCEPT_NOT_WRITTEN")
        row["verification_reasons"] = reasons
        row["verification_status"] = (
            "VERIFIED_HARD" if not reasons else "UNVERIFIED_HARD"
        )
        verified_rows.append(row)
    return verified_rows


_CAUSAL_OUTPUT_KEYS = {
    "c",
    "t",
    "normalization_t",
    "engine_c",
    "dashboard_c",
    "hard_c",
    "r_a_from_b",
    "r_b_from_a",
    "engine_r_a_from_b",
    "engine_r_b_from_a",
    "dashboard_r_a_from_b",
    "dashboard_r_b_from_a",
    "hard_r_a_from_b",
    "hard_r_b_from_a",
    "metric_a_from_b",
    "metric_b_from_a",
    "edited_metric",
    "edited_logits",
}


def _forbidden_manifest_paths(value: Any, *, path: str = "$") -> list[str]:
    """Return paths to causal/interchange outputs in a JSON-like value."""

    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            child_path = f"{path}.{key}"
            if normalized in _CAUSAL_OUTPUT_KEYS:
                violations.append(child_path)
            elif normalized.startswith("edited_") and normalized != "edited_metrics_included":
                violations.append(child_path)
            elif normalized.startswith("donor_") or normalized.startswith("patched_"):
                violations.append(child_path)
            violations.extend(_forbidden_manifest_paths(child, path=child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            violations.extend(_forbidden_manifest_paths(child, path=f"{path}[{index}]"))
    return violations


def validate_sanitized_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_model_id: str = MODEL_ID,
    expected_revision: str = MODEL_REVISION,
    require_rows: bool = True,
) -> dict[str, Any]:
    """Validate the clean-data firewall before any cheap READ computation.

    The returned shallow copy is safe to pass onward only after the recursive
    audit proves that no causal score, directional recovery, edited metric,
    patched value, or donor state is present.
    """

    if manifest.get("causal_interchange_outputs_included") is not False:
        raise ValueError("Manifest is not certified causal-interchange-output-free")
    if (
        "edited_metrics_included" in manifest
        and manifest.get("edited_metrics_included") is not False
    ):
        raise ValueError("Manifest is not certified edited-metric-free")
    violations = sorted(set(_forbidden_manifest_paths(manifest)))
    if violations:
        raise ValueError(f"Sanitized manifest contains forbidden fields: {violations}")
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Sanitized manifest requires model provenance")
    if str(model.get("id")) != expected_model_id:
        raise ValueError(
            f"Manifest model {model.get('id')!r} != expected {expected_model_id!r}"
        )
    if str(model.get("revision")) != expected_revision:
        raise ValueError("Manifest model revision differs from the pinned revision")
    dtype = str(model.get("dtype"))
    if dtype not in {"torch.bfloat16", "bfloat16"}:
        raise ValueError(f"Manifest dtype is not bf16: {dtype!r}")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Sanitized manifest requires frozen selection metadata")
    for key in ("layer", "position_rule", "written_threshold"):
        if key not in selection:
            raise ValueError(f"Sanitized manifest selection missing {key!r}")
    rows = manifest.get("rows")
    if require_rows and (not isinstance(rows, list) or not rows):
        raise ValueError("Sanitized manifest requires nonempty rows")
    if isinstance(rows, list):
        pair_ids = [str(row.get("pair_id")) for row in rows if isinstance(row, Mapping)]
        if len(pair_ids) != len(rows):
            raise ValueError("Every sanitized manifest row must be an object")
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("Sanitized manifest pair IDs must be unique")
    return dict(manifest)


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    payload: Mapping[str, Any],
) -> None:
    """Emit one small, JSON-friendly progress record when requested."""

    if callback is not None:
        callback(event, dict(payload))


def build_and_verify_dataset(
    bundle: Any,
    published_lens: Any,
    *,
    clean_state_and_logits: Callable[..., Mapping[str, Any]],
    symmetric_interchange: Callable[..., Mapping[str, Any]],
    token_difference_metric: Callable[[int, int], Callable[[torch.Tensor], torch.Tensor]],
    supplement_path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
    seed: int = SYMMETRIC_READ_SEED,
    layer_candidates: Sequence[int] = SYMMETRIC_LAYER_CANDIDATES,
    calibration_min_pairs: int = SYMMETRIC_CALIBRATION_MIN_PAIRS,
    n_folds: int = SYMMETRIC_N_FOLDS,
    kl_records_fn: Callable[..., Sequence[Mapping[str, Any]]] = hf_wrapper_logit_kl,
    kl_gate_fn: Callable[..., Mapping[str, Any]] = enforce_kl_gate,
    next_token_records_fn: Callable[..., Sequence[Mapping[str, Any]]] = (
        batched_next_token_records
    ),
    residual_matrices_fn: Callable[..., Mapping[int, torch.Tensor]] = (
        residual_prompt_matrices
    ),
    direction_bank_fn: Callable[..., Mapping[int, Mapping[int, torch.Tensor]]] = (
        jlens_direction_bank
    ),
    batch_size: int = 32,
    top_k: int = 5,
    kl_threshold: float = 1e-3,
    expected_candidate_count: int = 118,
    expected_calibration_count: int = 25,
    expected_evaluation_count: int = 93,
    expected_selected_layer: int = 16,
    expected_verified_count: int = 77,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the frozen dataset, layer-selection, and verification pipeline.

    Parameters
    ----------
    bundle, published_lens:
        An already-loaded pinned model bundle and its published Jacobian Lens.
        Loading and releasing GPU objects remain the caller's responsibility.
    clean_state_and_logits, symmetric_interchange, token_difference_metric:
        Injected causal primitives. They are used *only* on calibration groups
        to select the layer; this module deliberately does not import the
        intervention implementation.
    kl_records_fn, kl_gate_fn:
        Injectable HF/J-Lens agreement measurement and gate. The final run uses
        :func:`src.jlens_interface.hf_wrapper_logit_kl` and
        :func:`src.jlens_interface.enforce_kl_gate`.
    progress:
        Optional ``callback(event, payload)`` invoked after bounded stages and
        each candidate layer. Payloads contain scalars only.

    Returns
    -------
    dict
        Three caller-owned, in-memory payloads under ``full_dataset_artifact``,
        ``sanitized_manifest``, and ``direction_cache``. The manifest contains
        no causal/interchange output. Direction tensors are detached fp32 CPU
        tensors. The caller is responsible for serialization, paths, sizes,
        and content hashes, so no machine-local artifact reference is emitted.

    Notes
    -----
    The defaults reproduce the successful run: 20 KL prompts, 118 reciprocal
    candidates, layers 13--26, whole-group calibration, calibration-only
    symmetric full-residual causal selection, and the clean top-1 plus WRITTEN
    gate. Causal calibration rows stay exclusively in the full artifact.
    """

    if batch_size < 1 or top_k < 1:
        raise ValueError("batch_size and top_k must be positive")
    if min(
        expected_candidate_count,
        expected_calibration_count,
        expected_evaluation_count,
        expected_verified_count,
    ) < 1:
        raise ValueError("Expected frozen counts must be positive")
    layers = sorted(set(int(layer) for layer in layer_candidates))
    if not layers:
        raise ValueError("At least one layer candidate is required")
    if len(layers) != len(layer_candidates):
        raise ValueError("Layer candidates must be unique")
    set_seed(seed)
    for name, callback in (
        ("clean_state_and_logits", clean_state_and_logits),
        ("symmetric_interchange", symmetric_interchange),
        ("token_difference_metric", token_difference_metric),
        ("kl_records_fn", kl_records_fn),
        ("kl_gate_fn", kl_gate_fn),
        ("next_token_records_fn", next_token_records_fn),
        ("residual_matrices_fn", residual_matrices_fn),
        ("direction_bank_fn", direction_bank_fn),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")

    hf_model = bundle.hf_model
    tokenizer = bundle.tokenizer
    lens_model = bundle.lens_model
    if hf_model.training:
        raise ValueError("Dataset construction requires the model in eval mode")
    model_parameter = next(hf_model.parameters())
    if model_parameter.dtype != torch.bfloat16:
        raise ValueError(
            "The frozen dataset protocol requires a bf16 model; "
            f"got {model_parameter.dtype}"
        )
    model_id = str(getattr(bundle, "model_id", ""))
    model_revision = str(getattr(bundle, "revision", ""))
    if model_id != MODEL_ID or model_revision != MODEL_REVISION:
        raise ValueError(
            "Loaded model does not match the pinned final protocol: "
            f"{model_id!r}@{model_revision!r}"
        )
    lens_layers = {int(layer) for layer in published_lens.source_layers}
    missing_lens_layers = sorted(set(layers) - lens_layers)
    if missing_lens_layers:
        raise ValueError(
            f"Published lens does not cover candidate layers {missing_lens_layers}"
        )

    protocol = {
        "schema_version": "symmetric-causal-read-v1",
        "seed": int(seed),
        "model": {
            "id": model_id,
            "revision": model_revision,
            "dtype": str(model_parameter.dtype),
        },
        "candidate_source": "tracked reciprocal two-hop supplement",
        "candidate_count_required_min": 100,
        "calibration_group_rule": (
            "shuffle unordered concept dependency groups; take whole groups "
            f"until >={int(calibration_min_pairs)} pairs"
        ),
        "evaluation_folds": int(n_folds),
        "position_rule": SYMMETRIC_POSITION_RULE,
        "layer_candidates": layers,
        "layer_selection": (
            "calibration maximum median(|C_engine|)-median(|C_dashboard|), then "
            "engine median |C|, own>foil rate, and lower layer"
        ),
        "written_threshold": (
            "calibration maximum balanced accuracy with own recall>=0.80; "
            "then higher recall and lower threshold"
        ),
        "verification_gate": (
            "both engine targets clean top-1; own concept WRITTEN in both engine "
            "runs; dashboard target top-1 and concept WRITTEN in both "
            "same-context controls"
        ),
        "causal_truth": "signed symmetric full residual interchange; unclipped",
        "cheap_primary": "16-midpoint direction-defined integrated gradient",
        "go_rule": "READ_IG AUC>=0.70 and group-bootstrap CI95 lower>0.50",
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    kl_records = [dict(row) for row in kl_records_fn(bundle, list(G1_PROMPTS))]
    if len(G1_PROMPTS) != 20 or len(kl_records) != 20:
        raise RuntimeError(
            "HF/J-Lens agreement gate requires exactly 20 prompt records; "
            f"got {len(kl_records)}"
        )
    max_mean_kl = max(float(row["mean_kl"]) for row in kl_records)
    if not math.isfinite(max_mean_kl) or max_mean_kl >= kl_threshold:
        raise RuntimeError(
            "HF/J-Lens logit gate failed: "
            f"max mean KL={max_mean_kl!r}, required <{kl_threshold}"
        )
    logit_agreement = dict(
        kl_gate_fn(kl_records, threshold=float(kl_threshold))
    )
    if str(logit_agreement.get("status")) != "PASS":
        raise RuntimeError("Injected HF/J-Lens KL gate did not return PASS")
    _emit_progress(
        progress,
        "kl_gate",
        {"n": len(kl_records), "max_mean_kl": max_mean_kl, "status": "PASS"},
    )

    candidate_manifest = build_symmetric_causal_candidates(
        supplement_path,
        seed=seed,
        calibration_min_pairs=calibration_min_pairs,
        n_folds=n_folds,
    )
    if int(candidate_manifest["n_candidates"]) != expected_candidate_count:
        raise RuntimeError(
            "Frozen candidate count changed: "
            f"expected {expected_candidate_count}, "
            f"got {candidate_manifest['n_candidates']}"
        )
    if (
        int(candidate_manifest["n_calibration_pairs"])
        != expected_calibration_count
        or int(candidate_manifest["n_evaluation_pairs"])
        != expected_evaluation_count
    ):
        raise RuntimeError(
            "Frozen split counts changed: expected "
            f"{expected_calibration_count} calibration and "
            f"{expected_evaluation_count} evaluation, got "
            f"{candidate_manifest['n_calibration_pairs']} and "
            f"{candidate_manifest['n_evaluation_pairs']}"
        )
    tokenized_pairs: list[dict[str, Any]] = []
    tokenization_rejections: list[dict[str, Any]] = []
    for pair in candidate_manifest["pairs"]:
        try:
            tokenized_pairs.append(tokenize_symmetric_candidate(tokenizer, pair))
        except (ValueError, IndexError) as error:
            tokenization_rejections.append(
                {
                    **pair,
                    "verification_status": (
                        "CALIBRATION_ONLY"
                        if pair["split"] == "calibration"
                        else "UNVERIFIED"
                    ),
                    "verification_reasons": [f"TOKENIZATION_FAILURE: {error}"],
                    "engine_verified": False,
                    "control_verified": False,
                }
            )
    if not tokenized_pairs:
        raise RuntimeError("Every frozen candidate failed exact tokenization")
    _emit_progress(
        progress,
        "candidates",
        {
            "n_candidates": int(candidate_manifest["n_candidates"]),
            "n_calibration_pairs": int(candidate_manifest["n_calibration_pairs"]),
            "n_evaluation_pairs": int(candidate_manifest["n_evaluation_pairs"]),
            "n_tokenized": len(tokenized_pairs),
            "n_tokenization_rejections": len(tokenization_rejections),
        },
    )

    engine_prompts = [
        str(prompt)
        for pair in tokenized_pairs
        for prompt in (pair["engine_prompt_a"], pair["engine_prompt_b"])
    ]
    engine_targets = [
        int(token_id)
        for pair in tokenized_pairs
        for token_id in (pair["answer_a_token_id"], pair["answer_b_token_id"])
    ]
    dashboard_prompts = [
        str(prompt)
        for pair in tokenized_pairs
        for prompt in (pair["dashboard_prompt_a"], pair["dashboard_prompt_b"])
    ]
    dashboard_targets = [
        int(pair["dashboard_token_id"])
        for pair in tokenized_pairs
        for _ in range(2)
    ]
    engine_records = [
        dict(row)
        for row in next_token_records_fn(
            hf_model,
            tokenizer,
            engine_prompts,
            engine_targets,
            batch_size=batch_size,
            top_k=top_k,
        )
    ]
    dashboard_records = [
        dict(row)
        for row in next_token_records_fn(
            hf_model,
            tokenizer,
            dashboard_prompts,
            dashboard_targets,
            batch_size=batch_size,
            top_k=top_k,
        )
    ]
    context_prompts = [
        str(prompt)
        for pair in tokenized_pairs
        for prompt in (pair["concept_prefix_a"], pair["concept_prefix_b"])
    ]
    context_matrices = dict(
        residual_matrices_fn(
            lens_model,
            context_prompts,
            layers,
            position=-1,
            batch_size=batch_size,
        )
    )
    concept_token_ids = sorted(
        {
            int(token_id)
            for pair in tokenized_pairs
            for token_id in (
                pair["concept_a_token_id"],
                pair["concept_b_token_id"],
            )
        }
    )
    direction_bank = direction_bank_fn(
        published_lens,
        lens_model,
        concept_token_ids,
        layers,
        compute_device=model_parameter.device,
        output_device="cpu",
    )
    _emit_progress(
        progress,
        "clean_measurements",
        {
            "n_engine_runs": len(engine_records),
            "n_dashboard_runs": len(dashboard_records),
            "n_context_runs": len(context_prompts),
            "n_concept_directions": len(concept_token_ids),
            "n_layers": len(layers),
        },
    )

    calibration_indices = [
        index
        for index, pair in enumerate(tokenized_pairs)
        if pair["split"] == "calibration"
    ]
    if not calibration_indices:
        raise RuntimeError("No calibration pairs survived exact tokenization")
    layer_selection_rows: list[dict[str, Any]] = []
    for layer in layers:
        own_scores: list[float] = []
        foil_scores: list[float] = []
        matrix = context_matrices[layer]
        for index in calibration_indices:
            pair = tokenized_pairs[index]
            row_a, row_b = 2 * index, 2 * index + 1
            vector_a = direction_bank[int(pair["concept_a_token_id"])][layer]
            vector_b = direction_bank[int(pair["concept_b_token_id"])][layer]
            own_scores.extend(
                [
                    float(torch.dot(matrix[row_a], vector_a)),
                    float(torch.dot(matrix[row_b], vector_b)),
                ]
            )
            foil_scores.extend(
                [
                    float(torch.dot(matrix[row_a], vector_b)),
                    float(torch.dot(matrix[row_b], vector_a)),
                ]
            )
        margins = np.asarray(own_scores) - np.asarray(foil_scores)
        layer_selection_rows.append(
            {
                "layer": layer,
                "n_calibration_runs": len(margins),
                "own_greater_rate": float(np.mean(margins > 0.0)),
                "median_own_minus_foil": float(np.median(margins)),
                "mean_own_minus_foil": float(np.mean(margins)),
                "calibration_own_scores": own_scores,
                "calibration_foil_scores": foil_scores,
                "threshold_record": choose_written_threshold(
                    own_scores, foil_scores
                ),
            }
        )

    def encode(prompt: str) -> torch.Tensor:
        return tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model_parameter.device)

    for layer_record in layer_selection_rows:
        layer = int(layer_record["layer"])
        threshold = float(layer_record["threshold_record"]["threshold"])
        matrix = context_matrices[layer]
        calibration_causal_rows: list[dict[str, Any]] = []
        for index in calibration_indices:
            pair = tokenized_pairs[index]
            row_a, row_b = 2 * index, 2 * index + 1
            vector_a = direction_bank[int(pair["concept_a_token_id"])][layer]
            vector_b = direction_bank[int(pair["concept_b_token_id"])][layer]
            written = (
                float(torch.dot(matrix[row_a], vector_a)) >= threshold
                and float(torch.dot(matrix[row_b], vector_b)) >= threshold
            )
            clean_correct = all(
                (
                    int(engine_records[row_a]["rank"]) == 1,
                    int(engine_records[row_b]["rank"]) == 1,
                    int(dashboard_records[row_a]["rank"]) == 1,
                    int(dashboard_records[row_b]["rank"]) == 1,
                )
            )
            if not written or not clean_correct:
                continue

            position_a = int(pair["intervention_position_a"])
            position_b = int(pair["intervention_position_b"])
            engine_ids_a = encode(str(pair["engine_prompt_a"]))
            engine_ids_b = encode(str(pair["engine_prompt_b"]))
            engine_clean_a = clean_state_and_logits(
                hf_model,
                lens_model.layers,
                engine_ids_a,
                layer,
                position=position_a,
            )
            engine_clean_b = clean_state_and_logits(
                hf_model,
                lens_model.layers,
                engine_ids_b,
                layer,
                position=position_b,
            )
            engine = symmetric_interchange(
                hf_model,
                lens_model.layers,
                engine_ids_a,
                engine_ids_b,
                engine_clean_a,
                engine_clean_b,
                token_difference_metric(
                    int(pair["answer_a_token_id"]),
                    int(pair["answer_b_token_id"]),
                ),
                pair_id=str(pair["pair_id"]),
                task_kind="engine",
                layer=layer,
                position_a=position_a,
                position_b=position_b,
            )

            dashboard_ids_a = encode(str(pair["dashboard_prompt_a"]))
            dashboard_ids_b = encode(str(pair["dashboard_prompt_b"]))
            dashboard_clean_a = clean_state_and_logits(
                hf_model,
                lens_model.layers,
                dashboard_ids_a,
                layer,
                position=position_a,
            )
            dashboard_clean_b = clean_state_and_logits(
                hf_model,
                lens_model.layers,
                dashboard_ids_b,
                layer,
                position=position_b,
            )
            dashboard = symmetric_interchange(
                hf_model,
                lens_model.layers,
                dashboard_ids_a,
                dashboard_ids_b,
                dashboard_clean_a,
                dashboard_clean_b,
                token_difference_metric(
                    int(pair["dashboard_token_id"]),
                    int(pair["dashboard_distractor_token_id"]),
                ),
                pair_id=str(pair["pair_id"]),
                task_kind="dashboard",
                layer=layer,
                normalization_t=float(engine["T"]),
                position_a=position_a,
                position_b=position_b,
            )
            calibration_causal_rows.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "engine_C": float(engine["C"]),
                    "dashboard_C": float(dashboard["C"]),
                    "engine_R_a_from_b": float(engine["R_a_from_b"]),
                    "engine_R_b_from_a": float(engine["R_b_from_a"]),
                    "dashboard_R_a_from_b": float(dashboard["R_a_from_b"]),
                    "dashboard_R_b_from_a": float(dashboard["R_b_from_a"]),
                }
            )
        if len(calibration_causal_rows) < 10:
            raise RuntimeError(
                f"Layer {layer} has too few verified calibration pairs: "
                f"{len(calibration_causal_rows)}"
            )
        engine_abs = np.abs(
            [row["engine_C"] for row in calibration_causal_rows]
        )
        dashboard_abs = np.abs(
            [row["dashboard_C"] for row in calibration_causal_rows]
        )
        layer_record["calibration_causal_rows"] = calibration_causal_rows
        layer_record["n_causal_calibration_pairs"] = len(calibration_causal_rows)
        layer_record["engine_abs_C_median"] = float(np.median(engine_abs))
        layer_record["dashboard_abs_C_median"] = float(np.median(dashboard_abs))
        layer_record["causal_separation"] = float(
            np.median(engine_abs) - np.median(dashboard_abs)
        )
        _emit_progress(
            progress,
            "calibration_layer",
            {
                "layer": layer,
                "n_pairs": len(calibration_causal_rows),
                "engine_abs_C_median": layer_record["engine_abs_C_median"],
                "dashboard_abs_C_median": layer_record[
                    "dashboard_abs_C_median"
                ],
                "causal_separation": layer_record["causal_separation"],
            },
        )

    selected_record = select_layer_record(layer_selection_rows)
    selected_layer = int(selected_record["layer"])
    if selected_layer != int(expected_selected_layer):
        raise RuntimeError(
            "Calibration-only tie rule no longer selects the frozen layer: "
            f"expected L{expected_selected_layer}, got L{selected_layer}"
        )
    threshold_record = dict(selected_record["threshold_record"])
    written_threshold = float(threshold_record["threshold"])
    _emit_progress(
        progress,
        "selection",
        {
            "layer": selected_layer,
            "written_threshold": written_threshold,
            "causal_separation": float(selected_record["causal_separation"]),
        },
    )

    selected_context = context_matrices[selected_layer]
    written_scores_by_pair: list[tuple[float, float]] = []
    for index, pair in enumerate(tokenized_pairs):
        row_a, row_b = 2 * index, 2 * index + 1
        vector_a = direction_bank[int(pair["concept_a_token_id"])][selected_layer]
        vector_b = direction_bank[int(pair["concept_b_token_id"])][selected_layer]
        written_scores_by_pair.append(
            (
                float(torch.dot(selected_context[row_a], vector_a)),
                float(torch.dot(selected_context[row_b], vector_b)),
            )
        )
    verification_rows = apply_symmetric_verification_gate(
        tokenized_pairs,
        engine_records,
        dashboard_records,
        written_scores_by_pair,
        written_threshold=written_threshold,
    )
    verification_rows.extend(tokenization_rejections)
    verification_rows.sort(key=lambda row: str(row["pair_id"]))
    counts = {
        "candidates": len(verification_rows),
        "calibration_pairs": sum(
            row["verification_status"] == "CALIBRATION_ONLY"
            for row in verification_rows
        ),
        "evaluation_pairs": sum(
            row["split"] == "evaluation" for row in verification_rows
        ),
        "verified_pairs": sum(
            row["verification_status"] == "VERIFIED"
            for row in verification_rows
        ),
        "unverified_pairs": sum(
            row["verification_status"] == "UNVERIFIED"
            for row in verification_rows
        ),
        "engine_verified_evaluation_pairs": sum(
            row["split"] == "evaluation" and bool(row["engine_verified"])
            for row in verification_rows
        ),
    }
    if counts["candidates"] != expected_candidate_count:
        raise AssertionError("Verification roster no longer covers every candidate")
    if (
        counts["calibration_pairs"] != expected_calibration_count
        or counts["evaluation_pairs"] != expected_evaluation_count
        or counts["verified_pairs"] != expected_verified_count
    ):
        raise RuntimeError(
            "Frozen verification counts changed: expected "
            f"calibration={expected_calibration_count}, "
            f"evaluation={expected_evaluation_count}, "
            f"verified={expected_verified_count}; got {counts}"
        )
    _emit_progress(progress, "verification", counts)

    selected_directions = {
        int(token_id): direction_bank[int(token_id)][selected_layer]
        .detach()
        .float()
        .cpu()
        .contiguous()
        for token_id in concept_token_ids
    }
    direction_cache = {
        "schema_version": "symmetric-selected-directions-v1",
        "protocol_sha256": protocol_sha256,
        "model_id": model_id,
        "model_revision": model_revision,
        "selected_layer": selected_layer,
        "directions": selected_directions,
    }
    direction_cache_metadata = {
        "schema_version": direction_cache["schema_version"],
        "protocol_sha256": protocol_sha256,
        "model_id": model_id,
        "model_revision": model_revision,
        "selected_layer": selected_layer,
        "n_directions": len(selected_directions),
        "token_ids": concept_token_ids,
        "storage": "caller_managed",
    }

    sanitized_manifest = {
        "schema_version": "symmetric-clean-read-manifest-v1",
        "protocol_sha256": protocol_sha256,
        "model": {
            "id": model_id,
            "revision": model_revision,
            "dtype": str(model_parameter.dtype),
        },
        "selection": {
            "layer": selected_layer,
            "position_rule": SYMMETRIC_POSITION_RULE,
            "written_threshold": written_threshold,
        },
        "counts": counts,
        "rows": copy.deepcopy(verification_rows),
        "direction_cache": direction_cache_metadata,
        "causal_interchange_outputs_included": False,
        "edited_metrics_included": False,
    }
    sanitized_manifest = validate_sanitized_manifest(sanitized_manifest)

    full_dataset_artifact = {
        "schema_version": "symmetric-dataset-verification-v1",
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "model": {
            "id": model_id,
            "revision": model_revision,
            "dtype": str(model_parameter.dtype),
        },
        "logit_agreement": logit_agreement,
        "candidate_manifest": {
            key: value
            for key, value in candidate_manifest.items()
            if key != "pairs"
        },
        "tokenization_rejections": tokenization_rejections,
        "selection": {
            "layer": selected_layer,
            "position_rule": SYMMETRIC_POSITION_RULE,
            "layer_candidates": layer_selection_rows,
            "written_threshold": written_threshold,
            "threshold_record": threshold_record,
            "calibration_own_scores": selected_record[
                "calibration_own_scores"
            ],
            "calibration_foil_scores": selected_record[
                "calibration_foil_scores"
            ],
        },
        "counts": counts,
        "rows": copy.deepcopy(verification_rows),
        "direction_cache": direction_cache_metadata,
        "clean_read_manifest": {
            "schema_version": sanitized_manifest["schema_version"],
            "causal_interchange_outputs_included": False,
            "edited_metrics_included": False,
            "storage": "caller_managed",
        },
    }
    return {
        "full_dataset_artifact": full_dataset_artifact,
        "sanitized_manifest": sanitized_manifest,
        "direction_cache": direction_cache,
    }

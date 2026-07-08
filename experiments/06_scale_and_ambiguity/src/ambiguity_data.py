"""Strict loading and tokenization for the frozen ambiguity benchmark."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_AMBIGUITY_SPEC = (
    Path(__file__).resolve().parents[1] / "data" / "specs" / "ambiguity.json"
)
SCHEMA_VERSION = "1.0.0"
DATASET_NAME = "ambiguity_flagship"
DATASET_SEED = 1729
PROBE_TEMPLATE = (
    "Sentence: {sentence}\n"
    "Question: {question}\n"
    "A: {choice_for_A}\n"
    "B: {choice_for_B}\n"
    "Answer with exactly one letter.\n"
    "Answer:"
)
CATEGORY_COUNTS = {
    "lexical_ambiguity": 30,
    "pp_attachment": 30,
    "garden_path": 30,
    "ambiguous_pronoun": 30,
}
CATEGORY_ID_PREFIXES = {
    "lexical_ambiguity": "lexical",
    "pp_attachment": "pp_attachment",
    "garden_path": "garden_path",
    "ambiguous_pronoun": "ambiguous_pronoun",
}
CONSTRUCTION_COUNTS = {
    "garden_path": {
        "reduced_relative": 10,
        "np_z": 10,
        "gerund_participle": 10,
    },
    "ambiguous_pronoun": {
        "singular_they_subject": 10,
        "singular_they_possessive": 10,
        "singular_they_object": 10,
    },
}

_ROOT_KEYS = {"schema_version", "dataset", "seed", "provenance", "design", "items"}
_PROVENANCE_KEYS = {
    "origin",
    "method",
    "external_corpus",
    "canonical_examples",
    "license_note",
}
_DESIGN_KEYS = {
    "item_count",
    "category_counts",
    "id_contract",
    "deduplication",
    "concept_label_contract",
    "garden_path_reading_contract",
    "tokenizer_validation",
    "behavior_probe_contract",
}
_TOKENIZER_VALIDATION_KEYS = {
    "model_id",
    "revision",
    "concept_method",
    "probe_method",
    "concept_item_readings_passed",
    "probe_item_labels_passed",
    "model_inference_used",
}
_PROBE_CONTRACT_KEYS = {
    "type",
    "render_template",
    "primary_metric",
    "token_boundary",
    "counterbalancing",
    "interpretation",
}
_ITEM_BASE_KEYS = {"id", "category", "sentence", "readings", "behavior_probe"}
_READING_KEYS = {"id", "description", "concept_label"}
_PROBE_KEYS = {"type", "question", "target", "foil"}
_CHOICE_KEYS = {"reading_id", "token_label", "choice"}
_CONCEPT_LABEL = re.compile(r"[a-z]+")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual, key=repr)
    extra = sorted(actual - expected, key=repr)
    _require(not missing and not extra, f"{path} keys: missing={missing}, extra={extra}")


def _require_nonempty_text(value: Any, path: str) -> str:
    _require(isinstance(value, str), f"{path} must be a string")
    _require(bool(value.strip()), f"{path} must not be empty")
    _require(value == value.strip(), f"{path} must not have outer whitespace")
    return value


def _normalized_sentence(sentence: str) -> str:
    normalized = unicodedata.normalize("NFKC", sentence)
    return " ".join(normalized.casefold().split())


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_item(item_value: Any, *, path: str) -> Mapping[str, Any]:
    item = _require_mapping(item_value, path)
    category = item.get("category")
    _require(
        isinstance(category, str) and category in CATEGORY_COUNTS,
        f"{path}.category is not recognized",
    )
    expected_keys = set(_ITEM_BASE_KEYS)
    if category in CONSTRUCTION_COUNTS:
        expected_keys.add("construction")
    _require_exact_keys(item, expected_keys, path)

    item_id = _require_nonempty_text(item["id"], f"{path}.id")
    prefix = CATEGORY_ID_PREFIXES[category]
    _require(
        re.fullmatch(rf"{re.escape(prefix)}_\d{{3}}", item_id) is not None,
        f"{path}.id {item_id!r} does not match its category prefix",
    )
    sentence = _require_nonempty_text(item["sentence"], f"{path}.sentence")
    _require(sentence.endswith((".", "?", "!")), f"{path}.sentence lacks punctuation")

    if category in CONSTRUCTION_COUNTS:
        construction = item["construction"]
        _require(
            isinstance(construction, str)
            and construction in CONSTRUCTION_COUNTS[category],
            f"{path}.construction is not recognized for {category}",
        )

    readings = item["readings"]
    _require(isinstance(readings, list), f"{path}.readings must be an array")
    _require(len(readings) == 2, f"{path}.readings must contain exactly two readings")
    for reading_index, expected_id in enumerate(("r1", "r2")):
        reading_path = f"{path}.readings[{reading_index}]"
        reading = _require_mapping(readings[reading_index], reading_path)
        _require_exact_keys(reading, _READING_KEYS, reading_path)
        _require(reading["id"] == expected_id, f"{reading_path}.id must be {expected_id}")
        _require_nonempty_text(reading["description"], f"{reading_path}.description")
        concept = _require_nonempty_text(
            reading["concept_label"], f"{reading_path}.concept_label"
        )
        _require(
            _CONCEPT_LABEL.fullmatch(concept) is not None,
            f"{reading_path}.concept_label must contain only lowercase ASCII letters",
        )
    _require(
        readings[0]["concept_label"] != readings[1]["concept_label"],
        f"{path} concept labels must be distinct",
    )

    probe = _require_mapping(item["behavior_probe"], f"{path}.behavior_probe")
    _require_exact_keys(probe, _PROBE_KEYS, f"{path}.behavior_probe")
    _require(probe["type"] == "forced_choice", f"{path} probe must be forced_choice")
    question = _require_nonempty_text(probe["question"], f"{path}.behavior_probe.question")
    _require(question.endswith("?"), f"{path}.behavior_probe.question must end in ?")
    for role, reading_id, token_label in (("target", "r1", "A"), ("foil", "r2", "B")):
        choice_path = f"{path}.behavior_probe.{role}"
        choice = _require_mapping(probe[role], choice_path)
        _require_exact_keys(choice, _CHOICE_KEYS, choice_path)
        _require(
            choice["reading_id"] == reading_id,
            f"{choice_path}.reading_id must be {reading_id}",
        )
        _require(
            choice["token_label"] == token_label,
            f"{choice_path}.token_label must be {token_label}",
        )
        _require_nonempty_text(choice["choice"], f"{choice_path}.choice")
    _require(
        probe["target"]["choice"].casefold() != probe["foil"]["choice"].casefold(),
        f"{path} target and foil choices must differ",
    )
    return item


def validate_ambiguity_spec(payload_value: Any) -> None:
    """Validate the complete frozen spec, rejecting omissions and extensions."""

    payload = _require_mapping(payload_value, "spec")
    _require_exact_keys(payload, _ROOT_KEYS, "spec")
    _require(payload["schema_version"] == SCHEMA_VERSION, "unsupported schema_version")
    _require(payload["dataset"] == DATASET_NAME, "unexpected dataset name")
    _require(
        isinstance(payload["seed"], int) and not isinstance(payload["seed"], bool),
        "spec.seed must be an integer",
    )
    _require(payload["seed"] == DATASET_SEED, f"spec.seed must be {DATASET_SEED}")

    provenance = _require_mapping(payload["provenance"], "spec.provenance")
    _require_exact_keys(provenance, _PROVENANCE_KEYS, "spec.provenance")
    for key in _PROVENANCE_KEYS - {"external_corpus"}:
        _require_nonempty_text(provenance[key], f"spec.provenance.{key}")
    _require(provenance["external_corpus"] is None, "external_corpus must be null")

    design = _require_mapping(payload["design"], "spec.design")
    _require_exact_keys(design, _DESIGN_KEYS, "spec.design")
    expected_total = sum(CATEGORY_COUNTS.values())
    _require(design["item_count"] == expected_total, "design.item_count is not frozen")
    _require(design["category_counts"] == CATEGORY_COUNTS, "category_counts changed")
    for key in {
        "id_contract",
        "deduplication",
        "concept_label_contract",
        "garden_path_reading_contract",
    }:
        _require_nonempty_text(design[key], f"spec.design.{key}")

    tokenizer_validation = _require_mapping(
        design["tokenizer_validation"], "spec.design.tokenizer_validation"
    )
    _require_exact_keys(
        tokenizer_validation,
        _TOKENIZER_VALIDATION_KEYS,
        "spec.design.tokenizer_validation",
    )
    for key in {"model_id", "revision", "concept_method", "probe_method"}:
        _require_nonempty_text(
            tokenizer_validation[key], f"spec.design.tokenizer_validation.{key}"
        )
    _require(
        tokenizer_validation["concept_item_readings_passed"] == 2 * expected_total,
        "frozen concept tokenizer count changed",
    )
    _require(
        tokenizer_validation["probe_item_labels_passed"] == 2 * expected_total,
        "frozen probe tokenizer count changed",
    )
    _require(
        tokenizer_validation["model_inference_used"] is False,
        "tokenizer validation must not claim model inference",
    )

    probe_contract = _require_mapping(
        design["behavior_probe_contract"], "spec.design.behavior_probe_contract"
    )
    _require_exact_keys(
        probe_contract, _PROBE_CONTRACT_KEYS, "spec.design.behavior_probe_contract"
    )
    _require(probe_contract["type"] == "forced_choice", "probe contract type changed")
    _require(
        probe_contract["render_template"] == PROBE_TEMPLATE,
        "probe render template changed",
    )
    for key in _PROBE_CONTRACT_KEYS - {"type", "render_template"}:
        _require_nonempty_text(
            probe_contract[key], f"spec.design.behavior_probe_contract.{key}"
        )

    items = payload["items"]
    _require(isinstance(items, list), "spec.items must be an array")
    _require(len(items) == expected_total, f"spec.items must contain {expected_total} items")
    for index, item in enumerate(items):
        _validate_item(item, path=f"spec.items[{index}]")

    expected_ids = [
        f"{CATEGORY_ID_PREFIXES[category]}_{number:03d}"
        for category, count in CATEGORY_COUNTS.items()
        for number in range(1, count + 1)
    ]
    actual_ids = [item["id"] for item in items]
    _require(actual_ids == expected_ids, "item IDs or their frozen order changed")
    actual_counts = Counter(item["category"] for item in items)
    _require(actual_counts == CATEGORY_COUNTS, "item category balance changed")

    normalized_sentences = [_normalized_sentence(item["sentence"]) for item in items]
    _require(
        len(normalized_sentences) == len(set(normalized_sentences)),
        "duplicate normalized sentences found",
    )
    for category, expected in CONSTRUCTION_COUNTS.items():
        actual = Counter(
            item["construction"] for item in items if item["category"] == category
        )
        _require(actual == expected, f"{category} construction balance changed")


def load_ambiguity_spec(
    path: str | Path = DEFAULT_AMBIGUITY_SPEC,
) -> dict[str, Any]:
    """Load and strictly validate the authored JSON specification."""

    spec_path = Path(path)
    try:
        with spec_path.open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_unique_json_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load ambiguity spec at {spec_path}: {error}") from error
    validate_ambiguity_spec(payload)
    return payload


def load_ambiguity_items(
    path: str | Path = DEFAULT_AMBIGUITY_SPEC,
) -> list[dict[str, Any]]:
    """Return the validated frozen items in their stable source order."""

    payload = load_ambiguity_spec(path)
    return payload["items"]


def render_ambiguity_probe(item_value: Any, *, mirrored: bool = False) -> dict[str, Any]:
    """Render one original or mirrored probe with explicit orientation metadata."""

    item = _validate_item(item_value, path="item")
    _require(isinstance(mirrored, bool), "mirrored must be a boolean")
    probe = item["behavior_probe"]
    if mirrored:
        choice_for_a = probe["foil"]["choice"]
        choice_for_b = probe["target"]["choice"]
        reading_to_label = {"r1": "B", "r2": "A"}
        variant = "mirrored"
        fixed_ab_margin_sign = -1
        variant_index = 1
    else:
        choice_for_a = probe["target"]["choice"]
        choice_for_b = probe["foil"]["choice"]
        reading_to_label = {"r1": "A", "r2": "B"}
        variant = "original"
        fixed_ab_margin_sign = 1
        variant_index = 0
    label_to_reading = {label: reading for reading, label in reading_to_label.items()}
    prompt = PROBE_TEMPLATE.format(
        sentence=item["sentence"],
        question=probe["question"],
        choice_for_A=choice_for_a,
        choice_for_B=choice_for_b,
    )
    counterbalance = {
        "group_id": item["id"],
        "variant": variant,
        "variant_index": variant_index,
        "mirrored": mirrored,
        "reading_to_label": reading_to_label,
        "label_to_reading": label_to_reading,
        "target_reading_id": "r1",
        "foil_reading_id": "r2",
        "target_token_label": reading_to_label["r1"],
        "foil_token_label": reading_to_label["r2"],
        "fixed_ab_margin_sign": fixed_ab_margin_sign,
        "oriented_metric": (
            f"{fixed_ab_margin_sign:+d} * (logit(A) - logit(B))"
        ),
    }
    return {"prompt": prompt, "counterbalance": counterbalance}


def render_ambiguity_probe_pair(item: Any) -> list[dict[str, Any]]:
    """Render original then mirrored variants for one validated item."""

    return [
        render_ambiguity_probe(item, mirrored=False),
        render_ambiguity_probe(item, mirrored=True),
    ]


def _encode(tokenizer: Any, text: str, *, context: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except Exception as error:
        raise ValueError(f"Tokenizer failed for {context}: {error}") from error
    try:
        return [int(token_id) for token_id in encoded]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Tokenizer returned invalid IDs for {context}: {encoded!r}") from error


def _resolve_concept_token(
    tokenizer: Any, label: str, *, item_id: str, reading_id: str
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for surface in (f" {label}", label):
        token_ids = _encode(
            tokenizer,
            surface,
            context=f"{item_id}.{reading_id} concept {label!r}",
        )
        attempts.append({"surface": surface, "token_ids": token_ids})
        if len(token_ids) == 1:
            return {"label": label, "surface": surface, "token_id": token_ids[0]}
    raise ValueError(
        f"{item_id}.{reading_id} concept {label!r} is not one token: {attempts}"
    )


def _resolve_exact_continuation(
    tokenizer: Any, prompt: str, token_label: str, *, item_id: str, variant: str
) -> dict[str, Any]:
    surface = f" {token_label}"
    prompt_ids = _encode(
        tokenizer, prompt, context=f"{item_id}.{variant} prompt"
    )
    combined_ids = _encode(
        tokenizer,
        prompt + surface,
        context=f"{item_id}.{variant} continuation {surface!r}",
    )
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"{item_id}.{variant} continuation {surface!r} changes prompt tokenization"
        )
    appended_ids = combined_ids[len(prompt_ids) :]
    if len(appended_ids) != 1:
        raise ValueError(
            f"{item_id}.{variant} continuation {surface!r} appended "
            f"{len(appended_ids)} tokens: {appended_ids}"
        )
    return {
        "token_label": token_label,
        "surface": surface,
        "token_id": appended_ids[0],
    }


def tokenize_ambiguity_item(tokenizer: Any, item_value: Any) -> dict[str, Any]:
    """Resolve concept IDs and exact behavior IDs for both probe assignments."""

    item = _validate_item(item_value, path="item")
    concepts = {
        reading["id"]: _resolve_concept_token(
            tokenizer,
            reading["concept_label"],
            item_id=item["id"],
            reading_id=reading["id"],
        )
        for reading in item["readings"]
    }
    concept_ids = [concept["token_id"] for concept in concepts.values()]
    _require(len(set(concept_ids)) == 2, f"{item['id']} concept token IDs collide")

    prepared_probes: list[dict[str, Any]] = []
    for rendered in render_ambiguity_probe_pair(item):
        counterbalance = rendered["counterbalance"]
        variant = counterbalance["variant"]
        continuations = {
            label: _resolve_exact_continuation(
                tokenizer,
                rendered["prompt"],
                label,
                item_id=item["id"],
                variant=variant,
            )
            for label in ("A", "B")
        }
        continuation_ids = [entry["token_id"] for entry in continuations.values()]
        _require(
            len(set(continuation_ids)) == 2,
            f"{item['id']}.{variant} A/B continuation token IDs collide",
        )
        target_label = counterbalance["target_token_label"]
        foil_label = counterbalance["foil_token_label"]
        prepared_probes.append(
            {
                **rendered,
                "continuations": continuations,
                "target_token_id": continuations[target_label]["token_id"],
                "target_surface": continuations[target_label]["surface"],
                "foil_token_id": continuations[foil_label]["token_id"],
                "foil_surface": continuations[foil_label]["surface"],
            }
        )

    behavior_ids = {
        entry["token_id"]
        for probe in prepared_probes
        for entry in probe["continuations"].values()
    }
    _require(
        not behavior_ids.intersection(concept_ids),
        f"{item['id']} concept token overlaps an A/B behavior token",
    )
    return {
        **item,
        "concepts": concepts,
        "probe_variants": prepared_probes,
    }


def load_tokenized_ambiguity_items(
    tokenizer: Any,
    path: str | Path = DEFAULT_AMBIGUITY_SPEC,
) -> list[dict[str, Any]]:
    """Load the complete spec and fail if any item cannot be tokenized exactly."""

    return [
        tokenize_ambiguity_item(tokenizer, item)
        for item in load_ambiguity_items(path)
    ]


def deterministic_smoke_subset(
    items: Sequence[dict[str, Any]] | None = None,
    *,
    n_per_category: int = 2,
    seed: int = DATASET_SEED,
    path: str | Path = DEFAULT_AMBIGUITY_SPEC,
) -> list[dict[str, Any]]:
    """Choose a reproducible, category-balanced subset in source order."""

    _require(
        isinstance(n_per_category, int) and not isinstance(n_per_category, bool),
        "n_per_category must be an integer",
    )
    _require(n_per_category > 0, "n_per_category must be positive")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        "seed must be an integer",
    )
    source_items = load_ambiguity_items(path) if items is None else list(items)
    seen_ids: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {
        category: [] for category in CATEGORY_COUNTS
    }
    for index, item in enumerate(source_items):
        validated = _validate_item(item, path=f"items[{index}]")
        item_id = validated["id"]
        _require(item_id not in seen_ids, f"duplicate item ID {item_id!r} in subset source")
        seen_ids.add(item_id)
        grouped[validated["category"]].append(item)

    rng = random.Random(seed)
    selected_ids: set[str] = set()
    for category in CATEGORY_COUNTS:
        candidates = grouped[category]
        _require(
            len(candidates) >= n_per_category,
            f"category {category} has only {len(candidates)} candidates",
        )
        selected_ids.update(
            item["id"] for item in rng.sample(candidates, n_per_category)
        )
    return [item for item in source_items if item["id"] in selected_ids]

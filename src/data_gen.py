"""Deterministic dataset loading, token validation, and small gate subsets."""

from __future__ import annotations

import copy
import json
import random
import unicodedata
from pathlib import Path
from typing import Any

from src.model_utils import concept_token_id


UPSTREAM_JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
DEFAULT_JLENS_ROOT = Path.home() / "deps" / "jacobian-lens"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TWOHOP_SUPPLEMENT = PROJECT_ROOT / "data" / "specs" / "twohop_supplement.json"
UPSTREAM_TWOHOP_SOURCE = (
    "anthropics/jacobian-lens:data/experiments/probe-swap.json"
)
SUPPLEMENT_TWOHOP_SOURCE = "j-space-thoughts:data/specs/twohop_supplement.json"
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


def normalize_twohop_prompt(prompt: str) -> str:
    """Return the canonical key used for deterministic prompt deduplication."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Two-hop prompt must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", prompt)
    return " ".join(normalized.split()).casefold()


def _validated_twohop_items(items: Any, *, source: Path) -> list[dict]:
    if not isinstance(items, list) or not items:
        raise ValueError(f"Malformed two-hop dataset at {source}: non-empty items required")

    validated: list[dict] = []
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
        name = raw_item["name"]
        if name in names:
            raise ValueError(f"Duplicate item name {name!r} at {source}")
        names.add(name)
        validated.append(dict(raw_item))
    return validated


def load_twohop_supplement(
    path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
) -> dict[str, Any]:
    """Load and audit the project supplemental two-hop specification."""

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
    normalized_prompts = [normalize_twohop_prompt(item["prompt"]) for item in items]
    if len(normalized_prompts) != len(set(normalized_prompts)):
        raise ValueError(f"Supplemental prompts are not unique after normalization: {spec_path}")
    expected_count = payload["provenance"].get("supplement_item_count")
    if expected_count != len(items):
        raise ValueError(
            f"Supplemental count mismatch at {spec_path}: "
            f"declared={expected_count!r}, actual={len(items)}"
        )
    return {**payload, "items": items}


def _supplement_source_id(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return str(path.resolve())
    return f"j-space-thoughts:{relative.as_posix()}"


def _attach_supplement_provenance(payload: dict[str, Any], path: Path) -> list[dict]:
    dataset_provenance = {
        "schema_version": payload["schema_version"],
        "seed": payload["seed"],
        "declared_source": payload["source"],
        "provenance": payload["provenance"],
    }
    source_id = _supplement_source_id(path)
    return [
        {
            **item,
            "source": source_id,
            "dataset_provenance": copy.deepcopy(dataset_provenance),
        }
        for item in payload["items"]
    ]


def load_twohop_supplement_items(
    path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
) -> list[dict]:
    """Return supplemental items with their top-level provenance attached."""

    spec_path = Path(path)
    payload = load_twohop_supplement(spec_path)
    return _attach_supplement_provenance(payload, spec_path)


def continuation_token_id(
    tokenizer: Any,
    prompt: str,
    continuation: str,
) -> tuple[int, str]:
    """Resolve the exact next token after a fixed, already-tokenized prompt.

    A candidate is accepted only when tokenizing ``prompt + surface`` preserves
    every prompt token and appends exactly one token. This rejects boundary
    merges and multi-token answers instead of silently testing the wrong logit.
    """

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokenization is empty")
    if prompt[-1:].isspace():
        candidates = [continuation]
    else:
        candidates = [continuation] if continuation.startswith(" ") else [f" {continuation}", continuation]
    diagnostics: list[str] = []
    for surface in candidates:
        combined = tokenizer.encode(prompt + surface, add_special_tokens=False)
        appended = combined[len(prompt_ids) :] if combined[: len(prompt_ids)] == prompt_ids else []
        diagnostics.append(f"{surface!r}: full_tail={combined[-4:]}, appended={appended}")
        if len(appended) == 1:
            return int(appended[0]), surface
    raise ValueError(
        f"Continuation {continuation!r} is not one exact next token after prompt: "
        + "; ".join(diagnostics)
    )


def load_probe_swap_items(jlens_root: str | Path = DEFAULT_JLENS_ROOT) -> list[dict]:
    """Load the official Apache-2.0 probe-swap items without copying a corpus."""

    path = Path(jlens_root) / "data" / "experiments" / "probe-swap.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError(f"Malformed probe-swap dataset at {path}")
    return _validated_twohop_items(payload["items"], source=path)


def load_combined_twohop_collection(
    jlens_root: str | Path = DEFAULT_JLENS_ROOT,
    supplement_path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
) -> dict[str, Any]:
    """Build the audited first-wins union of upstream and supplemental prompts.

    Source order is fixed: official probe-swap items first, then the project
    supplement. Prompts are deduplicated by :func:`normalize_twohop_prompt`.
    Every retained item carries source provenance, and the aggregate record
    names each duplicate that was removed.
    """

    upstream_raw = load_probe_swap_items(jlens_root)
    upstream_provenance = {
        "source": UPSTREAM_TWOHOP_SOURCE,
        "commit": UPSTREAM_JLENS_COMMIT,
    }
    upstream = [
        {
            **item,
            "source": UPSTREAM_TWOHOP_SOURCE,
            "source_commit": UPSTREAM_JLENS_COMMIT,
            "dataset_provenance": copy.deepcopy(upstream_provenance),
        }
        for item in upstream_raw
    ]
    supplement_spec_path = Path(supplement_path)
    supplement_payload = load_twohop_supplement(supplement_spec_path)
    supplement = _attach_supplement_provenance(
        supplement_payload, supplement_spec_path
    )
    candidates = [*upstream, *supplement]

    retained: list[dict] = []
    seen: dict[str, dict] = {}
    duplicates: list[dict] = []
    for item in candidates:
        normalized = normalize_twohop_prompt(item["prompt"])
        kept = seen.get(normalized)
        if kept is not None:
            duplicates.append(
                {
                    "normalized_prompt": normalized,
                    "kept_name": kept["name"],
                    "kept_source": kept["source"],
                    "dropped_name": item["name"],
                    "dropped_source": item["source"],
                }
            )
            continue
        seen[normalized] = item
        retained.append(item)

    declared = supplement_payload["provenance"]
    observed_counts = {
        "upstream_item_count": len(upstream),
        "upstream_distinct_prompt_count": len(
            {normalize_twohop_prompt(item["prompt"]) for item in upstream}
        ),
        "supplement_item_count": len(supplement),
        "combined_item_count": len(candidates),
        "combined_distinct_prompt_count": len(retained),
    }
    mismatches = {
        key: (declared.get(key), actual)
        for key, actual in observed_counts.items()
        if declared.get(key) != actual
    }
    if mismatches:
        raise ValueError(f"Combined two-hop provenance count mismatch: {mismatches}")

    return {
        "schema_version": "combined-twohop-v1",
        "seed": supplement_payload["seed"],
        "source": "deterministic upstream-first union",
        "provenance": {
            "normalization": "Unicode NFKC, whitespace collapse, then casefold",
            "deduplication": "first occurrence in source order wins",
            "source_order": [UPSTREAM_TWOHOP_SOURCE, supplement[0]["source"]],
            "counts": observed_counts,
            "duplicates_removed": duplicates,
            "supplement": copy.deepcopy(
                {
                    "schema_version": supplement_payload["schema_version"],
                    "seed": supplement_payload["seed"],
                    "declared_source": supplement_payload["source"],
                    "provenance": supplement_payload["provenance"],
                }
            ),
        },
        "items": retained,
    }


def load_combined_twohop_items(
    jlens_root: str | Path = DEFAULT_JLENS_ROOT,
    supplement_path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
) -> list[dict]:
    """Return the deduplicated combined two-hop items in deterministic order."""

    return load_combined_twohop_collection(jlens_root, supplement_path)["items"]


def tokenize_twohop_item(tokenizer: Any, item: dict) -> dict:
    """Add exact concept and behavior token surfaces/IDs without losing provenance."""

    concept_id, concept_surface = concept_token_id(tokenizer, item["intermediate"])
    foil_concept_id, foil_concept_surface = concept_token_id(tokenizer, item["swap_to"])
    target_id, target_surface = continuation_token_id(
        tokenizer, item["prompt"], item["answer"]
    )
    foil_id, foil_surface = continuation_token_id(
        tokenizer, item["prompt"], item["swap_answer"]
    )
    ids = [concept_id, foil_concept_id, target_id, foil_id]
    if len(set(ids[:2])) != 2 or len(set(ids[2:])) != 2:
        raise ValueError(f"Concept or behavior pair collapsed for {item['name']!r}: {ids}")
    if concept_id in (target_id, foil_id) or foil_concept_id in (target_id, foil_id):
        raise ValueError(f"Concept token overlaps output token for {item['name']!r}")
    tokenized = {
        **item,
        "concept_token_id": concept_id,
        "concept_surface": concept_surface,
        "foil_concept_token_id": foil_concept_id,
        "foil_concept_surface": foil_concept_surface,
        "target_token_id": target_id,
        "target_surface": target_surface,
        "foil_token_id": foil_id,
        "foil_surface": foil_surface,
    }
    if "source" not in tokenized:
        tokenized["source"] = UPSTREAM_TWOHOP_SOURCE
    if (
        tokenized["source"] == UPSTREAM_TWOHOP_SOURCE
        and "source_commit" not in tokenized
    ):
        tokenized["source_commit"] = UPSTREAM_JLENS_COMMIT
    return tokenized


def tokenizable_twohop_items(tokenizer: Any) -> tuple[list[dict], list[dict]]:
    """Return accepted official items and explicit rejection diagnostics."""

    accepted: list[dict] = []
    rejected: list[dict] = []
    for item in load_probe_swap_items():
        try:
            accepted.append(tokenize_twohop_item(tokenizer, item))
        except (ValueError, IndexError) as error:
            rejected.append({"name": item.get("name"), "reason": str(error)})
    return accepted, rejected


def tokenizable_combined_twohop_items(
    tokenizer: Any,
    jlens_root: str | Path = DEFAULT_JLENS_ROOT,
    supplement_path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
) -> tuple[list[dict], list[dict]]:
    """Tokenize the audited combined collection with explicit rejection records."""

    accepted: list[dict] = []
    rejected: list[dict] = []
    for item in load_combined_twohop_items(jlens_root, supplement_path):
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
    return accepted, rejected


def deterministic_subset(items: list[dict], n: int, *, seed: int = 1729) -> list[dict]:
    """Select a reproducible subset while retaining source order in output."""

    if n > len(items):
        raise ValueError(f"Requested n={n} from only {len(items)} items")
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(items)), n))
    return [items[index] for index in selected_indices]


SYMMETRIC_READ_SEED = 1729
SYMMETRIC_CALIBRATION_MIN_PAIRS = 24
SYMMETRIC_N_FOLDS = 5

_SYMMETRIC_TASK_TEMPLATES = {
    "atomic-number-element-symbol": {
        "clue_prefix": "The chemical symbol of ",
        "question": "What is that element's chemical symbol?",
    },
    "us-city-state-capital": {
        "clue_prefix": "The capital of ",
        "question": "What is that US state's capital?",
    },
    "city-country-capital": {
        "clue_prefix": "The capital of ",
        "question": "What is that country's capital?",
    },
}


def _concept_answer_key(item: dict) -> tuple[str, str]:
    return (str(item["intermediate"]).casefold(), str(item["answer"]).casefold())


def _reciprocal_group_key(item: dict) -> tuple[str, tuple[str, str], tuple[str, str]]:
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
) -> tuple[str, str]:
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
    # Keep the concept latent, as in the original J-Lens prompt: the natural
    # description evokes it, while the WRITTEN gate decides whether it is
    # actually represented. ``concept`` remains an explicit function argument
    # so the caller cannot accidentally pair the clue with the wrong label.
    if not concept.strip():
        raise ValueError("Concept label must be non-empty")
    context = f"Consider {description}."
    question = "What is 2 + 2?" if dashboard else template["question"]
    return context, f"Context: {context} Question: {question} Answer:"


def build_symmetric_causal_candidates(
    supplement_path: str | Path = DEFAULT_TWOHOP_SUPPLEMENT,
    *,
    seed: int = SYMMETRIC_READ_SEED,
    calibration_min_pairs: int = SYMMETRIC_CALIBRATION_MIN_PAIRS,
    n_folds: int = SYMMETRIC_N_FOLDS,
) -> dict[str, Any]:
    """Build reciprocal matched prompts and a leakage-safe deterministic split.

    Every source row in the supplement has a reciprocal row. Some unordered
    concept pairs have several independently-authored natural contexts; those
    contexts remain separate prompt pairs but share one dependency group. Whole
    dependency groups, never individual contexts, are assigned to calibration
    or to one held-out fold.
    """

    if calibration_min_pairs < 1 or n_folds < 2:
        raise ValueError("calibration_min_pairs must be positive and n_folds >= 2")
    payload = load_twohop_supplement(supplement_path)
    grouped: dict[
        tuple[str, tuple[str, str], tuple[str, str]], dict[int, list[dict]]
    ] = {}
    for item in payload["items"]:
        key = _reciprocal_group_key(item)
        side = 0 if _concept_answer_key(item) == key[1] else 1
        grouped.setdefault(key, {0: [], 1: []})[side].append(item)

    pairs: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for key in sorted(grouped):
        category, low, high = key
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
            context_a, engine_a = _symmetric_prompt(
                category,
                str(concept_a),
                str(row_a["prompt"]),
                dashboard=False,
            )
            context_b, engine_b = _symmetric_prompt(
                category,
                str(concept_b),
                str(row_b["prompt"]),
                dashboard=False,
            )
            dash_context_a, dashboard_a = _symmetric_prompt(
                category,
                str(concept_a),
                str(row_a["prompt"]),
                dashboard=True,
            )
            dash_context_b, dashboard_b = _symmetric_prompt(
                category,
                str(concept_b),
                str(row_b["prompt"]),
                dashboard=True,
            )
            if context_a != dash_context_a or context_b != dash_context_b:
                raise AssertionError("Engine/dashboard contexts must be byte-identical")
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
    if len({(pair["engine_prompt_a"], pair["engine_prompt_b"]) for pair in pairs}) != len(
        pairs
    ):
        raise ValueError("Symmetric engine prompt pairs are not unique")
    if len(
        {(pair["dashboard_prompt_a"], pair["dashboard_prompt_b"]) for pair in pairs}
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
        group = pair["dependency_group"]
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


def tokenize_symmetric_candidate(tokenizer: Any, pair: dict) -> dict:
    """Attach exact concept/answer token IDs to one symmetric prompt pair."""

    concept_a_id, concept_a_surface = concept_token_id(tokenizer, pair["concept_a"])
    concept_b_id, concept_b_surface = concept_token_id(tokenizer, pair["concept_b"])
    answer_a_id, answer_a_surface = continuation_token_id(
        tokenizer, pair["engine_prompt_a"], pair["answer_a"]
    )
    answer_b_id, answer_b_surface = continuation_token_id(
        tokenizer, pair["engine_prompt_b"], pair["answer_b"]
    )
    dashboard_id_a, dashboard_surface_a = continuation_token_id(
        tokenizer, pair["dashboard_prompt_a"], pair["dashboard_answer"]
    )
    dashboard_id_b, dashboard_surface_b = continuation_token_id(
        tokenizer, pair["dashboard_prompt_b"], pair["dashboard_answer"]
    )
    distractor_id_a, distractor_surface_a = continuation_token_id(
        tokenizer, pair["dashboard_prompt_a"], pair["dashboard_distractor"]
    )
    distractor_id_b, distractor_surface_b = continuation_token_id(
        tokenizer, pair["dashboard_prompt_b"], pair["dashboard_distractor"]
    )
    if concept_a_id == concept_b_id or answer_a_id == answer_b_id:
        raise ValueError(f"Collapsed concept or answer tokens for {pair['pair_id']}")
    if dashboard_id_a != dashboard_id_b or distractor_id_a != distractor_id_b:
        raise ValueError(f"Dashboard tokenization differs across {pair['pair_id']}")
    if dashboard_id_a == distractor_id_a:
        raise ValueError(f"Dashboard target and distractor collapse for {pair['pair_id']}")
    engine_length_a = len(tokenizer.encode(pair["engine_prompt_a"], add_special_tokens=False))
    engine_length_b = len(tokenizer.encode(pair["engine_prompt_b"], add_special_tokens=False))
    dashboard_length_a = len(
        tokenizer.encode(pair["dashboard_prompt_a"], add_special_tokens=False)
    )
    dashboard_length_b = len(
        tokenizer.encode(pair["dashboard_prompt_b"], add_special_tokens=False)
    )
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
        "engine_n_tokens_a": engine_length_a,
        "engine_n_tokens_b": engine_length_b,
        "dashboard_n_tokens_a": dashboard_length_a,
        "dashboard_n_tokens_b": dashboard_length_b,
    }

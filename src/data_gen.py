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

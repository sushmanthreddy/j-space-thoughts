"""Deterministic dataset loading, token validation, and small gate subsets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.model_utils import concept_token_id


UPSTREAM_JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
DEFAULT_JLENS_ROOT = Path.home() / "deps" / "jacobian-lens"


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
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Malformed probe-swap dataset at {path}")
    required = {"name", "prompt", "intermediate", "answer", "swap_to", "swap_answer"}
    for index, item in enumerate(items):
        missing = required - set(item)
        if missing:
            raise ValueError(f"Item {index} missing fields {sorted(missing)}")
    return items


def tokenize_twohop_item(tokenizer: Any, item: dict) -> dict:
    """Add exact concept and behavior token surfaces/IDs to one official item."""

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
    return {
        **item,
        "concept_token_id": concept_id,
        "concept_surface": concept_surface,
        "foil_concept_token_id": foil_concept_id,
        "foil_concept_surface": foil_concept_surface,
        "target_token_id": target_id,
        "target_surface": target_surface,
        "foil_token_id": foil_id,
        "foil_surface": foil_surface,
        "source": "anthropics/jacobian-lens:data/experiments/probe-swap.json",
        "source_commit": UPSTREAM_JLENS_COMMIT,
    }


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


def deterministic_subset(items: list[dict], n: int, *, seed: int = 1729) -> list[dict]:
    """Select a reproducible subset while retaining source order in output."""

    if n > len(items):
        raise ValueError(f"Requested n={n} from only {len(items)} items")
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(items)), n))
    return [items[index] for index in selected_indices]


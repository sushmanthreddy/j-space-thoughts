"""Static and tokenizer-level validation for the supplemental two-hop spec.

These tests never load model weights.  The tokenizer test uses only the pinned
local Qwen tokenizer snapshot and skips cleanly when that snapshot is absent.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest

from src.data_gen import (
    SUPPLEMENT_TWOHOP_SOURCE,
    load_combined_twohop_collection,
    load_combined_twohop_items,
    load_twohop_supplement_items,
    normalize_twohop_prompt,
    tokenizable_combined_twohop_items,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "data" / "specs" / "twohop_supplement.json"
UPSTREAM_RELATIVE_PATH = Path("data/experiments/probe-swap.json")
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
    / TOKENIZER_REVISION
)
REQUIRED_ITEM_FIELDS = {
    "name",
    "category",
    "prompt",
    "intermediate",
    "answer",
    "swap_to",
    "swap_answer",
}


# This explicit table is the factual contract of the spec.  It also prevents a
# typo in either a target or foil from remaining self-consistent by accident.
EXPECTED_RELATIONS = {
    "atomic-number-element-symbol": {
        "hydrogen": "H",
        "helium": "He",
        "lithium": "Li",
        "neon": "Ne",
        "magnesium": "Mg",
        "aluminum": "Al",
        "silicon": "Si",
        "sulfur": "S",
        "chlorine": "Cl",
        "potassium": "K",
        "titanium": "Ti",
        "chromium": "Cr",
        "manganese": "Mn",
        "nickel": "Ni",
        "copper": "Cu",
        "silver": "Ag",
        "tin": "Sn",
        "platinum": "Pt",
        "lead": "Pb",
        "uranium": "U",
    },
    "us-city-state-capital": {
        "Texas": "Austin",
        "California": "Sacramento",
        "Illinois": "Springfield",
        "Michigan": "Lansing",
        "Wisconsin": "Madison",
        "Washington": "Olympia",
        "Nebraska": "Lincoln",
        "Ohio": "Columbus",
        "Tennessee": "Nashville",
        "Oregon": "Salem",
        "Alabama": "Montgomery",
        "Georgia": "Atlanta",
        "Delaware": "Dover",
        "Arizona": "Phoenix",
        "Colorado": "Denver",
        "Massachusetts": "Boston",
        "Idaho": "Boise",
        "Connecticut": "Hartford",
        "Indiana": "Indianapolis",
        "Maine": "Augusta",
    },
    "city-country-capital": {
        "Austria": "Vienna",
        "Belgium": "Brussels",
        "Chile": "Santiago",
        "Cuba": "Havana",
        "Iran": "Tehran",
        "Iraq": "Baghdad",
        "Ireland": "Dublin",
        "Kenya": "Nairobi",
        "Norway": "Oslo",
        "Pakistan": "Islamabad",
        "Portugal": "Lisbon",
        "Russia": "Moscow",
        "Syria": "Damascus",
        "Thailand": "Bangkok",
        "Turkey": "Ankara",
        "Australia": "Canberra",
        "Bulgaria": "Sofia",
        "Denmark": "Copenhagen",
        "Finland": "Helsinki",
        "Jamaica": "Kingston",
        "Tunisia": "Tunis",
        "Lebanon": "Beirut",
        "Afghanistan": "Kabul",
        "Philippines": "Manila",
        "Netherlands": "Amsterdam",
        "Peru": "Lima",
    },
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def items(payload: dict) -> list[dict]:
    return payload["items"]


def _normalized_prompt(prompt: str) -> str:
    return " ".join(prompt.split()).casefold()


def _upstream_path() -> Path:
    jlens_root = Path(
        os.environ.get("JLENS_ROOT", str(Path.home() / "deps" / "jacobian-lens"))
    )
    return jlens_root / UPSTREAM_RELATIVE_PATH


def test_header_and_schema(payload: dict, items: list[dict]) -> None:
    assert payload["schema_version"] == "twohop-supplement-v1"
    assert payload["seed"] == 1729
    assert isinstance(payload["source"], str) and payload["source"]
    provenance = payload["provenance"]
    assert provenance["upstream_commit"] == (
        "581d398613e5602a5af361e1c34d3a92ea82ba8e"
    )
    assert provenance["target_tokenizer_revision"] == TOKENIZER_REVISION
    assert provenance["supplement_item_count"] == len(items) == 206

    for item in items:
        assert set(item) == REQUIRED_ITEM_FIELDS
        assert all(isinstance(item[field], str) and item[field] for field in item)
        assert re.fullmatch(r"[a-z0-9-]+", item["name"])
        assert item["prompt"].endswith(" is")
        assert item["prompt"] == item["prompt"].rstrip()
        assert item["intermediate"] != item["swap_to"]
        assert item["answer"] != item["swap_answer"]
        for field in ("intermediate", "answer", "swap_to", "swap_answer"):
            assert item[field] == item[field].strip()
            # Both concept proxies and both possible outputs must remain
            # unspoken in the fixed prompt.
            assert not re.search(
                rf"(?<!\w){re.escape(item[field])}(?!\w)",
                item["prompt"],
                flags=re.IGNORECASE,
            )


def test_factual_tables_and_matched_reciprocal_foils(items: list[dict]) -> None:
    actual_relations: dict[str, dict[str, str]] = {
        category: {} for category in EXPECTED_RELATIONS
    }
    indexed: dict[tuple[str, str], dict] = {}
    for item in items:
        category = item["category"]
        assert category in EXPECTED_RELATIONS
        existing = actual_relations[category].setdefault(
            item["intermediate"], item["answer"]
        )
        assert existing == item["answer"]
        indexed[(category, item["intermediate"])] = item

    assert actual_relations == EXPECTED_RELATIONS
    assert Counter(item["category"] for item in items) == {
        "atomic-number-element-symbol": 20,
        "us-city-state-capital": 90,
        "city-country-capital": 96,
    }

    pair_counts = Counter(
        (
            item["category"],
            item["intermediate"],
            item["answer"],
            item["swap_to"],
            item["swap_answer"],
        )
        for item in items
    )
    for item in items:
        mate = indexed[(item["category"], item["swap_to"])]
        assert mate["answer"] == item["swap_answer"]
        forward = (
            item["category"],
            item["intermediate"],
            item["answer"],
            item["swap_to"],
            item["swap_answer"],
        )
        reverse = (
            item["category"],
            item["swap_to"],
            item["swap_answer"],
            item["intermediate"],
            item["answer"],
        )
        assert pair_counts[forward] == pair_counts[reverse]


def test_names_and_prompts_are_unique(items: list[dict]) -> None:
    names = [item["name"] for item in items]
    prompts = [normalize_twohop_prompt(item["prompt"]) for item in items]
    assert len(names) == len(set(names))
    assert len(prompts) == len(set(prompts))


def test_combined_with_pinned_upstream_has_at_least_150_distinct_prompts(
    payload: dict, items: list[dict]
) -> None:
    path = _upstream_path()
    assert path.is_file(), f"Pinned J-Lens probe-swap dataset not found: {path}"
    upstream = json.loads(path.read_text(encoding="utf-8"))["items"]
    provenance = payload["provenance"]
    assert len(upstream) == provenance["upstream_item_count"] == 90

    upstream_prompts = {_normalized_prompt(item["prompt"]) for item in upstream}
    supplemental_prompts = {_normalized_prompt(item["prompt"]) for item in items}
    assert len(upstream_prompts) == provenance["upstream_distinct_prompt_count"] == 86
    assert upstream_prompts.isdisjoint(supplemental_prompts)
    combined = upstream_prompts | supplemental_prompts
    assert len(upstream) + len(items) == provenance["combined_item_count"] == 296
    assert len(combined) == provenance["combined_distinct_prompt_count"] == 292
    assert len(combined) >= 150


def test_robust_loaders_preserve_provenance_and_audit_deduplication(
    payload: dict,
) -> None:
    supplemental = load_twohop_supplement_items()
    assert len(supplemental) == payload["provenance"]["supplement_item_count"] == 206
    assert all(item["source"] == SUPPLEMENT_TWOHOP_SOURCE for item in supplemental)
    assert all(item["dataset_provenance"]["seed"] == 1729 for item in supplemental)

    collection = load_combined_twohop_collection()
    combined = load_combined_twohop_items()
    assert collection["items"] == combined
    assert len(combined) == 292
    assert len({normalize_twohop_prompt(item["prompt"]) for item in combined}) == 292
    audit = collection["provenance"]
    assert audit["counts"] == {
        "upstream_item_count": 90,
        "upstream_distinct_prompt_count": 86,
        "supplement_item_count": 206,
        "combined_item_count": 296,
        "combined_distinct_prompt_count": 292,
    }
    assert len(audit["duplicates_removed"]) == 4
    assert {record["dropped_name"] for record in audit["duplicates_removed"]} == {
        "ex-element-state-80-8",
        "ex-city-currency-Toronto-Mumbai",
        "ex-element-state-26-80",
        "ex-element-state-8-80",
    }


def test_qwen_single_token_concept_and_answer_proxies(items: list[dict]) -> None:
    """Validate only tokenizer files; never instantiate or download a model."""

    if not TOKENIZER_SNAPSHOT.is_dir():
        pytest.skip(f"Pinned local tokenizer snapshot is absent: {TOKENIZER_SNAPSHOT}")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_SNAPSHOT,
        local_files_only=True,
    )

    for item in items:
        prompt_ids = tokenizer.encode(item["prompt"], add_special_tokens=False)
        concept_ids = []
        for field in ("intermediate", "swap_to"):
            token_ids = tokenizer.encode(f" {item[field]}", add_special_tokens=False)
            assert len(token_ids) == 1, (item["name"], field, token_ids)
            concept_ids.append(token_ids[0])

        answer_ids = []
        for field in ("answer", "swap_answer"):
            combined_ids = tokenizer.encode(
                f"{item['prompt']} {item[field]}", add_special_tokens=False
            )
            assert combined_ids[: len(prompt_ids)] == prompt_ids, (
                item["name"],
                field,
                "answer changed the fixed prompt tokenization",
            )
            appended_ids = combined_ids[len(prompt_ids) :]
            assert len(appended_ids) == 1, (item["name"], field, appended_ids)
            answer_ids.append(appended_ids[0])

        assert concept_ids[0] != concept_ids[1]
        assert answer_ids[0] != answer_ids[1]
        assert set(concept_ids).isdisjoint(answer_ids)

    accepted, rejected = tokenizable_combined_twohop_items(tokenizer)
    assert len(accepted) == 269
    assert len(rejected) == 23
    assert len(accepted) + len(rejected) == 292
    supplemental_names = {item["name"] for item in items}
    accepted_names = {item["name"] for item in accepted}
    rejected_names = {item["name"] for item in rejected}
    assert supplemental_names <= accepted_names
    assert supplemental_names.isdisjoint(rejected_names)
    assert all(
        item.get("dataset_provenance") is not None
        for item in accepted
        if item["name"] in supplemental_names
    )

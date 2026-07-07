from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy

import pytest

from src.ambiguity_data import (
    CATEGORY_COUNTS,
    deterministic_smoke_subset,
    load_ambiguity_items,
    load_ambiguity_spec,
    load_tokenized_ambiguity_items,
    render_ambiguity_probe_pair,
    tokenize_ambiguity_item,
    validate_ambiguity_spec,
)


class BoundaryTokenizer:
    """Small tokenizer double that treats leading-space words as units."""

    _pieces = re.compile(r"\s*[A-Za-z]+|\s*[^A-Za-z\s]")

    def __init__(
        self,
        *,
        split_exact: set[str] | None = None,
        merge_answer_a: bool = False,
    ) -> None:
        self.split_exact = split_exact or set()
        self.merge_answer_a = merge_answer_a
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text in self.split_exact:
            return [900_001, 900_002]
        pieces = self._pieces.findall(text)
        if self.merge_answer_a and text.endswith(" A"):
            pieces[-2:] = ["".join(pieces[-2:])]
        result: list[int] = []
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab[piece] = 100 + len(self.vocab)
            result.append(self.vocab[piece])
        return result


def test_frozen_spec_loads_and_rejects_structural_mutation(tmp_path) -> None:
    spec = load_ambiguity_spec()
    assert len(spec["items"]) == 120
    assert Counter(item["category"] for item in spec["items"]) == CATEGORY_COUNTS

    duplicate = deepcopy(spec)
    duplicate["items"][1]["sentence"] = duplicate["items"][0]["sentence"]
    with pytest.raises(ValueError, match="duplicate normalized sentences"):
        validate_ambiguity_spec(duplicate)

    extended = deepcopy(spec)
    extended["items"][0]["unexpected"] = True
    with pytest.raises(ValueError, match=r"extra=\['unexpected'\]"):
        validate_ambiguity_spec(extended)

    invalid_type = deepcopy(spec)
    invalid_type["items"][0]["category"] = []
    with pytest.raises(ValueError, match="category is not recognized"):
        validate_ambiguity_spec(invalid_type)

    duplicate_keys = tmp_path / "duplicate-keys.json"
    duplicate_keys.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key 'seed'"):
        load_ambiguity_spec(duplicate_keys)


def test_original_and_mirrored_rendering_swap_assignments() -> None:
    item = load_ambiguity_items()[0]
    original, mirrored = render_ambiguity_probe_pair(item)

    assert original["prompt"].startswith(f"Sentence: {item['sentence']}\n")
    assert "A: a financial institution\nB: land beside a river" in original["prompt"]
    assert "A: land beside a river\nB: a financial institution" in mirrored["prompt"]

    original_meta = original["counterbalance"]
    mirrored_meta = mirrored["counterbalance"]
    assert original_meta["group_id"] == mirrored_meta["group_id"] == item["id"]
    assert original_meta["reading_to_label"] == {"r1": "A", "r2": "B"}
    assert mirrored_meta["reading_to_label"] == {"r1": "B", "r2": "A"}
    assert original_meta["fixed_ab_margin_sign"] == 1
    assert mirrored_meta["fixed_ab_margin_sign"] == -1
    assert original_meta["target_token_label"] == "A"
    assert mirrored_meta["target_token_label"] == "B"


def test_tokenization_maps_concepts_and_exact_probe_continuations() -> None:
    item = load_ambiguity_items()[0]
    prepared = tokenize_ambiguity_item(BoundaryTokenizer(), item)

    assert prepared["readings"] == item["readings"]
    assert prepared["behavior_probe"] == item["behavior_probe"]
    assert prepared["concepts"]["r1"]["surface"] == " money"
    assert prepared["concepts"]["r2"]["surface"] == " river"
    assert prepared["concepts"]["r1"]["token_id"] != prepared["concepts"]["r2"]["token_id"]

    original, mirrored = prepared["probe_variants"]
    assert original["continuations"]["A"]["surface"] == " A"
    assert original["continuations"]["B"]["surface"] == " B"
    assert original["target_token_id"] == original["continuations"]["A"]["token_id"]
    assert original["foil_token_id"] == original["continuations"]["B"]["token_id"]
    assert mirrored["target_token_id"] == mirrored["continuations"]["B"]["token_id"]
    assert mirrored["foil_token_id"] == mirrored["continuations"]["A"]["token_id"]


def test_tokenization_fails_closed_on_split_or_boundary_merge() -> None:
    item = load_ambiguity_items()[0]
    split_concept = BoundaryTokenizer(split_exact={" money", "money"})
    with pytest.raises(ValueError, match="concept 'money' is not one token"):
        tokenize_ambiguity_item(split_concept, item)

    merging_tokenizer = BoundaryTokenizer(merge_answer_a=True)
    with pytest.raises(ValueError, match="changes prompt tokenization"):
        tokenize_ambiguity_item(merging_tokenizer, item)


def test_full_tokenization_and_smoke_subset_are_deterministic_and_balanced() -> None:
    prepared = load_tokenized_ambiguity_items(BoundaryTokenizer())
    assert len(prepared) == 120
    assert all(len(item["probe_variants"]) == 2 for item in prepared)

    items = load_ambiguity_items()
    before = [item["id"] for item in items]
    first = deterministic_smoke_subset(items, n_per_category=2, seed=19)
    second = deterministic_smoke_subset(items, n_per_category=2, seed=19)
    other_seed = deterministic_smoke_subset(items, n_per_category=2, seed=20)
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert [item["id"] for item in first] != [item["id"] for item in other_seed]
    assert Counter(item["category"] for item in first) == {
        category: 2 for category in CATEGORY_COUNTS
    }
    source_positions = {item_id: index for index, item_id in enumerate(before)}
    assert [source_positions[item["id"]] for item in first] == sorted(
        source_positions[item["id"]] for item in first
    )
    assert [item["id"] for item in items] == before

    with pytest.raises(ValueError, match="has only 30 candidates"):
        deterministic_smoke_subset(items, n_per_category=31)

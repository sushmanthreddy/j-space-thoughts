from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


SPEC_PATH = Path(__file__).parents[1] / "data" / "specs" / "ambiguity.json"
EXPECTED_COUNTS = {
    "lexical_ambiguity": 30,
    "pp_attachment": 30,
    "garden_path": 30,
    "ambiguous_pronoun": 30,
}
ID_PREFIXES = {
    "lexical_ambiguity": "lexical",
    "pp_attachment": "pp_attachment",
    "garden_path": "garden_path",
    "ambiguous_pronoun": "ambiguous_pronoun",
}


def _load_spec() -> dict:
    with SPEC_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_sentence(sentence: str) -> str:
    normalized = unicodedata.normalize("NFKC", sentence)
    return " ".join(normalized.casefold().split())


def test_ambiguity_spec_metadata_and_balance() -> None:
    spec = _load_spec()
    assert spec["schema_version"] == "1.0.0"
    assert spec["dataset"] == "ambiguity_flagship"
    assert isinstance(spec["seed"], int)
    assert spec["provenance"]["origin"]
    assert "no model output" in spec["provenance"]["method"]

    items = spec["items"]
    counts = Counter(item["category"] for item in items)
    assert len(items) == 120
    assert counts == EXPECTED_COUNTS
    assert spec["design"]["item_count"] == len(items)
    assert spec["design"]["category_counts"] == EXPECTED_COUNTS
    tokenizer_gate = spec["design"]["tokenizer_validation"]
    assert tokenizer_gate["concept_item_readings_passed"] == 2 * len(items)
    assert tokenizer_gate["probe_item_labels_passed"] == 2 * len(items)
    assert tokenizer_gate["model_inference_used"] is False


def test_ambiguity_items_have_stable_unique_identity_and_sentences() -> None:
    items = _load_spec()["items"]
    ids = [item["id"] for item in items]
    sentences = [_normalized_sentence(item["sentence"]) for item in items]
    assert len(ids) == len(set(ids))
    assert len(sentences) == len(set(sentences))

    seen_numbers: dict[str, list[int]] = {category: [] for category in EXPECTED_COUNTS}
    for item in items:
        category = item["category"]
        match = re.fullmatch(rf"{re.escape(ID_PREFIXES[category])}_(\d{{3}})", item["id"])
        assert match, item["id"]
        seen_numbers[category].append(int(match.group(1)))
        assert item["sentence"].strip() == item["sentence"]
        assert item["sentence"].endswith((".", "?", "!"))

    for category, numbers in seen_numbers.items():
        assert numbers == list(range(1, EXPECTED_COUNTS[category] + 1))


def test_each_item_defines_two_readings_and_a_forced_choice_probe() -> None:
    for item in _load_spec()["items"]:
        readings = item["readings"]
        assert len(readings) == 2
        assert [reading["id"] for reading in readings] == ["r1", "r2"]
        assert all(reading["description"].strip() for reading in readings)

        concept_labels = [reading["concept_label"] for reading in readings]
        assert len(set(concept_labels)) == 2
        assert all(re.fullmatch(r"[a-z]+", label) for label in concept_labels)

        probe = item["behavior_probe"]
        assert probe["type"] == "forced_choice"
        assert probe["question"].strip()
        target, foil = probe["target"], probe["foil"]
        assert target["reading_id"] == "r1"
        assert foil["reading_id"] == "r2"
        assert target["token_label"] == "A"
        assert foil["token_label"] == "B"
        assert target["token_label"] != foil["token_label"]
        assert target["choice"].strip()
        assert foil["choice"].strip()
        assert target["choice"].casefold() != foil["choice"].casefold()


def test_construction_subtypes_cover_the_nonlexical_families() -> None:
    items = _load_spec()["items"]
    garden = Counter(
        item["construction"] for item in items if item["category"] == "garden_path"
    )
    pronouns = Counter(
        item["construction"]
        for item in items
        if item["category"] == "ambiguous_pronoun"
    )
    assert garden == {"reduced_relative": 10, "np_z": 10, "gerund_participle": 10}
    assert pronouns == {
        "singular_they_subject": 10,
        "singular_they_possessive": 10,
        "singular_they_object": 10,
    }

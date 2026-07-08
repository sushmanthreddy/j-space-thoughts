"""Leakage-audited authored cues for mean-difference concept directions."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data_gen import DEFAULT_JLENS_ROOT, load_probe_swap_items


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data/specs/md_cues_v1.json"
WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall(_normalized_text(text))


def _contains_word(text: str, surface: str) -> bool:
    text_words = _words(text)
    surface_words = _words(surface)
    if not surface_words:
        return False
    width = len(surface_words)
    return any(
        text_words[index : index + width] == surface_words
        for index in range(len(text_words) - width + 1)
    )


def _contains_subsequence(sequence: Sequence[int], candidate: Sequence[int]) -> bool:
    if not candidate:
        return False
    width = len(candidate)
    return any(
        list(sequence[index : index + width]) == list(candidate)
        for index in range(len(sequence) - width + 1)
    )


def _has_shared_ngram(first: str, second: str, width: int) -> bool:
    first_words = _words(first)
    second_words = _words(second)
    if len(first_words) < width or len(second_words) < width:
        return False
    first_ngrams = {
        tuple(first_words[index : index + width])
        for index in range(len(first_words) - width + 1)
    }
    return any(
        tuple(second_words[index : index + width]) in first_ngrams
        for index in range(len(second_words) - width + 1)
    )


def load_md_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load the frozen authored-cue specification."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "md-cues-v1":
        raise ValueError(
            f"Unsupported MD cue schema: {payload.get('schema_version')!r}"
        )
    return payload


def render_cue(payload: Mapping[str, Any], cue: Mapping[str, str]) -> str:
    """Render one silent activation prompt from its frozen carrier template."""

    try:
        template = payload["templates"][cue["template_id"]]
    except KeyError as error:
        raise ValueError(f"Unknown cue template: {cue.get('template_id')!r}") from error
    rendered = template.format(cue=cue["text"])
    if not rendered.endswith(payload["anchor_suffix"]):
        raise ValueError(f"Cue {cue['cue_id']!r} does not end in the frozen anchor")
    return rendered


def concept_prompt_sets(payload: Mapping[str, Any], split: str) -> dict[str, list[str]]:
    """Return concept-keyed prompts in the manifest's matched template order."""

    if split not in {"train", "heldout"}:
        raise ValueError(f"Unknown split {split!r}")
    template_order = [
        template_id
        for template_id in payload["templates"]
        if template_id.startswith(f"{split}_")
    ]
    prompt_sets: dict[str, list[str]] = {}
    for concept in payload["concepts"]:
        by_template = {
            cue["template_id"]: cue for cue in concept["cues"] if cue["split"] == split
        }
        if set(by_template) != set(template_order):
            raise ValueError(
                f"Concept {concept['concept']!r} does not cover every {split} template"
            )
        prompt_sets[concept["concept"]] = [
            render_cue(payload, by_template[template_id])
            for template_id in template_order
        ]
    return prompt_sets


def baseline_exclusions(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    """Return the frozen leave-paired-foil-out baseline graph."""

    return {concept["concept"]: {concept["foil"]} for concept in payload["concepts"]}


def explicit_probe_prompt(cue: Mapping[str, str]) -> str:
    """Build a separate answer-eliciting probe never used for MD activations."""

    return (
        "Identify the entity described by this clue. Reply with its name only.\n"
        f"Clue: {cue['text']}\nAnswer:"
    )


def audit_md_manifest(
    payload: Mapping[str, Any],
    tokenizer: Any,
    *,
    jlens_root: str | Path = DEFAULT_JLENS_ROOT,
    max_length: int = 128,
) -> dict[str, Any]:
    """Fail closed on cue leakage, source drift, split reuse, and truncation."""

    selection = payload["selection"]
    concepts = payload["concepts"]
    pairs = payload["pairs"]
    if (
        len(concepts) != selection["n_unique_concepts"]
        or len(pairs) != selection["n_pairs"]
    ):
        raise ValueError("Manifest counts disagree with its frozen selection metadata")
    concept_names = [concept["concept"] for concept in concepts]
    if len(concept_names) != len(set(concept_names)):
        raise ValueError("Concept labels must be globally unique")

    upstream = {item["name"]: item for item in load_probe_swap_items(jlens_root)}
    pair_by_id = {pair["pair_id"]: pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("Pair IDs must be unique")
    for pair_id, pair in pair_by_id.items():
        if pair_id not in upstream:
            raise ValueError(f"Unknown upstream pair {pair_id!r}")
        item = upstream[pair_id]
        expected_concepts = [item["intermediate"], item["swap_to"]]
        expected_answers = [item["answer"], item["swap_answer"]]
        if (
            pair["concepts"] != expected_concepts
            or pair["behavior_answers"] != expected_answers
        ):
            raise ValueError(f"Pair {pair_id!r} has drifted from the pinned source")

    all_cue_ids: set[str] = set()
    all_fact_ids: set[str] = set()
    normalized_cues: set[str] = set()
    train_facts: set[str] = set()
    heldout_facts: set[str] = set()
    rendered_hashes: dict[str, str] = {}
    anchor_ids = tokenizer.encode(payload["anchor_suffix"], add_special_tokens=False)
    behavior_prompts = [item["prompt"] for item in upstream.values()]

    for concept in concepts:
        name = concept["concept"]
        pair = pair_by_id[concept["pair_id"]]
        if name not in pair["concepts"] or concept["foil"] not in pair["concepts"]:
            raise ValueError(f"Concept {name!r} has an invalid pair/foil mapping")
        if name == concept["foil"]:
            raise ValueError(f"Concept {name!r} cannot be its own foil")
        actual_ids = tokenizer.encode(
            concept["token_surface"], add_special_tokens=False
        )
        if actual_ids != [concept["token_id"]]:
            raise ValueError(
                f"Pinned token ID drift for concept {name!r}: {actual_ids}"
            )
        if name not in concept["token_surface"]:
            raise ValueError(f"Token surface does not contain concept label {name!r}")

        cues = concept["cues"]
        split_counts = {
            split: sum(cue["split"] == split for cue in cues)
            for split in ("train", "heldout")
        }
        expected_counts = {
            "train": selection["train_cues_per_concept"],
            "heldout": selection["heldout_cues_per_concept"],
        }
        if split_counts != expected_counts:
            raise ValueError(f"Wrong cue counts for {name!r}: {split_counts}")

        forbidden_surfaces = [name, concept["foil"], *pair["behavior_answers"]]
        forbidden_token_sequences: list[list[int]] = [
            tokenizer.encode(f" {surface}", add_special_tokens=False)
            for surface in forbidden_surfaces
        ]
        for cue in cues:
            cue_id = cue["cue_id"]
            fact_id = cue["fact_id"]
            normalized = _normalized_text(cue["text"])
            if (
                cue_id in all_cue_ids
                or fact_id in all_fact_ids
                or normalized in normalized_cues
            ):
                raise ValueError(f"Cue/fact/text reuse detected at {cue_id!r}")
            all_cue_ids.add(cue_id)
            all_fact_ids.add(fact_id)
            normalized_cues.add(normalized)
            (train_facts if cue["split"] == "train" else heldout_facts).add(fact_id)
            for surface in forbidden_surfaces:
                if _contains_word(cue["text"], surface):
                    raise ValueError(f"Forbidden surface {surface!r} in cue {cue_id!r}")
            if any(
                _has_shared_ngram(cue["text"], prompt, 6) for prompt in behavior_prompts
            ):
                raise ValueError(f"Behavior-prompt overlap in cue {cue_id!r}")

            rendered = render_cue(payload, cue)
            rendered_ids = tokenizer.encode(
                rendered,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            untruncated_ids = tokenizer.encode(rendered, add_special_tokens=False)
            if rendered_ids != untruncated_ids:
                raise ValueError(
                    f"Cue {cue_id!r} is truncated at max_length={max_length}"
                )
            if rendered_ids[-len(anchor_ids) :] != anchor_ids:
                raise ValueError(
                    f"Cue {cue_id!r} does not share the anchor token suffix"
                )
            if any(
                _contains_subsequence(rendered_ids, forbidden)
                for forbidden in forbidden_token_sequences
            ):
                raise ValueError(f"Forbidden token sequence in cue {cue_id!r}")
            rendered_hashes[cue_id] = hashlib.sha256(rendered.encode()).hexdigest()

    if train_facts & heldout_facts:
        raise ValueError("Train and held-out fact IDs overlap")
    train_sets = concept_prompt_sets(payload, "train")
    heldout_sets = concept_prompt_sets(payload, "heldout")
    return {
        "status": "PASS",
        "schema_version": payload["schema_version"],
        "n_pairs": len(pairs),
        "n_concepts": len(concepts),
        "n_train_cues": sum(map(len, train_sets.values())),
        "n_heldout_cues": sum(map(len, heldout_sets.values())),
        "max_length": max_length,
        "anchor_token_ids": anchor_ids,
        "rendered_sha256": rendered_hashes,
    }

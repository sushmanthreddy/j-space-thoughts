"""Integrity gates for the independently authored MD cue bank."""

from __future__ import annotations

import copy

import pytest

from src.md_manifest import (
    audit_md_manifest,
    baseline_exclusions,
    concept_prompt_sets,
    load_md_manifest,
)


TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            revision=TOKENIZER_REVISION,
            local_files_only=True,
        )
    except OSError as error:
        pytest.skip(f"Pinned tokenizer unavailable: {error}")


def test_manifest_has_balanced_concept_keyed_splits() -> None:
    payload = load_md_manifest()
    train = concept_prompt_sets(payload, "train")
    heldout = concept_prompt_sets(payload, "heldout")
    assert len(train) == len(heldout) == 40
    assert {len(prompts) for prompts in train.values()} == {4}
    assert {len(prompts) for prompts in heldout.values()} == {2}
    exclusions = baseline_exclusions(payload)
    assert all(
        exclusions[name] == {concept["foil"]}
        for name, concept in zip(exclusions, payload["concepts"], strict=True)
    )


def test_manifest_passes_full_leakage_audit(tokenizer) -> None:
    result = audit_md_manifest(load_md_manifest(), tokenizer)
    assert result["status"] == "PASS"
    assert result["n_train_cues"] == 160
    assert result["n_heldout_cues"] == 80
    assert len(result["rendered_sha256"]) == 240


def test_audit_rejects_concept_surface_leakage(tokenizer) -> None:
    payload = copy.deepcopy(load_md_manifest())
    payload["concepts"][0]["cues"][0]["text"] += " turtle"
    with pytest.raises(ValueError, match="Forbidden surface"):
        audit_md_manifest(payload, tokenizer)

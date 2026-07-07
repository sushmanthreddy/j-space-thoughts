from __future__ import annotations

from dataclasses import dataclass

from src.v2_repair import (
    exact_label_token_id,
    label_token_candidates,
    longest_contiguous_visible_band,
    paper_workspace_prior,
)


@dataclass
class _Tokenizer:
    mapping: dict[str, list[int]]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.mapping.get(text, [91, 92])

    def decode(self, token_ids):
        return next(
            (surface for surface, ids in self.mapping.items() if ids == token_ids),
            "?",
        )


def test_exact_label_resolution_precedes_leading_space() -> None:
    tokenizer = _Tokenizer({"ant": [517], " ant": [3196]})
    candidates = label_token_candidates(tokenizer, "ant")
    assert [row["token_id"] for row in candidates] == [517, 3196]
    assert exact_label_token_id(tokenizer, "ant") == (517, "ant")


def test_exact_label_resolution_falls_back_when_multitoken() -> None:
    tokenizer = _Tokenizer({"spider": [1, 2], " spider": [34354]})
    assert exact_label_token_id(tokenizer, "spider") == (34354, " spider")


def test_normalized_paper_prior_for_28_layer_model() -> None:
    assert paper_workspace_prior(28, list(range(27))) == list(range(11, 25))


def test_workspace_selection_uses_longest_clean_visibility_run() -> None:
    prior = list(range(11, 25))
    ranks = {
        "a": {layer: (5 if 13 <= layer <= 24 else 100) for layer in prior},
        "b": {layer: (6 if 13 <= layer <= 24 else 100) for layer in prior},
        "c": {layer: 50 for layer in prior},
    }
    band, diagnostics = longest_contiguous_visible_band(prior, ranks)
    assert band == list(range(13, 25))
    assert diagnostics[12]["active"] is False
    assert diagnostics[13]["active"] is True

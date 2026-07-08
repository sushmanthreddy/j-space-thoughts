from __future__ import annotations

from types import SimpleNamespace

import torch

from src.model_utils import batched_next_token_records


class _Tokenizer:
    def __call__(self, prompts, **kwargs):
        del kwargs
        lengths = [len(prompt) for prompt in prompts]
        width = max(lengths)
        input_ids = torch.zeros(len(prompts), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = length
            attention_mask[row, :length] = 1
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, token_ids):
        return f"token-{token_ids[0]}"


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        logits = torch.zeros(*input_ids.shape, 8, device=input_ids.device)
        logits.scatter_(-1, input_ids.unsqueeze(-1), 5.0)
        return SimpleNamespace(logits=logits + self.anchor)


def test_batched_next_token_records_respects_true_unpadded_position() -> None:
    rows = batched_next_token_records(
        _Model(),
        _Tokenizer(),
        ["aa", "bbb"],
        [2, 3],
        batch_size=2,
        top_k=3,
    )
    assert [row["rank"] for row in rows] == [1, 1]
    assert [row["n_tokens"] for row in rows] == [2, 3]
    assert all(row["top1_correct"] for row in rows)

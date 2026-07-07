from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.interventions import forward_logits, residual_edit_hooks
from src.read_scores import attribution_read


def _tiny_qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(3)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_attribution_read_matches_directional_finite_difference() -> None:
    model = _tiny_qwen()
    input_ids = torch.tensor([[1, 7, 9, 4]])
    direction = F.normalize(torch.arange(1, 17, dtype=torch.float32), dim=0)
    layer = 1
    result = attribution_read(
        model,
        model.model.layers,
        input_ids,
        {layer: direction},
        target_token_id=5,
        foil_token_id=6,
        intervention_positions=[-1],
    )

    epsilon = 1e-3

    def add_scaled(hidden: torch.Tensor, scale: float) -> torch.Tensor:
        edited = hidden.clone()
        edited[:, -1, :] = edited[:, -1, :] + scale * direction.to(hidden.dtype)
        return edited

    clean = forward_logits(model, input_ids)
    with residual_edit_hooks(
        model.model.layers, {layer: lambda hidden: add_scaled(hidden, epsilon)}
    ):
        plus = model(input_ids=input_ids, use_cache=False).logits.float()
    with residual_edit_hooks(
        model.model.layers, {layer: lambda hidden: add_scaled(hidden, -epsilon)}
    ):
        minus = model(input_ids=input_ids, use_cache=False).logits.float()
    plus_metric = plus[0, -1, 5] - plus[0, -1, 6]
    minus_metric = minus[0, -1, 5] - minus[0, -1, 6]
    finite_difference = float((plus_metric - minus_metric) / (2 * epsilon))

    assert abs(finite_difference - float(result.read[layer][0])) < 2e-3
    restored = forward_logits(model, input_ids)
    assert torch.equal(clean, restored)

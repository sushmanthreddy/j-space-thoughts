import torch

from src.interventions import ablation_edits


def test_ablation_edits_forward_shared_strength():
    direction = torch.tensor([1.0, 0.0])
    edits = ablation_edits({3: direction}, strength=0.25)
    hidden = torch.tensor([[[4.0, 2.0]]])
    assert torch.equal(edits[3](hidden), torch.tensor([[[3.0, 2.0]]]))

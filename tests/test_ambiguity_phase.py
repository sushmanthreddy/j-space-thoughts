from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.ambiguity_phase import (
    aggregate_counterbalanced_variants,
    analyze_meta_tokens,
    analyze_p3,
    infer_counterbalanced_commitment,
    plot_f8,
    probe_margin_from_logits,
    resolve_meta_token_candidates,
    shared_direction_attribution,
)


def _prepared_probe(*, mirrored: bool) -> dict:
    if mirrored:
        reading_to_label = {"r1": "B", "r2": "A"}
        sign = -1
        variant = "mirrored"
        index = 1
    else:
        reading_to_label = {"r1": "A", "r2": "B"}
        sign = 1
        variant = "original"
        index = 0
    return {
        "counterbalance": {
            "group_id": "synthetic_001",
            "variant": variant,
            "variant_index": index,
            "reading_to_label": reading_to_label,
            "fixed_ab_margin_sign": sign,
        },
        "continuations": {
            "A": {"token_id": 1},
            "B": {"token_id": 2},
        },
    }


def test_synthetic_logits_are_oriented_before_commitment() -> None:
    original_logits = torch.tensor([0.0, 5.0, 2.0, 0.0])
    mirrored_logits = torch.tensor([0.0, 4.0, 6.0, 0.0])
    original = probe_margin_from_logits(
        original_logits,
        _prepared_probe(mirrored=False),
    )
    mirrored = probe_margin_from_logits(
        mirrored_logits,
        _prepared_probe(mirrored=True),
    )

    assert original["fixed_ab_margin"] == pytest.approx(3.0)
    assert original["r1_minus_r2_margin"] == pytest.approx(3.0)
    assert mirrored["fixed_ab_margin"] == pytest.approx(-2.0)
    assert mirrored["r1_minus_r2_margin"] == pytest.approx(2.0)

    commitment = infer_counterbalanced_commitment([original, mirrored])
    assert commitment["uses_gold_reading"] is False
    assert commitment["committed_reading"] == "r1"
    assert commitment["committed_mean_margin"] == pytest.approx(2.5)
    assert commitment["counterbalance_agreement"] is True


def test_commitment_can_select_r2_and_retains_counterbalance_disagreement() -> None:
    original = probe_margin_from_logits(
        torch.tensor([0.0, 1.0, 5.0]),
        _prepared_probe(mirrored=False),
    )
    mirrored = probe_margin_from_logits(
        torch.tensor([0.0, 7.0, 5.0]),
        _prepared_probe(mirrored=True),
    )
    commitment = infer_counterbalanced_commitment([original, mirrored])
    assert commitment["committed_reading"] == "r2"
    assert commitment["mean_r1_minus_r2_margin"] == pytest.approx(-3.0)
    assert commitment["committed_margin_by_variant"] == pytest.approx([4.0, 2.0])

    disagreeing = dict(mirrored)
    disagreeing["r1_minus_r2_margin"] = 1.0
    disagreement = infer_counterbalanced_commitment([original, disagreeing])
    assert disagreement["committed_reading"] == "r2"
    assert disagreement["counterbalance_agreement"] is False


def _effect(clean: float, edited: float) -> dict[str, float]:
    return {
        "edited_committed_margin": edited,
        "delta": edited - clean,
        "positive_damage": clean - edited,
    }


def _variant_record(
    variant: str,
    *,
    clean: float,
    ablated: float,
    swapped: float,
    suppressed: float,
) -> dict:
    return {
        "counterbalance": {"variant": variant},
        "clean_committed_margin": clean,
        "attribution": {
            "aggregate": {
                "write_abs_mean": 2.0 if variant == "original" else 4.0,
                "read_abs_mean": 0.4 if variant == "original" else 0.6,
                "support_oriented_read": 0.3 if variant == "original" else 0.5,
                "first_order_predicted_delta": -1.0,
                "first_order_predicted_positive_damage": 1.0,
            },
            "alternate_concept": {
                "aggregate": {
                    "write_abs_mean": 1.0 if variant == "original" else 3.0,
                    "read_abs_mean": 0.2 if variant == "original" else 0.4,
                    "support_oriented_read": 0.1 if variant == "original" else 0.3,
                    "first_order_predicted_delta": -0.5,
                    "first_order_predicted_positive_damage": 0.5,
                }
            },
        },
        "ablation": _effect(clean, ablated),
        "clean_clamped_swap": _effect(clean, swapped),
        "output_suppression": {
            "committed_concept": _effect(clean, suppressed),
            "alternate_concept": _effect(clean, clean),
        },
    }


def test_counterbalanced_effect_aggregation_uses_oriented_margins() -> None:
    original = _variant_record(
        "original",
        clean=3.0,
        ablated=1.0,
        swapped=-2.0,
        suppressed=3.0,
    )
    mirrored = _variant_record(
        "mirrored",
        clean=1.0,
        ablated=0.0,
        swapped=-1.0,
        suppressed=1.0,
    )
    result = aggregate_counterbalanced_variants([original, mirrored])

    assert result["clean_committed_margin"] == pytest.approx(2.0)
    assert result["attribution"]["write_abs_mean"] == pytest.approx(3.0)
    assert result["attribution"]["support_oriented_read"] == pytest.approx(0.4)
    assert result["attribution"]["committed_concept"][
        "write_abs_mean"
    ] == pytest.approx(3.0)
    assert result["attribution"]["alternate_concept"][
        "write_abs_mean"
    ] == pytest.approx(2.0)
    assert result["attribution"]["alternate_concept"]["read_abs_mean"] == pytest.approx(
        0.3
    )
    assert result["ablation"]["positive_damage"] == pytest.approx(1.5)
    assert result["clean_clamped_swap"]["edited_committed_margin"] == pytest.approx(
        -1.5
    )
    assert result["clean_clamped_swap"]["flips_committed_mean_margin"] is True
    assert result["clean_clamped_swap"]["variant_flips_committed_margin"] == [
        True,
        True,
    ]
    assert result["clean_clamped_swap"]["counterbalance_robust_flip"] is True
    assert result["output_suppression"]["committed_concept"]["positive_damage"] == 0.0
    assert (
        result["output_suppression"]["interpretation"]["status"]
        == "STRUCTURAL_ZERO_NEGATIVE_CONTROL"
    )
    assert result["internal_minus_suppression_positive_damage"] == pytest.approx(1.5)


class _TinyResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(13)
        self.embedding = nn.Embedding(16, 4)
        self.blocks = nn.ModuleList(
            [nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)]
        )
        self.head = nn.Linear(4, 6, bias=False)
        self.forward_calls = 0
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden = torch.tanh(block(hidden))
        return SimpleNamespace(logits=self.head(hidden))


def test_shared_attribution_projects_one_backward_onto_all_meta_directions() -> None:
    model = _TinyResidualModel().eval()
    first = F.normalize(torch.tensor([1.0, 2.0, 3.0, 4.0]), dim=0)
    second = F.normalize(torch.tensor([2.0, -1.0, 1.0, 3.0]), dim=0)
    third = F.normalize(torch.tensor([-1.0, 3.0, 2.0, 1.0]), dim=0)
    banks = {
        "committed_concept": {0: first, 1: second},
        "alternate_concept": {0: second, 1: first},
        "meta::meaning": {0: second, 1: third},
        "meta::ambiguous": {0: third, 1: first},
    }

    result = shared_direction_attribution(
        model,
        model.blocks,
        torch.tensor([[1, 2, 3]]),
        banks,
        target_token_id=4,
        foil_token_id=5,
    )

    assert model.forward_calls == 1
    assert result["n_forward_backward"] == 1
    assert set(result["direction_results"]) == set(banks)
    for direction in result["direction_results"].values():
        assert direction["write"][0].shape == (3,)
        assert direction["read"][1].shape == (3,)
        exact = -sum(
            float(np.sum(direction["write"][layer] * direction["read"][layer]))
            for layer in (0, 1)
        )
        assert direction["predicted_delta"] == pytest.approx(exact)
        assert direction["aggregate"]["read_abs_mean"] >= 0.0


def _analysis_rows() -> list[dict]:
    categories = (
        "lexical_ambiguity",
        "pp_attachment",
        "garden_path",
        "ambiguous_pronoun",
    )
    rows = []
    for index in range(40):
        damage = 0.8 + 0.02 * index
        suppression = 0.0
        write = 1.0 + 0.03 * index
        read = 0.2 + 0.01 * index
        rows.append(
            {
                "id": f"synthetic_{index:03d}",
                "category": categories[index % len(categories)],
                "measurement_status": "OK",
                "commitment": {"counterbalance_agreement": index % 5 != 0},
                "counterbalanced": {
                    "clean_committed_margin": 2.0,
                    "attribution": {
                        "write_abs_mean": write,
                        "read_abs_mean": abs(read),
                        "support_oriented_read": read,
                        "first_order_predicted_positive_damage": write * read,
                        "alternate_concept": {
                            "write_abs_mean": 0.8 * write,
                            "read_abs_mean": 0.7 * abs(read),
                            "support_oriented_read": 0.7 * read,
                            "first_order_predicted_positive_damage": (
                                0.56 * write * read
                            ),
                        },
                    },
                    "ablation": {"positive_damage": damage},
                    "clean_clamped_swap": {
                        "edited_committed_margin": -0.4 - 0.01 * index,
                        "flips_committed_mean_margin": True,
                        "counterbalance_robust_flip": index % 4 != 0,
                    },
                    "output_suppression": {
                        "committed_concept": {"positive_damage": suppression}
                    },
                    "internal_minus_suppression_positive_damage": damage - suppression,
                    "ablation_damage_exceeds_suppression": True,
                },
                "meta_counterbalanced": {
                    "meaning_en": {
                        "status": "OK",
                        "mean_write_abs": 0.7 * write,
                        "mean_independent_read_abs": 0.8 * abs(read),
                        "ablation": {"positive_damage": 0.6 * damage},
                    },
                    "ambiguity_en": {
                        "status": "OK",
                        "mean_write_abs": 0.9 * write,
                        "mean_independent_read_abs": 0.9 * abs(read),
                        "ablation": {"positive_damage": 0.7 * damage},
                    },
                },
            }
        )
    return rows


def test_p3_bootstraps_and_f8_plot_are_cpu_only() -> None:
    rows = _analysis_rows()
    result = analyze_p3(rows, n_bootstrap=200, seed=7)

    assert result["overall"]["n"] == 40
    assert result["overall"]["statistics"]["swap_flip_rate"]["estimate"] == 1.0
    assert result["overall"]["statistics"]["counterbalance_robust_swap_flip_rate"][
        "estimate"
    ] == pytest.approx(0.75)
    assert (
        result["overall"]["statistics"]["internal_minus_suppression_damage"]["ci_low"]
        > 0.0
    )
    assert result["verdict"] == "supported"
    assert result["analysis_role"] == "diagnostic_upstream_strict_g2_failed"
    assert result["upstream_gate_context"]["strict_g2_status"] == "FAIL"
    assert (
        result["output_suppression_interpretation"]["status"]
        == "STRUCTURAL_ZERO_NEGATIVE_CONTROL"
    )
    assert (
        result["output_suppression_interpretation"]["observed_all_exact_zero"] is True
    )
    assert (
        result["output_suppression_interpretation"][
            "observed_damage_gap_equals_ablation"
        ]
        is True
    )
    assert set(result["by_category"]) == {
        "lexical_ambiguity",
        "pp_attachment",
        "garden_path",
        "ambiguous_pronoun",
    }

    figure, axes = plot_f8(rows)
    try:
        assert len(axes) == 3
        assert "WRITE" in axes[0].get_xlabel()
        assert "Independent" in axes[0].get_ylabel()
        assert "Meta-token" in axes[1].get_xlabel()
        assert "interpretive meta-tokens" in axes[1].get_title()
        assert "Diagnostic" in axes[2].get_xlabel()
        assert "Real all-band" in axes[2].get_ylabel()
    finally:
        plt.close(figure)


def test_meta_analysis_tests_independent_read_against_real_ablation() -> None:
    rng = np.random.default_rng(23)
    rows = []
    for index in range(60):
        write = float(rng.uniform(0.2, 2.0))
        read = float(rng.uniform(0.1, 1.5))
        damage = 1.8 * read + 0.05 * write + float(rng.normal(scale=0.04))
        rows.append(
            {
                "meta_counterbalanced": {
                    "meaning_en": {
                        "status": "OK",
                        "mean_write_abs": write,
                        "mean_independent_read_abs": read,
                        "mean_first_order_predicted_positive_damage": write * read,
                        "mean_lens_rank": 20.0 + index,
                        "mean_final_output_rank": 100.0 + index,
                        "ablation": {"positive_damage": damage},
                    }
                }
            }
        )
    resolution = {
        "selected": [
            {
                "key": "meaning_en",
                "label": "meaning",
                "language": "en",
                "gloss": "meaning",
                "token_id": 11,
                "surface": " meaning",
            }
        ]
    }

    result = analyze_meta_tokens(rows, resolution, n_bootstrap=200, seed=29)
    stats = result["by_candidate"]["meaning_en"]
    assert result["primary_association"].endswith("read_vs_ablation_damage")
    assert stats["read_vs_ablation_damage"]["estimate"] > 0.98
    assert stats["partial_causal_read_given_write"]["estimate"] > 0.98
    assert "first_order_prediction_vs_ablation_damage" in stats


class _MetaTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        mapping = {
            " meaning": [11],
            "meaning": [12],
            " 含义": [21, 22],
            "含义": [23],
            " broken": [31, 32],
            "broken": [33, 34],
        }
        return mapping[text]


def test_meta_candidates_are_selected_by_tokenization_without_outcomes() -> None:
    candidates = (
        {"key": "meaning_en", "label": "meaning", "language": "en", "gloss": "meaning"},
        {"key": "meaning_zh", "label": "含义", "language": "zh", "gloss": "meaning"},
        {"key": "broken", "label": "broken", "language": "en", "gloss": "broken"},
    )
    result = resolve_meta_token_candidates(_MetaTokenizer(), candidates)

    assert result["selection_uses_model_outputs"] is False
    assert [item["key"] for item in result["selected"]] == [
        "meaning_en",
        "meaning_zh",
    ]
    assert result["selected"][0]["surface"] == " meaning"
    assert result["selected"][1]["surface"] == "含义"
    assert result["rejected"][0]["reason"] == "no_exact_single_token_surface"

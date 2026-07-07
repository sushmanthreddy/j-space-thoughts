from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

import src.localization_phase as localization_phase
from src.localization_phase import (
    CONCEPT_WEIGHT_READ_DEFINITION,
    MD_DIRECTION_METHOD,
    NON_ADDITIVE_WARNING,
    PRIMARY_DIRECTION_METHOD,
    analyze_population_weight_read,
    capture_qwen_components,
    choose_f4_candidates,
    choose_source_layers,
    component_grad_delta_scores,
    flag_top_components,
    localize_source_direction,
    plot_f4_localization,
    plot_f6_weight_read_robustness,
    qwen_attention_weight_read_with_null,
    run_population_weight_read,
    select_localization_subset,
    spearman_rank_agreement,
    summarize_concept_weight_read,
    validate_population_source_rows,
    weight_read_for_flagged_components,
)
from src.read_scores import qwen_head_ov_read


def _selection_row(name: str, write: float, read: float, damage: float) -> dict:
    return {
        "name": name,
        "measurement_status": "OK",
        "direction_method": PRIMARY_DIRECTION_METHOD,
        "aggregate": {
            "write_abs_mean": write,
            "read_abs_mean": read,
        },
        "ablation": {"positive_damage": damage},
    }


def test_quantile_subset_is_deterministic_spans_four_cells_and_assigns_no_class() -> (
    None
):
    rows = [
        _selection_row("hh-extreme", 10.0, 10.0, 4.0),
        _selection_row("hh-inner", 9.0, 9.0, 3.0),
        _selection_row("hl-extreme", 10.0, 1.0, 0.2),
        _selection_row("hl-inner", 9.0, 2.0, 0.3),
        _selection_row("lh-extreme", 1.0, 10.0, 5.0),
        _selection_row("lh-inner", 2.0, 9.0, 4.0),
        _selection_row("ll-extreme", 1.0, 1.0, 0.1),
        _selection_row("ll-inner", 2.0, 2.0, 0.2),
        _selection_row("ignored-md", 100.0, 100.0, 100.0)
        | {"direction_method": "mean_difference"},
        _selection_row("ignored-error", 100.0, 100.0, 100.0)
        | {"measurement_status": "ERROR"},
    ]

    first = select_localization_subset(rows)
    repeated = select_localization_subset(list(reversed(rows)))

    assert [row["name"] for row in first["selected"]] == [
        "hh-extreme",
        "hl-extreme",
        "lh-extreme",
        "ll-extreme",
    ]
    assert [row["name"] for row in repeated["selected"]] == [
        row["name"] for row in first["selected"]
    ]
    assert {row["cell"] for row in first["selected"]} == {
        "high_write_high_read",
        "high_write_low_read",
        "low_write_high_read",
        "low_write_low_read",
    }
    assert all(row["strict_threshold_match"] for row in first["selected"])
    assert first["provenance"]["n_eligible_raw_rows"] == 8
    assert "No item is assigned" in first["provenance"]["class_guardrail"]

    candidates = choose_f4_candidates(first)
    assert candidates["driver_candidate"]["name"] == "lh-extreme"
    assert candidates["low_read_candidate"]["name"] == "hl-extreme"
    assert "not declared narration" in candidates["guardrail"]


def test_source_layer_selection_recomputes_exact_signed_products() -> None:
    row = {
        "name": "source-test",
        "raw_arrays": {
            "write_by_layer_position": {
                "3": [2.0, -1.0],
                "4": [3.0, 2.0],
                "5": [100.0, 100.0],
            },
            "attribution_read_by_layer_position": {
                "3": [0.5, 2.0],
                "4": [1.0, 1.0],
                "5": [1.0, 1.0],
            },
        },
    }

    result = choose_source_layers(row, n_source_layers=2, max_source_layer=4)

    assert result["selected_layers"] == [4, 3]
    assert result["selected"][0]["signed_first_order_positive_damage"] == 5.0
    assert result["selected"][1]["signed_first_order_positive_damage"] == -1.0
    assert result["selected"][1]["selection_score"] == 1.0
    assert "targeting only" in result["role"]


def test_component_grad_delta_formula_separates_heads_and_rejects_bad_shapes() -> None:
    clean_mlp = torch.zeros(1, 2, 4)
    perturbed_mlp = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]]])
    gradient_mlp = torch.full((1, 2, 4), 2.0)
    clean_attention = torch.zeros(1, 2, 4)
    perturbed_attention = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    gradient_attention = torch.ones(1, 2, 4)
    clean = {
        "mlp": {1: clean_mlp},
        "attention_pre_o_proj": {1: clean_attention},
    }
    perturbed = {
        "mlp": {1: perturbed_mlp},
        "attention_pre_o_proj": {1: perturbed_attention},
    }
    gradients = {
        "mlp": {1: gradient_mlp},
        "attention_pre_o_proj": {1: gradient_attention},
    }

    result = component_grad_delta_scores(
        clean,
        perturbed,
        gradients,
        head_geometry={1: (2, 2)},
    )

    assert result["mlps"][0]["score"] == pytest.approx(
        float((perturbed_mlp * gradient_mlp).sum())
    )
    assert [row["score"] for row in result["attention_heads"]] == pytest.approx(
        [1.0 + 2.0 + 5.0 + 6.0, 3.0 + 4.0 + 7.0 + 8.0]
    )
    assert result["attention_heads"][0]["score_by_position"] == pytest.approx(
        [3.0, 11.0]
    )

    bad = copy.deepcopy(perturbed)
    bad["mlp"][1] = torch.zeros(1, 2, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        component_grad_delta_scores(
            clean,
            bad,
            gradients,
            head_geometry={1: (2, 2)},
        )


def _tiny_qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(3)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_tiny_qwen_localization_weight_nulls_cleanup_and_f4(tmp_path) -> None:
    model = _tiny_qwen()
    blocks = model.model.layers
    input_ids = torch.tensor([[1, 7, 9, 4]])
    direction = F.normalize(torch.arange(1, 17, dtype=torch.float32), dim=0)
    original_mlp_hooks = [len(block.mlp._forward_hooks) for block in blocks]
    original_attention_hooks = [
        len(block.self_attn.o_proj._forward_pre_hooks) for block in blocks
    ]

    with pytest.raises(RuntimeError, match="deliberate"):
        with capture_qwen_components(blocks, [1, 2], start_graph=False):
            raise RuntimeError("deliberate hook-scope failure")
    assert [len(block.mlp._forward_hooks) for block in blocks] == original_mlp_hooks
    assert [
        len(block.self_attn.o_proj._forward_pre_hooks) for block in blocks
    ] == original_attention_hooks

    localization = localize_source_direction(
        model,
        blocks,
        input_ids,
        direction,
        source_layer=0,
        target_token_id=5,
        foil_token_id=6,
    )

    assert len(localization["mlps"]) == 3
    assert len(localization["attention_heads"]) == 12
    assert localization["component_layers"] == [1, 2, 3]
    assert np.isfinite(localization["actual_delta"])
    assert localization["positive_damage"] == pytest.approx(
        -localization["actual_delta"]
    )
    assert NON_ADDITIVE_WARNING in localization["localization_estimator"]["warning"]
    assert [len(block.mlp._forward_hooks) for block in blocks] == original_mlp_hooks
    assert [
        len(block.self_attn.o_proj._forward_pre_hooks) for block in blocks
    ] == original_attention_hooks

    flagged = flag_top_components(
        localization,
        top_k_mlps=3,
        top_k_heads=4,
    )
    weight = weight_read_for_flagged_components(
        blocks,
        direction,
        flagged,
        label_direction=direction,
        n_random=5,
        seed=19,
    )
    assert len(weight["mlps"]) == 3
    assert len(weight["attention_heads"]) == 4
    assert weight["metadata"]["raw_random_nulls_retained"] is True
    assert weight["metadata"]["direction"] == (
        "same supplied source-layer unit direction"
    )
    assert "raw" not in weight["metadata"]["direction"].lower()
    for row in weight["mlps"]:
        assert len(row["random_gains"]) == 5
        assert row["normalized_gain"] == pytest.approx(
            row["gain"] / row["random_median"]
        )
    for row in weight["attention_heads"]:
        assert len(row["random_ov_norms"]) == 5
        assert len(row["random_label_cosines"]) == 5
        assert row["normalized_ov_norm"] == pytest.approx(
            row["ov_norm"] / row["random_median_ov_norm"]
        )
        assert -1.0 <= row["label_cosine"] <= 1.0

    null_seed = 23
    direct_null = qwen_attention_weight_read_with_null(
        blocks[1].self_attn,
        direction,
        label_direction=direction,
        n_random=3,
        seed=null_seed,
    )
    generator = torch.Generator(device="cpu").manual_seed(null_seed)
    first_random = F.normalize(
        torch.randn(3, direction.numel(), generator=generator)[0], dim=0
    )
    first_helper = qwen_head_ov_read(
        blocks[1].self_attn,
        first_random,
        label_direction=direction,
    )
    for vectorized, scalar in zip(direct_null, first_helper, strict=True):
        assert vectorized["random_ov_norms"][0] == pytest.approx(
            scalar["ov_norm"], rel=1e-5, abs=1e-7
        )
        assert vectorized["random_label_cosines"][0] == pytest.approx(
            scalar["label_cosine"], rel=1e-5, abs=1e-7
        )

    second = copy.deepcopy(localization)
    for row in second["mlps"]:
        row["score"] *= -0.5
    for row in second["attention_heads"]:
        row["score"] *= -0.5
    records = [
        {
            "name": "driver",
            "source_layer": 0,
            "source_selection_rank": 1,
            "localization": localization,
        },
        {
            "name": "low-read",
            "source_layer": 0,
            "source_selection_rank": 1,
            "localization": second,
        },
    ]
    candidates = {
        "driver_candidate": {"name": "driver"},
        "low_read_candidate": {"name": "low-read"},
    }
    path = plot_f4_localization(records, candidates, tmp_path / "f4.png")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_spearman_rank_agreement_reports_exact_monotonic_order_and_null_case() -> None:
    increasing = [
        {"component": f"c{index}", "abs_score": float(index), "weight": index**2}
        for index in range(1, 6)
    ]
    result = spearman_rank_agreement(increasing, weight_key="weight")
    assert result["status"] == "ESTIMATED"
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n"] == 5
    assert result["paired_ranks"][0]["component_id"] == "c5"

    constant = [
        {"component": f"c{index}", "abs_score": float(index), "weight": 1.0}
        for index in range(1, 6)
    ]
    not_estimable = spearman_rank_agreement(constant, weight_key="weight")
    assert not_estimable["status"] == "NOT_ESTIMABLE"
    assert "constant" in not_estimable["reason"]


def _population_source_row(name: str, method: str, offset: float) -> dict:
    return {
        "name": name,
        "measurement_status": "OK",
        "direction_method": method,
        "direction_convention": (
            "normalize(W_U[token] @ J_layer)"
            if method == PRIMARY_DIRECTION_METHOD
            else "matched-slot mean difference"
        ),
        "aggregate": {
            "write_abs_mean": 1.0 + offset,
            "read_abs_mean": 0.2 + offset,
        },
        "clean_metric": 1.0,
        "ablation": {"positive_damage": 0.5 + offset},
        "raw_arrays": {
            "write_by_layer_position": {
                "1": [1.0 + offset, 2.0],
                "2": [0.5, 0.25],
            },
            "attribution_read_by_layer_position": {
                "1": [0.5, 1.0],
                "2": [0.1, 0.1],
            },
        },
        "prompt": f"Prompt {name}",
        "prompt_token_ids": [1, 2],
        "n_prompt_tokens": 2,
        "intervention_positions": [0, 1],
        "token_ids": {"concept": 3, "target": 4, "foil": 5},
        "token_surfaces": {"concept": " concept"},
        "intermediate": "concept",
        "workspace_layers": [1, 2],
    }


def test_population_source_validation_requires_complete_raw_and_md_counts() -> None:
    rows = [
        _population_source_row("raw-a", PRIMARY_DIRECTION_METHOD, 0.0),
        _population_source_row("raw-b", PRIMARY_DIRECTION_METHOD, 0.1),
        _population_source_row("md-a", MD_DIRECTION_METHOD, 0.2),
    ]
    payload = {
        "schema_version": "twohop-phase-v1",
        "metadata": {
            "primary_direction": PRIMARY_DIRECTION_METHOD,
            "rms_gain_folded_included": False,
            "workspace_layers": [1, 2],
        },
        "rows": rows,
        "sample_counts": {
            "n_by_method": {
                PRIMARY_DIRECTION_METHOD: {"successful": 2},
                MD_DIRECTION_METHOD: {"successful": 1},
            }
        },
        "analyses": {
            "ablation": {
                "by_method": {
                    PRIMARY_DIRECTION_METHOD: {"n": 2},
                    MD_DIRECTION_METHOD: {"n": 1},
                }
            }
        },
    }

    result = validate_population_source_rows(
        payload,
        expected_counts={PRIMARY_DIRECTION_METHOD: 2, MD_DIRECTION_METHOD: 1},
    )

    assert [row["name"] for row in result[PRIMARY_DIRECTION_METHOD]] == [
        "raw-a",
        "raw-b",
    ]
    assert [row["name"] for row in result[MD_DIRECTION_METHOD]] == ["md-a"]
    drifted = copy.deepcopy(payload)
    drifted["sample_counts"]["n_by_method"][MD_DIRECTION_METHOD]["successful"] = 2
    with pytest.raises(ValueError, match="Population count drift"):
        validate_population_source_rows(drifted)

    incomplete = copy.deepcopy(payload)
    del incomplete["rows"][0]["raw_arrays"]["write_by_layer_position"]["2"]
    with pytest.raises(ValueError, match="WRITE/READ layers do not align"):
        validate_population_source_rows(incomplete)

    short = copy.deepcopy(payload)
    short["rows"][0]["raw_arrays"]["write_by_layer_position"]["1"] = [1.0]
    short["rows"][0]["raw_arrays"]["attribution_read_by_layer_position"]["1"] = [0.5]
    with pytest.raises(ValueError, match="attribution-position coverage drift"):
        validate_population_source_rows(short)


def test_concept_weight_read_uses_frozen_component_means_and_32_nulls() -> None:
    weight = {
        "metadata": {"activation_independent": True, "n_random": 32},
        "mlps": [
            {"component": "L2.MLP", "normalized_gain": 1.0},
            {"component": "L3.MLP", "normalized_gain": 3.0},
        ],
        "attention_heads": [
            {"component": f"L2.H{head}", "label_weighted_normalized_ov": value}
            for head, value in enumerate((0.5, 1.0, 1.5, 2.0))
        ],
    }

    result = summarize_concept_weight_read(weight)

    assert result["mlp"]["estimate"] == pytest.approx(2.0)
    assert result["attention"]["estimate"] == pytest.approx(1.25)
    assert result["selection_conditioned"] is True
    assert result["definition"] == CONCEPT_WEIGHT_READ_DEFINITION
    with pytest.raises(ValueError, match="32 random"):
        summarize_concept_weight_read(
            weight | {"metadata": {"activation_independent": True, "n_random": 31}}
        )


def _population_analysis_records(method: str, *, seed: int, n: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    write = rng.uniform(0.5, 2.0, size=n)
    mlp = rng.uniform(0.3, 1.8, size=n)
    attention = rng.uniform(0.2, 1.6, size=n)
    causal = 1.7 * attention + 1.2 * mlp + 0.05 * write + rng.normal(scale=0.08, size=n)
    return [
        {
            "name": f"{method}-{index:03d}",
            "direction_method": method,
            "notebook02_summary": {
                "write_strength": float(write[index]),
                "attribution_read_strength": float(attention[index]),
                "all_band_ablation_positive_damage": float(causal[index]),
            },
            "concept_weight_read": {
                "mlp": {"estimate": float(mlp[index])},
                "attention": {"estimate": float(attention[index])},
            },
        }
        for index in range(n)
    ]


def _synthetic_twohop_for_f6() -> dict:
    def method(n: int, estimate: float) -> dict:
        return {
            "n": n,
            "partial_correlations": {
                "causal_read_given_write": {
                    "status": "ESTIMATED",
                    "estimate": estimate,
                    "ci_low": estimate - 0.1,
                    "ci_high": estimate + 0.1,
                },
                "causal_write_given_read": {
                    "status": "ESTIMATED",
                    "estimate": 0.05,
                    "ci_low": -0.1,
                    "ci_high": 0.2,
                },
            },
        }

    return {
        "analyses": {
            "ablation": {
                "by_method": {
                    PRIMARY_DIRECTION_METHOD: method(60, 0.7),
                    MD_DIRECTION_METHOD: method(30, 0.6),
                }
            }
        }
    }


def test_population_weight_analysis_and_f6_cover_both_methods_and_families(
    tmp_path,
) -> None:
    records = [
        *_population_analysis_records(PRIMARY_DIRECTION_METHOD, seed=31, n=60),
        *_population_analysis_records(MD_DIRECTION_METHOD, seed=37, n=30),
    ]

    analysis = analyze_population_weight_read(records, n_bootstrap=200, seed=41)

    assert set(analysis["by_method"]) == {
        PRIMARY_DIRECTION_METHOD,
        MD_DIRECTION_METHOD,
    }
    for method in (PRIMARY_DIRECTION_METHOD, MD_DIRECTION_METHOD):
        for family in ("mlp", "attention"):
            result = analysis["by_method"][method]["weight_families"][family]
            assert (
                result["partial_correlations"]["causal_weight_read_given_write"][
                    "status"
                ]
                == "ESTIMATED"
            )
            assert (
                result["partial_correlations"]["causal_write_given_weight_read"][
                    "status"
                ]
                == "ESTIMATED"
            )
            assert result["standardized_regression"]["status"] == "ESTIMATED"
    path = plot_f6_weight_read_robustness(
        _synthetic_twohop_for_f6(),
        analysis,
        tmp_path / "f6.png",
    )
    assert path.is_file()
    assert path.stat().st_size > 10_000


class _PopulationTokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        del prompt, add_special_tokens
        return [1, 2]

    def __call__(self, prompt: str, *, return_tensors: str, truncation: bool):
        del prompt
        assert return_tensors == "pt"
        assert truncation is False
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
        )


def test_population_orchestration_localizes_once_and_never_reuses_raw_for_md(
    tmp_path,
    monkeypatch,
) -> None:
    raw_a = _population_source_row("raw-a", PRIMARY_DIRECTION_METHOD, 0.0)
    raw_b = _population_source_row("raw-b", PRIMARY_DIRECTION_METHOD, 0.1)
    md_a = _population_source_row("md-a", MD_DIRECTION_METHOD, 0.2)
    artifact_path = tmp_path / "md.pt"
    payload = {
        "schema_version": "twohop-phase-v1",
        "metadata": {
            "model_id": "test/model",
            "model_revision": "revision",
            "primary_direction": PRIMARY_DIRECTION_METHOD,
            "rms_gain_folded_included": False,
            "workspace_layers": [1, 2],
        },
        "rows": [raw_a, raw_b, md_a],
        "sample_counts": {
            "n_by_method": {
                PRIMARY_DIRECTION_METHOD: {"successful": 2},
                MD_DIRECTION_METHOD: {"successful": 1},
            }
        },
        "analyses": {
            "ablation": {
                "by_method": {
                    PRIMARY_DIRECTION_METHOD: {"n": 2},
                    MD_DIRECTION_METHOD: {"n": 1},
                }
            }
        },
        "direction_coverage": {
            MD_DIRECTION_METHOD: {"artifact": {"path": str(artifact_path)}}
        },
    }
    model = torch.nn.Linear(2, 2, bias=False)
    bundle = SimpleNamespace(
        model_id="test/model",
        revision="revision",
        hf_model=model,
        tokenizer=_PopulationTokenizer(),
        lens_model=SimpleNamespace(
            d_model=2,
            n_layers=4,
            layers=[object(), object(), object(), object()],
        ),
    )
    lens = SimpleNamespace(d_model=2, source_layers=[1, 2])
    raw_direction = torch.tensor([1.0, 0.0])
    md_direction = torch.tensor([0.0, 1.0])
    monkeypatch.setattr(localization_phase, "validate_lens", lambda lens, model: None)
    monkeypatch.setattr(
        localization_phase,
        "jlens_direction_bank",
        lambda *args, **kwargs: {3: {1: raw_direction, 2: raw_direction}},
    )
    monkeypatch.setattr(
        localization_phase,
        "load_mean_difference_artifact",
        lambda *args, **kwargs: {
            "path": str(artifact_path.resolve()),
            "mean_difference": {"concept": {1: md_direction, 2: md_direction}},
            "canonical_lookup": {"concept": "concept"},
            "d_model_by_layer": {1: 2, 2: 2},
            "metadata": {"model_id": "test/model"},
            "layers": [1, 2],
            "source_dtypes": ["torch.float32"],
            "max_source_unit_norm_error": 0.0,
            "n_concepts": 1,
        },
    )
    payload["direction_coverage"][MD_DIRECTION_METHOD]["artifact"] = {
        "path": str(artifact_path.resolve()),
        "d_model_by_layer": {"1": 2, "2": 2},
        "metadata": {"model_id": "test/model"},
        "layers": [1, 2],
        "source_dtypes": ["torch.float32"],
        "max_source_unit_norm_error": 0.0,
        "n_concepts": 1,
    }
    directions_seen: list[torch.Tensor] = []

    def fake_localize(
        hf_model, blocks, input_ids, direction, *, source_layer, **kwargs
    ):
        del hf_model, blocks, input_ids, kwargs
        directions_seen.append(direction.detach().cpu())
        return {
            "source_layer": source_layer,
            "component_layers": [2, 3],
            "clean_metric": 1.0,
            "positive_damage": 0.25,
            "mlps": [
                {"component": "L2.MLP", "layer": 2, "score": 2.0, "abs_score": 2.0},
                {"component": "L3.MLP", "layer": 3, "score": 1.0, "abs_score": 1.0},
            ],
            "attention_heads": [
                {
                    "component": f"L2.H{head}",
                    "layer": 2,
                    "head": head,
                    "score": float(4 - head),
                    "abs_score": float(4 - head),
                }
                for head in range(4)
            ],
        }

    def fake_weight(blocks, direction, flagged, **kwargs):
        del blocks, direction, kwargs
        return {
            "metadata": {
                "activation_independent": True,
                "direction": "same supplied source-layer unit direction",
                "n_random": 32,
            },
            "mlps": [
                {
                    **row,
                    "normalized_gain": float(index + 1),
                    "random_gains": [1.0] * 32,
                }
                for index, row in enumerate(flagged["mlps"])
            ],
            "attention_heads": [
                {
                    **row,
                    "label_weighted_normalized_ov": float(index + 1),
                    "random_ov_norms": [1.0] * 32,
                    "random_label_cosines": [0.0] * 32,
                }
                for index, row in enumerate(flagged["attention_heads"])
            ],
        }

    monkeypatch.setattr(localization_phase, "localize_source_direction", fake_localize)
    monkeypatch.setattr(
        localization_phase,
        "weight_read_for_flagged_components",
        fake_weight,
    )

    result = run_population_weight_read(
        bundle,
        lens,
        payload,
        md_artifact_path=artifact_path,
        n_bootstrap=200,
        expected_counts={PRIMARY_DIRECTION_METHOD: 2, MD_DIRECTION_METHOD: 1},
    )

    assert result["sample_counts"]["n_total"] == 3
    assert len(directions_seen) == 3
    assert torch.equal(directions_seen[0], raw_direction)
    assert torch.equal(directions_seen[1], raw_direction)
    assert torch.equal(directions_seen[2], md_direction)
    md_record = result["records"][2]
    assert md_record["direction_method"] == MD_DIRECTION_METHOD
    assert md_record["direction_source"]["uses_raw_jlens_vector"] is False
    assert "raw" not in md_record["weight_read"]["metadata"]["direction"].lower()

    mismatched = copy.deepcopy(payload)
    mismatched["direction_coverage"][MD_DIRECTION_METHOD]["artifact"]["n_concepts"] = 2
    with pytest.raises(ValueError, match="artifact provenance"):
        run_population_weight_read(
            bundle,
            lens,
            mismatched,
            md_artifact_path=artifact_path,
            n_bootstrap=200,
            expected_counts={PRIMARY_DIRECTION_METHOD: 2, MD_DIRECTION_METHOD: 1},
        )

    clean_drift = copy.deepcopy(payload)
    clean_drift["rows"][0]["clean_metric"] = 2.0
    with pytest.raises(ValueError, match="clean metric drift"):
        run_population_weight_read(
            bundle,
            lens,
            clean_drift,
            md_artifact_path=artifact_path,
            n_bootstrap=200,
            expected_counts={PRIMARY_DIRECTION_METHOD: 2, MD_DIRECTION_METHOD: 1},
        )

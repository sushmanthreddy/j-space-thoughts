"""Executable correctness gates for notebook 00."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import torch

from src.data_gen import (
    G1_PROMPTS,
    deterministic_subset,
    tokenize_twohop_item,
    tokenizable_twohop_items,
)
from src.interventions import (
    ablation_edits,
    clamped_swap_edits,
    forward_logits,
    suppress_output_token,
)
from src.jlens_iface import (
    jlens_direction_bank,
    load_published_lens,
    token_rank,
    workspace_layers,
    write_by_position,
)
from src.metrics import (
    logit_difference,
    pearson_with_ci,
    save_json,
    signed_causal_delta,
)
from src.model_utils import (
    ModelBundle,
    capture_residuals,
    decode_topk,
    hf_wrapper_logit_kl,
    load_model,
    set_seed,
)
from src.plotting import save_figure, validation_scatter
from src.read_scores import attribution_read


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729


def _metric(logits: torch.Tensor, item: dict) -> float:
    return float(
        logit_difference(logits, item["target_token_id"], item["foil_token_id"])[
            0
        ].cpu()
    )


def _spider_item(tokenizer: Any) -> dict:
    return tokenize_twohop_item(
        tokenizer,
        {
            "name": "spider-legs",
            "category": "multihop",
            "prompt": "Fact: The number of legs on the animal that spins webs is ",
            "intermediate": "spider",
            "answer": "8",
            "swap_to": "ant",
            "swap_answer": "6",
        },
    )


def run_g1(bundle: ModelBundle) -> dict:
    """G1: exact wrapper reconstruction on all tokens of 20 prompts."""

    items = hf_wrapper_logit_kl(bundle, G1_PROMPTS)
    summary = {
        "n": len(items),
        "threshold_mean_kl": 1e-3,
        "max_prompt_mean_kl": max(item["mean_kl"] for item in items),
        "max_position_kl": max(item["max_kl"] for item in items),
        "max_abs_logit_error": max(item["max_abs_logit_error"] for item in items),
        "items": items,
    }
    summary["status"] = (
        "PASS" if summary["max_prompt_mean_kl"] < summary["threshold_mean_kl"] else "FAIL"
    )
    print(
        f"G1 {summary['status']}: N={summary['n']}, "
        f"max mean KL={summary['max_prompt_mean_kl']:.3g}, "
        f"max position KL={summary['max_position_kl']:.3g}"
    )
    return summary


def run_g2(bundle: ModelBundle, lens: Any, layers: list[int]) -> dict:
    """G2: spider WRITE plus clean-clamped spider/ant coordinate swap."""

    item = _spider_item(bundle.tokenizer)
    input_ids = bundle.lens_model.encode(item["prompt"])
    clean_logits = forward_logits(bundle.hf_model, input_ids)
    clean_metric = _metric(clean_logits, item)
    clean_top = decode_topk(bundle.tokenizer, clean_logits[0, -1], 20)
    residuals = capture_residuals(bundle.lens_model, input_ids, layers)
    token_ids = [item["concept_token_id"], item["foil_concept_token_id"]]

    raw_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        layers,
        fold_rms_gain=False,
    )
    effective_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        layers,
        fold_rms_gain=True,
    )
    raw_spider = raw_bank[item["concept_token_id"]]
    raw_ant = raw_bank[item["foil_concept_token_id"]]
    effective_spider = effective_bank[item["concept_token_id"]]
    effective_ant = effective_bank[item["foil_concept_token_id"]]

    raw_write = write_by_position(residuals, raw_spider)
    effective_write = write_by_position(residuals, effective_spider)
    lens_logits, _, _ = lens.apply(
        bundle.lens_model,
        item["prompt"],
        layers=layers,
        positions=None,
    )
    ranks = {
        layer: [
            token_rank(lens_logits[layer][position], item["concept_token_id"])
            for position in range(lens_logits[layer].shape[0])
        ]
        for layer in layers
    }

    ablated_logits = forward_logits(
        bundle.hf_model,
        input_ids,
        blocks=bundle.lens_model.layers,
        edits=ablation_edits(effective_spider),
    )
    raw_swap_logits = forward_logits(
        bundle.hf_model,
        input_ids,
        blocks=bundle.lens_model.layers,
        edits=clamped_swap_edits(residuals, raw_spider, raw_ant),
    )
    effective_swap_logits = forward_logits(
        bundle.hf_model,
        input_ids,
        blocks=bundle.lens_model.layers,
        edits=clamped_swap_edits(residuals, effective_spider, effective_ant),
    )
    suppressed_logits = suppress_output_token(clean_logits, item["concept_token_id"])

    variants = {}
    for name, logits in {
        "ablation_effective": ablated_logits,
        "swap_raw_WU_J": raw_swap_logits,
        "swap_rms_gain_folded": effective_swap_logits,
        "output_suppression": suppressed_logits,
    }.items():
        metric = _metric(logits, item)
        variants[name] = {
            "metric": metric,
            "delta": signed_causal_delta(clean_metric, metric),
            "target_rank": token_rank(logits[0, -1], item["target_token_id"]),
            "foil_rank": token_rank(logits[0, -1], item["foil_token_id"]),
            "top_tokens": decode_topk(bundle.tokenizer, logits[0, -1], 20),
        }

    min_rank = min(min(values) for values in ranks.values())
    clean_top_id = int(clean_logits[0, -1].argmax())
    swapped_top_id = int(effective_swap_logits[0, -1].argmax())
    concept_output_rank = token_rank(clean_logits[0, -1], item["concept_token_id"])
    high_write = min_rank <= 10
    strict_pass = (
        clean_top_id == item["target_token_id"]
        and high_write
        and swapped_top_id == item["foil_token_id"]
    )
    directional_pass = (
        variants["swap_rms_gain_folded"]["metric"] < 0
        and variants["swap_rms_gain_folded"]["foil_rank"]
        < variants["swap_rms_gain_folded"]["target_rank"]
    )
    summary = {
        "status": "PASS" if strict_pass else "FAIL",
        "directional_subgate": "PASS" if directional_pass else "FAIL",
        "strict_criterion": "clean top-1=8, spider J-Lens rank<=10, swapped top-1=6",
        "direction_conventions": {
            "primary_spec": "normalize(J.T @ lm_head[token])",
            "rms_gain_folded_sensitivity": (
                "normalize(J.T @ (lm_head[token] * final_rmsnorm_gain)); "
                "rank-equivalent to official lens.apply for Qwen"
            ),
            "multi_layer_swap": (
                "At every band layer, clamp current coefficients to the clean "
                "pass's swapped coefficients; preserve current orthogonal complement"
            ),
        },
        "item": item,
        "workspace_layers": layers,
        "clean_metric": clean_metric,
        "clean_top_tokens": clean_top,
        "clean_concept_output_rank": concept_output_rank,
        "min_spider_jlens_rank": min_rank,
        "jlens_ranks_by_layer": ranks,
        "raw_write_by_layer_position": {
            str(layer): [float(value) for value in raw_write[layer][0].cpu()]
            for layer in layers
        },
        "effective_write_by_layer_position": {
            str(layer): [float(value) for value in effective_write[layer][0].cpu()]
            for layer in layers
        },
        "variants": variants,
    }
    swapped = variants["swap_rms_gain_folded"]
    print(
        f"G2 {summary['status']} (directional subgate {summary['directional_subgate']}): "
        f"min spider rank={min_rank}, clean M={clean_metric:.3f}, "
        f"swap M={swapped['metric']:.3f}, swapped top={swapped['top_tokens'][0]['token']!r}"
    )
    if not strict_pass:
        print(
            "G2 failure documented: Qwen-7B does not reproduce the strict known-answer "
            "top-1 swap under the preregistered band. Downstream 7B results are diagnostic."
        )
    return summary


def _clean_validation_pool(bundle: ModelBundle, *, top_k: int = 10) -> tuple[list[dict], list[dict]]:
    tokenizable, rejected = tokenizable_twohop_items(bundle.tokenizer)
    accepted: list[dict] = []
    for item in tokenizable:
        input_ids = bundle.lens_model.encode(item["prompt"])
        logits = forward_logits(bundle.hf_model, input_ids)
        top_ids = logits[0, -1].topk(top_k).indices.tolist()
        reason = None
        if int(logits[0, -1].argmax()) != item["target_token_id"]:
            reason = "clean_top1_not_target"
        elif item["concept_token_id"] in top_ids:
            reason = f"concept_in_clean_top_{top_k}"
        elif item["foil_concept_token_id"] in top_ids:
            reason = f"foil_concept_in_clean_top_{top_k}"
        if reason is None:
            accepted.append(item)
        else:
            rejected.append({"name": item["name"], "reason": reason})
    return accepted, rejected


def run_g3(
    bundle: ModelBundle,
    lens: Any,
    layers: list[int],
    *,
    n_items: int = 20,
) -> dict:
    """G3: predicted versus real ablation on a held-out two-hop subset."""

    pool, rejected = _clean_validation_pool(bundle)
    if len(pool) < n_items:
        raise RuntimeError(
            f"Only {len(pool)} clean/token-valid items remain; need {n_items} for G3"
        )
    selected = deterministic_subset(pool, n_items, seed=SEED)
    direction_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        [item["concept_token_id"] for item in selected],
        layers,
        fold_rms_gain=True,
    )

    measurements: list[dict] = []
    for index, item in enumerate(selected, start=1):
        input_ids = bundle.lens_model.encode(item["prompt"])
        directions = direction_bank[item["concept_token_id"]]
        attribution = attribution_read(
            bundle.hf_model,
            bundle.lens_model.layers,
            input_ids,
            directions,
            target_token_id=item["target_token_id"],
            foil_token_id=item["foil_token_id"],
            intervention_positions=None,
        )
        clean_logits = forward_logits(bundle.hf_model, input_ids)
        ablated_logits = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=ablation_edits(directions),
        )
        suppressed_logits = suppress_output_token(
            clean_logits, item["concept_token_id"]
        )
        clean_metric = _metric(clean_logits, item)
        ablated_metric = _metric(ablated_logits, item)
        suppression_metric = _metric(suppressed_logits, item)
        measurements.append(
            {
                "item": item,
                "n_prompt_tokens": int(input_ids.shape[1]),
                "clean_metric": clean_metric,
                "ablated_metric": ablated_metric,
                "actual_delta": signed_causal_delta(clean_metric, ablated_metric),
                "predicted_delta": attribution.predicted_delta,
                "predicted_delta_by_layer": {
                    str(layer): value
                    for layer, value in attribution.predicted_delta_by_layer.items()
                },
                "write_by_layer_position": {
                    str(layer): [float(value) for value in values]
                    for layer, values in attribution.write.items()
                },
                "read_by_layer_position": {
                    str(layer): [float(value) for value in values]
                    for layer, values in attribution.read.items()
                },
                "suppression_metric": suppression_metric,
                "suppression_delta": signed_causal_delta(
                    clean_metric, suppression_metric
                ),
                "clean_top_tokens": decode_topk(
                    bundle.tokenizer, clean_logits[0, -1], 10
                ),
                "ablated_top_tokens": decode_topk(
                    bundle.tokenizer, ablated_logits[0, -1], 10
                ),
            }
        )
        print(
            f"G3 {index:02d}/{n_items} {item['name']}: "
            f"pred={measurements[-1]['predicted_delta']:.3f}, "
            f"actual={measurements[-1]['actual_delta']:.3f}"
        )

    predicted = [item["predicted_delta"] for item in measurements]
    actual = [item["actual_delta"] for item in measurements]
    correlation = pearson_with_ci(
        predicted,
        actual,
        n_bootstrap=5000,
        confidence=0.95,
        seed=SEED,
    )
    reliable = correlation["estimate"] >= 0.5 and correlation["ci_low"] > 0
    figure, _ = validation_scatter(
        predicted,
        actual,
        correlation=correlation["estimate"],
        n=n_items,
    )
    figure_path = save_figure(
        figure, ROOT / "results" / "figures" / "f5_attribution_vs_ablation_qwen7b.png"
    )
    summary = {
        "status": "PASS",
        "meaning": "validation computed; reliability is a measured outcome, not a pass condition",
        "attribution_reliable": reliable,
        "reliability_rule": "Pearson r >= 0.5 and 95% bootstrap CI lower bound > 0",
        "n": n_items,
        "seed": SEED,
        "workspace_layers": layers,
        "clean_pool_n": len(pool),
        "rejected": rejected,
        "correlation": correlation,
        "figure": str(figure_path.relative_to(ROOT)),
        "items": measurements,
    }
    print(
        f"G3 PASS (computed): N={n_items}, r={correlation['estimate']:.3f} "
        f"95% CI [{correlation['ci_low']:.3f}, {correlation['ci_high']:.3f}], "
        f"attribution_reliable={reliable}"
    )
    return summary


def run_all_gates(*, model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    """Run notebook-00 gates, save raw measurements, and return the payload."""

    set_seed(SEED)
    bundle = load_model(model_id)
    lens = load_published_lens(model_id)
    layers = workspace_layers(bundle.lens_model.n_layers, lens.source_layers)
    payload: dict[str, Any] = {
        "metadata": {
            "seed": SEED,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "model_id": model_id,
            "model_revision": bundle.revision,
            "lens_n_prompts": lens.n_prompts,
            "lens_source_layers": lens.source_layers,
            "workspace_layers": layers,
            "effect_sign": "delta_M = M_edited - M_clean",
        },
        "gates": {},
    }
    payload["gates"]["g1"] = run_g1(bundle)
    if payload["gates"]["g1"]["status"] != "PASS":
        raise RuntimeError("G1 failed; refusing to run activation interventions")
    payload["gates"]["g2"] = run_g2(bundle, lens, layers)
    payload["gates"]["g3"] = run_g3(bundle, lens, layers)

    raw_path = ROOT / "data" / "raw" / "00_gates_qwen7b.json"
    metrics_path = ROOT / "results" / "metrics.json"
    save_json(raw_path, payload)
    save_json(metrics_path, payload)
    print(f"Saved raw gates: {raw_path}")
    print(f"Saved curated metrics: {metrics_path}")
    return payload

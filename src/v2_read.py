"""Stage-1d attribution and layer-aligned weight READ validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from src.data_gen import continuation_token_id, load_probe_swap_items
from src.interventions import (
    ablation_edits,
    clamped_swap_edits,
    forward_logits,
    residual_edit_hooks,
)
from src.jlens_iface import jlens_direction_bank, token_rank
from src.localization_phase import (
    flag_top_components,
    localize_source_direction,
    qwen_attention_weight_read_with_null,
)
from src.metrics import (
    logit_difference,
    partial_correlation_with_ci,
    pearson_with_ci,
    save_json,
)
from src.model_utils import ModelBundle, capture_residuals, decode_topk, set_seed
from src.plotting import save_figure, set_style
from src.read_scores import attribution_read, qwen_mlp_gain
from src.v2_repair import (
    exact_label_token_id,
    load_calibration_items,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
N_VALIDATION = 20
LOCAL_ALPHA_GRID = (-0.25, -0.125, 0.0, 0.125, 0.25)
SWAP_ALPHA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def _metric(logits: torch.Tensor, item: Mapping[str, Any]) -> float:
    return float(
        logit_difference(
            logits,
            int(item["clean_answer_token_id"]),
            int(item["counterfactual_answer_token_id"]),
        )[0].cpu()
    )


def _tokenize_item(tokenizer: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    source_id, source_surface = exact_label_token_id(
        tokenizer, str(raw["intermediate"])
    )
    target_id, target_surface = exact_label_token_id(tokenizer, str(raw["swap_to"]))
    clean_id, clean_surface = continuation_token_id(
        tokenizer, str(raw["prompt"]), str(raw["answer"])
    )
    counterfactual_id, counterfactual_surface = continuation_token_id(
        tokenizer, str(raw["prompt"]), str(raw["swap_answer"])
    )
    return {
        **raw,
        "source_concept_token_id": source_id,
        "source_concept_surface": source_surface,
        "target_concept_token_id": target_id,
        "target_concept_surface": target_surface,
        "clean_answer_token_id": clean_id,
        "clean_answer_surface": clean_surface,
        "counterfactual_answer_token_id": counterfactual_id,
        "counterfactual_answer_surface": counterfactual_surface,
    }


def _select_validation_items(
    bundle: ModelBundle,
    lens: Any,
    layers: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select source-order clean-correct/readout-visible cases only."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in load_probe_swap_items():
        try:
            item = _tokenize_item(bundle.tokenizer, raw)
        except ValueError as error:
            rejected.append({"name": raw["name"], "reason": f"tokenization:{error}"})
            continue
        if item["source_concept_token_id"] in {
            item["clean_answer_token_id"],
            item["counterfactual_answer_token_id"],
        }:
            rejected.append({"name": raw["name"], "reason": "source_output_overlap"})
            continue
        input_ids = bundle.lens_model.encode(item["prompt"])
        clean = forward_logits(bundle.hf_model, input_ids)
        if int(clean[0, -1].argmax()) != item["clean_answer_token_id"]:
            rejected.append({"name": raw["name"], "reason": "clean_answer_not_top1"})
            continue
        readout, _, _ = lens.apply(
            bundle.lens_model,
            item["prompt"],
            layers=layers,
            positions=None,
        )
        best: tuple[int, int, int] | None = None
        for layer in layers:
            for position in range(readout[layer].shape[0]):
                rank = token_rank(
                    readout[layer][position], item["source_concept_token_id"]
                )
                candidate = (rank, layer, position)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        if best[0] > 10:
            rejected.append(
                {
                    "name": raw["name"],
                    "reason": "source_readout_rank_gt_10",
                    "minimum_rank": best[0],
                }
            )
            continue
        accepted.append(
            {
                **item,
                "minimum_source_readout_rank": best[0],
                "best_source_layer": best[1],
                "best_source_position": best[2],
                "clean_metric": _metric(clean, item),
                "clean_top_tokens": decode_topk(bundle.tokenizer, clean[0, -1], 5),
            }
        )
        if len(accepted) == N_VALIDATION:
            break
    if len(accepted) != N_VALIDATION:
        raise RuntimeError(f"Found only {len(accepted)} validation items")
    return accepted, rejected


def _differentiable_alpha_derivative(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    directions: Mapping[int, torch.Tensor],
    item: Mapping[str, Any],
) -> dict[str, float]:
    alpha = torch.zeros((), device=input_ids.device, requires_grad=True)
    edits = {}
    for layer, direction in directions.items():
        vector = direction.detach().to(input_ids.device, torch.float32)

        def edit(hidden, *, vector=vector):
            selected = hidden.float()
            projection = selected @ vector
            return (
                selected - alpha * projection.unsqueeze(-1) * vector
            ).to(hidden.dtype)

        edits[int(layer)] = edit
    with torch.enable_grad(), residual_edit_hooks(bundle.lens_model.layers, edits):
        logits = bundle.hf_model(input_ids=input_ids, use_cache=False).logits.float()
        metric = (
            logits[0, -1, int(item["clean_answer_token_id"])]
            - logits[0, -1, int(item["counterfactual_answer_token_id"])]
        )
        derivative = torch.autograd.grad(metric, alpha)[0]
    return {
        "metric_at_zero": float(metric.detach().cpu()),
        "dmetric_dalpha": float(derivative.detach().cpu()),
    }


def _attribution_record(
    bundle: ModelBundle,
    item: Mapping[str, Any],
    directions: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    input_ids = bundle.lens_model.encode(str(item["prompt"]))
    clean = forward_logits(bundle.hf_model, input_ids)
    attribution = attribution_read(
        bundle.hf_model,
        bundle.lens_model.layers,
        input_ids,
        directions,
        target_token_id=int(item["clean_answer_token_id"]),
        foil_token_id=int(item["counterfactual_answer_token_id"]),
    )
    dose: list[dict[str, float]] = []
    for alpha in LOCAL_ALPHA_GRID:
        if alpha == 0.0:
            logits = clean
        else:
            logits = forward_logits(
                bundle.hf_model,
                input_ids,
                blocks=bundle.lens_model.layers,
                edits=ablation_edits(directions, strength=alpha),
            )
        dose.append({"alpha": alpha, "metric": _metric(logits, item)})
    slope, intercept = np.polyfit(
        [row["alpha"] for row in dose],
        [row["metric"] for row in dose],
        deg=1,
    )
    full = forward_logits(
        bundle.hf_model,
        input_ids,
        blocks=bundle.lens_model.layers,
        edits=ablation_edits(directions, strength=1.0),
    )
    write_values = np.concatenate(list(attribution.write.values()))
    read_values = np.concatenate(list(attribution.read.values()))
    return {
        "name": item["name"],
        "prompt": item["prompt"],
        "intermediate": item["intermediate"],
        "source_concept_token_id": item["source_concept_token_id"],
        "source_concept_surface": item["source_concept_surface"],
        "clean_metric": _metric(clean, item),
        "minimum_source_readout_rank": item["minimum_source_readout_rank"],
        "attribution_predicted_derivative": attribution.predicted_delta,
        "predicted_by_layer": {
            str(layer): value
            for layer, value in attribution.predicted_delta_by_layer.items()
        },
        "write_abs_mean": float(np.mean(np.abs(write_values))),
        "read_abs_mean": float(np.mean(np.abs(read_values))),
        "write_by_layer_position": {
            str(layer): values.tolist() for layer, values in attribution.write.items()
        },
        "read_by_layer_position": {
            str(layer): values.tolist() for layer, values in attribution.read.items()
        },
        "local_dose_curve": dose,
        "local_ols_slope": float(slope),
        "local_ols_intercept": float(intercept),
        "full_alpha1_metric": _metric(full, item),
        "full_alpha1_delta": _metric(full, item) - _metric(clean, item),
        "full_alpha1_positive_damage": _metric(clean, item) - _metric(full, item),
    }


def _correlation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predicted = [float(row["attribution_predicted_derivative"]) for row in rows]
    local = [float(row["local_ols_slope"]) for row in rows]
    full = [float(row["full_alpha1_delta"]) for row in rows]
    damage = [float(row["full_alpha1_positive_damage"]) for row in rows]
    read = [float(row["read_abs_mean"]) for row in rows]
    write = [float(row["write_abs_mean"]) for row in rows]
    return {
        "predicted_vs_local_slope": pearson_with_ci(predicted, local),
        "predicted_vs_full_alpha1_delta": pearson_with_ci(predicted, full),
        "read_strength_vs_positive_damage": pearson_with_ci(read, damage),
        "read_strength_vs_positive_damage_given_write": partial_correlation_with_ci(
            damage, read, write
        ),
        "write_strength_vs_positive_damage": pearson_with_ci(write, damage),
        "local_sign_agreement": sum(
            int(np.sign(first) == np.sign(second))
            for first, second in zip(predicted, local, strict=True)
        )
        / len(rows),
        "full_sign_agreement": sum(
            int(np.sign(first) == np.sign(second))
            for first, second in zip(predicted, full, strict=True)
        )
        / len(rows),
    }


def _swap_dose_curve(
    bundle: ModelBundle,
    item: Mapping[str, Any],
    source: Mapping[int, torch.Tensor],
    target: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    input_ids = bundle.lens_model.encode(str(item["prompt"]))
    clean = forward_logits(bundle.hf_model, input_ids)
    residuals = capture_residuals(bundle.lens_model, input_ids, source)
    rows: list[dict[str, Any]] = []
    for alpha in SWAP_ALPHA_GRID:
        if alpha == 0.0:
            logits = clean
        else:
            logits = forward_logits(
                bundle.hf_model,
                input_ids,
                blocks=bundle.lens_model.layers,
                edits=clamped_swap_edits(
                    residuals, source, target, strength=alpha
                ),
            )
        rows.append(
            {
                "alpha": alpha,
                "metric": _metric(logits, item),
                "top_token": bundle.tokenizer.decode(
                    [int(logits[0, -1].argmax())]
                ),
                "counterfactual_rank": token_rank(
                    logits[0, -1], int(item["counterfactual_answer_token_id"])
                ),
            }
        )
    return {"name": item["name"], "rows": rows}


def _layer_aligned_weight_read(
    bundle: ModelBundle,
    directions: Mapping[int, torch.Tensor],
    flags: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    mlps: list[dict[str, Any]] = []
    for offset, flag in enumerate(flags["mlps"]):
        layer = int(flag["layer"])
        input_direction = directions[layer - 1]
        label_direction = directions[layer]
        weight = qwen_mlp_gain(
            bundle.lens_model.layers[layer],
            input_direction,
            n_random=32,
            seed=seed + 101 * offset,
        )
        block = bundle.lens_model.layers[layer]
        with torch.no_grad():
            vector = input_direction.to(
                next(block.parameters()).device, next(block.parameters()).dtype
            )
            output = block.mlp(block.post_attention_layernorm(vector)).float()
            label = label_direction.to(output.device, torch.float32)
            cosine = float(
                torch.nn.functional.cosine_similarity(output, label, dim=0).cpu()
            )
        null = np.asarray(weight["random_gains"], dtype=float)
        mlps.append(
            {
                **flag,
                **weight,
                "input_direction_layer": layer - 1,
                "label_direction_layer": layer,
                "label_cosine": cosine,
                "oriented_normalized_gain": float(weight["normalized_gain"]) * cosine,
                "gain_random_percentile": float(
                    np.mean(null <= float(weight["gain"]))
                ),
            }
        )
    heads: list[dict[str, Any]] = []
    for offset, flag in enumerate(flags["attention_heads"]):
        layer = int(flag["layer"])
        all_rows = qwen_attention_weight_read_with_null(
            bundle.lens_model.layers[layer].self_attn,
            directions[layer - 1],
            label_direction=directions[layer],
            n_random=32,
            seed=seed + 1009 + 101 * offset,
        )
        row = next(value for value in all_rows if int(value["head"]) == int(flag["head"]))
        heads.append(
            {
                **flag,
                **row,
                "input_direction_layer": layer - 1,
                "label_direction_layer": layer,
                "oriented_normalized_ov": float(row["normalized_ov_norm"])
                * float(row["label_cosine"]),
            }
        )
    mlp_primary = float(np.mean([row["normalized_gain"] for row in mlps]))
    head_primary = float(
        np.mean([row["label_weighted_normalized_ov"] for row in heads])
    )
    return {
        "mlps": mlps,
        "attention_heads": heads,
        "mlp_primary": mlp_primary,
        "attention_primary": head_primary,
        "equal_family_composite": 0.5 * mlp_primary + 0.5 * head_primary,
        "mlp_mean_random_percentile": float(
            np.mean([row["gain_random_percentile"] for row in mlps])
        ),
        "attention_mean_random_percentile": float(
            np.mean([row["ov_norm_random_percentile"] for row in heads])
        ),
        "mlp_mean_oriented": float(
            np.mean([row["oriented_normalized_gain"] for row in mlps])
        ),
        "attention_mean_oriented": float(
            np.mean([row["oriented_normalized_ov"] for row in heads])
        ),
        "metadata": {
            "activation_independent_primary_magnitude": True,
            "selection_conditioned": True,
            "input_direction": "v[layer-1]",
            "label_direction": "v[layer]",
            "n_random": 32,
        },
    }


def _clear_case_weight_validation(
    bundle: ModelBundle,
    items: list[dict[str, Any]],
    direction_bank: Mapping[int, Mapping[int, torch.Tensor]],
    attribution_lookup: Mapping[str, Mapping[str, Any]],
    layers: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        directions = {
            layer: direction_bank[item["source_concept_token_id"]][layer]
            for layer in layers
        }
        attribution = attribution_lookup[item["name"]]
        eligible_sources = layers[:-2]
        source_layer = max(
            eligible_sources,
            key=lambda layer: abs(float(attribution["predicted_by_layer"][str(layer)])),
        )
        input_ids = bundle.lens_model.encode(item["prompt"])
        localization = localize_source_direction(
            bundle.hf_model,
            bundle.lens_model.layers,
            input_ids,
            directions[source_layer],
            source_layer=source_layer,
            target_token_id=item["clean_answer_token_id"],
            foil_token_id=item["counterfactual_answer_token_id"],
            component_layers=list(range(source_layer + 1, max(layers) + 1)),
        )
        flags = flag_top_components(localization, top_k_mlps=2, top_k_heads=4)
        weight = _layer_aligned_weight_read(
            bundle,
            directions,
            flags,
            seed=SEED + 10_000 * index,
        )
        rows.append(
            {
                "name": item["name"],
                "source_layer": source_layer,
                "localization": localization,
                "flags": flags,
                "weight_read": weight,
            }
        )
    mlp_attr = [
        float(component["abs_score"])
        for row in rows
        for component in row["weight_read"]["mlps"]
    ]
    mlp_weight = [
        float(component["normalized_gain"])
        for row in rows
        for component in row["weight_read"]["mlps"]
    ]
    head_attr = [
        float(component["abs_score"])
        for row in rows
        for component in row["weight_read"]["attention_heads"]
    ]
    head_weight = [
        float(component["label_weighted_normalized_ov"])
        for row in rows
        for component in row["weight_read"]["attention_heads"]
    ]
    mlp_rho = spearmanr(mlp_attr, mlp_weight)
    head_rho = spearmanr(head_attr, head_weight)
    finite = all(
        math.isfinite(float(row["weight_read"][key]))
        for row in rows
        for key in (
            "mlp_primary",
            "attention_primary",
            "equal_family_composite",
        )
    )
    above_random_cases = sum(
        int(
            row["weight_read"]["mlp_mean_random_percentile"] >= 0.5
            and row["weight_read"]["attention_mean_random_percentile"] >= 0.5
        )
        for row in rows
    )
    orientation_cases = sum(
        int(
            row["weight_read"]["mlp_mean_oriented"] > 0
            and row["weight_read"]["attention_mean_oriented"] > 0
        )
        for row in rows
    )
    criteria = {
        "all_primary_values_finite": finite,
        "both_families_above_random_in_at_least_2_of_3_cases": above_random_cases >= 2,
        "attribution_weight_rank_rho_nonnegative_for_both_families": (
            float(mlp_rho.statistic) >= 0.0 and float(head_rho.statistic) >= 0.0
        ),
    }
    return {
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
        "n_above_random_cases": above_random_cases,
        "n_positive_orientation_cases": orientation_cases,
        "orientation_is_gate": False,
        "mlp_attribution_weight_spearman": {
            "rho": float(mlp_rho.statistic),
            "p_value": float(mlp_rho.pvalue),
            "n": len(mlp_attr),
            "comparison": "unsigned attribution magnitude vs unsigned weight gain",
        },
        "attention_attribution_weight_spearman": {
            "rho": float(head_rho.statistic),
            "p_value": float(head_rho.pvalue),
            "n": len(head_attr),
            "comparison": (
                "unsigned attribution magnitude vs label-weighted unsigned OV"
            ),
        },
        "rows": rows,
        "sign_note": (
            "Weight magnitude is unsigned. Positive layer-to-layer label orientation "
            "is a separate diagnostic, not the sign of behavioral attribution; "
            "it is therefore not a pass criterion."
        ),
    }


def _plot_validation(rows: list[dict[str, Any]], correlations: Mapping[str, Any]) -> str:
    predicted = np.asarray(
        [row["attribution_predicted_derivative"] for row in rows], dtype=float
    )
    local = np.asarray([row["local_ols_slope"] for row in rows], dtype=float)
    full = np.asarray([row["full_alpha1_delta"] for row in rows], dtype=float)
    set_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for axis, outcome, key, title in (
        (
            axes[0],
            local,
            "predicted_vs_local_slope",
            "Local dose slope (plumbing test)",
        ),
        (
            axes[1],
            full,
            "predicted_vs_full_alpha1_delta",
            "Full alpha=1 endpoint",
        ),
    ):
        axis.scatter(predicted, outcome, s=35, alpha=0.85)
        if np.ptp(predicted) > 0:
            coefficient = np.polyfit(predicted, outcome, 1)
            x = np.linspace(predicted.min(), predicted.max(), 100)
            axis.plot(x, np.polyval(coefficient, x), color="#B33A3A")
        statistic = correlations[key]
        axis.set(
            xlabel="Attribution first-order derivative",
            ylabel="Measured derivative / delta",
            title=f"{title}\nr={statistic['estimate']:.3f}, N={statistic['n']}",
        )
    path = ROOT / "results" / "figures" / "f5_repaired_read_validation.png"
    save_figure(figure, path)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def run_stage1d(
    bundle: ModelBundle,
    lens: Any,
    *,
    workspace_layers: list[int],
) -> dict[str, Any]:
    """Validate local attribution, nonlinear endpoints, and weight READ."""

    set_seed(SEED)
    items, rejected = _select_validation_items(bundle, lens, workspace_layers)
    clear_items = load_calibration_items(bundle.tokenizer)
    token_ids = {
        int(item["source_concept_token_id"]) for item in [*items, *clear_items]
    } | {int(item["target_concept_token_id"]) for item in clear_items}
    bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        workspace_layers,
        fold_rms_gain=False,
    )
    attribution_rows: list[dict[str, Any]] = []
    for item in items:
        directions = {
            layer: bank[item["source_concept_token_id"]][layer]
            for layer in workspace_layers
        }
        attribution_rows.append(_attribution_record(bundle, item, directions))
    attribution_lookup = {row["name"]: row for row in attribution_rows}
    # Ensure all three clear cases have a matching attribution record, even if
    # source-order validation selection did not include them.
    for item in clear_items:
        if item["name"] in attribution_lookup:
            continue
        readout = lens.apply(
            bundle.lens_model,
            item["prompt"],
            layers=workspace_layers,
            positions=None,
        )[0]
        item["minimum_source_readout_rank"] = min(
            token_rank(readout[layer][position], item["source_concept_token_id"])
            for layer in workspace_layers
            for position in range(readout[layer].shape[0])
        )
        directions = {
            layer: bank[item["source_concept_token_id"]][layer]
            for layer in workspace_layers
        }
        record = _attribution_record(bundle, item, directions)
        attribution_lookup[item["name"]] = record

    correlations = _correlation_summary(attribution_rows)
    exact_derivatives: list[dict[str, Any]] = []
    swap_doses: list[dict[str, Any]] = []
    for item in clear_items:
        source = {
            layer: bank[item["source_concept_token_id"]][layer]
            for layer in workspace_layers
        }
        target = {
            layer: bank[item["target_concept_token_id"]][layer]
            for layer in workspace_layers
        }
        exact = _differentiable_alpha_derivative(
            bundle, bundle.lens_model.encode(item["prompt"]), source, item
        )
        predicted = float(
            attribution_lookup[item["name"]]["attribution_predicted_derivative"]
        )
        exact_derivatives.append(
            {
                "name": item["name"],
                **exact,
                "attribution_predicted_derivative": predicted,
                "absolute_error": abs(exact["dmetric_dalpha"] - predicted),
            }
        )
        swap_doses.append(_swap_dose_curve(bundle, item, source, target))
    plumbing_pass = all(row["absolute_error"] <= 0.05 for row in exact_derivatives)
    local_stat = correlations["predicted_vs_local_slope"]
    local_reliable = bool(
        local_stat["estimate"] >= 0.8 and local_stat["ci_low"] > 0.0
    )
    full_stat = correlations["predicted_vs_full_alpha1_delta"]
    full_reliable = bool(full_stat["estimate"] >= 0.5 and full_stat["ci_low"] > 0.0)

    weight = _clear_case_weight_validation(
        bundle,
        clear_items,
        bank,
        attribution_lookup,
        workspace_layers,
    )
    if weight["status"] == "PASS" and plumbing_pass:
        primary = "WEIGHT_BASED"
        read_status = "PASS"
    else:
        primary = "UNRESOLVED"
        read_status = "FAIL"
    attribution_role = "SECONDARY"
    if full_reliable:
        attribution_role = "SECONDARY_ENDPOINT_VALIDATED"
    summary: dict[str, Any] = {
        "status": read_status,
        "model_id": bundle.model_id,
        "model_revision": bundle.revision,
        "workspace_layers": workspace_layers,
        "direction": "exact-label raw normalize(J.T @ W_U[token])",
        "validation_selection": {
            "rule": (
                "first 20 upstream items that are tokenizable, clean-answer top1, "
                "and source concept J-Lens rank<=10 in the repaired band"
            ),
            "n": len(items),
            "items": [item["name"] for item in items],
            "rejected_before_stop": rejected,
        },
        "attribution": {
            "plumbing_exact_derivatives": exact_derivatives,
            "plumbing_pass_abs_error_le_0.05": plumbing_pass,
            "local_reliable": local_reliable,
            "full_alpha1_endpoint_reliable": full_reliable,
            "reliability_rule": "r>=0.5 and bootstrap CI low>0 for full alpha1 endpoint",
            "role": attribution_role,
            "correlations": correlations,
            "rows": attribution_rows,
            "interpretation": (
                "The signed first-order product is distinct from READ magnitude; "
                "full nonlinear edits are evaluated separately from local plumbing."
            ),
        },
        "swap_dose_curves": swap_doses,
        "weight_read": weight,
        "primary_read": primary,
        "decision": (
            "Use layer-aligned random-normalized weight magnitude as primary; "
            "retain attribution as secondary."
            if read_status == "PASS"
            else "READ calibration unresolved; later calibration/science prohibited."
        ),
        "limitations": [
            "Weight magnitude is unsigned and selection-conditioned on attribution-flagged components.",
            "Layer-to-layer label orientation is not the sign of behavior-specific attribution.",
            "Alpha=2 coordinate swaps are strongly nonlinear and are not first-order attribution targets.",
        ],
    }
    summary["figure"] = _plot_validation(attribution_rows, correlations)
    raw_path = ROOT / "data" / "raw" / "03_read_validation_v2.json"
    save_json(raw_path, summary)
    summary["raw_artifact"] = str(raw_path.relative_to(ROOT))
    return summary


def _report_section(stage1d: Mapping[str, Any]) -> str:
    attr = stage1d["attribution"]
    corr = attr["correlations"]
    local = corr["predicted_vs_local_slope"]
    full = corr["predicted_vs_full_alpha1_delta"]
    read = corr["read_strength_vs_positive_damage"]
    partial = corr["read_strength_vs_positive_damage_given_write"]
    weight = stage1d["weight_read"]
    return f"""

## Stage 1d — READ repair and validation

The old `r=-0.36` label conflated two quantities: it compared the signed
first-order product `-sum(WRITE*READ)` with a large full ablation, not READ
magnitude with causal damage. V2 separates local gradient plumbing, the
nonlinear endpoint, and READ-strength association.

- Exact shared-alpha derivative check on the three G-SWAP cases: **{'PASS' if attr['plumbing_pass_abs_error_le_0.05'] else 'FAIL'}**.
- Attribution prediction vs local dose slope: r={local['estimate']:.3f}, 95% CI [{local['ci_low']:.3f}, {local['ci_high']:.3f}], N={local['n']}.
- Attribution prediction vs full alpha=1 ablation: r={full['estimate']:.3f}, 95% CI [{full['ci_low']:.3f}, {full['ci_high']:.3f}], N={full['n']}.
- Attribution READ magnitude vs positive damage: r={read['estimate']:.3f}, 95% CI [{read['ci_low']:.3f}, {read['ci_high']:.3f}].
- Partial READ-magnitude association given WRITE: r={partial['estimate']:.3f}, 95% CI [{partial['ci_low']:.3f}, {partial['ci_high']:.3f}].

Weight READ now feeds block `k` with `v[k-1]` and evaluates output/label
orientation against `v[k]`; the legacy code incorrectly reused one source-layer
vector at every downstream block. Components are flagged by attribution on the
three clear calibration cases, then evaluated against 32 seeded random
directions.

- Weight calibration: **{weight['status']}**; above-random in {weight['n_above_random_cases']}/3 cases. Positive MLP+attention orientation occurs in {weight['n_positive_orientation_cases']}/3 and is diagnostic only because weight magnitude is unsigned.
- MLP attribution/weight rank rho={weight['mlp_attribution_weight_spearman']['rho']:.3f} (N={weight['mlp_attribution_weight_spearman']['n']}).
- Attention attribution/weight rank rho={weight['attention_attribution_weight_spearman']['rho']:.3f} (N={weight['attention_attribution_weight_spearman']['n']}).

### READ decision

**{stage1d['status']} — primary READ: {stage1d['primary_read']}; attribution role:
{attr['role']}.** Weight magnitude is unsigned; signed label orientation is a
separate diagnostic and is not mislabeled as behavioral attribution.

![Repaired attribution validation](figures/f5_repaired_read_validation.png)

Stage-3 science remains prohibited until firing controls and G-POS pass.
"""


def persist_stage1d(stage1d: Mapping[str, Any]) -> dict[str, Any]:
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    repair = metrics["repair_v2"]
    if repair["gate_ledger"].get("g_dir") not in {"PASS", "DROPPED_MD"}:
        raise RuntimeError("READ validation cannot precede G-DIR")
    repair["stage1d_read_validation"] = dict(stage1d)
    repair["gate_ledger"]["read_validation"] = stage1d["status"]
    repair["gate_ledger"]["stage3_science"] = "PROHIBITED"
    repair["current_allowed_conclusion"] = (
        "READ_CALIBRATED_CONTROLS_PENDING"
        if stage1d["status"] == "PASS"
        else "READ_CALIBRATION_FAILED_STAGE4_REQUIRED"
    )
    save_json(metrics_path, metrics)
    report_path = ROOT / "results" / "RESULTS.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## Stage 1d — READ repair and validation"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    report_path.write_text(report.rstrip() + _report_section(stage1d), encoding="utf-8")
    return metrics

"""Stage-1 repair and hard known-answer coordinate-swap gate.

The repair is selected without looking at swap outcomes:

* resolve upstream JSON labels exactly when that exact string is one token;
* map the paper's normalized workspace prior (38--92% depth) to Qwen;
* within that prior, retain the longest contiguous run where the median clean
  source-concept J-Lens rank across three calibration prompts is top-10;
* use the paper-literal raw ``normalize(J.T @ W_U[token])`` direction;
* use the paper's documented double-strength swap, alpha=2;
* edit every real prompt position.

Only after freezing that configuration do we test the three declared
counterfactual answers and decide G-SWAP.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_gen import (
    G1_PROMPTS,
    continuation_token_id,
    load_probe_swap_items,
)
from src.interventions import (
    clamp_swapped_coordinates,
    clamped_swap_edits,
    forward_logits,
)
from src.jlens_iface import jlens_direction_bank, token_rank
from src.metrics import logit_difference, save_json
from src.model_utils import (
    ModelBundle,
    capture_residuals,
    decode_topk,
    hf_wrapper_logit_kl,
    set_seed,
)
from src.plotting import save_figure, set_style


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1729
CALIBRATION_ITEM_NAMES = (
    "spider-legs",
    "animal-legs-buffalo2",
    "chem-photosynthesis-Z",
)
PAPER_WORKSPACE_START_PERCENT = 38.0
PAPER_WORKSPACE_END_PERCENT = 92.0
VISIBILITY_RANK = 10
CANONICAL_STRENGTH = 2.0
ANT_CUE_PROMPTS = (
    "Fact: The number of legs on the insect that lives in organized colonies is ",
    "Fact: The insect that follows pheromone trails back to a colony is an ",
    "Fact: The tiny worker serving a queen in a colony is an ",
)


def label_token_candidates(tokenizer: Any, label: str) -> list[dict[str, Any]]:
    """Return distinct exact/leading-space one-token encodings in fixed order."""

    surfaces = [label] if label.startswith(" ") else [label, f" {label}"]
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for order, surface in enumerate(surfaces):
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(token_ids) != 1 or int(token_ids[0]) in seen:
            continue
        token_id = int(token_ids[0])
        seen.add(token_id)
        rows.append(
            {
                "selection_order": order,
                "surface": surface,
                "token_id": token_id,
                "decoded": tokenizer.decode([token_id]),
                "is_exact_label": surface == label,
            }
        )
    if not rows:
        raise ValueError(f"No single-token encoding for upstream label {label!r}")
    return rows


def exact_label_token_id(tokenizer: Any, label: str) -> tuple[int, str]:
    """Use the exact upstream label when possible, else a leading-space token."""

    row = label_token_candidates(tokenizer, label)[0]
    return int(row["token_id"]), str(row["surface"])


def leading_label_token_id(tokenizer: Any, label: str) -> tuple[int, str]:
    """Legacy leading-space preference retained only as a sensitivity arm."""

    candidates = label_token_candidates(tokenizer, label)
    leading = next(
        (row for row in candidates if str(row["surface"]).startswith(" ")), None
    )
    row = leading or candidates[0]
    return int(row["token_id"]), str(row["surface"])


def load_calibration_items(tokenizer: Any) -> list[dict[str, Any]]:
    """Load the three predeclared upstream cases with exact behavior tokens."""

    by_name = {item["name"]: item for item in load_probe_swap_items()}
    rows: list[dict[str, Any]] = []
    for name in CALIBRATION_ITEM_NAMES:
        raw = by_name[name]
        source_id, source_surface = exact_label_token_id(
            tokenizer, raw["intermediate"]
        )
        target_id, target_surface = exact_label_token_id(tokenizer, raw["swap_to"])
        clean_id, clean_surface = continuation_token_id(
            tokenizer, raw["prompt"], raw["answer"]
        )
        counterfactual_id, counterfactual_surface = continuation_token_id(
            tokenizer, raw["prompt"], raw["swap_answer"]
        )
        rows.append(
            {
                **raw,
                "source_concept_token_id": source_id,
                "source_concept_surface": source_surface,
                "target_concept_token_id": target_id,
                "target_concept_surface": target_surface,
                "clean_answer_token_id": clean_id,
                "clean_answer_surface": clean_surface,
                "counterfactual_answer_token_id": counterfactual_id,
                "counterfactual_answer_surface": counterfactual_surface,
                "source_token_candidates": label_token_candidates(
                    tokenizer, raw["intermediate"]
                ),
                "target_token_candidates": label_token_candidates(
                    tokenizer, raw["swap_to"]
                ),
            }
        )
    return rows


def paper_workspace_prior(n_layers: int, source_layers: Sequence[int]) -> list[int]:
    """Map the paper's ~L38--92 normalized depth range to this model."""

    denominator = max(1, int(n_layers) - 1)
    return [
        int(layer)
        for layer in sorted(set(int(value) for value in source_layers))
        if PAPER_WORKSPACE_START_PERCENT
        <= 100.0 * int(layer) / denominator
        <= PAPER_WORKSPACE_END_PERCENT
    ]


def longest_contiguous_visible_band(
    prior_layers: Sequence[int],
    min_ranks_by_item: Mapping[str, Mapping[int, int]],
    *,
    rank_threshold: int = VISIBILITY_RANK,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Select a band from clean readout visibility, never intervention outcomes."""

    diagnostics: dict[int, dict[str, Any]] = {}
    active: list[int] = []
    for layer in prior_layers:
        ranks = [int(values[int(layer)]) for values in min_ranks_by_item.values()]
        median = float(np.median(ranks))
        is_active = median <= rank_threshold
        diagnostics[int(layer)] = {
            "ranks": ranks,
            "median_rank": median,
            "active": is_active,
        }
        if is_active:
            active.append(int(layer))
    runs: list[list[int]] = []
    for layer in active:
        if not runs or layer != runs[-1][-1] + 1:
            runs.append([layer])
        else:
            runs[-1].append(layer)
    if not runs:
        raise ValueError("No clean-readout-visible workspace run was found")
    runs.sort(key=lambda run: (-len(run), run[0]))
    return runs[0], diagnostics


def _readout_ranks(
    bundle: ModelBundle,
    lens: Any,
    item: Mapping[str, Any],
    layers: Sequence[int],
) -> dict[str, Any]:
    input_ids = bundle.lens_model.encode(str(item["prompt"]))
    jlens_logits, _, _ = lens.apply(
        bundle.lens_model,
        str(item["prompt"]),
        layers=list(layers),
        positions=None,
    )
    candidates = {
        f"source:{row['surface']}": int(row["token_id"])
        for row in item["source_token_candidates"]
    }
    candidates.update(
        {
            f"target:{row['surface']}": int(row["token_id"])
            for row in item["target_token_candidates"]
        }
    )
    candidate_rows: dict[str, Any] = {}
    for label, token_id in candidates.items():
        by_layer: dict[str, list[int]] = {}
        best: tuple[int, int, int] | None = None
        for layer in layers:
            ranks = [
                token_rank(jlens_logits[int(layer)][position], token_id)
                for position in range(jlens_logits[int(layer)].shape[0])
            ]
            by_layer[str(layer)] = ranks
            layer_best = min(ranks)
            position = ranks.index(layer_best)
            candidate = (layer_best, int(layer), position)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        candidate_rows[label] = {
            "token_id": token_id,
            "minimum_rank": best[0],
            "best_layer": best[1],
            "best_position": best[2],
            "ranks_by_layer_position": by_layer,
        }
    source_key = f"source:{item['source_concept_surface']}"
    source_by_layer = {
        int(layer): min(candidate_rows[source_key]["ranks_by_layer_position"][str(layer)])
        for layer in layers
    }
    return {
        "prompt_token_ids": [int(value) for value in input_ids[0].cpu()],
        "prompt_tokens": [
            bundle.tokenizer.decode([int(value)]) for value in input_ids[0].cpu()
        ],
        "candidates": candidate_rows,
        "selected_source_min_rank_by_layer": source_by_layer,
    }


def _cue_surface_validation(
    bundle: ModelBundle,
    lens: Any,
    layers: Sequence[int],
) -> dict[str, Any]:
    candidates = label_token_candidates(bundle.tokenizer, "ant")
    rows: list[dict[str, Any]] = []
    for prompt in ANT_CUE_PROMPTS:
        logits, _, _ = lens.apply(
            bundle.lens_model, prompt, layers=list(layers), positions=None
        )
        rankings: list[dict[str, Any]] = []
        for candidate in candidates:
            best: tuple[int, int, int] | None = None
            for layer in layers:
                for position in range(logits[int(layer)].shape[0]):
                    rank = token_rank(
                        logits[int(layer)][position], int(candidate["token_id"])
                    )
                    row = (rank, int(layer), position)
                    if best is None or row < best:
                        best = row
            assert best is not None
            rankings.append(
                {
                    **candidate,
                    "minimum_rank": best[0],
                    "best_layer": best[1],
                    "best_position": best[2],
                }
            )
        rows.append({"prompt": prompt, "candidate_ranks": rankings})
    return {
        "selection_rule": "exact upstream label first; rank is validation, not selector",
        "prompts": rows,
        "limitation": (
            "The leading-space ant token can rank better on matched cues; the exact "
            "label convention is therefore retained as an explicit design choice."
        ),
    }


def _g1(bundle: ModelBundle) -> dict[str, Any]:
    rows = hf_wrapper_logit_kl(bundle, G1_PROMPTS)
    maximum = max(float(row["mean_kl"]) for row in rows)
    return {
        "status": "PASS" if maximum < 1e-3 else "FAIL",
        "threshold_mean_kl": 1e-3,
        "n": len(rows),
        "max_prompt_mean_kl": maximum,
        "max_position_kl": max(float(row["max_kl"]) for row in rows),
        "max_abs_logit_error": max(
            float(row["max_abs_logit_error"]) for row in rows
        ),
        "items": rows,
    }


def _metric(logits: torch.Tensor, item: Mapping[str, Any]) -> float:
    return float(
        logit_difference(
            logits,
            int(item["clean_answer_token_id"]),
            int(item["counterfactual_answer_token_id"]),
        )[0].cpu()
    )


def _coefficient_diagnostics(
    residuals: Mapping[int, torch.Tensor],
    source: Mapping[int, torch.Tensor],
    target: Mapping[int, torch.Tensor],
    *,
    layers: Sequence[int],
    strength: float,
    positions: Sequence[int] | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        clean = residuals[int(layer)]
        edited = clamp_swapped_coordinates(
            clean,
            clean,
            source[int(layer)],
            target[int(layer)],
            positions=positions,
            strength=strength,
        )
        if positions is None:
            clean_selected = clean.float()
            edited_selected = edited.float()
        else:
            clean_selected = clean[:, list(positions), :].float()
            edited_selected = edited[:, list(positions), :].float()
        basis = torch.stack(
            [source[int(layer)].float(), target[int(layer)].float()], dim=0
        )
        gram = basis @ basis.T
        inverse = torch.linalg.inv(gram)
        clean_coeff = (clean_selected @ basis.T) @ inverse
        edited_coeff = (edited_selected @ basis.T) @ inverse
        desired = clean_coeff + float(strength) * (
            clean_coeff.flip(-1) - clean_coeff
        )
        absolute_error = float((edited_coeff - desired).abs().max().cpu())
        desired_scale = float(desired.abs().max().cpu())
        rows.append(
            {
                "layer": int(layer),
                "direction_cosine": float(gram[0, 1].cpu()),
                "gram_condition": float(torch.linalg.cond(gram).cpu()),
                "max_abs_desired_coefficient": desired_scale,
                "max_abs_coefficient_error_after_bf16_edit": absolute_error,
                "max_relative_coefficient_error_after_bf16_edit": (
                    absolute_error / desired_scale if desired_scale > 0 else 0.0
                ),
            }
        )
    return {
        "layers": rows,
        "max_gram_condition": max(row["gram_condition"] for row in rows),
        "max_abs_coefficient_error_after_bf16_edit": max(
            row["max_abs_coefficient_error_after_bf16_edit"] for row in rows
        ),
        "max_relative_coefficient_error_after_bf16_edit": max(
            row["max_relative_coefficient_error_after_bf16_edit"] for row in rows
        ),
    }


def _attempt(
    *,
    bundle: ModelBundle,
    item: Mapping[str, Any],
    input_ids: torch.Tensor,
    clean_logits: torch.Tensor,
    residuals: Mapping[int, torch.Tensor],
    bank: Mapping[int, Mapping[int, torch.Tensor]],
    layers: Sequence[int],
    source_token_id: int,
    target_token_id: int,
    name: str,
    direction_convention: str,
    strength: float,
    positions: Sequence[int] | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    source = {int(layer): bank[int(source_token_id)][int(layer)] for layer in layers}
    target = {int(layer): bank[int(target_token_id)][int(layer)] for layer in layers}
    edits = clamped_swap_edits(
        {int(layer): residuals[int(layer)] for layer in layers},
        source,
        target,
        positions=positions,
        strength=strength,
    )
    outputs: list[torch.Tensor] = []
    top_ids: list[int] = []
    for _ in range(repeats):
        logits = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=edits,
        )
        outputs.append(logits)
        top_ids.append(int(logits[0, -1].argmax()))
    edited = outputs[0]
    vector = edited[0, -1]
    clean_id = int(item["clean_answer_token_id"])
    counterfactual_id = int(item["counterfactual_answer_token_id"])
    other = vector.clone()
    other[counterfactual_id] = -torch.inf
    top_other = float(other.max().cpu())
    repeat_error = max(
        (output - edited).abs().max().item() for output in outputs[1:]
    ) if len(outputs) > 1 else 0.0
    row = {
        "name": name,
        "item_name": item["name"],
        "source_label": item["intermediate"],
        "target_label": item["swap_to"],
        "source_token_id": int(source_token_id),
        "source_surface": bundle.tokenizer.decode([int(source_token_id)]),
        "target_token_id": int(target_token_id),
        "target_surface": bundle.tokenizer.decode([int(target_token_id)]),
        "direction_convention": direction_convention,
        "layers": [int(layer) for layer in layers],
        "positions": "all_prompt_positions" if positions is None else list(positions),
        "strength": float(strength),
        "clean_metric": _metric(clean_logits, item),
        "edited_metric": _metric(edited, item),
        "delta_metric": _metric(edited, item) - _metric(clean_logits, item),
        "clean_top_token_id": int(clean_logits[0, -1].argmax()),
        "clean_top_token": bundle.tokenizer.decode(
            [int(clean_logits[0, -1].argmax())]
        ),
        "edited_top_token_id": int(vector.argmax()),
        "edited_top_token": bundle.tokenizer.decode([int(vector.argmax())]),
        "clean_answer_rank_after_edit": token_rank(vector, clean_id),
        "counterfactual_answer_rank_after_edit": token_rank(vector, counterfactual_id),
        "counterfactual_logit": float(vector[counterfactual_id].cpu()),
        "clean_answer_logit": float(vector[clean_id].cpu()),
        "counterfactual_vs_clean_margin": float(
            (vector[counterfactual_id] - vector[clean_id]).cpu()
        ),
        "counterfactual_argmax_margin": float(
            vector[counterfactual_id].cpu()
        ) - top_other,
        "top_tokens": decode_topk(bundle.tokenizer, vector, 10),
        "repeat_count": repeats,
        "repeat_top_token_ids": top_ids,
        "repeat_max_abs_logit_difference": float(repeat_error),
    }
    row["strict_pass"] = bool(
        row["clean_top_token_id"] == clean_id
        and row["edited_top_token_id"] == counterfactual_id
        and row["counterfactual_argmax_margin"] > 0.0
        and len(set(top_ids)) == 1
        and repeat_error == 0.0
    )
    row["coefficient_diagnostics"] = _coefficient_diagnostics(
        residuals,
        source,
        target,
        layers=layers,
        strength=strength,
        positions=positions,
    )
    return row


def _cross_function_checks(
    bundle: ModelBundle,
    raw_bank: Mapping[int, Mapping[int, torch.Tensor]],
    *,
    source_token_id: int,
    target_token_id: int,
    layers: Sequence[int],
) -> list[dict[str, Any]]:
    prompts = [
        {
            "name": "spider_to_ant_first_letter",
            "prompt": "Fact: The first letter of the animal that spins webs is ",
            "source": source_token_id,
            "target": target_token_id,
        },
        {
            "name": "spider_to_ant_biological_class",
            "prompt": "Fact: The biological class of the animal that spins webs is ",
            "source": source_token_id,
            "target": target_token_id,
        },
        {
            "name": "reverse_ant_to_spider_legs",
            "prompt": ANT_CUE_PROMPTS[0],
            "source": target_token_id,
            "target": source_token_id,
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in prompts:
        input_ids = bundle.lens_model.encode(spec["prompt"])
        residuals = capture_residuals(bundle.lens_model, input_ids, layers)
        clean = forward_logits(bundle.hf_model, input_ids)
        source = {
            int(layer): raw_bank[int(spec["source"])][int(layer)] for layer in layers
        }
        target = {
            int(layer): raw_bank[int(spec["target"])][int(layer)] for layer in layers
        }
        edited = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=clamped_swap_edits(
                residuals, source, target, strength=CANONICAL_STRENGTH
            ),
        )
        rows.append(
            {
                "name": spec["name"],
                "prompt": spec["prompt"],
                "clean_top_tokens": decode_topk(bundle.tokenizer, clean[0, -1], 5),
                "edited_top_tokens": decode_topk(bundle.tokenizer, edited[0, -1], 5),
                "confirmatory": False,
            }
        )
    return rows


def _plot_gate(stage1: Mapping[str, Any]) -> str:
    canonical = stage1["g_swap"]["canonical_rows"]
    alpha1_by_item = {
        row["item_name"]: row
        for row in stage1["attempts"]
        if row["name"] == "exact_raw_alpha1_empirical_band"
    }
    labels = [str(row["item_name"]) for row in canonical]
    clean = [float(row["clean_metric"]) for row in canonical]
    alpha1 = [float(alpha1_by_item[label]["edited_metric"]) for label in labels]
    alpha2 = [float(row["edited_metric"]) for row in canonical]
    x = np.arange(len(labels))
    width = 0.25
    set_style()
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    axis.bar(x - width, clean, width, label="clean", color="#4C78A8")
    axis.bar(x, alpha1, width, label="swap α=1", color="#F2A541")
    axis.bar(x + width, alpha2, width, label="swap α=2", color="#2E8B57")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x, labels=labels, rotation=12, ha="right")
    axis.set_ylabel("M = logit(clean answer) − logit(counterfactual answer)")
    axis.set_title("G-SWAP calibration: one fixed empirical Qwen workspace band")
    axis.legend(frameon=False)
    path = ROOT / "results" / "figures" / "repair_gswap_calibration.png"
    save_figure(figure, path)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def run_stage1(bundle: ModelBundle, lens: Any) -> dict[str, Any]:
    """Execute the complete repair sweep and return JSON-safe evidence."""

    if bundle.model_id != MODEL_ID:
        raise ValueError(f"Stage-1 debug model must be {MODEL_ID}, got {bundle.model_id}")
    set_seed(SEED)
    g1 = _g1(bundle)
    if g1["status"] != "PASS":
        raise RuntimeError("G1 failed; residual intervention results are not usable")
    items = load_calibration_items(bundle.tokenizer)
    all_source_layers = sorted(int(layer) for layer in lens.source_layers)
    prior = paper_workspace_prior(bundle.lens_model.n_layers, all_source_layers)
    readout = {
        item["name"]: _readout_ranks(bundle, lens, item, all_source_layers)
        for item in items
    }
    min_ranks = {
        item["name"]: readout[item["name"]]["selected_source_min_rank_by_layer"]
        for item in items
    }
    empirical_band, band_diagnostics = longest_contiguous_visible_band(
        prior, min_ranks
    )
    union_layers = sorted(set(prior) | set(empirical_band) | {max(empirical_band) + 1})
    union_layers = [layer for layer in union_layers if layer in all_source_layers]

    contexts: dict[str, dict[str, Any]] = {}
    token_ids: set[int] = set()
    for item in items:
        for key in ("source_token_candidates", "target_token_candidates"):
            token_ids.update(int(row["token_id"]) for row in item[key])
        input_ids = bundle.lens_model.encode(item["prompt"])
        clean = forward_logits(bundle.hf_model, input_ids)
        contexts[item["name"]] = {
            "input_ids": input_ids,
            "clean_logits": clean,
            "residuals": capture_residuals(
                bundle.lens_model, input_ids, union_layers
            ),
        }
    raw_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        union_layers,
        fold_rms_gain=False,
    )
    folded_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        union_layers,
        fold_rms_gain=True,
    )

    attempts: list[dict[str, Any]] = []
    for item in items:
        context = contexts[item["name"]]
        for strength in (1.0, CANONICAL_STRENGTH):
            attempts.append(
                _attempt(
                    bundle=bundle,
                    item=item,
                    input_ids=context["input_ids"],
                    clean_logits=context["clean_logits"],
                    residuals=context["residuals"],
                    bank=raw_bank,
                    layers=empirical_band,
                    source_token_id=item["source_concept_token_id"],
                    target_token_id=item["target_concept_token_id"],
                    name=(
                        "exact_raw_alpha1_empirical_band"
                        if strength == 1.0
                        else "canonical_exact_raw_alpha2_empirical_band"
                    ),
                    direction_convention="normalize(J.T @ W_U[token])",
                    strength=strength,
                    repeats=3 if strength == CANONICAL_STRENGTH else 1,
                )
            )

    spider = items[0]
    spider_context = contexts[spider["name"]]
    legacy_source_id, _ = leading_label_token_id(
        bundle.tokenizer, spider["intermediate"]
    )
    legacy_target_id, _ = leading_label_token_id(bundle.tokenizer, spider["swap_to"])
    detailed_specs = [
        (
            "legacy_leading_folded_alpha1_generic_band",
            folded_bank,
            prior + ([max(prior) + 1] if max(prior) + 1 in all_source_layers else []),
            legacy_source_id,
            legacy_target_id,
            1.0,
            None,
            "normalize(J.T @ (gain * W_U[token]))",
        ),
        (
            "legacy_leading_raw_alpha1_generic_band",
            raw_bank,
            prior + ([max(prior) + 1] if max(prior) + 1 in all_source_layers else []),
            legacy_source_id,
            legacy_target_id,
            1.0,
            None,
            "normalize(J.T @ W_U[token])",
        ),
        (
            "alternate_leading_target_raw_alpha2_empirical_band",
            raw_bank,
            empirical_band,
            spider["source_concept_token_id"],
            legacy_target_id,
            2.0,
            None,
            "normalize(J.T @ W_U[token])",
        ),
        (
            "exact_folded_alpha2_empirical_band",
            folded_bank,
            empirical_band,
            spider["source_concept_token_id"],
            spider["target_concept_token_id"],
            2.0,
            None,
            "normalize(J.T @ (gain * W_U[token]))",
        ),
        (
            "exact_raw_alpha2_paper_prior",
            raw_bank,
            prior,
            spider["source_concept_token_id"],
            spider["target_concept_token_id"],
            2.0,
            None,
            "normalize(J.T @ W_U[token])",
        ),
        (
            "exact_raw_alpha2_adjacent_late_layer",
            raw_bank,
            sorted(set(empirical_band) | {max(empirical_band) + 1}),
            spider["source_concept_token_id"],
            spider["target_concept_token_id"],
            2.0,
            None,
            "normalize(J.T @ W_U[token])",
        ),
    ]
    source_key = f"source:{spider['source_concept_surface']}"
    best_position = int(
        readout[spider["name"]]["candidates"][source_key]["best_position"]
    )
    detailed_specs.extend(
        [
            (
                "exact_raw_alpha2_best_readout_position_only",
                raw_bank,
                empirical_band,
                spider["source_concept_token_id"],
                spider["target_concept_token_id"],
                2.0,
                [best_position],
                "normalize(J.T @ W_U[token])",
            ),
            (
                "exact_raw_alpha2_terminal_position_only",
                raw_bank,
                empirical_band,
                spider["source_concept_token_id"],
                spider["target_concept_token_id"],
                2.0,
                [-1],
                "normalize(J.T @ W_U[token])",
            ),
        ]
    )
    for (
        name,
        bank,
        layers,
        source_id,
        target_id,
        strength,
        positions,
        convention,
    ) in detailed_specs:
        attempts.append(
            _attempt(
                bundle=bundle,
                item=spider,
                input_ids=spider_context["input_ids"],
                clean_logits=spider_context["clean_logits"],
                residuals=spider_context["residuals"],
                bank=bank,
                layers=layers,
                source_token_id=source_id,
                target_token_id=target_id,
                name=name,
                direction_convention=convention,
                strength=strength,
                positions=positions,
            )
        )

    no_op_edits = clamped_swap_edits(
        {layer: spider_context["residuals"][layer] for layer in empirical_band},
        {
            layer: raw_bank[spider["source_concept_token_id"]][layer]
            for layer in empirical_band
        },
        {
            layer: raw_bank[spider["target_concept_token_id"]][layer]
            for layer in empirical_band
        },
        strength=0.0,
    )
    no_op = forward_logits(
        bundle.hf_model,
        spider_context["input_ids"],
        blocks=bundle.lens_model.layers,
        edits=no_op_edits,
    )
    no_op_max_error = float(
        (no_op - spider_context["clean_logits"]).abs().max().cpu()
    )

    canonical = [
        row
        for row in attempts
        if row["name"] == "canonical_exact_raw_alpha2_empirical_band"
    ]
    for row in canonical:
        item_readout = readout[row["item_name"]]
        item = next(value for value in items if value["name"] == row["item_name"])
        key = f"source:{item['source_concept_surface']}"
        ranks = item_readout["candidates"][key]["ranks_by_layer_position"]
        row["minimum_source_readout_rank_in_band"] = min(
            min(ranks[str(layer)]) for layer in empirical_band
        )
    g_swap_pass = bool(
        no_op_max_error == 0.0
        and len(canonical) == len(CALIBRATION_ITEM_NAMES)
        and all(row["strict_pass"] for row in canonical)
        and all(row["minimum_source_readout_rank_in_band"] <= 10 for row in canonical)
    )
    stage1: dict[str, Any] = {
        "status": "PASS" if g_swap_pass else "FAIL",
        "model": {
            "model_id": bundle.model_id,
            "revision": bundle.revision,
            "dtype": str(next(bundle.hf_model.parameters()).dtype),
            "n_layers": int(bundle.lens_model.n_layers),
        },
        "g1": g1,
        "token_resolution": {
            "rule": "exact upstream JSON label first, then leading-space fallback",
            "items": [
                {
                    key: value
                    for key, value in item.items()
                    if key
                    in {
                        "name",
                        "intermediate",
                        "swap_to",
                        "source_concept_token_id",
                        "source_concept_surface",
                        "target_concept_token_id",
                        "target_concept_surface",
                        "source_token_candidates",
                        "target_token_candidates",
                    }
                }
                for item in items
            ],
        },
        "workspace_discovery": {
            "selection_uses_swap_outcomes": False,
            "paper_source": "https://transformer-circuits.pub/2026/workspace/index.html",
            "paper_normalized_prior_percent": [
                PAPER_WORKSPACE_START_PERCENT,
                PAPER_WORKSPACE_END_PERCENT,
            ],
            "paper_prior_layers": prior,
            "visibility_rule": (
                "longest contiguous run in prior with median minimum source "
                "J-Lens rank <= 10 across three clean prompts"
            ),
            "selected_layers": empirical_band,
            "layer_diagnostics": {
                str(layer): value for layer, value in band_diagnostics.items()
            },
            "readout": readout,
        },
        "ant_surface_validation": _cue_surface_validation(bundle, lens, empirical_band),
        "attempts": attempts,
        "alpha0_no_op_max_abs_logit_error": no_op_max_error,
        "g_swap": {
            "status": "PASS" if g_swap_pass else "FAIL",
            "criterion": (
                "clean expected answer top-1; clean source concept rank <=10; "
                "counterfactual answer unique top-1 under one fixed exact/raw/"
                "alpha2/all-position configuration; three identical repeats; "
                "spider plus two predeclared upstream controls"
            ),
            "canonical_configuration": {
                "layers": empirical_band,
                "positions": "all_prompt_positions",
                "direction": "normalize(J.T @ W_U[token])",
                "strength": CANONICAL_STRENGTH,
                "token_resolution": "exact_label_first",
            },
            "canonical_rows": canonical,
            "n_pass": sum(int(row["strict_pass"]) for row in canonical),
            "n_required": len(CALIBRATION_ITEM_NAMES),
            "alpha0_no_op_max_abs_logit_error": no_op_max_error,
            "bidirectional_reliability_established": False,
        },
        "cross_function_checks": _cross_function_checks(
            bundle,
            raw_bank,
            source_token_id=spider["source_concept_token_id"],
            target_token_id=spider["target_concept_token_id"],
            layers=empirical_band,
        ),
        "limitations": [
            "The exact ant token is a design convention; leading-space ant can rank better on matched cues.",
            "Reverse ant-to-spider did not establish bidirectional swap reliability.",
            "The Qwen workspace band is a clean-readout calibration, not a band published by Anthropic for Qwen2.5.",
            "Alpha=2 is documented in the paper but is a stronger intervention than a unit coordinate exchange.",
        ],
    }
    stage1["figure"] = _plot_gate(stage1)
    return stage1


def _report(metrics: Mapping[str, Any]) -> str:
    repair = metrics["repair_v2"]
    preflight = repair["preflight"]
    stage0 = repair["stage0"]
    stage1 = repair["stage1"]
    gpu = preflight["gpu"]
    disk = preflight["disk"]
    rows = stage1["g_swap"]["canonical_rows"]
    table = "\n".join(
        "| {name} | `{source}`→`{target}` | `{clean}` | `{edited}` | "
        "{clean_m:.3f} | {edited_m:.3f} | {rank} | {status} |".format(
            name=row["item_name"],
            source=row["source_surface"],
            target=row["target_surface"],
            clean=row["clean_top_token"],
            edited=row["edited_top_token"],
            clean_m=row["clean_metric"],
            edited_m=row["edited_metric"],
            rank=row["minimum_source_readout_rank_in_band"],
            status="PASS" if row["strict_pass"] else "FAIL",
        )
        for row in rows
    )
    workspace = stage1["workspace_discovery"]
    return f"""# Repair-first replication report (v2)

## Current verdict

**G-SWAP {stage1['g_swap']['status']}; SCIENCE NOT YET RUN.** The causal swap
instrument now passes the three-case known-answer calibration, but the v2
hypothesis remains undecided until G-DIR, READ validation, firing controls, and
G-POS are completed. The v1 `NOT SUPPORTED` / `REFUTED` labels remain withdrawn
as scientific conclusions.

## Environment

- GPU: {gpu.get('name')}; {gpu.get('memory_total_mib')} MiB total; {gpu.get('memory_free_mib')} MiB free at recorded preflight.
- Home/HF-cache filesystem: {disk.get('total_gib'):.1f} GiB total; {disk.get('free_gib'):.1f} GiB free at recorded notebook preflight.
- Required tool/auth preflight: **{preflight['status']}**.

## Stage 0 — upstream diagnosis

- Released walkthrough readout: **{stage0['upstream_readout']['status']}** on `{stage0['upstream_readout']['model_name']}`.
- Released executable causal swap: **NOT RUNNABLE**; the public walkthrough is readout-only.
- Decision: `{stage0['decision']}`. This omission does not establish a Qwen model mismatch.

![F0 Stage-0 audit](figures/f0_stage0_upstream_audit.png)

## Stage 1a — repaired known-answer swap

The configuration was frozen before testing swap outcomes. The paper's
approximately 38–92% workspace-depth prior maps to layers
`{workspace['paper_prior_layers']}`. Within it, the longest contiguous run where
the median clean source-concept J-Lens rank across three prompts is top-10 is
**layers {workspace['selected_layers'][0]}–{workspace['selected_layers'][-1]}**.

The repaired convention is: exact upstream JSON label when it is one token
(notably `ant` id 517), paper-literal raw `J.T @ W_U`, all prompt positions,
and the paper's documented double-strength swap (`alpha=2`).

| item | concept swap | clean top-1 | edited top-1 | clean M | edited M | min source rank | gate |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
{table}

`M = logit(clean answer) - logit(counterfactual answer)`. All canonical runs
were repeated three times with identical logits/top-1 results. The alpha-zero
clean-clamp maximum logit error was
`{stage1['alpha0_no_op_max_abs_logit_error']:.3g}`.

![G-SWAP calibration](figures/repair_gswap_calibration.png)

### G-SWAP decision

**{stage1['g_swap']['status']} ({stage1['g_swap']['n_pass']}/{stage1['g_swap']['n_required']}).**
This licenses only the next repair/calibration notebooks. It does not license
P1–P3 or a claim about the WRITE-versus-READ hypothesis.

Important limits: the leading-space ` ant` token can have better clean readout
rank on matched cues, reverse ant→spider did not flip 6→8, and alpha=2 is a
stronger intervention than an unscaled coordinate exchange. Those facts are
persisted rather than hidden.

## Gate ledger

| gate | status | consequence |
| --- | --- | --- |
| Stage-0 preflight | {preflight['status']} | Environment usable |
| Upstream causal swap | NOT RUNNABLE | Release omission |
| G1 HF/J-Lens logits | {stage1['g1']['status']} | max mean KL={stage1['g1']['max_prompt_mean_kl']:.3e}, N={stage1['g1']['n']} |
| G-SWAP | {stage1['g_swap']['status']} | Proceed to G-DIR and READ validation only |
| G-DIR | NOT RUN IN V2 | Notebook 02 next |
| READ validation | NOT RUN IN V2 | Notebook 03 pending |
| G-POS / firing controls | NOT RUN IN V2 | Stage 2 pending |
| Stage-3 science | PROHIBITED | Calibration chain incomplete |
"""


def persist_stage1(stage1: Mapping[str, Any]) -> dict[str, Any]:
    """Persist Stage-1 evidence and update the live gate ledger/report."""

    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    repair = metrics["repair_v2"]
    repair["stage1"] = dict(stage1)
    repair["gate_ledger"]["g1"] = stage1["g1"]["status"]
    repair["gate_ledger"]["g_swap"] = stage1["g_swap"]["status"]
    repair["gate_ledger"]["g_dir"] = "NOT_RUN_V2"
    repair["gate_ledger"]["read_validation"] = "NOT_RUN_V2"
    repair["gate_ledger"]["stage3_science"] = "PROHIBITED"
    repair["current_allowed_conclusion"] = (
        "G_SWAP_REPAIRED_CALIBRATION_INCOMPLETE"
        if stage1["g_swap"]["status"] == "PASS"
        else "STAGE4_REPLICATION_FAILURE"
    )
    save_json(metrics_path, metrics)
    (ROOT / "results" / "RESULTS.md").write_text(
        _report(metrics), encoding="utf-8"
    )
    return metrics

"""Stage-2 firing controls, specificity checks, and narration positive gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jlens.hooks import ActivationRecorder

from src.controls_phase import (
    ABSENT_CONCEPT_SURFACES_V1,
    CAPABILITY_TEXTS_V1,
    load_known_narration_source,
    seeded_random_direction_bank,
    teacher_forced_nll,
)
from src.data_gen import continuation_token_id
from src.interventions import ablation_edits, clamped_swap_edits, forward_logits
from src.jlens_iface import jlens_direction_bank, token_rank, unembedding_weight
from src.metrics import logit_difference, save_json
from src.model_utils import ModelBundle, capture_residuals, decode_topk
from src.plotting import save_figure, set_style
from src.v2_read import _layer_aligned_weight_read
from src.localization_phase import flag_top_components, localize_source_direction
from src.v2_repair import load_calibration_items


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
RANDOM_DRAWS = 64
CONTINUATION_TOKENS = 16
# Frozen before any v2 intervention outcome.  These common, single-token
# function-word sets make a robust language-vs-English mass metric without
# selecting a favorable continuation token after seeing an edit.
LANGUAGE_MASS_SURFACES = {
    "English": (
        " the", " a", " an", " he", " she", " they", " it", " and",
        " but", " with", " for", " from", " was", " is", " in", " on",
    ),
    "French": (
        " le", " la", " les", " un", " une", " il", " elle", " ils",
        " elles", " et", " mais", " avec", " pour", " dans", " était",
        " est", " sur",
    ),
    "German": (
        " der", " die", " das", " ein", " eine", " er", " sie", " es",
        " und", " aber", " mit", " für", " von", " war", " ist", " im",
        " auf",
    ),
    "Spanish": (
        " el", " la", " los", " las", " un", " una", " él", " ella",
        " ellos", " ellas", " y", " pero", " con", " para", " de", " era",
        " es", " en",
    ),
    "Italian": (
        " il", " lo", " la", " i", " gli", " le", " un", " una", " lui",
        " lei", " loro", " e", " ma", " con", " per", " da", " era", " è",
    ),
}
DIRECT_PROBES = {
    "spider-legs": "Which animal spins webs? Answer with exactly one word:",
    "animal-legs-buffalo2": (
        "Which large horned bovine roamed the American plains in herds? "
        "Answer with exactly one word:"
    ),
    "chem-photosynthesis-Z": (
        "Which gas do plants release during photosynthesis? "
        "Answer with exactly one word:"
    ),
}


def subtract_output_logit(
    logits: torch.Tensor, token_id: int, amount: float = 1.0
) -> torch.Tensor:
    """Return a post-logit control arm with one vocabulary logit reduced."""

    if amount <= 0 or not math.isfinite(float(amount)):
        raise ValueError("Suppression amount must be finite and positive")
    edited = logits.clone()
    edited[..., int(token_id)] -= float(amount)
    return edited


def subtract_output_token_set(
    logits: torch.Tensor,
    token_ids: Sequence[int],
    positions: Sequence[int],
    amount: float = 1.0,
) -> torch.Tensor:
    """Suppress a complete token family at predeclared scoring positions."""

    if amount <= 0 or not math.isfinite(float(amount)):
        raise ValueError("Suppression amount must be finite and positive")
    ids = [int(token_id) for token_id in token_ids]
    places = [int(position) for position in positions]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Suppression token IDs must be nonempty and unique")
    if not places or len(set(places)) != len(places):
        raise ValueError("Suppression positions must be nonempty and unique")
    if min(ids) < 0 or max(ids) >= logits.shape[-1]:
        raise IndexError("Suppression token ID outside vocabulary")
    if min(places) < 0 or max(places) >= logits.shape[1]:
        raise IndexError("Suppression position outside sequence")
    edited = logits.clone()
    position_index = torch.tensor(places, device=edited.device)
    token_index = torch.tensor(ids, device=edited.device)
    edited[0, position_index[:, None], token_index[None, :]] -= float(amount)
    return edited


def language_mass_metric(
    logits: torch.Tensor,
    positions: Sequence[int],
    source_ids: Sequence[int],
    english_ids: Sequence[int],
) -> torch.Tensor:
    """Mean normalized language-family logit mass over fixed positions."""

    places = [int(position) for position in positions]
    source = [int(token_id) for token_id in source_ids]
    english = [int(token_id) for token_id in english_ids]
    if not places or not source or not english:
        raise ValueError("Language-mass metric inputs must be nonempty")
    if set(source) & set(english):
        raise ValueError("Source-language and English token families must be disjoint")
    values = logits[0, places].float()
    source_index = torch.tensor(source, device=values.device)
    english_index = torch.tensor(english, device=values.device)
    source_mass = torch.logsumexp(values[:, source_index], dim=-1) - math.log(
        len(source)
    )
    english_mass = torch.logsumexp(values[:, english_index], dim=-1) - math.log(
        len(english)
    )
    return (source_mass - english_mass).mean()


def _single_token_families(tokenizer: Any) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for language, surfaces in LANGUAGE_MASS_SURFACES.items():
        ids: list[int] = []
        for surface in surfaces:
            encoded = tokenizer.encode(surface, add_special_tokens=False)
            if len(encoded) != 1 or tokenizer.decode(encoded) != surface:
                raise ValueError(
                    f"Frozen language surface is not one exact token: {surface!r}"
                )
            ids.append(int(encoded[0]))
        if len(set(ids)) != len(ids):
            raise ValueError(f"Frozen {language} token family contains aliases")
        families[language] = {"surfaces": list(surfaces), "token_ids": ids}
    english = set(families["English"]["token_ids"])
    for language, family in families.items():
        if language != "English" and english & set(family["token_ids"]):
            raise ValueError(f"Frozen {language}/English token families overlap")
    return families


def _greedy_continuation(
    bundle: ModelBundle, prompt_ids: torch.Tensor
) -> torch.Tensor:
    """Freeze exactly 16 clean greedy tokens before any edited forward."""

    prefix = prompt_ids
    generated: list[torch.Tensor] = []
    special_ids = {
        value
        for value in (
            bundle.tokenizer.eos_token_id,
            bundle.tokenizer.pad_token_id,
            bundle.tokenizer.bos_token_id,
        )
        if value is not None
    }
    with torch.no_grad():
        for _ in range(CONTINUATION_TOKENS):
            logits = forward_logits(bundle.hf_model, prefix)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            if int(next_id.item()) in special_ids:
                raise RuntimeError("Clean narration rollout reached a special token")
            generated.append(next_id)
            prefix = torch.cat([prefix, next_id], dim=1)
    return torch.cat(generated, dim=1)


def _metric_residual_read(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    directions: Mapping[int, torch.Tensor],
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    intervention_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Secondary attribution diagnostic for an arbitrary scalar metric."""

    layers = sorted(int(layer) for layer in directions)
    positions = (
        list(range(input_ids.shape[1]))
        if intervention_positions is None
        else [int(position) for position in intervention_positions]
    )
    with torch.enable_grad(), ActivationRecorder(
        bundle.lens_model.layers, at=layers, start_graph_at=layers[0]
    ) as recorder:
        logits = bundle.hf_model(input_ids=input_ids, use_cache=False).logits.float()
        metric = metric_fn(logits)
        if metric.ndim != 0 or not torch.isfinite(metric):
            raise ValueError("Attribution metric_fn must return one finite scalar")
        activations = tuple(recorder.activations[layer] for layer in layers)
        gradients = torch.autograd.grad(metric, activations, allow_unused=False)
    rows: dict[str, list[float]] = {}
    all_reads: list[np.ndarray] = []
    for layer, activation, gradient in zip(
        layers, activations, gradients, strict=True
    ):
        vector = directions[layer].detach().to(activation.device, torch.float32)
        read = gradient[0, positions].detach().float() @ vector
        values = read.cpu().numpy()
        rows[str(layer)] = [float(value) for value in values]
        all_reads.append(values)
    flattened = np.concatenate(all_reads)
    return {
        "metric": float(metric.detach().cpu()),
        "read_by_layer_position": rows,
        "read_abs_mean": float(np.mean(np.abs(flattened))),
    }


def _direct_language_prompt(text: str) -> str:
    return (
        f"Here is a passage of text:\n\n{text}\n\n"
        "Question: Which language is this passage written in? "
        "Answer with exactly one word:"
    )


def _pair_metric(logits: torch.Tensor, source_id: int, foil_id: int) -> float:
    return float(logit_difference(logits, source_id, foil_id)[0].cpu())


def _canonical_swap(
    bundle: ModelBundle,
    item: dict[str, Any],
    source: dict[int, torch.Tensor],
    target: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    input_ids = bundle.lens_model.encode(item["prompt"])
    clean = forward_logits(bundle.hf_model, input_ids)
    residuals = capture_residuals(bundle.lens_model, input_ids, source)
    edited = forward_logits(
        bundle.hf_model,
        input_ids,
        blocks=bundle.lens_model.layers,
        edits=clamped_swap_edits(residuals, source, target, strength=2.0),
    )
    return clean, edited, residuals


def _reverify_gswap(
    bundle: ModelBundle,
    items: list[dict[str, Any]],
    bank: dict[int, dict[int, torch.Tensor]],
    layers: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        source = {
            layer: bank[item["source_concept_token_id"]][layer] for layer in layers
        }
        target = {
            layer: bank[item["target_concept_token_id"]][layer] for layer in layers
        }
        clean, edited, _ = _canonical_swap(bundle, item, source, target)
        clean_metric = _pair_metric(
            clean,
            item["clean_answer_token_id"],
            item["counterfactual_answer_token_id"],
        )
        edited_metric = _pair_metric(
            edited,
            item["clean_answer_token_id"],
            item["counterfactual_answer_token_id"],
        )
        passed = bool(
            int(clean[0, -1].argmax()) == item["clean_answer_token_id"]
            and int(edited[0, -1].argmax()) == item["counterfactual_answer_token_id"]
        )
        rows.append(
            {
                "name": item["name"],
                "clean_metric": clean_metric,
                "edited_metric": edited_metric,
                "delta": edited_metric - clean_metric,
                "clean_top": bundle.tokenizer.decode([int(clean[0, -1].argmax())]),
                "edited_top": bundle.tokenizer.decode([int(edited[0, -1].argmax())]),
                "pass": passed,
            }
        )
    return {
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "rows": rows,
    }


def _direct_concept_controls(
    bundle: ModelBundle,
    items: list[dict[str, Any]],
    bank: dict[int, dict[int, torch.Tensor]],
    layers: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        prompt = DIRECT_PROBES[item["name"]]
        source_output_id, source_output_surface = continuation_token_id(
            bundle.tokenizer, prompt, item["intermediate"]
        )
        foil_output_id, foil_output_surface = continuation_token_id(
            bundle.tokenizer, prompt, item["swap_to"]
        )
        input_ids = bundle.lens_model.encode(prompt)
        clean = forward_logits(bundle.hf_model, input_ids)
        source = {
            layer: bank[item["source_concept_token_id"]][layer] for layer in layers
        }
        target = {
            layer: bank[item["target_concept_token_id"]][layer] for layer in layers
        }
        residuals = capture_residuals(bundle.lens_model, input_ids, layers)
        internal = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=clamped_swap_edits(residuals, source, target, strength=2.0),
        )
        ablated = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=ablation_edits(source, strength=1.0),
        )
        suppressed = subtract_output_logit(clean, source_output_id)
        clean_metric = _pair_metric(clean, source_output_id, foil_output_id)
        internal_metric = _pair_metric(internal, source_output_id, foil_output_id)
        ablation_metric = _pair_metric(ablated, source_output_id, foil_output_id)
        suppression_metric = _pair_metric(suppressed, source_output_id, foil_output_id)
        rows.append(
            {
                "name": item["name"],
                "prompt": prompt,
                "source_output_token_id": source_output_id,
                "source_output_surface": source_output_surface,
                "foil_output_token_id": foil_output_id,
                "foil_output_surface": foil_output_surface,
                "clean_metric": clean_metric,
                "internal_metric": internal_metric,
                "internal_delta": internal_metric - clean_metric,
                "ablation_metric": ablation_metric,
                "ablation_delta": ablation_metric - clean_metric,
                "suppression_metric": suppression_metric,
                "suppression_delta": suppression_metric - clean_metric,
                "clean_pairwise_correct": clean_metric > 0,
                "suppression_fired": math.isclose(
                    suppression_metric - clean_metric, -1.0, abs_tol=1e-6
                ),
                "clean_top_tokens": decode_topk(bundle.tokenizer, clean[0, -1], 5),
                "internal_top_tokens": decode_topk(
                    bundle.tokenizer, internal[0, -1], 5
                ),
            }
        )
    return {
        "status": (
            "PASS"
            if all(row["clean_pairwise_correct"] and row["suppression_fired"] for row in rows)
            else "FAIL"
        ),
        "rows": rows,
        "control": "subtract 1.0 from the source concept-answer logit",
    }


def _instruction_positions(
    tokenizer: Any, prompt: str, passage: str
) -> list[int]:
    passage_end = prompt.index(passage) + len(passage)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    offsets = encoded.offset_mapping[0]
    positions = [
        index
        for index, (start, end) in enumerate(offsets.tolist())
        if end > start and start >= passage_end
    ]
    if not positions:
        raise ValueError("No instruction positions follow the language passage")
    return positions


def _task_read(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    directions: dict[int, torch.Tensor],
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    source_layer: int,
    intervention_positions: Sequence[int] | None,
    seed: int,
) -> dict[str, Any]:
    attribution = _metric_residual_read(
        bundle,
        input_ids,
        directions,
        metric_fn,
        intervention_positions=intervention_positions,
    )
    localization = localize_source_direction(
        bundle.hf_model,
        bundle.lens_model.layers,
        input_ids,
        directions[source_layer],
        source_layer=source_layer,
        target_token_id=0,
        foil_token_id=1,
        intervention_positions=intervention_positions,
        component_layers=list(
            range(source_layer + 1, len(bundle.lens_model.layers))
        ),
        metric_fn=metric_fn,
    )
    flags = flag_top_components(localization, top_k_mlps=2, top_k_heads=4)
    weight = _layer_aligned_weight_read(bundle, directions, flags, seed=seed)
    return {
        "attribution": attribution,
        "source_layer": source_layer,
        "flags": flags,
        "weight": weight,
        "selection_warning": (
            "Weight magnitude is activation-independent; task dependence enters "
            "through separately attribution-localized consumer sets at one shared "
            "WRITE-selected source layer."
        ),
    }


def _language_controls_and_gpos(
    bundle: ModelBundle,
    lens: Any,
    payload: dict[str, Any],
    bank: dict[int, dict[int, torch.Tensor]],
    language_direction_ids: dict[str, int],
    english_direction_id: int,
    intervention_layers: list[int],
    token_families: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run aligned, teacher-forced language narration controls."""

    family_json = json.dumps(
        token_families, sort_keys=True, separators=(",", ":")
    ).encode()
    family_sha256 = hashlib.sha256(family_json).hexdigest()
    rows: list[dict[str, Any]] = []
    for index, passage in enumerate(payload["passages"]):
        category = passage["category"]
        key = passage["key"]
        text = passage["text"]
        source_direction_id = language_direction_ids[category]
        directions = bank[source_direction_id]
        english_directions = bank[english_direction_id]
        swap_source = {
            layer: directions[layer] for layer in intervention_layers
        }
        swap_english = {
            layer: english_directions[layer] for layer in intervention_layers
        }
        source_family_ids = token_families[category]["token_ids"]
        english_family_ids = token_families["English"]["token_ids"]

        automatic_prompt = payload["task"]["automatic_q"].format(text=text)
        prompt_ids = bundle.lens_model.encode(automatic_prompt)
        prompt_length = int(prompt_ids.shape[1])
        continuation_ids = _greedy_continuation(bundle, prompt_ids)
        full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
        score_positions = list(
            range(prompt_length - 1, prompt_length + CONTINUATION_TOKENS - 1)
        )
        prompt_positions = list(range(prompt_length))

        def automatic_metric(logits: torch.Tensor) -> torch.Tensor:
            return language_mass_metric(
                logits,
                score_positions,
                source_family_ids,
                english_family_ids,
            )

        auto_clean = forward_logits(bundle.hf_model, full_ids)
        frozen_ids = [int(value) for value in continuation_ids[0].tolist()]
        teacher_forced_top = [
            int(auto_clean[0, position].argmax()) for position in score_positions
        ]
        teacher_forced_matches = sum(
            observed == expected
            for observed, expected in zip(
                teacher_forced_top, frozen_ids, strict=True
            )
        )
        auto_residuals = capture_residuals(
            bundle.lens_model, full_ids, intervention_layers
        )
        auto_internal = forward_logits(
            bundle.hf_model,
            full_ids,
            blocks=bundle.lens_model.layers,
            edits=clamped_swap_edits(
                auto_residuals,
                swap_source,
                swap_english,
                positions=prompt_positions,
                strength=2.0,
            ),
        )
        auto_clean_metric = float(automatic_metric(auto_clean).cpu())
        auto_internal_metric = float(automatic_metric(auto_internal).cpu())
        source_arm = subtract_output_token_set(
            auto_clean, source_family_ids, score_positions
        )
        english_arm = subtract_output_token_set(
            auto_clean, english_family_ids, score_positions
        )
        source_effect = float(automatic_metric(source_arm).cpu()) - auto_clean_metric
        english_effect = (
            float(automatic_metric(english_arm).cpu()) - auto_clean_metric
        )
        centered_sham = 0.5 * (source_effect + english_effect)

        readout, _, _ = lens.apply(
            bundle.lens_model,
            automatic_prompt,
            layers=intervention_layers,
            positions=None,
        )
        rank_by_layer_position = {
            int(layer): [
                token_rank(readout[layer][position], source_direction_id)
                for position in range(readout[layer].shape[0])
            ]
            for layer in intervention_layers
        }
        instruction_positions = _instruction_positions(
            bundle.tokenizer, automatic_prompt, text
        )
        instruction_rank_by_layer = {
            layer: [rank_by_layer_position[layer][position] for position in instruction_positions]
            for layer in intervention_layers
        }
        min_rank = min(
            rank for values in rank_by_layer_position.values() for rank in values
        )
        instruction_min_rank = min(
            rank for values in instruction_rank_by_layer.values() for rank in values
        )
        source_layer = min(
            intervention_layers,
            key=lambda layer: (min(rank_by_layer_position[layer]), layer),
        )

        direct_prompt = _direct_language_prompt(text)
        direct_source_id, direct_source_surface = continuation_token_id(
            bundle.tokenizer, direct_prompt, category
        )
        direct_english_id, direct_english_surface = continuation_token_id(
            bundle.tokenizer, direct_prompt, "English"
        )
        if (
            direct_source_id != source_direction_id
            or direct_english_id != english_direction_id
        ):
            raise RuntimeError("Language direction and direct-output coordinates differ")
        direct_ids = bundle.lens_model.encode(direct_prompt)

        def direct_metric(logits: torch.Tensor) -> torch.Tensor:
            return (
                logits[0, -1, direct_source_id].float()
                - logits[0, -1, direct_english_id].float()
            )

        direct_clean = forward_logits(bundle.hf_model, direct_ids)
        direct_residuals = capture_residuals(
            bundle.lens_model, direct_ids, intervention_layers
        )
        direct_internal = forward_logits(
            bundle.hf_model,
            direct_ids,
            blocks=bundle.lens_model.layers,
            edits=clamped_swap_edits(
                direct_residuals, swap_source, swap_english, strength=2.0
            ),
        )
        direct_clean_metric = float(direct_metric(direct_clean).cpu())
        direct_internal_metric = float(direct_metric(direct_internal).cpu())
        direct_suppressed = subtract_output_logit(direct_clean, direct_source_id)
        direct_suppression_effect = (
            float(direct_metric(direct_suppressed).cpu()) - direct_clean_metric
        )
        direct_clean_top_id = int(direct_clean[0, -1].argmax())
        direct_internal_top_id = int(direct_internal[0, -1].argmax())

        shared_seed = SEED + 100_000 + index * 1_000
        auto_read = _task_read(
            bundle,
            full_ids,
            directions,
            automatic_metric,
            source_layer=source_layer,
            intervention_positions=prompt_positions,
            seed=shared_seed,
        )
        direct_read = _task_read(
            bundle,
            direct_ids,
            directions,
            direct_metric,
            source_layer=source_layer,
            intervention_positions=None,
            seed=shared_seed,
        )
        direct_mlp = float(direct_read["weight"]["mlp_primary"])
        direct_attention = float(direct_read["weight"]["attention_primary"])
        read_valid = bool(
            math.isfinite(direct_mlp)
            and math.isfinite(direct_attention)
            and direct_mlp > 1e-8
            and direct_attention > 1e-8
        )
        mlp_ratio = (
            float(auto_read["weight"]["mlp_primary"]) / direct_mlp
            if read_valid
            else None
        )
        attention_ratio = (
            float(auto_read["weight"]["attention_primary"]) / direct_attention
            if read_valid
            else None
        )
        primary_weight_ratio = (
            max(mlp_ratio, attention_ratio)
            if mlp_ratio is not None and attention_ratio is not None
            else None
        )
        direct_attribution = direct_read["attribution"]["read_abs_mean"]
        attribution_ratio = (
            auto_read["attribution"]["read_abs_mean"] / direct_attribution
            if direct_attribution > 1e-12
            else None
        )
        auto_delta = auto_internal_metric - auto_clean_metric
        direct_delta = direct_internal_metric - direct_clean_metric
        checks = {
            "clean_continuation_capable": auto_clean_metric > 0.0,
            "high_write_all_prompt": min_rank <= 10,
            "low_causal_abs_delta": abs(auto_delta - centered_sham) <= 0.5,
            "low_causal_relative_to_direct": abs(auto_delta)
            <= 0.25 * abs(direct_delta),
            "direct_clean_top1_source": direct_clean_top_id == direct_source_id,
            "direct_internal_top1_english": (
                direct_internal_top_id == direct_english_id
            ),
            "low_primary_weight_read_ratio": (
                primary_weight_ratio is not None and primary_weight_ratio <= 0.5
            ),
            "continuation_suppression_arms_fire": math.isclose(
                source_effect, -1.0, abs_tol=1e-5
            )
            and math.isclose(english_effect, 1.0, abs_tol=1e-5),
            "direct_label_suppression_fires": math.isclose(
                direct_suppression_effect, -1.0, abs_tol=1e-5
            ),
        }
        rows.append(
            {
                "key": key,
                "category": category,
                "automatic_prompt": automatic_prompt,
                "language_direction_token_id": source_direction_id,
                "language_direction_surface": bundle.tokenizer.decode(
                    [source_direction_id]
                ),
                "english_direction_token_id": english_direction_id,
                "minimum_all_prompt_language_rank": min_rank,
                "minimum_instruction_span_language_rank_diagnostic": (
                    instruction_min_rank
                ),
                "language_rank_by_layer_position": {
                    str(layer): values
                    for layer, values in rank_by_layer_position.items()
                },
                "instruction_positions": instruction_positions,
                "write_selected_source_layer": source_layer,
                "frozen_continuation": {
                    "n_tokens": CONTINUATION_TOKENS,
                    "token_ids": frozen_ids,
                    "text": bundle.tokenizer.decode(frozen_ids),
                    "teacher_forced_argmax_matches": teacher_forced_matches,
                    "teacher_forced_argmax_match_fraction": (
                        teacher_forced_matches / CONTINUATION_TOKENS
                    ),
                    "teacher_forced_argmax_note": (
                        "Shape-dependent BF16 tie-breaking is recorded, not used "
                        "to select or replace the frozen clean rollout."
                    ),
                    "score_positions": score_positions,
                    "intervention_positions": prompt_positions,
                },
                "automatic_metric_definition": (
                    "mean normalized logsumexp(source-language token family) - "
                    "normalized logsumexp(English token family) over a frozen "
                    "16-token clean teacher-forced rollout"
                ),
                "automatic_clean_metric": auto_clean_metric,
                "automatic_internal_metric": auto_internal_metric,
                "automatic_internal_delta": auto_delta,
                "source_suppression_effect": source_effect,
                "english_suppression_effect": english_effect,
                "centered_firing_sham_effect": centered_sham,
                "centered_sham_note": (
                    "Algebraic center of two independently firing arms; not an "
                    "independent control observation."
                ),
                "direct_source": {
                    "token_id": direct_source_id,
                    "surface": direct_source_surface,
                },
                "direct_english": {
                    "token_id": direct_english_id,
                    "surface": direct_english_surface,
                },
                "direct_clean_metric": direct_clean_metric,
                "direct_internal_metric": direct_internal_metric,
                "direct_internal_delta": direct_delta,
                "direct_clean_top_id": direct_clean_top_id,
                "direct_internal_top_id": direct_internal_top_id,
                "direct_clean_top_tokens": decode_topk(
                    bundle.tokenizer, direct_clean[0, -1], 5
                ),
                "direct_internal_top_tokens": decode_topk(
                    bundle.tokenizer, direct_internal[0, -1], 5
                ),
                "direct_suppression_effect": direct_suppression_effect,
                "automatic_read": auto_read,
                "direct_read": direct_read,
                "mlp_weight_read_ratio_auto_over_direct": mlp_ratio,
                "attention_weight_read_ratio_auto_over_direct": attention_ratio,
                "primary_weight_read_ratio_auto_over_direct": primary_weight_ratio,
                "attribution_read_ratio_auto_over_direct_secondary": (
                    attribution_ratio
                ),
                "checks": checks,
                "joint_reproduction": all(checks.values()),
            }
        )
    firing = {
        "status": (
            "PASS"
            if all(
                row["checks"]["continuation_suppression_arms_fire"]
                and row["checks"]["direct_label_suppression_fires"]
                for row in rows
            )
            else "FAIL"
        ),
        "rows": [
            {
                "key": row["key"],
                "category": row["category"],
                "source_effect": row["source_suppression_effect"],
                "english_effect": row["english_suppression_effect"],
                "direct_label_effect": row["direct_suppression_effect"],
            }
            for row in rows
        ],
    }
    reproduced = [row for row in rows if row["joint_reproduction"]]
    categories = {row["category"] for row in reproduced}
    gpos = {
        "status": (
            "PASS" if len(reproduced) >= 6 and len(categories) >= 3 else "FAIL"
        ),
        "criterion": "6/8 joint passages across at least 3/4 languages",
        "n_reproduced": len(reproduced),
        "n_passages": len(rows),
        "categories_reproduced": sorted(categories),
        "token_family_manifest": token_families,
        "token_family_manifest_sha256": family_sha256,
        "metric_frozen_before_interventions": True,
        "attribution_role": "SECONDARY_DIAGNOSTIC_NOT_GATE",
        "weight_read_role": "PRIMARY_SELECTION_CONDITIONED",
        "rows": rows,
    }
    return firing, gpos


def _gram_matched_random_pair(
    source_reference: Mapping[int, torch.Tensor],
    target_reference: Mapping[int, torch.Tensor],
    *,
    item_name: str,
    draw_index: int,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, int],
    dict[int, int],
]:
    """Create a random unit pair preserving the real cosine at every layer."""

    random_source, source_seeds = seeded_random_direction_bank(
        source_reference,
        item_name=f"{item_name}:source",
        draw_index=draw_index,
        seed=SEED,
    )
    proposal, target_seeds = seeded_random_direction_bank(
        source_reference,
        item_name=f"{item_name}:target",
        draw_index=draw_index,
        seed=SEED,
    )
    random_target: dict[int, torch.Tensor] = {}
    for layer in sorted(source_reference):
        real_source = source_reference[layer].detach().float()
        real_target = target_reference[layer].detach().float()
        rho = float(torch.dot(real_source, real_target).clamp(-1.0, 1.0))
        first = random_source[layer].detach().float()
        orthogonal = proposal[layer].detach().float() - torch.dot(
            proposal[layer].detach().float(), first
        ) * first
        norm = orthogonal.norm()
        if not torch.isfinite(norm) or float(norm) <= 1e-8:
            raise RuntimeError("Unable to construct Gram-matched random pair")
        orthogonal = orthogonal / norm
        second = rho * first + math.sqrt(max(0.0, 1.0 - rho**2)) * orthogonal
        random_target[layer] = torch.nn.functional.normalize(second, dim=0).to(
            source_reference[layer].device
        )
        if not math.isclose(
            float(torch.dot(random_source[layer].float(), random_target[layer].float())),
            rho,
            abs_tol=2e-5,
        ):
            raise RuntimeError("Random null failed to preserve pair cosine")
    return random_source, random_target, source_seeds, target_seeds


def _random_pair_null(
    bundle: ModelBundle,
    items: list[dict[str, Any]],
    bank: dict[int, dict[int, torch.Tensor]],
    layers: list[int],
    real_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    real = {row["name"]: float(row["delta"]) for row in real_rows}
    rows: list[dict[str, Any]] = []
    for item in items:
        input_ids = bundle.lens_model.encode(item["prompt"])
        clean = forward_logits(bundle.hf_model, input_ids)
        clean_metric = _pair_metric(
            clean,
            item["clean_answer_token_id"],
            item["counterfactual_answer_token_id"],
        )
        residuals = capture_residuals(bundle.lens_model, input_ids, layers)
        source_reference = {
            layer: bank[item["source_concept_token_id"]][layer] for layer in layers
        }
        target_reference = {
            layer: bank[item["target_concept_token_id"]][layer] for layer in layers
        }
        draws: list[dict[str, Any]] = []
        for draw in range(RANDOM_DRAWS):
            source, target, source_seeds, target_seeds = _gram_matched_random_pair(
                source_reference,
                target_reference,
                item_name=item["name"],
                draw_index=draw,
            )
            edited = forward_logits(
                bundle.hf_model,
                input_ids,
                blocks=bundle.lens_model.layers,
                edits=clamped_swap_edits(
                    residuals, source, target, strength=2.0
                ),
            )
            metric = _pair_metric(
                edited,
                item["clean_answer_token_id"],
                item["counterfactual_answer_token_id"],
            )
            draws.append(
                {
                    "draw": draw,
                    "delta": metric - clean_metric,
                    "source_seeds": {str(k): v for k, v in source_seeds.items()},
                    "target_seeds": {str(k): v for k, v in target_seeds.items()},
                }
            )
        absolute = np.abs([row["delta"] for row in draws])
        threshold = float(np.quantile(absolute, 0.975))
        empirical_p = float(
            (1 + int(np.sum(absolute >= abs(real[item["name"]]))))
            / (RANDOM_DRAWS + 1)
        )
        rows.append(
            {
                "name": item["name"],
                "real_delta": real[item["name"]],
                "null_abs_q975": threshold,
                "empirical_two_sided_p": empirical_p,
                "specific_at_p_le_0.05": empirical_p <= 0.05,
                "draws": draws,
            }
        )
    return {
        "status": (
            "PASS" if all(row["specific_at_p_le_0.05"] for row in rows) else "FAIL"
        ),
        "n_draws_per_item": RANDOM_DRAWS,
        "null_geometry": "unit norm and per-layer source/target cosine matched",
        "rows": rows,
    }


def _absent_pair_null(
    bundle: ModelBundle,
    lens: Any,
    items: list[dict[str, Any]],
    bank: dict[int, dict[int, torch.Tensor]],
    absent_ids: list[int],
    layers: list[int],
    real_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    real = {row["name"]: float(row["delta"]) for row in real_rows}
    rows: list[dict[str, Any]] = []
    for item in items:
        input_ids = bundle.lens_model.encode(item["prompt"])
        clean = forward_logits(bundle.hf_model, input_ids)
        clean_metric = _pair_metric(
            clean,
            item["clean_answer_token_id"],
            item["counterfactual_answer_token_id"],
        )
        readout, _, _ = lens.apply(
            bundle.lens_model, item["prompt"], layers=layers, positions=None
        )
        ranks = {
            token_id: min(
                token_rank(readout[layer][position], token_id)
                for layer in layers
                for position in range(readout[layer].shape[0])
            )
            for token_id in absent_ids
        }
        selected = [token_id for token_id in absent_ids if ranks[token_id] >= 1000][:2]
        if len(selected) < 2:
            rows.append(
                {
                    "name": item["name"],
                    "status": "RANK_INFEASIBLE",
                    "ranks": {str(k): v for k, v in ranks.items()},
                }
            )
            continue
        residuals = capture_residuals(bundle.lens_model, input_ids, layers)
        source = {layer: bank[selected[0]][layer] for layer in layers}
        target = {layer: bank[selected[1]][layer] for layer in layers}
        edited = forward_logits(
            bundle.hf_model,
            input_ids,
            blocks=bundle.lens_model.layers,
            edits=clamped_swap_edits(residuals, source, target, strength=2.0),
        )
        metric = _pair_metric(
            edited,
            item["clean_answer_token_id"],
            item["counterfactual_answer_token_id"],
        )
        rows.append(
            {
                "name": item["name"],
                "status": "OK",
                "selected_token_ids": selected,
                "selected_surfaces": [
                    bundle.tokenizer.decode([token_id]) for token_id in selected
                ],
                "ranks": {str(k): v for k, v in ranks.items()},
                "delta": metric - clean_metric,
                "real_delta": real[item["name"]],
            }
        )
    eligible = [row for row in rows if row["status"] == "OK"]
    for row in eligible:
        row["abs_null_over_real_ratio"] = abs(row["delta"]) / max(
            abs(row["real_delta"]), 1e-12
        )
    ratio = (
        float(np.median([abs(row["delta"]) for row in eligible]))
        / max(float(np.median([abs(row["real_delta"]) for row in eligible])), 1e-12)
        if eligible
        else None
    )
    passed = len(eligible) == len(items) and all(
        row["abs_null_over_real_ratio"] < 0.25 for row in eligible
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "n_eligible": len(eligible),
        "median_abs_null_over_real_ratio": ratio,
        "criterion": "all three cases eligible and each |null|/|real| < 0.25",
        "rows": rows,
    }


def _capability_control(
    bundle: ModelBundle,
    items: list[dict[str, Any]],
    bank: dict[int, dict[int, torch.Tensor]],
    layers: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        source = {
            layer: bank[item["source_concept_token_id"]][layer] for layer in layers
        }
        target = {
            layer: bank[item["target_concept_token_id"]][layer] for layer in layers
        }
        for text in CAPABILITY_TEXTS_V1:
            input_ids = bundle.lens_model.encode(text["text"])
            clean = forward_logits(bundle.hf_model, input_ids)
            residuals = capture_residuals(bundle.lens_model, input_ids, layers)
            edited = forward_logits(
                bundle.hf_model,
                input_ids,
                blocks=bundle.lens_model.layers,
                edits=clamped_swap_edits(
                    residuals, source, target, strength=2.0
                ),
            )
            clean_nll = float(teacher_forced_nll(clean, input_ids)[0].cpu())
            edited_nll = float(teacher_forced_nll(edited, input_ids)[0].cpu())
            rows.append(
                {
                    "intervention": item["name"],
                    "text_id": text["id"],
                    "clean_nll": clean_nll,
                    "edited_nll": edited_nll,
                    "delta_nll": edited_nll - clean_nll,
                }
            )
    values = [row["delta_nll"] for row in rows]
    mean_delta = float(np.mean(values))
    per_intervention = []
    for item in items:
        item_values = [
            row["delta_nll"]
            for row in rows
            if row["intervention"] == item["name"]
        ]
        per_intervention.append(
            {
                "intervention": item["name"],
                "mean_delta_nll": float(np.mean(item_values)),
                "mean_abs_delta_nll": float(np.mean(np.abs(item_values))),
                "max_abs_delta_nll": float(np.max(np.abs(item_values))),
                "n": len(item_values),
            }
        )
    finite = all(math.isfinite(value) for value in values)
    preserved = finite and abs(mean_delta) < 0.25 and all(
        abs(row["mean_delta_nll"]) < 0.25 for row in per_intervention
    )
    return {
        "status": "PASS" if preserved else "FAIL",
        "mean_delta_nll": mean_delta,
        "mean_abs_delta_nll": float(np.mean(np.abs(values))),
        "n": len(rows),
        "threshold": 0.25,
        "criterion": (
            "absolute grand mean and every intervention-bank mean delta NLL < 0.25"
        ),
        "per_intervention": per_intervention,
        "rows": rows,
    }


def _identity_j_baseline(
    bundle: ModelBundle,
    items: list[dict[str, Any]],
    layers: list[int],
) -> dict[str, Any]:
    weight = unembedding_weight(bundle.lens_model)
    rows: list[dict[str, Any]] = []
    for item in items:
        source_vector = torch.nn.functional.normalize(
            weight[item["source_concept_token_id"]].detach().float(), dim=0
        ).to(next(bundle.hf_model.parameters()).device)
        target_vector = torch.nn.functional.normalize(
            weight[item["target_concept_token_id"]].detach().float(), dim=0
        ).to(next(bundle.hf_model.parameters()).device)
        source = {layer: source_vector for layer in layers}
        target = {layer: target_vector for layer in layers}
        clean, edited, _ = _canonical_swap(bundle, item, source, target)
        clean_metric = _pair_metric(
            clean,
            item["clean_answer_token_id"],
            item["counterfactual_answer_token_id"],
        )
        edited_metric = _pair_metric(
            edited,
            item["clean_answer_token_id"],
            item["counterfactual_answer_token_id"],
        )
        rows.append(
            {
                "name": item["name"],
                "clean_metric": clean_metric,
                "edited_metric": edited_metric,
                "delta": edited_metric - clean_metric,
                "edited_top": bundle.tokenizer.decode([int(edited[0, -1].argmax())]),
                "counterfactual_top1": (
                    int(edited[0, -1].argmax())
                    == item["counterfactual_answer_token_id"]
                ),
            }
        )
    return {
        "status": "COMPUTED_DIAGNOSTIC",
        "n_counterfactual_top1": sum(row["counterfactual_top1"] for row in rows),
        "n_cases": len(rows),
        "interpretation": (
            "Identity-J is diagnostic only; a canonical spider flip is not by "
            "itself specific to the fitted Jacobian directions."
        ),
        "rows": rows,
    }


def _plot_f3(direct: dict[str, Any]) -> str:
    rows = direct["rows"]
    set_style()
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    x = [row["ablation_delta"] for row in rows]
    y = [row["suppression_delta"] for row in rows]
    axis.scatter(x, y, s=70)
    for row in rows:
        axis.annotate(row["name"], (row["ablation_delta"], row["suppression_delta"]))
    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(0, color="black", linewidth=1)
    axis.set(
        xlabel="internal source-direction ablation delta",
        ylabel="firing output-suppression delta",
        title="F3 — Internal intervention vs a control that can fire",
    )
    path = ROOT / "results" / "figures" / "f3_firing_suppression_control.png"
    save_figure(figure, path)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _stage2_gate_criteria(
    *,
    gswap: Mapping[str, Any],
    controls_fire: Mapping[str, Any],
    random_null: Mapping[str, Any],
    absent: Mapping[str, Any],
    capability: Mapping[str, Any],
    gpos: Mapping[str, Any],
) -> dict[str, bool]:
    """Return the complete measured Stage-2 hard-gate vector."""

    return {
        "g_swap_reverified": gswap["status"] == "PASS",
        "controls_fire": controls_fire["status"] == "PASS",
        "random_pair_specific": random_null["status"] == "PASS",
        "absent_coordinate_specific": absent["status"] == "PASS",
        "capability_preserved": capability["status"] == "PASS",
        "g_pos_reproduced": gpos["status"] == "PASS",
    }


def run_stage2(
    bundle: ModelBundle,
    lens: Any,
    *,
    workspace_layers: list[int],
) -> dict[str, Any]:
    items = load_calibration_items(bundle.tokenizer)
    narration = load_known_narration_source()
    token_families = _single_token_families(bundle.tokenizer)
    language_labels = sorted({row["category"] for row in narration["passages"]})
    language_direction_ids: dict[str, int] = {}
    english_ids: set[int] = set()
    for label in language_labels:
        label_ids: set[int] = set()
        for passage in narration["passages"]:
            if passage["category"] != label:
                continue
            prompt = _direct_language_prompt(passage["text"])
            label_ids.add(continuation_token_id(bundle.tokenizer, prompt, label)[0])
            english_ids.add(
                continuation_token_id(bundle.tokenizer, prompt, "English")[0]
            )
        if len(label_ids) != 1:
            raise RuntimeError(f"Direct {label} output coordinate is not stable")
        language_direction_ids[label] = label_ids.pop()
    if len(english_ids) != 1:
        raise RuntimeError("Direct English output coordinate is not stable")
    english_direction_id = english_ids.pop()
    absent_ids = [
        int(bundle.tokenizer.encode(surface, add_special_tokens=False)[0])
        for surface in ABSENT_CONCEPT_SURFACES_V1
        if len(bundle.tokenizer.encode(surface, add_special_tokens=False)) == 1
    ]
    token_ids = {
        token_id
        for item in items
        for token_id in (
            item["source_concept_token_id"],
            item["target_concept_token_id"],
        )
    } | set(language_direction_ids.values()) | {english_direction_id} | set(absent_ids)
    read_layers = list(
        range(min(workspace_layers), len(bundle.lens_model.layers) - 1)
    )
    bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        read_layers,
        fold_rms_gain=False,
    )
    final_layer = len(bundle.lens_model.layers) - 1
    unembedding = unembedding_weight(bundle.lens_model)
    model_device = next(bundle.hf_model.parameters()).device
    for token_id in token_ids:
        bank[token_id][final_layer] = torch.nn.functional.normalize(
            unembedding[token_id].detach().float(), dim=0
        ).to(model_device)
    gswap = _reverify_gswap(bundle, items, bank, workspace_layers)
    direct = _direct_concept_controls(bundle, items, bank, workspace_layers)
    language_firing, gpos = _language_controls_and_gpos(
        bundle,
        lens,
        narration,
        bank,
        language_direction_ids,
        english_direction_id,
        workspace_layers,
        token_families,
    )
    controls_fire = {
        "status": (
            "PASS"
            if direct["status"] == "PASS" and language_firing["status"] == "PASS"
            else "FAIL"
        ),
        "direct_concept_probes": direct,
        "language_controls": language_firing,
    }
    random_null = _random_pair_null(
        bundle, items, bank, workspace_layers, gswap["rows"]
    )
    absent = _absent_pair_null(
        bundle,
        lens,
        items,
        bank,
        absent_ids,
        workspace_layers,
        gswap["rows"],
    )
    capability = _capability_control(bundle, items, bank, workspace_layers)
    identity = _identity_j_baseline(bundle, items, workspace_layers)
    criteria = _stage2_gate_criteria(
        gswap=gswap,
        controls_fire=controls_fire,
        random_null=random_null,
        absent=absent,
        capability=capability,
        gpos=gpos,
    )
    allowed = all(criteria.values())
    summary: dict[str, Any] = {
        "status": "PASS" if allowed else "FAIL",
        "workspace_layers": workspace_layers,
        "configuration": (
            "exact-label raw J-Lens directions for G-SWAP; leading-label raw "
            "directions for language controls; alpha=2; original prompt positions"
        ),
        "criteria": criteria,
        "g_swap_reverification": gswap,
        "controls_fire": controls_fire,
        "random_pair_null": random_null,
        "absent_coordinate_null": absent,
        "capability": capability,
        "identity_j_baseline": identity,
        "g_pos": gpos,
        "stage3_allowed": allowed,
        "stage4_required": not allowed,
        "limitations": [
            "Language READ is selection-conditioned because components are flagged by task attribution.",
            "The centered suppression sham averages two individually firing post-logit arms.",
            "Identity-J is a diagnostic baseline, not a calibrated causal instrument.",
        ],
    }
    summary["figure_f3"] = _plot_f3(direct)
    raw_path = ROOT / "data" / "raw" / "04_recalibration_v2.json"
    save_json(raw_path, summary)
    summary["raw_artifact"] = str(raw_path.relative_to(ROOT))
    return summary


def _report_section(stage2: dict[str, Any]) -> str:
    gpos = stage2["g_pos"]

    def display(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.3f}"

    rows = "\n".join(
        "| {key} | {category} | {rank} | {clean:.3f} | {delta:.3f} | "
        "{direct:.3f} | {wr} | {ar} | {joint} |".format(
            key=row["key"],
            category=row["category"],
            rank=row["minimum_all_prompt_language_rank"],
            clean=row["automatic_clean_metric"],
            delta=row["automatic_internal_delta"],
            direct=row["direct_internal_delta"],
            wr=display(row["primary_weight_read_ratio_auto_over_direct"]),
            ar=display(
                row["attribution_read_ratio_auto_over_direct_secondary"]
            ),
            joint="PASS" if row["joint_reproduction"] else "FAIL",
        )
        for row in gpos["rows"]
    )
    return f"""

## Stage 2 — recalibration, firing controls, and G-POS

- G-SWAP re-verification: **{stage2['g_swap_reverification']['status']}**.
- Controls that can fire: **{stage2['controls_fire']['status']}**. Direct
  concept/language-answer suppression and both continuation arms each changed a
  metric containing the suppressed token; they are not structural zeros.
- Matched random-pair null: **{stage2['random_pair_null']['status']}** ({stage2['random_pair_null']['n_draws_per_item']} draws per case).
- Absent-coordinate null: **{stage2['absent_coordinate_null']['status']}**; eligible={stage2['absent_coordinate_null']['n_eligible']}; median |null|/|real|={stage2['absent_coordinate_null']['median_abs_null_over_real_ratio']}.
- Capability: **{stage2['capability']['status']}**; mean delta NLL={stage2['capability']['mean_delta_nll']:.3f}, N={stage2['capability']['n']}.
- Identity-J diagnostic: {stage2['identity_j_baseline']['n_counterfactual_top1']}/{stage2['identity_j_baseline']['n_cases']} counterfactual top-1 flips, versus {sum(row['pass'] for row in stage2['g_swap_reverification']['rows'])}/{len(stage2['g_swap_reverification']['rows'])} for J-Lens.

![F3 firing suppression control](figures/f3_firing_suppression_control.png)

### Known-narration positive control

The v2 metric was frozen before intervention outcomes: a clean greedy 16-token
rollout is teacher-forced, and normalized source-language versus English token
family mass is scored across all 16 predictions. The aligned leading-space
language-label coordinate is used for WRITE, swap, direct classification, and
suppression. Attribution is secondary and is not a G-POS gate.

| passage | language | min all-prompt WRITE rank | clean continuation M | internal delta | direct-task delta | primary weight READ ratio | attribution ratio (secondary) | joint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{rows}

**G-POS {gpos['status']}: {gpos['n_reproduced']}/{gpos['n_passages']} passages,
languages={gpos['categories_reproduced']}.** The preregistered threshold is
at least 6/8 passages spanning at least 3/4 languages.

### Stage-2 decision

**{stage2['status']}**. {'Stage 3 is licensed.' if stage2['stage3_allowed'] else 'Stage 3 is blocked; the workflow switches to the Stage-4 replication-failure report. This is not a verdict on the hypothesis.'}
"""


def persist_stage2(stage2: dict[str, Any]) -> dict[str, Any]:
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    repair = metrics["repair_v2"]
    if repair["gate_ledger"].get("g_swap") != "PASS":
        raise RuntimeError("Stage 2 requires G-SWAP PASS")
    if repair["gate_ledger"].get("g_dir") not in {"PASS", "DROPPED_MD"}:
        raise RuntimeError("Stage 2 requires G-DIR PASS or an explicit MD drop")
    if repair["gate_ledger"].get("read_validation") != "PASS":
        raise RuntimeError("Stage 2 requires repaired READ validation")
    repair["stage2_recalibration"] = stage2
    repair["gate_ledger"]["g_swap_reverify"] = stage2["g_swap_reverification"][
        "status"
    ]
    repair["gate_ledger"]["controls_fire"] = stage2["controls_fire"]["status"]
    repair["gate_ledger"]["g_pos"] = stage2["g_pos"]["status"]
    repair["gate_ledger"]["random_pair_null"] = stage2["random_pair_null"][
        "status"
    ]
    repair["gate_ledger"]["absent_coordinate_null"] = stage2[
        "absent_coordinate_null"
    ]["status"]
    repair["gate_ledger"]["capability"] = stage2["capability"]["status"]
    repair["gate_ledger"]["stage3_science"] = (
        "ALLOWED" if stage2["stage3_allowed"] else "SKIPPED_PREREQUISITE"
    )
    repair["current_allowed_conclusion"] = (
        "CALIBRATED_SCIENCE_ALLOWED"
        if stage2["stage3_allowed"]
        else "STAGE4_REPLICATION_FAILURE_NO_HYPOTHESIS_INFERENCE"
    )
    repair["gate_ledger"]["stage4_report"] = (
        "NOT_REQUIRED" if stage2["stage3_allowed"] else "REQUIRED"
    )
    save_json(metrics_path, metrics)
    report_path = ROOT / "results" / "RESULTS.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## Stage 2 — recalibration, firing controls, and G-POS"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    report_path.write_text(report.rstrip() + _report_section(stage2), encoding="utf-8")
    return metrics

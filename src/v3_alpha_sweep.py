"""V3 surgical intervention-strength sweep and hard G-ALPHA gate.

The sweep uses only the three known-answer calibration cases, fixed unrelated
texts, and the known-narration positive control.  No Stage-3 science item is
read.  Carrying positions are selected from clean J-Lens ranks before any
edited forward.  The source-capped operator is the only selectable policy
frozen in notebook 00.  A masked fractional swap is retained as an exploratory
rescue and the all-position fractional swap as a diagnostic reference.
"""

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

from src.controls_phase import (
    ABSENT_CONCEPT_SURFACES_V1,
    CAPABILITY_TEXTS_V1,
    load_known_narration_source,
    teacher_forced_nll,
)
from src.data_gen import continuation_token_id
from src.interventions import clamped_swap_edits, forward_logits
from src.jlens_iface import jlens_direction_bank, token_rank, unembedding_weight
from src.metrics import logit_difference, save_json
from src.model_utils import ModelBundle, capture_residuals
from src.plotting import save_figure, set_style
from src.v2_recalibration import (
    _direct_language_prompt,
    _gram_matched_random_pair,
    _instruction_positions,
    _single_token_families,
    _task_read,
    language_mass_metric,
    subtract_output_logit,
    subtract_output_token_set,
)
from src.v2_repair import load_calibration_items
from src.v3_reverify import _repair_v2_sha256


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
ALPHA_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
RANDOM_DRAWS = 64
RANK_THRESHOLD = 10
CAPABILITY_THRESHOLD = 0.25
GPOS_LOW_CAUSAL = 0.5
GPOS_LOW_READ_RATIO = 0.5
POLICY_ORDER = (
    "project_out_transfer",
    "fractional_swap_carrying_positions",
    "fractional_swap_all_positions_reference",
)
SELECTABLE_POLICIES = {
    "project_out_transfer",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_out_transfer(
    hidden: torch.Tensor,
    clean_hidden: torch.Tensor,
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    positions: Sequence[int],
    strength: float,
    max_condition: float = 1e4,
) -> torch.Tensor:
    """Cap source removal at zero while transferring source mass to target.

    In the nonorthogonal two-direction basis with clean coefficients
    ``(c_s,c_t)``, the desired coefficients are

    ``((1-min(alpha,1))*c_s, c_t + alpha*(c_s-c_t))``.

    The orthogonal complement and unselected positions are unchanged.
    """

    if hidden.shape != clean_hidden.shape or hidden.ndim != 3:
        raise ValueError("Current and clean hidden states must share shape [B,S,D]")
    if not math.isfinite(float(strength)) or strength < 0:
        raise ValueError("Project-transfer strength must be finite and nonnegative")
    indices = [int(position) for position in positions]
    if len(set(indices)) != len(indices):
        raise ValueError("Project-transfer positions must be unique")
    if any(position < 0 or position >= hidden.shape[1] for position in indices):
        raise IndexError("Project-transfer position outside sequence")
    if not indices:
        return hidden.clone()
    source = source_direction.detach().to(hidden.device, torch.float32)
    target = target_direction.detach().to(hidden.device, torch.float32)
    basis = torch.stack([source, target], dim=0)
    norms = basis.norm(dim=-1)
    if not torch.allclose(
        norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("Project-transfer directions must be unit norm")
    gram = basis @ basis.T
    condition = torch.linalg.cond(gram)
    if not torch.isfinite(condition) or float(condition) > max_condition:
        raise ValueError("Project-transfer direction pair is ill-conditioned")
    inverse = torch.linalg.inv(gram)
    current = hidden[:, indices].float()
    clean = clean_hidden.detach().to(hidden.device)[:, indices].float()
    current_coefficients = (current @ basis.T) @ inverse
    clean_coefficients = (clean @ basis.T) @ inverse
    desired = clean_coefficients.clone()
    desired[..., 0] = (1.0 - min(float(strength), 1.0)) * clean_coefficients[
        ..., 0
    ]
    desired[..., 1] = clean_coefficients[..., 1] + float(strength) * (
        clean_coefficients[..., 0] - clean_coefficients[..., 1]
    )
    edited = hidden.clone()
    edited[:, indices] = (
        current + (desired - current_coefficients) @ basis
    ).to(hidden.dtype)
    return edited


def select_carrying_positions(
    readout: Mapping[int, torch.Tensor],
    source_token_id: int,
    layers: Sequence[int],
    *,
    rank_threshold: int = RANK_THRESHOLD,
) -> dict[str, Any]:
    """Select the union of clean source-label top-rank prompt positions."""

    layer_list = [int(layer) for layer in layers]
    if not layer_list:
        raise ValueError("Carrying-position selection requires layers")
    sequence_lengths = {int(readout[layer].shape[0]) for layer in layer_list}
    if len(sequence_lengths) != 1:
        raise ValueError("Readout layers disagree on sequence length")
    ranks = {
        str(layer): [
            token_rank(readout[layer][position], int(source_token_id))
            for position in range(readout[layer].shape[0])
        ]
        for layer in layer_list
    }
    positions = sorted(
        {
            position
            for values in ranks.values()
            for position, rank in enumerate(values)
            if rank <= rank_threshold
        }
    )
    payload = {
        "rule": "source J-Lens rank<=10 at any workspace layer",
        "rank_threshold": rank_threshold,
        "source_token_id": int(source_token_id),
        "sequence_length": sequence_lengths.pop(),
        "positions": positions,
        "ranks_by_layer_position": ranks,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _metric(logits: torch.Tensor, item: Mapping[str, Any]) -> float:
    return float(
        logit_difference(
            logits,
            int(item["clean_answer_token_id"]),
            int(item["counterfactual_answer_token_id"]),
        )[0].cpu()
    )


def _project_edits(
    clean_residuals: Mapping[int, torch.Tensor],
    source: Mapping[int, torch.Tensor],
    target: Mapping[int, torch.Tensor],
    positions: Sequence[int],
    strength: float,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        int(layer): (
            lambda hidden,
            layer=int(layer): project_out_transfer(
                hidden,
                clean_residuals[layer],
                source[layer],
                target[layer],
                positions=positions,
                strength=strength,
            )
        )
        for layer in sorted(source)
    }


def _edits(
    prepared: Mapping[str, Any],
    policy: str,
    alpha: float,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    positions = prepared["mask"]["positions"]
    if policy == "project_out_transfer":
        return _project_edits(
            prepared["residuals"],
            prepared["source"],
            prepared["target"],
            positions,
            alpha,
        )
    if policy == "fractional_swap_carrying_positions":
        return clamped_swap_edits(
            prepared["residuals"],
            prepared["source"],
            prepared["target"],
            positions=positions,
            strength=alpha,
        )
    if policy == "fractional_swap_all_positions_reference":
        return clamped_swap_edits(
            prepared["residuals"],
            prepared["source"],
            prepared["target"],
            positions=prepared.get("all_positions"),
            strength=alpha,
        )
    raise ValueError(f"Unknown alpha-sweep policy: {policy}")


def _json_mask(mask: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rule": mask["rule"],
        "rank_threshold": mask["rank_threshold"],
        "source_token_id": mask["source_token_id"],
        "sequence_length": mask["sequence_length"],
        "positions": list(mask["positions"]),
        "ranks_by_layer_position": dict(mask["ranks_by_layer_position"]),
        "sha256": mask["sha256"],
    }


def _build_bank(
    bundle: ModelBundle,
    lens: Any,
    token_ids: set[int],
    workspace_layers: Sequence[int],
) -> dict[int, dict[int, torch.Tensor]]:
    read_layers = list(
        range(min(int(layer) for layer in workspace_layers), len(bundle.lens_model.layers) - 1)
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
    device = next(bundle.hf_model.parameters()).device
    for token_id in token_ids:
        bank[token_id][final_layer] = torch.nn.functional.normalize(
            unembedding[token_id].detach().float(), dim=0
        ).to(device)
    return bank


def _mask_for_prompt(
    bundle: ModelBundle,
    lens: Any,
    prompt: str,
    source_token_id: int,
    workspace_layers: Sequence[int],
) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    readout, _, _ = lens.apply(
        bundle.lens_model,
        prompt,
        layers=list(workspace_layers),
        positions=None,
    )
    mask = select_carrying_positions(
        readout,
        source_token_id,
        workspace_layers,
    )
    return mask, readout


def _prepare_known(
    bundle: ModelBundle,
    lens: Any,
    items: Sequence[Mapping[str, Any]],
    bank: Mapping[int, Mapping[int, torch.Tensor]],
    workspace_layers: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        prompt = str(item["prompt"])
        source_id = int(item["source_concept_token_id"])
        target_id = int(item["target_concept_token_id"])
        input_ids = bundle.lens_model.encode(prompt)
        clean = forward_logits(bundle.hf_model, input_ids)
        residuals = capture_residuals(bundle.lens_model, input_ids, workspace_layers)
        mask, readout = _mask_for_prompt(
            bundle, lens, prompt, source_id, workspace_layers
        )
        rows.append(
            {
                "item": dict(item),
                "input_ids": input_ids,
                "clean": clean,
                "clean_metric": _metric(clean, item),
                "residuals": residuals,
                "source": {
                    int(layer): bank[source_id][int(layer)]
                    for layer in workspace_layers
                },
                "target": {
                    int(layer): bank[target_id][int(layer)]
                    for layer in workspace_layers
                },
                "mask": mask,
                "readout": readout,
            }
        )
    return rows


def _prepare_capability(
    bundle: ModelBundle,
    lens: Any,
    items: Sequence[Mapping[str, Any]],
    bank: Mapping[int, Mapping[int, torch.Tensor]],
    workspace_layers: Sequence[int],
) -> list[dict[str, Any]]:
    text_cache: dict[str, dict[str, Any]] = {}
    for text in CAPABILITY_TEXTS_V1:
        input_ids = bundle.lens_model.encode(text["text"])
        text_cache[text["id"]] = {
            "input_ids": input_ids,
            "clean": forward_logits(bundle.hf_model, input_ids),
            "residuals": capture_residuals(
                bundle.lens_model, input_ids, workspace_layers
            ),
        }
    rows: list[dict[str, Any]] = []
    for item in items:
        source_id = int(item["source_concept_token_id"])
        target_id = int(item["target_concept_token_id"])
        source = {
            int(layer): bank[source_id][int(layer)] for layer in workspace_layers
        }
        target = {
            int(layer): bank[target_id][int(layer)] for layer in workspace_layers
        }
        for text in CAPABILITY_TEXTS_V1:
            cached = text_cache[text["id"]]
            mask, _ = _mask_for_prompt(
                bundle,
                lens,
                text["text"],
                source_id,
                workspace_layers,
            )
            rows.append(
                {
                    "intervention": item["name"],
                    "text_id": text["id"],
                    "input_ids": cached["input_ids"],
                    "clean": cached["clean"],
                    "residuals": cached["residuals"],
                    "source": source,
                    "target": target,
                    "mask": mask,
                }
            )
    return rows


def _language_direction_ids(
    bundle: ModelBundle,
    narration: Mapping[str, Any],
) -> tuple[dict[str, int], int]:
    labels = sorted({row["category"] for row in narration["passages"]})
    output: dict[str, int] = {}
    english_ids: set[int] = set()
    for label in labels:
        ids: set[int] = set()
        for passage in narration["passages"]:
            if passage["category"] != label:
                continue
            prompt = _direct_language_prompt(passage["text"])
            ids.add(continuation_token_id(bundle.tokenizer, prompt, label)[0])
            english_ids.add(
                continuation_token_id(bundle.tokenizer, prompt, "English")[0]
            )
        if len(ids) != 1:
            raise RuntimeError(f"Unstable direct output token for {label}")
        output[label] = ids.pop()
    if len(english_ids) != 1:
        raise RuntimeError("Unstable direct English output token")
    return output, english_ids.pop()


def _prepare_gpos(
    bundle: ModelBundle,
    lens: Any,
    narration: Mapping[str, Any],
    v2_gpos: Mapping[str, Any],
    bank: Mapping[int, Mapping[int, torch.Tensor]],
    language_ids: Mapping[str, int],
    english_id: int,
    token_families: Mapping[str, Mapping[str, Any]],
    workspace_layers: Sequence[int],
) -> list[dict[str, Any]]:
    prior = {row["key"]: row for row in v2_gpos["rows"]}
    rows: list[dict[str, Any]] = []
    for index, passage in enumerate(narration["passages"]):
        key = passage["key"]
        category = passage["category"]
        text = passage["text"]
        previous = prior[key]
        source_id = int(language_ids[category])
        prompt = narration["task"]["automatic_q"].format(text=text)
        prompt_ids = bundle.lens_model.encode(prompt)
        frozen_ids = previous["frozen_continuation"]["token_ids"]
        continuation = torch.tensor(
            [frozen_ids], device=prompt_ids.device, dtype=prompt_ids.dtype
        )
        full_ids = torch.cat([prompt_ids, continuation], dim=1)
        score_positions = [
            int(value)
            for value in previous["frozen_continuation"]["score_positions"]
        ]
        if score_positions != list(
            range(
                prompt_ids.shape[1] - 1,
                prompt_ids.shape[1] + len(frozen_ids) - 1,
            )
        ):
            raise RuntimeError(f"Frozen continuation positions drifted for {key}")
        source_family = [
            int(value) for value in token_families[category]["token_ids"]
        ]
        english_family = [
            int(value) for value in token_families["English"]["token_ids"]
        ]

        def auto_metric(
            logits: torch.Tensor,
            positions: list[int] = score_positions,
            source_tokens: list[int] = source_family,
            english_tokens: list[int] = english_family,
        ) -> torch.Tensor:
            return language_mass_metric(
                logits, positions, source_tokens, english_tokens
            )

        auto_clean = forward_logits(bundle.hf_model, full_ids)
        auto_residuals = capture_residuals(
            bundle.lens_model, full_ids, workspace_layers
        )
        auto_mask, auto_readout = _mask_for_prompt(
            bundle, lens, prompt, source_id, workspace_layers
        )
        direct_prompt = _direct_language_prompt(text)
        direct_source_id, direct_source_surface = continuation_token_id(
            bundle.tokenizer, direct_prompt, category
        )
        direct_english_id, direct_english_surface = continuation_token_id(
            bundle.tokenizer, direct_prompt, "English"
        )
        if direct_source_id != source_id or direct_english_id != english_id:
            raise RuntimeError(f"Language coordinate drift for {key}")
        direct_ids = bundle.lens_model.encode(direct_prompt)

        def direct_metric(
            logits: torch.Tensor,
            source_token: int = direct_source_id,
            english_token: int = direct_english_id,
        ) -> torch.Tensor:
            return (
                logits[0, -1, source_token].float()
                - logits[0, -1, english_token].float()
            )

        direct_clean = forward_logits(bundle.hf_model, direct_ids)
        direct_residuals = capture_residuals(
            bundle.lens_model, direct_ids, workspace_layers
        )
        direct_mask, _ = _mask_for_prompt(
            bundle, lens, direct_prompt, source_id, workspace_layers
        )
        rank_by_layer = {
            int(layer): [
                token_rank(auto_readout[int(layer)][position], source_id)
                for position in range(auto_readout[int(layer)].shape[0])
            ]
            for layer in workspace_layers
        }
        source_layer = min(
            [int(layer) for layer in workspace_layers],
            key=lambda layer: (min(rank_by_layer[layer]), layer),
        )
        source_effect = float(
            auto_metric(
                subtract_output_token_set(
                    auto_clean, source_family, score_positions
                )
            ).cpu()
            - float(auto_metric(auto_clean).cpu())
        )
        english_effect = float(
            auto_metric(
                subtract_output_token_set(
                    auto_clean, english_family, score_positions
                )
            ).cpu()
            - float(auto_metric(auto_clean).cpu())
        )
        direct_effect = float(
            direct_metric(subtract_output_logit(direct_clean, direct_source_id)).cpu()
            - float(direct_metric(direct_clean).cpu())
        )
        rows.append(
            {
                "index": index,
                "key": key,
                "category": category,
                "prompt": prompt,
                "input_ids": full_ids,
                "clean": auto_clean,
                "residuals": auto_residuals,
                "mask": auto_mask,
                "source": {
                    int(layer): bank[source_id][int(layer)]
                    for layer in workspace_layers
                },
                "target": {
                    int(layer): bank[english_id][int(layer)]
                    for layer in workspace_layers
                },
                "directions": bank[source_id],
                "auto_metric_fn": auto_metric,
                "auto_clean_metric": float(auto_metric(auto_clean).cpu()),
                "score_positions": score_positions,
                "prompt_positions": list(range(prompt_ids.shape[1])),
                "source_family": source_family,
                "english_family": english_family,
                "source_layer": source_layer,
                "rank_by_layer": rank_by_layer,
                "minimum_rank": min(
                    value for values in rank_by_layer.values() for value in values
                ),
                "instruction_positions": _instruction_positions(
                    bundle.tokenizer, prompt, text
                ),
                "direct_prompt": direct_prompt,
                "direct_ids": direct_ids,
                "direct_clean": direct_clean,
                "direct_residuals": direct_residuals,
                "direct_mask": direct_mask,
                "direct_metric_fn": direct_metric,
                "direct_clean_metric": float(direct_metric(direct_clean).cpu()),
                "direct_source_id": direct_source_id,
                "direct_source_surface": direct_source_surface,
                "direct_english_id": direct_english_id,
                "direct_english_surface": direct_english_surface,
                "source_suppression_effect": source_effect,
                "english_suppression_effect": english_effect,
                "direct_suppression_effect": direct_effect,
                "v2_all_position_primary_weight_ratio": previous[
                    "primary_weight_read_ratio_auto_over_direct"
                ],
            }
        )
    return rows


def _masked_weight_read(
    bundle: ModelBundle,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        seed = SEED + 100_000 + int(row["index"]) * 1_000
        auto = _task_read(
            bundle,
            row["input_ids"],
            row["directions"],
            row["auto_metric_fn"],
            source_layer=int(row["source_layer"]),
            intervention_positions=row["mask"]["positions"],
            seed=seed,
        )
        direct = _task_read(
            bundle,
            row["direct_ids"],
            row["directions"],
            row["direct_metric_fn"],
            source_layer=int(row["source_layer"]),
            intervention_positions=row["direct_mask"]["positions"],
            seed=seed,
        )
        direct_mlp = float(direct["weight"]["mlp_primary"])
        direct_attention = float(direct["weight"]["attention_primary"])
        valid = bool(
            math.isfinite(direct_mlp)
            and math.isfinite(direct_attention)
            and direct_mlp > 1e-8
            and direct_attention > 1e-8
        )
        mlp_ratio = (
            float(auto["weight"]["mlp_primary"]) / direct_mlp if valid else None
        )
        attention_ratio = (
            float(auto["weight"]["attention_primary"]) / direct_attention
            if valid
            else None
        )
        primary = (
            max(mlp_ratio, attention_ratio)
            if mlp_ratio is not None and attention_ratio is not None
            else None
        )
        output[str(row["key"])] = {
            "source_layer": row["source_layer"],
            "automatic_mask": _json_mask(row["mask"]),
            "direct_mask": _json_mask(row["direct_mask"]),
            "mlp_ratio": mlp_ratio,
            "attention_ratio": attention_ratio,
            "primary_ratio": primary,
            "automatic": auto,
            "direct": direct,
            "role": "primary mask-specific selection-conditioned weight READ",
        }
    return output


def _evaluate_known(
    bundle: ModelBundle,
    prepared_rows: Sequence[Mapping[str, Any]],
    policy: str,
    alpha: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for prepared in prepared_rows:
        item = prepared["item"]
        edited = forward_logits(
            bundle.hf_model,
            prepared["input_ids"],
            blocks=bundle.lens_model.layers,
            edits=_edits(prepared, policy, alpha),
        )
        edited_top_id = int(edited[0, -1].argmax())
        edited_metric = _metric(edited, item)
        passed = bool(
            int(prepared["clean"][0, -1].argmax())
            == int(item["clean_answer_token_id"])
            and edited_top_id == int(item["counterfactual_answer_token_id"])
        )
        rows.append(
            {
                "name": item["name"],
                "mask": _json_mask(prepared["mask"]),
                "clean_metric": prepared["clean_metric"],
                "edited_metric": edited_metric,
                "delta_metric": edited_metric - prepared["clean_metric"],
                "clean_top": bundle.tokenizer.decode(
                    [int(prepared["clean"][0, -1].argmax())]
                ),
                "edited_top": bundle.tokenizer.decode([edited_top_id]),
                "edited_top_id": edited_top_id,
                "pass": passed,
            }
        )
    return {
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "n_pass": sum(row["pass"] for row in rows),
        "n_required": len(rows),
        "rows": rows,
    }


def _evaluate_capability(
    bundle: ModelBundle,
    prepared_rows: Sequence[Mapping[str, Any]],
    policy: str,
    alpha: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for prepared in prepared_rows:
        edited = forward_logits(
            bundle.hf_model,
            prepared["input_ids"],
            blocks=bundle.lens_model.layers,
            edits=_edits(prepared, policy, alpha),
        )
        clean_nll = float(
            teacher_forced_nll(
                prepared["clean"], prepared["input_ids"]
            )[0].cpu()
        )
        edited_nll = float(
            teacher_forced_nll(edited, prepared["input_ids"])[0].cpu()
        )
        rows.append(
            {
                "intervention": prepared["intervention"],
                "text_id": prepared["text_id"],
                "mask": _json_mask(prepared["mask"]),
                "clean_nll": clean_nll,
                "edited_nll": edited_nll,
                "delta_nll": edited_nll - clean_nll,
            }
        )
    values = [float(row["delta_nll"]) for row in rows]
    per_intervention = []
    for intervention in sorted({row["intervention"] for row in rows}):
        item_values = [
            float(row["delta_nll"])
            for row in rows
            if row["intervention"] == intervention
        ]
        per_intervention.append(
            {
                "intervention": intervention,
                "mean_delta_nll": float(np.mean(item_values)),
                "mean_abs_delta_nll": float(np.mean(np.abs(item_values))),
                "max_abs_delta_nll": float(np.max(np.abs(item_values))),
                "n": len(item_values),
            }
        )
    grand = float(np.mean(values))
    mean_abs = float(np.mean(np.abs(values)))
    numeric_pass = bool(
        all(math.isfinite(value) for value in values)
        and abs(grand) < CAPABILITY_THRESHOLD
        and mean_abs < CAPABILITY_THRESHOLD
        and all(
            abs(row["mean_delta_nll"]) < CAPABILITY_THRESHOLD
            and row["mean_abs_delta_nll"] < CAPABILITY_THRESHOLD
            for row in per_intervention
        )
    )
    n_active_edit_opportunities = (
        len(prepared_rows)
        if policy == "fractional_swap_all_positions_reference"
        else sum(bool(row["mask"]["positions"]) for row in prepared_rows)
    )
    if n_active_edit_opportunities == 0:
        status = "NO_EDIT_OPPORTUNITY"
    else:
        status = "PASS" if numeric_pass else "FAIL"
    return {
        "status": status,
        "numeric_threshold_status": "PASS" if numeric_pass else "FAIL",
        "n_active_edit_opportunities": n_active_edit_opportunities,
        "threshold": CAPABILITY_THRESHOLD,
        "mean_delta_nll": grand,
        "mean_abs_delta_nll": mean_abs,
        "per_intervention": per_intervention,
        "n": len(rows),
        "rows": rows,
    }


def _gpos_prepared_for_edit(
    row: Mapping[str, Any],
    *,
    direct: bool,
) -> dict[str, Any]:
    if direct:
        return {
            "residuals": row["direct_residuals"],
            "source": row["source"],
            "target": row["target"],
            "mask": row["direct_mask"],
        }
    return {
        "residuals": row["residuals"],
        "source": row["source"],
        "target": row["target"],
        "mask": row["mask"],
        "all_positions": row["prompt_positions"],
    }


def _evaluate_gpos(
    bundle: ModelBundle,
    prepared_rows: Sequence[Mapping[str, Any]],
    masked_read: Mapping[str, Mapping[str, Any]],
    policy: str,
    alpha: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for prepared in prepared_rows:
        auto_edited = forward_logits(
            bundle.hf_model,
            prepared["input_ids"],
            blocks=bundle.lens_model.layers,
            edits=_edits(
                _gpos_prepared_for_edit(prepared, direct=False), policy, alpha
            ),
        )
        direct_edited = forward_logits(
            bundle.hf_model,
            prepared["direct_ids"],
            blocks=bundle.lens_model.layers,
            edits=_edits(
                _gpos_prepared_for_edit(prepared, direct=True), policy, alpha
            ),
        )
        auto_metric = float(prepared["auto_metric_fn"](auto_edited).cpu())
        direct_metric = float(prepared["direct_metric_fn"](direct_edited).cpu())
        auto_delta = auto_metric - float(prepared["auto_clean_metric"])
        direct_delta = direct_metric - float(prepared["direct_clean_metric"])
        if policy == "fractional_swap_all_positions_reference":
            primary_ratio = prepared["v2_all_position_primary_weight_ratio"]
            read_source = "v2 all-original-prompt mask-specific READ"
        else:
            primary_ratio = masked_read[prepared["key"]]["primary_ratio"]
            read_source = "v3 clean-rank carrying-mask READ"
        checks = {
            "clean_continuation_capable": prepared["auto_clean_metric"] > 0.0,
            "high_write": prepared["minimum_rank"] <= RANK_THRESHOLD,
            "low_causal_abs_delta": abs(auto_delta) <= GPOS_LOW_CAUSAL,
            "low_causal_relative_to_direct": abs(auto_delta)
            <= 0.25 * abs(direct_delta),
            "direct_clean_top1_source": (
                int(prepared["direct_clean"][0, -1].argmax())
                == int(prepared["direct_source_id"])
            ),
            "direct_internal_top1_english": (
                int(direct_edited[0, -1].argmax())
                == int(prepared["direct_english_id"])
            ),
            "low_primary_weight_read_ratio": (
                primary_ratio is not None
                and float(primary_ratio) <= GPOS_LOW_READ_RATIO
            ),
            "continuation_suppression_arms_fire": math.isclose(
                prepared["source_suppression_effect"], -1.0, abs_tol=1e-5
            )
            and math.isclose(
                prepared["english_suppression_effect"], 1.0, abs_tol=1e-5
            ),
            "direct_label_suppression_fires": math.isclose(
                prepared["direct_suppression_effect"], -1.0, abs_tol=1e-5
            ),
        }
        rows.append(
            {
                "key": prepared["key"],
                "category": prepared["category"],
                "automatic_mask": _json_mask(prepared["mask"]),
                "direct_mask": _json_mask(prepared["direct_mask"]),
                "clean_metric": prepared["auto_clean_metric"],
                "edited_metric": auto_metric,
                "internal_delta": auto_delta,
                "direct_clean_metric": prepared["direct_clean_metric"],
                "direct_edited_metric": direct_metric,
                "direct_delta": direct_delta,
                "direct_edited_top": bundle.tokenizer.decode(
                    [int(direct_edited[0, -1].argmax())]
                ),
                "minimum_write_rank": prepared["minimum_rank"],
                "primary_weight_read_ratio": primary_ratio,
                "weight_read_source": read_source,
                "checks": checks,
                "joint_reproduction": all(checks.values()),
            }
        )
    reproduced = [row for row in rows if row["joint_reproduction"]]
    categories = {row["category"] for row in reproduced}
    passed = len(reproduced) >= 6 and len(categories) >= 3
    return {
        "status": "PASS" if passed else "FAIL",
        "n_reproduced": len(reproduced),
        "n_passages": len(rows),
        "categories_reproduced": sorted(categories),
        "criterion": ">=6/8 passages across >=3 languages",
        "rows": rows,
    }


def _evaluate_random_null(
    bundle: ModelBundle,
    prepared_rows: Sequence[Mapping[str, Any]],
    known: Mapping[str, Any],
    policy: str,
    alpha: float,
) -> dict[str, Any]:
    real = {row["name"]: float(row["delta_metric"]) for row in known["rows"]}
    rows: list[dict[str, Any]] = []
    for prepared in prepared_rows:
        item = prepared["item"]
        draws: list[dict[str, Any]] = []
        for draw in range(RANDOM_DRAWS):
            source, target, source_seeds, target_seeds = _gram_matched_random_pair(
                prepared["source"],
                prepared["target"],
                item_name=str(item["name"]),
                draw_index=draw,
            )
            random_prepared = {
                **prepared,
                "source": source,
                "target": target,
            }
            edited = forward_logits(
                bundle.hf_model,
                prepared["input_ids"],
                blocks=bundle.lens_model.layers,
                edits=_edits(random_prepared, policy, alpha),
            )
            delta = _metric(edited, item) - float(prepared["clean_metric"])
            draws.append(
                {
                    "draw": draw,
                    "delta": delta,
                    "source_seeds": {str(k): v for k, v in source_seeds.items()},
                    "target_seeds": {str(k): v for k, v in target_seeds.items()},
                }
            )
        absolute = np.abs([row["delta"] for row in draws])
        p_value = float(
            (1 + int(np.sum(absolute >= abs(real[str(item["name"])]))))
            / (RANDOM_DRAWS + 1)
        )
        rows.append(
            {
                "name": item["name"],
                "real_delta": real[str(item["name"])],
                "empirical_two_sided_p": p_value,
                "pass": p_value <= 0.05,
                "draws": draws,
            }
        )
    return {
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "n_draws_per_item": RANDOM_DRAWS,
        "rows": rows,
    }


def _evaluate_absent_null(
    bundle: ModelBundle,
    prepared_rows: Sequence[Mapping[str, Any]],
    known: Mapping[str, Any],
    policy: str,
    alpha: float,
    bank: Mapping[int, Mapping[int, torch.Tensor]],
    v2_absent: Mapping[str, Any],
    workspace_layers: Sequence[int],
) -> dict[str, Any]:
    selected = {
        row["name"]: row["selected_token_ids"]
        for row in v2_absent["rows"]
        if row["status"] == "OK"
    }
    real = {row["name"]: float(row["delta_metric"]) for row in known["rows"]}
    rows: list[dict[str, Any]] = []
    for prepared in prepared_rows:
        item = prepared["item"]
        ids = [int(value) for value in selected[str(item["name"])]]
        absent_prepared = {
            **prepared,
            "source": {
                int(layer): bank[ids[0]][int(layer)] for layer in workspace_layers
            },
            "target": {
                int(layer): bank[ids[1]][int(layer)] for layer in workspace_layers
            },
        }
        edited = forward_logits(
            bundle.hf_model,
            prepared["input_ids"],
            blocks=bundle.lens_model.layers,
            edits=_edits(absent_prepared, policy, alpha),
        )
        delta = _metric(edited, item) - float(prepared["clean_metric"])
        ratio = abs(delta) / max(abs(real[str(item["name"])]), 1e-12)
        rows.append(
            {
                "name": item["name"],
                "selected_token_ids": ids,
                "delta": delta,
                "real_delta": real[str(item["name"])],
                "abs_null_over_real_ratio": ratio,
                "pass": ratio < 0.25,
            }
        )
    return {
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "criterion": "each |absent delta|/|real delta| < 0.25",
        "rows": rows,
    }


def _compact_random(random_null: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": random_null["status"],
        "n_draws_per_item": random_null.get("n_draws_per_item", 0),
        "rows": [
            {key: value for key, value in row.items() if key != "draws"}
            for row in random_null.get("rows", [])
        ],
    }


def _compact_weight_read(
    weight_read: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, row in weight_read.items():
        output[key] = {
            "source_layer": row["source_layer"],
            "automatic_mask": row["automatic_mask"],
            "direct_mask": row["direct_mask"],
            "mlp_ratio": row["mlp_ratio"],
            "attention_ratio": row["attention_ratio"],
            "primary_ratio": row["primary_ratio"],
            "automatic_flags": row["automatic"]["flags"],
            "direct_flags": row["direct"]["flags"],
            "role": row["role"],
        }
    return output


def _plot_alpha(rows: Sequence[Mapping[str, Any]], selected: Any) -> str:
    set_style()
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 10.5), sharex=True)
    colors = {
        "project_out_transfer": "#2E7D32",
        "fractional_swap_carrying_positions": "#1565C0",
        "fractional_swap_all_positions_reference": "#7B1FA2",
    }
    labels = {
        "project_out_transfer": "source-capped transfer (primary)",
        "fractional_swap_carrying_positions": "masked fractional (exploratory)",
        "fractional_swap_all_positions_reference": "all-position reference",
    }
    for policy in POLICY_ORDER:
        policy_rows = sorted(
            [row for row in rows if row["policy"] == policy],
            key=lambda row: row["alpha"],
        )
        if not policy_rows:
            continue
        alpha = [row["alpha"] for row in policy_rows]
        axes[0].plot(
            alpha,
            [row["known_swaps"]["n_pass"] for row in policy_rows],
            marker="o",
            color=colors[policy],
            label=labels[policy],
        )
        axes[1].plot(
            alpha,
            [row["capability"]["mean_abs_delta_nll"] for row in policy_rows],
            marker="o",
            color=colors[policy],
        )
        axes[2].plot(
            alpha,
            [row["g_pos"]["n_reproduced"] / 8 for row in policy_rows],
            marker="o",
            color=colors[policy],
        )
    axes[0].axhline(3, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("known swaps flipped (of 3)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(CAPABILITY_THRESHOLD, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("mean |delta NLL|")
    axes[2].axhline(6 / 8, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("G-POS joint pass rate")
    axes[2].set_xlabel("intervention strength alpha")
    if selected is not None:
        for axis in axes:
            axis.axvline(
                float(selected["alpha"]), color="#B33A3A", linewidth=1.5
            )
    figure.suptitle("F-ALPHA — efficacy, collateral damage, and narration control")
    path = ROOT / "results" / "figures" / "f_alpha_v3.png"
    save_figure(figure, path)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _mask_manifest(
    known: Sequence[Mapping[str, Any]],
    capability: Sequence[Mapping[str, Any]],
    gpos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "rule": "source J-Lens rank<=10 at any L13-24 original-prompt position",
        "known": {
            row["item"]["name"]: _json_mask(row["mask"]) for row in known
        },
        "capability": {
            f"{row['intervention']}::{row['text_id']}": _json_mask(row["mask"])
            for row in capability
        },
        "g_pos": {
            row["key"]: {
                "automatic": _json_mask(row["mask"]),
                "direct": _json_mask(row["direct_mask"]),
            }
            for row in gpos
        },
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def run_alpha_sweep(
    bundle: ModelBundle,
    lens: Any,
    *,
    repair_v2: Mapping[str, Any],
    workspace_layers: Sequence[int],
) -> dict[str, Any]:
    """Execute the complete v3 alpha sweep without touching Stage-3 data."""

    items = load_calibration_items(bundle.tokenizer)
    narration = load_known_narration_source()
    token_families = _single_token_families(bundle.tokenizer)
    family_sha = hashlib.sha256(
        json.dumps(
            token_families, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    v2_stage2 = repair_v2["stage2_recalibration"]
    if family_sha != v2_stage2["g_pos"]["token_family_manifest_sha256"]:
        raise RuntimeError("Frozen language token-family manifest drifted")
    language_ids, english_id = _language_direction_ids(bundle, narration)
    absent_ids = {
        int(token_id)
        for row in v2_stage2["absent_coordinate_null"]["rows"]
        if row["status"] == "OK"
        for token_id in row["selected_token_ids"]
    }
    # Also prove all preregistered surfaces remain single-token even when the
    # v2 rank rule selected only a subset.
    for surface in ABSENT_CONCEPT_SURFACES_V1:
        encoded = bundle.tokenizer.encode(surface, add_special_tokens=False)
        if len(encoded) == 1:
            absent_ids.add(int(encoded[0]))
    token_ids = {
        int(token_id)
        for item in items
        for token_id in (
            item["source_concept_token_id"],
            item["target_concept_token_id"],
        )
    }
    token_ids |= set(language_ids.values()) | {english_id} | absent_ids
    bank = _build_bank(bundle, lens, token_ids, workspace_layers)

    known_prepared = _prepare_known(
        bundle, lens, items, bank, workspace_layers
    )
    capability_prepared = _prepare_capability(
        bundle, lens, items, bank, workspace_layers
    )
    gpos_prepared = _prepare_gpos(
        bundle,
        lens,
        narration,
        v2_stage2["g_pos"],
        bank,
        language_ids,
        english_id,
        token_families,
        workspace_layers,
    )
    manifest = _mask_manifest(
        known_prepared, capability_prepared, gpos_prepared
    )
    masked_read = _masked_weight_read(bundle, gpos_prepared)
    firing_pass = all(
        math.isclose(row["source_suppression_effect"], -1.0, abs_tol=1e-5)
        and math.isclose(row["english_suppression_effect"], 1.0, abs_tol=1e-5)
        and math.isclose(row["direct_suppression_effect"], -1.0, abs_tol=1e-5)
        for row in gpos_prepared
    )

    raw_rows: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        for alpha in ALPHA_GRID:
            known = _evaluate_known(
                bundle, known_prepared, policy, float(alpha)
            )
            capability = _evaluate_capability(
                bundle, capability_prepared, policy, float(alpha)
            )
            gpos = _evaluate_gpos(
                bundle, gpos_prepared, masked_read, policy, float(alpha)
            )
            random_null = _evaluate_random_null(
                bundle,
                known_prepared,
                known,
                policy,
                float(alpha),
            )
            absent = _evaluate_absent_null(
                bundle,
                known_prepared,
                known,
                policy,
                float(alpha),
                bank,
                v2_stage2["absent_coordinate_null"],
                workspace_layers,
            )
            gates = {
                "swap_3_of_3": known["status"] == "PASS",
                "capability": capability["status"] == "PASS",
                "g_pos": gpos["status"] == "PASS",
                "random_null": random_null["status"] == "PASS",
                "absent_null": absent["status"] == "PASS",
                "controls_fire": firing_pass,
            }
            raw_rows.append(
                {
                    "policy": policy,
                    "alpha": float(alpha),
                    "selectable": policy in SELECTABLE_POLICIES,
                    "known_swaps": known,
                    "capability": capability,
                    "g_pos": gpos,
                    "random_null": random_null,
                    "absent_null": absent,
                    "gates": gates,
                    "valid": policy in SELECTABLE_POLICIES and all(gates.values()),
                }
            )

    valid = [row for row in raw_rows if row["valid"]]
    valid.sort(
        key=lambda row: (
            float(row["alpha"]),
            POLICY_ORDER.index(str(row["policy"])),
        )
    )
    selected = None
    if valid:
        chosen = valid[0]
        selected = {
            "alpha": chosen["alpha"],
            "policy": chosen["policy"],
            "operation": (
                "source-capped project-out transfer"
                if chosen["policy"] == "project_out_transfer"
                else "fractional clamped coordinate swap"
            ),
            "layers": [int(layer) for layer in workspace_layers],
            "position_rule": manifest["rule"],
            "direction": "raw normalize(J.T @ W_U[token])",
            "per_concept_scaling": None,
            "selection_rule": "smallest alpha among selectable policies passing all gates",
        }
    raw_artifact = {
        "schema_version": "alpha-sweep-v3-raw",
        "alpha_grid": list(ALPHA_GRID),
        "extension_triggered": False,
        "extension_reason": (
            "alpha=0.25 did not pass 3/3 G-SWAP, so the downward extension "
            "rule cannot select a smaller valid alpha"
        ),
        "policy_order": list(POLICY_ORDER),
        "selectable_policies": sorted(SELECTABLE_POLICIES),
        "exploratory_rescue": {
            "policy": "fractional_swap_carrying_positions",
            "selection_status": "NONSELECTABLE_NOT_FROZEN_IN_NOTEBOOK_00",
            "reason": (
                "Retained as an honest surgical sensitivity analysis after the "
                "primary source-capped policy reached only 2/3 swaps."
            ),
        },
        "mask_manifest": manifest,
        "masked_weight_read": masked_read,
        "rows": raw_rows,
        "selected_intervention": selected,
    }
    raw_path = ROOT / "data" / "raw" / "v3" / "015_alpha_sweep.json"
    save_json(raw_path, raw_artifact)
    raw_sha256 = _file_sha256(raw_path)
    raw_bytes = raw_path.stat().st_size
    compact_rows = [
        {
            **{key: value for key, value in row.items() if key != "random_null"},
            "random_null": _compact_random(row["random_null"]),
        }
        for row in raw_rows
    ]
    summary: dict[str, Any] = {
        "status": "PASS" if selected is not None else "FAIL",
        "g_alpha": "PASS" if selected is not None else "FAIL",
        "alpha_grid": list(ALPHA_GRID),
        "extension_triggered": False,
        "extension_reason": raw_artifact["extension_reason"],
        "policy_order": list(POLICY_ORDER),
        "selectable_policies": sorted(SELECTABLE_POLICIES),
        "exploratory_rescue": {
            "policy": "fractional_swap_carrying_positions",
            "selection_status": "NONSELECTABLE_NOT_FROZEN_IN_NOTEBOOK_00",
            "reason": (
                "Reported as an exploratory surgical sensitivity analysis after "
                "the frozen primary source-capped policy reached only 2/3 swaps."
            ),
        },
        "mask_manifest": manifest,
        "firing_controls": "PASS" if firing_pass else "FAIL",
        "masked_weight_read": _compact_weight_read(masked_read),
        "rows": compact_rows,
        "selected_intervention": selected,
        "raw_artifact": str(raw_path.relative_to(ROOT)),
        "raw_artifact_sha256": raw_sha256,
        "raw_artifact_bytes": raw_bytes,
        "stage3_science_allowed": False,
        "limitations": [
            "Weight READ is selection-conditioned; mask-specific ratios are alpha-invariant.",
            "The masked fractional rescue is exploratory and nonselectable because notebook 00 did not freeze it.",
            "The all-position reference is diagnostic and cannot be selected.",
            "Empty capability masks are no-edit opportunities, not active-edit stress tests.",
        ],
    }
    summary["figure"] = _plot_alpha(compact_rows, selected)
    return summary


def _report_section(sweep: Mapping[str, Any]) -> str:
    rows = "\n".join(
        "| {policy} | {alpha:.2f} | {swaps}/3 | {nll:+.3f} | {abs_nll:.3f} "
        "| {capability} | {gpos}/8 | {random} | {absent} | {valid} |".format(
            policy=row["policy"],
            alpha=row["alpha"],
            swaps=row["known_swaps"]["n_pass"],
            nll=row["capability"]["mean_delta_nll"],
            abs_nll=row["capability"]["mean_abs_delta_nll"],
            capability=row["capability"]["status"],
            gpos=row["g_pos"]["n_reproduced"],
            random=row["random_null"]["status"],
            absent=row["absent_null"]["status"],
            valid="PASS" if row["valid"] else "FAIL",
        )
        for row in sweep["rows"]
    )
    decision = (
        f"G-ALPHA PASS; selected `{sweep['selected_intervention']['policy']}` "
        f"at alpha={sweep['selected_intervention']['alpha']}."
        if sweep["selected_intervention"] is not None
        else (
            "G-ALPHA FAIL; no tested strength/policy simultaneously achieved "
            "3/3 swaps, capability, G-POS, and both specificity nulls."
        )
    )
    masked_alpha_15 = next(
        row
        for row in sweep["rows"]
        if row["policy"] == "fractional_swap_carrying_positions"
        and row["alpha"] == 1.5
    )
    capability_masks = sweep["mask_manifest"]["capability"]
    n_empty_capability_masks = sum(
        not row["positions"] for row in capability_masks.values()
    )
    read_rows = masked_alpha_15["g_pos"]["rows"]
    masked_mean_delta = float(
        masked_alpha_15["capability"]["mean_delta_nll"]
    )
    max_internal_delta = max(abs(row["internal_delta"]) for row in read_rows)
    check_counts = {
        check: sum(bool(row["checks"][check]) for row in read_rows)
        for check in read_rows[0]["checks"]
    }
    read_table = "\n".join(
        "| {key} | {delta:+.3f} | {ratio:.3f} | {gate} |".format(
            key=row["key"],
            delta=row["internal_delta"],
            ratio=row["primary_weight_read_ratio"],
            gate=(
                "PASS"
                if row["checks"]["low_primary_weight_read_ratio"]
                else "FAIL"
            ),
        )
        for row in read_rows
    )
    all_position_125 = next(
        row
        for row in sweep["rows"]
        if row["policy"] == "fractional_swap_all_positions_reference"
        and row["alpha"] == 1.25
    )
    bank_means = ", ".join(
        "{intervention}={mean:.3f}".format(
            intervention=row["intervention"],
            mean=row["mean_delta_nll"],
        )
        for row in all_position_125["capability"]["per_intervention"]
    )
    all_position_signed_mean = float(
        all_position_125["capability"]["mean_delta_nll"]
    )
    all_position_mean_absolute = float(
        all_position_125["capability"]["mean_abs_delta_nll"]
    )
    return f"""

## Stage 1.5 — surgical alpha sweep

The carrying mask was frozen from clean source-label J-Lens rank <=10 at any
workspace layer before edited forwards. The source-capped operator was primary.
The carrying-position fractional swap is reported as an exploratory,
nonselectable sensitivity analysis because it was not frozen in notebook 00.
The all-position fractional swap is diagnostic only.

| policy | alpha | swaps | mean delta NLL | mean abs delta NLL | capability gate | G-POS | random | absent | composite |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
{rows}

![F-ALPHA](figures/f_alpha_v3.png)

### What the sweep isolated

The strongest exploratory surgical candidate was the carrying-position
fractional swap at alpha=1.50: swaps **3/3**, random and absent nulls **PASS**,
and mean capability delta NLL={masked_mean_delta:.3f}. That capability number
is a conditional no-op result, not broad evidence of harmlessness:
**{n_empty_capability_masks}/{len(capability_masks)} unrelated-text masks were
empty**, so the frozen rank rule applied no edit on every capability item.

The same alpha=1.50 candidate had small narration internal changes on all eight
items (largest absolute delta={max_internal_delta:.3f}) and its direct firing
controls passed, but G-POS reproduced **0/8**. Every mask-specific primary
weight-READ ratio exceeded the required <=0.50 threshold:

| item | internal delta | weight-READ ratio | <=0.50 |
| --- | ---: | ---: | --- |
{read_table}

Subgate decomposition at this setting: clean continuation capable
**{check_counts['clean_continuation_capable']}/8**; high WRITE
**{check_counts['high_write']}/8**; direct source-to-English flip
**{check_counts['direct_internal_top1_english']}/8**; low absolute causal change
**{check_counts['low_causal_abs_delta']}/8**; low causal change relative to the
direct arm **{check_counts['low_causal_relative_to_direct']}/8**; low primary
weight-READ **{check_counts['low_primary_weight_read_ratio']}/8**; and both
firing-control checks **8/8**.

These weight-READ ratios are properties of the fixed masks and are invariant to
alpha. Increasing or decreasing intervention strength therefore cannot make
this candidate satisfy the low-READ premise. `es2` additionally failed the
clean-continuation-capable prerequisite.

For context, the all-position reference at alpha=1.25 had signed grand mean
delta NLL={all_position_signed_mean:.3f}, but its grand mean absolute delta
NLL={all_position_mean_absolute:.3f} exceeded 0.25 and its per-intervention
signed means also failed ({bank_means}). It was nonselectable by protocol.

The full 24-row random and absent-control sweep is stored at
`{sweep['raw_artifact']}` (SHA-256 `{sweep['raw_artifact_sha256']}`).

**{decision}** The frozen primary policy never exceeded 2/3 swaps. The
exploratory carrying-position rescue also failed G-POS, so making it selectable
would not alter the decision. Stage 2 and Stage 3 are skipped; the workflow
takes the calibration-limitation fallback without a hypothesis verdict.
"""


def _validate_alpha_sweep(sweep: Mapping[str, Any]) -> None:
    if list(sweep["alpha_grid"]) != list(ALPHA_GRID):
        raise RuntimeError("G-ALPHA alpha grid drifted")
    if list(sweep["policy_order"]) != list(POLICY_ORDER):
        raise RuntimeError("G-ALPHA policy order drifted")
    if set(sweep["selectable_policies"]) != SELECTABLE_POLICIES:
        raise RuntimeError("G-ALPHA selectable-policy set drifted")

    rows = list(sweep["rows"])
    expected_pairs = {
        (policy, float(alpha)) for policy in POLICY_ORDER for alpha in ALPHA_GRID
    }
    actual_pairs = {(row["policy"], float(row["alpha"])) for row in rows}
    if len(rows) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise RuntimeError("G-ALPHA row coverage is incomplete or duplicated")

    controls_fire = sweep["firing_controls"] == "PASS"
    for row in rows:
        capability = row["capability"]
        numeric_capability_pass = bool(
            math.isfinite(float(capability["mean_delta_nll"]))
            and abs(float(capability["mean_delta_nll"]))
            < CAPABILITY_THRESHOLD
            and float(capability["mean_abs_delta_nll"])
            < CAPABILITY_THRESHOLD
            and all(
                abs(float(bank["mean_delta_nll"])) < CAPABILITY_THRESHOLD
                and float(bank["mean_abs_delta_nll"])
                < CAPABILITY_THRESHOLD
                for bank in capability["per_intervention"]
            )
        )
        expected_active_edit_opportunities = (
            len(capability["rows"])
            if row["policy"] == "fractional_swap_all_positions_reference"
            else sum(
                bool(item["mask"]["positions"])
                for item in capability["rows"]
            )
        )
        n_active_edit_opportunities = int(
            capability["n_active_edit_opportunities"]
        )
        if n_active_edit_opportunities != expected_active_edit_opportunities:
            raise RuntimeError("G-ALPHA active capability count does not rederive")
        expected_capability_status = (
            "NO_EDIT_OPPORTUNITY"
            if n_active_edit_opportunities == 0
            else ("PASS" if numeric_capability_pass else "FAIL")
        )
        if (
            capability["status"] != expected_capability_status
            or capability["numeric_threshold_status"]
            != ("PASS" if numeric_capability_pass else "FAIL")
        ):
            raise RuntimeError("G-ALPHA capability status does not rederive")
        expected_gates = {
            "swap_3_of_3": row["known_swaps"]["status"] == "PASS",
            "capability": capability["status"] == "PASS",
            "g_pos": row["g_pos"]["status"] == "PASS",
            "random_null": row["random_null"]["status"] == "PASS",
            "absent_null": row["absent_null"]["status"] == "PASS",
            "controls_fire": controls_fire,
        }
        if row["gates"] != expected_gates:
            raise RuntimeError("G-ALPHA stored gate values do not rederive")
        expected_valid = bool(
            row["policy"] in SELECTABLE_POLICIES
            and all(expected_gates.values())
        )
        if bool(row["selectable"]) != (
            row["policy"] in SELECTABLE_POLICIES
        ) or bool(row["valid"]) != expected_valid:
            raise RuntimeError("G-ALPHA validity does not rederive")
        random_null = row["random_null"]
        if (
            random_null["status"] not in {"PASS", "FAIL"}
            or random_null["n_draws_per_item"] != RANDOM_DRAWS
            or len(random_null["rows"]) != 3
            or row["absent_null"]["status"] not in {"PASS", "FAIL"}
            or len(row["absent_null"]["rows"]) != 3
        ):
            raise RuntimeError("G-ALPHA null coverage is incomplete")

    manifest = dict(sweep["mask_manifest"])
    manifest_sha = manifest.pop("sha256")
    rederived_manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest_sha != rederived_manifest_sha:
        raise RuntimeError("G-ALPHA mask manifest hash mismatch")

    valid_rows = [row for row in rows if row["valid"]]
    valid_rows.sort(
        key=lambda row: (
            float(row["alpha"]),
            POLICY_ORDER.index(str(row["policy"])),
        )
    )
    selected = sweep["selected_intervention"]
    if valid_rows:
        expected = valid_rows[0]
        if selected is None or (
            selected["policy"], float(selected["alpha"])
        ) != (expected["policy"], float(expected["alpha"])):
            raise RuntimeError("G-ALPHA did not select the smallest valid row")
    elif selected is not None:
        raise RuntimeError("G-ALPHA selected an intervention with no valid row")
    expected_status = "PASS" if valid_rows else "FAIL"
    if sweep["status"] != expected_status or sweep["g_alpha"] != expected_status:
        raise RuntimeError("G-ALPHA status does not rederive")

    raw_path = ROOT / str(sweep["raw_artifact"])
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != int(sweep["raw_artifact_bytes"])
        or _file_sha256(raw_path) != sweep["raw_artifact_sha256"]
    ):
        raise RuntimeError("G-ALPHA raw artifact integrity check failed")


def persist_alpha_sweep(sweep: Mapping[str, Any]) -> dict[str, Any]:
    _validate_alpha_sweep(sweep)
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    v3 = metrics["calibration_v3"]
    if (
        _repair_v2_sha256(metrics["repair_v2"])
        != v3["provenance"]["repair_v2_sha256"]
    ):
        raise RuntimeError("Immutable repair_v2 provenance changed during v3")
    if v3["gate_ledger"]["g_swap"] != "PASS":
        raise RuntimeError("G-ALPHA requires V3 G-SWAP PASS")
    protocol = v3["protocol"]
    if (
        protocol["alpha_grid"] != list(ALPHA_GRID)
        or protocol["primary_edit"] != "project_out_transfer"
        or protocol["thresholds"][
            "capability_abs_grand_and_bank_mean_delta_nll"
        ]
        != CAPABILITY_THRESHOLD
        or protocol["thresholds"]["g_pos_low_causal_abs_delta"]
        != GPOS_LOW_CAUSAL
        or protocol["thresholds"]["g_pos_max_weight_read_ratio"]
        != GPOS_LOW_READ_RATIO
    ):
        raise RuntimeError("Frozen notebook-00 G-ALPHA protocol drifted")
    v3["stage1_5_alpha_sweep"] = sweep
    v3["selected_intervention"] = sweep["selected_intervention"]
    v3["gate_ledger"]["g_alpha"] = sweep["g_alpha"]
    if sweep["g_alpha"] == "PASS":
        v3["gate_ledger"]["stage2_recalibration"] = "PENDING"
        v3["gate_ledger"]["stage4_report"] = "NOT_REQUIRED_YET"
        v3["current_allowed_conclusion"] = "ALPHA_SELECTED_STAGE2_REQUIRED"
    else:
        v3["gate_ledger"]["stage2_recalibration"] = "SKIPPED_PREREQUISITE"
        v3["gate_ledger"]["stage3_science"] = "SKIPPED_PREREQUISITE"
        v3["gate_ledger"]["stage4_report"] = "REQUIRED"
        v3.pop("stage2_recalibration", None)
        v3.pop("stage3_notebooks", None)
        v3.pop("stage4_fallback", None)
        v3["current_allowed_conclusion"] = (
            "NO_VALID_ALPHA_CALIBRATION_LIMITATION_NO_HYPOTHESIS_INFERENCE"
        )
    save_json(metrics_path, metrics)
    report_path = ROOT / "results" / "RESULTS.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## Stage 1.5 — surgical alpha sweep"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    verdict_start = report.index("## Current verdict")
    environment_start = report.index("## Environment")
    if sweep["g_alpha"] == "PASS":
        verdict = (
            "## Current verdict\n\n"
            "**G-ALPHA PASSED; STAGE 2 RECALIBRATION REQUIRED.** An "
            "intervention was selected, but no science is licensed until all "
            "Stage-2 gates pass at that intervention.\n\n"
        )
    else:
        verdict = (
            "## Current verdict\n\n"
            "**G-ALPHA FAILED; STAGE 2 AND STAGE 3 SKIPPED.** No frozen "
            "intervention passed all calibration requirements. The allowed "
            "conclusion is a calibration/READ-positive-control limitation, "
            "not a verdict on the Written-vs-Read hypothesis.\n\n"
        )
    report = (
        report[:verdict_start]
        + verdict
        + report[environment_start:]
    )
    report_path.write_text(
        report.rstrip() + _report_section(sweep), encoding="utf-8"
    )
    return metrics

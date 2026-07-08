"""Repair-first independent mean-difference concept-direction validation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from jlens.hooks import ActivationRecorder

from src.concept_vectors import (
    cosine_alignment,
    mean_difference_bank_from_matrices,
)
from src.data_gen import continuation_token_id
from src.jlens_iface import jlens_direction_bank
from src.md_manifest import (
    DEFAULT_MANIFEST,
    audit_md_manifest,
    baseline_exclusions,
    load_md_manifest,
    render_cue,
)
from src.md_validation import (
    fit_score_calibration,
    heldout_calibrated_retrieval,
    heldout_matched_baseline_deltas,
    leave_one_train_slot_out_stability,
)
from src.metrics import binomial_rate_with_ci, bootstrap_statistic, save_json
from src.model_utils import ModelBundle, batched_next_token_records, set_seed
from src.plotting import save_figure, set_style
from src.v2_repair import exact_label_token_id


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
EXPLICIT_TEMPLATES: dict[str, Callable[[str], str]] = {
    "legacy": lambda clue: (
        "Identify the entity described by this clue. Reply with its name only.\n"
        f"Clue: {clue}\nAnswer:"
    ),
    "exact_one_word": lambda clue: (
        "Infer the single entity described below. Answer with exactly its "
        f"one-word name and nothing else.\nClue: {clue}\nEntity:"
    ),
    "direct_completion": lambda clue: (
        f"Clue: {clue}\nThis clue describes the following single entity:"
    ),
    "question_one_word": lambda clue: (
        "Question: What single entity is described by this clue?\n"
        f"Clue: {clue}\nAnswer (one word):"
    ),
    "short": lambda clue: f"Name the entity.\n{clue}\nAnswer:",
}


def _rate(values: list[bool | int | float]) -> dict[str, Any]:
    return binomial_rate_with_ci(values)


def _mean(values: list[float]) -> dict[str, Any]:
    return bootstrap_statistic(
        [values], lambda array: float(np.mean(array)), seed=SEED
    )


def _probe_rows(
    bundle: ModelBundle,
    payload: Mapping[str, Any],
    *,
    split: str,
    renderer: Callable[[str], str],
) -> list[dict[str, Any]]:
    prompts: list[str] = []
    expected: list[int] = []
    metadata: list[dict[str, Any]] = []
    for concept in payload["concepts"]:
        for cue in concept["cues"]:
            if cue["split"] != split:
                continue
            prompt = renderer(cue["text"])
            token_id, surface = continuation_token_id(
                bundle.tokenizer, prompt, concept["concept"]
            )
            prompts.append(prompt)
            expected.append(token_id)
            metadata.append(
                {
                    "concept": concept["concept"],
                    "cue_id": cue["cue_id"],
                    "fact_id": cue["fact_id"],
                    "split": split,
                    "expected_surface": surface,
                }
            )
    rows = batched_next_token_records(
        bundle.hf_model,
        bundle.tokenizer,
        prompts,
        expected,
        batch_size=16,
        top_k=10,
    )
    for row, labels in zip(rows, metadata, strict=True):
        row.update(labels)
    return rows


def _silent_rows(
    bundle: ModelBundle, payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    prompts: list[str] = []
    expected: list[int] = []
    metadata: list[dict[str, Any]] = []
    for concept in payload["concepts"]:
        for cue in concept["cues"]:
            prompt = render_cue(payload, cue)
            token_id, surface = continuation_token_id(
                bundle.tokenizer, prompt, concept["concept"]
            )
            prompts.append(prompt)
            expected.append(token_id)
            metadata.append(
                {
                    "concept": concept["concept"],
                    "cue_id": cue["cue_id"],
                    "split": cue["split"],
                    "expected_surface": surface,
                }
            )
    rows = batched_next_token_records(
        bundle.hf_model,
        bundle.tokenizer,
        prompts,
        expected,
        batch_size=16,
        top_k=10,
    )
    for row, labels in zip(rows, metadata, strict=True):
        row.update(labels)
    return rows


def _cue_prompt_records(
    payload: Mapping[str, Any], split: str
) -> dict[str, list[dict[str, Any]]]:
    """Render audited cues while retaining their exact character spans."""

    template_order = [
        template_id
        for template_id in payload["templates"]
        if template_id.startswith(f"{split}_")
    ]
    records: dict[str, list[dict[str, Any]]] = {}
    for concept in payload["concepts"]:
        by_template = {
            cue["template_id"]: cue
            for cue in concept["cues"]
            if cue["split"] == split
        }
        rows: list[dict[str, Any]] = []
        for template_id in template_order:
            cue = by_template[template_id]
            prompt = render_cue(payload, cue)
            start = prompt.index(cue["text"])
            rows.append(
                {
                    "prompt": prompt,
                    "cue_id": cue["cue_id"],
                    "cue_char_start": start,
                    "cue_char_end": start + len(cue["text"]),
                }
            )
        records[concept["concept"]] = rows
    return records


@torch.no_grad()
def _cue_pooling_matrix_banks(
    lens_model: Any,
    records: Mapping[str, list[Mapping[str, Any]]],
    layers: list[int],
    *,
    batch_size: int = 16,
) -> dict[str, dict[str, dict[int, torch.Tensor]]]:
    """Capture anchor/cue-end/cue-last4/whole-cue residual summaries.

    Cue token indices are derived from tokenizer character offsets, so the
    common ``Keep the answer...`` suffix can never enter a cue pooling arm.
    """

    concepts = sorted(records)
    if not concepts or len({len(records[name]) for name in concepts}) != 1:
        raise ValueError("Cue records must have equal nonzero slots per concept")
    flat = [(concept, row) for concept in concepts for row in records[concept]]
    variants = ("anchor_final", "cue_end", "cue_last4", "cue_mean")
    chunks: dict[str, dict[str, dict[int, list[torch.Tensor]]]] = {
        variant: {
            concept: {layer: [] for layer in layers} for concept in concepts
        }
        for variant in variants
    }
    tokenizer = lens_model.tokenizer
    hf_model = getattr(lens_model, "_hf_model", None)
    if hf_model is None:
        raise TypeError("Cue pooling requires the official HF J-Lens adapter")
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        prompts = [str(row["prompt"]) for _, row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        input_ids = encoded.input_ids.to(lens_model.input_device)
        attention_mask = encoded.attention_mask.to(lens_model.input_device)
        pooling_positions: list[dict[str, list[int]]] = []
        for row_index, (_, record) in enumerate(batch):
            real = attention_mask[row_index].nonzero(as_tuple=False).flatten().tolist()
            cue_positions = [
                int(position)
                for position in real
                if int(offsets[row_index, position, 1])
                > int(record["cue_char_start"])
                and int(offsets[row_index, position, 0])
                < int(record["cue_char_end"])
                and int(offsets[row_index, position, 1])
                > int(offsets[row_index, position, 0])
            ]
            if not cue_positions:
                raise ValueError(f"No tokenizer tokens overlap cue {record['cue_id']}")
            pooling_positions.append(
                {
                    "anchor_final": [int(real[-1])],
                    "cue_end": cue_positions[-1:],
                    "cue_last4": cue_positions[-4:],
                    "cue_mean": cue_positions,
                }
            )
        with ActivationRecorder(lens_model.layers, at=layers) as recorder:
            hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        for row_index, (concept, _) in enumerate(batch):
            for variant in variants:
                positions = pooling_positions[row_index][variant]
                for layer in layers:
                    pooled = (
                        recorder.activations[layer][row_index, positions]
                        .float()
                        .mean(dim=0)
                        .detach()
                        .cpu()
                    )
                    chunks[variant][concept][layer].append(pooled)
    return {
        variant: {
            concept: {
                layer: torch.stack(values)
                for layer, values in per_layer.items()
            }
            for concept, per_layer in per_concept.items()
        }
        for variant, per_concept in chunks.items()
    }


def _training_pooling_selection(
    variant_matrices: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    exclusions: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Select pooling and layer by leave-one-train-template-out retrieval."""

    variants = sorted(variant_matrices)
    concepts = sorted(next(iter(variant_matrices.values())))
    layers = sorted(next(iter(next(iter(variant_matrices.values())).values())))
    n_slots = next(iter(variant_matrices.values()))[concepts[0]][layers[0]].shape[0]
    counts = {variant: {layer: 0 for layer in layers} for variant in variants}
    rows = {variant: {layer: 0 for layer in layers} for variant in variants}
    for variant in variants:
        matrices = variant_matrices[variant]
        for left_out in range(n_slots):
            kept = [slot for slot in range(n_slots) if slot != left_out]
            train_fold = {
                concept: {
                    layer: matrices[concept][layer][kept] for layer in layers
                }
                for concept in concepts
            }
            test_fold = {
                concept: {
                    layer: matrices[concept][layer][[left_out]] for layer in layers
                }
                for concept in concepts
            }
            directions, _ = mean_difference_bank_from_matrices(
                train_fold,
                baseline_exclusions=exclusions,
                matched_prompt_slots=True,
            )
            calibration = fit_score_calibration(train_fold, directions, exclusions)
            retrieval = heldout_calibrated_retrieval(
                test_fold,
                directions,
                calibration,
                n_permutations=1,
                permutation_seed=SEED,
            )
            for layer in layers:
                layer_rows = [
                    row
                    for row in retrieval["rows"]
                    if int(row["layer"]) == layer
                ]
                counts[variant][layer] += sum(int(row["top1"]) for row in layer_rows)
                rows[variant][layer] += len(layer_rows)
    grid = {
        variant: {
            str(layer): counts[variant][layer] / rows[variant][layer]
            for layer in layers
        }
        for variant in variants
    }
    selected_variant, selected_layer = sorted(
        (
            (variant, layer)
            for variant in variants
            for layer in layers
        ),
        key=lambda pair: (-grid[pair[0]][str(pair[1])], pair[0], pair[1]),
    )[0]
    return {
        "selection_uses_heldout": False,
        "method": "leave-one-train-template-out 40-way top1",
        "n_train_rows_per_candidate": len(concepts) * n_slots,
        "top1_grid": grid,
        "selected_pooling": selected_variant,
        "selected_layer": selected_layer,
        "selected_training_top1": grid[selected_variant][str(selected_layer)],
    }


def _select_explicit_template(
    bundle: ModelBundle, payload: Mapping[str, Any]
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    summaries: dict[str, Any] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for name, renderer in EXPLICIT_TEMPLATES.items():
        rows = _probe_rows(bundle, payload, split="train", renderer=renderer)
        all_rows[name] = rows
        summaries[name] = {
            "n": len(rows),
            "top1": _rate([row["top1_correct"] for row in rows]),
            "top5": _rate([row["top5_correct"] for row in rows]),
            "top10": _rate([row["top10_correct"] for row in rows]),
        }
    # Selection uses training cues only.  Top-5 is the gate target; top-1 and
    # name provide deterministic tie breaks.
    selected = sorted(
        summaries,
        key=lambda name: (
            -float(summaries[name]["top5"]["estimate"]),
            -float(summaries[name]["top1"]["estimate"]),
            name,
        ),
    )[0]
    return selected, summaries, all_rows[selected]


def _half_cpu_bank(
    bank: Mapping[str, Mapping[int, torch.Tensor]],
) -> dict[str, dict[int, torch.Tensor]]:
    return {
        concept: {
            int(layer): value.detach().half().cpu()
            for layer, value in per_layer.items()
        }
        for concept, per_layer in bank.items()
    }


def _alignment_summary(
    alignments: Mapping[str, Mapping[str, Mapping[int, float]]],
    fixed_layer: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "per_concept_layer": {
            concept: {
                convention: {str(layer): float(value) for layer, value in layers.items()}
                for convention, layers in conventions.items()
            }
            for concept, conventions in alignments.items()
        }
    }
    for convention in ("raw_WU_J", "rms_gain_folded"):
        fixed_values = [
            float(alignments[concept][convention][fixed_layer])
            for concept in sorted(alignments)
        ]
        band_values = [
            float(np.mean(list(alignments[concept][convention].values())))
            for concept in sorted(alignments)
        ]
        output[convention] = {
            "fixed_layer": _mean(fixed_values),
            "band_mean": _mean(band_values),
        }
    return output


def _plot_alignment(
    alignments: Mapping[str, Mapping[str, Mapping[int, float]]],
    layers: list[int],
) -> str:
    concepts = sorted(alignments)
    values = np.asarray(
        [
            [alignments[concept]["raw_WU_J"][layer] for layer in layers]
            for concept in concepts
        ]
    )
    set_style()
    figure, axis = plt.subplots(figsize=(10.5, 9.5))
    sns.heatmap(
        values,
        ax=axis,
        cmap="vlag",
        center=0,
        xticklabels=layers,
        yticklabels=concepts,
        cbar_kws={"label": "cosine(MD, exact-label raw J-Lens direction)"},
    )
    axis.set(
        xlabel="post-block layer",
        ylabel="concept",
        title="Repaired MD direction alignment across the validated Qwen band",
    )
    path = ROOT / "results" / "figures" / "repair_md_alignment.png"
    save_figure(figure, path)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def run_stage1c(
    bundle: ModelBundle,
    lens: Any,
    *,
    workspace_layers: list[int],
) -> dict[str, Any]:
    """Build, validate, and gate the independent MD direction family."""

    set_seed(SEED)
    if not workspace_layers:
        raise ValueError("The repaired workspace band is required")
    payload = load_md_manifest()
    audit = audit_md_manifest(payload, bundle.tokenizer)
    if audit["status"] != "PASS":
        raise RuntimeError("MD cue audit failed")
    exclusions = baseline_exclusions(payload)
    train_variants = _cue_pooling_matrix_banks(
        bundle.lens_model,
        _cue_prompt_records(payload, "train"),
        workspace_layers,
        batch_size=16,
    )
    position_selection = _training_pooling_selection(train_variants, exclusions)
    selected_pooling = str(position_selection["selected_pooling"])
    fixed_layer = int(position_selection["selected_layer"])
    heldout_variants = _cue_pooling_matrix_banks(
        bundle.lens_model,
        _cue_prompt_records(payload, "heldout"),
        workspace_layers,
        batch_size=16,
    )
    train_matrices = train_variants[selected_pooling]
    heldout_matrices = heldout_variants[selected_pooling]
    directions, train_means = mean_difference_bank_from_matrices(
        train_matrices,
        baseline_exclusions=exclusions,
        matched_prompt_slots=True,
        device="cuda",
    )
    calibration = fit_score_calibration(train_matrices, directions, exclusions)
    retrieval = heldout_calibrated_retrieval(
        heldout_matrices,
        directions,
        calibration,
        n_permutations=5000,
        permutation_seed=SEED,
    )
    sign_rows = heldout_matched_baseline_deltas(
        heldout_matrices, directions, exclusions
    )
    stability_rows = leave_one_train_slot_out_stability(
        train_matrices, directions, exclusions
    )

    fixed_rows = [
        row for row in retrieval["rows"] if int(row["layer"]) == fixed_layer
    ]
    retrieval_top1 = _rate([row["top1"] for row in fixed_rows])
    fixed_retrieval = retrieval["by_layer"][fixed_layer]
    retrieval_auroc = _mean(
        list(fixed_retrieval["per_concept_ovr_auroc"].values())
    )
    fixed_sign = [
        row for row in sign_rows if int(row["layer"]) == fixed_layer
    ]
    by_concept: dict[str, list[float]] = defaultdict(list)
    for row in fixed_sign:
        by_concept[str(row["concept"])].append(float(row["delta"]))
    concept_deltas = {
        concept: float(np.mean(values)) for concept, values in by_concept.items()
    }
    sign_positive = _rate([value > 0 for value in concept_deltas.values()])
    sign_delta = _mean(list(concept_deltas.values()))
    fixed_stability = [
        float(row["cosine_to_full_direction"])
        for row in stability_rows
        if int(row["layer"]) == fixed_layer
    ]
    stability_median = float(np.median(fixed_stability))

    selected_template, training_template_summaries, selected_training_rows = (
        _select_explicit_template(bundle, payload)
    )
    explicit_rows = _probe_rows(
        bundle,
        payload,
        split="heldout",
        renderer=EXPLICIT_TEMPLATES[selected_template],
    )
    explicit_top1 = _rate([row["top1_correct"] for row in explicit_rows])
    explicit_top5 = _rate([row["top5_correct"] for row in explicit_rows])
    silent_rows = _silent_rows(bundle, payload)
    silent_top1 = _rate([row["top1_correct"] for row in silent_rows])
    silent_top10 = _rate([row["top10_correct"] for row in silent_rows])

    concept_specs = {value["concept"]: value for value in payload["concepts"]}
    direction_tokens = {
        concept: exact_label_token_id(bundle.tokenizer, concept)
        for concept in concept_specs
    }
    token_ids = [token_id for token_id, _ in direction_tokens.values()]
    raw_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        workspace_layers,
        fold_rms_gain=False,
    )
    folded_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        workspace_layers,
        fold_rms_gain=True,
    )
    alignments: dict[str, dict[str, dict[int, float]]] = {}
    for concept, (token_id, _) in direction_tokens.items():
        alignments[concept] = {
            "raw_WU_J": cosine_alignment(directions[concept], raw_bank[token_id]),
            "rms_gain_folded": cosine_alignment(
                directions[concept], folded_bank[token_id]
            ),
        }
    alignment_summary = _alignment_summary(alignments, fixed_layer)

    maximum_norm_error = max(
        abs(float(value.norm()) - 1.0)
        for per_layer in directions.values()
        for value in per_layer.values()
    )
    chance = 1.0 / len(directions)
    criteria = {
        "retrieval_estimate_at_least_4x_chance": (
            retrieval_top1["estimate"] >= 4.0 * chance
        ),
        "retrieval_ci_above_chance": retrieval_top1["ci_low"] > chance,
        "retrieval_permutation_p_below_0.01": (
            fixed_retrieval["top1_fixed_label_permutation"]["p_value"] < 0.01
        ),
        "retrieval_auroc_ci_above_half": retrieval_auroc["ci_low"] > 0.5,
        "known_answer_top5_at_least_0.80": explicit_top5["estimate"] >= 0.80,
        "heldout_sign_fraction_at_least_0.80": sign_positive["estimate"] >= 0.80,
        "stability_median_at_least_0.70": stability_median >= 0.70,
        "silent_top10_at_most_0.25": silent_top10["estimate"] <= 0.25,
        "unit_norm_within_1e-5": maximum_norm_error <= 1e-5,
    }
    passed = all(criteria.values())

    manifest_sha256 = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
    artifact_path = ROOT / "data" / "directions" / "qwen2.5-7b_md_v2.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "schema_version": "md-repair-v2",
                "seed": SEED,
                "model_id": bundle.model_id,
                "model_revision": bundle.revision,
                "workspace_layers": workspace_layers,
                "fixed_validation_layer": fixed_layer,
                "manifest_sha256": manifest_sha256,
                "direction_tokens": direction_tokens,
                "baseline_exclusions": exclusions,
            },
            "mean_difference": _half_cpu_bank(directions),
            "train_means": _half_cpu_bank(train_means),
            "train_matrices": _half_cpu_bank(train_matrices),
            "heldout_matrices": _half_cpu_bank(heldout_matrices),
        },
        artifact_path,
    )

    summary: dict[str, Any] = {
        "status": "PASS" if passed else "DROPPED_MD",
        "md_use_permitted": passed,
        "gate_interpretation": (
            "validated independent direction"
            if passed
            else "MD arm dropped; failure is not evidence against hypothesis"
        ),
        "seed": SEED,
        "model_id": bundle.model_id,
        "model_revision": bundle.revision,
        "workspace_layers": workspace_layers,
        "fixed_validation_layer": fixed_layer,
        "fixed_layer_selection": (
            "joint layer/pooling choice by leave-one-train-template-out retrieval "
            "inside the clean-readout-selected Stage-1 workspace; no heldout use"
        ),
        "position_selection": position_selection,
        "n_concepts": len(directions),
        "n_train_cues_per_concept": 4,
        "n_heldout_cues_per_concept": 2,
        "chance_retrieval": chance,
        "cue_audit": audit,
        "manifest_sha256": manifest_sha256,
        "criteria": criteria,
        "retrieval": {
            "top1_at_fixed_layer": retrieval_top1,
            "fixed_layer_summary": fixed_retrieval,
            "macro_ovr_auroc_at_fixed_layer": retrieval_auroc,
            "by_layer": {
                str(layer): value for layer, value in retrieval["by_layer"].items()
            },
            "fixed_layer_rows": fixed_rows,
        },
        "heldout_sign": {
            "positive_concept_fraction": sign_positive,
            "mean_delta": sign_delta,
            "per_concept_mean_delta": concept_deltas,
        },
        "stability": {
            "median_cosine_at_fixed_layer": stability_median,
            "fixed_layer_rows": [
                row
                for row in stability_rows
                if int(row["layer"]) == fixed_layer
            ],
        },
        "explicit_known_answer": {
            "selection_uses_heldout": False,
            "selected_template": selected_template,
            "selected_template_source": EXPLICIT_TEMPLATES[selected_template]("{cue}"),
            "training_template_summaries": training_template_summaries,
            "selected_training_rows": selected_training_rows,
            "heldout_top1": explicit_top1,
            "heldout_top5": explicit_top5,
            "heldout_rows": explicit_rows,
        },
        "silent_output_contamination": {
            "top1": silent_top1,
            "top10": silent_top10,
        },
        "direction_tokens": {
            concept: {"token_id": token_id, "surface": surface}
            for concept, (token_id, surface) in direction_tokens.items()
        },
        "cosine_alignment": alignment_summary,
        "max_unit_norm_error": maximum_norm_error,
        "score_calibration": {
            concept: {str(layer): value for layer, value in per_layer.items()}
            for concept, per_layer in calibration.items()
        },
        "artifact": str(artifact_path.relative_to(ROOT)),
        "limitations": [
            "MD directions remain weakly aligned with exact-label J-Lens directions; cosine is diagnostic, not a gate.",
            "The explicit-probe wording was selected on training cues only.",
            "The cue set is authored and covers forty single-token concepts.",
        ],
    }
    summary["figure"] = _plot_alignment(alignments, workspace_layers)
    raw_path = ROOT / "data" / "raw" / "02_concept_finder_v2.json"
    save_json(
        raw_path,
        {
            "summary": summary,
            "all_retrieval_rows": retrieval["rows"],
            "all_sign_rows": sign_rows,
            "all_stability_rows": stability_rows,
            "silent_rows": silent_rows,
        },
    )
    summary["raw_artifact"] = str(raw_path.relative_to(ROOT))
    return summary


def _report_section(stage1c: Mapping[str, Any]) -> str:
    retrieval = stage1c["retrieval"]["top1_at_fixed_layer"]
    explicit = stage1c["explicit_known_answer"]
    alignment = stage1c["cosine_alignment"]["raw_WU_J"]["fixed_layer"]
    criteria = "\n".join(
        f"| {name} | {'PASS' if value else 'FAIL'} |"
        for name, value in stage1c["criteria"].items()
    )
    return f"""

## Stage 1c — independent concept finder (G-DIR)

The old analysis fixed validation at L18 and sampled the final common
instruction token; its 40-way held-out top-1 was 0.0625. Within the repaired
L{stage1c['workspace_layers'][0]}–L{stage1c['workspace_layers'][-1]} band, v2
selects **{stage1c['position_selection']['selected_pooling']} at
L{stage1c['fixed_validation_layer']}** by leave-one-training-template-out
retrieval. No held-out MD result entered that selection.

- Held-out retrieval: **{retrieval['estimate']:.3f}** ({retrieval['n_success']}/{retrieval['n']}); 95% Wilson CI [{retrieval['ci_low']:.3f}, {retrieval['ci_high']:.3f}]; chance={stage1c['chance_retrieval']:.3f}.
- Exact-token held-out known-answer top-5: **{explicit['heldout_top5']['estimate']:.3f}**; top-1={explicit['heldout_top1']['estimate']:.3f}; N={explicit['heldout_top5']['n']}.
- Explicit wording `{explicit['selected_template']}` was selected using training cues only.
- cosine(MD, exact-label raw J-Lens) at L{stage1c['fixed_validation_layer']}: **{alignment['estimate']:.3f}** (95% CI [{alignment['ci_low']:.3f}, {alignment['ci_high']:.3f}], N={alignment['n']}). Alignment is reported, not required to be high.

| criterion | result |
| --- | --- |
{criteria}

### G-DIR decision

**{stage1c['status']}**. {'The MD arm is validated for later robustness checks.' if stage1c['status'] == 'PASS' else 'The MD arm is dropped; its failure is not evidence against the hypothesis.'}

![MD/J-Lens alignment](figures/repair_md_alignment.png)

Stage-3 science remains prohibited pending READ validation, firing controls,
and G-POS.
"""


def persist_stage1c(stage1c: Mapping[str, Any]) -> dict[str, Any]:
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    repair = metrics["repair_v2"]
    if repair["gate_ledger"]["g_swap"] != "PASS":
        raise RuntimeError("G-DIR cannot be persisted before G-SWAP passes")
    repair["stage1c_concept_finder"] = dict(stage1c)
    repair["gate_ledger"]["g_dir"] = stage1c["status"]
    repair["gate_ledger"]["stage3_science"] = "PROHIBITED"
    repair["current_allowed_conclusion"] = "DIRECTION_CALIBRATED_READ_PENDING"
    save_json(metrics_path, metrics)
    report_path = ROOT / "results" / "RESULTS.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## Stage 1c — independent concept finder (G-DIR)"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    report_path.write_text(report.rstrip() + _report_section(stage1c), encoding="utf-8")
    return metrics

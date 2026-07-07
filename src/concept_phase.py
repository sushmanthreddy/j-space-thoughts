"""Notebook-01 orchestration for independently estimated concept directions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.concept_vectors import (
    cosine_alignment,
    mean_difference_bank_from_matrices,
    prompt_matrix_bank,
)
from src.jlens_iface import jlens_direction_bank, load_published_lens, workspace_layers
from src.md_manifest import (
    DEFAULT_MANIFEST,
    audit_md_manifest,
    baseline_exclusions,
    concept_prompt_sets,
    explicit_probe_prompt,
    load_md_manifest,
    render_cue,
)
from src.md_validation import (
    fit_score_calibration,
    heldout_calibrated_retrieval,
    heldout_matched_baseline_deltas,
    leave_one_train_slot_out_stability,
)
from src.metrics import bootstrap_statistic, save_json
from src.model_utils import (
    MODEL_REVISIONS,
    batched_next_token_records,
    load_model,
    set_seed,
)
from src.plotting import save_figure, set_style


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729


def _mean_ci(values: list[float], *, n_bootstrap: int = 5000) -> dict[str, Any]:
    return bootstrap_statistic(
        [values],
        lambda array: float(np.mean(array)),
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        seed=SEED,
    )


def _manifest_probe_metadata(
    payload: dict[str, Any],
    *,
    splits: set[str],
    explicit: bool,
) -> tuple[list[str], list[int], list[dict[str, str]]]:
    prompts: list[str] = []
    token_ids: list[int] = []
    metadata: list[dict[str, str]] = []
    for concept in payload["concepts"]:
        for cue in concept["cues"]:
            if cue["split"] not in splits:
                continue
            prompts.append(
                explicit_probe_prompt(cue) if explicit else render_cue(payload, cue)
            )
            token_ids.append(int(concept["token_id"]))
            metadata.append(
                {
                    "concept": concept["concept"],
                    "cue_id": cue["cue_id"],
                    "fact_id": cue["fact_id"],
                    "split": cue["split"],
                    "probe_type": "explicit_answer" if explicit else "silent_hold",
                }
            )
    return prompts, token_ids, metadata


def _run_probe_bank(
    bundle: Any,
    payload: dict[str, Any],
    *,
    splits: set[str],
    explicit: bool,
) -> list[dict[str, Any]]:
    prompts, expected_ids, metadata = _manifest_probe_metadata(
        payload,
        splits=splits,
        explicit=explicit,
    )
    rows = batched_next_token_records(
        bundle.hf_model,
        bundle.tokenizer,
        prompts,
        expected_ids,
        batch_size=8,
        top_k=10,
        max_length=128,
    )
    for row, labels in zip(rows, metadata, strict=True):
        row.update(labels)
    return rows


def _concept_level_sign_summary(
    rows: list[dict[str, Any]], fixed_layer: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_concept: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if int(row["layer"]) == fixed_layer:
            by_concept[row["concept"]].append(float(row["delta"]))
    concept_rows = [
        {
            "concept": concept,
            "mean_heldout_delta": float(np.mean(values)),
            "positive": int(float(np.mean(values)) > 0),
            "n_heldout_cues": len(values),
        }
        for concept, values in sorted(by_concept.items())
    ]
    positive_ci = _mean_ci([float(row["positive"]) for row in concept_rows])
    delta_ci = _mean_ci([float(row["mean_heldout_delta"]) for row in concept_rows])
    return concept_rows, positive_ci, delta_ci


def _alignment_summary(
    alignments: dict[str, dict[str, dict[int, float]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for convention in ("raw_WU_J", "rms_gain_folded"):
        values = [
            float(np.mean(list(alignments[name][convention].values())))
            for name in sorted(alignments)
        ]
        result[convention] = _mean_ci(values)
    result["per_concept_layer"] = {
        name: {
            convention: {str(layer): value for layer, value in per_layer.items()}
            for convention, per_layer in conventions.items()
        }
        for name, conventions in alignments.items()
    }
    return result


def _half_cpu_bank(
    bank: dict[str, dict[int, torch.Tensor]],
) -> dict[str, dict[int, torch.Tensor]]:
    return {
        concept: {
            layer: tensor.detach().half().cpu() for layer, tensor in layers.items()
        }
        for concept, layers in bank.items()
    }


def _half_cpu_matrices(
    bank: dict[str, dict[int, torch.Tensor]],
) -> dict[str, dict[int, torch.Tensor]]:
    return {
        concept: {
            layer: tensor.detach().half().cpu() for layer, tensor in layers.items()
        }
        for concept, layers in bank.items()
    }


def run_concept_vector_phase() -> dict[str, Any]:
    """Audit, construct, validate, persist, and summarize both direction families."""

    set_seed(SEED)
    payload = load_md_manifest()
    if payload["seed"] != SEED:
        raise ValueError("MD manifest seed differs from the project seed")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        payload["model"]["id"],
        revision=payload["model"]["revision"],
        local_files_only=True,
    )
    audit = audit_md_manifest(payload, tokenizer)
    print(
        f"MD CUE AUDIT {audit['status']}: {audit['n_concepts']} concepts, "
        f"{audit['n_train_cues']} train + {audit['n_heldout_cues']} held-out cues"
    )
    if audit["status"] != "PASS":
        raise RuntimeError("Refusing to collect activations from a failed cue audit")

    bundle = load_model(payload["model"]["id"])
    if bundle.revision != MODEL_REVISIONS[bundle.model_id]:
        raise ValueError(
            "Loaded model revision differs from the pinned project revision"
        )
    lens = load_published_lens(bundle.model_id)
    layers = workspace_layers(bundle.lens_model.n_layers, lens.source_layers)
    validation_plan = payload["validation_plan"]
    if layers != validation_plan["workspace_layers"]:
        raise ValueError(
            f"Workspace drift: computed {layers}, frozen {validation_plan['workspace_layers']}"
        )
    fixed_layer = int(validation_plan["fixed_validation_layer"])

    train_prompts = concept_prompt_sets(payload, "train")
    heldout_prompts = concept_prompt_sets(payload, "heldout")
    exclusions = baseline_exclusions(payload)
    print(
        f"Capturing silent MD cues in shared batches: layers={layers}, "
        f"fixed validation layer={fixed_layer}"
    )
    train_matrices = prompt_matrix_bank(
        bundle.lens_model,
        train_prompts,
        layers,
        batch_size=16,
    )
    heldout_matrices = prompt_matrix_bank(
        bundle.lens_model,
        heldout_prompts,
        layers,
        batch_size=16,
    )
    md_directions, train_means = mean_difference_bank_from_matrices(
        train_matrices,
        baseline_exclusions=exclusions,
        matched_prompt_slots=True,
        device="cuda",
    )

    calibration = fit_score_calibration(
        train_matrices,
        md_directions,
        exclusions,
        norm_tolerance=float(validation_plan["unit_norm_tolerance"]),
    )
    retrieval = heldout_calibrated_retrieval(
        heldout_matrices,
        md_directions,
        calibration,
        n_permutations=5000,
        permutation_seed=SEED,
    )
    sign_rows = heldout_matched_baseline_deltas(
        heldout_matrices,
        md_directions,
        exclusions,
        norm_tolerance=float(validation_plan["unit_norm_tolerance"]),
    )
    stability_rows = leave_one_train_slot_out_stability(
        train_matrices,
        md_directions,
        exclusions,
        norm_tolerance=float(validation_plan["unit_norm_tolerance"]),
    )

    concept_specs = {concept["concept"]: concept for concept in payload["concepts"]}
    token_ids = [int(concept["token_id"]) for concept in payload["concepts"]]
    raw_token_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        layers,
        fold_rms_gain=False,
    )
    effective_token_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        layers,
        fold_rms_gain=True,
    )
    alignments: dict[str, dict[str, dict[int, float]]] = {}
    for name, concept in concept_specs.items():
        token_id = int(concept["token_id"])
        alignments[name] = {
            "raw_WU_J": cosine_alignment(md_directions[name], raw_token_bank[token_id]),
            "rms_gain_folded": cosine_alignment(
                md_directions[name], effective_token_bank[token_id]
            ),
        }

    print(
        "Running separate explicit known-answer probes and silent contamination probes"
    )
    explicit_rows = _run_probe_bank(
        bundle,
        payload,
        splits={"heldout"},
        explicit=True,
    )
    silent_rows = _run_probe_bank(
        bundle,
        payload,
        splits={"train", "heldout"},
        explicit=False,
    )

    concept_sign_rows, sign_positive_ci, sign_delta_ci = _concept_level_sign_summary(
        sign_rows,
        fixed_layer,
    )
    fixed_retrieval_rows = [
        row for row in retrieval["rows"] if int(row["layer"]) == fixed_layer
    ]
    retrieval_top1_ci = _mean_ci([float(row["top1"]) for row in fixed_retrieval_rows])
    fixed_retrieval = retrieval["by_layer"][fixed_layer]
    retrieval_auroc_ci = _mean_ci(
        list(fixed_retrieval["per_concept_ovr_auroc"].values())
    )
    explicit_top1_ci = _mean_ci([float(row["top1_correct"]) for row in explicit_rows])
    explicit_top5_ci = _mean_ci([float(row["top5_correct"]) for row in explicit_rows])
    silent_top1_ci = _mean_ci([float(row["top1_correct"]) for row in silent_rows])
    silent_top10_ci = _mean_ci([float(row["top10_correct"]) for row in silent_rows])
    fixed_stability = [
        float(row["cosine_to_full_direction"])
        for row in stability_rows
        if int(row["layer"]) == fixed_layer
    ]
    stability_median = float(np.median(fixed_stability))

    max_norm_error = max(
        abs(float(direction.norm()) - 1.0)
        for directions in md_directions.values()
        for direction in directions.values()
    )
    chance = 1.0 / len(md_directions)
    criteria = {
        "heldout_sign_fraction_at_least_0.80": sign_positive_ci["estimate"] >= 0.80,
        "heldout_sign_mean_ci_excludes_zero": sign_delta_ci["ci_low"] > 0,
        "retrieval_top1_ci_above_chance": retrieval_top1_ci["ci_low"] > chance,
        "retrieval_auroc_ci_above_half": retrieval_auroc_ci["ci_low"] > 0.5,
        "retrieval_permutation_p_below_0.01": fixed_retrieval[
            "top1_fixed_label_permutation"
        ]["p_value"]
        < 0.01,
        "explicit_top1_at_least_0.50": explicit_top1_ci["estimate"] >= 0.50,
        "explicit_top5_at_least_0.80": explicit_top5_ci["estimate"] >= 0.80,
        "silent_top1_at_most_0.10": silent_top1_ci["estimate"] <= 0.10,
        "silent_top10_at_most_0.25": silent_top10_ci["estimate"] <= 0.25,
        "loo_median_cosine_at_least_0.70": stability_median >= 0.70,
        "unit_norm_within_tolerance": max_norm_error
        <= float(validation_plan["unit_norm_tolerance"]),
    }
    status = "PASS" if all(criteria.values()) else "FAIL"

    manifest_sha256 = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
    artifact_path = ROOT / "data" / "directions" / "qwen2.5-7b_concept_vectors.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "seed": SEED,
                "model_id": bundle.model_id,
                "model_revision": bundle.revision,
                "lens_n_prompts": lens.n_prompts,
                "workspace_layers": layers,
                "fixed_validation_layer": fixed_layer,
                "manifest_path": str(DEFAULT_MANIFEST.relative_to(ROOT)),
                "manifest_sha256": manifest_sha256,
                "concepts": payload["concepts"],
                "baseline_exclusions": exclusions,
            },
            "mean_difference": _half_cpu_bank(md_directions),
            "train_means": _half_cpu_bank(train_means),
            "train_matrices": _half_cpu_matrices(train_matrices),
            "heldout_matrices": _half_cpu_matrices(heldout_matrices),
        },
        artifact_path,
    )

    set_style()
    concept_order = sorted(alignments)
    heatmap = np.asarray(
        [
            [alignments[name]["raw_WU_J"][layer] for layer in layers]
            for name in concept_order
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 10))
    sns.heatmap(
        heatmap,
        ax=axis,
        cmap="vlag",
        center=0,
        xticklabels=layers,
        yticklabels=concept_order,
        cbar_kws={"label": "cosine(MD, raw J-Lens direction)"},
    )
    axis.set(
        xlabel="post-block layer",
        ylabel="concept",
        title="Independent MD vs. preregistered raw J-Lens directions",
    )
    alignment_figure = save_figure(
        figure,
        ROOT / "results" / "figures" / "concept_direction_alignment_qwen7b.png",
    )
    plt.close(figure)

    summary: dict[str, Any] = {
        "status": status,
        "criteria": criteria,
        "validation_plan": validation_plan,
        "seed": SEED,
        "n_concepts": len(md_directions),
        "n_pairs": len(payload["pairs"]),
        "n_train_cues_per_concept": 4,
        "n_heldout_cues_per_concept": 2,
        "workspace_layers": layers,
        "fixed_validation_layer": fixed_layer,
        "chance_retrieval": chance,
        "cue_audit": audit,
        "manifest_sha256": manifest_sha256,
        "max_unit_norm_error": max_norm_error,
        "heldout_sign": {
            "positive_concept_fraction": sign_positive_ci,
            "mean_delta": sign_delta_ci,
            "per_concept": concept_sign_rows,
            "all_rows": sign_rows,
        },
        "heldout_retrieval": {
            "top1_at_fixed_layer": retrieval_top1_ci,
            "macro_ovr_auroc_at_fixed_layer": retrieval_auroc_ci,
            "fixed_layer_summary": fixed_retrieval,
            "by_layer": {
                str(layer): values for layer, values in retrieval["by_layer"].items()
            },
            "all_rows": retrieval["rows"],
        },
        "explicit_known_answer": {
            "top1": explicit_top1_ci,
            "top5": explicit_top5_ci,
            "rows": explicit_rows,
        },
        "silent_output_contamination": {
            "top1": silent_top1_ci,
            "top10": silent_top10_ci,
            "rows": silent_rows,
        },
        "leave_one_slot_out_stability": {
            "median_cosine_at_fixed_layer": stability_median,
            "rows": stability_rows,
        },
        "score_calibration": {
            concept: {str(layer): values for layer, values in per_layer.items()}
            for concept, per_layer in calibration.items()
        },
        "cosine_alignment": _alignment_summary(alignments),
        "artifact": str(artifact_path.relative_to(ROOT)),
        "figure": str(alignment_figure.relative_to(ROOT)),
    }
    metrics_path = ROOT / "results" / "metrics.json"
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics["concept_vectors"] = summary
    save_json(metrics_path, metrics)
    save_json(ROOT / "data" / "raw" / "01_concept_vectors.json", summary)
    print(
        f"NB01 {status}: N={len(md_directions)} concepts; "
        f"retrieval top1={retrieval_top1_ci['estimate']:.3f} "
        f"95% CI [{retrieval_top1_ci['ci_low']:.3f}, "
        f"{retrieval_top1_ci['ci_high']:.3f}]; "
        f"explicit top1={explicit_top1_ci['estimate']:.3f}; "
        f"silent top10={silent_top10_ci['estimate']:.3f}; "
        f"LOO median cosine={stability_median:.3f}"
    )
    if status == "FAIL":
        failed = [name for name, passed in criteria.items() if not passed]
        print(f"Failed preregistered MD validation criteria: {failed}")
    return summary

"""Pinned Qwen-7B versus Qwen-14B scale-comparison orchestration.

Notebook 06 uses this module to validate the locally fitted 14B Jacobian lens,
run model-specific correctness gates, rebuild mean-difference (MD) directions
from the frozen silent-cue manifest, execute the existing two-hop pipeline, and
compare the two scales.  No 7B activation or direction tensor is ever accepted
as a 14B MD input.

The statistical comparison functions and F7 plotting function are deliberately
model-free.  They consume persisted payloads and therefore remain unit-testable
on CPU without downloading or loading either Qwen checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.concept_vectors import (
    cosine_alignment,
    mean_difference_bank_from_matrices,
    prompt_matrix_bank,
)
from src.gates import run_g1, run_g2
from src.jlens_iface import (
    jlens_direction_bank,
    load_local_lens,
    validate_lens,
    workspace_layers,
)
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
from src.metrics import (
    bootstrap_statistic,
    partial_correlation,
    pearson_r,
    save_json,
)
from src.model_utils import (
    MODEL_REVISIONS,
    batched_next_token_records,
    load_model,
    release_model,
    set_seed,
)
from src.plotting import save_figure, set_style
from src.twohop_phase import (
    MD_DIRECTION_METHOD,
    PRIMARY_DIRECTION_METHOD,
    run_twohop_phase,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 1729
SCHEMA_VERSION = "scale-phase-v1"

MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
MODEL_14B = "Qwen/Qwen2.5-14B-Instruct"
MODEL_TAGS = ("7B", "14B")
DIRECTION_METHODS = (PRIMARY_DIRECTION_METHOD, MD_DIRECTION_METHOD)

LOCAL_LENS_14B = ROOT / "data/lenses/qwen2.5-14b_jlens_100prompts.pt"
LOCAL_LENS_14B_METADATA = ROOT / "data/lenses/qwen2.5-14b_jlens_100prompts.json"
MD_ARTIFACT_14B = ROOT / "data/directions/qwen2.5-14b_concept_vectors.pt"

RAW_GATES_7B = ROOT / "data/raw/00_gates_qwen7b.json"
RAW_MD_7B = ROOT / "data/raw/01_concept_vectors.json"
RAW_TWOHOP_7B = ROOT / "data/raw/02_twohop_qwen2.5-7b.json"
RAW_GATES_14B = ROOT / "data/raw/06_gates_qwen14b.json"
RAW_MD_14B = ROOT / "data/raw/06_md_qwen14b.json"
RAW_TWOHOP_14B = ROOT / "data/raw/06_twohop_qwen14b.json"
RAW_SCALE = ROOT / "data/raw/06_scale_comparison.json"
CURATED_SCALE = ROOT / "results/scale_comparison.json"
F7_PATH = ROOT / "results/figures/f7_scale_comparison.png"

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_SELECTION = "first 100 train records with >=600 characters"
EXPECTED_MD_MANIFEST_SHA256 = (
    "9b7ee62d78f0d9922fd50fa12554500e3fc376e3d147eb759b78a87ac3a9a169"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {target}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return str(target.relative_to(ROOT.resolve()))
    except ValueError:
        return str(target)


def _resolve_declared_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def map_reference_layer(
    reference_layer: int,
    reference_n_layers: int,
    target_n_layers: int,
) -> int:
    """Map a frozen block index to another depth before inspecting outcomes."""

    if reference_n_layers < 2 or target_n_layers < 2:
        raise ValueError("Layer mapping requires models with at least two blocks")
    if not 0 <= reference_layer < reference_n_layers:
        raise ValueError("Reference layer is outside the reference model")
    relative_depth = reference_layer / (reference_n_layers - 1)
    return int(round(relative_depth * (target_n_layers - 1)))


def validate_qwen14b_lens_provenance(
    bundle: Any,
    lens: Any,
    metadata: Mapping[str, Any],
    *,
    lens_path: str | Path = LOCAL_LENS_14B,
    metadata_path: str | Path | None = LOCAL_LENS_14B_METADATA,
    expected_n_prompts: int = 100,
) -> dict[str, Any]:
    """Fail closed on 14B model, local-lens dimensions, and fit provenance."""

    if bundle.model_id != MODEL_14B:
        raise ValueError(f"Expected {MODEL_14B!r}, got {bundle.model_id!r}")
    expected_revision = MODEL_REVISIONS[MODEL_14B]
    if bundle.revision != expected_revision:
        raise ValueError("Loaded 14B model revision differs from the project pin")
    if metadata.get("model_id") != bundle.model_id:
        raise ValueError("Lens metadata model ID differs from the loaded model")
    if metadata.get("model_revision") != bundle.revision:
        raise ValueError("Lens metadata model revision differs from the loaded model")

    lens_file = Path(lens_path)
    if not lens_file.is_file() or lens_file.stat().st_size == 0:
        raise FileNotFoundError(f"Missing nonempty 14B lens file: {lens_file}")
    declared_lens = metadata.get("lens_path")
    if not isinstance(declared_lens, str):
        raise ValueError("Lens metadata must include its lens_path")
    if _resolve_declared_path(declared_lens) != lens_file.resolve():
        raise ValueError("Lens metadata lens_path does not identify the loaded file")

    if metadata_path is not None:
        metadata_file = Path(metadata_path)
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Missing 14B lens metadata: {metadata_file}")
        if _read_json(metadata_file) != dict(metadata):
            raise ValueError("In-memory metadata differs from the persisted JSON")
    else:
        metadata_file = None

    validate_lens(lens, bundle.lens_model)
    n_layers = int(bundle.lens_model.n_layers)
    d_model = int(bundle.lens_model.d_model)
    expected_workspace = workspace_layers(n_layers, range(n_layers - 1))
    lens_layers = [int(layer) for layer in lens.source_layers]
    metadata_layers = [int(layer) for layer in metadata.get("source_layers", [])]
    if lens_layers != expected_workspace:
        raise ValueError(
            "14B lens source layers are not the preregistered model-specific "
            f"workspace: lens={lens_layers}, expected={expected_workspace}"
        )
    if metadata_layers != lens_layers:
        raise ValueError("Lens metadata source layers differ from the loaded lens")
    if int(metadata.get("target_layer", -1)) != n_layers - 1:
        raise ValueError("Lens metadata target layer is not the final model block")
    if "d_model" in metadata and int(metadata["d_model"]) != d_model:
        raise ValueError("Lens metadata d_model differs from the loaded model")
    if "n_layers" in metadata and int(metadata["n_layers"]) != n_layers:
        raise ValueError("Lens metadata n_layers differs from the loaded model")

    jacobian_dtypes: set[str] = set()
    if set(map(int, lens.jacobians)) != set(lens_layers):
        raise ValueError("Loaded lens Jacobian keys differ from its source layers")
    for layer in lens_layers:
        jacobian = lens.jacobians[layer]
        if not isinstance(jacobian, torch.Tensor):
            raise TypeError(f"Lens Jacobian at layer {layer} is not a tensor")
        if tuple(jacobian.shape) != (d_model, d_model):
            raise ValueError(
                f"Lens Jacobian at layer {layer} has shape {tuple(jacobian.shape)}, "
                f"expected {(d_model, d_model)}"
            )
        if not jacobian.dtype.is_floating_point:
            raise TypeError(f"Lens Jacobian at layer {layer} is not floating point")
        jacobian_dtypes.add(str(jacobian.dtype))

    fitted = int(metadata.get("n_prompts_fitted", -1))
    requested = int(metadata.get("n_prompts_requested", -1))
    if requested != expected_n_prompts or fitted != expected_n_prompts:
        raise ValueError(
            f"14B lens must be the {expected_n_prompts}-prompt fit, "
            f"got requested={requested}, fitted={fitted}"
        )
    if int(lens.n_prompts) != fitted:
        raise ValueError("Loaded lens n_prompts differs from fit metadata")
    if metadata.get("wikitext_revision") != WIKITEXT_REVISION:
        raise ValueError("WikiText revision differs from the pinned fit revision")
    if metadata.get("selection") != WIKITEXT_SELECTION:
        raise ValueError("WikiText prompt-selection rule differs from the frozen rule")
    prompt_hashes = metadata.get("prompt_sha256")
    if not isinstance(prompt_hashes, list) or len(prompt_hashes) != expected_n_prompts:
        raise ValueError("Lens metadata must contain one hash per fitted prompt")
    if any(
        not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
        for value in prompt_hashes
    ):
        raise ValueError("Lens prompt hashes must be lowercase SHA-256 digests")
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ValueError("Lens prompt provenance contains duplicate prompt hashes")
    for field in ("dim_batch", "max_seq_len", "checkpoint_every"):
        value = metadata.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Lens metadata field {field!r} must be positive")

    return {
        "status": "PASS",
        "model_id": bundle.model_id,
        "model_revision": bundle.revision,
        "n_layers": n_layers,
        "d_model": d_model,
        "lens_n_prompts": fitted,
        "lens_source_layers": lens_layers,
        "workspace_layers": expected_workspace,
        "lens_jacobian_dtypes": sorted(jacobian_dtypes),
        "wikitext_revision": metadata["wikitext_revision"],
        "selection": metadata["selection"],
        "n_prompt_hashes": len(prompt_hashes),
        "lens_path": _relative_or_absolute(lens_file),
        "lens_bytes": lens_file.stat().st_size,
        "lens_sha256": _sha256(lens_file),
        "metadata_path": (
            _relative_or_absolute(metadata_file) if metadata_file is not None else None
        ),
        "metadata_sha256": (
            _sha256(metadata_file) if metadata_file is not None else None
        ),
    }


def _mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    return bootstrap_statistic(
        [values],
        lambda array: float(np.mean(array)),
        n_bootstrap=n_bootstrap,
        confidence=0.95,
        seed=seed,
    )


def _probe_rows(
    bundle: Any,
    manifest: Mapping[str, Any],
    *,
    splits: set[str],
    explicit: bool,
) -> list[dict[str, Any]]:
    prompts: list[str] = []
    expected_ids: list[int] = []
    labels: list[dict[str, str]] = []
    for concept in manifest["concepts"]:
        for cue in concept["cues"]:
            if cue["split"] not in splits:
                continue
            prompts.append(
                explicit_probe_prompt(cue) if explicit else render_cue(manifest, cue)
            )
            expected_ids.append(int(concept["token_id"]))
            labels.append(
                {
                    "concept": concept["concept"],
                    "cue_id": cue["cue_id"],
                    "fact_id": cue["fact_id"],
                    "split": cue["split"],
                    "probe_type": "explicit_answer" if explicit else "silent_hold",
                }
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
    for row, label in zip(rows, labels, strict=True):
        row.update(label)
    return rows


def _concept_sign_summary(
    rows: Sequence[Mapping[str, Any]],
    fixed_layer: int,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_concept: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if int(row["layer"]) == fixed_layer:
            by_concept[str(row["concept"])].append(float(row["delta"]))
    concept_rows = [
        {
            "concept": concept,
            "mean_heldout_delta": float(np.mean(values)),
            "positive": int(float(np.mean(values)) > 0.0),
            "n_heldout_cues": len(values),
        }
        for concept, values in sorted(by_concept.items())
    ]
    positive = _mean_ci(
        [float(row["positive"]) for row in concept_rows],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    delta = _mean_ci(
        [float(row["mean_heldout_delta"]) for row in concept_rows],
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    return concept_rows, positive, delta


def _half_cpu_bank(
    bank: Mapping[str, Mapping[int, torch.Tensor]],
) -> dict[str, dict[int, torch.Tensor]]:
    return {
        concept: {
            int(layer): tensor.detach().to(device="cpu", dtype=torch.float16)
            for layer, tensor in layer_bank.items()
        }
        for concept, layer_bank in bank.items()
    }


def _alignment_summary(
    alignments: Mapping[str, Mapping[str, Mapping[int, float]]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, convention in enumerate(("raw_WU_J", "rms_gain_folded")):
        per_concept = [
            float(np.mean(list(alignments[name][convention].values())))
            for name in sorted(alignments)
        ]
        result[convention] = _mean_ci(
            per_concept,
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
    result["per_concept_layer"] = {
        name: {
            convention: {str(layer): float(value) for layer, value in values.items()}
            for convention, values in conventions.items()
        }
        for name, conventions in alignments.items()
    }
    return result


def run_qwen14b_md_phase(
    bundle: Any,
    lens: Any,
    layers: Sequence[int],
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    artifact_path: str | Path = MD_ARTIFACT_14B,
    output_path: str | Path = RAW_MD_14B,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Rebuild and validate 14B MD directions from frozen silent cues."""

    if bundle.model_id != MODEL_14B or bundle.revision != MODEL_REVISIONS[MODEL_14B]:
        raise ValueError("The 14B MD phase requires the pinned 14B model")
    layer_list = sorted(set(int(layer) for layer in layers))
    expected_layers = workspace_layers(
        bundle.lens_model.n_layers,
        lens.source_layers,
    )
    if layer_list != expected_layers:
        raise ValueError(
            f"14B MD workspace drift: got {layer_list}, expected {expected_layers}"
        )
    manifest_file = Path(manifest_path)
    manifest_hash = _sha256(manifest_file)
    if manifest_hash != EXPECTED_MD_MANIFEST_SHA256:
        raise ValueError(
            "Frozen MD cue manifest hash drift: "
            f"got {manifest_hash}, expected {EXPECTED_MD_MANIFEST_SHA256}"
        )
    manifest = load_md_manifest(manifest_file)
    cue_audit = audit_md_manifest(manifest, bundle.tokenizer)
    if cue_audit["status"] != "PASS":
        raise RuntimeError("Refusing to fit 14B MD directions after a failed cue audit")

    reference_layer = int(manifest["validation_plan"]["fixed_validation_layer"])
    fixed_layer = map_reference_layer(
        reference_layer,
        reference_n_layers=28,
        target_n_layers=int(bundle.lens_model.n_layers),
    )
    if fixed_layer not in layer_list:
        raise ValueError(
            f"Depth-mapped validation layer {fixed_layer} is outside {layer_list}"
        )
    layer_mapping = {
        "rule": (
            "round((7B fixed layer / (7B n_layers-1)) * "
            "(target n_layers-1)); fixed before 14B activations are scored"
        ),
        "reference_model": MODEL_7B,
        "reference_n_layers": 28,
        "reference_layer": reference_layer,
        "target_model": MODEL_14B,
        "target_n_layers": int(bundle.lens_model.n_layers),
        "target_layer": fixed_layer,
    }

    exclusions = baseline_exclusions(manifest)
    train_matrices = prompt_matrix_bank(
        bundle.lens_model,
        concept_prompt_sets(manifest, "train"),
        layer_list,
        batch_size=16,
    )
    heldout_matrices = prompt_matrix_bank(
        bundle.lens_model,
        concept_prompt_sets(manifest, "heldout"),
        layer_list,
        batch_size=16,
    )
    directions, train_means = mean_difference_bank_from_matrices(
        train_matrices,
        baseline_exclusions=exclusions,
        matched_prompt_slots=True,
        device="cuda",
    )
    tolerance = float(manifest["validation_plan"]["unit_norm_tolerance"])
    calibration = fit_score_calibration(
        train_matrices,
        directions,
        exclusions,
        norm_tolerance=tolerance,
    )
    retrieval = heldout_calibrated_retrieval(
        heldout_matrices,
        directions,
        calibration,
        norm_tolerance=tolerance,
        n_permutations=n_permutations,
        permutation_seed=seed,
    )
    sign_rows = heldout_matched_baseline_deltas(
        heldout_matrices,
        directions,
        exclusions,
        norm_tolerance=tolerance,
    )
    stability_rows = leave_one_train_slot_out_stability(
        train_matrices,
        directions,
        exclusions,
        norm_tolerance=tolerance,
    )

    concept_specs = {item["concept"]: item for item in manifest["concepts"]}
    token_ids = [int(item["token_id"]) for item in manifest["concepts"]]
    raw_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        layer_list,
        fold_rms_gain=False,
    )
    effective_bank = jlens_direction_bank(
        lens,
        bundle.lens_model,
        token_ids,
        layer_list,
        fold_rms_gain=True,
    )
    alignments: dict[str, dict[str, dict[int, float]]] = {}
    for concept, spec in concept_specs.items():
        token_id = int(spec["token_id"])
        alignments[concept] = {
            "raw_WU_J": cosine_alignment(directions[concept], raw_bank[token_id]),
            "rms_gain_folded": cosine_alignment(
                directions[concept], effective_bank[token_id]
            ),
        }

    explicit_rows = _probe_rows(
        bundle,
        manifest,
        splits={"heldout"},
        explicit=True,
    )
    silent_rows = _probe_rows(
        bundle,
        manifest,
        splits={"train", "heldout"},
        explicit=False,
    )
    concept_sign, sign_positive_ci, sign_delta_ci = _concept_sign_summary(
        sign_rows,
        fixed_layer,
        n_bootstrap=n_bootstrap,
        seed=seed + 10,
    )
    fixed_retrieval_rows = [
        row for row in retrieval["rows"] if int(row["layer"]) == fixed_layer
    ]
    fixed_retrieval = retrieval["by_layer"][fixed_layer]
    retrieval_top1_ci = _mean_ci(
        [float(row["top1"]) for row in fixed_retrieval_rows],
        n_bootstrap=n_bootstrap,
        seed=seed + 20,
    )
    retrieval_auroc_ci = _mean_ci(
        list(fixed_retrieval["per_concept_ovr_auroc"].values()),
        n_bootstrap=n_bootstrap,
        seed=seed + 21,
    )
    explicit_top1_ci = _mean_ci(
        [float(row["top1_correct"]) for row in explicit_rows],
        n_bootstrap=n_bootstrap,
        seed=seed + 22,
    )
    explicit_top5_ci = _mean_ci(
        [float(row["top5_correct"]) for row in explicit_rows],
        n_bootstrap=n_bootstrap,
        seed=seed + 23,
    )
    silent_top1_ci = _mean_ci(
        [float(row["top1_correct"]) for row in silent_rows],
        n_bootstrap=n_bootstrap,
        seed=seed + 24,
    )
    silent_top10_ci = _mean_ci(
        [float(row["top10_correct"]) for row in silent_rows],
        n_bootstrap=n_bootstrap,
        seed=seed + 25,
    )
    fixed_stability = [
        float(row["cosine_to_full_direction"])
        for row in stability_rows
        if int(row["layer"]) == fixed_layer
    ]
    stability_median = float(np.median(fixed_stability))
    max_norm_error = max(
        abs(float(direction.detach().float().norm()) - 1.0)
        for layer_bank in directions.values()
        for direction in layer_bank.values()
    )
    chance = 1.0 / len(directions)
    criteria = {
        "heldout_sign_fraction_at_least_0.80": (sign_positive_ci["estimate"] >= 0.80),
        "heldout_sign_mean_ci_excludes_zero": sign_delta_ci["ci_low"] > 0.0,
        "retrieval_top1_ci_above_chance": retrieval_top1_ci["ci_low"] > chance,
        "retrieval_auroc_ci_above_half": retrieval_auroc_ci["ci_low"] > 0.5,
        "retrieval_permutation_p_below_0.01": (
            fixed_retrieval["top1_fixed_label_permutation"]["p_value"] < 0.01
        ),
        "explicit_top1_at_least_0.50": explicit_top1_ci["estimate"] >= 0.50,
        "explicit_top5_at_least_0.80": explicit_top5_ci["estimate"] >= 0.80,
        "silent_top1_at_most_0.10": silent_top1_ci["estimate"] <= 0.10,
        "silent_top10_at_most_0.25": silent_top10_ci["estimate"] <= 0.25,
        "loo_median_cosine_at_least_0.70": stability_median >= 0.70,
        "unit_norm_within_tolerance": max_norm_error <= tolerance,
    }
    status = "PASS" if all(criteria.values()) else "FAIL"

    artifact_file = Path(artifact_path)
    if (
        artifact_file.resolve()
        == (ROOT / "data/directions/qwen2.5-7b_concept_vectors.pt").resolve()
    ):
        raise ValueError("Refusing to overwrite or reuse the 7B MD artifact at 14B")
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "seed": seed,
                "model_id": bundle.model_id,
                "model_revision": bundle.revision,
                "lens_n_prompts": int(lens.n_prompts),
                "workspace_layers": layer_list,
                "fixed_validation_layer": fixed_layer,
                "validation_layer_mapping": layer_mapping,
                "manifest_path": _relative_or_absolute(manifest_file),
                "manifest_sha256": manifest_hash,
                "manifest_declared_model": manifest["model"],
                "runtime_model": {
                    "id": bundle.model_id,
                    "revision": bundle.revision,
                },
                "concepts": manifest["concepts"],
                "baseline_exclusions": {
                    name: sorted(values) for name, values in exclusions.items()
                },
                "direction_formula": (
                    "normalize(mean over matched train slots of "
                    "(concept residual - mean eligible residual)); paired foil "
                    "excluded"
                ),
                "source_model_reuse": False,
            },
            "mean_difference": _half_cpu_bank(directions),
            "train_means": _half_cpu_bank(train_means),
            "train_matrices": _half_cpu_bank(train_matrices),
            "heldout_matrices": _half_cpu_bank(heldout_matrices),
        },
        artifact_file,
    )

    summary: dict[str, Any] = {
        "schema_version": "md-scale-validation-v1",
        "status": status,
        "criteria": criteria,
        "metadata": {
            "seed": seed,
            "model_id": bundle.model_id,
            "model_revision": bundle.revision,
            "lens_n_prompts": int(lens.n_prompts),
            "workspace_layers": layer_list,
            "fixed_validation_layer": fixed_layer,
            "validation_layer_mapping": layer_mapping,
            "manifest_path": _relative_or_absolute(manifest_file),
            "manifest_sha256": manifest_hash,
            "independently_rebuilt_from_runtime_activations": True,
            "uses_7b_direction_tensors": False,
            "direction_formula": (
                "matched-template one-vs-other mean difference with paired foil "
                "left out of the baseline"
            ),
        },
        "n_concepts": len(directions),
        "n_pairs": len(manifest["pairs"]),
        "n_train_cues_per_concept": manifest["selection"]["train_cues_per_concept"],
        "n_heldout_cues_per_concept": manifest["selection"]["heldout_cues_per_concept"],
        "chance_retrieval": chance,
        "cue_audit": cue_audit,
        "max_unit_norm_error": max_norm_error,
        "heldout_sign": {
            "positive_concept_fraction": sign_positive_ci,
            "mean_delta": sign_delta_ci,
            "per_concept": concept_sign,
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
            concept: {str(layer): values for layer, values in layer_bank.items()}
            for concept, layer_bank in calibration.items()
        },
        "cosine_alignment": _alignment_summary(
            alignments,
            n_bootstrap=n_bootstrap,
            seed=seed + 30,
        ),
        "artifact": _relative_or_absolute(artifact_file),
        "artifact_bytes": artifact_file.stat().st_size,
        "artifact_sha256": _sha256(artifact_file),
    }
    save_json(output_path, summary)
    print(
        f"14B MD {status}: N={len(directions)} concepts, layer={fixed_layer}, "
        f"retrieval top1={retrieval_top1_ci['estimate']:.3f}, "
        f"explicit top1={explicit_top1_ci['estimate']:.3f}, "
        f"silent top10={silent_top10_ci['estimate']:.3f}"
    )
    if status == "FAIL":
        print(
            "14B MD validation failures: "
            f"{[name for name, passed in criteria.items() if not passed]}"
        )
    del raw_bank, effective_bank, directions, train_matrices, heldout_matrices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _safe_bootstrap(
    arrays: Sequence[Sequence[float]],
    statistic: Callable[..., float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    try:
        result = bootstrap_statistic(
            arrays,
            statistic,
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            seed=seed,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return {
            "status": "NOT_ESTIMABLE",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    if any(
        not np.isfinite(float(result[field]))
        for field in ("estimate", "ci_low", "ci_high")
    ):
        return {
            "status": "NOT_ESTIMABLE",
            "error_type": "NonFiniteStatistic",
            "error": "Bootstrap estimate or interval is non-finite",
        }
    return {"status": "ESTIMATED", **result}


def _validate_analysis_vectors(vectors: Mapping[str, Any]) -> int:
    fields = (
        "item_names",
        "write_strength",
        "read_strength",
        "causal_positive_damage",
        "predicted_positive_damage",
    )
    missing = [field for field in fields if field not in vectors]
    if missing:
        raise ValueError(f"Analysis vectors are missing fields: {missing}")
    lengths = {len(vectors[field]) for field in fields}
    if len(lengths) != 1:
        raise ValueError("Analysis vector fields must have identical lengths")
    n = lengths.pop()
    if n < 1:
        raise ValueError("Analysis vectors must be nonempty")
    names = [str(value) for value in vectors["item_names"]]
    if len(names) != len(set(names)):
        raise ValueError("Analysis item names must be unique within a method")
    for field in fields[1:]:
        values = np.asarray(vectors[field], dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Analysis vector {field!r} must be finite and flat")
    return n


def _method_scale_summary(
    twohop: Mapping[str, Any],
    method: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    try:
        statistics = twohop["analyses"]["ablation"]["by_method"][method]
    except KeyError:
        return {"status": "NOT_AVAILABLE", "n": 0}
    vectors = statistics["raw_analysis_vectors"]
    n = _validate_analysis_vectors(vectors)
    if int(statistics["n"]) != n:
        raise ValueError(f"Stored N disagrees with vectors for method {method!r}")
    mean_damage = _safe_bootstrap(
        [vectors["causal_positive_damage"]],
        lambda values: float(np.mean(values)),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "status": "ESTIMATED",
        "n": n,
        "partial_correlations": statistics["partial_correlations"],
        "mean_ablation_positive_damage": mean_damage,
        "attribution_predicted_vs_real": statistics["pearson"]["predicted_vs_real"],
        "analysis_vector_definitions": statistics.get("variables", {}),
    }


def _aligned_vectors(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    _validate_analysis_vectors(first)
    _validate_analysis_vectors(second)
    first_lookup = {str(name): index for index, name in enumerate(first["item_names"])}
    second_lookup = {
        str(name): index for index, name in enumerate(second["item_names"])
    }
    names = sorted(set(first_lookup) & set(second_lookup))
    fields = (
        "write_strength",
        "read_strength",
        "causal_positive_damage",
        "predicted_positive_damage",
    )
    aligned_first = {
        field: np.asarray(
            [first[field][first_lookup[name]] for name in names], dtype=float
        )
        for field in fields
    }
    aligned_second = {
        field: np.asarray(
            [second[field][second_lookup[name]] for name in names], dtype=float
        )
        for field in fields
    }
    return names, aligned_first, aligned_second


def _paired_scale_differences(
    twohop_7b: Mapping[str, Any],
    twohop_14b: Mapping[str, Any],
    method: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    try:
        vectors_7b = twohop_7b["analyses"]["ablation"]["by_method"][method][
            "raw_analysis_vectors"
        ]
        vectors_14b = twohop_14b["analyses"]["ablation"]["by_method"][method][
            "raw_analysis_vectors"
        ]
    except KeyError:
        return {"status": "NOT_AVAILABLE", "n_common": 0}
    names, seven, fourteen = _aligned_vectors(vectors_7b, vectors_14b)
    if len(names) < 4:
        return {
            "status": "NOT_ESTIMABLE",
            "n_common": len(names),
            "common_item_names": names,
            "error": "At least four common items are required",
        }
    arrays = [
        seven["causal_positive_damage"],
        seven["read_strength"],
        seven["write_strength"],
        seven["predicted_positive_damage"],
        fourteen["causal_positive_damage"],
        fourteen["read_strength"],
        fourteen["write_strength"],
        fourteen["predicted_positive_damage"],
    ]

    def read_partial_delta(c7, r7, w7, _p7, c14, r14, w14, _p14):
        return partial_correlation(c14, r14, w14) - partial_correlation(c7, r7, w7)

    def write_partial_delta(c7, r7, w7, _p7, c14, r14, w14, _p14):
        return partial_correlation(c14, w14, r14) - partial_correlation(c7, w7, r7)

    def mean_damage_delta(c7, _r7, _w7, _p7, c14, _r14, _w14, _p14):
        return float(np.mean(c14 - c7))

    def attribution_delta(c7, _r7, _w7, p7, c14, _r14, _w14, p14):
        return pearson_r(p14, c14) - pearson_r(p7, c7)

    return {
        "status": "ESTIMATED",
        "n_common": len(names),
        "common_item_names": names,
        "pairing_rule": "same frozen two-hop item name at both scales",
        "delta_14b_minus_7b": {
            "partial_causal_read_given_write": _safe_bootstrap(
                arrays,
                read_partial_delta,
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
            "partial_causal_write_given_read": _safe_bootstrap(
                arrays,
                write_partial_delta,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
            ),
            "mean_ablation_positive_damage": _safe_bootstrap(
                arrays,
                mean_damage_delta,
                n_bootstrap=n_bootstrap,
                seed=seed + 2,
            ),
            "attribution_predicted_vs_real_r": _safe_bootstrap(
                arrays,
                attribution_delta,
                n_bootstrap=n_bootstrap,
                seed=seed + 3,
            ),
        },
    }


def _gate_summary(gates_payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = gates_payload.get("gates", gates_payload)
    g1 = gates.get("g1", {})
    g2 = gates.get("g2", {})
    g3 = gates.get("g3", {})
    return {
        "g1": {
            key: g1.get(key)
            for key in ("status", "n", "threshold_mean_kl", "max_prompt_mean_kl")
        },
        "g2": {
            key: g2.get(key)
            for key in (
                "status",
                "directional_subgate",
                "strict_criterion",
                "clean_metric",
                "min_spider_jlens_rank",
            )
        },
        "g3_attribution_validation": {
            "status": g3.get("status"),
            "n": g3.get("n"),
            "attribution_reliable": g3.get("attribution_reliable"),
            "correlation": g3.get("correlation"),
            "meaning": g3.get("meaning"),
        },
        "strict_workspace_usable": (
            g1.get("status") == "PASS" and g2.get("status") == "PASS"
        ),
        "directional_workspace_usable": (
            g1.get("status") == "PASS" and g2.get("directional_subgate") == "PASS"
        ),
    }


def _validate_scale_run(tag: str, run: Mapping[str, Any]) -> tuple[str, str]:
    if tag not in MODEL_TAGS:
        raise ValueError(f"Unsupported scale tag {tag!r}")
    expected_model = MODEL_7B if tag == "7B" else MODEL_14B
    gates_metadata = run["gates"]["metadata"]
    twohop_metadata = run["twohop"]["metadata"]
    model_id = str(gates_metadata["model_id"])
    revision = str(gates_metadata["model_revision"])
    if model_id != expected_model:
        raise ValueError(
            f"{tag} run has model {model_id!r}, expected {expected_model!r}"
        )
    if revision != MODEL_REVISIONS[expected_model]:
        raise ValueError(f"{tag} run does not use the pinned model revision")
    if twohop_metadata.get("model_id") != model_id:
        raise ValueError(f"{tag} gates and two-hop model IDs differ")
    if twohop_metadata.get("model_revision") != revision:
        raise ValueError(f"{tag} gates and two-hop revisions differ")
    if list(gates_metadata["workspace_layers"]) != list(
        twohop_metadata["workspace_layers"]
    ):
        raise ValueError(f"{tag} gates and two-hop workspaces differ")
    md_metadata = run.get("md_validation", {}).get("metadata", {})
    if md_metadata and md_metadata.get("model_id") not in (None, model_id):
        raise ValueError(f"{tag} MD validation belongs to a different model")
    return model_id, revision


def compare_scale_runs(
    scale_runs: Mapping[str, Mapping[str, Any]],
    *,
    n_bootstrap: int = 5000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compare gates, P1 statistics, ablation, attribution, and sample sizes."""

    if set(scale_runs) != set(MODEL_TAGS):
        raise ValueError(f"Scale runs must contain exactly {MODEL_TAGS}")
    models: dict[str, Any] = {}
    for scale_index, tag in enumerate(MODEL_TAGS):
        run = scale_runs[tag]
        model_id, revision = _validate_scale_run(tag, run)
        twohop = run["twohop"]
        models[tag] = {
            "model_id": model_id,
            "model_revision": revision,
            "workspace_layers": list(twohop["metadata"]["workspace_layers"]),
            "lens_provenance": run.get("lens_provenance"),
            "gates": _gate_summary(run["gates"]),
            "md_validation": {
                "status": run.get("md_validation", {}).get("status"),
                "criteria": run.get("md_validation", {}).get("criteria"),
                "n_concepts": run.get("md_validation", {}).get("n_concepts"),
                "fixed_validation_layer": run.get("md_validation", {})
                .get("metadata", {})
                .get(
                    "fixed_validation_layer",
                    run.get("md_validation", {}).get("fixed_validation_layer"),
                ),
            },
            "twohop_status": twohop["status"],
            "corpus_criterion": twohop["corpus_criterion"],
            "sample_counts": twohop["sample_counts"],
            "methods": {
                method: _method_scale_summary(
                    twohop,
                    method,
                    n_bootstrap=n_bootstrap,
                    seed=seed + scale_index * 1000 + method_index * 100,
                )
                for method_index, method in enumerate(DIRECTION_METHODS)
            },
        }

    differences = {
        method: _paired_scale_differences(
            scale_runs["7B"]["twohop"],
            scale_runs["14B"]["twohop"],
            method,
            n_bootstrap=n_bootstrap,
            seed=seed + 10_000 + method_index * 100,
        )
        for method_index, method in enumerate(DIRECTION_METHODS)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPUTED",
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "confidence": 0.95,
        "models": models,
        "paired_14b_minus_7b": differences,
        "p1_interpretation": {
            "status": "DESCRIPTIVE_ESTIMATES_ONLY",
            "reason": (
                "The preregistration did not set a numerical threshold for a "
                "'large' READ partial correlation, an 'approximately zero' WRITE "
                "partial correlation, or scale sharpening. Estimates, bootstrap "
                "CIs, and paired scale differences are reported without inventing "
                "a pass threshold."
            ),
            "read_statistic": "partial corr(CAUSAL, READ | WRITE)",
            "write_statistic": "partial corr(CAUSAL, WRITE | READ)",
            "direction_methods": list(DIRECTION_METHODS),
        },
    }


def _plot_f7_panel(
    axis: Any,
    comparison: Mapping[str, Any],
    extractor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    title: str,
    ylabel: str,
) -> None:
    labels: list[str] = []
    estimates: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []
    for tag, color in (("7B", "#7A9CC6"), ("14B", "#D0794F")):
        for method, short, hatch in (
            (PRIMARY_DIRECTION_METHOD, "J-Lens", ""),
            (MD_DIRECTION_METHOD, "MD", "//"),
        ):
            method_summary = comparison["models"][tag]["methods"][method]
            if method_summary.get("status") != "ESTIMATED":
                continue
            statistic = extractor(method_summary)
            if statistic.get("status") != "ESTIMATED":
                continue
            labels.append(f"{tag}\n{short}\nN={method_summary['n']}")
            estimates.append(float(statistic["estimate"]))
            lows.append(float(statistic["ci_low"]))
            highs.append(float(statistic["ci_high"]))
            colors.append(color)
            hatches.append(hatch)
    axis.set(title=title, ylabel=ylabel)
    axis.axhline(0.0, color="black", linewidth=1)
    if not estimates:
        axis.text(0.5, 0.5, "Not estimable", ha="center", va="center")
        return
    positions = np.arange(len(estimates))
    bars = axis.bar(positions, estimates, color=colors, edgecolor="0.25")
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    errors = np.maximum(
        0.0,
        np.vstack(
            [
                np.asarray(estimates) - np.asarray(lows),
                np.asarray(highs) - np.asarray(estimates),
            ]
        ),
    )
    axis.errorbar(
        positions,
        estimates,
        yerr=errors,
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axis.set_xticks(positions, labels)


def plot_f7_scale_comparison(
    comparison: Mapping[str, Any],
    path: str | Path = F7_PATH,
) -> Path:
    """Save F7 with 95% bootstrap CIs for both scales and direction methods."""

    if set(comparison.get("models", {})) != set(MODEL_TAGS):
        raise ValueError("F7 requires validated 7B and 14B comparison payloads")
    set_style()
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    _plot_f7_panel(
        axes[0, 0],
        comparison,
        lambda method: method["partial_correlations"]["causal_read_given_write"],
        title="P1: READ conditional on WRITE",
        ylabel="partial correlation",
    )
    _plot_f7_panel(
        axes[0, 1],
        comparison,
        lambda method: method["partial_correlations"]["causal_write_given_read"],
        title="P1: WRITE conditional on READ",
        ylabel="partial correlation",
    )
    _plot_f7_panel(
        axes[1, 0],
        comparison,
        lambda method: method["mean_ablation_positive_damage"],
        title="Real all-band ablation effect",
        ylabel="mean positive damage",
    )
    _plot_f7_panel(
        axes[1, 1],
        comparison,
        lambda method: method["attribution_predicted_vs_real"],
        title="Attribution validation",
        ylabel="Pearson r (predicted vs real)",
    )
    strict = {
        tag: comparison["models"][tag]["gates"]["strict_workspace_usable"]
        for tag in MODEL_TAGS
    }
    directional = {
        tag: comparison["models"][tag]["gates"]["directional_workspace_usable"]
        for tag in MODEL_TAGS
    }
    figure.suptitle(
        "F7 — Qwen scale comparison (95% bootstrap CIs)\n"
        f"strict G2 usable: {strict}; directional G2 usable: {directional}"
    )
    target = save_figure(figure, path)
    plt.close(figure)
    return target


def qwen32b_disk_skip_record() -> dict[str, Any]:
    """Persist the pre-measured quota reason for not downloading Qwen-32B."""

    quota_gib = 100.0
    retained_7b_14b_gib = 41.7
    projected_all_three_gib = 102.8
    incremental_32b_gib = round(
        projected_all_three_gib - retained_7b_14b_gib,
        1,
    )
    return {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "status": "SKIPPED_DISK_CONSTRAINT",
        "download_attempted": False,
        "measurement_scope": "quota-aware home/HF-cache storage preflight",
        "measured_storage_quota_gib": quota_gib,
        "measured_7b_plus_14b_weights_gib": retained_7b_14b_gib,
        "estimated_incremental_32b_weights_gib": incremental_32b_gib,
        "projected_all_three_weights_gib": projected_all_three_gib,
        "minimum_shortfall_before_working_headroom_gib": (
            projected_all_three_gib - quota_gib
        ),
        "reason": (
            "The three weight sets alone project to 102.8 GiB on a 100 GiB "
            "quota, before lens artifacts, activations, checkpoints, or temporary "
            "files. The protected HF cache is retained; no 32B download was tried."
        ),
    }


def _twohop_attribution_gate(twohop: Mapping[str, Any]) -> dict[str, Any]:
    try:
        method = twohop["analyses"]["ablation"]["by_method"][PRIMARY_DIRECTION_METHOD]
        statistic = method["pearson"]["predicted_vs_real"]
    except KeyError as error:
        return {
            "status": "FAIL",
            "meaning": "Full-core attribution validation was not estimable",
            "attribution_reliable": False,
            "n": 0,
            "error": str(error),
            "source": "twohop.analyses.ablation.jlens_raw_wu_j.predicted_vs_real",
        }
    reliable = (
        statistic.get("status") == "ESTIMATED"
        and float(statistic["estimate"]) >= 0.5
        and float(statistic["ci_low"]) > 0.0
    )
    return {
        "status": "PASS" if statistic.get("status") == "ESTIMATED" else "FAIL",
        "meaning": (
            "Validation computed on the full frozen-clean-eligible two-hop core; "
            "reliability is a measured outcome, not a pass condition"
        ),
        "attribution_reliable": reliable,
        "reliability_rule": "Pearson r >= 0.5 and 95% bootstrap CI lower bound > 0",
        "n": method["n"],
        "correlation": statistic,
        "source": "twohop.analyses.ablation.jlens_raw_wu_j.predicted_vs_real",
    }


def _curated_scale_payload(
    *,
    provenance_14b: Mapping[str, Any],
    gates_14b: Mapping[str, Any],
    md_14b: Mapping[str, Any],
    twohop_14b: Mapping[str, Any],
    comparison: Mapping[str, Any],
    f7_path: str | Path,
    skip_32b: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPUTED",
        "qwen14b_lens_provenance": dict(provenance_14b),
        "qwen14b_gates": _gate_summary(gates_14b),
        "qwen14b_md_validation": {
            "status": md_14b["status"],
            "criteria": md_14b["criteria"],
            "metadata": md_14b["metadata"],
            "n_concepts": md_14b["n_concepts"],
            "artifact": md_14b["artifact"],
        },
        "qwen14b_twohop": {
            "status": twohop_14b["status"],
            "corpus_criterion": twohop_14b["corpus_criterion"],
            "sample_counts": twohop_14b["sample_counts"],
        },
        "comparison": dict(comparison),
        "f7": _relative_or_absolute(f7_path),
        "qwen32b": dict(skip_32b),
        "raw_outputs": {
            "gates_14b": _relative_or_absolute(RAW_GATES_14B),
            "md_14b": _relative_or_absolute(RAW_MD_14B),
            "twohop_14b": _relative_or_absolute(RAW_TWOHOP_14B),
            "scale_manifest": _relative_or_absolute(RAW_SCALE),
        },
    }


def run_scale_phase(
    *,
    lens_path: str | Path = LOCAL_LENS_14B,
    lens_metadata_path: str | Path = LOCAL_LENS_14B_METADATA,
    md_artifact_path: str | Path = MD_ARTIFACT_14B,
    gates_7b_path: str | Path = RAW_GATES_7B,
    md_7b_path: str | Path = RAW_MD_7B,
    twohop_7b_path: str | Path = RAW_TWOHOP_7B,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Execute notebook 06 end to end and persist raw plus curated outputs."""

    set_seed(seed)
    seven_gates = _read_json(gates_7b_path)
    seven_md = _read_json(md_7b_path)
    seven_twohop = _read_json(twohop_7b_path)
    lens_metadata = _read_json(lens_metadata_path)

    bundle = load_model(MODEL_14B)
    try:
        lens = load_local_lens(lens_path)
        provenance = validate_qwen14b_lens_provenance(
            bundle,
            lens,
            lens_metadata,
            lens_path=lens_path,
            metadata_path=lens_metadata_path,
        )
        layers = provenance["workspace_layers"]
        gates_14b: dict[str, Any] = {
            "metadata": {
                "seed": seed,
                "model_id": bundle.model_id,
                "model_revision": bundle.revision,
                "lens_n_prompts": int(lens.n_prompts),
                "lens_source_layers": list(map(int, lens.source_layers)),
                "workspace_layers": layers,
                "effect_sign": "delta_M = M_edited - M_clean",
                "lens_provenance": provenance,
            },
            "gates": {},
        }
        gates_14b["gates"]["g1"] = run_g1(bundle)
        save_json(RAW_GATES_14B, gates_14b)
        if gates_14b["gates"]["g1"]["status"] != "PASS":
            raise RuntimeError(
                "14B G1 failed; saved the failure and refused activation experiments"
            )
        gates_14b["gates"]["g2"] = run_g2(bundle, lens, layers)
        gates_14b["metadata"]["downstream_interpretation"] = (
            "MAIN_SCALE_RESULT"
            if gates_14b["gates"]["g2"]["status"] == "PASS"
            else "DIAGNOSTIC_STRICT_G2_FAILED"
        )
        save_json(RAW_GATES_14B, gates_14b)

        md_14b = run_qwen14b_md_phase(
            bundle,
            lens,
            layers,
            artifact_path=md_artifact_path,
            output_path=RAW_MD_14B,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed,
        )
        twohop_14b = run_twohop_phase(
            bundle,
            lens,
            md_artifact_path=md_artifact_path,
            output_path=RAW_TWOHOP_14B,
            layers=layers,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        gates_14b["gates"]["g3"] = _twohop_attribution_gate(twohop_14b)
        save_json(RAW_GATES_14B, gates_14b)

        scale_runs = {
            "7B": {
                "gates": seven_gates,
                "md_validation": seven_md,
                "twohop": seven_twohop,
                "lens_provenance": {
                    "kind": "pinned_published_lens",
                    "n_prompts": seven_gates["metadata"]["lens_n_prompts"],
                    "source_layers": seven_gates["metadata"]["lens_source_layers"],
                },
            },
            "14B": {
                "gates": gates_14b,
                "md_validation": md_14b,
                "twohop": twohop_14b,
                "lens_provenance": provenance,
            },
        }
        comparison = compare_scale_runs(
            scale_runs,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        f7 = plot_f7_scale_comparison(comparison, F7_PATH)
        skip_32b = qwen32b_disk_skip_record()
        raw_manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPUTED",
            "seed": seed,
            "inputs": {
                "gates_7b": _relative_or_absolute(gates_7b_path),
                "md_7b": _relative_or_absolute(md_7b_path),
                "twohop_7b": _relative_or_absolute(twohop_7b_path),
                "lens_14b": _relative_or_absolute(lens_path),
                "lens_metadata_14b": _relative_or_absolute(lens_metadata_path),
            },
            "outputs": {
                "gates_14b": _relative_or_absolute(RAW_GATES_14B),
                "md_14b": _relative_or_absolute(RAW_MD_14B),
                "md_artifact_14b": _relative_or_absolute(md_artifact_path),
                "twohop_14b": _relative_or_absolute(RAW_TWOHOP_14B),
                "curated": _relative_or_absolute(CURATED_SCALE),
                "f7": _relative_or_absolute(f7),
            },
            "qwen14b_lens_provenance": provenance,
            "qwen14b_gate_summary": _gate_summary(gates_14b),
            "qwen14b_md_status": {
                "status": md_14b["status"],
                "criteria": md_14b["criteria"],
                "artifact": md_14b["artifact"],
            },
            "qwen14b_twohop_status": {
                "status": twohop_14b["status"],
                "sample_counts": twohop_14b["sample_counts"],
            },
            "comparison": comparison,
            "qwen32b": skip_32b,
        }
        save_json(RAW_SCALE, raw_manifest)
        curated = _curated_scale_payload(
            provenance_14b=provenance,
            gates_14b=gates_14b,
            md_14b=md_14b,
            twohop_14b=twohop_14b,
            comparison=comparison,
            f7_path=f7,
            skip_32b=skip_32b,
        )
        save_json(CURATED_SCALE, curated)
        metrics_path = ROOT / "results/metrics.json"
        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        metrics["scale_comparison"] = curated
        # Keep the same per-item contract as notebook 02.  The raw file is the
        # complete audit trail, while metrics.json must still expose every
        # model's WRITE/READ/CAUSAL/suppression values for downstream reporting.
        metrics.setdefault("twohop", {})["qwen2.5-14b-instruct"] = twohop_14b
        save_json(metrics_path, metrics)
        print(
            "NB06 COMPUTED: 7B/14B comparison saved; "
            f"14B strict G2={gates_14b['gates']['g2']['status']}, "
            f"14B MD={md_14b['status']}, 32B={skip_32b['status']}"
        )
        return raw_manifest
    finally:
        release_model(bundle)


__all__ = [
    "compare_scale_runs",
    "map_reference_layer",
    "plot_f7_scale_comparison",
    "qwen32b_disk_skip_record",
    "run_qwen14b_md_phase",
    "run_scale_phase",
    "validate_qwen14b_lens_provenance",
]

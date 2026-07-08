"""Notebook 30 driver: matched dataset construction and verification gate."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.causal_read import (
    clean_state_and_logits,
    symmetric_interchange,
    token_difference_metric,
)
from src.concept_vectors import residual_prompt_matrices
from src.data_gen import (
    G1_PROMPTS,
    build_symmetric_causal_candidates,
    tokenize_symmetric_candidate,
)
from src.jlens_iface import jlens_direction_bank, load_published_lens
from src.metrics import save_json
from src.model_utils import (
    MODEL_REVISIONS,
    batched_next_token_records,
    hf_wrapper_logit_kl,
    load_model,
    release_model,
    set_seed,
)


ROOT = Path("/home/jovyan/j-space-thoughts")
RAW_DIR = ROOT / "data/raw/v6"
RAW_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = ROOT / "results/metrics.json"
CLEAN_MANIFEST_PATH = RAW_DIR / "30_clean_read_manifest.json"
FAILED_FORMAT_PATH = RAW_DIR / "30_dataset_and_verification_attempt1_unverified.json"
FAILED_DASHBOARD_PATH = RAW_DIR / "30_dataset_and_verification_attempt2_dashboard_void.json"
FAILED_L26_CAUSAL_PATH = RAW_DIR / "31_causal_ground_truth_attempt1_l26_void.json"
FAILED_LATENT_CONTEXT_PATH = (
    RAW_DIR / "30_dataset_and_verification_attempt3_latent_context_weak.json"
)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1729
LAYERS = list(range(13, 27))
POSITION_RULE = "explicit_concept_token_in_shared_context"


def command_output(arguments: list[str]) -> str:
    return subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


set_seed(SEED)
hf_path = shutil.which("hf")
if hf_path is None:
    raise RuntimeError("Mandatory preflight failed: hf is missing")
preflight = {
    "hf_path": hf_path,
    "hf_identity": command_output([hf_path, "auth", "whoami"]),
    "gpu_csv": command_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv",
        ]
    ),
    "hf_disk": command_output(["df", "-h", str(Path.home() / ".cache/huggingface")]),
}
print(json.dumps(preflight, indent=2))

protocol = {
    "schema_version": "symmetric-causal-read-v1",
    "seed": SEED,
    "model": {
        "id": MODEL_ID,
        "revision": MODEL_REVISIONS[MODEL_ID],
        "dtype": "torch.bfloat16",
    },
    "candidate_source": "tracked reciprocal two-hop supplement",
    "candidate_count_required_min": 100,
    "calibration_group_rule": (
        "shuffle unordered concept dependency groups; take whole groups until >=24 pairs"
    ),
    "evaluation_folds": 5,
    "position_rule": POSITION_RULE,
    "layer_candidates": LAYERS,
    "layer_selection": (
        "calibration maximum median(|C_engine|)-median(|C_dashboard|), then "
        "engine median |C|, own>foil rate, and lower layer"
    ),
    "written_threshold": (
        "calibration maximum balanced accuracy with own recall>=0.80; "
        "then higher recall and lower threshold"
    ),
    "verification_gate": (
        "both engine targets clean top-1; own concept WRITTEN in both engine runs; "
        "dashboard target top-1 and concept WRITTEN in both same-context controls"
    ),
    "causal_truth": "signed symmetric full residual interchange; unclipped",
    "cheap_primary": "16-midpoint direction-defined integrated gradient",
    "go_rule": "READ_IG AUC>=0.70 and group-bootstrap CI95 lower>0.50",
}
protocol_sha = hashlib.sha256(
    json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

bundle = load_model(MODEL_ID)
if next(bundle.hf_model.parameters()).dtype != torch.bfloat16:
    raise RuntimeError("Model did not load in bfloat16")
lens = load_published_lens(MODEL_ID)
if not set(LAYERS).issubset(set(int(layer) for layer in lens.source_layers)):
    raise RuntimeError(f"Published lens does not cover {LAYERS}")
kl_records = hf_wrapper_logit_kl(bundle, G1_PROMPTS)
max_mean_kl = max(record["mean_kl"] for record in kl_records)
if len(kl_records) != 20 or max_mean_kl >= 1e-3:
    raise RuntimeError(f"HF/J-Lens logit gate failed: max mean KL={max_mean_kl}")
print(f"G1 PASS: 20 prompts, max mean KL={max_mean_kl:.3e}")

candidate_manifest = build_symmetric_causal_candidates(seed=SEED)
tokenized_pairs = []
tokenization_rejections = []
for pair in candidate_manifest["pairs"]:
    try:
        tokenized_pairs.append(tokenize_symmetric_candidate(bundle.tokenizer, pair))
    except (ValueError, IndexError) as error:
        tokenization_rejections.append(
            {
                **pair,
                "verification_status": (
                    "CALIBRATION_ONLY" if pair["split"] == "calibration" else "UNVERIFIED"
                ),
                "verification_reasons": [f"TOKENIZATION_FAILURE: {error}"],
                "engine_verified": False,
                "control_verified": False,
            }
        )
if not tokenized_pairs:
    raise RuntimeError("Every frozen candidate failed exact tokenization")
print(
    "candidate pool",
    {
        key: candidate_manifest[key]
        for key in (
            "n_candidates",
            "n_dependency_groups",
            "n_calibration_pairs",
            "n_evaluation_pairs",
        )
    },
)

engine_prompts = [
    prompt
    for pair in tokenized_pairs
    for prompt in (pair["engine_prompt_a"], pair["engine_prompt_b"])
]
engine_targets = [
    token_id
    for pair in tokenized_pairs
    for token_id in (pair["answer_a_token_id"], pair["answer_b_token_id"])
]
dashboard_prompts = [
    prompt
    for pair in tokenized_pairs
    for prompt in (pair["dashboard_prompt_a"], pair["dashboard_prompt_b"])
]
dashboard_targets = [
    pair["dashboard_token_id"] for pair in tokenized_pairs for _ in range(2)
]
engine_records = batched_next_token_records(
    bundle.hf_model,
    bundle.tokenizer,
    engine_prompts,
    engine_targets,
    batch_size=32,
    top_k=5,
)
dashboard_records = batched_next_token_records(
    bundle.hf_model,
    bundle.tokenizer,
    dashboard_prompts,
    dashboard_targets,
    batch_size=32,
    top_k=5,
)
context_prompts = [
    prompt
    for pair in tokenized_pairs
    for prompt in (pair["concept_prefix_a"], pair["concept_prefix_b"])
]
context_matrices = residual_prompt_matrices(
    bundle.lens_model,
    context_prompts,
    LAYERS,
    position=-1,
    batch_size=32,
)
concept_token_ids = sorted(
    {
        int(token_id)
        for pair in tokenized_pairs
        for token_id in (pair["concept_a_token_id"], pair["concept_b_token_id"])
    }
)
direction_bank = jlens_direction_bank(
    lens,
    bundle.lens_model,
    concept_token_ids,
    LAYERS,
    compute_device="cuda",
    output_device="cpu",
)

calibration_indices = [
    index for index, pair in enumerate(tokenized_pairs) if pair["split"] == "calibration"
]
def choose_written_threshold(own_scores: list[float], foil_scores: list[float]) -> dict:
    own_array = np.asarray(own_scores)
    foil_array = np.asarray(foil_scores)
    threshold_rows = []
    for threshold in np.unique(np.concatenate([own_array, foil_array])):
        recall = float(np.mean(own_array >= threshold))
        specificity = float(np.mean(foil_array < threshold))
        if recall < 0.80:
            continue
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "own_recall": recall,
                "foil_specificity": specificity,
                "balanced_accuracy": 0.5 * (recall + specificity),
            }
        )
    if not threshold_rows:
        raise RuntimeError("No calibration WRITTEN threshold attains recall >=0.80")
    return max(
        threshold_rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["own_recall"],
            -row["threshold"],
        ),
    )


layer_selection_rows = []
for layer in LAYERS:
    own_scores: list[float] = []
    foil_scores: list[float] = []
    matrix = context_matrices[layer]
    for index in calibration_indices:
        pair = tokenized_pairs[index]
        row_a, row_b = 2 * index, 2 * index + 1
        vector_a = direction_bank[int(pair["concept_a_token_id"])][layer]
        vector_b = direction_bank[int(pair["concept_b_token_id"])][layer]
        own_scores.extend(
            [
                float(torch.dot(matrix[row_a], vector_a)),
                float(torch.dot(matrix[row_b], vector_b)),
            ]
        )
        foil_scores.extend(
            [
                float(torch.dot(matrix[row_a], vector_b)),
                float(torch.dot(matrix[row_b], vector_a)),
            ]
        )
    margins = np.asarray(own_scores) - np.asarray(foil_scores)
    threshold_record_for_layer = choose_written_threshold(own_scores, foil_scores)
    layer_selection_rows.append(
        {
            "layer": layer,
            "n_calibration_runs": len(margins),
            "own_greater_rate": float(np.mean(margins > 0.0)),
            "median_own_minus_foil": float(np.median(margins)),
            "mean_own_minus_foil": float(np.mean(margins)),
            "calibration_own_scores": own_scores,
            "calibration_foil_scores": foil_scores,
            "threshold_record": threshold_record_for_layer,
        }
    )


def encode(prompt: str) -> torch.Tensor:
    return bundle.tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(next(bundle.hf_model.parameters()).device)


for layer_record in layer_selection_rows:
    layer = int(layer_record["layer"])
    threshold = float(layer_record["threshold_record"]["threshold"])
    matrix = context_matrices[layer]
    calibration_causal_rows = []
    for index in calibration_indices:
        pair = tokenized_pairs[index]
        row_a, row_b = 2 * index, 2 * index + 1
        vector_a = direction_bank[int(pair["concept_a_token_id"])][layer]
        vector_b = direction_bank[int(pair["concept_b_token_id"])][layer]
        written = (
            float(torch.dot(matrix[row_a], vector_a)) >= threshold
            and float(torch.dot(matrix[row_b], vector_b)) >= threshold
        )
        clean_correct = all(
            (
                int(engine_records[row_a]["rank"]) == 1,
                int(engine_records[row_b]["rank"]) == 1,
                int(dashboard_records[row_a]["rank"]) == 1,
                int(dashboard_records[row_b]["rank"]) == 1,
            )
        )
        if not written or not clean_correct:
            continue
        position_a = int(pair["intervention_position_a"])
        position_b = int(pair["intervention_position_b"])
        engine_ids_a = encode(pair["engine_prompt_a"])
        engine_ids_b = encode(pair["engine_prompt_b"])
        engine_clean_a = clean_state_and_logits(
            bundle.hf_model,
            bundle.lens_model.layers,
            engine_ids_a,
            layer,
            position=position_a,
        )
        engine_clean_b = clean_state_and_logits(
            bundle.hf_model,
            bundle.lens_model.layers,
            engine_ids_b,
            layer,
            position=position_b,
        )
        engine = symmetric_interchange(
            bundle.hf_model,
            bundle.lens_model.layers,
            engine_ids_a,
            engine_ids_b,
            engine_clean_a,
            engine_clean_b,
            token_difference_metric(
                pair["answer_a_token_id"], pair["answer_b_token_id"]
            ),
            pair_id=pair["pair_id"],
            task_kind="engine",
            layer=layer,
            variant="full_residual",
            position_a=position_a,
            position_b=position_b,
        )
        dashboard_ids_a = encode(pair["dashboard_prompt_a"])
        dashboard_ids_b = encode(pair["dashboard_prompt_b"])
        dashboard_clean_a = clean_state_and_logits(
            bundle.hf_model,
            bundle.lens_model.layers,
            dashboard_ids_a,
            layer,
            position=position_a,
        )
        dashboard_clean_b = clean_state_and_logits(
            bundle.hf_model,
            bundle.lens_model.layers,
            dashboard_ids_b,
            layer,
            position=position_b,
        )
        dashboard = symmetric_interchange(
            bundle.hf_model,
            bundle.lens_model.layers,
            dashboard_ids_a,
            dashboard_ids_b,
            dashboard_clean_a,
            dashboard_clean_b,
            token_difference_metric(
                pair["dashboard_token_id"], pair["dashboard_distractor_token_id"]
            ),
            pair_id=pair["pair_id"],
            task_kind="dashboard",
            layer=layer,
            normalization_t=engine["T"],
            variant="full_residual",
            position_a=position_a,
            position_b=position_b,
        )
        calibration_causal_rows.append(
            {
                "pair_id": pair["pair_id"],
                "engine_C": engine["C"],
                "dashboard_C": dashboard["C"],
                "engine_R_a_from_b": engine["R_a_from_b"],
                "engine_R_b_from_a": engine["R_b_from_a"],
                "dashboard_R_a_from_b": dashboard["R_a_from_b"],
                "dashboard_R_b_from_a": dashboard["R_b_from_a"],
            }
        )
    if len(calibration_causal_rows) < 10:
        raise RuntimeError(f"Layer {layer} has too few verified calibration pairs")
    engine_abs = np.abs([row["engine_C"] for row in calibration_causal_rows])
    dashboard_abs = np.abs([row["dashboard_C"] for row in calibration_causal_rows])
    layer_record["calibration_causal_rows"] = calibration_causal_rows
    layer_record["n_causal_calibration_pairs"] = len(calibration_causal_rows)
    layer_record["engine_abs_C_median"] = float(np.median(engine_abs))
    layer_record["dashboard_abs_C_median"] = float(np.median(dashboard_abs))
    layer_record["causal_separation"] = float(
        np.median(engine_abs) - np.median(dashboard_abs)
    )
    print(
        f"calibration L{layer}: n={len(calibration_causal_rows)} "
        f"median|C| engine={layer_record['engine_abs_C_median']:.4f} "
        f"dashboard={layer_record['dashboard_abs_C_median']:.4f} "
        f"separation={layer_record['causal_separation']:.4f}"
    )

selected_record = max(
    layer_selection_rows,
    key=lambda row: (
        row["causal_separation"],
        row["engine_abs_C_median"],
        row["own_greater_rate"],
        -row["layer"],
    ),
)
selected_layer = int(selected_record["layer"])
threshold_record = selected_record["threshold_record"]
written_threshold = float(threshold_record["threshold"])
calibration_own = selected_record["calibration_own_scores"]
calibration_foil = selected_record["calibration_foil_scores"]
print(
    "frozen selection",
    {
        "layer": selected_layer,
        "position_rule": POSITION_RULE,
        "calibration_causal_separation": selected_record["causal_separation"],
        **threshold_record,
    },
)

verification_rows = []
selected_context = context_matrices[selected_layer]
for index, pair in enumerate(tokenized_pairs):
    row_a, row_b = 2 * index, 2 * index + 1
    vector_a = direction_bank[int(pair["concept_a_token_id"])][selected_layer]
    vector_b = direction_bank[int(pair["concept_b_token_id"])][selected_layer]
    engine_z_a = float(torch.dot(selected_context[row_a], vector_a))
    engine_z_b = float(torch.dot(selected_context[row_b], vector_b))
    dashboard_z_a = engine_z_a
    dashboard_z_b = engine_z_b
    engine_top1_a = int(engine_records[row_a]["rank"]) == 1
    engine_top1_b = int(engine_records[row_b]["rank"]) == 1
    dashboard_top1_a = int(dashboard_records[row_a]["rank"]) == 1
    dashboard_top1_b = int(dashboard_records[row_b]["rank"]) == 1
    engine_written_a = engine_z_a >= written_threshold
    engine_written_b = engine_z_b >= written_threshold
    dashboard_written_a = dashboard_z_a >= written_threshold
    dashboard_written_b = dashboard_z_b >= written_threshold
    engine_verified = all(
        (engine_top1_a, engine_top1_b, engine_written_a, engine_written_b)
    )
    control_verified = all(
        (
            engine_verified,
            dashboard_top1_a,
            dashboard_top1_b,
            dashboard_written_a,
            dashboard_written_b,
        )
    )
    reasons = []
    checks = {
        "ENGINE_A_TARGET_NOT_TOP1": engine_top1_a,
        "ENGINE_B_TARGET_NOT_TOP1": engine_top1_b,
        "ENGINE_A_CONCEPT_NOT_WRITTEN": engine_written_a,
        "ENGINE_B_CONCEPT_NOT_WRITTEN": engine_written_b,
        "DASHBOARD_A_TARGET_NOT_TOP1": dashboard_top1_a,
        "DASHBOARD_B_TARGET_NOT_TOP1": dashboard_top1_b,
        "DASHBOARD_A_CONCEPT_NOT_WRITTEN": dashboard_written_a,
        "DASHBOARD_B_CONCEPT_NOT_WRITTEN": dashboard_written_b,
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    if pair["split"] == "calibration":
        status = "CALIBRATION_ONLY"
    else:
        status = "VERIFIED" if control_verified else "UNVERIFIED"
    verification_rows.append(
        {
            **pair,
            "verification_status": status,
            "verification_reasons": reasons,
            "engine_verified": engine_verified,
            "control_verified": control_verified,
            "engine_top1_a": engine_top1_a,
            "engine_top1_b": engine_top1_b,
            "dashboard_top1_a": dashboard_top1_a,
            "dashboard_top1_b": dashboard_top1_b,
            "engine_z_a": engine_z_a,
            "engine_z_b": engine_z_b,
            "dashboard_z_a": dashboard_z_a,
            "dashboard_z_b": dashboard_z_b,
            "written_threshold": written_threshold,
            "engine_top_token_id_a": int(engine_records[row_a]["top_tokens"][0]["token_id"]),
            "engine_top_token_id_b": int(engine_records[row_b]["top_tokens"][0]["token_id"]),
            "dashboard_top_token_id_a": int(
                dashboard_records[row_a]["top_tokens"][0]["token_id"]
            ),
            "dashboard_top_token_id_b": int(
                dashboard_records[row_b]["top_tokens"][0]["token_id"]
            ),
        }
    )
verification_rows.extend(tokenization_rejections)
verification_rows.sort(key=lambda row: row["pair_id"])

counts = {
    "candidates": len(verification_rows),
    "calibration_pairs": sum(
        row["verification_status"] == "CALIBRATION_ONLY" for row in verification_rows
    ),
    "evaluation_pairs": sum(row["split"] == "evaluation" for row in verification_rows),
    "verified_pairs": sum(
        row["verification_status"] == "VERIFIED" for row in verification_rows
    ),
    "unverified_pairs": sum(
        row["verification_status"] == "UNVERIFIED" for row in verification_rows
    ),
    "engine_verified_evaluation_pairs": sum(
        row["split"] == "evaluation" and row["engine_verified"]
        for row in verification_rows
    ),
}
print("verification counts", counts)
if counts["verified_pairs"] < 60:
    print(
        f"TARGET SHORTFALL: {counts['verified_pairs']} verified pairs; "
        "continuing with every pair that passed the frozen gate"
    )

direction_path = RAW_DIR / "30_selected_directions.pt"
torch.save(
    {
        "schema_version": "symmetric-selected-directions-v1",
        "protocol_sha256": protocol_sha,
        "model_id": MODEL_ID,
        "model_revision": bundle.revision,
        "selected_layer": selected_layer,
        "directions": {
            int(token_id): direction_bank[int(token_id)][selected_layer].cpu()
            for token_id in concept_token_ids
        },
    },
    direction_path,
)
direction_sha = hashlib.sha256(direction_path.read_bytes()).hexdigest()
clean_read_manifest = {
    "schema_version": "symmetric-clean-read-manifest-v1",
    "protocol_sha256": protocol_sha,
    "model": {
        "id": bundle.model_id,
        "revision": bundle.revision,
        "dtype": str(next(bundle.hf_model.parameters()).dtype),
    },
    "selection": {
        "layer": selected_layer,
        "position_rule": POSITION_RULE,
        "written_threshold": written_threshold,
    },
    "counts": counts,
    "rows": verification_rows,
    "direction_cache": {
        "path": str(direction_path),
        "bytes": direction_path.stat().st_size,
        "sha256": direction_sha,
    },
    "causal_interchange_outputs_included": False,
}
serialized_clean_manifest = json.dumps(clean_read_manifest, sort_keys=True)
if '"C"' in serialized_clean_manifest or "metric_a_from_b" in serialized_clean_manifest:
    raise RuntimeError("Sanitized cheap READ manifest contains causal output fields")
save_json(CLEAN_MANIFEST_PATH, clean_read_manifest)
clean_manifest_sha = hashlib.sha256(CLEAN_MANIFEST_PATH.read_bytes()).hexdigest()
raw_artifact = {
    "schema_version": "symmetric-dataset-verification-v1",
    "protocol": protocol,
    "protocol_sha256": protocol_sha,
    "preflight": preflight,
    "model": {
        "id": bundle.model_id,
        "revision": bundle.revision,
        "dtype": str(next(bundle.hf_model.parameters()).dtype),
    },
    "logit_agreement": {
        "status": "PASS",
        "threshold": 1e-3,
        "n": len(kl_records),
        "max_mean_kl": max_mean_kl,
        "rows": kl_records,
    },
    "candidate_manifest": {
        key: value for key, value in candidate_manifest.items() if key != "pairs"
    },
    "tokenization_rejections": tokenization_rejections,
    "selection": {
        "layer": selected_layer,
        "position_rule": POSITION_RULE,
        "layer_candidates": layer_selection_rows,
        "written_threshold": written_threshold,
        "threshold_record": threshold_record,
        "calibration_own_scores": calibration_own,
        "calibration_foil_scores": calibration_foil,
    },
    "counts": counts,
    "rows": verification_rows,
    "direction_cache": {
        "path": str(direction_path),
        "bytes": direction_path.stat().st_size,
        "sha256": direction_sha,
    },
    "clean_read_manifest": {
        "path": str(CLEAN_MANIFEST_PATH),
        "bytes": CLEAN_MANIFEST_PATH.stat().st_size,
        "sha256": clean_manifest_sha,
        "causal_interchange_outputs_included": False,
    },
    "prompt_format_history": [
        {
            "status": "REJECTED_UNVERIFIED",
            "reason": (
                "Initial Question:/Answer: syntax made the model's immediate "
                "next token a prose prefix rather than the declared single answer"
            ),
            "selection_policy": (
                "Replacement direct-completion syntax chosen on calibration groups; "
                "no causal or READ outcome existed"
            ),
            "artifact_path": str(FAILED_FORMAT_PATH),
            "artifact_sha256": hashlib.sha256(FAILED_FORMAT_PATH.read_bytes()).hexdigest(),
            "counts": json.loads(FAILED_FORMAT_PATH.read_text())["counts"],
        },
        {
            "status": "REJECTED_DASHBOARD_CONTROL_VOID",
            "reason": (
                "Final-answer-token measurement displaced the latent concept in "
                "both arithmetic controls despite restored clean answers"
            ),
            "selection_policy": (
                "Semantic shared-context boundary fixed before any causal or READ outcome"
            ),
            "artifact_path": str(FAILED_DASHBOARD_PATH),
            "artifact_sha256": hashlib.sha256(
                FAILED_DASHBOARD_PATH.read_bytes()
            ).hexdigest(),
            "counts": json.loads(FAILED_DASHBOARD_PATH.read_text())["counts"],
        },
    ],
    "ground_truth_instrument_history": [
        {
            "status": "REJECTED_VOID_CAUSAL_INSTRUMENT",
            "layer": 26,
            "engine_abs_C_median": 0.002173954139019641,
            "dashboard_abs_C_median": 0.0,
            "reason": "WRITTEN-optimal layer left only one downstream block",
            "artifact_path": str(FAILED_L26_CAUSAL_PATH),
            "artifact_sha256": hashlib.sha256(
                FAILED_L26_CAUSAL_PATH.read_bytes()
            ).hexdigest(),
            "cheap_read_values_existed": False,
        },
        {
            "status": "REJECTED_WEAK_LATENT_CONTEXT_INTERCHANGE",
            "selected_layer": 22,
            "calibration_engine_abs_C_median": 0.0076,
            "calibration_dashboard_abs_C_median": 0.0039,
            "reason": (
                "No L13-L26 single latent-context-boundary state produced a "
                "task-selective engine-large/dashboard-zero causal instrument"
            ),
            "artifact_path": str(FAILED_LATENT_CONTEXT_PATH),
            "artifact_sha256": hashlib.sha256(
                FAILED_LATENT_CONTEXT_PATH.read_bytes()
            ).hexdigest(),
            "cheap_read_values_existed": False,
        },
    ],
}
raw_path = RAW_DIR / "30_dataset_and_verification.json"
save_json(raw_path, raw_artifact)
raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

metrics = json.loads(METRICS_PATH.read_text())
metrics["symmetric_causal_read_v6"] = {
    "schema_version": "symmetric-causal-read-v1",
    "status": "PHASE_1_DATASET_VERIFIED",
    "protocol": protocol,
    "protocol_sha256": protocol_sha,
    "preflight": preflight,
    "stage30": {
        "status": "COMPLETE",
        "counts": counts,
        "selection": raw_artifact["selection"],
        "logit_agreement": raw_artifact["logit_agreement"],
        "prompt_format_history": raw_artifact["prompt_format_history"],
        "verification_rows": verification_rows,
        "raw_artifact": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": raw_sha,
        },
        "direction_cache": raw_artifact["direction_cache"],
        "clean_read_manifest": raw_artifact["clean_read_manifest"],
        "ground_truth_instrument_history": raw_artifact[
            "ground_truth_instrument_history"
        ],
    },
}
save_json(METRICS_PATH, metrics)
print(
    json.dumps(
        {
            "raw_sha256": raw_sha,
            "direction_sha256": direction_sha,
            "clean_manifest_sha256": clean_manifest_sha,
        },
        indent=2,
    )
)

del context_matrices, direction_bank, lens
release_model(bundle)
gc.collect()
torch.cuda.empty_cache()

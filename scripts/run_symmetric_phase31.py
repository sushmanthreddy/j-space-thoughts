"""Notebook 31 driver: expensive symmetric causal ground truth."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.causal_read import (
    clean_state_and_logits,
    symmetric_interchange,
    token_difference_metric,
)
from src.metrics import save_json
from src.model_utils import load_model, release_model, set_seed


ROOT = Path("/home/jovyan/j-space-thoughts")
RAW_DIR = ROOT / "data/raw/v6"
METRICS_PATH = ROOT / "results/metrics.json"
VERIFY_PATH = RAW_DIR / "30_dataset_and_verification.json"
DIRECTION_PATH = RAW_DIR / "30_selected_directions.pt"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1729


set_seed(SEED)
verification = json.loads(VERIFY_PATH.read_text())
direction_payload = torch.load(DIRECTION_PATH, map_location="cpu", weights_only=False)
if direction_payload["protocol_sha256"] != verification["protocol_sha256"]:
    raise RuntimeError("Direction cache and verification protocol differ")
if hashlib.sha256(DIRECTION_PATH.read_bytes()).hexdigest() != verification[
    "direction_cache"
]["sha256"]:
    raise RuntimeError("Selected direction cache hash changed")
selected_layer = int(verification["selection"]["layer"])
position_rule = str(verification["selection"]["position_rule"])
verified_pairs = [
    row for row in verification["rows"] if row["verification_status"] == "VERIFIED"
]
if not verified_pairs:
    raise RuntimeError("No verified held-out pairs are available for causal truth")
print(f"computing causal truth for {len(verified_pairs)} verified pairs")

bundle = load_model(MODEL_ID)
if bundle.revision != verification["model"]["revision"]:
    raise RuntimeError("Model revision changed after verification")
directions = direction_payload["directions"]


def encode(prompt: str) -> torch.Tensor:
    return bundle.tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(next(bundle.hf_model.parameters()).device)


rows = []
for index, pair in enumerate(verified_pairs):
    position_a = int(pair["context_position_a"])
    position_b = int(pair["context_position_b"])
    vector_a = directions[int(pair["concept_a_token_id"])]
    vector_b = directions[int(pair["concept_b_token_id"])]
    engine_ids_a = encode(pair["engine_prompt_a"])
    engine_ids_b = encode(pair["engine_prompt_b"])
    engine_metric = token_difference_metric(
        pair["answer_a_token_id"], pair["answer_b_token_id"]
    )
    engine_clean_a = clean_state_and_logits(
        bundle.hf_model,
        bundle.lens_model.layers,
        engine_ids_a,
        selected_layer,
        position=position_a,
    )
    engine_clean_b = clean_state_and_logits(
        bundle.hf_model,
        bundle.lens_model.layers,
        engine_ids_b,
        selected_layer,
        position=position_b,
    )
    engine_full = symmetric_interchange(
        bundle.hf_model,
        bundle.lens_model.layers,
        engine_ids_a,
        engine_ids_b,
        engine_clean_a,
        engine_clean_b,
        engine_metric,
        pair_id=pair["pair_id"],
        task_kind="engine",
        layer=selected_layer,
        position_a=position_a,
        position_b=position_b,
        variant="full_residual",
    )
    engine_subspace = symmetric_interchange(
        bundle.hf_model,
        bundle.lens_model.layers,
        engine_ids_a,
        engine_ids_b,
        engine_clean_a,
        engine_clean_b,
        engine_metric,
        pair_id=pair["pair_id"],
        task_kind="engine",
        layer=selected_layer,
        position_a=position_a,
        position_b=position_b,
        variant="jlens_two_concept_subspace",
        direction_a=vector_a,
        direction_b=vector_b,
    )

    dashboard_ids_a = encode(pair["dashboard_prompt_a"])
    dashboard_ids_b = encode(pair["dashboard_prompt_b"])
    dashboard_metric = token_difference_metric(
        pair["dashboard_token_id"], pair["dashboard_distractor_token_id"]
    )
    dashboard_clean_a = clean_state_and_logits(
        bundle.hf_model,
        bundle.lens_model.layers,
        dashboard_ids_a,
        selected_layer,
        position=position_a,
    )
    dashboard_clean_b = clean_state_and_logits(
        bundle.hf_model,
        bundle.lens_model.layers,
        dashboard_ids_b,
        selected_layer,
        position=position_b,
    )
    dashboard_full = symmetric_interchange(
        bundle.hf_model,
        bundle.lens_model.layers,
        dashboard_ids_a,
        dashboard_ids_b,
        dashboard_clean_a,
        dashboard_clean_b,
        dashboard_metric,
        pair_id=pair["pair_id"],
        task_kind="dashboard",
        layer=selected_layer,
        position_a=position_a,
        position_b=position_b,
        normalization_t=engine_full["T"],
        variant="full_residual",
    )
    dashboard_subspace = symmetric_interchange(
        bundle.hf_model,
        bundle.lens_model.layers,
        dashboard_ids_a,
        dashboard_ids_b,
        dashboard_clean_a,
        dashboard_clean_b,
        dashboard_metric,
        pair_id=pair["pair_id"],
        task_kind="dashboard",
        layer=selected_layer,
        position_a=position_a,
        position_b=position_b,
        normalization_t=engine_full["T"],
        variant="jlens_two_concept_subspace",
        direction_a=vector_a,
        direction_b=vector_b,
    )
    rows.append(
        {
            "pair_id": pair["pair_id"],
            "dependency_group": pair["dependency_group"],
            "fold": pair["fold"],
            "category": pair["category"],
            "concept_a": pair["concept_a"],
            "concept_b": pair["concept_b"],
            "answer_a": pair["answer_a"],
            "answer_b": pair["answer_b"],
            "engine": {
                "full_residual": engine_full,
                "jlens_two_concept_subspace": engine_subspace,
            },
            "dashboard": {
                "full_residual": dashboard_full,
                "jlens_two_concept_subspace": dashboard_subspace,
            },
        }
    )
    if index < 5 or (index + 1) % 10 == 0 or index + 1 == len(verified_pairs):
        print(
            f"[{index + 1:03d}/{len(verified_pairs):03d}] {pair['pair_id']} "
            f"C_engine={engine_full['C']:.4f} C_dashboard={dashboard_full['C']:.4f} "
            f"direction_diff=({engine_full['directional_abs_difference']:.3f},"
            f"{dashboard_full['directional_abs_difference']:.3f})"
        )
    del (
        engine_clean_a,
        engine_clean_b,
        dashboard_clean_a,
        dashboard_clean_b,
    )
    if (index + 1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()

engine_c = np.asarray([row["engine"]["full_residual"]["C"] for row in rows])
dashboard_c = np.asarray([row["dashboard"]["full_residual"]["C"] for row in rows])
sanity = {
    "n_pairs": len(rows),
    "engine_C_median": float(np.median(engine_c)),
    "engine_abs_C_median": float(np.median(np.abs(engine_c))),
    "dashboard_C_median": float(np.median(dashboard_c)),
    "dashboard_abs_C_median": float(np.median(np.abs(dashboard_c))),
    "engine_sharp_directional_disagreement": int(
        sum(row["engine"]["full_residual"]["sharp_directional_disagreement"] for row in rows)
    ),
    "dashboard_sharp_directional_disagreement": int(
        sum(
            row["dashboard"]["full_residual"]["sharp_directional_disagreement"]
            for row in rows
        )
    ),
}
print("engine/dashboard causal sanity", json.dumps(sanity, indent=2))

artifact = {
    "schema_version": "symmetric-causal-ground-truth-v1",
    "protocol_sha256": verification["protocol_sha256"],
    "upstream_verification_sha256": hashlib.sha256(VERIFY_PATH.read_bytes()).hexdigest(),
    "model": verification["model"],
    "selected_layer": selected_layer,
    "position_rule": position_rule,
    "primary_truth": "full_residual",
    "signed_unclipped": True,
    "sanity": sanity,
    "rows": rows,
}
raw_path = RAW_DIR / "31_causal_ground_truth.json"
save_json(raw_path, artifact)
raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

metrics = json.loads(METRICS_PATH.read_text())
run = metrics["symmetric_causal_read_v6"]
if run["protocol_sha256"] != verification["protocol_sha256"]:
    raise RuntimeError("Metrics protocol changed after dataset verification")
run["status"] = "PHASE_1_COMPLETE"
run["stage31"] = {
    "status": "COMPLETE",
    "primary_truth": "full_residual",
    "signed_unclipped": True,
    "sanity": sanity,
    "rows": rows,
    "raw_artifact": {
        "path": str(raw_path),
        "bytes": raw_path.stat().st_size,
        "sha256": raw_sha,
    },
}
save_json(METRICS_PATH, metrics)
print(json.dumps({"raw_path": str(raw_path), "sha256": raw_sha}, indent=2))

release_model(bundle)
gc.collect()
torch.cuda.empty_cache()

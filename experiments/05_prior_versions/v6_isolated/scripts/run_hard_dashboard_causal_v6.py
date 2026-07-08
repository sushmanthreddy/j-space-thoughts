"""Compute expensive causal C for verified hard dashboards only."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch

from src.causal_read import (
    clean_state_and_logits,
    symmetric_interchange,
    token_difference_metric,
)
from src.model_utils import load_model, release_model, set_seed


ROOT = Path("/home/jovyan/j-space-thoughts")
HARD_MANIFEST_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_manifest.json"
FROZEN_CAUSAL_PATH = ROOT / "data/raw/v6/31_causal_ground_truth.json"
FROZEN_METRICS_PATH = ROOT / "results/metrics.json"
OUTPUT_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_causal.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1729


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


hard_manifest = json.loads(HARD_MANIFEST_PATH.read_text())
frozen_causal = json.loads(FROZEN_CAUSAL_PATH.read_text())
frozen_metrics = json.loads(FROZEN_METRICS_PATH.read_text())["symmetric_causal_read_v6"]
expected_causal_sha = frozen_metrics["stage31"]["raw_artifact"]["sha256"]
if sha256(FROZEN_CAUSAL_PATH) != expected_causal_sha:
    raise RuntimeError("Frozen causal ground-truth artifact hash changed")
if frozen_causal["protocol_sha256"] != hard_manifest["source_clean_manifest"][
    "protocol_sha256"
]:
    raise RuntimeError("Hard dashboards and frozen causal truth use different protocols")

hard_rows = [
    row for row in hard_manifest["rows"] if row["verification_status"] == "VERIFIED_HARD"
]
frozen_by_pair = {row["pair_id"]: row for row in frozen_causal["rows"]}
if not hard_rows or not {row["pair_id"] for row in hard_rows} <= set(frozen_by_pair):
    raise RuntimeError("Hard-dashboard pairs do not match frozen causal truth")

set_seed(SEED)
bundle = load_model(MODEL_ID)
if bundle.revision != hard_manifest["model"]["revision"]:
    raise RuntimeError("Pinned model revision differs from hard verification")
selected_layer = int(hard_manifest["selection"]["layer"])
device = next(bundle.hf_model.parameters()).device


def encode(prompt: str) -> torch.Tensor:
    return bundle.tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)


rows: list[dict] = []
for index, row in enumerate(hard_rows):
    input_ids_a = encode(row["hard_prompt_a"])
    input_ids_b = encode(row["hard_prompt_b"])
    position_a = int(row["intervention_position_a"])
    position_b = int(row["intervention_position_b"])
    clean_a = clean_state_and_logits(
        bundle.hf_model,
        bundle.lens_model.layers,
        input_ids_a,
        selected_layer,
        position=position_a,
    )
    clean_b = clean_state_and_logits(
        bundle.hf_model,
        bundle.lens_model.layers,
        input_ids_b,
        selected_layer,
        position=position_b,
    )
    metric_fn = token_difference_metric(
        int(row["hard_target_token_id"]), int(row["hard_distractor_token_id"])
    )
    frozen_engine = frozen_by_pair[row["pair_id"]]["engine"]["full_residual"]
    hard_causal = symmetric_interchange(
        bundle.hf_model,
        bundle.lens_model.layers,
        input_ids_a,
        input_ids_b,
        clean_a,
        clean_b,
        metric_fn,
        pair_id=str(row["pair_id"]),
        task_kind="dashboard",
        layer=selected_layer,
        normalization_t=float(frozen_engine["T"]),
        variant="full_residual",
        position_a=position_a,
        position_b=position_b,
    )
    if hard_causal["clean_top_token_id_a"] != int(row["hard_target_token_id"]):
        raise RuntimeError(f"Hard A verification drifted for {row['pair_id']}")
    if hard_causal["clean_top_token_id_b"] != int(row["hard_target_token_id"]):
        raise RuntimeError(f"Hard B verification drifted for {row['pair_id']}")
    if not hard_causal["signed_unclipped"]:
        raise RuntimeError("Hard causal result is not signed and unclipped")
    rows.append(
        {
            "pair_id": row["pair_id"],
            "dependency_group": row["dependency_group"],
            "fold": int(row["fold"]),
            "category": row["category"],
            "hard_template_id": row["hard_template_id"],
            "frozen_engine_C": float(frozen_engine["C"]),
            "frozen_engine_T": float(frozen_engine["T"]),
            "hard_dashboard": hard_causal,
        }
    )
    if index < 5 or (index + 1) % 10 == 0 or index + 1 == len(hard_rows):
        print(
            f"[{index + 1:03d}/{len(hard_rows):03d}] {row['pair_id']} "
            f"frozen engine C={frozen_engine['C']:.4f} "
            f"hard dashboard C={hard_causal['C']:.4f}"
        )
    del clean_a, clean_b, input_ids_a, input_ids_b
    if (index + 1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()

artifact = {
    "schema_version": "read-stress-v6-hard-dashboard-causal-v1",
    "model": hard_manifest["model"],
    "selected_layer": selected_layer,
    "position_rule": hard_manifest["selection"]["position_rule"],
    "source_hard_manifest": {
        "path": str(HARD_MANIFEST_PATH),
        "sha256": sha256(HARD_MANIFEST_PATH),
    },
    "frozen_causal_truth": {
        "path": str(FROZEN_CAUSAL_PATH),
        "sha256": sha256(FROZEN_CAUSAL_PATH),
        "engine_recomputed": False,
    },
    "rows": rows,
}
save_json(OUTPUT_PATH, artifact)
print("hard-dashboard causal artifact", OUTPUT_PATH, sha256(OUTPUT_PATH))

release_model(bundle)
del bundle
gc.collect()
torch.cuda.empty_cache()

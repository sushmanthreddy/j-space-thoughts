"""Notebook 32 driver: gradient-only cheap READ estimators."""

from __future__ import annotations

import ast
import gc
import hashlib
import json
from pathlib import Path

import torch

from src.cheap_read import (
    batch_token_difference_metric,
    symmetric_gradient_read,
    weight_norm_capacity_baseline,
)
from src.metrics import save_json
from src.model_utils import load_model, release_model, set_seed


ROOT = Path("/home/jovyan/j-space-thoughts")
RAW_DIR = ROOT / "data/raw/v6"
METRICS_PATH = ROOT / "results/metrics.json"
CLEAN_MANIFEST_PATH = RAW_DIR / "30_clean_read_manifest.json"
DIRECTION_PATH = RAW_DIR / "30_selected_directions.pt"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1729
IG_STEPS = 16


cheap_source_path = ROOT / "src/cheap_read.py"
cheap_source = cheap_source_path.read_text()
tree = ast.parse(cheap_source)
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        imports.append(node.module)
forbidden_import_roots = {
    "src.causal_read",
    "src.interventions",
    "src.read_scores",
    "src.read_validation",
}
forbidden_found = sorted(
    name for name in imports if any(name.startswith(root) for root in forbidden_import_roots)
)
if forbidden_found:
    raise RuntimeError(f"Anti-circularity import audit failed: {forbidden_found}")
driver_source = (ROOT / "scripts/run_symmetric_phase32.py").read_text()
forbidden_artifact_reference = "31_" + "causal_" + "ground_truth"
if forbidden_artifact_reference in driver_source:
    raise RuntimeError("Cheap READ driver references the causal ground-truth artifact")
anti_circularity_audit = {
    "status": "PASS",
    "cheap_module_imports": imports,
    "forbidden_imports_found": forbidden_found,
    "causal_artifact_path_referenced": False,
    "causal_outputs_consumed": False,
}
print("anti-circularity audit", json.dumps(anti_circularity_audit, indent=2))

set_seed(SEED)
verification = json.loads(CLEAN_MANIFEST_PATH.read_text())
if verification.get("causal_interchange_outputs_included") is not False:
    raise RuntimeError("Cheap READ manifest is not certified causal-output-free")
direction_payload = torch.load(DIRECTION_PATH, map_location="cpu", weights_only=False)
if direction_payload["protocol_sha256"] != verification["protocol_sha256"]:
    raise RuntimeError("Direction cache and verification protocol differ")
selected_layer = int(verification["selection"]["layer"])
position_rule = str(verification["selection"]["position_rule"])
verified_pairs = [
    row for row in verification["rows"] if row["verification_status"] == "VERIFIED"
]
if not verified_pairs:
    raise RuntimeError("No verified pairs are available for cheap READ")

bundle = load_model(MODEL_ID)
if any(parameter.requires_grad for parameter in bundle.hf_model.parameters()):
    raise RuntimeError("J-Lens wrapper did not freeze model parameters")
directions = direction_payload["directions"]
bundle.tokenizer.padding_side = "left"


def encode_pair(
    prompt_a: str,
    prompt_b: str,
    context_position_a: int,
    context_position_b: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    encoded = bundle.tokenizer(
        [prompt_a, prompt_b],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    if not bool(torch.all(encoded.attention_mask[:, -1] == 1)):
        raise RuntimeError("Left-padded pair does not end in real tokens")
    left_padding = encoded.input_ids.shape[1] - encoded.attention_mask.sum(dim=1)
    positions = [
        int(left_padding[0]) + int(context_position_a),
        int(left_padding[1]) + int(context_position_b),
    ]
    device = next(bundle.hf_model.parameters()).device
    return (
        encoded.input_ids.to(device),
        encoded.attention_mask.to(device),
        positions,
    )


raw_rows = []
compact_rows = []
for index, pair in enumerate(verified_pairs):
    vector_a = directions[int(pair["concept_a_token_id"])]
    vector_b = directions[int(pair["concept_b_token_id"])]
    engine_ids, engine_mask, engine_positions = encode_pair(
        pair["engine_prompt_a"],
        pair["engine_prompt_b"],
        pair["intervention_position_a"],
        pair["intervention_position_b"],
    )
    engine_metric = batch_token_difference_metric(
        pair["answer_a_token_id"], pair["answer_b_token_id"]
    )
    engine_read = symmetric_gradient_read(
        bundle.hf_model,
        bundle.lens_model.layers,
        engine_ids,
        engine_metric,
        vector_a,
        vector_b,
        layer=selected_layer,
        position=engine_positions,
        ig_steps=IG_STEPS,
        attention_mask=engine_mask,
    )
    dashboard_ids, dashboard_mask, dashboard_positions = encode_pair(
        pair["dashboard_prompt_a"],
        pair["dashboard_prompt_b"],
        pair["intervention_position_a"],
        pair["intervention_position_b"],
    )
    dashboard_metric = batch_token_difference_metric(
        pair["dashboard_token_id"], pair["dashboard_distractor_token_id"]
    )
    dashboard_read = symmetric_gradient_read(
        bundle.hf_model,
        bundle.lens_model.layers,
        dashboard_ids,
        dashboard_metric,
        vector_a,
        vector_b,
        layer=selected_layer,
        position=dashboard_positions,
        ig_steps=IG_STEPS,
        attention_mask=dashboard_mask,
    )
    baseline = weight_norm_capacity_baseline(
        bundle.lens_model.layers[selected_layer], vector_a, vector_b
    )
    metadata = {
        "pair_id": pair["pair_id"],
        "dependency_group": pair["dependency_group"],
        "fold": pair["fold"],
        "category": pair["category"],
        "concept_a": pair["concept_a"],
        "concept_b": pair["concept_b"],
    }
    raw_rows.append(
        {
            **metadata,
            "engine": engine_read,
            "dashboard": dashboard_read,
            "weight_norm_capacity_baseline": baseline,
        }
    )
    compact_rows.append(
        {
            **metadata,
            "engine": {
                "READ_IG": engine_read["READ_IG"],
                "READ_local": engine_read["READ_local"],
                "ig_abs_by_direction": engine_read["ig_abs_by_direction"],
                "local_abs_by_direction": engine_read["local_abs_by_direction"],
            },
            "dashboard": {
                "READ_IG": dashboard_read["READ_IG"],
                "READ_local": dashboard_read["READ_local"],
                "ig_abs_by_direction": dashboard_read["ig_abs_by_direction"],
                "local_abs_by_direction": dashboard_read["local_abs_by_direction"],
            },
            "weight_norm_baseline": baseline["weight_norm_baseline"],
            "baseline_label": baseline["baseline"],
        }
    )
    if index < 5 or (index + 1) % 10 == 0 or index + 1 == len(verified_pairs):
        print(
            f"[{index + 1:03d}/{len(verified_pairs):03d}] {pair['pair_id']} "
            f"IG=({engine_read['READ_IG']:.4f},{dashboard_read['READ_IG']:.4f}) "
            f"local=({engine_read['READ_local']:.4f},{dashboard_read['READ_local']:.4f})"
        )
    del engine_read, dashboard_read, engine_ids, dashboard_ids
    if (index + 1) % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()

artifact = {
    "schema_version": "symmetric-cheap-read-v1",
    "protocol_sha256": verification["protocol_sha256"],
    "upstream_clean_manifest_sha256": hashlib.sha256(
        CLEAN_MANIFEST_PATH.read_bytes()
    ).hexdigest(),
    "model": verification["model"],
    "selected_layer": selected_layer,
    "position_rule": position_rule,
    "ig_steps": IG_STEPS,
    "anti_circularity_audit": anti_circularity_audit,
    "rows": raw_rows,
}
raw_path = RAW_DIR / "32_cheap_read.json"
save_json(raw_path, artifact)
raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

metrics = json.loads(METRICS_PATH.read_text())
run = metrics["symmetric_causal_read_v6"]
if run["protocol_sha256"] != verification["protocol_sha256"]:
    raise RuntimeError("Metrics protocol changed after dataset verification")
run["status"] = "PHASE_2_COMPLETE"
run["stage32"] = {
    "status": "COMPLETE",
    "anti_circularity_audit": anti_circularity_audit,
    "ig_steps": IG_STEPS,
    "rows": compact_rows,
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

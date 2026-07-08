"""Compute hard-dashboard READ through the frozen cheap-only code path."""

from __future__ import annotations

import ast
import gc
import hashlib
import json
from pathlib import Path

import torch

from src.cheap_read import batch_token_difference_metric, symmetric_gradient_read
from src.model_utils import load_model, release_model, set_seed


ROOT = Path("/home/jovyan/j-space-thoughts")
HARD_MANIFEST_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_manifest.json"
DIRECTION_PATH = ROOT / "data/raw/v6/30_selected_directions.pt"
OUTPUT_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_cheap.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1729
IG_STEPS = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(all_keys(child))
    return keys


forbidden_import_roots = {
    "src.causal_read",
    "src.interventions",
    "src.read_scores",
    "src.read_validation",
}
audited_files = [ROOT / "src/cheap_read.py", Path(__file__).resolve()]
imports_by_file: dict[str, list[str]] = {}
for source_path in audited_files:
    tree = ast.parse(source_path.read_text())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    forbidden = sorted(
        name
        for name in imports
        if any(name.startswith(root) for root in forbidden_import_roots)
    )
    if forbidden:
        raise RuntimeError(f"Cheap-path forbidden imports in {source_path}: {forbidden}")
    imports_by_file[str(source_path.relative_to(ROOT))] = imports

hard_manifest = json.loads(HARD_MANIFEST_PATH.read_text())
if hard_manifest.get("causal_interchange_outputs_included") is not False:
    raise RuntimeError("Hard-dashboard manifest is not certified causal-output-free")
if hard_manifest.get("edited_metrics_included") is not False:
    raise RuntimeError("Hard-dashboard manifest contains edited metrics")
forbidden_keys = {
    "C",
    "R_a_from_b",
    "R_b_from_a",
    "metric_a_from_b",
    "metric_b_from_a",
}
present_forbidden_keys = sorted(all_keys(hard_manifest) & forbidden_keys)
if present_forbidden_keys:
    raise RuntimeError(f"Sanitized hard manifest contains {present_forbidden_keys}")
if sha256(DIRECTION_PATH) != hard_manifest["direction_cache"]["sha256"]:
    raise RuntimeError("Frozen direction-cache hash changed")

direction_payload = torch.load(DIRECTION_PATH, map_location="cpu", weights_only=False)
if direction_payload["protocol_sha256"] != hard_manifest["source_clean_manifest"][
    "protocol_sha256"
]:
    raise RuntimeError("Hard manifest and frozen directions use different protocols")

verified_rows = [
    row for row in hard_manifest["rows"] if row["verification_status"] == "VERIFIED_HARD"
]
if not verified_rows:
    raise RuntimeError("No verified hard dashboards are available for READ")

set_seed(SEED)
bundle = load_model(MODEL_ID)
if any(parameter.requires_grad for parameter in bundle.hf_model.parameters()):
    raise RuntimeError("Frozen model parameters unexpectedly require gradients")
if bundle.revision != hard_manifest["model"]["revision"]:
    raise RuntimeError("Pinned model revision differs from hard verification")
bundle.tokenizer.padding_side = "left"
directions = direction_payload["directions"]
selected_layer = int(hard_manifest["selection"]["layer"])


def encode_pair(row: dict) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    encoded = bundle.tokenizer(
        [row["hard_prompt_a"], row["hard_prompt_b"]],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    if not bool(torch.all(encoded.attention_mask[:, -1] == 1)):
        raise RuntimeError("Left-padded hard prompt does not end in a real token")
    left_padding = encoded.input_ids.shape[1] - encoded.attention_mask.sum(dim=1)
    positions = [
        int(left_padding[0]) + int(row["intervention_position_a"]),
        int(left_padding[1]) + int(row["intervention_position_b"]),
    ]
    device = next(bundle.hf_model.parameters()).device
    return (
        encoded.input_ids.to(device),
        encoded.attention_mask.to(device),
        positions,
    )


raw_rows: list[dict] = []
compact_rows: list[dict] = []
for index, row in enumerate(verified_rows):
    input_ids, attention_mask, positions = encode_pair(row)
    metric_fn = batch_token_difference_metric(
        int(row["hard_target_token_id"]), int(row["hard_distractor_token_id"])
    )
    estimate = symmetric_gradient_read(
        bundle.hf_model,
        bundle.lens_model.layers,
        input_ids,
        metric_fn,
        directions[int(row["concept_a_token_id"])],
        directions[int(row["concept_b_token_id"])],
        layer=selected_layer,
        position=positions,
        ig_steps=IG_STEPS,
        attention_mask=attention_mask,
    )
    if estimate.get("causal_outputs_consumed") is not False:
        raise RuntimeError("Frozen cheap estimator did not certify causal isolation")
    metadata = {
        "pair_id": row["pair_id"],
        "dependency_group": row["dependency_group"],
        "fold": int(row["fold"]),
        "category": row["category"],
        "hard_template_id": row["hard_template_id"],
    }
    raw_rows.append({**metadata, "hard_dashboard": estimate})
    compact_rows.append(
        {
            **metadata,
            "READ_IG": float(estimate["READ_IG"]),
            "READ_local": float(estimate["READ_local"]),
            "ig_abs_by_direction": estimate["ig_abs_by_direction"],
            "local_abs_by_direction": estimate["local_abs_by_direction"],
        }
    )
    if index < 5 or (index + 1) % 10 == 0 or index + 1 == len(verified_rows):
        print(
            f"[{index + 1:03d}/{len(verified_rows):03d}] {row['pair_id']} "
            f"hard READ_IG={estimate['READ_IG']:.6f} "
            f"READ_local={estimate['READ_local']:.6f}"
        )
    del estimate, input_ids, attention_mask
    if (index + 1) % 5 == 0:
        gc.collect()
        torch.cuda.empty_cache()

artifact = {
    "schema_version": "read-stress-v6-hard-dashboard-cheap-v1",
    "model": hard_manifest["model"],
    "selected_layer": selected_layer,
    "position_rule": hard_manifest["selection"]["position_rule"],
    "ig_steps": IG_STEPS,
    "source_hard_manifest": {
        "path": str(HARD_MANIFEST_PATH),
        "sha256": sha256(HARD_MANIFEST_PATH),
    },
    "anti_circularity_audit": {
        "status": "PASS",
        "imports_by_file": imports_by_file,
        "forbidden_imports_found": [],
        "forbidden_manifest_keys_found": present_forbidden_keys,
        "causal_artifact_read": False,
        "causal_outputs_consumed": False,
        "estimator_logic": "imported unchanged from src/cheap_read.py",
        "cheap_read_sha256": sha256(ROOT / "src/cheap_read.py"),
    },
    "rows": raw_rows,
    "compact_rows": compact_rows,
}
save_json(OUTPUT_PATH, artifact)
print("cheap hard-dashboard artifact", OUTPUT_PATH, sha256(OUTPUT_PATH))

release_model(bundle)
del bundle
gc.collect()
torch.cuda.empty_cache()

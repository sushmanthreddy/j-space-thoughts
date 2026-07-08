"""Build and clean-verify the isolated v6 hard-dashboard manifest."""

from __future__ import annotations

import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from src.data_gen_v6 import (
    HARD_DASHBOARD_TEMPLATES,
    build_hard_dashboard_candidates,
    verify_hard_dashboard_candidates,
)
from src.model_utils import load_model, release_model, set_seed


ROOT = Path("/home/jovyan/j-space-thoughts")
SOURCE_MANIFEST_PATH = ROOT / "data/raw/v6/30_clean_read_manifest.json"
OUTPUT_PATH = ROOT / "results/v6/raw/v6_2_hard_dashboard_manifest.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1729


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


set_seed(SEED)
source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
if source_manifest.get("causal_interchange_outputs_included") is not False:
    raise RuntimeError("Frozen clean source manifest is not causal-output-free")

bundle = load_model(MODEL_ID)
if bundle.revision != source_manifest["model"]["revision"]:
    raise RuntimeError("Pinned model revision differs from the frozen source run")

candidates = build_hard_dashboard_candidates(source_manifest["rows"], bundle.tokenizer)
verified_rows = verify_hard_dashboard_candidates(
    bundle.hf_model,
    bundle.tokenizer,
    candidates,
    batch_size=8,
    top_k=5,
)
counts = Counter(row["verification_status"] for row in verified_rows)
reason_counts = Counter(
    reason for row in verified_rows for reason in row["verification_reasons"]
)
if len(verified_rows) != 77:
    raise RuntimeError(f"Expected 77 hard-dashboard candidates, got {len(verified_rows)}")

artifact = {
    "schema_version": "read-stress-v6-hard-dashboard-manifest-v1",
    "seed": SEED,
    "model": source_manifest["model"],
    "source_clean_manifest": {
        "path": str(SOURCE_MANIFEST_PATH),
        "sha256": sha256(SOURCE_MANIFEST_PATH),
        "protocol_sha256": source_manifest["protocol_sha256"],
    },
    "selection": source_manifest["selection"],
    "direction_cache": source_manifest["direction_cache"],
    "design": {
        "control_kind": "fixed calibration-anchor fact with matched semantic relation and answer class",
        "numeric_example_not_applicable": (
            "The frozen engines output element symbols or capital-city names, never numbers; "
            "forcing numbers would preserve the answer-type mismatch under test."
        ),
        "template_selection_used_heldout_outcomes": False,
        "templates": HARD_DASHBOARD_TEMPLATES,
        "concept_irrelevance_by_construction": True,
        "exact_source_context_prefix_required": True,
    },
    "counts": {
        "candidates": len(verified_rows),
        "verified_hard": int(counts["VERIFIED_HARD"]),
        "unverified_hard": int(counts["UNVERIFIED_HARD"]),
        "dependency_groups_verified": len(
            {
                row["dependency_group"]
                for row in verified_rows
                if row["verification_status"] == "VERIFIED_HARD"
            }
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
    },
    "verification_contract": {
        "correctness": "frozen target token must be clean top-1 on both prompt sides",
        "written": "frozen layer-16 z must exceed the frozen threshold on both sides",
        "written_provenance": "exact causal prefix through concept token is unchanged",
        "failed_rows_excluded_not_relabeled": True,
    },
    "causal_interchange_outputs_included": False,
    "edited_metrics_included": False,
    "rows": verified_rows,
}
save_json(OUTPUT_PATH, artifact)

print("hard-dashboard verification", json.dumps(artifact["counts"], indent=2))
print("templates", json.dumps(HARD_DASHBOARD_TEMPLATES, indent=2))
print("sanitized manifest", OUTPUT_PATH, sha256(OUTPUT_PATH))

release_model(bundle)
del bundle
gc.collect()
torch.cuda.empty_cache()

"""Notebook 34 driver: GO-gated signed mediation and faithfulness."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.causal_read import (
    clean_state_and_logits,
    full_residual_interchange_edit,
    localize_signed_mediation,
    token_difference_metric,
)
from src.metrics import save_json
from src.model_utils import load_model, release_model, set_seed


ROOT = Path("/home/jovyan/j-space-thoughts")
RAW_DIR = ROOT / "data/raw/v6"
FIGURE_DIR = ROOT / "results/figures"
METRICS_PATH = ROOT / "results/metrics.json"
RESULTS_PATH = ROOT / "results/RESULTS.md"
VERIFY_PATH = RAW_DIR / "30_dataset_and_verification.json"
CAUSAL_PATH = RAW_DIR / "31_causal_ground_truth.json"
DIRECTION_PATH = RAW_DIR / "30_selected_directions.pt"
SEED = 1729


metrics = json.loads(METRICS_PATH.read_text())
run = metrics["symmetric_causal_read_v6"]
decision = run["decision"]
if decision != "GO":
    record = {
        "schema_version": "symmetric-localization-v1",
        "status": "SKIPPED_PREREQUISITE",
        "reason": "Phase 3 primary READ_IG decision was not GO",
        "phase3_decision": decision,
        "model_loaded": False,
        "mediation_computed": False,
        "faithfulness_computed": False,
    }
    raw_path = RAW_DIR / "34_localization.json"
    save_json(raw_path, record)
    run["stage34"] = {
        **record,
        "raw_artifact": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        },
    }
    run["status"] = "PHASE_4_NOT_APPLICABLE"
    save_json(METRICS_PATH, metrics)
    print(json.dumps(record, indent=2))
else:
    set_seed(SEED)
    verification = json.loads(VERIFY_PATH.read_text())
    causal = json.loads(CAUSAL_PATH.read_text())
    selected_layer = int(verification["selection"]["layer"])
    verified_by_id = {
        row["pair_id"]: row
        for row in verification["rows"]
        if row["verification_status"] == "VERIFIED"
    }
    stable_rows = [
        row
        for row in causal["rows"]
        if not row["engine"]["full_residual"]["sharp_directional_disagreement"]
        and abs(row["engine"]["full_residual"]["R_a_from_b"]) > 1e-6
    ]
    selected = sorted(
        stable_rows,
        key=lambda row: (
            -abs(row["engine"]["full_residual"]["R_a_from_b"]),
            row["pair_id"],
        ),
    )[:3]
    if not selected:
        raise RuntimeError("GO localization has no stable nonzero A<-B engine case")
    bundle = load_model("Qwen/Qwen2.5-7B-Instruct")
    component_layers = list(range(selected_layer + 1, len(bundle.lens_model.layers)))
    circuit_size = min(8, 2 * len(component_layers))

    def encode(prompt: str) -> torch.Tensor:
        return bundle.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(next(bundle.hf_model.parameters()).device)

    localization_rows = []
    for index, causal_row in enumerate(selected):
        pair = verified_by_id[causal_row["pair_id"]]
        position_a = int(pair["context_position_a"])
        position_b = int(pair["context_position_b"])
        ids_a = encode(pair["engine_prompt_a"])
        ids_b = encode(pair["engine_prompt_b"])
        clean_a = clean_state_and_logits(
            bundle.hf_model,
            bundle.lens_model.layers,
            ids_a,
            selected_layer,
            position=position_a,
        )
        clean_b = clean_state_and_logits(
            bundle.hf_model,
            bundle.lens_model.layers,
            ids_b,
            selected_layer,
            position=position_b,
        )
        source_edits = {
            selected_layer: full_residual_interchange_edit(
                clean_b["state"], position=position_a
            )
        }
        metric_fn = token_difference_metric(
            pair["answer_a_token_id"], pair["answer_b_token_id"]
        )
        localization = localize_signed_mediation(
            bundle.hf_model,
            bundle.lens_model.layers,
            ids_a,
            metric_fn,
            source_edits,
            component_layers=component_layers,
            circuit_size=circuit_size,
            go_authorized=True,
        )
        localization_rows.append(
            {
                "pair_id": pair["pair_id"],
                "dependency_group": pair["dependency_group"],
                "selection_role": (
                    "post-GO exploratory top absolute stable engine R_A<-B"
                ),
                "causal_R_a_from_b": causal_row["engine"]["full_residual"][
                    "R_a_from_b"
                ],
                "localization": localization,
            }
        )
        print(
            f"[{index + 1}/{len(selected)}] {pair['pair_id']} "
            f"faithfulness={localization['faithfulness']['faithfulness_fraction']:.4f} "
            f"circuit={localization['circuit_components']}"
        )

    exemplar = localization_rows[0]
    component_rows = sorted(
        exemplar["localization"]["component_rows"],
        key=lambda row: abs(row["READ_k"]),
        reverse=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), gridspec_kw={"width_ratios": [3, 1]})
    names = [row["component"] for row in component_rows]
    values = [row["READ_k"] for row in component_rows]
    colors = ["#1565C0" if value >= 0 else "#C62828" for value in values]
    axes[0].bar(np.arange(len(names)), values, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(np.arange(len(names)), names, rotation=65, ha="right", fontsize=8)
    axes[0].set_ylabel("signed component READ_k")
    axes[0].set_title(f"F5 — signed mediation ({exemplar['pair_id']})")
    faithfulness_values = [
        row["localization"]["faithfulness"]["faithfulness_fraction"]
        for row in localization_rows
    ]
    axes[1].bar(
        np.arange(len(localization_rows)),
        faithfulness_values,
        color="#6A1B9A",
    )
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_xticks(
        np.arange(len(localization_rows)),
        [row["pair_id"] for row in localization_rows],
        rotation=65,
        ha="right",
        fontsize=8,
    )
    axes[1].set_ylabel("outside-zero-ablation faithfulness")
    axes[1].set_title("Top-k circuit check")
    f5_path = FIGURE_DIR / "f5_signed_mediation_faithfulness.png"
    fig.tight_layout()
    fig.savefig(f5_path, dpi=180)
    plt.close(fig)

    record = {
        "schema_version": "symmetric-localization-v1",
        "status": "COMPLETE",
        "phase3_decision": decision,
        "source_layer": selected_layer,
        "component_layers": component_layers,
        "component_scope": "complete strictly-downstream attention and MLP outputs",
        "circuit_size": circuit_size,
        "rows": localization_rows,
        "figure": str(f5_path.relative_to(ROOT)),
    }
    raw_path = RAW_DIR / "34_localization.json"
    save_json(raw_path, record)
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    run["stage34"] = {
        **record,
        "raw_artifact": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": raw_sha,
        },
    }
    run["status"] = "PHASE_4_COMPLETE"
    save_json(METRICS_PATH, metrics)

    report = RESULTS_PATH.read_text()
    archive_marker = "\n---\n\n# Prior READ Go/No-Go validation"
    section = f"""

## GO-only signed localization

Phase 4 ran on three post-GO exploratory, directionally stable engine cases.
Every strictly downstream attention and MLP output was restored individually to
its clean value to obtain signed `READ_k`. The proposed top-{circuit_size}
circuits were then tested by zero-ablating every downstream component outside
the circuit in both clean and edited runs. Faithfulness fractions were
`{[round(value, 4) for value in faithfulness_values]}`.

![F5](figures/{f5_path.name})
"""
    report = report.replace(archive_marker, section.rstrip() + "\n" + archive_marker, 1)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "figure": str(f5_path)}, indent=2))
    release_model(bundle)
    gc.collect()
    torch.cuda.empty_cache()

"""Firewalled matched-metric READ execution for v7.

The cheap stage reads only the sanitized clean manifest, reconstructs the
frozen L16 J-Lens directions, and calls the unchanged 16-step READ estimator.
It has no dependency on intervention code or edited-model outputs.  Model-free
joining and statistical reporting are added only after this artifact is fixed.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import torch

from src.cheap_read import score_prompt_pair, weight_norm_capacity_baseline
from src.datasets import validate_sanitized_manifest
from src.jlens_interface import (
    MODEL_ID,
    MODEL_REVISION,
    jlens_direction_bank,
    load_model,
    load_published_lens,
    release_model,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "v7"
MANIFEST_PATH = RESULTS_DIR / "matched_manifest_v7.json"
READ_PATH = RESULTS_DIR / "raw" / "read_v7.json"
SEED = 1729
SELECTED_LAYER = 16
POSITION_RULE = "explicit_concept_token_in_shared_context"
METRIC_DEFINITION = "logit(answer_A) - logit(answer_B)"
IG_STEPS = 16
FROZEN_CHEAP_READ_SHA256 = (
    "a4a0ab35c50ce73dd153414118e6150891a708acf5f64bf9c8cb31225bb0caab"
)
FORBIDDEN_IMPORT_FRAGMENTS = (
    "causal",
    "interchange",
    "intervention",
    "patch",
    "read_scores",
    "read_validation",
)

ProgressFn = Callable[[int, int, Mapping[str, Any]], None]


def sha256_file(path: str | Path) -> str:
    """Return one file's byte SHA-256."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, value: Any) -> Path:
    """Persist deterministic finite JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _direction_digest(directions: Mapping[int, torch.Tensor], *, layer: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"v7-jlens-directions-layer-{int(layer)}\n".encode())
    for token_id in sorted(int(value) for value in directions):
        vector = directions[token_id].detach().float().contiguous().cpu()
        if vector.ndim != 1 or not torch.isfinite(vector).all():
            raise ValueError(f"Direction {token_id} is not finite and one-dimensional")
        if not torch.allclose(vector.norm(), torch.tensor(1.0), atol=1e-4, rtol=1e-4):
            raise ValueError(f"Direction {token_id} is not unit norm")
        digest.update(f"{token_id}:{vector.numel()}:".encode())
        digest.update(vector.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _build_direction_bank_read_v7(
    bundle: Any,
    token_ids: Sequence[int],
    *,
    layer: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Reconstruct raw J-Lens directions without importing a data-stage runner."""

    published_lens = load_published_lens(bundle.model_id, local_files_only=True)
    bank = jlens_direction_bank(
        published_lens,
        bundle.lens_model,
        token_ids,
        [int(layer)],
        fold_rms_gain=False,
        compute_device="cuda",
        output_device="cpu",
    )
    directions = {int(token_id): layers[int(layer)] for token_id, layers in bank.items()}
    provenance = {
        "convention": "normalize(J_layer.T @ W_U[token]); fold_rms_gain=False",
        "layer": int(layer),
        "n_directions": len(directions),
        "token_ids": sorted(directions),
        "ordered_tensor_digest_sha256": _direction_digest(directions, layer=layer),
    }
    del published_lens
    return directions, provenance


def imported_modules(path: str | Path) -> list[str]:
    """Return static import names for one Python source file."""

    source_path = Path(path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
            names.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
    return sorted(names)


def firewall_import_audit(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Fail if cheap-stage source imports any forbidden experiment module."""

    records: dict[str, list[str]] = {}
    violations: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        modules = imported_modules(path)
        label = str(path.relative_to(PROJECT_ROOT))
        records[label] = modules
        for module in modules:
            lowered = module.casefold()
            if any(fragment in lowered for fragment in FORBIDDEN_IMPORT_FRAGMENTS):
                violations.append({"path": label, "module": module})
    if violations:
        raise RuntimeError(f"Cheap READ firewall import violation: {violations}")
    return {
        "status": "PASS",
        "imports_by_file": records,
        "forbidden_import_fragments": list(FORBIDDEN_IMPORT_FRAGMENTS),
        "forbidden_imports_found": [],
    }


def _verified_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Sanitized manifest rows are missing")
    verified = [dict(row) for row in rows if row.get("verification_status") == "VERIFIED"]
    if len(verified) < 50:
        raise RuntimeError("Fewer than 50 verified rows reached frozen READ")
    pair_ids = [str(row["pair_id"]) for row in verified]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Verified READ pair IDs are not unique")
    return verified


@torch.no_grad()
def compute_read_rows_v7(
    bundle: Any,
    rows: Sequence[Mapping[str, Any]],
    directions: Mapping[int, torch.Tensor],
    *,
    layer: int = SELECTED_LAYER,
    ig_steps: int = IG_STEPS,
    progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Call the frozen READ scorer for both conditions with one matched metric."""

    if ig_steps != 16:
        raise ValueError("V7 must use the frozen 16-step READ_IG")
    if int(layer) != SELECTED_LAYER:
        raise ValueError("V7 must use frozen layer L16")
    if bundle.hf_model.training or any(
        parameter.requires_grad for parameter in bundle.hf_model.parameters()
    ):
        raise ValueError("Frozen READ requires an eval-mode, parameter-frozen model")
    blocks = bundle.lens_model.layers
    output: list[dict[str, Any]] = []
    for completed, row in enumerate(rows, start=1):
        if row.get("verification_status") != "VERIFIED":
            raise ValueError(f"{row.get('pair_id')} is not VERIFIED")
        pair_id = str(row["pair_id"])
        positive_id = int(row["answer_a_token_id"])
        negative_id = int(row["answer_b_token_id"])
        if positive_id != int(row["metric_positive_token_id"]):
            raise ValueError(f"{pair_id} positive metric token drifted")
        if negative_id != int(row["metric_negative_token_id"]):
            raise ValueError(f"{pair_id} negative metric token drifted")
        if row["engine_metric"] != row["dashboard_metric"]:
            raise ValueError(f"{pair_id} engine/dashboard metric text differs")
        if row["engine_metric"] != METRIC_DEFINITION:
            raise ValueError(f"{pair_id} metric differs from the frozen v7 definition")
        direction_a = directions[int(row["concept_a_token_id"])]
        direction_b = directions[int(row["concept_b_token_id"])]
        engine = score_prompt_pair(
            bundle.hf_model,
            blocks,
            bundle.tokenizer,
            prompt_a=str(row["engine_prompt_a"]),
            prompt_b=str(row["engine_prompt_b"]),
            position_a=int(row["engine_position_a"]),
            position_b=int(row["engine_position_b"]),
            positive_token_id=positive_id,
            negative_token_id=negative_id,
            direction_a=direction_a,
            direction_b=direction_b,
            layer=layer,
            ig_steps=ig_steps,
        )
        dashboard = score_prompt_pair(
            bundle.hf_model,
            blocks,
            bundle.tokenizer,
            prompt_a=str(row["dashboard_prompt_a"]),
            prompt_b=str(row["dashboard_prompt_b"]),
            position_a=int(row["dashboard_position_a"]),
            position_b=int(row["dashboard_position_b"]),
            positive_token_id=positive_id,
            negative_token_id=negative_id,
            direction_a=direction_a,
            direction_b=direction_b,
            layer=layer,
            ig_steps=ig_steps,
        )
        for condition, estimate in (("engine", engine), ("dashboard", dashboard)):
            if estimate.get("causal_outputs_consumed") is not False:
                raise RuntimeError(f"{pair_id} {condition} did not certify READ isolation")
            if int(estimate["ig_steps"]) != 16:
                raise RuntimeError(f"{pair_id} {condition} changed IG steps")
        expected_top = [positive_id, negative_id]
        if [int(value) for value in engine["clean_top_token_ids"]] != expected_top:
            raise RuntimeError(f"{pair_id} engine READ-batch clean top-1 drifted")
        if [int(value) for value in dashboard["clean_top_token_ids"]] != expected_top:
            raise RuntimeError(f"{pair_id} dashboard READ-batch clean top-1 drifted")
        baseline = weight_norm_capacity_baseline(
            blocks[layer], direction_a, direction_b
        )
        if baseline.get("behavior_metric_used") is not False:
            raise RuntimeError(f"{pair_id} capacity baseline used the behavior metric")
        if baseline.get("eligible_for_go") is not False:
            raise RuntimeError(f"{pair_id} capacity baseline eligibility drifted")
        result = {
            "pair_id": pair_id,
            "dependency_group": str(row["dependency_group"]),
            "fold": int(row["fold"]),
            "category": str(row["category"]),
            "concept_a": str(row["concept_a"]),
            "concept_b": str(row["concept_b"]),
            "answer_a": str(row["answer_a"]),
            "answer_b": str(row["answer_b"]),
            "verification_status": "VERIFIED",
            "metric": METRIC_DEFINITION,
            "metric_positive_token_id": positive_id,
            "metric_negative_token_id": negative_id,
            "same_metric_in_both_conditions": True,
            "engine": engine,
            "dashboard": dashboard,
            "capacity_baseline": baseline,
            "weight_norm_capacity_baseline": float(
                baseline["weight_norm_baseline"]
            ),
            "read_batch_clean_top_token_ids_expected": expected_top,
            "engine_read_batch_top1_consistent": (
                [int(value) for value in engine["clean_top_token_ids"]] == expected_top
            ),
            "dashboard_read_batch_top1_consistent": (
                [int(value) for value in dashboard["clean_top_token_ids"]]
                == expected_top
            ),
        }
        output.append(result)
        if progress is not None:
            progress(
                completed,
                len(rows),
                {
                    "pair_id": pair_id,
                    "status": "OK",
                    "same_metric_in_both_conditions": True,
                },
            )
    return output


def summarize_read_execution_v7(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return execution diagnostics without performing the decisive comparison."""

    if not rows:
        raise ValueError("READ execution summary requires rows")
    return {
        "n_pairs": len(rows),
        "n_dependency_groups": len({str(row["dependency_group"]) for row in rows}),
        "ig_steps": IG_STEPS,
        "engine_read_batch_top1_consistent": sum(
            bool(row["engine_read_batch_top1_consistent"]) for row in rows
        ),
        "dashboard_read_batch_top1_consistent": sum(
            bool(row["dashboard_read_batch_top1_consistent"]) for row in rows
        ),
        "finite_READ_IG": all(
            torch.isfinite(torch.tensor(float(row[condition]["READ_IG"])))
            for row in rows
            for condition in ("engine", "dashboard")
        ),
        "finite_READ_local": all(
            torch.isfinite(torch.tensor(float(row[condition]["READ_local"])))
            for row in rows
            for condition in ("engine", "dashboard")
        ),
        "capacity_baseline_pair_median": float(
            median(float(row["capacity_baseline"]["weight_norm_baseline"]) for row in rows)
        ),
        "condition_comparison_withheld_until_v7_4": True,
    }


def run_read_stage_v7(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    output_path: str | Path = READ_PATH,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Execute firewalled v7_3 without reading any edited-output artifact."""

    set_seed(SEED)
    source_path = Path(manifest_path)
    manifest_sha256_before = sha256_file(source_path)
    cheap_path = PROJECT_ROOT / "src" / "cheap_read.py"
    runner_path = PROJECT_ROOT / "src" / "matched_eval_v7.py"
    cheap_hash_before = sha256_file(cheap_path)
    if cheap_hash_before != FROZEN_CHEAP_READ_SHA256:
        raise RuntimeError("Frozen src/cheap_read.py hash changed")
    firewall = firewall_import_audit(
        [
            cheap_path,
            runner_path,
            PROJECT_ROOT / "src" / "datasets.py",
            PROJECT_ROOT / "src" / "jlens_interface.py",
        ]
    )
    manifest = validate_sanitized_manifest(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    if manifest.get("schema_version") != "matched-read-v7-sanitized-manifest-v1":
        raise ValueError("Unexpected sanitized v7 manifest schema")
    if manifest["metric_contract"]["engine"] != manifest["metric_contract"]["dashboard"]:
        raise ValueError("Sanitized manifest conditions do not share a metric")
    if manifest["metric_contract"]["engine"] != METRIC_DEFINITION:
        raise ValueError("Sanitized manifest metric differs from frozen v7")
    if int(manifest["selection"]["layer"]) != SELECTED_LAYER:
        raise ValueError("Sanitized manifest layer differs from L16")
    if str(manifest["selection"]["position_rule"]) != POSITION_RULE:
        raise ValueError("Sanitized manifest position rule drifted")
    verified = _verified_rows(manifest)

    bundle = load_model(local_files_only=True)
    try:
        observed_dtype = str(next(bundle.hf_model.parameters()).dtype)
        if bundle.model_id != MODEL_ID or bundle.revision != MODEL_REVISION:
            raise RuntimeError("Loaded model identity differs from the pinned v7 model")
        if observed_dtype != "torch.bfloat16":
            raise RuntimeError(f"Loaded model dtype is not bf16: {observed_dtype}")
        token_ids = [
            int(value) for value in manifest["direction_provenance"]["token_ids"]
        ]
        directions, direction_provenance = _build_direction_bank_read_v7(
            bundle, token_ids, layer=SELECTED_LAYER
        )
        if (
            direction_provenance["ordered_tensor_digest_sha256"]
            != manifest["direction_provenance"]["ordered_tensor_digest_sha256"]
        ):
            raise RuntimeError("Reconstructed READ directions differ from v7_1")
        rows = compute_read_rows_v7(
            bundle,
            verified,
            directions,
            layer=SELECTED_LAYER,
            ig_steps=IG_STEPS,
            progress=progress,
        )
        execution = summarize_read_execution_v7(rows)
        cheap_hash_after = sha256_file(cheap_path)
        if cheap_hash_after != cheap_hash_before:
            raise RuntimeError("Frozen src/cheap_read.py changed during v7_3")
        manifest_sha256_after = sha256_file(source_path)
        if manifest_sha256_after != manifest_sha256_before:
            raise RuntimeError("Sanitized manifest changed during v7_3")
        firewall.update(
            {
                "causal_artifact_read": False,
                "edited_metrics_read": False,
                "patch_outputs_read": False,
                "causal_outputs_consumed": False,
                "only_experimental_data_input": str(source_path.relative_to(PROJECT_ROOT)),
                "sanitized_manifest_sha256_before": manifest_sha256_before,
                "sanitized_manifest_sha256_after": manifest_sha256_after,
                "frozen_read_sha256_before": cheap_hash_before,
                "frozen_read_sha256_after": cheap_hash_after,
            }
        )
        artifact = {
            "schema_version": "matched-read-v7-cheap-v1",
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "dtype": "torch.bfloat16",
            },
            "selected_layer": SELECTED_LAYER,
            "position_rule": POSITION_RULE,
            "ig_steps": IG_STEPS,
            "metric_contract": {
                "engine": METRIC_DEFINITION,
                "dashboard": METRIC_DEFINITION,
                "identical_token_ids_enforced": True,
            },
            "source_manifest": {
                "path": str(source_path.relative_to(PROJECT_ROOT)),
                "sha256": manifest_sha256_after,
            },
            "direction_provenance": direction_provenance,
            "frozen_read": {
                "path": "src/cheap_read.py",
                "sha256": cheap_hash_after,
                "function": "src.cheap_read.score_prompt_pair",
                "ig_steps": IG_STEPS,
                "modified": False,
            },
            "firewall": firewall,
            "execution": execution,
            "rows": rows,
        }
        destination = save_json(output_path, artifact)
        return {
            "read_path": str(destination),
            "read_sha256": sha256_file(destination),
            "n_pairs": len(rows),
            "firewall": firewall,
            "execution": execution,
        }
    finally:
        release_model(bundle)


__all__ = [
    "FROZEN_CHEAP_READ_SHA256",
    "IG_STEPS",
    "READ_PATH",
    "compute_read_rows_v7",
    "firewall_import_audit",
    "imported_modules",
    "run_read_stage_v7",
    "summarize_read_execution_v7",
]

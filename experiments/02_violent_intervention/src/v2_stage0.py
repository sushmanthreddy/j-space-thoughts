"""Repair-first Stage-0 provenance, upstream audit, and report artifacts.

Stage 0 deliberately separates two questions that the first run conflated:

* Can the released Jacobian Lens walkthrough load a model/lens and produce a
  readout?
* Did the release provide an executable causal swap that can be run unchanged?

The public release answers the first question, but not the second.  This
module records that distinction without treating an unavailable reference
experiment as either a model failure or a successful intervention.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from src.metrics import save_json
from src.plotting import save_figure, set_style


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY = Path.home() / "deps" / "jacobian-lens"
WALKTHROUGH = DEPENDENCY / "walkthrough.ipynb"
PROBE_SWAP = DEPENDENCY / "data" / "experiments" / "probe-swap.json"
EXPECTED_DEPENDENCY_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
EXPECTED_WALKTHROUGH_SHA256 = (
    "96ba7c7945f0902e6cdacd32320309176dcfd891b571c2734a1aa60facfc5d4a"
)
EXPECTED_PROBE_SWAP_SHA256 = (
    "a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796"
)
UPSTREAM_CODE_CELLS = (1, 3, 5, 7)


def _run(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "output": completed.stdout.rstrip(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_preflight() -> dict[str, Any]:
    """Run and persist the required non-mutating environment checks."""

    path = os.environ.get("PATH", "")
    commands = {
        "codex_version": _run(["codex", "--version"]),
        "hf_auth": _run(["hf", "auth", "whoami"]),
        "git_global_config": _run(
            ["git", "-C", str(ROOT), "config", "--global", "--list"]
        ),
        "git_remote": _run(["git", "-C", str(ROOT), "remote", "-v"]),
        "gpu": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv",
            ]
        ),
        "disk": _run(["df", "-h", "/home/jovyan", str(Path.home() / ".cache/huggingface")]),
    }
    required_tools = {
        "codex": shutil.which("codex"),
        "hf": shutil.which("hf"),
        "git": shutil.which("git"),
        "nvidia-smi": shutil.which("nvidia-smi"),
    }
    tools_present = all(required_tools.values())
    commands_ok = all(value["returncode"] == 0 for value in commands.values())
    disk = shutil.disk_usage("/home/jovyan")
    gpu_csv = commands["gpu"]["output"].splitlines()
    gpu_values: dict[str, Any] = {}
    if len(gpu_csv) >= 2:
        fields = [part.strip() for part in gpu_csv[1].split(",")]
        if len(fields) == 3:
            gpu_values = {
                "name": fields[0],
                "memory_total_mib": int(fields[1].split()[0]),
                "memory_free_mib": int(fields[2].split()[0]),
            }
    payload = {
        "status": "PASS" if tools_present and commands_ok else "FAIL",
        "path": path,
        "tools": required_tools,
        "commands": commands,
        "gpu": gpu_values,
        "disk": {
            "path": "/home/jovyan",
            "total_gib": disk.total / 2**30,
            "used_gib": disk.used / 2**30,
            "free_gib": disk.free / 2**30,
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    return payload


def print_preflight(preflight: Mapping[str, Any]) -> None:
    """Print the same evidence retained in the executed notebook."""

    print("PATH=" + str(preflight["path"]))
    for name, value in preflight["tools"].items():
        print(f"{name}: {value}")
    for name, value in preflight["commands"].items():
        print(f"\n[{name}] returncode={value['returncode']}")
        print(value["output"])
    print(f"\nPREFLIGHT {preflight['status']}")


def audit_upstream_release() -> dict[str, Any]:
    """Prove what the pinned dependency release does and does not ship."""

    if not WALKTHROUGH.exists() or not PROBE_SWAP.exists():
        raise FileNotFoundError("Pinned Jacobian Lens checkout is incomplete")
    notebook = json.loads(WALKTHROUGH.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    selected_sources = {
        str(index): "".join(cells[index].get("source", []))
        for index in UPSTREAM_CODE_CELLS
    }
    all_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )
    forbidden_mutation_markers = (
        "register_forward_hook",
        "register_forward_pre_hook",
        "swap_coordinates",
        "ablate_direction",
        "residual_edit",
    )
    mutation_markers = {
        marker: marker in all_code for marker in forbidden_mutation_markers
    }
    dependency_commit = _run(["git", "rev-parse", "HEAD"], cwd=DEPENDENCY)[
        "output"
    ]
    dependency_status = _run(["git", "status", "--short"], cwd=DEPENDENCY)[
        "output"
    ]
    probe_payload = json.loads(PROBE_SWAP.read_text(encoding="utf-8"))
    probe_rows = probe_payload["items"]
    spider_rows = [
        row
        for row in probe_rows
        if row.get("intermediate") == "spider" and row.get("swap_to") == "ant"
    ]
    api_scan = _run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            "swap|ablat|interven|register_forward_hook|register_forward_pre_hook",
            "--",
            "jlens",
            "tests",
            "walkthrough.ipynb",
        ],
        cwd=DEPENDENCY,
    )
    causal_code_available = any(mutation_markers.values())
    return {
        "dependency_path": str(DEPENDENCY),
        "dependency_commit": dependency_commit,
        "expected_dependency_commit": EXPECTED_DEPENDENCY_COMMIT,
        "dependency_clean": dependency_status == "",
        "walkthrough_sha256": _sha256(WALKTHROUGH),
        "expected_walkthrough_sha256": EXPECTED_WALKTHROUGH_SHA256,
        "probe_swap_sha256": _sha256(PROBE_SWAP),
        "expected_probe_swap_sha256": EXPECTED_PROBE_SWAP_SHA256,
        "upstream_code_cells": list(UPSTREAM_CODE_CELLS),
        "upstream_code_cell_sha256": {
            index: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for index, source in selected_sources.items()
        },
        "walkthrough_mutation_markers": mutation_markers,
        "api_scan_returncode": api_scan["returncode"],
        "api_scan_output": api_scan["output"],
        "spider_probe_rows": spider_rows,
        "json_role": "prompts_only",
        "walkthrough_capability": "READOUT_ONLY",
        "causal_swap_code_available": causal_code_available,
        "canonical_swap_runnable_unchanged": False,
        "classification": "UPSTREAM_CAUSAL_SWAP_NOT_RUNNABLE_RELEASE_OMISSION",
        "g_swap_status": "UNTESTED",
        "model_mismatch_inference_permitted": False,
    }


def collect_upstream_readout(
    *,
    tokenizer: Any,
    jlens_logits: Mapping[int, torch.Tensor],
    logit_lens: Mapping[int, torch.Tensor],
    model_logits: torch.Tensor,
    layers: Sequence[int],
    model_name: str,
    lens_repo: str,
    lens_revision: str,
    lens_file: str,
    model_commit: str | None,
) -> dict[str, Any]:
    """Serialize outputs produced by the unchanged upstream readout cell."""

    def top(logits: torch.Tensor, k: int = 5) -> list[dict[str, Any]]:
        values, indices = logits.float().topk(k)
        return [
            {
                "token_id": int(index),
                "token": tokenizer.decode([int(index)]),
                "logit": float(value),
            }
            for value, index in zip(values.cpu(), indices.cpu(), strict=True)
        ]

    rows = []
    for layer in layers:
        rows.append(
            {
                "layer": int(layer),
                "logit_lens_top5": top(logit_lens[layer][0]),
                "jlens_top5": top(jlens_logits[layer][0]),
            }
        )
    return {
        "status": "PASS",
        "scope": "model_and_lens_load_plus_readout_only",
        "is_causal_intervention": False,
        "prompt": "Fact: The currency used in the country shaped like a boot is",
        "positions": [-2],
        "model_name": model_name,
        "model_commit": model_commit,
        "lens_repo": lens_repo,
        "lens_revision": lens_revision,
        "lens_file": lens_file,
        "layers": [int(layer) for layer in layers],
        "rows": rows,
        "model_top5": top(model_logits[0]),
    }


def _stage0_figure(stage0: Mapping[str, Any]) -> str:
    set_style()
    labels = [
        "Environment preflight",
        "Released walkthrough readout",
        "Released causal swap code",
        "Canonical G-SWAP",
    ]
    states = ["PASS", "PASS", "NOT RELEASED", "UNTESTED"]
    colors = ["#2E7D32", "#2E7D32", "#B26A00", "#6A6A6A"]
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    y = list(range(len(labels)))
    axis.barh(y, [1] * len(y), color=colors, height=0.62)
    axis.set_yticks(y, labels=labels)
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis.invert_yaxis()
    for row, state in enumerate(states):
        axis.text(
            0.5,
            row,
            state,
            color="white",
            weight="bold",
            ha="center",
            va="center",
        )
    axis.set_title("F0 — Repair-first Stage 0: upstream readout is not a causal swap")
    axis.text(
        0,
        -0.18,
        "The pinned public walkthrough contains no activation-mutation cell; "
        "therefore code-vs-model diagnosis is unresolved.",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    path = ROOT / "results" / "figures" / "f0_stage0_upstream_audit.png"
    save_figure(figure, path)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _interim_report(repair: Mapping[str, Any]) -> str:
    preflight = repair["preflight"]
    stage0 = repair["stage0"]
    gpu = preflight.get("gpu", {})
    disk = preflight.get("disk", {})
    readout = stage0["upstream_readout"]
    audit = stage0["upstream_release_audit"]
    return f"""# Repair-first replication report (v2)

## Current verdict

**INSTRUMENT NOT YET VALIDATED — NO SCIENCE VERDICT.** The earlier
`NOT SUPPORTED` / `REFUTED` labels are withdrawn as scientific conclusions.
They were computed downstream of failed calibration gates and remain only as
legacy diagnostics.

## Environment

- GPU: {gpu.get('name', 'unknown')}; {gpu.get('memory_total_mib', 'unknown')} MiB total; {gpu.get('memory_free_mib', 'unknown')} MiB free at preflight.
- Home/HF-cache filesystem: {disk.get('total_gib', float('nan')):.1f} GiB total; {disk.get('free_gib', float('nan')):.1f} GiB free at preflight.
- Required tool/auth preflight: **{preflight['status']}**.

## Stage 0 — upstream diagnosis

- Pinned dependency: `{audit['dependency_commit']}`; clean checkout={audit['dependency_clean']}.
- Walkthrough SHA-256: `{audit['walkthrough_sha256']}`.
- Unchanged released readout cells 1/3/5/7: **{readout['status']}** on `{readout['model_name']}` at layers {readout['layers']} and position `-2`.
- The released walkthrough performs model/lens loading and readout only. It never changes an activation or runs a swapped continuation.
- `data/experiments/probe-swap.json` contains the spider→ant prompt metadata, but the dependency explicitly describes the JSON files as prompts only.
- Executable upstream swap/ablation helper: **NOT RELEASED**.

### Stage-0 decision

`{audit['classification']}`. The requested unchanged canonical swap is not
runnable from the public release, so Stage 0 cannot distinguish a local code
bug from a Qwen model mismatch. This is not evidence that Qwen failed the
method. The strict G-SWAP state is **{audit['g_swap_status']}** pending an
honest repair/calibration attempt in our implementation.

![F0 Stage-0 audit](figures/f0_stage0_upstream_audit.png)

## Gate ledger

| gate | status | consequence |
| --- | --- | --- |
| Stage-0 preflight | {preflight['status']} | Environment usable |
| Upstream readout | {readout['status']} | Readout compatibility only |
| Unchanged upstream causal swap | NOT RUNNABLE | Release omission; no code-vs-model inference |
| G-SWAP | UNTESTED | Stage 2 and Stage 3 remain prohibited |
| G-DIR | NOT RUN IN V2 | Blocked behind G-SWAP |
| G-POS / firing controls | NOT RUN IN V2 | Blocked behind G-SWAP and recalibration |

## Interpretation

What is established so far is narrow: the released J-Lens can load and return
readouts on its demonstrated open model. What is not established is a causal
coordinate swap, a calibrated Qwen workspace band, or the truth or falsity of
the WRITE-versus-READ hypothesis.
"""


def persist_stage0(
    *,
    preflight: Mapping[str, Any],
    upstream_audit: Mapping[str, Any],
    upstream_readout: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge Stage-0 evidence into metrics and write F0 plus interim report."""

    metrics_path = ROOT / "results" / "metrics.json"
    repair = {
        "schema_version": "repair-first-v2",
        "legacy_v1_science_conclusions_valid": False,
        "current_allowed_conclusion": "INSTRUMENT_NOT_YET_VALIDATED",
        "preflight": dict(preflight),
        "stage0": {
            "status": "COMPLETE_WITH_RELEASE_OMISSION",
            "decision": upstream_audit["classification"],
            "upstream_release_audit": dict(upstream_audit),
            "upstream_readout": dict(upstream_readout),
            "g_swap_status": "UNTESTED",
            "proceed_to": "STAGE_1_CUSTOM_REPAIR",
            "science_allowed": False,
        },
        "gate_ledger": {
            "stage0_preflight": preflight["status"],
            "upstream_readout": upstream_readout["status"],
            "upstream_causal_swap": "NOT_RUNNABLE_RELEASE_OMISSION",
            "g_swap": "UNTESTED",
            "g_dir": "NOT_RUN_V2",
            "g_pos": "NOT_RUN_V2",
            "controls_fire": "NOT_RUN_V2",
            "stage3_science": "PROHIBITED",
        },
    }
    repair["stage0"]["figure"] = _stage0_figure(repair["stage0"])
    metrics = {
        "schema_version": "repair-first-v2",
        "metadata": {
            "sign_convention": "delta = M_edited - M_clean",
            "workflow": "stage0 -> stage1/G-SWAP -> stage2 -> stage3 or stage4",
        },
        "legacy_v1": {
            "status": "INVALIDATED_INSTRUMENT_FAILURE",
            "source_commit": "6666385cff42fe4053412e7230ec9f55b0259f79",
            "science_use_permitted": False,
            "note": (
                "Full legacy metrics remain reproducible from the source commit; "
                "they are not copied into the v2 science schema."
            ),
        },
        "repair_v2": repair,
    }
    save_json(metrics_path, metrics)
    (ROOT / "results" / "RESULTS.md").write_text(
        _interim_report(repair), encoding="utf-8"
    )
    return repair

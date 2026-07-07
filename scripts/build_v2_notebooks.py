"""Build the small, reviewable v2 notebooks from committed Python modules.

The upstream Stage-0 code cells are copied directly from the pinned Jacobian
Lens walkthrough.  The builder asserts their source is unchanged; surrounding
cells add provenance and persistence without pretending the readout is a
causal intervention.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_WALKTHROUGH = Path.home() / "deps" / "jacobian-lens" / "walkthrough.ipynb"


def _metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python (j-space-thoughts)",
            "language": "python",
            "name": "j-space-thoughts",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }


def build_stage0() -> Path:
    upstream = nbformat.read(UPSTREAM_WALKTHROUGH, as_version=4)
    selected = [copy.deepcopy(upstream.cells[index]) for index in (1, 3, 5, 7)]
    for copied, index in zip(selected, (1, 3, 5, 7), strict=True):
        assert copied.source == upstream.cells[index].source
        copied.outputs = []
        copied.execution_count = None
        copied.metadata = {}

    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 00 — Preflight and upstream Stage-0 diagnosis

This repair-first run withdraws the earlier scientific verdict until the
causal instrument passes calibration. Stage 0 first records the required
environment checks, then executes the released Jacobian Lens walkthrough's
model/lens/readout cells with byte-identical source. The public walkthrough
contains no causal intervention cell; readout success is therefore reported
separately from the unavailable canonical swap."""
        ),
        nbformat.v4.new_code_cell(
            """import os
import sys
from pathlib import Path

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(Path.home() / '.cache/huggingface')
os.environ['HF_HUB_CACHE'] = str(Path.home() / '.cache/huggingface/hub')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(Path.home() / '.cache/huggingface/hub')

from src.v2_stage0 import collect_preflight, print_preflight

preflight = collect_preflight()
print_preflight(preflight)
assert preflight['status'] == 'PASS', preflight"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Released walkthrough readout (unchanged source)

The next four code cells are exact copies of cells 1, 3, 5, and 7 from the
pinned upstream `walkthrough.ipynb`. They demonstrate loading and readout only.
No activation mutation or swapped continuation is present in the release."""
        ),
        *selected,
        nbformat.v4.new_markdown_cell(
            """## Release audit and Stage-0 decision

The audit below verifies dependency/file hashes, scans the released
walkthrough/API/tests, persists the readout output, and creates F0. Missing
upstream causal code is classified as a release omission—not as a Qwen model
failure and not as a successful swap."""
        ),
        nbformat.v4.new_code_cell(
            """import json

from src.v2_stage0 import (
    audit_upstream_release,
    collect_upstream_readout,
    persist_stage0,
)

upstream_audit = audit_upstream_release()
upstream_readout = collect_upstream_readout(
    tokenizer=tokenizer,
    jlens_logits=jlens_logits,
    logit_lens=logit_lens,
    model_logits=model_logits,
    layers=layers,
    model_name=MODEL_NAME,
    lens_repo=LENS_REPO,
    lens_revision=LENS_REVISION,
    lens_file=LENS_FILE,
    model_commit=getattr(hf_model.config, '_commit_hash', None),
)
repair = persist_stage0(
    preflight=preflight,
    upstream_audit=upstream_audit,
    upstream_readout=upstream_readout,
)
print(json.dumps({
    'stage0': repair['stage0']['status'],
    'decision': repair['stage0']['decision'],
    'upstream_readout': repair['gate_ledger']['upstream_readout'],
    'upstream_causal_swap': repair['gate_ledger']['upstream_causal_swap'],
    'g_swap': repair['gate_ledger']['g_swap'],
    'science_allowed': repair['stage0']['science_allowed'],
}, indent=2))
assert repair['gate_ledger']['g_swap'] == 'UNTESTED'
assert repair['stage0']['science_allowed'] is False"""
        ),
        nbformat.v4.new_code_cell(
            """import gc

del lens, model, tokenizer, hf_model
gc.collect()
torch.cuda.empty_cache()
print('Stage 0 complete: proceed only to Stage 1 custom repair; Stage 2/3 remain gated.')"""
        ),
    ]
    target = ROOT / "notebooks" / "00_preflight_and_stage0.ipynb"
    nbformat.write(notebook, target)
    return target


def main() -> None:
    targets = [build_stage0()]
    print(json.dumps([str(path.relative_to(ROOT)) for path in targets], indent=2))


if __name__ == "__main__":
    main()

"""Build the small, reviewable v2 notebooks from committed Python modules.

The upstream Stage-0 code cells are copied directly from the pinned Jacobian
Lens walkthrough.  The builder asserts their source is unchanged; surrounding
cells add provenance and persistence without pretending the readout is a
causal intervention.
"""

from __future__ import annotations

import argparse
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


def build_stage1() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 01 — Repair the coordinate swap (hard G-SWAP)

This notebook repairs the known spider→ant intervention before any science is
run. The workspace band is selected from clean J-Lens visibility only, then a
single fixed token/basis/strength/position configuration is tested on spider
and two predeclared upstream controls. Diagnostics retain the legacy
configuration, alternate token surface, RMS-folded basis, adjacent layer band,
and position-only edits."""
        ),
        nbformat.v4.new_code_cell(
            """import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(Path.home() / '.cache/huggingface')
os.environ['HF_HUB_CACHE'] = str(Path.home() / '.cache/huggingface/hub')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(Path.home() / '.cache/huggingface/hub')

prior = json.loads((ROOT / 'results/metrics.json').read_text())
assert prior['repair_v2']['stage0']['status'] == 'COMPLETE_WITH_RELEASE_OMISSION'
assert prior['repair_v2']['gate_ledger']['g_swap'] in {'UNTESTED', 'PASS', 'FAIL'}

from src.jlens_iface import load_published_lens
from src.model_utils import load_model
from src.v2_repair import MODEL_ID

bundle = load_model(MODEL_ID)
lens = load_published_lens(MODEL_ID)
print({
    'model': bundle.model_id,
    'revision': bundle.revision,
    'dtype': str(next(bundle.hf_model.parameters()).dtype),
    'n_layers': bundle.lens_model.n_layers,
    'lens_source_layers': [min(lens.source_layers), max(lens.source_layers)],
})"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Pre-outcome calibration and repair sweep

The paper's approximately 38–92% depth workspace prior is first mapped to the
28-block model. Inside that prior, the longest contiguous run where the median
minimum source-concept readout rank across three clean prompts is top-10 is
selected. Swap outcomes do not enter band selection. The strict arm uses exact
upstream labels, raw `J.T @ W_U`, `alpha=2`, and all prompt positions."""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_repair import run_stage1

stage1 = run_stage1(bundle, lens)
print('G1', stage1['g1']['status'], 'max mean KL', stage1['g1']['max_prompt_mean_kl'])
print('workspace prior', stage1['workspace_discovery']['paper_prior_layers'])
print('empirical band', stage1['workspace_discovery']['selected_layers'])
print()
print('Canonical rows:')
for row in stage1['g_swap']['canonical_rows']:
    print({
        'item': row['item_name'],
        'tokens': f"{row['source_surface']!r}->{row['target_surface']!r}",
        'clean_top': row['clean_top_token'],
        'edited_top': row['edited_top_token'],
        'clean_M': row['clean_metric'],
        'edited_M': row['edited_metric'],
        'cf_rank': row['counterfactual_answer_rank_after_edit'],
        'argmax_margin': row['counterfactual_argmax_margin'],
        'repeat_error': row['repeat_max_abs_logit_difference'],
        'pass': row['strict_pass'],
    })
print()
print('G-SWAP', stage1['g_swap']['status'])"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Persist the gate decision

A PASS licenses only the concept-direction and READ repair notebooks. It does
not license P1–P3. A FAIL would switch the workflow directly to the Stage-4
replication-failure path."""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_repair import persist_stage1

metrics = persist_stage1(stage1)
gate = metrics['repair_v2']['gate_ledger']['g_swap']
print(json.dumps({
    'G-SWAP': gate,
    'passed': stage1['g_swap']['n_pass'],
    'required': stage1['g_swap']['n_required'],
    'alpha0_no_op_max_error': stage1['alpha0_no_op_max_abs_logit_error'],
    'next': '02_concept_finder' if gate == 'PASS' else '08_report_stage4',
    'science_allowed': False,
}, indent=2))
assert gate == stage1['g_swap']['status']
assert metrics['repair_v2']['gate_ledger']['stage3_science'] == 'PROHIBITED'"""
        ),
        nbformat.v4.new_code_cell(
            """import gc
import torch

del stage1, metrics, lens, bundle
gc.collect()
torch.cuda.empty_cache()
print('Notebook 01 complete. Stage-3 science remains prohibited.')"""
        ),
    ]
    target = ROOT / "notebooks" / "01_repair_swap.ipynb"
    nbformat.write(notebook, target)
    return target


def build_stage2() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 02 — Repair and validate the independent concept finder

G-SWAP has passed, so this notebook may build the mean-difference (MD)
direction family. It reuses the leakage-audited 40-concept cue manifest, then
selects residual pooling and layer by leave-one-training-template-out retrieval
inside the clean-readout-selected Stage-1 workspace. Explicit probe wording is
also selected on training cues only; held-out cues decide G-DIR."""
        ),
        nbformat.v4.new_code_cell(
            """import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(Path.home() / '.cache/huggingface')
os.environ['HF_HUB_CACHE'] = str(Path.home() / '.cache/huggingface/hub')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(Path.home() / '.cache/huggingface/hub')

metrics = json.loads((ROOT / 'results/metrics.json').read_text())
repair = metrics['repair_v2']
assert repair['gate_ledger']['g_swap'] == 'PASS'
workspace_layers = repair['stage1']['g_swap']['canonical_configuration']['layers']

from src.jlens_iface import load_published_lens
from src.model_utils import load_model
from src.v2_repair import MODEL_ID

bundle = load_model(MODEL_ID)
lens = load_published_lens(MODEL_ID)
print('workspace layers fixed before MD heldout evaluation:', workspace_layers)"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_concept import run_stage1c

stage1c = run_stage1c(bundle, lens, workspace_layers=workspace_layers)
r = stage1c['retrieval']['top1_at_fixed_layer']
e = stage1c['explicit_known_answer']
a = stage1c['cosine_alignment']['raw_WU_J']['fixed_layer']
print({
    'G-DIR': stage1c['status'],
    'pooling': stage1c['position_selection']['selected_pooling'],
    'fixed_layer': stage1c['fixed_validation_layer'],
    'retrieval_top1': r['estimate'],
    'retrieval_ci': [r['ci_low'], r['ci_high']],
    'chance': stage1c['chance_retrieval'],
    'explicit_template': e['selected_template'],
    'heldout_exact_top5': e['heldout_top5']['estimate'],
    'cosine_md_raw_jlens': a['estimate'],
})
print('criteria:', stage1c['criteria'])"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_concept import persist_stage1c

metrics = persist_stage1c(stage1c)
gate = metrics['repair_v2']['gate_ledger']['g_dir']
print('Persisted G-DIR:', gate)
assert gate in {'PASS', 'DROPPED_MD'}
assert metrics['repair_v2']['gate_ledger']['stage3_science'] == 'PROHIBITED'"""
        ),
        nbformat.v4.new_code_cell(
            """import gc
import torch

del stage1c, metrics, lens, bundle
gc.collect()
torch.cuda.empty_cache()
print('Notebook 02 complete. Next: READ validation; science remains prohibited.')"""
        ),
    ]
    target = ROOT / "notebooks" / "02_concept_finder.ipynb"
    nbformat.write(notebook, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebooks",
        nargs="*",
        choices=("00", "01", "02"),
        default=("00", "01", "02"),
        help="Notebook numbers to rebuild; specify one to preserve other outputs.",
    )
    arguments = parser.parse_args()
    builders = {"00": build_stage0, "01": build_stage1, "02": build_stage2}
    targets = [builders[name]() for name in arguments.notebooks]
    print(json.dumps([str(path.relative_to(ROOT)) for path in targets], indent=2))


if __name__ == "__main__":
    main()

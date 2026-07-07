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


def build_stage3() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 03 — Repair READ estimators and validate attribution

This notebook separates three quantities previously conflated as one READ
test: the exact local attribution derivative, a measured small-dose slope, and
the nonlinear full-ablation endpoint. It also fixes weight READ's layer bug by
feeding block `k` with `v[k-1]` and evaluating output orientation against
`v[k]`. Weight magnitude remains unsigned and selection-conditioned."""
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
assert repair['gate_ledger']['g_dir'] in {'PASS', 'DROPPED_MD'}
workspace_layers = repair['stage1']['g_swap']['canonical_configuration']['layers']

from src.jlens_iface import load_published_lens
from src.model_utils import load_model
from src.v2_repair import MODEL_ID

bundle = load_model(MODEL_ID)
lens = load_published_lens(MODEL_ID)
print('READ validation band:', workspace_layers)"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_read import run_stage1d

stage1d = run_stage1d(bundle, lens, workspace_layers=workspace_layers)
a = stage1d['attribution']
w = stage1d['weight_read']
print({
    'READ status': stage1d['status'],
    'primary': stage1d['primary_read'],
    'attribution_role': a['role'],
    'exact_derivative_plumbing': a['plumbing_pass_abs_error_le_0.05'],
    'local_r': a['correlations']['predicted_vs_local_slope']['estimate'],
    'full_alpha1_r': a['correlations']['predicted_vs_full_alpha1_delta']['estimate'],
    'read_strength_r': a['correlations']['read_strength_vs_positive_damage']['estimate'],
    'weight_status': w['status'],
    'weight_above_random_cases': w['n_above_random_cases'],
    'weight_positive_orientation_cases': w['n_positive_orientation_cases'],
})
print('weight criteria:', w['criteria'])"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_read import persist_stage1d

metrics = persist_stage1d(stage1d)
gate = metrics['repair_v2']['gate_ledger']['read_validation']
print('Persisted READ validation:', gate)
assert gate in {'PASS', 'FAIL'}
assert metrics['repair_v2']['gate_ledger']['stage3_science'] == 'PROHIBITED'"""
        ),
        nbformat.v4.new_code_cell(
            """import gc
import torch

del stage1d, metrics, lens, bundle
gc.collect()
torch.cuda.empty_cache()
print('Notebook 03 complete. Controls/G-POS are next only if READ passed.')"""
        ),
    ]
    target = ROOT / "notebooks" / "03_read_and_validation.ipynb"
    nbformat.write(notebook, target)
    return target


def build_stage4() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 04 — Recalibration, firing controls, and G-POS

The repaired swap, independent direction gate, and READ validation have all
completed before this notebook runs. Stage 2 now re-verifies G-SWAP, executes
the redesigned controls whose measured metrics actually contain the
suppressed token, and tests the known-narration positive control. Every result
is persisted before the workflow chooses either Stage-3 science or the
Stage-4 replication-failure report."""
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
ledger = repair['gate_ledger']
assert ledger['g_swap'] == 'PASS'
assert ledger['g_dir'] in {'PASS', 'DROPPED_MD'}
assert ledger['read_validation'] == 'PASS'
assert ledger['stage3_science'] == 'PROHIBITED'
workspace_layers = repair['stage1']['g_swap']['canonical_configuration']['layers']

from src.jlens_iface import load_published_lens
from src.model_utils import load_model
from src.v2_repair import MODEL_ID

bundle = load_model(MODEL_ID)
lens = load_published_lens(MODEL_ID)
print({
    'model': bundle.model_id,
    'workspace_layers': workspace_layers,
    'g_swap': ledger['g_swap'],
    'g_dir': ledger['g_dir'],
    'read_validation': ledger['read_validation'],
    'science_before_recalibration': ledger['stage3_science'],
})"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Execute every Stage-2 calibration arm

Failure is recorded rather than raised here. In particular, the notebook
prints the direct and language suppression effects so a nominal control PASS
cannot hide another structural-zero design."""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_recalibration import run_stage2

stage2 = run_stage2(bundle, lens, workspace_layers=workspace_layers)
criteria = stage2.get('criteria', {
    'g_swap_reverified': stage2['g_swap_reverification']['status'] == 'PASS',
    'controls_fire': stage2['controls_fire']['status'] == 'PASS',
    'matched_random_specificity': stage2['random_pair_null']['status'] == 'PASS',
    'absent_coordinate_specificity': stage2['absent_coordinate_null']['status'] == 'PASS',
    'capability_preserved': stage2['capability']['status'] == 'PASS',
    'g_pos_reproduced': stage2['g_pos']['status'] == 'PASS',
})
print(json.dumps({
    'stage2': stage2['status'],
    'criteria': criteria,
    'identity_j_baseline': stage2['identity_j_baseline']['status'],
    'stage3_allowed': stage2['stage3_allowed'],
    'stage4_required': stage2['stage4_required'],
}, indent=2))

print('\\nDirect concept-answer controls:')
for row in stage2['controls_fire']['direct_concept_probes']['rows']:
    print({
        'item': row['name'],
        'clean_pairwise_correct': row['clean_pairwise_correct'],
        'internal_delta': row['internal_delta'],
        'suppression_delta': row['suppression_delta'],
        'suppression_fired': row['suppression_fired'],
    })

print('\\nLanguage controls with firing metrics:')
for row in stage2['controls_fire']['language_controls']['rows']:
    print(row)

print('\\nKnown-narration passage decisions:')
for row in stage2['g_pos']['rows']:
    print({
        'key': row['key'],
        'category': row['category'],
        'minimum_write_rank': row['minimum_all_prompt_language_rank'],
        'instruction_rank_diagnostic': row['minimum_instruction_span_language_rank_diagnostic'],
        'automatic_internal_delta': row['automatic_internal_delta'],
        'primary_weight_read_ratio': row['primary_weight_read_ratio_auto_over_direct'],
        'attribution_read_ratio_secondary': row['attribution_read_ratio_auto_over_direct_secondary'],
        'failed_checks': [
            name for name, passed in row['checks'].items() if not passed
        ],
        'joint_reproduction': row['joint_reproduction'],
    })"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Persist first, then branch

A failed calibration arm is itself the gate result. It sends the workflow to
notebook 08 without licensing P1–P3; it does not become an exception that
prevents the evidence from being saved."""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_recalibration import persist_stage2

metrics = persist_stage2(stage2)
ledger = metrics['repair_v2']['gate_ledger']
expected_branch = (
    'ALLOWED' if stage2['stage3_allowed'] else 'SKIPPED_PREREQUISITE'
)
assert stage2['status'] == ('PASS' if stage2['stage3_allowed'] else 'FAIL')
assert ledger['g_swap_reverify'] == stage2['g_swap_reverification']['status']
assert ledger['controls_fire'] == stage2['controls_fire']['status']
assert ledger['random_pair_null'] == stage2['random_pair_null']['status']
assert ledger['absent_coordinate_null'] == stage2['absent_coordinate_null']['status']
assert ledger['capability'] == stage2['capability']['status']
assert ledger['g_pos'] == stage2['g_pos']['status']
assert ledger['stage3_science'] == expected_branch
next_notebook = (
    '05_science_twohop'
    if expected_branch == 'ALLOWED'
    else '05-07_skip_guards_then_08_report_stage4'
)
print(json.dumps({
    'persisted_stage2': stage2['status'],
    'gate_ledger': ledger,
    'next': next_notebook,
    'hypothesis_inference_allowed': expected_branch == 'ALLOWED',
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """import gc
import torch

del stage2, metrics, lens, bundle
gc.collect()
torch.cuda.empty_cache()
print(
    'Notebook 04 complete. Continue to Stage 3 only when the persisted '
    'ledger says ALLOWED; otherwise execute the Stage-4 report.'
)"""
        ),
    ]
    target = ROOT / "notebooks" / "04_recalibration.ipynb"
    nbformat.write(notebook, target)
    return target


def _build_stage3_skip(
    *,
    number: str,
    title: str,
    filename: str,
    scope: str,
    next_notebook: str,
) -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            f"""# {number} — {title} — SKIPPED_PREREQUISITE

This is an executable gate record, not a Stage-3 science run. Stage 2 required
the replication-failure path, so no model is loaded and no {scope} measurement
is attempted. The cell below fails closed unless the persisted hard-gate
decision explicitly prohibits Stage 3."""
        ),
        nbformat.v4.new_code_cell(
            f"""import json
import os
from pathlib import Path

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)

metrics = json.loads((ROOT / 'results/metrics.json').read_text())
repair = metrics['repair_v2']
stage2 = repair['stage2_recalibration']
ledger = repair['gate_ledger']
assert stage2['stage4_required'] is True
assert stage2['stage3_allowed'] is False
assert ledger['stage3_science'] == 'SKIPPED_PREREQUISITE'

criteria = stage2.get('criteria')
if not isinstance(criteria, dict):
    criteria = {{
        'g_swap_reverified': stage2['g_swap_reverification']['status'] == 'PASS',
        'controls_fire': stage2['controls_fire']['status'] == 'PASS',
        'matched_random_specificity': stage2['random_pair_null']['status'] == 'PASS',
        'absent_coordinate_specificity': stage2['absent_coordinate_null']['status'] == 'PASS',
        'capability_preserved': stage2['capability']['status'] == 'PASS',
        'g_pos_reproduced': stage2['g_pos']['status'] == 'PASS',
    }}

blocking_criteria = {{
    name: value
    for name, value in criteria.items()
    if not (value is True or value == 'PASS')
}}
assert blocking_criteria, 'Stage 4 was required without a recorded failed criterion'

skip = {{
    'notebook': '{number}',
    'status': 'SKIPPED_PREREQUISITE',
    'science_executed': False,
    'model_inference_run': False,
    'scope': '{scope}',
    'blocking_criteria': blocking_criteria,
    'next': '{next_notebook}',
}}
assert skip['status'] == 'SKIPPED_PREREQUISITE'
assert skip['science_executed'] is False
assert skip['model_inference_run'] is False
repair.setdefault('stage3_notebooks', {{}})['{number}'] = skip
(ROOT / 'results/metrics.json').write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + chr(10)
)
print(json.dumps(skip, indent=2))"""
        ),
    ]
    target = ROOT / "notebooks" / filename
    nbformat.write(notebook, target)
    return target


def build_stage5_skip() -> Path:
    return _build_stage3_skip(
        number="05",
        title="Science two-hop",
        filename="05_science_twohop.ipynb",
        scope="P1/P2 two-hop",
        next_notebook="06_science_ambiguity",
    )


def build_stage6_skip() -> Path:
    return _build_stage3_skip(
        number="06",
        title="Science ambiguity",
        filename="06_science_ambiguity.ipynb",
        scope="P3 ambiguity",
        next_notebook="07_scale",
    )


def build_stage7_skip() -> Path:
    return _build_stage3_skip(
        number="07",
        title="Scale comparison",
        filename="07_scale.ipynb",
        scope="cross-scale P1",
        next_notebook="08_report_stage4",
    )


def build_stage8_report() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 08 — Stage-4 replication-failure report

Stage 2 failed its capability and known-narration positive-control gates, so
the three Stage-3 notebooks executed only prerequisite guards. This notebook
is intentionally model-free. It validates those persisted skips, records the
requested legacy predictor comparison with provenance, and closes the run
without making a hypothesis-level inference."""
        ),
        nbformat.v4.new_code_cell(
            """import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

metrics = json.loads((ROOT / 'results/metrics.json').read_text())
repair = metrics['repair_v2']
stage2 = repair['stage2_recalibration']
ledger = repair['gate_ledger']
assert stage2['status'] == 'FAIL'
assert stage2['stage4_required'] is True
assert stage2['stage3_allowed'] is False
assert ledger['stage3_science'] == 'SKIPPED_PREREQUISITE'
for number in ('05', '06', '07'):
    row = repair['stage3_notebooks'][number]
    assert row['status'] == 'SKIPPED_PREREQUISITE'
    assert row['science_executed'] is False
    assert row['model_inference_run'] is False
print({
    'stage2': stage2['status'],
    'blocking_criteria': [
        name for name, passed in stage2['criteria'].items() if not passed
    ],
    'stage3_science': ledger['stage3_science'],
    'model_loaded': False,
})"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v2_final_report import persist_stage4

metrics = persist_stage4()
stage4 = metrics['repair_v2']['stage4_report']
print(json.dumps({
    'classification': stage4['classification'],
    'custom_swap': stage4['custom_swap'],
    'calibration_blockers': stage4['calibration_blockers'],
    'predictions': stage4['predictions'],
    'skipped_notebooks': stage4['skipped_notebooks'],
    'legacy_fallback_comparison': stage4['legacy_fallback_comparison'],
    'claim_boundary': stage4['claim_boundary'],
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """report = (ROOT / 'results/RESULTS.md').read_text()
ledger = metrics['repair_v2']['gate_ledger']
assert ledger['stage4_report'] == 'COMPLETE'
assert ledger['stage3_science'] == 'SKIPPED_PREREQUISITE'
assert report.count('## Stage 4 — replication-failure fallback') == 1
assert 'P1 | **NOT_TESTED**' in report
assert 'P2 | **NOT_TESTED**' in report
assert 'P3 | **NOT_TESTED**' in report
assert 'does not establish that the WRITE-versus-READ hypothesis is' in report
assert all((ROOT / 'results' / row['path']).is_file() for row in stage4['valid_figures'])
print({
    'stage4_report': ledger['stage4_report'],
    'valid_figure_ids': [row['id'] for row in stage4['valid_figures']],
    'final_conclusion': metrics['repair_v2']['current_allowed_conclusion'],
})"""
        ),
    ]
    target = ROOT / "notebooks" / "08_report.ipynb"
    nbformat.write(notebook, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebooks",
        nargs="*",
        choices=("00", "01", "02", "03", "04", "05", "06", "07", "08"),
        default=("00", "01", "02", "03", "04"),
        help=(
            "Notebook numbers to rebuild; specify one to preserve other outputs. "
            "Fallback skip notebooks 05-07 must be requested after Stage 2 fails."
        ),
    )
    arguments = parser.parse_args()
    builders = {
        "00": build_stage0,
        "01": build_stage1,
        "02": build_stage2,
        "03": build_stage3,
        "04": build_stage4,
        "05": build_stage5_skip,
        "06": build_stage6_skip,
        "07": build_stage7_skip,
        "08": build_stage8_report,
    }
    targets = [builders[name]() for name in arguments.notebooks]
    print(json.dumps([str(path.relative_to(ROOT)) for path in targets], indent=2))


if __name__ == "__main__":
    main()

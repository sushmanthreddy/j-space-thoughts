"""Build one hard-gated v3 notebook at a time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


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
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 00 — V3 preflight and v2 instrument re-verification

V3 keeps the completed v2 evidence immutable and asks one new question: can a
smaller, carrying-position intervention retain 3/3 swap efficacy without the
alpha-2 collateral damage? This notebook first records the mandatory fresh
environment checks, then re-runs the bounded v2 sentinels. It does not select
alpha and cannot license science."""
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

from src.v2_stage0 import collect_preflight, print_preflight

preflight = collect_preflight()
print_preflight(preflight)
assert preflight['status'] == 'PASS', preflight"""
        ),
        nbformat.v4.new_code_cell(
            """prior = json.loads((ROOT / 'results/metrics.json').read_text())
repair_v2 = prior['repair_v2']
assert repair_v2['gate_ledger']['g_swap'] == 'PASS'
assert repair_v2['gate_ledger']['g_dir'] == 'PASS'
workspace_layers = repair_v2['stage1']['g_swap']['canonical_configuration']['layers']

from src.jlens_iface import load_published_lens
from src.model_utils import load_model
from src.v2_repair import MODEL_ID

bundle = load_model(MODEL_ID)
lens = load_published_lens(MODEL_ID)
print({
    'model': bundle.model_id,
    'revision': bundle.revision,
    'dtype': str(next(bundle.hf_model.parameters()).dtype),
    'workspace_layers': workspace_layers,
})"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v3_reverify import run_stage0_reverify

stage0 = run_stage0_reverify(
    bundle,
    lens,
    v2_metrics=repair_v2,
    workspace_layers=workspace_layers,
    preflight=preflight,
)
print(json.dumps({
    'status': stage0['status'],
    'checks': stage0['checks'],
    'g1_max_mean_kl': stage0['g1']['max_prompt_mean_kl'],
    'known_swaps': stage0['known_swaps']['n_pass'],
    'gdir_top1': stage0['gdir']['heldout_retrieval_top1'],
    'gdir_top5': stage0['gdir']['known_answer_top5'],
    'controls_fire': stage0['controls_fire']['status'],
}, indent=2))
for row in stage0['known_swaps']['rows']:
    print(row['name'], row['clean_top'], '->', row['edited_top'], row['pass'])"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v3_reverify import persist_stage0

metrics = persist_stage0(stage0)
v3 = metrics['calibration_v3']
assert v3['gate_ledger']['stage0_reverify'] == 'PASS'
assert v3['gate_ledger']['g_swap'] == 'NOT_RUN_V3'
assert v3['gate_ledger']['stage3_science'] == 'PROHIBITED'
print(json.dumps(v3['gate_ledger'], indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """import gc
import torch

del stage0, metrics, lens, bundle
gc.collect()
torch.cuda.empty_cache()
print('Notebook 00 complete. Only notebook 01 G-SWAP is now permitted.')"""
        ),
    ]
    target = ROOT / "notebooks" / "00_preflight_and_reverify.ipynb"
    nbformat.write(notebook, target)
    return target


def build_stage1() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 01 — Confirm the v3 G-SWAP prerequisite

Stage 0 passed. This notebook independently repeats the frozen v2 alpha-2,
all-position sentinel three times per item. It confirms that intervention
machinery has not drifted; alpha=2 is not selected for downstream use."""
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

metrics = json.loads((ROOT / 'results/metrics.json').read_text())
v3 = metrics['calibration_v3']
assert v3['gate_ledger']['stage0_reverify'] == 'PASS'
assert v3['gate_ledger']['stage3_science'] == 'PROHIBITED'
workspace_layers = v3['protocol']['workspace_layers']

from src.jlens_iface import load_published_lens
from src.model_utils import load_model
from src.v2_repair import MODEL_ID

bundle = load_model(MODEL_ID)
lens = load_published_lens(MODEL_ID)"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v3_reverify import run_stage1_confirm

stage1 = run_stage1_confirm(bundle, lens, workspace_layers=workspace_layers)
for row in stage1['g_swap']['rows']:
    print({
        'item': row['name'],
        'swap': f"{row['source']}->{row['target']}",
        'clean_top': row['clean_top'],
        'edited_top': row['edited_top'],
        'clean_M': row['clean_metric'],
        'edited_M': row['edited_metric'],
        'repeat_error': row['repeat_max_abs_logit_error'],
        'pass': row['pass'],
    })
print('G-SWAP', stage1['status'])"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v3_reverify import persist_stage1

metrics = persist_stage1(stage1)
v3 = metrics['calibration_v3']
assert v3['gate_ledger']['g_swap'] == stage1['status']
assert v3['gate_ledger']['stage3_science'] == 'PROHIBITED'
print(json.dumps({
    'g_swap': v3['gate_ledger']['g_swap'],
    'g_alpha': v3['gate_ledger']['g_alpha'],
    'next': '015_alpha_sweep' if stage1['status'] == 'PASS' else '08_report',
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """import gc
import torch

del stage1, metrics, lens, bundle
gc.collect()
torch.cuda.empty_cache()
print('Notebook 01 complete. Science remains prohibited pending G-ALPHA.')"""
        ),
    ]
    target = ROOT / "notebooks" / "01_confirm_swap.ipynb"
    nbformat.write(notebook, target)
    return target


def build_alpha_sweep() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 015 — Surgical intervention alpha sweep (G-ALPHA)

This is the new v3 core. The alpha grid, rank<=10 carrying-position rule,
thresholds, and source-capped primary operator were frozen in notebook 00.
A G-SWAP-only pilot then showed the source-capped operator could not flip the
spider case through alpha=2. The existing fractional coordinate swap restricted
to the same clean carrying mask is therefore included as an exploratory,
nonselectable sensitivity analysis; it was not frozen in notebook 00. The
all-position fractional swap is also a nonselectable reference. Random and
absent-coordinate controls are evaluated for every policy/alpha row."""
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
v3 = metrics['calibration_v3']
repair_v2 = metrics['repair_v2']
assert v3['gate_ledger']['stage0_reverify'] == 'PASS'
assert v3['gate_ledger']['g_swap'] == 'PASS'
assert v3['gate_ledger']['g_alpha'] in {'PENDING', 'FAIL'}
assert v3['gate_ledger']['stage3_science'] in {
    'PROHIBITED', 'SKIPPED_PREREQUISITE'
}
workspace_layers = v3['protocol']['workspace_layers']
print(json.dumps({
    'alpha_grid': v3['protocol']['alpha_grid'],
    'position_rule': v3['protocol']['position_rule'],
    'primary_edit': v3['protocol']['primary_edit'],
    'thresholds': v3['protocol']['thresholds'],
}, indent=2))

from src.jlens_iface import load_published_lens
from src.model_utils import load_model
from src.v2_repair import MODEL_ID

bundle = load_model(MODEL_ID)
lens = load_published_lens(MODEL_ID)"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v3_alpha_sweep import run_alpha_sweep

sweep = run_alpha_sweep(
    bundle,
    lens,
    repair_v2=repair_v2,
    workspace_layers=workspace_layers,
)
print('mask manifest SHA256', sweep['mask_manifest']['sha256'])
print('Known-answer carrying masks:')
for name, mask in sweep['mask_manifest']['known'].items():
    print(name, mask['positions'], f"{len(mask['positions'])}/{mask['sequence_length']}")
print()
print('Mask-specific primary weight-READ ratios:')
for key, row in sweep['masked_weight_read'].items():
    print(key, row['primary_ratio'], row['automatic_mask']['positions'])
print()
print('Full alpha table:')
for row in sweep['rows']:
    print({
        'policy': row['policy'],
        'alpha': row['alpha'],
        'swaps': row['known_swaps']['n_pass'],
        'mean_delta_nll': row['capability']['mean_delta_nll'],
        'gpos': row['g_pos']['n_reproduced'],
        'random': row['random_null']['status'],
        'absent': row['absent_null']['status'],
        'valid': row['valid'],
    })
print(json.dumps({
    'G-ALPHA': sweep['g_alpha'],
    'selected_intervention': sweep['selected_intervention'],
    'raw_artifact': sweep['raw_artifact'],
    'raw_artifact_sha256': sweep['raw_artifact_sha256'],
    'raw_artifact_bytes': sweep['raw_artifact_bytes'],
    'figure': sweep['figure'],
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v3_alpha_sweep import persist_alpha_sweep

metrics = persist_alpha_sweep(sweep)
v3 = metrics['calibration_v3']
assert v3['gate_ledger']['g_alpha'] == sweep['g_alpha']
assert v3['gate_ledger']['stage3_science'] in {
    'PROHIBITED', 'SKIPPED_PREREQUISITE'
}
next_notebook = (
    '04_recalibration_at_alpha'
    if sweep['g_alpha'] == 'PASS'
    else '04_recalibration_skip_then_stage4'
)
print(json.dumps({
    'g_alpha': v3['gate_ledger']['g_alpha'],
    'stage2': v3['gate_ledger']['stage2_recalibration'],
    'stage3': v3['gate_ledger']['stage3_science'],
    'next': next_notebook,
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """import gc
import torch

del sweep, metrics, lens, bundle
gc.collect()
torch.cuda.empty_cache()
print('Notebook 015 complete. No science was run.')"""
        ),
    ]
    target = ROOT / "notebooks" / "015_alpha_sweep.ipynb"
    nbformat.write(notebook, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", choices=("00", "01", "015"))
    arguments = parser.parse_args()
    builders = {"00": build_stage0, "01": build_stage1, "015": build_alpha_sweep}
    target = builders[arguments.notebook]()
    print(json.dumps({"built": str(target.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()

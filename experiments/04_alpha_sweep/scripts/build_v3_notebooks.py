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


def build_stage2_skip() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 04 — Stage-2 recalibration prerequisite record

G-ALPHA failed and no alpha* was selected. Stage 2 is therefore undefined.
This executed notebook records the skip without loading a model, recalibrating
at another strength, or relabeling Stage-0 sentinels as alpha*-specific gates."""
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
v3 = metrics['calibration_v3']
assert v3['gate_ledger']['g_alpha'] == 'FAIL'
assert v3['selected_intervention'] is None
assert v3['gate_ledger']['stage3_science'] == 'SKIPPED_PREREQUISITE'
print('Prerequisite guard: G-ALPHA FAIL; alpha*=None')"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v3_fallback import record_stage2_skip

metrics = record_stage2_skip()
stage2 = metrics['calibration_v3']['stage2_recalibration']
assert stage2['status'] == 'SKIPPED_PREREQUISITE'
assert stage2['model_forward_run'] is False
print(json.dumps(stage2, indent=2))
print('Notebook 04 complete. No model forward or Stage-2 gate was run.')"""
        ),
    ]
    target = ROOT / "notebooks" / "04_recalibration.ipynb"
    nbformat.write(notebook, target)
    return target


def build_stage3_skip(notebook_name: str) -> Path:
    metadata = {
        "05_science_twohop.ipynb": ("05", "P1/P2 two-hop and narration"),
        "06_science_ambiguity.ipynb": ("06", "P3 ambiguity"),
        "07_scale.ipynb": ("07", "P1 scale"),
    }
    number, scope = metadata[notebook_name]
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            f"""# {number} — {scope} prerequisite record

Stage 3 is prohibited because G-ALPHA failed and Stage 2 was skipped. This
notebook records `SKIPPED_PREREQUISITE` model-free. It does not import or
reinterpret v1/v2 science values."""
        ),
        nbformat.v4.new_code_cell(
            f"""import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

metrics = json.loads((ROOT / 'results/metrics.json').read_text())
v3 = metrics['calibration_v3']
assert v3['gate_ledger']['g_alpha'] == 'FAIL'
assert v3['stage2_recalibration']['status'] == 'SKIPPED_PREREQUISITE'
assert v3['selected_intervention'] is None

from src.v3_fallback import record_stage3_skip

metrics = record_stage3_skip('{notebook_name}')
entry = metrics['calibration_v3']['stage3_notebooks']['{notebook_name}']
assert entry['status'] == 'SKIPPED_PREREQUISITE'
assert entry['model_forward_run'] is False
assert entry['science_values_loaded'] is False
print(json.dumps(entry, indent=2))
print('Notebook {number} complete. No science was run.')"""
        ),
    ]
    target = ROOT / "notebooks" / notebook_name
    nbformat.write(notebook, target)
    return target


def build_stage4_report() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 08 — V3 Stage-4 calibration-limitation report

This notebook assembles only evidence licensed by the failed gate chain. It
publishes a calibration/READ-positive-control limitation and explicitly leaves
P1-P3 untested. The v1 correlation comparison is descriptive legacy evidence
from an invalidated instrument."""
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
v3 = metrics['calibration_v3']
assert v3['gate_ledger']['g_alpha'] == 'FAIL'
assert v3['selected_intervention'] is None
required = {
    '05_science_twohop.ipynb',
    '06_science_ambiguity.ipynb',
    '07_scale.ipynb',
}
assert set(v3['stage3_notebooks']) == required
assert all(
    row['status'] == 'SKIPPED_PREREQUISITE'
    for row in v3['stage3_notebooks'].values()
)
print('Gate chain verified: Stage 4 fallback is required.')"""
        ),
        nbformat.v4.new_code_cell(
            """from src.v3_fallback import record_stage4_fallback

metrics = record_stage4_fallback()
v3 = metrics['calibration_v3']
stage4 = v3['stage4_fallback']
assert stage4['status'] == 'COMPLETE'
assert stage4['claim_boundary']['hypothesis_status'] == 'NOT_TESTED'
assert stage4['claim_boundary']['hypothesis_false_established'] is False
assert v3['gate_ledger']['stage4_report'] == 'PASS'
print(json.dumps({
    'classification': stage4['classification'],
    'failed_gate': stage4['failed_gate'],
    'predictions': stage4['predictions'],
    'claim_boundary': stage4['claim_boundary'],
    'raw_artifact': stage4['raw_artifact'],
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """report = (ROOT / 'results/RESULTS.md').read_text()
required_phrases = [
    'V3 COMPLETE',
    'CALIBRATION_READ_POSITIVE_CONTROL_LIMITATION',
    'P1, P2, and P3 are **NOT TESTED**',
    'does **not** show that the Written-vs-Read hypothesis is false',
    '24/24',
    'N=155',
]
assert all(phrase in report for phrase in required_phrases)
print(report)
print('Notebook 08 complete. V3 report persisted without a hypothesis verdict.')"""
        ),
    ]
    target = ROOT / "notebooks" / "08_report.ipynb"
    nbformat.write(notebook, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook", choices=("00", "01", "015", "04", "05", "06", "07", "08")
    )
    arguments = parser.parse_args()
    builders = {
        "00": build_stage0,
        "01": build_stage1,
        "015": build_alpha_sweep,
        "04": build_stage2_skip,
        "05": lambda: build_stage3_skip("05_science_twohop.ipynb"),
        "06": lambda: build_stage3_skip("06_science_ambiguity.ipynb"),
        "07": lambda: build_stage3_skip("07_scale.ipynb"),
        "08": build_stage4_report,
    }
    target = builders[arguments.notebook]()
    print(json.dumps({"built": str(target.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()

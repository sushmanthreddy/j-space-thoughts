"""Build the fixed Road-A v4 notebooks without adding estimator variants."""

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


def build_notebook10() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 10 — The one behavior-specific READ attempt

This notebook freezes and executes the only new READ estimator permitted in
v4. It does not tune alpha, thresholds, component counts, or estimator form.

Protocol frozen before model outcomes:

- validation roster: the frozen v2 20-item READ roster plus the missing
  canonical `spider-legs` item, deduplicated (`N=21`);
- causal endpoint: the v4-authorized reuse of the exploratory v3 masked
  fractional source-to-foil swap, alpha=1.5, layers 13–24, clean rank<=10
  carrying-position union; this is a v4 protocol override, not a v3-selected
  intervention;
- path discovery: source-only unit projection deletion at one clean-only source
  layer (minimum source-token J-Lens rank; lower layer/position tie-break), on
  the same carrying positions; receivers are strictly downstream;
- path effect: exact edited-into-clean one-component patching, not grad×delta;
- `S_M`: all MLPs/heads with absolute exact patched contribution to `M` >=0.05
  logit units; no top-k or empty-set fallback;
- READ weights: the repaired v2/v3 layer alignment, 32 random directions and
  identical seeds; family means combined 50/50;
- an empty `S_M` is labeled `NO_PATH_DETECTED`, retained in the locked roster,
  and cannot silently pass validation.

The path-discovery deletion is deliberately distinct from the fixed swap used
for ground-truth CAUSAL, reducing endpoint reuse and avoiding target-injection
contamination in `S_M`."""
        ),
        nbformat.v4.new_code_cell(
            """import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(Path.home() / '.cache/huggingface')
os.environ['HF_HUB_CACHE'] = str(Path.home() / '.cache/huggingface/hub')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(Path.home() / '.cache/huggingface/hub')

from src.metrics import save_json
from src.v2_stage0 import collect_preflight, print_preflight
from src.v3_reverify import _repair_v2_sha256

preflight = collect_preflight()
print_preflight(preflight)
assert preflight['status'] == 'PASS'

metrics_path = ROOT / 'results/metrics.json'
metrics = json.loads(metrics_path.read_text())
v3 = metrics['calibration_v3']
assert v3['gate_ledger']['stage4_report'] == 'PASS'
assert v3['gate_ledger']['g_swap'] == 'PASS'
assert v3['gate_ledger']['g_alpha'] == 'FAIL'
assert v3['selected_intervention'] is None
assert _repair_v2_sha256(metrics['repair_v2']) == v3['provenance']['repair_v2_sha256']

canonical = ['spider-legs', 'animal-legs-buffalo2', 'chem-photosynthesis-Z']
prior_roster = metrics['repair_v2']['stage1d_read_validation']['validation_selection']['items']
roster = canonical + [name for name in prior_roster if name not in canonical]
assert len(roster) == 21 and all(name in roster for name in canonical)

protocol = {
    'schema_version': 'behavior-specific-read-v4-protocol',
    'new_read_estimators_permitted': 1,
    'estimator': 'behavior-specific path-restricted weight READ',
    'model_id': 'Qwen/Qwen2.5-7B-Instruct',
    'model_revision': v3['protocol']['model_revision'],
    'dtype': 'torch.bfloat16',
    'validation_roster': roster,
    'validation_n': len(roster),
    'workspace_layers': list(range(13, 25)),
    'causal_intervention': {
        'policy': 'fractional_swap_carrying_positions',
        'alpha': 1.5,
        'direction': 'raw normalize(J.T @ W_U[token])',
        'position_rule': 'clean source-label J-Lens rank<=10 union over L13-24',
        'provenance': 'v4-authorized override of v3 exploratory nonselectable policy',
        'alpha_resweep': False,
    },
    'path_discovery': {
        'perturbation': 'source-only unit projection deletion',
        'strength': 1.0,
        'source_layer_rule': 'minimum clean source-token J-Lens rank; lower layer then position tie-break',
        'positions': 'same frozen clean carrying-position union',
        'receiver_layers': 'strictly downstream through final block',
        'patch': 'edited component into otherwise clean run across all sequence positions',
        'component_types': ['MLP output', 'attention head immediately before o_proj'],
    },
    'path_threshold_abs_delta_m': 0.05,
    'path_selection': 'abs(exact patched contribution)>=0.05; no top-k/fallback',
    'weight_normalization': {
        'n_random': 32,
        'mlp': 'mean random-normalized MLP gain over S_M MLPs',
        'attention': 'mean random-normalized OV norm times abs label cosine over S_M heads',
        'composite': '0.5*MLP + 0.5*attention',
        'layer_alignment': 'v[layer-1] -> component -> compare with v[layer]',
        'seed_schedule': 'identical to repaired v2/v3',
    },
    'known_gate': {
        'rho_min': 0.4,
        'bootstrap_draws': 5000,
        'bootstrap_seed': 1729,
        'bootstrap_unit': 'source-concept cluster',
        'ci_lower_strictly_positive': True,
        'all_locked_rows_must_be_estimable': True,
    },
    'narration_gate': {
        'max_read_ratio': 0.5,
        'min_passages': 6,
        'min_languages': 3,
        'requires_low_causal_abs_delta': 0.5,
        'requires_clean_capable': True,
        'empty_auto_path_counts_as_low': False,
        'empty_direct_denominator': 'NOT_ESTIMABLE',
    },
}
protocol_sha = hashlib.sha256(
    json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()
).hexdigest()
existing = metrics.get('calibration_v4')
if existing is not None:
    assert existing['protocol_sha256'] == protocol_sha
    assert existing['stage10']['status'] in {'RUNNING', 'COMPLETE'}
metrics['calibration_v4'] = {
    'schema_version': 'road-a-one-shot-v4',
    'protocol': protocol,
    'protocol_sha256': protocol_sha,
    'preflight': preflight,
    'stage10': {'status': 'RUNNING'},
    'gate_ledger': {
        'g_readval': 'PENDING',
        'stage_a_science': 'PROHIBITED',
        'stage_b_report': 'PENDING',
    },
    'current_allowed_conclusion': 'READ_ESTIMATOR_ATTEMPT_IN_PROGRESS_NO_SCIENCE',
}
save_json(metrics_path, metrics)
print(json.dumps({'protocol_sha256': protocol_sha, 'roster': roster}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """import torch

from src.jlens_iface import load_published_lens
from src.model_utils import load_model
from src.v2_repair import MODEL_ID
from src.v3_reverify import _g1

bundle = load_model(MODEL_ID)
lens = load_published_lens(MODEL_ID)
assert next(bundle.hf_model.parameters()).dtype == torch.bfloat16
assert bundle.revision == protocol['model_revision']
g1 = _g1(bundle)
assert g1['status'] == 'PASS', g1
print(json.dumps({
    'model': bundle.model_id,
    'revision': bundle.revision,
    'dtype': str(next(bundle.hf_model.parameters()).dtype),
    'g1_max_mean_kl': g1['max_prompt_mean_kl'],
    'g1_n': g1['n'],
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """from src.controls_phase import load_known_narration_source
from src.data_gen import load_probe_swap_items
from src.jlens_iface import token_rank
from src.v2_read import _tokenize_item
from src.v3_alpha_sweep import (
    _build_bank,
    _language_direction_ids,
    _masked_weight_read,
    _prepare_gpos,
    _single_token_families,
)

raw_by_name = {row['name']: row for row in load_probe_swap_items()}
items = [_tokenize_item(bundle.tokenizer, raw_by_name[name]) for name in roster]
narration = load_known_narration_source()
token_families = _single_token_families(bundle.tokenizer)
language_ids, english_id = _language_direction_ids(bundle, narration)
token_ids = {
    int(token_id)
    for item in items
    for token_id in (item['source_concept_token_id'], item['target_concept_token_id'])
} | set(language_ids.values()) | {english_id}
workspace_layers = protocol['workspace_layers']
bank = _build_bank(bundle, lens, token_ids, workspace_layers)
assert all(set(range(13, 28)).issubset(bank[token_id]) for token_id in token_ids)

v2_gpos = metrics['repair_v2']['stage2_recalibration']['g_pos']
gpos_prepared = _prepare_gpos(
    bundle,
    lens,
    narration,
    v2_gpos,
    bank,
    language_ids,
    english_id,
    token_families,
    workspace_layers,
)
old_narration = _masked_weight_read(bundle, gpos_prepared)
print({'known_items': len(items), 'narration_items': len(gpos_prepared), 'direction_tokens': len(token_ids)})"""
        ),
        nbformat.v4.new_code_cell(
            """import gc
import math

from src.interventions import ablation_edits, clamped_swap_edits, forward_logits
from src.metrics import logit_difference
from src.model_utils import capture_residuals
from src.read_scores import behavior_specific_read, exact_path_patch_scores
from src.v2_recalibration import _task_read
from src.v3_alpha_sweep import _mask_for_prompt

PATH_THRESHOLD = protocol['path_threshold_abs_delta_m']
N_RANDOM = protocol['weight_normalization']['n_random']
SEED = protocol['known_gate']['bootstrap_seed']

def best_clean_source_layer(mask):
    candidates = [
        (rank, int(layer), position)
        for layer, ranks in mask['ranks_by_layer_position'].items()
        for position, rank in enumerate(ranks)
    ]
    rank, layer, position = min(candidates)
    return {'rank': int(rank), 'layer': layer, 'position': int(position)}

def compact_read(read):
    return {
        'status': read['status'],
        's_m': read['s_m'],
        'mlp_primary': read['mlp_primary'],
        'attention_primary': read['attention_primary'],
        'equal_family_composite': read['equal_family_composite'],
        'family_status': read['family_status'],
        'metadata': read['metadata'],
    }

known_rows = []
known_raw = []
clean_patch_audit = None
for index, item in enumerate(items):
    prompt = str(item['prompt'])
    input_ids = bundle.lens_model.encode(prompt)
    clean = forward_logits(bundle.hf_model, input_ids)
    clean_top = int(clean[0, -1].argmax())
    assert clean_top == int(item['clean_answer_token_id']), item['name']
    mask, _ = _mask_for_prompt(
        bundle,
        lens,
        prompt,
        int(item['source_concept_token_id']),
        workspace_layers,
    )
    assert mask['positions'], f"locked row has empty carrying mask: {item['name']}"
    source_layer_record = best_clean_source_layer(mask)
    source_layer = source_layer_record['layer']
    component_layers = list(range(source_layer + 1, len(bundle.lens_model.layers)))
    directions = bank[int(item['source_concept_token_id'])]
    target_directions = bank[int(item['target_concept_token_id'])]
    residuals = capture_residuals(bundle.lens_model, input_ids, workspace_layers)
    causal_edits = clamped_swap_edits(
        residuals,
        {layer: directions[layer] for layer in workspace_layers},
        {layer: target_directions[layer] for layer in workspace_layers},
        positions=mask['positions'],
        strength=1.5,
    )
    edited = forward_logits(
        bundle.hf_model,
        input_ids,
        blocks=bundle.lens_model.layers,
        edits=causal_edits,
    )
    target_id = int(item['clean_answer_token_id'])
    foil_id = int(item['counterfactual_answer_token_id'])

    def metric_fn(logits, target=target_id, foil=foil_id):
        return logits[0, -1, target].float() - logits[0, -1, foil].float()

    clean_metric = float(metric_fn(clean).cpu())
    edited_metric = float(metric_fn(edited).cpu())
    discovery_edits = ablation_edits(
        {source_layer: directions[source_layer]},
        positions=mask['positions'],
        strength=1.0,
    )
    path = exact_path_patch_scores(
        bundle.hf_model,
        bundle.lens_model.layers,
        input_ids,
        discovery_edits,
        metric_fn,
        component_layers=component_layers,
    )
    if clean_patch_audit is None:
        no_op = exact_path_patch_scores(
            bundle.hf_model,
            bundle.lens_model.layers,
            input_ids,
            {},
            metric_fn,
            component_layers=component_layers,
        )
        clean_patch_audit = {
            'item': item['name'],
            'actual_delta': no_op['actual_delta'],
            'max_abs_component_patch': max(
                row['abs_patched_contribution']
                for row in (*no_op['mlps'], *no_op['attention_heads'])
            ),
            'threshold': PATH_THRESHOLD,
        }
        assert clean_patch_audit['max_abs_component_patch'] < PATH_THRESHOLD
    behavior_read = behavior_specific_read(
        bundle.lens_model.layers,
        directions,
        path,
        path_threshold=PATH_THRESHOLD,
        n_random=N_RANDOM,
        seed=SEED + 10_000 * index,
    )
    old = _task_read(
        bundle,
        input_ids,
        directions,
        metric_fn,
        source_layer=source_layer,
        intervention_positions=mask['positions'],
        seed=SEED + 10_000 * index,
    )
    old_compact = {
        'mlp_primary': old['weight']['mlp_primary'],
        'attention_primary': old['weight']['attention_primary'],
        'equal_family_composite': old['weight']['equal_family_composite'],
        'component_ids': [
            row['component']
            for row in (*old['weight']['mlps'], *old['weight']['attention_heads'])
        ],
        'selection': 'legacy grad-delta top-2 MLP/top-4 head',
    }
    compact = {
        'name': item['name'],
        'prompt': prompt,
        'source_concept': item['intermediate'],
        'target_concept': item['swap_to'],
        'source_concept_token_id': int(item['source_concept_token_id']),
        'target_concept_token_id': int(item['target_concept_token_id']),
        'cluster': item['intermediate'],
        'carrying_mask': mask['positions'],
        'source_layer': source_layer_record,
        'clean_metric': clean_metric,
        'edited_metric': edited_metric,
        'causal_signed_clean_minus_edited': clean_metric - edited_metric,
        'causal_abs': abs(clean_metric - edited_metric),
        'clean_top': bundle.tokenizer.decode([clean_top]),
        'edited_top': bundle.tokenizer.decode([int(edited[0, -1].argmax())]),
        'behavior_specific_read': compact_read(behavior_read),
        'old_global_read': old_compact,
        'estimable': behavior_read['status'] == 'OK',
    }
    known_rows.append(compact)
    known_raw.append({
        **compact,
        'path_patch_scores': path,
        'behavior_specific_read_full': behavior_read,
        'old_global_read_full': old,
    })
    print(
        f"[{index + 1:02d}/{len(items)}] {item['name']}: "
        f"|CAUSAL|={compact['causal_abs']:.4f}, "
        f"READ={behavior_read['equal_family_composite']:.4f}, "
        f"|S_M|={behavior_read['s_m']['n_components']}"
    )
    del clean, edited, residuals, path, behavior_read, old
    gc.collect()
    torch.cuda.empty_cache()

assert len(known_rows) == len(roster)
print('clean-to-clean patch audit', clean_patch_audit)"""
        ),
        nbformat.v4.new_code_cell(
            """def behavior_ratio(auto, direct):
    ratios = {}
    for family, count_key, value_key in (
        ('mlp', 'n_mlps', 'mlp_primary'),
        ('attention', 'n_attention_heads', 'attention_primary'),
    ):
        auto_count = int(auto['s_m'][count_key])
        direct_count = int(direct['s_m'][count_key])
        if auto_count == 0:
            continue
        denominator = float(direct[value_key])
        if direct_count == 0 or not math.isfinite(denominator) or denominator <= 1e-8:
            return {
                'status': 'NOT_ESTIMABLE_EMPTY_DIRECT_FAMILY',
                'primary_ratio': None,
                'family_ratios': ratios,
            }
        ratios[family] = float(auto[value_key]) / denominator
    if not ratios:
        return {
            'status': 'NO_AUTO_PATH_DETECTED',
            'primary_ratio': None,
            'family_ratios': {},
        }
    return {
        'status': 'OK',
        'primary_ratio': max(ratios.values()),
        'family_ratios': ratios,
    }

narration_rows = []
narration_raw = []
for row in gpos_prepared:
    key = row['key']
    auto_source_layer = int(row['source_layer'])
    direct_source_layer = best_clean_source_layer(row['direct_mask'])['layer']
    auto_component_layers = list(range(auto_source_layer + 1, len(bundle.lens_model.layers)))
    direct_component_layers = list(range(direct_source_layer + 1, len(bundle.lens_model.layers)))
    auto_discovery = ablation_edits(
        {auto_source_layer: row['directions'][auto_source_layer]},
        positions=row['mask']['positions'],
        strength=1.0,
    )
    direct_discovery = ablation_edits(
        {direct_source_layer: row['directions'][direct_source_layer]},
        positions=row['direct_mask']['positions'],
        strength=1.0,
    )
    auto_path = exact_path_patch_scores(
        bundle.hf_model,
        bundle.lens_model.layers,
        row['input_ids'],
        auto_discovery,
        row['auto_metric_fn'],
        component_layers=auto_component_layers,
    )
    direct_path = exact_path_patch_scores(
        bundle.hf_model,
        bundle.lens_model.layers,
        row['direct_ids'],
        direct_discovery,
        row['direct_metric_fn'],
        component_layers=direct_component_layers,
    )
    base_seed = SEED + 100_000 + int(row['index']) * 1_000
    auto_read = behavior_specific_read(
        bundle.lens_model.layers,
        row['directions'],
        auto_path,
        path_threshold=PATH_THRESHOLD,
        n_random=N_RANDOM,
        seed=base_seed,
    )
    direct_read = behavior_specific_read(
        bundle.lens_model.layers,
        row['directions'],
        direct_path,
        path_threshold=PATH_THRESHOLD,
        n_random=N_RANDOM,
        seed=base_seed,
    )
    ratio = behavior_ratio(auto_read, direct_read)
    causal_edits = clamped_swap_edits(
        row['residuals'],
        row['source'],
        row['target'],
        positions=row['mask']['positions'],
        strength=1.5,
    )
    edited = forward_logits(
        bundle.hf_model,
        row['input_ids'],
        blocks=bundle.lens_model.layers,
        edits=causal_edits,
    )
    edited_metric = float(row['auto_metric_fn'](edited).cpu())
    causal_delta = edited_metric - float(row['auto_clean_metric'])
    global_row = old_narration[key]
    compact = {
        'key': key,
        'language': row['category'],
        'source_layer_auto': auto_source_layer,
        'source_layer_direct': direct_source_layer,
        'auto_mask': row['mask']['positions'],
        'direct_mask': row['direct_mask']['positions'],
        'clean_metric': float(row['auto_clean_metric']),
        'edited_metric': edited_metric,
        'causal_delta': causal_delta,
        'causal_abs': abs(causal_delta),
        'clean_capable': float(row['auto_clean_metric']) > 0.0,
        'old_global_primary_ratio': global_row['primary_ratio'],
        'old_global_mlp_ratio': global_row['mlp_ratio'],
        'old_global_attention_ratio': global_row['attention_ratio'],
        'behavior_specific_ratio': ratio,
        'behavior_specific_auto': compact_read(auto_read),
        'behavior_specific_direct': compact_read(direct_read),
    }
    narration_rows.append(compact)
    narration_raw.append({
        **compact,
        'auto_path_patch_scores': auto_path,
        'direct_path_patch_scores': direct_path,
        'behavior_specific_auto_full': auto_read,
        'behavior_specific_direct_full': direct_read,
        'old_global_full': global_row,
    })
    display_ratio = ratio['primary_ratio']
    print(
        key,
        'global=', global_row['primary_ratio'],
        'behavior-specific=', display_ratio,
        '|CAUSAL|=', abs(causal_delta),
        '|S_auto|=', auto_read['s_m']['n_components'],
        '|S_direct|=', direct_read['s_m']['n_components'],
    )
    del auto_path, direct_path, auto_read, direct_read, edited
    gc.collect()
    torch.cuda.empty_cache()

assert len(narration_rows) == 8"""
        ),
        nbformat.v4.new_code_cell(
            """import datetime

raw_artifact = {
    'schema_version': 'behavior-specific-read-v4-raw',
    'protocol': protocol,
    'protocol_sha256': protocol_sha,
    'model': {
        'id': bundle.model_id,
        'revision': bundle.revision,
        'dtype': str(next(bundle.hf_model.parameters()).dtype),
        'logit_agreement': g1,
    },
    'clean_patch_audit': clean_patch_audit,
    'known_answer_rows': known_raw,
    'narration_rows': narration_raw,
}
raw_path = ROOT / 'data/raw/v4/10_behavior_specific_read.json'
save_json(raw_path, raw_artifact)
raw_bytes = raw_path.stat().st_size
raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

metrics = json.loads(metrics_path.read_text())
assert metrics['calibration_v4']['protocol_sha256'] == protocol_sha
assert _repair_v2_sha256(metrics['repair_v2']) == metrics['calibration_v3']['provenance']['repair_v2_sha256']
metrics['calibration_v4']['stage10'] = {
    'status': 'COMPLETE',
    'completed_utc': datetime.datetime.now(datetime.UTC).isoformat(),
    'model': {
        'id': bundle.model_id,
        'revision': bundle.revision,
        'dtype': str(next(bundle.hf_model.parameters()).dtype),
        'logit_agreement': g1,
    },
    'clean_patch_audit': clean_patch_audit,
    'known_answer': {
        'n': len(known_rows),
        'rows': known_rows,
        'n_estimable': sum(row['estimable'] for row in known_rows),
    },
    'narration': {'n': len(narration_rows), 'rows': narration_rows},
    'raw_artifact': str(raw_path.relative_to(ROOT)),
    'raw_artifact_bytes': raw_bytes,
    'raw_artifact_sha256': raw_sha,
    'path_patching': {
        'threshold': PATH_THRESHOLD,
        'known_s_m_sizes': {
            row['name']: row['behavior_specific_read']['s_m']['n_components']
            for row in known_rows
        },
        'narration_auto_s_m_sizes': {
            row['key']: row['behavior_specific_auto']['s_m']['n_components']
            for row in narration_rows
        },
        'narration_direct_s_m_sizes': {
            row['key']: row['behavior_specific_direct']['s_m']['n_components']
            for row in narration_rows
        },
    },
    'contrast': 'legacy grad-delta top-k global READ versus exact-path-threshold restricted READ',
    'limitations': [
        'The 21-item roster is previously screened calibration data, not an untouched holdout.',
        'S_M is behavior-metric-specific and selection-conditioned, although discovered with source-only deletion distinct from the causal swap endpoint.',
        'Narration passages are reused positive controls, not independent generalization data.',
        'The alpha=1.5 masked swap is a v4-authorized override of a v3 exploratory nonselectable policy.',
    ],
}
save_json(metrics_path, metrics)

report = f'''# Behavior-specific READ validation report (v4)\n\n## Current status\n\n**ONE ESTIMATOR EXECUTED; G-READVAL PENDING. SCIENCE PROHIBITED.**\n\n## Preflight and fixed protocol\n\n- GPU: {preflight['gpu']['name']}; total VRAM {preflight['gpu']['memory_total_mib']} MiB; free {preflight['gpu']['memory_free_mib']} MiB.\n- Home/HF filesystem free: {preflight['disk']['free_gib']:.1f} GiB.\n- Model: `{bundle.model_id}` at `{bundle.revision}` in `{next(bundle.hf_model.parameters()).dtype}`.\n- HF/J-Lens max mean KL: {g1['max_prompt_mean_kl']:.3e} (N={g1['n']}, threshold 1e-3): **PASS**.\n- New READ estimators added: **1**. No alpha resweep was run.\n- Locked known-answer roster: N={len(known_rows)} (prior v2 20 plus spider).\n- Fixed causal endpoint: masked fractional source-to-foil swap, alpha=1.5, L13-24.\n- Path discovery: source-only unit deletion at a clean minimum-rank layer.\n- Exact path threshold: `|patched delta M| >= {PATH_THRESHOLD}`; no top-k/fallback.\n- Clean-to-clean maximum component patch: {clean_patch_audit['max_abs_component_patch']:.3e}.\n- Raw artifact: `{raw_path.relative_to(ROOT)}` (SHA-256 `{raw_sha}`).\n\n## Notebook 10 — path-restricted READ built\n\n- Known-answer estimable rows: {sum(row['estimable'] for row in known_rows)}/{len(known_rows)}.\n- Known-answer |S_M| range: {min(row['behavior_specific_read']['s_m']['n_components'] for row in known_rows)}–{max(row['behavior_specific_read']['s_m']['n_components'] for row in known_rows)}.\n- Narration auto |S_M| range: {min(row['behavior_specific_auto']['s_m']['n_components'] for row in narration_rows)}–{max(row['behavior_specific_auto']['s_m']['n_components'] for row in narration_rows)}.\n- Narration direct |S_M| range: {min(row['behavior_specific_direct']['s_m']['n_components'] for row in narration_rows)}–{max(row['behavior_specific_direct']['s_m']['n_components'] for row in narration_rows)}.\n\nNotebook 11 must now apply the frozen G-READVAL bars. No hypothesis science has run.\n'''
(ROOT / 'results/RESULTS.md').write_text(report, encoding='utf-8')
print(json.dumps({
    'stage10': 'COMPLETE',
    'known_n': len(known_rows),
    'known_estimable': sum(row['estimable'] for row in known_rows),
    'narration_n': len(narration_rows),
    'raw_bytes': raw_bytes,
    'raw_sha256': raw_sha,
    'next': '11_readval_gate',
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """del bank, gpos_prepared, old_narration, lens, bundle
gc.collect()
torch.cuda.empty_cache()
print('Notebook 10 complete. G-READVAL remains pending; science is prohibited.')"""
        ),
    ]
    target = ROOT / "notebooks" / "10_behavior_specific_read.ipynb"
    nbformat.write(notebook, target)
    return target


def build_notebook11() -> Path:
    notebook = nbformat.v4.new_notebook(metadata=_metadata())
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            """# 11 — Hard G-READVAL gate

This notebook applies the thresholds frozen before notebook-10 outcomes. It is
model-free and may not alter the estimator, path threshold, alpha, roster, or
empty-path policy.

Both subgates must pass:

1. locked `N=21` known-answer Spearman rho >=0.4 with source-concept-cluster
   bootstrap 95% CI lower bound >0, and every row estimable;
2. at least 6/8 narration passages across at least three languages must have a
   finite behavior-specific auto/direct ratio <=0.50, low causal change, and a
   clean-capable baseline. `NO_AUTO_PATH_DETECTED` is not a low-READ pass."""
        ),
        nbformat.v4.new_code_cell(
            """import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/home/jovyan/j-space-thoughts')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.metrics import save_json
from src.plotting import save_figure, set_style
from src.v3_reverify import _repair_v2_sha256

metrics_path = ROOT / 'results/metrics.json'
metrics = json.loads(metrics_path.read_text())
v4 = metrics['calibration_v4']
assert v4['stage10']['status'] == 'COMPLETE'
assert v4['gate_ledger']['g_readval'] == 'PENDING'
assert v4['gate_ledger']['stage_a_science'] == 'PROHIBITED'
assert v4['protocol']['new_read_estimators_permitted'] == 1
assert v4['protocol']['causal_intervention']['alpha_resweep'] is False
assert _repair_v2_sha256(metrics['repair_v2']) == metrics['calibration_v3']['provenance']['repair_v2_sha256']
protocol = v4['protocol']
known_rows = v4['stage10']['known_answer']['rows']
narration_rows = v4['stage10']['narration']['rows']
assert len(known_rows) == protocol['validation_n'] == 21
assert len(narration_rows) == 8"""
        ),
        nbformat.v4.new_code_cell(
            """read_values = np.asarray([
    row['behavior_specific_read']['equal_family_composite'] for row in known_rows
], dtype=float)
causal_values = np.asarray([row['causal_abs'] for row in known_rows], dtype=float)
all_estimable = bool(
    all(row['estimable'] for row in known_rows)
    and np.isfinite(read_values).all()
    and np.isfinite(causal_values).all()
)
rho_result = spearmanr(read_values, causal_values)
rho = float(rho_result.statistic)
p_value = float(rho_result.pvalue)

clusters = np.asarray([row['cluster'] for row in known_rows], dtype=object)
unique_clusters = np.asarray(sorted(set(clusters.tolist())), dtype=object)
cluster_indices = {
    cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters
}
rng = np.random.default_rng(protocol['known_gate']['bootstrap_seed'])
bootstrap_draws = protocol['known_gate']['bootstrap_draws']
bootstrap_rhos = []
for _ in range(bootstrap_draws):
    sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
    indices = np.concatenate([cluster_indices[cluster] for cluster in sampled])
    x = read_values[indices]
    y = causal_values[indices]
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        continue
    value = float(spearmanr(x, y).statistic)
    if np.isfinite(value):
        bootstrap_rhos.append(value)
bootstrap_rhos = np.asarray(bootstrap_rhos, dtype=float)
ci_low, ci_high = np.quantile(bootstrap_rhos, [0.025, 0.975])
known_checks = {
    'all_21_locked_rows_estimable': all_estimable,
    'spearman_rho_at_least_0_4': rho >= protocol['known_gate']['rho_min'],
    'bootstrap_ci_lower_strictly_positive': float(ci_low) > 0.0,
    'at_least_95_percent_bootstrap_draws_valid': len(bootstrap_rhos) >= 0.95 * bootstrap_draws,
}
known_status = 'PASS' if all(known_checks.values()) else 'FAIL'
known_gate = {
    'status': known_status,
    'n': len(known_rows),
    'n_source_concept_clusters': len(unique_clusters),
    'spearman_rho': rho,
    'p_value_two_sided_descriptive': p_value,
    'ci_low': float(ci_low),
    'ci_high': float(ci_high),
    'ci_level': 0.95,
    'bootstrap_method': 'source-concept-cluster percentile bootstrap',
    'bootstrap_draws_requested': bootstrap_draws,
    'bootstrap_draws_valid': len(bootstrap_rhos),
    'bootstrap_seed': protocol['known_gate']['bootstrap_seed'],
    'checks': known_checks,
    'rows': [
        {
            'name': row['name'],
            'cluster': row['cluster'],
            'causal_abs': row['causal_abs'],
            'behavior_specific_read': row['behavior_specific_read']['equal_family_composite'],
            'old_global_read': row['old_global_read']['equal_family_composite'],
            's_m_size': row['behavior_specific_read']['s_m']['n_components'],
            'estimable': row['estimable'],
        }
        for row in known_rows
    ],
}
print(json.dumps({
    'known_status': known_status,
    'N': len(known_rows),
    'clusters': len(unique_clusters),
    'rho': rho,
    'CI95': [float(ci_low), float(ci_high)],
    'bootstrap_valid': len(bootstrap_rhos),
    'checks': known_checks,
}, indent=2))"""
        ),
        nbformat.v4.new_code_cell(
            """frozen_v3 = metrics['calibration_v3']['stage1_5_alpha_sweep']['masked_weight_read']
narration_gate_rows = []
for row in narration_rows:
    ratio_record = row['behavior_specific_ratio']
    ratio = ratio_record['primary_ratio']
    finite_ratio = ratio is not None and np.isfinite(float(ratio))
    read_low = bool(finite_ratio and float(ratio) <= protocol['narration_gate']['max_read_ratio'])
    causal_low = row['causal_abs'] <= protocol['narration_gate']['requires_low_causal_abs_delta']
    joint = bool(read_low and causal_low and row['clean_capable'])
    narration_gate_rows.append({
        'key': row['key'],
        'language': row['language'],
        'frozen_v3_global_ratio': frozen_v3[row['key']]['primary_ratio'],
        'recomputed_old_global_ratio': row['old_global_primary_ratio'],
        'behavior_specific_ratio': ratio,
        'behavior_specific_status': ratio_record['status'],
        'auto_s_m_size': row['behavior_specific_auto']['s_m']['n_components'],
        'direct_s_m_size': row['behavior_specific_direct']['s_m']['n_components'],
        'causal_abs': row['causal_abs'],
        'clean_capable': row['clean_capable'],
        'read_low': read_low,
        'causal_low': causal_low,
        'joint_pass': joint,
    })
reproduced = [row for row in narration_gate_rows if row['joint_pass']]
languages = sorted({row['language'] for row in reproduced})
narration_checks = {
    'at_least_6_of_8_joint': len(reproduced) >= protocol['narration_gate']['min_passages'],
    'at_least_3_languages': len(languages) >= protocol['narration_gate']['min_languages'],
    'all_eight_causal_low': all(row['causal_low'] for row in narration_gate_rows),
    'empty_auto_paths_not_counted_as_low': all(
        not row['read_low']
        for row in narration_gate_rows
        if row['behavior_specific_status'] == 'NO_AUTO_PATH_DETECTED'
    ),
}
narration_status = 'PASS' if all(narration_checks.values()) else 'FAIL'
narration_gate = {
    'status': narration_status,
    'n': 8,
    'n_joint_pass': len(reproduced),
    'languages_joint_pass': languages,
    'n_finite_behavior_specific_ratios': sum(
        row['behavior_specific_ratio'] is not None for row in narration_gate_rows
    ),
    'checks': narration_checks,
    'rows': narration_gate_rows,
}
print(json.dumps({
    'narration_status': narration_status,
    'joint': len(reproduced),
    'languages': languages,
    'finite_behavior_specific_ratios': narration_gate['n_finite_behavior_specific_ratios'],
    'checks': narration_checks,
}, indent=2))
for row in narration_gate_rows:
    print(row)"""
        ),
        nbformat.v4.new_code_cell(
            """set_style()
fig, ax = plt.subplots(figsize=(8.2, 6.4))
ax.scatter(read_values, causal_values, color='#1565C0', s=55)
for row, x, y in zip(known_rows, read_values, causal_values, strict=True):
    ax.annotate(row['name'], (x, y), xytext=(3, 3), textcoords='offset points', fontsize=6)
ax.set_xlabel('behavior-specific path-restricted READ')
ax.set_ylabel('|CAUSAL| = |M_clean - M_edited|')
ax.set_title(
    f"F-READVAL-1 — known-answer validation\\n"
    f"Spearman rho={rho:.3f}, cluster-bootstrap 95% CI [{ci_low:.3f}, {ci_high:.3f}], N={len(known_rows)}"
)
f1_path = ROOT / 'results/figures/f_readval_1_v4.png'
save_figure(fig, f1_path)
plt.close(fig)

keys = [row['key'] for row in narration_gate_rows]
global_values = [row['recomputed_old_global_ratio'] for row in narration_gate_rows]
behavior_values = [
    0.0 if row['behavior_specific_ratio'] is None else row['behavior_specific_ratio']
    for row in narration_gate_rows
]
x = np.arange(len(keys)); width = 0.36
fig, ax = plt.subplots(figsize=(10.0, 5.8))
ax.bar(x - width / 2, global_values, width, label='old global/top-k READ', color='#6A1B9A')
bars = ax.bar(x + width / 2, behavior_values, width, label='behavior-specific READ', color='#00897B')
for bar, row in zip(bars, narration_gate_rows, strict=True):
    if row['behavior_specific_ratio'] is None:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.03,
            'NO PATH',
            ha='center',
            va='bottom',
            rotation=90,
            fontsize=7,
            color='#B71C1C',
        )
ax.axhline(0.5, color='black', linestyle='--', linewidth=1, label='low-READ bar 0.50')
ax.set_xticks(x, keys)
ax.set_ylabel('auto/direct primary READ ratio')
ax.set_title('F-READVAL-2 — narration global vs behavior-specific READ\\nNO PATH bars are not gate passes')
ax.legend(loc='upper left', fontsize=8)
f2_path = ROOT / 'results/figures/f_readval_2_v4.png'
save_figure(fig, f2_path)
plt.close(fig)
print(f1_path, f2_path)"""
        ),
        nbformat.v4.new_code_cell(
            """g_readval = 'PASS' if known_status == 'PASS' and narration_status == 'PASS' else 'FAIL'
raw_gate = {
    'schema_version': 'g-readval-v4-raw',
    'protocol_sha256': v4['protocol_sha256'],
    'known_predictivity': known_gate,
    'known_bootstrap_rhos': [float(value) for value in bootstrap_rhos],
    'narration_separation': narration_gate,
    'decision': g_readval,
}
raw_path = ROOT / 'data/raw/v4/11_readval_gate.json'
save_json(raw_path, raw_gate)
raw_bytes = raw_path.stat().st_size
raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

metrics = json.loads(metrics_path.read_text())
v4 = metrics['calibration_v4']
assert v4['protocol_sha256'] == raw_gate['protocol_sha256']
v4['stage11_readval'] = {
    'status': 'COMPLETE',
    'g_readval': g_readval,
    'known_predictivity': known_gate,
    'narration_separation': narration_gate,
    'figures': [
        str(f1_path.relative_to(ROOT)),
        str(f2_path.relative_to(ROOT)),
    ],
    'raw_artifact': str(raw_path.relative_to(ROOT)),
    'raw_artifact_bytes': raw_bytes,
    'raw_artifact_sha256': raw_sha,
    'decision_rule': 'PASS iff known predictivity AND narration separation pass',
}
v4['gate_ledger']['g_readval'] = g_readval
if g_readval == 'PASS':
    v4['gate_ledger']['stage_a_science'] = 'ALLOWED'
    v4['gate_ledger']['stage_b_report'] = 'NOT_REQUIRED'
    v4['current_allowed_conclusion'] = 'READ_VALIDATED_STAGE_A_SCIENCE_ALLOWED'
else:
    v4['gate_ledger']['stage_a_science'] = 'SKIPPED_PREREQUISITE'
    v4['gate_ledger']['stage_b_report'] = 'REQUIRED'
    v4['current_allowed_conclusion'] = 'G_READVAL_FAILED_METHODS_LIMITATION_NO_HYPOTHESIS_VERDICT'
save_json(metrics_path, metrics)

known_table = '\\n'.join(
    '| {name} | {causal:.3f} | {read:.3f} | {global_read:.3f} | {s_m} | {estimable} |'.format(
        name=row['name'],
        causal=row['causal_abs'],
        read=row['behavior_specific_read'],
        global_read=row['old_global_read'],
        s_m=row['s_m_size'],
        estimable='YES' if row['estimable'] else 'NO',
    )
    for row in known_gate['rows']
)
narration_table = '\\n'.join(
    '| {key} | {language} | {frozen:.3f} | {recomputed:.3f} | {behavior} | {auto_s} | {direct_s} | {causal:.3f} | {joint} |'.format(
        key=row['key'],
        language=row['language'],
        frozen=row['frozen_v3_global_ratio'],
        recomputed=row['recomputed_old_global_ratio'],
        behavior=('NA' if row['behavior_specific_ratio'] is None else f"{row['behavior_specific_ratio']:.3f}"),
        auto_s=row['auto_s_m_size'],
        direct_s=row['direct_s_m_size'],
        causal=row['causal_abs'],
        joint='PASS' if row['joint_pass'] else 'FAIL',
    )
    for row in narration_gate_rows
)
report_path = ROOT / 'results/RESULTS.md'
report = report_path.read_text()
marker = '\\n## Notebook 11 — G-READVAL'
if marker in report:
    report = report.split(marker, 1)[0].rstrip() + '\\n'
section = f'''\n## Notebook 11 — G-READVAL\n\n### (a) Known-answer predictivity\n\n- Status: **{known_status}**.\n- N={len(known_rows)} across {len(unique_clusters)} source-concept clusters.\n- Spearman rho={rho:.3f}; source-cluster bootstrap 95% CI [{ci_low:.3f}, {ci_high:.3f}] ({len(bootstrap_rhos)}/{bootstrap_draws} valid draws).\n- Frozen bar: rho>=0.4 and CI lower>0, with every locked row estimable.\n\n| item | |CAUSAL| | behavior-specific READ | old global READ | |S_M| | estimable |\n| --- | ---: | ---: | ---: | ---: | --- |\n{known_table}\n\n![F-READVAL-1](figures/f_readval_1_v4.png)\n\n### (b) Narration separation\n\n- Status: **{narration_status}**.\n- Joint low-READ/low-CAUSAL/clean-capable: {len(reproduced)}/8 across {len(languages)} languages.\n- Finite behavior-specific ratios: {narration_gate['n_finite_behavior_specific_ratios']}/8. Empty auto path sets are `NO_AUTO_PATH_DETECTED`, not low-READ passes.\n\n| item | language | frozen v3 global | recomputed global | behavior-specific | |S_auto| | |S_direct| | |CAUSAL| | joint |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n{narration_table}\n\n![F-READVAL-2](figures/f_readval_2_v4.png)\n\n### Decision\n\n**G-READVAL {g_readval}.** Both subgates were required. {'Stage A science is licensed.' if g_readval == 'PASS' else 'Stage A science is prohibited; the workflow must stop estimator work and take Road B.'}\n\nThe new estimator is selection-conditioned on exact path patching for the same behavior metric, but path discovery used a distinct source-only deletion rather than the causal swap endpoint. No threshold or alpha was tuned after outcomes.\n\nRaw gate artifact: `{raw_path.relative_to(ROOT)}` (SHA-256 `{raw_sha}`).\n'''
report_path.write_text(report.rstrip() + section, encoding='utf-8')
print(json.dumps({
    'G-READVAL': g_readval,
    'known': known_status,
    'narration': narration_status,
    'next': '12_science_twohop' if g_readval == 'PASS' else '12_skip_then_14_methods_report',
    'raw_sha256': raw_sha,
}, indent=2))"""
        ),
    ]
    target = ROOT / "notebooks" / "11_readval_gate.ipynb"
    nbformat.write(notebook, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", choices=("10", "11"))
    arguments = parser.parse_args()
    builders = {"10": build_notebook10, "11": build_notebook11}
    target = builders[arguments.notebook]()
    print(json.dumps({"built": str(target.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()

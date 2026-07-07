# Written vs. Read — repair-first replication

This repository tests whether a concept's causal influence is better explained
by downstream components **READING** its residual-stream direction than by the
strength with which that direction is **WRITTEN**. Version 2 is governed by a
hard calibration chain: no scientific conclusion is permitted until a known
concept swap and the positive/control gates pass.

## Current status

**G-SWAP passes; calibration is still incomplete and there is no v2 science
verdict.** The earlier
`NOT SUPPORTED` and `REFUTED` labels at commit
`6666385cff42fe4053412e7230ec9f55b0259f79` are retained only as legacy
diagnostics. Both old model scales failed the strict spider→ant top-1 swap,
the independent direction finder failed its gate, the narration positive
control reproduced 0/8, and the output-suppression metric was structurally
unable to fire. Those failures invalidate a hypothesis-level inference.

Stage 0 found that Anthropic's pinned public Jacobian Lens walkthrough loads a
model/lens and performs readout, but ships no executable causal swap or
ablation helper. The spider→ant row in `probe-swap.json` is prompt metadata,
not code. Therefore an unchanged upstream causal replication is **not
runnable**, and the release does not distinguish a local implementation bug
from a model mismatch. See [the live report](results/RESULTS.md).

Notebook 01 then selected Qwen2.5-7B layers 13–24 from clean readout
visibility, resolved exact upstream labels before leading-space alternatives,
and used the paper-literal raw direction with its documented double-strength
swap. One fixed all-position configuration produced the declared top-1
counterfactual for all three predeclared upstream cases: spider→ant (`8→6`),
buffalo→spider (`four→eight`), and oxygen→nitrogen (`8→7`). Each was repeated
three times with identical logits. This licenses G-DIR and READ validation,
not the science notebooks.

## Definitions and signs

- `WRITE = <h_l, v_c>`: projection of a clean post-block residual onto a unit
  concept direction.
- `READ`: downstream sensitivity, with weight-based READ primary after
  calibration and activation attribution secondary.
- `CAUSAL`: a measured behavior change under a validated residual intervention.

The stored signed convention is `delta = M_edited - M_clean`, where
`M = logit(target) - logit(foil)`. Positive ablation damage is explicitly
reported as `M_clean - M_edited`.

## Repair-first workflow

The required notebook chain is:

1. `00_preflight_and_stage0.ipynb` — environment, pinned upstream readout, and
   release audit.
2. `01_repair_swap.ipynb` — layers/positions/basis/token-surface/strength
   calibration and hard G-SWAP.
3. `02_concept_finder.ipynb` — independent mean-difference direction and G-DIR,
   only after G-SWAP.
4. `03_read_and_validation.ipynb` — reconcile attribution and weight READ.
5. `04_recalibration.ipynb` — firing controls and the narration G-POS gate.
6. `05_science_twohop.ipynb`, `06_science_ambiguity.ipynb`, and
   `07_scale.ipynb` — scientific tests only if every prerequisite passes.
7. `08_report.ipynb` — calibrated science report or an explicit Stage-4
   replication-failure report.

If a gate fails after an honest repair attempt, later notebooks execute only a
prerequisite guard and record `SKIPPED_PREREQUISITE`; they do not run model
inference or import legacy science values.

## Reproducing the environment

The recorded pod uses Python 3.11, PyTorch 2.5.1+cu124, Transformers 5.13.0,
and an NVIDIA H200. The dependency is pinned outside this repository:

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
export PIP_USER=false PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$HOME/.cache/huggingface/hub"

git -C "$HOME/deps/jacobian-lens" checkout \
  581d398613e5602a5af361e1c34d3a92ea82ba8e
cd "$HOME/j-space-thoughts"
.venv/bin/python -m pytest -q
```

The v2 notebooks are executed in place with the `j-space-thoughts` kernel and
a 14,400-second timeout. Model weights, fitted lenses, raw caches, and other
large intermediates remain ignored; curated metrics, executed notebooks, and
figures are committed.

## Integrity rule

Passing logit reconstruction is necessary but not sufficient. G-SWAP must
produce the declared counterfactual answer on the spider case and additional
known items, G-DIR must validate any independent direction before it is used,
and control/positive-control metrics must be capable of changing. A failed
instrument licenses a replication-failure report—not a claim that the
WRITE-versus-READ hypothesis is false.

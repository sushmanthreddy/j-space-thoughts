# Written vs. Read — surgical intervention calibration

This repository tests whether a concept's causal influence is better explained
by downstream components **READING** its residual-stream direction than by the
strength with which that direction is **WRITTEN**. Version 3 keeps the repaired
v2 instrument immutable, sweeps intervention strength and surgical position
rules, and permits no science until every calibration gate passes.

## Current status

**V3 is complete with a calibration/READ-positive-control limitation and no
hypothesis verdict.** Stage 0 and G-SWAP pass, but no frozen intervention passes
G-ALPHA, so no alpha* exists. Stage 2 and all Stage 3 science notebooks are
executed model-free prerequisite skips; P1–P3 remain untested.

The frozen source-capped carrying-position policy reaches at most 2/3 swaps.
An exploratory masked fractional swap reaches 3/3 at alpha 1.5 and passes the
random and absent-coordinate checks, but G-POS is 0/8 because every primary
weight-READ ratio exceeds the <=0.50 criterion. Its zero capability delta is a
structural no-op: all 24/24 unrelated-text masks are empty. See
[the live report](results/RESULTS.md) for the complete 24-row sweep and claim
boundary.

## Prior repair evidence (v2 context)

V2 established the working three-case swap, held-out direction retrieval,
firing controls, and the need to treat attribution READ as secondary. Its
alpha-2 all-position edit damaged unrelated capability (signed mean delta NLL
`+0.623`) and reproduced G-POS only 1/8. V3 preserves those metrics under the
immutable `repair_v2` namespace rather than rewriting them.

Stage 0 found that Anthropic's pinned public Jacobian Lens walkthrough loads a
model/lens and performs readout, but ships no executable causal swap or
ablation helper. The spider→ant row in `probe-swap.json` is prompt metadata,
not code. Therefore an unchanged upstream causal replication is **not
runnable**, and the release does not distinguish a local implementation bug
from a model mismatch.

Notebook 01 then selected Qwen2.5-7B layers 13–24 from clean readout
visibility, resolved exact upstream labels before leading-space alternatives,
and used the paper-literal raw direction with its documented double-strength
swap. One fixed all-position configuration produced the declared top-1
counterfactual for all three predeclared upstream cases: spider→ant (`8→6`),
buffalo→spider (`four→eight`), and oxygen→nitrogen (`8→7`). Each was repeated
three times with identical logits. This licenses G-DIR and READ validation,
not the science notebooks.

Notebook 02 validates the independent MD direction after a train-only
leave-one-template-out search chooses L24 and the mean residual over the last
four tokenizer tokens belonging to the clue (before the common instruction
suffix). Forty-way held-out retrieval is 44/80 (55.0%; chance 2.5%), and a
probe form selected on training cues reaches 71/80 (88.75%) exact-token top-5
on held-out cues. Mean cosine with the exact-label raw J-Lens direction is only
0.132 at L24, so the two families are validated but materially different.

Notebook 03 verifies attribution's shared-strength derivative against exact
autograd to about `1e-6`, but finds unreliable local finite-dose (`r=0.173`)
and full-alpha-1 endpoint (`r=0.062`) correlations. Attribution is therefore
secondary. Layer-aligned, random-normalized weight READ is primary: its
clear-case magnitudes are above random in 2/3 cases, while repaired
attribution/weight rank correlations are positive for MLPs (`rho=0.600`) and
attention (`rho=0.839`). Signed label orientation is retained as a diagnostic,
not mislabeled as the sign of an unsigned magnitude.

Notebook 04 freezes a 16-token teacher-forced language-mass metric before any
edited forward, aligns leading-space language-label coordinates across WRITE,
swap, direct classification, and suppression, and uses symmetric token-family
arms that each move the metric by exactly one logit unit. Those instrumentation
controls pass, as do 64-draw Gram-matched random-pair and three-case absent
nulls. However, the alpha-2 all-band edit damages unrelated-text NLL and G-POS
passes only Spanish `es1`; this is a failed calibration gate, not evidence that
the WRITE-versus-READ hypothesis is false.

## Definitions and signs

- `WRITE = <h_l, v_c>`: projection of a clean post-block residual onto a unit
  concept direction.
- `READ`: downstream sensitivity, with weight-based READ primary after
  calibration and activation attribution secondary.
- `CAUSAL`: a measured behavior change under a validated residual intervention.

The stored signed convention is `delta = M_edited - M_clean`, where
`M = logit(target) - logit(foil)`. Positive ablation damage is explicitly
reported as `M_clean - M_edited`.

## V3 calibration workflow

The required notebook chain is:

1. `00_preflight_and_reverify.ipynb` — fresh environment checks and bounded v2
   instrument re-verification.
2. `01_confirm_swap.ipynb` — repeat the three-case G-SWAP sentinel.
3. `015_alpha_sweep.ipynb` — eight strengths, three intervention policies,
   capability/G-POS/null gates, and F-ALPHA.
4. `04_recalibration.ipynb` — alpha*-specific gates, or an explicit no-alpha
   prerequisite record.
5. `05_science_twohop.ipynb`, `06_science_ambiguity.ipynb`, and
   `07_scale.ipynb` — science only after a valid alpha* and Stage-2 pass.
6. `08_report.ipynb` — calibrated science verdict or Stage-4 limitation report.

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

The v3 notebooks are executed in place with the `j-space-thoughts` kernel and
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

# Surgical intervention calibration report (v3)

## Current verdict

**V3 COMPLETE — CALIBRATION/READ-POSITIVE-CONTROL LIMITATION; NO HYPOTHESIS VERDICT.** G-SWAP passed, but no frozen alpha satisfied G-ALPHA. Stage 2 and Stage 3 were skipped by prerequisite, and P1-P3 remain untested.

## Environment

- GPU: NVIDIA H200; 143771 MiB total; 143072 MiB free at preflight.
- Home/HF-cache filesystem: 100.0 GiB total; 38.1 GiB free.
- Required tool/auth preflight: **PASS**.
- Model: `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28` in `torch.bfloat16`.

## Stage 0 — v2 instrument re-verification

- HF/J-Lens logit gate: **PASS**; max mean KL=1.660e-08, N=20.
- Known-answer alpha-2 swaps: **PASS** (3/3).
- Cached held-out G-DIR artifact: **PASS**; retrieval top-1=0.550, known-answer top-5=0.8875.
- Non-structural direct suppression controls: **PASS**.

Stage-0 decision: **PASS**. This licenses only G-SWAP confirmation
and the alpha sweep; it does not license Stage-2 recalibration or Stage-3 science.


## Stage 1 — G-SWAP confirmation

| item | concept swap | clean top-1 | edited top-1 | clean M | edited M | gate |
| --- | --- | --- | --- | ---: | ---: | --- |
| spider-legs | ` spider` -> `ant` | `8` | `6` | 6.500 | -6.500 | PASS |
| animal-legs-buffalo2 | ` buffalo` -> ` spider` | ` four` | ` eight` | 3.562 | -4.000 | PASS |
| chem-photosynthesis-Z | ` oxygen` -> ` nitrogen` | `8` | `7` | 5.375 | -5.250 | PASS |

**G-SWAP PASS (3/3).**
The next permitted step is the surgical alpha sweep. Science remains prohibited.

## Stage 1.5 — surgical alpha sweep

The carrying mask was frozen from clean source-label J-Lens rank <=10 at any
workspace layer before edited forwards. The source-capped operator was primary.
The carrying-position fractional swap is reported as an exploratory,
nonselectable sensitivity analysis because it was not frozen in notebook 00.
The all-position fractional swap is diagnostic only.

| policy | alpha | swaps | mean delta NLL | mean abs delta NLL | capability gate | G-POS | random | absent | composite |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| project_out_transfer | 0.25 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | FAIL | FAIL | FAIL |
| project_out_transfer | 0.50 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | PASS | FAIL |
| project_out_transfer | 0.75 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| project_out_transfer | 1.00 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| project_out_transfer | 1.25 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| project_out_transfer | 1.50 | 2/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| project_out_transfer | 1.75 | 2/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| project_out_transfer | 2.00 | 2/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| fractional_swap_carrying_positions | 0.25 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| fractional_swap_carrying_positions | 0.50 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| fractional_swap_carrying_positions | 0.75 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | FAIL | FAIL | FAIL |
| fractional_swap_carrying_positions | 1.00 | 0/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | FAIL | FAIL |
| fractional_swap_carrying_positions | 1.25 | 2/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | PASS | FAIL |
| fractional_swap_carrying_positions | 1.50 | 3/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | PASS | FAIL |
| fractional_swap_carrying_positions | 1.75 | 3/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | PASS | FAIL |
| fractional_swap_carrying_positions | 2.00 | 3/3 | +0.000 | 0.000 | NO_EDIT_OPPORTUNITY | 0/8 | PASS | PASS | FAIL |
| fractional_swap_all_positions_reference | 0.25 | 0/3 | -0.004 | 0.028 | PASS | 0/8 | FAIL | FAIL | FAIL |
| fractional_swap_all_positions_reference | 0.50 | 0/3 | +0.021 | 0.052 | PASS | 0/8 | FAIL | FAIL | FAIL |
| fractional_swap_all_positions_reference | 0.75 | 0/3 | +0.057 | 0.098 | PASS | 0/8 | FAIL | FAIL | FAIL |
| fractional_swap_all_positions_reference | 1.00 | 1/3 | +0.124 | 0.170 | FAIL | 1/8 | FAIL | FAIL | FAIL |
| fractional_swap_all_positions_reference | 1.25 | 2/3 | +0.230 | 0.269 | FAIL | 1/8 | PASS | PASS | FAIL |
| fractional_swap_all_positions_reference | 1.50 | 2/3 | +0.344 | 0.385 | FAIL | 1/8 | PASS | PASS | FAIL |
| fractional_swap_all_positions_reference | 1.75 | 3/3 | +0.474 | 0.515 | FAIL | 1/8 | PASS | PASS | FAIL |
| fractional_swap_all_positions_reference | 2.00 | 3/3 | +0.623 | 0.669 | FAIL | 1/8 | PASS | PASS | FAIL |

![F-ALPHA](figures/f_alpha_v3.png)

### What the sweep isolated

The strongest exploratory surgical candidate was the carrying-position
fractional swap at alpha=1.50: swaps **3/3**, random and absent nulls **PASS**,
and mean capability delta NLL=0.000. That capability number
is a conditional no-op result, not broad evidence of harmlessness:
**24/24 unrelated-text masks were
empty**, so the frozen rank rule applied no edit on every capability item.

The same alpha=1.50 candidate had small narration internal changes on all eight
items (largest absolute delta=0.215) and its direct firing
controls passed, but G-POS reproduced **0/8**. Every mask-specific primary
weight-READ ratio exceeded the required <=0.50 threshold:

| item | internal delta | weight-READ ratio | <=0.50 |
| --- | ---: | ---: | --- |
| fr1 | -0.215 | 0.849 | FAIL |
| fr2 | -0.159 | 1.000 | FAIL |
| de1 | +0.009 | 1.000 | FAIL |
| de2 | +0.025 | 1.118 | FAIL |
| es1 | +0.002 | 1.000 | FAIL |
| es2 | -0.059 | 1.000 | FAIL |
| it1 | -0.120 | 1.247 | FAIL |
| it2 | -0.032 | 1.121 | FAIL |

Subgate decomposition at this setting: clean continuation capable
**7/8**; high WRITE
**8/8**; direct source-to-English flip
**8/8**; low absolute causal change
**8/8**; low causal change relative to the
direct arm **8/8**; low primary
weight-READ **0/8**; and both
firing-control checks **8/8**.

These weight-READ ratios are properties of the fixed masks and are invariant to
alpha. Increasing or decreasing intervention strength therefore cannot make
this candidate satisfy the low-READ premise. `es2` additionally failed the
clean-continuation-capable prerequisite.

For context, the all-position reference at alpha=1.25 had signed grand mean
delta NLL=0.230, but its grand mean absolute delta
NLL=0.269 exceeded 0.25 and its per-intervention
signed means also failed (animal-legs-buffalo2=0.263, chem-photosynthesis-Z=0.074, spider-legs=0.354). It was nonselectable by protocol.

The full 24-row random and absent-control sweep is stored at
`data/raw/v3/015_alpha_sweep.json` (SHA-256 `ef53591af7e331607d7fafc43d6f3d08e2687e68197b9e925f7ee1a60e0e0cd5`).

**G-ALPHA FAIL; no tested strength/policy simultaneously achieved 3/3 swaps, capability, G-POS, and both specificity nulls.** The frozen primary policy never exceeded 2/3 swaps. The
exploratory carrying-position rescue also failed G-POS, so making it selectable
would not alter the decision. Stage 2 and Stage 3 are skipped; the workflow
takes the calibration-limitation fallback without a hypothesis verdict.

## Stage 2 — recalibration at alpha*

**SKIPPED_PREREQUISITE.** No alpha* exists, so no alpha*-specific recalibration is defined. No model forward was run in notebook
04. The following alpha*-specific checks therefore remain unmeasured:

- G-DIR at alpha*: **NOT_EVALUATED_NO_ALPHA**
- G-POS at alpha*: **NOT_EVALUATED_NO_ALPHA**
- G-SWAP at alpha*: **NOT_EVALUATED_NO_ALPHA**
- capability at alpha*: **NOT_EVALUATED_NO_ALPHA**
- weight-READ validation at alpha*: **NOT_EVALUATED_NO_ALPHA**

Stage-0 G-DIR re-verification is retained as an instrument sentinel, but it is
not relabeled as a Stage-2 result at a nonexistent alpha*.

## Stage 3 — science prerequisite records

| notebook | preregistered scope | result |
| --- | --- | --- |
| 05_science_twohop.ipynb | P1 and P2 | SKIPPED_PREREQUISITE |
| 06_science_ambiguity.ipynb | P3 | SKIPPED_PREREQUISITE |
| 07_scale.ipynb | P1 across model scale | SKIPPED_PREREQUISITE |

These notebooks are executed model-free guards. They do not import historical
science values or treat missing measurements as negative effects.

## Stage 4 — calibration-limitation result

**Classification: CALIBRATION_READ_POSITIVE_CONTROL_LIMITATION.** The working v2 intervention
was reproducible, and v3 again confirmed all three alpha-2 sentinel swaps.
However, the frozen source-capped surgical policy reached at most
**2/3** known-answer flips over the
full alpha grid, so it never met G-SWAP.

The strongest exploratory alternative, a carrying-position fractional swap at
alpha=1.50, flipped **3/3** cases and passed the random and absent-coordinate
checks. Its narration changes were small on **8/8** passages, but G-POS was
**0/8** because low primary weight-READ was **0/8**; the mask-specific ratios
were fr1=0.849, fr2=1.000, de1=1.000, de2=1.118, es1=1.000, es2=1.000, it1=1.247, it2=1.121, all above the <=0.50 criterion. One passage (`es2`) also
lacked clean continuation capability. These ratios are fixed-mask properties,
so tuning alpha cannot repair that subgate.

Capability delta NLL was exactly zero for the masked policies only because
**24/24**
unrelated-text masks were empty.
This is evidence that the detector did not fire on that fixed bank, not an
active-edit capability stress test. The all-position reference did actively
edit those texts; at alpha=2 its signed mean delta NLL was
**+0.623** and mean
absolute delta NLL was
**0.669**.

### Claim boundary

- P1, P2, and P3 are **NOT TESTED**.
- This run does **not** show that the Written-vs-Read hypothesis is false.
- It shows that the frozen intervention plus primary weight-READ positive
  control could not be jointly calibrated on open Qwen2.5-7B.
- Stage-2 independent weight-READ validation was never licensed and remains
  outstanding.

The requested legacy comparison is descriptive only: invalidated v1 reported
J-Lens `r=0.608` versus identity-J/logit-lens
`r=0.639` at `N=155`.
Those values come from commit `6666385cff42fe4053412e7230ec9f55b0259f79` and cannot be
used as evidence for P1-P3 because that instrument failed its gates.

The complete alpha sweep is in `results/metrics.json`; the full raw draw-level
artifact is `data/raw/v3/015_alpha_sweep.json` with SHA-256
`ef53591af7e331607d7fafc43d6f3d08e2687e68197b9e925f7ee1a60e0e0cd5`. F-ALPHA is the only new figure licensed by
the v3 gate chain.

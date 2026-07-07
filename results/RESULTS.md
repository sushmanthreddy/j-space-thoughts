# Road B: READ-operationalization methods limitation (v4)

## Preflight

- GPU: NVIDIA H200; 143771 MiB total, 143072 MiB free.
- Home/HF filesystem free: 37.9 GiB.
- Required tooling, Hugging Face authentication, repository remote, and model availability: **PASS**.

## G-READVAL decision

**FAIL.** Known-answer predictivity was `rho=-0.077` with source-concept-cluster bootstrap 95% CI `[-0.518, 0.409]`, N=21; narration separation was 0/8 with zero finite behavior-specific ratios.

## Stage B — methods limitation

The single permitted estimator did not validate, so the run stopped without testing P1–P3. The remainder is the required Road-B methods-limitation paper; it does not claim that Written-vs-Read is true or false.

## Abstract

On open Qwen2.5-7B we retained a working J-Lens instrument: canonical swaps reproduce 3/3, held-out concept retrieval is 0.55 top-1 / 0.8875 known-answer top-5, and the fixed masked intervention leaves all eight narration controls causally low with firing controls active. The remaining problem is READ. Attribution READ was uncorrelated with real alpha-1 effects (`r=0.062`), the inherited global weight-READ marked causally low narration concepts as strongly read (0/8 low), and the one permitted exact path-restricted estimator failed its preregistered validation (`rho=-0.077`, cluster-bootstrap 95% CI [-0.518, 0.409], N=21; narration 0/8). We therefore did not test Written-vs-Read.

## Environment and integrity

- GPU: NVIDIA H200; 143771 MiB total, 143072 MiB free.
- Home/HF filesystem free: 37.9 GiB.
- Model: `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28` in `torch.bfloat16`.
- HF/J-Lens max mean KL: 1.660e-08, N=20: **PASS** (<1e-3).
- New READ estimators: **exactly 1**. Alpha resweeps: **0**. Path thresholds tested: **[0.05]**.
- Clean-to-clean maximum exact component patch: 0.000e+00.

## What remained working

- G-SWAP: **PASS 3/3** (`8→6`, `four→eight`, `8→7`).
- G-DIR: **PASS** (top-1 0.55; known-answer top-5 0.8875).
- Narration CAUSAL-low: **8/8** under the fixed masked alpha=1.5 source-to-foil swap.
- Firing controls: **PASS**.
- Capability guardrail: masked unrelated-text rows were `NO_EDIT_OPPORTUNITY`; they are not presented as active-edit preservation.

The alpha=1.5 masked policy was exploratory/nonselectable in v3 and is used here only because v4 explicitly fixed it as the causal endpoint. No alpha was retuned.

## The single behavior-specific READ attempt

For each task, a clean-only source layer was selected by minimum source-token J-Lens rank. Path discovery used source-only unit projection deletion, distinct from the causal source-to-foil swap. Each downstream MLP output and attention head stream was patched exactly from the deleted run into an otherwise clean run. `S_M` retained every component with `|patched delta M| >= 0.05`; no top-k or fallback was allowed. The repaired v2/v3 random-normalized MLP and label-preserving OV weights were then averaged only within `S_M`, with the same 32 random directions, seeds, and `v[layer-1]→v[layer]` alignment.

Known-answer |S_M| ranged 2–153. Narration automatic |S_M| was 0–0; direct-task |S_M| was 21–204.

## Hard G-READVAL result

### Known-answer predictivity: **FAIL**

Spearman `rho=-0.077` with source-concept-cluster bootstrap 95% CI `[-0.518, 0.409]`, N=21 across 19 clusters. The frozen bar was rho>=0.4 with CI lower>0. All 21 rows were estimable, so failure is not due to post-hoc row removal.

| item | |CAUSAL| | behavior-specific READ | old global READ | |S_M| |
| --- | ---: | ---: | ---: | ---: |
| spider-legs | 7.875 | 1.250 | 1.391 | 36 |
| animal-legs-buffalo2 | 5.188 | 1.328 | 1.912 | 116 |
| chem-photosynthesis-Z | 8.000 | 1.498 | 2.104 | 58 |
| animal-nose-elephant | 14.375 | 1.237 | 1.339 | 51 |
| basketball-players | 2.625 | 0.934 | 0.675 | 56 |
| chem-organic-Z | 0.000 | 1.233 | 1.764 | 44 |
| city-state-Philadelphia | 28.031 | 1.325 | 2.534 | 145 |
| etym-saturn-position | 0.500 | 0.880 | 1.379 | 17 |
| etym-wargod-month | 0.250 | 1.682 | 1.863 | 12 |
| ex-city-capital-Lyon-Naples | 16.312 | 1.283 | 1.907 | 39 |
| ex-city-capital-Naples-Barcelona | 24.812 | 1.248 | 2.405 | 50 |
| ex-city-capital-Toronto-Lyon | 19.562 | 0.533 | 2.010 | 78 |
| ex-city-currency-Toronto-Beijing | 32.406 | 1.207 | 2.296 | 43 |
| ex-city-language-Lyon-Naples | 9.438 | 0.945 | 1.634 | 112 |
| ex-element-symbol-11-26 | 16.406 | 1.233 | 1.584 | 32 |
| ex-planet-color-third-fourth | 5.312 | 3.593 | 2.052 | 2 |
| ex2-city-language-Cairo | 26.156 | 1.806 | 2.897 | 104 |
| ex2-city-language-Moscow | 25.594 | 0.957 | 1.685 | 45 |
| ex2-language-capital-Greek | 34.938 | 1.147 | 2.758 | 48 |
| ex2-language-capital-Hungarian | 22.484 | 0.954 | 1.517 | 153 |
| ex2-language-capital-Polish | 19.469 | 1.459 | 2.226 | 116 |

![F-READVAL-1](figures/f_readval_1_v4.png)

### Narration separation: **FAIL**

All 8/8 remained causal-low, but behavior-specific auto/direct READ was finite for 0/8 and the joint gate reproduced 0/8 across zero languages. Every automatic narration `S_M` was empty at the fixed threshold; these are `NO_AUTO_PATH_DETECTED`, not low-READ successes. Direct-task path sets were nonempty. The frozen global ratios were fr1=0.849, fr2=1.000, de1=1.000, de2=1.118, es1=1.000, es2=1.000, it1=1.247, it2=1.121; all exceeded 0.50.

| item | language | frozen global | recomputed global | behavior-specific | |S_auto| | |S_direct| | |CAUSAL| |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| fr1 | French | 0.849 | 1.000 | NO PATH | 0 | 179 | 0.214 |
| fr2 | French | 1.000 | 1.000 | NO PATH | 0 | 182 | 0.167 |
| de1 | German | 1.000 | 0.876 | NO PATH | 0 | 21 | 0.000 |
| de2 | German | 1.118 | 1.118 | NO PATH | 0 | 73 | 0.018 |
| es1 | Spanish | 1.000 | 1.000 | NO PATH | 0 | 73 | 0.002 |
| es2 | Spanish | 1.000 | 1.000 | NO PATH | 0 | 93 | 0.059 |
| it1 | Italian | 1.247 | 1.247 | NO PATH | 0 | 204 | 0.131 |
| it2 | Italian | 1.121 | 1.121 | NO PATH | 0 | 198 | 0.028 |

![F-READVAL-2](figures/f_readval_2_v4.png)

The recomputed legacy top-k global value drifted for `fr1` and `de1` while the other six reproduced closely, further illustrating the instability of selection-conditioned global READ. The frozen v3 values remain the preregistered comparison.

## Decision and claim boundary

**G-READVAL FAIL. P1, P2, and P3 are NOT TESTED.** Notebooks 12–13 executed model-free prerequisite guards. This report neither supports nor refutes the Written-vs-Read hypothesis. It establishes a methods limitation: the READ side of the auditing story could not be operationalized here despite a working concept readout, canonical swap, and healthy causal/narration controls.

The new estimator is still behavior-metric selection-conditioned. The N=21 roster and all eight narration passages were previously used calibration data rather than untouched holdouts. Empty automatic path sets can arise mechanically for causally inert tasks; they were retained and disallowed from passing rather than converted to favorable zeros.

## Invalidated legacy comparison

For descriptive continuity only, invalidated v1 reported J-Lens `r=0.608` versus identity-J/logit-lens `r=0.639` at N=155. These values come from commit `6666385cff42fe4053412e7230ec9f55b0259f79` and are not evidence for P1–P3.

## What would be needed

A future test requires a READ estimator that prospectively predicts real fixed-intervention causal effects and produces finite behavior-specific narration scores under an independently validated, preferably cross-fitted path definition. This one-shot run permits no further estimator attempt.

## Reproducibility

- Notebook-10 raw artifact: `data/raw/v4/10_behavior_specific_read.json` (SHA-256 `77f129bf5f5366815e51819185f621e950e72770471b05627a605a657d06ff03`).
- Notebook-11 raw artifact: `data/raw/v4/11_readval_gate.json` (SHA-256 `abcf605278a4c9b82643e093038ecb3c06455310db34d692dd84bb9c67ab3360`).
- Protocol SHA-256: `83eaf54253f113ab8091da14cff672db0bf1efd3da762506ec6009cbb6f74050`.

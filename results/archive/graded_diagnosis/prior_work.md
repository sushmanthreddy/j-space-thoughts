# Prior graded-meter work inventory

## Search scope

This inventory was made at repository HEAD
`07ab3f6cec2e15a403bfab7cca099312c565124a` before running any new
result-dependent diagnostic. The search covered tracked root documentation,
`results/`, `paper/`, `experiments/`, notebooks, source files, and Git commit
messages for terms including `graded`, `range restriction`, `quantity
mismatch`, `normalization`, `directional`, `single-layer`, `power`, and the
reported engine-only correlation.

## Branch decision

**Partial prior work exists, but no dedicated root-cause diagnosis exists.**

The repository already contains a careful graded-*outcome* stress test. It
establishes that the current score does not rank causal magnitude and offers
range restriction as a possible limitation. It does not perform the H1-H7
diagnostics requested here. This audit will therefore review and reproduce the
prior numerical conclusions while filling the missing diagnostic analyses
after the evidence tests are frozen at Gate 1.

## Prior work being reviewed

### 1. Isolated v6 graded stress test

[`experiments/05_prior_versions/v6_isolated/results/RESULTS_v6.md`](../../../experiments/05_prior_versions/v6_isolated/results/RESULTS_v6.md)
is the closest prior analysis. It reports:

- 77 engines in 24 dependency groups;
- engine-only READ_IG Spearman rho `-0.179` with grouped interval
  `[-0.431, 0.126]`;
- a narrow engine absolute-C range of approximately `0.786` to `1.012`;
- no post-hoc weak/strong split because the frozen protocol supplied no such
  cutoff; and
- survival of binary separation under an answer-type-matched control.

Its supported conclusion is appropriately limited: binary relevant-versus-idle
discrimination survives, while positive graded resolution is not demonstrated.
It does not claim to know why the graded association fails.

### 2. Final integrated report and paper

[`results/RESULTS.md`](../RESULTS.md), the root `README.md`, and
[`paper/workshop_paper.md`](../paper/workshop_paper.md) reproduce the same
engine-only result and correctly reject the pooled engine/control correlation
as graded evidence. The paper says range restriction *may* hide a relation and
recommends a preregistered weak/medium/strong follow-up. It does not present
range restriction as proven causation.

### 3. Selection audit

[`results/selection_audit/VERDICT.md`](../selection_audit/VERDICT.md) rules out
one proposed selection mechanism within the visible population: all 77
L16-WRITTEN evaluation candidates pass every non-WRITTEN verification
condition. It leaves open whether the WRITTEN threshold itself excludes
weak-but-real causal concepts. That distinction is directly relevant to H1 and
H2 but is not itself a graded-meter diagnosis.

### 4. Earlier failed instruments and READ definitions

The research history in [`experiments/README.md`](../../../experiments/README.md)
and the paper records broken interventions, broad damaging edits, near-zero
earlier attribution correlations, empty paths, and a backward intermediate
classifier. These failures explain why the final instrument was narrowed and
firewalled. They do not identify the cause of the final READ_IG engine-only
correlation.

## What prior work did not test

No tracked report or commit was found that jointly provides the following for
the final 77-engine roster:

- variance and rank-resolution diagnostics for absolute C;
- a group-aware power or simulation analysis;
- correlations of READ_IG with concept amount, path norm, clean margin,
  normalization T, endpoint change, or IG completeness error;
- signed directional I_A/I_B agreement and cancellation analysis;
- relation-family correlations and leave-one-group-out stability;
- raw versus normalized causal-effect compression or high-margin saturation;
- a quantitative audit of the 16-step midpoint approximation; or
- evidence comparing L16/single-token READ against a preregistered multi-layer
  or multi-token alternative.

Accordingly, prior statements about range restriction and estimator scope are
plausible hypotheses, not completed causal diagnoses. Gate 1 will freeze the
evidence that would support or count against each explanation before these
missing analyses are run.

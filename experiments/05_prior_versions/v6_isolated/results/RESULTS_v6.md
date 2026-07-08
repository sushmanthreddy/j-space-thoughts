# v6 isolated stress test — ARTIFACT (partial)

## One-line decision

**ARTIFACT (partial): READ_IG survives the answer-type-matched control, but has no positive graded association within engines (rho=-0.179, 95% CI [-0.431, 0.126]); the perfect binary separation is not evidence of a graded causal-use meter.**

## Isolation and frozen protocol

This stress test starts from frozen commit `eb9e44144de7d05d4a8e93f975d1af1351b0d87d` and treats every
pre-existing source file, notebook, and result as read-only. All additions are
confined to `src/*_v6.py`, `scripts/*_v6.py`, `notebooks/v6_*.ipynb`, and
`results/v6/**`. The isolation audit found **no modified pre-v6 path**.

The causal truth, source layer L16, explicit-concept position, 16-step READ_IG,
READ_local, five dependency-group folds, and 10,000-draw seed-1729 group
bootstrap were reused unchanged. No estimator, layer, direction, fold, or score
transformation was retuned.

## CHECK 1 — graded signal within engines (decisive)

Only the 77 frozen verified engines were retained, spanning 24 dependency
groups. Correlations are against frozen signed-interchange `|C|`.

| estimator | engine-only Spearman rho | group-bootstrap 95% CI | interpretation |
| --- | ---: | --- | --- |
| READ_IG | -0.179 | [-0.431, 0.126] | CI spans zero |
| READ_local | -0.159 | [-0.450, 0.195] | CI spans zero |
| capacity baseline | -0.252 | [-0.492, 0.051] | CI spans zero |

READ_IG does not retain a positive graded-use signal inside engines; its point
estimate is negative. Engine `|C|` already lies in the narrow strong-causal
range `0.786` to
`1.012`. No within-engine AUC was run:
the frozen protocol has no weak/strong cutoff and the largest adjacent gap
isolates one row, so adding a cutoff now would be post-hoc.

![F_v6_1](figures/F_v6_1_engine_only_read_vs_c.png)

## CHECK 2 — answer-type-matched hard dashboard

The actual frozen engines never answer with numbers: they output chemical
symbols or capital-city names. A numeric control would therefore preserve the
answer-type mismatch. The hard controls instead use fixed calibration-only
anchors with the same relation and semantic answer class: platinum→Pt,
Netherlands→Amsterdam, and Alabama→Montgomery. The original natural context and
explicit concept token remain byte-for-byte unchanged, while the source concept
cannot determine the fixed anchor answer.

- VERIFIED_HARD: **77**;
  UNVERIFIED_HARD: **0**.
- Frozen engine median `|C|`: `0.9127`.
- Hard-dashboard median `|C|`: `0.0065`;
  sharp directional disagreements: `0`.
- Hard-control causal sanity: **PASS**.

| comparison | held-out READ_IG AUC | group-bootstrap 95% CI |
| --- | ---: | --- |
| engine vs old dashboard | 1.000 | [1.000, 1.000] |
| engine vs hard dashboard | 1.000 | [1.000, 1.000] |

The harder separation survives in all five frozen folds. Thus arithmetic answer
type is **not** the sole cause of the binary separation. This does not overturn
CHECK 1: surviving a relevant-vs-irrelevant classification does not establish a
graded causal-use meter.

![F_v6_2](figures/F_v6_2_old_vs_hard_dashboard_auc.png)

## CHECK 3 — raw READ_IG distributions

| class | minimum | median | maximum | IQR |
| --- | ---: | ---: | ---: | ---: |
| engine | 0.0350 | 0.2102 | 0.9928 | 0.1133 |
| old dashboard | 0.0009 | 0.0056 | 0.0234 | 0.0045 |
| hard dashboard | 0.0017 | 0.0063 | 0.0210 | 0.0068 |

The old dashboard scores are not identical, but they occupy a compressed low
band. The answer-type-matched hard dashboards occupy essentially the same band:
their observed ranges overlap across
`86.0%`
of the union. Both dashboard ranges are strictly disjoint from engines, with
gaps `0.0116` (old) and
`0.0139` (hard).

This pattern rules out a specifically arithmetic-gradient explanation, but it
supports the cautionary mechanism: on this roster READ_IG behaves like a binary
relevant-vs-irrelevant detector and offers no demonstrated graded resolution
among already-strong engines.

![F_v6_3](figures/F_v6_3_read_ig_distributions.png)

## Interpretation and scope

The original perfect class separation was inflated as evidence for a *graded*
USE score. A narrower claim survives: READ_IG robustly separates causally
relevant engines from two causally irrelevant control families, including a
semantic answer-type-matched control. It does not rank causal magnitude within
the 77 engines. The evidence is limited to Qwen2.5-7B, L16 explicit written
concepts, three frozen relation families, and an engine set whose causal effects
are all already strong.

## Firewall, reproducibility, and audit

- Cheap READ consumed only sanitized clean manifests and imported the unchanged
  `src/cheap_read.py`; it never read hard C, edited metrics, or interchange
  outputs. Firewall audit: **PASS**.
- Hard C was computed separately with the unchanged causal module and matched
  frozen engine T. C remained signed and unclipped.
- Required figures F_v6_1–F_v6_3 and all recorded raw-artifact hashes pass.
- Existing files modified: **none**. Isolation audit: **PASS**.
- The test suite, pytest, and Ruff were not run, as required.

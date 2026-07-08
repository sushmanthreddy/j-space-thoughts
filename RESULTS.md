# Results: compact numeric reference

Anthropic's J-Lens and J-Space supply the concept-readout foundation used in
this experiment. The contribution evaluated here is the donor-free `READ_IG`
detector, its matched controls, and its firewalled causal validation. The final
classification is: binary screening supported in the frozen setting; graded
measurement not supported for the current estimator/target pairing.

## Setup

| Item | Frozen value |
| --- | --- |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Dtype | bf16 |
| Measurement site | L16, explicit single concept token |
| Candidate split | 118 candidates; 25 calibration; 93 held-out evaluation |
| Final evaluation roster | 77 verified pairs in 24 dependency groups; 16 UNVERIFIED |
| Inference | five whole-group folds; 10,000 dependency-group bootstrap draws |
| Seed | 1729 |

The token position is the explicit concept token in each byte-identical shared
fact context; it is not one fixed integer index across prompts. ROC labels are
the constructed engine and idle tasks, validated independently by causal
interchange rather than created by thresholding `C`.

Full setup and pre-consolidation reporting:
[archived results](results/archive/RESULTS.md) and
[archived workshop draft](results/archive/paper/workshop_paper.md).

## Binary detector

The symmetric, signed, unclipped full-residual score `C` validates that the
matched engines are causally active and both dashboard families are idle.

| Class | N | Median absolute `C` | Grouped 95% CI |
| --- | ---: | ---: | --- |
| Relevant engine | 77 | 0.912714 | [0.896378, 0.929191] |
| Original idle dashboard | 77 | 0.005083 | [0.003587, 0.007752] |
| Answer-type-matched idle dashboard | 77 | 0.006466 | [0.004013, 0.010064] |

| Estimator and comparison | Held-out ROC AUC | Grouped 95% CI |
| --- | ---: | --- |
| `READ_IG`, engine vs original idle | 1.000000 | [1.000000, 1.000000] |
| `READ_IG`, engine vs answer-matched idle | 1.000000 | [1.000000, 1.000000] |
| `READ_local`, engine vs original idle | 0.914825 | [0.863661, 0.967161] |
| Static capacity control, engine vs original idle | 0.500000 | [0.500000, 0.500000] |

![Held-out binary AUC and baselines](results/figures/f2_binary_auc_and_baseline.png)

The answer-type-matched result rules out arithmetic answer type as the sole
source of separation. The static-capacity result also argues against a purely
behavior-independent high-gain explanation. These are ranking results on the
frozen roster; no deployment threshold or universal accuracy claim follows.

Full detail: [archived final results](results/archive/RESULTS.md).

## Graded result: not established

The decisive graded statistic is the association within the already-relevant
engine class.

| Quantity | Result |
| --- | --- |
| Engine-only `READ_IG` vs `|C|` | Spearman `rho=-0.179110`; grouped 95% CI `[-0.431377, 0.126014]` |
| Engine `|C|` range | `0.785789` to `1.012025` |
| Pooled engine/original-idle association | Spearman `rho=0.707412` |

The pooled value is **not graded evidence**: it is dominated by separation
between engines and original dashboards. It has no reported confidence interval
in the populated source artifacts, and it does not show that READ orders causal
magnitude among engines.

The diagnosis distinguishes what READ follows from what normalized
full-residual `C` measures:

| Association | Spearman rho | Grouped 95% interval |
| --- | ---: | --- |
| `READ_IG` vs mean absolute path endpoint change | 0.924200 | [0.821432, 0.961500] |
| Path endpoint vs absolute `C` | -0.144271 | [-0.405460, 0.140508] |
| `READ_IG` vs mean absolute concept amount | 0.649088 | [0.320006, 0.846221] |
| `READ_IG` vs mean delta norm | 0.629581 | [0.242093, 0.857630] |

This supports a measurement-design mismatch: READ ranks its
direction-defined, concept-scaled path, while `C` is normalized recovery after
replacing the complete residual state. No audited formula, row join, side,
answer orientation, layer, token, or padding defect was found.

Numerical completeness is also inadequate. The per-row relative
integrated-gradient completeness error has median `0.230672` and p95
`0.991759`; `0/76` nonzero-endpoint rows are at or below 1%, and `4/76` are at
or below 5%. The saved evidence does not isolate quadrature error from bf16
path/endpoint quantization.

The prospective power result is a conservative approximation, not an exact
simulation of the unequal 77-row/24-cluster design. Treating 24 dependency
groups as independent units gives power `0.48575` at true `rho=0.40` and an
80%-power MDE of `0.550`. The optimistic 77-independent-row scenario gives
power `0.95280` and MDE `0.325`. Limited prospective power constrains the test,
but it does not overturn the negative point estimate or realized grouped
interval.

Full detail:
[analysis](results/archive/graded_diagnosis/analysis.md),
[evidence ranking](results/archive/graded_diagnosis/ranking.md), and
[NO-GO recommendation](results/archive/graded_diagnosis/RECOMMENDATION.md).

## Selection audit

| Funnel stage | Count | Interpretation |
| --- | ---: | --- |
| Held-out evaluation candidates | 93 | Frozen pre-gate pool |
| Pass both engine clean-answer checks | 93 | No clean-answer removals |
| Pass WRITTEN in both reciprocal directions | 77 | 16 fail only WRITTEN |
| Visible rows passing both dashboard clean-answer checks | 77/77 | All visible rows pass |
| Visible rows rejected by any non-WRITTEN gate | 0 | No later-gate range restriction among visible concepts |

The verdict is **Finding A within the experiment's declared population of
L16-visible concepts**. Because every visible candidate passed the other
conditions, non-WRITTEN gates did not create the retained strong-only range.
The causal magnitude of the 16 not-WRITTEN candidates was not measured, so the
effect of the WRITTEN threshold itself remains unresolved.

Full detail:
[gate flow](results/archive/selection_audit/gate_flow.md),
[distribution comparison](results/archive/selection_audit/distribution_compare.md),
and [verdict](results/archive/selection_audit/VERDICT.md).

## Classification

| Claim | Classification | Scope |
| --- | --- | --- |
| Binary relevant-versus-idle detector | **Supported** | Frozen model, L16, explicit single-token concepts, matched engine/dashboard roster |
| Graded `READ_IG` / normalized full-residual `|C|` meter | **Principled NO-GO for the current pairing** | No positive within-engine ordering; no audited local code fix identified |
| Whether WRITTEN excludes a weak-but-real causal tail | **Open** | The 16 excluded candidates have no `C` measurement |

The conclusion should be stated as follows:

> In this frozen experiment, READ_IG is supported as a relevant-versus-idle
> screening score, but it is not supported as a graded estimator of normalized
> full-residual causal recovery within relevant examples.

Classification sources:
[selection verdict](results/archive/selection_audit/VERDICT.md) and
[graded recommendation](results/archive/graded_diagnosis/RECOMMENDATION.md).

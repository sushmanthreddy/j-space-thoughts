# WRITE / READ / CAUSAL results

## Overall verdict

**NOT SUPPORTED** — P3 was diagnostically refuted, P1 is unsupported (including a significantly negative main-scale raw READ partial), and P2 narration remains unestablished.

## Preregistered hypothesis

A concept's causal influence on behavior is governed by whether behavior-relevant downstream circuits READ its residual-stream direction, not merely by how strongly that direction is WRITTEN into the residual stream. The preregistered residual prediction is that CAUSAL tracks READ conditional on WRITE, while WRITE contributes approximately zero once READ is controlled. This pattern must survive both raw J-Lens and independent mean-difference directions and both attribution- and weight-based READ.

## Correctness gates and usable-workspace context

| model | G1 | G1 N | max mean KL | G2 strict | G2 directional | G3 computed | attribution reliable | attribution vs real | context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | PASS | 20 | 1.660e-08 | FAIL | PASS | PASS | NO | -0.360 (95% CI -0.736, 0.330; N=20) | diagnostic because strict workspace gate failed |
| 14B | PASS | 20 | 4.320e-08 | FAIL | PASS | PASS | YES | 0.514 (95% CI 0.386, 0.629; N=226) | diagnostic |

G3 is a held-out validation gate (N=20 at 7B); the scale table separately reports full-core attribution correlations. The directional G2 PASS is the RMS-gain-folded sensitivity variant. The raw direction used downstream did not move either scale's known-case answer top-1 to 6, so strict G2 failed.

## P1 — conditional READ versus WRITE

For these regressions, CAUSAL is positive ablation damage `M_clean - M_edited`; the repository-wide intervention delta stored per item is the opposite sign, `M_edited - M_clean`.

Attribution READ:

| model | direction | N | corr(WRITE, READ) | partial CAUSAL–READ \| WRITE | partial CAUSAL–WRITE \| READ | β READ (95% CI) | β WRITE (95% CI) | β WRITE×READ (95% CI) | R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | jlens_raw_wu_j | 226 | 0.116 (95% CI 0.001, 0.238; N=226) | -0.165 (95% CI -0.291, -0.037; N=226) | -0.144 (95% CI -0.253, -0.030; N=226) | -0.164 (-0.291, -0.034) | -0.142 (-0.250, -0.025) | -0.077 (-0.203, 0.065) | 0.052 |
| 14B | mean_difference | 16 | -0.119 (95% CI -0.708, 0.411; N=16) | 0.137 (95% CI -0.358, 0.574; N=16) | 0.181 (95% CI -0.337, 0.703; N=16) | 0.136 (-0.367, 0.603) | 0.181 (-0.331, 0.767) | -0.062 (-0.688, 0.744) | 0.046 |
| 7B | jlens_raw_wu_j | 155 | 0.132 (95% CI -0.029, 0.302; N=155) | 0.430 (95% CI 0.265, 0.565; N=155) | -0.107 (95% CI -0.250, 0.040; N=155) | 0.433 (0.268, 0.566) | -0.098 (-0.230, 0.034) | -0.131 (-0.313, -0.001) | 0.186 |
| 7B | mean_difference | 20 | 0.221 (95% CI -0.163, 0.599; N=20) | 0.010 (95% CI -0.499, 0.460; N=20) | -0.430 (95% CI -0.737, 0.118; N=20) | 0.009 (-0.504, 0.422) | -0.439 (-0.739, 0.146) | -0.135 (-0.846, 0.423) | 0.191 |

Weight READ (activation-independent after direction choice, but localization-selection-conditioned):

| model | direction | weight family | N | corr(WRITE, weight READ) | CAUSAL–READ \| WRITE | CAUSAL–WRITE \| READ | β weight READ (95% CI) | β WRITE (95% CI) | β WRITE×READ (95% CI) | R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | jlens_raw_wu_j | mlp | 155 | -0.149 (95% CI -0.371, 0.087; N=155) | -0.304 (95% CI -0.426, -0.169; N=155) | -0.090 (95% CI -0.226, 0.047; N=155) | -0.308 (-0.433, -0.165) | -0.087 (-0.222, 0.048) | -0.031 (-0.141, 0.036) | 0.094 |
| 7B | jlens_raw_wu_j | attention | 155 | -0.011 (95% CI -0.185, 0.161; N=155) | 0.014 (95% CI -0.139, 0.197; N=155) | -0.041 (95% CI -0.194, 0.110; N=155) | 0.014 (-0.138, 0.195) | -0.041 (-0.190, 0.108) | -0.009 (-0.180, 0.265) | 0.002 |
| 7B | mean_difference | mlp | 20 | -0.118 (95% CI -0.459, 0.184; N=20) | 0.035 (95% CI -0.467, 0.527; N=20) | -0.432 (95% CI -0.750, 0.046; N=20) | 0.032 (-0.427, 0.471) | -0.433 (-0.755, 0.043) | 0.246 (-0.589, 0.747) | 0.192 |
| 7B | mean_difference | attention | 20 | 0.009 (95% CI -0.527, 0.550; N=20) | -0.472 (95% CI -0.720, -0.186; N=20) | -0.479 (95% CI -0.744, -0.047; N=20) | -0.425 (-0.714, -0.185) | -0.433 (-0.689, -0.066) | -0.062 (-0.501, 0.405) | 0.371 |

**P1 verdict: UNSUPPORTED**. A model is labelled supported only when strict G1+G2 passes, raw and MD attribution plus MLP/attention weight READ are all available, every READ partial-correlation 95% CI lies above zero, and every WRITE partial-correlation 95% CI includes zero. This CI-sign rule does not supply the missing preregistered numeric definition of 'large'.
At the main 14B scale, the raw-direction READ partial is significantly negative, so this is contrary evidence rather than merely missing robustness.

### Direction and READ-estimator validation

| check | result |
| --- | --- |
| 7B MD validation | FAIL; raw/MD cosine=0.043 (95% CI 0.038, 0.048; N=40); held-out top-1=0.062 (95% CI 0.013, 0.113; N=80); explicit top-5=0.762 (95% CI 0.662, 0.850; N=80) |
| 14B MD validation | FAIL; failed criteria=explicit_top5_at_least_0.80, retrieval_top1_ci_above_chance |
| selected-head attribution/weight rank agreement | label-weighted OV ρ=0.724; OV norm ρ=0.850 |
| selected-MLP attribution/weight rank agreement | normalized gain ρ=-0.619 |

Agreement is mixed and selection-conditioned: attention ranks agree, while selected-MLP ranks are negatively associated.

## P2 — posthoc high-WRITE/low-READ quantile screen

Descriptive selection only: WRITE ≥ Q75 (26.577), READ ≤ Q25 (0.014), and |CAUSAL| ≤ Q25 (1.656).
These rows pass only the WRITE/READ/CAUSAL quantile screen. A full narration case must additionally have ablation approximately equal to suppression; 0 screened candidates meet that additional numerical screen.

| item | WRITE | READ | \|CAUSAL\| | \|suppression\| | role |
| --- | --- | --- | --- | --- | --- |
| ex2-language-capital-Greek | 33.494 | 0.013 | 1.656 | 0.000 | posthoc_quantile_screen_candidate |
| ex2-language-capital-Hungarian | 29.746 | 0.012 | 1.062 | 0.000 | posthoc_quantile_screen_candidate |
| world-cebu-country-capital | 28.289 | 0.013 | 1.203 | 0.000 | posthoc_quantile_screen_candidate |

**P2 verdict: UNESTABLISHED**. Narration remains unestablished: no candidate and control set met the full operational definition.
Known-narration control: FAIL; absent-null control: DESCRIPTIVE_NO_EQUIVALENCE_THRESHOLD.

## P3 — ambiguity commitment

| quantity | estimate |
| --- | --- |
| committed concept WRITE | 8.609 (95% CI 8.082, 9.154; N=120) |
| alternate concept WRITE | 8.599 (95% CI 8.044, 9.109; N=120) |
| committed concept attribution READ | 0.013 (95% CI 0.013, 0.014; N=120) |
| alternate concept attribution READ | 0.014 (95% CI 0.013, 0.014; N=120) |
| mean-margin swap flip rate | 0.033 (95% CI 0.013, 0.083; N=120) |
| variant 1 flip rate | 0.092 (95% CI 0.052, 0.157; N=120) |
| variant 2 flip rate | 0.075 (95% CI 0.040, 0.136; N=120) |
| both variants flip | 0.000 (95% CI 0.000, 0.031; N=120) |
| internal ablation damage | 1.151 (95% CI 0.949, 1.358; N=120) |
| internal minus suppression | 1.151 (95% CI 0.948, 1.353; N=120) |
| output suppression damage | 0.000 (95% CI 0.000, 0.000; N=120) |

**P3 diagnostic verdict: REFUTED**. G2 context: strict=FAIL, directional=PASS.
Committed, alternate, and meta-token directions use attribution READ only; weight READ and independent mean-difference direction robustness were not run for this diagnostic phase.

Caveat: The ambiguity behavior metric is a reading-answer logit difference, while output suppression clamps a separate concept-token logit. Its exact zero is therefore structural under this metric, so ablation > suppression is not an independent output-steering test.

Interpretive meta-token diagnostics (nonconfirmatory):

| association | estimate |
| --- | --- |
| candidate-mean READ vs damage | 0.783 (95% CI 0.477, 0.970; N=9) |
| candidate-mean partial READ \| WRITE | 0.671 (95% CI 0.172, 0.967; N=9) |
| pooled item×candidate READ vs damage | 0.253 (95% CI 0.174, 0.334; N=1080) |
| pooled item×candidate partial READ \| WRITE | 0.155 (95% CI 0.072, 0.241; N=1080) |

Observation-level bootstrap is descriptive and does not model within-item or within-candidate dependence.
Meta-token output suppression coverage: 1080/1080; all exact structural zeros=True.

## P4 — optional blackmail task

**NOT_RUN_OPTIONAL** — The optional blackmail phase was not run; this does not fail report completeness.

## Mandatory controls

| control | result |
| --- | --- |
| random-direction null | mean \|Δ\|=0.428 (95% CI 0.408, 0.450; N=155); observed−random \|Δ\|=2.961 (95% CI 2.642, 3.293; N=155) |
| absent-coordinate null | DESCRIPTIVE_NO_EQUIVALENCE_THRESHOLD; mean \|Δ\|=0.569 (95% CI 0.487, 0.656; N=153) |
| capability ΔNLL | 0.066 (95% CI 0.041, 0.092; N=155); 1240 text×intervention rows |
| two-hop accuracy clean → edited | 0.968 (95% CI 0.955, 0.981; N=155) → 0.952 (95% CI 0.935, 0.966; N=155); 620 off-target evaluations |
| known narration | FAIL; reproduced 0/8; high-WRITE 8/8; low-causal 1/8; clean-capable 5/8; mean WRITE=6.512 (95% CI 5.479, 7.674; N=8); mean READ=0.008 (95% CI 0.007, 0.009; N=8) |
| identity-J baseline | shared outcome: identity-J=0.639 (95% CI 0.552, 0.719; N=155); J-Lens=0.608 (95% CI 0.517, 0.693; N=155) |
| output-suppression completeness | PASS; 155 / 155 exact structural zeros; instrumentation only |

Random, absent-coordinate, capability, narration, and identity-J controls were run for the 7B two-hop phase; they were not repeated for 14B. Ambiguity has its own structural concept/meta-token suppression records.

## Scale comparison

Scale phase status: **COMPUTED**. Qwen-32B: **SKIPPED_DISK_CONSTRAINT**.

| scale | direction | N | strict usable | CAUSAL–READ \| WRITE | CAUSAL–WRITE \| READ | mean ablation damage | attribution r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | jlens_raw_wu_j | 226 | False | -0.165 (95% CI -0.291, -0.037; N=226) | -0.144 (95% CI -0.253, -0.030; N=226) | 0.224 (95% CI -0.270, 0.719; N=226) | 0.514 (95% CI 0.386, 0.629; N=226) |
| 14B | mean_difference | 16 | False | 0.137 (95% CI -0.358, 0.574; N=16) | 0.181 (95% CI -0.337, 0.703; N=16) | 1.262 (95% CI 0.154, 2.338; N=16) | 0.805 (95% CI 0.635, 0.925; N=16) |
| 7B | jlens_raw_wu_j | 155 | False | 0.430 (95% CI 0.265, 0.565; N=155) | -0.107 (95% CI -0.250, 0.040; N=155) | 3.099 (95% CI 2.698, 3.498; N=155) | 0.608 (95% CI 0.514, 0.695; N=155) |
| 7B | mean_difference | 20 | False | 0.010 (95% CI -0.499, 0.460; N=20) | -0.430 (95% CI -0.737, 0.118; N=20) | 1.302 (95% CI 0.148, 2.394; N=20) | 0.552 (95% CI 0.045, 0.834; N=20) |

Paired 14B − 7B differences on common frozen items:

| direction | N common | Δ READ partial | Δ WRITE partial | Δ mean damage | Δ attribution r |
| --- | --- | --- | --- | --- | --- |
| jlens_raw_wu_j | 135 | -0.611 (95% CI -0.821, -0.362; N=135) | 0.008 (95% CI -0.210, 0.228; N=135) | -3.417 (95% CI -4.060, -2.752; N=135) | -0.105 (95% CI -0.273, 0.037; N=135) |
| mean_difference | 14 | 0.010 (95% CI -0.547, 0.833; N=14) | 0.291 (95% CI -0.427, 1.025; N=14) | -0.100 (95% CI -1.877, 1.654; N=14) | 0.170 (95% CI -0.265, 0.950; N=14) |

Negative Δ READ partial means the conditional READ association weakened at 14B; the raw-direction CI excludes zero, so the preregistered pattern reversed rather than sharpened.

32B was not downloaded: published weight sizes project all three models to 102.8 GiB on the measured 100 GiB quota, before lens, checkpoint, activation, or temporary-file headroom.

## Figures

- [F1 — 7B CAUSAL versus READ/WRITE](figures/f1_twohop_qwen2.5-7b.png)
- [F1 — 14B CAUSAL versus READ/WRITE](figures/f1_twohop_qwen2.5-14b.png)
- [F2 — 7B conditional coefficients](figures/f2_twohop_qwen2.5-7b.png)
- [F2 — 14B conditional coefficients](figures/f2_twohop_qwen2.5-14b.png)
- [F3 — internal ablation versus output suppression](figures/f3_internal_vs_output_suppression.png)
- [F4 — READ localization](figures/f4_read_localization_qwen2.5-7b.png)
- [F5 — attribution versus real ablation](figures/f5_attribution_vs_ablation_qwen7b.png)
- [F6 — direction and weight-READ robustness](figures/f6_direction_robustness_qwen2.5-7b.png)
- [F6 — 14B direction robustness](figures/f6_direction_robustness_qwen2.5-14b.png)
- [F7 — scale comparison](figures/f7_scale_comparison.png)
- [F8 — ambiguity and meta-token diagnostics](figures/f8_ambiguity_write_read.png)

## Limitations

- Flash/SDPA attention kernels were not proven bitwise deterministic; seeded reruns can retain low-level nondeterminism despite fixed seeds.
- The 14B lens resumed from a legacy first-10-prompt checkpoint that predated the prompt-hash provenance sidecar; those first ten contributions cannot be cryptographically bound to the declared prompt list.
- The preregistration supplied no numerical cutoff for 'large' READ, 'approximately zero' WRITE, or scale sharpening; estimates and 95% CIs are primary, and the report's CI-sign rule is conservative bookkeeping.
- Weight READ is activation-independent only after direction choice and is selection-conditioned because components were first flagged by activation localization.
- Population weight READ was run for the 7B two-hop analysis, not for the 14B scale run or the ambiguity committed/alternate/meta-token directions; those analyses therefore do not satisfy a two-READ estimator claim.
- F4 contrasts measured driver and low-READ candidates, not a validated driver-versus-narration class, because the known-narration positive control did not reproduce.
- Concepts are restricted to exact single-token proxies; multi-token concepts are outside the fitted vocabulary-direction analysis.
- Random-direction, absent-coordinate, general capability, known-narration, and identity-J controls were run for 7B two-hop only, not repeated at 14B.
- The known-answer directional G2 sensitivity pass used RMS-gain-folded directions; the raw direction used downstream did not change top-1 to the swapped answer at either scale.
- The ambiguity behavior metric is a reading-answer logit difference, while output suppression clamps a separate concept-token logit. Its exact zero is therefore structural under this metric, so ablation > suppression is not an independent output-steering test.
- Qwen-7B failed the strict usable-workspace context; its downstream results are diagnostic.
- Qwen-14B failed the strict usable-workspace context; its downstream results are diagnostic.
- Independent MD direction validation status was FAIL; MD robustness is correspondingly limited.
- P1 did not survive every required direction/READ robustness check.
- P2 candidates are posthoc and do not establish a narration class.
- The absent-coordinate null has no preregistered equivalence margin or did not pass; near-zero specificity is not established.
- Final concept-token suppression is structurally zero for this target-minus-foil metric and checks instrumentation/direct-logit steering; it is not additional causal evidence.
- Random directions are norm- and layer-count matched, not geometry matched.
- The absent-coordinate null is conditional on a fixed rank threshold and candidate list.
- The retained concept-to-absent intervention removes the active concept coordinate and is a stress test, not a null.
- Capability NLL uses a small authored fixed text set, not a benchmark corpus.
- Known narration tests a preregistered first-token margin, not free-form generation quality.
- Within-direction identity-J and J-Lens validity checks use their own intervention directions; the headline baseline table instead joins both predictors to the same core-ablation outcome.

## Verdicts

| prediction | verdict |
| --- | --- |
| P1 | UNSUPPORTED |
| P2 | UNESTABLISHED |
| P3 | REFUTED |
| P4 | NOT_RUN_OPTIONAL |

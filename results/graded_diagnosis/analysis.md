# Graded-meter diagnosis: read-only analysis

## Provenance and invariants

No model, lens, causal intervention, or READ computation was rerun. This report
joins the 77 frozen engine rows in:

- `artifacts/final/02_causal.json`, SHA-256
  `9e5ad02e6ce2133e66d733e1f644fe14921254da2a0095b8a2603af890bc5402`;
- `artifacts/final/03_cheap.json`, SHA-256
  `da633369469bd6a2ca5ec359b89605207934a3f5c5486bfb77f32eef8a21cac5`;
  and
- `artifacts/final/01_clean_manifest.json` for frozen positions and lengths.

The artifacts have the same 77 unique pair IDs in the same order, covering 24
dependency groups. Category, concepts, group, fold, protocol hash, and model
metadata agree for every joined row. All nested records have status `OK`.

The implementation invariants reproduce:

- stored `READ_IG = mean(|I_A|, |I_B|)` within `2.98e-8`;
- `T=M_A-M_B`, both directional recoveries, and `C=N/T` within
  `1.11e-16`, where
  `N=((M_A-M_A_from_B)+(M_B_from_A-M_B))/2`; and
- stored completeness error `I-D` within `7.45e-9`.

These checks find no row join, formula, side-order, or arithmetic implementation
error.

## H1: how much causal variation is available?

| Quantity | N unique | Minimum | Q1 | Median | Q3 | Maximum | Sample SD | IQR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Absolute `C` | 77 | 0.785789 | 0.880435 | 0.912714 | 0.940546 | 1.012025 | 0.046374 | 0.060111 |
| `R_A_from_B` | 77 | 0.749304 | 0.867424 | 0.909181 | 0.953627 | 1.039344 | 0.064312 | 0.086202 |
| `R_B_from_A` | 77 | 0.623824 | 0.885794 | 0.927907 | 0.959538 | 1.005031 | 0.069189 | 0.073744 |
| Raw directional `N_A` | 69 | 12.875000 | 20.125000 | 23.437500 | 27.875000 | 33.234375 | 4.989635 | 7.750000 |
| Raw directional `N_B` | 75 | 13.750000 | 20.187500 | 23.375000 | 27.343750 | 34.539063 | 4.922589 | 7.156250 |
| Absolute raw numerator `N` | 72 | 13.375000 | 20.390625 | 23.187500 | 27.296875 | 33.121094 | 4.788080 | 6.906250 |
| Absolute `T` | 71 | 14.500000 | 21.875000 | 25.125000 | 30.093750 | 35.093750 | 4.930381 | 8.218750 |

For absolute C, sample variance is `0.00215057`, MAD is `0.0318882`,
coefficient of variation is `0.050956`, and the full range width is `0.226236`.
The raw directional numerator means are `23.678502` for `N_A` and `23.787591`
for `N_B`; every directional numerator is positive.
All 77 values are unique: there are no Spearman ties. Sorted adjacent gaps have
minimum `0.0000157`, median `0.0016004`, and maximum `0.0252262`; the largest
gap isolates only the minimum row. Seventy-six of 77 engines exceed `0.80`, 48
exceed `0.90`, 14 lie within `0.05` of full recovery, and two slightly exceed
1.0.

Thus the roster is substantively restricted to strong causal effects, but it
does retain 77 distinct ranks. Numeric narrowness alone cannot mechanically
destroy a Spearman association.

## H7: power with 24 independent groups

The frozen simulation used seed 1729, 20,000 Gaussian-copula datasets per
target rho, a two-sided alpha of 0.05, and true Spearman values from 0 to 0.90
in steps of 0.025. Latent Pearson correlation is
`2*sin(pi*rho_s/6)`. NumPy `Generator(PCG64)` is reset to seed 1729 separately
for each sample-size scenario. The primary `N=24` analysis treats dependency
groups as the independent units. `N=77` is an optimistic independence scenario
that treats every row as independent. The ordinary Spearman critical absolute
correlations are `0.404386` and `0.224174`, respectively.

| True rho | Power, 24 groups | Power, 77 independent rows |
| ---: | ---: | ---: |
| 0.000 | 0.0492 | 0.0527 |
| 0.025 | 0.0514 | 0.0541 |
| 0.050 | 0.0557 | 0.0720 |
| 0.075 | 0.0646 | 0.0988 |
| 0.100 | 0.0749 | 0.1351 |
| 0.125 | 0.0867 | 0.1860 |
| 0.150 | 0.1074 | 0.2496 |
| 0.175 | 0.1249 | 0.3248 |
| 0.200 | 0.1535 | 0.4111 |
| 0.225 | 0.1765 | 0.5041 |
| 0.250 | 0.2140 | 0.5991 |
| 0.275 | 0.2452 | 0.6754 |
| 0.300 | 0.2915 | 0.7595 |
| 0.325 | 0.3311 | 0.8224 |
| 0.350 | 0.3803 | 0.8779 |
| 0.375 | 0.4331 | 0.9205 |
| 0.400 | 0.4858 | 0.9528 |
| 0.425 | 0.5354 | 0.9719 |
| 0.450 | 0.5947 | 0.9856 |
| 0.475 | 0.6473 | 0.9918 |
| 0.500 | 0.7028 | 0.9964 |
| 0.525 | 0.7481 | 0.9984 |
| 0.550 | 0.8031 | 0.9992 |
| 0.575 | 0.8395 | 0.9998 |
| 0.600 | 0.8791 | 1.0000 |
| 0.625 | 0.9094 | 1.0000 |
| 0.650 | 0.9364 | 1.0000 |
| 0.675 | 0.9566 | 1.0000 |
| 0.700 | 0.9706 | 1.0000 |
| 0.725 | 0.9812 | 1.0000 |
| 0.750 | 0.9905 | 1.0000 |
| 0.775 | 0.9952 | 1.0000 |
| 0.800 | 0.9973 | 1.0000 |
| 0.825 | 0.9991 | 1.0000 |
| 0.850 | 0.9996 | 1.0000 |
| 0.875 | 1.0000 | 1.0000 |
| 0.900 | 1.0000 | 1.0000 |

Displayed powers are rounded to four decimals. In particular, the 24-group
value at rho 0.875 is `0.99995`, while the 24-group value at 0.900 and the
77-row values from 0.600 onward were exactly 1.0 in this simulation.

The dependency-aware 80%-power minimum detectable rho is **0.550**; power at
the preregistered useful rho of 0.40 is only **0.48575** (Monte Carlo SE
`0.00353`). The optimistic 77-row values are MDE `0.325` and power `0.95280`.
Fisher-z calculations agree: continuous MDE `0.54508` for 24 and `0.31463` for
77, with analytic rho-0.40 power `0.49264` and `0.95395`. The observed canonical
group-bootstrap result remains rho `-0.179110`, CI
`[-0.431377, 0.126014]`; it was not used to tune the simulation.

Under the frozen H7 criterion, the primary design was not reliably powered to
detect a moderate rho of 0.40. This supports insufficient prospective power;
it does not by itself reinterpret the realized negative estimate or its
bootstrap interval.

## H4: what READ and C actually track

The following point estimates use all 77 rows. Diagnostic intervals use 10,000
whole-group bootstrap draws with a fixed post-Gate-1 seed; they are not used to
select covariates.

| Association with READ_IG | Spearman rho | Grouped 95% interval |
| --- | ---: | --- |
| Absolute C | -0.179110 | [-0.438718, 0.120088] |
| Mean absolute concept amount | 0.649088 | [0.320006, 0.846221] |
| Mean delta norm | 0.629581 | [0.242093, 0.857630] |
| Mean absolute clean READ margin | -0.285307 | [-0.616693, 0.182680] |
| Causal normalization `|T|` | -0.279594 | [-0.612610, 0.185884] |
| Mean absolute path endpoint change | 0.924200 | [0.821432, 0.961500] |
| Mean absolute completeness error | 0.139729 | [-0.079850, 0.308629] |
| Relative completeness error | -0.594826 | [-0.738355, -0.403710] |
| Absolute raw causal numerator `N` | -0.290167 | [-0.599845, 0.173431] |

The side-specific correlations reinforce the scale result: READ versus concept
amount A/B is `0.560387/0.533177`, and versus delta norm A/B is
`0.550791/0.620301`. Mean amount and mean delta norm correlate at `0.856617`.
The relative-error correlation is partly denominator-coupled because endpoint
change appears in that denominator; it should not be treated as independent
causal evidence.

Normalization is strongly compressive:

- `rho(|N|,|T|)=0.950229`, grouped interval `[0.818830, 0.975717]`;
- `rho(|N|,|C|)=0.359836`, interval `[0.150432, 0.580115]`;
- `rho(|T|,|C|)=0.097251`, interval `[-0.196505, 0.355614]`;
- coefficient of variation falls from `0.201747` for `|N|` to `0.050956`
  for `|C|`, a 74.7% reduction; and
- IQR/median falls from `0.297844` to `0.065860`, a 77.9% reduction.

The raw swap effect and the clean denominator share almost the same scale, so
their ratio is close to a constant full-recovery fraction. The algebra is exact;
this is target normalization by design, not a division bug. READ, meanwhile,
tracks representation/path scale much more strongly than normalized C. H4 is
supported as a measurement-design explanation, not a software defect.

## H3 and the 16-step completeness check

READ strongly follows the endpoint of its own direction-defined path, but that
endpoint is rank-discordant with full-residual interchange:

| Association | Spearman rho | Grouped 95% interval |
| --- | ---: | --- |
| READ versus path endpoint magnitude | 0.924200 | [0.821432, 0.961500] |
| Signed `I_A` versus its endpoint `D_A` | 0.907271 | [0.789078, 0.957446] |
| Signed `I_B` versus its endpoint `D_B` | 0.887051 | [0.744502, 0.946498] |
| Path endpoint versus absolute C | -0.144271 | [-0.405460, 0.140508] |
| Path endpoint versus absolute N | -0.261167 | [-0.596480, 0.203508] |
| Path endpoint versus absolute T | -0.252462 | [-0.610470, 0.205370] |
| `J_A=-I_A` versus `R_A_from_B` | -0.013118 | [-0.249793, 0.213123] |
| `J_B=I_B` versus `R_B_from_A` | -0.101477 | [-0.312892, 0.136901] |
| `J_A=-I_A` versus raw `N_A` | -0.407540 | [-0.684216, 0.049540] |
| `J_B=I_B` versus raw `N_B` | -0.153204 | [-0.539353, 0.326967] |
| Oriented endpoint `-D_A` versus `R_A_from_B` | 0.013177 | [-0.219055, 0.229415] |
| Oriented endpoint `D_B` versus `R_B_from_A` | -0.026294 | [-0.241478, 0.189450] |

However, the preregistered completeness bar fails:

- per-row relative error `sum|I-D|/sum|D|`: median `0.230672`, p95
  `0.991759`;
- excluding the one row with both endpoint changes exactly zero: median
  `0.227971`, p95 `0.911997`, maximum `6.56746`;
- only `0/76` nonzero-endpoint rows are at or below 1%, `4/76` are at or below
  5%, and `13/76` are at or below 10%;
- global `sum|I-D|/sum|D|` is `0.207833`; and
- mean absolute directional error has median `0.045251`, p95 `0.102392`, and
  maximum `0.145896`.

Despite poor magnitude completeness, signed integral/end-point ranks remain
high (`rho=0.907271` for A and `0.887051` for B). Endpoint means have only 38
unique values and 54/77 rows participate in ties, consistent with coarse bf16
endpoint deltas. Source inspection provides a plausible mechanism: offsets are
formed in fp32 but cast back to the recorded bf16 hidden dtype before the
forward pass; logits are cast to float only after that bf16 computation, while
gradient-dot-delta accumulation is fp32. Exact finite-precision completeness is
therefore not guaranteed.

A secondary saved-integrand odd/even interleaving check has median relative
disagreement `1.67%` for A and `1.51%` for B, with p95 `7.20%/4.37%`. This is
much smaller than the endpoint completeness error, suggesting 16-point
quadrature alone is unlikely to explain all of it, but it is not a formal
convergence proof.

The endpoint/full-state discordance supports H3, but READ-versus-endpoint rho
`0.924200` is below the frozen particularly-clear threshold of 0.95, and the
decisive H3 criterion required accurate completeness. It is therefore partial
rather than decisive evidence. The stored artifacts cannot separate quadrature
error from bf16 path/endpoint quantization well enough to call READ a calibrated
magnitude of even its own path.

## H5: do signs or directional cancellation explain the failure?

- `I_A<0` in 76/77 rows; `I_B>0` in 77/77; both expected signs hold in 76/77.
- The only violation is `symmetric-032` (Finland/Jamaica):
  `I_A=+0.026877`, `I_B=+0.220764`, READ `0.123820`, and C `0.913758`.
  Its A endpoint is `-0.093750`, making it a completeness/sign anomaly rather
  than a common cancellation pattern.
- READ equals the predeclared oriented mean `(J_A+J_B)/2` within float tolerance
  for 76/77 rows. The oriented mean still has rho `-0.179610` with absolute C,
  grouped interval `[-0.436835, 0.122026]`.
- Direction magnitude imbalance has median `0.200260`, IQR `0.208682`, and
  maximum `0.965648`; A is larger in 39 rows and B in 38.
- `rho(|I_A|,|I_B|)=0.351543`. For the fixed oriented values,
  `rho(J_A,J_B)=0.350886`, grouped interval `[-0.148713, 0.714302]`.
- Fixed directions do not track their corresponding causal recoveries:
  `J_A/R_A rho=-0.013118` and `J_B/R_B rho=-0.101477`.
- Both causal recoveries are positive in 77/77 rows. Causal directional
  disagreement has median `0.065163`, maximum `0.323929`, and zero sharp flags.

Absolute averaging therefore hides a sign violation in one row, not a systemic
graded signal. H5 is strongly disfavored as the main explanation.

## Relation families and leave-one-group-out stability

| Relation family | Rows / groups | READ-versus-C rho | Grouped 95% interval |
| --- | ---: | ---: | --- |
| Element symbol | 8 / 8 | -0.833333 | [-1.000000, -0.199054] |
| Country capital | 41 / 10 | -0.179443 | [-0.386139, 0.227623] |
| US-state capital | 28 / 6 | -0.340996 | [-0.655867, 0.354348] |

All three point estimates are negative. The element family is very small, so
its interval should not be generalized beyond those eight rows.

Across 24 leave-one-whole-group-out analyses, every rho remains negative. The
range is `[-0.239171, -0.122794]`, median `-0.177024`, and IQR `0.039697`.
Removing Australia/Turkey gives the most negative value; removing
Colorado/Massachusetts gives the least negative. No single dependency group
drives the overall null/negative result.

## H6: layer and token alignment

READ and C use L16 in all 77 rows. C uses the raw manifest concept position;
READ adds the exact left-padding offset for paired batching. Re-tokenizing from
the frozen local tokenizer gives:

- zero prompt-length mismatches;
- zero solo or padded concept-token mismatches;
- zero recorded-position mismatches; and
- the intended concept token at all 154 targeted A/B positions.

All 154 clean top-token IDs also agree. Batched READ versus solo causal forwards
show small bf16 numerical drift: implied T ranks correlate at `0.998580`,
median absolute T difference is `0.09375`, maximum `0.375`, median relative
difference `0.003759`, and maximum `0.016502`. Drift has no association with
padding magnitude (`rho=0.0366`, p `0.752`). Both metric signs remain correct in
77/77 pairs.

There is no evidence of a layer, token, padding, A/B wiring, answer-token, or
join bug. Broader multi-layer or multi-token causal use remains untested, but it
cannot explain failure against the same-site L16 full-residual C. H6's current
code-flaw branch is ruled out.

## H2 remains unidentified

The selection audit found 16 candidates that failed only WRITTEN and zero
visible candidates rejected by later conditions. Those 16 have no causal C by
design. This diagnosis therefore cannot determine whether they contain a
weak-but-real causal tail. Their existence is not evidence for H2; the direct
causal-only follow-up specified at Gate 1 would be required.

## Phase 2 factual summary

The current data contain four simultaneous facts:

1. C is restricted to strong recovery but retains unique ranks.
2. Twenty-four independent groups provide less than 50% power for a true rho of
   0.40.
3. READ ranks concept/path scale and its own path endpoint, while the endpoint
   does not rank full-residual recovery; normalized C is strongly compressed by
   `N/T`.
4. The implementation alignment and algebra are correct, but finite-precision
   IG completeness is not accurate enough to certify a calibrated magnitude.

These observations are ranked against the frozen hypotheses in Phase 3; no
score has been changed or selected from them.

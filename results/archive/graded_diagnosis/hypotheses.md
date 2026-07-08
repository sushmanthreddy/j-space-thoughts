# Graded-meter diagnosis: frozen hypotheses and evidence tests

## Gate 1 status

These tests are frozen before any new row-level diagnostic correlation, power
simulation, family breakdown, leave-one-group-out result, or completeness
summary is computed. They may not be edited after Phase 2 begins.

## Rules shared by all hypotheses

- Analyze all 77 frozen verified engines and preserve their 24 dependency
  groups. Do not exclude rows or invent a weak/strong split.
- The graded target remains the existing absolute full-residual causal score
  `|C|`. Signed or directional components are diagnostic decompositions only;
  they do not replace the target.
- Do not flip a sign based on an observed correlation. The only directional
  orientation allowed below is fixed by the answer metric definition before
  results: A-to-B should reduce `M=logit(y_A)-logit(y_B)`, so `J_A=-I_A`, while
  B-to-A should increase it, so `J_B=+I_B`.
- Use Spearman correlation for monotonic associations. Where uncertainty is
  reported, resample complete dependency groups with 10,000 draws and seed
  1729 plus a fixed hypothesis-specific integer offset.
- Derived quantities below are read-only diagnostics, not replacement scores.
  No diagnostic may flow back into READ, C, row selection, or a new threshold.
- A code flaw requires evidence that implementation, indexing, or algebra
  differs from the frozen formula. A faithfully implemented but unsuitable
  estimand is a design limitation, not a software bug.
- Never use the pooled engine/control correlation as graded evidence.

## Fixed diagnostic quantities

For each engine pair, Phase 2 will read the stored directional values and form:

- `concept_amount_mean = mean(abs(h_A·v_A), abs(h_B·v_B))`;
- `delta_norm_mean = mean(||Delta_A||, ||Delta_B||)`;
- `clean_margin_mean = mean(abs(M_A), abs(M_B))`;
- `T_abs = abs(M_A-M_B)` from causal truth;
- `path_endpoint_mean = mean(abs(D_A), abs(D_B))`, where each `D` is the stored
  direction-path endpoint metric change;
- `ig_error_mean = mean(abs(I_A-D_A), abs(I_B-D_B))`;
- `ig_relative_error = sum(abs(I-D)) / max(sum(abs(D)), 1e-12)`;
- causal directional numerators `N_A=M_A-M_A_from_B` and
  `N_B=M_B_from_A-M_B`, with `N=(N_A+N_B)/2`; and
- fixed oriented IG effects `J_A=-I_A`, `J_B=+I_B`.

## H1 — Range restriction / insufficient target variation

**Type:** dataset/target limitation, not a code flaw.

**Test.** Report variance, standard deviation, IQR, MAD, full range, unique
value count, tie multiplicities, adjacent rank gaps, and distance from full
recovery `C=1` for engine `|C|`. Compare with both directional recoveries and
raw `|N|`. Interpret these together with H7 power.

**Supports H1.** Values occupy a near-ceiling band; ranks have substantial ties
or gaps comparable to numerical/measurement instability; raw `|N|` varies
materially more than `|C|`; and moderate monotonic effects have poor power.

**Counts against H1.** The target has many precisely ordered unique values, a
broad weak-to-strong continuum, little ceiling pile-up, and adequate power for
rho around 0.40.

**Decisive pattern.** Narrow scale plus rank loss/ceiling and an 80%-power MDE
at or above 0.50 supports H1. A narrow but continuous, well-resolved ordering is
not decisive because Spearman is invariant to numeric scale.

## H2 — The upstream WRITTEN threshold removes the low causal end

**Type:** dataset-selection limitation, not a code flaw.

**Test.** Use the completed selection audit to count below-WRITTEN candidates
that otherwise pass clean-answer gates. Direct support would require a future,
separately preregistered causal-only measurement of those excluded rows using
unchanged full-residual C and no READ. Within retained rows, report visibility
amount versus `|C|` only as indirect evidence.

**Supports H2.** A preregistered causal-only follow-up shows that below-WRITTEN
candidates contain reproducible nonzero effects that systematically extend the
causal distribution downward.

**Counts against H2.** Below-WRITTEN candidates are causally null/invalid or do
not occupy a lower causal range.

**Decisive pattern.** A substantial lower-effect tail appears exclusively below
the frozen visibility cutoff. Merely observing exclusions is not support; with
no C for those rows, H2 must remain unresolved.

## H3 — READ and C measure different interventions

**Type:** principled estimator/estimand mismatch, not a defect when both are
implemented faithfully.

**Test.** Compare each stored signed IG integral `I_i` with its own path endpoint
change `D_i` and quantify completeness. Separately compare fixed oriented
effects `J_A,J_B` and `path_endpoint_mean` with full-residual directional
recoveries, raw `|N|`, and normalized `|C|`.

**Supports H3.** IG accurately measures its prescribed direction path—median
relative completeness error below 1% and 95th percentile below 5%—while path
effects have weak rank association with full-residual recovery. A particularly
clear pattern is rho at least 0.95 for READ versus its endpoint magnitude but
absolute rho at most 0.20 versus `|C|`.

**Counts against H3.** The direction-path endpoint effects closely track the
full-residual directional effects (rho at least 0.60), while stored READ does
not, or IG completeness is too inaccurate to represent its own path.

**Decisive pattern.** Accurate completeness plus persistent endpoint/full-state
discordance means principled not-graded: adding IG steps cannot make the two
interventions the same.

## H4 — Path scaling or causal normalization confounds magnitude

**Type:** possible measurement-design limitation; a code flaw only if source
algebra or row alignment is wrong.

**Test.** Correlate READ_IG with the frozen concept amounts, delta norms, clean
margins, `|T|`, endpoint magnitude, and completeness error. Reconstruct `N_A`,
`N_B`, and `N`; verify `C=N/T` numerically; compare dispersion and ranks of
`|N|`, `|T|`, and `|C|`. Do not divide or residualize READ to create a new
predictor.

**Supports H4.** READ has absolute rho at least 0.50 with path scale,
representation amount, or confidence geometry while remaining flat against
`|C|`; or `|N|` is rank-rich but proportional to `|T|`, causing `|N/T|` to
collapse near a constant. A grouped interval excluding zero strengthens the
case.

**Counts against H4.** Non-endpoint covariates have absolute rho below 0.20 with
READ, normalization preserves numerator ranks, and source/algebra checks pass.

**Decisive pattern.** READ being essentially a path-scale rank implicates READ
scaling; broad `|N|` collapsing after division implicates target normalization.
Only a formula, indexing, or join mismatch counts as a code bug.

## H5 — Absolute averaging loses directional information

**Type:** possible aggregation-design limitation; not automatically a code bug
because absolute mean is the frozen estimator.

**Test.** Report expected-sign rates (`I_A<0`, `I_B>0`), fixed oriented effects
`J_A,J_B`, their magnitude imbalance, sign/cancellation patterns, and the two
causal directional recoveries. Compare `J_A` with `R_A_from_B` and `J_B` with
`R_B_from_A` only in this predeclared semantic orientation.

**Supports H5.** Expected-sign violations or severe directional imbalance are
common and absolute averaging hides them; oriented direction-level association
is at least 0.50 while absolute-mean READ remains at most 0.20 against `|C|`.

**Counts against H5.** Expected signs hold nearly universally and the two
oriented magnitudes are reasonably concordant. In that case
`READ_IG=(J_A+J_B)/2`, so the absolute value has not discarded sign information
on this roster.

**Decisive pattern.** Universal expected signs rule H5 out for this dataset.
Frequent violations converted into large positive scores support it, but any
new signed score would still require a separately frozen experiment.

## H6 — One layer and one token miss distributed causal magnitude

**Type:** measurement-scope limitation; a code flaw only if READ and C use
different layer/position indices.

**Test.** Audit that both paths use L16 and the same explicit-concept token after
padding resolution. Use family and leave-one-group-out stability only as
indirect heterogeneity evidence. A direct test requires a future fixed
layer/token grid and aggregation chosen without evaluation C.

**Supports H6.** A current indexing mismatch is found, or a preregistered
multi-site study finds a large, example-varying fraction of causal recovery
outside the selected site.

**Counts against H6.** Layer/position alignment is exact and a future fixed-site
audit finds other sites negligible or redundant.

**Decisive pattern.** An indexing mismatch is a code flaw. Otherwise a large,
variable outside-site fraction supports H6 for global causal use, but cannot by
itself explain failure against same-site full-residual C; that subspace mismatch
belongs to H3.

## H7 — Insufficient statistical power

**Type:** statistical-design limitation, not a code flaw.

**Test.** Freeze seed 1729, positive true rho, two-sided alpha 0.05, 80% power,
and Gaussian-copula simulation with Pearson latent correlation
`2*sin(pi*rho_s/6)`. At each target Spearman rho from 0.00 to 0.90 in increments
of 0.025, run 20,000 Monte Carlo datasets for two predeclared bounds:

1. 24 independent observations, treating dependency groups as the effective
   units (primary conservative design);
2. 77 independent observations, treating every row as independent (optimistic
   upper bound).

Use the ordinary two-sided Spearman test in each simulation and report the
smallest grid rho reaching 80% rejection power. Cross-check both with Fisher-z
analytic calculations. Also report the existing whole-group-bootstrap interval
as observed-design context, but do not use its point estimate to tune the
simulation.

**Supports H7.** The scientifically useful moderate association fixed here at
rho 0.40 has less than 80% power under the 24-group design, or the 80%-power MDE
is at least 0.50.

**Counts against H7.** The 24-group design has at least 80% power at rho 0.40;
an MDE at or below 0.30 would be strong counterevidence.

**Decisive pattern.** The frozen power curve determines testability. Low power
means the negative result cannot distinguish no relationship from a moderate
one; adequate power makes the flat association substantive evidence against a
useful graded meter.

## Interpretation guardrails fixed at Gate 1

- Support for H1, H2, or H7 implies not-testable-on-this-data, not broken code.
- Support for H3 implies a principled screen-not-meter interpretation.
- H4, H5, or H6 justify code-fix language only if a formula/index/alignment
  invariant fails. Otherwise they are alternative-measurement hypotheses.
- No hypothesis may be rescued by post-hoc score transformation. Any future
  estimator or dataset must be frozen independently of its evaluation C.

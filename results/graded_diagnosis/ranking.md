# Graded-meter diagnosis: evidence ranking

## Overall interpretation

No flaw was found in the audited formula, join, side-order, layer, or position
invariants. Pair joins, answer orientation, L16, concept-token positions,
padding adjustment, READ aggregation, directional causal formulas, and `C=N/T`
all reproduce from source and artifacts. The evidence is best organized as:

1. a **measurement-design mismatch**: READ ranks a concept-scaled direction
   path, while C is a normalized full-residual recovery fraction;
2. a **strong-only target population** whose normalized C values are compressed
   near full recovery; and
3. a conservative power analysis showing limited prospective sensitivity to a
   moderate graded association at 24 effective units.

There is also a real numerical completeness limitation: 16-step stored IG does
not meet its completeness bar. That weakens claims that READ is a calibrated
magnitude of its own path. The current artifacts do not isolate quadrature from
bf16 path/endpoint quantization, but they do rule out a side, index, join, or
algebra bug.

## Ranked causes

### 1. H4 — path scaling and causal normalization confound magnitude

**Evidence status: strongly supported as measurement design; code-bug branch
rejected.**

Evidence for:

- READ correlates with mean concept amount at rho `0.649088`, grouped interval
  `[0.320006, 0.846221]`, and mean delta norm at `0.629581`
  `[0.242093, 0.857630]`.
- READ correlates most strongly with its own endpoint magnitude: `0.924200`
  `[0.821432, 0.961500]`, while its correlation with absolute C is `-0.179110`.
- Raw causal numerator `|N|` and denominator `|T|` have rho `0.950229`
  `[0.818830, 0.975717]`.
- Normalization reduces coefficient of variation from `0.201747` for `|N|` to
  `0.050956` for `|C|` (74.7%) and IQR/median by 77.9%.

Evidence against or limiting the claim:

- READ correlations with clean margin (`-0.285307`) and `|T|` (`-0.279594`)
  have intervals spanning zero; READ is not simply a clean-confidence score.
- `|N|` retains a modest positive association with `|C|`, rho `0.359836`
  `[0.150432, 0.580115]`; normalization does not destroy every rank.
- Every formula and join reproduces: maximum C algebra error is `1.11e-16`,
  and stored READ differs from mean directional magnitude by at most `2.98e-8`.

Conclusion: the intended formulas measure differently scaled quantities. The
`h·v` multiplier naturally makes READ sensitive to representation/path scale,
while C intentionally removes the clean pair scale. This is a confound for
graded comparison, not incorrect code.

### 2. H7 — insufficient statistical power

**Evidence status: supported by the frozen prospective criterion.**

Evidence for:

- With 24 independent group units, simulated power for true rho `0.40` is only
  `0.48575`.
- The 80%-power grid MDE is `0.550`; Fisher-z gives continuous MDE `0.54508`.
- This clears the preregistered H7 support criterion of MDE at least 0.50.

Evidence against or limiting the claim:

- Under the optimistic 77-independent-row scenario, rho-0.40 power is
  `0.95280` and MDE is `0.325`.
- The realized grouped interval is `[-0.431377, 0.126014]`; its upper endpoint
  is small and the observed point estimate is negative. Low prospective power
  does not turn those realized data into positive evidence.

Conclusion: under the preregistered conservative 24-unit approximation, the
experiment was not reliably sensitive to a moderate positive association.
Actual power for the unequal 77-row/24-cluster design was not estimated. H7
limits how definitively prospective sensitivity can be characterized; it does
not explain the negative point estimate, override the realized interval's upper
endpoint of 0.126, or rescue the existing meter.

### 3. H3 — direction-path sensitivity and full-residual C are different estimands

**Evidence status: strongly plausible but not decisive because completeness
fails.**

Evidence for:

- READ follows its own path endpoint at rho `0.924200`, while endpoint magnitude
  versus C is `-0.144271` `[-0.405460, 0.140508]`.
- Fixed oriented `J_A` versus causal `R_A` is `-0.013118`; `J_B` versus `R_B`
  is `-0.101477`. The direction-path ranks do not resemble full-state recovery
  ranks.
- The source definitions are genuinely different: Delta moves only along two
  J-Lens directions, whereas C replaces the complete residual vector.

Evidence against or limiting the claim:

- READ-versus-endpoint rho is below the frozen particularly-clear threshold of
  0.95.
- Median relative IG completeness error is `23.1%`, p95 `99.2%`; only `4/76`
  nonzero-endpoint rows meet 5% error. The decisive H3 condition required
  accurate completeness.
- The artifacts cannot distinguish remaining quadrature error from bf16
  endpoint/path quantization. A numerical measurement limitation may contribute
  alongside estimand mismatch.

Conclusion: different interventions are a principled reason a binary screen
need not be a graded C meter, but this run cannot cleanly isolate that reason
from incomplete finite-precision IG magnitude. Adding steps alone is not proven
to help; the saved-integrand check suggests it is not the entire problem.

### 4. H1 — range restriction / no useful causal continuum

**Evidence status: partially supported for construct coverage; not decisively
supported as a rank-statistic explanation.**

Evidence for:

- Absolute C is confined to `[0.785789, 1.012025]`, median `0.912714`, IQR
  `0.060111`, and coefficient of variation `0.050956`.
- `76/77` engines exceed C `0.80`; there is no weak or medium causal tail.
- C normalization compresses relative dispersion by about 75% compared with
  raw N, and H7 shows poor power for moderate effects.

Evidence against:

- All 77 C values are unique with no ties. Median adjacent rank gap is
  `0.001600`; the largest gap `0.025226` isolates only one row.
- Spearman is invariant to numeric scale, so a narrow range does not itself
  erase rank order.
- Family and leave-one-group-out results remain negative rather than becoming
  noisy mixtures of positive and negative subgroup effects.

Conclusion: the dataset does not cover the causal continuum a useful graded
meter should rank. That limits the scientific test, but “no variance to rank”
is too strong: ranks exist and READ fails to follow them.

### 5. H2 — WRITTEN excludes weak-but-real causal concepts

**Evidence status: unresolved.**

Evidence for or motivating concern:

- `16/93` evaluation candidates fail only WRITTEN while passing all clean-answer
  checks. They are the only available place a lower causal tail might exist.
- Every visible candidate passes all non-WRITTEN conditions, so those later
  conditions cannot explain the strong-only retained range.

Evidence against or missing:

- The 16 excluded rows have no C by audit design. Exclusion does not establish
  that they are causal, weak, or part of a continuous lower tail.
- The selection audit found zero visible candidates discarded by another gate.

Conclusion: current artifacts neither support nor refute H2. It must not be
promoted to an explanation without a preregistered causal-only study of
below-WRITTEN concepts.

### 6. H6 — one layer/token or an indexing mistake

**Evidence status: current code-flaw branch ruled out; broader scope untested.**

Evidence for or motivating concern:

- The study measures only one explicit token at calibration-selected L16, so it
  cannot establish global multi-site graded use.
- No preregistered multi-layer/token comparison exists.

Evidence against a current implementation cause:

- All 77 READ and C records use L16 and the same semantic concept position after
  exact left-padding adjustment; all 154 targeted token IDs match.
- All pair IDs, metadata, answer orientation, top tokens, and joins match.
- Batched-versus-solo clean T ranks correlate `0.998580`; median relative drift
  is `0.003759`, maximum `0.016502`, and there is no detectable rank
  association with padding magnitude (`rho=0.0366`, p `0.752`).
- The target being diagnosed is itself same-site L16 full-residual C. Missing
  causal effects elsewhere do not explain failure against that target.

Conclusion: no layer/position code bug exists. Multi-site READ might answer a
different global-use question, but it is not a fix for this same-site result.

### 7. H5 — absolute averaging discards the graded signal

**Evidence status: strongly disfavored.**

Evidence for:

- One row, `symmetric-032`, violates the expected A-direction sign, and absolute
  averaging hides that violation.
- Directional magnitude imbalance has median `0.200260` and one extreme value
  `0.965648`.

Evidence against:

- Expected signs hold in `76/77` A directions and `77/77` B directions; both
  hold in `76/77` pairs.
- For those 76 pairs, absolute READ equals the predeclared oriented mean. Across
  all rows the oriented mean still correlates `-0.179610` with C, interval
  `[-0.436835, 0.122026]`.
- Oriented directions do not recover a hidden relation: `J_A/R_A=-0.013118` and
  `J_B/R_B=-0.101477`, far below the frozen 0.50 support threshold.
- Both causal directional recoveries are positive in all 77 rows, with zero
  sharp disagreement flags.

Conclusion: replacing the absolute mean with the preregistered oriented mean
does not recover graded association; sign loss is not a systemic explanation on
this roster. H5 explains one numerical anomaly, not the failed meter.

## Code flaw versus design limitation

| Category | Finding |
| --- | --- |
| Audited software/alignment flaw | **Not supported.** Formula, joins, layer, token, A/B orientation, and positions pass. |
| Numerical measurement limitation | **Supported.** bf16/16-step IG fails strict completeness; exact source is not isolated. |
| Dataset/statistical limitation | **Supported.** Strong-only C coverage and 24-group power limit the test; WRITTEN's effect remains unknown. |
| Principled estimator property | **Supported but not cleanly isolated.** Direction-path sensitivity and normalized full-state recovery rank different effects. |

No specific local code correction is supported by this audit; the unresolved
numerical completeness issue requires a separately frozen precision experiment.
The honest conclusion is: **no flaw was found in the audited formula/alignment
invariants; the current estimator is a validated binary screen, while gradedness
is poorly posed against this normalized strong-only target and prospective
sensitivity is limited under the conservative 24-unit power approximation.**

# Graded-meter recommendation

## Decision: NO-GO

**Do not pursue further tuning of the current `READ_IG` estimator as a graded
meter of normalized full-residual `|C|`. Keep the validated binary
relevant-versus-idle screen as the project deliverable.**

This is a NO-GO for the current estimator/target pairing, not a theorem that no
future independently designed graded-use estimator can exist.

## Why NO-GO follows from the evidence

The negative result is not explained by one correctable defect:

1. **The estimands differ.** READ follows its own direction-path endpoint at
   rho `0.924`, but that endpoint does not rank full-residual `|C|`
   (`rho=-0.144`). The fixed oriented directional associations with causal
   recoveries are `-0.013` and `-0.101`.
2. **The target is compressed by design.** Raw swap numerator and clean
   denominator correlate `0.950`; division reduces relative dispersion by
   roughly 75%. READ instead follows concept amount (`0.649`) and path norm
   (`0.630`). No formula or join error was found.
3. **Magnitude completeness is inadequate.** Median relative IG completeness
   error is `23.1%`, p95 `99.2%`. Improving numerical precision could make READ
   a better measure of its own path, but the measured endpoint itself is already
   rank-discordant with C.
4. **The obvious local alternatives are unsupported.** Expected signs hold in
   76/77 pairs and the preregistered oriented mean remains flat, so a signed
   aggregation is not justified. Layer/token alignment is exact, and a
   multi-site score would target a different quantity from same-site L16 C.
5. **Coverage and power limit generality but do not provide a repair.** The
   roster is strong-only, and the conservative 24-unit simulation has only
   48.6% power for true rho 0.40. Yet all 77 C ranks are unique, every family
   point estimate is negative, all 24 leave-one-group-out estimates remain
   negative, and the realized interval's upper endpoint is `0.126`.

No single permitted change resolves all three central barriers: a broader
dataset would not repair path-versus-full-state mismatch or completeness; a
precision change would not create a causal continuum or make the path endpoint
rank C; and a rescaling, signed variant, or new aggregation selected after
seeing evaluation C would be post-hoc estimator search.

## Classification of the failure

- **Audited software flaw:** not supported.
- **Numerical completeness limitation:** supported, source not fully isolated.
- **Dataset/testability limitation:** supported for strong-only coverage and
  conservative prospective sensitivity; WRITTEN's selection effect remains
  unresolved.
- **Principled estimator property:** supported as the main scope conclusion.
  The score is behavior-sensitive enough to screen used versus idle concepts,
  but is not calibrated to normalized full-state recovery magnitude.

The appropriate label is therefore:

> **Principled not-graded for the current READ_IG/normalized-C pairing, with
> additional not-testable-on-this-data limitations.**

## Frozen writeup sentence

Use this sentence without strengthening it:

> In this frozen experiment, READ_IG is supported as a relevant-versus-idle
> screening score, but it is not supported as a graded estimator of normalized
> full-residual causal recovery within relevant examples.

Do not say that graded use is impossible, that low power proves a hidden
positive relation, or that a code fix has been identified.

## Reopening rule for any future independent program

This diagnosis does not recommend another tuning cycle. If a future project is
proposed independently, the NO-GO should be reconsidered only under all of the
following rules frozen before its model runs:

1. **Direction:** the primary association must be positive; no sign flip is
   allowed.
2. **Independent magnitude order:** task dependency levels must be authored by
   an ex ante prompt/data parameter—such as a fixed coefficient controlling the
   contribution of a concept-derived term while holding visibility high—not by
   sorting, thresholding, or selecting on evaluation C. The coefficient must
   change the prompt/task dependency; it may not directly multiply READ, the
   behavior metric, or C.
3. **Firewall and one-shot roster:** freeze the complete roster, levels, splits,
   estimator, dtype/precision, and decision rules first. Compute and hash READ
   while blinded to all evaluation C and before evaluation C exists. Only then
   compute C for every frozen row. C may validate the authored continuum but may
   not change row inclusion, levels, prompts, estimator, precision, or whether
   the READ result is reported.
4. **Independent units and matched power rule:** use at least 60 whole
   dependency groups, with the final count chosen by a prospective clustered
   simulation. At a separately frozen true rho of `0.50`, that simulation must
   give at least 80% probability of meeting the exact compound meter-success
   event in Rule 7. The 60-group floor, rho-0.50 design alternative, and later
   thresholds are new normative reopening criteria, not estimates from the
   current data.
5. **Target validity:** authored level medians of absolute C must be monotonic,
   and the median of the highest-dependency level minus the median of the
   lowest-dependency level must be at least `0.40`. Evaluation C validates this
   frozen order but never defines or revises it.
6. **Numerical validity:** median IG relative completeness error must be below
   1% and p95 below 5%, matching the Gate-1 criterion and frozen without
   reference to READ/C correlation.
7. **Meter success bar:** on untouched groups, the positive-direction
   engine-only rho must be at least `0.40` and its whole-group-bootstrap 95%
   lower bound must exceed `0.20`. The 0.20 lower-bound requirement is a new,
   deliberately stringent reopening bar.
8. **Failure criterion:** if target and numerical validity pass but the meter
   success bar fails, retain NO-GO and stop pursuing current READ_IG as a graded
   C meter. If either validity gate fails, report benchmark failure, retain the
   current NO-GO, and draw no new estimator conclusion. In either case, do not
   recycle those held-out groups, revise levels/prompts, or rescale,
   residualize, sign-flip, or change aggregation after inspection; reopening
   would require a wholly new preregistration and untouched groups.

These are conditions for overturning the conclusion, not a recommendation to
spend resources on that experiment now.

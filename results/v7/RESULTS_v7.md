# V7 matched-comparison results

## Pre-registered decision rule

Pre-registered on 2026-07-09 before generating, inspecting, or evaluating any
v7 matched-design model outputs.

The primary held-out test compares ENGINE with the matched idle-DASHBOARD using
the identical measured quantity in both conditions:

`M = logit(answer_A) - logit(answer_B)`

`READ_IG` **passes** only if its dependency-group-held-out ROC AUC is at least
`0.70` and the lower endpoint of its 10,000-draw dependency-group bootstrap
confidence interval is strictly greater than `0.50`.

The frozen 16-step `READ_IG` estimator and frozen `READ_local` estimator will
not be tuned, sign-flipped, or redefined after inspecting v7 results. Failed
verification items are `UNVERIFIED`, excluded from confirmatory evaluation,
and never relabeled.

The final decision will use exactly one of these forms:

- **SURVIVES:** AUC stays high on the matched design; use-vs-idle detection is
  real in this frozen setting and is not explained by the old mismatched logit
  comparison.
- **COLLAPSES:** AUC drops toward chance on the matched design; the prior
  `1.000` was a mismatched-comparison design artifact and is corrected here.

## Results

Pending execution of the pre-registered v7 protocol.

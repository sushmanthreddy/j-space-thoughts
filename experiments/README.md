# Experimental archive

This directory is the audit trail for approaches that failed, were exploratory,
or were superseded by the final five-notebook pipeline. The files are preserved
as they were used; they are not maintained as a second runnable implementation.
Use the root `src/`, `notebooks/`, and `results/` directories for the final
result.

The attribution boundary matters throughout this history. The Jacobian Lens
(J-Lens/J-Space) method, lens readout, and canonical swap demonstrations come
from Anthropic's `jacobian-lens` project. Reproducing and calibrating those
pieces here is upstream validation, not our contribution. Our work begins with
the question asked after a concept is visibly **WRITTEN**: can a donor-free,
gradient-only **READ** score predict whether that concept is causally used? The
final contribution is the detector and its firewalled validation, not J-Lens
itself.

Historical numbers below describe the run in the named archive. They should not
be mixed with the final result; final claims and provenance live in `results/`.

## 01 — `01_broken_instrument/`

This first run tried to reproduce J-Lens concept readout and use broad
source-to-foil edits for two-hop questions, controls, and attribution analyses.
The basic calibration failed. On the canonical legs example the clean answer
was `8`; replacing the represented concept *spider* with *ant* should have moved
the answer to `6`, but the edited run returned `4`. The aggregate record marks
the entire generation `INVALIDATED_INSTRUMENT_FAILURE`.

That means the apparent negative result from this run was not evidence against
concept use: an instrument that misses its known positive control cannot test
the hypothesis. The lesson was to require model-wrapper agreement and exact,
repeatable known swaps before interpreting any new causal or READ statistic.

## 02 — `02_violent_intervention/`

The repair run audited the upstream package boundary, resolved token surfaces
and direction conventions, and recovered all `3/3` declared swaps under one
fixed configuration. Its 20-prompt wrapper check had maximum mean KL
`1.6602172081547906e-08`, well below the `1e-3` gate. These were successful
reproduction and plumbing checks.

The intervention was nevertheless too blunt for science: it edited all prompt
positions across a layer band at strength `alpha=2`. Across `24` unrelated-text
measurements, mean delta NLL was `0.6233231921990713` and mean absolute delta NLL
was `0.6690807243188223`, both incompatible with the `0.25` preservation bar;
only `1/8` language-control passages met the intended positive-control rule.
The causal effect could therefore reflect general model damage rather than use
of the named concept. The lesson was to localize the intervention to places
where the concept is present and to count an empty edit mask as no measurement,
not as reassuring zero damage.

## 03 — `03_read_attempts_failed/`

This folder collects several substantive failures rather than one discarded
implementation:

- The local attribution READ estimate was almost unrelated to the nonlinear
  causal endpoint: Pearson `r=0.061844689450020945`, 95% bootstrap CI
  `[-0.40641509476840315, 0.4824036702258359]`, `N=20`.
- A static downstream weight-capacity score called none of the `8` causally
  quiet narration cases low-READ (`0/8`). It measured capacity to respond to a
  direction, not whether the current behavior used it.
- The preregistered behavior-specific path score was also unsuccessful on the
  `21` known-answer cases: Spearman `rho=-0.07662337662337662`, 95% cluster
  bootstrap CI `[-0.5176576077155814, 0.4089291552379375]`. All `8` narration
  paths were empty, so those controls were non-estimable rather than low-READ.
- The later v5 validation expanded to `163` cases (`155` engine and `8`
  dashboard cases; `75` versus only `4` independent semantic concepts). The two
  causal targets agreed only moderately (`rho=0.4101667482090183`, `N=155`),
  and `49/163` declared labels failed verification. The only complete score,
  R1, ranked the classes backward: held-out AUC `0.07833333333333332`, 95%
  dependency-cluster bootstrap CI
  `[0.02142857142857143, 0.15203244109494102]`. R2 and R3 remained undefined
  under the frozen complete-coverage rule; missing values were not replaced by
  favorable zeros.

These attempts were not rescued by flipping a sign after seeing the outcome.
They were superseded because the target, labels, and domains were not clean
enough to answer the question. The lesson was to construct verified, matched
engine/idle prompts, use full-residual interchange as causal truth, freeze the
gradient-only estimator before reading that truth, and keep dependency groups
intact during evaluation.

## 04 — `04_alpha_sweep/`

This run tested eight strengths from `0.25` through `2.0` under three masking or
swap policies (`24` configurations total). The only preregistered selectable
policy never passed the joint calibration gate: it achieved `0/3` known swaps
at `alpha=0.25` and at most `2/3` at `alpha=1.5` and `2.0`. Consequently no
intervention was selected and the downward-extension rule was not triggered.

An exploratory carrying-position policy did recover `3/3` swaps at
`alpha=1.5`, but it was explicitly nonselectable because it had not been frozen
in advance. Moreover, all `24/24` capability masks were empty, so their reported
zero delta NLL represented `NO_EDIT_OPPORTUNITY`, not a preservation result.
The run failed `G-ALPHA` and made no hypothesis-level inference. The lesson was
that sweeping a broken policy cannot validate it, and that a post-hoc surgical
variant can motivate the next protocol but cannot retroactively pass a frozen
gate.

## 05 — `05_prior_versions/`

This directory preserves the immediate predecessors of the cleaned pipeline.
They include successful work as well as instrument corrections, so “archived”
here means superseded rather than necessarily failed.

### `symmetric_go/`

The symmetric run introduced the core matched design. Its setup went through
three recorded corrections before any cheap READ score was used: the first
prompt format left all cases unverified; a final-answer-token dashboard made the
WRITTEN check void; and full-residual interchange at L26 had engine median
`|C|=0.0022`. A latent context-boundary variant was also too weak (best
calibration medians `|C_engine|=0.0076` and `|C_dashboard|=0.0039`). Those
failures led to an explicit single-token concept and a calibration-selected L16
intervention.

The corrected run produced `118` candidates, `25` calibration pairs, `93`
evaluation pairs, and `77` verified pairs in `24` evaluation dependency groups.
Median engine `C` was `0.9127144298688193`; old-dashboard median `|C|` was
`0.005082592121982211`. Firewalled READ_IG achieved held-out AUC `1.0`, 95% CI
`[1.0, 1.0]`, while the static capacity baseline was `0.5`. This established a
binary signal, but the pooled association was later recognized as dominated by
between-class separation. The scripts and figure here were superseded by the
clean modules and thin notebooks in the repository root.

### `v6_isolated/`

The isolated v6 stress test asked whether that perfect binary separation was a
graded meter and whether it survived answer-type-matched controls. Among the
same `77` verified engines, READ_IG had engine-only Spearman
`rho=-0.17911036332088964`, 95% group-bootstrap CI
`[-0.4313773341424785, 0.12601395213937172]`; the causal range was already narrow
and strong (`|C|=0.7857889237199582` to `1.0120253164556963`). Thus graded use
was not supported.

All `77/77` hard dashboards verified. Their median `|C|` was
`0.00646551724137931`, and engine-versus-hard-dashboard READ_IG AUC remained
`1.0`, 95% CI `[1.0, 1.0]`. This ruled out arithmetic answer type as the sole
explanation for the binary separation, while leaving the graded claim false.
The honest verdict was `ARTIFACT (partial)`: binary relevant-versus-idle
detection survived; a graded causal-use meter did not. This isolated tree is
now integrated into the final pipeline and retained for auditability.

### `shared_helpers/` and `results/`

`shared_helpers/` contains the former model, direction, intervention, and metric
utilities plus their historical tests. Their working behavior was consolidated
into the final `src/` modules; these copies remain unchanged so earlier runs can
be interpreted in their original context. `results/metrics_all_generations.json`
is the machine-readable aggregate spanning the invalid v1 instrument through
the symmetric run, and `RESULTS_pre_v6_mixed.md` is the corresponding pre-clean
narrative. They are provenance records, not the final reporting surface.

## 06 — `06_scale_and_ambiguity/`

This branch explored whether earlier claims transferred to Qwen2.5-14B and to a
120-item ambiguity benchmark. The locally fitted 14B lens used `100` prompts;
`226` two-hop items were clean-eligible and the 20-prompt wrapper gate passed
with maximum mean KL `4.3200365951179265e-08`. However, the strict canonical
swap gate still failed (the directional subgate passed), and the independent
40-concept mean-difference validation failed its top-5 and top-1-above-chance
criteria. A 32B run was not downloaded: projected model weights alone were
`102.8 GiB` against a `100 GiB` quota.

The ambiguity arm completed all `120` items but was explicitly diagnostic
because the upstream strict G2 instrument gate had failed. Only `4/120` raw
swaps flipped the committed reading, and `0/120` were robust in both
counterbalanced probes (robust flip rate `0.0`, 95% Wilson CI
`[0.0, 0.031019166418703472]`). It was therefore neither a valid confirmation
nor part of the final explicit-concept result. The lesson was to stop an
inferential branch when its causal instrument fails, even if downstream
notebooks can still produce descriptive numbers.

## 07 — `07_go_only_localization/`

This post-GO exploratory run localized signed mediation in three
directionally-stable, high-effect engines and proposed a top-8 downstream
attention/MLP circuit for each. When every component outside each proposed
circuit was ablated, the faithfulness fractions were
`0.3649447949526814`, `0.2386919315403423`, and `0.36233766233766235`.

Those values do not support a faithful compact-circuit claim. Localization was
not needed for the binary detector and was archived instead of being folded
into the headline result. The lesson was that successfully detecting use does
not by itself identify a small mechanism that faithfully carries the effect.

## Metric provenance for this narrative

- The broken spider-to-ant output and the plain-language failure chronology are
  recorded in `03_read_attempts_failed/results/WRITEUP_v5.md`.
- Repair, calibration, alpha-sweep, READ-v2/v4/v5, symmetric-run, and
  localization values come from
  `05_prior_versions/results/metrics_all_generations.json`; its companion
  `05_prior_versions/results/RESULTS_pre_v6_mixed.md` documents the frozen
  protocol and pre-READ instrument amendments.
- The isolated stress-test values come from
  `05_prior_versions/v6_isolated/results/metrics_v6.json` and
  `05_prior_versions/v6_isolated/results/RESULTS_v6.md`.
- Scale values come from
  `06_scale_and_ambiguity/results/scale_comparison.json`; the ambiguity item
  count, raw flip rate, and counterbalanced robust-flip interval are preserved
  in the executed output of
  `06_scale_and_ambiguity/notebooks/05_ambiguity_flagship.ipynb`.
- Final headline values are independently snapshotted in
  `../results/PROVENANCE_pre_refactor.json`; after the clean rerun, use the
  corresponding post-refactor provenance and root `results/` files as the
  authoritative final record.

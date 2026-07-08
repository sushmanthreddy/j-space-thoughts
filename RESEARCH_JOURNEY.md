# Research journey: from visible concepts to validated use screening

## Problem and motivation

Anthropic's Jacobian Lens (J-Lens) and J-Space provide the representational
foundation for finding verbalizable concepts in model activations. This project
starts after that detection step: a visible concept can be either an **engine**
that affects the current answer or a **dashboard** that reflects context without
driving the answer. Direct intervention can test that distinction, but it is an
awkward routine monitor, so the contribution tested here is a donor-free,
gradient-only READ detector and a firewalled validation of it—not J-Lens,
J-Space, or a new lens.

The distinction matters for safety monitoring because treating every visible,
harmful-looking trace as active can create false alarms and false mechanistic
explanations.

The detector is called “cheap” in that operational sense: it does not require a
matched donor state or edited evaluation forward pass. No runtime, memory, or
throughput comparison was run, so the work does not claim a measured speedup.

## The arc of failures and fixes

### 1. The first apparent result used a broken instrument

The initial intervention failed its most basic positive control. Replacing the
represented concept *spider* with *ant* should have moved a leg-count answer
from `8` to `6`; the edited run returned `4`. The apparent scientific result
was discarded. The repair required exact token surfaces, the correct Jacobian
convention, the correct intervention configuration, and wrapper agreement
before the three canonical swaps reproduced `3/3`.

That episode established a rule used throughout the later work: a READ result
is uninterpretable until the causal instrument passes known cases.

### 2. A working swap was still too violent

The repaired intervention edited every prompt position across a layer band at
`alpha=2`. It damaged unrelated prediction behavior, and only `1/8` language
controls met the intended positive-control rule. Under that intervention, an
apparent causal effect could be manufactured by general model disruption rather
than by changing the named concept's role.

Restricting the edit to positions where the source concept was visible made the
intervention surgical. It retained the known swaps while the eight narration
passages spanning four concepts stayed causally quiet. This was not a universal
preservation result: the separate unrelated-text masks were empty, so their
zero changes were no-op measurements. Still, the narration controls were
causally quiet under this surgical measurement, exposing READ—not causal
measurement—as the main bottleneck.

### 3. Plausible READ definitions measured capacity, not current use

Several attempts failed under their frozen rules:

- Local attribution was nearly unrelated to its nonlinear intervention
  endpoint: Pearson `r=0.062`.
- A global static-capacity score classified `0/8` causally quiet narration
  cases as low-READ.
- A behavior-specific, path-restricted score reached Spearman `rho=-0.077` on
  the known-answer cases; all eight narration paths were empty and therefore
  non-estimable, not favorable zeros.

The static score exposed the central conceptual distinction: a model component
can be wired to respond to a direction without consulting that direction for
the answer currently being produced. The local and path-restricted estimators
instead failed empirically or could not produce complete measurements. None of
the results was rescued by flipping signs, filling missing scores, or continuing
estimator search after inspection.

### 4. The final pipeline separated the roles cleanly

The working protocol replaced heterogeneous engines and dashboards with
reciprocal, matched prompts. The same explicit concept and fact context appear
in both conditions: an engine asks for an answer that depends on the concept,
while a dashboard asks for an unrelated answer. A harder dashboard keeps the
engine's semantic answer type, removing arithmetic output type as the obvious
separator.

The pipeline then assigns distinct jobs to four stages:

1. **Verify visibility and task behavior.** At the explicit concept token and
   L16, both reciprocal concepts must pass the frozen WRITTEN criterion, and
   clean engine and dashboard answers must be correct.
2. **Establish causal truth.** Swap the complete residual state at that token
   in both directions. The normalized score `C` is the signed, unclipped mean
   recovery across those directions.
3. **Freeze cheap READ independently.** `READ_IG` integrates answer gradients
   along the prespecified path between paired J-Lens directions. It cannot read
   donor states, evaluation `C`, or intervention outputs.
4. **Evaluate only after both artifacts exist.** Pair IDs are joined after the
   causal and gradient paths are fixed, and dependency groups remain intact in
   folds and bootstrap draws.

This ordering matters as much as the formula. It prevents an evaluation
intervention from choosing an example's sign, transformation, exclusion, or
READ score.

## What was established

Within this frozen Qwen2.5-7B-Instruct experiment, `READ_IG` separates all 77
verified engines from both idle-control families across 24 dependency groups.
The held-out ROC AUC is `1.000000` with grouped interval
`[1.000000, 1.000000]` against both the original and answer-type-matched
dashboards. The causal sanity check independently shows strong engine recovery
and near-zero dashboard effects.

This supports a binary relevant-versus-idle screening result in the declared
setting. It is perfect empirical ranking on this roster, not universal
accuracy, a deployment threshold, or evidence about future false-positive
rates. See [RESULTS.md](RESULTS.md) for the compact numeric record.

## What was not established

`READ_IG` is not supported as a graded causal-use meter. Within the 77 already
strong engines, its Spearman association with normalized full-residual `|C|`
is `-0.179110`, with grouped interval `[-0.431377, 0.126014]`. The pooled
engine/dashboard correlation is not graded evidence because it mostly restates
the binary class gap.

The appropriate conclusion is narrower than “graded use is impossible.” It is
a NO-GO for further tuning of the current `READ_IG` / normalized full-residual
`|C|` pairing on these data. A future estimator or an independently designed
magnitude benchmark remains an open research question.

## Why the two stress tests matter

### Selection audit

The held-out funnel contains 93 evaluation candidates. Of these, 77 pass
WRITTEN in both directions and 16 fail only WRITTEN; every visible candidate
passes the later clean-answer conditions, so those later gates reject `0`
visible candidates. This supports Finding A only within the declared population
of L16-visible concepts: non-WRITTEN gates did not create the strong-only causal
range. It does **not** establish that the WRITTEN threshold itself is
magnitude-neutral, because no causal `C` exists for the excluded 16.

### Graded diagnosis

The read-only diagnosis found no formula, join, answer-orientation, layer,
token, padding, or side-order defect. Instead, `READ_IG` closely follows the
endpoint of its own direction-defined path (`rho=0.924200`), while that endpoint
does not rank normalized full-residual `|C|` (`rho=-0.144271`). READ also tracks
concept amount and path scale, whereas division by the clean contrast compresses
the full-residual recovery target.

The diagnosis adds two limitations rather than a repair. The engine roster is
strong-only, and a conservative approximation that treats the 24 dependency
groups as independent units has limited prospective power for a moderate
association. In addition, the stored 16-step IG fails its completeness bar, and
the contributions of quadrature and bf16 path/endpoint quantization remain
unresolved. Those issues constrain interpretation, but neither converts the
observed negative association into evidence for gradedness. The principled
decision is therefore to retain the binary screen and stop tuning the current
graded pairing.

## Limitations and open questions

The evidence comes from one pinned model, one calibration-selected layer,
explicit single-token concepts, three structured relation families, and 24
observed dependency groups. Some scope limitations enter with J-Lens/J-Space as
the verbalizable-direction foundation; this protocol narrows them further by
testing only one token and L16. Transfer to other models, layers, languages,
implicit or multi-token concepts, free-form reasoning, and distribution shift
is not established.

Full-residual interchange is behavioral evidence, not proof that one J-Lens
direction is the unique causal mechanism. The study also does not supply a
deployment cutoff, realistic-prevalence error rates, a runtime advantage, or a
faithful compact circuit. The WRITTEN-excluded causal tail remains unresolved,
and graded measurement remains unestablished.

If gradedness is revisited, it should be a new, preregistered program with an
independently authored weak-to-strong dependency order, untouched groups,
adequate group-aware power, and a numerical completeness gate fixed before
causal evaluation. It should not be another tuning cycle on the current score
and roster.

Detailed source records are preserved in [results/archive](results/archive/),
and the untouched historical sequence remains in
[experiments/README.md](experiments/README.md).

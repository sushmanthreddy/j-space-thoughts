# Visible but Idle: gradient-only detection of concept use

> A model can visibly represent a concept without using it for the answer it is
> producing. This project asks whether that difference can be detected without
> running a causal intervention every time.

The ending is deliberately mixed. On one frozen Qwen2.5-7B-Instruct
experiment, `READ_IG` works as a binary relevant-versus-idle screen. It does
not work as a graded meter of causal strength. The failed instruments and the
negative graded result are part of the result, not material hidden behind it.

## The question that started the project

Anthropic's Jacobian Lens (J-Lens) made it practical to surface a verbalizable
concept from an intermediate model activation, even when that concept never
appeared in the model's output. Anthropic's work also showed that some J-Lens
directions could be manipulated causally. That created the representational
foundation for this project—but also exposed a gap.

Seeing a concept is not the same as knowing that the current computation uses
it.

Neel Nanda had stated this methodological problem clearly in earlier work:

> “Probes on their own can mislead, and don't necessarily tell us that the
> model uses this representation.”

— [Neel Nanda, *Actually, Othello-GPT Has A Linear Emergent World
Representation*](https://www.neelnanda.io/mechanistic-interpretability/othello)

When reviewing Anthropic's global-workspace paper, he sharpened the same gap
for J-Lens:

> “J-Lens seems clearly useful as a hypothesis generation tool, but less useful
> for validating hypotheses.”

He explicitly asked for more evidence about reliability and false-positive
rates. [His public review](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper)
and the [Anthropic-hosted commentary, pp. 41–42](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf#page=42)
are the direct intellectual motivation for the question tested here.

The corresponding causal principle also appears in Neel's own TransformerLens
work:

> “Activation patching is about studying the counterfactual effect of a
> specific activation.”

— [Neel-authored TransformerLens source](https://github.com/TransformerLensOrg/TransformerLens/blob/649d3be19b0f7283fd81d06e4f94aef8cb6b2cfe/transformer_lens/patching.py#L61-L66)
([authorship record](https://github.com/TransformerLensOrg/TransformerLens/commit/649d3be19b0f7283fd81d06e4f94aef8cb6b2cfe))

Together, these ideas supplied the shape of the project: use a cheap signal to
generate a hypothesis about concept use, then judge that hypothesis against an
independent counterfactual intervention.

The question became:

> Once J-Lens says that a concept is visible, can a cheap, behavior-conditioned
> signal tell whether that concept is actually driving this answer?

This is inspiration, not authorship. Anthropic introduced J-Lens and J-Space.
Neel and his MATS scholars Camila Blank and Agam Bhatia independently
replicated several core J-Lens findings on Qwen3.6-27B, and his
representation-versus-use concern motivated this experiment. He did not design
`READ_IG`, author this repository, or endorse its results.

## The research did not begin with a working detector

### First, the causal ruler was broken

The first experiment appeared to produce a scientific result. Then its most
basic calibration failed. Replacing the represented concept *spider* with
*ant* should have moved a leg-count answer from `8` to `6`; the edited run
returned `4`.

That invalidated the apparent conclusion. A method that misses its known
positive control cannot say whether the hypothesis is wrong. The intervention
had to be repaired before any READ score could be interpreted.

### Then the repaired intervention was a sledgehammer

After correcting token surfaces, Jacobian conventions, and the intervention
configuration, the canonical swaps reproduced. But the edit changed every
prompt position across a broad layer band. It damaged unrelated behavior, so
an apparent “causal effect” could simply be general model disruption.

The next version restricted edits to positions where the source concept was
visible. Under that surgical measurement, the known swaps still worked and the
narration controls stayed causally quiet. This made the causal instrument more
credible—and exposed READ as the real bottleneck.

### Plausible READ scores still failed

The early scores each failed differently:

- Local attribution was nearly unrelated to its nonlinear intervention
  endpoint: Pearson `r=0.062`.
- A global downstream-capacity score called `0/8` causally quiet narration
  cases low-READ. It measured whether the network *could* respond to a
  direction, not whether this answer used it.
- A behavior-specific path score reached Spearman `rho=-0.077` on the
  known-answer cases, while the narration paths were non-estimable.

Those failures supplied the key design lesson: do not keep redefining READ
until something correlates. Instead, build matched tasks with verified labels,
freeze the cheap estimator before seeing evaluation causal results, and judge
it once.

## The final experiment

The final protocol uses an engine/dashboard comparison.

- In an **engine**, the explicit concept is needed for the answer—for example,
  the model must use a country concept to produce its capital.
- In a **dashboard**, the same concept and fact context remain visible, but the
  requested answer is unrelated.
- A harder dashboard also matches the engine's semantic answer type, so the
  detector cannot succeed only by separating arithmetic answers from cities or
  symbols.

The experiment then assigns four different jobs to four stages:

1. **WRITTEN — verify visibility.** The concept must be represented at the
   explicit concept token at L16, and the clean task answers must be correct.
2. **Causal truth — verify use.** Exchange the complete residual state between
   matched concepts in both directions. The signed, unclipped recovery score
   `C` records how much the answer follows that exchange.
3. **READ — freeze the cheap signal.** `READ_IG` integrates answer gradients
   along a prespecified path between paired J-Lens directions. It receives no
   donor state, evaluation `C`, or intervention output.
4. **Evaluation — join only afterward.** The causal and READ artifacts meet
   only after both are fixed, and whole dependency groups stay together during
   inference.

In the engine analogy, WRITTEN asks whether a light is visible on the
dashboard. READ asks whether the corresponding part is connected to the
behavior currently being produced.

“Gradient-only” and “cheap” mean donor-free and intervention-output-free in
this repository. No runtime, memory, throughput, or cost benchmark was run, so
the project does not claim a measured speedup.

## What the experiment found

The result is easiest to understand as five questions answered in order. Each
plot earns one inference; none should be stretched into the next one.

### 1. Did the intervention establish a real engine/dashboard difference?

![Causal intervention sanity check: engines show strong recovery while visible-but-idle dashboards remain near zero](results/figures/f1_causal_sanity.png)

**What to look at.** Swapping the complete concept-token residual state changes
the engine answer strongly: median signed `C` is `0.912714`. The same operation
barely moves the original dashboard answer: median signed `C` is `-0.002043`.
The engine/dashboard labels are constructed by the tasks and then supported by
this causal check; they are not created by thresholding `C`.

**What this earns.** The matched prompts behave like used versus idle concepts
under the declared L16 full-residual intervention.

**What it does not earn.** Full-residual interchange does not prove that the
J-Lens direction is the unique causal mechanism. The swapped state can contain
other correlated information.

### 2. Does the cheap score visibly separate the two classes?

![Raw READ-IG distributions: engines occupy a higher band than both idle-control families](results/figures/f5_read_ig_distributions.png)

**What to look at.** The vertical axis is logarithmic. The lowest engine score
is `0.034970`; the highest original-idle score is `0.023411`, and the highest
answer-matched-idle score is `0.021032`. No idle point overlaps the engine band
on this frozen roster.

**What this earns.** The successful AUC is visible in the raw observations; it
is not an artifact of one favorable threshold or a hidden transformation.

**What it does not earn.** The gap does not supply a deployment threshold. A
new model, concept family, or prompt distribution could move either band, and
deployment prevalence would change the score's operational predictive value.

### 3. Is integrated READ doing more than a local gradient or static wiring score?

![Held-out binary AUC comparison for READ-IG, READ-local, and static capacity](results/figures/f2_binary_auc_and_baseline.png)

**What to look at.** `READ_IG` reaches held-out ROC AUC `1.000000` with grouped
interval `[1.000000, 1.000000]`. `READ_local` reaches `0.914825`
`[0.863661, 0.967161]`; the behavior-independent static-capacity control is
`0.500000` `[0.500000, 0.500000]`.

**What this earns.** Static sensitivity to a concept direction is not enough to
explain the result. The integrated path achieves cleaner binary ranking than
the local score on this roster.

**What it does not earn.** The comparison is statistical, not a runtime
benchmark. `READ_IG` is donor-free, but no measured speed or cost advantage is
claimed.

### 4. Is the result only an arithmetic-versus-city shortcut?

![READ-IG AUC remains perfect under the harder answer-type-matched idle control](results/figures/f4_hard_dashboard_auc.png)

**What to look at.** Engine discrimination is `1.000000` against the original
arithmetic dashboards and remains `1.000000` against dashboards with the same
semantic answer type as the engines.

**What this earns.** Arithmetic answer type is not the sole explanation for
the binary separation.

**What it does not earn.** One harder negative set cannot rule out every
lexical, syntactic, difficulty, or distributional shortcut. More adversarial
controls remain useful future work.

### 5. Does READ measure degree of causal use within engines?

![Engine-only graded check: READ-IG does not positively rank normalized causal recovery](results/figures/f3_engine_only_graded_check.png)

**What to look at.** Within the 77 already-relevant engines, `READ_IG` has
Spearman `rho=-0.179110` with normalized full-residual `|C|`; the grouped
interval is `[-0.431377, 0.126014]`. The point estimate is negative and the
interval contains zero.

**What this earns.** No positive graded-use evidence. The plot is the reason
the project reports a binary detector rather than a causal-strength ruler.

**What it does not earn.** It does not prove that every future graded estimator
is impossible. The present roster is strong-only, and a separate diagnosis
found limited prospective sensitivity under a conservative power approximation
and poor numerical completeness. Those limitations motivate a new benchmark;
they do not rescue this estimator/target pairing.

A later audit found no formula, join, layer, token, sign, or padding bug that
rescues the graded result. READ follows the scale and endpoint of its own
direction-defined path; normalized full-residual `C` measures a different,
compressed quantity.

That leaves a clear boundary:

> **Used versus idle is supported here. How strongly used is not.**

| Question | Status | Meaning |
| --- | --- | --- |
| Can `READ_IG` distinguish an answer-relevant concept from a visible but idle one? | **Supported in this frozen setting** | The engine/dashboard ranking succeeds against both idle-control families. |
| Does `READ_IG` rank how strongly relevant concepts are used? | **Not supported; current pairing is a NO-GO** | The within-engine association is not positive, and no audited local code correction rescues it. |
| Did later verification gates remove weak visible engines? | **No, within the L16-visible population** | Every visible evaluation candidate passed the non-WRITTEN conditions. |
| Does the WRITTEN threshold itself exclude a weak-but-real causal tail? | **Open** | The below-threshold candidates have no causal measurement in this study. |
| Does the result generalize to other models, layers, tasks, or implicit concepts? | **Open** | No transfer claim was established. |

## Why the negative result matters

A pooled engine/dashboard correlation could have looked like a graded meter,
but it mostly restated the class boundary. Testing only within engines exposed
that mistake. The diagnosis also found that the retained engines occupy a
strong-only causal range and that the stored integrated gradients have poor
numerical completeness. Neither limitation turns the negative association into
positive graded evidence.

The selection audit narrowed one possible explanation: all 77 L16-visible
evaluation candidates passed the later verification conditions, so those gates
did not remove weak visible engines. The effect of the WRITTEN threshold itself
remains open because the 16 below-threshold candidates have no causal `C`.

The honest deliverable is therefore a binary screen in one controlled setting,
not a universal thought reader, causal-strength ruler, or deployment-ready
safety monitor.

## Read the full record

- [Research journey](RESEARCH_JOURNEY.md) — the complete failure-and-repair
  narrative, limitations, and open questions.
- [Results](RESULTS.md) — the compact numeric reference, graded diagnosis,
  selection audit, and final classification.
- [Consolidation notes](results/CONSOLIDATION_NOTES.md) — source inventory and
  number-consistency audit.
- [Detailed research archive](results/archive/) — the prior workshop draft,
  original results, frozen hypotheses, selection audit, analysis, and
  recommendation.
- [Experimental history](experiments/README.md) — untouched failed,
  exploratory, and superseded runs.
- [Machine-readable metrics](results/metrics.json) — populated final metrics.

## Attribution boundary

Anthropic introduced the Jacobian Lens and J-Space in
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
and released the
[Jacobian Lens reference implementation](https://github.com/anthropics/jacobian-lens).
This repository does not claim J-Lens, J-Space, the published lens weights, or
Anthropic's canonical swaps as original work.

Neel Nanda's role is intellectual motivation and independent review, not
project authorship. His earlier causal-validation principle and his explicit
request for J-Lens reliability and false-positive evidence shaped the research
question.

This project's contribution begins after concept detection: the `READ_IG`
estimator, matched engine/dashboard prompts, signed symmetric full-residual
validation, the anti-circularity firewall, and the report of both the successful
binary test and failed graded test.

## Repository map

```text
RESEARCH_JOURNEY.md       synthesized research narrative
RESULTS.md                synthesized numeric reference
src/                      final implementation
notebooks/                five-stage executable pipeline
results/
  metrics.json            machine-readable final metrics
  figures/                canonical result figures
  archive/                superseded detailed research records
  CONSOLIDATION_NOTES.md  inventory and consistency audit
experiments/              untouched historical runs and failure record
data/specs/               authored reciprocal prompt specification
```

The implementation modules separate model/lens loading, causal interchange,
cheap READ, dataset construction, grouped evaluation, and plotting. The five
notebooks follow the same order from frozen dataset construction through final
figures.

## Reproduction

The recorded run used the versions in `requirements.txt`, the pinned model and
published J-Lens weights, and a CUDA-capable environment. Intermediate model
artifacts are intentionally ignored under `artifacts/final/`.

Run the notebooks sequentially from the repository root:

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
cd "$HOME/j-space-thoughts"

for notebook in \
  notebooks/01_build_and_verify_dataset.ipynb \
  notebooks/02_causal_ground_truth.ipynb \
  notebooks/03_cheap_read.ipynb \
  notebooks/04_trust_check_and_stress_test.ipynb \
  notebooks/05_results_and_figures.ipynb
do
  .venv/bin/python -m jupyter nbconvert \
    --to notebook --execute --inplace "$notebook" \
    --ExecutePreprocessor.timeout=21600 \
    --ExecutePreprocessor.kernel_name=j-space-thoughts
done
```

Notebook 05 regenerates the computational metrics, figures, and historical
`results/RESULTS.md` report path. The pre-consolidation report is preserved at
[`results/archive/RESULTS.md`](results/archive/RESULTS.md); the curated root
[`RESULTS.md`](RESULTS.md) is the external-review reference.

## Anti-circularity firewall

`src/cheap_read.py` imports no causal, patching, or intervention module. The
cheap notebook consumes the sanitized manifest and direction cache, not the
evaluation causal artifact. Hard-control READ values are frozen before their
causal truth is computed. The final evaluation joins independent artifacts by
pair ID after both are fixed.

## License

Original contributions are provided under Apache-2.0 as described in
[`LICENSE`](LICENSE). Anthropic's separately distributed code and data retain
their own notices, and model weights remain subject to their respective
licenses.

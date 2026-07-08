# Visible but Idle: gradient-only detection of concept use

Anthropic's Jacobian Lens can surface a verbalizable concept in a language
model. This project asks the next question: **is the visible concept affecting
the current answer, or is it merely present and idle?**

The honest result is narrow. On one frozen Qwen2.5-7B-Instruct experiment,
`READ_IG` works as a binary relevant-versus-idle screen. It does not work as a
graded meter of causal strength. The repository reports both outcomes and keeps
the failed approaches and detailed audits available for review.

## Start here

- [Research journey](RESEARCH_JOURNEY.md) — the problem, failed instruments,
  working protocol, negative graded result, and open questions in one narrative.
- [Results](RESULTS.md) — the compact numeric reference, including the binary
  result, graded diagnosis, selection audit, and final classification.
- [Consolidation notes](results/CONSOLIDATION_NOTES.md) — source inventory and
  number-consistency audit.
- [Detailed research archive](results/archive/) — the prior workshop draft,
  original result report, frozen hypotheses, selection audit, analysis, and
  recommendation.
- [Experimental history](experiments/README.md) — untouched failed,
  exploratory, and superseded runs.
- [Machine-readable metrics](results/metrics.json) — populated final metrics.

## Status in plain language

| Question | Status | Meaning |
| --- | --- | --- |
| Can `READ_IG` distinguish an answer-relevant concept from a visible but idle one? | **Supported in this frozen setting** | The engine/dashboard ranking succeeds against both original and answer-type-matched idle controls. |
| Does `READ_IG` rank how strongly relevant concepts are used? | **Not supported; current pairing is a NO-GO** | The within-engine association is not positive, and the diagnosis found no local code correction that rescues it. |
| Did later verification gates remove weak visible engines? | **No, within the L16-visible population** | Every visible evaluation candidate passed the non-WRITTEN conditions. |
| Does the WRITTEN threshold itself exclude a weak-but-real causal tail? | **Open** | The below-threshold candidates have no causal measurement in this study. |
| Does the result generalize to other models, layers, tasks, or implicit concepts? | **Open** | No transfer claim was established. |

![Held-out binary AUC and baselines](results/figures/f2_binary_auc_and_baseline.png)

The figure shows ranking on the frozen roster, not universal accuracy or a
deployment-ready threshold. See [RESULTS.md](RESULTS.md) for the values and
grouped intervals.

## The idea

The matched tasks use an engine/dashboard comparison:

- An **engine** prompt contains an explicit concept that is needed for the
  answer.
- A **dashboard** retains the concept and fact context but asks for an unrelated
  answer. A harder dashboard also matches the engine's semantic answer type.
- Symmetric full-residual interchange supplies expensive causal truth `C`.
- `READ_IG` is frozen separately from that truth and uses answer gradients along
  a prespecified J-Lens direction path.

WRITTEN therefore asks whether the concept is visible. READ asks whether the
current answer is sensitive to it. Evaluation joins the causal and READ
artifacts only after both paths are fixed.

“Gradient-only” and “cheap” mean donor-free and intervention-output-free here.
No runtime, memory, throughput, or cost benchmark was run.

## Attribution boundary

Anthropic introduced the Jacobian Lens and J-Space in
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
and released the
[Jacobian Lens reference implementation](https://github.com/anthropics/jacobian-lens).
This repository uses that representational foundation and does not claim the
lens, J-Space, published lens weights, or canonical swaps as original work.

The contribution evaluated here begins after concept detection: the READ
estimator, matched idle controls, signed symmetric causal validation, and the
anti-circularity firewall.

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

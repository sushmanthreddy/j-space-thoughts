# Visible but Idle: gradient-only detection of concept use

Anthropic's Jacobian Lens can surface a verbalizable concept signal inside a language model. This project asks the next question: **is that concept affecting the current answer, or is it merely visible and idle?**

On a frozen Qwen2.5-7B-Instruct experiment, the gradient-only `READ_IG` score separated every verified relevant example from two families of idle controls, including answer-type-matched controls. It did **not** rank how strongly the relevant examples were used. The result supports a binary use-versus-idle detector in this setting, not a graded causal-use meter.

**Final verdict:** binary detector `SUPPORTED`; graded meter `NOT_SUPPORTED`; broader stress-test label `ARTIFACT (partial)`. Here, “partial” means the broader graded claim failed while the narrower binary result survived—it does not mean the run or repository is incomplete.

[Full results](results/RESULTS.md) · [Machine-readable metrics](results/metrics.json) · [Workshop paper](paper/workshop_paper.md) · [Research history](experiments/README.md)

## Verdict at a glance

| Question | Status | What the experiment says |
| --- | --- | --- |
| Can `READ_IG` distinguish a relevant concept from a visible-but-idle one? | **Supported in this setting** | Group-bootstrap ROC AUC 1.000 [1.000, 1.000] against both idle families, using 77 matched concept pairs across 24 dependency groups. |
| Does `READ_IG` measure how strongly a relevant concept is used? | **Not supported** | Within engines, Spearman rho is -0.179 with grouped 95% CI [-0.431, 0.126]. |
| Does the result transfer to other models, layers, tasks, languages, or implicit concepts? | **Open** | No valid transfer result was established in the final protocol. |
| Is there a deployment-ready score cutoff or known future false-positive rate? | **Open** | The study measures ranking by AUC; it does not calibrate an operating threshold. |
| Is `READ_IG` empirically faster than causal interchange? | **Not measured** | It is donor-free, but no runtime, memory, or throughput benchmark was run. |
| Did the archived exploratory work identify a small faithful circuit? | **Not supported** | Top-eight localization retained only 0.239–0.365 of the relevant effect. |

## The idea in everyday language

Imagine two prompts that contain the same fact and explicit concept:

- In the **engine**, the answer depends on that concept—for example, the model needs a country concept to produce its capital.
- In the **dashboard**, the concept is still visible in the prompt, but the requested answer is unrelated. A harder dashboard also matches the engine's answer type.

J-Lens can say “the concept is present” in both prompts. This project tests whether the answer would be sensitive to moving that concept representation toward a matched alternative.

The experiment has four stages:

1. **Find visibility.** Require the concept to be clearly represented at the explicit concept token. This is called **WRITTEN**.
2. **Establish causal truth.** Exchange the complete residual state—the model's full hidden-state vector at that token—between matched concepts in both directions. Engines should change; idle controls should not.
3. **Compute READ independently.** Integrate answer gradients along a direction-defined activation path from each clean state toward the paired concept. This produces `READ_IG` without reading evaluation causal scores or donor states.
4. **Evaluate after freezing.** Join the separate artifacts and test whether high READ scores belong to engines and low scores to idle controls.

The important distinction is:

> **WRITTEN asks whether a concept is visible. READ asks whether the current answer is sensitive to it.**

## What worked

### 1. The matched tasks passed the causal sanity check

Changing the complete concept-token residual state strongly changed the engine answer-logit preference but had almost no effect on idle answers.

| Class | Median signed `C` (grouped 95% CI) | Median absolute `C` (grouped 95% CI) |
| --- | ---: | ---: |
| Relevant engine | 0.912714 [0.896378, 0.929191] | 0.912714 [0.896378, 0.929191] |
| Original idle dashboard | -0.002043 [-0.003652, 0.002326] | 0.005083 [0.003587, 0.007752] |
| Answer-matched idle dashboard | 0.001235 [-0.001546, 0.005780] | 0.006466 [0.004013, 0.010064] |

![Causal effects distinguish relevant engines from original idle controls](results/figures/f1_causal_sanity.png)

*The figure shows engines and original dashboards. The answer-matched dashboards were checked separately and were also causally idle.*

### 2. `READ_IG` separated relevant from idle concepts

| Estimator and comparison | Held-out ROC AUC | Grouped 95% CI |
| --- | ---: | ---: |
| `READ_IG`, engine vs original idle | 1.000000 | [1.000000, 1.000000] |
| `READ_IG`, engine vs answer-matched idle | 1.000000 | [1.000000, 1.000000] |
| `READ_local`, engine vs original idle | 0.914825 | [0.863661, 0.967161] |
| Static capacity control, engine vs original idle | 0.500000 | [0.500000, 0.500000] |

The labels are the constructed engine and idle tasks. Causal interchange validates those classes; it does not create them by thresholding `C`.

The engine scores ranged from 0.034970 to 0.992770. Original idle scores ranged from 0.000881 to 0.023411, and answer-matched idle scores ranged from 0.001650 to 0.021032. No idle example outranked an engine in this frozen roster.

![READ_IG separates relevant and idle concepts in this evaluation](results/figures/f5_read_ig_distributions.png)

*Every verified engine scores above both idle-control families on this frozen roster; the vertical axis is logarithmic.*

This is **perfect empirical ranking on this dataset**, not “100% accuracy” on future prompts. The implementation returns a continuous score; no deployment threshold or universal false-positive rate has been established.

### 3. The harder control survived

The original idle prompts used arithmetic answers, which could have been an easy shortcut. The harder controls instead use the same semantic answer class as the engines while keeping the source concept irrelevant. `READ_IG` still reached AUC 1.000 [1.000, 1.000], so arithmetic answer type is not the sole explanation.

### 4. The result is protected against causal-label leakage

The cheap path cannot read evaluation causal results:

- `src/cheap_read.py` imports no causal, patching, or interchange module and performs no artifact file I/O.
- Notebook 03's scientific input artifacts are the sanitized manifest and frozen direction cache; it reads no causal artifact.
- Hard-control READ scores are frozen before hard-control causal truth is computed.
- Evaluation joins the separate artifacts only after both are fixed.

Calibration-group causal separation is allowed to choose one global layer. Evaluation-group causal scores cannot choose an example's sign, transformation, exclusion, or READ value.

## What did not work

### `READ_IG` did not rank causal strength within engines

Among the 77 already-causal engines, `READ_IG` had Spearman rho -0.179110 with `|C|`, with grouped 95% CI [-0.431377, 0.126014]. The interval includes zero and the point estimate is negative, so the experiment provides no positive graded-use evidence.

![READ_IG does not rank causal strength within relevant engines](results/figures/f3_engine_only_graded_check.png)

*Within engines, the confidence interval spans zero, so READ does not provide positive graded-use evidence.*

The pooled rho of 0.707412 mainly restates the gap between engines and controls. It must not be interpreted as evidence that READ measures degree of use. No weak/strong threshold was invented after seeing the data.

Earlier experiments also rejected several plausible ideas: a broken positive-control instrument, an overly broad intervention that damaged unrelated behavior, earlier READ definitions with near-zero or backward results, and an unsuccessful compact-circuit localization. They remain documented in the [experiment archive](experiments/README.md).

## What needs more research

The binary result is promising but narrow. The next study should test:

- **A wider causal range:** preregister weak, medium, and strong engines so graded measurement receives a fair test.
- **Transfer:** evaluate other models, sizes, layers, relations, languages, and prompt styles.
- **Harder concepts:** include implicit, multi-token, ambiguous, and naturally occurring concepts.
- **More adversarial controls:** match wording, syntax, answer type, difficulty, and distribution more tightly.
- **Deployment calibration:** select thresholds on calibration data and report sensitivity, specificity, precision, recall, and false-alarm rates under realistic prevalence.
- **Efficiency:** benchmark runtime, memory, throughput, and cost against full-residual interchange.
- **Mechanistic specificity:** determine whether the J-Lens direction itself is causal rather than merely correlated with information carried by the full residual state.

The current result should therefore be described as:

> A narrow, firewalled relevant-versus-idle screening score on one frozen model and structured dataset—not a general thought reader, causal-strength meter, or deployment-ready safety monitor.

## Scope and protocol

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, bf16.
- Lens: published Qwen2.5-7B J-Lens, loaded through Anthropic's official package.
- Measurement: explicit single concept token at calibration-selected L16; WRITTEN threshold 2.482431.
- Data: 118 reciprocal candidates across element symbols, country capitals, and US-state capitals; 25 calibration, 93 evaluation, 77 verified and 16 unverified.
- Inference: 24 held-out dependency groups, five whole-group folds, and 10,000 whole-group bootstrap draws with seed 1729.
- `READ_IG`: 16 gradient-bearing midpoint evaluations per direction. No runtime comparison was performed.

The ROC labels are the constructed engine and idle task classes. They are not created by thresholding `C`; separately produced causal interchange validates the intended task contrast.

## Attribution boundary

Anthropic introduced the Jacobian Lens and J-Space in [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) and released the [Apache-2.0 reference implementation](https://github.com/anthropics/jacobian-lens). This project reproduces that foundation and does not claim J-Lens, J-Space, the published lens, or Anthropic's canonical swap experiments as original work.

This project's contribution begins after detection:

1. distinguish WRITTEN concept visibility from READ behavior sensitivity;
2. define the donor-free `READ_IG` estimator;
3. add matched idle controls and an anti-circularity firewall; and
4. report the successful binary result together with the failed graded result.

## Repository map

```text
src/
  jlens_interface.py   pinned model/lens loading, J-Lens directions, WRITTEN
  causal_read.py       expensive signed symmetric full-residual truth C
  cheap_read.py        READ_IG, READ_local, static capacity control
  datasets.py          matched prompts, gates, answer-matched hard controls
  evaluation.py        grouped AUC/correlation/bootstrap and provenance gate
  plotting.py          six pure final figure builders
notebooks/
  01_build_and_verify_dataset.ipynb
  02_causal_ground_truth.ipynb
  03_cheap_read.ipynb
  04_trust_check_and_stress_test.ipynb
  05_results_and_figures.ipynb
results/               canonical metrics, report, figures, pre/post provenance
paper/                 workshop paper and its figure copies
experiments/           untouched failed, exploratory, and superseded runs
data/specs/            authored reciprocal prompt specification
```

[`experiments/README.md`](experiments/README.md) gives the research arc: the broken positive-control instrument, the intervention that damaged unrelated behavior, failed READ definitions, the alpha sweep, the corrected symmetric protocol, the v6 stress test, scale/ambiguity work, and the unsuccessful compact-circuit localization. These failures are retained rather than rewritten.

## Exact reproduction

The recorded run used an NVIDIA H200 and the versions pinned in `requirements.txt`. The notebooks are sequential and write ignored intermediate artifacts under `artifacts/final/`; notebook 05 writes the tracked final results.

<details>
<summary>Full pinned setup and execution commands</summary>

### 1. Environment and upstream dependency

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_CACHE=/home/jovyan/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/home/jovyan/.cache/huggingface/hub

cd "$HOME/j-space-thoughts"
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

git clone https://github.com/anthropics/jacobian-lens "$HOME/deps/jacobian-lens"
git -C "$HOME/deps/jacobian-lens" checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
.venv/bin/pip install -e "$HOME/deps/jacobian-lens"

.venv/bin/python -m ipykernel install --user \
  --name j-space-thoughts --display-name j-space-thoughts
```

If the upstream checkout already exists, verify its commit instead of cloning over it.

### 2. Download the pinned model and published lens

```bash
hf auth whoami
hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28
hf download neuronpedia/jacobian-lens \
  qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt \
  --revision 16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a
```

The code uses `local_files_only=True`; a missing pinned download fails rather than silently resolving another revision.

### 3. Preflight

```bash
command -v hf
hf auth whoami
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
.venv/bin/python -c "import torch; print(torch.__version__)"
```

Notebook 01 repeats these checks and enforces a 20-prompt HF/J-Lens wrapper gate with maximum mean KL below `1e-3`.

### 4. Execute the pipeline

```bash
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

</details>

The stages produce:

1. a sanitized dataset manifest and frozen directions;
2. base engine/original-control causal truth;
3. base and hard-control READ values without causal access;
4. hard-control causal truth plus all grouped evaluation; and
5. canonical metrics, figures, `RESULTS.md`, and the post-refactor provenance comparison.

## Anti-circularity firewall

`src/cheap_read.py` imports no causal, patching, or intervention module. Notebook 03's scientific input artifacts are the sanitized clean manifest and frozen direction cache; it does not read notebook 02's causal artifact. It freezes hard-control READ values before notebook 04 computes hard-control `C`. Notebook 03 also reads source text for its static import/hash audit and loads the pinned model and lens. The final release records a source-level firewall check.

To inspect the boundary directly:

```bash
grep -nE '^(from|import).*(causal_read|intervention|patching)' src/cheap_read.py
```

The expected result is no import. Mentions in documentation or audit strings are not computational dependencies.

## Refactor provenance

`results/PROVENANCE_pre_refactor.json` freezes the original headline values. A clean rerun creates `results/PROVENANCE_post_refactor.json`; notebook 05 compares 80 scientific leaf fields using

```text
max(absolute tolerance 1e-3, relative tolerance 1e-3 × |pre|)
```

Counts, identifiers, booleans, and decisions must match exactly. Any out-of-tolerance difference stops notebook 05. The machine-readable comparison is `results/PROVENANCE_comparison.json`.

## License

Original contributions in this repository are provided under Apache-2.0 as described in [`LICENSE`](LICENSE). Anthropic's separately distributed `jacobian-lens` code and synthetic data retain their own Apache-2.0 attribution and notices. Model weights and other downloads remain subject to their respective licenses.

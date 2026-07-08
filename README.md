# Visible but Idle: gradient-only detection of concept use

This repository asks a narrow question after a concept has been found inside a language model: is that concept actually being used for the current answer, or is it merely visible and idle? We reproduce Anthropic's Jacobian Lens (J-Lens) readout on pinned Qwen2.5-7B-Instruct and add a firewalled gradient-only estimator, `READ_IG`, which is validated against expensive symmetric full-residual interchange. On 77 verified held-out matched pairs, `READ_IG` separates relevant engines from both original and answer-type-matched idle controls with ROC AUC 1.000 (group-bootstrap 95% CI [1.000, 1.000]). It does **not** rank causal magnitude within engines (Spearman rho -0.179, CI [-0.431, 0.126]). The supported claim is therefore a binary relevant-versus-idle detector in this setting, not a graded causal-use meter.

The complete numerical report is in [`results/RESULTS.md`](results/RESULTS.md), and the workshop paper is in [`paper/workshop_paper.md`](paper/workshop_paper.md).

## Attribution boundary

Anthropic introduced the Jacobian Lens and J-Space in [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) and released the [Apache-2.0 reference implementation](https://github.com/anthropics/jacobian-lens). This project reproduces that detection machinery; it does not claim J-Lens, J-Space, the published lens, or Anthropic's canonical swap experiments as original work.

Our contribution begins after detection:

1. distinguish **WRITTEN** (the selected activation projects onto a J-Lens concept direction) from **READ** (the selected behavior is sensitive to that concept);
2. define a donor-free, gradient-only `READ_IG` score;
3. validate it against independently produced causal truth behind an anti-circularity firewall; and
4. report both the successful binary result and the failed graded-use result.

“Gradient-only” and “donor-free” describe the computation. No wall-clock comparison was run, so the repository does not claim a measured speedup over causal interchange.

## Final scope and result

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, bf16.
- Lens: published Qwen2.5-7B J-Lens, loaded through Anthropic's official package.
- Position/layer: the explicit single concept token at calibration-selected L16.
- Data: 118 reciprocal candidates; 25 calibration, 93 held out, 77 verified in 24 dependency groups.
- Causal sanity: median engine `C=0.912714`; original idle median `|C|=0.005083`; answer-matched idle median `|C|=0.006466`.
- Binary result: engine versus either idle family `READ_IG` AUC 1.000, CI [1.000, 1.000].
- Secondary controls: `READ_local` AUC 0.914825 [0.863661, 0.967161]; static capacity AUC 0.500.
- Graded stress test: engine-only rho -0.179110 [-0.431377, 0.126014], so graded use is not supported.
- Honest final label: `ARTIFACT (partial)`—the broader graded interpretation fails while the narrower binary detector survives.

The AUC labels are the constructed relevant-engine and idle-control task classes. Causal interchange validates that construction; AUC is not computed from labels made by thresholding `C`.

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

The stages produce:

1. a sanitized dataset manifest and frozen directions;
2. base engine/original-control causal truth;
3. base and hard-control READ values without causal access;
4. hard-control causal truth plus all grouped evaluation; and
5. canonical metrics, figures, `RESULTS.md`, and the post-refactor provenance comparison.

## Anti-circularity firewall

`src/cheap_read.py` imports no causal, patching, or intervention module. Notebook 03 reads only the sanitized clean manifest and the frozen direction cache; it does not read notebook 02's causal artifact. It freezes hard-control READ values before notebook 04 computes hard-control `C`. Notebook 03 performs a static import audit, and the final release also records a source-level firewall check.

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

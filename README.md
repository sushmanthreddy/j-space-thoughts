# Written vs. Read

This repository tests when a concept visible in a Jacobian-lens readout is
actually load-bearing for a language model's behavior. The preregistered
mechanistic distinction is:

- **WRITE**: projection of a clean post-block residual onto a unit concept
  direction.
- **READ**: downstream sensitivity to that direction, estimated independently
  by activation attribution and component weights.
- **CAUSAL**: the measured behavior-logit change under a real residual-stream
  ablation or concept/foil coordinate swap.

The signed convention is always `delta = M_edited - M_clean`, where
`M = logit(target) - logit(foil)`. Thus the first-order ablation prediction is
`-sum(WRITE * READ)`. Positive-is-damage plots are explicitly labelled
`M_clean - M_edited` and never silently reverse this convention.

The research is executed in notebooks, with reusable implementation and tests
in `src/` and `tests/`. Generated model weights, fitted lenses, and caches are
not versioned. Curated raw per-item measurements and figures are written under
`results/`.

## Environment setup

The recorded runs use Python 3.11, PyTorch 2.5.1+cu124, Transformers 5.13.0,
and an NVIDIA H200. The two environment overrides below matter on the supplied
pod: it exports `PIP_USER=true`, and its unrelated global user packages are not
compatible with Transformers 5.

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
export PIP_USER=false
export PYTHONNOUSERSITE=1

mkdir -p "$HOME/deps"
test -d "$HOME/deps/jacobian-lens/.git" || \
  git clone https://github.com/anthropics/jacobian-lens.git "$HOME/deps/jacobian-lens"
git -C "$HOME/deps/jacobian-lens" checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e

cd "$HOME/j-space-thoughts"
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e "$HOME/deps/jacobian-lens"
.venv/bin/python -m ipykernel install --user \
  --name j-space-thoughts --display-name "Python (j-space-thoughts)"
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q -p no:cacheprovider tests
.venv/bin/python -m pytest -q -p no:cacheprovider "$HOME/deps/jacobian-lens/tests"
```

Do not run the dependency repository's `uv sync`: its lock currently selects a
different PyTorch/CUDA stack. Do not clear `~/.cache/huggingface` while a scale
is still needed.

## Pinned downloads

Download a model explicitly before the notebook that uses it:

```bash
hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28
hf download neuronpedia/jacobian-lens \
  qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt \
  --revision 16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a

# Run only after the 7B correctness gates pass.
hf download Qwen/Qwen2.5-14B-Instruct \
  --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
```

The 7B and 14B weights require about 41.7 GiB together. The optional 32B model
would raise weights alone to about 102.8 GiB, exceeding the pod's 100 GB home
volume, so it is skipped unless storage is expanded.

## Exact run order

Run notebooks from the repository root so relative output paths are stable:

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
export PIP_USER=false
export PYTHONNOUSERSITE=1
cd "$HOME/j-space-thoughts"

for notebook in \
  notebooks/_jlens_walkthrough_qwen25_7b.ipynb \
  notebooks/00_setup_and_gates.ipynb \
  notebooks/01_lens_and_concept_vectors.ipynb \
  notebooks/02_twohop_core.ipynb \
  notebooks/03_controls.ipynb \
  notebooks/04_read_localization.ipynb \
  notebooks/05_ambiguity_flagship.ipynb \
  notebooks/06_scale_comparison.ipynb \
  notebooks/08_report.ipynb
do
  jupyter nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.kernel_name=j-space-thoughts \
    --ExecutePreprocessor.timeout=14400 "$notebook"
done
```

Notebook 07 is optional and is run only if the documented blackmail-action
base-rate gate reaches 15%. Each required notebook prints a short PASS/FAIL
summary and writes its raw measurements before later notebooks consume them.

## Correctness and integrity gates

Downstream claims are not trusted until notebook 00 records:

1. HF-vs-J-Lens-wrapper KL below `1e-3` on 20 prompts.
2. The known spider-to-eight case, including a spider/ant coordinate swap.
3. Pearson correlation between attribution-predicted and real ablation effects
   on a held-out two-hop validation set.

The output-token suppression control is reported for every concept. A residual
ablation effect comparable to output suppression is treated as unproven rather
than as evidence of internal causal use. Null directions, absent concepts,
capability damage, the known language-narration case, and logit-lens baselines
are retained in the final metrics even when they refute the hypothesis.

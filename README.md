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

## Result

The preregistered hypothesis is **not supported** in these runs. Both 7B and
14B pass the logit-equivalence gate but fail the strict known-answer workspace
gate; P1 is unsupported, P2 is unestablished, and the diagnostic ambiguity P3
is refuted. P4 was optional and was not run. See the numerical tables, CIs,
controls, and limitations in [results/RESULTS.md](results/RESULTS.md).

The research is executed in notebooks, with reusable implementation and tests
in `src/` and `tests/`. Full raw per-item JSON is written to ignored
`data/raw/`; fitted lenses and direction tensors are written to ignored
`data/lenses/` and `data/directions/`. Versioned `results/` contains the
curated per-item scalar measurements and correlations in `metrics.json`, the
figures, the compact scale comparison, and the report. A fresh clone does not
contain the ignored intermediates, so run the notebooks in the order below.

## Environment setup

The recorded runs use Python 3.11, PyTorch 2.5.1+cu124, Transformers 5.13.0,
and an NVIDIA H200. The two environment overrides below matter on the supplied
pod: it exports `PIP_USER=true`, and its unrelated global user packages are not
compatible with Transformers 5.

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
export PIP_USER=false
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export JUPYTER_PATH="$HOME/.local/share/jupyter"

mkdir -p "$HOME/deps"
test -d "$HOME/deps/jacobian-lens/.git" || \
  git clone https://github.com/anthropics/jacobian-lens.git "$HOME/deps/jacobian-lens"
git -C "$HOME/deps/jacobian-lens" checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e

cd "$HOME/j-space-thoughts"
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt -c constraints-recorded.txt
.venv/bin/python -m pip install --no-deps -e "$HOME/deps/jacobian-lens"
.venv/bin/python -m ipykernel install --user \
  --name j-space-thoughts --display-name "Python (j-space-thoughts)"
.venv/bin/python -m pytest -q -p no:cacheprovider tests
.venv/bin/python -m pytest -q -p no:cacheprovider "$HOME/deps/jacobian-lens/tests"
```

`constraints-recorded.txt` pins the package versions used for the recorded
numbers. The supplied pod already provides PyTorch 2.5.1+cu124; on a fresh
CUDA 12.4 environment, install that PyTorch build before applying the
constraints. Verify the core runtime with:

```bash
.venv/bin/python - <<'PY'
import sys, torch, transformers
print(sys.version)
print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("transformers", transformers.__version__)
PY
git -C "$HOME/deps/jacobian-lens" rev-parse HEAD
```

Because this venv exposes the supplied pod's system packages, `pip check` also
sees an unrelated preinstalled `emotion-vectors` package and reports its stale
NumPy/SciPy/Transformers constraints. This project does not import that
package; the exact version check, project tests, and upstream J-Lens tests are
the relevant environment checks.

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

# Run after notebook 00 records the 7B gates. If strict G2 fails, all later
# scale results must remain explicitly diagnostic, as in the recorded run.
hf download Qwen/Qwen2.5-14B-Instruct \
  --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
```

The 7B and 14B weights require about 41.7 GiB together. The optional 32B model
would raise weights alone to about 102.8 GiB, exceeding the pod's 100 GB home
volume, so it is skipped unless storage is expanded.

If the ignored 14B lens is absent, notebook 06 fits it from the first 100
qualifying WikiText training records at pinned revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Expect roughly 1.22 GiB for the
final lens, 2.44 GiB for its resumable checkpoint, and about 80 MiB for the 14B
mean-difference directions under `data/`. The fit uses source layers 19–43 and
writes prompt hashes and provenance sidecars before the scale analysis.

## Exact run order

Run the reference walkthrough from its ignored artifact directory, then run
the research notebooks from the repository root so relative output paths are
stable:

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
export PIP_USER=false
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export JUPYTER_PATH="$HOME/.local/share/jupyter"
ROOT="$HOME/j-space-thoughts"

mkdir -p "$ROOT/artifacts/walkthrough"
(
  cd "$ROOT/artifacts/walkthrough"
  "$ROOT/.venv/bin/python" -m jupyter nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.kernel_name=j-space-thoughts \
    --ExecutePreprocessor.timeout=14400 \
    "$ROOT/notebooks/_jlens_walkthrough_qwen25_7b.ipynb"
)

cd "$ROOT"

for notebook in \
  notebooks/00_setup_and_gates.ipynb \
  notebooks/01_lens_and_concept_vectors.ipynb \
  notebooks/02_twohop_core.ipynb \
  notebooks/03_controls.ipynb \
  notebooks/04_read_localization.ipynb \
  notebooks/05_ambiguity_flagship.ipynb \
  notebooks/06_scale_comparison.ipynb \
  notebooks/08_report.ipynb
do
  .venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.kernel_name=j-space-thoughts \
    --ExecutePreprocessor.timeout=14400 "$notebook"
done
```

Notebook 07/P4 was not run and no notebook 07 is included; it is optional and
is recorded as `NOT_RUN_OPTIONAL`. Each required notebook prints a short
status or verdict summary and writes raw measurements before later notebooks
consume them. A strict G2 failure is recorded rather than bypassed: subsequent
results at that scale are diagnostic.

After notebook 08 reports `REPORT COMPLETENESS PASS`, rerun the local checks:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests
.venv/bin/python -m pytest -q -p no:cacheprovider "$HOME/deps/jacobian-lens/tests"
.venv/bin/python -m ruff check src tests
```

## Correctness and integrity gates

Downstream claims are not trusted until notebook 00 records:

1. HF-vs-J-Lens-wrapper KL below `1e-3` on 20 prompts.
2. The known spider-to-eight case, including a spider/ant coordinate swap.
3. Pearson correlation between attribution-predicted and real ablation effects
   on a held-out two-hop validation set.

The output-token suppression control is reported for every concept. Here its
exact zero is structural because the concept-token logit is disjoint from the
target-minus-foil behavior logits, so it is an instrumentation/direct-logit
steering check rather than independent causal evidence. Null directions,
absent concepts, capability damage, the known language-narration case, and
logit-lens baselines are retained in the final metrics even when they refute
the hypothesis.

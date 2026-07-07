# Written vs. Read — behavior-specific READ one-shot

## Result

**V4 is complete with a READ-operationalization methods limitation and no hypothesis verdict.** The single permitted behavior-specific estimator failed G-READVAL: known-answer `rho=-0.077` (cluster-bootstrap 95% CI `[-0.518, 0.409]`, N=21) and narration separation `0/8`. P1–P3 were not run.

The working evidence retained from prior repairs is real but narrower: canonical swaps pass 3/3, concept retrieval passes, firing controls work, and narration causal changes remain low 8/8. Masked capability rows were `NO_EDIT_OPPORTUNITY`, not active-edit preservation.

See [the full methods report](results/RESULTS.md), [notebook 10](notebooks/10_behavior_specific_read.ipynb), [notebook 11](notebooks/11_readval_gate.ipynb), and [notebook 14](notebooks/14_report.ipynb).

## One-shot method

Exactly one new estimator was added in `src/read_scores.py`: exact path-patch thresholding followed by the inherited random-normalized weight READ restricted to `S_M`. The threshold was fixed at `|delta M|>=0.05`; no alpha or threshold sweep and no estimator fallback was used.

## Notebook chain

1. `10_behavior_specific_read.ipynb` — builds exact path sets and global/restricted READ.
2. `11_readval_gate.ipynb` — applies the hard known-answer and narration gates.
3. `12_science_twohop.ipynb` and `13_science_ambiguity.ipynb` — executed model-free skips because G-READVAL failed.
4. `14_report.ipynb` — Road-B methods-limitation paper.

## Reproduction

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$HOME/.cache/huggingface/hub"
cd "$HOME/j-space-thoughts"
.venv/bin/python -m pytest -q
```

Model weights and full raw data remain ignored. Executed notebooks, compact metrics, figures, and the report are committed.

# Symmetric causal READ

## Result

**GO on Qwen2.5-7B-Instruct:** the gradient-only `READ_IG` estimator predicts
held-out causal use with ROC AUC `1.000` and unordered-concept-group bootstrap
95% CI `[1.000, 1.000]`. `READ_local` reaches AUC `0.915` (`[0.864, 0.967]`),
while the known-broken static MLP capacity baseline remains at chance (`0.500`).

The expensive truth is signed, symmetric, one-token full-residual interchange.
Across 77 verified held-out prompt pairs, median engine `C=0.913` and median
dashboard `|C|=0.0051`; no pair triggered the pre-registered directional-
disagreement flag. The cheap estimator never imports interchange code or reads
causal outputs.

This validates the estimator only for explicitly written concepts in
`Qwen/Qwen2.5-7B-Instruct`. Earlier latent-context attempts did not yield a
sound one-position causal instrument and are retained as rejected raw artifacts.
GO-only localization found large signed component mediation, but top-8 circuits
preserved only `0.239–0.365` of the effect under outside-component zero
ablation, so no faithful compact-circuit claim is made.

See [the full report](results/RESULTS.md) and
[the machine-readable metrics](results/metrics.json).

## Separation of truth and predictor

- `src/causal_read.py` owns full-residual/subspace interchange C and GO-only
  signed mediation.
- `src/cheap_read.py` owns clean-forward gradients, 16-step midpoint IG,
  local sensitivity, and the labelled static capacity baseline. It imports no
  causal or patching module.
- `src/data_gen.py` builds 118 distinct natural reciprocal prompt pairs,
  leakage-safe concept-group splits, exact token contracts, and task-matched
  engine/dashboard controls.

## Executed notebook chain

1. `30_dataset_and_verification.ipynb` — preflight, 20-prompt KL gate,
   calibration-only instrument selection, and VERIFIED/UNVERIFIED logging.
2. `31_causal_ground_truth.ipynb` — signed symmetric full-residual C and the
   diagnostic two-concept J-Lens subspace variant.
3. `32_cheap_read.ipynb` — isolated `READ_IG`, `READ_local`, and capacity
   baseline.
4. `33_trust_check.ipynb` — five grouped held-out folds, 10,000-draw grouped
   bootstrap, F1–F4, and the GO decision.
5. `34_localization.ipynb` — GO-only signed mediation and faithfulness (F5).
6. `35_report.ipynb` — artifact-by-artifact completion audit.

## Reproduction

```bash
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_CACHE=/home/jovyan/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/home/jovyan/.cache/huggingface/hub
cd "$HOME/j-space-thoughts"

for notebook in \
  notebooks/30_dataset_and_verification.ipynb \
  notebooks/31_causal_ground_truth.ipynb \
  notebooks/32_cheap_read.ipynb \
  notebooks/33_trust_check.ipynb \
  notebooks/34_localization.ipynb \
  notebooks/35_report.ipynb
do
  .venv/bin/python -m jupyter nbconvert \
    --to notebook --execute --inplace "$notebook" \
    --ExecutePreprocessor.timeout=14400 \
    --ExecutePreprocessor.kernel_name=j-space-thoughts
done
```

The protocol explicitly forbids running the test suite, Ruff, or pytest for
this experimental run. Model weights and full raw artifacts remain gitignored;
executed notebooks, compact metrics, figures, and reports are committed.

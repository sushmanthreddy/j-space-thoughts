# Data layout

Only compact, authored dataset specifications in `data/specs/` are versioned.
Generated prompts, fitted Jacobian lenses, cached activations, and raw model
outputs are reproducible artifacts and are ignored by Git. Curated per-item
measurements used by the report are stored in `results/metrics.json`.


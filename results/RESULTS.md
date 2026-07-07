# Behavior-specific READ validation report (v4)

## Current status

**ONE ESTIMATOR EXECUTED; G-READVAL PENDING. SCIENCE PROHIBITED.**

## Preflight and fixed protocol

- GPU: NVIDIA H200; total VRAM 143771 MiB; free 143072 MiB.
- Home/HF filesystem free: 37.9 GiB.
- Model: `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28` in `torch.bfloat16`.
- HF/J-Lens max mean KL: 1.660e-08 (N=20, threshold 1e-3): **PASS**.
- New READ estimators added: **1**. No alpha resweep was run.
- Locked known-answer roster: N=21 (prior v2 20 plus spider).
- Fixed causal endpoint: masked fractional source-to-foil swap, alpha=1.5, L13-24.
- Path discovery: source-only unit deletion at a clean minimum-rank layer.
- Exact path threshold: `|patched delta M| >= 0.05`; no top-k/fallback.
- Clean-to-clean maximum component patch: 0.000e+00.
- Raw artifact: `data/raw/v4/10_behavior_specific_read.json` (SHA-256 `77f129bf5f5366815e51819185f621e950e72770471b05627a605a657d06ff03`).

## Notebook 10 — path-restricted READ built

- Known-answer estimable rows: 21/21.
- Known-answer |S_M| range: 2–153.
- Narration auto |S_M| range: 0–0.
- Narration direct |S_M| range: 21–204.

Notebook 11 must now apply the frozen G-READVAL bars. No hypothesis science has run.

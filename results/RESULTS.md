# Surgical intervention calibration report (v3)

## Current verdict

**V3 CALIBRATION IN PROGRESS; SCIENCE PROHIBITED.** V2 established a working
three-case swap but failed capability and G-POS at alpha=2. V3 will sweep alpha
and carrying-position edits before any hypothesis test.

## Environment

- GPU: NVIDIA H200; 143771 MiB total; 143072 MiB free at preflight.
- Home/HF-cache filesystem: 100.0 GiB total; 38.1 GiB free.
- Required tool/auth preflight: **PASS**.
- Model: `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28` in `torch.bfloat16`.

## Stage 0 — v2 instrument re-verification

- HF/J-Lens logit gate: **PASS**; max mean KL=1.660e-08, N=20.
- Known-answer alpha-2 swaps: **PASS** (3/3).
- Cached held-out G-DIR artifact: **PASS**; retrieval top-1=0.550, known-answer top-5=0.8875.
- Non-structural direct suppression controls: **PASS**.

Stage-0 decision: **PASS**. This licenses only G-SWAP confirmation
and the alpha sweep; it does not license Stage-2 recalibration or Stage-3 science.

# Repair-first replication report (v2)

## Current verdict

**INSTRUMENT NOT YET VALIDATED — NO SCIENCE VERDICT.** The earlier
`NOT SUPPORTED` / `REFUTED` labels are withdrawn as scientific conclusions.
They were computed downstream of failed calibration gates and remain only as
legacy diagnostics.

## Environment

- GPU: NVIDIA H200; 143771 MiB total; 143072 MiB free at preflight.
- Home/HF-cache filesystem: 100.0 GiB total; 38.1 GiB free at preflight.
- Required tool/auth preflight: **PASS**.

## Stage 0 — upstream diagnosis

- Pinned dependency: `581d398613e5602a5af361e1c34d3a92ea82ba8e`; clean checkout=True.
- Walkthrough SHA-256: `96ba7c7945f0902e6cdacd32320309176dcfd891b571c2734a1aa60facfc5d4a`.
- Unchanged released readout cells 1/3/5/7: **PASS** on `Qwen/Qwen3.5-4B` at layers [8, 16, 24, 30] and position `-2`.
- The released walkthrough performs model/lens loading and readout only. It never changes an activation or runs a swapped continuation.
- `data/experiments/probe-swap.json` contains the spider→ant prompt metadata, but the dependency explicitly describes the JSON files as prompts only.
- Executable upstream swap/ablation helper: **NOT RELEASED**.

### Stage-0 decision

`UPSTREAM_CAUSAL_SWAP_NOT_RUNNABLE_RELEASE_OMISSION`. The requested unchanged canonical swap is not
runnable from the public release, so Stage 0 cannot distinguish a local code
bug from a Qwen model mismatch. This is not evidence that Qwen failed the
method. The strict G-SWAP state is **UNTESTED** pending an
honest repair/calibration attempt in our implementation.

![F0 Stage-0 audit](figures/f0_stage0_upstream_audit.png)

## Gate ledger

| gate | status | consequence |
| --- | --- | --- |
| Stage-0 preflight | PASS | Environment usable |
| Upstream readout | PASS | Readout compatibility only |
| Unchanged upstream causal swap | NOT RUNNABLE | Release omission; no code-vs-model inference |
| G-SWAP | UNTESTED | Stage 2 and Stage 3 remain prohibited |
| G-DIR | NOT RUN IN V2 | Blocked behind G-SWAP |
| G-POS / firing controls | NOT RUN IN V2 | Blocked behind G-SWAP and recalibration |

## Interpretation

What is established so far is narrow: the released J-Lens can load and return
readouts on its demonstrated open model. What is not established is a causal
coordinate swap, a calibrated Qwen workspace band, or the truth or falsity of
the WRITE-versus-READ hypothesis.

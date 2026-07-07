# READ Go/No-Go validation — ground truth complete

## Preflight

- GPU: NVIDIA H200; 143771 MiB total, 143072 MiB free.
- HF filesystem free: 38 GiB.
- Qwen3 scale arm: **SKIPPED** — no comparable validated Qwen3 J-Lens instrument; 32B/30B also exceed disk.

## Frozen scope

Validation only. P1/P2/P3 are not tested. Exactly R1/R2/R3 and seven fixed candidate summaries are permitted. Protocol SHA-256: `06e8ecf624b1aae7060426fbbc17429e60809caf9e24ed53b79dfb6ae8190d31`.

## Notebook 20

- 163 cases: 155 engines and all 8 dashboards.
- 79 semantic concepts: 75 engines and 4 dashboard languages.
- Coordinate-resampling A defined for 155/163; fixed masked alpha=1.5 source-to-foil B defined for 161/163.
- A/B case-level correlation: Pearson r=0.33548809947421065, Spearman rho=0.4101667482090183 (N=155).
- Declared label verification: **FAILED_DECLARED_LABEL_COVERAGE**, failures=49 (failures retained).

No READ score or AUC has been inspected yet. Notebook 21 is next.

# Consolidation notes

## Phase 1 inventory of research Markdown

The inventory below covers every tracked Markdown file outside `experiments/`
that contains research status, methods, results, audits, hypotheses, diagnosis,
recommendations, or an integrated research narrative. Paths are the original
pre-consolidation locations.

- `README.md` — project-facing status, plain-language framing, headline results, scope, attribution boundary, repository map, and reproduction instructions; retain as the concise landing page and update its entry-point links.
- `paper/workshop_paper.md` — workshop-style abstract, motivation, J-Lens/J-Space background, full method, final results, failure history, limitations, and references; archive as a superseded integrated draft.
- `results/RESULTS.md` — canonical pre-consolidation numeric report for setup, causal sanity, binary detection, graded failure, score distributions, and provenance; archive after its values are transferred verbatim to the new root `RESULTS.md`.
- `results/selection_audit/gate_flow.md` — reconstruction of the candidate pool, verification code, exact selection funnel, rejection reasons, and Gate 1 decision; archive as detailed selection-audit evidence.
- `results/selection_audit/distribution_compare.md` — retained-versus-eligible-rejected causal distributions, gate breakdown, relation-family summaries, and the scope of the selection inference; archive as detailed selection-audit evidence.
- `results/selection_audit/VERDICT.md` — scoped Finding A, deciding counts, plain-language interpretation, limitations, and audit-integrity statement; archive as the selection-audit decision record.
- `results/graded_diagnosis/prior_work.md` — inventory of earlier graded-meter analyses and the diagnostic questions they had not answered; archive as diagnosis provenance.
- `results/graded_diagnosis/hypotheses.md` — preregistered H1–H7 hypotheses, fixed quantities, evidence criteria, and interpretation guardrails; archive as the frozen diagnosis plan.
- `results/graded_diagnosis/analysis.md` — read-only checks of joins and formulas plus target variation, power, covariates, path completeness, directionality, subgroup stability, alignment, and unresolved selection; archive as the full diagnostic analysis.
- `results/graded_diagnosis/ranking.md` — evidence-ranked causes of the graded failure and separation of software, numerical, dataset, and estimator limitations; archive as the diagnostic synthesis.
- `results/graded_diagnosis/RECOMMENDATION.md` — current-pairing NO-GO decision, frozen writeup sentence, classification, and strict conditions for any independent reopening study; archive as the final graded-meter decision record.

`data/README.md` is operational documentation of the data layout rather than a
research record, so it remains in place and is not superseded. Generated or
third-party Markdown under `.venv/` and `.pytest_cache/` is untracked and out
of scope. Markdown under `experiments/` is explicitly protected from changes;
it remains an archival source rather than a consolidation target.

## Section-to-source map

### `RESEARCH_JOURNEY.md`

- **Problem, motivation, and attribution:** `paper/workshop_paper.md` §§1–2,
  `README.md` (opening and attribution boundary), and the attribution statement
  in `experiments/README.md`.
- **Initial instrument failure, violent intervention, surgical repair, and
  failed READ attempts:** `experiments/README.md` §§01–04 and
  `experiments/03_read_attempts_failed/results/WRITEUP_v5.md`. These protected
  files are sources only and will not move.
- **Final matched and firewalled pipeline:** `paper/workshop_paper.md` §3,
  `README.md` (idea, firewall, and protocol), and `results/RESULTS.md`.
- **Binary result and failed graded interpretation:** `results/RESULTS.md`,
  `paper/workshop_paper.md` §4, and
  `results/graded_diagnosis/RECOMMENDATION.md`.
- **Selection stress test:** `results/selection_audit/gate_flow.md`,
  `results/selection_audit/distribution_compare.md`, and
  `results/selection_audit/VERDICT.md`.
- **Graded diagnosis and NO-GO:** `results/graded_diagnosis/analysis.md`,
  `results/graded_diagnosis/ranking.md`, and
  `results/graded_diagnosis/RECOMMENDATION.md`.
- **Limitations and open questions:** `paper/workshop_paper.md` §5.2,
  `README.md` (scope and open research),
  `results/selection_audit/VERDICT.md`, and the graded diagnosis records.

### `RESULTS.md`

- **Setup:** `results/RESULTS.md` (scope and dataset),
  `paper/workshop_paper.md` §3.1, and `results/metrics.json`.
- **Binary detector and causal sanity:** `results/RESULTS.md` (causal sanity and
  binary detection) and `results/metrics.json`.
- **Graded result and pooled-result warning:** `results/RESULTS.md` (graded-use
  stress test).
- **Discriminating graded diagnostics, completeness, and power:**
  `results/graded_diagnosis/analysis.md` and
  `results/graded_diagnosis/ranking.md`.
- **Selection funnel and scoped verdict:**
  `results/selection_audit/gate_flow.md` and
  `results/selection_audit/VERDICT.md`.
- **Final classification:** `results/RESULTS.md`,
  `results/selection_audit/VERDICT.md`, and
  `results/graded_diagnosis/RECOMMENDATION.md`.

## Planned archive layout

The workshop draft will move to
`results/archive/paper/workshop_paper.md`; the old results report to
`results/archive/RESULTS.md`; and the two detailed audit directories to
`results/archive/selection_audit/` and
`results/archive/graded_diagnosis/`. The relative grouping is retained so the
records remain easy to inspect. The root `README.md` will be edited in place.

## Phase 5 consistency check

Pending.

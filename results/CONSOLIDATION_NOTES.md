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

## Archive layout

The workshop draft and its six figure assets moved to
`results/archive/paper/`; the old results report moved to
`results/archive/RESULTS.md`; and the two detailed audit directories moved to
`results/archive/selection_audit/` and
`results/archive/graded_diagnosis/`. The selection audit's machine-readable
`rejected_C.json` moved with its Markdown records. Relative links were repaired
after the moves, and the root `README.md` was edited in place.

## Phase 5 consistency check

**Outcome: PASS.** No untraceable scientific value remains; the
untraceable-source TODO list is empty.

### Number traceability

Every displayed scientific number in the two synthesis documents was checked
at its written precision against a populated source. The source groups are:

- `RESEARCH_JOURNEY.md` historical values (`8` to `6`, observed `4`, `3/3`,
  `alpha=2`, `1/8`, `r=0.062`, `0/8`, and `rho=-0.077`) come from the protected
  `experiments/README.md` and
  `experiments/03_read_attempts_failed/results/WRITEUP_v5.md`.
- Final setup, causal medians, binary AUCs, the canonical within-engine graded
  result, causal range, and pooled descriptive rho come verbatim from
  `results/archive/RESULTS.md`. The six-decimal Markdown displays were retained
  rather than replaced with differently represented raw floating-point values.
- Graded diagnostic correlations, completeness statistics, and power/MDE
  values come verbatim from
  `results/archive/graded_diagnosis/analysis.md`, with interpretation checked
  against `results/archive/graded_diagnosis/ranking.md` and
  `results/archive/graded_diagnosis/RECOMMENDATION.md`.
- Selection counts and Finding A's scope come verbatim from
  `results/archive/selection_audit/gate_flow.md` and
  `results/archive/selection_audit/VERDICT.md`.

Two potential fabrication traps were checked explicitly. The token location is
reported as the explicit concept token rather than an invented fixed integer
position. The pooled rho has no populated confidence interval, so none is
reported; the canonical within-engine interval is not reused for it. The power
statement is labeled as the conservative 24-independent-group approximation,
not the unestimated power of the unequal clustered design.

### Archive integrity

The numerical token stream of each of the ten moved Markdown records was
compared with its source at commit
`af44ce35452a32b07326ccb22d867c4d0d4d2418`; all ten comparisons passed.
Scientific prose and values were preserved. Three archived records received
relative-link-only repairs necessitated by their deeper paths; the workshop
figures moved with the draft, and the selection JSON moved with its audit.

### Protected-tree integrity

A sorted SHA-256 manifest was captured before the consolidation for every file
under `experiments/` and compared with a fresh manifest afterward. All `152/152`
files are byte-identical. Both manifests hash to
`170905b0f6d9d617228b2c138956b79c8b33fe62b253ede2932aa79442538f84`.
`git diff` from branch base
`b876a1df814ec1d713890c54c7ffdd5d2fd99641` also reports no change under
`experiments/`.

### Links and deliverables

A local-target check over `README.md`, both synthesis documents, this file, and
all archived Markdown found no broken relative link. The final layout contains
exactly the two requested new synthesis documents, `RESEARCH_JOURNEY.md` and
root `RESULTS.md`, plus this required working/audit record. The prior detailed
research Markdown is preserved under `results/archive/`, and `data/README.md`
remains unchanged as operational documentation.

No test suite, pytest, Ruff, model run, causal intervention, READ computation,
or result recomputation was performed, as required for this writing and
archival task.

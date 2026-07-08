# Selection audit: verification-gate flow

## Scope and provenance

This is a read-only diagnostic of the frozen selection process at source commit
`1a1c6ff77f25389234d4b661f2cfd49fec4c4926`. It does not compute or consume
READ. It reconstructs the candidate funnel from records that existed before
this audit.

The pre-gate pool is real and recoverable:

- The tracked source [`data/specs/twohop_supplement.json`](../../../data/specs/twohop_supplement.json)
  contains 236 validated reciprocal items. Its SHA-256 is
  `a27a79a28831e49c22b2f1a4981ab4308806b0ba2bd5a9b362dd4f5ab6274701`.
- `build_symmetric_causal_candidates` pairs those items deterministically into
  118 candidate pairs in 33 dependency groups. The builder is in
  [`src/datasets.py`](../../../src/datasets.py#L286-L429).
- The frozen local record `artifacts/final/01_dataset.json` preserves all 118
  rows, including every evaluation rejection and its reasons. Its SHA-256 is
  `4d15652b0491aef16df59756d0d7cf0245957f591045574c26fe92dd1908fb8a`.
  The identical 118-row roster is also present in
  `artifacts/final/01_clean_manifest.json`.
- The full artifact records 25 calibration pairs, 93 evaluation pairs, all
  group assignments, and `tokenization_rejections=[]`.

`artifacts/final/` is intentionally ignored by Git, so the tracked supplement
is the durable candidate source and the local frozen artifacts are the
authoritative record of model-dependent gate outcomes.

## What the code actually gates

The final verification function computes a joint set of booleans rather than
short-circuiting rows one gate at a time. In
[`apply_symmetric_verification_gate`](../../../src/datasets.py#L625-L705):

1. `engine_verified` requires both engine targets to be clean top-1 and both
   own concepts to exceed the global WRITTEN threshold.
2. `control_verified` additionally requires both original-dashboard targets to
   be clean top-1.
3. Evaluation status is `VERIFIED` exactly when `control_verified` is true.
   Calibration rows are always labeled `CALIBRATION_ONLY`.

The dashboard WRITTEN reason codes reuse the engine WRITTEN booleans because
engine and dashboard share the byte-identical fact context at the measured
token. They are aliases, not two additional visibility tests.

There is **no held-out reciprocal/interchange-consistency rejection gate**.
Reciprocity is part of candidate construction and the clean/WRITTEN conditions
must pass in both directions. Causal interchange is used on calibration rows
to select a single global layer, then is measured on the 77 verified evaluation
rows; directional disagreement is reported later and does not remove rows.

## Candidate funnel

The split is an assignment, not a rejection: the 118 pairs become 25
calibration pairs and 93 held-out evaluation pairs. Exact tokenization then
passes all 118 pairs.

The table below expresses the joint evaluation checks in protocol order. Since
all clean-answer checks pass independently, this sequential presentation has
the same counts as the joint implementation.

| Stage | Enter | Pass / leave | Removed at stage | Evidence |
| --- | ---: | ---: | ---: | --- |
| Reciprocal candidates constructed | 118 | 118 | 0 | 236 source items, 33 dependency groups |
| Assign calibration split | 118 | 25 calibration | 0 | whole-group allocation; not rejection |
| Assign evaluation split | 118 | 93 evaluation | 0 | remaining whole groups |
| Exact tokenization | 118 | 118 | 0 | `tokenization_rejections=[]` |
| Evaluation: both engine targets clean top-1 | 93 | 93 | 0 | all `engine_top1_a/b=true` |
| Evaluation: both own concepts WRITTEN at L16 | 93 | 77 | 16 | threshold `2.482430934906006` |
| Evaluation: both dashboard targets clean top-1 | 77 | 77 | 0 | all 93 pass this check independently |
| Final evaluation status | 93 | 77 `VERIFIED` | 16 `UNVERIFIED` | exact frozen count |

For completeness, all 25 calibration rows pass both engine and dashboard
clean-answer checks; 24 pass both WRITTEN checks and one does not. Calibration
rows are not part of the held-out 77/16 decision.

## Rejection characterization available at Gate 1

All 16 held-out `UNVERIFIED` rows fail only the WRITTEN visibility condition:

| Failure pattern | Count |
| --- | ---: |
| Concept A below WRITTEN threshold only | 7 |
| Concept B below WRITTEN threshold only | 8 |
| Both concepts below WRITTEN threshold | 1 |
| Any engine clean-answer failure | 0 |
| Any dashboard clean-answer failure | 0 |
| Any separate reciprocal/interchange-consistency failure | 0 |

The 16 WRITTEN exclusions comprise 12 country-capital, 3 US-state-capital, and
1 element-symbol pair. The evaluation pool and final verified roster are:

| Relation family | Evaluation candidates | Verified | WRITTEN-only exclusions |
| --- | ---: | ---: | ---: |
| Element symbol | 9 | 8 | 1 |
| Country capital | 53 | 41 | 12 |
| US-state capital | 31 | 28 | 3 |
| **Total** | **93** | **77** | **16** |

Under the audit specification, a rejected candidate is eligible for rejected-C
measurement only if the concepts are WRITTEN and the row was rejected for some
other reason. The number of such candidates is **zero**. The specification
explicitly excludes concept-not-WRITTEN rows from evidence of magnitude bias.

## Gate 1 decision

**PASS.** The pre-gate candidate pool and rejection reasons are preserved, so
the audit does not need to reconstruct or fabricate a pool. Phase 2 therefore
proceeds with an empty, evidence-backed rejected-C roster: zero visible
candidates were removed by the jointly evaluated clean-answer conditions, and
no interchange-consistency rejection condition exists.

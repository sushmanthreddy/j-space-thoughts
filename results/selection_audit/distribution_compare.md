# Selection audit: retained versus rejected causal magnitude

## Comparison population

The retained population contains the 77 held-out `VERIFIED` engines in
`artifacts/final/02_causal.json` (SHA-256
`9e5ad02e6ce2133e66d733e1f644fe14921254da2a0095b8a2603af890bc5402`).
Their causal score is the existing signed, unclipped, symmetric full-residual
interchange `C`; this report summarizes its absolute magnitude.

The rejected-but-real population is empty. Of 93 held-out candidates, all 93
pass both engine clean-answer checks and both dashboard clean-answer checks.
The 16 exclusions fail only WRITTEN. Under the audit's explicit eligibility
rule, concept-not-WRITTEN exclusions are not evidence of magnitude bias and do
not receive a causal recomputation. Consequently:

- eligible rejected candidates: **0**;
- rejected-candidate causal computations: **0**;
- READ invocations: **0**.

The empty roster and all 16 excluded IDs are recorded in
[`rejected_C.json`](rejected_C.json).

## Requested distribution summaries

Quartiles use NumPy's default linear quantile convention, matching the
read-only recomputation from the frozen causal artifact.

| Population | N | Median absolute C | Q1 | Q3 | IQR | Full range |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Retained engines | 77 | 0.912714 | 0.880435 | 0.940546 | 0.060111 | [0.785789, 1.012025] |
| Eligible rejected-but-real engines | 0 | N/A | N/A | N/A | N/A | N/A |

The requested fraction of rejected-but-real candidates below the retained
minimum of `0.785789` is **undefined (0/0)**, not 0%. Reporting 0% would imply a
measured rejected distribution when none exists.

## Breakdown by rejecting gate

| Gate or condition | Held-out rows excluded | Eligible for rejected C | C computed | Fraction below retained minimum |
| --- | ---: | ---: | ---: | --- |
| Exact tokenization | 0 | 0 | 0 | N/A |
| Engine clean top-1, both directions | 0 | 0 | 0 | N/A |
| Dashboard clean top-1, both directions | 0 | 0 | 0 | N/A |
| Separate reciprocal/interchange consistency | 0 | 0 | 0 | N/A; no such row-level gate exists |
| WRITTEN visibility, both directions | 16 | 0 | 0 | Out of scope by audit rule |

Among L16-visible held-out candidates, 77 enter the post-WRITTEN checks and all
77 are retained. Therefore the retained causal range is the entire observed
causal range for the audit-eligible visible population; no later verification
gate can have compressed it.

## Retained engines by relation family

| Relation family | N | Dependency groups | Median absolute C | Q1 | Q3 | IQR | Full range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Element symbol | 8 | 8 | 0.943443 | 0.908542 | 0.957150 | 0.048609 | [0.826996, 0.973753] |
| Country capital | 41 | 10 | 0.906790 | 0.871822 | 0.924188 | 0.052366 | [0.811015, 1.012025] |
| US-state capital | 28 | 6 | 0.929675 | 0.900042 | 0.943100 | 0.043058 | [0.785789, 0.990667] |

No one relation family supplies the entire high-magnitude range. The family
medians lie between 0.906790 and 0.943443, their ranges overlap, the global
maximum occurs in country capitals, and the global minimum occurs in US-state
capitals.

For context, the WRITTEN pass counts are 8/9 element-symbol, 41/53
country-capital, and 28/31 US-state-capital evaluation pairs. The lower
country-capital visibility pass rate may matter for representational coverage,
but the audit specification correctly prevents interpreting it as causal
magnitude selection without measuring C for not-WRITTEN concepts.

## Plain-language result

There is no evidence that clean-answer or interchange-consistency gates removed
weak-but-real **visible** engines: those gates removed zero such rows. Within
the population the study defines as eligible—concepts that are WRITTEN at L16—
the verification process retained every candidate that reached the later
checks.

This is narrower than proving the entire dataset construction is
magnitude-neutral. The audit does not determine whether the WRITTEN threshold
itself excludes causally real but poorly represented concepts, because the
specification places those 16 rows outside the rejected-C population. It also
does not undo calibration's use of causal separation to select L16.

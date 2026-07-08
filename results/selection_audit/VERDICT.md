# Selection audit verdict

## Finding A — observed post-WRITTEN gates are magnitude-neutral

The audit supports **Finding A within the experiment's declared population of
L16-visible concepts**. The clean-answer and downstream verification checks did
not preferentially discard weak-but-real visible engines, because they did not
discard any visible engine at all.

## Deciding numbers

1. The frozen held-out pool contains **93** evaluation candidates.
2. **77** pass WRITTEN in both reciprocal directions; **16** fail only WRITTEN.
3. Of the 77 visible candidates, **77/77** pass both engine clean-answer checks
   and both dashboard clean-answer checks; **0** visible candidates are rejected
   by any non-WRITTEN gate.
4. The retained engines have median absolute `C = 0.912714`, IQR
   `[0.880435, 0.940546]`, and full range `[0.785789, 1.012025]`. Because the
   eligible rejected population has `N=0`, its distribution and the requested
   below-minimum fraction are undefined rather than zero.

There is no separate held-out reciprocal/interchange-consistency gate. The
reciprocal structure is built into candidate construction and both directions
must pass the clean/WRITTEN checks. Causal directional disagreement is reported
after verification and removes no rows.

## Plain-language explanation

For concepts that meet the study's WRITTEN definition, the feared selection
effect is not present: every held-out candidate that was visible at L16 made it
into the final engine set. Therefore the later verification gates cannot be the
reason the retained engines occupy a narrow, strong causal range. The binary
result remains supported as originally scoped—distinguishing relevant from idle
among explicit concepts that are detectably WRITTEN at the selected layer.

This finding does **not** show that the WRITTEN eligibility rule itself is
causal-magnitude-neutral. The 16 excluded pairs were not-WRITTEN, and the audit
specification correctly forbids using them as evidence about magnitude or
computing rejected-C for them. It also does not revisit calibration's causal
selection of L16. A separate preregistered study would be needed to ask whether
causally real but weakly represented concepts fall below the visibility
threshold.

## Audit integrity

- The candidate pool came from an actual tracked supplement and frozen
  118-row artifact; no pool was fabricated.
- The eligible rejected-C roster is explicitly empty in
  [`rejected_C.json`](rejected_C.json).
- No model or GPU causal run was performed because there were zero eligible
  rejected candidates.
- READ was never imported, invoked, or recomputed.
- Existing validated-result files were not modified.

Supporting details:

- [Exact gate funnel](gate_flow.md)
- [Retained/rejected distribution report](distribution_compare.md)
- [Machine-readable rejected-C roster](rejected_C.json)

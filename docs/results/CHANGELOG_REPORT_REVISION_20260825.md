# Experiment 01 — report revision changelog

**Revision date:** 2026-08-25  
**Scope:** narrative and read-only diagnostics only

## Invariants

- Phase-I, Phase-II and Phase-III-R numerical result tables are unchanged.
- Frozen splits, subsets, targets, budgets, seeds, alpha grid, whitening grid,
  readers, thresholds and technical outcomes are unchanged.
- No encoder was trained and no scientific fit was rerun for this revision.
- The frozen Phase-I classification remains `A1`; the frozen Phase-III-R
  classification remains `R3`.

## Phase I

- Made `A1` a frozen secondary technical classification rather than the
  scientific headline.
- Opened the report with three separate diagnostics: directional spectral
  anti-alignment, the `last_concat512 → meanK_concatS` pooling interaction and
  the Phase-I normalized gaps for direction, volatility and timing.
- Separated the operational linear ceiling gap, normalized recovery gap and
  whitening mediation.
- Corrected whitening depth language to `k_50gap=128` and
  `k_nonrobust=508`; no few-PC concentration is claimed.
- Added the complete k=8/16/32/64 non-monotonicity audit, budget `n/D`, raw
  critical-budget metrics, common-alpha source verification, parity and a
  machine-readable claim table.
- Labelled current intervals as computational-robustness intervals and exposed
  grouped stock-day uncertainty as pending.

## Phase II

- Replaced copied narrative literals with values read from the frozen Phase-II
  artifacts and derived the current Phase-III-R status rather than repeating
  the historical execution-time flag.
- Restricted causal language: predictive-mass/whitening agreement is a
  descriptive bridge, not proof of a causal spectral mechanism.
- Corrected the `17:32` versus `33:64` discussion. The bands contain 16 and 32
  directions, respectively, so their raw paired difference is retained only as
  a legacy audit. The report now surfaces each band's own dimension-matched
  Haar null and descriptive predictive mass per direction.
- Preserved all Phase-II results, including the original negative conclusion
  about the proposed local non-monotonic explanation.

## Phase III-R

- Kept `R3` unchanged but restricted its meaning to the frozen selected MLP
  family and conditioning transforms.
- Moved raw R², medians, ranges, negative-score fractions, ceilings and
  eligibility ahead of normalized-gap interpretation.
- Marked the attenuation values as algebraic technical-rule outputs rather
  than stable effect sizes when low-budget raw R² is negative.
- Removed any general claim that nonlinear readers preserve the same
  accessibility mechanism.

## Research note and project summary

- Kept temporal averaging, token-role projection and PCA as three distinct
  operators.
- Reframed the historical Hadamard result as a fixed-projection operational
  contrast. The 128-dimensional common block and 384-dimensional complement
  are not directly dimension matched, and their independently fitted
  out-of-sample R² values are not additive.
- Gated any privileged token-role or unified mechanism on the pending
  structured Haar null.
- Distinguished the historical post-P0 nonlinear reader from Phase III-R.

## Corrective diagnostic completed on 2026-08-26

- Added the separately preregistered T2 token-role matched-null diagnostic; it
  does not modify any Phase-I/II/III-R result or threshold.
- Reproduced all 1,188 historical full/common/complement cells within the
  registered tolerance before evaluating the null.
- Evaluated 100 dimension- and structure-matched role-Haar draws for all three
  arms, three encoder seeds and both readouts, plus the generic and shuffled
  controls.
- The all-ones common direction was not unusually weak in all three seeds and
  its complement was not unusually strong in all three seeds for any
  arm/readout.
- Replaced the earlier pending status with the completed conclusion: retain the
  fixed-projection observation, but omit a privileged Hadamard/relational
  mechanism from the mathematical simulator.

## Provenance note

Phase III-R's identity gate attests the original Phase-II manifest SHA-256
`1a30b67f6739a1a0440eae1866ee55f72cddf94248e5edf336a7e605461144c2`.
The Phase-II directory manifest was subsequently regenerated after report-code
revision, so it is not presented as that historical identity. The provenance
record retains both roles explicitly and verifies that the frozen Phase-II
result and summary hashes did not change.

See `REPORT_REVISION_PROVENANCE_20260825.json` for exact hashes.

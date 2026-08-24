# Changelog — Experiment 01 Phase I narrative revision

Date: 2026-07-31.

## Narrative changes

- Replaced prior report SHA-256: `37b382ce04fedf3362a44ad948314910bfe69f09eca053acb22d9aaeabd51812`.
- Moved `A1` from the report headline to a secondary technical
  classification; its value and preregistered rule are unchanged.
- Reframed the primary result around three distinct specificity diagnostics:
  established spectral anti-alignment, established `last → meanK` fragility,
  and Phase-I normalized finite-sample gaps across target blocks.
- Separated the robust `0.165405` operational ceiling gap, robust
  normalized-recovery gap, and whitening mediation.
- Corrected whitening language: `k_50gap=128` halves but does not
  eliminate the gap; `k_nonrobust=508` is near-complete whitening;
  no few-PC concentration is
  claimed.
- Replaced any possible “coexisting B” reading with “A1 with a robust ceiling
  gap.”

## Added read-only diagnostics

- Added hierarchical intervals and paired adjacent-depth differences for
  `k=8,16,32,64`, plus results by encoder seed and target block. The diagnostic
  reads frozen recovery points only and is explicitly post hoc.
- Diagnostic row counts: 12 depth intervals, 9 hierarchical paired
  differences, 36 depth-by-encoder rows and 27 paired-difference-by-encoder
  rows. Directional paired results (`to−from`) are:
  `8→16 0.000794 [-0.008151, 0.009747]; 16→32 0.017974 [0.009575, 0.028355]; 32→64 -0.113101 [-0.178963, -0.071438]`.
- Added the trace-scale audit (`last` horizon/supervised ≈ `1.40`), the exact
  regularization formula, and an explicit verification that figure 06 uses
  `ridge_raw_common_alpha`. Fixed-absolute-lambda comparisons are absent.
- Added historical reproduction parity, production full-budget test results,
  and new-test min-norm OLS as a separate diagnostic with no old/new split
  equality requirement.

## Unchanged protected artifacts

- `results.parquet`: `ecf4e410c595baa32d06a1998bbd5151794d02ff141499af3c1f56268e110ffb`
- `summary/summary.json`: `7978961be69e50881ac022a67bfd7fea4f619c9806374121b57d6d4cbac1d4a6`
- Technical outcome: `A1`.
- Thresholds: `k_50gap=128`,
  `k_nonrobust=508`, `delta=0.1`.
- No feature generation, reader fitting, Phase II, or Phase III was executed.

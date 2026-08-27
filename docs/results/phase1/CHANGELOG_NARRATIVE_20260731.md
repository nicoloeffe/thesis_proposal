# Changelog — Experiment 01 Phase I narrative revision

Revision date: 2026-08-25.

## Narrative changes

- Superseded the earlier narrative artifact; its exact identity remains in
  version-control history rather than in this deterministically regenerated
  report bundle.
- Restored `A1` as the explicit frozen preregistered technical
  classification while separating it from the scientific interpretation.
- Removed copied PCA/null, pooling, gap, ratio, budget, whitening-depth and
  execution-status literals. Narrative values now come from hashed inputs.
- Replaced the older PCA-ladder literals with the detected completed Phase-II
  summary and marked that evidence as later diagnostic context.
- Separated the robust `0.165405` operational ceiling gap, robust
  normalized-recovery gap, and whitening mediation.
- Whitening wording is generated from the frozen candidates:
  `k_50gap=128`, `k_nonrobust=508` and reduction
  `0.556291`; no few-PC concentration is claimed.
- Replaced any possible “coexisting B” reading with “A1 with a robust ceiling
  gap.”
- Added the supervised-pretraining-label limitation and restricted the reader
  result to frozen-representation accessibility.

## Added read-only diagnostics

- Added hierarchical intervals and paired adjacent-depth differences for
  `8`, `16`, `32`, `64`, plus results by encoder seed and target block. The diagnostic
  reads frozen recovery points only and is explicitly post hoc.
- Diagnostic row counts: 12 depth intervals,
  9 hierarchical paired differences,
  36 depth-by-encoder rows and
  27 paired-difference-by-encoder
  rows. Directional paired results (`to−from`) are:
  `8→16 0.000794 [-0.008151, 0.009747]; 16→32 0.017974 [0.009575, 0.028355]; 32→64 -0.113101 [-0.178963, -0.071438]`.
- Added the trace-scale audit (observed `last` horizon/supervised ratio
  `1.396123`), the exact
  regularization formula, and an explicit verification that figure 06 uses
  common alpha when generated. Fixed-absolute-lambda comparisons are absent.
- Added historical reproduction parity, production full-budget test results,
  and new-test min-norm OLS as a separate diagnostic with no old/new split
  equality requirement.
- Added raw/normalized decisive-budget table and a hashed claim table.
- Corrected uncertainty language: existing intervals are computational
  robustness intervals; grouped stock/day uncertainty is pending.
- Replaced stale Phase-II/III scope prose with detected artifact status:
  Phase II `complete`, Phase III-R
  `complete`.

## Unchanged protected artifacts

- `results.parquet`: `ecf4e410c595baa32d06a1998bbd5151794d02ff141499af3c1f56268e110ffb`
- `summary/summary.json`: `7978961be69e50881ac022a67bfd7fea4f619c9806374121b57d6d4cbac1d4a6`
- Technical outcome: `A1`.
- Thresholds: `k_50gap=128`,
  `k_nonrobust=508`, `delta=0.1`.
- This revision did not generate features, fit readers, train encoders or run a
  new experimental phase.

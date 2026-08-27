# Experiment 01 — F16 label-matched supervision diagnostic

**Status:** completed fixed-test diagnostic with post-hoc read-only corrective reaggregation

**Phase-I outcome:** A1 frozen and unchanged

**Scientific F16 conclusion:** rapid transition at the minimum supervision budget, not a verified smooth dose response

## Result

F16 varies target-aligned supervision while keeping the encoder architecture
matched. Its strongest result survives the audit: all
`12/12` primary
label-matched supervised-minus-horizon gaps are positive, every grouped
stock→stock-day interval excludes zero, and all
`84/84`
leave-one-stock-out gaps are positive.

The broader smooth-dose claim does not survive family deduplication.
`whitening_k128` is empirically the same diagnostic as Axis B: the maximum raw
R² difference is `0.000340`, the
raw-value correlation is `0.999991`, and the
entire whitening ladder changes test R² by at most
`0.000920`. Counting it separately
turns three distinct passing families into four nominal families. With the
original requirement of four passing families, the deduplicated smooth flag is
**False** (`3/5` distinct families
pass).

At the smallest budget, `7,116` labelled rows
(`0.108%` of full train), several geometry metrics
have already moved 82–89% of the horizon→supervised path. The observed pattern
is therefore a sharp early transition followed by saturation and small rank
inversions, not a graded response proportional to label volume.

## Decision audit: before amendment, after amendment, after deduplication

The exact Spearman boundary correction was mathematically justified: exact
`rho=0.8` had been serialized as `0.7999999999999999`. It was nevertheless
applied `6.48` minutes after test unlock and
changed three family decisions. The complete effect on all five flags is:

| flag | pre-boundary correction | post-boundary six-family | corrective deduplicated reading |
| --- | ---: | ---: | --- |
| supervised_like_at_low_label_volume | True | True | True |
| smooth_label_volume_dependence | False | True | False |
| accessibility_without_measured_geometry_change | True | False | True |
| low_budget_optimization_floor | False | False | False |
| directionality_specific_coadaptation | False | False | not identified |

The corrected `accessibility_without_measured_geometry_change=True` flag has a
narrow technical meaning: fewer than two **distinct geometry families** show
the all-seed rank pattern. It does not mean geometry is unchanged; the
minimum-budget shifts below are large.

The historical `directionality_specific_coadaptation=False` is not evidence
against specificity. Its volatility normalization divides by anchor gaps
`0.0163, 0.0206, 0.0125`,
all below the post-hoc `0.05` interpretability floor, producing unstable
ratios. Its scientific status is therefore **not identified**.

## Primary label-matched comparison

| budget | seed | F16 supervised R² | horizon-JEPA R² | paired gap | grouped 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| b_1 | 0 | 0.3162 | 0.1208 | 0.1954 | [0.1453, 0.2755] |
| b_1 | 1 | 0.2668 | 0.0550 | 0.2118 | [0.1503, 0.3120] |
| b_1 | 2 | 0.3054 | 0.0359 | 0.2695 | [0.1290, 0.5520] |
| b_16 | 0 | 0.3670 | 0.2299 | 0.1371 | [0.0908, 0.1821] |
| b_16 | 1 | 0.3708 | 0.2188 | 0.1520 | [0.1102, 0.1937] |
| b_16 | 2 | 0.3674 | 0.2236 | 0.1438 | [0.1056, 0.1854] |
| b_1_4 | 0 | 0.2516 | 0.0689 | 0.1827 | [0.1555, 0.2220] |
| b_1_4 | 1 | 0.1912 | 0.0188 | 0.1724 | [0.1425, 0.2140] |
| b_1_4 | 2 | 0.2134 | 0.0299 | 0.1835 | [0.1271, 0.2832] |
| b_4 | 0 | 0.3548 | 0.1980 | 0.1568 | [0.0857, 0.2277] |
| b_4 | 1 | 0.3493 | 0.1969 | 0.1524 | [0.0884, 0.2097] |
| b_4 | 2 | 0.3493 | 0.1896 | 0.1596 | [0.0922, 0.2343] |

## Frozen rank checks and monotonicity

Every metric is oriented so horizon-JEPA is 0 and canonical supervised is 1.
With only four budgets, `rho=0.8` permits one adjacent rank inversion; it is not
strict monotonicity. Only Axis A is strictly increasing in every seed.

| family | seed 0 rho | seed 1 rho | seed 2 rho | post-amendment pass | strictly monotone all seeds | audit role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| axis_a_accessibility | 1.000 | 1.000 | 1.000 | true | true | independent |
| axis_b_accessibility | 0.800 | 1.000 | 0.800 | true | false | independent |
| role_retention | 0.400 | 0.400 | 0.400 | false | false | independent |
| topk_predictive_mass | 0.400 | 0.800 | 0.200 | false | false | independent |
| pooling_loss | 1.000 | 1.000 | 0.800 | true | false | independent |
| whitening_k128 | 0.800 | 1.000 | 0.800 | true | false | duplicate of Axis B |

Several ordering differences are at or below the cohort-resolution scale. For
Axis B, the largest raw inversion is
`0.0088`
against a `0.02` convergence tolerance. Rank-based pass/fail is therefore not
evidence for a precisely resolved continuous law.

## Minimum-budget saturation

The table reports the mean and seed range of the horizon→supervised oriented
coordinate at `b_1_4`.

| distinct family | mean path completed | seed range |
| --- | ---: | ---: |
| axis_a_accessibility | 0.578 | [0.520, 0.643] |
| axis_b_accessibility | 0.451 | [0.381, 0.550] |
| pooling_loss | 0.863 | [0.789, 0.927] |
| role_retention | 0.893 | [0.855, 0.921] |
| topk_predictive_mass | 0.819 | [0.764, 0.892] |

The preregistered low-volume threshold rule passes only at `b_1` (28,446
rows), not at `b_1_4`, because the joint Axis-B midpoint condition fails at the
floor. “Supervised-like” is a threshold label, not statistical equivalence to
the canonical supervised encoder.

## Interpretation and limits

F16 supports two claims: target-aligned supervision rapidly changes the
learned representation, and label-matched F16 encoders outperform
horizon-JEPA on the directional readout. It does not support a smooth
six-family dose-response law. The dominant empirical shape is an early step
and plateau, consistent with the possibility that even a small amount of
target-aligned gradient selects a different geometry.

F16 changes label volume, target exposure and optimization trajectory together.
It does not isolate a universal causal effect of label count. Validation labels
used for checkpoint selection are not included in the nominal training-label
budget. The intervals cover seven stocks from one market and remain
descriptive despite grouped bootstrap and leave-one-stock-out checks.

## Integrity

- frozen `f16_results.parquet`, `f16_geometry.parquet`, grouped uncertainty,
  selections, thresholds and checkpoints are unchanged;
- no test reopening, new training or new fit was performed;
- the original post-amendment summary is retained as historical technical
  output;
- this correction is a deterministic reaggregation of frozen artifacts.

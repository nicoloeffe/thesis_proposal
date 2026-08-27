# Experiment 01 — F16 label-matched supervised dose response

**Status:** complete fixed-test diagnostic  
**Phase-I outcome:** A1 with robust ceiling gap, frozen and unchanged  
**F16 pattern:** `multiple_preregistered_patterns_passed_report_separately`

## Result

F16 varies the amount of target-aligned supervision used to train an otherwise
matched supervised encoder. Axis A fits a fresh reader on the same labelled
budget; Axis B holds the reader budget fixed at `b_16` and diagnoses the
representation. The test was opened once only after all checkpoint and alpha
selections were hash-frozen. No test result changed a selection.

The preregistered interpretation flags are:

- supervised-like at low label volume: **True**;
- smooth label-volume dependence: **True**;
- accessibility change without measured geometry change: **False**;
- low-budget optimization floor: **False**;
- directionality-specific co-adaptation: **False**.

These flags are reported separately. Mixed evidence is not collapsed into a
new A/B/C/D outcome, and F16 does not modify Phase I.

## Primary label-matched comparison

The table reports block-mean independent-target R² under `last_concat512`.
Intervals are paired hierarchical test-set intervals that resample stocks and
then stock-days; seven-stock leave-one-out estimates are in the complete
external artifact `f16_grouped_uncertainty.parquet`.

| budget | seed | F16 supervised R² | horizon-JEPA R² | paired gap | grouped 95% interval |
|---|---:|---:|---:|---:|---:|
| b_1_4 | 0 | 0.2516 | 0.0689 | 0.1827 | [0.1555, 0.2220] |
| b_1_4 | 1 | 0.1912 | 0.0188 | 0.1724 | [0.1425, 0.2140] |
| b_1_4 | 2 | 0.2134 | 0.0299 | 0.1835 | [0.1271, 0.2832] |
| b_1 | 0 | 0.3162 | 0.1208 | 0.1954 | [0.1453, 0.2755] |
| b_1 | 1 | 0.2668 | 0.0550 | 0.2118 | [0.1503, 0.3120] |
| b_1 | 2 | 0.3054 | 0.0359 | 0.2695 | [0.1290, 0.5520] |
| b_4 | 0 | 0.3548 | 0.1980 | 0.1568 | [0.0857, 0.2277] |
| b_4 | 1 | 0.3493 | 0.1969 | 0.1524 | [0.0884, 0.2097] |
| b_4 | 2 | 0.3493 | 0.1896 | 0.1596 | [0.0922, 0.2343] |
| b_16 | 0 | 0.3670 | 0.2299 | 0.1371 | [0.0908, 0.1821] |
| b_16 | 1 | 0.3708 | 0.2188 | 0.1520 | [0.1102, 0.1937] |
| b_16 | 2 | 0.3674 | 0.2236 | 0.1438 | [0.1056, 0.1854] |

## Dose-response checks

Every metric is oriented so horizon-JEPA is 0 and the canonical supervised
anchor is 1. Spearman correlation is computed over the four frozen budgets
inside each encoder seed.

| family | seed 0 ρ | seed 1 ρ | seed 2 ρ | all seeds ≥0.8 |
|---|---:|---:|---:|---:|
| Axis-A accessibility | 1.000 | 1.000 | 1.000 | true |
| Axis-B accessibility | 0.800 | 1.000 | 0.800 | true |
| role retention | 0.400 | 0.400 | 0.400 | false |
| top-k predictive mass | 0.400 | 0.800 | 0.200 | false |
| pooling loss | 1.000 | 1.000 | 0.800 | true |
| whitening k=128 | 0.800 | 1.000 | 0.800 | true |

The overall smooth rule requires at least four of the six families; exactly
four pass. No training cell met the preregistered instability definition.

The low-volume supervised-like rule passes at `b_1`. At `b_1_4`, Axis A is
already on the supervised side of the anchor midpoint in all three seeds, but
Axis B is not; the stronger joint rule therefore does not pass at the floor.

## Geometry and controls

The complete geometry artifact reports common/full and contrast/full retention
non-additively, cumulative predictive mass, `last→meanK` loss, covariance
spectra and the frozen whitening bridge `k=0,8,16,32,64,128,256,508`.
Accessibility, pooling loss and whitening-k128 show the all-seed ordered
pattern; role retention and top-k predictive mass do not. Volatility and timing
remain specificity controls. The preregistered directionality-specific
co-adaptation condition does not pass in any encoder seed.

## Uncertainty and limits

Grouped intervals use 5,000 deterministic resamples with seed `20260827`.
All twelve primary intervals exclude zero; all 84 corresponding
leave-one-stock-out gaps are positive. With only seven stocks these intervals
remain descriptive, which is why the full leave-one-stock-out table is retained.

F16 changes label volume and target-aligned feature exposure together. A
low-budget effect cannot by itself separate label scarcity from exposure or
optimization, and a dose response does not establish a universal causal law
of representation learning.

## Integrity

- test cohort: 11,136 endpoints across 87 stock-days and seven stocks;
- checkpoint grid: 12 best plus 12 epoch-20 sensitivity checkpoints;
- training failures: 0;
- selections changed after unlock: false;
- Phase II/III, MLP, VICReg and simulators: not run by F16;
- two post-test amendments are limited to strict JSON serialization and the
  numerical representation of the exact Spearman boundary at 0.8.

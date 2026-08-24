# Report — Experiment 01 Phase I

## Primary result: directional specificity across three distinct diagnostics

The primary result is not the taxonomy label. It is the convergence of three
distinct specificity diagnostics:

1. **Established directional spectral anti-alignment.** At `m/D=1/32`, the
   horizon-JEPA final readout recovers `0.0050` of its full linear directional
   score, below the `0.0563` empirical random-subspace null. Supervised recovers
   `0.8971`, compared with its `0.6118` null. This establishes variance–task
   anti-alignment rather than a generic absence of predictive content.
2. **Established directional pooling fragility.** Under `last → meanK`, the
   production full-budget linear directional R² changes from `0.2199` to
   `0.0701` for horizon-JEPA, while supervised remains approximately stable
   (`0.3853 → 0.3941`). This is a readout interaction, not a second A/B/D
   outcome.
3. **Phase-I finite-sample specificity.** The mean normalized gaps are:

| target block | mean normalized finite-sample gap | directional / control |
| --- | --- | --- |
| directional | 0.5460 | — |
| volatility | 0.1838 | 2.97× |
| timing | 0.1528 | 3.57× |

The directional penalty is therefore approximately **3–3.5 times larger**
than the controls (exact ratios `2.97×` versus volatility and `3.57×` versus
timing). Volatility and timing remain specificity controls and are not pooled
into the directional result.

## Three effects that must remain separate

### Robust operational linear ceiling gap

At full production budget with tuned raw ridge, the supervised minus
horizon-JEPA directional R² gap is `0.165405`, with hierarchical
95% interval `[0.160753, 0.168857]`. The interval
excludes zero. This is a robust operational linear ceiling gap; it is not, by
itself, a normalized sample-efficiency statement.

### Robust normalized-recovery gap

After target-wise normalization by each representation's eligible operational
ceiling, the directional finite-sample gap remains robust across adjacent low
budgets. The mean over all six preregistered low budgets is
`0.546020`; the mean over the decisive `0.125` and `0.25`
days/stock cells is `0.700972`.

### Mediation by progressive whitening

The unchanged technical thresholds are `k_50gap = 128` and
`k_nonrobust = 508`. Partial whitening at `k=128` reduces
the decisive-budget normalized gap by `55.6%` but does not
eliminate it: the gap remains robust. Non-robustness requires `k=508`, i.e.
near-complete whitening of a 512-dimensional readout. The evidence therefore
does **not** justify saying that the problem is concentrated in a few leading
principal components.

## Secondary technical classification

The preregistered taxonomy is retained unchanged as a secondary technical
classification: **A1 with a robust ceiling gap**.

Technical rule satisfied: Native finite-sample gap is robust at adjacent low budgets and progressive whitening reduces it by at least 50% and makes it non-robust. The robust ceiling gap
above is reported alongside A1; `B` is not described as a coexisting outcome.

The complete unchanged machine-readable classification record remains in
`summary/summary.json` and is reproduced here for auditability:

```json
{
  "absolute_ceiling_gap": {
    "lower": 0.1607531894900738,
    "mean": 0.16540515477348147,
    "robust": true,
    "upper": 0.16885745649215148
  },
  "decisive_budgets": [
    0.125,
    0.25
  ],
  "delta": 0.1,
  "k_50gap": 128,
  "k_nonrobust": 508,
  "large_sample_ceiling_meaningful": true,
  "low_budget_mean_normalized_gap": 0.5460198970445955,
  "native_adjacent_robust_pairs": [
    [
      0.125,
      0.25
    ],
    [
      0.25,
      0.5
    ],
    [
      0.5,
      1
    ],
    [
      1,
      2
    ],
    [
      2,
      4
    ]
  ],
  "native_low_budget_mean_gap": 0.7009716425183607,
  "outcome": "A1",
  "reason": "native finite-sample gap is robust at adjacent low budgets and progressive whitening reduces it by at least 50% and makes it non-robust",
  "whitening_candidates": [
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 0,
      "mean_gap": 0.7009716425183556,
      "reduction_fraction": 7.327471962526033e-15
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 1,
      "mean_gap": 0.6913073400240535,
      "reduction_fraction": 0.013787009214219603
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 2,
      "mean_gap": 0.6844773584182045,
      "reduction_fraction": 0.02353060109664029
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 4,
      "mean_gap": 0.6743530908242632,
      "reduction_fraction": 0.037973792489616076
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 8,
      "mean_gap": 0.6577122128628066,
      "reduction_fraction": 0.0617135231036412
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 16,
      "mean_gap": 0.6585066964384894,
      "reduction_fraction": 0.060580119799581
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 32,
      "mean_gap": 0.6764809187948003,
      "reduction_fraction": 0.03493825176090337
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 64,
      "mean_gap": 0.5633800038814455,
      "reduction_fraction": 0.19628702545311782
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 128,
      "mean_gap": 0.31102723764057083,
      "reduction_fraction": 0.5562912694682596
    },
    {
      "all_nonrobust": false,
      "all_robust": true,
      "k": 256,
      "mean_gap": 0.1602048230004235,
      "reduction_fraction": 0.771453203977182
    },
    {
      "all_nonrobust": true,
      "all_robust": false,
      "k": 508,
      "mean_gap": 0.05212363313798933,
      "reduction_fraction": 0.9256408819182381
    }
  ]
}
```

## Whitening-depth non-monotonicity diagnostic

This added diagnostic uses only frozen Phase-I recovery points. Within each
encoder/subsample cell, it first averages the paired supervised–horizon gap
over the two decisive budgets (`0.125`, `0.25`), then applies 5,000-draw
hierarchical resampling of encoder seeds followed by paired cells within
encoder. It does not refit a reader and is not outcome-defining.

### Hierarchical intervals by target block and depth

| target block | k | mean gap | 95% interval | encoders / paired cells |
| --- | --- | --- | --- | --- |
| directional | 8 | 0.6577 | [0.6367, 0.6789] | 3 / 30 |
| directional | 16 | 0.6585 | [0.6386, 0.6785] | 3 / 30 |
| directional | 32 | 0.6765 | [0.6569, 0.6973] | 3 / 30 |
| directional | 64 | 0.5634 | [0.4906, 0.6122] | 3 / 30 |
| timing | 8 | 0.3489 | [0.2476, 0.4655] | 3 / 30 |
| timing | 16 | 0.3171 | [0.2440, 0.3940] | 3 / 30 |
| timing | 32 | 0.3643 | [0.2892, 0.4436] | 3 / 30 |
| timing | 64 | 0.4786 | [0.4035, 0.5478] | 3 / 30 |
| volatility | 8 | 0.2582 | [0.2151, 0.3049] | 3 / 30 |
| volatility | 16 | 0.2430 | [0.2042, 0.2832] | 3 / 30 |
| volatility | 32 | 0.2563 | [0.2153, 0.2988] | 3 / 30 |
| volatility | 64 | 0.3061 | [0.2693, 0.3408] | 3 / 30 |

### Paired differences between adjacent inspected depths

Differences are `gap(to k) − gap(from k)`, paired by encoder seed and subsample.

| target block | depth pair | paired Δ gap | 95% interval | excludes zero |
| --- | --- | --- | --- | --- |
| directional | 8→16 | 0.0008 | [-0.0082, 0.0097] | false |
| directional | 16→32 | 0.0180 | [0.0096, 0.0284] | true |
| directional | 32→64 | -0.1131 | [-0.1790, -0.0714] | true |
| timing | 8→16 | -0.0318 | [-0.0820, 0.0020] | false |
| timing | 16→32 | 0.0472 | [0.0306, 0.0645] | true |
| timing | 32→64 | 0.1143 | [0.0716, 0.1542] | true |
| volatility | 8→16 | -0.0152 | [-0.0418, 0.0007] | false |
| volatility | 16→32 | 0.0132 | [0.0058, 0.0208] | true |
| volatility | 32→64 | 0.0498 | [0.0219, 0.0775] | true |

### Means by encoder seed and target block

| target block | encoder seed | k=8 | k=16 | k=32 | k=64 |
| --- | --- | --- | --- | --- | --- |
| directional | 0 | 0.6396 | 0.6483 | 0.6697 | 0.4891 |
| directional | 1 | 0.6591 | 0.6508 | 0.6632 | 0.5898 |
| directional | 2 | 0.6744 | 0.6764 | 0.6966 | 0.6113 |
| timing | 0 | 0.3472 | 0.3363 | 0.3848 | 0.5214 |
| timing | 1 | 0.4519 | 0.3706 | 0.4096 | 0.4940 |
| timing | 2 | 0.2475 | 0.2442 | 0.2985 | 0.4205 |
| volatility | 0 | 0.2585 | 0.2561 | 0.2658 | 0.3176 |
| volatility | 1 | 0.2787 | 0.2372 | 0.2518 | 0.3024 |
| volatility | 2 | 0.2376 | 0.2359 | 0.2511 | 0.2982 |

For direction, `8→16` is indistinguishable from zero, `16→32` shows a small
positive paired change, and `32→64` a larger negative paired change; the latter
two intervals exclude zero. Volatility and timing show different local
patterns. This verifies local non-monotonicity in the inspected cells, but the
diagnostic is post hoc and does not modify the preregistered interpretation or
support a “few-PC” account. Full paired values per encoder are retained in the
four `15_whitening_nonmonotonicity_*` artifacts.

## Global covariance scale and regularization parity

`trace_cov_over_dim` is **not matched**. On `last_concat512`, the mean trace
scale is `0.360919`
for horizon-JEPA and
`0.258515`
for supervised, a horizon/supervised ratio of
`1.396`
(approximately `1.40`). The all-branch max/min ratio is
`3.306` because masked-JEPA
has a still larger trace.

Scientific common-regularization comparisons use the dimensionless parameter

`lambda = alpha * trace(covariance) / D`.

Figure 06 is verified to select `reader_family = ridge_raw_common_alpha` and
therefore compares **common alpha**, not common absolute lambda. No
fixed-absolute-lambda comparison is included in this report; with unmatched
trace scale, such a comparison would be marked confounded.

```json
{
  "last_concat512": {
    "approximately_matched_within_10pct": false,
    "jepa_horizon_over_supervised_ratio": 1.3961231190072076,
    "jepa_horizon_supervised_trace_matched_within_10pct": false,
    "max_over_min_ratio": 3.306248430729335,
    "trace_cov_over_dim_by_branch": {
      "jepa_horizon": 0.36091869271667787,
      "jepa_masked": 0.8547146345258958,
      "supervised": 0.2585149459979787
    }
  },
  "meanK_concatS": {
    "approximately_matched_within_10pct": false,
    "jepa_horizon_over_supervised_ratio": 1.0159637122972474,
    "jepa_horizon_supervised_trace_matched_within_10pct": true,
    "max_over_min_ratio": 4.996132087662248,
    "trace_cov_over_dim_by_branch": {
      "jepa_horizon": 0.16902756978152125,
      "jepa_masked": 0.8312147912995064,
      "supervised": 0.16637166045953
    }
  }
}
```

## Historical and production parity

### Historical reproduction gate

This is the mandatory old-split min-norm OLS reproduction check:

| branch | observed | historical reference | absolute difference | gate passed |
| --- | --- | --- | --- | --- |
| jepa_horizon | 0.211129 | 0.2111 | 0.000029 | true |
| supervised | 0.375636 | 0.3756 | 0.000036 | true |

The observed values reproduce the historical rounded references within the
frozen tolerance.

### Production full-budget test

The following scores use the new canonical test split and are deliberately
reported separately:

| branch | tuned ridge R² [95%] | min-norm OLS R² [95%] |
| --- | --- | --- |
| jepa_horizon | 0.219943 [0.216481, 0.224736] | 0.219791 [0.216256, 0.224649] |
| jepa_masked | 0.107020 [0.094023, 0.115318] | 0.107020 [0.094023, 0.115318] |
| supervised | 0.385349 [0.385218, 0.385489] | 0.385344 [0.385216, 0.385489] |

The new-test min-norm OLS values are diagnostics. They are **not required** to
equal the old-split reproduction values because validation and test are the
two chronological halves of the former held-out stock-days.

## Scope and leakage controls

- frozen post-P0 representations only;
- label budgets are nested stock-day groups;
- covariance/whitening uses all unlabelled train features only;
- alpha is selected on the fixed complete validation split;
- the fixed complete test split is evaluated only after configuration fixing;
- directional, volatility and timing are summarized separately;
- normalized recovery is target-wise and only uses full-budget R² at least 0.01.

## Specificity and time-of-day controls

```json
{
  "directional": {
    "full_train_raw_r2": {
      "jepa_horizon": 0.21994342851651347,
      "jepa_masked": 0.10702033978211407,
      "supervised": 0.38534858328999494
    },
    "interpretation_scope": "primary directional specificity result",
    "n_recovery_curve_rows": 39
  },
  "timing": {
    "full_train_raw_r2": {
      "jepa_horizon": 0.5430492625434953,
      "jepa_masked": 0.5127684048482807,
      "supervised": 0.6063927375948467
    },
    "interpretation_scope": "preregistered specificity control",
    "n_recovery_curve_rows": 39
  },
  "volatility": {
    "full_train_raw_r2": {
      "jepa_horizon": 0.49403668565822834,
      "jepa_masked": 0.4739590590772728,
      "supervised": 0.5165553854860679
    },
    "interpretation_scope": "preregistered specificity control",
    "n_recovery_curve_rows": 39
  }
}
```

Opening, middle and closing contiguous blocks are reported separately in
`time_of_day_sensitivity_summary.parquet` (18 rows).
These sensitivity cells are not pooled into the random-anchor curves.

## Uncertainty

Intervals use hierarchical resampling of encoder seeds followed by subsampling
seeds within encoder. Companion Parquet tables expose
`sd_subsample_within_encoder` and `sd_encoder_between_means`; all
encoder-specific curves are retained in figure 10.

## Figures and diagnostics

- `01_raw_directional_r2.png`
- `02_directional_normalized_recovery.png`
- `03_supervised_jepa_normalized_gap.png`
- `04_gap_vs_whitening_depth.png`
- `05_raw_vs_whitened_learning_curves.png`
- `06_common_alpha_gap_surface.png`
- `07_min_norm_ols_learning_curves.png`
- `09_low_budget_subsample_distributions.png`
- `10_encoder_specific_curves.png`
- `11_variance_decomposition.png`
- `12_target_specificity_panels.png`
- `13_readout_interaction_panels.png`
- `14_ceiling_eligibility_map.png`
- `06_common_alpha_audit.json`
- `15_whitening_nonmonotonicity_manifest.json`
- `15_whitening_nonmonotonicity_intervals.parquet`
- `15_whitening_nonmonotonicity_paired_differences.parquet`
- `15_whitening_nonmonotonicity_by_encoder.parquet`
- `15_whitening_nonmonotonicity_paired_by_encoder.parquet`

Phase II (PCA/random subspaces) and Phase III (MLP) were not run. This revision
changes narrative ordering and adds read-only diagnostics only; it does not
change Phase-I results, thresholds, the technical outcome, or the fit pipeline.

# Report — Experiment 01 Phase I

## Primary Phase-I result: reader-relative finite-sample accessibility

The Phase-I result supports the following restricted statement: conditional on
the frozen representations and a newly fitted reader, the supervised
representation is more accessible at low reader-label budgets. It does not
establish that supervised pretraining is intrinsically more label-efficient
end to end, because the supervised encoder saw directional and volatility
labels during pretraining.

### Directional spectral organization

Phase II places only `0.000094` of horizon-JEPA's directional predictive mass in its first 8 PCs, versus `0.751775` for supervised. The fraction of Haar draws beating top-PCA averages `1.000` for horizon-JEPA and `0.000` for supervised across the recorded encoder seeds.

When present, these values come from the completed Phase-II machine summary,
not from literals copied from the older post-P0 PCA ladder. They are diagnostic
evidence and do not alter Phase I.

### Pooling interaction

At full budget, changing `last_concat512 → meanK_concatS` changes directional test R² from `0.219943` to `0.070085` for horizon-JEPA and from `0.385349` to `0.394098` for supervised.

This matched readout contrast is reported as an interaction with pooling, not
as an additional A/B/D outcome.

### Finite-sample specificity

The table distinguishes normalized gaps over the frozen low-budget grid from
raw-R² gaps averaged over the recorded decisive budgets:

| target block | normalized gap (low-budget grid) | raw R² gap (decisive budgets) | normalized directional/control | raw directional/control |
| --- | --- | --- | --- | --- |
| directional | 0.5460 | 0.3053 | — | — |
| volatility | 0.1838 | 0.1536 | 2.97× | 1.99× |
| timing | 0.1528 | 0.1301 | 3.57× | 2.35× |

The descriptive directional/control ratios are `2.97×` and `3.57×` on the normalized-recovery scale, versus `1.99×` and `2.35×` on the raw-R² scale. The magnitude is therefore scale-dependent. These are point summaries, not an independence-adjusted target-block interaction test. Volatility and timing remain separate controls. These
ratios alone do not establish an interaction because target families are
correlated and grouped stock-day uncertainty has not yet been computed.

## Phase-I effects kept separate

### Operational linear ceiling gap

The supervised-minus-horizon operational ceiling gap is `0.165405` with computational-robustness interval `[0.160753, 0.168857]`. The interval excludes zero. When available, this is an operational linear ceiling
statement, not a normalized sample-efficiency statement.

### Normalized-recovery gap

Recovery is normalized target-wise by each representation's eligible
operational ceiling. The mean normalized directional gap over the frozen low-budget grid is `0.546020`; over the recorded decisive budgets `0.125`, `0.25` it is `0.700972`. The frozen summary records the adjacent robust pairs as
`0.125→0.25`, `0.25→0.5`, `0.5→1`, `1→2`, `2→4`.

### Mediation by progressive whitening

At `k_50gap=128`, whitening reduces the decisive-budget gap by `55.6%` but does not eliminate it. At the historical technical field `k_nonrobust=508`, the gap no longer meets the compound preregistered criterion `lower > 0 and mean ≥ δ=0.10` at both decisive budgets; this is an effect-threshold transition, not a confidence interval crossing zero. The decisive-budget mean gaps are `0.035524`, `0.068723`, and their lower interval bounds remain positive (`0.000085`, `0.036869`). It is the maximum tested valid whitening depth. The mean decisive-budget gap is reduced by `92.6%` there. This pattern does not support concentration in only a few leading PCs.

## Frozen preregistered classification and δ sensitivity

At the preregistered primary threshold `δ=0.10`, the
frozen Phase-I technical classifier returns **A1**. This label
is reported secondarily to the empirical accessibility result. In the
historical rule, “robust” is a
compound practical-effect criterion: the interval lower bound must exceed zero
and the point estimate must reach `δ`. It does not mean only “statistically
different from zero.”

| δ | technical class | k_50gap | k_nonrobust |
| --- | --- | --- | --- |
| 0.05 | D | 128 | — |
| 0.10 | A1 | 128 | 508 |
| 0.15 | A1 | 128 | 508 |

The `δ=0.05` and `δ=0.15` rows are mandatory preregistered sensitivities, not
alternative primary outcomes. A label change across this grid means that the
taxonomy is threshold-sensitive; it does not alter any measured gap. The
result rows, thresholds and classification logic have not been modified. The
operational ceiling result is stated separately as **A1 with
a robust ceiling gap**; `B` is not a coexisting outcome.

The frozen machine reason is retained verbatim in the record below. Its phrase
“makes it non-robust” must be read according to the compound criterion above,
not as a confidence interval crossing zero.

The complete machine-readable record is reproduced for auditability:

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

## Raw and normalized metrics at decisive budgets

The table co-reports raw test R², operational ceiling and normalized recovery.
Ranges are over the eligible target/encoder/subsample cells represented in the
frozen result table; they are not population intervals.

| block | budget | branch | raw R² mean / median | ceiling mean [range] | recovery mean / median [range] | eligible targets min–max | negative raw fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| directional | 0.125 | jepa_horizon | 0.0261 / 0.0222 | 0.2199 [0.0654, 0.3738] | 0.1447 / 0.1652 [-0.5340, 0.4060] | 12–12 | 0.131 |
| directional | 0.125 | supervised | 0.3319 / 0.3235 | 0.3853 [0.2736, 0.4658] | 0.8544 / 0.8518 [0.5903, 0.9708] | 12–12 | 0.000 |
| directional | 0.25 | jepa_horizon | 0.0382 / 0.0338 | 0.2199 [0.0654, 0.3738] | 0.1928 / 0.1960 [-0.2847, 0.5105] | 12–12 | 0.047 |
| directional | 0.25 | supervised | 0.3430 / 0.3321 | 0.3853 [0.2736, 0.4658] | 0.8851 / 0.8821 [0.6736, 0.9764] | 12–12 | 0.000 |
| timing | 0.125 | jepa_horizon | 0.3525 / 0.3766 | 0.5430 [0.5408, 0.5444] | 0.6491 / 0.6944 [0.1970, 0.8020] | 1–1 | 0.000 |
| timing | 0.125 | supervised | 0.4893 / 0.5049 | 0.6064 [0.6056, 0.6076] | 0.8068 / 0.8328 [0.4913, 0.8766] | 1–1 | 0.000 |
| timing | 0.25 | jepa_horizon | 0.3695 / 0.3881 | 0.5430 [0.5408, 0.5444] | 0.6804 / 0.7132 [0.2203, 0.8001] | 1–1 | 0.000 |
| timing | 0.25 | supervised | 0.4929 / 0.5154 | 0.6064 [0.6056, 0.6076] | 0.8129 / 0.8501 [0.4932, 0.9166] | 1–1 | 0.000 |
| volatility | 0.125 | jepa_horizon | 0.2344 / 0.2532 | 0.4940 [0.4358, 0.5518] | 0.4723 / 0.5416 [-0.0521, 0.7152] | 2–2 | 0.050 |
| volatility | 0.125 | supervised | 0.3970 / 0.4149 | 0.5166 [0.4614, 0.5715] | 0.7702 / 0.8213 [0.4668, 0.9238] | 2–2 | 0.000 |
| volatility | 0.25 | jepa_horizon | 0.2672 / 0.2759 | 0.4940 [0.4358, 0.5518] | 0.5404 / 0.5991 [0.2119, 0.8058] | 2–2 | 0.000 |
| volatility | 0.25 | supervised | 0.4118 / 0.4169 | 0.5166 [0.4614, 0.5715] | 0.7987 / 0.8371 [0.5092, 0.9487] | 2–2 | 0.000 |

The machine-readable version is `16_critical_budget_metrics.parquet`.

## Whitening-depth non-monotonicity diagnostic

This added diagnostic uses only frozen Phase-I recovery points. Within each
encoder/subsample cell, it first averages the paired supervised–horizon gap
over the decisive budgets (`0.125`, `0.25`), then applies 5,000-draw
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

`8→16` has a positive point change of `0.0008` and its interval includes zero; `16→32` has a positive point change of `0.0180` and its interval excludes zero; `32→64` has a negative point change of `-0.1131` and its interval excludes zero. The alternating point-estimate signs establish local non-monotonicity in the inspected grid. This diagnostic is post hoc and does not modify the
preregistered interpretation. Full paired values per encoder are retained in
the `15_whitening_nonmonotonicity_*` artifacts.

## Global covariance scale and regularization parity

On `last_concat512`, mean `trace_cov_over_dim` is `0.360919` for horizon-JEPA and `0.258515` for supervised, a ratio of `1.396123`. The two traces are not matched within the report's 10% diagnostic tolerance. The all-branch max/min ratio is `3.306248`.

Scientific common-regularization comparisons use the dimensionless parameter

`lambda = alpha * trace(covariance) / D`.

Figure 06 passed its source audit: every plotted cell comes from `ridge_raw_common_alpha`, and its axis is dimensionless alpha rather than absolute lambda. No fixed-absolute-lambda comparison is included; when trace
scales differ, such a comparison is confounded.

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

The historical reproduction gate is available and passed.

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
- normalized recovery is target-wise and only uses full-budget R² at least 0.01;
- the `R² ≥ 0.01` ceiling-eligibility rule is evaluated on full-budget test
  outcomes; it is part of metric definition, not validation-time hyperparameter
  selection.

The smallest observed fractional-budget `n/D` is `6.951`, so the original grid does not enter the `n/D < 1` regime.

## External-validity limits

- The dataset contains seven stocks from one market/domain.
- The historical split is stock-day-group-disjoint but not globally
  chronological. Within each stock, train days span almost the full calendar
  year and occur both before and after held-out validation/test days; this is
  not a forward-only temporal-generalization design.
- Validation and test are chronological halves of a historically explored
  held-out set, so the new test is not a pristine external confirmation set.
- Fractional budgets vary within-day endpoint coverage while retaining seven
  stock-day groups.
- Supervised pretraining used directional and volatility labels later probed by
  Experiment 01; timing was not a direct training target but may be correlated
  with those labels.

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
seeds within encoder. They are **computational-robustness intervals**, not
population-generalization confidence intervals. Grouped stock/day uncertainty
and leave-one-stock-out sensitivity remain pending. Companion tables expose
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
- `16_critical_budget_metrics.parquet`
- `15_whitening_nonmonotonicity_intervals.parquet`
- `15_whitening_nonmonotonicity_paired_differences.parquet`
- `15_whitening_nonmonotonicity_by_encoder.parquet`
- `15_whitening_nonmonotonicity_paired_by_encoder.parquet`

Phase II status: `complete`; Phase III-R status: `complete`. These later diagnostics do not change the frozen Phase-I outcome. This revision changes narrative and read-only report
diagnostics only; it does not change Phase-I results, thresholds, the technical
outcome or the fit pipeline.

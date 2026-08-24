# Experiment 01 — Predictability-dependent spectral allocation

## Status and sample

Preregistered decision: **fail**.

The diagnostic fits on 100,000 historical train endpoints and evaluates intrinsic predictability on 50,000 held-out endpoints. This is 2.048% of the 7,323,510 valid endpoints and 1.866% of all 8,039,246 dataset rows; no model, covariance or cross-covariance is fit on the full dataset.

Coverage is broad: all seven stocks; 1,527 train stock-days and 170 disjoint validation stock-days. Their union covers 99.941% (1,697/1,698) of valid stock-days; the omitted groups contain 4 valid endpoints. The train ratios are 125.0 observations/dimension for the 800-D raw window and 195.3 for the 512-D representation.

## Estimands

P_j is validation R² from trace-normalized linear ridge on the exact normalized raw K=20 LOB window. Alpha is selected separately per target by stock-day-grouped cross-validation inside train.

M_j is the fraction of full-rank predictive mass in the top 8 covariance directions. Its percentile uses 999 matched Haar subspaces and the same fractional mass estimand.

## Target-level relationship

| branch | seed | Spearman ρ(P,F) | ρ(P,null percentile) | low-P below null |
|---|---:|---:|---:|---:|
| jepa_horizon | 0 | 0.2475 | 0.2526 | 2 |
| jepa_horizon | 1 | 0.2132 | 0.2134 | 2 |
| jepa_horizon | 2 | 0.2059 | 0.2181 | 3 |
| jepa_masked | 0 | 0.3578 | 0.0476 | 0 |
| jepa_masked | 1 | 0.3554 | 0.1401 | 0 |
| jepa_masked | 2 | 0.3211 | 0.1029 | 0 |
| supervised | 0 | -0.1348 | -0.1937 | 0 |
| supervised | 1 | -0.1838 | -0.1853 | 0 |
| supervised | 2 | -0.1054 | -0.1362 | 0 |

| branch | mean ρ(P,F) | SD across seeds |
|---|---:|---:|
| jepa_horizon | 0.2222 | 0.0222 |
| jepa_masked | 0.3448 | 0.0206 |
| supervised | -0.1413 | 0.0396 |

The paired mean contrast is Δρ = ρ(JEPA-horizon) − ρ(supervised) = 0.3636.

## Intrinsic predictability

| target | family | P_raw_linear | selected alpha |
|---|---|---:|---:|
| time_to_next_mid_move | timing | 0.316357 | 0.01 |
| d_log_depth_top5@20 | depth | 0.120168 | 0.158489 |
| d_imbalance_top5@20 | imbalance | 0.115488 | 0.158489 |
| d_log_depth_top5@10 | depth | 0.106907 | 0.0630957 |
| d_imbalance_top5@10 | imbalance | 0.094079 | 0.158489 |
| d_log_depth_top5@5 | depth | 0.084280 | 0.158489 |
| d_imbalance_top5@5 | imbalance | 0.073584 | 0.158489 |
| d_imbalance_all@20 | imbalance | 0.072955 | 0.0630957 |
| d_log_depth_all@20 | depth | 0.068920 | 0.0251189 |
| d_log_depth_all@10 | depth | 0.066809 | 0.0251189 |
| d_imbalance_all@10 | imbalance | 0.066059 | 0.0630957 |
| d_log_depth_all@5 | depth | 0.061314 | 0.0251189 |
| d_imbalance_all@5 | imbalance | 0.056196 | 0.0630957 |
| d_log_depth_all@1 | depth | 0.053963 | 0.0251189 |
| d_log_depth_top5@1 | depth | 0.053146 | 0.158489 |
| d_imbalance_all@1 | imbalance | 0.040166 | 0.0630957 |
| d_imbalance_top5@1 | imbalance | 0.038506 | 0.158489 |

## Dependency and family sensitivity

The 17 targets are correlated and are not treated as 17 independent inferential units. Family medians (imbalance, depth, timing; n=3) are a conservative directional sensitivity check, not a standalone significance test.

| branch | seed | family-median ρ(P,F) |
|---|---:|---:|
| jepa_horizon | 0 | 0.5000 |
| jepa_horizon | 1 | 0.5000 |
| jepa_horizon | 2 | 0.5000 |
| jepa_masked | 0 | 0.5000 |
| jepa_masked | 1 | 0.5000 |
| jepa_masked | 2 | 1.0000 |
| supervised | 0 | 1.0000 |
| supervised | 1 | 1.0000 |
| supervised | 2 | 1.0000 |

## Preregistered gates

- rho_horizon_each_seed: fail
- delta_rho_mean: pass
- horizon_low_p_below_null: pass
- supervised_low_p_below_null: pass

## Interpretation boundary

A pass supports a predictability–spectral-allocation association and low-predictability anti-alignment in these frozen encoders. It does not identify a discontinuous SNR threshold, Bayes predictability, or a causal effect of the objective alone. The masked-JEPA arm is descriptive; the preregistered contrast is horizon-JEPA versus supervised.

These targets and the historical held-out split were previously used exploratorily. The analysis is therefore prospective after exploration, not pristine confirmatory evidence.

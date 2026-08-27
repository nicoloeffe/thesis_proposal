# Narrative summary — Experiment 01 Phase I

## Primary scientific result: specificity across three diagnostics

Conditional on the frozen representations and a newly fitted linear reader,
the supervised representation is more accessible at low reader-label budgets.
This is not an end-to-end label-efficiency claim because the supervised encoder
was itself trained with directional and volatility labels.

Phase II places only `0.000094` of horizon-JEPA's directional predictive mass in its first 8 PCs, versus `0.751775` for supervised. The fraction of Haar draws beating top-PCA averages `1.000` for horizon-JEPA and `0.000` for supervised across the recorded encoder seeds.

At full budget, changing `last_concat512 → meanK_concatS` changes directional test R² from `0.219943` to `0.070085` for horizon-JEPA and from `0.385349` to `0.394098` for supervised.

The Phase-I normalized finite-sample gaps are `0.546020` for
direction, `0.183807` for volatility and
`0.152758` for timing. The descriptive directional/control ratios are `2.97×` and `3.57×`. They are point summaries, not an independence-adjusted specificity test.

## Separate Phase-I effects

- The supervised-minus-horizon operational ceiling gap is `0.165405` with computational-robustness interval `[0.160753, 0.168857]`. The interval excludes zero.
- The mean normalized directional gap over the frozen low-budget grid is `0.546020`; over the recorded decisive budgets `0.125`, `0.25` it is `0.700972`.
- At `k_50gap=128`, whitening reduces the decisive-budget gap by `55.6%` but does not eliminate it. Non-robustness first appears at `k_nonrobust=508`; this is the maximum tested valid whitening depth. This pattern does not support concentration in only a few leading PCs.

## Secondary frozen technical classification

The preregistered Phase-I classification remains **A1**.
Nothing in this narrative revision changes its thresholds, result rows or
decision rule. The separate operational ceiling fact is reported as
**A1 with a robust ceiling gap**; `B` is not treated as a
coexisting outcome.

## Parity and scope

The historical reproduction gate is available and passed. Production full-budget scores and new-test min-norm
OLS remain separate because the new chronological test half is not required to
equal the old validation split. Phase II status: `complete`; Phase III-R status: `complete`. These later diagnostics do not change the frozen Phase-I outcome.

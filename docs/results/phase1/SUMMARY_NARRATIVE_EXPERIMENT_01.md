# Narrative summary — Experiment 01 Phase I

## Primary Phase-I result: reader-relative finite-sample accessibility

Conditional on the frozen representations and a newly fitted linear reader,
the supervised representation is more accessible at low reader-label budgets.
This is not an end-to-end label-efficiency claim because the supervised encoder
was itself trained with directional and volatility labels.

Phase II places only `0.000094` of horizon-JEPA's directional predictive mass in its first 8 PCs, versus `0.751775` for supervised. The fraction of Haar draws beating top-PCA averages `1.000` for horizon-JEPA and `0.000` for supervised across the recorded encoder seeds.

At full budget, changing `last_concat512 → meanK_concatS` changes directional test R² from `0.219943` to `0.070085` for horizon-JEPA and from `0.385349` to `0.394098` for supervised.

The Phase-I normalized finite-sample gaps are `0.546020` for
direction, `0.183807` for volatility and
`0.152758` for timing. The descriptive directional/control ratios are `2.97×` and `3.57×` on the normalized-recovery scale, versus `1.99×` and `2.35×` on the raw-R² scale. The magnitude is therefore scale-dependent. These are point summaries, not an independence-adjusted target-block interaction test.

## Separate Phase-I effects

- The supervised-minus-horizon operational ceiling gap is `0.165405` with computational-robustness interval `[0.160753, 0.168857]`. The interval excludes zero.
- The mean normalized directional gap over the frozen low-budget grid is `0.546020`; over the recorded decisive budgets `0.125`, `0.25` it is `0.700972`.
- At `k_50gap=128`, whitening reduces the decisive-budget gap by `55.6%` but does not eliminate it. At the historical technical field `k_nonrobust=508`, the gap no longer meets the compound preregistered criterion `lower > 0 and mean ≥ δ=0.10` at both decisive budgets; this is an effect-threshold transition, not a confidence interval crossing zero. The decisive-budget mean gaps are `0.035524`, `0.068723`, and their lower interval bounds remain positive (`0.000085`, `0.036869`). It is the maximum tested valid whitening depth. The mean decisive-budget gap is reduced by `92.6%` there. This pattern does not support concentration in only a few leading PCs.

## Frozen technical classification and mandatory sensitivity

At the preregistered primary threshold `δ=0.10`, the
frozen Phase-I classifier returns **A1**. This technical label
is secondary to the empirical accessibility result. The mandatory sensitivity
grid is:

| δ | technical class | k_50gap | k_nonrobust |
| --- | --- | --- | --- |
| 0.05 | D | 128 | — |
| 0.10 | A1 | 128 | 508 |
| 0.15 | A1 | 128 | 508 |

These sensitivity rows do not replace the primary threshold. They show that
the taxonomy label is threshold-sensitive even though the underlying gap curve
is unchanged. Nothing in this narrative revision changes thresholds, result
rows or the decision rule. The separate operational ceiling fact is reported as
**A1 with a robust ceiling gap**; `B` is not treated as a
coexisting outcome.

## Parity and scope

The historical reproduction gate is available and passed. Production full-budget scores and new-test min-norm
OLS remain separate because the new chronological test half is not required to
equal the old validation split. Phase II status: `complete`; Phase III-R status: `complete`. These later diagnostics do not change the frozen Phase-I outcome.

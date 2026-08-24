# Narrative summary — Experiment 01 Phase I

## Primary result: directional specificity across distinct diagnostics

Three separate diagnostics point to a direction-specific accessibility
penalty. First, the previously established spectral diagnostic shows
directional variance–task anti-alignment: at `m/D=1/32`, horizon-JEPA recovers
`0.0050` against a `0.0563` random-subspace null, whereas supervised recovers
`0.8971` against `0.6118`. Second, the previously established `last → meanK`
diagnostic shows directional pooling fragility; the production linear
full-budget check has horizon-JEPA `0.2199 → 0.0701`, while supervised is
approximately stable (`0.3853 → 0.3941`). Third, Phase I measures the normalized
finite-sample gap as `0.5460` for direction, `0.1838` for volatility and
`0.1528` for timing. The directional penalty is therefore approximately
`3–3.5×` larger than the controls (exact ratios `2.97×` and `3.57×`).

## Separate Phase-I effects

- The operational linear ceiling gap is robust: supervised minus horizon-JEPA
  is `0.1654` with hierarchical 95% interval
  `[0.1608, 0.1689]`.
- The normalized recovery gap is independently robust at adjacent low budgets;
  its six-low-budget mean is `0.5460`.
- Whitening mediates the second component: `k_50gap=128` halves but does
  not eliminate the gap, while non-robustness appears only at
  `k_nonrobust=508`,
  i.e. near-complete whitening. These results do not support concentration of
  the problem in a few leading PCs.

## Secondary technical classification

The unchanged preregistered classification is **A1 with a robust ceiling gap**.
`B` is not a coexisting outcome. The local
non-monotonicity diagnostic at `k=8,16,32,64` is post hoc and does not alter
this technical classification.

## Parity and scope

The historical reproduction gate passed (`0.211129` versus `0.2111` for
horizon-JEPA; `0.375636` versus `0.3756` for supervised). Production
full-budget scores and new-test min-norm OLS are reported separately because
the new chronological test split is not required to equal the old validation
split. Phase II and Phase III were not run.

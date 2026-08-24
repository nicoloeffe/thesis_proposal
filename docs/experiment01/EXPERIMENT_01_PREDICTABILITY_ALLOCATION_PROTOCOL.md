# Experiment 01 — Predictability-dependent spectral allocation

Status: **implemented, frozen and executed**. The preregistered outcome is
**`fail`**; this diagnostic does not alter the Phase I technical outcome A1 or
any Phase I–III result.

This diagnostic tests whether encoder spectral allocation across the 17 frozen
held-out targets is associated with an encoder-independent operational measure
of target predictability. It does not modify or rerun Experiment 01 Phase I,
II, or III.

## Deliberately fractional sample

The analysis is forbidden from fitting on the full dataset. Its frozen sample
contract is:

- 100,000 historical post-P0 train endpoints;
- 50,000 historical post-P0 held-out endpoints;
- 150,000 selected endpoints in total;
- 2.0482% of the 7,323,510 valid endpoints;
- 1.8658% of the 8,039,246 source rows.

The fraction is small computationally but broad in coverage. The audited split
contains all seven stocks, 1,527 train stock-days and 170 disjoint validation
stock-days. Their union covers 1,697 of 1,698 valid stock-days (99.9411%). The
only omitted group is `(stock_id=5, day_id=349)`, which contains four valid
endpoints. The train has 125 observations per dimension for the flattened
800-D raw input and 195.3 observations per dimension for the 512-D encoder
representation.

The gate rejects a run unless all of the following remain true:

- exact endpoint counts and hashes match the historical artifacts;
- the selected fraction is between 1% and 5% of valid endpoints;
- train and validation endpoints and stock-days are disjoint;
- every split contains all seven stocks;
- every stock contributes at least 10,000 train and 5,000 validation rows;
- train contains at least 1,000 stock-days and validation at least 100;
- selected endpoints cover at least 99% of valid stock-days;
- train supplies at least 100 observations per fitted dimension;
- no covariance, cross-covariance, PCA, or ridge fit uses unselected rows.

Each of the five raw-input cross-validation folds must also contain every stock,
at least 100 stock-days, and at least 10,000 rows.

The compressed source NPZ is opened to gather the exact K=20 windows at the
selected endpoints. This is source-data access, not a full-dataset fit.

## Estimands

For target `j`, intrinsic predictability `P_j` is validation R² from a
trace-normalized linear ridge reader fitted to the exact normalized flattened
K=20 LOB input. Alpha is selected separately per target with five-fold
stock-day-grouped cross-validation inside the 100,000-row train set. The
50,000-row historical validation set is evaluation-only.

For each branch and encoder seed, allocation is the fractional predictive mass

`F_j(8) = M_j(8) / M_j(D_valid)`.

The matched null uses 999 deterministic Haar subspaces of dimension eight and
the same fractional-mass estimand. A target is below the null when its smoothed
empirical percentile is at most 0.05.

The confirmatory contrast is horizon-JEPA versus supervised. Masked-JEPA is
descriptive only. `last_concat512` is the only readout in this diagnostic.

## Frozen decision rule

The following thresholds were approved and frozen before outcome computation:

- `rho(P,F) > 0.6` for horizon-JEPA in each encoder seed;
- mean paired `rho_horizon - rho_supervised > 0.3`;
- among the bottom third of targets by `P`, at least two below-null targets in
  horizon-JEPA for each seed;
- zero below-null low-P targets in supervised for each seed.

A target-level pass with positive family-median direction in all three
horizon-JEPA seeds is reported as `pass`. A target-level pass that does not
survive that conservative family-direction check is
`ambiguous_target_level_only`; otherwise the result is `fail`.

The 17 targets are correlated. Target-level rank correlations and within-family
results are reported, together with a three-family median sensitivity analysis.
The latter has `n=3` and is not assigned a standalone inferential p-value. An
effective-rank diagnostic is also emitted.

## Interpretation boundary

Even a pass supports only a predictability–allocation association and
low-predictability anti-alignment in the frozen encoders. It does not establish
a discontinuous SNR threshold, Bayes predictability, or a causal contribution
of the training objective alone. Because the targets and historical held-out
split have already appeared in exploratory analysis, the status is
prospective-after-exploration rather than pristine confirmatory evidence.

## Commands

Create a draft without reading outcomes:

```bash
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01_predictability_allocation draft \
  --out validation/experiment01/predictability_allocation/protocol_draft.json
```

Run the outcome-blind audit:

```bash
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01_predictability_allocation audit \
  --out validation/experiment01/predictability_allocation/input_audit.json
```

Freeze and input-bind the protocol (already completed for the canonical run):

```bash
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01_predictability_allocation freeze \
  --draft validation/experiment01/predictability_allocation/protocol_draft.json \
  --out validation/experiment01/predictability_allocation/protocol_frozen.json \
  --scientific-approver '<name>' \
  --approve-proposed-thresholds \
  --acknowledge-exploratory-status
```

Only a frozen protocol can be executed:

```bash
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01_predictability_allocation run \
  --protocol validation/experiment01/predictability_allocation/protocol_frozen.json \
  --out-dir validation/experiment01/predictability_allocation/run
```

## Canonical execution

The canonical execution completed in 17.45 seconds at:

`validation/experiment01/predictability_allocation_20260819/run`

It used exactly 100,000 train and 50,000 validation endpoints. Horizon-JEPA
Spearman correlations were `0.247549`, `0.213235`, and `0.205882`, so the
per-seed `rho > 0.6` condition failed. The mean paired delta versus supervised
was `0.363562`, and the low-predictability/null sign condition passed, but the
conjunction of frozen criteria therefore yielded `fail`. This rejects the
strong quantitative mechanism proposed here; it does not negate Phase I's
specificity result, Phase II's spectral localization, or Phase III-R's reader
accessibility result.

The frozen protocol SHA-256 is
`18e67f04dfa1e3418a333966ac2ce0629feb0670c50a48edf3800e38cb338ad1`.
The canonical run-manifest file SHA-256 is
`31d348cee4374a8ee7cdd29d6d578b60a99b5f0dabca2a374a991adecfc84e61`.

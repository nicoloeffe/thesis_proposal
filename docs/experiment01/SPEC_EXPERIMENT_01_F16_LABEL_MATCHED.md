# Experiment 01 — F16 label-matched supervised dose response

**Protocol date:** 2026-08-26
**Status:** preregistered before any F16 encoder training or F16 test access
**Scope:** diagnostic extension; frozen Phase I/II/III-R results are unchanged

## 1. Question and estimand

F16 tests how target-aligned supervised pretraining changes representation
geometry and fresh-reader accessibility as the number of labelled encoder
training examples increases.

The primary comparison is a newly trained budgeted `supervised` encoder versus
the frozen `jepa_horizon` encoder. Frozen `jepa_masked` is a secondary internal
control. The historical 500,000-row epoch-20 supervised checkpoint is a
descriptive upper anchor; it is not treated as if it had followed the new
stopping, row-identity or validation protocol.

F16 does not reclassify the frozen Phase-I technical outcome A1. It separates:

1. accessibility of a frozen representation to a fresh linear reader;
2. direct target co-adaptation during supervised encoder training;
3. dependence of representation geometry on supervised label volume.

## 2. Frozen inputs and barriers

The experiment uses only:

- `data/lobench_processed.npz`, SHA-256
  `7617dbbfcee56377f980a606267397861f6613017f0a2aca1e218407726ef862`;
- production bundle `validation/experiment01_bundle_20260730`, manifest SHA-256
  `bdded4ebd03c29d47e5dfdba106590f24763cc06bb7e6e5ea379eb4b34201c0b`;
- the nine canonical epoch-20 checkpoints in
  `CHECKPOINTS_MULTISEED_MANIFEST.json`;
- the exact Phase-I seed-0 labelled subset manifests;
- the fixed train/validation/test split already used by Experiment 01.

The checkpoint and hyperparameter audit is
[`TRAINING_PROTOCOL.md`](TRAINING_PROTOCOL.md), backed by
`TRAINING_PROTOCOL_AUDIT.json`. It establishes that historical same-seed arm
comparisons used identical 500,000-row train samples, but the three encoder
seeds changed the row sample as well as initialization and minibatch order.
F16 removes that confound.

### Test barrier

Before the barrier is unlocked, code may read test **row metadata only** to
construct and hash a target-blind cohort. It may not load test targets, test
features, test sufficient statistics or any result derived from them.

The barrier may be unlocked exactly once only after:

1. the specification, cohort manifest and production job inventory are frozen;
2. all 12 supervised runs have terminated or have a recorded terminal failure;
3. validation-selected checkpoint, reader alpha and other permitted selections
   are frozen and hash-pinned;
4. the failure table is frozen;
5. an unlock record names every selected artifact and its SHA-256.

No intermediate F16 test result may change training, cohort, checkpoint,
reader, whitening or reporting decisions.

## 3. Label budgets and accounting

Encoder training uses the exact Phase-I `subsample_seed=0` row identities:

| label | rows | canonical stock-days | stock-day equivalents |
|---|---:|---:|---:|
| `b_1_4` | 7,116 | 7 partial days | 1.75 |
| `b_1` | 28,446 | 7 full days | 7 |
| `b_4` | 122,099 | 28 full days | 28 |
| `b_16` | 490,937 | 112 full days | 112 |

The canonical stock-day identity is `(stock_id, trading_date)`. The four row
sets must be nested. Their exact Parquet bytes, row-key sequence, endpoint-index
sequence and source Phase-I files are hash-pinned by `f16_cohort_manifest.json`.
Any mismatch is terminal.

For every budget `L` and encoder seed in `{0,1,2}`:

- the supervised encoder receives gradients only from the exact labelled rows
  in `L`;
- its Axis-A fresh reader trains on exactly `L`;
- the frozen JEPA Axis-A reader trains on exactly `L`;
- Axis B uses the same fixed `b_16` reader-training reference for every encoder;
- validation labels are fixed overhead common to all budgets and are reported
  separately as `|V|` and `|L union V|`;
- target standardization is fit on `L` only and then reused across all three
  encoder seeds at that budget.

The primary estimand is incremental training-label efficiency conditional on a
fixed validation overhead. Frozen JEPA encoders had access to a larger
unlabelled pretraining corpus; a low-budget supervised collapse therefore
cannot distinguish label scarcity from feature-exposure or optimization limits.

## 4. Encoder, objective and deterministic training

The new supervised encoder preserves the canonical backbone and target
inventory:

- input window `K=20`, four role tokens per timestep, `d_model=128`;
- two-layer spatial and two-layer non-causal temporal transformers;
- attention-pooling head followed by a 22-output linear layer;
- the 20 future-feature targets at horizons `1,5,10,20` and realized
  volatility at `5,20`;
- train-only per-target z-scoring;
- equal-weight mean MSE across the 22 standardized targets.

Timing is not a direct supervised target. It remains a specificity control.

Frozen optimization values:

| property | value |
|---|---:|
| optimizer | AdamW |
| betas / epsilon | `(0.9,0.999)` / `1e-8` |
| initial LR | `3e-4` |
| weight decay | `1e-4` |
| gradient clip | global norm `1.0` |
| batch size | 256 |
| train loader | deterministic shuffle, `drop_last=False` |
| warm-up | 1,953 optimizer updates |
| LR decay | per-update cosine |
| terminal LR | `3e-6` at update 39,060 |
| maximum updates | 39,060 |
| validation cadence | every 500 updates, plus update 0 and epoch-20 sensitivity |
| minimum material improvement | validation MSE decrease `> 1e-4` |
| patience | 8 scheduled validation checks |

### Derivation of the maximum-update cap

The canonical 500,000-row trainer performed
`floor(500000/256) × 20 = 39,060` updates. The pre-F16 validation-only audit
found the supervised minima at epochs 20, 20 and 19 across seeds. Therefore a
10-epoch cap is not supported by the canonical validation histories, and the
20-epoch-equivalent 39,060-update cap is frozen. No F16 result was used to
choose it.

The historical scheduler advanced only between epochs. F16 intentionally makes
the schedule genuinely per update. This prospective correction is shared by
all 12 new runs; it is why the old epoch-20 checkpoint is only a descriptive
anchor.

### Checkpoint selection and stopping

Validation selection minimizes the equal-weight 22-target standardized MSE.
A candidate replaces the best checkpoint only when the decrease is strictly
greater than `1e-4`; ties and smaller changes retain the earliest update.

Early stopping is not eligible until both conditions hold:

1. at least 4,000 updates have completed;
2. the checkpoint after 20 complete passes through that budget has been saved.

The epoch-20-sensitivity update is therefore:

| budget | batches/pass | epoch-20 update |
|---|---:|---:|
| `b_1_4` | 28 | 560 |
| `b_1` | 112 | 2,240 |
| `b_4` | 477 | 9,540 |
| `b_16` | 1,918 | 38,360 |

After early stopping becomes eligible, eight consecutive scheduled checks
without a material improvement stop the run. At the maximum, the
validation-selected checkpoint remains primary even if it is earlier than the
last or epoch-20 checkpoint.

Each resumable checkpoint must include model, optimizer, scheduler, global
update, completed passes, target statistics, train/validation row hashes,
PyTorch/NumPy/Python RNG states and DataLoader generator state. Resume fails
closed if any manifest or source fingerprint differs.

Non-finite input, target, loss, parameter, gradient or validation metric is a
terminal numerical failure. A gradient norm above `1e6` before clipping is a
terminal explosion. An empty loader, missing target, row mismatch or checkpoint
reload mismatch is terminal. Failures are recorded and never silently retried
under altered hyperparameters.

## 5. Fixed union cohort and convergence gate

The experiment must not extract all endpoints for every new checkpoint. It
constructs a fixed union of:

1. all labelled train rows through `b_16`;
2. a target-blind covariance cohort spanning every train stock-day;
3. a validation cohort spanning every validation stock-day;
4. a sealed test cohort spanning every test stock-day.

Within each `(stock_id, trading_date)`, rows are ordered by the SHA-256 of the
domain-separated string
`experiment01-f16-cohort-v1 + NUL + split + NUL + row_key`. Candidate cohorts
take the first `min(cap, n_day)` rows. This makes candidates nested and
independent of targets and features.

Candidate endpoint caps per stock-day are fixed at `128,256,512,1024`.

### Convergence reference and cells

The benchmark uses only the already frozen `jepa_horizon` and `supervised`
checkpoints, encoder seeds `0,1,2`, and both fixed readouts. The reference uses
the complete production train covariance statistics, exact `b_16` reader rows
and complete validation split. No test array is opened.

For each candidate, compare against the reference:

| quantity | frozen tolerance |
|---|---:|
| directional full-rank validation R² | absolute error ≤ 0.020 |
| directional top-8 predictive mass | absolute error ≤ 0.020 |
| directional top-16 predictive mass | absolute error ≤ 0.020 |
| common/full role retention | absolute error ≤ 0.030 |
| contrast/full role retention | absolute error ≤ 0.030 |
| directional `last → meanK` R² gap | absolute error ≤ 0.020 |
| covariance trace | relative error ≤ 0.050 |
| cumulative explained variance at 8,16,32 | absolute error ≤ 0.020 |
| normalized leading-16 eigenvalue profile | L1 error ≤ 0.030 |

Only the 12 independent directional targets enter the primary convergence
scores; redundant bid/ask targets are excluded. Ridge uses the Phase-I
trace-normalized common-alpha grid, with alpha selected on complete validation
for the frozen reference and then held fixed while scoring candidate validation
cohorts.

A cap passes only if every required metric passes for every audited arm, seed
and applicable readout/pair. Select the smallest passing cap. If none passes,
cohort construction fails closed and no F16 encoder is trained; the candidate
grid and tolerances are not expanded after inspection.

PCA/covariance uses only feature rows from the target-blind train cohort. The
cross-covariance needed for predictive mass uses only the exact `b_16` labelled
reference, projected into that PCA basis. Cohort sampling never changes a
labelled budget.

## 6. Evaluation axes and permitted selection

### Axis A — label-matched end-to-end point

For an encoder trained on `L`, fit a fresh reader on the same `L`. Compare it
with readers on frozen JEPA encoders trained on that identical manifest. This
is the primary equal-incremental-label comparison.

### Axis B — fixed-reader representation diagnostic

For every budgeted encoder, fit the reader on fixed `b_16`. Use this axis for
ceilings, role projections, spectral geometry and pooling comparisons. Axis B
does not claim total label matching.

Reader alpha is selected on validation from the unchanged Phase-I alpha grid
using `lambda = alpha × trace(covariance) / D`. The primary tie break is the
largest alpha within `1e-12` of the best aggregate validation R². No absolute
lambda comparison is scientific.

Whitening uses only the already selected Phase-I bridge depths
`0,8,16,32,64,128,256,508`; F16 cannot select a new depth. `k_50gap=128` and
`k_nonrobust=508` remain frozen Phase-I diagnostics.

## 7. Outcomes and measurements

For each budget and encoder seed, primary outputs are:

- raw directional R² under `last_concat512`;
- Axis-A label-matched R²;
- Axis-B fixed-`b_16` R² and ceiling;
- common/full and contrast/full role retention, reported non-additively;
- top-8, top-16 and cumulative predictive mass;
- directional `last → meanK` loss.

Secondary outputs are:

- volatility and timing specificity controls;
- covariance spectrum and trace;
- the selected Phase-I whitening bridge;
- target-level estimates with redundant-target flags;
- stock and stock-day grouped uncertainty;
- leave-one-stock-out sensitivity;
- epoch-20 checkpoint sensitivity.

All primary summaries report the three encoder seeds separately and paired
across arms. Grouped intervals resample stocks first and stock-days within
stocks; they are descriptive with seven stock clusters. No target count is
treated as an independent sample size when targets are correlated.

## 8. Pre-registered interpretation matrix

The main object is the dose-response curve, not a new A/B/C/D outcome.

1. **Supervised-like at low label volume.** At `b_1_4` or `b_1`, a primary
   geometry metric is closer to the canonical supervised anchor than to the
   paired horizon anchor for all three seeds, and Axis-B directional R² is on
   the supervised side of the paired horizon/supervised midpoint.
2. **Smooth label-volume dependence.** For a metric oriented from the horizon
   anchor toward the canonical supervised anchor, Spearman correlation with
   ordered budget is at least `0.8` in all three seeds. Call the overall pattern
   smooth only if at least four of the six primary geometry/accessibility
   families satisfy this rule.
3. **Accessibility without measured geometry change.** Axis-A or Axis-B R² has
   the all-seed ordered pattern, but fewer than two of role retention, top-k
   mass, pooling loss and whitening bridge do.
4. **Low-budget optimization floor.** A cell is unstable if it terminates in a
   numerical failure or its best validation MSE does not beat the update-0
   value by `0.01`. An interpretation depending on an unstable budget is not
   made.
5. **Target-block heterogeneity.** Directionality-specific co-adaptation is
   supported only when the paired dose response is larger for directional than
   for both volatility and timing in all three seeds after ceiling scaling.

Mixed-seed or mixed-metric patterns are reported as heterogeneous, not forced
into a mechanism. These rules affect the F16 narrative only and never the
historical Phase-I A1 label.

## 9. Required artifacts and execution order

The production directory must contain:

```text
f16_manifest.json
f16_job_inventory.parquet
f16_training_curves.parquet
f16_checkpoint_manifest.json
f16_cohort_manifest.json
f16_cohort_convergence.parquet
f16_results.parquet
f16_geometry.parquet
f16_grouped_uncertainty.parquet
f16_failures.parquet
f16_summary.json
REPORT_EXPERIMENT_01_F16.md
```

Execution order is fixed:

1. record Git commit, dirty status, specification SHA-256 and input hashes;
2. freeze label/covariance/validation/test row manifests;
3. run the validation-only cohort convergence gate on existing checkpoints;
4. record the selected cap or fail closed;
5. run only `b_1_4`, seed 0 as a train/validation smoke and runtime benchmark;
6. verify target identity, deterministic reload/resume and no test access;
7. freeze the 12-cell job inventory;
8. train the remaining cells sequentially and resumably;
9. freeze validation-selected and epoch-20 checkpoint identities;
10. extract/process only the selected fixed union cohort;
11. run all train/validation selection and geometry analyses;
12. freeze selections and unlock test once;
13. run test, grouped uncertainty, summary, report and integrity audit.

The smoke benchmark does not authorize the 12-run grid automatically. Its
runtime, VRAM, RAM and storage estimate are reported first. Production requires
an explicit go/no-go decision, and the protocol may not be weakened merely to
fit a preferred runtime.

No Phase II or Phase III training, MLP, VICReg or simulator is part of F16.

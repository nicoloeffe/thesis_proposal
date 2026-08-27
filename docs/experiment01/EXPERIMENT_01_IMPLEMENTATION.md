# Experiment 01 — implementation and execution contract

This document describes the implementation of
[`SPEC_EXPERIMENT_01_SAMPLE_EFFICIENCY_20260730.md`](SPEC_EXPERIMENT_01_SAMPLE_EFFICIENCY_20260730.md)
version 2.0. Phase III is additionally governed by the later definitive
[`SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md`](SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md),
SHA-256
`78ca15821ac40355c35e5f40ecaf5086f5e6bbb6f339255a85b13fc7d952a151`.
That later document replaces the eligibility rule in the earlier optional MLP
section, so the executed Phase-III `b_1_4` floor is protocol-eligible.

## Current status

- dataset and reference-artifact equivalence gates: complete
  (`AUDIT_EXPERIMENT_01.md`);
- reference OLS reproduction gate: passed (`REPRODUCTION_GATE_EXPERIMENT_01.json`);
- Phase-I implementation: complete;
- canonical metadata/equivalence gate: passed, 8,039,246/8,039,246 rows;
- preregistered three-way split: complete;
- complete target extraction/equivalence gate: passed;
- feature pre-extraction gate: passed, 36/36 comparisons bit-identical;
- complete sharded feature extraction: complete, 18/18 feature sets and
  1,332/1,332 shards;
- production bundle preflight: passed, including full SHA-256 and finite-value
  scans;
- production Phase-I statistical grid, summary and corrected report: complete;
- Phase-II spectral diagnostic: complete, zero failures, Phase-I unchanged;
- original Phase-III 21,456-model design: stopped before selection freeze and
  before production-test access because it was computationally disproportionate;
- compute-feasible Phase-III-R: complete with frozen outcome `R3`, Phase-I
  technical outcome unchanged;
- T2 token-role matched-null diagnostic: complete, no privileged Hadamard axis;
- F16 checkpoint/training audit: complete and fail-closed;
- F16 target-blind cohort convergence: complete, cap 128 selected on 504/504
  passing cells without test-target or test-feature access;
- F16 production: complete, 12/12 training cells, 0 failures, 24 frozen
  best/epoch-20 checkpoints and 33/33 validation sufficient-statistic caches;
- F16 fixed test, grouped uncertainty, summary and report: complete; no
  selection changed after the one-time unlock;
- active software verification: 187 tests passed after F16 additions (one
  harmless empty-legend warning in a synthetic Phase-I plotting smoke test).

The authoritative current overview is
[`PROJECT_STATE.md`](../../PROJECT_STATE.md). This implementation contract
retains frozen reproduction commands and protocol details for auditability.

The seven original raw CSVs are present under `data/lobench/raw/`; their
SHA-256 hashes match the canonical audit file by file. Full reconstruction
from CSV is therefore available. The processed NPZ, canonical sidecars,
production bundle, checkpoints and frozen Phase-I–III outputs remain unchanged.

The sampled post-P0 dumps are used only as equivalence controls. The corrected
input is built from the full NPZ plus a CSV-derived metadata sidecar and the
preregistered three-way split described below.

## Files

```text
scripts/experiment01/run_experiment_01.py  CLI

experiment01/
  metadata.py              CSV sidecar + full numerical CSV↔NPZ gate
  split3.py                reference train + chronological held-out halves
  bundle.py                complete rows/target shards + storage estimate
  sharded.py               verified row-addressable sharded NPY arrays
  extraction.py            benchmarks, readout equivalence, resumable extraction
  schema.py                fail-closed three-way input contract
  subsets.py               nested stock-day/fractional-day budgets
  linear.py                sufficient statistics, OLS, ridge, whitening
  pipeline.py              streaming Phase-I runner and compute log
  results.py               operational ceilings and hierarchical summaries
  summary.py               gaps, curve quantities and A1/A2/B/D assignment
  reporting.py             Phase-I figures and Markdown report
  reproduction.py          read-only validation and OLS reproduction gate
  phase2_reproduction.py   frozen PCA-ladder reproduction gate
  training_audit.py       nine-checkpoint protocol and row-identity audit
  f16.py                  frozen label/cohort row manifests and test barrier
  f16_convergence.py      validation-only cap-selection gate
  f16_training.py         deterministic resumable supervised training
  f16_planning.py         post-pilot inventory and compute/storage bounds
  reference/               frozen extraction and ladder equivalence stack

tests/test_experiment01.py synthetic compliance/regression tests
```

## Required input bundle

The runner accepts only a directory with `manifest.json` using:

```text
schema_name    = thesis.experiment01.input
schema_version = 1
```

It requires:

1. provenance fields declaring the corrected post-P0 source commit, dataset
   SHA-256, split fingerprint and target-manifest fingerprint;
2. exactly three fixed, complete splits: `train`, `validation`, `test`;
   plus explicit verified flags for context-window boundaries, target-horizon
   boundaries and row/feature/target alignment under `K=20`,
   `max_horizon=20`;
3. one Parquet row table per split with:

   ```text
   row_key
   stock_id
   stock_symbol
   stock_day_id
   trading_date
   endpoint_index
   endpoint_order
   timestamp_ns
   ```

4. complete canonical endpoint order `0..n-1` within every stock-day;
5. globally disjoint row keys and stock-days;
6. complete sharded target NPY arrays plus explicit target/block/independence
   metadata;
7. the exact timing semantic:
   `log1p_observed_or_capped_all_rows:max_look=600`;
8. one logical float32 array, composed of ordered NPY shards, per exact
   `branch × encoder_seed × readout × split`;
9. exactly the canonical feature inventory and readout definitions;
10. file SHA-256, byte size, shape, dtype and row-key SHA-256 for every array.

Every shard records path, byte size, SHA-256, row interval, shape, dtype and
row-key SHA-256. Shards cover each logical array exactly once with no gaps or
overlap. Phase I presents them through a row-addressable array and processes
them sequentially, so none of the 18 complete matrices is loaded at once.

The preflight rejects missing timestamps, inferred symbols, incomplete days,
row-key mismatches, stock-day leakage, NaN/Inf, stale hashes, unexpected seeds
or modified pooling definitions.

## Corrected metadata and split

Canonical metadata is read directly from the seven CSVs, never inferred from
the processed NPZ. The sidecar contains:

```text
global_row_index, stock_id, stock_symbol, timestamp_ns, trading_date,
day_id, endpoint_order, raw_csv_row_index
```

The canonical stock-day identity is `(stock_id, trading_date)`.

The equivalence gate compares every reconstructed `book`, `mid_z`,
`stock_ids` and `day_ids` value against `lobench_processed.npz`. It passed with
zero mismatches and identical array hashes.

The complete historical `grouped_split_by_stock_day.v1` assignment is rebuilt
with `split_seed=0` and `val_frac=0.1`. All 1,528 historical training
stock-days remain train. Within each stock, the 170 historical held-out days
are ordered chronologically: the first `floor(n/2)` become validation and the
remainder become test. This produces:

| split | stock-days | endpoints |
|---|---:|---:|
| train | 1,528 | 6,596,688 |
| validation | 83 | 352,931 |
| test | 87 | 373,891 |

Validation and test therefore derive from the previous held-out set, already
used in historical exploratory analyses. The Experiment 01 test portion is
nevertheless forbidden for alpha, whitening-k or any other hyperparameter
selection.

## Pre-extraction gates and storage

- 23 targets: the canonical 20 directional, 2 volatility and capped timing
  target;
- historical target comparison: bit-identical for sampled train/held-out;
- feature comparison: bit-identical for all 9 checkpoints, both readouts and
  all 100,000/50,000 historical sampled endpoints;
- storage estimate: 269.97 GB features, 0.67 GB targets, 297.71 GB required
  with 10% headroom;
- physical feature layout: 1,332 NPY shards (74 per logical matrix), with
  100,000 rows per full shard;
- completed bundle size: 270,781,826,562 bytes (253 GiB reported by `du`);
- final bundle manifest SHA-256:
  `bdded4ebd03c29d47e5dfdba106590f24763cc06bb7e6e5ea379eb4b34201c0b`;
- measured free storage before extraction: 634.16 GB;
- benchmark: 14,075 rows/s on one stock-day and 21,241 rows/s on one complete
  stock using `supervised_seed0_ep020`;
- observed complete feature-extraction wall time: approximately 61 minutes;
- observed full production preflight wall time: approximately 7 minutes.

## Phase-I implementation

### Label budgets

`subsets.py` implements:

- `b ∈ {1/8, 1/4, 1/2, 1, 2, 4, ...}`;
- `balanced_max` when distinct;
- exactly one deterministic `full_train`;
- independent within-stock day permutations;
- one first day shared by fractional levels and `b=1`;
- contiguous nested fractional blocks around one reproducible uniform anchor;
- ceiling rounding of an unavoidable fractional endpoint count;
- 10/5/3/1 adaptive subsampling seeds;
- opening/middle/closing sensitivity manifests and result table.

Every subset is serialized with stable row keys and a row-key SHA-256.

### Linear readers

The implementation uses additive float64 statistics:

```text
n, sum(X), sum(Y), XᵀX, XᵀY, YᵀY
```

Nested budgets add only new stock-day rows. One eigendecomposition per
design/transform evaluates the full 32-value alpha grid:

```text
alpha = 0 + 31 log-spaced values from 1e-8 to 1e4
lambda = alpha * trace(covariance) / D
```

Raw coordinates use labelled-subset centering/equivalent unpenalized
intercepts and never coordinate-wise standardization. Alpha selection maximizes
the canonical mean validation R² over fixed independent targets. Exact
machine-equal ties choose the larger alpha; this documents the deterministic
fallback because the repository has no existing one-standard-error convention.

Min-norm OLS uses alpha zero, a symmetric eigensolver/pseudoinverse and a
recorded machine-precision rank tolerance. There is no hidden shrinkage.

### Whitening

For every exact feature set, mean/covariance and eigensystem are fit once from
all unlabelled train features. The cached transform metadata is reused at every
label budget.

The requested grid is:

```text
0, 1, 2, 4, 8, 16, 32, 64, 128, 256, D_valid
```

No scientific eigenvalue floor is applied. A request beyond numerical rank is
an explicit invalid cell. Every transformed labelled design recomputes its
trace scale before converting alpha to lambda.

### Evaluation and result handling

- validation and test remain fixed and complete;
- whitening never uses validation/test features;
- alpha selection uses validation only;
- common-alpha configurations are fixed by preregistration;
- test metrics are not used for model selection;
- constant targets receive explicit invalid status;
- all targets, including redundant directional copies, remain in raw rows;
- normalized recovery uses the exact full-budget protocol/configuration;
- full-budget R² below 0.01 is ineligible and never divided;
- negative low-budget recoveries are not clipped;
- directional blocks with fewer than two eligible independent targets are
  marked non-interpretable;
- hierarchical intervals resample encoder seeds, then subsampling seeds;
- within-encoder and between-encoder standard deviations remain separate.

The runner writes:

```text
results.parquet
failures.parquet
time_of_day_sensitivity.parquet
metadata.json
subset_manifest.json
subset_manifests/
time_of_day_sensitivity_subsets/
transforms/
```

The summarizer and reporter add the required uncertainty tables, gap
sensitivities, curve quantities, figures, `summary.json` and
`REPORT_EXPERIMENT_01.md`.

## Commands

Use the repository's ROCm environment:

```bash
# Build and fully verify CSV-derived metadata
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 build-sidecar \
  --raw-dir data/lobench/raw \
  --dataset data/lobench_processed.npz \
  --out-dir validation/experiment01_inputs_20260730/sidecar

# Preregister historical train + chronological held-out halves
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 build-three-way-split \
  --sidecar-dir validation/experiment01_inputs_20260730/sidecar \
  --dataset data/lobench_processed.npz \
  --reference-split validation/readouts_v2_20260728/split.npz \
  --out-dir validation/experiment01_inputs_20260730/split3

# Write complete rows/target shards and the storage estimate
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 prepare-bundle \
  --split-dir validation/experiment01_inputs_20260730/split3 \
  --dataset data/lobench_processed.npz \
  --reference-dir validation/readouts_v2_20260728 \
  --out-dir validation/experiment01_bundle_20260730

# Mandatory benchmarks and feature-equivalence gate
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 preextract-gate \
  --bundle validation/experiment01_bundle_20260730 \
  --reference-dir validation/readouts_v2_20260728 \
  --device cuda

# Resumable checkpoint/split/shard extraction
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 extract-features \
  --bundle validation/experiment01_bundle_20260730 \
  --device cuda

# Verify the historical post-P0 battery and display blockers
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 audit-reference \
  --in-dir validation/readouts_v2_20260728

# Recompute the historical free OLS gate
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 reproduce \
  --in-dir validation/readouts_v2_20260728 \
  --out validation/experiment01/reproduction_gate_recomputed.json

# Validate the completed production bundle
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 preflight \
  --bundle validation/experiment01_bundle_20260730

# Inspect/serialize all label subsets before compute
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 subsets \
  --bundle /absolute/path/to/experiment01_input \
  --out-dir validation/experiment01/subsets_review

# Phase I
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 run-phase1 \
  --bundle /absolute/path/to/experiment01_input \
  --out-dir validation/experiment01/phase1

# Hierarchical uncertainty and outcome
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 summarize \
  --results validation/experiment01/phase1/results.parquet \
  --out-dir validation/experiment01/phase1/summary

# Figures and report
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 report \
  --results validation/experiment01/phase1/results.parquet \
  --summary-dir validation/experiment01/phase1/summary \
  --out-dir validation/experiment01/phase1/report
```

## Tests

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
../rocm_env/bin/python -m pytest -q
```

The tests cover deterministic/nested group sampling, fractional anchors,
adaptive replication, split and row identity, direct-vs-Gram solvers,
min-norm rank handling, dimensionless lambda, progressive whitening,
constant/shuffled targets, ceiling eligibility, unclipped recovery,
hierarchical variance decomposition, bundle preflight and a streaming Phase-I
smoke run.

## Production gate — passed

The corrected three-way rule supplied after the initial audit is
preregistered. The following fail-closed production conditions have all
passed:

1. all 18 logical feature arrays have complete verified shard coverage;
2. `manifest.json` has `status=complete`;
3. the full bundle preflight rechecks hashes, row identities, finite values and
   exact feature inventory.

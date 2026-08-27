# Experiment 01 Phase III — historical pre-implementation audit

> **Document scope.** This is the frozen pre-execution audit. Its statements
> about result availability describe that point in the execution timeline, not
> the current repository state. Phase III-R subsequently completed; see
> `REPORT_EXPERIMENT_01_PHASE3.md` for the current result and classification.
>
> **Protocol version.** The governing document is the later definitive
> [`SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md`](https://github.com/nicoloeffe/thesis_proposal/blob/main/docs/experiment01/SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md),
> SHA-256
> `78ca15821ac40355c35e5f40ecaf5086f5e6bbb6f339255a85b13fc7d952a151`.
> It replaces the eligibility rule in the earlier optional MLP section; hence
> `b_min_mlp = b_1_4` is not an ineligible executed budget.

Date: 2026-08-01  
Repository commit at audit start: `6a94bd5890539037a00fa7f635776707ed647183`  
Protocol status: Phase I and Phase II are frozen; Phase-I technical outcome `A1` is unchanged.

## Audit scope and ordering

This document was created before adding or modifying Phase-III production code. It records the historical reader semantics, frozen-artifact identities, subset eligibility, and the acceptance gates that must pass before production MLP training.

The requested project-state file `STATO_TESI_POST_PHASE2_20260731.md` was not present in the repository or the inspected user workspace at audit time. The audit therefore uses the frozen Phase-I and Phase-II reports, summaries, manifests, source, and the definitive Phase-III specification. This absence does not authorize inference or weakening of any artifact identity.

The working tree was already substantially dirty at audit start, with pre-existing deleted and untracked research artifacts. Those changes are treated as user-owned and are not restored or overwritten. Phase III writes only to its new output directory plus explicitly named Phase-III source/tests.

## Frozen artifact identity ledger

The following identities were read from the existing frozen artifacts before Phase-III implementation:

| Artifact | SHA-256 |
|---|---|
| production bundle `manifest.json` | `bdded4ebd03c29d47e5dfdba106590f24763cc06bb7e6e5ea379eb4b34201c0b` |
| Phase-I `results.parquet` | `ecf4e410c595baa32d06a1998bbd5151794d02ff141499af3c1f56268e110ffb` |
| Phase-I technical `summary/summary.json` | `7978961be69e50881ac022a67bfd7fea4f619c9806374121b57d6d4cbac1d4a6` |
| Phase-I `subset_manifest.json` | `3a4412bf110706f685b1502c64ba277a9b3129cf3c64aa77ccd53eefe5ef1471` |
| Phase-II `phase2_results.parquet` | `a0a3a5ea609f8347af0ce29c2ef4fd000cd847ecf24b9df544d7b1f600449fb3` |
| Phase-II `predictive_mass.parquet` | `fecf798a2cf042d5a872a5d173c9710849e038d016a00d93eea6beff86dd6727` |
| Phase-II `manifest.json` | `1a30b67f6739a1a0440eae1866ee55f72cddf94248e5edf336a7e605461144c2` |
| Phase-II `summary.json` | `bfc2c9f000d85d1555f3004bad73aa08728d20f54e9f28c86ba31f8a159a432e` |

The complete production-bundle traversal passed before implementation: 18 feature sets, all recorded encoder checkpoints and feature shards, 23 targets, and 6,596,688/352,931/373,891 train/validation/test rows were verified. It took 385.03 seconds and peaked at 4.25 GB resident RAM. The 166 Phase-II artifacts and five recorded Phase-II source files also matched their size and SHA-256 records. All 78 Phase-I subset files matched their size and SHA-256 records; independently recomputed row-key hashes and row counts matched every subset record. A future mismatch remains a hard stop.

## Historical post-P0 MLP reconstruction

Canonical implementation: `ladder_accessibility.py::mlp_ceiling`  
Source SHA-256: `a34c8574b2914efa25c9677f1b404f23ebf8dec579fe1bf914455d220711ddd6`

### Architecture

The historical MLP was:

```text
Linear(d, 256)
GELU
Linear(256, 256)
GELU
Linear(256, T)
```

It used neither dropout, BatchNorm, nor LayerNorm. It is not the one-hidden-layer Phase-III primary architecture.

### Input and target preprocessing

The historical Stage-1 train sample was deterministically permuted with `split_seed=0` and split into 90% reader-train and 10% internal validation. Both the coordinate-wise input mean/standard deviation and target mean/standard deviation were fitted on that 90% reader-training subset. Near-constant coordinate or target standard deviations were replaced by one. The 50,000-row outer validation set was transformed with those statistics and used only for the reported historical score.

Coordinate-wise input standardization was therefore present historically. This is an intentional difference from the Phase-III native arm, which permits only subtraction of the mean fitted once on all unlabelled production-train features. Phase III also forbids hidden normalization.

### Optimizer, regularization, and model selection

- optimizer: AdamW;
- learning rate: `1e-3`;
- weight decay: `1e-4`;
- batch size: `4096`;
- maximum epochs: `80`;
- internal-validation patience: `10` epochs;
- improvement rule: any strictly lower internal standardized-target MSE;
- checkpoint: best internal-validation loss;
- gradient clipping: absent;
- reader seeds: `0,1,2,3,4`, with a shared split and seed-specific initialization/order.

The historical run did not select over a hyperparameter grid. It did not use the production Phase-III validation/test split.

### Split and target inventory

The historical Stage-1 artifacts contain 100,000 train rows and 50,000 outer-validation rows, with `split_seed=0`. One joint MLP was fitted to 22 raw targets: 20 directional columns and two realized-volatility columns. The reported directional aggregate keeps 12 independent columns (`d_spread_z`, `d_microprice_rel`, and `d_top_imbalance` at horizons 1, 5, 10, and 20) and drops the eight algebraically redundant bid/ask-relative columns from aggregation. Timing was held out and is not part of the two historical reference scores.

### Frozen historical reference

| Branch/readout/block | mean R2 | encoder SD | reader SD | mean best epoch |
|---|---:|---:|---:|---:|
| `jepa_horizon/last_concat512/directional` | 0.3191358981 | 0.006436 | 0.004462 | 68.8667 |
| `supervised/last_concat512/directional` | 0.3880910480 | 0.000433 | 0.003656 | 24.4667 |

Historical table identities:

- `mlp_agg.csv`: `bb39df20bd28bb11c9fa59ed33af643d5d77f2fc68ae66d4b7644790e41ec8a5`;
- `mlp_reader_runs.csv`: `52b661f57f7ce9bc815da92166738ee9b11b6bc288bb9e57f4c2927af562b0bc`;
- historical analysis manifest: `9ae4fb10374f4e5175cc846388bdd7b082422c66f9c50bc6803a6f1cad27316c`.

No saved historical MLP predictions or checkpoints are present. The acceptance gate must therefore retrain the exact historical reader and reproduce each aggregate within absolute tolerance `0.015`, reporting reader/encoder dispersion. Until that succeeds, Phase-III production training is blocked.

## Phase-III reader boundary

The new primary reader is intentionally different and must remain exactly:

```text
Linear(d, 256, bias=True)
GELU
Dropout(0.10)
Linear(256, T, bias=True)
```

Native input receives only the frozen all-unlabelled-train mean subtraction. Full whitening uses the frozen train-only valid-rank transform with dimension 508. Target standardization is fitted independently on the exact labelled Phase-I subset. AdamW uses `lr=1e-3`, weight decay selected from `{0,1e-5,1e-3}`, global gradient clipping at 5.0, and the preregistered step-based stopping rule. Validation performs checkpoint and weight-decay selection; test access remains blocked until the selection manifest is frozen and hashed.

## Exact subset eligibility and preregistered grids

The Phase-I subset inventory gives a minimum row count below 4096 at `b_1_8` (3,559) and above 4096 at `b_1_4` (7,116). Therefore:

```text
b_min_mlp = b_1_4 = 0.25
```

No intermediate subset will be generated.

- primary directional grid: every existing Phase-I level from `b_1_4` through `full_train`, including `balanced_max`;
- low-budget set: `b_1_4`, `b_1_2`, `b_1`, `b_2`, `b_4`;
- volatility/timing grid: `b_1_4`, `b_1`, `b_4`, `balanced_max`, `full_train`;
- spectral grid: `b_1_4`, `b_4`, `full_train`;
- capacity sensitivity: widths 128 and 512 at `b_1_4` and `full_train`, directional/native, both primary branches.

All cells reuse the exact Phase-I subset files and row-key hashes. Subsampling seeds and reader seeds are distinct identity axes.

## Pre-production stop state

After the identity traversal passed, production MLP training remained blocked pending all of the following:

1. frozen Phase-I branch-whitening derived table;
2. implementation and full tests for the native/whitened reader, selection boundary, streaming, and restartability;
3. synthetic nonlinear and anisotropic-conditioning gates;
4. historical stochastic reproduction gate;
5. full-budget linear row/hash parity and exact PCA-band identity;
6. serialization and inspection of the complete job inventory.

No Phase-III result or R1/R2/R3/R4 classification is available at this stage.

## Acceptance-gate closure and compute inventory

All pre-production gates subsequently passed without accessing the production test split:

- synthetic nonlinear gate: linear R2 `0.000261`, Phase-III MLP R2 `0.991556`;
- synthetic conditioning gate: native R2 `-0.019069`, train-only-whitened R2 `0.986712`, isotropic reference `0.986984`;
- frozen full-budget linear parity: 90 native plus 90 full-whitened target rows joined one-to-one;
- PCA identity: exact four-band union of PCs 1:508 and top-128/full-rank projection parity for both branches and all encoder seeds;
- historical stochastic reproduction: horizon-JEPA `0.319563` versus `0.319136`, supervised `0.390844` versus `0.388091`, both within `0.015`;
- complete repository test suite: 150 passed.

The protocol was frozen before production test access with SHA-256 `39e94319804ab6b52fb29bac8124e683210735c3e5ee662f877ed4c53fe150da`.

The exact inventory contains 2,796 logical validation-selection cells, 8,388 weight-decay candidate fits, and 13,068 independent evaluation fits: 21,456 trained models in total. A two-cell validation-only benchmark measured 6.92 seconds per 1,000 steps in native coordinates and 6.56 seconds after full whitening. Peak VRAM was 0.43 GB and peak system RAM was 3.09 GB. The strict all-model lower bound is approximately 40.2 hours if every model stops at 1,000 steps; the all-20,000-step bound is approximately 803.7 hours. These are compute bounds, not a post-hoc protocol change.

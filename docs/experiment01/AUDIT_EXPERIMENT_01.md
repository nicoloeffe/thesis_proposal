# Experiment 01 — integrity audit

- **Audit status:** complete
- **Frozen experiment version:** 2.0
- **Dataset rows:** 8,039,246
- **Canonical encoder inventory:** 3 objectives × 3 seeds

This document records the integrity gates that bind the published Experiment
01 results to their data, stock-day split, frozen encoders and production
bundle. It is an audit of identity and equivalence, not a scientific outcome.

## 1. Canonical inputs

The processed dataset is built by
`scripts/dataset/build_encoder_dataset_lobench.py` from seven source CSVs under
`data/lobench/raw/`. The builder applies the declared market-time and collapsed
book filters, assigns `stock_id` in source-file order and concatenates the
filtered instruments deterministically.

The metadata sidecar is read directly from the source CSVs and contains, in the
same global order as `data/lobench_processed.npz`:

```text
global_row_index
stock_id
stock_symbol
timestamp_ns
trading_date
day_id
endpoint_order
raw_csv_row_index
```

The canonical stock-day identity is `(stock_id, trading_date)`, and
`endpoint_order` is consecutive within every filtered stock-day.

## 2. CSV ↔ processed-dataset equivalence

The fail-closed gate passed for all 8,039,246 rows. It verifies:

- global and per-stock row counts;
- `stock_ids` and `day_ids`;
- numerical equality of `book` and `mid_z`;
- source-symbol and timestamp provenance;
- identical global order after the builder filters.

Canonical processed-dataset SHA-256:

```text
7617dbbfcee56377f980a606267397861f6613017f0a2aca1e218407726ef862
```

Raw-source SHA-256 inventory:

| symbol | SHA-256 |
|---|---|
| `sz000001` | `cfc88e926c06b87f7e82506ec0973d07afde838d1b949353c21a6c7ab049842b` |
| `sz000002` | `eaf43ffda67970fb467e38fdc0984784a94c2e141f1e90c9525d18fef77e3465` |
| `sz000651` | `527e082a61f30f42e4ce5ec117cb2d99f42b3eeb6798de4f8237a9d2b14fea59` |
| `sz000858` | `d9ad8f2f341e3868c59bcc1e382e761038ea3ddb86c1f28c89c59f8ef136b14f` |
| `sz002415` | `2c801af4e923e3abf1bc2fec35ddbc9289027e9ceb2f95d17d61975cba60073a` |
| `sz300147` | `60bfb8fee288b028f773b389066696ed18878d3e1c26ffeffdd9636738f97062` |
| `sz300750` | `7ed3d0b250871c19fb5829a4270777c028a6847845e264dfbef9541bf25ac938` |

## 3. Split integrity

The three-way split is defined over complete stock-days. All days assigned to
the encoder-training side of the reference split remain in train. Within each
stock, held-out days are ordered by `trading_date`; the first half is assigned
to validation and the second half to test, with an odd remainder assigned to
test.

The manifest verifies:

- complete enumeration of stock-days;
- train/validation/test disjointness;
- no encoder-training day in test;
- temporal ordering within each held-out half;
- exact row-key and endpoint-index hashes;
- validation-only hyperparameter selection.

Validation and test derive from a held-out group previously exposed to
exploratory analysis. The frozen Experiment 01 test is not used to select
alpha, whitening depth, reader capacity or any other hyperparameter.

## 4. Encoder and readout identity

The canonical inventory contains the epoch-20 checkpoint for each combination
of:

```text
objective ∈ {supervised, jepa_horizon, jepa_masked}
seed      ∈ {0, 1, 2}
```

All nine checkpoint identities and SHA-256 hashes are declared in
[CHECKPOINTS_MULTISEED_MANIFEST.json](CHECKPOINTS_MULTISEED_MANIFEST.json).
Their shared stock-normalization statistics have SHA-256:

```text
0938cfda578710436220d0a31fd5d6101c496dafa1512784a4dddef3a44f7900
```

The two fixed 512-dimensional readouts are:

- `last_concat512 = grid[:, -1, :, :].reshape(B, 512)`;
- `meanK_concatS = grid.mean(dim=1).reshape(B, 512)`.

The pre-extraction gate reproduced all 36 available checkpoint/readout/split
reference matrices bit-for-bit before production extraction.

## 5. Production bundle

The bundle is sharded and row-addressable, so complete feature matrices do not
need to coexist in memory. It contains exactly:

```text
3 objectives × 3 seeds × 2 readouts × 3 splits = 54 logical feature arrays
```

The complete extraction produced 1,332 ordered shards. The final preflight
verified file hashes, sizes, shapes, dtypes, finite values, row-key alignment,
target alignment, stock-day completeness and split isolation.

Production bundle manifest SHA-256:

```text
bdded4ebd03c29d47e5dfdba106590f24763cc06bb7e6e5ea379eb4b34201c0b
```

## 6. Reproduction and phase gates

- Full-rank min-norm OLS reference reproduction: passed.
- Phase I production preflight and fixed-test isolation: passed.
- Phase II PCA-ladder reproduction: 3,960 cells passed with maximum absolute
  error below the frozen tolerance.
- Phase I ↔ Phase II full-rank parity: passed for all feature/target cells.
- Phase III-R prerequisite, selection-isolation and fixed-test gates: passed.
- Predictability-allocation protocol binding and fractional-sample audit:
  passed; its scientific decision is reported separately as `fail`.

Technical failure count in the completed production analyses: zero.

## 7. Verification boundary

The tracked reports, figures, metadata and checksums permit scientific review
without the large local arrays. Exact recomputation additionally requires the
raw or processed dataset, the nine checkpoint archive and sufficient storage
for the production bundle. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

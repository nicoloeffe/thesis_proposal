# LOB representation experiments

Repository for the LOBench representation-learning study and the frozen
Experiment 01 analyses. The obsolete top-level `legacy/` tree was removed on
2026-08-24; active code has no import dependency on it.

The authoritative operational and scientific status is
[PROJECT_STATE.md](PROJECT_STATE.md). Historical reports remain available for
audit, but they should not be used as current workspace inventories.

External reviewers should start with the tracked
[results package](docs/results/README.md) and
[reproducibility guide](docs/REPRODUCIBILITY.md).

## Active layout

```text
experiment01/                         Experiment 01 implementation
experiment01/historical/              corrected post-P0 reproduction stack
models/model_tokenizer_t.py           shared encoder architecture
training/                             active encoder training entrypoints
scripts/dataset/                      canonical processed-dataset builder
scripts/evaluation/                   evaluation and orchestration utilities
scripts/experiment01/                 Experiment 01 command-line entrypoints
tests/                                active regression/compliance suite
docs/                                 protocols, historical snapshots and notes
```

`experiment01/legacy.py` and `experiment01/phase2_legacy.py` are deliberately
retained historical-reproduction modules. Their names describe protocol
compatibility; they do not refer to the deleted directory.

`training/historical/` contains checkpoint-compatible pre-fix definitions used
by historical evaluation scripts. Current training entrypoints remain directly
under `training/`.

## Current frozen assets

| asset | path | status |
|---|---|---|
| canonical raw CSVs | `data/lobench/raw/` | present, 7 files, hashes verified |
| processed dataset | `data/lobench_processed.npz` | present, 8,039,246 rows |
| production bundle | `validation/experiment01_bundle_20260730` | complete, about 253 GiB |
| Phase I–III outputs | `validation/experiment01/execution_20260730` | complete |
| P→M diagnostic | `validation/experiment01/predictability_allocation_20260819/run` | complete |
| 3×3 encoder checkpoints | `checkpoints/multiseed` | present |

The seven original raw CSVs are present in `data/lobench/raw/`. Their SHA-256
hashes match the historical audit exactly, so CSV→NPZ/metadata reconstruction
is available again. No frozen dataset, sidecar, bundle or result was regenerated
when the raw sources were restored.

## Verification

Use the repository ROCm environment:

```bash
../rocm_env/bin/python -m pytest -q
```

Current result after the repository reorganization and artifact packaging:
**165 passed**.

Inspect the production bundle without rerunning analysis:

```bash
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 preflight \
  --bundle validation/experiment01_bundle_20260730
```

Audit the historical post-P0 artifacts:

```bash
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 audit-historical \
  --in-dir validation/readouts_v2_20260728
```

The old command name `audit-legacy` remains an alias for reproducibility.

## Principal reports

- [Phase I](docs/results/phase1/REPORT_EXPERIMENT_01.md)
- [Phase II](docs/results/phase2/REPORT_EXPERIMENT_01_PHASE2.md)
- [Phase III-R](docs/results/phase3r/REPORT_EXPERIMENT_01_PHASE3.md)
- [Predictability-allocation diagnostic](docs/results/predictability_allocation/REPORT_EXPERIMENT_01_PREDICTABILITY_ALLOCATION.md)

The nine canonical multiseed checkpoints are described by a tracked
[manifest](docs/experiment01/CHECKPOINTS_MULTISEED_MANIFEST.json) and packaged
as a deterministic 84-MB release artifact. The redundant 210-file training
directory is not committed to Git.

## Snapshot and recovery

The state immediately before removal of `legacy/` is recoverable from:

```text
commit  f77dc7468b10fdb7bc7272d41fcc49348341e80a
tag     project-snapshot-2026-08-24-experiment01
```

Bulk ignored artifacts are not duplicated in Git; their canonical hashes are
recorded in manifests and in
[PROJECT_SNAPSHOT_20260824.md](docs/history/PROJECT_SNAPSHOT_20260824.md).

The documentation index is [docs/README.md](docs/README.md).

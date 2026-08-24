# Reproducibility and artifact availability

This document separates what is versioned in Git from the large scientific
artifacts required for full recomputation. The canonical scientific status is
summarized in [`PROJECT_STATE.md`](../PROJECT_STATE.md).

## Versioned in Git

- source code and tests;
- the frozen Experiment 01 specification and implementation contract;
- reference parity and checkpoint manifests with SHA-256 identifiers;
- publication copies of the Phase I, Phase II and Phase III-R reports,
  summaries and figures under `docs/results/`.

## Software environment

The completed experiments used:

```text
Python          3.12.3
PyTorch         2.9.1+rocm6.3
HIP             6.3.42134-a9a80e791
NumPy           2.2.4
Pandas          2.2.3
SciPy           1.15.2
scikit-learn    1.6.1
PyArrow         24.0.0
Matplotlib      3.10.1
```

For CPU inspection and tests:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m pytest -q
```

Production GPU recomputation requires a PyTorch build matching the local ROCm
stack. Installing `requirements.txt` from the default Python index is sufficient
for CPU tests but does not guarantee GPU/ROCm compatibility.

## Data tiers

| tier | local path | size | Git status | role |
|---|---|---:|---|---|
| raw source | `data/lobench/raw/` | 6.76 GB | ignored | seven canonical LOBench CSVs |
| processed dataset | `data/lobench_processed.npz` | 162 MiB | ignored | 8,039,246 filtered rows |
| production bundle | `validation/experiment01_bundle_20260730/` | 253 GiB | ignored | complete sharded features/targets |
| complete outputs | `validation/experiment01/` | about 6.1 GiB | ignored | full result tables and diagnostics |
| publication results | `docs/results/` | about 3.7 MiB | tracked | reports, summaries and figures |

The raw data and production bundle are deliberately excluded from Git. Their
canonical hashes, row counts and provenance are recorded in `PROJECT_STATE.md`
and the Experiment 01 documents. Data redistribution must follow the terms of
the original LOBench source.

## Encoder checkpoints

Experiment 01 uses exactly nine epoch-20 checkpoints: three arms by three
encoder seeds. They total 84,199,395 bytes. The full local 210-file training
directory is not a canonical distribution artifact because it includes every
intermediate epoch plus redundant `best` and `last` aliases.

The nine files and their hashes are defined in
[`CHECKPOINTS_MULTISEED_MANIFEST.json`](experiment01/CHECKPOINTS_MULTISEED_MANIFEST.json).
Packaging and verification instructions are in
[`CHECKPOINTS.md`](experiment01/CHECKPOINTS.md).

## Verification boundary

The tracked reports make the scientific conclusions reviewable without large
downloads. Recomputing the frozen representations requires the nine checkpoint
archive plus the processed dataset. Rebuilding the production bundle from zero
additionally requires the seven canonical raw CSVs. Re-running the complete
statistical phases requires the 253-GiB production bundle or equivalent compute
and storage to regenerate it.

# Scripts

## Dataset

`dataset/build_encoder_dataset_lobench.py` is the canonical CSV-to-NPZ builder.
The seven verified inputs live outside Git under `data/lobench/raw/`.

## Experiment 01

`experiment01/` contains the Phase I, Phase II, original Phase III, reduced
Phase III-R and predictability-allocation command-line entrypoints. Invoke them
as modules from the repository root, for example:

```bash
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 --help
```

The frozen reference extraction and ladder utilities live under
`experiment01/reference/` in the source package, where Phase I–III import them
as explicit equivalence and reproduction gates.

Exploratory one-off probes and superseded execution managers are intentionally
excluded from the publication package.

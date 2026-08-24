# Scripts

## Dataset

`dataset/build_encoder_dataset_lobench.py` is the canonical CSV-to-NPZ builder.
The seven verified inputs live outside Git under `data/lobench/raw/`.

## Evaluation

`evaluation/` contains exploratory probes and the shell managers used by the
stopped and reduced Phase-III executions.

## Experiment 01

`experiment01/` contains the Phase I, Phase II, original Phase III, reduced
Phase III-R and predictability-allocation command-line entrypoints. Invoke them
as modules from the repository root, for example:

```bash
../rocm_env/bin/python -m scripts.experiment01.run_experiment_01 --help
```

The corrected pre-Experiment-01 extraction and ladder utilities live under
`experiment01/historical/` in the source package, where Phase I–III can import
them as explicit reproduction gates.

Scripts importing `training.historical` are compatibility analyses for pre-fix
checkpoints; they are not current training code.

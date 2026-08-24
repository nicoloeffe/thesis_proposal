# Experiment 01 Phase III-R — final report

## Compute-feasible preregistered amendment

Phase III v1 was terminated before selection freeze and before any production
test access because its 21,456-model inventory was computationally
disproportionate. Phase III-R was frozen before test with 1,296 models: 504
primary, 576 specificity-control and 216 focused spectral fits. It preserves
all scientific semantics and outcome thresholds while reducing replication to
three paired encoder, subset and reader seeds where applicable. The capacity
sweep and redundant spectral arms are explicitly outside Phase III-R.


## Preregistered outcome

The directional `last_concat512` primary outcome is **R3: persistent difficulty beyond linearity and second-order conditioning**. Phase-I technical outcome **A1 remains frozen and unchanged**. Phase III changes only the reader-relative diagnosis, not the Phase-I result.

The frozen native-ridge low-budget normalized gap is 0.6975. The native-MLP gap is 1.5301, giving reader attenuation -1.1937. The full-whitened MLP gap is 4.9594, giving within-MLP whitening attenuation -2.2412.

## Frozen Phase-I/II facts

- Phase-I `A1` and all Phase-I thresholds/results are unchanged.
- Phase II localized directional signal deeply along the covariance spectrum; predictive mass remains a linear covariance diagnostic.
- The production bundle, Phase-I subsets, Phase-I transforms, Phase-II caches and canonical checkpoints passed their hash gates.
- The historical MLP gate reproduced horizon-JEPA and supervised within absolute tolerance 0.015; that historical reader used coordinate-wise standardization and is not the Phase-III primary reader.

## New Phase-III reader result

The primary reader is exactly `Linear(d,256)-GELU-Dropout(0.10)-Linear(256,T)`, with no coordinate-wise native input standardization, BatchNorm or LayerNorm. Weight decay was selected on the fixed validation split. The selection manifest was frozen and hashed before one-shot test inference.

Encoder-specific native directional gaps are `{"0": 1.2768411854630035, "1": 1.7423459793309604, "2": 1.5712472605087682}`. Encoder-specific full-whitened gaps are `{"0": 4.692356626927102, "1": 5.2814637547446335, "2": 4.9044829429059655}`. Meaningful-ceiling status is `True`.

## Conditioning and reader decomposition

Phase III separates: (1) the operational full-budget ceiling of each reader, (2) finite-sample recovery relative to its own target-wise ceiling, (3) dependence on the invertible train-only whitening transform, and (4) dependence on enlarging the reader class from frozen ridge to the preregistered MLP. The reader-by-conditioning interaction is descriptive and does not change R3.

## Target specificity

Directional, volatility and timing results are reported separately. They are never pooled. Volatility and timing are specificity controls; the preregistered outcome is directional only.

- `directional/full_whitened`: mean normalized gap 4.9594
- `directional/native`: mean normalized gap 1.5301
- `timing/full_whitened`: mean normalized gap 2.0578
- `timing/native`: mean normalized gap 1.0293
- `volatility/full_whitened`: mean normalized gap 1.1727
- `volatility/native`: mean normalized gap 0.4864

## Spectral diagnostics

The focused nonlinear spectral contrast is restricted to
`jepa_horizon` directional targets: head PCs 1:127, deep PCs 382:508 and the
full valid rank. Detailed intermediate-band, supervised and timing spectral
localization remains the frozen Phase-II result; Phase III-R does not repeat
it. The MLP does not “recover predictive mass.”

- `jepa_horizon/directional`: head 1:127 0.3497, deep 382:508 0.0200, full-rank 0.3704

The cross-fitted spectral control was not invoked by the reduced report; the frozen Phase-II controls remain unchanged.

## Secondary ceiling statement

Full-budget MLP ceiling gaps, nonlinear lift over frozen ridge, MLP-to-supervised ratios and target-specific ceiling eligibility are reported in `phase3_ceiling_and_lift.parquet`. They are operational reader results, not Bayes-content estimates.

- `jepa_horizon/directional/full_whitened`: ceiling 0.3609, lift 0.1408, supervised ratio 0.9119
- `jepa_horizon/directional/native`: ceiling 0.3448, lift 0.1248, supervised ratio 0.8602
- `jepa_horizon/timing/full_whitened`: ceiling 0.5657, lift 0.0411, supervised ratio 0.9264
- `jepa_horizon/timing/native`: ceiling 0.5641, lift 0.0210, supervised ratio 0.9131
- `jepa_horizon/volatility/full_whitened`: ceiling 0.5362, lift 0.0421, supervised ratio 0.9938
- `jepa_horizon/volatility/native`: ceiling 0.5022, lift 0.0082, supervised ratio 0.9325
- `supervised/directional/full_whitened`: ceiling 0.3946, lift 0.0092, supervised ratio 1.0000
- `supervised/directional/native`: ceiling 0.3986, lift 0.0132, supervised ratio 1.0000
- `supervised/timing/full_whitened`: ceiling 0.6107, lift 0.0105, supervised ratio 1.0000
- `supervised/timing/native`: ceiling 0.6178, lift 0.0114, supervised ratio 1.0000
- `supervised/volatility/full_whitened`: ceiling 0.5393, lift 0.0227, supervised ratio 1.0000
- `supervised/volatility/native`: ceiling 0.5383, lift 0.0217, supervised ratio 1.0000

## Capacity sensitivity

Capacity sensitivity was removed by the frozen compute-feasible amendment. No
architecture sweep was used to select or redefine the width-256 primary
reader.

## Limitations and prohibited interpretations

Equal MLP performance would not prove equal information, and a persistent MLP gap would not prove information loss. Full whitening is a post-hoc train-only invertible coordinate transform, not a training-time encoder intervention. No claim is made that VICReg/SIGReg must reproduce it, that top-128 failure proves tail causality, or that these results generalize beyond this domain.

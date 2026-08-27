# Experiment 01 Phase III-R — final report

## Compute-feasible preregistered amendment

Phase III is governed by the later definitive specification
[`SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md`](https://github.com/nicoloeffe/thesis_proposal/blob/main/docs/experiment01/SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md)
(SHA-256 `78ca15821ac40355c35e5f40ecaf5086f5e6bbb6f339255a85b13fc7d952a151`).
It replaces the eligibility rule in the earlier optional MLP section; the
executed `b_1_4` floor is protocol-eligible.

Phase III v1 was terminated before selection freeze and before any production
test access with status `terminated_pre_test_compute_infeasible` and
`0` test-inference claims. The recorded
reason is: observed production runtime made the 21,456-model grid disproportionate to the diagnostic question. Phase III-R was frozen before
test with 1296 models:
504 primary,
576 specificity-control and
216 focused spectral fits. It
preserves the frozen outcome thresholds while reducing replication to the
protocol-recorded paired seeds. The capacity sweep and omitted spectral arms
remain outside Phase III-R.


## Preregistered outcome

The directional `last_concat512` primary outcome is **R3: the frozen gap criterion persists for the selected MLP after whitening**. Phase-I technical outcome **A1 remains frozen and unchanged**. Phase III changes only the reader-relative diagnosis, not the Phase-I result. For R3 specifically, “persists” refers only to the frozen selected MLP family and transforms; it is not a claim about nonlinear readers in general.

The frozen native-ridge low-budget normalized gap is 0.6975. The native-MLP gap is 1.5301, giving reader attenuation -1.1937. The full-whitened MLP gap is 4.9594, giving within-MLP whitening attenuation -2.2412.

The two attenuation quantities above are algebraic outputs of the frozen
classification rule, not stable effect-size estimates: low-budget raw R² is
negative in many primary cells. Raw scores and ceiling eligibility therefore
come before the normalized-gap interpretation.

## Raw and normalized budget metrics

The table below accompanies normalized recovery with the underlying raw test
R² distribution, operational ceiling range, eligibility count and fraction of
negative raw scores. It is generated from `phase3_results.parquet` and is also
serialized as `phase3_report_metrics.parquet`.

| block | budget | branch | transform | raw R² mean/median [range] | recovery mean/median [range] | ceiling mean [range] | eligible | negative raw fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| directional | b_1_2 | jepa_horizon | full_whitened | -1.378/-1.455 [-2.595, -0.266] | -4.028/-3.880 [-8.897, -0.629] | 0.361 [0.247, 0.438] | 12 | 1.000 |
| directional | b_1_2 | jepa_horizon | native | -0.175/-0.108 [-1.262, 0.178] | -0.514/-0.304 [-3.483, 0.478] | 0.345 [0.213, 0.422] | 12 | 0.731 |
| directional | b_1_2 | supervised | full_whitened | -0.033/-0.006 [-0.302, 0.142] | -0.120/-0.015 [-1.068, 0.310] | 0.395 [0.283, 0.466] | 12 | 0.528 |
| directional | b_1_2 | supervised | native | 0.317/0.315 [0.127, 0.432] | 0.788/0.799 [0.337, 0.938] | 0.399 [0.285, 0.468] | 12 | 0.000 |
| directional | b_1_4 | jepa_horizon | full_whitened | -2.416/-2.503 [-4.630, -0.453] | -7.117/-6.527 [-18.779, -1.070] | 0.361 [0.247, 0.438] | 12 | 1.000 |
| directional | b_1_4 | jepa_horizon | native | -0.387/-0.259 [-2.511, 0.056] | -1.126/-0.844 [-6.930, 0.173] | 0.345 [0.213, 0.422] | 12 | 0.966 |
| directional | b_1_4 | supervised | full_whitened | -0.416/-0.383 [-0.861, -0.106] | -1.106/-0.948 [-2.976, -0.228] | 0.395 [0.283, 0.466] | 12 | 1.000 |
| directional | b_1_4 | supervised | native | 0.255/0.250 [0.076, 0.400] | 0.632/0.658 [0.187, 0.855] | 0.399 [0.285, 0.468] | 12 | 0.000 |
| directional | full_train | jepa_horizon | full_whitened | 0.361/0.372 [0.244, 0.438] | 1.000/1.000 [0.978, 1.015] | 0.361 [0.247, 0.438] | 12 | 0.000 |
| directional | full_train | jepa_horizon | native | 0.345/0.361 [0.209, 0.423] | 1.000/1.000 [0.971, 1.021] | 0.345 [0.213, 0.422] | 12 | 0.000 |
| directional | full_train | supervised | full_whitened | 0.395/0.400 [0.283, 0.467] | 1.000/1.000 [0.990, 1.006] | 0.395 [0.283, 0.466] | 12 | 0.000 |
| directional | full_train | supervised | native | 0.399/0.406 [0.285, 0.468] | 1.000/1.000 [0.991, 1.006] | 0.399 [0.285, 0.468] | 12 | 0.000 |
| timing | b_1_4 | jepa_horizon | full_whitened | -1.654/-1.642 [-2.543, -0.945] | -2.926/-2.926 [-4.533, -1.666] | 0.566 [0.561, 0.569] | 1 | 1.000 |
| timing | b_1_4 | jepa_horizon | native | -0.316/-0.078 [-1.348, 0.211] | -0.558/-0.139 [-2.376, 0.378] | 0.564 [0.560, 0.568] | 1 | 0.556 |
| timing | b_1_4 | supervised | full_whitened | -0.530/-0.448 [-1.120, -0.284] | -0.868/-0.734 [-1.834, -0.467] | 0.611 [0.609, 0.612] | 1 | 1.000 |
| timing | b_1_4 | supervised | native | 0.291/0.312 [0.077, 0.402] | 0.472/0.505 [0.124, 0.651] | 0.618 [0.617, 0.619] | 1 | 0.000 |
| timing | full_train | jepa_horizon | full_whitened | 0.566/0.568 [0.558, 0.571] | 1.000/0.999 [0.994, 1.008] | 0.566 [0.561, 0.569] | 1 | 0.000 |
| timing | full_train | jepa_horizon | native | 0.564/0.566 [0.553, 0.571] | 1.000/1.000 [0.988, 1.012] | 0.564 [0.560, 0.568] | 1 | 0.000 |
| timing | full_train | supervised | full_whitened | 0.611/0.611 [0.607, 0.615] | 1.000/1.000 [0.996, 1.004] | 0.611 [0.609, 0.612] | 1 | 0.000 |
| timing | full_train | supervised | native | 0.618/0.618 [0.615, 0.622] | 1.000/1.001 [0.993, 1.005] | 0.618 [0.617, 0.619] | 1 | 0.000 |
| volatility | b_1_4 | jepa_horizon | full_whitened | -0.846/-0.738 [-1.700, -0.257] | -1.611/-1.536 [-3.424, -0.434] | 0.536 [0.479, 0.592] | 2 | 1.000 |
| volatility | b_1_4 | jepa_horizon | native | -0.069/-0.112 [-0.816, 0.344] | -0.123/-0.223 [-1.460, 0.664] | 0.502 [0.446, 0.559] | 2 | 0.611 |
| volatility | b_1_4 | supervised | full_whitened | -0.228/-0.268 [-0.460, 0.022] | -0.439/-0.458 [-0.947, 0.037] | 0.539 [0.486, 0.595] | 2 | 0.963 |
| volatility | b_1_4 | supervised | native | 0.193/0.176 [0.067, 0.344] | 0.363/0.337 [0.114, 0.654] | 0.538 [0.482, 0.595] | 2 | 0.000 |
| volatility | full_train | jepa_horizon | full_whitened | 0.536/0.536 [0.478, 0.595] | 1.000/1.000 [0.994, 1.005] | 0.536 [0.479, 0.592] | 2 | 0.000 |
| volatility | full_train | jepa_horizon | native | 0.502/0.502 [0.439, 0.562] | 1.000/1.000 [0.983, 1.018] | 0.502 [0.446, 0.559] | 2 | 0.000 |
| volatility | full_train | supervised | full_whitened | 0.539/0.539 [0.484, 0.596] | 1.000/1.000 [0.997, 1.002] | 0.539 [0.486, 0.595] | 2 | 0.000 |
| volatility | full_train | supervised | native | 0.538/0.538 [0.480, 0.598] | 1.000/1.000 [0.994, 1.006] | 0.538 [0.482, 0.595] | 2 | 0.000 |

## Frozen Phase-I/II facts

- Phase-I `A1` and all Phase-I thresholds/results are unchanged.
- Phase II localized directional signal deeply along the covariance spectrum; predictive mass remains a linear covariance diagnostic.
- The production bundle, Phase-I subsets, Phase-I transforms, Phase-II caches and canonical checkpoints passed their hash gates.
- The historical MLP gate reproduced its recorded branches within absolute tolerance 0.015; that historical reader used coordinate-wise standardization and is not the Phase-III primary reader.

## New Phase-III reader result

The primary reader is exactly `Linear(d,256)-GELU-Dropout(0.10)-Linear(256,T)`, with no coordinate-wise native input standardization, BatchNorm or LayerNorm. Weight decay was selected from `[0.0, 1e-05, 0.001]` on the fixed validation split. The selection manifest was frozen and hashed before one-shot test inference.

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

These normalized-gap summaries are descriptive. In particular, values above
one are differences between normalized recoveries and must not be read as
“times worse.” A target-block interaction requires grouped stock-day
uncertainty, which is not present in the current aggregate artifacts.

## Spectral diagnostics

The focused nonlinear spectral contrast is restricted to
`jepa_horizon` directional targets with arms
`band_1_127, band_382_508, full_valid_rank`. Detailed intermediate-band,
supervised and timing spectral
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

Equal MLP performance would not prove equal information, and a persistent MLP
gap does not prove information loss or a general nonlinear-accessibility
mechanism. Full whitening is a post-hoc train-only invertible coordinate
transform, not a training-time encoder intervention. No claim is made that
VICReg/SIGReg must reproduce it or that head-band failure proves tail
causality.

The existing intervals resample encoder, subset and reader seeds and therefore
measure computational robustness, not population generalization. Grouped
stock/day uncertainty and leave-one-stock-out sensitivity remain pending. The
dataset contains seven stocks from one market/domain; the historical split is
not globally chronological, validation and test derive from a historically
explored held-out set, and the test is not a pristine external confirmation
set. The supervised encoder also saw directional and volatility labels during
pretraining, so this phase cannot support an end-to-end label-efficiency claim.

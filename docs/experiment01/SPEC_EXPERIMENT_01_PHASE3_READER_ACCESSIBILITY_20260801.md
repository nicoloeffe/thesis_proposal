# Experiment 01 — Phase III: reader-relative and conditioning-relative accessibility

**Version:** 1.0 — preregistered implementation specification
**Date:** 2026-08-01
**Status:** definitive Phase-III specification; Phase I and Phase II are frozen
**Primary implementation agent:** Codex, after mandatory repository and artifact audit

---

## 0. Purpose and scientific status

Experiment 01 Phase III is the final empirical phase before the definitive mathematical formalization of the thesis.

Phase I established that, for directional targets under `last_concat512`, `jepa_horizon` has:

1. a lower full-budget linear operational ceiling than `supervised`;
2. a robust additional finite-sample recovery gap after normalizing each branch by its own ceiling;
3. a gap that is strongly reduced by progressive whitening and becomes non-robust only under almost-complete whitening.

Phase II established that this effect is not generic compression. Inside the same `jepa_horizon` representation and the same leading principal components, the top-eight cumulative predictive mass is approximately:

| target block | top-8 predictive mass |
|---|---:|
| directional | **0.0001** |
| volatility | 0.5601 |
| timing | 0.8141 |

For `jepa_horizon` directional targets, 100/100 matched Haar subspaces outperform top-PCA at both `k=8` and `k=16`, separately for every encoder seed. The transition from PCA underperformance to PCA superiority occurs as directional predictive mass accumulates deeper in the spectrum.

Phase III must answer the remaining empirical question:

> **How much of the observed accessibility penalty is specific to the linear reader class, and how much persists for a nonlinear reader after covariance conditioning is removed?**

This phase does **not**:

- retrain encoders;
- modify Phase-I or Phase-II results;
- reclassify the frozen technical outcome `A1`;
- claim to measure Bayes content;
- test VICReg, SIGReg or other training-time interventions;
- construct a simulator;
- search broadly over neural architectures.

The experiment is a controlled reader study on frozen representations.

---

# 1. Frozen empirical starting point

The following are accepted as established and must not be re-estimated as competing outcomes.

## 1.1 Directional specificity

For `jepa_horizon/last_concat512`, the leading spectral directions expose timing and volatility but almost no directional signal. This within-encoder comparison is the primary empirical fact motivating Phase III.

## 1.2 Linear finite-sample penalty

On the canonical three-way production bundle, tuned full-rank linear readers show:

- a robust directional ceiling gap;
- a robust low-budget normalized-recovery gap;
- a directional penalty approximately 2.97 times the volatility penalty and 3.57 times the timing penalty.

## 1.3 Conditioning mediation

Progressive whitening reduces the directional low-budget gap by at least 50% at `k=128` and makes it non-robust only at the maximum valid depth `k=508`.

This supports a distributed covariance-mediated effect. It does not prove that the whole phenomenon is linear, nor that whitening changes the abstract information content.

## 1.4 Pooling fragility

`C_pooling` remains frozen:

- `jepa_horizon`, directional MLP: `last 0.3191 -> meanK 0.0494`;
- `supervised`, directional MLP: `last 0.3881 -> meanK 0.3917`.

Phase III uses `last_concat512` as its primary readout. It does not reopen the pooling classification.

## 1.5 Structural rank

The numerical covariance rank is `508 = 4 × 127`, matching the maximum dimension implied by four independently LayerNorm-constrained 128-dimensional tokens. Directions outside the valid numerical rank must never be inverted or treated as scientific features.

---

# 2. Scientific questions

## 2.1 Primary reader-relativity question

For frozen `supervised` and `jepa_horizon` representations, does a nonlinear MLP recover each branch's own full-budget directional ceiling more rapidly than the tuned linear reader?

For reader family `q`, branch `r`, target `t` and labelled stock-day budget `g`, report:

\[
R^2_{q,r,t}(g)
\]

and target-wise normalized recovery:

\[
A_{q,r,t}(g)
=
\frac{R^2_{q,r,t}(g)}{R^2_{q,r,t}(\mathrm{full})}.
\]

The branch-level normalized gap is:

\[
G_q(g)
=
\overline{A}_{q,\mathrm{sup}}(g)
-
\overline{A}_{q,\mathrm{hor}}(g),
\]

where the overline is the canonical average over eligible independent targets and the hierarchical seed structure.

The primary comparison is:

\[
G_{\mathrm{MLP,native}}(g)
\quad\text{versus}\quad
G_{\mathrm{ridge,native}}(g),
\]

with the ridge curve read unchanged from frozen Phase I.

## 2.2 Conditioning-by-reader interaction

Does full whitening still improve sample efficiency when the reader is nonlinear?

The primary 2×2 design is:

| reader | native coordinates | full whitening |
|---|---|---|
| tuned ridge | frozen Phase I | frozen Phase I, `k=508` |
| MLP | new Phase III | new Phase III |

This separates:

1. changing reader class at fixed native coordinates;
2. changing conditioning at fixed reader class;
3. interaction between reader nonlinearity and conditioning.

## 2.3 Nonlinear ceiling question

At full labelled budget, how much additional operational signal is recovered by the MLP relative to tuned ridge?

Define the nonlinear lift:

\[
L_{r,t}
=
R^2_{\mathrm{MLP},r,t}(\mathrm{full})
-
R^2_{\mathrm{ridge},r,t}(\mathrm{full}).
\]

This is an operational reader-family comparison, not a Bayes-content estimate.

## 2.4 Target-specificity question

Is reader relativity itself target-specific?

Directional is the primary block. Volatility and timing are preregistered controls and remain separate. No global average across the three blocks is allowed.

## 2.5 Spectral necessity question

For `jepa_horizon`, does a nonlinear reader require spectral bands outside the leading subspace to recover directional performance?

This is a secondary diagnostic. It must not alter the primary Phase-III outcome.

## 2.6 Whitening-target interaction question

The frozen Phase-I bridge reports only supervised-minus-JEPA gaps. Before MLP training, decompose the whitening effect by branch:

\[
D_{r,t}(k,g)
=
A_{\mathrm{ridge},r,t}^{(k)}(g)
-
A_{\mathrm{ridge},r,t}^{(0)}(g).
\]

This determines whether partial whitening directly helps or harms each branch, rather than attributing a gap change to `jepa_horizon` without decomposition.

---

# 3. Scope

## 3.1 Mandatory primary scope

- branches: `supervised`, `jepa_horizon`;
- encoder seeds: `0,1,2`;
- readout: `last_concat512`;
- target block: directional, 12 canonical independent targets;
- readers/transforms:
  - native MLP;
  - full-whitened MLP;
  - frozen native tuned ridge;
  - frozen full-whitened tuned ridge;
- labelled subsets: exact Phase-I nested stock-day manifests;
- evaluation: canonical fixed validation and test splits.

## 3.2 Mandatory specificity controls

Repeat the primary native/full-whitened MLP comparison for:

- volatility, 2 targets;
- timing, 1 target.

These controls may use a reduced budget grid defined in Section 7. They must never be pooled with directional.

## 3.3 Mandatory spectral diagnostic

For `supervised` and `jepa_horizon`, directional and timing, use equal-dimensional disjoint PCA bands within the valid 508-dimensional eigenspace:

- `band_1_127`: PCs 1–127;
- `band_128_254`: PCs 128–254;
- `band_255_381`: PCs 255–381;
- `band_382_508`: PCs 382–508.

Each band has exactly 127 dimensions and therefore the same MLP parameter count under the fixed architecture.

Also include:

- `full_valid_rank`: all 508 valid PCA coordinates;
- `top_128`: PCs 1–128, for direct comparability with `k_50gap=128`.

Band results are diagnostic and must be described as such.

## 3.4 Explicitly excluded from the outcome-deciding grid

- `jepa_masked`;
- `meanK_concatS` learning curves;
- raw 20-target directional duplication in the training loss;
- arbitrary learned pooling;
- architecture sweeps beyond the preregistered sensitivity;
- new encoder training;
- VICReg/SIGReg;
- simulator experiments.

`jepa_masked` and `meanK_concatS` may appear only in optional full-budget descriptive checks after the primary report is frozen.

---

# 4. Reader definitions

## 4.1 Frozen linear readers

Do not refit or alter the Phase-I linear pipeline. Read the following from the frozen Phase-I artifacts:

- `ridge_raw_tuned_alpha`;
- `ridge_whitened_k508_tuned_alpha`;
- min-norm OLS diagnostics where needed.

Any Phase-III summary that includes linear curves must reference their exact artifact hash.

## 4.2 Historical MLP audit and gate

Before implementing the new reader, audit the repository for the historical MLP that produced the post-P0 scores:

- `jepa_horizon/last`: approximately `0.3191`;
- `supervised/last`: approximately `0.3881`.

The audit must identify:

- architecture;
- input preprocessing;
- target preprocessing;
- optimizer and regularization;
- model-selection rule;
- random seeds;
- split and target inventory;
- whether coordinate-wise standardization, BatchNorm or LayerNorm was used.

If exact historical predictions or checkpoints exist, recompute the scores exactly from them. If only stochastic retraining is possible, reproduce the aggregate within absolute tolerance `0.015` and report seed dispersion.

The historical reader is a gate and reference only. It is not automatically the Phase-III primary reader.

## 4.3 Primary Phase-III MLP

The primary MLP must isolate reader nonlinearity without silently whitening or standardizing coordinates.

### Architecture

For input dimension `d` and block output dimension `T`:

```text
Linear(d, 256, bias=True)
GELU
Dropout(p=0.10)
Linear(256, T, bias=True)
```

No additional hidden layer is permitted in the primary reader.

### Forbidden in the primary reader

- input coordinate-wise standardization;
- BatchNorm;
- LayerNorm;
- whitening except in the explicit whitening arm;
- weight normalization;
- target-dependent feature selection;
- learned PCA or learned pooling.

### Input centering

For native coordinates, subtract the mean fitted once on **all unlabelled train features** of the exact feature set. The mean is fixed across all labelled budgets.

For PCA-band inputs, use the frozen Phase-II train-only PCA basis and center from the same unlabelled train mean.

For the full-whitening arm, use the exact valid-rank Phase-I/II transform fitted on all unlabelled train features. The transformed input dimension is 508.

### Target handling

Train one independent multi-output MLP per target block:

- directional: 12 independent outputs;
- volatility: 2 outputs;
- timing: 1 output.

Within each labelled subset, standardize each target using the labelled-subset mean and standard deviation. Inverse-transform predictions before computing R².

Do not use labels outside the selected labelled subset to standardize targets.

### Loss

Use equal-weight mean squared error over standardized independent targets:

\[
\mathcal L
=
\frac{1}{T}
\sum_{t=1}^{T}
\frac{1}{n}
\sum_{i=1}^{n}
(\widetilde y_{it}-\widehat{\widetilde y}_{it})^2.
\]

No target is weighted by raw variance, historical score or downstream importance.

### Optimizer

- AdamW;
- learning rate: `1e-3`;
- beta parameters: PyTorch defaults;
- epsilon: PyTorch default;
- gradient clipping: global norm `5.0`;
- mixed precision is allowed only if a float32 parity smoke test passes.

### Regularization grid

The only primary hyperparameter grid is AdamW weight decay:

```text
0, 1e-5, 1e-3
```

Dropout, width and learning rate are fixed.

Weight decay is selected separately for each exact:

```text
branch × encoder_seed × target_block × transform × budget × subsample_seed
```

using validation only.

Exact validation ties choose the larger weight decay.

### Step-based training

To avoid confounding small budgets with fewer optimizer updates:

- mini-batch size: `min(4096, n_train)`;
- batches sampled uniformly over rows from the fixed labelled subset;
- maximum optimizer steps: `20,000`;
- minimum steps before stopping: `1,000`;
- full-validation evaluation every `500` steps;
- early-stopping patience: `6` evaluations;
- minimum validation improvement: `1e-5` in canonical mean block R²;
- restore the best validation checkpoint.

The complete fixed validation split is used for checkpoint and weight-decay selection. Test is not read during training or model selection.

## 4.4 Capacity sensitivity

A limited sensitivity is mandatory at:

- the smallest MLP-eligible budget;
- full train;
- directional, native coordinates;
- both primary branches.

Additional widths:

```text
128, 512
```

All other settings remain fixed. This sensitivity does not choose the primary architecture and does not alter the primary outcome.

---

# 5. Hyperparameter selection and test blinding

## 5.1 Selection stage

For each exact cell, use one deterministic selection seed:

```text
selection_seed = 7919
```

Train the three weight-decay candidates and choose the candidate with maximum canonical validation R² averaged over independent targets in the block.

Write a frozen `selection_manifest.json` containing:

- selected weight decay;
- selected checkpoint step;
- validation metrics;
- training seed;
- input transform hash;
- subset hash;
- model-definition hash.

Hash the manifest before any production test evaluation.

## 5.2 Evaluation seeds

After selection, retrain the selected configuration from scratch with independent reader seeds.

Adaptive reader-seed count:

- budgets in the low-budget set: 5 reader seeds;
- higher non-full budgets: 3 reader seeds;
- full train: 3 reader seeds.

Reader seeds are nested and fixed:

```text
0, 1, 2, 3, 4
```

Higher-budget cells use the prefix required by their replication count.

Each evaluation seed uses validation-only early stopping. The test split is evaluated exactly once from its best validation checkpoint.

## 5.3 No retrospective tuning

After the first production test evaluation:

- architecture cannot change;
- budget grid cannot change;
- weight-decay grid cannot change;
- seed counts cannot change;
- target inventory cannot change;
- outcome thresholds cannot change.

Any later architecture is a new experiment and must not be merged into Phase III.

---

# 6. Label budgets and subset identity

## 6.1 Exact subset reuse

Reuse the frozen Phase-I subset manifests and exact row-key hashes. Do not regenerate stock-day permutations or fractional anchors.

For every Phase-III cell, record the Phase-I subset-manifest hash.

## 6.2 MLP eligibility threshold

A budget is primary-eligible only if every required branch/encoder/subsample cell has:

\[
n_{\mathrm{rows}}\ge4096.
\]

The threshold is checked before training and reported.

Let `b_min_mlp` be the smallest existing Phase-I budget satisfying this condition.

No new intermediate budget may be invented.

## 6.3 Primary directional budget grid

Use every existing Phase-I budget from `b_min_mlp` through `full_train`, including `balanced_max` where distinct.

The full budget is deterministic and appears once per encoder seed and reader seed; it has no subsampling-seed replication.

## 6.4 Low-budget set

Define the preregistered low-budget set as:

\[
\mathcal B_{\mathrm{low}}
=
\{b\in\mathcal B_{\mathrm{MLP}}: b\le4\}.
\]

If fewer than two eligible levels exist, use the first three eligible non-full levels and mark the deviation automatically in metadata. Do not choose levels after seeing MLP results.

## 6.5 Specificity-control budget grid

For volatility and timing, use:

1. `b_min_mlp`;
2. the smallest existing budget at least `4 × b_min_mlp`;
3. the smallest existing budget at least `16 × b_min_mlp`;
4. `balanced_max`, if distinct;
5. `full_train`.

Deduplicate coincident levels while preserving order.

## 6.6 Spectral diagnostic budgets

Use:

- `b_min_mlp`;
- the smallest existing budget at least `16 × b_min_mlp`;
- `full_train`.

Spectral band diagnostics are limited to directional and timing.

---

# 7. Metrics

## 7.1 Test R²

Use the exact canonical Phase-I definition per target:

\[
R^2
=
1-
\frac{\sum_i(y_i-\widehat y_i)^2}
{\max\{\sum_i(y_i-\overline y_{\mathrm{eval}})^2,10^{-12}\}}.
\]

The baseline mean is the mean of the evaluation split, as in Phase I.

## 7.2 Target-wise normalized recovery

For every exact reader/branch/transform/target:

\[
A(g)
=
\frac{R^2(g)}{R^2(\mathrm{full})}.
\]

The full-budget denominator must use the same:

- reader family;
- architecture;
- transform;
- target;
- evaluation protocol.

Do not clip negative recoveries.

## 7.3 Ceiling eligibility

A target is eligible for normalized recovery only when:

\[
R^2(\mathrm{full})\ge0.01.
\]

Never divide by a smaller ceiling. Report ineligible targets and reasons.

A directional block is interpretable only with at least two eligible independent targets.

## 7.4 Block aggregation

Within each encoder seed and exact experimental cell:

1. compute target-wise metrics;
2. average only over eligible independent targets;
3. preserve every target row;
4. aggregate encoder/subsample/reader uncertainty hierarchically.

Directional, volatility and timing remain separate.

## 7.5 Nonlinear lift

At each budget and especially full train:

\[
L_r(g)
=
R^2_{\mathrm{MLP},r}(g)
-
R^2_{\mathrm{ridge},r}(g).
\]

Report raw and normalized versions. Do not interpret positive lift as proof of nonlinear Bayes information.

## 7.6 Reader attenuation of the Phase-I gap

On common eligible budgets:

\[
\operatorname{Atten}_{\mathrm{reader}}
=
1-
\frac{\overline G_{\mathrm{MLP,native}}}
{\overline G_{\mathrm{ridge,native}}},
\]

where means are over `B_low` and eligible targets.

Only compute the ratio when the ridge denominator is positive. Retain the absolute difference as the primary robust quantity:

\[
\Delta_{\mathrm{reader}}
=
\overline G_{\mathrm{ridge,native}}
-
\overline G_{\mathrm{MLP,native}}.
\]

## 7.7 Conditioning attenuation within MLP

\[
\operatorname{Atten}_{\mathrm{white|MLP}}
=
1-
\frac{\overline G_{\mathrm{MLP,white}}}
{\overline G_{\mathrm{MLP,native}}}.
\]

Also report the absolute reduction:

\[
\Delta_{\mathrm{white|MLP}}
=
\overline G_{\mathrm{MLP,native}}
-
\overline G_{\mathrm{MLP,white}}.
\]

## 7.8 Reader × conditioning interaction

Define:

\[
I(g)
=
\left[G_{\mathrm{MLP,native}}(g)-G_{\mathrm{MLP,white}}(g)\right]
-
\left[G_{\mathrm{ridge,native}}(g)-G_{\mathrm{ridge,white}}(g)\right].
\]

This is descriptive. It asks whether whitening changes the between-branch gap differently for MLP and ridge.

---

# 8. Uncertainty and paired comparisons

## 8.1 Hierarchical structure

Resample in this order:

1. encoder seeds;
2. subsampling seeds within encoder;
3. reader seeds within encoder × subset.

Use paired resampling across branches and transforms whenever the exact encoder seed, subset seed and reader seed are matched.

## 8.2 Variance decomposition

Report separately:

- `sd_reader_within_subset_encoder`;
- `sd_subsample_within_encoder`;
- `sd_encoder_between_means`.

Do not report only a pooled standard deviation.

## 8.3 Confidence intervals

Use hierarchical bootstrap 95% intervals with a fixed bootstrap seed and at least 10,000 bootstrap draws for final summaries.

Preserve encoder-specific and reader-seed-specific curves.

## 8.4 Robust gap definition

Retain the Phase-I practical threshold:

\[
\delta=0.10.
\]

A normalized gap is robust at a budget when:

1. its hierarchical 95% interval has lower bound above zero;
2. its point estimate is at least `0.10`.

Report sensitivity for `delta ∈ {0.05,0.15}` without changing the primary Phase-III classification.

---

# 9. Preregistered Phase-III outcomes

The primary outcome is evaluated on:

- directional targets;
- `last_concat512`;
- `supervised` versus `jepa_horizon`;
- common MLP-eligible low budgets;
- native and full-whitened full-rank readers.

Phase-I `A1` remains frozen regardless of the Phase-III outcome.

## Outcome R1 — predominantly reader-class-mediated accessibility

Requirements:

1. the frozen native ridge gap is robust on at least two adjacent MLP-eligible low budgets;
2. native MLP reduces the low-budget mean normalized gap by at least 50% relative to native tuned ridge;
3. native MLP has no robust gap on two adjacent low-budget levels;
4. both MLP full-budget ceilings are meaningful;
5. the result is not driven by one encoder or reader seed.

Interpretation:

> Much of the finite-sample penalty is relative to the linear reader class. A nonlinear reader can exploit the frozen JEPA representation more sample-efficiently without changing the encoder.

A remaining full-budget MLP ceiling gap is reported separately and may coexist with R1.

## Outcome R2 — conditioning-mediated accessibility persists for MLP

Requirements:

1. native MLP retains a robust gap on at least two adjacent low-budget levels;
2. full whitening reduces the MLP low-budget mean gap by at least 50%;
3. full-whitened MLP has no robust gap on two adjacent low-budget levels;
4. both MLP ceilings are meaningful and the result is stable across seeds.

Interpretation:

> Reader nonlinearity alone is insufficient. Covariance conditioning remains an important accessibility determinant even for the tested MLP.

## Outcome R3 — persistent difficulty beyond linearity and second-order conditioning

Requirements:

1. native MLP retains a robust low-budget gap;
2. full-whitened MLP also retains a robust gap on at least two adjacent low-budget levels;
3. the full-whitened low-budget mean gap remains at least `0.10`;
4. the pattern is stable across encoder, subset and reader seeds.

Interpretation:

> The tested accessibility penalty is not explained solely by the linear reader prior or covariance conditioning. The downstream map remains more difficult to estimate from `jepa_horizon` under the tested nonlinear reader.

This is not automatically a Bayes-content claim.

## Outcome R4 — mixed or indeterminate reader result

Assign R4 when:

- attenuation is materially below 50% but robustness changes inconsistently;
- signs or conclusions vary strongly across encoder seeds;
- reader-seed variance dominates the between-branch effect;
- ceilings are ineligible;
- optimization failures or unstable validation selection prevent a clean conclusion.

Interpretation:

> Phase III does not support a simple reader-relative or conditioning-relative mechanism under the tested MLP.

Report all quantitative results without forcing R1–R3.

## Secondary ceiling statement

Independently of R1–R4, report:

- full-budget MLP ceiling gap;
- nonlinear lift for each branch;
- MLP-to-supervised performance ratio;
- target-specific ceiling differences.

Do not call these Bayes-content differences.

---

# 10. Spectral reader diagnostics

## 10.1 Purpose

These diagnostics test whether nonlinear recovery of directional signal requires deep spectral bands. They do not redefine predictive mass and do not change R1–R4.

## 10.2 Equal-dimensional bands

Train the same 127-input MLP on each frozen disjoint band:

```text
PC 1:127
PC 128:254
PC 255:381
PC 382:508
```

Because all bands have equal dimension, architecture and parameter count are exactly matched.

Report:

- raw test R²;
- normalized recovery within each band when eligible;
- nonlinear lift over the matched linear band reader from Phase II;
- directional and timing separately;
- per-encoder and hierarchical intervals.

## 10.3 Full-rank and top-128 comparison

Compare:

- full valid rank MLP;
- top-128 MLP;
- best individual 127-dimensional band.

Interpretation rules:

- full rank substantially above top-128 for `jepa_horizon` directional supports tail necessity;
- a deep 127-dimensional band above the head band supports nonlinear usability of deep coordinates;
- timing head band close to full rank supports head sufficiency;
- no band claim may be made from training predictive mass alone.

## 10.4 No “MLP recovers predictive mass” wording

Predictive mass is a linear covariance diagnostic. The report may say that MLP performance is **consistent or inconsistent with the spectral localization measured in Phase II**, but must not say that MLP directly recovers predictive mass.

---

# 11. Cross-fitted spectral control

If Phase-III interpretation quantitatively relates MLP behavior to predictive-mass depth, construct a cross-fitted spectral control.

## 11.1 Procedure

1. keep PCA fixed from all unlabelled train features;
2. partition train stock-days into two deterministic balanced folds within stock;
3. estimate target–PC cross-covariances on fold A;
4. evaluate the resulting linear spectral predictor on fold B;
5. swap A and B;
6. average the two directions;
7. preserve target-wise and encoder-wise results.

## 11.2 Purpose

This control distinguishes train-only population-style mass from out-of-fold spectral recovery. It is secondary and does not alter Phase-II results.

## 11.3 Leakage rule

The test split is never used in the cross-fitting construction. Validation remains reserved for MLP selection.

---

# 12. Free pre-MLP diagnostic: branch-specific whitening effects

Before MLP training, generate a derived table from frozen Phase-I results.

For every:

```text
branch × encoder_seed × target × budget × whitening_depth
```

compute:

\[
D_{r,t}(k,g)
=
A_{r,t}^{(k)}(g)-A_{r,t}^{(0)}(g).
\]

Outputs must show separately whether partial whitening:

- helps `jepa_horizon`;
- harms `jepa_horizon`;
- helps `supervised` more;
- harms `supervised` less;
- changes both in the same direction.

The target-head/tail hypothesis is evaluated descriptively:

> Partial whitening may benefit tail-loaded directional information while temporarily degrading accessibility for head-loaded timing or intermediate volatility.

This hypothesis remains post hoc and cannot alter the MLP grid.

Required output:

```text
phase1_branch_whitening_effects.parquet
phase1_branch_whitening_effects_summary.json
```

---

# 13. Reproduction and acceptance gates

## 13.1 Artifact identity

Before any training, verify hashes for:

- production bundle manifest;
- Phase-I `results.parquet` and technical `summary.json`;
- Phase-II `phase2_results.parquet`, `predictive_mass.parquet` and PCA transforms;
- exact Phase-I subset manifests;
- encoder checkpoints and feature shards.

Any mismatch blocks production execution.

## 13.2 Historical MLP gate

Pass the historical MLP audit/gate described in Section 4.2.

If historical preprocessing included coordinate standardization or hidden normalization, document the difference and continue only after confirming that the new primary MLP intentionally excludes it.

## 13.3 Synthetic nonlinear gate

Create a synthetic representation where:

\[
Y=Z_1^2+\varepsilon
\]

with symmetric `Z_1`.

Required behavior:

- linear reader near zero test R²;
- Phase-III MLP materially positive;
- no test leakage;
- reproducible seed behavior.

This verifies that the MLP pipeline can detect genuine nonlinear accessibility.

## 13.4 Conditioning gate

On synthetic data related by a known invertible anisotropic transform:

- verify equal information and equal oracle function;
- verify native finite-sample sensitivity;
- verify that explicit whitening reduces the induced coordinate penalty;
- verify correct transform fitting on unlabelled train only.

## 13.5 Full-budget linear parity

Read, do not recompute, the frozen Phase-I full-budget linear metrics and verify their hashes and row identities before joining them to MLP results.

## 13.6 PCA-band identity

Verify that:

- the four 127-dimensional bands are disjoint;
- their union is exactly PCs `1:508`;
- each band has dimension 127;
- `top_128` and full-rank projections reproduce Phase-II projections within numerical tolerance.

## 13.7 Train/validation/test isolation

Tests must fail if:

- test metrics are accessed before `selection_manifest.json` is frozen;
- target standardization uses labels outside the labelled subset;
- PCA or whitening uses validation/test features;
- subset row hashes differ from Phase I;
- reader-seed and subsampling-seed identities are conflated.

---

# 14. Compute, checkpointing and restartability

## 14.1 Streaming inputs

Use the existing sharded production bundle. Do not create duplicate 270-GB feature copies.

Cache only:

- exact transformed feature shards when economically justified;
- PCA-band projections;
- whitening transforms;
- selected checkpoints;
- result tables.

## 14.2 Restartability

Every exact cell must have a deterministic job key containing:

```text
branch
encoder_seed
readout
target_block
transform
budget
subsample_seed
reader_seed
weight_decay
```

Completed cells are immutable and skipped on restart only after output-hash verification.

## 14.3 Failure handling

Do not silently retry with changed settings.

Every failure row must record:

- job key;
- exception;
- last completed step;
- validation state;
- GPU/CPU memory status;
- whether the cell is scientifically required.

A primary conclusion cannot be assigned if required cells are systematically missing for one branch.

## 14.4 Runtime report

Report:

- extraction/projection time;
- hyperparameter-selection time;
- evaluation-training time;
- test inference time;
- peak GPU memory;
- peak system RAM;
- output storage;
- number of trained models and failed cells.

---

# 15. Required outputs

## 15.1 Core tables

```text
phase3_results.parquet
phase3_normalized_recovery.parquet
phase3_reader_gap.parquet
phase3_ceiling_and_lift.parquet
phase3_reader_conditioning_interaction.parquet
phase3_variance_components.parquet
phase3_spectral_bands.parquet
phase3_capacity_sensitivity.parquet
phase1_branch_whitening_effects.parquet
crossfitted_spectral_control.parquet        # if invoked by the report
failures.parquet
```

Every raw result row must include:

```text
branch
encoder_seed
readout
target_block
target_name
transform
spectral_arm
budget
n_stock_days
n_rows
subsample_seed
reader_seed
weight_decay
best_step
validation_r2
test_r2
full_budget_ceiling
ceiling_eligible
normalized_recovery
subset_hash
transform_hash
selection_manifest_hash
```

## 15.2 Metadata and manifests

```text
metadata.json
selection_manifest.json
selection_manifest.sha256
phase3_manifest.json
compute_log.json
```

## 15.3 Figures

At minimum:

1. `01_directional_raw_mlp_learning_curves.png`
2. `02_directional_normalized_mlp_recovery.png`
3. `03_linear_vs_mlp_gap.png`
4. `04_native_vs_whitened_mlp_gap.png`
5. `05_reader_conditioning_2x2.png`
6. `06_full_budget_ceiling_and_nonlinear_lift.png`
7. `07_target_specificity_reader_gap.png`
8. `08_reader_seed_variance.png`
9. `09_encoder_specific_mlp_curves.png`
10. `10_equal_dimensional_spectral_bands.png`
11. `11_full_vs_top128_mlp.png`
12. `12_branch_specific_whitening_effects.png`
13. `13_capacity_sensitivity.png`
14. `14_ceiling_eligibility_map.png`

Figures must preserve individual encoder curves or provide companion panels/tables.

## 15.4 Reports

```text
AUDIT_EXPERIMENT_01_PHASE3.md
REPORT_EXPERIMENT_01_PHASE3.md
SUMMARY_NARRATIVE_EXPERIMENT_01_PHASE3.md
CHANGELOG_PHASE3.md
```

The report must clearly separate:

- frozen Phase-I/II facts;
- new Phase-III results;
- preregistered outcome;
- secondary spectral diagnostics;
- post-hoc observations;
- limitations.

---

# 16. Required tests

At minimum, add tests for:

1. exact Phase-I subset reuse;
2. MLP budget eligibility;
3. target standardization using labelled subset only;
4. no coordinate-wise input standardization in native arm;
5. train-only centering/PCA/whitening;
6. valid-rank dimension 508;
7. exact 127-dimensional spectral bands and union;
8. deterministic selection manifest;
9. test access blocked before selection freeze;
10. adaptive reader-seed replication;
11. step-based training and early stopping;
12. weight-decay tie rule;
13. target-wise ceiling normalization;
14. ceiling threshold `0.01`;
15. unclipped negative recovery;
16. hierarchical reader/subsample/encoder variance decomposition;
17. paired branch and transform comparisons;
18. nonlinear synthetic gate;
19. anisotropic-transform conditioning gate;
20. historical MLP gate;
21. join parity with frozen Phase-I results;
22. streaming smoke run across native and whitened MLP;
23. restartability and completed-cell hash verification.

Production execution requires the complete test suite to pass.

---

# 17. Interpretation rules

## 17.1 Allowed conclusions

Depending on outcome, the report may conclude that accessibility is:

- strongly reader-relative;
- conditioning-relative even for a nonlinear reader;
- persistent beyond the tested reader and second-order transform;
- target-specific;
- dependent on deep spectral coordinates.

## 17.2 Forbidden conclusions

Do not state:

- that the MLP estimates Bayes content;
- that equal MLP performance proves equal information;
- that a persistent MLP gap proves information loss;
- that the MLP “recovers predictive mass”;
- that full whitening is a training-time intervention;
- that VICReg or SIGReg must reproduce post-hoc whitening;
- that top-128 failure alone proves tail causality;
- that Phase III changes the Phase-I A1 outcome;
- that any result generalizes beyond the current domain without additional evidence.

## 17.3 Correct final decomposition

The final empirical decomposition should be written as:

1. **full-budget operational ceiling** — how much the tested reader family ultimately recovers;
2. **finite-sample accessibility** — how quickly it recovers its own ceiling;
3. **conditioning dependence** — how recovery changes under an invertible train-only transform;
4. **reader dependence** — how recovery changes when the decoder family is enlarged;
5. **spectral dependence** — which frozen coordinate bands are necessary or sufficient for the tested reader.

---

# 18. Decision after Phase III

After the report is frozen:

- do not immediately add another empirical phase;
- consolidate the post-Phase-III project-state document;
- bring the complete empirical picture to the prospective advisor;
- construct the definitive mathematical framework jointly or under supervision.

The mathematical program should then formalize:

\[
\mathcal R^*(Z;Y),
\quad
\mathcal R_{\mathcal Q}^*(Z;Y),
\quad
\mathcal R_{\mathcal A,n}(Z;Y),
\]

and characterize their invariance or non-invariance under invertible changes of coordinates.

The empirical result to be explained is not merely that JEPA is anisotropic. It is:

> **The same frozen horizon-JEPA representation exposes head-loaded timing, intermediate volatility and deeply buried directional information; these allocations produce different pooling, spectral and finite-sample behavior, and Phase III determines how much of that behavior is relative to the downstream reader class.**

Only after this formalization should the project decide whether to add:

- a controlled solution-selection model;
- a known isotropic regularizer as a causal control;
- a task-aware geometric intervention;
- a second empirical domain.

---

# 19. Implementation sequence for Codex

1. Read:
   - `STATO_TESI_POST_PHASE2_20260731.md`;
   - frozen Phase-I and Phase-II reports/manifests;
   - this specification.
2. Audit the historical MLP and write `AUDIT_EXPERIMENT_01_PHASE3.md` before modifying production code.
3. Verify all artifact hashes and frozen subset identities.
4. Implement and test the branch-specific whitening decomposition using frozen results.
5. Implement the MLP pipeline, selection/test-blinding boundary and synthetic gates.
6. Pass the historical MLP gate.
7. Serialize and inspect all eligible Phase-III job cells before compute.
8. Run hyperparameter selection using validation only.
9. Freeze and hash `selection_manifest.json`.
10. Run independent evaluation seeds and evaluate test once per selected checkpoint.
11. Run the equal-dimensional spectral diagnostics.
12. Produce summaries, variance decomposition, figures and report.
13. Assign exactly one of `R1`, `R2`, `R3`, `R4` for the directional primary outcome.
14. Do not start encoder retraining, VICReg, Phase IV, simulator work or mathematical-proof implementation.

---

# 20. Stop conditions

Stop before production training and report the blocker if:

- historical MLP semantics cannot be reconstructed sufficiently to perform the gate;
- Phase-I subset hashes or Phase-II PCA hashes do not match;
- the native MLP path applies hidden coordinate normalization;
- train/validation/test isolation cannot be guaranteed;
- full whitening cannot reproduce the Phase-I valid-rank transform;
- required spectral bands do not map exactly to the frozen PCA basis;
- the test split is accessed during selection;
- the synthetic nonlinear or conditioning gate fails.

Do not infer missing identities or silently weaken the protocol.

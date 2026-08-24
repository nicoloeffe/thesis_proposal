# Experiment 01 — Finite-sample accessibility of frozen LOB representations

**Version:** 2.0 — preregistered specification  
**Date:** 2026-07-30  
**Status:** definitive implementation specification, superseding the previous draft  
**Primary implementation agent:** Codex, after mandatory repository audit

---

## 0. Purpose and status of the experiment

This is the first decisive experiment after the consolidated representation-geometry analysis.

It does **not** train new encoders and does **not** attempt to prove a mechanistic explanation. It tests whether the already observed organization of predictive signal has a measurable finite-sample cost for downstream learning, and whether any such cost is attributable to:

1. a lower operational ceiling;
2. covariance conditioning and coordinate scale;
3. a genuinely harder downstream function;
4. an information-destroying readout interface.

The experiment is explicitly conditioned on an already established empirical fact:

> **Pooling fragility is already present.** Passing from `last_concat512` to `meanK_concatS` changes directional performance from `0.3191` to `0.0494` for `jepa_horizon`, while supervised changes from `0.3881` to `0.3917`.

Therefore pooling fragility is **not** an outcome competing with the finite-sample hypotheses. It is a known property, denoted below by `C_pooling`, that coexists with whichever finite-sample outcome is observed.

The main unresolved question is:

> Conditional on `C_pooling`, does `jepa_horizon` require more labelled stock-days to recover the same fraction of its own large-sample operational ceiling, and does the gap disappear when covariance conditioning is removed?

---

## 1. Scientific questions

### 1.1 Primary question

For frozen `supervised` and `jepa_horizon` representations evaluated with the full-rank `last_concat512` readout on the canonical directional targets:

> How does downstream performance evolve as the labelled training budget increases by independent stock-day groups?

Both of the following curves must be reported:

\[
R^2_r(g),
\]

and

\[
\widetilde A_r(g)
=
\frac{R^2_r(g)}{R^2_r(\mathrm{full})},
\]

where:

- \(r\) identifies the representation and reader configuration;
- \(g\) is the labelled group budget;
- `full` is the full canonical labelled training split under the same reader protocol.

The raw curve measures absolute downstream utility. The normalized curve measures recovery of the representation's **own operational ceiling**.

### 1.2 Conditioning question

If a normalized sample-efficiency gap exists in the native representation coordinates:

> Does it disappear after progressively whitening the leading covariance directions using only unlabelled training features?

The relevant object is not one learning curve, but the change in the supervised–JEPA gap across reader priors and whitening depth.

### 1.3 Target-specificity question

Is any finite-sample cost:

- specific to directional targets;
- shared with volatility targets;
- shared with the held-out timing target?

Directional targets determine the primary conclusion. Volatility and timing are preregistered specificity controls and must be reported separately, never pooled into one global average.

### 1.4 Readout-interface question

How does finite-sample behavior interact with the already established difference between:

- `last_concat512`;
- `meanK_concatS`?

`last_concat512` is the primary readout for deciding A1/A2/B/D. `meanK_concatS` is a secondary readout used to characterize interaction with the known pooling fragility `C_pooling`.

---

## 2. Preregistered factual starting point

The following are treated as established and are not re-tested as competing outcomes:

1. `jepa_horizon` has substantial directional operational content under `last_concat512`.
2. Supervised has a higher directional large-sample score under the same readout.
3. JEPA directional signal is weakly aligned with the leading variance directions.
4. Supervised directional signal is strongly aligned with leading variance directions.
5. `meanK_concatS` strongly damages `jepa_horizon` directional performance but not supervised directional performance.
6. `jepa_masked/meanK` has a directional score near zero and must not be assigned an interpretable ceiling-normalized curve when the full-budget ceiling is below the declared threshold.

The current experiment adds the **statistical axis**. It does not replace the established geometry and pooling results.

---

## 3. Preregistered outcomes

The outcome taxonomy is evaluated primarily on:

- directional targets;
- `last_concat512`;
- full-rank representations;
- tuned raw ridge versus top-\(k\)-whitened ridge;
- normalized ceiling-recovery curves.

Let

\[
G_r(g)
=
\widetilde A_{\mathrm{sup},r}(g)
-
\widetilde A_{\mathrm{hor},r}(g),
\]

where \(r\) denotes the reader/transform branch.

Define a practically meaningful normalized-recovery gap as:

\[
\delta = 0.10.
\]

A gap is called **robust** at a budget when:

1. the hierarchical 95% confidence interval for \(G_r(g)\) has lower bound above zero; and
2. the point estimate is at least \(\delta\).

Sensitivity results for \(\delta\in\{0.05,0.15\}\) must also be reported, but the primary classification uses \(0.10\).

### Outcome A1 — finite-sample cost mediated by conditioning

Requirements:

1. a robust normalized gap exists in native coordinates for at least two adjacent low-budget levels;
2. progressive top-\(k\) whitening reduces the low-budget gap by at least 50%;
3. for at least one preregistered \(k\), the gap is no longer robust at the same budget levels;
4. the large-sample directional ceiling remains meaningful for both representations.

Interpretation:

> The JEPA signal is statistically more expensive to extract in native coordinates, but much of the cost is explained by the covariance geometry and the reader's coordinate-dependent prior.

This is a basis/interface-conditioning result, not evidence that the abstract downstream function is intrinsically harder.

### Outcome A2 — finite-sample cost persistent after conditioning correction

Requirements:

1. a robust normalized gap exists in native coordinates for at least two adjacent low-budget levels;
2. the gap remains robust after broad top-\(k\) whitening, including the maximum numerically valid whitening depth;
3. tuned ridge and min-norm OLS do not eliminate the pattern;
4. the effect is not driven by a single encoder seed or subsample realization.

Interpretation:

> Covariance scale and linear conditioning are insufficient to explain the cost. The downstream map from the JEPA representation is genuinely more difficult to estimate with finite labels within the tested reader families.

This is the strongest preregistered outcome.

### Outcome B — difference predominantly in operational ceiling

Requirements:

1. raw absolute curves differ materially at large sample size;
2. normalized recovery curves are similar under tuned full-rank readers;
3. no robust \(\delta=0.10\) gap persists across adjacent low-budget levels;
4. the result is stable across encoder and subsampling seeds.

Interpretation:

> `jepa_horizon` preserves less operational directional content, but the content it does preserve is not substantially more label-expensive to learn.

The thesis should emphasize content difference, variance ordering and pooling fragility rather than a general label-efficiency penalty.

### Outcome D — no robust or stable finite-sample conclusion

Evidence may include:

- large sign changes across encoder seeds;
- large within-encoder subsample variability;
- no stable separation between A1/A2/B;
- effects confined to isolated budgets or hyperparameters;
- failure to reproduce canonical full-budget controls.

Interpretation:

> The finite-sample signature is not sufficiently stable to support a mechanism or a thesis claim.

The instability itself must be reported, but no simulator may be designed around an unstable signature.

### Established condition C_pooling

Independently of A1/A2/B/D:

> `meanK_concatS` is already known to be a destructive interface for `jepa_horizon` directional signal and a non-destructive interface for supervised directional signal.

The current experiment asks whether the statistical cost seen under `last_concat512` is amplified, erased or made uninterpretable by this pooling operation.

---

## 4. Scope and staged execution

The experiment is divided into three phases to preserve interpretability and control compute.

### Phase I — primary full-rank experiment

Mandatory and outcome-deciding:

- frozen representations;
- full-rank features;
- `last_concat512` primary;
- `meanK_concatS` secondary;
- directional, volatility and timing blocks;
- min-norm OLS;
- raw ridge with common dimensionless regularization grid;
- raw ridge tuned per branch;
- top-\(k\)-whitened ridge;
- group-structured nested label budgets;
- adaptive subsampling replication.

### Phase II — preregistered geometric extension

Executed after Phase I passes all tests:

- top-PCA subspaces;
- matched random subspaces;
- selected budgets covering low, intermediate and full-data regimes;
- same reader conventions as Phase I where computationally feasible.

Phase II characterizes how truncation interacts with the finite-sample result. It does **not** determine the primary A1/A2/B/D outcome.

### Phase III — secondary nonlinear reader

Optional only after the linear pipeline is validated:

- MLP reader;
- no interpretation below the declared minimum labelled-data threshold;
- no role in primary outcome assignment.

### Explicitly excluded

- training or fine-tuning encoders;
- learned pooling;
- VICReg, SIGReg or other geometry interventions;
- simulator construction;
- causal claims about the encoder objective;
- target redefinition;
- test-set model selection;
- silently replacing unavailable canonical artifacts.

---

## 5. Mandatory repository audit before implementation

Codex must inspect the repository before changing code and produce `AUDIT_EXPERIMENT_01.md` containing:

1. exact paths and hashes of canonical post-P0 feature dumps;
2. branch names and all canonical encoder seeds;
3. exact readout construction for `last_concat512` and `meanK_concatS`;
4. canonical train/validation/test stock-day manifests;
5. stock identifiers and number of training stock-days per stock;
6. endpoint timestamps and row identity keys;
7. target arrays and target-block manifest;
8. identification of algebraically redundant directional targets;
9. exact canonical `R²` implementation and aggregation semantics;
10. exact canonical linear-ladder preprocessing and min-norm solver;
11. any existing ridge, PCA, whitening or random-subspace utilities;
12. dimensionality and dtype of every feature artifact;
13. current handling of intercepts, centering and missing values;
14. environment, dependency versions and expected compute backend.

The implementation must fail loudly if it cannot verify that all inputs belong to the corrected post-P0 pipeline.

No filename, split, target identity, seed or readout definition may be inferred silently.

---

## 6. Experimental factors

### 6.1 Encoder branch

Mandatory:

- `supervised`;
- `jepa_horizon`;
- `jepa_masked`.

Every canonical encoder seed must be included. Expected count is three per branch, but the audit is authoritative.

The primary A1/A2/B/D contrast is:

- `supervised` versus `jepa_horizon`.

`jepa_masked` remains a preregistered comparison but may be excluded from ceiling-normalized summaries when the denominator is below threshold.

### 6.2 Readout

- `last_concat512` — primary;
- `meanK_concatS` — secondary interaction with `C_pooling`.

No readout implementation may be modified.

### 6.3 Target blocks

Use the canonical project target manifest.

#### Primary block: directional

- use the independent directional targets only;
- exclude algebraically redundant copies from aggregate tests;
- preserve per-target scores for all targets, including redundant ones, in the raw output table;
- use the canonical aggregate metric for absolute performance;
- use target-wise normalized recovery for the primary normalized summary.

#### Specificity block: volatility

Use all canonical volatility targets, with the same independence and redundancy rules defined by the target manifest.

#### Specificity block: timing

Use the canonical held-out timing target and preserve its exact capped-target semantics.

Do not average directional, volatility and timing into one score.

### 6.4 Feature view

Phase I:

- `full_rank_raw`;
- `full_rank_whiten_topk`.

Phase II:

- `top_pca_m`;
- `random_subspace_m`.

### 6.5 Reader family

Mandatory Phase I reader branches:

1. `min_norm_ols_raw`;
2. `ridge_raw_common_alpha`;
3. `ridge_raw_tuned_alpha`;
4. `ridge_whiten_topk_common_alpha`;
5. `ridge_whiten_topk_tuned_alpha`.

Optional Phase III:

6. `mlp`.

These are not interchangeable implementation details. They answer different scientific questions.

---

## 7. Group-structured labelled-budget design

### 7.1 Why row-wise sampling is forbidden

LOB windows overlap strongly. Sampling arbitrary rows would make the nominal label count a poor proxy for independent information and would compress the learning curves precisely in the regime of interest.

All primary subsampling must therefore be based on stock-day groups.

### 7.2 Budget unit

Let \(b\) denote labelled **equivalent days per stock**.

The preregistered budget sequence is:

\[
b\in
\left\{
\tfrac18,\tfrac14,\tfrac12,1,2,4,8,16,32,64,\ldots,b_{\max}
\right\},
\]

truncated according to the audited number of available training days per stock.

Add two terminal levels when distinct:

- `balanced_max`: all days up to the minimum stock-specific day count;
- `full_train`: every canonical training stock-day, including any stock-specific excess days.

The plot axis must report both:

- total labelled stock-day equivalents;
- actual number of labelled windows.

Also report:

\[
\frac{n_{\mathrm{rows}}}{D}
\]

for every cell, because the transition near \(n\approx D\) is scientifically central.

### 7.3 Sub-day levels below one full day per stock

For

\[
b\in\left\{\tfrac18,\tfrac14,\tfrac12\right\},
\]

select exactly one training stock-day per stock and retain one contiguous block of canonical endpoints from that day.

The block lengths are:

- \(1/8\) of the valid endpoint sequence;
- \(1/4\) of the valid endpoint sequence;
- \(1/2\) of the valid endpoint sequence.

Requirements:

1. blocks are contiguous in endpoint order;
2. fractional blocks are nested within a subsampling seed;
3. the same stock-days and blocks are used for all branches, encoder seeds, readouts, readers and targets;
4. each block remains inside one stock-day;
5. no context window may cross a day boundary;
6. the block anchor is reproducible from the subsampling seed;
7. the primary anchor is sampled uniformly over feasible positions, then the nested blocks are constructed around the same anchor;
8. a sensitivity table must report opening/middle/closing block position, so a result driven only by time of day is visible.

This design preserves group independence across days while reaching regimes with \(n/D<1\).

### 7.4 Full-day levels

For integer \(b\ge1\):

1. independently permute training stock-days within each stock using the subsampling seed;
2. choose the first \(b\) days per stock;
3. include every valid canonical window from each selected day;
4. make subsets nested as \(b\) increases;
5. use the same subset manifest for every compared representation and reader.

The day selected for the fractional levels must be the first day in the corresponding full-day permutation, ensuring nesting from \(1/8\) through \(b=1\).

### 7.5 Adaptive number of subsampling seeds

Use the following minimum replication schedule:

| Budget | Minimum subsampling seeds |
|---|---:|
| \(b\in\{1/8,1/4,1/2,1,2\}\) | 10 |
| \(b\in\{4,8,16\}\) | 5 |
| \(b\ge32\), excluding exact `full_train` | 3 |
| `full_train` | 1 deterministic realization |

More seeds may be added, but never fewer without a documented compute failure.

The schedule is intentionally concentrated in the low-budget region that decides A versus D.

### 7.6 Uncertainty decomposition

Do not collapse encoder and subsampling variability into one error bar.

For each experimental configuration report:

1. within-encoder mean and variance across subsampling seeds;
2. between-encoder variance of the encoder-specific means;
3. hierarchical total confidence interval;
4. raw points or distributions by encoder seed;
5. number of valid subsampling realizations.

Primary figures should show the hierarchical interval. Companion tables must separately expose:

- `sd_subsample_within_encoder`;
- `sd_encoder_between_means`.

A result driven by one encoder seed cannot be classified as A1 or A2.

---

## 8. Data alignment and leakage rules

### 8.1 Canonical splits

- preserve the existing stock-day grouped train/validation/test split;
- no stock-day may appear in more than one split;
- no endpoint row may appear in more than one split;
- do not reconstruct splits from random row sampling.

### 8.2 Matched subsets

Every subset manifest must be keyed by stable row identity and reused across:

- branches;
- encoder seeds;
- readouts;
- feature transforms;
- target blocks;
- reader families.

The pipeline must prove identity of selected row keys, not merely equality of row counts.

### 8.3 Validation and test

- validation and test sets remain fixed and complete across label budgets;
- all hyperparameter selection uses validation only;
- the test set is evaluated only after the reader configuration is fixed;
- the report must state that the experiment measures training-label efficiency conditional on the fixed canonical validation set.

### 8.4 Unlabelled training features

All train-split representation vectors may be used without target labels to estimate:

- feature means required by a transform;
- covariance matrices;
- PCA bases;
- top-\(k\) whitening transforms;
- trace normalization constants.

Validation and test features must never be used to fit these transformations.

This isolates label efficiency from the cost of estimating representation covariance, which is appropriate for a self-supervised setting.

---

## 9. Raw-coordinate preprocessing

The native-coordinate branches must preserve the representation geometry.

### 9.1 Centering

Use the exact canonical linear-probe convention identified in the audit.

At minimum:

- include an unpenalized intercept or equivalent centering;
- never use validation/test means;
- record whether centering is based on the labelled subset or all unlabelled train features.

The canonical reproduction test is authoritative. If an alternative convention is evaluated, it must be labeled as a sensitivity analysis.

### 9.2 No coordinate-wise standardization in raw branches

Do **not** independently standardize every feature coordinate to unit variance in `raw` branches. Such scaling would already remove part of the variance geometry under investigation.

Allowed raw preprocessing is limited to:

- canonical centering/intercept handling;
- optional global scalar normalization that is explicitly recorded and applied identically within a branch.

---

## 10. Covariance scale and dimensionless ridge regularization

### 10.1 Objective

Ridge is defined as:

\[
\widehat W_{\lambda}
=
\arg\min_W
\left
\{
\frac1n\|Y-XW\|_F^2
+
\lambda\|W\|_F^2
\right\}.
\]

Therefore:

\[
(\widehat\Sigma+\lambda I)\widehat W
=
\frac{X^\top Y}{n},
\qquad
\widehat\Sigma=\frac{X^\top X}{n}.
\]

### 10.2 Trace-normalized regularization

For every design matrix define:

\[
\bar s
=
\frac{\operatorname{tr}(\widehat\Sigma)}{D}.
\]

Use a dimensionless regularization parameter \(\alpha\) and set:

\[
\lambda=\alpha\bar s.
\]

The common-grid comparison is therefore common in \(\alpha\), not necessarily in absolute \(\lambda\).

This prevents global representation scale from being mistaken for conditioning.

### 10.3 Common alpha grid

Use:

```text
alpha = 0 plus 31 log-spaced values from 1e-8 to 1e4
```

The exact values must be serialized in metadata.

For `ridge_raw_common_alpha`, every \(\alpha\) is reported and the same grid is used for every branch.

For `ridge_raw_tuned_alpha`, validation selects \(\alpha\) independently for each:

```text
branch × encoder_seed × readout × target_block × budget × subsample_seed
```

Ties must be resolved deterministically using the largest regularization within one validation standard error if the existing project convention supports this; otherwise select the smallest validation loss and document the tie rule.

### 10.4 Mandatory trace diagnostics

For every branch, encoder seed, readout, transform and budget record:

- `trace_cov`;
- `trace_cov_over_dim`;
- largest eigenvalue;
- smallest numerically valid eigenvalue;
- condition number;
- numerical rank.

At full unlabelled-train scale, report pairwise ratios of `trace_cov_over_dim` across branches.

The report must explicitly verify whether LayerNorm has approximately matched global representation scale. This may not be assumed from average \(|\gamma|\) alone.

---

## 11. Min-norm OLS branch

`min_norm_ols_raw` uses \(\alpha=0\) and the minimum-Euclidean-norm solution.

Requirements:

- use a stable eigendecomposition, SVD or pseudoinverse;
- define numerical rank using a standard machine-precision criterion;
- record rank and tolerance;
- do not add hidden regularization;
- preserve the canonical intercept/centering convention.

This branch measures behavior without explicit ridge shrinkage, though it still reflects the implicit minimum-norm prior when the system is underdetermined.

---

## 12. Top-k whitening as an experimental axis

### 12.1 Estimation data

For each:

```text
branch × encoder_seed × readout
```

estimate mean and covariance from **all unlabelled canonical training features**.

Never refit the whitening transform as the labelled budget changes.

### 12.2 Transform definition

Let the training covariance eigendecomposition be:

\[
\Sigma=U\operatorname{diag}(s_1,\ldots,s_D)U^\top,
\qquad
s_1\ge\cdots\ge s_D\ge0.
\]

For a whitening depth \(k\), define:

\[
T_k
=
U\operatorname{diag}(t_1,\ldots,t_D)U^\top,
\]

where

\[
t_j=
\begin{cases}
1/\sqrt{s_j}, & j\le k,\\
1, & j>k.
\end{cases}
\]

Apply:

\[
X^{(k)}=(X-\mu)T_k.
\]

Thus the first \(k\) covariance directions are whitened and the remaining directions are left unchanged.

### 12.3 k grid

Use:

```text
k ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, D_valid}
```

where:

- `k=0` is the native centered representation;
- `D_valid` is the numerical rank available for stable inversion;
- invalid duplicate levels are removed.

The grid is deliberately dense around the observed participation-ratio scale.

### 12.4 No scientific eigenvalue floor

Do not introduce an arbitrary absolute or relative eigenvalue floor as a model hyperparameter.

Only a standard numerical-rank tolerance derived from machine precision and matrix scale may be used to avoid division by numerical zero. Record:

- requested `k`;
- effective `k`;
- numerical tolerance;
- smallest inverted eigenvalue;
- transform condition number.

If a requested `k` is numerically invalid, mark the cell invalid rather than silently clipping eigenvalues.

### 12.5 Interpretation

The primary whitening result is the curve:

\[
k\mapsto G_k(g),
\]

not a binary raw-versus-whitened comparison.

Report:

- the smallest \(k\) reducing the low-budget gap by at least 50%;
- the smallest \(k\) at which the gap ceases to be robust;
- whether these values are stable across encoder seeds;
- whether full or near-full whitening is required.

Because whitening changes the transformed trace, every whitened ridge cell must recompute \(\bar s=\operatorname{tr}(\widehat\Sigma_k)/D\) before converting \(\alpha\) to \(\lambda\).

---

## 13. Operational ceilings and normalized recovery

### 13.1 Raw performance

Always report canonical absolute test performance:

\[
R^2(g).
\]

No normalized plot may replace the raw curve.

### 13.2 Ceiling definition

For each exact configuration:

```text
branch × encoder_seed × readout × target × reader_family × transform
```

define the operational ceiling as the score at `full_train` under the same fitting and model-selection protocol.

Do not normalize a raw reader curve by a whitened ceiling or a fixed-alpha curve by a tuned-alpha ceiling.

### 13.3 Target-wise normalized recovery

For target \(t\):

\[
\widetilde A_t(g)
=
\frac{R_t^2(g)}{R_t^2(\mathrm{full})}.
\]

Do not clip negative values at low budget.

A target is eligible for normalized analysis only if:

\[
R_t^2(\mathrm{full})\ge0.01.
\]

Eligibility is fixed from the full-budget result before inspecting low-budget behavior.

### 13.4 Block-level normalized summary

The primary block summary is the mean and median of target-wise normalized recoveries over the fixed eligible independent targets.

Also report the ratio of aggregate block scores as a secondary descriptive quantity, but do not use it alone for A1/A2/B/D classification.

If fewer than two directional targets are eligible for one branch-reader configuration, declare the block-level normalized result non-interpretable.

`jepa_masked/meanK` is expected to fail the ceiling threshold and must be treated accordingly rather than assigned an unstable ratio.

---

## 14. Phase II: PCA and matched random subspaces

Phase II begins only after Phase I reproduces the canonical controls.

### 14.1 PCA fitting

Fit PCA separately for each:

```text
branch × encoder_seed × readout
```

using all unlabelled training features only.

Reuse the basis across label budgets.

### 14.2 Dimensions

Minimum preregistered fractions:

```text
m/D ∈ {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}
```

Record exact rounded dimensions.

A first compute-saving pass may use:

```text
m/D ∈ {1/32, 1/8, 1}
```

at selected low, middle and full budgets, then expand after validation.

### 14.3 Random subspaces

- use orthonormal random bases;
- match PCA dimension exactly;
- use deterministic seeds;
- use at least 20 draws per cell, preferably 50 where compute allows;
- never select a draw by test performance.

### 14.4 Purpose

Phase II asks whether the known top-PCA deficit persists and changes with label budget. It must not be used to redefine the primary full-rank finite-sample outcome after seeing the data.

---

## 15. MLP secondary analysis

MLP curves are optional and cannot determine A1/A2/B/D.

A cell is eligible only if both conditions hold:

\[
b\ge8\ \text{full days per stock}
\]

and

\[
n_{\mathrm{rows}}\ge4096.
\]

Requirements:

- frozen encoder;
- fixed preregistered architecture or exact reuse of the canonical MLP probe;
- validation-only early stopping and hyperparameter selection;
- parameter count recorded;
- train/validation gap reported;
- no interpretation of excluded small-budget cells.

The MLP analysis asks whether a nonlinear reader changes the large- and medium-budget comparison. It is not a clean probe of representation sample complexity in the severely overparameterized low-budget regime.

---

## 16. Efficient linear-algebra implementation

### 16.1 Sufficient statistics

For each labelled subset and transformed design compute once:

\[
G=\frac{X^\top X}{n},
\qquad
C=\frac{X^\top Y}{n},
\]

plus all statistics needed for an unpenalized intercept.

All \(\alpha\) values must reuse these matrices.

### 16.2 Eigendecomposition reuse

Compute one eigendecomposition or SVD of \(G\) per:

```text
branch × encoder_seed × readout × transform × target_block
× budget × subsample_seed
```

Then evaluate all ridge values using the shared eigenbasis in \(O(D^2)\) per \(\alpha\), not by refitting from raw rows.

### 16.3 Nested-budget updates

Because budget subsets are nested, the implementation should maintain incremental sufficient statistics:

- row count;
- feature sum;
- target sum;
- \(X^\top X\);
- \(X^\top Y\);
- \(Y^\top Y\).

When moving from one full-day budget to the next, update these statistics by adding the new stock-day blocks rather than rescanning all previous rows.

Correctness is mandatory; incremental updates must be checked against direct recomputation on sampled cells.

### 16.4 Whitening implementation

Do not materialize a separate full transformed dataset for every \(k\) unless memory and runtime are demonstrably acceptable.

Prefer algebraic transformation of sufficient statistics:

\[
G_k=T_k^\top G T_k,
\qquad
C_k=T_k^\top C.
\]

Cache:

- train covariance eigensystem;
- each \(T_k\) or compact equivalent;
- transformed Gram matrices where reused.

### 16.5 Compute log

Record:

- number of raw feature loads;
- number of Gram computations;
- number of eigendecompositions;
- cache hits;
- runtime by stage;
- peak memory where available.

A pipeline that independently refits every \(\alpha\) from raw data is non-compliant.

---

## 17. Mandatory free canonical reproduction test

Before running the learning-curve grid, reproduce the existing canonical full-linear directional ladder result at:

```text
budget       = full_train
readout      = last_concat512
feature_view = full_rank_raw
reader       = min_norm_ols_raw
target_block = directional
```

Expected aggregate test `R²` values under the exact canonical aggregation semantics:

- `jepa_horizon`: **0.2111**;
- `supervised`: **0.3756**.

Tolerance:

```text
absolute difference <= 0.005
```

unless the audit demonstrates that the published values are an aggregation across encoder seeds requiring a different but exactly reproducible comparison. In that case the test must reproduce the same aggregation and document it.

If either value fails tolerance:

1. stop the experiment;
2. write a diagnostic report;
3. inspect row alignment, target block, centering, intercept, aggregation and split identity;
4. do not run the full grid.

This is a hard acceptance gate.

---

## 18. Additional sanity checks and automated tests

At minimum:

1. identical subsampling seed reproduces identical stock-day order, block anchors and row identities;
2. fractional budgets are nested within each seed;
3. integer full-day budgets are nested within each seed;
4. the \(b=1\) day is the same day used for the fractional budgets;
5. no stock-day overlaps across canonical splits;
6. feature and target row keys remain exactly aligned after every transform and subset;
7. all compared branches receive the identical row-key manifest;
8. whitening and PCA use train features only;
9. whitening transforms are fixed across labelled budgets;
10. raw branches do not perform coordinate-wise variance standardization;
11. random-subspace bases are orthonormal to tolerance;
12. every reported \(\alpha\) belongs to the declared grid;
13. every reported \(\lambda\) equals \(\alpha\operatorname{tr}(G)/D\) within tolerance;
14. Gram-based ridge agrees with a direct solver on small sampled cells;
15. incremental sufficient statistics agree with direct recomputation;
16. min-norm OLS uses the declared numerical-rank rule;
17. shuffled-target controls produce near-zero or negative test `R²`;
18. constant-target handling does not produce NaNs silently;
19. result tables contain no duplicate experimental keys;
20. ceiling eligibility is determined only from full-budget scores;
21. `full_train` is evaluated once per deterministic configuration, not redundantly across fake subsample seeds;
22. invalid whitening depths are marked explicitly rather than clipped silently.

The implementation is not accepted until all mandatory tests pass.

---

## 19. Required result schema

Produce one tidy Parquet file, with CSV optional, containing one row per target-level experimental cell.

Minimum columns:

```text
experiment_version
commit_hash
branch
encoder_seed
readout
target_block
target_name
target_independent
budget_kind
budget_days_per_stock
budget_stock_day_equivalents
n_stock_days
n_rows
n_rows_over_dim
subsample_seed
block_anchor_quantile
feature_view
feature_dim
whiten_k_requested
whiten_k_effective
pca_fraction
subspace_seed
reader_family
alpha
lambda_absolute
alpha_selected
trace_cov
trace_cov_over_dim
lambda_max_cov
lambda_min_valid_cov
condition_number
numerical_rank
train_r2
val_r2
test_r2
full_budget_test_r2
normalized_recovery
ceiling_eligible
fit_status
failure_reason
runtime_seconds
```

Also produce:

- `metadata.json`;
- `subset_manifests/` with stable row keys;
- `transforms/` or deterministic regeneration metadata;
- `failures.parquet`;
- `AUDIT_EXPERIMENT_01.md`;
- `REPORT_EXPERIMENT_01.md`.

---

## 20. Required summaries and figures

### 20.1 Primary figures

1. raw directional `R²` versus labelled stock-day equivalents for tuned raw ridge;
2. directional normalized recovery versus labelled stock-day equivalents;
3. supervised–JEPA normalized gap versus budget;
4. gap versus whitening depth \(k\) at each low-budget level;
5. tuned raw ridge versus tuned whitened ridge learning curves;
6. fixed common-\(\alpha\) gap surfaces over `(budget, alpha)`;
7. min-norm OLS learning curves;
8. `n_rows / D` marked on or below the budget axis.

### 20.2 Uncertainty figures

9. within-encoder subsampling distributions at low budgets;
10. encoder-specific mean curves;
11. between-encoder versus within-encoder variance decomposition.

### 20.3 Specificity figures

12. directional, volatility and timing panels using identical axes where possible;
13. `last_concat512` versus `meanK_concatS` interaction panels;
14. ceiling-eligibility map by target and branch.

### 20.4 Phase II figures

15. top-PCA, random-subspace and full-rank curves at selected dimensions;
16. PCA percentile within the random-subspace null across budgets;
17. heatmap over `(budget, m/D)`.

### 20.5 Summary quantities

Report:

- area under the raw learning curve against log labelled-group budget;
- area under the normalized learning curve;
- smallest budget reaching 50%, 80% and 90% of own ceiling;
- right-censor unreached thresholds;
- low-budget mean normalized gap;
- `k_50gap`: smallest whitening depth reducing the gap by at least 50%;
- `k_nonrobust`: smallest whitening depth making the gap non-robust;
- target-block-specific outcome classification.

Do not extrapolate sample requirements beyond observed budgets.

---

## 21. Statistical summaries

### 21.1 Hierarchical uncertainty

Use a hierarchical bootstrap or equivalent two-level resampling procedure:

1. resample encoder seeds;
2. within each encoder seed, resample subsampling seeds.

Report 95% intervals.

Because the number of encoder seeds is small, also show all encoder-specific estimates and do not rely solely on asymptotic standard errors.

### 21.2 Adjacent-budget robustness

A1/A2 require a robust effect at two adjacent low-budget levels. Isolated significant points do not qualify.

The preregistered low-budget set is:

\[
\left\{
\tfrac18,\tfrac14,\tfrac12,1,2,4
\right\}
\]

subject to data availability.

### 21.3 Target aggregation

- preserve per-target outcomes;
- aggregate only independent targets;
- fix target eligibility using full-budget ceilings;
- report mean, median and range across eligible targets;
- do not allow one high-ceiling target to dominate the normalized block result silently.

---

## 22. Interpretation constraints

The report may claim only what the executed branches identify.

### Allowed under A1

- a finite-sample gap exists in native coordinates;
- much of the gap is mediated by covariance conditioning;
- whitening leading directions improves accessibility without adding target labels.

Do not claim that JEPA contains the same Bayes information as supervised.

### Allowed under A2

- a finite-sample gap persists after tested linear conditioning corrections;
- the tested linear reader families require more labelled groups for JEPA;
- covariance scale alone is insufficient.

Do not claim a universal intrinsic difficulty across all possible readers.

### Allowed under B

- ceiling differences dominate the comparison;
- normalized label recovery is similar;
- the geometry remains important for PCA and pooling but not as a robust full-rank label-efficiency penalty.

### Allowed under D

- the finite-sample result is unstable or unresolved;
- variance components and failure modes are characterized.

Do not retrofit a mechanism to noisy results.

### Always allowed from prior evidence

- `C_pooling`: `meanK_concatS` is a fragile interface for `jepa_horizon` directional information relative to `last_concat512`.

---

## 23. Acceptance criteria

Phase I is complete only when:

1. repository audit is written;
2. post-P0 artifact identities are verified;
3. canonical OLS reproduction gate passes;
4. all mandatory branches and canonical encoder seeds are included;
5. group-structured nested budgets are generated and serialized;
6. adaptive seed counts meet the preregistered minima;
7. raw and normalized curves are reported;
8. global covariance traces are verified and reported;
9. common-\(\alpha\), tuned-\(\alpha\), OLS and top-\(k\)-whitening branches are executed;
10. directional, volatility and timing results are separate;
11. uncertainty is decomposed into subsample and encoder components;
12. Gram/eigendecomposition reuse is implemented and verified;
13. one machine-readable table, metadata file and Markdown report are produced;
14. the report assigns directional `last_concat512` to A1, A2, B or D using the preregistered rules;
15. `C_pooling` is reported as a known coexisting condition, not reassigned as an outcome.

Phase II begins only after Phase I acceptance.

---

## 24. Implementation order for Codex

1. Read the master thesis document and this specification completely.
2. Audit repository and canonical artifacts.
3. Implement stable row-key and stock-day manifests.
4. Implement nested group and fractional-day subset generation.
5. Reproduce the canonical full-data OLS scores `0.2111` and `0.3756`.
6. Implement raw sufficient statistics and direct-solver cross-checks.
7. Implement common-\(\alpha\) and tuned-\(\alpha\) ridge using Gram reuse.
8. Implement adaptive subsampling replication and variance decomposition.
9. Implement unlabelled-train covariance audit.
10. Implement top-\(k\) whitening and its numerical diagnostics.
11. Execute directional `last_concat512` Phase I first.
12. Validate results and runtime.
13. Expand to volatility and timing.
14. Expand to `meanK_concatS`.
15. Generate Phase I report and outcome classification.
16. Only after approval, execute PCA/random-subspace Phase II.
17. Only after approval and budget eligibility, execute MLP Phase III.

---

## 25. Required Codex handoff summary

At completion, Codex must provide a concise handoff containing:

- files added or modified;
- exact execution commands;
- canonical artifact paths used;
- tests passed and failed;
- reproduction scores;
- compute and runtime summary;
- missing cells or deviations;
- location of machine-readable results;
- directional outcome A1/A2/B/D;
- separate volatility and timing findings;
- whether the whitening-depth curve supports a conditioning interpretation;
- whether the result is sufficiently stable to motivate a mechanistic simulator.

---

## 26. Decision after Experiment 01

### If A1

Develop the mathematical treatment of reader-relative accessibility and covariance conditioning. Design a controlled model in which the same predictive content is presented under different geometries, then test a target-free conditioning intervention.

### If A2

Develop finite-sample accessibility beyond covariance scale, including function complexity and reader-class dependence. The simulator must reproduce a persistent post-whitening gap.

### If B

Prioritize content loss, predictive sufficiency and pooling-interface analysis. Do not center the thesis on general label inefficiency.

### If D

Increase replication only if the variance decomposition indicates that more seeds can resolve the result. Otherwise retain the established pooling and geometry findings and do not build a mechanism around the learning curves.

In every case, the next mathematical chapter must be written to explain the measured phenomenon rather than to justify a conclusion chosen in advance.

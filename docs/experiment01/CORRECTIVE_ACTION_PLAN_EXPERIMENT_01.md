# Experiment 01 — corrective action plan

**Version:** 1.0 — completed corrective execution record

**Date:** 2026-08-25

**Scope:** Phase I, Phase II, Phase III-R, token-role diagnostic and F16

**Status:** T0–T4 complete; T5 delivered in the supervisor and claim-boundary documents. The frozen specifications, not this retrospective record, authorized each production analysis.

## 1. Objective

The objective is to arrive at the supervisor meeting with:

1. an internally consistent and reproducible account of the frozen Experiment 01;
2. every fixable methodological objection either corrected or tested;
3. every structural limitation explicitly bounded;
4. a preregistered diagnosis of the label-in-pretraining confound;
5. an empirical specification for the mathematical simulator that contains no
   unverified token-role claim.

“Complete” does not mean that every possible extension has been run. It means
that no known issue is hidden, no reported number is disconnected from its
artifact, and every unresolved scientific fork has a named experiment and a
decision rule.

## 2. Frozen baseline and non-negotiable invariants

The existing Phase-I, Phase-II and Phase-III-R numerical artifacts remain
frozen. Corrective work must not silently overwrite or reinterpret them as a
new preregistered experiment.

The following remain unchanged:

- production bundle and row identities;
- encoder checkpoints and their hashes;
- historical train/validation/test assignment;
- Phase-I budgets, subset manifests, seeds, alpha grid and whitening grid;
- target blocks and target-independence declarations;
- Phase-I A1/A2/B/D thresholds and the frozen technical outcome `A1`;
- Phase-III R1/R2/R3/R4 thresholds and the recorded technical outcome;
- Phase-II predictive-mass definition and PCA fitting convention.

Every new result must use a new output namespace and must declare whether it is:

- a reproduction of an existing result;
- a recomputation of an existing statistic;
- a post-hoc diagnostic;
- a new preregistered corrective experiment.

No result from this plan may be merged retroactively into the original outcome
rules.

## 3. Current representation and readout contract

All three encoder arms produce the same shaped contextual grid:

```text
B × K × S × D = B × 20 × 4 × 128.
```

The four role tokens at each timestep are, in fixed order:

```text
bid_top, bid_deep, ask_top, ask_deep.
```

The two canonical 512-dimensional readouts are:

```text
last_concat512 = grid[:, -1, :, :].reshape(B, 512)
meanK_concatS  = grid.mean(axis=1).reshape(B, 512)
```

Therefore:

- `last → meanK` is a temporal operation over the 20 positions;
- Hadamard common/contrast is a role-axis operation over the four contextual
  tokens after selecting a readout;
- PCA is fitted in the resulting 512-dimensional endpoint feature space.

These three operations must remain terminologically separate in code, reports
and the mathematical simulator.

## 4. Workstreams and dependency order

```text
T0  Integrity and protocol repair
 ├── T1  Frozen-result reporting and statistical repair
 ├── T2  Token-role matched-null diagnostic
 └── T3  F16 preregistration
          └── T4  F16 production execution

T1 + T2 + T4
 └── T5  Simulator decision brief and supervisor package
```

`T0` is mandatory before any production execution. `T2` may run on the CPU
while the F16 training jobs run, but its manifest must be frozen first. `T5`
cannot be finalized from intermediate results.

## 5. T0 — integrity and protocol repair

### T0.1 Track the definitive Phase-III specification

Import the exact definitive Phase-III v1 specification into the repository as:

```text
docs/experiment01/SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md
```

The imported bytes must have SHA-256:

```text
78ca15821ac40355c35e5f40ecaf5086f5e6bbb6f339255a85b13fc7d952a151
```

The document, Phase-III amendment, implementation note, audit and report must
state that this later definitive specification governs Phase III. It replaces
the eligibility rule of the earlier optional MLP section for Phase-III jobs.
The apparent `b >= 8` versus `b_min_mlp = b_1_4` conflict is therefore a
versioning issue, not a validity failure.

Acceptance gate:

- tracked specification hash equals the constant in `experiment01/phase3.py`;
- every Phase-III report links to the tracked specification;
- no current document calls the executed Phase-III budgets ineligible.

### T0.2 Make reporting artifact-derived

Remove hard-coded scientific numbers and unconditional conclusions from
`experiment01/reporting.py`. In particular:

- PCA/null values;
- pooling values;
- finite-sample gaps and specificity ratios;
- decisive budget names;
- `k_50gap` and `k_nonrobust`;
- interval-excludes-zero statements;
- execution-status statements about Phase II and Phase III.

Every value in prose must be derived from a validated table or manifest.
Assertions such as “interval excludes zero” must be conditional on the actual
bounds. A missing or inconsistent artifact must fail closed.

Required tests:

1. mutate a synthetic result fixture and verify that every corresponding prose
   value changes;
2. provide an interval crossing zero and verify that the significance sentence
   changes;
3. change decisive budgets and whitening depths in a fixture and verify that no
   stale literal survives;
4. omit a required artifact and verify fail-closed behavior;
5. scan generated reports for superseded execution-status language.

### T0.3 Write the training protocol and hyperparameter audit

Create `docs/experiment01/TRAINING_PROTOCOL.md` covering all nine canonical
checkpoints:

- architecture and tokenization;
- objective and target inventory;
- optimizer, learning rate, weight decay and schedule;
- batch size, epoch count and effective optimizer updates;
- `max_train_samples` and `max_val_samples` actually recorded in checkpoints;
- stopping and checkpoint-selection rule;
- encoder seed, data split seed and initialization policy;
- compute budget and any arm-specific difference;
- checkpoint path, size and SHA-256.

The audit must distinguish matched hyperparameters from objective-required
differences. It must explicitly record that the supervised encoder was trained
on the 20 directional and two volatility targets later probed by Experiment 01,
whereas timing was not directly included but is correlated with training
targets.

The checkpoint metadata currently record `max_train_samples=500000`,
`max_val_samples=50000`, 20 epochs and batch size 256 for all three arms. These
values must be verified for every seed. Do not claim that the JEPA checkpoints
saw approximately 14 times more training rows than supervised unless a
different effective-row audit demonstrates it; the recorded caps do not support
that statement.

### T0.4 Complete the repository audit

Extend `docs/experiment01/AUDIT_EXPERIMENT_01.md` to include all required SPEC
items, including:

- counts of stock-days by stock and split;
- endpoint timestamp, row-key and stock-day identity contract;
- redundant target identities and exclusion rules;
- canonical preprocessing for each reader branch;
- dtype and dimensionality of every distributed artifact;
- environment, package and accelerator versions;
- runtime integrity gates from `pipeline.py`;
- compute-log counters and trace diagnostics;
- production failure-table status.

The audit must say that `pipeline.py` exists and has been inspected. It must not
repeat the scope limitation of the earlier incomplete external-review packet.

### T0.5 State the limits of external validity

Add one shared limitations section, reused by all reports:

- seven stocks from one market/domain;
- group-disjoint but non-chronological historical split;
- possible same-calendar-day presence across stock-specific split sides;
- validation and test derived from a historically explored held-out set;
- test is not a pristine external confirmation set;
- original fractional budgets do not reach `n/D < 1`;
- fractional budgets vary within-day endpoint coverage while retaining seven
  stock-day groups.

These are limitations, not unqualified claims of leakage or invalidity.

## 6. T1 — frozen-result reporting and statistical repair

T1 does not retrain encoders and does not change Phase-I/II/III-R outcomes.

### T1.1 Reproduce every headline from machine artifacts

Build a single claim table containing:

```text
claim_id
phase
metric
value
source_artifact
source_columns
filter
aggregation
artifact_sha256
report_locations
```

Recompute the headline values from the frozen parquets/JSON and compare them
with the existing reports. Any mismatch blocks report regeneration.

### T1.2 Separate technical outcome from scientific interpretation

The report must present:

- `A1` as the frozen preregistered technical classification;
- the absolute linear ceiling gap as a separate operational fact;
- normalized finite-sample recovery as a separate statistic;
- specificity across target blocks as a diagnostic, not a replacement outcome;
- label-in-pretraining as a limit on causal and end-to-end label-efficiency
  interpretation.

Allowed formulation:

> Conditional on the frozen representations and a fresh reader, the supervised
> representation is more accessible at low reader-label budgets.

Disallowed without F16 support:

> The supervised objective is intrinsically more label-efficient end to end.

### T1.3 Report raw and normalized metrics together

For every target block and critical budget, report:

- raw test R2;
- full-budget ceiling;
- normalized recovery mean;
- normalized recovery median;
- eligible-target count and ceiling range;
- minimum and maximum target-level recovery;
- fraction of negative raw test R2 values.

Phase-III normalized gaps above one must never be described as “times worse”.
They must be accompanied by the underlying raw R2 distribution.

### T1.4 Replace seed-only uncertainty language

Retain the existing encoder/subsample-seed intervals, but label them as
computational robustness intervals. Do not call them population-generalization
intervals.

Add test-set grouped uncertainty using saved or recomputed row-level
predictions:

1. resample stocks as the outer level;
2. resample stock-days within sampled stocks;
3. keep every endpoint from a sampled stock-day together;
4. recompute block-level R2 and paired branch differences;
5. report leave-one-stock-out sensitivity separately.

With seven stocks, report the full leave-one-stock-out table and do not rely on
the bootstrap interval alone.

If row-level predictions cannot be recovered without refitting, record that
fact before compute and restrict reruns to the selected frozen models needed to
produce predictions.

This artifact inventory is a Day-1 gate. Check explicitly for:

- serialized fitted coefficients and intercepts;
- row-level test predictions and row keys;
- test sufficient statistics by stock-day;
- train sufficient statistics for every selected budget/transform;
- enough cached information to reconstruct the validation-selected model.

The existing aggregate transform and Phase-II cache files do not by themselves
provide grouped test resampling. If coefficients or per-stock-day statistics
are absent, estimate the exact streaming/refit cost before scheduling Days 5–6.

Extend grouped uncertainty to Phase-II headline claims. Predictive mass is a
train-estimated statistic, so its grouped uncertainty must resample train
stock-days and recompute the covariance eigensystem and cross-covariance, or use
a clearly labelled grouped jackknife when the full bootstrap is too expensive.
Report stock-level leave-one-out sensitivity for top-8/top-16 mass. Test-side
PCA-ladder R2 uncertainty remains a separate grouped evaluation problem and
must not be conflated with train-side predictive-mass uncertainty.

### T1.5 Test specificity rather than quoting a ratio

The ratios `directional/volatility` and `directional/timing` may remain as
descriptive point summaries, but the primary specificity analysis must include:

- raw-gap contrasts across target blocks;
- normalized-gap contrasts across target blocks;
- grouped-bootstrap intervals for the differences;
- target-level sensitivity;
- denominator/ceiling sensitivity;
- interaction interpretation only when the corresponding interval excludes
  zero.

If robustness is present only on the normalized scale, describe it as
scale-dependent.

### T1.6 Whitening-depth and budget-selection stability

Using existing Phase-I results, report:

- `k_50gap` separately for each encoder seed;
- `k_nonrobust` separately for each encoder seed;
- the gap at every registered whitening depth;
- results for every adjacent low-budget pair, not only the first robust pair;
- the originally selected decisive pair clearly marked as data-selected;
- sensitivity of the bridge to a fixed `b_1_8/b_1_4` pair and to the complete
  low-budget set.

No threshold or Phase-I outcome changes.

### T1.7 Repair Phase-II band comparisons

The direct comparison of `17:32` with `33:64` is not dimension matched. It must
be removed as evidence or replaced by:

- equal-width sub-bands;
- per-direction descriptive mass, clearly labelled descriptive;
- the existing matched-dimension random-subspace control;
- raw R2, leave-band-out R2 and predictive mass reported separately.

The original negative conclusion about the proposed non-monotonic mechanism
remains frozen; this repair cannot turn it into a positive preregistered result.

### T1.8 Surface existing controls

Report production evidence for controls already implemented:

- shuffled-target near-zero/negative behavior;
- incremental versus direct sufficient statistics;
- Gram solver versus direct solver;
- rank-threshold parity;
- feature-load, eigendecomposition and cache counters;
- trace covariance ratios for every budget, not only full train;
- time-of-day sensitivity across opening, middle and closing anchors;
- figure-06 common-alpha selection verified from data, not asserted by a
  literal boolean.

### T1.9 Correct the `n/D` statement

The original experiment did not reach `n/D < 1`. The frozen report must say so.
Do not relabel the existing fractional grid.

A new endpoint-budget `n/D` diagnostic is optional and must be a separately
preregistered experiment with its own sampling unit and autocorrelation caveat.
It is not a gate for the supervisor package unless the mathematical simulator
explicitly requires empirical calibration across the interpolation threshold.

### T1.10 Reframe Phase III-R

Phase III-R remains protocol-valid under the definitive Phase-III
specification. Its reporting must nevertheless:

- keep the technical R outcome unchanged;
- lead with raw R2 and ceiling eligibility;
- treat large normalized gaps with negative low-budget R2 as unstable ratios;
- separate low-budget reader behavior from full-budget head/deep diagnostics;
- preserve the matched 127-dimensional head/deep result as an exploratory
  within-JEPA observation;
- avoid claiming a general nonlinear accessibility mechanism from the reduced
  grid.

## 7. T2 — token-role matched-null diagnostic

### T2.1 Scientific question

The existing results already establish an operational cross-arm fact: the same
128-dimensional all-ones role projection retains almost all supervised linear
performance and only a small fraction of JEPA performance. T2 does not need to
rediscover that contrast.

T2 asks the narrower mechanistic question: for the four contextual role tokens
at the selected readout, are the all-ones common direction and its zero-sum
complement special relative to dimension- and structure-matched random role
subspaces? It also tests whether the common role direction is a high-variance
direction and whether its predictive performance is unusual conditional on the
variance it captures.

This is a post-hoc diagnostic. It does not alter A1/A2/B/D or R1/R2/R3/R4.

### T2.2 Reproduction gate and established baseline

Before any null calculation, reproduce from the exact historical readout dumps:

| arm | full | common 128D | contrasts 384D |
|---|---:|---:|---:|
| `jepa_horizon` | 0.211129 | 0.041423 | 0.204832 |
| `supervised` | 0.375636 | 0.372956 | 0.333051 |
| `jepa_masked` | 0.100645 | 0.015770 | 0.091788 |

The gate must match per encoder seed and target, not only aggregate means.

Once reproduced, the matched cross-arm baseline may be reported directly:

| arm | common/full | contrasts/full |
|---|---:|---:|
| `jepa_horizon` | 19.6% | 97.0% |
| `supervised` | 99.3% | 88.7% |
| `jepa_masked` | 15.7% | 91.2% |

This establishes selective loss under the fixed common-role projection. It does
not, by itself, show that the Hadamard axis is exceptional among role axes, nor
does it identify whether the cross-arm difference is caused by objective family
or target-aligned supervision.

The historical analysis also already contains the Hadamard decomposition after
temporal averaging. Reproduce it rather than scheduling it as a new analysis:

| arm | `meanK` full | `meanK` common | `meanK` contrasts |
|---|---:|---:|---:|
| `jepa_horizon` | 0.063059 | 0.008961 | 0.057360 |
| `supervised` | 0.386520 | 0.389158 | 0.307328 |
| `jepa_masked` | 0.004130 | 0.000798 | 0.004419 |

Values slightly above full are possible out of sample because the projected
models are fitted independently; they must not be interpreted as additive
content creation.

### T2.3 Primary structured Haar null

For each draw:

1. sample a deterministic Haar matrix `Q` in role space `R^(4x4)`;
2. take one column `q` as a random common-like role direction;
3. take the remaining three columns `Q_perp` as its complement;
4. lift them to feature space using:

```text
B_common   = q      tensor I_128   -> 512 x 128
B_contrast = Q_perp tensor I_128   -> 512 x 384
```

5. fit both projected designs using the same train-centered min-norm OLS as the
   historical Hadamard analysis;
6. evaluate on the identical fixed evaluation side recorded by the source; the
   historical reproduction source uses its validation side, not the new
   production test.

Use 100 deterministic draws per arm and encoder seed. Store every draw. The
minimum attainable empirical p is therefore `1/101` using the plus-one
convention.

Run the structured null for both `last_concat512` and `meanK_concatS`. For each
observed and random subspace, also compute:

- fraction of covariance trace captured by the 128D direction and 384D
  complement;
- raw and trace-normalized predictive performance;
- principal-angle overlap with leading covariance eigenspaces;
- residual predictive performance relative to the empirical R2-versus-variance
  trend across draws.

This tests, rather than assumes, that the all-ones role direction is a dominant
variance direction and that JEPA is unusually anti-aligned with it.

### T2.4 Secondary generic feature-space null

As a secondary diagnostic only, compare with generic 128D and 384D Haar
subspaces in `R^512`. This answers whether the role-structured null differs
from arbitrary channel-and-role mixing. It must not replace the structured
null in the primary conclusion.

### T2.5 Required outputs

For all three arms, three encoder seeds and every independent directional
target, report:

- common and contrast raw R2;
- retention relative to full R2;
- variance fraction captured by each subspace;
- R2-versus-variance scatter and residual;
- overlap with leading covariance eigenspaces;
- percentile within the matched null;
- empirical p;
- `full - common` and `full - contrast`;
- shared/commonality term;
- two-block Shapley attribution, labelled descriptive;
- role-signal-span energy as a separate geometric metric;
- block aggregate only after seed/target tables;
- shuffled-target control;
- numerical failures and rank diagnostics.

Suggested artifacts:

```text
token_role_observed.parquet
token_role_structured_null.parquet
token_role_generic_null.parquet
token_role_commonality.parquet
token_role_summary.json
token_role_failures.parquet
token_role_manifest.json
REPORT_EXPERIMENT_01_TOKEN_ROLE.md
```

### T2.6 Interpretation rules

The report must distinguish the following outcomes:

1. **JEPA common unusually weak conditional on variance, complement unusually
   strong; supervised not:** role-axis-specific representation geometry.
   Causal attribution between objective family and label exposure still
   requires F16.
2. **Same unusual pattern in all arms:** source/architecture-associated role
   structure, not JEPA-specific organization.
3. **Observed subspaces typical under the structured null:** no evidence that
   the Hadamard role axis is exceptional among role axes; retain the already
   established operational common-pooling loss across encoders.
4. **Seed- or target-dependent result:** mixed post-hoc diagnostic; do not add a
   token-role constraint to the simulator.

No claim may use the raw `0.041 versus 0.205` ratio as a dimension-corrected
effect size.

### T2.7 Simulator gate from T2

Before T2, the simulator may encode:

- anti-alignment of predictive signal with high-variance directions;
- selective destruction under temporal averaging.
- selective destruction of JEPA, but not supervised, under the fixed all-ones
  role projection.

The last item is an operational constraint, not yet a general mechanism. The
simulator may identify it with a special high-variance role-common axis only if
the structured null and variance diagnostics support that identification.

Treat the proposition that PCA anti-alignment, role-common loss and temporal
mean loss are one mechanism as a new unification hypothesis, not an established
fact. PCA directions can mix roles and latent channels; the role-common
direction need not be a leading covariance direction; temporally contextualized
positions need not form a Fourier-like stationary axis. Temporal averaging and
role averaging remain separate operators unless the diagnostics establish the
claimed link.

A temporal-frequency decomposition may be proposed after verifying whether the
full `K x S x D` grids or adequate sufficient statistics are available. It is
not assumed to be compute-free and is not a T2 completion gate.

## 8. T3 — F16 preregistration

### T3.1 Estimand

F16 tests how target-aligned supervised pretraining changes representation
geometry and fresh-reader accessibility as the number of labelled training
examples increases.

It does not reclassify the frozen Phase-I outcome. It distinguishes:

- representation-conditional reader accessibility;
- direct target co-adaptation;
- dependence of geometry on supervised label volume.

The primary arm comparison is budgeted `supervised` versus frozen
`jepa_horizon`. Frozen `jepa_masked` is a secondary internal control. The
existing full-budget supervised checkpoint is a descriptive upper anchor and
is not silently treated as if it followed the new stopping rule.

### T3.2 Label budgets

Use the exact nested Phase-I manifests:

| label | rows at seed 0 | stock-days represented | stock-day equivalents |
|---|---:|---:|---:|
| `b_1_4` | 7,116 | 7 partial days | 1.75 |
| `b_1` | 28,446 | 7 full days | 7 |
| `b_4` | 122,099 | 28 full days | 28 |
| `b_16` | 490,937 | 112 full days | 112 |

Use encoder seeds `0,1,2`. Do not use `b_64`: its 1,930,201 rows exceed the
canonical supervised `max_train_samples=500000`, while `b_16` already matches
that effective upper scale.

`b_1_4` is a declared floor, not a presumed failure. Degeneracy, instability or
failure to learn must be determined by preregistered gates.

Do not call `b_1_4` “approximately random initialization” on the basis of the
parameter-to-row ratio. If an untrained-geometry reference is scientifically
needed, add an explicit zero-update checkpoint as a separately declared
control.

`b_16` is a near-match to the canonical 500,000-row cap and should be compared
with the canonical supervised checkpoint. That comparison does not isolate the
stopping rule alone: the exact stock-day support, row selection and validation
selection also differ and must be reported.

### T3.3 Label accounting

For each budget `L`:

- the budgeted supervised encoder trains on the exact labelled rows in `L`;
- the fresh reader for that encoder trains on the same `L` manifest;
- frozen JEPA readers train on the same `L` manifest;
- validation labels are a fixed overhead common to all budgets and must be
  counted separately;
- report `|L|`, `|V|` and `|L union V|` in rows and stock-days.

Also record, for every arm and training stage, whether `V` is used for encoder
checkpoint selection, reader hyperparameter selection or both. Equal access to
unique labels and equal adaptive use of those labels are different properties
and must not be conflated.

The primary estimand is therefore incremental training-label efficiency
conditional on a fixed validation set, not an unconditional count of every
label consumed by model development.

The budgeted supervised encoder also receives gradient-bearing feature
exposure only on `L`, whereas the frozen JEPA encoder was pretrained on the
larger unlabeled corpus. Therefore:

- persistence of supervised geometry at low `L` is strong evidence despite
  this disadvantage;
- collapse at low `L` remains ambiguous between label scarcity, feature
  exposure and optimization/capacity limits;
- resolving that ambiguity would require a separately designed semi-supervised
  or target-ablation control, not an improvised change to F16.

### T3.4 Encoder stopping rule

The preregistration must freeze, before production:

- a common maximum optimizer-update budget;
- validation cadence expressed in updates;
- patience expressed in validation checks;
- checkpoint tie-break rule;
- gradient and numerical-failure rules;
- the exact validation set;
- a test-access barrier.

Primary checkpoint selection is validation based. Record steps-to-best and the
complete train/validation trajectory. Preserve the epoch-20 checkpoint from
each run as a sensitivity analysis because the canonical encoders use epoch 20.

The maximum-update value must be derived from a validation-only benchmark and
the canonical training protocol, then frozen in the F16 specification. It must
not be changed after any F16 test result is seen.

### T3.5 Cohort and storage gate

Do not re-extract every endpoint for every new checkpoint.

Construct a fixed union of:

1. every labelled training row required through `b_16`;
2. a target-blind unlabeled-train covariance cohort spanning all train
   stock-days;
3. the fixed validation cohort;
4. a fixed test cohort spanning all test stock-days.

Choose the within-stock-day endpoint cap by a convergence benchmark on existing
checkpoints. Candidate caps must be fixed before inspection, for example:

```text
128, 256, 512, 1024 endpoints per stock-day.
```

Select the smallest cap reproducing predefined reference metrics within frozen
tolerances. At minimum benchmark:

- full-rank directional R2;
- top-8 and top-16 predictive mass;
- common/full and contrast/full retention;
- `last → meanK` gap;
- covariance trace and leading eigenvalues.

Sampling inside the covariance/evaluation cohort must never modify the frozen
Phase-I labelled training manifests. All row keys and hashes must be stored.

### T3.6 F16 primary and secondary measurements

F16 must keep two evaluation axes separate.

**Axis A — label-matched end-to-end point.** For an encoder trained with `L`,
fit its fresh reader on the same `L`. Compare with frozen JEPA readers trained
on `L`. This measures the complete tested pipeline at equal incremental unique
training labels.

**Axis B — fixed-reader representation diagnostic.** For every budgeted
encoder, fit the reader using the same fixed `b_16` labelled reference. Use
this axis for ceiling, role decomposition and cross-encoder geometry so that a
smaller reader budget is not mistaken for a worse representation. Axis B is a
diagnostic and does not claim total label matching.

For every budget and encoder seed:

Primary:

- raw directional R2 under `last_concat512`;
- Axis-A label-matched R2;
- Axis-B fixed-`b_16` R2 and ceiling;
- common/full and contrast/full role retention;
- top-8, top-16 and cumulative predictive mass;
- `last → meanK` loss.

Secondary:

- volatility and timing controls;
- covariance spectrum and trace;
- progressive-whitening bridge at selected frozen depths;
- target-level results;
- grouped stock/stock-day uncertainty;
- leave-one-stock-out sensitivity;
- epoch-20 stopping sensitivity.

The most informative F16 curve is the geometry dose-response, not a binary
“supervised wins” comparison.

### T3.7 F16 hypotheses and interpretations

1. **Supervised-like geometry already at low `L`:** geometry is strongly
   associated with the supervised objective and does not require full label
   volume under the tested optimization protocol.
2. **Smooth transition from JEPA-like to supervised-like geometry:** geometry
   depends on the volume of target-aligned supervision.
3. **Accessibility changes without spectral/pooling geometry changing:** the
   original finite-sample gap is not explained by the measured second-order or
   role geometry alone.
4. **No stable learning at low `L`:** low-budget encoder capacity/optimization
   floor; no causal conclusion from the collapsed cells.
5. **Strong target-block heterogeneity:** direct co-adaptation is supported when
   directional geometry changes with `L` more strongly than timing, while
   accounting for target correlations and ceiling scale.

No F16 outcome changes the historical A1 label. F16 changes the permissible
causal narrative and simulator-selection mechanism.

### T3.8 F16 required artifacts

```text
SPEC_EXPERIMENT_01_F16_LABEL_MATCHED.md
f16_manifest.json
f16_job_inventory.parquet
f16_training_curves.parquet
f16_checkpoint_manifest.json
f16_cohort_manifest.json
f16_results.parquet
f16_geometry.parquet
f16_grouped_uncertainty.parquet
f16_failures.parquet
f16_summary.json
REPORT_EXPERIMENT_01_F16.md
```

Every checkpoint and result artifact must be hash-pinned. Jobs must be
resumable, idempotent and fail closed on manifest drift.

## 9. T4 — F16 production execution order

1. Record Git commit and clean/dirty status.
2. Freeze the F16 specification and its SHA-256.
3. Freeze label, covariance, validation and test cohort manifests.
4. Run the convergence benchmark using only existing checkpoints.
5. Run one-seed smoke tests without test access.
6. Verify shapes, targets, losses, checkpoint reload and deterministic row
   identity.
7. Serialize the complete production job inventory.
8. Train all budgeted supervised encoders sequentially and resumably.
9. Freeze the validation-selected checkpoint manifest.
10. Extract only the fixed union cohort.
11. Run reader and geometry analyses on train/validation.
12. Freeze all hyperparameter and checkpoint selections.
13. Unlock test evaluation exactly once.
14. Generate grouped uncertainty and leave-one-stock-out analyses.
15. Run `summarize`, `report` and final integrity audit.
16. Record runtime, peak RAM/VRAM, storage, failures and artifact hashes.

Intermediate test results must not be interpreted or used to change the job
inventory.

## 10. T5 — simulator decision brief

**T2 execution record (2026-08-26).** The historical gate reproduced 1,188
cells with maximum absolute error `1.79e-12`; the 100-draw structured role-Haar
grid completed with zero failures. No arm/readout satisfied both the
all-three-seed `unusually_weak` common rule and the all-three-seed
`unusually_strong` complement rule. The T2 decision-matrix branch is therefore
`not exceptional`: retain the fixed all-ones projection only as an operational
constraint and omit a privileged Hadamard mechanism from the simulator.

The supervisor brief must separate empirical constraints from causal
interpretations.

### 10.1 Constraints already safe

The minimal simulator may reproduce:

1. predictive content that is not concentrated in the leading covariance
   directions;
2. representations with similar content but different finite-sample linear
   accessibility;
3. selective information loss under a temporal averaging operator;
4. conditioning improvement under train-fitted whitening.
5. selective JEPA loss under the fixed all-ones role projection, contrasted
   with near-complete supervised retention.

### 10.2 Conditional role-axis mechanism

The observed all-ones projection may be represented as an empirical operator
from the start. Treat it as a privileged high-variance role axis, or unify it
with the PCA mechanism, only if T2 rejects the structured role-null explanation
and verifies the variance relationship consistently across seeds and targets.

### 10.3 Conditional solution-selection mechanism

Use F16 to decide whether the mathematical model should parameterize:

- objective-associated geometry present at low supervised label volume;
- a continuous geometry transition with target-aligned supervision;
- only representation-conditional accessibility, without a causal training
  claim.

### 10.4 Decision matrix

| T2 role null | F16 dose-response | Simulator implication |
|---|---|---|
| JEPA-specific | supervised-like already at low `L` | model objective-associated role geometry |
| JEPA-specific | smooth transition with `L` | model supervision-dependent solution selection |
| not exceptional | any | retain the observed all-ones projection as an operational constraint, but omit a privileged Hadamard mechanism |
| mixed | mixed or failed | keep only spectral and temporal constraints; role mechanism remains open |

The mathematical work may begin immediately on the population linear-Gaussian
family and invertible-reparameterization analysis, but it must leave the
conditional mechanisms above as explicit switches until T2 and F16 finish.

## 11. Supervisor-ready package

The final package must contain:

1. scientific README and research note;
2. master Experiment-01 SPEC;
3. definitive Phase-III SPEC and amendment;
4. F16 preregistration and manifest;
5. corrected Phase-I, Phase-II and Phase-III-R reports;
6. token-role diagnostic report;
7. F16 report, if production completes;
8. training protocol and checkpoint manifest;
9. completed integrity audit and limitations section;
10. compact machine-readable result tables and hashes;
11. simulator decision brief;
12. exact commands for reproduction.

The package must distinguish portable public artifacts from local large
artifacts. Absolute local paths must not appear in public manifests.

## 12. Suggested seven-day execution schedule

### Day 1 — freeze and repair

- T0.1 Phase-III specification import and links;
- F16 training-protocol audit;
- T2 and F16 draft specifications;
- cohort convergence benchmark launch;
- reporting hard-code removal begins.
- inventory fitted coefficients, row-level predictions and per-stock-day
  sufficient statistics; revise the grouped-bootstrap compute estimate before
  scheduling production reruns.

### Day 2 — gates

- reporting tests and claim-table reproduction;
- T2 historical reproduction gate;
- F16 one-seed train/validation smoke test;
- freeze job and cohort manifests.

### Days 2–5 — production compute

- sequential/resumable F16 encoder training;
- T2 structured role-null on CPU;
- T1 seed/budget/whitening and band reanalyses;
- no interpretation of partial F16 test results.

### Days 5–6 — fixed evaluation

- validation selection freeze;
- one-time F16 test evaluation;
- stock/stock-day bootstrap and leave-one-stock-out;
- raw/normalized/median tables;
- specificity interaction and target sensitivity.

### Day 7 — synthesis

- regenerate all corrected reports;
- complete audits and manifests;
- write simulator decision brief;
- run the full test suite;
- record Git status, commit and final artifact hashes;
- assemble the supervisor package.

If F16 training exceeds the window, do not weaken the protocol. Deliver the
completed T0–T2 package, frozen F16 specification, completed cells and explicit
remaining inventory.

## 13. Definition of done

The corrective programme is complete when:

- all distributed specifications are tracked and hash-consistent;
- no report contains a scientific number disconnected from an artifact;
- the frozen technical outcomes are unchanged and clearly labelled;
- raw and normalized metrics, means and medians are co-reported;
- uncertainty scope is accurately named and grouped test uncertainty is
  available for headline claims;
- unmatched spectral and role-subspace comparisons are repaired or removed;
- every headline claim that does not survive grouped uncertainty is explicitly
  downgraded in the narrative, not merely left with a wider interval in a
  table;
- Phase III is governed by the correct specification and cautiously framed;
- the supervised label-in-pretraining confound is explicitly stated;
- T2 has a complete matched-null result for all three arms;
- F16 has either completed under a frozen protocol or is presented as a frozen,
  fully inventoried pending experiment;
- the simulator brief includes only empirically supported constraints;
- all tests pass and all output manifests verify.

## 14. Explicitly out of scope

This corrective programme does not include:

- Phase II/III expansion for its own sake;
- broad MLP architecture or capacity sweeps;
- VICReg, SIGReg or new encoder objectives;
- P-to-M target battery analysis;
- topology experiments;
- a full simulator implementation before the supervisor decision;
- claims of pristine temporal or external-market validation;
- silent replacement of the historical split or production bundle.

Those are separate experiments or later thesis phases.

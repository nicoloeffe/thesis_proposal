# Experiment 01 — T2 token-role matched-null diagnostic

**Version:** 1.0
**Date:** 2026-08-26
**Status:** preregistered corrective diagnostic; frozen before production
**Scientific role:** post-hoc mechanism test; no Phase-I or Phase-III reclassification

## 1. Question

The existing post-P0 analysis applies a fixed Hadamard transform to the four
contextual token roles at a selected 512-dimensional endpoint readout. It
reports a 128-dimensional all-ones role block (“common”) and its
384-dimensional zero-sum complement (“contrasts”).

The established operational observation is that the common block retains much
less directional linear performance for the two JEPA representations than for
the supervised representation. The unresolved question is narrower:

> Are the all-ones role direction and its zero-sum complement exceptional
> relative to random role directions and their matched complements, after
> matching dimension and recording captured covariance scale?

This diagnostic does not assume that role averaging, temporal averaging and
PCA anti-alignment are the same mechanism.

## 2. Frozen inputs

The diagnostic uses only the historical post-P0 artifacts in
`validation/readouts_v2_20260728`:

- extraction manifest SHA-256
  `96eaf6b2e6829779697c224d6919364ce159e016c0caf1975fc0ee752ccb2e91`;
- trained-target artifact SHA-256
  `f2ab87577875e8c535d9e7ebdd4b60df991f20c2564cb9c8d57aeaa9ac9e9ac9`;
- split artifact SHA-256
  `0c5149c1260c153c8bdbe3ac8a453750816b4ef62eaa6b54ac03ffb396245cc3`;
- historical per-target ladder SHA-256
  `62dfeb18c0d9a8c792e7c788b803e3666f8ecf423956119f15076573e1d05785`;
- historical aggregate ladder SHA-256
  `fc32ec77807a6cff4a04ea1358a2ea8a6a977a332a711f7960ef724cd51086db`;
- nine epoch-20 readout dumps, whose hashes are taken from the frozen
  extraction manifest and reverified before use.

The historical arrays contain 100,000 train endpoints and 50,000 validation
endpoints. Validation is the evaluation side because this diagnostic
reproduces the historical analysis. It does not use the later production test
split and is not presented as external confirmation.

## 3. Arms, seeds, readouts and targets

Required encoder arms:

- `jepa_horizon`;
- `jepa_masked`;
- `supervised`.

Required encoder seeds: `0, 1, 2` at epoch 20.

Required readouts:

- `last_concat512`;
- `meanK_concatS`, sourced from historical `tmean_concat512` arrays.

Each readout is ordered as four contextual roles with 128 channels per role:

```text
bid_top, bid_deep, ask_top, ask_deep.
```

The primary target family contains the 12 independent trained directional
targets retained by the historical aggregate. The eight exact best-bid and
best-ask copies of spread are reproduced by the gate but excluded from the
primary block aggregate. Volatility targets may be serialized as controls but
cannot alter the token-role conclusion.

## 4. Reader and evaluation convention

All projected readers use the exact historical convention:

- train-fitted feature and target means;
- min-norm OLS with `numpy.linalg.lstsq(..., rcond=None)`;
- train-mean intercept restored at evaluation;
- per-target validation R² with the validation target mean as denominator;
- no ridge selection, target standardization, clipping or test-set tuning.

Full, common and complement models are fitted independently. Their
out-of-sample R² values are not assumed additive.

## 5. Historical reproduction gate

Before any null result is generated, the implementation must reconstruct from
the raw readout matrices every historical full-rank per-target cell for:

- full 512D readout;
- Hadamard common 128D block;
- Hadamard contrast 384D block;
- both required readouts;
- all three arms and all three encoder seeds.

Every recomputed per-target R² must match `ladder_long.csv` within absolute
tolerance `5e-10`. The aggregate means must reproduce:

| readout | arm | full | common 128D | complement 384D |
|---|---|---:|---:|---:|
| last | horizon-JEPA | 0.211129 | 0.041423 | 0.204832 |
| last | masked-JEPA | 0.100645 | 0.015770 | 0.091788 |
| last | supervised | 0.375636 | 0.372956 | 0.333051 |
| meanK | horizon-JEPA | 0.063059 | 0.008961 | 0.057360 |
| meanK | masked-JEPA | 0.004130 | 0.000798 | 0.004419 |
| meanK | supervised | 0.386520 | 0.389158 | 0.307328 |

The gate is fail-closed. No null table or report may be produced after a gate
failure.

## 6. Observed role subspaces

Let `H4` be the normalized four-by-four Hadamard matrix and `I128` the channel
identity. For a 512D role-major readout:

```text
B_common     = H4[:, 0:1] tensor I128   # 512 x 128
B_complement = H4[:, 1:4] tensor I128   # 512 x 384
```

The first column is the normalized all-ones role direction. The complement is
the zero-sum role subspace. The names “common” and “complement” refer to these
linear operators, not to statistically independent information components.

## 7. Primary structured Haar null

Use 100 deterministic draws shared across every arm, encoder seed and readout.
The base seed is `20260826`.

For draw `b`, generate a Haar orthogonal matrix `Q_b` in `R^(4x4)` using a
Gaussian matrix, QR decomposition and deterministic diagonal-sign correction.
The per-draw random seed is derived from `SeedSequence([20260826, b, 4])`.

```text
B_common(b)     = Q_b[:, 0:1] tensor I128   # 512 x 128
B_complement(b) = Q_b[:, 1:4] tensor I128   # 512 x 384
```

Both subspaces are exact complements and have the same dimensions as the
observed Hadamard blocks. Bases and draw IDs are shared to preserve pairing.

## 8. Secondary generic feature-space null

As a secondary diagnostic, use 100 deterministic generic Haar subspaces in
`R^512` for each of dimensions 128 and 384. Seeds are derived from
`SeedSequence([20260826, b, m, 512])`.

The 128D and 384D generic draws are sampled independently and are not treated
as complementary blocks. This null tests arbitrary channel-and-role mixing;
it cannot replace the structured role-space null in the primary conclusion.

## 9. Required statistics

For every arm, encoder seed, readout, target, subspace kind and draw, record:

- raw validation R²;
- retention relative to full-readout R² when the full value exceeds `0.01`;
- subspace dimension and numerical rank;
- covariance-trace fraction captured by the subspace;
- descriptive `R² / trace_fraction`, invalid when the denominator is
  numerically zero;
- overlap with the variance-matched leading PCA span,
  `||U_m^T B||_F² / m`;
- energy of the full-readout directional coefficient span inside the subspace;
- fit status and failure reason.

For the observed common/complement pair also report:

- `full - common` and `full - complement`;
- intercept-only validation R²;
- shared/commonality term
  `common + complement - full - intercept_only`;
- two-block Shapley attribution:

```text
phi_common = 0.5 * [(common - intercept) + (full - complement)]
phi_complement = 0.5 * [(complement - intercept) + (full - common)]
```

Shapley and commonality are descriptive reader decompositions, not unique
information decompositions.

For each null family, fit the descriptive R²-versus-trace relationship using
null draws only and report the observed residual. No observed point is used to
fit its null trend.

## 10. Empirical probabilities and aggregation

For an observed common block:

```text
p_lower = (1 + count(null_R2 <= observed_R2)) / 101
```

For an observed complement:

```text
p_upper = (1 + count(null_R2 >= observed_R2)) / 101
```

Also report the observed percentile in the null using the raw count fraction.
The minimum attainable plus-one p-value is `1/101`.

The primary block statistic is the unweighted mean raw R² across the 12
independent directional targets within encoder seed. Per-target rows always
remain available. Encoder seeds are never collapsed before their individual
null comparisons are shown.

## 11. Decision rules

For an arm/readout, the common direction is `unusually_weak` only if its block
aggregate lower-tail p-value is at most `0.05` in all three encoder seeds. The
complement is `unusually_strong` only if its block aggregate upper-tail p-value
is at most `0.05` in all three seeds.

Interpretation is then restricted to:

1. JEPA common unusually weak and complement unusually strong, supervised not:
   role-axis-specific representation geometry; causal attribution to objective
   family remains blocked by supervised label exposure.
2. The same pattern in every arm: source/architecture-associated role
   structure, not a JEPA-specific mechanism.
3. Observed blocks typical under the structured null: no evidence that the
   Hadamard role axis is exceptional; retain only the fixed-projection
   operational observation.
4. Seed-, target- or readout-dependent results: mixed post-hoc diagnostic; no
   token-role mechanism is added to the simulator.

Generic-null results and per-direction ratios cannot override the structured
null classification.

## 12. Controls and fail-closed conditions

A deterministic shuffled-target control uses independent train and validation
row permutations derived from `SeedSequence([20260826, encoder_seed,
readout_index, split_index, 991])`. It is evaluated with the same observed and
structured-null bases.

The run fails closed for:

- any input, endpoint or historical-output hash mismatch;
- missing arm, seed, readout or target;
- non-finite inputs or outputs;
- basis non-orthogonality above `1e-10`;
- reproduction error above `5e-10`;
- missing or duplicated draw IDs;
- dimension/rank mismatch without an explicit invalid row;
- any report claim without a source artifact and hash.

## 13. Compute policy

Readout dumps are processed sequentially. Full sufficient statistics are
cached once per exact arm × encoder seed × readout and reused for every basis.
The implementation must not refit by rescanning 100,000 rows for every draw.

Before the full 100-draw grid:

1. run the complete historical reproduction gate;
2. benchmark one arm × seed × readout with five structured draws;
3. report projected runtime, peak RAM and output storage;
4. verify deterministic resume behavior.

CPU is the reference device. No encoder, MLP or GPU training is authorized.

## 14. Required artifacts

Output namespace:

```text
validation/experiment01/token_role_20260826/
```

Required files:

```text
protocol_frozen.json
input_gate.json
sufficient_statistics/
reproduction_per_target.parquet
reproduction_summary.parquet
reproduction_gate.json
benchmark.json
token_role_observed.parquet
token_role_structured_null.parquet
token_role_generic_null.parquet
token_role_commonality.parquet
token_role_null_summary.parquet
token_role_shuffled_control.parquet
token_role_failures.parquet
token_role_summary.json
figures/
REPORT_EXPERIMENT_01_TOKEN_ROLE.md
manifest.json
```

The manifest records all artifact hashes, source hashes, the specification hash
and the historical manifest identities. Public copies must not contain
absolute local paths.

## 15. Prohibited interpretations

This diagnostic cannot:

- change `A1`, `R3` or any prior threshold;
- establish information-theoretic loss;
- identify objective-family causality;
- treat `0.041` versus `0.205` as a dimension-corrected effect size;
- infer that role projection, temporal averaging and PCA are one mechanism;
- use the generic 512D null in place of the structured role-space null;
- use validation results to alter draws, thresholds or decision rules.

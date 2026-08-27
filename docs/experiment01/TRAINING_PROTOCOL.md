# Experiment 01 — training protocol and hyperparameter audit

- **Audit status:** passed, fail-closed
- **Canonical inventory:** 3 objectives × 3 encoder seeds
- **Scientific checkpoint rule:** epoch 20 for every arm and seed
- **Machine audit SHA-256:** `fd3b1f24fa3237a9657b8cfc66ba6651326d6e138c272a4316f328c2dff6c0cb`

This document separates what was genuinely matched across the historical
encoder arms from objective-required differences and from residual training
confounds. The machine-readable evidence is
[`TRAINING_PROTOCOL_AUDIT.json`](TRAINING_PROTOCOL_AUDIT.json).

## 1. Shared encoder and tokenization

Every arm uses the same non-causal grid backbone. A 20-snapshot input
window produces four role tokens at each timestep, so the encoder output is
`20 × 4 × 128`. Experiment 01 reads either the four final-timestep tokens
(`last_concat512`) or the four tokens after averaging each role through
time (`meanK_concatS`).

| property | frozen value |
|---|---:|
| window length `K` | 20 |
| role tokens per timestep `S` | 4 |
| model dimension | 128 |
| spatial transformer | 2 layers, 4 heads, FFN 256 |
| temporal transformer | 2 layers, 4 heads, FFN 256, non-causal |
| dropout | 0.1 |
| stocks | 7 |
| encoder parameters | 535,680 |

The token roles and normalization are shared. The nine checkpoints agree on
the stock-statistics fingerprint `0938cfda578710436220d0a31fd5d6101c496dafa1512784a4dddef3a44f7900`.

## 2. Shared optimization envelope

| property | frozen value |
|---|---:|
| grouped split seed | 0 |
| candidate train cap | 500,000 endpoints |
| candidate validation cap | 50,000 endpoints |
| batch size | 256 |
| epochs | 20 |
| train batches per epoch (`drop_last=True`) | 1,953 |
| gradient-bearing rows per epoch | 499,968 |
| optimizer updates | 39,060 |
| optimizer | AdamW, betas=(0.9, 0.999), eps=1e-8 |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| gradient-norm clip | 1.0 |
| final LR fraction | 0.01 |

These equal row and update caps do **not** imply equal FLOPs. JEPA also
evaluates a predictor and EMA target network, whereas supervised training
uses an attention-pooling regression head.

## 3. Objective-required differences

| arm | objective | auxiliary trainable parameters | schedule |
|---|---|---:|---|
| `jepa_horizon` | L1 latent prediction at horizons 0,1,5,10,20; EMA target | 152,704 | epoch-level cosine |
| `jepa_masked` | L1 structured masked latent prediction; mask ratio 0.50–0.65; EMA target | 152,448 | epoch-level cosine |
| `supervised` | MSE on 22 standardized future targets | 36,246 | warm-up + cosine, block-stepped per epoch |

The supervised target inventory contains 20 future-feature targets
(`spread`, `microprice`, `best bid`, `best ask`, `top imbalance` at
horizons 1, 5, 10 and 20) plus realized volatility at horizons 5 and
20. These include the target-aligned quantities later probed by
Experiment 01. Timing was not a direct supervised target, although it is
correlated with the trained targets. Consequently, Phase-I reader-label
comparisons are representation-conditional and are not an end-to-end
label-efficiency comparison. F16 is designed to measure this dependence.

### Historical supervised scheduler detail

The archived implementation describes the scheduler as per-update, but
calls `scheduler.step()` 1,953 times in a loop only after completing each
epoch. The learning rate is therefore constant during an epoch and jumps
between epochs. This is a historical implementation property, not a
reinterpretation of the checkpoints. F16 uses an explicitly update-based
scheduler and treats the canonical supervised checkpoint only as a
descriptive anchor.

## 4. Exact historical row-identity reconstruction

The canonical dataset hash is `7617dbbfcee56377f980a606267397861f6613017f0a2aca1e218407726ef862`. After
the archived endpoint filters there are
7,323,510 valid endpoints:
6,596,688 on the historical train
side and 726,822 on its
held-out side.

For a fixed encoder seed, all three arms select the **same 500,000 train
endpoints**. Thus arm comparisons paired at the same seed are row-matched.
Across encoder seeds, however, the sampling seed changes the training-row
sample as well as initialization and minibatch order:

| seed pair | shared train rows | fraction of each 500k sample |
|---|---:|---:|
| 0–1 | 37,824 | 7.565% |
| 0–2 | 38,064 | 7.613% |
| 1–2 | 37,587 | 7.517% |

Therefore the historical three-seed dispersion combines initialization,
minibatch and data-subsample variation; it is not a pure optimization-seed
error bar. F16 removes this confound by freezing the exact Phase-I seed-0
row manifest and reusing it for encoder seeds 0, 1 and 2.

The historical validation subsets are not matched between supervised and
JEPA: supervised reuses the RNG after drawing train rows, while JEPA uses
a fresh RNG seeded with `seed+1`. This does not alter the frozen epoch-20
Phase-I representation comparison, but it prevents interpreting the old
training-time validation metrics as a perfectly matched arm comparison.

Row hashes were not embedded in the checkpoints. Their identities above
are deterministic reconstructions from the canonical dataset and archived
sampling implementation; this provenance limit is explicit in the JSON
audit.

## 5. Checkpoint selection and validation history

The distributed scientific inventory deliberately selects epoch 20 for
every cell. The training scripts also maintained a lowest-validation-loss
`best.pt`, but that alias is not the canonical scientific checkpoint.

| arm | seed | epoch-20 validation objective | best epoch | best objective |
|---|---:|---:|---:|---:|
| `jepa_horizon` | 0 | 0.139546623 | 20 | 0.139546623 |
| `jepa_horizon` | 1 | 0.146000893 | 20 | 0.146000893 |
| `jepa_horizon` | 2 | 0.147569372 | 19 | 0.147285903 |
| `jepa_masked` | 0 | 0.268552691 | 8 | 0.250730215 |
| `jepa_masked` | 1 | 0.265289530 | 6 | 0.243376044 |
| `jepa_masked` | 2 | 0.259140984 | 7 | 0.242791347 |
| `supervised` | 0 | 0.580000290 | 20 | 0.580000290 |
| `supervised` | 1 | 0.585118810 | 20 | 0.585118810 |
| `supervised` | 2 | 0.577262107 | 19 | 0.577002257 |

The masked objective reached its historical validation minimum at epochs
6–8 and was worse by epoch 20. Horizon seed 2 and supervised seed 2 were
minimally better at epoch 19. These facts are reported descriptively; the
frozen Phase-I checkpoint rule is not changed.

## 6. Canonical checkpoint inventory

| arm | seed | path | bytes | SHA-256 |
|---|---:|---|---:|---|
| `jepa_horizon` | 0 | `checkpoints/multiseed/jepa_horizon/seed0/epoch_020.pt` | 10,562,059 | `756aa9dfd88b65eb5cfabca8e2d93c6fefa52994e39ae321f3ad23435f5ea619` |
| `jepa_horizon` | 1 | `checkpoints/multiseed/jepa_horizon/seed1/epoch_020.pt` | 10,562,059 | `fa3fc130f3421895130ea2174a88f698fb7db1692f8de6d76368153d2a3a096a` |
| `jepa_horizon` | 2 | `checkpoints/multiseed/jepa_horizon/seed2/epoch_020.pt` | 10,562,059 | `c4a5d3c011ff79b1cc887f0f7045f55aba043ecacd0b054280a470507f0b9ecc` |
| `jepa_masked` | 0 | `checkpoints/multiseed/jepa_masked/seed0/epoch_020.pt` | 10,560,779 | `4a20451e3ab47a3ba6dab14cbbaaa076027fb075a384f7ab749259492e57d189` |
| `jepa_masked` | 1 | `checkpoints/multiseed/jepa_masked/seed1/epoch_020.pt` | 10,560,779 | `75a74630448b0244e99df781e4c4db1140e79b9bc3d8dd9ccf54d88365b9289f` |
| `jepa_masked` | 2 | `checkpoints/multiseed/jepa_masked/seed2/epoch_020.pt` | 10,560,779 | `67ee7db0ee92b4402225e3d1161fe7c71d55c545cdbc72c1a5b115baddecb94d` |
| `supervised` | 0 | `checkpoints/multiseed/supervised/seed0/epoch_020.pt` | 6,943,627 | `92657fe4c1b6c1ee1e3e0b0f3b31d1ca92980cb1c63cbe08c7c010c9cfd468db` |
| `supervised` | 1 | `checkpoints/multiseed/supervised/seed1/epoch_020.pt` | 6,943,627 | `1ec5fa7df081e6c4c392a19c13a21bd24895507d13998efb68680aaa8424f8ff` |
| `supervised` | 2 | `checkpoints/multiseed/supervised/seed2/epoch_020.pt` | 6,943,627 | `29ff331f3f531cdbba9c14a223b577014f03a97efe42fada9eec03737b68d7ed` |

## 7. Evidentiary limits and F16 consequences

The checkpoint payloads contain configuration, optimizer state, validation
metrics, normalization statistics and target inventory. They do not contain
a Git commit, training-source hash, hardware identity or wall-clock log.
The source hashes in the machine audit describe the current archived
implementation, which is consistent with the payloads but cannot prove the
exact training-time checkout.

F16 therefore freezes prospectively:

- exact row-key manifests shared across encoder seeds;
- an exact validation cohort and validation-only stopping rule;
- a genuinely per-update LR schedule and common maximum-update cap;
- RNG state, optimizer/scheduler state and resumable checkpoint identity;
- a sealed test barrier until every validation selection is frozen;
- the canonical epoch-20 model only as a descriptive upper anchor.

No statement in this audit changes Phase I, its thresholds or technical
classification A1.

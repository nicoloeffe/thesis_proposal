# Experiment 01 — T2 token-role matched-null diagnostic

## Result

The dimension-matched structured role-Haar null is **not rejected**. The fixed
all-ones role projection remains an operational cross-encoder contrast, but the
data do not support treating the Hadamard common direction or its zero-sum
complement as a privileged relational mechanism.

This is a preregistered corrective, post-hoc diagnostic on the historical
100,000/50,000 train/validation endpoint sample. It does not use the production
test split and does not change Phase-I `A1` or Phase-III-R `R3`.

## Reproduction gate

Before evaluating a null, the implementation reproduced every historical
full/common/complement OLS cell for three arms, three encoder seeds, two
readouts and 22 targets:

- cells: `1,188`;
- per-target maximum absolute error: `1.7927e-12`;
- registered tolerance: `5e-10`;
- status: passed.

The independent-directional aggregate was reproduced as follows:

| readout | arm | full 512D | common 128D | complement 384D |
|---|---|---:|---:|---:|
| last | horizon-JEPA | 0.211129 | 0.041423 | 0.204832 |
| last | masked-JEPA | 0.100645 | 0.015770 | 0.091788 |
| last | supervised | 0.375636 | 0.372956 | 0.333051 |
| meanK | horizon-JEPA | 0.063059 | 0.008961 | 0.057360 |
| meanK | masked-JEPA | 0.004130 | 0.000798 | 0.004419 |
| meanK | supervised | 0.386520 | 0.389158 | 0.307328 |

These independently fitted out-of-sample R² values are not additive
information components.

## Primary structured null

Each observed block was compared with 100 deterministic Haar rotations in the
four-dimensional role space. Every rotation preserves the 128D common-like
direction and its matched 384D complement after lifting through the 128 feature
channels. The statistic below is the unweighted mean over the 12 independent
directional targets within encoder seed.

For the primary `last_concat512` horizon-JEPA comparison:

| seed | block | observed R² | null mean±sd | preregistered p | trace fraction | trace-conditioned residual |
|---:|---|---:|---:|---:|---:|---:|
| 0 | common | 0.048167 | 0.089126±0.037702 | 0.0990 | 0.5742 | -0.010459 |
| 0 | complement | 0.208835 | 0.197175±0.014996 | 0.1287 | 0.4258 | -0.001110 |
| 1 | common | 0.036529 | 0.083854±0.036528 | 0.0594 | 0.5744 | -0.022594 |
| 1 | complement | 0.203157 | 0.190879±0.015993 | 0.1287 | 0.4256 | 0.002384 |
| 2 | common | 0.039572 | 0.081253±0.035459 | 0.0990 | 0.4932 | -0.026166 |
| 2 | complement | 0.202503 | 0.189373±0.016318 | 0.0990 | 0.5068 | 0.008664 |

For common, `p` is the plus-one lower-tail probability. For complement it is
the plus-one upper-tail probability. The preregistered decision requires
`p <= 0.05` in all three encoder seeds.

## Decision table

| arm/readout | common p by seed | complement p by seed | common weak all seeds | complement strong all seeds |
|---|---|---|---|---|
| horizon/last | .0990 / .0594 / .0990 | .1287 / .1287 / .0990 | no | no |
| horizon/meanK | .0693 / .0495 / .0990 | .3168 / .1089 / .2277 | no | no |
| masked/last | .1881 / .1287 / .2178 | .1980 / .0594 / .2673 | no | no |
| masked/meanK | .0891 / .2970 / .1683 | .2871 / .2475 / .4455 | no | no |
| supervised/last | .9901 / .9208 / .8812 | 1.0000 / 1.0000 / 1.0000 | no | no |
| supervised/meanK | .9703 / .8812 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | no | no |

The same joint “common unusually weak and complement unusually strong” pattern
therefore fails for every arm/readout. One isolated horizon/meanK common cell
reaches `0.0495`; the all-three-seed rule prevents treating it as a robust
mechanism.

## Controls and compute

- Generic 512D subspaces, trace-conditioned residuals, PCA overlap,
  directional coefficient-span energy and commonality/Shapley decompositions
  are serialized as secondary diagnostics; they cannot override the structured
  null decision.
- Shuffled-target directional block means range from `-0.0042` to `-0.0011`.
- Grid runtime: `177.08 s`.
- Peak RAM: `518,615,040` bytes (about `495 MiB`).
- Output size: `97,123,838` bytes (about `92.6 MiB`).
- Failures: `0`.

## Interpretation for the simulator

The fixed all-ones projection can remain in the empirical record because its
cross-encoder retention difference is real. It should not be encoded as a
special high-variance relational axis, nor unified with temporal pooling or
PCA anti-alignment. The simulator can safely target the dimension-matched
spectral anti-alignment and the selective temporal-pooling fragility; the role
mechanism remains absent.

## Integrity

- specification SHA-256:
  `51db7780a12fefe41e3191e5fef77ce4e87deee5becf861a0000fc8c30864a68`;
- production manifest SHA-256:
  `ef23d6517d20252c1cfd58a0e89e86f8093b91ca7867a92274d240df9b0fdc83`;
- canonical output:
  `validation/experiment01/token_role_20260826/`;
- canonical report:
  `validation/experiment01/token_role_20260826/REPORT_EXPERIMENT_01_TOKEN_ROLE.md`.

The canonical manifest hashes 145 required artifacts, including all sufficient
statistics and resumable feature-set shards.

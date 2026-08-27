# Experiment 01 Phase III-R — compute-feasible amendment

This amendment is subordinate to the later definitive Phase-III v1
specification,
[`SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md`](https://github.com/nicoloeffe/thesis_proposal/blob/main/docs/experiment01/SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md),
SHA-256
`78ca15821ac40355c35e5f40ecaf5086f5e6bbb6f339255a85b13fc7d952a151`.
That specification replaces the eligibility rule in the earlier optional MLP
section: the executed `b_1_4` floor is therefore eligible under the governing
Phase-III contract.

Phase III v1 was stopped for computational infeasibility before the selection
manifest was frozen and before any production test access. Phase III-R changes
only the job inventory. It preserves the frozen bundle, splits, subset row
identities, feature transforms, target blocks, MLP architecture, optimizer,
step schedule, weight-decay grid, validation-only selection, test boundary,
metrics, thresholds, bootstrap settings and R1--R4 outcome rules.

The primary grid keeps directional `last_concat512`, both branches, all three
encoder seeds, native/full-whitened coordinates, adjacent low budgets `b_1_4`
and `b_1_2`, full train, subset seeds 0--2 at low budgets and reader seeds
0--2. Volatility and timing remain separate controls at `b_1_4` and full train.
The spectral diagnostic is restricted to horizon-JEPA directional head
`1:127`, deep `382:508` and `full_valid_rank`. Capacity sensitivity is omitted.

The amendment was selected from scientific contrast requirements and measured
runtime/completeness only. Aggregate validation performance was not inspected.
During the post-confirmation implementation audit, one individual timing-cell
validation value was incidentally displayed while inspecting artifact schema;
it was not aggregated, interpreted or used to alter this already confirmed
inventory. Any insufficient precision yields R4; the grid is not expanded
after test access.

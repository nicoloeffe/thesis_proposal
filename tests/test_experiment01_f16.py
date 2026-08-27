from __future__ import annotations

import pandas as pd
import numpy as np

from experiment01.f16 import CAP_CANDIDATES, cohort_for_cap, select_nested_cohort
from experiment01.f16_convergence import pc_stats_to_raw, projection_stats, role_projections
from experiment01.f16_training import (
    DeterministicPassBatchSampler,
    F16TrainingConfig,
    learning_rate_multiplier,
)
from experiment01.f16_planning import projected_updates
from experiment01.f16_production import GRID
from experiment01.f16_evaluation import (
    _largest_alpha_index_within_1e12,
    _poolings,
    _stats_arrays,
    _stats_from_npz,
    _target_blocks,
    WHITENING_DEPTHS,
)
from experiment01.f16_test import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    _bootstrap_weights,
    _weighted_block_r2,
)
from experiment01.f16_posttest import strict_json_safe_fingerprint
from experiment01.f16_posttest_threshold import BOUNDARY_TOLERANCE, SPEARMAN_THRESHOLD
from experiment01.io import canonical_json_sha256
from experiment01.linear import SufficientStats, sufficient_stats


def _rows() -> pd.DataFrame:
    records = []
    position = 0
    for stock_id, date, count in ((0, "2019-01-02", 7), (0, "2019-01-03", 4), (1, "2019-01-02", 9)):
        for order in range(count):
            records.append(
                {
                    "row_key": f"{stock_id}|{date}|{order}",
                    "stock_id": stock_id,
                    "stock_symbol": f"s{stock_id}",
                    "stock_day_id": position // max(count, 1),
                    "trading_date": date,
                    "endpoint_index": 100 + position,
                    "endpoint_order": order,
                    "timestamp_ns": 1_000_000 + position,
                }
            )
            position += 1
    return pd.DataFrame(records)


def test_nested_cohort_is_deterministic_and_covers_every_stock_day() -> None:
    rows = _rows()
    first = select_nested_cohort(rows, "validation", max_cap=5)
    second = select_nested_cohort(rows, "validation", max_cap=5)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5 + 4 + 5
    assert first[["stock_id", "trading_date"]].drop_duplicates().shape[0] == 3


def test_cap_prefixes_are_nested() -> None:
    rows = _rows()
    cohort = select_nested_cohort(rows, "train", max_cap=max(CAP_CANDIDATES))
    prior: set[str] = set()
    for cap in CAP_CANDIDATES:
        current = set(cohort_for_cap(cohort, cap)["row_key"])
        assert prior.issubset(current)
        prior = current


def test_split_domain_changes_deterministic_selection() -> None:
    rows = _rows()
    train = select_nested_cohort(rows, "train", max_cap=3)
    validation = select_nested_cohort(rows, "validation", max_cap=3)
    assert set(train["row_key"]) != set(validation["row_key"])


def test_pc_stats_roundtrip_recovers_raw_sufficient_statistics() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 8))
    y = rng.normal(size=(80, 3))
    mean = rng.normal(size=8)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    z = (x - mean) @ q
    recovered = pc_stats_to_raw(sufficient_stats(z, y), mean, q)
    expected = sufficient_stats(x, y)
    assert recovered.n == expected.n
    np.testing.assert_allclose(recovered.x_sum, expected.x_sum, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(recovered.xtx, expected.xtx, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(recovered.xty, expected.xty, rtol=1e-12, atol=1e-12)


def test_role_projections_are_orthogonal_and_stats_match_direct_projection() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(40, 512))
    y = rng.normal(size=(40, 2))
    stats = sufficient_stats(x, y)
    common, contrast = role_projections()
    for matrix in (common, contrast):
        projected = projection_stats(stats, matrix)
        expected = sufficient_stats(x @ matrix, y)
        np.testing.assert_allclose(projected.x_sum, expected.x_sum, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(projected.xtx, expected.xtx, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(projected.xty, expected.xty, rtol=1e-12, atol=1e-12)


def test_f16_batch_order_is_resumable_from_global_update() -> None:
    full = DeterministicPassBatchSampler(11, 4, seed=2, start_update=0, maximum_updates=8)
    batches = list(full)
    resumed = DeterministicPassBatchSampler(11, 4, seed=2, start_update=5, maximum_updates=8)
    assert list(resumed) == batches[5:]
    assert full.steps_per_pass == 3
    assert sorted(batches[0] + batches[1] + batches[2]) == list(range(11))


def test_f16_learning_rate_schedule_hits_frozen_endpoints() -> None:
    config = F16TrainingConfig()
    assert learning_rate_multiplier(0, config) == 1 / config.warmup_updates
    assert learning_rate_multiplier(config.warmup_updates - 1, config) == 1.0
    assert learning_rate_multiplier(config.maximum_updates - 1, config) == config.terminal_lr_fraction


def test_f16_projected_updates_respect_epoch20_and_maximum() -> None:
    assert projected_updates(7_116) == (28, 560, 6_500, 39_060)
    assert projected_updates(122_099) == (477, 9_540, 9_540, 39_060)
    assert projected_updates(490_937) == (1_918, 38_360, 38_360, 39_060)


def test_f16_production_grid_has_frozen_budget_major_order() -> None:
    assert GRID == (
        ("b_1_4", 0),
        ("b_1_4", 1),
        ("b_1_4", 2),
        ("b_1", 0),
        ("b_1", 1),
        ("b_1", 2),
        ("b_4", 0),
        ("b_4", 1),
        ("b_4", 2),
        ("b_16", 0),
        ("b_16", 1),
        ("b_16", 2),
    )


def test_f16_whitening_depths_are_exact_phase1_bridge() -> None:
    assert WHITENING_DEPTHS == (0, 8, 16, 32, 64, 128, 256, 508)


def test_f16_poolings_preserve_role_major_512_contract() -> None:
    grid = __import__("torch").arange(2 * 3 * 4 * 128).reshape(2, 3, 4, 128).float()
    values = _poolings(grid)
    assert values["last_concat512"].shape == (2, 512)
    assert values["meanK_concatS"].shape == (2, 512)
    np.testing.assert_array_equal(values["last_concat512"][0], grid[0, -1].reshape(-1).numpy())
    np.testing.assert_array_equal(values["meanK_concatS"][0], grid[0].mean(0).reshape(-1).numpy())


def test_f16_sufficient_stats_serialization_roundtrip() -> None:
    rng = np.random.default_rng(12)
    stats = sufficient_stats(rng.normal(size=(20, 5)), rng.normal(size=(20, 3)))
    arrays = _stats_arrays(stats, "sample")
    recovered = _stats_from_npz(arrays, "sample")
    assert recovered.n == stats.n
    np.testing.assert_allclose(recovered.x_sum, stats.x_sum)
    np.testing.assert_allclose(recovered.y_sum, stats.y_sum)
    np.testing.assert_allclose(recovered.xtx, stats.xtx)
    np.testing.assert_allclose(recovered.xty, stats.xty)
    np.testing.assert_allclose(recovered.yty, stats.yty)


def test_f16_target_blocks_use_independent_offsets_within_each_block() -> None:
    definitions = [
        {"block": "directional", "independent": True},
        {"block": "directional", "independent": False},
        {"block": "volatility", "independent": True},
        {"block": "timing", "independent": True},
    ]
    blocks = _target_blocks(definitions)
    np.testing.assert_array_equal(blocks["directional"][0], [0, 1])
    np.testing.assert_array_equal(blocks["directional"][1], [0])
    np.testing.assert_array_equal(blocks["volatility"][1], [0])
    np.testing.assert_array_equal(blocks["timing"][1], [0])


def test_f16_alpha_tie_break_is_largest_grid_index_within_absolute_1e12() -> None:
    scores = np.asarray([0.2, 0.2 - 2e-12, 0.2 - 5e-13])
    assert _largest_alpha_index_within_1e12(scores) == 2


def test_f16_grouped_bootstrap_is_hierarchical_and_deterministic() -> None:
    groups = pd.DataFrame(
        {
            "stock_id": [0, 0, 1, 1, 1],
            "trading_date": ["a", "b", "a", "b", "c"],
        }
    )
    first = _bootstrap_weights(groups)
    second = _bootstrap_weights(groups)
    assert BOOTSTRAP_DRAWS == 5000
    assert BOOTSTRAP_SEED == 20260827
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5000, 5)
    assert np.all(first.sum(axis=1) >= 4)


def test_f16_grouped_r2_reconstructs_weighted_residual_definition() -> None:
    groups = pd.DataFrame({"stock_id": [0, 1], "trading_date": ["a", "b"]})
    frame = pd.DataFrame(
        {
            "stock_id": [0, 1],
            "trading_date": ["a", "b"],
            "target_index": [0, 0],
            "target_independent": [True, True],
            "n_rows": [2, 2],
            "y_sum": [1.0, 5.0],
            "yty": [1.0, 13.0],
            "residual_ss": [0.25, 0.75],
        }
    )
    observed = _weighted_block_r2(frame, np.ones((1, 2)), groups)[0]
    expected = 1.0 - 1.0 / (14.0 - 36.0 / 4.0)
    assert observed == expected


def test_f16_posttest_serialization_maps_nan_to_strict_json_null() -> None:
    assert strict_json_safe_fingerprint({"value": float("nan")}) == canonical_json_sha256(
        {"value": None}
    )


def test_f16_spearman_boundary_treats_binary_float_0_8_as_at_least_0_8() -> None:
    observed = 0.7999999999999999
    assert observed < SPEARMAN_THRESHOLD
    assert observed >= SPEARMAN_THRESHOLD - BOUNDARY_TOLERANCE

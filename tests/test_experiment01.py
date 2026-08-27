from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiment01.constants import (
    ALPHA_GRID,
    INPUT_SCHEMA,
    INPUT_SCHEMA_VERSION,
    READOUTS,
    RESULT_COLUMNS,
)
from experiment01.bundle import timing_target_all_rows
from experiment01.errors import ExperimentIntegrityError
from experiment01.io import canonical_json_sha256, sha256_array, sha256_file
from experiment01.linear import (
    SufficientStats,
    direct_ridge_solution,
    eigensystem,
    evaluate,
    evaluate_stats,
    fit_alpha,
    fit_unlabelled_covariance,
    predict,
    r2_per_target,
    sufficient_stats,
    transformed_design,
    tune_alpha,
    whitening_transform,
)
from experiment01.metadata import build_metadata_sidecar
from experiment01.pipeline import Phase1Config, run_phase1
from experiment01.reporting import generate_phase1_report
from experiment01.results import (
    attach_operational_ceilings,
    hierarchical_interval,
    paired_gap_points,
    uncertainty_summary,
    validate_result_schema,
)
from experiment01.schema import (
    READOUT_DEFINITIONS,
    FeatureSet,
    InputBundle,
    TargetDefinition,
    load_input_bundle,
)
from experiment01.sharded import (
    SHARDED_STORAGE,
    ArrayShard,
    ShardedArray,
    sharded_record_fingerprint_payload,
)
from experiment01.split3 import chronological_heldout_halves
from experiment01.subsets import (
    anchor_sensitivity,
    budget_schedule,
    generate_all_selections,
    nested_interval,
    selections_for_seed,
)
from experiment01.summary import summarize_phase1


def _training_rows(days=(5, 6), rows_per_day=16) -> pd.DataFrame:
    values = []
    raw_index = 0
    for stock, count in enumerate(days):
        for day in range(count):
            for order in range(rows_per_day):
                values.append(
                    {
                        "row_key": f"{stock}|{day}|{raw_index}",
                        "stock_id": stock,
                        "stock_symbol": f"S{stock}",
                        "stock_day_id": day,
                        "trading_date": f"2026-01-{day + 1:02d}",
                        "endpoint_index": raw_index,
                        "endpoint_order": order,
                        "timestamp_ns": 1_000_000_000 * (raw_index + 1),
                    }
                )
                raw_index += 1
    return pd.DataFrame(values)


def test_alpha_grid_is_exact_preregistered_grid():
    assert len(ALPHA_GRID) == 32
    assert ALPHA_GRID[0] == 0.0
    np.testing.assert_array_equal(ALPHA_GRID[1:], np.logspace(-8, 4, 31))


def test_chronological_heldout_halves_puts_odd_extra_day_in_test():
    validation, test = chronological_heldout_halves(
        ["2019-01-05", "2019-01-01", "2019-01-03", "2019-01-02", "2019-01-04"]
    )
    assert validation == ("2019-01-01", "2019-01-02")
    assert test == ("2019-01-03", "2019-01-04", "2019-01-05")


def test_vectorized_timing_target_matches_canonical_loop():
    mid = np.asarray([1, 1, 2, 2, 2, 2, 2, 3, 3, 3], dtype=np.float32)
    stock = np.asarray([0] * 5 + [0] * 5, dtype=np.int32)
    day = np.asarray([0] * 5 + [1] * 5, dtype=np.int32)
    observed = timing_target_all_rows(mid, stock, day, max_look=3)
    expected_duration = []
    for index in range(len(mid)):
        step = 1
        duration = None
        while (
            index + step < len(mid)
            and stock[index + step] == stock[index]
            and day[index + step] == day[index]
            and step <= 3
        ):
            if mid[index + step] != mid[index]:
                duration = step
                break
            step += 1
        expected_duration.append(min(step, 3) if duration is None else duration)
    expected = np.log1p(expected_duration).astype(np.float32)
    np.testing.assert_array_equal(observed, expected)


def test_metadata_sidecar_reads_csv_identity_and_gates_full_npz(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    chunks = []
    for stock_id in range(7):
        rows = []
        for timestamp, collapsed in (
            ("2019-01-02 09:30:00", False),
            ("2019-01-02 14:57:00", False),
            ("2019-01-02 09:31:00", True),
        ):
            row = {"index": timestamp}
            for level in range(1, 11):
                row[f"BidPrice{level}"] = (
                    1.0 if collapsed else 1.0 - (level - 1) * 0.01
                )
                row[f"AskPrice{level}"] = (
                    1.01 if collapsed else 1.01 + (level - 1) * 0.01
                )
                row[f"BidVolume{level}"] = float(level)
                row[f"AskVolume{level}"] = float(level + 1)
            rows.append(row)
        path = raw_root / f"sz{stock_id:06d}-level10_processed.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        chunks.append(
            __import__(
                "scripts.dataset.build_encoder_dataset_lobench",
                fromlist=["process_csv"],
            ).process_csv(path, stock_id)
        )
    dataset = tmp_path / "processed.npz"
    np.savez_compressed(
        dataset,
        book=np.concatenate([chunk["book"] for chunk in chunks]),
        mid_z=np.concatenate([chunk["mid_z"] for chunk in chunks]),
        stock_ids=np.concatenate([chunk["stock_ids"] for chunk in chunks]),
        day_ids=np.concatenate([chunk["day_ids"] for chunk in chunks]),
        min_spread_z_per_stock=np.asarray(
            [chunk["min_spread_z"] for chunk in chunks], dtype=np.float32
        ),
        price_std_rmb_per_stock=np.asarray(
            [chunk["price_std_rmb"] for chunk in chunks], dtype=np.float32
        ),
    )
    payload = build_metadata_sidecar(
        raw_root,
        dataset,
        tmp_path / "sidecar",
        chunk_rows=2,
        expected_total_rows=7,
    )
    assert payload["equivalence"]["passed"] is True
    sidecar = pd.read_parquet(
        tmp_path / "sidecar" / "metadata_sidecar.parquet"
    )
    assert sidecar["global_row_index"].tolist() == list(range(7))
    assert sidecar["stock_symbol"].tolist() == [
        f"sz{stock_id:06d}" for stock_id in range(7)
    ]
    assert sidecar["raw_csv_row_index"].tolist() == [0] * 7
    assert sidecar["endpoint_order"].tolist() == [0] * 7
    assert sidecar["trading_date"].tolist() == ["2019-01-02"] * 7


def test_sharded_array_supports_slices_fancy_indices_and_columns(tmp_path):
    values = np.arange(42, dtype=np.float32).reshape(14, 3)
    paths = []
    shards = []
    for index, (start, stop) in enumerate(((0, 5), (5, 9), (9, 14))):
        path = tmp_path / f"part-{index}.npy"
        np.save(path, values[start:stop])
        paths.append(path)
        shards.append(ArrayShard(path, start, stop))
    sharded = ShardedArray(shards, values.shape, values.dtype)
    np.testing.assert_array_equal(sharded[3:12], values[3:12])
    np.testing.assert_array_equal(
        sharded[np.asarray([11, 0, 8, 8])],
        values[np.asarray([11, 0, 8, 8])],
    )
    np.testing.assert_array_equal(sharded[2:10, 1], values[2:10, 1])
    np.testing.assert_array_equal(sharded[-1], values[-1])


def test_sufficient_stats_and_covariance_stream_sharded_arrays(tmp_path):
    rng = np.random.default_rng(901)
    x = rng.normal(size=(31, 5)).astype(np.float32)
    y = rng.normal(size=(31, 3)).astype(np.float32)
    x_shards = []
    y_shards = []
    for index, (start, stop) in enumerate(((0, 7), (7, 19), (19, 31))):
        x_path = tmp_path / f"x-{index}.npy"
        y_path = tmp_path / f"y-{index}.npy"
        np.save(x_path, x[start:stop])
        np.save(y_path, y[start:stop])
        x_shards.append(ArrayShard(x_path, start, stop))
        y_shards.append(ArrayShard(y_path, start, stop))
    sharded_x = ShardedArray(x_shards, x.shape, x.dtype)
    sharded_y = ShardedArray(y_shards, y.shape, y.dtype)
    observed = sufficient_stats(sharded_x, sharded_y, chunk_rows=6)
    expected = sufficient_stats(x, y, chunk_rows=6)
    for name in ("x_sum", "y_sum", "xtx", "xty", "yty"):
        np.testing.assert_allclose(
            getattr(observed, name), getattr(expected, name)
        )
    observed_cov = fit_unlabelled_covariance(sharded_x, chunk_rows=6)
    expected_cov = fit_unlabelled_covariance(x, chunk_rows=6)
    np.testing.assert_allclose(observed_cov.mean, expected_cov.mean)
    np.testing.assert_allclose(
        observed_cov.covariance, expected_cov.covariance
    )


@pytest.mark.parametrize("n_rows", range(1, 18))
def test_nested_center_intervals_are_nested(n_rows):
    for anchor in range(n_rows):
        previous = set()
        for length in range(1, n_rows + 1):
            start, end = nested_interval(n_rows, length, anchor)
            current = set(range(start, end))
            assert len(current) == length
            assert anchor in current
            assert previous.issubset(current)
            previous = current


def test_subsets_are_deterministic_and_nested():
    rows = _training_rows()
    left = selections_for_seed(rows, 3)
    right = selections_for_seed(rows, 3)
    assert [value.budget.label for value in left] == [
        value.budget.label for value in right
    ]
    for first, second in zip(left, right):
        np.testing.assert_array_equal(first.row_indices, second.row_indices)
        assert first.row_key_sha256 == second.row_key_sha256
    ordered = sorted(
        [value for value in left if not value.budget.is_full_train],
        key=lambda value: value.budget.days_per_stock,
    )
    for previous, current in zip(ordered, ordered[1:]):
        assert set(previous.row_indices).issubset(current.row_indices)


def test_fractional_day_is_first_integer_day_for_every_stock():
    rows = _training_rows()
    selections = selections_for_seed(rows, 0)
    fractional = [
        value for value in selections if value.budget.days_per_stock == 0.5
    ][0]
    one_day = [
        value for value in selections if value.budget.days_per_stock == 1.0
    ][0]
    fractional_groups = set(
        zip(
            rows.iloc[fractional.row_indices]["stock_id"],
            rows.iloc[fractional.row_indices]["stock_day_id"],
        )
    )
    one_day_groups = set(
        zip(
            rows.iloc[one_day.row_indices]["stock_id"],
            rows.iloc[one_day.row_indices]["stock_day_id"],
        )
    )
    assert fractional_groups == one_day_groups


def test_adaptive_seed_schedule_and_full_train_once():
    rows = _training_rows()
    selections = generate_all_selections(rows)
    counts = {}
    for value in selections:
        counts.setdefault(value.budget.label, set()).add(value.subsample_seed)
    schedule = {value.label: value for value in budget_schedule(rows)}
    for label, budget in schedule.items():
        if budget.is_full_train:
            assert counts[label] == {-1}
        else:
            assert len(counts[label]) == budget.minimum_seeds


def test_time_of_day_sensitivity_uses_declared_positions():
    rows = _training_rows()
    values = anchor_sensitivity(rows, seeds=[0])
    means = sorted(
        {
            round(float(value.anchor_quantile), 8)
            for value in values
            if value.budget.days_per_stock == 0.5
        }
    )
    assert means[0] == 0.0
    assert means[-1] == 1.0
    assert 0.45 <= means[1] <= 0.55


def test_incremental_stats_equal_direct_recomputation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, 7))
    y = rng.normal(size=(90, 3))
    incremental = SufficientStats.zeros(7, 3)
    incremental.add_rows(x[:20], y[:20])
    incremental.add_rows(x[20:55], y[20:55])
    incremental.add_rows(x[55:], y[55:])
    direct = sufficient_stats(x, y)
    assert incremental.n == direct.n
    for name in ("x_sum", "y_sum", "xtx", "xty", "yty", "gram", "cross"):
        np.testing.assert_allclose(getattr(incremental, name), getattr(direct, name))


@pytest.mark.parametrize("alpha", [0.0, 1e-8, 1e-3, 1.0, 1e4])
def test_gram_ridge_matches_direct_solver(alpha):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 9))
    x[:, -1] = x[:, 0] + x[:, 1]
    y = rng.normal(size=(120, 2))
    design = transformed_design(sufficient_stats(x, y))
    gram = fit_alpha(design, alpha)
    direct = direct_ridge_solution(x, y, alpha)
    np.testing.assert_allclose(
        predict(gram, x), predict(direct, x), rtol=2e-7, atol=2e-8
    )


def test_direct_min_norm_uses_the_declared_gram_rank_threshold():
    rng = np.random.default_rng(41)
    n_rows, dimension = 2048, 12
    basis = rng.normal(size=(n_rows, dimension))
    basis -= basis.mean(axis=0, keepdims=True)
    basis, _ = np.linalg.qr(basis)
    singular_values = np.ones(dimension)
    singular_values[-1] = 1e-8
    x = basis * singular_values
    y = rng.normal(size=(n_rows, 3))
    design = transformed_design(sufficient_stats(x, y))
    gram = fit_alpha(design, 0.0)
    direct = direct_ridge_solution(x, y, 0.0)
    assert gram.numerical_rank == dimension - 1
    np.testing.assert_allclose(
        predict(gram, x), predict(direct, x), rtol=2e-7, atol=2e-8
    )


def test_dimensionless_lambda_uses_trace_over_dimension():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80, 6)) * np.arange(1, 7)
    y = rng.normal(size=(80, 2))
    design = transformed_design(sufficient_stats(x, y))
    model = fit_alpha(design, 0.25)
    expected = 0.25 * np.trace(design.gram) / design.gram.shape[0]
    assert model.lambda_absolute == pytest.approx(expected)


def test_min_norm_has_no_hidden_regularization_and_uses_rank_rule():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(30, 50))
    y = rng.normal(size=(30, 2))
    design = transformed_design(sufficient_stats(x, y))
    model = fit_alpha(design, 0.0)
    assert model.lambda_absolute == 0.0
    assert model.numerical_rank <= 29  # centering removes one row degree
    assert model.numerical_tolerance > 0.0


def test_progressive_whitening_whitens_only_requested_eigendirections():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3000, 6)) @ np.diag([5, 3, 2, 1, 0.5, 0.2])
    fit = fit_unlabelled_covariance(x)
    transform = whitening_transform(fit, 3)
    assert transform.valid
    transformed_covariance = transform.matrix.T @ fit.covariance @ transform.matrix
    in_original_basis = (
        fit.eigensystem.eigenvectors.T
        @ transformed_covariance
        @ fit.eigensystem.eigenvectors
    )
    np.testing.assert_allclose(np.diag(in_original_basis)[:3], 1.0, atol=1e-9)
    np.testing.assert_allclose(
        np.diag(in_original_basis)[3:],
        fit.eigensystem.eigenvalues[3:],
        rtol=1e-10,
        atol=1e-10,
    )


def test_invalid_whitening_depth_is_marked_not_clipped():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 5))
    x[:, -1] = x[:, 0]
    fit = fit_unlabelled_covariance(x)
    transform = whitening_transform(fit, 5)
    assert not transform.valid
    assert transform.effective_k is None
    assert transform.failure_reason == "requested_k_exceeds_numerical_rank"


def test_transformed_stats_and_raw_prediction_agree():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(150, 5))
    y = rng.normal(size=(150, 2))
    fit = fit_unlabelled_covariance(x)
    transform = whitening_transform(fit, 2)
    stats = sufficient_stats(x, y)
    model = fit_alpha(transformed_design(stats, transform.matrix), 0.1)
    x_transformed = (x - fit.mean) @ transform.matrix
    transformed_mean = (stats.x_mean - fit.mean) @ transform.matrix
    explicit = (
        (x_transformed - transformed_mean) @ model.weights_transformed
        + stats.y_mean
    )
    np.testing.assert_allclose(predict(model, x), explicit, atol=1e-10)


def test_stats_evaluation_agrees_with_direct_evaluation():
    rng = np.random.default_rng(10)
    x_train = rng.normal(size=(100, 7))
    y_train = rng.normal(size=(100, 3))
    x_test = rng.normal(size=(50, 7))
    y_test = rng.normal(size=(50, 3))
    model = fit_alpha(transformed_design(sufficient_stats(x_train, y_train)), 0.5)
    direct = evaluate(model, x_test, y_test)
    from_stats = evaluate_stats(model, sufficient_stats(x_test, y_test))
    np.testing.assert_allclose(direct.values, from_stats.values, atol=1e-12)


def test_constant_target_is_explicitly_invalid():
    y = np.ones((20, 2))
    prediction = np.column_stack([np.ones(20), np.arange(20)])
    scores = r2_per_target(y, prediction)
    assert not scores.valid.any()
    assert scores.reasons == ("constant_target", "constant_target")
    assert np.isnan(scores.values).all()


def test_validation_only_alpha_tuning_is_deterministic():
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(100, 6))
    beta = rng.normal(size=(6, 2))
    y_train = x_train @ beta + rng.normal(scale=0.4, size=(100, 2))
    x_val = rng.normal(size=(50, 6))
    y_val = x_val @ beta + rng.normal(scale=0.4, size=(50, 2))
    design = transformed_design(sufficient_stats(x_train, y_train))
    validation_stats = sufficient_stats(x_val, y_val)
    first = tune_alpha(
        design, None, None, [0, 1], validation_stats=validation_stats
    )
    second = tune_alpha(
        design, None, None, [0, 1], validation_stats=validation_stats
    )
    assert first.alpha in ALPHA_GRID
    assert first.alpha == second.alpha
    assert first.index == second.index
    assert first.validation_score == second.validation_score
    np.testing.assert_array_equal(first.scores_by_alpha, second.scores_by_alpha)


def test_shuffled_target_control_is_near_zero_or_negative():
    rng = np.random.default_rng(12)
    x_train = rng.normal(size=(300, 8))
    beta = rng.normal(size=(8, 1))
    y_train = x_train @ beta + rng.normal(scale=0.2, size=(300, 1))
    x_test = rng.normal(size=(200, 8))
    y_test = x_test @ beta + rng.normal(scale=0.2, size=(200, 1))
    rng.shuffle(y_test)
    model = fit_alpha(transformed_design(sufficient_stats(x_train, y_train)), 1e-3)
    assert evaluate(model, x_test, y_test).values[0] < 0.05


def _result_row(**updates):
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "experiment_version": "2.0",
            "commit_hash": "a" * 40,
            "branch": "supervised",
            "encoder_seed": 0,
            "readout": "last_concat512",
            "target_block": "directional",
            "target_name": "target",
            "target_independent": True,
            "budget_kind": "integer_days",
            "budget_days_per_stock": 1.0,
            "budget_stock_day_equivalents": 2.0,
            "n_stock_days": 2,
            "n_rows": 20,
            "n_rows_over_dim": 2.0,
            "subsample_seed": 0,
            "block_anchor_quantile": np.nan,
            "feature_view": "full_rank_raw",
            "feature_dim": 10,
            "reader_family": "ridge_raw_tuned_alpha",
            "alpha": float(ALPHA_GRID[10]),
            "lambda_absolute": float(ALPHA_GRID[10]),
            "alpha_selected": True,
            "test_r2": 0.1,
            "fit_status": "ok",
            "failure_reason": "",
            "runtime_seconds": 0.1,
            "ceiling_eligible": False,
        }
    )
    row.update(updates)
    return row


def test_ceiling_normalization_uses_same_tuned_protocol_and_does_not_clip():
    low = _result_row(test_r2=-0.05, alpha=float(ALPHA_GRID[20]))
    full = _result_row(
        budget_kind="full_train",
        budget_days_per_stock=np.nan,
        budget_stock_day_equivalents=20.0,
        n_stock_days=20,
        n_rows=200,
        n_rows_over_dim=20.0,
        subsample_seed=-1,
        test_r2=0.2,
        alpha=float(ALPHA_GRID[15]),
    )
    result = attach_operational_ceilings(pd.DataFrame([low, full]))
    low_result = result.iloc[0]
    assert low_result["full_budget_test_r2"] == pytest.approx(0.2)
    assert low_result["normalized_recovery"] == pytest.approx(-0.25)
    assert bool(low_result["ceiling_eligible"])


def test_ineligible_ceiling_has_no_normalized_ratio():
    low = _result_row(test_r2=0.0)
    full = _result_row(
        budget_kind="full_train",
        budget_days_per_stock=np.nan,
        budget_stock_day_equivalents=20.0,
        n_stock_days=20,
        n_rows=200,
        n_rows_over_dim=20.0,
        subsample_seed=-1,
        test_r2=0.009,
    )
    result = attach_operational_ceilings(pd.DataFrame([low, full]))
    assert not bool(result.iloc[0]["ceiling_eligible"])
    assert np.isnan(result.iloc[0]["normalized_recovery"])


def test_duplicate_experimental_keys_are_rejected():
    frame = pd.DataFrame([_result_row(), _result_row()])
    with pytest.raises(ExperimentIntegrityError, match="duplicate"):
        validate_result_schema(frame, finalized=False)


def test_hierarchical_interval_exposes_two_variance_components():
    points = pd.DataFrame(
        {
            "encoder_seed": np.repeat([0, 1, 2], 4),
            "subsample_seed": list(range(4)) * 3,
            "value": [0.1, 0.2, 0.0, 0.1, 0.3, 0.4, 0.2, 0.3, 0.5, 0.6, 0.4, 0.5],
        }
    )
    first = hierarchical_interval(points, "value", n_bootstrap=500, seed=0)
    second = hierarchical_interval(points, "value", n_bootstrap=500, seed=0)
    assert first == second
    assert first.sd_subsample_within_encoder > 0
    assert first.sd_encoder_between_means > 0
    assert first.n_encoders == 3


def test_parallel_uncertainty_is_bitwise_identical_to_serial():
    rows = []
    for branch_index, branch in enumerate(("supervised", "jepa_horizon")):
        for budget in (0.5, 1.0):
            for encoder_seed in range(3):
                for subsample_seed in range(3):
                    rows.append(
                        {
                            "branch": branch,
                            "encoder_seed": encoder_seed,
                            "readout": "last_concat512",
                            "target_block": "directional",
                            "budget_kind": "fractional",
                            "budget_days_per_stock": budget,
                            "budget_stock_day_equivalents": budget * 7,
                            "subsample_seed": subsample_seed,
                            "feature_view": "full_rank_raw",
                            "whiten_k_requested": np.nan,
                            "whiten_k_effective": np.nan,
                            "reader_family": "ridge_raw_tuned_alpha",
                            "alpha": 1e-3,
                            "recovery_mean": (
                                branch_index
                                + encoder_seed / 10
                                + subsample_seed / 100
                                + budget / 1000
                            ),
                        }
                    )
    points = pd.DataFrame(rows)
    serial = uncertainty_summary(points, n_bootstrap=200, n_workers=1)
    parallel = uncertainty_summary(points, n_bootstrap=200, n_workers=2)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)


def test_gap_pairing_matches_common_alpha_but_not_tuned_alpha():
    base = {
        "encoder_seed": 0,
        "readout": "last_concat512",
        "target_block": "directional",
        "budget_kind": "integer_days",
        "budget_days_per_stock": 1.0,
        "budget_stock_day_equivalents": 2.0,
        "subsample_seed": 0,
        "feature_view": "full_rank_raw",
        "whiten_k_requested": np.nan,
        "whiten_k_effective": np.nan,
    }
    common = pd.DataFrame(
        [
            {
                **base,
                "branch": branch,
                "reader_family": "ridge_raw_common_alpha",
                "alpha": alpha,
                "recovery_mean": value,
            }
            for branch, values in (
                ("supervised", (0.4, 0.5)),
                ("jepa_horizon", (0.2, 0.3)),
            )
            for alpha, value in zip((ALPHA_GRID[4], ALPHA_GRID[8]), values)
        ]
    )
    assert len(paired_gap_points(common)) == 2
    tuned = pd.DataFrame(
        [
            {
                **base,
                "branch": "supervised",
                "reader_family": "ridge_raw_tuned_alpha",
                "alpha": ALPHA_GRID[4],
                "recovery_mean": 0.5,
            },
            {
                **base,
                "branch": "jepa_horizon",
                "reader_family": "ridge_raw_tuned_alpha",
                "alpha": ALPHA_GRID[8],
                "recovery_mean": 0.3,
            },
        ]
    )
    assert len(paired_gap_points(tuned)) == 1


def _write_npy_record(root: Path, relative: str, array: np.ndarray, row_hash: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "row_key_sha256": row_hash,
    }


def _write_rows(root: Path, split: str, rows: pd.DataFrame):
    relative = f"rows/{split}.parquet"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    counts_by_stock = []
    for stock_id, group in rows.groupby("stock_id", sort=True, observed=True):
        counts_by_stock.append(
            {
                "stock_id": int(stock_id),
                "stock_symbol": str(group["stock_symbol"].iloc[0]),
                "n_stock_days": int(group["trading_date"].nunique()),
                "n_endpoints": len(group),
                "first_trading_date": str(group["trading_date"].min()),
                "last_trading_date": str(group["trading_date"].max()),
            }
        )
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "n_rows": len(rows),
        "n_stock_days": len(
            rows[["stock_id", "trading_date"]].drop_duplicates()
        ),
        "first_trading_date": str(rows["trading_date"].min()),
        "last_trading_date": str(rows["trading_date"].max()),
        "counts_by_stock": counts_by_stock,
        "row_key_sha256": sha256_array(
            rows["row_key"].astype(str).to_numpy(dtype="U")
        ),
        "complete_stock_days": True,
    }


def _convert_record_to_shards(
    root: Path, record: dict, rows: pd.DataFrame
) -> dict:
    source = np.load(root / record["path"], allow_pickle=False)
    cuts = (0, len(source) // 2, len(source))
    shards = []
    for index, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
        relative = f"sharded/{Path(record['path']).stem}-{index}.npy"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, source[start:stop])
        shards.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "shape": list(source[start:stop].shape),
                "dtype": source.dtype.name,
                "row_start": start,
                "row_stop": stop,
                "row_key_sha256": sha256_array(
                    rows["row_key"].iloc[start:stop].astype(str).to_numpy(
                        dtype="U"
                    )
                ),
            }
        )
    result = {
        "storage": SHARDED_STORAGE,
        "shape": list(source.shape),
        "dtype": source.dtype.name,
        "row_key_sha256": record["row_key_sha256"],
        "shards": shards,
    }
    result["shard_manifest_sha256"] = canonical_json_sha256(
        sharded_record_fingerprint_payload(result)
    )
    return result


def _make_bundle(root: Path) -> Path:
    rng = np.random.default_rng(13)
    split_rows = {}
    offset = 0
    for split, day in (("train", 10), ("validation", 20), ("test", 30)):
        values = []
        for stock in (0, 1):
            for order in range(4):
                endpoint = offset
                values.append(
                    {
                        "row_key": (
                            f"{stock}|2026-02-{day - 9:02d}|{order}"
                        ),
                        "stock_id": stock,
                        "stock_symbol": f"S{stock}",
                        "stock_day_id": day,
                        "trading_date": f"2026-02-{day - 9:02d}",
                        "endpoint_index": endpoint,
                        "endpoint_order": order,
                        "timestamp_ns": (endpoint + 1) * 1_000_000_000,
                    }
                )
                offset += 1
        split_rows[split] = pd.DataFrame(values)
    split_records = {
        split: _write_rows(root, split, rows)
        for split, rows in split_rows.items()
    }
    definitions = [
        {"name": "d_a@1", "block": "directional", "independent": True, "redundant_with": []},
        {"name": "d_b@1", "block": "directional", "independent": True, "redundant_with": []},
        {"name": "realized_vol@5", "block": "volatility", "independent": True, "redundant_with": []},
        {
            "name": "time_to_next_mid_move",
            "block": "timing",
            "independent": True,
            "redundant_with": [],
            "semantics": "log1p_observed_or_capped_all_rows:max_look=600",
        },
    ]
    target_arrays = {}
    for split, rows in split_rows.items():
        target_arrays[split] = _write_npy_record(
            root,
            f"targets/{split}.npy",
            rng.normal(size=(len(rows), len(definitions))).astype(np.float32),
            split_records[split]["row_key_sha256"],
        )
    feature_sets = []
    seeds = {"supervised": [0, 1, 2], "jepa_horizon": [0, 1, 2], "jepa_masked": [0, 1, 2]}
    checkpoints = {}
    for branch, branch_seeds in seeds.items():
        for seed in branch_seeds:
            tag = f"{branch}_seed{seed}_ep020"
            checkpoint_path = root / "checkpoints" / f"{tag}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(tag.encode("utf-8"))
            checkpoints[tag] = {
                "arm": branch,
                "seed": seed,
                "epoch": 20,
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            }
            for readout in READOUTS:
                arrays = {}
                for split, rows in split_rows.items():
                    arrays[split] = _write_npy_record(
                        root,
                        f"features/{branch}_s{seed}_{readout}_{split}.npy",
                        rng.normal(size=(len(rows), 512)).astype(np.float32),
                        split_records[split]["row_key_sha256"],
                    )
                feature_sets.append(
                    {
                        "branch": branch,
                        "encoder_seed": seed,
                        "readout": readout,
                        "dimension": 512,
                        "dtype": "float32",
                        "checkpoint_tag": tag,
                        "checkpoint_sha256": checkpoints[tag]["sha256"],
                        "arrays": arrays,
                    }
                )
    gate_schemas = {
        "storage_estimate": "thesis.experiment01.storage_estimate",
        "pre_extraction_report": "thesis.experiment01.pre_extraction_gate",
        "target_equivalence_report": "thesis.experiment01.target_equivalence",
    }
    for name, schema_name in gate_schemas.items():
        path = root / f"{name}.json"
        path.write_text(
            json.dumps({"schema_name": schema_name, "passed": True}),
            encoding="utf-8",
        )
    pre_extraction = {
        "storage_estimate": {
            "path": "storage_estimate.json",
            "sha256": sha256_file(root / "storage_estimate.json"),
            "size_bytes": (root / "storage_estimate.json").stat().st_size,
            "passed": True,
        },
        "benchmark_and_feature_equivalence": {
            "path": "pre_extraction_report.json",
            "sha256": sha256_file(root / "pre_extraction_report.json"),
            "size_bytes": (root / "pre_extraction_report.json").stat().st_size,
            "passed": True,
        },
    }
    manifest = {
        "schema_name": INPUT_SCHEMA,
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "provenance": {
            "corrected_post_p0": True,
            "source_commit": "a" * 40,
            "dataset_path": "synthetic.npz",
            "dataset_sha256": "b" * 64,
            "split_protocol_fingerprint": "split-fixture",
            "target_manifest_fingerprint": "targets-fixture",
        },
        "historical_heldout_reuse_disclosed": True,
        "validation_is_fixed_and_complete": True,
        "test_is_fixed_and_complete": True,
        "training_features_are_full_unlabelled_split": True,
        "context_window_within_stock_day_verified": True,
        "target_horizon_within_stock_day_verified": True,
        "row_feature_target_alignment_verified": True,
        "protocol": {
            "K": 20,
            "max_horizon": 20,
            "grouping": "stock_id+trading_date",
            "test_hyperparameter_selection": "forbidden",
        },
        "pre_extraction": pre_extraction,
        "canonical_encoder_seeds": seeds,
        "canonical_checkpoints": checkpoints,
        "readout_definitions": READOUT_DEFINITIONS,
        "splits": split_records,
        "targets": {
            "definitions": definitions,
            "arrays": target_arrays,
            "equivalence_report": {
                "path": "target_equivalence_report.json",
                "sha256": sha256_file(root / "target_equivalence_report.json"),
                "size_bytes": (
                    root / "target_equivalence_report.json"
                ).stat().st_size,
                "passed": True,
            },
        },
        "feature_sets": feature_sets,
    }
    manifest["feature_inventory_sha256"] = canonical_json_sha256(
        {"feature_sets": feature_sets}
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_three_way_bundle_preflight_passes(tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    bundle = load_input_bundle(root)
    assert set(bundle.rows) == {"train", "validation", "test"}
    assert len(bundle.feature_sets) == 18
    assert bundle.load_targets("test").shape == (8, 4)


def test_three_way_bundle_with_sharded_targets_and_features_passes(tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_rows = pd.read_parquet(root / manifest["splits"]["train"]["path"])
    test_rows = pd.read_parquet(root / manifest["splits"]["test"]["path"])
    manifest["targets"]["arrays"]["train"] = _convert_record_to_shards(
        root, manifest["targets"]["arrays"]["train"], train_rows
    )
    manifest["feature_sets"][0]["arrays"]["test"] = _convert_record_to_shards(
        root, manifest["feature_sets"][0]["arrays"]["test"], test_rows
    )
    manifest["feature_inventory_sha256"] = canonical_json_sha256(
        {"feature_sets": manifest["feature_sets"]}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = load_input_bundle(root)
    assert isinstance(bundle.target_paths["train"], ShardedArray)
    assert any(
        isinstance(feature.paths["test"], ShardedArray)
        for feature in bundle.feature_sets
    )
    np.testing.assert_array_equal(
        bundle.load_targets("train")[:],
        np.load(root / "targets/train.npy"),
    )


def test_sharded_row_identity_mismatch_fails_loudly(tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = pd.read_parquet(root / manifest["splits"]["train"]["path"])
    record = _convert_record_to_shards(
        root, manifest["targets"]["arrays"]["train"], rows
    )
    record["shards"][0]["row_key_sha256"] = "0" * 64
    record["shard_manifest_sha256"] = canonical_json_sha256(
        sharded_record_fingerprint_payload(record)
    )
    manifest["targets"]["arrays"]["train"] = record
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExperimentIntegrityError, match="row identity mismatch"):
        load_input_bundle(root)


def test_bundle_without_test_split_fails_loudly(tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    manifest = json.loads((root / "manifest.json").read_text())
    del manifest["splits"]["test"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ExperimentIntegrityError, match="splits must be exactly"):
        load_input_bundle(root, verify_hashes=False)


def test_feature_row_identity_mismatch_fails_loudly(tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["feature_sets"][0]["arrays"]["test"]["row_key_sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ExperimentIntegrityError, match="row identity mismatch"):
        load_input_bundle(root, verify_hashes=False)


def test_raw_endpoint_overlap_across_splits_fails_loudly(tmp_path):
    root = _make_bundle(tmp_path / "bundle")
    manifest = json.loads((root / "manifest.json").read_text())
    train = pd.read_parquet(root / manifest["splits"]["train"]["path"])
    validation_path = root / manifest["splits"]["validation"]["path"]
    validation = pd.read_parquet(validation_path)
    validation.loc[0, "endpoint_index"] = int(train.loc[0, "endpoint_index"])
    validation.to_parquet(validation_path, index=False)
    manifest["splits"]["validation"]["size_bytes"] = validation_path.stat().st_size
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ExperimentIntegrityError, match="overlap across splits"):
        load_input_bundle(root, verify_hashes=False)


def test_legacy_two_way_directory_is_not_silently_accepted():
    with pytest.raises(ExperimentIntegrityError, match="manifest.json is missing"):
        load_input_bundle("validation/readouts_v2_20260728", verify_hashes=False)


def test_phase1_streaming_smoke_writes_finalized_artifacts(tmp_path):
    rng = np.random.default_rng(21)
    root = tmp_path / "manual_bundle"
    root.mkdir()
    (root / "manifest.json").write_text("{}")
    train_rows = _training_rows(days=(2, 2), rows_per_day=12)
    validation_rows = _training_rows(days=(1, 1), rows_per_day=8)
    test_rows = _training_rows(days=(1, 1), rows_per_day=8)
    # Stable keys and raw identities must be disjoint even in this manually
    # constructed bundle, which bypasses the full schema preflight on purpose.
    for prefix, frame, offset in (
        ("validation", validation_rows, 10_000),
        ("test", test_rows, 20_000),
    ):
        frame["row_key"] = prefix + "|" + frame["row_key"]
        frame["endpoint_index"] += offset
        frame["timestamp_ns"] += offset * 1_000_000_000
        frame["stock_day_id"] += offset
    beta = rng.normal(size=(4, 2))
    paths = {}
    target_paths = {}
    for split, frame in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        x = rng.normal(size=(len(frame), 4)).astype(np.float32)
        y = (x @ beta + rng.normal(scale=0.2, size=(len(frame), 2))).astype(
            np.float32
        )
        x_path = root / f"x_{split}.npy"
        y_path = root / f"y_{split}.npy"
        np.save(x_path, x)
        np.save(y_path, y)
        paths[split] = x_path
        target_paths[split] = y_path
    definitions = (
        TargetDefinition("d_a@1", "directional", True, (), None),
        TargetDefinition("d_b@1", "directional", True, (), None),
    )
    feature = FeatureSet(
        "supervised",
        0,
        "last_concat512",
        4,
        np.dtype("float32"),
        paths,
    )
    bundle = InputBundle(
        root=root,
        manifest={
            "splits": {
                "train": {
                    "row_key_sha256": sha256_array(
                        train_rows["row_key"].to_numpy(dtype="U")
                    )
                }
            },
            "canonical_encoder_seeds": {
                "supervised": [0],
                "jepa_horizon": [0],
                "jepa_masked": [0],
            },
        },
        rows={
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
        },
        target_paths=target_paths,
        target_definitions=definitions,
        feature_sets=(feature,),
    )
    out = tmp_path / "run"
    metadata = run_phase1(
        bundle,
        out,
        Phase1Config(
            branches=("supervised",),
            readouts=("last_concat512",),
            target_blocks=("directional",),
            run_common_alpha=False,
            run_tuned_alpha=True,
            run_min_norm=True,
            run_whitening=False,
            run_anchor_sensitivity=True,
            chunk_rows=16,
            direct_crosscheck_rows=16,
        ),
    )
    results = pd.read_parquet(out / "results.parquet")
    assert len(results) > 0
    assert results["fit_status"].eq("ok").all()
    assert results["full_budget_test_r2"].notna().all()
    assert (out / "metadata.json").is_file()
    assert (out / "failures.parquet").is_file()
    assert (out / "time_of_day_sensitivity.parquet").is_file()
    assert metadata["artifacts"]["results"]["n_rows"] == len(results)
    assert metadata["artifacts"]["time_of_day_sensitivity"]["n_rows"] > 0
    summary = summarize_phase1(
        out / "results.parquet", out / "summary", n_bootstrap=20
    )
    assert (
        summary["directional_last_concat512_outcome"]["outcome"] == "D"
    )
    report = generate_phase1_report(
        out / "results.parquet", out / "summary", out / "report"
    )
    assert report["outcome"]["outcome"] == "D"
    report_path = out / "report" / "REPORT_EXPERIMENT_01.md"
    assert report_path.is_file()
    initial_text = report_path.read_text()
    assert "Phase II status: `not_present`" in initial_text
    assert "Phase II (PCA/random subspaces) and Phase III (MLP) were not run" not in initial_text
    assert initial_text.index("## Primary scientific result") < initial_text.index(
        "## Secondary frozen preregistered classification"
    )
    narrative_text = (
        out / "report" / "SUMMARY_NARRATIVE_EXPERIMENT_01.md"
    ).read_text()
    assert narrative_text.index("## Primary scientific result") < narrative_text.index(
        "## Secondary frozen technical classification"
    )
    claim_table = pd.read_parquet(out / "report" / "17_claim_table.parquet")
    assert not claim_table["source_artifact"].str.startswith("/").any()
    report_manifest = json.loads(
        (out / "report" / "report_manifest.json").read_text()
    )
    assert all(
        not record["path"].startswith("/")
        for record in report_manifest["protected_inputs"].values()
    )

    # Report prose is artifact-derived: changing the machine summary changes
    # the corresponding numbers, budget names, thresholds and interval claim.
    summary_path = out / "summary" / "summary.json"
    mutated = json.loads(summary_path.read_text())
    mutated["target_block_gap_signatures"]["directional"][
        "low_budget_mean_normalized_gap"
    ] = 0.123456
    outcome = mutated["directional_last_concat512_outcome"]
    outcome["absolute_ceiling_gap"] = {
        "mean": 0.02,
        "lower": -0.03,
        "upper": 0.07,
        "robust": False,
    }
    outcome["decisive_budgets"] = [1.0, 2.0]
    outcome["native_low_budget_mean_gap"] = 0.234567
    outcome["k_50gap"] = 7
    outcome["k_nonrobust"] = 9
    outcome["whitening_candidates"] = [
        {
            "k": 7,
            "mean_gap": 0.09,
            "reduction_fraction": 0.6,
            "all_robust": True,
            "all_nonrobust": False,
        },
        {
            "k": 9,
            "mean_gap": 0.01,
            "reduction_fraction": 0.9,
            "all_robust": False,
            "all_nonrobust": True,
        },
    ]
    summary_path.write_text(json.dumps(mutated))
    generate_phase1_report(
        out / "results.parquet", out / "summary", out / "report"
    )
    mutated_text = report_path.read_text()
    assert "`0.123456`" in mutated_text
    assert "`1`, `2`" in mutated_text
    assert "`k_50gap=7`" in mutated_text
    assert "`k_nonrobust=9`" in mutated_text
    assert "The interval includes zero." in mutated_text
    assert "k_50gap=128" not in mutated_text

    mutated["directional_last_concat512_outcome"]["absolute_ceiling_gap"] = {
        "mean": 0.04,
        "lower": 0.01,
        "upper": 0.08,
        "robust": True,
    }
    summary_path.write_text(json.dumps(mutated))
    generate_phase1_report(
        out / "results.parquet", out / "summary", out / "report"
    )
    assert "The interval excludes zero." in report_path.read_text()

    summary_path.unlink()
    with pytest.raises(ValueError, match="cannot read Phase-I technical summary"):
        generate_phase1_report(
            out / "results.parquet", out / "summary", out / "report"
        )

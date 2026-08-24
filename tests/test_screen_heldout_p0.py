from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from experiment01.historical import analysis_artifacts as artifacts
from experiment01.historical import screen_heldout_gate1 as gate1


def _write_gate_dataset(
    path: Path, stock_ids: np.ndarray, day_ids: np.ndarray
) -> None:
    n_rows = len(stock_ids)
    np.savez(
        path,
        book=np.zeros((n_rows, 2, 2, 2), dtype=np.float32),
        mid_z=np.arange(n_rows, dtype=np.float32),
        stock_ids=np.asarray(stock_ids, dtype=np.int64),
        day_ids=np.asarray(day_ids, dtype=np.int64),
        min_spread_z_per_stock=np.ones(
            int(np.max(stock_ids)) + 1, dtype=np.float32
        ),
    )


def _dataset_sha256(path: Path) -> str:
    return artifacts.sha256_file(path)


def _field(split, name: str):
    if isinstance(split, dict):
        return split[name]
    return getattr(split, name)


def _save_split(path: Path, split, dataset_sha256: str) -> None:
    artifacts.save_split(
        path,
        split,
        dataset_sha256=dataset_sha256,
        split_config={
            "K": 3,
            "max_horizon": 2,
            "val_frac": 0.34,
            "split_seed": 7,
            "n_train": 8,
            "n_val": 6,
            "subsample_seed": 19,
            "grouping": "stock_day",
        },
    )


def _gate_argv(dataset: Path, split_path: Path) -> list[str]:
    return [
        "screen_heldout_gate1.py",
        "--dataset",
        str(dataset),
        "--subsample",
        str(split_path),
        "--K",
        "3",
        "--max_h",
        "2",
        "--val_frac",
        "0.34",
        "--split_seed",
        "7",
        "--n_train",
        "8",
        "--n_val",
        "6",
        "--subsample_seed",
        "19",
    ]


def test_gate1_reuse_uses_raw_train_and_validation_endpoints(
    monkeypatch, tmp_path
):
    block_size = 12
    stock_ids: list[int] = []
    day_ids: list[int] = []
    for stock_id in range(2):
        for day_id in range(3):
            stock_ids.extend([stock_id] * block_size)
            day_ids.extend([day_id] * block_size)
    stock_ids_array = np.asarray(stock_ids, dtype=np.int64)
    day_ids_array = np.asarray(day_ids, dtype=np.int64)

    dataset = tmp_path / "gate_dataset.npz"
    _write_gate_dataset(dataset, stock_ids_array, day_ids_array)
    split = artifacts.build_endpoint_split(
        stock_ids_array,
        day_ids_array,
        np.ones(len(stock_ids_array), dtype=bool),
        K=3,
        max_horizon=2,
        val_frac=0.34,
        split_seed=7,
        n_train=8,
        n_val=6,
        subsample_seed=19,
    )
    split_path = tmp_path / "split.npz"
    _save_split(split_path, split, _dataset_sha256(dataset))
    train_t = np.asarray(_field(split, "train_t"))
    val_t = np.asarray(_field(split, "val_t"))

    future_calls: list[np.ndarray] = []
    vol_calls: list[np.ndarray] = []
    timing_calls: list[np.ndarray] = []

    def fake_derive_raw_features(book, mid_z, stock_ids, n_stocks):
        return np.zeros((len(stock_ids), 1), dtype=np.float32), ["synthetic"]

    def fake_future_targets(raw_feat, endpoints, features, horizons):
        future_calls.append(np.asarray(endpoints).copy())
        return np.zeros(
            (len(endpoints), len(features) * len(horizons)), dtype=np.float32
        )

    def fake_vol_targets(
        mid_z, endpoints, horizons, min_spread_per_stock, stock_ids
    ):
        vol_calls.append(np.asarray(endpoints).copy())
        return np.zeros((len(endpoints), len(horizons)), dtype=np.float32)

    def fake_timing(mid_z, stock_ids, day_ids, endpoints, max_look=600):
        timing_calls.append(np.asarray(endpoints).copy())
        durations = np.arange(1, len(endpoints) + 1, dtype=np.int32)
        return durations, np.zeros(len(endpoints), dtype=bool)

    monkeypatch.setattr(
        gate1, "derive_raw_features_array", fake_derive_raw_features
    )
    monkeypatch.setattr(
        gate1, "compute_future_feature_targets", fake_future_targets
    )
    monkeypatch.setattr(gate1, "compute_vol_targets", fake_vol_targets)
    monkeypatch.setattr(gate1, "time_to_next_mid_move", fake_timing)
    monkeypatch.setattr(gate1, "ols_r2", lambda *args: 0.0)
    monkeypatch.setattr(sys, "argv", _gate_argv(dataset, split_path))

    gate1.main()

    expected_future_calls = []
    for _ in range(1 + len(gate1.CAND_FEATS)):
        expected_future_calls.extend([train_t, val_t])
    assert len(future_calls) == len(expected_future_calls)
    for observed, expected in zip(future_calls, expected_future_calls):
        np.testing.assert_array_equal(observed, expected)

    assert len(vol_calls) == 2
    np.testing.assert_array_equal(vol_calls[0], train_t)
    np.testing.assert_array_equal(vol_calls[1], val_t)
    assert len(timing_calls) == 2
    np.testing.assert_array_equal(timing_calls[0], train_t)
    np.testing.assert_array_equal(timing_calls[1], val_t)


def test_gate1_reuse_rejects_legacy_position_only_subsample(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "gate_dataset.npz"
    _write_gate_dataset(
        dataset,
        np.zeros(20, dtype=np.int64),
        np.zeros(20, dtype=np.int64),
    )
    legacy = tmp_path / "legacy_subsample.npz"
    np.savez(
        legacy,
        train_pos=np.array([0, 2], dtype=np.int64),
        val_pos=np.array([1, 3], dtype=np.int64),
    )

    def must_not_reach_target_construction(*args, **kwargs):
        pytest.fail("Gate 1 accepted a legacy position-only split")

    monkeypatch.setattr(
        gate1,
        "derive_raw_features_array",
        must_not_reach_target_construction,
    )
    monkeypatch.setattr(sys, "argv", _gate_argv(dataset, legacy))

    with pytest.raises(ValueError, match=r"(?i)(schema|legacy|train_t|endpoint)"):
        gate1.main()


def test_time_to_next_mid_move_never_crosses_stock_or_day_boundary():
    mid_z = np.array(
        [0, 0, 10, 10, 10, 20, 20, 20, 21, 21], dtype=np.float32
    )
    stock_ids = np.array(
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64
    )
    day_ids = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64
    )

    durations, censored = gate1.time_to_next_mid_move(
        mid_z,
        stock_ids,
        day_ids,
        np.array([1, 4, 6], dtype=np.int64),
    )

    # Changes immediately across a stock boundary (t=1) or day boundary (t=4)
    # are not observations of the same series and are therefore censored.
    np.testing.assert_array_equal(durations, np.array([1, 1, 2]))
    np.testing.assert_array_equal(censored, np.array([True, True, False]))


def test_time_to_next_mid_move_caps_exhausted_search_at_max_look():
    mid_z = np.zeros(10, dtype=np.float32)
    stock_ids = np.zeros(10, dtype=np.int64)
    day_ids = np.zeros(10, dtype=np.int64)

    durations, censored = gate1.time_to_next_mid_move(
        mid_z,
        stock_ids,
        day_ids,
        np.array([0], dtype=np.int64),
        max_look=3,
    )

    np.testing.assert_array_equal(durations, np.array([3]))
    np.testing.assert_array_equal(censored, np.array([True]))


def test_timing_target_is_identical_for_screening_and_saved_rows(
    monkeypatch, tmp_path
):
    block_size = 12
    stock_ids = np.repeat(np.arange(6, dtype=np.int64), block_size)
    day_ids = np.tile(
        np.repeat(np.arange(3, dtype=np.int64), 2 * block_size),
        1,
    )
    dataset = tmp_path / "gate_dataset.npz"
    _write_gate_dataset(dataset, stock_ids, day_ids)
    split = artifacts.build_endpoint_split(
        stock_ids,
        day_ids,
        np.ones(len(stock_ids), dtype=bool),
        K=3,
        max_horizon=2,
        val_frac=0.34,
        split_seed=7,
        n_train=8,
        n_val=6,
        subsample_seed=19,
    )
    split_path = tmp_path / "split.npz"
    _save_split(split_path, split, _dataset_sha256(dataset))
    train_t = np.asarray(_field(split, "train_t"))
    val_t = np.asarray(_field(split, "val_t"))
    seen = []

    monkeypatch.setattr(
        gate1,
        "derive_raw_features_array",
        lambda *args: (np.zeros((len(stock_ids), 9), dtype=np.float32), []),
    )
    monkeypatch.setattr(
        gate1,
        "compute_future_feature_targets",
        lambda raw, endpoints, features, horizons: np.zeros(
            (len(endpoints), len(features) * len(horizons)), dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        gate1,
        "compute_vol_targets",
        lambda mid, endpoints, horizons, spreads, stocks: np.zeros(
            (len(endpoints), len(horizons)), dtype=np.float32
        ),
    )

    def fake_timing(mid, stocks, days, endpoints, max_look=600):
        values = np.arange(1, len(endpoints) + 1, dtype=np.int32)
        values[-1] = max_look
        censored = np.zeros(len(endpoints), dtype=bool)
        censored[-1] = True
        return values, censored

    def record_ols(xtr, ytr, xva, yva):
        if np.asarray(ytr).ndim == 1:
            seen.append((np.asarray(ytr).copy(), np.asarray(yva).copy()))
        return 0.0

    heldout = tmp_path / "heldout.npz"
    monkeypatch.setattr(gate1, "time_to_next_mid_move", fake_timing)
    monkeypatch.setattr(gate1, "ols_r2", record_ols)
    monkeypatch.setattr(
        sys,
        "argv",
        _gate_argv(dataset, split_path)
        + ["--save_heldout", str(heldout)],
    )

    gate1.main()

    with np.load(heldout, allow_pickle=False) as saved:
        timing_col = list(saved["heldout_names"]).index(
            "time_to_next_mid_move"
        )
        np.testing.assert_allclose(
            saved["y_train_heldout"][:, timing_col], seen[-1][0]
        )
        np.testing.assert_allclose(
            saved["y_val_heldout"][:, timing_col], seen[-1][1]
        )

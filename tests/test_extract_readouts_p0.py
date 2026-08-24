from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from experiment01.historical import extract_readouts_multiseed as stage1


def _write_tiny_dataset(
    path: Path,
    book: np.ndarray,
    *,
    day_ids: np.ndarray | None = None,
) -> None:
    n_rows = book.shape[0]
    if day_ids is None:
        day_ids = np.zeros(n_rows, dtype=np.int64)
    np.savez(
        path,
        book=book,
        mid_z=np.arange(n_rows, dtype=np.float32),
        stock_ids=np.zeros(n_rows, dtype=np.int64),
        day_ids=np.asarray(day_ids, dtype=np.int64),
        min_spread_z_per_stock=np.ones(1, dtype=np.float32),
    )


def _run_without_checkpoints(
    monkeypatch,
    *,
    dataset: Path,
    checkpoint_root: Path,
    output_dir: Path,
) -> None:
    checkpoint_root.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_readouts_multiseed.py",
            "--dataset",
            str(dataset),
            "--ckpt_root",
            str(checkpoint_root),
            "--out_dir",
            str(output_dir),
            "--K",
            "3",
            "--max_h",
            "2",
            "--val_frac",
            "0.25",
            "--split_seed",
            "7",
            "--n_train",
            "100",
            "--n_val",
            "100",
            "--arms",
            "",
            "--num_workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    stage1.main()


def test_stage1_maps_split_positions_to_raw_endpoints_before_building_targets(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "tiny_lob.npz"
    book = np.zeros((20, 2, 2, 2), dtype=np.float32)
    day_ids = np.zeros(20, dtype=np.int64)
    day_ids[[8, 14]] = 1
    _write_tiny_dataset(dataset, book, day_ids=day_ids)

    valid_t = np.array([5, 8, 11, 14], dtype=np.int64)
    train_pos = np.array([0, 2], dtype=np.int64)
    val_pos = np.array([1, 3], dtype=np.int64)
    target_endpoint_calls: list[np.ndarray] = []

    def fake_compute_valid_endpoints(stock_ids, day_ids, K, max_horizon, vol_mask):
        np.testing.assert_array_equal(stock_ids, np.zeros(20, dtype=np.int64))
        expected_days = np.zeros(20, dtype=np.int64)
        expected_days[[8, 14]] = 1
        np.testing.assert_array_equal(day_ids, expected_days)
        assert K == 3
        assert max_horizon == 2
        return valid_t.copy()

    def fake_grouped_split(stock_ids, day_ids, endpoints, val_frac, seed):
        np.testing.assert_array_equal(endpoints, valid_t)
        assert val_frac == 0.25
        assert seed == 7
        return train_pos.copy(), val_pos.copy()

    def fake_build_raw_targets(
        book, mid_z, stock_ids, endpoint_subset, min_spread_per_stock
    ):
        target_endpoint_calls.append(np.asarray(endpoint_subset).copy())
        return (
            np.zeros((len(endpoint_subset), 1), dtype=np.float32),
            ["synthetic_target"],
        )

    monkeypatch.setattr(
        stage1, "compute_valid_endpoints", fake_compute_valid_endpoints
    )
    monkeypatch.setattr(
        stage1, "grouped_split_by_stock_day", fake_grouped_split
    )
    monkeypatch.setattr(stage1, "build_raw_targets", fake_build_raw_targets)

    _run_without_checkpoints(
        monkeypatch,
        dataset=dataset,
        checkpoint_root=tmp_path / "checkpoints",
        output_dir=tmp_path / "output",
    )

    assert len(target_endpoint_calls) == 2
    np.testing.assert_array_equal(target_endpoint_calls[0], valid_t[train_pos])
    np.testing.assert_array_equal(target_endpoint_calls[1], valid_t[val_pos])


def test_stage1_passes_absolute_inclusive_volume_mask_to_endpoint_builder(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "tiny_lob.npz"
    book = np.zeros((20, 2, 2, 2), dtype=np.float32)
    book[2, 0, 0, 1] = 5.0
    book[3, 0, 1, 1] = -5.0
    book[4, 0, 0, 1] = 5.01
    book[5, 1, 1, 1] = -6.0
    day_ids = np.zeros(20, dtype=np.int64)
    day_ids[[8, 14]] = 1
    _write_tiny_dataset(dataset, book, day_ids=day_ids)

    expected_vol_mask = np.ones(20, dtype=bool)
    expected_vol_mask[[4, 5]] = False
    observed_vol_masks: list[np.ndarray] = []

    def fake_compute_valid_endpoints(stock_ids, day_ids, K, max_horizon, vol_mask):
        observed_vol_masks.append(np.asarray(vol_mask).copy())
        return np.array([5, 8, 11, 14], dtype=np.int64)

    def fake_grouped_split(stock_ids, day_ids, valid_t, val_frac, seed):
        return (
            np.array([0, 2], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
        )

    def fake_build_raw_targets(
        book, mid_z, stock_ids, endpoint_subset, min_spread_per_stock
    ):
        return (
            np.zeros((len(endpoint_subset), 1), dtype=np.float32),
            ["synthetic_target"],
        )

    monkeypatch.setattr(
        stage1, "compute_valid_endpoints", fake_compute_valid_endpoints
    )
    monkeypatch.setattr(
        stage1, "grouped_split_by_stock_day", fake_grouped_split
    )
    monkeypatch.setattr(stage1, "build_raw_targets", fake_build_raw_targets)

    _run_without_checkpoints(
        monkeypatch,
        dataset=dataset,
        checkpoint_root=tmp_path / "checkpoints",
        output_dir=tmp_path / "output",
    )

    assert len(observed_vol_masks) == 1
    np.testing.assert_array_equal(observed_vol_masks[0], expected_vol_mask)

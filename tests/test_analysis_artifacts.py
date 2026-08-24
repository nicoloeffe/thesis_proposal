from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from experiment01.reference import analysis_artifacts as artifacts


@dataclass(frozen=True)
class SyntheticGroups:
    stock_ids: np.ndarray
    day_ids: np.ndarray
    vol_mask: np.ndarray
    K: int = 3
    max_horizon: int = 2


@pytest.fixture
def synthetic_groups() -> SyntheticGroups:
    block_size = 12
    stock_ids: list[int] = []
    day_ids: list[int] = []
    for stock_id in range(2):
        for day_id in range(3):
            stock_ids.extend([stock_id] * block_size)
            day_ids.extend([day_id] * block_size)

    vol_mask = np.ones(len(stock_ids), dtype=bool)
    # These invalidate endpoints whose complete past/future span touches them.
    vol_mask[[6, 42]] = False
    return SyntheticGroups(
        stock_ids=np.asarray(stock_ids, dtype=np.int64),
        day_ids=np.asarray(day_ids, dtype=np.int64),
        vol_mask=vol_mask,
    )


def _field(split, name: str):
    if isinstance(split, dict):
        return split[name]
    return getattr(split, name)


def _build_split(groups: SyntheticGroups):
    return artifacts.build_endpoint_split(
        groups.stock_ids,
        groups.day_ids,
        groups.vol_mask,
        K=groups.K,
        max_horizon=groups.max_horizon,
        val_frac=0.34,
        split_seed=7,
        n_train=13,
        n_val=9,
        subsample_seed=19,
    )


def _assert_complete_endpoint_contract(
    endpoints: np.ndarray,
    groups: SyntheticGroups,
) -> None:
    offsets = np.arange(
        -(groups.K - 1), groups.max_horizon + 1, dtype=np.int64
    )
    for endpoint in np.asarray(endpoints, dtype=np.int64):
        indices = endpoint + offsets
        assert indices[0] >= 0
        assert indices[-1] < len(groups.stock_ids)
        assert np.all(groups.stock_ids[indices] == groups.stock_ids[endpoint])
        assert np.all(groups.day_ids[indices] == groups.day_ids[endpoint])
        assert np.all(groups.vol_mask[indices])


def _endpoint_hash(values: np.ndarray) -> str:
    hash_fn = getattr(artifacts, "endpoint_sha256", None)
    if hash_fn is None:
        hash_fn = getattr(artifacts, "sha256_array")
    return hash_fn(np.asarray(values, dtype=np.int64))


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
            "n_train": 13,
            "n_val": 9,
            "subsample_seed": 19,
            "grouping": "stock_day",
        },
    )


def test_build_endpoint_split_maps_positions_to_raw_endpoints(synthetic_groups):
    split = _build_split(synthetic_groups)
    valid_t = _field(split, "valid_t")
    train_pos = _field(split, "train_pos")
    val_pos = _field(split, "val_pos")
    train_t = _field(split, "train_t")
    val_t = _field(split, "val_t")

    np.testing.assert_array_equal(train_t, valid_t[train_pos])
    np.testing.assert_array_equal(val_t, valid_t[val_pos])
    assert np.any(train_t != train_pos)
    assert np.any(val_t != val_pos)


def test_build_endpoint_split_has_disjoint_stock_days(synthetic_groups):
    split = _build_split(synthetic_groups)
    train_t = _field(split, "train_t")
    val_t = _field(split, "val_t")

    train_groups = set(
        zip(
            synthetic_groups.stock_ids[train_t].tolist(),
            synthetic_groups.day_ids[train_t].tolist(),
        )
    )
    val_groups = set(
        zip(
            synthetic_groups.stock_ids[val_t].tolist(),
            synthetic_groups.day_ids[val_t].tolist(),
        )
    )
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)


def test_build_endpoint_split_respects_window_horizon_and_vol_mask(
    synthetic_groups,
):
    split = _build_split(synthetic_groups)
    _assert_complete_endpoint_contract(_field(split, "train_t"), synthetic_groups)
    _assert_complete_endpoint_contract(_field(split, "val_t"), synthetic_groups)


def test_validation_endpoints_are_all_members_of_valid_t(synthetic_groups):
    split = _build_split(synthetic_groups)
    valid_t = _field(split, "valid_t")
    val_t = _field(split, "val_t")

    positions = np.searchsorted(valid_t, val_t)
    assert np.all(positions < len(valid_t))
    np.testing.assert_array_equal(valid_t[positions], val_t)


def test_endpoint_hash_is_deterministic_and_order_sensitive():
    endpoints = np.array([5, 8, 11, 14], dtype=np.int64)
    assert _endpoint_hash(endpoints) == _endpoint_hash(endpoints.copy())
    assert _endpoint_hash(endpoints) != _endpoint_hash(endpoints[::-1])


def test_split_roundtrip_preserves_raw_endpoints_and_hashes(
    synthetic_groups, tmp_path
):
    split = _build_split(synthetic_groups)
    dataset_sha256 = "a" * 64
    path = tmp_path / "split.npz"
    _save_split(path, split, dataset_sha256)

    loaded = artifacts.load_split(
        path, expected_dataset_sha256=dataset_sha256
    )
    for name in ("valid_t", "train_pos", "val_pos", "train_t", "val_t"):
        np.testing.assert_array_equal(_field(loaded, name), _field(split, name))

    with np.load(path, allow_pickle=False) as archive:
        assert int(archive["schema_version"]) >= 2
        assert str(archive["dataset_sha256"]) == dataset_sha256
        assert str(archive["train_endpoint_sha256"]) == _endpoint_hash(
            _field(split, "train_t")
        )
        assert str(archive["val_endpoint_sha256"]) == _endpoint_hash(
            _field(split, "val_t")
        )


def test_load_split_rejects_tampered_endpoint_hash(synthetic_groups, tmp_path):
    split = _build_split(synthetic_groups)
    path = tmp_path / "split.npz"
    _save_split(path, split, "a" * 64)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["train_endpoint_sha256"] = np.asarray("0" * 64)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=r"(?i)(hash|integrity|endpoint)"):
        artifacts.load_split(path, expected_dataset_sha256="a" * 64)


def test_load_split_rejects_dataset_fingerprint_mismatch(
    synthetic_groups, tmp_path
):
    split = _build_split(synthetic_groups)
    path = tmp_path / "split.npz"
    _save_split(path, split, "a" * 64)

    with pytest.raises(ValueError, match=r"(?i)dataset"):
        artifacts.load_split(path, expected_dataset_sha256="b" * 64)


def test_load_split_rejects_legacy_position_only_archive(tmp_path):
    path = tmp_path / "legacy_subsample.npz"
    np.savez(
        path,
        train_pos=np.array([0, 2], dtype=np.int64),
        val_pos=np.array([1, 3], dtype=np.int64),
    )

    with pytest.raises(ValueError, match=r"(?i)(schema|legacy|train_t|endpoint)"):
        artifacts.load_split(path)


class _ExplodingArray:
    def __array__(self, dtype=None, copy=None):
        raise RuntimeError("synthetic serialization failure")


def test_atomic_savez_failure_preserves_existing_destination(tmp_path):
    output_dir = tmp_path / "npz"
    output_dir.mkdir()
    destination = output_dir / "artifact.npz"
    artifacts.atomic_savez(
        destination, stable=np.array([1, 2, 3], dtype=np.int64)
    )
    original = destination.read_bytes()

    with pytest.raises(RuntimeError, match="synthetic serialization failure"):
        artifacts.atomic_savez(destination, broken=_ExplodingArray())

    assert destination.read_bytes() == original
    assert list(output_dir.iterdir()) == [destination]
    with np.load(destination, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["stable"], np.array([1, 2, 3], dtype=np.int64)
        )


def test_atomic_json_failure_preserves_existing_destination(tmp_path):
    output_dir = tmp_path / "json"
    output_dir.mkdir()
    destination = output_dir / "manifest.json"
    artifacts.atomic_write_json(destination, {"stable": True})
    original = destination.read_bytes()

    with pytest.raises(TypeError):
        artifacts.atomic_write_json(destination, {"broken": object()})

    assert destination.read_bytes() == original
    assert list(output_dir.iterdir()) == [destination]
    assert json.loads(destination.read_text()) == {"stable": True}


def test_real_lobench_split_obeys_full_endpoint_contract():
    dataset = Path(__file__).resolve().parents[1] / "data" / "lobench_processed.npz"
    if not dataset.is_file():
        pytest.skip(f"real dataset is absent: {dataset}")

    with np.load(dataset, allow_pickle=False) as raw:
        stock_ids = raw["stock_ids"]
        day_ids = raw["day_ids"]
        book = raw["book"]

    vol_mask = np.empty(len(stock_ids), dtype=bool)
    chunk_size = 100_000
    for start in range(0, len(stock_ids), chunk_size):
        stop = min(start + chunk_size, len(stock_ids))
        volumes = book[start:stop, :, :, 1]
        vol_mask[start:stop] = (
            np.max(np.abs(volumes), axis=(1, 2)) <= 5.0
        )
    del book

    K = 20
    max_horizon = 20
    split = artifacts.build_endpoint_split(
        stock_ids,
        day_ids,
        vol_mask,
        K=K,
        max_horizon=max_horizon,
        val_frac=0.10,
        split_seed=0,
        n_train=100_000,
        n_val=50_000,
        subsample_seed=0,
    )
    valid_t = _field(split, "valid_t")
    train_pos = _field(split, "train_pos")
    val_pos = _field(split, "val_pos")
    train_t = _field(split, "train_t")
    val_t = _field(split, "val_t")

    np.testing.assert_array_equal(train_t, valid_t[train_pos])
    np.testing.assert_array_equal(val_t, valid_t[val_pos])

    train_groups = set(
        zip(stock_ids[train_t].tolist(), day_ids[train_t].tolist())
    )
    val_groups = set(zip(stock_ids[val_t].tolist(), day_ids[val_t].tolist()))
    assert train_groups.isdisjoint(val_groups)

    val_membership_pos = np.searchsorted(valid_t, val_t)
    assert np.all(val_membership_pos < len(valid_t))
    np.testing.assert_array_equal(valid_t[val_membership_pos], val_t)

    offsets = np.arange(-(K - 1), max_horizon + 1, dtype=np.int64)
    endpoints = np.concatenate([train_t, val_t])
    assert np.all(vol_mask[endpoints])
    for start in range(0, len(endpoints), 10_000):
        endpoint_chunk = endpoints[start : start + 10_000]
        indices = endpoint_chunk[:, None] + offsets[None, :]
        assert indices.min() >= 0
        assert indices.max() < len(stock_ids)
        assert np.all(
            stock_ids[indices] == stock_ids[endpoint_chunk, None]
        )
        assert np.all(day_ids[indices] == day_ids[endpoint_chunk, None])
        assert np.all(vol_mask[indices])

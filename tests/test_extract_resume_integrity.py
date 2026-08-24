from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiment01.historical import analysis_artifacts as artifacts
from experiment01.historical import extract_readouts_multiseed as stage1


def _readout_metadata() -> dict[str, object]:
    return {
        "schema_name": stage1.READOUT_SCHEMA,
        "schema_version": stage1.READOUT_VERSION,
        "artifact_fingerprint": "artifact-fingerprint",
        "protocol_fingerprint": "protocol-fingerprint",
        "split_fingerprint": "split-fingerprint",
        "dataset_sha256": "d" * 64,
        "train_endpoint_sha256": "t" * 64,
        "val_endpoint_sha256": "v" * 64,
        "checkpoint_sha256": "c" * 64,
        "stock_stats_sha256": "s" * 64,
        "arm": "jepa_horizon",
        "seed": 0,
        "epoch": 20,
        "n_train": 2,
        "n_val": 1,
    }


def _readout_arrays() -> dict[str, np.ndarray]:
    return {
        "last_concat512_train": np.zeros((2, 512), dtype=np.float32),
        "last_concat512_val": np.zeros((1, 512), dtype=np.float32),
        "tmean_concat512_train": np.ones((2, 512), dtype=np.float32),
        "tmean_concat512_val": np.ones((1, 512), dtype=np.float32),
    }


def _write_readout(
    path: Path,
    metadata: dict[str, object],
    *,
    arrays: dict[str, np.ndarray] | None = None,
    omit: set[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {**metadata, **(arrays or _readout_arrays())}
    for key in omit or set():
        payload.pop(key)
    stage1.atomic_savez(path, **payload)
    return {
        "path": str(path),
        "file_sha256": stage1.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "artifact_fingerprint": metadata["artifact_fingerprint"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "stock_stats_sha256": metadata["stock_stats_sha256"],
        "train_endpoint_sha256": metadata["train_endpoint_sha256"],
        "val_endpoint_sha256": metadata["val_endpoint_sha256"],
        "arrays": {
            "last_concat512_train": {
                "shape": [2, 512],
                "dtype": "float32",
            },
            "last_concat512_val": {
                "shape": [1, 512],
                "dtype": "float32",
            },
            "tmean_concat512_train": {
                "shape": [2, 512],
                "dtype": "float32",
            },
            "tmean_concat512_val": {
                "shape": [1, 512],
                "dtype": "float32",
            },
        },
    }


@pytest.fixture
def valid_readout(tmp_path):
    path = tmp_path / "readout.npz"
    expected = _readout_metadata()
    record = _write_readout(path, expected)
    return path, record, expected


def test_validate_readout_for_resume_accepts_valid_dump_and_manifest_entry(
    valid_readout,
):
    path, record, expected = valid_readout
    ok, reason = stage1.validate_readout_for_resume(
        path, record, expected, n_train=2, n_val=1
    )
    assert ok
    assert reason == "ok"


def test_validate_readout_for_resume_rejects_orphan_without_manifest_entry(
    valid_readout,
):
    path, _, expected = valid_readout
    ok, reason = stage1.validate_readout_for_resume(
        path, None, expected, n_train=2, n_val=1
    )
    assert not ok
    assert "manifest entry is missing" in reason


def test_validate_readout_for_resume_rejects_file_sha_tamper(valid_readout):
    path, record, expected = valid_readout
    with path.open("ab") as handle:
        handle.write(b"tampered")

    ok, reason = stage1.validate_readout_for_resume(
        path, record, expected, n_train=2, n_val=1
    )
    assert not ok
    assert "size differs" in reason or "SHA-256 differs" in reason


@pytest.mark.parametrize(
    ("field", "replacement", "reason_fragment"),
    [
        ("artifact_fingerprint", "stale-artifact", "artifact_fingerprint"),
        ("checkpoint_sha256", "stale-checkpoint", "checkpoint_sha256"),
        ("stock_stats_sha256", "stale-stock-stats", "stock_stats_sha256"),
        (
            "train_endpoint_sha256",
            "stale-train-endpoints",
            "train_endpoint_sha256",
        ),
        (
            "val_endpoint_sha256",
            "stale-validation-endpoints",
            "val_endpoint_sha256",
        ),
        (
            "arrays",
            {
                "last_concat512_train": {
                    "shape": [999, 512],
                    "dtype": "float32",
                }
            },
            "array schema",
        ),
    ],
)
def test_validate_readout_for_resume_rejects_stale_manifest_record(
    valid_readout, field, replacement, reason_fragment
):
    path, record, expected = valid_readout
    stale_record = deepcopy(record)
    stale_record[field] = replacement

    ok, reason = stage1.validate_readout_for_resume(
        path, stale_record, expected, n_train=2, n_val=1
    )
    assert not ok
    assert reason_fragment in reason


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("train_endpoint_sha256", "wrong-train-endpoints"),
        ("val_endpoint_sha256", "wrong-validation-endpoints"),
        ("checkpoint_sha256", "wrong-checkpoint"),
        ("stock_stats_sha256", "wrong-stock-stats"),
        ("protocol_fingerprint", "wrong-protocol-or-source"),
        ("split_fingerprint", "wrong-split"),
        ("dataset_sha256", "wrong-dataset"),
    ],
)
def test_validate_readout_for_resume_rejects_identity_mismatch(
    valid_readout, field, replacement
):
    path, record, expected = valid_readout
    incompatible = {**expected, field: replacement}

    ok, reason = stage1.validate_readout_for_resume(
        path, record, incompatible, n_train=2, n_val=1
    )
    assert not ok
    assert field in reason


def test_validate_readout_for_resume_rejects_schema_key_set(tmp_path):
    path = tmp_path / "legacy_readout.npz"
    expected = _readout_metadata()
    record = _write_readout(
        path, expected, omit={"schema_name", "train_endpoint_sha256"}
    )

    ok, reason = stage1.validate_readout_for_resume(
        path, record, expected, n_train=2, n_val=1
    )
    assert not ok
    assert "key set differs" in reason


@pytest.mark.parametrize(
    ("array_name", "bad_shape"),
    [
        ("last_concat512_train", (2, 511)),
        ("last_concat512_val", (2, 512)),
        ("tmean_concat512_train", (1, 512)),
        ("tmean_concat512_val", (1, 511)),
    ],
)
def test_validate_readout_for_resume_rejects_wrong_array_shape(
    tmp_path, array_name, bad_shape
):
    path = tmp_path / f"{array_name}.npz"
    expected = _readout_metadata()
    arrays = _readout_arrays()
    arrays[array_name] = np.zeros(bad_shape, dtype=np.float32)
    record = _write_readout(path, expected, arrays=arrays)

    ok, reason = stage1.validate_readout_for_resume(
        path, record, expected, n_train=2, n_val=1
    )
    assert not ok
    assert array_name in reason
    assert "shape mismatch" in reason


def _target_metadata() -> dict[str, object]:
    return {
        "schema_name": stage1.TARGET_SCHEMA,
        "schema_version": stage1.TARGET_VERSION,
        "artifact_fingerprint": "target-artifact-fingerprint",
        "protocol_fingerprint": "protocol-fingerprint",
        "split_fingerprint": "split-fingerprint",
        "dataset_sha256": "d" * 64,
        "train_endpoint_sha256": "t" * 64,
        "val_endpoint_sha256": "v" * 64,
        "n_train": 2,
        "n_val": 1,
    }


def _write_targets(
    path: Path, metadata: dict[str, object]
) -> dict[str, object]:
    stage1.atomic_savez(
        path,
        **metadata,
        y_train_raw=np.zeros((2, 3), dtype=np.float32),
        y_val_raw=np.zeros((1, 3), dtype=np.float32),
        target_names=np.asarray(["a", "b", "c"]),
    )
    return {
        "path": str(path),
        "file_sha256": stage1.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "artifact_fingerprint": metadata["artifact_fingerprint"],
        "shape_train": [2, 3],
        "shape_val": [1, 3],
        "train_endpoint_sha256": metadata["train_endpoint_sha256"],
        "val_endpoint_sha256": metadata["val_endpoint_sha256"],
    }


def test_validate_targets_for_resume_accepts_valid_dump_and_manifest_entry(
    tmp_path,
):
    path = tmp_path / "targets.npz"
    expected = _target_metadata()
    record = _write_targets(path, expected)

    ok, reason = stage1.validate_targets_for_resume(
        path, record, expected, n_train=2, n_val=1, n_targets=3
    )
    assert ok
    assert reason == "ok"


@pytest.mark.parametrize(
    ("field", "replacement", "reason_fragment"),
    [
        ("artifact_fingerprint", "stale-artifact", "artifact_fingerprint"),
        (
            "train_endpoint_sha256",
            "stale-train-endpoints",
            "train_endpoint_sha256",
        ),
        (
            "val_endpoint_sha256",
            "stale-validation-endpoints",
            "val_endpoint_sha256",
        ),
        ("shape_train", [999, 3], "train shape"),
        ("shape_val", [999, 3], "validation shape"),
    ],
)
def test_validate_targets_for_resume_rejects_stale_manifest_record(
    tmp_path, field, replacement, reason_fragment
):
    path = tmp_path / "targets.npz"
    expected = _target_metadata()
    record = _write_targets(path, expected)
    record[field] = replacement

    ok, reason = stage1.validate_targets_for_resume(
        path, record, expected, n_train=2, n_val=1, n_targets=3
    )
    assert not ok
    assert reason_fragment in reason


def _compatible_manifests() -> tuple[dict, dict]:
    requested = {
        "jepa_horizon_seed0_ep020": {
            "path": "/checkpoints/epoch_020.pt",
            "sha256": "c" * 64,
        }
    }
    existing = {
        "schema": {
            "name": stage1.MANIFEST_SCHEMA,
            "version": stage1.MANIFEST_VERSION,
        },
        "git": {"commit": "old-git-sha", "dirty": False},
        "dataset": {"sha256": "d" * 64},
        "source": {"fingerprint": "source-fingerprint"},
        "protocol": {"fingerprint": "protocol-fingerprint"},
        "split": {"fingerprint": "split-fingerprint"},
        "requested_checkpoints": requested,
    }
    planned = deepcopy(existing)
    planned["git"] = {"commit": "new-unrelated-git-sha", "dirty": False}
    return existing, planned


def test_manifest_compatibility_allows_unrelated_git_sha_change():
    existing, planned = _compatible_manifests()
    stage1._manifest_compatible(existing, planned)


@pytest.mark.parametrize(
    ("section", "label"),
    [
        ("source", "source fingerprint"),
        ("protocol", "protocol fingerprint"),
    ],
)
def test_manifest_compatibility_rejects_relevant_fingerprint_change(
    section, label
):
    existing, planned = _compatible_manifests()
    planned[section]["fingerprint"] = f"changed-{section}"

    with pytest.raises(RuntimeError, match=label):
        stage1._manifest_compatible(existing, planned)


def test_main_resume_keeps_manifest_inventory_exact_and_rejects_orphan(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.npz"
    book = np.zeros((20, 2, 2, 2), dtype=np.float32)
    day_ids = np.zeros(20, dtype=np.int64)
    day_ids[[8, 14]] = 1
    np.savez(
        dataset,
        book=book,
        mid_z=np.arange(20, dtype=np.float32),
        stock_ids=np.zeros(20, dtype=np.int64),
        day_ids=day_ids,
        min_spread_z_per_stock=np.ones(1, dtype=np.float32),
    )

    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = (
        checkpoint_root / "jepa_horizon" / "seed0" / "epoch_020.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint identity")
    output_dir = tmp_path / "output"

    valid_t = np.array([5, 8, 11, 14], dtype=np.int64)
    train_pos = np.array([0, 2], dtype=np.int64)
    val_pos = np.array([1, 3], dtype=np.int64)
    encoder_loads: list[Path] = []
    fake_stats = {"synthetic_stat": np.array([1.0], dtype=np.float32)}
    fake_stats_sha256 = stage1.stock_stats_fingerprint(fake_stats)

    def fake_compute_valid_endpoints(stock_ids, days, K, max_horizon, vol_mask):
        return valid_t.copy()

    def fake_grouped_split(stock_ids, days, endpoints, val_frac, seed):
        return train_pos.copy(), val_pos.copy()

    def fake_targets(book, mid_z, stock_ids, endpoints, min_spread):
        return (
            np.asarray(endpoints, dtype=np.float32)[:, None],
            ["synthetic_target"],
        )

    def fake_load_encoder(arm, checkpoint_path, device):
        encoder_loads.append(Path(checkpoint_path))
        return object(), fake_stats

    def fake_extract_poolings(encode, dataset, batch_size, num_workers, device):
        n_rows = len(dataset)
        values = np.arange(n_rows * 512, dtype=np.float32).reshape(
            n_rows, 512
        )
        return {
            "last_concat512": values,
            "tmean_concat512": values + 1.0,
        }

    monkeypatch.setattr(
        stage1, "compute_valid_endpoints", fake_compute_valid_endpoints
    )
    monkeypatch.setattr(
        stage1, "grouped_split_by_stock_day", fake_grouped_split
    )
    monkeypatch.setattr(stage1, "build_raw_targets", fake_targets)
    monkeypatch.setattr(
        stage1,
        "checkpoint_stock_stats_fingerprint",
        lambda path: fake_stats_sha256,
    )
    monkeypatch.setattr(stage1, "load_encoder", fake_load_encoder)
    monkeypatch.setattr(stage1, "extract_poolings", fake_extract_poolings)
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
            "2",
            "--n_val",
            "2",
            "--subsample_seed",
            "0",
            "--arms",
            "jepa_horizon",
            "--seeds",
            "0",
            "--epochs",
            "20",
            "--num_workers",
            "0",
            "--device",
            "cpu",
        ],
    )

    stage1.main()
    assert encoder_loads == [checkpoint]

    # A second identical run must validate and reuse the existing dump. It must
    # not touch the encoder, and the reused entry must remain in the manifest.
    monkeypatch.setattr(
        stage1,
        "load_encoder",
        lambda *args, **kwargs: pytest.fail(
            "a valid readout was regenerated instead of reused"
        ),
    )
    stage1.main()

    tag = "jepa_horizon_seed0_ep020"
    readout_paths = {
        path.relative_to(output_dir).as_posix()
        for path in (output_dir / "readouts").glob("*.npz")
    }
    manifest = json.loads(
        (output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    manifest_paths = {
        record["path"] for record in manifest["readouts"].values()
    }
    assert readout_paths == manifest_paths == {f"readouts/{tag}.npz"}
    assert set(manifest["readouts"]) == {tag}
    assert (
        manifest["requested_checkpoints"][tag]["stock_stats_sha256"]
        == fake_stats_sha256
    )
    assert manifest["readouts"][tag]["stock_stats_sha256"] == fake_stats_sha256
    assert manifest["summary"] == {
        "expected": 1,
        "complete": 1,
        "missing": [],
    }
    assert manifest["status"] == "complete"

    required_manifest_sections = {
        "schema",
        "status",
        "created_at_utc",
        "updated_at_utc",
        "git",
        "environment",
        "source",
        "dataset",
        "protocol",
        "split",
        "targets",
        "requested_checkpoints",
        "readouts",
        "summary",
    }
    assert required_manifest_sections <= set(manifest)
    assert manifest["schema"] == {
        "name": stage1.MANIFEST_SCHEMA,
        "version": stage1.MANIFEST_VERSION,
    }
    assert {"commit", "dirty"} <= set(manifest["git"])
    assert {
        "python",
        "numpy",
        "torch",
        "rocm",
        "actual_device",
        "device_name",
    } <= set(manifest["environment"])
    assert {"fingerprint", "files"} <= set(manifest["source"])
    assert set(stage1.SOURCE_FILES) == set(manifest["source"]["files"])
    assert {"path", "sha256", "size_bytes"} <= set(manifest["dataset"])
    assert {
        "fingerprint",
        "K",
        "max_horizon",
        "vol_clip",
        "val_frac",
        "split_seed",
        "subsample_seed",
        "requested_n_train",
        "requested_n_val",
        "grouping",
        "split_algorithm",
        "poolings",
        "batch_size",
        "num_workers",
    } <= set(manifest["protocol"])
    assert {
        "path",
        "file_sha256",
        "size_bytes",
        "fingerprint",
        "n_valid_total",
        "n_train",
        "n_val",
        "train_endpoint_sha256",
        "val_endpoint_sha256",
    } <= set(manifest["split"])
    assert {
        "path",
        "file_sha256",
        "size_bytes",
        "artifact_fingerprint",
        "shape_train",
        "shape_val",
        "train_endpoint_sha256",
        "val_endpoint_sha256",
    } <= set(manifest["targets"])

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(stage1.__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = ""
    if git_commit:
        assert manifest["git"]["commit"] == git_commit

    split_artifact = output_dir / manifest["split"]["path"]
    target_artifact = output_dir / manifest["targets"]["path"]
    readout_artifact = output_dir / manifest["readouts"][tag]["path"]
    with np.load(split_artifact, allow_pickle=False) as split_npz:
        assert str(split_npz["schema_name"]) == artifacts.SPLIT_SCHEMA
        assert int(split_npz["schema_version"]) == artifacts.SPLIT_VERSION
        split_train_hash = str(split_npz["train_endpoint_sha256"])
        split_val_hash = str(split_npz["val_endpoint_sha256"])
        assert split_train_hash == stage1.endpoint_sha256(split_npz["train_t"])
        assert split_val_hash == stage1.endpoint_sha256(split_npz["val_t"])
    with np.load(target_artifact, allow_pickle=False) as targets_npz:
        assert str(targets_npz["schema_name"]) == stage1.TARGET_SCHEMA
        assert int(targets_npz["schema_version"]) == stage1.TARGET_VERSION
        target_train_hash = str(targets_npz["train_endpoint_sha256"])
        target_val_hash = str(targets_npz["val_endpoint_sha256"])
    with np.load(readout_artifact, allow_pickle=False) as readout_npz:
        assert str(readout_npz["schema_name"]) == stage1.READOUT_SCHEMA
        assert int(readout_npz["schema_version"]) == stage1.READOUT_VERSION
        readout_train_hash = str(readout_npz["train_endpoint_sha256"])
        readout_val_hash = str(readout_npz["val_endpoint_sha256"])
        assert str(readout_npz["stock_stats_sha256"]) == fake_stats_sha256

    assert {
        split_train_hash,
        target_train_hash,
        readout_train_hash,
        manifest["split"]["train_endpoint_sha256"],
        manifest["targets"]["train_endpoint_sha256"],
        manifest["readouts"][tag]["train_endpoint_sha256"],
    } == {split_train_hash}
    assert {
        split_val_hash,
        target_val_hash,
        readout_val_hash,
        manifest["split"]["val_endpoint_sha256"],
        manifest["targets"]["val_endpoint_sha256"],
        manifest["readouts"][tag]["val_endpoint_sha256"],
    } == {split_val_hash}
    assert manifest["split"]["file_sha256"] == stage1.sha256_file(
        split_artifact
    )
    assert manifest["targets"]["file_sha256"] == stage1.sha256_file(
        target_artifact
    )
    assert manifest["readouts"][tag]["file_sha256"] == stage1.sha256_file(
        readout_artifact
    )

    # Equal counts are not enough: any undeclared NPZ makes the directory
    # incompatible and must be rejected before an encoder is loaded.
    stage1.atomic_savez(
        output_dir / "readouts" / "orphan.npz",
        unexpected=np.array([1], dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match=r"unexpected readout NPZ"):
        stage1.main()

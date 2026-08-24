from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiment01.historical import analysis_artifacts as artifacts
from experiment01.historical import ladder_accessibility as ladder


def _build_stage1_tree(
    root: Path,
    *,
    target_names: list[str] | None = None,
    with_heldout: bool = False,
    heldout_train_hash: str | None = None,
) -> tuple[Path, Path | None]:
    root.mkdir()
    (root / "readouts").mkdir()
    dataset_sha = "d" * 64
    protocol_fingerprint = "p" * 64
    split = artifacts.SplitArtifact(
        valid_t=np.array([3, 5, 7, 9], dtype=np.int64),
        train_pos=np.array([0, 2], dtype=np.int64),
        val_pos=np.array([1, 3], dtype=np.int64),
        train_t=np.array([3, 7], dtype=np.int64),
        val_t=np.array([5, 9], dtype=np.int64),
    )
    split_path = root / "split.npz"
    artifacts.save_split(
        split_path,
        split,
        dataset_sha256=dataset_sha,
        split_config={
            "K": 3,
            "max_horizon": 2,
            "vol_clip": 5.0,
            "val_frac": 0.25,
            "split_seed": 0,
            "subsample_seed": 0,
            "n_train": 2,
            "n_val": 2,
            "grouping": "stock_id+day_id",
        },
    )

    canonical_names = ladder.dir_indices()["names"]
    names = canonical_names if target_names is None else target_names
    target_fingerprint = "t" * 64
    targets_path = root / "targets_shared.npz"
    artifacts.atomic_savez(
        targets_path,
        schema_name=np.asarray(artifacts.TARGET_SCHEMA),
        schema_version=np.asarray(artifacts.TARGET_VERSION, dtype=np.int64),
        artifact_fingerprint=np.asarray(target_fingerprint),
        protocol_fingerprint=np.asarray(protocol_fingerprint),
        split_fingerprint=np.asarray(split.split_fingerprint),
        dataset_sha256=np.asarray(dataset_sha),
        train_endpoint_sha256=np.asarray(split.train_endpoint_sha256),
        val_endpoint_sha256=np.asarray(split.val_endpoint_sha256),
        n_train=np.asarray(2, dtype=np.int64),
        n_val=np.asarray(2, dtype=np.int64),
        y_train_raw=np.zeros((2, len(names)), dtype=np.float32),
        y_val_raw=np.zeros((2, len(names)), dtype=np.float32),
        target_names=np.asarray(names, dtype=str),
    )

    tag = "supervised_seed0_ep020"
    checkpoint_sha = "c" * 64
    stock_stats_sha = "u" * 64
    readout_fingerprint = "r" * 64
    readout_path = root / "readouts" / f"{tag}.npz"
    arrays = {
        "last_concat512_train": np.zeros((2, 512), dtype=np.float32),
        "last_concat512_val": np.zeros((2, 512), dtype=np.float32),
        "tmean_concat512_train": np.zeros((2, 512), dtype=np.float32),
        "tmean_concat512_val": np.zeros((2, 512), dtype=np.float32),
    }
    artifacts.atomic_savez(
        readout_path,
        schema_name=np.asarray(artifacts.READOUT_SCHEMA),
        schema_version=np.asarray(artifacts.READOUT_VERSION, dtype=np.int64),
        artifact_fingerprint=np.asarray(readout_fingerprint),
        protocol_fingerprint=np.asarray(protocol_fingerprint),
        split_fingerprint=np.asarray(split.split_fingerprint),
        dataset_sha256=np.asarray(dataset_sha),
        train_endpoint_sha256=np.asarray(split.train_endpoint_sha256),
        val_endpoint_sha256=np.asarray(split.val_endpoint_sha256),
        checkpoint_sha256=np.asarray(checkpoint_sha),
        stock_stats_sha256=np.asarray(stock_stats_sha),
        arm=np.asarray("supervised"),
        seed=np.asarray(0, dtype=np.int64),
        epoch=np.asarray(20, dtype=np.int64),
        n_train=np.asarray(2, dtype=np.int64),
        n_val=np.asarray(2, dtype=np.int64),
        **arrays,
    )
    array_record = {
        key: {"shape": list(value.shape), "dtype": "float32"}
        for key, value in arrays.items()
    }
    manifest = {
        "schema": {
            "name": artifacts.MANIFEST_SCHEMA,
            "version": artifacts.MANIFEST_VERSION,
        },
        "status": "complete",
        "dataset": {"sha256": dataset_sha},
        "protocol": {"fingerprint": protocol_fingerprint},
        "split": {
            "path": "split.npz",
            "file_sha256": artifacts.sha256_file(split_path),
            "size_bytes": split_path.stat().st_size,
            "fingerprint": split.split_fingerprint,
            "n_valid_total": 4,
            "n_train": 2,
            "n_val": 2,
            "train_endpoint_sha256": split.train_endpoint_sha256,
            "val_endpoint_sha256": split.val_endpoint_sha256,
        },
        "targets": {
            "path": "targets_shared.npz",
            "file_sha256": artifacts.sha256_file(targets_path),
            "size_bytes": targets_path.stat().st_size,
            "artifact_fingerprint": target_fingerprint,
            "shape_train": [2, len(names)],
            "shape_val": [2, len(names)],
            "train_endpoint_sha256": split.train_endpoint_sha256,
            "val_endpoint_sha256": split.val_endpoint_sha256,
        },
        "requested_checkpoints": {
            tag: {
                "arm": "supervised",
                "seed": 0,
                "epoch": 20,
                "sha256": checkpoint_sha,
                "stock_stats_sha256": stock_stats_sha,
            }
        },
        "readouts": {
            tag: {
                "path": f"readouts/{tag}.npz",
                "file_sha256": artifacts.sha256_file(readout_path),
                "size_bytes": readout_path.stat().st_size,
                "artifact_fingerprint": readout_fingerprint,
                "checkpoint_sha256": checkpoint_sha,
                "stock_stats_sha256": stock_stats_sha,
                "train_endpoint_sha256": split.train_endpoint_sha256,
                "val_endpoint_sha256": split.val_endpoint_sha256,
                "arrays": array_record,
            }
        },
        "summary": {"expected": 1, "complete": 1, "missing": []},
    }

    heldout_path = None
    if with_heldout:
        heldout_path = root / "targets_heldout.npz"
        heldout_fingerprint = "h" * 64
        recorded_train_hash = (
            split.train_endpoint_sha256
            if heldout_train_hash is None
            else heldout_train_hash
        )
        artifacts.atomic_savez(
            heldout_path,
            schema_name=np.asarray(artifacts.TARGET_SCHEMA),
            schema_version=np.asarray(artifacts.TARGET_VERSION, dtype=np.int64),
            artifact_kind=np.asarray("heldout_targets"),
            artifact_fingerprint=np.asarray(heldout_fingerprint),
            source_sha256=np.asarray("s" * 64),
            split_schema_name=np.asarray(artifacts.SPLIT_SCHEMA),
            split_schema_version=np.asarray(
                artifacts.SPLIT_VERSION, dtype=np.int64
            ),
            split_fingerprint=np.asarray(split.split_fingerprint),
            dataset_sha256=np.asarray(dataset_sha),
            train_endpoint_sha256=np.asarray(recorded_train_hash),
            val_endpoint_sha256=np.asarray(split.val_endpoint_sha256),
            n_train=np.asarray(2, dtype=np.int64),
            n_val=np.asarray(2, dtype=np.int64),
            y_train_heldout=np.zeros((2, 1), dtype=np.float32),
            y_val_heldout=np.zeros((2, 1), dtype=np.float32),
            heldout_names=np.asarray(["d_imbalance_all@1"]),
        )
        manifest["heldout_targets"] = {
            "path": heldout_path.name,
            "file_sha256": artifacts.sha256_file(heldout_path),
            "size_bytes": heldout_path.stat().st_size,
            "artifact_fingerprint": heldout_fingerprint,
            "shape_train": [2, 1],
            "shape_val": [2, 1],
            "train_endpoint_sha256": recorded_train_hash,
            "val_endpoint_sha256": split.val_endpoint_sha256,
        }

    with (root / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return readout_path, heldout_path


def test_preflight_accepts_exact_v2_inventory(tmp_path):
    readout_path, heldout_path = _build_stage1_tree(
        tmp_path / "stage1", with_heldout=True
    )

    inventory = ladder.validate_stage1_inputs(
        tmp_path / "stage1", str(heldout_path)
    )

    assert inventory["readout_paths"] == [readout_path]
    assert inventory["heldout_path"] == heldout_path.resolve()


def test_preflight_rejects_noncanonical_trained_target_order(tmp_path):
    names = ladder.dir_indices()["names"]
    names[0], names[1] = names[1], names[0]
    _build_stage1_tree(tmp_path / "stage1", target_names=names)

    with pytest.raises(RuntimeError, match="target_names.*canonical order"):
        ladder.validate_stage1_inputs(tmp_path / "stage1")


def test_preflight_rejects_same_shape_heldout_from_different_endpoints(tmp_path):
    _, heldout_path = _build_stage1_tree(
        tmp_path / "stage1",
        with_heldout=True,
        heldout_train_hash="x" * 64,
    )

    with pytest.raises(RuntimeError, match="held-out metadata mismatch"):
        ladder.validate_stage1_inputs(
            tmp_path / "stage1", str(heldout_path)
        )


def test_preflight_rejects_extra_readout_file(tmp_path):
    _build_stage1_tree(tmp_path / "stage1")
    np.savez(tmp_path / "stage1" / "readouts" / "extra.npz", x=np.ones(1))

    with pytest.raises(RuntimeError, match="inventory exactly"):
        ladder.validate_stage1_inputs(tmp_path / "stage1")


def test_preflight_rejects_readout_whose_sha_differs_from_manifest(tmp_path):
    readout_path, _ = _build_stage1_tree(tmp_path / "stage1")
    with readout_path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(RuntimeError, match=r"readout .*(size|SHA-256)"):
        ladder.validate_stage1_inputs(tmp_path / "stage1")

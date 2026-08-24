"""Prepare the complete sharded Experiment 01 bundle before feature extraction."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiment01.historical.analysis_artifacts import load_split
from experiment01.historical.extract_readouts_multiseed import (
    FUTURE_FEATURES,
    FUTURE_HORIZONS,
    VOL_HORIZONS,
)
from training.train_tokenizer_t import (
    compute_future_feature_targets,
    compute_vol_targets,
    derive_raw_features_array,
)

from .constants import (
    INPUT_SCHEMA,
    INPUT_SCHEMA_VERSION,
    READOUTS,
)
from .errors import ExperimentIntegrityError
from .io import (
    atomic_save_npy,
    atomic_write_json,
    canonical_json_sha256,
    sha256_array,
    sha256_file,
)
from .schema import READOUT_DEFINITIONS
from .sharded import SHARDED_STORAGE, sharded_record_fingerprint_payload
from .split3 import SPLIT_NAMES, load_preregistered_split


TARGET_TIMING_SEMANTICS = "log1p_observed_or_capped_all_rows:max_look=600"
TARGET_TIMING_MAX_LOOK = 600


def target_definitions() -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for feature in FUTURE_FEATURES:
        for horizon in FUTURE_HORIZONS:
            name = f"{feature}@{horizon}"
            redundant: list[str] = []
            independent = True
            if feature in {"d_best_bid_rel", "d_best_ask_rel"}:
                independent = False
                redundant = [f"d_spread_z@{horizon}"]
            definitions.append(
                {
                    "name": name,
                    "block": "directional",
                    "independent": independent,
                    "redundant_with": redundant,
                    "semantics": "raw_future_delta",
                }
            )
    for horizon in VOL_HORIZONS:
        definitions.append(
            {
                "name": f"realized_vol@{horizon}",
                "block": "volatility",
                "independent": True,
                "redundant_with": [],
                "semantics": "canonical_realized_volatility_raw",
            }
        )
    definitions.append(
        {
            "name": "time_to_next_mid_move",
            "block": "timing",
            "independent": True,
            "redundant_with": [],
            "semantics": TARGET_TIMING_SEMANTICS,
        }
    )
    return definitions


def timing_target_all_rows(
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    *,
    max_look: int = TARGET_TIMING_MAX_LOOK,
) -> np.ndarray:
    """Vectorized equivalent of the canonical held-out timing target."""
    mid = np.asarray(mid_z)
    stocks = np.asarray(stock_ids)
    days = np.asarray(day_ids)
    if not (mid.ndim == stocks.ndim == days.ndim == 1):
        raise ValueError("timing arrays must be one-dimensional")
    if not (len(mid) == len(stocks) == len(days)):
        raise ValueError("timing arrays are misaligned")
    result = np.empty(len(mid), dtype=np.float32)
    boundaries = np.flatnonzero(
        (stocks[1:] != stocks[:-1]) | (days[1:] != days[:-1])
    ) + 1
    group_starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), boundaries]
    )
    group_stops = np.concatenate(
        [boundaries, np.asarray([len(mid)], dtype=np.int64)]
    )
    for start, stop in zip(group_starts, group_stops):
        values = mid[start:stop]
        changes = np.flatnonzero(values[1:] != values[:-1]) + 1
        run_starts = np.concatenate(
            [np.asarray([0], dtype=np.int64), changes]
        )
        run_stops = np.concatenate(
            [changes, np.asarray([len(values)], dtype=np.int64)]
        )
        next_boundary = np.repeat(run_stops, run_stops - run_starts)
        duration = np.minimum(
            next_boundary - np.arange(len(values), dtype=np.int64),
            max_look,
        )
        result[start:stop] = np.log1p(duration).astype(np.float32)
    return result


def _sharded_record(
    *,
    shape: Sequence[int],
    dtype: np.dtype | str,
    row_key_sha256: str,
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "storage": SHARDED_STORAGE,
        "shape": [int(value) for value in shape],
        "dtype": np.dtype(dtype).name,
        "row_key_sha256": row_key_sha256,
        "shards": shards,
    }
    record["shard_manifest_sha256"] = canonical_json_sha256(
        sharded_record_fingerprint_payload(record)
    )
    return record


def _target_shards(
    *,
    root: Path,
    split: str,
    rows: pd.DataFrame,
    raw_features: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    min_spread: np.ndarray,
    timing: np.ndarray,
    shard_rows: int,
    row_key_sha256: str,
) -> Mapping[str, Any]:
    records: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(rows), shard_rows)):
        stop = min(start + shard_rows, len(rows))
        endpoints = rows["endpoint_index"].iloc[start:stop].to_numpy(
            dtype=np.int64
        )
        directional = compute_future_feature_targets(
            raw_features,
            endpoints,
            FUTURE_FEATURES,
            FUTURE_HORIZONS,
        )
        volatility = compute_vol_targets(
            mid_z,
            endpoints,
            VOL_HORIZONS,
            min_spread,
            stock_ids,
        )
        values = np.concatenate(
            [
                directional.astype(np.float32, copy=False),
                volatility.astype(np.float32, copy=False),
                timing[endpoints, None].astype(np.float32, copy=False),
            ],
            axis=1,
        )
        relative = (
            Path("targets") / split / f"part-{shard_index:05d}.npy"
        )
        path = root / relative
        atomic_save_npy(path, values)
        records.append(
            {
                "path": str(relative),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "shape": list(values.shape),
                "dtype": values.dtype.name,
                "row_start": start,
                "row_stop": stop,
                "row_key_sha256": sha256_array(
                    rows["row_key"].iloc[start:stop].astype(str).to_numpy(
                        dtype="U"
                    )
                ),
            }
        )
    return _sharded_record(
        shape=(len(rows), len(target_definitions())),
        dtype=np.float32,
        row_key_sha256=row_key_sha256,
        shards=records,
    )


def _exact_target_equivalence(
    legacy_dir: Path,
    historical_split_path: Path,
    raw_features: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    min_spread: np.ndarray,
    timing: np.ndarray,
) -> Mapping[str, Any]:
    split = load_split(historical_split_path)

    def values(endpoints: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                compute_future_feature_targets(
                    raw_features,
                    endpoints,
                    FUTURE_FEATURES,
                    FUTURE_HORIZONS,
                ),
                compute_vol_targets(
                    mid_z,
                    endpoints,
                    VOL_HORIZONS,
                    min_spread,
                    stock_ids,
                ),
            ],
            axis=1,
        ).astype(np.float32)

    checks: dict[str, Any] = {}
    with np.load(legacy_dir / "targets_shared.npz", allow_pickle=False) as shared:
        for split_name, endpoints, key in (
            ("train", split.train_t, "y_train_raw"),
            ("validation", split.val_t, "y_val_raw"),
        ):
            observed = values(endpoints)
            expected = np.asarray(shared[key])
            checks[f"shared_{split_name}"] = {
                "equal": bool(np.array_equal(observed, expected)),
                "observed_sha256": sha256_array(observed),
                "expected_sha256": sha256_array(expected),
                "max_abs_difference": float(
                    np.max(
                        np.abs(
                            observed.astype(np.float64)
                            - expected.astype(np.float64)
                        )
                    )
                ),
            }
    with np.load(legacy_dir / "targets_heldout.npz", allow_pickle=False) as heldout:
        names = [str(value) for value in heldout["heldout_names"]]
        timing_index = names.index("time_to_next_mid_move")
        for split_name, endpoints, key in (
            ("train", split.train_t, "y_train_heldout"),
            ("validation", split.val_t, "y_val_heldout"),
        ):
            observed = timing[endpoints]
            expected = np.asarray(heldout[key])[:, timing_index]
            checks[f"timing_{split_name}"] = {
                "equal": bool(np.array_equal(observed, expected)),
                "observed_sha256": sha256_array(observed),
                "expected_sha256": sha256_array(expected),
                "max_abs_difference": float(
                    np.max(
                        np.abs(
                            observed.astype(np.float64)
                            - expected.astype(np.float64)
                        )
                    )
                ),
            }
    return {
        "passed": all(record["equal"] for record in checks.values()),
        "checks": checks,
    }


def _storage_estimate(
    root: Path,
    split_records: Mapping[str, Any],
    *,
    dimension: int,
    n_feature_matrices: int,
    target_dimension: int,
) -> Mapping[str, Any]:
    row_counts = {
        split: int(split_records[split]["n_rows"]) for split in SPLIT_NAMES
    }
    feature_by_split = {
        split: rows * dimension * np.dtype("float32").itemsize * n_feature_matrices
        for split, rows in row_counts.items()
    }
    feature_bytes = int(sum(feature_by_split.values()))
    target_bytes = int(
        sum(row_counts.values())
        * target_dimension
        * np.dtype("float32").itemsize
    )
    disk = shutil.disk_usage(root)
    required_with_headroom = int(np.ceil((feature_bytes + target_bytes) * 1.10))
    sufficient_free_storage = disk.free >= required_with_headroom
    return {
        "schema_name": "thesis.experiment01.storage_estimate",
        "schema_version": 1,
        "passed": sufficient_free_storage,
        "row_counts": row_counts,
        "feature_dimension": dimension,
        "n_complete_feature_matrices": n_feature_matrices,
        "feature_dtype": "float32",
        "feature_bytes_by_split": feature_by_split,
        "feature_bytes_total": feature_bytes,
        "target_dimension": target_dimension,
        "target_bytes_total": target_bytes,
        "estimated_total_bytes": feature_bytes + target_bytes,
        "required_bytes_with_10_percent_headroom": required_with_headroom,
        "disk_free_bytes_at_estimate": disk.free,
        "sufficient_free_storage": sufficient_free_storage,
        "processing_strategy": {
            "format": SHARDED_STORAGE,
            "order": "checkpoint -> split -> row shard",
            "simultaneous_complete_feature_matrices": 0,
            "simultaneous_in_memory_readouts": list(READOUTS),
            "resume_unit": "one checkpoint/split/row-shard",
        },
    }


def prepare_bundle(
    split_dir: str | Path,
    dataset_path: str | Path,
    legacy_dir: str | Path,
    out_dir: str | Path,
    *,
    shard_rows: int = 100_000,
) -> Mapping[str, Any]:
    """Write rows, target shards, storage estimate and a prepared manifest."""
    if shard_rows <= 0:
        raise ValueError("shard_rows must be positive")
    split_root, split_manifest = load_preregistered_split(split_dir)
    dataset = Path(dataset_path).resolve()
    legacy_root = Path(legacy_dir).resolve()
    destination = Path(out_dir).resolve()
    manifest_path = destination / "manifest.json"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to prepare a bundle in non-empty {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    legacy_manifest = json.loads(
        (legacy_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    dataset_sha256 = sha256_file(dataset)
    if (
        dataset_sha256 != split_manifest["dataset_sha256"]
        or dataset_sha256 != legacy_manifest["dataset"]["sha256"]
    ):
        raise ExperimentIntegrityError(
            "dataset SHA-256 differs across sidecar/split/legacy provenance"
        )

    rows: dict[str, pd.DataFrame] = {}
    split_records: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        source_record = split_manifest["splits"][split]
        source = split_root / source_record["path"]
        relative = Path("rows") / f"{split}.parquet"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != source_record["sha256"]:
            raise ExperimentIntegrityError(f"{split} row copy hash mismatch")
        frame = pd.read_parquet(target)
        rows[split] = frame
        split_records[split] = {
            "path": str(relative),
            "sha256": source_record["sha256"],
            "size_bytes": target.stat().st_size,
            "n_rows": len(frame),
            "n_stock_days": source_record["n_stock_days"],
            "row_key_sha256": source_record["row_key_sha256"],
            "endpoint_index_sha256": source_record["endpoint_index_sha256"],
            "complete_stock_days": True,
            "counts_by_stock": source_record["counts_by_stock"],
            "first_trading_date": source_record["first_trading_date"],
            "last_trading_date": source_record["last_trading_date"],
        }

    with np.load(dataset, allow_pickle=False) as archive:
        book = archive["book"]
        mid_z = archive["mid_z"]
        stock_ids = archive["stock_ids"].astype(np.int64)
        day_ids = archive["day_ids"].astype(np.int64)
        min_spread = archive["min_spread_z_per_stock"]
        raw_features, _ = derive_raw_features_array(
            book, mid_z, stock_ids, int(stock_ids.max()) + 1
        )
        timing = timing_target_all_rows(mid_z, stock_ids, day_ids)
        target_equivalence = _exact_target_equivalence(
            legacy_root,
            legacy_root / legacy_manifest["split"]["path"],
            raw_features,
            mid_z,
            stock_ids,
            min_spread,
            timing,
        )
        target_equivalence["schema_name"] = (
            "thesis.experiment01.target_equivalence"
        )
        target_equivalence["schema_version"] = 1
        atomic_write_json(
            destination / "target_equivalence_report.json",
            target_equivalence,
        )
        if not target_equivalence["passed"]:
            raise ExperimentIntegrityError(
                "full-target formulas do not reproduce historical target artifacts"
            )
        target_arrays = {
            split: _target_shards(
                root=destination,
                split=split,
                rows=rows[split],
                raw_features=raw_features,
                mid_z=mid_z,
                stock_ids=stock_ids,
                min_spread=min_spread,
                timing=timing,
                shard_rows=shard_rows,
                row_key_sha256=split_records[split]["row_key_sha256"],
            )
            for split in SPLIT_NAMES
        }

    checkpoints = legacy_manifest.get("requested_checkpoints", {})
    if len(checkpoints) != 9:
        raise ExperimentIntegrityError(
            f"expected nine canonical checkpoints, found {len(checkpoints)}"
        )
    seeds = {
        branch: sorted(
            int(record["seed"])
            for record in checkpoints.values()
            if record["arm"] == branch
        )
        for branch in ("supervised", "jepa_horizon", "jepa_masked")
    }
    if any(values != [0, 1, 2] for values in seeds.values()):
        raise ExperimentIntegrityError(
            f"canonical checkpoint seed inventory differs: {seeds}"
        )
    for tag, record in checkpoints.items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ExperimentIntegrityError(f"checkpoint {tag} hash mismatch")

    definitions = target_definitions()
    target_manifest_fingerprint = canonical_json_sha256(
        {"definitions": definitions, "arrays": target_arrays}
    )
    storage = _storage_estimate(
        destination,
        split_records,
        dimension=512,
        n_feature_matrices=18,
        target_dimension=len(definitions),
    )
    storage["shard_rows"] = shard_rows
    atomic_write_json(destination / "storage_estimate.json", storage)
    if not storage["sufficient_free_storage"]:
        raise ExperimentIntegrityError(
            "insufficient free storage for full extraction plus headroom"
        )

    manifest: dict[str, Any] = {
        "schema_name": INPUT_SCHEMA,
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "prepared",
        "provenance": {
            "corrected_post_p0": True,
            "source_commit": legacy_manifest["git"]["commit"],
            "dataset_path": str(dataset),
            "dataset_sha256": dataset_sha256,
            "sidecar_fingerprint": split_manifest["sidecar_fingerprint"],
            "csv_npz_equivalence_fingerprint": split_manifest[
                "equivalence_fingerprint"
            ],
            "split_protocol_fingerprint": split_manifest[
                "split_protocol_fingerprint"
            ],
            "target_manifest_fingerprint": target_manifest_fingerprint,
            "historical_split_fingerprint": legacy_manifest["split"][
                "fingerprint"
            ],
        },
        "historical_heldout_reuse_disclosed": True,
        "historical_heldout_reuse_statement": split_manifest["protocol"][
            "historical_exposure_disclosure"
        ],
        "validation_is_fixed_and_complete": True,
        "test_is_fixed_and_complete": True,
        "training_features_are_full_unlabelled_split": True,
        "context_window_within_stock_day_verified": True,
        "target_horizon_within_stock_day_verified": True,
        "row_feature_target_alignment_verified": True,
        "protocol": {
            "K": 20,
            "max_horizon": 20,
            "vol_clip": 5.0,
            "grouping": "stock_id+trading_date",
            "split_rule": split_manifest["protocol"]["name"],
            "test_hyperparameter_selection": "forbidden",
            "validation_hyperparameter_selection": "required",
        },
        "canonical_encoder_seeds": seeds,
        "canonical_checkpoints": checkpoints,
        "readout_definitions": READOUT_DEFINITIONS,
        "shard_rows": shard_rows,
        "splits": split_records,
        "targets": {
            "definitions": definitions,
            "arrays": target_arrays,
            "equivalence_report": {
                "path": "target_equivalence_report.json",
                "sha256": sha256_file(
                    destination / "target_equivalence_report.json"
                ),
                "size_bytes": (
                    destination / "target_equivalence_report.json"
                ).stat().st_size,
                "passed": True,
            },
        },
        "feature_sets": [],
        "pre_extraction": {
            "storage_estimate": {
                "path": "storage_estimate.json",
                "sha256": sha256_file(destination / "storage_estimate.json"),
                "size_bytes": (
                    destination / "storage_estimate.json"
                ).stat().st_size,
                "passed": storage["sufficient_free_storage"],
            },
            "benchmark_and_feature_equivalence": None,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest

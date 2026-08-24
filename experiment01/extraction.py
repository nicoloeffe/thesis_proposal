"""Pre-extraction gates and sequential sharded feature extraction."""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiment01.reference.analysis_artifacts import load_split
from experiment01.reference.extract_readouts_multiseed import (
    RawWindowDataset,
    load_encoder,
)

from .constants import BRANCHES, READOUTS, SPLITS
from .errors import ExperimentIntegrityError
from .io import (
    atomic_save_npy,
    atomic_write_json,
    canonical_json_sha256,
    sha256_array,
    sha256_file,
)
from .sharded import SHARDED_STORAGE, sharded_record_fingerprint_payload


PREEXTRACT_SCHEMA = "thesis.experiment01.pre_extraction_gate"
PREEXTRACT_SCHEMA_VERSION = 1
EXTRACTION_STATE_SCHEMA = "thesis.experiment01.feature_extraction_state"
EXTRACTION_STATE_VERSION = 1


def _load_manifest(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / "manifest.json"
    if not path.is_file():
        raise ExperimentIntegrityError("bundle manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"prepared", "extracting", "complete"}:
        raise ExperimentIntegrityError("bundle is not prepared for extraction")
    return manifest


def _device(name: str) -> torch.device:
    value = torch.device(name)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentIntegrityError(
            f"requested device {name!r} is unavailable"
        )
    return value


@torch.inference_mode()
def _iter_poolings(
    encode,
    dataset: RawWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    for book, stock_ids in loader:
        book = book.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        grid = encode(book, stock_ids)
        batch = grid.shape[0]
        last = (
            grid[:, -1, :, :]
            .reshape(batch, -1)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        mean = (
            grid.mean(dim=1)
            .reshape(batch, -1)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        if last.shape[1] != 512 or mean.shape[1] != 512:
            raise ExperimentIntegrityError(
                f"encoder returned non-canonical readout shapes "
                f"{last.shape}/{mean.shape}"
            )
        yield last, mean


def _extract_arrays(
    encode,
    dataset: RawWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    last_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    for last, mean in _iter_poolings(
        encode,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    ):
        last_parts.append(last)
        mean_parts.append(mean)
    if not last_parts:
        return (
            np.empty((0, 512), dtype=np.float32),
            np.empty((0, 512), dtype=np.float32),
        )
    return np.concatenate(last_parts), np.concatenate(mean_parts)


def _benchmark(
    encode,
    dataset: RawWindowDataset,
    *,
    label: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Mapping[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    rows = 0
    checksum = 0.0
    for last, mean in _iter_poolings(
        encode,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    ):
        rows += len(last)
        checksum += float(last[:, 0].sum(dtype=np.float64))
        checksum += float(mean[:, 0].sum(dtype=np.float64))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "n_rows": rows,
        "runtime_seconds": elapsed,
        "rows_per_second": rows / elapsed if elapsed > 0 else None,
        "readout_checksum": checksum,
    }


def _comparison(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> Mapping[str, Any]:
    if observed.shape != expected.shape:
        return {
            "passed": False,
            "shape_equal": False,
            "observed_shape": list(observed.shape),
            "expected_shape": list(expected.shape),
        }
    delta = np.abs(
        observed.astype(np.float64) - expected.astype(np.float64)
    )
    exact = np.equal(observed, expected)
    return {
        "passed": bool(np.allclose(observed, expected, rtol=rtol, atol=atol)),
        "shape_equal": True,
        "dtype_equal": observed.dtype == expected.dtype,
        "exact_equal": bool(np.all(exact)),
        "exact_mismatch_count": int(exact.size - np.count_nonzero(exact)),
        "max_abs_difference": float(delta.max(initial=0.0)),
        "mean_abs_difference": float(delta.mean()) if delta.size else 0.0,
        "observed_sha256": sha256_array(observed),
        "expected_sha256": sha256_array(expected),
        "rtol": rtol,
        "atol": atol,
    }


def run_pre_extraction_gate(
    bundle_dir: str | Path,
    legacy_dir: str | Path,
    *,
    device_name: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 2,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    benchmark_checkpoint: str = "supervised_seed0_ep020",
) -> Mapping[str, Any]:
    """Benchmark and reproduce every legacy readout before full extraction."""
    root = Path(bundle_dir).resolve()
    legacy_root = Path(legacy_dir).resolve()
    manifest = _load_manifest(root)
    report_path = root / "pre_extraction_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    storage_record = manifest.get("pre_extraction", {}).get(
        "storage_estimate", {}
    )
    storage_path = root / str(storage_record.get("path", ""))
    if (
        not storage_path.is_file()
        or sha256_file(storage_path) != storage_record.get("sha256")
        or storage_record.get("passed") is not True
    ):
        raise ExperimentIntegrityError("storage estimate gate is not valid")
    legacy_manifest = json.loads(
        (legacy_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    if (
        legacy_manifest["dataset"]["sha256"]
        != manifest["provenance"]["dataset_sha256"]
    ):
        raise ExperimentIntegrityError(
            "historical readout/bundle dataset provenance differs"
        )
    device = _device(device_name)
    dataset_path = Path(manifest["provenance"]["dataset_path"])
    historical = load_split(
        legacy_root / legacy_manifest["split"]["path"],
        expected_dataset_sha256=manifest["provenance"]["dataset_sha256"],
    )
    equivalence_records: list[dict[str, Any]] = []
    benchmark_records: list[Mapping[str, Any]] = []
    total_valid_rows = int(
        sum(manifest["splits"][split]["n_rows"] for split in SPLITS)
    )

    with np.load(dataset_path, allow_pickle=False) as archive:
        book = archive["book"]
        mid_z = archive["mid_z"]
        stock_ids = archive["stock_ids"].astype(np.int64)
        checkpoints = manifest["canonical_checkpoints"]
        for tag, checkpoint in sorted(checkpoints.items()):
            checkpoint_path = Path(checkpoint["path"])
            if sha256_file(checkpoint_path) != checkpoint["sha256"]:
                raise ExperimentIntegrityError(
                    f"checkpoint {tag} SHA-256 mismatch"
                )
            encode, stats = load_encoder(
                checkpoint["arm"], str(checkpoint_path), device
            )
            legacy_dump_record = legacy_manifest["readouts"].get(tag)
            if not isinstance(legacy_dump_record, Mapping):
                raise ExperimentIntegrityError(
                    f"legacy readout manifest lacks {tag}"
                )
            legacy_dump_path = legacy_root / legacy_dump_record["path"]
            if sha256_file(legacy_dump_path) != legacy_dump_record["file_sha256"]:
                raise ExperimentIntegrityError(
                    f"legacy readout {tag} SHA-256 mismatch"
                )
            with np.load(legacy_dump_path, allow_pickle=False) as expected:
                for split_name, endpoints, suffix in (
                    ("train", historical.train_t, "train"),
                    ("validation", historical.val_t, "val"),
                ):
                    dataset = RawWindowDataset(
                        book,
                        mid_z,
                        stock_ids,
                        endpoints,
                        stats,
                        20,
                    )
                    last, mean = _extract_arrays(
                        encode,
                        dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        device=device,
                    )
                    for readout, observed, key in (
                        (
                            "last_concat512",
                            last,
                            f"last_concat512_{suffix}",
                        ),
                        (
                            "meanK_concatS",
                            mean,
                            f"tmean_concat512_{suffix}",
                        ),
                    ):
                        comparison = _comparison(
                            observed,
                            np.asarray(expected[key]),
                            rtol=rtol,
                            atol=atol,
                        )
                        equivalence_records.append(
                            {
                                "checkpoint": tag,
                                "branch": checkpoint["arm"],
                                "encoder_seed": int(checkpoint["seed"]),
                                "split": split_name,
                                "readout": readout,
                                "n_rows": len(endpoints),
                                **comparison,
                            }
                        )
                    del last, mean

            if tag == benchmark_checkpoint:
                all_rows = pd.concat(
                    [
                        pd.read_parquet(
                            root / manifest["splits"][split]["path"],
                            columns=[
                                "stock_id",
                                "trading_date",
                                "endpoint_index",
                            ],
                        )
                        for split in SPLITS
                    ],
                    ignore_index=True,
                ).sort_values("endpoint_index")
                first = all_rows.iloc[0]
                day_mask = (
                    (all_rows["stock_id"] == first["stock_id"])
                    & (all_rows["trading_date"] == first["trading_date"])
                )
                stock_mask = all_rows["stock_id"] == first["stock_id"]
                for label, endpoints in (
                    (
                        "one_stock_day",
                        all_rows.loc[day_mask, "endpoint_index"].to_numpy(
                            dtype=np.int64
                        ),
                    ),
                    (
                        "one_stock",
                        all_rows.loc[stock_mask, "endpoint_index"].to_numpy(
                            dtype=np.int64
                        ),
                    ),
                ):
                    benchmark_records.append(
                        _benchmark(
                            encode,
                            RawWindowDataset(
                                book,
                                mid_z,
                                stock_ids,
                                endpoints,
                                stats,
                                20,
                            ),
                            label=label,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            device=device,
                        )
                    )
            del encode
            if device.type == "cuda":
                torch.cuda.empty_cache()

    equivalence_passed = all(
        record["passed"] for record in equivalence_records
    )
    benchmark_passed = (
        {record["label"] for record in benchmark_records}
        == {"one_stock_day", "one_stock"}
        and all(
            record["n_rows"] > 0
            and record["runtime_seconds"] > 0
            and record["rows_per_second"] > 0
            for record in benchmark_records
        )
    )
    stock_rate = next(
        (
            float(record["rows_per_second"])
            for record in benchmark_records
            if record["label"] == "one_stock"
        ),
        math.nan,
    )
    projected_seconds = (
        total_valid_rows * len(manifest["canonical_checkpoints"]) / stock_rate
        if np.isfinite(stock_rate) and stock_rate > 0
        else None
    )
    report: dict[str, Any] = {
        "schema_name": PREEXTRACT_SCHEMA,
        "schema_version": PREEXTRACT_SCHEMA_VERSION,
        "passed": equivalence_passed and benchmark_passed,
        "fail_closed": True,
        "storage_estimate_sha256": storage_record["sha256"],
        "dataset_sha256": manifest["provenance"]["dataset_sha256"],
        "legacy_split_fingerprint": historical.split_fingerprint,
        "numeric_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "feature_equivalence": {
            "passed": equivalence_passed,
            "scope": (
                "all nine canonical checkpoints, both readouts, and every "
                "endpoint in the historical 100k train/50k held-out dumps"
            ),
            "records": equivalence_records,
        },
        "benchmarks": {
            "passed": benchmark_passed,
            "checkpoint": benchmark_checkpoint,
            "records": benchmark_records,
            "projected_full_extraction_seconds": projected_seconds,
            "projected_full_extraction_hours": (
                projected_seconds / 3600.0
                if projected_seconds is not None
                else None
            ),
            "projection_basis": (
                "one-stock measured throughput × total valid rows × 9 "
                "encoders; both readouts are emitted in the same pass"
            ),
        },
    }
    report["gate_fingerprint"] = canonical_json_sha256(report)
    atomic_write_json(report_path, report)
    manifest["pre_extraction"]["benchmark_and_feature_equivalence"] = {
        "path": report_path.name,
        "sha256": sha256_file(report_path),
        "size_bytes": report_path.stat().st_size,
        "fingerprint": report["gate_fingerprint"],
        "passed": report["passed"],
    }
    atomic_write_json(root / "manifest.json", manifest)
    if not report["passed"]:
        raise ExperimentIntegrityError(
            f"pre-extraction gate failed; see {report_path}"
        )
    return report


def _feature_array_record(
    rows: pd.DataFrame,
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "storage": SHARDED_STORAGE,
        "shape": [len(rows), 512],
        "dtype": "float32",
        "row_key_sha256": sha256_array(
            rows["row_key"].astype(str).to_numpy(dtype="U")
        ),
        "shards": shards,
    }
    record["shard_manifest_sha256"] = canonical_json_sha256(
        sharded_record_fingerprint_payload(record)
    )
    return record


def _load_or_initialize_state(
    root: Path,
    manifest: Mapping[str, Any],
    preextract_sha256: str,
) -> dict[str, Any]:
    path = root / "feature_extraction_state.json"
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            state.get("schema_name") != EXTRACTION_STATE_SCHEMA
            or state.get("schema_version") != EXTRACTION_STATE_VERSION
            or state.get("dataset_sha256")
            != manifest["provenance"]["dataset_sha256"]
            or state.get("pre_extraction_report_sha256") != preextract_sha256
        ):
            raise ExperimentIntegrityError(
                "existing feature extraction state is incompatible"
            )
        return state
    return {
        "schema_name": EXTRACTION_STATE_SCHEMA,
        "schema_version": EXTRACTION_STATE_VERSION,
        "dataset_sha256": manifest["provenance"]["dataset_sha256"],
        "pre_extraction_report_sha256": preextract_sha256,
        "completed_shards": {},
    }


def _valid_existing_shard(
    root: Path,
    state_record: Mapping[str, Any] | None,
    *,
    relative: Path,
    expected_shape: tuple[int, int],
    expected_row_hash: str,
) -> bool:
    path = root / relative
    if not isinstance(state_record, Mapping) or not path.is_file():
        return False
    return (
        state_record.get("path") == str(relative)
        and state_record.get("shape") == list(expected_shape)
        and state_record.get("dtype") == "float32"
        and state_record.get("row_key_sha256") == expected_row_hash
        and state_record.get("size_bytes") == path.stat().st_size
        and state_record.get("sha256") == sha256_file(path)
    )


def extract_full_features(
    bundle_dir: str | Path,
    *,
    device_name: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 2,
    checkpoint_tags: Iterable[str] | None = None,
) -> Mapping[str, Any]:
    """Extract checkpoint-by-checkpoint and persist only bounded row shards."""
    root = Path(bundle_dir).resolve()
    manifest = _load_manifest(root)
    gate = manifest.get("pre_extraction", {}).get(
        "benchmark_and_feature_equivalence"
    )
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise ExperimentIntegrityError(
            "benchmark/feature-equivalence gate must pass before full extraction"
        )
    gate_path = root / str(gate.get("path", ""))
    if not gate_path.is_file() or sha256_file(gate_path) != gate.get("sha256"):
        raise ExperimentIntegrityError("pre-extraction gate report hash mismatch")
    device = _device(device_name)
    checkpoints = manifest["canonical_checkpoints"]
    requested = (
        sorted(checkpoints)
        if checkpoint_tags is None
        else list(checkpoint_tags)
    )
    if not requested or any(tag not in checkpoints for tag in requested):
        raise ValueError("checkpoint_tags contains an unknown or empty selection")
    state_path = root / "feature_extraction_state.json"
    state = _load_or_initialize_state(root, manifest, gate["sha256"])
    rows = {
        split: pd.read_parquet(root / manifest["splits"][split]["path"])
        for split in SPLITS
    }
    shard_rows = int(manifest["shard_rows"])
    dataset_path = Path(manifest["provenance"]["dataset_path"])
    existing_features = {
        (
            record["branch"],
            int(record["encoder_seed"]),
            record["readout"],
        ): record
        for record in manifest.get("feature_sets", [])
    }
    manifest["status"] = "extracting"
    atomic_write_json(root / "manifest.json", manifest)

    with np.load(dataset_path, allow_pickle=False) as archive:
        book = archive["book"]
        mid_z = archive["mid_z"]
        stock_ids = archive["stock_ids"].astype(np.int64)
        for tag in requested:
            checkpoint = checkpoints[tag]
            checkpoint_path = Path(checkpoint["path"])
            if sha256_file(checkpoint_path) != checkpoint["sha256"]:
                raise ExperimentIntegrityError(
                    f"checkpoint {tag} SHA-256 mismatch"
                )
            encode, stats = load_encoder(
                checkpoint["arm"], str(checkpoint_path), device
            )
            split_shards: dict[str, dict[str, list[dict[str, Any]]]] = {
                readout: {split: [] for split in SPLITS}
                for readout in READOUTS
            }
            for split in SPLITS:
                frame = rows[split]
                for shard_index, start in enumerate(
                    range(0, len(frame), shard_rows)
                ):
                    stop = min(start + shard_rows, len(frame))
                    endpoints = frame["endpoint_index"].iloc[
                        start:stop
                    ].to_numpy(dtype=np.int64)
                    row_hash = sha256_array(
                        frame["row_key"].iloc[start:stop].astype(str).to_numpy(
                            dtype="U"
                        )
                    )
                    relative_by_readout = {
                        readout: (
                            Path("features")
                            / checkpoint["arm"]
                            / f"seed{int(checkpoint['seed'])}"
                            / readout
                            / split
                            / f"part-{shard_index:05d}.npy"
                        )
                        for readout in READOUTS
                    }
                    state_records = {
                        readout: state["completed_shards"].get(
                            str(relative_by_readout[readout])
                        )
                        for readout in READOUTS
                    }
                    reusable = all(
                        _valid_existing_shard(
                            root,
                            state_records[readout],
                            relative=relative_by_readout[readout],
                            expected_shape=(stop - start, 512),
                            expected_row_hash=row_hash,
                        )
                        for readout in READOUTS
                    )
                    if not reusable:
                        for readout, relative in relative_by_readout.items():
                            path = root / relative
                            if path.exists() and not _valid_existing_shard(
                                root,
                                state_records[readout],
                                relative=relative,
                                expected_shape=(stop - start, 512),
                                expected_row_hash=row_hash,
                            ):
                                raise ExperimentIntegrityError(
                                    f"unverified existing feature shard {path}"
                                )
                        last, mean = _extract_arrays(
                            encode,
                            RawWindowDataset(
                                book,
                                mid_z,
                                stock_ids,
                                endpoints,
                                stats,
                                20,
                            ),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            device=device,
                        )
                        for readout, values in (
                            ("last_concat512", last),
                            ("meanK_concatS", mean),
                        ):
                            relative = relative_by_readout[readout]
                            path = root / relative
                            atomic_save_npy(path, values)
                            state_record = {
                                "path": str(relative),
                                "sha256": sha256_file(path),
                                "size_bytes": path.stat().st_size,
                                "shape": list(values.shape),
                                "dtype": values.dtype.name,
                                "row_start": start,
                                "row_stop": stop,
                                "row_key_sha256": row_hash,
                            }
                            state["completed_shards"][
                                str(relative)
                            ] = state_record
                        atomic_write_json(state_path, state)
                        del last, mean
                    for readout in READOUTS:
                        record = dict(
                            state["completed_shards"][
                                str(relative_by_readout[readout])
                            ]
                        )
                        split_shards[readout][split].append(record)

            for readout in READOUTS:
                arrays = {
                    split: _feature_array_record(
                        rows[split], split_shards[readout][split]
                    )
                    for split in SPLITS
                }
                record = {
                    "branch": checkpoint["arm"],
                    "encoder_seed": int(checkpoint["seed"]),
                    "readout": readout,
                    "dimension": 512,
                    "dtype": "float32",
                    "checkpoint_tag": tag,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "arrays": arrays,
                }
                existing_features[
                    (
                        checkpoint["arm"],
                        int(checkpoint["seed"]),
                        readout,
                    )
                ] = record
            manifest["feature_sets"] = [
                existing_features[key] for key in sorted(existing_features)
            ]
            atomic_write_json(root / "manifest.json", manifest)
            del encode
            if device.type == "cuda":
                torch.cuda.empty_cache()

    expected = {
        (branch, seed, readout)
        for branch in BRANCHES
        for seed in manifest["canonical_encoder_seeds"][branch]
        for readout in READOUTS
    }
    actual = set(existing_features)
    manifest["status"] = "complete" if actual == expected else "extracting"
    manifest["feature_inventory_sha256"] = canonical_json_sha256(
        {"feature_sets": manifest["feature_sets"]}
    )
    atomic_write_json(root / "manifest.json", manifest)
    return {
        "status": manifest["status"],
        "completed_feature_sets": len(actual),
        "expected_feature_sets": len(expected),
        "missing": sorted(expected - actual),
    }

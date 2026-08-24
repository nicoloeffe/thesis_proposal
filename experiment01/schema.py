"""Fail-closed input contract for the preregistered three-way bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .constants import (
    BRANCHES,
    INPUT_SCHEMA,
    INPUT_SCHEMA_VERSION,
    READOUTS,
    SPLITS,
)
from .errors import ExperimentIntegrityError
from .io import canonical_json_sha256, sha256_array, sha256_file
from .sharded import (
    SHARDED_STORAGE,
    ArrayShard,
    ShardedArray,
    sharded_record_fingerprint_payload,
)


ROW_COLUMNS = (
    "row_key",
    "stock_id",
    "stock_symbol",
    "stock_day_id",
    "trading_date",
    "endpoint_index",
    "endpoint_order",
    "timestamp_ns",
)

READOUT_DEFINITIONS = {
    "last_concat512": "grid[:, -1, :, :].reshape(B, S * d_model)",
    "meanK_concatS": "grid.mean(axis=1).reshape(B, S * d_model)",
}


def _fail(message: str) -> None:
    raise ExperimentIntegrityError(f"Experiment 01 input integrity failure: {message}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a list")
    return value


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{label}.path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{label}.path must remain inside the bundle")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        _fail(f"{label}.path escapes the bundle")
    return resolved


def stable_row_keys(frame: pd.DataFrame) -> np.ndarray:
    """Return row keys as canonical unicode values in the manifest row order."""
    return frame["row_key"].astype(str).to_numpy(dtype="U", copy=True)


def _validate_recorded_file(
    root: Path, record: Mapping[str, Any], label: str, verify_hashes: bool
) -> Path:
    path = _safe_relative(root, record.get("path"), label)
    if not path.is_file():
        _fail(f"{label} is missing: {path}")
    expected_size = record.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size != path.stat().st_size:
        _fail(f"{label} size differs from the manifest")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        _fail(f"{label} has no valid SHA-256")
    if verify_hashes and sha256_file(path) != expected_hash:
        _fail(f"{label} SHA-256 differs from the manifest")
    return path


def _validate_json_gate(
    root: Path,
    record: Mapping[str, Any],
    label: str,
    *,
    schema_name: str,
    verify_hashes: bool,
) -> Mapping[str, Any]:
    if record.get("passed") is not True:
        _fail(f"{label} did not pass")
    path = _validate_recorded_file(root, record, label, verify_hashes)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {exc}")
    if payload.get("schema_name") != schema_name:
        _fail(f"{label} schema differs")
    if payload.get("passed") is not True:
        _fail(f"{label} payload did not pass")
    return payload


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    block: str
    independent: bool
    redundant_with: tuple[str, ...]
    semantics: str | None


@dataclass(frozen=True)
class FeatureSet:
    branch: str
    encoder_seed: int
    readout: str
    dimension: int
    dtype: np.dtype
    paths: Mapping[str, Path | ShardedArray]

    @property
    def key(self) -> tuple[str, int, str]:
        return self.branch, self.encoder_seed, self.readout


@dataclass
class InputBundle:
    root: Path
    manifest: Mapping[str, Any]
    rows: Mapping[str, pd.DataFrame]
    target_paths: Mapping[str, Path | ShardedArray]
    target_definitions: tuple[TargetDefinition, ...]
    feature_sets: tuple[FeatureSet, ...]

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.target_definitions)

    @property
    def target_blocks(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(target.block for target in self.target_definitions))

    @property
    def encoder_seeds(self) -> Mapping[str, tuple[int, ...]]:
        return {
            branch: tuple(
                int(seed)
                for seed in self.manifest["canonical_encoder_seeds"][branch]
            )
            for branch in BRANCHES
        }

    def load_targets(
        self, split: str, mmap_mode: str | None = "r"
    ) -> np.ndarray | ShardedArray:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        source = self.target_paths[split]
        if isinstance(source, ShardedArray):
            return source
        return np.load(source, mmap_mode=mmap_mode, allow_pickle=False)

    def load_features(
        self, feature_set: FeatureSet, split: str, mmap_mode: str | None = "r"
    ) -> np.ndarray | ShardedArray:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        source = feature_set.paths[split]
        if isinstance(source, ShardedArray):
            return source
        return np.load(source, mmap_mode=mmap_mode, allow_pickle=False)

    def feature_set(self, branch: str, seed: int, readout: str) -> FeatureSet:
        matches = [
            item
            for item in self.feature_sets
            if item.key == (branch, int(seed), readout)
        ]
        if len(matches) != 1:
            raise KeyError((branch, seed, readout))
        return matches[0]


def _validate_rows(
    root: Path,
    split_records: Mapping[str, Any],
    verify_hashes: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    rows: dict[str, pd.DataFrame] = {}
    key_hashes: dict[str, str] = {}
    endpoint_seen = np.zeros(0, dtype=bool)
    timestamps_by_stock: dict[int, list[np.ndarray]] = {}
    symbol_by_stock: dict[int, str] = {}
    date_by_stock_day_id: dict[tuple[int, int], str] = {}
    split_groups: dict[str, set[tuple[int, str]]] = {}
    for split in SPLITS:
        record = _mapping(split_records.get(split), f"splits.{split}")
        if record.get("complete_stock_days") is not True:
            _fail(f"splits.{split} is not declared to contain complete stock-days")
        path = _validate_recorded_file(
            root, record, f"splits.{split}.rows", verify_hashes
        )
        if path.suffix != ".parquet":
            _fail(f"splits.{split}.rows must be Parquet")
        frame = pd.read_parquet(path)
        missing = set(ROW_COLUMNS) - set(frame.columns)
        if missing:
            _fail(f"splits.{split}.rows is missing columns {sorted(missing)}")
        frame = frame.loc[:, ROW_COLUMNS].copy()
        if len(frame) != int(record.get("n_rows", -1)):
            _fail(f"splits.{split}.n_rows differs from the row table")
        if len(frame) == 0:
            _fail(f"splits.{split} is empty")
        if frame.isna().any().any():
            _fail(f"splits.{split}.rows contains missing values")
        keys = frame["row_key"].astype(str)
        if keys.duplicated().any():
            _fail(f"splits.{split}.rows contains duplicate row keys")
        key_hash = sha256_array(keys.to_numpy(dtype="U"))
        if record.get("row_key_sha256") != key_hash:
            _fail(f"splits.{split}.row_key_sha256 mismatch")
        key_hashes[split] = key_hash

        for column in ("stock_id", "stock_day_id", "endpoint_index", "endpoint_order"):
            if not pd.api.types.is_integer_dtype(frame[column]):
                _fail(f"splits.{split}.{column} must have an integer dtype")
        if not pd.api.types.is_integer_dtype(frame["timestamp_ns"]):
            _fail(f"splits.{split}.timestamp_ns must be integer nanoseconds")
        if frame["endpoint_index"].duplicated().any():
            _fail(f"splits.{split} contains duplicate raw endpoint indices")
        if frame.duplicated(["stock_id", "timestamp_ns"]).any():
            _fail(
                f"splits.{split} contains duplicate endpoint timestamps "
                "within a stock"
            )
        endpoints = frame["endpoint_index"].to_numpy(dtype=np.int64)
        if np.any(endpoints < 0):
            _fail(f"splits.{split} contains negative raw endpoint indices")
        largest_endpoint = int(endpoints.max())
        if largest_endpoint >= len(endpoint_seen):
            grown = np.zeros(largest_endpoint + 1, dtype=bool)
            grown[: len(endpoint_seen)] = endpoint_seen
            endpoint_seen = grown
        if endpoint_seen[endpoints].any():
            example = int(endpoints[endpoint_seen[endpoints]][0])
            _fail(
                f"raw endpoint indices overlap across splits "
                f"(example: {example})"
            )
        endpoint_seen[endpoints] = True
        for stock, stock_frame in frame.groupby(
            "stock_id", sort=False, observed=True
        ):
            timestamps_by_stock.setdefault(int(stock), []).append(
                stock_frame["timestamp_ns"].to_numpy(dtype=np.int64)
            )

        parsed_dates = pd.to_datetime(frame["trading_date"], errors="coerce")
        if parsed_dates.isna().any():
            _fail(f"splits.{split}.trading_date contains invalid dates")
        if (parsed_dates.dt.strftime("%Y-%m-%d") != frame["trading_date"]).any():
            _fail(f"splits.{split}.trading_date is not canonical YYYY-MM-DD")
        groups: set[tuple[int, str]] = set()
        for (stock, day), group in frame.groupby(
            ["stock_id", "trading_date"], sort=False, observed=True
        ):
            order = group["endpoint_order"].to_numpy(dtype=np.int64)
            if not np.array_equal(order, np.arange(len(order), dtype=np.int64)):
                _fail(
                    f"{split} stock-day ({stock}, {day}) is not in complete "
                    "canonical endpoint order 0..n-1"
                )
            endpoint = group["endpoint_index"].to_numpy(dtype=np.int64)
            timestamp = group["timestamp_ns"].to_numpy(dtype=np.int64)
            if np.any(endpoint[1:] <= endpoint[:-1]):
                _fail(f"{split} stock-day ({stock}, {day}) endpoint order is invalid")
            if np.any(timestamp[1:] <= timestamp[:-1]):
                _fail(f"{split} stock-day ({stock}, {day}) timestamp order is invalid")
            if group["stock_symbol"].nunique() != 1:
                _fail(f"{split} stock-day ({stock}, {day}) has multiple symbols")
            if group["stock_day_id"].nunique() != 1:
                _fail(f"{split} stock-day ({stock}, {day}) has multiple day IDs")
            symbol = str(group["stock_symbol"].iloc[0])
            previous_symbol = symbol_by_stock.setdefault(int(stock), symbol)
            if previous_symbol != symbol:
                _fail(f"stock_id {stock} maps to multiple stock symbols")
            stock_day_id = int(group["stock_day_id"].iloc[0])
            day_key = (int(stock), stock_day_id)
            previous_date = date_by_stock_day_id.setdefault(day_key, str(day))
            if previous_date != str(day):
                _fail(
                    f"stock/day ID {day_key} maps to multiple trading dates"
                )
            expected_keys = (
                group["stock_id"].astype(str)
                + "|"
                + group["trading_date"].astype(str)
                + "|"
                + group["endpoint_order"].astype(str)
            )
            if not np.array_equal(
                expected_keys.to_numpy(dtype="U"),
                group["row_key"].astype(str).to_numpy(dtype="U"),
            ):
                _fail(
                    f"{split} stock-day ({stock}, {day}) row keys do not "
                    "encode canonical stock/date/order identity"
                )
            groups.add((int(stock), str(day)))
        if record.get("n_stock_days") != len(groups):
            _fail(f"splits.{split}.n_stock_days differs from the row table")
        if record.get("first_trading_date") != str(frame["trading_date"].min()):
            _fail(f"splits.{split}.first_trading_date differs from the row table")
        if record.get("last_trading_date") != str(frame["trading_date"].max()):
            _fail(f"splits.{split}.last_trading_date differs from the row table")
        actual_counts_by_stock = []
        for stock_id, stock_group in frame.groupby(
            "stock_id", sort=True, observed=True
        ):
            actual_counts_by_stock.append(
                {
                    "stock_id": int(stock_id),
                    "stock_symbol": str(stock_group["stock_symbol"].iloc[0]),
                    "n_stock_days": int(
                        stock_group["trading_date"].nunique()
                    ),
                    "n_endpoints": len(stock_group),
                    "first_trading_date": str(
                        stock_group["trading_date"].min()
                    ),
                    "last_trading_date": str(
                        stock_group["trading_date"].max()
                    ),
                }
            )
        if record.get("counts_by_stock") != actual_counts_by_stock:
            _fail(f"splits.{split}.counts_by_stock differs from the row table")
        split_groups[split] = groups
        rows[split] = frame

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_groups[left] & split_groups[right]
            if overlap:
                _fail(
                    f"stock-day overlap between {left} and {right}: "
                    f"{next(iter(overlap))}"
                )
    for stock, timestamp_parts in timestamps_by_stock.items():
        timestamps = np.concatenate(timestamp_parts)
        timestamps.sort()
        duplicate = timestamps[1:] == timestamps[:-1]
        if duplicate.any():
            example = int(timestamps[1:][duplicate][0])
            _fail(
                f"stock/timestamp identities overlap across splits "
                f"(example: ({stock}, {example}))"
            )
    # Canonical row keys are (stock_id, trading_date, endpoint_order). Their
    # cross-split disjointness follows from the complete, disjoint stock-days
    # checked above, without materialising millions of Python strings in a set.
    return rows, key_hashes


def _validate_array_source(
    root: Path,
    record: Mapping[str, Any],
    label: str,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: tuple[np.dtype, ...],
    row_keys: np.ndarray,
    expected_row_key_sha256: str,
    shard_row_hash_cache: dict[tuple[int, int], str],
    verify_hashes: bool,
    check_finite: bool,
) -> tuple[Path | ShardedArray, np.dtype]:
    """Validate one monolithic or sharded array without weakening identity checks."""
    if record.get("row_key_sha256") != expected_row_key_sha256:
        _fail(f"{label} row identity mismatch")
    if record.get("storage") == SHARDED_STORAGE:
        if record.get("shape") != list(expected_shape):
            _fail(f"{label}.shape differs from expected")
        try:
            dtype = np.dtype(record.get("dtype"))
        except TypeError:
            _fail(f"{label}.dtype is invalid")
        if dtype not in expected_dtype:
            _fail(f"{label} has invalid dtype {dtype}")
        shard_records = _list(record.get("shards"), f"{label}.shards")
        if not shard_records:
            _fail(f"{label}.shards is empty")
        expected_fingerprint = canonical_json_sha256(
            sharded_record_fingerprint_payload(record)
        )
        if record.get("shard_manifest_sha256") != expected_fingerprint:
            _fail(f"{label}.shard_manifest_sha256 mismatch")
        shards: list[ArrayShard] = []
        cursor = 0
        for index, raw in enumerate(shard_records):
            shard_record = _mapping(raw, f"{label}.shards[{index}]")
            start = shard_record.get("row_start")
            stop = shard_record.get("row_stop")
            if (
                not isinstance(start, int)
                or not isinstance(stop, int)
                or start != cursor
                or stop <= start
                or stop > expected_shape[0]
            ):
                _fail(f"{label}.shards[{index}] row interval is invalid")
            path = _validate_recorded_file(
                root,
                shard_record,
                f"{label}.shards[{index}]",
                verify_hashes,
            )
            if path.suffix != ".npy":
                _fail(f"{label}.shards[{index}] must be an NPY file")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shard_shape = (stop - start, *expected_shape[1:])
            if array.shape != shard_shape or array.dtype != dtype:
                _fail(
                    f"{label}.shards[{index}] has {array.shape}/{array.dtype}; "
                    f"expected {shard_shape}/{dtype}"
                )
            if shard_record.get("shape") != list(shard_shape):
                _fail(f"{label}.shards[{index}] shape metadata mismatch")
            if shard_record.get("dtype") != dtype.name:
                _fail(f"{label}.shards[{index}] dtype metadata mismatch")
            interval = (start, stop)
            expected_row_hash = shard_row_hash_cache.get(interval)
            if expected_row_hash is None:
                expected_row_hash = sha256_array(row_keys[start:stop])
                shard_row_hash_cache[interval] = expected_row_hash
            if shard_record.get("row_key_sha256") != expected_row_hash:
                _fail(f"{label}.shards[{index}] row identity mismatch")
            if check_finite and not _all_finite(array):
                _fail(f"{label}.shards[{index}] contains NaN or infinity")
            shards.append(ArrayShard(path, start, stop))
            cursor = stop
        if cursor != expected_shape[0]:
            _fail(f"{label}.shards do not cover all rows")
        return ShardedArray(shards, expected_shape, dtype), dtype

    path = _validate_recorded_file(root, record, label, verify_hashes)
    if path.suffix != ".npy":
        _fail(f"{label} must be an NPY file")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != expected_shape:
        _fail(f"{label} has shape {array.shape}, expected {expected_shape}")
    if array.dtype not in expected_dtype:
        _fail(f"{label} has invalid dtype {array.dtype}")
    if record.get("shape") != list(expected_shape):
        _fail(f"{label}.shape differs from the file")
    if record.get("dtype") != array.dtype.name:
        _fail(f"{label}.dtype differs from the file")
    if check_finite and not _all_finite(array):
        _fail(f"{label} contains NaN or infinity")
    return path, array.dtype


def _validate_targets(
    root: Path,
    manifest: Mapping[str, Any],
    rows: Mapping[str, pd.DataFrame],
    row_key_hashes: Mapping[str, str],
    verify_hashes: bool,
    check_finite: bool,
) -> tuple[dict[str, Path | ShardedArray], tuple[TargetDefinition, ...]]:
    target_section = _mapping(manifest.get("targets"), "targets")
    target_gate = _mapping(
        target_section.get("equivalence_report"),
        "targets.equivalence_report",
    )
    _validate_json_gate(
        root,
        target_gate,
        "targets.equivalence_report",
        schema_name="thesis.experiment01.target_equivalence",
        verify_hashes=verify_hashes,
    )
    definitions_raw = _list(target_section.get("definitions"), "targets.definitions")
    definitions: list[TargetDefinition] = []
    names: set[str] = set()
    allowed_blocks = {"directional", "volatility", "timing"}
    for index, raw in enumerate(definitions_raw):
        record = _mapping(raw, f"targets.definitions[{index}]")
        name = record.get("name")
        block = record.get("block")
        independent = record.get("independent")
        redundant = record.get("redundant_with", [])
        semantics = record.get("semantics")
        if not isinstance(name, str) or not name or name in names:
            _fail(f"target name {name!r} is missing or duplicated")
        if block not in allowed_blocks:
            _fail(f"target {name!r} has invalid block {block!r}")
        if not isinstance(independent, bool):
            _fail(f"target {name!r} has no boolean independent flag")
        if not isinstance(redundant, list) or not all(
            isinstance(value, str) for value in redundant
        ):
            _fail(f"target {name!r}.redundant_with must be a string list")
        if semantics is not None and not isinstance(semantics, str):
            _fail(f"target {name!r}.semantics must be a string or null")
        definitions.append(
            TargetDefinition(
                name=name,
                block=block,
                independent=independent,
                redundant_with=tuple(redundant),
                semantics=semantics,
            )
        )
        names.add(name)
    if not definitions:
        _fail("targets.definitions is empty")
    for block in allowed_blocks:
        if not any(target.block == block for target in definitions):
            _fail(f"target block {block!r} is absent")
    if (
        sum(
            target.block == "directional" and target.independent
            for target in definitions
        )
        < 2
    ):
        _fail("fewer than two independent directional targets are declared")
    timing = [target for target in definitions if target.block == "timing"]
    if len(timing) != 1:
        _fail("exactly one canonical timing target is required")
    if timing[0].semantics != "log1p_observed_or_capped_all_rows:max_look=600":
        _fail("timing target does not preserve the canonical capped semantics")

    arrays = _mapping(target_section.get("arrays"), "targets.arrays")
    paths: dict[str, Path | ShardedArray] = {}
    for split in SPLITS:
        record = _mapping(arrays.get(split), f"targets.arrays.{split}")
        expected_shape = (len(rows[split]), len(definitions))
        source, _ = _validate_array_source(
            root,
            record,
            f"targets.arrays.{split}",
            expected_shape=expected_shape,
            expected_dtype=(np.dtype("float32"), np.dtype("float64")),
            row_keys=stable_row_keys(rows[split]),
            expected_row_key_sha256=row_key_hashes[split],
            shard_row_hash_cache={},
            verify_hashes=verify_hashes,
            check_finite=check_finite,
        )
        paths[split] = source
    return paths, tuple(definitions)


def _all_finite(array: np.ndarray, chunk_rows: int = 65536) -> bool:
    for start in range(0, len(array), chunk_rows):
        if not np.isfinite(np.asarray(array[start : start + chunk_rows])).all():
            return False
    return True


def _validate_features(
    root: Path,
    manifest: Mapping[str, Any],
    rows: Mapping[str, pd.DataFrame],
    row_key_hashes: Mapping[str, str],
    verify_hashes: bool,
    check_finite: bool,
) -> tuple[FeatureSet, ...]:
    definitions = _mapping(manifest.get("readout_definitions"), "readout_definitions")
    if dict(definitions) != READOUT_DEFINITIONS:
        _fail("readout definitions differ from the frozen canonical definitions")
    canonical_seeds = _mapping(
        manifest.get("canonical_encoder_seeds"), "canonical_encoder_seeds"
    )
    checkpoint_records = _mapping(
        manifest.get("canonical_checkpoints"), "canonical_checkpoints"
    )
    checkpoint_by_key: dict[tuple[str, int], tuple[str, str]] = {}
    for tag, raw_checkpoint in checkpoint_records.items():
        checkpoint = _mapping(
            raw_checkpoint, f"canonical_checkpoints.{tag}"
        )
        branch = checkpoint.get("arm")
        seed = checkpoint.get("seed")
        epoch = checkpoint.get("epoch")
        digest = checkpoint.get("sha256")
        path_value = checkpoint.get("path")
        if (
            branch not in BRANCHES
            or not isinstance(seed, int)
            or epoch != 20
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(path_value, str)
            or not path_value
        ):
            _fail(f"canonical checkpoint {tag!r} metadata is invalid")
        key = (str(branch), int(seed))
        if key in checkpoint_by_key:
            _fail(f"duplicate canonical checkpoint key {key!r}")
        checkpoint_path = Path(path_value)
        if not checkpoint_path.is_file():
            _fail(f"canonical checkpoint {tag!r} is missing")
        if verify_hashes and sha256_file(checkpoint_path) != digest:
            _fail(f"canonical checkpoint {tag!r} SHA-256 mismatch")
        checkpoint_by_key[key] = (str(tag), digest)
    expected_keys: set[tuple[str, int, str]] = set()
    for branch in BRANCHES:
        raw_seeds = canonical_seeds.get(branch)
        if not isinstance(raw_seeds, list) or not raw_seeds:
            _fail(f"canonical_encoder_seeds.{branch} is missing or empty")
        if any(not isinstance(seed, int) for seed in raw_seeds):
            _fail(f"canonical_encoder_seeds.{branch} must contain integers")
        if len(set(raw_seeds)) != len(raw_seeds):
            _fail(f"canonical_encoder_seeds.{branch} contains duplicates")
        for seed in raw_seeds:
            for readout in READOUTS:
                expected_keys.add((branch, int(seed), readout))

    feature_records = _list(manifest.get("feature_sets"), "feature_sets")
    row_key_arrays = {
        split: stable_row_keys(rows[split]) for split in SPLITS
    }
    shard_row_hash_caches: dict[
        str, dict[tuple[int, int], str]
    ] = {split: {} for split in SPLITS}
    feature_sets: list[FeatureSet] = []
    seen: set[tuple[str, int, str]] = set()
    for index, raw in enumerate(feature_records):
        record = _mapping(raw, f"feature_sets[{index}]")
        branch = record.get("branch")
        seed = record.get("encoder_seed")
        readout = record.get("readout")
        key = (branch, seed, readout)
        if key not in expected_keys:
            _fail(f"feature_sets[{index}] has non-canonical key {key!r}")
        if key in seen:
            _fail(f"duplicate feature set {key!r}")
        checkpoint_key = (str(branch), int(seed))
        expected_checkpoint = checkpoint_by_key.get(checkpoint_key)
        if expected_checkpoint is None:
            _fail(f"feature set {key!r} has no canonical checkpoint")
        if (
            record.get("checkpoint_tag") != expected_checkpoint[0]
            or record.get("checkpoint_sha256") != expected_checkpoint[1]
        ):
            _fail(f"feature set {key!r} checkpoint provenance mismatch")
        seen.add(key)
        dimension = record.get("dimension")
        if not isinstance(dimension, int) or dimension <= 0:
            _fail(f"feature set {key!r} has invalid dimension")
        if dimension != 512:
            _fail(f"canonical readout {readout!r} must have dimension 512")
        try:
            dtype = np.dtype(record.get("dtype"))
        except TypeError:
            _fail(f"feature set {key!r} has invalid dtype")
        if dtype != np.dtype("float32"):
            _fail(f"feature set {key!r} must preserve canonical float32 dtype")
        array_records = _mapping(record.get("arrays"), f"feature set {key!r}.arrays")
        paths: dict[str, Path | ShardedArray] = {}
        for split in SPLITS:
            array_record = _mapping(
                array_records.get(split), f"feature set {key!r}.arrays.{split}"
            )
            expected_shape = (len(rows[split]), dimension)
            source, source_dtype = _validate_array_source(
                root,
                array_record,
                f"feature set {key!r}.arrays.{split}",
                expected_shape=expected_shape,
                expected_dtype=(dtype,),
                row_keys=row_key_arrays[split],
                expected_row_key_sha256=row_key_hashes[split],
                shard_row_hash_cache=shard_row_hash_caches[split],
                verify_hashes=verify_hashes,
                check_finite=check_finite,
            )
            if source_dtype != dtype:
                _fail(f"feature set {key!r}/{split} dtype mismatch")
            paths[split] = source
        feature_sets.append(
            FeatureSet(
                branch=str(branch),
                encoder_seed=int(seed),
                readout=str(readout),
                dimension=dimension,
                dtype=dtype,
                paths=paths,
            )
        )
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        extra = sorted(seen - expected_keys)
        _fail(f"feature inventory is not exact; missing={missing}, extra={extra}")
    if set(checkpoint_by_key) != {
        (branch, seed)
        for branch in BRANCHES
        for seed in canonical_seeds[branch]
    }:
        _fail("canonical checkpoint inventory is not exact")
    return tuple(sorted(feature_sets, key=lambda item: item.key))


def load_input_bundle(
    bundle_root: str | Path,
    *,
    verify_hashes: bool = True,
    check_finite: bool = True,
) -> InputBundle:
    """Validate and load a complete Experiment 01 input bundle.

    No fallback to the legacy two-way dump format is provided on purpose.
    """
    root = Path(bundle_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        _fail(f"manifest.json is missing from {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read manifest.json: {exc}")
    manifest = _mapping(manifest, "manifest")
    if manifest.get("schema_name") != INPUT_SCHEMA:
        _fail(f"schema_name must be {INPUT_SCHEMA!r}")
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        _fail(f"schema_version must be {INPUT_SCHEMA_VERSION}")
    if manifest.get("status") != "complete":
        _fail("bundle status is not complete")
    provenance = _mapping(manifest.get("provenance"), "provenance")
    if provenance.get("corrected_post_p0") is not True:
        _fail("provenance.corrected_post_p0 is not true")
    for field in (
        "source_commit",
        "dataset_path",
        "dataset_sha256",
        "split_protocol_fingerprint",
        "target_manifest_fingerprint",
    ):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            _fail(f"provenance.{field} is missing")
    if len(provenance["dataset_sha256"]) != 64:
        _fail("provenance.dataset_sha256 is not a SHA-256")
    if manifest.get("validation_is_fixed_and_complete") is not True:
        _fail("validation split is not declared fixed and complete")
    if manifest.get("test_is_fixed_and_complete") is not True:
        _fail("test split is not declared fixed and complete")
    if manifest.get("training_features_are_full_unlabelled_split") is not True:
        _fail("train features are not the full unlabelled canonical split")
    for flag in (
        "context_window_within_stock_day_verified",
        "target_horizon_within_stock_day_verified",
        "row_feature_target_alignment_verified",
    ):
        if manifest.get(flag) is not True:
            _fail(f"{flag} is not true")
    protocol = _mapping(manifest.get("protocol"), "protocol")
    if protocol.get("K") != 20 or protocol.get("max_horizon") != 20:
        _fail("protocol must preserve K=20 and max_horizon=20")
    if protocol.get("grouping") != "stock_id+trading_date":
        _fail("protocol grouping must be stock_id+trading_date")
    if protocol.get("test_hyperparameter_selection") != "forbidden":
        _fail("test hyperparameter selection is not explicitly forbidden")
    if manifest.get("historical_heldout_reuse_disclosed") is not True:
        _fail("historical held-out reuse is not disclosed")
    pre_extraction = _mapping(
        manifest.get("pre_extraction"), "pre_extraction"
    )
    for name, schema_name in (
        ("storage_estimate", "thesis.experiment01.storage_estimate"),
        (
            "benchmark_and_feature_equivalence",
            "thesis.experiment01.pre_extraction_gate",
        ),
    ):
        record = _mapping(pre_extraction.get(name), f"pre_extraction.{name}")
        _validate_json_gate(
            root,
            record,
            f"pre_extraction.{name}",
            schema_name=schema_name,
            verify_hashes=verify_hashes,
        )

    split_records = _mapping(manifest.get("splits"), "splits")
    if set(split_records) != set(SPLITS):
        _fail(f"splits must be exactly {list(SPLITS)}")
    rows, row_key_hashes = _validate_rows(root, split_records, verify_hashes)
    target_paths, target_definitions = _validate_targets(
        root,
        manifest,
        rows,
        row_key_hashes,
        verify_hashes,
        check_finite,
    )
    feature_sets = _validate_features(
        root,
        manifest,
        rows,
        row_key_hashes,
        verify_hashes,
        check_finite,
    )
    expected_inventory_hash = canonical_json_sha256(
        {"feature_sets": manifest.get("feature_sets")}
    )
    if manifest.get("feature_inventory_sha256") != expected_inventory_hash:
        _fail("feature inventory fingerprint mismatch")
    return InputBundle(
        root=root,
        manifest=manifest,
        rows=rows,
        target_paths=target_paths,
        target_definitions=target_definitions,
        feature_sets=feature_sets,
    )


def iter_target_indices(
    definitions: Iterable[TargetDefinition], block: str
) -> tuple[np.ndarray, np.ndarray]:
    values = list(definitions)
    all_indices = np.asarray(
        [index for index, target in enumerate(values) if target.block == block],
        dtype=np.int64,
    )
    independent = np.asarray(
        [
            index
            for index, target in enumerate(values)
            if target.block == block and target.independent
        ],
        dtype=np.int64,
    )
    return all_indices, independent

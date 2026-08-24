"""Canonical schemas, fingerprints and atomic I/O for stage-1 analysis artifacts.

This module is intentionally lightweight: it imports NumPy but not PyTorch.  It
is shared by readout extraction, Gate 1 and the regression tests so that a split
can never silently change meaning from "positions in valid_t" to raw endpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np


SPLIT_SCHEMA_NAME = SPLIT_SCHEMA = "thesis.stage1.split"
SPLIT_SCHEMA_VERSION = SPLIT_VERSION = 2
TARGET_SCHEMA_NAME = TARGET_SCHEMA = "thesis.stage1.targets"
TARGET_VERSION = 2
READOUT_SCHEMA_NAME = READOUT_SCHEMA = "thesis.stage1.readout"
READOUT_VERSION = 2
MANIFEST_SCHEMA_NAME = MANIFEST_SCHEMA = "thesis.stage1.extraction_manifest"
MANIFEST_VERSION = 2
HASH_ALGORITHM = "sha256-array-v1"


def sha256_file(path: os.PathLike[str] | str, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash dtype, shape and bytes in a platform-independent little-endian form."""
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays cannot be fingerprinted canonically")
    canonical_dtype = value.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
    header = json.dumps(
        {"dtype": canonical_dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"thesis.sha256-array.v1\0")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def endpoint_sha256(endpoints: np.ndarray) -> str:
    value = np.asarray(endpoints, dtype="<i8")
    if value.ndim != 1:
        raise ValueError(f"endpoints must be one-dimensional, got {value.shape}")
    return sha256_array(value)


def canonical_sha256(mapping: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"thesis.canonical-json.v1\0")
    digest.update(encoded)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # The replace has already succeeded. Some filesystems do not
            # support directory fsync; that must not turn a committed write
            # into a reported failure.
            pass
    finally:
        os.close(descriptor)


def atomic_savez(path: os.PathLike[str] | str, **arrays: Any) -> None:
    """Write an uncompressed NPZ via a same-directory temp and atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _as_int64_vector(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must have an integer dtype, got {array.dtype}")
    result = np.ascontiguousarray(array, dtype=np.int64)
    if len(result) and result[0] < 0:
        raise ValueError(f"{name} integrity failure: values must be non-negative")
    if len(result) and np.any(result[1:] <= result[:-1]):
        raise ValueError(
            f"{name} integrity failure: values must be strictly increasing and unique"
        )
    return result


def _normalized_split_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    def first(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in config:
                return config[name]
        return default

    required = {
        "K": first("K"),
        "max_horizon": first("max_horizon", "max_h"),
        "val_frac": first("val_frac"),
        "split_seed": first("split_seed"),
        "subsample_seed": first("subsample_seed"),
        "requested_n_train": first("requested_n_train", "n_train"),
        "requested_n_val": first("requested_n_val", "n_val"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"split config is missing required fields: {missing}")
    return {
        "K": int(required["K"]),
        "max_horizon": int(required["max_horizon"]),
        "vol_clip": float(first("vol_clip", default=5.0)),
        "val_frac": float(required["val_frac"]),
        "split_seed": int(required["split_seed"]),
        "subsample_seed": int(required["subsample_seed"]),
        "requested_n_train": int(required["requested_n_train"]),
        "requested_n_val": int(required["requested_n_val"]),
        "grouping": str(first("grouping", default="stock_id+day_id")),
        "split_algorithm": str(
            first("split_algorithm", default="grouped_split_by_stock_day.v1")
        ),
        "subsample_algorithm": str(
            first(
                "subsample_algorithm",
                default="numpy.default_rng.choice_without_replacement_then_sort.v1",
            )
        ),
    }


@dataclass
class SplitArtifact:
    valid_t: np.ndarray
    train_pos: np.ndarray
    val_pos: np.ndarray
    train_t: np.ndarray
    val_t: np.ndarray
    dataset_sha256: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    split_fingerprint: str = ""

    def __post_init__(self) -> None:
        self.valid_t = _as_int64_vector("valid_t", self.valid_t)
        self.train_pos = _as_int64_vector("train_pos", self.train_pos)
        self.val_pos = _as_int64_vector("val_pos", self.val_pos)
        self.train_t = _as_int64_vector("train_t", self.train_t)
        self.val_t = _as_int64_vector("val_t", self.val_t)
        self.validate_mapping()
        if not self.split_fingerprint:
            self.split_fingerprint = self.compute_fingerprint()

    @property
    def valid_t_sha256(self) -> str:
        return endpoint_sha256(self.valid_t)

    @property
    def train_pos_sha256(self) -> str:
        return endpoint_sha256(self.train_pos)

    @property
    def val_pos_sha256(self) -> str:
        return endpoint_sha256(self.val_pos)

    @property
    def train_endpoint_sha256(self) -> str:
        return endpoint_sha256(self.train_t)

    @property
    def val_endpoint_sha256(self) -> str:
        return endpoint_sha256(self.val_t)

    def validate_mapping(self) -> None:
        if len(self.train_pos) and self.train_pos[-1] >= len(self.valid_t):
            raise ValueError("train_pos is out of bounds for valid_t")
        if len(self.val_pos) and self.val_pos[-1] >= len(self.valid_t):
            raise ValueError("val_pos is out of bounds for valid_t")
        if not np.array_equal(self.train_t, self.valid_t[self.train_pos]):
            raise ValueError("train endpoint integrity failure: train_t != valid_t[train_pos]")
        if not np.array_equal(self.val_t, self.valid_t[self.val_pos]):
            raise ValueError("validation endpoint integrity failure: val_t != valid_t[val_pos]")
        if np.intersect1d(self.train_pos, self.val_pos, assume_unique=True).size:
            raise ValueError("train_pos and val_pos overlap")

    def fingerprint_payload(self) -> Dict[str, Any]:
        return {
            "schema_name": SPLIT_SCHEMA,
            "schema_version": SPLIT_VERSION,
            "dataset_sha256": self.dataset_sha256,
            "config": self.config,
            "n_valid_total": len(self.valid_t),
            "valid_t_sha256": self.valid_t_sha256,
            "train_pos_sha256": self.train_pos_sha256,
            "val_pos_sha256": self.val_pos_sha256,
            "train_endpoint_sha256": self.train_endpoint_sha256,
            "val_endpoint_sha256": self.val_endpoint_sha256,
        }

    def compute_fingerprint(self) -> str:
        return canonical_sha256(self.fingerprint_payload())

    def bind(self, *, dataset_sha256: str, config: Mapping[str, Any]) -> "SplitArtifact":
        self.dataset_sha256 = str(dataset_sha256)
        self.config = _normalized_split_config(config)
        self.split_fingerprint = self.compute_fingerprint()
        return self


def build_endpoint_split(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    vol_mask: np.ndarray,
    *,
    K: int,
    max_horizon: int,
    val_frac: float,
    split_seed: int,
    n_train: int,
    n_val: int,
    subsample_seed: int,
    compute_valid_endpoints_fn: Optional[Callable[..., np.ndarray]] = None,
    grouped_split_fn: Optional[
        Callable[..., Tuple[np.ndarray, np.ndarray]]
    ] = None,
) -> SplitArtifact:
    stock_ids = np.asarray(stock_ids, dtype=np.int64)
    day_ids = np.asarray(day_ids, dtype=np.int64)
    vol_mask = np.asarray(vol_mask, dtype=bool)
    if not (stock_ids.ndim == day_ids.ndim == vol_mask.ndim == 1):
        raise ValueError("stock_ids, day_ids and vol_mask must be one-dimensional")
    if not (len(stock_ids) == len(day_ids) == len(vol_mask)):
        raise ValueError("stock_ids, day_ids and vol_mask lengths differ")
    if compute_valid_endpoints_fn is None or grouped_split_fn is None:
        from training.train_tokenizer_t import (
            compute_valid_endpoints,
            grouped_split_by_stock_day,
        )

        compute_valid_endpoints_fn = (
            compute_valid_endpoints_fn or compute_valid_endpoints
        )
        grouped_split_fn = grouped_split_fn or grouped_split_by_stock_day

    valid_t = np.asarray(
        compute_valid_endpoints_fn(
            stock_ids, day_ids, K, max_horizon, vol_mask
        ),
        dtype=np.int64,
    )
    train_pos, val_pos = grouped_split_fn(
        stock_ids, day_ids, valid_t, val_frac, split_seed
    )
    train_pos = np.asarray(train_pos, dtype=np.int64)
    val_pos = np.asarray(val_pos, dtype=np.int64)
    rng = np.random.default_rng(subsample_seed)
    if len(train_pos) > n_train:
        train_pos = np.sort(
            rng.choice(train_pos, size=n_train, replace=False)
        )
    if len(val_pos) > n_val:
        val_pos = np.sort(rng.choice(val_pos, size=n_val, replace=False))
    train_pos = np.sort(train_pos)
    val_pos = np.sort(val_pos)
    result = SplitArtifact(
        valid_t=valid_t,
        train_pos=train_pos,
        val_pos=val_pos,
        train_t=valid_t[train_pos],
        val_t=valid_t[val_pos],
        config=_normalized_split_config(
            {
                "K": K,
                "max_horizon": max_horizon,
                "val_frac": val_frac,
                "split_seed": split_seed,
                "subsample_seed": subsample_seed,
                "n_train": n_train,
                "n_val": n_val,
            }
        ),
    )
    train_groups = set(
        zip(stock_ids[result.train_t].tolist(), day_ids[result.train_t].tolist())
    )
    val_groups = set(
        zip(stock_ids[result.val_t].tolist(), day_ids[result.val_t].tolist())
    )
    if not train_groups.isdisjoint(val_groups):
        raise ValueError("grouped split integrity failure: stock-day overlap")
    return result


def save_split(
    path: os.PathLike[str] | str,
    split: SplitArtifact,
    *,
    dataset_sha256: str,
    split_config: Mapping[str, Any],
) -> None:
    split.bind(dataset_sha256=dataset_sha256, config=split_config)
    config = split.config
    atomic_savez(
        path,
        schema_name=np.asarray(SPLIT_SCHEMA),
        schema_version=np.asarray(SPLIT_VERSION, dtype=np.int64),
        hash_algorithm=np.asarray(HASH_ALGORITHM),
        dataset_sha256=np.asarray(split.dataset_sha256),
        split_fingerprint=np.asarray(split.split_fingerprint),
        valid_t_sha256=np.asarray(split.valid_t_sha256),
        train_pos_sha256=np.asarray(split.train_pos_sha256),
        val_pos_sha256=np.asarray(split.val_pos_sha256),
        train_endpoint_sha256=np.asarray(split.train_endpoint_sha256),
        val_endpoint_sha256=np.asarray(split.val_endpoint_sha256),
        K=np.asarray(config["K"], dtype=np.int64),
        max_horizon=np.asarray(config["max_horizon"], dtype=np.int64),
        vol_clip=np.asarray(config["vol_clip"], dtype=np.float64),
        val_frac=np.asarray(config["val_frac"], dtype=np.float64),
        split_seed=np.asarray(config["split_seed"], dtype=np.int64),
        subsample_seed=np.asarray(config["subsample_seed"], dtype=np.int64),
        requested_n_train=np.asarray(
            config["requested_n_train"], dtype=np.int64
        ),
        requested_n_val=np.asarray(config["requested_n_val"], dtype=np.int64),
        grouping=np.asarray(config["grouping"]),
        split_algorithm=np.asarray(config["split_algorithm"]),
        subsample_algorithm=np.asarray(config["subsample_algorithm"]),
        n_valid_total=np.asarray(len(split.valid_t), dtype=np.int64),
        valid_t=split.valid_t.astype("<i8", copy=False),
        train_pos=split.train_pos.astype("<i8", copy=False),
        val_pos=split.val_pos.astype("<i8", copy=False),
        train_t=split.train_t.astype("<i8", copy=False),
        val_t=split.val_t.astype("<i8", copy=False),
    )


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"split field {key} must be scalar")
    return value.item()


def load_split(
    path: os.PathLike[str] | str,
    *,
    expected_dataset_sha256: Optional[str] = None,
) -> SplitArtifact:
    required = {
        "schema_name", "schema_version", "hash_algorithm", "dataset_sha256",
        "split_fingerprint", "valid_t_sha256", "train_pos_sha256",
        "val_pos_sha256", "train_endpoint_sha256", "val_endpoint_sha256",
        "K", "max_horizon", "vol_clip", "val_frac", "split_seed",
        "subsample_seed", "requested_n_train", "requested_n_val", "grouping",
        "split_algorithm", "subsample_algorithm", "n_valid_total", "valid_t",
        "train_pos", "val_pos", "train_t", "val_t",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    "legacy or invalid split schema: missing "
                    + ", ".join(sorted(missing))
                )
            if _scalar(data, "schema_name") != SPLIT_SCHEMA:
                raise ValueError("unsupported split schema name")
            if int(_scalar(data, "schema_version")) != SPLIT_VERSION:
                raise ValueError("unsupported split schema version")
            if _scalar(data, "hash_algorithm") != HASH_ALGORITHM:
                raise ValueError("unsupported split hash algorithm")
            dataset_sha256 = str(_scalar(data, "dataset_sha256"))
            if (
                expected_dataset_sha256 is not None
                and dataset_sha256 != expected_dataset_sha256
            ):
                raise ValueError("split dataset SHA-256 mismatch")
            config = {
                "K": int(_scalar(data, "K")),
                "max_horizon": int(_scalar(data, "max_horizon")),
                "vol_clip": float(_scalar(data, "vol_clip")),
                "val_frac": float(_scalar(data, "val_frac")),
                "split_seed": int(_scalar(data, "split_seed")),
                "subsample_seed": int(_scalar(data, "subsample_seed")),
                "requested_n_train": int(_scalar(data, "requested_n_train")),
                "requested_n_val": int(_scalar(data, "requested_n_val")),
                "grouping": str(_scalar(data, "grouping")),
                "split_algorithm": str(_scalar(data, "split_algorithm")),
                "subsample_algorithm": str(_scalar(data, "subsample_algorithm")),
            }
            result = SplitArtifact(
                valid_t=data["valid_t"].copy(),
                train_pos=data["train_pos"].copy(),
                val_pos=data["val_pos"].copy(),
                train_t=data["train_t"].copy(),
                val_t=data["val_t"].copy(),
                dataset_sha256=dataset_sha256,
                config=config,
                split_fingerprint=str(_scalar(data, "split_fingerprint")),
            )
            expected_hashes = {
                "valid_t_sha256": result.valid_t_sha256,
                "train_pos_sha256": result.train_pos_sha256,
                "val_pos_sha256": result.val_pos_sha256,
                "train_endpoint_sha256": result.train_endpoint_sha256,
                "val_endpoint_sha256": result.val_endpoint_sha256,
            }
            for key, expected in expected_hashes.items():
                if str(_scalar(data, key)) != expected:
                    raise ValueError(f"split integrity hash mismatch: {key}")
            if int(_scalar(data, "n_valid_total")) != len(result.valid_t):
                raise ValueError("split integrity mismatch: n_valid_total")
    except (OSError, KeyError) as exc:
        raise ValueError(f"cannot read split schema: {exc}") from exc
    if result.split_fingerprint != result.compute_fingerprint():
        raise ValueError("split integrity fingerprint mismatch")
    return result

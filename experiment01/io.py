"""Canonical hashing and atomic serialization helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays do not have a canonical Experiment 01 hash")
    canonical_dtype = value.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
    header = json.dumps(
        {"dtype": canonical_dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"thesis.experiment01.array.v1\0")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


class StreamingArrayHasher:
    """Incrementally compute the exact ``sha256_array`` digest.

    Chunks must cover the declared array in row-major row order.  This is used
    by the CSV↔NPZ equivalence gate so the reconstructed 8M-row book never has
    to be materialized a second time merely for hashing.
    """

    def __init__(self, dtype: np.dtype | str, shape: Sequence[int]):
        self.dtype = np.dtype(dtype).newbyteorder("<")
        self.shape = tuple(int(value) for value in shape)
        if not self.shape or any(value < 0 for value in self.shape):
            raise ValueError(f"invalid streaming array shape {self.shape}")
        header = json.dumps(
            {"dtype": self.dtype.str, "shape": list(self.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._digest = hashlib.sha256()
        self._digest.update(b"thesis.experiment01.array.v1\0")
        self._digest.update(len(header).to_bytes(8, "little"))
        self._digest.update(header)
        self._rows = 0

    def update(self, chunk: np.ndarray) -> None:
        value = np.asarray(chunk)
        expected_tail = self.shape[1:]
        if value.ndim != len(self.shape) or value.shape[1:] != expected_tail:
            raise ValueError(
                f"streaming hash chunk has shape {value.shape}; "
                f"expected (*, {expected_tail})"
            )
        if self._rows + len(value) > self.shape[0]:
            raise ValueError("streaming hash received more rows than declared")
        canonical = np.ascontiguousarray(
            value.astype(self.dtype, copy=False)
        )
        self._digest.update(canonical.tobytes(order="C"))
        self._rows += len(value)

    def hexdigest(self) -> str:
        if self._rows != self.shape[0]:
            raise ValueError(
                f"streaming hash received {self._rows} rows, "
                f"expected {self.shape[0]}"
            )
        return self._digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"thesis.experiment01.json.v1\0")
    digest.update(encoded)
    return digest.hexdigest()


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy/non-finite values to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_parquet(frame, path: str | os.PathLike[str]) -> None:
    """Write a DataFrame through a same-directory temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".parquet.tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_savez(path: str | os.PathLike[str], **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz.tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def atomic_save_npy(path: str | os.PathLike[str], array: np.ndarray) -> None:
    """Atomically write one NPY array without adding a filename suffix."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npy.tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise

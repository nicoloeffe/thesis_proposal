"""Row-addressable, fail-closed sharded NPY arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SHARDED_STORAGE = "sharded_npy_v1"


@dataclass(frozen=True)
class ArrayShard:
    path: Path
    row_start: int
    row_stop: int

    @property
    def n_rows(self) -> int:
        return self.row_stop - self.row_start


class ShardedArray:
    """Minimal NumPy-like row array backed by verified NPY shards."""

    def __init__(
        self,
        shards: Sequence[ArrayShard],
        shape: Sequence[int],
        dtype: np.dtype | str,
    ):
        self.shards = tuple(shards)
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        if len(self.shape) < 1:
            raise ValueError("sharded arrays must have a row dimension")
        cursor = 0
        for shard in self.shards:
            if shard.row_start != cursor or shard.row_stop <= shard.row_start:
                raise ValueError("shard intervals are not contiguous and non-empty")
            cursor = shard.row_stop
        if cursor != self.shape[0]:
            raise ValueError("shards do not cover the declared array shape")
        self._stops = np.asarray(
            [shard.row_stop for shard in self.shards], dtype=np.int64
        )

    def __len__(self) -> int:
        return self.shape[0]

    def _load(self, index: int) -> np.ndarray:
        return np.load(
            self.shards[index].path, mmap_mode="r", allow_pickle=False
        )

    def _slice(self, value: slice) -> np.ndarray:
        start, stop, step = value.indices(len(self))
        if step != 1:
            return self._fancy(np.arange(start, stop, step, dtype=np.int64))
        if stop <= start:
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        first = int(np.searchsorted(self._stops, start, side="right"))
        last = int(np.searchsorted(self._stops, stop - 1, side="right"))
        parts = []
        for shard_index in range(first, last + 1):
            shard = self.shards[shard_index]
            local_start = max(start, shard.row_start) - shard.row_start
            local_stop = min(stop, shard.row_stop) - shard.row_start
            parts.append(np.asarray(self._load(shard_index)[local_start:local_stop]))
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def _fancy(self, value: np.ndarray) -> np.ndarray:
        indices = np.asarray(value)
        if indices.dtype == bool:
            if indices.ndim != 1 or len(indices) != len(self):
                raise IndexError("boolean shard index has invalid shape")
            indices = np.flatnonzero(indices)
        indices = indices.astype(np.int64, copy=False)
        if indices.ndim != 1:
            raise IndexError("sharded arrays support one-dimensional row indices")
        indices = np.where(indices < 0, indices + len(self), indices)
        if len(indices) and (
            int(indices.min()) < 0 or int(indices.max()) >= len(self)
        ):
            raise IndexError("sharded array row index out of range")
        result = np.empty((len(indices), *self.shape[1:]), dtype=self.dtype)
        shard_indices = np.searchsorted(self._stops, indices, side="right")
        for shard_index in np.unique(shard_indices):
            output_positions = np.flatnonzero(shard_indices == shard_index)
            shard = self.shards[int(shard_index)]
            local = indices[output_positions] - shard.row_start
            result[output_positions] = self._load(int(shard_index))[local]
        return result

    def __getitem__(self, value):
        column_index = None
        row_index = value
        if isinstance(value, tuple):
            if not value:
                raise IndexError("empty array index")
            row_index = value[0]
            column_index = value[1:]
        if isinstance(row_index, (int, np.integer)):
            index = int(row_index)
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("sharded array row index out of range")
            shard_index = int(np.searchsorted(self._stops, index, side="right"))
            shard = self.shards[shard_index]
            result = np.asarray(
                self._load(shard_index)[index - shard.row_start]
            )
        elif isinstance(row_index, slice):
            result = self._slice(row_index)
        else:
            result = self._fancy(np.asarray(row_index))
        if column_index is not None:
            result = result[(..., *column_index)]
        return result


def sharded_record_fingerprint_payload(
    record: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "storage": record.get("storage"),
        "shape": record.get("shape"),
        "dtype": record.get("dtype"),
        "row_key_sha256": record.get("row_key_sha256"),
        "shards": record.get("shards"),
    }

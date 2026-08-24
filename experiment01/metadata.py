"""Canonical LOBench metadata sidecar and CSV↔NPZ equivalence gate."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.dataset.build_encoder_dataset_lobench import (
    ASK_PRICE_COLS,
    ASK_VOL_COLS,
    BID_PRICE_COLS,
    BID_VOL_COLS,
    filter_canonical_rows,
)

from .errors import ExperimentIntegrityError
from .io import (
    StreamingArrayHasher,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)


SIDECAR_SCHEMA = "thesis.experiment01.lobench_metadata"
SIDECAR_SCHEMA_VERSION = 1
EQUIVALENCE_SCHEMA = "thesis.experiment01.csv_npz_equivalence"
EQUIVALENCE_SCHEMA_VERSION = 1
EXPECTED_TOTAL_ROWS = 8_039_246
SYMBOL_PATTERN = re.compile(r"^(sz\d+)-level10_processed$")

SIDECAR_COLUMNS = (
    "global_row_index",
    "stock_id",
    "stock_symbol",
    "timestamp_ns",
    "trading_date",
    "day_id",
    "endpoint_order",
    "raw_csv_row_index",
)

SIDECAR_ARROW_SCHEMA = pa.schema(
    [
        pa.field("global_row_index", pa.int64(), nullable=False),
        pa.field("stock_id", pa.int32(), nullable=False),
        pa.field("stock_symbol", pa.string(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("trading_date", pa.string(), nullable=False),
        pa.field("day_id", pa.int32(), nullable=False),
        pa.field("endpoint_order", pa.int32(), nullable=False),
        pa.field("raw_csv_row_index", pa.int64(), nullable=False),
    ]
)


def discover_raw_csvs(raw_dir: str | Path) -> tuple[Path, ...]:
    paths = tuple(sorted(Path(raw_dir).resolve().glob("sz*_processed.csv")))
    if len(paths) != 7:
        raise ExperimentIntegrityError(
            f"expected exactly seven canonical raw CSVs, found {len(paths)}"
        )
    symbols = [stock_symbol(path) for path in paths]
    if len(set(symbols)) != len(symbols):
        raise ExperimentIntegrityError("raw CSV stock symbols are not unique")
    return paths


def stock_symbol(path: str | Path) -> str:
    match = SYMBOL_PATTERN.fullmatch(Path(path).stem)
    if match is None:
        raise ExperimentIntegrityError(
            f"cannot derive canonical stock symbol from {Path(path).name!r}"
        )
    return match.group(1)


def _book_from_frame(frame: pd.DataFrame) -> np.ndarray:
    result = np.empty((len(frame), 2, 10, 2), dtype=np.float32)
    result[:, 0, :, 0] = frame[BID_PRICE_COLS].to_numpy(dtype=np.float32)
    result[:, 0, :, 1] = frame[BID_VOL_COLS].to_numpy(dtype=np.float32)
    result[:, 1, :, 0] = frame[ASK_PRICE_COLS].to_numpy(dtype=np.float32)
    result[:, 1, :, 1] = frame[ASK_VOL_COLS].to_numpy(dtype=np.float32)
    return result


def _difference(left: np.ndarray, right: np.ndarray) -> Mapping[str, Any]:
    if left.shape != right.shape:
        return {
            "equal": False,
            "mismatch_count": None,
            "max_abs_difference": None,
            "shape_left": list(left.shape),
            "shape_right": list(right.shape),
        }
    equal = np.equal(left, right)
    mismatch_count = int(equal.size - np.count_nonzero(equal))
    if mismatch_count:
        maximum = float(
            np.max(
                np.abs(
                    np.asarray(left, dtype=np.float64)
                    - np.asarray(right, dtype=np.float64)
                )
            )
        )
    else:
        maximum = 0.0
    return {
        "equal": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "max_abs_difference": maximum,
    }


class _OrderState:
    def __init__(self) -> None:
        self.last_timestamp_ns: int | None = None
        self.last_date: str | None = None
        self.next_order = 0
        self.seen_dates: set[str] = set()

    def assign(
        self, timestamp_ns: np.ndarray, trading_dates: np.ndarray
    ) -> np.ndarray:
        if len(timestamp_ns) == 0:
            return np.empty(0, dtype=np.int32)
        if self.last_timestamp_ns is not None and int(timestamp_ns[0]) <= self.last_timestamp_ns:
            raise ExperimentIntegrityError(
                "raw CSV timestamps are not globally strictly increasing within stock"
            )
        if np.any(timestamp_ns[1:] <= timestamp_ns[:-1]):
            raise ExperimentIntegrityError(
                "raw CSV timestamps are not strictly increasing within a chunk"
            )
        result = np.empty(len(timestamp_ns), dtype=np.int32)
        start = 0
        while start < len(result):
            date = str(trading_dates[start])
            stop = start + 1
            while stop < len(result) and str(trading_dates[stop]) == date:
                stop += 1
            if date != self.last_date:
                if date in self.seen_dates:
                    raise ExperimentIntegrityError(
                        f"trading date {date} is split into non-contiguous CSV blocks"
                    )
                self.seen_dates.add(date)
                self.last_date = date
                self.next_order = 0
            result[start:stop] = np.arange(
                self.next_order,
                self.next_order + stop - start,
                dtype=np.int32,
            )
            self.next_order += stop - start
            start = stop
        self.last_timestamp_ns = int(timestamp_ns[-1])
        return result


def _hashers(
    n_rows: int,
) -> dict[str, StreamingArrayHasher]:
    return {
        "book": StreamingArrayHasher(np.dtype("float32"), (n_rows, 2, 10, 2)),
        "mid_z": StreamingArrayHasher(np.dtype("float32"), (n_rows,)),
        "stock_ids": StreamingArrayHasher(np.dtype("int32"), (n_rows,)),
        "day_ids": StreamingArrayHasher(np.dtype("int32"), (n_rows,)),
    }


def _update_hashers(
    hashers: Mapping[str, StreamingArrayHasher],
    *,
    book: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
) -> None:
    hashers["book"].update(book)
    hashers["mid_z"].update(mid_z)
    hashers["stock_ids"].update(stock_ids)
    hashers["day_ids"].update(day_ids)


def _digest_hashers(
    hashers: Mapping[str, StreamingArrayHasher],
) -> dict[str, str]:
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def build_metadata_sidecar(
    raw_dir: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path,
    *,
    chunk_rows: int = 200_000,
    expected_total_rows: int = EXPECTED_TOTAL_ROWS,
) -> Mapping[str, Any]:
    """Build the sidecar and publish it only after full numerical equivalence.

    The gate compares every reconstructed value, not samples.  Any mismatch
    leaves an equivalence report with ``passed=false`` but does not publish a
    canonical sidecar manifest.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    raw_paths = discover_raw_csvs(raw_dir)
    dataset = Path(dataset_path).resolve()
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sidecar_path = destination / "metadata_sidecar.parquet"
    temporary_path = destination / ".metadata_sidecar.parquet.building"
    report_path = destination / "equivalence_report.json"
    manifest_path = destination / "sidecar_manifest.json"
    if sidecar_path.exists() or manifest_path.exists() or temporary_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing sidecar build in {destination}"
        )

    source_columns = [
        "index",
        *BID_PRICE_COLS,
        *BID_VOL_COLS,
        *ASK_PRICE_COLS,
        *ASK_VOL_COLS,
    ]
    builder_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dataset"
        / "build_encoder_dataset_lobench.py"
    )
    source_records = [
        {
            "stock_id": stock_id,
            "stock_symbol": stock_symbol(path),
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for stock_id, path in enumerate(raw_paths)
    ]
    dataset_sha256 = sha256_file(dataset)

    stock_reports: list[dict[str, Any]] = []
    global_mismatch = {
        name: 0 for name in ("book", "mid_z", "stock_ids", "day_ids")
    }
    global_max_abs = {name: 0.0 for name in global_mismatch}
    writer: pq.ParquetWriter | None = None
    global_offset = 0
    passed = False
    try:
        with np.load(dataset, allow_pickle=False) as archive:
            required = {
                "book",
                "mid_z",
                "stock_ids",
                "day_ids",
                "min_spread_z_per_stock",
                "price_std_rmb_per_stock",
            }
            missing = required - set(archive.files)
            if missing:
                raise ExperimentIntegrityError(
                    f"processed NPZ is missing fields {sorted(missing)}"
                )
            book_npz = archive["book"]
            mid_npz = archive["mid_z"]
            stocks_npz = archive["stock_ids"]
            days_npz = archive["day_ids"]
            if len(book_npz) != expected_total_rows:
                raise ExperimentIntegrityError(
                    f"NPZ has {len(book_npz):,} rows; expected "
                    f"{expected_total_rows:,}"
                )
            if not (
                len(mid_npz) == len(stocks_npz) == len(days_npz) == len(book_npz)
            ):
                raise ExperimentIntegrityError("processed NPZ arrays are misaligned")
            expected_counts = np.bincount(
                stocks_npz.astype(np.int64), minlength=len(raw_paths)
            )
            raw_global_hashers = _hashers(len(book_npz))
            npz_global_hashers = _hashers(len(book_npz))
            writer = pq.ParquetWriter(
                temporary_path,
                SIDECAR_ARROW_SCHEMA,
                compression="zstd",
                use_dictionary=["stock_symbol", "trading_date"],
            )

            for stock_id, raw_path in enumerate(raw_paths):
                symbol = stock_symbol(raw_path)
                expected_count = int(expected_counts[stock_id])
                raw_stock_hashers = _hashers(expected_count)
                npz_stock_hashers = _hashers(expected_count)
                raw_row_offset = 0
                stock_rows = 0
                day_origin: int | None = None
                order_state = _OrderState()
                first_timestamp: str | None = None
                last_timestamp: str | None = None
                for source in pd.read_csv(
                    raw_path,
                    usecols=source_columns,
                    chunksize=chunk_rows,
                ):
                    raw_indices = np.arange(
                        raw_row_offset,
                        raw_row_offset + len(source),
                        dtype=np.int64,
                    )
                    raw_row_offset += len(source)
                    source = source.copy()
                    source["__raw_csv_row_index"] = raw_indices
                    frame = filter_canonical_rows(source)
                    if frame.empty:
                        continue
                    timestamp = pd.to_datetime(frame["index"], errors="raise")
                    timestamp_ns = timestamp.astype("int64").to_numpy(
                        dtype=np.int64
                    )
                    trading_date = timestamp.dt.strftime("%Y-%m-%d").to_numpy(
                        dtype=str
                    )
                    day_of_year = timestamp.dt.dayofyear.to_numpy(dtype=np.int32)
                    if day_origin is None:
                        day_origin = int(day_of_year.min())
                    if int(day_of_year.min()) < day_origin:
                        raise ExperimentIntegrityError(
                            f"{symbol}: CSV date order invalidates canonical day_id"
                        )
                    day_id = (day_of_year - day_origin).astype(np.int32)
                    endpoint_order = order_state.assign(
                        timestamp_ns, trading_date
                    )
                    if first_timestamp is None:
                        first_timestamp = str(timestamp.iloc[0])
                    last_timestamp = str(timestamp.iloc[-1])

                    raw_book = _book_from_frame(frame)
                    raw_mid = (
                        raw_book[:, 0, 0, 0] + raw_book[:, 1, 0, 0]
                    ) / np.float32(2.0)
                    raw_stock_ids = np.full(
                        len(frame), stock_id, dtype=np.int32
                    )
                    stop = global_offset + len(frame)
                    if stop > len(book_npz):
                        raise ExperimentIntegrityError(
                            "CSV reconstruction contains more rows than the NPZ"
                        )
                    expected_book = np.asarray(book_npz[global_offset:stop])
                    expected_mid = np.asarray(mid_npz[global_offset:stop])
                    expected_stock_ids = np.asarray(
                        stocks_npz[global_offset:stop]
                    )
                    expected_day_ids = np.asarray(days_npz[global_offset:stop])
                    values = {
                        "book": (raw_book, expected_book),
                        "mid_z": (raw_mid.astype(np.float32), expected_mid),
                        "stock_ids": (raw_stock_ids, expected_stock_ids),
                        "day_ids": (day_id, expected_day_ids),
                    }
                    for name, (raw_value, expected_value) in values.items():
                        comparison = _difference(raw_value, expected_value)
                        global_mismatch[name] += int(
                            comparison["mismatch_count"] or 0
                        )
                        global_max_abs[name] = max(
                            global_max_abs[name],
                            float(comparison["max_abs_difference"] or 0.0),
                        )
                    _update_hashers(
                        raw_global_hashers,
                        book=raw_book,
                        mid_z=raw_mid,
                        stock_ids=raw_stock_ids,
                        day_ids=day_id,
                    )
                    _update_hashers(
                        npz_global_hashers,
                        book=expected_book,
                        mid_z=expected_mid,
                        stock_ids=expected_stock_ids,
                        day_ids=expected_day_ids,
                    )
                    _update_hashers(
                        raw_stock_hashers,
                        book=raw_book,
                        mid_z=raw_mid,
                        stock_ids=raw_stock_ids,
                        day_ids=day_id,
                    )
                    _update_hashers(
                        npz_stock_hashers,
                        book=expected_book,
                        mid_z=expected_mid,
                        stock_ids=expected_stock_ids,
                        day_ids=expected_day_ids,
                    )

                    table = pa.Table.from_pydict(
                        {
                            "global_row_index": np.arange(
                                global_offset, stop, dtype=np.int64
                            ),
                            "stock_id": raw_stock_ids,
                            "stock_symbol": np.repeat(symbol, len(frame)),
                            "timestamp_ns": timestamp_ns,
                            "trading_date": trading_date,
                            "day_id": day_id,
                            "endpoint_order": endpoint_order,
                            "raw_csv_row_index": frame[
                                "__raw_csv_row_index"
                            ].to_numpy(dtype=np.int64),
                        },
                        schema=SIDECAR_ARROW_SCHEMA,
                    )
                    writer.write_table(table)
                    global_offset = stop
                    stock_rows += len(frame)

                stock_reports.append(
                    {
                        "stock_id": stock_id,
                        "stock_symbol": symbol,
                        "raw_rows_before_filter": raw_row_offset,
                        "rows_after_filter": stock_rows,
                        "npz_rows": expected_count,
                        "count_equal": stock_rows == expected_count,
                        "first_timestamp": first_timestamp,
                        "last_timestamp": last_timestamp,
                        "reconstructed_hashes": _digest_hashers(
                            raw_stock_hashers
                        ),
                        "npz_hashes": _digest_hashers(npz_stock_hashers),
                    }
                )
            writer.close()
            writer = None
            reconstructed_hashes = _digest_hashers(raw_global_hashers)
            npz_hashes = _digest_hashers(npz_global_hashers)

        passed = (
            global_offset == expected_total_rows
            and all(value == 0 for value in global_mismatch.values())
            and all(record["count_equal"] for record in stock_reports)
            and reconstructed_hashes == npz_hashes
        )
        report: dict[str, Any] = {
            "schema_name": EQUIVALENCE_SCHEMA,
            "schema_version": EQUIVALENCE_SCHEMA_VERSION,
            "passed": passed,
            "fail_closed": True,
            "dataset": {
                "path": str(dataset),
                "sha256": dataset_sha256,
                "size_bytes": dataset.stat().st_size,
            },
            "builder": {
                "path": str(builder_path),
                "sha256": sha256_file(builder_path),
                "filter_function": "filter_canonical_rows",
                "subsample_stride": 1,
            },
            "raw_sources": source_records,
            "checks": {
                "expected_total_rows": expected_total_rows,
                "reconstructed_total_rows": global_offset,
                "total_rows_equal": global_offset == expected_total_rows,
                "stock_counts_equal": all(
                    record["count_equal"] for record in stock_reports
                ),
                "stock_ids_equal": global_mismatch["stock_ids"] == 0,
                "day_ids_equal": global_mismatch["day_ids"] == 0,
                "book_numerically_equal": global_mismatch["book"] == 0,
                "mid_z_numerically_equal": global_mismatch["mid_z"] == 0,
                "global_order_equal": all(
                    value == 0 for value in global_mismatch.values()
                ),
            },
            "mismatch_counts": global_mismatch,
            "max_abs_differences": global_max_abs,
            "reconstructed_hashes": reconstructed_hashes,
            "npz_hashes": npz_hashes,
            "stocks": stock_reports,
        }
        report["equivalence_fingerprint"] = canonical_json_sha256(report)
        atomic_write_json(report_path, report)
        if not passed:
            temporary_path.unlink(missing_ok=True)
            raise ExperimentIntegrityError(
                f"CSV↔NPZ equivalence gate failed; see {report_path}"
            )
        os.replace(temporary_path, sidecar_path)
        manifest: dict[str, Any] = {
            "schema_name": SIDECAR_SCHEMA,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "status": "verified",
            "canonical_stock_day_identity": ["stock_id", "trading_date"],
            "columns": list(SIDECAR_COLUMNS),
            "n_rows": global_offset,
            "sidecar": {
                "path": sidecar_path.name,
                "sha256": sha256_file(sidecar_path),
                "size_bytes": sidecar_path.stat().st_size,
            },
            "equivalence_report": {
                "path": report_path.name,
                "sha256": sha256_file(report_path),
                "fingerprint": report["equivalence_fingerprint"],
                "passed": True,
            },
            "dataset_sha256": dataset_sha256,
            "builder_sha256": sha256_file(builder_path),
            "raw_source_sha256": {
                record["stock_symbol"]: record["sha256"]
                for record in source_records
            },
        }
        manifest["sidecar_fingerprint"] = canonical_json_sha256(manifest)
        atomic_write_json(manifest_path, manifest)
        return {"manifest": manifest, "equivalence": report}
    except BaseException:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise


def load_verified_sidecar_manifest(
    sidecar_dir: str | Path,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    root = Path(sidecar_dir).resolve()
    manifest_path = root / "sidecar_manifest.json"
    report_path = root / "equivalence_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ExperimentIntegrityError("verified sidecar manifest/report is missing")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_name") != SIDECAR_SCHEMA
        or manifest.get("schema_version") != SIDECAR_SCHEMA_VERSION
        or manifest.get("status") != "verified"
        or report.get("schema_name") != EQUIVALENCE_SCHEMA
        or report.get("schema_version") != EQUIVALENCE_SCHEMA_VERSION
        or report.get("passed") is not True
    ):
        raise ExperimentIntegrityError("sidecar equivalence status is not valid")
    manifest_payload = dict(manifest)
    recorded_sidecar_fingerprint = manifest_payload.pop(
        "sidecar_fingerprint", None
    )
    if canonical_json_sha256(manifest_payload) != recorded_sidecar_fingerprint:
        raise ExperimentIntegrityError("sidecar manifest fingerprint mismatch")
    report_payload = dict(report)
    recorded_equivalence_fingerprint = report_payload.pop(
        "equivalence_fingerprint", None
    )
    if canonical_json_sha256(report_payload) != recorded_equivalence_fingerprint:
        raise ExperimentIntegrityError("equivalence report fingerprint mismatch")
    if (
        manifest.get("equivalence_report", {}).get("fingerprint")
        != recorded_equivalence_fingerprint
    ):
        raise ExperimentIntegrityError(
            "sidecar/equivalence fingerprint linkage mismatch"
        )
    sidecar_record = manifest.get("sidecar", {})
    sidecar = root / str(sidecar_record.get("path", ""))
    if not sidecar.is_file():
        raise ExperimentIntegrityError("sidecar Parquet is missing")
    if sha256_file(sidecar) != sidecar_record.get("sha256"):
        raise ExperimentIntegrityError("sidecar Parquet SHA-256 mismatch")
    if sha256_file(report_path) != manifest.get("equivalence_report", {}).get(
        "sha256"
    ):
        raise ExperimentIntegrityError("equivalence report SHA-256 mismatch")
    return sidecar, manifest, report

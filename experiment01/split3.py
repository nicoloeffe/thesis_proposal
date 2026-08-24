"""Preregistered three-way split derived from the historical held-out groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiment01.reference.analysis_artifacts import load_split
from training.train_tokenizer_t import (
    compute_valid_endpoints,
    grouped_split_by_stock_day,
)

from .errors import ExperimentIntegrityError
from .io import (
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_sha256,
    sha256_array,
    sha256_file,
)
from .metadata import load_verified_sidecar_manifest


SPLIT3_SCHEMA = "thesis.experiment01.three_way_split"
SPLIT3_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")


def chronological_heldout_halves(
    trading_dates: list[str] | tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return first floor(n/2) dates as validation and the rest as test."""
    dates = tuple(sorted(str(value) for value in trading_dates))
    if len(set(dates)) != len(dates):
        raise ValueError("held-out trading dates must be unique")
    cut = len(dates) // 2
    return dates[:cut], dates[cut:]


def _stock_day_rows(
    metadata: pd.DataFrame,
    assignments: np.ndarray,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    working = metadata.loc[
        :,
        [
            "stock_id",
            "stock_symbol",
            "trading_date",
            "day_id",
            "global_row_index",
            "timestamp_ns",
        ],
    ].copy()
    working["__assignment"] = assignments
    for (stock_id, trading_date), group in working.groupby(
        ["stock_id", "trading_date"], sort=True, observed=True
    ):
        unique_assignments = np.unique(group["__assignment"])
        if len(unique_assignments) != 1:
            raise ExperimentIntegrityError(
                f"stock-day ({stock_id}, {trading_date}) spans split assignments"
            )
        split_index = int(unique_assignments[0])
        if split_index not in range(len(SPLIT_NAMES)):
            raise ExperimentIntegrityError("an endpoint has no split assignment")
        if group["day_id"].nunique() != 1 or group["stock_symbol"].nunique() != 1:
            raise ExperimentIntegrityError(
                f"stock-day ({stock_id}, {trading_date}) has inconsistent metadata"
            )
        values.append(
            {
                "stock_id": int(stock_id),
                "stock_symbol": str(group["stock_symbol"].iloc[0]),
                "trading_date": str(trading_date),
                "day_id": int(group["day_id"].iloc[0]),
                "split": SPLIT_NAMES[split_index],
                "n_endpoints": len(group),
                "first_global_row_index": int(group["global_row_index"].iloc[0]),
                "last_global_row_index": int(group["global_row_index"].iloc[-1]),
                "first_timestamp_ns": int(group["timestamp_ns"].iloc[0]),
                "last_timestamp_ns": int(group["timestamp_ns"].iloc[-1]),
            }
        )
    return values


def _row_table(metadata: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    source = metadata.iloc[positions].copy()
    source["stock_symbol"] = source["stock_symbol"].astype(str)
    source["trading_date"] = source["trading_date"].astype(str)
    source["endpoint_order"] = (
        source.groupby(
            ["stock_id", "trading_date"], sort=False, observed=True
        ).cumcount()
    ).astype(np.int32)
    order_text = source["endpoint_order"].astype(str)
    source["row_key"] = (
        source["stock_id"].astype(str)
        + "|"
        + source["trading_date"]
        + "|"
        + order_text
    )
    result = pd.DataFrame(
        {
            "row_key": source["row_key"],
            "stock_id": source["stock_id"].astype(np.int32),
            "stock_symbol": source["stock_symbol"],
            "stock_day_id": source["day_id"].astype(np.int32),
            "trading_date": source["trading_date"],
            "endpoint_index": source["global_row_index"].astype(np.int64),
            "endpoint_order": source["endpoint_order"].astype(np.int32),
            "timestamp_ns": source["timestamp_ns"].astype(np.int64),
        }
    )
    return result


def _counts_by_stock(frame: pd.DataFrame) -> list[dict[str, Any]]:
    values = []
    for stock_id, group in frame.groupby("stock_id", sort=True, observed=True):
        values.append(
            {
                "stock_id": int(stock_id),
                "stock_symbol": str(group["stock_symbol"].iloc[0]),
                "n_stock_days": int(group["trading_date"].nunique()),
                "n_endpoints": len(group),
                "first_trading_date": str(group["trading_date"].min()),
                "last_trading_date": str(group["trading_date"].max()),
            }
        )
    return values


def build_three_way_split(
    sidecar_dir: str | Path,
    dataset_path: str | Path,
    historical_split_path: str | Path,
    out_dir: str | Path,
) -> Mapping[str, Any]:
    """Rebuild the historical grouping and split its held-out groups by time."""
    sidecar_path, sidecar_manifest, equivalence = (
        load_verified_sidecar_manifest(sidecar_dir)
    )
    dataset = Path(dataset_path).resolve()
    historical_path = Path(historical_split_path).resolve()
    destination = Path(out_dir).resolve()
    manifest_path = destination / "split_manifest.json"
    rows_root = destination / "rows"
    if manifest_path.exists() or rows_root.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing split build in {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if sha256_file(dataset) != sidecar_manifest["dataset_sha256"]:
        raise ExperimentIntegrityError("dataset differs from verified sidecar source")

    historical = load_split(
        historical_path,
        expected_dataset_sha256=sidecar_manifest["dataset_sha256"],
    )
    required_config = {
        "K": 20,
        "max_horizon": 20,
        "vol_clip": 5.0,
        "val_frac": 0.1,
        "split_seed": 0,
        "grouping": "stock_id+day_id",
        "split_algorithm": "grouped_split_by_stock_day.v1",
    }
    for name, expected in required_config.items():
        if historical.config.get(name) != expected:
            raise ExperimentIntegrityError(
                f"historical split {name}={historical.config.get(name)!r}; "
                f"expected {expected!r}"
            )

    table = pq.read_table(sidecar_path)
    metadata = table.to_pandas(
        categories=["stock_symbol", "trading_date"],
        self_destruct=True,
    )
    if not np.array_equal(
        metadata["global_row_index"].to_numpy(dtype=np.int64),
        np.arange(len(metadata), dtype=np.int64),
    ):
        raise ExperimentIntegrityError("sidecar global row order is not canonical")

    with np.load(dataset, allow_pickle=False) as archive:
        book = archive["book"]
        stock_ids = archive["stock_ids"].astype(np.int64)
        day_ids = archive["day_ids"].astype(np.int64)
        bid_volume = book[:, 0, :, 1]
        ask_volume = book[:, 1, :, 1]
        volume_mask = (
            np.abs(bid_volume).max(axis=1) <= required_config["vol_clip"]
        ) & (
            np.abs(ask_volume).max(axis=1) <= required_config["vol_clip"]
        )
        valid_t = np.asarray(
            compute_valid_endpoints(
                stock_ids,
                day_ids,
                required_config["K"],
                required_config["max_horizon"],
                volume_mask,
            ),
            dtype=np.int64,
        )
        if not np.array_equal(valid_t, historical.valid_t):
            raise ExperimentIntegrityError(
                "integrally reconstructed valid endpoints differ from historical split"
            )
        historical_train_pos, historical_heldout_pos = (
            grouped_split_by_stock_day(
                stock_ids,
                day_ids,
                valid_t,
                required_config["val_frac"],
                required_config["split_seed"],
            )
        )
    historical_train_pos = np.asarray(historical_train_pos, dtype=np.int64)
    historical_heldout_pos = np.asarray(
        historical_heldout_pos, dtype=np.int64
    )
    if len(historical_train_pos) + len(historical_heldout_pos) != len(valid_t):
        raise ExperimentIntegrityError("historical full split does not cover valid_t")

    valid_metadata = metadata.iloc[valid_t].reset_index(drop=True)
    if not np.array_equal(
        valid_metadata["stock_id"].to_numpy(dtype=np.int64),
        stock_ids[valid_t],
    ) or not np.array_equal(
        valid_metadata["day_id"].to_numpy(dtype=np.int64),
        day_ids[valid_t],
    ):
        raise ExperimentIntegrityError("sidecar/NPZ endpoint metadata misalignment")

    assignments = np.full(len(valid_t), -1, dtype=np.int8)
    assignments[historical_train_pos] = 0
    heldout = valid_metadata.iloc[historical_heldout_pos]
    heldout_day_split: dict[tuple[int, str], int] = {}
    heldout_partition: list[dict[str, Any]] = []
    for stock_id, group in heldout.groupby("stock_id", sort=True, observed=True):
        dates = sorted(group["trading_date"].astype(str).unique().tolist())
        validation_dates, test_dates = chronological_heldout_halves(dates)
        for date in validation_dates:
            heldout_day_split[(int(stock_id), date)] = 1
        for date in test_dates:
            heldout_day_split[(int(stock_id), date)] = 2
        heldout_partition.append(
            {
                "stock_id": int(stock_id),
                "n_historical_heldout_days": len(dates),
                "n_new_validation_days": len(validation_dates),
                "n_new_test_days": len(test_dates),
                "validation_dates": validation_dates,
                "test_dates": test_dates,
            }
        )
    for position in historical_heldout_pos:
        row = valid_metadata.iloc[int(position)]
        key = (int(row["stock_id"]), str(row["trading_date"]))
        assignments[int(position)] = heldout_day_split[key]
    if np.any(assignments < 0):
        raise ExperimentIntegrityError("three-way split left endpoints unassigned")

    # Prove that the sampled legacy endpoints retain their historical side.
    legacy_train_positions = np.searchsorted(valid_t, historical.train_t)
    legacy_val_positions = np.searchsorted(valid_t, historical.val_t)
    if (
        not np.array_equal(valid_t[legacy_train_positions], historical.train_t)
        or not np.array_equal(valid_t[legacy_val_positions], historical.val_t)
        or np.any(assignments[legacy_train_positions] != 0)
        or np.any(assignments[legacy_val_positions] == 0)
    ):
        raise ExperimentIntegrityError(
            "legacy sampled endpoints do not preserve historical train/held-out sides"
        )

    stock_days = _stock_day_rows(valid_metadata, assignments)
    split_day_sets = {
        split: {
            (row["stock_id"], row["trading_date"])
            for row in stock_days
            if row["split"] == split
        }
        for split in SPLIT_NAMES
    }
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if split_day_sets[left] & split_day_sets[right]:
                raise ExperimentIntegrityError(
                    f"stock-day overlap between {left} and {right}"
                )

    split_records: dict[str, Any] = {}
    for split_index, split in enumerate(SPLIT_NAMES):
        positions = np.flatnonzero(assignments == split_index).astype(np.int64)
        frame = _row_table(valid_metadata, positions)
        relative = Path("rows") / f"{split}.parquet"
        path = destination / relative
        atomic_write_parquet(frame, path)
        row_key_hash = sha256_array(
            frame["row_key"].astype(str).to_numpy(dtype="U")
        )
        split_records[split] = {
            "path": str(relative),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "n_rows": len(frame),
            "n_stock_days": int(frame["trading_date"].nunique())
            if frame["stock_id"].nunique() == 1
            else len(split_day_sets[split]),
            "row_key_sha256": row_key_hash,
            "endpoint_index_sha256": sha256_array(
                frame["endpoint_index"].to_numpy(dtype=np.int64)
            ),
            "complete_stock_days": True,
            "counts_by_stock": _counts_by_stock(frame),
            "first_trading_date": str(frame["trading_date"].min()),
            "last_trading_date": str(frame["trading_date"].max()),
        }

    protocol = {
        "name": "historical_train_plus_chronological_heldout_halves.v1",
        "historical_algorithm": "grouped_split_by_stock_day.v1",
        "historical_split_seed": 0,
        "historical_val_frac": 0.1,
        "grouping": "stock_id+trading_date",
        "train_rule": "all stock-days assigned to historical train remain train",
        "heldout_rule": (
            "within each stock sort historical held-out stock-days by "
            "trading_date; first floor(n/2) -> validation; remainder -> test"
        ),
        "odd_day_rule": "additional held-out day goes to test",
        "test_selection_policy": (
            "test is never used to select alpha, whitening-k, or any other "
            "Experiment 01 hyperparameter"
        ),
        "historical_exposure_disclosure": (
            "validation and test both derive from the previous held-out set, "
            "which was used in historical exploratory analyses"
        ),
    }
    manifest: dict[str, Any] = {
        "schema_name": SPLIT3_SCHEMA,
        "schema_version": SPLIT3_SCHEMA_VERSION,
        "status": "preregistered",
        "dataset_sha256": sidecar_manifest["dataset_sha256"],
        "sidecar_fingerprint": sidecar_manifest["sidecar_fingerprint"],
        "equivalence_fingerprint": equivalence["equivalence_fingerprint"],
        "historical_split": {
            "path": str(historical_path),
            "sha256": sha256_file(historical_path),
            "fingerprint": historical.split_fingerprint,
            "n_valid_endpoints": len(valid_t),
            "n_sampled_train": len(historical.train_t),
            "n_sampled_heldout": len(historical.val_t),
        },
        "protocol": protocol,
        "splits": split_records,
        "heldout_partition_by_stock": heldout_partition,
        "stock_days": stock_days,
        "stock_day_list_sha256": canonical_json_sha256(
            {"stock_days": stock_days}
        ),
        "checks": {
            "valid_t_exactly_reconstructed": True,
            "historical_sample_membership_verified": True,
            "all_historical_train_stock_days_retained": True,
            "no_historical_train_day_moved_to_test": True,
            "stock_day_disjoint": True,
            "endpoint_disjoint": True,
            "complete_coverage": int(sum(
                record["n_rows"] for record in split_records.values()
            )) == len(valid_t),
        },
    }
    manifest["split_protocol_fingerprint"] = canonical_json_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_preregistered_split(
    split_dir: str | Path,
) -> tuple[Path, Mapping[str, Any]]:
    root = Path(split_dir).resolve()
    path = root / "split_manifest.json"
    if not path.is_file():
        raise ExperimentIntegrityError("preregistered split manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_name") != SPLIT3_SCHEMA
        or manifest.get("schema_version") != SPLIT3_SCHEMA_VERSION
        or manifest.get("status") != "preregistered"
    ):
        raise ExperimentIntegrityError("three-way split manifest is invalid")
    recorded_fingerprint = manifest.get("split_protocol_fingerprint")
    payload = dict(manifest)
    payload.pop("split_protocol_fingerprint", None)
    if canonical_json_sha256(payload) != recorded_fingerprint:
        raise ExperimentIntegrityError("three-way split manifest fingerprint mismatch")
    for split in SPLIT_NAMES:
        record = manifest.get("splits", {}).get(split, {})
        row_path = root / str(record.get("path", ""))
        if not row_path.is_file() or sha256_file(row_path) != record.get("sha256"):
            raise ExperimentIntegrityError(f"{split} row manifest file mismatch")
    return root, manifest

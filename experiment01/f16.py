"""F16 preregistration, row-manifest and test-barrier utilities.

This module deliberately keeps target/feature access out of cohort creation.
The test cohort is selected from row metadata only and is sealed before any
F16 training starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiment01.io import atomic_write_json, canonical_json_sha256, sha256_array, sha256_file


F16_SCHEMA_VERSION = 1
CAP_CANDIDATES = (128, 256, 512, 1024)
MAX_CAP = max(CAP_CANDIDATES)
BUDGETS = ("b_1_4", "b_1", "b_4", "b_16")
BUDGET_EXPECTATIONS = {
    "b_1_4": {"rows": 7_116, "stock_days": 7, "equivalents": 1.75},
    "b_1": {"rows": 28_446, "stock_days": 7, "equivalents": 7.0},
    "b_4": {"rows": 122_099, "stock_days": 28, "equivalents": 28.0},
    "b_16": {"rows": 490_937, "stock_days": 112, "equivalents": 112.0},
}
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
LABEL_COLUMNS = ("source_row_position", *ROW_COLUMNS)
COHORT_DOMAIN = b"experiment01-f16-cohort-v1"


class F16IntegrityError(RuntimeError):
    """A fail-closed F16 identity or protocol violation."""


def sha256_string_sequence(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"thesis.experiment01.string-sequence.v1\0")
    count = 0
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        count += 1
    digest.update(count.to_bytes(8, "little"))
    return digest.hexdigest()


def _cohort_digest(split: str, row_key: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(COHORT_DOMAIN)
    digest.update(b"\0")
    digest.update(split.encode("utf-8"))
    digest.update(b"\0")
    digest.update(row_key.encode("utf-8"))
    return digest.digest()


def select_nested_cohort(
    rows: pd.DataFrame,
    split: str,
    max_cap: int = MAX_CAP,
) -> pd.DataFrame:
    """Select a nested, target-blind max-cap cohort from every stock-day."""
    missing = set(ROW_COLUMNS) - set(rows.columns)
    if missing:
        raise F16IntegrityError(f"{split} rows missing columns {sorted(missing)}")
    if max_cap <= 0:
        raise ValueError("max_cap must be positive")
    if rows["row_key"].astype(str).duplicated().any():
        raise F16IntegrityError(f"{split} row keys are not unique")
    source = rows.loc[:, ROW_COLUMNS].copy()
    source.insert(0, "source_row_position", np.arange(len(source), dtype=np.int64))
    selected: list[pd.DataFrame] = []
    groups = source.groupby(["stock_id", "trading_date"], sort=False, observed=True)
    for _identity, group in groups:
        keys = group["row_key"].astype(str).tolist()
        digests = [_cohort_digest(split, key) for key in keys]
        order = sorted(range(len(group)), key=lambda index: (digests[index], keys[index]))
        count = min(max_cap, len(group))
        chosen = order[:count]
        value = group.iloc[chosen].copy()
        value["cohort_rank"] = np.arange(count, dtype=np.int32)
        value["cohort_digest"] = [digests[index].hex() for index in chosen]
        value["split"] = split
        selected.append(value)
    if not selected:
        raise F16IntegrityError(f"{split} cohort is empty")
    cohort = pd.concat(selected, ignore_index=True)
    cohort = cohort.sort_values("source_row_position", kind="stable").reset_index(drop=True)
    if cohort.duplicated("row_key").any():
        raise F16IntegrityError(f"{split} cohort contains duplicate row keys")
    expected_groups = rows[["stock_id", "trading_date"]].drop_duplicates().shape[0]
    actual_groups = cohort[["stock_id", "trading_date"]].drop_duplicates().shape[0]
    if actual_groups != expected_groups:
        raise F16IntegrityError(
            f"{split} cohort covers {actual_groups} stock-days, expected {expected_groups}"
        )
    return cohort


def cohort_for_cap(cohort: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap not in CAP_CANDIDATES:
        raise ValueError(f"cap {cap} is outside {CAP_CANDIDATES}")
    return cohort.loc[cohort["cohort_rank"].astype(int) < cap].copy()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise F16IntegrityError(f"path is outside repository root: {path}") from exc


def _parquet_record(path: Path, frame: pd.DataFrame, repo_root: Path) -> dict[str, Any]:
    endpoint_index = frame["endpoint_index"].to_numpy(dtype=np.int64, copy=False)
    source_positions = frame["source_row_position"].to_numpy(dtype=np.int64, copy=False)
    return {
        "path": _relative(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": list(frame.columns),
        "row_key_sequence_sha256": sha256_string_sequence(frame["row_key"].astype(str)),
        "endpoint_index_sha256": sha256_array(endpoint_index),
        "source_row_position_sha256": sha256_array(source_positions),
        "stock_days": int(
            frame[["stock_id", "trading_date"]].drop_duplicates().shape[0]
        ),
        "first_trading_date": str(frame["trading_date"].min()),
        "last_trading_date": str(frame["trading_date"].max()),
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "working_tree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "working_tree_status": status,
    }


def _load_bundle_rows(bundle_root: Path) -> tuple[Mapping[str, Any], dict[str, pd.DataFrame]]:
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file():
        raise F16IntegrityError(f"missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise F16IntegrityError("production bundle is not complete")
    rows: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        record = manifest["splits"][split]
        path = bundle_root / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise F16IntegrityError(f"bundle row file hash mismatch: {split}")
        frame = pd.read_parquet(path)
        missing = set(ROW_COLUMNS) - set(frame.columns)
        if missing:
            raise F16IntegrityError(f"bundle {split} rows missing {sorted(missing)}")
        if len(frame) != int(record["n_rows"]):
            raise F16IntegrityError(f"bundle {split} row count mismatch")
        rows[split] = frame.loc[:, ROW_COLUMNS].copy()
    return manifest, rows


def _validate_and_copy_labels(
    repo_root: Path,
    train_rows: pd.DataFrame,
    subset_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    records: dict[str, Any] = {}
    key_sets: dict[str, set[str]] = {}
    for budget in BUDGETS:
        source_path = subset_root / budget / "seed_000.parquet"
        if not source_path.is_file():
            raise F16IntegrityError(f"missing Phase-I label manifest: {source_path}")
        source = pd.read_parquet(source_path)
        missing = set(LABEL_COLUMNS) - set(source.columns)
        if missing:
            raise F16IntegrityError(f"{source_path} missing {sorted(missing)}")
        label = source.loc[:, LABEL_COLUMNS].copy()
        expectation = BUDGET_EXPECTATIONS[budget]
        if len(label) != expectation["rows"]:
            raise F16IntegrityError(
                f"{budget} has {len(label)} rows, expected {expectation['rows']}"
            )
        stock_days = label[["stock_id", "trading_date"]].drop_duplicates().shape[0]
        if stock_days != expectation["stock_days"]:
            raise F16IntegrityError(
                f"{budget} has {stock_days} stock-days, expected {expectation['stock_days']}"
            )
        positions = label["source_row_position"].to_numpy(dtype=np.int64)
        if len(np.unique(positions)) != len(positions):
            raise F16IntegrityError(f"{budget} has duplicate source positions")
        if len(positions) and (positions.min() < 0 or positions.max() >= len(train_rows)):
            raise F16IntegrityError(f"{budget} source positions are out of range")
        canonical = train_rows.iloc[positions].reset_index(drop=True)
        for column in ROW_COLUMNS:
            left = label[column].astype(str).to_numpy()
            right = canonical[column].astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise F16IntegrityError(f"{budget} differs from train rows in {column}")
        if label["row_key"].astype(str).duplicated().any():
            raise F16IntegrityError(f"{budget} has duplicate row keys")
        key_sets[budget] = set(label["row_key"].astype(str))
        destination = output_root / "labels" / f"{budget}.parquet"
        _atomic_write_parquet(label, destination)
        record = _parquet_record(destination, label, repo_root)
        record.update(
            {
                "budget": budget,
                "stock_day_equivalents": expectation["equivalents"],
                "source_path": _relative(source_path, repo_root),
                "source_sha256": sha256_file(source_path),
                "subsample_seed": 0,
            }
        )
        records[budget] = record

    for smaller, larger in zip(BUDGETS[:-1], BUDGETS[1:]):
        if not key_sets[smaller].issubset(key_sets[larger]):
            raise F16IntegrityError(f"label budgets are not nested: {smaller} -> {larger}")
    return records, key_sets


def _cohort_records(
    repo_root: Path,
    rows: Mapping[str, pd.DataFrame],
    output_root: Path,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        cohort = select_nested_cohort(rows[split], split=split, max_cap=MAX_CAP)
        path = output_root / "cohorts" / f"{split}_max{MAX_CAP}.parquet"
        _atomic_write_parquet(cohort, path)
        record = _parquet_record(path, cohort, repo_root)
        record.update(
            {
                "split": split,
                "max_cap_per_stock_day": MAX_CAP,
                "candidate_rows": {
                    str(cap): int((cohort["cohort_rank"].astype(int) < cap).sum())
                    for cap in CAP_CANDIDATES
                },
                "selection_domain": COHORT_DOMAIN.decode("ascii"),
                "outcome_arrays_accessed": False,
            }
        )
        records[split] = record
    return records


def freeze_f16_candidates(
    repo_root: Path,
    bundle_root: Path,
    subset_root: Path,
    spec_path: Path,
    training_audit_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze F16 label and target-blind cohort candidate manifests."""
    repo_root = repo_root.resolve()
    bundle_root = bundle_root.resolve()
    subset_root = subset_root.resolve()
    spec_path = spec_path.resolve()
    training_audit_path = training_audit_path.resolve()
    output_root = output_root.resolve()
    for required in (spec_path, training_audit_path):
        if not required.is_file():
            raise F16IntegrityError(f"missing preregistration input: {required}")
    training_audit = json.loads(training_audit_path.read_text(encoding="utf-8"))
    if training_audit.get("status") != "passed" or training_audit.get("failures"):
        raise F16IntegrityError("training protocol audit has not passed")

    bundle_manifest, rows = _load_bundle_rows(bundle_root)
    output_root.mkdir(parents=True, exist_ok=True)
    label_records, _ = _validate_and_copy_labels(
        repo_root, rows["train"], subset_root, output_root
    )
    cohort_records = _cohort_records(repo_root, rows, output_root)

    candidate_manifest: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_cohort_candidates",
        "schema_version": F16_SCHEMA_VERSION,
        "status": "frozen_pending_convergence",
        "specification": {
            "path": _relative(spec_path, repo_root),
            "sha256": sha256_file(spec_path),
        },
        "inputs": {
            "bundle_manifest_path": _relative(bundle_root / "manifest.json", repo_root),
            "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
            "dataset_sha256": bundle_manifest["provenance"]["dataset_sha256"],
            "training_audit_path": _relative(training_audit_path, repo_root),
            "training_audit_sha256": sha256_file(training_audit_path),
        },
        "candidate_caps_per_stock_day": list(CAP_CANDIDATES),
        "selection_rule": (
            "within each split and (stock_id,trading_date), sort by SHA-256 of "
            "domain + NUL + split + NUL + row_key; take rank < cap"
        ),
        "canonical_stock_day_identity": ["stock_id", "trading_date"],
        "label_budgets": label_records,
        "cohorts": cohort_records,
        "label_budgets_nested": True,
        "test_barrier": {
            "status": "locked",
            "test_row_metadata_accessed": True,
            "test_targets_accessed": False,
            "test_features_accessed": False,
            "test_statistics_accessed": False,
        },
        "failures": [],
    }
    candidate_manifest["manifest_fingerprint"] = canonical_json_sha256(candidate_manifest)
    candidate_path = output_root / "f16_cohort_candidates_manifest.json"
    atomic_write_json(candidate_path, candidate_manifest)

    git = _git_state(repo_root)
    freeze_dir = output_root / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    (freeze_dir / "git_commit.txt").write_text(git["commit"] + "\n", encoding="utf-8")
    (freeze_dir / "working_tree_status.txt").write_text(
        git["working_tree_status"] + ("\n" if git["working_tree_status"] else ""),
        encoding="utf-8",
    )
    (freeze_dir / "spec.sha256").write_text(
        f"{sha256_file(spec_path)}  {_relative(spec_path, repo_root)}\n", encoding="utf-8"
    )

    protocol_manifest: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_protocol",
        "schema_version": F16_SCHEMA_VERSION,
        "status": "preregistered_candidates_frozen",
        "git": {key: value for key, value in git.items() if key != "working_tree_status"},
        "specification": candidate_manifest["specification"],
        "training_protocol_audit": candidate_manifest["inputs"]["training_audit_path"],
        "training_protocol_audit_sha256": candidate_manifest["inputs"][
            "training_audit_sha256"
        ],
        "bundle_manifest_sha256": candidate_manifest["inputs"]["bundle_manifest_sha256"],
        "cohort_candidates_manifest": _relative(candidate_path, repo_root),
        "cohort_candidates_manifest_sha256": sha256_file(candidate_path),
        "budgets": list(BUDGETS),
        "encoder_seeds": [0, 1, 2],
        "new_training_cells": 12,
        "maximum_optimizer_updates": 39_060,
        "validation_cadence_updates": 500,
        "patience_validation_checks": 8,
        "minimum_improvement_mse": 1e-4,
        "test_barrier": "locked",
        "cohort_selected_cap": None,
        "production_grid_authorized": False,
        "failures": [],
    }
    protocol_manifest["manifest_fingerprint"] = canonical_json_sha256(protocol_manifest)
    atomic_write_json(output_root / "f16_manifest.json", protocol_manifest)
    return protocol_manifest, candidate_manifest

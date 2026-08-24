"""Predictability-dependent spectral-allocation diagnostic.

This module implements the prospective diagnostic proposed for thesis section
14.5 without modifying Experiments 01 Phase I--III.  It consumes the frozen
post-P0 100k/50k readout artifacts and their 17 held-out targets.

The two quantities are deliberately reader- and source-qualified:

``P_raw_linear``
    Out-of-sample R2 of a trace-normalized ridge on the exact normalized raw
    K-window supplied to the encoders.  Alpha is selected by grouped
    cross-validation inside the historical train split; the historical
    validation split is evaluation-only.

``F_top_k``
    The fraction of full-rank linear predictive mass allocated to the first k
    covariance eigenvectors.  The matched Haar null is computed on this same
    fraction; it is never mixed with a random-subspace test-R2 null.

The historical targets have already appeared in exploratory work.  The
protocol therefore records this analysis as prospective-after-exploration,
not as a pristine confirmatory test.  A run is refused unless its protocol has
explicitly been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from experiment01.historical.analysis_artifacts import (
    canonical_sha256 as historical_canonical_sha256,
    sha256_array as historical_sha256_array,
)

from .constants import ALPHA_GRID, BRANCHES
from .errors import ExperimentIntegrityError
from .io import (
    atomic_savez,
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_sha256,
    sha256_file,
)
from .linear import (
    SufficientStats,
    evaluate_stats,
    fit_alpha,
    sufficient_stats,
    transformed_design,
)


SCHEMA_NAME = "thesis.experiment01.predictability_allocation"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "1.0"
READOUT = "last_concat512"
EXPECTED_DIMENSION = 512
EXPECTED_VALID_RANK = 508
EXPECTED_K = 20
EXPECTED_N_TRAIN = 100_000
EXPECTED_N_VALIDATION = 50_000
EXPECTED_N_VALID_TOTAL = 7_323_510
EXPECTED_N_DATASET_ROWS = 8_039_246
EXPECTED_N_STOCKS = 7
RAW_INPUT_DIMENSION = EXPECTED_K * 2 * 10 * 2
MIN_TRAIN_OBSERVATIONS_PER_DIMENSION = 100.0
MIN_TRAIN_STOCK_DAYS = 1_000
MIN_VALIDATION_STOCK_DAYS = 100
MIN_VALID_STOCK_DAY_COVERAGE = 0.99
MIN_TRAIN_ROWS_PER_STOCK = 10_000
MIN_VALIDATION_ROWS_PER_STOCK = 5_000
MIN_RAW_CV_STOCK_DAYS_PER_FOLD = 100
MIN_RAW_CV_ROWS_PER_FOLD = 10_000
MIN_SAMPLE_FRACTION = 0.01
MAX_SAMPLE_FRACTION = 0.05
EXPECTED_TARGET_NAMES = (
    "d_imbalance_top5@1",
    "d_imbalance_top5@5",
    "d_imbalance_top5@10",
    "d_imbalance_top5@20",
    "d_imbalance_all@1",
    "d_imbalance_all@5",
    "d_imbalance_all@10",
    "d_imbalance_all@20",
    "d_log_depth_top5@1",
    "d_log_depth_top5@5",
    "d_log_depth_top5@10",
    "d_log_depth_top5@20",
    "d_log_depth_all@1",
    "d_log_depth_all@5",
    "d_log_depth_all@10",
    "d_log_depth_all@20",
    "time_to_next_mid_move",
)


def target_family(name: str) -> str:
    value = str(name)
    if value.startswith("d_imbalance_"):
        return "imbalance"
    if value.startswith("d_log_depth_"):
        return "depth"
    if value == "time_to_next_mid_move":
        return "timing"
    raise ValueError(f"unknown held-out target {value!r}")


def target_horizon(name: str) -> int | None:
    value = str(name)
    if "@" not in value:
        return None
    return int(value.rsplit("@", 1)[1])


def validate_sample_contract(
    n_train: int,
    n_validation: int,
    n_valid_total: int,
    *,
    n_dataset_rows: int = EXPECTED_N_DATASET_ROWS,
) -> Mapping[str, Any]:
    """Fail closed unless the frozen, deliberately fractional sample is used."""
    observed = (int(n_train), int(n_validation), int(n_valid_total))
    expected = (EXPECTED_N_TRAIN, EXPECTED_N_VALIDATION, EXPECTED_N_VALID_TOTAL)
    if observed != expected or int(n_dataset_rows) != EXPECTED_N_DATASET_ROWS:
        raise ExperimentIntegrityError(
            "sample does not match the frozen 100k-train/50k-validation contract"
        )
    selected = int(n_train) + int(n_validation)
    valid_fraction = selected / int(n_valid_total)
    dataset_fraction = selected / int(n_dataset_rows)
    if not MIN_SAMPLE_FRACTION <= valid_fraction <= MAX_SAMPLE_FRACTION:
        raise ExperimentIntegrityError(
            "selected endpoint fraction is too small or accidentally near-full"
        )
    if selected >= n_valid_total or selected >= n_dataset_rows:
        raise ExperimentIntegrityError("full-dataset fitting is forbidden")
    return {
        "n_train": int(n_train),
        "n_validation": int(n_validation),
        "n_selected": selected,
        "n_valid_total": int(n_valid_total),
        "n_dataset_rows": int(n_dataset_rows),
        "fraction_of_valid_endpoints": valid_fraction,
        "fraction_of_dataset_rows": dataset_fraction,
        "fit_selected_endpoints_only": True,
        "full_dataset_fit": False,
    }


def sample_coverage_diagnostics(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    valid_endpoints: np.ndarray,
    train_endpoints: np.ndarray,
    validation_endpoints: np.ndarray,
) -> Mapping[str, Any]:
    """Check that the fractional sample remains broad across stocks and days."""
    stocks = np.asarray(stock_ids, dtype=np.int64)
    days = np.asarray(day_ids, dtype=np.int64)
    valid = np.asarray(valid_endpoints, dtype=np.int64)
    train = np.asarray(train_endpoints, dtype=np.int64)
    validation = np.asarray(validation_endpoints, dtype=np.int64)
    for label, endpoint in (
        ("valid", valid),
        ("train", train),
        ("validation", validation),
    ):
        if endpoint.ndim != 1 or len(endpoint) == 0:
            raise ExperimentIntegrityError(f"{label} endpoints are empty or malformed")
        if int(endpoint.min()) < 0 or int(endpoint.max()) >= len(stocks):
            raise ExperimentIntegrityError(f"{label} endpoint is out of bounds")
        if len(np.unique(endpoint)) != len(endpoint):
            raise ExperimentIntegrityError(f"{label} endpoints contain duplicates")
    if len(stocks) != len(days):
        raise ExperimentIntegrityError("stock/day arrays are misaligned")
    if np.intersect1d(train, validation, assume_unique=True).size:
        raise ExperimentIntegrityError("train and validation endpoints overlap")

    expected_stocks = np.arange(EXPECTED_N_STOCKS, dtype=np.int64)
    by_split: dict[str, Any] = {}
    group_sets: dict[str, set[tuple[int, int]]] = {}
    for label, endpoint in (
        ("valid", valid),
        ("train", train),
        ("validation", validation),
    ):
        split_stocks = stocks[endpoint]
        observed_stocks = np.unique(split_stocks)
        if not np.array_equal(observed_stocks, expected_stocks):
            raise ExperimentIntegrityError(f"{label} does not cover all seven stocks")
        pairs = np.unique(
            np.column_stack([split_stocks, days[endpoint]]), axis=0
        )
        group_sets[label] = {
            (int(stock), int(day)) for stock, day in pairs.tolist()
        }
        by_split[label] = {
            "n_rows": int(len(endpoint)),
            "n_stock_days": int(len(pairs)),
            "rows_by_stock": np.bincount(
                split_stocks, minlength=EXPECTED_N_STOCKS
            ).astype(int).tolist(),
            "stock_days_by_stock": np.bincount(
                pairs[:, 0], minlength=EXPECTED_N_STOCKS
            ).astype(int).tolist(),
        }
    if group_sets["train"] & group_sets["validation"]:
        raise ExperimentIntegrityError("train and validation stock-days overlap")
    selected_groups = group_sets["train"] | group_sets["validation"]
    if not selected_groups.issubset(group_sets["valid"]):
        raise ExperimentIntegrityError(
            "fractional sample contains a stock-day outside the valid inventory"
        )
    missing_groups = group_sets["valid"] - selected_groups
    stock_day_coverage = len(selected_groups) / len(group_sets["valid"])
    if stock_day_coverage < MIN_VALID_STOCK_DAY_COVERAGE:
        raise ExperimentIntegrityError("valid stock-day coverage is too narrow")
    missing_valid_rows = int(
        sum(
            np.count_nonzero(
                (stocks[valid] == stock) & (days[valid] == day)
            )
            for stock, day in missing_groups
        )
    )
    if by_split["train"]["n_stock_days"] < MIN_TRAIN_STOCK_DAYS:
        raise ExperimentIntegrityError("too few training stock-days")
    if by_split["validation"]["n_stock_days"] < MIN_VALIDATION_STOCK_DAYS:
        raise ExperimentIntegrityError("too few validation stock-days")
    if min(by_split["train"]["rows_by_stock"]) < MIN_TRAIN_ROWS_PER_STOCK:
        raise ExperimentIntegrityError("too few training rows for at least one stock")
    if (
        min(by_split["validation"]["rows_by_stock"])
        < MIN_VALIDATION_ROWS_PER_STOCK
    ):
        raise ExperimentIntegrityError(
            "too few validation rows for at least one stock"
        )
    raw_ratio = len(train) / RAW_INPUT_DIMENSION
    representation_ratio = len(train) / EXPECTED_DIMENSION
    if min(raw_ratio, representation_ratio) < MIN_TRAIN_OBSERVATIONS_PER_DIMENSION:
        raise ExperimentIntegrityError("too few training observations per dimension")
    return {
        "passed": True,
        "all_stocks_in_each_split": True,
        "stock_day_disjoint": True,
        "all_valid_stock_days_represented": not missing_groups,
        "valid_stock_day_coverage": stock_day_coverage,
        "minimum_valid_stock_day_coverage": MIN_VALID_STOCK_DAY_COVERAGE,
        "missing_valid_stock_days": [list(value) for value in sorted(missing_groups)],
        "missing_valid_endpoint_rows": missing_valid_rows,
        "raw_train_rows_over_dimension": raw_ratio,
        "representation_train_rows_over_dimension": representation_ratio,
        "minimum_train_rows_over_dimension": (
            MIN_TRAIN_OBSERVATIONS_PER_DIMENSION
        ),
        "splits": by_split,
    }


@dataclass(frozen=True)
class AllocationProtocol:
    status: str
    top_k: int = 8
    haar_draws: int = 999
    haar_seed: int = 20260818
    raw_cv_folds: int = 5
    raw_cv_seed: int = 20260818
    raw_chunk_rows: int = 2048
    feature_chunk_rows: int = 8192
    full_mass_floor: float = 1e-12
    low_predictability_fraction: float = 1.0 / 3.0
    below_null_quantile: float = 0.05
    rho_horizon_min_each_seed: float = 0.6
    delta_rho_min_mean: float = 0.3
    min_horizon_below_null_low_p: int = 2
    max_supervised_below_null_low_p: int = 0

    def validate(self, *, require_frozen: bool = False) -> None:
        if self.status not in {"draft", "frozen"}:
            raise ValueError("protocol status must be draft or frozen")
        if require_frozen and self.status != "frozen":
            raise ExperimentIntegrityError(
                "predictability-allocation protocol is not frozen"
            )
        if not 0 < self.top_k <= EXPECTED_VALID_RANK:
            raise ValueError("top_k is outside the valid spectrum")
        if self.haar_draws < 99:
            raise ValueError("at least 99 matched Haar draws are required")
        if self.raw_cv_folds < 2:
            raise ValueError("raw_cv_folds must be at least two")
        if self.raw_chunk_rows <= 0 or self.feature_chunk_rows <= 0:
            raise ValueError("chunk sizes must be positive")
        if not 0.0 < self.full_mass_floor < 1.0:
            raise ValueError("full_mass_floor must lie in (0,1)")
        if not 0.0 < self.low_predictability_fraction < 1.0:
            raise ValueError("low_predictability_fraction must lie in (0,1)")
        if not 0.0 < self.below_null_quantile < 0.5:
            raise ValueError("below_null_quantile must lie in (0,0.5)")
        if not -1.0 <= self.rho_horizon_min_each_seed <= 1.0:
            raise ValueError("invalid rho threshold")
        if not -2.0 <= self.delta_rho_min_mean <= 2.0:
            raise ValueError("invalid delta-rho threshold")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AllocationProtocol":
        if value.get("schema_name") != SCHEMA_NAME:
            raise ExperimentIntegrityError("protocol schema differs")
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ExperimentIntegrityError("protocol schema version differs")
        analysis = value.get("analysis")
        thresholds = value.get("thresholds")
        if not isinstance(analysis, Mapping) or not isinstance(thresholds, Mapping):
            raise ExperimentIntegrityError("protocol sections are missing")
        result = cls(
            status=str(value.get("status")),
            top_k=int(analysis["top_k"]),
            haar_draws=int(analysis["haar_draws"]),
            haar_seed=int(analysis["haar_seed"]),
            raw_cv_folds=int(analysis["raw_cv_folds"]),
            raw_cv_seed=int(analysis["raw_cv_seed"]),
            raw_chunk_rows=int(analysis["raw_chunk_rows"]),
            feature_chunk_rows=int(analysis["feature_chunk_rows"]),
            full_mass_floor=float(analysis["full_mass_floor"]),
            low_predictability_fraction=float(
                thresholds["low_predictability_fraction"]
            ),
            below_null_quantile=float(thresholds["below_null_quantile"]),
            rho_horizon_min_each_seed=float(
                thresholds["rho_horizon_min_each_seed"]
            ),
            delta_rho_min_mean=float(thresholds["delta_rho_min_mean"]),
            min_horizon_below_null_low_p=int(
                thresholds["min_horizon_below_null_low_p"]
            ),
            max_supervised_below_null_low_p=int(
                thresholds["max_supervised_below_null_low_p"]
            ),
        )
        result.validate()
        return result


def default_protocol_payload(*, status: str = "draft") -> dict[str, Any]:
    protocol = AllocationProtocol(status=status)
    protocol.validate()
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "scientific_status": "prospective_after_historical_exploration",
        "changes_phase1_phase2_phase3": False,
        "scope": {
            "branches": list(BRANCHES),
            "confirmatory_contrast": ["jepa_horizon", "supervised"],
            "jepa_masked_role": "descriptive_only",
            "encoder_seeds": [0, 1, 2],
            "readout": READOUT,
            "target_names": list(EXPECTED_TARGET_NAMES),
            "target_families": ["imbalance", "depth", "timing"],
        },
        "estimands": {
            "P_j": (
                "validation R2 of trace-normalized ridge on the exact normalized "
                "raw K=20 LOB window; alpha selected per target by grouped "
                "cross-validation inside train"
            ),
            "M_j": "F_top_k=M(top_k)/M(D_valid)",
            "haar_null": "matched distribution of F=M(Q)/M(D_valid)",
            "null_percentile": "(1 + count(F_haar <= F_top_k))/(B + 1)",
        },
        "analysis": {
            "top_k": protocol.top_k,
            "haar_draws": protocol.haar_draws,
            "haar_seed": protocol.haar_seed,
            "raw_cv_folds": protocol.raw_cv_folds,
            "raw_cv_seed": protocol.raw_cv_seed,
            "raw_chunk_rows": protocol.raw_chunk_rows,
            "feature_chunk_rows": protocol.feature_chunk_rows,
            "full_mass_floor": protocol.full_mass_floor,
            "raw_input": "train_only_normalized_full_window_K20_flattened",
            "sample_contract": {
                "n_train": EXPECTED_N_TRAIN,
                "n_validation": EXPECTED_N_VALIDATION,
                "n_valid_total": EXPECTED_N_VALID_TOTAL,
                "n_dataset_rows": EXPECTED_N_DATASET_ROWS,
                "selected_fraction_min": MIN_SAMPLE_FRACTION,
                "selected_fraction_max": MAX_SAMPLE_FRACTION,
                "fit_selected_endpoints_only": True,
                "full_dataset_fit_forbidden": True,
                "all_seven_stocks_each_split": True,
                "stock_day_disjoint": True,
                "minimum_valid_stock_day_coverage": (
                    MIN_VALID_STOCK_DAY_COVERAGE
                ),
                "minimum_train_stock_days": MIN_TRAIN_STOCK_DAYS,
                "minimum_validation_stock_days": MIN_VALIDATION_STOCK_DAYS,
                "minimum_train_rows_per_stock": MIN_TRAIN_ROWS_PER_STOCK,
                "minimum_validation_rows_per_stock": (
                    MIN_VALIDATION_ROWS_PER_STOCK
                ),
                "minimum_train_rows_over_dimension": (
                    MIN_TRAIN_OBSERVATIONS_PER_DIMENSION
                ),
                "note": (
                    "the compressed source NPZ may be opened to gather the frozen "
                    "endpoint windows; covariance, cross-covariance, PCA and ridge "
                    "are fit only on the selected endpoints"
                ),
            },
            "ridge_regularization": "lambda=alpha*trace(covariance)/D",
            "alpha_grid": [float(value) for value in ALPHA_GRID],
        },
        "thresholds": {
            "low_predictability_fraction": protocol.low_predictability_fraction,
            "below_null_quantile": protocol.below_null_quantile,
            "rho_horizon_min_each_seed": protocol.rho_horizon_min_each_seed,
            "delta_rho_min_mean": protocol.delta_rho_min_mean,
            "min_horizon_below_null_low_p": (
                protocol.min_horizon_below_null_low_p
            ),
            "max_supervised_below_null_low_p": (
                protocol.max_supervised_below_null_low_p
            ),
            "approval_status": "requires_scientific_approval_before_freeze",
        },
        "interpretation": {
            "supported_if_passed": (
                "predictability-dependent top-spectral allocation with low-P "
                "anti-alignment for the frozen encoders"
            ),
            "not_identified": [
                "a discontinuous SNR threshold",
                "Bayes predictability",
                "causal contribution of the training objective alone",
            ],
            "family_analysis": (
                "sensitivity analysis; n=3 family medians has no standalone "
                "inferential p-value"
            ),
        },
    }


def load_protocol(path: str | Path, *, require_frozen: bool = False) -> tuple[
    AllocationProtocol, Mapping[str, Any]
]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    protocol = AllocationProtocol.from_mapping(payload)
    protocol.validate(require_frozen=require_frozen)
    scope = payload.get("scope", {})
    analysis = payload.get("analysis", {})
    sample = analysis.get("sample_contract", {})
    if tuple(scope.get("branches", ())) != tuple(BRANCHES):
        raise ExperimentIntegrityError("protocol branch inventory differs")
    if tuple(scope.get("encoder_seeds", ())) != (0, 1, 2):
        raise ExperimentIntegrityError("protocol encoder-seed inventory differs")
    if tuple(scope.get("target_names", ())) != EXPECTED_TARGET_NAMES:
        raise ExperimentIntegrityError("protocol target inventory differs")
    if scope.get("readout") != READOUT:
        raise ExperimentIntegrityError("protocol readout differs")
    expected_sample = {
        "n_train": EXPECTED_N_TRAIN,
        "n_validation": EXPECTED_N_VALIDATION,
        "n_valid_total": EXPECTED_N_VALID_TOTAL,
        "n_dataset_rows": EXPECTED_N_DATASET_ROWS,
    }
    if any(sample.get(key) != value for key, value in expected_sample.items()):
        raise ExperimentIntegrityError("protocol sample contract differs")
    if not sample.get("full_dataset_fit_forbidden"):
        raise ExperimentIntegrityError("protocol does not forbid full-dataset fitting")
    if not np.array_equal(
        np.asarray(analysis.get("alpha_grid", ()), dtype=np.float64),
        ALPHA_GRID,
    ):
        raise ExperimentIntegrityError("protocol alpha grid differs")
    if require_frozen:
        thresholds = payload.get("thresholds", {})
        freeze = payload.get("freeze", {})
        if thresholds.get("approval_status") != "approved_before_outcome_access":
            raise ExperimentIntegrityError(
                "frozen protocol lacks prospective threshold approval"
            )
        if not isinstance(freeze, Mapping) or not freeze.get(
            "scientific_design_approved"
        ):
            raise ExperimentIntegrityError("scientific design was not approved")
        inventory_hash = freeze.get("input_inventory_sha256")
        if not isinstance(inventory_hash, str) or len(inventory_hash) != 64:
            raise ExperimentIntegrityError("frozen protocol is not input-bound")
    return protocol, payload


def freeze_protocol_payload(
    draft_payload: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    *,
    scientific_approver: str,
    approved_at_utc: str,
) -> Mapping[str, Any]:
    """Bind an approved draft to audited inputs without reading outcomes."""
    protocol = AllocationProtocol.from_mapping(draft_payload)
    protocol.validate()
    if protocol.status != "draft":
        raise ExperimentIntegrityError("only a draft protocol can be frozen")
    if not scientific_approver.strip() or not approved_at_utc.strip():
        raise ValueError("approver and approval timestamp are required")
    if not input_audit.get("passed") or input_audit.get("outcomes_read") is not False:
        raise ExperimentIntegrityError("input audit is not outcome-blind and complete")
    inventory_hash = input_audit.get("inventory_sha256")
    if not isinstance(inventory_hash, str) or len(inventory_hash) != 64:
        raise ExperimentIntegrityError("input audit lacks a canonical hash")
    frozen = json.loads(json.dumps(draft_payload))
    frozen["status"] = "frozen"
    frozen["thresholds"]["approval_status"] = "approved_before_outcome_access"
    frozen["freeze"] = {
        "scientific_design_approved": True,
        "scientific_approver": scientific_approver.strip(),
        "approved_at_utc": approved_at_utc.strip(),
        "input_inventory_sha256": inventory_hash,
        "draft_payload_sha256": canonical_json_sha256(draft_payload),
        "outcomes_read_before_freeze": False,
    }
    AllocationProtocol.from_mapping(frozen).validate(require_frozen=True)
    return frozen


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExperimentIntegrityError(f"{label} path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ExperimentIntegrityError(f"{label} path is not safely relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ExperimentIntegrityError(f"{label} path escapes input root") from exc
    if not path.is_file():
        raise ExperimentIntegrityError(f"{label} file is missing: {path}")
    return path


def _verify_recorded_file(
    root: Path,
    record: Mapping[str, Any],
    label: str,
    *,
    verify_hashes: bool,
) -> Path:
    path = _safe_relative(root, record.get("path"), label)
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise ExperimentIntegrityError(f"{label} size differs from manifest")
    expected = record.get("file_sha256") or record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ExperimentIntegrityError(f"{label} has no SHA-256")
    if verify_hashes and sha256_file(path) != expected:
        raise ExperimentIntegrityError(f"{label} SHA-256 differs from manifest")
    return path


def _parse_readout_key(key: str) -> tuple[str, int]:
    for branch in BRANCHES:
        prefix = f"{branch}_seed"
        if key.startswith(prefix) and key.endswith("_ep020"):
            return branch, int(key[len(prefix) : -len("_ep020")])
    raise ExperimentIntegrityError(f"unexpected readout key {key!r}")


def audit_historical_inputs(
    input_dir: str | Path,
    dataset_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> Mapping[str, Any]:
    """Validate metadata and hashes without reading scientific outcomes."""
    root = Path(input_dir).resolve()
    manifest_path = root / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise ExperimentIntegrityError("historical analysis manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ExperimentIntegrityError("historical readout manifest is not complete")
    protocol = manifest.get("protocol", {})
    if int(protocol.get("K", -1)) != EXPECTED_K:
        raise ExperimentIntegrityError("historical readouts do not use K=20")
    dataset = Path(dataset_path).resolve()
    if not dataset.is_file():
        raise ExperimentIntegrityError("dataset is missing")
    dataset_sha = sha256_file(dataset) if verify_hashes else manifest["dataset"]["sha256"]
    if dataset_sha != manifest.get("dataset", {}).get("sha256"):
        raise ExperimentIntegrityError("dataset hash differs from historical manifest")

    split_record = manifest.get("split", {})
    split_path = _verify_recorded_file(
        root, split_record, "split", verify_hashes=verify_hashes
    )
    target_record = manifest.get("heldout_targets", {})
    target_path = _verify_recorded_file(
        root, target_record, "heldout targets", verify_hashes=verify_hashes
    )
    with np.load(split_path, allow_pickle=False) as split:
        valid_t = np.asarray(split["valid_t"], dtype=np.int64)
        train_t = np.asarray(split["train_t"], dtype=np.int64)
        val_t = np.asarray(split["val_t"], dtype=np.int64)
        n_valid_total = int(np.asarray(split["n_valid_total"]).item())
        train_endpoint_sha = str(split["train_endpoint_sha256"].item())
        val_endpoint_sha = str(split["val_endpoint_sha256"].item())
    sample = validate_sample_contract(len(train_t), len(val_t), n_valid_total)
    if len(valid_t) != n_valid_total:
        raise ExperimentIntegrityError("valid endpoint inventory is incomplete")
    with np.load(dataset, allow_pickle=False) as raw:
        stock_ids = np.asarray(raw["stock_ids"], dtype=np.int64)
        day_ids = np.asarray(raw["day_ids"], dtype=np.int64)
    if len(stock_ids) != EXPECTED_N_DATASET_ROWS:
        raise ExperimentIntegrityError("dataset row count differs")
    coverage = sample_coverage_diagnostics(
        stock_ids, day_ids, valid_t, train_t, val_t
    )
    if train_endpoint_sha != split_record.get("train_endpoint_sha256"):
        raise ExperimentIntegrityError("train endpoint hash differs")
    if val_endpoint_sha != split_record.get("val_endpoint_sha256"):
        raise ExperimentIntegrityError("validation endpoint hash differs")
    with np.load(target_path, allow_pickle=False) as heldout:
        names = tuple(str(value) for value in heldout["heldout_names"].tolist())
        target_train_sha = str(heldout["train_endpoint_sha256"].item())
        target_val_sha = str(heldout["val_endpoint_sha256"].item())
    train_shape = tuple(int(value) for value in target_record.get("shape_train", ()))
    val_shape = tuple(int(value) for value in target_record.get("shape_val", ()))
    if names != EXPECTED_TARGET_NAMES:
        raise ExperimentIntegrityError("held-out target inventory or order differs")
    if train_shape != (len(train_t), len(names)) or val_shape != (
        len(val_t),
        len(names),
    ):
        raise ExperimentIntegrityError("held-out target shapes differ")
    if target_train_sha != train_endpoint_sha or target_val_sha != val_endpoint_sha:
        raise ExperimentIntegrityError("held-out targets are endpoint-misaligned")

    records = manifest.get("readouts")
    if not isinstance(records, Mapping) or len(records) != 9:
        raise ExperimentIntegrityError("exactly nine readout artifacts are required")
    observed_keys: set[tuple[str, int]] = set()
    readout_rows = []
    stock_stats_hashes = set()
    for key, raw_record in sorted(records.items()):
        if not isinstance(raw_record, Mapping):
            raise ExperimentIntegrityError("readout record is malformed")
        branch, seed = _parse_readout_key(str(key))
        observed_keys.add((branch, seed))
        path = _verify_recorded_file(
            root, raw_record, f"readout {key}", verify_hashes=verify_hashes
        )
        shape = raw_record.get("arrays", {}).get(
            f"{READOUT}_train", {}
        ).get("shape")
        if shape != [len(train_t), EXPECTED_DIMENSION]:
            raise ExperimentIntegrityError(f"readout {key} has wrong train shape")
        if raw_record.get("train_endpoint_sha256") != train_endpoint_sha:
            raise ExperimentIntegrityError(f"readout {key} train endpoints differ")
        if raw_record.get("val_endpoint_sha256") != val_endpoint_sha:
            raise ExperimentIntegrityError(f"readout {key} validation endpoints differ")
        stock_stats_hashes.add(str(raw_record.get("stock_stats_sha256")))
        readout_rows.append(
            {
                "key": key,
                "branch": branch,
                "encoder_seed": seed,
                "path": str(path),
                "file_sha256": raw_record.get("file_sha256"),
            }
        )
    expected_keys = {(branch, seed) for branch in BRANCHES for seed in (0, 1, 2)}
    if observed_keys != expected_keys:
        raise ExperimentIntegrityError("readout branch/seed inventory differs")
    if len(stock_stats_hashes) != 1 or len(next(iter(stock_stats_hashes))) != 64:
        raise ExperimentIntegrityError("readouts do not share one stock-stats hash")
    checkpoint_records = manifest.get("requested_checkpoints", {})
    checkpoint_candidates = [
        Path(value["path"])
        for value in checkpoint_records.values()
        if isinstance(value, Mapping)
        and value.get("stock_stats_sha256") == next(iter(stock_stats_hashes))
    ]
    if not checkpoint_candidates or not checkpoint_candidates[0].is_file():
        raise ExperimentIntegrityError("no checkpoint supplies canonical stock stats")
    inventory = {
        "schema_name": f"{SCHEMA_NAME}.input_audit",
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "input_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_sha,
        "split_path": str(split_path),
        "split_file_sha256": split_record.get("file_sha256"),
        "heldout_target_path": str(target_path),
        "heldout_target_file_sha256": target_record.get("file_sha256"),
        "target_names": list(names),
        "target_families": {name: target_family(name) for name in names},
        "n_train": len(train_t),
        "n_validation": len(val_t),
        "n_valid_total": n_valid_total,
        "n_dataset_rows": EXPECTED_N_DATASET_ROWS,
        "sample_contract": sample,
        "sample_coverage": coverage,
        "selected_endpoint_fraction": sample["fraction_of_valid_endpoints"],
        "selected_dataset_row_fraction": sample["fraction_of_dataset_rows"],
        "full_dataset_fit": False,
        "train_endpoint_sha256": train_endpoint_sha,
        "validation_endpoint_sha256": val_endpoint_sha,
        "readouts": readout_rows,
        "stock_stats_sha256": next(iter(stock_stats_hashes)),
        "stock_stats_checkpoint": str(checkpoint_candidates[0].resolve()),
        "outcomes_read": False,
        "verify_hashes": bool(verify_hashes),
    }
    return {**inventory, "inventory_sha256": canonical_json_sha256(inventory)}


def load_stock_stats(
    checkpoint_path: str | Path, *, expected_sha256: str
) -> Mapping[str, np.ndarray]:
    import torch

    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    raw = checkpoint.get("stock_stats")
    if not isinstance(raw, Mapping):
        raise ExperimentIntegrityError("checkpoint has no stock_stats mapping")
    stats = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in raw.items()
    }
    required = {
        "depth_scale_per_stock",
        "vol_min_per_stock",
        "vol_scale_per_stock",
    }
    if set(stats) != required:
        raise ExperimentIntegrityError("checkpoint stock_stats inventory differs")
    fingerprint = historical_canonical_sha256(
        {
            key: historical_sha256_array(value)
            for key, value in sorted(stats.items())
        }
    )
    if fingerprint != expected_sha256:
        raise ExperimentIntegrityError("checkpoint stock_stats fingerprint differs")
    return stats


def normalized_raw_windows(
    book: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    endpoints: np.ndarray,
    stock_stats: Mapping[str, np.ndarray],
    *,
    K: int = EXPECTED_K,
) -> np.ndarray:
    """Vectorized exact equivalent of ``normalize_book_window`` then flatten."""
    endpoint = np.asarray(endpoints, dtype=np.int64)
    if endpoint.ndim != 1 or len(endpoint) == 0:
        raise ValueError("endpoints must be a non-empty vector")
    offsets = np.arange(-(K - 1), 1, dtype=np.int64)
    indices = endpoint[:, None] + offsets[None, :]
    if int(indices.min()) < 0 or int(indices.max()) >= len(book):
        raise ExperimentIntegrityError("raw window endpoint is out of bounds")
    endpoint_stock = stock_ids[endpoint].astype(np.int64, copy=False)
    endpoint_day = day_ids[endpoint].astype(np.int64, copy=False)
    if not np.all(stock_ids[indices] == endpoint_stock[:, None]):
        raise ExperimentIntegrityError("raw window crosses a stock boundary")
    if not np.all(day_ids[indices] == endpoint_day[:, None]):
        raise ExperimentIntegrityError("raw window crosses a day boundary")
    raw = np.asarray(book[indices], dtype=np.float32).copy()
    middle = np.asarray(mid_z[indices], dtype=np.float32)
    depth = np.asarray(
        stock_stats["depth_scale_per_stock"], dtype=np.float32
    )[endpoint_stock]
    volume_min = np.asarray(
        stock_stats["vol_min_per_stock"], dtype=np.float32
    )[endpoint_stock]
    volume_scale = np.asarray(
        stock_stats["vol_scale_per_stock"], dtype=np.float32
    )[endpoint_stock]
    if np.any(depth <= 0.0) or np.any(volume_scale <= 0.0):
        raise ExperimentIntegrityError("raw normalization scale is non-positive")
    for side in range(2):
        relative = raw[:, :, side, :, 0] - middle[:, :, None]
        clip = 3.0 * depth[:, None, None]
        raw[:, :, side, :, 0] = np.clip(relative, -clip, clip) / depth[
            :, None, None
        ]
    raw[:, :, :, :, 1] = (
        raw[:, :, :, :, 1] - volume_min[:, None, None, None]
    ) / volume_scale[:, None, None, None]
    result = raw.reshape(len(endpoint), -1)
    if result.shape[1] != K * int(np.prod(book.shape[1:])):
        raise ExperimentIntegrityError("flattened raw-window dimension differs")
    if not np.isfinite(result).all():
        raise ExperimentIntegrityError("normalized raw windows contain non-finite values")
    return result


def grouped_fold_ids(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    endpoints: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    endpoint = np.asarray(endpoints, dtype=np.int64)
    groups = np.column_stack(
        [stock_ids[endpoint].astype(np.int64), day_ids[endpoint].astype(np.int64)]
    )
    unique, inverse = np.unique(groups, axis=0, return_inverse=True)
    if len(unique) < n_folds:
        raise ExperimentIntegrityError("fewer stock-days than raw CV folds")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), len(unique)]))
    order = rng.permutation(len(unique))
    assignment = np.empty(len(unique), dtype=np.int64)
    assignment[order] = np.arange(len(unique), dtype=np.int64) % n_folds
    fold = assignment[inverse]
    for value in range(n_folds):
        selected = fold == value
        if not np.any(selected):
            raise ExperimentIntegrityError("a raw CV fold is empty")
        if int(selected.sum()) < MIN_RAW_CV_ROWS_PER_FOLD:
            raise ExperimentIntegrityError("a raw CV fold has too few rows")
        fold_groups = np.unique(groups[selected], axis=0)
        if len(fold_groups) < MIN_RAW_CV_STOCK_DAYS_PER_FOLD:
            raise ExperimentIntegrityError("a raw CV fold has too few stock-days")
        if not np.array_equal(
            np.unique(fold_groups[:, 0]),
            np.arange(EXPECTED_N_STOCKS, dtype=np.int64),
        ):
            raise ExperimentIntegrityError("a raw CV fold does not cover every stock")
    return fold


def add_stats(values: Sequence[SufficientStats]) -> SufficientStats:
    if not values:
        raise ValueError("at least one sufficient-statistic object is required")
    result = SufficientStats.zeros(values[0].dimension, values[0].n_targets)
    for value in values:
        result.add(value)
    return result


def subtract_stats(total: SufficientStats, part: SufficientStats) -> SufficientStats:
    if total.dimension != part.dimension or total.n_targets != part.n_targets:
        raise ValueError("sufficient statistics are incompatible")
    if not 0 < part.n < total.n:
        raise ValueError("subtracted statistics must be a strict non-empty subset")
    return SufficientStats(
        n=total.n - part.n,
        x_sum=total.x_sum - part.x_sum,
        y_sum=total.y_sum - part.y_sum,
        xtx=total.xtx - part.xtx,
        xty=total.xty - part.xty,
        yty=total.yty - part.yty,
    )


def _stats_arrays(prefix: str, stats: SufficientStats) -> Mapping[str, np.ndarray]:
    return {
        f"{prefix}_n": np.asarray(stats.n, dtype=np.int64),
        f"{prefix}_x_sum": stats.x_sum,
        f"{prefix}_y_sum": stats.y_sum,
        f"{prefix}_xtx": stats.xtx,
        f"{prefix}_xty": stats.xty,
        f"{prefix}_yty": stats.yty,
    }


def _stats_from_npz(data: Any, prefix: str) -> SufficientStats:
    return SufficientStats(
        n=int(np.asarray(data[f"{prefix}_n"]).item()),
        x_sum=np.asarray(data[f"{prefix}_x_sum"], dtype=np.float64),
        y_sum=np.asarray(data[f"{prefix}_y_sum"], dtype=np.float64),
        xtx=np.asarray(data[f"{prefix}_xtx"], dtype=np.float64),
        xty=np.asarray(data[f"{prefix}_xty"], dtype=np.float64),
        yty=np.asarray(data[f"{prefix}_yty"], dtype=np.float64),
    )


def build_raw_statistics(
    dataset_path: str | Path,
    split_path: str | Path,
    heldout_target_path: str | Path,
    stock_stats: Mapping[str, np.ndarray],
    *,
    n_folds: int,
    fold_seed: int,
    chunk_rows: int,
) -> tuple[list[SufficientStats], SufficientStats, Mapping[str, Any]]:
    """Scan historical raw windows once and return train-fold/validation stats."""
    with np.load(split_path, allow_pickle=False) as split:
        train_t = np.asarray(split["train_t"], dtype=np.int64)
        val_t = np.asarray(split["val_t"], dtype=np.int64)
    with np.load(heldout_target_path, allow_pickle=False) as heldout:
        y_train = np.asarray(heldout["y_train_heldout"], dtype=np.float32)
        y_val = np.asarray(heldout["y_val_heldout"], dtype=np.float32)
    with np.load(dataset_path, allow_pickle=False) as raw:
        book = np.asarray(raw["book"])
        mid_z = np.asarray(raw["mid_z"])
        stock_ids = np.asarray(raw["stock_ids"], dtype=np.int64)
        day_ids = np.asarray(raw["day_ids"], dtype=np.int64)
        validate_sample_contract(
            len(train_t),
            len(val_t),
            EXPECTED_N_VALID_TOTAL,
            n_dataset_rows=len(book),
        )
        fold_id = grouped_fold_ids(
            stock_ids,
            day_ids,
            train_t,
            n_folds=n_folds,
            seed=fold_seed,
        )
        dimension = EXPECTED_K * int(np.prod(book.shape[1:]))
        folds = [
            SufficientStats.zeros(dimension, len(EXPECTED_TARGET_NAMES))
            for _ in range(n_folds)
        ]
        validation = SufficientStats.zeros(dimension, len(EXPECTED_TARGET_NAMES))
        for start in range(0, len(train_t), chunk_rows):
            stop = min(start + chunk_rows, len(train_t))
            x = normalized_raw_windows(
                book,
                mid_z,
                stock_ids,
                day_ids,
                train_t[start:stop],
                stock_stats,
            )
            y = y_train[start:stop]
            local_fold = fold_id[start:stop]
            for fold in range(n_folds):
                selected = np.flatnonzero(local_fold == fold)
                if len(selected):
                    folds[fold].add_rows(x[selected], y[selected])
        for start in range(0, len(val_t), chunk_rows):
            stop = min(start + chunk_rows, len(val_t))
            x = normalized_raw_windows(
                book,
                mid_z,
                stock_ids,
                day_ids,
                val_t[start:stop],
                stock_stats,
            )
            validation.add_rows(x, y_val[start:stop])
    if sum(value.n for value in folds) != len(train_t):
        raise ExperimentIntegrityError("raw train folds do not cover all rows")
    if validation.n != len(val_t):
        raise ExperimentIntegrityError("raw validation statistics are incomplete")
    metadata = {
        "dimension": dimension,
        "n_train": len(train_t),
        "n_validation": len(val_t),
        "n_folds": n_folds,
        "fold_rows": [value.n for value in folds],
        "fold_seed": fold_seed,
        "chunk_rows": chunk_rows,
    }
    return folds, validation, metadata


def save_raw_statistics_cache(
    path: str | Path,
    folds: Sequence[SufficientStats],
    validation: SufficientStats,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "n_folds": np.asarray(len(folds), dtype=np.int64)
    }
    for index, stats in enumerate(folds):
        arrays.update(_stats_arrays(f"fold_{index}", stats))
    arrays.update(_stats_arrays("validation", validation))
    atomic_savez(path, **arrays)


def load_raw_statistics_cache(
    path: str | Path,
) -> tuple[list[SufficientStats], SufficientStats]:
    with np.load(path, allow_pickle=False) as data:
        n_folds = int(np.asarray(data["n_folds"]).item())
        folds = [_stats_from_npz(data, f"fold_{index}") for index in range(n_folds)]
        validation = _stats_from_npz(data, "validation")
    return folds, validation


def raw_ridge_predictability(
    folds: Sequence[SufficientStats],
    validation: SufficientStats,
    target_names: Sequence[str] = EXPECTED_TARGET_NAMES,
    *,
    alpha_grid: Iterable[float] = ALPHA_GRID,
) -> pd.DataFrame:
    """Per-target grouped-CV alpha selection and untouched-validation R2."""
    full = add_stats(folds)
    if validation.dimension != full.dimension or validation.n_targets != full.n_targets:
        raise ValueError("raw train and validation statistics are incompatible")
    grid = np.asarray(list(alpha_grid), dtype=np.float64)
    if len(grid) == 0 or np.any(grid < 0.0) or not np.isfinite(grid).all():
        raise ValueError("alpha grid is invalid")
    cv = np.full((len(folds), len(grid), full.n_targets), np.nan, dtype=np.float64)
    for fold_index, heldout in enumerate(folds):
        train = subtract_stats(full, heldout)
        design = transformed_design(train)
        for alpha_index, alpha in enumerate(grid):
            model = fit_alpha(design, float(alpha))
            scores = evaluate_stats(model, heldout)
            cv[fold_index, alpha_index, scores.valid] = scores.values[scores.valid]
    mean_cv = np.nanmean(cv, axis=0)
    if np.any(~np.isfinite(mean_cv).any(axis=0)):
        raise ExperimentIntegrityError("raw CV has an invalid target")
    chosen_indices = np.empty(full.n_targets, dtype=np.int64)
    for target_index in range(full.n_targets):
        values = mean_cv[:, target_index]
        best = float(np.nanmax(values))
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(best)) * 16.0
        tied = np.flatnonzero(values >= best - tolerance)
        chosen_indices[target_index] = int(tied[-1])
    full_design = transformed_design(full)
    validation_r2 = np.full(full.n_targets, np.nan, dtype=np.float64)
    lambda_absolute = np.full(full.n_targets, np.nan, dtype=np.float64)
    for alpha_index in np.unique(chosen_indices):
        model = fit_alpha(full_design, float(grid[alpha_index]))
        scores = evaluate_stats(model, validation)
        selected = np.flatnonzero(chosen_indices == alpha_index)
        if not scores.valid[selected].all():
            raise ExperimentIntegrityError("raw validation target is numerically invalid")
        validation_r2[selected] = scores.values[selected]
        lambda_absolute[selected] = model.lambda_absolute
    rows = []
    for index, name in enumerate(target_names):
        rows.append(
            {
                "target_name": str(name),
                "target_family": target_family(str(name)),
                "target_horizon": target_horizon(str(name)),
                "P_raw_linear": float(validation_r2[index]),
                "alpha": float(grid[chosen_indices[index]]),
                "lambda_absolute": float(lambda_absolute[index]),
                "cv_r2_mean": float(mean_cv[chosen_indices[index], index]),
                "cv_r2_sd": float(
                    np.nanstd(cv[:, chosen_indices[index], index], ddof=1)
                ),
                "n_train": full.n,
                "n_validation": validation.n,
                "raw_dimension": full.dimension,
                "cv_folds": len(folds),
                "estimand": "operational_raw_linear_predictability",
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PredictiveMassResult:
    eigenvalues: np.ndarray
    numerical_rank: int
    numerical_tolerance: float
    target_variance: np.ndarray
    pc_cross: np.ndarray
    full_mass: np.ndarray
    top_mass: np.ndarray
    top_fraction: np.ndarray


def predictive_mass_fraction(
    stats: SufficientStats,
    *,
    top_k: int,
    full_mass_floor: float,
) -> PredictiveMassResult:
    design = transformed_design(stats)
    spectrum = design.eigensystem
    rank = spectrum.diagnostics.numerical_rank
    if rank < top_k:
        raise ExperimentIntegrityError("top-k exceeds numerical covariance rank")
    values = spectrum.eigenvalues[:rank]
    if np.any(values <= spectrum.diagnostics.numerical_tolerance):
        raise ExperimentIntegrityError("valid covariance spectrum is inconsistent")
    variance = stats.target_centered_ss / stats.n
    target_scale = np.maximum(stats.yty / stats.n, 1.0)
    target_tolerance = np.finfo(np.float64).eps * target_scale
    if np.any(variance <= target_tolerance):
        raise ExperimentIntegrityError("constant or numerically invalid held-out target")
    pc_cross = spectrum.eigenvectors[:, :rank].T @ stats.cross
    mass_by_direction = np.square(pc_cross) / values[:, None] / variance[None, :]
    if not np.isfinite(mass_by_direction).all() or np.any(mass_by_direction < 0.0):
        raise ExperimentIntegrityError("predictive mass is non-finite or negative")
    full_mass = mass_by_direction.sum(axis=0)
    if np.any(full_mass <= full_mass_floor):
        raise ExperimentIntegrityError("full-rank predictive mass is below its gate")
    top_mass = mass_by_direction[:top_k].sum(axis=0)
    fraction = top_mass / full_mass
    tolerance = np.finfo(np.float64).eps * 1024.0
    if np.any(fraction < -tolerance) or np.any(fraction > 1.0 + tolerance):
        raise ExperimentIntegrityError("top-k predictive-mass fraction is outside [0,1]")
    fraction = np.clip(fraction, 0.0, 1.0)
    return PredictiveMassResult(
        eigenvalues=values,
        numerical_rank=rank,
        numerical_tolerance=spectrum.diagnostics.numerical_tolerance,
        target_variance=variance,
        pc_cross=pc_cross,
        full_mass=full_mass,
        top_mass=top_mass,
        top_fraction=fraction,
    )


def deterministic_haar_basis(
    dimension: int,
    subspace_dimension: int,
    *,
    seed: int,
    branch_index: int,
    encoder_seed: int,
    draw: int,
) -> np.ndarray:
    if not 0 < subspace_dimension <= dimension:
        raise ValueError("Haar subspace dimension is invalid")
    sequence = np.random.SeedSequence(
        [int(seed), int(branch_index), int(encoder_seed), int(draw), int(dimension)]
    )
    rng = np.random.default_rng(sequence)
    q, r = np.linalg.qr(
        rng.standard_normal((dimension, subspace_dimension)), mode="reduced"
    )
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    result = q * signs[None, :]
    np.testing.assert_allclose(
        result.T @ result,
        np.eye(subspace_dimension),
        rtol=2e-12,
        atol=2e-12,
    )
    return result


def haar_fraction_null(
    mass: PredictiveMassResult,
    *,
    top_k: int,
    draws: int,
    seed: int,
    branch_index: int,
    encoder_seed: int,
) -> np.ndarray:
    result = np.empty((draws, len(mass.full_mass)), dtype=np.float64)
    diagonal = mass.eigenvalues
    for draw in range(draws):
        q = deterministic_haar_basis(
            mass.numerical_rank,
            top_k,
            seed=seed,
            branch_index=branch_index,
            encoder_seed=encoder_seed,
            draw=draw,
        )
        covariance = q.T @ (diagonal[:, None] * q)
        covariance = (covariance + covariance.T) * 0.5
        projected_cross = q.T @ mass.pc_cross
        try:
            weights = np.linalg.solve(covariance, projected_cross)
        except np.linalg.LinAlgError as exc:
            raise ExperimentIntegrityError("matched Haar covariance is singular") from exc
        subspace_mass = (
            np.einsum("kt,kt->t", projected_cross, weights)
            / mass.target_variance
        )
        fraction = subspace_mass / mass.full_mass
        tolerance = np.finfo(np.float64).eps * 4096.0
        if (
            not np.isfinite(fraction).all()
            or np.any(fraction < -tolerance)
            or np.any(fraction > 1.0 + tolerance)
        ):
            raise ExperimentIntegrityError("matched Haar mass fraction is invalid")
        result[draw] = np.clip(fraction, 0.0, 1.0)
    return result


def allocation_tables(
    x_train: np.ndarray,
    y_train: np.ndarray,
    target_names: Sequence[str],
    *,
    branch: str,
    encoder_seed: int,
    protocol: AllocationProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = sufficient_stats(x_train, y_train, chunk_rows=protocol.feature_chunk_rows)
    return allocation_tables_from_stats(
        stats,
        target_names,
        branch=branch,
        encoder_seed=encoder_seed,
        protocol=protocol,
    )


def allocation_tables_from_stats(
    stats: SufficientStats,
    target_names: Sequence[str],
    *,
    branch: str,
    encoder_seed: int,
    protocol: AllocationProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if branch not in BRANCHES:
        raise ValueError("unknown branch")
    if stats.n_targets != len(target_names):
        raise ExperimentIntegrityError("representation target count differs")
    mass = predictive_mass_fraction(
        stats,
        top_k=protocol.top_k,
        full_mass_floor=protocol.full_mass_floor,
    )
    null = haar_fraction_null(
        mass,
        top_k=protocol.top_k,
        draws=protocol.haar_draws,
        seed=protocol.haar_seed,
        branch_index=BRANCHES.index(branch),
        encoder_seed=encoder_seed,
    )
    percentile = (1.0 + np.sum(null <= mass.top_fraction[None, :], axis=0)) / (
        protocol.haar_draws + 1.0
    )
    allocation_rows = []
    null_rows = []
    for target_index, name in enumerate(target_names):
        allocation_rows.append(
            {
                "branch": branch,
                "encoder_seed": int(encoder_seed),
                "readout": READOUT,
                "target_name": str(name),
                "target_family": target_family(str(name)),
                "target_horizon": target_horizon(str(name)),
                "top_k": protocol.top_k,
                "valid_dimension": mass.numerical_rank,
                "numerical_tolerance": mass.numerical_tolerance,
                "target_variance": float(mass.target_variance[target_index]),
                "full_predictive_mass": float(mass.full_mass[target_index]),
                "top_predictive_mass": float(mass.top_mass[target_index]),
                "F_top_k": float(mass.top_fraction[target_index]),
                "haar_fraction_mean": float(np.mean(null[:, target_index])),
                "haar_fraction_median": float(np.median(null[:, target_index])),
                "haar_fraction_q05": float(np.quantile(null[:, target_index], 0.05)),
                "haar_fraction_q95": float(np.quantile(null[:, target_index], 0.95)),
                "haar_percentile": float(percentile[target_index]),
                "below_null_q05": bool(
                    percentile[target_index] <= protocol.below_null_quantile
                ),
                "n_train": stats.n,
                "fit_status": "ok",
                "failure_reason": "",
            }
        )
        for draw in range(protocol.haar_draws):
            null_rows.append(
                {
                    "branch": branch,
                    "encoder_seed": int(encoder_seed),
                    "readout": READOUT,
                    "target_name": str(name),
                    "target_family": target_family(str(name)),
                    "top_k": protocol.top_k,
                    "draw": draw,
                    "haar_fraction": float(null[draw, target_index]),
                    "top_fraction": float(mass.top_fraction[target_index]),
                }
            )
    return pd.DataFrame(allocation_rows), pd.DataFrame(null_rows)


def save_representation_statistics_cache(
    path: str | Path, stats: SufficientStats
) -> None:
    atomic_savez(path, **_stats_arrays("train", stats))


def load_representation_statistics_cache(path: str | Path) -> SufficientStats:
    with np.load(path, allow_pickle=False) as data:
        return _stats_from_npz(data, "train")


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return float("nan")
    xr = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def effective_rank_diagnostics(y: np.ndarray) -> Mapping[str, Any]:
    value = np.asarray(y, dtype=np.float64)
    if value.ndim != 2 or len(value) < 2:
        raise ValueError("target matrix must be a non-empty two-dimensional array")
    scale = value.std(axis=0, ddof=1)
    if np.any(scale <= 0.0):
        raise ExperimentIntegrityError("held-out target is constant")
    standardized = (value - value.mean(axis=0)) / scale
    singular = np.linalg.svd(standardized, compute_uv=False)
    eigenvalues = np.square(singular) / (len(value) - 1)
    weights = eigenvalues / eigenvalues.sum()
    positive = weights > 0.0
    entropy_rank = float(np.exp(-np.sum(weights[positive] * np.log(weights[positive]))))
    participation = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    return {
        "numerical_rank": int(np.linalg.matrix_rank(standardized)),
        "entropy_effective_rank": entropy_rank,
        "participation_ratio": participation,
        "components_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "components_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
        "components_99pct": int(np.searchsorted(cumulative, 0.99) + 1),
        "eigenvalues": eigenvalues.tolist(),
        "interpretation": "redundancy diagnostic, not an inferential sample size",
    }


def relationship_tables(
    predictability: pd.DataFrame,
    allocation: pd.DataFrame,
    *,
    protocol: AllocationProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = allocation.merge(
        predictability[["target_name", "P_raw_linear"]],
        on="target_name",
        how="inner",
        validate="many_to_one",
    )
    expected = len(allocation)
    if len(merged) != expected:
        raise ExperimentIntegrityError("P/M target merge is incomplete")
    ordered_targets = predictability.sort_values(
        ["P_raw_linear", "target_name"], kind="mergesort"
    )["target_name"].tolist()
    low_count = max(
        1,
        int(math.ceil(len(ordered_targets) * protocol.low_predictability_fraction)),
    )
    low_targets = set(ordered_targets[:low_count])
    merged["low_predictability"] = merged["target_name"].isin(low_targets)

    correlation_rows = []
    family_rows = []
    for (branch, seed), group in merged.groupby(
        ["branch", "encoder_seed"], observed=True
    ):
        rho_fraction = spearman_rho(group["P_raw_linear"], group["F_top_k"])
        rho_percentile = spearman_rho(
            group["P_raw_linear"], group["haar_percentile"]
        )
        low = group[group["low_predictability"]]
        correlation_rows.append(
            {
                "table_level": "encoder_target_level",
                "branch": branch,
                "encoder_seed": int(seed),
                "rho_P_F": rho_fraction,
                "rho_P_haar_percentile": rho_percentile,
                "n_targets": len(group),
                "n_low_predictability": len(low),
                "below_null_all": int(group["below_null_q05"].sum()),
                "below_null_low_predictability": int(low["below_null_q05"].sum()),
            }
        )
        family_medians = group.groupby("target_family", observed=True).agg(
            P_raw_linear=("P_raw_linear", "median"),
            F_top_k=("F_top_k", "median"),
            haar_percentile=("haar_percentile", "median"),
            n_targets=("target_name", "size"),
        ).reset_index()
        family_rows.extend(
            {
                "branch": branch,
                "encoder_seed": int(seed),
                **row,
            }
            for row in family_medians.to_dict(orient="records")
        )
        correlation_rows.append(
            {
                "table_level": "encoder_family_medians",
                "branch": branch,
                "encoder_seed": int(seed),
                "rho_P_F": spearman_rho(
                    family_medians["P_raw_linear"], family_medians["F_top_k"]
                ),
                "rho_P_haar_percentile": spearman_rho(
                    family_medians["P_raw_linear"],
                    family_medians["haar_percentile"],
                ),
                "n_targets": len(family_medians),
                "n_low_predictability": np.nan,
                "below_null_all": np.nan,
                "below_null_low_predictability": np.nan,
            }
        )
        for family, family_group in group.groupby("target_family", observed=True):
            correlation_rows.append(
                {
                    "table_level": "encoder_within_family",
                    "branch": branch,
                    "encoder_seed": int(seed),
                    "target_family": family,
                    "rho_P_F": spearman_rho(
                        family_group["P_raw_linear"], family_group["F_top_k"]
                    ),
                    "rho_P_haar_percentile": spearman_rho(
                        family_group["P_raw_linear"],
                        family_group["haar_percentile"],
                    ),
                    "n_targets": len(family_group),
                    "n_low_predictability": int(
                        family_group["low_predictability"].sum()
                    ),
                    "below_null_all": int(family_group["below_null_q05"].sum()),
                    "below_null_low_predictability": int(
                        family_group.loc[
                            family_group["low_predictability"], "below_null_q05"
                        ].sum()
                    ),
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    target_level = correlations[
        correlations["table_level"].eq("encoder_target_level")
    ].copy()
    aggregates = target_level.groupby("branch", observed=True).agg(
        rho_P_F_mean=("rho_P_F", "mean"),
        rho_P_F_sd=("rho_P_F", "std"),
        rho_P_haar_percentile_mean=("rho_P_haar_percentile", "mean"),
        rho_P_haar_percentile_sd=("rho_P_haar_percentile", "std"),
        below_null_low_predictability_mean=(
            "below_null_low_predictability",
            "mean",
        ),
        n_encoder_seeds=("encoder_seed", "nunique"),
    ).reset_index()
    paired = target_level.pivot(
        index="encoder_seed", columns="branch", values="rho_P_F"
    )
    if not {"jepa_horizon", "supervised"}.issubset(paired.columns):
        raise ExperimentIntegrityError("primary branch contrast is incomplete")
    delta = (
        paired["jepa_horizon"] - paired["supervised"]
    ).rename("delta_rho_horizon_minus_supervised")
    delta_rows = pd.DataFrame(
        {
            "encoder_seed": delta.index.astype(int),
            "delta_rho_horizon_minus_supervised": delta.to_numpy(),
        }
    )
    summary_row = pd.DataFrame(
        {
            "encoder_seed": [-1],
            "delta_rho_horizon_minus_supervised": [float(delta.mean())],
            "delta_rho_sd": [float(delta.std(ddof=1))],
        }
    )
    delta_rows = pd.concat([delta_rows, summary_row], ignore_index=True, sort=False)
    return merged, correlations, pd.concat(
        [aggregates.assign(table_level="branch_aggregate"), delta_rows.assign(table_level="paired_delta")],
        ignore_index=True,
        sort=False,
    ), pd.DataFrame(family_rows)


def evaluate_preregistered_decision(
    correlations: pd.DataFrame,
    branch_summary: pd.DataFrame,
    *,
    protocol: AllocationProtocol,
) -> Mapping[str, Any]:
    target = correlations[
        correlations["table_level"].eq("encoder_target_level")
    ]
    horizon = target[target["branch"].eq("jepa_horizon")].sort_values("encoder_seed")
    supervised = target[target["branch"].eq("supervised")].sort_values("encoder_seed")
    delta_rows = branch_summary[
        branch_summary["table_level"].eq("paired_delta")
        & branch_summary["encoder_seed"].eq(-1)
    ]
    if len(horizon) != 3 or len(supervised) != 3 or len(delta_rows) != 1:
        raise ExperimentIntegrityError("decision inventory is incomplete")
    rho_gate = bool(
        np.all(horizon["rho_P_F"] > protocol.rho_horizon_min_each_seed)
    )
    delta_value = float(
        delta_rows["delta_rho_horizon_minus_supervised"].iloc[0]
    )
    delta_gate = delta_value > protocol.delta_rho_min_mean
    horizon_null_gate = bool(
        np.all(
            horizon["below_null_low_predictability"]
            >= protocol.min_horizon_below_null_low_p
        )
    )
    supervised_null_gate = bool(
        np.all(
            supervised["below_null_low_predictability"]
            <= protocol.max_supervised_below_null_low_p
        )
    )
    target_pass = rho_gate and delta_gate and horizon_null_gate and supervised_null_gate
    family = correlations[
        correlations["table_level"].eq("encoder_family_medians")
        & correlations["branch"].eq("jepa_horizon")
    ]
    family_direction_consistent = bool(
        len(family) == 3 and np.all(family["rho_P_F"] > 0.0)
    )
    if target_pass and family_direction_consistent:
        outcome = "pass"
    elif target_pass:
        outcome = "ambiguous_target_level_only"
    else:
        outcome = "fail"
    return {
        "schema_name": f"{SCHEMA_NAME}.decision",
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome,
        "target_level_pass": target_pass,
        "family_direction_consistent": family_direction_consistent,
        "gates": {
            "rho_horizon_each_seed": rho_gate,
            "delta_rho_mean": delta_gate,
            "horizon_low_p_below_null": horizon_null_gate,
            "supervised_low_p_below_null": supervised_null_gate,
        },
        "observed": {
            "rho_horizon_by_seed": horizon["rho_P_F"].tolist(),
            "rho_supervised_by_seed": supervised["rho_P_F"].tolist(),
            "delta_rho_mean": delta_value,
            "horizon_below_null_low_p_by_seed": horizon[
                "below_null_low_predictability"
            ].astype(int).tolist(),
            "supervised_below_null_low_p_by_seed": supervised[
                "below_null_low_predictability"
            ].astype(int).tolist(),
        },
        "claim_scope": (
            "association/anti-alignment diagnostic only; no discontinuous SNR "
            "threshold or objective-only causal effect is identified"
        ),
    }


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def render_allocation_report(
    audit: Mapping[str, Any],
    predictability: pd.DataFrame,
    correlations: pd.DataFrame,
    branch_summary: pd.DataFrame,
    family: pd.DataFrame,
    decision: Mapping[str, Any],
    *,
    protocol: AllocationProtocol,
) -> str:
    """Render a completion-only report with explicitly bounded claims."""
    coverage = audit["sample_coverage"]
    splits = coverage["splits"]
    target_level = correlations[
        correlations["table_level"].eq("encoder_target_level")
    ].sort_values(["branch", "encoder_seed"])
    aggregate = branch_summary[
        branch_summary["table_level"].eq("branch_aggregate")
    ].sort_values("branch")
    delta = branch_summary[
        branch_summary["table_level"].eq("paired_delta")
        & branch_summary["encoder_seed"].eq(-1)
    ]
    delta_value = float(delta["delta_rho_horizon_minus_supervised"].iloc[0])
    lines = [
        "# Experiment 01 — Predictability-dependent spectral allocation",
        "",
        "## Status and sample",
        "",
        f"Preregistered decision: **{decision['outcome']}**.",
        "",
        (
            f"The diagnostic fits on {audit['n_train']:,} historical train "
            f"endpoints and evaluates intrinsic predictability on {audit['n_validation']:,} "
            "held-out endpoints. This is "
            f"{100.0 * audit['selected_endpoint_fraction']:.3f}% of the "
            f"{audit['n_valid_total']:,} valid endpoints and "
            f"{100.0 * audit['selected_dataset_row_fraction']:.3f}% of all "
            f"{audit['n_dataset_rows']:,} dataset rows; no model, covariance or "
            "cross-covariance is fit on the full dataset."
        ),
        "",
        (
            f"Coverage is broad: all seven stocks; {splits['train']['n_stock_days']:,} "
            f"train stock-days and {splits['validation']['n_stock_days']:,} "
            f"disjoint validation stock-days. Their union covers "
            f"{100.0 * coverage['valid_stock_day_coverage']:.3f}% "
            f"({splits['train']['n_stock_days'] + splits['validation']['n_stock_days']:,}/"
            f"{splits['valid']['n_stock_days']:,}) of valid stock-days; the "
            f"omitted groups contain {coverage['missing_valid_endpoint_rows']:,} "
            "valid endpoints. The train "
            f"ratios are {coverage['raw_train_rows_over_dimension']:.1f} "
            "observations/dimension for the 800-D raw window and "
            f"{coverage['representation_train_rows_over_dimension']:.1f} for "
            "the 512-D representation."
        ),
        "",
        "## Estimands",
        "",
        (
            "P_j is validation R² from trace-normalized linear ridge on the exact "
            "normalized raw K=20 LOB window. Alpha is selected separately per "
            "target by stock-day-grouped cross-validation inside train."
        ),
        "",
        (
            f"M_j is the fraction of full-rank predictive mass in the top "
            f"{protocol.top_k} covariance directions. Its percentile uses "
            f"{protocol.haar_draws} matched Haar subspaces and the same fractional "
            "mass estimand."
        ),
        "",
        "## Target-level relationship",
        "",
        "| branch | seed | Spearman ρ(P,F) | ρ(P,null percentile) | low-P below null |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in target_level.itertuples(index=False):
        lines.append(
            f"| {row.branch} | {int(row.encoder_seed)} | {row.rho_P_F:.4f} | "
            f"{row.rho_P_haar_percentile:.4f} | "
            f"{int(row.below_null_low_predictability)} |"
        )
    lines.extend(
        [
            "",
            "| branch | mean ρ(P,F) | SD across seeds |",
            "|---|---:|---:|",
        ]
    )
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.branch} | {row.rho_P_F_mean:.4f} | {row.rho_P_F_sd:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "The paired mean contrast is Δρ = ρ(JEPA-horizon) − "
                f"ρ(supervised) = {delta_value:.4f}."
            ),
            "",
            "## Intrinsic predictability",
            "",
            "| target | family | P_raw_linear | selected alpha |",
            "|---|---|---:|---:|",
        ]
    )
    for row in predictability.sort_values("P_raw_linear", ascending=False).itertuples(
        index=False
    ):
        lines.append(
            f"| {row.target_name} | {row.target_family} | "
            f"{row.P_raw_linear:.6f} | {row.alpha:.6g} |"
        )
    family_correlations = correlations[
        correlations["table_level"].eq("encoder_family_medians")
    ].sort_values(["branch", "encoder_seed"])
    lines.extend(
        [
            "",
            "## Dependency and family sensitivity",
            "",
            (
                "The 17 targets are correlated and are not treated as 17 "
                "independent inferential units. Family medians (imbalance, depth, "
                "timing; n=3) are a conservative directional sensitivity check, "
                "not a standalone significance test."
            ),
            "",
            "| branch | seed | family-median ρ(P,F) |",
            "|---|---:|---:|",
        ]
    )
    for row in family_correlations.itertuples(index=False):
        lines.append(
            f"| {row.branch} | {int(row.encoder_seed)} | {row.rho_P_F:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gates",
            "",
        ]
    )
    for name, passed in decision["gates"].items():
        lines.append(f"- {name}: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "A pass supports a predictability–spectral-allocation association "
                "and low-predictability anti-alignment in these frozen encoders. "
                "It does not identify a discontinuous SNR threshold, Bayes "
                "predictability, or a causal effect of the objective alone. The "
                "masked-JEPA arm is descriptive; the preregistered contrast is "
                "horizon-JEPA versus supervised."
            ),
            "",
            (
                "These targets and the historical held-out split were previously "
                "used exploratorily. The analysis is therefore prospective after "
                "exploration, not pristine confirmatory evidence."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_predictability_allocation(
    input_dir: str | Path,
    dataset_path: str | Path,
    protocol_path: str | Path,
    out_dir: str | Path,
    *,
    verify_hashes: bool = True,
) -> Mapping[str, Any]:
    """Run the frozen diagnostic.  This is the only outcome-reading entrypoint."""
    started = time.time()
    protocol, protocol_payload = load_protocol(protocol_path, require_frozen=True)
    audit = audit_historical_inputs(
        input_dir, dataset_path, verify_hashes=verify_hashes
    )
    if protocol_payload["freeze"]["input_inventory_sha256"] != audit[
        "inventory_sha256"
    ]:
        raise ExperimentIntegrityError(
            "current audited inputs differ from the protocol-bound inventory"
        )
    root = Path(input_dir).resolve()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "input_audit.json", audit)
    atomic_write_json(output / "protocol_frozen.json", protocol_payload)

    target_path = Path(audit["heldout_target_path"])
    split_path = Path(audit["split_path"])
    with np.load(target_path, allow_pickle=False) as heldout:
        y_train = np.asarray(heldout["y_train_heldout"], dtype=np.float32)
        target_names = tuple(str(value) for value in heldout["heldout_names"].tolist())
        target_rank = effective_rank_diagnostics(y_train)
    stock_stats = load_stock_stats(
        audit["stock_stats_checkpoint"],
        expected_sha256=str(audit["stock_stats_sha256"]),
    )
    cache_path = output / "raw_window_sufficient_statistics.npz"
    cache_metadata_path = output / "raw_window_sufficient_statistics.json"
    cache_source = {
        "dataset_sha256": audit["dataset_sha256"],
        "split_file_sha256": audit["split_file_sha256"],
        "heldout_target_file_sha256": audit["heldout_target_file_sha256"],
        "stock_stats_sha256": audit["stock_stats_sha256"],
        "raw_cv_folds": protocol.raw_cv_folds,
        "raw_cv_seed": protocol.raw_cv_seed,
        "raw_chunk_rows": protocol.raw_chunk_rows,
    }
    if cache_path.is_file() and cache_metadata_path.is_file():
        cache_metadata = json.loads(cache_metadata_path.read_text())
        if cache_metadata.get("source") != cache_source:
            raise ExperimentIntegrityError("raw-window statistics cache is stale")
        if cache_metadata.get("sha256") != sha256_file(cache_path):
            raise ExperimentIntegrityError("raw-window statistics cache hash differs")
        folds, validation = load_raw_statistics_cache(cache_path)
    else:
        folds, validation, raw_metadata = build_raw_statistics(
            dataset_path,
            split_path,
            target_path,
            stock_stats,
            n_folds=protocol.raw_cv_folds,
            fold_seed=protocol.raw_cv_seed,
            chunk_rows=protocol.raw_chunk_rows,
        )
        save_raw_statistics_cache(cache_path, folds, validation)
        cache_metadata = {
            "schema_name": f"{SCHEMA_NAME}.raw_stats_cache",
            "schema_version": SCHEMA_VERSION,
            "source": cache_source,
            "statistics": raw_metadata,
            "sha256": sha256_file(cache_path),
            "size_bytes": cache_path.stat().st_size,
        }
        atomic_write_json(cache_metadata_path, cache_metadata)
    predictability = raw_ridge_predictability(folds, validation, target_names)
    atomic_write_parquet(predictability, output / "intrinsic_predictability.parquet")

    allocation_parts = []
    null_parts = []
    representation_cache_artifacts: list[str] = []
    representation_cache_dir = output / "representation_statistics"
    representation_cache_dir.mkdir(parents=True, exist_ok=True)
    for record in audit["readouts"]:
        key = str(record["key"])
        cache_path = representation_cache_dir / f"{key}.npz"
        metadata_path = representation_cache_dir / f"{key}.json"
        cache_source = {
            "readout_file_sha256": record["file_sha256"],
            "heldout_target_file_sha256": audit["heldout_target_file_sha256"],
            "train_endpoint_sha256": audit["train_endpoint_sha256"],
            "readout": READOUT,
            "n_train": EXPECTED_N_TRAIN,
            "dimension": EXPECTED_DIMENSION,
            "n_targets": len(target_names),
        }
        if cache_path.is_file() and metadata_path.is_file():
            cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cache_metadata.get("source") != cache_source:
                raise ExperimentIntegrityError(
                    f"representation-statistics cache is stale for {key}"
                )
            if cache_metadata.get("sha256") != sha256_file(cache_path):
                raise ExperimentIntegrityError(
                    f"representation-statistics cache hash differs for {key}"
                )
            stats = load_representation_statistics_cache(cache_path)
        else:
            if cache_path.exists() or metadata_path.exists():
                raise ExperimentIntegrityError(
                    f"representation-statistics cache is incomplete for {key}"
                )
            with np.load(record["path"], allow_pickle=False) as readout:
                if str(readout["train_endpoint_sha256"].item()) != audit[
                    "train_endpoint_sha256"
                ]:
                    raise ExperimentIntegrityError(
                        "readout endpoint hash changed at run"
                    )
                x_train = np.asarray(
                    readout[f"{READOUT}_train"], dtype=np.float32
                )
                stats = sufficient_stats(
                    x_train,
                    y_train,
                    chunk_rows=protocol.feature_chunk_rows,
                )
                del x_train
            save_representation_statistics_cache(cache_path, stats)
            cache_metadata = {
                "schema_name": f"{SCHEMA_NAME}.representation_stats_cache",
                "schema_version": SCHEMA_VERSION,
                "source": cache_source,
                "statistics": {
                    "n": stats.n,
                    "dimension": stats.dimension,
                    "n_targets": stats.n_targets,
                },
                "sha256": sha256_file(cache_path),
                "size_bytes": cache_path.stat().st_size,
            }
            atomic_write_json(metadata_path, cache_metadata)
        if (
            stats.n != EXPECTED_N_TRAIN
            or stats.dimension != EXPECTED_DIMENSION
            or stats.n_targets != len(target_names)
        ):
            raise ExperimentIntegrityError(
                f"representation statistics have the wrong shape for {key}"
            )
        allocation, null = allocation_tables_from_stats(
            stats,
            target_names,
            branch=str(record["branch"]),
            encoder_seed=int(record["encoder_seed"]),
            protocol=protocol,
        )
        allocation_parts.append(allocation)
        null_parts.append(null)
        representation_cache_artifacts.extend(
            [
                str(cache_path.relative_to(output)),
                str(metadata_path.relative_to(output)),
            ]
        )
    allocation = pd.concat(allocation_parts, ignore_index=True)
    null = pd.concat(null_parts, ignore_index=True)
    atomic_write_parquet(allocation, output / "spectral_allocation.parquet")
    atomic_write_parquet(null, output / "matched_haar_null.parquet")
    merged, correlations, branch_summary, family = relationship_tables(
        predictability, allocation, protocol=protocol
    )
    atomic_write_parquet(merged, output / "target_relationships.parquet")
    atomic_write_parquet(correlations, output / "correlations.parquet")
    atomic_write_parquet(branch_summary, output / "branch_summary.parquet")
    atomic_write_parquet(family, output / "family_diagnostics.parquet")
    decision = evaluate_preregistered_decision(
        correlations, branch_summary, protocol=protocol
    )
    atomic_write_json(output / "decision.json", decision)
    report = render_allocation_report(
        audit,
        predictability,
        correlations,
        branch_summary,
        family,
        decision,
        protocol=protocol,
    )
    _atomic_write_text(
        output / "REPORT_EXPERIMENT_01_PREDICTABILITY_ALLOCATION.md", report
    )
    atomic_write_parquet(
        pd.DataFrame(columns=["stage", "branch", "encoder_seed", "target_name", "reason"]),
        output / "failures.parquet",
    )
    artifact_names = (
        "input_audit.json",
        "protocol_frozen.json",
        "raw_window_sufficient_statistics.npz",
        "raw_window_sufficient_statistics.json",
        "intrinsic_predictability.parquet",
        "spectral_allocation.parquet",
        "matched_haar_null.parquet",
        "target_relationships.parquet",
        "correlations.parquet",
        "branch_summary.parquet",
        "family_diagnostics.parquet",
        "decision.json",
        "REPORT_EXPERIMENT_01_PREDICTABILITY_ALLOCATION.md",
        "failures.parquet",
    ) + tuple(representation_cache_artifacts)
    artifacts = {
        name: {
            "sha256": sha256_file(output / name),
            "size_bytes": (output / name).stat().st_size,
        }
        for name in artifact_names
    }
    metadata = {
        "schema_name": f"{SCHEMA_NAME}.metadata",
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "analysis_version": ANALYSIS_VERSION,
        "protocol_file_sha256": sha256_file(protocol_path),
        "input_inventory_sha256": audit["inventory_sha256"],
        "git_commit": _git_commit(Path.cwd()),
        "runtime_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "target_rank_diagnostic": target_rank,
        "decision": decision,
        "artifacts": artifacts,
        "phase1_phase2_phase3_modified": False,
    }
    atomic_write_json(output / "metadata.json", metadata)
    manifest_payload = {
        "schema_name": f"{SCHEMA_NAME}.manifest",
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "metadata_sha256": sha256_file(output / "metadata.json"),
        "artifacts": {
            **artifacts,
            "metadata.json": {
                "sha256": sha256_file(output / "metadata.json"),
                "size_bytes": (output / "metadata.json").stat().st_size,
            },
        },
    }
    manifest_payload["payload_sha256"] = canonical_json_sha256(manifest_payload)
    atomic_write_json(output / "manifest.json", manifest_payload)
    return metadata

"""Experiment 01 Phase III: reader-relative accessibility diagnostics.

Phase I and Phase II are immutable inputs.  This module writes only Phase-III
artifacts and enforces the validation/test boundary explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from .errors import ExperimentIntegrityError
from .io import (
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_sha256,
    sha256_array,
    sha256_file,
)
from .schema import FeatureSet, InputBundle, TargetDefinition


PHASE1_RESULTS_SHA256 = (
    "ecf4e410c595baa32d06a1998bbd5151794d02ff141499af3c1f56268e110ffb"
)
PHASE1_SUMMARY_SHA256 = (
    "7978961be69e50881ac022a67bfd7fea4f619c9806374121b57d6d4cbac1d4a6"
)
PHASE1_SUBSET_MANIFEST_SHA256 = (
    "3a4412bf110706f685b1502c64ba277a9b3129cf3c64aa77ccd53eefe5ef1471"
)
PHASE2_MANIFEST_SHA256 = (
    "1a30b67f6739a1a0440eae1866ee55f72cddf94248e5edf336a7e605461144c2"
)
BUNDLE_MANIFEST_SHA256 = (
    "bdded4ebd03c29d47e5dfdba106590f24763cc06bb7e6e5ea379eb4b34201c0b"
)

PRIMARY_BRANCHES = ("jepa_horizon", "supervised")
PRIMARY_READOUT = "last_concat512"
SECONDARY_READOUT = "meanK_concatS"
VALID_RANK = 508
SELECTION_SEED = 7919
WEIGHT_DECAYS = (0.0, 1e-5, 1e-3)
READER_SEEDS = (0, 1, 2, 3, 4)
MLP_MIN_ROWS = 4096
SPECTRAL_BANDS = (
    ("band_1_127", 0, 127),
    ("band_128_254", 127, 254),
    ("band_255_381", 254, 381),
    ("band_382_508", 381, 508),
)
SPECTRAL_ARMS = tuple(name for name, _, _ in SPECTRAL_BANDS) + (
    "full_valid_rank",
    "top_128",
)
PRIMARY_BUDGETS = (
    "b_1_4",
    "b_1_2",
    "b_1",
    "b_2",
    "b_4",
    "b_8",
    "b_16",
    "b_32",
    "b_64",
    "b_128",
    "balanced_max",
    "full_train",
)
LOW_BUDGETS = ("b_1_4", "b_1_2", "b_1", "b_2", "b_4")
CONTROL_BUDGETS = ("b_1_4", "b_1", "b_4", "balanced_max", "full_train")
SPECTRAL_BUDGETS = ("b_1_4", "b_4", "full_train")
FAILURE_COLUMNS = (
    "job_key",
    "stage",
    "exception",
    "last_completed_step",
    "validation_state",
    "gpu_memory_bytes",
    "system_rss_bytes",
    "scientifically_required",
)
PHASE1_METADATA_SHA256 = (
    "3bae44567d2873d67ef5b59e92b9c0eacc693b88309f94866a86829b9b6bf5e3"
)
HISTORICAL_MLP_SOURCE_SHA256 = (
    "a34c8574b2914efa25c9677f1b404f23ebf8dec579fe1bf914455d220711ddd6"
)
HISTORICAL_MLP_REFERENCES = {
    "jepa_horizon": 0.3191358981,
    "supervised": 0.3880910480,
}
PHASE3_SPECIFICATION_SHA256 = (
    "78ca15821ac40355c35e5f40ecaf5086f5e6bbb6f339255a85b13fc7d952a151"
)
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_DRAWS = 10_000


@dataclass(frozen=True)
class Phase3MLPConfig:
    width: int = 256
    dropout: float = 0.10
    learning_rate: float = 1e-3
    max_steps: int = 20_000
    min_steps: int = 1_000
    validation_interval: int = 500
    patience_evaluations: int = 6
    minimum_validation_improvement: float = 1e-5
    gradient_clip_norm: float = 5.0
    evaluation_chunk_rows: int = 65_536

    def validate(
        self, *, primary: bool = True, enforce_preregistered_schedule: bool = True
    ) -> None:
        if self.width <= 0:
            raise ValueError("MLP width must be positive")
        if primary and self.width != 256:
            raise ValueError("the primary Phase-III MLP width must be 256")
        if self.dropout != 0.10:
            raise ValueError("Phase III fixes dropout at 0.10")
        if self.learning_rate != 1e-3:
            raise ValueError("Phase III fixes learning rate at 1e-3")
        if enforce_preregistered_schedule:
            if (
                self.max_steps != 20_000
                or self.min_steps != 1_000
                or self.validation_interval != 500
                or self.patience_evaluations != 6
            ):
                raise ValueError("Phase III step/stopping schedule is frozen")
            if self.minimum_validation_improvement != 1e-5:
                raise ValueError("Phase III validation improvement is frozen")
            if self.gradient_clip_norm != 5.0:
                raise ValueError("Phase III gradient clipping is frozen")
        elif (
            self.max_steps <= 0
            or self.min_steps <= 0
            or self.min_steps > self.max_steps
            or self.validation_interval <= 0
            or self.patience_evaluations <= 0
            or self.minimum_validation_improvement < 0
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError("invalid smoke/gate MLP schedule")
        if self.evaluation_chunk_rows <= 0:
            raise ValueError("evaluation_chunk_rows must be positive")


@dataclass(frozen=True)
class TargetStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    source_subset_hash: str
    n_rows: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(values, dtype=np.float32) - self.mean)
            / self.scale
        ).astype(np.float32, copy=False)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float32) * self.scale + self.mean
        ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class FrozenFeatureTransform:
    kind: str
    mean: np.ndarray
    basis: np.ndarray | None
    scales: np.ndarray | None
    input_dimension: int
    output_dimension: int
    transform_hash: str
    source_transform_sha256: str
    spectral_arm: str = "none"

    def apply_numpy(self, values: np.ndarray) -> np.ndarray:
        centered = np.asarray(values, dtype=np.float32) - self.mean
        if self.basis is None:
            transformed = centered
        else:
            transformed = centered @ self.basis
        if self.scales is not None:
            transformed = transformed * self.scales
        result = np.asarray(transformed, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != self.output_dimension:
            raise ExperimentIntegrityError("feature transform output shape mismatch")
        if not np.isfinite(result).all():
            raise ExperimentIntegrityError("non-finite transformed Phase-III input")
        return result


def make_primary_mlp(input_dimension: int, output_dimension: int, *, width: int = 256):
    """Construct the exact one-hidden-layer reader; no hidden normalization."""

    import torch.nn as nn

    if input_dimension <= 0 or output_dimension <= 0 or width <= 0:
        raise ValueError("MLP dimensions must be positive")
    return nn.Sequential(
        nn.Linear(input_dimension, width, bias=True),
        nn.GELU(),
        nn.Dropout(p=0.10),
        nn.Linear(width, output_dimension, bias=True),
    )


def target_indices_for_block(
    definitions: Sequence[TargetDefinition], block: str
) -> tuple[int, ...]:
    indices = tuple(
        index
        for index, target in enumerate(definitions)
        if target.block == block and target.independent
    )
    expected = {"directional": 12, "volatility": 2, "timing": 1}
    if block not in expected or len(indices) != expected[block]:
        raise ExperimentIntegrityError(
            f"target block {block!r} has {len(indices)} independent targets"
        )
    return indices


def fit_target_standardizer(
    target_source,
    positions: np.ndarray,
    target_indices: Sequence[int],
    *,
    subset_hash: str,
    chunk_rows: int = 65_536,
) -> TargetStandardizer:
    """Fit target scaling on the selected labelled rows and nowhere else."""

    rows = np.asarray(positions, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("labelled target positions must be a non-empty vector")
    if len(np.unique(rows)) != len(rows):
        raise ExperimentIntegrityError("labelled subset positions are not unique")
    columns = np.asarray(tuple(target_indices), dtype=np.int64)
    if columns.ndim != 1 or len(columns) == 0:
        raise ValueError("target_indices must be non-empty")
    total = np.zeros(len(columns), dtype=np.float64)
    second = np.zeros(len(columns), dtype=np.float64)
    count = 0
    for start in range(0, len(rows), chunk_rows):
        values = np.asarray(
            target_source[rows[start : start + chunk_rows]], dtype=np.float64
        )[:, columns]
        if not np.isfinite(values).all():
            raise ExperimentIntegrityError("non-finite labelled targets")
        total += values.sum(axis=0)
        second += np.einsum("nt,nt->t", values, values)
        count += len(values)
    mean = total / count
    variance = np.maximum(second / count - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return TargetStandardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        source_subset_hash=str(subset_hash),
        n_rows=count,
    )


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ExperimentIntegrityError(f"missing frozen {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ExperimentIntegrityError(
            f"frozen {label} hash mismatch: expected {expected}, observed {observed}"
        )


def initialize_compute_device(device: str) -> dict[str, Any]:
    """Initialize ROCm/CUDA before large PyArrow/NumPy artifact scans."""

    import torch

    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm requested but unavailable")
        torch.cuda.init()
        return {
            "device": str(dev),
            "device_name": torch.cuda.get_device_name(dev),
            "hip": torch.version.hip,
            "mixed_precision": False,
        }
    return {
        "device": str(dev),
        "device_name": "cpu",
        "hip": torch.version.hip,
        "mixed_precision": False,
    }


def verify_phase1_inputs(phase1_dir: str | Path) -> None:
    """Fail closed if the frozen Phase-I inputs needed here have changed."""

    root = Path(phase1_dir)
    _require_sha256(root / "results.parquet", PHASE1_RESULTS_SHA256, "Phase-I results")
    _require_sha256(
        root.parent / "summary" / "summary.json",
        PHASE1_SUMMARY_SHA256,
        "Phase-I technical summary",
    )
    _require_sha256(
        root / "subset_manifest.json",
        PHASE1_SUBSET_MANIFEST_SHA256,
        "Phase-I subset manifest",
    )


def _phase1_whitening_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "target_independent",
        "budget_kind",
        "budget_days_per_stock",
        "budget_stock_day_equivalents",
        "n_stock_days",
        "n_rows",
        "subsample_seed",
        "block_anchor_quantile",
        "feature_view",
        "whiten_k_requested",
        "whiten_k_effective",
        "reader_family",
        "alpha",
        "alpha_selected",
        "test_r2",
        "full_budget_test_r2",
        "normalized_recovery",
        "ceiling_eligible",
        "fit_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ExperimentIntegrityError(
            f"Phase-I results missing whitening columns: {missing}"
        )
    selected = frame.loc[
        (frame["feature_view"] == "full_rank_whiten_topk")
        & (frame["reader_family"] == "ridge_whiten_topk_tuned_alpha")
        & frame["alpha_selected"].astype(bool)
        & (frame["fit_status"] == "ok")
        & frame["target_independent"].astype(bool)
    ].copy()
    if selected.empty:
        raise ExperimentIntegrityError("no selected Phase-I whitening rows found")
    selected["whitening_depth"] = selected["whiten_k_effective"].astype(int)
    return selected


def build_phase1_branch_whitening_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive branch-specific effects from frozen Phase-I normalized recovery.

    Each row is paired with the exact k=0 cell.  Subsampling seed remains an
    explicit identity even though the prose formula abbreviates it as budget g.
    """

    selected = _phase1_whitening_rows(frame)
    identity = [
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "budget_kind",
        "budget_days_per_stock",
        "budget_stock_day_equivalents",
        "subsample_seed",
    ]
    exact_identity = ["branch", *identity]
    baseline = selected.loc[selected["whitening_depth"] == 0].copy()
    duplicates = baseline.duplicated(exact_identity, keep=False)
    if duplicates.any():
        raise ExperimentIntegrityError("Phase-I k=0 whitening baseline is not unique")
    expected = selected[exact_identity].drop_duplicates()
    missing = expected.merge(
        baseline[exact_identity], on=exact_identity, how="left", indicator=True
    )
    if (missing["_merge"] != "both").any():
        raise ExperimentIntegrityError("Phase-I whitening cell lacks an exact k=0 row")

    base_columns = exact_identity + [
        "test_r2",
        "normalized_recovery",
        "alpha",
        "ceiling_eligible",
    ]
    baseline = baseline[base_columns].rename(
        columns={
            "test_r2": "test_r2_k0",
            "normalized_recovery": "normalized_recovery_k0",
            "alpha": "alpha_k0",
            "ceiling_eligible": "ceiling_eligible_k0",
        }
    )
    out = selected.merge(baseline, on=exact_identity, how="left", validate="many_to_one")
    out["delta_test_r2_vs_k0"] = out["test_r2"] - out["test_r2_k0"]
    out["delta_normalized_recovery_vs_k0"] = (
        out["normalized_recovery"] - out["normalized_recovery_k0"]
    )
    tolerance = 1e-12
    delta = out["delta_normalized_recovery_vs_k0"]
    out["branch_effect"] = np.select(
        [delta > tolerance, delta < -tolerance],
        ["helps", "harms"],
        default="unchanged",
    )

    primary = out.loc[out["branch"].isin(PRIMARY_BRANCHES)].copy()
    pair_key = identity + ["whitening_depth"]
    pair_counts = primary.groupby(pair_key, dropna=False)["branch"].nunique()
    if not bool((pair_counts == len(PRIMARY_BRANCHES)).all()):
        raise ExperimentIntegrityError(
            "primary branch whitening rows are not exactly pairable"
        )
    paired = primary.loc[
        primary["branch"] == "jepa_horizon",
        pair_key + ["delta_normalized_recovery_vs_k0"],
    ].rename(
        columns={
            "delta_normalized_recovery_vs_k0": "jepa_horizon_delta_recovery"
        }
    )
    supervised_pair = primary.loc[
        primary["branch"] == "supervised",
        pair_key + ["delta_normalized_recovery_vs_k0"],
    ].rename(
        columns={
            "delta_normalized_recovery_vs_k0": "supervised_delta_recovery"
        }
    )
    paired = paired.merge(
        supervised_pair, on=pair_key, how="inner", validate="one_to_one"
    )
    jepa = paired["jepa_horizon_delta_recovery"]
    supervised = paired["supervised_delta_recovery"]
    same_direction = (
        ((jepa > tolerance) & (supervised > tolerance))
        | ((jepa < -tolerance) & (supervised < -tolerance))
        | ((jepa.abs() <= tolerance) & (supervised.abs() <= tolerance))
    )
    paired["changes_both_same_direction"] = same_direction
    paired["helps_jepa_horizon"] = jepa > tolerance
    paired["harms_jepa_horizon"] = jepa < -tolerance
    paired["helps_supervised_more"] = supervised > jepa + tolerance
    paired["harms_supervised_less"] = (
        (jepa < -tolerance)
        & (supervised < -tolerance)
        & (supervised > jepa + tolerance)
    )
    paired["supervised_minus_jepa_delta"] = supervised - jepa
    out = out.merge(paired, on=pair_key, how="left", validate="many_to_one")
    out.insert(0, "phase1_results_sha256", PHASE1_RESULTS_SHA256)
    out = out.sort_values(
        [
            "readout",
            "target_block",
            "target_name",
            "budget_stock_day_equivalents",
            "subsample_seed",
            "encoder_seed",
            "whitening_depth",
            "branch",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return out


def summarize_phase1_branch_whitening_effects(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = []
    columns = ["branch", "readout", "target_block", "whitening_depth"]
    for key, group in frame.groupby(columns, observed=True, sort=True):
        delta = group["delta_normalized_recovery_vs_k0"].to_numpy(dtype=float)
        finite = np.isfinite(delta)
        values = delta[finite]
        grouped.append(
            {
                **dict(zip(columns, key)),
                "n_rows": int(len(group)),
                "n_finite": int(finite.sum()),
                "mean_delta_normalized_recovery": (
                    float(values.mean()) if len(values) else None
                ),
                "median_delta_normalized_recovery": (
                    float(np.median(values)) if len(values) else None
                ),
                "fraction_helped": (
                    float(np.mean(values > 1e-12)) if len(values) else None
                ),
                "fraction_harmed": (
                    float(np.mean(values < -1e-12)) if len(values) else None
                ),
            }
        )
    primary = frame.loc[
        (frame["branch"] == "jepa_horizon")
        & (frame["readout"] == PRIMARY_READOUT)
    ]
    relations = []
    for (block, depth), group in primary.groupby(
        ["target_block", "whitening_depth"], observed=True, sort=True
    ):
        relations.append(
            {
                "target_block": block,
                "whitening_depth": int(depth),
                "n_paired_cells": int(len(group)),
                "fraction_helps_jepa_horizon": float(group["helps_jepa_horizon"].mean()),
                "fraction_harms_jepa_horizon": float(group["harms_jepa_horizon"].mean()),
                "fraction_helps_supervised_more": float(group["helps_supervised_more"].mean()),
                "fraction_harms_supervised_less": float(group["harms_supervised_less"].mean()),
                "fraction_changes_both_same_direction": float(
                    group["changes_both_same_direction"].mean()
                ),
            }
        )
    return {
        "schema_name": "thesis.experiment01.phase3.branch_whitening_summary",
        "schema_version": 1,
        "source_phase1_results_sha256": PHASE1_RESULTS_SHA256,
        "interpretation_status": "post_hoc_descriptive",
        "hypothesis": (
            "Partial whitening may benefit tail-loaded directional information "
            "while temporarily degrading accessibility for head-loaded timing or "
            "intermediate volatility."
        ),
        "grouped_effects": grouped,
        "primary_branch_relations": relations,
    }


def write_phase1_branch_whitening_effects(
    phase1_dir: str | Path, out_dir: str | Path
) -> dict[str, Any]:
    """Verify frozen inputs and write the mandatory free Phase-III diagnostic."""

    verify_phase1_inputs(phase1_dir)
    source = Path(phase1_dir) / "results.parquet"
    frame = pd.read_parquet(source)
    effects = build_phase1_branch_whitening_effects(frame)
    summary = summarize_phase1_branch_whitening_effects(effects)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_path = destination / "phase1_branch_whitening_effects.parquet"
    summary_path = destination / "phase1_branch_whitening_effects_summary.json"
    atomic_write_parquet(effects, table_path)
    atomic_write_json(summary_path, summary)
    return {
        "table": str(table_path),
        "table_rows": int(len(effects)),
        "table_sha256": sha256_file(table_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }


def _phase1_transform_record(
    phase1_dir: Path, feature: FeatureSet
) -> tuple[Mapping[str, Any], Path]:
    metadata_path = phase1_dir / "metadata.json"
    _require_sha256(metadata_path, PHASE1_METADATA_SHA256, "Phase-I metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in metadata.get("transforms", [])
        if record.get("branch") == feature.branch
        and int(record.get("encoder_seed", -1)) == feature.encoder_seed
        and record.get("readout") == feature.readout
    ]
    if len(matches) != 1:
        raise ExperimentIntegrityError(
            f"frozen Phase-I transform is not unique for {feature.key}"
        )
    record = matches[0]
    path = phase1_dir / str(record["path"])
    _require_sha256(path, str(record["sha256"]), f"transform {feature.key}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise ExperimentIntegrityError(f"transform size mismatch for {feature.key}")
    return record, path


def load_frozen_feature_transform(
    phase1_dir: str | Path,
    feature: FeatureSet,
    *,
    kind: str,
    spectral_arm: str = "none",
) -> FrozenFeatureTransform:
    """Load a train-only frozen transform without using labels or held-out data."""

    root = Path(phase1_dir)
    record, path = _phase1_transform_record(root, feature)
    with np.load(path, allow_pickle=False) as data:
        mean = np.asarray(data["unlabelled_train_mean"], dtype=np.float64)
        eigenvalues = np.asarray(data["covariance_eigenvalues"], dtype=np.float64)
        eigenvectors = np.asarray(data["covariance_eigenvectors"], dtype=np.float64)
        tolerance = float(np.asarray(data["numerical_tolerance"]).item())
        rank = int(np.asarray(data["numerical_rank"]).item())
    dimension = feature.dimension
    if (
        mean.shape != (dimension,)
        or eigenvalues.shape != (dimension,)
        or eigenvectors.shape != (dimension, dimension)
        or rank != VALID_RANK
    ):
        raise ExperimentIntegrityError(
            f"invalid frozen transform shape/rank for {feature.key}"
        )
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(eigenvalues).all()
        or not np.isfinite(eigenvectors).all()
        or not np.all(eigenvalues[:rank] > tolerance)
        or np.any(eigenvalues[rank:] > tolerance)
    ):
        raise ExperimentIntegrityError(
            f"invalid frozen transform numerics for {feature.key}"
        )
    orthogonality = eigenvectors.T @ eigenvectors
    if not np.allclose(orthogonality, np.eye(dimension), atol=2e-8, rtol=2e-8):
        raise ExperimentIntegrityError(
            f"frozen PCA basis is not orthonormal for {feature.key}"
        )

    basis: np.ndarray | None
    scales: np.ndarray | None
    arm = spectral_arm
    if kind == "native":
        if arm != "none":
            raise ValueError("native transform cannot declare a spectral arm")
        basis = None
        scales = None
        output_dimension = dimension
    elif kind == "full_whitened":
        if arm != "none":
            raise ValueError("full whitening cannot declare a spectral arm")
        basis = eigenvectors[:, :rank]
        scales = 1.0 / np.sqrt(eigenvalues[:rank])
        output_dimension = rank
    elif kind == "pca_coordinates":
        if arm in dict((name, (start, stop)) for name, start, stop in SPECTRAL_BANDS):
            start, stop = dict(
                (name, (left, right)) for name, left, right in SPECTRAL_BANDS
            )[arm]
        elif arm == "full_valid_rank":
            start, stop = 0, rank
        elif arm == "top_128":
            start, stop = 0, 128
        else:
            raise ValueError(f"unknown PCA spectral arm {arm!r}")
        basis = eigenvectors[:, start:stop]
        scales = None
        output_dimension = stop - start
    else:
        raise ValueError(f"unknown Phase-III feature transform {kind!r}")
    payload = {
        "algorithm": "phase3_train_only_frozen_feature_transform.v1",
        "kind": kind,
        "spectral_arm": arm,
        "source_transform_sha256": record["sha256"],
        "source_phase1_metadata_sha256": PHASE1_METADATA_SHA256,
        "input_dimension": dimension,
        "output_dimension": output_dimension,
        "valid_rank": rank,
        "centering": "all_unlabelled_train_mean",
        "coordinate_standardization": False,
    }
    return FrozenFeatureTransform(
        kind=kind,
        mean=mean.astype(np.float32),
        basis=None if basis is None else basis.astype(np.float32),
        scales=None if scales is None else scales.astype(np.float32),
        input_dimension=dimension,
        output_dimension=output_dimension,
        transform_hash=canonical_json_sha256(payload),
        source_transform_sha256=str(record["sha256"]),
        spectral_arm=arm,
    )


def verify_spectral_band_identity() -> dict[str, Any]:
    coverage = np.zeros(VALID_RANK, dtype=np.int8)
    for name, start, stop in SPECTRAL_BANDS:
        if stop - start != 127:
            raise ExperimentIntegrityError(f"spectral band {name} is not 127-D")
        if start < 0 or stop > VALID_RANK or stop <= start:
            raise ExperimentIntegrityError(f"spectral band {name} is invalid")
        coverage[start:stop] += 1
    if not np.array_equal(coverage, np.ones(VALID_RANK, dtype=np.int8)):
        raise ExperimentIntegrityError(
            "spectral bands are not a disjoint union of PCs 1:508"
        )
    return {
        "status": "pass",
        "valid_rank": VALID_RANK,
        "bands": [
            {"name": name, "pc_start_inclusive": start + 1, "pc_stop_inclusive": stop}
            for name, start, stop in SPECTRAL_BANDS
        ],
        "coverage_sha256": sha256_array(coverage),
    }


def _load_subset_manifest(phase1_dir: Path) -> Mapping[str, Any]:
    path = phase1_dir / "subset_manifest.json"
    _require_sha256(path, PHASE1_SUBSET_MANIFEST_SHA256, "Phase-I subset manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("subsets")
    if not isinstance(records, list) or not records:
        raise ExperimentIntegrityError("Phase-I subset manifest has no subsets")
    for record in records:
        subset_path = phase1_dir / str(record.get("path", ""))
        if (
            not subset_path.is_file()
            or subset_path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(subset_path) != record.get("sha256")
        ):
            raise ExperimentIntegrityError(
                f"Phase-I subset file identity mismatch: {record.get('path')}"
            )
    return payload


def _records_by_budget(payload: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in payload["subsets"]:
        grouped.setdefault(str(record["budget_label"]), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda value: int(value["subsample_seed"]))
    missing = sorted(set(PRIMARY_BUDGETS) - set(grouped))
    if missing:
        raise ExperimentIntegrityError(f"missing Phase-I Phase-III budgets: {missing}")
    ineligible = [
        (label, int(record["n_rows"]))
        for label in PRIMARY_BUDGETS
        for record in grouped[label]
        if int(record["n_rows"]) < MLP_MIN_ROWS
    ]
    if ineligible:
        raise ExperimentIntegrityError(
            f"primary MLP budget unexpectedly below 4096 rows: {ineligible[:3]}"
        )
    lower = grouped.get("b_1_8", [])
    if not lower or min(int(value["n_rows"]) for value in lower) >= MLP_MIN_ROWS:
        raise ExperimentIntegrityError("b_min_mlp=0.25 is not supported by the manifest")
    return grouped


def adaptive_reader_seeds(budget_label: str) -> tuple[int, ...]:
    if budget_label in LOW_BUDGETS:
        return READER_SEEDS
    if budget_label in PRIMARY_BUDGETS:
        return READER_SEEDS[:3]
    raise ValueError(f"unknown Phase-III budget {budget_label!r}")


def _model_definition_hash(width: int, input_dimension: int, target_block: str) -> str:
    output_dimensions = {"directional": 12, "volatility": 2, "timing": 1}
    return canonical_json_sha256(
        {
            "architecture": [
                ["Linear", input_dimension, width, True],
                ["GELU"],
                ["Dropout", 0.10],
                ["Linear", width, output_dimensions[target_block], True],
            ],
            "input_coordinate_standardization": False,
            "batch_norm": False,
            "layer_norm": False,
        }
    )


def _job_key(record: Mapping[str, Any]) -> str:
    identity_names = (
        "stage",
        "job_family",
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "transform",
        "spectral_arm",
        "budget_label",
        "subsample_seed",
        "reader_seed",
        "weight_decay",
        "width",
    )
    return canonical_json_sha256({name: record.get(name) for name in identity_names})


def _logical_job_key(record: Mapping[str, Any]) -> str:
    identity_names = (
        "job_family",
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "transform",
        "spectral_arm",
        "budget_label",
        "subsample_seed",
        "width",
    )
    return canonical_json_sha256({name: record.get(name) for name in identity_names})


def build_phase3_job_inventory(subset_payload: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Serialize every selection candidate and evaluation blueprint pre-compute."""

    grouped = _records_by_budget(subset_payload)
    logical: list[dict[str, Any]] = []

    def append_cells(
        *,
        family: str,
        target_block: str,
        budget_labels: Sequence[str],
        transforms: Sequence[tuple[str, str, int]],
        width: int = 256,
    ) -> None:
        for branch in PRIMARY_BRANCHES:
            for encoder_seed in (0, 1, 2):
                for transform, spectral_arm, input_dimension in transforms:
                    for label in budget_labels:
                        for subset in grouped[label]:
                            logical.append(
                                {
                                    "job_family": family,
                                    "branch": branch,
                                    "encoder_seed": encoder_seed,
                                    "readout": PRIMARY_READOUT,
                                    "target_block": target_block,
                                    "transform": transform,
                                    "spectral_arm": spectral_arm,
                                    "budget_label": label,
                                    "budget_days_per_stock": subset.get(
                                        "budget_days_per_stock"
                                    ),
                                    "budget_stock_day_equivalents": subset.get(
                                        "budget_stock_day_equivalents"
                                    ),
                                    "n_stock_days": int(subset["n_stock_days"]),
                                    "n_rows": int(subset["n_rows"]),
                                    "subsample_seed": int(subset["subsample_seed"]),
                                    "subset_path": str(subset["path"]),
                                    "subset_hash": str(subset["row_key_sha256"]),
                                    "subset_file_sha256": str(subset["sha256"]),
                                    "input_dimension": input_dimension,
                                    "width": width,
                                    "model_definition_hash": _model_definition_hash(
                                        width, input_dimension, target_block
                                    ),
                                }
                            )

    primary_transforms = (
        ("native", "none", 512),
        ("full_whitened", "none", VALID_RANK),
    )
    append_cells(
        family="primary_directional",
        target_block="directional",
        budget_labels=PRIMARY_BUDGETS,
        transforms=primary_transforms,
    )
    for target_block in ("volatility", "timing"):
        append_cells(
            family="specificity_control",
            target_block=target_block,
            budget_labels=CONTROL_BUDGETS,
            transforms=primary_transforms,
        )
    spectral_transforms = tuple(
        ("pca_coordinates", name, stop - start)
        for name, start, stop in SPECTRAL_BANDS
    ) + (
        ("pca_coordinates", "full_valid_rank", VALID_RANK),
        ("pca_coordinates", "top_128", 128),
    )
    for target_block in ("directional", "timing"):
        append_cells(
            family="spectral_diagnostic",
            target_block=target_block,
            budget_labels=SPECTRAL_BUDGETS,
            transforms=spectral_transforms,
        )
    for width in (128, 512):
        append_cells(
            family="capacity_sensitivity",
            target_block="directional",
            budget_labels=("b_1_4", "full_train"),
            transforms=(("native", "none", 512),),
            width=width,
        )

    logical_frame = pd.DataFrame(logical)
    duplicate_columns = [
        "job_family",
        "branch",
        "encoder_seed",
        "target_block",
        "transform",
        "spectral_arm",
        "budget_label",
        "subsample_seed",
        "width",
    ]
    if logical_frame.duplicated(duplicate_columns).any():
        raise ExperimentIntegrityError("Phase-III logical inventory contains duplicates")

    selection_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for logical_record in logical:
        logical_record["logical_job_key"] = _logical_job_key(logical_record)
        for weight_decay in WEIGHT_DECAYS:
            row = {
                **logical_record,
                "stage": "selection",
                "reader_seed": SELECTION_SEED,
                "weight_decay": float(weight_decay),
            }
            row["job_key"] = _job_key(row)
            selection_rows.append(row)
        for reader_seed in adaptive_reader_seeds(logical_record["budget_label"]):
            row = {
                **logical_record,
                "stage": "evaluation",
                "reader_seed": int(reader_seed),
                "weight_decay": np.nan,
            }
            row["job_key"] = _job_key({**row, "weight_decay": "selected"})
            evaluation_rows.append(row)
    selection = pd.DataFrame(selection_rows).sort_values("job_key").reset_index(drop=True)
    evaluation = pd.DataFrame(evaluation_rows).sort_values("job_key").reset_index(drop=True)
    if selection["job_key"].duplicated().any() or evaluation["job_key"].duplicated().any():
        raise ExperimentIntegrityError("Phase-III job keys are not unique")
    if len(selection) != len(logical_frame) * len(WEIGHT_DECAYS):
        raise ExperimentIntegrityError("selection inventory cardinality mismatch")
    return selection, evaluation


def write_phase3_job_inventory(
    phase1_dir: str | Path, out_dir: str | Path
) -> dict[str, Any]:
    root = Path(phase1_dir)
    verify_phase1_inputs(root)
    band_gate = verify_spectral_band_identity()
    payload = _load_subset_manifest(root)
    selection, evaluation = build_phase3_job_inventory(payload)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selection_path = destination / "selection_job_inventory.parquet"
    evaluation_path = destination / "evaluation_job_inventory.parquet"
    metadata_path = destination / "job_inventory_summary.json"
    atomic_write_parquet(selection, selection_path)
    atomic_write_parquet(evaluation, evaluation_path)
    logical_cells = len(selection) // len(WEIGHT_DECAYS)
    summary = {
        "schema_name": "thesis.experiment01.phase3.job_inventory",
        "schema_version": 1,
        "b_min_mlp": 0.25,
        "minimum_rows": MLP_MIN_ROWS,
        "low_budget_labels": list(LOW_BUDGETS),
        "selection_seed": SELECTION_SEED,
        "weight_decays": list(WEIGHT_DECAYS),
        "logical_selection_cells": logical_cells,
        "selection_models": int(len(selection)),
        "evaluation_models": int(len(evaluation)),
        "total_models": int(len(selection) + len(evaluation)),
        "selection_by_family": {
            str(key): int(value)
            for key, value in selection.groupby("job_family").size().items()
        },
        "evaluation_by_family": {
            str(key): int(value)
            for key, value in evaluation.groupby("job_family").size().items()
        },
        "spectral_band_gate": band_gate,
        "phase1_subset_manifest_sha256": PHASE1_SUBSET_MANIFEST_SHA256,
        "selection_inventory_sha256": sha256_file(selection_path),
        "evaluation_inventory_sha256": sha256_file(evaluation_path),
    }
    atomic_write_json(metadata_path, summary)
    summary["summary_sha256"] = sha256_file(metadata_path)
    return summary


def freeze_phase3_protocol(out_dir: str | Path) -> dict[str, Any]:
    """Serialize the definitive preregistered protocol before production test."""

    destination = Path(out_dir)
    path = destination / "protocol_frozen.json"
    hash_path = destination / "protocol_frozen.sha256"
    payload = {
        "schema_name": "thesis.experiment01.phase3.protocol",
        "schema_version": 1,
        "status": "frozen_pre_test",
        "specification_sha256": PHASE3_SPECIFICATION_SHA256,
        "phase1_outcome_unchanged": "A1",
        "primary": {
            "branches": list(PRIMARY_BRANCHES),
            "encoder_seeds": [0, 1, 2],
            "readout": PRIMARY_READOUT,
            "target_block": "directional",
            "transforms": ["native", "full_whitened"],
            "budget_labels": list(PRIMARY_BUDGETS),
            "low_budget_labels": list(LOW_BUDGETS),
        },
        "controls": {
            "target_blocks": ["volatility", "timing"],
            "budget_labels": list(CONTROL_BUDGETS),
        },
        "spectral": {
            "target_blocks": ["directional", "timing"],
            "budget_labels": list(SPECTRAL_BUDGETS),
            "arms": list(SPECTRAL_ARMS),
            "valid_rank": VALID_RANK,
        },
        "reader": {
            "architecture": "Linear(d,256)-GELU-Dropout(0.10)-Linear(256,T)",
            "native_input_coordinate_standardization": False,
            "target_standardization": "exact_labelled_subset_only",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay_grid": list(WEIGHT_DECAYS),
            "selection_seed": SELECTION_SEED,
            "reader_seeds": list(READER_SEEDS),
            "gradient_clip_norm": 5.0,
            "max_steps": 20_000,
            "min_steps": 1_000,
            "validation_interval": 500,
            "patience_evaluations": 6,
            "minimum_validation_improvement": 1e-5,
        },
        "metrics": {
            "ceiling_eligibility_threshold": 0.01,
            "robust_gap_delta": 0.10,
            "delta_sensitivity": [0.05, 0.15],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "negative_recovery_clipped": False,
        },
        "outcomes": {
            "R1_reader_attenuation_min": 0.50,
            "R2_whitening_attenuation_min": 0.50,
            "R3_full_whitened_mean_gap_min": 0.10,
            "robust_required_adjacent_levels": 2,
            "stability_rule": (
                "required qualitative gap/attenuation sign must hold in each of "
                "the three encoder-specific means; otherwise R4"
            ),
        },
        "test_policy": "blocked_until_selection_manifest_frozen_and_hashed",
    }
    if path.exists() or hash_path.exists():
        if not path.is_file() or not hash_path.is_file():
            raise ExperimentIntegrityError("partial Phase-III protocol freeze exists")
        observed = sha256_file(path)
        recorded = hash_path.read_text().strip().split()[0]
        if observed != recorded or json.loads(path.read_text()) != payload:
            raise ExperimentIntegrityError("frozen Phase-III protocol differs")
        return {"status": "already_frozen", "sha256": observed}
    atomic_write_json(path, payload)
    digest = sha256_file(path)
    hash_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return {"status": "frozen", "sha256": digest}


def load_subset_positions(
    phase1_dir: str | Path,
    subset_record: Mapping[str, Any],
    *,
    train_rows: pd.DataFrame | None = None,
) -> np.ndarray:
    """Load exact Phase-I row positions and optionally recheck source row identity."""

    path = Path(phase1_dir) / str(subset_record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(subset_record["size_bytes"])
        or sha256_file(path) != subset_record["sha256"]
    ):
        raise ExperimentIntegrityError(f"subset artifact mismatch: {path}")
    columns = ["source_row_position", "row_key"]
    subset = pd.read_parquet(path, columns=columns)
    if len(subset) != int(subset_record["n_rows"]):
        raise ExperimentIntegrityError(f"subset row count mismatch: {path}")
    row_keys = subset["row_key"].astype(str).to_numpy(dtype="U")
    if sha256_array(row_keys) != subset_record["row_key_sha256"]:
        raise ExperimentIntegrityError(f"subset row-key hash mismatch: {path}")
    positions = subset["source_row_position"].to_numpy(dtype=np.int64)
    if len(np.unique(positions)) != len(positions) or np.any(positions < 0):
        raise ExperimentIntegrityError(f"subset positions are invalid: {path}")
    if train_rows is not None:
        if len(positions) and int(positions.max()) >= len(train_rows):
            raise ExperimentIntegrityError(f"subset position is out of range: {path}")
        source_keys = (
            train_rows.iloc[positions]["row_key"].astype(str).to_numpy(dtype="U")
        )
        if not np.array_equal(source_keys, row_keys):
            raise ExperimentIntegrityError(f"subset/source row identity mismatch: {path}")
    return positions


def canonical_r2_from_sums(
    sse: np.ndarray, y_sum: np.ndarray, yty: np.ndarray, n_rows: int
) -> np.ndarray:
    total = np.asarray(yty, dtype=np.float64) - (
        np.asarray(y_sum, dtype=np.float64) ** 2 / int(n_rows)
    )
    return 1.0 - np.asarray(sse, dtype=np.float64) / np.maximum(total, 1e-12)


class TorchFeatureTransform:
    """Device-resident copy of one frozen, label-free feature transform."""

    def __init__(self, transform: FrozenFeatureTransform, device: str):
        import torch

        self.transform = transform
        self.device = torch.device(device)
        self.mean = torch.as_tensor(
            transform.mean, dtype=torch.float32, device=self.device
        )
        self.basis = (
            None
            if transform.basis is None
            else torch.as_tensor(
                transform.basis, dtype=torch.float32, device=self.device
            )
        )
        self.scales = (
            None
            if transform.scales is None
            else torch.as_tensor(
                transform.scales, dtype=torch.float32, device=self.device
            )
        )

    def __call__(self, values):
        import torch

        # Sharded NPY slices may be read-only memmaps; tensor() makes the
        # explicit safe device copy required for training.
        x = torch.tensor(values, dtype=torch.float32, device=self.device)
        x = x - self.mean
        if self.basis is not None:
            x = x @ self.basis
        if self.scales is not None:
            x = x * self.scales
        return x


def _torch_transform(values, transform: FrozenFeatureTransform, device):
    """Compatibility helper for smoke calls; training caches this object."""

    import torch

    del torch
    return TorchFeatureTransform(transform, str(device))(values)


def assert_test_access_allowed(
    selection_manifest_path: str | Path,
    selection_hash_path: str | Path | None = None,
) -> str:
    """Return the frozen manifest hash or fail before test features/labels load."""

    manifest_path = Path(selection_manifest_path)
    hash_path = (
        Path(selection_hash_path)
        if selection_hash_path is not None
        else manifest_path.with_suffix(".sha256")
    )
    if not manifest_path.is_file() or not hash_path.is_file():
        raise ExperimentIntegrityError(
            "test access blocked: selection manifest is not frozen and hashed"
        )
    observed = sha256_file(manifest_path)
    recorded = hash_path.read_text(encoding="utf-8").strip().split()[0]
    if observed != recorded:
        raise ExperimentIntegrityError(
            "test access blocked: selection manifest hash mismatch"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or payload.get("test_accessed") is not False:
        raise ExperimentIntegrityError(
            "test access blocked: selection manifest state is not frozen/pre-test"
        )
    return observed


def freeze_selection_manifest(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    selection_inventory_sha256: str,
) -> str:
    """Atomically freeze validation-only selections before any test evaluation."""

    destination = Path(path)
    hash_path = destination.with_suffix(".sha256")
    if destination.exists() or hash_path.exists():
        raise FileExistsError("refusing to overwrite a frozen selection manifest")
    required = {
        "job_key",
        "selected_weight_decay",
        "selected_checkpoint_step",
        "validation_r2",
        "training_seed",
        "input_transform_hash",
        "subset_hash",
        "model_definition_hash",
    }
    normalized: list[dict[str, Any]] = []
    for record in records:
        missing = sorted(required - set(record))
        if missing:
            raise ExperimentIntegrityError(
                f"selection record missing fields: {missing}"
            )
        if int(record["training_seed"]) != SELECTION_SEED:
            raise ExperimentIntegrityError("selection record uses a noncanonical seed")
        if float(record["selected_weight_decay"]) not in WEIGHT_DECAYS:
            raise ExperimentIntegrityError("selection record chose invalid weight decay")
        if not np.isfinite(float(record["validation_r2"])):
            raise ExperimentIntegrityError("selection validation R2 is non-finite")
        normalized.append(dict(record))
    normalized.sort(key=lambda value: str(value["job_key"]))
    if len({str(value["job_key"]) for value in normalized}) != len(normalized):
        raise ExperimentIntegrityError("selection manifest contains duplicate job keys")
    payload = {
        "schema_name": "thesis.experiment01.phase3.selection_manifest",
        "schema_version": 1,
        "status": "frozen",
        "test_accessed": False,
        "selection_seed": SELECTION_SEED,
        "tie_rule": "maximum_validation_r2_then_larger_weight_decay",
        "weight_decay_grid": list(WEIGHT_DECAYS),
        "selection_inventory_sha256": selection_inventory_sha256,
        "records": normalized,
    }
    atomic_write_json(destination, payload)
    digest = sha256_file(destination)
    hash_path.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return digest


def select_weight_decay(candidate_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Apply the exact validation tie rule: larger weight decay wins exact ties."""

    if len(candidate_rows) != len(WEIGHT_DECAYS):
        raise ExperimentIntegrityError("weight-decay selection needs three candidates")
    observed = {float(value["weight_decay"]) for value in candidate_rows}
    if observed != set(WEIGHT_DECAYS):
        raise ExperimentIntegrityError("weight-decay candidate grid is incomplete")
    if any(not np.isfinite(float(value["validation_r2"])) for value in candidate_rows):
        raise ExperimentIntegrityError("non-finite validation metric in selection")
    return max(
        candidate_rows,
        key=lambda value: (
            float(value["validation_r2"]), float(value["weight_decay"])
        ),
    )


def evaluate_mlp_r2(
    model,
    feature_source,
    target_source,
    transform: FrozenFeatureTransform,
    target_indices: Sequence[int],
    target_scaler: TargetStandardizer,
    *,
    split: str,
    device: str,
    chunk_rows: int,
    torch_transform: TorchFeatureTransform | None = None,
    selection_manifest_path: str | Path | None = None,
) -> np.ndarray:
    """Evaluate canonical per-target R2 without fitting any held-out statistic."""

    import torch

    if split == "test":
        if selection_manifest_path is None:
            raise ExperimentIntegrityError(
                "test access blocked: no frozen selection manifest supplied"
            )
        assert_test_access_allowed(selection_manifest_path)
    elif split not in {"validation", "synthetic"}:
        raise ValueError(f"unsupported MLP evaluation split {split!r}")
    dev = torch.device(device)
    transform_on_device = torch_transform or TorchFeatureTransform(
        transform, device
    )
    columns = np.asarray(tuple(target_indices), dtype=np.int64)
    sse = np.zeros(len(columns), dtype=np.float64)
    y_sum = np.zeros(len(columns), dtype=np.float64)
    yty = np.zeros(len(columns), dtype=np.float64)
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feature_source), chunk_rows):
            stop = min(start + chunk_rows, len(feature_source))
            raw_x = np.asarray(feature_source[start:stop], dtype=np.float32)
            raw_y = np.asarray(target_source[start:stop], dtype=np.float32)[:, columns]
            prediction_standard = model(
                transform_on_device(raw_x)
            ).detach().cpu().numpy()
            prediction = target_scaler.inverse(prediction_standard)
            residual = raw_y.astype(np.float64) - prediction.astype(np.float64)
            y64 = raw_y.astype(np.float64)
            sse += np.einsum("nt,nt->t", residual, residual)
            y_sum += y64.sum(axis=0)
            yty += np.einsum("nt,nt->t", y64, y64)
            count += len(raw_y)
    if count != len(feature_source):
        raise ExperimentIntegrityError("evaluation row count mismatch")
    return canonical_r2_from_sums(sse, y_sum, yty, count)


def train_validation_only_mlp(
    feature_train,
    target_train,
    labelled_positions: np.ndarray,
    feature_validation,
    target_validation,
    transform: FrozenFeatureTransform,
    target_indices: Sequence[int],
    target_scaler: TargetStandardizer,
    *,
    reader_seed: int,
    weight_decay: float,
    device: str,
    config: Phase3MLPConfig = Phase3MLPConfig(),
    primary_width: bool = True,
    enforce_preregistered_schedule: bool = True,
) -> dict[str, Any]:
    """Train with validation-only early stopping; this function has no test input."""

    import torch

    config.validate(
        primary=primary_width,
        enforce_preregistered_schedule=enforce_preregistered_schedule,
    )
    if float(weight_decay) not in WEIGHT_DECAYS and enforce_preregistered_schedule:
        raise ValueError("weight decay is outside the preregistered grid")
    positions = np.asarray(labelled_positions, dtype=np.int64)
    if len(positions) != target_scaler.n_rows:
        raise ExperimentIntegrityError("target scaler was not fit on this subset size")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm requested but unavailable")
    torch.manual_seed(int(reader_seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(reader_seed))
        torch.cuda.reset_peak_memory_stats(dev)
    rng = np.random.default_rng(int(reader_seed))
    model = make_primary_mlp(
        transform.output_dimension, len(tuple(target_indices)), width=config.width
    ).to(dev)
    transform_on_device = TorchFeatureTransform(transform, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=float(weight_decay)
    )
    loss_function = torch.nn.MSELoss(reduction="mean")
    batch_size = min(4096, len(positions))
    columns = np.asarray(tuple(target_indices), dtype=np.int64)
    best_score = -float("inf")
    best_step = 0
    best_state: dict[str, Any] | None = None
    best_target_r2: np.ndarray | None = None
    bad_evaluations = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_step = 0
    for step in range(1, config.max_steps + 1):
        sampled = rng.integers(0, len(positions), size=batch_size, endpoint=False)
        rows = positions[sampled]
        x_batch = np.asarray(feature_train[rows], dtype=np.float32)
        y_batch = np.asarray(target_train[rows], dtype=np.float32)[:, columns]
        y_standard = target_scaler.transform(y_batch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(transform_on_device(x_batch))
        truth = torch.as_tensor(y_standard, dtype=torch.float32, device=dev)
        loss = loss_function(prediction, truth)
        if not torch.isfinite(loss):
            raise ExperimentIntegrityError("non-finite Phase-III MLP training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise ExperimentIntegrityError("non-finite Phase-III MLP gradient norm")
        optimizer.step()
        last_step = step
        if step % config.validation_interval:
            continue
        target_r2 = evaluate_mlp_r2(
            model,
            feature_validation,
            target_validation,
            transform,
            target_indices,
            target_scaler,
            split="validation",
            device=device,
            chunk_rows=config.evaluation_chunk_rows,
            torch_transform=transform_on_device,
        )
        canonical_score = float(np.mean(target_r2))
        if not np.isfinite(canonical_score):
            raise ExperimentIntegrityError("non-finite Phase-III validation R2")
        improved = canonical_score > (
            best_score + config.minimum_validation_improvement
        )
        history.append(
            {
                "step": step,
                "training_loss": float(loss.detach().cpu().item()),
                "validation_r2": canonical_score,
                "improved": bool(improved),
            }
        )
        if improved:
            best_score = canonical_score
            best_step = step
            best_target_r2 = target_r2.copy()
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            bad_evaluations = 0
        else:
            bad_evaluations += 1
        if (
            step >= config.min_steps
            and bad_evaluations >= config.patience_evaluations
        ):
            break
    if best_state is None or best_target_r2 is None:
        raise ExperimentIntegrityError("MLP early stopping recorded no checkpoint")
    model.load_state_dict(best_state)
    peak_gpu = (
        int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
    )
    return {
        "model": model,
        "best_state": best_state,
        "best_step": int(best_step),
        "last_step": int(last_step),
        "validation_r2": float(best_score),
        "validation_r2_per_target": best_target_r2,
        "history": history,
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_bytes": peak_gpu,
        "batch_size": batch_size,
    }


def atomic_torch_save(state: Mapping[str, Any], path: str | Path) -> str:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(state), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(destination)


def _numpy_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    residual = np.einsum("nt,nt->t", truth - prediction, truth - prediction)
    centered = truth - truth.mean(axis=0, keepdims=True)
    total = np.einsum("nt,nt->t", centered, centered)
    return 1.0 - residual / np.maximum(total, 1e-12)


def run_synthetic_nonlinear_gate(
    *, device: str, seed: int = 314159
) -> dict[str, Any]:
    """Verify that the Phase-III reader detects a symmetric quadratic signal."""

    rng = np.random.default_rng(seed)
    n_train, n_validation, n_test, dimension = 12_288, 4_096, 4_096, 8
    x_train = rng.normal(size=(n_train, dimension)).astype(np.float32)
    x_validation = rng.normal(size=(n_validation, dimension)).astype(np.float32)
    x_test = rng.normal(size=(n_test, dimension)).astype(np.float32)

    def target(x: np.ndarray, noise: np.ndarray) -> np.ndarray:
        return (x[:, :1] ** 2 + 0.10 * noise[:, None]).astype(np.float32)

    y_train = target(x_train, rng.normal(size=n_train))
    y_validation = target(x_validation, rng.normal(size=n_validation))
    y_test = target(x_test, rng.normal(size=n_test))
    positions = np.arange(n_train, dtype=np.int64)
    subset_hash = sha256_array(positions)
    mean = x_train.astype(np.float64).mean(axis=0).astype(np.float32)
    transform_payload = {
        "gate": "symmetric_quadratic",
        "seed": seed,
        "mean_sha256": sha256_array(mean),
        "fit_split": "train_only",
    }
    transform = FrozenFeatureTransform(
        kind="native",
        mean=mean,
        basis=None,
        scales=None,
        input_dimension=dimension,
        output_dimension=dimension,
        transform_hash=canonical_json_sha256(transform_payload),
        source_transform_sha256="synthetic_train_only",
    )
    scaler = fit_target_standardizer(
        y_train, positions, (0,), subset_hash=subset_hash
    )
    design = np.column_stack(
        [np.ones(n_train), x_train.astype(np.float64) - mean]
    )
    beta = np.linalg.lstsq(design, y_train.astype(np.float64), rcond=None)[0]
    linear_prediction = np.column_stack(
        [np.ones(n_test), x_test.astype(np.float64) - mean]
    ) @ beta
    linear_r2 = float(_numpy_r2(y_test, linear_prediction)[0])
    smoke = Phase3MLPConfig(
        max_steps=2_000,
        min_steps=400,
        validation_interval=100,
        patience_evaluations=8,
        evaluation_chunk_rows=4_096,
    )
    trained = train_validation_only_mlp(
        x_train,
        y_train,
        positions,
        x_validation,
        y_validation,
        transform,
        (0,),
        scaler,
        reader_seed=seed,
        weight_decay=0.0,
        device=device,
        config=smoke,
        enforce_preregistered_schedule=False,
    )
    nonlinear_r2 = float(
        evaluate_mlp_r2(
            trained["model"],
            x_test,
            y_test,
            transform,
            (0,),
            scaler,
            split="synthetic",
            device=device,
            chunk_rows=4_096,
        )[0]
    )
    passed = abs(linear_r2) <= 0.05 and nonlinear_r2 >= 0.50
    result = {
        "schema_name": "thesis.experiment01.phase3.synthetic_nonlinear_gate",
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "seed": seed,
        "data": {
            "n_train": n_train,
            "n_validation": n_validation,
            "n_test": n_test,
            "dimension": dimension,
            "equation": "Y=Z_1^2+epsilon",
            "test_hash": sha256_array(np.column_stack([x_test, y_test])),
        },
        "linear_test_r2": linear_r2,
        "phase3_mlp_test_r2": nonlinear_r2,
        "mlp_best_step": trained["best_step"],
        "mlp_validation_r2": trained["validation_r2"],
        "training_signature_has_no_test_input": True,
        "thresholds": {"abs_linear_r2_max": 0.05, "mlp_r2_min": 0.50},
    }
    if not passed:
        raise ExperimentIntegrityError(f"synthetic nonlinear gate failed: {result}")
    return result


def _ridge_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    mean_x = x_train.mean(axis=0, keepdims=True)
    mean_y = y_train.mean(axis=0, keepdims=True)
    train = x_train - mean_x
    covariance = train.T @ train / len(train)
    cross = train.T @ (y_train - mean_y) / len(train)
    penalty = alpha * float(np.trace(covariance) / covariance.shape[0])
    weights = np.linalg.solve(
        covariance + penalty * np.eye(covariance.shape[0]), cross
    )
    return mean_y + (x_test - mean_x) @ weights


def run_synthetic_conditioning_gate(seed: int = 271828) -> dict[str, Any]:
    """Check finite-sample coordinate sensitivity and train-only whitening relief."""

    rng = np.random.default_rng(seed)
    n_train, n_test, dimension = 512, 8_192, 32
    z_train = rng.normal(size=(n_train, dimension))
    z_test = rng.normal(size=(n_test, dimension))
    beta = np.zeros((dimension, 1), dtype=np.float64)
    beta[0, 0] = 1.0
    y_train = z_train @ beta + 0.10 * rng.normal(size=(n_train, 1))
    y_test = z_test @ beta + 0.10 * rng.normal(size=(n_test, 1))
    scales = np.geomspace(1.0, 100.0, dimension)
    scales[0] = 1e-3
    x_train = z_train * scales
    x_test = z_test * scales
    oracle_x_beta = beta / scales[:, None]
    oracle_difference = float(
        np.max(np.abs((x_test @ oracle_x_beta) - (z_test @ beta)))
    )
    reconstruction_error = float(np.max(np.abs(x_test / scales - z_test)))

    train_mean = x_train.mean(axis=0)
    centered = x_train - train_mean
    covariance = centered.T @ centered / len(centered)
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if np.any(values <= 0.0):
        raise ExperimentIntegrityError("conditioning gate covariance lost rank")
    white_train = centered @ vectors / np.sqrt(values)
    white_test = (x_test - train_mean) @ vectors / np.sqrt(values)
    alpha = 0.05
    native_prediction = _ridge_prediction(
        x_train, y_train, x_test, alpha=alpha
    )
    white_prediction = _ridge_prediction(
        white_train, y_train, white_test, alpha=alpha
    )
    isotropic_prediction = _ridge_prediction(
        z_train, y_train, z_test, alpha=alpha
    )
    native_r2 = float(_numpy_r2(y_test, native_prediction)[0])
    white_r2 = float(_numpy_r2(y_test, white_prediction)[0])
    isotropic_r2 = float(_numpy_r2(y_test, isotropic_prediction)[0])
    penalty_reduction = white_r2 - native_r2
    restoration_error = abs(white_r2 - isotropic_r2)
    combined_mean = np.vstack([x_train, x_test]).mean(axis=0)
    train_only_witness = float(np.max(np.abs(train_mean - combined_mean)))
    passed = (
        oracle_difference <= 1e-10
        and reconstruction_error <= 1e-12
        and native_r2 + 0.25 < isotropic_r2
        and penalty_reduction >= 0.25
        and restoration_error <= 0.08
        and train_only_witness > 0.0
    )
    result = {
        "schema_name": "thesis.experiment01.phase3.synthetic_conditioning_gate",
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "seed": seed,
        "invertible_scale_condition_number": float(scales.max() / scales.min()),
        "oracle_prediction_max_abs_difference": oracle_difference,
        "inverse_transform_max_abs_error": reconstruction_error,
        "native_test_r2": native_r2,
        "whitened_test_r2": white_r2,
        "isotropic_reference_test_r2": isotropic_r2,
        "whitening_r2_improvement": penalty_reduction,
        "whitened_isotropic_abs_difference": restoration_error,
        "whitening_fit_split": "unlabelled_train_only",
        "train_vs_combined_mean_witness": train_only_witness,
        "ridge_alpha_trace_normalized": alpha,
    }
    if not passed:
        raise ExperimentIntegrityError(f"synthetic conditioning gate failed: {result}")
    return result


def verify_full_budget_linear_parity(
    phase1_dir: str | Path, definitions: Sequence[TargetDefinition]
) -> dict[str, Any]:
    """Read (never refit) exact frozen full-budget linear rows."""

    root = Path(phase1_dir)
    verify_phase1_inputs(root)
    expected_targets = {
        target.name
        for target in definitions
        if target.independent and target.block in {"directional", "volatility", "timing"}
    }
    columns = [
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "target_independent",
        "budget_kind",
        "subsample_seed",
        "feature_view",
        "whiten_k_effective",
        "reader_family",
        "alpha_selected",
        "test_r2",
        "fit_status",
    ]
    frame = pd.read_parquet(root / "results.parquet", columns=columns)
    common = (
        frame["branch"].isin(PRIMARY_BRANCHES)
        & frame["encoder_seed"].isin((0, 1, 2))
        & (frame["readout"] == PRIMARY_READOUT)
        & frame["target_independent"].astype(bool)
        & frame["target_name"].isin(expected_targets)
        & (frame["budget_kind"] == "full_train")
        & (frame["subsample_seed"] == -1)
        & frame["alpha_selected"].astype(bool)
        & (frame["fit_status"] == "ok")
    )
    native = frame.loc[
        common
        & (frame["reader_family"] == "ridge_raw_tuned_alpha")
        & (frame["feature_view"] == "full_rank_raw")
    ].copy()
    whitened = frame.loc[
        common
        & (frame["reader_family"] == "ridge_whiten_topk_tuned_alpha")
        & (frame["feature_view"] == "full_rank_whiten_topk")
        & (frame["whiten_k_effective"] == VALID_RANK)
    ].copy()
    key = ["branch", "encoder_seed", "target_block", "target_name"]
    expected_count = len(PRIMARY_BRANCHES) * 3 * len(expected_targets)
    if (
        len(native) != expected_count
        or len(whitened) != expected_count
        or native.duplicated(key).any()
        or whitened.duplicated(key).any()
        or set(native["target_name"]) != expected_targets
        or set(whitened["target_name"]) != expected_targets
    ):
        raise ExperimentIntegrityError("frozen full-budget linear row identity failed")
    joined = native[key + ["test_r2"]].merge(
        whitened[key + ["test_r2"]],
        on=key,
        suffixes=("_native", "_full_whitened"),
        validate="one_to_one",
    )
    return {
        "status": "pass",
        "source_results_sha256": PHASE1_RESULTS_SHA256,
        "row_count_per_reader": expected_count,
        "joined_row_count": int(len(joined)),
        "row_identity_sha256": sha256_array(
            joined[key].astype(str).agg("\x1f".join, axis=1).to_numpy(dtype="U")
        ),
        "block_means": [
            {
                "branch": branch,
                "target_block": block,
                "native_test_r2": float(group["test_r2_native"].mean()),
                "full_whitened_test_r2": float(
                    group["test_r2_full_whitened"].mean()
                ),
            }
            for (branch, block), group in joined.groupby(
                ["branch", "target_block"], observed=True, sort=True
            )
        ],
    }


def verify_phase2_projection_identity(
    bundle: InputBundle,
    phase1_dir: str | Path,
    phase2_dir: str | Path,
    *,
    rows: int = 4096,
) -> dict[str, Any]:
    """Verify exact transform provenance and independent PCA projection parity."""

    phase2_root = Path(phase2_dir)
    _require_sha256(
        phase2_root / "manifest.json", PHASE2_MANIFEST_SHA256, "Phase-II manifest"
    )
    checks = []
    for branch in PRIMARY_BRANCHES:
        for encoder_seed in (0, 1, 2):
            feature = bundle.feature_set(branch, encoder_seed, PRIMARY_READOUT)
            record, transform_path = _phase1_transform_record(Path(phase1_dir), feature)
            cache_json_path = (
                phase2_root
                / "cache"
                / f"{branch}_seed{encoder_seed}_{PRIMARY_READOUT}.json"
            )
            cache = json.loads(cache_json_path.read_text(encoding="utf-8"))
            if cache["source_fingerprint"]["transform_sha256"] != record["sha256"]:
                raise ExperimentIntegrityError("Phase-II PCA transform provenance mismatch")
            if int(cache["valid_dimension"]) != VALID_RANK:
                raise ExperimentIntegrityError("Phase-II valid rank differs from 508")
            top = load_frozen_feature_transform(
                phase1_dir, feature, kind="pca_coordinates", spectral_arm="top_128"
            )
            full = load_frozen_feature_transform(
                phase1_dir,
                feature,
                kind="pca_coordinates",
                spectral_arm="full_valid_rank",
            )
            raw = np.asarray(
                bundle.load_features(feature, "train")[:rows], dtype=np.float32
            )
            with np.load(transform_path, allow_pickle=False) as data:
                mean = np.asarray(data["unlabelled_train_mean"], dtype=np.float64)
                vectors = np.asarray(data["covariance_eigenvectors"], dtype=np.float64)
            phase2_full = (raw.astype(np.float64) - mean) @ vectors[:, :VALID_RANK]
            phase2_top = phase2_full[:, :128]
            observed_full = full.apply_numpy(raw).astype(np.float64)
            observed_top = top.apply_numpy(raw).astype(np.float64)
            full_error = float(np.max(np.abs(observed_full - phase2_full)))
            top_error = float(np.max(np.abs(observed_top - phase2_top)))
            tolerance = 2e-4
            if full_error > tolerance or top_error > tolerance:
                raise ExperimentIntegrityError(
                    f"Phase-II PCA projection parity failed for {feature.key}"
                )
            checks.append(
                {
                    "branch": branch,
                    "encoder_seed": encoder_seed,
                    "readout": PRIMARY_READOUT,
                    "rows": min(rows, len(raw)),
                    "top128_max_abs_error": top_error,
                    "full_rank_max_abs_error": full_error,
                    "tolerance": tolerance,
                    "source_transform_sha256": record["sha256"],
                    "top128_transform_hash": top.transform_hash,
                    "full_rank_transform_hash": full.transform_hash,
                }
            )
    return {
        "status": "pass",
        "phase2_manifest_sha256": PHASE2_MANIFEST_SHA256,
        "spectral_band_identity": verify_spectral_band_identity(),
        "checks": checks,
    }


def run_historical_mlp_gate(
    historical_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str,
    tolerance: float = 0.015,
) -> dict[str, Any]:
    """Retrain the exact stochastic post-P0 reader and compare frozen aggregates."""

    device_record = initialize_compute_device(device)
    source_path = (
        Path(__file__).resolve().parent
        / "historical/ladder_accessibility.py"
    )
    _require_sha256(
        source_path, HISTORICAL_MLP_SOURCE_SHA256, "historical MLP implementation"
    )
    from experiment01.historical.ladder_accessibility import (
        dir_indices,
        mlp_ceiling,
        validate_stage1_inputs,
    )

    root = Path(historical_dir).resolve()
    inventory = validate_stage1_inputs(root, None)
    if int(inventory["manifest"]["protocol"]["split_seed"]) != 0:
        raise ExperimentIntegrityError("historical MLP split seed differs from zero")
    indices = dir_indices()["dir_indep"]
    names = dir_indices()["names"]
    directional_names = [names[index] for index in indices]
    with np.load(inventory["targets_path"], allow_pickle=False) as targets:
        y_train = targets["y_train_raw"].astype(np.float64)
        y_validation = targets["y_val_raw"].astype(np.float64)
    destination = Path(out_dir)
    shards = destination / "historical_mlp_gate_shards"
    shards.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    started = time.perf_counter()
    for branch in PRIMARY_BRANCHES:
        for encoder_seed in (0, 1, 2):
            readout_path = (
                root
                / "readouts"
                / f"{branch}_seed{encoder_seed}_ep020.npz"
            )
            if not readout_path.is_file():
                raise ExperimentIntegrityError(
                    f"historical readout is missing: {readout_path}"
                )
            shard_path = shards / f"{branch}_seed{encoder_seed}.parquet"
            state_path = shards / f"{branch}_seed{encoder_seed}.json"
            fingerprint = {
                "historical_source_sha256": HISTORICAL_MLP_SOURCE_SHA256,
                "readout_sha256": sha256_file(readout_path),
                "targets_sha256": sha256_file(inventory["targets_path"]),
                "branch": branch,
                "encoder_seed": encoder_seed,
                "reader_seeds": list(READER_SEEDS),
                "epochs": 80,
                "patience": 10,
                "split_seed": 0,
                "internal_validation_fraction": 0.10,
                "device": device,
            }
            if shard_path.is_file() and state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if (
                    state.get("fingerprint") != fingerprint
                    or state.get("sha256") != sha256_file(shard_path)
                ):
                    raise ExperimentIntegrityError(
                        f"stale historical gate shard: {branch}/seed{encoder_seed}"
                    )
                all_rows.append(pd.read_parquet(shard_path))
                continue
            with np.load(readout_path, allow_pickle=False) as readout:
                x_train = np.asarray(
                    readout["last_concat512_train"], dtype=np.float32
                )
                x_validation = np.asarray(
                    readout["last_concat512_val"], dtype=np.float32
                )
            result = mlp_ceiling(
                x_train,
                x_validation,
                y_train,
                y_validation,
                device,
                hidden=256,
                epochs=80,
                lr=1e-3,
                wd=1e-4,
                mlp_seeds=5,
                patience=10,
                split_seed=0,
                internal_val_fraction=0.10,
                batch_size=4096,
            )
            rows = []
            for reader_index, reader_seed in enumerate(result["reader_seeds"]):
                for target_index in indices:
                    rows.append(
                        {
                            "branch": branch,
                            "encoder_seed": encoder_seed,
                            "readout": PRIMARY_READOUT,
                            "target_block": "directional",
                            "target_name": names[target_index],
                            "reader_seed": int(reader_seed),
                            "test_split_role": "historical_outer_validation",
                            "r2": float(result["runs"][reader_index, target_index]),
                            "best_epoch": int(result["epochs_used"][reader_index]),
                        }
                    )
            frame = pd.DataFrame(rows).sort_values(
                ["reader_seed", "target_name"], kind="stable"
            )
            atomic_write_parquet(frame, shard_path)
            atomic_write_json(
                state_path,
                {
                    "schema_name": "thesis.experiment01.phase3.historical_mlp_gate_shard",
                    "schema_version": 1,
                    "fingerprint": fingerprint,
                    "sha256": sha256_file(shard_path),
                    "rows": len(frame),
                },
            )
            all_rows.append(frame)
    runs = pd.concat(all_rows, ignore_index=True)
    expected_rows = len(PRIMARY_BRANCHES) * 3 * 5 * len(directional_names)
    if len(runs) != expected_rows:
        raise ExperimentIntegrityError("historical MLP gate run inventory is incomplete")
    summaries = []
    passed = True
    for branch in PRIMARY_BRANCHES:
        branch_rows = runs.loc[runs["branch"] == branch]
        encoder_means = []
        encoder_reader_variances = []
        for encoder_seed, encoder in branch_rows.groupby("encoder_seed", sort=True):
            target_reader = encoder.pivot(
                index="reader_seed", columns="target_name", values="r2"
            )
            target_means = target_reader.mean(axis=0).to_numpy(dtype=float)
            target_reader_sd = target_reader.std(axis=0, ddof=0).to_numpy(dtype=float)
            encoder_means.append(float(target_means.mean()))
            encoder_reader_variances.append(float(np.mean(target_reader_sd**2)))
        observed = float(np.mean(encoder_means))
        reference = HISTORICAL_MLP_REFERENCES[branch]
        difference = observed - reference
        branch_passed = abs(difference) <= tolerance
        passed = passed and branch_passed
        summaries.append(
            {
                "branch": branch,
                "observed_r2": observed,
                "reference_r2": reference,
                "difference": difference,
                "absolute_tolerance": tolerance,
                "encoder_std": float(np.std(encoder_means, ddof=0)),
                "reader_std": float(np.sqrt(np.mean(encoder_reader_variances))),
                "reader_seed_min_mean": float(
                    branch_rows.groupby("reader_seed")["r2"].mean().min()
                ),
                "reader_seed_max_mean": float(
                    branch_rows.groupby("reader_seed")["r2"].mean().max()
                ),
                "passed": branch_passed,
            }
        )
    runs_path = destination / "historical_mlp_gate_runs.parquet"
    gate_path = destination / "historical_mlp_gate.json"
    atomic_write_parquet(runs, runs_path)
    gate = {
        "schema_name": "thesis.experiment01.phase3.historical_mlp_gate",
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "reproduction_mode": "stochastic_retraining_no_saved_checkpoint_or_predictions",
        "historical_source_sha256": HISTORICAL_MLP_SOURCE_SHA256,
        "runs_sha256": sha256_file(runs_path),
        "summaries": summaries,
        "runtime_seconds": float(time.perf_counter() - started),
        "device": device_record,
        "semantics": {
            "coordinatewise_input_standardization": True,
            "batch_norm": False,
            "layer_norm": False,
            "architecture": "512-256-256-T with GELU",
            "intentional_phase3_difference_confirmed": True,
        },
    }
    atomic_write_json(gate_path, gate)
    if not passed:
        raise ExperimentIntegrityError(f"historical MLP gate failed: {summaries}")
    return gate


def run_phase3_preproduction_gates(
    bundle_dir: str | Path,
    phase1_dir: str | Path,
    phase2_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Run all non-historical fail-closed acceptance gates."""

    from .schema import load_input_bundle

    device_record = initialize_compute_device(device)
    bundle_root = Path(bundle_dir)
    _require_sha256(
        bundle_root / "manifest.json", BUNDLE_MANIFEST_SHA256, "bundle manifest"
    )
    destination = Path(out_dir)
    identity_path = destination / "artifact_identity_gate.json"
    if not identity_path.is_file():
        raise ExperimentIntegrityError("the complete artifact identity gate is missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("status") != "pass" or identity.get("bundle", {}).get(
        "verify_hashes"
    ) is not True:
        raise ExperimentIntegrityError("the complete artifact identity gate did not pass")
    bundle = load_input_bundle(
        bundle_root, verify_hashes=False, check_finite=False
    )
    gates: dict[str, Any] = {
        "conditioning": run_synthetic_conditioning_gate(),
        "nonlinear": run_synthetic_nonlinear_gate(device=device),
        "linear_parity": verify_full_budget_linear_parity(
            phase1_dir, bundle.target_definitions
        ),
        "pca_band_identity": verify_phase2_projection_identity(
            bundle, phase1_dir, phase2_dir
        ),
    }
    for name, payload in gates.items():
        atomic_write_json(destination / f"{name}_gate.json", payload)
    result = {
        "schema_name": "thesis.experiment01.phase3.preproduction_gates",
        "schema_version": 1,
        "status": "pass",
        "device": device_record,
        "gates": {
            name: {
                "status": payload["status"],
                "path": f"{name}_gate.json",
                "sha256": sha256_file(destination / f"{name}_gate.json"),
            }
            for name, payload in gates.items()
        },
        "historical_gate_required_separately": True,
    }
    atomic_write_json(destination / "preproduction_gates.json", result)
    return result


def add_targetwise_normalized_recovery(results: pd.DataFrame) -> pd.DataFrame:
    """Attach same-reader-family target ceilings without clipping recovery."""

    required = {
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "transform",
        "spectral_arm",
        "budget_label",
        "reader_seed",
        "width",
        "test_r2",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Phase-III results missing recovery columns: {missing}")
    group_key = [
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "transform",
        "spectral_arm",
        "width",
    ]
    clean = results.drop(
        columns=[
            name
            for name in (
                "full_budget_ceiling",
                "ceiling_eligible",
                "normalized_recovery",
            )
            if name in results.columns
        ]
    )
    full = clean.loc[clean["budget_label"] == "full_train"].copy()
    ceilings = (
        full.groupby(group_key, dropna=False, observed=True)["test_r2"]
        .mean()
        .rename("full_budget_ceiling")
        .reset_index()
    )
    if ceilings.duplicated(group_key).any():
        raise ExperimentIntegrityError("full-budget ceiling is not unique")
    out = clean.merge(ceilings, on=group_key, how="left", validate="many_to_one")
    if out["full_budget_ceiling"].isna().any():
        raise ExperimentIntegrityError("a result row lacks its full-budget ceiling")
    out["ceiling_eligible"] = out["full_budget_ceiling"] >= 0.01
    out["normalized_recovery"] = np.where(
        out["ceiling_eligible"],
        out["test_r2"] / out["full_budget_ceiling"],
        np.nan,
    )
    return out


def variance_components(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Separate reader, subset, and encoder dispersion hierarchically."""

    identity = [
        "job_family",
        "readout",
        "target_block",
        "target_name",
        "transform",
        "spectral_arm",
        "budget_label",
        "branch",
        "width",
    ]
    required = set(identity) | {
        "encoder_seed",
        "subsample_seed",
        "reader_seed",
        metric,
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"variance input missing columns: {missing}")
    rows = []
    for key, group in results.groupby(identity, dropna=False, observed=True, sort=True):
        subset_reader = group.groupby(
            ["encoder_seed", "subsample_seed"], observed=True
        )[metric]
        reader_variances = subset_reader.var(ddof=0).fillna(0.0).to_numpy(dtype=float)
        subset_means = subset_reader.mean().reset_index()
        subsample_variances = (
            subset_means.groupby("encoder_seed", observed=True)[metric]
            .var(ddof=0)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        encoder_means = (
            subset_means.groupby("encoder_seed", observed=True)[metric]
            .mean()
            .to_numpy(dtype=float)
        )
        rows.append(
            {
                **dict(zip(identity, key)),
                "metric": metric,
                "sd_reader_within_subset_encoder": float(
                    np.sqrt(np.mean(reader_variances))
                ),
                "sd_subsample_within_encoder": float(
                    np.sqrt(np.mean(subsample_variances))
                ),
                "sd_encoder_between_means": float(np.std(encoder_means, ddof=0)),
                "n_encoder_seeds": int(len(encoder_means)),
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def paired_metric_difference(
    results: pd.DataFrame,
    *,
    metric: str,
    comparison_column: str,
    left_value: str,
    right_value: str,
) -> pd.DataFrame:
    """Return strictly paired left-minus-right reader/subset/encoder differences."""

    pair_key = [
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "budget_label",
        "subsample_seed",
        "reader_seed",
        "width",
    ]
    retain = pair_key + [comparison_column, metric]
    subset = results.loc[
        results[comparison_column].isin((left_value, right_value)), retain
    ]
    if subset.duplicated(pair_key + [comparison_column]).any():
        raise ExperimentIntegrityError("paired comparison cells are not unique")
    left = subset.loc[subset[comparison_column] == left_value, pair_key + [metric]].rename(
        columns={metric: f"{metric}_{left_value}"}
    )
    right = subset.loc[
        subset[comparison_column] == right_value, pair_key + [metric]
    ].rename(columns={metric: f"{metric}_{right_value}"})
    paired = left.merge(right, on=pair_key, how="inner", validate="one_to_one")
    expected = max(len(left), len(right))
    if len(paired) != expected:
        raise ExperimentIntegrityError("paired comparison has unmatched cells")
    paired["difference"] = (
        paired[f"{metric}_{left_value}"] - paired[f"{metric}_{right_value}"]
    )
    paired["comparison"] = f"{left_value}_minus_{right_value}"
    return paired


def audit_historical_mlp_semantics(repo_root: str | Path) -> dict[str, Any]:
    """Static identity gate for the code and frozen historical reference tables."""

    root = Path(repo_root)
    source = root / "experiment01/historical/ladder_accessibility.py"
    _require_sha256(source, HISTORICAL_MLP_SOURCE_SHA256, "historical MLP source")
    aggregate = (
        root
        / "validation/readouts_v2_20260728/analysis_consolidation_20260728/mlp_agg.csv"
    )
    runs = aggregate.with_name("mlp_reader_runs.csv")
    _require_sha256(
        aggregate,
        "bb39df20bd28bb11c9fa59ed33af643d5d77f2fc68ae66d4b7644790e41ec8a5",
        "historical MLP aggregate",
    )
    _require_sha256(
        runs,
        "52b661f57f7ce9bc815da92166738ee9b11b6bc288bb9e57f4c2927af562b0bc",
        "historical MLP reader runs",
    )
    frame = pd.read_csv(aggregate)
    observed = frame.loc[
        (frame["pooling"] == PRIMARY_READOUT)
        & (frame["block"] == "dir")
        & frame["arm"].isin(PRIMARY_BRANCHES)
    ].set_index("arm")["r2_mean"]
    if set(observed.index) != set(PRIMARY_BRANCHES):
        raise ExperimentIntegrityError("historical aggregate identity is incomplete")
    return {
        "status": "pass",
        "source_sha256": HISTORICAL_MLP_SOURCE_SHA256,
        "coordinatewise_input_standardization": True,
        "batch_norm": False,
        "layer_norm": False,
        "hidden_layers": [256, 256],
        "dropout": 0.0,
        "references": {branch: float(observed[branch]) for branch in PRIMARY_BRANCHES},
    }


def verify_completed_cell(
    state_path: str | Path, expected_fingerprint: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Verify immutable completed-cell outputs before a restart skips the job."""

    path = Path(state_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise ExperimentIntegrityError("cell state is not complete")
    if state.get("fingerprint") != dict(expected_fingerprint):
        raise ExperimentIntegrityError("completed-cell fingerprint mismatch")
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ExperimentIntegrityError("completed cell has no artifact records")
    for artifact in artifacts:
        artifact_path = path.parent / str(artifact.get("path", ""))
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != int(artifact.get("size_bytes", -1))
            or sha256_file(artifact_path) != artifact.get("sha256")
        ):
            raise ExperimentIntegrityError(
                f"completed-cell artifact failed identity: {artifact_path}"
            )
    return state


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _row_record(row: pd.Series) -> dict[str, Any]:
    return {str(key): _json_scalar(value) for key, value in row.to_dict().items()}


def _phase3_prerequisites(out_dir: Path, *, historical: bool) -> None:
    protocol = out_dir / "protocol_frozen.json"
    protocol_hash = out_dir / "protocol_frozen.sha256"
    if not protocol.is_file() or not protocol_hash.is_file():
        raise ExperimentIntegrityError("the frozen Phase-III protocol is missing")
    observed_protocol_hash = sha256_file(protocol)
    recorded_protocol_hash = protocol_hash.read_text(encoding="utf-8").strip().split()[0]
    if observed_protocol_hash != recorded_protocol_hash:
        raise ExperimentIntegrityError("the frozen Phase-III protocol hash differs")
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    if protocol_payload.get("status") != "frozen_pre_test":
        raise ExperimentIntegrityError("the Phase-III protocol is not frozen pre-test")
    identity = out_dir / "artifact_identity_gate.json"
    preproduction = out_dir / "preproduction_gates.json"
    for path in (identity, preproduction):
        if not path.is_file() or json.loads(path.read_text()).get("status") != "pass":
            raise ExperimentIntegrityError(f"Phase-III prerequisite did not pass: {path}")
    if historical:
        path = out_dir / "historical_mlp_gate.json"
        if not path.is_file() or json.loads(path.read_text()).get("status") != "pass":
            raise ExperimentIntegrityError("historical MLP reproduction gate did not pass")
    if protocol_payload.get("schema_name") == "thesis.experiment01.phase3_reduced.protocol":
        preparation = out_dir / "reduced_preparation_gate.json"
        if (
            not preparation.is_file()
            or json.loads(preparation.read_text(encoding="utf-8")).get("status")
            != "pass"
        ):
            raise ExperimentIntegrityError("Phase III-R preparation gate did not pass")
        inventory = protocol_payload.get("inventory", {})
        for name, expected in (
            ("selection_job_inventory.parquet", inventory.get("selection_inventory_sha256")),
            ("evaluation_job_inventory.parquet", inventory.get("evaluation_inventory_sha256")),
        ):
            path = out_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ExperimentIntegrityError(f"Phase III-R inventory differs: {name}")


class _ExecutionCaches:
    def __init__(
        self,
        bundle: InputBundle,
        phase1_dir: Path,
        subset_records: Mapping[str, Mapping[str, Any]],
    ):
        self.bundle = bundle
        self.phase1_dir = phase1_dir
        self.subset_records = subset_records
        self.positions: dict[str, np.ndarray] = {}
        self.target_scalers: dict[tuple[str, str], TargetStandardizer] = {}
        self.transforms: dict[tuple[str, int, str, str], FrozenFeatureTransform] = {}
        self.target_train = bundle.load_targets("train")
        self.target_validation = bundle.load_targets("validation")

    def positions_for(self, subset_path: str) -> np.ndarray:
        if subset_path not in self.positions:
            self.positions[subset_path] = load_subset_positions(
                self.phase1_dir,
                self.subset_records[subset_path],
                train_rows=self.bundle.rows["train"],
            )
        return self.positions[subset_path]

    def scaler_for(
        self, subset_path: str, target_block: str
    ) -> tuple[TargetStandardizer, tuple[int, ...]]:
        key = (subset_path, target_block)
        indices = target_indices_for_block(
            self.bundle.target_definitions, target_block
        )
        if key not in self.target_scalers:
            record = self.subset_records[subset_path]
            self.target_scalers[key] = fit_target_standardizer(
                self.target_train,
                self.positions_for(subset_path),
                indices,
                subset_hash=str(record["row_key_sha256"]),
            )
        return self.target_scalers[key], indices

    def transform_for(
        self, feature: FeatureSet, kind: str, spectral_arm: str
    ) -> FrozenFeatureTransform:
        key = (feature.branch, feature.encoder_seed, kind, spectral_arm)
        if key not in self.transforms:
            self.transforms[key] = load_frozen_feature_transform(
                self.phase1_dir,
                feature,
                kind=kind,
                spectral_arm=spectral_arm,
            )
        return self.transforms[key]


def _selection_candidate_fingerprint(
    job: Mapping[str, Any], transform: FrozenFeatureTransform
) -> dict[str, Any]:
    return {
        "schema": "phase3_selection_candidate.v1",
        "job": dict(job),
        "transform_hash": transform.transform_hash,
        "phase1_results_sha256": PHASE1_RESULTS_SHA256,
        "phase1_subset_manifest_sha256": PHASE1_SUBSET_MANIFEST_SHA256,
        "phase2_manifest_sha256": PHASE2_MANIFEST_SHA256,
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
        "training_config": asdict(Phase3MLPConfig(width=int(job["width"]))),
        "mixed_precision": False,
    }


def _write_completed_state(
    directory: Path,
    fingerprint: Mapping[str, Any],
    artifact_paths: Sequence[Path],
) -> None:
    artifacts = [
        {
            "path": str(path.relative_to(directory)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in artifact_paths
    ]
    atomic_write_json(
        directory / "complete.json",
        {
            "schema_name": "thesis.experiment01.phase3.completed_cell",
            "schema_version": 1,
            "status": "complete",
            "fingerprint": dict(fingerprint),
            "artifacts": artifacts,
        },
    )


def _run_selection_candidate(
    bundle: InputBundle,
    phase1_dir: Path,
    out_dir: Path,
    caches: _ExecutionCaches,
    job: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    job_dir = out_dir / "selection_jobs" / str(job["job_key"])
    state_path = job_dir / "complete.json"
    feature = bundle.feature_set(
        str(job["branch"]), int(job["encoder_seed"]), str(job["readout"])
    )
    transform = caches.transform_for(
        feature, str(job["transform"]), str(job["spectral_arm"])
    )
    if transform.output_dimension != int(job["input_dimension"]):
        raise ExperimentIntegrityError("job/transform dimension mismatch")
    fingerprint = _selection_candidate_fingerprint(job, transform)
    if state_path.is_file():
        verify_completed_cell(state_path, fingerprint)
        return json.loads((job_dir / "metrics.json").read_text())
    job_dir.mkdir(parents=True, exist_ok=True)
    positions = caches.positions_for(str(job["subset_path"]))
    scaler, target_indices = caches.scaler_for(
        str(job["subset_path"]), str(job["target_block"])
    )
    config = Phase3MLPConfig(width=int(job["width"]))
    trained = train_validation_only_mlp(
        bundle.load_features(feature, "train"),
        caches.target_train,
        positions,
        bundle.load_features(feature, "validation"),
        caches.target_validation,
        transform,
        target_indices,
        scaler,
        reader_seed=SELECTION_SEED,
        weight_decay=float(job["weight_decay"]),
        device=device,
        config=config,
        primary_width=int(job["width"]) == 256,
    )
    checkpoint_path = job_dir / "checkpoint.pt"
    checkpoint_sha = atomic_torch_save(trained["best_state"], checkpoint_path)
    history_path = job_dir / "history.parquet"
    atomic_write_parquet(pd.DataFrame(trained["history"]), history_path)
    target_names = [bundle.target_names[index] for index in target_indices]
    metrics = {
        "job_key": str(job["job_key"]),
        "logical_job_key": str(job["logical_job_key"]),
        "weight_decay": float(job["weight_decay"]),
        "validation_r2": float(trained["validation_r2"]),
        "validation_r2_per_target": {
            name: float(value)
            for name, value in zip(target_names, trained["validation_r2_per_target"])
        },
        "best_step": int(trained["best_step"]),
        "last_step": int(trained["last_step"]),
        "runtime_seconds": float(trained["runtime_seconds"]),
        "peak_gpu_memory_bytes": int(trained["peak_gpu_memory_bytes"]),
        "checkpoint_path": str(checkpoint_path.relative_to(out_dir)),
        "checkpoint_sha256": checkpoint_sha,
        "input_transform_hash": transform.transform_hash,
        "source_transform_sha256": transform.source_transform_sha256,
        "target_scaler_mean_sha256": sha256_array(scaler.mean),
        "target_scaler_scale_sha256": sha256_array(scaler.scale),
        "target_scaler_subset_hash": scaler.source_subset_hash,
    }
    metrics_path = job_dir / "metrics.json"
    atomic_write_json(metrics_path, metrics)
    _write_completed_state(
        job_dir, fingerprint, (metrics_path, checkpoint_path, history_path)
    )
    return metrics


def run_phase3_selection(
    bundle_dir: str | Path,
    phase1_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Run all validation-only candidates and freeze selection before test."""

    from .schema import load_input_bundle

    initialize_compute_device(device)
    destination = Path(out_dir)
    _phase3_prerequisites(destination, historical=True)
    manifest_path = destination / "selection_manifest.json"
    if manifest_path.exists():
        digest = assert_test_access_allowed(manifest_path)
        return {
            "status": "already_frozen",
            "selection_manifest_sha256": digest,
        }
    inventory_path = destination / "selection_job_inventory.parquet"
    summary = json.loads((destination / "job_inventory_summary.json").read_text())
    if sha256_file(inventory_path) != summary["selection_inventory_sha256"]:
        raise ExperimentIntegrityError("selection inventory hash mismatch")
    inventory = pd.read_parquet(inventory_path)
    bundle = load_input_bundle(bundle_dir, verify_hashes=False, check_finite=False)
    subset_payload = _load_subset_manifest(Path(phase1_dir))
    subset_records = {
        str(record["path"]): record for record in subset_payload["subsets"]
    }
    caches = _ExecutionCaches(bundle, Path(phase1_dir), subset_records)
    selection_records = []
    failures = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    order = [
        "branch",
        "encoder_seed",
        "transform",
        "spectral_arm",
        "target_block",
        "budget_stock_day_equivalents",
        "subsample_seed",
        "width",
        "weight_decay",
    ]
    inventory = inventory.sort_values(order, kind="stable")
    grouped_inventory = inventory.groupby(
        "logical_job_key", sort=False, observed=True
    )
    for logical_index, (logical_key, candidates) in enumerate(grouped_inventory, start=1):
        candidate_metrics = []
        try:
            for _, row in candidates.sort_values("weight_decay").iterrows():
                candidate_metrics.append(
                    _run_selection_candidate(
                        bundle,
                        Path(phase1_dir),
                        destination,
                        caches,
                        _row_record(row),
                        device=device,
                    )
                )
                peak_rss = max(peak_rss, process.memory_info().rss)
            selected = select_weight_decay(candidate_metrics)
            representative = _row_record(candidates.iloc[0])
            selection_records.append(
                {
                    "job_key": str(logical_key),
                    "selected_candidate_job_key": selected["job_key"],
                    "selected_weight_decay": selected["weight_decay"],
                    "selected_checkpoint_step": selected["best_step"],
                    "selected_checkpoint_path": selected["checkpoint_path"],
                    "selected_checkpoint_sha256": selected["checkpoint_sha256"],
                    "validation_r2": selected["validation_r2"],
                    "validation_r2_per_target": selected[
                        "validation_r2_per_target"
                    ],
                    "training_seed": SELECTION_SEED,
                    "input_transform_hash": selected["input_transform_hash"],
                    "subset_hash": representative["subset_hash"],
                    "subset_file_sha256": representative["subset_file_sha256"],
                    "model_definition_hash": representative[
                        "model_definition_hash"
                    ],
                    "identity": {
                        name: representative[name]
                        for name in (
                            "job_family",
                            "branch",
                            "encoder_seed",
                            "readout",
                            "target_block",
                            "transform",
                            "spectral_arm",
                            "budget_label",
                            "subsample_seed",
                            "width",
                        )
                    },
                }
            )
            if logical_index == 1 or logical_index % 10 == 0:
                progress = {
                    "stage": "selection",
                    "logical_cells_completed": len(selection_records),
                    "logical_cells_total": int(summary["logical_selection_cells"]),
                    "candidate_models_completed_or_verified": logical_index * 3,
                    "candidate_models_total": int(len(inventory)),
                    "runtime_seconds": float(time.perf_counter() - started),
                    "peak_system_ram_bytes": int(peak_rss),
                    "test_accessed": False,
                }
                atomic_write_json(destination / "selection_progress.json", progress)
                print(json.dumps(progress, sort_keys=True), flush=True)
        except BaseException as exc:
            failures.append(
                {
                    "job_key": str(logical_key),
                    "stage": "selection",
                    "exception": repr(exc),
                    "last_completed_step": None,
                    "validation_state": "failed_before_selection_freeze",
                    "gpu_memory_bytes": None,
                    "system_rss_bytes": process.memory_info().rss,
                    "scientifically_required": True,
                }
            )
            break
    failures_path = destination / "selection_failures.parquet"
    atomic_write_parquet(pd.DataFrame(failures, columns=FAILURE_COLUMNS), failures_path)
    if failures or len(selection_records) != int(summary["logical_selection_cells"]):
        raise ExperimentIntegrityError(
            "selection incomplete; manifest was not frozen and test remains blocked"
        )
    digest = freeze_selection_manifest(
        selection_records,
        manifest_path,
        selection_inventory_sha256=summary["selection_inventory_sha256"],
    )
    result = {
        "status": "complete",
        "models_trained_or_verified": int(len(inventory)),
        "logical_cells": len(selection_records),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_system_ram_bytes": int(peak_rss),
        "selection_manifest_sha256": digest,
        "failures": 0,
    }
    atomic_write_json(destination / "selection_compute_log.json", result)
    return result


def _evaluation_fingerprint(
    job: Mapping[str, Any],
    transform: FrozenFeatureTransform,
    selection_record: Mapping[str, Any],
    selection_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "phase3_evaluation_cell.v1",
        "job": dict(job),
        "selected_weight_decay": selection_record["selected_weight_decay"],
        "transform_hash": transform.transform_hash,
        "selection_manifest_sha256": selection_manifest_sha256,
        "phase1_results_sha256": PHASE1_RESULTS_SHA256,
        "phase1_subset_manifest_sha256": PHASE1_SUBSET_MANIFEST_SHA256,
        "phase2_manifest_sha256": PHASE2_MANIFEST_SHA256,
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
        "training_config": asdict(Phase3MLPConfig(width=int(job["width"]))),
        "mixed_precision": False,
    }


def _recover_or_block_test_inference(
    claim_path: Path, metrics_path: Path, expected_job_key: str
) -> pd.DataFrame | None:
    if not claim_path.exists():
        return None
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if claim.get("job_key") != expected_job_key:
        raise ExperimentIntegrityError("test inference claim job-key mismatch")
    if not metrics_path.is_file():
        raise ExperimentIntegrityError(
            "test inference may have occurred but no atomic metrics artifact exists; "
            "refusing to evaluate test a second time"
        )
    frame = pd.read_parquet(metrics_path)
    if (
        frame.empty
        or set(frame["job_key"].astype(str)) != {expected_job_key}
        or not np.isfinite(frame["test_r2"].to_numpy(dtype=float)).all()
    ):
        raise ExperimentIntegrityError("existing test metrics are incomplete or invalid")
    observed = sha256_file(metrics_path)
    if claim.get("status") == "complete":
        if claim.get("metrics_sha256") != observed:
            raise ExperimentIntegrityError("completed test inference metrics hash mismatch")
    elif claim.get("status") == "started":
        atomic_write_json(
            claim_path,
            {
                **claim,
                "status": "complete",
                "metrics_sha256": observed,
                "recovered_after_atomic_metrics_write": True,
            },
        )
    else:
        raise ExperimentIntegrityError("invalid test inference claim state")
    return frame


def _run_evaluation_cell(
    bundle: InputBundle,
    phase1_dir: Path,
    out_dir: Path,
    caches: _ExecutionCaches,
    job: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    *,
    selection_manifest_path: Path,
    selection_manifest_sha256: str,
    device: str,
) -> pd.DataFrame:
    job_dir = out_dir / "evaluation_jobs" / str(job["job_key"])
    state_path = job_dir / "complete.json"
    feature = bundle.feature_set(
        str(job["branch"]), int(job["encoder_seed"]), str(job["readout"])
    )
    transform = caches.transform_for(
        feature, str(job["transform"]), str(job["spectral_arm"])
    )
    fingerprint = _evaluation_fingerprint(
        job, transform, selection_record, selection_manifest_sha256
    )
    if state_path.is_file():
        verify_completed_cell(state_path, fingerprint)
        return pd.read_parquet(job_dir / "test_metrics.parquet")
    job_dir.mkdir(parents=True, exist_ok=True)
    positions = caches.positions_for(str(job["subset_path"]))
    scaler, target_indices = caches.scaler_for(
        str(job["subset_path"]), str(job["target_block"])
    )
    config = Phase3MLPConfig(width=int(job["width"]))
    trained = train_validation_only_mlp(
        bundle.load_features(feature, "train"),
        caches.target_train,
        positions,
        bundle.load_features(feature, "validation"),
        caches.target_validation,
        transform,
        target_indices,
        scaler,
        reader_seed=int(job["reader_seed"]),
        weight_decay=float(selection_record["selected_weight_decay"]),
        device=device,
        config=config,
        primary_width=int(job["width"]) == 256,
    )
    checkpoint_path = job_dir / "checkpoint.pt"
    checkpoint_sha = atomic_torch_save(trained["best_state"], checkpoint_path)
    history_path = job_dir / "history.parquet"
    atomic_write_parquet(pd.DataFrame(trained["history"]), history_path)
    claim_path = job_dir / "test_inference_claim.json"
    metrics_path = job_dir / "test_metrics.parquet"
    existing = _recover_or_block_test_inference(
        claim_path, metrics_path, str(job["job_key"])
    )
    if existing is not None:
        _write_completed_state(
            job_dir,
            fingerprint,
            (metrics_path, checkpoint_path, history_path, claim_path),
        )
        return existing
    assert_test_access_allowed(selection_manifest_path)
    atomic_write_json(
        claim_path,
        {
            "schema_name": "thesis.experiment01.phase3.test_inference_claim",
            "schema_version": 1,
            "status": "started",
            "job_key": str(job["job_key"]),
            "selection_manifest_sha256": selection_manifest_sha256,
            "test_evaluation_limit": 1,
        },
    )
    # These are deliberately loaded only after the frozen-manifest assertion
    # and the one-shot inference claim have been serialized.
    feature_test = bundle.load_features(feature, "test")
    target_test = bundle.load_targets("test")
    test_r2 = evaluate_mlp_r2(
        trained["model"],
        feature_test,
        target_test,
        transform,
        target_indices,
        scaler,
        split="test",
        device=device,
        chunk_rows=config.evaluation_chunk_rows,
        selection_manifest_path=selection_manifest_path,
    )
    target_names = [bundle.target_names[index] for index in target_indices]
    validation_r2 = np.asarray(trained["validation_r2_per_target"], dtype=float)
    rows = []
    for target_name, val_r2, tst_r2 in zip(target_names, validation_r2, test_r2):
        rows.append(
            {
                "branch": job["branch"],
                "encoder_seed": int(job["encoder_seed"]),
                "readout": job["readout"],
                "target_block": job["target_block"],
                "target_name": target_name,
                "transform": job["transform"],
                "spectral_arm": job["spectral_arm"],
                "budget": job["budget_label"],
                "budget_label": job["budget_label"],
                "n_stock_days": int(job["n_stock_days"]),
                "n_rows": int(job["n_rows"]),
                "subsample_seed": int(job["subsample_seed"]),
                "reader_seed": int(job["reader_seed"]),
                "weight_decay": float(selection_record["selected_weight_decay"]),
                "best_step": int(trained["best_step"]),
                "validation_r2": float(val_r2),
                "test_r2": float(tst_r2),
                "full_budget_ceiling": np.nan,
                "ceiling_eligible": False,
                "normalized_recovery": np.nan,
                "subset_hash": job["subset_hash"],
                "transform_hash": transform.transform_hash,
                "selection_manifest_hash": selection_manifest_sha256,
                "selection_manifest_sha256": selection_manifest_sha256,
                "job_key": job["job_key"],
                "logical_job_key": job["logical_job_key"],
                "job_family": job["job_family"],
                "width": int(job["width"]),
                "checkpoint_sha256": checkpoint_sha,
                "runtime_seconds": float(trained["runtime_seconds"]),
                "peak_gpu_memory_bytes": int(trained["peak_gpu_memory_bytes"]),
            }
        )
    frame = pd.DataFrame(rows)
    atomic_write_parquet(frame, metrics_path)
    atomic_write_json(
        claim_path,
        {
            "schema_name": "thesis.experiment01.phase3.test_inference_claim",
            "schema_version": 1,
            "status": "complete",
            "job_key": str(job["job_key"]),
            "selection_manifest_sha256": selection_manifest_sha256,
            "test_evaluation_limit": 1,
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    _write_completed_state(
        job_dir,
        fingerprint,
        (metrics_path, checkpoint_path, history_path, claim_path),
    )
    return frame


def run_phase3_evaluation(
    bundle_dir: str | Path,
    phase1_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Retrain selected configurations and perform one-shot fixed-test inference."""

    from .schema import load_input_bundle

    initialize_compute_device(device)
    destination = Path(out_dir)
    _phase3_prerequisites(destination, historical=True)
    selection_path = destination / "selection_manifest.json"
    selection_hash = assert_test_access_allowed(selection_path)
    selection_payload = json.loads(selection_path.read_text())
    selected = {
        str(record["job_key"]): record for record in selection_payload["records"]
    }
    inventory_path = destination / "evaluation_job_inventory.parquet"
    inventory_summary = json.loads(
        (destination / "job_inventory_summary.json").read_text()
    )
    if sha256_file(inventory_path) != inventory_summary["evaluation_inventory_sha256"]:
        raise ExperimentIntegrityError("evaluation inventory hash mismatch")
    inventory = pd.read_parquet(inventory_path)
    if set(inventory["logical_job_key"].astype(str)) != set(selected):
        raise ExperimentIntegrityError("selection/evaluation logical inventory mismatch")
    bundle = load_input_bundle(bundle_dir, verify_hashes=False, check_finite=False)
    subset_payload = _load_subset_manifest(Path(phase1_dir))
    subset_records = {
        str(record["path"]): record for record in subset_payload["subsets"]
    }
    caches = _ExecutionCaches(bundle, Path(phase1_dir), subset_records)
    frames = []
    failures = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    order = [
        "branch",
        "encoder_seed",
        "transform",
        "spectral_arm",
        "target_block",
        "budget_stock_day_equivalents",
        "subsample_seed",
        "width",
        "reader_seed",
    ]
    ordered_inventory = inventory.sort_values(order, kind="stable")
    for model_index, (_, row) in enumerate(ordered_inventory.iterrows(), start=1):
        job = _row_record(row)
        try:
            frames.append(
                _run_evaluation_cell(
                    bundle,
                    Path(phase1_dir),
                    destination,
                    caches,
                    job,
                    selected[str(job["logical_job_key"])],
                    selection_manifest_path=selection_path,
                    selection_manifest_sha256=selection_hash,
                    device=device,
                )
            )
            peak_rss = max(peak_rss, process.memory_info().rss)
            if model_index == 1 or model_index % 10 == 0:
                progress = {
                    "stage": "evaluation",
                    "models_completed_or_verified": len(frames),
                    "models_total": int(len(inventory)),
                    "runtime_seconds": float(time.perf_counter() - started),
                    "peak_system_ram_bytes": int(peak_rss),
                    "selection_manifest_sha256": selection_hash,
                }
                atomic_write_json(destination / "evaluation_progress.json", progress)
                print(json.dumps(progress, sort_keys=True), flush=True)
        except BaseException as exc:
            failures.append(
                {
                    "job_key": job["job_key"],
                    "stage": "evaluation",
                    "exception": repr(exc),
                    "last_completed_step": None,
                    "validation_state": "failed_before_or_during_one_shot_test",
                    "gpu_memory_bytes": None,
                    "system_rss_bytes": process.memory_info().rss,
                    "scientifically_required": True,
                }
            )
    failures_path = destination / "failures.parquet"
    atomic_write_parquet(pd.DataFrame(failures, columns=FAILURE_COLUMNS), failures_path)
    if failures:
        raise ExperimentIntegrityError(
            f"{len(failures)} Phase-III evaluation cells failed; results not finalized"
        )
    raw = pd.concat(frames, ignore_index=True)
    expected_rows = sum(
        {"directional": 12, "volatility": 2, "timing": 1}[block]
        for block in inventory["target_block"]
    )
    if len(raw) != expected_rows:
        raise ExperimentIntegrityError("Phase-III raw target-row inventory mismatch")
    finalized = add_targetwise_normalized_recovery(raw)
    results_path = destination / "phase3_results.parquet"
    atomic_write_parquet(finalized, results_path)
    result = {
        "status": "complete",
        "evaluation_models": int(len(inventory)),
        "target_rows": int(len(finalized)),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_system_ram_bytes": int(peak_rss),
        "selection_manifest_sha256": selection_hash,
        "results_sha256": sha256_file(results_path),
        "failures": 0,
    }
    atomic_write_json(destination / "evaluation_compute_log.json", result)
    return result


def run_phase3_benchmark(
    bundle_dir: str | Path,
    phase1_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Benchmark the minimum 1,000-step window without reading production test."""

    from .schema import load_input_bundle

    device_record = initialize_compute_device(device)
    destination = Path(out_dir)
    _phase3_prerequisites(destination, historical=True)
    inventory = pd.read_parquet(destination / "selection_job_inventory.parquet")
    bundle = load_input_bundle(bundle_dir, verify_hashes=False, check_finite=False)
    subset_payload = _load_subset_manifest(Path(phase1_dir))
    subset_records = {
        str(record["path"]): record for record in subset_payload["subsets"]
    }
    caches = _ExecutionCaches(bundle, Path(phase1_dir), subset_records)
    records = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    for transform_kind in ("native", "full_whitened"):
        matches = inventory.loc[
            (inventory["job_family"] == "primary_directional")
            & (inventory["branch"] == "jepa_horizon")
            & (inventory["encoder_seed"] == 0)
            & (inventory["transform"] == transform_kind)
            & (inventory["budget_label"] == "b_1_4")
            & (inventory["subsample_seed"] == 0)
            & (inventory["weight_decay"] == 0.0)
        ]
        if len(matches) != 1:
            raise ExperimentIntegrityError("benchmark inventory cell is not unique")
        job = _row_record(matches.iloc[0])
        feature = bundle.feature_set("jepa_horizon", 0, PRIMARY_READOUT)
        transform = caches.transform_for(feature, transform_kind, "none")
        positions = caches.positions_for(str(job["subset_path"]))
        scaler, target_indices = caches.scaler_for(
            str(job["subset_path"]), "directional"
        )
        config = Phase3MLPConfig(
            max_steps=1_000,
            min_steps=1_000,
            validation_interval=500,
            patience_evaluations=6,
        )
        trained = train_validation_only_mlp(
            bundle.load_features(feature, "train"),
            caches.target_train,
            positions,
            bundle.load_features(feature, "validation"),
            caches.target_validation,
            transform,
            target_indices,
            scaler,
            reader_seed=SELECTION_SEED,
            weight_decay=0.0,
            device=device,
            config=config,
            enforce_preregistered_schedule=False,
        )
        peak_rss = max(peak_rss, process.memory_info().rss)
        records.append(
            {
                "transform": transform_kind,
                "input_dimension": transform.output_dimension,
                "n_rows": len(positions),
                "steps": trained["last_step"],
                "validation_evaluations": len(trained["history"]),
                "runtime_seconds": trained["runtime_seconds"],
                "seconds_per_1000_steps_with_two_full_validations": trained[
                    "runtime_seconds"
                ],
                "best_validation_r2": trained["validation_r2"],
                "peak_gpu_memory_bytes": trained["peak_gpu_memory_bytes"],
            }
        )
    inventory_summary = json.loads(
        (destination / "job_inventory_summary.json").read_text()
    )
    mean_runtime = float(np.mean([value["runtime_seconds"] for value in records]))
    total_models = int(inventory_summary["total_models"])
    lower_seconds = total_models * mean_runtime
    maximum_seconds = total_models * mean_runtime * 20.0
    result = {
        "schema_name": "thesis.experiment01.phase3.compute_benchmark",
        "schema_version": 1,
        "status": "complete",
        "device": device_record,
        "benchmark_cells": records,
        "production_models": total_models,
        "estimate_assumptions": {
            "lower_bound": "every model stops at the minimum 1000 steps",
            "upper_bound": "every model reaches 20000 steps with proportional validation cost",
            "test_inference_not_included": True,
            "historical_gate_or_feature_projection_not_included": True,
        },
        "estimated_compute_seconds_minimum_steps": lower_seconds,
        "estimated_compute_hours_minimum_steps": lower_seconds / 3600.0,
        "estimated_compute_seconds_all_max_steps": maximum_seconds,
        "estimated_compute_hours_all_max_steps": maximum_seconds / 3600.0,
        "peak_system_ram_bytes": int(peak_rss),
        "test_accessed": False,
    }
    atomic_write_json(destination / "compute_benchmark.json", result)
    return result

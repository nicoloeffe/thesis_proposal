"""Validation-only convergence gate for the frozen F16 cohort candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiment01.constants import ALPHA_GRID
from experiment01.f16 import (
    CAP_CANDIDATES,
    F16IntegrityError,
    _atomic_write_parquet,
    _parquet_record,
    _relative,
    cohort_for_cap,
)
from experiment01.io import atomic_write_json, canonical_json_sha256, sha256_file
from experiment01.linear import (
    SufficientStats,
    eigensystem,
    evaluate_stats,
    fit_alpha,
    select_targets,
    transformed_design,
    tune_alpha,
)
from experiment01.sharded import ArrayShard, ShardedArray


AUDIT_ARMS = ("jepa_horizon", "supervised")
AUDIT_SEEDS = (0, 1, 2)
AUDIT_READOUTS = ("last_concat512", "meanK_concatS")
TOLERANCES = {
    "directional_full_rank_validation_r2": ("absolute", 0.020),
    "directional_top8_predictive_mass": ("absolute", 0.020),
    "directional_top16_predictive_mass": ("absolute", 0.020),
    "common_full_role_retention": ("absolute", 0.030),
    "contrast_full_role_retention": ("absolute", 0.030),
    "directional_last_to_meanK_gap": ("absolute", 0.020),
    "covariance_trace": ("relative", 0.050),
    "cumulative_explained_variance_8": ("absolute", 0.020),
    "cumulative_explained_variance_16": ("absolute", 0.020),
    "cumulative_explained_variance_32": ("absolute", 0.020),
    "normalized_leading16_eigenvalue_profile": ("absolute", 0.030),
}


@dataclass
class FeatureMoments:
    n: int
    x_sum: np.ndarray
    xtx: np.ndarray

    @classmethod
    def zeros(cls, dimension: int) -> "FeatureMoments":
        return cls(
            n=0,
            x_sum=np.zeros(dimension, dtype=np.float64),
            xtx=np.zeros((dimension, dimension), dtype=np.float64),
        )

    def add_rows(self, x: np.ndarray) -> None:
        value = np.asarray(x, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != self.x_sum.shape[0]:
            raise ValueError("feature chunk has incompatible shape")
        if not np.isfinite(value).all():
            raise F16IntegrityError("non-finite feature in cohort convergence")
        self.n += len(value)
        self.x_sum += value.sum(axis=0, dtype=np.float64)
        self.xtx += value.T @ value

    def add(self, other: "FeatureMoments") -> None:
        if self.x_sum.shape != other.x_sum.shape:
            raise ValueError("feature moments have incompatible dimensions")
        self.n += other.n
        self.x_sum += other.x_sum
        self.xtx += other.xtx

    def copy(self) -> "FeatureMoments":
        return FeatureMoments(self.n, self.x_sum.copy(), self.xtx.copy())

    @property
    def covariance(self) -> np.ndarray:
        if self.n <= 0:
            raise ValueError("feature moments are empty")
        mean = self.x_sum / self.n
        covariance = self.xtx / self.n - np.outer(mean, mean)
        return (covariance + covariance.T) * 0.5


def _stats_from_npz(data: Mapping[str, np.ndarray], prefix: str) -> SufficientStats:
    return SufficientStats(
        n=int(np.asarray(data[f"{prefix}_n"]).item()),
        x_sum=np.asarray(data[f"{prefix}_x_sum"], dtype=np.float64),
        y_sum=np.asarray(data[f"{prefix}_y_sum"], dtype=np.float64),
        xtx=np.asarray(data[f"{prefix}_xtx"], dtype=np.float64),
        xty=np.asarray(data[f"{prefix}_xty"], dtype=np.float64),
        yty=np.asarray(data[f"{prefix}_yty"], dtype=np.float64),
    )


def pc_stats_to_raw(
    stats: SufficientStats,
    train_mean: np.ndarray,
    eigenvectors: np.ndarray,
) -> SufficientStats:
    """Invert the exact train-centered PCA coordinate transformation."""
    mean = np.asarray(train_mean, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if vectors.shape != (stats.dimension, stats.dimension) or mean.shape != (
        stats.dimension,
    ):
        raise ValueError("PCA transform has incompatible shape")
    centered_sum = vectors @ stats.x_sum
    centered_xtx = vectors @ stats.xtx @ vectors.T
    centered_xtx = (centered_xtx + centered_xtx.T) * 0.5
    x_sum = stats.n * mean + centered_sum
    xtx = (
        centered_xtx
        + np.outer(mean, centered_sum)
        + np.outer(centered_sum, mean)
        + stats.n * np.outer(mean, mean)
    )
    xty = vectors @ stats.xty + np.outer(mean, stats.y_sum)
    return SufficientStats(
        n=stats.n,
        x_sum=x_sum,
        y_sum=stats.y_sum.copy(),
        xtx=(xtx + xtx.T) * 0.5,
        xty=xty,
        yty=stats.yty.copy(),
    )


def projection_stats(stats: SufficientStats, projection: np.ndarray) -> SufficientStats:
    matrix = np.asarray(projection, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != stats.dimension:
        raise ValueError("projection has incompatible shape")
    return SufficientStats(
        n=stats.n,
        x_sum=matrix.T @ stats.x_sum,
        y_sum=stats.y_sum.copy(),
        xtx=matrix.T @ stats.xtx @ matrix,
        xty=matrix.T @ stats.xty,
        yty=stats.yty.copy(),
    )


def role_projections(role_dim: int = 128) -> tuple[np.ndarray, np.ndarray]:
    hadamard = np.asarray(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=np.float64,
    ) / 2.0
    identity = np.eye(role_dim, dtype=np.float64)
    common = np.vstack([hadamard[0, role] * identity for role in range(4)])
    contrast = np.zeros((4 * role_dim, 3 * role_dim), dtype=np.float64)
    for role in range(4):
        for contrast_index in range(3):
            start_row = role * role_dim
            start_column = contrast_index * role_dim
            contrast[
                start_row : start_row + role_dim,
                start_column : start_column + role_dim,
            ] = hadamard[contrast_index + 1, role] * identity
    np.testing.assert_allclose(common.T @ common, identity, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        contrast.T @ contrast,
        np.eye(3 * role_dim),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(common.T @ contrast, 0.0, rtol=0.0, atol=1e-14)
    return common, contrast


def _array_from_record(bundle_root: Path, record: Mapping[str, Any]) -> ShardedArray:
    if record.get("storage") != "sharded_npy_v1":
        raise F16IntegrityError("F16 convergence requires the sharded production bundle")
    shards = [
        ArrayShard(
            bundle_root / str(shard["path"]),
            int(shard["row_start"]),
            int(shard["row_stop"]),
        )
        for shard in record["shards"]
    ]
    return ShardedArray(shards, record["shape"], record["dtype"])


def _feature_array(
    bundle_root: Path,
    bundle_manifest: Mapping[str, Any],
    arm: str,
    seed: int,
    readout: str,
    split: str,
) -> ShardedArray:
    matches = [
        record
        for record in bundle_manifest["feature_sets"]
        if record["branch"] == arm
        and int(record["encoder_seed"]) == seed
        and record["readout"] == readout
    ]
    if len(matches) != 1:
        raise F16IntegrityError(f"feature record is not unique: {arm}/{seed}/{readout}")
    return _array_from_record(bundle_root, matches[0]["arrays"][split])


def _target_array(
    bundle_root: Path, bundle_manifest: Mapping[str, Any], split: str
) -> ShardedArray:
    if split == "test":
        raise F16IntegrityError("test target access is forbidden during convergence")
    return _array_from_record(bundle_root, bundle_manifest["targets"]["arrays"][split])


def _bucket_positions(cohort: pd.DataFrame) -> list[np.ndarray]:
    edges = (0, *CAP_CANDIDATES)
    values: list[np.ndarray] = []
    rank = cohort["cohort_rank"].to_numpy(dtype=np.int64)
    positions = cohort["source_row_position"].to_numpy(dtype=np.int64)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = np.sort(positions[(rank >= lower) & (rank < upper)])
        values.append(selected)
    return values


def _feature_moments_by_cap(
    x: ShardedArray,
    cohort: pd.DataFrame,
    chunk_rows: int,
) -> dict[int, FeatureMoments]:
    buckets: list[FeatureMoments] = []
    for positions in _bucket_positions(cohort):
        moments = FeatureMoments.zeros(x.shape[1])
        for start in range(0, len(positions), chunk_rows):
            moments.add_rows(x[positions[start : start + chunk_rows]])
        buckets.append(moments)
    result: dict[int, FeatureMoments] = {}
    cumulative = FeatureMoments.zeros(x.shape[1])
    for cap, bucket in zip(CAP_CANDIDATES, buckets):
        cumulative.add(bucket)
        result[cap] = cumulative.copy()
    return result


def _validation_stats_by_cap(
    x: ShardedArray,
    y: ShardedArray,
    cohort: pd.DataFrame,
    chunk_rows: int,
) -> dict[int, SufficientStats]:
    buckets: list[SufficientStats] = []
    for positions in _bucket_positions(cohort):
        stats = SufficientStats.zeros(x.shape[1], y.shape[1])
        for start in range(0, len(positions), chunk_rows):
            indices = positions[start : start + chunk_rows]
            stats.add_rows(x[indices], y[indices])
        buckets.append(stats)
    result: dict[int, SufficientStats] = {}
    cumulative = SufficientStats.zeros(x.shape[1], y.shape[1])
    for cap, bucket in zip(CAP_CANDIDATES, buckets):
        cumulative.add(bucket)
        result[cap] = cumulative.copy()
    return result


def _directional_indices(bundle_manifest: Mapping[str, Any]) -> np.ndarray:
    indices = [
        index
        for index, target in enumerate(bundle_manifest["targets"]["definitions"])
        if target["block"] == "directional" and bool(target["independent"])
    ]
    if len(indices) != 12:
        raise F16IntegrityError(
            f"expected 12 independent directional targets, found {len(indices)}"
        )
    return np.asarray(indices, dtype=np.int64)


def _aggregate_r2(model, stats: SufficientStats, target_indices: np.ndarray) -> float:
    scores = evaluate_stats(model, stats)
    if not scores.valid[target_indices].all():
        raise F16IntegrityError("constant independent directional target")
    return float(np.mean(scores.values[target_indices]))


def _fit_reference_reader(
    train_stats: SufficientStats,
    validation_stats: SufficientStats,
    target_indices: np.ndarray,
):
    design = transformed_design(train_stats)
    tuned = tune_alpha(
        design,
        None,
        None,
        target_indices,
        alpha_grid=ALPHA_GRID,
        validation_stats=validation_stats,
    )
    return fit_alpha(design, tuned.alpha), tuned


def _predictive_mass(
    train_stats: SufficientStats,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    target_indices: np.ndarray,
    k: int,
    tolerance: float,
) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if k > len(values) or np.any(values[:k] <= tolerance):
        raise F16IntegrityError(f"top-{k} covariance directions are numerically invalid")
    target_variance = train_stats.target_centered_ss / train_stats.n
    if np.any(target_variance[target_indices] <= 0.0):
        raise F16IntegrityError("non-positive target variance in predictive mass")
    projected = vectors[:, :k].T @ train_stats.cross[:, target_indices]
    mass = np.square(projected) / (
        values[:k, None] * target_variance[target_indices][None, :]
    )
    if not np.isfinite(mass).all():
        raise F16IntegrityError("non-finite predictive mass")
    return float(np.mean(mass.sum(axis=0)))


def _spectrum_metrics(values: np.ndarray) -> dict[str, float]:
    eigenvalues = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    trace = float(eigenvalues.sum())
    if not np.isfinite(trace) or trace <= 0.0:
        raise F16IntegrityError("invalid covariance trace")
    result = {"covariance_trace": trace}
    for k in (8, 16, 32):
        result[f"cumulative_explained_variance_{k}"] = float(
            eigenvalues[:k].sum() / trace
        )
    return result


def _metric_row(
    *,
    arm: str,
    seed: int,
    readout: str,
    cap: int,
    metric: str,
    reference: float,
    candidate: float,
) -> dict[str, Any]:
    kind, tolerance = TOLERANCES[metric]
    absolute_error = abs(candidate - reference)
    if kind == "relative":
        if reference == 0.0:
            relative_error = float("inf")
        else:
            relative_error = absolute_error / abs(reference)
        passed = bool(np.isfinite(relative_error) and relative_error <= tolerance)
        comparison_error = relative_error
    else:
        relative_error = absolute_error / abs(reference) if reference != 0.0 else np.nan
        passed = bool(np.isfinite(absolute_error) and absolute_error <= tolerance)
        comparison_error = absolute_error
    return {
        "arm": arm,
        "encoder_seed": seed,
        "readout": readout,
        "cap_per_stock_day": cap,
        "metric": metric,
        "reference_value": reference,
        "candidate_value": candidate,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "tolerance_kind": kind,
        "tolerance": tolerance,
        "comparison_error": comparison_error,
        "passed": passed,
    }


def _load_reference_stats(
    phase1_dir: Path,
    phase2_dir: Path,
    arm: str,
    seed: int,
    readout: str,
) -> tuple[SufficientStats, SufficientStats, np.ndarray, np.ndarray, float, dict[str, Any]]:
    tag = f"{arm}_seed{seed}_{readout}"
    cache_path = phase2_dir / "cache" / f"{tag}.npz"
    cache_metadata_path = phase2_dir / "cache" / f"{tag}.json"
    if not cache_path.is_file() or not cache_metadata_path.is_file():
        raise F16IntegrityError(f"missing Phase-II cache: {tag}")
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    if cache_metadata.get("sha256") != sha256_file(cache_path):
        raise F16IntegrityError(f"Phase-II cache hash mismatch: {tag}")
    transform_path = phase1_dir / "transforms" / f"{tag}.npz"
    expected_transform_sha = cache_metadata["source_fingerprint"]["transform_sha256"]
    if sha256_file(transform_path) != expected_transform_sha:
        raise F16IntegrityError(f"Phase-I transform hash mismatch: {tag}")
    with np.load(transform_path, allow_pickle=False) as transform:
        mean = np.asarray(transform["unlabelled_train_mean"], dtype=np.float64)
        eigenvalues = np.asarray(transform["covariance_eigenvalues"], dtype=np.float64)
        eigenvectors = np.asarray(transform["covariance_eigenvectors"], dtype=np.float64)
        tolerance = float(np.asarray(transform["numerical_tolerance"]).item())
    with np.load(cache_path, allow_pickle=False) as cache:
        train_pc = _stats_from_npz(cache, "budget_b_16")
        validation_pc = _stats_from_npz(cache, "validation")
    return (
        pc_stats_to_raw(train_pc, mean, eigenvectors),
        pc_stats_to_raw(validation_pc, mean, eigenvectors),
        eigenvalues,
        eigenvectors,
        tolerance,
        {
            "cache_path": cache_path,
            "cache_sha256": sha256_file(cache_path),
            "transform_path": transform_path,
            "transform_sha256": expected_transform_sha,
        },
    )


def _feature_metrics(
    train_stats: SufficientStats,
    full_validation: SufficientStats,
    candidate_validation: Mapping[int, SufficientStats],
    reference_eigenvalues: np.ndarray,
    reference_eigenvectors: np.ndarray,
    reference_tolerance: float,
    candidate_moments: Mapping[int, FeatureMoments],
    target_indices: np.ndarray,
) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[str, Any]]:
    full_model, tuned_full = _fit_reference_reader(
        train_stats, full_validation, target_indices
    )
    common_projection, contrast_projection = role_projections()
    common_train = projection_stats(train_stats, common_projection)
    common_validation = projection_stats(full_validation, common_projection)
    contrast_train = projection_stats(train_stats, contrast_projection)
    contrast_validation = projection_stats(full_validation, contrast_projection)
    common_model, tuned_common = _fit_reference_reader(
        common_train, common_validation, target_indices
    )
    contrast_model, tuned_contrast = _fit_reference_reader(
        contrast_train, contrast_validation, target_indices
    )
    reference_full_r2 = _aggregate_r2(full_model, full_validation, target_indices)
    reference_common_r2 = _aggregate_r2(
        common_model, common_validation, target_indices
    )
    reference_contrast_r2 = _aggregate_r2(
        contrast_model, contrast_validation, target_indices
    )
    if abs(reference_full_r2) < 1e-12:
        raise F16IntegrityError("reference full-rank directional R2 is zero")
    reference_spectrum = _spectrum_metrics(reference_eigenvalues)
    reference_profile = np.maximum(reference_eigenvalues[:16], 0.0) / reference_spectrum[
        "covariance_trace"
    ]
    reference = {
        "directional_full_rank_validation_r2": reference_full_r2,
        "common_full_role_retention": reference_common_r2 / reference_full_r2,
        "contrast_full_role_retention": reference_contrast_r2 / reference_full_r2,
        "directional_top8_predictive_mass": _predictive_mass(
            train_stats,
            reference_eigenvalues,
            reference_eigenvectors,
            target_indices,
            8,
            reference_tolerance,
        ),
        "directional_top16_predictive_mass": _predictive_mass(
            train_stats,
            reference_eigenvalues,
            reference_eigenvectors,
            target_indices,
            16,
            reference_tolerance,
        ),
        **reference_spectrum,
        "normalized_leading16_eigenvalue_profile": 0.0,
    }
    candidates: dict[int, dict[str, float]] = {}
    for cap in CAP_CANDIDATES:
        validation = candidate_validation[cap]
        full_r2 = _aggregate_r2(full_model, validation, target_indices)
        common_r2 = _aggregate_r2(
            common_model,
            projection_stats(validation, common_projection),
            target_indices,
        )
        contrast_r2 = _aggregate_r2(
            contrast_model,
            projection_stats(validation, contrast_projection),
            target_indices,
        )
        covariance = candidate_moments[cap].covariance
        spectrum = eigensystem(covariance, candidate_moments[cap].n)
        spectrum_metrics = _spectrum_metrics(spectrum.eigenvalues)
        profile = np.maximum(spectrum.eigenvalues[:16], 0.0) / spectrum_metrics[
            "covariance_trace"
        ]
        candidates[cap] = {
            "directional_full_rank_validation_r2": full_r2,
            "common_full_role_retention": common_r2 / full_r2,
            "contrast_full_role_retention": contrast_r2 / full_r2,
            "directional_top8_predictive_mass": _predictive_mass(
                train_stats,
                spectrum.eigenvalues,
                spectrum.eigenvectors,
                target_indices,
                8,
                spectrum.diagnostics.numerical_tolerance,
            ),
            "directional_top16_predictive_mass": _predictive_mass(
                train_stats,
                spectrum.eigenvalues,
                spectrum.eigenvectors,
                target_indices,
                16,
                spectrum.diagnostics.numerical_tolerance,
            ),
            **spectrum_metrics,
            "normalized_leading16_eigenvalue_profile": float(
                np.abs(profile - reference_profile).sum()
            ),
        }
    selection = {
        "full_alpha": tuned_full.alpha,
        "common_alpha": tuned_common.alpha,
        "contrast_alpha": tuned_contrast.alpha,
    }
    return reference, candidates, selection


def run_convergence_gate(
    repo_root: Path,
    output_root: Path,
    bundle_root: Path,
    phase1_dir: Path,
    phase2_dir: Path,
    *,
    chunk_rows: int = 32_768,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    bundle_root = bundle_root.resolve()
    phase1_dir = phase1_dir.resolve()
    phase2_dir = phase2_dir.resolve()
    candidate_manifest_path = output_root / "f16_cohort_candidates_manifest.json"
    protocol_manifest_path = output_root / "f16_manifest.json"
    for path in (candidate_manifest_path, protocol_manifest_path):
        if not path.is_file():
            raise F16IntegrityError(f"missing frozen F16 manifest: {path}")
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    protocol_manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("status") != "frozen_pending_convergence":
        raise F16IntegrityError("candidate cohort manifest is not frozen")
    if protocol_manifest.get("test_barrier") != "locked":
        raise F16IntegrityError("test barrier is not locked")
    if candidate_manifest["test_barrier"] != {
        "status": "locked",
        "test_row_metadata_accessed": True,
        "test_targets_accessed": False,
        "test_features_accessed": False,
        "test_statistics_accessed": False,
    }:
        raise F16IntegrityError("candidate manifest test barrier is invalid")

    bundle_manifest_path = bundle_root / "manifest.json"
    if sha256_file(bundle_manifest_path) != candidate_manifest["inputs"][
        "bundle_manifest_sha256"
    ]:
        raise F16IntegrityError("production bundle manifest drift")
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    target_indices = _directional_indices(bundle_manifest)

    train_cohort_path = repo_root / candidate_manifest["cohorts"]["train"]["path"]
    validation_cohort_path = (
        repo_root / candidate_manifest["cohorts"]["validation"]["path"]
    )
    if sha256_file(train_cohort_path) != candidate_manifest["cohorts"]["train"][
        "sha256"
    ]:
        raise F16IntegrityError("train cohort candidate drift")
    if sha256_file(validation_cohort_path) != candidate_manifest["cohorts"][
        "validation"
    ]["sha256"]:
        raise F16IntegrityError("validation cohort candidate drift")
    train_cohort = pd.read_parquet(train_cohort_path)
    validation_cohort = pd.read_parquet(validation_cohort_path)
    validation_targets = _target_array(bundle_root, bundle_manifest, "validation")

    rows: list[dict[str, Any]] = []
    feature_outputs: dict[tuple[str, int, str], tuple[dict[str, float], dict[int, dict[str, float]]]] = {}
    source_records: list[dict[str, Any]] = []
    alpha_selections: list[dict[str, Any]] = []
    for arm in AUDIT_ARMS:
        for seed in AUDIT_SEEDS:
            for readout in AUDIT_READOUTS:
                x_train = _feature_array(
                    bundle_root, bundle_manifest, arm, seed, readout, "train"
                )
                x_validation = _feature_array(
                    bundle_root, bundle_manifest, arm, seed, readout, "validation"
                )
                candidate_moments = _feature_moments_by_cap(
                    x_train, train_cohort, chunk_rows
                )
                candidate_validation = _validation_stats_by_cap(
                    x_validation, validation_targets, validation_cohort, chunk_rows
                )
                (
                    train_stats,
                    full_validation,
                    reference_values,
                    reference_vectors,
                    reference_tolerance,
                    source_record,
                ) = _load_reference_stats(
                    phase1_dir, phase2_dir, arm, seed, readout
                )
                reference, candidates, alphas = _feature_metrics(
                    train_stats,
                    full_validation,
                    candidate_validation,
                    reference_values,
                    reference_vectors,
                    reference_tolerance,
                    candidate_moments,
                    target_indices,
                )
                feature_outputs[(arm, seed, readout)] = (reference, candidates)
                source_records.append(
                    {
                        "arm": arm,
                        "encoder_seed": seed,
                        "readout": readout,
                        "phase2_cache_path": _relative(source_record["cache_path"], repo_root),
                        "phase2_cache_sha256": source_record["cache_sha256"],
                        "phase1_transform_path": _relative(
                            source_record["transform_path"], repo_root
                        ),
                        "phase1_transform_sha256": source_record["transform_sha256"],
                    }
                )
                alpha_selections.append(
                    {
                        "arm": arm,
                        "encoder_seed": seed,
                        "readout": readout,
                        **alphas,
                    }
                )
                for cap in CAP_CANDIDATES:
                    for metric in (
                        "directional_full_rank_validation_r2",
                        "directional_top8_predictive_mass",
                        "directional_top16_predictive_mass",
                        "common_full_role_retention",
                        "contrast_full_role_retention",
                        "covariance_trace",
                        "cumulative_explained_variance_8",
                        "cumulative_explained_variance_16",
                        "cumulative_explained_variance_32",
                        "normalized_leading16_eigenvalue_profile",
                    ):
                        rows.append(
                            _metric_row(
                                arm=arm,
                                seed=seed,
                                readout=readout,
                                cap=cap,
                                metric=metric,
                                reference=reference[metric],
                                candidate=candidates[cap][metric],
                            )
                        )

    for arm in AUDIT_ARMS:
        for seed in AUDIT_SEEDS:
            last_reference, last_candidates = feature_outputs[
                (arm, seed, "last_concat512")
            ]
            mean_reference, mean_candidates = feature_outputs[
                (arm, seed, "meanK_concatS")
            ]
            reference_gap = (
                last_reference["directional_full_rank_validation_r2"]
                - mean_reference["directional_full_rank_validation_r2"]
            )
            for cap in CAP_CANDIDATES:
                candidate_gap = (
                    last_candidates[cap]["directional_full_rank_validation_r2"]
                    - mean_candidates[cap]["directional_full_rank_validation_r2"]
                )
                rows.append(
                    _metric_row(
                        arm=arm,
                        seed=seed,
                        readout="last_minus_meanK",
                        cap=cap,
                        metric="directional_last_to_meanK_gap",
                        reference=reference_gap,
                        candidate=candidate_gap,
                    )
                )

    results = pd.DataFrame(rows)
    expected_rows = (len(AUDIT_ARMS) * len(AUDIT_SEEDS) * len(AUDIT_READOUTS) * 10 + len(AUDIT_ARMS) * len(AUDIT_SEEDS)) * len(CAP_CANDIDATES)
    if len(results) != expected_rows:
        raise F16IntegrityError(
            f"convergence grid has {len(results)} rows, expected {expected_rows}"
        )
    pass_by_cap = results.groupby("cap_per_stock_day", sort=True)["passed"].all()
    passing_caps = [int(cap) for cap, passed in pass_by_cap.items() if bool(passed)]
    if not passing_caps:
        selected_cap = None
        status = "failed_no_candidate_passed"
    else:
        selected_cap = min(passing_caps)
        status = "passed"

    convergence_path = output_root / "f16_cohort_convergence.parquet"
    _atomic_write_parquet(results, convergence_path)
    decision: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_cohort_convergence",
        "schema_version": 1,
        "status": status,
        "candidate_manifest_path": _relative(candidate_manifest_path, repo_root),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "bundle_manifest_sha256": sha256_file(bundle_manifest_path),
        "phase1_metadata_sha256": sha256_file(phase1_dir / "metadata.json"),
        "phase2_metadata_sha256": sha256_file(phase2_dir / "metadata.json"),
        "candidate_caps_per_stock_day": list(CAP_CANDIDATES),
        "selection_rule": "smallest cap for which every preregistered row passes",
        "selected_cap_per_stock_day": selected_cap,
        "pass_by_cap": {str(cap): bool(value) for cap, value in pass_by_cap.items()},
        "rows_by_cap": {
            str(cap): int((results["cap_per_stock_day"] == cap).sum())
            for cap in CAP_CANDIDATES
        },
        "failed_rows_by_cap": {
            str(cap): int(
                (
                    (results["cap_per_stock_day"] == cap)
                    & (~results["passed"].astype(bool))
                ).sum()
            )
            for cap in CAP_CANDIDATES
        },
        "tolerances": {
            metric: {"kind": kind, "value": value}
            for metric, (kind, value) in TOLERANCES.items()
        },
        "audit_arms": list(AUDIT_ARMS),
        "audit_seeds": list(AUDIT_SEEDS),
        "audit_readouts": list(AUDIT_READOUTS),
        "independent_directional_target_indices": target_indices.tolist(),
        "alpha_selections": alpha_selections,
        "source_records": source_records,
        "convergence_table": {
            "path": _relative(convergence_path, repo_root),
            "sha256": sha256_file(convergence_path),
            "size_bytes": convergence_path.stat().st_size,
            "rows": len(results),
        },
        "test_barrier": {
            "status": "locked",
            "test_targets_accessed": False,
            "test_features_accessed": False,
            "test_statistics_accessed": False,
        },
        "failures": [] if selected_cap is not None else ["no_candidate_cap_passed"],
    }
    decision["manifest_fingerprint"] = canonical_json_sha256(decision)
    decision_path = output_root / "f16_cohort_decision.json"
    atomic_write_json(decision_path, decision)
    if selected_cap is None:
        raise F16IntegrityError(
            "no preregistered cohort cap passed; F16 training is blocked"
        )

    selected_records: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        source_path = repo_root / candidate_manifest["cohorts"][split]["path"]
        source = pd.read_parquet(source_path)
        selected = cohort_for_cap(source, selected_cap)
        path = output_root / "cohorts" / "selected" / f"{split}.parquet"
        _atomic_write_parquet(selected, path)
        record = _parquet_record(path, selected, repo_root)
        record.update(
            {
                "split": split,
                "cap_per_stock_day": selected_cap,
                "outcome_arrays_accessed_during_selection": False,
            }
        )
        selected_records[split] = record
    selected_manifest: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_cohort",
        "schema_version": 1,
        "status": "selected_and_frozen",
        "selected_cap_per_stock_day": selected_cap,
        "candidate_manifest_path": _relative(candidate_manifest_path, repo_root),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "decision_path": _relative(decision_path, repo_root),
        "decision_sha256": sha256_file(decision_path),
        "label_budgets": candidate_manifest["label_budgets"],
        "cohorts": selected_records,
        "test_barrier": decision["test_barrier"],
        "failures": [],
    }
    selected_manifest["manifest_fingerprint"] = canonical_json_sha256(selected_manifest)
    atomic_write_json(output_root / "f16_cohort_manifest.json", selected_manifest)
    return results, decision

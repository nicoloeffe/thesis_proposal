"""Experiment 01 Phase-II preregistered spectral diagnostics.

The implementation is deliberately isolated from Phase I.  It consumes the
frozen production bundle, the frozen Phase-I covariance transforms and the
already serialized Phase-I subset manifests.  Feature arrays are scanned once
per exact feature set and all subsequent PCA, band and random-subspace fits are
performed from cached sufficient statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from .constants import BRANCHES, READOUTS
from .errors import ExperimentIntegrityError
from .io import (
    atomic_savez,
    atomic_write_json,
    atomic_write_parquet,
    sha256_file,
)
from .linear import (
    SufficientStats,
    evaluate_stats,
    fit_alpha,
    sufficient_stats,
    transformed_design,
    tune_alpha,
)
from .schema import FeatureSet, InputBundle, iter_target_indices


PHASE2_VERSION = "1.0"
PHASE2_K_BASE = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PHASE2_SELECTED_BUDGETS = (
    ("b_1_8", 0.125, 0),
    ("b_1_4", 0.25, 0),
    ("b_4", 4.0, 0),
    ("b_16", 16.0, 0),
    ("full_train", None, -1),
)
PHASE2_BANDS = (
    ("1:8", 1, 8),
    ("9:16", 9, 16),
    ("17:32", 17, 32),
    ("33:64", 33, 64),
    ("65:128", 65, 128),
    ("129:256", 129, 256),
    ("257:D_valid", 257, None),
)

PHASE2_RESULT_COLUMNS = (
    "phase2_version",
    "commit_hash",
    "branch",
    "encoder_seed",
    "readout",
    "target_block",
    "target_name",
    "target_independent",
    "budget_label",
    "budget_days_per_stock",
    "subsample_seed",
    "n_rows",
    "subspace_order",
    "subspace_dimension",
    "valid_dimension",
    "dimension_fraction",
    "reader_family",
    "alpha",
    "lambda_absolute",
    "alpha_selected_on_validation",
    "train_r2",
    "validation_r2",
    "test_r2",
    "trace_cov",
    "trace_cov_over_dim",
    "numerical_rank",
    "numerical_tolerance",
    "fit_status",
    "failure_reason",
)


@dataclass(frozen=True)
class Phase2Config:
    random_draws: int = 100
    chunk_rows: int = 65536
    bundle_hashes_verified_this_run: bool = False
    branches: tuple[str, ...] = BRANCHES
    readouts: tuple[str, ...] = READOUTS
    target_blocks: tuple[str, ...] = (
        "directional",
        "volatility",
        "timing",
    )

    def validate(self) -> None:
        if self.random_draws != 100:
            raise ValueError("Phase II requires exactly 100 Haar draws")
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if not isinstance(self.bundle_hashes_verified_this_run, bool):
            raise ValueError("bundle_hashes_verified_this_run must be boolean")
        if set(self.branches) != set(BRANCHES):
            raise ValueError("Phase II requires all canonical branches")
        if set(self.readouts) != set(READOUTS):
            raise ValueError("Phase II requires both canonical readouts")
        if set(self.target_blocks) != {"directional", "volatility", "timing"}:
            raise ValueError("Phase II requires all three target blocks")


@dataclass
class FeatureStatistics:
    eigenvalues: np.ndarray
    numerical_rank: int
    numerical_tolerance: float
    budgets: Mapping[str, SufficientStats]
    validation: SufficientStats
    test: SufficientStats


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


def feature_tag(feature: FeatureSet) -> str:
    return f"{feature.branch}_seed{feature.encoder_seed}_{feature.readout}"


def phase2_schedule(valid_dimension: int) -> tuple[int, ...]:
    rank = int(valid_dimension)
    if rank <= 0:
        raise ValueError("valid_dimension must be positive")
    return tuple(sorted({value for value in PHASE2_K_BASE if value <= rank} | {rank}))


def spectral_bands(valid_dimension: int) -> tuple[tuple[str, int, int], ...]:
    rank = int(valid_dimension)
    rows = []
    for label, one_based_start, one_based_stop in PHASE2_BANDS:
        stop = rank if one_based_stop is None else min(one_based_stop, rank)
        if one_based_start <= stop:
            rows.append((label, one_based_start - 1, stop))
    if not rows or rows[-1][2] != rank:
        raise ExperimentIntegrityError("spectral bands do not cover D_valid")
    covered = np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for _, start, stop in rows]
    )
    if not np.array_equal(covered, np.arange(rank, dtype=np.int64)):
        raise ExperimentIntegrityError("spectral bands overlap or have gaps")
    return tuple(rows)


def deterministic_haar_basis(
    valid_dimension: int,
    subspace_dimension: int,
    *,
    branch_index: int,
    encoder_seed: int,
    readout_index: int,
    draw: int,
) -> np.ndarray:
    """Return one deterministic Haar-distributed partial orthonormal basis."""
    rank = int(valid_dimension)
    m = int(subspace_dimension)
    if not 0 < m <= rank:
        raise ValueError("random subspace dimension is outside D_valid")
    if m == rank:
        return np.eye(rank, dtype=np.float64)
    seed = np.random.SeedSequence(
        [20260731, branch_index, int(encoder_seed), readout_index, m, int(draw)]
    )
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((rank, m)), mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    result = q * signs[None, :]
    np.testing.assert_allclose(
        result.T @ result,
        np.eye(m),
        rtol=2e-12,
        atol=2e-12,
    )
    return result


def _selected_subset_records(phase1_dir: Path) -> Mapping[str, Mapping[str, object]]:
    manifest_path = phase1_dir / "subset_manifest.json"
    payload = json.loads(manifest_path.read_text())
    records: dict[str, Mapping[str, object]] = {}
    for label, days, seed in PHASE2_SELECTED_BUDGETS:
        matches = [
            row
            for row in payload["subsets"]
            if row["budget_label"] == label
            and int(row["subsample_seed"]) == seed
            and (
                days is None
                or float(row["budget_days_per_stock"]) == float(days)
            )
        ]
        if len(matches) != 1:
            raise ExperimentIntegrityError(
                f"Phase-II selected subset {label}/seed{seed} is not unique"
            )
        record = matches[0]
        path = phase1_dir / str(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ExperimentIntegrityError(
                f"Phase-II selected subset {label}/seed{seed} failed its hash"
            )
        records[label] = {**record, "absolute_path": path}
    return records


def _positions_by_budget(
    records: Mapping[str, Mapping[str, object]], n_train: int
) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for label, record in records.items():
        if label == "full_train":
            continue
        frame = pd.read_parquet(
            record["absolute_path"], columns=["source_row_position"]
        )
        positions = frame["source_row_position"].to_numpy(dtype=np.int64)
        if len(positions) != int(record["n_rows"]):
            raise ExperimentIntegrityError(f"subset {label} row count differs")
        if (
            len(positions)
            and (
                positions[0] < 0
                or positions[-1] >= n_train
                or np.any(positions[1:] <= positions[:-1])
            )
        ):
            raise ExperimentIntegrityError(f"subset {label} positions are invalid")
        result[label] = positions
    return result


def _to_train_centered_pc_stats(
    stats: SufficientStats, train_mean: np.ndarray, eigenvectors: np.ndarray
) -> SufficientStats:
    mean = np.asarray(train_mean, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    centered_sum = stats.x_sum - stats.n * mean
    centered_xtx = (
        stats.xtx
        - np.outer(mean, stats.x_sum)
        - np.outer(stats.x_sum, mean)
        + stats.n * np.outer(mean, mean)
    )
    centered_xtx = (centered_xtx + centered_xtx.T) * 0.5
    centered_xty = stats.xty - np.outer(mean, stats.y_sum)
    x_sum = vectors.T @ centered_sum
    xtx = vectors.T @ centered_xtx @ vectors
    xtx = (xtx + xtx.T) * 0.5
    xty = vectors.T @ centered_xty
    return SufficientStats(
        n=stats.n,
        x_sum=x_sum,
        y_sum=stats.y_sum.copy(),
        xtx=xtx,
        xty=xty,
        yty=stats.yty.copy(),
    )


def _stream_feature_training_statistics(
    x_train,
    y_train,
    *,
    positions: Mapping[str, np.ndarray],
    train_mean: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    chunk_rows: int,
) -> tuple[SufficientStats, Mapping[str, SufficientStats], Mapping[str, int]]:
    dimension = int(x_train.shape[1])
    n_targets = int(y_train.shape[1])
    partial = {
        label: SufficientStats.zeros(dimension, n_targets)
        for label in positions
    }
    x_sum = np.zeros(dimension, dtype=np.float64)
    y_sum = np.zeros(n_targets, dtype=np.float64)
    xty = np.zeros((dimension, n_targets), dtype=np.float64)
    yty = np.zeros(n_targets, dtype=np.float64)
    updates = {label: 0 for label in positions}
    n_rows = len(x_train)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        x = np.asarray(x_train[start:stop], dtype=np.float64)
        y = np.asarray(y_train[start:stop], dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ExperimentIntegrityError("non-finite train feature/target chunk")
        x_sum += x.sum(axis=0, dtype=np.float64)
        y_sum += y.sum(axis=0, dtype=np.float64)
        xty += x.T @ y
        yty += np.einsum("nt,nt->t", y, y)
        for label, selected in positions.items():
            left = int(np.searchsorted(selected, start, side="left"))
            right = int(np.searchsorted(selected, stop, side="left"))
            if right > left:
                local = selected[left:right] - start
                partial[label].add_rows(x[local], y[local])
                updates[label] += 1
    observed_mean = x_sum / n_rows
    np.testing.assert_allclose(
        observed_mean,
        train_mean,
        rtol=2e-9,
        atol=2e-9,
        err_msg="Phase-II train mean differs from frozen Phase-I covariance fit",
    )
    y_mean = y_sum / n_rows
    centered_cross = xty / n_rows - np.outer(train_mean, y_mean)
    cross_pc = eigenvectors.T @ centered_cross
    full_pc = SufficientStats(
        n=n_rows,
        x_sum=np.zeros(dimension, dtype=np.float64),
        y_sum=y_sum,
        xtx=np.diag(eigenvalues * n_rows),
        xty=cross_pc * n_rows,
        yty=yty,
    )
    partial_pc = {
        label: _to_train_centered_pc_stats(value, train_mean, eigenvectors)
        for label, value in partial.items()
    }
    for label, selected in positions.items():
        if partial_pc[label].n != len(selected):
            raise ExperimentIntegrityError(f"subset {label} was not fully accumulated")
    return full_pc, partial_pc, updates


def _stats_arrays(prefix: str, stats: SufficientStats) -> Mapping[str, np.ndarray]:
    return {
        f"{prefix}_n": np.asarray(stats.n, dtype=np.int64),
        f"{prefix}_x_sum": stats.x_sum,
        f"{prefix}_y_sum": stats.y_sum,
        f"{prefix}_xtx": stats.xtx,
        f"{prefix}_xty": stats.xty,
        f"{prefix}_yty": stats.yty,
    }


def _stats_from_npz(data, prefix: str) -> SufficientStats:
    return SufficientStats(
        n=int(np.asarray(data[f"{prefix}_n"]).item()),
        x_sum=np.asarray(data[f"{prefix}_x_sum"], dtype=np.float64),
        y_sum=np.asarray(data[f"{prefix}_y_sum"], dtype=np.float64),
        xtx=np.asarray(data[f"{prefix}_xtx"], dtype=np.float64),
        xty=np.asarray(data[f"{prefix}_xty"], dtype=np.float64),
        yty=np.asarray(data[f"{prefix}_yty"], dtype=np.float64),
    )


def _feature_transform_record(
    phase1_metadata: Mapping[str, object], feature: FeatureSet
) -> Mapping[str, object]:
    matches = [
        record
        for record in phase1_metadata["transforms"]
        if record["branch"] == feature.branch
        and int(record["encoder_seed"]) == feature.encoder_seed
        and record["readout"] == feature.readout
    ]
    if len(matches) != 1:
        raise ExperimentIntegrityError(
            f"frozen Phase-I transform is not unique for {feature.key}"
        )
    return matches[0]


def prepare_feature_cache(
    bundle: InputBundle,
    feature: FeatureSet,
    *,
    phase1_dir: Path,
    phase1_metadata: Mapping[str, object],
    cache_dir: Path,
    subset_records: Mapping[str, Mapping[str, object]],
    chunk_rows: int,
) -> tuple[FeatureStatistics, Mapping[str, object]]:
    tag = feature_tag(feature)
    cache_path = cache_dir / f"{tag}.npz"
    metadata_path = cache_dir / f"{tag}.json"
    transform_record = _feature_transform_record(phase1_metadata, feature)
    transform_path = phase1_dir / str(transform_record["path"])
    if sha256_file(transform_path) != transform_record["sha256"]:
        raise ExperimentIntegrityError(f"frozen Phase-I transform hash failed: {tag}")
    source_fingerprint = {
        "cache_algorithm": "train_centered_pc_sufficient_statistics.v1",
        "phase2_version": PHASE2_VERSION,
        "bundle_manifest_sha256": sha256_file(bundle.root / "manifest.json"),
        "phase1_metadata_sha256": sha256_file(phase1_dir / "metadata.json"),
        "transform_sha256": transform_record["sha256"],
        "subset_sha256": {
            label: record["sha256"] for label, record in subset_records.items()
        },
        "chunk_rows": chunk_rows,
    }
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("source_fingerprint") != source_fingerprint:
            raise ExperimentIntegrityError(f"stale Phase-II cache metadata: {tag}")
        if metadata.get("sha256") != sha256_file(cache_path):
            raise ExperimentIntegrityError(f"Phase-II cache hash failed: {tag}")
        return load_feature_cache(cache_path), metadata

    cache_dir.mkdir(parents=True, exist_ok=True)
    with np.load(transform_path, allow_pickle=False) as transform:
        train_mean = np.asarray(transform["unlabelled_train_mean"], dtype=np.float64)
        eigenvalues = np.asarray(transform["covariance_eigenvalues"], dtype=np.float64)
        eigenvectors = np.asarray(
            transform["covariance_eigenvectors"], dtype=np.float64
        )
        tolerance = float(np.asarray(transform["numerical_tolerance"]).item())
        rank = int(np.asarray(transform["numerical_rank"]).item())
    if (
        train_mean.shape != (feature.dimension,)
        or eigenvalues.shape != (feature.dimension,)
        or eigenvectors.shape != (feature.dimension, feature.dimension)
    ):
        raise ExperimentIntegrityError(f"invalid Phase-I transform shape: {tag}")
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ExperimentIntegrityError(f"non-finite Phase-I transform: {tag}")
    if not np.all(eigenvalues[:rank] > tolerance):
        raise ExperimentIntegrityError(f"Phase-I numerical rank is inconsistent: {tag}")
    if np.any(eigenvalues[rank:] > tolerance):
        raise ExperimentIntegrityError(f"Phase-I invalid tail exceeds tolerance: {tag}")

    x_train = bundle.load_features(feature, "train")
    y_train = bundle.load_targets("train")
    positions = _positions_by_budget(subset_records, len(x_train))
    full, partial, updates = _stream_feature_training_statistics(
        x_train,
        y_train,
        positions=positions,
        train_mean=train_mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        chunk_rows=chunk_rows,
    )
    budgets: dict[str, SufficientStats] = {**partial, "full_train": full}
    for split in ("validation", "test"):
        raw = sufficient_stats(
            bundle.load_features(feature, split),
            bundle.load_targets(split),
            chunk_rows=chunk_rows,
        )
        value = _to_train_centered_pc_stats(raw, train_mean, eigenvectors)
        if split == "validation":
            validation = value
        else:
            test = value
    arrays: dict[str, np.ndarray] = {
        "eigenvalues": eigenvalues,
        "numerical_rank": np.asarray(rank, dtype=np.int64),
        "numerical_tolerance": np.asarray(tolerance, dtype=np.float64),
        "budget_labels": np.asarray(list(budgets), dtype="U32"),
    }
    for label, value in budgets.items():
        arrays.update(_stats_arrays(f"budget_{label}", value))
    arrays.update(_stats_arrays("validation", validation))
    arrays.update(_stats_arrays("test", test))
    atomic_savez(cache_path, **arrays)
    metadata = {
        "schema_name": "thesis.experiment01.phase2_feature_cache",
        "schema_version": 1,
        "feature": {
            "branch": feature.branch,
            "encoder_seed": feature.encoder_seed,
            "readout": feature.readout,
        },
        "source_fingerprint": source_fingerprint,
        "sha256": sha256_file(cache_path),
        "size_bytes": cache_path.stat().st_size,
        "valid_dimension": rank,
        "numerical_tolerance": tolerance,
        "budget_rows": {label: value.n for label, value in budgets.items()},
        "streaming_subset_updates": updates,
    }
    atomic_write_json(metadata_path, metadata)
    return load_feature_cache(cache_path), metadata


def load_feature_cache(path: str | Path) -> FeatureStatistics:
    with np.load(path, allow_pickle=False) as data:
        labels = [str(value) for value in data["budget_labels"].tolist()]
        budgets = {
            label: _stats_from_npz(data, f"budget_{label}") for label in labels
        }
        return FeatureStatistics(
            eigenvalues=np.asarray(data["eigenvalues"], dtype=np.float64),
            numerical_rank=int(np.asarray(data["numerical_rank"]).item()),
            numerical_tolerance=float(
                np.asarray(data["numerical_tolerance"]).item()
            ),
            budgets=budgets,
            validation=_stats_from_npz(data, "validation"),
            test=_stats_from_npz(data, "test"),
        )


def select_coordinate_stats(
    stats: SufficientStats,
    coordinates: Sequence[int],
    target_indices: Sequence[int] | None = None,
) -> SufficientStats:
    coord = np.asarray(coordinates, dtype=np.int64)
    targets = (
        np.arange(stats.n_targets, dtype=np.int64)
        if target_indices is None
        else np.asarray(target_indices, dtype=np.int64)
    )
    return SufficientStats(
        n=stats.n,
        x_sum=stats.x_sum[coord],
        y_sum=stats.y_sum[targets],
        xtx=stats.xtx[np.ix_(coord, coord)],
        xty=stats.xty[np.ix_(coord, targets)],
        yty=stats.yty[targets],
    )


def project_stats(
    stats: SufficientStats,
    basis: np.ndarray,
    target_indices: Sequence[int] | None = None,
) -> SufficientStats:
    q = np.asarray(basis, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != stats.dimension:
        raise ValueError("basis has incompatible shape")
    targets = (
        np.arange(stats.n_targets, dtype=np.int64)
        if target_indices is None
        else np.asarray(target_indices, dtype=np.int64)
    )
    xtx = q.T @ stats.xtx @ q
    return SufficientStats(
        n=stats.n,
        x_sum=q.T @ stats.x_sum,
        y_sum=stats.y_sum[targets],
        xtx=(xtx + xtx.T) * 0.5,
        xty=q.T @ stats.xty[:, targets],
        yty=stats.yty[targets],
    )


def predictive_mass_table(
    bundle: InputBundle,
    feature: FeatureSet,
    statistics: FeatureStatistics,
) -> pd.DataFrame:
    full = statistics.budgets["full_train"]
    eigenvalues = statistics.eigenvalues
    rank = statistics.numerical_rank
    tolerance = statistics.numerical_tolerance
    cross = full.cross
    target_variance = full.target_centered_ss / full.n
    target_scale = np.maximum(full.yty / full.n, 1.0)
    target_tolerance = np.finfo(np.float64).eps * target_scale
    target_valid = target_variance > target_tolerance
    direction_valid = np.arange(len(eigenvalues)) < rank
    direction_valid &= eigenvalues > tolerance
    mass = np.full((len(eigenvalues), full.n_targets), np.nan, dtype=np.float64)
    valid_rows = np.flatnonzero(direction_valid)
    valid_targets = np.flatnonzero(target_valid)
    mass[np.ix_(valid_rows, valid_targets)] = (
        np.square(cross[np.ix_(valid_rows, valid_targets)])
        / eigenvalues[valid_rows, None]
        / target_variance[valid_targets][None, :]
    )
    if np.any(mass[np.isfinite(mass)] < -1e-14):
        raise ExperimentIntegrityError("predictive mass became negative")
    cumulative = np.nancumsum(np.where(np.isfinite(mass), mass, 0.0), axis=0)
    total = cumulative[rank - 1]
    variance_total = float(eigenvalues[:rank].sum())
    schedule = set(phase2_schedule(rank))
    definitions = bundle.target_definitions
    rows = []
    for direction in range(len(eigenvalues)):
        valid = bool(direction_valid[direction])
        for target_index, target in enumerate(definitions):
            target_ok = bool(target_valid[target_index])
            denominator = total[target_index]
            rows.append(
                {
                    "branch": feature.branch,
                    "encoder_seed": feature.encoder_seed,
                    "readout": feature.readout,
                    "target_block": target.block,
                    "target_name": target.name,
                    "target_independent": target.independent,
                    "direction_index": direction + 1,
                    "eigenvalue": float(eigenvalues[direction]),
                    "numerical_tolerance": tolerance,
                    "valid_dimension": rank,
                    "direction_valid": valid,
                    "target_variance": float(target_variance[target_index]),
                    "target_valid": target_ok,
                    "predictive_mass": (
                        float(mass[direction, target_index])
                        if valid and target_ok
                        else np.nan
                    ),
                    "cumulative_predictive_mass": (
                        float(cumulative[direction, target_index])
                        if target_ok
                        else np.nan
                    ),
                    "total_predictive_mass_valid": (
                        float(denominator) if target_ok else np.nan
                    ),
                    "cumulative_mass_fraction": (
                        float(cumulative[direction, target_index] / denominator)
                        if target_ok and denominator > 0.0
                        else np.nan
                    ),
                    "variance_fraction": (
                        float(eigenvalues[direction] / variance_total)
                        if valid and variance_total > 0.0
                        else np.nan
                    ),
                    "cumulative_variance_fraction": (
                        float(eigenvalues[: direction + 1].sum() / variance_total)
                        if direction < rank and variance_total > 0.0
                        else np.nan
                    ),
                    "curve_schedule_point": (direction + 1) in schedule,
                    "fit_status": "ok" if valid and target_ok else "invalid",
                    "failure_reason": (
                        ""
                        if valid and target_ok
                        else (
                            "numerically_invalid_covariance_direction"
                            if not valid
                            else "constant_or_numerically_invalid_target"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _reader_rows(
    bundle: InputBundle,
    feature: FeatureSet,
    statistics: FeatureStatistics,
    *,
    budget_label: str,
    budget_days: float | None,
    subsample_seed: int,
    order: str,
    coordinates: np.ndarray,
    commit: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rank = statistics.numerical_rank
    train_all = statistics.budgets[budget_label]
    for block in ("directional", "volatility", "timing"):
        global_indices, independent_global = iter_target_indices(
            bundle.target_definitions, block
        )
        local_independent = np.flatnonzero(
            np.isin(global_indices, independent_global)
        ).astype(np.int64)
        train = select_coordinate_stats(train_all, coordinates, global_indices)
        validation = select_coordinate_stats(
            statistics.validation, coordinates, global_indices
        )
        test = select_coordinate_stats(statistics.test, coordinates, global_indices)
        design = transformed_design(train)
        tuned = tune_alpha(
            design,
            None,
            None,
            local_independent,
            validation_stats=validation,
        )
        candidates = (
            ("min_norm_ols_diagnostic", fit_alpha(design, 0.0), False),
            (
                "ridge_trace_normalized_tuned",
                fit_alpha(design, tuned.alpha),
                True,
            ),
        )
        for reader_family, model, selected in candidates:
            train_scores = evaluate_stats(model, train)
            validation_scores = evaluate_stats(model, validation)
            test_scores = evaluate_stats(model, test)
            diagnostics = design.eigensystem.diagnostics
            for local, global_index in enumerate(global_indices):
                reasons = []
                for split, scores in (
                    ("train", train_scores),
                    ("validation", validation_scores),
                    ("test", test_scores),
                ):
                    if not scores.valid[local]:
                        reasons.append(f"{split}:{scores.reasons[local]}")
                target = bundle.target_definitions[int(global_index)]
                rows.append(
                    {
                        "phase2_version": PHASE2_VERSION,
                        "commit_hash": commit,
                        "branch": feature.branch,
                        "encoder_seed": feature.encoder_seed,
                        "readout": feature.readout,
                        "target_block": block,
                        "target_name": target.name,
                        "target_independent": target.independent,
                        "budget_label": budget_label,
                        "budget_days_per_stock": (
                            np.nan if budget_days is None else budget_days
                        ),
                        "subsample_seed": subsample_seed,
                        "n_rows": train.n,
                        "subspace_order": order,
                        "subspace_dimension": len(coordinates),
                        "valid_dimension": rank,
                        "dimension_fraction": len(coordinates) / rank,
                        "reader_family": reader_family,
                        "alpha": model.alpha,
                        "lambda_absolute": model.lambda_absolute,
                        "alpha_selected_on_validation": selected,
                        "train_r2": float(train_scores.values[local]),
                        "validation_r2": float(validation_scores.values[local]),
                        "test_r2": float(test_scores.values[local]),
                        "trace_cov": diagnostics.trace_cov,
                        "trace_cov_over_dim": diagnostics.trace_cov_over_dim,
                        "numerical_rank": diagnostics.numerical_rank,
                        "numerical_tolerance": diagnostics.numerical_tolerance,
                        "fit_status": "ok" if not reasons else "invalid",
                        "failure_reason": ";".join(reasons),
                    }
                )
    return rows


def phase2_ladder_table(
    bundle: InputBundle,
    feature: FeatureSet,
    statistics: FeatureStatistics,
    *,
    commit: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rank = statistics.numerical_rank
    schedule = phase2_schedule(rank)
    budget_map = {label: (days, seed) for label, days, seed in PHASE2_SELECTED_BUDGETS}
    for budget_label, (days, seed) in budget_map.items():
        for m in schedule:
            top = np.arange(m, dtype=np.int64)
            bottom = np.arange(rank - m, rank, dtype=np.int64)
            rows.extend(
                _reader_rows(
                    bundle,
                    feature,
                    statistics,
                    budget_label=budget_label,
                    budget_days=days,
                    subsample_seed=seed,
                    order="top_pca",
                    coordinates=top,
                    commit=commit,
                )
            )
            rows.extend(
                _reader_rows(
                    bundle,
                    feature,
                    statistics,
                    budget_label=budget_label,
                    budget_days=days,
                    subsample_seed=seed,
                    order="bottom_pca",
                    coordinates=bottom,
                    commit=commit,
                )
            )
    return pd.DataFrame(rows, columns=PHASE2_RESULT_COLUMNS)


def full_rank_phase1_parity_gate(
    bundle: InputBundle,
    feature: FeatureSet,
    statistics: FeatureStatistics,
    ladder: pd.DataFrame,
    phase1_full_min_norm: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> Mapping[str, object]:
    """Gate the Phase-II sufficient-statistic path against frozen Phase I."""
    observed = ladder[
        ladder["budget_label"].eq("full_train")
        & ladder["subspace_order"].eq("top_pca")
        & ladder["subspace_dimension"].eq(statistics.numerical_rank)
        & ladder["reader_family"].eq("min_norm_ols_diagnostic")
    ][
        [
            "branch",
            "encoder_seed",
            "readout",
            "target_block",
            "target_name",
            "train_r2",
            "validation_r2",
            "test_r2",
        ]
    ]
    expected = phase1_full_min_norm[
        phase1_full_min_norm["branch"].eq(feature.branch)
        & phase1_full_min_norm["encoder_seed"].eq(feature.encoder_seed)
        & phase1_full_min_norm["readout"].eq(feature.readout)
    ]
    keys = ["branch", "encoder_seed", "readout", "target_block", "target_name"]
    if len(observed) != len(bundle.target_definitions) or len(expected) != len(
        bundle.target_definitions
    ):
        raise ExperimentIntegrityError(
            f"Phase-I/II full-rank parity inventory differs for {feature_tag(feature)}"
        )
    merged = observed.merge(
        expected,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_phase2", "_phase1"),
    )
    if len(merged) != len(bundle.target_definitions):
        raise ExperimentIntegrityError(
            f"Phase-I/II full-rank parity keys differ for {feature_tag(feature)}"
        )
    split_columns = {
        "train": ("train_r2_phase2", "train_r2_phase1"),
        "validation": ("validation_r2", "val_r2"),
        "test": ("test_r2_phase2", "test_r2_phase1"),
    }
    differences = {}
    for split, (left, right) in split_columns.items():
        values = np.abs(
            merged[left].to_numpy(dtype=np.float64)
            - merged[right].to_numpy(dtype=np.float64)
        )
        if not np.isfinite(values).all():
            raise ExperimentIntegrityError(
                f"non-finite Phase-I/II parity difference for {feature_tag(feature)}"
            )
        differences[split] = float(values.max(initial=0.0))
    maximum = max(differences.values())
    if maximum > tolerance:
        raise ExperimentIntegrityError(
            f"Phase-I/II full-rank parity failed for {feature_tag(feature)}: "
            f"{maximum:.6g} > {tolerance:.6g}"
        )
    return {
        "passed": True,
        "feature": list(feature.key),
        "n_targets": len(merged),
        "tolerance": tolerance,
        "max_abs_difference": maximum,
        "max_abs_difference_by_split": differences,
    }


def _evaluate_random_min_norm(
    full_train: SufficientStats,
    test: SufficientStats,
    basis: np.ndarray,
) -> np.ndarray:
    train_projected = project_stats(full_train, basis)
    test_projected = project_stats(test, basis)
    design = transformed_design(train_projected)
    model = fit_alpha(design, 0.0)
    scores = evaluate_stats(model, test_projected)
    if not scores.valid.all():
        raise ExperimentIntegrityError("random subspace has invalid test target")
    return scores.values


def random_subspace_table(
    bundle: InputBundle,
    feature: FeatureSet,
    statistics: FeatureStatistics,
    ladder: pd.DataFrame,
    *,
    n_draws: int,
) -> pd.DataFrame:
    rank = statistics.numerical_rank
    bands = spectral_bands(rank)
    band_widths = {stop - start for _, start, stop in bands}
    ladder_dimensions = set(phase2_schedule(rank))
    dimensions = sorted(ladder_dimensions | band_widths)
    independent_by_block = {
        block: iter_target_indices(bundle.target_definitions, block)[1]
        for block in ("directional", "volatility", "timing")
    }
    top = ladder[
        ladder["budget_label"].eq("full_train")
        & ladder["subspace_order"].eq("top_pca")
        & ladder["reader_family"].eq("min_norm_ols_diagnostic")
        & ladder["target_independent"].eq(True)
    ]
    top_scores = (
        top.groupby(["subspace_dimension", "target_block"], observed=True)[
            "test_r2"
        ]
        .mean()
        .to_dict()
    )
    branch_index = BRANCHES.index(feature.branch)
    readout_index = READOUTS.index(feature.readout)
    eigenvalues = statistics.eigenvalues[:rank]
    variance_total = float(eigenvalues.sum())
    valid_coordinates = np.arange(rank, dtype=np.int64)
    full_train = select_coordinate_stats(
        statistics.budgets["full_train"], valid_coordinates
    )
    test = select_coordinate_stats(statistics.test, valid_coordinates)
    rows = []
    for m in dimensions:
        full_rank_scores = None
        if m == rank:
            full_rank_scores = _evaluate_random_min_norm(
                full_train,
                test,
                np.eye(rank, dtype=np.float64),
            )
        for draw in range(n_draws):
            if full_rank_scores is None:
                q = deterministic_haar_basis(
                    rank,
                    m,
                    branch_index=branch_index,
                    encoder_seed=feature.encoder_seed,
                    readout_index=readout_index,
                    draw=draw,
                )
                scores = _evaluate_random_min_norm(full_train, test, q)
                variance_fraction = float(
                    np.einsum("dm,d,dm->", q, eigenvalues, q)
                    / variance_total
                )
            else:
                scores = full_rank_scores
                variance_fraction = 1.0
            for block, target_indices in independent_by_block.items():
                values = scores[target_indices]
                rows.append(
                    {
                        "branch": feature.branch,
                        "encoder_seed": feature.encoder_seed,
                        "readout": feature.readout,
                        "target_block": block,
                        "subspace_dimension": m,
                        "valid_dimension": rank,
                        "dimension_fraction": m / rank,
                        "subspace_seed": draw,
                        "seed_algorithm": (
                            "SeedSequence(20260731,branch,encoder,readout,m,draw)"
                        ),
                        "reader_family": "min_norm_ols_diagnostic",
                        "test_r2_mean": float(np.mean(values)),
                        "test_r2_median": float(np.median(values)),
                        "n_independent_targets": len(values),
                        "variance_fraction": variance_fraction,
                        "ladder_dimension": m in ladder_dimensions,
                        "band_matched_dimension": m in band_widths,
                        "top_pca_test_r2_mean": top_scores.get((m, block), np.nan),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["top_pca_percentile"] = np.nan
    frame["empirical_p_random_exceeds_top"] = np.nan
    for _, index in frame.groupby(
        ["subspace_dimension", "target_block"], observed=True
    ).groups.items():
        group = frame.loc[index]
        top_score = float(group["top_pca_test_r2_mean"].iloc[0])
        if np.isfinite(top_score):
            random_values = group["test_r2_mean"].to_numpy(dtype=np.float64)
            tie_tolerance = (
                np.finfo(np.float64).eps
                * max(1.0, abs(top_score), float(np.max(np.abs(random_values))))
                * 256.0
            )
            frame.loc[index, "top_pca_percentile"] = 100.0 * float(
                np.mean(random_values <= top_score + tie_tolerance)
            )
            frame.loc[index, "empirical_p_random_exceeds_top"] = float(
                np.mean(random_values > top_score + tie_tolerance)
            )
    return frame


def _fit_block_aggregate(
    train: SufficientStats,
    validation: SufficientStats,
    test: SufficientStats,
    independent_local: np.ndarray,
    *,
    tuned: bool,
) -> tuple[float, float, float, float]:
    design = transformed_design(train)
    if tuned:
        choice = tune_alpha(
            design,
            None,
            None,
            independent_local,
            validation_stats=validation,
        )
        model = fit_alpha(design, choice.alpha)
    else:
        model = fit_alpha(design, 0.0)
    train_scores = evaluate_stats(model, train)
    validation_scores = evaluate_stats(model, validation)
    test_scores = evaluate_stats(model, test)
    if not (
        train_scores.valid[independent_local].all()
        and validation_scores.valid[independent_local].all()
        and test_scores.valid[independent_local].all()
    ):
        raise ExperimentIntegrityError("spectral band has invalid target score")
    return (
        model.alpha,
        float(np.mean(train_scores.values[independent_local])),
        float(np.mean(validation_scores.values[independent_local])),
        float(np.mean(test_scores.values[independent_local])),
    )


def spectral_band_table(
    bundle: InputBundle,
    feature: FeatureSet,
    statistics: FeatureStatistics,
    predictive_mass: pd.DataFrame,
    random_null: pd.DataFrame,
) -> pd.DataFrame:
    rank = statistics.numerical_rank
    eigenvalues = statistics.eigenvalues[:rank]
    variance_total = float(eigenvalues.sum())
    full_train = statistics.budgets["full_train"]
    rows = []
    for label, start, stop in spectral_bands(rank):
        band = np.arange(start, stop, dtype=np.int64)
        leave = np.concatenate(
            [
                np.arange(0, start, dtype=np.int64),
                np.arange(stop, rank, dtype=np.int64),
            ]
        )
        width = len(band)
        variance_fraction = float(eigenvalues[start:stop].sum() / variance_total)
        for block in ("directional", "volatility", "timing"):
            global_indices, independent_global = iter_target_indices(
                bundle.target_definitions, block
            )
            independent_local = np.flatnonzero(
                np.isin(global_indices, independent_global)
            ).astype(np.int64)
            mass_rows = predictive_mass[
                predictive_mass["target_block"].eq(block)
                & predictive_mass["target_independent"].eq(True)
                & predictive_mass["direction_index"].between(start + 1, stop)
                & predictive_mass["direction_valid"].eq(True)
            ]
            mass_by_target = mass_rows.groupby("target_name", observed=True).agg(
                band_mass=("predictive_mass", "sum"),
                total_mass=("total_predictive_mass_valid", "first"),
            )
            mass_fraction = mass_by_target["band_mass"] / mass_by_target["total_mass"]
            train_band = select_coordinate_stats(full_train, band, global_indices)
            validation_band = select_coordinate_stats(
                statistics.validation, band, global_indices
            )
            test_band = select_coordinate_stats(statistics.test, band, global_indices)
            train_leave = select_coordinate_stats(full_train, leave, global_indices)
            validation_leave = select_coordinate_stats(
                statistics.validation, leave, global_indices
            )
            test_leave = select_coordinate_stats(statistics.test, leave, global_indices)
            null = random_null[
                random_null["subspace_dimension"].eq(width)
                & random_null["target_block"].eq(block)
            ]["test_r2_mean"].to_numpy(dtype=np.float64)
            if len(null) != 100:
                raise ExperimentIntegrityError(
                    f"band {label}/{block} does not have 100 matched null draws"
                )
            for reader, tuned in (
                ("min_norm_ols_diagnostic", False),
                ("ridge_trace_normalized_tuned", True),
            ):
                alpha_band, train_r2_band, val_r2_band, test_r2_band = (
                    _fit_block_aggregate(
                        train_band,
                        validation_band,
                        test_band,
                        independent_local,
                        tuned=tuned,
                    )
                )
                alpha_leave, train_r2_leave, val_r2_leave, test_r2_leave = (
                    _fit_block_aggregate(
                        train_leave,
                        validation_leave,
                        test_leave,
                        independent_local,
                        tuned=tuned,
                    )
                )
                rows.append(
                    {
                        "branch": feature.branch,
                        "encoder_seed": feature.encoder_seed,
                        "readout": feature.readout,
                        "target_block": block,
                        "band": label,
                        "direction_start": start + 1,
                        "direction_stop": stop,
                        "band_dimension": width,
                        "valid_dimension": rank,
                        "variance_fraction": variance_fraction,
                        "predictive_mass_mean_independent": float(
                            mass_by_target["band_mass"].mean()
                        ),
                        "predictive_mass_fraction_mean_independent": float(
                            mass_fraction.mean()
                        ),
                        "reader_family": reader,
                        "alpha_band_only": alpha_band,
                        "train_r2_band_only": train_r2_band,
                        "validation_r2_band_only": val_r2_band,
                        "test_r2_band_only": test_r2_band,
                        "alpha_leave_band_out": alpha_leave,
                        "train_r2_leave_band_out": train_r2_leave,
                        "validation_r2_leave_band_out": val_r2_leave,
                        "test_r2_leave_band_out": test_r2_leave,
                        "matched_random_test_r2_mean": (
                            float(null.mean()) if not tuned else np.nan
                        ),
                        "matched_random_test_r2_std": (
                            float(null.std(ddof=1)) if not tuned else np.nan
                        ),
                        "band_only_percentile_in_random_null": (
                            100.0 * float(np.mean(null <= test_r2_band))
                            if not tuned
                            else np.nan
                        ),
                        "empirical_p_random_exceeds_band": (
                            float(np.mean(null > test_r2_band))
                            if not tuned
                            else np.nan
                        ),
                        "random_comparison_status": (
                            "matched_min_norm"
                            if not tuned
                            else "not_run_for_tuned_ridge"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _feature_artifact_valid(
    directory: Path, source_fingerprint: Mapping[str, object]
) -> bool:
    complete = directory / "complete.json"
    if not complete.is_file():
        return False
    payload = json.loads(complete.read_text())
    if payload.get("source_fingerprint") != source_fingerprint:
        raise ExperimentIntegrityError(f"stale Phase-II feature shard: {directory}")
    for record in payload["artifacts"].values():
        path = directory / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ExperimentIntegrityError(
                f"Phase-II feature shard hash failed: {path}"
            )
    return True


def _write_feature_outputs(
    directory: Path,
    tables: Mapping[str, pd.DataFrame],
    source_fingerprint: Mapping[str, object],
    diagnostics: Mapping[str, object],
) -> Mapping[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, table in tables.items():
        filename = f"{name}.parquet"
        path = directory / filename
        atomic_write_parquet(table, path)
        artifacts[name] = {
            "path": filename,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "n_rows": len(table),
        }
    payload = {
        "schema_name": "thesis.experiment01.phase2_feature_shard",
        "schema_version": 1,
        "source_fingerprint": source_fingerprint,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }
    atomic_write_json(directory / "complete.json", payload)
    return payload


def _read_feature_outputs(directory: Path) -> Mapping[str, pd.DataFrame]:
    payload = json.loads((directory / "complete.json").read_text())
    return {
        name: pd.read_parquet(directory / record["path"])
        for name, record in payload["artifacts"].items()
    }


def phase1_phase2_bridge(
    predictive_mass: pd.DataFrame,
    phase1_summary_dir: Path,
    phase1_summary: Mapping[str, object],
) -> pd.DataFrame:
    gap = pd.read_parquet(phase1_summary_dir / "gap_summary_delta_010.parquet")
    requested_k = (0, 8, 16, 32, 64, 128, 256, 508)
    gap = gap[
        gap["readout"].eq("last_concat512")
        & gap["reader_family"].eq("ridge_whiten_topk_tuned_alpha")
        & gap["feature_view"].eq("full_rank_whiten_topk")
        & gap["whiten_k_requested"].isin(requested_k)
        & gap["budget_days_per_stock"].isin([0.125, 0.25])
    ].copy()
    mass = predictive_mass[
        predictive_mass["readout"].eq("last_concat512")
        & predictive_mass["target_independent"].eq(True)
        & predictive_mass["direction_valid"].eq(True)
    ]
    mass_rows = []
    for (branch, seed, block), group in mass.groupby(
        ["branch", "encoder_seed", "target_block"], observed=True
    ):
        rank = int(group["valid_dimension"].iloc[0])
        for k in requested_k:
            effective = min(k, rank)
            if effective == 0:
                by_target = group.groupby("target_name", observed=True).agg(
                    total=("total_predictive_mass_valid", "first")
                )
                cumulative_mean = 0.0
                fraction_mean = 0.0
            else:
                selected = group[group["direction_index"].eq(effective)]
                by_target = selected.set_index("target_name")
                cumulative_mean = float(
                    by_target["cumulative_predictive_mass"].mean()
                )
                fraction_mean = float(by_target["cumulative_mass_fraction"].mean())
            mass_rows.append(
                {
                    "branch": branch,
                    "encoder_seed": int(seed),
                    "target_block": block,
                    "k_requested": k,
                    "k_effective": effective,
                    "predictive_mass_cumulative_mean_independent": cumulative_mean,
                    "predictive_mass_cumulative_fraction_mean_independent": fraction_mean,
                }
            )
    mass_frame = pd.DataFrame(mass_rows)
    rows = []
    outcome = phase1_summary["directional_last_concat512_outcome"]
    for gap_row in gap.itertuples():
        matched = mass_frame[
            mass_frame["target_block"].eq(gap_row.target_block)
            & mass_frame["k_requested"].eq(int(gap_row.whiten_k_requested))
        ]
        for mass_row in matched.itertuples():
            rows.append(
                {
                    "target_block": gap_row.target_block,
                    "readout": "last_concat512",
                    "k_requested": int(gap_row.whiten_k_requested),
                    "k_effective": int(mass_row.k_effective),
                    "budget_days_per_stock": float(gap_row.budget_days_per_stock),
                    "phase1_gap_mean": float(gap_row.mean),
                    "phase1_gap_lower": float(gap_row.lower),
                    "phase1_gap_upper": float(gap_row.upper),
                    "phase1_gap_robust": bool(gap_row.robust),
                    "branch": mass_row.branch,
                    "encoder_seed": int(mass_row.encoder_seed),
                    "predictive_mass_cumulative_mean_independent": float(
                        mass_row.predictive_mass_cumulative_mean_independent
                    ),
                    "predictive_mass_cumulative_fraction_mean_independent": float(
                        mass_row.predictive_mass_cumulative_fraction_mean_independent
                    ),
                    "is_k_50gap": int(gap_row.whiten_k_requested)
                    == int(outcome["k_50gap"]),
                    "is_k_nonrobust": int(gap_row.whiten_k_requested)
                    == int(outcome["k_nonrobust"]),
                    "phase1_outcome_unchanged": outcome["outcome"],
                }
            )
    return pd.DataFrame(rows)


def run_phase2(
    bundle: InputBundle,
    out_dir: str | Path,
    *,
    phase1_dir: str | Path,
    reproduction_gate: str | Path,
    config: Phase2Config = Phase2Config(),
) -> Mapping[str, object]:
    config.validate()
    destination = Path(out_dir).resolve()
    phase1_root = Path(phase1_dir).resolve()
    gate_path = Path(reproduction_gate).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final_names = (
        "phase2_results.parquet",
        "predictive_mass.parquet",
        "random_subspace_null.parquet",
        "spectral_bands.parquet",
        "phase1_phase2_bridge.parquet",
        "failures.parquet",
        "metadata.json",
    )
    if any((destination / name).exists() for name in final_names):
        raise FileExistsError("refusing to overwrite finalized Phase-II artifacts")
    gate = json.loads(gate_path.read_text())
    if gate.get("passed") is not True:
        raise ExperimentIntegrityError("Phase-II reproduction gate did not pass")
    phase1_metadata_path = phase1_root / "metadata.json"
    phase1_metadata = json.loads(phase1_metadata_path.read_text())
    phase1_summary_dir = phase1_root.parent / "summary"
    phase1_summary_path = phase1_summary_dir / "summary.json"
    phase1_summary = json.loads(phase1_summary_path.read_text())
    if phase1_summary["directional_last_concat512_outcome"]["outcome"] != "A1":
        raise ExperimentIntegrityError("frozen Phase-I technical outcome is not A1")
    subset_records = _selected_subset_records(phase1_root)
    input_fingerprint = {
        "bundle_manifest_sha256": sha256_file(bundle.root / "manifest.json"),
        "phase1_results_sha256": sha256_file(phase1_root / "results.parquet"),
        "phase1_metadata_sha256": sha256_file(phase1_metadata_path),
        "phase1_summary_sha256": sha256_file(phase1_summary_path),
        "reproduction_gate_sha256": sha256_file(gate_path),
        "phase1_outcome": "A1",
        "phase2_implementation_sha256": sha256_file(__file__),
    }
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    commit = _git_commit(Path(__file__).resolve().parents[1])
    cache_dir = destination / "cache"
    shards_dir = destination / "feature_shards"
    selected = [
        feature
        for feature in bundle.feature_sets
        if feature.branch in config.branches and feature.readout in config.readouts
    ]
    if len(selected) != 18:
        raise ExperimentIntegrityError("Phase II requires exactly 18 feature sets")
    phase1_full_min_norm = pd.read_parquet(
        phase1_root / "results.parquet",
        columns=[
            "branch",
            "encoder_seed",
            "readout",
            "target_block",
            "target_name",
            "budget_kind",
            "reader_family",
            "train_r2",
            "val_r2",
            "test_r2",
        ],
        filters=[
            ("budget_kind", "==", "full_train"),
            ("reader_family", "==", "min_norm_ols_raw"),
        ],
    )
    if len(phase1_full_min_norm) != 18 * len(bundle.target_definitions):
        raise ExperimentIntegrityError(
            "frozen Phase-I full-train min-norm inventory is not exact"
        )
    cache_metadata = []
    feature_runtime = {}
    for index, feature in enumerate(selected):
        feature_started = time.perf_counter()
        tag = feature_tag(feature)
        shard_dir = shards_dir / tag
        feature_fingerprint = {
            **input_fingerprint,
            "feature": list(feature.key),
            "random_draws": config.random_draws,
            "selected_budgets": [list(value) for value in PHASE2_SELECTED_BUDGETS],
            "k_base": list(PHASE2_K_BASE),
            "bands": [list(value) for value in PHASE2_BANDS],
        }
        if _feature_artifact_valid(shard_dir, feature_fingerprint):
            print(f"[Experiment 01 Phase II] {index + 1}/18 {tag} cached")
            continue
        statistics, cache_record = prepare_feature_cache(
            bundle,
            feature,
            phase1_dir=phase1_root,
            phase1_metadata=phase1_metadata,
            cache_dir=cache_dir,
            subset_records=subset_records,
            chunk_rows=config.chunk_rows,
        )
        cache_metadata.append(cache_record)
        mass = predictive_mass_table(bundle, feature, statistics)
        ladder = phase2_ladder_table(
            bundle, feature, statistics, commit=commit
        )
        parity = full_rank_phase1_parity_gate(
            bundle,
            feature,
            statistics,
            ladder,
            phase1_full_min_norm,
        )
        random_null = random_subspace_table(
            bundle,
            feature,
            statistics,
            ladder,
            n_draws=config.random_draws,
        )
        bands = spectral_band_table(
            bundle, feature, statistics, mass, random_null
        )
        failures = ladder[~ladder["fit_status"].eq("ok")].copy()
        _write_feature_outputs(
            shard_dir,
            {
                "phase2_results": ladder,
                "predictive_mass": mass,
                "random_subspace_null": random_null,
                "spectral_bands": bands,
                "failures": failures,
            },
            feature_fingerprint,
            {"phase1_full_rank_min_norm_parity": parity},
        )
        feature_runtime[tag] = time.perf_counter() - feature_started
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(f"[Experiment 01 Phase II] {index + 1}/18 {tag} complete")

    combined: dict[str, list[pd.DataFrame]] = {
        "phase2_results": [],
        "predictive_mass": [],
        "random_subspace_null": [],
        "spectral_bands": [],
        "failures": [],
    }
    feature_shards = []
    for feature in selected:
        tag = feature_tag(feature)
        directory = shards_dir / tag
        tables = _read_feature_outputs(directory)
        for name in combined:
            combined[name].append(tables[name])
        feature_shards.append(json.loads((directory / "complete.json").read_text()))
    finalized = {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in combined.items()
    }
    bridge = phase1_phase2_bridge(
        finalized["predictive_mass"], phase1_summary_dir, phase1_summary
    )
    finalized["phase1_phase2_bridge"] = bridge
    output_names = {
        "phase2_results": "phase2_results.parquet",
        "predictive_mass": "predictive_mass.parquet",
        "random_subspace_null": "random_subspace_null.parquet",
        "spectral_bands": "spectral_bands.parquet",
        "phase1_phase2_bridge": "phase1_phase2_bridge.parquet",
        "failures": "failures.parquet",
    }
    artifacts = {}
    for name, filename in output_names.items():
        path = destination / filename
        atomic_write_parquet(finalized[name], path)
        artifacts[name] = {
            "path": filename,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "n_rows": len(finalized[name]),
        }
    metadata = {
        "schema_name": "thesis.experiment01.phase2_metadata",
        "schema_version": 1,
        "status": "complete",
        "phase2_version": PHASE2_VERSION,
        "commit_hash": commit,
        "input_fingerprint": input_fingerprint,
        "protocol": {
            "scope": "preregistered_spectral_diagnostic_only",
            "pca_fit": "all_unlabelled_canonical_train_features_only",
            "validation_policy": "alpha_selection_only",
            "test_policy": "fixed_test_once_after_validation_selection",
            "readouts": list(READOUTS),
            "branches": list(BRANCHES),
            "encoder_seeds": [0, 1, 2],
            "target_blocks": ["directional", "volatility", "timing"],
            "k_base": list(PHASE2_K_BASE),
            "bands": [list(value) for value in PHASE2_BANDS],
            "selected_budgets": [list(value) for value in PHASE2_SELECTED_BUDGETS],
            "random_draws": config.random_draws,
            "random_reader": "min_norm_ols_diagnostic",
            "ridge": "lambda=alpha*trace(labelled_design_covariance)/m",
            "phase1_modified": False,
            "phase1_outcome_modified": False,
            "phase3_started": False,
            "bundle_hashes_verified_this_run": (
                config.bundle_hashes_verified_this_run
            ),
        },
        "compute": {
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss,
            "feature_runtime_seconds": feature_runtime,
            "feature_scan_policy": "one_train_scan_plus_one_validation_and_test_stats_scan_per_feature_set",
            "feature_shards_resumable": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "cache_records_created_this_run": cache_metadata,
        "feature_shards": feature_shards,
        "phase1_full_rank_min_norm_parity": [
            record["diagnostics"]["phase1_full_rank_min_norm_parity"]
            for record in feature_shards
        ],
        "artifacts": artifacts,
    }
    atomic_write_json(destination / "metadata.json", metadata)
    return metadata

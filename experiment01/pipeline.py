"""Streaming Phase-I runner for the preregistered Experiment 01 grid."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

from .constants import (
    ALPHA_GRID,
    BRANCHES,
    EXPERIMENT_VERSION,
    READOUTS,
    RESULT_COLUMNS,
)
from .errors import ExperimentIntegrityError
from .io import atomic_savez, atomic_write_json, sha256_file
from .linear import (
    Design,
    Eigensystem,
    SufficientStats,
    WhiteningFit,
    WhiteningTransform,
    direct_ridge_solution,
    evaluate_stats,
    fit_alpha,
    fit_unlabelled_covariance,
    select_targets,
    sufficient_stats,
    transformed_design,
    tune_alpha,
    whitening_k_grid,
    whitening_transform,
)
from .results import attach_operational_ceilings
from .schema import FeatureSet, InputBundle, iter_target_indices
from .subsets import (
    SubsetSelection,
    anchor_sensitivity,
    generate_all_selections,
    write_subset_manifests,
)


@dataclass(frozen=True)
class Phase1Config:
    branches: tuple[str, ...] = BRANCHES
    readouts: tuple[str, ...] = READOUTS
    target_blocks: tuple[str, ...] = ("directional", "volatility", "timing")
    run_common_alpha: bool = True
    run_tuned_alpha: bool = True
    run_min_norm: bool = True
    run_whitening: bool = True
    run_anchor_sensitivity: bool = True
    chunk_rows: int = 65536
    direct_crosscheck_rows: int = 2048

    def validate(self) -> None:
        if not self.branches or not set(self.branches).issubset(BRANCHES):
            raise ValueError("Phase1Config.branches is invalid")
        if not self.readouts or not set(self.readouts).issubset(READOUTS):
            raise ValueError("Phase1Config.readouts is invalid")
        if not self.target_blocks or not set(self.target_blocks).issubset(
            {"directional", "volatility", "timing"}
        ):
            raise ValueError("Phase1Config.target_blocks is invalid")
        if not (self.run_common_alpha or self.run_tuned_alpha or self.run_min_norm):
            raise ValueError("at least one reader family must be enabled")
        if self.chunk_rows <= 0 or self.direct_crosscheck_rows <= 0:
            raise ValueError("chunk sizes must be positive")


class ParquetSink:
    """Append homogeneously typed pandas batches to one Parquet file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.n_rows = 0

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        ordered = frame.loc[:, RESULT_COLUMNS].copy()
        table = pa.Table.from_pandas(ordered, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=True,
            )
        else:
            assert self.schema is not None
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.n_rows += len(frame)

    def close(self) -> None:
        if self.writer is None:
            # Keep required artifacts present even for a completely failed run.
            pd.DataFrame(columns=RESULT_COLUMNS).to_parquet(self.path, index=False)
        else:
            self.writer.close()
            self.writer = None


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


def _feature_tag(feature: FeatureSet) -> str:
    return f"{feature.branch}_seed{feature.encoder_seed}_{feature.readout}"


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _target_layout(
    bundle: InputBundle, blocks: Sequence[str]
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    definitions = bundle.target_definitions
    result = []
    for block in blocks:
        global_indices, independent_global = iter_target_indices(definitions, block)
        local_independent = np.flatnonzero(
            np.isin(global_indices, independent_global)
        ).astype(np.int64)
        independent_flags = np.asarray(
            [definitions[index].independent for index in global_indices],
            dtype=bool,
        )
        result.append(
            (block, global_indices, local_independent, independent_flags)
        )
    return result


def _selection_pathways(
    selections: Sequence[SubsetSelection],
) -> list[list[SubsetSelection]]:
    full = [value for value in selections if value.budget.is_full_train]
    if len(full) != 1:
        raise ExperimentIntegrityError("expected exactly one full_train subset")
    pathways: list[list[SubsetSelection]] = []
    nonfull_seeds = sorted(
        {
            value.subsample_seed
            for value in selections
            if value.subsample_seed >= 0
        }
    )
    for seed in nonfull_seeds:
        values = [
            value for value in selections if value.subsample_seed == seed
        ]
        values.sort(key=lambda value: float(value.budget.days_per_stock))
        if seed == 0:
            values.append(full[0])
        pathways.append(values)
    return pathways


def _label_sensitivity_selections(
    selections: Sequence[SubsetSelection],
) -> tuple[SubsetSelection, ...]:
    values = []
    for selection in selections:
        anchor = float(selection.anchor_quantile)
        if anchor < 0.2:
            label = "opening"
        elif anchor > 0.8:
            label = "closing"
        else:
            label = "middle"
        values.append(
            replace(
                selection,
                budget=replace(
                    selection.budget,
                    kind=f"fractional_sensitivity_{label}",
                    label=f"{selection.budget.label}_{label}",
                ),
            )
        )
    return tuple(values)


def _add_positions(
    accumulator: SufficientStats,
    x: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    chunk_rows: int,
    compute_log: dict[str, object],
) -> None:
    for start in range(0, len(positions), chunk_rows):
        index = positions[start : start + chunk_rows]
        accumulator.add_rows(x[index], y[index])
        compute_log["gram_updates"] = int(compute_log["gram_updates"]) + 1


def _assert_stats_equal(left: SufficientStats, right: SufficientStats) -> None:
    if left.n != right.n:
        raise ExperimentIntegrityError("incremental/direct row counts differ")
    for name in ("x_sum", "y_sum", "xtx", "xty", "yty"):
        np.testing.assert_allclose(
            getattr(left, name),
            getattr(right, name),
            rtol=2e-11,
            atol=2e-8,
            err_msg=f"incremental sufficient statistic mismatch: {name}",
        )


def _direct_solver_crosscheck(
    x: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    maximum_rows: int,
) -> None:
    chosen = positions[: min(len(positions), maximum_rows)]
    if len(chosen) < 2:
        return
    x_small = np.asarray(x[chosen], dtype=np.float64)
    y_small = np.asarray(y[chosen], dtype=np.float64)
    stats = sufficient_stats(x_small, y_small)
    design = transformed_design(stats)
    for alpha in (0.0, 1e-3, 1.0):
        gram_model = fit_alpha(design, alpha)
        direct_model = direct_ridge_solution(x_small, y_small, alpha)
        np.testing.assert_allclose(
            x_small @ gram_model.beta_raw + gram_model.intercept,
            x_small @ direct_model.beta_raw + direct_model.intercept,
            rtol=2e-7,
            atol=2e-8,
            err_msg=f"Gram/direct solver mismatch at alpha={alpha}",
        )


def _base_row(
    *,
    commit: str,
    feature: FeatureSet,
    block: str,
    target_name: str,
    target_independent: bool,
    selection: SubsetSelection,
    design: Design | None,
    feature_view: str,
    whiten_requested: float,
    whiten_effective: float,
    reader_family: str,
    alpha: float,
    lambda_absolute: float,
    alpha_selected: bool,
    train_r2: float,
    val_r2: float,
    test_r2: float,
    fit_status: str,
    failure_reason: str,
    runtime_seconds: float,
) -> dict[str, object]:
    diagnostics = design.eigensystem.diagnostics if design is not None else None
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "commit_hash": commit,
        "branch": feature.branch,
        "encoder_seed": feature.encoder_seed,
        "readout": feature.readout,
        "target_block": block,
        "target_name": target_name,
        "target_independent": bool(target_independent),
        "budget_kind": selection.budget.kind,
        "budget_days_per_stock": (
            np.nan
            if selection.budget.days_per_stock is None
            else float(selection.budget.days_per_stock)
        ),
        "budget_stock_day_equivalents": selection.stock_day_equivalents,
        "n_stock_days": selection.n_stock_days,
        "n_rows": selection.n_rows,
        "n_rows_over_dim": selection.n_rows / feature.dimension,
        "subsample_seed": selection.subsample_seed,
        "block_anchor_quantile": (
            np.nan
            if selection.anchor_quantile is None
            else selection.anchor_quantile
        ),
        "feature_view": feature_view,
        "feature_dim": feature.dimension,
        "whiten_k_requested": whiten_requested,
        "whiten_k_effective": whiten_effective,
        "pca_fraction": np.nan,
        "subspace_seed": np.nan,
        "reader_family": reader_family,
        "alpha": alpha,
        "lambda_absolute": lambda_absolute,
        "alpha_selected": bool(alpha_selected),
        "trace_cov": (
            np.nan if diagnostics is None else diagnostics.trace_cov
        ),
        "trace_cov_over_dim": (
            np.nan if diagnostics is None else diagnostics.trace_cov_over_dim
        ),
        "lambda_max_cov": (
            np.nan if diagnostics is None else diagnostics.lambda_max_cov
        ),
        "lambda_min_valid_cov": (
            np.nan
            if diagnostics is None
            else diagnostics.lambda_min_valid_cov
        ),
        "condition_number": (
            np.nan if diagnostics is None else diagnostics.condition_number
        ),
        "numerical_rank": (
            np.nan if diagnostics is None else diagnostics.numerical_rank
        ),
        "numerical_tolerance": (
            np.nan
            if diagnostics is None
            else diagnostics.numerical_tolerance
        ),
        "train_r2": train_r2,
        "val_r2": val_r2,
        "test_r2": test_r2,
        "full_budget_test_r2": np.nan,
        "normalized_recovery": np.nan,
        "ceiling_eligible": False,
        "fit_status": fit_status,
        "failure_reason": failure_reason,
        "runtime_seconds": runtime_seconds,
    }


def _models_for_design(
    *,
    commit: str,
    feature: FeatureSet,
    block: str,
    target_names: Sequence[str],
    independent_flags: np.ndarray,
    independent_indices: np.ndarray,
    selection: SubsetSelection,
    design: Design,
    train_stats: SufficientStats,
    validation_stats: SufficientStats,
    test_stats: SufficientStats,
    feature_view: str,
    whiten_requested: float,
    whiten_effective: float,
    include_min_norm: bool,
    include_common: bool,
    include_tuned: bool,
    compute_log: dict[str, object],
) -> pd.DataFrame:
    started = time.perf_counter()
    models = [fit_alpha(design, float(alpha)) for alpha in ALPHA_GRID]
    compute_log["ridge_models"] = int(compute_log["ridge_models"]) + len(models)
    tuned = tune_alpha(
        design,
        None,
        None,
        independent_indices,
        validation_stats=validation_stats,
    )
    evaluations = [
        (
            evaluate_stats(model, train_stats),
            evaluate_stats(model, validation_stats),
            evaluate_stats(model, test_stats),
        )
        for model in models
    ]
    emitted: list[tuple[str, int, bool]] = []
    if include_min_norm:
        emitted.append(("min_norm_ols_raw", 0, False))
    if include_common:
        family = (
            "ridge_raw_common_alpha"
            if feature_view == "full_rank_raw"
            else "ridge_whiten_topk_common_alpha"
        )
        emitted.extend(
            (family, index, index == tuned.index)
            for index in range(len(ALPHA_GRID))
        )
    if include_tuned:
        family = (
            "ridge_raw_tuned_alpha"
            if feature_view == "full_rank_raw"
            else "ridge_whiten_topk_tuned_alpha"
        )
        emitted.append((family, tuned.index, True))
    elapsed = time.perf_counter() - started
    per_model_runtime = elapsed / max(1, len(emitted))
    rows: list[dict[str, object]] = []
    for family, model_index, selected in emitted:
        model = models[model_index]
        train_scores, validation_scores, test_scores = evaluations[model_index]
        for target_index, target_name in enumerate(target_names):
            reasons = []
            for split_name, scores in (
                ("train", train_scores),
                ("validation", validation_scores),
                ("test", test_scores),
            ):
                if not scores.valid[target_index]:
                    reasons.append(f"{split_name}:{scores.reasons[target_index]}")
            status = "ok" if not reasons else "invalid"
            rows.append(
                _base_row(
                    commit=commit,
                    feature=feature,
                    block=block,
                    target_name=target_name,
                    target_independent=bool(independent_flags[target_index]),
                    selection=selection,
                    design=design,
                    feature_view=feature_view,
                    whiten_requested=whiten_requested,
                    whiten_effective=whiten_effective,
                    reader_family=family,
                    alpha=model.alpha,
                    lambda_absolute=model.lambda_absolute,
                    alpha_selected=selected,
                    train_r2=float(train_scores.values[target_index]),
                    val_r2=float(validation_scores.values[target_index]),
                    test_r2=float(test_scores.values[target_index]),
                    fit_status=status,
                    failure_reason=";".join(reasons),
                    runtime_seconds=per_model_runtime,
                )
            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _invalid_whitening_rows(
    *,
    commit: str,
    feature: FeatureSet,
    block: str,
    target_names: Sequence[str],
    independent_flags: np.ndarray,
    selection: SubsetSelection,
    requested_k: int,
    reason: str,
    include_common: bool,
    include_tuned: bool,
) -> pd.DataFrame:
    rows = []
    families: list[tuple[str, float]] = []
    if include_common:
        families.extend(
            ("ridge_whiten_topk_common_alpha", float(alpha))
            for alpha in ALPHA_GRID
        )
    if include_tuned:
        families.append(("ridge_whiten_topk_tuned_alpha", np.nan))
    for family, alpha in families:
        for index, name in enumerate(target_names):
            rows.append(
                _base_row(
                    commit=commit,
                    feature=feature,
                    block=block,
                    target_name=name,
                    target_independent=bool(independent_flags[index]),
                    selection=selection,
                    design=None,
                    feature_view="full_rank_whiten_topk",
                    whiten_requested=float(requested_k),
                    whiten_effective=np.nan,
                    reader_family=family,
                    alpha=alpha,
                    lambda_absolute=np.nan,
                    alpha_selected=False,
                    train_r2=np.nan,
                    val_r2=np.nan,
                    test_r2=np.nan,
                    fit_status="invalid",
                    failure_reason=reason,
                    runtime_seconds=0.0,
                )
            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _selection_rows(
    *,
    bundle: InputBundle,
    feature: FeatureSet,
    selection: SubsetSelection,
    all_train_stats: SufficientStats,
    validation_stats: SufficientStats,
    test_stats: SufficientStats,
    whitening_transforms: Mapping[int, WhiteningTransform],
    target_layout: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    config: Phase1Config,
    commit: str,
    compute_log: dict[str, object],
) -> pd.DataFrame:
    frames = []
    for block, global_indices, local_independent, independent_flags in target_layout:
        train_block = select_targets(all_train_stats, global_indices)
        validation_block = select_targets(validation_stats, global_indices)
        test_block = select_targets(test_stats, global_indices)
        names = [bundle.target_definitions[index].name for index in global_indices]
        raw_design = transformed_design(train_block)
        compute_log["eigendecompositions"] = (
            int(compute_log["eigendecompositions"]) + 1
        )
        frames.append(
            _models_for_design(
                commit=commit,
                feature=feature,
                block=block,
                target_names=names,
                independent_flags=independent_flags,
                independent_indices=local_independent,
                selection=selection,
                design=raw_design,
                train_stats=train_block,
                validation_stats=validation_block,
                test_stats=test_block,
                feature_view="full_rank_raw",
                whiten_requested=np.nan,
                whiten_effective=np.nan,
                include_min_norm=config.run_min_norm,
                include_common=config.run_common_alpha,
                include_tuned=config.run_tuned_alpha,
                compute_log=compute_log,
            )
        )
        if not config.run_whitening:
            continue
        for requested_k, transform in sorted(whitening_transforms.items()):
            if not transform.valid:
                frames.append(
                    _invalid_whitening_rows(
                        commit=commit,
                        feature=feature,
                        block=block,
                        target_names=names,
                        independent_flags=independent_flags,
                        selection=selection,
                        requested_k=requested_k,
                        reason=transform.failure_reason,
                        include_common=config.run_common_alpha,
                        include_tuned=config.run_tuned_alpha,
                    )
                )
                continue
            assert transform.matrix is not None
            design = transformed_design(train_block, transform.matrix)
            compute_log["eigendecompositions"] = (
                int(compute_log["eigendecompositions"]) + 1
            )
            compute_log["transform_cache_hits"] = (
                int(compute_log["transform_cache_hits"]) + 1
            )
            frames.append(
                _models_for_design(
                    commit=commit,
                    feature=feature,
                    block=block,
                    target_names=names,
                    independent_flags=independent_flags,
                    independent_indices=local_independent,
                    selection=selection,
                    design=design,
                    train_stats=train_block,
                    validation_stats=validation_block,
                    test_stats=test_block,
                    feature_view="full_rank_whiten_topk",
                    whiten_requested=float(requested_k),
                    whiten_effective=float(transform.effective_k),
                    include_min_norm=False,
                    include_common=config.run_common_alpha,
                    include_tuned=config.run_tuned_alpha,
                    compute_log=compute_log,
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=RESULT_COLUMNS
    )


def _covariance_record(feature: FeatureSet, fit: WhiteningFit) -> dict[str, object]:
    diagnostics = fit.eigensystem.diagnostics
    return {
        "branch": feature.branch,
        "encoder_seed": feature.encoder_seed,
        "readout": feature.readout,
        "n_unlabelled_train_rows": fit.n_rows,
        "dimension": feature.dimension,
        "trace_cov": _finite_or_none(diagnostics.trace_cov),
        "trace_cov_over_dim": _finite_or_none(diagnostics.trace_cov_over_dim),
        "lambda_max_cov": _finite_or_none(diagnostics.lambda_max_cov),
        "lambda_min_valid_cov": _finite_or_none(
            diagnostics.lambda_min_valid_cov
        ),
        "condition_number": _finite_or_none(diagnostics.condition_number),
        "numerical_rank": diagnostics.numerical_rank,
        "numerical_tolerance": diagnostics.numerical_tolerance,
    }


def _trace_ratios(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    frame = pd.DataFrame(records)
    for (seed, readout), group in frame.groupby(
        ["encoder_seed", "readout"], observed=True
    ):
        values = {
            str(row.branch): float(row.trace_cov_over_dim)
            for row in group.itertuples()
        }
        for left_index, left in enumerate(BRANCHES):
            for right in BRANCHES[left_index + 1 :]:
                if left in values and right in values:
                    rows.append(
                        {
                            "encoder_seed": int(seed),
                            "readout": str(readout),
                            "numerator": left,
                            "denominator": right,
                            "trace_cov_over_dim_ratio": values[left] / values[right],
                        }
                    )
    return rows


def run_phase1(
    bundle: InputBundle,
    out_dir: str | Path,
    config: Phase1Config = Phase1Config(),
) -> Mapping[str, object]:
    """Execute Phase I and write a finalized tidy Parquet table.

    The caller must already have passed ``load_input_bundle``.  Test targets are
    never used for fitting, transform estimation or alpha selection.
    """
    config.validate()
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.parquet"
    failures_path = destination / "failures.parquet"
    sensitivity_path = destination / "time_of_day_sensitivity.parquet"
    if results_path.exists() or failures_path.exists() or sensitivity_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing Experiment 01 result artifacts"
        )
    started = time.perf_counter()
    commit = _git_commit(Path(__file__).resolve().parents[1])
    selections = generate_all_selections(bundle.rows["train"])
    train_key_hash = bundle.manifest["splits"]["train"]["row_key_sha256"]
    subset_metadata = write_subset_manifests(
        bundle.rows["train"],
        selections,
        destination,
        source_row_key_sha256=train_key_hash,
    )
    sensitivity_values = _label_sensitivity_selections(
        anchor_sensitivity(bundle.rows["train"])
    )
    sensitivity_subset_metadata = write_subset_manifests(
        bundle.rows["train"],
        sensitivity_values,
        destination / "time_of_day_sensitivity_subsets",
        source_row_key_sha256=train_key_hash,
    )
    pathways = _selection_pathways(selections)
    target_layout = _target_layout(bundle, config.target_blocks)
    y_train = bundle.load_targets("train")
    y_validation = bundle.load_targets("validation")
    y_test = bundle.load_targets("test")

    result_sink = ParquetSink(results_path)
    failure_sink = ParquetSink(failures_path)
    sensitivity_sink = ParquetSink(sensitivity_path)
    compute_log: dict[str, object] = {
        "raw_feature_loads": 0,
        "gram_updates": 0,
        "eigendecompositions": 0,
        "ridge_models": 0,
        "transform_cache_hits": 0,
        "incremental_direct_checks": 0,
        "direct_solver_checks": 0,
        "stage_runtime_seconds": {},
    }
    covariance_records: list[dict[str, object]] = []
    transform_records: list[dict[str, object]] = []
    selected_features = [
        feature
        for feature in bundle.feature_sets
        if feature.branch in config.branches and feature.readout in config.readouts
    ]
    if not selected_features:
        raise ExperimentIntegrityError("Phase I selection contains no feature set")
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    try:
        for feature_index, feature in enumerate(selected_features):
            feature_started = time.perf_counter()
            x_train = bundle.load_features(feature, "train")
            x_validation = bundle.load_features(feature, "validation")
            x_test = bundle.load_features(feature, "test")
            compute_log["raw_feature_loads"] = int(compute_log["raw_feature_loads"]) + 3
            whitening_fit = fit_unlabelled_covariance(
                x_train, chunk_rows=config.chunk_rows
            )
            compute_log["eigendecompositions"] = (
                int(compute_log["eigendecompositions"]) + 1
            )
            fixed_whitening_transforms = {
                requested_k: whitening_transform(whitening_fit, requested_k)
                for requested_k in whitening_k_grid(whitening_fit)
            }
            transform_relative = Path("transforms") / f"{_feature_tag(feature)}.npz"
            transform_path = destination / transform_relative
            atomic_savez(
                transform_path,
                unlabelled_train_mean=whitening_fit.mean,
                covariance_eigenvalues=whitening_fit.eigensystem.eigenvalues,
                covariance_eigenvectors=whitening_fit.eigensystem.eigenvectors,
                numerical_tolerance=np.asarray(
                    whitening_fit.eigensystem.diagnostics.numerical_tolerance,
                    dtype=np.float64,
                ),
                numerical_rank=np.asarray(
                    whitening_fit.valid_dimension, dtype=np.int64
                ),
                k_grid=np.asarray(
                    sorted(fixed_whitening_transforms), dtype=np.int64
                ),
            )
            transform_records.append(
                {
                    "branch": feature.branch,
                    "encoder_seed": feature.encoder_seed,
                    "readout": feature.readout,
                    "path": str(transform_relative),
                    "sha256": sha256_file(transform_path),
                    "size_bytes": transform_path.stat().st_size,
                    "algorithm": "progressive_topk_whitening.v1",
                    "k_diagnostics": [
                        {
                            "requested_k": int(requested_k),
                            "effective_k": transform.effective_k,
                            "valid": transform.valid,
                            "failure_reason": transform.failure_reason,
                            "numerical_tolerance": transform.numerical_tolerance,
                            "smallest_inverted_eigenvalue": (
                                _finite_or_none(
                                    transform.smallest_inverted_eigenvalue
                                )
                            ),
                            "transform_condition_number": (
                                _finite_or_none(transform.condition_number)
                            ),
                        }
                        for requested_k, transform in sorted(
                            fixed_whitening_transforms.items()
                        )
                    ],
                }
            )
            covariance_records.append(_covariance_record(feature, whitening_fit))
            validation_stats = sufficient_stats(
                x_validation, y_validation, chunk_rows=config.chunk_rows
            )
            test_stats = sufficient_stats(
                x_test, y_test, chunk_rows=config.chunk_rows
            )
            compute_log["gram_updates"] = int(compute_log["gram_updates"]) + (
                int(np.ceil(len(x_validation) / config.chunk_rows))
                + int(np.ceil(len(x_test) / config.chunk_rows))
            )
            crosschecked = False
            full_reference: pd.DataFrame | None = None
            buffered_seed_zero: list[pd.DataFrame] = []
            for pathway in pathways:
                accumulator = SufficientStats.zeros(
                    feature.dimension, len(bundle.target_definitions)
                )
                previous = np.empty(0, dtype=np.int64)
                for selection in pathway:
                    added = np.setdiff1d(
                        selection.row_indices, previous, assume_unique=True
                    )
                    _add_positions(
                        accumulator,
                        x_train,
                        y_train,
                        added,
                        config.chunk_rows,
                        compute_log,
                    )
                    if not crosschecked:
                        direct = sufficient_stats(
                            x_train[selection.row_indices],
                            y_train[selection.row_indices],
                            chunk_rows=config.chunk_rows,
                        )
                        _assert_stats_equal(accumulator, direct)
                        _direct_solver_crosscheck(
                            x_train,
                            y_train,
                            selection.row_indices,
                            config.direct_crosscheck_rows,
                        )
                        compute_log["incremental_direct_checks"] = (
                            int(compute_log["incremental_direct_checks"]) + 1
                        )
                        compute_log["direct_solver_checks"] = (
                            int(compute_log["direct_solver_checks"]) + 3
                        )
                        crosschecked = True
                    batch = _selection_rows(
                        bundle=bundle,
                        feature=feature,
                        selection=selection,
                        all_train_stats=accumulator,
                        validation_stats=validation_stats,
                        test_stats=test_stats,
                        whitening_transforms=fixed_whitening_transforms,
                        target_layout=target_layout,
                        config=config,
                        commit=commit,
                        compute_log=compute_log,
                    )
                    if pathway[0].subsample_seed == 0:
                        buffered_seed_zero.append(batch)
                        if selection.budget.is_full_train:
                            combined = pd.concat(
                                buffered_seed_zero, ignore_index=True
                            )
                            finalized = attach_operational_ceilings(combined)
                            full_reference = combined[
                                combined["budget_kind"].eq("full_train")
                            ].copy()
                            result_sink.append(finalized)
                            failure_sink.append(
                                finalized[~finalized["fit_status"].eq("ok")]
                            )
                            buffered_seed_zero.clear()
                    else:
                        if full_reference is None:
                            raise ExperimentIntegrityError(
                                "full_train ceilings were not produced before low-budget rows"
                            )
                        combined = pd.concat(
                            [full_reference, batch], ignore_index=True
                        )
                        finalized = attach_operational_ceilings(combined)
                        current = finalized.iloc[len(full_reference) :].copy()
                        result_sink.append(current)
                        failure_sink.append(
                            current[~current["fit_status"].eq("ok")]
                        )
                    previous = selection.row_indices
                    peak_rss = max(peak_rss, process.memory_info().rss)
            if (
                config.run_anchor_sensitivity
                and config.run_tuned_alpha
                and feature.branch in {"supervised", "jepa_horizon"}
                and feature.readout == "last_concat512"
            ):
                if full_reference is None:
                    raise ExperimentIntegrityError(
                        "time-of-day sensitivity requires full_train ceilings"
                    )
                sensitivity_config = Phase1Config(
                    branches=(feature.branch,),
                    readouts=(feature.readout,),
                    target_blocks=("directional",),
                    run_common_alpha=False,
                    run_tuned_alpha=True,
                    run_min_norm=False,
                    run_whitening=False,
                    run_anchor_sensitivity=False,
                    chunk_rows=config.chunk_rows,
                    direct_crosscheck_rows=config.direct_crosscheck_rows,
                )
                grouped_sensitivity: dict[tuple[str, int], list[SubsetSelection]] = {}
                for labelled in sensitivity_values:
                    anchor_label = labelled.budget.kind.rsplit("_", 1)[-1]
                    grouped_sensitivity.setdefault(
                        (anchor_label, labelled.subsample_seed), []
                    ).append(labelled)
                for _, values in sorted(grouped_sensitivity.items()):
                    values.sort(key=lambda item: float(item.budget.days_per_stock))
                    accumulator = SufficientStats.zeros(
                        feature.dimension, len(bundle.target_definitions)
                    )
                    previous = np.empty(0, dtype=np.int64)
                    for selection in values:
                        added = np.setdiff1d(
                            selection.row_indices, previous, assume_unique=True
                        )
                        _add_positions(
                            accumulator,
                            x_train,
                            y_train,
                            added,
                            config.chunk_rows,
                            compute_log,
                        )
                        batch = _selection_rows(
                            bundle=bundle,
                            feature=feature,
                            selection=selection,
                            all_train_stats=accumulator,
                            validation_stats=validation_stats,
                            test_stats=test_stats,
                            whitening_transforms=fixed_whitening_transforms,
                            target_layout=_target_layout(
                                bundle, sensitivity_config.target_blocks
                            ),
                            config=sensitivity_config,
                            commit=commit,
                            compute_log=compute_log,
                        )
                        combined = pd.concat(
                            [full_reference, batch], ignore_index=True
                        )
                        finalized = attach_operational_ceilings(combined)
                        current = finalized.iloc[len(full_reference) :].copy()
                        sensitivity_sink.append(current)
                        failure_sink.append(
                            current[~current["fit_status"].eq("ok")]
                        )
                        previous = selection.row_indices
            if buffered_seed_zero:
                raise ExperimentIntegrityError(
                    f"{_feature_tag(feature)} did not reach full_train"
                )
            compute_log["stage_runtime_seconds"][_feature_tag(feature)] = (
                time.perf_counter() - feature_started
            )
            print(
                f"[Experiment 01] {feature_index + 1}/{len(selected_features)} "
                f"{_feature_tag(feature)} complete"
            )
    finally:
        result_sink.close()
        failure_sink.close()
        sensitivity_sink.close()

    total_runtime = time.perf_counter() - started
    compute_log["runtime_seconds_total"] = total_runtime
    compute_log["peak_rss_bytes"] = peak_rss
    metadata: dict[str, object] = {
        "experiment_version": EXPERIMENT_VERSION,
        "commit_hash": commit,
        "input_bundle": str(bundle.root),
        "input_manifest_sha256": sha256_file(bundle.root / "manifest.json"),
        "result_schema": list(RESULT_COLUMNS),
        "alpha_grid": ALPHA_GRID.tolist(),
        "alpha_tie_rule": (
            "maximize mean validation R2 over fixed independent targets; "
            "machine-equal ties choose largest alpha"
        ),
        "centering": "labelled-subset X/Y means; equivalent unpenalized intercept",
        "raw_coordinate_standardization": "none",
        "whitening_estimation": "all unlabelled canonical train features only",
        "validation_policy": "fixed complete validation; alpha selection only",
        "test_policy": "fixed complete test; evaluation after configuration fixed",
        "phase1_config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in config.__dict__.items()
        },
        "subset_manifest": subset_metadata,
        "time_of_day_sensitivity_subset_manifest": sensitivity_subset_metadata,
        "covariance_diagnostics": covariance_records,
        "transforms": transform_records,
        "trace_cov_over_dim_pairwise_ratios": _trace_ratios(covariance_records),
        "compute_log": compute_log,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            "results": {
                "path": "results.parquet",
                "sha256": sha256_file(results_path),
                "size_bytes": results_path.stat().st_size,
                "n_rows": result_sink.n_rows,
            },
            "failures": {
                "path": "failures.parquet",
                "sha256": sha256_file(failures_path),
                "size_bytes": failures_path.stat().st_size,
                "n_rows": failure_sink.n_rows,
            },
            "time_of_day_sensitivity": {
                "path": "time_of_day_sensitivity.parquet",
                "sha256": sha256_file(sensitivity_path),
                "size_bytes": sensitivity_path.stat().st_size,
                "n_rows": sensitivity_sink.n_rows,
            },
        },
    }
    atomic_write_json(destination / "metadata.json", metadata)
    return metadata

"""Final summaries and reports for Experiment 01 Phase III.

This module is intentionally downstream of the immutable selection manifest
and one-shot evaluation artifacts.  It never trains a reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .errors import ExperimentIntegrityError
from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .phase3 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    LOW_BUDGETS,
    PHASE1_RESULTS_SHA256,
    PHASE2_MANIFEST_SHA256,
    PRIMARY_BRANCHES,
    PRIMARY_READOUT,
    VALID_RANK,
    add_targetwise_normalized_recovery,
    assert_test_access_allowed,
    load_frozen_feature_transform,
    variance_components,
)
from .linear import SufficientStats, evaluate_stats, fit_alpha, transformed_design, tune_alpha
from .phase2 import load_feature_cache, project_stats, select_coordinate_stats
from .schema import load_input_bundle


BUDGET_ORDER = {
    label: index
    for index, label in enumerate(
        (
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
    )
}
MATCHED_SPECTRAL_ARMS = {
    "band_1_127": np.arange(0, 127, dtype=np.int64),
    "band_128_254": np.arange(127, 254, dtype=np.int64),
    "band_255_381": np.arange(254, 381, dtype=np.int64),
    "band_382_508": np.arange(381, 508, dtype=np.int64),
    "full_valid_rank": np.arange(0, 508, dtype=np.int64),
    "top_128": np.arange(0, 128, dtype=np.int64),
}


def _is_reduced_phase3(root: Path) -> bool:
    protocol = root / "protocol_frozen.json"
    if not protocol.is_file():
        return False
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    return payload.get("schema_name") == "thesis.experiment01.phase3_reduced.protocol"


def _phase1_budget_key(
    kind: str, days: float | None, seed: int, n_rows: int
) -> tuple[str, float | None, int, int]:
    value = None if days is None or not np.isfinite(float(days)) else float(days)
    return str(kind), value, int(seed), int(n_rows)


def load_frozen_ridge_rows(
    phase1_dir: str | Path, subset_manifest: str | Path
) -> pd.DataFrame:
    root = Path(phase1_dir)
    if sha256_file(root / "results.parquet") != PHASE1_RESULTS_SHA256:
        raise ExperimentIntegrityError("frozen Phase-I results hash mismatch")
    subset = json.loads(Path(subset_manifest).read_text())
    label_by_key = {
        _phase1_budget_key(
            record["budget_kind"],
            record.get("budget_days_per_stock"),
            record["subsample_seed"],
            record["n_rows"],
        ): record["budget_label"]
        for record in subset["subsets"]
    }
    columns = [
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
        "feature_view",
        "whiten_k_effective",
        "reader_family",
        "alpha_selected",
        "val_r2",
        "test_r2",
        "full_budget_test_r2",
        "normalized_recovery",
        "ceiling_eligible",
        "fit_status",
    ]
    frame = pd.read_parquet(root / "results.parquet", columns=columns)
    common = (
        frame["branch"].isin(PRIMARY_BRANCHES)
        & (frame["readout"] == PRIMARY_READOUT)
        & frame["target_independent"].astype(bool)
        & frame["alpha_selected"].astype(bool)
        & (frame["fit_status"] == "ok")
    )
    native = frame.loc[
        common
        & (frame["feature_view"] == "full_rank_raw")
        & (frame["reader_family"] == "ridge_raw_tuned_alpha")
    ].copy()
    native["transform"] = "native"
    white = frame.loc[
        common
        & (frame["feature_view"] == "full_rank_whiten_topk")
        & (frame["reader_family"] == "ridge_whiten_topk_tuned_alpha")
        & (frame["whiten_k_effective"] == VALID_RANK)
    ].copy()
    white["transform"] = "full_whitened"
    result = pd.concat([native, white], ignore_index=True)
    labels = []
    for row in result.itertuples(index=False):
        key = _phase1_budget_key(
            row.budget_kind,
            row.budget_days_per_stock,
            row.subsample_seed,
            row.n_rows,
        )
        if key not in label_by_key:
            raise ExperimentIntegrityError(f"cannot map frozen ridge budget: {key}")
        labels.append(label_by_key[key])
    result["budget_label"] = labels
    result["reader_class"] = "ridge"
    result = result.rename(columns={"val_r2": "validation_r2"})
    result["spectral_arm"] = "none"
    result["width"] = 0
    key = [
        "branch",
        "encoder_seed",
        "target_name",
        "transform",
        "budget_label",
        "subsample_seed",
    ]
    if result.duplicated(key).any():
        raise ExperimentIntegrityError("frozen ridge target rows are not unique")
    return result


def build_phase3_reader_gap(results: pd.DataFrame) -> pd.DataFrame:
    primary = results.loc[
        results["branch"].isin(PRIMARY_BRANCHES)
        & (results["width"] == 256)
        & results["transform"].isin(("native", "full_whitened"))
        & (results["spectral_arm"] == "none")
    ].copy()
    pair_key = [
        "job_family",
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "transform",
        "budget_label",
        "n_stock_days",
        "n_rows",
        "subsample_seed",
        "reader_seed",
        "width",
    ]
    columns = pair_key + [
        "test_r2",
        "full_budget_ceiling",
        "ceiling_eligible",
        "normalized_recovery",
    ]
    left = primary.loc[primary["branch"] == "supervised", columns].rename(
        columns={
            "test_r2": "supervised_test_r2",
            "full_budget_ceiling": "supervised_full_budget_ceiling",
            "ceiling_eligible": "supervised_ceiling_eligible",
            "normalized_recovery": "supervised_normalized_recovery",
        }
    )
    right = primary.loc[primary["branch"] == "jepa_horizon", columns].rename(
        columns={
            "test_r2": "jepa_horizon_test_r2",
            "full_budget_ceiling": "jepa_horizon_full_budget_ceiling",
            "ceiling_eligible": "jepa_horizon_ceiling_eligible",
            "normalized_recovery": "jepa_horizon_normalized_recovery",
        }
    )
    paired = left.merge(right, on=pair_key, how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ExperimentIntegrityError("Phase-III branch gap has unmatched rows")
    paired["both_ceiling_eligible"] = (
        paired["supervised_ceiling_eligible"]
        & paired["jepa_horizon_ceiling_eligible"]
    )
    paired["raw_r2_gap"] = (
        paired["supervised_test_r2"] - paired["jepa_horizon_test_r2"]
    )
    paired["normalized_recovery_gap"] = np.where(
        paired["both_ceiling_eligible"],
        paired["supervised_normalized_recovery"]
        - paired["jepa_horizon_normalized_recovery"],
        np.nan,
    )
    return paired


def _hierarchical_point(cell: pd.DataFrame, metric: str) -> float:
    reader_means = cell.groupby(
        ["encoder_seed", "subsample_seed"], observed=True
    )[metric].mean()
    subset_means = reader_means.groupby(level="encoder_seed").mean()
    return float(subset_means.mean())


def hierarchical_bootstrap_gap(
    paired: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Encoder -> subset -> reader paired bootstrap for block gap summaries."""

    group_columns = [
        "job_family",
        "readout",
        "target_block",
        "transform",
        "budget_label",
        "width",
    ]
    rows = []
    root_rng = np.random.default_rng(seed)
    for key, group in paired.groupby(
        group_columns, observed=True, dropna=False, sort=True
    ):
        valid = group.loc[
            group["both_ceiling_eligible"]
            & np.isfinite(group["normalized_recovery_gap"])
        ]
        if valid.empty:
            rows.append(
                {
                    **dict(zip(group_columns, key)),
                    "mean": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "n_encoders": 0,
                    "n_subsamples": 0,
                    "n_reader_seeds": 0,
                    "eligible_targets": 0,
                    "robust_delta_005": False,
                    "robust_delta_010": False,
                    "robust_delta_015": False,
                }
            )
            continue
        cell = (
            valid.groupby(
                ["encoder_seed", "subsample_seed", "reader_seed"],
                observed=True,
            )["normalized_recovery_gap"]
            .mean()
            .reset_index()
        )
        encoders = sorted(cell["encoder_seed"].unique())
        hierarchy: dict[int, dict[int, np.ndarray]] = {}
        for encoder in encoders:
            hierarchy[int(encoder)] = {
                int(subset_seed): subset["normalized_recovery_gap"].to_numpy(
                    dtype=float
                )
                for subset_seed, subset in cell.loc[
                    cell["encoder_seed"] == encoder
                ].groupby("subsample_seed", observed=True)
            }
        rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
        samples = np.empty(draws, dtype=np.float64)
        for draw in range(draws):
            encoder_values = []
            for sampled_encoder in rng.choice(encoders, size=len(encoders), replace=True):
                subsets = hierarchy[int(sampled_encoder)]
                subset_keys = tuple(subsets)
                subset_values = []
                for sampled_subset in rng.choice(
                    subset_keys, size=len(subset_keys), replace=True
                ):
                    reader_values = subsets[int(sampled_subset)]
                    subset_values.append(
                        float(
                            rng.choice(
                                reader_values,
                                size=len(reader_values),
                                replace=True,
                            ).mean()
                        )
                    )
                encoder_values.append(float(np.mean(subset_values)))
            samples[draw] = float(np.mean(encoder_values))
        point = _hierarchical_point(cell, "normalized_recovery_gap")
        lower, upper = np.quantile(samples, [0.025, 0.975])
        eligible_counts = (
            valid.groupby("target_name", observed=True)["both_ceiling_eligible"]
            .any()
            .sum()
        )
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "mean": point,
                "lower": float(lower),
                "upper": float(upper),
                "n_encoders": len(encoders),
                "n_subsamples": int(
                    cell[["encoder_seed", "subsample_seed"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_reader_seeds": int(cell["reader_seed"].nunique()),
                "eligible_targets": int(eligible_counts),
                "robust_delta_005": bool(lower > 0.0 and point >= 0.05),
                "robust_delta_010": bool(lower > 0.0 and point >= 0.10),
                "robust_delta_015": bool(lower > 0.0 and point >= 0.15),
            }
        )
    return pd.DataFrame(rows)


def build_ceiling_and_lift(
    mlp_results: pd.DataFrame, ridge_rows: pd.DataFrame
) -> pd.DataFrame:
    mlp = mlp_results.loc[
        (mlp_results["width"] == 256)
        & mlp_results["transform"].isin(("native", "full_whitened"))
        & (mlp_results["spectral_arm"] == "none")
    ].copy()
    ridge_key = [
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "transform",
        "budget_label",
        "subsample_seed",
    ]
    ridge = ridge_rows[
        ridge_key
        + [
            "test_r2",
            "full_budget_test_r2",
            "normalized_recovery",
            "ceiling_eligible",
        ]
    ].rename(
        columns={
            "test_r2": "ridge_test_r2",
            "full_budget_test_r2": "ridge_full_budget_ceiling",
            "normalized_recovery": "ridge_normalized_recovery",
            "ceiling_eligible": "ridge_ceiling_eligible",
        }
    )
    joined = mlp.merge(ridge, on=ridge_key, how="left", validate="many_to_one")
    if joined["ridge_test_r2"].isna().any():
        raise ExperimentIntegrityError("MLP/ridge join is incomplete")
    joined["nonlinear_lift_raw"] = joined["test_r2"] - joined["ridge_test_r2"]
    joined["nonlinear_lift_normalized"] = (
        joined["normalized_recovery"] - joined["ridge_normalized_recovery"]
    )
    joined["mlp_to_supervised_performance_ratio"] = np.nan
    full = joined["budget_label"] == "full_train"
    supervised = (
        joined.loc[full & (joined["branch"] == "supervised")]
        .groupby(
            ["encoder_seed", "target_block", "target_name", "transform"],
            observed=True,
        )["test_r2"]
        .mean()
        .rename("supervised_mlp_ceiling")
        .reset_index()
    )
    joined = joined.merge(
        supervised,
        on=["encoder_seed", "target_block", "target_name", "transform"],
        how="left",
        validate="many_to_one",
    )
    joined["mlp_to_supervised_performance_ratio"] = (
        joined["full_budget_ceiling"] / joined["supervised_mlp_ceiling"]
    )
    return joined


def build_reader_conditioning_interaction(
    mlp_gap_summary: pd.DataFrame,
    frozen_ridge_gap_summary: pd.DataFrame,
) -> pd.DataFrame:
    mlp = mlp_gap_summary.pivot_table(
        index=["readout", "target_block", "budget_label"],
        columns="transform",
        values="mean",
        aggfunc="first",
    ).reset_index()
    if not {"native", "full_whitened"}.issubset(mlp.columns):
        raise ExperimentIntegrityError("MLP conditioning interaction lacks transforms")
    mlp["mlp_native_minus_white_gap"] = mlp["native"] - mlp["full_whitened"]
    ridge = frozen_ridge_gap_summary.pivot_table(
        index=["readout", "target_block", "budget_label"],
        columns="transform",
        values="mean",
        aggfunc="first",
    ).reset_index()
    ridge["ridge_native_minus_white_gap"] = ridge["native"] - ridge["full_whitened"]
    out = mlp.merge(
        ridge[
            [
                "readout",
                "target_block",
                "budget_label",
                "ridge_native_minus_white_gap",
            ]
        ],
        on=["readout", "target_block", "budget_label"],
        how="left",
        validate="one_to_one",
    )
    out["reader_conditioning_interaction"] = (
        out["mlp_native_minus_white_gap"]
        - out["ridge_native_minus_white_gap"]
    )
    return out


def compute_matched_spectral_ridge(
    bundle_dir: str | Path,
    phase2_dir: str | Path,
    selection_manifest_path: str | Path,
) -> pd.DataFrame:
    """Use frozen Phase-II sufficient statistics for exact seed-0 matched bands."""

    selection_hash = assert_test_access_allowed(selection_manifest_path)
    phase2_root = Path(phase2_dir)
    if sha256_file(phase2_root / "manifest.json") != PHASE2_MANIFEST_SHA256:
        raise ExperimentIntegrityError("Phase-II manifest changed before spectral join")
    bundle = load_input_bundle(bundle_dir, verify_hashes=False, check_finite=False)
    rows = []
    budget_seed = {"b_1_4": 0, "b_4": 0, "full_train": -1}
    budget_days = {"b_1_4": 0.25, "b_4": 4.0, "full_train": None}
    for branch in PRIMARY_BRANCHES:
        for encoder_seed in (0, 1, 2):
            tag = f"{branch}_seed{encoder_seed}_{PRIMARY_READOUT}"
            cache_path = phase2_root / "cache" / f"{tag}.npz"
            cache_json = json.loads(
                (phase2_root / "cache" / f"{tag}.json").read_text()
            )
            statistics = load_feature_cache(cache_path)
            if statistics.numerical_rank != VALID_RANK:
                raise ExperimentIntegrityError("matched spectral cache rank differs")
            for budget_label, subsample_seed in budget_seed.items():
                train_all = statistics.budgets[budget_label]
                for block in ("directional", "timing"):
                    target_indices = np.asarray(
                        [
                            index
                            for index, target in enumerate(bundle.target_definitions)
                            if target.block == block and target.independent
                        ],
                        dtype=np.int64,
                    )
                    local = np.arange(len(target_indices), dtype=np.int64)
                    for arm, coordinates in MATCHED_SPECTRAL_ARMS.items():
                        train = select_coordinate_stats(
                            train_all, coordinates, target_indices
                        )
                        validation = select_coordinate_stats(
                            statistics.validation, coordinates, target_indices
                        )
                        test = select_coordinate_stats(
                            statistics.test, coordinates, target_indices
                        )
                        design = transformed_design(train)
                        selected = tune_alpha(
                            design,
                            None,
                            None,
                            local,
                            validation_stats=validation,
                        )
                        model = fit_alpha(design, selected.alpha)
                        validation_scores = evaluate_stats(model, validation)
                        test_scores = evaluate_stats(model, test)
                        for target_local, target_global in enumerate(target_indices):
                            if (
                                not validation_scores.valid[target_local]
                                or not test_scores.valid[target_local]
                            ):
                                raise ExperimentIntegrityError(
                                    "matched spectral ridge produced an invalid target"
                                )
                            rows.append(
                                {
                                    "branch": branch,
                                    "encoder_seed": encoder_seed,
                                    "readout": PRIMARY_READOUT,
                                    "target_block": block,
                                    "target_name": bundle.target_names[target_global],
                                    "budget_label": budget_label,
                                    "budget_days_per_stock": budget_days[budget_label],
                                    "subsample_seed": subsample_seed,
                                    "spectral_arm": arm,
                                    "subspace_dimension": len(coordinates),
                                    "alpha": model.alpha,
                                    "lambda_absolute": model.lambda_absolute,
                                    "validation_r2": float(
                                        validation_scores.values[target_local]
                                    ),
                                    "test_r2": float(test_scores.values[target_local]),
                                    "phase2_cache_sha256": sha256_file(cache_path),
                                    "phase2_transform_sha256": cache_json[
                                        "source_fingerprint"
                                    ]["transform_sha256"],
                                    "selection_manifest_sha256": selection_hash,
                                }
                            )
    result = pd.DataFrame(rows)
    key = [
        "branch",
        "encoder_seed",
        "target_name",
        "budget_label",
        "subsample_seed",
        "spectral_arm",
    ]
    if result.duplicated(key).any():
        raise ExperimentIntegrityError("matched spectral ridge rows are duplicated")
    return result


def _crossfit_predictor_r2(
    fit_stats: SufficientStats,
    evaluation_stats: SufficientStats,
    eigenvalues: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    fit = select_coordinate_stats(fit_stats, coordinates)
    evaluation = select_coordinate_stats(evaluation_stats, coordinates)
    values = np.asarray(eigenvalues, dtype=np.float64)[coordinates]
    if np.any(values <= 0.0):
        raise ExperimentIntegrityError("cross-fitted PCA predictor has invalid eigenvalue")
    beta = fit.cross / values[:, None]
    intercept = fit.y_mean - fit.x_mean @ beta
    covariance = evaluation.gram
    cross = evaluation.cross
    centered_sse = (
        evaluation.target_centered_ss
        - 2.0 * evaluation.n * np.einsum("dt,dt->t", beta, cross)
        + evaluation.n * np.einsum("dt,de,et->t", beta, covariance, beta)
    )
    mean_residual = evaluation.y_mean - intercept - evaluation.x_mean @ beta
    sse = centered_sse + evaluation.n * mean_residual**2
    return 1.0 - sse / np.maximum(evaluation.target_centered_ss, 1e-12)


def compute_crossfitted_spectral_control(
    bundle_dir: str | Path,
    phase1_dir: str | Path,
    phase2_dir: str | Path,
    *,
    chunk_rows: int = 65_536,
) -> pd.DataFrame:
    """Two-fold within-stock stock-day cross-fit using no validation/test rows."""

    bundle = load_input_bundle(bundle_dir, verify_hashes=False, check_finite=False)
    rows = bundle.rows["train"]
    day_table = (
        rows[["stock_id", "stock_day_id", "trading_date"]]
        .drop_duplicates()
        .sort_values(["stock_id", "trading_date", "stock_day_id"], kind="stable")
    )
    day_table["fold"] = (
        day_table.groupby("stock_id", observed=True).cumcount() % 2
    ).astype(np.int8)
    stock_values = rows["stock_id"].to_numpy(dtype=np.int64)
    day_values = rows["stock_day_id"].to_numpy(dtype=np.int64)
    fold_values = np.empty(len(rows), dtype=np.int8)
    for stock, stock_days in day_table.groupby("stock_id", observed=True):
        mask = stock_values == int(stock)
        mapping = dict(
            zip(
                stock_days["stock_day_id"].astype(int),
                stock_days["fold"].astype(int),
            )
        )
        mapped = pd.Series(day_values[mask], copy=False).map(mapping)
        if mapped.isna().any():
            raise ExperimentIntegrityError("cross-fit fold mapping missed stock-days")
        fold_values[mask] = mapped.to_numpy(dtype=np.int8)
    fold_positions = {
        fold: np.flatnonzero(fold_values == fold).astype(np.int64)
        for fold in (0, 1)
    }
    if any(len(value) == 0 for value in fold_positions.values()):
        raise ExperimentIntegrityError("cross-fitted stock-day fold is empty")
    y_train = bundle.load_targets("train")
    result_rows = []
    schedule = np.asarray([1, 2, 4, 8, 16, 32, 64, 128, 256, VALID_RANK])
    for branch in PRIMARY_BRANCHES:
        for encoder_seed in (0, 1, 2):
            feature = bundle.feature_set(branch, encoder_seed, PRIMARY_READOUT)
            x_train = bundle.load_features(feature, "train")
            fold_stats = {}
            for fold, positions in fold_positions.items():
                accumulator = SufficientStats.zeros(512, len(bundle.target_definitions))
                for start in range(0, len(positions), chunk_rows):
                    selected = positions[start : start + chunk_rows]
                    accumulator.add_rows(x_train[selected], y_train[selected])
                fold_stats[fold] = accumulator
            transform = load_frozen_feature_transform(
                phase1_dir,
                feature,
                kind="pca_coordinates",
                spectral_arm="full_valid_rank",
            )
            fold_pc = {
                fold: project_stats(stats, transform.basis)
                for fold, stats in fold_stats.items()
            }
            cache = load_feature_cache(
                Path(phase2_dir) / "cache" / f"{branch}_seed{encoder_seed}_{PRIMARY_READOUT}.npz"
            )
            eigenvalues = cache.eigenvalues[:VALID_RANK]
            for block in ("directional", "volatility", "timing"):
                target_indices = np.asarray(
                    [
                        index
                        for index, target in enumerate(bundle.target_definitions)
                        if target.block == block and target.independent
                    ],
                    dtype=np.int64,
                )
                fold_target = {
                    fold: select_coordinate_stats(
                        stats, np.arange(VALID_RANK), target_indices
                    )
                    for fold, stats in fold_pc.items()
                }
                for k in schedule:
                    coordinates = np.arange(int(k), dtype=np.int64)
                    a_to_b = _crossfit_predictor_r2(
                        fold_target[0], fold_target[1], eigenvalues, coordinates
                    )
                    b_to_a = _crossfit_predictor_r2(
                        fold_target[1], fold_target[0], eigenvalues, coordinates
                    )
                    for local, global_index in enumerate(target_indices):
                        result_rows.append(
                            {
                                "branch": branch,
                                "encoder_seed": encoder_seed,
                                "readout": PRIMARY_READOUT,
                                "target_block": block,
                                "target_name": bundle.target_names[global_index],
                                "k": int(k),
                                "fold_a_to_b_r2": float(a_to_b[local]),
                                "fold_b_to_a_r2": float(b_to_a[local]),
                                "crossfitted_r2": float(
                                    (a_to_b[local] + b_to_a[local]) * 0.5
                                ),
                                "fold_a_rows": len(fold_positions[0]),
                                "fold_b_rows": len(fold_positions[1]),
                                "fold_assignment": "chronological_within_stock_alternating_stock_days.v1",
                                "pca_fit_split": "all_unlabelled_train",
                                "labels_used": "opposite_train_fold_only",
                                "validation_used": False,
                                "test_used": False,
                                "transform_hash": transform.transform_hash,
                            }
                        )
    return pd.DataFrame(result_rows)


def _adjacent_robust(
    summary: pd.DataFrame,
    transform: str,
    budget_labels: Sequence[str] = LOW_BUDGETS,
) -> bool:
    values = summary.loc[
        (summary["transform"] == transform)
        & summary["budget_label"].isin(budget_labels)
    ].set_index("budget_label")["robust_delta_010"]
    flags = [bool(values.get(label, False)) for label in budget_labels]
    return any(left and right for left, right in zip(flags[:-1], flags[1:]))


def _encoder_gap_means(paired: pd.DataFrame, transform: str) -> pd.Series:
    values = paired.loc[
        (paired["target_block"] == "directional")
        & (paired["transform"] == transform)
        & paired["budget_label"].isin(LOW_BUDGETS)
        & paired["both_ceiling_eligible"]
    ]
    return values.groupby("encoder_seed", observed=True)[
        "normalized_recovery_gap"
    ].mean()


def assign_phase3_outcome(
    mlp_results: pd.DataFrame,
    paired_gap: pd.DataFrame,
    mlp_gap_summary: pd.DataFrame,
    frozen_ridge_gap_summary: pd.DataFrame,
) -> dict[str, Any]:
    primary_summary = mlp_gap_summary.loc[
        (mlp_gap_summary["readout"] == PRIMARY_READOUT)
        & (mlp_gap_summary["target_block"] == "directional")
        & (mlp_gap_summary["width"] == 256)
    ].copy()
    low = primary_summary.loc[primary_summary["budget_label"].isin(LOW_BUDGETS)]
    observed_low_budget_labels = tuple(
        label for label in LOW_BUDGETS if label in set(low["budget_label"])
    )
    if len(observed_low_budget_labels) < 2:
        raise ExperimentIntegrityError(
            "Phase-III outcome needs at least two adjacent observed low budgets"
        )
    native_gap = float(low.loc[low["transform"] == "native", "mean"].mean())
    white_gap = float(
        low.loc[low["transform"] == "full_whitened", "mean"].mean()
    )
    ridge_low = frozen_ridge_gap_summary.loc[
        (frozen_ridge_gap_summary["target_block"] == "directional")
        & frozen_ridge_gap_summary["budget_label"].isin(observed_low_budget_labels)
    ]
    ridge_native_gap = float(
        ridge_low.loc[ridge_low["transform"] == "native", "mean"].mean()
    )
    reader_attenuation = (
        1.0 - native_gap / ridge_native_gap if ridge_native_gap > 0 else np.nan
    )
    whitening_attenuation = (
        1.0 - white_gap / native_gap if native_gap > 0 else np.nan
    )
    native_robust_adjacent = _adjacent_robust(
        primary_summary, "native", observed_low_budget_labels
    )
    white_robust_adjacent = _adjacent_robust(
        primary_summary, "full_whitened", observed_low_budget_labels
    )
    ridge_robust_adjacent = _adjacent_robust(
        frozen_ridge_gap_summary.loc[
            frozen_ridge_gap_summary["target_block"] == "directional"
        ],
        "native",
        observed_low_budget_labels,
    )
    full = mlp_results.loc[
        (mlp_results["job_family"] == "primary_directional")
        & (mlp_results["budget_label"] == "full_train")
        & (mlp_results["width"] == 256)
    ]
    ceiling_counts = (
        full.loc[full["ceiling_eligible"]]
        .groupby(["branch", "transform"], observed=True)["target_name"]
        .nunique()
        .to_dict()
    )
    ceilings_meaningful = all(
        ceiling_counts.get((branch, transform), 0) >= 2
        for branch in PRIMARY_BRANCHES
        for transform in ("native", "full_whitened")
    )
    native_encoder = _encoder_gap_means(paired_gap, "native")
    white_encoder = _encoder_gap_means(paired_gap, "full_whitened")
    native_reader_cells = paired_gap.loc[
        (paired_gap["target_block"] == "directional")
        & (paired_gap["transform"] == "native")
        & paired_gap["budget_label"].isin(LOW_BUDGETS)
        & paired_gap["both_ceiling_eligible"]
    ]
    reader_variances = (
        native_reader_cells.groupby(
            ["encoder_seed", "subsample_seed"], observed=True
        )["normalized_recovery_gap"]
        .var(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    native_reader_sd = float(np.sqrt(np.mean(reader_variances)))
    reader_variance_dominates = native_reader_sd > abs(native_gap)
    encoder_inventory_complete = (
        set(native_encoder.index) == {0, 1, 2}
        and set(white_encoder.index) == {0, 1, 2}
    )
    r1_stable = encoder_inventory_complete and bool((native_encoder >= 0).all())
    r2_stable = encoder_inventory_complete and bool(
        ((native_encoder - white_encoder) >= 0).all()
    )
    r3_stable = encoder_inventory_complete and bool((white_encoder >= 0.10).all())

    requirements = {
        "analyzed_low_budget_labels": list(observed_low_budget_labels),
        "ridge_native_robust_adjacent": ridge_robust_adjacent,
        "native_mlp_robust_adjacent": native_robust_adjacent,
        "full_whitened_mlp_robust_adjacent": white_robust_adjacent,
        "native_low_budget_mean_gap": native_gap,
        "full_whitened_low_budget_mean_gap": white_gap,
        "ridge_native_low_budget_mean_gap": ridge_native_gap,
        "reader_attenuation": reader_attenuation,
        "whitening_attenuation_within_mlp": whitening_attenuation,
        "ceilings_meaningful": ceilings_meaningful,
        "native_encoder_gap_means": {
            str(key): float(value) for key, value in native_encoder.items()
        },
        "full_whitened_encoder_gap_means": {
            str(key): float(value) for key, value in white_encoder.items()
        },
        "native_reader_within_cell_sd": native_reader_sd,
        "reader_variance_dominates_native_gap": reader_variance_dominates,
    }
    if (
        ridge_robust_adjacent
        and np.isfinite(reader_attenuation)
        and reader_attenuation >= 0.50
        and not native_robust_adjacent
        and ceilings_meaningful
        and r1_stable
        and not reader_variance_dominates
    ):
        outcome = "R1"
    elif (
        native_robust_adjacent
        and np.isfinite(whitening_attenuation)
        and whitening_attenuation >= 0.50
        and not white_robust_adjacent
        and ceilings_meaningful
        and r2_stable
        and not reader_variance_dominates
    ):
        outcome = "R2"
    elif (
        native_robust_adjacent
        and white_robust_adjacent
        and white_gap >= 0.10
        and r3_stable
        and not reader_variance_dominates
    ):
        outcome = "R3"
    else:
        outcome = "R4"
    return {
        "outcome": outcome,
        "phase1_outcome_unchanged": "A1",
        "requirements": requirements,
        "stability": {
            "R1": r1_stable,
            "R2": r2_stable,
            "R3": r3_stable,
        },
    }


def _frozen_ridge_gap_summary(
    execution_root: Path, subset_manifest: Mapping[str, Any]
) -> pd.DataFrame:
    path = execution_root / "summary" / "gap_summary_delta_010.parquet"
    frame = pd.read_parquet(path)
    frame = frame.loc[
        (frame["readout"] == PRIMARY_READOUT)
        & frame["target_block"].isin(("directional", "volatility", "timing"))
        & (
            (frame["reader_family"] == "ridge_raw_tuned_alpha")
            | (
                (frame["reader_family"] == "ridge_whiten_topk_tuned_alpha")
                & (frame["whiten_k_effective"] == VALID_RANK)
            )
        )
    ].copy()
    frame["transform"] = np.where(
        frame["reader_family"] == "ridge_raw_tuned_alpha",
        "native",
        "full_whitened",
    )
    equivalents = {
        float(record["budget_stock_day_equivalents"]): record["budget_label"]
        for record in subset_manifest["subsets"]
    }
    frame["budget_label"] = [
        equivalents[float(value)] for value in frame["budget_stock_day_equivalents"]
    ]
    frame["robust_delta_010"] = frame["robust"].astype(bool)
    return frame


def summarize_phase3(
    phase3_dir: str | Path,
    phase1_dir: str | Path,
    *,
    bundle_dir: str | Path,
    phase2_dir: str | Path,
) -> dict[str, Any]:
    root = Path(phase3_dir)
    reduced = _is_reduced_phase3(root)
    selection_hash = assert_test_access_allowed(root / "selection_manifest.json")
    results_path = root / "phase3_results.parquet"
    results = pd.read_parquet(results_path)
    if set(results["selection_manifest_sha256"].astype(str)) != {selection_hash}:
        raise ExperimentIntegrityError("Phase-III result selection hash mismatch")
    if results["job_key"].isna().any() or not np.isfinite(results["test_r2"]).all():
        raise ExperimentIntegrityError("Phase-III result table is incomplete")
    subset_path = Path(phase1_dir) / "subset_manifest.json"
    subset_payload = json.loads(subset_path.read_text())
    ridge = load_frozen_ridge_rows(phase1_dir, subset_path)
    paired = build_phase3_reader_gap(results)
    gap_summary = hierarchical_bootstrap_gap(paired)
    frozen_gap = _frozen_ridge_gap_summary(Path(phase1_dir).parent, subset_payload)
    ceiling = build_ceiling_and_lift(results, ridge)
    interaction = build_reader_conditioning_interaction(gap_summary, frozen_gap)
    variance = variance_components(results, "normalized_recovery")
    matched_spectral_path = root / "matched_spectral_ridge.parquet"
    if matched_spectral_path.is_file():
        matched_spectral = pd.read_parquet(matched_spectral_path)
    else:
        matched_spectral = compute_matched_spectral_ridge(
            bundle_dir, phase2_dir, root / "selection_manifest.json"
        )
        atomic_write_parquet(matched_spectral, matched_spectral_path)
    spectral = results.loc[results["job_family"] == "spectral_diagnostic"].copy()
    spectral = spectral.merge(
        matched_spectral.rename(
            columns={
                "validation_r2": "matched_ridge_validation_r2",
                "test_r2": "matched_ridge_test_r2",
                "alpha": "matched_ridge_alpha",
                "lambda_absolute": "matched_ridge_lambda_absolute",
            }
        )[
            [
                "branch",
                "encoder_seed",
                "target_block",
                "target_name",
                "budget_label",
                "subsample_seed",
                "spectral_arm",
                "matched_ridge_validation_r2",
                "matched_ridge_test_r2",
                "matched_ridge_alpha",
                "matched_ridge_lambda_absolute",
                "phase2_cache_sha256",
                "phase2_transform_sha256",
            ]
        ],
        on=[
            "branch",
            "encoder_seed",
            "target_block",
            "target_name",
            "budget_label",
            "subsample_seed",
            "spectral_arm",
        ],
        how="left",
        validate="many_to_one",
    )
    spectral["nonlinear_lift_over_matched_ridge"] = (
        spectral["test_r2"] - spectral["matched_ridge_test_r2"]
    )
    crossfit_path = root / "crossfitted_spectral_control.parquet"
    if not reduced and not crossfit_path.is_file():
        crossfit = compute_crossfitted_spectral_control(
            bundle_dir, phase1_dir, phase2_dir
        )
        atomic_write_parquet(crossfit, crossfit_path)
    capacity = results.loc[results["job_family"] == "capacity_sensitivity"].copy()
    outcome = assign_phase3_outcome(results, paired, gap_summary, frozen_gap)

    tables = {
        "phase3_normalized_recovery.parquet": results,
        "phase3_reader_gap.parquet": pd.concat(
            [
                paired.assign(table_level="target_paired"),
                gap_summary.assign(table_level="block_summary"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "phase3_ceiling_and_lift.parquet": ceiling,
        "phase3_reader_conditioning_interaction.parquet": interaction,
        "phase3_variance_components.parquet": variance,
        "phase3_spectral_bands.parquet": spectral,
        "phase3_capacity_sensitivity.parquet": capacity,
    }
    hashes = {}
    for name, frame in tables.items():
        path = root / name
        atomic_write_parquet(frame, path)
        hashes[name] = sha256_file(path)
    summary = {
        "schema_name": "thesis.experiment01.phase3.summary",
        "schema_version": 1,
        "status": "complete",
        "protocol_variant": (
            "phase3_r_compute_feasible.v1" if reduced else "phase3_v1_definitive"
        ),
        "selection_manifest_sha256": selection_hash,
        "phase3_results_sha256": sha256_file(results_path),
        "phase1_results_sha256": PHASE1_RESULTS_SHA256,
        "phase2_manifest_sha256": PHASE2_MANIFEST_SHA256,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "outcome": outcome,
        "tables": hashes,
        "matched_spectral_ridge_sha256": sha256_file(matched_spectral_path),
        "crossfitted_spectral_control_sha256": (
            sha256_file(crossfit_path) if crossfit_path.is_file() else None
        ),
        "crossfitted_spectral_control_status": (
            "present" if crossfit_path.is_file() else "not_invoked_by_reduced_report"
        ),
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def _budget_x(frame: pd.DataFrame) -> np.ndarray:
    if "budget_stock_day_equivalents" in frame and frame[
        "budget_stock_day_equivalents"
    ].notna().any():
        return frame["budget_stock_day_equivalents"].to_numpy(dtype=float)
    return np.asarray(
        [BUDGET_ORDER.get(str(value), -1) + 1 for value in frame["budget_label"]],
        dtype=float,
    )


def generate_phase3_figures(phase3_dir: str | Path) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(phase3_dir)
    reduced = _is_reduced_phase3(root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = pd.read_parquet(root / "phase3_results.parquet")
    gap_table = pd.read_parquet(root / "phase3_reader_gap.parquet")
    gap = gap_table.loc[gap_table["table_level"] == "block_summary"].copy()
    ceiling = pd.read_parquet(root / "phase3_ceiling_and_lift.parquet")
    variance = pd.read_parquet(root / "phase3_variance_components.parquet")
    spectral = pd.read_parquet(root / "phase3_spectral_bands.parquet")
    capacity = pd.read_parquet(root / "phase3_capacity_sensitivity.parquet")
    whitening = pd.read_parquet(root / "phase1_branch_whitening_effects.parquet")
    records = []

    def save(name: str, figure) -> None:
        path = figures / name
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        records.append(
            {"path": f"figures/{name}", "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )

    primary = results.loc[
        (results["job_family"] == "primary_directional")
        & (results["width"] == 256)
    ]
    for normalized, name, ylabel in (
        (False, "01_directional_raw_mlp_learning_curves.png", "test R2"),
        (True, "02_directional_normalized_mlp_recovery.png", "normalized recovery"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        metric = "normalized_recovery" if normalized else "test_r2"
        for axis, transform in zip(axes, ("native", "full_whitened")):
            subset = primary.loc[primary["transform"] == transform]
            cell = (
                subset.groupby(
                    ["branch", "encoder_seed", "budget_label"], observed=True
                )[metric]
                .mean()
                .reset_index()
            )
            for (branch, encoder), curve in cell.groupby(
                ["branch", "encoder_seed"], observed=True
            ):
                curve = curve.assign(
                    order=curve["budget_label"].map(BUDGET_ORDER)
                ).sort_values("order")
                axis.plot(
                    curve["order"],
                    curve[metric],
                    marker="o",
                    alpha=0.75,
                    label=f"{branch}/seed{encoder}",
                )
            axis.set_title(transform)
            axis.set_xlabel("preregistered budget index")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(fontsize=7, ncol=2)
        save(name, fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    directional_gap = gap.loc[
        (gap["target_block"] == "directional")
        & (gap["width"] == 256)
    ].copy()
    for transform, curve in directional_gap.groupby("transform", observed=True):
        curve = curve.assign(order=curve["budget_label"].map(BUDGET_ORDER)).sort_values("order")
        axis.plot(curve["order"], curve["mean"], marker="o", label=f"MLP/{transform}")
        axis.fill_between(curve["order"], curve["lower"], curve["upper"], alpha=0.18)
    ridge = pd.read_parquet(
        root.parent / "summary" / "gap_summary_delta_010.parquet"
    )
    ridge = ridge.loc[
        (ridge["readout"] == PRIMARY_READOUT)
        & (ridge["target_block"] == "directional")
        & (ridge["reader_family"] == "ridge_raw_tuned_alpha")
        & (ridge["budget_stock_day_equivalents"] >= 1.75)
    ].sort_values("budget_stock_day_equivalents")
    axis.plot(
        np.arange(len(ridge)), ridge["mean"], color="black", linestyle="--", marker="s", label="frozen ridge/native"
    )
    axis.axhline(0.10, color="grey", linewidth=1, linestyle=":")
    axis.set(xlabel="preregistered budget index", ylabel="supervised - horizon-JEPA normalized gap")
    axis.grid(alpha=0.25)
    axis.legend()
    save("03_linear_vs_mlp_gap.png", fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for transform, curve in directional_gap.groupby("transform", observed=True):
        curve = curve.assign(order=curve["budget_label"].map(BUDGET_ORDER)).sort_values("order")
        axis.errorbar(
            curve["order"], curve["mean"],
            yerr=[curve["mean"] - curve["lower"], curve["upper"] - curve["mean"]],
            marker="o", capsize=2, label=transform,
        )
    axis.axhline(0.10, color="grey", linestyle=":")
    axis.set(xlabel="budget index", ylabel="normalized gap", title="Native vs full-whitened MLP")
    axis.legend(); axis.grid(alpha=0.25)
    save("04_native_vs_whitened_mlp_gap.png", fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    low = directional_gap.loc[directional_gap["budget_label"].isin(LOW_BUDGETS)]
    mlp_values = low.groupby("transform", observed=True)["mean"].mean()
    if reduced:
        requirement_values = json.loads((root / "summary.json").read_text())[
            "outcome"
        ]["requirements"]
        common_low_labels = requirement_values["analyzed_low_budget_labels"]
        subset_payload = json.loads(
            (root.parent / "phase1" / "subset_manifest.json").read_text()
        )
        frozen_gap = _frozen_ridge_gap_summary(root.parent, subset_payload)
        frozen_common = frozen_gap.loc[
            (frozen_gap["target_block"] == "directional")
            & frozen_gap["budget_label"].isin(common_low_labels)
        ]
        ridge_native = float(
            frozen_common.loc[frozen_common["transform"] == "native", "mean"].mean()
        )
        ridge_white_value = float(
            frozen_common.loc[
                frozen_common["transform"] == "full_whitened", "mean"
            ].mean()
        )
    else:
        ridge_native = float(ridge.loc[ridge["budget_stock_day_equivalents"] <= 28, "mean"].head(5).mean())
    ridge_all = pd.read_parquet(root.parent / "summary" / "gap_summary_delta_010.parquet")
    ridge_white = ridge_all.loc[
        (ridge_all["readout"] == PRIMARY_READOUT)
        & (ridge_all["target_block"] == "directional")
        & (ridge_all["reader_family"] == "ridge_whiten_topk_tuned_alpha")
        & (ridge_all["whiten_k_effective"] == VALID_RANK)
        & ridge_all["budget_stock_day_equivalents"].between(1.75, 28)
    ]
    if not reduced:
        ridge_white_value = float(ridge_white["mean"].mean())
    values = [ridge_native, ridge_white_value, mlp_values.get("native", np.nan), mlp_values.get("full_whitened", np.nan)]
    axis.bar([0, 1, 2, 3], values, color=["#444444", "#999999", "#1f77b4", "#ff7f0e"])
    axis.set_xticks([0, 1, 2, 3], ["ridge\nnative", "ridge\nwhite", "MLP\nnative", "MLP\nwhite"])
    axis.set_ylabel("low-budget directional normalized gap")
    save("05_reader_conditioning_2x2.png", fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    full = ceiling.loc[ceiling["budget_label"] == "full_train"]
    full.groupby(["branch", "transform"], observed=True)["test_r2"].mean().unstack().plot.bar(ax=axes[0])
    full.groupby(["branch", "transform"], observed=True)["nonlinear_lift_raw"].mean().unstack().plot.bar(ax=axes[1])
    axes[0].set_title("MLP operational ceiling"); axes[0].set_ylabel("test R2")
    axes[1].set_title("nonlinear lift over frozen ridge"); axes[1].set_ylabel("delta R2")
    save("06_full_budget_ceiling_and_nonlinear_lift.png", fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    specificity = gap.loc[(gap["width"] == 256) & (gap["transform"] == "native")]
    for block, curve in specificity.groupby("target_block", observed=True):
        curve = curve.assign(order=curve["budget_label"].map(BUDGET_ORDER)).sort_values("order")
        axis.plot(curve["order"], curve["mean"], marker="o", label=block)
    axis.axhline(0.10, color="grey", linestyle=":")
    axis.set(xlabel="budget index", ylabel="normalized gap"); axis.legend(); axis.grid(alpha=0.25)
    save("07_target_specificity_reader_gap.png", fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    var = variance.loc[(variance["target_block"] == "directional") & (variance["width"] == 256)]
    var.groupby(["branch", "transform"], observed=True)[
        "sd_reader_within_subset_encoder"
    ].mean().unstack().plot.bar(ax=axis)
    axis.set_ylabel("reader-seed SD")
    save("08_reader_seed_variance.png", fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    native = primary.loc[primary["transform"] == "native"]
    cell = native.groupby(["branch", "encoder_seed", "budget_label"], observed=True)["normalized_recovery"].mean().reset_index()
    for encoder, axis in enumerate(axes):
        for branch, curve in cell.loc[cell["encoder_seed"] == encoder].groupby("branch", observed=True):
            curve = curve.assign(order=curve["budget_label"].map(BUDGET_ORDER)).sort_values("order")
            axis.plot(curve["order"], curve["normalized_recovery"], marker="o", label=branch)
        axis.set_title(f"encoder seed {encoder}"); axis.grid(alpha=0.25)
    axes[0].set_ylabel("normalized recovery"); axes[-1].legend()
    save("09_encoder_specific_mlp_curves.png", fig)

    spectral_mean = spectral.groupby(
        ["target_block", "branch", "spectral_arm"], observed=True
    )["test_r2"].mean().reset_index()
    if reduced:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        table = spectral_mean.pivot(
            index="spectral_arm", columns="branch", values="test_r2"
        )
        table.plot.bar(ax=axis)
        axis.set_title("horizon-JEPA directional: head, deep, full rank")
        axis.set_ylabel("test R2")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
        for axis, block in zip(axes, ("directional", "timing")):
            table = spectral_mean.loc[spectral_mean["target_block"] == block].pivot(
                index="spectral_arm", columns="branch", values="test_r2"
            )
            table.plot.bar(ax=axis); axis.set_title(block); axis.set_ylabel("test R2")
    save("10_equal_dimensional_spectral_bands.png", fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    comparison_arms = (
        ("band_1_127", "band_382_508", "full_valid_rank")
        if reduced
        else ("full_valid_rank", "top_128")
    )
    compare = spectral.loc[spectral["spectral_arm"].isin(comparison_arms)]
    compare.groupby(["branch", "spectral_arm"], observed=True)["test_r2"].mean().unstack().plot.bar(ax=axis)
    axis.set_ylabel("test R2")
    save(
        "11_head_deep_full_mlp.png" if reduced else "11_full_vs_top128_mlp.png",
        fig,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    whit = whitening.loc[
        (whitening["readout"] == PRIMARY_READOUT)
        & (whitening["target_block"] == "directional")
    ]
    whit.groupby(["branch", "whitening_depth"], observed=True)[
        "delta_normalized_recovery_vs_k0"
    ].mean().unstack(0).plot(ax=axis, marker="o")
    axis.set(xlabel="whitening depth k", ylabel="delta recovery vs k=0")
    axis.grid(alpha=0.25)
    save("12_branch_specific_whitening_effects.png", fig)

    if not reduced:
        fig, axis = plt.subplots(figsize=(8, 5))
        capacity.groupby(["branch", "width"], observed=True)["test_r2"].mean().unstack().plot.bar(ax=axis)
        axis.set_ylabel("test R2")
        save("13_capacity_sensitivity.png", fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    eligibility = (
        results.groupby(["target_block", "branch", "transform"], observed=True)["ceiling_eligible"]
        .mean().unstack(["branch", "transform"])
    )
    image = axis.imshow(eligibility.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_yticks(np.arange(len(eligibility.index)), eligibility.index)
    axis.set_xticks(np.arange(len(eligibility.columns)), ["/".join(map(str, value)) for value in eligibility.columns], rotation=45, ha="right")
    fig.colorbar(image, ax=axis, label="eligible target fraction")
    save("14_ceiling_eligibility_map.png", fig)
    return records


def _format_num(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_phase3_reports(phase3_dir: str | Path) -> dict[str, str]:
    root = Path(phase3_dir)
    reduced = _is_reduced_phase3(root)
    summary = json.loads((root / "summary.json").read_text())
    outcome = summary["outcome"]
    requirements = outcome["requirements"]
    result = outcome["outcome"]
    outcome_text = {
        "R1": "predominantly reader-class-mediated accessibility",
        "R2": "conditioning-mediated accessibility persists for MLP",
        "R3": "persistent difficulty beyond linearity and second-order conditioning",
        "R4": "mixed or indeterminate reader result",
    }[result]
    gap_table = pd.read_parquet(root / "phase3_reader_gap.parquet")
    gap_summary = gap_table.loc[gap_table["table_level"] == "block_summary"]
    specificity = (
        gap_summary.loc[
            gap_summary["budget_label"].isin(LOW_BUDGETS)
            & (gap_summary["width"] == 256)
        ]
        .groupby(["target_block", "transform"], observed=True)["mean"]
        .mean()
        .reset_index()
    )
    specificity_lines = "\n".join(
        f"- `{row.target_block}/{row.transform}`: mean normalized gap {_format_num(row.mean)}"
        for row in specificity.itertuples(index=False)
    )
    ceiling_table = pd.read_parquet(root / "phase3_ceiling_and_lift.parquet")
    ceiling_summary = (
        ceiling_table.loc[ceiling_table["budget_label"] == "full_train"]
        .groupby(["branch", "target_block", "transform"], observed=True)
        .agg(
            mlp_test_r2=("test_r2", "mean"),
            nonlinear_lift=("nonlinear_lift_raw", "mean"),
            mlp_to_supervised_ratio=("mlp_to_supervised_performance_ratio", "mean"),
        )
        .reset_index()
    )
    ceiling_lines = "\n".join(
        f"- `{row.branch}/{row.target_block}/{row.transform}`: ceiling {_format_num(row.mlp_test_r2)}, "
        f"lift {_format_num(row.nonlinear_lift)}, supervised ratio {_format_num(row.mlp_to_supervised_ratio)}"
        for row in ceiling_summary.itertuples(index=False)
    )
    spectral_table = pd.read_parquet(root / "phase3_spectral_bands.parquet")
    spectral_full = (
        spectral_table.loc[spectral_table["budget_label"] == "full_train"]
        .groupby(["branch", "target_block", "spectral_arm"], observed=True)["test_r2"]
        .mean()
        .reset_index()
    )
    spectral_lines = []
    band_names = [
        "band_1_127",
        "band_128_254",
        "band_255_381",
        "band_382_508",
    ]
    for (branch, block), cell in spectral_full.groupby(
        ["branch", "target_block"], observed=True
    ):
        by_arm = cell.set_index("spectral_arm")["test_r2"]
        if reduced:
            spectral_lines.append(
                f"- `{branch}/{block}`: head 1:127 "
                f"{_format_num(by_arm.get('band_1_127'))}, deep 382:508 "
                f"{_format_num(by_arm.get('band_382_508'))}, full-rank "
                f"{_format_num(by_arm.get('full_valid_rank'))}"
            )
            continue
        available_bands = by_arm.reindex(band_names).dropna()
        best_band = str(available_bands.idxmax()) if len(available_bands) else "NA"
        spectral_lines.append(
            f"- `{branch}/{block}`: full-rank {_format_num(by_arm.get('full_valid_rank'))}, "
            f"top-128 {_format_num(by_arm.get('top_128'))}, best equal band `{best_band}` "
            f"at {_format_num(available_bands.max() if len(available_bands) else np.nan)}"
        )
    spectral_text = "\n".join(spectral_lines)
    capacity = pd.read_parquet(root / "phase3_capacity_sensitivity.parquet")
    capacity_lines = "\n".join(
        f"- `{row.branch}/width{int(row.width)}/{row.budget_label}`: test R2 {_format_num(row.test_r2)}"
        for row in (
            capacity.groupby(["branch", "width", "budget_label"], observed=True)["test_r2"]
            .mean()
            .reset_index()
            .itertuples(index=False)
        )
    )
    phase_title = "Experiment 01 Phase III-R" if reduced else "Experiment 01 Phase III"
    amendment_section = (
        """## Compute-feasible preregistered amendment

Phase III v1 was terminated before selection freeze and before any production
test access because its 21,456-model inventory was computationally
disproportionate. Phase III-R was frozen before test with 1,296 models: 504
primary, 576 specificity-control and 216 focused spectral fits. It preserves
all scientific semantics and outcome thresholds while reducing replication to
three paired encoder, subset and reader seeds where applicable. The capacity
sweep and redundant spectral arms are explicitly outside Phase III-R.
"""
        if reduced
        else ""
    )
    spectral_scope_text = (
        """The focused nonlinear spectral contrast is restricted to
`jepa_horizon` directional targets: head PCs 1:127, deep PCs 382:508 and the
full valid rank. Detailed intermediate-band, supervised and timing spectral
localization remains the frozen Phase-II result; Phase III-R does not repeat
it. The MLP does not “recover predictive mass.”"""
        if reduced
        else """The four equal 127-dimensional PCA bands are disjoint and cover PCs 1:508 exactly. `full_valid_rank` and `top_128` are reported separately. MLP performance is compared with matched frozen-Phase-II sufficient-statistic ridge readers where the exact seed-0 subset exists. These results may be consistent or inconsistent with Phase-II spectral localization; the MLP does not “recover predictive mass.”"""
    )
    crossfit_text = (
        "The cross-fitted spectral control was not invoked by the reduced report; the frozen Phase-II controls remain unchanged."
        if reduced
        else "The train-only two-fold stock-day cross-fitted spectral control is serialized separately and never uses validation or test rows in its construction."
    )
    capacity_section = (
        """## Capacity sensitivity

Capacity sensitivity was removed by the frozen compute-feasible amendment. No
architecture sweep was used to select or redefine the width-256 primary
reader."""
        if reduced
        else f"""## Capacity sensitivity

Widths 128 and 512 are descriptive sensitivity checks at the preregistered minimum and full budgets; they do not select or redefine the width-256 primary reader.

{capacity_lines}"""
    )
    report = f"""# {phase_title} — final report

{amendment_section}

## Preregistered outcome

The directional `last_concat512` primary outcome is **{result}: {outcome_text}**. Phase-I technical outcome **A1 remains frozen and unchanged**. Phase III changes only the reader-relative diagnosis, not the Phase-I result.

The frozen native-ridge low-budget normalized gap is {_format_num(requirements['ridge_native_low_budget_mean_gap'])}. The native-MLP gap is {_format_num(requirements['native_low_budget_mean_gap'])}, giving reader attenuation {_format_num(requirements['reader_attenuation'])}. The full-whitened MLP gap is {_format_num(requirements['full_whitened_low_budget_mean_gap'])}, giving within-MLP whitening attenuation {_format_num(requirements['whitening_attenuation_within_mlp'])}.

## Frozen Phase-I/II facts

- Phase-I `A1` and all Phase-I thresholds/results are unchanged.
- Phase II localized directional signal deeply along the covariance spectrum; predictive mass remains a linear covariance diagnostic.
- The production bundle, Phase-I subsets, Phase-I transforms, Phase-II caches and canonical checkpoints passed their hash gates.
- The historical MLP gate reproduced horizon-JEPA and supervised within absolute tolerance 0.015; that historical reader used coordinate-wise standardization and is not the Phase-III primary reader.

## New Phase-III reader result

The primary reader is exactly `Linear(d,256)-GELU-Dropout(0.10)-Linear(256,T)`, with no coordinate-wise native input standardization, BatchNorm or LayerNorm. Weight decay was selected on the fixed validation split. The selection manifest was frozen and hashed before one-shot test inference.

Encoder-specific native directional gaps are `{json.dumps(requirements['native_encoder_gap_means'], sort_keys=True)}`. Encoder-specific full-whitened gaps are `{json.dumps(requirements['full_whitened_encoder_gap_means'], sort_keys=True)}`. Meaningful-ceiling status is `{requirements['ceilings_meaningful']}`.

## Conditioning and reader decomposition

Phase III separates: (1) the operational full-budget ceiling of each reader, (2) finite-sample recovery relative to its own target-wise ceiling, (3) dependence on the invertible train-only whitening transform, and (4) dependence on enlarging the reader class from frozen ridge to the preregistered MLP. The reader-by-conditioning interaction is descriptive and does not change {result}.

## Target specificity

Directional, volatility and timing results are reported separately. They are never pooled. Volatility and timing are specificity controls; the preregistered outcome is directional only.

{specificity_lines}

## Spectral diagnostics

{spectral_scope_text}

{spectral_text}

{crossfit_text}

## Secondary ceiling statement

Full-budget MLP ceiling gaps, nonlinear lift over frozen ridge, MLP-to-supervised ratios and target-specific ceiling eligibility are reported in `phase3_ceiling_and_lift.parquet`. They are operational reader results, not Bayes-content estimates.

{ceiling_lines}

{capacity_section}

## Limitations and prohibited interpretations

Equal MLP performance would not prove equal information, and a persistent MLP gap would not prove information loss. Full whitening is a post-hoc train-only invertible coordinate transform, not a training-time encoder intervention. No claim is made that VICReg/SIGReg must reproduce it, that top-128 failure proves tail causality, or that these results generalize beyond this domain.
"""
    narrative = f"""# {phase_title} — narrative summary

{phase_title} assigns **{result} ({outcome_text})** for the directional `last_concat512` comparison. The low-budget gap changes from {_format_num(requirements['ridge_native_low_budget_mean_gap'])} for the frozen native ridge to {_format_num(requirements['native_low_budget_mean_gap'])} for the native MLP and {_format_num(requirements['full_whitened_low_budget_mean_gap'])} after full train-only whitening. Phase-I **A1 remains unchanged**.

The empirical decomposition is reader-specific: full-budget operational ceiling, finite-sample accessibility relative to that ceiling, conditioning dependence under an invertible train-only transform, reader dependence, and spectral dependence. Volatility and timing remain separate controls. Spectral MLP performance is discussed only as consistent or inconsistent with Phase-II localization, never as recovery of predictive mass.
"""
    if reduced:
        changelog = """# Experiment 01 Phase III-R — changelog

- Terminated the 21,456-model Phase-III v1 grid before selection freeze and before test access because observed runtime made it infeasible.
- Froze a compute-feasible 1,296-model amendment before test access.
- Retained two adjacent primary low budgets, full-budget ceilings, both branches, three encoder seeds, three paired subset seeds and three reader seeds.
- Retained volatility and timing as separate low/full specificity controls.
- Focused the nonlinear spectral diagnostic on horizon-JEPA directional head, deep and full-rank coordinates.
- Removed capacity sensitivity and redundant nonlinear spectral arms; frozen Phase II remains the detailed spectral analysis.
- Reused only exact v1 selection cells after fingerprint and artifact-hash verification.
- Preserved the MLP, optimizer, stopping schedule, weight-decay grid, splits, targets, metrics, thresholds, R1--R4 rules and Phase-I A1 outcome.
"""
    else:
        changelog = """# Experiment 01 Phase III — changelog

- Added a pre-implementation audit of the exact historical post-P0 MLP.
- Verified the complete production bundle, Phase-I/II artifacts, checkpoints, shards and 78 exact subset identities.
- Added the frozen branch-specific Phase-I whitening-effect decomposition.
- Added the one-hidden-layer primary MLP with native centering only and explicit full-whitening/PCA arms.
- Added labelled-subset-only target standardization, validation-only weight-decay selection, deterministic tie rule and adaptive reader seeds.
- Added a fail-closed selection-manifest/test boundary and one-shot restart-safe test inference claims.
- Added synthetic nonlinear and anisotropic-conditioning gates, historical stochastic reproduction, full-budget linear parity and PCA-band identity gates.
- Added exact selection/evaluation job inventories, checkpoint hashing, failure records and runtime/RAM/VRAM accounting.
- Added target-wise ceilings/recovery, hierarchical variance, paired gaps, reader-conditioning interaction, spectral and capacity summaries.
- Added 14 required figures and the final R1/R2/R3/R4 classifier.
- Did not modify Phase I, Phase II, their reports, thresholds or technical outcome A1; did not train encoders or start a later empirical phase.
"""
    paths = {
        "REPORT_EXPERIMENT_01_PHASE3.md": report,
        "SUMMARY_NARRATIVE_EXPERIMENT_01_PHASE3.md": narrative,
        "CHANGELOG_PHASE3.md": changelog,
    }
    hashes = {}
    for name, content in paths.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        hashes[name] = sha256_file(path)
    return hashes


def finalize_phase3(
    phase3_dir: str | Path,
    phase1_dir: str | Path,
    *,
    bundle_dir: str | Path,
    phase2_dir: str | Path,
) -> dict[str, Any]:
    root = Path(phase3_dir)
    reduced = _is_reduced_phase3(root)
    summary = summarize_phase3(
        root,
        phase1_dir,
        bundle_dir=bundle_dir,
        phase2_dir=phase2_dir,
    )
    figures = generate_phase3_figures(root)
    reports = write_phase3_reports(root)
    compute_parts = {}
    for name in (
        "compute_benchmark.json",
        "selection_compute_log.json",
        "evaluation_compute_log.json",
        "historical_mlp_gate.json",
    ):
        path = root / name
        if path.is_file():
            compute_parts[name] = json.loads(path.read_text())
    output_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    compute_log = {
        "schema_name": "thesis.experiment01.phase3.compute_log",
        "schema_version": 1,
        "parts": compute_parts,
        "output_storage_bytes_before_manifest": output_bytes,
        "trained_models": int(
            compute_parts.get("selection_compute_log.json", {}).get(
                "models_trained_or_verified", 0
            )
            + compute_parts.get("evaluation_compute_log.json", {}).get(
                "evaluation_models", 0
            )
        ),
        "failed_cells": int(
            compute_parts.get("selection_compute_log.json", {}).get("failures", 0)
            + compute_parts.get("evaluation_compute_log.json", {}).get("failures", 0)
        ),
    }
    atomic_write_json(root / "compute_log.json", compute_log)
    metadata = {
        "schema_name": "thesis.experiment01.phase3.metadata",
        "schema_version": 1,
        "status": "complete",
        "protocol_variant": (
            "phase3_r_compute_feasible.v1" if reduced else "phase3_v1_definitive"
        ),
        "phase1_outcome_unchanged": "A1",
        "phase3_outcome": summary["outcome"]["outcome"],
        "selection_manifest_sha256": summary["selection_manifest_sha256"],
        "phase1_results_sha256": PHASE1_RESULTS_SHA256,
        "phase2_manifest_sha256": PHASE2_MANIFEST_SHA256,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "figures": figures,
        "reports": reports,
        "compute_log_sha256": sha256_file(root / "compute_log.json"),
    }
    atomic_write_json(root / "metadata.json", metadata)
    excluded = {"phase3_manifest.json"}
    artifacts = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    ]
    source_root = Path(__file__).resolve().parents[1]
    source_paths = (
        source_root / "experiment01/phase3.py",
        source_root / "experiment01/phase3_reporting.py",
        source_root / "scripts/experiment01/run_experiment_01_phase3.py",
        source_root / "tests/test_experiment01_phase3.py",
    )
    if reduced:
        source_paths = source_paths + (
            source_root / "experiment01/phase3_reduced.py",
            source_root
            / "scripts/experiment01/run_experiment_01_phase3_reduced.py",
            source_root / "tests/test_experiment01_phase3_reduced.py",
        )
    manifest = {
        "schema_name": "thesis.experiment01.phase3.manifest",
        "schema_version": 1,
        "status": "complete",
        "protocol_variant": (
            "phase3_r_compute_feasible.v1" if reduced else "phase3_v1_definitive"
        ),
        "phase1_outcome_unchanged": "A1",
        "phase3_outcome": summary["outcome"]["outcome"],
        "artifacts": artifacts,
        "source_files": [
            {
                "path": str(path.relative_to(source_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in source_paths
        ],
    }
    atomic_write_json(root / "phase3_manifest.json", manifest)
    manifest_digest = sha256_file(root / "phase3_manifest.json")
    (root / "phase3_manifest.sha256").write_text(
        f"{manifest_digest}  phase3_manifest.json\n", encoding="utf-8"
    )
    return {
        "status": "complete",
        "outcome": summary["outcome"]["outcome"],
        "manifest_sha256": manifest_digest,
        "artifacts": len(artifacts),
        "output_storage_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
    }

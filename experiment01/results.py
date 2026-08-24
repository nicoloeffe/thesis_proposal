"""Result-schema validation, ceiling normalization and uncertainty summaries."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .constants import (
    ALPHA_GRID,
    CEILING_THRESHOLD,
    EXPERIMENTAL_KEY_COLUMNS,
    LOW_BUDGETS,
    PRIMARY_GAP_DELTA,
    RESULT_COLUMNS,
)
from .errors import ExperimentIntegrityError


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def validate_result_schema(frame: pd.DataFrame, *, finalized: bool) -> None:
    missing = set(RESULT_COLUMNS) - set(frame.columns)
    if missing:
        raise ExperimentIntegrityError(
            f"result table is missing columns {sorted(missing)}"
        )
    duplicates = frame.duplicated(list(EXPERIMENTAL_KEY_COLUMNS), keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, list(EXPERIMENTAL_KEY_COLUMNS)].iloc[0]
        raise ExperimentIntegrityError(
            "result table contains duplicate experimental keys: "
            + repr(example.to_dict())
        )
    for alpha in frame["alpha"].dropna().astype(float).unique():
        if not np.any(np.isclose(alpha, ALPHA_GRID, rtol=0.0, atol=0.0)):
            raise ExperimentIntegrityError(
                f"result alpha {alpha!r} is outside the declared grid"
            )
    full = frame[frame["budget_kind"] == "full_train"]
    if not full.empty:
        keys = full.apply(_normalization_key, axis=1)
        if keys.duplicated().any():
            raise ExperimentIntegrityError(
                "full_train is evaluated more than once per deterministic configuration"
            )
    if finalized:
        successful = frame["fit_status"].eq("ok")
        if frame.loc[successful, "full_budget_test_r2"].isna().any():
            raise ExperimentIntegrityError(
                "successful rows are missing a full-budget ceiling"
            )


def _normalization_key(row: pd.Series) -> tuple[object, ...]:
    values: list[object] = [
        row["branch"],
        row["encoder_seed"],
        row["readout"],
        row["target_block"],
        row["target_name"],
        bool(row["target_independent"]),
        row["feature_view"],
        _nan_key(row["whiten_k_requested"]),
        _nan_key(row["whiten_k_effective"]),
        _nan_key(row["pca_fraction"]),
        _nan_key(row["subspace_seed"]),
        row["reader_family"],
    ]
    if "common_alpha" in str(row["reader_family"]):
        values.append(float(row["alpha"]))
    return tuple(values)


def _nan_key(value: object) -> object:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def attach_operational_ceilings(
    frame: pd.DataFrame, threshold: float = CEILING_THRESHOLD
) -> pd.DataFrame:
    """Attach exact-configuration full-train ceilings and target eligibility."""
    result = frame.copy()
    if "full_budget_test_r2" not in result:
        result["full_budget_test_r2"] = np.nan
    if "normalized_recovery" not in result:
        result["normalized_recovery"] = np.nan
    if "ceiling_eligible" not in result:
        result["ceiling_eligible"] = False
    successful_full = result[
        result["budget_kind"].eq("full_train") & result["fit_status"].eq("ok")
    ]
    ceilings: dict[tuple[object, ...], float] = {}
    for _, row in successful_full.iterrows():
        key = _normalization_key(row)
        if key in ceilings:
            raise ExperimentIntegrityError(
                f"duplicate full-budget ceiling for configuration {key!r}"
            )
        ceilings[key] = float(row["test_r2"])
    for index, row in result.iterrows():
        if row["fit_status"] != "ok":
            continue
        key = _normalization_key(row)
        if key not in ceilings:
            raise ExperimentIntegrityError(
                f"missing full-budget ceiling for configuration {key!r}"
            )
        ceiling = ceilings[key]
        eligible = bool(np.isfinite(ceiling) and ceiling >= threshold)
        result.at[index, "full_budget_test_r2"] = ceiling
        result.at[index, "ceiling_eligible"] = eligible
        if eligible:
            result.at[index, "normalized_recovery"] = (
                float(row["test_r2"]) / ceiling
            )
    validate_result_schema(result, finalized=True)
    return result


def block_recovery_points(frame: pd.DataFrame) -> pd.DataFrame:
    """One mean/median target-wise recovery point per encoder/subsample cell."""
    values = frame[
        frame["fit_status"].eq("ok")
        & frame["target_independent"].eq(True)
        & frame["ceiling_eligible"].eq(True)
    ].copy()
    if values.empty:
        return pd.DataFrame()
    key = [
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "budget_kind",
        "budget_days_per_stock",
        "budget_stock_day_equivalents",
        "n_rows",
        "n_rows_over_dim",
        "subsample_seed",
        "block_anchor_quantile",
        "feature_view",
        "whiten_k_requested",
        "whiten_k_effective",
        "reader_family",
        "alpha_selected",
        "alpha",
    ]
    grouped = values.groupby(key, dropna=False, observed=True)
    result = grouped["normalized_recovery"].agg(
        recovery_mean="mean",
        recovery_median="median",
        recovery_min="min",
        recovery_max="max",
        n_eligible_targets="count",
    ).reset_index()
    aggregate_scores = grouped[["test_r2", "full_budget_test_r2"]].mean().reset_index(
        drop=True
    )
    result["aggregate_block_test_r2"] = aggregate_scores["test_r2"].to_numpy()
    result["aggregate_block_full_budget_test_r2"] = aggregate_scores[
        "full_budget_test_r2"
    ].to_numpy()
    result["aggregate_block_score_ratio"] = (
        result["aggregate_block_test_r2"]
        / result["aggregate_block_full_budget_test_r2"]
    )
    noninterpretable = (
        result["target_block"].eq("directional")
        & (result["n_eligible_targets"] < 2)
    )
    result.loc[noninterpretable, [
        "recovery_mean",
        "recovery_median",
        "recovery_min",
        "recovery_max",
    ]] = np.nan
    result["interpretable"] = ~noninterpretable
    return result


@dataclass(frozen=True)
class HierarchicalInterval:
    mean: float
    lower: float
    upper: float
    sd_subsample_within_encoder: float
    sd_encoder_between_means: float
    n_encoders: int
    n_subsamples: int


def _hierarchical_group_task(
    payload: tuple[
        tuple[object, ...],
        pd.DataFrame,
        str,
        str,
        str,
        int,
        int,
    ],
) -> tuple[tuple[object, ...], HierarchicalInterval]:
    (
        key,
        group,
        value_column,
        encoder_column,
        subsample_column,
        n_bootstrap,
        seed,
    ) = payload
    return (
        key,
        hierarchical_interval(
            group,
            value_column,
            encoder_column=encoder_column,
            subsample_column=subsample_column,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    )


def _hierarchical_group_intervals(
    grouped,
    *,
    value_column: str,
    encoder_column: str,
    subsample_column: str,
    n_bootstrap: int,
    seed: int,
    n_workers: int,
):
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    selected_columns = list(
        dict.fromkeys([encoder_column, subsample_column, value_column])
    )
    tasks = (
        (
            key if isinstance(key, tuple) else (key,),
            group.loc[:, selected_columns],
            value_column,
            encoder_column,
            subsample_column,
            n_bootstrap,
            seed,
        )
        for key, group in grouped
    )
    if n_workers == 1:
        yield from map(_hierarchical_group_task, tasks)
        return
    # Every group resets the same preregistered RNG seed, so groups are
    # independent.  imap preserves group order and therefore produces exactly
    # the serial table while reducing only wall-clock time.
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        yield from pool.imap(_hierarchical_group_task, tasks, chunksize=64)


def hierarchical_interval(
    points: pd.DataFrame,
    value_column: str,
    *,
    encoder_column: str = "encoder_seed",
    subsample_column: str = "subsample_seed",
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> HierarchicalInterval:
    valid = points[
        np.isfinite(points[value_column].to_numpy(dtype=np.float64))
    ].copy()
    encoders = sorted(valid[encoder_column].unique().tolist())
    if not encoders:
        return HierarchicalInterval(
            *(float("nan"),) * 5, n_encoders=0, n_subsamples=0
        )
    arrays = {
        encoder: valid.loc[
            valid[encoder_column] == encoder, value_column
        ].to_numpy(dtype=np.float64)
        for encoder in encoders
    }
    encoder_means = np.asarray(
        [arrays[encoder].mean() for encoder in encoders], dtype=np.float64
    )
    within_variances = np.asarray(
        [
            arrays[encoder].var(ddof=1) if len(arrays[encoder]) > 1 else 0.0
            for encoder in encoders
        ]
    )
    within = float(np.sqrt(np.mean(within_variances)))
    between = float(
        encoder_means.std(ddof=1) if len(encoder_means) > 1 else 0.0
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        sampled_encoders = rng.choice(encoders, size=len(encoders), replace=True)
        encoder_draws = []
        for encoder in sampled_encoders:
            values = arrays[encoder]
            encoder_draws.append(
                float(rng.choice(values, size=len(values), replace=True).mean())
            )
        draws[draw] = float(np.mean(encoder_draws))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return HierarchicalInterval(
        mean=float(encoder_means.mean()),
        lower=float(lower),
        upper=float(upper),
        sd_subsample_within_encoder=within,
        sd_encoder_between_means=between,
        n_encoders=len(encoders),
        n_subsamples=len(valid),
    )


def uncertainty_summary(
    points: pd.DataFrame,
    *,
    value_column: str = "recovery_mean",
    n_bootstrap: int = 5000,
    seed: int = 0,
    n_workers: int = 1,
) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    points = points.copy()
    points["_comparison_alpha"] = np.where(
        points["reader_family"].astype(str).str.contains("common_alpha"),
        points["alpha"].astype(float),
        np.nan,
    )
    group_columns = [
        "branch",
        "readout",
        "target_block",
        "budget_kind",
        "budget_days_per_stock",
        "budget_stock_day_equivalents",
        "feature_view",
        "whiten_k_requested",
        "whiten_k_effective",
        "reader_family",
        "_comparison_alpha",
    ]
    rows = []
    grouped = points.groupby(group_columns, dropna=False, observed=True)
    for key, interval in _hierarchical_group_intervals(
        grouped,
        value_column=value_column,
        encoder_column="encoder_seed",
        subsample_column="subsample_seed",
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_workers=n_workers,
    ):
        row = dict(zip(group_columns, key))
        row["alpha"] = row.pop("_comparison_alpha")
        row.update(interval.__dict__)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gap_points(
    points: pd.DataFrame,
    left_branch: str = "supervised",
    right_branch: str = "jepa_horizon",
) -> pd.DataFrame:
    points = points.copy()
    points["_comparison_alpha"] = np.where(
        points["reader_family"].astype(str).str.contains("common_alpha"),
        points["alpha"].astype(float),
        np.nan,
    )
    key = [
        "encoder_seed",
        "readout",
        "target_block",
        "budget_kind",
        "budget_days_per_stock",
        "budget_stock_day_equivalents",
        "subsample_seed",
        "feature_view",
        "whiten_k_requested",
        "whiten_k_effective",
        "reader_family",
        "_comparison_alpha",
    ]
    left = points[points["branch"] == left_branch][
        key + ["recovery_mean"]
    ].rename(columns={"recovery_mean": "left_recovery"})
    right = points[points["branch"] == right_branch][
        key + ["recovery_mean"]
    ].rename(columns={"recovery_mean": "right_recovery"})
    paired = left.merge(right, on=key, how="inner", validate="one_to_one")
    paired["normalized_gap"] = (
        paired["left_recovery"] - paired["right_recovery"]
    )
    return paired


def gap_summary(
    paired: pd.DataFrame,
    *,
    delta: float = PRIMARY_GAP_DELTA,
    n_bootstrap: int = 5000,
    seed: int = 0,
    n_workers: int = 1,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame(
            columns=[
                "readout",
                "target_block",
                "budget_kind",
                "budget_days_per_stock",
                "budget_stock_day_equivalents",
                "feature_view",
                "whiten_k_requested",
                "whiten_k_effective",
                "reader_family",
                "alpha",
                "mean",
                "lower",
                "upper",
                "sd_subsample_within_encoder",
                "sd_encoder_between_means",
                "n_encoders",
                "n_subsamples",
                "delta",
                "robust",
            ]
        )
    group_columns = [
        "readout",
        "target_block",
        "budget_kind",
        "budget_days_per_stock",
        "budget_stock_day_equivalents",
        "feature_view",
        "whiten_k_requested",
        "whiten_k_effective",
        "reader_family",
        "_comparison_alpha",
    ]
    rows = []
    grouped = paired.groupby(group_columns, dropna=False, observed=True)
    for key, interval in _hierarchical_group_intervals(
        grouped,
        value_column="normalized_gap",
        encoder_column="encoder_seed",
        subsample_column="subsample_seed",
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_workers=n_workers,
    ):
        row = dict(zip(group_columns, key))
        row["alpha"] = row.pop("_comparison_alpha")
        row.update(interval.__dict__)
        row["delta"] = float(delta)
        row["robust"] = bool(interval.lower > 0.0 and interval.mean >= delta)
        rows.append(row)
    return pd.DataFrame(rows)


def adjacent_robust_low_budgets(summary: pd.DataFrame) -> list[tuple[float, float]]:
    if summary.empty or not {
        "budget_days_per_stock",
        "robust",
    }.issubset(summary.columns):
        return []
    values = summary[
        summary["budget_days_per_stock"].isin(LOW_BUDGETS)
        & summary["robust"].eq(True)
    ]["budget_days_per_stock"].astype(float)
    robust = set(values.tolist())
    ordered = [value for value in LOW_BUDGETS if value in robust]
    return [
        (left, right)
        for left, right in zip(ordered, ordered[1:])
        if LOW_BUDGETS.index(right) == LOW_BUDGETS.index(left) + 1
    ]


def threshold_budget(
    points: pd.DataFrame,
    threshold: float,
    value_column: str = "mean",
) -> tuple[float | None, bool]:
    ordered = points.sort_values("budget_stock_day_equivalents")
    reached = ordered[ordered[value_column] >= threshold]
    if reached.empty:
        return None, True
    return float(reached.iloc[0]["budget_stock_day_equivalents"]), False


def area_under_log_budget(
    budgets: Sequence[float], values: Sequence[float]
) -> float:
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    return float(np.trapezoid(y[order], np.log(x[order])))

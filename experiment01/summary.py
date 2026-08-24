"""Preregistered summaries, hierarchical uncertainty and outcome assignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import GAP_SENSITIVITIES, LOW_BUDGETS, PRIMARY_GAP_DELTA
from .io import atomic_write_json, atomic_write_parquet
from .results import (
    adjacent_robust_low_budgets,
    area_under_log_budget,
    block_recovery_points,
    gap_summary,
    hierarchical_interval,
    paired_gap_points,
    threshold_budget,
    uncertainty_summary,
    validate_result_schema,
)


def raw_block_points(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[
        frame["fit_status"].eq("ok") & frame["target_independent"].eq(True)
    ].copy()
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
        "feature_view",
        "whiten_k_requested",
        "whiten_k_effective",
        "reader_family",
        "alpha",
    ]
    return (
        values.groupby(key, dropna=False, observed=True)["test_r2"]
        .agg(raw_r2_mean="mean", raw_r2_median="median", n_targets="count")
        .reset_index()
    )


def _reader_subset(
    points: pd.DataFrame,
    *,
    readout: str,
    block: str,
    reader: str,
    feature_view: str,
) -> pd.DataFrame:
    return points[
        points["readout"].eq(readout)
        & points["target_block"].eq(block)
        & points["reader_family"].eq(reader)
        & points["feature_view"].eq(feature_view)
    ].copy()


def _gap_at_budgets(
    summary: pd.DataFrame, budgets: Sequence[float]
) -> pd.DataFrame:
    return summary[
        summary["budget_days_per_stock"].astype(float).isin(
            [float(value) for value in budgets]
        )
    ]


def _all_encoder_gap_positive(paired: pd.DataFrame, budgets: Sequence[float]) -> bool:
    values = paired[
        paired["budget_days_per_stock"].astype(float).isin(budgets)
    ]
    if values.empty:
        return False
    means = values.groupby("encoder_seed", observed=True)["normalized_gap"].mean()
    return len(means) >= 2 and bool((means > 0.0).all())


def _ceiling_is_meaningful(frame: pd.DataFrame) -> bool:
    values = frame[
        frame["budget_kind"].eq("full_train")
        & frame["readout"].eq("last_concat512")
        & frame["target_block"].eq("directional")
        & frame["reader_family"].eq("ridge_raw_tuned_alpha")
        & frame["feature_view"].eq("full_rank_raw")
        & frame["branch"].isin(["supervised", "jepa_horizon"])
        & frame["target_independent"].eq(True)
        & frame["ceiling_eligible"].eq(True)
    ]
    counts = values.groupby(["branch", "encoder_seed"], observed=True).size()
    return (
        len(counts) >= 4
        and bool((counts >= 2).all())
        and set(values["branch"]) == {"supervised", "jepa_horizon"}
    )


def _absolute_ceiling_gap(
    raw: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
) -> Mapping[str, float | bool]:
    values = _reader_subset(
        raw,
        readout="last_concat512",
        block="directional",
        reader="ridge_raw_tuned_alpha",
        feature_view="full_rank_raw",
    )
    values = values[values["budget_kind"].eq("full_train")]
    left = values[values["branch"].eq("supervised")][
        ["encoder_seed", "raw_r2_mean"]
    ]
    right = values[values["branch"].eq("jepa_horizon")][
        ["encoder_seed", "raw_r2_mean"]
    ]
    paired = left.merge(
        right, on="encoder_seed", suffixes=("_supervised", "_jepa")
    )
    if paired.empty:
        return {"mean": float("nan"), "lower": float("nan"), "robust": False}
    paired["gap"] = (
        paired["raw_r2_mean_supervised"] - paired["raw_r2_mean_jepa"]
    )
    interval = hierarchical_interval(
        paired.assign(subsample_seed=-1),
        "gap",
        n_bootstrap=n_bootstrap,
        seed=0,
    )
    return {
        "mean": interval.mean,
        "lower": interval.lower,
        "upper": interval.upper,
        "robust": bool(interval.lower > 0.0),
    }


def classify_directional_outcome(
    frame: pd.DataFrame,
    recovery: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    delta: float = PRIMARY_GAP_DELTA,
    n_bootstrap: int = 5000,
    n_workers: int = 1,
) -> Mapping[str, object]:
    primary = _reader_subset(
        recovery,
        readout="last_concat512",
        block="directional",
        reader="ridge_raw_tuned_alpha",
        feature_view="full_rank_raw",
    )
    native_paired = paired_gap_points(primary)
    native_summary = gap_summary(
        native_paired,
        delta=delta,
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
    )
    adjacent = adjacent_robust_low_budgets(native_summary)
    native_low = (
        native_summary[
            native_summary["budget_days_per_stock"].isin(LOW_BUDGETS)
        ]["mean"]
        if not native_summary.empty
        else pd.Series(dtype=float)
    )
    meaningful = _ceiling_is_meaningful(frame)
    ceiling_gap = _absolute_ceiling_gap(raw, n_bootstrap=n_bootstrap)
    details: dict[str, object] = {
        "delta": delta,
        "native_adjacent_robust_pairs": [list(value) for value in adjacent],
        "large_sample_ceiling_meaningful": meaningful,
        "absolute_ceiling_gap": dict(ceiling_gap),
        "low_budget_mean_normalized_gap": (
            float(native_low.mean()) if len(native_low) else float("nan")
        ),
    }
    if adjacent and meaningful:
        decisive_budgets = list(adjacent[0])
        raw_gap = _gap_at_budgets(native_summary, decisive_budgets)
        baseline = float(raw_gap["mean"].mean())
        whiten = _reader_subset(
            recovery,
            readout="last_concat512",
            block="directional",
            reader="ridge_whiten_topk_tuned_alpha",
            feature_view="full_rank_whiten_topk",
        )
        whiten_paired = paired_gap_points(whiten)
        whiten_summary = gap_summary(
            whiten_paired,
            delta=delta,
            n_bootstrap=n_bootstrap,
            n_workers=n_workers,
        )
        candidates = []
        for requested_k, group in whiten_summary.groupby(
            "whiten_k_requested", dropna=True, observed=True
        ):
            same = _gap_at_budgets(group, decisive_budgets)
            if len(same) != len(decisive_budgets):
                continue
            mean_gap = float(same["mean"].mean())
            candidates.append(
                {
                    "k": int(requested_k),
                    "mean_gap": mean_gap,
                    "reduction_fraction": (
                        float("nan")
                        if baseline == 0.0
                        else 1.0 - mean_gap / baseline
                    ),
                    "all_nonrobust": bool((~same["robust"]).all()),
                    "all_robust": bool(same["robust"].all()),
                }
            )
        candidates.sort(key=lambda row: row["k"])
        k_50 = next(
            (
                row["k"]
                for row in candidates
                if row["reduction_fraction"] >= 0.5
            ),
            None,
        )
        k_nonrobust = next(
            (row["k"] for row in candidates if row["all_nonrobust"]),
            None,
        )
        details.update(
            {
                "decisive_budgets": decisive_budgets,
                "native_low_budget_mean_gap": baseline,
                "whitening_candidates": candidates,
                "k_50gap": k_50,
                "k_nonrobust": k_nonrobust,
            }
        )
        if (
            k_50 is not None
            and k_nonrobust is not None
            and _all_encoder_gap_positive(native_paired, decisive_budgets)
        ):
            return {
                "outcome": "A1",
                "reason": (
                    "native finite-sample gap is robust at adjacent low budgets "
                    "and progressive whitening reduces it by at least 50% and "
                    "makes it non-robust"
                ),
                **details,
            }
        maximum = candidates[-1] if candidates else None
        ols = _reader_subset(
            recovery,
            readout="last_concat512",
            block="directional",
            reader="min_norm_ols_raw",
            feature_view="full_rank_raw",
        )
        ols_summary = gap_summary(
            paired_gap_points(ols),
            delta=delta,
            n_bootstrap=n_bootstrap,
            n_workers=n_workers,
        )
        ols_same = _gap_at_budgets(ols_summary, decisive_budgets)
        persistent_ols = (
            len(ols_same) == len(decisive_budgets)
            and bool(ols_same["robust"].all())
        )
        persistent_maximum = bool(maximum and maximum["all_robust"])
        details.update(
            {
                "maximum_matched_whitening": maximum,
                "min_norm_persistent": persistent_ols,
            }
        )
        if (
            persistent_maximum
            and persistent_ols
            and _all_encoder_gap_positive(native_paired, decisive_budgets)
        ):
            return {
                "outcome": "A2",
                "reason": (
                    "native gap persists through the maximum matched valid "
                    "whitening depth and min-norm OLS, with positive means for "
                    "every paired encoder seed"
                ),
                **details,
            }

    if not adjacent and bool(ceiling_gap["robust"]):
        return {
            "outcome": "B",
            "reason": (
                "large-sample raw ceiling gap is robust while no normalized "
                "delta=0.10 gap persists at adjacent low budgets"
            ),
            **details,
        }
    return {
        "outcome": "D",
        "reason": (
            "the executed evidence does not satisfy the complete preregistered "
            "conditions for A1, A2 or B"
        ),
        **details,
    }


def curve_quantities(
    uncertainty: pd.DataFrame,
    *,
    value_column: str = "mean",
) -> pd.DataFrame:
    if uncertainty.empty:
        return pd.DataFrame()
    group_columns = [
        "branch",
        "readout",
        "target_block",
        "feature_view",
        "whiten_k_requested",
        "reader_family",
        "alpha",
    ]
    rows = []
    for key, group in uncertainty.groupby(
        group_columns, dropna=False, observed=True
    ):
        group = group.sort_values("budget_stock_day_equivalents")
        row = dict(zip(group_columns, key))
        row["area_under_log_budget_curve"] = area_under_log_budget(
            group["budget_stock_day_equivalents"], group[value_column]
        )
        for threshold in (0.5, 0.8, 0.9):
            budget, censored = threshold_budget(group, threshold, value_column)
            label = int(threshold * 100)
            row[f"budget_reaching_{label}pct"] = budget
            row[f"budget_reaching_{label}pct_right_censored"] = censored
        rows.append(row)
    return pd.DataFrame(rows)


def specificity_gap_signatures(
    recovery: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    n_workers: int = 1,
) -> Mapping[str, object]:
    signatures: dict[str, object] = {}
    for block in ("directional", "volatility", "timing"):
        points = _reader_subset(
            recovery,
            readout="last_concat512",
            block=block,
            reader="ridge_raw_tuned_alpha",
            feature_view="full_rank_raw",
        )
        summary = gap_summary(
            paired_gap_points(points),
            delta=PRIMARY_GAP_DELTA,
            n_bootstrap=n_bootstrap,
            n_workers=n_workers,
        )
        low = (
            summary[summary["budget_days_per_stock"].isin(LOW_BUDGETS)]
            if not summary.empty
            else summary
        )
        signatures[block] = {
            "role": (
                "primary A1/A2/B/D classification"
                if block == "directional"
                else "specificity control; not pooled into primary outcome"
            ),
            "adjacent_robust_low_budget_pairs": [
                list(value) for value in adjacent_robust_low_budgets(summary)
            ],
            "low_budget_mean_normalized_gap": (
                float(low["mean"].mean()) if len(low) else float("nan")
            ),
        }
    return signatures


def summarize_phase1(
    results_path: str | Path,
    out_dir: str | Path,
    *,
    n_bootstrap: int = 5000,
    n_workers: int = 1,
) -> Mapping[str, object]:
    frame = pd.read_parquet(results_path)
    validate_result_schema(frame, finalized=True)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    recovery = block_recovery_points(frame)
    raw = raw_block_points(frame)
    recovery_uncertainty = uncertainty_summary(
        recovery,
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
    )
    raw_uncertainty = uncertainty_summary(
        raw.rename(columns={"raw_r2_mean": "recovery_mean"}),
        value_column="recovery_mean",
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
    ).rename(
        columns={
            "mean": "raw_r2_mean",
            "lower": "raw_r2_lower",
            "upper": "raw_r2_upper",
        }
    )
    paired = paired_gap_points(recovery)
    gaps = {
        str(delta): gap_summary(
            paired,
            delta=delta,
            n_bootstrap=n_bootstrap,
            n_workers=n_workers,
        )
        for delta in GAP_SENSITIVITIES
    }
    outcome = classify_directional_outcome(
        frame,
        recovery,
        raw,
        delta=PRIMARY_GAP_DELTA,
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
    )
    quantities = curve_quantities(recovery_uncertainty)
    if not quantities.empty:
        quantities = quantities.rename(
            columns={
                "area_under_log_budget_curve": (
                    "area_under_normalized_log_budget_curve"
                )
            }
        )
    raw_for_auc = raw_uncertainty.rename(columns={"raw_r2_mean": "mean"})
    raw_quantities = curve_quantities(raw_for_auc)
    if not raw_quantities.empty:
        raw_quantities = raw_quantities[
            [
                column
                for column in raw_quantities.columns
                if not column.startswith("budget_reaching_")
            ]
        ].rename(
            columns={
                "area_under_log_budget_curve": (
                    "area_under_raw_log_budget_curve"
                )
            }
        )
    artifacts = {
        "block_recovery_points.parquet": recovery,
        "raw_block_points.parquet": raw,
        "recovery_uncertainty.parquet": recovery_uncertainty,
        "raw_uncertainty.parquet": raw_uncertainty,
        "gap_summary_delta_005.parquet": gaps["0.05"],
        "gap_summary_delta_010.parquet": gaps["0.1"],
        "gap_summary_delta_015.parquet": gaps["0.15"],
        "curve_quantities.parquet": quantities,
        "raw_curve_quantities.parquet": raw_quantities,
    }
    for name, table in artifacts.items():
        atomic_write_parquet(table, destination / name)
    payload: dict[str, object] = {
        "directional_last_concat512_outcome": outcome,
        "target_block_gap_signatures": specificity_gap_signatures(
            recovery,
            n_bootstrap=n_bootstrap,
            n_workers=n_workers,
        ),
        "established_coexisting_condition": "C_pooling",
        "bootstrap": {
            "algorithm": "encoder_then_within_encoder_subsample_resampling",
            "n_draws": n_bootstrap,
            "seed": 0,
            "workers": n_workers,
        },
        "artifacts": {
            name: {"n_rows": len(table)} for name, table in artifacts.items()
        },
    }
    atomic_write_json(destination / "summary.json", payload)
    return payload

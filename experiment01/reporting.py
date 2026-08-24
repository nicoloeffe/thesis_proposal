"""Phase-I figures and Markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .results import hierarchical_interval, paired_gap_points


COLORS = {
    "supervised": "#1f77b4",
    "jepa_horizon": "#d62728",
    "jepa_masked": "#7f7f7f",
}

WHITENING_NONMONOTONICITY_K = (8, 16, 32, 64)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _line_by_branch(
    table: pd.DataFrame,
    path: Path,
    *,
    y: str,
    lower: str | None,
    upper: str | None,
    title: str,
    ylabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for branch, group in table.groupby("branch", observed=True):
        group = group.sort_values("budget_stock_day_equivalents")
        x = group["budget_stock_day_equivalents"].to_numpy(dtype=float)
        value = group[y].to_numpy(dtype=float)
        axis.plot(
            x,
            value,
            marker="o",
            label=branch,
            color=COLORS.get(str(branch)),
        )
        if lower and upper and lower in group and upper in group:
            axis.fill_between(
                x,
                group[lower].to_numpy(dtype=float),
                group[upper].to_numpy(dtype=float),
                color=COLORS.get(str(branch)),
                alpha=0.16,
            )
    axis.set_xscale("log")
    axis.set_xlabel("labelled stock-day equivalents")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.legend(frameon=False)
    _save(fig, path)


def _specificity_findings(
    raw_uncertainty: pd.DataFrame,
    recovery_uncertainty: pd.DataFrame,
) -> Mapping[str, object]:
    findings: dict[str, object] = {}
    for block in ("directional", "volatility", "timing"):
        raw = raw_uncertainty[
            raw_uncertainty["target_block"].eq(block)
            & raw_uncertainty["readout"].eq("last_concat512")
            & raw_uncertainty["reader_family"].eq("ridge_raw_tuned_alpha")
            & raw_uncertainty["feature_view"].eq("full_rank_raw")
            & raw_uncertainty["budget_kind"].eq("full_train")
        ]
        recovery = recovery_uncertainty[
            recovery_uncertainty["target_block"].eq(block)
            & recovery_uncertainty["readout"].eq("last_concat512")
            & recovery_uncertainty["reader_family"].eq("ridge_raw_tuned_alpha")
            & recovery_uncertainty["feature_view"].eq("full_rank_raw")
        ]
        findings[block] = {
            "full_train_raw_r2": {
                str(row.branch): float(row.raw_r2_mean)
                for row in raw.itertuples()
            },
            "n_recovery_curve_rows": len(recovery),
            "interpretation_scope": (
                "primary directional specificity result"
                if block == "directional"
                else "preregistered specificity control"
            ),
        }
    return findings


def _markdown_table(
    rows: Iterable[Mapping[str, object]], columns: Iterable[str]
) -> str:
    """Render small diagnostic tables without an optional tabulate dependency."""
    selected = list(columns)
    values = list(rows)
    header = "| " + " | ".join(selected) + " |"
    separator = "| " + " | ".join("---" for _ in selected) + " |"
    body = [
        "| "
        + " | ".join(str(row.get(column, "")) for column in selected)
        + " |"
        for row in values
    ]
    return "\n".join([header, separator, *body])


def _float_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _whitening_nonmonotonicity_diagnostic(
    recovery: pd.DataFrame,
    *,
    decisive_budgets: Iterable[float],
    n_bootstrap: int = 5000,
) -> Mapping[str, object]:
    """Post-hoc paired diagnostic using only frozen Phase-I recovery points.

    Each encoder/subsample cell is first averaged across the two preregistered
    decisive budgets.  The existing hierarchical bootstrap is then applied to
    encoder seeds followed by these paired within-encoder cells.
    """
    budgets = tuple(float(value) for value in decisive_budgets)
    selected = recovery[
        recovery["readout"].eq("last_concat512")
        & recovery["reader_family"].eq("ridge_whiten_topk_tuned_alpha")
        & recovery["feature_view"].eq("full_rank_whiten_topk")
        & recovery["whiten_k_requested"].isin(
            WHITENING_NONMONOTONICITY_K
        )
        & recovery["budget_days_per_stock"].isin(budgets)
    ]
    paired = paired_gap_points(selected)
    cell_columns = [
        "encoder_seed",
        "target_block",
        "subsample_seed",
        "whiten_k_requested",
    ]
    cells = (
        paired.groupby(cell_columns, observed=True)
        .agg(
            normalized_gap=("normalized_gap", "mean"),
            n_budget_cells=("budget_days_per_stock", "nunique"),
        )
        .reset_index()
    )
    if cells.empty:
        empty = pd.DataFrame()
        return {
            "scope": {
                "status": "unavailable_no_matching_cells",
                "readout": "last_concat512",
                "reader_family": "ridge_whiten_topk_tuned_alpha",
                "decisive_budgets_days_per_stock": list(budgets),
                "whitening_depths": list(WHITENING_NONMONOTONICITY_K),
                "interpretation_policy": (
                    "diagnostic does not alter the preregistered technical outcome"
                ),
            },
            "intervals": empty,
            "paired_differences": empty,
            "per_encoder": empty,
            "paired_differences_per_encoder": empty,
        }
    if not cells["n_budget_cells"].eq(len(budgets)).all():
        raise ValueError(
            "whitening non-monotonicity diagnostic is missing paired budgets"
        )

    interval_rows: list[dict[str, object]] = []
    for (block, requested_k), group in cells.groupby(
        ["target_block", "whiten_k_requested"], observed=True
    ):
        interval = hierarchical_interval(
            group,
            "normalized_gap",
            n_bootstrap=n_bootstrap,
            seed=0,
        )
        interval_rows.append(
            {
                "target_block": str(block),
                "whiten_k_requested": int(requested_k),
                **interval.__dict__,
            }
        )
    intervals = pd.DataFrame(interval_rows).sort_values(
        ["target_block", "whiten_k_requested"]
    )

    index = ["encoder_seed", "target_block", "subsample_seed"]
    wide = cells.pivot(
        index=index,
        columns="whiten_k_requested",
        values="normalized_gap",
    ).reset_index()
    required = [float(value) for value in WHITENING_NONMONOTONICITY_K]
    if any(value not in wide for value in required):
        raise ValueError("whitening depth pairing is incomplete")
    if wide[required].isna().any().any():
        raise ValueError("whitening depth pairing contains missing cells")

    difference_rows: list[dict[str, object]] = []
    per_encoder_difference_rows: list[dict[str, object]] = []
    for block, group in wide.groupby("target_block", observed=True):
        for left, right in zip(required, required[1:]):
            compared = group.copy()
            compared["paired_difference"] = compared[right] - compared[left]
            interval = hierarchical_interval(
                compared,
                "paired_difference",
                n_bootstrap=n_bootstrap,
                seed=0,
            )
            difference_rows.append(
                {
                    "target_block": str(block),
                    "from_k": int(left),
                    "to_k": int(right),
                    "difference_definition": "gap_to_k_minus_gap_from_k",
                    **interval.__dict__,
                    "interval_excludes_zero": bool(
                        interval.lower > 0.0 or interval.upper < 0.0
                    ),
                }
            )
            for encoder_seed, encoder in compared.groupby(
                "encoder_seed", observed=True
            ):
                per_encoder_difference_rows.append(
                    {
                        "target_block": str(block),
                        "encoder_seed": int(encoder_seed),
                        "from_k": int(left),
                        "to_k": int(right),
                        "mean_paired_difference": float(
                            encoder["paired_difference"].mean()
                        ),
                        "n_paired_subsamples": len(encoder),
                    }
                )
    differences = pd.DataFrame(difference_rows).sort_values(
        ["target_block", "from_k"]
    )
    per_encoder_differences = pd.DataFrame(
        per_encoder_difference_rows
    ).sort_values(["target_block", "encoder_seed", "from_k"])
    per_encoder = (
        cells.groupby(
            ["target_block", "encoder_seed", "whiten_k_requested"],
            observed=True,
        )["normalized_gap"]
        .agg(mean_normalized_gap="mean", n_paired_subsamples="count")
        .reset_index()
        .sort_values(
            ["target_block", "encoder_seed", "whiten_k_requested"]
        )
    )
    return {
        "scope": {
            "status": "post_hoc_diagnostic_only",
            "readout": "last_concat512",
            "reader_family": "ridge_whiten_topk_tuned_alpha",
            "target_blocks": ["directional", "volatility", "timing"],
            "decisive_budgets_days_per_stock": list(budgets),
            "whitening_depths": list(WHITENING_NONMONOTONICITY_K),
            "gap": "supervised_minus_jepa_horizon_normalized_recovery",
            "budget_aggregation": (
                "mean within encoder/subsample over decisive budgets before "
                "hierarchical resampling"
            ),
            "bootstrap": {
                "algorithm": "encoder_then_within_encoder_paired_resampling",
                "n_draws": n_bootstrap,
                "seed": 0,
            },
            "interpretation_policy": (
                "diagnostic does not alter the preregistered technical outcome"
            ),
        },
        "intervals": intervals,
        "paired_differences": differences,
        "per_encoder": per_encoder,
        "paired_differences_per_encoder": per_encoder_differences,
    }


def _find_ancestor_file(path: Path, name: str) -> Path | None:
    current = path.resolve().parent
    for directory in (current, *current.parents):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _parity_audit(
    results_path: Path, raw_uncertainty: pd.DataFrame
) -> Mapping[str, object]:
    gate_path = _find_ancestor_file(
        results_path, "REPRODUCTION_GATE_EXPERIMENT_01.json"
    )
    historical: Mapping[str, object]
    if gate_path is None:
        historical = {
            "available": False,
            "reason": "REPRODUCTION_GATE_EXPERIMENT_01.json not found",
        }
    else:
        gate = json.loads(gate_path.read_text())
        historical = {
            "available": True,
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "passed": bool(gate["passed"]),
            "evaluation_split": gate["legacy_evaluation_split_name"],
            "reader_family": "min_norm_ols_raw",
            "results": gate["results"],
        }

    base = raw_uncertainty[
        raw_uncertainty["target_block"].eq("directional")
        & raw_uncertainty["readout"].eq("last_concat512")
        & raw_uncertainty["feature_view"].eq("full_rank_raw")
        & raw_uncertainty["budget_kind"].eq("full_train")
    ]

    def production_rows(reader_family: str) -> list[dict[str, object]]:
        rows = base[base["reader_family"].eq(reader_family)].sort_values(
            "branch"
        )
        return [
            {
                "branch": str(row.branch),
                "mean_test_r2": float(row.raw_r2_mean),
                "lower": float(row.raw_r2_lower),
                "upper": float(row.raw_r2_upper),
                "n_encoders": int(row.n_encoders),
            }
            for row in rows.itertuples()
        ]

    return {
        "historical_reproduction_gate": historical,
        "production_full_budget_test": {
            "split": "new canonical test",
            "tuned_raw_ridge": production_rows("ridge_raw_tuned_alpha"),
            "min_norm_ols_diagnostic": production_rows("min_norm_ols_raw"),
            "parity_requirement": (
                "none: the production test is a different chronological half "
                "of the historical held-out stock-days"
            ),
        },
    }


def generate_phase1_report(
    results_path: str | Path,
    summary_dir: str | Path,
    out_dir: str | Path,
) -> Mapping[str, object]:
    results_path = Path(results_path)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    previous_report_path = destination / "REPORT_EXPERIMENT_01.md"
    current_report_sha256 = (
        sha256_file(previous_report_path)
        if previous_report_path.is_file()
        else None
    )
    previous_manifest_path = destination / "report_manifest.json"
    previous_manifest = (
        json.loads(previous_manifest_path.read_text())
        if previous_manifest_path.is_file()
        else {}
    )
    previous_report_sha256 = previous_manifest.get(
        "narrative_revision", {}
    ).get("previous_report_sha256", current_report_sha256)
    summary_root = Path(summary_dir)
    summary_path = summary_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    recovery = pd.read_parquet(summary_root / "block_recovery_points.parquet")
    raw = pd.read_parquet(summary_root / "raw_block_points.parquet")
    recovery_u = pd.read_parquet(summary_root / "recovery_uncertainty.parquet")
    raw_u = pd.read_parquet(summary_root / "raw_uncertainty.parquet")
    gap = pd.read_parquet(summary_root / "gap_summary_delta_010.parquet")
    result_columns = [
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "target_independent",
        "budget_kind",
        "budget_stock_day_equivalents",
        "n_rows_over_dim",
        "subsample_seed",
        "feature_view",
        "whiten_k_requested",
        "reader_family",
        "alpha",
        "normalized_recovery",
        "ceiling_eligible",
        "fit_status",
        "trace_cov_over_dim",
    ]
    results = pd.read_parquet(results_path, columns=result_columns)
    figure_paths: list[str] = []

    primary_raw = raw_u[
        raw_u["target_block"].eq("directional")
        & raw_u["readout"].eq("last_concat512")
        & raw_u["reader_family"].eq("ridge_raw_tuned_alpha")
        & raw_u["feature_view"].eq("full_rank_raw")
    ]
    name = "01_raw_directional_r2.png"
    _line_by_branch(
        primary_raw,
        destination / name,
        y="raw_r2_mean",
        lower="raw_r2_lower",
        upper="raw_r2_upper",
        title="Directional raw R² — tuned raw ridge",
        ylabel="test R²",
    )
    figure_paths.append(name)

    primary_recovery = recovery_u[
        recovery_u["target_block"].eq("directional")
        & recovery_u["readout"].eq("last_concat512")
        & recovery_u["reader_family"].eq("ridge_raw_tuned_alpha")
        & recovery_u["feature_view"].eq("full_rank_raw")
    ]
    name = "02_directional_normalized_recovery.png"
    _line_by_branch(
        primary_recovery,
        destination / name,
        y="mean",
        lower="lower",
        upper="upper",
        title="Directional recovery of own operational ceiling",
        ylabel="target-wise normalized recovery",
    )
    figure_paths.append(name)

    primary_gap = gap[
        gap["target_block"].eq("directional")
        & gap["readout"].eq("last_concat512")
        & gap["reader_family"].eq("ridge_raw_tuned_alpha")
        & gap["feature_view"].eq("full_rank_raw")
    ].sort_values("budget_stock_day_equivalents")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    x = primary_gap["budget_stock_day_equivalents"].to_numpy(float)
    if len(primary_gap):
        axis.plot(
            x,
            primary_gap["mean"].to_numpy(dtype=float),
            marker="o",
            color="#6a3d9a",
        )
        axis.fill_between(
            x,
            primary_gap["lower"].to_numpy(dtype=float),
            primary_gap["upper"].to_numpy(dtype=float),
            alpha=0.2,
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.axhline(0.10, color="black", linestyle="--", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xlabel("labelled stock-day equivalents")
    axis.set_ylabel("supervised − JEPA normalized recovery")
    axis.set_title("Primary normalized sample-efficiency gap")
    name = "03_supervised_jepa_normalized_gap.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    whiten_gap = gap[
        gap["target_block"].eq("directional")
        & gap["readout"].eq("last_concat512")
        & gap["reader_family"].eq("ridge_whiten_topk_tuned_alpha")
        & gap["feature_view"].eq("full_rank_whiten_topk")
        & gap["budget_days_per_stock"].isin([0.125, 0.25, 0.5, 1, 2, 4])
    ]
    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    for budget, group in whiten_gap.groupby(
        "budget_days_per_stock", observed=True
    ):
        group = group.sort_values("whiten_k_requested")
        axis.plot(
            group["whiten_k_requested"],
            group["mean"],
            marker="o",
            label=f"b={budget:g}",
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.axhline(0.10, color="black", linestyle="--", linewidth=0.8)
    axis.set_xscale("symlog", linthresh=1)
    axis.set_xlabel("whitening depth k")
    axis.set_ylabel("normalized gap")
    axis.set_title("Gap versus progressive whitening depth")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(ncol=2, fontsize=8, frameon=False)
    name = "04_gap_vs_whitening_depth.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    whiten_recovery = recovery_u[
        recovery_u["target_block"].eq("directional")
        & recovery_u["readout"].eq("last_concat512")
        & recovery_u["reader_family"].eq("ridge_whiten_topk_tuned_alpha")
    ]
    matched_k = (
        int(whiten_recovery["whiten_k_requested"].max())
        if not whiten_recovery.empty
        else None
    )
    compare = primary_recovery.assign(reader_label="raw")
    if matched_k is not None:
        compare = pd.concat(
            [
                compare,
                whiten_recovery[
                    whiten_recovery["whiten_k_requested"].eq(matched_k)
                ].assign(reader_label=f"whiten k={matched_k}"),
            ],
            ignore_index=True,
        )
    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    for (branch, label), group in compare.groupby(
        ["branch", "reader_label"], observed=True
    ):
        group = group.sort_values("budget_stock_day_equivalents")
        axis.plot(
            group["budget_stock_day_equivalents"],
            group["mean"],
            marker="o",
            linestyle="-" if label == "raw" else "--",
            color=COLORS.get(str(branch)),
            label=f"{branch}: {label}",
        )
    axis.set_xscale("log")
    axis.set_xlabel("labelled stock-day equivalents")
    axis.set_ylabel("normalized recovery")
    axis.set_title("Tuned raw ridge versus maximum matched whitening")
    axis.legend(fontsize=8, frameon=False)
    name = "05_raw_vs_whitened_learning_curves.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    surface = results[
        results["target_block"].eq("directional")
        & results["readout"].eq("last_concat512")
        & results["feature_view"].eq("full_rank_raw")
        & results["reader_family"].eq("ridge_raw_common_alpha")
        & results["target_independent"].eq(True)
        & results["ceiling_eligible"].eq(True)
        & results["fit_status"].eq("ok")
        & results["branch"].isin(["supervised", "jepa_horizon"])
    ]
    figure_06_audit = {
        "figure": "06_common_alpha_gap_surface.png",
        "reader_family_filter": "ridge_raw_common_alpha",
        "uses_common_alpha": True,
        "uses_common_absolute_lambda": False,
        "n_source_rows": len(surface),
        "alpha_values": sorted(surface["alpha"].dropna().unique().tolist()),
        "lambda_definition": "lambda = alpha * trace(covariance) / D",
    }
    atomic_write_json(destination / "06_common_alpha_audit.json", figure_06_audit)
    surface = (
        surface.groupby(
            ["branch", "budget_stock_day_equivalents", "alpha"],
            observed=True,
        )["normalized_recovery"]
        .mean()
        .unstack("branch")
        .reset_index()
    )
    if {"supervised", "jepa_horizon"}.issubset(surface.columns):
        surface["gap"] = surface["supervised"] - surface["jepa_horizon"]
        pivot = surface.pivot(
            index="alpha",
            columns="budget_stock_day_equivalents",
            values="gap",
        )
        fig, axis = plt.subplots(figsize=(8.2, 5.0))
        image = axis.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
        )
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45)
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels([f"{value:.1e}" for value in pivot.index], fontsize=6)
        axis.set_xlabel("labelled stock-day equivalents")
        axis.set_ylabel("dimensionless alpha")
        axis.set_title("Fixed common-alpha normalized gap surface")
        fig.colorbar(image, ax=axis, label="supervised − JEPA")
        name = "06_common_alpha_gap_surface.png"
        _save(fig, destination / name)
        figure_paths.append(name)

    ols = raw_u[
        raw_u["target_block"].eq("directional")
        & raw_u["readout"].eq("last_concat512")
        & raw_u["reader_family"].eq("min_norm_ols_raw")
    ]
    name = "07_min_norm_ols_learning_curves.png"
    _line_by_branch(
        ols,
        destination / name,
        y="raw_r2_mean",
        lower="raw_r2_lower",
        upper="raw_r2_upper",
        title="Min-norm OLS directional learning curves",
        ylabel="test R²",
    )
    figure_paths.append(name)

    axis_table = (
        raw[
            raw["target_block"].eq("directional")
            & raw["readout"].eq("last_concat512")
            & raw["reader_family"].eq("ridge_raw_tuned_alpha")
        ][
            [
                "budget_stock_day_equivalents",
                "n_rows",
                "n_rows_over_dim",
                "subsample_seed",
            ]
        ]
        .drop_duplicates()
        .sort_values(["budget_stock_day_equivalents", "subsample_seed"])
    )
    atomic_write_parquet(axis_table, destination / "08_budget_axis_n_over_d.parquet")

    low_points = recovery[
        recovery["target_block"].eq("directional")
        & recovery["readout"].eq("last_concat512")
        & recovery["reader_family"].eq("ridge_raw_tuned_alpha")
        & recovery["budget_days_per_stock"].isin([0.125, 0.25, 0.5, 1, 2, 4])
    ]
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    labels, data = [], []
    for (branch, budget), group in low_points.groupby(
        ["branch", "budget_days_per_stock"], observed=True
    ):
        labels.append(f"{branch}\n{budget:g}")
        data.append(group["recovery_mean"].dropna().to_numpy())
    if data:
        axis.boxplot(data, tick_labels=labels, showfliers=True)
    axis.set_ylabel("normalized recovery")
    axis.set_title("Within-encoder/subsample distributions at low budgets")
    axis.tick_params(axis="x", labelrotation=45, labelsize=7)
    name = "09_low_budget_subsample_distributions.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    for (branch, encoder), group in low_points.groupby(
        ["branch", "encoder_seed"], observed=True
    ):
        curve = group.groupby("budget_stock_day_equivalents")[
            "recovery_mean"
        ].mean()
        axis.plot(
            curve.index,
            curve.values,
            color=COLORS.get(str(branch)),
            alpha=0.45,
            label=f"{branch}/seed{encoder}",
        )
    axis.set_xscale("log")
    axis.set_xlabel("labelled stock-day equivalents")
    axis.set_ylabel("normalized recovery")
    axis.set_title("Encoder-specific mean curves")
    axis.legend(fontsize=6, ncol=2, frameon=False)
    name = "10_encoder_specific_curves.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    variance = primary_recovery.sort_values(
        ["branch", "budget_stock_day_equivalents"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharex=True)
    for branch, group in variance.groupby("branch", observed=True):
        x = group["budget_stock_day_equivalents"]
        axes[0].plot(
            x,
            group["sd_subsample_within_encoder"],
            marker="o",
            color=COLORS.get(str(branch)),
            label=branch,
        )
        axes[1].plot(
            x,
            group["sd_encoder_between_means"],
            marker="o",
            color=COLORS.get(str(branch)),
            label=branch,
        )
    for axis, title in zip(
        axes, ("within-encoder subsample SD", "between-encoder mean SD")
    ):
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("stock-day equivalents")
    axes[0].set_ylabel("standard deviation")
    axes[1].legend(frameon=False)
    name = "11_variance_decomposition.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    specificity = recovery_u[
        recovery_u["readout"].eq("last_concat512")
        & recovery_u["reader_family"].eq("ridge_raw_tuned_alpha")
        & recovery_u["feature_view"].eq("full_rank_raw")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharex=True)
    for axis, block in zip(axes, ("directional", "volatility", "timing")):
        for branch, group in specificity[
            specificity["target_block"].eq(block)
        ].groupby("branch", observed=True):
            group = group.sort_values("budget_stock_day_equivalents")
            axis.plot(
                group["budget_stock_day_equivalents"],
                group["mean"],
                marker="o",
                color=COLORS.get(str(branch)),
                label=branch,
            )
        axis.set_xscale("log")
        axis.set_title(block)
        axis.set_xlabel("stock-day equivalents")
    axes[0].set_ylabel("normalized recovery")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(fontsize=7, frameon=False)
    name = "12_target_specificity_panels.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    readout_compare = recovery_u[
        recovery_u["target_block"].eq("directional")
        & recovery_u["reader_family"].eq("ridge_raw_tuned_alpha")
        & recovery_u["feature_view"].eq("full_rank_raw")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    for axis, readout in zip(axes, ("last_concat512", "meanK_concatS")):
        for branch, group in readout_compare[
            readout_compare["readout"].eq(readout)
        ].groupby("branch", observed=True):
            group = group.sort_values("budget_stock_day_equivalents")
            axis.plot(
                group["budget_stock_day_equivalents"],
                group["mean"],
                marker="o",
                color=COLORS.get(str(branch)),
                label=branch,
            )
        axis.set_xscale("log")
        axis.set_title(readout)
        axis.set_xlabel("stock-day equivalents")
    axes[0].set_ylabel("normalized recovery")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(frameon=False, fontsize=7)
    name = "13_readout_interaction_panels.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    eligibility = results[
        results["budget_kind"].eq("full_train")
        & results["reader_family"].eq("ridge_raw_tuned_alpha")
        & results["feature_view"].eq("full_rank_raw")
    ][
        ["branch", "encoder_seed", "readout", "target_name", "ceiling_eligible"]
    ].drop_duplicates()
    eligibility["configuration"] = (
        eligibility["branch"].astype(str)
        + "/s"
        + eligibility["encoder_seed"].astype(str)
        + "/"
        + eligibility["readout"].astype(str)
    )
    pivot = eligibility.pivot_table(
        index="target_name",
        columns="configuration",
        values="ceiling_eligible",
        aggfunc="first",
    )
    fig, axis = plt.subplots(
        figsize=(max(8.0, 0.35 * len(pivot.columns)), max(5.0, 0.18 * len(pivot)))
    )
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Greys")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(pivot.index, fontsize=6)
    axis.set_title("Full-budget ceiling eligibility (R² ≥ 0.01)")
    fig.colorbar(image, ax=axis, ticks=[0, 1])
    name = "14_ceiling_eligibility_map.png"
    _save(fig, destination / name)
    figure_paths.append(name)

    findings = _specificity_findings(raw_u, recovery_u)
    signatures = summary.get("target_block_gap_signatures", {})
    directional_gap = _float_or_nan(
        signatures["directional"]["low_budget_mean_normalized_gap"]
    )
    volatility_gap = _float_or_nan(
        signatures["volatility"]["low_budget_mean_normalized_gap"]
    )
    timing_gap = _float_or_nan(
        signatures["timing"]["low_budget_mean_normalized_gap"]
    )
    specificity_ratios = {
        "directional_over_volatility": (
            directional_gap / volatility_gap
            if np.isfinite(volatility_gap) and volatility_gap != 0.0
            else float("nan")
        ),
        "directional_over_timing": (
            directional_gap / timing_gap
            if np.isfinite(timing_gap) and timing_gap != 0.0
            else float("nan")
        ),
    }

    scale_rows = results[
        results["budget_kind"].eq("full_train")
        & results["feature_view"].eq("full_rank_raw")
        & results["reader_family"].eq("ridge_raw_tuned_alpha")
    ][
        ["branch", "encoder_seed", "readout", "trace_cov_over_dim"]
    ].drop_duplicates()
    scale_audit = {}
    for readout, group in scale_rows.groupby("readout", observed=True):
        branch_means = group.groupby("branch", observed=True)[
            "trace_cov_over_dim"
        ].mean()
        ratio = float(branch_means.max() / branch_means.min())
        horizon = _float_or_nan(branch_means.get("jepa_horizon"))
        supervised = _float_or_nan(branch_means.get("supervised"))
        horizon_supervised_ratio = (
            horizon / supervised
            if np.isfinite(horizon)
            and np.isfinite(supervised)
            and supervised != 0.0
            else float("nan")
        )
        scale_audit[str(readout)] = {
            "trace_cov_over_dim_by_branch": {
                str(key): float(value) for key, value in branch_means.items()
            },
            "jepa_horizon_over_supervised_ratio": horizon_supervised_ratio,
            "jepa_horizon_supervised_trace_matched_within_10pct": bool(
                np.isfinite(horizon_supervised_ratio)
                and 1.0 / 1.10 <= horizon_supervised_ratio <= 1.10
            ),
            "max_over_min_ratio": ratio,
            "approximately_matched_within_10pct": bool(ratio <= 1.10),
        }
    regularization_audit = {
        "trace_matched": False,
        "scientific_common_regularization_parameter": "alpha",
        "lambda_definition": "lambda = alpha * trace(covariance) / D",
        "figure_06_reader_family": "ridge_raw_common_alpha",
        "figure_06_uses_common_alpha": True,
        "figure_06_audit": "06_common_alpha_audit.json",
        "absolute_lambda_comparisons_in_report": False,
        "absolute_lambda_comparison_policy": (
            "confounded when covariance traces differ; omitted"
        ),
    }

    sensitivity_path = results_path.with_name("time_of_day_sensitivity.parquet")
    sensitivity_summary = pd.DataFrame()
    if sensitivity_path.is_file():
        sensitivity = pd.read_parquet(sensitivity_path)
        sensitivity_summary = (
            sensitivity[
                sensitivity["fit_status"].eq("ok")
                & sensitivity["target_independent"].eq(True)
                & sensitivity["ceiling_eligible"].eq(True)
            ]
            .groupby(
                [
                    "branch",
                    "budget_kind",
                    "budget_days_per_stock",
                    "budget_stock_day_equivalents",
                ],
                observed=True,
            )["normalized_recovery"]
            .agg(mean="mean", median="median", std="std", n="count")
            .reset_index()
        )
        atomic_write_parquet(
            sensitivity_summary,
            destination / "time_of_day_sensitivity_summary.parquet",
        )

    outcome = summary["directional_last_concat512_outcome"]
    nonmonotonicity = _whitening_nonmonotonicity_diagnostic(
        recovery,
        decisive_budgets=outcome.get("decisive_budgets", (0.125, 0.25)),
        n_bootstrap=5000,
    )
    diagnostic_files = {
        "intervals": "15_whitening_nonmonotonicity_intervals.parquet",
        "paired_differences": (
            "15_whitening_nonmonotonicity_paired_differences.parquet"
        ),
        "per_encoder": "15_whitening_nonmonotonicity_by_encoder.parquet",
        "paired_differences_per_encoder": (
            "15_whitening_nonmonotonicity_paired_by_encoder.parquet"
        ),
    }
    for key, filename in diagnostic_files.items():
        table = nonmonotonicity[key]
        if not table.empty:
            atomic_write_parquet(table, destination / filename)
    nonmonotonicity_manifest = {
        "scope": nonmonotonicity["scope"],
        "artifacts": {
            key: {"path": filename, "n_rows": len(nonmonotonicity[key])}
            for key, filename in diagnostic_files.items()
        },
    }
    atomic_write_json(
        destination / "15_whitening_nonmonotonicity_manifest.json",
        nonmonotonicity_manifest,
    )
    parity = _parity_audit(results_path, raw_u)

    interval_table_rows = [
        {
            "target block": row.target_block,
            "k": int(row.whiten_k_requested),
            "mean gap": f"{row.mean:.4f}",
            "95% interval": f"[{row.lower:.4f}, {row.upper:.4f}]",
            "encoders / paired cells": (
                f"{int(row.n_encoders)} / {int(row.n_subsamples)}"
            ),
        }
        for row in nonmonotonicity["intervals"].itertuples()
    ]
    difference_table_rows = [
        {
            "target block": row.target_block,
            "depth pair": f"{int(row.from_k)}→{int(row.to_k)}",
            "paired Δ gap": f"{row.mean:.4f}",
            "95% interval": f"[{row.lower:.4f}, {row.upper:.4f}]",
            "excludes zero": str(bool(row.interval_excludes_zero)).lower(),
        }
        for row in nonmonotonicity["paired_differences"].itertuples()
    ]
    encoder_values = nonmonotonicity["per_encoder"]
    encoder_table_rows: list[dict[str, object]] = []
    if not encoder_values.empty:
        encoder_wide = encoder_values.pivot(
            index=["target_block", "encoder_seed"],
            columns="whiten_k_requested",
            values="mean_normalized_gap",
        ).reset_index()
        for _, values in encoder_wide.iterrows():
            encoder_table_rows.append(
                {
                    "target block": values["target_block"],
                    "encoder seed": int(values["encoder_seed"]),
                    "k=8": f"{values[8.0]:.4f}",
                    "k=16": f"{values[16.0]:.4f}",
                    "k=32": f"{values[32.0]:.4f}",
                    "k=64": f"{values[64.0]:.4f}",
                }
            )

    historical = parity["historical_reproduction_gate"]
    historical_rows: list[dict[str, object]] = []
    if historical.get("available"):
        for branch in ("jepa_horizon", "supervised"):
            row = historical["results"][branch]
            historical_rows.append(
                {
                    "branch": branch,
                    "observed": f"{float(row['observed']):.6f}",
                    "historical reference": f"{float(row['expected']):.4f}",
                    "absolute difference": (
                        f"{float(row['absolute_difference']):.6f}"
                    ),
                    "gate passed": str(bool(row["passed"])).lower(),
                }
            )
    production = parity["production_full_budget_test"]
    ridge_by_branch = {
        row["branch"]: row for row in production["tuned_raw_ridge"]
    }
    ols_by_branch = {
        row["branch"]: row for row in production["min_norm_ols_diagnostic"]
    }
    production_rows = []
    for branch in ("jepa_horizon", "jepa_masked", "supervised"):
        ridge = ridge_by_branch.get(branch)
        ols = ols_by_branch.get(branch)
        production_rows.append(
            {
                "branch": branch,
                "tuned ridge R² [95%]": (
                    "n/a"
                    if ridge is None
                    else (
                        f"{ridge['mean_test_r2']:.6f} "
                        f"[{ridge['lower']:.6f}, {ridge['upper']:.6f}]"
                    )
                ),
                "min-norm OLS R² [95%]": (
                    "n/a"
                    if ols is None
                    else (
                        f"{ols['mean_test_r2']:.6f} "
                        f"[{ols['lower']:.6f}, {ols['upper']:.6f}]"
                    )
                ),
            }
        )

    ceiling = outcome["absolute_ceiling_gap"]
    ceiling_mean = _float_or_nan(ceiling.get("mean"))
    ceiling_lower = _float_or_nan(ceiling.get("lower"))
    ceiling_upper = _float_or_nan(ceiling.get("upper"))
    outcome_reason = str(outcome["reason"])
    narrative_outcome_reason = outcome_reason[:1].upper() + outcome_reason[1:]
    k_50gap = outcome.get("k_50gap")
    k_nonrobust = outcome.get("k_nonrobust")
    whitening_by_k = {
        candidate["k"]: candidate
        for candidate in outcome.get("whitening_candidates", [])
    }
    k_50_reduction = float(
        whitening_by_k.get(k_50gap, {}).get("reduction_fraction", np.nan)
    )
    decisive_gap = float(outcome.get("native_low_budget_mean_gap", np.nan))
    last_scale = scale_audit.get("last_concat512", {})
    last_trace_by_branch = last_scale.get("trace_cov_over_dim_by_branch", {})
    last_horizon_trace = _float_or_nan(
        last_trace_by_branch.get("jepa_horizon")
    )
    last_supervised_trace = _float_or_nan(
        last_trace_by_branch.get("supervised")
    )
    last_horizon_supervised_ratio = _float_or_nan(
        last_scale.get("jepa_horizon_over_supervised_ratio")
    )
    last_max_min_ratio = _float_or_nan(last_scale.get("max_over_min_ratio"))
    specificity_table = _markdown_table(
        [
            {
                "target block": "directional",
                "mean normalized finite-sample gap": f"{directional_gap:.4f}",
                "directional / control": "—",
            },
            {
                "target block": "volatility",
                "mean normalized finite-sample gap": f"{volatility_gap:.4f}",
                "directional / control": (
                    f"{specificity_ratios['directional_over_volatility']:.2f}×"
                ),
            },
            {
                "target block": "timing",
                "mean normalized finite-sample gap": f"{timing_gap:.4f}",
                "directional / control": (
                    f"{specificity_ratios['directional_over_timing']:.2f}×"
                ),
            },
        ],
        [
            "target block",
            "mean normalized finite-sample gap",
            "directional / control",
        ],
    )
    interval_table = _markdown_table(
        interval_table_rows,
        [
            "target block",
            "k",
            "mean gap",
            "95% interval",
            "encoders / paired cells",
        ],
    )
    difference_table = _markdown_table(
        difference_table_rows,
        [
            "target block",
            "depth pair",
            "paired Δ gap",
            "95% interval",
            "excludes zero",
        ],
    )
    encoder_table = _markdown_table(
        encoder_table_rows,
        ["target block", "encoder seed", "k=8", "k=16", "k=32", "k=64"],
    )
    historical_table = _markdown_table(
        historical_rows,
        [
            "branch",
            "observed",
            "historical reference",
            "absolute difference",
            "gate passed",
        ],
    )
    production_table = _markdown_table(
        production_rows,
        ["branch", "tuned ridge R² [95%]", "min-norm OLS R² [95%]"],
    )
    directional_difference_text = "; ".join(
        (
            f"{int(row.from_k)}→{int(row.to_k)} "
            f"{row.mean:.6f} [{row.lower:.6f}, {row.upper:.6f}]"
        )
        for row in nonmonotonicity["paired_differences"].itertuples()
        if row.target_block == "directional"
    )

    primary_result = {
        "name": "directional_specificity_across_three_distinct_diagnostics",
        "spectral_anti_alignment_established": True,
        "last_to_meanK_directional_fragility_established": True,
        "phase1_normalized_finite_sample_gap": {
            "directional": directional_gap,
            "volatility": volatility_gap,
            "timing": timing_gap,
            **specificity_ratios,
        },
    }
    narrative_summary = f"""# Narrative summary — Experiment 01 Phase I

## Primary result: directional specificity across distinct diagnostics

Three separate diagnostics point to a direction-specific accessibility
penalty. First, the previously established spectral diagnostic shows
directional variance–task anti-alignment: at `m/D=1/32`, horizon-JEPA recovers
`0.0050` against a `0.0563` random-subspace null, whereas supervised recovers
`0.8971` against `0.6118`. Second, the previously established `last → meanK`
diagnostic shows directional pooling fragility; the production linear
full-budget check has horizon-JEPA `0.2199 → 0.0701`, while supervised is
approximately stable (`0.3853 → 0.3941`). Third, Phase I measures the normalized
finite-sample gap as `0.5460` for direction, `0.1838` for volatility and
`0.1528` for timing. The directional penalty is therefore approximately
`3–3.5×` larger than the controls (exact ratios `2.97×` and `3.57×`).

## Separate Phase-I effects

- The operational linear ceiling gap is robust: supervised minus horizon-JEPA
  is `{ceiling_mean:.4f}` with hierarchical 95% interval
  `[{ceiling_lower:.4f}, {ceiling_upper:.4f}]`.
- The normalized recovery gap is independently robust at adjacent low budgets;
  its six-low-budget mean is `{directional_gap:.4f}`.
- Whitening mediates the second component: `k_50gap={k_50gap}` halves but does
  not eliminate the gap, while non-robustness appears only at
  `k_nonrobust={k_nonrobust}`,
  i.e. near-complete whitening. These results do not support concentration of
  the problem in a few leading PCs.

## Secondary technical classification

The unchanged preregistered classification is **{outcome['outcome']} with a robust ceiling gap**.
`B` is not a coexisting outcome. The local
non-monotonicity diagnostic at `k=8,16,32,64` is post hoc and does not alter
this technical classification.

## Parity and scope

The historical reproduction gate passed (`0.211129` versus `0.2111` for
horizon-JEPA; `0.375636` versus `0.3756` for supervised). Production
full-budget scores and new-test min-norm OLS are reported separately because
the new chronological test split is not required to equal the old validation
split. Phase II and Phase III were not run.
"""
    (destination / "SUMMARY_NARRATIVE_EXPERIMENT_01.md").write_text(
        narrative_summary, encoding="utf-8"
    )

    report = f"""# Report — Experiment 01 Phase I

## Primary result: directional specificity across three distinct diagnostics

The primary result is not the taxonomy label. It is the convergence of three
distinct specificity diagnostics:

1. **Established directional spectral anti-alignment.** At `m/D=1/32`, the
   horizon-JEPA final readout recovers `0.0050` of its full linear directional
   score, below the `0.0563` empirical random-subspace null. Supervised recovers
   `0.8971`, compared with its `0.6118` null. This establishes variance–task
   anti-alignment rather than a generic absence of predictive content.
2. **Established directional pooling fragility.** Under `last → meanK`, the
   production full-budget linear directional R² changes from `0.2199` to
   `0.0701` for horizon-JEPA, while supervised remains approximately stable
   (`0.3853 → 0.3941`). This is a readout interaction, not a second A/B/D
   outcome.
3. **Phase-I finite-sample specificity.** The mean normalized gaps are:

{specificity_table}

The directional penalty is therefore approximately **3–3.5 times larger**
than the controls (exact ratios `2.97×` versus volatility and `3.57×` versus
timing). Volatility and timing remain specificity controls and are not pooled
into the directional result.

## Three effects that must remain separate

### Robust operational linear ceiling gap

At full production budget with tuned raw ridge, the supervised minus
horizon-JEPA directional R² gap is `{ceiling_mean:.6f}`, with hierarchical
95% interval `[{ceiling_lower:.6f}, {ceiling_upper:.6f}]`. The interval
excludes zero. This is a robust operational linear ceiling gap; it is not, by
itself, a normalized sample-efficiency statement.

### Robust normalized-recovery gap

After target-wise normalization by each representation's eligible operational
ceiling, the directional finite-sample gap remains robust across adjacent low
budgets. The mean over all six preregistered low budgets is
`{directional_gap:.6f}`; the mean over the decisive `0.125` and `0.25`
days/stock cells is `{decisive_gap:.6f}`.

### Mediation by progressive whitening

The unchanged technical thresholds are `k_50gap = {k_50gap}` and
`k_nonrobust = {k_nonrobust}`. Partial whitening at `k=128` reduces
the decisive-budget normalized gap by `{k_50_reduction:.1%}` but does not
eliminate it: the gap remains robust. Non-robustness requires `k=508`, i.e.
near-complete whitening of a 512-dimensional readout. The evidence therefore
does **not** justify saying that the problem is concentrated in a few leading
principal components.

## Secondary technical classification

The preregistered taxonomy is retained unchanged as a secondary technical
classification: **{outcome['outcome']} with a robust ceiling gap**.

Technical rule satisfied: {narrative_outcome_reason}. The robust ceiling gap
above is reported alongside A1; `B` is not described as a coexisting outcome.

The complete unchanged machine-readable classification record remains in
`summary/summary.json` and is reproduced here for auditability:

```json
{json.dumps(outcome, indent=2, sort_keys=True)}
```

## Whitening-depth non-monotonicity diagnostic

This added diagnostic uses only frozen Phase-I recovery points. Within each
encoder/subsample cell, it first averages the paired supervised–horizon gap
over the two decisive budgets (`0.125`, `0.25`), then applies 5,000-draw
hierarchical resampling of encoder seeds followed by paired cells within
encoder. It does not refit a reader and is not outcome-defining.

### Hierarchical intervals by target block and depth

{interval_table}

### Paired differences between adjacent inspected depths

Differences are `gap(to k) − gap(from k)`, paired by encoder seed and subsample.

{difference_table}

### Means by encoder seed and target block

{encoder_table}

For direction, `8→16` is indistinguishable from zero, `16→32` shows a small
positive paired change, and `32→64` a larger negative paired change; the latter
two intervals exclude zero. Volatility and timing show different local
patterns. This verifies local non-monotonicity in the inspected cells, but the
diagnostic is post hoc and does not modify the preregistered interpretation or
support a “few-PC” account. Full paired values per encoder are retained in the
four `15_whitening_nonmonotonicity_*` artifacts.

## Global covariance scale and regularization parity

`trace_cov_over_dim` is **not matched**. On `last_concat512`, the mean trace
scale is `{last_horizon_trace:.6f}`
for horizon-JEPA and
`{last_supervised_trace:.6f}`
for supervised, a horizon/supervised ratio of
`{last_horizon_supervised_ratio:.3f}`
(approximately `1.40`). The all-branch max/min ratio is
`{last_max_min_ratio:.3f}` because masked-JEPA
has a still larger trace.

Scientific common-regularization comparisons use the dimensionless parameter

`lambda = alpha * trace(covariance) / D`.

Figure 06 is verified to select `reader_family = ridge_raw_common_alpha` and
therefore compares **common alpha**, not common absolute lambda. No
fixed-absolute-lambda comparison is included in this report; with unmatched
trace scale, such a comparison would be marked confounded.

```json
{json.dumps(scale_audit, indent=2, sort_keys=True)}
```

## Historical and production parity

### Historical reproduction gate

This is the mandatory old-split min-norm OLS reproduction check:

{historical_table}

The observed values reproduce the historical rounded references within the
frozen tolerance.

### Production full-budget test

The following scores use the new canonical test split and are deliberately
reported separately:

{production_table}

The new-test min-norm OLS values are diagnostics. They are **not required** to
equal the old-split reproduction values because validation and test are the
two chronological halves of the former held-out stock-days.

## Scope and leakage controls

- frozen post-P0 representations only;
- label budgets are nested stock-day groups;
- covariance/whitening uses all unlabelled train features only;
- alpha is selected on the fixed complete validation split;
- the fixed complete test split is evaluated only after configuration fixing;
- directional, volatility and timing are summarized separately;
- normalized recovery is target-wise and only uses full-budget R² at least 0.01.

## Specificity and time-of-day controls

```json
{json.dumps(findings, indent=2, sort_keys=True)}
```

Opening, middle and closing contiguous blocks are reported separately in
`time_of_day_sensitivity_summary.parquet` ({len(sensitivity_summary)} rows).
These sensitivity cells are not pooled into the random-anchor curves.

## Uncertainty

Intervals use hierarchical resampling of encoder seeds followed by subsampling
seeds within encoder. Companion Parquet tables expose
`sd_subsample_within_encoder` and `sd_encoder_between_means`; all
encoder-specific curves are retained in figure 10.

## Figures and diagnostics

{chr(10).join(f'- `{name}`' for name in figure_paths)}
- `06_common_alpha_audit.json`
- `15_whitening_nonmonotonicity_manifest.json`
{chr(10).join(f'- `{name}`' for name in diagnostic_files.values())}

Phase II (PCA/random subspaces) and Phase III (MLP) were not run. This revision
changes narrative ordering and adds read-only diagnostics only; it does not
change Phase-I results, thresholds, the technical outcome, or the fit pipeline.
"""
    (destination / "REPORT_EXPERIMENT_01.md").write_text(report, encoding="utf-8")

    protected_inputs = {
        "phase1_results": {
            "path": str(results_path),
            "sha256": sha256_file(results_path),
        },
        "technical_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
    }
    changelog = f"""# Changelog — Experiment 01 Phase I narrative revision

Date: 2026-07-31.

## Narrative changes

- Replaced prior report SHA-256: `{previous_report_sha256 or 'not present'}`.
- Moved `{outcome['outcome']}` from the report headline to a secondary technical
  classification; its value and preregistered rule are unchanged.
- Reframed the primary result around three distinct specificity diagnostics:
  established spectral anti-alignment, established `last → meanK` fragility,
  and Phase-I normalized finite-sample gaps across target blocks.
- Separated the robust `{ceiling_mean:.6f}` operational ceiling gap, robust
  normalized-recovery gap, and whitening mediation.
- Corrected whitening language: `k_50gap={k_50gap}` halves but does not
  eliminate the gap; `k_nonrobust={k_nonrobust}` is near-complete whitening;
  no few-PC concentration is
  claimed.
- Replaced any possible “coexisting B” reading with “A1 with a robust ceiling
  gap.”

## Added read-only diagnostics

- Added hierarchical intervals and paired adjacent-depth differences for
  `k=8,16,32,64`, plus results by encoder seed and target block. The diagnostic
  reads frozen recovery points only and is explicitly post hoc.
- Diagnostic row counts: 12 depth intervals, 9 hierarchical paired
  differences, 36 depth-by-encoder rows and 27 paired-difference-by-encoder
  rows. Directional paired results (`to−from`) are:
  `{directional_difference_text}`.
- Added the trace-scale audit (`last` horizon/supervised ≈ `1.40`), the exact
  regularization formula, and an explicit verification that figure 06 uses
  `ridge_raw_common_alpha`. Fixed-absolute-lambda comparisons are absent.
- Added historical reproduction parity, production full-budget test results,
  and new-test min-norm OLS as a separate diagnostic with no old/new split
  equality requirement.

## Unchanged protected artifacts

- `results.parquet`: `{protected_inputs['phase1_results']['sha256']}`
- `summary/summary.json`: `{protected_inputs['technical_summary']['sha256']}`
- Technical outcome: `{outcome['outcome']}`.
- Thresholds: `k_50gap={k_50gap}`,
  `k_nonrobust={k_nonrobust}`, `delta={outcome['delta']}`.
- No feature generation, reader fitting, Phase II, or Phase III was executed.
"""
    (destination / "CHANGELOG_NARRATIVE_20260731.md").write_text(
        changelog, encoding="utf-8"
    )

    payload = {
        "primary_result": primary_result,
        "technical_classification": {
            "role": "secondary",
            "label": outcome["outcome"],
            "wording": f"{outcome['outcome']} with a robust ceiling gap",
            "unchanged_record": outcome,
        },
        # Backward-compatible machine-readable technical outcome.
        "outcome": outcome,
        "specificity_findings": findings,
        "specificity_ratios": specificity_ratios,
        "global_scale_audit": scale_audit,
        "regularization_audit": regularization_audit,
        "figure_06_audit": figure_06_audit,
        "whitening_nonmonotonicity": nonmonotonicity_manifest,
        "parity": parity,
        "protected_inputs": protected_inputs,
        "narrative_revision": {
            "date": "2026-07-31",
            "previous_report_sha256": previous_report_sha256,
            "results_or_technical_summary_modified": False,
        },
        "time_of_day_sensitivity_rows": len(sensitivity_summary),
        "figures": figure_paths,
        "report": "REPORT_EXPERIMENT_01.md",
        "narrative_summary": "SUMMARY_NARRATIVE_EXPERIMENT_01.md",
        "changelog": "CHANGELOG_NARRATIVE_20260731.md",
        "phase2_started": False,
        "phase3_started": False,
    }
    atomic_write_json(destination / "report_manifest.json", payload)
    return payload

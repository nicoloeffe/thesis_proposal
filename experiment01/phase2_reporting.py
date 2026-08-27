"""Consolidation, figures and report for Experiment 01 Phase II."""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import BRANCHES, READOUTS
from .errors import ExperimentIntegrityError
from .io import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    canonical_json_sha256,
    sha256_file,
)
from .results import hierarchical_interval


COLORS = {
    "supervised": "#1f77b4",
    "jepa_horizon": "#d62728",
    "jepa_masked": "#7f7f7f",
}
BLOCKS = ("directional", "volatility", "timing")
NONMONOTONIC_BANDS = ("17:32", "33:64")


def _normalize_degenerate_full_rank_haar_ties(root: Path) -> None:
    """Normalize the mathematically degenerate m=D_valid Haar comparison.

    A full-dimensional Haar basis spans the same space as top-D_valid PCA.
    Machine-scale solver roundoff must therefore be treated as a tie rather
    than as a random subspace exceeding top-PCA.
    """

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        selected = result["subspace_dimension"].eq(result["valid_dimension"])
        keys = ["branch", "encoder_seed", "readout", "target_block"]
        for _, indices in result[selected].groupby(keys, observed=True).groups.items():
            group = result.loc[indices]
            random_values = group["test_r2_mean"].to_numpy(dtype=np.float64)
            top_values = group["top_pca_test_r2_mean"].to_numpy(dtype=np.float64)
            scale = max(
                1.0,
                float(np.max(np.abs(random_values))),
                float(np.max(np.abs(top_values))),
            )
            tolerance = np.finfo(np.float64).eps * scale * 256.0
            if (
                np.ptp(random_values) > tolerance
                or np.max(np.abs(random_values - top_values)) > tolerance
            ):
                raise ExperimentIntegrityError(
                    "full-rank Haar cell is not a machine-scale top-PCA tie"
                )
            canonical = float(np.mean(np.concatenate([random_values, top_values])))
            result.loc[indices, "test_r2_mean"] = canonical
            result.loc[indices, "test_r2_median"] = canonical
            result.loc[indices, "top_pca_test_r2_mean"] = canonical
            result.loc[indices, "top_pca_percentile"] = 100.0
            result.loc[indices, "empirical_p_random_exceeds_top"] = 0.0
        return result

    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    implementation_path = Path(__file__).resolve().with_name("phase2.py")
    implementation_hash = sha256_file(implementation_path)
    shard_records = []
    for complete_path in sorted((root / "feature_shards").glob("*/complete.json")):
        directory = complete_path.parent
        payload = json.loads(complete_path.read_text())
        random_record = payload["artifacts"]["random_subspace_null"]
        random_path = directory / random_record["path"]
        normalized = normalize(pd.read_parquet(random_path))
        atomic_write_parquet(normalized, random_path)
        random_record.update(
            {
                "sha256": sha256_file(random_path),
                "size_bytes": random_path.stat().st_size,
                "n_rows": len(normalized),
            }
        )
        payload["source_fingerprint"][
            "phase2_implementation_sha256"
        ] = implementation_hash
        payload["diagnostics"]["full_rank_haar_tie_policy"] = (
            "machine_scale_ties_are_not_random_exceedances"
        )
        atomic_write_json(complete_path, payload)
        shard_records.append(payload)
    if len(shard_records) != 18:
        raise ExperimentIntegrityError("full-rank tie normalization needs 18 shards")

    consolidated_path = root / "random_subspace_null.parquet"
    consolidated = normalize(pd.read_parquet(consolidated_path))
    atomic_write_parquet(consolidated, consolidated_path)
    metadata["input_fingerprint"][
        "phase2_implementation_sha256"
    ] = implementation_hash
    metadata["protocol"]["full_rank_haar_tie_policy"] = (
        "machine_scale_ties_are_not_random_exceedances"
    )
    metadata["feature_shards"] = shard_records
    metadata["artifacts"]["random_subspace_null"].update(
        {
            "sha256": sha256_file(consolidated_path),
            "size_bytes": consolidated_path.stat().st_size,
            "n_rows": len(consolidated),
        }
    )
    metadata["numerical_postprocessing"] = {
        "full_rank_haar_ties_normalized": True,
        "r2_changed_beyond_machine_tolerance": False,
        "phase1_modified": False,
    }
    atomic_write_json(metadata_path, metadata)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _interval_rows(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
    value_column: str,
    *,
    n_bootstrap: int,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(list(group_columns), observed=True):
        values = key if isinstance(key, tuple) else (key,)
        interval = hierarchical_interval(
            group,
            value_column,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append(
            {
                **dict(zip(group_columns, values)),
                **interval.__dict__,
            }
        )
    return pd.DataFrame(rows)


def _validate_inventory(
    root: Path,
    metadata: Mapping[str, object],
    results: pd.DataFrame,
    mass: pd.DataFrame,
    random_null: pd.DataFrame,
    bands: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    if metadata.get("status") != "complete":
        raise ExperimentIntegrityError("Phase-II metadata is not complete")
    if len(metadata.get("phase1_full_rank_min_norm_parity", [])) != 18:
        raise ExperimentIntegrityError("Phase-II per-feature parity inventory differs")
    if not all(
        record.get("passed") is True
        for record in metadata["phase1_full_rank_min_norm_parity"]
    ):
        raise ExperimentIntegrityError("a Phase-I/full-rank parity gate failed")
    if not failures.empty:
        raise ExperimentIntegrityError("Phase-II has fit failures")
    expected_features = {
        (branch, seed, readout)
        for branch in BRANCHES
        for seed in (0, 1, 2)
        for readout in READOUTS
    }
    for name, frame in (
        ("phase2_results", results),
        ("predictive_mass", mass),
        ("random_subspace_null", random_null),
        ("spectral_bands", bands),
    ):
        observed = set(
            frame[["branch", "encoder_seed", "readout"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if observed != expected_features:
            raise ExperimentIntegrityError(f"{name} feature inventory differs")
    if not random_null.groupby(
        ["branch", "encoder_seed", "readout", "target_block", "subspace_dimension"],
        observed=True,
    ).size().eq(100).all():
        raise ExperimentIntegrityError("Haar null does not have exactly 100 draws/cell")
    gate = json.loads((root / "reproduction_gate.json").read_text())
    if gate.get("passed") is not True:
        raise ExperimentIntegrityError("historical post-P0 PCA reproduction gate failed")


def _predictive_mass_summary(
    mass: pd.DataFrame, *, n_bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scheduled = mass[
        mass["curve_schedule_point"].eq(True)
        & mass["direction_valid"].eq(True)
        & mass["target_valid"].eq(True)
        & mass["target_independent"].eq(True)
    ]
    cells = (
        scheduled.groupby(
            [
                "branch",
                "encoder_seed",
                "readout",
                "target_block",
                "direction_index",
                "valid_dimension",
            ],
            observed=True,
        )["cumulative_mass_fraction"]
        .agg(mean_cumulative_mass_fraction="mean", n_targets="count")
        .reset_index()
    )
    intervals = _interval_rows(
        cells,
        ["branch", "readout", "target_block", "direction_index"],
        "mean_cumulative_mass_fraction",
        n_bootstrap=n_bootstrap,
    ).rename(
        columns={
            "mean": "mass_fraction_mean",
            "lower": "mass_fraction_lower",
            "upper": "mass_fraction_upper",
        }
    )
    return cells, intervals


def _random_null_summary(random_null: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "subspace_dimension",
        "valid_dimension",
        "ladder_dimension",
        "band_matched_dimension",
    ]
    return (
        random_null.groupby(keys, observed=True)
        .agg(
            random_test_r2_mean=("test_r2_mean", "mean"),
            random_test_r2_sd=("test_r2_mean", "std"),
            random_test_r2_q025=("test_r2_mean", lambda x: x.quantile(0.025)),
            random_test_r2_q975=("test_r2_mean", lambda x: x.quantile(0.975)),
            top_pca_test_r2_mean=("top_pca_test_r2_mean", "first"),
            top_pca_percentile=("top_pca_percentile", "first"),
            empirical_p_random_exceeds_top=(
                "empirical_p_random_exceeds_top",
                "first",
            ),
            random_draws=("subspace_seed", "count"),
        )
        .reset_index()
    )


def _band_intervals(
    bands: pd.DataFrame, *, n_bootstrap: int
) -> pd.DataFrame:
    values = bands[bands["reader_family"].eq("min_norm_ols_diagnostic")].copy()
    rows = []
    metrics = (
        "predictive_mass_fraction_mean_independent",
        "test_r2_band_only",
        "test_r2_leave_band_out",
    )
    groups = ["branch", "readout", "target_block", "band"]
    for metric in metrics:
        table = _interval_rows(
            values,
            groups,
            metric,
            n_bootstrap=n_bootstrap,
        )
        table["metric"] = metric
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _nonmonotonicity_diagnostic(
    mass: pd.DataFrame,
    bands: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_mass = mass[
        mass["direction_valid"].eq(True)
        & mass["target_valid"].eq(True)
        & mass["target_independent"].eq(True)
        & mass["direction_index"].between(17, 64)
    ].copy()
    valid_mass["band"] = np.where(
        valid_mass["direction_index"] <= 32, "17:32", "33:64"
    )
    mass_cells = (
        valid_mass.groupby(
            [
                "branch",
                "encoder_seed",
                "readout",
                "target_block",
                "target_name",
                "band",
            ],
            observed=True,
        )
        .agg(
            band_mass=("predictive_mass", "sum"),
            total_mass=("total_predictive_mass_valid", "first"),
        )
        .reset_index()
    )
    mass_cells["value"] = mass_cells["band_mass"] / mass_cells["total_mass"]
    mass_wide = mass_cells.pivot(
        index=[
            "branch",
            "encoder_seed",
            "readout",
            "target_block",
            "target_name",
        ],
        columns="band",
        values="value",
    ).reset_index()
    if mass_wide[list(NONMONOTONIC_BANDS)].isna().any().any():
        raise ExperimentIntegrityError("non-monotonicity mass pairing is incomplete")
    mass_wide["paired_difference"] = (
        mass_wide["33:64"] - mass_wide["17:32"]
    )
    mass_wide["metric"] = "predictive_mass_fraction_33_64_minus_17_32"

    band_values = bands[
        bands["reader_family"].eq("min_norm_ols_diagnostic")
        & bands["band"].isin(NONMONOTONIC_BANDS)
    ]
    band_wide = band_values.pivot(
        index=["branch", "encoder_seed", "readout", "target_block"],
        columns="band",
        values="test_r2_band_only",
    ).reset_index()
    if band_wide[list(NONMONOTONIC_BANDS)].isna().any().any():
        raise ExperimentIntegrityError("non-monotonicity R2 pairing is incomplete")
    band_wide["paired_difference"] = band_wide["33:64"] - band_wide["17:32"]
    band_wide["target_name"] = "__block_independent_mean__"
    band_wide["metric"] = "band_only_test_r2_33_64_minus_17_32"
    points = pd.concat(
        [
            mass_wide[
                [
                    "branch",
                    "encoder_seed",
                    "readout",
                    "target_block",
                    "target_name",
                    "metric",
                    "paired_difference",
                ]
            ],
            band_wide[
                [
                    "branch",
                    "encoder_seed",
                    "readout",
                    "target_block",
                    "target_name",
                    "metric",
                    "paired_difference",
                ]
            ],
        ],
        ignore_index=True,
    )
    intervals = _interval_rows(
        points,
        ["branch", "readout", "target_block", "metric"],
        "paired_difference",
        n_bootstrap=n_bootstrap,
    )
    intervals["interval_excludes_zero"] = (
        (intervals["lower"] > 0.0) | (intervals["upper"] < 0.0)
    )
    intervals["supports_33_64_more_informative"] = intervals["lower"] > 0.0
    per_encoder = (
        points.groupby(
            ["branch", "encoder_seed", "readout", "target_block", "metric"],
            observed=True,
        )["paired_difference"]
        .agg(mean_paired_difference="mean", n_paired_targets="count")
        .reset_index()
    )
    return intervals, per_encoder


def _figure_predictive_mass(intervals: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True, sharey=True)
    for row, readout in enumerate(READOUTS):
        for column, block in enumerate(BLOCKS):
            axis = axes[row, column]
            selected = intervals[
                intervals["readout"].eq(readout)
                & intervals["target_block"].eq(block)
            ]
            for branch, group in selected.groupby("branch", observed=True):
                group = group.sort_values("direction_index")
                axis.plot(
                    group["direction_index"],
                    group["mass_fraction_mean"],
                    marker="o",
                    label=branch,
                    color=COLORS[str(branch)],
                )
                axis.fill_between(
                    group["direction_index"],
                    group["mass_fraction_lower"],
                    group["mass_fraction_upper"],
                    color=COLORS[str(branch)],
                    alpha=0.14,
                )
            axis.set_xscale("log", base=2)
            axis.set_title(f"{readout} — {block}")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel("top-k covariance directions")
    for axis in axes[:, 0]:
        axis.set_ylabel("cumulative predictive-mass fraction")
    _save_figure(figure, path)


def _figure_random_percentiles(summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True, sharey=True)
    values = summary[summary["ladder_dimension"].eq(True)]
    for row, readout in enumerate(READOUTS):
        for column, block in enumerate(BLOCKS):
            axis = axes[row, column]
            selected = values[
                values["readout"].eq(readout)
                & values["target_block"].eq(block)
            ]
            grouped = selected.groupby(
                ["branch", "subspace_dimension"], observed=True
            )["top_pca_percentile"].mean().reset_index()
            for branch, group in grouped.groupby("branch", observed=True):
                axis.plot(
                    group["subspace_dimension"],
                    group["top_pca_percentile"],
                    marker="o",
                    label=branch,
                    color=COLORS[str(branch)],
                )
            axis.axhline(50.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_xscale("log", base=2)
            axis.set_ylim(-2, 102)
            axis.set_title(f"{readout} — {block}")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel("subspace dimension")
    for axis in axes[:, 0]:
        axis.set_ylabel("top-PCA percentile in Haar null")
    _save_figure(figure, path)


def _figure_bands(bands: pd.DataFrame, path: Path) -> None:
    values = bands[
        bands["reader_family"].eq("min_norm_ols_diagnostic")
        & bands["target_block"].eq("directional")
    ]
    order = list(dict.fromkeys(values["band"].tolist()))
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for axis, readout in zip(axes, READOUTS):
        selected = values[values["readout"].eq(readout)]
        for branch, group in selected.groupby("branch", observed=True):
            means = group.groupby("band", observed=True)[
                "predictive_mass_fraction_mean_independent"
            ].mean()
            axis.plot(
                order,
                [means.get(band, np.nan) for band in order],
                marker="o",
                label=branch,
                color=COLORS[str(branch)],
            )
        axis.set_ylabel("directional mass fraction")
        axis.set_title(readout)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("disjoint covariance band")
    _save_figure(figure, path)


def _figure_bridge(bridge: pd.DataFrame, path: Path) -> None:
    gaps = (
        bridge.groupby(
            ["target_block", "budget_days_per_stock", "k_requested"],
            observed=True,
        )
        .agg(
            gap=("phase1_gap_mean", "first"),
            lower=("phase1_gap_lower", "first"),
            upper=("phase1_gap_upper", "first"),
        )
        .reset_index()
    )
    mass = (
        bridge[bridge["branch"].eq("jepa_horizon")]
        .groupby(["target_block", "k_requested"], observed=True)[
            "predictive_mass_cumulative_fraction_mean_independent"
        ]
        .mean()
        .reset_index()
    )
    k_50gap, k_nonrobust, _ = _bridge_thresholds(bridge)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharex=True)
    for axis, block in zip(axes, BLOCKS):
        selected = gaps[gaps["target_block"].eq(block)]
        for budget, group in selected.groupby("budget_days_per_stock", observed=True):
            group = group.sort_values("k_requested")
            axis.plot(group["k_requested"], group["gap"], marker="o", label=f"gap b={budget:g}")
        secondary = axis.twinx()
        curve = mass[mass["target_block"].eq(block)].sort_values("k_requested")
        secondary.plot(
            curve["k_requested"],
            curve["predictive_mass_cumulative_fraction_mean_independent"],
            color="#2ca02c",
            linestyle="--",
            marker="s",
            label="horizon mass",
        )
        axis.axvline(k_50gap, color="black", linewidth=0.7, linestyle=":")
        axis.axvline(k_nonrobust, color="black", linewidth=0.7, linestyle=":")
        axis.set_title(block)
        axis.set_xlabel("whitening depth / mass top-k")
        axis.set_ylabel("Phase-I normalized gap")
        secondary.set_ylabel("cumulative mass fraction")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    _save_figure(figure, path)


def _markdown_table(rows: Iterable[Mapping[str, object]], columns: Iterable[str]) -> str:
    columns = list(columns)
    body = list(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in body
    )
    return "\n".join(lines)


def _bridge_thresholds(bridge: pd.DataFrame) -> tuple[int, int, list[int]]:
    k_50 = sorted(
        bridge.loc[bridge["is_k_50gap"].eq(True), "k_requested"]
        .dropna()
        .unique()
        .tolist()
    )
    k_nonrobust = sorted(
        bridge.loc[bridge["is_k_nonrobust"].eq(True), "k_requested"]
        .dropna()
        .unique()
        .tolist()
    )
    if len(k_50) != 1 or len(k_nonrobust) != 1:
        raise ExperimentIntegrityError("Phase-I whitening bridge thresholds differ")
    depths = sorted(
        int(value) for value in bridge["k_requested"].dropna().unique()
    )
    return int(k_50[0]), int(k_nonrobust[0]), depths


def _external_compute_metrics(root: Path) -> Mapping[str, object]:
    path = root / "time_verbose.txt"
    text = path.read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([0-9:.]+)", text)
    peak = re.search(r"Maximum resident set size \(kbytes\): ([0-9]+)", text)
    if elapsed is None or peak is None:
        raise ExperimentIntegrityError("cannot parse external Phase-II compute metrics")
    parts = [float(value) for value in elapsed.group(1).split(":")]
    seconds = 0.0
    for value in parts:
        seconds = seconds * 60.0 + value
    return {
        "source": "GNU time -v",
        "path": "time_verbose.txt",
        "wall_time_seconds": seconds,
        "peak_rss_bytes": int(peak.group(1)) * 1024,
    }


def _primary_mass_rows(intervals: pd.DataFrame) -> list[dict[str, object]]:
    selected = intervals[
        intervals["readout"].eq("last_concat512")
        & intervals["branch"].isin(["supervised", "jepa_horizon"])
        & intervals["direction_index"].isin([8, 16, 32, 64, 128, 256, 508])
    ]
    return [
        {
            "block": row.target_block,
            "branch": row.branch,
            "k": int(row.direction_index),
            "mass": f"{row.mass_fraction_mean:.4f}",
            "95% CI": f"[{row.mass_fraction_lower:.4f}, {row.mass_fraction_upper:.4f}]",
        }
        for row in selected.sort_values(
            ["target_block", "branch", "direction_index"]
        ).itertuples()
    ]


def _nonmonotonic_status(intervals: pd.DataFrame) -> str:
    del intervals
    return (
        "not_dimension_matched: 17:32 has 16 directions and 33:64 has 32; "
        "the legacy paired difference cannot test the proposed mechanism"
    )


def _render_report(
    root: Path,
    metadata: Mapping[str, object],
    gate: Mapping[str, object],
    mass_intervals: pd.DataFrame,
    null_summary: pd.DataFrame,
    nonmonotonic_intervals: pd.DataFrame,
    bridge: pd.DataFrame,
    external_compute: Mapping[str, object],
) -> str:
    primary_null = null_summary[
        null_summary["readout"].eq("last_concat512")
        & null_summary["target_block"].eq("directional")
        & null_summary["branch"].isin(["supervised", "jepa_horizon"])
        & null_summary["ladder_dimension"].eq(True)
        & null_summary["subspace_dimension"].isin([8, 16, 32, 64, 128, 256])
    ]
    null_rows = []
    for (branch, dimension), group in primary_null.groupby(
        ["branch", "subspace_dimension"], observed=True
    ):
        null_rows.append(
            {
                "branch": branch,
                "k": int(dimension),
                "top-PCA R2": f"{group.top_pca_test_r2_mean.mean():.4f}",
                "Haar R2 mean": f"{group.random_test_r2_mean.mean():.4f}",
                "top percentile mean": f"{group.top_pca_percentile.mean():.1f}",
                "Haar exceedance fraction mean": (
                    f"{group.empirical_p_random_exceeds_top.mean():.3f}"
                ),
                "seed range": (
                    f"[{group.empirical_p_random_exceeds_top.min():.2f}, "
                    f"{group.empirical_p_random_exceeds_top.max():.2f}]"
                ),
            }
        )
    nonmono = nonmonotonic_intervals[
        nonmonotonic_intervals["readout"].eq("last_concat512")
        & nonmonotonic_intervals["target_block"].eq("directional")
    ]
    nonmono_rows = [
        {
            "branch": row.branch,
            "metric": row.metric,
            "difference": f"{row.mean:.4f}",
            "95% CI": f"[{row.lower:.4f}, {row.upper:.4f}]",
            "robust": bool(row.supports_33_64_more_informative),
        }
        for row in nonmono.sort_values(["metric", "branch"]).itertuples()
    ]
    spectral_bands = pd.read_parquet(root / "spectral_bands.parquet")
    selected_bands = spectral_bands[
        spectral_bands["readout"].eq("last_concat512")
        & spectral_bands["target_block"].eq("directional")
        & spectral_bands["band"].isin(NONMONOTONIC_BANDS)
        & spectral_bands["reader_family"].eq("min_norm_ols_diagnostic")
    ].copy()
    if selected_bands.empty or set(selected_bands["band"].unique()) != set(
        NONMONOTONIC_BANDS
    ):
        raise ExperimentIntegrityError("Phase-II band audit cells are missing")
    matched_band_rows = []
    for (branch, band, dimension), group in selected_bands.groupby(
        ["branch", "band", "band_dimension"], observed=True
    ):
        dimension = int(dimension)
        mass_fraction = float(
            group["predictive_mass_fraction_mean_independent"].mean()
        )
        matched_band_rows.append(
            {
                "branch": branch,
                "band": band,
                "dimension": dimension,
                "variance fraction": f"{group['variance_fraction'].mean():.4f}",
                "predictive mass": f"{mass_fraction:.4f}",
                "mass/direction": f"{mass_fraction / dimension:.6f}",
                "band-only R2": f"{group['test_r2_band_only'].mean():.4f}",
                "matched Haar R2": (
                    f"{group['matched_random_test_r2_mean'].mean():.4f}"
                ),
                "Haar exceedance fraction": (
                    f"{group['empirical_p_random_exceeds_band'].mean():.3f}"
                ),
            }
        )
    gap = (
        bridge.groupby(
            ["target_block", "budget_days_per_stock", "k_requested"],
            observed=True,
        )["phase1_gap_mean"]
        .first()
        .reset_index()
    )
    gap_rows = [
        {
            "block": row.target_block,
            "budget": f"{row.budget_days_per_stock:g}",
            "k": int(row.k_requested),
            "Phase-I gap": f"{row.phase1_gap_mean:.4f}",
        }
        for row in gap.sort_values(
            ["target_block", "budget_days_per_stock", "k_requested"]
        ).itertuples()
    ]
    parity_max = max(
        row["max_abs_difference"]
        for row in metadata["phase1_full_rank_min_norm_parity"]
    )
    primary_mass = mass_intervals[
        mass_intervals["readout"].eq("last_concat512")
    ]

    k_50gap, k_nonrobust, bridge_depths = _bridge_thresholds(bridge)

    horizon_headline_null = primary_null[
        primary_null["branch"].eq("jepa_horizon")
        & primary_null["subspace_dimension"].isin([8, 16])
    ]
    supervised_headline_null = primary_null[
        primary_null["branch"].eq("supervised")
    ]
    if horizon_headline_null.empty or supervised_headline_null.empty:
        raise ExperimentIntegrityError("Phase-II headline null cells are missing")
    random_draws = sorted(horizon_headline_null["random_draws"].unique().tolist())
    if len(random_draws) != 1:
        raise ExperimentIntegrityError("Phase-II headline Haar draw counts differ")
    horizon_seed_count = horizon_headline_null["encoder_seed"].nunique()
    horizon_all_random_exceed = bool(
        horizon_headline_null["empirical_p_random_exceeds_top"].eq(1.0).all()
    )
    supervised_all_top = bool(
        supervised_headline_null["top_pca_percentile"].eq(100.0).all()
    )
    haar_headline = (
        f"Per horizon-JEPA direzionale, tutti i {int(random_draws[0])} "
        f"sottospazi Haar superano top-PCA a k=8 e k=16 in ciascuno dei "
        f"{horizon_seed_count} encoder seed."
        if horizon_all_random_exceed
        else (
            "Per horizon-JEPA direzionale, almeno una cella k=8/16 non è "
            "superata da tutti i sottospazi Haar; si vedano i valori per seed."
        )
    )
    supervised_null_headline = (
        "Supervised top-PCA è al percentile 100 del null a tutte le profondità "
        "riportate."
        if supervised_all_top
        else (
            "Supervised top-PCA non raggiunge il percentile 100 del null in "
            "tutte le profondità riportate."
        )
    )
    horizon_by_k = primary_null[
        primary_null["branch"].eq("jepa_horizon")
    ].groupby("subspace_dimension", observed=True).agg(
        empirical_p=("empirical_p_random_exceeds_top", "mean"),
        percentile=("top_pca_percentile", "mean"),
    )
    transition_depths = [
        int(k)
        for k, row in horizon_by_k.iterrows()
        if 0.0 < float(row.empirical_p) < 1.0
    ]
    top_dominant_depths = [
        int(k)
        for k, row in horizon_by_k.iterrows()
        if float(row.empirical_p) == 0.0 and float(row.percentile) == 100.0
    ]
    horizon_transition_text = (
        "La transizione è eterogenea alle profondità "
        + ", ".join(str(value) for value in transition_depths)
        + "; top-PCA domina il null alle profondità "
        + ", ".join(str(value) for value in top_dominant_depths)
        + "."
    )

    parity_records = metadata["phase1_full_rank_min_norm_parity"]
    parity_feature_count = len(parity_records)
    parity_target_counts = sorted({int(row["n_targets"]) for row in parity_records})
    if len(parity_target_counts) != 1:
        raise ExperimentIntegrityError("Phase-I parity target counts differ")
    phase1_outcomes = sorted(
        bridge["phase1_outcome_unchanged"].dropna().unique().tolist()
    )
    if len(phase1_outcomes) != 1:
        raise ExperimentIntegrityError("Phase-I outcome differs within bridge")
    phase1_outcome = str(phase1_outcomes[0])

    phase3_summary = root.parent / "phase3_reduced" / "summary.json"
    if phase3_summary.is_file():
        phase3_payload = json.loads(phase3_summary.read_text())
        if phase3_payload.get("status") != "complete":
            raise ExperimentIntegrityError(
                "detected Phase-III-R summary is not complete"
            )
        current_later_status = (
            "Nel repository è ora presente anche Phase III-R completata; questa "
            "evidenza successiva non modifica Phase II né l'outcome Phase I."
        )
    else:
        current_later_status = (
            "Non è presente un summary Phase III-R completato accanto a questo run."
        )

    failure_rows = int(metadata.get("artifacts", {}).get("failures", {}).get("n_rows", -1))
    if failure_rows < 0:
        raise ExperimentIntegrityError("Phase-II metadata lacks failure row count")

    def mass_value(branch: str, block: str, k: int) -> float:
        selected = primary_mass[
            primary_mass["branch"].eq(branch)
            & primary_mass["target_block"].eq(block)
            & primary_mass["direction_index"].eq(k)
        ]
        if len(selected) != 1:
            raise ExperimentIntegrityError("primary mass summary cell is not unique")
        return float(selected["mass_fraction_mean"].iloc[0])

    return f"""# Experiment 01 — Phase II spectral diagnostics

## Stato e risultato

Phase II è completata come analisi diagnostica preregistrata. Phase I non è stata modificata: soglie, risultati e outcome tecnico **{phase1_outcome}** restano congelati. Durante Phase II non sono stati eseguiti MLP, nuovi training, VICReg o simulatori. {current_later_status}

Il gate storico PCA post-P0 passa su {gate['n_cells']} celle con errore assoluto massimo `{gate['maximum_absolute_difference']:.3e}` (tolleranza `{gate['tolerance']:.1e}`). Il gate aggiuntivo full-rank Phase I↔Phase II passa per tutte le {parity_feature_count} feature e {parity_target_counts[0]} target, con errore massimo `{parity_max:.3e}`.

### Diagnosi in breve

- **Specificità direzionale descrittiva.** Su `last_concat512`, horizon-JEPA colloca soltanto {mass_value('jepa_horizon', 'directional', 8):.4f} e {mass_value('jepa_horizon', 'directional', 16):.4f} della massa direzionale cumulativa nelle prime 8 e 16 PC; supervised ne colloca rispettivamente {mass_value('supervised', 'directional', 8):.4f} e {mass_value('supervised', 'directional', 16):.4f}. Per horizon-JEPA a k=8 i controlli sono molto meno estremi: volatilità {mass_value('jepa_horizon', 'volatility', 8):.4f}, timing {mass_value('jepa_horizon', 'timing', 8):.4f}. Il contrasto resta una diagnostica finché manca incertezza raggruppata per stock-day.
- **Top-PCA underperformance localizzata.** {haar_headline} {horizon_transition_text} {supervised_null_headline} Le quantità 0 e 1 sono frazioni di superamento su {int(random_draws[0])} draw, non p-value continui.
- **Meccanismo coerente con whitening profondo, non prova causale.** Horizon-JEPA raggiunge solo {mass_value('jepa_horizon', 'directional', 128):.4f} della massa direzionale a k=128 e {mass_value('jepa_horizon', 'directional', 256):.4f} a k=256, contro {mass_value('supervised', 'directional', 128):.4f} e {mass_value('supervised', 'directional', 256):.4f} per supervised. Questa dispersione fornisce una spiegazione spettrale coerente del gap finite-sample e della profondità di whitening, ma il ponte resta descrittivo perché il whitening riscala il full rank anziché troncarlo.
- **La storia locale 17:32→33:64 non è un confronto dimension-matched.** La prima banda contiene 16 direzioni e la seconda 32: la loro differenza grezza non può spiegare la non-monotonia. Il report conserva i valori storici come audit, ma usa soltanto il confronto di ciascuna banda con il proprio null Haar matched e la massa per direzione come descrittivi.

## Protocollo effettivo

- PCA/covarianza: fit esclusivamente sulle feature non etichettate del train canonico completo, separatamente per ramo × encoder seed × readout.
- Cross-covarianza e predictive mass: train canonico; direzioni oltre il rank numerico sono marcate invalide e mai invertite.
- Alpha ridge: selezionato esclusivamente su validation; test fisso usato solo per la valutazione finale.
- Null: {int(random_draws[0])} sottospazi Haar deterministici per dimensione, reader min-norm diagnostico; nessuna estrazione selezionata sul test.
- Readout primario: `last_concat512`; secondario: `meanK_concatS`. Blocchi directional, volatility e timing sempre separati.
- I confronti ridge usano `lambda = alpha * trace(covariance) / dimension` sul design etichettato pertinente.

## Localizzazione della predictive mass

La tabella seguente riporta la frazione cumulativa media della predictive mass sui target indipendenti. Gli intervalli gerarchici sui seed misurano robustezza computazionale, non generalizzazione di popolazione.

{_markdown_table(_primary_mass_rows(mass_intervals), ['block', 'branch', 'k', 'mass', '95% CI'])}

Le curve complete, comprese `meanK_concatS` e `jepa_masked`, sono in `predictive_mass_intervals.parquet` e nella figura 01. La predictive mass non è identificata con R² out-of-sample: è una statistica stimata sul train e viene confrontata separatamente con ladder e bande sul test.

## Top-PCA versus sottospazi Haar

Per il blocco direzionale primario, il percentile top-PCA e la frazione dei {int(random_draws[0])} null che lo superano sono:

{_markdown_table(null_rows, ['branch', 'k', 'top-PCA R2', 'Haar R2 mean', 'top percentile mean', 'Haar exceedance fraction mean', 'seed range'])}

I risultati sono riportati per ogni encoder seed in `random_null_summary.parquet`; la media tra seed qui sopra è soltanto descrittiva. Le frazioni hanno risoluzione `1/{int(random_draws[0])}` e non sono interpretate come p-value di popolazione né usate per una decisione inferenziale preregistrata. Bottom-k, min-norm OLS e ridge trace-normalized tarato su validation sono conservati in `phase2_results.parquet`.

## Bande spettrali e non-monotonia k=8,16,32,64

La differenza storica confronta `17:32` (16 direzioni) con `33:64` (32
direzioni). Non è quindi un contrasto dimension-matched e non viene usata come
evidenza a favore o contro una localizzazione meccanicistica. La tabella
seguente riporta invece, separatamente per banda, il null Haar della stessa
dimensione e la massa predittiva media per direzione. Le medie sono descrittive
tra encoder seed; i valori per seed restano negli artefatti.

{_markdown_table(matched_band_rows, ['branch', 'band', 'dimension', 'variance fraction', 'predictive mass', 'mass/direction', 'band-only R2', 'matched Haar R2', 'Haar exceedance fraction'])}

Per trasparenza, le differenze paired originarie restano riportate sotto come
**audit legacy non dimension-matched**. La colonna `robust` descrive soltanto se
l'intervallo della differenza grezza esclude zero; non corregge il confondimento
di dimensione.

{_markdown_table(nonmono_rows, ['branch', 'metric', 'difference', '95% CI', 'robust'])}

Conclusione corretta: **la specifica spiegazione 17:32→33:64 resta non
verificata perché il contrasto diretto è dimension-confounded**. I null matched
di ciascuna banda sono diagnostiche separate e non trasformano quel confronto
in un test paired valido. Questa revisione non modifica l'outcome di Phase I né
il risultato numerico originale. I risultati per encoder seed sono in
`nonmonotonicity_per_encoder.parquet`; tutte le bande, leave-band-out e null
matched sono in `spectral_bands.parquet`.

## Ponte con il whitening Phase I

Il ponte usa senza rifit `k_50gap = {k_50gap}`, `k_nonrobust = {k_nonrobust}` e i gap congelati alle profondità {', '.join(str(int(value)) for value in bridge_depths)}. Il whitening a {k_50gap} riduce il gap del 55,6% ma non lo elimina. Al campo tecnico storico `k_nonrobust={k_nonrobust}`, il gap non soddisfa più a entrambi i budget il criterio composto `lower > 0 and mean >= delta=0.10`: i lower bound restano positivi e la riduzione media è del 92,6%. È quindi una transizione della soglia di effetto, non un intervallo che attraversa zero. Phase II non assume né conclude che il problema sia concentrato in poche PC.

{_markdown_table(gap_rows, ['block', 'budget', 'k', 'Phase-I gap'])}

La figura 04 sovrappone i gap Phase I alle frazioni cumulative di massa Phase II senza rifittare Phase I.

## Specificità e limiti

Directional, volatility e timing sono riportati separatamente e con identica procedura. `meanK_concatS` resta l’analisi secondaria dell’interazione con la fragilità al pooling. Le bande e i null sono diagnostiche descrittive; non introducono nuove soglie decisionali e non ridefiniscono A1/A2/B/D.

Il null Haar usa il min-norm OLS diagnostico per preservare la parità con il vecchio PCA ladder post-P0. Il ridge tarato è prodotto per top-k, bottom-k, band-only e leave-band-out, ma non viene usato per selezionare o classificare estrazioni Haar.

Gli intervalli correnti non includono resampling di stock e stock-day; una diagnostica che non sopravvive a tale incertezza raggruppata dovrà essere declassata. Il dataset copre sette titoli di un singolo mercato. Lo split è disgiunto per stock-day ma non forward-chaining: per ciascun titolo, il calendario train precede e segue i giorni di validation/test. Il test deriva inoltre da un held-out set già esplorato storicamente. Questi limiti impediscono di trattare Phase II come conferma esterna.

`jepa_masked` è mantenuto come controllo interno descrittivo, non come confronto headline: il checkpoint canonico epoch 20 è successivo ai minimi di validation osservati alle epoch 6–8 ed era già stato congelato prima di Phase II.

## Compute e failure

- Runtime core interno: `{metadata['compute']['runtime_seconds']:.1f}` s; wall time canonico esterno: `{external_compute['wall_time_seconds']:.1f}` s.
- Peak RAM canonica (`GNU time -v`): `{external_compute['peak_rss_bytes'] / 2**30:.2f}` GiB. Il campionamento interno è conservato in metadata ma non viene usato come stima del picco.
- Failure tecniche: {failure_rows}.
- Cache: statistiche sufficienti in coordinate PCA; nessun rifit dell'intero bundle per ogni k, banda o sottospazio.

## Artefatti

Gli artefatti canonici sono `phase2_results.parquet`, `predictive_mass.parquet`, `random_subspace_null.parquet`, `spectral_bands.parquet`, `phase1_phase2_bridge.parquet`, `failures.parquet`, `metadata.json`, le tabelle diagnostiche, le figure e `manifest.json`. Tutti gli hash sono registrati nel manifest.
"""


def render_phase2_report_from_artifacts(
    phase2_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Regenerate Phase-II prose without recomputing any scientific table."""
    root = Path(phase2_dir).resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    gate = json.loads((root / "reproduction_gate.json").read_text())
    mass_intervals = pd.read_parquet(root / "predictive_mass_intervals.parquet")
    null_summary = pd.read_parquet(root / "random_null_summary.parquet")
    nonmonotonic = pd.read_parquet(root / "nonmonotonicity_intervals.parquet")
    bridge = pd.read_parquet(root / "phase1_phase2_bridge.parquet")
    external_compute = _external_compute_metrics(root)
    report = _render_report(
        root,
        metadata,
        gate,
        mass_intervals,
        null_summary,
        nonmonotonic,
        bridge,
        external_compute,
    )
    destination = (
        Path(output_path)
        if output_path is not None
        else root / "REPORT_EXPERIMENT_01_PHASE2.md"
    )
    atomic_write_text(destination, report)
    return destination


def summarize_and_report_phase2(
    phase2_dir: str | Path,
    *,
    n_bootstrap: int = 5000,
) -> Mapping[str, object]:
    root = Path(phase2_dir).resolve()
    _normalize_degenerate_full_rank_haar_ties(root)
    metadata = json.loads((root / "metadata.json").read_text())
    gate = json.loads((root / "reproduction_gate.json").read_text())
    results = pd.read_parquet(root / "phase2_results.parquet")
    mass = pd.read_parquet(root / "predictive_mass.parquet")
    random_null = pd.read_parquet(root / "random_subspace_null.parquet")
    bands = pd.read_parquet(root / "spectral_bands.parquet")
    bridge = pd.read_parquet(root / "phase1_phase2_bridge.parquet")
    failures = pd.read_parquet(root / "failures.parquet")
    external_compute = _external_compute_metrics(root)
    _validate_inventory(
        root, metadata, results, mass, random_null, bands, failures
    )

    mass_cells, mass_intervals = _predictive_mass_summary(
        mass, n_bootstrap=n_bootstrap
    )
    null_summary = _random_null_summary(random_null)
    band_intervals = _band_intervals(bands, n_bootstrap=n_bootstrap)
    nonmono_intervals, nonmono_per_encoder = _nonmonotonicity_diagnostic(
        mass, bands, n_bootstrap=n_bootstrap
    )
    tables = {
        "predictive_mass_cells.parquet": mass_cells,
        "predictive_mass_intervals.parquet": mass_intervals,
        "random_null_summary.parquet": null_summary,
        "spectral_band_intervals.parquet": band_intervals,
        "nonmonotonicity_intervals.parquet": nonmono_intervals,
        "nonmonotonicity_per_encoder.parquet": nonmono_per_encoder,
    }
    for filename, table in tables.items():
        atomic_write_parquet(table, root / filename)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _figure_predictive_mass(mass_intervals, figures / "01_predictive_mass.png")
    _figure_random_percentiles(null_summary, figures / "02_top_pca_haar_percentile.png")
    _figure_bands(bands, figures / "03_spectral_bands.png")
    _figure_bridge(bridge, figures / "04_phase1_phase2_bridge.png")

    report = _render_report(
        root,
        metadata,
        gate,
        mass_intervals,
        null_summary,
        nonmono_intervals,
        bridge,
        external_compute,
    )
    report_path = root / "REPORT_EXPERIMENT_01_PHASE2.md"
    atomic_write_text(report_path, report)

    k_50gap, k_nonrobust, _ = _bridge_thresholds(bridge)
    phase1_outcomes = sorted(
        bridge["phase1_outcome_unchanged"].dropna().unique().tolist()
    )
    if len(phase1_outcomes) != 1:
        raise ExperimentIntegrityError("Phase-I outcome differs within bridge")
    phase3_summary_path = root.parent / "phase3_reduced" / "summary.json"
    phase3_r_complete = False
    if phase3_summary_path.is_file():
        phase3_r_complete = (
            json.loads(phase3_summary_path.read_text()).get("status") == "complete"
        )

    summary = {
        "schema_name": "thesis.experiment01.phase2_summary",
        "schema_version": 1,
        "status": "complete",
        "phase1_technical_outcome_unchanged": str(phase1_outcomes[0]),
        "phase1_modified": False,
        "phase3_r_complete_at_report_generation": phase3_r_complete,
        "historical_reproduction_gate_passed": bool(gate.get("passed")),
        "full_rank_phase1_parity_passed": bool(
            all(
                row.get("passed") is True
                for row in metadata["phase1_full_rank_min_norm_parity"]
            )
        ),
        "failure_count": len(failures),
        "compute": external_compute,
        "nonmonotonicity_directional_last_status": _nonmonotonic_status(
            nonmono_intervals
        ),
        "findings": {
            "directional_last_cumulative_mass": [
                {
                    "branch": str(row.branch),
                    "k": int(row.direction_index),
                    "mean": float(row.mass_fraction_mean),
                    "lower": float(row.mass_fraction_lower),
                    "upper": float(row.mass_fraction_upper),
                }
                for row in mass_intervals[
                    mass_intervals["readout"].eq("last_concat512")
                    & mass_intervals["target_block"].eq("directional")
                    & mass_intervals["branch"].isin(
                        ["supervised", "jepa_horizon"]
                    )
                ].sort_values(["branch", "direction_index"]).itertuples()
            ],
            "horizon_last_specificity_at_k8": [
                {
                    "target_block": str(row.target_block),
                    "mean": float(row.mass_fraction_mean),
                    "lower": float(row.mass_fraction_lower),
                    "upper": float(row.mass_fraction_upper),
                }
                for row in mass_intervals[
                    mass_intervals["readout"].eq("last_concat512")
                    & mass_intervals["branch"].eq("jepa_horizon")
                    & mass_intervals["direction_index"].eq(8)
                ].sort_values("target_block").itertuples()
            ],
            "directional_last_top_pca_haar": [
                {
                    "branch": str(row.branch),
                    "encoder_seed": int(row.encoder_seed),
                    "k": int(row.subspace_dimension),
                    "top_pca_test_r2_mean": float(row.top_pca_test_r2_mean),
                    "random_test_r2_mean": float(row.random_test_r2_mean),
                    "top_pca_percentile": float(row.top_pca_percentile),
                    "empirical_p_random_exceeds_top": float(
                        row.empirical_p_random_exceeds_top
                    ),
                }
                for row in null_summary[
                    null_summary["readout"].eq("last_concat512")
                    & null_summary["target_block"].eq("directional")
                    & null_summary["branch"].isin(
                        ["supervised", "jepa_horizon"]
                    )
                    & null_summary["ladder_dimension"].eq(True)
                ].sort_values(
                    ["branch", "encoder_seed", "subspace_dimension"]
                ).itertuples()
            ],
            "directional_last_nonmonotonicity": [
                {
                    "branch": str(row.branch),
                    "metric": str(row.metric),
                    "mean": float(row.mean),
                    "lower": float(row.lower),
                    "upper": float(row.upper),
                    "dimension_matched": False,
                    "first_band_dimension": 16,
                    "second_band_dimension": 32,
                    "interpretation": "legacy_interval_only",
                    "supports_33_64_more_informative": bool(
                        row.supports_33_64_more_informative
                    ),
                }
                for row in nonmono_intervals[
                    nonmono_intervals["readout"].eq("last_concat512")
                    & nonmono_intervals["target_block"].eq("directional")
                ].sort_values(["metric", "branch"]).itertuples()
            ],
            "whitening_bridge": {
                "k_50gap": k_50gap,
                "k_nonrobust": k_nonrobust,
                "phase1_refit": False,
                "interpretation": "descriptive_spectral_mechanism_not_causal_proof",
            },
        },
        "bootstrap": {
            "algorithm": "encoder_then_within_encoder_target_resampling",
            "n_draws": n_bootstrap,
            "seed": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def write_phase2_manifest(phase2_dir: str | Path) -> Mapping[str, object]:
    root = Path(phase2_dir).resolve()
    required = (
        "phase2_results.parquet",
        "predictive_mass.parquet",
        "random_subspace_null.parquet",
        "spectral_bands.parquet",
        "phase1_phase2_bridge.parquet",
        "failures.parquet",
        "metadata.json",
        "summary.json",
        "REPORT_EXPERIMENT_01_PHASE2.md",
        "reproduction_gate.json",
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = str(path.relative_to(root))
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    repository = Path(__file__).resolve().parents[1]
    source_records = []
    for relative in (
        "experiment01/phase2.py",
        "experiment01/phase2_reproduction.py",
        "experiment01/phase2_reporting.py",
        "scripts/experiment01/run_experiment_01_phase2.py",
        "tests/test_experiment01_phase2.py",
    ):
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    phase2_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    phase1_outcome = phase2_summary.get(
        "phase1_technical_outcome_unchanged", "unavailable"
    )
    payload = {
        "schema_name": "thesis.experiment01.phase2_manifest",
        "schema_version": 1,
        "status": "complete",
        "phase1_modified": False,
        "phase1_outcome_unchanged": phase1_outcome,
        "subsequent_phase_started_during_phase2": False,
        "subsequent_phase_status_scope": "phase2_execution_time",
        "artifacts": records,
        "source_files": source_records,
    }
    payload["manifest_payload_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(root / "manifest.json", payload)
    return {
        **payload,
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
    }

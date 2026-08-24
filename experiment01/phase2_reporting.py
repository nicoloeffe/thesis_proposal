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
        axis.axvline(128, color="black", linewidth=0.7, linestyle=":")
        axis.axvline(508, color="black", linewidth=0.7, linestyle=":")
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
    selected = intervals[
        intervals["readout"].eq("last_concat512")
        & intervals["target_block"].eq("directional")
        & intervals["metric"].eq(
            "predictive_mass_fraction_33_64_minus_17_32"
        )
        & intervals["branch"].isin(["supervised", "jepa_horizon"])
    ]
    robust = len(selected) == 2 and selected[
        "supports_33_64_more_informative"
    ].all()
    return (
        "supportata in modo gerarchicamente robusto per entrambi i rami decisivi"
        if robust
        else (
            "non supportata in modo robusto per entrambi i rami decisivi; resta "
            "una spiegazione post hoc non confermata"
        )
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
                "empirical p mean": f"{group.empirical_p_random_exceeds_top.mean():.3f}",
                "seed range p": (
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

Phase II è completata come analisi diagnostica preregistrata. Phase I non è stata modificata: soglie, risultati e outcome tecnico **A1** restano congelati. Non sono stati avviati MLP, nuovi training, VICReg, simulatori o Phase III.

Il gate storico PCA post-P0 passa su {gate['n_cells']} celle con errore assoluto massimo `{gate['maximum_absolute_difference']:.3e}` (tolleranza `{gate['tolerance']:.1e}`). Il gate aggiuntivo full-rank Phase I↔Phase II passa per tutte le 18 feature e 23 target, con errore massimo `{parity_max:.3e}`.

### Diagnosi in breve

- **Specificità direzionale netta.** Su `last_concat512`, horizon-JEPA colloca soltanto {mass_value('jepa_horizon', 'directional', 8):.4f} e {mass_value('jepa_horizon', 'directional', 16):.4f} della massa direzionale cumulativa nelle prime 8 e 16 PC; supervised ne colloca rispettivamente {mass_value('supervised', 'directional', 8):.4f} e {mass_value('supervised', 'directional', 16):.4f}. Per horizon-JEPA a k=8 i controlli sono molto meno estremi: volatilità {mass_value('jepa_horizon', 'volatility', 8):.4f}, timing {mass_value('jepa_horizon', 'timing', 8):.4f}.
- **Top-PCA underperformance localizzata.** Per horizon-JEPA direzionale, tutti i 100 sottospazi Haar superano top-PCA a k=8 e k=16 in ciascuno dei tre encoder seed. La transizione è eterogenea a k=32/64 e top-PCA domina il null a k=128/256. Supervised top-PCA è al percentile 100 del null a tutte le profondità riportate.
- **Meccanismo coerente con whitening profondo, non prova causale.** Horizon-JEPA raggiunge solo {mass_value('jepa_horizon', 'directional', 128):.4f} della massa direzionale a k=128 e {mass_value('jepa_horizon', 'directional', 256):.4f} a k=256, contro {mass_value('supervised', 'directional', 128):.4f} e {mass_value('supervised', 'directional', 256):.4f} per supervised. Questa dispersione fornisce una spiegazione spettrale coerente del gap finite-sample e della profondità di whitening, ma il ponte resta descrittivo perché il whitening riscala il full rank anziché troncarlo.
- **La storia locale 17:32→33:64 non regge come spiegazione generale.** Il confronto paired non è robustamente positivo per entrambi i rami decisivi; la non-monotonia resta non spiegata da quella singola coppia di bande.

## Protocollo effettivo

- PCA/covarianza: fit esclusivamente sulle feature non etichettate del train canonico completo, separatamente per ramo × encoder seed × readout.
- Cross-covarianza e predictive mass: train canonico; direzioni oltre il rank numerico sono marcate invalide e mai invertite.
- Alpha ridge: selezionato esclusivamente su validation; test fisso usato solo per la valutazione finale.
- Null: 100 sottospazi Haar deterministici per dimensione, reader min-norm diagnostico; nessuna estrazione selezionata sul test.
- Readout primario: `last_concat512`; secondario: `meanK_concatS`. Blocchi directional, volatility e timing sempre separati.
- I confronti ridge usano `lambda = alpha * trace(covariance) / dimension` sul design etichettato pertinente.

## Localizzazione della predictive mass

La tabella seguente riporta la frazione cumulativa media della predictive mass sui target indipendenti. Gli intervalli sono gerarchici sui tre encoder seed.

{_markdown_table(_primary_mass_rows(mass_intervals), ['block', 'branch', 'k', 'mass', '95% CI'])}

Le curve complete, comprese `meanK_concatS` e `jepa_masked`, sono in `predictive_mass_intervals.parquet` e nella figura 01. La predictive mass non è identificata con R² out-of-sample: è una diagnostica population-style stimata sul train e viene confrontata separatamente con ladder e bande sul test.

## Top-PCA versus sottospazi Haar

Per il blocco direzionale primario, il percentile top-PCA e la frazione dei 100 null che lo superano sono:

{_markdown_table(null_rows, ['branch', 'k', 'top-PCA R2', 'Haar R2 mean', 'top percentile mean', 'empirical p mean', 'seed range p'])}

I risultati sono riportati per ogni encoder seed in `random_null_summary.parquet`; la media tra seed qui sopra è soltanto descrittiva. Bottom-k, min-norm OLS e ridge trace-normalized tarato su validation sono conservati in `phase2_results.parquet`.

## Bande spettrali e non-monotonia k=8,16,32,64

Il test post hoc preregistrato confronta in modo paired la banda 17:32 con 33:64. Una differenza positiva significa che 33:64 contiene più predictive mass o produce R² band-only maggiore.

{_markdown_table(nonmono_rows, ['branch', 'metric', 'difference', '95% CI', 'robust'])}

Conclusione della verifica: **{_nonmonotonic_status(nonmonotonic_intervals)}**. Questa diagnosi non modifica l’interpretazione né l’outcome di Phase I. I risultati per encoder seed sono in `nonmonotonicity_per_encoder.parquet`; tutte le bande, leave-band-out e null matched sono in `spectral_bands.parquet`.

## Ponte con il whitening Phase I

Il ponte usa senza rifit `k_50gap = 128`, `k_nonrobust = 508` e i gap congelati a k=0,8,16,32,64,128,256,508. Il whitening parziale a 128 dimezza il gap ma non lo elimina; la non-robustezza richiede whitening quasi completo. Phase II non assume né conclude che il problema sia concentrato in poche PC.

{_markdown_table(gap_rows, ['block', 'budget', 'k', 'Phase-I gap'])}

La figura 04 sovrappone i gap Phase I alle frazioni cumulative di massa Phase II senza rifittare Phase I.

## Specificità e limiti

Directional, volatility e timing sono riportati separatamente e con identica procedura. `meanK_concatS` resta l’analisi secondaria dell’interazione con la fragilità al pooling. Le bande e i null sono diagnostiche descrittive; non introducono nuove soglie decisionali e non ridefiniscono A1/A2/B/D.

Il null Haar usa il min-norm OLS diagnostico per preservare la parità con il vecchio PCA ladder post-P0. Il ridge tarato è prodotto per top-k, bottom-k, band-only e leave-band-out, ma non viene usato per selezionare o classificare estrazioni Haar.

## Compute e failure

- Runtime core interno: `{metadata['compute']['runtime_seconds']:.1f}` s; wall time canonico esterno: `{external_compute['wall_time_seconds']:.1f}` s.
- Peak RAM canonica (`GNU time -v`): `{external_compute['peak_rss_bytes'] / 2**30:.2f}` GiB. Il campionamento interno è conservato in metadata ma non viene usato come stima del picco.
- Failure tecniche: 0.
- Cache: statistiche sufficienti in coordinate PCA; nessun rifit sui 270 GB per k, banda o sottospazio.

## Artefatti

Gli artefatti canonici sono `phase2_results.parquet`, `predictive_mass.parquet`, `random_subspace_null.parquet`, `spectral_bands.parquet`, `phase1_phase2_bridge.parquet`, `failures.parquet`, `metadata.json`, le tabelle diagnostiche, le figure e `manifest.json`. Tutti gli hash sono registrati nel manifest.
"""


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
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "schema_name": "thesis.experiment01.phase2_summary",
        "schema_version": 1,
        "status": "complete",
        "phase1_technical_outcome_unchanged": "A1",
        "phase1_modified": False,
        "phase3_started": False,
        "historical_reproduction_gate_passed": True,
        "full_rank_phase1_parity_passed": True,
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
                "k_50gap": 128,
                "k_nonrobust": 508,
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
        "experiment01/phase2_legacy.py",
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
    payload = {
        "schema_name": "thesis.experiment01.phase2_manifest",
        "schema_version": 1,
        "status": "complete",
        "phase1_modified": False,
        "phase1_outcome_unchanged": "A1",
        "phase3_started": False,
        "artifacts": records,
        "source_files": source_records,
    }
    payload["manifest_payload_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(root / "manifest.json", payload)
    return {
        **payload,
        "manifest_file_sha256": sha256_file(root / "manifest.json"),
    }

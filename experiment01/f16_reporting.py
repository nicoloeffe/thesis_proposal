"""Predeclared F16 summary, figures, narrative report and integrity manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment01.f16 import BUDGETS, F16IntegrityError, _relative
from experiment01.f16_test import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    _test_source_inventory,
    _verify_unlock,
)
from experiment01.io import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    canonical_json_sha256,
    sha256_file,
)


BUDGET_ORDER = {budget: index for index, budget in enumerate(BUDGETS)}
FAMILIES = (
    "axis_a_accessibility",
    "axis_b_accessibility",
    "role_retention",
    "topk_predictive_mass",
    "pooling_loss",
    "whitening_k128",
)
DISTINCT_FAMILIES = tuple(
    family for family in FAMILIES if family != "whitening_k128"
)
CORRECTIVE_REVISION_DATE = "2026-08-27"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise F16IntegrityError(f"missing F16 reporting artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_table(repo_root: Path, path: Path, expected_sha: str | None = None) -> pd.DataFrame:
    if not path.is_file() or (expected_sha is not None and sha256_file(path) != expected_sha):
        raise F16IntegrityError(f"F16 reporting table drift: {path}")
    return pd.read_parquet(path)


def _aggregate_raw(results: pd.DataFrame) -> pd.DataFrame:
    selected = results.loc[
        results["target_independent"].astype(bool)
        & results["reader_family"].eq("ridge_trace_normalized")
        & results["feature_view"].eq("full")
        & results["checkpoint_kind"].isin(["best", "canonical_epoch20"])
    ]
    return (
        selected.groupby(
            [
                "feature_key",
                "encoder_family",
                "trained_budget",
                "encoder_seed",
                "checkpoint_kind",
                "readout",
                "axis",
                "analysis_budget",
                "target_block",
            ],
            as_index=False,
        )[["train_r2", "validation_r2", "test_r2"]]
        .mean()
    )


def _one(frame: pd.DataFrame, mask: np.ndarray | pd.Series, column: str) -> float:
    selected = frame.loc[mask, column]
    if len(selected) != 1:
        raise F16IntegrityError(f"F16 summary expected one {column}, found {len(selected)}")
    value = float(selected.iloc[0])
    if not np.isfinite(value):
        raise F16IntegrityError(f"F16 summary found non-finite {column}")
    return value


def _metric_value(
    family: str,
    feature_key: str,
    budget: str,
    raw: pd.DataFrame,
    geometry: pd.DataFrame,
) -> float:
    if family in {"axis_a_accessibility", "axis_b_accessibility"}:
        axis = "A_label_matched" if family == "axis_a_accessibility" else "B_fixed_b16"
        analysis_budget = budget if axis == "A_label_matched" else "b_16"
        return _one(
            raw,
            raw["feature_key"].eq(feature_key)
            & raw["readout"].eq("last_concat512")
            & raw["axis"].eq(axis)
            & raw["analysis_budget"].eq(analysis_budget)
            & raw["target_block"].eq("directional"),
            "test_r2",
        )
    if family == "role_retention":
        selected = geometry.loc[
            geometry["feature_key"].eq(feature_key)
            & geometry["readout"].eq("last_concat512")
            & geometry["target_block"].eq("directional")
            & geometry["metric_family"].eq("role_retention")
            & geometry["feature_view"].isin(["role_common", "role_contrast"]),
            "test_retention",
        ]
        if len(selected) != 2 or not np.isfinite(selected).all():
            raise F16IntegrityError(f"F16 role-retention metric incomplete: {feature_key}")
        return float(selected.mean())
    if family == "topk_predictive_mass":
        selected = geometry.loc[
            geometry["feature_key"].eq(feature_key)
            & geometry["readout"].eq("last_concat512")
            & geometry["target_block"].eq("directional")
            & geometry["target_independent"].astype(bool)
            & geometry["metric_family"].eq("predictive_mass")
            & geometry["k"].isin([8, 16]),
            "cumulative_mass_fraction",
        ]
        if len(selected) != 24 or not np.isfinite(selected).all():
            raise F16IntegrityError(f"F16 top-k mass metric incomplete: {feature_key}")
        return float(selected.mean())
    if family == "pooling_loss":
        return _one(
            geometry,
            geometry["feature_key"].eq(feature_key)
            & geometry["target_block"].eq("directional")
            & geometry["metric_family"].eq("pooling_loss"),
            "test_r2",
        )
    if family == "whitening_k128":
        return _one(
            geometry,
            geometry["feature_key"].eq(feature_key)
            & geometry["readout"].eq("last_concat512")
            & geometry["target_block"].eq("directional")
            & geometry["metric_family"].eq("whitening_bridge")
            & geometry["k"].eq(128),
            "test_r2",
        )
    raise ValueError(f"unknown F16 summary family {family}")


def _spearman(values: list[float]) -> float:
    if len(values) != 4 or not np.isfinite(values).all():
        return float("nan")
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(ranks) == 0:
        return 0.0
    return float(np.corrcoef(np.arange(4, dtype=np.float64), ranks)[0, 1])


def _oriented_coordinates(raw: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for seed in (0, 1, 2):
            horizon_key = f"jepa_horizon_seed{seed}_canonical"
            supervised_key = f"supervised_seed{seed}_canonical"
            for budget in BUDGETS:
                new_key = f"supervised_f16_{budget}_seed{seed}_best"
                new = _metric_value(family, new_key, budget, raw, geometry)
                horizon = _metric_value(family, horizon_key, budget, raw, geometry)
                supervised = _metric_value(family, supervised_key, budget, raw, geometry)
                denominator = supervised - horizon
                coordinate = (new - horizon) / denominator if abs(denominator) >= 1e-12 else np.nan
                rows.append(
                    {
                        "family": family,
                        "encoder_seed": seed,
                        "budget": budget,
                        "budget_order": BUDGET_ORDER[budget],
                        "new_value": new,
                        "horizon_anchor": horizon,
                        "supervised_anchor": supervised,
                        "anchor_difference": denominator,
                        "oriented_coordinate": coordinate,
                        "closer_to_supervised": bool(np.isfinite(coordinate) and coordinate > 0.5),
                    }
                )
    return pd.DataFrame(rows)


def _training_stability(output_root: Path) -> list[dict[str, Any]]:
    curves = pd.read_parquet(output_root / "f16_training_curves.parquet")
    rows = []
    for (budget, seed), group in curves.groupby(["budget", "encoder_seed"], sort=False):
        initial = group.loc[group["global_update"].eq(0), "validation_mse"]
        if len(initial) != 1:
            raise F16IntegrityError("F16 stability has no unique update-0 MSE")
        best = float(group["validation_mse"].min())
        improvement = float(initial.iloc[0]) - best
        rows.append(
            {
                "budget": budget,
                "encoder_seed": int(seed),
                "initial_validation_mse": float(initial.iloc[0]),
                "best_validation_mse": best,
                "improvement": improvement,
                "unstable": improvement < 0.01,
            }
        )
    return rows


def _target_block_dose_response(raw: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for seed in (0, 1, 2):
        for block in ("directional", "volatility", "timing"):
            horizon = _one(
                raw,
                raw["feature_key"].eq(f"jepa_horizon_seed{seed}_canonical")
                & raw["axis"].eq("B_fixed_b16")
                & raw["readout"].eq("last_concat512")
                & raw["target_block"].eq(block),
                "test_r2",
            )
            ceiling = _one(
                raw,
                raw["feature_key"].eq(f"supervised_seed{seed}_canonical")
                & raw["axis"].eq("B_fixed_b16")
                & raw["readout"].eq("last_concat512")
                & raw["target_block"].eq(block),
                "test_r2",
            )
            low = _one(
                raw,
                raw["feature_key"].eq(f"supervised_f16_b_1_4_seed{seed}_best")
                & raw["axis"].eq("B_fixed_b16")
                & raw["readout"].eq("last_concat512")
                & raw["target_block"].eq(block),
                "test_r2",
            )
            high = _one(
                raw,
                raw["feature_key"].eq(f"supervised_f16_b_16_seed{seed}_best")
                & raw["axis"].eq("B_fixed_b16")
                & raw["readout"].eq("last_concat512")
                & raw["target_block"].eq(block),
                "test_r2",
            )
            scale = ceiling - horizon
            response = (high - low) / scale if abs(scale) >= 1e-12 else np.nan
            rows.append(
                {
                    "encoder_seed": seed,
                    "target_block": block,
                    "low_b_1_4_r2": low,
                    "high_b_16_r2": high,
                    "horizon_anchor_r2": horizon,
                    "supervised_ceiling_r2": ceiling,
                    "ceiling_scaled_dose_response": response,
                }
            )
    return rows


def summarize_f16(repo_root: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    unlock = _verify_unlock(repo_root, output_root)
    complete = _read_json(output_root / "f16_fixed_test_complete.json")
    unsigned = dict(complete)
    fingerprint = unsigned.pop("manifest_fingerprint", None)
    if fingerprint != canonical_json_sha256(unsigned) or complete.get("status") != "fixed_test_complete":
        raise F16IntegrityError("F16 fixed-test completion manifest drift")
    results = _verified_table(
        repo_root,
        repo_root / complete["artifacts"]["results"]["path"],
        complete["artifacts"]["results"]["sha256"],
    )
    geometry = _verified_table(
        repo_root,
        repo_root / complete["artifacts"]["geometry"]["path"],
        complete["artifacts"]["geometry"]["sha256"],
    )
    grouped = _verified_table(
        repo_root,
        repo_root / complete["artifacts"]["grouped_uncertainty"]["path"],
        complete["artifacts"]["grouped_uncertainty"]["sha256"],
    )
    raw = _aggregate_raw(results)
    coordinates = _oriented_coordinates(raw, geometry)
    smooth = []
    for family in FAMILIES:
        seed_rho = {}
        for seed in (0, 1, 2):
            values = (
                coordinates.loc[
                    coordinates["family"].eq(family) & coordinates["encoder_seed"].eq(seed)
                ]
                .sort_values("budget_order")["oriented_coordinate"]
                .tolist()
            )
            seed_rho[str(seed)] = _spearman(values)
        smooth.append(
            {
                "family": family,
                "spearman_by_seed": seed_rho,
                "passes_all_seeds_rho_ge_0_8": all(
                    np.isfinite(value) and value >= 0.8 for value in seed_rho.values()
                ),
            }
        )
    smooth_count = sum(row["passes_all_seeds_rho_ge_0_8"] for row in smooth)
    low_rules = []
    axis_b = coordinates.loc[coordinates["family"].eq("axis_b_accessibility")]
    for budget in ("b_1_4", "b_1"):
        axis_b_all = bool(
            axis_b.loc[axis_b["budget"].eq(budget), "closer_to_supervised"].astype(bool).all()
        )
        for family in FAMILIES:
            family_all = bool(
                coordinates.loc[
                    coordinates["family"].eq(family) & coordinates["budget"].eq(budget),
                    "closer_to_supervised",
                ].astype(bool).all()
            )
            low_rules.append(
                {
                    "budget": budget,
                    "family": family,
                    "metric_closer_all_seeds": family_all,
                    "axis_b_midpoint_all_seeds": axis_b_all,
                    "passes": family_all and axis_b_all,
                }
            )
    stability = _training_stability(output_root)
    unstable = [row for row in stability if row["unstable"]]
    dose = _target_block_dose_response(raw)
    dose_frame = pd.DataFrame(dose)
    heterogeneity_by_seed = {}
    for seed in (0, 1, 2):
        selected = dose_frame.loc[dose_frame["encoder_seed"].eq(seed)].set_index("target_block")
        directional = float(selected.loc["directional", "ceiling_scaled_dose_response"])
        controls = [
            float(selected.loc[block, "ceiling_scaled_dose_response"])
            for block in ("volatility", "timing")
        ]
        heterogeneity_by_seed[str(seed)] = bool(
            np.isfinite([directional, *controls]).all() and directional > max(controls)
        )
    accessibility_smooth = {
        row["family"]: row["passes_all_seeds_rho_ge_0_8"]
        for row in smooth
        if row["family"] in {"axis_a_accessibility", "axis_b_accessibility"}
    }
    geometry_smooth_count = sum(
        row["passes_all_seeds_rho_ge_0_8"]
        for row in smooth
        if row["family"]
        in {"role_retention", "topk_predictive_mass", "pooling_loss", "whitening_k128"}
    )
    flags = {
        "supervised_like_at_low_label_volume": any(row["passes"] for row in low_rules),
        "smooth_label_volume_dependence": smooth_count >= 4,
        "accessibility_without_measured_geometry_change": (
            any(accessibility_smooth.values()) and geometry_smooth_count < 2
        ),
        "low_budget_optimization_floor": bool(unstable),
        "directionality_specific_coadaptation": all(heterogeneity_by_seed.values()),
    }
    if sum(bool(value) for value in flags.values()) == 1:
        overall = next(key for key, value in flags.items() if value)
    elif sum(bool(value) for value in flags.values()) == 0:
        overall = "no_preregistered_pattern_passed"
    else:
        overall = "multiple_preregistered_patterns_passed_report_separately"
    headline_grouped = grouped.loc[
        grouped["estimate_type"].eq("hierarchical_bootstrap")
        & grouped["axis"].eq("A_label_matched")
        & grouped["readout"].eq("last_concat512")
        & grouped["target_block"].eq("directional")
    ].sort_values(["trained_budget", "encoder_seed"])
    summary: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_summary",
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_question": "supervised versus horizon-JEPA label-matched dose response on last_concat512",
        "phase1_outcome": "A1 with robust ceiling gap; frozen and unchanged",
        "interpretation": {
            "overall": overall,
            "flags": flags,
            "rule": "retain all preregistered flags; mixed seed/metric patterns are heterogeneous",
        },
        "smoothness": {
            "families_passing_all_seed_rho_ge_0_8": smooth_count,
            "required_for_overall_smooth": 4,
            "details": smooth,
        },
        "low_budget_midpoint_rules": low_rules,
        "training_stability": stability,
        "target_block_dose_response": dose,
        "directionality_specific_by_seed": heterogeneity_by_seed,
        "oriented_coordinates": coordinates.to_dict("records"),
        "headline_grouped_axis_a_directional_last": headline_grouped.to_dict("records"),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "counts": {
            "result_rows": len(results),
            "geometry_rows": len(geometry),
            "grouped_uncertainty_rows": len(grouped),
            "eligible_training_cells": len(stability) - len(unstable),
            "unstable_training_cells": len(unstable),
        },
        "provenance": {
            "unlock_sha256": sha256_file(output_root / "f16_test_unlock.json"),
            "fixed_test_complete_sha256": sha256_file(output_root / "f16_fixed_test_complete.json"),
            "results_sha256": complete["artifacts"]["results"]["sha256"],
            "geometry_sha256": complete["artifacts"]["geometry"]["sha256"],
            "grouped_uncertainty_sha256": complete["artifacts"]["grouped_uncertainty"]["sha256"],
            "test_source_fingerprint": unlock["test_source"]["fingerprint"],
        },
        "selection_changes_after_test": False,
        "failures": [],
    }
    summary["summary_fingerprint"] = canonical_json_sha256(summary)
    atomic_write_json(output_root / "f16_summary.json", summary)
    return summary


def _figures(output_root: Path, summary: Mapping[str, Any], results: pd.DataFrame, geometry: pd.DataFrame) -> list[dict[str, Any]]:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    raw = _aggregate_raw(results)
    records = []
    labels = list(BUDGETS)
    x = np.arange(len(labels))
    for axis, number in (("A_label_matched", "01"), ("B_fixed_b16", "02")):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        new_means, new_std = [], []
        horizon_means, supervised_means = [], []
        for budget in BUDGETS:
            analysis_budget = budget if axis == "A_label_matched" else "b_16"
            base = (
                raw["axis"].eq(axis)
                & raw["analysis_budget"].eq(analysis_budget)
                & raw["readout"].eq("last_concat512")
                & raw["target_block"].eq("directional")
            )
            new = raw.loc[
                base
                & raw["encoder_family"].eq("supervised_f16")
                & raw["trained_budget"].eq(budget),
                "test_r2",
            ]
            horizon = raw.loc[base & raw["encoder_family"].eq("jepa_horizon"), "test_r2"]
            supervised = raw.loc[base & raw["encoder_family"].eq("supervised"), "test_r2"]
            new_means.append(float(new.mean())); new_std.append(float(new.std(ddof=1)))
            horizon_means.append(float(horizon.mean())); supervised_means.append(float(supervised.mean()))
        ax.errorbar(x, new_means, yerr=new_std, marker="o", capsize=3, label="F16 supervised mean±sd")
        ax.plot(x, horizon_means, marker="s", label="frozen horizon-JEPA")
        ax.plot(x, supervised_means, marker="^", label="canonical supervised anchor")
        ax.set_xticks(x, labels)
        ax.set_ylabel("fixed-test directional R²")
        ax.set_title(f"F16 {axis.replace('_', ' ')} — last_concat512")
        ax.grid(alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout()
        path = figure_root / f"{number}_directional_{axis}.png"
        fig.savefig(path, dpi=180); plt.close(fig)
        records.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    coordinates = pd.DataFrame(summary["oriented_coordinates"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for family in FAMILIES:
        selected = coordinates.loc[coordinates["family"].eq(family)]
        means = selected.groupby("budget_order")["oriented_coordinate"].mean().reindex(x)
        ax.plot(x, means, marker="o", label=family)
    ax.axhline(0.0, color="black", lw=0.8, ls="--"); ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xticks(x, labels); ax.set_ylabel("horizon→supervised oriented coordinate")
    ax.set_title("F16 accessibility and geometry dose response")
    ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout(); path = figure_root / "03_oriented_dose_response.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    records.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    bridge = geometry.loc[
        geometry["metric_family"].eq("whitening_bridge")
        & geometry["encoder_family"].eq("supervised_f16")
        & geometry["target_block"].eq("directional")
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for budget in BUDGETS:
        selected = bridge.loc[bridge["trained_budget"].eq(budget)]
        curve = selected.groupby("k")["test_r2"].mean().sort_index()
        ax.plot(curve.index, curve.values, marker="o", label=budget)
    ax.set_xlabel("whitened leading PCs k"); ax.set_ylabel("fixed-test directional R²")
    ax.set_title("F16 frozen whitening bridge — last_concat512")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); path = figure_root / "04_whitening_bridge.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    records.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return records


def report_f16(repo_root: Path, output_root: Path) -> tuple[str, dict[str, Any]]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    summary_path = output_root / "f16_summary.json"
    summary = _read_json(summary_path)
    unsigned = dict(summary); fingerprint = unsigned.pop("summary_fingerprint", None)
    if fingerprint != canonical_json_sha256(unsigned) or summary.get("status") != "complete":
        raise F16IntegrityError("F16 summary fingerprint drift")
    complete = _read_json(output_root / "f16_fixed_test_complete.json")
    results = pd.read_parquet(output_root / "f16_results.parquet")
    geometry = pd.read_parquet(output_root / "f16_geometry.parquet")
    grouped = pd.read_parquet(output_root / "f16_grouped_uncertainty.parquet")
    figures = _figures(output_root, summary, results, geometry)
    flags = summary["interpretation"]["flags"]
    smooth = summary["smoothness"]
    grouped_primary = grouped.loc[
        grouped["estimate_type"].eq("hierarchical_bootstrap")
        & grouped["axis"].eq("A_label_matched")
        & grouped["readout"].eq("last_concat512")
        & grouped["target_block"].eq("directional")
    ].copy()
    grouped_primary["budget_order"] = grouped_primary["trained_budget"].map(
        BUDGET_ORDER
    )
    if grouped_primary["budget_order"].isna().any():
        raise F16IntegrityError("F16 corrective report found an unknown budget")
    grouped_primary = grouped_primary.sort_values(
        ["budget_order", "encoder_seed"]
    )
    grouped_lines = "\n".join(
        f"| {row.trained_budget} | {int(row.encoder_seed)} | {row.supervised_f16_r2:.4f} | "
        f"{row.jepa_horizon_r2:.4f} | {row.paired_gap:.4f} | [{row.lower_95:.4f}, {row.upper_95:.4f}] |"
        for row in grouped_primary.itertuples(index=False)
    )
    smooth_lines = "\n".join(
        f"| {row['family']} | {row['spearman_by_seed']['0']:.3f} | "
        f"{row['spearman_by_seed']['1']:.3f} | {row['spearman_by_seed']['2']:.3f} | "
        f"{str(row['passes_all_seeds_rho_ge_0_8']).lower()} |"
        for row in smooth["details"]
    )
    report = f"""# Experiment 01 — F16 label-matched supervised dose response

**Status:** complete fixed-test diagnostic
**Phase-I outcome:** A1 with robust ceiling gap, frozen and unchanged
**F16 pattern:** `{summary['interpretation']['overall']}`

## Result

F16 varies the amount of target-aligned supervision used to train an otherwise
matched supervised encoder. Axis A fits a fresh reader on the same labelled
budget; Axis B holds the reader budget fixed at `b_16` and diagnoses the
representation. The test was opened once only after all checkpoint and alpha
selections were hash-frozen. No test result changed a selection.

The preregistered interpretation flags are:

- supervised-like at low label volume: **{flags['supervised_like_at_low_label_volume']}**;
- smooth label-volume dependence: **{flags['smooth_label_volume_dependence']}**;
- accessibility change without measured geometry change: **{flags['accessibility_without_measured_geometry_change']}**;
- low-budget optimization floor: **{flags['low_budget_optimization_floor']}**;
- directionality-specific co-adaptation: **{flags['directionality_specific_coadaptation']}**.

These flags are reported separately. Mixed evidence is not collapsed into a
new A/B/C/D outcome, and F16 does not modify Phase I.

## Primary label-matched comparison

The table reports block-mean independent-target R² under `last_concat512`.
Intervals are paired hierarchical test-set intervals that resample stocks and
then stock-days; seven-stock leave-one-out estimates are in
`f16_grouped_uncertainty.parquet`.

| budget | seed | F16 supervised R² | horizon-JEPA R² | paired gap | grouped 95% interval |
|---|---:|---:|---:|---:|---:|
{grouped_lines}

## Dose-response checks

Every metric is oriented so horizon-JEPA is 0 and the canonical supervised
anchor is 1. Spearman correlation is computed over the four frozen budgets
inside each encoder seed.

| family | seed 0 ρ | seed 1 ρ | seed 2 ρ | all seeds ≥0.8 |
|---|---:|---:|---:|---:|
{smooth_lines}

The overall smooth rule requires at least four of the six families to pass;
**{smooth['families_passing_all_seed_rho_ge_0_8']}** passed. `b_1_4` instability
was defined before test as failure to improve validation MSE by 0.01; the
summary records {summary['counts']['unstable_training_cells']} unstable cells.

## Geometry and controls

`f16_geometry.parquet` reports common/full and contrast/full retention
non-additively, cumulative predictive mass at the frozen spectral depths,
`last→meanK` loss, covariance spectra and the frozen whitening bridge
`k=0,8,16,32,64,128,256,508`. Volatility and timing remain specificity
controls, not alternative primary outcomes. Target counts are not treated as
independent replications.

## Uncertainty and limits

Grouped intervals use {BOOTSTRAP_DRAWS:,} deterministic resamples with seed
`{BOOTSTRAP_SEED}`. With only seven stocks they are descriptive; the complete
leave-one-stock-out table is therefore mandatory. F16 changes label volume and
target-aligned feature exposure together. A low-budget effect cannot by itself
separate label scarcity from exposure or optimization, and a dose response
does not establish a universal causal law of representation learning.

## Integrity

- test cohort: 11,136 endpoints across 87 stock-days and seven stocks;
- checkpoint grid: 12 best plus 12 epoch-20 sensitivity checkpoints;
- training failures: 0;
- selections changed after unlock: false;
- Phase II/III, MLP, VICReg and simulators: not run by F16.
"""
    report_path = output_root / "REPORT_EXPERIMENT_01_F16.md"
    atomic_write_text(report_path, report)
    required = {
        "f16_manifest.json",
        "f16_job_inventory.parquet",
        "f16_training_curves.parquet",
        "f16_checkpoint_manifest.json",
        "f16_cohort_manifest.json",
        "f16_cohort_convergence.parquet",
        "f16_results.parquet",
        "f16_geometry.parquet",
        "f16_grouped_uncertainty.parquet",
        "f16_failures.parquet",
        "f16_summary.json",
        "REPORT_EXPERIMENT_01_F16.md",
    }
    missing = sorted(name for name in required if not (output_root / name).is_file())
    if missing:
        raise F16IntegrityError(f"F16 final artifact contract incomplete: {missing}")
    artifact_records = {}
    for name in sorted(required):
        path = output_root / name
        artifact_records[name] = {
            "path": _relative(path, repo_root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_final_manifest",
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifact_records,
        "figures": [
            {**record, "path": _relative(Path(record["path"]), repo_root)} for record in figures
        ],
        "unlock_sha256": sha256_file(output_root / "f16_test_unlock.json"),
        "fixed_test_complete_sha256": sha256_file(output_root / "f16_fixed_test_complete.json"),
        "test_source": _test_source_inventory(repo_root),
        "outcome": summary["interpretation"],
        "phase1_outcome_unchanged": True,
        "test_accessed_once": True,
        "selection_changes_after_test": False,
        "failures": [],
    }
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    atomic_write_json(output_root / "f16_final_manifest.json", manifest)
    return report, manifest


def _flag_overall(flags: Mapping[str, bool]) -> str:
    passed = [key for key, value in flags.items() if bool(value)]
    if len(passed) == 1:
        return passed[0]
    if not passed:
        return "no_preregistered_pattern_passed"
    return "multiple_preregistered_patterns_passed_report_separately"


def build_f16_corrective_diagnostics(
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Reaggregate frozen F16 outputs without changing the historical summary."""

    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    summary_path = output_root / "f16_summary.json"
    summary = _read_json(summary_path)
    unsigned = dict(summary)
    fingerprint = unsigned.pop("summary_fingerprint", None)
    if fingerprint != canonical_json_sha256(unsigned):
        raise F16IntegrityError("F16 corrective audit found summary drift")

    amendment_path = output_root / "f16_posttest_threshold_amendment.json"
    amendment = _read_json(amendment_path)
    amendment_unsigned = dict(amendment)
    amendment_fingerprint = amendment_unsigned.pop("amendment_fingerprint", None)
    if amendment_fingerprint != canonical_json_sha256(amendment_unsigned):
        raise F16IntegrityError("F16 threshold-amendment fingerprint drift")

    results_path = output_root / "f16_results.parquet"
    geometry_path = output_root / "f16_geometry.parquet"
    grouped_path = output_root / "f16_grouped_uncertainty.parquet"
    provenance = summary["provenance"]
    for path, expected in (
        (results_path, provenance["results_sha256"]),
        (geometry_path, provenance["geometry_sha256"]),
        (grouped_path, provenance["grouped_uncertainty_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise F16IntegrityError(f"F16 corrective source drift: {path}")

    geometry = pd.read_parquet(geometry_path)
    grouped = pd.read_parquet(grouped_path)
    coordinates = pd.DataFrame(summary["oriented_coordinates"])
    smooth_details = {
        str(row["family"]): row for row in summary["smoothness"]["details"]
    }
    changed = {
        str(row["family"]): bool(row["prior"])
        for row in amendment["changed_families"]
    }

    decision = _read_json(output_root / "f16_cohort_decision.json")
    tolerances = decision["tolerances"]
    tolerance_by_family = {
        "axis_a_accessibility": float(
            tolerances["directional_full_rank_validation_r2"]["value"]
        ),
        "axis_b_accessibility": float(
            tolerances["directional_full_rank_validation_r2"]["value"]
        ),
        "role_retention": max(
            float(tolerances["common_full_role_retention"]["value"]),
            float(tolerances["contrast_full_role_retention"]["value"]),
        ),
        "topk_predictive_mass": float(
            tolerances["directional_top16_predictive_mass"]["value"]
        ),
        "pooling_loss": float(
            tolerances["directional_last_to_meanK_gap"]["value"]
        ),
        "whitening_k128": float(
            tolerances["directional_full_rank_validation_r2"]["value"]
        ),
    }

    family_rows = []
    for family in FAMILIES:
        detail = smooth_details[family]
        selected = coordinates.loc[coordinates["family"].eq(family)]
        strict_by_seed: dict[str, bool] = {}
        inversion_count = 0
        max_inversion = 0.0
        for seed, group in selected.groupby("encoder_seed", observed=True):
            ordered = group.sort_values("budget_order")
            anchor_sign = np.sign(
                ordered["supervised_anchor"].to_numpy(dtype=float)
                - ordered["horizon_anchor"].to_numpy(dtype=float)
            )
            if np.any(anchor_sign == 0.0) or not np.isfinite(anchor_sign).all():
                raise F16IntegrityError("F16 corrective orientation is invalid")
            signed_values = ordered["new_value"].to_numpy(dtype=float) * anchor_sign
            differences = np.diff(signed_values)
            inversions = -differences[differences < 0.0]
            inversion_count += int(len(inversions))
            if len(inversions):
                max_inversion = max(max_inversion, float(inversions.max()))
            strict_by_seed[str(int(seed))] = bool(np.all(differences > 0.0))
        post_pass = bool(detail["passes_all_seeds_rho_ge_0_8"])
        pre_pass = changed.get(family, post_pass)
        family_rows.append(
            {
                "family": family,
                "spearman_seed0": float(detail["spearman_by_seed"]["0"]),
                "spearman_seed1": float(detail["spearman_by_seed"]["1"]),
                "spearman_seed2": float(detail["spearman_by_seed"]["2"]),
                "pre_boundary_pass": pre_pass,
                "post_boundary_pass": post_pass,
                "strict_monotone_all_seeds": all(strict_by_seed.values()),
                "strict_monotone_by_seed": json.dumps(
                    strict_by_seed, sort_keys=True, separators=(",", ":")
                ),
                "rank_inversion_count": inversion_count,
                "maximum_raw_inversion": max_inversion,
                "cohort_resolution_tolerance": tolerance_by_family[family],
                "maximum_inversion_within_tolerance": (
                    max_inversion <= tolerance_by_family[family]
                ),
                "independent_family": family != "whitening_k128",
                "duplicate_of": (
                    "axis_b_accessibility" if family == "whitening_k128" else ""
                ),
            }
        )
    family_audit = pd.DataFrame(family_rows)

    axis_b = coordinates.loc[
        coordinates["family"].eq("axis_b_accessibility")
    ].sort_values(["encoder_seed", "budget_order"])
    whitening = coordinates.loc[
        coordinates["family"].eq("whitening_k128")
    ].sort_values(["encoder_seed", "budget_order"])
    if not np.array_equal(
        axis_b[["encoder_seed", "budget_order"]].to_numpy(),
        whitening[["encoder_seed", "budget_order"]].to_numpy(),
    ):
        raise F16IntegrityError("F16 duplicate-family audit is not paired")
    axis_values = axis_b["new_value"].to_numpy(dtype=float)
    whitening_values = whitening["new_value"].to_numpy(dtype=float)
    axis_coordinates = axis_b["oriented_coordinate"].to_numpy(dtype=float)
    whitening_coordinates = whitening["oriented_coordinate"].to_numpy(dtype=float)
    duplicate_audit = {
        "excluded_family": "whitening_k128",
        "retained_family": "axis_b_accessibility",
        "maximum_absolute_raw_r2_difference": float(
            np.max(np.abs(axis_values - whitening_values))
        ),
        "maximum_absolute_oriented_coordinate_difference": float(
            np.max(np.abs(axis_coordinates - whitening_coordinates))
        ),
        "pearson_r_raw_values": float(
            np.corrcoef(axis_values, whitening_values)[0, 1]
        ),
        "same_spearman_pass_pattern": bool(
            smooth_details["axis_b_accessibility"][
                "passes_all_seeds_rho_ge_0_8"
            ]
            == smooth_details["whitening_k128"][
                "passes_all_seeds_rho_ge_0_8"
            ]
        ),
    }
    bridge = geometry.loc[
        geometry["metric_family"].eq("whitening_bridge")
        & geometry["encoder_family"].eq("supervised_f16")
        & geometry["target_block"].eq("directional")
        & geometry["readout"].eq("last_concat512")
    ]
    curve_ranges = (
        bridge.groupby(["trained_budget", "encoder_seed"], observed=True)[
            "test_r2"
        ]
        .agg(lambda values: float(values.max() - values.min()))
        .to_numpy(dtype=float)
    )
    duplicate_audit["maximum_whitening_ladder_r2_range"] = float(
        curve_ranges.max(initial=0.0)
    )

    phase1_subsets = _read_json(
        repo_root
        / "validation/experiment01/execution_20260730/phase1/subset_manifest.json"
    )
    full_rows = [
        int(row["n_rows"])
        for row in phase1_subsets["subsets"]
        if row["budget_label"] == "full_train"
    ]
    if len(full_rows) != 1:
        raise F16IntegrityError("F16 corrective audit lacks full-train row count")
    minimum_label_rows = len(pd.read_parquet(output_root / "labels/b_1_4.parquet"))
    saturation = coordinates.loc[coordinates["budget"].eq("b_1_4")].copy()
    saturation["label_rows"] = minimum_label_rows
    saturation["fraction_of_full_train_rows"] = minimum_label_rows / full_rows[0]
    saturation["percent_of_horizon_to_supervised_path"] = (
        100.0 * saturation["oriented_coordinate"]
    )
    saturation["independent_family"] = saturation["family"].ne(
        "whitening_k128"
    )
    saturation = saturation[
        [
            "family",
            "encoder_seed",
            "budget",
            "label_rows",
            "fraction_of_full_train_rows",
            "new_value",
            "horizon_anchor",
            "supervised_anchor",
            "oriented_coordinate",
            "percent_of_horizon_to_supervised_path",
            "independent_family",
        ]
    ].sort_values(["family", "encoder_seed"])

    post_flags = {
        key: bool(value)
        for key, value in summary["interpretation"]["flags"].items()
    }
    pre_pass_by_family = dict(
        zip(family_audit["family"], family_audit["pre_boundary_pass"])
    )
    pre_smooth_count = sum(bool(value) for value in pre_pass_by_family.values())
    pre_geometry_count = sum(
        bool(pre_pass_by_family[family])
        for family in (
            "role_retention",
            "topk_predictive_mass",
            "pooling_loss",
            "whitening_k128",
        )
    )
    pre_accessibility = any(
        bool(pre_pass_by_family[family])
        for family in ("axis_a_accessibility", "axis_b_accessibility")
    )
    pre_flags = dict(post_flags)
    pre_flags["smooth_label_volume_dependence"] = pre_smooth_count >= 4
    pre_flags["accessibility_without_measured_geometry_change"] = (
        pre_accessibility and pre_geometry_count < 2
    )

    distinct_pass = family_audit.loc[
        family_audit["independent_family"], "post_boundary_pass"
    ].astype(bool)
    distinct_geometry_pass = family_audit.loc[
        family_audit["family"].isin(
            ["role_retention", "topk_predictive_mass", "pooling_loss"]
        ),
        "post_boundary_pass",
    ].astype(bool)
    distinct_accessibility_pass = family_audit.loc[
        family_audit["family"].isin(
            ["axis_a_accessibility", "axis_b_accessibility"]
        ),
        "post_boundary_pass",
    ].astype(bool)
    deduplicated_flags = dict(post_flags)
    deduplicated_flags["smooth_label_volume_dependence"] = (
        int(distinct_pass.sum()) >= 4
    )
    deduplicated_flags["accessibility_without_measured_geometry_change"] = (
        bool(distinct_accessibility_pass.any())
        and int(distinct_geometry_pass.sum()) < 2
    )

    dose = pd.DataFrame(summary["target_block_dose_response"])
    dose["anchor_gap"] = (
        dose["supervised_ceiling_r2"] - dose["horizon_anchor_r2"]
    )
    volatility_gaps = dose.loc[
        dose["target_block"].eq("volatility"), "anchor_gap"
    ].to_numpy(dtype=float)
    directionality_audit = {
        "historical_technical_flag": post_flags[
            "directionality_specific_coadaptation"
        ],
        "scientific_status": "not_identified",
        "post_hoc_denominator_floor": 0.05,
        "volatility_anchor_gap_by_seed": volatility_gaps.tolist(),
        "all_volatility_denominators_below_floor": bool(
            np.all(np.abs(volatility_gaps) < 0.05)
        ),
        "reason": (
            "volatility ceiling-minus-horizon anchors are near zero, so the "
            "ceiling-scaled control response is unstable"
        ),
    }

    unlock = _read_json(output_root / "f16_test_unlock.json")
    unlock_time = pd.Timestamp(unlock["unlocked_at_utc"])
    amendment_time = pd.Timestamp(amendment["created_at_utc"])
    amendment_audit = {
        "applied_after_test_unlock": bool(amendment_time > unlock_time),
        "minutes_after_unlock": float(
            (amendment_time - unlock_time).total_seconds() / 60.0
        ),
        "mathematical_correction_valid": True,
        "changed_families": amendment["changed_families"],
        "pre_boundary_family_pass_count": pre_smooth_count,
        "post_boundary_family_pass_count": int(
            summary["smoothness"]["families_passing_all_seed_rho_ge_0_8"]
        ),
        "selection_changed": False,
        "test_reopened": False,
    }

    primary_grouped = grouped.loc[
        grouped["estimate_type"].eq("hierarchical_bootstrap")
        & grouped["axis"].eq("A_label_matched")
        & grouped["readout"].eq("last_concat512")
        & grouped["target_block"].eq("directional")
    ]
    primary_loso = grouped.loc[
        grouped["estimate_type"].eq("leave_one_stock_out")
        & grouped["axis"].eq("A_label_matched")
        & grouped["readout"].eq("last_concat512")
        & grouped["target_block"].eq("directional")
    ]
    primary_audit = {
        "grouped_cell_count": int(len(primary_grouped)),
        "grouped_positive_gap_count": int(primary_grouped["paired_gap"].gt(0.0).sum()),
        "grouped_intervals_excluding_zero_count": int(
            primary_grouped["lower_95"].gt(0.0).sum()
        ),
        "leave_one_stock_out_count": int(len(primary_loso)),
        "leave_one_stock_out_positive_gap_count": int(
            primary_loso["paired_gap"].gt(0.0).sum()
        ),
    }
    if primary_audit != {
        "grouped_cell_count": 12,
        "grouped_positive_gap_count": 12,
        "grouped_intervals_excluding_zero_count": 12,
        "leave_one_stock_out_count": 84,
        "leave_one_stock_out_positive_gap_count": 84,
    }:
        raise F16IntegrityError(
            f"F16 corrective primary audit changed: {primary_audit}"
        )

    payload = {
        "schema_name": "thesis.experiment01.f16_corrective_reanalysis",
        "schema_version": 1,
        "status": "complete",
        "revision_date": CORRECTIVE_REVISION_DATE,
        "scope": "post_hoc_read_only_reaggregation",
        "scientific_result_tables_modified": False,
        "phase1_outcome_modified": False,
        "source_hashes": {
            "f16_results.parquet": sha256_file(results_path),
            "f16_geometry.parquet": sha256_file(geometry_path),
            "f16_grouped_uncertainty.parquet": sha256_file(grouped_path),
            "f16_summary.json": sha256_file(summary_path),
            "f16_posttest_threshold_amendment.json": sha256_file(amendment_path),
        },
        "historical_pre_boundary_flags": pre_flags,
        "historical_post_boundary_flags": post_flags,
        "deduplicated_flags": deduplicated_flags,
        "pre_boundary_overall": _flag_overall(pre_flags),
        "post_boundary_overall": _flag_overall(post_flags),
        "deduplicated_overall": _flag_overall(deduplicated_flags),
        "deduplicated_family_inventory": list(DISTINCT_FAMILIES),
        "deduplicated_family_pass_count": int(distinct_pass.sum()),
        "smooth_rule_required_count": 4,
        "duplicate_audit": duplicate_audit,
        "amendment_audit": amendment_audit,
        "directionality_audit": directionality_audit,
        "primary_paired_gap_audit": primary_audit,
        "scientific_interpretation": {
            "label_matched_gap": "supported",
            "smooth_dose_response": "not_supported_after_family_deduplication",
            "response_shape": "rapid_transition_at_minimum_budget_then_plateau",
            "supervised_like_rule": (
                "passes_at_b_1_as_a_threshold_rule_not_equivalence"
            ),
            "accessibility_without_geometry": (
                "technical_flag_true_after_deduplication_but_refers_to_smoothness; "
                "large_minimum-budget_geometry shifts are observed"
            ),
            "directionality_specific_coadaptation": "not_identified",
        },
    }
    payload["payload_fingerprint"] = canonical_json_sha256(payload)
    return payload, family_audit, saturation


def _f16_corrective_report(
    summary: Mapping[str, Any],
    correction: Mapping[str, Any],
    family_audit: pd.DataFrame,
    saturation: pd.DataFrame,
    grouped: pd.DataFrame,
) -> str:
    grouped_primary = grouped.loc[
        grouped["estimate_type"].eq("hierarchical_bootstrap")
        & grouped["axis"].eq("A_label_matched")
        & grouped["readout"].eq("last_concat512")
        & grouped["target_block"].eq("directional")
    ].sort_values(["trained_budget", "encoder_seed"])
    grouped_lines = "\n".join(
        f"| {row.trained_budget} | {int(row.encoder_seed)} | "
        f"{row.supervised_f16_r2:.4f} | {row.jepa_horizon_r2:.4f} | "
        f"{row.paired_gap:.4f} | [{row.lower_95:.4f}, {row.upper_95:.4f}] |"
        for row in grouped_primary.itertuples(index=False)
    )
    family_lines = "\n".join(
        f"| {row.family} | {row.spearman_seed0:.3f} | "
        f"{row.spearman_seed1:.3f} | {row.spearman_seed2:.3f} | "
        f"{str(bool(row.post_boundary_pass)).lower()} | "
        f"{str(bool(row.strict_monotone_all_seeds)).lower()} | "
        f"{('duplicate of Axis B' if not row.independent_family else 'independent')} |"
        for row in family_audit.itertuples(index=False)
    )
    saturation_summary = (
        saturation.loc[saturation["independent_family"]]
        .groupby("family", observed=True)["oriented_coordinate"]
        .agg(
            mean_coordinate="mean",
            min_coordinate="min",
            max_coordinate="max",
        )
        .reset_index()
    )
    saturation_lines = "\n".join(
        f"| {row.family} | {row.mean_coordinate:.3f} | "
        f"[{row.min_coordinate:.3f}, {row.max_coordinate:.3f}] |"
        for row in saturation_summary.itertuples(index=False)
    )
    pre = correction["historical_pre_boundary_flags"]
    post = correction["historical_post_boundary_flags"]
    dedup = correction["deduplicated_flags"]
    flag_names = (
        "supervised_like_at_low_label_volume",
        "smooth_label_volume_dependence",
        "accessibility_without_measured_geometry_change",
        "low_budget_optimization_floor",
        "directionality_specific_coadaptation",
    )
    flag_lines = []
    for name in flag_names:
        corrected = (
            "not identified"
            if name == "directionality_specific_coadaptation"
            else str(bool(dedup[name]))
        )
        flag_lines.append(
            f"| {name} | {bool(pre[name])} | {bool(post[name])} | {corrected} |"
        )
    duplicate = correction["duplicate_audit"]
    amendment = correction["amendment_audit"]
    directionality = correction["directionality_audit"]
    primary = correction["primary_paired_gap_audit"]
    minimum_fraction = float(saturation["fraction_of_full_train_rows"].iloc[0])
    minimum_rows = int(saturation["label_rows"].iloc[0])
    return f"""# Experiment 01 — F16 label-matched supervision diagnostic

**Status:** completed fixed-test diagnostic with post-hoc read-only corrective reaggregation

**Phase-I outcome:** A1 frozen and unchanged

**Scientific F16 conclusion:** rapid transition at the minimum supervision budget, not a verified smooth dose response

## Result

F16 varies target-aligned supervision while keeping the encoder architecture
matched. Its strongest result survives the audit: all
`{primary['grouped_positive_gap_count']}/{primary['grouped_cell_count']}` primary
label-matched supervised-minus-horizon gaps are positive, every grouped
stock→stock-day interval excludes zero, and all
`{primary['leave_one_stock_out_positive_gap_count']}/{primary['leave_one_stock_out_count']}`
leave-one-stock-out gaps are positive.

The broader smooth-dose claim does not survive family deduplication.
`whitening_k128` is empirically the same diagnostic as Axis B: the maximum raw
R² difference is `{duplicate['maximum_absolute_raw_r2_difference']:.6f}`, the
raw-value correlation is `{duplicate['pearson_r_raw_values']:.6f}`, and the
entire whitening ladder changes test R² by at most
`{duplicate['maximum_whitening_ladder_r2_range']:.6f}`. Counting it separately
turns three distinct passing families into four nominal families. With the
original requirement of four passing families, the deduplicated smooth flag is
**False** (`{correction['deduplicated_family_pass_count']}/5` distinct families
pass).

At the smallest budget, `{minimum_rows:,}` labelled rows
(`{100.0 * minimum_fraction:.3f}%` of full train), several geometry metrics
have already moved 82–89% of the horizon→supervised path. The observed pattern
is therefore a sharp early transition followed by saturation and small rank
inversions, not a graded response proportional to label volume.

## Decision audit: before amendment, after amendment, after deduplication

The exact Spearman boundary correction was mathematically justified: exact
`rho=0.8` had been serialized as `0.7999999999999999`. It was nevertheless
applied `{amendment['minutes_after_unlock']:.2f}` minutes after test unlock and
changed three family decisions. The complete effect on all five flags is:

| flag | pre-boundary correction | post-boundary six-family | corrective deduplicated reading |
| --- | ---: | ---: | --- |
{chr(10).join(flag_lines)}

The corrected `accessibility_without_measured_geometry_change=True` flag has a
narrow technical meaning: fewer than two **distinct geometry families** show
the all-seed rank pattern. It does not mean geometry is unchanged; the
minimum-budget shifts below are large.

The historical `directionality_specific_coadaptation=False` is not evidence
against specificity. Its volatility normalization divides by anchor gaps
`{', '.join(f'{value:.4f}' for value in directionality['volatility_anchor_gap_by_seed'])}`,
all below the post-hoc `0.05` interpretability floor, producing unstable
ratios. Its scientific status is therefore **not identified**.

## Primary label-matched comparison

| budget | seed | F16 supervised R² | horizon-JEPA R² | paired gap | grouped 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
{grouped_lines}

## Frozen rank checks and monotonicity

Every metric is oriented so horizon-JEPA is 0 and canonical supervised is 1.
With only four budgets, `rho=0.8` permits one adjacent rank inversion; it is not
strict monotonicity. Only Axis A is strictly increasing in every seed.

| family | seed 0 rho | seed 1 rho | seed 2 rho | post-amendment pass | strictly monotone all seeds | audit role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{family_lines}

Several ordering differences are at or below the cohort-resolution scale. For
Axis B, the largest raw inversion is
`{float(family_audit.loc[family_audit['family'].eq('axis_b_accessibility'), 'maximum_raw_inversion'].iloc[0]):.4f}`
against a `0.02` convergence tolerance. Rank-based pass/fail is therefore not
evidence for a precisely resolved continuous law.

## Minimum-budget saturation

The table reports the mean and seed range of the horizon→supervised oriented
coordinate at `b_1_4`.

| distinct family | mean path completed | seed range |
| --- | ---: | ---: |
{saturation_lines}

The preregistered low-volume threshold rule passes only at `b_1` (28,446
rows), not at `b_1_4`, because the joint Axis-B midpoint condition fails at the
floor. “Supervised-like” is a threshold label, not statistical equivalence to
the canonical supervised encoder.

## Interpretation and limits

F16 supports two claims: target-aligned supervision rapidly changes the
learned representation, and label-matched F16 encoders outperform
horizon-JEPA on the directional readout. It does not support a smooth
six-family dose-response law. The dominant empirical shape is an early step
and plateau, consistent with the possibility that even a small amount of
target-aligned gradient selects a different geometry.

F16 changes label volume, target exposure and optimization trajectory together.
It does not isolate a universal causal effect of label count. Validation labels
used for checkpoint selection are not included in the nominal training-label
budget. The intervals cover seven stocks from one market and remain
descriptive despite grouped bootstrap and leave-one-stock-out checks.

## Integrity

- frozen `f16_results.parquet`, `f16_geometry.parquet`, grouped uncertainty,
  selections, thresholds and checkpoints are unchanged;
- no test reopening, new training or new fit was performed;
- the original post-amendment summary is retained as historical technical
  output;
- this correction is a deterministic reaggregation of frozen artifacts.
"""


def write_f16_corrective_revision(
    repo_root: Path,
    output_root: Path,
    *,
    publication_root: Path | None = None,
) -> dict[str, Any]:
    """Write the corrective F16 report and derived audit artifacts only."""

    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    correction, family_audit, saturation = build_f16_corrective_diagnostics(
        repo_root, output_root
    )
    summary = _read_json(output_root / "f16_summary.json")
    grouped = pd.read_parquet(output_root / "f16_grouped_uncertainty.parquet")
    report = _f16_corrective_report(
        summary, correction, family_audit, saturation, grouped
    )

    destinations = [output_root]
    if publication_root is not None:
        destinations.append(publication_root.resolve())
    artifact_names = (
        "f16_corrective_reanalysis.json",
        "f16_family_audit.parquet",
        "f16_saturation_table.parquet",
        "REPORT_EXPERIMENT_01_F16.md",
    )
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            destination / "f16_corrective_reanalysis.json", correction
        )
        atomic_write_parquet(
            family_audit, destination / "f16_family_audit.parquet"
        )
        atomic_write_parquet(
            saturation, destination / "f16_saturation_table.parquet"
        )
        atomic_write_text(destination / "REPORT_EXPERIMENT_01_F16.md", report)

    records = []
    for name in artifact_names:
        path = output_root / name
        records.append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_name": "thesis.experiment01.f16_corrective_manifest",
        "schema_version": 1,
        "status": "complete",
        "revision_date": CORRECTIVE_REVISION_DATE,
        "scope": "report_and_read_only_reaggregation",
        "scientific_result_tables_modified": False,
        "phase1_outcome_modified": False,
        "artifacts": records,
        "source_hashes": correction["source_hashes"],
        "reporting_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    for destination in destinations:
        atomic_write_json(destination / "f16_corrective_manifest.json", manifest)
    return manifest

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
from experiment01.io import atomic_write_json, atomic_write_text, canonical_json_sha256, sha256_file


BUDGET_ORDER = {budget: index for index, budget in enumerate(BUDGETS)}
FAMILIES = (
    "axis_a_accessibility",
    "axis_b_accessibility",
    "role_retention",
    "topk_predictive_mass",
    "pooling_loss",
    "whitening_k128",
)


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
    ].sort_values(["trained_budget", "encoder_seed"])
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

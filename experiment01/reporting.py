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

from .io import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    sha256_file,
)
from .results import (
    adjacent_robust_low_budgets,
    hierarchical_interval,
    paired_gap_points,
)


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


def _require_finite(value: object, label: str) -> float:
    result = _float_or_nan(value)
    if not np.isfinite(result):
        raise ValueError(f"required report metric is missing or non-finite: {label}")
    return result


def _format_budget(value: float) -> str:
    return f"{float(value):g}"


def _format_budget_list(values: Iterable[float]) -> str:
    return ", ".join(f"`{_format_budget(value)}`" for value in values)


def _format_optional(value: object, digits: int = 4) -> str:
    number = _float_or_nan(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"


def _interval_zero_statement(lower: float, upper: float) -> str:
    if lower > 0.0 or upper < 0.0:
        return "The interval excludes zero."
    return "The interval includes zero."


def _load_json_object(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def _spectral_diagnostic_from_phase2(
    phase2_summary_path: Path,
) -> Mapping[str, object]:
    """Load the Phase-II headline from its completed machine summary.

    The Phase-I narrative may mention later spectral evidence, but it must not
    silently reuse literals from an older PCA ladder.  If a Phase-II summary is
    present, every field used here is therefore required and validated.
    """
    payload = _load_json_object(
        phase2_summary_path, label="Phase-II summary"
    )
    if payload.get("status") != "complete" or payload.get("failure_count") != 0:
        raise ValueError("detected Phase-II summary is not complete and failure-free")
    if payload.get("phase1_modified") is not False:
        raise ValueError("Phase-II summary does not attest that Phase I was unchanged")
    findings = payload.get("findings")
    if not isinstance(findings, Mapping):
        raise ValueError("Phase-II summary has no findings object")
    mass_rows = findings.get("directional_last_cumulative_mass")
    null_rows = findings.get("directional_last_top_pca_haar")
    if not isinstance(mass_rows, list) or not isinstance(null_rows, list):
        raise ValueError("Phase-II summary lacks spectral headline tables")

    headline_k = 8

    def mass_for(branch: str) -> Mapping[str, object]:
        selected = [
            row
            for row in mass_rows
            if row.get("branch") == branch and row.get("k") == headline_k
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Phase-II summary needs one {branch} mass row at k={headline_k}"
            )
        row = selected[0]
        return {
            "mean": _require_finite(row.get("mean"), f"{branch} top-k mass"),
            "lower": _require_finite(row.get("lower"), f"{branch} mass lower"),
            "upper": _require_finite(row.get("upper"), f"{branch} mass upper"),
        }

    def null_for(branch: str) -> Mapping[str, object]:
        selected = [
            row
            for row in null_rows
            if row.get("branch") == branch and row.get("k") == headline_k
        ]
        if not selected:
            raise ValueError(
                f"Phase-II summary has no {branch} Haar rows at k={headline_k}"
            )
        probabilities = np.asarray(
            [
                _require_finite(
                    row.get("empirical_p_random_exceeds_top"),
                    f"{branch} Haar empirical p",
                )
                for row in selected
            ],
            dtype=float,
        )
        return {
            "n_encoder_seeds": len(selected),
            "empirical_p_mean": float(probabilities.mean()),
            "empirical_p_min": float(probabilities.min()),
            "empirical_p_max": float(probabilities.max()),
        }

    return {
        "available": True,
        "source_path": str(phase2_summary_path),
        "source_sha256": sha256_file(phase2_summary_path),
        "k": headline_k,
        "jepa_horizon": {
            "predictive_mass": mass_for("jepa_horizon"),
            "haar": null_for("jepa_horizon"),
        },
        "supervised": {
            "predictive_mass": mass_for("supervised"),
            "haar": null_for("supervised"),
        },
    }


def _later_phase_context(results_path: Path) -> Mapping[str, object]:
    execution_root = results_path.resolve().parent.parent
    phase2_path = execution_root / "phase2" / "summary.json"
    spectral: Mapping[str, object]
    if phase2_path.is_file():
        spectral = _spectral_diagnostic_from_phase2(phase2_path)
        phase2 = {
            "status": "complete",
            "summary_path": str(phase2_path),
            "summary_sha256": sha256_file(phase2_path),
        }
    else:
        spectral = {"available": False, "reason": "Phase-II summary not present"}
        phase2 = {"status": "not_present"}

    phase3_path = execution_root / "phase3_reduced" / "summary.json"
    if phase3_path.is_file():
        phase3_payload = _load_json_object(
            phase3_path, label="Phase-III-R summary"
        )
        if phase3_payload.get("status") != "complete":
            raise ValueError("detected Phase-III-R summary is not complete")
        phase3 = {
            "status": "complete",
            "protocol_variant": phase3_payload.get("protocol_variant"),
            "technical_outcome": (
                phase3_payload.get("outcome", {}).get("outcome")
                if isinstance(phase3_payload.get("outcome"), Mapping)
                else None
            ),
            "summary_path": str(phase3_path),
            "summary_sha256": sha256_file(phase3_path),
        }
    else:
        phase3 = {"status": "not_present"}
    return {"phase2": phase2, "phase3_r": phase3, "spectral": spectral}


def _pooling_diagnostic(raw_uncertainty: pd.DataFrame) -> Mapping[str, object]:
    selected = raw_uncertainty[
        raw_uncertainty["target_block"].eq("directional")
        & raw_uncertainty["reader_family"].eq("ridge_raw_tuned_alpha")
        & raw_uncertainty["feature_view"].eq("full_rank_raw")
        & raw_uncertainty["budget_kind"].eq("full_train")
        & raw_uncertainty["branch"].isin(["jepa_horizon", "supervised"])
        & raw_uncertainty["readout"].isin(
            ["last_concat512", "meanK_concatS"]
        )
    ]
    values: dict[str, dict[str, float]] = {}
    for branch in ("jepa_horizon", "supervised"):
        values[branch] = {}
        for readout in ("last_concat512", "meanK_concatS"):
            rows = selected[
                selected["branch"].eq(branch)
                & selected["readout"].eq(readout)
            ]
            if len(rows) != 1:
                return {
                    "available": False,
                    "reason": (
                        "complete two-branch last/meanK full-budget cells "
                        "are not present"
                    ),
                }
            values[branch][readout] = _require_finite(
                rows.iloc[0]["raw_r2_mean"], f"{branch}/{readout} raw R2"
            )
    return {
        "available": True,
        "metric": "mean full-budget directional test R2",
        "reader_family": "ridge_raw_tuned_alpha",
        "values": values,
    }


def _critical_budget_metrics(
    results: pd.DataFrame, decisive_budgets: Iterable[float]
) -> pd.DataFrame:
    selected = results[
        results["readout"].eq("last_concat512")
        & results["reader_family"].eq("ridge_raw_tuned_alpha")
        & results["feature_view"].eq("full_rank_raw")
        & results["budget_days_per_stock"].isin(tuple(decisive_budgets))
        & results["target_independent"].eq(True)
        & results["fit_status"].eq("ok")
        & results["branch"].isin(["jepa_horizon", "supervised"])
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["target_block", "budget_days_per_stock", "branch"]
    for key, group in selected.groupby(keys, observed=True):
        block, budget, branch = key
        eligible = group[
            group["ceiling_eligible"].eq(True)
            & group["normalized_recovery"].map(np.isfinite)
        ]
        if eligible.empty:
            continue
        ceiling_cells = eligible[
            [
                "encoder_seed",
                "target_name",
                "full_budget_test_r2",
                "ceiling_eligible",
            ]
        ].drop_duplicates()
        counts = ceiling_cells.groupby("encoder_seed", observed=True)[
            "target_name"
        ].nunique()
        raw_values = group["test_r2"].to_numpy(dtype=float)
        recovery_values = eligible["normalized_recovery"].to_numpy(dtype=float)
        ceiling_values = ceiling_cells["full_budget_test_r2"].to_numpy(
            dtype=float
        )
        if not (
            np.isfinite(raw_values).all()
            and np.isfinite(recovery_values).all()
            and np.isfinite(ceiling_values).all()
        ):
            raise ValueError("critical-budget table contains non-finite metrics")
        rows.append(
            {
                "target_block": str(block),
                "budget_days_per_stock": float(budget),
                "branch": str(branch),
                "raw_test_r2_mean": float(raw_values.mean()),
                "raw_test_r2_median": float(np.median(raw_values)),
                "negative_raw_test_r2_fraction": float(
                    np.mean(raw_values < 0.0)
                ),
                "full_budget_ceiling_mean": float(ceiling_values.mean()),
                "full_budget_ceiling_min": float(ceiling_values.min()),
                "full_budget_ceiling_max": float(ceiling_values.max()),
                "normalized_recovery_mean": float(recovery_values.mean()),
                "normalized_recovery_median": float(
                    np.median(recovery_values)
                ),
                "normalized_recovery_min": float(recovery_values.min()),
                "normalized_recovery_max": float(recovery_values.max()),
                "eligible_target_count_min_per_encoder": int(counts.min()),
                "eligible_target_count_max_per_encoder": int(counts.max()),
                "n_raw_cells": int(len(raw_values)),
                "n_recovery_cells": int(len(recovery_values)),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _raw_specificity_from_critical_budgets(
    critical_metrics: pd.DataFrame,
) -> Mapping[str, object]:
    """Derive raw-scale arm gaps from the already frozen decisive cells."""
    required = {
        "target_block",
        "budget_days_per_stock",
        "branch",
        "raw_test_r2_mean",
    }
    if critical_metrics.empty or not required.issubset(critical_metrics.columns):
        return {"available": False, "reason": "critical-budget cells unavailable"}
    pivot = critical_metrics.pivot_table(
        index=["target_block", "budget_days_per_stock"],
        columns="branch",
        values="raw_test_r2_mean",
        aggfunc="first",
    )
    if not {"jepa_horizon", "supervised"}.issubset(pivot.columns):
        return {"available": False, "reason": "paired arm cells unavailable"}
    pivot = pivot.dropna(subset=["jepa_horizon", "supervised"]).copy()
    pivot["raw_gap"] = pivot["supervised"] - pivot["jepa_horizon"]
    gaps = {
        str(block): float(group["raw_gap"].mean())
        for block, group in pivot.reset_index().groupby(
            "target_block", observed=True
        )
    }
    directional = _float_or_nan(gaps.get("directional"))
    ratios: dict[str, float] = {}
    for control in ("volatility", "timing"):
        value = _float_or_nan(gaps.get(control))
        ratios[f"directional_over_{control}"] = (
            directional / value
            if np.isfinite(directional) and np.isfinite(value) and value != 0.0
            else float("nan")
        )
    return {
        "available": bool(gaps),
        "definition": (
            "mean across decisive budgets of supervised minus horizon-JEPA "
            "raw test R2"
        ),
        "raw_gap_by_target_block": gaps,
        **ratios,
    }


def _gap_summary_subset(
    frame: pd.DataFrame,
    *,
    reader_family: str,
    feature_view: str,
) -> pd.DataFrame:
    return frame[
        frame["readout"].eq("last_concat512")
        & frame["target_block"].eq("directional")
        & frame["reader_family"].eq(reader_family)
        & frame["feature_view"].eq(feature_view)
    ].copy()


def _all_encoder_directional_gaps_positive(
    recovery: pd.DataFrame, budgets: Iterable[float]
) -> bool:
    selected = recovery[
        recovery["readout"].eq("last_concat512")
        & recovery["target_block"].eq("directional")
        & recovery["reader_family"].eq("ridge_raw_tuned_alpha")
        & recovery["feature_view"].eq("full_rank_raw")
    ]
    paired = paired_gap_points(selected)
    paired = paired[
        paired["budget_days_per_stock"].astype(float).isin(
            [float(value) for value in budgets]
        )
    ]
    if paired.empty:
        return False
    means = paired.groupby("encoder_seed", observed=True)[
        "normalized_gap"
    ].mean()
    return len(means) >= 2 and bool((means > 0.0).all())


def _classify_frozen_gap_sensitivity(
    gap_summary: pd.DataFrame,
    recovery: pd.DataFrame,
    *,
    expected_delta: float | None = None,
    large_sample_ceiling_meaningful: bool,
    ceiling_gap_robust: bool,
) -> Mapping[str, object]:
    """Reapply the frozen taxonomy to serialized gap-summary artifacts.

    This performs no bootstrap and no model fit.  It exists so the mandatory
    delta sensitivity is reported from the already serialized summaries rather
    than silently recomputed during report generation.
    """
    finite_delta = gap_summary["delta"].dropna().astype(float).unique()
    if len(finite_delta) == 0 and expected_delta is not None:
        delta = float(expected_delta)
    elif len(finite_delta) == 1:
        delta = float(finite_delta[0])
        if expected_delta is not None and not np.isclose(delta, expected_delta):
            raise ValueError("gap sensitivity artifact delta does not match its path")
    else:
        raise ValueError("gap sensitivity artifact must contain exactly one delta")
    native = _gap_summary_subset(
        gap_summary,
        reader_family="ridge_raw_tuned_alpha",
        feature_view="full_rank_raw",
    )
    adjacent = adjacent_robust_low_budgets(native)
    details: dict[str, object] = {
        "delta": delta,
        "native_adjacent_robust_pairs": [list(pair) for pair in adjacent],
        "k_50gap": None,
        "k_nonrobust": None,
    }
    if adjacent and large_sample_ceiling_meaningful:
        decisive = tuple(float(value) for value in adjacent[0])
        native_decisive = native[
            native["budget_days_per_stock"].astype(float).isin(decisive)
        ]
        baseline = float(native_decisive["mean"].mean())
        whiten = _gap_summary_subset(
            gap_summary,
            reader_family="ridge_whiten_topk_tuned_alpha",
            feature_view="full_rank_whiten_topk",
        )
        candidates: list[dict[str, object]] = []
        for requested_k, group in whiten.groupby(
            "whiten_k_requested", dropna=True, observed=True
        ):
            same = group[
                group["budget_days_per_stock"].astype(float).isin(decisive)
            ]
            if len(same) != len(decisive):
                continue
            mean_gap = float(same["mean"].mean())
            robust = same["robust"].astype(bool)
            candidates.append(
                {
                    "k": int(requested_k),
                    "mean_gap": mean_gap,
                    "reduction_fraction": (
                        float("nan")
                        if baseline == 0.0
                        else 1.0 - mean_gap / baseline
                    ),
                    "all_nonrobust": bool((~robust).all()),
                    "all_robust": bool(robust.all()),
                }
            )
        candidates.sort(key=lambda row: int(row["k"]))
        k_50gap = next(
            (
                int(row["k"])
                for row in candidates
                if float(row["reduction_fraction"]) >= 0.5
            ),
            None,
        )
        k_nonrobust = next(
            (
                int(row["k"])
                for row in candidates
                if bool(row["all_nonrobust"])
            ),
            None,
        )
        details.update(
            {
                "decisive_budgets": list(decisive),
                "k_50gap": k_50gap,
                "k_nonrobust": k_nonrobust,
            }
        )
        positive_by_encoder = _all_encoder_directional_gaps_positive(
            recovery, decisive
        )
        if k_50gap is not None and k_nonrobust is not None and positive_by_encoder:
            return {"outcome": "A1", **details}

        maximum = candidates[-1] if candidates else None
        ols = _gap_summary_subset(
            gap_summary,
            reader_family="min_norm_ols_raw",
            feature_view="full_rank_raw",
        )
        ols_decisive = ols[
            ols["budget_days_per_stock"].astype(float).isin(decisive)
        ]
        persistent_ols = len(ols_decisive) == len(decisive) and bool(
            ols_decisive["robust"].astype(bool).all()
        )
        if (
            maximum is not None
            and bool(maximum["all_robust"])
            and persistent_ols
            and positive_by_encoder
        ):
            return {"outcome": "A2", **details}
    if not adjacent and ceiling_gap_robust:
        return {"outcome": "B", **details}
    return {"outcome": "D", **details}


def _directional_nonmonotonicity_text(differences: pd.DataFrame) -> str:
    if differences.empty or "target_block" not in differences:
        return "No complete directional adjacent-depth diagnostic is available."
    selected = differences[differences["target_block"].eq("directional")]
    if selected.empty:
        return "No complete directional adjacent-depth diagnostic is available."
    clauses = []
    signs = set()
    for row in selected.sort_values("from_k").itertuples():
        mean = _require_finite(row.mean, "directional adjacent-depth mean")
        lower = _require_finite(row.lower, "directional adjacent-depth lower")
        upper = _require_finite(row.upper, "directional adjacent-depth upper")
        sign = "positive" if mean > 0.0 else "negative" if mean < 0.0 else "zero"
        signs.add(sign)
        interval = "excludes" if lower > 0.0 or upper < 0.0 else "includes"
        clauses.append(
            f"`{int(row.from_k)}→{int(row.to_k)}` has a {sign} point change "
            f"of `{mean:.4f}` and its interval {interval} zero"
        )
    pattern = (
        "The alternating point-estimate signs establish local non-monotonicity "
        "in the inspected grid."
        if "positive" in signs and "negative" in signs
        else "The point-estimate signs do not establish local non-monotonicity."
    )
    return "; ".join(clauses) + f". {pattern}"


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
    if gate_path is None:
        repository_gate = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "experiment01"
            / "REPRODUCTION_GATE_EXPERIMENT_01.json"
        )
        if repository_gate.is_file():
            gate_path = repository_gate
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
    summary_root = Path(summary_dir)
    summary_path = summary_root / "summary.json"
    summary = _load_json_object(summary_path, label="Phase-I technical summary")
    outcome = summary.get("directional_last_concat512_outcome")
    if not isinstance(outcome, Mapping):
        raise ValueError("Phase-I summary lacks the directional outcome record")
    decisive_budgets = tuple(
        _require_finite(value, "decisive budget")
        for value in outcome.get("decisive_budgets", ())
    )
    adjacent_pairs = outcome.get("native_adjacent_robust_pairs", ())
    low_budgets = sorted(
        {
            _require_finite(value, "low-budget grid value")
            for pair in adjacent_pairs
            for value in pair
        }
    )
    if not low_budgets:
        low_budgets = list(decisive_budgets)
    later_phases = _later_phase_context(results_path)
    recovery = pd.read_parquet(summary_root / "block_recovery_points.parquet")
    raw = pd.read_parquet(summary_root / "raw_block_points.parquet")
    recovery_u = pd.read_parquet(summary_root / "recovery_uncertainty.parquet")
    raw_u = pd.read_parquet(summary_root / "raw_uncertainty.parquet")
    gap_paths = {
        0.05: summary_root / "gap_summary_delta_005.parquet",
        0.10: summary_root / "gap_summary_delta_010.parquet",
        0.15: summary_root / "gap_summary_delta_015.parquet",
    }
    gap_sensitivity_frames = {
        delta: pd.read_parquet(path) for delta, path in gap_paths.items()
    }
    gap = gap_sensitivity_frames[0.10]
    result_columns = [
        "branch",
        "encoder_seed",
        "readout",
        "target_block",
        "target_name",
        "target_independent",
        "budget_kind",
        "budget_days_per_stock",
        "budget_stock_day_equivalents",
        "n_rows_over_dim",
        "subsample_seed",
        "feature_view",
        "whiten_k_requested",
        "reader_family",
        "alpha",
        "lambda_absolute",
        "test_r2",
        "full_budget_test_r2",
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
        & gap["budget_days_per_stock"].isin(low_budgets)
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
    surface_source_rows = len(surface)
    source_families = sorted(surface["reader_family"].dropna().unique().tolist())
    surface = (
        surface.groupby(
            ["branch", "budget_stock_day_equivalents", "alpha"],
            observed=True,
        )["normalized_recovery"]
        .mean()
        .unstack("branch")
        .reset_index()
    )
    required_surface_columns = {"supervised", "jepa_horizon"}
    paired_surface = pd.DataFrame()
    if required_surface_columns.issubset(surface.columns):
        paired_surface = surface.dropna(
            subset=["supervised", "jepa_horizon", "alpha"]
        ).copy()
    figure_06_generated = not paired_surface.empty
    figure_06_uses_common_alpha = bool(
        figure_06_generated
        and source_families == ["ridge_raw_common_alpha"]
        and paired_surface["alpha"].map(np.isfinite).all()
    )
    if figure_06_generated and not figure_06_uses_common_alpha:
        raise ValueError("figure 06 source does not verify common-alpha parity")
    figure_06_audit = {
        "figure": "06_common_alpha_gap_surface.png",
        "status": "generated" if figure_06_generated else "not_generated_no_pairs",
        "reader_families_observed": source_families,
        "uses_common_alpha": figure_06_uses_common_alpha,
        "uses_common_absolute_lambda": False if figure_06_generated else None,
        "n_source_rows": int(surface_source_rows),
        "n_paired_surface_cells": int(len(paired_surface)),
        "alpha_values": (
            sorted(paired_surface["alpha"].dropna().unique().tolist())
            if "alpha" in paired_surface
            else []
        ),
        "lambda_definition": "lambda = alpha * trace(covariance) / D",
    }
    atomic_write_json(destination / "06_common_alpha_audit.json", figure_06_audit)
    if figure_06_generated:
        paired_surface["gap"] = (
            paired_surface["supervised"] - paired_surface["jepa_horizon"]
        )
        pivot = paired_surface.pivot(
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
        & recovery["budget_days_per_stock"].isin(low_budgets)
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
    block_gaps: dict[str, float] = {}
    for block in ("directional", "volatility", "timing"):
        record = signatures.get(block) if isinstance(signatures, Mapping) else None
        if not isinstance(record, Mapping):
            block_gaps[block] = float("nan")
            continue
        block_gaps[block] = _float_or_nan(
            record.get("low_budget_mean_normalized_gap")
        )
    directional_gap = block_gaps["directional"]
    volatility_gap = block_gaps["volatility"]
    timing_gap = block_gaps["timing"]
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
    last_scale_for_regularization = scale_audit.get("last_concat512", {})
    trace_matched = bool(
        last_scale_for_regularization.get(
            "jepa_horizon_supervised_trace_matched_within_10pct", False
        )
    )
    regularization_audit = {
        "trace_matched": trace_matched,
        "scientific_common_regularization_parameter": "alpha",
        "lambda_definition": "lambda = alpha * trace(covariance) / D",
        "figure_06_status": figure_06_audit["status"],
        "figure_06_reader_families_observed": figure_06_audit[
            "reader_families_observed"
        ],
        "figure_06_uses_common_alpha": figure_06_audit["uses_common_alpha"],
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

    nonmonotonicity = _whitening_nonmonotonicity_diagnostic(
        recovery,
        decisive_budgets=decisive_budgets,
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
    pooling = _pooling_diagnostic(raw_u)
    critical_metrics = _critical_budget_metrics(results, decisive_budgets)
    if not critical_metrics.empty:
        atomic_write_parquet(
            critical_metrics, destination / "16_critical_budget_metrics.parquet"
        )

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
            table_row = {
                "target block": values["target_block"],
                "encoder seed": int(values["encoder_seed"]),
            }
            for depth in nonmonotonicity["scope"]["whitening_depths"]:
                table_row[f"k={int(depth)}"] = _format_optional(
                    values.get(float(depth))
                )
            encoder_table_rows.append(table_row)

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

    ceiling = outcome.get("absolute_ceiling_gap")
    if not isinstance(ceiling, Mapping):
        raise ValueError("Phase-I outcome lacks the absolute ceiling gap")
    ceiling_mean = _float_or_nan(ceiling.get("mean"))
    ceiling_lower = _float_or_nan(ceiling.get("lower"))
    ceiling_upper = _float_or_nan(ceiling.get("upper"))
    ceiling_available = all(
        np.isfinite(value)
        for value in (ceiling_mean, ceiling_lower, ceiling_upper)
    )
    ceiling_excludes_zero = bool(
        ceiling_available and (ceiling_lower > 0.0 or ceiling_upper < 0.0)
    )
    ceiling_robust = bool(ceiling.get("robust"))
    if ceiling_available and ceiling_robust != ceiling_excludes_zero:
        raise ValueError(
            "ceiling robust flag is inconsistent with its reported interval"
        )
    sensitivity_outcomes = [
        _classify_frozen_gap_sensitivity(
            gap_sensitivity_frames[delta],
            recovery,
            expected_delta=delta,
            large_sample_ceiling_meaningful=bool(
                outcome.get("large_sample_ceiling_meaningful")
            ),
            ceiling_gap_robust=ceiling_robust,
        )
        for delta in sorted(gap_sensitivity_frames)
    ]
    primary_sensitivity = next(
        row for row in sensitivity_outcomes if np.isclose(row["delta"], 0.10)
    )
    if (
        not gap_sensitivity_frames[0.10].empty
        and primary_sensitivity["outcome"] != outcome["outcome"]
    ):
        raise ValueError(
            "serialized delta=0.10 gap summary does not reproduce the frozen outcome"
        )
    raw_specificity = _raw_specificity_from_critical_budgets(critical_metrics)
    raw_gaps = raw_specificity.get("raw_gap_by_target_block", {})
    if not isinstance(raw_gaps, Mapping):
        raw_gaps = {}
    k_50gap = outcome.get("k_50gap")
    k_nonrobust = outcome.get("k_nonrobust")
    whitening_by_k = {
        candidate["k"]: candidate
        for candidate in outcome.get("whitening_candidates", [])
    }
    k_50_reduction = _float_or_nan(
        whitening_by_k.get(k_50gap, {}).get("reduction_fraction")
    )
    decisive_gap = _float_or_nan(outcome.get("native_low_budget_mean_gap"))
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
    scale_available = all(
        np.isfinite(value)
        for value in (
            last_horizon_trace,
            last_supervised_trace,
            last_horizon_supervised_ratio,
            last_max_min_ratio,
        )
    )
    specificity_table_rows = []
    for block, normalized_gap in (
        ("directional", directional_gap),
        ("volatility", volatility_gap),
        ("timing", timing_gap),
    ):
        normalized_ratio = (
            "—"
            if block == "directional"
            else f"{_format_optional(specificity_ratios[f'directional_over_{block}'], 2)}×"
        )
        raw_ratio = (
            "—"
            if block == "directional"
            else f"{_format_optional(raw_specificity.get(f'directional_over_{block}'), 2)}×"
        )
        specificity_table_rows.append(
            {
                "target block": block,
                "normalized gap (low-budget grid)": _format_optional(
                    normalized_gap
                ),
                "raw R² gap (decisive budgets)": _format_optional(
                    raw_gaps.get(block)
                ),
                "normalized directional/control": normalized_ratio,
                "raw directional/control": raw_ratio,
            }
        )
    specificity_table = _markdown_table(
        specificity_table_rows,
        [
            "target block",
            "normalized gap (low-budget grid)",
            "raw R² gap (decisive budgets)",
            "normalized directional/control",
            "raw directional/control",
        ],
    )
    sensitivity_table = _markdown_table(
        [
            {
                "δ": f"{float(row['delta']):.2f}",
                "technical class": row["outcome"],
                "k_50gap": row.get("k_50gap") or "—",
                "k_nonrobust": row.get("k_nonrobust") or "—",
            }
            for row in sensitivity_outcomes
        ],
        ["δ", "technical class", "k_50gap", "k_nonrobust"],
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
    diagnostic_depths = tuple(
        int(value) for value in nonmonotonicity["scope"]["whitening_depths"]
    )
    encoder_table = _markdown_table(
        encoder_table_rows,
        [
            "target block",
            "encoder seed",
            *(f"k={value}" for value in diagnostic_depths),
        ],
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

    critical_table_rows = [
        {
            "block": row.target_block,
            "budget": _format_budget(row.budget_days_per_stock),
            "branch": row.branch,
            "raw R² mean / median": (
                f"{row.raw_test_r2_mean:.4f} / {row.raw_test_r2_median:.4f}"
            ),
            "ceiling mean [range]": (
                f"{row.full_budget_ceiling_mean:.4f} "
                f"[{row.full_budget_ceiling_min:.4f}, "
                f"{row.full_budget_ceiling_max:.4f}]"
            ),
            "recovery mean / median [range]": (
                f"{row.normalized_recovery_mean:.4f} / "
                f"{row.normalized_recovery_median:.4f} "
                f"[{row.normalized_recovery_min:.4f}, "
                f"{row.normalized_recovery_max:.4f}]"
            ),
            "eligible targets min–max": (
                f"{int(row.eligible_target_count_min_per_encoder)}–"
                f"{int(row.eligible_target_count_max_per_encoder)}"
            ),
            "negative raw fraction": f"{row.negative_raw_test_r2_fraction:.3f}",
        }
        for row in critical_metrics.itertuples()
    ]
    critical_table = _markdown_table(
        critical_table_rows,
        [
            "block",
            "budget",
            "branch",
            "raw R² mean / median",
            "ceiling mean [range]",
            "recovery mean / median [range]",
            "eligible targets min–max",
            "negative raw fraction",
        ],
    )

    spectral = later_phases["spectral"]
    if spectral.get("available"):
        spectral_k = int(spectral["k"])
        horizon_mass = spectral["jepa_horizon"]["predictive_mass"]
        supervised_mass = spectral["supervised"]["predictive_mass"]
        horizon_haar = spectral["jepa_horizon"]["haar"]
        supervised_haar = spectral["supervised"]["haar"]
        spectral_summary_text = (
            f"Phase II places only `{horizon_mass['mean']:.6f}` of horizon-JEPA's "
            f"directional predictive mass in its first {spectral_k} PCs, versus "
            f"`{supervised_mass['mean']:.6f}` for supervised. The fraction of "
            f"Haar draws beating top-PCA averages `{horizon_haar['empirical_p_mean']:.3f}` "
            f"for horizon-JEPA and `{supervised_haar['empirical_p_mean']:.3f}` for "
            "supervised across the recorded encoder seeds."
        )
    else:
        spectral_summary_text = (
            "No Phase-II machine summary is present beside these Phase-I artifacts, "
            "so this report makes no numerical spectral claim."
        )

    if pooling.get("available"):
        horizon_pool = pooling["values"]["jepa_horizon"]
        supervised_pool = pooling["values"]["supervised"]
        pooling_summary_text = (
            "At full budget, changing `last_concat512 → meanK_concatS` changes "
            f"directional test R² from `{horizon_pool['last_concat512']:.6f}` to "
            f"`{horizon_pool['meanK_concatS']:.6f}` for horizon-JEPA and from "
            f"`{supervised_pool['last_concat512']:.6f}` to "
            f"`{supervised_pool['meanK_concatS']:.6f}` for supervised."
        )
    else:
        pooling_summary_text = (
            "The complete matched last/meanK pooling cells are unavailable in the "
            "supplied Phase-I summary, so no pooling contrast is asserted."
        )

    finite_ratios = [
        specificity_ratios["directional_over_volatility"],
        specificity_ratios["directional_over_timing"],
    ]
    raw_ratios = [
        _float_or_nan(raw_specificity.get("directional_over_volatility")),
        _float_or_nan(raw_specificity.get("directional_over_timing")),
    ]
    if all(np.isfinite(value) for value in [*finite_ratios, *raw_ratios]):
        specificity_ratio_text = (
            "The descriptive directional/control ratios are "
            f"`{finite_ratios[0]:.2f}×` and `{finite_ratios[1]:.2f}×` on the "
            f"normalized-recovery scale, versus `{raw_ratios[0]:.2f}×` and "
            f"`{raw_ratios[1]:.2f}×` on the raw-R² scale. The magnitude is "
            "therefore scale-dependent. These are point summaries, not an "
            "independence-adjusted target-block interaction test."
        )
    else:
        specificity_ratio_text = (
            "One or both raw/normalized control-block ratios are unavailable, so "
            "no cross-scale ratio claim is made."
        )

    if k_50gap is not None and k_nonrobust is not None:
        if not np.isfinite(k_50_reduction):
            raise ValueError("k_50gap has no matching whitening candidate")
        maximum_tested_k = max(int(value) for value in whitening_by_k)
        k_nonrobust_reduction = _float_or_nan(
            whitening_by_k.get(k_nonrobust, {}).get("reduction_fraction")
        )
        transition_rows = _gap_summary_subset(
            gap,
            reader_family="ridge_whiten_topk_tuned_alpha",
            feature_view="full_rank_whiten_topk",
        )
        transition_rows = transition_rows[
            transition_rows["whiten_k_requested"].eq(float(k_nonrobust))
            & transition_rows["budget_days_per_stock"].astype(float).isin(
                [float(value) for value in decisive_budgets]
            )
        ].sort_values("budget_days_per_stock")
        positive_interval_text = ""
        if len(transition_rows) == len(decisive_budgets):
            means = ", ".join(f"`{value:.6f}`" for value in transition_rows["mean"])
            lowers = ", ".join(
                f"`{value:.6f}`" for value in transition_rows["lower"]
            )
            if bool((transition_rows["lower"] > 0.0).all()):
                positive_interval_text = (
                    f" The decisive-budget mean gaps are {means}, and their lower "
                    f"interval bounds remain positive ({lowers})."
                )
        whitening_summary_text = (
            f"At `k_50gap={int(k_50gap)}`, whitening reduces the decisive-budget "
            f"gap by `{k_50_reduction:.1%}` but does not eliminate it. "
            f"At the historical technical field `k_nonrobust={int(k_nonrobust)}`, "
            f"the gap no longer meets the compound preregistered criterion "
            f"`lower > 0 and mean ≥ δ={float(outcome['delta']):.2f}` at both "
            "decisive budgets; this is an effect-threshold transition, not a "
            "confidence interval crossing zero."
            f"{positive_interval_text} "
            + (
                "It is the maximum tested valid whitening depth. "
                if int(k_nonrobust) == maximum_tested_k
                else "It is below the maximum tested valid whitening depth. "
            )
            + (
                f"The mean decisive-budget gap is reduced by "
                f"`{k_nonrobust_reduction:.1%}` there. "
                if np.isfinite(k_nonrobust_reduction)
                else ""
            )
            + "This pattern does not support concentration in only a few leading PCs."
        )
    else:
        whitening_summary_text = (
            "The frozen outcome has no finite whitening transition, so no whitening-"
            "depth claim is made."
        )

    historical_status_text = (
        "The historical reproduction gate is available and passed."
        if historical.get("available") and historical.get("passed")
        else "The historical reproduction gate is not available in this report context."
    )
    later_phase_status_text = (
        f"Phase II status: `{later_phases['phase2']['status']}`; Phase III-R status: "
        f"`{later_phases['phase3_r']['status']}`. These later diagnostics do not "
        "change the frozen Phase-I outcome."
    )
    nonmonotonicity_text = _directional_nonmonotonicity_text(
        nonmonotonicity["paired_differences"]
    )
    fractional_n_over_d = results[
        results["budget_kind"].eq("fractional")
        & results["readout"].eq("last_concat512")
        & results["feature_view"].eq("full_rank_raw")
        & results["reader_family"].eq("ridge_raw_tuned_alpha")
    ]["n_rows_over_dim"]
    min_fractional_n_over_d = (
        _require_finite(fractional_n_over_d.min(), "minimum fractional n/D")
        if not fractional_n_over_d.empty
        else float("nan")
    )

    primary_result = {
        "name": "frozen_phase1_outcome_with_separate_scientific_diagnostics",
        "spectral_diagnostic": spectral,
        "pooling_diagnostic": pooling,
        "phase1_normalized_finite_sample_gap": {
            "directional": directional_gap,
            "volatility": volatility_gap,
            "timing": timing_gap,
            **specificity_ratios,
        },
        "phase1_raw_decisive_budget_gap": raw_specificity,
        "delta_sensitivity": sensitivity_outcomes,
    }

    adjacent_pair_text = ", ".join(
        f"`{_format_budget(left)}→{_format_budget(right)}`"
        for left, right in adjacent_pairs
    )
    trace_match_text = (
        "matched within the report's 10% diagnostic tolerance"
        if last_scale.get("jepa_horizon_supervised_trace_matched_within_10pct")
        else "not matched within the report's 10% diagnostic tolerance"
    )
    if figure_06_audit["status"] == "generated":
        figure_06_text = (
            "Figure 06 passed its source audit: every plotted cell comes from "
            "`ridge_raw_common_alpha`, and its axis is dimensionless alpha rather "
            "than absolute lambda."
        )
    else:
        figure_06_text = (
            "Figure 06 was not generated because no complete paired common-alpha "
            "surface was available; no regularization-parity claim is made from it."
        )
    n_over_d_text = (
        f"The smallest observed fractional-budget `n/D` is "
        f"`{min_fractional_n_over_d:.3f}`, so the original grid does not enter "
        "the `n/D < 1` regime."
        if np.isfinite(min_fractional_n_over_d)
        else "No fractional-budget `n/D` value is available in these artifacts."
    )
    decisive_budget_text = (
        _format_budget_list(decisive_budgets)
        if decisive_budgets
        else "`none recorded`"
    )
    if ceiling_available:
        ceiling_summary_text = (
            f"The supervised-minus-horizon operational ceiling gap is "
            f"`{ceiling_mean:.6f}` with computational-robustness interval "
            f"`[{ceiling_lower:.6f}, {ceiling_upper:.6f}]`. "
            f"{_interval_zero_statement(ceiling_lower, ceiling_upper)}"
        )
    else:
        ceiling_summary_text = (
            "A two-branch operational ceiling gap is unavailable in these artifacts, "
            "so no ceiling-gap claim is made."
        )
    if np.isfinite(directional_gap):
        recovery_summary_text = (
            f"The mean normalized directional gap over the frozen low-budget grid "
            f"is `{directional_gap:.6f}`; over the recorded decisive budgets "
            f"{decisive_budget_text} it is `{_format_optional(decisive_gap, 6)}`."
        )
    else:
        recovery_summary_text = (
            "A two-branch normalized directional gap is unavailable in these "
            "artifacts, so no recovery-gap claim is made."
        )
    if scale_available:
        trace_summary_text = (
            f"On `last_concat512`, mean `trace_cov_over_dim` is "
            f"`{last_horizon_trace:.6f}` for horizon-JEPA and "
            f"`{last_supervised_trace:.6f}` for supervised, a ratio of "
            f"`{last_horizon_supervised_ratio:.6f}`. The two traces are "
            f"{trace_match_text}. The all-branch max/min ratio is "
            f"`{last_max_min_ratio:.6f}`."
        )
    else:
        trace_summary_text = (
            "The report inputs do not contain both horizon-JEPA and supervised "
            "trace-scale cells, so no covariance-scale parity claim is made."
        )

    narrative_summary = f"""# Narrative summary — Experiment 01 Phase I

## Primary Phase-I result: reader-relative finite-sample accessibility

Conditional on the frozen representations and a newly fitted linear reader,
the supervised representation is more accessible at low reader-label budgets.
This is not an end-to-end label-efficiency claim because the supervised encoder
was itself trained with directional and volatility labels.

{spectral_summary_text}

{pooling_summary_text}

The Phase-I normalized finite-sample gaps are `{_format_optional(directional_gap, 6)}` for
direction, `{_format_optional(volatility_gap, 6)}` for volatility and
`{_format_optional(timing_gap, 6)}` for timing. {specificity_ratio_text}

## Separate Phase-I effects

- {ceiling_summary_text}
- {recovery_summary_text}
- {whitening_summary_text}

## Frozen technical classification and mandatory sensitivity

At the preregistered primary threshold `δ={float(outcome['delta']):.2f}`, the
frozen Phase-I classifier returns **{outcome['outcome']}**. This technical label
is secondary to the empirical accessibility result. The mandatory sensitivity
grid is:

{sensitivity_table}

These sensitivity rows do not replace the primary threshold. They show that
the taxonomy label is threshold-sensitive even though the underlying gap curve
is unchanged. Nothing in this narrative revision changes thresholds, result
rows or the decision rule. The separate operational ceiling fact is reported as
**{outcome['outcome']} with a robust ceiling gap**; `B` is not treated as a
coexisting outcome.

## Parity and scope

{historical_status_text} Production full-budget scores and new-test min-norm
OLS remain separate because the new chronological test half is not required to
equal the old validation split. {later_phase_status_text}
"""
    atomic_write_text(
        destination / "SUMMARY_NARRATIVE_EXPERIMENT_01.md", narrative_summary
    )

    report = f"""# Report — Experiment 01 Phase I

## Primary Phase-I result: reader-relative finite-sample accessibility

The Phase-I result supports the following restricted statement: conditional on
the frozen representations and a newly fitted reader, the supervised
representation is more accessible at low reader-label budgets. It does not
establish that supervised pretraining is intrinsically more label-efficient
end to end, because the supervised encoder saw directional and volatility
labels during pretraining.

### Directional spectral organization

{spectral_summary_text}

When present, these values come from the completed Phase-II machine summary,
not from literals copied from the older post-P0 PCA ladder. They are diagnostic
evidence and do not alter Phase I.

### Pooling interaction

{pooling_summary_text}

This matched readout contrast is reported as an interaction with pooling, not
as an additional A/B/D outcome.

### Finite-sample specificity

The table distinguishes normalized gaps over the frozen low-budget grid from
raw-R² gaps averaged over the recorded decisive budgets:

{specificity_table}

{specificity_ratio_text} Volatility and timing remain separate controls. These
ratios alone do not establish an interaction because target families are
correlated and grouped stock-day uncertainty has not yet been computed.

## Phase-I effects kept separate

### Operational linear ceiling gap

{ceiling_summary_text} When available, this is an operational linear ceiling
statement, not a normalized sample-efficiency statement.

### Normalized-recovery gap

Recovery is normalized target-wise by each representation's eligible
operational ceiling. {recovery_summary_text} The frozen summary records the adjacent robust pairs as
{adjacent_pair_text or '`none`'}.

### Mediation by progressive whitening

{whitening_summary_text}

## Frozen preregistered classification and δ sensitivity

At the preregistered primary threshold `δ={float(outcome['delta']):.2f}`, the
frozen Phase-I technical classifier returns **{outcome['outcome']}**. This label
is reported secondarily to the empirical accessibility result. In the
historical rule, “robust” is a
compound practical-effect criterion: the interval lower bound must exceed zero
and the point estimate must reach `δ`. It does not mean only “statistically
different from zero.”

{sensitivity_table}

The `δ=0.05` and `δ=0.15` rows are mandatory preregistered sensitivities, not
alternative primary outcomes. A label change across this grid means that the
taxonomy is threshold-sensitive; it does not alter any measured gap. The
result rows, thresholds and classification logic have not been modified. The
operational ceiling result is stated separately as **{outcome['outcome']} with
a robust ceiling gap**; `B` is not a coexisting outcome.

The frozen machine reason is retained verbatim in the record below. Its phrase
“makes it non-robust” must be read according to the compound criterion above,
not as a confidence interval crossing zero.

The complete machine-readable record is reproduced for auditability:

```json
{json.dumps(outcome, indent=2, sort_keys=True)}
```

## Raw and normalized metrics at decisive budgets

The table co-reports raw test R², operational ceiling and normalized recovery.
Ranges are over the eligible target/encoder/subsample cells represented in the
frozen result table; they are not population intervals.

{critical_table}

The machine-readable version is `16_critical_budget_metrics.parquet`.

## Whitening-depth non-monotonicity diagnostic

This added diagnostic uses only frozen Phase-I recovery points. Within each
encoder/subsample cell, it first averages the paired supervised–horizon gap
over the decisive budgets ({decisive_budget_text}), then applies 5,000-draw
hierarchical resampling of encoder seeds followed by paired cells within
encoder. It does not refit a reader and is not outcome-defining.

### Hierarchical intervals by target block and depth

{interval_table}

### Paired differences between adjacent inspected depths

Differences are `gap(to k) − gap(from k)`, paired by encoder seed and subsample.

{difference_table}

### Means by encoder seed and target block

{encoder_table}

{nonmonotonicity_text} This diagnostic is post hoc and does not modify the
preregistered interpretation. Full paired values per encoder are retained in
the `15_whitening_nonmonotonicity_*` artifacts.

## Global covariance scale and regularization parity

{trace_summary_text}

Scientific common-regularization comparisons use the dimensionless parameter

`lambda = alpha * trace(covariance) / D`.

{figure_06_text} No fixed-absolute-lambda comparison is included; when trace
scales differ, such a comparison is confounded.

```json
{json.dumps(scale_audit, indent=2, sort_keys=True)}
```

## Historical and production parity

### Historical reproduction gate

This is the mandatory old-split min-norm OLS reproduction check:

{historical_table}

{historical_status_text}

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
- normalized recovery is target-wise and only uses full-budget R² at least 0.01;
- the `R² ≥ 0.01` ceiling-eligibility rule is evaluated on full-budget test
  outcomes; it is part of metric definition, not validation-time hyperparameter
  selection.

{n_over_d_text}

## External-validity limits

- The dataset contains seven stocks from one market/domain.
- The historical split is stock-day-group-disjoint but not globally
  chronological. Within each stock, train days span almost the full calendar
  year and occur both before and after held-out validation/test days; this is
  not a forward-only temporal-generalization design.
- Validation and test are chronological halves of a historically explored
  held-out set, so the new test is not a pristine external confirmation set.
- Fractional budgets vary within-day endpoint coverage while retaining seven
  stock-day groups.
- Supervised pretraining used directional and volatility labels later probed by
  Experiment 01; timing was not a direct training target but may be correlated
  with those labels.

## Specificity and time-of-day controls

```json
{json.dumps(findings, indent=2, sort_keys=True)}
```

Opening, middle and closing contiguous blocks are reported separately in
`time_of_day_sensitivity_summary.parquet` ({len(sensitivity_summary)} rows).
These sensitivity cells are not pooled into the random-anchor curves.

## Uncertainty

Intervals use hierarchical resampling of encoder seeds followed by subsampling
seeds within encoder. They are **computational-robustness intervals**, not
population-generalization confidence intervals. Grouped stock/day uncertainty
and leave-one-stock-out sensitivity remain pending. Companion tables expose
`sd_subsample_within_encoder` and `sd_encoder_between_means`; all
encoder-specific curves are retained in figure 10.

## Figures and diagnostics

{chr(10).join(f'- `{name}`' for name in figure_paths)}
- `06_common_alpha_audit.json`
- `15_whitening_nonmonotonicity_manifest.json`
- `16_critical_budget_metrics.parquet`
{chr(10).join(f'- `{name}`' for name in diagnostic_files.values())}

{later_phase_status_text} This revision changes narrative and read-only report
diagnostics only; it does not change Phase-I results, thresholds, the technical
outcome or the fit pipeline.
"""
    atomic_write_text(destination / "REPORT_EXPERIMENT_01.md", report)

    repository_root = Path(__file__).resolve().parents[1]

    def portable_source_path(path: Path | str) -> str:
        candidate = Path(path)
        try:
            return str(candidate.resolve().relative_to(repository_root.resolve()))
        except ValueError:
            return candidate.name

    def portable_payload(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): portable_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable_payload(item) for item in value]
        if isinstance(value, tuple):
            return [portable_payload(item) for item in value]
        if isinstance(value, str) and Path(value).is_absolute():
            return portable_source_path(value)
        return value

    protected_inputs = {
        "phase1_results": {
            "path": portable_source_path(results_path),
            "sha256": sha256_file(results_path),
        },
        "technical_summary": {
            "path": portable_source_path(summary_path),
            "sha256": sha256_file(summary_path),
        },
    }
    for delta, path in gap_paths.items():
        protected_inputs[f"gap_summary_delta_{delta:.2f}"] = {
            "path": portable_source_path(path),
            "sha256": sha256_file(path),
        }
    if spectral.get("available"):
        protected_inputs["phase2_summary"] = {
            "path": portable_source_path(str(spectral["source_path"])),
            "sha256": spectral["source_sha256"],
        }
    if later_phases["phase3_r"]["status"] == "complete":
        protected_inputs["phase3_r_summary"] = {
            "path": portable_source_path(
                str(later_phases["phase3_r"]["summary_path"])
            ),
            "sha256": later_phases["phase3_r"]["summary_sha256"],
        }

    claim_rows: list[dict[str, object]] = []

    def add_claim(
        claim_id: str,
        phase: str,
        metric: str,
        value: object,
        source_path: Path,
        source_columns: str,
        filter_text: str,
        aggregation: str,
        report_locations: str,
    ) -> None:
        claim_rows.append(
            {
                "claim_id": claim_id,
                "phase": phase,
                "metric": metric,
                "value_json": json.dumps(value, sort_keys=True),
                "source_artifact": portable_source_path(source_path),
                "source_columns": source_columns,
                "filter": filter_text,
                "aggregation": aggregation,
                "artifact_sha256": sha256_file(source_path),
                "report_locations": report_locations,
            }
        )

    add_claim(
        "phase1.technical_outcome",
        "I",
        "A1/A2/B/D classification",
        outcome["outcome"],
        summary_path,
        "directional_last_concat512_outcome.outcome",
        "directional/last_concat512",
        "frozen preregistered rule",
        "summary; report/frozen preregistered outcome",
    )
    if ceiling_available:
        add_claim(
            "phase1.ceiling_gap",
            "I",
            "supervised_minus_horizon_full_budget_raw_r2",
            {"mean": ceiling_mean, "lower": ceiling_lower, "upper": ceiling_upper},
            summary_path,
            "directional_last_concat512_outcome.absolute_ceiling_gap",
            "directional/last_concat512/full_train/ridge_raw_tuned_alpha",
            "hierarchical encoder/subsample interval",
            "summary; report/operational linear ceiling gap",
        )
    for block, value in block_gaps.items():
        if np.isfinite(value):
            add_claim(
                f"phase1.normalized_gap.{block}",
                "I",
                "low_budget_mean_normalized_gap",
                value,
                summary_path,
                f"target_block_gap_signatures.{block}.low_budget_mean_normalized_gap",
                f"{block}/last_concat512/ridge_raw_tuned_alpha",
                "mean over frozen low-budget grid",
                "summary; report/finite-sample specificity",
            )
    for row in sensitivity_outcomes:
        delta = float(row["delta"])
        add_claim(
            f"phase1.delta_sensitivity.{delta:.2f}",
            "I",
            "A1/A2/B/D sensitivity classification",
            row,
            gap_paths[delta],
            "mean,lower,upper,delta,robust",
            "directional/last_concat512/frozen reader and whitening families",
            "frozen taxonomy reapplied to serialized gap summaries",
            "summary; report/delta sensitivity",
        )
    if raw_specificity.get("available"):
        add_claim(
            "phase1.raw_specificity.decisive_budgets",
            "I",
            "raw supervised-minus-horizon gap and directional/control ratios",
            raw_specificity,
            destination / "16_critical_budget_metrics.parquet",
            "raw_test_r2_mean",
            "last_concat512;independent targets;decisive budgets",
            "paired arm difference, then mean across decisive budgets",
            "summary; report/finite-sample specificity",
        )
    if pooling.get("available"):
        raw_uncertainty_path = summary_root / "raw_uncertainty.parquet"
        for branch, values in pooling["values"].items():
            for readout, value in values.items():
                add_claim(
                    f"phase1.pooling.{branch}.{readout}",
                    "I",
                    "full_budget_directional_test_r2",
                    value,
                    raw_uncertainty_path,
                    "raw_r2_mean",
                    (
                        f"branch={branch};readout={readout};directional;full_train;"
                        "ridge_raw_tuned_alpha;full_rank_raw"
                    ),
                    "hierarchical point mean",
                    "summary; report/pooling interaction",
                )
    if spectral.get("available"):
        phase2_source = Path(str(spectral["source_path"]))
        for branch in ("jepa_horizon", "supervised"):
            add_claim(
                f"phase2.predictive_mass.{branch}.top{spectral['k']}",
                "II",
                "cumulative_directional_predictive_mass",
                spectral[branch]["predictive_mass"],
                phase2_source,
                "findings.directional_last_cumulative_mass",
                f"branch={branch};k={spectral['k']};last_concat512;directional",
                "hierarchical mean and interval",
                "summary; report/directional spectral organization",
            )
            add_claim(
                f"phase2.haar.{branch}.top{spectral['k']}",
                "II",
                "empirical_p_random_exceeds_top",
                spectral[branch]["haar"],
                phase2_source,
                "findings.directional_last_top_pca_haar",
                f"branch={branch};k={spectral['k']};last_concat512;directional",
                "mean and range over encoder seeds",
                "summary; report/directional spectral organization",
            )
    claim_table = pd.DataFrame(claim_rows)
    atomic_write_parquet(claim_table, destination / "17_claim_table.parquet")

    changelog = f"""# Changelog — Experiment 01 Phase I narrative revision

Latest consolidation date: 2026-08-27.

## Adversarial-audit consolidation — 2026-08-27

- Kept the frozen `{outcome['outcome']}` classifier output at the preregistered
  primary threshold `δ={float(outcome['delta']):.2f}` and added the mandatory `δ=0.05/0.15`
  classifications from the existing serialized gap summaries.
- Defined the historical `robust` flag explicitly as the compound condition
  `lower > 0 and mean ≥ δ`. At `k_nonrobust={k_nonrobust}`, the report now
  distinguishes failure of that practical-effect criterion from a confidence
  interval crossing zero.
- Added the full-depth gap reduction
  `{_format_optional(whitening_by_k.get(k_nonrobust, {}).get('reduction_fraction'), 6)}`
  while retaining `k_50gap={k_50gap}` and every frozen whitening result.
- Added raw decisive-budget arm gaps and raw directional/control ratios beside
  normalized recovery, with an explicit scale-dependence label.
- Clarified that train stock-days occur both before and after held-out days and
  that ceiling eligibility is a test-outcome metric definition, not a selected
  hyperparameter.
- Added an artifact-derived Phase-I claim map; no feature, reader or encoder was
  regenerated.

## Earlier narrative revision — 2026-08-25

## Narrative changes

- Superseded the earlier narrative artifact; its exact identity remains in
  version-control history rather than in this deterministically regenerated
  report bundle.
- Restored `{outcome['outcome']}` as the explicit frozen preregistered technical
  classification while separating it from the scientific interpretation.
- Removed copied PCA/null, pooling, gap, ratio, budget, whitening-depth and
  execution-status literals. Narrative values now come from hashed inputs.
- Replaced the older PCA-ladder literals with the detected completed Phase-II
  summary and marked that evidence as later diagnostic context.
- Separated the robust `{ceiling_mean:.6f}` operational ceiling gap, robust
  normalized-recovery gap, and whitening mediation.
- Whitening wording is generated from the frozen candidates:
  `k_50gap={k_50gap}`, `k_nonrobust={k_nonrobust}` and reduction
  `{_format_optional(k_50_reduction, 6)}`; no few-PC concentration is claimed.
- Replaced any possible “coexisting B” reading with “A1 with a robust ceiling
  gap.”
- Added the supervised-pretraining-label limitation and restricted the reader
  result to frozen-representation accessibility.

## Added read-only diagnostics

- Added hierarchical intervals and paired adjacent-depth differences for
  {_format_budget_list(diagnostic_depths)}, plus results by encoder seed and target block. The diagnostic
  reads frozen recovery points only and is explicitly post hoc.
- Diagnostic row counts: {len(nonmonotonicity['intervals'])} depth intervals,
  {len(nonmonotonicity['paired_differences'])} hierarchical paired differences,
  {len(nonmonotonicity['per_encoder'])} depth-by-encoder rows and
  {len(nonmonotonicity['paired_differences_per_encoder'])} paired-difference-by-encoder
  rows. Directional paired results (`to−from`) are:
  `{directional_difference_text}`.
- Added the trace-scale audit (observed `last` horizon/supervised ratio
  `{last_horizon_supervised_ratio:.6f}`), the exact
  regularization formula, and an explicit verification that figure 06 uses
  common alpha when generated. Fixed-absolute-lambda comparisons are absent.
- Added historical reproduction parity, production full-budget test results,
  and new-test min-norm OLS as a separate diagnostic with no old/new split
  equality requirement.
- Added raw/normalized decisive-budget table and a hashed claim table.
- Corrected uncertainty language: existing intervals are computational
  robustness intervals; grouped stock/day uncertainty is pending.
- Replaced stale Phase-II/III scope prose with detected artifact status:
  Phase II `{later_phases['phase2']['status']}`, Phase III-R
  `{later_phases['phase3_r']['status']}`.

## Unchanged protected artifacts

- `results.parquet`: `{protected_inputs['phase1_results']['sha256']}`
- `summary/summary.json`: `{protected_inputs['technical_summary']['sha256']}`
- Technical outcome: `{outcome['outcome']}`.
- Thresholds: `k_50gap={k_50gap}`,
  `k_nonrobust={k_nonrobust}`, `delta={outcome['delta']}`.
- This revision did not generate features, fit readers, train encoders or run a
  new experimental phase.
"""
    atomic_write_text(
        destination / "CHANGELOG_NARRATIVE_20260731.md", changelog
    )

    payload = {
        "primary_result": primary_result,
        "technical_classification": {
            "role": "frozen_preregistered_technical_outcome",
            "label": outcome["outcome"],
            "wording": f"{outcome['outcome']} with a robust ceiling gap",
            "unchanged_record": outcome,
        },
        # Backward-compatible machine-readable technical outcome.
        "outcome": outcome,
        "specificity_findings": findings,
        "specificity_ratios": specificity_ratios,
        "raw_specificity": raw_specificity,
        "delta_sensitivity": sensitivity_outcomes,
        "global_scale_audit": scale_audit,
        "regularization_audit": regularization_audit,
        "figure_06_audit": figure_06_audit,
        "whitening_nonmonotonicity": nonmonotonicity_manifest,
        "parity": parity,
        "later_phase_context": later_phases,
        "critical_budget_metrics": {
            "path": "16_critical_budget_metrics.parquet",
            "n_rows": len(critical_metrics),
        },
        "claim_table": {
            "path": "17_claim_table.parquet",
            "n_rows": len(claim_table),
        },
        "protected_inputs": protected_inputs,
        "narrative_revision": {
            "date": "2026-08-27",
            "superseded_report_identity": "version_control_history",
            "results_or_technical_summary_modified": False,
        },
        "time_of_day_sensitivity_rows": len(sensitivity_summary),
        "figures": figure_paths,
        "report": "REPORT_EXPERIMENT_01.md",
        "narrative_summary": "SUMMARY_NARRATIVE_EXPERIMENT_01.md",
        "changelog": "CHANGELOG_NARRATIVE_20260731.md",
        "phase2_status": later_phases["phase2"]["status"],
        "phase3_r_status": later_phases["phase3_r"]["status"],
    }
    payload = portable_payload(payload)
    atomic_write_json(destination / "report_manifest.json", payload)
    return payload

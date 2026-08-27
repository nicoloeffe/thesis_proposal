"""Audited numerical-boundary correction for the frozen F16 summary rule."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiment01.f16_posttest import summarize_f16_serialization_fix
from experiment01.io import atomic_write_json, canonical_json_sha256, json_safe, sha256_file


SPEARMAN_THRESHOLD = 0.8
BOUNDARY_TOLERANCE = 1e-12


def summarize_f16_boundary_corrected(repo_root: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    preliminary = summarize_f16_serialization_fix(repo_root, output_root)
    preliminary_sha = sha256_file(output_root / "f16_summary.json")
    summary = dict(preliminary)
    smoothness = dict(summary["smoothness"])
    details = [dict(row) for row in smoothness["details"]]
    changed = []
    for row in details:
        prior = bool(row["passes_all_seeds_rho_ge_0_8"])
        corrected = all(
            np.isfinite(float(value))
            and float(value) >= SPEARMAN_THRESHOLD - BOUNDARY_TOLERANCE
            for value in row["spearman_by_seed"].values()
        )
        row["passes_all_seeds_rho_ge_0_8"] = corrected
        if corrected != prior:
            changed.append(
                {
                    "family": row["family"],
                    "prior": prior,
                    "corrected": corrected,
                    "spearman_by_seed": row["spearman_by_seed"],
                }
            )
    count = sum(bool(row["passes_all_seeds_rho_ge_0_8"]) for row in details)
    smoothness["details"] = details
    smoothness["families_passing_all_seed_rho_ge_0_8"] = count
    smoothness["numerical_boundary_policy"] = {
        "mathematical_threshold": SPEARMAN_THRESHOLD,
        "absolute_tolerance": BOUNDARY_TOLERANCE,
        "reason": "floating representation of exact rank correlation 0.8",
    }
    summary["smoothness"] = smoothness
    flags = dict(summary["interpretation"]["flags"])
    flags["smooth_label_volume_dependence"] = count >= 4
    accessibility = {
        row["family"]: bool(row["passes_all_seeds_rho_ge_0_8"])
        for row in details
        if row["family"] in {"axis_a_accessibility", "axis_b_accessibility"}
    }
    geometry_count = sum(
        bool(row["passes_all_seeds_rho_ge_0_8"])
        for row in details
        if row["family"]
        in {"role_retention", "topk_predictive_mass", "pooling_loss", "whitening_k128"}
    )
    flags["accessibility_without_measured_geometry_change"] = (
        any(accessibility.values()) and geometry_count < 2
    )
    if sum(bool(value) for value in flags.values()) == 1:
        overall = next(key for key, value in flags.items() if value)
    elif sum(bool(value) for value in flags.values()) == 0:
        overall = "no_preregistered_pattern_passed"
    else:
        overall = "multiple_preregistered_patterns_passed_report_separately"
    interpretation = dict(summary["interpretation"])
    interpretation["flags"] = flags
    interpretation["overall"] = overall
    summary["interpretation"] = interpretation
    summary["posttest_numerical_boundary_correction"] = {
        "applied": True,
        "changed_families": changed,
        "scientific_inputs_changed": False,
        "selection_changed": False,
    }
    summary.pop("summary_fingerprint", None)
    summary["summary_fingerprint"] = canonical_json_sha256(json_safe(summary))
    atomic_write_json(output_root / "f16_summary.json", summary)
    amendment = {
        "schema_name": "thesis.experiment01.f16_posttest_threshold_amendment",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trigger": "exact Spearman rho=0.8 represented as 0.7999999999999999",
        "frozen_rule": "rho >= 0.8 in all three encoder seeds",
        "correction": "compare rho >= 0.8 - 1e-12",
        "changed_families": changed,
        "prior_summary_sha256": preliminary_sha,
        "corrected_summary_sha256": sha256_file(output_root / "f16_summary.json"),
        "results_sha256": sha256_file(output_root / "f16_results.parquet"),
        "geometry_sha256": sha256_file(output_root / "f16_geometry.parquet"),
        "grouped_uncertainty_sha256": sha256_file(
            output_root / "f16_grouped_uncertainty.parquet"
        ),
        "test_reopened": False,
        "checkpoint_selection_changed": False,
        "alpha_selection_changed": False,
        "outcome_rule_changed": False,
    }
    amendment["amendment_fingerprint"] = canonical_json_sha256(amendment)
    atomic_write_json(output_root / "f16_posttest_threshold_amendment.json", amendment)
    return summary

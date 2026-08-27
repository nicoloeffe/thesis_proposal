"""Compute-feasible, pre-test amendment for Experiment 01 Phase III.

Phase III-R changes only the job inventory.  It reuses the frozen bundle,
features, transforms, splits, MLP definition, optimizer, stopping schedule,
weight-decay grid, test boundary, metrics, thresholds, and R1--R4 outcomes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .errors import ExperimentIntegrityError
from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .phase3 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    PHASE1_SUBSET_MANIFEST_SHA256,
    PRIMARY_BRANCHES,
    PRIMARY_READOUT,
    READER_SEEDS,
    SELECTION_SEED,
    VALID_RANK,
    WEIGHT_DECAYS,
    build_phase3_job_inventory,
    verify_completed_cell,
    verify_phase1_inputs,
)


PHASE3_REDUCED_VARIANT = "phase3_r_compute_feasible.v1"
PHASE3_REDUCED_PRIMARY_BUDGETS = ("b_1_4", "b_1_2", "full_train")
PHASE3_REDUCED_CONTROL_BUDGETS = ("b_1_4", "full_train")
PHASE3_REDUCED_SPECTRAL_BUDGETS = ("b_1_4", "full_train")
PHASE3_REDUCED_LOW_SUBSAMPLE_SEEDS = (0, 1, 2)
PHASE3_REDUCED_READER_SEEDS = READER_SEEDS[:3]
PHASE3_REDUCED_SPECTRAL_ARMS = (
    "band_1_127",
    "band_382_508",
    "full_valid_rank",
)
EXPECTED_SELECTION_MODELS = 648
EXPECTED_EVALUATION_MODELS = 648
EXPECTED_LOGICAL_CELLS = 216
EXPECTED_REUSABLE_SELECTION_MODELS = 252


_PREREQUISITE_ARTIFACTS = (
    "AUDIT_EXPERIMENT_01_PHASE3.md",
    "artifact_identity_gate.json",
    "compute_benchmark.json",
    "conditioning_gate.json",
    "historical_mlp_gate.json",
    "historical_mlp_gate_runs.parquet",
    "linear_parity_gate.json",
    "nonlinear_gate.json",
    "pca_band_identity_gate.json",
    "phase1_branch_whitening_effects.parquet",
    "phase1_branch_whitening_effects_summary.json",
    "preproduction_gates.json",
    "test_results.txt",
)


def _eligible_subset(frame: pd.DataFrame) -> pd.Series:
    return frame["budget_label"].eq("full_train") | frame[
        "subsample_seed"
    ].astype(int).isin(PHASE3_REDUCED_LOW_SUBSAMPLE_SEEDS)


def build_phase3_reduced_inventory(
    subset_payload: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the definitive v1 inventory to the confirmed Phase III-R design."""

    selection, evaluation = build_phase3_job_inventory(subset_payload)

    def reduced(frame: pd.DataFrame) -> pd.DataFrame:
        primary = frame.loc[
            frame["job_family"].eq("primary_directional")
            & frame["budget_label"].isin(PHASE3_REDUCED_PRIMARY_BUDGETS)
            & _eligible_subset(frame)
        ]
        controls = frame.loc[
            frame["job_family"].eq("specificity_control")
            & frame["budget_label"].isin(PHASE3_REDUCED_CONTROL_BUDGETS)
            & _eligible_subset(frame)
        ]
        spectral = frame.loc[
            frame["job_family"].eq("spectral_diagnostic")
            & frame["branch"].eq("jepa_horizon")
            & frame["target_block"].eq("directional")
            & frame["spectral_arm"].isin(PHASE3_REDUCED_SPECTRAL_ARMS)
            & frame["budget_label"].isin(PHASE3_REDUCED_SPECTRAL_BUDGETS)
            & _eligible_subset(frame)
        ]
        return (
            pd.concat([primary, controls, spectral], ignore_index=True)
            .sort_values("job_key", kind="stable")
            .reset_index(drop=True)
        )

    selection = reduced(selection)
    evaluation = reduced(evaluation)
    evaluation = evaluation.loc[
        evaluation["reader_seed"].astype(int).isin(PHASE3_REDUCED_READER_SEEDS)
    ].reset_index(drop=True)
    validate_phase3_reduced_inventory(selection, evaluation)
    return selection, evaluation


def validate_phase3_reduced_inventory(
    selection: pd.DataFrame, evaluation: pd.DataFrame
) -> None:
    """Fail closed on cardinality, pairing, and the exact confirmed amendment."""

    if len(selection) != EXPECTED_SELECTION_MODELS:
        raise ExperimentIntegrityError(
            f"Phase III-R selection count differs: {len(selection)}"
        )
    if len(evaluation) != EXPECTED_EVALUATION_MODELS:
        raise ExperimentIntegrityError(
            f"Phase III-R evaluation count differs: {len(evaluation)}"
        )
    if selection["job_key"].duplicated().any() or evaluation["job_key"].duplicated().any():
        raise ExperimentIntegrityError("Phase III-R job keys are not unique")
    if set(selection["weight_decay"].astype(float)) != set(WEIGHT_DECAYS):
        raise ExperimentIntegrityError("Phase III-R weight-decay grid changed")
    if set(selection["reader_seed"].astype(int)) != {SELECTION_SEED}:
        raise ExperimentIntegrityError("Phase III-R selection seed changed")
    if set(evaluation["reader_seed"].astype(int)) != set(
        PHASE3_REDUCED_READER_SEEDS
    ):
        raise ExperimentIntegrityError("Phase III-R reader seeds changed")
    if set(selection["width"].astype(int)) != {256}:
        raise ExperimentIntegrityError("Phase III-R contains a capacity sweep")
    if len(set(selection["logical_job_key"].astype(str))) != EXPECTED_LOGICAL_CELLS:
        raise ExperimentIntegrityError("Phase III-R logical selection count changed")
    if len(set(evaluation["logical_job_key"].astype(str))) != EXPECTED_LOGICAL_CELLS:
        raise ExperimentIntegrityError("Phase III-R logical evaluation count changed")
    selection_replication = selection.groupby("logical_job_key", observed=True).size()
    evaluation_replication = evaluation.groupby("logical_job_key", observed=True).size()
    if not selection_replication.eq(3).all() or not evaluation_replication.eq(3).all():
        raise ExperimentIntegrityError("Phase III-R replication is not exactly three")
    if set(selection["logical_job_key"].astype(str)) != set(
        evaluation["logical_job_key"].astype(str)
    ):
        raise ExperimentIntegrityError("Phase III-R selection/evaluation cells differ")

    family_counts = selection.groupby("job_family", observed=True).size().to_dict()
    if family_counts != {
        "primary_directional": 252,
        "specificity_control": 288,
        "spectral_diagnostic": 108,
    }:
        raise ExperimentIntegrityError(
            f"Phase III-R family counts differ: {family_counts}"
        )
    spectral = selection.loc[selection["job_family"] == "spectral_diagnostic"]
    if (
        set(spectral["branch"]) != {"jepa_horizon"}
        or set(spectral["target_block"]) != {"directional"}
        or set(spectral["spectral_arm"]) != set(PHASE3_REDUCED_SPECTRAL_ARMS)
    ):
        raise ExperimentIntegrityError("Phase III-R spectral contrast changed")

    paired = selection.loc[
        selection["job_family"].isin(
            ("primary_directional", "specificity_control")
        )
    ]
    pairing_key = [
        "job_family",
        "encoder_seed",
        "target_block",
        "transform",
        "budget_label",
        "subsample_seed",
        "weight_decay",
    ]
    branch_counts = paired.groupby(pairing_key, observed=True)["branch"].nunique()
    if not branch_counts.eq(len(PRIMARY_BRANCHES)).all():
        raise ExperimentIntegrityError("Phase III-R branch pairing is incomplete")


def _json_record(row: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        record[str(key)] = value
    return record


def _copy_verified_file(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise ExperimentIntegrityError(f"missing source prerequisite: {source}")
    source_hash = sha256_file(source)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise ExperimentIntegrityError(
                f"existing Phase III-R prerequisite differs: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        "path": destination.name,
        "source_path": str(source),
        "sha256": source_hash,
        "size_bytes": source.stat().st_size,
    }


def _copy_verified_selection_cell(
    source_dir: Path, destination_dir: Path, expected_job: Mapping[str, Any]
) -> dict[str, Any]:
    state_path = source_dir / "complete.json"
    if not state_path.is_file():
        raise ExperimentIntegrityError(f"source cell is incomplete: {source_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    fingerprint = state.get("fingerprint")
    if not isinstance(fingerprint, dict) or fingerprint.get("job") != dict(expected_job):
        raise ExperimentIntegrityError(
            f"source cell job fingerprint differs: {source_dir.name}"
        )
    verify_completed_cell(state_path, fingerprint)
    if destination_dir.exists():
        destination_state = destination_dir / "complete.json"
        verify_completed_cell(destination_state, fingerprint)
    else:
        destination_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{destination_dir.name}.", dir=destination_dir.parent)
        )
        temporary_cell = temporary_root / destination_dir.name
        try:
            shutil.copytree(source_dir, temporary_cell)
            verify_completed_cell(temporary_cell / "complete.json", fingerprint)
            os.replace(temporary_cell, destination_dir)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
    if sha256_file(destination_dir / "complete.json") != sha256_file(state_path):
        raise ExperimentIntegrityError("reused completed-state hash differs after copy")
    return {
        "job_key": source_dir.name,
        "source_complete_sha256": sha256_file(state_path),
        "destination_complete_sha256": sha256_file(
            destination_dir / "complete.json"
        ),
        "source_path": str(source_dir),
        "destination_path": str(destination_dir),
        "copy_mode": "independent_copy",
    }


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    if path.exists():
        if _load_json(path) != normalized:
            raise ExperimentIntegrityError(f"frozen artifact differs: {path}")
    else:
        atomic_write_json(path, normalized)
    return sha256_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentIntegrityError(f"expected JSON object: {path}")
    return payload


def prepare_phase3_reduced(
    phase1_dir: str | Path,
    source_phase3_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Freeze Phase III-R and import only exact, hash-verified v1 checkpoints."""

    phase1_root = Path(phase1_dir)
    source = Path(source_phase3_dir)
    destination = Path(out_dir)
    verify_phase1_inputs(phase1_root)
    if (source / "selection_manifest.json").exists() or (
        source / "phase3_results.parquet"
    ).exists():
        raise ExperimentIntegrityError(
            "Phase III v1 accessed/froze post-selection state; amendment is forbidden"
        )
    test_claims = list(source.glob("evaluation_jobs/*/test_inference_claim.json"))
    if test_claims:
        raise ExperimentIntegrityError("Phase III v1 contains test-inference claims")
    for required in (
        "selection_job_inventory.parquet",
        "evaluation_job_inventory.parquet",
        "job_inventory_summary.json",
        "protocol_frozen.json",
        "protocol_frozen.sha256",
    ):
        if not (source / required).is_file():
            raise ExperimentIntegrityError(f"Phase III v1 artifact is missing: {required}")

    source_summary = _load_json(source / "job_inventory_summary.json")
    if sha256_file(source / "selection_job_inventory.parquet") != source_summary.get(
        "selection_inventory_sha256"
    ) or sha256_file(source / "evaluation_job_inventory.parquet") != source_summary.get(
        "evaluation_inventory_sha256"
    ):
        raise ExperimentIntegrityError("Phase III v1 inventory hash mismatch")

    subset_payload = _load_json(phase1_root / "subset_manifest.json")
    if sha256_file(phase1_root / "subset_manifest.json") != PHASE1_SUBSET_MANIFEST_SHA256:
        raise ExperimentIntegrityError("Phase-I subset manifest changed")
    selection, evaluation = build_phase3_reduced_inventory(subset_payload)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "selection_manifest.json").exists() or list(
        destination.glob("evaluation_jobs/*/test_inference_claim.json")
    ):
        raise ExperimentIntegrityError("refusing to re-prepare Phase III-R after test boundary")

    prerequisite_records = [
        _copy_verified_file(source / name, destination / name)
        for name in _PREREQUISITE_ARTIFACTS
    ]

    amendment_text = """# Experiment 01 Phase III-R — compute-feasible amendment

This amendment is subordinate to the later definitive Phase-III v1
specification,
[`SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md`](https://github.com/nicoloeffe/thesis_proposal/blob/main/docs/experiment01/SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md),
SHA-256
`78ca15821ac40355c35e5f40ecaf5086f5e6bbb6f339255a85b13fc7d952a151`.
That specification replaces the eligibility rule in the earlier optional MLP
section: the executed `b_1_4` floor is therefore eligible under the governing
Phase-III contract.

Phase III v1 was stopped for computational infeasibility before the selection
manifest was frozen and before any production test access. Phase III-R changes
only the job inventory. It preserves the frozen bundle, splits, subset row
identities, feature transforms, target blocks, MLP architecture, optimizer,
step schedule, weight-decay grid, validation-only selection, test boundary,
metrics, thresholds, bootstrap settings and R1--R4 outcome rules.

The primary grid keeps directional `last_concat512`, both branches, all three
encoder seeds, native/full-whitened coordinates, adjacent low budgets `b_1_4`
and `b_1_2`, full train, subset seeds 0--2 at low budgets and reader seeds
0--2. Volatility and timing remain separate controls at `b_1_4` and full train.
The spectral diagnostic is restricted to horizon-JEPA directional head
`1:127`, deep `382:508` and `full_valid_rank`. Capacity sensitivity is omitted.

The amendment was selected from scientific contrast requirements and measured
runtime/completeness only. Aggregate validation performance was not inspected.
During the post-confirmation implementation audit, one individual timing-cell
validation value was incidentally displayed while inspecting artifact schema;
it was not aggregated, interpreted or used to alter this already confirmed
inventory. Any insufficient precision yields R4; the grid is not expanded
after test access.
"""
    amendment_path = destination / "PROTOCOL_AMENDMENT_PHASE3_R.md"
    if amendment_path.exists():
        if amendment_path.read_text(encoding="utf-8") != amendment_text:
            raise ExperimentIntegrityError("Phase III-R amendment text differs")
    else:
        amendment_path.write_text(amendment_text, encoding="utf-8")

    selection_path = destination / "selection_job_inventory.parquet"
    evaluation_path = destination / "evaluation_job_inventory.parquet"
    atomic_write_parquet(selection, selection_path)
    atomic_write_parquet(evaluation, evaluation_path)
    inventory_summary = {
        "schema_name": "thesis.experiment01.phase3_reduced.job_inventory",
        "schema_version": 1,
        "variant": PHASE3_REDUCED_VARIANT,
        "logical_selection_cells": EXPECTED_LOGICAL_CELLS,
        "selection_models": len(selection),
        "evaluation_models": len(evaluation),
        "total_models": len(selection) + len(evaluation),
        "new_models_before_reuse": len(selection) + len(evaluation),
        "selection_seed": SELECTION_SEED,
        "reader_seeds": list(PHASE3_REDUCED_READER_SEEDS),
        "weight_decays": list(WEIGHT_DECAYS),
        "selection_by_family": {
            str(key): int(value)
            for key, value in selection.groupby("job_family").size().items()
        },
        "evaluation_by_family": {
            str(key): int(value)
            for key, value in evaluation.groupby("job_family").size().items()
        },
        "selection_inventory_sha256": sha256_file(selection_path),
        "evaluation_inventory_sha256": sha256_file(evaluation_path),
        "phase1_subset_manifest_sha256": PHASE1_SUBSET_MANIFEST_SHA256,
    }
    _write_once_json(destination / "job_inventory_summary.json", inventory_summary)

    source_complete = {
        path.parent.name: path.parent
        for path in source.glob("selection_jobs/*/complete.json")
    }
    reusable = selection.loc[
        selection["job_key"].astype(str).isin(source_complete)
    ].copy()
    if len(reusable) != EXPECTED_REUSABLE_SELECTION_MODELS:
        raise ExperimentIntegrityError(
            f"reusable selection count differs: {len(reusable)}"
        )
    reuse_records = []
    for row in reusable.sort_values("job_key", kind="stable").itertuples(index=False):
        series = pd.Series(row._asdict())
        job = _json_record(series)
        key = str(job["job_key"])
        reuse_records.append(
            _copy_verified_selection_cell(
                source_complete[key], destination / "selection_jobs" / key, job
            )
        )
    reuse_path = destination / "reused_selection_cells.parquet"
    atomic_write_parquet(pd.DataFrame(reuse_records), reuse_path)
    reuse_manifest = {
        "schema_name": "thesis.experiment01.phase3_reduced.checkpoint_reuse",
        "schema_version": 1,
        "status": "verified",
        "source_phase3_dir": str(source.resolve()),
        "reused_selection_models": len(reuse_records),
        "new_selection_models_remaining": len(selection) - len(reuse_records),
        "evaluation_models_remaining": len(evaluation),
        "new_training_models_remaining": len(selection)
        - len(reuse_records)
        + len(evaluation),
        "records_sha256": sha256_file(reuse_path),
    }
    _write_once_json(destination / "checkpoint_reuse_manifest.json", reuse_manifest)

    observed_runtimes = [
        float(
            json.loads(
                (destination / "selection_jobs" / record["job_key"] / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )["runtime_seconds"]
        )
        for record in reuse_records
    ]
    ordered_runtimes = sorted(observed_runtimes)
    mean_runtime = statistics.mean(observed_runtimes)
    compute_forecast = {
        "schema_name": "thesis.experiment01.phase3_reduced.compute_forecast",
        "schema_version": 1,
        "basis": "runtime metadata from exact reusable selection cells; no validation performance aggregation",
        "original_total_models": int(source_summary["total_models"]),
        "reduced_total_models": len(selection) + len(evaluation),
        "model_count_reduction_fraction": 1.0
        - (len(selection) + len(evaluation)) / int(source_summary["total_models"]),
        "reused_models": len(reuse_records),
        "new_models_remaining": reuse_manifest["new_training_models_remaining"],
        "observed_runtime_models": len(observed_runtimes),
        "observed_runtime_mean_seconds": mean_runtime,
        "observed_runtime_median_seconds": statistics.median(observed_runtimes),
        "observed_runtime_p90_seconds": ordered_runtimes[
            int(0.90 * (len(ordered_runtimes) - 1))
        ],
        "point_forecast_new_training_hours": reuse_manifest[
            "new_training_models_remaining"
        ]
        * mean_runtime
        / 3600.0,
        "operational_forecast_hours": [36.0, 60.0],
        "forecast_includes_reporting_overhead": True,
    }
    _write_once_json(destination / "reduced_compute_forecast.json", compute_forecast)

    termination = {
        "schema_name": "thesis.experiment01.phase3_v1.termination_record",
        "schema_version": 1,
        "status": "terminated_pre_test_compute_infeasible",
        "source_phase3_dir": str(source.resolve()),
        "selection_complete_models": len(source_complete),
        "selection_manifest_exists": False,
        "evaluation_complete_models": 0,
        "test_inference_claims": 0,
        "phase3_results_exists": False,
        "source_protocol_sha256": sha256_file(source / "protocol_frozen.json"),
        "source_selection_inventory_sha256": sha256_file(
            source / "selection_job_inventory.parquet"
        ),
        "termination_reason": "observed production runtime made the 21,456-model grid disproportionate to the diagnostic question",
    }
    _write_once_json(destination / "PHASE3_V1_TERMINATION_RECORD.json", termination)

    protocol = {
        "schema_name": "thesis.experiment01.phase3_reduced.protocol",
        "schema_version": 1,
        "status": "frozen_pre_test",
        "variant": PHASE3_REDUCED_VARIANT,
        "amendment_sha256": sha256_file(amendment_path),
        "source_v1_protocol_sha256": termination["source_protocol_sha256"],
        "phase1_outcome_unchanged": "A1",
        "primary": {
            "branches": list(PRIMARY_BRANCHES),
            "encoder_seeds": [0, 1, 2],
            "readout": PRIMARY_READOUT,
            "target_block": "directional",
            "transforms": ["native", "full_whitened"],
            "budget_labels": list(PHASE3_REDUCED_PRIMARY_BUDGETS),
            "low_budget_labels": ["b_1_4", "b_1_2"],
            "low_budget_subsample_seeds": list(
                PHASE3_REDUCED_LOW_SUBSAMPLE_SEEDS
            ),
        },
        "controls": {
            "target_blocks": ["volatility", "timing"],
            "budget_labels": list(PHASE3_REDUCED_CONTROL_BUDGETS),
            "role": "specificity_only",
        },
        "spectral": {
            "branches": ["jepa_horizon"],
            "target_blocks": ["directional"],
            "budget_labels": list(PHASE3_REDUCED_SPECTRAL_BUDGETS),
            "arms": list(PHASE3_REDUCED_SPECTRAL_ARMS),
            "valid_rank": VALID_RANK,
            "role": "secondary_head_deep_full_contrast",
        },
        "omitted": {
            "capacity_sensitivity": True,
            "intermediate_spectral_bands": ["band_128_254", "band_255_381"],
            "top_128": True,
            "supervised_and_timing_spectral_mlp": True,
        },
        "reader": {
            "architecture": "Linear(d,256)-GELU-Dropout(0.10)-Linear(256,T)",
            "reader_seeds": list(PHASE3_REDUCED_READER_SEEDS),
            "selection_seed": SELECTION_SEED,
            "weight_decay_grid": list(WEIGHT_DECAYS),
            "max_steps": 20_000,
            "min_steps": 1_000,
            "validation_interval": 500,
            "patience_evaluations": 6,
            "minimum_validation_improvement": 1e-5,
            "all_other_reader_semantics": "unchanged_from_phase3_v1",
        },
        "metrics_and_outcomes": {
            "ceiling_eligibility_threshold": 0.01,
            "robust_gap_delta": 0.10,
            "reader_attenuation_threshold": 0.50,
            "whitening_attenuation_threshold": 0.50,
            "robust_required_adjacent_levels": 2,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "outcomes": ["R1", "R2", "R3", "R4"],
            "insufficient_precision_outcome": "R4",
        },
        "inventory": {
            "selection_models": len(selection),
            "evaluation_models": len(evaluation),
            "total_models": len(selection) + len(evaluation),
            "selection_inventory_sha256": sha256_file(selection_path),
            "evaluation_inventory_sha256": sha256_file(evaluation_path),
        },
        "test_policy": "blocked_until_reduced_selection_manifest_frozen_and_hashed",
        "performance_based_adaptation": False,
        "aggregate_validation_performance_inspected": False,
        "incidental_single_schema_example_metric_exposed_after_grid_confirmation": True,
    }
    protocol_hash = _write_once_json(destination / "protocol_frozen.json", protocol)
    hash_path = destination / "protocol_frozen.sha256"
    expected_hash_text = f"{protocol_hash}  protocol_frozen.json\n"
    if hash_path.exists():
        if hash_path.read_text(encoding="utf-8") != expected_hash_text:
            raise ExperimentIntegrityError("Phase III-R protocol hash sidecar differs")
    else:
        hash_path.write_text(expected_hash_text, encoding="utf-8")

    preparation = {
        "schema_name": "thesis.experiment01.phase3_reduced.preparation",
        "schema_version": 1,
        "status": "pass",
        "variant": PHASE3_REDUCED_VARIANT,
        "protocol_sha256": protocol_hash,
        "inventory_summary_sha256": sha256_file(
            destination / "job_inventory_summary.json"
        ),
        "checkpoint_reuse_manifest_sha256": sha256_file(
            destination / "checkpoint_reuse_manifest.json"
        ),
        "termination_record_sha256": sha256_file(
            destination / "PHASE3_V1_TERMINATION_RECORD.json"
        ),
        "prerequisite_artifacts": prerequisite_records,
        "test_accessed": False,
    }
    _write_once_json(destination / "reduced_preparation_gate.json", preparation)
    return {
        "status": "pass",
        "variant": PHASE3_REDUCED_VARIANT,
        "selection_models": len(selection),
        "evaluation_models": len(evaluation),
        "reused_selection_models": len(reuse_records),
        "new_training_models_remaining": reuse_manifest[
            "new_training_models_remaining"
        ],
        "protocol_sha256": protocol_hash,
    }

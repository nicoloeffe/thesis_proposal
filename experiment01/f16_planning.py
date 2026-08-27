"""Post-pilot job inventory and compute/storage estimate for F16."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from experiment01.f16 import BUDGETS, F16IntegrityError, _atomic_write_parquet, _relative
from experiment01.io import atomic_write_json, canonical_json_sha256, sha256_file


EXPECTED_ROWS = {"b_1_4": 7_116, "b_1": 28_446, "b_4": 122_099, "b_16": 490_937}
STOCK_DAYS = {"b_1_4": 7, "b_1": 7, "b_4": 28, "b_16": 112}
BATCH_SIZE = 256
MAX_UPDATES = 39_060
MINIMUM_UPDATES = 4_000
PATIENCE_CHECKS = 8
VALIDATION_CADENCE = 500


def projected_updates(rows: int, pilot_updates: int = 6_500) -> tuple[int, int, int, int]:
    steps_per_pass = math.ceil(rows / BATCH_SIZE)
    epoch20_update = steps_per_pass * 20
    earliest_stop = max(MINIMUM_UPDATES, epoch20_update)
    pilot_pattern = max(int(pilot_updates), earliest_stop)
    return steps_per_pass, epoch20_update, min(pilot_pattern, MAX_UPDATES), MAX_UPDATES


def _verify_pilot_sources(repo_root: Path, complete: Mapping[str, Any]) -> None:
    for relative, record in complete["source_inventory"]["files"].items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise F16IntegrityError(f"training source changed after pilot: {relative}")


def build_pilot_report_and_inventory(
    repo_root: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    complete_path = output_root / "runs" / "b_1_4" / "seed0" / "complete.json"
    cohort_path = output_root / "f16_cohort_manifest.json"
    decision_path = output_root / "f16_cohort_decision.json"
    for path in (complete_path, cohort_path, decision_path):
        if not path.is_file():
            raise F16IntegrityError(f"missing F16 planning input: {path}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("failures"):
        raise F16IntegrityError("F16 pilot is not complete")
    if complete.get("test_accessed") is not False:
        raise F16IntegrityError("F16 pilot test barrier violation")
    if decision.get("status") != "passed" or cohort.get("selected_cap_per_stock_day") != 128:
        raise F16IntegrityError("F16 cohort gate is not frozen at cap 128")
    _verify_pilot_sources(repo_root, complete)

    history_record = complete["history"]
    history_path = repo_root / history_record["path"]
    if not history_path.is_file() or sha256_file(history_path) != history_record["sha256"]:
        raise F16IntegrityError("F16 pilot history drift")
    history = pd.read_parquet(history_path)
    initial_rows = history[history["global_update"].astype(int) == 0]
    if len(initial_rows) != 1:
        raise F16IntegrityError("F16 pilot history has no unique update-0 row")
    initial_validation_mse = float(initial_rows.iloc[0]["validation_mse"])

    seconds_per_update = float(complete["runtime"]["wall_seconds"]) / int(
        complete["final_update"]
    )
    jobs: list[dict[str, Any]] = []
    for budget in BUDGETS:
        rows = EXPECTED_ROWS[budget]
        steps, epoch20, scenario_updates, upper_updates = projected_updates(
            rows, pilot_updates=int(complete["final_update"])
        )
        for seed in (0, 1, 2):
            is_pilot = budget == "b_1_4" and seed == 0
            payload = {
                "budget": budget,
                "encoder_seed": seed,
                "train_rows": rows,
                "stock_days": STOCK_DAYS[budget],
                "label_manifest_sha256": cohort["label_budgets"][budget]["sha256"],
                "validation_manifest_sha256": cohort["cohorts"]["validation"]["sha256"],
                "source_fingerprint": complete["source_fingerprint"],
                "maximum_updates": MAX_UPDATES,
            }
            jobs.append(
                {
                    "job_id": canonical_json_sha256(payload),
                    **payload,
                    "steps_per_pass": steps,
                    "epoch20_update": epoch20,
                    "early_stop_eligible_update": max(MINIMUM_UPDATES, epoch20),
                    "pilot_pattern_updates": scenario_updates,
                    "upper_bound_updates": upper_updates,
                    "pilot_pattern_runtime_seconds": scenario_updates * seconds_per_update,
                    "upper_bound_runtime_seconds": upper_updates * seconds_per_update,
                    "status": "complete" if is_pilot else "pending_authorization",
                    "is_pilot": is_pilot,
                    "test_access_permitted": False,
                }
            )
    inventory = pd.DataFrame(jobs)
    if len(inventory) != 12 or inventory["job_id"].duplicated().any():
        raise F16IntegrityError("invalid F16 production job inventory")
    inventory_path = output_root / "f16_job_inventory.parquet"
    _atomic_write_parquet(inventory, inventory_path)

    remaining = inventory[~inventory["is_pilot"]]
    training_bytes_per_run = int(complete["runtime"]["run_output_bytes"])
    label_b16 = pd.read_parquet(
        repo_root / cohort["label_budgets"]["b_16"]["path"], columns=["row_key"]
    )
    covariance_train = pd.read_parquet(
        repo_root / cohort["cohorts"]["train"]["path"], columns=["row_key"]
    )
    train_union_rows = len(set(label_b16["row_key"]) | set(covariance_train["row_key"]))
    evaluation_rows = int(cohort["cohorts"]["validation"]["rows"]) + int(
        cohort["cohorts"]["test"]["rows"]
    )
    dense_feature_bytes_per_encoder = (train_union_rows + evaluation_rows) * 512 * 4 * 2
    report: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_pilot_benchmark",
        "schema_version": 1,
        "status": "pilot_complete_production_awaiting_authorization",
        "pilot": {
            "budget": complete["budget"],
            "encoder_seed": complete["encoder_seed"],
            "train_rows": complete["train_rows"],
            "validation_rows": complete["validation_rows"],
            "final_update": complete["final_update"],
            "best_update": complete["best_update"],
            "epoch20_update": complete["epoch20_update"],
            "best_validation_mse": complete["best_validation_mse"],
            "initial_validation_mse": initial_validation_mse,
            "validation_improvement": initial_validation_mse
            - float(complete["best_validation_mse"]),
            "stop_reason": complete["stop_reason"],
            "wall_seconds": complete["runtime"]["wall_seconds"],
            "seconds_per_update_including_validation": seconds_per_update,
            "peak_ram_bytes": complete["runtime"]["peak_ram_bytes"],
            "peak_vram_bytes": complete["runtime"]["peak_vram_bytes"],
            "output_bytes": complete["runtime"]["run_output_bytes"],
            "checkpoint_reload_mse": complete[
                "selected_checkpoint_reload_validation_mse"
            ],
            "checkpoint_reload_exact_within_1e_8": True,
            "unstable_by_preregistered_rule": (
                initial_validation_mse - float(complete["best_validation_mse"]) < 0.01
            ),
            "test_accessed": False,
        },
        "cohort": {
            "selected_cap_per_stock_day": 128,
            "covariance_train_rows": cohort["cohorts"]["train"]["rows"],
            "validation_rows": cohort["cohorts"]["validation"]["rows"],
            "sealed_test_rows": cohort["cohorts"]["test"]["rows"],
            "all_504_convergence_cells_passed": (
                sum(int(value) for value in decision["failed_rows_by_cap"].values()) == 0
                and sum(int(value) for value in decision["rows_by_cap"].values()) == 504
            ),
        },
        "remaining_training_estimate": {
            "pending_cells": len(remaining),
            "pilot_pattern_seconds": float(
                remaining["pilot_pattern_runtime_seconds"].sum()
            ),
            "pilot_pattern_hours": float(
                remaining["pilot_pattern_runtime_seconds"].sum() / 3600.0
            ),
            "maximum_cap_seconds": float(
                remaining["upper_bound_runtime_seconds"].sum()
            ),
            "maximum_cap_hours": float(
                remaining["upper_bound_runtime_seconds"].sum() / 3600.0
            ),
            "interpretation": (
                "pilot-pattern assumes each cell stops at max(6500, epoch20); "
                "maximum-cap assumes every pending cell reaches 39060 updates"
            ),
        },
        "storage_estimate": {
            "checkpoint_outputs_all_12_bytes": training_bytes_per_run * 12,
            "dense_two_readout_features_per_encoder_bytes_if_materialized": dense_feature_bytes_per_encoder,
            "dense_two_readout_features_all_12_bytes_if_materialized": dense_feature_bytes_per_encoder
            * 12,
            "production_policy": (
                "do not persist dense full-cohort matrices; extract sequentially "
                "and retain sufficient/grouped statistics plus hashes"
            ),
            "expected_persistent_f16_output_upper_bytes": 2_000_000_000,
        },
        "inventory": {
            "path": _relative(inventory_path, repo_root),
            "sha256": sha256_file(inventory_path),
            "size_bytes": inventory_path.stat().st_size,
            "rows": len(inventory),
        },
        "pilot_complete_path": _relative(complete_path, repo_root),
        "pilot_complete_sha256": sha256_file(complete_path),
        "cohort_manifest_sha256": sha256_file(cohort_path),
        "source_fingerprint": complete["source_fingerprint"],
        "production_grid_authorized": False,
        "test_barrier": "locked",
        "failures": [],
    }
    report["manifest_fingerprint"] = canonical_json_sha256(report)
    atomic_write_json(output_root / "f16_pilot_benchmark.json", report)

    failures = pd.DataFrame(
        columns=[
            "job_id",
            "budget",
            "encoder_seed",
            "stage",
            "error_type",
            "error",
            "terminal",
        ]
    )
    _atomic_write_parquet(failures, output_root / "f16_failures.parquet")
    return inventory, report

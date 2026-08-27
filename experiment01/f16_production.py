"""Fail-closed production orchestration for the frozen F16 training grid."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from experiment01.f16 import BUDGETS, F16IntegrityError, _relative
from experiment01.f16_training import train_f16_cell, write_failure
from experiment01.io import (
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_sha256,
    sha256_file,
)


AUTHORIZATION_FILENAME = "f16_production_authorization.json"
PROGRESS_FILENAME = "f16_production_progress.json"
FAILURES_FILENAME = "f16_failures.parquet"
GRID = tuple((budget, seed) for budget in BUDGETS for seed in (0, 1, 2))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise F16IntegrityError(f"missing F16 production artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pinned_inputs(repo_root: Path, output_root: Path) -> dict[str, Any]:
    pilot_path = output_root / "f16_pilot_benchmark.json"
    inventory_path = output_root / "f16_job_inventory.parquet"
    cohort_path = output_root / "f16_cohort_manifest.json"
    decision_path = output_root / "f16_cohort_decision.json"
    protocol_path = output_root / "f16_manifest.json"
    for path in (pilot_path, inventory_path, cohort_path, decision_path, protocol_path):
        if not path.is_file():
            raise F16IntegrityError(f"missing F16 production input: {path}")
    pilot = _read_json(pilot_path)
    if pilot.get("status") != "pilot_complete_production_awaiting_authorization":
        raise F16IntegrityError("F16 pilot is not awaiting production authorization")
    if pilot.get("production_grid_authorized") is not False:
        raise F16IntegrityError("frozen pilot report authorization field drift")
    if pilot.get("test_barrier") != "locked" or pilot["pilot"].get("test_accessed") is not False:
        raise F16IntegrityError("F16 test barrier was not preserved by the pilot")
    inventory = pd.read_parquet(inventory_path)
    observed = list(zip(inventory["budget"], inventory["encoder_seed"].astype(int)))
    if observed != list(GRID) or len(inventory) != 12 or inventory["job_id"].duplicated().any():
        raise F16IntegrityError("F16 job inventory is not the frozen 4x3 grid")
    if sha256_file(inventory_path) != pilot["inventory"]["sha256"]:
        raise F16IntegrityError("F16 job inventory drift")
    if sha256_file(cohort_path) != pilot["cohort_manifest_sha256"]:
        raise F16IntegrityError("F16 cohort manifest drift after pilot")
    return {
        "pilot_benchmark": {
            "path": _relative(pilot_path, repo_root),
            "sha256": sha256_file(pilot_path),
        },
        "job_inventory": {
            "path": _relative(inventory_path, repo_root),
            "sha256": sha256_file(inventory_path),
        },
        "cohort_manifest": {
            "path": _relative(cohort_path, repo_root),
            "sha256": sha256_file(cohort_path),
        },
        "cohort_decision": {
            "path": _relative(decision_path, repo_root),
            "sha256": sha256_file(decision_path),
        },
        "protocol_manifest": {
            "path": _relative(protocol_path, repo_root),
            "sha256": sha256_file(protocol_path),
        },
        "source_fingerprint": pilot["source_fingerprint"],
        "pilot_complete_sha256": pilot["pilot_complete_sha256"],
    }


def authorize_production_grid(
    repo_root: Path,
    output_root: Path,
    *,
    authorization_text: str,
) -> dict[str, Any]:
    """Record explicit authorization without mutating preregistered artifacts."""
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    normalized = authorization_text.strip()
    if not normalized:
        raise ValueError("authorization_text must be non-empty")
    pins = _pinned_inputs(repo_root, output_root)
    payload: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_production_authorization",
        "schema_version": 1,
        "status": "authorized",
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_text": normalized,
        "scope": {
            "training_cells": 12,
            "already_complete_pilot_cells": 1,
            "remaining_cells_authorized": 11,
            "training_only": True,
            "test_access_permitted": False,
            "phase1_outcome_mutation_permitted": False,
        },
        "pinned_inputs": pins,
        "test_barrier": "locked",
    }
    payload["authorization_fingerprint"] = canonical_json_sha256(payload)
    path = output_root / AUTHORIZATION_FILENAME
    if path.is_file():
        existing = _read_json(path)
        # Authorization is append-never and idempotent once recorded.
        comparable = dict(existing)
        fingerprint = comparable.pop("authorization_fingerprint", None)
        if fingerprint != canonical_json_sha256(comparable):
            raise F16IntegrityError("existing F16 authorization fingerprint drift")
        if existing.get("pinned_inputs") != pins or existing.get("test_barrier") != "locked":
            raise F16IntegrityError("existing F16 authorization no longer matches frozen inputs")
        return existing
    atomic_write_json(path, payload)
    return payload


def _verify_authorization(repo_root: Path, output_root: Path) -> dict[str, Any]:
    authorization = _read_json(output_root / AUTHORIZATION_FILENAME)
    fingerprint = authorization.get("authorization_fingerprint")
    unsigned = dict(authorization)
    unsigned.pop("authorization_fingerprint", None)
    if fingerprint != canonical_json_sha256(unsigned):
        raise F16IntegrityError("F16 production authorization fingerprint drift")
    if authorization.get("status") != "authorized" or authorization.get("test_barrier") != "locked":
        raise F16IntegrityError("F16 production grid is not authorized with a locked test")
    if authorization.get("scope", {}).get("test_access_permitted") is not False:
        raise F16IntegrityError("F16 production authorization permits forbidden test access")
    if authorization.get("pinned_inputs") != _pinned_inputs(repo_root, output_root):
        raise F16IntegrityError("F16 production input drift after authorization")
    return authorization


def _cell_state(output_root: Path, budget: str, seed: int) -> tuple[str, dict[str, Any] | None]:
    run_dir = output_root / "runs" / budget / f"seed{seed}"
    complete_path = run_dir / "complete.json"
    failure_path = run_dir / "failure.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("status") != "complete" or complete.get("test_accessed") is not False:
            raise F16IntegrityError(f"invalid completed F16 cell {budget}/seed{seed}")
        return "complete", complete
    if failure_path.is_file():
        return "failed", _read_json(failure_path)
    if (run_dir / "last.pt").is_file():
        return "resumable", None
    return "pending", None


def production_status(repo_root: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    cells = []
    for budget, seed in GRID:
        state, record = _cell_state(output_root, budget, seed)
        cells.append(
            {
                "budget": budget,
                "encoder_seed": seed,
                "status": state,
                "final_update": record.get("final_update") if record else None,
                "best_update": record.get("best_update") if record else None,
            }
        )
    counts = pd.Series([cell["status"] for cell in cells]).value_counts().to_dict()
    return {
        "schema_name": "thesis.experiment01.f16_production_progress",
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if counts.get("complete", 0) == len(GRID) else "running_or_interrupted",
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "cells": cells,
        "authorization_sha256": (
            sha256_file(output_root / AUTHORIZATION_FILENAME)
            if (output_root / AUTHORIZATION_FILENAME).is_file()
            else None
        ),
        "test_barrier": "locked",
        "test_accessed": False,
    }


def _write_progress(repo_root: Path, output_root: Path, active_cell: Mapping[str, Any] | None) -> dict[str, Any]:
    progress = production_status(repo_root, output_root)
    progress["active_cell"] = dict(active_cell) if active_cell is not None else None
    atomic_write_json(output_root / PROGRESS_FILENAME, progress)
    return progress


def _freeze_failures(repo_root: Path, output_root: Path) -> pd.DataFrame:
    rows = []
    inventory = pd.read_parquet(output_root / "f16_job_inventory.parquet")
    jobs = {(str(row.budget), int(row.encoder_seed)): str(row.job_id) for row in inventory.itertuples()}
    for budget, seed in GRID:
        failure_path = output_root / "runs" / budget / f"seed{seed}" / "failure.json"
        if not failure_path.is_file():
            continue
        failure = _read_json(failure_path)
        rows.append(
            {
                "job_id": jobs[(budget, seed)],
                "budget": budget,
                "encoder_seed": seed,
                "stage": "training",
                "error_type": failure.get("error_type"),
                "error": failure.get("error"),
                "terminal": True,
            }
        )
    failures = pd.DataFrame(
        rows,
        columns=("job_id", "budget", "encoder_seed", "stage", "error_type", "error", "terminal"),
    )
    atomic_write_parquet(failures, output_root / FAILURES_FILENAME)
    return failures


def run_production_grid(
    repo_root: Path,
    output_root: Path,
    bundle_root: Path,
    dataset_path: Path,
    checkpoint_manifest_path: Path,
    *,
    device_name: str = "cuda",
    num_workers: int = 2,
    trainer: Callable[..., Mapping[str, Any]] = train_f16_cell,
) -> dict[str, Any]:
    """Run or resume the grid in frozen order, stopping at the first failure."""
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    _verify_authorization(repo_root, output_root)
    _write_progress(repo_root, output_root, None)
    for budget, seed in GRID:
        state, _record = _cell_state(output_root, budget, seed)
        if state == "complete":
            continue
        if state == "failed":
            raise F16IntegrityError(f"terminal F16 failure already recorded for {budget}/seed{seed}")
        active = {"budget": budget, "encoder_seed": seed, "resume": state == "resumable"}
        _write_progress(repo_root, output_root, active)
        print(f"F16 production: starting {budget}/seed{seed} resume={active['resume']}", flush=True)
        try:
            complete = trainer(
                repo_root,
                output_root,
                bundle_root,
                dataset_path,
                checkpoint_manifest_path,
                budget=budget,
                seed=seed,
                device_name=device_name,
                num_workers=num_workers,
            )
        except BaseException as exc:
            write_failure(repo_root, output_root, budget, seed, exc)
            _freeze_failures(repo_root, output_root)
            _write_progress(repo_root, output_root, None)
            raise
        if complete.get("status") != "complete" or complete.get("test_accessed") is not False:
            raise F16IntegrityError(f"F16 trainer returned invalid completion for {budget}/seed{seed}")
        _write_progress(repo_root, output_root, None)
    failures = _freeze_failures(repo_root, output_root)
    if not failures.empty:
        raise F16IntegrityError("F16 grid ended with recorded training failures")
    progress = _write_progress(repo_root, output_root, None)
    if progress["counts"] != {"complete": 12}:
        raise F16IntegrityError("F16 grid did not finish all 12 cells")
    print("F16 production: all 12 training cells complete; test remains locked", flush=True)
    return progress

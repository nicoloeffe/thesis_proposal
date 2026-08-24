#!/usr/bin/env python3
"""Run Experiment 01 Phase III locally without an active Codex session.

The scientific commands remain implemented by
``scripts.experiment01.run_experiment_01_phase3``. This module only provides
deterministic, fail-closed orchestration around those commands: a single-process
lock, durable logs, a heartbeat, and stage checks.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA_NAME = "thesis.experiment01.phase3.local_runner"
SCHEMA_VERSION = 1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _selection_complete(out_dir: Path) -> bool:
    path = out_dir / "selection_manifest.json"
    if not path.is_file():
        return False
    payload = _load_json(path)
    return payload.get("status") == "frozen" and bool(payload.get("records"))


def _evaluation_complete(out_dir: Path) -> bool:
    results = out_dir / "phase3_results.parquet"
    log_path = out_dir / "evaluation_compute_log.json"
    if not results.is_file() or not log_path.is_file():
        return False
    payload = _load_json(log_path)
    return (
        payload.get("status") == "complete"
        and payload.get("failures") == 0
        and payload.get("results_sha256") == _sha256(results)
    )


def _summary_complete(out_dir: Path) -> bool:
    path = out_dir / "summary.json"
    if not path.is_file():
        return False
    payload = _load_json(path)
    if payload.get("status") != "complete":
        return False
    tables = payload.get("tables")
    if not isinstance(tables, dict) or not tables:
        return False
    return all(
        (out_dir / name).is_file() and _sha256(out_dir / name) == digest
        for name, digest in tables.items()
    )


def _finalization_complete(out_dir: Path) -> bool:
    manifest_path = out_dir / "phase3_manifest.json"
    report_path = out_dir / "REPORT_EXPERIMENT_01_PHASE3.md"
    metadata_path = out_dir / "metadata.json"
    if not all(path.is_file() for path in (manifest_path, report_path, metadata_path)):
        return False
    manifest = _load_json(manifest_path)
    metadata = _load_json(metadata_path)
    return manifest.get("status") == "complete" and metadata.get("status") == "complete"


def _current_scientific_progress(out_dir: Path, stage: str) -> dict[str, Any] | None:
    name = {
        "selection": "selection_progress.json",
        "evaluation": "evaluation_progress.json",
    }.get(stage)
    if name is None:
        return None
    path = out_dir / name
    try:
        return _load_json(path) if path.is_file() else None
    except (OSError, ValueError, RuntimeError):
        return {"status": "temporarily_unreadable"}


def _validate_inputs(args: argparse.Namespace) -> None:
    required_dirs = (args.bundle, args.phase1_dir, args.phase2_dir, args.out_dir)
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise RuntimeError(f"missing required directories: {missing_dirs}")
    if importlib.util.find_spec(args.phase3_module) is None:
        raise RuntimeError(f"Phase-III module not found: {args.phase3_module}")
    if not Path(sys.executable).is_file():
        raise RuntimeError(f"Python interpreter not found: {sys.executable}")
    prerequisite_files = (
        args.out_dir / "protocol_frozen.json",
        args.out_dir / "protocol_frozen.sha256",
        args.out_dir / "preproduction_gates.json",
        args.out_dir / "selection_job_inventory.parquet",
        args.out_dir / "evaluation_job_inventory.parquet",
        args.out_dir / "job_inventory_summary.json",
    )
    missing = [str(path) for path in prerequisite_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase-III prerequisites are incomplete: {missing}")
    gates = _load_json(args.out_dir / "preproduction_gates.json")
    if gates.get("status") != "pass":
        raise RuntimeError("Phase-III preproduction gate is not passing")


def _stage_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    common = [
        "--bundle",
        str(args.bundle),
        "--phase1-dir",
        str(args.phase1_dir),
    ]
    return [
        {
            "name": "selection",
            "command": [
                sys.executable,
                "-m",
                args.phase3_module,
                "select",
                *common,
                "--out-dir",
                str(args.out_dir),
                "--device",
                args.device,
            ],
            "complete": lambda: _selection_complete(args.out_dir),
            "always_run": True,
        },
        {
            "name": "evaluation",
            "command": [
                sys.executable,
                "-m",
                args.phase3_module,
                "evaluate",
                *common,
                "--out-dir",
                str(args.out_dir),
                "--device",
                args.device,
            ],
            "complete": lambda: _evaluation_complete(args.out_dir),
            "always_run": False,
        },
        {
            "name": "summarize",
            "command": [
                sys.executable,
                "-m",
                args.phase3_module,
                "summarize",
                *common,
                "--phase2-dir",
                str(args.phase2_dir),
                "--out-dir",
                str(args.out_dir),
            ],
            "complete": lambda: _summary_complete(args.out_dir),
            "always_run": False,
        },
        {
            "name": "finalize",
            "command": [
                sys.executable,
                "-m",
                args.phase3_module,
                "finalize",
                *common,
                "--phase2-dir",
                str(args.phase2_dir),
                "--out-dir",
                str(args.out_dir),
            ],
            "complete": lambda: _finalization_complete(args.out_dir),
            "always_run": False,
        },
    ]


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    execution_root = project_root / "validation/experiment01/execution_20260730"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=project_root / "validation/experiment01_bundle_20260730",
    )
    parser.add_argument("--phase1-dir", type=Path, default=execution_root / "phase1")
    parser.add_argument("--phase2-dir", type=Path, default=execution_root / "phase2")
    parser.add_argument("--out-dir", type=Path, default=execution_root / "phase3")
    parser.add_argument(
        "--runner-dir", type=Path, default=execution_root / "phase3_runner"
    )
    parser.add_argument(
        "--phase3-module",
        default="scripts.experiment01.run_experiment_01_phase3",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args()
    for name in (
        "bundle",
        "phase1_dir",
        "phase2_dir",
        "out_dir",
        "runner_dir",
    ):
        setattr(parsed, name, getattr(parsed, name).expanduser().resolve())
    if parsed.heartbeat_seconds < 5:
        parser.error("--heartbeat-seconds must be at least 5")
    return parsed


def main() -> int:
    args = _parse_args()
    _validate_inputs(args)
    stages = _stage_commands(args)
    plan = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "python": sys.executable,
        "project_root": str(Path(__file__).resolve().parents[2]),
        "bundle": str(args.bundle),
        "phase1_dir": str(args.phase1_dir),
        "phase2_dir": str(args.phase2_dir),
        "out_dir": str(args.out_dir),
        "runner_dir": str(args.runner_dir),
        "device": args.device,
        "phase3_module": args.phase3_module,
        "stages": [
            {
                "name": stage["name"],
                "already_complete": bool(stage["complete"]()),
                "command": stage["command"],
            }
            for stage in stages
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    args.runner_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.runner_dir / "runner.lock"
    status_path = args.runner_dir / "status.json"
    events_path = args.runner_dir / "events.jsonl"
    log_dir = args.runner_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another Phase-III local runner already holds the lock") from exc

    status: dict[str, Any] = {
        **plan,
        "state": "running",
        "pid": os.getpid(),
        "started_at": _now(),
        "updated_at": _now(),
        "current_stage": None,
        "completed_stages": [],
        "skipped_stages": [],
        "failure": None,
    }
    _atomic_json(status_path, status)
    _append_event(events_path, {"event": "runner_started", "at": _now(), "pid": os.getpid()})

    child: subprocess.Popen[bytes] | None = None
    stop_signal: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_signal
        stop_signal = signum
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("CODEX_API_KEY", None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("MPLCONFIGDIR", str(args.runner_dir / "matplotlib"))

    try:
        for stage in stages:
            name = str(stage["name"])
            if not stage["always_run"] and stage["complete"]():
                status["skipped_stages"].append(name)
                status["updated_at"] = _now()
                _atomic_json(status_path, status)
                _append_event(
                    events_path,
                    {"event": "stage_skipped_already_complete", "stage": name, "at": _now()},
                )
                continue
            status["current_stage"] = name
            status["stage_started_at"] = _now()
            status["updated_at"] = _now()
            status["scientific_progress"] = _current_scientific_progress(
                args.out_dir, name
            )
            _atomic_json(status_path, status)
            _append_event(events_path, {"event": "stage_started", "stage": name, "at": _now()})
            log_path = log_dir / f"{name}.log"
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"\n[{_now()}] START {name}\n"
                    + "COMMAND "
                    + json.dumps(stage["command"])
                    + "\n"
                )
                log_handle.flush()
                child = subprocess.Popen(
                    stage["command"],
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                status["child_pid"] = child.pid
                _atomic_json(status_path, status)
                stage_started_monotonic = time.monotonic()
                while True:
                    try:
                        return_code = int(
                            child.wait(timeout=args.heartbeat_seconds)
                        )
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    status["updated_at"] = _now()
                    status["stage_elapsed_seconds"] = max(
                        0.0, time.monotonic() - stage_started_monotonic
                    )
                    status["scientific_progress"] = _current_scientific_progress(
                        args.out_dir, name
                    )
                    _atomic_json(status_path, status)
                log_handle.write(f"[{_now()}] END {name} return_code={return_code}\n")
                log_handle.flush()
            child = None
            status.pop("child_pid", None)
            if stop_signal is not None:
                raise RuntimeError(f"runner interrupted by signal {stop_signal}")
            if return_code != 0:
                raise RuntimeError(
                    f"stage {name} failed with return code {return_code}; see {log_path}"
                )
            if not stage["complete"]():
                raise RuntimeError(
                    f"stage {name} returned success but its completion gate failed"
                )
            status["completed_stages"].append(name)
            status["current_stage"] = None
            status["updated_at"] = _now()
            status["scientific_progress"] = None
            _atomic_json(status_path, status)
            _append_event(events_path, {"event": "stage_completed", "stage": name, "at": _now()})

        status["state"] = "complete"
        status["current_stage"] = None
        status["completed_at"] = _now()
        status["updated_at"] = _now()
        _atomic_json(status_path, status)
        _append_event(events_path, {"event": "runner_completed", "at": _now()})
        return 0
    except BaseException as exc:
        status["state"] = "failed"
        status["failure"] = repr(exc)
        status["failed_at"] = _now()
        status["updated_at"] = _now()
        _atomic_json(status_path, status)
        _append_event(
            events_path,
            {"event": "runner_failed", "at": _now(), "failure": repr(exc)},
        )
        print(f"Phase-III local runner failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

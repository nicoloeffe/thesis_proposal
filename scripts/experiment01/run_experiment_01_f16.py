#!/usr/bin/env python3
"""Experiment 01 F16 orchestration (preregistration and staged execution)."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment01.f16 import freeze_f16_candidates
from experiment01.f16_convergence import run_convergence_gate
from experiment01.f16_training import train_f16_cell, write_failure
from experiment01.f16_planning import build_pilot_report_and_inventory
from experiment01.f16_production import (
    authorize_production_grid,
    production_status,
    run_production_grid,
)
from experiment01.f16_evaluation import (
    analyze_f16_validation,
    extract_f16_validation_statistics,
    freeze_f16_checkpoints,
)
from experiment01.f16_test import run_f16_fixed_test, unlock_f16_test
from experiment01.f16_reporting import report_f16
from experiment01.f16_posttest_threshold import summarize_f16_boundary_corrected


def _path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-candidates")
    freeze.add_argument("--repo-root", type=Path, default=Path("."))
    freeze.add_argument(
        "--bundle", type=Path, default=Path("validation/experiment01_bundle_20260730")
    )
    freeze.add_argument(
        "--subsets",
        type=Path,
        default=Path(
            "validation/experiment01/execution_20260730/phase1/subset_manifests"
        ),
    )
    freeze.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/experiment01/SPEC_EXPERIMENT_01_F16_LABEL_MATCHED.md"),
    )
    freeze.add_argument(
        "--training-audit",
        type=Path,
        default=Path("docs/experiment01/TRAINING_PROTOCOL_AUDIT.json"),
    )
    freeze.add_argument(
        "--out",
        type=Path,
        default=Path("validation/experiment01/f16_20260826"),
    )
    convergence = subparsers.add_parser("convergence-gate")
    convergence.add_argument("--repo-root", type=Path, default=Path("."))
    convergence.add_argument(
        "--bundle", type=Path, default=Path("validation/experiment01_bundle_20260730")
    )
    convergence.add_argument(
        "--phase1",
        type=Path,
        default=Path("validation/experiment01/execution_20260730/phase1"),
    )
    convergence.add_argument(
        "--phase2",
        type=Path,
        default=Path("validation/experiment01/execution_20260730/phase2"),
    )
    convergence.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    convergence.add_argument("--chunk-rows", type=int, default=32768)
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--repo-root", type=Path, default=Path("."))
    pilot.add_argument(
        "--bundle", type=Path, default=Path("validation/experiment01_bundle_20260730")
    )
    pilot.add_argument(
        "--dataset", type=Path, default=Path("data/lobench_processed.npz")
    )
    pilot.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=Path("docs/experiment01/CHECKPOINTS_MULTISEED_MANIFEST.json"),
    )
    pilot.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    pilot.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    pilot.add_argument("--num-workers", type=int, default=2)
    pilot_report = subparsers.add_parser("pilot-report")
    pilot_report.add_argument("--repo-root", type=Path, default=Path("."))
    pilot_report.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    authorize = subparsers.add_parser("authorize-production")
    authorize.add_argument("--repo-root", type=Path, default=Path("."))
    authorize.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    authorize.add_argument("--authorization-text", required=True)
    production = subparsers.add_parser("production-grid")
    production.add_argument("--repo-root", type=Path, default=Path("."))
    production.add_argument(
        "--bundle", type=Path, default=Path("validation/experiment01_bundle_20260730")
    )
    production.add_argument(
        "--dataset", type=Path, default=Path("data/lobench_processed.npz")
    )
    production.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=Path("docs/experiment01/CHECKPOINTS_MULTISEED_MANIFEST.json"),
    )
    production.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    production.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    production.add_argument("--num-workers", type=int, default=2)
    status = subparsers.add_parser("production-status")
    status.add_argument("--repo-root", type=Path, default=Path("."))
    status.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    freeze_checkpoints = subparsers.add_parser("freeze-checkpoints")
    freeze_checkpoints.add_argument("--repo-root", type=Path, default=Path("."))
    freeze_checkpoints.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    extract_validation = subparsers.add_parser("extract-validation")
    extract_validation.add_argument("--repo-root", type=Path, default=Path("."))
    extract_validation.add_argument(
        "--bundle", type=Path, default=Path("validation/experiment01_bundle_20260730")
    )
    extract_validation.add_argument(
        "--dataset", type=Path, default=Path("data/lobench_processed.npz")
    )
    extract_validation.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    extract_validation.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    extract_validation.add_argument("--batch-size", type=int, default=512)
    extract_validation.add_argument("--num-workers", type=int, default=2)
    extract_validation.add_argument("--chunk-rows", type=int, default=8192)
    analyze_validation = subparsers.add_parser("analyze-validation")
    analyze_validation.add_argument("--repo-root", type=Path, default=Path("."))
    analyze_validation.add_argument(
        "--bundle", type=Path, default=Path("validation/experiment01_bundle_20260730")
    )
    analyze_validation.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    unlock_test = subparsers.add_parser("unlock-test")
    unlock_test.add_argument("--repo-root", type=Path, default=Path("."))
    unlock_test.add_argument(
        "--bundle", type=Path, default=Path("validation/experiment01_bundle_20260730")
    )
    unlock_test.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    run_test = subparsers.add_parser("run-test")
    run_test.add_argument("--repo-root", type=Path, default=Path("."))
    run_test.add_argument(
        "--bundle", type=Path, default=Path("validation/experiment01_bundle_20260730")
    )
    run_test.add_argument(
        "--dataset", type=Path, default=Path("data/lobench_processed.npz")
    )
    run_test.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    run_test.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run_test.add_argument("--batch-size", type=int, default=512)
    run_test.add_argument("--num-workers", type=int, default=2)
    run_test.add_argument("--chunk-rows", type=int, default=8192)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--repo-root", type=Path, default=Path("."))
    summarize.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    report = subparsers.add_parser("report")
    report.add_argument("--repo-root", type=Path, default=Path("."))
    report.add_argument(
        "--out", type=Path, default=Path("validation/experiment01/f16_20260826")
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.command == "freeze-candidates":
        protocol, cohorts = freeze_f16_candidates(
            root,
            _path(root, args.bundle),
            _path(root, args.subsets),
            _path(root, args.spec),
            _path(root, args.training_audit),
            _path(root, args.out),
        )
        print(f"status={protocol['status']}")
        print(f"candidate_caps={cohorts['candidate_caps_per_stock_day']}")
        for split, record in cohorts["cohorts"].items():
            print(f"{split}_max_rows={record['rows']}")
    elif args.command == "convergence-gate":
        results, decision = run_convergence_gate(
            root,
            _path(root, args.out),
            _path(root, args.bundle),
            _path(root, args.phase1),
            _path(root, args.phase2),
            chunk_rows=args.chunk_rows,
        )
        print(f"status={decision['status']}")
        print(f"selected_cap={decision['selected_cap_per_stock_day']}")
        print(f"rows={len(results)}")
    elif args.command == "pilot":
        try:
            complete = train_f16_cell(
                root,
                _path(root, args.out),
                _path(root, args.bundle),
                _path(root, args.dataset),
                _path(root, args.checkpoint_manifest),
                budget="b_1_4",
                seed=0,
                device_name=args.device,
                num_workers=args.num_workers,
            )
        except BaseException as exc:
            write_failure(root, _path(root, args.out), "b_1_4", 0, exc)
            raise
        print(f"status={complete['status']}")
        print(f"final_update={complete['final_update']}")
        print(f"best_update={complete['best_update']}")
        print(f"wall_seconds={complete['runtime']['wall_seconds']:.3f}")
        print(f"peak_vram_bytes={complete['runtime']['peak_vram_bytes']}")
    elif args.command == "pilot-report":
        inventory, report = build_pilot_report_and_inventory(root, _path(root, args.out))
        estimate = report["remaining_training_estimate"]
        print(f"status={report['status']}")
        print(f"jobs={len(inventory)}")
        print(f"pending={estimate['pending_cells']}")
        print(f"pilot_pattern_hours={estimate['pilot_pattern_hours']:.3f}")
        print(f"maximum_cap_hours={estimate['maximum_cap_hours']:.3f}")
    elif args.command == "authorize-production":
        authorization = authorize_production_grid(
            root,
            _path(root, args.out),
            authorization_text=args.authorization_text,
        )
        print(f"status={authorization['status']}")
        print(f"authorization_fingerprint={authorization['authorization_fingerprint']}")
        print("test_barrier=locked")
    elif args.command == "production-grid":
        progress = run_production_grid(
            root,
            _path(root, args.out),
            _path(root, args.bundle),
            _path(root, args.dataset),
            _path(root, args.checkpoint_manifest),
            device_name=args.device,
            num_workers=args.num_workers,
        )
        print(f"status={progress['status']}")
        print(f"counts={progress['counts']}")
        print("test_barrier=locked")
    elif args.command == "production-status":
        progress = production_status(root, _path(root, args.out))
        print(f"status={progress['status']}")
        print(f"counts={progress['counts']}")
        print(f"test_barrier={progress['test_barrier']}")
    elif args.command == "freeze-checkpoints":
        checkpoints, manifest = freeze_f16_checkpoints(root, _path(root, args.out))
        print(f"status={manifest['status']}")
        print(f"checkpoints={len(checkpoints)}")
        print("test_barrier=locked")
    elif args.command == "extract-validation":
        state = extract_f16_validation_statistics(
            root,
            _path(root, args.out),
            _path(root, args.bundle),
            _path(root, args.dataset),
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            chunk_rows=args.chunk_rows,
        )
        print(f"status={state['status']}")
        print(f"feature_sets={len(state['feature_sets'])}")
        print("test_barrier=locked")
    elif args.command == "analyze-validation":
        results, geometry, manifest = analyze_f16_validation(
            root,
            _path(root, args.out),
            _path(root, args.bundle),
        )
        print(f"status={manifest['status']}")
        print(f"results={len(results)}")
        print(f"geometry={len(geometry)}")
        print("test_barrier=locked")
    elif args.command == "unlock-test":
        unlock = unlock_f16_test(
            root, _path(root, args.out), _path(root, args.bundle)
        )
        print(f"status={unlock['status']}")
        print(f"unlock_fingerprint={unlock['unlock_fingerprint']}")
        print("selection_changes_permitted=false")
    elif args.command == "run-test":
        complete = run_f16_fixed_test(
            root,
            _path(root, args.out),
            _path(root, args.bundle),
            _path(root, args.dataset),
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            chunk_rows=args.chunk_rows,
        )
        print(f"status={complete['status']}")
        print(f"runtime_seconds={complete['runtime_seconds_this_invocation']:.3f}")
        print("selection_changes_after_unlock=false")
    elif args.command == "summarize":
        summary = summarize_f16_boundary_corrected(root, _path(root, args.out))
        print(f"status={summary['status']}")
        print(f"pattern={summary['interpretation']['overall']}")
        print("phase1_outcome_unchanged=true")
    elif args.command == "report":
        _report, manifest = report_f16(root, _path(root, args.out))
        print(f"status={manifest['status']}")
        print(f"artifacts={len(manifest['artifacts'])}")
        print("phase1_outcome_unchanged=true")


if __name__ == "__main__":
    main()

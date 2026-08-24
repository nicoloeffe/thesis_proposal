#!/usr/bin/env python3
"""Command-line entry point for Experiment 01 Phase III."""

from __future__ import annotations

import argparse
import json

from experiment01.phase3 import (
    run_historical_mlp_gate,
    run_phase3_benchmark,
    run_phase3_evaluation,
    run_phase3_preproduction_gates,
    run_phase3_selection,
    write_phase1_branch_whitening_effects,
    write_phase3_job_inventory,
)
from experiment01.phase3_reporting import finalize_phase3, summarize_phase3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    whitening = subparsers.add_parser(
        "derive-branch-whitening",
        help="derive the frozen Phase-I branch-specific whitening diagnostic",
    )
    whitening.add_argument("--phase1-dir", required=True)
    whitening.add_argument("--out-dir", required=True)
    inventory = subparsers.add_parser(
        "inventory",
        help="serialize and validate all preregistered Phase-III job cells",
    )
    inventory.add_argument("--phase1-dir", required=True)
    inventory.add_argument("--out-dir", required=True)
    gates = subparsers.add_parser(
        "preproduction-gates",
        help="run synthetic, parity, transform, and isolation prerequisites",
    )
    gates.add_argument("--bundle", required=True)
    gates.add_argument("--phase1-dir", required=True)
    gates.add_argument("--phase2-dir", required=True)
    gates.add_argument("--out-dir", required=True)
    gates.add_argument("--device", default="cuda")
    reference = subparsers.add_parser(
        "reference-gate",
        aliases=["historical-gate"],
        help="retrain and reproduce the frozen reference MLP",
    )
    reference.add_argument(
        "--reference-dir",
        "--historical-dir",
        dest="reference_dir",
        required=True,
    )
    reference.add_argument("--out-dir", required=True)
    reference.add_argument("--device", default="cuda")
    selection = subparsers.add_parser(
        "select",
        help="run the full validation-only weight-decay selection grid",
    )
    selection.add_argument("--bundle", required=True)
    selection.add_argument("--phase1-dir", required=True)
    selection.add_argument("--out-dir", required=True)
    selection.add_argument("--device", default="cuda")
    evaluation = subparsers.add_parser(
        "evaluate",
        help="retrain selected readers and perform one-shot fixed-test inference",
    )
    evaluation.add_argument("--bundle", required=True)
    evaluation.add_argument("--phase1-dir", required=True)
    evaluation.add_argument("--out-dir", required=True)
    evaluation.add_argument("--device", default="cuda")
    benchmark = subparsers.add_parser(
        "benchmark",
        help="benchmark two 1000-step validation-only cells before production",
    )
    benchmark.add_argument("--bundle", required=True)
    benchmark.add_argument("--phase1-dir", required=True)
    benchmark.add_argument("--out-dir", required=True)
    benchmark.add_argument("--device", default="cuda")
    summarize = subparsers.add_parser(
        "summarize",
        help="build all Phase-III derived tables after evaluation",
    )
    summarize.add_argument("--bundle", required=True)
    summarize.add_argument("--phase1-dir", required=True)
    summarize.add_argument("--phase2-dir", required=True)
    summarize.add_argument("--out-dir", required=True)
    finalize = subparsers.add_parser(
        "finalize",
        help="generate figures, reports, metadata and the final manifest",
    )
    finalize.add_argument("--bundle", required=True)
    finalize.add_argument("--phase1-dir", required=True)
    finalize.add_argument("--phase2-dir", required=True)
    finalize.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.command == "derive-branch-whitening":
        result = write_phase1_branch_whitening_effects(
            args.phase1_dir, args.out_dir
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "inventory":
        result = write_phase3_job_inventory(args.phase1_dir, args.out_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "preproduction-gates":
        result = run_phase3_preproduction_gates(
            args.bundle,
            args.phase1_dir,
            args.phase2_dir,
            args.out_dir,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command in {"reference-gate", "historical-gate"}:
        result = run_historical_mlp_gate(
            args.reference_dir, args.out_dir, device=args.device
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "select":
        result = run_phase3_selection(
            args.bundle, args.phase1_dir, args.out_dir, device=args.device
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate":
        result = run_phase3_evaluation(
            args.bundle, args.phase1_dir, args.out_dir, device=args.device
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "benchmark":
        result = run_phase3_benchmark(
            args.bundle, args.phase1_dir, args.out_dir, device=args.device
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "summarize":
        result = summarize_phase3(
            args.out_dir,
            args.phase1_dir,
            bundle_dir=args.bundle,
            phase2_dir=args.phase2_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "finalize":
        result = finalize_phase3(
            args.out_dir,
            args.phase1_dir,
            bundle_dir=args.bundle,
            phase2_dir=args.phase2_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

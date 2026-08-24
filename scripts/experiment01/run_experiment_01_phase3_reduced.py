#!/usr/bin/env python3
"""Command-line entry point for the compute-feasible Experiment 01 Phase III-R."""

from __future__ import annotations

import argparse
import json

from experiment01.phase3 import run_phase3_evaluation, run_phase3_selection
from experiment01.phase3_reduced import prepare_phase3_reduced
from experiment01.phase3_reporting import finalize_phase3, summarize_phase3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--phase1-dir", required=True)
    prepare.add_argument("--source-phase3-dir", required=True)
    prepare.add_argument("--out-dir", required=True)
    for name in ("select", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--bundle", required=True)
        command.add_argument("--phase1-dir", required=True)
        command.add_argument("--out-dir", required=True)
        command.add_argument("--device", default="cuda")
    for name in ("summarize", "finalize"):
        command = subparsers.add_parser(name)
        command.add_argument("--bundle", required=True)
        command.add_argument("--phase1-dir", required=True)
        command.add_argument("--phase2-dir", required=True)
        command.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_phase3_reduced(
            args.phase1_dir, args.source_phase3_dir, args.out_dir
        )
    elif args.command == "select":
        result = run_phase3_selection(
            args.bundle, args.phase1_dir, args.out_dir, device=args.device
        )
    elif args.command == "evaluate":
        result = run_phase3_evaluation(
            args.bundle, args.phase1_dir, args.out_dir, device=args.device
        )
    elif args.command == "summarize":
        result = summarize_phase3(
            args.out_dir,
            args.phase1_dir,
            bundle_dir=args.bundle,
            phase2_dir=args.phase2_dir,
        )
    else:
        result = finalize_phase3(
            args.out_dir,
            args.phase1_dir,
            bundle_dir=args.bundle,
            phase2_dir=args.phase2_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

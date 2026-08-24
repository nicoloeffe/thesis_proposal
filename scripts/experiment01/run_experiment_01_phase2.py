#!/usr/bin/env python3
"""CLI for the isolated Experiment 01 Phase-II spectral diagnostic."""

from __future__ import annotations

import argparse
import json

from experiment01.phase2 import Phase2Config, run_phase2
from experiment01.phase2_legacy import reproduce_post_p0_pca_ladder
from experiment01.phase2_reporting import (
    summarize_and_report_phase2,
    write_phase2_manifest,
)
from experiment01.schema import load_input_bundle


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    reproduce = commands.add_parser(
        "reproduce", help="run the mandatory corrected post-P0 PCA ladder gate"
    )
    reproduce.add_argument("--in-dir", required=True)
    reproduce.add_argument("--reference-ladder", required=True)
    reproduce.add_argument("--out", required=True)
    reproduce.add_argument("--tolerance", type=float, default=5e-10)
    run = commands.add_parser(
        "run", help="execute the fail-closed preregistered Phase-II grid"
    )
    run.add_argument("--bundle", required=True)
    run.add_argument("--phase1-dir", required=True)
    run.add_argument("--reproduction-gate", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--chunk-rows", type=int, default=65536)
    run.add_argument(
        "--verify-bundle-hashes",
        action="store_true",
        help=(
            "rehash every bundle shard before execution; otherwise rely on the "
            "frozen manifest and its completed production preflight"
        ),
    )
    finalize = commands.add_parser(
        "finalize", help="validate, summarize, report and hash Phase II"
    )
    finalize.add_argument("--phase2-dir", required=True)
    finalize.add_argument("--bootstrap-draws", type=int, default=5000)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "reproduce":
        payload = reproduce_post_p0_pca_ladder(
            args.in_dir,
            args.reference_ladder,
            args.out,
            tolerance=args.tolerance,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "run":
        bundle = load_input_bundle(
            args.bundle,
            verify_hashes=args.verify_bundle_hashes,
            check_finite=False,
        )
        payload = run_phase2(
            bundle,
            args.out_dir,
            phase1_dir=args.phase1_dir,
            reproduction_gate=args.reproduction_gate,
            config=Phase2Config(
                random_draws=100,
                chunk_rows=args.chunk_rows,
                bundle_hashes_verified_this_run=args.verify_bundle_hashes,
            ),
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "finalize":
        summary = summarize_and_report_phase2(
            args.phase2_dir,
            n_bootstrap=args.bootstrap_draws,
        )
        manifest = write_phase2_manifest(args.phase2_dir)
        print(
            json.dumps(
                {
                    "summary": summary,
                    "manifest_file_sha256": manifest["manifest_file_sha256"],
                    "manifest_payload_sha256": manifest[
                        "manifest_payload_sha256"
                    ],
                },
                indent=2,
            )
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()

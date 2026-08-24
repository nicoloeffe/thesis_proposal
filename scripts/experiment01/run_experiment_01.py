#!/usr/bin/env python3
"""Command-line entry point for Experiment 01."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment01.reproduction import (
    canonical_reproduction_gate,
    reference_input_diagnosis,
)
from experiment01.pipeline import Phase1Config, run_phase1
from experiment01.reporting import generate_phase1_report
from experiment01.schema import load_input_bundle
from experiment01.subsets import (
    anchor_sensitivity,
    generate_all_selections,
    write_subset_manifests,
)
from experiment01.summary import summarize_phase1


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 01 — finite-sample accessibility"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    reference = commands.add_parser(
        "audit-reference",
        aliases=["audit-historical", "audit-legacy"],
        help="verify corrected v2 dumps and explain why they cannot run Phase I v2",
    )
    reference.add_argument("--in-dir", required=True)

    reproduce = commands.add_parser(
        "reproduce",
        help="run the mandatory reference full-rank min-norm OLS gate",
    )
    reproduce.add_argument("--in-dir", required=True)
    reproduce.add_argument("--out", required=True)
    reproduce.add_argument("--tolerance", type=float, default=0.005)

    sidecar = commands.add_parser(
        "build-sidecar",
        help="build CSV-derived metadata and run the full CSV↔NPZ gate",
    )
    sidecar.add_argument("--raw-dir", required=True)
    sidecar.add_argument("--dataset", required=True)
    sidecar.add_argument("--out-dir", required=True)
    sidecar.add_argument("--chunk-rows", type=int, default=200_000)

    split3 = commands.add_parser(
        "build-three-way-split",
        help="retain reference train and chronologically halve held-out days",
    )
    split3.add_argument("--sidecar-dir", required=True)
    split3.add_argument("--dataset", required=True)
    split3.add_argument(
        "--reference-split",
        "--historical-split",
        dest="reference_split",
        required=True,
    )
    split3.add_argument("--out-dir", required=True)

    prepare = commands.add_parser(
        "prepare-bundle",
        help="write complete rows/target shards and pre-extraction storage estimate",
    )
    prepare.add_argument("--split-dir", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument(
        "--reference-dir",
        "--historical-dir",
        "--legacy-dir",
        dest="reference_dir",
        required=True,
        help="frozen reference readout directory",
    )
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--shard-rows", type=int, default=100_000)

    preextract = commands.add_parser(
        "preextract-gate",
        help="benchmark day/stock extraction and reproduce all post-P0 readouts",
    )
    preextract.add_argument("--bundle", required=True)
    preextract.add_argument(
        "--reference-dir",
        "--historical-dir",
        "--legacy-dir",
        dest="reference_dir",
        required=True,
        help="frozen reference readout directory",
    )
    preextract.add_argument("--device", default="cuda")
    preextract.add_argument("--batch-size", type=int, default=512)
    preextract.add_argument("--num-workers", type=int, default=2)
    preextract.add_argument("--rtol", type=float, default=1e-5)
    preextract.add_argument("--atol", type=float, default=1e-6)
    preextract.add_argument(
        "--benchmark-checkpoint", default="supervised_seed0_ep020"
    )

    extract = commands.add_parser(
        "extract-features",
        help="sequentially extract verified sharded readouts",
    )
    extract.add_argument("--bundle", required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--batch-size", type=int, default=512)
    extract.add_argument("--num-workers", type=int, default=2)
    extract.add_argument(
        "--checkpoint-tags",
        default=None,
        help="optional comma-separated canonical checkpoint tags",
    )

    preflight = commands.add_parser(
        "preflight", help="validate a complete three-way input bundle"
    )
    preflight.add_argument("--bundle", required=True)
    preflight.add_argument(
        "--skip-finite-scan",
        action="store_true",
        help="skip only the expensive NaN/Inf scan; identities and hashes remain mandatory",
    )

    subsets = commands.add_parser(
        "subsets", help="serialize primary and time-of-day sensitivity subsets"
    )
    subsets.add_argument("--bundle", required=True)
    subsets.add_argument("--out-dir", required=True)

    run = commands.add_parser("run-phase1", help="execute the Phase-I grid")
    run.add_argument("--bundle", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--branches", default="supervised,jepa_horizon,jepa_masked")
    run.add_argument("--readouts", default="last_concat512,meanK_concatS")
    run.add_argument("--target-blocks", default="directional,volatility,timing")
    run.add_argument("--chunk-rows", type=int, default=65536)
    run.add_argument("--no-common-alpha", action="store_true")
    run.add_argument("--no-tuned-alpha", action="store_true")
    run.add_argument("--no-min-norm", action="store_true")
    run.add_argument("--no-whitening", action="store_true")

    summarize = commands.add_parser(
        "summarize", help="build uncertainty tables and outcome classification"
    )
    summarize.add_argument("--results", required=True)
    summarize.add_argument("--out-dir", required=True)
    summarize.add_argument("--bootstrap-draws", type=int, default=5000)
    summarize.add_argument(
        "--bootstrap-workers",
        type=int,
        default=1,
        help="parallel worker processes for independent bootstrap groups",
    )

    report = commands.add_parser(
        "report", help="generate Phase-I figures and REPORT_EXPERIMENT_01.md"
    )
    report.add_argument("--results", required=True)
    report.add_argument("--summary-dir", required=True)
    report.add_argument("--out-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command in {"audit-reference", "audit-historical", "audit-legacy"}:
        print(json.dumps(reference_input_diagnosis(args.in_dir), indent=2))
        return
    if args.command == "reproduce":
        payload = canonical_reproduction_gate(
            args.in_dir,
            output_path=args.out,
            tolerance=args.tolerance,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "build-sidecar":
        from experiment01.metadata import build_metadata_sidecar

        payload = build_metadata_sidecar(
            args.raw_dir,
            args.dataset,
            args.out_dir,
            chunk_rows=args.chunk_rows,
        )
        print(
            json.dumps(
                {
                    "passed": payload["equivalence"]["passed"],
                    "n_rows": payload["manifest"]["n_rows"],
                    "sidecar_fingerprint": payload["manifest"][
                        "sidecar_fingerprint"
                    ],
                },
                indent=2,
            )
        )
        return
    if args.command == "build-three-way-split":
        from experiment01.split3 import build_three_way_split

        payload = build_three_way_split(
            args.sidecar_dir,
            args.dataset,
            args.reference_split,
            args.out_dir,
        )
        print(
            json.dumps(
                {
                    "split_protocol_fingerprint": payload[
                        "split_protocol_fingerprint"
                    ],
                    "splits": {
                        split: {
                            "n_rows": record["n_rows"],
                            "n_stock_days": record["n_stock_days"],
                        }
                        for split, record in payload["splits"].items()
                    },
                },
                indent=2,
            )
        )
        return
    if args.command == "prepare-bundle":
        from experiment01.bundle import prepare_bundle

        payload = prepare_bundle(
            args.split_dir,
            args.dataset,
            args.reference_dir,
            args.out_dir,
            shard_rows=args.shard_rows,
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "target_manifest_fingerprint": payload["provenance"][
                        "target_manifest_fingerprint"
                    ],
                    "storage_estimate": payload["pre_extraction"][
                        "storage_estimate"
                    ],
                },
                indent=2,
            )
        )
        return
    if args.command == "preextract-gate":
        from experiment01.extraction import run_pre_extraction_gate

        payload = run_pre_extraction_gate(
            args.bundle,
            args.reference_dir,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rtol=args.rtol,
            atol=args.atol,
            benchmark_checkpoint=args.benchmark_checkpoint,
        )
        print(
            json.dumps(
                {
                    "passed": payload["passed"],
                    "feature_equivalence_passed": payload[
                        "feature_equivalence"
                    ]["passed"],
                    "benchmarks": payload["benchmarks"],
                },
                indent=2,
            )
        )
        return
    if args.command == "extract-features":
        from experiment01.extraction import extract_full_features

        tags = (
            None
            if args.checkpoint_tags is None
            else _csv_tuple(args.checkpoint_tags)
        )
        payload = extract_full_features(
            args.bundle,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            checkpoint_tags=tags,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "preflight":
        bundle = load_input_bundle(
            args.bundle,
            check_finite=not args.skip_finite_scan,
        )
        payload = {
            "status": "passed",
            "bundle": str(bundle.root),
            "n_rows": {key: len(value) for key, value in bundle.rows.items()},
            "encoder_seeds": bundle.encoder_seeds,
            "target_names": bundle.target_names,
            "n_feature_sets": len(bundle.feature_sets),
        }
        print(json.dumps(payload, indent=2))
        return
    if args.command == "subsets":
        bundle = load_input_bundle(args.bundle)
        destination = Path(args.out_dir)
        primary = generate_all_selections(bundle.rows["train"])
        write_subset_manifests(
            bundle.rows["train"],
            primary,
            destination / "primary",
            source_row_key_sha256=bundle.manifest["splits"]["train"][
                "row_key_sha256"
            ],
        )
        sensitivity = anchor_sensitivity(bundle.rows["train"])
        for label, predicate in (
            ("opening", lambda value: value < 0.2),
            ("middle", lambda value: 0.2 <= value <= 0.8),
            ("closing", lambda value: value > 0.8),
        ):
            write_subset_manifests(
                bundle.rows["train"],
                [
                    value
                    for value in sensitivity
                    if predicate(float(value.anchor_quantile))
                ],
                destination / "time_of_day_sensitivity" / label,
                source_row_key_sha256=bundle.manifest["splits"]["train"][
                    "row_key_sha256"
                ],
            )
        print(f"wrote {len(primary)} primary and {len(sensitivity)} sensitivity subsets")
        return
    if args.command == "run-phase1":
        bundle = load_input_bundle(args.bundle)
        config = Phase1Config(
            branches=_csv_tuple(args.branches),
            readouts=_csv_tuple(args.readouts),
            target_blocks=_csv_tuple(args.target_blocks),
            run_common_alpha=not args.no_common_alpha,
            run_tuned_alpha=not args.no_tuned_alpha,
            run_min_norm=not args.no_min_norm,
            run_whitening=not args.no_whitening,
            chunk_rows=args.chunk_rows,
        )
        payload = run_phase1(bundle, args.out_dir, config)
        print(json.dumps(payload["artifacts"], indent=2))
        return
    if args.command == "summarize":
        payload = summarize_phase1(
            args.results,
            args.out_dir,
            n_bootstrap=args.bootstrap_draws,
            n_workers=args.bootstrap_workers,
        )
        print(
            json.dumps(
                payload["directional_last_concat512_outcome"], indent=2
            )
        )
        return
    if args.command == "report":
        payload = generate_phase1_report(
            args.results, args.summary_dir, args.out_dir
        )
        print(json.dumps(payload, indent=2))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()

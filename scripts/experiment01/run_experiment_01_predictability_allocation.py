#!/usr/bin/env python3
"""CLI for the outcome-blind preregistration and execution of the §14.5 test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiment01.io import atomic_write_json, canonical_json_sha256
from experiment01.predictability_allocation import (
    audit_historical_inputs,
    default_protocol_payload,
    freeze_protocol_payload,
    run_predictability_allocation,
)


DEFAULT_INPUT_DIR = "validation/readouts_v2_20260728"
DEFAULT_DATASET = "data/lobench_processed.npz"


def _output_path(value: str, *, force: bool) -> Path:
    path = Path(value)
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force explicitly")
    return path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    draft = commands.add_parser(
        "draft",
        help="write the proposed protocol without reading any scientific outcome",
    )
    draft.add_argument("--out", required=True)
    draft.add_argument("--force", action="store_true")

    audit = commands.add_parser(
        "audit",
        help="verify hashes and the deliberately fractional sample contract",
    )
    audit.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    audit.add_argument("--dataset", default=DEFAULT_DATASET)
    audit.add_argument("--out")
    audit.add_argument("--force", action="store_true")

    freeze = commands.add_parser(
        "freeze",
        help="freeze an approved draft and bind it to the audited input inventory",
    )
    freeze.add_argument("--draft", required=True)
    freeze.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    freeze.add_argument("--dataset", default=DEFAULT_DATASET)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--scientific-approver", required=True)
    freeze.add_argument("--approve-proposed-thresholds", action="store_true")
    freeze.add_argument("--acknowledge-exploratory-status", action="store_true")
    freeze.add_argument("--force", action="store_true")

    run = commands.add_parser(
        "run",
        help="execute only a frozen, input-bound protocol",
    )
    run.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    run.add_argument("--dataset", default=DEFAULT_DATASET)
    run.add_argument("--protocol", required=True)
    run.add_argument("--out-dir", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "draft":
        path = _output_path(args.out, force=args.force)
        payload = default_protocol_payload()
        atomic_write_json(path, payload)
        print(
            json.dumps(
                {
                    "status": "draft",
                    "path": str(path.resolve()),
                    "payload_sha256": canonical_json_sha256(payload),
                    "outcomes_read": False,
                },
                indent=2,
            )
        )
        return
    if args.command == "audit":
        payload = audit_historical_inputs(
            args.input_dir, args.dataset, verify_hashes=True
        )
        if args.out:
            path = _output_path(args.out, force=args.force)
            atomic_write_json(path, payload)
        print(json.dumps(payload, indent=2))
        return
    if args.command == "freeze":
        if not args.approve_proposed_thresholds:
            raise ValueError("threshold approval is required before freezing")
        if not args.acknowledge_exploratory_status:
            raise ValueError("exploratory-status acknowledgement is required")
        path = _output_path(args.out, force=args.force)
        draft = json.loads(Path(args.draft).read_text(encoding="utf-8"))
        audit = audit_historical_inputs(
            args.input_dir, args.dataset, verify_hashes=True
        )
        payload = freeze_protocol_payload(
            draft,
            audit,
            scientific_approver=args.scientific_approver,
            approved_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(path, payload)
        print(
            json.dumps(
                {
                    "status": "frozen",
                    "path": str(path.resolve()),
                    "payload_sha256": canonical_json_sha256(payload),
                    "input_inventory_sha256": audit["inventory_sha256"],
                    "outcomes_read": False,
                },
                indent=2,
            )
        )
        return
    if args.command == "run":
        metadata = run_predictability_allocation(
            args.input_dir,
            args.dataset,
            args.protocol,
            args.out_dir,
            verify_hashes=True,
        )
        print(json.dumps(metadata, indent=2))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()

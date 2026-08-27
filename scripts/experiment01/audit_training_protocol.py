#!/usr/bin/env python3
"""Audit canonical Experiment 01 training checkpoints and emit JSON evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment01.io import sha256_file
from experiment01.training_audit import (
    audit_training_protocol,
    render_training_protocol,
    write_audit_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/experiment01/CHECKPOINTS_MULTISEED_MANIFEST.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="also reconstruct historical train/validation endpoint hashes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiment01/TRAINING_PROTOCOL_AUDIT.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("docs/experiment01/TRAINING_PROTOCOL.md"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    dataset = args.dataset
    if dataset is not None and not dataset.is_absolute():
        dataset = root / dataset
    output = args.out if args.out.is_absolute() else root / args.out
    markdown_output = (
        args.markdown_out if args.markdown_out.is_absolute() else root / args.markdown_out
    )
    audit = audit_training_protocol(root, manifest, dataset)
    write_audit_json(output, audit)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_training_protocol(audit, sha256_file(output)), encoding="utf-8"
    )
    print(f"status={audit['status']}")
    print(f"checkpoints={len(audit['checkpoints'])}")
    print(f"out={output.relative_to(root)}")
    print(f"markdown={markdown_output.relative_to(root)}")


if __name__ == "__main__":
    main()

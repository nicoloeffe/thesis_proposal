#!/usr/bin/env python3
"""CLI for Experiment 01 T2 token-role matched-null diagnostics."""

from __future__ import annotations

import argparse
import json

from experiment01.token_role import (
    benchmark_token_role,
    finalize_token_role,
    freeze_protocol,
    reproduce_historical_token_role,
    run_token_role_nulls,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("freeze", "validate inputs and freeze the preregistered T2 protocol"),
        ("reproduce", "run the fail-closed historical OLS reproduction gate"),
        ("benchmark", "benchmark five matched structured-null draws"),
        ("run", "run the resumable 100-draw structured/generic grid"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--in-dir", required=True)
        command.add_argument("--out-dir", required=True)
    finalize = commands.add_parser(
        "finalize", help="summarize, plot, report and hash the completed T2 run"
    )
    finalize.add_argument("--out-dir", required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "freeze":
        result = freeze_protocol(args.in_dir, args.out_dir)
    elif args.command == "reproduce":
        result = reproduce_historical_token_role(args.in_dir, args.out_dir)
    elif args.command == "benchmark":
        result = benchmark_token_role(args.in_dir, args.out_dir)
    elif args.command == "run":
        result = run_token_role_nulls(args.in_dir, args.out_dir)
    elif args.command == "finalize":
        result = finalize_token_role(args.out_dir)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

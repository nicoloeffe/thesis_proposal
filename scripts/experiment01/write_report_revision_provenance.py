#!/usr/bin/env python3
"""Write a provenance record for report-only Experiment 01 revisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment01.io import atomic_write_json, sha256_file


FROZEN_SCIENTIFIC_HASHES = {
    "validation/experiment01/execution_20260730/phase1/results.parquet": (
        "ecf4e410c595baa32d06a1998bbd5151794d02ff141499af3c1f56268e110ffb"
    ),
    "validation/experiment01/execution_20260730/summary/summary.json": (
        "7978961be69e50881ac022a67bfd7fea4f619c9806374121b57d6d4cbac1d4a6"
    ),
    "validation/experiment01/execution_20260730/phase2/phase2_results.parquet": (
        "a0a3a5ea609f8347af0ce29c2ef4fd000cd847ecf24b9df544d7b1f600449fb3"
    ),
    "validation/experiment01/execution_20260730/phase2/predictive_mass.parquet": (
        "fecf798a2cf042d5a872a5d173c9710849e038d016a00d93eea6beff86dd6727"
    ),
    "validation/experiment01/execution_20260730/phase2/summary.json": (
        "bfc2c9f000d85d1555f3004bad73aa08728d20f54e9f28c86ba31f8a159a432e"
    ),
    "validation/experiment01/execution_20260730/phase3_reduced/phase3_results.parquet": (
        "f31fe3926fe8ff5512a9f77069c233209b75c3b9c73f348e3464176c2a9e190e"
    ),
    "validation/experiment01/execution_20260730/phase3_reduced/summary.json": (
        "e5e707a40c040cd5062439670a1aa33a0b7ea21dd5b23c59557f2e9299d6c4d6"
    ),
}

HISTORICAL_PHASE2_MANIFEST_SHA256 = (
    "1a30b67f6739a1a0440eae1866ee55f72cddf94248e5edf336a7e605461144c2"
)
HISTORICAL_PHASE3_MANIFEST_SHA256 = (
    "31a260a084571e19114d810f7ae2efd4f35b6e1c75ed5c6a06bc7a288366f104"
)
TOKEN_ROLE_MANIFEST_SHA256 = (
    "ef23d6517d20252c1cfd58a0e89e86f8093b91ca7867a92274d240df9b0fdc83"
)

REPORT_PAIRS = {
    "docs/results/phase1/REPORT_EXPERIMENT_01.md": (
        "validation/experiment01/execution_20260730/report/REPORT_EXPERIMENT_01.md"
    ),
    "docs/results/phase1/SUMMARY_NARRATIVE_EXPERIMENT_01.md": (
        "validation/experiment01/execution_20260730/report/"
        "SUMMARY_NARRATIVE_EXPERIMENT_01.md"
    ),
    "docs/results/phase1/CHANGELOG_NARRATIVE_20260731.md": (
        "validation/experiment01/execution_20260730/report/"
        "CHANGELOG_NARRATIVE_20260731.md"
    ),
    "docs/results/phase1/16_critical_budget_metrics.parquet": (
        "validation/experiment01/execution_20260730/report/"
        "16_critical_budget_metrics.parquet"
    ),
    "docs/results/phase1/17_claim_table.parquet": (
        "validation/experiment01/execution_20260730/report/17_claim_table.parquet"
    ),
    "docs/results/phase1/report_manifest.json": (
        "validation/experiment01/execution_20260730/report/report_manifest.json"
    ),
    "docs/results/phase2/REPORT_EXPERIMENT_01_PHASE2.md": (
        "validation/experiment01/execution_20260730/phase2/"
        "REPORT_EXPERIMENT_01_PHASE2.md"
    ),
    "docs/results/phase3r/AUDIT_EXPERIMENT_01_PHASE3.md": (
        "validation/experiment01/execution_20260730/phase3_reduced/"
        "AUDIT_EXPERIMENT_01_PHASE3.md"
    ),
    "docs/results/phase3r/REPORT_EXPERIMENT_01_PHASE3.md": (
        "validation/experiment01/execution_20260730/phase3_reduced/"
        "REPORT_EXPERIMENT_01_PHASE3.md"
    ),
    "docs/results/phase3r/SUMMARY_NARRATIVE_EXPERIMENT_01_PHASE3.md": (
        "validation/experiment01/execution_20260730/phase3_reduced/"
        "SUMMARY_NARRATIVE_EXPERIMENT_01_PHASE3.md"
    ),
    "docs/results/phase3r/CHANGELOG_PHASE3.md": (
        "validation/experiment01/execution_20260730/phase3_reduced/"
        "CHANGELOG_PHASE3.md"
    ),
    "docs/results/phase3r/phase3_report_metrics.parquet": (
        "validation/experiment01/execution_20260730/phase3_reduced/"
        "phase3_report_metrics.parquet"
    ),
}

REVISED_REPOSITORY_FILES = (
    "README.md",
    "docs/research/RESEARCH_NOTE_GEOMETRY_ACCESSIBILITY.md",
    "docs/results/README.md",
    "docs/results/CHANGELOG_REPORT_REVISION_20260825.md",
    "docs/results/token_role/REPORT_EXPERIMENT_01_TOKEN_ROLE.md",
    "docs/experiment01/SPEC_EXPERIMENT_01_TOKEN_ROLE_MATCHED_NULL_20260826.md",
    "PROJECT_STATE.md",
    "experiment01/io.py",
    "experiment01/reporting.py",
    "experiment01/phase2_reporting.py",
    "experiment01/phase3_reporting.py",
    "scripts/experiment01/write_report_revision_provenance.py",
)


def _verified_record(root: Path, relative: str, expected: str) -> dict[str, object]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"frozen scientific artifact changed: {relative}: "
            f"expected {expected}, observed {observed}"
        )
    return {
        "path": relative,
        "sha256": observed,
        "size_bytes": path.stat().st_size,
        "verified_against_frozen_hash": True,
    }


def build_provenance(repository_root: str | Path) -> dict[str, object]:
    root = Path(repository_root).resolve()
    scientific = [
        _verified_record(root, relative, expected)
        for relative, expected in FROZEN_SCIENTIFIC_HASHES.items()
    ]

    identity_gate_path = (
        root
        / "validation/experiment01/execution_20260730/phase3_reduced/"
        "artifact_identity_gate.json"
    )
    identity_gate = json.loads(identity_gate_path.read_text(encoding="utf-8"))
    gate_phase2_hash = identity_gate["phase2"]["manifest_sha256"]
    if gate_phase2_hash != HISTORICAL_PHASE2_MANIFEST_SHA256:
        raise RuntimeError("Phase-III gate does not attest the frozen Phase-II manifest")

    phase3_manifest_path = (
        root
        / "validation/experiment01/execution_20260730/phase3_reduced/"
        "phase3_manifest.json"
    )
    phase3_manifest_hash = sha256_file(phase3_manifest_path)
    if phase3_manifest_hash != HISTORICAL_PHASE3_MANIFEST_SHA256:
        raise RuntimeError("frozen Phase-III manifest identity changed")

    current_phase2_manifest = (
        root / "validation/experiment01/execution_20260730/phase2/manifest.json"
    )
    token_role_manifest = root / "validation/experiment01/token_role_20260826/manifest.json"
    if sha256_file(token_role_manifest) != TOKEN_ROLE_MANIFEST_SHA256:
        raise RuntimeError("completed T2 token-role manifest identity changed")
    report_records = []
    for published_relative, canonical_relative in REPORT_PAIRS.items():
        published = root / published_relative
        canonical = root / canonical_relative
        if not published.is_file() or not canonical.is_file():
            raise FileNotFoundError(
                published if not published.is_file() else canonical
            )
        published_hash = sha256_file(published)
        canonical_hash = sha256_file(canonical)
        if published_hash != canonical_hash:
            raise RuntimeError(
                f"published/canonical report revision differs: {published_relative}"
            )
        report_records.append(
            {
                "published_path": published_relative,
                "canonical_path": canonical_relative,
                "sha256": published_hash,
                "size_bytes": published.stat().st_size,
                "byte_identical": True,
            }
        )

    repository_records = []
    for relative in REVISED_REPOSITORY_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        repository_records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "schema_name": "thesis.experiment01.report_revision_provenance",
        "schema_version": 1,
        "revision_date": "2026-08-25",
        "scope": "narrative_and_read_only_report_diagnostics_only",
        "scientific_results_modified": False,
        "thresholds_modified": False,
        "technical_outcomes_modified": False,
        "scientific_compute_pipeline_modified": False,
        "reporting_code_modified": True,
        "frozen_scientific_artifacts": scientific,
        "historical_execution_manifests": {
            "phase2": {
                "sha256": HISTORICAL_PHASE2_MANIFEST_SHA256,
                "attested_by": (
                    "validation/experiment01/execution_20260730/phase3_reduced/"
                    "artifact_identity_gate.json"
                ),
                "artifact_hashes_verified_before_phase3": int(
                    identity_gate["phase2"]["artifact_hashes_verified"]
                ),
            },
            "phase3_reduced": {
                "path": (
                    "validation/experiment01/execution_20260730/phase3_reduced/"
                    "phase3_manifest.json"
                ),
                "sha256": phase3_manifest_hash,
            },
        },
        "current_phase2_directory_manifest": {
            "path": (
                "validation/experiment01/execution_20260730/phase2/manifest.json"
            ),
            "sha256": sha256_file(current_phase2_manifest),
            "role": "post_hoc_report_revision_state",
            "is_original_phase3_gate_identity": False,
            "original_phase3_gate_identity_preserved_above": True,
        },
        "corrective_diagnostics": {
            "token_role_t2": {
                "path": "validation/experiment01/token_role_20260826/manifest.json",
                "sha256": TOKEN_ROLE_MANIFEST_SHA256,
                "status": "complete",
                "phase1_outcome_modified": False,
                "phase3_outcome_modified": False,
            }
        },
        "report_revision_artifacts": report_records,
        "revised_repository_files": repository_records,
        "interpretation": {
            "phase1": "frozen A1; revised narrative and read-only diagnostics",
            "phase2": "frozen spectral results; revised report only",
            "phase3_r": (
                "frozen R3 technical classification; revised reader-specific "
                "interpretation and raw-metric disclosure"
            ),
            "token_role": (
                "historical fixed-projection observation retained; completed "
                "structured role-Haar null does not support a privileged "
                "role-contrast mechanism"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--out",
        default="docs/results/REPORT_REVISION_PROVENANCE_20260825.json",
    )
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    output = root / args.out
    atomic_write_json(output, build_provenance(root))
    print(json.dumps({"path": str(output.relative_to(root)), "sha256": sha256_file(output)}))


if __name__ == "__main__":
    main()

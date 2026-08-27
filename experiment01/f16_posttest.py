"""Audited post-test serialization adapter for the frozen F16 reporter.

The fixed test and every scientific selection were already complete before
this adapter existed.  It changes only strict-JSON serialization of non-finite
informational fields (notably the inapplicable omitted-stock identifier on a
bootstrap-summary row) from NaN to null.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from experiment01 import f16_reporting as frozen_reporting
from experiment01.io import (
    atomic_write_json,
    canonical_json_sha256,
    json_safe,
    sha256_file,
)


def strict_json_safe_fingerprint(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(json_safe(payload))


def summarize_f16_serialization_fix(repo_root: Path, output_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    original_hash_function = frozen_reporting.canonical_json_sha256
    frozen_reporting.canonical_json_sha256 = strict_json_safe_fingerprint
    try:
        summary = frozen_reporting.summarize_f16(repo_root, output_root)
    finally:
        frozen_reporting.canonical_json_sha256 = original_hash_function
    adapter_path = Path(__file__).resolve()
    amendment = {
        "schema_name": "thesis.experiment01.f16_posttest_serialization_amendment",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trigger": "strict JSON rejected NaN in an inapplicable informational field",
        "scope": "serialization only: recursively map non-finite JSON values to null before fingerprinting",
        "scientific_values_recomputed": False,
        "test_reopened": False,
        "checkpoint_selection_changed": False,
        "alpha_selection_changed": False,
        "outcome_rule_changed": False,
        "frozen_reporting_source": {
            "path": "experiment01/f16_reporting.py",
            "sha256": sha256_file(repo_root / "experiment01/f16_reporting.py"),
        },
        "adapter_source": {
            "path": "experiment01/f16_posttest.py",
            "sha256": sha256_file(adapter_path),
        },
        "fixed_test_complete_sha256": sha256_file(
            output_root / "f16_fixed_test_complete.json"
        ),
        "results_sha256": sha256_file(output_root / "f16_results.parquet"),
        "geometry_sha256": sha256_file(output_root / "f16_geometry.parquet"),
        "grouped_uncertainty_sha256": sha256_file(
            output_root / "f16_grouped_uncertainty.parquet"
        ),
        "summary_sha256": sha256_file(output_root / "f16_summary.json"),
    }
    amendment["amendment_fingerprint"] = canonical_json_sha256(amendment)
    atomic_write_json(output_root / "f16_posttest_serialization_amendment.json", amendment)
    return summary

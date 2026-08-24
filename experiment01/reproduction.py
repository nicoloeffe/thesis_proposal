"""Read-only checks against the corrected reference dump battery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from experiment01.reference.analysis_artifacts import atomic_write_json
from experiment01.reference.consolidation_geometry import (
    derive_pooling,
    linear_stats,
    r2_from_basis,
)
from experiment01.reference.ladder_accessibility import (
    dir_indices,
    validate_stage1_inputs,
)

from .errors import ExperimentIntegrityError


EXPECTED_REPRODUCTION = {
    "jepa_horizon": 0.2111,
    "supervised": 0.3756,
}


def reference_input_diagnosis(in_dir: str | Path) -> Mapping[str, object]:
    root = Path(in_dir).resolve()
    inventory = validate_stage1_inputs(
        root, str(root / "targets_heldout.npz")
    )
    split = inventory["split"]
    blockers = [
        "legacy schema exposes train/val only; canonical test split is absent",
        "train_t is a 100000-row random endpoint subsample, not complete stock-days",
        "feature dumps contain only the sampled train/val endpoints",
        (
            "processed NPZ/readout dumps omit endpoint timestamps and stock "
            "symbols; reconstruct them from the verified raw CSVs before "
            "building the production bundle"
        ),
    ]
    return {
        "post_p0_integrity_verified": True,
        "n_readout_dumps": len(inventory["readout_paths"]),
        "split_fingerprint": split.split_fingerprint,
        "n_train": len(split.train_t),
        "n_validation": len(split.val_t),
        "n_test": 0,
        "phase1_v2_eligible": False,
        "blockers": blockers,
    }


def canonical_reproduction_gate(
    in_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    tolerance: float = 0.005,
) -> Mapping[str, object]:
    """Recompute the historical full-rank OLS control from verified dumps."""
    root = Path(in_dir).resolve()
    inventory = validate_stage1_inputs(
        root, str(root / "targets_heldout.npz")
    )
    target_indices = dir_indices()["dir_indep"]
    with np.load(inventory["targets_path"], allow_pickle=False) as targets:
        y_train = targets["y_train_raw"].astype(np.float64)[:, target_indices]
        y_validation = targets["y_val_raw"].astype(np.float64)[:, target_indices]
    by_branch: dict[str, list[dict[str, float]]] = {
        branch: [] for branch in EXPECTED_REPRODUCTION
    }
    for path in inventory["readout_paths"]:
        with np.load(path, allow_pickle=False) as dump:
            branch = str(np.asarray(dump["arm"]).item())
            if branch not in by_branch:
                continue
            seed = int(np.asarray(dump["seed"]).item())
            x_train = derive_pooling(dump, "last_concat512", "train")
            x_validation = derive_pooling(dump, "last_concat512", "val")
            stats = linear_stats(x_train, y_train, x_validation, y_validation)
            scores = r2_from_basis(stats, np.eye(stats.dimension))
            by_branch[branch].append(
                {
                    "encoder_seed": seed,
                    "aggregate_directional_r2": float(np.mean(scores)),
                }
            )
    results: dict[str, object] = {}
    passed = True
    for branch, expected in EXPECTED_REPRODUCTION.items():
        seed_rows = sorted(
            by_branch[branch], key=lambda row: row["encoder_seed"]
        )
        if len(seed_rows) != 3:
            raise ExperimentIntegrityError(
                f"reproduction expected three {branch} encoder seeds"
            )
        aggregate = float(
            np.mean([row["aggregate_directional_r2"] for row in seed_rows])
        )
        difference = abs(aggregate - expected)
        branch_passed = difference <= tolerance
        passed = passed and branch_passed
        results[branch] = {
            "expected": expected,
            "observed": aggregate,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": branch_passed,
            "per_encoder_seed": seed_rows,
        }
    payload: dict[str, object] = {
        "gate": (
            "full_train/last_concat512/full_rank_raw/"
            "min_norm_ols_raw/directional"
        ),
        "legacy_evaluation_split_name": "val",
        "aggregation": (
            "mean target R2 over 12 independent directional targets within "
            "encoder seed, then mean over three encoder seeds"
        ),
        "post_p0_inputs_verified": True,
        "passed": passed,
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, payload)
    if not passed:
        diagnostic = {
            **payload,
            "required_action": (
                "stop before Phase I; inspect row alignment, target block, "
                "centering, intercept, aggregation and split identity"
            ),
        }
        if output_path is not None:
            path = Path(output_path)
            diagnostic_path = path.with_name(
                "DIAGNOSTIC_REPRODUCTION_GATE_EXPERIMENT_01.json"
            )
            atomic_write_json(diagnostic_path, diagnostic)
        raise ExperimentIntegrityError(
            "canonical OLS reproduction gate failed: "
            + json.dumps(results, sort_keys=True)
        )
    return payload

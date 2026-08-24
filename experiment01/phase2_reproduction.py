"""Fail-closed reproduction gate for the corrected reference PCA ladder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from experiment01.reference.consolidation_geometry import (
    derive_pooling,
    ladder_from_stats,
    linear_stats,
    pca_from_stats,
)
from experiment01.reference.ladder_accessibility import validate_stage1_inputs

from .errors import ExperimentIntegrityError
from .io import atomic_write_json, sha256_file


REFERENCE_PHASE2_POOLINGS = ("last_concat512", "meanK_concatS")
REFERENCE_PHASE2_SCHEDULE = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def reproduce_post_p0_pca_ladder(
    in_dir: str | Path,
    reference_ladder: str | Path,
    output_path: str | Path,
    *,
    tolerance: float = 5e-10,
    poolings: Sequence[str] = REFERENCE_PHASE2_POOLINGS,
) -> Mapping[str, object]:
    """Recompute every trained-target PCA-ladder cell used by Phase II.

    The gate intentionally uses the historical 100k/50k train/validation dump
    and the same post-P0 sufficient-statistic implementation that produced the
    frozen reference.  Production Phase II must not start unless this gate
    passes.
    """
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    root = Path(in_dir).resolve()
    reference_path = Path(reference_ladder).resolve()
    heldout_path = root / "targets_heldout.npz"
    inventory = validate_stage1_inputs(root, str(heldout_path))
    with np.load(inventory["targets_path"], allow_pickle=False) as targets:
        y_train = targets["y_train_raw"].astype(np.float64)
        y_validation = targets["y_val_raw"].astype(np.float64)
        target_names = [str(value) for value in targets["target_names"].tolist()]

    reference = pd.read_csv(reference_path)
    required_columns = {
        "arm",
        "seed",
        "pooling",
        "target",
        "m",
        "r2",
    }
    if not required_columns.issubset(reference.columns):
        raise ExperimentIntegrityError(
            "post-P0 PCA reference is missing required columns"
        )
    expected = reference[
        reference["pooling"].isin(poolings)
        & reference["target"].isin(target_names)
        & reference["m"].isin(REFERENCE_PHASE2_SCHEDULE)
    ][["arm", "seed", "pooling", "target", "m", "r2"]].copy()
    key = ["arm", "seed", "pooling", "target", "m"]
    if expected.duplicated(key).any():
        raise ExperimentIntegrityError("post-P0 PCA reference keys are duplicated")

    rows: list[dict[str, object]] = []
    for path in inventory["readout_paths"]:
        with np.load(path, allow_pickle=False) as dump:
            branch = str(np.asarray(dump["arm"]).item())
            encoder_seed = int(np.asarray(dump["seed"]).item())
            for pooling in poolings:
                x_train = derive_pooling(dump, pooling, "train")
                x_validation = derive_pooling(dump, pooling, "val")
                stats = linear_stats(
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                )
                _, eigenvectors = pca_from_stats(stats)
                ladder = ladder_from_stats(
                    stats, eigenvectors, REFERENCE_PHASE2_SCHEDULE
                )
                for m, scores in ladder.items():
                    for target_name, score in zip(target_names, scores):
                        rows.append(
                            {
                                "arm": branch,
                                "seed": encoder_seed,
                                "pooling": pooling,
                                "target": target_name,
                                "m": int(m),
                                "observed_r2": float(score),
                            }
                        )
    observed = pd.DataFrame(rows)
    if observed.duplicated(key).any():
        raise ExperimentIntegrityError("recomputed PCA ladder keys are duplicated")
    compared = expected.merge(
        observed,
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    missing = compared[compared["_merge"].ne("both")]
    if len(missing):
        payload = {
            "schema_name": "thesis.experiment01.phase2_reproduction_gate",
            "schema_version": 1,
            "passed": False,
            "reason": "reference/recomputed PCA key sets differ",
            "n_unmatched": len(missing),
            "examples": missing.head(20).to_dict("records"),
        }
        atomic_write_json(output_path, payload)
        raise ExperimentIntegrityError(
            "Phase-II post-P0 PCA reproduction key mismatch"
        )
    compared["absolute_difference"] = np.abs(
        compared["observed_r2"] - compared["r2"]
    )
    maximum = float(compared["absolute_difference"].max())
    passed = bool(maximum <= tolerance)
    by_group = []
    for (branch, pooling), group in compared.groupby(
        ["arm", "pooling"], observed=True
    ):
        by_group.append(
            {
                "branch": str(branch),
                "readout": str(pooling),
                "n_cells": len(group),
                "maximum_absolute_difference": float(
                    group["absolute_difference"].max()
                ),
                "mean_absolute_difference": float(
                    group["absolute_difference"].mean()
                ),
            }
        )
    worst = compared.nlargest(20, "absolute_difference")[
        key + ["r2", "observed_r2", "absolute_difference"]
    ]
    payload = {
        "schema_name": "thesis.experiment01.phase2_reproduction_gate",
        "schema_version": 1,
        "passed": passed,
        "gate": "corrected_post_p0_pca_ladder.v1",
        "tolerance": tolerance,
        "n_cells": len(compared),
        "maximum_absolute_difference": maximum,
        "poolings": list(poolings),
        "schedule": list(REFERENCE_PHASE2_SCHEDULE),
        "reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        },
        "post_p0_inputs": {
            "root": str(root),
            "analysis_manifest_sha256": sha256_file(root / "analysis_manifest.json"),
            "inventory_verified": True,
        },
        "by_branch_readout": by_group,
        "worst_cells": worst.to_dict("records"),
    }
    atomic_write_json(output_path, payload)
    if not passed:
        raise ExperimentIntegrityError(
            "Phase-II post-P0 PCA reproduction gate failed: "
            + json.dumps(
                {
                    "maximum_absolute_difference": maximum,
                    "tolerance": tolerance,
                },
                sort_keys=True,
            )
        )
    return payload

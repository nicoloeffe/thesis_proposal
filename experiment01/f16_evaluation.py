"""Train/validation-only F16 checkpoint freezing, extraction and analysis.

The module never opens test targets or test feature arrays.  It reduces each
fixed feature set to additive sufficient statistics so the full endpoint
matrices are never persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

from experiment01.constants import ALPHA_GRID
from experiment01.f16 import BUDGETS, F16IntegrityError, _relative, sha256_string_sequence
from experiment01.f16_convergence import (
    FeatureMoments,
    _array_from_record,
    _feature_array,
    projection_stats,
    role_projections,
)
from experiment01.io import (
    atomic_savez,
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_sha256,
    sha256_array,
    sha256_file,
)
from experiment01.linear import (
    SufficientStats,
    eigensystem,
    evaluate_stats,
    fit_alpha,
    select_targets,
    transformed_design,
)
from experiment01.reference.extract_readouts_multiseed import RawWindowDataset
from training.train_jepa_horizon import HorizonJEPAEncoderConfig
from training.train_supervised_grid import ReadoutConfig, SupervisedGrid


EVALUATION_SCHEMA_VERSION = 1
READOUTS = ("last_concat512", "meanK_concatS")
CHECKPOINT_KINDS = ("best", "epoch20")
ANCHOR_ARMS = ("jepa_horizon", "jepa_masked", "supervised")
WHITENING_DEPTHS = (0, 8, 16, 32, 64, 128, 256, 508)
SOURCE_FILES = (
    "experiment01/f16_evaluation.py",
    "experiment01/f16_convergence.py",
    "experiment01/linear.py",
    "training/train_supervised_grid.py",
    "training/train_jepa_horizon.py",
    "training/train_tokenizer_t.py",
)


@dataclass
class F16FeatureStats:
    covariance: FeatureMoments | None
    budgets: dict[str, SufficientStats]
    validation: SufficientStats


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise F16IntegrityError(f"missing F16 evaluation artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source_inventory(repo_root: Path) -> dict[str, Any]:
    files = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise F16IntegrityError(f"missing F16 evaluation source: {relative}")
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {
        "files": files,
        "fingerprint": canonical_json_sha256(
            {key: value["sha256"] for key, value in sorted(files.items())}
        ),
    }


def _verify_locked_inputs(repo_root: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    progress = _read_json(output_root / "f16_production_progress.json")
    cohort = _read_json(output_root / "f16_cohort_manifest.json")
    if progress.get("status") != "complete" or progress.get("counts") != {"complete": 12}:
        raise F16IntegrityError("F16 production training is not 12/12 complete")
    if progress.get("test_barrier") != "locked" or progress.get("test_accessed") is not False:
        raise F16IntegrityError("F16 production test barrier drift")
    if cohort.get("status") != "selected_and_frozen" or cohort.get("selected_cap_per_stock_day") != 128:
        raise F16IntegrityError("F16 selected cohort is not frozen at cap 128")
    barrier = cohort.get("test_barrier", {})
    if barrier != {
        "status": "locked",
        "test_features_accessed": False,
        "test_statistics_accessed": False,
        "test_targets_accessed": False,
    }:
        raise F16IntegrityError("F16 cohort test barrier was mutated")
    failure_path = output_root / "f16_failures.parquet"
    failures = pd.read_parquet(failure_path)
    if not failures.empty:
        raise F16IntegrityError("F16 training failures exist")
    return progress, cohort


def freeze_f16_checkpoints(repo_root: Path, output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hash-pin best and epoch-20 checkpoints after all 12 runs terminate."""
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    progress, cohort = _verify_locked_inputs(repo_root, output_root)
    source = _source_inventory(repo_root)
    checkpoint_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    training_source_fingerprints: set[str] = set()
    for budget in BUDGETS:
        for seed in (0, 1, 2):
            run_dir = output_root / "runs" / budget / f"seed{seed}"
            complete_path = run_dir / "complete.json"
            complete = _read_json(complete_path)
            if (
                complete.get("status") != "complete"
                or complete.get("budget") != budget
                or int(complete.get("encoder_seed", -1)) != seed
                or complete.get("test_accessed") is not False
                or complete.get("failures")
            ):
                raise F16IntegrityError(f"invalid F16 completion: {budget}/seed{seed}")
            training_source_fingerprints.add(str(complete["source_fingerprint"]))
            history_path = repo_root / complete["history"]["path"]
            if sha256_file(history_path) != complete["history"]["sha256"]:
                raise F16IntegrityError(f"F16 history drift: {budget}/seed{seed}")
            history = pd.read_parquet(history_path)
            history.insert(0, "encoder_seed", seed)
            history.insert(0, "budget", budget)
            curve_frames.append(history)
            for kind in CHECKPOINT_KINDS:
                record = complete["checkpoints"][kind]
                path = repo_root / record["path"]
                if not path.is_file() or sha256_file(path) != record["sha256"]:
                    raise F16IntegrityError(f"F16 checkpoint drift: {budget}/seed{seed}/{kind}")
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(path, map_location="cpu")
                if (
                    payload.get("budget") != budget
                    or int(payload.get("encoder_seed", -1)) != seed
                    or payload.get("test_accessed") is not False
                    or payload.get("source_inventory", {}).get("fingerprint")
                    != complete["source_fingerprint"]
                ):
                    raise F16IntegrityError(f"F16 checkpoint payload drift: {budget}/seed{seed}/{kind}")
                expected_update = int(complete["best_update"] if kind == "best" else complete["epoch20_update"])
                if int(payload.get("global_update", -1)) != expected_update:
                    raise F16IntegrityError(f"F16 checkpoint update drift: {budget}/seed{seed}/{kind}")
                checkpoint_rows.append(
                    {
                        "encoder_family": "supervised_f16",
                        "trained_budget": budget,
                        "encoder_seed": seed,
                        "checkpoint_kind": kind,
                        "global_update": expected_update,
                        "validation_mse": float(payload["validation_metrics"]["mse"]),
                        "path": _relative(path, repo_root),
                        "sha256": record["sha256"],
                        "size_bytes": path.stat().st_size,
                        "training_source_fingerprint": complete["source_fingerprint"],
                        "test_accessed": False,
                    }
                )
    if len(training_source_fingerprints) != 1:
        raise F16IntegrityError("F16 production cells do not share one training source fingerprint")
    checkpoints = pd.DataFrame(checkpoint_rows).sort_values(
        ["trained_budget", "encoder_seed", "checkpoint_kind"], kind="stable"
    )
    curves = pd.concat(curve_frames, ignore_index=True).sort_values(
        ["budget", "encoder_seed", "global_update", "event"], kind="stable"
    )
    if len(checkpoints) != 24 or checkpoints.duplicated(
        ["trained_budget", "encoder_seed", "checkpoint_kind"]
    ).any():
        raise F16IntegrityError("F16 checkpoint inventory is not exactly 4x3x2")
    checkpoint_table_path = output_root / "f16_checkpoint_inventory.parquet"
    curves_path = output_root / "f16_training_curves.parquet"
    atomic_write_parquet(checkpoints, checkpoint_table_path)
    atomic_write_parquet(curves, curves_path)
    manifest: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_checkpoint_manifest",
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_checkpoint_rule": "minimum validation MSE with >1e-4 improvement and earliest tie retention",
        "sensitivity_checkpoint": "epoch20",
        "checkpoint_inventory": {
            "path": _relative(checkpoint_table_path, repo_root),
            "sha256": sha256_file(checkpoint_table_path),
            "rows": len(checkpoints),
        },
        "training_curves": {
            "path": _relative(curves_path, repo_root),
            "sha256": sha256_file(curves_path),
            "rows": len(curves),
        },
        "training_source_fingerprint": next(iter(training_source_fingerprints)),
        "evaluation_source": source,
        "production_progress_sha256": sha256_file(output_root / "f16_production_progress.json"),
        "cohort_manifest_sha256": sha256_file(output_root / "f16_cohort_manifest.json"),
        "failure_table_sha256": sha256_file(output_root / "f16_failures.parquet"),
        "test_barrier": "locked",
        "test_accessed": False,
        "failures": [],
    }
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    atomic_write_json(output_root / "f16_checkpoint_manifest.json", manifest)
    return checkpoints, manifest


def _verify_checkpoint_manifest(repo_root: Path, output_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    _progress, cohort = _verify_locked_inputs(repo_root, output_root)
    manifest_path = output_root / "f16_checkpoint_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "frozen" or manifest.get("test_barrier") != "locked":
        raise F16IntegrityError("F16 checkpoint manifest is not frozen with locked test")
    if manifest.get("evaluation_source") != _source_inventory(repo_root):
        raise F16IntegrityError("F16 evaluation source changed after checkpoint freeze")
    record = manifest["checkpoint_inventory"]
    path = repo_root / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise F16IntegrityError("F16 checkpoint inventory drift")
    checkpoints = pd.read_parquet(path)
    if len(checkpoints) != 24:
        raise F16IntegrityError("F16 checkpoint inventory row count drift")
    return checkpoints, manifest, cohort


def _verify_parquet(repo_root: Path, record: Mapping[str, Any]) -> pd.DataFrame:
    path = repo_root / str(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise F16IntegrityError(f"F16 parquet drift: {path}")
    return pd.read_parquet(path)


def _row_plan(repo_root: Path, cohort: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labels = {budget: _verify_parquet(repo_root, cohort["label_budgets"][budget]) for budget in BUDGETS}
    covariance = _verify_parquet(repo_root, cohort["cohorts"]["train"])
    validation = _verify_parquet(repo_root, cohort["cohorts"]["validation"])
    b16 = labels["b_16"].copy()
    union = pd.concat([b16, covariance], ignore_index=True)
    duplicate = union[union.duplicated("row_key", keep=False)]
    if not duplicate.empty:
        identity_counts = duplicate.groupby("row_key")[["source_row_position", "endpoint_index"]].nunique()
        if (identity_counts > 1).any().any():
            raise F16IntegrityError("F16 train union has ambiguous duplicated row identity")
    union = union.drop_duplicates("row_key", keep="first").sort_values(
        "source_row_position", kind="stable"
    ).reset_index(drop=True)
    if union["source_row_position"].duplicated().any():
        raise F16IntegrityError("F16 train union source positions are not unique")
    label_keys = {budget: set(frame["row_key"].astype(str)) for budget, frame in labels.items()}
    if not (
        label_keys["b_1_4"] <= label_keys["b_1"] <= label_keys["b_4"] <= label_keys["b_16"]
    ):
        raise F16IntegrityError("F16 label budgets are not nested")
    union_keys = union["row_key"].astype(str)
    for budget in BUDGETS:
        union[f"member_{budget}"] = union_keys.isin(label_keys[budget]).to_numpy()
        if int(union[f"member_{budget}"].sum()) != len(labels[budget]):
            raise F16IntegrityError(f"F16 union lost rows from {budget}")
    covariance_keys = set(covariance["row_key"].astype(str))
    union["member_covariance"] = union_keys.isin(covariance_keys).to_numpy()
    if int(union["member_covariance"].sum()) != len(covariance):
        raise F16IntegrityError("F16 union lost covariance rows")
    plan = {
        "train_union_rows": len(union),
        "validation_rows": len(validation),
        "train_union_row_key_sha256": sha256_string_sequence(union_keys),
        "train_union_endpoint_sha256": sha256_array(union["endpoint_index"].to_numpy(dtype=np.int64)),
        "validation_row_key_sha256": sha256_string_sequence(validation["row_key"].astype(str)),
        "validation_endpoint_sha256": sha256_array(validation["endpoint_index"].to_numpy(dtype=np.int64)),
        "budget_rows": {budget: len(labels[budget]) for budget in BUDGETS},
        "covariance_rows": len(covariance),
    }
    return union, validation, plan


def _target_arrays(
    repo_root: Path,
    bundle_root: Path,
    bundle_manifest: Mapping[str, Any],
    union: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    del repo_root
    train = _array_from_record(bundle_root, bundle_manifest["targets"]["arrays"]["train"])
    val = _array_from_record(bundle_root, bundle_manifest["targets"]["arrays"]["validation"])
    y_union = np.asarray(
        train[union["source_row_position"].to_numpy(dtype=np.int64)], dtype=np.float32
    )
    y_validation = np.asarray(
        val[validation["source_row_position"].to_numpy(dtype=np.int64)], dtype=np.float32
    )
    if y_union.shape != (len(union), 23) or y_validation.shape != (len(validation), 23):
        raise F16IntegrityError("F16 train/validation target shape drift")
    if not np.isfinite(y_union).all() or not np.isfinite(y_validation).all():
        raise F16IntegrityError("non-finite F16 train/validation targets")
    return y_union, y_validation


def _stats_arrays(stats: SufficientStats, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_n": np.asarray(stats.n, dtype=np.int64),
        f"{prefix}_x_sum": stats.x_sum,
        f"{prefix}_y_sum": stats.y_sum,
        f"{prefix}_xtx": stats.xtx,
        f"{prefix}_xty": stats.xty,
        f"{prefix}_yty": stats.yty,
    }


def _stats_from_npz(data: Mapping[str, np.ndarray], prefix: str) -> SufficientStats:
    return SufficientStats(
        n=int(np.asarray(data[f"{prefix}_n"]).item()),
        x_sum=np.asarray(data[f"{prefix}_x_sum"], dtype=np.float64),
        y_sum=np.asarray(data[f"{prefix}_y_sum"], dtype=np.float64),
        xtx=np.asarray(data[f"{prefix}_xtx"], dtype=np.float64),
        xty=np.asarray(data[f"{prefix}_xty"], dtype=np.float64),
        yty=np.asarray(data[f"{prefix}_yty"], dtype=np.float64),
    )


def _save_feature_stats(
    path: Path,
    stats_by_readout: Mapping[str, F16FeatureStats],
    source_fingerprint: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "readouts": np.asarray(list(READOUTS), dtype="U32"),
        "source_fingerprint": np.asarray(canonical_json_sha256(source_fingerprint)),
    }
    inventory: dict[str, Any] = {}
    for readout, stats in stats_by_readout.items():
        prefix = "last" if readout == "last_concat512" else "meanK"
        arrays[f"{prefix}_budget_labels"] = np.asarray(list(stats.budgets), dtype="U16")
        for budget, value in stats.budgets.items():
            arrays.update(_stats_arrays(value, f"{prefix}_budget_{budget}"))
        arrays.update(_stats_arrays(stats.validation, f"{prefix}_validation"))
        if stats.covariance is not None:
            arrays[f"{prefix}_covariance_n"] = np.asarray(stats.covariance.n, dtype=np.int64)
            arrays[f"{prefix}_covariance_x_sum"] = stats.covariance.x_sum
            arrays[f"{prefix}_covariance_xtx"] = stats.covariance.xtx
        inventory[readout] = {
            "budgets": list(stats.budgets),
            "validation_rows": stats.validation.n,
            "covariance_rows": stats.covariance.n if stats.covariance is not None else 0,
        }
    atomic_savez(path, **arrays)
    return {
        "path": _relative(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "source_fingerprint": source_fingerprint,
        "inventory": inventory,
    }


def load_feature_stats(repo_root: Path, record: Mapping[str, Any]) -> dict[str, F16FeatureStats]:
    path = repo_root / str(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise F16IntegrityError(f"F16 sufficient-statistics cache drift: {path}")
    expected_fingerprint = canonical_json_sha256(record["source_fingerprint"])
    result = {}
    with np.load(path, allow_pickle=False) as data:
        if str(data["source_fingerprint"].item()) != expected_fingerprint:
            raise F16IntegrityError(f"F16 sufficient-statistics fingerprint drift: {path}")
        for readout in READOUTS:
            prefix = "last" if readout == "last_concat512" else "meanK"
            budgets = [str(value) for value in data[f"{prefix}_budget_labels"].tolist()]
            covariance = None
            if f"{prefix}_covariance_n" in data.files:
                covariance = FeatureMoments(
                    n=int(data[f"{prefix}_covariance_n"].item()),
                    x_sum=np.asarray(data[f"{prefix}_covariance_x_sum"], dtype=np.float64),
                    xtx=np.asarray(data[f"{prefix}_covariance_xtx"], dtype=np.float64),
                )
            result[readout] = F16FeatureStats(
                covariance=covariance,
                budgets={budget: _stats_from_npz(data, f"{prefix}_budget_{budget}") for budget in budgets},
                validation=_stats_from_npz(data, f"{prefix}_validation"),
            )
    return result


def _empty_stats(budgets: Iterable[str], with_covariance: bool) -> dict[str, F16FeatureStats]:
    return {
        readout: F16FeatureStats(
            covariance=FeatureMoments.zeros(512) if with_covariance else None,
            budgets={budget: SufficientStats.zeros(512, 23) for budget in budgets},
            validation=SufficientStats.zeros(512, 23),
        )
        for readout in READOUTS
    }


def _add_train_batch(
    accumulators: Mapping[str, F16FeatureStats],
    features: Mapping[str, np.ndarray],
    targets: np.ndarray,
    masks: Mapping[str, np.ndarray],
    covariance_mask: np.ndarray | None,
) -> None:
    for readout, x in features.items():
        value = accumulators[readout]
        if covariance_mask is not None and value.covariance is not None and covariance_mask.any():
            value.covariance.add_rows(x[covariance_mask])
        for budget, mask in masks.items():
            if mask.any():
                value.budgets[budget].add_rows(x[mask], targets[mask])


def _add_validation_batch(
    accumulators: Mapping[str, F16FeatureStats],
    features: Mapping[str, np.ndarray],
    targets: np.ndarray,
) -> None:
    for readout, x in features.items():
        accumulators[readout].validation.add_rows(x, targets)


def _poolings(grid: torch.Tensor) -> dict[str, np.ndarray]:
    batch = grid.shape[0]
    last = (
        grid[:, -1]
        .reshape(batch, -1)
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    mean = (
        grid.mean(dim=1)
        .reshape(batch, -1)
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if last.shape[1] != 512 or mean.shape[1] != 512:
        raise F16IntegrityError("F16 encoder emitted non-canonical readout dimension")
    return {"last_concat512": last, "meanK_concatS": mean}


def _load_f16_encoder(path: Path, expected: Mapping[str, Any], device: torch.device):
    if sha256_file(path) != expected["sha256"]:
        raise F16IntegrityError(f"F16 extraction checkpoint hash drift: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = SupervisedGrid(
        HorizonJEPAEncoderConfig.from_dict(payload["encoder_config"]),
        ReadoutConfig(**payload["readout_config"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    stock_stats = {key: np.asarray(value) for key, value in payload["stock_stats"].items()}
    return model.encoder, stock_stats


@torch.inference_mode()
def _extract_new_checkpoint_stats(
    checkpoint: Mapping[str, Any],
    *,
    dataset_path: Path,
    union: pd.DataFrame,
    validation: pd.DataFrame,
    y_union: np.ndarray,
    y_validation: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, F16FeatureStats]:
    kind = str(checkpoint["checkpoint_kind"])
    trained_budget = str(checkpoint["trained_budget"])
    budgets = BUDGETS if kind == "best" else (trained_budget,)
    with_covariance = kind == "best"
    accumulators = _empty_stats(budgets, with_covariance)
    with np.load(dataset_path, allow_pickle=False) as raw:
        book = raw["book"]
        mid_z = raw["mid_z"]
        stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
        encoder, stock_stats = _load_f16_encoder(
            Path(checkpoint["absolute_path"]), checkpoint, device
        )
        train_rows = union if kind == "best" else union.loc[union[f"member_{trained_budget}"]].copy()
        train_positions = train_rows.index.to_numpy(dtype=np.int64)
        train_loader = DataLoader(
            RawWindowDataset(
                book,
                mid_z,
                stock_ids,
                train_rows["endpoint_index"].to_numpy(dtype=np.int64),
                stock_stats,
                20,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
        cursor = 0
        for book_batch, stock_batch in train_loader:
            n = len(book_batch)
            positions = train_positions[cursor : cursor + n]
            grid = encoder(
                book_batch.to(device, non_blocking=True),
                stock_batch.to(device, non_blocking=True),
            )
            features = _poolings(grid)
            masks = {
                budget: union.loc[positions, f"member_{budget}"].to_numpy(dtype=bool)
                for budget in budgets
            }
            covariance_mask = (
                union.loc[positions, "member_covariance"].to_numpy(dtype=bool)
                if with_covariance
                else None
            )
            _add_train_batch(
                accumulators,
                features,
                y_union[positions],
                masks,
                covariance_mask,
            )
            cursor += n
        if cursor != len(train_rows):
            raise F16IntegrityError("F16 train extraction cursor mismatch")
        validation_loader = DataLoader(
            RawWindowDataset(
                book,
                mid_z,
                stock_ids,
                validation["endpoint_index"].to_numpy(dtype=np.int64),
                stock_stats,
                20,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
        cursor = 0
        for book_batch, stock_batch in validation_loader:
            n = len(book_batch)
            grid = encoder(
                book_batch.to(device, non_blocking=True),
                stock_batch.to(device, non_blocking=True),
            )
            _add_validation_batch(accumulators, _poolings(grid), y_validation[cursor : cursor + n])
            cursor += n
        if cursor != len(validation):
            raise F16IntegrityError("F16 validation extraction cursor mismatch")
        del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return accumulators


def _extract_anchor_stats(
    bundle_root: Path,
    bundle_manifest: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    union: pd.DataFrame,
    validation: pd.DataFrame,
    y_union: np.ndarray,
    y_validation: np.ndarray,
    chunk_rows: int,
) -> dict[str, F16FeatureStats]:
    accumulators = _empty_stats(BUDGETS, True)
    union_positions = union["source_row_position"].to_numpy(dtype=np.int64)
    validation_positions = validation["source_row_position"].to_numpy(dtype=np.int64)
    for readout in READOUTS:
        x_train = _feature_array(bundle_root, bundle_manifest, arm, seed, readout, "train")
        x_validation = _feature_array(bundle_root, bundle_manifest, arm, seed, readout, "validation")
        value = accumulators[readout]
        for start in range(0, len(union), chunk_rows):
            stop = min(start + chunk_rows, len(union))
            x = np.asarray(x_train[union_positions[start:stop]], dtype=np.float32)
            for budget in BUDGETS:
                mask = union[f"member_{budget}"].to_numpy(dtype=bool)[start:stop]
                if mask.any():
                    value.budgets[budget].add_rows(x[mask], y_union[start:stop][mask])
            covariance_mask = union["member_covariance"].to_numpy(dtype=bool)[start:stop]
            if covariance_mask.any() and value.covariance is not None:
                value.covariance.add_rows(x[covariance_mask])
        for start in range(0, len(validation), chunk_rows):
            stop = min(start + chunk_rows, len(validation))
            x = np.asarray(x_validation[validation_positions[start:stop]], dtype=np.float32)
            value.validation.add_rows(x, y_validation[start:stop])
    return accumulators


def extract_f16_validation_statistics(
    repo_root: Path,
    output_root: Path,
    bundle_root: Path,
    dataset_path: Path,
    *,
    device_name: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 2,
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    """Extract all fixed train/validation statistics; test access is impossible here."""
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    bundle_root = bundle_root.resolve()
    dataset_path = dataset_path.resolve()
    checkpoints, checkpoint_manifest, cohort = _verify_checkpoint_manifest(repo_root, output_root)
    bundle_manifest_path = bundle_root / "manifest.json"
    protocol = _read_json(output_root / "f16_manifest.json")
    if sha256_file(bundle_manifest_path) != protocol["bundle_manifest_sha256"]:
        raise F16IntegrityError("F16 production bundle manifest drift")
    bundle_manifest = _read_json(bundle_manifest_path)
    if sha256_file(dataset_path) != bundle_manifest["provenance"]["dataset_sha256"]:
        raise F16IntegrityError("F16 canonical dataset drift before extraction")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise F16IntegrityError("F16 validation extraction requested unavailable GPU")
    if batch_size <= 0 or chunk_rows <= 0 or not 0 <= num_workers <= 8:
        raise ValueError("invalid F16 extraction runtime parameter")
    device = torch.device(device_name)
    union, validation, row_plan = _row_plan(repo_root, cohort)
    y_union, y_validation = _target_arrays(
        repo_root, bundle_root, bundle_manifest, union, validation
    )
    target_hashes = {
        "train_union_targets_sha256": sha256_array(y_union),
        "validation_targets_sha256": sha256_array(y_validation),
    }
    cache_root = output_root / "sufficient_statistics"
    cache_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "f16_validation_extraction_state.json"
    source = _source_inventory(repo_root)
    base_fingerprint = {
        "algorithm": "f16_fixed_union_sufficient_statistics.v1",
        "evaluation_source_fingerprint": source["fingerprint"],
        "checkpoint_manifest_sha256": sha256_file(output_root / "f16_checkpoint_manifest.json"),
        "cohort_manifest_sha256": sha256_file(output_root / "f16_cohort_manifest.json"),
        "bundle_manifest_sha256": sha256_file(bundle_manifest_path),
        "row_plan": row_plan,
        "target_hashes": target_hashes,
        "test_access": "forbidden",
    }
    state = (
        _read_json(state_path)
        if state_path.is_file()
        else {
            "schema_name": "thesis.experiment01.f16_validation_extraction_state",
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "extracting",
            "base_fingerprint": base_fingerprint,
            "feature_sets": {},
            "test_barrier": "locked",
            "test_accessed": False,
        }
    )
    if state.get("base_fingerprint") != base_fingerprint or state.get("test_accessed") is not False:
        raise F16IntegrityError("stale or test-contaminated F16 validation extraction state")
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def valid_existing(key: str, fingerprint: Mapping[str, Any]) -> bool:
        record = state["feature_sets"].get(key)
        if not isinstance(record, Mapping) or record.get("source_fingerprint") != fingerprint:
            return False
        path = repo_root / str(record.get("path", ""))
        return path.is_file() and sha256_file(path) == record.get("sha256")

    ordered = checkpoints.sort_values(
        ["checkpoint_kind", "trained_budget", "encoder_seed"], kind="stable"
    )
    for row in ordered.itertuples(index=False):
        key = f"supervised_f16_{row.trained_budget}_seed{int(row.encoder_seed)}_{row.checkpoint_kind}"
        fingerprint = {
            **base_fingerprint,
            "feature_key": key,
            "checkpoint_sha256": row.sha256,
            "checkpoint_kind": row.checkpoint_kind,
            "trained_budget": row.trained_budget,
            "encoder_seed": int(row.encoder_seed),
        }
        if valid_existing(key, fingerprint):
            print(f"F16 validation extraction: skip verified {key}", flush=True)
            continue
        print(f"F16 validation extraction: start {key}", flush=True)
        checkpoint = row._asdict()
        checkpoint["absolute_path"] = str(repo_root / row.path)
        stats = _extract_new_checkpoint_stats(
            checkpoint,
            dataset_path=dataset_path,
            union=union,
            validation=validation,
            y_union=y_union,
            y_validation=y_validation,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        cache_path = cache_root / f"{key}.npz"
        state["feature_sets"][key] = {
            "encoder_family": "supervised_f16",
            "trained_budget": row.trained_budget,
            "encoder_seed": int(row.encoder_seed),
            "checkpoint_kind": row.checkpoint_kind,
            "checkpoint_sha256": row.sha256,
            **_save_feature_stats(cache_path, stats, fingerprint, repo_root),
        }
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(state_path, state)
        peak_rss = max(peak_rss, process.memory_info().rss)

    for arm in ANCHOR_ARMS:
        for seed in (0, 1, 2):
            key = f"{arm}_seed{seed}_canonical"
            checkpoint_records = [
                record
                for record in bundle_manifest["canonical_checkpoints"].values()
                if record["arm"] == arm and int(record["seed"]) == seed
            ]
            if len(checkpoint_records) != 1:
                raise F16IntegrityError(f"canonical F16 anchor is not unique: {arm}/seed{seed}")
            checkpoint = checkpoint_records[0]
            path = Path(checkpoint["path"])
            if not path.is_absolute():
                path = repo_root / path
            if sha256_file(path) != checkpoint["sha256"]:
                raise F16IntegrityError(f"canonical F16 anchor checkpoint drift: {arm}/seed{seed}")
            fingerprint = {
                **base_fingerprint,
                "feature_key": key,
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_kind": "canonical_epoch20",
                "trained_budget": "historical_500k",
                "encoder_seed": seed,
            }
            if valid_existing(key, fingerprint):
                print(f"F16 validation extraction: skip verified {key}", flush=True)
                continue
            print(f"F16 validation extraction: start {key}", flush=True)
            stats = _extract_anchor_stats(
                bundle_root,
                bundle_manifest,
                arm=arm,
                seed=seed,
                union=union,
                validation=validation,
                y_union=y_union,
                y_validation=y_validation,
                chunk_rows=chunk_rows,
            )
            cache_path = cache_root / f"{key}.npz"
            state["feature_sets"][key] = {
                "encoder_family": arm,
                "trained_budget": "historical_500k",
                "encoder_seed": seed,
                "checkpoint_kind": "canonical_epoch20",
                "checkpoint_sha256": checkpoint["sha256"],
                **_save_feature_stats(cache_path, stats, fingerprint, repo_root),
            }
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(state_path, state)
            peak_rss = max(peak_rss, process.memory_info().rss)
    expected = 24 + 9
    if len(state["feature_sets"]) != expected:
        raise F16IntegrityError(
            f"F16 validation extraction has {len(state['feature_sets'])} feature sets, expected {expected}"
        )
    state["status"] = "complete"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["runtime"] = {
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "peak_ram_bytes_this_invocation": max(
            peak_rss, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
        "peak_vram_bytes_this_invocation": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "chunk_rows": chunk_rows,
    }
    state["test_barrier"] = "locked"
    state["test_accessed"] = False
    state["manifest_fingerprint"] = canonical_json_sha256(
        {key: value for key, value in state.items() if key != "manifest_fingerprint"}
    )
    atomic_write_json(state_path, state)
    return state


def _target_blocks(definitions: Sequence[Mapping[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for block in ("directional", "volatility", "timing"):
        indices = np.asarray(
            [index for index, target in enumerate(definitions) if target["block"] == block],
            dtype=np.int64,
        )
        independent = np.asarray(
            [offset for offset, index in enumerate(indices) if bool(definitions[int(index)]["independent"])],
            dtype=np.int64,
        )
        if not len(indices) or not len(independent):
            raise F16IntegrityError(f"F16 target block is empty: {block}")
        result[block] = (indices, independent)
    return result


def _largest_alpha_index_within_1e12(scores: np.ndarray) -> int:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).any():
        raise F16IntegrityError("F16 alpha selection has no finite validation score")
    best = float(np.nanmax(values))
    eligible = np.flatnonzero(np.isfinite(values) & (values >= best - 1e-12))
    if not len(eligible):
        raise F16IntegrityError("F16 alpha tie set is empty")
    return int(eligible[-1])


def _tune_f16_alpha(
    design,
    validation_stats: SufficientStats,
    independent_target_indices: Sequence[int],
):
    independent = np.asarray(independent_target_indices, dtype=np.int64)
    if independent.ndim != 1 or not len(independent):
        raise ValueError("F16 alpha tuning requires independent targets")
    scores = np.full(len(ALPHA_GRID), -np.inf, dtype=np.float64)
    for index, alpha in enumerate(ALPHA_GRID):
        model = fit_alpha(design, float(alpha))
        evaluated = evaluate_stats(model, validation_stats)
        eligible = independent[evaluated.valid[independent]]
        if len(eligible):
            scores[index] = float(np.mean(evaluated.values[eligible]))
    chosen = _largest_alpha_index_within_1e12(scores)
    return float(ALPHA_GRID[chosen]), chosen, float(scores[chosen]), scores


def _reader_rows(
    *,
    feature: Mapping[str, Any],
    readout: str,
    axis: str,
    analysis_budget: str,
    train_stats: SufficientStats,
    validation_stats: SufficientStats,
    definitions: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, tuple[np.ndarray, np.ndarray]],
    feature_view: str = "full",
    projection: np.ndarray | None = None,
    whitening_k: int | None = None,
    include_ols: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    projected_train = projection_stats(train_stats, projection) if projection is not None else train_stats
    projected_validation = (
        projection_stats(validation_stats, projection) if projection is not None else validation_stats
    )
    for block, (indices, independent_local) in blocks.items():
        train_block = select_targets(projected_train, indices)
        validation_block = select_targets(projected_validation, indices)
        design = transformed_design(train_block)
        tuned_alpha, tuned_index, tuned_score, _scores = _tune_f16_alpha(
            design, validation_block, independent_local
        )
        model = fit_alpha(design, tuned_alpha)
        train_score = evaluate_stats(model, train_block)
        validation_score = evaluate_stats(model, validation_block)
        selection_key = {
            "feature_key": feature["feature_key"],
            "readout": readout,
            "axis": axis,
            "analysis_budget": analysis_budget,
            "target_block": block,
            "feature_view": feature_view,
            "whitening_k": whitening_k,
            "reader_family": "ridge_trace_normalized",
        }
        selections.append(
            {
                **selection_key,
                "alpha": tuned_alpha,
                "alpha_grid_index": tuned_index,
                "validation_r2_mean_independent": tuned_score,
                "lambda_absolute": model.lambda_absolute,
                "tie_break": "largest alpha within numerical 1e-12-equivalent tie",
                "selected_without_test": True,
            }
        )
        for local, global_index in enumerate(indices):
            target = definitions[int(global_index)]
            results.append(
                {
                    **selection_key,
                    "encoder_family": feature["encoder_family"],
                    "trained_budget": feature["trained_budget"],
                    "encoder_seed": feature["encoder_seed"],
                    "checkpoint_kind": feature["checkpoint_kind"],
                    "checkpoint_sha256": feature["checkpoint_sha256"],
                    "target_index": int(global_index),
                    "target_name": target["name"],
                    "target_independent": bool(target["independent"]),
                    "n_train_rows": train_block.n,
                    "n_validation_rows": validation_block.n,
                    "feature_dimension": train_block.dimension,
                    "alpha": tuned_alpha,
                    "lambda_absolute": model.lambda_absolute,
                    "train_r2": float(train_score.values[local]),
                    "validation_r2": float(validation_score.values[local]),
                    "fit_status": "ok" if validation_score.valid[local] else "invalid",
                    "failure_reason": validation_score.reasons[local],
                    "test_r2": np.nan,
                    "test_accessed": False,
                }
            )
        if include_ols:
            ols = fit_alpha(design, 0.0)
            ols_train = evaluate_stats(ols, train_block)
            ols_validation = evaluate_stats(ols, validation_block)
            for local, global_index in enumerate(indices):
                target = definitions[int(global_index)]
                results.append(
                    {
                        **{**selection_key, "reader_family": "min_norm_ols"},
                        "encoder_family": feature["encoder_family"],
                        "trained_budget": feature["trained_budget"],
                        "encoder_seed": feature["encoder_seed"],
                        "checkpoint_kind": feature["checkpoint_kind"],
                        "checkpoint_sha256": feature["checkpoint_sha256"],
                        "target_index": int(global_index),
                        "target_name": target["name"],
                        "target_independent": bool(target["independent"]),
                        "n_train_rows": train_block.n,
                        "n_validation_rows": validation_block.n,
                        "feature_dimension": train_block.dimension,
                        "alpha": 0.0,
                        "lambda_absolute": 0.0,
                        "train_r2": float(ols_train.values[local]),
                        "validation_r2": float(ols_validation.values[local]),
                        "fit_status": "ok" if ols_validation.valid[local] else "invalid",
                        "failure_reason": ols_validation.reasons[local],
                        "test_r2": np.nan,
                        "test_accessed": False,
                    }
                )
    return results, selections


def _geometry_rows(
    feature: Mapping[str, Any],
    readout: str,
    stats: F16FeatureStats,
    definitions: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    if stats.covariance is None or "b_16" not in stats.budgets:
        return []
    spectrum = eigensystem(stats.covariance.covariance, stats.covariance.n)
    values = spectrum.eigenvalues
    vectors = spectrum.eigenvectors
    rank = spectrum.diagnostics.numerical_rank
    tolerance = spectrum.diagnostics.numerical_tolerance
    labelled = stats.budgets["b_16"]
    target_variance = labelled.target_centered_ss / labelled.n
    cross_pc = vectors.T @ labelled.cross
    direction_valid = (np.arange(len(values)) < rank) & (values > tolerance)
    target_valid = target_variance > (
        np.finfo(np.float64).eps * np.maximum(labelled.yty / labelled.n, 1.0)
    )
    mass = np.full((len(values), labelled.n_targets), np.nan, dtype=np.float64)
    good_directions = np.flatnonzero(direction_valid)
    good_targets = np.flatnonzero(target_valid)
    mass[np.ix_(good_directions, good_targets)] = (
        np.square(cross_pc[np.ix_(good_directions, good_targets)])
        / values[good_directions, None]
        / target_variance[good_targets][None, :]
    )
    cumulative = np.nancumsum(np.where(np.isfinite(mass), mass, 0.0), axis=0)
    total = cumulative[rank - 1]
    trace = float(values[:rank].sum())
    schedule = tuple(
        sorted({value for value in (1, 2, 4, 8, 16, 32, 64, 128, 256, 508, rank) if value <= rank})
    )
    rows: list[dict[str, Any]] = []
    base = {
        "feature_key": feature["feature_key"],
        "encoder_family": feature["encoder_family"],
        "trained_budget": feature["trained_budget"],
        "encoder_seed": feature["encoder_seed"],
        "checkpoint_kind": feature["checkpoint_kind"],
        "checkpoint_sha256": feature["checkpoint_sha256"],
        "readout": readout,
        "covariance_rows": stats.covariance.n,
        "reader_rows": labelled.n,
        "covariance_trace": trace,
        "covariance_trace_over_dim": trace / len(values),
        "numerical_rank": rank,
        "numerical_tolerance": tolerance,
        "test_accessed": False,
    }
    for direction in range(len(values)):
        rows.append(
            {
                **base,
                "metric_family": "covariance_spectrum",
                "target_block": "all",
                "target_index": -1,
                "target_name": "__none__",
                "target_independent": False,
                "k": direction + 1,
                "eigenvalue": float(values[direction]),
                "cumulative_variance_fraction": (
                    float(values[: direction + 1].sum() / trace) if direction < rank and trace > 0 else np.nan
                ),
                "predictive_mass": np.nan,
                "cumulative_predictive_mass": np.nan,
                "cumulative_mass_fraction": np.nan,
                "validation_r2": np.nan,
                "retention": np.nan,
                "feature_view": "pca_direction",
            }
        )
    for k in schedule:
        for target_index, target in enumerate(definitions):
            rows.append(
                {
                    **base,
                    "metric_family": "predictive_mass",
                    "target_block": target["block"],
                    "target_index": target_index,
                    "target_name": target["name"],
                    "target_independent": bool(target["independent"]),
                    "k": k,
                    "eigenvalue": float(values[k - 1]),
                    "cumulative_variance_fraction": float(values[:k].sum() / trace),
                    "predictive_mass": float(mass[k - 1, target_index]) if target_valid[target_index] else np.nan,
                    "cumulative_predictive_mass": (
                        float(cumulative[k - 1, target_index]) if target_valid[target_index] else np.nan
                    ),
                    "cumulative_mass_fraction": (
                        float(cumulative[k - 1, target_index] / total[target_index])
                        if target_valid[target_index] and total[target_index] > 0
                        else np.nan
                    ),
                    "validation_r2": np.nan,
                    "retention": np.nan,
                    "feature_view": "top_pca_cumulative",
                }
            )
    return rows


def analyze_f16_validation(
    repo_root: Path,
    output_root: Path,
    bundle_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Freeze all reader/geometry selections using train and validation only."""
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    bundle_root = bundle_root.resolve()
    _checkpoints, checkpoint_manifest, _cohort = _verify_checkpoint_manifest(repo_root, output_root)
    extraction_path = output_root / "f16_validation_extraction_state.json"
    extraction = _read_json(extraction_path)
    if extraction.get("status") != "complete" or extraction.get("test_accessed") is not False:
        raise F16IntegrityError("F16 validation extraction is incomplete or test-contaminated")
    unsigned = {key: value for key, value in extraction.items() if key != "manifest_fingerprint"}
    if extraction.get("manifest_fingerprint") != canonical_json_sha256(unsigned):
        raise F16IntegrityError("F16 validation extraction manifest fingerprint drift")
    if extraction.get("base_fingerprint", {}).get("evaluation_source_fingerprint") != _source_inventory(repo_root)["fingerprint"]:
        raise F16IntegrityError("F16 evaluation source drift before validation analysis")
    bundle_manifest = _read_json(bundle_root / "manifest.json")
    definitions = bundle_manifest["targets"]["definitions"]
    if len(definitions) != 23:
        raise F16IntegrityError("F16 target inventory is not canonical 23-target set")
    blocks = _target_blocks(definitions)
    feature_records = []
    for key, record in sorted(extraction["feature_sets"].items()):
        feature_records.append({"feature_key": key, **record})
    results: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    started = time.perf_counter()
    for feature in feature_records:
        print(f"F16 validation analysis: {feature['feature_key']}", flush=True)
        cached = load_feature_stats(repo_root, feature)
        is_new = feature["encoder_family"] == "supervised_f16"
        is_best = feature["checkpoint_kind"] in {"best", "canonical_epoch20"}
        for readout in READOUTS:
            stats = cached[readout]
            if is_new:
                axis_a_budgets = (str(feature["trained_budget"]),)
            else:
                axis_a_budgets = BUDGETS
            for budget in axis_a_budgets:
                reader_rows, selection_rows = _reader_rows(
                    feature=feature,
                    readout=readout,
                    axis="A_label_matched",
                    analysis_budget=budget,
                    train_stats=stats.budgets[budget],
                    validation_stats=stats.validation,
                    definitions=definitions,
                    blocks=blocks,
                )
                results.extend(reader_rows)
                selections.extend(selection_rows)
            if is_best and "b_16" in stats.budgets:
                reader_rows, selection_rows = _reader_rows(
                    feature=feature,
                    readout=readout,
                    axis="B_fixed_b16",
                    analysis_budget="b_16",
                    train_stats=stats.budgets["b_16"],
                    validation_stats=stats.validation,
                    definitions=definitions,
                    blocks=blocks,
                    include_ols=True,
                )
                results.extend(reader_rows)
                selections.extend(selection_rows)
                common, contrast = role_projections()
                for view, projection in (("role_common", common), ("role_contrast", contrast)):
                    role_rows, role_selections = _reader_rows(
                        feature=feature,
                        readout=readout,
                        axis="B_fixed_b16",
                        analysis_budget="b_16",
                        train_stats=stats.budgets["b_16"],
                        validation_stats=stats.validation,
                        definitions=definitions,
                        blocks=blocks,
                        feature_view=view,
                        projection=projection,
                    )
                    results.extend(role_rows)
                    selections.extend(role_selections)
                geometry.extend(_geometry_rows(feature, readout, stats, definitions, blocks))
                if readout == "last_concat512" and stats.covariance is not None:
                    spectrum = eigensystem(stats.covariance.covariance, stats.covariance.n)
                    for k in WHITENING_DEPTHS:
                        if k > spectrum.diagnostics.numerical_rank:
                            raise F16IntegrityError(
                                f"frozen F16 whitening k={k} exceeds rank for {feature['feature_key']}"
                            )
                        scales = np.ones(512, dtype=np.float64)
                        if k:
                            scales[:k] = 1.0 / np.sqrt(spectrum.eigenvalues[:k])
                        transform = (spectrum.eigenvectors * scales[None, :]) @ spectrum.eigenvectors.T
                        white_rows, white_selections = _reader_rows(
                            feature=feature,
                            readout=readout,
                            axis="B_fixed_b16",
                            analysis_budget="b_16",
                            train_stats=stats.budgets["b_16"],
                            validation_stats=stats.validation,
                            definitions=definitions,
                            blocks=blocks,
                            feature_view="whiten_topk",
                            projection=transform,
                            whitening_k=k,
                        )
                        results.extend(white_rows)
                        selections.extend(white_selections)
    results_frame = pd.DataFrame(results)
    geometry_frame = pd.DataFrame(geometry)
    selections_frame = pd.DataFrame(selections)
    if results_frame.empty or geometry_frame.empty or selections_frame.empty:
        raise F16IntegrityError("F16 validation analysis produced an empty required table")
    if results_frame["test_accessed"].astype(bool).any() or geometry_frame["test_accessed"].astype(bool).any():
        raise F16IntegrityError("F16 validation result claims test access")
    if selections_frame.duplicated(
        [
            "feature_key",
            "readout",
            "axis",
            "analysis_budget",
            "target_block",
            "feature_view",
            "whitening_k",
            "reader_family",
        ]
    ).any():
        raise F16IntegrityError("F16 validation selection keys are not unique")
    aggregate_source = results_frame.loc[
        results_frame["target_independent"].astype(bool)
        & results_frame["reader_family"].eq("ridge_trace_normalized")
        & results_frame["axis"].eq("B_fixed_b16")
    ]
    aggregate = (
        aggregate_source.groupby(
            [
                "feature_key",
                "encoder_family",
                "trained_budget",
                "encoder_seed",
                "checkpoint_kind",
                "checkpoint_sha256",
                "readout",
                "target_block",
                "feature_view",
                "whitening_k",
            ],
            dropna=False,
            as_index=False,
        )["validation_r2"]
        .mean()
    )
    derived_geometry: list[dict[str, Any]] = []
    role = aggregate.loc[aggregate["feature_view"].isin(["full", "role_common", "role_contrast"])]
    for keys, group in role.groupby(
        [
            "feature_key",
            "encoder_family",
            "trained_budget",
            "encoder_seed",
            "checkpoint_kind",
            "checkpoint_sha256",
            "readout",
            "target_block",
        ],
        sort=False,
    ):
        values = dict(zip(group["feature_view"], group["validation_r2"]))
        if not {"full", "role_common", "role_contrast"} <= set(values):
            continue
        if abs(float(values["full"])) < 1e-12:
            raise F16IntegrityError(f"zero full validation R2 in role retention: {keys[0]}")
        base = dict(
            zip(
                (
                    "feature_key",
                    "encoder_family",
                    "trained_budget",
                    "encoder_seed",
                    "checkpoint_kind",
                    "checkpoint_sha256",
                    "readout",
                    "target_block",
                ),
                keys,
            )
        )
        for view in ("role_common", "role_contrast"):
            derived_geometry.append(
                {
                    **base,
                    "metric_family": "role_retention",
                    "target_index": -1,
                    "target_name": "__block_mean_independent__",
                    "target_independent": True,
                    "k": np.nan,
                    "eigenvalue": np.nan,
                    "cumulative_variance_fraction": np.nan,
                    "predictive_mass": np.nan,
                    "cumulative_predictive_mass": np.nan,
                    "cumulative_mass_fraction": np.nan,
                    "validation_r2": float(values[view]),
                    "full_validation_r2": float(values["full"]),
                    "retention": float(values[view] / values["full"]),
                    "feature_view": view,
                    "whitening_k": np.nan,
                    "test_accessed": False,
                }
            )
    full = aggregate.loc[aggregate["feature_view"].eq("full")]
    for keys, group in full.groupby(
        [
            "feature_key",
            "encoder_family",
            "trained_budget",
            "encoder_seed",
            "checkpoint_kind",
            "checkpoint_sha256",
            "target_block",
        ],
        sort=False,
    ):
        values = dict(zip(group["readout"], group["validation_r2"]))
        if set(READOUTS) <= set(values):
            derived_geometry.append(
                {
                    **dict(
                        zip(
                            (
                                "feature_key",
                                "encoder_family",
                                "trained_budget",
                                "encoder_seed",
                                "checkpoint_kind",
                                "checkpoint_sha256",
                                "target_block",
                            ),
                            keys,
                        )
                    ),
                    "readout": "last_minus_meanK",
                    "metric_family": "pooling_loss",
                    "target_index": -1,
                    "target_name": "__block_mean_independent__",
                    "target_independent": True,
                    "k": np.nan,
                    "eigenvalue": np.nan,
                    "cumulative_variance_fraction": np.nan,
                    "predictive_mass": np.nan,
                    "cumulative_predictive_mass": np.nan,
                    "cumulative_mass_fraction": np.nan,
                    "validation_r2": float(values["last_concat512"] - values["meanK_concatS"]),
                    "full_validation_r2": float(values["last_concat512"]),
                    "retention": (
                        float(values["meanK_concatS"] / values["last_concat512"])
                        if abs(float(values["last_concat512"])) >= 1e-12
                        else np.nan
                    ),
                    "feature_view": "last_to_meanK",
                    "whitening_k": np.nan,
                    "test_accessed": False,
                }
            )
    whiten = aggregate.loc[aggregate["feature_view"].eq("whiten_topk")]
    for row in whiten.itertuples(index=False):
        derived_geometry.append(
            {
                "feature_key": row.feature_key,
                "encoder_family": row.encoder_family,
                "trained_budget": row.trained_budget,
                "encoder_seed": row.encoder_seed,
                "checkpoint_kind": row.checkpoint_kind,
                "checkpoint_sha256": row.checkpoint_sha256,
                "readout": row.readout,
                "target_block": row.target_block,
                "metric_family": "whitening_bridge",
                "target_index": -1,
                "target_name": "__block_mean_independent__",
                "target_independent": True,
                "k": int(row.whitening_k),
                "eigenvalue": np.nan,
                "cumulative_variance_fraction": np.nan,
                "predictive_mass": np.nan,
                "cumulative_predictive_mass": np.nan,
                "cumulative_mass_fraction": np.nan,
                "validation_r2": float(row.validation_r2),
                "full_validation_r2": np.nan,
                "retention": np.nan,
                "feature_view": "whiten_topk",
                "whitening_k": int(row.whitening_k),
                "test_accessed": False,
            }
        )
    if derived_geometry:
        geometry_frame = pd.concat(
            [geometry_frame, pd.DataFrame(derived_geometry)], ignore_index=True, sort=False
        )
    validation_results_path = output_root / "f16_validation_results.parquet"
    validation_geometry_path = output_root / "f16_validation_geometry.parquet"
    selection_table_path = output_root / "f16_validation_selections.parquet"
    atomic_write_parquet(results_frame, validation_results_path)
    atomic_write_parquet(geometry_frame, validation_geometry_path)
    atomic_write_parquet(selections_frame, selection_table_path)
    selection_manifest: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_validation_selections",
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "validation_selections_frozen_test_locked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "alpha_grid": [float(value) for value in ALPHA_GRID],
        "alpha_rule": "maximize aggregate validation R2 over independent targets; largest alpha wins numerical tie",
        "whitening_depths": list(WHITENING_DEPTHS),
        "checkpoint_manifest_sha256": sha256_file(output_root / "f16_checkpoint_manifest.json"),
        "validation_extraction_state_sha256": sha256_file(extraction_path),
        "failure_table_sha256": sha256_file(output_root / "f16_failures.parquet"),
        "artifacts": {
            "validation_results": {
                "path": _relative(validation_results_path, repo_root),
                "sha256": sha256_file(validation_results_path),
                "rows": len(results_frame),
            },
            "validation_geometry": {
                "path": _relative(validation_geometry_path, repo_root),
                "sha256": sha256_file(validation_geometry_path),
                "rows": len(geometry_frame),
            },
            "selection_table": {
                "path": _relative(selection_table_path, repo_root),
                "sha256": sha256_file(selection_table_path),
                "rows": len(selections_frame),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
        "test_barrier": "locked",
        "test_accessed": False,
        "failures": [],
    }
    selection_manifest["manifest_fingerprint"] = canonical_json_sha256(selection_manifest)
    atomic_write_json(output_root / "f16_validation_selection_manifest.json", selection_manifest)
    return results_frame, geometry_frame, selection_manifest

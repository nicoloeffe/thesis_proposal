"""Fail-closed audit of the nine canonical Experiment 01 encoders.

The checkpoint payloads are the primary evidence.  The repository training
sources are inspected separately: their hashes and relevant implementation
properties are recorded, but they are not presented as cryptographic proof of
the exact source tree used to create checkpoints that do not embed a source
fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from experiment01.io import sha256_array, sha256_file


SCHEMA_NAME = "thesis.experiment01.training_protocol_audit"
SCHEMA_VERSION = 1

EXPECTED_TARGETS = [
    f"{feature}@{horizon}"
    for feature in (
        "d_spread_z",
        "d_microprice_rel",
        "d_best_bid_rel",
        "d_best_ask_rel",
        "d_top_imbalance",
    )
    for horizon in (1, 5, 10, 20)
] + ["realized_vol@5", "realized_vol@20"]

SHARED_ENCODER_CONFIG = {
    "L": 10,
    "n_stocks": 7,
    "K": 20,
    "S": 4,
    "raw_per_token": 10,
    "d_model": 128,
    "d_latent": 32,
    "spatial_n_layers": 2,
    "spatial_n_heads": 4,
    "spatial_d_ffn": 256,
    "temporal_n_layers": 2,
    "temporal_n_heads": 4,
    "temporal_d_ffn": 256,
    "temporal_causal": False,
    "dropout": 0.1,
    "stock_emb_init_scale": 0.02,
}

SHARED_TRAIN_ARGS = {
    "max_train_samples": 500_000,
    "max_val_samples": 50_000,
    "val_frac": 0.1,
    "vol_clip": 5.0,
    "split_seed": 0,
    "epochs": 20,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "eta_min_frac": 0.01,
}

SOURCE_FILES = (
    "models/model_tokenizer_t.py",
    "training/train_tokenizer_t.py",
    "training/train_jepa_horizon.py",
    "training/train_jepa_masked.py",
    "training/train_supervised_grid.py",
)


class TrainingAuditError(RuntimeError):
    """Raised when any canonical training-protocol gate fails."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TrainingAuditError("PyTorch is required to inspect checkpoints") from exc
    return torch


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    torch = _torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location="cpu")


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"thesis.canonical-json.v1\0")
    digest.update(payload)
    return digest.hexdigest()


def _stock_stats_fingerprint(stock_stats: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(
        {
            key: _legacy_sha256_array(np.asarray(value, dtype=np.float32))
            for key, value in sorted(stock_stats.items())
        }
    )


def _legacy_sha256_array(array: np.ndarray) -> str:
    """Stage-1 array fingerprint used by the canonical checkpoint manifest."""
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays cannot be fingerprinted")
    dtype = value.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(value.astype(dtype, copy=False))
    header = json.dumps(
        {"dtype": dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"thesis.sha256-array.v1\0")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _state_numel(state: Mapping[str, Any], prefix: str | None = None) -> int:
    total = 0
    for name, value in state.items():
        if prefix is not None and not str(name).startswith(prefix):
            continue
        if hasattr(value, "numel"):
            total += int(value.numel())
    return total


def _read_history(path: Path, expected_epochs: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TrainingAuditError(f"missing history: {path}")
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != expected_epochs:
        raise TrainingAuditError(
            f"{path}: expected {expected_epochs} history rows, got {len(history)}"
        )
    epochs = [int(row.get("epoch", -1)) for row in history]
    if epochs != list(range(1, expected_epochs + 1)):
        raise TrainingAuditError(f"{path}: non-consecutive epoch history {epochs}")
    return history


def _metric_from_history(arm: str, row: Mapping[str, Any]) -> float:
    if arm == "supervised":
        return float(row["val_mse"])
    metric = "L_total" if arm == "jepa_horizon" else "L_jepa"
    return float(row["val"][metric])


def _metric_from_checkpoint(arm: str, checkpoint: Mapping[str, Any]) -> float:
    metric = {
        "supervised": "mse",
        "jepa_horizon": "L_total",
        "jepa_masked": "L_jepa",
    }[arm]
    return float(checkpoint["val_metrics"][metric])


def _optimizer_settings(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    groups = checkpoint["optimizer_state_dict"].get("param_groups", [])
    if len(groups) != 1:
        raise TrainingAuditError(
            f"expected one optimizer parameter group, found {len(groups)}"
        )
    group = groups[0]
    return {
        "algorithm": "AdamW",
        "lr_at_checkpoint": float(group["lr"]),
        "initial_lr": float(group.get("initial_lr", checkpoint["train_args"]["lr"])),
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group.get("amsgrad", False)),
    }


def _assert_equal(
    errors: list[str], label: str, actual: Any, expected: Any, tolerance: float = 0.0
) -> None:
    if isinstance(expected, float):
        equal = math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance)
    else:
        equal = actual == expected
    if not equal:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def _source_inventory(repo_root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise TrainingAuditError(f"missing training source: {relative}")
        files[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    supervised_source = (repo_root / "training/train_supervised_grid.py").read_text(
        encoding="utf-8"
    )
    horizon_source = (repo_root / "training/train_jepa_horizon.py").read_text(
        encoding="utf-8"
    )
    masked_source = (repo_root / "training/train_jepa_masked.py").read_text(
        encoding="utf-8"
    )
    observations = {
        "encoder_seed_controls_train_row_subsample_all_arms": all(
            "np.random.default_rng(args.seed)" in source
            for source in (supervised_source, horizon_source, masked_source)
        ),
        "jepa_validation_subsample_uses_seed_plus_one": all(
            "np.random.default_rng(args.seed + 1)" in source
            for source in (horizon_source, masked_source)
        ),
        "supervised_validation_reuses_advanced_train_rng": (
            supervised_source.count("np.random.default_rng(args.seed)") == 1
            and "rng.choice(train_pos" in supervised_source
            and "rng.choice(val_pos" in supervised_source
        ),
        "supervised_scheduler_steps_after_epoch_in_batch_count_loop": (
            "for _ in range(steps_per_epoch):" in supervised_source
            and "scheduler.step()" in supervised_source
            and supervised_source.index("for _ in range(steps_per_epoch):")
            < supervised_source.index("va_loss, va_pred, va_true")
        ),
        "jepa_scheduler_is_epoch_level_cosine": all(
            "CosineAnnealingLR" in source and "scheduler.step()" in source
            for source in (horizon_source, masked_source)
        ),
    }
    if not all(observations.values()):
        missing = [key for key, value in observations.items() if not value]
        raise TrainingAuditError(
            "repository implementation no longer supports audited observations: "
            + ", ".join(missing)
        )
    return {
        "files": files,
        "source_fingerprint": _canonical_json_sha256(
            {key: value["sha256"] for key, value in sorted(files.items())}
        ),
        "observations": observations,
        "evidentiary_limit": (
            "The checkpoints do not embed a Git commit or source hash. These are "
            "hashes of the current archived implementation, which is consistent "
            "with the payload metadata but is not cryptographic proof of the exact "
            "training-time source tree."
        ),
    }


def _audit_checkpoint(
    repo_root: Path,
    record: Mapping[str, Any],
    expected_stock_stats_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    relative = Path(str(record["path"]))
    path = repo_root / relative
    if not path.is_file():
        return {}, [f"{relative}: checkpoint is missing"]
    _assert_equal(errors, f"{relative} size", path.stat().st_size, record["size_bytes"])
    _assert_equal(errors, f"{relative} SHA-256", sha256_file(path), record["sha256"])

    checkpoint = _load_checkpoint(path)
    arm = str(record["arm"])
    seed = int(record["seed"])
    args = checkpoint.get("train_args", {})
    cfg = checkpoint.get("enc_cfg", {})
    _assert_equal(errors, f"{relative} epoch", int(checkpoint.get("epoch", -1)), 20)
    _assert_equal(errors, f"{relative} manifest epoch", int(record["epoch"]), 20)
    _assert_equal(errors, f"{relative} seed", int(args.get("seed", -1)), seed)
    for key, expected in SHARED_ENCODER_CONFIG.items():
        _assert_equal(errors, f"{relative} enc_cfg.{key}", cfg.get(key), expected)
    for key, expected in SHARED_TRAIN_ARGS.items():
        _assert_equal(errors, f"{relative} train_args.{key}", args.get(key), expected)

    stock_stats_sha256 = _stock_stats_fingerprint(checkpoint.get("stock_stats", {}))
    _assert_equal(
        errors,
        f"{relative} stock_stats SHA-256",
        stock_stats_sha256,
        expected_stock_stats_sha256,
    )

    history_path = path.parent / "history.json"
    history = _read_history(history_path, expected_epochs=20)
    history_metric = _metric_from_history(arm, history[-1])
    checkpoint_metric = _metric_from_checkpoint(arm, checkpoint)
    _assert_equal(
        errors,
        f"{relative} epoch-20 validation metric",
        checkpoint_metric,
        history_metric,
        tolerance=1e-12,
    )
    best_row = min(history, key=lambda row: _metric_from_history(arm, row))

    if arm == "supervised":
        _assert_equal(
            errors,
            f"{relative} target inventory",
            checkpoint.get("target_names"),
            EXPECTED_TARGETS,
        )
        state = checkpoint["model_state_dict"]
        encoder_numel = _state_numel(state, prefix="encoder.")
        head_numel = _state_numel(state) - encoder_numel
        objective = {
            "name": "standardized 22-target supervised regression",
            "loss": "mean-squared error",
            "target_count": 22,
            "target_names": EXPECTED_TARGETS,
            "target_standardization": "per target, training-row mean and std",
            "timing_directly_supervised": False,
        }
        schedule = {
            "family": "linear warm-up plus cosine decay",
            "warmup_fraction": float(args["warmup_frac"]),
            "implementation_cadence": (
                "the scheduler is advanced steps_per_epoch times only after each "
                "training epoch; learning rate is therefore constant inside an epoch"
            ),
        }
    else:
        state = checkpoint["online_state_dict"]
        encoder_numel = _state_numel(state)
        head_numel = _state_numel(checkpoint["predictor_state_dict"])
        if arm == "jepa_horizon":
            _assert_equal(
                errors,
                f"{relative} horizons",
                list(checkpoint.get("horizons", [])),
                [0, 1, 5, 10, 20],
            )
            objective = {
                "name": "horizon latent prediction",
                "loss": str(args["loss_type"]),
                "horizons": [0, 1, 5, 10, 20],
                "ema_target": True,
                "ema_tau": [float(args["tau_start"]), float(args["tau_end"])],
            }
        else:
            objective = {
                "name": "structured masked latent prediction",
                "loss": str(args["loss_type"]),
                "mask_ratio": [
                    float(args["mask_ratio_low"]),
                    float(args["mask_ratio_high"]),
                ],
                "ema_target": True,
                "ema_tau": [float(args["tau_start"]), float(args["tau_end"])],
            }
        schedule = {
            "family": "epoch-level cosine annealing",
            "T_max_epochs": int(args["epochs"]),
            "eta_min_fraction": float(args["eta_min_frac"]),
        }

    steps_per_epoch = int(args["max_train_samples"]) // int(args["batch_size"])
    effective_rows_per_epoch = steps_per_epoch * int(args["batch_size"])
    record_out = {
        "id": record["id"],
        "arm": arm,
        "seed": seed,
        "epoch": 20,
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "history_path": history_path.relative_to(repo_root).as_posix(),
        "history_sha256": sha256_file(history_path),
        "format_version": checkpoint.get("format_version"),
        "encoder_parameters": encoder_numel,
        "objective_head_parameters": head_numel,
        "optimizer": _optimizer_settings(checkpoint),
        "schedule": schedule,
        "objective": objective,
        "train_rows_sampled": int(args["max_train_samples"]),
        "effective_rows_per_epoch_drop_last": effective_rows_per_epoch,
        "validation_rows": int(args["max_val_samples"]),
        "steps_per_epoch": steps_per_epoch,
        "effective_optimizer_updates": steps_per_epoch * int(args["epochs"]),
        "canonical_epoch20_validation_metric": checkpoint_metric,
        "history_best_epoch": int(best_row["epoch"]),
        "history_best_validation_metric": _metric_from_history(arm, best_row),
        "stock_stats_sha256": stock_stats_sha256,
        "train_args": {
            key: args[key]
            for key in sorted(set(SHARED_TRAIN_ARGS) | {"seed", "num_workers"})
        },
    }
    return record_out, errors


def _volatility_mask(book: np.ndarray, clip: float, chunk_rows: int = 100_000) -> np.ndarray:
    mask = np.empty(book.shape[0], dtype=bool)
    for start in range(0, book.shape[0], chunk_rows):
        stop = min(start + chunk_rows, book.shape[0])
        volumes = book[start:stop, :, :, 1]
        mask[start:stop] = np.max(np.abs(volumes), axis=(1, 2)) <= clip
    return mask


def _subsample_positions(
    train_pos: np.ndarray,
    validation_pos: np.ndarray,
    seed: int,
    arm: str,
    n_train: int = 500_000,
    n_validation: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    if arm == "supervised":
        rng = np.random.default_rng(seed)
        train = np.sort(rng.choice(train_pos, size=n_train, replace=False))
        validation = np.sort(rng.choice(validation_pos, size=n_validation, replace=False))
    else:
        train_rng = np.random.default_rng(seed)
        validation_rng = np.random.default_rng(seed + 1)
        train = np.sort(train_rng.choice(train_pos, size=n_train, replace=False))
        validation = np.sort(
            validation_rng.choice(validation_pos, size=n_validation, replace=False)
        )
    return train, validation


def reconstruct_historical_row_identity(dataset_path: Path) -> dict[str, Any]:
    """Reconstruct the row-selection rules used by the archived trainers."""
    from training.train_tokenizer_t import (
        compute_valid_endpoints,
        grouped_split_by_stock_day,
    )

    if not dataset_path.is_file():
        raise TrainingAuditError(f"dataset is missing: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        book = data["book"]
        stock_ids = data["stock_ids"].astype(np.int64, copy=False)
        day_ids = data["day_ids"].astype(np.int64, copy=False)
        vol_mask = _volatility_mask(book, clip=5.0)
    valid_t = compute_valid_endpoints(stock_ids, day_ids, 20, 20, vol_mask)
    train_pos, validation_pos = grouped_split_by_stock_day(
        stock_ids, day_ids, valid_t, 0.1, 0
    )

    rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, int, str], np.ndarray] = {}
    for arm in ("jepa_horizon", "jepa_masked", "supervised"):
        for seed in (0, 1, 2):
            train_selected, validation_selected = _subsample_positions(
                train_pos, validation_pos, seed, arm
            )
            for split, positions in (
                ("train", train_selected),
                ("validation", validation_selected),
            ):
                endpoints = valid_t[positions].astype(np.int64, copy=False)
                selected[(arm, seed, split)] = endpoints
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "split": split,
                        "rows": int(endpoints.size),
                        "raw_endpoint_sha256": sha256_array(endpoints),
                    }
                )

    same_seed_train = {
        str(seed): len(
            {
                row["raw_endpoint_sha256"]
                for row in rows
                if row["seed"] == seed and row["split"] == "train"
            }
        )
        == 1
        for seed in (0, 1, 2)
    }
    same_seed_validation = {
        str(seed): len(
            {
                row["raw_endpoint_sha256"]
                for row in rows
                if row["seed"] == seed and row["split"] == "validation"
            }
        )
        == 1
        for seed in (0, 1, 2)
    }
    train_seed_overlap: list[dict[str, Any]] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        a = selected[("jepa_horizon", left, "train")]
        b = selected[("jepa_horizon", right, "train")]
        overlap = int(np.intersect1d(a, b, assume_unique=True).size)
        train_seed_overlap.append(
            {
                "seed_pair": [left, right],
                "intersection_rows": overlap,
                "fraction_of_500000": overlap / 500_000.0,
            }
        )
    if not all(same_seed_train.values()):
        raise TrainingAuditError("same-seed historical train rows differ across arms")
    if any(same_seed_validation.values()):
        raise TrainingAuditError(
            "expected supervised and JEPA historical validation subsamples to differ"
        )
    return {
        "dataset_path": dataset_path.as_posix(),
        "dataset_sha256": sha256_file(dataset_path),
        "valid_endpoints": int(valid_t.size),
        "grouped_train_candidates": int(train_pos.size),
        "grouped_validation_candidates": int(validation_pos.size),
        "selection_records": rows,
        "same_seed_train_rows_matched_across_arms": same_seed_train,
        "same_seed_validation_rows_matched_across_arms": same_seed_validation,
        "train_overlap_between_encoder_seeds": train_seed_overlap,
        "checkpoint_provenance_limit": (
            "Row hashes were not embedded in the historical checkpoints. These "
            "identities are a deterministic reconstruction from the canonical "
            "dataset and archived sampling implementations."
        ),
    }


def audit_training_protocol(
    repo_root: Path,
    manifest_path: Path,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    _assert_equal(errors, "manifest status", manifest.get("status"), "complete")
    _assert_equal(errors, "manifest checkpoint count", manifest.get("canonical_file_count"), 9)
    _assert_equal(errors, "manifest selection rule", manifest.get("selection_rule"),
                  "epoch 20 for each frozen arm and encoder seed")

    records: list[dict[str, Any]] = []
    for checkpoint_record in manifest.get("checkpoints", []):
        audited, checkpoint_errors = _audit_checkpoint(
            repo_root,
            checkpoint_record,
            str(manifest["stock_stats_sha256"]),
        )
        records.append(audited)
        errors.extend(checkpoint_errors)
    if len(records) != 9:
        errors.append(f"expected 9 audited checkpoint records, got {len(records)}")
    identities = {(row.get("arm"), row.get("seed")) for row in records}
    expected_identities = {
        (arm, seed)
        for arm in ("jepa_horizon", "jepa_masked", "supervised")
        for seed in (0, 1, 2)
    }
    _assert_equal(errors, "arm/seed inventory", identities, expected_identities)
    if errors:
        raise TrainingAuditError("training audit failed:\n- " + "\n- ".join(errors))

    audit: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "canonical_manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "canonical_manifest_sha256": sha256_file(manifest_path),
        "canonical_bundle_manifest_sha256": manifest["source_bundle_manifest_sha256"],
        "stock_stats_sha256": manifest["stock_stats_sha256"],
        "shared_contract": {
            "encoder_config": SHARED_ENCODER_CONFIG,
            "train_args": SHARED_TRAIN_ARGS,
            "input_grid": "20 timesteps x 4 role tokens x 128 dimensions",
            "train_drop_last": True,
            "effective_rows_per_epoch": 499_968,
            "effective_optimizer_updates": 39_060,
            "canonical_selection": "epoch 20 for every arm and seed",
            "seed_policy": (
                "split_seed=0 is shared; encoder seed controls initialization, "
                "minibatch order and the capped historical row subsample"
            ),
        },
        "checkpoints": records,
        "source_audit": _source_inventory(repo_root),
        "matched_and_confounded": {
            "matched": [
                "encoder architecture and tokenization",
                "candidate stock-day split and split_seed=0",
                "500000-row training cap and 50000-row validation cap",
                "20 epochs, batch size 256 and 39060 optimizer updates",
                "AdamW learning rate, weight decay, gradient clipping and final LR fraction",
                "same-seed historical training-row identities across all three arms",
            ],
            "objective_required_differences": [
                "supervised MSE head versus JEPA predictor and EMA target",
                "horizon prediction versus structured masking",
                "objective-specific validation loss and compute per update",
            ],
            "historical_confounds_or_limits": [
                "encoder seeds change the capped training-row sample as well as initialization and minibatch order",
                "supervised validation sampling reuses the train RNG, while JEPA validation uses seed+1",
                "the supervised LR scheduler advances in a post-epoch batch-count loop, so LR is constant within each epoch",
                "epoch 20 is the scientific checkpoint even where best validation occurred earlier",
                "checkpoint payloads omit training-time Git/source and hardware/runtime fingerprints",
                "supervised pretraining directly used the 20 future-feature and two volatility targets later probed; timing was not a direct target but is correlated with them",
            ],
        },
        "failures": [],
    }
    if dataset_path is not None:
        row_audit = reconstruct_historical_row_identity(dataset_path.resolve())
        row_audit["dataset_path"] = dataset_path.resolve().relative_to(repo_root).as_posix()
        audit["historical_row_identity"] = row_audit
    return audit


def write_audit_json(path: Path, audit: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_training_protocol(audit: Mapping[str, Any], audit_sha256: str) -> str:
    """Render the human-readable protocol from the validated audit artifact."""
    if audit.get("status") != "passed" or audit.get("failures"):
        raise TrainingAuditError("refusing to render a non-passing training audit")
    checkpoints = list(audit["checkpoints"])
    row_identity = audit.get("historical_row_identity")
    if row_identity is None:
        raise TrainingAuditError(
            "historical row-identity reconstruction is required for the public protocol"
        )

    lines = [
        "# Experiment 01 — training protocol and hyperparameter audit",
        "",
        "- **Audit status:** passed, fail-closed",
        "- **Canonical inventory:** 3 objectives × 3 encoder seeds",
        "- **Scientific checkpoint rule:** epoch 20 for every arm and seed",
        f"- **Machine audit SHA-256:** `{audit_sha256}`",
        "",
        "This document separates what was genuinely matched across the historical",
        "encoder arms from objective-required differences and from residual training",
        "confounds. The machine-readable evidence is",
        "[`TRAINING_PROTOCOL_AUDIT.json`](TRAINING_PROTOCOL_AUDIT.json).",
        "",
        "## 1. Shared encoder and tokenization",
        "",
        "Every arm uses the same non-causal grid backbone. A 20-snapshot input",
        "window produces four role tokens at each timestep, so the encoder output is",
        "`20 × 4 × 128`. Experiment 01 reads either the four final-timestep tokens",
        "(`last_concat512`) or the four tokens after averaging each role through",
        "time (`meanK_concatS`).",
        "",
        "| property | frozen value |",
        "|---|---:|",
        "| window length `K` | 20 |",
        "| role tokens per timestep `S` | 4 |",
        "| model dimension | 128 |",
        "| spatial transformer | 2 layers, 4 heads, FFN 256 |",
        "| temporal transformer | 2 layers, 4 heads, FFN 256, non-causal |",
        "| dropout | 0.1 |",
        "| stocks | 7 |",
        f"| encoder parameters | {checkpoints[0]['encoder_parameters']:,} |",
        "",
        "The token roles and normalization are shared. The nine checkpoints agree on",
        f"the stock-statistics fingerprint `{audit['stock_stats_sha256']}`.",
        "",
        "## 2. Shared optimization envelope",
        "",
        "| property | frozen value |",
        "|---|---:|",
        "| grouped split seed | 0 |",
        "| candidate train cap | 500,000 endpoints |",
        "| candidate validation cap | 50,000 endpoints |",
        "| batch size | 256 |",
        "| epochs | 20 |",
        "| train batches per epoch (`drop_last=True`) | 1,953 |",
        "| gradient-bearing rows per epoch | 499,968 |",
        "| optimizer updates | 39,060 |",
        "| optimizer | AdamW, betas=(0.9, 0.999), eps=1e-8 |",
        "| learning rate | 3e-4 |",
        "| weight decay | 1e-4 |",
        "| gradient-norm clip | 1.0 |",
        "| final LR fraction | 0.01 |",
        "",
        "These equal row and update caps do **not** imply equal FLOPs. JEPA also",
        "evaluates a predictor and EMA target network, whereas supervised training",
        "uses an attention-pooling regression head.",
        "",
        "## 3. Objective-required differences",
        "",
        "| arm | objective | auxiliary trainable parameters | schedule |",
        "|---|---|---:|---|",
    ]
    exemplar = {row["arm"]: row for row in checkpoints}
    lines.extend(
        [
            "| `jepa_horizon` | L1 latent prediction at horizons 0,1,5,10,20; EMA target | "
            f"{exemplar['jepa_horizon']['objective_head_parameters']:,} | epoch-level cosine |",
            "| `jepa_masked` | L1 structured masked latent prediction; mask ratio 0.50–0.65; EMA target | "
            f"{exemplar['jepa_masked']['objective_head_parameters']:,} | epoch-level cosine |",
            "| `supervised` | MSE on 22 standardized future targets | "
            f"{exemplar['supervised']['objective_head_parameters']:,} | warm-up + cosine, block-stepped per epoch |",
            "",
            "The supervised target inventory contains 20 future-feature targets",
            "(`spread`, `microprice`, `best bid`, `best ask`, `top imbalance` at",
            "horizons 1, 5, 10 and 20) plus realized volatility at horizons 5 and",
            "20. These include the target-aligned quantities later probed by",
            "Experiment 01. Timing was not a direct supervised target, although it is",
            "correlated with the trained targets. Consequently, Phase-I reader-label",
            "comparisons are representation-conditional and are not an end-to-end",
            "label-efficiency comparison. F16 is designed to measure this dependence.",
            "",
            "### Historical supervised scheduler detail",
            "",
            "The archived implementation describes the scheduler as per-update, but",
            "calls `scheduler.step()` 1,953 times in a loop only after completing each",
            "epoch. The learning rate is therefore constant during an epoch and jumps",
            "between epochs. This is a historical implementation property, not a",
            "reinterpretation of the checkpoints. F16 uses an explicitly update-based",
            "scheduler and treats the canonical supervised checkpoint only as a",
            "descriptive anchor.",
            "",
            "## 4. Exact historical row-identity reconstruction",
            "",
            f"The canonical dataset hash is `{row_identity['dataset_sha256']}`. After",
            "the archived endpoint filters there are",
            f"{row_identity['valid_endpoints']:,} valid endpoints:",
            f"{row_identity['grouped_train_candidates']:,} on the historical train",
            f"side and {row_identity['grouped_validation_candidates']:,} on its",
            "held-out side.",
            "",
            "For a fixed encoder seed, all three arms select the **same 500,000 train",
            "endpoints**. Thus arm comparisons paired at the same seed are row-matched.",
            "Across encoder seeds, however, the sampling seed changes the training-row",
            "sample as well as initialization and minibatch order:",
            "",
            "| seed pair | shared train rows | fraction of each 500k sample |",
            "|---|---:|---:|",
        ]
    )
    for overlap in row_identity["train_overlap_between_encoder_seeds"]:
        left, right = overlap["seed_pair"]
        lines.append(
            f"| {left}–{right} | {overlap['intersection_rows']:,} | "
            f"{100.0 * overlap['fraction_of_500000']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "Therefore the historical three-seed dispersion combines initialization,",
            "minibatch and data-subsample variation; it is not a pure optimization-seed",
            "error bar. F16 removes this confound by freezing the exact Phase-I seed-0",
            "row manifest and reusing it for encoder seeds 0, 1 and 2.",
            "",
            "The historical validation subsets are not matched between supervised and",
            "JEPA: supervised reuses the RNG after drawing train rows, while JEPA uses",
            "a fresh RNG seeded with `seed+1`. This does not alter the frozen epoch-20",
            "Phase-I representation comparison, but it prevents interpreting the old",
            "training-time validation metrics as a perfectly matched arm comparison.",
            "",
            "Row hashes were not embedded in the checkpoints. Their identities above",
            "are deterministic reconstructions from the canonical dataset and archived",
            "sampling implementation; this provenance limit is explicit in the JSON",
            "audit.",
            "",
            "## 5. Checkpoint selection and validation history",
            "",
            "The distributed scientific inventory deliberately selects epoch 20 for",
            "every cell. The training scripts also maintained a lowest-validation-loss",
            "`best.pt`, but that alias is not the canonical scientific checkpoint.",
            "",
            "| arm | seed | epoch-20 validation objective | best epoch | best objective |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in checkpoints:
        lines.append(
            f"| `{row['arm']}` | {row['seed']} | "
            f"{row['canonical_epoch20_validation_metric']:.9f} | "
            f"{row['history_best_epoch']} | {row['history_best_validation_metric']:.9f} |"
        )
    lines.extend(
        [
            "",
            "The masked objective reached its historical validation minimum at epochs",
            "6–8 and was worse by epoch 20. Horizon seed 2 and supervised seed 2 were",
            "minimally better at epoch 19. These facts are reported descriptively; the",
            "frozen Phase-I checkpoint rule is not changed.",
            "",
            "## 6. Canonical checkpoint inventory",
            "",
            "| arm | seed | path | bytes | SHA-256 |",
            "|---|---:|---|---:|---|",
        ]
    )
    for row in checkpoints:
        lines.append(
            f"| `{row['arm']}` | {row['seed']} | `{row['path']}` | "
            f"{row['size_bytes']:,} | `{row['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## 7. Evidentiary limits and F16 consequences",
            "",
            "The checkpoint payloads contain configuration, optimizer state, validation",
            "metrics, normalization statistics and target inventory. They do not contain",
            "a Git commit, training-source hash, hardware identity or wall-clock log.",
            "The source hashes in the machine audit describe the current archived",
            "implementation, which is consistent with the payloads but cannot prove the",
            "exact training-time checkout.",
            "",
            "F16 therefore freezes prospectively:",
            "",
            "- exact row-key manifests shared across encoder seeds;",
            "- an exact validation cohort and validation-only stopping rule;",
            "- a genuinely per-update LR schedule and common maximum-update cap;",
            "- RNG state, optimizer/scheduler state and resumable checkpoint identity;",
            "- a sealed test barrier until every validation selection is frozen;",
            "- the canonical epoch-20 model only as a descriptive upper anchor.",
            "",
            "No statement in this audit changes Phase I, its thresholds or technical",
            "classification A1.",
            "",
        ]
    )
    return "\n".join(lines)

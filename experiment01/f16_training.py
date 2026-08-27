"""Resumable, validation-only supervised encoder training for F16."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import random
import resource
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from experiment01.f16 import BUDGETS, F16IntegrityError, _relative, sha256_string_sequence
from experiment01.f16_convergence import _array_from_record
from experiment01.io import atomic_write_json, canonical_json_sha256, sha256_array, sha256_file
from experiment01.training_audit import EXPECTED_TARGETS, _stock_stats_fingerprint
from training.train_jepa_horizon import HorizonJEPAEncoderConfig
from training.train_supervised_grid import (
    ReadoutConfig,
    SupervisedGrid,
    SupervisedGridDataset,
    r2_per_target,
    standardize_targets,
    summarize_r2,
)


TRAINING_SCHEMA_VERSION = 1
ORDER_DOMAIN = 20260826
SOURCE_FILES = (
    "experiment01/f16.py",
    "experiment01/f16_training.py",
    "training/train_supervised_grid.py",
    "training/train_jepa_horizon.py",
    "training/train_tokenizer_t.py",
)


@dataclass(frozen=True)
class F16TrainingConfig:
    batch_size: int = 256
    validation_batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip: float = 1.0
    gradient_explosion_threshold: float = 1e6
    warmup_updates: int = 1953
    maximum_updates: int = 39060
    terminal_lr_fraction: float = 0.01
    validation_cadence_updates: int = 500
    patience_checks: int = 8
    minimum_improvement_mse: float = 1e-4
    minimum_updates: int = 4000
    encoder_pass_sensitivity: int = 20
    K: int = 20
    S: int = 4
    raw_per_token: int = 10
    d_model: int = 128
    spatial_n_layers: int = 2
    spatial_n_heads: int = 4
    spatial_d_ffn: int = 256
    temporal_n_layers: int = 2
    temporal_n_heads: int = 4
    temporal_d_ffn: int = 256
    temporal_causal: bool = False
    dropout: float = 0.1
    readout_dropout: float = 0.0

    def validate(self) -> None:
        frozen = F16TrainingConfig()
        if self != frozen:
            raise F16IntegrityError("F16 scientific training configuration drift")


class DeterministicPassBatchSampler(Sampler[list[int]]):
    """Stateless pass permutations addressed only by the global update cursor."""

    def __init__(
        self,
        n_rows: int,
        batch_size: int,
        seed: int,
        start_update: int,
        maximum_updates: int,
    ):
        if n_rows <= 0 or batch_size <= 0:
            raise ValueError("n_rows and batch_size must be positive")
        if not 0 <= start_update <= maximum_updates:
            raise ValueError("invalid update interval")
        self.n_rows = int(n_rows)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_update = int(start_update)
        self.maximum_updates = int(maximum_updates)
        self.steps_per_pass = math.ceil(self.n_rows / self.batch_size)

    @staticmethod
    def pass_seed(seed: int, pass_index: int) -> int:
        sequence = np.random.SeedSequence([ORDER_DOMAIN, int(seed), int(pass_index)])
        return int(sequence.generate_state(1, dtype=np.uint64)[0] & np.uint64(0x7FFF_FFFF_FFFF_FFFF))

    def permutation(self, pass_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.pass_seed(self.seed, pass_index))
        permutation = torch.randperm(self.n_rows, generator=generator)
        return permutation, generator.get_state()

    def __iter__(self) -> Iterator[list[int]]:
        cursor = self.start_update
        while cursor < self.maximum_updates:
            pass_index, batch_index = divmod(cursor, self.steps_per_pass)
            permutation, _ = self.permutation(pass_index)
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, self.n_rows)
            yield permutation[start:stop].tolist()
            cursor += 1

    def __len__(self) -> int:
        return self.maximum_updates - self.start_update

    def cursor_record(self, next_update: int) -> dict[str, Any]:
        pass_index, batch_index = divmod(int(next_update), self.steps_per_pass)
        _, state = self.permutation(pass_index)
        return {
            "domain": ORDER_DOMAIN,
            "seed": self.seed,
            "next_global_update": int(next_update),
            "next_pass_index": int(pass_index),
            "next_batch_index": int(batch_index),
            "steps_per_pass": self.steps_per_pass,
            "generator_state_sha256": sha256_array(state.numpy()),
            "generator_state": state,
        }


def learning_rate_multiplier(update_index: int, config: F16TrainingConfig) -> float:
    """Multiplier used by update ``update_index`` (zero-based)."""
    if update_index < 0:
        raise ValueError("update index must be non-negative")
    if update_index < config.warmup_updates:
        return (update_index + 1) / config.warmup_updates
    progress = (update_index - config.warmup_updates) / max(
        1, config.maximum_updates - config.warmup_updates - 1
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.terminal_lr_fraction + (1.0 - config.terminal_lr_fraction) * cosine


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npy", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, np.asarray(array), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_inventory(repo_root: Path) -> dict[str, Any]:
    files = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise F16IntegrityError(f"missing F16 training source: {relative}")
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {
        "files": files,
        "fingerprint": canonical_json_sha256(
            {key: value["sha256"] for key, value in sorted(files.items())}
        ),
    }


def _load_frozen_manifests(repo_root: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol_path = output_root / "f16_manifest.json"
    cohort_path = output_root / "f16_cohort_manifest.json"
    decision_path = output_root / "f16_cohort_decision.json"
    for path in (protocol_path, cohort_path, decision_path):
        if not path.is_file():
            raise F16IntegrityError(f"missing frozen F16 artifact: {path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if protocol.get("test_barrier") != "locked":
        raise F16IntegrityError("F16 protocol test barrier is not locked")
    if cohort.get("status") != "selected_and_frozen" or decision.get("status") != "passed":
        raise F16IntegrityError("F16 cohort convergence has not passed")
    if cohort.get("selected_cap_per_stock_day") != 128:
        raise F16IntegrityError("unexpected F16 selected cohort cap")
    if cohort["test_barrier"].get("test_targets_accessed") is not False:
        raise F16IntegrityError("test target barrier was violated")
    return protocol, cohort, decision


def prepare_target_cache(
    repo_root: Path,
    output_root: Path,
    bundle_root: Path,
    bundle_manifest: Mapping[str, Any],
    cohort_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract only train-budget and selected-validation targets; never test."""
    cache_root = output_root / "target_cache"
    manifest_path = cache_root / "manifest.json"
    source_fingerprint = {
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "cohort_manifest_sha256": sha256_file(output_root / "f16_cohort_manifest.json"),
        "target_names": EXPECTED_TARGETS,
        "target_columns": list(range(22)),
        "test_access": "forbidden",
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_fingerprint") != source_fingerprint:
            raise F16IntegrityError("stale F16 target cache manifest")
        for record in existing["arrays"].values():
            path = repo_root / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise F16IntegrityError("F16 target cache file drift")
        return existing

    definitions = bundle_manifest["targets"]["definitions"]
    names = [str(record["name"]) for record in definitions[:22]]
    if names != EXPECTED_TARGETS:
        raise F16IntegrityError("bundle supervised target inventory drift")
    train_targets = _array_from_record(
        bundle_root, bundle_manifest["targets"]["arrays"]["train"]
    )
    validation_targets = _array_from_record(
        bundle_root, bundle_manifest["targets"]["arrays"]["validation"]
    )
    arrays: dict[str, Any] = {}
    for budget in BUDGETS:
        label_record = cohort_manifest["label_budgets"][budget]
        label_path = repo_root / label_record["path"]
        if sha256_file(label_path) != label_record["sha256"]:
            raise F16IntegrityError(f"label manifest drift: {budget}")
        label = pd.read_parquet(label_path)
        positions = label["source_row_position"].to_numpy(dtype=np.int64)
        target = np.asarray(train_targets[positions, :22], dtype=np.float32)
        if target.shape != (len(label), 22) or not np.isfinite(target).all():
            raise F16IntegrityError(f"invalid target cache values: {budget}")
        path = cache_root / f"{budget}.npy"
        _atomic_save_npy(target, path)
        arrays[budget] = {
            "path": _relative(path, repo_root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "shape": list(target.shape),
            "dtype": target.dtype.name,
            "row_key_sequence_sha256": label_record["row_key_sequence_sha256"],
        }

    validation_record = cohort_manifest["cohorts"]["validation"]
    validation_path = repo_root / validation_record["path"]
    if sha256_file(validation_path) != validation_record["sha256"]:
        raise F16IntegrityError("selected validation cohort drift")
    validation_rows = pd.read_parquet(validation_path)
    positions = validation_rows["source_row_position"].to_numpy(dtype=np.int64)
    validation = np.asarray(validation_targets[positions, :22], dtype=np.float32)
    if validation.shape != (len(validation_rows), 22) or not np.isfinite(validation).all():
        raise F16IntegrityError("invalid validation target cache values")
    path = cache_root / "validation.npy"
    _atomic_save_npy(validation, path)
    arrays["validation"] = {
        "path": _relative(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(validation.shape),
        "dtype": validation.dtype.name,
        "row_key_sequence_sha256": validation_record["row_key_sequence_sha256"],
    }
    manifest = {
        "schema_name": "thesis.experiment01.f16_target_cache",
        "schema_version": 1,
        "status": "complete",
        "source_fingerprint": source_fingerprint,
        "arrays": arrays,
        "test_targets_accessed": False,
        "failures": [],
    }
    manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest


def _load_stock_stats(checkpoint_path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stock_stats"].items()}
    if _stock_stats_fingerprint(stats) != expected_sha256:
        raise F16IntegrityError("canonical stock-statistics fingerprint mismatch")
    return stats


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in value:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _model_configs(config: F16TrainingConfig) -> tuple[HorizonJEPAEncoderConfig, ReadoutConfig]:
    encoder = HorizonJEPAEncoderConfig(
        L=10,
        n_stocks=7,
        K=config.K,
        S=config.S,
        raw_per_token=config.raw_per_token,
        d_model=config.d_model,
        spatial_n_layers=config.spatial_n_layers,
        spatial_n_heads=config.spatial_n_heads,
        spatial_d_ffn=config.spatial_d_ffn,
        temporal_n_layers=config.temporal_n_layers,
        temporal_n_heads=config.temporal_n_heads,
        temporal_d_ffn=config.temporal_d_ffn,
        temporal_causal=config.temporal_causal,
        dropout=config.dropout,
    )
    return encoder, ReadoutConfig(
        d_model=config.d_model, out_dim=22, dropout=config.readout_dropout
    )


def _validate_model(
    model: SupervisedGrid,
    loader: DataLoader,
    device: torch.device,
    target_names: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise F16IntegrityError("non-finite F16 model parameter")
    total_loss = 0.0
    n_total = 0
    predictions: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    with torch.no_grad():
        for book, target, stock_ids in loader:
            book = book.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            stock_ids = stock_ids.to(device, non_blocking=True)
            prediction = model(book, stock_ids)
            if not torch.isfinite(prediction).all():
                raise F16IntegrityError("non-finite F16 validation prediction")
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise F16IntegrityError("non-finite F16 validation loss")
            total_loss += float(loss.item()) * len(book)
            n_total += len(book)
            predictions.append(prediction.float().cpu().numpy())
            truth.append(target.float().cpu().numpy())
    if n_total == 0:
        raise F16IntegrityError("empty F16 validation loader")
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(truth, axis=0)
    r2 = r2_per_target(y_true, y_pred)
    summary = summarize_r2(list(target_names), r2)
    return {
        "mse": total_loss / n_total,
        "n_rows": n_total,
        "r2_mean_all": summary["mean_all"],
        "r2_mean_future": summary["mean_future"],
        "r2_mean_vol": summary["mean_vol"],
    }


def _checkpoint_payload(
    *,
    model: SupervisedGrid,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: F16TrainingConfig,
    budget: str,
    seed: int,
    global_update: int,
    sampler: DeterministicPassBatchSampler,
    best_mse: float,
    best_update: int,
    no_improvement_checks: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    stock_stats: Mapping[str, np.ndarray],
    row_identity: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    encoder_config, readout_config = _model_configs(config)
    return {
        "schema_name": "thesis.experiment01.f16_checkpoint",
        "schema_version": TRAINING_SCHEMA_VERSION,
        "budget": budget,
        "encoder_seed": seed,
        "global_update": global_update,
        "completed_passes": global_update // sampler.steps_per_pass,
        "batch_in_next_pass": global_update % sampler.steps_per_pass,
        "steps_per_pass": sampler.steps_per_pass,
        "epoch20_update": sampler.steps_per_pass * config.encoder_pass_sensitivity,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": _rng_state(),
        "dataloader_state": sampler.cursor_record(global_update),
        "best_validation_mse": best_mse,
        "best_update": best_update,
        "no_improvement_checks": no_improvement_checks,
        "target_names": EXPECTED_TARGETS,
        "target_mean": np.asarray(target_mean, dtype=np.float32),
        "target_std": np.asarray(target_std, dtype=np.float32),
        "stock_stats": {key: np.asarray(value) for key, value in stock_stats.items()},
        "encoder_config": encoder_config.to_dict(),
        "readout_config": asdict(readout_config),
        "training_config": asdict(config),
        "row_identity": dict(row_identity),
        "source_inventory": dict(source_inventory),
        "validation_metrics": dict(validation_metrics),
        "test_accessed": False,
    }


def _load_resume_checkpoint(
    path: Path,
    model: SupervisedGrid,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    *,
    budget: str,
    seed: int,
    row_identity: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("budget") != budget or int(checkpoint.get("encoder_seed", -1)) != seed:
        raise F16IntegrityError("resume checkpoint job identity mismatch")
    if checkpoint.get("row_identity") != dict(row_identity):
        raise F16IntegrityError("resume checkpoint row identity drift")
    if checkpoint.get("source_inventory") != dict(source_inventory):
        raise F16IntegrityError("resume checkpoint source drift")
    if checkpoint.get("test_accessed") is not False:
        raise F16IntegrityError("resume checkpoint test barrier violation")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    _restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def _checkpoint_record(path: Path, repo_root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": _relative(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "global_update": int(payload["global_update"]),
        "validation_mse": float(payload["validation_metrics"]["mse"]),
    }


def _output_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def train_f16_cell(
    repo_root: Path,
    output_root: Path,
    bundle_root: Path,
    dataset_path: Path,
    checkpoint_manifest_path: Path,
    *,
    budget: str,
    seed: int,
    device_name: str = "cuda",
    num_workers: int = 2,
) -> dict[str, Any]:
    config = F16TrainingConfig()
    config.validate()
    if budget not in BUDGETS or seed not in (0, 1, 2):
        raise ValueError("F16 budget/seed is outside the frozen grid")
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    bundle_root = bundle_root.resolve()
    dataset_path = dataset_path.resolve()
    checkpoint_manifest_path = checkpoint_manifest_path.resolve()
    protocol, cohort_manifest, decision = _load_frozen_manifests(repo_root, output_root)
    if protocol.get("production_grid_authorized") is not False:
        raise F16IntegrityError("unexpected mutation of frozen production authorization")
    if decision.get("selected_cap_per_stock_day") != 128:
        raise F16IntegrityError("F16 convergence decision drift")
    if not 0 <= num_workers <= 8:
        raise ValueError("num_workers must be in [0,8]")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise F16IntegrityError("CUDA/ROCm device requested but unavailable")
    device = torch.device(device_name)
    source_inventory = _source_inventory(repo_root)

    bundle_manifest_path = bundle_root / "manifest.json"
    if sha256_file(bundle_manifest_path) != protocol["bundle_manifest_sha256"]:
        raise F16IntegrityError("production bundle manifest drift")
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(dataset_path) != bundle_manifest["provenance"]["dataset_sha256"]:
        raise F16IntegrityError("canonical dataset hash drift")
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    stock_stats_sha = str(checkpoint_manifest["stock_stats_sha256"])
    horizon_seed0 = [
        record
        for record in checkpoint_manifest["checkpoints"]
        if record["arm"] == "jepa_horizon" and int(record["seed"]) == 0
    ]
    if len(horizon_seed0) != 1:
        raise F16IntegrityError("canonical horizon seed-0 checkpoint is not unique")
    stock_stats_path = repo_root / horizon_seed0[0]["path"]
    if sha256_file(stock_stats_path) != horizon_seed0[0]["sha256"]:
        raise F16IntegrityError("canonical stock-stats checkpoint drift")
    stock_stats = _load_stock_stats(stock_stats_path, stock_stats_sha)

    target_cache = prepare_target_cache(
        repo_root, output_root, bundle_root, bundle_manifest, cohort_manifest
    )
    label_record = cohort_manifest["label_budgets"][budget]
    label_path = repo_root / label_record["path"]
    if sha256_file(label_path) != label_record["sha256"]:
        raise F16IntegrityError("F16 label manifest drift")
    train_rows = pd.read_parquet(label_path)
    validation_record = cohort_manifest["cohorts"]["validation"]
    validation_path = repo_root / validation_record["path"]
    if sha256_file(validation_path) != validation_record["sha256"]:
        raise F16IntegrityError("F16 validation manifest drift")
    validation_rows = pd.read_parquet(validation_path)
    y_train_raw = np.load(
        repo_root / target_cache["arrays"][budget]["path"], allow_pickle=False
    )
    y_validation_raw = np.load(
        repo_root / target_cache["arrays"]["validation"]["path"], allow_pickle=False
    )
    y_train, y_validation, target_mean, target_std = standardize_targets(
        y_train_raw, y_validation_raw
    )
    if not np.isfinite(y_train).all() or not np.isfinite(y_validation).all():
        raise F16IntegrityError("non-finite standardized F16 target")

    row_identity = {
        "cohort_manifest_sha256": sha256_file(output_root / "f16_cohort_manifest.json"),
        "label_manifest_sha256": label_record["sha256"],
        "label_row_key_sequence_sha256": label_record["row_key_sequence_sha256"],
        "label_endpoint_index_sha256": label_record["endpoint_index_sha256"],
        "validation_manifest_sha256": validation_record["sha256"],
        "validation_row_key_sequence_sha256": validation_record[
            "row_key_sequence_sha256"
        ],
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
    }

    raw = np.load(dataset_path, allow_pickle=False)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    train_endpoints = train_rows["endpoint_index"].to_numpy(dtype=np.int64)
    validation_endpoints = validation_rows["endpoint_index"].to_numpy(dtype=np.int64)
    train_dataset = SupervisedGridDataset(
        book, mid_z, stock_ids, train_endpoints, stock_stats, y_train, config.K
    )
    validation_dataset = SupervisedGridDataset(
        book,
        mid_z,
        stock_ids,
        validation_endpoints,
        stock_stats,
        y_validation,
        config.K,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    encoder_config, readout_config = _model_configs(config)
    model = SupervisedGrid(encoder_config, readout_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda update: learning_rate_multiplier(update, config)
    )

    run_dir = output_root / "runs" / budget / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("source_fingerprint") != source_inventory["fingerprint"]:
            raise F16IntegrityError("completed F16 cell source drift")
        if complete.get("row_identity") != row_identity:
            raise F16IntegrityError("completed F16 cell row identity drift")
        for record in complete["checkpoints"].values():
            path = repo_root / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise F16IntegrityError("completed F16 cell checkpoint drift")
        return complete

    sampler_template = DeterministicPassBatchSampler(
        len(train_dataset),
        config.batch_size,
        seed,
        0,
        config.maximum_updates,
    )
    epoch20_update = sampler_template.steps_per_pass * config.encoder_pass_sensitivity
    early_stop_eligible_update = max(config.minimum_updates, epoch20_update)
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.parquet"
    history: list[dict[str, Any]] = []
    global_update = 0
    best_mse = float("inf")
    best_update = 0
    no_improvement_checks = 0
    if last_path.is_file():
        checkpoint = _load_resume_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            budget=budget,
            seed=seed,
            row_identity=row_identity,
            source_inventory=source_inventory,
        )
        global_update = int(checkpoint["global_update"])
        best_mse = float(checkpoint["best_validation_mse"])
        best_update = int(checkpoint["best_update"])
        no_improvement_checks = int(checkpoint["no_improvement_checks"])
        if history_path.is_file():
            history = pd.read_parquet(history_path).to_dict("records")

    sampler = DeterministicPassBatchSampler(
        len(train_dataset),
        config.batch_size,
        seed,
        global_update,
        config.maximum_updates,
    )
    print(
        f"F16 {budget}/seed{seed}: train={len(train_dataset):,}, "
        f"validation={len(validation_dataset):,}, steps/pass={sampler.steps_per_pass}, "
        f"epoch20_update={epoch20_update:,}, max_updates={config.maximum_updates:,}, "
        f"device={device}",
        flush=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    process = psutil.Process()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    peak_rss = process.memory_info().rss
    rolling_loss_sum = 0.0
    rolling_rows = 0
    stop_reason = "maximum_updates"
    selected_payload: Mapping[str, Any] | None = None

    def evaluate_and_checkpoint(reason: str, counts_for_patience: bool) -> Mapping[str, Any]:
        nonlocal best_mse, best_update, no_improvement_checks, rolling_loss_sum, rolling_rows
        metrics = _validate_model(model, validation_loader, device, EXPECTED_TARGETS)
        improved = bool(metrics["mse"] < best_mse - config.minimum_improvement_mse)
        if best_mse == float("inf"):
            improved = True
        if improved:
            best_mse = float(metrics["mse"])
            best_update = global_update
            no_improvement_checks = 0
        elif counts_for_patience:
            no_improvement_checks += 1
        record = {
            "global_update": global_update,
            "completed_passes": global_update / sampler.steps_per_pass,
            "event": reason,
            "counts_for_patience": counts_for_patience,
            "improved": improved,
            "no_improvement_checks": no_improvement_checks,
            "learning_rate_next_update": float(optimizer.param_groups[0]["lr"]),
            "train_mse_since_prior_check": (
                rolling_loss_sum / rolling_rows if rolling_rows else np.nan
            ),
            **{f"validation_{key}": value for key, value in metrics.items()},
            "wall_seconds": time.perf_counter() - start_wall,
        }
        history.append(record)
        print(
            f"update={global_update:>5d} event={reason} "
            f"train_mse={record['train_mse_since_prior_check']:.6f} "
            f"val_mse={metrics['mse']:.6f} best={best_mse:.6f}@{best_update} "
            f"patience={no_improvement_checks}/{config.patience_checks}",
            flush=True,
        )
        rolling_loss_sum = 0.0
        rolling_rows = 0
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            budget=budget,
            seed=seed,
            global_update=global_update,
            sampler=sampler,
            best_mse=best_mse,
            best_update=best_update,
            no_improvement_checks=no_improvement_checks,
            target_mean=target_mean,
            target_std=target_std,
            stock_stats=stock_stats,
            row_identity=row_identity,
            source_inventory=source_inventory,
            validation_metrics=metrics,
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, run_dir / "best.pt")
        if global_update == 0:
            _atomic_torch_save(payload, run_dir / "initial.pt")
        if global_update == epoch20_update:
            _atomic_torch_save(payload, run_dir / "epoch20.pt")
        pd.DataFrame(history).to_parquet(history_path, index=False)
        return payload

    if global_update == 0:
        selected_payload = evaluate_and_checkpoint("initial", counts_for_patience=False)

    model.train()
    for book_batch, target_batch, stock_batch in train_loader:
        if global_update >= config.maximum_updates:
            break
        book_batch = book_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)
        stock_batch = stock_batch.to(device, non_blocking=True)
        if not torch.isfinite(book_batch).all() or not torch.isfinite(target_batch).all():
            raise F16IntegrityError("non-finite F16 train input or target")
        prediction = model(book_batch, stock_batch)
        loss = F.mse_loss(prediction, target_batch)
        if not torch.isfinite(loss):
            raise F16IntegrityError("non-finite F16 training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if not torch.isfinite(grad_norm):
            raise F16IntegrityError("non-finite F16 gradient norm")
        if float(grad_norm) > config.gradient_explosion_threshold:
            raise F16IntegrityError("F16 gradient explosion threshold exceeded")
        optimizer.step()
        scheduler.step()
        global_update += 1
        rolling_loss_sum += float(loss.item()) * len(book_batch)
        rolling_rows += len(book_batch)
        peak_rss = max(peak_rss, process.memory_info().rss)

        cadence = global_update % config.validation_cadence_updates == 0
        sensitivity = global_update == epoch20_update
        if cadence or sensitivity or global_update == config.maximum_updates:
            event = "epoch20_sensitivity" if sensitivity and not cadence else "scheduled"
            selected_payload = evaluate_and_checkpoint(event, counts_for_patience=cadence)
            model.train()
            if (
                cadence
                and global_update >= early_stop_eligible_update
                and no_improvement_checks >= config.patience_checks
            ):
                stop_reason = "early_stopping"
                break

    if selected_payload is None:
        raise F16IntegrityError("F16 training produced no checkpoint payload")
    if not (run_dir / "epoch20.pt").is_file():
        raise F16IntegrityError("F16 run ended without epoch-20 sensitivity checkpoint")
    best_path = run_dir / "best.pt"
    epoch20_path = run_dir / "epoch20.pt"
    if not best_path.is_file():
        raise F16IntegrityError("F16 run ended without best checkpoint")

    # Reload the selected checkpoint and reproduce its stored validation metric.
    try:
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    except TypeError:
        best_checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    reloaded = _validate_model(model, validation_loader, device, EXPECTED_TARGETS)
    stored_mse = float(best_checkpoint["validation_metrics"]["mse"])
    if not math.isclose(reloaded["mse"], stored_mse, rel_tol=0.0, abs_tol=1e-8):
        raise F16IntegrityError(
            f"selected checkpoint reload MSE mismatch: {reloaded['mse']} vs {stored_mse}"
        )

    wall_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu
    max_rss_bytes = max(
        peak_rss,
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    )
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    checkpoints = {
        "best": _checkpoint_record(best_path, repo_root, best_checkpoint),
        "epoch20": _checkpoint_record(
            epoch20_path,
            repo_root,
            torch.load(epoch20_path, map_location="cpu", weights_only=False),
        ),
        "last": _checkpoint_record(last_path, repo_root, selected_payload),
        "initial": _checkpoint_record(
            run_dir / "initial.pt",
            repo_root,
            torch.load(run_dir / "initial.pt", map_location="cpu", weights_only=False),
        ),
    }
    complete = {
        "schema_name": "thesis.experiment01.f16_training_cell",
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": "complete",
        "budget": budget,
        "encoder_seed": seed,
        "stop_reason": stop_reason,
        "final_update": global_update,
        "best_update": int(best_checkpoint["global_update"]),
        "best_validation_mse": stored_mse,
        "epoch20_update": epoch20_update,
        "early_stop_eligible_update": early_stop_eligible_update,
        "steps_per_pass": sampler.steps_per_pass,
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "target_mean_sha256": sha256_array(np.asarray(target_mean, dtype=np.float32)),
        "target_std_sha256": sha256_array(np.asarray(target_std, dtype=np.float32)),
        "row_identity": row_identity,
        "source_fingerprint": source_inventory["fingerprint"],
        "source_inventory": source_inventory,
        "training_config": asdict(config),
        "checkpoints": checkpoints,
        "history": {
            "path": _relative(history_path, repo_root),
            "sha256": sha256_file(history_path),
            "size_bytes": history_path.stat().st_size,
            "rows": len(history),
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "peak_ram_bytes": max_rss_bytes,
            "peak_vram_bytes": peak_vram,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
            "rocm_version": getattr(torch.version, "hip", None),
        },
        "selected_checkpoint_reload_validation_mse": reloaded["mse"],
        "test_accessed": False,
        "failures": [],
    }
    atomic_write_json(complete_path, complete)
    # Output size is recorded after complete.json exists; it is informational and
    # deliberately excludes no scientific input.
    complete["runtime"]["run_output_bytes"] = _output_size(run_dir)
    atomic_write_json(complete_path, complete)
    return complete


def write_failure(
    repo_root: Path,
    output_root: Path,
    budget: str,
    seed: int,
    exc: BaseException,
) -> None:
    run_dir = output_root / "runs" / budget / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": "thesis.experiment01.f16_training_failure",
        "schema_version": 1,
        "status": "failed",
        "budget": budget,
        "encoder_seed": seed,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "test_accessed": False,
    }
    atomic_write_json(run_dir / "failure.json", payload)

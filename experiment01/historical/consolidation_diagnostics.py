#!/usr/bin/env python3
"""Read-only diagnostics for the consolidation package.

This script never changes checkpoints.  It extracts final-normalization scales,
supervised attention weights, and a one-seed temporal-position screen from the
canonical v2 split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment01.historical.analysis_artifacts import (
    atomic_write_json,
    load_split,
    sha256_file,
)
from experiment01.historical.consolidation_geometry import (
    ladder_from_stats,
    linear_stats,
    pca_from_stats,
)
from experiment01.historical.extract_readouts_multiseed import (
    HorizonJEPAEncoderConfig,
    RawWindowDataset,
    ReadoutConfig,
    SupervisedGrid,
    load_encoder,
    robust_torch_load,
    to_numpy_stats,
)
from experiment01.historical.ladder_accessibility import dir_indices


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {name!r} requested but CUDA/ROCm is unavailable"
        )
    return requested


def _gamma_from_checkpoint(arm: str, checkpoint: dict) -> np.ndarray:
    if arm == "supervised":
        state = checkpoint["model_state_dict"]
        candidates = [
            value
            for key, value in state.items()
            if key == "encoder.final_norm.weight"
        ]
    else:
        state = checkpoint.get(
            "online_state_dict", checkpoint.get("encoder_state_dict")
        )
        if state is None:
            raise ValueError(f"{arm}: missing online encoder state")
        candidates = [
            value for key, value in state.items() if key == "final_norm.weight"
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"{arm}: expected one final_norm.weight, got {len(candidates)}"
        )
    return candidates[0].detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def supervised_attention(
    checkpoint_path: Path,
    dataset: RawWindowDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    checkpoint = robust_torch_load(checkpoint_path, device)
    model = SupervisedGrid(
        HorizonJEPAEncoderConfig.from_dict(checkpoint["enc_cfg"]),
        ReadoutConfig(**checkpoint["readout_cfg"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    weight_sum = None
    entropy_sum = 0.0
    entropy_square_sum = 0.0
    n_rows = 0
    for book, stock_ids in loader:
        book = book.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        grid = model.encoder(book, stock_ids)
        flat = grid.reshape(grid.shape[0], -1, grid.shape[-1])
        keys = model.readout.k(flat)
        scores = (
            keys * model.readout.q.view(1, 1, -1)
        ).sum(dim=-1) / math.sqrt(keys.shape[-1])
        weights = torch.softmax(scores, dim=1)
        entropy = -(
            weights * torch.log(torch.clamp(weights, min=1e-12))
        ).sum(dim=1) / math.log(weights.shape[1])
        batch_sum = weights.sum(dim=0).detach().cpu().numpy()
        weight_sum = batch_sum if weight_sum is None else weight_sum + batch_sum
        entropy_sum += float(entropy.sum().item())
        entropy_square_sum += float((entropy * entropy).sum().item())
        n_rows += int(weights.shape[0])
    if weight_sum is None or n_rows == 0:
        raise RuntimeError("attention diagnostic received no rows")
    mean_weights = weight_sum / n_rows
    mean_entropy = entropy_sum / n_rows
    entropy_variance = max(
        entropy_square_sum / n_rows - mean_entropy * mean_entropy, 0.0
    )
    return mean_weights, {
        "n_rows": n_rows,
        "normalized_entropy_mean": mean_entropy,
        "normalized_entropy_std": math.sqrt(entropy_variance),
    }


@torch.no_grad()
def _extract_temporal_memmap(
    encode,
    dataset: RawWindowDataset,
    destination: Path,
    K: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.memmap:
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(K, len(dataset), 512),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    start = 0
    for book, stock_ids in loader:
        book = book.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        grid = encode(book, stock_ids)
        concat = grid.reshape(grid.shape[0], K, 512)
        values = concat.permute(1, 0, 2).float().cpu().numpy()
        output[:, start : start + len(book)] = values
        start += len(book)
    if start != len(dataset):
        raise RuntimeError(f"temporal extraction wrote {start}/{len(dataset)} rows")
    output.flush()
    return output


def temporal_screen(
    arm: str,
    checkpoint_path: Path,
    train_dataset: RawWindowDataset,
    val_dataset: RawWindowDataset,
    y_train: np.ndarray,
    y_val: np.ndarray,
    K: int,
    m: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict]:
    encode, _ = load_encoder(arm, str(checkpoint_path), device)
    directional = dir_indices()["dir_indep"]
    with tempfile.TemporaryDirectory(
        prefix=f"temporal-{arm}-", dir="/tmp"
    ) as temporary:
        root = Path(temporary)
        train_map = _extract_temporal_memmap(
            encode,
            train_dataset,
            root / "train.npy",
            K,
            batch_size,
            num_workers,
            device,
        )
        val_map = _extract_temporal_memmap(
            encode,
            val_dataset,
            root / "val.npy",
            K,
            batch_size,
            num_workers,
            device,
        )
        rows = []
        for timestep in range(K):
            stats = linear_stats(
                train_map[timestep], y_train, val_map[timestep], y_val
            )
            _, vectors = pca_from_stats(stats)
            curve = ladder_from_stats(stats, vectors, [m, 512])
            low = float(np.mean(curve[m][directional]))
            full = float(np.mean(curve[512][directional]))
            rows.append(
                {
                    "arm": arm,
                    "timestep": timestep + 1,
                    "m": m,
                    "dimension": 512,
                    "m_fraction": m / 512,
                    "r2_m": low,
                    "r2_full": full,
                    "r2_fraction": (
                        low / full if full >= 0.01 else float("nan")
                    ),
                    "denominator_reliable": full >= 0.01,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only consolidation diagnostics"
    )
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--temporal_seed", type=int, default=0)
    parser.add_argument("--temporal_m", type=int, default=16)
    parser.add_argument("--skip_attention", action="store_true")
    parser.add_argument("--skip_temporal", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else in_dir / "analysis" / "diagnostics"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (in_dir / "analysis_manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise RuntimeError("stage-1 manifest is not complete")
    K = int(manifest["protocol"]["K"])
    if K != 20:
        raise RuntimeError(f"temporal diagnostic requires K=20, got {K}")
    split = load_split(
        in_dir / "split.npz",
        expected_dataset_sha256=manifest["dataset"]["sha256"],
    )
    checkpoints = manifest["requested_checkpoints"]

    gamma_rows = []
    for tag, record in sorted(checkpoints.items()):
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"{tag}: checkpoint SHA-256 mismatch")
        checkpoint = robust_torch_load(path, torch.device("cpu"))
        gamma = np.abs(_gamma_from_checkpoint(record["arm"], checkpoint))
        gamma_rows.append(
            {
                "arm": record["arm"],
                "seed": record["seed"],
                "epoch": record["epoch"],
                "gamma_abs_min": float(gamma.min()),
                "gamma_abs_max": float(gamma.max()),
                "gamma_abs_mean": float(gamma.mean()),
                "gamma_abs_std": float(gamma.std()),
            }
        )
    _write_csv(
        out_dir / "final_norm_gamma.csv",
        gamma_rows,
        [
            "arm", "seed", "epoch", "gamma_abs_min", "gamma_abs_max",
            "gamma_abs_mean", "gamma_abs_std",
        ],
    )

    if args.skip_attention and args.skip_temporal:
        print(f"Saved final_norm diagnostics -> {out_dir}")
        return
    device = _device(args.device)
    dataset_path = Path(manifest["dataset"]["path"])
    if sha256_file(dataset_path) != manifest["dataset"]["sha256"]:
        raise RuntimeError("dataset SHA-256 mismatch")
    with np.load(dataset_path, allow_pickle=False) as raw:
        book = raw["book"]
        mid_z = raw["mid_z"]
        stock_ids = raw["stock_ids"]

        if not args.skip_attention:
            attention_rows = []
            attention_summary = []
            for tag, record in sorted(checkpoints.items()):
                if record["arm"] != "supervised":
                    continue
                checkpoint = robust_torch_load(
                    Path(record["path"]), torch.device("cpu")
                )
                stats = to_numpy_stats(checkpoint["stock_stats"])
                dataset = RawWindowDataset(
                    book, mid_z, stock_ids, split.val_t, stats, K
                )
                weights, summary = supervised_attention(
                    Path(record["path"]),
                    dataset,
                    args.batch_size,
                    args.num_workers,
                    device,
                )
                matrix = weights.reshape(K, 4)
                for timestep in range(K):
                    for token in range(4):
                        attention_rows.append(
                            {
                                "seed": record["seed"],
                                "epoch": record["epoch"],
                                "timestep": timestep + 1,
                                "token": token + 1,
                                "mean_weight": float(matrix[timestep, token]),
                            }
                        )
                attention_summary.append(
                    {
                        "seed": record["seed"],
                        "epoch": record["epoch"],
                        **summary,
                        "first_timestep_weight": float(matrix[0].sum()),
                        "last_timestep_weight": float(matrix[-1].sum()),
                        "max_position_weight": float(matrix.max()),
                        "min_position_weight": float(matrix.min()),
                    }
                )
            _write_csv(
                out_dir / "supervised_attention_positions.csv",
                attention_rows,
                ["seed", "epoch", "timestep", "token", "mean_weight"],
            )
            _write_csv(
                out_dir / "supervised_attention_summary.csv",
                attention_summary,
                [
                    "seed", "epoch", "n_rows", "normalized_entropy_mean",
                    "normalized_entropy_std", "first_timestep_weight",
                    "last_timestep_weight", "max_position_weight",
                    "min_position_weight",
                ],
            )

        temporal_rows = []
        if not args.skip_temporal:
            with np.load(
                in_dir / "targets_shared.npz", allow_pickle=False
            ) as targets:
                y_train = targets["y_train_raw"].astype(np.float64)
                y_val = targets["y_val_raw"].astype(np.float64)
            for arm in ("jepa_horizon", "jepa_masked", "supervised"):
                matches = [
                    record
                    for record in checkpoints.values()
                    if record["arm"] == arm
                    and int(record["seed"]) == args.temporal_seed
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"{arm}: expected one seed {args.temporal_seed} checkpoint"
                    )
                record = matches[0]
                checkpoint = robust_torch_load(
                    Path(record["path"]), torch.device("cpu")
                )
                stats = to_numpy_stats(checkpoint["stock_stats"])
                train_dataset = RawWindowDataset(
                    book, mid_z, stock_ids, split.train_t, stats, K
                )
                val_dataset = RawWindowDataset(
                    book, mid_z, stock_ids, split.val_t, stats, K
                )
                temporal_rows.extend(
                    temporal_screen(
                        arm,
                        Path(record["path"]),
                        train_dataset,
                        val_dataset,
                        y_train,
                        y_val,
                        K,
                        args.temporal_m,
                        args.batch_size,
                        args.num_workers,
                        device,
                    )
                )
            _write_csv(
                out_dir / "temporal_screen.csv",
                temporal_rows,
                [
                    "arm", "timestep", "m", "dimension", "m_fraction",
                    "r2_m", "r2_full", "r2_fraction",
                    "denominator_reliable",
                ],
            )

    atomic_write_json(
        out_dir / "diagnostics_manifest.json",
        {
            "schema": {
                "name": "consolidation_diagnostics_manifest",
                "version": 1,
            },
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": sha256_file(
                in_dir / "analysis_manifest.json"
            ),
            "source_sha256": sha256_file(Path(__file__)),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "torch_version": torch.__version__,
            "rocm_version": getattr(torch.version, "hip", None),
            "K": K,
            "attention_all_supervised_seeds": not args.skip_attention,
            "temporal_seed": args.temporal_seed,
            "temporal_m": args.temporal_m,
            "temporal_all_20_positions": not args.skip_temporal,
            "outputs": {
                path.name: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(out_dir.glob("*.csv"))
            },
        },
    )
    print(f"Saved consolidation diagnostics -> {out_dir}")


if __name__ == "__main__":
    main()

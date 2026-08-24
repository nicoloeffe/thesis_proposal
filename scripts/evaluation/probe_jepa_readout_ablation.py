#!/usr/bin/env python3
"""
probe_jepa_readout_ablation.py

Readout ablation on one frozen Horizon JEPA grid representation.

This script intentionally keeps the dataset/split/target protocol aligned with
scripts/evaluation/probe_jepa_grid_readout.py and changes only the readout put
on top of the frozen JEPA encoder. The goal is to test whether the low R2 of the
current single-query AttnPoolReadout comes from the readout bottleneck rather
than missing information in the JEPA grid.

Example
-------
python -m scripts.evaluation.probe_jepa_readout_ablation \\
  --dataset data/lobench_processed.npz \\
  --jepa_ckpt checkpoints/jepa_horizon/v1_500k/epoch_012.pt \\
  --out_dir validation/jepa_readout_ablation/v1_100k \\
  --max_train_samples 100000 \\
  --max_val_samples 50000 \\
  --epochs 80 \\
  --batch_size 512 \\
  --probe_batch_size 1024 \\
  --num_workers 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent, _THIS.parent.parent.parent,
           _THIS.parent.parent.parent.parent, Path.cwd(), *Path.cwd().parents]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.historical.train_jepa_horizon import (  # type: ignore
    HorizonJEPAEncoder, HorizonJEPAEncoderConfig,
)
from training.train_tokenizer_t import (  # type: ignore
    compute_valid_endpoints, normalize_book_window, grouped_split_by_stock_day,
    derive_raw_features_array,
)
from training.historical.train_supervised_grid import (  # type: ignore
    AttnPoolReadout, ReadoutConfig, build_targets, standardize_targets,
    r2_per_target, FUTURE_HORIZONS, VOL_HORIZONS,
)


# =============================================================================
# Loading / extraction
# =============================================================================

def robust_torch_load(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_jepa_encoder(ckpt_path: str, device: torch.device) -> Tuple[HorizonJEPAEncoder, Dict, Dict]:
    ckpt = robust_torch_load(ckpt_path, device)
    enc_cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    enc = HorizonJEPAEncoder(enc_cfg).to(device)
    # Resolve encoder weights across checkpoint formats:
    #   JEPA online            -> "online_state_dict"
    #   encoder-only probe     -> "encoder_state_dict"
    #   supervised_grid model  -> "model_state_dict" with "encoder."-prefixed keys
    if "online_state_dict" in ckpt:
        enc_sd = ckpt["online_state_dict"]
    elif "encoder_state_dict" in ckpt:
        enc_sd = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        enc_sd = {k.removeprefix("encoder."): v
                  for k, v in ckpt["model_state_dict"].items()
                  if k.startswith("encoder.")}
        if not enc_sd:
            raise KeyError("model_state_dict senza chiavi 'encoder.'")
    else:
        raise KeyError("nessun peso encoder trovato nel checkpoint")
    enc.load_state_dict(enc_sd)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    stock_stats = {
        k: np.asarray(v, dtype=np.float32) if not isinstance(v, (int, float)) else v
        for k, v in ckpt["stock_stats"].items()
    }
    return enc, ckpt, stock_stats


class WindowDataset(Dataset):
    def __init__(
        self,
        book: np.ndarray,
        mid_z: np.ndarray,
        stock_ids: np.ndarray,
        valid_t: np.ndarray,
        stock_stats: Dict,
        K: int,
    ):
        self.book = book
        self.mid_z = mid_z
        self.stock_ids = stock_ids
        self.valid_t = valid_t
        self.stock_stats = stock_stats
        self.K = K

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, idx: int):
        t = int(self.valid_t[idx])
        s = int(self.stock_ids[t])
        K = self.K
        w = normalize_book_window(
            self.book[t - K + 1:t + 1],
            self.mid_z[t - K + 1:t + 1],
            s,
            self.stock_stats,
        )
        return torch.from_numpy(w).float(), torch.tensor(s, dtype=torch.long)


@torch.no_grad()
def extract_grids(
    enc: HorizonJEPAEncoder,
    book: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    valid_t: np.ndarray,
    stock_stats: Dict,
    K: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    label: str,
) -> np.ndarray:
    ds = WindowDataset(book, mid_z, stock_ids, valid_t, stock_stats, K)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    out = []
    n = 0
    t0 = time.time()
    for W, sid in dl:
        W = W.to(device, non_blocking=True)
        sid = sid.to(device, non_blocking=True)
        g = enc(W, sid)                                    # (B,K,S,D)
        out.append(g.detach().cpu().numpy().astype(np.float32))
        n += W.shape[0]
        if n and n % (batch_size * 20) == 0:
            print(f"    {label}: extracted {n:,} in {time.time() - t0:.1f}s")
    grids = np.concatenate(out, axis=0)
    print(f"  {label}: grid {grids.shape} extracted in {time.time() - t0:.1f}s")
    return grids


# =============================================================================
# Datasets
# =============================================================================

class ArrayTargetDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.x = x
        self.y = y.astype(np.float32, copy=False)
        self.mean = None if mean is None else mean.astype(np.float32, copy=False)
        self.std = None if std is None else std.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        x = self.x[idx].astype(np.float32, copy=False)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.from_numpy(self.y[idx])


def feature_standardizer(x_train: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)
    return mean, std


# =============================================================================
# Readouts
# =============================================================================

class LinearBottleneck(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, z_dim)
        self.head = nn.Linear(z_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.proj(x))


class MLPReadout(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiQueryAttentionReadout(nn.Module):
    """Multi-query attention over all K*S grid tokens, then linear head."""
    def __init__(self, d_model: int, n_queries: int, out_dim: int, dropout: float):
        super().__init__()
        self.n_queries = n_queries
        self.q = nn.Parameter(torch.randn(n_queries, d_model) / math.sqrt(d_model))
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(n_queries * d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_queries * d_model, out_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        B = grid.shape[0]
        x = grid.reshape(B, -1, grid.shape[-1])            # (B,T,D)
        k = self.k(x)
        v = self.v(x)
        scores = torch.einsum("btd,qd->btq", k, self.q) / math.sqrt(k.shape[-1])
        w = torch.softmax(scores, dim=1)                   # (B,T,Q)
        r = torch.einsum("btq,btd->bqd", w, v).reshape(B, -1)
        return self.head(self.drop(self.norm(r)))


# =============================================================================
# Training / metrics
# =============================================================================

def summarize_r2(names: List[str], r2: np.ndarray) -> Dict[str, float]:
    future_idx = [i for i, n in enumerate(names) if "realized_vol" not in n]
    vol_idx = [i for i, n in enumerate(names) if "realized_vol" in n]
    return {
        "mean_all": float(np.mean(r2)),
        "median_all": float(np.median(r2)),
        "mean_future_delta": float(np.mean(r2[future_idx])) if future_idx else float("nan"),
        "mean_realized_vol": float(np.mean(r2[vol_idx])) if vol_idx else float("nan"),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
    train: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(train)
    total_loss, n_total = 0.0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = x.shape[0]
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = F.mse_loss(pred, y)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.item()) * B
        n_total += B
        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
    return (
        total_loss / max(n_total, 1),
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_true, axis=0),
    )


def train_probe(
    name: str,
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    target_names: List[str],
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict:
    dl_tr = DataLoader(
        train_ds,
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=args.loader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.loader_workers > 0,
        drop_last=False,
    )
    dl_va = DataLoader(
        val_ds,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.loader_workers > 0,
        drop_last=False,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * args.eta_min_frac
    )

    best = {
        "epoch": 0,
        "val_mse": float("inf"),
        "state_dict": None,
        "r2": None,
        "summary": None,
        "train_mse": None,
    }
    bad = 0
    history = []
    t0 = time.time()
    print(f"\nTraining {name}...")
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr_loss, _, _ = run_epoch(model, dl_tr, optimizer, device, args.grad_clip, train=True)
        va_loss, va_pred, va_true = run_epoch(model, dl_va, None, device, args.grad_clip, train=False)
        scheduler.step()
        r2 = r2_per_target(va_true, va_pred)
        summary = summarize_r2(target_names, r2)
        history.append({
            "epoch": epoch,
            "train_mse": float(tr_loss),
            "val_mse": float(va_loss),
            "val_r2": summary,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        improved = va_loss < best["val_mse"] - args.min_delta
        if improved:
            best = {
                "epoch": epoch,
                "val_mse": float(va_loss),
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "r2": r2.astype(float).tolist(),
                "summary": summary,
                "train_mse": float(tr_loss),
            }
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs or improved:
            print(
                f"  ep {epoch:03d}/{args.epochs} [{time.time() - ep_t0:.1f}s] "
                f"train_mse={tr_loss:.4f} val_mse={va_loss:.4f} "
                f"R2 all={summary['mean_all']:.4f} "
                f"future={summary['mean_future_delta']:.4f} "
                f"vol={summary['mean_realized_vol']:.4f} "
                f"best_ep={best['epoch']}"
            )
        if bad >= args.patience:
            print(f"  early stopping at epoch {epoch} (patience={args.patience})")
            break

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    best["elapsed_sec"] = float(time.time() - t0)
    result = {
        "readout": name,
        "best_epoch": int(best["epoch"]),
        "best_val_mse": float(best["val_mse"]),
        "best_train_mse": None if best["train_mse"] is None else float(best["train_mse"]),
        "summary": best["summary"],
        "r2": best["r2"],
        "history": history,
        "elapsed_sec": best["elapsed_sec"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"metrics_{safe_name(name)}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def print_final_table(results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 100)
    print("READOUT ABLATION SUMMARY")
    print("=" * 100)
    print(
        f"{'readout':36s} {'mean_all':>10s} {'future':>10s} "
        f"{'vol':>10s} {'best_ep':>8s} {'val_mse':>10s}"
    )
    print("-" * 100)
    for name, res in results.items():
        s = res["summary"]
        print(
            f"{name:36s} {s['mean_all']:10.4f} {s['mean_future_delta']:10.4f} "
            f"{s['mean_realized_vol']:10.4f} {res['best_epoch']:8d} "
            f"{res['best_val_mse']:10.4f}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Readout ablation on frozen JEPA grids")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--jepa_ckpt", type=str, default="checkpoints/jepa_horizon/v1_500k/epoch_012.pt")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--max_train_samples", type=int, default=100000)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2, help="Workers for JEPA grid extraction")
    p.add_argument("--loader_workers", type=int, default=0, help="Workers for probe DataLoaders")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for JEPA extraction")
    p.add_argument("--probe_batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min_delta", type=float, default=1e-7)
    p.add_argument("--eta_min_frac", type=float, default=0.03)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--flat_bottleneck_dim", type=int, default=256)
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--readout_dropout", type=float, default=0.0)
    p.add_argument("--multi_query", type=int, default=4)
    p.add_argument("--run_full_mlp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run_multi_query", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--standardize_vector_features", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("JEPA READOUT ABLATION — same frozen grid, same split, same targets")
    print("=" * 100)
    print(f"dataset       : {args.dataset}")
    print(f"jepa_ckpt     : {args.jepa_ckpt}")
    print(f"out_dir       : {args.out_dir}")
    print(f"device        : {device}")

    print("\n[1/6] Loading frozen JEPA encoder...")
    enc, ckpt, stock_stats = load_jepa_encoder(args.jepa_ckpt, device)
    K, S, D = int(enc.cfg.K), int(enc.cfg.S), int(enc.cfg.d_model)
    print(f"  epoch={ckpt.get('epoch', 'N/A')}  K={K} S={S} d_model={D}")

    print("\n[2/6] Loading dataset & reproducing grid-probe split...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids_arr = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    min_spread_per_stock = raw["min_spread_z_per_stock"].astype(np.float32)
    n_stocks = int(min_spread_per_stock.shape[0])
    raw_feat, _ = derive_raw_features_array(book, mid_z, stock_ids_arr, n_stocks)

    bid_v, ask_v = book[:, 0, :, 1], book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & \
               (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    max_h = max(max(FUTURE_HORIZONS), max(VOL_HORIZONS))
    valid_t = compute_valid_endpoints(stock_ids_arr, day_ids, K, max_h, vol_mask)

    train_pos, val_pos = grouped_split_by_stock_day(
        stock_ids_arr, day_ids, valid_t, args.val_frac, args.seed
    )
    rng = np.random.default_rng(args.seed)
    if args.max_train_samples > 0 and len(train_pos) > args.max_train_samples:
        train_pos = np.sort(rng.choice(train_pos, args.max_train_samples, replace=False))
    if args.max_val_samples > 0 and len(val_pos) > args.max_val_samples:
        val_pos = np.sort(rng.choice(val_pos, args.max_val_samples, replace=False))
    t_train, t_val = valid_t[train_pos], valid_t[val_pos]
    print(f"  valid={len(valid_t):,} train={len(t_train):,} val={len(t_val):,}")

    print("\n[3/6] Building the same 22 targets and train-only standardization...")
    y_train_raw, target_names = build_targets(
        book, mid_z, stock_ids_arr, t_train, raw_feat, min_spread_per_stock
    )
    y_val_raw, _ = build_targets(
        book, mid_z, stock_ids_arr, t_val, raw_feat, min_spread_per_stock
    )
    y_train, y_val, target_mean, target_std = standardize_targets(y_train_raw, y_val_raw)
    print(f"  y_train={y_train.shape} y_val={y_val.shape}")

    print("\n[4/6] Extracting frozen JEPA grids once...")
    g_train = extract_grids(
        enc, book, mid_z, stock_ids_arr, t_train, stock_stats, K,
        device, args.batch_size, args.num_workers, "train"
    )
    g_val = extract_grids(
        enc, book, mid_z, stock_ids_arr, t_val, stock_stats, K,
        device, args.batch_size, args.num_workers, "val"
    )

    print("\n[5/6] Preparing readout feature views...")
    last_train = g_train[:, -1, :, :]
    last_val = g_val[:, -1, :, :]
    last_concat_train = np.ascontiguousarray(last_train.reshape(last_train.shape[0], -1))
    last_concat_val = np.ascontiguousarray(last_val.reshape(last_val.shape[0], -1))
    full_flat_train = g_train.reshape(g_train.shape[0], -1)
    full_flat_val = g_val.reshape(g_val.shape[0], -1)
    print(f"  grid             train={g_train.shape} val={g_val.shape}")
    print(f"  last_tokens      train={last_train.shape} val={last_val.shape}")
    print(f"  last_concat512   train={last_concat_train.shape} val={last_concat_val.shape}")
    print(f"  full_flat        train={full_flat_train.shape} val={full_flat_val.shape}")

    vector_stats = {}
    if args.standardize_vector_features:
        vector_stats["last_concat512"] = feature_standardizer(last_concat_train)
        vector_stats["full_flat"] = feature_standardizer(full_flat_train)
        print("  vector readouts use train-only feature z-score standardization")
    else:
        print("  vector readouts use raw frozen features")

    def vec_ds(xtr, xva, key):
        mean, std = vector_stats.get(key, (None, None))
        return (
            ArrayTargetDataset(xtr, y_train, mean, std),
            ArrayTargetDataset(xva, y_val, mean, std),
        )

    results: Dict[str, Dict] = {}
    print("\n[6/6] Training ablation readouts...")

    # 1. Exact current readout from train_supervised_grid.py.
    rd_cfg = ReadoutConfig(d_model=D, out_dim=y_train.shape[1], dropout=args.readout_dropout)
    train_ds = ArrayTargetDataset(g_train, y_train)
    val_ds = ArrayTargetDataset(g_val, y_val)
    name = "attn_pool_single_query_grid"
    results[name] = train_probe(
        name, AttnPoolReadout(rd_cfg), train_ds, val_ds, target_names, device, args, out_dir
    )

    # 2. Critical test: last_concat512 -> LinearBottleneck z32 -> 22.
    train_ds, val_ds = vec_ds(last_concat_train, last_concat_val, "last_concat512")
    name = f"last_concat512_linear_bottleneck_z{args.z_dim}"
    results[name] = train_probe(
        name,
        LinearBottleneck(last_concat_train.shape[1], args.z_dim, y_train.shape[1]),
        train_ds,
        val_ds,
        target_names,
        device,
        args,
        out_dir,
    )

    # 3. last_concat512 -> MLP.
    name = f"last_concat512_mlp_h{args.mlp_hidden}"
    results[name] = train_probe(
        name,
        MLPReadout(last_concat_train.shape[1], args.mlp_hidden, y_train.shape[1], args.dropout),
        train_ds,
        val_ds,
        target_names,
        device,
        args,
        out_dir,
    )

    # 4. full_flat -> Linear.
    train_ds, val_ds = vec_ds(full_flat_train, full_flat_val, "full_flat")
    name = "full_flat_linear"
    results[name] = train_probe(
        name,
        nn.Linear(full_flat_train.shape[1], y_train.shape[1]),
        train_ds,
        val_ds,
        target_names,
        device,
        args,
        out_dir,
    )

    # 5. full_flat -> bottleneck 256 -> 22.
    name = f"full_flat_linear_bottleneck_z{args.flat_bottleneck_dim}"
    results[name] = train_probe(
        name,
        LinearBottleneck(full_flat_train.shape[1], args.flat_bottleneck_dim, y_train.shape[1]),
        train_ds,
        val_ds,
        target_names,
        device,
        args,
        out_dir,
    )

    # 6. optional full_flat -> MLP.
    if args.run_full_mlp:
        name = f"full_flat_mlp_h{args.mlp_hidden}"
        results[name] = train_probe(
            name,
            MLPReadout(full_flat_train.shape[1], args.mlp_hidden, y_train.shape[1], args.dropout),
            train_ds,
            val_ds,
            target_names,
            device,
            args,
            out_dir,
        )

    # 7. optional multi-query attention over grid.
    if args.run_multi_query:
        train_ds = ArrayTargetDataset(g_train, y_train)
        val_ds = ArrayTargetDataset(g_val, y_val)
        name = f"multi_query_attention_grid_q{args.multi_query}"
        results[name] = train_probe(
            name,
            MultiQueryAttentionReadout(D, args.multi_query, y_train.shape[1], args.readout_dropout),
            train_ds,
            val_ds,
            target_names,
            device,
            args,
            out_dir,
        )

    meta = {
        "args": vars(args),
        "checkpoint": {
            "path": args.jepa_ckpt,
            "epoch": ckpt.get("epoch", None),
            "enc_cfg": enc.cfg.to_dict(),
        },
        "target_names": target_names,
        "n_train": int(len(t_train)),
        "n_val": int(len(t_val)),
        "feature_shapes": {
            "grid": list(g_train.shape[1:]),
            "last_tokens": list(last_train.shape[1:]),
            "last_concat512": [int(last_concat_train.shape[1])],
            "full_flat": [int(full_flat_train.shape[1])],
        },
        "target_standardizer": {
            "mean": target_mean.tolist(),
            "std": target_std.tolist(),
        },
        "vector_feature_standardization": bool(args.standardize_vector_features),
    }
    summary = {"meta": meta, "results": results}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_final_table(results)
    print(f"\nSaved: {out_dir / 'summary.json'}")
    print(f"Saved per-readout metrics_<readout>.json files in: {out_dir}")


if __name__ == "__main__":
    main()

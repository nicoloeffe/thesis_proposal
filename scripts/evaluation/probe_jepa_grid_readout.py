#!/usr/bin/env python3
"""
probe_jepa_grid_readout.py — symmetric readout probe on a frozen JEPA grid encoder.

This is the JEPA-side counterpart of train_supervised_grid.py. It trains the
*same* AttnPoolReadout (attention pool over all K·S tokens, no bottleneck) on a
*frozen* JEPA encoder, against the same 22 observable targets, same split.

Result: a "supervised vs JEPA" R² comparison where the only difference is
whether the encoder was trained by the supervised loss (end-to-end) or by the
JEPA-horizon SSL objective (frozen).

The grounded z32 head (with bottleneck) lives elsewhere — that's the probe of
compressibility, not the main fair comparison.

Usage
-----
python -m scripts.evaluation.probe_jepa_grid_readout \\
  --dataset data/lobench_processed.npz \\
  --jepa_ckpt checkpoints/jepa_horizon/v1_500k/epoch_012.pt \\
  --out_dir validation/jepa_grid_readout/v1 \\
  --epochs 30 --batch_size 512
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
           _THIS.parent.parent.parent.parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.historical.train_jepa_horizon import (  # type: ignore
    HorizonJEPAEncoder, HorizonJEPAEncoderConfig,
)
from training.train_tokenizer_t import (  # type: ignore
    compute_valid_endpoints, normalize_book_window, grouped_split_by_stock_day,
    derive_raw_features_array, compute_future_feature_targets, compute_vol_targets,
)
# Reuse the SAME readout and helpers from the supervised trainer so the two arms
# share one source of truth for the readout architecture and target machinery.
from training.historical.train_supervised_grid import (  # type: ignore
    AttnPoolReadout, ReadoutConfig, build_targets, standardize_targets,
    r2_per_target, summarize_r2, FUTURE_HORIZONS, VOL_HORIZONS,
)


# =============================================================================
#  Load frozen JEPA encoder
# =============================================================================

def load_jepa_encoder(ckpt_path: str, device: torch.device
                      ) -> Tuple[HorizonJEPAEncoder, Dict, Dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    enc_cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    enc = HorizonJEPAEncoder(enc_cfg).to(device)
    state_key = "online_state_dict" if "online_state_dict" in ckpt else "encoder_state_dict"
    enc.load_state_dict(ckpt[state_key])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    stock_stats = {k: np.asarray(v, dtype=np.float32) if not isinstance(v, (int, float)) else v
                   for k, v in ckpt["stock_stats"].items()}
    return enc, ckpt, stock_stats


# =============================================================================
#  Grid extraction (run frozen encoder once, cache grids)
# =============================================================================

class WindowDataset(Dataset):
    def __init__(self, book: np.ndarray, mid_z: np.ndarray, stock_ids: np.ndarray,
                 valid_t: np.ndarray, stock_stats: Dict, K: int):
        self.book = book; self.mid_z = mid_z; self.stock_ids = stock_ids
        self.valid_t = valid_t; self.stock_stats = stock_stats; self.K = K
    def __len__(self): return len(self.valid_t)
    def __getitem__(self, idx):
        t = int(self.valid_t[idx]); s = int(self.stock_ids[t]); K = self.K
        w = normalize_book_window(self.book[t-K+1:t+1], self.mid_z[t-K+1:t+1], s, self.stock_stats)
        return torch.from_numpy(w).float(), torch.tensor(s, dtype=torch.long)


@torch.no_grad()
def extract_grids(enc: HorizonJEPAEncoder, book, mid_z, stock_ids, valid_t,
                  stock_stats, K: int, device: torch.device, batch_size: int = 512,
                  num_workers: int = 4) -> np.ndarray:
    ds = WindowDataset(book, mid_z, stock_ids, valid_t, stock_stats, K)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=(device.type == "cuda"))
    out = []
    t0 = time.time()
    for W, sid in dl:
        W = W.to(device, non_blocking=True); sid = sid.to(device, non_blocking=True)
        g = enc(W, sid)                                  # (B,K,S,d)
        out.append(g.cpu().numpy())
        if (len(out) * batch_size) % (batch_size * 20) == 0:
            print(f"    extracted {sum(x.shape[0] for x in out):,} in {time.time()-t0:.1f}s")
    grids = np.concatenate(out, axis=0)
    return grids


# =============================================================================
#  Grid dataset for readout training
# =============================================================================

class GridTargetDataset(Dataset):
    def __init__(self, grids: np.ndarray, y: np.ndarray):
        self.g = torch.from_numpy(grids).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return self.g.shape[0]
    def __getitem__(self, idx): return self.g[idx], self.y[idx]


# =============================================================================
#  Train readout on frozen grids
# =============================================================================

def run_readout_epoch(readout: AttnPoolReadout, loader: DataLoader,
                     optimizer: Optional[torch.optim.Optimizer], device: torch.device,
                     grad_clip: float, train: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    readout.train(train)
    total_loss, n_total = 0.0, 0
    all_pred, all_true = [], []
    for g, y in loader:
        g = g.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        B = g.shape[0]
        with torch.set_grad_enabled(train):
            y_pred = readout(g)
            loss = F.mse_loss(y_pred, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(readout.parameters(), grad_clip)
            optimizer.step()
        total_loss += float(loss.item()) * B
        n_total += B
        all_pred.append(y_pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
    return total_loss / max(n_total, 1), np.concatenate(all_pred), np.concatenate(all_true)


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Train AttnPoolReadout on frozen JEPA grids")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--jepa_ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--max_train_samples", type=int, default=500000)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--readout_dropout", type=float, default=0.0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print("JEPA GRID READOUT PROBE (symmetric counterpart of supervised_grid)")
    print("=" * 92)

    print("[1/6] Loading frozen JEPA encoder...")
    enc, ckpt, stock_stats = load_jepa_encoder(args.jepa_ckpt, device)
    K = enc.cfg.K; S = enc.cfg.S; D = enc.cfg.d_model
    print(f"  K={K}, S={S}, d_model={D}")

    print("[2/6] Loading dataset & splitting...")
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

    train_pos, val_pos = grouped_split_by_stock_day(stock_ids_arr, day_ids, valid_t,
                                                    args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed)
    if args.max_train_samples > 0 and len(train_pos) > args.max_train_samples:
        train_pos = np.sort(rng.choice(train_pos, args.max_train_samples, replace=False))
    if args.max_val_samples > 0 and len(val_pos) > args.max_val_samples:
        val_pos = np.sort(rng.choice(val_pos, args.max_val_samples, replace=False))
    t_train, t_val = valid_t[train_pos], valid_t[val_pos]
    print(f"  train: {len(t_train):,}   val: {len(t_val):,}")

    print("[3/6] Building targets...")
    y_train_raw, target_names = build_targets(book, mid_z, stock_ids_arr,
                                              t_train, raw_feat, min_spread_per_stock)
    y_val_raw, _ = build_targets(book, mid_z, stock_ids_arr,
                                 t_val, raw_feat, min_spread_per_stock)
    y_train, y_val, target_mean, target_std = standardize_targets(y_train_raw, y_val_raw)

    print("[4/6] Extracting frozen grids (one pass over the data)...")
    print("  train grids...")
    g_train = extract_grids(enc, book, mid_z, stock_ids_arr, t_train, stock_stats, K,
                            device, args.batch_size, args.num_workers)
    print(f"    g_train: {g_train.shape}")
    print("  val grids...")
    g_val = extract_grids(enc, book, mid_z, stock_ids_arr, t_val, stock_stats, K,
                          device, args.batch_size, args.num_workers)
    print(f"    g_val: {g_val.shape}")

    print("[5/6] Training readout on frozen grids...")
    readout_cfg = ReadoutConfig(d_model=D, out_dim=22, dropout=args.readout_dropout)
    readout = AttnPoolReadout(readout_cfg).to(device)
    n_rd = sum(p.numel() for p in readout.parameters())
    print(f"  readout params: {n_rd:,}")

    optimizer = torch.optim.AdamW(readout.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    ds_tr = GridTargetDataset(g_train, y_train); ds_va = GridTargetDataset(g_val, y_val)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=2, pin_memory=(device.type == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=2, pin_memory=(device.type == "cuda"))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_val_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_p, tr_t = run_readout_epoch(readout, dl_tr, optimizer, device,
                                                args.grad_clip, train=True)
        va_loss, va_p, va_t = run_readout_epoch(readout, dl_va, None, device,
                                                args.grad_clip, train=False)
        tr_s = summarize_r2(target_names, r2_per_target(tr_t, tr_p))
        va_s = summarize_r2(target_names, r2_per_target(va_t, va_p))
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs} [{dt:.1f}s] "
              f"train_mse={tr_loss:.4f} val_mse={va_loss:.4f} | "
              f"val R²: all={va_s['mean_all']:.4f} future={va_s['mean_future']:.4f} "
              f"vol={va_s['mean_vol']:.4f}")
        history.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": va_loss,
                        "train_r2": tr_s, "val_r2": va_s})
        if va_loss < best_val_mse:
            best_val_mse = va_loss
            torch.save({
                "model_type": "jepa_grid_readout",
                "readout_state_dict": readout.state_dict(),
                "readout_cfg": {"d_model": D, "out_dim": 22, "dropout": args.readout_dropout},
                "enc_cfg": enc.cfg.to_dict(),
                "target_names": target_names,
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist(),
                "stock_stats": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                for k, v in stock_stats.items()},
                "val_metrics": {"mse": va_loss, **{f"r2_{k}": v for k, v in va_s.items()}},
                "train_args": vars(args),
            }, out_dir / "best.pt")

    print("[6/6] Done.")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  best val_mse: {best_val_mse:.4f}")
    print(f"  saved: {out_dir/'best.pt'}, {out_dir/'history.json'}")


if __name__ == "__main__":
    main()

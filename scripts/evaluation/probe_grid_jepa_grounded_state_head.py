#!/usr/bin/env python3
"""
probe_grid_jepa_grounded_state_head.py

Grounded compression probe for Horizon Grid-JEPA.

Goal
====
A trained Horizon JEPA grid encoder produces a distributed representation

    H_t = encoder(W_t)  with shape (B, K, S, D) = (B, 20, 4, 128).

This script freezes that encoder and learns a small grounded state head

    H_t -> z_t in R^d -> observable targets

where the observable targets are the same future microstructure/volatility targets
used in A1-T probes. This is NOT a JEPA loss: the compression is grounded by
external, non-co-adaptable targets.

Decision use
============
If z_d keeps most of the predictive ceiling of the grid readout, z_d is a
candidate JEPA-derived state for downstream WM. If it fails, Horizon JEPA remains
a distributed representation learner rather than a compact state model.

Example
-------
python -m scripts.evaluation.probe_grid_jepa_grounded_state_head \
  --dataset data/lobench_processed.npz \
  --horizon_ckpt checkpoints/jepa_horizon/v1/best.pt \
  --out_dir validation/grounded_state_head/jepa_horizon_best \
  --z_dims 32,64 \
  --readout attn_all \
  --max_train_samples 100000 \
  --max_val_samples 50000 \
  --batch_size 512 \
  --probe_batch_size 1024 \
  --epochs 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve()
for p in [HERE.parent, *HERE.parents, Path.cwd(), *Path.cwd().parents]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from training.historical.train_jepa_horizon import (  # type: ignore
        HorizonJEPAEncoder,
        HorizonJEPAEncoderConfig,
    )
except Exception as e:
    raise SystemExit(
        "Cannot import HorizonJEPAEncoder from training/train_jepa_horizon.py. "
        "Run from the thesis project root. Original error: " + repr(e)
    )

try:
    from training.train_tokenizer_t import (  # type: ignore
        compute_valid_endpoints,
        normalize_book_window,
        derive_raw_features_array,
        compute_future_feature_targets,
        compute_vol_targets,
        fit_target_standardizer,
        apply_standardizer,
        grouped_split_by_stock_day,
        compute_stock_stats_train_only,
    )
except Exception as e:
    raise SystemExit(
        "Cannot import tokenizer utilities from training/train_tokenizer_t.py. "
        "Run from the thesis project root. Original error: " + repr(e)
    )


# =============================================================================
# Utilities
# =============================================================================

def robust_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def maybe_subsample(pos: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n is None or max_n <= 0 or len(pos) <= max_n:
        return pos
    rng = np.random.default_rng(seed)
    out = rng.choice(pos, size=max_n, replace=False)
    out.sort()
    return out


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    yc = y_true - y_true.mean(axis=0, keepdims=True)
    ss_tot = (yc ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def summarize_r2(names: List[str], r2: np.ndarray) -> Dict[str, float]:
    future_idx = [i for i, n in enumerate(names) if not n.startswith("realized_vol")]
    vol_idx = [i for i, n in enumerate(names) if n.startswith("realized_vol")]
    return {
        "mean_all": float(np.mean(r2)),
        "median_all": float(np.median(r2)),
        "mean_future_delta": float(np.mean(r2[future_idx])) if future_idx else float("nan"),
        "mean_realized_vol": float(np.mean(r2[vol_idx])) if vol_idx else float("nan"),
    }


def to_numpy_stats(stock_stats: Dict) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stock_stats.items()}


@torch.no_grad()
def latent_diagnostics(z: torch.Tensor) -> Dict[str, float]:
    std = z.std(dim=0)
    zc = z - z.mean(dim=0, keepdim=True)
    B, d = z.shape
    cov = (zc.T @ zc) / max(B - 1, 1)
    offdiag_abs = cov.abs().sum() - cov.diagonal().abs().sum()
    cov_offdiag = (offdiag_abs / max(d * d - d, 1)).item()
    try:
        s = torch.linalg.svdvals(zc)
        p = s / (s.sum() + 1e-12)
        p = p[p > 1e-12]
        eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()
    except Exception:
        eff_rank = float("nan")
    return {
        "z_std_mean": float(std.mean().item()),
        "z_std_min": float(std.min().item()),
        "z_eff_rank": float(eff_rank),
        "z_norm": float(z.norm(dim=-1).mean().item()),
        "z_cov_offdiag": float(cov_offdiag),
    }


# =============================================================================
# Dataset and encoder extraction
# =============================================================================

class RawWindowDataset(Dataset):
    def __init__(
        self,
        book: np.ndarray,
        mid_z: np.ndarray,
        stock_ids: np.ndarray,
        valid_t: np.ndarray,
        stock_stats: Dict[str, np.ndarray],
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
        book_win = self.book[t - K + 1: t + 1]
        mid_win = self.mid_z[t - K + 1: t + 1]
        book_norm = normalize_book_window(book_win, mid_win, s, self.stock_stats)
        return torch.from_numpy(book_norm).float(), torch.tensor(s, dtype=torch.long)


def load_horizon_jepa_encoder(ckpt_path: str, device: torch.device) -> Tuple[HorizonJEPAEncoder, Dict]:
    ckpt = robust_torch_load(ckpt_path, device)
    if ckpt.get("model_type") == "compact_horizon_jepa":
        raise ValueError("This script expects a GRID Horizon JEPA checkpoint, not compact_horizon_jepa.")
    enc_cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    enc = HorizonJEPAEncoder(enc_cfg).to(device)
    state = ckpt.get("online_state_dict", ckpt.get("encoder_state_dict", None))
    if state is None:
        raise ValueError("Horizon JEPA checkpoint must contain online_state_dict or encoder_state_dict")
    enc.load_state_dict(state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, ckpt


@torch.no_grad()
def extract_grids(
    encoder: HorizonJEPAEncoder,
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label: str,
) -> np.ndarray:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    grids = []
    t0 = time.time()
    n = 0
    for book, stock_ids in loader:
        book = book.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        grid = encoder(book, stock_ids, mask=None)       # (B,K,S,D)
        grids.append(grid.detach().cpu().numpy().astype(np.float32))
        n += book.shape[0]
    out = np.concatenate(grids, axis=0)
    dt = time.time() - t0
    print(f"  extracted {label}: {out.shape} in {dt:.1f}s ({n/max(dt,1e-9):.0f}/s)")
    return out


# =============================================================================
# Grounded state-head models
# =============================================================================

class AttnAllStateHead(nn.Module):
    """Learned query attention over all K*S grid tokens -> z_d."""
    def __init__(self, token_dim: int, z_dim: int, attn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(attn_dim) / math.sqrt(attn_dim))
        self.k = nn.Linear(token_dim, attn_dim)
        self.v = nn.Linear(token_dim, attn_dim)
        self.norm_r = nn.LayerNorm(attn_dim)
        self.proj_z = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (B,K,S,D)
        B = grid.shape[0]
        x = grid.reshape(B, -1, grid.shape[-1])
        k = self.k(x)                                      # (B,T,A)
        v = self.v(x)                                      # (B,T,A)
        scores = (k * self.q.view(1, 1, -1)).sum(dim=-1) / math.sqrt(k.shape[-1])
        w = torch.softmax(scores, dim=1)                    # (B,T)
        r = (w.unsqueeze(-1) * v).sum(dim=1)                # (B,A)
        r = self.norm_r(r)
        return self.proj_z(r)                              # (B,z)

    @torch.no_grad()
    def attention_weights(self, grid: torch.Tensor) -> torch.Tensor:
        B = grid.shape[0]
        x = grid.reshape(B, -1, grid.shape[-1])
        k = self.k(x)
        scores = (k * self.q.view(1, 1, -1)).sum(dim=-1) / math.sqrt(k.shape[-1])
        return torch.softmax(scores, dim=1)                 # (B,K*S)


class MeanAllStateHead(nn.Module):
    def __init__(self, token_dim: int, z_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = grid.mean(dim=(1, 2))                           # (B,D)
        return self.net(x)


class LastConcatStateHead(nn.Module):
    def __init__(self, K: int, S: int, token_dim: int, z_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.S = S
        self.token_dim = token_dim
        in_dim = S * token_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
            nn.LayerNorm(z_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        last = grid[:, -1, :, :].reshape(grid.shape[0], -1)  # (B,S*D)
        return self.net(last)


class TargetHead(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GroundedStateProbe(nn.Module):
    def __init__(self, state_head: nn.Module, target_head: nn.Module):
        super().__init__()
        self.state_head = state_head
        self.target_head = target_head

    def encode_state(self, grid: torch.Tensor) -> torch.Tensor:
        return self.state_head(grid)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        z = self.state_head(grid)
        return self.target_head(z)


class FlattenGridDataset(Dataset):
    def __init__(self, grids: np.ndarray, y: np.ndarray):
        self.grids = torch.from_numpy(grids).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return self.grids.shape[0]
    def __getitem__(self, idx):
        return self.grids[idx], self.y[idx]


@torch.no_grad()
def predict_and_states(model: GroundedStateProbe, grids: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    preds, zs = [], []
    for i in range(0, grids.shape[0], batch_size):
        xb = torch.from_numpy(grids[i:i + batch_size]).float().to(device)
        z = model.encode_state(xb)
        pred = model.target_head(z)
        zs.append(z.detach().cpu())
        preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0).numpy().astype(np.float32), torch.cat(zs, dim=0).numpy().astype(np.float32)


def make_state_head(readout: str, K: int, S: int, D: int, z_dim: int, hidden: int, dropout: float) -> nn.Module:
    if readout == "attn_all":
        return AttnAllStateHead(D, z_dim, attn_dim=hidden, dropout=dropout)
    if readout == "mean_all":
        return MeanAllStateHead(D, z_dim, hidden=hidden, dropout=dropout)
    if readout == "last_concat":
        return LastConcatStateHead(K, S, D, z_dim, hidden=max(hidden, 256), dropout=dropout)
    raise ValueError(f"Unknown readout {readout!r}")


def train_grounded_probe(
    model: GroundedStateProbe,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    model = model.to(device)
    train_ds = FlattenGridDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.03)
    best = {"val_mse": float("inf"), "epoch": 0, "state": None}
    bad = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        sched.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, x_val.shape[0], batch_size * 4):
                xb = torch.from_numpy(x_val[i:i + batch_size * 4]).float().to(device)
                preds.append(model(xb).detach())
            pred_val = torch.cat(preds, dim=0)
            val_mse = float(F.mse_loss(pred_val, y_val_t).item())
        if val_mse < best["val_mse"] - 1e-7:
            best = {
                "val_mse": val_mse,
                "epoch": ep,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            print(f"    {label:24s} ep={ep:03d} train_mse={total/max(n,1):.5f} val_mse={val_mse:.5f} best_ep={best['epoch']}")
        if bad >= patience:
            print(f"    early stop at ep={ep}, best_ep={best['epoch']}")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    yhat, z_val = predict_and_states(model, x_val, device, batch_size=batch_size * 4)
    info = {
        "best_epoch": int(best["epoch"]),
        "best_val_mse": float(best["val_mse"]),
        "elapsed_sec": float(time.time() - t0),
    }
    return yhat, z_val, info


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon_ckpt", required=True)
    p.add_argument("--out_dir", default="validation/grounded_state_head/jepa_horizon")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--max_train_samples", type=int, default=100000)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for frozen encoder extraction")
    p.add_argument("--probe_batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--z_dims", type=str, default="32,64")
    p.add_argument("--readouts", type=str, default="attn_all", help="Comma-separated: attn_all,mean_all,last_concat")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--state_hidden", type=int, default=128)
    p.add_argument("--target_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--future_features", type=str, default="d_spread_z,d_microprice_rel,d_best_bid_rel,d_best_ask_rel,d_top_imbalance")
    p.add_argument("--future_horizons", type=str, default="1,5,10,20")
    p.add_argument("--vol_horizons", type=str, default="5,20")
    p.add_argument("--gate_threshold", type=float, default=0.25)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    z_dims = [int(x) for x in args.z_dims.split(",") if x.strip()]
    readouts = [x.strip() for x in args.readouts.split(",") if x.strip()]
    future_features = [x.strip() for x in args.future_features.split(",") if x.strip()]
    future_horizons = [int(x) for x in args.future_horizons.split(",") if x.strip()]
    vol_horizons = [int(x) for x in args.vol_horizons.split(",") if x.strip()]
    max_h = max(max(future_horizons), max(vol_horizons))

    print("=" * 100)
    print("GRID-JEPA GROUNDED STATE HEAD PROBE")
    print("=" * 100)
    print(f"dataset      : {args.dataset}")
    print(f"horizon_ckpt : {args.horizon_ckpt}")
    print(f"device       : {device}")
    print(f"z_dims       : {z_dims}")
    print(f"readouts     : {readouts}")
    print(f"gate R2      : {args.gate_threshold:.3f}")

    print("\n[1/7] Loading frozen Horizon Grid-JEPA encoder...")
    encoder, ckpt = load_horizon_jepa_encoder(args.horizon_ckpt, device)
    K, S, D = int(encoder.cfg.K), int(encoder.cfg.S), int(encoder.cfg.d_model)
    print(f"  epoch={ckpt.get('epoch','N/A')}  K={K} S={S} D={D}")

    print("\n[2/7] Loading raw LOBench...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    n_stocks = int(raw["min_spread_z_per_stock"].shape[0]) if "min_spread_z_per_stock" in raw.files else int(stock_ids.max() + 1)
    print(f"  N={len(mid_z):,} n_stocks={n_stocks} L={book.shape[2]}")

    print("\n[3/7] Valid endpoints and grouped split...")
    bid_v = book[:, 0, :, 1]
    ask_v = book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    valid_t = compute_valid_endpoints(stock_ids, day_ids, K, max_h, vol_mask)
    train_pos, val_pos = grouped_split_by_stock_day(stock_ids, day_ids, valid_t, args.val_frac, args.split_seed)
    train_pos = maybe_subsample(train_pos, args.max_train_samples, args.seed + 11)
    val_pos = maybe_subsample(val_pos, args.max_val_samples, args.seed + 17)
    train_t = valid_t[train_pos]
    val_t = valid_t[val_pos]
    print(f"  valid_t={len(valid_t):,} train={len(train_t):,} val={len(val_t):,} max_h={max_h}")

    print("\n[4/7] Observable targets...")
    t0 = time.time()
    raw_feat, _raw_names = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)
    fut_train_raw = compute_future_feature_targets(raw_feat, train_t, future_features, future_horizons)
    fut_val_raw = compute_future_feature_targets(raw_feat, val_t, future_features, future_horizons)
    vol_train_raw = compute_vol_targets(mid_z, train_t, vol_horizons, raw["min_spread_z_per_stock"], stock_ids)
    vol_val_raw = compute_vol_targets(mid_z, val_t, vol_horizons, raw["min_spread_z_per_stock"], stock_ids)
    y_train_raw = np.concatenate([fut_train_raw, vol_train_raw], axis=1).astype(np.float32)
    y_val_raw = np.concatenate([fut_val_raw, vol_val_raw], axis=1).astype(np.float32)
    y_mu, y_sd = fit_target_standardizer(y_train_raw)
    y_train = apply_standardizer(y_train_raw, y_mu, y_sd).astype(np.float32)
    y_val = apply_standardizer(y_val_raw, y_mu, y_sd).astype(np.float32)
    target_names = [f"{f}@{h}" for f in future_features for h in future_horizons] + [f"realized_vol@{h}" for h in vol_horizons]
    print(f"  y_train={y_train.shape} y_val={y_val.shape} built in {time.time()-t0:.1f}s")

    print("\n[5/7] Normalization stats and datasets...")
    if "stock_stats" in ckpt:
        stock_stats = to_numpy_stats(ckpt["stock_stats"])
        print("  using stock_stats from Horizon JEPA checkpoint")
    else:
        stock_stats = compute_stock_stats_train_only(book, mid_z, stock_ids, day_ids, train_t, n_stocks)
        print("  computed train-only stock_stats")
    ds_train = RawWindowDataset(book, mid_z, stock_ids, train_t, stock_stats, K)
    ds_val = RawWindowDataset(book, mid_z, stock_ids, val_t, stock_stats, K)

    print("\n[6/7] Extracting frozen grid representations...")
    grid_train = extract_grids(encoder, ds_train, args.batch_size, args.num_workers, device, "train")
    grid_val = extract_grids(encoder, ds_val, args.batch_size, args.num_workers, device, "val")

    print("\n[7/7] Training grounded state heads...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    best_label = None
    best_mean_all = -float("inf")

    for readout in readouts:
        for z_dim in z_dims:
            label = f"{readout}_z{z_dim}"
            print(f"\n  Training {label}...")
            state_head = make_state_head(readout, K, S, D, z_dim, args.state_hidden, args.dropout)
            target_head = TargetHead(z_dim, out_dim=y_train.shape[1], hidden=args.target_hidden, dropout=args.dropout)
            model = GroundedStateProbe(state_head, target_head)
            yhat, z_val, info = train_grounded_probe(
                model, grid_train, y_train, grid_val, y_val, device,
                batch_size=args.probe_batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                label=label,
            )
            r2 = r2_per_target(y_val, yhat)
            summary = summarize_r2(target_names, r2)
            z_diag_t = latent_diagnostics(torch.from_numpy(z_val).float())
            print(f"  {label} R²: mean_all={summary['mean_all']:.4f}  future={summary['mean_future_delta']:.4f}  vol={summary['mean_realized_vol']:.4f}")
            print("  z diagnostics: " + ", ".join(f"{k}={v:.4f}" for k, v in z_diag_t.items()))

            state_path = out_dir / f"state_head_{label}.pt"
            torch.save({
                "model_type": "grid_jepa_grounded_state_head",
                "readout": readout,
                "z_dim": z_dim,
                "K": K,
                "S": S,
                "D": D,
                "state_hidden": args.state_hidden,
                "target_hidden": args.target_hidden,
                "dropout": args.dropout,
                "state_dict": model.state_dict(),
                "state_head_state_dict": model.state_head.state_dict(),
                "target_head_state_dict": model.target_head.state_dict(),
                "horizon_ckpt": args.horizon_ckpt,
                "target_names": target_names,
                "y_standardizer": {"mean": y_mu.tolist(), "std": y_sd.tolist()},
                "metrics": {"r2": r2.tolist(), "summary": summary, "z_diagnostics": z_diag_t, "train_info": info},
                "args": vars(args),
            }, state_path)
            print(f"  saved state head: {state_path}")

            results[label] = {
                "readout": readout,
                "z_dim": z_dim,
                "r2": r2.tolist(),
                "summary": summary,
                "z_diagnostics": z_diag_t,
                "train_info": info,
                "state_head_path": str(state_path),
            }
            if summary["mean_all"] > best_mean_all:
                best_mean_all = summary["mean_all"]
                best_label = label

    print("\n" + "=" * 100)
    print("GROUNDING PROBE SUMMARY")
    print("=" * 100)
    for label, r in results.items():
        s = r["summary"]
        decision = "PASS" if s["mean_all"] >= args.gate_threshold else "fail"
        print(f"  {label:20s} mean_all={s['mean_all']:.4f} future={s['mean_future_delta']:.4f} vol={s['mean_realized_vol']:.4f}  {decision}")
    print(f"\nBest: {best_label} mean_all={best_mean_all:.4f}")
    if best_mean_all >= args.gate_threshold:
        print("Decision: grounded compression passes gate; this z is a candidate JEPA-derived WM state.")
    else:
        print("Decision: grounded compression fails gate; grid-JEPA remains distributed representation, not compact WM state.")

    meta = {
        "args": vars(args),
        "checkpoint": {"path": args.horizon_ckpt, "epoch": ckpt.get("epoch", None), "enc_cfg": ckpt.get("enc_cfg", None)},
        "target_names": target_names,
        "n_train": int(len(train_t)),
        "n_val": int(len(val_t)),
        "gate_threshold": float(args.gate_threshold),
        "best_label": best_label,
        "best_mean_all": float(best_mean_all),
        "y_standardizer": {"mean": y_mu.tolist(), "std": y_sd.tolist()},
    }
    with open(out_dir / "grounded_state_head_metrics.json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    np.savez_compressed(
        out_dir / "grounded_state_head_val_arrays.npz",
        y_val=y_val,
        **{f"z_val_{label}": np.zeros((1,), dtype=np.float32) for label in []},
    )
    print(f"\nSaved metrics to: {out_dir / 'grounded_state_head_metrics.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
probe_jepa_horizon_readouts.py

Diagnostic readout probes for Horizon JEPA horizon-conditioned JEPA token grids.

Goal
----
Given a trained Horizon JEPA checkpoint, extract the unmasked token grid
    H_t = encoder(W_t)  with shape (B, K, S=4, d_model=128)
and test whether the predictive information found in the last-timestep concat
can be compressed into a compact state.

Readouts tested:
  1. last_concat512 + linear bottleneck z_dim -> targets
     (low-rank supervised compression, not PCA)
  2. attention pooling over the four last-timestep semantic tokens -> z_dim -> targets
  3. optional MLP probe from last_concat512 -> targets

This script is intentionally a probe/eval file. It does NOT change the JEPA
trainer. It is meant to be run on Horizon JEPA checkpoints such as epoch_005.pt, best.pt,
or last.pt. The extracted readouts are the same as in Masked JEPA: last_mean128,
last_concat512, learned bottleneck, attention pooling, and MLP diagnostics.

Example
-------
python -m scripts.evaluation.probe_jepa_horizon_readouts \
  --dataset data/lobench_processed.npz \
  --horizon_ckpt checkpoints/jepa_horizon/v1/epoch_005.pt \
  --out_dir validation/readout_diag/jepa_horizon_epoch005 \
  --max_train_samples 100000 \
  --max_val_samples 50000 \
  --batch_size 512 \
  --num_workers 2
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

HERE = Path(__file__).resolve()
for p in [HERE.parent, *HERE.parents, Path.cwd(), *Path.cwd().parents]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# ---- Project imports ----
try:
    from training.historical.train_jepa_horizon import (
        HorizonJEPAEncoder,
        HorizonJEPAEncoderConfig,
    )
except Exception as e:
    raise SystemExit(
        "Cannot import Horizon JEPA classes from training/train_jepa_horizon.py. "
        "Run this script from the thesis project root. Original error: " + repr(e)
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
        "Cannot import tokenizer data utilities from training/train_tokenizer_t.py. "
        "Run this script from the thesis project root. Original error: " + repr(e)
    )


# =============================================================================
# Utilities
# =============================================================================

def robust_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def local_grouped_split_by_stock_day(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    valid_t: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_day = int(day_ids.max()) + 1
    composite = stock_ids[valid_t].astype(np.int64) * max_day + day_ids[valid_t].astype(np.int64)
    groups = np.unique(composite)
    rng.shuffle(groups)
    n_val = max(1, int(round(val_frac * len(groups))))
    val_groups = set(groups[:n_val].tolist())
    val_mask = np.array([g in val_groups for g in composite])
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def maybe_subsample(pos: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n is None or max_n <= 0 or len(pos) <= max_n:
        return pos
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pos, size=max_n, replace=False))


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    yc = y_true - y_true.mean(axis=0, keepdims=True)
    ss_tot = (yc ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def summarize_r2(names: List[str], r2: np.ndarray) -> Dict[str, float]:
    future_idx = [i for i, n in enumerate(names) if not n.startswith("realized_vol")]
    vol_idx = [i for i, n in enumerate(names) if n.startswith("realized_vol")]
    out = {
        "mean_all": float(np.mean(r2)),
        "median_all": float(np.median(r2)),
        "mean_future_delta": float(np.mean(r2[future_idx])) if future_idx else float("nan"),
        "mean_realized_vol": float(np.mean(r2[vol_idx])) if vol_idx else float("nan"),
    }
    return out


def print_summary(label: str, names: List[str], r2: np.ndarray) -> Dict[str, float]:
    s = summarize_r2(names, r2)
    print(f"\n  {label} R² summary:")
    print(f"    mean all targets        : {s['mean_all']:.4f}")
    print(f"    median all targets      : {s['median_all']:.4f}")
    print(f"    mean future_delta       : {s['mean_future_delta']:.4f}")
    print(f"    mean realized_vol       : {s['mean_realized_vol']:.4f}")
    return s


def standardize_x(train: np.ndarray, val: np.ndarray, eps: float = 1e-6):
    mu = train.mean(axis=0, keepdims=True).astype(np.float32)
    sd = train.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(sd, eps)
    return ((train - mu) / sd).astype(np.float32), ((val - mu) / sd).astype(np.float32), mu, sd


# =============================================================================
# Dataset / extraction
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


def to_numpy_stats(stock_stats: Dict) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stock_stats.items()}


def load_horizon_jepa_encoder(ckpt_path: str, device: torch.device) -> Tuple[HorizonJEPAEncoder, Dict]:
    ckpt = robust_torch_load(ckpt_path, device)
    enc_cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    enc = HorizonJEPAEncoder(enc_cfg).to(device)
    state = ckpt.get("online_state_dict", ckpt.get("encoder_state_dict", None))
    if state is None:
        raise ValueError("Horizon JEPA checkpoint must contain online_state_dict")
    enc.load_state_dict(state)
    enc.eval()
    return enc, ckpt


@torch.no_grad()
def extract_token_grids(
    encoder: HorizonJEPAEncoder,
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label: str,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    last_mean, last_concat, last_tokens = [], [], []
    t0 = time.time()
    n = 0
    for book, stock_ids in loader:
        book = book.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        grid = encoder(book, stock_ids, mask=None)                  # (B,K,S,D)
        last = grid[:, -1, :, :]                                    # (B,4,128)
        last_tokens.append(last.detach().cpu().numpy().astype(np.float32))
        last_mean.append(last.mean(dim=1).detach().cpu().numpy().astype(np.float32))
        last_concat.append(last.reshape(last.shape[0], -1).detach().cpu().numpy().astype(np.float32))
        n += book.shape[0]
    dt = time.time() - t0
    print(f"  Horizon JEPA {label}: extracted {n:,} token readouts in {dt:.1f}s ({n/max(dt,1e-9):.0f}/s)")
    out = {
        "last_tokens": np.concatenate(last_tokens, axis=0),       # (N,4,128)
        "last_mean128": np.concatenate(last_mean, axis=0),
        "last_concat512": np.concatenate(last_concat, axis=0),
    }
    for k, v in out.items():
        print(f"    {k:18s} shape={v.shape}")
    return out


# =============================================================================
# Supervised readout probes
# =============================================================================

class LinearBottleneckProbe(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, z_dim, bias=True)
        self.head = nn.Linear(z_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        return self.head(z)


class AttentionPoolProbe(nn.Module):
    def __init__(self, token_dim: int, z_dim: int, out_dim: int, n_tokens: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(z_dim) / math.sqrt(z_dim))
        self.k = nn.Linear(token_dim, z_dim)
        self.v = nn.Linear(token_dim, z_dim)
        self.head = nn.Linear(z_dim, out_dim)
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,4,128)
        k = self.k(x)                                               # (B,4,z)
        v = self.v(x)                                               # (B,4,z)
        scores = (k * self.q.view(1, 1, -1)).sum(dim=-1) / math.sqrt(k.shape[-1])
        w = torch.softmax(scores, dim=1)                             # (B,4)
        z = (w.unsqueeze(-1) * v).sum(dim=1)                          # (B,z)
        z = self.norm(z)
        return self.head(z)

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k(x)
        scores = (k * self.q.view(1, 1, -1)).sum(dim=-1) / math.sqrt(k.shape[-1])
        return torch.softmax(scores, dim=1)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
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


def train_torch_probe(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 15,
    label: str = "probe",
) -> Tuple[np.ndarray, Dict]:
    model = model.to(device)
    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    x_val_t = torch.from_numpy(x_val).float().to(device)
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
        with torch.no_grad():
            # Chunk val if needed to keep memory bounded.
            preds = []
            for i in range(0, x_val_t.shape[0], batch_size * 4):
                preds.append(model(x_val_t[i:i + batch_size * 4]).detach())
            pred_val = torch.cat(preds, dim=0)
            val_mse = float(F.mse_loss(pred_val, y_val_t).item())

        if val_mse < best["val_mse"] - 1e-7:
            best = {"val_mse": val_mse, "epoch": ep, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            print(f"    {label:28s} ep={ep:03d} train_mse={total/max(n,1):.5f} val_mse={val_mse:.5f} best_ep={best['epoch']}")
        if bad >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, x_val_t.shape[0], batch_size * 4):
            preds.append(model(x_val_t[i:i + batch_size * 4]).detach().cpu())
        y_pred = torch.cat(preds, dim=0).numpy().astype(np.float32)
    info = {"best_epoch": int(best["epoch"]), "best_val_mse": float(best["val_mse"]), "elapsed_sec": time.time() - t0}
    return y_pred, info


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon_ckpt", required=True)
    p.add_argument("--out_dir", default="validation/readout_diag/jepa_horizon")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--max_train_samples", type=int, default=100000)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=512, help="Batch size for encoder extraction")
    p.add_argument("--probe_batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Probe controls
    p.add_argument("--z_dims", type=str, default="32,64", help="Comma-separated bottleneck dims")
    p.add_argument("--probe_epochs", type=int, default=80)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_weight_decay", type=float, default=1e-3)
    p.add_argument("--probe_patience", type=int, default=15)
    p.add_argument("--run_mlp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mlp_hidden", type=int, default=256)

    # Targets
    p.add_argument("--future_features", type=str, default="d_spread_z,d_microprice_rel,d_best_bid_rel,d_best_ask_rel,d_top_imbalance")
    p.add_argument("--future_horizons", type=str, default="1,5,10,20")
    p.add_argument("--vol_horizons", type=str, default="5,20")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    z_dims = [int(x) for x in args.z_dims.split(",") if x.strip()]
    future_features = [x.strip() for x in args.future_features.split(",") if x.strip()]
    future_horizons = [int(x) for x in args.future_horizons.split(",") if x.strip()]
    vol_horizons = [int(x) for x in args.vol_horizons.split(",") if x.strip()]
    max_h = max(max(future_horizons), max(vol_horizons))

    print("=" * 92)
    print("Horizon JEPA READOUT DIAGNOSTICS — learned bottleneck / attention pooling / MLP")
    print("=" * 92)
    print(f"dataset   : {args.dataset}")
    print(f"horizon_ckpt : {args.horizon_ckpt}")
    print(f"device    : {device}")
    print(f"z_dims    : {z_dims}")

    print("\n[1/7] Loading Horizon JEPA checkpoint...")
    encoder, ckpt = load_horizon_jepa_encoder(args.horizon_ckpt, device)
    K = int(encoder.cfg.K)
    print(f"  epoch={ckpt.get('epoch', 'N/A')}  K={K} S={encoder.cfg.S} d_model={encoder.cfg.d_model}")

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
    splitter = grouped_split_by_stock_day if grouped_split_by_stock_day is not None else local_grouped_split_by_stock_day
    train_pos, val_pos = splitter(stock_ids, day_ids, valid_t, args.val_frac, args.split_seed)
    train_pos = maybe_subsample(train_pos, args.max_train_samples, args.seed + 11)
    val_pos = maybe_subsample(val_pos, args.max_val_samples, args.seed + 17)
    train_t = valid_t[train_pos]
    val_t = valid_t[val_pos]
    print(f"  valid_t={len(valid_t):,} train={len(train_t):,} val={len(val_t):,} max_h={max_h}")

    print("\n[4/7] Building observable targets...")
    t0 = time.time()
    raw_feat, raw_names = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)
    fut_train_raw = compute_future_feature_targets(raw_feat, train_t, future_features, future_horizons)
    fut_val_raw = compute_future_feature_targets(raw_feat, val_t, future_features, future_horizons)
    vol_train_raw = compute_vol_targets(mid_z, train_t, vol_horizons, raw["min_spread_z_per_stock"], stock_ids)
    vol_val_raw = compute_vol_targets(mid_z, val_t, vol_horizons, raw["min_spread_z_per_stock"], stock_ids)
    y_train_raw = np.concatenate([fut_train_raw, vol_train_raw], axis=1).astype(np.float32)
    y_val_raw = np.concatenate([fut_val_raw, vol_val_raw], axis=1).astype(np.float32)
    y_mu, y_sd = fit_target_standardizer(y_train_raw)
    y_train = apply_standardizer(y_train_raw, y_mu, y_sd).astype(np.float32)
    y_val = apply_standardizer(y_val_raw, y_mu, y_sd).astype(np.float32)
    target_names = []
    for f in future_features:
        for h in future_horizons:
            target_names.append(f"{f}@{h}")
    for h in vol_horizons:
        target_names.append(f"realized_vol@{h}")
    print(f"  targets train={y_train.shape} val={y_val.shape} built in {time.time()-t0:.1f}s")

    print("\n[5/7] Normalization stats and datasets...")
    if "stock_stats" in ckpt:
        stock_stats = to_numpy_stats(ckpt["stock_stats"])
        print("  using stock_stats from Horizon JEPA checkpoint")
    elif compute_stock_stats_train_only is not None:
        stock_stats = compute_stock_stats_train_only(book, mid_z, stock_ids, day_ids, train_t, n_stocks)
        print("  computed train-only stock_stats")
    else:
        raise RuntimeError("No stock_stats in checkpoint and compute_stock_stats_train_only unavailable")

    ds_train = RawWindowDataset(book, mid_z, stock_ids, train_t, stock_stats, K)
    ds_val = RawWindowDataset(book, mid_z, stock_ids, val_t, stock_stats, K)

    print("\n[6/7] Extracting Horizon JEPA readouts...")
    r_train = extract_token_grids(encoder, ds_train, args.batch_size, args.num_workers, device, "train")
    r_val = extract_token_grids(encoder, ds_val, args.batch_size, args.num_workers, device, "val")

    results = {}
    preds_by_model = {}

    print("\n[7/7] Training readout probes...")

    # A) Linear low-rank bottleneck from concat512.
    xtr, xva, xmu, xsd = standardize_x(r_train["last_concat512"], r_val["last_concat512"])
    for z in z_dims:
        label = f"HorizonJEPA_concat512_linear_z{z}"
        print(f"\n  Training {label}...")
        model = LinearBottleneckProbe(in_dim=xtr.shape[1], z_dim=z, out_dim=y_train.shape[1])
        yhat, info = train_torch_probe(
            model, xtr, y_train, xva, y_val, device,
            batch_size=args.probe_batch_size,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            patience=args.probe_patience,
            label=label,
        )
        r2 = r2_per_target(y_val, yhat)
        summary = print_summary(label, target_names, r2)
        results[label] = {"r2": r2.tolist(), "summary": summary, "train_info": info}
        preds_by_model[label] = yhat

    # B) Attention pooling over last tokens (4 x 128).
    # Standardize flattened tokens, then reshape back so each feature dimension is normalized.
    xtok_train = r_train["last_tokens"]
    xtok_val = r_val["last_tokens"]
    xtok_flat_train, xtok_flat_val, _, _ = standardize_x(
        xtok_train.reshape(xtok_train.shape[0], -1),
        xtok_val.reshape(xtok_val.shape[0], -1),
    )
    xtok_train_std = xtok_flat_train.reshape(xtok_train.shape).astype(np.float32)
    xtok_val_std = xtok_flat_val.reshape(xtok_val.shape).astype(np.float32)
    for z in z_dims:
        label = f"HorizonJEPA_attn_pool_z{z}"
        print(f"\n  Training {label}...")
        model = AttentionPoolProbe(token_dim=xtok_train.shape[-1], z_dim=z, out_dim=y_train.shape[1])
        yhat, info = train_torch_probe(
            model, xtok_train_std, y_train, xtok_val_std, y_val, device,
            batch_size=args.probe_batch_size,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            patience=args.probe_patience,
            label=label,
        )
        r2 = r2_per_target(y_val, yhat)
        summary = print_summary(label, target_names, r2)
        # average attention weights on validation
        model.eval().to(device)
        with torch.no_grad():
            chunks = []
            x_val_t = torch.from_numpy(xtok_val_std).float().to(device)
            for i in range(0, x_val_t.shape[0], args.probe_batch_size * 4):
                chunks.append(model.attention_weights(x_val_t[i:i + args.probe_batch_size * 4]).detach().cpu())
            attn_w = torch.cat(chunks, dim=0).mean(dim=0).numpy().astype(float).tolist()
        results[label] = {"r2": r2.tolist(), "summary": summary, "train_info": info, "mean_attention_weights": attn_w,
                          "token_order": ["bid_top", "bid_deep", "ask_top", "ask_deep"]}
        preds_by_model[label] = yhat
        print(f"    mean attn weights [bid_top,bid_deep,ask_top,ask_deep]: {[round(w, 3) for w in attn_w]}")

    # C) Optional nonlinear MLP diagnostic from concat512.
    if args.run_mlp:
        label = f"HorizonJEPA_concat512_mlp_h{args.mlp_hidden}"
        print(f"\n  Training {label}...")
        model = MLPProbe(in_dim=xtr.shape[1], hidden=args.mlp_hidden, out_dim=y_train.shape[1], dropout=0.1)
        yhat, info = train_torch_probe(
            model, xtr, y_train, xva, y_val, device,
            batch_size=args.probe_batch_size,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            patience=args.probe_patience,
            label=label,
        )
        r2 = r2_per_target(y_val, yhat)
        summary = print_summary(label, target_names, r2)
        results[label] = {"r2": r2.tolist(), "summary": summary, "train_info": info}
        preds_by_model[label] = yhat

    # Compact table.
    print("\n" + "=" * 92)
    print("Compact comparison")
    labels = list(results.keys())
    header = "  target".ljust(34) + "".join([lab[-24:].rjust(26) for lab in labels])
    print(header)
    for i, name in enumerate(target_names):
        row = f"  {name:<32s}" + "".join([f"{results[lab]['r2'][i]:26.4f}" for lab in labels])
        print(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args),
        "checkpoint": {"path": args.horizon_ckpt, "epoch": ckpt.get("epoch", None), "enc_cfg": ckpt.get("enc_cfg", None)},
        "target_names": target_names,
        "n_train": int(len(train_t)),
        "n_val": int(len(val_t)),
        "y_standardizer": {"mean": y_mu.tolist(), "std": y_sd.tolist()},
    }
    with open(out_dir / "readout_probe_metrics.json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    np.savez_compressed(
        out_dir / "readout_probe_arrays.npz",
        y_val=y_val,
        **{f"pred_{k}": v for k, v in preds_by_model.items()},
    )
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()

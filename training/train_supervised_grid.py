#!/usr/bin/env python3
"""
train_supervised_grid.py — Supervised predictive backbone on the JEPA grid encoder.

Purpose
=======
Drop-in supervised arm for the controlled comparison:

    Same encoder architecture as Horizon JEPA (HorizonJEPAEncoder, 4 region tokens,
    non-causal, token-preserving). Same data, same split, same target set.
    Only difference vs F1: trained end-to-end with a supervised MSE loss on the
    22 observable future targets (5 microstructure deltas at horizons {1,5,10,20}
    plus realized vol at {5,20}).

Readout
-------
Token-preserving final stage — NO bottleneck. A learnable query attends to all
K·S grid tokens producing r ∈ R^{d_model}; then Linear(d_model -> 22).

    grid (B, K=20, S=4, d=128)
      |  q (learnable, d=128)
      v
    cross-attention over K·S=80 tokens, scaled dot-product
      v
    r ∈ R^{d_model}
      v
    Linear(d_model -> 22)

This is the readout we agreed for both arms: minimal, identical, non-bottlenecking.
The grounded z32 head from earlier probes lives elsewhere (probe of compressibility).

Output checkpoint format is symmetric with the JEPA-grid readout probe so that
`scripts/evaluation/compare_supervised_jepa.py` consumes both with the same code
path.

Usage
-----
python -m training.train_supervised_grid \\
  --dataset data/lobench_processed.npz \\
  --jepa_ckpt checkpoints/jepa_horizon/v1_500k/epoch_012.pt \\
  --out_dir checkpoints/supervised_grid/v1 \\
  --epochs 15 --batch_size 256

Smoke test (no data needed):
  python -m training.train_supervised_grid --smoke_test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
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


# =============================================================================
#  Project imports
# =============================================================================

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent, _THIS.parent.parent.parent,
           _THIS.parent.parent.parent.parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from training.train_jepa_horizon import (  # type: ignore
        HorizonJEPAEncoder,
        HorizonJEPAEncoderConfig,
    )
except Exception:
    HorizonJEPAEncoder = None
    HorizonJEPAEncoderConfig = None

try:
    from training.train_tokenizer_t import (  # type: ignore
        compute_valid_endpoints,
        normalize_book_window,
        grouped_split_by_stock_day,
        derive_raw_features_array,
        compute_future_feature_targets,
        compute_vol_targets,
    )
except Exception:
    compute_valid_endpoints = None
    normalize_book_window = None
    grouped_split_by_stock_day = None
    derive_raw_features_array = None
    compute_future_feature_targets = None
    compute_vol_targets = None


def _check_imports_or_exit() -> None:
    missing = []
    if HorizonJEPAEncoder is None:
        missing.append("train_jepa_horizon (HorizonJEPAEncoder)")
    if any(x is None for x in [compute_valid_endpoints, normalize_book_window,
                               grouped_split_by_stock_day, derive_raw_features_array,
                               compute_future_feature_targets, compute_vol_targets]):
        missing.append("train_tokenizer_t (data/target utilities)")
    if missing:
        raise SystemExit("Cannot import: " + "; ".join(missing) + ". Run from project root.")


# =============================================================================
#  Readout: attention-pool over the full grid (no bottleneck), then Linear -> 22
# =============================================================================

@dataclass
class ReadoutConfig:
    """Minimal readout: query-attention pool to d_model, then Linear to out_dim."""
    d_model: int = 128
    out_dim: int = 22
    dropout: float = 0.0


class AttnPoolReadout(nn.Module):
    """Token-preserving readout: learnable query, scaled dot-product attention
    over all K·S grid tokens, producing r ∈ R^{d_model}. Then Linear -> out_dim.

    No bottleneck (d_model in, d_model out of the pool). The pool replaces a
    flatten + MLP and keeps the representation at the natural token dimension.

    forward:
        grid: (B, K, S, d)
        return: (B, out_dim)
    """

    def __init__(self, cfg: ReadoutConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Learnable query in d_model space (no auxiliary attn_dim).
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(d, cfg.out_dim)

    def pool(self, grid: torch.Tensor) -> torch.Tensor:
        """grid (B, K, S, d) -> r (B, d). Exposed for diagnostics/probing."""
        B = grid.shape[0]
        x = grid.reshape(B, -1, grid.shape[-1])          # (B, K*S, d)
        k = self.k(x)                                     # (B, K*S, d)
        v = self.v(x)                                     # (B, K*S, d)
        # Scaled dot-product with the learnable query q.
        scores = (k * self.q.view(1, 1, -1)).sum(dim=-1) / math.sqrt(k.shape[-1])
        w = torch.softmax(scores, dim=1)                  # (B, K*S)
        r = (w.unsqueeze(-1) * v).sum(dim=1)              # (B, d)
        return self.norm(r)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        r = self.pool(grid)
        return self.head(self.drop(r))


# =============================================================================
#  Full supervised model: encoder + readout
# =============================================================================

class SupervisedGrid(nn.Module):
    """Encoder + AttnPoolReadout. Single nn.Module, trained end-to-end."""

    def __init__(self, enc_cfg: HorizonJEPAEncoderConfig, readout_cfg: ReadoutConfig):
        super().__init__()
        assert enc_cfg.d_model == readout_cfg.d_model, \
            f"d_model mismatch encoder={enc_cfg.d_model} readout={readout_cfg.d_model}"
        self.encoder = HorizonJEPAEncoder(enc_cfg)
        self.readout = AttnPoolReadout(readout_cfg)

    def forward(self, book: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        grid = self.encoder(book, stock_ids)              # (B, K, S, d)
        return self.readout(grid)                         # (B, out_dim)

    @torch.no_grad()
    def encode_grid(self, book: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(book, stock_ids)


# =============================================================================
#  Dataset: (W_t, y_22, stock_id) with precomputed targets
# =============================================================================

class SupervisedGridDataset(Dataset):
    """
    Returns:
        W_t:       (K, 2, L, 2)   normalized window
        y:         (out_dim,)     standardized target vector (22 dims)
        stock_id:  scalar long
    """

    def __init__(
        self,
        book: np.ndarray,
        mid_z: np.ndarray,
        stock_ids: np.ndarray,
        valid_t: np.ndarray,
        stock_stats: Dict[str, np.ndarray],
        y_standardized: np.ndarray,
        K: int,
    ):
        if len(valid_t) != y_standardized.shape[0]:
            raise ValueError(f"valid_t ({len(valid_t)}) and targets "
                             f"({y_standardized.shape[0]}) must align")
        self.book = book
        self.mid_z = mid_z
        self.stock_ids = stock_ids
        self.valid_t = valid_t
        self.stock_stats = stock_stats
        self.y = y_standardized.astype(np.float32, copy=False)
        self.K = K
        self.L = book.shape[2]

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, idx: int):
        t = int(self.valid_t[idx])
        s = int(self.stock_ids[t])
        K = self.K
        w_t = normalize_book_window(
            self.book[t - K + 1 : t + 1],
            self.mid_z[t - K + 1 : t + 1],
            s, self.stock_stats,
        )
        return (
            torch.from_numpy(w_t).float(),
            torch.from_numpy(self.y[idx]).float(),
            torch.tensor(s, dtype=torch.long),
        )


# =============================================================================
#  Target building and standardization
# =============================================================================

FUTURE_FEATURES = ["d_spread_z", "d_microprice_rel", "d_best_bid_rel",
                   "d_best_ask_rel", "d_top_imbalance"]
FUTURE_HORIZONS = [1, 5, 10, 20]
VOL_HORIZONS = [5, 20]


def build_targets(
    book: np.ndarray, mid_z: np.ndarray, stock_ids: np.ndarray,
    valid_t: np.ndarray, raw_feat: np.ndarray, min_spread_per_stock: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Compute the 22-dim raw target vector for each valid_t.

    Column order:
        future: feature_outer × horizon_inner
            d_spread_z@1, d_spread_z@5, ..., d_top_imbalance@20    (20 cols)
        vol: realized_vol@5, realized_vol@20                        (2 cols)
    """
    fut = compute_future_feature_targets(raw_feat, valid_t, FUTURE_FEATURES, FUTURE_HORIZONS)
    vol = compute_vol_targets(mid_z, valid_t, VOL_HORIZONS, min_spread_per_stock, stock_ids)
    y_raw = np.concatenate([fut, vol], axis=1).astype(np.float32)
    names = [f"{f}@{h}" for f in FUTURE_FEATURES for h in FUTURE_HORIZONS] \
            + [f"realized_vol@{h}" for h in VOL_HORIZONS]
    return y_raw, names


def standardize_targets(y_train_raw: np.ndarray, y_val_raw: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-target z-score using TRAIN statistics. Returns (y_train_z, y_val_z, mean, std)."""
    mean = y_train_raw.mean(axis=0).astype(np.float32)
    std = y_train_raw.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-8)
    y_train_z = (y_train_raw - mean) / std
    y_val_z = (y_val_raw - mean) / std
    return y_train_z, y_val_z, mean, std


# =============================================================================
#  Metrics
# =============================================================================

def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-target R² on the standardized predictions (which equals R² on raw)."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + eps
    return 1.0 - ss_res / ss_tot


def summarize_r2(names: List[str], r2: np.ndarray) -> Dict[str, float]:
    future_idx = [i for i, n in enumerate(names) if "realized_vol" not in n]
    vol_idx = [i for i, n in enumerate(names) if "realized_vol" in n]
    return {
        "mean_all": float(np.mean(r2)),
        "mean_future": float(np.mean(r2[future_idx])) if future_idx else float("nan"),
        "mean_vol": float(np.mean(r2[vol_idx])) if vol_idx else float("nan"),
    }


# =============================================================================
#  Epoch
# =============================================================================

def run_epoch(model: SupervisedGrid, loader: DataLoader,
              optimizer: Optional[torch.optim.Optimizer], device: torch.device,
              grad_clip: float, train: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(train)
    total_loss, n_total = 0.0, 0
    all_pred, all_true = [], []
    for W_t, y, stock_ids in loader:
        W_t = W_t.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        B = W_t.shape[0]

        with torch.set_grad_enabled(train):
            y_pred = model(W_t, stock_ids)
            loss = F.mse_loss(y_pred, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()) * B
        n_total += B
        all_pred.append(y_pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())

    avg_loss = total_loss / max(n_total, 1)
    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    return avg_loss, y_pred, y_true


# =============================================================================
#  Checkpoint
# =============================================================================

def save_checkpoint(path: Path, epoch: int, model: SupervisedGrid,
                    optimizer: torch.optim.Optimizer,
                    enc_cfg: HorizonJEPAEncoderConfig, readout_cfg: ReadoutConfig,
                    target_names: List[str], target_mean: np.ndarray, target_std: np.ndarray,
                    stock_stats: Dict[str, np.ndarray], train_args: dict,
                    val_metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_type": "supervised_grid",
        "format_version": "supervised_grid_v1",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "enc_cfg": enc_cfg.to_dict(),
        "readout_cfg": asdict(readout_cfg),
        "target_names": list(target_names),
        "target_mean": np.asarray(target_mean, dtype=np.float32).tolist(),
        "target_std": np.asarray(target_std, dtype=np.float32).tolist(),
        "stock_stats": {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in stock_stats.items()},
        "train_args": train_args,
        "val_metrics": val_metrics,
    }
    torch.save(state, path)


# =============================================================================
#  Stock_stats loader from a JEPA checkpoint (to share normalization)
# =============================================================================

def load_stock_stats_from_jepa(jepa_ckpt: str, device: torch.device) -> Dict[str, np.ndarray]:
    try:
        ckpt = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(jepa_ckpt, map_location=device)
    if "stock_stats" not in ckpt:
        raise ValueError(f"No stock_stats in checkpoint {jepa_ckpt}")
    return {k: np.asarray(v, dtype=np.float32) if not isinstance(v, (int, float)) else v
            for k, v in ckpt["stock_stats"].items()}


# =============================================================================
#  Smoke test
# =============================================================================

def smoke_test() -> None:
    print("=" * 80)
    print("Supervised Grid — SMOKE TEST")
    print("=" * 80)
    if HorizonJEPAEncoder is None:
        raise SystemExit("Run from project root: HorizonJEPAEncoder not importable.")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, K, L = 4, 20, 10
    enc_cfg = HorizonJEPAEncoderConfig(L=L, n_stocks=7, K=K, S=4, raw_per_token=10,
                                       d_model=64, spatial_n_layers=1, temporal_n_layers=1,
                                       spatial_d_ffn=128, temporal_d_ffn=128)
    readout_cfg = ReadoutConfig(d_model=64, out_dim=22)
    model = SupervisedGrid(enc_cfg, readout_cfg).to(device)

    book = torch.randn(B, K, 2, L, 2, device=device)
    stock_ids = torch.randint(0, 7, (B,), device=device)
    y = torch.randn(B, 22, device=device)

    y_pred = model(book, stock_ids)
    assert y_pred.shape == (B, 22), f"y_pred {y_pred.shape}, expected {(B, 22)}"
    loss = F.mse_loss(y_pred, y)
    loss.backward()
    print(f"  forward OK  : y {tuple(y_pred.shape)}, loss={loss.item():.4f}")

    grid = model.encode_grid(book, stock_ids)
    assert grid.shape == (B, K, 4, 64)
    print(f"  encode_grid : grid {tuple(grid.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params      : {n_params:,}")
    print("\nSmoke test passed.")


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Supervised Grid: encoder + attn-pool readout")
    p.add_argument("--smoke_test", action="store_true")

    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--jepa_ckpt", type=str, default=None,
                   help="JEPA checkpoint to inherit stock_stats from (recommended for fair comparison)")
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--max_train_samples", type=int, default=500000)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    # split_seed is DECOUPLED from --seed: it controls ONLY the grouped
    # train/val split, held FIXED across seeds and arms so multi-seed bars
    # measure optimization variance, not split variance. --seed varies.
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--eta_min_frac", type=float, default=0.01)

    # Encoder config — defaults match the JEPA backbone exactly.
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--S", type=int, default=4)
    p.add_argument("--raw_per_token", type=int, default=10)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--spatial_n_layers", type=int, default=2)
    p.add_argument("--spatial_n_heads", type=int, default=4)
    p.add_argument("--spatial_d_ffn", type=int, default=256)
    p.add_argument("--temporal_n_layers", type=int, default=2)
    p.add_argument("--temporal_n_heads", type=int, default=4)
    p.add_argument("--temporal_d_ffn", type=int, default=256)
    p.add_argument("--temporal_causal", action="store_true",
                   help="Default OFF (non-causal), matching JEPA. Use only if comparing to a causal variant.")
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--readout_dropout", type=float, default=0.0)
    p.add_argument("--save_every", type=int, default=5)
    args = p.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.dataset is None or args.out_dir is None:
        raise SystemExit("--dataset and --out_dir required unless --smoke_test")
    _check_imports_or_exit()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print("SUPERVISED GRID — backbone-fair supervised arm")
    print("=" * 92)
    print(f"device       : {device}")
    print(f"dataset      : {args.dataset}")
    print(f"jepa_ckpt    : {args.jepa_ckpt}  (stock_stats source)")
    print(f"out_dir      : {args.out_dir}")
    print(f"backbone     : HorizonJEPAEncoder, S={args.S}, d_model={args.d_model}, "
          f"temporal_causal={args.temporal_causal}")

    print("\n[1/8] Loading raw LOBench dataset...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids_arr = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    min_spread_per_stock = raw["min_spread_z_per_stock"].astype(np.float32)
    n_stocks = int(min_spread_per_stock.shape[0])
    N, L = len(mid_z), book.shape[2]
    print(f"  N={N:,}  n_stocks={n_stocks}  L={L}")

    print("\n[2/8] Building derived raw features and vol mask...")
    raw_feat, _names = derive_raw_features_array(book, mid_z, stock_ids_arr, n_stocks)
    bid_v, ask_v = book[:, 0, :, 1], book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & \
               (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    max_h = max(max(FUTURE_HORIZONS), max(VOL_HORIZONS))
    print(f"  vol_mask pass: {vol_mask.sum():,}/{N:,}; max_h={max_h}")

    print("\n[3/8] Valid endpoints and grouped split...")
    valid_t = compute_valid_endpoints(stock_ids_arr, day_ids, args.K, max_h, vol_mask)
    train_pos, val_pos = grouped_split_by_stock_day(
        stock_ids_arr, day_ids, valid_t, args.val_frac, args.split_seed)
    rng = np.random.default_rng(args.seed)
    if args.max_train_samples > 0 and len(train_pos) > args.max_train_samples:
        train_pos = np.sort(rng.choice(train_pos, size=args.max_train_samples, replace=False))
    if args.max_val_samples > 0 and len(val_pos) > args.max_val_samples:
        val_pos = np.sort(rng.choice(val_pos, size=args.max_val_samples, replace=False))
    t_train, t_val = valid_t[train_pos], valid_t[val_pos]
    print(f"  train endpoints: {len(t_train):,}   val endpoints: {len(t_val):,}")

    print("\n[4/8] Building 22-target vectors and standardizing (TRAIN-only stats)...")
    y_train_raw, target_names = build_targets(book, mid_z, stock_ids_arr,
                                              t_train, raw_feat, min_spread_per_stock)
    y_val_raw, _ = build_targets(book, mid_z, stock_ids_arr,
                                 t_val, raw_feat, min_spread_per_stock)
    y_train, y_val, target_mean, target_std = standardize_targets(y_train_raw, y_val_raw)
    print(f"  y_train: {y_train.shape}   y_val: {y_val.shape}   names: {len(target_names)}")

    print("\n[5/8] Stock stats from JEPA checkpoint (shared normalization)...")
    if args.jepa_ckpt is None:
        raise SystemExit("--jepa_ckpt is required to inherit stock_stats for fair comparison.")
    stock_stats = load_stock_stats_from_jepa(args.jepa_ckpt, device)
    print(f"  loaded stock_stats with keys: {list(stock_stats.keys())}")

    print("\n[6/8] Datasets & loaders...")
    ds_train = SupervisedGridDataset(book, mid_z, stock_ids_arr, t_train,
                                     stock_stats, y_train, args.K)
    ds_val = SupervisedGridDataset(book, mid_z, stock_ids_arr, t_val,
                                   stock_stats, y_val, args.K)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                          persistent_workers=args.num_workers > 0, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0, drop_last=False)

    print("\n[7/8] Model...")
    enc_cfg = HorizonJEPAEncoderConfig(
        L=L, n_stocks=n_stocks, K=args.K, S=args.S, raw_per_token=args.raw_per_token,
        d_model=args.d_model,
        spatial_n_layers=args.spatial_n_layers, spatial_n_heads=args.spatial_n_heads,
        spatial_d_ffn=args.spatial_d_ffn,
        temporal_n_layers=args.temporal_n_layers, temporal_n_heads=args.temporal_n_heads,
        temporal_d_ffn=args.temporal_d_ffn, temporal_causal=args.temporal_causal,
        dropout=args.dropout,
    )
    readout_cfg = ReadoutConfig(d_model=args.d_model, out_dim=22, dropout=args.readout_dropout)
    model = SupervisedGrid(enc_cfg, readout_cfg).to(device)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_rd = sum(p.numel() for p in model.readout.parameters())
    print(f"  encoder params={n_enc:,}  readout params={n_rd:,}  total={n_enc+n_rd:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(dl_train))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    eta_min = args.lr * args.eta_min_frac

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return (eta_min / args.lr) + (1.0 - eta_min / args.lr) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("\n[8/8] Training...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_pred, tr_true = run_epoch(model, dl_train, optimizer, device,
                                              args.grad_clip, train=True)
        # The scheduler is per-step inside run_epoch's optimizer loop. Step here
        # `len(dl_train)` times to match. Simpler: step at the end of the epoch
        # by the number of steps performed. Equivalent to per-step for cosine.
        for _ in range(steps_per_epoch):
            scheduler.step()
        va_loss, va_pred, va_true = run_epoch(model, dl_val, None, device,
                                              args.grad_clip, train=False)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        tr_r2 = r2_per_target(tr_true, tr_pred)
        va_r2 = r2_per_target(va_true, va_pred)
        tr_s = summarize_r2(target_names, tr_r2)
        va_s = summarize_r2(target_names, va_r2)

        print(f"epoch {epoch:3d}/{args.epochs}  lr={lr_now:.2e}  [{dt:.1f}s]")
        print(f"  train  mse={tr_loss:.4f}  R²: all={tr_s['mean_all']:.4f} "
              f"future={tr_s['mean_future']:.4f}  vol={tr_s['mean_vol']:.4f}")
        print(f"  val    mse={va_loss:.4f}  R²: all={va_s['mean_all']:.4f} "
              f"future={va_s['mean_future']:.4f}  vol={va_s['mean_vol']:.4f}")

        history.append({"epoch": epoch, "lr": lr_now, "train_mse": tr_loss,
                        "val_mse": va_loss, "train_r2": tr_s, "val_r2": va_s})

        if va_loss < best_val_mse:
            best_val_mse = va_loss
            save_checkpoint(out_dir / "best.pt", epoch, model, optimizer,
                            enc_cfg, readout_cfg, target_names, target_mean, target_std,
                            stock_stats, vars(args),
                            {"mse": va_loss, **{f"r2_{k}": v for k, v in va_s.items()}})
            print(f"  -> saved best.pt  (val_mse={best_val_mse:.4f})")
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", epoch, model, optimizer,
                            enc_cfg, readout_cfg, target_names, target_mean, target_std,
                            stock_stats, vars(args),
                            {"mse": va_loss, **{f"r2_{k}": v for k, v in va_s.items()}})

    save_checkpoint(out_dir / "last.pt", args.epochs, model, optimizer,
                    enc_cfg, readout_cfg, target_names, target_mean, target_std,
                    stock_stats, vars(args),
                    {"mse": va_loss, **{f"r2_{k}": v for k, v in va_s.items()}})

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "target_names.json", "w") as f:
        json.dump(target_names, f, indent=2)
    print(f"\nDone. Best val_mse={best_val_mse:.4f}. History: {out_dir/'history.json'}")


if __name__ == "__main__":
    main()

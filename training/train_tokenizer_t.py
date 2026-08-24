"""
train_tokenizer_t.py — Trainer for A1-T temporal LOB tokenizer.

Combines into a single file:
  - K-window dataset construction with valid index precomputation
  - Future / vol / stats target computation (ground truth, train-only standardized)
  - Curriculum loss weighting (4 phases)
  - Training loop with per-component logging

Coherence with the probe:
  derive_raw_features_array() must produce feature LEVELS in EXACTLY the same
  order and definition as latent_raw_probe.derive_raw_features(). The probe
  computes targets as raw_feat[t+1, idx] - raw_feat[t, idx]; the trainer must
  produce raw_feat[t+h, idx] - raw_feat[t, idx] for h in future_horizons. Any
  drift between the two functions invalidates the latent->raw probe as a
  diagnostic of A1-T success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, *_THIS.parents]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.model_tokenizer_t import TokenizerConfigT, LOBAutoTokenizerT


VOL_CLIP = 5.0


# =============================================================================
#  Section 2 — Raw feature extraction (must match latent_raw_probe.py exactly)
# =============================================================================

# Single source of truth for feature names and order. The probe imports this
# same list (after this trainer is committed). Until then, both files declare
# the list independently — they MUST agree.
RAW_FEATURE_NAMES: List[str] = [
    "mid_z_level",
    "spread_z",
    "top_imbalance",
    "imbalance_top5",
    "imbalance_all",
    "log_depth_top5",
    "log_depth_all",
    "microprice_rel",
    "best_bid_rel",
    "best_ask_rel",
    "depth_width_z",
]


def derive_raw_features_array(
    book: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    n_stocks: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute long-memory raw feature LEVELS in the same order as the probe.

    Returns:
        feat:  (N, F) float32, where F = len(RAW_FEATURE_NAMES)
        names: list[str], the canonical feature ordering

    Each entry is a LEVEL value (not a delta). Targets are computed as
    raw_feat[t+h] - raw_feat[t] downstream.
    """
    bid_p = book[:, 0, :, 0]
    ask_p = book[:, 1, :, 0]
    bid_vz = book[:, 0, :, 1]
    ask_vz = book[:, 1, :, 1]
    L = book.shape[2]

    vol_min = np.zeros(n_stocks, dtype=np.float32)
    for s in range(n_stocks):
        m = stock_ids == s
        if m.any():
            vol_min[s] = float(min(bid_vz[m].min(), ask_vz[m].min()))
    vshift = vol_min[stock_ids][:, None]
    bid_v = np.maximum(bid_vz - vshift, 0.0)
    ask_v = np.maximum(ask_vz - vshift, 0.0)
    eps = 1e-8

    bid1 = bid_p[:, 0]
    ask1 = ask_p[:, 0]
    bv1 = bid_v[:, 0]
    av1 = ask_v[:, 0]

    spread_z = ask1 - bid1
    best_bid_rel = bid1 - mid_z
    best_ask_rel = ask1 - mid_z
    microprice = (ask1 * bv1 + bid1 * av1) / (bv1 + av1 + eps)
    microprice_rel = microprice - mid_z

    top_imb = (bv1 - av1) / (bv1 + av1 + eps)
    k5 = min(5, L)
    bid_top5 = bid_v[:, :k5].sum(axis=1)
    ask_top5 = ask_v[:, :k5].sum(axis=1)
    bid_all = bid_v.sum(axis=1)
    ask_all = ask_v.sum(axis=1)
    imb_top5 = (bid_top5 - ask_top5) / (bid_top5 + ask_top5 + eps)
    imb_all = (bid_all - ask_all) / (bid_all + ask_all + eps)
    log_depth_top5 = np.log1p(bid_top5 + ask_top5)
    log_depth_all = np.log1p(bid_all + ask_all)
    depth_width_z = ask_p[:, -1] - bid_p[:, -1]

    feats = np.stack([
        mid_z,
        spread_z,
        top_imb,
        imb_top5,
        imb_all,
        log_depth_top5,
        log_depth_all,
        microprice_rel,
        best_bid_rel,
        best_ask_rel,
        depth_width_z,
    ], axis=1).astype(np.float32)
    return feats, RAW_FEATURE_NAMES


# =============================================================================
#  Section 3 — Per-stock book normalization (replicates v1 LOBenchDataset)
# =============================================================================

def compute_stock_stats(
    book: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    n_stocks: int,
) -> Dict[str, np.ndarray]:
    """
    Compute the same per-stock stats as v1 LOBenchDataset.

    Returns dict with three arrays of shape (n_stocks,):
        depth_scale_per_stock
        vol_min_per_stock
        vol_scale_per_stock
    """
    depth_scale = np.zeros(n_stocks, dtype=np.float32)
    vol_min = np.zeros(n_stocks, dtype=np.float32)
    vol_scale = np.ones(n_stocks, dtype=np.float32)

    for s in range(n_stocks):
        m = stock_ids == s
        if not m.any():
            continue
        bid_p = book[m, 0, :, 0]
        ask_p = book[m, 1, :, 0]
        rel = np.concatenate([
            (bid_p - mid_z[m, None]).ravel(),
            (ask_p - mid_z[m, None]).ravel(),
        ])
        # depth_scale = std of rel-to-mid prices for this stock
        depth_scale[s] = float(rel.std() + 1e-6)

        bid_vz = book[m, 0, :, 1]
        ask_vz = book[m, 1, :, 1]
        vmin = float(min(bid_vz.min(), ask_vz.min()))
        vol_min[s] = vmin
        vshift_b = np.maximum(bid_vz - vmin, 0.0)
        vshift_a = np.maximum(ask_vz - vmin, 0.0)
        # vol_scale: per-stock spread of shifted volumes
        vs = float(np.concatenate([vshift_b.ravel(), vshift_a.ravel()]).std() + 1e-6)
        vol_scale[s] = vs

    return {
        "depth_scale_per_stock": depth_scale,
        "vol_min_per_stock": vol_min,
        "vol_scale_per_stock": vol_scale,
    }


def compute_stock_stats_train_only(
    book: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    valid_t_train: np.ndarray,
    n_stocks: int,
) -> Dict[str, np.ndarray]:
    """
    Compute per-stock normalization stats using train stock-day groups only.

    The grouped split keeps each (stock, day) entirely in train or val, so this
    avoids using validation-day marginal scales while sharing one normalization
    dictionary for both train and validation datasets.
    """
    max_day = int(day_ids.max()) + 1
    group_all = stock_ids.astype(np.int64) * max_day + day_ids.astype(np.int64)
    train_groups = np.unique(group_all[valid_t_train])
    train_mask = np.isin(group_all, train_groups)

    depth_scale = np.zeros(n_stocks, dtype=np.float32)
    vol_min = np.zeros(n_stocks, dtype=np.float32)
    vol_scale = np.ones(n_stocks, dtype=np.float32)

    for s_id in range(n_stocks):
        m = (stock_ids == s_id) & train_mask
        if not m.any():
            raise ValueError(
                f"No training snapshots for stock {s_id}; cannot compute train-only normalization stats. "
                "Use a less aggressive max_train_samples or adjust the grouped split."
            )

        bid_p = book[m, 0, :, 0]
        ask_p = book[m, 1, :, 0]
        rel = np.concatenate([
            (bid_p - mid_z[m, None]).ravel(),
            (ask_p - mid_z[m, None]).ravel(),
        ])
        depth_scale[s_id] = float(rel.std() + 1e-6)

        bid_vz = book[m, 0, :, 1]
        ask_vz = book[m, 1, :, 1]
        vmin = float(min(bid_vz.min(), ask_vz.min()))
        vol_min[s_id] = vmin
        vshift_b = np.maximum(bid_vz - vmin, 0.0)
        vshift_a = np.maximum(ask_vz - vmin, 0.0)
        vol_scale[s_id] = float(np.concatenate([vshift_b.ravel(), vshift_a.ravel()]).std() + 1e-6)

    return {
        "depth_scale_per_stock": depth_scale,
        "vol_min_per_stock": vol_min,
        "vol_scale_per_stock": vol_scale,
    }


def normalize_book_window(
    book_raw_window: np.ndarray,
    mid_z_window: np.ndarray,
    stock_id: int,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Normalize a (K, 2, L, 2) raw book window for a single stock.
    Replicates v1 normalization but applied per-snapshot in the window.
    """
    depth_s = float(stats["depth_scale_per_stock"][stock_id])
    vmin_s = float(stats["vol_min_per_stock"][stock_id])
    vscale_s = float(stats["vol_scale_per_stock"][stock_id])

    bk = book_raw_window.astype(np.float32, copy=True)

    # Prices: (price - mid) / depth, clipped at ±3 depth.
    for side in range(2):
        rel = bk[:, side, :, 0] - mid_z_window[:, None]
        clip = 3.0 * depth_s
        rel = np.clip(rel, -clip, clip)
        bk[:, side, :, 0] = rel / depth_s

    # Volumes: (vol - vmin) / vscale.
    bk[:, :, :, 1] = (book_raw_window[:, :, :, 1].astype(np.float32) - vmin_s) / vscale_s
    return bk


# =============================================================================
#  Section 4 — Valid index precomputation
# =============================================================================

def compute_valid_endpoints(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    K: int,
    max_horizon: int,
    vol_mask: np.ndarray,
) -> np.ndarray:
    """
    Vectorized precomputation of valid endpoint indices t such that:
        - [t-K+1, ..., t+max_horizon] are entirely within the same (stock, day);
        - all snapshots in that range pass vol_clip (vol_mask is True).

    Args:
        stock_ids:  (N,) int64
        day_ids:    (N,) int64
        K:          window length (past)
        max_horizon: largest future offset needed
        vol_mask:   (N,) bool, True if snapshot passes vol_clip

    Returns:
        valid_t: int64 array of valid endpoint indices.
    """
    N = len(stock_ids)
    if N == 0:
        return np.array([], dtype=np.int64)

    # Composite group id per snapshot (stock, day) -> single int.
    # Use a stable mapping: g = stock_ids * (max_day+1) + day_ids.
    max_day = int(day_ids.max()) + 1
    g = (stock_ids.astype(np.int64) * max_day) + day_ids.astype(np.int64)

    # For each position i, span_back[i] = (i - first index of current block).
    # Vectorized via cumulative count since group change.
    block_change = np.zeros(N, dtype=bool)
    block_change[0] = True
    block_change[1:] = g[1:] != g[:-1]
    block_idx = np.cumsum(block_change) - 1                 # group label (0..n_blocks-1)
    block_starts = np.where(block_change)[0]
    block_ends = np.concatenate([block_starts[1:], np.array([N])])  # exclusive

    # span_back[i] = i - block_starts[block_idx[i]]
    span_back = np.arange(N) - block_starts[block_idx]
    # span_fwd[i] = block_ends[block_idx[i]] - 1 - i
    span_fwd = (block_ends[block_idx] - 1) - np.arange(N)

    # Endpoint t valid w.r.t. boundaries iff:
    #   span_back[t] >= K-1 AND span_fwd[t] >= max_horizon
    bound_ok = (span_back >= (K - 1)) & (span_fwd >= max_horizon)

    # Vol-mask cumulative sum: vol_window_sum[i] = sum(vol_mask[i-K+1..i+max_horizon])
    # We need this == K + max_horizon for valid t.
    win_len = K + max_horizon
    vol_int = vol_mask.astype(np.int32)
    cum = np.concatenate([[0], np.cumsum(vol_int)])         # length N+1
    # window [a, b] inclusive with a = t-K+1, b = t+max_horizon
    # Sum = cum[b+1] - cum[a]
    # We compute only for indices where bound_ok already holds.
    valid_candidates = np.where(bound_ok)[0]
    a = valid_candidates - (K - 1)
    b = valid_candidates + max_horizon
    win_sum = cum[b + 1] - cum[a]
    vol_ok = win_sum == win_len

    valid_t = valid_candidates[vol_ok]
    return valid_t.astype(np.int64)


# =============================================================================
#  Section 5 — Target computation (ground truth, pre-standardization)
# =============================================================================

def _features_to_indices(features: List[str], all_names: List[str]) -> List[int]:
    """Map target names like 'd_spread_z' to indices in raw_feat."""
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    out = []
    for f in features:
        # Strip 'd_' prefix if present (target = delta of feature).
        base = f[2:] if f.startswith("d_") else f
        if base not in name_to_idx:
            raise ValueError(f"Feature {f!r} (base {base!r}) not in raw feature set {all_names}")
        out.append(name_to_idx[base])
    return out


def compute_future_feature_targets(
    raw_feat: np.ndarray,
    valid_t: np.ndarray,
    future_features: List[str],
    horizons: List[int],
) -> np.ndarray:
    """
    For each valid endpoint t and each (feature, horizon) pair, compute:
        target[t, f, h] = raw_feat[t+h, idx_of_f] - raw_feat[t, idx_of_f]

    Flattened ordering: (feature_outer, horizon_inner).
    For features=[A,B,C] and horizons=[1,5,10,20], the output column order is:
        A@1, A@5, A@10, A@20, B@1, B@5, B@10, B@20, C@1, C@5, C@10, C@20

    This is the order that FutureFeatureHead's Linear output is interpreted as.

    Returns:
        future_targets: (n_valid, n_features × n_horizons) float32
    """
    feat_idx = _features_to_indices(future_features, RAW_FEATURE_NAMES)
    n_valid = len(valid_t)
    F_count = len(future_features)
    H_count = len(horizons)
    out = np.empty((n_valid, F_count * H_count), dtype=np.float32)

    feat_t = raw_feat[valid_t][:, feat_idx]                  # (n_valid, F_count)
    col = 0
    for f_pos in range(F_count):
        idx = feat_idx[f_pos]
        for h in horizons:
            feat_th = raw_feat[valid_t + h, idx]
            out[:, col] = feat_th - feat_t[:, f_pos]
            col += 1
    return out


def compute_vol_targets(
    mid_z: np.ndarray,
    valid_t: np.ndarray,
    vol_horizons: List[int],
    min_spread_per_stock: np.ndarray,
    stock_ids: np.ndarray,
) -> np.ndarray:
    """
    For each valid t and each H in vol_horizons:
        rv = std(diff(mid_z[t : t+H+1]))
        vol_target[t, H] = rv / min_spread_z[stock(t)]

    Tick-physical units (matches v1 realized_vol_K convention).

    Returns:
        vol_targets: (n_valid, n_vol_horizons) float32
    """
    n_valid = len(valid_t)
    H_count = len(vol_horizons)
    out = np.empty((n_valid, H_count), dtype=np.float32)
    min_spread_per_t = min_spread_per_stock[stock_ids[valid_t]].astype(np.float32)

    for h_pos, H in enumerate(vol_horizons):
        # mid_z window of size H+1 ending at t+H. Diffs of size H.
        # Vectorize: build (n_valid, H+1) window via broadcasting.
        offsets = np.arange(H + 1)
        mid_window = mid_z[valid_t[:, None] + offsets[None, :]]
        diffs = np.diff(mid_window, axis=1)
        rv = diffs.std(axis=1)
        out[:, h_pos] = (rv / np.maximum(min_spread_per_t, 1e-8)).astype(np.float32)
    return out


def compute_stats_targets_normalized(
    book: np.ndarray,
    mid_z: np.ndarray,
    stock_ids: np.ndarray,
    valid_t: np.ndarray,
    stats: Dict[str, np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute the 4 immediate scalars [spread_norm, imbalance, bid_conc, ask_conc]
    on the NORMALIZED snapshot t for every valid endpoint.

    Done in batch with explicit vectorization to avoid touching __getitem__.
    Returns: (n_valid, 4) float32.
    """
    book_t = book[valid_t]                                   # (B, 2, L, 2)
    mid_t = mid_z[valid_t]                                   # (B,)
    s_t = stock_ids[valid_t]                                 # (B,)

    depth_s = stats["depth_scale_per_stock"][s_t].astype(np.float32)[:, None]      # (B, 1)
    vmin_s = stats["vol_min_per_stock"][s_t].astype(np.float32)[:, None, None]     # (B, 1, 1)
    vscale_s = stats["vol_scale_per_stock"][s_t].astype(np.float32)[:, None, None] # (B, 1, 1)

    bk = book_t.astype(np.float32, copy=True)
    # Prices: (price - mid) / depth, clipped at ±3 depth.
    for side in range(2):
        rel = bk[:, side, :, 0] - mid_t[:, None]             # (B, L)
        clip = 3.0 * depth_s                                 # (B, 1) broadcasts to (B, L)
        rel = np.clip(rel, -clip, clip)
        bk[:, side, :, 0] = rel / depth_s
    # Volumes: (vol - vmin) / vscale.
    bk[:, :, :, 1] = (book_t[:, :, :, 1].astype(np.float32) - vmin_s) / vscale_s

    bid_p0 = bk[:, 0, 0, 0]
    ask_p0 = bk[:, 1, 0, 0]
    bid_v0 = bk[:, 0, 0, 1]
    ask_v0 = bk[:, 1, 0, 1]

    spread_norm = ask_p0 - bid_p0
    imbalance = (bid_v0 - ask_v0) / (bid_v0 + ask_v0 + eps)
    bid_vtot = bk[:, 0, :, 1].sum(axis=1) + eps
    ask_vtot = bk[:, 1, :, 1].sum(axis=1) + eps
    bid_conc = bid_v0 / bid_vtot
    ask_conc = ask_v0 / ask_vtot

    return np.stack([spread_norm, imbalance, bid_conc, ask_conc], axis=1).astype(np.float32)


# =============================================================================
#  Section 6 — Target standardization (train-only stats)
# =============================================================================

def fit_target_standardizer(targets_train: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = targets_train.mean(axis=0).astype(np.float32)
    std = targets_train.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return mean, std


def apply_standardizer(targets: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((targets - mean) / std).astype(np.float32)


# =============================================================================
#  Section 7 — Dataset
# =============================================================================

class LOBenchKWindowDataset(Dataset):
    """
    Yields (books_window, stock_id, stats_t, future_targets, vol_targets)
    for each valid endpoint.
    """

    def __init__(
        self,
        book_raw: np.ndarray,
        mid_z: np.ndarray,
        stock_ids: np.ndarray,
        valid_t: np.ndarray,
        future_targets: np.ndarray,
        vol_targets: np.ndarray,
        stats_targets: np.ndarray,
        stock_stats: Dict[str, np.ndarray],
        K: int,
    ):
        self.book_raw = book_raw
        self.mid_z = mid_z
        self.stock_ids = stock_ids
        self.valid_t = valid_t
        self.future_targets = future_targets
        self.vol_targets = vol_targets
        self.stats_targets = stats_targets
        self.stock_stats = stock_stats
        self.K = K

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, idx: int):
        t = int(self.valid_t[idx])
        s = int(self.stock_ids[t])
        K = self.K
        # Window [t-K+1, ..., t]
        book_win = self.book_raw[t - K + 1 : t + 1]          # (K, 2, L, 2)
        mid_win = self.mid_z[t - K + 1 : t + 1]              # (K,)
        book_norm = normalize_book_window(book_win, mid_win, s, self.stock_stats)

        return (
            torch.from_numpy(book_norm).float(),
            torch.tensor(s, dtype=torch.long),
            torch.from_numpy(self.stats_targets[idx]).float(),
            torch.from_numpy(self.future_targets[idx]).float(),
            torch.from_numpy(self.vol_targets[idx]).float(),
        )


# =============================================================================
#  Section 8 — Train/val split (grouped by stock-day)
# =============================================================================

def grouped_split_by_stock_day(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    valid_t: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_idx, val_idx) as positions in valid_t (not raw indices).
    A stock-day appears entirely in train or entirely in val.
    """
    rng = np.random.default_rng(seed)
    s_t = stock_ids[valid_t]
    d_t = day_ids[valid_t]
    composite = s_t.astype(np.int64) * (int(d_t.max()) + 1) + d_t.astype(np.int64)
    unique_groups = np.unique(composite)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(round(val_frac * len(unique_groups))))
    val_groups = set(unique_groups[:n_val_groups].tolist())
    val_mask = np.array([g in val_groups for g in composite])
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx


# =============================================================================
#  Section 9 — Curriculum
# =============================================================================

def curriculum_weights(epoch: int, args, cfg: TokenizerConfigT) -> Dict[str, float]:
    """
    Returns a dict of weights for the 7 losses for the current epoch.

    Phases:
      phase 1 [1 .. p2_start-1]:        recon + struct
      phase 2 [p2_start .. p3_start-1]: + stats + cov
      phase 3 [p3_start .. p4_start-1]: + future + vol
      phase 4 [p4_start ..]:            + dyn

    Each loss's "on" weight is its cfg default; "off" weight is 0.0.
    """
    p2 = args.phase2_start
    p3 = args.phase3_start
    p4 = args.phase4_start

    on_recon = True
    on_struct = True
    on_stats = epoch >= p2
    on_cov = epoch >= p2
    on_future = epoch >= p3
    on_vol = epoch >= p3
    on_dyn = epoch >= p4

    return {
        "recon": cfg.w_recon if on_recon else 0.0,
        "struct": cfg.w_struct if on_struct else 0.0,
        "stats": cfg.w_stats if on_stats else 0.0,
        "future": cfg.w_future if on_future else 0.0,
        "vol": cfg.w_vol if on_vol else 0.0,
        "cov": cfg.w_cov if on_cov else 0.0,
        "dyn": cfg.w_dyn if on_dyn else 0.0,
    }


# =============================================================================
#  Section 10 — Train loop
# =============================================================================

def run_epoch(
    model: LOBAutoTokenizerT,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    weights: Dict[str, float],
    train: bool,
    grad_clip: float,
) -> Dict[str, float]:
    model.train(train)
    sums = {k: 0.0 for k in ["recon", "struct", "stats", "future", "vol", "cov", "dyn", "total"]}
    n_total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            books_window, stock_ids, stats_t, future_targets, vol_targets = batch
            books_window = books_window.to(device, non_blocking=True)
            stock_ids = stock_ids.to(device, non_blocking=True)
            stats_t = stats_t.to(device, non_blocking=True)
            future_targets = future_targets.to(device, non_blocking=True)
            vol_targets = vol_targets.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            out = model(books_window, stock_ids, stats_t, future_targets, vol_targets)
            losses = out["losses"]
            total = (
                weights["recon"] * losses["recon"]
                + weights["struct"] * losses["struct"]
                + weights["stats"] * losses["stats"]
                + weights["future"] * losses["future"]
                + weights["vol"] * losses["vol"]
                + weights["cov"] * losses["cov"]
                + weights["dyn"] * losses["dyn"]
            )

            if train:
                total.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            b = books_window.shape[0]
            n_total += b
            for k in ["recon", "struct", "stats", "future", "vol", "cov", "dyn"]:
                sums[k] += float(losses[k].item()) * b
            sums["total"] += float(total.item()) * b

    return {k: v / max(n_total, 1) for k, v in sums.items()}


def save_ckpt(
    path: Path,
    model: LOBAutoTokenizerT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_metrics: Dict[str, float],
    stock_stats: Dict[str, np.ndarray],
    target_standardizers: Dict[str, Dict[str, np.ndarray]],
    args,
    cfg: TokenizerConfigT,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "val_metrics": val_metrics,
        "cfg": cfg.to_dict(),
        "stock_stats": {k: v.tolist() for k, v in stock_stats.items()},
        "target_standardizers": {
            tk: {sk: sv.tolist() for sk, sv in tv.items()}
            for tk, tv in target_standardizers.items()
        },
        "future_features": cfg.future_features,
        "future_horizons": cfg.future_horizons,
        "vol_horizons": cfg.vol_horizons,
        "raw_feature_names": RAW_FEATURE_NAMES,
        "train_args": vars(args),
    }
    torch.save(ckpt, path)


# =============================================================================
#  Section 11 — Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--ckpt_dir", default="checkpoints/world_model/encoder/a1_t", type=str)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vol_clip", type=float, default=VOL_CLIP)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)

    # Architecture / config (override TokenizerConfigT defaults)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--d_latent", type=int, default=32)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dyn_max_batch", type=int, default=512)

    # Loss weight overrides (defaults from cfg)
    p.add_argument("--w_recon", type=float, default=None)
    p.add_argument("--w_struct", type=float, default=None)
    p.add_argument("--w_stats", type=float, default=None)
    p.add_argument("--w_future", type=float, default=None)
    p.add_argument("--w_vol", type=float, default=None)
    p.add_argument("--w_cov", type=float, default=None)
    p.add_argument("--w_dyn", type=float, default=None)

    # Curriculum: epoch (1-indexed) at which each phase begins.
    p.add_argument("--phase2_start", type=int, default=4)
    p.add_argument("--phase3_start", type=int, default=7)
    p.add_argument("--phase4_start", type=int, default=13)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    # =========== Load raw LOBench ===========
    print("\n[1/12] Loading raw LOBench dataset...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)        # (N, 2, L, 2)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    min_spread_per_stock = raw["min_spread_z_per_stock"].astype(np.float32, copy=False)
    n_stocks = int(min_spread_per_stock.shape[0])
    N = len(mid_z)
    L = book.shape[2]
    print(f"  N={N:,}  n_stocks={n_stocks}  L={L}")

    # =========== Per-stock stats ===========
    print("\n[2/12] Computing per-stock normalization stats...")
    stock_stats = compute_stock_stats(book, mid_z, stock_ids, n_stocks)
    for s in range(n_stocks):
        print(f"  stock {s}: depth={stock_stats['depth_scale_per_stock'][s]:.4f}  "
              f"vol_min={stock_stats['vol_min_per_stock'][s]:+.3f}  "
              f"vol_scale={stock_stats['vol_scale_per_stock'][s]:.3f}")

    # =========== Raw feature levels ===========
    print("\n[3/12] Deriving raw feature levels (probe-aligned)...")
    raw_feat, raw_names = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)
    print(f"  raw_feat shape: {raw_feat.shape}  features: {raw_names}")

    # =========== Vol mask ===========
    print("\n[4/12] Computing vol_clip mask...")
    bid_v = book[:, 0, :, 1]
    ask_v = book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    n_pass = int(vol_mask.sum())
    print(f"  pass vol_clip: {n_pass:,}/{N:,} ({100.0*n_pass/N:.2f}%)")

    # =========== Build cfg ===========
    cfg = TokenizerConfigT()
    cfg.K = args.K
    cfg.L = L
    cfg.n_stocks = n_stocks
    cfg.d_latent = args.d_latent
    cfg.d_model = args.d_model
    cfg.dropout = args.dropout
    cfg.dyn_max_batch = args.dyn_max_batch
    if args.w_recon is not None: cfg.w_recon = args.w_recon
    if args.w_struct is not None: cfg.w_struct = args.w_struct
    if args.w_stats is not None: cfg.w_stats = args.w_stats
    if args.w_future is not None: cfg.w_future = args.w_future
    if args.w_vol is not None: cfg.w_vol = args.w_vol
    if args.w_cov is not None: cfg.w_cov = args.w_cov
    if args.w_dyn is not None: cfg.w_dyn = args.w_dyn

    max_horizon = max(max(cfg.future_horizons), max(cfg.vol_horizons))
    print(f"\n[5/12] Curriculum config: phases at epochs {args.phase2_start}/{args.phase3_start}/{args.phase4_start}")
    print(f"  K={cfg.K}  d_latent={cfg.d_latent}  d_model={cfg.d_model}  max_horizon={max_horizon}")

    # =========== Valid endpoints ===========
    print("\n[6/12] Computing valid endpoints...")
    t0 = time.time()
    valid_t = compute_valid_endpoints(stock_ids, day_ids, cfg.K, max_horizon, vol_mask)
    print(f"  valid_t: {len(valid_t):,}  ({time.time()-t0:.1f}s)")

    # =========== Targets (raw, pre-standardization) ===========
    print("\n[7/12] Computing future feature targets (raw)...")
    t0 = time.time()
    future_targets_raw = compute_future_feature_targets(
        raw_feat, valid_t, cfg.future_features, cfg.future_horizons,
    )
    print(f"  future_targets shape: {future_targets_raw.shape}  ({time.time()-t0:.1f}s)")

    print("\n[8/12] Computing vol targets (raw)...")
    t0 = time.time()
    vol_targets_raw = compute_vol_targets(
        mid_z, valid_t, cfg.vol_horizons, min_spread_per_stock, stock_ids,
    )
    print(f"  vol_targets shape: {vol_targets_raw.shape}  ({time.time()-t0:.1f}s)")

    print("\n[9/12] Computing stats targets (normalized snapshot t)...")
    t0 = time.time()
    stats_targets_raw = compute_stats_targets_normalized(
        book, mid_z, stock_ids, valid_t, stock_stats,
    )
    print(f"  stats_targets shape: {stats_targets_raw.shape}  ({time.time()-t0:.1f}s)")

    # =========== Train/val split ===========
    print("\n[10/12] Grouped split by stock-day...")
    train_idx, val_idx = grouped_split_by_stock_day(
        stock_ids, day_ids, valid_t, args.val_frac, args.seed,
    )
    if args.max_train_samples and args.max_train_samples > 0 and len(train_idx) > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_idx = np.sort(rng.choice(train_idx, size=args.max_train_samples, replace=False))
    if args.max_val_samples and args.max_val_samples > 0 and len(val_idx) > args.max_val_samples:
        rng = np.random.default_rng(args.seed + 1)
        val_idx = np.sort(rng.choice(val_idx, size=args.max_val_samples, replace=False))
    print(f"  train: {len(train_idx):,}  val: {len(val_idx):,}")

    # =========== Standardize targets (TRAIN ONLY) ===========
    print("\n[11/12] Fitting target standardizers on train only...")
    fut_mean, fut_std = fit_target_standardizer(future_targets_raw[train_idx])
    vol_mean, vol_std = fit_target_standardizer(vol_targets_raw[train_idx])
    sts_mean, sts_std = fit_target_standardizer(stats_targets_raw[train_idx])

    future_targets = apply_standardizer(future_targets_raw, fut_mean, fut_std)
    vol_targets = apply_standardizer(vol_targets_raw, vol_mean, vol_std)
    stats_targets = apply_standardizer(stats_targets_raw, sts_mean, sts_std)

    target_standardizers = {
        "future": {"mean": fut_mean, "std": fut_std},
        "vol": {"mean": vol_mean, "std": vol_std},
        "stats": {"mean": sts_mean, "std": sts_std},
    }
    print(f"  future target std (per dim, sample): {fut_std[:5]}")
    print(f"  vol target std: {vol_std}")
    print(f"  stats target std: {sts_std}")

    # =========== Datasets / loaders ===========
    train_ds = LOBenchKWindowDataset(
        book, mid_z, stock_ids, valid_t[train_idx],
        future_targets[train_idx], vol_targets[train_idx], stats_targets[train_idx],
        stock_stats, cfg.K,
    )
    val_ds = LOBenchKWindowDataset(
        book, mid_z, stock_ids, valid_t[val_idx],
        future_targets[val_idx], vol_targets[val_idx], stats_targets[val_idx],
        stock_stats, cfg.K,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    # =========== Model / optimizer / scheduler ===========
    print("\n[12/12] Building model...")
    model = LOBAutoTokenizerT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({
            "args": vars(args),
            "cfg": cfg.to_dict(),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_valid_total": int(len(valid_t)),
        }, f, indent=2)

    # =========== Train loop ===========
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    last_phase = -1
    print("\n" + "=" * 110)
    print("TRAINING")
    print("=" * 110)
    print(f"Curriculum phases start at epochs: 1 / {args.phase2_start} / {args.phase3_start} / {args.phase4_start}")
    print(f"Note: best_val and patience are reset at each phase boundary so curriculum")
    print(f"      activations don't trigger spurious early stopping.")

    def _phase_for(ep: int) -> int:
        if ep >= args.phase4_start: return 4
        if ep >= args.phase3_start: return 3
        if ep >= args.phase2_start: return 2
        return 1

    for epoch in range(1, args.epochs + 1):
        weights = curriculum_weights(epoch, args, cfg)
        active = [k for k, v in weights.items() if v > 0]

        # Reset best/patience when phase changes (val_total has discontinuous
        # meaning across phases when new losses turn on).
        phase = _phase_for(epoch)
        if phase != last_phase:
            if last_phase != -1:
                print(f"  [curriculum] phase change {last_phase} -> {phase}: "
                      f"resetting best_val and patience counter.")
            best_val = float("inf")
            epochs_no_improve = 0
            last_phase = phase

        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, device, weights, train=True,
                       grad_clip=args.grad_clip)
        va = run_epoch(model, val_loader, optimizer=None, device=device, weights=weights,
                       train=False, grad_clip=0.0)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  ph={phase}  "
            f"active={active}  "
            f"recon t/v {tr['recon']:.4f}/{va['recon']:.4f}  "
            f"struct {tr['struct']:.4f}/{va['struct']:.4f}  "
            f"stats {tr['stats']:.4f}/{va['stats']:.4f}  "
            f"future {tr['future']:.4f}/{va['future']:.4f}  "
            f"vol {tr['vol']:.4f}/{va['vol']:.4f}  "
            f"cov {tr['cov']:.4f}/{va['cov']:.4f}  "
            f"dyn {tr['dyn']:.4f}/{va['dyn']:.4f}  "
            f"total t/v {tr['total']:.4f}/{va['total']:.4f}  "
            f"lr={lr_now:.2e}  t={dt:.1f}s"
        )

        if va["total"] < best_val:
            best_val = va["total"]
            best_epoch = epoch
            epochs_no_improve = 0
            save_ckpt(ckpt_dir / "encoder_best.pt", model, optimizer, scheduler,
                      epoch, va, stock_stats, target_standardizers, args, cfg)
        else:
            epochs_no_improve += 1

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_ckpt(ckpt_dir / f"encoder_epoch_{epoch}.pt", model, optimizer, scheduler,
                      epoch, va, stock_stats, target_standardizers, args, cfg)
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch {best_epoch}, val {best_val:.4f}, phase {phase})")
            break

    print(f"\nBest val total: {best_val:.6f} at epoch {best_epoch}")
    print(f"Saved best: {ckpt_dir / 'encoder_best.pt'}")


if __name__ == "__main__":
    main()

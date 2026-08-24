"""
eval_tokenizer_t.py — Evaluation suite for A1-T temporal LOB tokenizer.

This evaluator is the A1-T counterpart of eval_encoder_lobench_tokenizer.py.
It keeps the old tokenizer checks where meaningful and adds A1-T specific checks:

Old / geometry checks:
  - current-snapshot reconstruction: price + volume
  - decoder structural validity: monotonic ladder, crossed spread, side violations
  - immediate stats-head R²
  - latent covariance/correlation/PCA/effective rank
  - kNN consistency / injectivity diagnostics
  - latent autocorrelation at multiple lags

A1-T temporal checks:
  - future feature head R² by feature and horizon
  - realized volatility head R² by horizon
  - dynamics-aware metric alignment diagnostic on validation batch

Typical use:
  python -m scripts.evaluation.eval_tokenizer_t \
    --ckpt checkpoints/tokenizer_t/v1/encoder_best.pt \
    --dataset data/lobench_processed.npz \
    --n_samples 50000 \
    --out_dir validation/tokenizer_t/v1_eval \
    --no_tsne

Notes:
  - The evaluator uses the stock_stats and target_standardizers saved inside
    the A1-T checkpoint.
  - By default, endpoint validity follows the A1-T trainer used for the current
    checkpoint: raw-volume vol_clip and future horizon availability. Use
    --vol_filter normalized to evaluate under v1/v2-style normalized-volume
    filtering.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist
except Exception as e:
    raise RuntimeError("This evaluator requires sklearn and scipy.") from e

# --- Robust project-root import setup ---
from pathlib import Path
import sys

_THIS = Path(__file__).resolve()

# Add current dir + all parents up to filesystem root.
# This lets the script work whether launched from repo root, scripts/, or directly.
for _p in [_THIS.parent, *_THIS.parents]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.model_tokenizer_t import TokenizerConfigT, LOBAutoTokenizerT
REGIME_NAMES = ["low_vol", "mid_vol", "high_vol"]


# =============================================================================
# Raw features, normalization, valid endpoints
# =============================================================================

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


def _as_np_stats(stats_obj: Dict) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats_obj.items()}


def load_a1t(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TokenizerConfigT.from_dict(ckpt.get("cfg", {}))
    model = LOBAutoTokenizerT(cfg).to(device)
    state = ckpt.get("model_state_dict", None)
    if state is None:
        raise ValueError("Checkpoint does not contain model_state_dict. Is this an A1-T checkpoint?")
    model.load_state_dict(state, strict=True)
    model.eval()

    stock_stats = _as_np_stats(ckpt["stock_stats"])
    target_standardizers = ckpt.get("target_standardizers", {})
    print(f"A1-T loaded: {ckpt_path}")
    print(f"  epoch={ckpt.get('epoch', 'n/a')}  val_metrics={ckpt.get('val_metrics', {})}")
    print(f"  K={cfg.K}  d_latent={cfg.d_latent}  d_model={cfg.d_model}  n_stocks={cfg.n_stocks}")
    return model, cfg, ckpt, stock_stats, target_standardizers


def normalize_book_window(book_raw_window: np.ndarray, mid_z_window: np.ndarray, stock_id: int, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize a (K,2,L,2) raw book window with A1-T/v1-style stock stats."""
    depth_s = float(stats["depth_scale_per_stock"][stock_id])
    vmin_s = float(stats["vol_min_per_stock"][stock_id])
    vscale_s = float(stats["vol_scale_per_stock"][stock_id])
    bk = book_raw_window.astype(np.float32, copy=True)
    for side in range(2):
        rel = bk[:, side, :, 0] - mid_z_window[:, None]
        clip = 3.0 * depth_s
        rel = np.clip(rel, -clip, clip)
        bk[:, side, :, 0] = rel / depth_s
    bk[:, :, :, 1] = (book_raw_window[:, :, :, 1].astype(np.float32) - vmin_s) / vscale_s
    return bk


def normalize_book_batch(book_raw: np.ndarray, mid_z: np.ndarray, stock_ids: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize (N,2,L,2) snapshots; keeps same indices."""
    bk = book_raw.astype(np.float32, copy=True)
    depth = stats["depth_scale_per_stock"][stock_ids].astype(np.float32)
    vmin = stats["vol_min_per_stock"][stock_ids].astype(np.float32)
    vscale = stats["vol_scale_per_stock"][stock_ids].astype(np.float32)
    for side in range(2):
        rel = bk[:, side, :, 0] - mid_z[:, None]
        clip = (3.0 * depth)[:, None]
        rel = np.clip(rel, -clip, clip)
        bk[:, side, :, 0] = rel / depth[:, None]
    bk[:, :, :, 1] = (book_raw[:, :, :, 1].astype(np.float32) - vmin[:, None, None]) / vscale[:, None, None]
    return bk


def derive_raw_features_array(book: np.ndarray, mid_z: np.ndarray, stock_ids: np.ndarray, n_stocks: int) -> Tuple[np.ndarray, List[str]]:
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


def compute_valid_endpoints(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    K: int,
    max_horizon: int,
    vol_mask: np.ndarray,
) -> np.ndarray:
    """Same endpoint logic as the A1-T trainer: [t-K+1, ..., t+max_horizon] valid."""
    N = len(stock_ids)
    max_day = int(day_ids.max()) + 1
    g = stock_ids.astype(np.int64) * max_day + day_ids.astype(np.int64)

    block_change = np.zeros(N, dtype=bool)
    block_change[0] = True
    block_change[1:] = g[1:] != g[:-1]
    block_idx = np.cumsum(block_change) - 1
    block_starts = np.where(block_change)[0]
    block_ends = np.concatenate([block_starts[1:], np.array([N])])

    span_back = np.arange(N) - block_starts[block_idx]
    span_fwd = (block_ends[block_idx] - 1) - np.arange(N)
    bound_ok = (span_back >= (K - 1)) & (span_fwd >= max_horizon)

    win_len = K + max_horizon
    cum = np.concatenate([[0], np.cumsum(vol_mask.astype(np.int32))])
    candidates = np.where(bound_ok)[0]
    a = candidates - (K - 1)
    b = candidates + max_horizon
    win_sum = cum[b + 1] - cum[a]
    return candidates[win_sum == win_len].astype(np.int64)


def compute_vol_mask(raw_book: np.ndarray, stock_ids: np.ndarray, stats: Dict[str, np.ndarray], vol_clip: float, mode: str) -> np.ndarray:
    if mode == "raw":
        bid_v = raw_book[:, 0, :, 1]
        ask_v = raw_book[:, 1, :, 1]
        return (np.abs(bid_v).max(axis=1) <= vol_clip) & (np.abs(ask_v).max(axis=1) <= vol_clip)
    if mode == "normalized":
        vmin = stats["vol_min_per_stock"][stock_ids]
        vscale = stats["vol_scale_per_stock"][stock_ids]
        vol_norm = (raw_book[:, :, :, 1] - vmin[:, None, None]) / vscale[:, None, None]
        return (vol_norm <= vol_clip).all(axis=(1, 2))
    raise ValueError(f"Unknown vol_filter mode: {mode}")


def grouped_split_by_stock_day(stock_ids: np.ndarray, day_ids: np.ndarray, valid_t: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    max_day = int(day_ids.max()) + 1
    composite = stock_ids[valid_t].astype(np.int64) * max_day + day_ids[valid_t].astype(np.int64)
    unique_groups = np.unique(composite)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(round(val_frac * len(unique_groups))))
    val_groups = set(unique_groups[:n_val_groups].tolist())
    val_mask = np.array([g in val_groups for g in composite])
    train_pos = np.where(~val_mask)[0]
    val_pos = np.where(val_mask)[0]
    return train_pos, val_pos, unique_groups[:n_val_groups]


def _features_to_indices(features: List[str], all_names: List[str]) -> List[int]:
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    out = []
    for f in features:
        base = f[2:] if f.startswith("d_") else f
        if base not in name_to_idx:
            raise ValueError(f"Feature {f!r} not found in {all_names}")
        out.append(name_to_idx[base])
    return out


def compute_future_feature_targets(raw_feat: np.ndarray, valid_t: np.ndarray, future_features: List[str], horizons: List[int]) -> np.ndarray:
    feat_idx = _features_to_indices(future_features, RAW_FEATURE_NAMES)
    out = np.empty((len(valid_t), len(feat_idx) * len(horizons)), dtype=np.float32)
    feat_t = raw_feat[valid_t][:, feat_idx]
    col = 0
    for f_pos, idx in enumerate(feat_idx):
        for h in horizons:
            out[:, col] = raw_feat[valid_t + h, idx] - feat_t[:, f_pos]
            col += 1
    return out


def compute_vol_targets(mid_z: np.ndarray, valid_t: np.ndarray, vol_horizons: List[int], min_spread_per_stock: np.ndarray, stock_ids: np.ndarray) -> np.ndarray:
    out = np.empty((len(valid_t), len(vol_horizons)), dtype=np.float32)
    min_spread = min_spread_per_stock[stock_ids[valid_t]].astype(np.float32)
    for j, H in enumerate(vol_horizons):
        offs = np.arange(H + 1)
        mid_win = mid_z[valid_t[:, None] + offs[None, :]]
        rv = np.diff(mid_win, axis=1).std(axis=1)
        out[:, j] = (rv / np.maximum(min_spread, 1e-8)).astype(np.float32)
    return out


def compute_stats_targets_normalized(book_norm_t: np.ndarray) -> np.ndarray:
    eps = 1e-8
    bid_p0 = book_norm_t[:, 0, 0, 0]
    ask_p0 = book_norm_t[:, 1, 0, 0]
    bid_v0 = book_norm_t[:, 0, 0, 1]
    ask_v0 = book_norm_t[:, 1, 0, 1]
    spread = ask_p0 - bid_p0
    imbalance = (bid_v0 - ask_v0) / (bid_v0 + ask_v0 + eps)
    bid_vtot = book_norm_t[:, 0, :, 1].sum(axis=1) + eps
    ask_vtot = book_norm_t[:, 1, :, 1].sum(axis=1) + eps
    bid_conc = bid_v0 / bid_vtot
    ask_conc = ask_v0 / ask_vtot
    return np.stack([spread, imbalance, bid_conc, ask_conc], axis=1).astype(np.float32)


def apply_standardizer(x: np.ndarray, standardizer: Optional[Dict]) -> np.ndarray:
    if not standardizer:
        return x.astype(np.float32)
    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    std = np.asarray(standardizer["std"], dtype=np.float32)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def inverse_standardizer(x: np.ndarray, standardizer: Optional[Dict]) -> np.ndarray:
    if not standardizer:
        return x.astype(np.float32)
    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    std = np.asarray(standardizer["std"], dtype=np.float32)
    return (x * std + mean).astype(np.float32)


def compute_regime_labels(stock_ids: np.ndarray, day_ids: np.ndarray, mid_z: np.ndarray, n_stocks: int) -> np.ndarray:
    regimes = np.zeros(len(stock_ids), dtype=np.int64)
    for s in range(n_stocks):
        s_mask = stock_ids == s
        s_days = np.unique(day_ids[s_mask])
        vol_per_day = {}
        for d in s_days:
            m = s_mask & (day_ids == d)
            md = mid_z[m]
            vol_per_day[d] = float(np.diff(md).std()) if len(md) >= 2 else 0.0
        vols = np.array([vol_per_day[d] for d in s_days])
        q33, q67 = np.percentile(vols, [33, 67])
        for d in s_days:
            v = vol_per_day[d]
            r = 0 if v < q33 else 1 if v < q67 else 2
            regimes[s_mask & (day_ids == d)] = r
    return regimes


# =============================================================================
# Dataset encoding
# =============================================================================

@torch.no_grad()
def encode_sample(
    model: LOBAutoTokenizerT,
    cfg: TokenizerConfigT,
    raw: Dict,
    stats: Dict[str, np.ndarray],
    target_standardizers: Dict,
    valid_t: np.ndarray,
    sample_pos: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict:
    book = raw["book"]
    mid_z = raw["mid_z"]
    stock_ids = raw["stock_ids"]
    min_spread = raw["min_spread_z_per_stock"]
    n = len(sample_pos)
    endpoints = valid_t[sample_pos]

    raw_feat, _ = derive_raw_features_array(book, mid_z, stock_ids, int(min_spread.shape[0]))
    future_raw = compute_future_feature_targets(raw_feat, endpoints, cfg.future_features, cfg.future_horizons)
    vol_raw = compute_vol_targets(mid_z, endpoints, cfg.vol_horizons, min_spread, stock_ids)

    book_t_norm = normalize_book_batch(book[endpoints], mid_z[endpoints], stock_ids[endpoints], stats)
    stats_raw = compute_stats_targets_normalized(book_t_norm)

    future_std = apply_standardizer(future_raw, target_standardizers.get("future"))
    vol_std = apply_standardizer(vol_raw, target_standardizers.get("vol"))
    stats_std = apply_standardizer(stats_raw, target_standardizers.get("stats"))

    Z, book_pred, stats_pred, future_pred, vol_pred = [], [], [], [], []
    dyn_losses = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bsz = end - start
        bw = np.empty((bsz, cfg.K, 2, cfg.L, 2), dtype=np.float32)
        sids = stock_ids[endpoints[start:end]].astype(np.int64)
        for j, t in enumerate(endpoints[start:end]):
            bw[j] = normalize_book_window(book[t - cfg.K + 1 : t + 1], mid_z[t - cfg.K + 1 : t + 1], int(stock_ids[t]), stats)
        bw_t = torch.from_numpy(bw).to(device)
        sid_t = torch.from_numpy(sids).to(device)
        st_t = torch.from_numpy(stats_std[start:end]).to(device)
        fut_t = torch.from_numpy(future_std[start:end]).to(device)
        vol_t = torch.from_numpy(vol_std[start:end]).to(device)
        out = model(bw_t, sid_t, st_t, fut_t, vol_t)
        Z.append(out["z"].cpu().numpy())
        book_pred.append(out["book_pred"].cpu().numpy())
        stats_pred.append(out["stats_pred"].cpu().numpy())
        future_pred.append(out["future_pred"].cpu().numpy())
        vol_pred.append(out["vol_pred"].cpu().numpy())
        dyn_losses.append(float(out["losses"]["dyn"].detach().cpu()))

    Z = np.concatenate(Z, axis=0)
    book_pred = np.concatenate(book_pred, axis=0)
    stats_pred_std = np.concatenate(stats_pred, axis=0)
    future_pred_std = np.concatenate(future_pred, axis=0)
    vol_pred_std = np.concatenate(vol_pred, axis=0)

    # Inverse standardize predictions for interpretable R²/MSE. R² is affine-invariant,
    # but raw units help inspection and JSON output.
    stats_pred_raw = inverse_standardizer(stats_pred_std, target_standardizers.get("stats"))
    future_pred_raw = inverse_standardizer(future_pred_std, target_standardizers.get("future"))
    vol_pred_raw = inverse_standardizer(vol_pred_std, target_standardizers.get("vol"))

    regimes_all = compute_regime_labels(stock_ids, raw["day_ids"], mid_z, int(min_spread.shape[0]))

    return {
        "endpoints": endpoints,
        "Z": Z,
        "book_norm": book_t_norm,
        "book_pred": book_pred,
        "stats_true_raw": stats_raw,
        "stats_true_std": stats_std,
        "stats_pred_raw": stats_pred_raw,
        "stats_pred_std": stats_pred_std,
        "future_true_raw": future_raw,
        "future_true_std": future_std,
        "future_pred_raw": future_pred_raw,
        "future_pred_std": future_pred_std,
        "vol_true_raw": vol_raw,
        "vol_true_std": vol_std,
        "vol_pred_raw": vol_pred_raw,
        "vol_pred_std": vol_pred_std,
        "dyn_loss_mean": float(np.mean(dyn_losses)) if dyn_losses else float("nan"),
        "stock_ids": stock_ids[endpoints],
        "day_ids": raw["day_ids"][endpoints],
        "regimes": regimes_all[endpoints],
        "mid_z": mid_z[endpoints],
    }


# =============================================================================
# Metrics
# =============================================================================


def _r2_dict(pred: np.ndarray, true: np.ndarray, names: List[str]) -> Dict[str, float]:
    out = {}
    for i, name in enumerate(names):
        y = true[:, i]
        p = pred[:, i]
        var = y.var()
        mse = ((p - y) ** 2).mean()
        out[name] = float(1.0 - mse / max(var, 1e-12))
    return out


def _mse_dict(pred: np.ndarray, true: np.ndarray, names: List[str]) -> Dict[str, float]:
    return {name: float(((pred[:, i] - true[:, i]) ** 2).mean()) for i, name in enumerate(names)}


def reconstruction_metrics(data: Dict, L: int) -> Dict:
    book_true = data["book_norm"]
    book_pred = data["book_pred"]
    regimes = data["regimes"]

    vt = book_true[:, :, :, 1]
    vp = book_pred[:, :, :, 1]
    pt = book_true[:, :, :, 0]
    pp = book_pred[:, :, :, 0]

    w = np.ones(L, dtype=np.float64)
    w[0] = 4.0
    if L > 1:
        w[1] = 2.0
    w = w / w.sum()

    per_level_vol_mse = ((vp - vt) ** 2).mean(axis=(0, 1))
    per_level_vol_mae = np.abs(vp - vt).mean(axis=(0, 1))
    per_level_price_mse = ((pp - pt) ** 2).mean(axis=(0, 1))
    per_level_price_mae = np.abs(pp - pt).mean(axis=(0, 1))

    bid = pp[:, 0, :]
    ask = pp[:, 1, :]
    spread_pred = ask[:, 0] - bid[:, 0]
    spread_true = pt[:, 1, 0] - pt[:, 0, 0]

    struct = {
        "bid_mono_viol_rate": float((bid[:, 1:] > bid[:, :-1]).mean()),
        "ask_mono_viol_rate": float((ask[:, 1:] < ask[:, :-1]).mean()),
        "bid_side_viol_rate": float((bid > 0).mean()),
        "ask_side_viol_rate": float((ask < 0).mean()),
        "crossed_viol_rate": float((bid[:, 0] >= ask[:, 0]).mean()),
        "spread_MAE": float(np.abs(spread_pred - spread_true).mean()),
        "spread_MSE": float(((spread_pred - spread_true) ** 2).mean()),
    }

    per_regime = {}
    for r, rname in enumerate(REGIME_NAMES):
        m = regimes == r
        if m.sum() == 0:
            continue
        per_regime[rname] = {
            "n": int(m.sum()),
            "vol_mse": float(((vp[m] - vt[m]) ** 2).mean()),
            "price_mse": float(((pp[m] - pt[m]) ** 2).mean()),
        }

    return {
        "volume_MSE": float(per_level_vol_mse.mean()),
        "volume_MAE": float(per_level_vol_mae.mean()),
        "volume_wMSE": float((per_level_vol_mse * w).sum()),
        "volume_top_MSE": float(per_level_vol_mse[:2].mean()),
        "volume_deep_MSE": float(per_level_vol_mse[2:].mean()) if L > 2 else 0.0,
        "price_MSE": float(per_level_price_mse.mean()),
        "price_MAE": float(per_level_price_mae.mean()),
        "price_wMSE": float((per_level_price_mse * w).sum()),
        "price_top_MSE": float(per_level_price_mse[:2].mean()),
        "price_deep_MSE": float(per_level_price_mse[2:].mean()) if L > 2 else 0.0,
        "pv_wMSE": float((per_level_vol_mse * w).sum() + (per_level_price_mse * w).sum()),
        "per_level_vol_mse": per_level_vol_mse,
        "per_level_price_mse": per_level_price_mse,
        "per_level_vol_mae": per_level_vol_mae,
        "per_level_price_mae": per_level_price_mae,
        "struct": struct,
        "per_regime": per_regime,
    }


def head_metrics(data: Dict, cfg: TokenizerConfigT) -> Dict:
    stats_names = ["spread_norm", "imbalance", "bid_conc", "ask_conc"]
    future_names = []
    for f in cfg.future_features:
        for h in cfg.future_horizons:
            future_names.append(f"{f}@{h}")
    vol_names = [f"realized_vol@{h}" for h in cfg.vol_horizons]

    return {
        "stats_r2": _r2_dict(data["stats_pred_raw"], data["stats_true_raw"], stats_names),
        "stats_mse": _mse_dict(data["stats_pred_raw"], data["stats_true_raw"], stats_names),
        "future_r2": _r2_dict(data["future_pred_raw"], data["future_true_raw"], future_names),
        "future_mse": _mse_dict(data["future_pred_raw"], data["future_true_raw"], future_names),
        "vol_r2": _r2_dict(data["vol_pred_raw"], data["vol_true_raw"], vol_names),
        "vol_mse": _mse_dict(data["vol_pred_raw"], data["vol_true_raw"], vol_names),
    }


def latent_geometry(data: Dict) -> Dict:
    Z = data["Z"]
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = np.cov(Zc.T)
    corr = np.corrcoef(Z.T)
    off = np.abs(corr - np.eye(corr.shape[0]))

    pca = PCA(n_components=min(10, Z.shape[1]))
    Zs = StandardScaler().fit_transform(Z)
    pca.fit(Zs)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 1e-12)
    p = eig / eig.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    norms = np.linalg.norm(Z, axis=1)

    return {
        "z_mean_mean": float(Z.mean(axis=0).mean()),
        "z_std_mean": float(Z.std(axis=0).mean()),
        "z_std_min": float(Z.std(axis=0).min()),
        "z_std_max": float(Z.std(axis=0).max()),
        "norm_mean": float(norms.mean()),
        "norm_p95": float(np.percentile(norms, 95)),
        "corr": corr,
        "max_off_diag": float(off.max()),
        "mean_off_diag": float(off[off > 0].mean()),
        "pca_explained": pca.explained_variance_ratio_,
        "effective_rank": eff_rank,
    }


def injectivity_analysis(data: Dict, n_pairs: int = 50000, n_points: int = 5000) -> Dict:
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(data["Z"]), min(n_points, len(data["Z"])), replace=False)
    Z = data["Z"][sub_idx]
    O = data["book_norm"][sub_idx].reshape(len(sub_idx), -1)
    S = data["stock_ids"][sub_idx]
    i_arr = rng.integers(0, len(Z), size=n_pairs)
    j_arr = rng.integers(0, len(Z), size=n_pairs)
    same = (S[i_arr] == S[j_arr]) & (i_arr != j_arr)
    i_arr, j_arr = i_arr[same], j_arr[same]
    dz = np.linalg.norm(Z[i_arr] - Z[j_arr], axis=1)
    do = np.linalg.norm(O[i_arr] - O[j_arr], axis=1)
    valid = dz > 1e-6
    c = do[valid] / dz[valid]
    if len(c) == 0:
        return {"median": float("nan"), "p95": float("nan"), "p99": float("nan"), "p99_over_median": float("nan")}
    return {
        "median": float(np.median(c)),
        "p95": float(np.percentile(c, 95)),
        "p99": float(np.percentile(c, 99)),
        "max": float(c.max()),
        "p99_over_median": float(np.percentile(c, 99) / max(np.median(c), 1e-12)),
    }


def knn_consistency(data: Dict, n: int = 2000, k: int = 10) -> Dict:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(data["Z"]), min(n, len(data["Z"])), replace=False)
    Z = data["Z"][idx]
    O = data["book_norm"][idx].reshape(len(idx), -1)
    S = data["stock_ids"][idx]
    Dz = cdist(Z, Z)
    Do = cdist(O, O)
    same = S[:, None] == S[None, :]
    Dz = np.where(same, Dz, np.inf)
    np.fill_diagonal(Dz, np.inf)
    knn_idx = np.argsort(Dz, axis=1)[:, :k]
    knn_do = np.take_along_axis(Do, knn_idx, axis=1).mean(axis=1)
    rand_do = np.zeros(len(idx))
    for i in range(len(idx)):
        pool = np.where(same[i])[0]
        pool = pool[pool != i]
        if len(pool) == 0:
            rand_do[i] = 1.0
            continue
        pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
        rand_do[i] = Do[i, pick].mean()
    ratio = knn_do / (rand_do + 1e-12)
    return {"median_ratio": float(np.median(ratio)), "p95_ratio": float(np.percentile(ratio, 95))}


def regression_probe(Z: np.ndarray, y: np.ndarray, name: str) -> Dict:
    Zs = StandardScaler().fit_transform(Z)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Zs))
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]
    reg = Ridge(alpha=1.0).fit(Zs[tr], y[tr])
    pred = reg.predict(Zs[te])
    return {
        "name": name,
        "R2": float(r2_score(y[te], pred)),
        "MSE": float(mean_squared_error(y[te], pred)),
        "MAE": float(mean_absolute_error(y[te], pred)),
    }


def classification_probe(Z: np.ndarray, y: np.ndarray, name: str) -> Dict:
    if len(np.unique(y)) < 2:
        return {"name": name, "accuracy": float("nan"), "baseline": float("nan")}
    Zs = StandardScaler().fit_transform(Z)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Zs))
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(Zs[tr], y[tr])
    acc = clf.score(Zs[te], y[te])
    base = np.bincount(y[te]).max() / len(te)
    return {"name": name, "accuracy": float(acc), "baseline": float(base)}


def downstream_probes(data: Dict) -> Dict:
    Z = data["Z"]
    # simple diagnostics from existing eval: immediate stats and regimes.
    stats = data["stats_true_raw"]
    out = {
        "spread_norm": regression_probe(Z, stats[:, 0], "spread_norm"),
        "imbalance": regression_probe(Z, stats[:, 1], "imbalance"),
        "bid_conc": regression_probe(Z, stats[:, 2], "bid_conc"),
        "ask_conc": regression_probe(Z, stats[:, 3], "ask_conc"),
        "regime_tri": classification_probe(Z, data["regimes"], "regime_tri"),
    }
    m = data["regimes"] != 1
    if m.sum() > 200:
        out["regime_bin"] = classification_probe(Z[m], (data["regimes"][m] == 2).astype(np.int64), "regime_bin")
    return out


@torch.no_grad()
def latent_autocorrelation(
    model: LOBAutoTokenizerT,
    cfg: TokenizerConfigT,
    raw: Dict,
    stats: Dict[str, np.ndarray],
    vol_mask: np.ndarray,
    device: torch.device,
    Ks: List[int],
    n_samples: int,
    batch_size: int,
    seed: int,
) -> Dict:
    stock_ids = raw["stock_ids"]
    day_ids = raw["day_ids"]
    book = raw["book"]
    mid_z = raw["mid_z"]
    Kmax = max(Ks)
    valid_t = compute_valid_endpoints(stock_ids, day_ids, cfg.K, Kmax, vol_mask)
    valid_t = valid_t[valid_t + Kmax < len(stock_ids)]
    if len(valid_t) == 0:
        return {}
    rng = np.random.default_rng(seed)
    endpoints = rng.choice(valid_t, size=min(n_samples, len(valid_t)), replace=False)

    def encode_endpoints(ts: np.ndarray) -> np.ndarray:
        outs = []
        for start in range(0, len(ts), batch_size):
            end = min(start + batch_size, len(ts))
            bsz = end - start
            bw = np.empty((bsz, cfg.K, 2, cfg.L, 2), dtype=np.float32)
            sids = stock_ids[ts[start:end]].astype(np.int64)
            for j, t in enumerate(ts[start:end]):
                bw[j] = normalize_book_window(book[t - cfg.K + 1 : t + 1], mid_z[t - cfg.K + 1 : t + 1], int(stock_ids[t]), stats)
            z = model.encode(torch.from_numpy(bw).to(device), torch.from_numpy(sids).to(device))
            outs.append(z.cpu().numpy())
        return np.concatenate(outs, axis=0)

    Z0 = encode_endpoints(endpoints)
    out = {}
    for lag in Ks:
        Z1 = encode_endpoints(endpoints + lag)
        c = []
        for d in range(Z0.shape[1]):
            x, y = Z0[:, d], Z1[:, d]
            if x.std() < 1e-8 or y.std() < 1e-8:
                c.append(0.0)
            else:
                c.append(float(np.corrcoef(x, y)[0, 1]))
        c = np.asarray(c)
        out[int(lag)] = {
            "mean": float(c.mean()),
            "median": float(np.median(c)),
            "min": float(c.min()),
            "max": float(c.max()),
            "per_dim": c,
        }
    return out


# =============================================================================
# Printing / plotting
# =============================================================================


def print_reconstruction(rec: Dict):
    print("\n" + "=" * 78)
    print("[1] RECONSTRUCTION / STRUCTURAL VALIDITY")
    print("=" * 78)
    print(f"  volume MSE={rec['volume_MSE']:.6f}  MAE={rec['volume_MAE']:.6f}  wMSE={rec['volume_wMSE']:.6f}")
    print(f"  price  MSE={rec['price_MSE']:.6f}  MAE={rec['price_MAE']:.6f}  wMSE={rec['price_wMSE']:.6f}")
    print(f"  combined pv_wMSE={rec['pv_wMSE']:.6f}")
    st = rec["struct"]
    print("  structural:")
    print(f"    bid_mono={100*st['bid_mono_viol_rate']:.4f}%  ask_mono={100*st['ask_mono_viol_rate']:.4f}%  crossed={100*st['crossed_viol_rate']:.4f}%")
    print(f"    bid_side={100*st['bid_side_viol_rate']:.4f}%  ask_side={100*st['ask_side_viol_rate']:.4f}%  spread_MAE={st['spread_MAE']:.6f}")


def print_heads(hm: Dict, cfg: TokenizerConfigT):
    print("\n" + "=" * 78)
    print("[2] A1-T HEADS")
    print("=" * 78)
    print("  stats head R²:")
    for k, v in hm["stats_r2"].items():
        print(f"    {k:<18s} {v:+.4f}")
    print("\n  future feature head R²:")
    for k, v in hm["future_r2"].items():
        print(f"    {k:<28s} {v:+.4f}")
    print("\n  realized vol head R²:")
    for k, v in hm["vol_r2"].items():
        print(f"    {k:<28s} {v:+.4f}")


def print_geometry(g: Dict, inj: Dict, knn: Dict):
    print("\n" + "=" * 78)
    print("[3] LATENT GEOMETRY")
    print("=" * 78)
    pca_cum = np.cumsum(g["pca_explained"])
    print(f"  z_std mean={g['z_std_mean']:.4f} min={g['z_std_min']:.4f} max={g['z_std_max']:.4f}")
    print(f"  norm mean={g['norm_mean']:.4f} p95={g['norm_p95']:.4f}")
    print(f"  PCA: first5={100*pca_cum[min(4, len(pca_cum)-1)]:.1f}% first8={100*pca_cum[min(7, len(pca_cum)-1)]:.1f}%")
    print(f"  effective_rank={g['effective_rank']:.2f}")
    print(f"  corr max|off|={g['max_off_diag']:.4f} mean|off|={g['mean_off_diag']:.4f}")
    print(f"  injectivity p99/median={inj['p99_over_median']:.2f} median={inj['median']:.4f} p99={inj['p99']:.4f}")
    print(f"  kNN median_ratio={knn['median_ratio']:.4f} p95={knn['p95_ratio']:.4f}")


def print_autocorr(ac: Dict):
    print("\n" + "=" * 78)
    print("[4] LATENT AUTOCORRELATION")
    print("=" * 78)
    if not ac:
        print("  no valid autocorrelation sample")
        return
    print(f"  {'lag':>6s} {'~sec':>6s} {'mean':>9s} {'median':>9s} {'min':>9s} {'max':>9s}")
    for k in sorted(ac.keys()):
        v = ac[k]
        print(f"  {k:6d} {3*k:6d} {v['mean']:+9.4f} {v['median']:+9.4f} {v['min']:+9.4f} {v['max']:+9.4f}")


def plot_summary(out_dir: Path, rec: Dict, hm: Dict, geom: Dict, ac: Dict):
    if plt is None:
        return
    # Figure 1: recon per level
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(rec["per_level_vol_mse"]))
    ax.plot(x, rec["per_level_vol_mse"], marker="o", label="volume MSE")
    ax.plot(x, rec["per_level_price_mse"], marker="o", label="price MSE")
    ax.set_xlabel("LOB level")
    ax.set_ylabel("MSE")
    ax.set_title("A1-T reconstruction per level")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "a1t_reconstruction_per_level.png", dpi=160)
    plt.close(fig)

    # Figure 2: PCA explained variance
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = geom["pca_explained"]
    ax.bar(np.arange(1, len(vals)+1), 100*vals)
    ax.plot(np.arange(1, len(vals)+1), 100*np.cumsum(vals), marker="o")
    ax.set_xlabel("PC")
    ax.set_ylabel("variance explained (%)")
    ax.set_title("A1-T latent PCA")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "a1t_latent_pca.png", dpi=160)
    plt.close(fig)

    # Figure 3: autocorr
    if ac:
        fig, ax = plt.subplots(figsize=(7, 4))
        ks = sorted(ac.keys())
        ax.plot(ks, [ac[k]["mean"] for k in ks], marker="o", label="mean")
        ax.plot(ks, [ac[k]["median"] for k in ks], marker="o", label="median")
        ax.set_xlabel("lag")
        ax.set_ylabel("corr(z_t, z_{t+lag})")
        ax.set_title("A1-T latent autocorrelation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "a1t_latent_autocorrelation.png", dpi=160)
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main():
    p = argparse.ArgumentParser(description="Evaluate A1-T temporal LOB tokenizer")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out_dir", default="validation/tokenizer/a1_T_eval")
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vol_clip", type=float, default=None)
    p.add_argument("--vol_filter", choices=["raw", "normalized"], default="raw",
                   help="raw matches current A1-T trainer; normalized matches v1/v2-style filtering")
    p.add_argument("--split", choices=["val", "train", "all"], default="val",
                   help="Evaluate on A1-T validation stock-days by default")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--autocorr_lags", default="1,2,5,10,20,50,100")
    p.add_argument("--autocorr_samples", type=int, default=5000)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    model, cfg, ckpt, stock_stats, target_standardizers = load_a1t(args.ckpt, device)

    raw_npz = np.load(args.dataset)
    raw = {
        "book": raw_npz["book"].astype(np.float32, copy=False),
        "mid_z": raw_npz["mid_z"].astype(np.float32, copy=False),
        "stock_ids": raw_npz["stock_ids"].astype(np.int64, copy=False),
        "day_ids": raw_npz["day_ids"].astype(np.int64, copy=False),
        "min_spread_z_per_stock": raw_npz["min_spread_z_per_stock"].astype(np.float32, copy=False),
    }
    N = len(raw["mid_z"])
    if args.vol_clip is None:
        args.vol_clip = float(ckpt.get("train_args", {}).get("vol_clip", 5.0))

    max_horizon = max(max(cfg.future_horizons), max(cfg.vol_horizons))
    print("\nBuilding valid endpoints...")
    vol_mask = compute_vol_mask(raw["book"], raw["stock_ids"], stock_stats, args.vol_clip, args.vol_filter)
    valid_t = compute_valid_endpoints(raw["stock_ids"], raw["day_ids"], cfg.K, max_horizon, vol_mask)
    print(f"  vol_filter={args.vol_filter} vol_clip={args.vol_clip}")
    print(f"  valid_t={len(valid_t):,}/{N:,} ({100*len(valid_t)/N:.2f}%)")

    val_frac = float(ckpt.get("train_args", {}).get("val_frac", 0.15))
    split_seed = int(ckpt.get("train_args", {}).get("seed", 42))
    train_pos, val_pos, val_episode_ids = grouped_split_by_stock_day(raw["stock_ids"], raw["day_ids"], valid_t, val_frac, split_seed)
    if args.split == "val":
        pool = val_pos
    elif args.split == "train":
        pool = train_pos
    else:
        pool = np.arange(len(valid_t))
    rng = np.random.default_rng(args.seed)
    if len(pool) > args.n_samples:
        sample_pos = np.sort(rng.choice(pool, size=args.n_samples, replace=False))
    else:
        sample_pos = np.sort(pool)
    print(f"  split={args.split} pool={len(pool):,} sampled={len(sample_pos):,}")

    print("\nEncoding sampled endpoints and collecting heads...")
    t0 = time.time()
    data = encode_sample(model, cfg, raw, stock_stats, target_standardizers, valid_t, sample_pos, device, args.batch_size)
    print(f"  encoded Z={data['Z'].shape} in {time.time()-t0:.1f}s")
    print(f"  dyn alignment loss mean={data['dyn_loss_mean']:.4f}")

    rec = reconstruction_metrics(data, cfg.L)
    hm = head_metrics(data, cfg)
    geom = latent_geometry(data)
    inj = injectivity_analysis(data)
    knn = knn_consistency(data)
    ds = downstream_probes(data)

    ac_lags = [int(x) for x in args.autocorr_lags.split(",") if x.strip()]
    ac = latent_autocorrelation(model, cfg, raw, stock_stats, vol_mask, device, ac_lags, args.autocorr_samples, args.batch_size, args.seed)

    print_reconstruction(rec)
    print_heads(hm, cfg)
    print_geometry(geom, inj, knn)
    print_autocorr(ac)

    print("\n" + "=" * 78)
    print("[5] SIMPLE DOWNSTREAM DIAGNOSTIC PROBES")
    print("=" * 78)
    for k, v in ds.items():
        if "R2" in v:
            print(f"  {k:<18s} R²={v['R2']:+.4f}")
        elif "accuracy" in v:
            print(f"  {k:<18s} acc={100*v['accuracy']:.2f}% baseline={100*v['baseline']:.2f}%")

    # Save JSON-friendly metrics.
    def _np_to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        return x

    metrics = {
        "args": vars(args),
        "cfg": cfg.to_dict(),
        "n_eval": int(len(sample_pos)),
        "valid_fraction": float(len(valid_t) / N),
        "dyn_loss_mean": data["dyn_loss_mean"],
        "reconstruction": {k: _np_to_list(v) for k, v in rec.items() if k not in ["per_regime"]},
        "reconstruction_per_regime": rec["per_regime"],
        "heads": hm,
        "latent_geometry": {k: _np_to_list(v) for k, v in geom.items() if k != "corr"},
        "injectivity": inj,
        "knn": knn,
        "autocorrelation": {str(k): {kk: _np_to_list(vv) for kk, vv in val.items()} for k, val in ac.items()},
        "downstream": ds,
    }
    with open(out_dir / "eval_tokenizer_t_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        out_dir / "eval_tokenizer_t_arrays.npz",
        Z=data["Z"],
        endpoints=data["endpoints"],
        regimes=data["regimes"],
        latent_corr=geom["corr"],
        pca_explained=geom["pca_explained"],
        recon_vol_per_level=rec["per_level_vol_mse"],
        recon_price_per_level=rec["per_level_price_mse"],
    )
    if not args.no_plots:
        plot_summary(out_dir, rec, hm, geom, ac)
    print(f"\nSaved metrics to: {out_dir}")


if __name__ == "__main__":
    main()

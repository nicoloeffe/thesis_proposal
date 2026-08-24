"""
latent_raw_probe.py — Diagnostic probe: do A1 latent histories preserve raw LOB long-memory features?

Patched version:
  - optional --tokenizer_ckpt argument.
  - if --tokenizer_ckpt is provided, the probe validation set is built from the
    same stock-day validation split used by A1-T training, reconstructed from
    the checkpoint train_args/cfg if val_episode_ids are not saved.
  - without --tokenizer_ckpt, it falls back to the legacy independent grouped split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ----------------------------- utils -----------------------------

def find_key(npz, candidates: Iterable[str], ndim: Optional[int] = None):
    for k in candidates:
        if k in npz.files and (ndim is None or npz[k].ndim == ndim):
            return k
    return None


def grouped_split(n: int, group_ids: Optional[np.ndarray], val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if group_ids is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * n)))
        return np.sort(idx[n_val:]), np.sort(idx[:n_val])
    unique = np.unique(group_ids)
    rng.shuffle(unique)
    n_val_groups = max(1, int(round(val_frac * len(unique))))
    val_groups = set(unique[:n_val_groups].tolist())
    mask = np.array([g in val_groups for g in group_ids])
    return np.where(~mask)[0], np.where(mask)[0]


def subsample(idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    out = rng.choice(idx, size=max_n, replace=False)
    out.sort()
    return out


# ----------------------------- A1-T split reconstruction -----------------------------

def compute_a1t_valid_endpoints(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    K: int,
    max_horizon: int,
    vol_mask: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the exact valid_t criterion used by training/train_tokenizer_t.py.

    Endpoint t is valid iff:
      - [t-K+1, ..., t+max_horizon] is inside the same stock-day block;
      - all snapshots in that interval pass vol_mask.
    """
    N = len(stock_ids)
    if N == 0:
        return np.array([], dtype=np.int64)

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
    vol_int = vol_mask.astype(np.int32)
    cum = np.concatenate([[0], np.cumsum(vol_int)])

    candidates = np.where(bound_ok)[0]
    a = candidates - (K - 1)
    b = candidates + max_horizon
    win_sum = cum[b + 1] - cum[a]
    vol_ok = win_sum == win_len
    return candidates[vol_ok].astype(np.int64)


def reconstruct_a1t_val_episode_ids(tokenizer_ckpt: str, raw_npz) -> Tuple[np.ndarray, str, Dict]:
    """
    Return A1-T validation stock-day composite ids.

    Future checkpoints may save val_episode_ids directly. For existing checkpoints,
    reconstruct them deterministically from train_args/cfg and raw data.
    Composite key matches training/train_tokenizer_t.py:
        stock_id * (day_ids.max()+1) + day_id
    """
    ckpt = torch.load(tokenizer_ckpt, map_location="cpu", weights_only=False)

    if "val_episode_ids" in ckpt:
        val_episode_ids = np.asarray(ckpt["val_episode_ids"], dtype=np.int64)
        meta = ckpt.get("split_meta", {}) or {}
        return val_episode_ids, "checkpoint:val_episode_ids", meta

    args = ckpt.get("train_args", {}) or {}
    cfg = ckpt.get("cfg", {}) or {}

    stock_ids = raw_npz["stock_ids"].astype(np.int64)
    day_ids = raw_npz["day_ids"].astype(np.int64)
    book = raw_npz["book"].astype(np.float32, copy=False)

    K = int(cfg.get("K", args.get("K", 20)))
    future_horizons = cfg.get("future_horizons", [1, 5, 10, 20])
    vol_horizons = cfg.get("vol_horizons", [5, 20])
    max_horizon = int(max(max(future_horizons), max(vol_horizons)))
    val_frac = float(args.get("val_frac", 0.15))
    seed = int(args.get("seed", 42))
    vol_clip = float(args.get("vol_clip", 5.0))

    # This intentionally mirrors the current A1-T trainer that produced the checkpoint:
    # raw z-scored volume mask, not v1/v2 normalized-volume mask.
    bid_v = book[:, 0, :, 1]
    ask_v = book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= vol_clip) & (np.abs(ask_v).max(axis=1) <= vol_clip)

    valid_t = compute_a1t_valid_endpoints(stock_ids, day_ids, K, max_horizon, vol_mask)

    max_day = int(day_ids.max()) + 1
    s_t = stock_ids[valid_t]
    d_t = day_ids[valid_t]
    composite = s_t.astype(np.int64) * max_day + d_t.astype(np.int64)

    unique_groups = np.unique(composite)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(round(val_frac * len(unique_groups))))
    val_episode_ids = unique_groups[:n_val_groups].astype(np.int64)

    meta = {
        "kind": "reconstructed_a1t_stock_day",
        "K": K,
        "max_horizon": max_horizon,
        "val_frac": val_frac,
        "seed": seed,
        "vol_clip": vol_clip,
        "n_valid_t": int(len(valid_t)),
        "n_unique_groups": int(len(unique_groups)),
        "max_day": int(max_day),
    }
    source = f"reconstructed:A1T val_frac={val_frac} seed={seed} K={K} max_horizon={max_horizon}"
    return val_episode_ids, source, meta


def split_by_a1t_checkpoint(
    tokenizer_ckpt: str,
    raw_npz,
    raw_idx_2d: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    """Split latent sequences using the validation stock-days of A1-T training."""
    val_episode_ids, source, split_meta = reconstruct_a1t_val_episode_ids(tokenizer_ckpt, raw_npz)

    stock_ids_raw = raw_npz["stock_ids"].astype(np.int64)
    day_ids_raw = raw_npz["day_ids"].astype(np.int64)
    max_day = int(day_ids_raw.max()) + 1

    # Sequence is entirely one stock-day; use the endpoint before target transition
    # (seq_len-1) as the reference. First token would be equivalent by construction.
    seq_ref_raw = raw_idx_2d[:, seq_len - 1]
    seq_episode_ids = (
        stock_ids_raw[seq_ref_raw].astype(np.int64) * max_day
        + day_ids_raw[seq_ref_raw].astype(np.int64)
    )

    val_set = set(val_episode_ids.tolist())
    val_mask = np.array([e in val_set for e in seq_episode_ids], dtype=bool)
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]

    if len(va_idx) == 0:
        raise RuntimeError(
            "A1-T checkpoint split produced zero validation sequences. "
            "Check tokenizer_ckpt/raw dataset compatibility."
        )
    if len(tr_idx) == 0:
        raise RuntimeError(
            "A1-T checkpoint split produced zero training sequences. "
            "Check tokenizer_ckpt/raw dataset compatibility."
        )

    mode = f"A1T_checkpoint_split ({source})"
    meta = {
        **split_meta,
        "n_val_episode_ids": int(len(val_episode_ids)),
        "n_train_sequences_before_subsample": int(len(tr_idx)),
        "n_val_sequences_before_subsample": int(len(va_idx)),
    }
    return tr_idx, va_idx, mode, meta


# ----------------------------- raw features -----------------------------

def derive_raw_features(raw_npz) -> Tuple[np.ndarray, List[str], Dict]:
    """Return feature matrix (N, F) from raw LOBench processed NPZ."""
    book = raw_npz["book"].astype(np.float32, copy=False)
    mid_z = raw_npz["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw_npz["stock_ids"].astype(np.int64, copy=False)
    N, _, L, _ = book.shape

    bid_p = book[:, 0, :, 0]
    ask_p = book[:, 1, :, 0]
    bid_vz = book[:, 0, :, 1]
    ask_vz = book[:, 1, :, 1]

    # Shift volume z-scores to non-negative per stock, as in LOBenchDataset.
    n_stocks = int(stock_ids.max()) + 1
    vol_min = np.zeros(n_stocks, dtype=np.float32)
    for s in range(n_stocks):
        m = stock_ids == s
        vol_min[s] = float(min(bid_vz[m].min(), ask_vz[m].min())) if m.any() else 0.0
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

    names = [
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

    meta = {"n_raw": int(N), "n_stocks": int(n_stocks), "feature_names": names}
    return feats, names, meta


# ----------------------------- alignment via raw_indices -----------------------------

def load_alignment(lat_npz, n_seq: int, seq_len: int) -> Tuple[np.ndarray, str]:
    """
    Load explicit alignment from the latent NPZ.

    Returns raw_idx_2d of shape (n_seq, seq_len+1), where raw_idx_2d[i, j] is
    the index in the raw LOBench dataset of token j of latent sequence i.
    """
    if "raw_indices" not in lat_npz.files:
        raise ValueError(
            "Latent dataset has no `raw_indices` key. Cannot align to raw LOBench."
            " Re-run build_wm_dataset_lobench_tokenizer.py to regenerate."
        )
    raw_idx = lat_npz["raw_indices"]
    if raw_idx.ndim != 2:
        raise ValueError(f"`raw_indices` must be 2D, got ndim={raw_idx.ndim}")
    if raw_idx.shape[0] < n_seq:
        raise ValueError(f"`raw_indices` has {raw_idx.shape[0]} rows, expected >= {n_seq}")
    if raw_idx.shape[1] < seq_len + 1:
        raise ValueError(
            f"`raw_indices` has {raw_idx.shape[1]} cols, expected >= seq_len+1={seq_len+1}"
        )
    return raw_idx[:n_seq, : seq_len + 1].astype(np.int64), "explicit:raw_indices_2d"


def alignment_sanity_check(raw_idx_2d: np.ndarray, raw_npz, n_check: int = 64) -> None:
    """Sanity-check same (stock, day) and monotonicity inside latent sequence."""
    stock_ids = raw_npz["stock_ids"].astype(np.int64)
    day_ids = raw_npz["day_ids"].astype(np.int64) if "day_ids" in raw_npz.files else None

    n_check = min(n_check, raw_idx_2d.shape[0])
    pick = np.arange(n_check)
    rows = raw_idx_2d[pick]

    s_per_token = stock_ids[rows]
    monotonic = (np.diff(rows, axis=1) > 0).all(axis=1)
    same_stock = (s_per_token == s_per_token[:, :1]).all(axis=1)

    if day_ids is not None:
        d_per_token = day_ids[rows]
        same_day = (d_per_token == d_per_token[:, :1]).all(axis=1)
    else:
        same_day = np.ones(n_check, dtype=bool)

    n_ok = int((monotonic & same_stock & same_day).sum())
    print(f"  alignment sanity: {n_ok}/{n_check} sequences pass "
          f"(monotonic & same-stock & same-day)")
    if n_ok < n_check:
        bad_idx = np.where(~(monotonic & same_stock & same_day))[0][:3]
        for i in bad_idx:
            print(f"    bad seq {i}: rows[:5]={rows[i, :5].tolist()}, "
                  f"stocks[:5]={s_per_token[i, :5].tolist()}")


# ----------------------------- probe model -----------------------------

class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.0):
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

    def forward(self, x):
        return self.net(x)


def standardize_train_val(xtr, xva, eps=1e-8):
    mu = xtr.mean(axis=0, keepdims=True)
    sd = xtr.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (xtr - mu) / sd, (xva - mu) / sd, mu.squeeze(0), sd.squeeze(0)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_names: List[str],
    args,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    Xtr, Xva, _, _ = standardize_train_val(X_train.astype(np.float32), X_val.astype(np.float32))
    ytr, yva, y_mu, y_sd = standardize_train_val(y_train.astype(np.float32), y_val.astype(np.float32))

    ds = TensorDataset(torch.from_numpy(Xtr.astype(np.float32)), torch.from_numpy(ytr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = MLPProbe(Xtr.shape[1], ytr.shape[1], hidden=args.hidden, dropout=args.probe_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_state = None
    best_val = float("inf")
    epochs_no_improve = 0
    Xva_t = torch.from_numpy(Xva.astype(np.float32)).to(device)
    yva_t = torch.from_numpy(yva.astype(np.float32)).to(device)

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        scheduler.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                pred = model(Xva_t)
                val_loss = float(((pred - yva_t) ** 2).mean().cpu())
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += args.eval_every
            if args.patience > 0 and epochs_no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds_std = []
    with torch.no_grad():
        for i in range(0, len(Xva), args.eval_batch_size):
            xb = torch.from_numpy(Xva[i:i+args.eval_batch_size].astype(np.float32)).to(device)
            preds_std.append(model(xb).cpu().numpy())
    pred_std = np.concatenate(preds_std, axis=0)
    pred = pred_std * y_sd[None, :] + y_mu[None, :]

    mse = ((pred - y_val) ** 2).mean(axis=0)
    var = y_val.var(axis=0)
    r2 = 1.0 - mse / np.maximum(var, 1e-12)

    return {
        "r2_per_target": {name: float(r2[i]) for i, name in enumerate(target_names)},
        "mse_per_target": {name: float(mse[i]) for i, name in enumerate(target_names)},
        "r2_mean": float(r2.mean()),
        "r2_min": float(r2.min()),
        "r2_max": float(r2.max()),
    }


# ----------------------------- data assembly -----------------------------

def choose_target_indices(feature_names: List[str], targets_arg: str) -> Tuple[List[int], List[str]]:
    requested = [x.strip() for x in targets_arg.split(",") if x.strip()]
    idx = []
    names = []
    for name in requested:
        if name not in feature_names:
            raise ValueError(f"Unknown target feature {name!r}. Available={feature_names}")
        idx.append(feature_names.index(name))
        names.append("d_" + name)
    return idx, names


def make_arrays_for_lag(
    latent_seq: np.ndarray,
    raw_feat: np.ndarray,
    raw_idx_2d: np.ndarray,
    example_idx: np.ndarray,
    lag: int,
    seq_len: int,
    target_idx: List[int],
    raw_input_mode: str = "all_features_delta",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_lat, X_raw, y).
    Target transition is token seq_len-1 -> seq_len.
    """
    z = latent_seq[example_idx, seq_len - lag : seq_len, :]
    X_lat = z.reshape(len(example_idx), -1).astype(np.float32)

    r_target = raw_idx_2d[example_idx, seq_len]
    r_prev = raw_idx_2d[example_idx, seq_len - 1]
    y = raw_feat[r_target][:, target_idx] - raw_feat[r_prev][:, target_idx]

    if lag > seq_len - 1:
        raise ValueError(
            f"raw baseline with lag={lag} requires seq_len-1>=lag, "
            f"but seq_len={seq_len} (max lag = {seq_len - 1})"
        )

    if raw_input_mode == "all_features_delta":
        parts = []
        for j in range(lag, 0, -1):
            r_a = raw_idx_2d[example_idx, seq_len - 1 - j]
            r_b = raw_idx_2d[example_idx, seq_len - j]
            d = raw_feat[r_b] - raw_feat[r_a]
            parts.append(d)
        X_raw = np.concatenate(parts, axis=1)

    elif raw_input_mode == "target_features_delta":
        parts = []
        for j in range(lag, 0, -1):
            r_a = raw_idx_2d[example_idx, seq_len - 1 - j]
            r_b = raw_idx_2d[example_idx, seq_len - j]
            d = raw_feat[r_b][:, target_idx] - raw_feat[r_a][:, target_idx]
            parts.append(d)
        X_raw = np.concatenate(parts, axis=1)

    elif raw_input_mode == "all_features_level":
        parts = []
        for j in range(lag, 0, -1):
            r_pos = raw_idx_2d[example_idx, seq_len - j]
            parts.append(raw_feat[r_pos])
        X_raw = np.concatenate(parts, axis=1)

    else:
        raise ValueError(f"Unknown raw_input_mode={raw_input_mode}")

    return X_lat.astype(np.float32), X_raw.astype(np.float32), y.astype(np.float32)


# ----------------------------- main -----------------------------

def main():
    p = argparse.ArgumentParser(description="Latent-to-raw predictive probe for A1 temporal information.")
    p.add_argument("--latent_dataset", required=True)
    p.add_argument("--raw_dataset", required=True)
    p.add_argument("--tokenizer_ckpt", type=str, default=None,
                   help="Optional A1-T checkpoint. If provided, use A1-T validation stock-days for probe split.")
    p.add_argument("--out_dir", default="validation/world_model/latent_raw_probe")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--lags", type=str, default="1,2,5,10")
    p.add_argument("--targets", type=str, default="spread_z,microprice_rel,best_bid_rel,best_ask_rel,top_imbalance")
    p.add_argument("--raw_input_mode",
                   choices=["all_features_delta", "target_features_delta", "all_features_level"],
                   default="all_features_delta")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=200000)
    p.add_argument("--max_val_samples", type=int, default=50000)

    # Probe training.
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--probe_dropout", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_npz = np.load(args.latent_dataset)
    raw_npz = np.load(args.raw_dataset)
    seq_key = find_key(lat_npz, ["sequences", "z_seq", "Z_seq", "seqs", "X", "Z"], ndim=3)
    if seq_key is None:
        raise ValueError(f"No latent sequence array found. Keys={lat_npz.files}")
    latent_seq = lat_npz[seq_key].astype(np.float32, copy=False)
    if latent_seq.shape[1] < args.seq_len + 1:
        raise ValueError(f"latent sequences have length {latent_seq.shape[1]}, need {args.seq_len+1}")
    latent_seq = latent_seq[:, :args.seq_len + 1, :]
    n_seq = latent_seq.shape[0]

    # Legacy group ids for fallback split.
    group_arrays = []
    for candidates in [["episode_ids", "episode_id", "episodes", "ep_id"],
                       ["day_ids", "day_id"], ["stock_ids", "stock_id"]]:
        k = find_key(lat_npz, candidates, ndim=1)
        if k is not None and len(lat_npz[k]) == n_seq:
            group_arrays.append(lat_npz[k].astype(np.int64))
    if group_arrays:
        stacked = np.vstack(group_arrays).T
        _, group_id = np.unique(stacked, axis=0, return_inverse=True)
        fallback_split_mode = "grouped"
    else:
        group_id = None
        fallback_split_mode = "random"

    raw_idx_2d, align_mode = load_alignment(lat_npz, n_seq, args.seq_len)

    raw_feat, feature_names, raw_meta = derive_raw_features(raw_npz)
    target_idx, target_names = choose_target_indices(feature_names, args.targets)
    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    if max(lags) > args.seq_len - 1:
        raise ValueError(
            f"max lag {max(lags)} > seq_len-1 = {args.seq_len - 1}. "
            f"Raw baseline needs one extra position before context."
        )

    split_meta = {}
    if args.tokenizer_ckpt is not None:
        tr_idx, va_idx, split_mode, split_meta = split_by_a1t_checkpoint(
            args.tokenizer_ckpt,
            raw_npz,
            raw_idx_2d,
            args.seq_len,
        )
    else:
        tr_idx, va_idx = grouped_split(n_seq, group_id, args.val_frac, args.split_seed)
        split_mode = fallback_split_mode + " independent_probe_split (leakage-prone for trained tokenizers)"

    n_train_before_subsample = int(len(tr_idx))
    n_val_before_subsample = int(len(va_idx))
    tr_idx = subsample(tr_idx, args.max_train_samples, args.seed)
    va_idx = subsample(va_idx, args.max_val_samples, args.seed + 1)

    print("=" * 88)
    print("LATENT → RAW LONG-MEMORY PREDICTIVE PROBE")
    print("=" * 88)
    print(f"latent_dataset : {args.latent_dataset}")
    print(f"raw_dataset    : {args.raw_dataset}")
    if args.tokenizer_ckpt is not None:
        print(f"tokenizer_ckpt : {args.tokenizer_ckpt}")
    print(f"seq_key        : {seq_key}  latent_shape={latent_seq.shape}")
    print(f"raw_shape      : {raw_npz['book'].shape}")
    print(f"alignment      : {align_mode}  raw_idx_2d_shape={raw_idx_2d.shape}")
    alignment_sanity_check(raw_idx_2d, raw_npz, n_check=64)
    print(f"split          : {split_mode}")
    print(f"                 train={len(tr_idx):,} / {n_train_before_subsample:,} before subsample")
    print(f"                 val  ={len(va_idx):,} / {n_val_before_subsample:,} before subsample")
    print(f"targets        : {target_names}")
    print(f"lags           : {lags}")
    print(f"raw baseline   : {args.raw_input_mode}")
    print("=" * 88)

    results = {
        "meta": {
            "args": vars(args),
            "seq_key": seq_key,
            "latent_shape": list(latent_seq.shape),
            "raw_shape": list(raw_npz["book"].shape),
            "alignment": align_mode,
            "feature_names": feature_names,
            "target_names": target_names,
            "split_mode": split_mode,
            "split_meta": split_meta,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "n_train_before_subsample": n_train_before_subsample,
            "n_val_before_subsample": n_val_before_subsample,
            "raw_meta": raw_meta,
        },
        "lags": {},
    }

    for lag in lags:
        t0 = time.time()
        Xlat_tr, Xraw_tr, y_tr = make_arrays_for_lag(
            latent_seq, raw_feat, raw_idx_2d, tr_idx,
            lag, args.seq_len, target_idx, args.raw_input_mode,
        )
        Xlat_va, Xraw_va, y_va = make_arrays_for_lag(
            latent_seq, raw_feat, raw_idx_2d, va_idx,
            lag, args.seq_len, target_idx, args.raw_input_mode,
        )

        print(f"\n[L={lag}] latent probe: X={Xlat_tr.shape} target={y_tr.shape}")
        lat_res = train_probe(Xlat_tr, y_tr, Xlat_va, y_va, target_names,
                              args, seed=args.seed + 1000 + lag)
        print(f"  latent R2 mean={lat_res['r2_mean']:.4f}  " +
              " ".join([f"{k}={v:.3f}" for k, v in lat_res["r2_per_target"].items()]))

        print(f"[L={lag}] raw baseline: X={Xraw_tr.shape} target={y_tr.shape}")
        raw_res = train_probe(Xraw_tr, y_tr, Xraw_va, y_va, target_names,
                              args, seed=args.seed + 2000 + lag)
        print(f"  raw    R2 mean={raw_res['r2_mean']:.4f}  " +
              " ".join([f"{k}={v:.3f}" for k, v in raw_res["r2_per_target"].items()]))

        gap = {k: raw_res["r2_per_target"][k] - lat_res["r2_per_target"][k] for k in target_names}
        ratio = {
            k: (lat_res["r2_per_target"][k] / raw_res["r2_per_target"][k])
            if abs(raw_res["r2_per_target"][k]) > 1e-8 else float("nan")
            for k in target_names
        }
        print(f"  gap raw-lat mean={np.mean(list(gap.values())):.4f}  "
              f"ratio lat/raw mean={np.nanmean(list(ratio.values())):.3f}  "
              f"elapsed={time.time()-t0:.1f}s")

        results["lags"][str(lag)] = {
            "latent": lat_res,
            "raw_baseline": raw_res,
            "gap_raw_minus_latent": gap,
            "ratio_latent_over_raw": ratio,
        }

        with open(out_dir / "latent_raw_probe_results.json", "w") as f:
            json.dump(results, f, indent=2)

    csv_lines = ["lag,probe,target,r2,mse"]
    for lag in lags:
        item = results["lags"][str(lag)]
        for probe_key, probe_name in [("latent", "latent"), ("raw_baseline", "raw")]:
            r2d = item[probe_key]["r2_per_target"]
            msed = item[probe_key]["mse_per_target"]
            for t in target_names:
                csv_lines.append(f"{lag},{probe_name},{t},{r2d[t]:.8f},{msed[t]:.8e}")
    (out_dir / "latent_raw_probe_results.csv").write_text("\n".join(csv_lines))

    if (not args.no_plots) and plt is not None:
        for t in target_names:
            fig, ax = plt.subplots(figsize=(7, 4))
            lat_vals = [results["lags"][str(l)]["latent"]["r2_per_target"][t] for l in lags]
            raw_vals = [results["lags"][str(l)]["raw_baseline"]["r2_per_target"][t] for l in lags]
            ax.plot(lags, lat_vals, marker="o", label="A1 latent history")
            ax.plot(lags, raw_vals, marker="o", label="raw feature history")
            ax.set_xlabel("context length L")
            ax.set_ylabel("validation R²")
            ax.set_title(t)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"r2_vs_lag_{t}.png", dpi=160)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        lat_mean = [results["lags"][str(l)]["latent"]["r2_mean"] for l in lags]
        raw_mean = [results["lags"][str(l)]["raw_baseline"]["r2_mean"] for l in lags]
        ax.plot(lags, lat_mean, marker="o", label="A1 latent history")
        ax.plot(lags, raw_mean, marker="o", label="raw feature history")
        ax.set_xlabel("context length L")
        ax.set_ylabel("mean validation R²")
        ax.set_title("Mean R² over long-memory raw delta targets")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "r2_vs_lag_mean.png", dpi=160)
        plt.close(fig)

    print("\n" + "=" * 88)
    print("Saved results to:", out_dir)
    print("Key file:", out_dir / "latent_raw_probe_results.json")
    print("=" * 88)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compare_f0_f1_latent_readouts.py

Compare two frozen z32 state representations on the same LOBench endpoints and
observable targets:

  F0 / A1-T:
      W_t -> z_A1T in R^32                         (native supervised tokenizer)

  F1 / JEPA-grounded:
      W_t -> H_JEPA in R^{K x S x D} -> state_head -> z_JEPA in R^32
      (frozen grid-Horizon JEPA + frozen grounded state head)

The script evaluates:
  1) frozen native heads:
       A1-T future/vol heads on z_A1T
       JEPA grounded TargetHead on z_JEPA
  2) same fresh probes on both latents:
       linear probe and small MLP probe trained on z -> 22 observable targets
  3) cross-head compatibility:
       A1-T heads on z_JEPA
       JEPA TargetHead on z_A1T

All evaluations use the same grouped stock-day split and the same target
standardizer fit on the probe train split.

Example
-------
python -m scripts.evaluation.compare_f0_f1_latent_readouts \
  --dataset data/lobench_processed.npz \
  --a1t_ckpt checkpoints/tokenizer/a1_T_K20_d32_500k/encoder_best.pt \
  --jepa_ckpt checkpoints/jepa_horizon/v1/best.pt \
  --jepa_head validation/grounded_state_head/jepa_horizon_best/state_head_attn_all_z32.pt \
  --out_dir validation/latent_compare/f0_f1 \
  --max_train_samples 100000 \
  --max_val_samples 50000 \
  --batch_size 512 \
  --probe_batch_size 1024 \
  --probe_epochs 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

# ----------------------------- project imports -----------------------------
try:
    from models.model_tokenizer_t import TokenizerConfigT, LOBAutoTokenizerT  # type: ignore
except Exception as e1:
    try:
        from model_tokenizer_t import TokenizerConfigT, LOBAutoTokenizerT  # type: ignore
    except Exception as e2:
        raise SystemExit(
            "Cannot import A1-T model. Run from project root. Errors: "
            + repr(e1) + " / " + repr(e2)
        )

try:
    from training.historical.train_jepa_horizon import (  # type: ignore
        HorizonJEPAEncoder,
        HorizonJEPAEncoderConfig,
    )
except Exception as e1:
    try:
        from train_jepa_horizon import HorizonJEPAEncoderConfig, HorizonJEPAEncoder  # type: ignore
    except Exception as e2:
        raise SystemExit(
            "Cannot import Horizon JEPA encoder. Run from project root. Errors: "
            + repr(e1) + " / " + repr(e2)
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
except Exception as e1:
    try:
        from train_tokenizer_t import (  # type: ignore
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
    except Exception as e2:
        raise SystemExit(
            "Cannot import tokenizer data utilities. Run from project root. Errors: "
            + repr(e1) + " / " + repr(e2)
        )


# =============================================================================
# Utility functions
# =============================================================================

def robust_torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def maybe_subsample(pos: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n is None or max_n <= 0 or len(pos) <= max_n:
        return pos
    rng = np.random.default_rng(seed)
    out = rng.choice(pos, size=max_n, replace=False)
    out.sort()
    return out


def to_numpy_stats(stock_stats: Dict) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stock_stats.items()}


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    yc = y_true - y_true.mean(axis=0, keepdims=True)
    ss_tot = (yc ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def summarize_r2(target_names: List[str], r2: np.ndarray) -> Dict[str, float]:
    future_idx = [i for i, n in enumerate(target_names) if not n.startswith("realized_vol")]
    vol_idx = [i for i, n in enumerate(target_names) if n.startswith("realized_vol")]
    return {
        "mean_all": float(np.mean(r2)),
        "median_all": float(np.median(r2)),
        "mean_future_delta": float(np.mean(r2[future_idx])) if future_idx else float("nan"),
        "mean_realized_vol": float(np.mean(r2[vol_idx])) if vol_idx else float("nan"),
        "min": float(np.min(r2)),
        "max": float(np.max(r2)),
    }


def standardize_x(x_train: np.ndarray, x_val: np.ndarray, eps: float = 1e-6):
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True) + eps
    return ((x_train - mu) / sd).astype(np.float32), ((x_val - mu) / sd).astype(np.float32), mu, sd


@torch.no_grad()
def latent_diagnostics(z: np.ndarray) -> Dict[str, float]:
    zt = torch.from_numpy(z).float()
    std = zt.std(dim=0)
    zc = zt - zt.mean(dim=0, keepdim=True)
    B, d = zt.shape
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
        "z_norm": float(zt.norm(dim=-1).mean().item()),
        "z_cov_offdiag": float(cov_offdiag),
    }


def concat_standardizers(fut_mu, fut_sd, vol_mu, vol_sd) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.concatenate([np.asarray(fut_mu, dtype=np.float32), np.asarray(vol_mu, dtype=np.float32)], axis=0)
    sd = np.concatenate([np.asarray(fut_sd, dtype=np.float32), np.asarray(vol_sd, dtype=np.float32)], axis=0)
    return mu.astype(np.float32), sd.astype(np.float32)


def convert_pred_standardizer(pred_src_std: np.ndarray, src_mu: np.ndarray, src_sd: np.ndarray,
                              dst_mu: np.ndarray, dst_sd: np.ndarray) -> np.ndarray:
    """Convert predictions expressed in src-standardized target coordinates into
    dst-standardized coordinates."""
    raw = pred_src_std * src_sd.reshape(1, -1) + src_mu.reshape(1, -1)
    return ((raw - dst_mu.reshape(1, -1)) / dst_sd.reshape(1, -1)).astype(np.float32)


# =============================================================================
# JEPA grounded head classes, copied for robust checkpoint loading
# =============================================================================

class AttnAllStateHead(nn.Module):
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
        B = grid.shape[0]
        x = grid.reshape(B, -1, grid.shape[-1])
        k = self.k(x)
        v = self.v(x)
        scores = (k * self.q.view(1, 1, -1)).sum(dim=-1) / math.sqrt(k.shape[-1])
        w = torch.softmax(scores, dim=1)
        r = (w.unsqueeze(-1) * v).sum(dim=1)
        r = self.norm_r(r)
        return self.proj_z(r)

class MeanAllStateHead(nn.Module):
    def __init__(self, token_dim: int, z_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, z_dim), nn.LayerNorm(z_dim),
        )
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid.mean(dim=(1, 2)))

class LastConcatStateHead(nn.Module):
    def __init__(self, K: int, S: int, token_dim: int, z_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        in_dim = S * token_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, z_dim), nn.LayerNorm(z_dim),
        )
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid[:, -1, :, :].reshape(grid.shape[0], -1))

class TargetHead(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
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


def make_state_head(readout: str, K: int, S: int, D: int, z_dim: int, hidden: int, dropout: float) -> nn.Module:
    if readout == "attn_all":
        return AttnAllStateHead(D, z_dim, attn_dim=hidden, dropout=dropout)
    if readout == "mean_all":
        return MeanAllStateHead(D, z_dim, hidden=hidden, dropout=dropout)
    if readout == "last_concat":
        return LastConcatStateHead(K, S, D, z_dim, hidden=max(hidden, 256), dropout=dropout)
    raise ValueError(f"Unknown readout {readout!r}")


# =============================================================================
# Datasets and model loading
# =============================================================================

class RawWindowDataset(Dataset):
    def __init__(self, book: np.ndarray, mid_z: np.ndarray, stock_ids: np.ndarray,
                 valid_t: np.ndarray, stock_stats: Dict[str, np.ndarray], K: int):
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


def load_a1t(ckpt_path: str, device: torch.device):
    ckpt = robust_torch_load(ckpt_path, device)
    cfg_dict = ckpt.get("cfg", ckpt.get("config", {}))
    cfg = TokenizerConfigT.from_dict(cfg_dict) if cfg_dict else TokenizerConfigT()
    model = LOBAutoTokenizerT(cfg).to(device)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [A1-T load] missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            print(f"    unexpected sample: {unexpected[:5]}")
        if missing:
            print(f"    missing sample: {missing[:5]}")
    model.eval()
    return model, ckpt


def load_jepa_encoder(ckpt_path: str, device: torch.device):
    ckpt = robust_torch_load(ckpt_path, device)
    cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    enc = HorizonJEPAEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["online_state_dict"])
    enc.eval()
    return enc, ckpt


def load_jepa_grounded_head(head_path: str, device: torch.device):
    ckpt = robust_torch_load(head_path, device)
    readout = ckpt["readout"]
    z_dim = int(ckpt["z_dim"])
    K, S, D = int(ckpt["K"]), int(ckpt["S"]), int(ckpt["D"])
    state_hidden = int(ckpt.get("state_hidden", 128))
    target_hidden = int(ckpt.get("target_hidden", 256))
    dropout = float(ckpt.get("dropout", 0.0))
    target_names = list(ckpt.get("target_names", []))
    out_dim = len(target_names) if target_names else 22
    state_head = make_state_head(readout, K, S, D, z_dim, state_hidden, dropout)
    target_head = TargetHead(z_dim, out_dim=out_dim, hidden=target_hidden, dropout=dropout)
    model = GroundedStateProbe(state_head, target_head).to(device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.state_head.load_state_dict(ckpt["state_head_state_dict"])
        model.target_head.load_state_dict(ckpt["target_head_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_latents_and_native_preds(
    a1t: LOBAutoTokenizerT,
    jepa: HorizonJEPAEncoder,
    jepa_head: GroundedStateProbe,
    ds_a1t: Dataset,
    ds_jepa: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label: str,
):
    """Extract z_A1T, native A1T predictions, z_JEPA, native JEPA-head predictions.

    ds_a1t and ds_jepa must have the same endpoints/order, but may use different
    stock normalization stats.
    """
    dl_a = DataLoader(ds_a1t, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0)
    dl_j = DataLoader(ds_jepa, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0)
    za, zj, pa, pj = [], [], [], []
    n = 0
    t0 = time.time()
    for (wa, sa), (wj, sj) in zip(dl_a, dl_j):
        wa = wa.to(device, non_blocking=True)
        sa = sa.to(device, non_blocking=True)
        wj = wj.to(device, non_blocking=True)
        sj = sj.to(device, non_blocking=True)
        z_a = a1t.encode(wa, sa)
        pred_a = torch.cat([a1t.future_head(z_a), a1t.vol_head(z_a)], dim=-1)
        grid = jepa(wj, sj, mask=None)
        z_j = jepa_head.encode_state(grid)
        pred_j = jepa_head.target_head(z_j)
        za.append(z_a.detach().cpu())
        zj.append(z_j.detach().cpu())
        pa.append(pred_a.detach().cpu())
        pj.append(pred_j.detach().cpu())
        n += wa.shape[0]
        if n % (batch_size * 20) == 0:
            print(f"  [{label}] extracted {n:,} windows ({time.time()-t0:.1f}s)")
    return (
        torch.cat(za, dim=0).numpy().astype(np.float32),
        torch.cat(zj, dim=0).numpy().astype(np.float32),
        torch.cat(pa, dim=0).numpy().astype(np.float32),
        torch.cat(pj, dim=0).numpy().astype(np.float32),
    )


# =============================================================================
# Probe models
# =============================================================================

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.net(x)

class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_probe(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray,
                x_val: np.ndarray, y_val: np.ndarray, device: torch.device,
                batch_size: int, epochs: int, lr: float, weight_decay: float,
                patience: int, label: str):
    model = model.to(device)
    ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.03)
    best = {"mse": float("inf"), "epoch": 0, "state": None}
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, x_val_t.shape[0], 8192):
                preds.append(model(x_val_t[i:i+8192]))
            pred = torch.cat(preds, dim=0)
            mse = float(F.mse_loss(pred, y_val_t).item())
        if mse < best["mse"] - 1e-6:
            best = {"mse": mse, "epoch": ep, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    preds = []
    with torch.no_grad():
        xv = torch.from_numpy(x_val).float().to(device)
        for i in range(0, xv.shape[0], 8192):
            preds.append(model(xv[i:i+8192]).detach().cpu())
    yhat = torch.cat(preds, dim=0).numpy().astype(np.float32)
    print(f"  {label:24s} best_ep={best['epoch']:3d} val_mse={best['mse']:.5f}")
    return yhat, {"best_epoch": best["epoch"], "best_val_mse": best["mse"]}


def eval_prediction(name: str, y_val: np.ndarray, yhat: np.ndarray, target_names: List[str]):
    r2 = r2_per_target(y_val, yhat)
    summary = summarize_r2(target_names, r2)
    print(f"  {name:28s} R² mean={summary['mean_all']:+.4f}  future={summary['mean_future_delta']:+.4f}  vol={summary['mean_realized_vol']:+.4f}")
    return r2, summary


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Compare F0 A1-T z32 vs F1 JEPA-grounded z32 readouts/probes")
    p.add_argument("--dataset", required=True)
    p.add_argument("--a1t_ckpt", required=True)
    p.add_argument("--jepa_ckpt", required=True)
    p.add_argument("--jepa_head", required=True)
    p.add_argument("--out_dir", default="validation/latent_compare/f0_f1")

    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--max_train_samples", type=int, default=100000)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--future_features", type=str, default="d_spread_z,d_microprice_rel,d_best_bid_rel,d_best_ask_rel,d_top_imbalance")
    p.add_argument("--future_horizons", type=str, default="1,5,10,20")
    p.add_argument("--vol_horizons", type=str, default="5,20")

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")

    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--probe_batch_size", type=int, default=1024)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_weight_decay", type=float, default=1e-3)
    p.add_argument("--probe_patience", type=int, default=15)
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--save_latents", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("COMPARE F0/F1 LATENT READOUTS")
    print("=" * 100)
    print(f"device    : {device}")
    print(f"dataset   : {args.dataset}")
    print(f"A1-T ckpt : {args.a1t_ckpt}")
    print(f"JEPA ckpt : {args.jepa_ckpt}")
    print(f"JEPA head : {args.jepa_head}")

    # ----- load models -----
    print("\n[1/7] Loading frozen models...")
    a1t, a1t_ckpt = load_a1t(args.a1t_ckpt, device)
    jepa, jepa_ckpt = load_jepa_encoder(args.jepa_ckpt, device)
    jepa_head, head_ckpt = load_jepa_grounded_head(args.jepa_head, device)
    print(f"  A1-T: d_latent={a1t.cfg.d_latent}, K={a1t.cfg.K}")
    print(f"  JEPA: K={jepa.cfg.K}, S={jepa.cfg.S}, D={jepa.cfg.d_model}")
    print(f"  head: readout={head_ckpt.get('readout')} z_dim={head_ckpt.get('z_dim')}")

    # ----- raw data and targets -----
    print("\n[2/7] Loading LOBench and building fixed split/targets...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    min_spread = raw["min_spread_z_per_stock"].astype(np.float32, copy=False)
    n_stocks = int(min_spread.shape[0])

    future_features = [x.strip() for x in args.future_features.split(",") if x.strip()]
    future_horizons = [int(x) for x in args.future_horizons.split(",") if x.strip()]
    vol_horizons = [int(x) for x in args.vol_horizons.split(",") if x.strip()]
    max_h = max(max(future_horizons), max(vol_horizons))

    bid_v, ask_v = book[:, 0, :, 1], book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    valid_t = compute_valid_endpoints(stock_ids, day_ids, args.K, max_h, vol_mask)
    train_pos, val_pos = grouped_split_by_stock_day(stock_ids, day_ids, valid_t, args.val_frac, args.split_seed)
    train_pos = maybe_subsample(train_pos, args.max_train_samples, args.seed + 11)
    val_pos = maybe_subsample(val_pos, args.max_val_samples, args.seed + 17)
    train_t = valid_t[train_pos]
    val_t = valid_t[val_pos]
    print(f"  N={len(mid_z):,} valid_t={len(valid_t):,} train={len(train_t):,} val={len(val_t):,}")

    raw_feat, _ = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)
    fut_train_raw = compute_future_feature_targets(raw_feat, train_t, future_features, future_horizons)
    fut_val_raw = compute_future_feature_targets(raw_feat, val_t, future_features, future_horizons)
    vol_train_raw = compute_vol_targets(mid_z, train_t, vol_horizons, min_spread, stock_ids)
    vol_val_raw = compute_vol_targets(mid_z, val_t, vol_horizons, min_spread, stock_ids)
    y_train_raw = np.concatenate([fut_train_raw, vol_train_raw], axis=1).astype(np.float32)
    y_val_raw = np.concatenate([fut_val_raw, vol_val_raw], axis=1).astype(np.float32)
    y_mu, y_sd = fit_target_standardizer(y_train_raw)
    y_train = apply_standardizer(y_train_raw, y_mu, y_sd).astype(np.float32)
    y_val = apply_standardizer(y_val_raw, y_mu, y_sd).astype(np.float32)
    target_names = [f"{f}@{h}" for f in future_features for h in future_horizons] + [f"realized_vol@{h}" for h in vol_horizons]
    print(f"  targets: train={y_train.shape} val={y_val.shape}")

    # ----- stock stats per branch -----
    print("\n[3/7] Resolving branch-specific stock normalization stats...")
    if "stock_stats" in a1t_ckpt and a1t_ckpt["stock_stats"]:
        a1t_stats = to_numpy_stats(a1t_ckpt["stock_stats"])
        print("  A1-T stock_stats: checkpoint")
    else:
        a1t_stats = compute_stock_stats_train_only(book, mid_z, stock_ids, day_ids, train_t, n_stocks)
        print("  A1-T stock_stats: recomputed on probe train")
    if "stock_stats" in jepa_ckpt and jepa_ckpt["stock_stats"]:
        jepa_stats = to_numpy_stats(jepa_ckpt["stock_stats"])
        print("  JEPA stock_stats: checkpoint")
    else:
        jepa_stats = compute_stock_stats_train_only(book, mid_z, stock_ids, day_ids, train_t, n_stocks)
        print("  JEPA stock_stats: recomputed on probe train")

    ds_train_a = RawWindowDataset(book, mid_z, stock_ids, train_t, a1t_stats, args.K)
    ds_val_a = RawWindowDataset(book, mid_z, stock_ids, val_t, a1t_stats, args.K)
    ds_train_j = RawWindowDataset(book, mid_z, stock_ids, train_t, jepa_stats, args.K)
    ds_val_j = RawWindowDataset(book, mid_z, stock_ids, val_t, jepa_stats, args.K)

    # ----- latent extraction -----
    print("\n[4/7] Extracting frozen latents and native predictions...")
    z_a_train, z_j_train, pred_a_train, pred_j_train = extract_latents_and_native_preds(
        a1t, jepa, jepa_head, ds_train_a, ds_train_j, args.batch_size, args.num_workers, device, "train")
    z_a_val, z_j_val, pred_a_val, pred_j_val = extract_latents_and_native_preds(
        a1t, jepa, jepa_head, ds_val_a, ds_val_j, args.batch_size, args.num_workers, device, "val")
    print(f"  z_A1T train/val: {z_a_train.shape} / {z_a_val.shape}")
    print(f"  z_JEPA train/val: {z_j_train.shape} / {z_j_val.shape}")
    print("  diag A1-T val : " + ", ".join(f"{k}={v:.4f}" for k, v in latent_diagnostics(z_a_val).items()))
    print("  diag JEPA val : " + ", ".join(f"{k}={v:.4f}" for k, v in latent_diagnostics(z_j_val).items()))

    # ----- native head standardizer conversions -----
    print("\n[5/7] Evaluating frozen native heads and cross-heads...")
    # A1-T native heads output in A1-T training target standardization.
    ts = a1t_ckpt.get("target_standardizers", {})
    if "future" in ts and "vol" in ts:
        a_mu, a_sd = concat_standardizers(ts["future"]["mean"], ts["future"]["std"], ts["vol"]["mean"], ts["vol"]["std"])
    else:
        print("  [warn] A1-T target standardizers missing; assuming probe standardizer")
        a_mu, a_sd = y_mu, y_sd
    # JEPA grounded target head output in its own training standardization.
    yh = head_ckpt.get("y_standardizer", {})
    if "mean" in yh and "std" in yh:
        j_mu, j_sd = np.asarray(yh["mean"], dtype=np.float32), np.asarray(yh["std"], dtype=np.float32)
    else:
        print("  [warn] JEPA head target standardizer missing; assuming probe standardizer")
        j_mu, j_sd = y_mu, y_sd

    pred_a_native_val = convert_pred_standardizer(pred_a_val, a_mu, a_sd, y_mu, y_sd)
    pred_j_native_val = convert_pred_standardizer(pred_j_val, j_mu, j_sd, y_mu, y_sd)
    rows = []
    r2_tables = {}
    def add_result(name: str, yhat: np.ndarray, kind: str):
        r2, summ = eval_prediction(name, y_val, yhat, target_names)
        r2_tables[name] = {"kind": kind, "r2": r2.tolist(), "summary": summ}
        for i, tn in enumerate(target_names):
            rows.append({"model": name, "kind": kind, "target_name": tn, "r2": float(r2[i])})

    add_result("A1T_z + A1T_native_head", pred_a_native_val, "native")
    add_result("JEPA_z + JEPA_native_head", pred_j_native_val, "native")

    # Cross-heads: dimensions must match.
    with torch.no_grad():
        za_val_t = torch.from_numpy(z_a_val).float().to(device)
        zj_val_t = torch.from_numpy(z_j_val).float().to(device)
        cross_j_on_a = []
        cross_a_on_j = []
        for i in range(0, z_a_val.shape[0], 8192):
            cross_j_on_a.append(jepa_head.target_head(za_val_t[i:i+8192]).detach().cpu())
            cross_a_on_j.append(torch.cat([a1t.future_head(zj_val_t[i:i+8192]), a1t.vol_head(zj_val_t[i:i+8192])], dim=-1).detach().cpu())
        cross_j_on_a = torch.cat(cross_j_on_a, dim=0).numpy().astype(np.float32)
        cross_a_on_j = torch.cat(cross_a_on_j, dim=0).numpy().astype(np.float32)
    add_result("A1T_z + JEPA_head", convert_pred_standardizer(cross_j_on_a, j_mu, j_sd, y_mu, y_sd), "cross")
    add_result("JEPA_z + A1T_head", convert_pred_standardizer(cross_a_on_j, a_mu, a_sd, y_mu, y_sd), "cross")

    # ----- same fresh probes -----
    print("\n[6/7] Training same fresh probes on both z spaces...")
    z_a_tr_s, z_a_va_s, _, _ = standardize_x(z_a_train, z_a_val)
    z_j_tr_s, z_j_va_s, _, _ = standardize_x(z_j_train, z_j_val)

    probe_specs = [
        ("A1T_z + linear_probe", z_a_tr_s, z_a_va_s, LinearProbe(z_a_tr_s.shape[1], y_train.shape[1])),
        ("JEPA_z + linear_probe", z_j_tr_s, z_j_va_s, LinearProbe(z_j_tr_s.shape[1], y_train.shape[1])),
        ("A1T_z + MLP_probe", z_a_tr_s, z_a_va_s, MLPProbe(z_a_tr_s.shape[1], args.mlp_hidden, y_train.shape[1], dropout=0.1)),
        ("JEPA_z + MLP_probe", z_j_tr_s, z_j_va_s, MLPProbe(z_j_tr_s.shape[1], args.mlp_hidden, y_train.shape[1], dropout=0.1)),
    ]
    for name, xtr, xva, model in probe_specs:
        yhat, info = train_probe(model, xtr, y_train, xva, y_val, device,
                                 batch_size=args.probe_batch_size,
                                 epochs=args.probe_epochs,
                                 lr=args.probe_lr,
                                 weight_decay=args.probe_weight_decay,
                                 patience=args.probe_patience,
                                 label=name)
        add_result(name, yhat, "fresh_probe")
        r2_tables[name]["train_info"] = info

    # ----- save outputs -----
    print("\n[7/7] Saving outputs...")
    # compact summary CSV
    with open(out_dir / "summary.csv", "w") as f:
        f.write("model,kind,mean_all,mean_future_delta,mean_realized_vol,median_all,min,max\n")
        for name, rec in r2_tables.items():
            s = rec["summary"]
            f.write(f"{name},{rec['kind']},{s['mean_all']:.6f},{s['mean_future_delta']:.6f},{s['mean_realized_vol']:.6f},{s['median_all']:.6f},{s['min']:.6f},{s['max']:.6f}\n")
    with open(out_dir / "per_target_r2.csv", "w") as f:
        f.write("model,kind,target_name,r2\n")
        for r in rows:
            f.write(f"{r['model']},{r['kind']},{r['target_name']},{r['r2']:.6f}\n")

    meta = {
        "args": vars(args),
        "target_names": target_names,
        "n_train": int(len(train_t)),
        "n_val": int(len(val_t)),
        "a1t_ckpt_epoch": a1t_ckpt.get("epoch", None),
        "jepa_ckpt_epoch": jepa_ckpt.get("epoch", None),
        "jepa_head_meta": {k: head_ckpt.get(k) for k in ["readout", "z_dim", "K", "S", "D", "state_hidden", "target_hidden"]},
        "target_standardizer_probe": {"mean": y_mu.tolist(), "std": y_sd.tolist()},
        "latent_diagnostics": {
            "A1T_val": latent_diagnostics(z_a_val),
            "JEPA_val": latent_diagnostics(z_j_val),
        },
        "results": r2_tables,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_latents:
        np.savez_compressed(
            out_dir / "latents_and_targets.npz",
            train_t=train_t, val_t=val_t,
            z_a1t_train=z_a_train, z_a1t_val=z_a_val,
            z_jepa_train=z_j_train, z_jepa_val=z_j_val,
            y_train=y_train, y_val=y_val,
            y_train_raw=y_train_raw, y_val_raw=y_val_raw,
            target_names=np.array(target_names),
        )
        print(f"  saved latents: {out_dir / 'latents_and_targets.npz'}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for name, rec in r2_tables.items():
        s = rec["summary"]
        print(f"  {name:28s} [{rec['kind']:11s}] mean={s['mean_all']:+.4f} future={s['mean_future_delta']:+.4f} vol={s['mean_realized_vol']:+.4f}")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()

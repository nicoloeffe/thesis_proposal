#!/usr/bin/env python3
"""
probe_anisotropy_vs_nonlinearity.py

Separates two hypotheses that, until now, were fused under "the JEPA organizes
the signal differently / less accessibly":

    (A) ANISOTROPIC-but-LINEAR : the directional signal IS linearly decodable,
        but it lives in low-variance directions of the representation, so
        practical (regularized / rank-limited) readouts miss it.
    (N) GENUINELY NON-LINEAR   : no linear functional recovers it at any scale;
        a capacity-controlled MLP is required.

This is a PROBE / EVAL file. It does NOT modify the trainer, the dataset, or the
existing probes. It imports and reuses them. Run it once per frozen encoder
checkpoint; compare encoders offline from the emitted CSVs.

------------------------------------------------------------------------------
Locked design decisions (agreed before implementation):

1. LINEAR CEILING := min-norm OLS via the pseudo-inverse (ridge with lambda->0),
   NOT ridge-CV. A ceiling must not shrink: ridge shrinkage suppresses exactly
   the low-variance directions we want to detect, which would manufacture false
   non-linearity. The pseudo-inverse is well-defined even when the JEPA
   representation is rank-deficient (null space carries zero variance ->
   zero information -> correctly ignored; low-variance-but-nonzero directions
   are used in full). It is whitening-invariant and invariant to per-dimension
   standardization (both are invertible linear maps; min-norm OLS R^2 is
   unchanged on the row space).

2. AXIS-2 PCA for V90 is computed on the MEAN-CENTERED representation, NOT the
   per-dim-standardized one. Per-dim z-scoring is a diagonal invertible map: it
   is invisible to the ceiling (good) but it rewrites the covariance eigen-
   spectrum, which is precisely the geometry V90 is supposed to measure
   ("signal in low-variance directions of the representation AS THE ENCODER
   PRODUCES IT"). Standardizing before PCA would isotropize per-coordinate
   variances and measure the wrong geometry.

3. WHITENING (ZCA) acts ONLY on the practical / regularized readouts (axis 3),
   never on the ceiling. As a correctness check, the pinv ceiling MUST be
   invariant to whitening; if it moves, eps is too large or the rank handling
   is wrong. The whitening result is reported with the eps actually used.

------------------------------------------------------------------------------
Three axes (all on last_concat512 = 4 last-timestep tokens x 128, per encoder):

  AXIS 1  discriminator      gap_nl(t) = R2_mlp(t) - R2_lin(t)
            R2_lin  = pinv ceiling (closed form, no reg, full rank)
            R2_mlp  = MLP, width sweep to val plateau, weight_decay=0
            gap ~ 0      -> linear  -> read V90 (axis 2)
            gap >> 0 rob -> non-linear (vs the PINV ceiling, not vs z32)

  AXIS 2  spectral location  V90(t) on centered-PCA
            smallest cumulative-variance fraction whose top-k linear fit
            reaches 0.90 * full linear ceiling. small = aligned/high-variance,
            large = anisotropic/low-variance tail.

  AXIS 3  practical readouts  Delta across {ridge-CV, z32-bottleneck, attn-pool}
            + whitening test on the concat-based readouts (ridge-CV): if R2
            rises toward the pinv ceiling after ZCA, the inaccessibility was
            CONDITIONING -> anisotropy. Ceiling-invariance asserted.

  Internal control: realized_vol@{5,20}. The dichotomy must behave as predicted
  on the DIRECTIONAL targets and VANISH on vol. If it appears on vol too, the
  probe is measuring itself, not the representation. Never aggregate directional
  and vol together.

------------------------------------------------------------------------------
Usage
-----
    python -m scripts.evaluation.probe_anisotropy_vs_nonlinearity \
        --dataset data/lobench_processed.npz \
        --encoder_ckpt checkpoints/jepa_horizon/v1_500k/epoch_012.pt \
        --encoder_type jepa \
        --out_dir validation/aniso_vs_nonlin/jepa_h_ep012 \
        --max_train_samples 100000 --max_val_samples 50000 \
        --batch_size 512 --num_workers 2 --probe_seeds 0,1,2

    # supervised encoder (uses the .encoder inside SupervisedGrid):
    python -m scripts.evaluation.probe_anisotropy_vs_nonlinearity \
        --encoder_ckpt checkpoints/supervised_grid/v1/best.pt \
        --encoder_type supervised  ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Path bootstrap (same dance as the sibling probes: makes both                 #
# `from training.X import ...` and `import probe_jepa_horizon_readouts` work)  #
# --------------------------------------------------------------------------- #
_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent, _THIS.parent.parent.parent,
           _THIS.parent.parent.parent.parent, Path.cwd()]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---- Reused building blocks (no re-implementation) ----
from training.historical.train_jepa_horizon import (  # type: ignore
    HorizonJEPAEncoder, HorizonJEPAEncoderConfig,
)
from training.train_tokenizer_t import (  # type: ignore
    compute_valid_endpoints, grouped_split_by_stock_day, derive_raw_features_array,
    fit_target_standardizer, apply_standardizer, compute_stock_stats_train_only,
)
from training.historical.train_supervised_grid import (  # type: ignore
    build_targets, r2_per_target, FUTURE_FEATURES, FUTURE_HORIZONS, VOL_HORIZONS,
    ReadoutConfig, SupervisedGrid,
)
# Heavy lifters reused verbatim from the existing readout probe:
from probe_jepa_horizon_readouts import (  # type: ignore
    RawWindowDataset, extract_token_grids, load_horizon_jepa_encoder,
    standardize_x, MLPProbe, AttentionPoolProbe, LinearBottleneckProbe,
    train_torch_probe, robust_torch_load, to_numpy_stats,
)


# --------------------------------------------------------------------------- #
# Small numeric helpers (the only genuinely new code)                          #
# --------------------------------------------------------------------------- #

def _augment_bias(X: np.ndarray) -> np.ndarray:
    """Append a column of ones for the intercept."""
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)


def linear_ceiling(x_tr: np.ndarray, y_tr: np.ndarray,
                   x_va: np.ndarray, y_va: np.ndarray) -> np.ndarray:
    """Honest LINEAR ceiling: min-norm OLS via the Moore-Penrose pseudo-inverse.

    No regularization, no rank truncation. float64 solve. Handles rank-deficient
    features (np.linalg.lstsq with rcond=None == gelsd == minimum-norm solution).
    Returns per-target R^2 on the val split. Invariant to invertible linear
    reparametrization of x (per-dim standardization, whitening) on the row space.
    """
    Xtr = _augment_bias(x_tr.astype(np.float64))
    Xva = _augment_bias(x_va.astype(np.float64))
    W, *_ = np.linalg.lstsq(Xtr, y_tr.astype(np.float64), rcond=None)
    yhat = (Xva @ W).astype(np.float32)
    return r2_per_target(y_va.astype(np.float32), yhat)


def ridge_cv(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray,
             lambdas: Optional[List[float]] = None) -> Tuple[np.ndarray, float]:
    """Practical regularized linear readout. Closed-form ridge; lambda picked by
    MEAN val R^2 (a *practical* readout, NOT the ceiling). Intercept unpenalized.
    Returns (per-target R^2 at best lambda, best lambda)."""
    if lambdas is None:
        lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    Xtr = _augment_bias(x_tr.astype(np.float64))
    Xva = _augment_bias(x_va.astype(np.float64))
    p = Xtr.shape[1]
    G = Xtr.T @ Xtr
    rhs = Xtr.T @ y_tr.astype(np.float64)
    P = np.eye(p); P[-1, -1] = 0.0  # don't penalize the bias term
    best_r2, best_lam, best_mean = None, None, -np.inf
    for lam in lambdas:
        W = np.linalg.solve(G + lam * P, rhs)
        yhat = (Xva @ W).astype(np.float32)
        r2 = r2_per_target(y_va.astype(np.float32), yhat)
        m = float(np.mean(r2))
        if m > best_mean:
            best_mean, best_r2, best_lam = m, r2, lam
    return best_r2, best_lam


def center_only(train: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-center using TRAIN mean only (no per-dim scaling). For axis-2 PCA."""
    mu = train.mean(axis=0, keepdims=True).astype(np.float32)
    return (train - mu).astype(np.float32), (val - mu).astype(np.float32), mu


def pca_basis(xc_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """PCA on already-centered train features. Returns (components V [p,p],
    cumulative explained-variance fraction [p])."""
    # SVD of centered data: columns of Vt.T are principal directions.
    _, s, Vt = np.linalg.svd(xc_tr.astype(np.float64), full_matrices=False)
    ev = (s ** 2)
    cumvar = np.cumsum(ev) / np.sum(ev)
    return Vt.T.astype(np.float64), cumvar.astype(np.float64)


def _k_grid(p: int) -> List[int]:
    g = list(range(1, min(20, p) + 1))
    g += list(range(22, min(64, p) + 1, 2))
    g += list(range(72, p + 1, 8))
    if p not in g:
        g.append(p)
    return sorted(set(g))


def v90(xc_tr: np.ndarray, y_tr: np.ndarray, xc_va: np.ndarray, y_va: np.ndarray,
        r2_full_centered: np.ndarray, frac: float = 0.90, eps_r2: float = 0.01
        ) -> Tuple[np.ndarray, Dict]:
    """V90 per target: smallest cumulative-variance fraction (top-k centered PCA)
    whose linear (pinv) fit reaches frac * full linear ceiling. First crossing on
    val. Targets with full ceiling <= eps_r2 are returned as NaN (no signal to
    locate). Also returns the full R2-vs-cumvar curve per target for plotting.
    """
    V, cumvar = pca_basis(xc_tr)
    p = V.shape[0]
    ks = _k_grid(p)
    n_t = y_tr.shape[1]
    curve_r2 = np.full((len(ks), n_t), np.nan, dtype=np.float64)
    for i, k in enumerate(ks):
        Vk = V[:, :k]
        ztr = xc_tr.astype(np.float64) @ Vk
        zva = xc_va.astype(np.float64) @ Vk
        curve_r2[i] = linear_ceiling(ztr, y_tr, zva, y_va)
    out = np.full(n_t, np.nan, dtype=np.float64)
    for t in range(n_t):
        if not np.isfinite(r2_full_centered[t]) or r2_full_centered[t] <= eps_r2:
            continue
        thresh = frac * r2_full_centered[t]
        hit = np.where(curve_r2[:, t] >= thresh)[0]
        out[t] = float(cumvar[ks[hit[0]] - 1]) if len(hit) else 1.0
    return out, {"ks": ks, "cumvar_at_k": [float(cumvar[k - 1]) for k in ks],
                 "curve_r2": curve_r2.tolist()}


def zca_whiten(xc_tr: np.ndarray, xc_va: np.ndarray, eps: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """ZCA whitening from TRAIN covariance: W = U (Lambda+eps)^(-1/2) U^T, applied
    to centered features (stays in the original coordinate frame). eps regularizes
    the rank-deficient spectrum and is reported by the caller."""
    Xc = xc_tr.astype(np.float64)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    w_, U = np.linalg.eigh(cov)
    w_ = np.clip(w_, 0.0, None)
    Wmat = U @ np.diag(1.0 / np.sqrt(w_ + eps)) @ U.T
    return (Xc @ Wmat).astype(np.float32), (xc_va.astype(np.float64) @ Wmat).astype(np.float32)


def mlp_ceiling(x_tr, y_tr, x_va, y_va, device, widths: List[int],
                plateau_tol: float, seed: int, epochs: int, patience: int,
                batch_size: int) -> Tuple[np.ndarray, int, Dict]:
    """Non-linear ceiling: MLP with width swept until the val mean-R^2 gain
    between successive widths drops below plateau_tol. weight_decay=0 (for a
    ceiling the honest regularizer is early-stopping, not L2). Returns
    (best per-target R^2, width at plateau, curve)."""
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim, out_dim = x_tr.shape[1], y_tr.shape[1]
    curve = {}
    best_r2, best_w, prev_mean = None, None, -np.inf
    for w in widths:
        model = MLPProbe(in_dim=in_dim, hidden=w, out_dim=out_dim, dropout=0.1)
        yhat, _info = train_torch_probe(
            model, x_tr, y_tr, x_va, y_va, device,
            batch_size=batch_size, epochs=epochs, lr=1e-3,
            weight_decay=0.0, patience=patience, label=f"mlp_w{w}",
        )
        r2 = r2_per_target(y_va, yhat)
        m = float(np.mean(r2))
        curve[w] = m
        if best_r2 is None or m > float(np.mean(best_r2)):
            best_r2, best_w = r2, w
        if m - prev_mean < plateau_tol and prev_mean > -np.inf:
            break  # plateau reached
        prev_mean = m
    return best_r2, best_w, {"width_mean_r2": curve}


def permutation_floor(x_tr, y_tr, x_va, y_va, seed: int = 0) -> np.ndarray:
    """Sanity floor: linear ceiling on label-permuted train. Should be ~0.
    Guards against a probe manufacturing signal from capacity/noise."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y_tr.shape[0])
    return linear_ceiling(x_tr, y_tr[perm], x_va, y_va)


# --------------------------------------------------------------------------- #
# Encoder loading (jepa or supervised) + setup mirrored from compare_*         #
# --------------------------------------------------------------------------- #

def load_encoder(ckpt_path: str, encoder_type: str, device: torch.device):
    """Return (encoder_module, ckpt). For 'supervised' we take SupervisedGrid.encoder.
    The encoder exposes forward(book, stock_ids, mask=None) -> (B,K,S,D), matching
    what extract_token_grids expects."""
    if encoder_type == "jepa":
        enc, ckpt = load_horizon_jepa_encoder(ckpt_path, device)
        return enc, ckpt
    elif encoder_type == "supervised":
        ckpt = robust_torch_load(ckpt_path, device)
        enc_cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
        rd_cfg = ReadoutConfig(**ckpt["readout_cfg"])
        model = SupervisedGrid(enc_cfg, rd_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model.encoder, ckpt
    raise ValueError(f"unknown encoder_type: {encoder_type}")


def setup_data(args, ckpt, device):
    """Build val+train endpoints, observable targets, and per-stock stats.
    Mirrors the verified recipe in compare_supervised_jepa / probe_jepa_*."""
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    min_spread = raw["min_spread_z_per_stock"].astype(np.float32)
    n_stocks = int(min_spread.shape[0])
    raw_feat, _ = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)

    bid_v, ask_v = book[:, 0, :, 1], book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & \
               (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    K = int(ckpt["enc_cfg"]["K"]) if "enc_cfg" in ckpt and "K" in ckpt["enc_cfg"] \
        else int(getattr(getattr(ckpt, "cfg", None), "K", 20) or 20)
    max_h = max(max(FUTURE_HORIZONS), max(VOL_HORIZONS))
    valid_t = compute_valid_endpoints(stock_ids, day_ids, K, max_h, vol_mask)
    train_pos, val_pos = grouped_split_by_stock_day(
        stock_ids, day_ids, valid_t, args.val_frac, args.seed)

    rng = np.random.default_rng(args.seed + 1)
    def subsample(pos, cap):
        if cap > 0 and len(pos) > cap:
            return np.sort(rng.choice(pos, cap, replace=False))
        return pos
    train_pos = subsample(train_pos, args.max_train_samples)
    val_pos = subsample(val_pos, args.max_val_samples)
    train_t, val_t = valid_t[train_pos], valid_t[val_pos]

    y_tr_raw, names = build_targets(book, mid_z, stock_ids, train_t, raw_feat, min_spread)
    y_va_raw, _ = build_targets(book, mid_z, stock_ids, val_t, raw_feat, min_spread)
    y_mu, y_sd = fit_target_standardizer(y_tr_raw)
    y_tr = apply_standardizer(y_tr_raw, y_mu, y_sd).astype(np.float32)
    y_va = apply_standardizer(y_va_raw, y_mu, y_sd).astype(np.float32)

    if "stock_stats" in ckpt:
        stock_stats = to_numpy_stats(ckpt["stock_stats"])
    else:
        stock_stats = compute_stock_stats_train_only(
            book, mid_z, stock_ids, day_ids, train_t, n_stocks)

    # directional = future feature deltas; vol = internal control.
    n_dir = len(FUTURE_FEATURES) * len(FUTURE_HORIZONS)
    is_vol = np.array([i >= n_dir for i in range(len(names))])
    return dict(book=book, mid_z=mid_z, stock_ids=stock_ids, K=K,
                train_t=train_t, val_t=val_t, stock_stats=stock_stats,
                y_tr=y_tr, y_va=y_va, target_names=names, is_vol=is_vol)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Anisotropy vs non-linearity probe")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder_ckpt", required=True)
    ap.add_argument("--encoder_type", choices=["jepa", "supervised"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_train_samples", type=int, default=100000)
    ap.add_argument("--max_val_samples", type=int, default=50000)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--vol_clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=2)
    # axis-1 MLP ceiling
    ap.add_argument("--mlp_widths", type=str, default="64,128,256,512,1024")
    ap.add_argument("--mlp_plateau_tol", type=float, default=0.005)
    ap.add_argument("--mlp_epochs", type=int, default=80)
    ap.add_argument("--mlp_patience", type=int, default=15)
    ap.add_argument("--probe_seeds", type=str, default="0,1,2")
    # axis-3
    ap.add_argument("--z_bottleneck", type=int, default=32)
    ap.add_argument("--whiten_eps", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    widths = [int(x) for x in args.mlp_widths.split(",") if x.strip()]
    seeds = [int(x) for x in args.probe_seeds.split(",") if x.strip()]

    print("=" * 92)
    print(f"ANISOTROPY vs NON-LINEARITY  |  {args.encoder_type}  |  {args.encoder_ckpt}")
    print("=" * 92)

    print("[1/5] Loading encoder + data setup...")
    encoder, ckpt = load_encoder(args.encoder_ckpt, args.encoder_type, device)
    D = setup_data(args, ckpt, device)
    names, is_vol = D["target_names"], D["is_vol"]

    print("[2/5] Extracting token grids (last_concat512, last_tokens)...")
    ds_tr = RawWindowDataset(D["book"], D["mid_z"], D["stock_ids"], D["train_t"],
                             D["stock_stats"], D["K"])
    ds_va = RawWindowDataset(D["book"], D["mid_z"], D["stock_ids"], D["val_t"],
                             D["stock_stats"], D["K"])
    r_tr = extract_token_grids(encoder, ds_tr, args.batch_size, args.num_workers, device, "train")
    r_va = extract_token_grids(encoder, ds_va, args.batch_size, args.num_workers, device, "val")
    concat_tr, concat_va = r_tr["last_concat512"], r_va["last_concat512"]
    tok_tr, tok_va = r_tr["last_tokens"], r_va["last_tokens"]
    y_tr, y_va = D["y_tr"], D["y_va"]

    # standardized concat -> ceiling + practical readouts ; centered concat -> V90 + whitening
    xs_tr, xs_va, _, _ = standardize_x(concat_tr, concat_va)
    xc_tr, xc_va, _ = center_only(concat_tr, concat_va)

    print("[3/5] AXIS 1 - linear ceiling (pinv) and non-linear ceiling (MLP sweep)...")
    r2_lin = linear_ceiling(xs_tr, y_tr, xs_va, y_va)
    r2_lin_centered = linear_ceiling(xc_tr, y_tr, xc_va, y_va)  # used as V90 reference
    floor = permutation_floor(xs_tr, y_tr, xs_va, y_va, seed=0)
    mlp_seeds = []
    best_widths = []
    for s in seeds:
        r2_mlp_s, bw, _curve = mlp_ceiling(
            xs_tr, y_tr, xs_va, y_va, device, widths,
            args.mlp_plateau_tol, s, args.mlp_epochs, args.mlp_patience, args.batch_size)
        mlp_seeds.append(r2_mlp_s); best_widths.append(bw)
    r2_mlp = np.mean(np.stack(mlp_seeds, axis=0), axis=0)
    r2_mlp_std = np.std(np.stack(mlp_seeds, axis=0), axis=0)
    gap_nl = r2_mlp - r2_lin

    print("[4/5] AXIS 2 - V90 spectral location (centered PCA)...")
    v90_vals, v90_curve = v90(xc_tr, y_tr, xc_va, y_va, r2_lin_centered)

    print("[5/5] AXIS 3 - practical readouts + whitening...")
    r2_ridge, best_lam = ridge_cv(xs_tr, y_tr, xs_va, y_va)
    # z32 low-rank practical readout (the OLD 'linear' probe, reclassified):
    torch.manual_seed(0)
    z_model = LinearBottleneckProbe(in_dim=xs_tr.shape[1], z_dim=args.z_bottleneck,
                                    out_dim=y_tr.shape[1])
    z_yhat, _ = train_torch_probe(z_model, xs_tr, y_tr, xs_va, y_va, device,
                                  batch_size=1024, epochs=80, lr=1e-3,
                                  weight_decay=1e-3, patience=15, label=f"z{args.z_bottleneck}")
    r2_z = r2_per_target(y_va, z_yhat)
    # attention-pool practical readout on last_tokens:
    torch.manual_seed(0)
    ap_model = AttentionPoolProbe(token_dim=tok_tr.shape[-1], z_dim=args.z_bottleneck,
                                  out_dim=y_tr.shape[1], n_tokens=tok_tr.shape[1])
    ap_yhat, _ = train_torch_probe(ap_model, tok_tr, y_tr, tok_va, y_va, device,
                                   batch_size=1024, epochs=80, lr=1e-3,
                                   weight_decay=1e-3, patience=15, label="attnpool")
    r2_attn = r2_per_target(y_va, ap_yhat)

    practical = np.stack([r2_ridge, r2_z, r2_attn], axis=0)  # (3, n_t)
    delta_readout = practical.max(axis=0) - practical.min(axis=0)

    # whitening test: only on concat-based readouts. Ridge-CV must rise toward the
    # ceiling if inaccessibility was conditioning. Ceiling MUST stay invariant.
    xw_tr, xw_va = zca_whiten(xc_tr, xc_va, args.whiten_eps)
    r2_ridge_white, lam_white = ridge_cv(xw_tr, y_tr, xw_va, y_va)
    r2_lin_white = linear_ceiling(xw_tr, y_tr, xw_va, y_va)
    ceiling_invariance_max_abs = float(np.max(np.abs(r2_lin_white - r2_lin_centered)))

    # ----------------------------- emit -----------------------------
    def grp(mask, arr):  # mean over a target group, ignoring NaN
        v = arr[mask]; v = v[np.isfinite(v)]
        return float(np.mean(v)) if len(v) else float("nan")
    dir_mask, vol_mask_t = ~is_vol, is_vol

    per_target = []
    for i, nm in enumerate(names):
        per_target.append(dict(
            target=nm, is_vol=bool(is_vol[i]),
            r2_lin=float(r2_lin[i]), r2_mlp=float(r2_mlp[i]),
            r2_mlp_std=float(r2_mlp_std[i]), gap_nl=float(gap_nl[i]),
            perm_floor=float(floor[i]), v90=float(v90_vals[i]),
            r2_ridge=float(r2_ridge[i]), r2_z=float(r2_z[i]),
            r2_attn=float(r2_attn[i]), delta_readout=float(delta_readout[i]),
            r2_ridge_white=float(r2_ridge_white[i]),
        ))

    import csv
    with open(out_dir / "per_target.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_target[0].keys()))
        w.writeheader(); w.writerows(per_target)

    # V90-vs-ceiling scatter table (apples-to-apples guard): compare V90 only at
    # matched ceiling. One row per target with (r2_lin_centered, v90).
    with open(out_dir / "v90_vs_ceiling.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["target", "is_vol", "r2_lin_centered", "v90"])
        for i, nm in enumerate(names):
            w.writerow([nm, int(is_vol[i]), f"{r2_lin_centered[i]:.6f}", f"{v90_vals[i]:.6f}"])

    summary = dict(
        encoder_type=args.encoder_type, encoder_ckpt=args.encoder_ckpt,
        n_train=int(len(D["train_t"])), n_val=int(len(D["val_t"])),
        best_mlp_widths=best_widths, ridge_best_lambda=best_lam,
        ridge_best_lambda_white=lam_white, whiten_eps=args.whiten_eps,
        ceiling_invariance_max_abs=ceiling_invariance_max_abs,
        directional=dict(
            r2_lin=grp(dir_mask, r2_lin), r2_mlp=grp(dir_mask, r2_mlp),
            gap_nl=grp(dir_mask, gap_nl), v90=grp(dir_mask, v90_vals),
            delta_readout=grp(dir_mask, delta_readout),
            r2_ridge=grp(dir_mask, r2_ridge), r2_ridge_white=grp(dir_mask, r2_ridge_white),
        ),
        vol_control=dict(
            r2_lin=grp(vol_mask_t, r2_lin), r2_mlp=grp(vol_mask_t, r2_mlp),
            gap_nl=grp(vol_mask_t, gap_nl), v90=grp(vol_mask_t, v90_vals),
            delta_readout=grp(vol_mask_t, delta_readout),
        ),
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "v90_curve.json", "w") as f:
        json.dump({"target_names": names, **v90_curve}, f)

    # ----------------------------- report -----------------------------
    print("\n" + "=" * 92); print("SUMMARY (directional targets | vol control)"); print("=" * 92)
    print(f"  ceiling-invariance under whitening (max |dR2|): {ceiling_invariance_max_abs:.4f}  "
          f"({'OK' if ceiling_invariance_max_abs < 0.01 else 'WARN: eps too large / rank issue'})")
    sd, sv = summary["directional"], summary["vol_control"]
    print(f"  {'':<16}{'R2_lin':>9}{'R2_mlp':>9}{'gap_nl':>9}{'V90':>9}{'D_read':>9}")
    print(f"  {'directional':<16}{sd['r2_lin']:>9.3f}{sd['r2_mlp']:>9.3f}"
          f"{sd['gap_nl']:>9.3f}{sd['v90']:>9.3f}{sd['delta_readout']:>9.3f}")
    print(f"  {'vol (control)':<16}{sv['r2_lin']:>9.3f}{sv['r2_mlp']:>9.3f}"
          f"{sv['gap_nl']:>9.3f}{sv['v90']:>9.3f}{sv['delta_readout']:>9.3f}")
    print(f"  ridge directional: {sd['r2_ridge']:.3f} -> whitened {sd['r2_ridge_white']:.3f} "
          f"(rise = conditioning/anisotropy)")
    print(f"\n  interpretation guide (directional):")
    print(f"    gap_nl ~ 0  -> LINEAR  -> read V90: small=aligned, large=anisotropic")
    print(f"    gap_nl >> 0 -> NON-LINEAR (vs the pinv ceiling, not z32)")
    print(f"  written: per_target.csv, v90_vs_ceiling.csv, summary.json, v90_curve.json -> {out_dir}")


if __name__ == "__main__":
    main()

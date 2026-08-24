#!/usr/bin/env python3
"""
pacf_analysis.py — PACF / temporal-dependence diagnostics for latent WM datasets.

Reads a prebuilt latent sequence dataset, typically:
    data/wm_lobench_tokenizer_a1_cov.npz
with sequences shaped (N, seq_len+1, d_latent), and computes:
  - marginal PACF for latent levels s_t per dimension
  - marginal PACF for residuals / deltas Δs_t per dimension
  - effective lag summaries using practical and significance thresholds
  - cross-lag correlation matrices Corr(x_t^i, x_{t-k}^j), mainly lag 1 by default
  - optional group/regime summaries using stock/day/episode ids when present

Important note:
PACF is estimated from the available prebuilt windows. If windows overlap heavily,
this is still useful diagnostically, but not a perfect replacement for true raw
contiguous trajectories. When raw_indices/day_ids/stock_ids are available, the script
can reduce duplication by taking one representative stride-subsample per group.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def find_key(npz, candidates, ndim: Optional[int] = None):
    for k in candidates:
        if k in npz.files and (ndim is None or npz[k].ndim == ndim):
            return k
    return None


def robust_standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.nanmean(x)) / (np.nanstd(x) + eps)


def autocorr_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation rho[0:max_lag+1] using FFT, normalized with rho[0]=1."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        out = np.zeros(max_lag + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    x = x - x.mean()
    var = np.dot(x, x)
    if var <= 1e-30:
        out = np.zeros(max_lag + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    size = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, size)
    acov = np.fft.irfft(fx * np.conjugate(fx), size)[: max_lag + 1]
    # Biased normalization. Good enough and stable for PACF diagnostics.
    rho = acov / acov[0]
    rho[0] = 1.0
    return rho


def pacf_yw(x: np.ndarray, max_lag: int) -> np.ndarray:
    """PACF via Levinson-Durbin/Yule-Walker recursion. Returns pacf[0..max_lag]."""
    rho = autocorr_fft(x, max_lag)
    pacf = np.zeros(max_lag + 1, dtype=np.float64)
    pacf[0] = 1.0
    if max_lag == 0:
        return pacf

    phi = np.zeros(max_lag + 1, dtype=np.float64)
    sigma = 1.0
    for k in range(1, max_lag + 1):
        if k == 1:
            alpha = rho[1]
        else:
            numer = rho[k] - np.dot(phi[1:k], rho[1:k][::-1])
            alpha = numer / max(sigma, 1e-12)
        alpha = float(np.clip(alpha, -0.999999, 0.999999))
        old_phi = phi.copy()
        phi[k] = alpha
        if k > 1:
            phi[1:k] = old_phi[1:k] - alpha * old_phi[k - 1 : 0 : -1]
        sigma *= max(1.0 - alpha * alpha, 1e-12)
        pacf[k] = alpha
    return pacf


def effective_lag(pacf: np.ndarray, sig_thr: float, practical_thr: float, min_lag: int = 1) -> Dict[str, int]:
    lags = np.arange(len(pacf))
    valid = lags >= min_lag
    abs_p = np.abs(pacf)
    sig_lags = lags[valid & (abs_p > sig_thr)]
    practical_lags = lags[valid & (abs_p > practical_thr)]
    return {
        "last_sig_lag": int(sig_lags.max()) if sig_lags.size else 0,
        "last_practical_lag": int(practical_lags.max()) if practical_lags.size else 0,
        "first_below_practical_after_1": int(next((int(k) for k in range(1, len(pacf)) if abs_p[k] < practical_thr), 0)),
    }


def flatten_sequences(seqs: np.ndarray, max_tokens: int, seed: int) -> np.ndarray:
    """Flatten windows into token cloud/time-ish stream. Diagnostic, not perfect raw trajectory."""
    N, T, D = seqs.shape
    flat = seqs.reshape(-1, D)
    if max_tokens and flat.shape[0] > max_tokens:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(flat.shape[0], size=max_tokens, replace=False))
        flat = flat[idx]
    return flat.astype(np.float64, copy=False)


def build_group_ids(npz, n: int) -> Tuple[Optional[np.ndarray], Dict]:
    meta = {}
    group_arrays = []
    for candidates in [["episode_ids", "episode_id", "episodes", "ep_id"], ["day_ids", "day_id", "days"], ["stock_ids", "stock_id", "stocks"]]:
        k = find_key(npz, candidates, ndim=1)
        if k is not None and len(npz[k]) == n:
            group_arrays.append(npz[k].astype(np.int64))
            meta[f"group_key_{len(group_arrays)}"] = k
    if not group_arrays:
        return None, meta
    stacked = np.vstack(group_arrays).T
    _, gid = np.unique(stacked, axis=0, return_inverse=True)
    meta["n_groups"] = int(gid.max() + 1)
    return gid, meta


def representative_windows(seqs: np.ndarray, group_id: Optional[np.ndarray], max_windows: int, seed: int) -> np.ndarray:
    """Subsample windows with light grouping awareness."""
    N = len(seqs)
    if max_windows <= 0 or N <= max_windows:
        return seqs
    rng = np.random.default_rng(seed)
    if group_id is None:
        idx = np.sort(rng.choice(N, size=max_windows, replace=False))
        return seqs[idx]

    unique = np.unique(group_id)
    per_group = max(1, int(math.ceil(max_windows / len(unique))))
    chosen = []
    for g in unique:
        ids = np.where(group_id == g)[0]
        if ids.size <= per_group:
            chosen.extend(ids.tolist())
        else:
            # spread across group order rather than pure random, to reduce overlapping-window redundancy
            take_pos = np.linspace(0, ids.size - 1, per_group).round().astype(int)
            chosen.extend(ids[take_pos].tolist())
    chosen = np.array(chosen, dtype=np.int64)
    if chosen.size > max_windows:
        chosen = np.sort(rng.choice(chosen, size=max_windows, replace=False))
    else:
        chosen = np.sort(chosen)
    return seqs[chosen]


def cross_lag_corr(x: np.ndarray, lag: int) -> np.ndarray:
    """Corr matrix C[i,j]=corr(x_t^i, x_{t-lag}^j). x: (T,D)."""
    if lag <= 0:
        return np.corrcoef(x, rowvar=False)
    cur = x[lag:]
    past = x[:-lag]
    cur = (cur - cur.mean(axis=0)) / (cur.std(axis=0) + 1e-12)
    past = (past - past.mean(axis=0)) / (past.std(axis=0) + 1e-12)
    return (cur.T @ past) / max(cur.shape[0] - 1, 1)


def summarize_pacf_matrix(pacf_mat: np.ndarray, n_eff: int, practical_thr: float) -> Dict:
    sig_thr = 2.0 / math.sqrt(max(n_eff, 1))
    per_dim = []
    for d in range(pacf_mat.shape[0]):
        eff = effective_lag(pacf_mat[d], sig_thr=sig_thr, practical_thr=practical_thr)
        eff["dim"] = d
        eff["max_abs_pacf_lag1_plus"] = float(np.max(np.abs(pacf_mat[d, 1:]))) if pacf_mat.shape[1] > 1 else 0.0
        per_dim.append(eff)
    last_sig = np.array([r["last_sig_lag"] for r in per_dim])
    last_pr = np.array([r["last_practical_lag"] for r in per_dim])
    return {
        "n_eff": int(n_eff),
        "significance_threshold_approx": float(sig_thr),
        "practical_threshold": float(practical_thr),
        "last_sig_lag_mean": float(last_sig.mean()),
        "last_sig_lag_median": float(np.median(last_sig)),
        "last_sig_lag_max": int(last_sig.max()),
        "last_practical_lag_mean": float(last_pr.mean()),
        "last_practical_lag_median": float(np.median(last_pr)),
        "last_practical_lag_max": int(last_pr.max()),
        "per_dim": per_dim,
    }


def save_plots(out_dir: Path, pacf_level: np.ndarray, pacf_delta: np.ndarray, cross_level: Dict[int, np.ndarray], cross_delta: Dict[int, np.ndarray], max_lag: int):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    lags = np.arange(max_lag + 1)

    def plot_pacf(mat, name):
        fig, ax = plt.subplots(figsize=(10, 5))
        for d in range(mat.shape[0]):
            ax.plot(lags[1:], mat[d, 1:], alpha=0.55, linewidth=1)
        ax.axhline(0.05, linestyle="--", linewidth=1)
        ax.axhline(-0.05, linestyle="--", linewidth=1)
        ax.set_title(f"PACF by latent dimension — {name}")
        ax.set_xlabel("lag")
        ax.set_ylabel("PACF")
        fig.tight_layout()
        fig.savefig(out_dir / f"pacf_{name}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        q = np.quantile(np.abs(mat[:, 1:]), [0.25, 0.5, 0.75, 0.9, 0.95], axis=0)
        x = np.arange(1, max_lag + 1)
        ax.plot(x, q[1], label="median |PACF|")
        ax.fill_between(x, q[0], q[2], alpha=0.25, label="IQR")
        ax.plot(x, q[3], linestyle="--", label="q90")
        ax.plot(x, q[4], linestyle=":", label="q95")
        ax.axhline(0.05, linestyle="--", linewidth=1, color="black")
        ax.set_title(f"PACF absolute quantiles — {name}")
        ax.set_xlabel("lag")
        ax.set_ylabel("|PACF|")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"pacf_abs_quantiles_{name}.png", dpi=160)
        plt.close(fig)

    plot_pacf(pacf_level, "level")
    plot_pacf(pacf_delta, "delta")

    for lag, mat in cross_level.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"Cross-lag corr level: Corr(s_t^i, s_t-{lag}^j)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"cross_lag_corr_level_lag{lag}.png", dpi=160)
        plt.close(fig)
    for lag, mat in cross_delta.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"Cross-lag corr delta: Corr(ds_t^i, ds_t-{lag}^j)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"cross_lag_corr_delta_lag{lag}.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", default="validation/world_model/pacf_analysis")
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--max_lag", type=int, default=50)
    ap.add_argument("--max_windows", type=int, default=200000, help="Subsample windows before flattening. 0=all.")
    ap.add_argument("--max_tokens", type=int, default=2000000, help="Subsample flattened tokens for PACF. 0=all.")
    ap.add_argument("--practical_thr", type=float, default=0.05)
    ap.add_argument("--cross_lags", type=str, default="1,2,5,10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.dataset)
    seq_key = find_key(npz, ["sequences", "z_seq", "Z_seq", "seqs", "X", "Z"], ndim=3)
    if seq_key is None:
        raise ValueError(f"No 3D sequence key found. Keys={npz.files}")
    seqs = npz[seq_key].astype(np.float32)
    need_len = args.seq_len + 1
    if seqs.shape[1] >= need_len:
        seqs = seqs[:, :need_len]
    N, T, D = seqs.shape
    group_id, group_meta = build_group_ids(npz, N)
    seqs_sub = representative_windows(seqs, group_id, args.max_windows, args.seed)

    print("=" * 88)
    print("PACF / TEMPORAL DEPENDENCE ANALYSIS FOR LATENT WM DATASET")
    print("=" * 88)
    print(f"dataset      : {args.dataset}")
    print(f"seq_key      : {seq_key}")
    print(f"full shape   : {seqs.shape}")
    print(f"used windows : {seqs_sub.shape[0]:,}")
    print(f"d_latent     : {D}")
    print(f"max_lag      : {args.max_lag}")
    print(f"groups       : {group_meta}")

    level = flatten_sequences(seqs_sub, args.max_tokens, args.seed)
    delta = np.diff(seqs_sub, axis=1).reshape(-1, D)
    if args.max_tokens and delta.shape[0] > args.max_tokens:
        rng = np.random.default_rng(args.seed + 1)
        idx = np.sort(rng.choice(delta.shape[0], size=args.max_tokens, replace=False))
        delta = delta[idx]

    pacf_level = np.zeros((D, args.max_lag + 1), dtype=np.float64)
    pacf_delta = np.zeros((D, args.max_lag + 1), dtype=np.float64)
    for d in range(D):
        pacf_level[d] = pacf_yw(level[:, d], args.max_lag)
        pacf_delta[d] = pacf_yw(delta[:, d], args.max_lag)

    cross_lag_list = [int(x.strip()) for x in args.cross_lags.split(",") if x.strip()]
    cross_level = {lag: cross_lag_corr(level, lag) for lag in cross_lag_list}
    cross_delta = {lag: cross_lag_corr(delta, lag) for lag in cross_lag_list}

    summary = {
        "dataset": args.dataset,
        "seq_key": seq_key,
        "full_shape": list(seqs.shape),
        "used_windows": int(seqs_sub.shape[0]),
        "level_tokens": int(level.shape[0]),
        "delta_tokens": int(delta.shape[0]),
        "d_latent": int(D),
        "max_lag": int(args.max_lag),
        "group_meta": group_meta,
        "level": summarize_pacf_matrix(pacf_level, n_eff=level.shape[0], practical_thr=args.practical_thr),
        "delta": summarize_pacf_matrix(pacf_delta, n_eff=delta.shape[0], practical_thr=args.practical_thr),
        "cross_lag_level": {},
        "cross_lag_delta": {},
    }

    for lag, mat in cross_level.items():
        off = mat.copy()
        # keep all cross-dim entries, exclude same-dim autocorr-ish diagonal for summary
        mask = ~np.eye(D, dtype=bool)
        vals = np.abs(off[mask])
        summary["cross_lag_level"][str(lag)] = {
            "mean_abs_offdiag": float(vals.mean()),
            "max_abs_offdiag": float(vals.max()),
            "fro_offdiag": float(np.linalg.norm(off[mask])),
        }
    for lag, mat in cross_delta.items():
        mask = ~np.eye(D, dtype=bool)
        vals = np.abs(mat[mask])
        summary["cross_lag_delta"][str(lag)] = {
            "mean_abs_offdiag": float(vals.mean()),
            "max_abs_offdiag": float(vals.max()),
            "fro_offdiag": float(np.linalg.norm(mat[mask])),
        }

    print("\n[Effective PACF order — levels]")
    print(f"  last practical lag mean/median/max : {summary['level']['last_practical_lag_mean']:.2f} / {summary['level']['last_practical_lag_median']:.1f} / {summary['level']['last_practical_lag_max']}")
    print(f"  last significant lag mean/median/max: {summary['level']['last_sig_lag_mean']:.2f} / {summary['level']['last_sig_lag_median']:.1f} / {summary['level']['last_sig_lag_max']}")

    print("\n[Effective PACF order — deltas]")
    print(f"  last practical lag mean/median/max : {summary['delta']['last_practical_lag_mean']:.2f} / {summary['delta']['last_practical_lag_median']:.1f} / {summary['delta']['last_practical_lag_max']}")
    print(f"  last significant lag mean/median/max: {summary['delta']['last_sig_lag_mean']:.2f} / {summary['delta']['last_sig_lag_median']:.1f} / {summary['delta']['last_sig_lag_max']}")

    print("\n[Cross-lag correlation offdiag — deltas]")
    for lag, st in summary["cross_lag_delta"].items():
        print(f"  lag {lag:>2}: mean|off|={st['mean_abs_offdiag']:.4f} max|off|={st['max_abs_offdiag']:.4f} fro={st['fro_offdiag']:.4f}")

    with open(out_dir / "pacf_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out_dir / "pacf_arrays.npz",
        pacf_level=pacf_level,
        pacf_delta=pacf_delta,
        **{f"cross_level_lag{lag}": mat for lag, mat in cross_level.items()},
        **{f"cross_delta_lag{lag}": mat for lag, mat in cross_delta.items()},
    )

    if not args.no_plots:
        save_plots(out_dir, pacf_level, pacf_delta, cross_level, cross_delta, args.max_lag)
    print("=" * 88)
    print(f"Saved PACF summary/arrays/figures to: {out_dir}")


if __name__ == "__main__":
    main()

"""
raw_lob_pacf_analysis.py — PACF diagnostics on raw/interpretable LOBench features.

Purpose
-------
Compare the temporal dependence of raw LOB features with the PACF already
observed in A1 latent tokens. This helps distinguish:
  (i) genuinely short-memory LOB innovations;
  (ii) an encoder/tokenizer that artificially flattened temporal dependence.

Expected dataset
----------------
NPZ produced by build_encoder_dataset_lobench.py, with keys:
  book:      (N, 2, L, 2), side 0=bid, side 1=ask, feature [price_z, volume_z]
  mid_z:     (N,)
  stock_ids: (N,)
  day_ids:   (N,)

The script computes interpretable raw features, including level features and
innovation features, then estimates PACF up to max_lag within contiguous
(stock, day) groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------- numerical helpers -----------------------------

def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / np.maximum(np.abs(b), eps)


def durbin_levinson_pacf(acf: np.ndarray, max_lag: int) -> np.ndarray:
    """PACF from autocorrelation using Durbin-Levinson recursion.

    acf[0] should be 1. Returns PACF lags 0..max_lag with pacf[0]=1.
    """
    pacf = np.zeros(max_lag + 1, dtype=np.float64)
    pacf[0] = 1.0
    if max_lag == 0:
        return pacf

    phi = np.zeros((max_lag + 1, max_lag + 1), dtype=np.float64)
    sigma = np.zeros(max_lag + 1, dtype=np.float64)
    sigma[0] = max(float(acf[0]), 1e-12)

    for k in range(1, max_lag + 1):
        if k == 1:
            num = acf[1]
        else:
            num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
        den = max(sigma[k - 1], 1e-12)
        phi[k, k] = np.clip(num / den, -0.999999, 0.999999)
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        sigma[k] = sigma[k - 1] * max(1.0 - phi[k, k] ** 2, 1e-12)
        pacf[k] = phi[k, k]
    return pacf


def aggregate_acf(groups: List[np.ndarray], max_lag: int, center: str = "group") -> np.ndarray:
    """Aggregate autocorrelation across disjoint contiguous groups.

    groups: list of 1D arrays.
    center: 'group' or 'global'. Group-centering avoids artificial day/stock level shifts.
    """
    clean = []
    if center == "global":
        all_x = np.concatenate([g[np.isfinite(g)] for g in groups if len(g) > max_lag])
        mu = float(all_x.mean()) if len(all_x) else 0.0
    else:
        mu = 0.0

    for g in groups:
        x = np.asarray(g, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) <= max_lag + 1:
            continue
        if center == "group":
            x = x - x.mean()
        else:
            x = x - mu
        clean.append(x)

    gamma = np.zeros(max_lag + 1, dtype=np.float64)
    counts = np.zeros(max_lag + 1, dtype=np.float64)
    for x in clean:
        for lag in range(max_lag + 1):
            if lag == 0:
                a = x
                b = x
            else:
                a = x[lag:]
                b = x[:-lag]
            if len(a) == 0:
                continue
            gamma[lag] += float(np.dot(a, b))
            counts[lag] += len(a)
    gamma = gamma / np.maximum(counts, 1.0)
    if gamma[0] <= 1e-12:
        acf = np.zeros(max_lag + 1, dtype=np.float64)
        acf[0] = 1.0
        return acf
    return gamma / gamma[0]


def effective_lags(pacf: np.ndarray, n_eff: int, practical_thr: float = 0.05) -> Dict[str, float]:
    vals = np.abs(pacf[1:])
    lags = np.arange(1, len(pacf))
    sig_thr = 2.0 / np.sqrt(max(n_eff, 1))

    def last_above(thr: float) -> int:
        idx = np.where(vals > thr)[0]
        return int(lags[idx[-1]]) if len(idx) else 0

    return {
        "last_practical_lag": last_above(practical_thr),
        "last_significant_lag": last_above(sig_thr),
        "significance_threshold": float(sig_thr),
        "practical_threshold": float(practical_thr),
        "max_abs_pacf": float(vals.max()) if len(vals) else 0.0,
        "sum_abs_pacf": float(vals.sum()) if len(vals) else 0.0,
    }


def corr_lag_matrix(groups_X: List[np.ndarray], lag: int) -> np.ndarray:
    """Corr between X_t features and X_{t-lag} features within groups.

    Each group array shape (T,F). Returns (F,F) correlation matrix.
    """
    xs = []
    ys = []
    for X in groups_X:
        if len(X) <= lag + 1:
            continue
        X = np.asarray(X, dtype=np.float64)
        X = X - np.nanmean(X, axis=0, keepdims=True)
        xs.append(X[lag:])
        ys.append(X[:-lag])
    if not xs:
        return np.eye(1)
    A = np.concatenate(xs, axis=0)
    B = np.concatenate(ys, axis=0)
    A = np.nan_to_num(A)
    B = np.nan_to_num(B)
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    Astd = A.std(axis=0, keepdims=True) + 1e-12
    Bstd = B.std(axis=0, keepdims=True) + 1e-12
    Az = A / Astd
    Bz = B / Bstd
    return (Az.T @ Bz) / max(len(Az) - 1, 1)


def offdiag_stats(mat: np.ndarray) -> Dict[str, float]:
    F = mat.shape[0]
    mask = ~np.eye(F, dtype=bool)
    vals = np.abs(mat[mask]) if F > 1 else np.array([0.0])
    return {
        "mean_abs_offdiag": float(vals.mean()),
        "max_abs_offdiag": float(vals.max()),
        "fro_offdiag": float(np.linalg.norm(mat[mask])) if F > 1 else 0.0,
    }


# ----------------------------- feature construction -----------------------------

def make_positive_volumes(book: np.ndarray, stock_ids: np.ndarray) -> np.ndarray:
    """Approximate nonnegative volumes from LOBench z-volume.

    The tokenizer dataset uses vol_z - per-stock min, then rescales by p99.
    For PACF/correlation, scale is irrelevant, but nonnegativity matters for
    imbalance/microprice, so we shift by per-stock minimum.
    """
    vols = book[:, :, :, 1].astype(np.float64)
    out = np.empty_like(vols)
    for s in np.unique(stock_ids):
        mask = stock_ids == s
        vm = np.nanmin(vols[mask])
        out[mask] = vols[mask] - vm
    return np.maximum(out, 0.0)


def compute_raw_features(book: np.ndarray, mid_z: np.ndarray, stock_ids: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Build interpretable raw LOB feature matrix (N,F)."""
    price = book[:, :, :, 0].astype(np.float64)
    vol = make_positive_volumes(book, stock_ids)

    bid_p = price[:, 0, :]
    ask_p = price[:, 1, :]
    bid_v = vol[:, 0, :]
    ask_v = vol[:, 1, :]

    bid1 = bid_p[:, 0]
    ask1 = ask_p[:, 0]
    bv1 = bid_v[:, 0]
    av1 = ask_v[:, 0]

    spread = ask1 - bid1
    top_imb = safe_div(bv1 - av1, bv1 + av1)

    k5 = min(5, bid_v.shape[1])
    bid_depth_5 = bid_v[:, :k5].sum(axis=1)
    ask_depth_5 = ask_v[:, :k5].sum(axis=1)
    imb5 = safe_div(bid_depth_5 - ask_depth_5, bid_depth_5 + ask_depth_5)
    total_depth_5 = np.log1p(bid_depth_5 + ask_depth_5)

    bid_depth_all = bid_v.sum(axis=1)
    ask_depth_all = ask_v.sum(axis=1)
    imball = safe_div(bid_depth_all - ask_depth_all, bid_depth_all + ask_depth_all)
    total_depth_all = np.log1p(bid_depth_all + ask_depth_all)

    micro = safe_div(ask1 * bv1 + bid1 * av1, bv1 + av1)
    micro_rel = micro - mid_z.astype(np.float64)

    # Relative book geometry in z-units; robust to stock price level.
    best_bid_rel = bid1 - mid_z
    best_ask_rel = ask1 - mid_z
    depth_width = (ask_p[:, -1] - bid_p[:, -1])

    X = np.column_stack([
        mid_z.astype(np.float64),
        spread,
        top_imb,
        imb5,
        imball,
        total_depth_5,
        total_depth_all,
        micro_rel,
        best_bid_rel,
        best_ask_rel,
        depth_width,
    ]).astype(np.float64)
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
    return X, names


def build_groups(X: np.ndarray, stock_ids: np.ndarray, day_ids: np.ndarray, max_tokens: int) -> List[np.ndarray]:
    """Return contiguous stock-day groups, preserving original order."""
    keys = stock_ids.astype(np.int64) * 100000 + day_ids.astype(np.int64)
    groups = []
    used = 0
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        if len(idx) < 10:
            continue
        # Ensure contiguous order from original processed CSV.
        g = X[idx]
        groups.append(g)
        used += len(g)
        if max_tokens and max_tokens > 0 and used >= max_tokens:
            break
    return groups


def to_delta_groups(groups: List[np.ndarray], feature_names: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    dg = []
    for g in groups:
        if len(g) > 2:
            dg.append(np.diff(g, axis=0))
    return dg, ["d_" + n for n in feature_names]


# ----------------------------- main analysis -----------------------------

def analyze_feature_groups(groups: List[np.ndarray], feature_names: List[str], max_lag: int, practical_thr: float) -> Dict:
    F = len(feature_names)
    n_eff = int(sum(max(0, len(g) - max_lag) for g in groups))
    pacf_arr = np.zeros((F, max_lag + 1), dtype=np.float64)
    acf_arr = np.zeros((F, max_lag + 1), dtype=np.float64)
    rows = []
    for i, name in enumerate(feature_names):
        series_groups = [g[:, i] for g in groups if len(g) > max_lag + 1]
        acf = aggregate_acf(series_groups, max_lag=max_lag, center="group")
        pacf = durbin_levinson_pacf(acf, max_lag=max_lag)
        acf_arr[i] = acf
        pacf_arr[i] = pacf
        row = {"feature": name, **effective_lags(pacf, n_eff=n_eff, practical_thr=practical_thr)}
        rows.append(row)
    practical = np.array([r["last_practical_lag"] for r in rows], dtype=float)
    significant = np.array([r["last_significant_lag"] for r in rows], dtype=float)
    return {
        "n_effective": n_eff,
        "features": rows,
        "summary": {
            "last_practical_lag_mean": float(practical.mean()),
            "last_practical_lag_median": float(np.median(practical)),
            "last_practical_lag_max": int(practical.max()),
            "last_significant_lag_mean": float(significant.mean()),
            "last_significant_lag_median": float(np.median(significant)),
            "last_significant_lag_max": int(significant.max()),
        },
        "pacf": pacf_arr,
        "acf": acf_arr,
    }


def save_plots(out_dir: Path, levels: Dict, deltas: Dict, feature_names: List[str], delta_names: List[str], max_lag: int, cross: Dict):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    lags = np.arange(max_lag + 1)

    def plot_pacf_grid(arr: np.ndarray, names: List[str], title: str, filename: str):
        n = len(names)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows), sharex=True)
        axes = np.ravel(axes)
        for i, ax in enumerate(axes):
            if i >= n:
                ax.axis("off")
                continue
            ax.axhline(0.0, linewidth=0.8)
            ax.axhline(0.05, linestyle="--", linewidth=0.8)
            ax.axhline(-0.05, linestyle="--", linewidth=0.8)
            ax.plot(lags[1:], arr[i, 1:], marker="o", markersize=2, linewidth=1)
            ax.set_title(names[i])
            ax.set_ylim(-0.5, 0.5)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    plot_pacf_grid(levels["pacf"], feature_names, "Raw LOB feature PACF — levels", "raw_levels_pacf_grid.png")
    plot_pacf_grid(deltas["pacf"], delta_names, "Raw LOB feature PACF — deltas", "raw_deltas_pacf_grid.png")

    for key, names, filename in [
        ("levels", feature_names, "cross_lag_levels.png"),
        ("deltas", delta_names, "cross_lag_deltas.png"),
    ]:
        mats = cross[key]
        fig, axes = plt.subplots(1, len(mats), figsize=(4.5 * len(mats), 4))
        if len(mats) == 1:
            axes = [axes]
        for ax, (lag, mat) in zip(axes, mats.items()):
            im = ax.imshow(mat, vmin=-0.5, vmax=0.5, cmap="coolwarm")
            ax.set_title(f"lag {lag}")
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(range(len(names)), fontsize=7)
            ax.set_yticklabels(range(len(names)), fontsize=7)
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        fig.suptitle(filename.replace("_", " "))
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/lobench_processed.npz")
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=2_000_000)
    p.add_argument("--practical_thr", type=float, default=0.05)
    p.add_argument("--out_dir", default="validation/world_model/raw_lob_pacf")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    data = np.load(args.dataset)
    required = ["book", "mid_z", "stock_ids", "day_ids"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Missing keys {missing}. Available keys: {data.files}")

    book = data["book"]
    mid_z = data["mid_z"]
    stock_ids = data["stock_ids"]
    day_ids = data["day_ids"]

    X, names = compute_raw_features(book, mid_z, stock_ids)
    groups = build_groups(X, stock_ids, day_ids, max_tokens=args.max_tokens)
    delta_groups, delta_names = to_delta_groups(groups, names)

    levels = analyze_feature_groups(groups, names, args.max_lag, args.practical_thr)
    deltas = analyze_feature_groups(delta_groups, delta_names, args.max_lag, args.practical_thr)

    cross_lags = [1, 2, 5, 10]
    cross = {"levels": {}, "deltas": {}, "level_stats": {}, "delta_stats": {}}
    for lag in cross_lags:
        mat_l = corr_lag_matrix(groups, lag=lag)
        mat_d = corr_lag_matrix(delta_groups, lag=lag)
        cross["levels"][lag] = mat_l
        cross["deltas"][lag] = mat_d
        cross["level_stats"][lag] = offdiag_stats(mat_l)
        cross["delta_stats"][lag] = offdiag_stats(mat_d)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": args.dataset,
        "book_shape": list(book.shape),
        "n_groups_used": len(groups),
        "n_tokens_used": int(sum(len(g) for g in groups)),
        "max_lag": args.max_lag,
        "practical_threshold": args.practical_thr,
        "feature_names": names,
        "delta_feature_names": delta_names,
        "levels": {k: v for k, v in levels.items() if k not in ["pacf", "acf"]},
        "deltas": {k: v for k, v in deltas.items() if k not in ["pacf", "acf"]},
        "cross_lag_stats": {
            "levels": {str(k): v for k, v in cross["level_stats"].items()},
            "deltas": {str(k): v for k, v in cross["delta_stats"].items()},
        },
    }
    with open(out_dir / "raw_pacf_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_dict = {
        "levels_pacf": levels["pacf"],
        "levels_acf": levels["acf"],
        "deltas_pacf": deltas["pacf"],
        "deltas_acf": deltas["acf"],
        "feature_names": np.array(names, dtype=object),
        "delta_feature_names": np.array(delta_names, dtype=object),
    }
    for lag, mat in cross["levels"].items():
        save_dict[f"cross_levels_lag{lag}"] = mat
    for lag, mat in cross["deltas"].items():
        save_dict[f"cross_deltas_lag{lag}"] = mat
    np.savez_compressed(out_dir / "raw_pacf_arrays.npz", **save_dict)

    if not args.no_plots:
        save_plots(out_dir, levels, deltas, names, delta_names, args.max_lag, cross)

    print("=" * 88)
    print("RAW LOB PACF / TEMPORAL DEPENDENCE ANALYSIS")
    print("=" * 88)
    print(f"dataset      : {args.dataset}")
    print(f"book shape   : {book.shape}")
    print(f"groups used  : {len(groups)}")
    print(f"tokens used  : {sum(len(g) for g in groups):,}")
    print(f"max_lag      : {args.max_lag}")

    print("\n[Effective PACF order — raw feature levels]")
    s = levels["summary"]
    print(f"  last practical lag mean/median/max : {s['last_practical_lag_mean']:.2f} / {s['last_practical_lag_median']:.1f} / {s['last_practical_lag_max']}")
    print(f"  last significant lag mean/median/max: {s['last_significant_lag_mean']:.2f} / {s['last_significant_lag_median']:.1f} / {s['last_significant_lag_max']}")

    print("\n[Effective PACF order — raw feature deltas]")
    s = deltas["summary"]
    print(f"  last practical lag mean/median/max : {s['last_practical_lag_mean']:.2f} / {s['last_practical_lag_median']:.1f} / {s['last_practical_lag_max']}")
    print(f"  last significant lag mean/median/max: {s['last_significant_lag_mean']:.2f} / {s['last_significant_lag_median']:.1f} / {s['last_significant_lag_max']}")

    print("\n[Per-feature raw delta practical lags]")
    for row in deltas["features"]:
        print(f"  {row['feature']:<24s} practical={row['last_practical_lag']:>2d} significant={row['last_significant_lag']:>2d} max|pacf|={row['max_abs_pacf']:.4f}")

    print("\n[Cross-lag correlation offdiag — raw feature deltas]")
    for lag in cross_lags:
        st = cross["delta_stats"][lag]
        print(f"  lag {lag:2d}: mean|off|={st['mean_abs_offdiag']:.4f} max|off|={st['max_abs_offdiag']:.4f} fro={st['fro_offdiag']:.4f}")
    print("=" * 88)
    print(f"Saved raw PACF summary/arrays/figures to: {out_dir}")


if __name__ == "__main__":
    main()

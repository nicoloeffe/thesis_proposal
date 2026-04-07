"""
eval_encoder.py — Valutazione oggettiva dell'encoder LOB.

Tre sezioni:
  1. RICOSTRUZIONE  — MSE, MAE, wMSE su volumi, decomposte per livello
  2. DOWNSTREAM     — probe lineari su target t+1:
                       reward, |shock_mid| (volatilità), OFI
                       + classificazione regime
  3. GEOMETRIA      — PCA, varianza spiegata, correlazione PC vs feature

Output: eval_encoder.png + stampa tabelle a terminale.

Uso:
  python scripts/eval_encoder.py
  python scripts/eval_encoder.py --ckpt checkpoints/encoder_best.pt \\
                                  --dataset data/dataset.npz \\
                                  --n_samples 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))

from models.encoder import LOBEncoder, EncoderConfig
from training.train_encoder import LOBDataset
from simulate import REGIMES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_encoder(ckpt_path: str, device: torch.device):
    from models.encoder import LOBAutoEncoder
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = EncoderConfig()
    # Load config from checkpoint (handles d_latent changes)
    if "cfg" in ckpt:
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    ae   = LOBAutoEncoder(cfg).to(device)
    ae.encoder.load_state_dict(ckpt["encoder"])
    if "decoder" in ckpt:
        ae.decoder.load_state_dict(ckpt["decoder"])
    ae.eval()
    print(f"Encoder loaded  : {ckpt_path}")
    print(f"  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")
    print(f"  d_latent={cfg.d_latent}")
    return ae.encoder, ckpt["stats"], ae, cfg


@torch.no_grad()
def build_encoded_dataset(
    encoder: LOBEncoder,
    dataset_path: str,
    stats: dict,
    device: torch.device,
    n_samples: int = 50_000,
    batch_size: int = 1024,
    seed: int = 42,
    autoencoder=None,   # optional: pass LOBAutoEncoder to get vol_pred
) -> dict:
    """
    Encode a random subset. Returns a dict with:
      Z          : (N, d_latent)      latent vectors at t
      Z_next     : (N, d_latent)      latent vectors at t+1
      vol_pred   : (N, 2, L)          reconstructed volumes
      vol_true   : (N, 2, L)          ground-truth volumes
      scalars    : (N, 4)             normalised [mid, spread, imb, inv]
      regimes    : (N,)               0/1/2
      rewards    : (N,)               r_t
      raw_obs    : (N, obs_dim)       raw (unnormalised) observations
      raw_next   : (N, obs_dim)       raw (unnormalised) next observations
    """
    raw_data  = np.load(dataset_path)
    obs_all   = raw_data["observations"]        # (N_total, 44)
    next_all  = raw_data["next_observations"]   # (N_total, 44)
    rew_all   = raw_data["rewards"]             # (N_total,)
    reg_all   = raw_data["regimes"]             # (N_total,)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rew_all), size=min(n_samples, len(rew_all)), replace=False))

    ds      = LOBDataset(dataset_path, stats=stats)
    ds_next = LOBDataset(dataset_path, stats=stats)
    # reuse the same normalisation stats for next_obs
    # we build next manually using the same preprocessing
    L    = ds.L
    tick = ds.tick_size
    s    = stats

    def preprocess(obs_array: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        book_flat = obs_array[:, :2*L*2].reshape(-1, 2, L, 2).astype(np.float32)
        sc_raw    = obs_array[:, 2*L*2:].astype(np.float32)
        mids      = sc_raw[:, 0]
        book_norm = book_flat.copy()
        for side in range(2):
            book_norm[:, side, :, 0] = (book_flat[:, side, :, 0] - mids[:, None]) / tick
        book_norm[:, :, :, 1] /= s["vol_scale"]
        sc_norm = np.stack([
            (sc_raw[:, 0] - s["mid_mean"]) / s["mid_std"],
            sc_raw[:, 1] / tick,
            sc_raw[:, 2],
            sc_raw[:, 3] / s["inv_scale"],
        ], axis=1)
        return torch.from_numpy(book_norm), torch.from_numpy(sc_norm)

    obs_sub  = obs_all[idx]
    next_sub = next_all[idx]

    books,    scalars    = preprocess(obs_sub)
    books_n,  scalars_n  = preprocess(next_sub)

    Z_list, Zn_list, vp_list, vt_list = [], [], [], []

    for i in range(0, len(idx), batch_size):
        b  = books[i:i+batch_size].to(device)
        sc = scalars[i:i+batch_size].to(device)
        bn = books_n[i:i+batch_size].to(device)
        sn = scalars_n[i:i+batch_size].to(device)

        z  = encoder(b, sc)
        zn = encoder(bn, sn)

        Z_list.append(z.cpu().numpy())
        Zn_list.append(zn.cpu().numpy())
        vt_list.append(b[:, :, :, 1].cpu().numpy())   # true volumes (normalised)

        if autoencoder is not None:
            with torch.no_grad():
                book_pred = autoencoder.decoder(z)   # (B, 2, L, 2)
                vp_list.append(book_pred.cpu().numpy())

    vol_pred = np.concatenate(vp_list, axis=0) if vp_list else None

    return {
        "Z":        np.concatenate(Z_list,  axis=0),
        "Z_next":   np.concatenate(Zn_list, axis=0),
        "vol_true": np.concatenate(vt_list, axis=0),
        "vol_pred": vol_pred,
        "scalars":  scalars.numpy()[: len(idx)],
        "regimes":  reg_all[idx],
        "rewards":  rew_all[idx],
        "raw_obs":  obs_sub,
        "raw_next": next_sub,
    }


# ---------------------------------------------------------------------------
# Section 1 — Reconstruction metrics
# ---------------------------------------------------------------------------

def reconstruction_metrics(data: dict, L: int, vol_scale: float) -> dict:
    """
    MSE, MAE, wMSE on normalised volumes, decomposed by level and regime.
    """
    vt = data["vol_true"]   # (N, 2, L)
    bp = data["vol_pred"]   # (N, 2, L, 2) or None
    regimes = data["regimes"]

    if bp is None:
        zeros = np.zeros(L)
        return {"wMSE": 0, "MSE": 0, "MAE": 0, "top_MSE": 0, "deep_MSE": 0,
                "per_level_mse": zeros, "per_level_mae": zeros,
                "per_regime": {}}

    vp = bp[:, :, :, 1]   # (N, 2, L)

    w = np.ones(L)
    w[0] = 4.0
    if L > 1: w[1] = 2.0
    w = w / w.sum()

    per_level_mse = ((vp - vt) ** 2).mean(axis=(0, 1))
    per_level_mae = np.abs(vp - vt).mean(axis=(0, 1))

    wmse      = (per_level_mse * w).sum()
    total_mse = per_level_mse.mean()
    total_mae = per_level_mae.mean()
    top_mse   = per_level_mse[:2].mean()
    deep_mse  = per_level_mse[2:].mean() if L > 2 else 0.0

    # Per-regime breakdown
    per_regime = {}
    for r_idx, rname in enumerate(["low_vol", "mid_vol", "high_vol"]):
        mask = (regimes == r_idx)
        if mask.sum() == 0:
            continue
        vt_r = vt[mask]
        vp_r = vp[mask]
        mse_abs = ((vp_r - vt_r) ** 2).mean()
        # Relative MSE: normalise by mean true volume² (scale-invariant)
        vol_mean_sq = (vt_r ** 2).mean() + 1e-10
        mse_rel = mse_abs / vol_mean_sq
        per_level_mse_r = ((vp_r - vt_r) ** 2).mean(axis=(0, 1))
        per_regime[rname] = {
            "mse_abs": mse_abs,
            "mse_rel": mse_rel,
            "per_level_mse": per_level_mse_r,
            "n": int(mask.sum()),
        }

    return {
        "wMSE":          wmse,
        "MSE":           total_mse,
        "MAE":           total_mae,
        "top_MSE":       top_mse,
        "deep_MSE":      deep_mse,
        "per_level_mse": per_level_mse,
        "per_level_mae": per_level_mae,
        "per_regime":    per_regime,
    }


# ---------------------------------------------------------------------------
# Section 2 — Downstream probes
# ---------------------------------------------------------------------------

def _split(X, y, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n   = int(len(X) * train_frac)
    return X[idx[:n]], X[idx[n:]], y[idx[:n]], y[idx[n:]]


def probe_regression(Z: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Ridge regression from z to scalar target."""
    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)
    X_tr, X_te, y_tr, y_te = _split(Z_s, y)
    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    # Baseline: predict mean
    baseline_mse = mean_squared_error(y_te, np.full_like(y_te, y_tr.mean()))
    return {"name": name, "MSE": mse, "MAE": mae, "R2": r2,
            "baseline_MSE": baseline_mse}


def probe_classification(Z: np.ndarray, regimes: np.ndarray) -> dict:
    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)
    X_tr, X_te, y_tr, y_te = _split(Z_s, regimes)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc    = clf.score(X_te, y_te)
    report = classification_report(y_te, y_pred,
                                   target_names=[r["name"] for r in REGIMES])
    baseline = max(np.bincount(y_te)) / len(y_te)
    return {
        "accuracy": acc, "baseline": baseline, "report": report,
        "y_true": y_te, "y_pred": y_pred,
    }


def downstream_probes(data: dict) -> dict:
    Z      = data["Z"]
    Z_next = data["Z_next"]
    raw_next = data["raw_next"]
    L        = 10
    tick     = 0.01

    # --- Reward t+1 ---
    rewards_next = data["rewards"]   # reward_t (aligned with obs_t)
    res_reward = probe_regression(Z, rewards_next, "reward_t")

    # --- |shock_mid| = volatility proxy ---
    mid_t      = data["raw_obs"][:, 2*L*2 + 0]
    mid_t1     = raw_next[:, 2*L*2 + 0]
    shock      = np.abs(mid_t1 - mid_t)
    res_vol = probe_regression(Z, shock, "|shock_mid|")

    # --- OFI: (bid_vol_best - ask_vol_best) / total_vol_best ---
    bid_vol = raw_next[:, 1]          # book[0, 0, 1] — bid best volume
    ask_vol = raw_next[:, 2*L + 1]    # book[1, 0, 1] — ask best volume
    denom   = bid_vol + ask_vol + 1e-8
    ofi     = (bid_vol - ask_vol) / denom
    res_ofi = probe_regression(Z, ofi, "OFI_t+1")

    # --- Spread t+1 ---
    spread_next = raw_next[:, 2*L*2 + 1]
    res_spread = probe_regression(Z, spread_next, "spread_t+1")

    # --- Regime classification ---
    res_regime = probe_classification(Z, data["regimes"])

    return {
        "reward":  res_reward,
        "vol":     res_vol,
        "ofi":     res_ofi,
        "spread":  res_spread,
        "regime":  res_regime,
    }


# ---------------------------------------------------------------------------
# Section 3 — Latent geometry
# ---------------------------------------------------------------------------

def latent_geometry(data: dict) -> dict:
    Z       = data["Z"]
    scalars = data["scalars"]   # [mid_norm, spread_norm, imb, inv_norm]
    regimes = data["regimes"]

    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)

    pca = PCA(n_components=10)
    Z_pca = pca.fit_transform(Z_s)

    # Correlations: top-5 PCs vs scalars (skip spread if constant)
    n_pc = min(5, Z_pca.shape[1])
    scalar_names = ["mid", "spread", "imbalance", "inventory"]
    corr = np.zeros((n_pc, 4))
    for i in range(n_pc):
        for j in range(4):
            std = scalars[:, j].std()
            if std < 1e-8:
                corr[i, j] = float("nan")
            else:
                corr[i, j] = np.corrcoef(Z_pca[:, i], scalars[:, j])[0, 1]

    return {
        "Z_pca":        Z_pca,
        "pca":          pca,
        "corr":         corr,
        "scalar_names": scalar_names,
        "regimes":      regimes,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_reconstruction(rec: dict) -> None:
    print("\n" + "="*60)
    print("RECONSTRUCTION (normalised volumes)")
    print("="*60)
    print(f"  wMSE={rec['wMSE']:.6f}  MSE={rec['MSE']:.6f}  MAE={rec['MAE']:.6f}")
    print(f"  top-of-book MSE={rec['top_MSE']:.6f}  deep MSE={rec['deep_MSE']:.6f}")
    if rec.get("per_regime"):
        print(f"\n  {'regime':<10s}  {'MSE_abs':>10s}  {'MSE_rel':>10s}  {'n':>7s}")
        print(f"  {'─'*42}")
        for rname, rv in rec["per_regime"].items():
            print(f"  {rname:<10s}  {rv['mse_abs']:>10.6f}  "
                  f"{rv['mse_rel']:>10.4f}  {rv['n']:>7d}")


def print_downstream(ds: dict) -> None:
    print("\n" + "="*60)
    print("DOWNSTREAM PROBES (linear from z)")
    print("="*60)
    print(f"\n  {'probe':<15s}  {'R²':>8s}  {'MSE':>10s}")
    print(f"  {'─'*36}")
    for key in ["reward", "vol", "ofi", "spread"]:
        r = ds[key]
        print(f"  {r['name']:<15s}  {r['R2']:>+8.4f}  {r['MSE']:>10.6f}")

    r = ds["regime"]
    print(f"\n  Regime classification: {r['accuracy']*100:.1f}% "
          f"(baseline: {r['baseline']*100:.1f}%)")
    print(f"{r['report']}")


def print_geometry(geo: dict) -> None:
    pca = geo["pca"]
    corr = geo["corr"]
    names = geo["scalar_names"]
    print("\n" + "="*60)
    print("LATENT GEOMETRY")
    print("="*60)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_pc = len(pca.explained_variance_ratio_)
    print(f"\n  PCA: {cumvar[min(7, n_pc-1)]:.0f}% in 8 PCs, "
          f"{cumvar[min(4, n_pc-1)]:.0f}% in 5 PCs")
    print(f"\n  {'PC':<5s}  {'var%':>6s}  {'cum%':>6s}")
    for i in range(min(n_pc, 10)):
        print(f"  PC{i+1:<2d}  {pca.explained_variance_ratio_[i]*100:>5.1f}%  "
              f"{cumvar[i]:>5.1f}%")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(rec: dict, ds: dict, geo: dict, data: dict, out_path: str) -> None:
    """
    Comprehensive evaluation: 3×3 layout.
      Row 0 (Reconstruction): Vol+MSE per level | Recon examples | MSE per regime
      Row 1 (Latent space):   PCA explained var | t-SNE by regime | Per-dim std
      Row 2 (Downstream):     Probe R² bars    | Confusion matrix | Temporal Δz
    """

    L    = len(rec["per_level_mse"])
    pca  = geo["pca"]
    corr = geo["corr"]
    Z    = data["Z"]
    regimes = data["regimes"]
    d_latent = Z.shape[1]
    regime_names = [r["name"] for r in REGIMES]
    regime_colors = ["#2ecc71", "#3498db", "#e74c3c"]

    GRAY  = "#555555"
    BLUE  = "#2c7bb6"
    GREEN = "#1a9641"
    RED   = "#d7191c"

    fig = plt.figure(figsize=(18, 15))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ===================================================================
    # Row 0 — Reconstruction quality
    # ===================================================================

    # --- 0a. Volume distribution & MSE per level ---
    ax = fig.add_subplot(gs[0, 0])
    vt = data["vol_true"]
    vol_per_level = vt.mean(axis=1)
    vol_mean = vol_per_level.mean(axis=0)
    vol_std  = vol_per_level.std(axis=0)
    ax2 = ax.twinx()
    ax.bar(range(L), vol_mean, color=[RED if i <= 1 else BLUE for i in range(L)],
           alpha=0.7, label="Mean vol")
    ax.errorbar(range(L), vol_mean, yerr=vol_std, fmt="none",
                ecolor=GRAY, capsize=3, lw=0.8)
    ax2.plot(range(L), rec["per_level_mse"], "k--o", markersize=4, lw=1.2, label="MSE")
    ax.set_title("Volume & reconstruction MSE\nper book level", fontsize=10)
    ax.set_xlabel("Level (0=best)", fontsize=8)
    ax.set_ylabel("Mean volume (norm.)", fontsize=8)
    ax2.set_ylabel("Recon MSE", fontsize=8)
    ax.set_xticks(range(L))
    ax.tick_params(labelsize=7)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    # --- 0b. Reconstruction examples (1 per regime) ---
    ax = fig.add_subplot(gs[0, 1])
    if data["vol_pred"] is not None:
        for r_idx, (rname, rcol) in enumerate(zip(regime_names, regime_colors)):
            mask = (regimes == r_idx)
            if mask.sum() == 0:
                continue
            # Pick a representative sample (median MSE for this regime)
            r_true = vt[mask].mean(axis=1)          # (N_r, L) avg bid+ask
            r_pred = data["vol_pred"][mask][:, :, :, 1].mean(axis=1)  # (N_r, L) avg bid+ask
            # Actually let's just show the mean profile
            true_mean = r_true.mean(axis=0)
            pred_mean = r_pred.mean(axis=0)
            offset = r_idx * 0.15
            ax.plot(range(L), true_mean, "o-", color=rcol, markersize=3,
                    lw=1.5, label=f"{rname} true")
            ax.plot(range(L), pred_mean, "x--", color=rcol, markersize=4,
                    lw=1, alpha=0.7)
        ax.set_title("Reconstruction: true (●) vs pred (×)\nmean profile per regime", fontsize=10)
        ax.set_xlabel("Level", fontsize=8)
        ax.set_ylabel("Mean volume (norm.)", fontsize=8)
        ax.legend(fontsize=7, ncol=1)
        ax.set_xticks(range(L))
        ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "No decoder output", transform=ax.transAxes, ha="center")

    # --- 0c. MSE per regime ---
    ax = fig.add_subplot(gs[0, 2])
    if data["vol_pred"] is not None:
        regime_mse = []
        for r_idx in range(len(REGIMES)):
            mask = (regimes == r_idx)
            if mask.sum() == 0:
                regime_mse.append(0)
                continue
            r_true = vt[mask]
            r_pred = data["vol_pred"][mask][:, :, :, 1]
            mse = ((r_true - r_pred) ** 2).mean()
            regime_mse.append(mse)
        bars = ax.bar(regime_names, regime_mse, color=regime_colors, alpha=0.8)
        ax.set_title("Reconstruction MSE per regime", fontsize=10)
        ax.set_ylabel("MSE (normalised volumes)", fontsize=8)
        ax.tick_params(labelsize=8)
        for bar, v in zip(bars, regime_mse):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.0001,
                    f"{v:.4f}", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No decoder output", transform=ax.transAxes, ha="center")

    # ===================================================================
    # Row 1 — Latent space structure
    # ===================================================================

    # --- 1a. PCA explained variance ---
    ax = fig.add_subplot(gs[1, 0])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_show = min(d_latent, 10)
    ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show] * 100,
           color=BLUE, alpha=0.8, label="Per-PC")
    ax.plot(range(1, n_show + 1), cumvar[:n_show], "o-", color=RED, markersize=4,
            lw=1.5, label="Cumulative")
    ax.axhline(80, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.set_title(f"PCA — explained variance\n({d_latent}-dim z)", fontsize=10)
    ax.set_xlabel("PC", fontsize=8)
    ax.set_ylabel("Variance (%)", fontsize=8)
    ax.set_xticks(range(1, n_show + 1))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    # --- 1b. t-SNE colored by regime ---
    ax = fig.add_subplot(gs[1, 1])
    n_tsne = min(8000, len(Z))
    rng = np.random.default_rng(42)
    tsne_idx = rng.choice(len(Z), size=n_tsne, replace=False)
    Z_sub = Z[tsne_idx]
    reg_sub = regimes[tsne_idx]
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    Z_2d = tsne.fit_transform(Z_sub)
    for r_idx, (rname, rcol) in enumerate(zip(regime_names, regime_colors)):
        mask = (reg_sub == r_idx)
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=rcol, s=3, alpha=0.4,
                   label=rname, rasterized=True)
    ax.set_title("t-SNE of z colored by regime", fontsize=10)
    ax.legend(fontsize=8, markerscale=4)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- 1c. Per-dimension std (VICReg check) ---
    ax = fig.add_subplot(gs[1, 2])
    per_dim_std = Z.std(axis=0)
    ax.bar(range(d_latent), per_dim_std, color=BLUE, alpha=0.8)
    ax.axhline(1.0, color=RED, ls="--", lw=1.5, alpha=0.7, label="target = 1.0")
    ax.set_title("Per-dim std(z)\n(VICReg variance check)", fontsize=10)
    ax.set_xlabel("dimension", fontsize=8)
    ax.set_ylabel("std", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(per_dim_std.max() * 1.3, 1.5))

    # ===================================================================
    # Row 2 — Downstream & dynamics
    # ===================================================================

    # --- 2a. Downstream probe R² ---
    ax = fig.add_subplot(gs[2, 0])
    probe_items = []
    for key, label, col in [
        ("ofi", "OFI", GREEN),
        ("spread", "spread_{t+1}", BLUE),
        ("vol", "|Δmid|", RED),
    ]:
        if key in ds:
            probe_items.append((label, ds[key]["R2"], col))
    if probe_items:
        names, vals, cols = zip(*probe_items)
        ax.barh(list(names), list(vals), color=list(cols), alpha=0.8)
        ax.set_xlabel("R² (linear probe from z)", fontsize=8)
        ax.set_title("Downstream probes\n(single snapshot)", fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.tick_params(labelsize=8)
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")

    # --- 2b. Confusion matrix ---
    ax = fig.add_subplot(gs[2, 1])
    if "regime" in ds and "y_pred" in ds["regime"]:
        y_true = ds["regime"]["y_true"]
        y_pred = ds["regime"]["y_pred"]
    else:
        # Recompute quickly
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        N = len(Z)
        split = int(0.8 * N)
        scaler = StandardScaler().fit(Z[:split])
        Zs_tr, Zs_te = scaler.transform(Z[:split]), scaler.transform(Z[split:])
        clf = LogisticRegression(max_iter=500, C=1.0).fit(Zs_tr, regimes[:split])
        y_true = regimes[split:]
        y_pred = clf.predict(Zs_te)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(regime_names)))
    ax.set_xticklabels(regime_names, fontsize=8)
    ax.set_yticks(range(len(regime_names)))
    ax.set_yticklabels(regime_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(f"Regime confusion matrix\n(acc={ds['regime']['accuracy']*100:.1f}%)", fontsize=10)
    for i in range(len(regime_names)):
        for j in range(len(regime_names)):
            v = cm_norm[i, j]
            col = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}\n({cm[i,j]})", ha="center", va="center",
                    fontsize=8, color=col)

    # --- 2c. Temporal smoothness: ‖z_{t+1} - z_t‖ per regime ---
    ax = fig.add_subplot(gs[2, 2])
    dz = np.linalg.norm(data["Z_next"] - data["Z"], axis=1)  # (N,)
    for r_idx, (rname, rcol) in enumerate(zip(regime_names, regime_colors)):
        mask = (regimes == r_idx)
        if mask.sum() == 0:
            continue
        dz_r = dz[mask]
        ax.hist(dz_r, bins=50, alpha=0.5, density=True, color=rcol, label=rname)
    ax.set_title("Temporal smoothness: ‖z_{t+1} - z_t‖\nper regime", fontsize=10)
    ax.set_xlabel("‖Δz‖", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)

    plt.suptitle("LOB Encoder — Evaluation Report", fontsize=14, y=1.01, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder, stats, ae, cfg = load_encoder(args.ckpt, device)

    print(f"\nEncoding {args.n_samples:,} campioni...")
    data = build_encoded_dataset(
        encoder, args.dataset, stats, device,
        n_samples=args.n_samples,
        autoencoder=ae,
    )
    print(f"  Z shape: {data['Z'].shape}")

    # Section 1
    rec = reconstruction_metrics(data, L=10, vol_scale=stats["vol_scale"])
    print_reconstruction(rec)

    # Section 2
    print("\nCalcolo downstream probes...")
    ds = downstream_probes(data)
    print_downstream(ds)

    # Section 3
    print("\nAnalisi geometria latente...")
    geo = latent_geometry(data)
    print_geometry(geo)

    # Plot
    plot_all(rec, ds, geo, data, out_path=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LOB Encoder")
    parser.add_argument("--ckpt",      type=str,  default="checkpoints/encoder_best.pt")
    parser.add_argument("--dataset",   type=str,  default="data/dataset.npz")
    parser.add_argument("--n_samples", type=int,  default=50_000)
    parser.add_argument("--out",       type=str,  default="eval_encoder.png")
    args = parser.parse_args()
    main(args)
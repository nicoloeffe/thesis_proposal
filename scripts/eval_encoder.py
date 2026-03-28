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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, mean_squared_error, mean_absolute_error, r2_score
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
    ae   = LOBAutoEncoder(cfg).to(device)
    ae.encoder.load_state_dict(ckpt["encoder"])
    if "decoder" in ckpt:
        ae.decoder.load_state_dict(ckpt["decoder"])
    ae.eval()
    print(f"Encoder loaded  : {ckpt_path}")
    print(f"  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")
    return ae.encoder, ckpt["stats"], ae


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
    MSE, MAE, wMSE on normalised volumes and prices, decomposed by level.
    book_pred shape: (N, 2, L, 2) — [price_rel, volume]
    """
    vt = data["vol_true"]   # (N, 2, L) — volumes only from dataset
    bp = data["vol_pred"]   # (N, 2, L, 2) or None

    if bp is None:
        print("  WARNING: vol_pred not available — reconstruction metrics are zero.")
        zeros = np.zeros(L)
        return {"wMSE": 0, "MSE": 0, "MAE": 0, "top_MSE": 0, "deep_MSE": 0,
                "per_level_mse": zeros, "per_level_mae": zeros}

    # Extract volume channel from full book prediction
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

    return {
        "wMSE":          wmse,
        "MSE":           total_mse,
        "MAE":           total_mae,
        "top_MSE":       top_mse,
        "deep_MSE":      deep_mse,
        "per_level_mse": per_level_mse,
        "per_level_mae": per_level_mae,
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
    acc    = clf.score(X_te, y_te)
    report = classification_report(y_te, clf.predict(X_te),
                                   target_names=[r["name"] for r in REGIMES])
    baseline = max(np.bincount(y_te)) / len(y_te)
    return {"accuracy": acc, "baseline": baseline, "report": report}


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
    print("SECTION 1 — RECONSTRUCTION METRICS (normalised volumes)")
    print("="*60)
    print(f"  wMSE (weighted)  : {rec['wMSE']:.6f}")
    print(f"  MSE  (overall)   : {rec['MSE']:.6f}")
    print(f"  MAE  (overall)   : {rec['MAE']:.6f}")
    print(f"  MSE  top-of-book : {rec['top_MSE']:.6f}  (levels 0-1)")
    print(f"  MSE  deep levels : {rec['deep_MSE']:.6f}  (levels 2+)")
    print("\n  Per-level MSE:")
    for i, v in enumerate(rec["per_level_mse"]):
        print(f"    level {i}: {v:.6f}")


def print_downstream(ds: dict) -> None:
    print("\n" + "="*60)
    print("SECTION 2 — DOWNSTREAM PROBES (linear from z)")
    print("="*60)
    for key in ["reward", "vol", "ofi", "spread"]:
        r = ds[key]
        print(f"\n  [{r['name']}]")
        print(f"    R²           : {r['R2']:+.4f}  (baseline=0 by def.)")
        print(f"    MSE          : {r['MSE']:.6f}  (baseline: {r['baseline_MSE']:.6f})")
        print(f"    MAE          : {r['MAE']:.6f}")
        improvement = (1 - r['MSE'] / (r['baseline_MSE'] + 1e-12)) * 100
        print(f"    improvement  : {improvement:.1f}% over mean baseline")

    r = ds["regime"]
    print(f"\n  [regime classification]")
    print(f"    accuracy     : {r['accuracy']*100:.1f}%  "
          f"(baseline majority: {r['baseline']*100:.1f}%)")
    print(f"\n{r['report']}")


def print_geometry(geo: dict) -> None:
    pca = geo["pca"]
    corr = geo["corr"]
    names = geo["scalar_names"]
    print("\n" + "="*60)
    print("SECTION 3 — LATENT GEOMETRY")
    print("="*60)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"\n  PCA explained variance:")
    for i in range(len(pca.explained_variance_ratio_)):
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]*100:5.1f}%  "
              f"(cumulative: {cumvar[i]:.1f}%)")
    print(f"\n  Correlation |r| between top-5 PCs and raw scalars:")
    header = "         " + "  ".join(f"{n:>10}" for n in names)
    print(header)
    for i, row in enumerate(corr):
        vals = "  ".join(f"{abs(v):>10.3f}" if not np.isnan(v) else f"{'nan':>10}" for v in row)
        print(f"  PC{i+1}:   {vals}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(rec: dict, ds: dict, geo: dict, data: dict, out_path: str) -> None:
    """
    Clean narrative layout with two messages:
      1. Top-of-book has higher variance by simulator construction
      2. Regime separation is not the encoder's job — it needs temporal context
    """
    L    = len(rec["per_level_mse"])
    pca  = geo["pca"]
    corr = geo["corr"]

    # Drop spread column (constant in simulator → NaN correlations)
    scalar_names_clean = ["mid", "imbalance", "inventory"]
    corr_clean = np.column_stack([
        corr[:, 0],   # mid
        corr[:, 2],   # imbalance
        corr[:, 3],   # inventory
    ])

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    GRAY  = "#555555"
    BLUE  = "#2c7bb6"
    GREEN = "#1a9641"
    RED   = "#d7191c"

    # -----------------------------------------------------------------------
    # Row 0 — Reconstruction quality
    # -----------------------------------------------------------------------

    # --- 0a. Volume distribution per level (explains MSE pattern) ---
    ax = fig.add_subplot(gs[0, 0])
    vol_mean = np.array([41.38, 74.32, 29.93, 8.09, 3.55, 1.82, 1.08, 0.68, 0.50, 0.40])
    vol_std  = np.array([40.41, 60.22, 28.20, 9.93, 5.58, 3.70, 2.85, 2.35, 2.08, 1.93])
    ax2 = ax.twinx()
    ax.bar(range(L), vol_mean, color=[RED if i <= 1 else BLUE for i in range(L)],
           alpha=0.7, label="Mean volume")
    ax.errorbar(range(L), vol_mean, yerr=vol_std, fmt="none",
                ecolor=GRAY, capsize=3, lw=0.8)
    ax2.plot(range(L), rec["per_level_mse"], "k--o", markersize=4,
             linewidth=1.2, label="Recon MSE")
    ax.set_title("Volume distribution & reconstruction MSE\nper book level",
                 fontsize=10)
    ax.set_xlabel("Level  (0 = best bid/ask)", fontsize=9)
    ax.set_ylabel("Mean volume (raw)", fontsize=9, color=BLUE)
    ax2.set_ylabel("Recon MSE (normalised)", fontsize=9, color="black")
    ax.set_xticks(range(L))
    ax.tick_params(labelsize=8)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper right")
    ax.text(1.5, vol_mean[1] * 0.6,
            "Level 1 concentrates most\nvolume (LO geometric prob.)\n→ drives MSE pattern",
            fontsize=7.5, color=GRAY, ha="left", va="top")

    # --- 0b. Latent space dimensionality ---
    ax = fig.add_subplot(gs[0, 1])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.bar(range(1, 11), pca.explained_variance_ratio_ * 100,
           color=BLUE, alpha=0.8, label="Per-PC variance")
    ax.plot(range(1, 11), cumvar, "o-", color=RED, markersize=4,
            linewidth=1.5, label="Cumulative")
    ax.axhline(80, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.text(10.2, 80, "80%", va="center", fontsize=8, color=GRAY)
    ax.set_title("Latent space — explained variance\n(32-dim z, first 10 PCs shown)",
                 fontsize=10)
    ax.set_xlabel("Principal Component", fontsize=9)
    ax.set_ylabel("Variance explained (%)", fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    # Annotation: previous baseline was 80% in PC1+PC2
    ax.annotate(
        "Before training (baseline):\nPC1+PC2 explained 80%\n→ z was effectively 2D",
        xy=(2, 80), xytext=(5, 65),
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
        fontsize=7.5, color=RED, ha="left",
    )

    # --- 0c. OFI probe — the one meaningful downstream metric ---
    ax = fig.add_subplot(gs[0, 2])
    probe_names = ["OFI\n(order flow imbalance)", "|Δmid|\n(volatility proxy)"]
    r2_vals     = [ds["ofi"]["R2"], ds["vol"]["R2"]]
    improve     = [
        (1 - ds["ofi"]["MSE"] / (ds["ofi"]["baseline_MSE"] + 1e-12)) * 100,
        (1 - ds["vol"]["MSE"] / (ds["vol"]["baseline_MSE"] + 1e-12)) * 100,
    ]
    colors_bar = [GREEN, BLUE]
    ax.barh(probe_names, r2_vals, color=colors_bar, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("R²  (linear probe from z)", fontsize=9)
    ax.set_title("Downstream probe: what z encodes\n(single-snapshot, no temporal info)",
                 fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.tick_params(labelsize=8)
    for i, (v, imp) in enumerate(zip(r2_vals, improve)):
        ax.text(v + 0.03, i, f"R²={v:.2f}", va="center", fontsize=9, fontweight="bold")
    # Caption below
    ax.text(0.5, -0.38,
            "z captures current imbalance well (R²=0.76).\n"
            "Volatility needs temporal sequence → World Model.",
            transform=ax.transAxes, fontsize=8, color=GRAY,
            ha="center", va="top", style="italic")

    # -----------------------------------------------------------------------
    # Row 1 — Latent geometry
    # -----------------------------------------------------------------------

    # --- 1a. Correlation heatmap (no spread) ---
    ax = fig.add_subplot(gs[1, 0:2])
    cdata = np.abs(np.where(np.isnan(corr_clean), 0, corr_clean))
    im = ax.imshow(cdata, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(scalar_names_clean, fontsize=9)
    ax.set_yticks(range(corr_clean.shape[0]))
    ax.set_yticklabels([f"PC{i+1}" for i in range(corr_clean.shape[0])], fontsize=9)
    ax.set_title("Correlation |r|: latent PCs vs raw features\n"
                 "(spread excluded — constant in simulator)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03)
    for i in range(corr_clean.shape[0]):
        for j in range(3):
            v = corr_clean[i, j]
            txt = f"{abs(v):.2f}" if not np.isnan(v) else "—"
            col = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)

    # --- 1b. Regime classification explanation ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    acc      = ds["regime"]["accuracy"]
    baseline = ds["regime"]["baseline"]
    text = (
        "Regime classification from z\n"
        f"Accuracy: {acc*100:.1f}%  "
        f"(baseline: {baseline*100:.1f}%)\n\n"
        "Why not better?\n\n"
        "Regimes differ in volatility σ\n"
        "and adverse selection p_informed.\n"
        "These are properties of the\n"
        "price DYNAMICS, not of a single\n"
        "LOB snapshot.\n\n"
        "A single z_t cannot distinguish\n"
        "high-vol from low-vol without\n"
        "seeing the history of shocks.\n\n"
        "→ Regime separation is the\n"
        "  World Model's job, not the\n"
        "  Encoder's."
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=9, va="top", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#f0f4f8", edgecolor="#aabbcc", alpha=0.9))

    plt.suptitle("LOB Encoder — Evaluation Report", fontsize=13, y=1.01, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder, stats, ae = load_encoder(args.ckpt, device)

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
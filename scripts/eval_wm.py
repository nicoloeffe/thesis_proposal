"""
eval_wm.py — Valutazione completa del World Model (Modulo B).

Il WM non è un forecaster: fornisce P(z_{t+1} | h_t) come GMM, che diventa
la distribuzione nominale P₀ nel problema DRO. Ciò che conta è:

  1. CALIBRAZIONE σ   — se σ ≈ |errore reale|, il raggio ε ha senso fisico
  2. COPERTURA         — centroidi separati → direzioni di perturbazione DRO significative
  3. REGIME-AWARENESS  — ambiguity set diversi per regime

Layout di valutazione (4 pannelli narrativi):

  Pannello A — "Il WM è calibrato?"
               σ/|err| per regime, binned calibration.
  Pannello B — "La GMM genera scenari diversi?"
               Separazione centroidi, K_eff, π per regime heatmap.
  Pannello C — "Il WM distingue i regimi?"
               NLL per regime, effetto contesto, entropia π.
  Pannello D — "CRPS sanity check"
               GMM vs gaussiana baseline. La GMM non deve essere peggio.

Le metriche downstream (V_robust curves, worst-case qualitativi, GMM vs Gaussian
DRO A/B) si calcolano nella Fase 5 (stress test), non qui.

Uso:
  python scripts/eval_wm.py
  python scripts/eval_wm.py --wm_ckpt checkpoints/wm_best.pt \
                             --dataset data/wm_dataset.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from models.world_model import LOBWorldModel, WorldModelConfig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_world_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = WorldModelConfig()
    for k, v in ckpt["cfg"].items():
        setattr(cfg, k, v)
    model = LOBWorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"World model loaded: {ckpt_path}")
    print(f"  epoch={ckpt['epoch']}  val_nll={ckpt['val_nll']:.4f}")
    print(f"  K={cfg.n_gmm}  d_latent={cfg.d_latent}  d_model={cfg.d_model}")
    return model, cfg


def load_dataset(path: str, n_samples: int | None = None, seed: int = 42):
    data = np.load(path)
    sequences = torch.from_numpy(data["sequences"])
    actions   = torch.from_numpy(data["actions"])
    rewards   = torch.from_numpy(data["rewards"])
    regimes   = torch.from_numpy(data["regimes"].astype(np.int64))

    if n_samples is not None and n_samples < len(sequences):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sequences), size=n_samples, replace=False)
        sequences = sequences[idx]
        actions   = actions[idx]
        rewards   = rewards[idx]
        regimes   = regimes[idx]

    M, Np1, D = sequences.shape
    print(f"Dataset: {M:,} sequences, N={Np1-1}, d_latent={D}")
    return sequences, actions, rewards, regimes


# ---------------------------------------------------------------------------
# Core inference — one-step predictions + raw outputs
# ---------------------------------------------------------------------------

@torch.no_grad()
def one_step_metrics(
    model: LOBWorldModel,
    sequences: torch.Tensor,
    actions: torch.Tensor,
    regimes: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    """
    1-step prediction: collect all GMM outputs for downstream analysis.
    """
    M, Np1, D = sequences.shape
    N = Np1 - 1

    all_nll = []
    all_pi  = []
    all_mu  = []
    all_log_sig = []
    all_errors  = []
    all_pos_nll = []
    regime_nll = {0: [], 1: [], 2: []}

    for i in range(0, M, batch_size):
        z_seq = sequences[i:i+batch_size].to(device)
        a_seq = actions[i:i+batch_size].to(device)
        reg   = regimes[i:i+batch_size].to(device)

        pi, mu, log_sig = model(z_seq, a_seq)
        z_next = z_seq[:, 1:, :]

        # Weighted mean prediction
        z_pred = (pi.unsqueeze(-1) * mu).sum(dim=2)
        err = z_pred - z_next

        # Total NLL
        nll = model.nll_loss(pi, mu, log_sig, z_next)

        # Per-position NLL
        B_cur, N_cur, K, D_cur = mu.shape
        z_exp = z_next.unsqueeze(2)
        sig = torch.exp(log_sig)
        log_norm = -0.5 * (
            D_cur * math.log(2 * math.pi)
            + 2 * log_sig.sum(dim=-1)
            + ((z_exp - mu) ** 2 / (sig ** 2 + 1e-8)).sum(dim=-1)
        )
        log_pi = torch.log(pi + 1e-8)
        log_p  = torch.logsumexp(log_pi + log_norm, dim=-1)
        pos_nll = -log_p

        all_nll.append(nll.item())
        all_pi.append(pi.cpu())
        all_mu.append(mu.cpu())
        all_log_sig.append(log_sig.cpu())
        all_errors.append(err.cpu())
        all_pos_nll.append(pos_nll.cpu())

        # --- FIX CALCOLO NLL PER REGIME ---
        # 1. Espandiamo reg se è 1D (per-episode) a 2D (per-step)
        if reg.dim() == 1:
            reg_exp = reg.unsqueeze(1).expand(-1, N_cur)
        else:
            reg_exp = reg

        # 2. Flatten (B, N) -> (B*N,)
        pi_flat  = pi.reshape(B_cur * N_cur, K)
        mu_flat  = mu.reshape(B_cur * N_cur, K, D_cur)
        ls_flat  = log_sig.reshape(B_cur * N_cur, K, D_cur)
        zn_flat  = z_next.reshape(B_cur * N_cur, D_cur)
        reg_flat = reg_exp.reshape(-1)

        for r in [0, 1, 2]:
            mask = (reg_flat == r)
            if mask.sum() > 0:
                # Aggiungiamo unsqueeze(1) per ripristinare la dimensione N=1
                # in modo che nll_loss riceva shape (n_true, 1, K, D)
                nll_r = model.nll_loss(
                    pi_flat[mask].unsqueeze(1),
                    mu_flat[mask].unsqueeze(1),
                    ls_flat[mask].unsqueeze(1),
                    zn_flat[mask].unsqueeze(1)
                ).item()
                regime_nll[r].append(nll_r)

    pi_all    = torch.cat(all_pi).numpy()
    mu_all    = torch.cat(all_mu).numpy()
    errors    = torch.cat(all_errors).numpy()
    log_sig_c = torch.cat(all_log_sig).numpy()
    pos_nll   = torch.cat(all_pos_nll).numpy()

    per_dim_mse = (errors ** 2).mean(axis=(0, 1))
    ent = -(pi_all * np.log(pi_all + 1e-8)).sum(axis=-1)

    return {
        "nll_mean":     float(np.mean(all_nll)),
        "mse_mean":     float(per_dim_mse.mean()),
        "per_dim_mse":  per_dim_mse,
        "entropy_pi":   ent,
        "pi_all":       pi_all,
        "mu_all":       mu_all,
        "errors":       errors,
        "log_sig_cat":  log_sig_c,
        "regime_nll":   {r: np.mean(v) for r, v in regime_nll.items() if v},
        "pos_nll_mean": pos_nll.mean(axis=0),
    }


# ---------------------------------------------------------------------------
# Pannello A — Calibrazione σ/|err| (globale + per regime)
# ---------------------------------------------------------------------------

def calibration_analysis(metrics: dict, regimes: torch.Tensor) -> dict:
    """
    σ vs |error|: globale, per dimensione, per regime, e binned.
    Se σ è calibrata, ε nel DRO ha la scala giusta.
    """
    errors  = metrics["errors"]       # (M, N, D)
    log_sig = metrics["log_sig_cat"]  # (M, N, K, D)
    pi      = metrics["pi_all"]       # (M, N, K)
    reg     = regimes.numpy()

    sig = np.exp(log_sig)
    # Weighted average σ across components
    sig_avg = (pi[..., np.newaxis] * sig).sum(axis=2)  # (M, N, D)

    abs_err = np.abs(errors)

    # --- Global per-dim ---
    abs_err_per_dim = abs_err.mean(axis=(0, 1))
    sig_per_dim     = sig_avg.mean(axis=(0, 1))
    calib_ratio_per_dim = sig_per_dim / (abs_err_per_dim + 1e-8)

    # --- Per-regime σ/|err| ---
    calib_per_regime = {}
    
    # Expand regimes for per-step masking if necessary
    if reg.ndim == 1:
        reg_exp = np.broadcast_to(reg[:, None], (errors.shape[0], errors.shape[1]))
    else:
        reg_exp = reg

    for r in [0, 1, 2]:
        mask = (reg_exp == r)
        if mask.sum() > 0:
            sig_r = sig_avg[mask].mean()
            err_r = abs_err[mask].mean()
            calib_per_regime[r] = float(sig_r / (err_r + 1e-8))

    # --- Binned calibration ---
    sig_flat = sig_avg.reshape(-1)
    err_flat = abs_err.reshape(-1)
    n_bins = 10
    percentiles = np.percentile(sig_flat, np.linspace(0, 100, n_bins + 1))
    bin_sig, bin_err = [], []
    for b in range(n_bins):
        lo, hi = percentiles[b], percentiles[b + 1] + 1e-10
        mask = (sig_flat >= lo) & (sig_flat < hi)
        if mask.sum() > 0:
            bin_sig.append(sig_flat[mask].mean())
            bin_err.append(err_flat[mask].mean())

    return {
        "abs_err_per_dim":   abs_err_per_dim,
        "sig_per_dim":       sig_per_dim,
        "calib_ratio":       calib_ratio_per_dim,
        "calib_ratio_mean":  float(calib_ratio_per_dim.mean()),
        "calib_per_regime":  calib_per_regime,
        "bin_sig":           np.array(bin_sig),
        "bin_err":           np.array(bin_err),
    }


# ---------------------------------------------------------------------------
# Pannello B — Copertura GMM: separazione centroidi, K_eff
# ---------------------------------------------------------------------------

def coverage_analysis(metrics: dict, regimes: torch.Tensor) -> dict:
    """
    - Separazione centroidi: distanza media tra coppie di μ_k, normalizzata per σ.
    - K_eff: numero effettivo di componenti (1/Σπ²), media su tutti i sample.
    - π per regime heatmap.
    """
    pi      = metrics["pi_all"]       # (M, N, K)
    mu      = metrics["mu_all"]       # (M, N, K, D)
    log_sig = metrics["log_sig_cat"]  # (M, N, K, D)
    reg     = regimes.numpy()

    M, N, K, D = mu.shape

    # --- Centroid separation (in units of σ) ---
    mu_avg  = mu.mean(axis=(0, 1))   # (K, D)
    sig_avg = np.exp(log_sig).mean(axis=(0, 1))  # (K, D)
    avg_sig = sig_avg.mean()  # scalar

    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            d = np.linalg.norm(mu_avg[i] - mu_avg[j])
            dists.append(d)
    mean_dist = float(np.mean(dists)) if dists else 0.0
    separation_sigma = mean_dist / (avg_sig + 1e-8)

    # --- K_eff = 1 / Σ π_k² ---
    pi_flat = pi.reshape(-1, K)
    k_eff_per_sample = 1.0 / (pi_flat ** 2).sum(axis=-1)
    k_eff_mean = float(k_eff_per_sample.mean())

    # --- π per regime ---
    pi_per_regime = {}
    
    if reg.ndim == 1:
        reg_exp = np.broadcast_to(reg[:, None], (M, N))
    else:
        reg_exp = reg
        
    for r in [0, 1, 2]:
        mask = (reg_exp == r)
        if mask.sum() > 0:
            pi_per_regime[r] = pi[mask].mean(axis=0)

    # --- Component usage (argmax) ---
    assignments = pi.reshape(-1, K).argmax(axis=-1)
    usage_counts = np.bincount(assignments, minlength=K)
    usage_frac   = usage_counts / usage_counts.sum()

    ent = metrics["entropy_pi"]

    return {
        "mu_avg":             mu_avg,
        "separation_l2":      mean_dist,
        "separation_sigma":   separation_sigma,
        "k_eff_mean":         k_eff_mean,
        "K":                  K,
        "pi_per_regime":      pi_per_regime,
        "usage_frac":         usage_frac,
        "entropy_mean":       float(ent.mean()),
        "entropy_std":        float(ent.std()),
        "max_entropy":        float(np.log(K)),
    }


# ---------------------------------------------------------------------------
# Pannello D — CRPS sanity check (GMM vs Gaussian baseline)
# ---------------------------------------------------------------------------

def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Closed-form CRPS for a single Gaussian N(mu, sigma²)."""
    z = (y - mu) / (sigma + 1e-8)
    phi = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    crps_val = sigma * (z * (2 * phi - 1) + 2 * pdf - 1 / math.sqrt(math.pi))
    return float(crps_val.mean())


def crps_sanity_check(metrics: dict) -> dict:
    """
    CRPS della GMM vs gaussiana moment-matched.
    La GMM non deve essere peggio — ma il valore vero è nella struttura,
    non nel CRPS.
    """
    errors  = metrics["errors"]       # (M, N, D)
    pi      = metrics["pi_all"]       # (M, N, K)
    mu      = metrics["mu_all"]       # (M, N, K, D)
    log_sig = metrics["log_sig_cat"]  # (M, N, K, D)

    sig = np.exp(log_sig)

    # GMM weighted mean and variance
    gmm_mean = (pi[..., np.newaxis] * mu).sum(axis=2)  # (M, N, D)
    gmm_var  = (pi[..., np.newaxis] * (sig**2 + mu**2)).sum(axis=2) - gmm_mean**2
    gmm_std  = np.sqrt(np.maximum(gmm_var, 1e-8))

    # z_next = gmm_mean - errors  (errors = z_pred - z_next, z_pred = gmm_mean)
    z_next = gmm_mean - errors

    # Gaussian baseline CRPS (moment-matched)
    crps_gauss = crps_gaussian(gmm_mean, gmm_std, z_next)

    # GMM CRPS (Monte Carlo: energy form)
    N_mc = 200
    M_, N_, K_, D_ = mu.shape
    rng = np.random.default_rng(42)

    pi_flat  = pi.reshape(-1, K_)
    mu_flat  = mu.reshape(-1, K_, D_)
    sig_flat = sig.reshape(-1, K_, D_)
    z_flat   = z_next.reshape(-1, D_)
    n_points = pi_flat.shape[0]

    max_pts = 50_000
    if n_points > max_pts:
        idx = rng.choice(n_points, size=max_pts, replace=False)
        pi_flat  = pi_flat[idx]
        mu_flat  = mu_flat[idx]
        sig_flat = sig_flat[idx]
        z_flat   = z_flat[idx]
        n_points = max_pts

    # Draw samples from GMM
    comp_idx = np.array([
        rng.choice(K_, size=N_mc, p=pi_flat[i]) for i in range(n_points)
    ])  # (n_points, N_mc)

    samples = np.zeros((n_points, N_mc, D_))
    for s in range(N_mc):
        cidx = comp_idx[:, s]
        mu_s  = mu_flat[np.arange(n_points), cidx]
        sig_s = sig_flat[np.arange(n_points), cidx]
        samples[:, s] = mu_s + sig_s * rng.standard_normal((n_points, D_))

    # E|X - y|
    term1 = np.abs(samples - z_flat[:, np.newaxis, :]).mean(axis=(1, 2))
    # E|X - X'|
    half = N_mc // 2
    term2 = np.abs(samples[:, :half] - samples[:, half:2*half]).mean(axis=(1, 2))

    crps_gmm = float((term1 - 0.5 * term2).mean())

    return {
        "crps_gmm":      crps_gmm,
        "crps_gaussian":  crps_gauss,
        "crps_ratio":     crps_gmm / (crps_gauss + 1e-8),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_all(metrics: dict, calib: dict, coverage: dict, crps: dict) -> None:
    names = ["low_vol", "mid_vol", "high_vol"]

    print("\n" + "=" * 70)
    print('PANNELLO A — IL WM È CALIBRATO?  (σ/|err| → ε ha senso fisico)')
    print("=" * 70)
    print(f"  σ/|err| globale: {calib['calib_ratio_mean']:.3f}  (ideale ≈ 1.0)")
    print(f"\n  σ/|err| per regime:")
    for r, name in enumerate(names):
        if r in calib["calib_per_regime"]:
            print(f"    {name:10s}: {calib['calib_per_regime'][r]:.3f}")

    print(f"\n  Per-dim calibration (first 8 dims):")
    for i in range(min(8, len(calib["calib_ratio"]))):
        print(f"    dim {i:2d}: σ={calib['sig_per_dim'][i]:.4f}  "
              f"|err|={calib['abs_err_per_dim'][i]:.4f}  "
              f"ratio={calib['calib_ratio'][i]:.3f}")

    print("\n" + "=" * 70)
    print('PANNELLO B — LA GMM GENERA SCENARI DIVERSI?  (copertura failure modes)')
    print("=" * 70)
    K = coverage["K"]
    print(f"  K = {K} componenti")
    print(f"  Separazione centroidi: {coverage['separation_l2']:.4f} L2  "
          f"= {coverage['separation_sigma']:.2f}σ")
    print(f"  K_eff (1/Σπ²): {coverage['k_eff_mean']:.2f} / {K}")
    print(f"  Entropia π: {coverage['entropy_mean']:.3f} ± {coverage['entropy_std']:.3f}  "
          f"(max = ln({K}) = {coverage['max_entropy']:.3f})")
    print(f"\n  Component usage (argmax):")
    for k in range(K):
        print(f"    comp {k}: {coverage['usage_frac'][k]*100:.1f}%")
    print(f"\n  Media π per regime:")
    header = "           " + "  ".join(f"comp_{k}" for k in range(K))
    print(header)
    for r, name in enumerate(names):
        if r in coverage["pi_per_regime"]:
            vals = "  ".join(f"{v:.4f}" for v in coverage["pi_per_regime"][r])
            print(f"    {name:10s} {vals}")

    print("\n" + "=" * 70)
    print('PANNELLO C — IL WM DISTINGUE I REGIMI?  (ambiguity set regime-dipendente)')
    print("=" * 70)
    print(f"  NLL globale: {metrics['nll_mean']:.4f}")
    print(f"  NLL per regime:")
    for r, name in enumerate(names):
        if r in metrics["regime_nll"]:
            print(f"    {name:10s}: {metrics['regime_nll'][r]:.4f}")
    pnll = metrics["pos_nll_mean"]
    improvement = (pnll[0] - pnll[-1]) / abs(pnll[0]) * 100
    print(f"\n  NLL position 1 → {len(pnll)}: {pnll[0]:.4f} → {pnll[-1]:.4f}  "
          f"(improvement: {improvement:.1f}%)")

    print("\n" + "=" * 70)
    print('PANNELLO D — CRPS SANITY CHECK  (GMM non deve essere peggio)')
    print("=" * 70)
    print(f"  CRPS GMM:      {crps['crps_gmm']:.6f}")
    print(f"  CRPS Gaussian: {crps['crps_gaussian']:.6f}")
    ratio_pct = (1 - crps["crps_ratio"]) * 100
    status = "✓ GMM migliore" if crps["crps_ratio"] < 1.0 else "⚠ GMM peggiore"
    print(f"  Ratio GMM/Gauss: {crps['crps_ratio']:.4f}  ({status}, Δ={ratio_pct:+.1f}%)")

# ---------------------------------------------------------------------------
# Plotting — 8 Pannelli (Layout 4x2)
# ---------------------------------------------------------------------------

def plot_all(metrics: dict, calib: dict, coverage: dict, crps: dict,
             out_path: str) -> None:
    names_regime = ["low_vol", "mid_vol", "high_vol"]
    K = coverage["K"]
    D = len(metrics["per_dim_mse"])
    colors_regime = ["tab:blue", "tab:orange", "tab:red"]

    # Griglia 4x2 pulita e simmetrica
    fig = plt.figure(figsize=(18, 20))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.25)

    # 1. Predicted σ vs absolute error per dimension
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(D)
    w = 0.35
    ax.bar(x - w/2, calib["sig_per_dim"], w, label="Predicted σ", color="steelblue")
    ax.bar(x + w/2, calib["abs_err_per_dim"], w, label="|Error|", color="salmon")
    ax.set_title("Predicted σ vs absolute error per dimension")
    ax.set_xlabel("Dimension"); ax.set_ylabel("Mean value")
    ax.legend(fontsize=8); ax.set_xticks(x)

    # 2. Calibration ratio by regime
    ax = fig.add_subplot(gs[0, 1])
    r_names, r_vals = [], []
    for r, name in enumerate(names_regime):
        if r in calib["calib_per_regime"]:
            r_names.append(name)
            r_vals.append(calib["calib_per_regime"][r])
    ax.bar(r_names, r_vals, color=colors_regime[:len(r_names)])
    ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="Ideal = 1.0")
    ax.set_title("Calibration ratio by regime")
    ax.set_ylabel("Ratio (σ / |err|)")
    ax.legend(fontsize=8)
    for i, v in enumerate(r_vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # 3. Binned calibration
    ax = fig.add_subplot(gs[1, 0])
    if len(calib["bin_sig"]) > 0:
        ax.scatter(calib["bin_sig"], calib["bin_err"], c="steelblue", s=40, zorder=3)
        lims = [0, max(calib["bin_sig"].max(), calib["bin_err"].max()) * 1.1]
        ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Predicted σ (binned)")
        ax.set_ylabel("|Error| (binned)")
        ax.set_title("Binned calibration")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Mean mixture weights by regime
    ax = fig.add_subplot(gs[1, 1])
    pi_matrix = np.zeros((3, K))
    for r in range(3):
        if r in coverage["pi_per_regime"]:
            pi_matrix[r] = coverage["pi_per_regime"][r]
    im = ax.imshow(pi_matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"comp_{k}" for k in range(K)])
    ax.set_yticks(range(3))
    ax.set_yticklabels(names_regime)
    ax.set_title(f"Mean mixture weights by regime (K_eff={coverage['k_eff_mean']:.2f}/{K}, sep={coverage['separation_sigma']:.1f}σ)")
    plt.colorbar(im, ax=ax)
    for r in range(3):
        for k in range(K):
            ax.text(k, r, f"{pi_matrix[r, k]:.3f}",
                    ha="center", va="center", fontsize=9)

    # 5. NLL by regime
    ax = fig.add_subplot(gs[2, 0])
    reg_names, reg_vals = [], []
    for r, name in enumerate(names_regime):
        if r in metrics["regime_nll"]:
            reg_names.append(name)
            reg_vals.append(metrics["regime_nll"][r])
    ax.bar(reg_names, reg_vals, color=colors_regime[:len(reg_names)])
    ax.set_title("NLL by regime")
    ax.set_ylabel("NLL")
    for i, v in enumerate(reg_vals):
        ax.text(i, v - 0.3 if v > 0 else v + 0.1, f"{v:.2f}", ha="center", va="top" if v > 0 else "bottom",
                fontsize=9, color="white" if v > 0 else "black", fontweight="bold")

    # 6. NLL vs sequence position
    ax = fig.add_subplot(gs[2, 1])
    pnll = metrics["pos_nll_mean"]
    ax.plot(range(1, len(pnll) + 1), pnll, "b-o", markersize=3)
    ax.set_title("NLL vs sequence position (no systematic trend)")
    ax.set_xlabel("Position (t)"); ax.set_ylabel("NLL")
    ax.grid(True, alpha=0.3)

    # 7. CRPS: GMM vs moment-matched Gaussian
    ax = fig.add_subplot(gs[3, 0])
    crps_names = ["GMM", "Gaussian\n(moment-matched)"]
    crps_vals  = [crps["crps_gmm"], crps["crps_gaussian"]]
    ax.bar(crps_names, crps_vals, color=["steelblue", "lightcoral"], width=0.6)
    ax.set_title("CRPS: GMM vs moment-matched Gaussian")
    ax.set_ylabel("CRPS (lower = better)")
    for i, v in enumerate(crps_vals):
        ax.text(i, v + (max(crps_vals)*0.01), f"{v:.5f}", ha="center", fontsize=9)

    # 8. Per-dimension calibration ratio
    ax = fig.add_subplot(gs[3, 1])
    ax.bar(range(D), calib["calib_ratio"], color="seagreen", alpha=0.7)
    ax.axhline(1.0, color="red", ls="--", label="Ideal (ratio=1.0)")
    ax.set_title("Per-dimension calibration ratio")
    ax.set_xlabel("Dimension"); ax.set_ylabel("Ratio (σ / |err|)")
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, D, max(1, D // 8)))

    plt.suptitle("World Model Evaluation", fontsize=18, fontweight="bold", y=0.93)
                 
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_world_model(args.wm_ckpt, device)
    sequences, actions, rewards, regimes = load_dataset(
        args.dataset, n_samples=args.n_samples
    )

    print("\nComputing one-step metrics...")
    metrics = one_step_metrics(model, sequences, actions, regimes, device)

    print("Computing calibration analysis (Pannello A)...")
    calib = calibration_analysis(metrics, regimes)

    print("Computing coverage analysis (Pannello B)...")
    coverage = coverage_analysis(metrics, regimes)

    print("Computing CRPS sanity check (Pannello D)...")
    crps = crps_sanity_check(metrics)

    print_all(metrics, calib, coverage, crps)
    plot_all(metrics, calib, coverage, crps, out_path=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate World Model (Pannelli A-D)")
    parser.add_argument("--wm_ckpt",    type=str,  default="checkpoints/wm_best.pt")
    parser.add_argument("--dataset",    type=str,  default="data/wm_dataset.npz")
    parser.add_argument("--n_samples",  type=int,  default=10_000)
    parser.add_argument("--out",        type=str,  default="eval_wm.png")
    args = parser.parse_args()
    main(args)
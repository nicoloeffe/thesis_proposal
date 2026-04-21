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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
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

        pi, mu, log_sig = model(z_seq)
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
#
# v2 (closed-form per-dimension CRPS):
#   Entrambi CRPS_GMM e CRPS_Gaussian sono calcolati come media per-dim di
#   CRPS univariati in closed-form, quindi sono matematicamente confrontabili.
#
#   CRPS univariato per una N(μ, σ²):
#     CRPS(N(μ,σ²), y) = σ · [ 2φ((y-μ)/σ) + (y-μ)/σ · (2Φ((y-μ)/σ) - 1) ]
#                        - σ/√π
#
#   CRPS univariato per una GMM Σ_k π_k N(μ_k, σ_k²) (Grimit et al. 2006):
#     CRPS = Σ_k π_k · A(μ_k, σ_k, y)
#            - 0.5 · Σ_{k,k'} π_k π_{k'} · A(μ_k - μ_{k'}, √(σ_k² + σ_{k'}²), 0)
#
#   dove A(μ, σ, y) = E_{X~N(μ,σ²)}[|X - y|] è la parte "expected |X - y|".

def _gaussian_expected_abs(mu: np.ndarray, sigma: np.ndarray,
                           y: np.ndarray | float = 0.0) -> np.ndarray:
    """
    E_{X~N(μ,σ²)}[|X - y|]  =  σ · [ 2φ((y-μ)/σ) + (y-μ)/σ · (2Φ((y-μ)/σ) - 1) ]
    Accetta broadcast.
    """
    from scipy.special import erf
    s = np.maximum(sigma, 1e-12)
    z = (y - mu) / s
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2)))
    return s * (2.0 * phi + z * (2.0 * Phi - 1.0))


def crps_gaussian_perdim(mu: np.ndarray, sigma: np.ndarray,
                         y: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS univariato per una singola gaussiana N(μ, σ²), per-dim.
    Returns array with same shape as mu/sigma/y.
    """
    exp_abs_y = _gaussian_expected_abs(mu, sigma, y)               # E|X-y|
    # E|X-X'| per X,X' iid N(μ,σ²) = 2σ/√π
    exp_abs_xx = 2.0 * sigma / np.sqrt(np.pi)
    return exp_abs_y - 0.5 * exp_abs_xx


def crps_gmm_perdim(pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                    y: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS univariato per una GMM diagonale, per-dim.

    Formula (Grimit et al. 2006, Section 3.2):
      CRPS(F, y) = Σ_k π_k A(μ_k, σ_k, y)
                   - 0.5 Σ_{k,k'} π_k π_{k'} A(μ_k - μ_{k'}, √(σ_k²+σ_{k'}²), 0)

    Args shapes:
      pi     : (..., K)       — mixture weights
      mu     : (..., K, D)    — per-component means
      sigma  : (..., K, D)    — per-component stds
      y      : (..., D)       — observed value

    Returns:
      crps : (..., D)         — CRPS per dim

    Sanity-verified:
      - K=1 collapses to single-Gaussian CRPS
      - Bimodal GMM beats moment-matched Gaussian when evaluated under the
        true bimodal distribution (on-mode points dominate)
    """
    # Termine 1: Σ_k π_k · E|X_k - y|
    y_expanded = y[..., np.newaxis, :]                             # (..., 1, D)
    term1_per_k = _gaussian_expected_abs(mu, sigma, y_expanded)    # (..., K, D)
    term1 = (pi[..., np.newaxis] * term1_per_k).sum(axis=-2)       # (..., D)

    # Termine 2: 0.5 · Σ_{k,k'} π_k π_{k'} · E|X_k - X_{k'}|
    # dove X_k ~ N(μ_k, σ_k²), X_{k'} ~ N(μ_{k'}, σ_{k'}²) indipendenti
    # → X_k - X_{k'} ~ N(μ_k - μ_{k'}, σ_k² + σ_{k'}²)
    mu_k      = mu[..., :, np.newaxis, :]                          # (..., K, 1, D)
    mu_kp     = mu[..., np.newaxis, :, :]                          # (..., 1, K, D)
    sig2_k    = (sigma ** 2)[..., :, np.newaxis, :]                # (..., K, 1, D)
    sig2_kp   = (sigma ** 2)[..., np.newaxis, :, :]                # (..., 1, K, D)
    diff_mu   = mu_k - mu_kp                                        # (..., K, K, D)
    sum_sig   = np.sqrt(sig2_k + sig2_kp)                          # (..., K, K, D)

    exp_abs = _gaussian_expected_abs(diff_mu, sum_sig, 0.0)        # (..., K, K, D)

    pi_outer = pi[..., :, np.newaxis] * pi[..., np.newaxis, :]     # (..., K, K)
    term2    = 0.5 * (pi_outer[..., np.newaxis] * exp_abs).sum(axis=(-3, -2))  # (..., D)

    return term1 - term2


def crps_sanity_check(metrics: dict) -> dict:
    """
    CRPS della GMM vs gaussiana moment-matched.

    v2: entrambi closed-form univariati per-dim, poi mediati. Così sono
    matematicamente confrontabili.

    Interpretazione:
      - ratio < 1: GMM meglio della gaussiana moment-matched (cattura multi-
                   modalità o code non-gaussiane)
      - ratio ≈ 1: distribuzione vera essenzialmente gaussiana per-regime,
                   GMM è equivalente a una singola gaussiana.
      - ratio > 1: sarebbe patologico (GMM sub-ottima); non dovrebbe succedere
                   se la GMM è ben allenata.
    """
    errors  = metrics["errors"]       # (M, N, D)
    pi      = metrics["pi_all"]       # (M, N, K)
    mu      = metrics["mu_all"]       # (M, N, K, D)
    log_sig = metrics["log_sig_cat"]  # (M, N, K, D)
    sig     = np.exp(log_sig)

    # Reconstruct z_next: errors = z_pred - z_next, z_pred = GMM weighted mean
    gmm_mean = (pi[..., np.newaxis] * mu).sum(axis=2)             # (M, N, D)
    gmm_var  = (pi[..., np.newaxis] * (sig ** 2 + mu ** 2)).sum(axis=2) - gmm_mean ** 2
    gmm_std  = np.sqrt(np.maximum(gmm_var, 1e-10))                # (M, N, D)
    z_next   = gmm_mean - errors                                   # (M, N, D)

    # Flatten (M, N, ...) → (M*N, ...)
    M_, N_, K_, D_ = mu.shape
    pi_f    = pi.reshape(-1, K_)
    mu_f    = mu.reshape(-1, K_, D_)
    sig_f   = sig.reshape(-1, K_, D_)
    zn_f    = z_next.reshape(-1, D_)
    gmean_f = gmm_mean.reshape(-1, D_)
    gstd_f  = gmm_std.reshape(-1, D_)

    # Subsample if too many points (formula is O(N*K² * D), manageable but
    # memory spikes on large D*K*K)
    max_pts = 50_000
    if pi_f.shape[0] > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(pi_f.shape[0], size=max_pts, replace=False)
        pi_f, mu_f, sig_f = pi_f[idx], mu_f[idx], sig_f[idx]
        zn_f, gmean_f, gstd_f = zn_f[idx], gmean_f[idx], gstd_f[idx]

    # --- CRPS per-dim for GMM (closed-form) ---
    crps_gmm_pd = crps_gmm_perdim(pi_f, mu_f, sig_f, zn_f)         # (n, D)
    # --- CRPS per-dim for moment-matched Gaussian (closed-form) ---
    crps_gauss_pd = crps_gaussian_perdim(gmean_f, gstd_f, zn_f)    # (n, D)

    # Global means
    crps_gmm   = float(crps_gmm_pd.mean())
    crps_gauss = float(crps_gauss_pd.mean())

    # Per-dim means for diagnostics
    crps_gmm_per_dim   = crps_gmm_pd.mean(axis=0)                  # (D,)
    crps_gauss_per_dim = crps_gauss_pd.mean(axis=0)                # (D,)

    ratio = crps_gmm / (crps_gauss + 1e-12)

    return {
        "crps_gmm":           crps_gmm,
        "crps_gaussian":      crps_gauss,
        "crps_ratio":         ratio,
        "crps_gmm_per_dim":   crps_gmm_per_dim,
        "crps_gauss_per_dim": crps_gauss_per_dim,
        "crps_gain_per_dim":  crps_gauss_per_dim - crps_gmm_per_dim,
    }



# ---------------------------------------------------------------------------
# Pannello E — Density calibration (PIT + reliability + MMD)
# ---------------------------------------------------------------------------

def gmm_marginal_cdf(
    pi:      np.ndarray,   # (N, K)
    mu:      np.ndarray,   # (N, K, D)
    sigma:   np.ndarray,   # (N, K, D)
    z_true:  np.ndarray,   # (N, D)
) -> np.ndarray:
    """
    Compute F_GMM(z_true_d) per-dimension for a diagonal-covariance GMM.

    For each sample n, for each dim d:
      F(z) = Σ_k π_k * Φ((z - μ_k,d) / σ_k,d)

    Returns u ∈ [0, 1]^(N, D). Under perfect calibration, u ~ Uniform[0,1].
    """
    from scipy.special import erf
    # Standard normal CDF: Φ(x) = 0.5 * (1 + erf(x/√2))
    z = z_true[:, np.newaxis, :]                           # (N, 1, D)
    std_z = (z - mu) / (sigma + 1e-12)                     # (N, K, D)
    phi_per_comp = 0.5 * (1.0 + erf(std_z / np.sqrt(2)))   # (N, K, D)
    u = (pi[..., np.newaxis] * phi_per_comp).sum(axis=1)   # (N, D)
    return u


def ks_uniform(u: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov distance between empirical CDF of u and Uniform[0,1].
    Returns (KS statistic, approximate p-value).
    """
    from scipy.stats import kstest
    res = kstest(u, "uniform")
    return float(res.statistic), float(res.pvalue)


def coverage_curve(u: np.ndarray, n_levels: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Reliability curve: for each α level, compute empirical coverage
    of central predictive interval.

    Central α interval = [u low, u high] = [(1-α)/2, (1+α)/2]
    Coverage = fraction of u in that interval.
    """
    alphas = np.linspace(0.05, 0.99, n_levels)
    covs = []
    for a in alphas:
        lo = (1 - a) / 2
        hi = (1 + a) / 2
        c = ((u >= lo) & (u <= hi)).mean()
        covs.append(c)
    return alphas, np.array(covs)


def gmm_sample(
    pi:    np.ndarray,   # (N, K)
    mu:    np.ndarray,   # (N, K, D)
    sigma: np.ndarray,   # (N, K, D)
    n_samples: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample from batch of GMMs. Returns (N, n_samples, D) if n_samples > 1,
    else (N, D).
    """
    rng = np.random.default_rng(seed)
    N_, K_, D_ = mu.shape

    if n_samples == 1:
        comp = np.array([rng.choice(K_, p=pi[i]) for i in range(N_)])   # (N,)
        mu_s    = mu[np.arange(N_), comp]
        sigma_s = sigma[np.arange(N_), comp]
        return mu_s + sigma_s * rng.standard_normal((N_, D_))            # (N, D)
    else:
        out = np.zeros((N_, n_samples, D_))
        for s in range(n_samples):
            comp = np.array([rng.choice(K_, p=pi[i]) for i in range(N_)])
            mu_s = mu[np.arange(N_), comp]
            sigma_s = sigma[np.arange(N_), comp]
            out[:, s] = mu_s + sigma_s * rng.standard_normal((N_, D_))
        return out


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """
    Maximum Mean Discrepancy between two samples X, Y using RBF kernel.
    MMD² = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]

    If sigma is None, use median heuristic.
    Returns MMD² (unbiased estimator).
    """
    from scipy.spatial.distance import cdist
    n = len(X); m = len(Y)

    # Median heuristic for bandwidth
    if sigma is None:
        combined = np.vstack([X, Y])
        dists = cdist(combined[:1000], combined[:1000], metric="sqeuclidean")
        sigma = float(np.sqrt(0.5 * np.median(dists[dists > 0])))
        sigma = max(sigma, 1e-6)

    def K(A, B):
        D = cdist(A, B, metric="sqeuclidean")
        return np.exp(-D / (2 * sigma ** 2))

    K_xx = K(X, X)
    K_yy = K(Y, Y)
    K_xy = K(X, Y)

    # Unbiased MMD² (exclude diagonal)
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)
    mmd2 = K_xx.sum() / (n * (n - 1)) + K_yy.sum() / (m * (m - 1)) - 2 * K_xy.mean()
    return float(mmd2)


def calibration_pit_analysis(
    metrics: dict,
    sequences: torch.Tensor,
    regimes: torch.Tensor,
    n_mmd_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Density calibration:
      - PIT per-dim: u_{n,d} = F_GMM(z_true_{n,d}) ~ Uniform[0,1]
      - KS statistic + p-value per-dim
      - Reliability curve (empirical vs nominal coverage)
      - Sharpness: mean of predicted σ
      - Two-sample MMD: between real z_{t+1} and GMM samples
    """
    pi      = metrics["pi_all"]        # (M, N, K)
    mu      = metrics["mu_all"]        # (M, N, K, D)
    log_sig = metrics["log_sig_cat"]   # (M, N, K, D)
    sig     = np.exp(log_sig)

    # z_next reconstruction: errors = z_pred - z_next, z_pred = GMM weighted mean
    z_pred = (pi[..., np.newaxis] * mu).sum(axis=2)          # (M, N, D)
    z_next = z_pred - metrics["errors"]                      # (M, N, D)

    M_, N_, K_, D_ = mu.shape

    # Flatten to (M*N, ...)
    pi_f  = pi.reshape(-1, K_)
    mu_f  = mu.reshape(-1, K_, D_)
    sig_f = sig.reshape(-1, K_, D_)
    zn_f  = z_next.reshape(-1, D_)
    total_points = pi_f.shape[0]

    # Subsample for MMD (computationally expensive)
    rng = np.random.default_rng(seed)
    if total_points > n_mmd_samples:
        idx = rng.choice(total_points, size=n_mmd_samples, replace=False)
    else:
        idx = np.arange(total_points)

    # --- PIT per-dim (su tutti i punti) ---
    u = gmm_marginal_cdf(pi_f, mu_f, sig_f, zn_f)   # (total, D)

    # KS per-dim
    ks_per_dim = []
    pval_per_dim = []
    for d in range(D_):
        ks, pv = ks_uniform(u[:, d])
        ks_per_dim.append(ks)
        pval_per_dim.append(pv)
    ks_per_dim = np.array(ks_per_dim)
    pval_per_dim = np.array(pval_per_dim)

    # --- Reliability curve (aggregato su tutte le dim) ---
    alphas, covs = coverage_curve(u.flatten())
    calib_error_l1 = float(np.abs(alphas - covs).mean())

    # --- Sharpness ---
    # Weighted σ per-sample per-dim, then mean
    sig_avg = (pi_f[..., np.newaxis] * sig_f).sum(axis=1)   # (total, D)
    sharpness = float(sig_avg.mean())

    # --- Two-sample MMD (tra z_next veri e sample dalla GMM) ---
    pi_sub  = pi_f[idx]
    mu_sub  = mu_f[idx]
    sig_sub = sig_f[idx]
    zn_sub  = zn_f[idx]
    z_gen = gmm_sample(pi_sub, mu_sub, sig_sub, n_samples=1, seed=seed)
    mmd2 = mmd_rbf(zn_sub, z_gen)

    return {
        "u_pit":          u,              # (total, D) per plotting
        "ks_per_dim":     ks_per_dim,
        "pval_per_dim":   pval_per_dim,
        "ks_mean":        float(ks_per_dim.mean()),
        "alphas":         alphas,
        "coverages":      covs,
        "calib_error_l1": calib_error_l1,
        "sharpness":      sharpness,
        "mmd2":           mmd2,
        "mmd_sigma_used": "median_heuristic",
    }


# ---------------------------------------------------------------------------
# Pannello F — Autoregressive rollout (diagnostic, NOT a training objective)
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_autoregressive(
    model: LOBWorldModel,
    context_seq: torch.Tensor,    # (B, N_ctx, D) — initial real context
    horizon: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """
    Autoregressive rollout via stochastic sampling from GMM.

    At each step t:
      1. Forward pass on current sequence → GMM parameters
      2. Sample z_{t+1} from the GMM at last position
      3. Append to sequence, repeat

    Note: the model is NOT trained autoregressively (teacher forcing only).
    Rollouts test generalization beyond training distribution (exposure bias).

    Returns: (B, N_ctx + horizon, D) full trajectory including context.
    """
    torch.manual_seed(seed)
    B, N_ctx, D = context_seq.shape
    z_hist = context_seq.clone().to(device)

    for _ in range(horizon):
        # Forward takes sequence of shape (B, T, D) and returns predictions
        # at each position. We use LAST position's GMM to sample z_next.
        # The model expects (B, T+1, D) to predict T positions; we pass
        # z_hist + dummy last token, then take position -1 prediction.

        # Trick: append a placeholder zero vector, the model will predict
        # based on positions 0..T-1. We use pred at position -1 (predicting
        # from full z_hist context).
        placeholder = torch.zeros(B, 1, D, device=device)
        z_input = torch.cat([z_hist, placeholder], dim=1)    # (B, T+1, D)
        pi, mu, log_sig = model(z_input)                     # (B, T, K) etc
        sig = torch.exp(log_sig)

        # Last position: (B, K), (B, K, D), (B, K, D)
        pi_last  = pi[:, -1, :]
        mu_last  = mu[:, -1, :, :]
        sig_last = sig[:, -1, :, :]

        # Sample component
        comp = torch.multinomial(pi_last, num_samples=1).squeeze(-1)   # (B,)
        b_idx = torch.arange(B, device=device)
        mu_s  = mu_last[b_idx, comp]                                    # (B, D)
        sig_s = sig_last[b_idx, comp]                                   # (B, D)

        # Sample z_{t+1} ~ N(mu_s, diag(sig_s²))
        noise = torch.randn(B, D, device=device)
        z_new = mu_s + sig_s * noise                                    # (B, D)

        z_hist = torch.cat([z_hist, z_new.unsqueeze(1)], dim=1)

    return z_hist   # (B, N_ctx + horizon, D)


def train_regime_probe_on_latents(
    sequences: np.ndarray,    # (M, N+1, D)  — normalized z
    regimes: np.ndarray,      # (M, N) or (M,) — per-step or per-ep
    seed: int = 42,
) -> tuple[object, object]:
    """
    Train a shallow MLP probe (sklearn) to classify regime from single z.
    Returns (model, scaler).

    Used to check whether autoregressive rollouts preserve the regime
    of their initial context (consistency test).
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    M, Np1, D = sequences.shape

    # Flatten to per-step samples
    if regimes.ndim == 2:
        reg_flat = regimes.flatten()                     # (M*N,)
        z_flat = sequences[:, :-1, :].reshape(-1, D)     # (M*N, D), align with reg
    else:
        reg_flat = np.repeat(regimes, Np1)
        z_flat = sequences.reshape(-1, D)

    # Subsample for speed
    if len(z_flat) > 50_000:
        idx = rng.choice(len(z_flat), size=50_000, replace=False)
        z_flat = z_flat[idx]
        reg_flat = reg_flat[idx]

    scaler = StandardScaler().fit(z_flat)
    z_scaled = scaler.transform(z_flat)

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        verbose=False,
    )
    clf.fit(z_scaled, reg_flat.astype(int))
    train_acc = clf.score(z_scaled, reg_flat.astype(int))
    print(f"  regime probe trained: accuracy={train_acc:.3f}  n_iter={clf.n_iter_}")
    return clf, scaler


def rollout_analysis(
    model: LOBWorldModel,
    sequences: torch.Tensor,     # (M, N+1, D) all data
    regimes: torch.Tensor,       # (M, N) per-step
    device: torch.device,
    n_rollouts: int = 500,
    horizon: int = 50,
    ctx_len: int = 20,
    seed: int = 42,
) -> dict:
    """
    Autoregressive rollout test.

    For n_rollouts sequences:
      1. Use first ctx_len steps as real context
      2. Rollout forward for `horizon` steps via stochastic sampling
      3. Compare with the trajectory the model would produce under
         teacher forcing (not applicable here — no real z_{t+1} for steps
         beyond dataset end)

    Since sequences have length N+1 = ctx_len+1 = 21, we can only
    compare with truth for h=1 step. For h > 1 we can only measure:
      - Rollout stability (does ||z|| explode?)
      - Distribution matching (are rollout z's from same distribution as real z?)
      - Regime consistency (do rolled-out z's preserve their initial regime?)

    Returns dict with per-horizon metrics.
    """
    rng = np.random.default_rng(seed)
    M = len(sequences)
    idx = rng.choice(M, size=min(n_rollouts, M), replace=False)

    seq_sub = sequences[idx]                          # (n, N+1, D)
    reg_sub = regimes[idx] if regimes.dim() == 2 else regimes[idx].unsqueeze(1).expand(-1, sequences.shape[1] - 1)

    # Determine initial regime per rollout (take first step's regime)
    reg_init = reg_sub[:, 0].cpu().numpy()            # (n,)
    print(f"  initial regime distribution: {np.bincount(reg_init)}")

    # Context: first ctx_len of the sequence (require ctx_len <= N+1)
    N_data = seq_sub.shape[1]
    if ctx_len >= N_data:
        ctx_len = N_data - 1
        print(f"  ctx_len adjusted to {ctx_len} (data seq_len={N_data})")

    context = seq_sub[:, :ctx_len, :]                 # (n, ctx_len, D)

    # --- Rollout ---
    print(f"  rolling out {len(idx)} trajectories for {horizon} steps (ctx={ctx_len})...")
    full = rollout_autoregressive(model, context, horizon, device, seed=seed)
    full_np = full.cpu().numpy()                      # (n, ctx_len+horizon, D)

    # Extract the rolled-out segment (after context)
    rolled = full_np[:, ctx_len:, :]                  # (n, horizon, D)

    # --- Metric 1: Rollout norm stability ---
    norms = np.linalg.norm(rolled, axis=-1)           # (n, horizon)
    norm_mean_per_step  = norms.mean(axis=0)          # (horizon,)
    norm_p5_per_step    = np.percentile(norms, 5, axis=0)
    norm_p95_per_step   = np.percentile(norms, 95, axis=0)

    # --- Metric 2: Distribution matching (mean & std per dim over rollouts vs real) ---
    # Real z distribution: use all z from the sequences (flatten)
    real_z = sequences.cpu().numpy().reshape(-1, sequences.shape[-1])   # (M*(N+1), D)
    real_mean = real_z.mean(axis=0)
    real_std  = real_z.std(axis=0)

    # Rollout z distribution per horizon step
    rolled_mean_per_step = rolled.mean(axis=0)         # (horizon, D)
    rolled_std_per_step  = rolled.std(axis=0)          # (horizon, D)

    # L2 distance of mean/std profiles from expected
    mean_dev_per_step = np.linalg.norm(rolled_mean_per_step - real_mean, axis=-1)  # (horizon,)
    std_dev_per_step  = np.linalg.norm(rolled_std_per_step - real_std, axis=-1)

    # --- Metric 3: Regime consistency (train probe on real z, apply to rollouts) ---
    print(f"  training regime probe on real latents...")
    probe, scaler = train_regime_probe_on_latents(
        sequences.cpu().numpy(),
        regimes.cpu().numpy(),
        seed=seed,
    )

    # Classify each rolled-out z, compute fraction matching initial regime
    regime_consistency_per_step = np.zeros(horizon)
    for h in range(horizon):
        z_h = rolled[:, h, :]                               # (n, D)
        z_h_scaled = scaler.transform(z_h)
        pred_h = probe.predict(z_h_scaled)                  # (n,)
        regime_consistency_per_step[h] = (pred_h == reg_init).mean()

    # Per-regime breakdown at h=50 (end of rollout)
    consistency_per_regime_end = {}
    h_last = horizon - 1
    z_last = rolled[:, h_last, :]
    z_last_scaled = scaler.transform(z_last)
    pred_last = probe.predict(z_last_scaled)
    for r in [0, 1, 2]:
        mask = reg_init == r
        if mask.sum() > 0:
            consistency_per_regime_end[r] = float((pred_last[mask] == r).mean())

    return {
        "ctx_len":                   ctx_len,
        "horizon":                   horizon,
        "n_rollouts":                len(idx),
        "norm_mean_per_step":        norm_mean_per_step,
        "norm_p5_per_step":          norm_p5_per_step,
        "norm_p95_per_step":         norm_p95_per_step,
        "real_norm_mean":            float(np.linalg.norm(real_mean)),
        "real_std_mean":             float(real_std.mean()),
        "mean_dev_per_step":         mean_dev_per_step,
        "std_dev_per_step":          std_dev_per_step,
        "regime_consistency_per_step": regime_consistency_per_step,
        "consistency_per_regime_end": consistency_per_regime_end,
        "regime_init_counts":        np.bincount(reg_init, minlength=3).tolist(),
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


def print_calibration(calibpit: dict) -> None:
    print("\n" + "=" * 70)
    print('PANNELLO E — DENSITY CALIBRATION (PIT + reliability + MMD)')
    print("=" * 70)

    ks_pd = calibpit["ks_per_dim"]
    pv_pd = calibpit["pval_per_dim"]
    print(f"\n  KS statistic (ideal ≈ 0):")
    print(f"    mean across dims : {calibpit['ks_mean']:.4f}")
    print(f"    worst dim        : {ks_pd.max():.4f} (dim {int(ks_pd.argmax())})")
    print(f"    best dim         : {ks_pd.min():.4f} (dim {int(ks_pd.argmin())})")

    n_pass = (pv_pd > 0.05).sum()
    print(f"\n  Dims passing KS (p > 0.05): {n_pass}/{len(pv_pd)}")

    print(f"\n  Reliability curve (coverage):")
    print(f"    mean |coverage - α| : {calibpit['calib_error_l1']:.4f}  "
          f"(ideal = 0)")

    print(f"\n  Sharpness (mean predicted σ): {calibpit['sharpness']:.4f}")
    print(f"  Two-sample MMD² (real vs GMM samples): {calibpit['mmd2']:.6f}  "
          f"(ideal ≈ 0)")

    # Interpretative verdict
    print(f"\n  Interpretation:")
    if calibpit['ks_mean'] < 0.05 and calibpit['calib_error_l1'] < 0.03:
        print(f"    [OK ] density well-calibrated — predictive quantiles match data")
    elif calibpit['ks_mean'] < 0.10:
        print(f"    [~~ ] density moderately calibrated — usable for DRO")
    else:
        print(f"    [!! ] density miscalibrated — ε in DRO may be wrong scale")


def print_rollout(ro: dict) -> None:
    print("\n" + "=" * 70)
    print('PANNELLO F — AUTOREGRESSIVE ROLLOUT (diagnostic off-training-distribution)')
    print("=" * 70)
    names = ["low_vol", "mid_vol", "high_vol"]

    print(f"\n  Setup: ctx={ro['ctx_len']} steps, horizon={ro['horizon']} steps, "
          f"n_rollouts={ro['n_rollouts']}")
    print(f"  Initial regime counts: {ro['regime_init_counts']}")

    nm = ro["norm_mean_per_step"]
    print(f"\n  Rollout norm stability:")
    print(f"    h=1  : mean_norm={nm[0]:.3f}")
    print(f"    h=10 : mean_norm={nm[9]:.3f}"  if len(nm) > 9  else "")
    print(f"    h=25 : mean_norm={nm[24]:.3f}" if len(nm) > 24 else "")
    print(f"    h=50 : mean_norm={nm[-1]:.3f}")
    print(f"    real  : mean_norm={ro['real_norm_mean']:.3f}  "
          f"(on normalized latents ≈ √d ≈ {np.sqrt(16):.2f})")

    md = ro["mean_dev_per_step"]
    sd = ro["std_dev_per_step"]
    print(f"\n  Distribution matching (L2 deviation from real moments):")
    print(f"    h=1  : ||μ_dev||={md[0]:.4f}  ||σ_dev||={sd[0]:.4f}")
    print(f"    h=50 : ||μ_dev||={md[-1]:.4f}  ||σ_dev||={sd[-1]:.4f}")

    rc = ro["regime_consistency_per_step"]
    print(f"\n  Regime consistency (rolled z classified as initial regime):")
    print(f"    h=1  : {rc[0]*100:.1f}%")
    print(f"    h=10 : {rc[9]*100:.1f}%"  if len(rc) > 9  else "")
    print(f"    h=25 : {rc[24]*100:.1f}%" if len(rc) > 24 else "")
    print(f"    h=50 : {rc[-1]*100:.1f}%")

    print(f"\n  Regime consistency at h={ro['horizon']} per regime:")
    for r, name in enumerate(names):
        if r in ro["consistency_per_regime_end"]:
            print(f"    {name:10s}: {ro['consistency_per_regime_end'][r]*100:.1f}%")

    # Interpretative verdict
    print(f"\n  Interpretation:")
    final_rc = rc[-1]
    norm_ratio = nm[-1] / (ro['real_norm_mean'] + 1e-6)
    if final_rc > 0.75 and 0.7 < norm_ratio < 1.4:
        print(f"    [OK ] rollouts stable + preserve regime — WM generalizes off-distribution")
    elif final_rc > 0.5 and 0.5 < norm_ratio < 2.0:
        print(f"    [~~ ] rollouts degrade gradually — expected from teacher-forced training")
    else:
        print(f"    [!! ] rollouts unstable or regime-drifting — exposure bias dominates")
    print(f"  (Not a blocker for Module C, which uses only 1-step predictions.)")


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
    ax.set_title("NLL vs sequence position (transformer uses temporal info)")
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
# Plotting — Calibration (PIT + reliability + MMD)
# ---------------------------------------------------------------------------

def plot_calibration(calibpit: dict, out_path: str) -> None:
    """
    Figura dedicata alla calibrazione della densità predittiva:
      - Grid 4x4: PIT histograms per tutte le 16 dimensioni
      - Sotto: reliability diagram + summary bars
    Total layout: 5x4 (top 4x4 for PIT, bottom row full-width split)
    """
    u = calibpit["u_pit"]                          # (total, D)
    ks_pd = calibpit["ks_per_dim"]
    pv_pd = calibpit["pval_per_dim"]
    alphas = calibpit["alphas"]
    coverages = calibpit["coverages"]

    D = u.shape[1]
    n_cols = 4
    n_rows_pit = (D + n_cols - 1) // n_cols        # 4 rows for 16 dims
    n_rows_total = n_rows_pit + 2                  # +2 for reliability + summary

    fig = plt.figure(figsize=(16, 3 * n_rows_total))
    gs  = gridspec.GridSpec(n_rows_total, n_cols, figure=fig,
                            hspace=0.55, wspace=0.35)

    # --- Top: PIT histogram per-dim ---
    for d in range(D):
        r = d // n_cols
        c = d % n_cols
        ax = fig.add_subplot(gs[r, c])
        ax.hist(u[:, d], bins=20, range=(0, 1), density=True,
                color="steelblue", edgecolor="white", alpha=0.85)
        ax.axhline(1.0, color="red", ls="--", lw=1.2, alpha=0.7,
                   label="uniform target")
        color = "green" if pv_pd[d] > 0.05 else ("orange" if pv_pd[d] > 0.01 else "red")
        ax.set_title(f"dim {d}: KS={ks_pd[d]:.3f}  p={pv_pd[d]:.2g}",
                     fontsize=9, color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(3.0, ax.get_ylim()[1]))
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("density", fontsize=8)
        if r == n_rows_pit - 1:
            ax.set_xlabel("u = F(z_true)", fontsize=8)

    # --- Reliability diagram (covering 2 columns) ---
    ax = fig.add_subplot(gs[n_rows_pit, 0:2])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(alphas, coverages, "o-", color="steelblue", lw=2, markersize=5,
            label="Empirical coverage")
    ax.fill_between(alphas, alphas - 0.05, alphas + 0.05,
                    color="gray", alpha=0.2, label="±5% band")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal α (predicted coverage)", fontsize=10)
    ax.set_ylabel("Empirical coverage", fontsize=10)
    ax.set_title(f"Reliability diagram  "
                 f"(mean |cov-α| = {calibpit['calib_error_l1']:.4f})",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Summary bar: KS per dim ---
    ax = fig.add_subplot(gs[n_rows_pit, 2:4])
    dim_colors = ["green" if p > 0.05 else ("orange" if p > 0.01 else "red")
                  for p in pv_pd]
    ax.bar(range(D), ks_pd, color=dim_colors, edgecolor="white")
    ax.set_title(f"KS statistic per dim  (mean={calibpit['ks_mean']:.4f})",
                 fontsize=11)
    ax.set_xlabel("dimension", fontsize=10)
    ax.set_ylabel("KS stat (lower = better)", fontsize=10)
    ax.set_xticks(range(D))
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # --- Bottom row: key numbers as big text ---
    ax = fig.add_subplot(gs[n_rows_pit + 1, :])
    ax.axis("off")
    n_pass = (pv_pd > 0.05).sum()
    summary = (
        f"Density Calibration Summary\n"
        f"─────────────────────────────────────────────────────────────────\n"
        f"  KS mean:       {calibpit['ks_mean']:.4f}     (ideal ≈ 0)\n"
        f"  Dims passing KS (p>0.05):   {n_pass}/{D}\n"
        f"  Reliability L1 error:       {calibpit['calib_error_l1']:.4f}     (ideal = 0)\n"
        f"  Sharpness (mean σ):         {calibpit['sharpness']:.4f}\n"
        f"  Two-sample MMD² (RBF):      {calibpit['mmd2']:.6f}     (ideal ≈ 0)"
    )
    ax.text(0.05, 0.5, summary, fontsize=12, family="monospace",
            verticalalignment="center")

    plt.suptitle("World Model — Density Calibration (PIT + Reliability + MMD)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Plotting — Autoregressive rollout
# ---------------------------------------------------------------------------

def plot_rollout(ro: dict, out_path: str) -> None:
    """
    Rollout diagnostic: norm stability, distribution matching,
    regime consistency per horizon step.
    Grid 2x2.
    """
    horizon = ro["horizon"]
    h_steps = np.arange(1, horizon + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # --- (0,0) Norm stability ---
    ax = axes[0, 0]
    ax.plot(h_steps, ro["norm_mean_per_step"], "b-", lw=2, label="mean ‖z‖")
    ax.fill_between(h_steps, ro["norm_p5_per_step"], ro["norm_p95_per_step"],
                    alpha=0.25, color="steelblue", label="p5-p95 band")
    ax.axhline(ro["real_norm_mean"], color="red", ls="--", lw=1.5,
               label=f"real mean ‖z‖ = {ro['real_norm_mean']:.2f}")
    ax.set_xlabel("Horizon step", fontsize=10)
    ax.set_ylabel("‖z‖₂ per step", fontsize=10)
    ax.set_title("Rollout norm stability", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Distribution deviation ---
    ax = axes[0, 1]
    ax.plot(h_steps, ro["mean_dev_per_step"], "o-", color="steelblue",
            markersize=3, label="||μ_dev||")
    ax.plot(h_steps, ro["std_dev_per_step"], "s-", color="salmon",
            markersize=3, label="||σ_dev||")
    ax.set_xlabel("Horizon step", fontsize=10)
    ax.set_ylabel("L2 deviation from real distribution", fontsize=10)
    ax.set_title("Distribution matching (lower = better)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Regime consistency per step ---
    ax = axes[1, 0]
    rc = ro["regime_consistency_per_step"]
    ax.plot(h_steps, rc * 100, "g-", lw=2, marker="o", markersize=3)
    ax.axhline(33.3, color="gray", ls=":", lw=1.0,
               label="chance (3-class)")
    ax.axhline(75.0, color="orange", ls="--", lw=1.0, alpha=0.6,
               label="good threshold = 75%")
    ax.set_xlabel("Horizon step", fontsize=10)
    ax.set_ylabel("Rollouts matching initial regime (%)", fontsize=10)
    ax.set_title("Regime consistency vs horizon", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,1) Per-regime breakdown at final horizon ---
    ax = axes[1, 1]
    regime_names = ["low_vol", "mid_vol", "high_vol"]
    vals = []
    labels = []
    colors = ["tab:blue", "tab:orange", "tab:red"]
    colors_used = []
    for r, name in enumerate(regime_names):
        if r in ro["consistency_per_regime_end"]:
            vals.append(ro["consistency_per_regime_end"][r] * 100)
            labels.append(name)
            colors_used.append(colors[r])
    bars = ax.bar(labels, vals, color=colors_used, edgecolor="white",
                  linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(33.3, color="gray", ls=":", lw=1.0)
    ax.axhline(75.0, color="orange", ls="--", lw=1.0, alpha=0.6)
    ax.set_ylabel("Consistency at final horizon (%)", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title(f"Per-regime consistency at h={horizon}", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"World Model — Autoregressive Rollout Diagnostic  "
                 f"(ctx={ro['ctx_len']}, horizon={horizon}, "
                 f"n_rollouts={ro['n_rollouts']})",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Rollout plot saved: {out_path}")


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

    # --- Pannello E: Density calibration ---
    if not args.skip_calibration:
        print("\nComputing density calibration (Pannello E)...")
        calibpit = calibration_pit_analysis(metrics, sequences, regimes,
                                            n_mmd_samples=args.mmd_samples,
                                            seed=args.seed)
        print_calibration(calibpit)
        plot_calibration(calibpit, out_path=args.out_calibration)

    # --- Pannello F: Autoregressive rollout ---
    if not args.skip_rollout:
        print("\nComputing autoregressive rollout (Pannello F)...")
        ro = rollout_analysis(
            model, sequences, regimes, device,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            ctx_len=args.ctx_len,
            seed=args.seed,
        )
        print_rollout(ro)
        plot_rollout(ro, out_path=args.out_rollout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate World Model (Pannelli A-F)")
    parser.add_argument("--wm_ckpt",    type=str,  default="checkpoints/wm_best.pt")
    parser.add_argument("--dataset",    type=str,  default="data/wm_dataset.npz")
    parser.add_argument("--n_samples",  type=int,  default=10_000)
    parser.add_argument("--out",             type=str, default="validation/world_model/eval_wm.png",
                        help="Main evaluation figure (Pannelli A-D)")
    parser.add_argument("--out_calibration", type=str, default="validation/world_model/eval_wm_calibration.png",
                        help="Density calibration figure (Pannello E)")
    parser.add_argument("--out_rollout",     type=str, default="validation/world_model/eval_wm_rollout.png",
                        help="Autoregressive rollout figure (Pannello F)")
    parser.add_argument("--n_rollouts", type=int, default=500,
                        help="Number of trajectories for rollout test")
    parser.add_argument("--horizon",    type=int, default=50,
                        help="Rollout horizon in steps")
    parser.add_argument("--ctx_len",    type=int, default=20,
                        help="Initial context length for rollout")
    parser.add_argument("--mmd_samples", type=int, default=1000,
                        help="Number of samples for MMD computation")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip Pannello E (density calibration)")
    parser.add_argument("--skip_rollout",     action="store_true",
                        help="Skip Pannello F (autoregressive rollout)")
    args = parser.parse_args()
    main(args)
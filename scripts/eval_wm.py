"""
eval_wm.py — Valutazione completa del World Model (Modulo B).

Focus: qualità della predizione ONE-STEP, coerente con il Modulo C (DRO one-step).
Il world model serve a fornire P(z_{t+1} | h_t) come GMM; il DRO perturba
quella distribuzione in un intorno Wasserstein. Quindi ciò che conta è:

  1. ONE-STEP PREDICTION  — NLL, MSE, MAE tra GMM mean e z_t+1 reale,
                             per-dim MSE, breakdown per regime
  2. CALIBRAZIONE GMM     — predicted σ vs actual |error|:
                             se σ è calibrata, il raggio ε del DRO ha senso fisico
  3. ANALISI COMPONENTI   — entropia π, specializzazione per regime,
                             component usage (mode collapse check)
  4. POSITION ANALYSIS    — NLL per posizione nella sequenza:
                             il modello migliora con più contesto?

Output: eval_wm.png + stampe a terminale.

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
# Section 1 — One-step prediction
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
    1-step prediction quality using teacher forcing on all positions.
    """
    M, Np1, D = sequences.shape
    N = Np1 - 1

    all_mse = []
    all_mae = []
    all_nll = []
    all_pi  = []
    all_log_sig = []
    all_errors  = []
    all_pos_nll = []
    regime_nll = {0: [], 1: [], 2: []}

    for i in range(0, M, batch_size):
        z_seq = sequences[i:i+batch_size].to(device)
        a_seq = actions[i:i+batch_size].to(device)
        reg   = regimes[i:i+batch_size]

        pi, mu, log_sig = model(z_seq, a_seq)
        z_next = z_seq[:, 1:, :]

        # Weighted mean prediction
        z_pred = (pi.unsqueeze(-1) * mu).sum(dim=2)

        err = z_pred - z_next
        mse = (err ** 2).mean(dim=-1)
        mae = err.abs().mean(dim=-1)

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

        all_mse.append(mse.cpu())
        all_mae.append(mae.cpu())
        all_nll.append(nll.item())
        all_pi.append(pi.cpu())
        all_log_sig.append(log_sig.cpu())
        all_errors.append(err.cpu())
        all_pos_nll.append(pos_nll.cpu())

        for r in [0, 1, 2]:
            mask = (reg == r)
            if mask.sum() > 0:
                nll_r = model.nll_loss(
                    pi[mask], mu[mask], log_sig[mask], z_next[mask]
                ).item()
                regime_nll[r].append(nll_r)

    mse_all   = torch.cat(all_mse).numpy()
    mae_all   = torch.cat(all_mae).numpy()
    pi_all    = torch.cat(all_pi).numpy()
    errors    = torch.cat(all_errors).numpy()
    log_sig_c = torch.cat(all_log_sig).numpy()
    pos_nll   = torch.cat(all_pos_nll).numpy()

    per_dim_mse = (errors ** 2).mean(axis=(0, 1))
    ent = -(pi_all * np.log(pi_all + 1e-8)).sum(axis=-1)

    return {
        "mse_mean":     float(mse_all.mean()),
        "mse_std":      float(mse_all.std()),
        "mae_mean":     float(mae_all.mean()),
        "nll_mean":     float(np.mean(all_nll)),
        "per_dim_mse":  per_dim_mse,
        "entropy_pi":   ent,
        "pi_all":       pi_all,
        "errors":       errors,
        "log_sig_cat":  log_sig_c,
        "regime_nll":   {r: np.mean(v) for r, v in regime_nll.items() if v},
        "pos_nll_mean": pos_nll.mean(axis=0),
    }


# ---------------------------------------------------------------------------
# Section 2 — GMM Calibration
# ---------------------------------------------------------------------------

def calibration_analysis(metrics: dict) -> dict:
    """
    Compare predicted σ with actual errors.
    Well-calibrated: |error| ≈ σ on average per dimension.
    Critical for DRO: if σ is miscalibrated, ε has wrong scale.
    """
    errors  = metrics["errors"]
    log_sig = metrics["log_sig_cat"]
    pi      = metrics["pi_all"]

    sig = np.exp(log_sig)
    sig_avg = (pi[..., np.newaxis] * sig).sum(axis=2)

    abs_err_per_dim = np.abs(errors).mean(axis=(0, 1))
    sig_per_dim     = sig_avg.mean(axis=(0, 1))

    calib_ratio_per_dim = sig_per_dim / (abs_err_per_dim + 1e-8)

    # Binned calibration
    sig_flat = sig_avg.reshape(-1)
    err_flat = np.abs(errors).reshape(-1)
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
        "abs_err_per_dim":  abs_err_per_dim,
        "sig_per_dim":      sig_per_dim,
        "calib_ratio":      calib_ratio_per_dim,
        "calib_ratio_mean": float(calib_ratio_per_dim.mean()),
        "bin_sig":          np.array(bin_sig),
        "bin_err":          np.array(bin_err),
    }


# ---------------------------------------------------------------------------
# Section 3 — Component analysis
# ---------------------------------------------------------------------------

def component_analysis(metrics: dict, regimes: torch.Tensor) -> dict:
    pi = metrics["pi_all"]
    M, N, K = pi.shape
    reg = regimes.numpy()

    pi_per_regime = {}
    for r in [0, 1, 2]:
        mask = (reg == r)
        if mask.sum() > 0:
            pi_per_regime[r] = pi[mask].mean(axis=(0, 1))

    assignments = pi.reshape(-1, K).argmax(axis=-1)
    usage_counts = np.bincount(assignments, minlength=K)
    usage_frac   = usage_counts / usage_counts.sum()

    ent = metrics["entropy_pi"]
    max_ent = np.log(K)

    return {
        "pi_per_regime":  pi_per_regime,
        "usage_frac":     usage_frac,
        "entropy_mean":   float(ent.mean()),
        "entropy_std":    float(ent.std()),
        "max_entropy":    float(max_ent),
        "K":              K,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_all(metrics: dict, calib: dict, comp: dict) -> None:
    names = ["low_vol", "mid_vol", "high_vol"]

    print("\n" + "=" * 70)
    print("SECTION 1 — ONE-STEP PREDICTION (teacher forcing)")
    print("=" * 70)
    print(f"  NLL (mean)       : {metrics['nll_mean']:.4f}")
    print(f"  MSE (mean ± std) : {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"  MAE (mean)       : {metrics['mae_mean']:.6f}")
    print(f"\n  NLL per regime:")
    for r, name in enumerate(names):
        if r in metrics["regime_nll"]:
            print(f"    {name:10s}: {metrics['regime_nll'][r]:.4f}")

    print(f"\n  Per-dim MSE (top-5 highest / lowest):")
    pdm = metrics["per_dim_mse"]
    ranked = np.argsort(pdm)
    print(f"    highest: ", end="")
    for i in ranked[-5:][::-1]:
        print(f"d{i}={pdm[i]:.6f}  ", end="")
    print(f"\n    lowest:  ", end="")
    for i in ranked[:5]:
        print(f"d{i}={pdm[i]:.6f}  ", end="")
    print()

    print("\n" + "=" * 70)
    print("SECTION 2 — GMM CALIBRATION (σ vs |error|)")
    print("=" * 70)
    print(f"  Mean calibration ratio σ/|err| : {calib['calib_ratio_mean']:.3f}  (ideal ≈ 1.0)")
    print(f"\n  Per-dim calibration (first 8 dims):")
    for i in range(min(8, len(calib["calib_ratio"]))):
        print(f"    dim {i:2d}: σ={calib['sig_per_dim'][i]:.4f}  "
              f"|err|={calib['abs_err_per_dim'][i]:.4f}  "
              f"ratio={calib['calib_ratio'][i]:.3f}")

    print("\n" + "=" * 70)
    print("SECTION 3 — COMPONENT ANALYSIS")
    print("=" * 70)
    print(f"  K = {comp['K']} components")
    print(f"  Entropy π: {comp['entropy_mean']:.3f} ± {comp['entropy_std']:.3f}  "
          f"(max = ln({comp['K']}) = {comp['max_entropy']:.3f})")
    print(f"\n  Component usage (argmax assignment):")
    for k in range(comp["K"]):
        print(f"    comp {k}: {comp['usage_frac'][k]*100:.1f}%")
    print(f"\n  Mean π per regime:")
    header = "           " + "  ".join(f"comp_{k}" for k in range(comp["K"]))
    print(header)
    for r, name in enumerate(names):
        if r in comp["pi_per_regime"]:
            vals = "  ".join(f"{v:.4f}" for v in comp["pi_per_regime"][r])
            print(f"    {name:10s} {vals}")

    print("\n" + "=" * 70)
    print("SECTION 4 — POSITION ANALYSIS (NLL vs context length)")
    print("=" * 70)
    pnll = metrics["pos_nll_mean"]
    print(f"  NLL at position  1 (min context): {pnll[0]:.4f}")
    print(f"  NLL at position  5              : {pnll[min(4, len(pnll)-1)]:.4f}")
    print(f"  NLL at position 10              : {pnll[min(9, len(pnll)-1)]:.4f}")
    print(f"  NLL at position {len(pnll)} (max context): {pnll[-1]:.4f}")
    improvement = (pnll[0] - pnll[-1]) / abs(pnll[0]) * 100
    print(f"  Improvement first→last: {improvement:.1f}%")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(metrics: dict, calib: dict, comp: dict, out_path: str) -> None:
    names_regime = ["low_vol", "mid_vol", "high_vol"]
    K = comp["K"]
    D = len(metrics["per_dim_mse"])

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.40)

    # --- 1. Per-dim MSE ---
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(D), metrics["per_dim_mse"], color="steelblue")
    ax.set_title("MSE per dimensione latente")
    ax.set_xlabel("dim")
    ax.set_ylabel("MSE")

    # --- 2. NLL per regime ---
    ax = fig.add_subplot(gs[0, 1])
    reg_names, reg_vals = [], []
    for r, name in enumerate(names_regime):
        if r in metrics["regime_nll"]:
            reg_names.append(name)
            reg_vals.append(metrics["regime_nll"][r])
    colors_r = ["tab:blue", "tab:orange", "tab:red"][:len(reg_names)]
    ax.bar(reg_names, reg_vals, color=colors_r)
    ax.set_title("NLL per regime")
    ax.set_ylabel("NLL")
    for i, v in enumerate(reg_vals):
        ax.text(i, v - 0.5, f"{v:.2f}", ha="center", va="top",
                fontsize=9, color="white", fontweight="bold")

    # --- 3. NLL vs position ---
    ax = fig.add_subplot(gs[0, 2])
    pnll = metrics["pos_nll_mean"]
    ax.plot(range(1, len(pnll) + 1), pnll, "b-o", markersize=3)
    ax.set_title("NLL vs posizione (effetto contesto)")
    ax.set_xlabel("posizione (1 = min contesto)")
    ax.set_ylabel("NLL")
    ax.grid(True, alpha=0.3)

    # --- 4. Entropy π distribution ---
    ax = fig.add_subplot(gs[0, 3])
    ent = metrics["entropy_pi"].reshape(-1)
    ax.hist(ent, bins=50, color="teal", alpha=0.7, density=True)
    ax.axvline(comp["max_entropy"], color="red", ls="--",
               label=f"max = ln({K}) = {comp['max_entropy']:.2f}")
    ax.axvline(comp["entropy_mean"], color="orange", ls="--",
               label=f"mean = {comp['entropy_mean']:.2f}")
    ax.set_title("Distribuzione entropia π")
    ax.set_xlabel("H(π)")
    ax.legend(fontsize=7)

    # --- 5. Calibration: σ vs |error| per dim ---
    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(D)
    w = 0.35
    ax.bar(x - w/2, calib["sig_per_dim"], w, label="predicted σ", color="steelblue")
    ax.bar(x + w/2, calib["abs_err_per_dim"], w, label="actual |error|", color="salmon")
    ax.set_title("Calibrazione GMM: σ predetta vs |errore| reale")
    ax.set_xlabel("dimensione latente")
    ax.set_ylabel("valore medio")
    ax.legend()
    ax.set_xticks(x)

    # --- 6. Calibration: binned ---
    ax = fig.add_subplot(gs[1, 2])
    if len(calib["bin_sig"]) > 0:
        ax.scatter(calib["bin_sig"], calib["bin_err"], c="steelblue", s=40, zorder=3)
        lims = [0, max(calib["bin_sig"].max(), calib["bin_err"].max()) * 1.1]
        ax.plot(lims, lims, "r--", alpha=0.5, label="perfect calibration")
        ax.set_xlabel("predicted σ (binned)")
        ax.set_ylabel("actual |error| (binned)")
        ax.set_title("Calibrazione binned")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- 7. Component usage ---
    ax = fig.add_subplot(gs[1, 3])
    ax.bar(range(K), comp["usage_frac"] * 100, color="mediumpurple")
    ax.set_title("Uso componenti (argmax)")
    ax.set_xlabel("componente k")
    ax.set_ylabel("% assegnazioni")
    ax.set_xticks(range(K))
    for k in range(K):
        ax.text(k, comp["usage_frac"][k] * 100 + 0.5,
                f"{comp['usage_frac'][k]*100:.1f}%", ha="center", fontsize=8)

    # --- 8. π per regime heatmap ---
    ax = fig.add_subplot(gs[2, :2])
    pi_matrix = np.zeros((3, K))
    for r in range(3):
        if r in comp["pi_per_regime"]:
            pi_matrix[r] = comp["pi_per_regime"][r]
    im = ax.imshow(pi_matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"comp_{k}" for k in range(K)])
    ax.set_yticks(range(3))
    ax.set_yticklabels(names_regime)
    ax.set_title("Media π per regime e componente")
    plt.colorbar(im, ax=ax)
    for r in range(3):
        for k in range(K):
            ax.text(k, r, f"{pi_matrix[r, k]:.3f}",
                    ha="center", va="center", fontsize=9)

    # --- 9. Calibration ratio per dim ---
    ax = fig.add_subplot(gs[2, 2:])
    ax.bar(range(D), calib["calib_ratio"], color="seagreen", alpha=0.7)
    ax.axhline(1.0, color="red", ls="--", label="perfect (ratio=1.0)")
    ax.set_title("Rapporto σ / |errore| per dimensione")
    ax.set_xlabel("dimensione latente")
    ax.set_ylabel("σ / |error|")
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, D, max(1, D // 8)))

    plt.suptitle("World Model — Valutazione Completa (one-step)",
                 fontsize=14, y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot salvato in: {out_path}")


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

    print("Computing calibration analysis...")
    calib = calibration_analysis(metrics)

    print("Computing component analysis...")
    comp = component_analysis(metrics, regimes)

    print_all(metrics, calib, comp)
    plot_all(metrics, calib, comp, out_path=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate World Model (one-step)")
    parser.add_argument("--wm_ckpt",    type=str,  default="checkpoints/wm_best.pt")
    parser.add_argument("--dataset",    type=str,  default="data/wm_dataset.npz")
    parser.add_argument("--n_samples",  type=int,  default=10_000)
    parser.add_argument("--out",        type=str,  default="eval_wm.png")
    args = parser.parse_args()
    main(args)
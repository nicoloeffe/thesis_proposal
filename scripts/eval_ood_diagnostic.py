"""
eval_ood_diagnostic.py — Diagnostica OOD degli stati avversariali x*.

Misura quanto gli x* prodotti dal DRO inner solver escono dal supporto
della distribuzione di z vista durante il training del critic.

Per ogni ε:
  1. Seleziona traiettorie dal dataset
  2. Per ogni timestep, risolve il DRO inner problem → x*
  3. Confronta x* con la distribuzione di training di z:
     - % dimensioni fuori [μ-2σ, μ+2σ] per-dim
     - % dimensioni fuori [μ-3σ, μ+3σ] per-dim
     - Distanza di Mahalanobis media
     - Max |x*_d - μ_d| / σ_d  (worst-case per-dim deviation)
  4. Plot diagnostico

Output: eval_ood_diagnostic.png

Uso:
  python scripts/eval_ood_diagnostic.py
  python scripts/eval_ood_diagnostic.py --trust_radius 0.2 --n_traj 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.world_model import LOBWorldModel, WorldModelConfig
from models.critic import ValueNetwork
from models.dro import WassersteinDRO, DROConfig


# ---------------------------------------------------------------------------
# Loaders (same as stress_test.py)
# ---------------------------------------------------------------------------

def load_world_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = WorldModelConfig()
    for k, v in ckpt["cfg"].items():
        setattr(cfg, k, v)
    model = LOBWorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_critic(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    critic = ValueNetwork(
        d_latent=cfg["d_latent"], hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
    ).to(device)
    critic.load_state_dict(ckpt["model"])
    critic.eval()
    return critic


# ---------------------------------------------------------------------------
# Compute z training distribution stats
# ---------------------------------------------------------------------------

def compute_z_stats(sequences: torch.Tensor) -> dict:
    """
    Compute per-dimension stats of z from the training sequences.

    Args:
        sequences: (M, N+1, D) — all latent sequences from wm_dataset
    Returns:
        dict with mean (D,), std (D,), q01 (D,), q99 (D,), cov (D, D)
    """
    # Flatten all z vectors: (M*(N+1), D)
    M, Np1, D = sequences.shape
    z_all = sequences.reshape(-1, D)

    stats = {
        "mean": z_all.mean(dim=0),          # (D,)
        "std":  z_all.std(dim=0),           # (D,)
        "min":  z_all.min(dim=0).values,    # (D,)
        "max":  z_all.max(dim=0).values,    # (D,)
        "q01":  z_all.quantile(0.01, dim=0),  # (D,)
        "q99":  z_all.quantile(0.99, dim=0),  # (D,)
        "N":    z_all.shape[0],
        "D":    D,
    }

    # Covariance for Mahalanobis
    z_c = z_all - stats["mean"]
    cov = (z_c.T @ z_c) / (z_all.shape[0] - 1)
    # Regularise for invertibility
    cov += 1e-6 * torch.eye(D)
    stats["cov_inv"] = torch.linalg.inv(cov)

    print(f"z training stats: {z_all.shape[0]:,} vectors, D={D}")
    print(f"  mean norm: {stats['mean'].norm():.4f}")
    print(f"  std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")

    return stats


# ---------------------------------------------------------------------------
# Measure OOD for a set of adversarial x*
# ---------------------------------------------------------------------------

def measure_ood(
    x_star: torch.Tensor,     # (N_points, D)
    y_nominal: torch.Tensor,  # (N_points, D)
    z_stats: dict,
) -> dict:
    """
    Compute OOD metrics for adversarial states x* relative to training z.

    Returns dict with:
      - frac_outside_2sigma: fraction of (point, dim) pairs outside ±2σ
      - frac_outside_3sigma: fraction of (point, dim) pairs outside ±3σ
      - frac_points_any_ood: fraction of points with at least 1 dim outside ±3σ
      - mahalanobis_mean/std: Mahalanobis distance from training mean
      - max_zscore: worst-case |x*_d - μ_d| / σ_d across all dims
      - transport_per_dim: mean |x* - y| per dimension
      - per_dim_ood_rate: (D,) — fraction of points OOD per dimension
    """
    mu = z_stats["mean"].to(x_star.device)
    sigma = z_stats["std"].to(x_star.device)
    cov_inv = z_stats["cov_inv"].to(x_star.device)

    N, D = x_star.shape

    # Z-scores: |x*_d - μ_d| / σ_d
    zscores = ((x_star - mu) / (sigma + 1e-8)).abs()  # (N, D)

    outside_2s = (zscores > 2.0).float()
    outside_3s = (zscores > 3.0).float()

    # Mahalanobis: sqrt((x - μ)^T Σ^{-1} (x - μ))
    diff = x_star - mu  # (N, D)
    mahal = torch.sqrt((diff @ cov_inv * diff).sum(dim=-1))  # (N,)

    # Transport per dim
    transport = (x_star - y_nominal).abs()  # (N, D)

    # Also compute z-scores of the NOMINAL points for comparison
    zscores_nom = ((y_nominal - mu) / (sigma + 1e-8)).abs()

    return {
        "frac_outside_2sigma": outside_2s.mean().item(),
        "frac_outside_3sigma": outside_3s.mean().item(),
        "frac_points_any_ood_3s": (outside_3s.sum(dim=-1) > 0).float().mean().item(),
        "mahalanobis_mean": mahal.mean().item(),
        "mahalanobis_std": mahal.std().item(),
        "mahalanobis_max": mahal.max().item(),
        "max_zscore": zscores.max().item(),
        "mean_zscore": zscores.mean().item(),
        "transport_per_dim": transport.mean(dim=0).cpu(),   # (D,)
        "per_dim_ood_rate_3s": outside_3s.mean(dim=0).cpu(),  # (D,)
        # Nominal comparison
        "nominal_frac_outside_3sigma": (zscores_nom > 3.0).float().mean().item(),
        "nominal_mahalanobis_mean": torch.sqrt(
            ((y_nominal - mu) @ cov_inv * (y_nominal - mu)).sum(dim=-1)
        ).mean().item(),
    }


# ---------------------------------------------------------------------------
# Run OOD diagnostic across epsilons
# ---------------------------------------------------------------------------

def run_ood_diagnostic(
    world_model: LOBWorldModel,
    critic: ValueNetwork,
    sequences: torch.Tensor,    # (M, N+1, D) on device
    actions: torch.Tensor,      # (M, N, 3) on device
    rewards: torch.Tensor,      # (M, N) on device
    z_stats: dict,
    epsilons: list[float],
    dro_cfg: DROConfig,
    n_traj: int = 30,
    n_timesteps: int = 10,      # sample timesteps per trajectory
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    For each epsilon, run DRO on sampled (trajectory, timestep) pairs
    and collect OOD metrics.
    """
    M = sequences.shape[0]
    device = sequences.device
    rng = torch.Generator().manual_seed(seed)

    traj_idx = torch.randperm(M, generator=rng)[:n_traj]
    N = actions.shape[1]
    time_idx = torch.randperm(N, generator=rng)[:min(n_timesteps, N)]

    dro = WassersteinDRO(critic, dro_cfg)

    results = {}

    for eps in epsilons:
        if verbose:
            print(f"\n  ε = {eps:.4f}  ", end="", flush=True)

        all_x_star = []
        all_y_nominal = []

        for i, ti in enumerate(traj_idx):
            z_seq = sequences[ti:ti+1]
            a_seq = actions[ti:ti+1]

            with torch.no_grad():
                pi_full, mu_full, log_sig_full = world_model(z_seq, a_seq)

            for t in time_idx:
                t_val = t.item()
                if t_val >= N:
                    continue

                pi_t = pi_full[0, t_val, :]
                mu_t = mu_full[0, t_val, :, :]
                log_sig_t = log_sig_full[0, t_val, :, :]

                with torch.enable_grad():
                    result = dro.solve_one_step(pi_t, mu_t, log_sig_t, eps)

                all_x_star.append(result["x_star"].cpu())
                all_y_nominal.append(mu_t.cpu())

            if verbose and (i + 1) % 10 == 0:
                print(f"[{i+1}/{n_traj}]", end=" ", flush=True)

        x_star_all = torch.cat(all_x_star, dim=0)      # (N_total, D)
        y_nom_all = torch.cat(all_y_nominal, dim=0)    # (N_total, D) — K centroids per step

        ood = measure_ood(x_star_all, y_nom_all, z_stats)
        ood["n_points"] = x_star_all.shape[0]
        results[eps] = ood

        if verbose:
            print(f"  n={ood['n_points']}  "
                  f"OOD_3σ={ood['frac_outside_3sigma']*100:.1f}%  "
                  f"Mahal={ood['mahalanobis_mean']:.2f}±{ood['mahalanobis_std']:.2f}  "
                  f"max_z={ood['max_zscore']:.2f}")

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_diagnostic(results: dict, epsilons: list[float], out_path: str,
                    D: int) -> None:
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    eps_arr = np.array(epsilons)

    ood_3s = np.array([results[e]["frac_outside_3sigma"] * 100 for e in epsilons])
    ood_2s = np.array([results[e]["frac_outside_2sigma"] * 100 for e in epsilons])
    pts_ood = np.array([results[e]["frac_points_any_ood_3s"] * 100 for e in epsilons])
    mahal = np.array([results[e]["mahalanobis_mean"] for e in epsilons])
    mahal_s = np.array([results[e]["mahalanobis_std"] for e in epsilons])
    max_z = np.array([results[e]["max_zscore"] for e in epsilons])
    nom_3s = np.array([results[e]["nominal_frac_outside_3sigma"] * 100 for e in epsilons])

    # --- 1. OOD fraction vs ε ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(eps_arr, ood_3s, "r-o", label="x* outside ±3σ (%)", linewidth=2)
    ax.plot(eps_arr, ood_2s, "orange", marker="s", ls="--", label="x* outside ±2σ (%)")
    ax.plot(eps_arr, nom_3s, "g--", marker="^", alpha=0.6, label="nominal y outside ±3σ (%)")
    ax.set_xlabel("ε")
    ax.set_ylabel("% (dim, point) pairs")
    ax.set_title("Tasso OOD vs ε")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2. % of POINTS with any OOD dim ---
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(len(epsilons)), pts_ood, color="salmon", alpha=0.8)
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"{e:.2f}" for e in epsilons], rotation=45, fontsize=8)
    ax.set_xlabel("ε")
    ax.set_ylabel("% punti")
    ax.set_title("Punti con almeno 1 dim fuori ±3σ")
    for i, v in enumerate(pts_ood):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=7)

    # --- 3. Mahalanobis distance ---
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(eps_arr, mahal, "b-o", linewidth=2, label="x* (adversarial)")
    ax.fill_between(eps_arr, mahal - mahal_s, mahal + mahal_s, alpha=0.2)
    nom_mahal = np.array([results[e]["nominal_mahalanobis_mean"] for e in epsilons])
    ax.plot(eps_arr, nom_mahal, "g--^", alpha=0.6, label="y (nominal)")
    # Expected Mahalanobis for D-dim Gaussian: ~sqrt(D)
    ax.axhline(np.sqrt(D), color="gray", ls=":", alpha=0.5,
               label=f"E[Mahal] Gaussian ≈ √{D} = {np.sqrt(D):.1f}")
    ax.set_xlabel("ε")
    ax.set_ylabel("Mahalanobis distance")
    ax.set_title("Distanza di Mahalanobis da μ_train")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 4. Max z-score per ε ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(eps_arr, max_z, "r-s", linewidth=2)
    ax.axhline(3.0, color="orange", ls="--", alpha=0.5, label="3σ threshold")
    ax.axhline(5.0, color="red", ls="--", alpha=0.5, label="5σ (extreme OOD)")
    ax.set_xlabel("ε")
    ax.set_ylabel("max |x*_d - μ_d| / σ_d")
    ax.set_title("Worst-case z-score (max across all dims & points)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 5. Per-dim OOD rate for a mid-range ε ---
    mid_eps = epsilons[len(epsilons) // 2]
    ax = fig.add_subplot(gs[1, 1])
    per_dim = results[mid_eps]["per_dim_ood_rate_3s"].numpy() * 100
    ax.bar(range(D), per_dim, color="steelblue", alpha=0.8)
    ax.set_xlabel("dimensione latente")
    ax.set_ylabel("% OOD (>3σ)")
    ax.set_title(f"Tasso OOD per dimensione (ε={mid_eps:.2f})")
    ax.grid(True, alpha=0.3, axis="y")

    # --- 6. Transport per dim for same ε ---
    ax = fig.add_subplot(gs[1, 2])
    transport = results[mid_eps]["transport_per_dim"].numpy()
    ax.bar(range(D), transport, color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("dimensione latente")
    ax.set_ylabel("|x* - y| medio")
    ax.set_title(f"Trasporto medio per dim (ε={mid_eps:.2f})")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Diagnostica OOD — Stati Avversariali x* vs Distribuzione Training z",
                 fontsize=14, y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot salvato in: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict, epsilons: list[float]) -> None:
    print("\n" + "=" * 85)
    print("DIAGNOSTICA OOD — RIEPILOGO")
    print("=" * 85)
    print(f"  {'ε':>6s}  {'n_pts':>6s}  {'OOD_2σ%':>8s}  {'OOD_3σ%':>8s}  "
          f"{'pts_any':>7s}  {'Mahal':>8s}  {'max_z':>6s}  "
          f"{'nom_3σ%':>8s}")
    print("-" * 85)
    for eps in epsilons:
        r = results[eps]
        print(f"  {eps:>6.3f}  {r['n_points']:>6d}  "
              f"{r['frac_outside_2sigma']*100:>7.1f}%  "
              f"{r['frac_outside_3sigma']*100:>7.1f}%  "
              f"{r['frac_points_any_ood_3s']*100:>6.1f}%  "
              f"{r['mahalanobis_mean']:>7.2f}  "
              f"{r['max_zscore']:>6.2f}  "
              f"{r['nominal_frac_outside_3sigma']*100:>7.1f}%")
    print("=" * 85)

    # Interpretation
    worst_eps = epsilons[-1]
    worst = results[worst_eps]
    print("\nInterpretazione:")
    if worst["frac_outside_3sigma"] > 0.10:
        print(f"  ⚠  A ε={worst_eps:.2f}, {worst['frac_outside_3sigma']*100:.0f}% delle "
              f"coppie (punto, dim) sono fuori ±3σ.")
        print(f"     Il critic viene valutato in regioni dove non è stato allenato.")
        print(f"     → Ridurre trust_radius o usare un critic Lipschitz-bounded.")
    elif worst["frac_outside_3sigma"] > 0.02:
        print(f"  ⚡  OOD moderato ({worst['frac_outside_3sigma']*100:.1f}% > 3σ). "
              f"Risultati plausibili ma da validare.")
    else:
        print(f"  ✓  OOD minimo ({worst['frac_outside_3sigma']*100:.1f}% > 3σ). "
              f"Gli x* restano nel supporto di z. Risultati affidabili.")

    if worst["max_zscore"] > 5.0:
        print(f"  ⚠  Max z-score = {worst['max_zscore']:.1f}σ — "
              f"almeno un punto è in zona di estrapolazione estrema.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load
    world_model = load_world_model(args.wm_ckpt, device)
    critic = load_critic(args.critic_ckpt, device)

    data = np.load(args.dataset)
    sequences = torch.from_numpy(data["sequences"]).to(device)
    actions = torch.from_numpy(data["actions"]).to(device)
    rewards = torch.from_numpy(data["rewards"]).to(device)
    M, Np1, D = sequences.shape
    print(f"Dataset: {M:,} sequences, N={Np1-1}, D={D}")

    # Compute z stats from ALL training data
    z_stats = compute_z_stats(sequences.cpu())

    # DRO config
    dro_cfg = DROConfig(
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_steps=args.outer_steps,
        lambda_init=args.lambda_init,
        trust_radius=args.trust_radius,
        gamma=args.gamma,
    )
    print(f"\nDRO config: inner_steps={dro_cfg.inner_steps}  "
          f"trust_radius={dro_cfg.trust_radius}")

    epsilons = [float(e) for e in args.epsilons.split(",")]
    print(f"Epsilons: {epsilons}")

    # Run
    t0 = time.time()
    results = run_ood_diagnostic(
        world_model, critic, sequences, actions, rewards,
        z_stats, epsilons, dro_cfg,
        n_traj=args.n_traj,
        n_timesteps=args.n_timesteps,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nTempo totale: {elapsed:.1f}s")

    # Output
    print_summary(results, epsilons)
    plot_diagnostic(results, epsilons, args.out, D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OOD Diagnostic for DRO Adversarial States")
    parser.add_argument("--wm_ckpt",      type=str, default="checkpoints/wm_best.pt")
    parser.add_argument("--critic_ckpt",  type=str, default="checkpoints/critic_best.pt")
    parser.add_argument("--dataset",      type=str, default="data/wm_dataset.npz")
    parser.add_argument("--epsilons",     type=str,
                        default="0.0,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--n_traj",       type=int, default=30)
    parser.add_argument("--n_timesteps",  type=int, default=10,
                        help="Timesteps to sample per trajectory")
    parser.add_argument("--inner_steps",  type=int, default=100)
    parser.add_argument("--inner_lr",     type=float, default=0.05)
    parser.add_argument("--outer_steps",  type=int, default=30)
    parser.add_argument("--lambda_init",  type=float, default=50.0)
    parser.add_argument("--trust_radius", type=float, default=0.2,
                        help="Must match stress_test.py and backward DP")
    parser.add_argument("--gamma",        type=float, default=0.95)
    parser.add_argument("--out",          type=str, default="eval_ood_diagnostic.png")
    args = parser.parse_args()
    main(args)

"""
stress_test.py — Fase 5: Stress testing distribuzionale.

Produce la curva V_rob(ε) — l'output principale del framework.

Per ogni ε nell'insieme {0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}:
  1. Seleziona n traiettorie nominali dal dataset
  2. Per ogni traiettoria e ogni timestep t:
     - Il world model produce la GMM P(z_{t+1} | h_t)
     - Il Modulo C risolve il duale Wasserstein e trova x*_worst
     - Calcola il backup robusto y_rob = r_t + γ * V(x*_worst)
  3. Accumula V_rob(ε) = media delle somme scontate

Output: stress_test.png con la curva V_rob(ε) + diagnostiche.

Uso:
  python scripts/stress_test.py
  python scripts/stress_test.py --wm_ckpt checkpoints/wm_best.pt \
                                 --critic_ckpt checkpoints/critic_best.pt \
                                 --dataset data/wm_dataset.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from models.world_model import LOBWorldModel, WorldModelConfig
from models.critic import ValueNetwork
from models.dro import WassersteinDRO, StressTestRunner, DROConfig


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_world_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = WorldModelConfig()
    for k, v in ckpt["cfg"].items():
        setattr(cfg, k, v)
    model = LOBWorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"World model loaded: {path}  (epoch={ckpt['epoch']})")
    return model


def load_critic(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    critic = ValueNetwork(
        d_latent=cfg["d_latent"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
    ).to(device)
    critic.load_state_dict(ckpt["model"])
    critic.eval()
    # NOTE: do NOT freeze critic gradients — DRO inner loop needs
    # torch.autograd.grad(V(x), x) which requires the critic's
    # forward pass to build a graph w.r.t. input x.
    print(f"Critic loaded: {path}  (epoch={ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.6f})")
    return critic


def load_dataset(path: str, device: torch.device):
    data = np.load(path)
    sequences = torch.from_numpy(data["sequences"]).to(device)
    actions   = torch.from_numpy(data["actions"]).to(device)
    rewards   = torch.from_numpy(data["rewards"]).to(device)
    regimes   = torch.from_numpy(data["regimes"].astype(np.int64))
    M, Np1, D = sequences.shape
    print(f"Dataset: {M:,} sequences, N={Np1-1}, d_latent={D}")
    return sequences, actions, rewards, regimes


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    results: dict,
    epsilons: list[float],
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Collect data
    eps_arr   = np.array(epsilons)
    v_rob     = np.array([results[e]["v_rob_mean"] for e in epsilons])
    v_rob_std = np.array([results[e]["v_rob_std"] for e in epsilons])
    v_nom     = results[epsilons[0]]["v_nominal_mean"]  # same for all ε
    degrad    = np.array([results[e]["degradation"] * 100 for e in epsilons])
    lambdas   = np.array([results[e]["mean_lambda"] for e in epsilons])
    transport = np.array([results[e]["mean_transport"] for e in epsilons])

    # --- 1. Main plot: V_rob(ε) curve ---
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(eps_arr, v_rob, "b-o", markersize=6, linewidth=2, label="$V^{rob}(\\varepsilon)$")
    ax.fill_between(eps_arr, v_rob - v_rob_std, v_rob + v_rob_std,
                    alpha=0.2, color="blue")
    ax.axhline(v_nom, color="green", ls="--", linewidth=1.5,
               label=f"$V^{{nom}}$ = {v_nom:.4f}")
    ax.set_xlabel("ε (Wasserstein radius)", fontsize=12)
    ax.set_ylabel("$V^{rob}(\\varepsilon)$", fontsize=12)
    ax.set_title("Curva di valore robusto $V^{rob}(\\varepsilon)$", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- 2. Degradation % ---
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(range(len(epsilons)), degrad, color="salmon", alpha=0.8)
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"{e:.2f}" for e in epsilons], rotation=45, fontsize=8)
    ax.set_xlabel("ε")
    ax.set_ylabel("degradazione %")
    ax.set_title("Degradazione relativa")
    for i, v in enumerate(degrad):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=7)

    # --- 3. λ* vs ε ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(eps_arr, lambdas, "r-s", markersize=5)
    ax.set_xlabel("ε")
    ax.set_ylabel("λ*")
    ax.set_title("Moltiplicatore ottimo λ*(ε)")
    ax.grid(True, alpha=0.3)

    # --- 4. Mean transport vs ε ---
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(eps_arr, transport, "g-^", markersize=5)
    ax.plot(eps_arr, eps_arr, "k--", alpha=0.4, label="ε (complementary slackness)")
    ax.set_xlabel("ε")
    ax.set_ylabel("$\\frac{1}{m} \\sum \\|x^* - y\\|^2$")
    ax.set_title("Costo di trasporto medio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 5. Per-trajectory V_rob distribution for selected ε ---
    ax = fig.add_subplot(gs[1, 2])
    # Pick a mid-range ε
    mid_eps = epsilons[len(epsilons) // 2]
    per_traj = results[mid_eps]["per_traj_v_rob"]
    ax.hist(per_traj, bins=20, color="steelblue", alpha=0.7, density=True)
    ax.axvline(np.mean(per_traj), color="red", ls="--",
               label=f"mean={np.mean(per_traj):.3f}")
    ax.axvline(v_nom, color="green", ls="--",
               label=f"nominal={v_nom:.3f}")
    ax.set_xlabel("$V^{rob}$")
    ax.set_title(f"Distribuzione $V^{{rob}}$ (ε={mid_eps:.2f})")
    ax.legend(fontsize=8)

    plt.suptitle("Stress Test Distribuzionale — Modulo C",
                 fontsize=14, y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot salvato in: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load models
    world_model = load_world_model(args.wm_ckpt, device)
    critic      = load_critic(args.critic_ckpt, device)

    # Load data
    sequences, actions, rewards, regimes = load_dataset(args.dataset, device)

    # DRO config
    dro_cfg = DROConfig(
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_steps=args.outer_steps,
        lambda_init=args.lambda_init,
        trust_radius=args.trust_radius,
        gamma=args.gamma,
    )
    print(f"\nDRO config:")
    print(f"  inner_steps={dro_cfg.inner_steps}  inner_lr={dro_cfg.inner_lr} (adaptive)")
    print(f"  outer_steps={dro_cfg.outer_steps} (bisection)")
    print(f"  lambda_init={dro_cfg.lambda_init}  trust_radius={dro_cfg.trust_radius}")
    print(f"  gamma={dro_cfg.gamma}")

    # Epsilons to test
    epsilons = [float(e) for e in args.epsilons.split(",")]
    print(f"  epsilons: {epsilons}")

    # Run stress test
    runner = StressTestRunner(world_model, critic, dro_cfg)

    print(f"\nRunning stress test on {args.n_traj} trajectories...")
    t0 = time.time()

    results = runner.run_stress_test(
        sequences, actions, rewards,
        epsilons=epsilons,
        n_trajectories=args.n_traj,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Print summary table
    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS — V_rob(ε)")
    print("=" * 70)
    print(f"  {'ε':>8s}  {'V_rob':>10s}  {'V_nom':>10s}  "
          f"{'degrad%':>8s}  {'λ*':>8s}  {'transport':>10s}")
    print("-" * 70)
    for eps in epsilons:
        r = results[eps]
        print(f"  {eps:>8.4f}  {r['v_rob_mean']:>10.4f}  "
              f"{r['v_nominal_mean']:>10.4f}  "
              f"{r['degradation']*100:>7.1f}%  "
              f"{r['mean_lambda']:>8.3f}  "
              f"{r['mean_transport']:>10.6f}")

    # Plot
    plot_results(results, epsilons, args.out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRO Stress Test (Module C)")
    parser.add_argument("--wm_ckpt",      type=str, default="checkpoints/wm_best.pt")
    parser.add_argument("--critic_ckpt",  type=str, default="checkpoints/critic_best.pt")
    parser.add_argument("--dataset",      type=str, default="data/wm_dataset.npz")
    parser.add_argument("--epsilons",     type=str, default="0.0,0.01,0.05,0.1,0.2,0.5,1.0",
                        help="Comma-separated list of ε values")
    parser.add_argument("--n_traj",       type=int, default=50,
                        help="Number of trajectories to evaluate")
    parser.add_argument("--inner_steps",  type=int, default=100,
                        help="Gradient steps for inner x optimization")
    parser.add_argument("--inner_lr",     type=float, default=0.05,
                        help="Base inner lr (auto-scaled by 1/(2λ+1))")
    parser.add_argument("--outer_steps",  type=int, default=30,
                        help="Bisection iterations for λ")
    parser.add_argument("--lambda_init",  type=float, default=50.0,
                        help="Upper bound for λ bisection")
    parser.add_argument("--trust_radius", type=float, default=3.0,
                        help="Max per-dim deviation from centroid (OOD guard)")
    parser.add_argument("--gamma",        type=float, default=0.95)
    parser.add_argument("--out",          type=str, default="stress_test.png")
    args = parser.parse_args()
    main(args)
"""
validate_robust_critic.py — Validation of Backward Robust DP.

FAIR COMPARISON: both methods produce V(z_0, t=0) via backward sweep.

  - MULTI-STEP (backward DP): at step t, DRO uses V_rob(·, t+1)
    The adversary knows the future is also adversarial.

  - ONE-STEP: at step t, DRO uses V_nom (always the nominal critic)
    The adversary thinks the future is nominal — myopic.

  Both fit a last-layer V(·, t) at each step and produce V(z_0, t=0).
  Same critic architecture, same frozen φ, same DRO solver.
  Only difference: what continuation value the adversary sees.

Tests:
  1. Consistency: V_rob(ε=0) ≈ V_nom for both methods
  2. Monotonicity: V_rob(ε) decreasing for both
  3. Multi-step ≤ One-step: backward DP is more conservative

Uso:
  python scripts/validate_robust_critic.py \
      --n_sequences 20000 --epsilons "0,0.01,0.02,0.05,0.10,0.20"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from models.critic import ValueNetwork
from models.world_model import LOBWorldModel, WorldModelConfig
from models.dro import DROConfig

from train_robust_critic import (
    FrozenPhiCritic,
    batched_inner_solve,
    batched_robust_targets,
    precompute_gmm_by_timestep,
    fit_last_layer,
)


# ---------------------------------------------------------------------------
# Nominal expectation (no DRO) — helper for ε=0
# ---------------------------------------------------------------------------

def nominal_expectation(
    critic: nn.Module,
    pi_t: torch.Tensor,    # (M, K)
    mu_t: torch.Tensor,    # (M, K, D)
    r_t: torch.Tensor,     # (M,)
    gamma: float,
    device: torch.device,
    chunk_size: int = 10000,
) -> torch.Tensor:
    """E_P[r + γ V(z')] using GMM centroids."""
    M, K, D = mu_t.shape
    targets = torch.zeros(M)

    for c_start in range(0, M, chunk_size):
        c_end = min(c_start + chunk_size, M)
        mu_c = mu_t[c_start:c_end].to(device)
        pi_c = pi_t[c_start:c_end].to(device)
        n = c_end - c_start
        with torch.no_grad():
            v_k = critic(mu_c.reshape(n * K, D)).reshape(n, K)
        v_nom = (pi_c * v_k).sum(dim=-1).cpu()
        targets[c_start:c_end] = r_t[c_start:c_end] + gamma * v_nom

    return targets


# ---------------------------------------------------------------------------
# Generic backward sweep (works for both multi-step and one-step)
# ---------------------------------------------------------------------------

def run_backward_sweep(
    critic: FrozenPhiCritic,
    nominal_state: dict,
    sequences: torch.Tensor,
    rewards: torch.Tensor,
    pi_all: torch.Tensor,
    mu_all: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epsilon: float,
    cfg: DROConfig,
    device: torch.device,
    use_robust_continuation: bool,  # True = multi-step, False = one-step
    fit_epochs: int = 10,
    fit_batch_size: int = 2048,
    fit_lr: float = 1e-3,
    chunk_size: int = 10000,
) -> dict:
    """
    Backward sweep producing V(z_0, t=0).

    use_robust_continuation=True:
      DRO at step t uses V_rob(·, t+1) from previous backward step.
      This is the full multi-step backward DP.

    use_robust_continuation=False:
      DRO at step t uses V_nom (nominal critic, always same weights).
      This is the one-step baseline — adversary is myopic.
    """
    M, Np1, D = sequences.shape
    N = Np1 - 1

    layer_weights = {}

    for t in range(N - 1, -1, -1):
        z_t = sequences[:, t, :]
        r_t = rewards[:, t]

        if t == N - 1:
            targets = r_t.clone()
        else:
            # Choose which V to use for computing targets
            if use_robust_continuation:
                # Multi-step: use V_rob(·, t+1) from previous backward step
                critic.set_last_layer_state(layer_weights[t + 1])
            else:
                # One-step: always use V_nom
                critic.set_last_layer_state(nominal_state)

            critic.critic.eval()

            pi_t_gmm = pi_all[:, t, :]
            mu_t_gmm = mu_all[:, t, :, :]

            if epsilon < 1e-8:
                targets = nominal_expectation(
                    critic, pi_t_gmm, mu_t_gmm, r_t,
                    gamma=cfg.gamma, device=device, chunk_size=chunk_size,
                )
            else:
                targets = batched_robust_targets(
                    critic, pi_t_gmm, mu_t_gmm, r_t,
                    epsilon=epsilon, cfg=cfg,
                    device=device, chunk_size=chunk_size,
                    verbose=False,
                )

        # Warm start for fitting: from t+1 if available, else nominal
        if t < N - 1:
            critic.set_last_layer_state(layer_weights[t + 1])
        else:
            critic.set_last_layer_state(nominal_state)

        # Fit V(·, t)
        fit_last_layer(
            critic,
            z_t, targets,
            train_mask, val_mask,
            device,
            epochs=fit_epochs,
            batch_size=fit_batch_size,
            lr=fit_lr,
        )

        layer_weights[t] = critic.get_last_layer_state()

    # V at t=0
    critic.set_last_layer_state(layer_weights[0])
    critic.critic.eval()
    with torch.no_grad():
        v_0 = critic(sequences[:, 0, :].to(device)).cpu().mean().item()

    return {
        "epsilon": epsilon,
        "v_0": v_0,
        "target_mean_0": targets.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load models ---
    wm_ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    wm_cfg  = WorldModelConfig()
    for k, v in wm_ckpt["cfg"].items():
        setattr(wm_cfg, k, v)
    world_model = LOBWorldModel(wm_cfg).to(device)
    world_model.load_state_dict(wm_ckpt["model"])
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    cr_ckpt = torch.load(args.critic_ckpt, map_location=device, weights_only=False)
    cr_cfg  = cr_ckpt["cfg"]

    base_critic = ValueNetwork(
        d_latent=cr_cfg["d_latent"],
        hidden=cr_cfg["hidden"],
        n_layers=cr_cfg["n_layers"],
    ).to(device)
    base_critic.load_state_dict(cr_ckpt["model"])

    critic = FrozenPhiCritic(base_critic)
    nominal_state = critic.get_last_layer_state()

    print(f"Critic: {critic.n_frozen:,} frozen + {critic.n_trainable:,} trainable")

    # --- Dataset ---
    data = np.load(args.dataset)
    sequences   = torch.from_numpy(data["sequences"])
    actions     = torch.from_numpy(data["actions"])
    rewards     = torch.from_numpy(data["rewards"])
    episode_ids = data["episode_ids"]

    M, Np1, D = sequences.shape
    N = Np1 - 1

    if args.n_sequences < M:
        rng = np.random.default_rng(42)
        seq_idx = rng.choice(M, size=args.n_sequences, replace=False)
        sequences   = sequences[seq_idx]
        actions     = actions[seq_idx]
        rewards     = rewards[seq_idx]
        episode_ids = episode_ids[seq_idx]
        M = args.n_sequences

    # Episode split
    unique_eps = np.unique(episode_ids)
    rng_split  = np.random.default_rng(123)
    rng_split.shuffle(unique_eps)
    n_val_eps  = max(1, int(len(unique_eps) * 0.1))
    val_eps    = set(unique_eps[-n_val_eps:])
    is_val     = np.array([eid in val_eps for eid in episode_ids], dtype=bool)
    train_mask = torch.from_numpy(~is_val)
    val_mask   = torch.from_numpy(is_val)

    print(f"Dataset: {M:,} sequences, N={N}")
    print(f"Split: {train_mask.sum().item():,} train, {val_mask.sum().item():,} val")

    # --- GMMs ---
    print("Pre-computing GMMs...")
    pi_all, mu_all = precompute_gmm_by_timestep(
        world_model, sequences, actions, device
    )

    # --- Nominal V ---
    critic.set_last_layer_state(nominal_state)
    critic.critic.eval()
    with torch.no_grad():
        v_nom = critic(sequences[:, 0, :].to(device)).cpu().mean().item()
    print(f"V_nom(t=0) = {v_nom:.4f}")

    # --- DRO config ---
    dro_cfg = DROConfig(
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_steps=args.outer_steps,
        lambda_init=args.lambda_init,
        trust_radius=args.trust_radius,
        gamma=args.gamma,
    )

    epsilons = [float(e) for e in args.epsilons.split(",")]
    print(f"\nEpsilons: {epsilons}")

    # =====================================================================
    # MULTI-STEP: Backward DP with robust continuation
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"MULTI-STEP BACKWARD DP (adversary sees V_rob)")
    print(f"{'='*60}")

    multistep_results = []
    for eps in epsilons:
        print(f"\n--- ε = {eps} ---")
        t0 = time.time()
        critic.set_last_layer_state(nominal_state)

        result = run_backward_sweep(
            critic, nominal_state,
            sequences, rewards, pi_all, mu_all,
            train_mask, val_mask,
            epsilon=eps, cfg=dro_cfg, device=device,
            use_robust_continuation=True,
            fit_epochs=args.fit_epochs,
            fit_batch_size=args.fit_batch_size,
            fit_lr=args.fit_lr,
            chunk_size=args.chunk_size,
        )

        elapsed = time.time() - t0
        multistep_results.append(result)
        print(f"  V_rob(ε={eps}) = {result['v_0']:.4f}  ({elapsed:.0f}s)")

    # =====================================================================
    # ONE-STEP: Backward DP with nominal continuation
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"ONE-STEP BACKWARD (adversary sees V_nom)")
    print(f"{'='*60}")

    onestep_results = []
    for eps in epsilons:
        print(f"\n--- ε = {eps} ---")
        t0 = time.time()
        critic.set_last_layer_state(nominal_state)

        result = run_backward_sweep(
            critic, nominal_state,
            sequences, rewards, pi_all, mu_all,
            train_mask, val_mask,
            epsilon=eps, cfg=dro_cfg, device=device,
            use_robust_continuation=False,
            fit_epochs=args.fit_epochs,
            fit_batch_size=args.fit_batch_size,
            fit_lr=args.fit_lr,
            chunk_size=args.chunk_size,
        )

        elapsed = time.time() - t0
        onestep_results.append(result)
        print(f"  V_1step(ε={eps}) = {result['v_0']:.4f}  ({elapsed:.0f}s)")

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"V_nom = {v_nom:.4f}")
    print(f"")
    print(f"{'ε':>8s}  {'Multi-step':>12s}  {'One-step':>12s}  "
          f"{'Δ(M-O)':>10s}  {'Δ%multi':>10s}  {'Δ%one':>10s}")
    print(f"{'─'*70}")
    for mr, osr in zip(multistep_results, onestep_results):
        eps = mr["epsilon"]
        vm  = mr["v_0"]
        vo  = osr["v_0"]
        delta = vm - vo
        pct_m = 100 * (vm - v_nom) / (abs(v_nom) + 1e-8) if abs(v_nom) > 1e-8 else 0
        pct_o = 100 * (vo - v_nom) / (abs(v_nom) + 1e-8) if abs(v_nom) > 1e-8 else 0
        print(f"{eps:8.3f}  {vm:12.4f}  {vo:12.4f}  "
              f"{delta:10.4f}  {pct_m:9.1f}%  {pct_o:9.1f}%")

    # =====================================================================
    # Consistency check
    # =====================================================================
    eps0_multi = [r for r in multistep_results if r["epsilon"] < 1e-8]
    eps0_one   = [r for r in onestep_results if r["epsilon"] < 1e-8]
    print(f"\nCONSISTENCY CHECK (ε=0):")
    if eps0_multi:
        gap_m = abs(eps0_multi[0]["v_0"] - v_nom)
        print(f"  Multi-step: V(ε=0) = {eps0_multi[0]['v_0']:.4f}, "
              f"gap = {gap_m:.4f}  {'✓' if gap_m < 0.15 else '✗'}")
    if eps0_one:
        gap_o = abs(eps0_one[0]["v_0"] - v_nom)
        print(f"  One-step:   V(ε=0) = {eps0_one[0]['v_0']:.4f}, "
              f"gap = {gap_o:.4f}  {'✓' if gap_o < 0.15 else '✗'}")

    # Monotonicity
    mv = [r["v_0"] for r in multistep_results]
    ov = [r["v_0"] for r in onestep_results]
    mono_m = all(mv[i] >= mv[i+1] - 0.02 for i in range(len(mv)-1))
    mono_o = all(ov[i] >= ov[i+1] - 0.02 for i in range(len(ov)-1))
    print(f"\nMONOTONICITY:")
    print(f"  Multi-step: {'✓' if mono_m else '✗'}")
    print(f"  One-step:   {'✓' if mono_o else '✗'}")

    # Multi-step ≤ One-step check
    ms_le_os = all(mv[i] <= ov[i] + 0.02 for i in range(len(mv)))
    print(f"\nMULTI-STEP ≤ ONE-STEP:")
    print(f"  {'✓' if ms_le_os else '✗'} (multi-step should be more conservative)")

    # =====================================================================
    # Plot
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute values
    ax = axes[0]
    eps_list = [r["epsilon"] for r in multistep_results]
    ax.plot(eps_list, mv, "o-", color="C0", linewidth=2, markersize=8,
            label="Multi-step (backward DP)")
    ax.plot(eps_list, ov, "s--", color="C1", linewidth=2, markersize=8,
            label="One-step (myopic adversary)")
    ax.axhline(v_nom, color="gray", linestyle=":", linewidth=1,
               label=f"V_nom = {v_nom:.3f}")
    ax.set_xlabel("Wasserstein radius ε", fontsize=12)
    ax.set_ylabel("V(z₀, t=0)", fontsize=12)
    ax.set_title("Robust Value: V(z₀, t=0)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: relative degradation
    ax = axes[1]
    if abs(v_nom) > 1e-8:
        pct_m = [100 * (v - v_nom) / abs(v_nom) for v in mv]
        pct_o = [100 * (v - v_nom) / abs(v_nom) for v in ov]
    else:
        pct_m = [v - v_nom for v in mv]
        pct_o = [v - v_nom for v in ov]
    ax.plot(eps_list, pct_m, "o-", color="C0", linewidth=2, markersize=8,
            label="Multi-step")
    ax.plot(eps_list, pct_o, "s--", color="C1", linewidth=2, markersize=8,
            label="One-step")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Wasserstein radius ε", fontsize=12)
    ax.set_ylabel("Δ% from nominal", fontsize=12)
    ax.set_title("Relative Degradation", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.out_plot)
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")

    # Save
    results = {
        "v_nom":     v_nom,
        "multistep": multistep_results,
        "onestep":   onestep_results,
        "epsilons":  epsilons,
    }
    torch.save(results, Path(args.ckpt_dir) / "validation_results.pt")
    print(f"Results saved: {Path(args.ckpt_dir) / 'validation_results.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Backward Robust DP")
    parser.add_argument("--critic_ckpt",   type=str, default="checkpoints/critic_best.pt")
    parser.add_argument("--wm_ckpt",       type=str, default="checkpoints/wm_best.pt")
    parser.add_argument("--dataset",       type=str, default="data/wm_dataset.npz")
    parser.add_argument("--epsilons",      type=str, default="0,0.01,0.02,0.05,0.10,0.20")
    parser.add_argument("--n_sequences",   type=int, default=20000)
    parser.add_argument("--inner_steps",   type=int, default=50)
    parser.add_argument("--inner_lr",      type=float, default=0.05)
    parser.add_argument("--outer_steps",   type=int, default=20)
    parser.add_argument("--lambda_init",   type=float, default=50.0)
    parser.add_argument("--trust_radius",  type=float, default=0.2)
    parser.add_argument("--gamma",         type=float, default=0.95)
    parser.add_argument("--chunk_size",    type=int, default=10000)
    parser.add_argument("--fit_epochs",    type=int, default=10)
    parser.add_argument("--fit_batch_size",type=int, default=2048)
    parser.add_argument("--fit_lr",        type=float, default=1e-3)
    parser.add_argument("--out_plot",      type=str, default="stress_test_multistep.png")
    parser.add_argument("--ckpt_dir",      type=str, default="checkpoints")
    args = parser.parse_args()
    main(args)

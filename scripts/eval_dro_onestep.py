"""
eval_dro_onestep.py — Validazione completa DRO one-step.

Esegue un sweep su ε ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5} e per ogni ε
valuta il solver DRO su n campioni dal dataset. Produce un pannello 2×3:

  Row 1 — La storia principale:
    A1: V_rob(ε) globale con ±std e V_nom baseline
    A2: V_rob(ε) per regime (3 curve)
    A3: Δ% degradazione per regime (bar chart per ε selezionati)

  Row 2 — Diagnostiche del solver:
    B1: Transport vs ε per regime (deve seguire la diagonale)
    B2: λ*(ε) per regime
    B3: (x* - y) / σ histogram (per un ε rappresentativo)

Criteri di validazione:
  ✓ V_rob monotonicamente decrescente con ε
  ✓ V_rob < V_nom sempre (zero violations)
  ✓ Transport ≈ ε (complementary slackness)
  ✓ Displacement entro ±3σ

Uso:
    python scripts/eval_dro_onestep.py
    python scripts/eval_dro_onestep.py --n_samples 300 --epsilons 0.0,0.01,0.05,0.1,0.2,0.5
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.critic import ValueNetwork
from models.world_model import LOBWorldModel, WorldModelConfig
from models.dro import WassersteinDRO, DROConfig


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_critic(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    critic = ValueNetwork(d_latent=cfg["d_latent"], hidden=cfg["hidden"],
                          n_layers=cfg["n_layers"]).to(device)
    critic.load_state_dict(ckpt["model"])
    critic.eval()
    L = ckpt.get("lipschitz_estimate", "?")
    print(f"Critic: {path}  val={ckpt['val_loss']:.4f}  L={L}")
    return critic


def load_world_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = WorldModelConfig()
    for k, v in ckpt["cfg"].items():
        setattr(cfg, k, v)
    wm = LOBWorldModel(cfg).to(device)
    wm.load_state_dict(ckpt["model"])
    wm.eval()
    print(f"WM: {path}  epoch={ckpt['epoch']}")
    return wm


def load_dataset(path: str, n_samples: int, seed: int = 42):
    data = np.load(path)
    seqs = torch.from_numpy(data["sequences"].astype(np.float32))
    acts = torch.from_numpy(data["actions"].astype(np.float32))
    rews = torch.from_numpy(data["rewards"].astype(np.float32))
    regs = data["regimes"].astype(np.int64)
    if regs.ndim > 1:
        regs = regs[:, 0]

    M = len(seqs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(M, size=min(n_samples, M), replace=False)
    return seqs[idx], acts[idx], rews[idx], regs[idx]


# ---------------------------------------------------------------------------
# Core eval: run DRO at t=0 for all samples at a single ε
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_gmm_at_t(wm, z_seq, a_seq, t):
    pi, mu, log_sig = wm(z_seq, a_seq)
    return pi[0, t], mu[0, t], log_sig[0, t]


def eval_single_epsilon(
    critic, wm, dro, sequences, actions, rewards, regimes,
    epsilon: float, device: torch.device,
) -> dict:
    """Run DRO one-step at t=0 for each sample. Returns per-sample results."""
    M = len(sequences)

    v_rob, v_nom = np.zeros(M), np.zeros(M)
    lam_arr, transport_arr = np.zeros(M), np.zeros(M)
    delta_norm_list = []

    for i in range(M):
        z_seq = sequences[i:i+1].to(device)
        a_seq = actions[i:i+1].to(device)

        pi_t, mu_t, ls_t = get_gmm_at_t(wm, z_seq, a_seq, t=0)
        r_t = rewards[i, 0].item()

        with torch.enable_grad():
            res = dro.robust_bellman_backup(r_t, pi_t, mu_t, ls_t, epsilon)

        v_rob[i] = res["v_robust"]
        v_nom[i] = res["v_nominal"]
        lam_arr[i] = res["lambda_star"]
        transport_arr[i] = res["transport"]

        # Normalized displacement
        y, _, sigma = dro._sample_nominal(pi_t, mu_t, ls_t)
        delta = ((res["x_star"] - y) / (sigma + 1e-8)).cpu().numpy().reshape(-1)
        delta_norm_list.append(delta)

    return {
        "v_rob": v_rob, "v_nom": v_nom,
        "lam": lam_arr, "transport": transport_arr,
        "delta_norm": np.concatenate(delta_norm_list),
        "regimes": regimes,
    }


# ---------------------------------------------------------------------------
# Full ε sweep
# ---------------------------------------------------------------------------

def run_sweep(
    critic, wm, sequences, actions, rewards, regimes,
    epsilons: list[float], device: torch.device, dro_cfg: DROConfig,
) -> dict:
    """Run eval for each ε. Returns results[ε] = {...}."""
    dro = WassersteinDRO(critic, dro_cfg)
    all_results = {}

    for eps in epsilons:
        t0 = time.time()
        print(f"  ε={eps:.3f}  ", end="", flush=True)

        res = eval_single_epsilon(
            critic, wm, dro, sequences, actions, rewards, regimes,
            epsilon=eps, device=device,
        )
        elapsed = time.time() - t0

        # Per-regime stats
        regime_stats = {}
        for r, name in enumerate(["low_vol", "mid_vol", "high_vol"]):
            mask = (regimes == r)
            if mask.sum() == 0:
                continue
            vr, vn = res["v_rob"][mask], res["v_nom"][mask]
            regime_stats[r] = {
                "v_rob_mean": vr.mean(), "v_rob_std": vr.std(),
                "v_nom_mean": vn.mean(),
                "lam_mean": res["lam"][mask].mean(),
                "transport_mean": res["transport"][mask].mean(),
                "violations": int((vr > vn).sum()),
                "n": int(mask.sum()),
            }

        all_results[eps] = {
            "raw": res,
            "regime_stats": regime_stats,
            "v_rob_mean": res["v_rob"].mean(),
            "v_rob_std": res["v_rob"].std(),
            "v_nom_mean": res["v_nom"].mean(),
            "lam_mean": res["lam"].mean(),
            "transport_mean": res["transport"].mean(),
        }

        print(f"V_rob={res['v_rob'].mean():.4f}  "
              f"V_nom={res['v_nom'].mean():.4f}  "
              f"W={res['transport'].mean():.4f}  "
              f"λ*={res['lam'].mean():.2f}  "
              f"({elapsed:.1f}s)")

    return all_results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(results: dict, epsilons: list[float]) -> None:
    names = ["low_vol", "mid_vol", "high_vol"]

    print(f"\n{'='*80}")
    print(f"DRO ONE-STEP VALIDATION — SUMMARY")
    print(f"{'='*80}")

    # Global table
    print(f"\n{'ε':>6s}  {'V_rob':>8s}  {'V_nom':>8s}  {'Δ%':>7s}  "
          f"{'λ*':>6s}  {'W':>8s}  {'violations':>10s}")
    print(f"{'─'*65}")
    for eps in epsilons:
        r = results[eps]
        vr, vn = r["v_rob_mean"], r["v_nom_mean"]
        deg = (vn - vr) / (abs(vn) + 1e-8) * 100
        total_viol = sum(
            rs["violations"] for rs in r["regime_stats"].values()
        )
        total_n = sum(rs["n"] for rs in r["regime_stats"].values())
        print(f"{eps:6.3f}  {vr:8.4f}  {vn:8.4f}  {deg:6.1f}%  "
              f"{r['lam_mean']:6.2f}  {r['transport_mean']:8.4f}  "
              f"{total_viol}/{total_n}")

    # Per-regime detail for a representative ε
    eps_detail = [e for e in epsilons if e > 0]
    if eps_detail:
        eps_rep = eps_detail[len(eps_detail) // 2]
        print(f"\nDetail per regime (ε={eps_rep}):")
        for r_id, name in enumerate(names):
            rs = results[eps_rep]["regime_stats"].get(r_id)
            if rs is None:
                continue
            deg = (rs["v_nom_mean"] - rs["v_rob_mean"]) / (abs(rs["v_nom_mean"]) + 1e-8) * 100
            print(f"  {name:10s}  V_nom={rs['v_nom_mean']:.4f}  "
                  f"V_rob={rs['v_rob_mean']:.4f}  Δ={deg:.1f}%  "
                  f"λ*={rs['lam_mean']:.2f}  W={rs['transport_mean']:.4f}")

    # Displacement stats
    for eps in epsilons:
        if eps > 0:
            dn = results[eps]["raw"]["delta_norm"]
            pct = (np.abs(dn) <= 3.0).mean() * 100
            if pct < 100:
                print(f"\n  ⚠ ε={eps}: {pct:.1f}% displacement entro ±3σ")
            break
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(results: dict, epsilons: list[float], out_path: str) -> None:
    names = ["low_vol", "mid_vol", "high_vol"]
    colors = ["#2166ac", "#f4a582", "#b2182b"]  # blue, orange, red
    eps_pos = [e for e in epsilons if e > 0]  # exclude ε=0 for some plots

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # =====================================================================
    # ROW 1 — La storia principale
    # =====================================================================

    # --- A1: V_rob(ε) globale ---
    ax = fig.add_subplot(gs[0, 0])
    eps_arr = np.array(epsilons)
    vr_arr = np.array([results[e]["v_rob_mean"] for e in epsilons])
    vr_std = np.array([results[e]["v_rob_std"] for e in epsilons])
    vn = results[epsilons[0]]["v_nom_mean"]

    ax.plot(eps_arr, vr_arr, "o-", color="#2166ac", linewidth=2, markersize=6,
            label="$V^{rob}(\\varepsilon)$", zorder=3)
    ax.fill_between(eps_arr, vr_arr - vr_std, vr_arr + vr_std,
                    alpha=0.15, color="#2166ac")
    ax.axhline(vn, color="#4daf4a", ls="--", lw=1.5,
               label=f"$V^{{nom}}$ = {vn:.3f}")
    ax.set_xlabel("ε (Wasserstein radius)")
    ax.set_ylabel("$V^{rob}(\\varepsilon)$")
    ax.set_title("Curva di valore robusto", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # --- A2: V_rob(ε) per regime ---
    ax = fig.add_subplot(gs[0, 1])
    for r_id, (name, c) in enumerate(zip(names, colors)):
        vr_regime = []
        vn_regime = []
        for eps in epsilons:
            rs = results[eps]["regime_stats"].get(r_id)
            if rs:
                vr_regime.append(rs["v_rob_mean"])
                vn_regime.append(rs["v_nom_mean"])
            else:
                vr_regime.append(np.nan)
                vn_regime.append(np.nan)
        ax.plot(eps_arr, vr_regime, "o-", color=c, linewidth=1.8,
                markersize=5, label=name)
        # Nominal baseline per regime (dashed, at ε=0)
        if not np.isnan(vn_regime[0]):
            ax.axhline(vn_regime[0], color=c, ls=":", alpha=0.4, lw=1)

    ax.set_xlabel("ε")
    ax.set_ylabel("$V^{rob}(\\varepsilon)$")
    ax.set_title("$V^{rob}(\\varepsilon)$ per regime", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # --- A3: Degradazione assoluta per regime (bar chart) ---
    ax = fig.add_subplot(gs[0, 2])
    # Pick 3 representative ε values
    eps_show = [e for e in [0.01, 0.05, 0.1, 0.2] if e in results]
    if not eps_show:
        eps_show = eps_pos[:3]
    n_eps = len(eps_show)
    n_reg = 3
    bar_w = 0.8 / n_eps

    for j, eps in enumerate(eps_show):
        for r_id, (name, c) in enumerate(zip(names, colors)):
            rs = results[eps]["regime_stats"].get(r_id)
            if rs is None:
                continue
            delta = rs["v_nom_mean"] - rs["v_rob_mean"]
            x_pos = r_id + (j - n_eps/2 + 0.5) * bar_w
            alpha = 0.4 + 0.6 * (j / max(n_eps - 1, 1))
            bar = ax.bar(x_pos, delta, width=bar_w * 0.9, color=c, alpha=alpha)
            if j == n_eps - 1:  # label only last group
                ax.text(x_pos, delta + 0.005, f"{delta:.3f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(n_reg))
    ax.set_xticklabels(names)
    ax.set_ylabel("$V^{nom} - V^{rob}$ (assoluto)")
    ax.set_title("Degradazione per regime", fontweight="bold")

    # Custom legend for ε values
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="gray", alpha=0.4 + 0.6 * (j / max(n_eps-1, 1)),
                     label=f"ε={eps_show[j]}")
               for j in range(n_eps)]
    ax.legend(handles=handles, fontsize=8)

    # =====================================================================
    # ROW 2 — Diagnostiche del solver
    # =====================================================================

    # --- B1: Transport vs ε (complementary slackness) ---
    ax = fig.add_subplot(gs[1, 0])
    for r_id, (name, c) in enumerate(zip(names, colors)):
        tr_regime = []
        for eps in eps_pos:
            rs = results[eps]["regime_stats"].get(r_id)
            tr_regime.append(rs["transport_mean"] if rs else np.nan)
        ax.plot(eps_pos, tr_regime, "o-", color=c, markersize=5, label=name)

    eps_pos_arr = np.array(eps_pos)
    ax.plot(eps_pos_arr, eps_pos_arr, "k--", alpha=0.3, label="W = ε (ideale)")
    ax.set_xlabel("ε")
    ax.set_ylabel("Transport W")
    ax.set_title("Complementary slackness", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- B2: λ*(ε) per regime ---
    ax = fig.add_subplot(gs[1, 1])
    for r_id, (name, c) in enumerate(zip(names, colors)):
        lam_regime = []
        for eps in eps_pos:
            rs = results[eps]["regime_stats"].get(r_id)
            lam_regime.append(rs["lam_mean"] if rs else np.nan)
        ax.plot(eps_pos, lam_regime, "s-", color=c, markersize=5, label=name)

    ax.set_xlabel("ε")
    ax.set_ylabel("λ*")
    ax.set_title("Moltiplicatore duale λ*(ε)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- B3: Displacement histogram ---
    ax = fig.add_subplot(gs[1, 2])
    # Use a mid-range ε
    eps_hist = eps_pos[len(eps_pos) // 2] if eps_pos else 0.1
    if eps_hist in results:
        dn = results[eps_hist]["raw"]["delta_norm"]
        ax.hist(dn, bins=80, color="#2166ac", alpha=0.7, density=True,
                edgecolor="white", linewidth=0.3)
        ax.axvline(-3, color="#b2182b", ls="--", alpha=0.6)
        ax.axvline(3, color="#b2182b", ls="--", alpha=0.6, label="±3σ")
        pct = (np.abs(dn) <= 3.0).mean() * 100
        ax.set_title(f"$(x^* - y) / \\sigma$  (ε={eps_hist}, {pct:.0f}% in ±3σ)",
                     fontweight="bold")
        ax.set_xlabel("$(x^* - y) / \\sigma$")
        ax.set_ylabel("density")
        # Add stats text
        ax.text(0.97, 0.95, f"μ={dn.mean():.3f}\nσ={dn.std():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8)

    plt.suptitle("DRO One-Step Validation", fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot salvato: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    critic = load_critic(args.critic_ckpt, device)
    wm = load_world_model(args.wm_ckpt, device)

    sequences, actions, rewards, regimes = load_dataset(
        args.dataset, n_samples=args.n_samples
    )
    print(f"Samples: {len(sequences)}  "
          f"(low={sum(regimes==0)}, mid={sum(regimes==1)}, high={sum(regimes==2)})\n")

    epsilons = [float(e) for e in args.epsilons.split(",")]

    dro_cfg = DROConfig(
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps,
        cost_type=args.cost_type,
        n_samples_per_component=args.n_samples_per_comp,
        trust_radius_sigma=args.trust_radius,
    )
    print(f"DRO: cost={dro_cfg.cost_type}  n_s={dro_cfg.n_samples_per_component}  "
          f"R={dro_cfg.trust_radius_sigma}σ  inner={dro_cfg.inner_steps}\n")

    print(f"Running ε sweep: {epsilons}")
    results = run_sweep(
        critic, wm, sequences, actions, rewards, regimes,
        epsilons=epsilons, device=device, dro_cfg=dro_cfg,
    )

    print_summary(results, epsilons)
    plot_all(results, epsilons, out_path=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRO One-Step Validation")
    parser.add_argument("--critic_ckpt",      type=str,   default="checkpoints/critic_best.pt")
    parser.add_argument("--wm_ckpt",          type=str,   default="checkpoints/wm_best.pt")
    parser.add_argument("--dataset",          type=str,   default="data/wm_dataset.npz")
    parser.add_argument("--epsilons",         type=str,   default="0.0,0.01,0.05,0.1,0.2,0.5")
    parser.add_argument("--n_samples",        type=int,   default=200)
    parser.add_argument("--inner_steps",      type=int,   default=100)
    parser.add_argument("--outer_steps",      type=int,   default=30)
    parser.add_argument("--cost_type",        type=str,   default="mahalanobis")
    parser.add_argument("--n_samples_per_comp", type=int, default=3)
    parser.add_argument("--trust_radius",     type=float, default=3.0)
    parser.add_argument("--out",              type=str,   default="eval_dro_onestep.png")
    args = parser.parse_args()
    main(args)
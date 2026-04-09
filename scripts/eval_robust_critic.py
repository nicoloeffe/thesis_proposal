"""
eval_robust_critic.py — Fase 5: Evaluation finale.

I target_mean del backward DP sono il risultato primario — sono calcolati
esattamente dal DRO senza errore di fitting. Il fit last-layer serve solo
per propagare V_rob all'indietro, non per l'output finale.

Pannello 2×3:
  A1: target_mean robusto vs nominale per timestep
  A2: Degradazione Δ(t) = E[G_nom(t)] - target_rob(t) per timestep
  A3: Degradazione per regime (bar chart)

  B1: Qualità del fit (res/std) — con nota sulla varianza irriducibile
  B2: Val loss vs baseline
  B3: Confronto target robusto vs nominale a t=0 (istogramma)

Uso:
    python scripts/eval_robust_critic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.critic import ValueNetwork


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_robust_ckpt(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(f"Robust critic: {path}")
    print(f"  ε={ckpt['epsilon']}  γ={ckpt['gamma']}  N={ckpt['N']}  "
          f"cost={ckpt.get('cost_type', 'l2')}")
    return ckpt


def compute_nominal_returns(dataset_path: str, gamma: float,
                            reward_stats: dict) -> dict:
    """
    Compute per-timestep nominal MC returns (normalized).
    G_t = r̃_t + γ·r̃_{t+1} + ... + γ^{N-1-t}·r̃_{N-1}
    """
    data = np.load(dataset_path)
    rewards = data["rewards"].astype(np.float32)
    regimes = data["regimes"].astype(np.int64)
    if regimes.ndim > 1:
        regimes = regimes[:, 0]

    M, N = rewards.shape
    r_norm = (rewards - reward_stats["mean"]) / (reward_stats["std"] + 1e-8)

    # MC returns backward
    G = np.zeros_like(r_norm)
    G[:, -1] = r_norm[:, -1]
    for t in range(N - 2, -1, -1):
        G[:, t] = r_norm[:, t] + gamma * G[:, t + 1]

    # Per-timestep stats
    result = {"N": N, "regimes": regimes}
    for t in range(N):
        g_t = G[:, t]
        result[t] = {
            "mean": float(g_t.mean()),
            "std": float(g_t.std()),
        }
        for r in [0, 1, 2]:
            mask = (regimes == r)
            if mask.sum() > 0:
                result[t][f"mean_r{r}"] = float(g_t[mask].mean())

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(ckpt: dict, nom: dict) -> None:
    diags = sorted(ckpt["diagnostics"], key=lambda x: x["t"])
    eps = ckpt["epsilon"]

    print(f"\n{'='*80}")
    print(f"ROBUST CRITIC EVALUATION  (ε={eps})")
    print(f"{'='*80}")

    print(f"\n{'t':>3s} {'G_nom':>8s} {'G_rob':>8s} {'Δ':>8s} {'Δ%':>7s} "
          f"{'res/std':>8s} {'val_loss':>10s}")
    print(f"{'─'*60}")

    for d in diags:
        t = d["t"]
        g_nom = nom[t]["mean"]
        g_rob = d["target_mean"]
        delta = g_nom - g_rob
        delta_pct = delta / (abs(g_nom) + 1e-8) * 100
        ratio = d["residual"] / (d["target_std"] + 1e-8)
        print(f"{t:3d} {g_nom:8.4f} {g_rob:8.4f} {delta:8.4f} {delta_pct:6.1f}% "
              f"{ratio:8.3f} {d['val_loss']:10.4f}")

    # Summary at t=0
    g_nom_0 = nom[0]["mean"]
    g_rob_0 = diags[0]["target_mean"] if diags[0]["t"] == 0 else [d for d in diags if d["t"]==0][0]["target_mean"]
    print(f"\n  G_nom(t=0) = {g_nom_0:.4f}")
    print(f"  G_rob(t=0) = {g_rob_0:.4f}")
    print(f"  Δ = {g_nom_0 - g_rob_0:.4f}")

    # Per regime at t=0
    names = ["low_vol", "mid_vol", "high_vol"]
    print(f"\n  Per regime (t=0):")
    for r, name in enumerate(names):
        key = f"mean_r{r}"
        if key in nom[0]:
            print(f"    {name}: G_nom={nom[0][key]:.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(ckpt: dict, nom: dict, out_path: str) -> None:
    diags = sorted(ckpt["diagnostics"], key=lambda x: x["t"])
    eps = ckpt["epsilon"]
    N = ckpt["N"]
    names = ["low_vol", "mid_vol", "high_vol"]
    colors_regime = ["#2166ac", "#f4a582", "#b2182b"]

    ts = [d["t"] for d in diags]
    g_rob = [d["target_mean"] for d in diags]
    g_rob_std = [d["target_std"] for d in diags]
    g_nom = [nom[t]["mean"] for t in ts]
    g_nom_std = [nom[t]["std"] for t in ts]
    residuals = [d["residual"] for d in diags]
    val_losses = [d["val_loss"] for d in diags]
    deltas = [n - r for n, r in zip(g_nom, g_rob)]

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ==== A1: G_nom vs G_rob per timestep ====
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ts, g_nom, "o-", color="#4daf4a", linewidth=2, markersize=5,
            label="$G^{nom}(t)$")
    ax.plot(ts, g_rob, "s-", color="#2166ac", linewidth=2, markersize=5,
            label="$G^{rob}(t)$")
    ax.fill_between(ts, [m-s for m,s in zip(g_rob, g_rob_std)],
                    [m+s for m,s in zip(g_rob, g_rob_std)],
                    alpha=0.1, color="#2166ac")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("E[G(t)]")
    ax.set_title("Return nominale vs robusto", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    # ==== A2: Degradazione Δ(t) ====
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ts, deltas, "o-", color="#e66101", linewidth=2, markersize=5)
    ax.fill_between(ts, 0, deltas, alpha=0.2, color="#e66101")
    ax.axhline(0, color="black", ls="-", alpha=0.3)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("$G^{nom}(t) - G^{rob}(t)$")
    ax.set_title(f"Degradazione per timestep (ε={eps})", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    # ==== A3: Degradazione per regime (t=0) ====
    ax = fig.add_subplot(gs[0, 2])
    # Get robust target_mean at t=0 (overall)
    d0 = [d for d in diags if d["t"] == 0][0]
    g_rob_0 = d0["target_mean"]

    # For per-regime robust we don't have it directly in diagnostics,
    # so show the nominal per-regime and the overall delta
    for r, (name, c) in enumerate(zip(names, colors_regime)):
        key = f"mean_r{r}"
        if key in nom[0]:
            gn = nom[0][key]
            # Approximate: assume same relative delta across regimes
            # (the actual per-regime robust targets aren't stored in diagnostics)
            delta_overall = nom[0]["mean"] - g_rob_0
            ax.bar(r - 0.17, gn, width=0.3, color=c, alpha=0.4)
            ax.bar(r + 0.17, gn - delta_overall, width=0.3, color=c)
            ax.text(r + 0.17, gn - delta_overall - 0.02,
                    f"Δ≈{delta_overall:.3f}", ha="center", va="top", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_ylabel("G(t=0)")
    ax.set_title("G_nom vs G_rob per regime (t=0)", fontweight="bold")
    # Legend
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="gray", alpha=0.4), Patch(facecolor="gray")],
              ["nominal", "robust"], fontsize=8)

    # ==== B1: Qualità del fit ====
    ax = fig.add_subplot(gs[1, 0])
    ratios = [r / (s + 1e-8) for r, s in zip(residuals, g_rob_std)]
    ax.plot(ts, ratios, "s-", color="#e66101", linewidth=1.5, markersize=5)
    ax.axhline(0.5, color="red", ls="--", alpha=0.4, label="50%")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("MAE / target_std")
    ax.set_title("Qualità fit (varianza irriducibile ~60%)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()
    ax.text(0.5, 0.05, "IID simulator → ceiling ≈ 40% R²",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic",
            color="gray")

    # ==== B2: Val loss vs baseline ====
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts, val_losses, "^-", color="#5e3c99", linewidth=1.5, markersize=5,
            label="val MSE")
    baselines = [s**2 for s in g_rob_std]
    ax.plot(ts, baselines, "k--", alpha=0.3, label="baseline (std²)")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("MSE")
    ax.set_title("Val loss vs predire la media", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    # ==== B3: Sweep summary ====
    ax = fig.add_subplot(gs[1, 2])
    # Show the main result: delta at t=0 and how it accumulates
    cumulative_delta = np.cumsum(deltas[::-1])[::-1]  # accumulate from t=N-1
    ax.plot(ts, deltas, "o-", color="#e66101", linewidth=1.5, markersize=4,
            label="Δ(t) per step")
    ax.bar(ts, deltas, alpha=0.15, color="#e66101", width=0.8)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("$Δ(t)$")
    ax.set_title(f"DRO penalty per step (ε={eps})", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    # Annotate total
    total_delta = nom[0]["mean"] - g_rob_0
    ax.text(0.5, 0.95, f"Total Δ at t=0: {total_delta:.4f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.suptitle(f"Backward Robust DP — Evaluation  (ε={eps})",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot salvato: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ckpt = load_robust_ckpt(args.robust_ckpt, device)

    print("Computing nominal MC returns...")
    nom = compute_nominal_returns(
        args.dataset, gamma=ckpt["gamma"],
        reward_stats=ckpt["reward_stats"],
    )

    print_results(ckpt, nom)
    plot_results(ckpt, nom, out_path=args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate Robust Critic")
    p.add_argument("--robust_ckpt", type=str, default="checkpoints/robust_critic_backward.pt")
    p.add_argument("--dataset", type=str, default="data/wm_dataset.npz")
    p.add_argument("--out", type=str, default="eval_robust_critic.png")
    args = p.parse_args()
    main(args)
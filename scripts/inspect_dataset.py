"""
inspect_dataset.py — Visualizza le differenze tra regimi nel dataset.

Genera un plot diagnostico con:
  Row 1: mid price trajectory per regime (+ mixed episode)
  Row 2: book depth profile medio per regime (bid+ask volume per level)
  Row 3: distribuzioni di spread, imbalance, reward per regime
  Row 4: un episodio mixed con switch point evidenziato

Uso (dalla root del progetto):
  python scripts/inspect_dataset.py
  python scripts/inspect_dataset.py --dataset data/dataset.npz
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

from config import EnvConfig
from simulate import generate_dataset, save_dataset, REGIMES


def ensure_dataset(path: str = "dataset_inspect.npz") -> str:
    if os.path.exists(path):
        print(f"Using existing dataset: {path}")
        return path

    cfg = EnvConfig()
    cfg.N_episodes = 30
    cfg.T_max = 500
    cfg.dataset_path = path
    cfg.mixed_regime_frac = 0.30

    print(f"Generating inspection dataset: {cfg.N_episodes} episodes")
    dataset = generate_dataset(cfg, seed=123, shuffle=False)
    save_dataset(dataset, path)
    return path


def load_dataset(path: str):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def plot_regime_comparison(data: dict, out_path: str) -> None:
    L = EnvConfig.L
    book_flat_dim = 2 * L * 2
    n_regimes = len(REGIMES)

    obs = data["observations"]
    rewards = data["rewards"]
    regimes = data["regimes"]
    ep_ids = data["episode_ids"]
    switch_mask = data["switch_mask"]

    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    # Helper: shade background by regime
    def shade_regimes(ax, t, ep_reg, alpha=0.12):
        changes = np.where(np.diff(ep_reg) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes, [len(ep_reg)]])
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            r = ep_reg[start]
            ax.axvspan(t[start], t[min(end-1, len(t)-1)],
                      alpha=alpha, color=colors[r], linewidth=0)

    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.35,
                           height_ratios=[1, 1, 1, 1.2, 0.4])

    # ─── Row 1: Mid price per regime (3 cols) + depth comparison ───────
    for col in range(n_regimes):
        regime_mask = (regimes == col)
        regime_eps = np.unique(ep_ids[regime_mask])
        for ep in regime_eps:
            ep_mask = (ep_ids == ep)
            if np.all(regimes[ep_mask] == col) and switch_mask[ep_mask].sum() == 0:
                break

        ep_obs = obs[ep_mask]
        t = np.arange(len(ep_obs))
        mids = ep_obs[:, book_flat_dim + 0]

        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, mids, lw=0.6, color=colors[col])
        ax.set_title(f"{REGIMES[col]['name']}\n"
                     f"σ={REGIMES[col]['sigma_mid']}  "
                     f"λ_mo={REGIMES[col]['lambda_mo_buy']:.2f}  "
                     f"p_inf={REGIMES[col]['p_informed']:.2f}",
                     fontsize=9)
        if col == 0:
            ax.set_ylabel("mid price")
        ax.set_xlabel("step")

    # Depth comparison
    ax = fig.add_subplot(gs[0, 3])
    levels = np.arange(L)
    for col in range(n_regimes):
        regime_mask = (regimes == col)
        regime_obs = obs[regime_mask]
        books = regime_obs[:, :book_flat_dim].reshape(-1, 2, L, 2)
        total_vol = books[:, :, :, 1].mean(axis=(0, 1))
        ax.plot(levels, total_vol, 'o-', color=colors[col], label=REGIMES[col]["name"])
    ax.set_xlabel("level")
    ax.set_ylabel("mean volume (bid+ask avg)")
    ax.set_title("Depth comparison", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ─── Row 2: Book depth per regime + table ──────────────────────────
    for col in range(n_regimes):
        regime_mask = (regimes == col)
        regime_obs = obs[regime_mask]
        books = regime_obs[:, :book_flat_dim].reshape(-1, 2, L, 2)
        bid_vols = books[:, 0, :, 1].mean(axis=0)
        ask_vols = books[:, 1, :, 1].mean(axis=0)

        ax = fig.add_subplot(gs[1, col])
        ax.barh(levels - 0.15, bid_vols, height=0.3, color=colors[col], alpha=0.6, label="bid vol")
        ax.barh(levels + 0.15, ask_vols, height=0.3, color=colors[col], alpha=0.9, label="ask vol")
        ax.set_yticks(levels)
        ax.set_yticklabels([f"L{i}" for i in levels])
        ax.invert_yaxis()
        ax.set_xlabel("mean volume")
        ax.set_title(f"Book depth — {REGIMES[col]['name']}", fontsize=9)
        if col == 0:
            ax.legend(fontsize=7)

    # Summary table
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    headers = ["", "low", "mid", "high"]
    metrics = ["L0 vol", "L0-L4 vol", "reward μ", "reward σ", "|inv| μ"]
    rows = []
    for metric in metrics:
        row = [metric]
        for col in range(n_regimes):
            mask = (regimes == col)
            books = obs[mask, :book_flat_dim].reshape(-1, 2, L, 2)
            if metric == "L0 vol":
                row.append(f"{books[:, :, 0, 1].mean():.1f}")
            elif metric == "L0-L4 vol":
                row.append(f"{books[:, :, :5, 1].mean():.1f}")
            elif metric == "reward μ":
                row.append(f"{rewards[mask].mean():.4f}")
            elif metric == "reward σ":
                row.append(f"{rewards[mask].std():.3f}")
            elif metric == "|inv| μ":
                row.append(f"{np.abs(obs[mask, book_flat_dim + 3]).mean():.1f}")
        rows.append(row)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("Key metrics by regime", fontsize=9)

    # ─── Row 3: Distributions ──────────────────────────────────────────
    metric_names = ["spread", "imbalance", "reward", "inventory"]
    metric_indices = [book_flat_dim + 1, book_flat_dim + 2, None, book_flat_dim + 3]

    for m, (name, idx) in enumerate(zip(metric_names, metric_indices)):
        ax = fig.add_subplot(gs[2, m])
        for col in range(n_regimes):
            regime_mask = (regimes == col)
            vals = rewards[regime_mask] if name == "reward" else obs[regime_mask, idx]
            q01, q99 = np.percentile(vals, [1, 99])
            vals_clip = vals[(vals >= q01) & (vals <= q99)]
            ax.hist(vals_clip, bins=50, alpha=0.5, density=True,
                    color=colors[col], label=REGIMES[col]["name"])
        ax.set_title(f"Distribution: {name}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel(name)

    # ─── Row 4: Mixed episode detail (4 panels) ───────────────────────
    mixed_eps = np.unique(ep_ids[switch_mask == 1])
    if len(mixed_eps) > 0:
        ep = mixed_eps[0]
        ep_mask = (ep_ids == ep)
        ep_obs = obs[ep_mask]
        ep_rew = rewards[ep_mask]
        ep_reg = regimes[ep_mask]
        ep_sw = switch_mask[ep_mask]
        t = np.arange(len(ep_obs))

        mids = ep_obs[:, book_flat_dim + 0]
        books = ep_obs[:, :book_flat_dim].reshape(-1, 2, L, 2)
        bid_l0 = books[:, 0, 0, 1]
        ask_l0 = books[:, 1, 0, 1]
        imbalance = ep_obs[:, book_flat_dim + 2]
        sw_points = np.where(ep_sw == 1)[0]

        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor=colors[r], alpha=0.25,
                               label=REGIMES[r]["name"]) for r in range(n_regimes)]

        # Panel 1: Mid price
        ax = fig.add_subplot(gs[3, 0])
        shade_regimes(ax, t, ep_reg)
        ax.plot(t, mids, lw=0.7, color="black")
        for sp in sw_points:
            ax.axvline(sp, color="red", ls="--", lw=1.5, alpha=0.8)
        ax.set_title("Mixed ep — Mid price", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("mid")
        ax.legend(handles=legend_patches, fontsize=7, loc="upper right")

        # Panel 2: L0 volume bid + ask
        ax = fig.add_subplot(gs[3, 1])
        shade_regimes(ax, t, ep_reg)
        # Smooth with rolling mean for readability
        window = 20
        bid_smooth = np.convolve(bid_l0, np.ones(window)/window, mode='same')
        ask_smooth = np.convolve(ask_l0, np.ones(window)/window, mode='same')
        ax.plot(t, bid_smooth, lw=0.8, color="green", alpha=0.8, label="bid L0")
        ax.plot(t, ask_smooth, lw=0.8, color="red", alpha=0.8, label="ask L0")
        for sp in sw_points:
            ax.axvline(sp, color="red", ls="--", lw=1.5, alpha=0.8)
        ax.set_title("Mixed ep — L0 volume (smoothed)", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("volume")
        ax.legend(fontsize=7)

        # Panel 3: Cumulative reward
        ax = fig.add_subplot(gs[3, 2])
        shade_regimes(ax, t, ep_reg)
        cum_rew = np.cumsum(ep_rew)
        ax.plot(t, cum_rew, lw=1, color="black")
        for sp in sw_points:
            ax.axvline(sp, color="red", ls="--", lw=1.5, alpha=0.8)
        ax.set_title("Mixed ep — Cumulative reward", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("Σ reward")

        # Panel 4: Imbalance
        ax = fig.add_subplot(gs[3, 3])
        shade_regimes(ax, t, ep_reg)
        imb_smooth = np.convolve(imbalance, np.ones(window)/window, mode='same')
        ax.plot(t, imb_smooth, lw=0.8, color="purple")
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        for sp in sw_points:
            ax.axvline(sp, color="red", ls="--", lw=1.5, alpha=0.8)
        ax.set_title("Mixed ep — Imbalance (smoothed)", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("imbalance")

    # ─── Row 5: Full parameters table ─────────────────────────────────
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    headers = ["Regime", "σ_mid", "p_inf", "λ_mo", "MO_size",
               "λ_lo", "λ_cancel", "n_trans",
               "spread μ±σ", "reward μ±σ", "|inv| μ"]
    rows = []
    for col in range(n_regimes):
        r = REGIMES[col]
        mask = (regimes == col)
        n = mask.sum()
        sp = obs[mask, book_flat_dim + 1]
        rew = rewards[mask]
        inv = np.abs(obs[mask, book_flat_dim + 3])
        rows.append([
            r["name"],
            f"{r['sigma_mid']:.3f}",
            f"{r['p_informed']:.2f}",
            f"{r['lambda_mo_buy']:.2f}",
            f"{r['mo_size_lambda']:.1f}",
            f"{r['lambda_lo_bid']:.1f}",
            f"{r['lambda_cancel_bid']:.2f}",
            f"{n:,}",
            f"{sp.mean():.3f}±{sp.std():.3f}",
            f"{rew.mean():.4f}±{rew.std():.3f}",
            f"{inv.mean():.1f}",
        ])
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#d5d8dc")

    fig.suptitle("Dataset Inspection — Diversified Regimes + Mixed Episodes",
                 fontsize=14, y=0.98)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


def print_summary(data: dict) -> None:
    regimes = data["regimes"]
    ep_ids = data["episode_ids"]
    switch_mask = data["switch_mask"]

    print(f"\n{'='*60}")
    print(f"Dataset summary")
    print(f"  Total transitions: {len(regimes):,}")
    print(f"  Total episodes:    {len(np.unique(ep_ids))}")
    for i, r in enumerate(REGIMES):
        cnt = (regimes == i).sum()
        print(f"  Regime {i} ({r['name']:10s}): {cnt:,} transitions "
              f"({cnt/len(regimes)*100:.1f}%)")

    n_switches = switch_mask.sum()
    mixed_eps = len(np.unique(ep_ids[switch_mask == 1])) if n_switches > 0 else 0
    print(f"  Regime switches:   {n_switches}")
    print(f"  Mixed episodes:    {mixed_eps}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--out", type=str, default="inspect_regimes.png")
    args = parser.parse_args()

    if args.dataset and os.path.exists(args.dataset):
        path = args.dataset
    else:
        path = ensure_dataset("dataset_inspect.npz")

    data = load_dataset(path)
    print_summary(data)
    plot_regime_comparison(data, args.out)
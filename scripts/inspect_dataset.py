"""
inspect_dataset.py — Visualizza un episodio per ciascun regime del dataset.

Genera un plot con 3 colonne (regimi) x 3 righe (mid, inventory, reward).

Uso (dalla root del progetto):
  python scripts/inspect_dataset.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add simulator directory to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

from config import EnvConfig
from simulate import run_regime, save_dataset, REGIMES


def ensure_dataset(path: str = "dataset_small.npz") -> str:
    if os.path.exists(path):
        print(f"Using existing dataset: {path}")
        return path

    cfg = EnvConfig()
    cfg.N_episodes = 9   # 3 per regime
    cfg.T_max = 500
    cfg.dataset_path = path

    print(f"Generating inspection dataset: {cfg.N_episodes} episodes x {cfg.T_max} steps")

    rng = np.random.default_rng(123)
    all_parts = []
    eps_per_regime = cfg.N_episodes // len(REGIMES)

    for i in range(len(REGIMES)):
        part = run_regime(cfg, eps_per_regime, regime_idx=i, rng=rng, verbose=True)
        all_parts.append(part)

    dataset = {
        key: np.concatenate([p[key] for p in all_parts], axis=0)
        for key in all_parts[0].keys()
    }

    np.savez_compressed(path, **dataset)
    print(f"Inspection dataset saved to {path}")
    return path


def load_and_inspect(path: str) -> None:
    cfg = EnvConfig()
    L = cfg.L
    book_flat_dim = 2 * L * 2

    data = np.load(path)
    obs     = data["observations"]
    rewards = data["rewards"]
    regimes = data["regimes"]
    N = len(rewards)

    print(f"\nDataset: {path}")
    print(f"  Total transitions : {N}")
    print(f"  obs_dim           : {obs.shape[1]}")
    for i, r in enumerate(REGIMES):
        cnt = (regimes == i).sum()
        print(f"  regime {i} ({r['name']:10s}): {cnt} transitions")

    n_regimes = len(REGIMES)
    fig, axes = plt.subplots(3, n_regimes, figsize=(5 * n_regimes, 9))
    fig.suptitle("One episode per regime", fontsize=13)

    for col, regime in enumerate(REGIMES):
        idx = np.where(regimes == col)[0]
        if len(idx) == 0:
            continue

        ep_len  = min(cfg.T_max, len(idx))
        ep_obs  = obs[idx[:ep_len]]
        ep_rews = rewards[idx[:ep_len]]
        t = np.arange(ep_len)

        mids        = ep_obs[:, book_flat_dim + 0]
        inventories = ep_obs[:, book_flat_dim + 3]

        ax = axes[0, col]
        ax.plot(t, mids, lw=0.8)
        ax.set_title(f"regime: {regime['name']}\n"
                     f"σ={regime['sigma_mid']}  p_inf={regime['p_informed']}", fontsize=9)
        ax.set_ylabel("mid price" if col == 0 else "")
        ax.tick_params(labelbottom=False)

        ax = axes[1, col]
        ax.plot(t, inventories, lw=0.8, color="tab:orange")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel("inventory" if col == 0 else "")
        ax.tick_params(labelbottom=False)

        ax = axes[2, col]
        ax.plot(t, ep_rews, lw=0.8, color="tab:green")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel("reward" if col == 0 else "")
        ax.set_xlabel("time step")

        print(f"\nRegime {col} ({regime['name']}) — first {ep_len} transitions")
        print(f"  mid   : mean={mids.mean():.3f}  std={mids.std():.3f}")
        print(f"  inv   : mean={inventories.mean():.2f}  std={inventories.std():.2f}  "
              f"range=[{inventories.min():.1f}, {inventories.max():.1f}]")
        print(f"  reward: mean={ep_rews.mean():.4f}  std={ep_rews.std():.4f}  "
              f"range=[{ep_rews.min():.3f}, {ep_rews.max():.3f}]")
        print(f"  neg rewards: {(ep_rews < 0).sum()} / {len(ep_rews)}")

    plt.tight_layout()
    out = "inspect_regimes.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    path = ensure_dataset("dataset_small.npz")
    load_and_inspect(path)
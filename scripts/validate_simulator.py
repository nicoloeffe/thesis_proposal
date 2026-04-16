#!/usr/bin/env python3
"""
validate_simulator.py — Validazione 3×3 del simulatore LOB.

Produce una singola figura 3×3 + scorecard testuale.

Layout:
  Row 1 — Book microstructure:   Volume profile | Depth & L0 ratio | L0 vuoto %
  Row 2 — Price dynamics & MM:   Returns dist   | Imbalance & Inventory  | σ vs Spread vs MM
  Row 3 — Actions & episodes:    k distribution | PNL      | Mixed episode

Riferimenti:
  [CST10] Cont, Stoikov, Talreja 2010 — depth ratio 3-5x
  [BMP02] Bouchaud, Mézard, Potters 2002 — book shape
  [C01]   Cont 2001 — fat tails, vol clustering
  [AS08]  Avellaneda, Stoikov 2008 — optimal spread

Uso:
  python validate_simulator.py --dataset dataset.npz
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
from config import EnvConfig
from simulate import REGIMES, generate_dataset, save_dataset

# ---------------------------------------------------------------------------
# Constants & Style
# ---------------------------------------------------------------------------
L = EnvConfig.L
BFD = 2 * L * 2  # book flat dim
TICK = EnvConfig.tick_size
N_R = len(REGIMES)
RNAMES = [r["name"].replace("_", " ") for r in REGIMES]

# Palette
C = ["#1976D2", "#F57C00", "#388E3C"]   # blue, orange, green — muted, professional
C_LIGHT = ["#BBDEFB", "#FFE0B2", "#C8E6C9"]
BG = "#FAFAFA"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
    "legend.framealpha": 0.8,
})


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract(data: dict) -> dict:
    obs = data["observations"]
    books = obs[:, :BFD].reshape(-1, 2, L, 2)
    return dict(
        books=books,
        mid=obs[:, BFD], spread=obs[:, BFD+1],
        imbalance=obs[:, BFD+2], inventory=obs[:, BFD+3],
        actions=data["actions"], rewards=data["rewards"],
        regimes=data["regimes"], ep_ids=data["episode_ids"],
        switch_mask=data["switch_mask"],
    )


def episode_returns(d):
    """Calcola i ritorni purificati, evitando salti tra episodi e regimi diversi."""
    rets = {r: [] for r in range(len(REGIMES))}
    for ep in np.unique(d["ep_ids"]):
        mask = d["ep_ids"] == ep
        mid = d["mid"][mask]
        reg = d["regimes"][mask]
        if len(mid) < 2: 
            continue
        # Differenza ESATTAMENTE interna all'episodio
        ep_rets = np.diff(mid)
        # Allineiamo il regime al ritorno (prendiamo il regime dello step di arrivo)
        ep_reg = reg[1:]
        for r in range(len(REGIMES)):
            rets[r].append(ep_rets[ep_reg == r])
            
    # Uniamo tutte le liste in array numpy
    return {r: np.concatenate(v) if len(v) > 0 else np.array([]) for r, v in rets.items()}


def acf(x, max_lag=30):
    x = x - x.mean()
    n, v = len(x), np.var(x)
    if v < 1e-15:
        return np.zeros(max_lag)
    return np.array([np.sum(x[:n-l] * x[l:]) / (n * v) for l in range(1, max_lag+1)])


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def make_figure(d, out_path):
    rets = episode_returns(d)
    books, regimes, actions = d["books"], d["regimes"], d["actions"]
    spread, mid = d["spread"], d["mid"]

    fig = plt.figure(figsize=(17, 14.5))
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.38, wspace=0.30,
        left=0.06, right=0.97, top=0.92, bottom=0.04,
    )
    fig.suptitle(
        "LOB Simulator Validation",
        fontsize=16, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.945,
        "[CST10] Cont-Stoikov-Talreja 2010    [C01] Cont 2001    "
        "[AS08] Avellaneda-Stoikov 2008    [BMP02] Bouchaud-Mézard-Potters 2002",
        ha="center", fontsize=7.5, color="#666666", style="italic",
    )

    scores = {}

    # ═══════════════════════════════════════════════════════════════════
    # ROW 1 — BOOK MICROSTRUCTURE
    # ═══════════════════════════════════════════════════════════════════

    # (0,0) Volume profile
    ax = fig.add_subplot(gs[0, 0])
    for r in range(N_R):
        vols = books[regimes == r, :, :, 1].mean(axis=(0, 1))
        ax.plot(range(L), vols, "o-", color=C[r], label=RNAMES[r],
                lw=2, markersize=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_title("Volume profile per livello\n[BMP02]", fontweight="bold")
    ax.set_xlabel("Level (L0 = best)")
    ax.set_ylabel("Mean volume")
    ax.legend()

    # (0,1) Depth + L0 bars
    ax = fig.add_subplot(gs[0, 1])
    depths, l0s = [], []
    for r in range(N_R):
        m = regimes == r
        depths.append(books[m, :, :, 1].sum(axis=(1, 2)).mean())
        l0s.append(books[m, :, 0, 1].mean())
    x = np.arange(N_R)
    ax.bar(x - 0.18, depths, 0.34, label="Total depth", color=C_LIGHT,
           edgecolor=C, linewidth=0.8)
    ax.bar(x + 0.18, l0s, 0.34, label="L0 volume", color=C,
           edgecolor=C, linewidth=0.8)
    dr = depths[0] / max(depths[-1], 1e-6)
    lr = l0s[0] / max(l0s[-1], 1e-6)
    scores["depth_ratio"] = dr
    scores["l0_ratio"] = lr
    ax.set_xticks(x)
    ax.set_xticklabels(RNAMES)
    ax.set_title(f"Depth ratio: {dr:.1f}×  |  L0 ratio: {lr:.1f}×\n[CST10] target 1.5–5×",
                 fontweight="bold")
    ax.set_ylabel("Volume")
    ax.legend(loc="upper right")

    # (0,2) L0 empty %
    ax = fig.add_subplot(gs[0, 2])
    pcts = []
    for r in range(N_R):
        v = books[regimes == r, :, 0, 1]
        pcts.append((v == 0).sum() / v.size * 100)
    bars = ax.bar(RNAMES, pcts, color=C, edgecolor="white", linewidth=1.5)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width()/2, p + 1.2, f"{p:.1f}%",
                ha="center", fontsize=10, fontweight="bold", color="#333")
    ax.set_title("% time L0 empty\nLow <15% | Mid 20–35% | High 35–55%", fontweight="bold")
    ax.set_ylabel("%")
    scores["l0_empty"] = pcts

    # ═══════════════════════════════════════════════════════════════════
    # ROW 2 — PRICE DYNAMICS & MARKET MAKER
    # ═══════════════════════════════════════════════════════════════════

    # (1,0) Returns distribution
    ax = fig.add_subplot(gs[1, 0])
    kurtoses = {}
    for r in range(N_R):
        if r not in rets:
            continue
        rv = rets[r]
        K = stats.kurtosis(rv, fisher=True)
        kurtoses[r] = K
        
        # Usiamo i rendimenti RAW, non standardizzati
        kde = stats.gaussian_kde(rv, bw_method=0.8)
        
        # Generiamo la x su una scala realistica per i rendimenti veri
        xx = np.linspace(-0.15, 0.15, 300) 
        
        # Plottiamo e mettiamo la deviazione standard reale nella legenda
        ax.plot(xx, kde(xx), color=C[r], lw=2, label=f"{RNAMES[r]} (σ={rv.std():.3f})")
        ax.fill_between(xx, 0, kde(xx), color=C[r], alpha=0.15)

    ax.set_xlim(-0.15, 0.15) # Zoom sui veri movimenti di prezzo!
    ax.set_title("Raw Returns Distribution\nRandom Walk Volatility Dispersion", fontweight="bold")
    ax.set_xlabel("Actual Price Change")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    scores["kurtosis"] = kurtoses
# # ═══════════════════════════════════════════════════════════════════
    # (1,1) Inventory & Imbalance (Split in due mini-grafici orizzontali)
    # ═══════════════════════════════════════════════════════════════════
    
    # Creiamo una mini-griglia 2 righe x 1 colonna dentro lo slot centrale
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.5)
    
    # --- Top: Inventory ---
    ax_inv = fig.add_subplot(inner_gs[0, 0])
    for r in range(N_R):
        inv = d["inventory"][d["regimes"] == r]
        if len(inv) < 2: continue
        kde = stats.gaussian_kde(inv, bw_method=0.4)
        xx = np.linspace(-15, 15, 200) # Range corretto per inventario
        ax_inv.plot(xx, kde(xx), color=C[r], lw=1.5)
        ax_inv.fill_between(xx, 0, kde(xx), color=C[r], alpha=0.15)
        
    ax_inv.axvline(0, color="black", ls="--", alpha=0.5, lw=1)
    ax_inv.set_title("MM Inventory (Risk Position)", fontsize=9, fontweight="bold")
    ax_inv.set_xlim(-15, 15)
    ax_inv.tick_params(axis='both', labelsize=7)

    # --- Bottom: Imbalance ---
    ax_imb = fig.add_subplot(inner_gs[1, 0])
    for r in range(N_R):
        imb = d["imbalance"][d["regimes"] == r]
        if len(imb) < 2: continue
        kde = stats.gaussian_kde(imb, bw_method=0.3)
        xx = np.linspace(-1, 1, 200) # Range corretto per imbalance
        ax_imb.plot(xx, kde(xx), color=C[r], lw=1.5)
        ax_imb.fill_between(xx, 0, kde(xx), color=C[r], alpha=0.15)
        
    ax_imb.axvline(0, color="black", ls="--", alpha=0.5, lw=1)
    ax_imb.set_title("Order Flow Imbalance (Toxicity)", fontsize=9, fontweight="bold")
    ax_imb.set_xlim(-1, 1)
    ax_imb.set_xticks([-0.75, -0.25, 0.25, 0.75])
    ax_imb.tick_params(axis='both', labelsize=7)

    # (1,2) σ vs market spread vs MM spread
    ax = fig.add_subplot(gs[1, 2])
    sigmas, sp_mkt, sp_mm = [], [], []
    for r in range(N_R):
        sigmas.append(rets[r].std() if r in rets else 0)
        sp_mkt.append(spread[regimes == r].mean())
        k_all = actions[regimes == r, 0] + actions[regimes == r, 1]
        sp_mm.append(k_all.mean() * TICK)
    x = np.arange(N_R)
    w = 0.24
    ax.bar(x - w, sigmas, w, label="σ (return std)", color="#5C6BC0")
    ax.bar(x, sp_mkt, w, label="Market spread", color="#EF5350")
    ax.bar(x + w, sp_mm, w, label="MM spread (A-S)", color="#26A69A")
    for i in range(N_R):
        if sigmas[i] > 0:
            rm = sp_mkt[i] / sigmas[i]
            rmm = sp_mm[i] / sigmas[i]
            y = max(sp_mkt[i], sp_mm[i])
            ax.text(i, y + 0.003, f"mkt {rm:.1f}×\nmm {rmm:.1f}×",
                    ha="center", fontsize=7, fontweight="bold", color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(RNAMES)
    ax.set_title("σ  vs  Market spread  vs  MM spread\n[AS08] spread scales with volatility",
                 fontweight="bold")
    ax.set_ylabel("Price units")
    ax.legend(fontsize=7, loc="upper left")
    scores["spread_sigma"] = {r: sp_mkt[r] / max(sigmas[r], 1e-10) for r in range(N_R)}

    # ═══════════════════════════════════════════════════════════════════
    # ROW 3 — ACTIONS & MIXED EPISODE
    # ═══════════════════════════════════════════════════════════════════

    # (2,0) k distribution
    ax = fig.add_subplot(gs[2, 0])
    for r in range(N_R):
        k_all = np.concatenate([actions[regimes == r, 0], actions[regimes == r, 1]])
        ax.hist(k_all, bins=np.arange(0.5, L + 1.5, 1), alpha=0.50, density=True,
                color=C[r], edgecolor=C[r], linewidth=0.5,
                label=f"{RNAMES[r]}  μ={k_all.mean():.1f}")
    ax.set_title("A-S quote offset k (ticks from mid)\n[AS08] k increases with σ",
                 fontweight="bold")
    ax.set_xlabel("k (ticks)")
    ax.set_ylabel("Density")
    ax.legend()
    # ═══════════════════════════════════════════════════════════════════
    # (2,1) PnL / Reward Distribution
    # ═══════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2, 1])
    for r in range(N_R):
        rew = d["rewards"][d["regimes"] == r]
        if len(rew) < 10: continue
        kde = stats.gaussian_kde(rew, bw_method=0.4)
        
        # Filtriamo gli estremi rari (0.5% e 99.5%) per evitare che l'asse X si schiacci troppo
        q_low, q_high = np.percentile(rew, [0.5, 99.5]) 
        xx = np.linspace(q_low, q_high, 200)
        
        ax.plot(xx, kde(xx), color=C[r], lw=2, label=f"{RNAMES[r]} (μ={rew.mean():.3f})")
        ax.fill_between(xx, 0, kde(xx), color=C[r], alpha=0.2)

    ax.axvline(0, color="black", ls="--", alpha=0.6)
    ax.set_title("Market Maker Step-Reward (PnL)\nProfitability across regimes", fontweight="bold")
    ax.set_xlabel("Reward per step")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    # (2,2) Mixed episode — mid price with regime shading
    ax = fig.add_subplot(gs[2, 2])
    mixed_eps = np.unique(d["ep_ids"][d["switch_mask"] == 1])
    if len(mixed_eps) > 0:
        ep = mixed_eps[0]
        m = d["ep_ids"] == ep
        t = np.arange(m.sum())
        ep_mid = d["mid"][m]
        ep_reg = d["regimes"][m]
        sw = np.where(d["switch_mask"][m] == 1)[0]

        # Shade regimes
        changes = np.where(np.diff(ep_reg) != 0)[0] + 1
        bounds = np.concatenate([[0], changes, [len(ep_reg)]])
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            ax.axvspan(t[s], t[min(e - 1, len(t) - 1)],
                       alpha=0.15, color=C[ep_reg[s]], lw=0)

        ax.plot(t, ep_mid, lw=0.6, color="#212121")
        for s in sw:
            ax.axvline(s, color="#D32F2F", ls="--", lw=1.2, alpha=0.7)

        patches = [Patch(facecolor=C[r], alpha=0.3, label=RNAMES[r]) for r in range(N_R)]
        ax.legend(handles=patches, loc="best", fontsize=7)
    ax.set_title("Mixed episode — regime switching\nred dashes = switch points",
                 fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mid price")

    # ─── Row labels ───
    row_labels = [
        "Book Microstructure",
        "Price Dynamics & Market Maker",
        "Actions & Regime Switching",
    ]
    for i, label in enumerate(row_labels):
        fig.text(0.01, 0.88 - i * 0.325, label, rotation=90,
                 fontsize=10, fontweight="bold", color="#888", va="center")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")
    return scores



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="validation")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading: {args.dataset}")
        raw = np.load(args.dataset)
        data = {k: raw[k] for k in raw.files}
    else:
        print(f"Generating {args.episodes} episodes (unshuffled)...")
        cfg = EnvConfig()
        cfg.N_episodes = args.episodes
        cfg.mixed_regime_frac = 0.30
        data = generate_dataset(cfg, seed=args.seed, shuffle=False)
        save_dataset(data, os.path.join(args.out_dir, "dataset.npz"))

    d = extract(data)
    n = len(d["rewards"])
    print(f"Dataset: {n:,} transitions, {len(np.unique(d['ep_ids']))} episodes\n")

    scores = make_figure(d, os.path.join(args.out_dir, "simulator_validation.png"))


if __name__ == "__main__":
    main()
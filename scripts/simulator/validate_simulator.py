#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parent.parent / "simulator"
sys.path.insert(0, str(SIM_ROOT))
from config import EnvConfig
from simulate import REGIMES, generate_dataset, save_dataset

CFG0 = EnvConfig()
L = CFG0.L
TICK = CFG0.tick_size
N_R = len(REGIMES)
RNAMES = [r["name"].replace("_", " ") for r in REGIMES]
BFD = 2 * L * 2

C = ["#1976D2", "#F57C00", "#388E3C"]
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
    "legend.framealpha": 0.85,
})


def sort_dataset_temporally(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "episode_ids" not in data or "timesteps" not in data:
        return data
    order = np.lexsort((data["timesteps"], data["episode_ids"]))
    out = {}
    n = len(order)
    for k, v in data.items():
        if isinstance(v, np.ndarray) and len(v.shape) >= 1 and v.shape[0] == n:
            out[k] = v[order]
        else:
            out[k] = v
    return out


def extract(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    obs = data["observations"]
    books = obs[:, :BFD].reshape(-1, 2, L, 2)
    return {
        "books": books,
        "mid": obs[:, BFD],
        "spread": obs[:, BFD + 1],
        "imbalance": obs[:, BFD + 2],
        "inventory": obs[:, BFD + 3],
        "actions": data["actions"],
        "rewards": data["rewards"],
        "regimes": data["regimes"],
        "ep_ids": data["episode_ids"],
        "switch_mask": data["switch_mask"],
        "timesteps": data["timesteps"],
    }


def episode_returns(d: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    rets = {r: [] for r in range(N_R)}
    for ep in np.unique(d["ep_ids"]):
        mask = d["ep_ids"] == ep
        mid = d["mid"][mask]
        reg = d["regimes"][mask]
        if len(mid) < 2:
            continue
        ep_rets = np.diff(mid)
        ep_reg = reg[1:]
        for r in range(N_R):
            rr = ep_rets[ep_reg == r]
            if len(rr) > 0:
                rets[r].append(rr)
    return {r: np.concatenate(v) if len(v) else np.array([]) for r, v in rets.items()}


def lo_profile_from_regime(regime: dict) -> np.ndarray:
    levels = np.arange(L)
    alpha = regime["lo_alpha"]
    f0 = regime["lo_best_supply"]
    exp_profile = np.exp(-alpha * levels)
    shape = exp_profile.copy()
    shape[0] = f0
    shape *= exp_profile.sum() / max(shape.sum(), 1e-12)
    return shape


def safe_kde(ax, values, color, label=None, xgrid=None, bw=0.4, fill_alpha=0.15, lw=2.0):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or np.std(x) < 1e-12:
        return False
    try:
        kde = stats.gaussian_kde(x, bw_method=bw)
    except Exception:
        return False
    if xgrid is None:
        lo, hi = np.percentile(x, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = x.mean() - 1.0, x.mean() + 1.0
        xgrid = np.linspace(lo, hi, 250)
    y = kde(xgrid)
    ax.plot(xgrid, y, color=color, lw=lw, label=label)
    ax.fill_between(xgrid, 0, y, color=color, alpha=fill_alpha)
    return True


def stack_switch_windows(values: np.ndarray, ep_ids: np.ndarray, switch_mask: np.ndarray, pre=25, post=60):
    rows = []
    values = np.asarray(values, dtype=float)
    for ep in np.unique(ep_ids):
        idx = np.where(ep_ids == ep)[0]
        if len(idx) == 0:
            continue
        local_switches = np.where(switch_mask[idx] == 1)[0]
        for s in local_switches:
            row = np.full(pre + post + 1, np.nan)
            a = max(0, s - pre)
            b = min(len(idx), s + post + 1)
            dest_a = pre - (s - a)
            dest_b = dest_a + (b - a)
            row[dest_a:dest_b] = values[idx[a:b]]
            rows.append(row)
    x = np.arange(-pre, post + 1)
    if not rows:
        return x, np.full_like(x, np.nan, dtype=float), np.full_like(x, np.nan, dtype=float), 0
    arr = np.vstack(rows)
    mean = np.nanmean(arr, axis=0)
    count = np.sum(~np.isnan(arr), axis=0)
    std = np.nanstd(arr, axis=0)
    se = std / np.sqrt(np.maximum(count, 1))
    return x, mean, se, arr.shape[0]


def binned_mean(x, y, bins):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centers, means = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (x >= a) & (x < b)
        if m.sum() == 0:
            centers.append(0.5 * (a + b) if np.isfinite(a) and np.isfinite(b) else np.nan)
            means.append(np.nan)
        else:
            centers.append(np.median(x[m]))
            means.append(np.mean(y[m]))
    return np.array(centers), np.array(means)


def compute_score_rows(d, rets):
    books = d["books"]
    regimes = d["regimes"]
    actions = d["actions"]
    rows = []
    for r, reg in enumerate(REGIMES):
        m = regimes == r
        prof = books[m, :, :, 1].mean(axis=(0, 1))
        rows.append({
            "regime": reg["name"],
            "latent_sigma": reg["sigma_mid"],
            "empirical_sigma": float(rets[r].std()) if len(rets[r]) else np.nan,
            "lambda_cancel_per_share": reg["lambda_cancel_per_share"],
            "mean_total_depth": float(books[m, :, :, 1].sum(axis=(1, 2)).mean()),
            "mean_L0_volume": float(books[m, :, 0, 1].mean()),
            "L0_empty_pct": float(((books[m, :, 0, 1] == 0).sum() / max(books[m, :, 0, 1].size, 1)) * 100.0),
            "market_spread_mean": float(d["spread"][m].mean()),
            "mm_spread_mean": float((actions[m, 0] + actions[m, 1]).mean() * TICK),
            "k_mean": float(np.concatenate([actions[m, 0], actions[m, 1]]).mean()),
            "reward_mean": float(d["rewards"][m].mean()),
            "empirical_peak_level": int(np.argmax(prof)),
        })
    return rows


def save_scorecard_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_overview_figure(d, out_path):
    books, regimes, actions = d["books"], d["regimes"], d["actions"]
    spread = d["spread"]
    rets = episode_returns(d)

    fig = plt.figure(figsize=(17, 14.5))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.30,
                           left=0.06, right=0.97, top=0.92, bottom=0.04)
    fig.suptitle("LOB Simulator Validation — Overview", fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.945,
             "Stylized facts: book shape, depth, L0 fragility, returns, spreads, actions and mixed episodes",
             ha="center", fontsize=8, color="#666666", style="italic")

    ax = fig.add_subplot(gs[0, 0])
    for r in range(N_R):
        vols = books[regimes == r, :, :, 1].mean(axis=(0, 1))
        ax.plot(range(L), vols, "o-", color=C[r], label=RNAMES[r], lw=2, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_title("Volume profile per livello", fontweight="bold")
    ax.set_xlabel("Level (L0 = best)")
    ax.set_ylabel("Mean volume")
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    depths, l0s = [], []
    for r in range(N_R):
        m = regimes == r
        depths.append(books[m, :, :, 1].sum(axis=(1, 2)).mean())
        l0s.append(books[m, :, 0, 1].mean())
    x = np.arange(N_R)
    ax.bar(x - 0.18, depths, 0.34, label="Total depth", color=C_LIGHT, edgecolor=C, linewidth=0.8)
    ax.bar(x + 0.18, l0s, 0.34, label="L0 volume", color=C, edgecolor=C, linewidth=0.8)
    dr = depths[0] / max(depths[-1], 1e-9)
    lr = l0s[0] / max(l0s[-1], 1e-9)
    ax.set_xticks(x)
    ax.set_xticklabels(RNAMES)
    ax.set_title(f"Depth ratio low/high = {dr:.1f}x | L0 ratio = {lr:.1f}x", fontweight="bold")
    ax.set_ylabel("Volume")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[0, 2])
    pcts = []
    for r in range(N_R):
        v = books[regimes == r, :, 0, 1]
        pcts.append((v == 0).sum() / max(v.size, 1) * 100)
    bars = ax.bar(RNAMES, pcts, color=C, edgecolor="white", linewidth=1.5)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.8, f"{p:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Percent time L0 is empty", fontweight="bold")
    ax.set_ylabel("%")

    ax = fig.add_subplot(gs[1, 0])
    for r in range(N_R):
        rv = rets[r]
        if len(rv) == 0:
            continue
        xx = np.linspace(-0.15, 0.15, 300)
        safe_kde(ax, rv, C[r], label=f"{RNAMES[r]} (std={rv.std():.3f})", xgrid=xx, bw=0.8, fill_alpha=0.15)
    ax.set_xlim(-0.15, 0.15)
    ax.set_title("Raw returns distribution", fontweight="bold")
    ax.set_xlabel("Price change")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], hspace=0.45)
    ax_inv = fig.add_subplot(inner[0, 0])
    for r in range(N_R):
        inv = d["inventory"][regimes == r]
        xx = np.linspace(np.percentile(inv, 0.5), np.percentile(inv, 99.5), 250) if len(inv) else None
        safe_kde(ax_inv, inv, C[r], xgrid=xx, bw=0.4, fill_alpha=0.15, lw=1.6)
    ax_inv.axvline(0, color="black", ls="--", alpha=0.5, lw=1)
    ax_inv.set_title("MM inventory", fontsize=9, fontweight="bold")
    ax_inv.tick_params(axis="both", labelsize=7)

    ax_imb = fig.add_subplot(inner[1, 0])
    for r in range(N_R):
        imb = d["imbalance"][regimes == r]
        xx = np.linspace(-1.0, 1.0, 250)
        safe_kde(ax_imb, imb, C[r], xgrid=xx, bw=0.3, fill_alpha=0.15, lw=1.6)
    ax_imb.axvline(0, color="black", ls="--", alpha=0.5, lw=1)
    ax_imb.set_xlim(-1, 1)
    ax_imb.set_title("Order-flow imbalance", fontsize=9, fontweight="bold")
    ax_imb.tick_params(axis="both", labelsize=7)

    ax = fig.add_subplot(gs[1, 2])
    sigmas, sp_mkt, sp_mm = [], [], []
    for r in range(N_R):
        sigmas.append(rets[r].std() if len(rets[r]) else 0.0)
        sp_mkt.append(spread[regimes == r].mean())
        k_all = actions[regimes == r, 0] + actions[regimes == r, 1]
        sp_mm.append(k_all.mean() * TICK)
    x = np.arange(N_R)
    w = 0.24
    ax.bar(x - w, sigmas, w, label="Empirical sigma", color="#5C6BC0")
    ax.bar(x, sp_mkt, w, label="Market spread", color="#EF5350")
    ax.bar(x + w, sp_mm, w, label="MM spread", color="#26A69A")
    ax.set_xticks(x)
    ax.set_xticklabels(RNAMES)
    ax.set_title("Empirical sigma vs market and MM spread", fontweight="bold")
    ax.set_ylabel("Price units")
    ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(gs[2, 0])
    for r in range(N_R):
        k_all = np.concatenate([actions[regimes == r, 0], actions[regimes == r, 1]])
        ax.hist(k_all, bins=np.arange(0.5, L + 1.5, 1), alpha=0.50, density=True,
                color=C[r], edgecolor=C[r], linewidth=0.5, label=f"{RNAMES[r]} mu={k_all.mean():.1f}")
    ax.set_title("A-S quote offset k", fontweight="bold")
    ax.set_xlabel("k (ticks from mid)")
    ax.set_ylabel("Density")
    ax.legend()

    ax = fig.add_subplot(gs[2, 1])
    for r in range(N_R):
        rew = d["rewards"][regimes == r]
        if len(rew) < 20:
            continue
        lo, hi = np.percentile(rew, [0.5, 99.5])
        xx = np.linspace(lo, hi, 250)
        safe_kde(ax, rew, C[r], label=f"{RNAMES[r]} (mu={rew.mean():.3f})", xgrid=xx, bw=0.4, fill_alpha=0.18)
    ax.axvline(0, color="black", ls="--", alpha=0.6)
    ax.set_title("Market-maker step reward", fontweight="bold")
    ax.set_xlabel("Reward per step")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[2, 2])
    mixed_eps = np.unique(d["ep_ids"][d["switch_mask"] == 1])
    if len(mixed_eps) > 0:
        counts = {ep: int((d["switch_mask"][d["ep_ids"] == ep] == 1).sum()) for ep in mixed_eps}
        ep = max(counts, key=counts.get)
        m = d["ep_ids"] == ep
        t = np.arange(m.sum())
        ep_mid = d["mid"][m]
        ep_reg = d["regimes"][m]
        sw = np.where(d["switch_mask"][m] == 1)[0]
        changes = np.where(np.diff(ep_reg) != 0)[0] + 1
        bounds = np.concatenate([[0], changes, [len(ep_reg)]])
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            ax.axvspan(t[s], t[min(e - 1, len(t) - 1)], alpha=0.15, color=C[ep_reg[s]], lw=0)
        ax.plot(t, ep_mid, lw=0.8, color="#212121")
        for s in sw:
            ax.axvline(s, color="#D32F2F", ls="--", lw=1.2, alpha=0.7)
        patches = [Patch(facecolor=C[r], alpha=0.3, label=RNAMES[r]) for r in range(N_R)]
        ax.legend(handles=patches, loc="best", fontsize=7)
    ax.set_title("Mixed episode with regime shading", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mid price")

    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def make_mechanism_figure(d, out_path, pre=25, post=60):
    books, regimes, actions = d["books"], d["regimes"], d["actions"]
    rets = episode_returns(d)
    depth_total = books[:, :, :, 1].sum(axis=(1, 2))

    fig = plt.figure(figsize=(18, 14.5))
    gs = gridspec.GridSpec(
        3, 3, figure=fig, hspace=0.42, wspace=0.30,
        left=0.06, right=0.97, top=0.92, bottom=0.05
    )
    fig.suptitle("LOB Simulator Validation — Mechanism Checks", fontsize=16, fontweight="bold", y=0.97)
    fig.text(
        0.5, 0.945,
        "Checks of internal mechanisms: LO profile, endogenous depth, volatility transmission, switch relaxation, inventory control",
        ha="center", fontsize=8, color="#666666", style="italic"
    )

    # ------------------------------------------------------------------
    # Row 1
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for r, reg in enumerate(REGIMES):
        prof = lo_profile_from_regime(reg)
        prof = prof / max(prof.sum(), 1e-12)
        ax.plot(range(L), prof, "o--", color=C[r], lw=2, markersize=4, label=RNAMES[r])
    ax.set_title("Theoretical LO intensity shape", fontweight="bold")
    ax.set_xlabel("Level")
    ax.set_ylabel("Normalized intensity")
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    for r in range(N_R):
        prof = books[regimes == r, :, :, 1].mean(axis=(0, 1))
        prof = prof / max(prof.sum(), 1e-12)
        peak = int(np.argmax(prof))
        ax.plot(range(L), prof, "o-", color=C[r], lw=2, markersize=4, label=f"{RNAMES[r]} peak=L{peak}")
    ax.set_title("Empirical mean book shape", fontweight="bold")
    ax.set_xlabel("Level")
    ax.set_ylabel("Normalized mean volume")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    cancel_rates = [reg["lambda_cancel_per_share"] for reg in REGIMES]
    depth_means = [depth_total[regimes == r].mean() for r in range(N_R)]
    l0_means = [books[regimes == r, :, 0, 1].mean() for r in range(N_R)]

    ax.plot(cancel_rates, depth_means, "o-", color="#455A64", lw=2, label="Total depth")
    ax.plot(cancel_rates, l0_means, "s--", color="#D81B60", lw=2, label="L0 volume")
    for r in range(N_R):
        ax.scatter(cancel_rates[r], depth_means[r], s=80, color=C[r], zorder=3)
        ax.text(cancel_rates[r], depth_means[r], f" {RNAMES[r]}", va="center", fontsize=8)
    ax.set_title("Endogenous depth vs cancellation rate", fontweight="bold")
    ax.set_xlabel("lambda_cancel_per_share")
    ax.set_ylabel("Mean volume")
    ax.legend()

    # ------------------------------------------------------------------
    # Row 2
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    latent = [reg["sigma_mid"] for reg in REGIMES]
    empirical = [rets[r].std() if len(rets[r]) else np.nan for r in range(N_R)]
    x = np.arange(N_R)
    w = 0.35
    ax.bar(x - w / 2, latent, width=w, color="#90CAF9", edgecolor="#1976D2", label="Latent sigma_mid")
    ax.bar(x + w / 2, empirical, width=w, color="#A5D6A7", edgecolor="#388E3C", label="Empirical return std")
    ax.set_xticks(x)
    ax.set_xticklabels(RNAMES)
    ax.set_title("Latent sigma vs observed volatility", fontweight="bold")
    ax.set_ylabel("Price units")
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    xw, mu, se, nsw = stack_switch_windows(depth_total, d["ep_ids"], d["switch_mask"], pre=pre, post=post)
    ax.plot(xw, mu, color="#3949AB", lw=2)
    ax.fill_between(xw, mu - se, mu + se, color="#9FA8DA", alpha=0.35)
    ax.axvline(0, color="#D32F2F", ls="--", lw=1.2)
    ax.set_title(f"Switch event study: total depth (n={nsw})", fontweight="bold")
    ax.set_xlabel("Steps from switch")
    ax.set_ylabel("Depth")

    ax = fig.add_subplot(gs[1, 2])
    xw, mu, se, nsw = stack_switch_windows(d["spread"], d["ep_ids"], d["switch_mask"], pre=pre, post=post)
    ax.plot(xw, mu, color="#00897B", lw=2)
    ax.fill_between(xw, mu - se, mu + se, color="#80CBC4", alpha=0.35)
    ax.axvline(0, color="#D32F2F", ls="--", lw=1.2)
    ax.set_title(f"Switch event study: market spread (n={nsw})", fontweight="bold")
    ax.set_xlabel("Steps from switch")
    ax.set_ylabel("Spread")

    # ------------------------------------------------------------------
    # Row 3
    # ------------------------------------------------------------------
    inv = d["inventory"]
    k_bid = actions[:, 0]
    k_ask = actions[:, 1]
    size_skew = np.log((actions[:, 2] + 1e-8) / (actions[:, 3] + 1e-8))
    bins = np.array([-15, -8, -4, -2, 0, 2, 4, 8, 15], dtype=float)

    ax = fig.add_subplot(gs[2, 0])
    centers, means = binned_mean(inv, k_bid, bins)
    m = np.isfinite(centers) & np.isfinite(means)
    ax.plot(centers[m], means[m], "o-", color="#1565C0", lw=2)
    ax.axvline(0, color="black", ls="--", alpha=0.5, lw=1)
    ax.set_title(r"Inventory -> $E[k_{bid}\mid inv]$", fontweight="bold")
    ax.set_xlabel("Inventory bin center")
    ax.set_ylabel("Mean bid offset")

    ax = fig.add_subplot(gs[2, 1])
    centers, means = binned_mean(inv, k_ask, bins)
    m = np.isfinite(centers) & np.isfinite(means)
    ax.plot(centers[m], means[m], "s--", color="#EF6C00", lw=2)
    ax.axvline(0, color="black", ls="--", alpha=0.5, lw=1)
    ax.set_title(r"Inventory -> $E[k_{ask}\mid inv]$", fontweight="bold")
    ax.set_xlabel("Inventory bin center")
    ax.set_ylabel("Mean ask offset")

    ax = fig.add_subplot(gs[2, 2])
    centers, means = binned_mean(inv, size_skew, bins)
    m = np.isfinite(centers) & np.isfinite(means)
    ax.plot(centers[m], means[m], "o-", color="#AD1457", lw=2)
    ax.axhline(0, color="black", ls="--", alpha=0.5)
    ax.axvline(0, color="black", ls=":", alpha=0.4)
    ax.set_title(r"Inventory -> size skew $\log(q_{bid}/q_{ask})$", fontweight="bold")
    ax.set_xlabel("Inventory bin center")
    ax.set_ylabel("Mean size skew")

    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Complete validation for the stylised LOB simulator")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="validation/simulator")
    p.add_argument("--episodes", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pre", type=int, default=25)
    p.add_argument("--post", type=int, default=60)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset and os.path.exists(args.dataset):
        raw = np.load(args.dataset)
        data = {k: raw[k] for k in raw.files}
    else:
        cfg = EnvConfig()
        cfg.N_episodes = args.episodes
        cfg.mixed_regime_frac = 0.30
        data = generate_dataset(cfg, seed=args.seed, shuffle=False)
        save_dataset(data, os.path.join(args.out_dir, "dataset_unshuffled.npz"))

    data = sort_dataset_temporally(data)
    d = extract(data)
    rets = episode_returns(d)
    rows = compute_score_rows(d, rets)

    overview_path = os.path.join(args.out_dir, "simulator_validation_overview.png")
    mechanism_path = os.path.join(args.out_dir, "simulator_validation_mechanisms.png")
    csv_path = os.path.join(args.out_dir, "simulator_scorecard.csv")

    make_overview_figure(d, overview_path)
    make_mechanism_figure(d, mechanism_path, pre=args.pre, post=args.post)
    save_scorecard_csv(rows, csv_path)

    print(f"Saved: {overview_path}")
    print(f"Saved: {mechanism_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
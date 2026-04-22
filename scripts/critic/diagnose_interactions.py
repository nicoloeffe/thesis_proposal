"""
diagnose_interactions.py — Test diagnostico sulle interazioni inv × tl × regime.

Domanda centrale: il return MC G_t varia genuinamente con il regime a parità
di (inv, tl), oppure è risk-neutral? E se sì, il critico lo cattura?

Tre domande empiriche:
  Q1: G_mean(inv, tl, regime) ha dipendenza significativa dal regime?
      → i dati hanno l'interazione (intuizione di Nicolò)
  Q2: G_std(inv, tl, regime) varia con il regime?
      → conferma rumore regime-dipendente (narrativa DRO)
  Q3: V_mean(inv, tl, regime) ≈ G_mean(inv, tl, regime)?
      → il critico sta imparando ciò che è presente nei dati?

Output:
  - Tabella testuale (60 bucket) con G_mean/G_sem/G_std/V_mean/V_err
  - Kruskal-Wallis test per bucket (inv, tl) → #bucket con p<0.05
  - Figura 3×2: heatmap G_mean (Row1) e V_mean (Row2), una per regime
  - Heatmap addizionale di |V_err| per vedere dove il critico sbaglia di più

Uso:
  python scripts/diagnose_interactions.py
  python scripts/diagnose_interactions.py --ckpt checkpoints/critic_best.pt \
                                           --dataset data/wm_dataset.npz \
                                           --val_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from scipy.stats import kruskal

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.critic import ValueNetwork
except ImportError:
    ValueNetwork = None


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

INV_EDGES = np.array([-1.00, -0.50, -0.15, 0.15, 0.50, 1.00])
INV_NAMES = ["very_short", "short", "flat", "long", "very_long"]

TL_EDGES = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
TL_NAMES = ["end", "mid_late", "mid_early", "start"]

REGIME_NAMES = ["low_vol", "mid_vol", "high_vol"]


def bucket_inv(inv_norm: np.ndarray) -> np.ndarray:
    """Assegna bucket inv_id ∈ [0, 4] per ogni sample."""
    # digitize returns 1..5 for edges, want 0..4
    b = np.digitize(inv_norm, INV_EDGES[1:-1])  # boundaries interni
    return np.clip(b, 0, len(INV_NAMES) - 1)


def bucket_tl(tl: np.ndarray) -> np.ndarray:
    """Assegna bucket tl_id ∈ [0, 3]. tl=1 (start) → bucket 3."""
    b = np.digitize(tl, TL_EDGES[1:-1])
    return np.clip(b, 0, len(TL_NAMES) - 1)


# ---------------------------------------------------------------------------
# Data loading & critic inference
# ---------------------------------------------------------------------------

def load_critic(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    version = cfg.get("version", "v1")

    kwargs = dict(
        d_state=cfg["d_state"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
    )
    if version in ("v2", "v3"):
        kwargs["d_z"]          = cfg.get("d_z", cfg["d_state"] - 2)
        kwargs["use_spectral"] = cfg.get("use_spectral", True)
    if version == "v3":
        kwargs["d_action"] = cfg.get("d_action", 4)

    critic = ValueNetwork(**kwargs).to(device)
    critic.load_state_dict(ckpt["model"])
    critic.eval()
    print(f"Critic: {ckpt_path}  ({version}, epoch={ckpt['epoch']}, "
          f"val={ckpt['val_loss']:.4f})")
    return critic, ckpt


def build_dataset(
    dataset_path: str,
    reward_stats: dict,
    g_stats:      dict | None,
    gamma:        float,
    inv_max:      float,
    max_t_for_mc: int,
    val_only:     bool,
    seed:         int = 42,
    action_stats: dict | None = None,
    use_actions:  bool = False,
    z_stats:      dict | None = None,
) -> dict:
    """Costruisce (s, G, reg, inv_norm, tl) per l'analisi. v3.1: opzionalmente z_norm."""
    data = np.load(dataset_path)
    sequences   = data["sequences"].astype(np.float32)
    inventories = data["inventories"].astype(np.float32)
    time_left   = data["time_left"].astype(np.float32)
    rewards     = data["rewards"].astype(np.float32)
    regimes     = data["regimes"].astype(np.int64)
    episode_ids = data["episode_ids"]

    M, Np1, d_z = sequences.shape
    N = Np1 - 1

    # v3.1: z-score per-dim
    if z_stats is not None:
        sequences = (sequences - z_stats["mean"]) / z_stats["std"]
        print(f"  v3.1: z normalized")

    # Stato aumentato
    inv_norm_full = np.clip(inventories / inv_max, -1.0, 1.0)
    tl_full       = time_left
    parts = [sequences, inv_norm_full[..., None], tl_full[..., None]]

    # v3: actions
    if use_actions:
        if "actions" not in data.files:
            raise ValueError("Dataset missing 'actions' but use_actions=True")
        actions_raw = data["actions"].astype(np.float32)   # (M, N, 4)
        L_levels = (action_stats.get("L_levels", 10) if action_stats else 10)
        q_max    = (action_stats.get("q_max", 1.0)  if action_stats else 1.0)

        k = actions_raw[..., :2]
        q = actions_raw[..., 2:]
        k_norm = np.clip((k - 1.0) / max(1e-8, L_levels - 1), 0.0, 1.0)
        q_norm = np.clip(q / q_max, 0.0, 1.0)
        a_norm = np.concatenate([k_norm, q_norm], axis=-1)
        a_full = np.concatenate([a_norm, a_norm[:, -1:, :]], axis=1)
        parts.append(a_full)
        print(f"  v3 actions: q_max={q_max:.3f}, L={L_levels}")

    aug = np.concatenate(parts, axis=-1)

    # MC returns: v2.2 respects winsor_active flag from g_stats
    if g_stats is not None:
        G = np.zeros_like(rewards)
        G[:, -1] = rewards[:, -1]
        for t in range(N - 2, -1, -1):
            G[:, t] = rewards[:, t] + gamma * G[:, t + 1]

        # Check winsor_active flag (v2.2); retrocompat v2.1 controlla winsor_low/high
        winsor_on = g_stats.get("winsor_active", False)
        if not winsor_on:
            winsor_on = (g_stats.get("winsor_low", 0.0) > 0.0 or
                         g_stats.get("winsor_high", 100.0) < 100.0)

        if winsor_on and "p_low" in g_stats and "p_high" in g_stats:
            G = np.clip(G, g_stats["p_low"], g_stats["p_high"])
            print(f"Target: v2 G = clip(MC_raw, p_low, p_high) / g_std  (winsorized + z-scored)")
        else:
            print(f"Target: v2 G = MC_raw / g_std  (NO winsor, z-scored only)")

        G = (G - g_stats["mean"]) / (g_stats["std"] + 1e-8)
    else:
        r_norm = (rewards - reward_stats["mean"]) / (reward_stats["std"] + 1e-8)
        G = np.zeros_like(r_norm)
        G[:, -1] = r_norm[:, -1]
        for t in range(N - 2, -1, -1):
            G[:, t] = r_norm[:, t] + gamma * G[:, t + 1]
        print(f"Target: v1 G = reward-normalized MC")

    # Early steps only (MC unbiased)
    max_t_for_mc = min(max_t_for_mc, N)
    s        = aug[:, :max_t_for_mc].reshape(-1, aug.shape[-1])
    g_flat   = G[:, :max_t_for_mc].reshape(-1)
    inv_flat = inv_norm_full[:, :max_t_for_mc].reshape(-1)
    tl_flat  = tl_full[:, :max_t_for_mc].reshape(-1)

    if regimes.ndim == 2:
        reg_flat = regimes[:, :max_t_for_mc].reshape(-1)
    else:
        reg_flat = np.repeat(regimes, max_t_for_mc)

    ep_flat = np.repeat(episode_ids, max_t_for_mc)

    # Val only filter
    if val_only:
        rng = np.random.default_rng(seed)
        unique_eps = np.unique(episode_ids)
        rng.shuffle(unique_eps)
        n_val = max(1, int(len(unique_eps) * 0.1))
        val_eps = set(unique_eps[:n_val])
        mask = np.array([ep in val_eps for ep in ep_flat])
        s        = s[mask]
        g_flat   = g_flat[mask]
        reg_flat = reg_flat[mask]
        inv_flat = inv_flat[mask]
        tl_flat  = tl_flat[mask]
        print(f"val-only: {len(s):,} samples ({n_val} episodes)")
    else:
        print(f"full dataset: {len(s):,} samples")

    return {
        "s":         torch.from_numpy(s).float(),
        "g":         g_flat.astype(np.float32),
        "reg":       reg_flat.astype(np.int64),
        "inv_norm":  inv_flat.astype(np.float32),
        "tl":        tl_flat.astype(np.float32),
    }


@torch.no_grad()
def predict_v_all(critic, s: torch.Tensor, device: torch.device,
                  batch_size: int = 8192) -> np.ndarray:
    critic.eval()
    preds = []
    for i in range(0, len(s), batch_size):
        v = critic(s[i:i+batch_size].to(device))
        preds.append(v.cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Bucket statistics
# ---------------------------------------------------------------------------

def compute_bucket_stats(data: dict, V: np.ndarray) -> dict:
    """
    Per ogni combinazione (inv_bucket, tl_bucket, regime) calcola:
      n, G_mean, G_std, G_sem, V_mean, V_std
    """
    G   = data["g"]
    reg = data["reg"]
    inv = data["inv_norm"]
    tl  = data["tl"]

    inv_b = bucket_inv(inv)
    tl_b  = bucket_tl(tl)

    # Allocate (n_inv, n_tl, n_reg)
    n_inv, n_tl, n_reg = len(INV_NAMES), len(TL_NAMES), len(REGIME_NAMES)
    result = {
        "n":      np.zeros((n_inv, n_tl, n_reg), dtype=np.int64),
        "G_mean": np.full((n_inv, n_tl, n_reg), np.nan),
        "G_std":  np.full((n_inv, n_tl, n_reg), np.nan),
        "G_sem":  np.full((n_inv, n_tl, n_reg), np.nan),
        "V_mean": np.full((n_inv, n_tl, n_reg), np.nan),
        "V_std":  np.full((n_inv, n_tl, n_reg), np.nan),
        "raw_G":  {},    # per Kruskal-Wallis test
    }

    for i in range(n_inv):
        for t in range(n_tl):
            for r in range(n_reg):
                mask = (inv_b == i) & (tl_b == t) & (reg == r)
                n = int(mask.sum())
                result["n"][i, t, r] = n
                if n < 20:   # skip under-sampled
                    continue
                Gb = G[mask]
                Vb = V[mask]
                result["G_mean"][i, t, r] = float(Gb.mean())
                result["G_std"][i, t, r]  = float(Gb.std())
                result["G_sem"][i, t, r]  = float(Gb.std() / np.sqrt(n))
                result["V_mean"][i, t, r] = float(Vb.mean())
                result["V_std"][i, t, r]  = float(Vb.std())
                result["raw_G"][(i, t, r)] = Gb

    return result


def kruskal_test_per_bucket(stats: dict) -> dict:
    """Per ogni (inv, tl), testa se G differisce significativamente tra regimi."""
    n_inv, n_tl = len(INV_NAMES), len(TL_NAMES)
    p_vals = np.full((n_inv, n_tl), np.nan)
    for i in range(n_inv):
        for t in range(n_tl):
            groups = []
            for r in range(len(REGIME_NAMES)):
                if (i, t, r) in stats["raw_G"]:
                    groups.append(stats["raw_G"][(i, t, r)])
            if len(groups) >= 2 and all(len(g) >= 20 for g in groups):
                try:
                    _, p = kruskal(*groups)
                    p_vals[i, t] = p
                except Exception:
                    pass
    n_sig = int((p_vals < 0.05).sum())
    n_total = int(np.isfinite(p_vals).sum())
    return {"p_vals": p_vals, "n_sig": n_sig, "n_total": n_total}


def compute_cross_bucket_ranking(data: dict, V: np.ndarray,
                                  n_pairs: int = 100_000,
                                  seed: int = 42) -> dict:
    """
    Pairwise accuracy cross-bucket: il critico ordina correttamente
    stati che appartengono a bucket (inv, tl, regime) diversi?

    Questo è il test chiave per ranking critic: anche se V_assoluto è
    indeterminato, il SIGN di V_a - V_b deve tracciare SIGN di G_a - G_b.

    Returns:
      global: pairwise accuracy globale
      cross_regime: pairwise accuracy solo su coppie cross-regime
      within_regime: pairwise accuracy su coppie stesso regime
    """
    rng = np.random.default_rng(seed)
    G   = data["g"]
    reg = data["reg"]
    N   = len(G)

    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N, size=n_pairs)
    mask_diff = idx_a != idx_b
    idx_a, idx_b = idx_a[mask_diff], idx_b[mask_diff]

    diff_v = V[idx_a] - V[idx_b]
    diff_g = G[idx_a] - G[idx_b]

    # Tie filter (exact)
    valid = diff_g != 0
    diff_v = diff_v[valid]
    diff_g = diff_g[valid]
    reg_a  = reg[idx_a][valid]
    reg_b  = reg[idx_b][valid]

    acc_global = float((np.sign(diff_v) == np.sign(diff_g)).mean())

    mask_cross = reg_a != reg_b
    mask_within = reg_a == reg_b
    result = {
        "n_pairs":       len(diff_v),
        "global":        acc_global,
    }
    if mask_cross.sum() > 0:
        result["cross_regime"] = float(
            (np.sign(diff_v[mask_cross]) == np.sign(diff_g[mask_cross])).mean()
        )
    if mask_within.sum() > 0:
        result["within_regime"] = float(
            (np.sign(diff_v[mask_within]) == np.sign(diff_g[mask_within])).mean()
        )
    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_full_table(stats: dict, kw: dict) -> None:
    print("\n" + "=" * 92)
    print("BUCKET STATISTICS — G (MC return observed) vs V (critic prediction)")
    print("=" * 92)
    print("Legend:")
    print("  n      : samples in bucket")
    print("  G      : MC return mean ± SEM (std/√n)   [z-scored units]")
    print("  V      : critic prediction mean")
    print("  err    : V - G   (positive = critic overestimates)")
    print("  p      : Kruskal-Wallis p-value across regimes (this inv,tl)")

    for i, inv_name in enumerate(INV_NAMES):
        for t, tl_name in enumerate(TL_NAMES):
            p_val = kw["p_vals"][i, t]
            p_str = f"p_KW={p_val:.3g}" if np.isfinite(p_val) else "p_KW=n/a"
            sig   = "  [*]" if np.isfinite(p_val) and p_val < 0.05 else ""
            print(f"\n  inv={inv_name:>10s}  tl={tl_name:>10s}  {p_str}{sig}")
            for r, reg_name in enumerate(REGIME_NAMES):
                n = stats["n"][i, t, r]
                if n < 20:
                    print(f"    {reg_name:>8s}: n={n:>6d}  [under-sampled]")
                    continue
                Gm  = stats["G_mean"][i, t, r]
                Gse = stats["G_sem"][i, t, r]
                Gsd = stats["G_std"][i, t, r]
                Vm  = stats["V_mean"][i, t, r]
                err = Vm - Gm
                print(f"    {reg_name:>8s}: n={n:>6d}  "
                      f"G={Gm:+6.3f} ± {Gse:.3f}  σ_G={Gsd:.3f}  "
                      f"V={Vm:+6.3f}  err={err:+6.3f}")


def print_summary(stats: dict, kw: dict) -> None:
    print("\n" + "=" * 92)
    print("SUMMARY — le 3 domande diagnostiche")
    print("=" * 92)

    # Q1 — G_mean varia col regime?
    n_populated_inv_tl = int(np.isfinite(kw["p_vals"]).sum())
    n_total_inv_tl = kw["p_vals"].size
    n_undersampled = n_total_inv_tl - n_populated_inv_tl

    print("\n  Q1: G_mean varia significativamente col regime (a parità di inv, tl)?")
    print(f"      Kruskal-Wallis: {kw['n_sig']}/{kw['n_total']} "
          f"(inv,tl) buckets popolati con p<0.05  "
          f"({100*kw['n_sig']/max(1,kw['n_total']):.0f}%)")
    if n_undersampled > 0:
        print(f"      ({n_undersampled} bucket under-sampled, esclusi dal test)")

    # G_mean range across regimes per bucket — soppreso warning All-NaN
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            g_range = np.nanmax(stats["G_mean"], axis=2) - np.nanmin(stats["G_mean"], axis=2)

    valid = np.isfinite(g_range)
    if valid.any():
        print(f"      G_mean range tra regimi (median over populated buckets): "
              f"{np.nanmedian(g_range[valid]):.3f}")
        print(f"      G_mean range tra regimi (max over populated buckets):    "
              f"{np.nanmax(g_range[valid]):.3f}  "
              f"(in bucket: {_argmax_bucket_name(g_range)})")

    # Q2 — σ(G) varia col regime?
    print("\n  Q2: G_std varia col regime? (atteso sì per narrativa DRO)")
    for r, name in enumerate(REGIME_NAMES):
        vals = stats["G_std"][..., r]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            print(f"      {name:>8s}: median σ(G) = {np.median(vals):.3f}   "
                  f"max = {vals.max():.3f}")

    # Q3 — V tracks G?
    print("\n  Q3: Il critico V tracks G per bucket?")
    err = stats["V_mean"] - stats["G_mean"]
    err_valid = err[np.isfinite(err)]
    if len(err_valid) > 0:
        print(f"      median |V−G| = {np.median(np.abs(err_valid)):.3f}")
        print(f"      max    |V−G| = {np.abs(err_valid).max():.3f}")
        print(f"      V bias (mean V−G) = {err_valid.mean():+.3f}")

        # Breakdown per regime: where is the critic worst?
        for r, name in enumerate(REGIME_NAMES):
            e_r = err[..., r]
            e_r = e_r[np.isfinite(e_r)]
            if len(e_r) > 0:
                print(f"      {name:>8s}: median |err| = {np.median(np.abs(e_r)):.3f}   "
                      f"worst bucket err = {e_r[np.argmax(np.abs(e_r))]:+.3f}")


def _argmax_bucket_name(matrix_inv_tl: np.ndarray) -> str:
    """Nome del bucket (inv, tl) con valore massimo."""
    if not np.isfinite(matrix_inv_tl).any():
        return "n/a (all NaN)"
    flat_idx = np.nanargmax(matrix_inv_tl)
    i, t = np.unravel_index(flat_idx, matrix_inv_tl.shape)
    return f"inv={INV_NAMES[i]}, tl={TL_NAMES[t]}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(stats: dict, kw: dict, out_path: str) -> None:
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Compute shared color limits for G and V heatmaps (easier visual comparison)
    all_G = stats["G_mean"][np.isfinite(stats["G_mean"])]
    all_V = stats["V_mean"][np.isfinite(stats["V_mean"])]
    if len(all_G) > 0 and len(all_V) > 0:
        vmin = min(all_G.min(), all_V.min())
        vmax = max(all_G.max(), all_V.max())
    else:
        vmin, vmax = -1, 1

    # -----------------------------------------------------------------------
    # Row 1: G_mean heatmap per regime
    # -----------------------------------------------------------------------
    for r, name in enumerate(REGIME_NAMES):
        ax = fig.add_subplot(gs[0, r])
        mat = stats["G_mean"][..., r]   # (n_inv, n_tl)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(TL_NAMES)))
        ax.set_xticklabels(TL_NAMES, fontsize=8, rotation=30)
        ax.set_yticks(range(len(INV_NAMES)))
        ax.set_yticklabels(INV_NAMES, fontsize=8)
        ax.set_xlabel("time_left bucket", fontsize=9)
        ax.set_ylabel("inv bucket", fontsize=9)
        ax.set_title(f"$\\bar{{G}}$ observed — {name}", fontsize=11)
        # annotate
        for i in range(mat.shape[0]):
            for t in range(mat.shape[1]):
                if np.isfinite(mat[i, t]):
                    ax.text(t, i, f"{mat[i,t]:+.2f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, shrink=0.7)

    # -----------------------------------------------------------------------
    # Row 2: V_mean heatmap per regime (same scale)
    # -----------------------------------------------------------------------
    for r, name in enumerate(REGIME_NAMES):
        ax = fig.add_subplot(gs[1, r])
        mat = stats["V_mean"][..., r]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(TL_NAMES)))
        ax.set_xticklabels(TL_NAMES, fontsize=8, rotation=30)
        ax.set_yticks(range(len(INV_NAMES)))
        ax.set_yticklabels(INV_NAMES, fontsize=8)
        ax.set_xlabel("time_left bucket", fontsize=9)
        ax.set_ylabel("inv bucket", fontsize=9)
        ax.set_title(f"$\\bar{{V}}$ critic — {name}", fontsize=11)
        for i in range(mat.shape[0]):
            for t in range(mat.shape[1]):
                if np.isfinite(mat[i, t]):
                    ax.text(t, i, f"{mat[i,t]:+.2f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, shrink=0.7)

    # -----------------------------------------------------------------------
    # Row 3: Diagnostic panels
    # -----------------------------------------------------------------------

    import warnings

    # (2,0): V_err heatmap (max abs across regimes) — where is critic worst?
    ax = fig.add_subplot(gs[2, 0])
    err = stats["V_mean"] - stats["G_mean"]   # (inv, tl, reg)
    # Max abs err over regimes per (inv, tl) — soppresso All-NaN warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        err_max_abs = np.nanmax(np.abs(err), axis=2)
        em = np.nanmax(np.abs(err_max_abs)) if np.isfinite(err_max_abs).any() else 1
    im = ax.imshow(err_max_abs, aspect="auto", cmap="Reds",
                   vmin=0, vmax=em, origin="lower")
    ax.set_xticks(range(len(TL_NAMES)))
    ax.set_xticklabels(TL_NAMES, fontsize=8, rotation=30)
    ax.set_yticks(range(len(INV_NAMES)))
    ax.set_yticklabels(INV_NAMES, fontsize=8)
    ax.set_xlabel("time_left bucket", fontsize=9)
    ax.set_ylabel("inv bucket", fontsize=9)
    ax.set_title("max |V − G| across regimes\n(hotspot = dove il critico sbaglia)",
                 fontsize=10)
    for i in range(err_max_abs.shape[0]):
        for t in range(err_max_abs.shape[1]):
            if np.isfinite(err_max_abs[i, t]):
                ax.text(t, i, f"{err_max_abs[i,t]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.7)

    # (2,1): G_range across regimes heatmap
    ax = fig.add_subplot(gs[2, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        g_range = np.nanmax(stats["G_mean"], axis=2) - np.nanmin(stats["G_mean"], axis=2)
        gm = np.nanmax(g_range) if np.isfinite(g_range).any() else 1
    im = ax.imshow(g_range, aspect="auto", cmap="Blues",
                   vmin=0, vmax=gm, origin="lower")
    ax.set_xticks(range(len(TL_NAMES)))
    ax.set_xticklabels(TL_NAMES, fontsize=8, rotation=30)
    ax.set_yticks(range(len(INV_NAMES)))
    ax.set_yticklabels(INV_NAMES, fontsize=8)
    ax.set_xlabel("time_left bucket", fontsize=9)
    ax.set_ylabel("inv bucket", fontsize=9)
    ax.set_title("max $\\bar{G}$ − min $\\bar{G}$ across regimes\n"
                 "(hotspot = dove i dati hanno interazione)", fontsize=10)
    for i in range(g_range.shape[0]):
        for t in range(g_range.shape[1]):
            if np.isfinite(g_range[i, t]):
                ax.text(t, i, f"{g_range[i,t]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.7)

    # (2,2): Kruskal-Wallis p-values heatmap
    ax = fig.add_subplot(gs[2, 2])
    pvals = kw["p_vals"]
    log_p = -np.log10(np.clip(pvals, 1e-30, 1))
    lm = np.nanmax(log_p) if np.isfinite(log_p).any() else 3
    im = ax.imshow(log_p, aspect="auto", cmap="Purples",
                   vmin=0, vmax=lm, origin="lower")
    ax.set_xticks(range(len(TL_NAMES)))
    ax.set_xticklabels(TL_NAMES, fontsize=8, rotation=30)
    ax.set_yticks(range(len(INV_NAMES)))
    ax.set_yticklabels(INV_NAMES, fontsize=8)
    ax.set_xlabel("time_left bucket", fontsize=9)
    ax.set_ylabel("inv bucket", fontsize=9)
    ax.set_title("−log10(p) Kruskal-Wallis\n(alto = regime differisce)",
                 fontsize=10)
    for i in range(log_p.shape[0]):
        for t in range(log_p.shape[1]):
            if np.isfinite(log_p[i, t]):
                sig = "*" if pvals[i, t] < 0.05 else " "
                ax.text(t, i, f"{log_p[i,t]:.1f}{sig}",
                        ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle("Diagnostica interazioni inv × tl × regime",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load critic
    critic, ckpt = load_critic(args.ckpt, device)
    cfg          = ckpt["cfg"]
    reward_stats = ckpt["reward_stats"]
    g_stats      = ckpt.get("g_stats")
    action_stats = ckpt.get("action_stats")
    z_stats      = ckpt.get("z_stats")
    gamma        = cfg["gamma"]
    inv_max      = cfg["inv_max"]
    use_actions  = cfg.get("use_actions", False)
    z_norm_flag  = cfg.get("z_norm", False)

    print(f"gamma={gamma}  inv_max={inv_max}  use_actions={use_actions}  "
          f"z_norm={z_norm_flag}")
    if g_stats is not None:
        print(f"g_stats: μ={g_stats['mean']:+.4f}  σ={g_stats['std']:.4f}")
    if use_actions and action_stats is not None:
        print(f"action_stats: q_max={action_stats['q_max']:.3f}  "
              f"L_levels={action_stats.get('L_levels', 10)}")
    if z_norm_flag and z_stats is not None:
        print(f"z_stats: per-dim std mean={z_stats['std'].mean():.4f}")

    # Build dataset
    data = build_dataset(
        args.dataset, reward_stats=reward_stats, g_stats=g_stats,
        gamma=gamma, inv_max=inv_max,
        max_t_for_mc=args.max_t_for_mc,
        val_only=args.val_only,
        seed=args.seed,
        action_stats=action_stats,
        use_actions=use_actions,
        z_stats=z_stats if z_norm_flag else None,
    )

    # Predict V on all samples
    print("\nPredicting V(s) on all samples...")
    V = predict_v_all(critic, data["s"], device, batch_size=args.batch_size)
    print(f"  V range: [{V.min():+.3f}, {V.max():+.3f}]  mean={V.mean():+.3f}")

    # Bucket statistics
    print("\nComputing bucket statistics (5×4×3 = 60 buckets)...")
    stats = compute_bucket_stats(data, V)
    n_populated = (stats["n"] >= 20).sum()
    print(f"  {n_populated}/60 bucket popolati (>=20 samples)")

    # Kruskal-Wallis test
    print("\nRunning Kruskal-Wallis tests...")
    kw = kruskal_test_per_bucket(stats)

    # Cross-bucket pairwise ranking accuracy (rank-critic friendly)
    print("\nComputing cross-bucket pairwise ranking accuracy...")
    pw = compute_cross_bucket_ranking(data, V, n_pairs=args.n_pairwise,
                                      seed=args.seed)

    # Print outputs
    print_full_table(stats, kw)
    print_summary(stats, kw)

    # Pairwise accuracy section
    mode = cfg.get("mode", "mc")
    print("\n" + "=" * 92)
    print(f"PAIRWISE RANKING ACCURACY  (mode={mode}; scale-invariant metric)")
    print("=" * 92)
    print(f"  n pairs              : {pw['n_pairs']:,}")
    print(f"  Global accuracy      : {pw['global']:.4f}")
    if "within_regime" in pw:
        print(f"  Within-regime pairs  : {pw['within_regime']:.4f}")
    if "cross_regime" in pw:
        print(f"  Cross-regime pairs   : {pw['cross_regime']:.4f}")
    print(f"\n  Random baseline      : 0.5000")
    print(f"  Interpretation:")
    print(f"    Cross-regime high  → il critico distingue stati cross-regime correttamente")
    print(f"    Within-regime high → il critico ordina correttamente anche a regime fisso")
    if mode == "rank":
        print(f"\n  Note: questo è LA metrica principale per ranking critic.")
        print(f"        R²/bias sono scale-dipendenti e non significativi qui.")

    # Plot
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_all(stats, kw, out_path=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose inv × tl × regime interactions")
    parser.add_argument("--ckpt",     type=str, default="checkpoints/critic_best.pt")
    parser.add_argument("--dataset",  type=str, default="data/wm_dataset.npz")
    parser.add_argument("--val_only", action="store_true",
                        help="Use only val split (10% episodes)")
    parser.add_argument("--max_t_for_mc", type=int, default=15,
                        help="MC returns only for t ∈ [0, max_t_for_mc]")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--n_pairwise", type=int, default=100_000,
                        help="N random pairs for pairwise accuracy")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out",        type=str, default="diagnose_interactions.png")
    args = parser.parse_args()
    main(args)
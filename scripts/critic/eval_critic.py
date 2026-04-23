"""
eval_critic.py — Valutazione completa del critico V_θ(s) (Modulo C, part 1).

Il critico è allenato per predire il valore atteso E[G_t | s_t] dove
G_t = Σ_{k≥0} γ^k r_{t+k} (MC return normalizzato).

Lo stato è aumentato: s_t = [z_t, inv_norm_t, tl_t] ∈ R^{d_z + 2}.
z_t è il latente LOB (16-dim), inv_norm_t ∈ [-1,1], tl_t ∈ [0,1].

Tre blocchi di validazione (3x3 panels + scorecard):

  Row 1 — PREDICTION QUALITY ("il critico predice bene i return?")
    (0,0): V_pred vs G_true scatter + Spearman ρ globale + R²
    (0,1): Per-regime metrics (bar MSE + Spearman)
    (0,2): Distribuzione errori V - G (detecta bias e modi)

  Row 2 — FEATURE IMPORTANCE ("il critico usa tutte le feature?")
    (1,0): Ablation R² — drop di R² quando si shuffla z / inv / tl
           Scale-free: misura l'impatto reale di ogni feature
    (1,1): Partial dependence su inv (per regime)
    (1,2): Partial dependence su tl (per regime)

  Row 3 — SMOOTHNESS & DRO COMPATIBILITY
    (2,0): Lipschitz empirica globale + per regime
    (2,1): PGD adversarial Lipschitz (worst-case)
    (2,2): Calibration (reliability diagram)

Scorecard con verdetto A/B/C/D:
  A: tutto ok, critico pronto per DRO
  B: prediction ok ma z contribuisce poco (ablation) → DRO con leverage limitato
  C: prediction debole → ri-allenare
  D: entrambi problematici → ripensare

Uso:
  python scripts/eval_critic.py
  python scripts/eval_critic.py --ckpt checkpoints/critic_best.pt \
                                 --dataset data/wm_dataset.npz \
                                 --n_samples 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from models.critic import ValueNetwork
except ImportError:
    # Fallback for standalone/testing
    ValueNetwork = None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_critic(ckpt_path: str, device: torch.device) -> tuple[ValueNetwork, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    version = cfg.get("version", "v1")

    # Build model with appropriate args for v1/v2/v3
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
    print(f"Critic loaded: {ckpt_path}  (version={version})")
    print(f"  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}  "
          f"mode={cfg.get('mode', '?')}")
    extra = ""
    if version == "v3":
        extra = f"  d_action={cfg.get('d_action')}"
    print(f"  d_state={cfg['d_state']}  hidden={cfg['hidden']}  "
          f"n_layers={cfg['n_layers']}  inv_max={cfg.get('inv_max', '?')}{extra}")
    return critic, ckpt


def load_data(
    dataset_path:    str,
    reward_stats:    dict,
    gamma:           float,
    inv_max:         float,
    n_samples:       int | None,
    val_only:        bool,
    max_t_for_mc:    int = 15,
    seed:            int = 42,
    g_stats:         dict | None = None,
    action_stats:    dict | None = None,
    use_actions:     bool = False,
    z_stats:         dict | None = None,        # v3.1
) -> dict:
    """
    Carica dataset, costruisce stati aumentati e MC returns.

    v3.1: se z_stats fornite, applica z-score per-dim al latente.
    v3: se use_actions, concatena le quote A-S normalizzate allo stato.
    """
    data = np.load(dataset_path)
    sequences   = data["sequences"].astype(np.float32)
    inventories = data["inventories"].astype(np.float32)
    time_left   = data["time_left"].astype(np.float32)
    rewards     = data["rewards"].astype(np.float32)
    regimes     = data["regimes"].astype(np.int64)
    episode_ids = data["episode_ids"]

    M, Np1, d_z = sequences.shape
    N = Np1 - 1

    # v3.1: z-score per-dim (if z_stats provided)
    if z_stats is not None:
        sequences = (sequences - z_stats["mean"]) / z_stats["std"]
        print(f"  v3.1: z normalized (per-dim z-score)")

    # Augmented state: [z, inv_norm, tl]
    inv_norm = np.clip(inventories / inv_max, -1.0, 1.0)[..., None]
    tl       = time_left[..., None]
    parts = [sequences, inv_norm, tl]

    # v3: actions
    if use_actions:
        if "actions" not in data.files:
            raise ValueError("Dataset missing 'actions' but use_actions=True")
        actions_raw = data["actions"].astype(np.float32)    # (M, N, 4)

        L_levels = (action_stats.get("L_levels", 10) if action_stats else 10)
        q_max    = (action_stats.get("q_max", 1.0)  if action_stats else 1.0)

        k = actions_raw[..., :2]
        q = actions_raw[..., 2:]
        k_norm = np.clip((k - 1.0) / max(1e-8, L_levels - 1), 0.0, 1.0)
        q_norm = np.clip(q / q_max, 0.0, 1.0)
        a_norm = np.concatenate([k_norm, q_norm], axis=-1)       # (M, N, 4)
        a_full = np.concatenate([a_norm, a_norm[:, -1:, :]], axis=1)  # (M, N+1, 4)
        parts.append(a_full)
        print(f"  v3 state: d_state={sum(p.shape[-1] for p in parts)} "
              f"(including 4 action features)")

    aug = np.concatenate(parts, axis=-1)

    # MC returns: v2 uses raw rewards + z-score; v1 uses reward-normalized
    if g_stats is not None:
        # v2: MC on raw rewards, then optional winsorize + z-score
        G = np.zeros_like(rewards)
        G[:, -1] = rewards[:, -1]
        for t in range(N - 2, -1, -1):
            G[:, t] = rewards[:, t] + gamma * G[:, t + 1]

        # v2.2: winsorize solo se attivata nel ckpt (winsor_active)
        winsor_on = g_stats.get("winsor_active", False)
        # Retrocompat v2.1: aveva sempre p_low/p_high ma no winsor_active flag
        if not winsor_on and "p_low" in g_stats and "p_high" in g_stats:
            p_low, p_high = g_stats["p_low"], g_stats["p_high"]
            # Check if percentili sono diversi da min/max del raw G
            # (se sì = v2.1 con winsor attiva; se no = v2.2 con winsor off)
            winsor_on = (g_stats.get("winsor_low", 0.0) > 0.0 or
                         g_stats.get("winsor_high", 100.0) < 100.0)

        if winsor_on:
            p_low, p_high = g_stats["p_low"], g_stats["p_high"]
            n_clip = ((G < p_low) | (G > p_high)).sum()
            G = np.clip(G, p_low, p_high)
            print(f"  Target: G winsorized [{p_low:+.3f}, {p_high:+.3f}] "
                  f"({n_clip:,} clipped) + z-scored (v2.1)")
        else:
            print(f"  Target: G z-scored, no winsorization (v2.2)")

        G = (G - g_stats["mean"]) / (g_stats["std"] + 1e-8)
        print(f"    g_mean={g_stats['mean']:+.4f}  g_std={g_stats['std']:.4f}")
    else:
        # v1: reward-normalized MC
        r_norm = (rewards - reward_stats["mean"]) / (reward_stats["std"] + 1e-8)
        G = np.zeros_like(r_norm)
        G[:, -1] = r_norm[:, -1]
        for t in range(N - 2, -1, -1):
            G[:, t] = r_norm[:, t] + gamma * G[:, t + 1]
        print(f"  Target: reward-normalized MC (v1)")

    # Use only early steps (MC less biased)
    max_t_for_mc = min(max_t_for_mc, N)
    s_flat = aug[:, :max_t_for_mc, :].reshape(-1, aug.shape[-1])
    g_flat = G[:, :max_t_for_mc].reshape(-1)

    # Regimes flat
    if regimes.ndim == 2:
        reg_flat = regimes[:, :max_t_for_mc].reshape(-1)
    else:
        reg_flat = np.repeat(regimes, max_t_for_mc)

    # Episode ids flat (for val split if needed)
    ep_flat = np.repeat(episode_ids, max_t_for_mc)

    # Val split (episode-based) if requested
    if val_only:
        rng = np.random.default_rng(seed)
        unique_eps = np.unique(episode_ids)
        rng.shuffle(unique_eps)
        n_val = max(1, int(len(unique_eps) * 0.1))
        val_eps = set(unique_eps[:n_val])
        mask = np.array([ep in val_eps for ep in ep_flat])
        s_flat   = s_flat[mask]
        g_flat   = g_flat[mask]
        reg_flat = reg_flat[mask]
        print(f"  val-only: {len(s_flat):,} samples from {n_val} episodes")

    # Subsample
    if n_samples is not None and n_samples < len(s_flat):
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(len(s_flat), size=n_samples, replace=False)
        s_flat   = s_flat[idx]
        g_flat   = g_flat[idx]
        reg_flat = reg_flat[idx]

    print(f"Dataset loaded: {len(s_flat):,} (s, G) pairs  d_state={aug.shape[-1]}")
    print(f"  G: mean={g_flat.mean():+.4f}  std={g_flat.std():.4f}  "
          f"[{g_flat.min():+.2f}, {g_flat.max():+.2f}]")
    print(f"  Regime counts: {np.bincount(reg_flat.astype(int), minlength=3).tolist()}")

    return {
        "s":        torch.from_numpy(s_flat).float(),
        "g":        torch.from_numpy(g_flat).float(),
        "reg":      torch.from_numpy(reg_flat).long(),
        "d_z":      d_z,
        "d_action": 4 if use_actions else 0,
        "d_state":  aug.shape[-1],
    }


# ===========================================================================
# SECTION 1 — Prediction quality
# ===========================================================================

@torch.no_grad()
def predict_all(critic: ValueNetwork, s: torch.Tensor, device: torch.device,
                batch_size: int = 4096) -> np.ndarray:
    """Batched V prediction."""
    critic.eval()
    preds = []
    for i in range(0, len(s), batch_size):
        batch = s[i:i+batch_size].to(device)
        v = critic(batch)
        preds.append(v.cpu().numpy())
    return np.concatenate(preds)


def prediction_quality(
    critic:     ValueNetwork,
    data:       dict,
    device:     torch.device,
) -> dict:
    """
    Misura la qualità di V come estimatore di G.
    Metriche: MSE, MAE, Spearman ρ, Pearson r, R².
    Per-regime e globale.
    """
    s   = data["s"]
    g   = data["g"].numpy()
    reg = data["reg"].numpy()

    V = predict_all(critic, s, device)                    # (N,)

    # Global metrics
    err = V - g
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    rho, _ = spearmanr(V, g)
    r = np.corrcoef(V, g)[0, 1]
    # R² as explained variance
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    # Per-regime
    per_regime = {}
    for r_id in [0, 1, 2]:
        mask = (reg == r_id)
        if mask.sum() == 0:
            continue
        err_r = V[mask] - g[mask]
        rho_r, _ = spearmanr(V[mask], g[mask])
        r_pear = np.corrcoef(V[mask], g[mask])[0, 1]
        ss_res_r = np.sum(err_r ** 2)
        ss_tot_r = np.sum((g[mask] - g[mask].mean()) ** 2)
        r2_r = float(1.0 - ss_res_r / (ss_tot_r + 1e-12))
        per_regime[r_id] = {
            "n":        int(mask.sum()),
            "mse":      float(np.mean(err_r ** 2)),
            "mae":      float(np.mean(np.abs(err_r))),
            "bias":     float(err_r.mean()),
            "spearman": float(rho_r),
            "pearson":  float(r_pear),
            "r2":       r2_r,
        }

    return {
        "V":        V,
        "G":        g,
        "reg":      reg,
        "errors":   err,
        "mse":      mse,
        "mae":      mae,
        "bias":     bias,
        "spearman": float(rho),
        "pearson":  float(r),
        "r2":       r2,
        "per_regime": per_regime,
    }


# ===========================================================================
# SECTION 2 — Sensibility
# ===========================================================================

def gradient_magnitudes(
    critic:     ValueNetwork,
    data:       dict,
    device:     torch.device,
    n_samples:  int = 5000,
    seed:       int = 42,
) -> dict:
    """
    Il test killer: ||∇_z V|| vs |∂V/∂inv| vs |∂V/∂tl|.
    Se il critico è degenere su z, il gradient su z è ~0 e il DRO è inutile.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(data["s"]))
    idx = rng.choice(len(data["s"]), size=n, replace=False)
    s = data["s"][idx].to(device).requires_grad_(True)
    d_z      = data["d_z"]
    d_action = data.get("d_action", 0)

    v = critic(s)
    grads = torch.autograd.grad(v.sum(), s, create_graph=False)[0]  # (n, d_state)

    # Split: first d_z dims are z, then inv, then tl, then optionally actions
    grad_z   = grads[:, :d_z]              # (n, d_z)
    grad_inv = grads[:, d_z]               # (n,)
    grad_tl  = grads[:, d_z + 1]           # (n,)

    # Per-sample norms
    norm_z   = grad_z.norm(dim=-1).cpu().numpy()
    abs_inv  = grad_inv.abs().cpu().numpy()
    abs_tl   = grad_tl.abs().cpu().numpy()

    # Per-dim mean |grad| for z
    per_dim_z = grad_z.abs().mean(dim=0).cpu().numpy()

    # v3: gradient per action dims
    abs_actions = None
    per_unit_actions = 0.0
    norm_actions = None
    if d_action > 0:
        grad_actions = grads[:, d_z + 2:d_z + 2 + d_action]     # (n, d_action)
        norm_actions = grad_actions.norm(dim=-1).cpu().numpy()
        abs_actions  = grad_actions.abs().mean(dim=0).cpu().numpy()  # (d_action,)
        per_unit_actions = float(norm_actions.mean() / np.sqrt(d_action))

    # Per-unit-dim magnitudes
    per_unit_z   = float(norm_z.mean() / np.sqrt(d_z))
    per_unit_inv = float(abs_inv.mean())
    per_unit_tl  = float(abs_tl.mean())

    # Ratio: how much does z contribute vs scalars (v2 metric, still reported)?
    ratio_z_vs_scalars = per_unit_z / ((per_unit_inv + per_unit_tl) / 2 + 1e-8)

    # v3: ratio includes actions
    if d_action > 0:
        denom_v3 = (per_unit_inv + per_unit_tl + per_unit_actions) / 3
        ratio_z_vs_all = per_unit_z / (denom_v3 + 1e-8)
    else:
        ratio_z_vs_all = ratio_z_vs_scalars

    return {
        "norm_z":            norm_z,
        "abs_inv":           abs_inv,
        "abs_tl":            abs_tl,
        "abs_actions":       abs_actions,                # (d_action,) or None
        "per_unit_z":        per_unit_z,
        "per_unit_inv":      per_unit_inv,
        "per_unit_tl":       per_unit_tl,
        "per_unit_actions":  per_unit_actions,
        "per_unit_z_median": float(np.median(norm_z) / np.sqrt(d_z)),
        "per_dim_z":         per_dim_z,
        "ratio_z_vs_scalars": float(ratio_z_vs_scalars),
        "ratio_z_vs_all":     float(ratio_z_vs_all),
        "d_action":           d_action,
    }


def partial_dependence(
    critic:     ValueNetwork,
    data:       dict,
    device:     torch.device,
    variable:   str,            # "inv" or "tl"
    n_grid:     int = 21,
    n_base:     int = 500,
    seed:       int = 42,
) -> dict:
    """
    Partial dependence di V rispetto a inv o tl.
    Per n_base stati di base, fissa z e l'altra variabile scalare,
    varia la variabile target sulla sua range. Media per regime.
    """
    rng = np.random.default_rng(seed)
    d_z = data["d_z"]
    n_base = min(n_base, len(data["s"]))
    idx = rng.choice(len(data["s"]), size=n_base, replace=False)
    s_base = data["s"][idx].clone()                 # (n_base, d_state)
    reg_base = data["reg"][idx].numpy()

    if variable == "inv":
        grid = np.linspace(-1, 1, n_grid)
        var_idx = d_z
    elif variable == "tl":
        grid = np.linspace(0, 1, n_grid)
        var_idx = d_z + 1
    else:
        raise ValueError(f"Unknown variable: {variable}")

    # For each grid value, replace var column and evaluate
    curves_per_regime = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for v_val in grid:
            s_mod = s_base.clone()
            s_mod[:, var_idx] = float(v_val)
            v = critic(s_mod.to(device)).cpu().numpy()

            for r_id in [0, 1, 2]:
                mask = (reg_base == r_id)
                if mask.sum() > 0:
                    curves_per_regime[r_id].append(v[mask].mean())

    # Convert to arrays
    curves = {r: np.array(vals) for r, vals in curves_per_regime.items()
              if len(vals) == n_grid}

    return {
        "grid":     grid,
        "curves":   curves,         # {regime_id: (n_grid,)}
        "variable": variable,
    }


# ===========================================================================
# SECTION 2b — Ablation & Calibration (replacing gradient ratio)
# ===========================================================================

def ablation_r2(
    critic:     ValueNetwork,
    data:       dict,
    device:     torch.device,
    n_samples:  int = 5000,
    seed:       int = 42,
) -> dict:
    """
    Ablation test: misura il calo di R² e Spearman ρ quando si shuffla
    ciascun gruppo di feature (z, inv, tl, actions).

    Questo è scale-free: non confronta gradienti su scale diverse,
    ma l'impatto di ogni feature sulla qualità predittiva del critico.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(data["s"]))
    idx = rng.choice(len(data["s"]), size=n, replace=False)
    s = data["s"][idx]
    g = data["g"][idx].numpy()
    d_z = data["d_z"]
    d_action = data.get("d_action", 0)

    # Baseline prediction
    V_base = predict_all(critic, s, device)
    ss_tot = np.sum((g - g.mean()) ** 2) + 1e-12
    r2_base = float(1.0 - np.sum((V_base - g) ** 2) / ss_tot)
    rho_base = float(spearmanr(V_base, g)[0])

    results = {
        "baseline": {"r2": r2_base, "spearman": rho_base},
    }

    # Ablation groups: shuffle each group independently
    groups = {
        "z":   (0, d_z),
        "inv": (d_z, d_z + 1),
        "tl":  (d_z + 1, d_z + 2),
    }
    if d_action > 0:
        groups["actions"] = (d_z + 2, d_z + 2 + d_action)

    for name, (start, end) in groups.items():
        s_abl = s.clone()
        # Shuffle the feature group across samples (breaks correlation)
        perm = torch.from_numpy(rng.permutation(n))
        s_abl[:, start:end] = s_abl[perm, start:end]

        V_abl = predict_all(critic, s_abl, device)
        r2_abl = float(1.0 - np.sum((V_abl - g) ** 2) / ss_tot)
        rho_abl = float(spearmanr(V_abl, g)[0])

        results[name] = {
            "r2": r2_abl,
            "spearman": rho_abl,
            "r2_drop": r2_base - r2_abl,
            "rho_drop": rho_base - rho_abl,
        }

    return results


def calibration_analysis(
    V:          np.ndarray,
    G:          np.ndarray,
    n_bins:     int = 15,
) -> dict:
    """
    Calibration (reliability diagram): per bin di V predetto,
    calcola la media di G osservato. Un critico calibrato ha
    E[G | V ∈ bin] ≈ center(bin).
    """
    # Bin by predicted V
    v_min, v_max = V.min(), V.max()
    edges = np.linspace(v_min, v_max, n_bins + 1)
    bin_centers = []
    g_means = []
    g_stds = []
    counts = []

    for i in range(n_bins):
        mask = (V >= edges[i]) & (V < edges[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = mask | (V == edges[i + 1])
        n = mask.sum()
        if n < 10:
            continue
        bin_centers.append((edges[i] + edges[i + 1]) / 2)
        g_means.append(float(G[mask].mean()))
        g_stds.append(float(G[mask].std()))
        counts.append(int(n))

    return {
        "bin_centers": np.array(bin_centers),
        "g_means":     np.array(g_means),
        "g_stds":      np.array(g_stds),
        "counts":      np.array(counts),
    }


# ===========================================================================
# SECTION 3 — Smoothness & DRO compatibility
# ===========================================================================

def lipschitz_analysis(
    critic:     ValueNetwork,
    data:       dict,
    device:     torch.device,
    n_samples:  int = 5000,
    seed:       int = 42,
) -> dict:
    """
    Lipschitz empirica globale + per regime, calcolata SOLO rispetto a z.
    Lipschitz = ||∇_z V(z, inv, tl)|| (L2 norm sul vettore z).
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(data["s"]))
    idx = rng.choice(len(data["s"]), size=n, replace=False)
    s = data["s"][idx].to(device).requires_grad_(True)
    reg = data["reg"][idx].numpy()

    v = critic(s)
    grads = torch.autograd.grad(v.sum(), s, create_graph=False)[0]
    
    # Isolate gradients with respect to z only
    grad_z = grads[:, :critic.d_z]
    lip = grad_z.norm(dim=-1).cpu().numpy()               # (n,)

    # Global
    result = {
        "median":   float(np.median(lip)),
        "p95":      float(np.percentile(lip, 95)),
        "max":      float(lip.max()),
    }

    # Per-regime
    per_regime = {}
    for r_id in [0, 1, 2]:
        mask = (reg == r_id)
        if mask.sum() > 0:
            per_regime[r_id] = {
                "median": float(np.median(lip[mask])),
                "p95":    float(np.percentile(lip[mask], 95)),
                "max":    float(lip[mask].max()),
            }

    result["per_regime"] = per_regime
    result["lip_samples"] = lip
    return result


def pgd_adversarial_lipschitz(
    critic:     ValueNetwork,
    data:       dict,
    device:     torch.device,
    eps_rel:    float = 0.01,
    n_steps:    int = 20,
    n_samples:  int = 500,
    seed:       int = 42,
) -> dict:
    """
    PGD per trovare la direzione che massimizza |V(s+δ) - V(s)| / ||δ||.
    Simile all'encoder eval: worst-case Lipschitz.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(data["s"]))
    idx = rng.choice(len(data["s"]), size=n, replace=False)
    s0 = data["s"][idx].to(device)

    # Per-sample eps based on ||s||
    s_norm = s0.norm(dim=-1, keepdim=True)
    eps = eps_rel * s_norm

    # Init δ randomly
    delta = torch.randn_like(s0) * 1e-4
    delta.requires_grad_(True)

    v0 = critic(s0).detach()

    for _ in range(n_steps):
        v_pert = critic(s0 + delta)
        loss = (v_pert - v0).abs().sum()    # maximize |V(s+δ) - V(s)|
        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        # Sign-based step (like L_infinity PGD, but we normalize by ||δ||)
        delta_new = delta + eps * grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
        # Project to ||δ|| <= eps
        dn = delta_new.norm(dim=-1, keepdim=True)
        scale = torch.where(dn > eps, eps / dn, torch.ones_like(dn))
        delta = (delta_new * scale).detach().requires_grad_(True)

    with torch.no_grad():
        v_pert = critic(s0 + delta)
        diff = (v_pert - v0).abs()
        delta_norm = delta.norm(dim=-1)
        lip_adv = (diff / (delta_norm + 1e-8)).cpu().numpy()

    return {
        "eps_rel":  eps_rel,
        "n_steps":  n_steps,
        "median":   float(np.median(lip_adv)),
        "p95":      float(np.percentile(lip_adv, 95)),
        "max":      float(lip_adv.max()),
        "lip_adv_samples": lip_adv,
    }


# ===========================================================================
# VERDICT
# ===========================================================================

def verdict(pred: dict, lip: dict) -> tuple[str, list[str]]:
    """
    Tre livelli:
      A: prediction + smoothness ok → pronto per DRO
      B: un aspetto debole
      C: entrambi problematici
    """
    issues = []

    # Prediction criterion
    if pred["spearman"] < 0.3:
        issues.append(f"Spearman ρ={pred['spearman']:.2f} (target >0.3)")
    if pred["r2"] < 0.05:
        issues.append(f"R²={pred['r2']:.3f} (target >0.05)")

    # Smoothness — too high Lipschitz breaks DRO solver
    if lip["p95"] > 10.0:
        issues.append(f"Lipschitz p95={lip['p95']:.1f} (target <10)")

    if not issues:
        return "A", []
    if len(issues) == 1:
        return "B", issues
    return "C", issues


# ===========================================================================
# PRINTING
# ===========================================================================

def print_scorecard(pred: dict, pw: dict, lip: dict) -> None:
    names = ["low_vol", "mid_vol", "high_vol"]

    print("\n" + "=" * 70)
    print("PREDICTION QUALITY (Scatter & Correlation)")
    print("=" * 70)
    print(f"  N samples : {len(pred['V']):,}")
    print(f"  R²        : {pred['r2']:+.4f}")
    print(f"  Spearman ρ: {pred['spearman']:+.4f}")
    print(f"  MAE       : {pred['mae']:.4f}")

    print(f"\n  Per regime:")
    for r, name in enumerate(names):
        if r in pred["per_regime"]:
            p = pred["per_regime"][r]
            print(f"    {name:10s} (n={p['n']:>7,}): "
                  f"ρ={p['spearman']:+.3f}  R²={p['r2']:+.3f}")

    print("\n" + "=" * 70)
    print("PAIRWISE RANKING ACCURACY (Scale-invariant ranking)")
    print("=" * 70)
    print(f"  Global     : {pw['global']:.4f}")
    for r, name in enumerate(names):
        k = f"regime_{r}"
        if k in pw:
            print(f"  {name:10s} : {pw[k]:.4f}")

    print("\n" + "=" * 70)
    print("SMOOTHNESS (Lipschitz rispetto a z)")
    print("=" * 70)
    print(f"  Global     : med={lip['median']:.3f}  p95={lip['p95']:.3f}")
    for r, name in enumerate(names):
        if r in lip["per_regime"]:
            pr = lip["per_regime"][r]
            print(f"  {name:10s} : med={pr['median']:.3f}  p95={pr['p95']:.3f}")
    print("=" * 70 + "\n")


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_all(
    pred:       dict,
    pw:         dict,
    lip:        dict,
    out_path:   str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    regime_names = ["low_vol", "mid_vol", "high_vol"]
    regime_colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # -----------------------------------------------------------------------
    # Row 1 — Prediction quality (Scatter plots per regime)
    # -----------------------------------------------------------------------
    V, G = pred["V"], pred["G"]
    reg = pred["reg"]

    for r in range(3):
        ax = fig.add_subplot(gs[0, r])
        mask = (reg == r)
        
        if mask.sum() == 0:
            ax.set_title(f"{regime_names[r]} (no data)")
            ax.axis("off")
            continue
            
        Vr = V[mask]
        Gr = G[mask]
        
        if len(Vr) > 3000:
            rng = np.random.default_rng(r)
            sub = rng.choice(len(Vr), size=3000, replace=False)
            V_plot, G_plot = Vr[sub], Gr[sub]
        else:
            V_plot, G_plot = Vr, Gr

        ax.scatter(G_plot, V_plot, s=4, alpha=0.3, c=regime_colors[r])
        
        lims = [min(V_plot.min(), G_plot.min()), max(V_plot.max(), G_plot.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
        
        ax.set_xlabel("G (True Return)", fontsize=10)
        if r == 0:
            ax.set_ylabel("V (Critic Prediction)", fontsize=10)
            
        r2_val = pred["per_regime"][r]["r2"] if r in pred["per_regime"] else np.nan
        rho_val = pred["per_regime"][r]["spearman"] if r in pred["per_regime"] else np.nan
        var_g = np.var(Gr)
        
        ax.set_title(f"{regime_names[r]} Scatter\n"
                     f"R²={r2_val:+.2f}  |  ρ={rho_val:+.2f}  |  Var(G)={var_g:.2f}",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # -----------------------------------------------------------------------
    # Row 2 — DRO validation & Rankings
    # -----------------------------------------------------------------------

    # (1,0) Pairwise Ranking Accuracy
    ax = fig.add_subplot(gs[1, 0])
    regs_pw = [r for r in [0, 1, 2] if f"regime_{r}" in pw]
    x = np.arange(len(regs_pw))
    w = 0.5
    pw_vals = [pw[f"regime_{r}"] for r in regs_pw]
    names_pw = [regime_names[r] for r in regs_pw]
    colors_pw = [regime_colors[r] for r in regs_pw]

    bars = ax.bar(x, pw_vals, w, color=colors_pw, alpha=0.8, edgecolor="white")
    ax.axhline(0.5, color="black", ls="--", lw=1, alpha=0.5, label="Random baseline (0.5)")
    for bar, v in zip(bars, pw_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names_pw, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(f"Pairwise Ranking Accuracy\n"
                 f"(Global = {pw['global']:.3f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # (1,1) Lipschitz Smoothness
    ax = fig.add_subplot(gs[1, 1])
    regs_lip = [r for r in [0, 1, 2] if r in lip["per_regime"]]
    x = np.arange(len(regs_lip))
    w = 0.35
    med = [lip["per_regime"][r]["median"] for r in regs_lip]
    p95 = [lip["per_regime"][r]["p95"]    for r in regs_lip]
    names_lip = [regime_names[r] for r in regs_lip]

    ax.bar(x - w/2, med, w, color="#3498db", label="median")
    ax.bar(x + w/2, p95, w, color="#e67e22", label="p95")

    ax.set_xticks(x)
    ax.set_xticklabels(names_lip, fontsize=10)
    ax.set_ylabel(r"$\|\nabla_z V\|_2$", fontsize=10)
    ax.set_title(f"Lipschitz Smoothness (w.r.t z)\n"
                 f"(Global p95 = {lip['p95']:.3f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (1,2) Summary Metrics Block
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    summary_lines = [
        "GLOBAL METRICS",
        "------------------",
        f"Target        : MC Returns (G)",
        f"Model R²      : {pred['r2']:+.3f}",
        f"Spearman ρ    : {pred['spearman']:+.3f}",
        f"MAE           : {pred['mae']:.3f}",
        f"Bias          : {pred['bias']:+.3f}",
        f"Ranking Acc   : {pw['global']:.1%}",
        f"Lipschitz p95 : {lip['p95']:.1f}",
    ]

    ax.text(0.5, 0.5, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=12, va="center", ha="center", family="monospace",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa",
                      edgecolor="#dee2e6", alpha=0.9))

    plt.suptitle("Critic V_θ(s) Evaluation Scorecard",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


def pairwise_accuracy(V: np.ndarray, G: np.ndarray, reg: np.ndarray,
                      n_pairs: int = 50000, seed: int = 42) -> dict:
    """
    Pairwise accuracy: % di coppie (a,b) dove sign(V_a - V_b) == sign(G_a - G_b).
    Metrica principale per ranking critic (scale-invariant).
    """
    rng = np.random.default_rng(seed)
    N = len(V)
    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N, size=n_pairs)
    # Avoid same-index pairs
    mask_diff = idx_a != idx_b
    idx_a = idx_a[mask_diff]
    idx_b = idx_b[mask_diff]

    diff_v = V[idx_a] - V[idx_b]
    diff_g = G[idx_a] - G[idx_b]

    # Ignore ties in G (exact)
    valid = diff_g != 0
    diff_v = diff_v[valid]
    diff_g = diff_g[valid]

    acc_global = float((np.sign(diff_v) == np.sign(diff_g)).mean())

    # Per-regime: pair is "regime r" if both ends have regime r
    result = {"global": acc_global, "n_pairs": len(diff_v)}
    reg_a = reg[idx_a][valid]
    reg_b = reg[idx_b][valid]
    for r_id in [0, 1, 2]:
        mask_r = (reg_a == r_id) & (reg_b == r_id)
        if mask_r.sum() > 20:
            a = (np.sign(diff_v[mask_r]) == np.sign(diff_g[mask_r])).mean()
            result[f"regime_{r_id}"] = float(a)

    # Cross-regime pairs
    mask_cross = (reg_a != reg_b)
    if mask_cross.sum() > 20:
        a = (np.sign(diff_v[mask_cross]) == np.sign(diff_g[mask_cross])).mean()
        result["cross_regime"] = float(a)

    return result


# ===========================================================================
# MAIN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load critic ---
    critic, ckpt = load_critic(args.ckpt, device)
    cfg = ckpt["cfg"]
    reward_stats = ckpt["reward_stats"]
    g_stats = ckpt.get("g_stats")    # v2 only
    gamma   = cfg["gamma"]
    inv_max = cfg["inv_max"]
    mode    = cfg.get("mode", "mc")
    is_rank = (mode == "rank")
    use_actions  = cfg.get("use_actions", False)
    action_stats = ckpt.get("action_stats")
    z_stats      = ckpt.get("z_stats")              # v3.1
    z_norm_flag  = cfg.get("z_norm", False)
    print(f"  reward_stats μ={reward_stats['mean']:+.6f}  σ={reward_stats['std']:.6f}")
    if g_stats is not None:
        print(f"  g_stats μ={g_stats['mean']:+.6f}  σ={g_stats['std']:.6f}  (v2)")
    if use_actions and action_stats is not None:
        print(f"  action_stats: q_max={action_stats['q_max']:.3f}  "
              f"L_levels={action_stats.get('L_levels', 10)}  (v3)")
    if z_norm_flag and z_stats is not None:
        print(f"  z_stats: per-dim std mean={z_stats['std'].mean():.4f}  (v3.1)")
    print(f"  gamma={gamma}  inv_max={inv_max}  mode={mode}  "
          f"use_actions={use_actions}  z_norm={z_norm_flag}")
    if is_rank:
        print(f"  [!] RANKING CRITIC: V scale is indeterminate, "
              f"focus on ordinal metrics (Spearman, pairwise acc)")

    # --- Load data ---
    data = load_data(
        args.dataset, reward_stats, gamma, inv_max,
        n_samples=args.n_samples,
        val_only=args.val_only,
        max_t_for_mc=args.max_t_for_mc,
        seed=args.seed,
        g_stats=g_stats,
        action_stats=action_stats,
        use_actions=use_actions,
        z_stats=z_stats if z_norm_flag else None,
    )

    # --- Section 1: Prediction quality ---
    print("\n[1/3] Evaluating prediction quality...")
    pred = prediction_quality(critic, data, device)

    # --- Section 2: Pairwise Ranking Accuracy ---
    print("\n[2/3] Evaluating pairwise ranking accuracy...")
    pw = pairwise_accuracy(
        pred["V"], pred["G"], pred["reg"],
        n_pairs=args.n_pairwise, seed=args.seed,
    )

    # --- Section 3: Smoothness ---
    print("\n[3/3] Evaluating smoothness (Lipschitz)...")
    lip = lipschitz_analysis(critic, data, device,
                             n_samples=args.n_lip_samples,
                             seed=args.seed)

    # --- Print scorecard ---
    print_scorecard(pred, pw, lip)

    # --- Plot ---
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_all(pred, pw, lip, out_path=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Critic V_θ(s)")
    parser.add_argument("--ckpt",     type=str, default="checkpoints/critic_rank/critic_best.pt")
    parser.add_argument("--dataset",  type=str, default="data/wm_dataset.npz")
    parser.add_argument("--n_samples", type=int, default=20_000,
                        help="Number of (s, G) pairs for prediction eval")
    parser.add_argument("--val_only", action="store_true",
                        help="Use only val split (episode-based, 10%%)")
    parser.add_argument("--max_t_for_mc", type=int, default=15,
                        help="Use MC returns only for t in [0, max_t_for_mc]")
    parser.add_argument("--n_pd_samples",  type=int, default=500)
    parser.add_argument("--n_lip_samples", type=int, default=5000)
    parser.add_argument("--n_pairwise", type=int, default=100_000,
                        help="N random pairs for pairwise accuracy (ranking mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default="eval_critic.png")
    args = parser.parse_args()
    main(args)
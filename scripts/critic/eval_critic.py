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

  Row 2 — SENSIBILITY ("il critico usa tutti gli input?")
    (1,0): Gradient magnitudes — IL TEST KILLER
           ||∇_z V|| / √d_z  vs  |∂V/∂inv|  vs  |∂V/∂tl|
    (1,1): Partial dependence su inv (per regime)
    (1,2): Partial dependence su tl (per regime)

  Row 3 — SMOOTHNESS & DRO COMPATIBILITY
    (2,0): Lipschitz empirica globale + per regime
    (2,1): PGD adversarial Lipschitz (worst-case)
    (2,2): Per-dimension sensitivity (quale dim di z conta di più?)

Scorecard con verdetto A/B/C/D:
  A: tutto ok, critico pronto per DRO
  B: prediction ok ma gradient su z debole → DRO funzionerà con poco leverage
  C: prediction debole ma sensibility ok → ri-allenare
  D: entrambi rotti → ripensare

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
sys.path.insert(0, str(Path(__file__).parent.parent))

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
) -> dict:
    """
    Carica dataset, costruisce stati aumentati e MC returns.

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
    Lipschitz empirica globale + per regime.
    Lipschitz = ||∇_s V(s)|| (per sample), poi statistiche.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(data["s"]))
    idx = rng.choice(len(data["s"]), size=n, replace=False)
    s = data["s"][idx].to(device).requires_grad_(True)
    reg = data["reg"][idx].numpy()

    v = critic(s)
    grads = torch.autograd.grad(v.sum(), s, create_graph=False)[0]
    lip = grads.norm(dim=-1).cpu().numpy()               # (n,)

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

def verdict(pred: dict, sens: dict, lip: dict) -> tuple[str, list[str]]:
    """
    Quattro livelli:
      A: tutto ok → pronto per DRO
      B: prediction ok ma gradient su z debole → DRO funzionerà con poco leverage
      C: prediction debole ma sensibility ok → ri-allenare
      D: entrambi rotti → ripensare
    """
    prediction_issues = []
    sensibility_issues = []

    # Prediction criterion (MC returns are noisy, thresholds soft)
    if pred["spearman"] < 0.3:
        prediction_issues.append(f"Spearman ρ={pred['spearman']:.2f} (target >0.3)")
    if pred["r2"] < 0.05:
        prediction_issues.append(f"R²={pred['r2']:.3f} (target >0.05 on noisy MC)")

    # Sensibility criterion — the key one for DRO
    ratio = sens["ratio_z_vs_scalars"]
    if ratio < 0.05:
        sensibility_issues.append(
            f"gradient ratio z/[inv,tl]={ratio:.3f} — "
            "critico essenzialmente ignora z (DRO senza leverage)"
        )
    elif ratio < 0.15:
        sensibility_issues.append(
            f"gradient ratio z/[inv,tl]={ratio:.3f} — "
            "leverage su z debole"
        )

    # Smoothness — too high Lipschitz breaks DRO solver
    if lip["p95"] > 10.0:
        sensibility_issues.append(f"Lipschitz p95={lip['p95']:.1f} — alta")

    if not prediction_issues and not sensibility_issues:
        return "A", []
    if not prediction_issues and sensibility_issues:
        return "B", sensibility_issues
    if prediction_issues and not sensibility_issues:
        return "C", prediction_issues
    return "D", prediction_issues + sensibility_issues


# ===========================================================================
# PRINTING
# ===========================================================================

def print_scorecard(
    pred:       dict,
    sens:       dict,
    lip:        dict,
    lip_adv:    dict,
    verdict_letter: str,
    verdict_issues: list[str],
) -> None:
    names = ["low_vol", "mid_vol", "high_vol"]

    print("\n" + "=" * 70)
    print("PREDICTION QUALITY — il critico predice bene G?")
    print("=" * 70)
    print(f"  Global:")
    print(f"    N samples : {len(pred['V']):,}")
    print(f"    MSE       : {pred['mse']:.4f}")
    print(f"    MAE       : {pred['mae']:.4f}")
    print(f"    bias      : {pred['bias']:+.4f}")
    print(f"    Spearman ρ: {pred['spearman']:+.4f}   (correlation rank-based)")
    print(f"    Pearson r : {pred['pearson']:+.4f}")
    print(f"    R²        : {pred['r2']:+.4f}  (MC noisy; >0.05 già rilevante)")

    print(f"\n  Per regime:")
    for r, name in enumerate(names):
        if r in pred["per_regime"]:
            p = pred["per_regime"][r]
            print(f"    {name:10s} (n={p['n']:>7,}): "
                  f"MSE={p['mse']:.4f}  ρ={p['spearman']:+.3f}  R²={p['r2']:+.3f}")

    print("\n" + "=" * 70)
    print("SENSIBILITY — il critico usa tutte le feature?")
    print("=" * 70)
    d_action = sens.get("d_action", 0)
    print(f"  Gradient magnitudes (mean over samples):")
    print(f"    ||∇_z V||/√d_z : {sens['per_unit_z']:.4f}    (per-unit-dim)")
    print(f"    |∂V/∂inv|      : {sens['per_unit_inv']:.4f}")
    print(f"    |∂V/∂tl|       : {sens['per_unit_tl']:.4f}")
    if d_action > 0:
        print(f"    ||∇_a V||/√d_a : {sens['per_unit_actions']:.4f}    "
              f"(per-unit-dim, d_action={d_action})")

    ratio = sens["ratio_z_vs_scalars"]
    status = ("OK " if ratio > 0.15 else "~~ " if ratio > 0.05 else "!! ")
    print(f"    ratio z vs (inv+tl)/2       = {ratio:.3f}   [{status}]")
    if d_action > 0:
        r_all = sens["ratio_z_vs_all"]
        s_all = ("OK " if r_all > 0.15 else "~~ " if r_all > 0.05 else "!! ")
        print(f"    ratio z vs (inv+tl+a)/3     = {r_all:.3f}   [{s_all}]  (v3)")

    print(f"\n  Per-dim |∂V/∂z_d|  (media):")
    pd_z = sens["per_dim_z"]
    for i in range(0, len(pd_z), 8):
        chunk = pd_z[i:i+8]
        s_line = "  ".join(f"d{i+j:02d}={chunk[j]:.3f}" for j in range(len(chunk)))
        print(f"    {s_line}")
    print(f"    max={pd_z.max():.3f}  min={pd_z.min():.3f}  "
          f"max/min={pd_z.max()/(pd_z.min()+1e-8):.1f}")

    if d_action > 0 and sens.get("abs_actions") is not None:
        abs_a = sens["abs_actions"]
        labels = ["k_bid", "k_ask", "q_bid", "q_ask"]
        print(f"\n  Per-dim |∂V/∂action_d|:")
        print(f"    " + "  ".join(
            f"{lbl}={abs_a[i]:.4f}" for i, lbl in enumerate(labels)
        ))

    print("\n" + "=" * 70)
    print("SMOOTHNESS — compatibile con DRO solver?")
    print("=" * 70)
    print(f"  Lipschitz empirica (random sampling):")
    print(f"    median={lip['median']:.3f}  p95={lip['p95']:.3f}  max={lip['max']:.3f}")
    print(f"\n  Per regime (median):")
    for r, name in enumerate(names):
        if r in lip["per_regime"]:
            pr = lip["per_regime"][r]
            print(f"    {name:10s}: med={pr['median']:.3f}  p95={pr['p95']:.3f}")

    print(f"\n  PGD adversarial Lipschitz (eps_rel={lip_adv['eps_rel']:.0e}, "
          f"n_steps={lip_adv['n_steps']}):")
    print(f"    median={lip_adv['median']:.3f}  p95={lip_adv['p95']:.3f}  "
          f"max={lip_adv['max']:.3f}")

    print("\n" + "=" * 70)
    print(f"VERDETTO: {verdict_letter}")
    print("=" * 70)
    if verdict_letter == "A":
        print("  Critico validato e pronto per il DRO (Modulo C, parte 2).")
    elif verdict_letter == "B":
        print("  Prediction OK ma gradient su z debole.")
        for iss in verdict_issues:
            print(f"    - {iss}")
        print("  Impatto: il DRO avrà poco leverage sulla perturbazione di z.")
        print("  Possibili cause: reward dominato da inv/tl, poco segnale di")
        print("  mercato nelle dinamiche del simulatore.")
    elif verdict_letter == "C":
        print("  Prediction debole.")
        for iss in verdict_issues:
            print(f"    - {iss}")
        print("  Possibili fix: più epoche, gp_weight diverso, learning rate.")
    else:
        print("  Critico non funzionante.")
        for iss in verdict_issues:
            print(f"    - {iss}")
        print("  Ripensare training o architettura.")
    print("=" * 70 + "\n")


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_all(
    pred:       dict,
    sens:       dict,
    pd_inv:     dict,
    pd_tl:      dict,
    lip:        dict,
    lip_adv:    dict,
    out_path:   str,
) -> None:
    regime_names = ["low_vol", "mid_vol", "high_vol"]
    regime_colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig = plt.figure(figsize=(17, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.30)

    # -----------------------------------------------------------------------
    # Row 1 — Prediction quality
    # -----------------------------------------------------------------------

    # (0,0) scatter V vs G
    ax = fig.add_subplot(gs[0, 0])
    V, G = pred["V"], pred["G"]
    # Subsample for plotting (too many points otherwise)
    if len(V) > 5000:
        rng = np.random.default_rng(0)
        sub = rng.choice(len(V), size=5000, replace=False)
        V_plot = V[sub]; G_plot = G[sub]; reg_plot = pred["reg"][sub]
    else:
        V_plot = V; G_plot = G; reg_plot = pred["reg"]

    for r in range(3):
        mask = (reg_plot == r)
        if mask.sum() > 0:
            ax.scatter(G_plot[mask], V_plot[mask], s=4, alpha=0.4,
                       c=regime_colors[r], label=regime_names[r])
    lims = [min(V_plot.min(), G_plot.min()), max(V_plot.max(), G_plot.max())]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="identity")
    ax.set_xlabel("G (MC return, true)", fontsize=9)
    ax.set_ylabel("V (critic prediction)", fontsize=9)
    ax.set_title(f"V vs G  —  Spearman ρ={pred['spearman']:.3f}  "
                 f"R²={pred['r2']:.3f}", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # (0,1) per-regime metrics bars
    ax = fig.add_subplot(gs[0, 1])
    regs_with_data = [r for r in [0, 1, 2] if r in pred["per_regime"]]
    x = np.arange(len(regs_with_data))
    w = 0.35
    mse_vals = [pred["per_regime"][r]["mse"] for r in regs_with_data]
    rho_vals = [pred["per_regime"][r]["spearman"] for r in regs_with_data]
    names_plot = [regime_names[r] for r in regs_with_data]
    cols_plot = [regime_colors[r] for r in regs_with_data]

    # Primary axis: MSE
    ax.bar(x - w/2, mse_vals, w, color=cols_plot, alpha=0.7, label="MSE")
    ax.set_ylabel("MSE", fontsize=9, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(names_plot, fontsize=8)
    for i, v in enumerate(mse_vals):
        ax.text(i - w/2, v + max(mse_vals)*0.02, f"{v:.3f}", ha="center", fontsize=7)

    # Secondary axis: Spearman
    ax2 = ax.twinx()
    ax2.bar(x + w/2, rho_vals, w, color=cols_plot, alpha=0.4, hatch="//",
            edgecolor="black", linewidth=0.5, label="Spearman ρ")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Spearman ρ", fontsize=9, color="#666")
    ax2.set_ylim(-0.1, max(1.0, max(rho_vals) * 1.1))
    for i, v in enumerate(rho_vals):
        ax2.text(i + w/2, v + 0.03 if v >= 0 else v - 0.05, f"{v:+.3f}",
                 ha="center", fontsize=7)

    ax.set_title("Per-regime prediction quality", fontsize=10)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)

    # (0,2) error distribution
    ax = fig.add_subplot(gs[0, 2])
    err = pred["errors"]
    for r in range(3):
        if r in pred["per_regime"]:
            mask = (pred["reg"] == r)
            ax.hist(err[mask], bins=40, alpha=0.5, density=True,
                    color=regime_colors[r], label=regime_names[r])
    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.7)
    ax.axvline(pred["bias"], color="red", lw=1.5, ls=":",
               label=f"bias={pred['bias']:+.3f}")
    ax.set_xlabel("V - G", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title("Distribuzione errori per regime", fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # -----------------------------------------------------------------------
    # Row 2 — Sensibility (THE KILLER ROW)
    # -----------------------------------------------------------------------

    # (1,0) gradient magnitudes — the key panel
    ax = fig.add_subplot(gs[1, 0])
    labels = [r"$\|\nabla_z V\|/\sqrt{d_z}$", r"$|\partial V/\partial\,\mathrm{inv}|$",
              r"$|\partial V/\partial\,\mathrm{tl}|$"]
    vals   = [sens["per_unit_z"], sens["per_unit_inv"], sens["per_unit_tl"]]
    colors_g = ["#8e44ad", "#e67e22", "#16a085"]
    bars = ax.bar(labels, vals, color=colors_g, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ratio = sens["ratio_z_vs_scalars"]
    status = "OK" if ratio > 0.15 else "WEAK" if ratio > 0.05 else "ZERO LEVERAGE"
    ax.set_title(f"Gradient magnitudes (mean)\n"
                 f"ratio z/([inv,tl]/2) = {ratio:.3f}  [{status}]", fontsize=10)
    ax.set_ylabel("Magnitude", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (1,1) partial dependence on inv
    ax = fig.add_subplot(gs[1, 1])
    for r, name in enumerate(regime_names):
        if r in pd_inv["curves"]:
            ax.plot(pd_inv["grid"], pd_inv["curves"][r],
                    color=regime_colors[r], lw=2, label=name, marker="o", markersize=3)
    ax.axvline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("inv_norm  (−1=short  +1=long)", fontsize=9)
    ax.set_ylabel("V (mean over z, tl fixed)", fontsize=9)
    ax.set_title("Partial dependence on inventory", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # (1,2) partial dependence on tl
    ax = fig.add_subplot(gs[1, 2])
    for r, name in enumerate(regime_names):
        if r in pd_tl["curves"]:
            ax.plot(pd_tl["grid"], pd_tl["curves"][r],
                    color=regime_colors[r], lw=2, label=name, marker="o", markersize=3)
    ax.set_xlabel("time_left  (0=end  1=start)", fontsize=9)
    ax.set_ylabel("V (mean over z, inv fixed)", fontsize=9)
    ax.set_title("Partial dependence on time_left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # -----------------------------------------------------------------------
    # Row 3 — Smoothness
    # -----------------------------------------------------------------------

    # (2,0) Lipschitz per regime
    ax = fig.add_subplot(gs[2, 0])
    regs_with_lip = [r for r in [0, 1, 2] if r in lip["per_regime"]]
    x = np.arange(len(regs_with_lip))
    w = 0.25
    med = [lip["per_regime"][r]["median"] for r in regs_with_lip]
    p95 = [lip["per_regime"][r]["p95"]    for r in regs_with_lip]
    mx  = [lip["per_regime"][r]["max"]    for r in regs_with_lip]
    names_plot = [regime_names[r] for r in regs_with_lip]

    ax.bar(x - w,   med, w, color="#3498db", label="median")
    ax.bar(x,       p95, w, color="#e67e22", label="p95")
    ax.bar(x + w,   mx,  w, color="#e74c3c", label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(names_plot, fontsize=8)
    ax.set_ylabel(r"$\|\nabla_s V\|_2$", fontsize=9)
    ax.set_title(f"Lipschitz per regime\n"
                 f"global median={lip['median']:.2f}  "
                 f"p95={lip['p95']:.2f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # (2,1) PGD adversarial Lipschitz — hist
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(lip_adv["lip_adv_samples"], bins=40, color="#e67e22",
            edgecolor="white", alpha=0.85)
    ax.axvline(lip_adv["median"], color="black", ls="--", lw=1.5,
               label=f"median={lip_adv['median']:.2f}")
    ax.axvline(lip_adv["p95"], color="red", ls="--", lw=1.5,
               label=f"p95={lip_adv['p95']:.2f}")
    ax.set_xlabel("|V(s+δ) - V(s)| / ||δ||", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(f"PGD adversarial Lipschitz\n"
                 f"worst-case, eps_rel={lip_adv['eps_rel']:.0e}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # (2,2) per-dim sensitivity |∂V/∂z_d|
    ax = fig.add_subplot(gs[2, 2])
    pd_z = sens["per_dim_z"]
    d_z  = len(pd_z)
    mean_val = pd_z.mean()
    # Sort descending for clarity
    order = np.argsort(-pd_z)
    cols = ["#8e44ad" if pd_z[i] > mean_val * 0.5 else "#bdc3c7" for i in order]
    ax.bar(range(d_z), pd_z[order], color=cols, edgecolor="white")
    # Reference lines
    ax.axhline(sens["per_unit_inv"], color="#e67e22", ls="--", lw=1,
               label=f"|∂V/∂inv|={sens['per_unit_inv']:.3f}")
    ax.axhline(sens["per_unit_tl"], color="#16a085", ls="--", lw=1,
               label=f"|∂V/∂tl|={sens['per_unit_tl']:.3f}")
    ax.set_xlabel("dim of z (sorted by |∂V/∂z_d|)", fontsize=9)
    ax.set_ylabel("|∂V/∂z_d|  (mean)", fontsize=9)
    ax.set_title(f"Per-dim latent sensitivity\n"
                 f"max/min ratio = {pd_z.max()/(pd_z.min()+1e-8):.1f}",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Critic V_θ(s) — Evaluation Scorecard",
                 fontsize=15, fontweight="bold", y=0.995)
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
    print(f"  reward_stats μ={reward_stats['mean']:+.6f}  σ={reward_stats['std']:.6f}")
    if g_stats is not None:
        print(f"  g_stats μ={g_stats['mean']:+.6f}  σ={g_stats['std']:.6f}  (v2)")
    if use_actions and action_stats is not None:
        print(f"  action_stats: q_max={action_stats['q_max']:.3f}  "
              f"L_levels={action_stats.get('L_levels', 10)}  (v3)")
    print(f"  gamma={gamma}  inv_max={inv_max}  mode={mode}  "
          f"use_actions={use_actions}")
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
    )

    # --- Section 1: Prediction quality ---
    print("\n[1/3] Evaluating prediction quality...")
    pred = prediction_quality(critic, data, device)

    # --- Extra section for ranking: pairwise accuracy ---
    pw = None
    if is_rank:
        print("[1b/3] Evaluating pairwise ranking accuracy...")
        pw = pairwise_accuracy(
            pred["V"], pred["G"], pred["reg"],
            n_pairs=args.n_pairwise, seed=args.seed,
        )

    # --- Section 2: Sensibility ---
    print("[2/3] Evaluating sensibility (gradient analysis + partial dependence)...")
    sens   = gradient_magnitudes(critic, data, device,
                                 n_samples=args.n_gradient_samples,
                                 seed=args.seed)
    pd_inv = partial_dependence(critic, data, device, variable="inv",
                                n_grid=21, n_base=args.n_pd_samples,
                                seed=args.seed)
    pd_tl  = partial_dependence(critic, data, device, variable="tl",
                                n_grid=21, n_base=args.n_pd_samples,
                                seed=args.seed)

    # --- Section 3: Smoothness ---
    print("[3/3] Evaluating smoothness (Lipschitz + PGD)...")
    lip    = lipschitz_analysis(critic, data, device,
                                n_samples=args.n_lip_samples,
                                seed=args.seed)
    lip_adv = pgd_adversarial_lipschitz(critic, data, device,
                                         eps_rel=args.pgd_eps,
                                         n_steps=args.pgd_steps,
                                         n_samples=args.n_pgd_samples,
                                         seed=args.seed)

    # --- Verdict ---
    verdict_letter, verdict_issues = verdict(pred, sens, lip)

    # --- Print scorecard ---
    print_scorecard(pred, sens, lip, lip_adv, verdict_letter, verdict_issues)

    # --- Ranking-specific section ---
    if pw is not None:
        print("\n" + "=" * 70)
        print("PAIRWISE RANKING ACCURACY (scale-invariant)")
        print("=" * 70)
        print(f"  Global     (n={pw['n_pairs']:,} pairs): {pw['global']:.4f}")
        for r_id, name in enumerate(["low_vol", "mid_vol", "high_vol"]):
            k = f"regime_{r_id}"
            if k in pw:
                print(f"  {name:>10s} within-regime: {pw[k]:.4f}")
        if "cross_regime" in pw:
            print(f"  cross-regime              : {pw['cross_regime']:.4f}")
        print(f"\n  Random baseline: 0.500")
        print(f"  Perfect critic : 1.000 (impossibile dato il rumore di G)")
        print(f"  Interpretazione:")
        print(f"    >0.70 = ordinamento forte")
        print(f"    0.55-0.70 = ordinamento moderato (realistico per MC rumoroso)")
        print(f"    <0.55 = ordinamento debole")

    # --- Plot ---
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_all(pred, sens, pd_inv, pd_tl, lip, lip_adv, out_path=args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Critic V_θ(s)")
    parser.add_argument("--ckpt",     type=str, default="checkpoints/critic_best.pt")
    parser.add_argument("--dataset",  type=str, default="data/wm_dataset.npz")
    parser.add_argument("--n_samples", type=int, default=20_000,
                        help="Number of (s, G) pairs for prediction eval")
    parser.add_argument("--val_only", action="store_true",
                        help="Use only val split (episode-based, 10%)")
    parser.add_argument("--max_t_for_mc", type=int, default=15,
                        help="Use MC returns only for t in [0, max_t_for_mc] "
                             "(later steps have biased MC due to truncation)")
    parser.add_argument("--n_gradient_samples", type=int, default=5000)
    parser.add_argument("--n_pd_samples",       type=int, default=500)
    parser.add_argument("--n_lip_samples",      type=int, default=5000)
    parser.add_argument("--n_pgd_samples",      type=int, default=500)
    parser.add_argument("--pgd_eps",   type=float, default=0.01)
    parser.add_argument("--pgd_steps", type=int,   default=20)
    parser.add_argument("--n_pairwise", type=int, default=100_000,
                        help="N random pairs for pairwise accuracy (ranking mode)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--out",       type=str,   default="eval_critic.png")
    args = parser.parse_args()
    main(args)
#!/usr/bin/env python3
"""
Oracle test: misura il Bayes floor per pairwise ranking accuracy su G.

DOMANDA: il critico (0.648) è al limite information-theoretic di ciò che è
possibile fare, o lascia segnale sul tavolo?

METODO: l'oracle conosce (inv, tl, regime) di ogni stato e predice il MIGLIOR
stimatore puntuale possibile dato queste informazioni, ossia E[G | bucket].
Poi misuriamo quanto bene questo oracle ordina coppie random di G.

Se Oracle arriva a ~0.68-0.70 → il critico è già quasi ottimo su questo
segnale; la varianza residua di G dato (inv, tl, regime) è irriducibile.

Se Oracle arriva a 0.80+ → il critico lascia segnale sul tavolo, probabilmente
non sa combinare bene inv×tl×regime. Da indagare architetture diverse.

L'oracle include anche la granularità raffinata su z: per ogni bucket
calcoliamo E[G|bucket] globale, ma anche E[G|bucket, z] in modo non-parametrico
(kNN su z dentro il bucket). Questo dà l'upper bound realistico di quanto
segnale c'è dentro z.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


# Bucket definitions (from diagnose_interactions.py)
INV_NAMES = ["very_short", "short", "flat", "long", "very_long"]
INV_EDGES = [-np.inf, -10.0, -2.0, 2.0, 10.0, np.inf]    # on raw inv
TL_NAMES  = ["end", "mid_late", "mid_early", "start"]
TL_EDGES  = [0.0, 0.25, 0.5, 0.75, 1.0]
REGIME_NAMES = ["low_vol", "mid_vol", "high_vol"]


def bucket_inv(inv_raw: np.ndarray) -> np.ndarray:
    """inv raw → bucket id."""
    return np.digitize(inv_raw, INV_EDGES[1:-1])


def bucket_tl(tl: np.ndarray) -> np.ndarray:
    """tl in [0,1] → bucket id (tl=1 start, tl=0 end)."""
    b = np.digitize(tl, TL_EDGES[1:-1])
    return np.clip(b, 0, len(TL_NAMES) - 1)


def compute_g_mean_per_bucket(
    G:     np.ndarray,        # (N,)
    inv_b: np.ndarray,        # (N,) bucket ids
    tl_b:  np.ndarray,        # (N,) bucket ids
    reg:   np.ndarray,        # (N,) regime ids
) -> np.ndarray:
    """
    Per ogni bucket (inv_b, tl_b, reg), calcola E[G|bucket] usando TUTTI i sample.
    Buckets under-sampled (<20) usano la mean globale come fallback.

    Returns:
        g_oracle : (N,) — per ogni sample, la mean di G nel suo bucket
    """
    n_inv = len(INV_NAMES)
    n_tl  = len(TL_NAMES)
    n_reg = len(REGIME_NAMES)

    g_oracle = np.zeros_like(G)
    g_global = G.mean()

    for i in range(n_inv):
        for t in range(n_tl):
            for r in range(n_reg):
                mask = (inv_b == i) & (tl_b == t) & (reg == r)
                n = mask.sum()
                if n >= 20:
                    g_oracle[mask] = G[mask].mean()
                else:
                    g_oracle[mask] = g_global

    return g_oracle


def pairwise_accuracy(
    predictor: np.ndarray,
    G:         np.ndarray,
    n_pairs:   int = 200_000,
    seed:      int = 42,
) -> dict:
    """
    Pairwise ranking accuracy: % di coppie (a,b) dove
    sign(predictor_a - predictor_b) == sign(G_a - G_b).

    Ignora i pair dove diff_G == 0 (eseguendo un filtro sui ties).
    """
    rng = np.random.default_rng(seed)
    N = len(G)

    idx_a = rng.integers(0, N, size=n_pairs)
    idx_b = rng.integers(0, N, size=n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    diff_p = predictor[idx_a] - predictor[idx_b]
    diff_g = G[idx_a] - G[idx_b]

    # Ignore ties in G (exact) AND in predictor (when oracle has coarse bucketing,
    # many pair share same E[G|bucket] so diff_p=0 which would count as wrong)
    valid = diff_g != 0
    diff_p = diff_p[valid]
    diff_g = diff_g[valid]

    # For ties in predictor (common with bucket oracle), count as 0.5
    sign_p = np.sign(diff_p)
    sign_g = np.sign(diff_g)

    correct = (sign_p == sign_g).astype(np.float32)
    ties    = (sign_p == 0)
    correct[ties] = 0.5   # tie in predictor → coin flip

    acc = float(correct.mean())
    frac_ties = float(ties.mean())

    return {
        "acc":       acc,
        "n_pairs":   len(diff_p),
        "frac_ties": frac_ties,
    }


def main(args: argparse.Namespace) -> None:
    print(f"Loading dataset: {args.dataset}")
    data = np.load(args.dataset)
    inventories = data["inventories"].astype(np.float32)   # (M, N+1)
    time_left   = data["time_left"].astype(np.float32)     # (M, N+1)
    rewards     = data["rewards"].astype(np.float32)       # (M, N)
    regimes     = data["regimes"].astype(np.int64)         # (M,) or (M, N)
    episode_ids = data["episode_ids"]

    M, Np1 = inventories.shape
    N = Np1 - 1

    # Compute MC returns (raw G)
    print(f"Computing MC returns (γ={args.gamma})...")
    G = np.zeros_like(rewards)
    G[:, -1] = rewards[:, -1]
    for t in range(N - 2, -1, -1):
        G[:, t] = rewards[:, t] + args.gamma * G[:, t + 1]

    # Regime per-step
    if regimes.ndim == 1:
        reg_step = regimes[:, None].repeat(N, axis=1)
    else:
        reg_step = regimes[:, :N]

    # Flat
    inv_flat = inventories[:, :N].reshape(-1)
    tl_flat  = time_left[:, :N].reshape(-1)
    reg_flat = reg_step.reshape(-1)
    G_flat   = G.reshape(-1)

    # Val-only subset (same seed del training)
    if args.val_only:
        rng = np.random.default_rng(42)
        unique_eps = np.unique(episode_ids)
        rng.shuffle(unique_eps)
        n_val = max(1, int(len(unique_eps) * args.val_frac))
        val_eps = set(unique_eps[:n_val])
        # Mask per-step
        ep_rep = np.repeat(episode_ids, N)
        val_mask = np.array([e in val_eps for e in ep_rep])
        inv_flat = inv_flat[val_mask]
        tl_flat  = tl_flat[val_mask]
        reg_flat = reg_flat[val_mask]
        G_flat   = G_flat[val_mask]
        print(f"  val split: {len(G_flat):,} samples from {n_val} episodes")
    else:
        print(f"  full dataset: {len(G_flat):,} samples")

    print(f"  G range: [{G_flat.min():+.2f}, {G_flat.max():+.2f}]  "
          f"mean={G_flat.mean():+.3f}  std={G_flat.std():.3f}")

    # --- Oracle 1: global mean (trivial baseline) ---
    print("\n" + "=" * 70)
    print("ORACLE 1 — Trivial: predictor = G_mean (costante)")
    print("=" * 70)
    g_trivial = np.full_like(G_flat, G_flat.mean())
    r1 = pairwise_accuracy(g_trivial, G_flat, n_pairs=args.n_pairs, seed=args.seed)
    print(f"  Pairwise acc: {r1['acc']:.4f}  "
          f"(n_pairs={r1['n_pairs']:,}, frac_ties={r1['frac_ties']:.3f})")
    print(f"  Expected: 0.5 (tutto tie → coin flip). Sanity check.")

    # --- Oracle 2: bucket oracle (inv, tl, regime) ---
    print("\n" + "=" * 70)
    print("ORACLE 2 — Bucket oracle: predictor = E[G | inv_b, tl_b, regime]")
    print("=" * 70)
    inv_b = bucket_inv(inv_flat)
    tl_b  = bucket_tl(tl_flat)
    g_oracle = compute_g_mean_per_bucket(G_flat, inv_b, tl_b, reg_flat)
    r2 = pairwise_accuracy(g_oracle, G_flat, n_pairs=args.n_pairs, seed=args.seed)
    print(f"  Pairwise acc: {r2['acc']:.4f}  "
          f"(n_pairs={r2['n_pairs']:,}, frac_ties={r2['frac_ties']:.3f})")
    print(f"  Interpretazione:")
    print(f"    Questo è il BAYES FLOOR empirico usando solo (inv, tl, regime).")
    print(f"    Qualunque critico che vede (inv, tl, regime) come feature può al")
    print(f"    massimo arrivare qui (modulo piccoli guadagni da z).")

    # --- Oracle 3: bucket oracle (solo inv + tl, NO regime) ---
    # Simula un critico che ignora completamente il regime
    print("\n" + "=" * 70)
    print("ORACLE 3 — Bucket oracle SENZA regime: predictor = E[G | inv_b, tl_b]")
    print("=" * 70)
    # Compute E[G | inv_b, tl_b] ignoring regime
    g_no_regime = np.zeros_like(G_flat)
    for i in range(len(INV_NAMES)):
        for t in range(len(TL_NAMES)):
            mask = (inv_b == i) & (tl_b == t)
            if mask.sum() >= 20:
                g_no_regime[mask] = G_flat[mask].mean()
    r3 = pairwise_accuracy(g_no_regime, G_flat, n_pairs=args.n_pairs, seed=args.seed)
    print(f"  Pairwise acc: {r3['acc']:.4f}  "
          f"(n_pairs={r3['n_pairs']:,}, frac_ties={r3['frac_ties']:.3f})")
    print(f"  Δ vs Oracle 2: {r2['acc'] - r3['acc']:+.4f}")
    print(f"  (quanto ACC aggiunge il regime? Se piccolo, z contribuisce poco)")

    # --- Oracle 4: solo regime ---
    print("\n" + "=" * 70)
    print("ORACLE 4 — Solo regime: predictor = E[G | regime]")
    print("=" * 70)
    g_only_regime = np.zeros_like(G_flat)
    for r_id in range(3):
        mask = (reg_flat == r_id)
        if mask.sum() >= 20:
            g_only_regime[mask] = G_flat[mask].mean()
    r4 = pairwise_accuracy(g_only_regime, G_flat, n_pairs=args.n_pairs, seed=args.seed)
    print(f"  Pairwise acc: {r4['acc']:.4f}")

    # --- Oracle 5: solo inv ---
    print("\n" + "=" * 70)
    print("ORACLE 5 — Solo inv: predictor = E[G | inv_b]")
    print("=" * 70)
    g_only_inv = np.zeros_like(G_flat)
    for i in range(len(INV_NAMES)):
        mask = (inv_b == i)
        if mask.sum() >= 20:
            g_only_inv[mask] = G_flat[mask].mean()
    r5 = pairwise_accuracy(g_only_inv, G_flat, n_pairs=args.n_pairs, seed=args.seed)
    print(f"  Pairwise acc: {r5['acc']:.4f}")

    # --- Oracle 6: solo tl ---
    print("\n" + "=" * 70)
    print("ORACLE 6 — Solo tl: predictor = E[G | tl_b]")
    print("=" * 70)
    g_only_tl = np.zeros_like(G_flat)
    for t in range(len(TL_NAMES)):
        mask = (tl_b == t)
        if mask.sum() >= 20:
            g_only_tl[mask] = G_flat[mask].mean()
    r6 = pairwise_accuracy(g_only_tl, G_flat, n_pairs=args.n_pairs, seed=args.seed)
    print(f"  Pairwise acc: {r6['acc']:.4f}")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("SINTESI — Quanto segnale hanno le diverse features?")
    print("=" * 70)
    print(f"  Random baseline              : 0.500")
    print(f"  Oracle solo inv              : {r5['acc']:.4f}  "
          f"(gain vs random: {r5['acc'] - 0.5:+.4f})")
    print(f"  Oracle solo tl               : {r6['acc']:.4f}  "
          f"(gain vs random: {r6['acc'] - 0.5:+.4f})")
    print(f"  Oracle solo regime           : {r4['acc']:.4f}  "
          f"(gain vs random: {r4['acc'] - 0.5:+.4f})")
    print(f"  Oracle inv + tl              : {r3['acc']:.4f}  "
          f"(gain vs random: {r3['acc'] - 0.5:+.4f})")
    print(f"  Oracle inv + tl + regime     : {r2['acc']:.4f}  "
          f"(gain vs random: {r2['acc'] - 0.5:+.4f})")
    print()
    print(f"  Marginal gain di regime      : {r2['acc'] - r3['acc']:+.4f}")
    print(f"  → questo è L'UPPER BOUND su quanto z può aggiungere alla accuracy,")
    print(f"    perché z è l'UNICO segnale usato dal critico per inferire il regime.")
    print()
    print(f"  Confronto al critico attuale:")
    print(f"    Critico v2.3 / v3 rank (misurato): ~0.648")
    print(f"    Bayes floor (inv + tl + regime)  : {r2['acc']:.4f}")
    gap = r2['acc'] - 0.648
    if gap < 0.02:
        verdict = "Critico al Bayes floor. Niente da guadagnare."
    elif gap < 0.05:
        verdict = "Critico vicino al Bayes floor. Guadagni marginali possibili."
    else:
        verdict = "Critico sotto il Bayes floor. Architettura da rivedere."
    print(f"    Gap: {gap:+.4f}  → {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle test: Bayes floor for pairwise ranking")
    parser.add_argument("--dataset",  type=str, default="data/wm_dataset.npz")
    parser.add_argument("--gamma",    type=float, default=0.95)
    parser.add_argument("--val_only", action="store_true",
                        help="Use only val split (same seed as training)")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--n_pairs",  type=int,   default=200_000)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()
    main(args)
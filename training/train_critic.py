"""
train_critic.py — Training offline del critico V_θ(s).

Stato aumentato:
  s_t = [z_t, inv_norm_t, tl_t]  ∈ R^(d_z + 2)

  z_t        : stato latente LOB da wm_dataset.npz
  inv_norm_t : clip(inventory_t / inv_max, -1, 1)
  tl_t       : time_left_t ∈ [0, 1], già nel dataset

Due modalità:
  --mode td : TD(0) con target network EMA + switch_mask (default)
  --mode mc : Monte Carlo returns

Gradient penalty (--gp_weight): mantiene Lipschitz bound soft.
Switch mask (TD mode): esclude i passi di cambio regime dal TD loss
  — il bootstrap V(z_{t+1}) è invalido se il regime cambia esogenamente.

Split episodico: sequenze dello stesso episodio non vengono divise tra
  train e val (evita leakage da sequenze sovrapposte).

Uso:
  python training/train_critic.py --mode mc --gp_weight 0.1
  python training/train_critic.py --mode td --gp_weight 0.1 --inv_max 5.0
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.critic import ValueNetwork

# ---------------------------------------------------------------------------
# Augmented state builder
# ---------------------------------------------------------------------------

def build_augmented_sequences(
    sequences:   np.ndarray,   # (M, N+1, d_z)
    inventories: np.ndarray,   # (M, N+1)
    time_left:   np.ndarray,   # (M, N+1)
    inv_max:     float,
    actions:     np.ndarray | None = None,   # (M, N, 4) or None — v3
    action_stats: dict | None = None,
    L_levels:    int = 10,
    z_stats:     dict | None = None,         # v3.1: per-dim z-score
) -> np.ndarray:
    """
    Concatena z_t (opzionalmente z-scored) con inv_norm_t, tl_t e opzionalmente action_t.

    v3.1 z-normalization (optional):
      Se z_stats è fornito ({"mean": (d_z,), "std": (d_z,)}), applica
      z_norm = (z - z_mean) / z_std prima di concatenare. Questo allinea
      le scale delle feature (z tipicamente ha std~0.02 per dim, inv/tl
      hanno range [0,1]/[-1,1]).

    v3 state augmentation (optional):
      Se actions fornite, concatena anche [k_bid, k_ask, q_bid, q_ask]
      normalized in [0,1].

    Returns:
        aug : (M, N+1, d_z + 2 + d_action)
    """
    # v3.1: z-score per-dim (se z_stats fornite)
    if z_stats is not None:
        z_mean = z_stats["mean"]        # (d_z,)
        z_std  = z_stats["std"]         # (d_z,)
        sequences = (sequences - z_mean) / z_std

    inv_norm = np.clip(inventories / inv_max, -1.0, 1.0)[..., None]   # (M, N+1, 1)
    tl       = time_left[..., None]                                     # (M, N+1, 1)

    parts = [sequences, inv_norm, tl]

    if actions is not None:
        M, N, d_a = actions.shape
        assert d_a == 4, f"Expected actions shape (M, N, 4), got (M, N, {d_a})"
        # Normalize k (indices 0, 1) and q (indices 2, 3)
        k = actions[..., :2]                           # (M, N, 2)
        q = actions[..., 2:]                           # (M, N, 2)

        k_norm = np.clip((k - 1.0) / max(1e-8, L_levels - 1), 0.0, 1.0)

        if action_stats is not None and "q_max" in action_stats:
            q_max = action_stats["q_max"]
        else:
            q_max = float(np.percentile(q, 99.5))      # robust max
            q_max = max(q_max, 1e-6)

        q_norm = np.clip(q / q_max, 0.0, 1.0)          # (M, N, 2)

        a_norm = np.concatenate([k_norm, q_norm], axis=-1)  # (M, N, 4)

        # Align: pad last step by repeating a_{N-1}
        a_full = np.concatenate([a_norm, a_norm[:, -1:, :]], axis=1)   # (M, N+1, 4)
        parts.append(a_full)

    return np.concatenate(parts, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Episode-based split (evita leakage da sequenze sovrapposte)
# ---------------------------------------------------------------------------

def episode_split(
    episode_ids: np.ndarray,
    val_frac:    float = 0.1,
    seed:        int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    unique_eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)
    n_val   = max(1, int(len(unique_eps) * val_frac))
    val_set = set(unique_eps[:n_val])
    train_idx = np.where([ep not in val_set for ep in episode_ids])[0]
    val_idx   = np.where([ep in val_set     for ep in episode_ids])[0]
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CriticSequenceDataset(Dataset):
    """
    Sequence-level dataset per TD learning.

    Ogni sample: (s_seq, r_seq, switch_mask, regime)
      s_seq       : (N+1, d_z+2)  — stati aumentati
      r_seq       : (N,)           — reward raw
      switch_mask : (N,)  int8     — 1 = cambio regime a step t
      regime      : scalar         — regime prevalente della sequenza
    """

    def __init__(self, path: str, inv_max: float) -> None:
        data = np.load(path)

        aug = build_augmented_sequences(
            data["sequences"].astype(np.float32),
            data["inventories"].astype(np.float32),
            data["time_left"].astype(np.float32),
            inv_max=inv_max,
        )

        self.sequences   = torch.from_numpy(aug)
        self.rewards     = torch.from_numpy(data["rewards"].astype(np.float32))
        self.regimes     = torch.from_numpy(data["regimes"].astype(np.int64))
        self.episode_ids = data["episode_ids"]

        if "switch_mask" in data:
            self.switch_mask = torch.from_numpy(data["switch_mask"].astype(np.int8))
        else:
            M, Np1 = self.sequences.shape[:2]
            self.switch_mask = torch.zeros(M, Np1 - 1, dtype=torch.int8)

        # regimes per-episode scalar → espandi a per-step se necessario
        if self.regimes.dim() == 1 and self.regimes.shape[0] == len(self.sequences):
            N = self.sequences.shape[1] - 1
            self.regimes = self.regimes.unsqueeze(1).expand(-1, N)

        M, Np1, D = self.sequences.shape
        print(f"CriticSequenceDataset: {M:,} seq  N={Np1-1}  d_state={D}")

    def __len__(self)  -> int: return len(self.sequences)
    def __getitem__(self, i):
        return (
            self.sequences[i],    # (N+1, d_state)
            self.rewards[i],      # (N,)
            self.switch_mask[i],  # (N,)
            self.regimes[i],      # (N,)
        )


class CriticFlatDataset(Dataset):
    """
    Flat (s_t, G_t) pairs per MC learning.

    v2 changes:
      - G_t ora è z-scored sulle stats globali (g_mean, g_std) computate
        sul dataset (o su un subset fornito esternamente). Questo rende
        il target stabile numericamente, range [-3, +3] tipico.
      - reward_stats non serve più per calcolare G (usiamo reward raw),
        ma lo teniamo nel checkpoint per compat.

    G_t = Σ_{k≥0} γ^k r_{t+k}  (su reward RAW, non normalizzati)
    G_t_norm = (G_t - g_mean) / g_std
    """

    def __init__(
        self,
        path:          str,
        inv_max:       float,
        gamma:         float,
        g_stats:       dict | None = None,
        use_actions:   bool = False,              # v3: include actions in state
        action_stats:  dict | None = None,        # v3: q_max, L_levels
        z_stats:       dict | None = None,        # v3.1: per-dim z-score
    ) -> None:
        data = np.load(path)

        # v3: load actions if requested
        actions = None
        if use_actions:
            if "actions" not in data.files:
                raise ValueError(
                    "Dataset missing 'actions' key but use_actions=True. "
                    "Rigenera wm_dataset con build_wm_dataset.py (include actions)."
                )
            actions = data["actions"].astype(np.float32)
            # Expected shape: (M, N, 4)  [k_bid, k_ask, q_bid, q_ask]
            assert actions.ndim == 3, f"actions shape {actions.shape} (expected 3D)"
            # Il dataset potrebbe avere 3 o 4 componenti a seconda della versione del simulatore.
            # Forziamo 4 componenti: se ne abbiamo 3, c'è un bug upstream, solleviamo.
            assert actions.shape[-1] == 4, (
                f"actions last dim = {actions.shape[-1]}, expected 4 "
                f"([k_bid, k_ask, q_bid, q_ask]). Controlla il dataset."
            )

        aug = build_augmented_sequences(
            data["sequences"].astype(np.float32),
            data["inventories"].astype(np.float32),
            data["time_left"].astype(np.float32),
            inv_max=inv_max,
            actions=actions,
            action_stats=action_stats,
            L_levels=(action_stats.get("L_levels", 10) if action_stats else 10),
            z_stats=z_stats,
        )

        rews = data["rewards"].astype(np.float32)
        regs = data["regimes"].astype(np.int64)
        self.episode_ids = data["episode_ids"]

        M, Np1, D = aug.shape
        N = Np1 - 1

        # Calcola MC returns su reward RAW (no normalizzazione reward-level)
        returns = np.zeros_like(rews)
        returns[:, -1] = rews[:, -1]
        for t in range(N - 2, -1, -1):
            returns[:, t] = rews[:, t] + gamma * returns[:, t + 1]

        # v2.2: winsorization opzionale (attivata solo se winsor_active=True nel g_stats)
        if g_stats is not None and g_stats.get("winsor_active", False):
            p_low  = g_stats["p_low"]
            p_high = g_stats["p_high"]
            n_clip_low  = (returns < p_low).sum()
            n_clip_high = (returns > p_high).sum()
            returns = np.clip(returns, p_low, p_high)
            print(f"  Winsorized: clipped {n_clip_low:,} low (<{p_low:.3f}) "
                  f"and {n_clip_high:,} high (>{p_high:.3f}) samples")

        # Z-score G: se g_stats fornite, usa quelle; altrimenti compute qui
        if g_stats is None:
            g_mean = float(returns.mean())
            g_std  = float(returns.std())
        else:
            g_mean = g_stats["mean"]
            g_std  = g_stats["std"]

        returns_norm = (returns - g_mean) / (g_std + 1e-8)

        # Flat
        self.s   = torch.from_numpy(aug[:, :N].reshape(-1, D))
        self.g   = torch.from_numpy(returns_norm.reshape(-1))
        self.reg = torch.from_numpy(
            regs[:, None].repeat(N, axis=1).reshape(-1)
            if regs.ndim == 1
            else regs.reshape(-1)
        )

        # Keep stats for checkpoint
        self.g_mean = g_mean
        self.g_std  = g_std

        print(
            f"CriticFlatDataset (v2.1): {len(self.s):,} samples  d_state={D}  "
            f"G_processed mean={returns.mean():+.4f}  std={returns.std():.4f}"
        )
        print(
            f"  z-scored G_norm  mean={self.g.mean().item():+.4f}  "
            f"std={self.g.std().item():.4f}  "
            f"range=[{self.g.min().item():+.2f}, {self.g.max().item():+.2f}]"
        )

    def __len__(self)  -> int: return len(self.s)
    def __getitem__(self, i): return self.s[i], self.g[i], self.reg[i]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_reward_stats(path: str) -> dict:
    data  = np.load(path)
    r     = data["rewards"].astype(np.float32).reshape(-1)
    stats = {"mean": float(r.mean()), "std": float(r.std())}
    print(f"Reward stats: μ={stats['mean']:.6f}  σ={stats['std']:.6f}")
    return stats


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(1 - tau).add_(p_o.data, alpha=tau)


def _compute_action_stats_from_train_split(
    dataset_path: str,
    val_frac:     float,
    seed:         int = 42,
    L_levels:     int = 10,
) -> dict:
    """
    Calcola q_max per normalizzazione delle actions sul train split
    (episode-based), per evitare leakage.

    actions: (M, N, 4)  [k_bid, k_ask, q_bid, q_ask]
    - k è già in [1, L_levels]; normalizzazione deterministica: (k-1)/(L-1)
    - q ha range variabile; usiamo il 99.5° percentile come q_max robusto

    Returns:
        dict con q_max, L_levels
    """
    data = np.load(dataset_path)
    if "actions" not in data.files:
        raise ValueError(f"Dataset {dataset_path} missing 'actions'")

    ep_ids = data["episode_ids"]
    actions = data["actions"].astype(np.float32)   # (M, N, 4)

    train_idx, _ = episode_split(ep_ids, val_frac=val_frac, seed=seed)
    train_q = actions[train_idx, :, 2:].reshape(-1)   # q_bid & q_ask

    q_max = float(np.percentile(train_q, 99.5))
    q_min = float(train_q.min())
    q_mean = float(train_q.mean())

    print(f"Action stats (train split):")
    print(f"  q range raw : [{q_min:.2f}, {train_q.max():.2f}]  mean={q_mean:.2f}")
    print(f"  q_max (p99.5)={q_max:.2f}  (usato per normalizzazione)")
    print(f"  L_levels={L_levels}  (k normalizzata come (k-1)/(L-1))")

    return {
        "q_max":    q_max,
        "L_levels": L_levels,
        "q_min":    q_min,
        "q_mean":   q_mean,
    }


def _compute_z_stats_from_train_split(
    dataset_path: str,
    val_frac:     float,
    seed:         int = 42,
) -> dict:
    """
    v3.1: z-score per-dim del latente z sul train split (episode-based).

    L'encoder v5 ha output z con std ≈0.02 per dim, mentre le altre feature
    dello stato (inv_norm, tl, actions_norm) sono in [0,1] o [-1,1] con std
    dell'ordine di 0.3-0.6. Questa discrepanza di scala fa sì che i gradient
    magnitudes di V rispetto a z appaiano sistematicamente piccoli vs le
    altre feature, indipendentemente dal contenuto informativo.

    Fix: normalizzazione per-dim z sul train split. Le stats vengono salvate
    nel ckpt per uso in eval (zero leakage).

    Returns:
        dict con "mean" (d_z,), "std" (d_z,)
    """
    data = np.load(dataset_path)
    ep_ids = data["episode_ids"]
    sequences = data["sequences"].astype(np.float32)    # (M, N+1, d_z)

    train_idx, _ = episode_split(ep_ids, val_frac=val_frac, seed=seed)
    train_z = sequences[train_idx].reshape(-1, sequences.shape[-1])  # (M_tr*(N+1), d_z)

    z_mean = train_z.mean(axis=0).astype(np.float32)   # (d_z,)
    z_std  = train_z.std(axis=0).astype(np.float32)    # (d_z,)
    z_std  = np.maximum(z_std, 1e-6)                   # avoid div-by-zero

    print(f"Z stats (train split, {len(train_idx):,} sequences):")
    print(f"  per-dim mean: range=[{z_mean.min():+.4f}, {z_mean.max():+.4f}]  "
          f"|mean|.mean={np.abs(z_mean).mean():.4f}")
    print(f"  per-dim std : range=[{z_std.min():.4f}, {z_std.max():.4f}]  "
          f"mean={z_std.mean():.4f}")

    return {
        "mean": z_mean,
        "std":  z_std,
    }


def _compute_g_stats_from_train_split(
    dataset_path: str,
    gamma:        float,
    val_frac:     float,
    seed:         int = 42,
    winsor_low:   float = 0.0,
    winsor_high:  float = 100.0,
) -> dict:
    """
    Calcola g_mean, g_std sul train split (episode-based) per z-score.

    v2.2 (no winsorization):
      Dopo analisi diagnostica abbiamo visto che la winsorization a 1-99%
      taglia proprio la parte più informativa del segnale (bucket inv=|estremo|
      + tl=end dove G varia tra regimi). Ora usiamo G raw, affidandoci alla
      Huber loss con delta alto per la robustezza agli outlier.

      Se winsor_low > 0 o winsor_high < 100, winsorization è riattivata
      (per ablation).

    Args:
        winsor_low:  percentile low clip (0.0 = disabilitato)
        winsor_high: percentile high clip (100.0 = disabilitato)
    """
    data = np.load(dataset_path)
    ep_ids = data["episode_ids"]
    rews   = data["rewards"].astype(np.float32)
    M, N = rews.shape

    # Compute MC returns (raw reward)
    returns = np.zeros_like(rews)
    returns[:, -1] = rews[:, -1]
    for t in range(N - 2, -1, -1):
        returns[:, t] = rews[:, t] + gamma * returns[:, t + 1]

    # Train split
    train_idx, _ = episode_split(ep_ids, val_frac=val_frac, seed=seed)
    train_returns = returns[train_idx].reshape(-1)

    # Winsorization (opzionale, default disabled in v2.2)
    apply_winsor = (winsor_low > 0.0) or (winsor_high < 100.0)
    if apply_winsor:
        p_low  = float(np.percentile(train_returns, winsor_low))
        p_high = float(np.percentile(train_returns, winsor_high))
        processed = np.clip(train_returns, p_low, p_high)
    else:
        p_low  = float(train_returns.min())
        p_high = float(train_returns.max())
        processed = train_returns

    stats = {
        "mean":   float(processed.mean()),
        "std":    float(processed.std()),
        "p_low":  p_low,
        "p_high": p_high,
        "winsor_low":  winsor_low,
        "winsor_high": winsor_high,
        "winsor_active": apply_winsor,
    }
    print(f"G stats (train split, {len(train_idx):,} sequences):")
    print(f"  raw   : mean={train_returns.mean():+.4f}  std={train_returns.std():.4f}  "
          f"range=[{train_returns.min():+.2f}, {train_returns.max():+.2f}]")
    if apply_winsor:
        print(f"  winsor: p{winsor_low}={p_low:+.4f}  p{winsor_high}={p_high:+.4f}  "
              f"(clipped {(train_returns < p_low).mean()*100:.2f}% low, "
              f"{(train_returns > p_high).mean()*100:.2f}% high)")
        print(f"  final : mean={stats['mean']:+.4f}  std={stats['std']:.4f}")
    else:
        print(f"  no winsor (v2.2): using raw returns for z-score")
    return stats


# ---------------------------------------------------------------------------
# MC Training (v2)
# ---------------------------------------------------------------------------

def train_mc(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Mode: MC  loss=Huber(delta={args.huber_delta})  "
          f"spectral_norm={not args.no_spectral}  "
          f"use_actions={args.use_actions}")

    # reward_stats compat, ma non usato per z-score (lo facciamo su G direttamente)
    reward_stats = compute_reward_stats(args.dataset)

    # Compute g_stats on train split (episode-based), to be used for z-scoring
    g_stats = _compute_g_stats_from_train_split(
        args.dataset, gamma=args.gamma,
        val_frac=args.val_frac, seed=42,
        winsor_low=args.winsor_low,
        winsor_high=args.winsor_high,
    )

    # v3: action_stats per normalizzazione delle quote A-S
    action_stats = None
    if args.use_actions:
        action_stats = _compute_action_stats_from_train_split(
            args.dataset, val_frac=args.val_frac, seed=42,
            L_levels=args.L_levels,
        )

    # v3.1: z_stats per normalizzazione per-dim del latente
    z_stats = None
    if args.z_norm:
        z_stats = _compute_z_stats_from_train_split(
            args.dataset, val_frac=args.val_frac, seed=42,
        )

    # Build dataset with z-scored G using train-split stats (zero leakage)
    full_ds = CriticFlatDataset(
        args.dataset, inv_max=args.inv_max,
        gamma=args.gamma, g_stats=g_stats,
        use_actions=args.use_actions,
        action_stats=action_stats,
        z_stats=z_stats,
    )

    # Split episodico — le sequenze flat da stesso episodio vanno nello stesso split
    ep_ids_per_seq = np.load(args.dataset)["episode_ids"]
    M_seq = len(ep_ids_per_seq)
    N_per_seq = len(full_ds) // M_seq

    train_seq_idx, val_seq_idx = episode_split(ep_ids_per_seq, args.val_frac, seed=42)
    train_flat = np.concatenate([
        np.arange(i * N_per_seq, (i + 1) * N_per_seq) for i in train_seq_idx
    ])
    val_flat = np.concatenate([
        np.arange(i * N_per_seq, (i + 1) * N_per_seq) for i in val_seq_idx
    ])
    train_ds = Subset(full_ds, train_flat)
    val_ds   = Subset(full_ds, val_flat)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # v3: stato layout = [z, inv, tl, (actions)]
    d_state  = full_ds.s.shape[-1]
    d_action = 4 if args.use_actions else 0
    d_z      = d_state - 2 - d_action

    critic = ValueNetwork(
        d_state=d_state, d_z=d_z, d_action=d_action,
        hidden=args.hidden, n_layers=args.n_layers,
        use_spectral=not args.no_spectral,
    ).to(device)
    version_str = "v3" if args.use_actions else "v2"
    print(f"Critic ({version_str}): {sum(p.numel() for p in critic.parameters()):,} params  "
          f"d_state={d_state}  d_z={d_z}  d_action={d_action}  hidden={args.hidden}")

    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Huber loss (smooth L1): robusta a outlier nel target MC noisy
    huber = nn.HuberLoss(delta=args.huber_delta, reduction="mean")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        critic.train()
        t0 = time.time()
        tr_loss = 0.0

        for s, g, reg in train_loader:
            s = s.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)

            v_pred = critic(s)
            loss   = huber(v_pred, g)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        critic.eval()
        val_loss = 0.0
        val_per_regime: dict[int, list] = {0: [], 1: [], 2: []}
        with torch.no_grad():
            for s, g, reg in val_loader:
                s   = s.to(device,   non_blocking=True)
                g   = g.to(device,   non_blocking=True)
                reg = reg.to(device, non_blocking=True)
                v   = critic(s)
                val_loss += huber(v, g).item()
                for r in [0, 1, 2]:
                    m = (reg == r)
                    if m.sum() > 0:
                        val_per_regime[r].append(
                            huber(v[m], g[m]).item())

        val_loss /= len(val_loader)
        scheduler.step()

        reg_str = "  ".join(
            f"{['low','mid','high'][r]}={np.mean(v):.4f}"
            for r, v in val_per_regime.items() if v
        )
        print(f"Ep {epoch:3d}/{args.epochs}  train={tr_loss:.4f}  "
              f"val={val_loss:.4f}  [{reg_str}]  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.1f}s")

        ckpt = {
            "epoch": epoch, "val_loss": val_loss,
            "model": critic.state_dict(),
            "reward_stats": reward_stats,
            "g_stats":      g_stats,
            "action_stats": action_stats,                  # v3: None se use_actions=False
            "z_stats":      z_stats,                       # v3.1: None se z_norm=False
            "cfg": {
                "d_state":     d_state,
                "d_z":         d_z,
                "d_action":    d_action,                    # v3
                "hidden":      args.hidden,
                "n_layers":    args.n_layers,
                "gamma":       args.gamma,
                "inv_max":     args.inv_max,
                "mode":        "mc",
                "version":     version_str,                 # "v2" or "v3"
                "use_actions": args.use_actions,
                "L_levels":    args.L_levels,
                "z_norm":      args.z_norm,                 # v3.1
                "use_spectral": not args.no_spectral,
                "huber_delta": args.huber_delta,
            },
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "critic_best.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"critic_ep{epoch:03d}.pt")

    critic.eval()
    s_val  = full_ds.s[torch.randperm(len(full_ds.s))[:5000]].to(device)
    L_emp  = critic.estimate_lipschitz(s_val)
    print(f"\nBest val loss: {best_val:.4f}  Lipschitz: {L_emp:.2f}")

    best_ckpt = torch.load(ckpt_dir / "critic_best.pt", weights_only=False)
    best_ckpt["lipschitz_estimate"] = L_emp
    torch.save(best_ckpt, ckpt_dir / "critic_best.pt")
    print(f"Checkpoint: {ckpt_dir / 'critic_best.pt'}")


# ---------------------------------------------------------------------------
# TD Training
# ---------------------------------------------------------------------------

def train_td(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Mode: TD(0)  τ={args.tau}  gp_weight={args.gp_weight}")

    reward_stats = compute_reward_stats(args.dataset)
    full_ds = CriticSequenceDataset(args.dataset, inv_max=args.inv_max)

    train_idx, val_idx = episode_split(full_ds.episode_ids, args.val_frac)
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    d_state = full_ds.sequences.shape[-1]
    critic  = ValueNetwork(d_state=d_state, hidden=args.hidden,
                           n_layers=args.n_layers).to(device)
    target  = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}  "
          f"d_state={d_state}")

    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    r_mean = torch.tensor(reward_stats["mean"], device=device)
    r_std  = torch.tensor(reward_stats["std"] + 1e-8, device=device)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        critic.train()
        t0 = time.time()
        tr_loss = tr_gp = 0.0

        for s_seq, r_seq, sw_mask, reg in train_loader:
            s_seq   = s_seq.to(device,   non_blocking=True)   # (B, N+1, d_state)
            r_seq   = r_seq.to(device,   non_blocking=True)   # (B, N)
            sw_mask = sw_mask.to(device, non_blocking=True)   # (B, N)
            B, Np1, D = s_seq.shape
            N = Np1 - 1

            r_norm = (r_seq - r_mean) / r_std                 # (B, N)
            s_t    = s_seq[:, :N].reshape(B * N, D)           # (B*N, d_state)
            v_pred = critic(s_t).reshape(B, N)                # (B, N)

            with torch.no_grad():
                s_next  = s_seq[:, 1:].reshape(B * N, D)
                v_next  = target(s_next).reshape(B, N)
                td_tgt  = r_norm + args.gamma * v_next        # (B, N)

            # Maschera switch steps: il bootstrap è invalido se il regime
            # cambia esogenamente tra t e t+1
            valid   = (sw_mask == 0).float()                  # (B, N)
            n_valid = valid.sum().clamp(min=1.0)
            mse     = ((v_pred - td_tgt) ** 2 * valid).sum() / n_valid

            loss = mse
            if args.gp_weight > 0:
                gp   = critic.gradient_penalty(s_t)
                loss = loss + args.gp_weight * gp
                tr_gp += gp.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            ema_update(target, critic, args.tau)
            tr_loss += mse.item()

        n_b     = len(train_loader)
        tr_loss /= n_b
        tr_gp   /= n_b

        critic.eval()
        val_loss = 0.0
        val_per_regime: dict[int, list] = {0: [], 1: [], 2: []}
        with torch.no_grad():
            for s_seq, r_seq, sw_mask, reg in val_loader:
                s_seq   = s_seq.to(device,   non_blocking=True)
                r_seq   = r_seq.to(device,   non_blocking=True)
                sw_mask = sw_mask.to(device, non_blocking=True)
                reg     = reg.to(device,     non_blocking=True)
                B, Np1, D = s_seq.shape
                N = Np1 - 1

                r_norm  = (r_seq - r_mean) / r_std
                s_t     = s_seq[:, :N].reshape(B * N, D)
                v_pred  = critic(s_t).reshape(B, N)
                s_next  = s_seq[:, 1:].reshape(B * N, D)
                v_next  = target(s_next).reshape(B, N)
                td_tgt  = r_norm + args.gamma * v_next

                valid   = (sw_mask == 0).float()
                n_valid = valid.sum().clamp(min=1.0)
                batch_loss = ((v_pred - td_tgt) ** 2 * valid).sum() / n_valid
                val_loss  += batch_loss.item()

                # Per-regime: usa il regime al primo step della sequenza
                reg_scalar = reg[:, 0] if reg.dim() == 2 else reg
                for r in [0, 1, 2]:
                    m = (reg_scalar == r)
                    if m.sum() > 0:
                        vp_r  = v_pred[m]
                        td_r  = td_tgt[m]
                        vld_r = valid[m]
                        n_r   = vld_r.sum().clamp(min=1.0)
                        val_per_regime[r].append(
                            (((vp_r - td_r) ** 2 * vld_r).sum() / n_r).item()
                        )

        val_loss /= len(val_loader)
        scheduler.step()

        reg_str = "  ".join(
            f"{['low','mid','high'][r]}={np.mean(v):.4f}"
            for r, v in val_per_regime.items() if v
        )
        gp_str = f"  gp={tr_gp:.4f}" if args.gp_weight > 0 else ""
        print(f"Ep {epoch:3d}/{args.epochs}  train={tr_loss:.4f}{gp_str}  "
              f"val={val_loss:.4f}  [{reg_str}]  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.1f}s")

        ckpt = {
            "epoch": epoch, "val_loss": val_loss,
            "model":        critic.state_dict(),
            "target_model": target.state_dict(),
            "reward_stats": reward_stats,
            "cfg": {"d_state": d_state, "hidden": args.hidden,
                    "n_layers": args.n_layers, "gamma": args.gamma,
                    "inv_max": args.inv_max, "mode": "td"},
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "critic_best.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"critic_ep{epoch:03d}.pt")

    critic.eval()
    s_est  = full_ds.sequences[:500, 0, :].to(device)
    L_emp  = critic.estimate_lipschitz(s_est)
    print(f"\nBest val loss: {best_val:.4f}  Lipschitz: {L_emp:.2f}")

    best_ckpt = torch.load(ckpt_dir / "critic_best.pt", weights_only=False)
    best_ckpt["lipschitz_estimate"] = L_emp
    torch.save(best_ckpt, ckpt_dir / "critic_best.pt")
    print(f"Checkpoint: {ckpt_dir / 'critic_best.pt'}")


# ---------------------------------------------------------------------------
# Rank Training (RankNet pairwise logistic)
# ---------------------------------------------------------------------------

def _rank_loss(v_a: torch.Tensor, v_b: torch.Tensor,
               g_a: torch.Tensor, g_b: torch.Tensor,
               margin: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RankNet pairwise logistic loss.

    L(a,b) = -log σ(sign(G_a - G_b) · (V_a - V_b))

    Se |G_a - G_b| < margin, il pair è "tie" e viene ignorato (loss 0).

    Returns:
        loss: scalare, media sui pair non-tie
        acc:  pairwise accuracy (sign(V_a - V_b) == sign(G_a - G_b))
    """
    diff_g = g_a - g_b                       # (B,)
    diff_v = v_a - v_b                       # (B,)

    # Ignore ties (|G_a - G_b| < margin)
    valid = diff_g.abs() > margin
    if valid.sum() == 0:
        return torch.tensor(0.0, device=v_a.device, requires_grad=True), \
               torch.tensor(0.0, device=v_a.device)

    sign_g = torch.sign(diff_g[valid])                  # (B',), ±1
    diff_v_v = diff_v[valid]                             # (B',)

    # log σ(sign · diff_v) = -log(1 + exp(-sign·diff_v))
    # Using F.logsigmoid for stability
    loss = -torch.nn.functional.logsigmoid(sign_g * diff_v_v).mean()

    # Pairwise accuracy (for monitoring)
    with torch.no_grad():
        acc = ((torch.sign(diff_v_v) == sign_g).float().mean())

    return loss, acc


def train_rank(args: argparse.Namespace) -> None:
    """
    Ranking training. Il critico non predice G, impara a ordinare gli stati.

    Procedura:
      1. Batch standard (s, G)
      2. All'interno del batch, creo coppie randomizzando l'ordine
         (b_1 vs b_2, b_3 vs b_4, ...) = batch_size/2 pairs per batch
      3. Applico RankNet logistic loss

    Metriche:
      - Pairwise accuracy (% coppie correttamente ordinate)
      - Per-regime pairwise accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Mode: RANK (RankNet)  "
          f"spectral_norm={not args.no_spectral}  "
          f"use_actions={args.use_actions}")

    reward_stats = compute_reward_stats(args.dataset)

    # For ranking we DON'T need winsorization (scale doesn't matter)
    # but keep g_stats for downstream DRO normalization compat
    g_stats = _compute_g_stats_from_train_split(
        args.dataset, gamma=args.gamma,
        val_frac=args.val_frac, seed=42,
        winsor_low=0.0, winsor_high=100.0,    # always raw for ranking
    )

    # v3: action_stats
    action_stats = None
    if args.use_actions:
        action_stats = _compute_action_stats_from_train_split(
            args.dataset, val_frac=args.val_frac, seed=42,
            L_levels=args.L_levels,
        )

    # v3.1: z_stats
    z_stats = None
    if args.z_norm:
        z_stats = _compute_z_stats_from_train_split(
            args.dataset, val_frac=args.val_frac, seed=42,
        )

    full_ds = CriticFlatDataset(
        args.dataset, inv_max=args.inv_max,
        gamma=args.gamma, g_stats=g_stats,
        use_actions=args.use_actions,
        action_stats=action_stats,
        z_stats=z_stats,
    )

    ep_ids_per_seq = np.load(args.dataset)["episode_ids"]
    M_seq = len(ep_ids_per_seq)
    N_per_seq = len(full_ds) // M_seq

    train_seq_idx, val_seq_idx = episode_split(ep_ids_per_seq, args.val_frac, seed=42)
    train_flat = np.concatenate([
        np.arange(i * N_per_seq, (i + 1) * N_per_seq) for i in train_seq_idx
    ])
    val_flat = np.concatenate([
        np.arange(i * N_per_seq, (i + 1) * N_per_seq) for i in val_seq_idx
    ])
    train_ds = Subset(full_ds, train_flat)
    val_ds   = Subset(full_ds, val_flat)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    # Batch size must be even for pair creation
    bs = args.batch_size
    if bs % 2 != 0:
        bs -= 1
        print(f"  batch_size rounded to {bs} (needs to be even)")

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # v3: stato layout = [z, inv, tl, (actions)]
    d_state  = full_ds.s.shape[-1]
    d_action = 4 if args.use_actions else 0
    d_z      = d_state - 2 - d_action
    critic   = ValueNetwork(
        d_state=d_state, d_z=d_z, d_action=d_action,
        hidden=args.hidden, n_layers=args.n_layers,
        use_spectral=not args.no_spectral,
    ).to(device)
    version_str = "v3" if args.use_actions else "v2"
    print(f"Critic ranking ({version_str}): "
          f"{sum(p.numel() for p in critic.parameters()):,} params  "
          f"d_state={d_state}  d_z={d_z}  d_action={d_action}  hidden={args.hidden}")

    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")         # we track val_loss but the real metric is val_acc
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        critic.train()
        t0 = time.time()
        tr_loss = tr_acc = 0.0
        n_batches = 0

        for s, g, reg in train_loader:
            s = s.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)

            # Split batch in two halves → create pairs
            B = s.shape[0]
            half = B // 2
            s_a, s_b = s[:half], s[half:2*half]
            g_a, g_b = g[:half], g[half:2*half]

            v_a = critic(s_a)
            v_b = critic(s_b)

            loss, acc = _rank_loss(v_a, v_b, g_a, g_b, margin=args.margin)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            tr_loss += loss.item()
            tr_acc  += acc.item()
            n_batches += 1

        tr_loss /= max(1, n_batches)
        tr_acc  /= max(1, n_batches)

        critic.eval()
        val_loss = val_acc = 0.0
        val_per_regime: dict[int, list] = {0: [], 1: [], 2: []}
        n_val_batches = 0
        with torch.no_grad():
            for s, g, reg in val_loader:
                s   = s.to(device,   non_blocking=True)
                g   = g.to(device,   non_blocking=True)
                reg = reg.to(device, non_blocking=True)

                B = s.shape[0]
                half = B // 2
                s_a, s_b = s[:half], s[half:2*half]
                g_a, g_b = g[:half], g[half:2*half]
                r_a = reg[:half]

                v_a = critic(s_a)
                v_b = critic(s_b)
                loss, acc = _rank_loss(v_a, v_b, g_a, g_b, margin=args.margin)
                val_loss += loss.item()
                val_acc  += acc.item()
                n_val_batches += 1

                # Per-regime accuracy: classify pair by regime of s_a
                for r_id in [0, 1, 2]:
                    m = (r_a == r_id)
                    if m.sum() > 0:
                        diff_g = g_a[m] - g_b[m]
                        diff_v = v_a[m] - v_b[m]
                        sign_g = torch.sign(diff_g)
                        sign_v = torch.sign(diff_v)
                        valid = diff_g.abs() > args.margin
                        if valid.sum() > 0:
                            a = (sign_v[valid] == sign_g[valid]).float().mean().item()
                            val_per_regime[r_id].append(a)

        val_loss /= max(1, n_val_batches)
        val_acc  /= max(1, n_val_batches)
        scheduler.step()

        reg_str = "  ".join(
            f"{['low','mid','high'][r]}={np.mean(v):.3f}"
            for r, v in val_per_regime.items() if v
        )
        print(f"Ep {epoch:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"[{reg_str}]  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.1f}s")

        ckpt = {
            "epoch": epoch, "val_loss": val_loss,
            "val_acc": val_acc,
            "model": critic.state_dict(),
            "reward_stats": reward_stats,
            "g_stats":      g_stats,
            "action_stats": action_stats,
            "z_stats":      z_stats,                       # v3.1
            "cfg": {
                "d_state":     d_state,
                "d_z":         d_z,
                "d_action":    d_action,
                "hidden":      args.hidden,
                "n_layers":    args.n_layers,
                "gamma":       args.gamma,
                "inv_max":     args.inv_max,
                "mode":        "rank",
                "version":     version_str,
                "use_actions": args.use_actions,
                "L_levels":    args.L_levels,
                "z_norm":      args.z_norm,                # v3.1
                "use_spectral": not args.no_spectral,
                "margin":      args.margin,
            },
        }
        # Best by pairwise accuracy (the metric we care about)
        if val_acc > best_acc:
            best_acc = val_acc
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "critic_best.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"critic_ep{epoch:03d}.pt")

    critic.eval()
    s_val  = full_ds.s[torch.randperm(len(full_ds.s))[:5000]].to(device)
    L_emp  = critic.estimate_lipschitz(s_val)
    print(f"\nBest val_acc: {best_acc:.3f}  val_loss: {best_val:.4f}  "
          f"Lipschitz: {L_emp:.2f}")

    best_ckpt = torch.load(ckpt_dir / "critic_best.pt", weights_only=False)
    best_ckpt["lipschitz_estimate"] = L_emp
    torch.save(best_ckpt, ckpt_dir / "critic_best.pt")
    print(f"Checkpoint: {ckpt_dir / 'critic_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Critic (TD / MC / RANK) — v2.3")
    parser.add_argument("--dataset",     type=str,   default="data/wm_dataset.npz")
    parser.add_argument("--mode",        type=str,   default="mc",
                        choices=["td", "mc", "rank"])
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.95)
    parser.add_argument("--tau",         type=float, default=0.05,
                        help="EMA rate for TD target network (TD mode only)")
    parser.add_argument("--gp_weight",   type=float, default=0.0,
                        help="Gradient penalty weight (TD mode only in v2)")
    parser.add_argument("--hidden",      type=int,   default=128,
                        help="Hidden dim per branch (v2 has two branches)")
    parser.add_argument("--n_layers",    type=int,   default=3)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_frac",    type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--inv_max",     type=float, default=25.0,
                        help="Normalizzazione inventory: clip(inv/inv_max, -1, 1).")
    parser.add_argument("--huber_delta", type=float, default=5.0,
                        help="Huber loss delta (v2.2 default: 5.0, era 1.0 in v2.1). "
                             "Con delta alto la zona quadratica copre i target estremi, "
                             "così il critico può apprendere le code invece di censurarle.")
    parser.add_argument("--no_spectral", action="store_true",
                        help="Disattiva spectral_norm (per ablation)")
    parser.add_argument("--winsor_low",  type=float, default=0.0,
                        help="Percentile low clip per G (v2.2 default: 0.0 = no winsor). "
                             "v2.1 usava 1.0 ma dopo diagnostica abbiamo visto che "
                             "tagliare le code rimuove il segnale più informativo.")
    parser.add_argument("--winsor_high", type=float, default=100.0,
                        help="Percentile high clip per G (v2.2 default: 100.0 = no winsor)")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Margin for ranking loss (rank mode): pairs with "
                             "|G_a - G_b| < margin are ignored as ties. "
                             "Default 0 = no tie filtering.")
    parser.add_argument("--use_actions", action="store_true",
                        help="v3: augment state with A-S quotes "
                             "[k_bid, k_ask, q_bid, q_ask] normalized. "
                             "Helps critic capture inv × σ interactions directly.")
    parser.add_argument("--L_levels", type=int, default=10,
                        help="LOB levels for k normalization (used with --use_actions). "
                             "Default 10 (simulator default).")
    parser.add_argument("--z_norm", action="store_true",
                        help="v3.1: z-score normalization per-dim del latente z "
                             "usando stats del train split. Allinea le scale con le "
                             "altre feature dello stato (inv, tl, actions).")
    parser.add_argument("--ckpt_dir",    type=str,   default="checkpoints")
    args = parser.parse_args()

    if args.mode == "td":
        train_td(args)
    elif args.mode == "rank":
        train_rank(args)
    else:
        train_mc(args)
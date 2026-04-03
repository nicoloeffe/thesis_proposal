"""
train_robust_critic.py — Backward Robust Dynamic Programming.

Instead of fitted value iteration (which suffers from bootstrap instability),
this uses backward induction on finite-horizon episodes:

    t = T: V_rob(z, T) = 0
    t = T-1: V_rob(z, T-1) = inf_Q E_Q[ r_{T-1} + γ V_rob(z', T) ]
    t = T-2: V_rob(z, T-2) = inf_Q E_Q[ r_{T-2} + γ V_rob(z', T-1) ]
    ...
    t = 0: V_rob(z, 0) = inf_Q E_Q[ r_0 + γ V_rob(z', 1) ]

Key property: when computing V_rob(·, t), the function V_rob(·, t+1) is
ALREADY COMPUTED AND FROZEN. There is no bootstrap, no circularity, no
fixed-point iteration. Each backward step is a single supervised regression
on fixed targets — exactly like the nominal critic training that worked
perfectly.

The DRO one-step module (which already works) is used at each step.

The critic is time-conditioned: V(z, t) = w_t^T · φ(z) + b_t
where φ is the frozen feature extractor from the nominal critic and
each timestep t has its own last-layer weights (w_t, b_t).

Uso:
  python training/train_robust_critic.py \
      --n_transitions 100000 --epsilon 0.05
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.critic import ValueNetwork
from models.world_model import LOBWorldModel, WorldModelConfig
from models.dro import DROConfig


# ---------------------------------------------------------------------------
# Frozen-φ critic with swappable last layer
# ---------------------------------------------------------------------------

class FrozenPhiCritic(nn.Module):
    """
    Wraps a ValueNetwork with frozen hidden layers.
    The last Linear layer weights can be swapped per-timestep.
    """

    def __init__(self, critic: ValueNetwork):
        super().__init__()
        self.critic = critic

        # Identify and freeze all layers except the last Linear
        self.last_layer_name = None
        for name, module in critic.named_modules():
            if isinstance(module, nn.Linear):
                self.last_layer_name = name

        # Freeze everything except last layer
        for name, param in critic.named_parameters():
            if not name.startswith(self.last_layer_name):
                param.requires_grad_(False)

        # Get reference to last layer
        parts = self.last_layer_name.split(".")
        obj = critic
        for p in parts:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        self.last_layer = obj

        # Count params
        self.n_frozen = sum(
            p.numel() for n, p in critic.named_parameters()
            if not n.startswith(self.last_layer_name)
        )
        self.n_trainable = sum(
            p.numel() for n, p in critic.named_parameters()
            if n.startswith(self.last_layer_name)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_last_layer_state(self) -> dict[str, torch.Tensor]:
        return {
            "weight": self.last_layer.weight.data.clone(),
            "bias": self.last_layer.bias.data.clone(),
        }

    def set_last_layer_state(self, state: dict[str, torch.Tensor]) -> None:
        self.last_layer.weight.data.copy_(state["weight"])
        self.last_layer.bias.data.copy_(state["bias"])

    def reset_last_layer(self) -> None:
        """Re-initialize last layer for fresh fitting."""
        nn.init.xavier_uniform_(self.last_layer.weight)
        nn.init.zeros_(self.last_layer.bias)

    def trainable_params(self) -> list[nn.Parameter]:
        return [self.last_layer.weight, self.last_layer.bias]


# ---------------------------------------------------------------------------
# Batched inner solver
# ---------------------------------------------------------------------------

def batched_inner_solve(
    critic: nn.Module,
    y: torch.Tensor,
    lam: torch.Tensor,
    inner_steps: int = 50,
    base_lr: float = 0.05,
    trust_radius: float = 0.2,
) -> torch.Tensor:
    R = trust_radius
    y_det = y.detach()
    lr = torch.clamp(0.5 / (2.0 * lam + 1.0), max=base_lr).unsqueeze(-1)
    x = y.clone().detach().requires_grad_(True)

    for step in range(inner_steps):
        v = critic(x)
        penalty = lam * ((x - y_det) ** 2).sum(dim=-1)
        obj = (v + penalty).sum()
        grad = torch.autograd.grad(obj, x, create_graph=False)[0]
        x_new = x - lr * grad
        x_new = torch.clamp(x_new, y_det - R, y_det + R)
        x = x_new.detach().requires_grad_(True)

    return x.detach()


# ---------------------------------------------------------------------------
# Batched robust targets (single critic — no twin needed!)
# ---------------------------------------------------------------------------

def batched_robust_targets(
    critic: nn.Module,
    pi_t: torch.Tensor, # (M, K)
    mu_t: torch.Tensor, # (M, K, D)
    r_t: torch.Tensor, # (M,)
    epsilon: float,
    cfg: DROConfig,
    device: torch.device,
    chunk_size: int = 10000,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute robust Bellman targets for all sequences at a single timestep.
    The critic contains V_{t+1} — already frozen from previous backward step.
    """
    M, K, D = mu_t.shape
    gamma = cfg.gamma
    targets = torch.zeros(M)

    t0 = time.time()
    n_chunks = (M + chunk_size - 1) // chunk_size

    for c in range(n_chunks):
        i_start = c * chunk_size
        i_end = min((c + 1) * chunk_size, M)
        n_chunk = i_end - i_start

        pi_c = pi_t[i_start:i_end].to(device)
        mu_c = mu_t[i_start:i_end].to(device)
        r_c = r_t[i_start:i_end]
        y_flat = mu_c.reshape(n_chunk * K, D)

        # Bisection
        lam_low = torch.full((n_chunk,), 1e-4, device=device)
        lam_high = torch.full((n_chunk,), cfg.lambda_init, device=device)

        for bisect_step in range(cfg.outer_steps):
            lam_mid = (lam_low + lam_high) / 2
            lam_flat = lam_mid.repeat_interleave(K)

            x_star = batched_inner_solve(
                critic, y_flat, lam_flat,
                inner_steps=cfg.inner_steps,
                base_lr=cfg.inner_lr,
                trust_radius=cfg.trust_radius,
            )

            x_reshaped = x_star.reshape(n_chunk, K, D)
            transport_k = ((x_reshaped - mu_c) ** 2).sum(dim=-1)
            w_transport = (pi_c * transport_k).sum(dim=-1)

            too_high = w_transport > epsilon
            lam_low = torch.where(too_high, lam_mid, lam_low)
            lam_high = torch.where(too_high, lam_high, lam_mid)

            if ((lam_high - lam_low) / (lam_high + 1e-8) < 0.01).all():
                break

        # Final solve
        lam_final = ((lam_low + lam_high) / 2).repeat_interleave(K)
        x_star = batched_inner_solve(
            critic, y_flat, lam_final,
            inner_steps=cfg.inner_steps,
            base_lr=cfg.inner_lr,
            trust_radius=cfg.trust_radius,
        )

        with torch.no_grad():
            v_star = critic(x_star)

        v_reshaped = v_star.reshape(n_chunk, K)
        v_robust = (pi_c * v_reshaped).sum(dim=-1)
        targets[i_start:i_end] = r_c + gamma * v_robust.cpu()

        if verbose and (c + 1) % 5 == 0:
            elapsed = time.time() - t0
            mean_t = w_transport.mean().item()
            mean_lam = ((lam_low + lam_high) / 2).mean().item()
            print(f" chunk {c+1}/{n_chunks} "
                  f"transport={mean_t:.4f} λ*={mean_lam:.3f} "
                  f"t={elapsed:.1f}s", end="\r")

    if verbose:
        elapsed = time.time() - t0
        print(f" Done: {M} seqs in {elapsed:.1f}s "
              f"mean={targets.mean():.4f} std={targets.std():.4f} ")

    return targets


# ---------------------------------------------------------------------------
# Pre-compute GMMs (keep timestep structure!)
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_gmm_by_timestep(
    world_model: LOBWorldModel,
    sequences: torch.Tensor, # (M, N+1, D)
    actions: torch.Tensor, # (M, N, 3)
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns GMM params indexed by timestep:
        pi_all: (M, N, K)
        mu_all: (M, N, K, D)
    """
    M, Np1, D = sequences.shape
    N = Np1 - 1

    pi_list, mu_list = [], []

    for i in range(0, M, batch_size):
        z_seq = sequences[i:i+batch_size].to(device)
        a_seq = actions[i:i+batch_size].to(device)
        pi, mu, _ = world_model(z_seq, a_seq)
        pi_list.append(pi.cpu()) # (B, N, K)
        mu_list.append(mu.cpu()) # (B, N, K, D)

    pi_all = torch.cat(pi_list, dim=0) # (M, N, K)
    mu_all = torch.cat(mu_list, dim=0) # (M, N, K, D)

    return pi_all, mu_all


# ---------------------------------------------------------------------------
# Fit last layer on targets
# ---------------------------------------------------------------------------

def fit_last_layer(
    critic: FrozenPhiCritic,
    z: torch.Tensor,
    targets: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
) -> float:
    """
    Fit only the last layer (w, b) to regress targets.
    Split is by episode (masks precomputed), not random.
    Returns best validation loss.
    """
    train_ds = TensorDataset(z[train_mask], targets[train_mask])
    val_ds = TensorDataset(z[val_mask], targets[val_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.Adam(critic.trainable_params(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # IMPORTANT: keep eval() mode! The hidden layers are frozen and have
        # Dropout — train() would make φ(z) stochastic, corrupting the fit.
        # Gradients still flow to the last layer via loss.backward().
        critic.critic.eval()
        tr_loss = 0.0
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            pred = critic(zb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        critic.critic.eval()
        vl = 0.0
        with torch.no_grad():
            for zb, yb in val_loader:
                zb, yb = zb.to(device), yb.to(device)
                vl += loss_fn(critic(zb), yb).item()
        vl /= len(val_loader)

        if vl < best_val:
            best_val = vl

    return best_val


# ---------------------------------------------------------------------------
# Main: Backward Robust DP
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- World model ---
    wm_ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    wm_cfg = WorldModelConfig()
    for k, v in wm_ckpt["cfg"].items():
        setattr(wm_cfg, k, v)
    world_model = LOBWorldModel(wm_cfg).to(device)
    world_model.load_state_dict(wm_ckpt["model"])
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)
    print(f"World model loaded (epoch={wm_ckpt['epoch']})")

    # --- Load nominal critic → frozen-φ critic ---
    cr_ckpt = torch.load(args.critic_ckpt, map_location=device, weights_only=False)
    cr_cfg = cr_ckpt["cfg"]

    base_critic = ValueNetwork(
        d_latent=cr_cfg["d_latent"],
        hidden=cr_cfg["hidden"],
        n_layers=cr_cfg["n_layers"],
    ).to(device)
    base_critic.load_state_dict(cr_ckpt["model"])

    critic = FrozenPhiCritic(base_critic)
    print(f"Critic: {critic.n_frozen:,} frozen (φ) + "
          f"{critic.n_trainable:,} trainable (last layer)")

    # --- Load dataset (keep sequence structure!) ---
    data = np.load(args.dataset)
    sequences = torch.from_numpy(data["sequences"]) # (M, N+1, D)
    actions = torch.from_numpy(data["actions"]) # (M, N, 3)
    rewards = torch.from_numpy(data["rewards"]) # (M, N)
    episode_ids = data["episode_ids"] # (M,)

    M, Np1, D = sequences.shape
    N = Np1 - 1
    print(f"Dataset: {M:,} sequences, N={N} steps, d_latent={D}")

    # Subsample sequences (not transitions!)
    if args.n_sequences < M:
        rng = np.random.default_rng(42)
        seq_idx = rng.choice(M, size=args.n_sequences, replace=False)
        sequences = sequences[seq_idx]
        actions = actions[seq_idx]
        rewards = rewards[seq_idx]
        episode_ids = episode_ids[seq_idx]
        M = args.n_sequences
    print(f"Using {M:,} sequences")

    # --- Episode-based train/val split (no leakage!) ---
    unique_eps = np.unique(episode_ids)
    rng_split = np.random.default_rng(123)
    rng_split.shuffle(unique_eps)
    n_val_eps = max(1, int(len(unique_eps) * 0.1))
    val_eps = set(unique_eps[-n_val_eps:])
    is_val = np.array([eid in val_eps for eid in episode_ids], dtype = bool)
    train_mask = torch.from_numpy(~is_val) # (M,) bool
    val_mask = torch.from_numpy(is_val) # (M,) bool
    print(f"Split by episode: {len(unique_eps)} episodes, "
          f"{train_mask.sum().item():,} train, {val_mask.sum().item():,} val")

    # --- Pre-compute GMMs by timestep ---
    print("\nPre-computing GMMs (keeping timestep structure)...")
    pi_all, mu_all = precompute_gmm_by_timestep(
        world_model, sequences, actions, device
    )
    print(f"GMMs: pi {pi_all.shape}, mu {mu_all.shape}")

    # --- DRO config ---
    dro_cfg = DROConfig(
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_steps=args.outer_steps,
        lambda_init=args.lambda_init,
        trust_radius=args.trust_radius,
        gamma=args.gamma,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    print(f"\n{'='*70}")
    print(f"BACKWARD ROBUST DYNAMIC PROGRAMMING")
    print(f" ε = {args.epsilon} γ = {args.gamma}")
    print(f" Horizon: N = {N} steps (backward from t={N-1} to t=0)")
    print(f" Inner: {dro_cfg.inner_steps} steps Trust R: {dro_cfg.trust_radius}")
    print(f" Representation: FROZEN | Last layer: fitted per timestep")
    print(f"{'='*70}")

    # Storage for per-timestep last-layer weights
    layer_weights = {} # t → {"weight": ..., "bias": ...}

    # Also store per-timestep diagnostics
    diagnostics = []

    t_total = time.time()

    # =====================================================================
    # BACKWARD SWEEP: from t = N-1 down to t = 0
    # =====================================================================

    for t in range(N - 1, -1, -1):
        print(f"\n{'─'*50}")
        print(f" Backward step t={t} ({N-1-t+1}/{N})")
        print(f"{'─'*50}")

        # States and rewards at timestep t
        z_t = sequences[:, t, :] # (M, D)
        r_t = rewards[:, t] # (M,)

        if t == N - 1:
            # ---- TERMINAL STEP ----
            # V_rob(z, N) = 0 (no future value)
            # Target: y = r_{N-1} + γ · 0 = r_{N-1}
            # But we still apply DRO on the "next step value = 0"
            # Which means: inf_Q E_Q[r + 0] = r (no perturbation changes a constant)
            # So targets are just the immediate rewards
            targets = r_t.clone()
            print(f" Terminal: targets = r_{t} "
                  f"mean={targets.mean():.4f} std={targets.std():.4f}")

        else:
            # ---- BACKWARD STEP ----
            # Load V_{t+1} into critic (from previous backward step)
            critic.set_last_layer_state(layer_weights[t + 1])
            critic.critic.eval()
            # Note: no need to enable requires_grad on parameters.
            # autograd.grad(obj, x) computes dobj/dx through the forward
            # pass regardless of param.requires_grad — only x needs grad.

            # GMM params at timestep t
            pi_t = pi_all[:, t, :] # (M, K)
            mu_t = mu_all[:, t, :, :] # (M, K, D)

            # Compute robust targets using DRO with V_{t+1}
            print(f" Computing robust targets with V_rob(·, {t+1})...")
            targets = batched_robust_targets(
                critic, pi_t, mu_t, r_t,
                epsilon=args.epsilon,
                cfg=dro_cfg,
                device=device,
                chunk_size=args.chunk_size,
            )

        # ---- FIT V_rob(·, t) ----
        # Reset last layer and fit on targets
        critic.reset_last_layer()
        print(f" Fitting V_rob(·, {t})...")
        val_loss = fit_last_layer(
            critic, z_t, targets, train_mask, val_mask, device,
            epochs=args.fit_epochs,
            batch_size=args.fit_batch_size,
            lr=args.fit_lr,
        )

        # Save this timestep's weights
        layer_weights[t] = critic.get_last_layer_state()

        # Compute residual: how well does V_rob(z_t, t) match targets?
        critic.critic.eval()
        with torch.no_grad():
            v_pred = critic(z_t.to(device)).cpu()
        residual = (v_pred - targets).abs().mean().item()

        diag = {
            "t": t,
            "target_mean": targets.mean().item(),
            "target_std": targets.std().item(),
            "val_loss": val_loss,
            "residual": residual,
        }
        diagnostics.append(diag)

        print(f" → t={t} target_mean={diag['target_mean']:.4f} "
              f"val={val_loss:.6f} residual={residual:.6f}")

    elapsed_total = time.time() - t_total

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"BACKWARD ROBUST DP COMPLETE")
    print(f"Total time: {elapsed_total:.1f}s")
    print(f"{'─'*50}")
    print(f"{'t':>3s} {'target_mean':>12s} {'residual':>10s} {'val_loss':>10s}")
    print(f"{'─'*50}")
    for d in sorted(diagnostics, key=lambda x: x["t"]):
        print(f"{d['t']:3d} {d['target_mean']:12.4f} "
              f"{d['residual']:10.6f} {d['val_loss']:10.6f}")

    # Save all timestep weights + diagnostics
    ckpt = {
        "layer_weights": layer_weights, # t → {weight, bias}
        "diagnostics": diagnostics,
        "cfg": cr_cfg,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "N": N,
        "critic_base": base_critic.state_dict(), # frozen φ
    }
    torch.save(ckpt, ckpt_dir / "robust_critic_backward.pt")
    print(f"\nSaved: {ckpt_dir / 'robust_critic_backward.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backward Robust DP — Frozen φ")
    parser.add_argument("--critic_ckpt",   type=str,   default="checkpoints/critic_best.pt")
    parser.add_argument("--wm_ckpt",       type=str,   default="checkpoints/wm_best.pt")
    parser.add_argument("--dataset",       type=str,   default="data/wm_dataset.npz")
    parser.add_argument("--epsilon",       type=float, default=0.05)
    parser.add_argument("--n_sequences",   type=int,   default=50000)
    parser.add_argument("--inner_steps",   type=int,   default=50)
    parser.add_argument("--inner_lr",      type=float, default=0.05)
    parser.add_argument("--outer_steps",   type=int,   default=20)
    parser.add_argument("--lambda_init",   type=float, default=50.0)
    parser.add_argument("--trust_radius",  type=float, default=0.2)
    parser.add_argument("--gamma",         type=float, default=0.95)
    parser.add_argument("--chunk_size",    type=int,   default=10000)
    parser.add_argument("--fit_epochs",    type=int,   default=10)
    parser.add_argument("--fit_batch_size",type=int,   default=2048)
    parser.add_argument("--fit_lr",        type=float, default=1e-3)
    parser.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    args = parser.parse_args()
    main(args)
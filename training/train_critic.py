"""
train_critic.py — Fase 4: Training offline del critico V_θ(z).

Approccio: Monte Carlo returns (NO bootstrap).

Il dataset contiene sequenze di 20 step con reward:
    r_0, r_1, ..., r_19

Per ogni posizione t, calcoliamo il return cumulato DAI DATI:
    G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{N-1-t}*r_{N-1}

Poi il training è una regressione supervisionata:
    L = MSE( V_θ(z_t), G_t )

Nessuna target network, nessun soft update, nessuna circolarità.
G_t è un numero fisso calcolato una volta sola prima del training.

Uso:
    python training/train_critic.py
    python training/train_critic.py --dataset data/wm_dataset.npz \
                                     --epochs 50 --gamma 0.95
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.critic import ValueNetwork


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CriticDataset(Dataset):
    """
    Pre-computes Monte Carlo returns from sequence rewards.

    Input: wm_dataset.npz con sequences (M, N+1, D) e rewards (M, N)

    Per ogni sequenza e ogni posizione t:
        G_t = Σ_{k=0}^{N-1-t} γ^k * r_{t+k}

    Output campioni: (z_t, G_t, regime)
    """

    def __init__(self, path: str, gamma: float = 0.95) -> None:
        data = np.load(path)
        sequences = data["sequences"].astype(np.float32)  # (M, N+1, D)
        rewards   = data["rewards"].astype(np.float32)    # (M, N)
        regimes   = data["regimes"].astype(np.int64)      # (M,)

        M, Np1, D = sequences.shape
        N = Np1 - 1

        # --- Pre-compute Monte Carlo returns ---
        # G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        # Computed backwards for efficiency:
        #   G_{N-1} = r_{N-1}
        #   G_t     = r_t + γ * G_{t+1}
        returns = np.zeros_like(rewards)  # (M, N)
        returns[:, -1] = rewards[:, -1]
        for t in range(N - 2, -1, -1):
            returns[:, t] = rewards[:, t] + gamma * returns[:, t + 1]

        # Print return statistics
        print(f"Monte Carlo returns (γ={gamma}, horizon={N}):")
        print(f"  G mean: {returns.mean():.4f}  std: {returns.std():.4f}")
        print(f"  G range: [{returns.min():.4f}, {returns.max():.4f}]")

        # Flatten: every (z_t, G_t) pair becomes a training sample
        z_all = sequences[:, :N, :].reshape(-1, D)   # (M*N, D)
        g_all = returns.reshape(-1)                    # (M*N,)
        reg   = np.repeat(regimes, N)                  # (M*N,)

        self.z = torch.from_numpy(z_all)
        self.g = torch.from_numpy(g_all)
        self.reg = torch.from_numpy(reg)

        print(f"CriticDataset: {len(self.z):,} samples  d_latent={D}")
        for r, name in enumerate(["low_vol", "mid_vol", "high_vol"]):
            mask = (self.reg == r)
            cnt  = mask.sum().item()
            g_r  = self.g[mask]
            print(f"  regime {r} ({name:10s}): {cnt:,}  "
                  f"G_mean={g_r.mean():.4f}  G_std={g_r.std():.4f}")

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, idx: int):
        return self.z[idx], self.g[idx], self.reg[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Gamma  : {args.gamma}")

    # --- Dataset ---
    full_ds = CriticDataset(args.dataset, gamma=args.gamma)
    val_size   = int(len(full_ds) * args.val_frac)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {train_size:,}  Val: {val_size:,}  "
          f"Steps/epoch: {len(train_loader)}")

    # --- Model (no target network needed) ---
    d_latent = full_ds.z.shape[-1]
    critic = ValueNetwork(d_latent=d_latent, hidden=args.hidden,
                          n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"Critic params: {n_params:,}")

    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    loss_fn = nn.MSELoss()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        critic.train()
        t0 = time.time()
        tr_loss = 0.0

        for z, g, reg in train_loader:
            z = z.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)

            v_pred = critic(z)          # (B,)
            loss   = loss_fn(v_pred, g)  # MSE(V(z), G) — that's it

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()

            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # ---- Validate ----
        critic.eval()
        val_loss = 0.0
        val_bias = 0.0
        val_per_regime = {0: [], 1: [], 2: []}

        with torch.no_grad():
            for z, g, reg in val_loader:
                z   = z.to(device, non_blocking=True)
                g   = g.to(device, non_blocking=True)
                reg = reg.to(device, non_blocking=True)

                v_pred = critic(z)
                val_loss += loss_fn(v_pred, g).item()
                val_bias += (v_pred - g).mean().item()

                # Per-regime MSE
                for r in [0, 1, 2]:
                    mask = (reg == r)
                    if mask.sum() > 0:
                        mse_r = loss_fn(v_pred[mask], g[mask]).item()
                        val_per_regime[r].append(mse_r)

        n_val = len(val_loader)
        val_loss /= n_val
        val_bias /= n_val

        reg_str = "  ".join(
            f"{['low', 'mid', 'high'][r]}={np.mean(v):.6f}"
            for r, v in val_per_regime.items() if v
        )

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={tr_loss:.6f}  val={val_loss:.6f}  "
              f"bias={val_bias:+.4f}  "
              f"[{reg_str}]  "
              f"lr={lr_now:.2e}  t={elapsed:.1f}s")

        ckpt = {
            "epoch":    epoch,
            "val_loss": val_loss,
            "model":    critic.state_dict(),
            "cfg": {
                "d_latent": d_latent,
                "hidden":   args.hidden,
                "n_layers": args.n_layers,
                "gamma":    args.gamma,
            },
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "critic_best.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"critic_ep{epoch:03d}.pt")

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Critico salvato in {ckpt_dir / 'critic_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Critic (Monte Carlo returns)")
    parser.add_argument("--dataset",     type=str,   default="data/wm_dataset.npz")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.95)
    parser.add_argument("--hidden",      type=int,   default=256)
    parser.add_argument("--n_layers",    type=int,   default=3)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--val_frac",    type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--ckpt_dir",    type=str,   default="checkpoints")
    args = parser.parse_args()
    train(args)
"""
train_critic.py — Training offline del critico V_θ(z).

Due modalità:
  --mode td   : TD(0) con target network EMA  (default)
  --mode mc   : Monte Carlo returns

Gradient penalty (--gp_weight): penalizza ‖∇_z V‖² per mantenere il
Lipschitz bound soft. Il bound empirico viene stimato a fine training
e salvato nel checkpoint per il DRO.

Reward normalization: running mean/std calcolati prima del training,
salvati nel checkpoint.

Uso:
    python training/train_critic.py --mode mc --gp_weight 0.1
    python training/train_critic.py --mode td --gp_weight 0.1
"""

from __future__ import annotations

import argparse
import copy
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
# Datasets
# ---------------------------------------------------------------------------

class CriticSequenceDataset(Dataset):
    """Sequence-level dataset for TD learning."""

    def __init__(self, path: str) -> None:
        data = np.load(path)
        self.sequences = torch.from_numpy(data["sequences"].astype(np.float32))
        self.rewards   = torch.from_numpy(data["rewards"].astype(np.float32))
        self.regimes   = torch.from_numpy(data["regimes"].astype(np.int64))
        M, Np1, D = self.sequences.shape
        print(f"CriticSequenceDataset: {M:,} seq, N={Np1-1}, D={D}")

    def __len__(self): return len(self.sequences)
    def __getitem__(self, i): return self.sequences[i], self.rewards[i], self.regimes[i]


class CriticFlatDataset(Dataset):
    """Flat (z_t, G_t) pairs for MC learning."""

    def __init__(self, path: str, gamma: float, reward_stats: dict) -> None:
        data = np.load(path)
        seqs = data["sequences"].astype(np.float32)
        rews = data["rewards"].astype(np.float32)
        regs = data["regimes"].astype(np.int64)
        M, Np1, D = seqs.shape; N = Np1 - 1

        r_norm = (rews - reward_stats["mean"]) / (reward_stats["std"] + 1e-8)
        returns = np.zeros_like(r_norm)
        returns[:, -1] = r_norm[:, -1]
        for t in range(N - 2, -1, -1):
            returns[:, t] = r_norm[:, t] + gamma * returns[:, t + 1]

        self.z   = torch.from_numpy(seqs[:, :N].reshape(-1, D))
        self.g   = torch.from_numpy(returns.reshape(-1))
        self.reg = torch.from_numpy(np.repeat(regs, N))
        print(f"CriticFlatDataset: {len(self.z):,} samples  "
              f"G mean={self.g.mean():.4f} std={self.g.std():.4f}")

    def __len__(self): return len(self.z)
    def __getitem__(self, i): return self.z[i], self.g[i], self.reg[i]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_reward_stats(path: str) -> dict:
    data = np.load(path)
    r = data["rewards"].astype(np.float32).reshape(-1)
    stats = {"mean": float(r.mean()), "std": float(r.std())}
    print(f"Reward stats: μ={stats['mean']:.6f}  σ={stats['std']:.6f}")
    return stats


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(1 - tau).add_(p_o.data, alpha=tau)


# ---------------------------------------------------------------------------
# MC Training
# ---------------------------------------------------------------------------

def train_mc(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nMode: MC  gp_weight={args.gp_weight}")

    reward_stats = compute_reward_stats(args.dataset)
    full_ds = CriticFlatDataset(args.dataset, args.gamma, reward_stats)

    val_size = int(len(full_ds) * args.val_frac)
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    d_latent = full_ds.z.shape[-1]
    critic = ValueNetwork(d_latent=d_latent, hidden=args.hidden,
                          n_layers=args.n_layers).to(device)
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        critic.train(); t0 = time.time(); tr_loss = 0.0; tr_gp = 0.0

        for z, g, reg in train_loader:
            z = z.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)

            v_pred = critic(z)
            mse = nn.functional.mse_loss(v_pred, g)

            loss = mse
            if args.gp_weight > 0:
                gp = critic.gradient_penalty(z)
                loss = loss + args.gp_weight * gp
                tr_gp += gp.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            tr_loss += mse.item()

        n_batches = len(train_loader)
        tr_loss /= n_batches
        tr_gp /= n_batches

        # Validate
        critic.eval(); val_loss = 0.0; val_per_regime = {0: [], 1: [], 2: []}
        with torch.no_grad():
            for z, g, reg in val_loader:
                z   = z.to(device, non_blocking=True)
                g   = g.to(device, non_blocking=True)
                reg = reg.to(device, non_blocking=True)
                v = critic(z)
                val_loss += nn.functional.mse_loss(v, g).item()
                for r in [0, 1, 2]:
                    m = (reg == r)
                    if m.sum() > 0:
                        val_per_regime[r].append(nn.functional.mse_loss(v[m], g[m]).item())

        val_loss /= len(val_loader)
        scheduler.step()

        reg_str = "  ".join(f"{['low','mid','high'][r]}={np.mean(v):.4f}"
                            for r, v in val_per_regime.items() if v)
        gp_str = f"  gp={tr_gp:.4f}" if args.gp_weight > 0 else ""
        print(f"Ep {epoch:3d}/{args.epochs}  train={tr_loss:.4f}{gp_str}  "
              f"val={val_loss:.4f}  [{reg_str}]  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.1f}s")

        ckpt = {
            "epoch": epoch, "val_loss": val_loss,
            "model": critic.state_dict(),
            "reward_stats": reward_stats,
            "cfg": {"d_latent": d_latent, "hidden": args.hidden,
                    "n_layers": args.n_layers, "gamma": args.gamma, "mode": "mc"},
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "critic_best.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"critic_ep{epoch:03d}.pt")

    # Estimate Lipschitz constant on validation data
    critic.eval()
    z_val = full_ds.z[torch.randperm(len(full_ds.z))[:5000]].to(device)
    L_emp = critic.estimate_lipschitz(z_val)
    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Empirical Lipschitz: {L_emp:.2f}")

    # Re-save best with Lipschitz estimate
    best_ckpt = torch.load(ckpt_dir / "critic_best.pt", weights_only=False)
    best_ckpt["lipschitz_estimate"] = L_emp
    torch.save(best_ckpt, ckpt_dir / "critic_best.pt")
    print(f"Checkpoint: {ckpt_dir / 'critic_best.pt'}")


# ---------------------------------------------------------------------------
# TD Training
# ---------------------------------------------------------------------------

def train_td(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nMode: TD(0)  τ={args.tau}  gp_weight={args.gp_weight}")

    reward_stats = compute_reward_stats(args.dataset)
    full_ds = CriticSequenceDataset(args.dataset)

    val_size = int(len(full_ds) * args.val_frac)
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    d_latent = full_ds.sequences.shape[-1]
    critic = ValueNetwork(d_latent=d_latent, hidden=args.hidden,
                          n_layers=args.n_layers).to(device)
    target = copy.deepcopy(critic)
    for p in target.parameters(): p.requires_grad_(False)
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    r_mean = torch.tensor(reward_stats["mean"], device=device)
    r_std  = torch.tensor(reward_stats["std"] + 1e-8, device=device)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        critic.train(); t0 = time.time(); tr_loss = 0.0; tr_gp = 0.0

        for z_seq, r_seq, reg in train_loader:
            z_seq = z_seq.to(device, non_blocking=True)
            r_seq = r_seq.to(device, non_blocking=True)
            B, Np1, D = z_seq.shape; N = Np1 - 1

            r_norm = (r_seq - r_mean) / r_std
            z_t = z_seq[:, :N].reshape(B * N, D)
            v_pred = critic(z_t).reshape(B, N)

            with torch.no_grad():
                z_next = z_seq[:, 1:].reshape(B * N, D)
                v_next = target(z_next).reshape(B, N)
                td_target = r_norm + args.gamma * v_next

            mse = nn.functional.mse_loss(v_pred, td_target)
            loss = mse
            if args.gp_weight > 0:
                gp = critic.gradient_penalty(z_t)
                loss = loss + args.gp_weight * gp
                tr_gp += gp.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            ema_update(target, critic, args.tau)
            tr_loss += mse.item()

        n_batches = len(train_loader)
        tr_loss /= n_batches
        tr_gp /= n_batches

        # Validate
        critic.eval(); val_loss = 0.0; val_per_regime = {0: [], 1: [], 2: []}
        with torch.no_grad():
            for z_seq, r_seq, reg in val_loader:
                z_seq = z_seq.to(device, non_blocking=True)
                r_seq = r_seq.to(device, non_blocking=True)
                reg   = reg.to(device, non_blocking=True)
                B, Np1, D = z_seq.shape; N = Np1 - 1

                r_norm = (r_seq - r_mean) / r_std
                z_t = z_seq[:, :N].reshape(B * N, D)
                v_pred = critic(z_t).reshape(B, N)
                z_next = z_seq[:, 1:].reshape(B * N, D)
                v_next = target(z_next).reshape(B, N)
                td_target = r_norm + args.gamma * v_next

                val_loss += nn.functional.mse_loss(v_pred, td_target).item()
                for r in [0, 1, 2]:
                    m = (reg == r)
                    if m.sum() > 0:
                        val_per_regime[r].append(
                            nn.functional.mse_loss(v_pred[m], td_target[m]).item())

        val_loss /= len(val_loader)
        scheduler.step()

        reg_str = "  ".join(f"{['low','mid','high'][r]}={np.mean(v):.4f}"
                            for r, v in val_per_regime.items() if v)
        gp_str = f"  gp={tr_gp:.4f}" if args.gp_weight > 0 else ""
        print(f"Ep {epoch:3d}/{args.epochs}  train={tr_loss:.4f}{gp_str}  "
              f"val={val_loss:.4f}  [{reg_str}]  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.1f}s")

        ckpt = {
            "epoch": epoch, "val_loss": val_loss,
            "model": critic.state_dict(),
            "target_model": target.state_dict(),
            "reward_stats": reward_stats,
            "cfg": {"d_latent": d_latent, "hidden": args.hidden,
                    "n_layers": args.n_layers, "gamma": args.gamma, "mode": "td"},
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, ckpt_dir / "critic_best.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"critic_ep{epoch:03d}.pt")

    # Estimate Lipschitz
    critic.eval()
    z_est = full_ds.sequences[:500, 0, :].to(device)
    L_emp = critic.estimate_lipschitz(z_est)
    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Empirical Lipschitz: {L_emp:.2f}")

    best_ckpt = torch.load(ckpt_dir / "critic_best.pt", weights_only=False)
    best_ckpt["lipschitz_estimate"] = L_emp
    torch.save(best_ckpt, ckpt_dir / "critic_best.pt")
    print(f"Checkpoint: {ckpt_dir / 'critic_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Critic (TD or MC)")
    parser.add_argument("--dataset",     type=str,   default="data/wm_dataset.npz")
    parser.add_argument("--mode",        type=str,   default="mc", choices=["td", "mc"])
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.95)
    parser.add_argument("--tau",         type=float, default=0.05)
    parser.add_argument("--gp_weight",   type=float, default=0.1)
    parser.add_argument("--hidden",      type=int,   default=256)
    parser.add_argument("--n_layers",    type=int,   default=3)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--val_frac",    type=float, default=0.1)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--ckpt_dir",    type=str,   default="checkpoints")
    args = parser.parse_args()

    if args.mode == "td":
        train_td(args)
    else:
        train_mc(args)
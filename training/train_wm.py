"""
train_wm.py — Training del World Model (Modulo B).

Pipeline:
  1. Carica wm_dataset.npz (sequenze latenti pre-calcolate dall'encoder)
  2. Allena il Causal Transformer + GMM con NLL loss
  3. Logga NLL totale, entropia π (mode collapse), NLL per regime
  4. Salva checkpoint con model, config, stats

Uso:
  python training/train_wm.py
  python training/train_wm.py --dataset data/wm_dataset.npz \\
                               --epochs 50 --seq_len 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.world_model import LOBWorldModel, WorldModelConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WMDataset(Dataset):
    """
    Dataset di sequenze latenti per il world model.

    Ogni campione è:
      sequences   : (N+1, d_latent)  — z_0..z_N
      actions     : (N,   d_action)  — a_0..a_{N-1}
      rewards     : (N,)             — r_0..r_{N-1}
      regimes     : (N,) or scalar   — per-step or per-episode regime (0/1/2)
      episode_id  : scalar           — episodio di origine
    """

    def __init__(self, path: str, indices: np.ndarray | None = None) -> None:
        data = np.load(path)
        sequences  = torch.from_numpy(data["sequences"])
        actions    = torch.from_numpy(data["actions"])
        rewards    = torch.from_numpy(data["rewards"])
        regimes    = torch.from_numpy(data["regimes"].astype(np.int64))
        ep_ids     = torch.from_numpy(data["episode_ids"].astype(np.int64))

        if indices is not None:
            sequences = sequences[indices]
            actions   = actions[indices]
            rewards   = rewards[indices]
            regimes   = regimes[indices]
            ep_ids    = ep_ids[indices]

        self.sequences   = sequences
        self.actions     = actions
        self.rewards     = rewards
        self.regimes     = regimes
        self.episode_ids = ep_ids

        M, Np1, D = self.sequences.shape
        self.N = Np1 - 1
        self.D = D

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            self.sequences[idx],
            self.actions[idx],
            self.rewards[idx],
            self.regimes[idx],
        )


def episode_split(
    dataset_path: str,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split sequences by episode — all segments of an episode go to the same split.
    Returns (train_indices, val_indices).
    """
    data    = np.load(dataset_path)
    ep_ids  = data["episode_ids"]
    unique_eps = np.unique(ep_ids)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)

    n_val_eps = max(1, int(len(unique_eps) * val_frac))
    val_eps   = set(unique_eps[:n_val_eps])
    train_eps = set(unique_eps[n_val_eps:])

    train_idx = np.where([ep not in val_eps   for ep in ep_ids])[0]
    val_idx   = np.where([ep in val_eps        for ep in ep_ids])[0]

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}")

    # --- Dataset — split by episode to avoid sequence overlap leak ---
    print(f"Building episode-based train/val split (val_frac={args.val_frac})...")
    train_idx, val_idx = episode_split(args.dataset, val_frac=args.val_frac)
    train_ds = WMDataset(args.dataset, indices=train_idx)
    val_ds   = WMDataset(args.dataset, indices=val_idx)
    print(f"  train: {len(train_idx):,}  val: {len(val_idx):,}")

    # Infer d_latent from data
    d_latent = train_ds.D
    regimes_per_step = (train_ds.regimes[0].dim() == 1)
    print(f"  d_latent={d_latent}  seq_len={train_ds.N}  "
          f"regimes_per_step={regimes_per_step}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Batch: {args.batch_size}  Steps/epoch: {len(train_loader)}")

    # --- Model ---
    cfg = WorldModelConfig()
    cfg.d_latent = d_latent
    cfg.n_gmm = args.n_gmm
    cfg.lambda_regime = args.lambda_regime
    model = LOBWorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}  lambda_regime={cfg.lambda_regime}")

    # --- Optimiser + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        t0 = time.time()
        tr_nll = 0.0
        tr_reg_acc = 0.0

        for z_seq, a_seq, rewards, regimes in train_loader:
            z_seq   = z_seq.to(device, non_blocking=True)
            a_seq   = a_seq.to(device, non_blocking=True)
            regimes = regimes.to(device, non_blocking=True)

            # Expand per-episode regimes to per-step if needed
            if regimes.dim() == 1:
                regimes = regimes.unsqueeze(1).expand(-1, a_seq.shape[1])

            optimizer.zero_grad()
            pi, mu, log_sig = model(z_seq, a_seq)
            z_next = z_seq[:, 1:, :]

            loss_nll = model.nll_loss(pi, mu, log_sig, z_next)
            loss_reg, reg_acc = model.regime_loss(regimes)
            loss = loss_nll + cfg.lambda_regime * loss_reg

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            tr_nll     += loss_nll.item()
            tr_reg_acc += reg_acc

        n_tr = len(train_loader)
        tr_nll /= n_tr
        tr_reg_acc /= n_tr

        # ---- Validate ----
        model.eval()
        val_nll = 0.0
        val_ent = 0.0
        val_reg_acc = 0.0
        nll_per_regime = {0: 0.0, 1: 0.0, 2: 0.0}
        cnt_per_regime = {0: 0,   1: 0,   2: 0}

        with torch.no_grad():
            for z_seq, a_seq, rewards, regimes in val_loader:
                z_seq   = z_seq.to(device, non_blocking=True)
                a_seq   = a_seq.to(device, non_blocking=True)
                regimes = regimes.to(device, non_blocking=True)

                if regimes.dim() == 1:
                    regimes = regimes.unsqueeze(1).expand(-1, a_seq.shape[1])

                pi, mu, log_sig = model(z_seq, a_seq)
                z_next = z_seq[:, 1:, :]

                val_nll += model.nll_loss(pi, mu, log_sig, z_next).item()

                # Entropy of π
                ent = -(pi * torch.log(pi + 1e-8)).sum(dim=-1).mean().item()
                val_ent += ent

                # Regime head accuracy
                logits = model.regime_head(model._last_hidden)
                val_reg_acc += (logits.argmax(dim=-1) == regimes).float().mean().item()

                # NLL per regime (flatten to per-step)
                reg_flat = regimes.reshape(-1)
                B, N, K, D = mu.shape
                for reg_id in [0, 1, 2]:
                    mask = (reg_flat == reg_id)
                    if mask.sum() == 0:
                        continue
                    pi_f = pi.reshape(B*N, K)[mask].unsqueeze(1)
                    mu_f = mu.reshape(B*N, K, D)[mask].unsqueeze(1)
                    ls_f = log_sig.reshape(B*N, K, D)[mask].unsqueeze(1)
                    zn_f = z_next.reshape(B*N, D)[mask].unsqueeze(1)
                    nll_per_regime[reg_id] += model.nll_loss(pi_f, mu_f, ls_f, zn_f).item()
                    cnt_per_regime[reg_id] += 1

        n_val = len(val_loader)
        val_nll /= n_val
        val_ent /= n_val
        val_reg_acc /= n_val
        for r in [0, 1, 2]:
            if cnt_per_regime[r] > 0:
                nll_per_regime[r] /= cnt_per_regime[r]

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"nll={tr_nll:.4f}  reg_acc={tr_reg_acc:.3f}  "
            f"| val_nll={val_nll:.4f}  ent_π={val_ent:.3f}  "
            f"reg_acc={val_reg_acc:.3f}  "
            f"[low={nll_per_regime[0]:.3f}  "
            f"mid={nll_per_regime[1]:.3f}  "
            f"high={nll_per_regime[2]:.3f}]  "
            f"lr={lr_now:.2e}  t={elapsed:.1f}s"
        )

        ckpt = {
            "epoch":    epoch,
            "val_nll":  val_nll,
            "model":    model.state_dict(),
            "cfg":      cfg.__dict__,
        }
        if val_nll < best_val:
            best_val = val_nll
            epochs_no_improve = 0
            torch.save(ckpt, ckpt_dir / "wm_best.pt")
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"wm_ep{epoch:03d}.pt")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    print(f"\nBest val NLL: {best_val:.4f}")
    print(f"Saved to {ckpt_dir / 'wm_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LOB World Model")
    parser.add_argument("--dataset",       type=str,   default="data/wm_dataset.npz")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--val_frac",      type=float, default=0.1)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--n_gmm",         type=int,   default=5)
    parser.add_argument("--lambda_regime", type=float, default=0.1,
                        help="Weight for regime classification auxiliary loss")
    parser.add_argument("--patience",      type=int,   default=15,
                        help="Early stopping patience")
    parser.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    args = parser.parse_args()
    train(args)
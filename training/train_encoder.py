"""
train_encoder.py — Training del LOB Autoencoder (Fase 1, Modulo A) (v5).

Pipeline:
  1. Carica dataset.npz
  2. Normalizza book: prezzi relativi al mid (in tick), volumi / vol_scale
  3. Training loop con Adam + cosine LR schedule + gradient clipping
  4. Salva encoder congelato in checkpoints/encoder.pt

Note (v5):
  L'encoder apprende una STATIC REPRESENTATION del book. La dinamica è
  compito esclusivo del Modulo B (world model).

  Loss: L = L_recon + λ_stats * L_stats + λ_contr * L_contr

  Rimossi VICReg (var + decorr): empiricamente neutri sui probe downstream,
  introducevano whitening dannoso per la struttura dei regimi.
  Sostituiti dall'on-manifold contractive loss che fornisce bi-Lipschitz
  empirica sulla data manifold (condizione per stabilità del Modulo C).

  Early stopping: val_total.

Uso:
  python training/train_encoder.py
  python training/train_encoder.py --dataset data/dataset.npz --epochs 50 \\
                                    --lambda_contr 0.1 --contr_tau_percentile 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encoder import LOBAutoEncoder, EncoderConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LOBDataset(Dataset):
    """
    Loads observations from dataset.npz and exposes books only.

    Normalisation:
      - book prices  : subtract mid, divide by tick_size → relative tick offsets
      - book volumes : divide by vol_scale (99th percentile of training set)
      - mid/inv stats are computed and saved for downstream use, but NOT
        passed to the encoder model.

    v4: rimosso load_next e book_next — l'encoder è statico.
    """

    def __init__(
        self,
        path: str,
        L: int = 10,
        tick_size: float = 0.01,
        stats: dict | None = None,
    ) -> None:
        data = np.load(path)
        obs = data["observations"].astype(np.float32)

        book_flat_dim = 2 * L * 2
        self.L = L
        self.tick_size = tick_size

        book_flat   = obs[:, :book_flat_dim]
        scalars_raw = obs[:, book_flat_dim:]
        book        = book_flat.reshape(-1, 2, L, 2)

        # Compute normalisation stats (on full data or passed in)
        if stats is None:
            vols = book[:, :, :, 1].reshape(-1)
            vol_scale = float(np.percentile(vols[vols > 0], 99)) if (vols > 0).any() else 1.0
            mid_mean  = float(scalars_raw[:, 0].mean())
            mid_std   = float(scalars_raw[:, 0].std()) + 1e-8
            inv_scale = float(np.percentile(np.abs(scalars_raw[:, 3]), 99)) + 1e-8
            self.stats = {
                "vol_scale": vol_scale,
                "mid_mean":  mid_mean,
                "mid_std":   mid_std,
                "inv_scale": inv_scale,
            }
        else:
            self.stats = stats

        s = self.stats

        # Normalize book using mid from scalars (internal only)
        mids = scalars_raw[:, 0]
        bk_n = book.copy()
        for side in range(2):
            bk_n[:, side, :, 0] = (book[:, side, :, 0] - mids[:, None]) / tick_size
        bk_n[:, :, :, 1] /= s["vol_scale"]

        self.book = torch.from_numpy(bk_n)

    def __len__(self) -> int:
        return len(self.book)

    def __getitem__(self, idx: int):
        return (self.book[idx],)


# ---------------------------------------------------------------------------
# Stats and splitting
# ---------------------------------------------------------------------------

def compute_stats_from_indices(
    path: str,
    indices: list[int] | np.ndarray,
    L: int = 10,
    tick_size: float = 0.01,
) -> dict:
    """
    Compute normalization stats from a SUBSET of the dataset (train only).
    Prevents val data from influencing normalization — avoids data leakage.
    """
    data = np.load(path)
    obs = data["observations"][indices].astype(np.float32)
    book_flat_dim = 2 * L * 2
    book = obs[:, :book_flat_dim].reshape(-1, 2, L, 2)
    scalars_raw = obs[:, book_flat_dim:]

    vols = book[:, :, :, 1].reshape(-1)
    vol_scale = float(np.percentile(vols[vols > 0], 99)) if (vols > 0).any() else 1.0
    mid_mean  = float(scalars_raw[:, 0].mean())
    mid_std   = float(scalars_raw[:, 0].std()) + 1e-8
    inv_scale = float(np.percentile(np.abs(scalars_raw[:, 3]), 99)) + 1e-8

    return {
        "vol_scale": vol_scale,
        "mid_mean":  mid_mean,
        "mid_std":   mid_std,
        "inv_scale": inv_scale,
    }


def episode_split_indices(
    path: str,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int] | None, list[int] | None]:
    """
    Split by episode: all transitions from one episode go to the same split.
    Prevents temporal leakage — even if the encoder is static, splitting
    by timestep would put nearly-identical consecutive snapshots in both
    train and val, inflating val performance artificially.
    """
    data = np.load(path)
    if "episode_ids" not in data:
        return None, None

    ep_ids = data["episode_ids"]
    unique_eps = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_eps)

    n_val_eps = max(1, int(len(unique_eps) * val_frac))
    val_eps = set(unique_eps[:n_val_eps])

    train_idx = [i for i, e in enumerate(ep_ids) if e not in val_eps]
    val_idx   = [i for i, e in enumerate(ep_ids) if e in val_eps]
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}")

    # --- Episode-based split ---
    train_idx, val_idx = episode_split_indices(
        args.dataset, val_frac=args.val_frac
    )

    if train_idx is None:
        print("WARNING: no episode_ids in dataset, using random split")
        full_ds = LOBDataset(args.dataset)
        stats = full_ds.stats
        val_size = int(len(full_ds) * args.val_frac)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        stats = compute_stats_from_indices(args.dataset, train_idx)
        print(f"Stats computed on {len(train_idx):,} TRAIN transitions only")

        full_ds = LOBDataset(args.dataset, stats=stats)

        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
        print(f"Episode split: {len(train_idx):,} train, {len(val_idx):,} val")

    print(f"Norm stats    : vol_scale={stats['vol_scale']:.2f}  "
          f"mid_std={stats['mid_std']:.4f}  inv_scale={stats['inv_scale']:.2f}")
    print(f"Lambda stats={args.lambda_stats}  contr={args.lambda_contr}  "
          f"tau_pct={args.contr_tau_percentile}")

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
    cfg = EncoderConfig()
    cfg.d_latent = args.d_latent
    model = LOBAutoEncoder(cfg).to(device)
    print(f"Model params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"d_latent      : {cfg.d_latent}")

    # --- Optimiser + scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
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
        tr_total = tr_recon = tr_stats = tr_contr = 0.0

        for (book,) in train_loader:
            book = book.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, _, losses = model(
                book,
                lambda_stats=args.lambda_stats,
                lambda_contr=args.lambda_contr,
                contr_tau_percentile=args.contr_tau_percentile,
            )
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            tr_total += losses["total"].item()
            tr_recon += losses["recon"].item()
            tr_stats += losses["stats"].item()
            tr_contr += losses["contr"].item()

        n = len(train_loader)
        tr_total /= n; tr_recon /= n; tr_stats /= n; tr_contr /= n

        # ---- Validate ----
        model.eval()
        val_total = val_recon = val_stats = val_contr = 0.0
        with torch.no_grad():
            for (book,) in val_loader:
                book = book.to(device, non_blocking=True)
                _, _, losses = model(
                    book,
                    lambda_stats=args.lambda_stats,
                    lambda_contr=args.lambda_contr,
                    contr_tau_percentile=args.contr_tau_percentile,
                )
                val_total += losses["total"].item()
                val_recon += losses["recon"].item()
                val_stats += losses["stats"].item()
                val_contr += losses["contr"].item()
        val_n = len(val_loader)
        val_total /= val_n; val_recon /= val_n; val_stats /= val_n; val_contr /= val_n

        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"recon={tr_recon:.4f}  stats={tr_stats:.4f}  "
              f"contr={tr_contr:.4f}  "
              f"| vrecon={val_recon:.4f}  vstats={val_stats:.4f}  "
              f"vcontr={val_contr:.4f}  vtotal={val_total:.4f}  "
              f"lr={lr_now:.2e}  t={elapsed:.1f}s")

        ckpt = {
            "epoch":                epoch,
            "val_loss":             val_total,
            "val_recon":            val_recon,
            "encoder":              model.encoder.state_dict(),
            "decoder":              model.decoder.state_dict(),
            "stats_head":           model.stats_head.state_dict(),
            "cfg":                  cfg.__dict__,
            "stats":                stats,
            "lambda_stats":         args.lambda_stats,
            "lambda_contr":         args.lambda_contr,
            "contr_tau_percentile": args.contr_tau_percentile,
        }

        # Early stopping on val_total — now comparable across epochs
        # (no warmup/ramp to change the scale).
        if val_total < best_val:
            best_val = val_total
            epochs_no_improve = 0
            torch.save(ckpt, ckpt_dir / "encoder_best.pt")
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"encoder_ep{epoch:03d}.pt")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    print(f"\nBest val_total: {best_val:.6f}")
    print(f"Encoder saved to {ckpt_dir / 'encoder_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LOB Autoencoder (v5)")
    parser.add_argument("--dataset",       type=str,   default="data/dataset.npz")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--val_frac",      type=float, default=0.1)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--d_latent",      type=int,   default=16,
                        help="Latent dimension (default: 16)")
    parser.add_argument("--lambda_stats",  type=float, default=3.0,
                        help="Weight for aggregate book stats loss")
    parser.add_argument("--lambda_contr",  type=float, default=0.1,
                        help="Weight for on-manifold contractive loss (v5)")
    parser.add_argument("--contr_tau_percentile", type=float, default=10.0,
                        help="Percentile cutoff for 'near pairs' in contractive loss "
                             "(e.g. 10 = prende il 10%% di coppie più vicine in input)")
    parser.add_argument("--patience",      type=int,   default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    args = parser.parse_args()
    train(args)
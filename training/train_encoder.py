"""
train_encoder.py — Training del LOB Autoencoder (Fase 1, Modulo A).

Pipeline:
  1. Carica dataset.npz
  2. Normalizza book (volumi) e scalari (mid, spread, imbalance, inventory)
  3. Training loop con Adam + cosine LR schedule + gradient clipping
  4. Salva encoder congelato in checkpoints/encoder.pt

Uso:
  python training/train_encoder.py
  python training/train_encoder.py --dataset data/dataset.npz --epochs 50
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
from models.encoder import LOBAutoEncoder, EncoderConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LOBDataset(Dataset):
    """
    Loads observations from dataset.npz and exposes (book, scalars) pairs.

    Normalisation:
      - book prices  : subtract mid, divide by (L * tick_size) → relative offsets
      - book volumes : divide by vol_scale (99th percentile of training set)
      - mid          : subtract mean, divide by std
      - spread       : divide by tick_size
      - imbalance    : already in [-1, 1], keep as-is
      - inventory    : divide by inv_scale (99th percentile)
    """

    def __init__(
        self,
        path: str,
        L: int = 10,
        tick_size: float = 0.01,
        stats: dict | None = None,
        load_next: bool = False,
    ) -> None:
        data = np.load(path)
        obs      = data["observations"].astype(np.float32)       # (N, 44)
        obs_next = data["next_observations"].astype(np.float32)  # (N, 44)

        book_flat_dim = 2 * L * 2  # 40
        self.L = L
        self.tick_size = tick_size
        self.load_next = load_next

        def split(arr):
            book_flat  = arr[:, :book_flat_dim]
            sc_raw     = arr[:, book_flat_dim:]
            return book_flat.reshape(-1, 2, L, 2), sc_raw

        book,      scalars_raw      = split(obs)
        book_next, scalars_raw_next = split(obs_next)

        # Compute normalisation stats
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

        def normalize(bk, sc_raw):
            mids = sc_raw[:, 0]
            bk_n = bk.copy()
            for side in range(2):
                bk_n[:, side, :, 0] = (bk[:, side, :, 0] - mids[:, None]) / tick_size
            bk_n[:, :, :, 1] /= s["vol_scale"]
            sc_n = np.stack([
                (sc_raw[:, 0] - s["mid_mean"]) / s["mid_std"],
                sc_raw[:, 1] / tick_size,
                sc_raw[:, 2],
                sc_raw[:, 3] / s["inv_scale"],
            ], axis=1)
            return torch.from_numpy(bk_n), torch.from_numpy(sc_n)

        self.book,      self.scalars      = normalize(book,      scalars_raw)
        self.book_next, self.scalars_next = normalize(book_next, scalars_raw_next)

    def __len__(self) -> int:
        return len(self.book)

    def __getitem__(self, idx: int):
        if self.load_next:
            return (self.book[idx], self.scalars[idx],
                    self.book_next[idx], self.scalars_next[idx])
        return self.book[idx], self.scalars[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}")

    # --- Dataset & DataLoader ---
    full_ds = LOBDataset(args.dataset, load_next=True)
    stats   = full_ds.stats
    print(f"Dataset size  : {len(full_ds):,} transitions")
    print(f"Norm stats    : vol_scale={stats['vol_scale']:.2f}  "
          f"mid_std={stats['mid_std']:.4f}  inv_scale={stats['inv_scale']:.2f}")
    print(f"Lambda temp   : {args.lambda_temp}   Lambda decorr: {args.lambda_decorr}")

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
          f"Batch: {args.batch_size}  Steps/epoch: {len(train_loader)}")

    # --- Model ---
    cfg   = EncoderConfig()
    model = LOBAutoEncoder(cfg).to(device)
    print(f"Model params  : {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimiser + scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        t0 = time.time()
        tr_total = tr_recon = tr_temp = tr_decorr = 0.0

        for book, scalars, book_next, sc_next in train_loader:
            book      = book.to(device, non_blocking=True)
            scalars   = scalars.to(device, non_blocking=True)
            book_next = book_next.to(device, non_blocking=True)
            sc_next   = sc_next.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, _, losses = model(
                book, scalars, book_next, sc_next,
                lambda_temp=args.lambda_temp,
                lambda_decorr=args.lambda_decorr,
            )
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            tr_total  += losses["total"].item()
            tr_recon  += losses["recon"].item()
            tr_temp   += losses["temporal"].item()
            tr_decorr += losses["decorr"].item()

        n = len(train_loader)
        tr_total /= n; tr_recon /= n; tr_temp /= n; tr_decorr /= n

        # ---- Validate ----
        model.eval()
        val_total = val_recon = val_temp = val_decorr = 0.0
        with torch.no_grad():
            for book, scalars, book_next, sc_next in val_loader:
                book      = book.to(device, non_blocking=True)
                scalars   = scalars.to(device, non_blocking=True)
                book_next = book_next.to(device, non_blocking=True)
                sc_next   = sc_next.to(device, non_blocking=True)
                _, _, losses = model(
                    book, scalars, book_next, sc_next,
                    lambda_temp=args.lambda_temp,
                    lambda_decorr=args.lambda_decorr,
                )
                val_total  += losses["total"].item()
                val_recon  += losses["recon"].item()
                val_temp   += losses["temporal"].item()
                val_decorr += losses["decorr"].item()
        val_total  /= len(val_loader)
        val_recon  /= len(val_loader)
        val_temp   /= len(val_loader)
        val_decorr /= len(val_loader)

        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"total={tr_total:.4f}  recon={tr_recon:.4f}  "
              f"temp={tr_temp:.4f}  decorr={tr_decorr:.4f}  "
              f"| val={val_total:.4f}  vrecon={val_recon:.4f}  "
              f"vtemp={val_temp:.4f}  vdecorr={val_decorr:.4f}  "
              f"lr={lr_now:.2e}  t={elapsed:.1f}s")

        ckpt = {
            "epoch":         epoch,
            "val_loss":      val_total,
            "encoder":       model.encoder.state_dict(),
            "decoder":       model.decoder.state_dict(),
            "temporal":      model.temporal.state_dict(),
            "cfg":           cfg.__dict__,
            "stats":         stats,
            "lambda_temp":   args.lambda_temp,
            "lambda_decorr": args.lambda_decorr,
        }
        if val_total < best_val:
            best_val = val_total
            torch.save(ckpt, ckpt_dir / "encoder_best.pt")
        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"encoder_ep{epoch:03d}.pt")

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Encoder saved to {ckpt_dir / 'encoder_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LOB Autoencoder (Proposal C)")
    parser.add_argument("--dataset",       type=str,   default="data/dataset.npz")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--val_frac",      type=float, default=0.1)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--lambda_temp",   type=float, default=0.3)
    parser.add_argument("--lambda_decorr", type=float, default=0.05)
    parser.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    args = parser.parse_args()
    train(args)
"""
train_encoder.py — Training del LOB Autoencoder (Fase 1, Modulo A) (v2).

Pipeline:
  1. Carica dataset.npz
  2. Normalizza book: prezzi relativi al mid (in tick), volumi / vol_scale
  3. Training loop con Adam + cosine LR schedule + gradient clipping
  4. Salva encoder congelato in checkpoints/encoder.pt

Note (v2):
  L'encoder riceve SOLO il book, niente scalari. Mid è usato internamente
  per normalizzare i prezzi del book, ma non viene passato al modello.
  Le stats di normalizzazione (vol_scale, mid_mean, mid_std, inv_scale)
  vengono salvate nel checkpoint per uso downstream (critico, world model).

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
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encoder import LOBAutoEncoder, EncoderConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LOBDataset(Dataset):
    """
    Loads observations from dataset.npz and exposes (book, book_next) pairs.

    Normalisation:
      - book prices  : subtract mid, divide by tick_size → relative tick offsets
      - book volumes : divide by vol_scale (99th percentile of training set)
      - mid/inv stats are computed and saved for downstream use, but NOT
        passed to the encoder model.
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
        obs      = data["observations"].astype(np.float32)
        obs_next = data["next_observations"].astype(np.float32)

        book_flat_dim = 2 * L * 2
        self.L = L
        self.tick_size = tick_size
        self.load_next = load_next

        def split(arr):
            book_flat = arr[:, :book_flat_dim]
            sc_raw    = arr[:, book_flat_dim:]
            return book_flat.reshape(-1, 2, L, 2), sc_raw

        book,      scalars_raw      = split(obs)
        book_next, scalars_raw_next = split(obs_next)

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

        def normalize_book(bk, sc_raw):
            """Normalize book using mid from scalars (internal only)."""
            mids = sc_raw[:, 0]
            bk_n = bk.copy()
            for side in range(2):
                bk_n[:, side, :, 0] = (bk[:, side, :, 0] - mids[:, None]) / tick_size
            bk_n[:, :, :, 1] /= s["vol_scale"]
            return torch.from_numpy(bk_n)

        self.book      = normalize_book(book,      scalars_raw)
        self.book_next = normalize_book(book_next, scalars_raw_next)

    def __len__(self) -> int:
        return len(self.book)

    def __getitem__(self, idx: int):
        if self.load_next:
            return self.book[idx], self.book_next[idx]
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
    Prevents temporal leakage in the L_temporal auxiliary loss.
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
        full_ds = LOBDataset(args.dataset, load_next=True)
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

        full_ds = LOBDataset(args.dataset, stats=stats, load_next=True)

        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
        print(f"Episode split: {len(train_idx):,} train, {len(val_idx):,} val")

    print(f"Norm stats    : vol_scale={stats['vol_scale']:.2f}  "
          f"mid_std={stats['mid_std']:.4f}  inv_scale={stats['inv_scale']:.2f}")
    print(f"Lambda temp={args.lambda_temp}  decorr={args.lambda_decorr}  "
          f"var={args.lambda_var}")

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

    # --- Temporal warmup schedule ---
    # Without scalars the encoder needs time to learn reconstruction
    # before the temporal loss can provide useful gradients.
    # Phase 1 (epoch 1..warmup):   lambda_temp = 0    (recon + VICReg only)
    # Phase 2 (warmup..warmup+ramp): linear ramp 0 → target
    # Phase 3 (warmup+ramp..end):   lambda_temp = target
    warmup = args.temp_warmup
    ramp   = args.temp_ramp
    print(f"Temporal warmup: {warmup} epochs off + {ramp} epochs ramp "
          f"→ lambda_temp={args.lambda_temp}")
    print(f"Lambda stats={args.lambda_stats}")

    for epoch in range(1, args.epochs + 1):
        # Compute effective lambda_temp with warmup schedule
        if epoch <= warmup:
            eff_lambda_temp = 0.0
        elif epoch <= warmup + ramp:
            eff_lambda_temp = args.lambda_temp * (epoch - warmup) / ramp
        else:
            eff_lambda_temp = args.lambda_temp

        # ---- Train ----
        model.train()
        t0 = time.time()
        tr_total = tr_recon = tr_stats = tr_temp = tr_decorr = tr_var = 0.0

        for book, book_next in train_loader:
            book      = book.to(device, non_blocking=True)
            book_next = book_next.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, _, losses = model(
                book, book_next,
                lambda_temp=eff_lambda_temp,
                lambda_stats=args.lambda_stats,
                lambda_decorr=args.lambda_decorr,
                lambda_var=args.lambda_var,
            )
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            tr_total  += losses["total"].item()
            tr_recon  += losses["recon"].item()
            tr_stats  += losses["stats"].item()
            tr_temp   += losses["temporal"].item()
            tr_decorr += losses["decorr"].item()
            tr_var    += losses["var"].item()

        n = len(train_loader)
        tr_total /= n; tr_recon /= n; tr_stats /= n
        tr_temp /= n; tr_decorr /= n; tr_var /= n

        # ---- Validate ----
        model.eval()
        val_total = val_recon = val_stats = val_temp = val_decorr = val_var = 0.0
        with torch.no_grad():
            for book, book_next in val_loader:
                book      = book.to(device, non_blocking=True)
                book_next = book_next.to(device, non_blocking=True)
                _, _, losses = model(
                    book, book_next,
                    lambda_temp=eff_lambda_temp,
                    lambda_stats=args.lambda_stats,
                    lambda_decorr=args.lambda_decorr,
                    lambda_var=args.lambda_var,
                )
                val_total  += losses["total"].item()
                val_recon  += losses["recon"].item()
                val_stats  += losses["stats"].item()
                val_temp   += losses["temporal"].item()
                val_decorr += losses["decorr"].item()
                val_var    += losses["var"].item()
        val_n = len(val_loader)
        val_total /= val_n; val_recon /= val_n; val_stats /= val_n
        val_temp /= val_n; val_decorr /= val_n; val_var /= val_n

        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        phase = ("warmup" if epoch <= warmup
                 else f"ramp({eff_lambda_temp:.3f})" if epoch <= warmup + ramp
                 else "full")
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"recon={tr_recon:.4f}  stats={tr_stats:.4f}  "
              f"temp={tr_temp:.4f}  decorr={tr_decorr:.4f}  var={tr_var:.4f}  "
              f"| vrecon={val_recon:.4f}  vstats={val_stats:.4f}  "
              f"vtemp={val_temp:.4f}  "
              f"lr={lr_now:.2e}  [{phase}]  t={elapsed:.1f}s")

        ckpt = {
            "epoch":         epoch,
            "val_loss":      val_total,
            "val_recon":     val_recon,
            "encoder":       model.encoder.state_dict(),
            "decoder":       model.decoder.state_dict(),
            "temporal":      model.temporal.state_dict(),
            "stats_head":    model.stats_head.state_dict(),
            "cfg":           cfg.__dict__,
            "stats":         stats,
            "lambda_temp":   args.lambda_temp,
            "lambda_stats":  args.lambda_stats,
            "lambda_decorr": args.lambda_decorr,
            "lambda_var":    args.lambda_var,
        }

        # Track val_recon for early stopping — NOT val_total.
        # val_total changes scale when lambda_temp ramps, making
        # cross-epoch comparison meaningless. val_recon is the
        # true metric of encoder quality.
        if epoch > warmup:
            if val_recon < best_val:
                best_val = val_recon
                epochs_no_improve = 0
                torch.save(ckpt, ckpt_dir / "encoder_best.pt")
            else:
                epochs_no_improve += 1
        else:
            if epoch == warmup:
                torch.save(ckpt, ckpt_dir / "encoder_warmup_end.pt")

        if epoch % 10 == 0:
            torch.save(ckpt, ckpt_dir / f"encoder_ep{epoch:03d}.pt")

        # Early stopping (only after warmup)
        if epoch > warmup and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    print(f"\nBest val_recon: {best_val:.6f}")
    print(f"Encoder saved to {ckpt_dir / 'encoder_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LOB Autoencoder")
    parser.add_argument("--dataset",       type=str,   default="data/dataset.npz")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--val_frac",      type=float, default=0.1)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--d_latent",      type=int,   default=24,
                        help="Latent dimension (default: 24)")
    parser.add_argument("--lambda_temp",   type=float, default=0.3)
    parser.add_argument("--lambda_stats",  type=float, default=1.0,
                        help="Weight for aggregate book stats loss")
    parser.add_argument("--lambda_decorr", type=float, default=0.05)
    parser.add_argument("--lambda_var",    type=float, default=0.1,
                        help="VICReg variance loss weight")
    parser.add_argument("--temp_warmup",   type=int,   default=5,
                        help="Epochs with lambda_temp=0 (let recon converge first)")
    parser.add_argument("--temp_ramp",     type=int,   default=5,
                        help="Epochs to linearly ramp lambda_temp from 0 to target")
    parser.add_argument("--patience",      type=int,   default=15,
                        help="Early stopping patience (epochs, counted after warmup)")
    parser.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    args = parser.parse_args()
    train(args)
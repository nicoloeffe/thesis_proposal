"""
build_wm_dataset.py — Costruisce il dataset di sequenze latenti per il World Model.

Pipeline:
  1. Carica il dataset non shufflato (con episode_ids)
  2. Usa l'encoder congelato per produrre z_t per ogni transizione
  3. Raggruppa le transizioni in sequenze contigue per episodio
  4. Salva il dataset in formato .npz pronto per il training del world model

Struttura del dataset salvato (wm_dataset.npz):
  sequences  : (N_seq, N+1, d_latent)   — z_0..z_N per ogni sequenza
  actions    : (N_seq, N, d_action)      — a_0..a_{N-1}
  rewards    : (N_seq, N)                — r_0..r_{N-1}
  regimes    : (N_seq, N)                — per-step regime labels (0/1/2)
  episode_ids: (N_seq,)                  — episodio di origine

Uso:
  python scripts/build_wm_dataset.py
  python scripts/build_wm_dataset.py --dataset data/dataset_ordered.npz \\
                                      --ckpt checkpoints/encoder_best.pt \\
                                      --seq_len 20 --out data/wm_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))

from models.encoder import LOBEncoder, EncoderConfig
from training.train_encoder import LOBDataset


# ---------------------------------------------------------------------------
# Encode all transitions
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_all(
    encoder: LOBEncoder,
    dataset_path: str,
    stats: dict,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Encode all observations in the dataset.
    Returns Z : (N, d_latent)
    """
    ds = LOBDataset(dataset_path, stats=stats, load_next=False)
    N  = len(ds)
    # Infer d_latent from encoder's output projection
    d  = encoder.latent_proj.bias.shape[0]
    Z  = np.zeros((N, d), dtype=np.float32)

    encoder.eval()
    for i in range(0, N, batch_size):
        book    = ds.book[i:i+batch_size].to(device)
        scalars = ds.scalars[i:i+batch_size].to(device)
        z = encoder(book, scalars)
        Z[i:i+len(z)] = z.cpu().numpy()

        if (i // batch_size) % 50 == 0:
            print(f"  encoded {i:>8d}/{N}  ({100*i/N:.1f}%)", end="\r")

    print(f"  encoded {N}/{N}  (100.0%)")
    return Z


# ---------------------------------------------------------------------------
# Build sequences
# ---------------------------------------------------------------------------

def build_sequences(
    Z:           np.ndarray,   # (N, d_latent)
    actions:     np.ndarray,   # (N, 3)
    rewards:     np.ndarray,   # (N,)
    regimes:     np.ndarray,   # (N,)
    episode_ids: np.ndarray,   # (N,)
    seq_len:     int = 20,
    stride:      int = 1,
) -> dict[str, np.ndarray]:
    """
    Slice encoded transitions into overlapping sequences of length seq_len+1.
    Each sequence contains:
      - z_t, z_{t+1}, ..., z_{t+seq_len}  → shape (seq_len+1, d_latent)
      - a_t, ..., a_{t+seq_len-1}          → shape (seq_len, 3)
      - r_t, ..., r_{t+seq_len-1}          → shape (seq_len,)

    Only slices within the same episode are kept.
    """
    seqs_z  = []
    seqs_a  = []
    seqs_r  = []
    seqs_reg = []
    seqs_ep  = []

    unique_eps = np.unique(episode_ids)
    print(f"  Building sequences from {len(unique_eps)} episodes...")

    for ep_id in unique_eps:
        mask = episode_ids == ep_id
        idx  = np.where(mask)[0]

        # Ensure contiguous
        if len(idx) < seq_len + 1:
            continue

        ep_Z   = Z[idx]
        ep_A   = actions[idx]
        ep_R   = rewards[idx]
        ep_reg = regimes[idx]  # per-step regime labels

        for start in range(0, len(idx) - seq_len, stride):
            end = start + seq_len + 1  # seq_len+1 latents, seq_len actions/rewards
            seqs_z.append(ep_Z[start:end])              # (seq_len+1, d_latent)
            seqs_a.append(ep_A[start:start+seq_len])    # (seq_len, 3)
            seqs_r.append(ep_R[start:start+seq_len])    # (seq_len,)
            seqs_reg.append(ep_reg[start:start+seq_len])# (seq_len,) — per-step!
            seqs_ep.append(ep_id)

    print(f"  Total sequences: {len(seqs_z):,}")

    return {
        "sequences":   np.array(seqs_z,   dtype=np.float32),  # (N_seq, seq_len+1, d_latent)
        "actions":     np.array(seqs_a,   dtype=np.float32),  # (N_seq, seq_len, 3)
        "rewards":     np.array(seqs_r,   dtype=np.float32),  # (N_seq, seq_len)
        "regimes":     np.array(seqs_reg, dtype=np.int8),     # (N_seq, seq_len) — per-step!
        "episode_ids": np.array(seqs_ep,  dtype=np.int32),    # (N_seq,)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Encoder : {args.ckpt}")
    print(f"Seq len : {args.seq_len}  Stride: {args.stride}")

    # --- Load encoder ---
    ckpt    = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_enc = EncoderConfig()
    if "cfg" in ckpt:
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg_enc, k):
                setattr(cfg_enc, k, v)
    encoder = LOBEncoder(cfg_enc).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    stats = ckpt["stats"]
    print(f"Encoder loaded (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}, "
          f"d_latent={cfg_enc.d_latent})")

    # --- Check dataset has episode_ids ---
    raw = np.load(args.dataset)
    if "episode_ids" not in raw:
        raise ValueError(
            "Dataset does not contain episode_ids. "
            "Regenerate with: python simulator/simulate.py --no_shuffle --out data/dataset_ordered.npz"
        )

    print(f"\nEncoding {len(raw['rewards']):,} transitions...")
    Z = encode_all(encoder, args.dataset, stats, device, batch_size=args.batch_size)
    print(f"Z shape: {Z.shape}")

    print("\nBuilding sequences...")
    dataset = build_sequences(
        Z            = Z,
        actions      = raw["actions"],
        rewards      = raw["rewards"],
        regimes      = raw["regimes"],
        episode_ids  = raw["episode_ids"],
        seq_len      = args.seq_len,
        stride       = args.stride,
    )

    # --- Stats ---
    N_seq = len(dataset["sequences"])
    d_lat = dataset["sequences"].shape[-1]
    print(f"\nWorld model dataset:")
    print(f"  sequences shape : {dataset['sequences'].shape}  — (N_seq, {args.seq_len+1}, {d_lat})")
    print(f"  actions shape   : {dataset['actions'].shape}")
    print(f"  rewards shape   : {dataset['rewards'].shape}")
    counts = np.bincount(dataset["regimes"].flatten().astype(int), minlength=3)
    total_steps = dataset["regimes"].size
    for i, name in enumerate(["low_vol", "mid_vol", "high_vol"]):
        print(f"  regime {i} ({name:10s}): {counts[i]:7d} steps ({100*counts[i]/total_steps:.1f}%)")
    # Check for mixed sequences (regime changes within a sequence)
    n_mixed = sum(
        len(np.unique(dataset["regimes"][j])) > 1
        for j in range(N_seq)
    )
    print(f"  mixed sequences  : {n_mixed:,} / {N_seq:,} ({100*n_mixed/N_seq:.1f}%)")

    # --- Shuffle sequences before saving ---
    rng  = np.random.default_rng(42)
    perm = rng.permutation(N_seq)
    dataset = {k: v[perm] for k, v in dataset.items()}

    out_path = args.out
    np.savez_compressed(out_path, **dataset)
    print(f"\nSaved to {out_path}  ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build world model latent sequence dataset")
    parser.add_argument("--dataset",    type=str,  default="data/dataset_ordered.npz",
                        help="Ordered (non-shuffled) dataset with episode_ids")
    parser.add_argument("--ckpt",       type=str,  default="checkpoints/encoder_best.pt")
    parser.add_argument("--seq_len",    type=int,  default=20,
                        help="Sequence length N (world model context)")
    parser.add_argument("--stride",     type=int,  default=5,
                        help="Stride between sequences (1=max overlap, seq_len=no overlap)")
    parser.add_argument("--batch_size", type=int,  default=2048)
    parser.add_argument("--out",        type=str,  default="data/wm_dataset.npz")
    args = parser.parse_args()
    main(args)
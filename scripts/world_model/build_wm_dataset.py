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
  inventories: (N_seq, N+1)              — inventory per ogni step (incluso z_N)
  time_left  : (N_seq, N+1)             — time left normalizzato in [0,1]
  switch_mask: (N_seq, N)               — 1 se cambio di regime al passo t

Uso:
  python scripts/build_wm_dataset.py
  python scripts/build_wm_dataset.py --dataset data/dataset_ordered.npz \
                                      --ckpt checkpoints/encoder_best.pt \
                                      --seq_len 20 --out data/wm_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "simulator"))

from models.encoder import LOBEncoder, EncoderConfig
from training.train_encoder import LOBDataset


# ---------------------------------------------------------------------------
# Encode all transitions
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_all(
    encoder:      LOBEncoder,
    dataset_path: str,
    stats:        dict,
    device:       torch.device,
    d_latent:     int,
    batch_size:   int = 2048,
) -> np.ndarray:
    """
    Encode all observations in the dataset.
    Returns Z : (N, d_latent)
    """
    ds = LOBDataset(dataset_path, stats=stats)   # v5: no load_next argument
    N  = len(ds)
    Z  = np.zeros((N, d_latent), dtype=np.float32)

    encoder.eval()
    for i in range(0, N, batch_size):
        book = ds.book[i:i+batch_size].to(device)

        # L'encoder v5 prende in input SOLO il book. Niente scalari.
        z = encoder(book)

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
    inventories: np.ndarray,   # (N,)   ← nuovo
    time_left:   np.ndarray,   # (N,)   ← nuovo
    switch_mask: np.ndarray,   # (N,)   ← nuovo
    seq_len:     int = 20,
    stride:      int = 1,
) -> dict[str, np.ndarray]:
    """
    Slice encoded transitions into overlapping sequences of length seq_len+1.

    Per ogni sequenza salva anche inventories e time_left su N+1 step
    (incluso lo stato finale z_N) e switch_mask su N step.
    """
    seqs_z    = []
    seqs_a    = []
    seqs_r    = []
    seqs_reg  = []
    seqs_ep   = []
    seqs_inv  = []   # ← nuovo
    seqs_tl   = []   # ← nuovo
    seqs_sw   = []   # ← nuovo

    unique_eps = np.unique(episode_ids)
    print(f"  Building sequences from {len(unique_eps)} episodes...")

    for ep_id in unique_eps:
        mask = episode_ids == ep_id
        idx  = np.where(mask)[0]

        if len(idx) < seq_len + 1:
            continue

        ep_Z   = Z[idx]
        ep_A   = actions[idx]
        ep_R   = rewards[idx]
        ep_reg = regimes[idx]
        ep_inv = inventories[idx]   # ← nuovo
        ep_tl  = time_left[idx]     # ← nuovo
        ep_sw  = switch_mask[idx]   # ← nuovo

        for start in range(0, len(idx) - seq_len, stride):
            end = start + seq_len + 1   # seq_len+1 latents

            seqs_z.append(ep_Z[start:end])                   # (N+1, d_latent)
            seqs_a.append(ep_A[start:start+seq_len])         # (N, 3)
            seqs_r.append(ep_R[start:start+seq_len])         # (N,)
            seqs_reg.append(ep_reg[start:start+seq_len])     # (N,)
            seqs_ep.append(ep_id)
            seqs_inv.append(ep_inv[start:end])               # (N+1,) ← nuovo
            seqs_tl.append(ep_tl[start:end])                 # (N+1,) ← nuovo
            seqs_sw.append(ep_sw[start:start+seq_len])       # (N,)   ← nuovo

    print(f"  Total sequences: {len(seqs_z):,}")

    return {
        "sequences":   np.array(seqs_z,   dtype=np.float32),  # (N_seq, N+1, d_latent)
        "actions":     np.array(seqs_a,   dtype=np.float32),  # (N_seq, N, 3)
        "rewards":     np.array(seqs_r,   dtype=np.float32),  # (N_seq, N)
        "regimes":     np.array(seqs_reg, dtype=np.int8),     # (N_seq, N)
        "episode_ids": np.array(seqs_ep,  dtype=np.int32),    # (N_seq,)
        "inventories": np.array(seqs_inv, dtype=np.float32),  # (N_seq, N+1) ← nuovo
        "time_left":   np.array(seqs_tl,  dtype=np.float32),  # (N_seq, N+1) ← nuovo
        "switch_mask": np.array(seqs_sw,  dtype=np.int8),     # (N_seq, N)   ← nuovo
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

    # --- Verifica campi richiesti ---
    raw = np.load(args.dataset)
    required = ["episode_ids", "inventories", "time_left", "switch_mask"]
    missing  = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"Dataset manca di: {missing}\n"
            f"Chiavi presenti: {list(raw.keys())}\n"
            "Rigenera con: python simulator/simulate.py --no_shuffle --out data/dataset_ordered.npz"
        )

    print(f"\nEncoding {len(raw['rewards']):,} transitions...")
    Z = encode_all(encoder, args.dataset, stats, device,
                   d_latent=cfg_enc.d_latent, batch_size=args.batch_size)
    print(f"Z shape: {Z.shape}")

    print("\nBuilding sequences...")
    dataset = build_sequences(
        Z            = Z,
        actions      = raw["actions"],
        rewards      = raw["rewards"],
        regimes      = raw["regimes"],
        episode_ids  = raw["episode_ids"],
        inventories  = raw["inventories"],
        time_left    = raw["time_left"],
        switch_mask  = raw["switch_mask"],
        seq_len      = args.seq_len,
        stride       = args.stride,
    )

    # --- Normalizzazione di z: (z - mean) / std per-dim ---
    #
    # Problema motivante: l'encoder v5 con contractive loss produce latenti con
    # std ~0.02, scala a cui la GMM subisce mode collapse (una sola componente
    # domina perché NLL è minimizzata concentrando tutta la probabilità su un
    # punto, vista la densità peak ∝ 1/σ^d).
    #
    # Fix: standardizziamo z dopo l'encoding. Le stats sono calcolate SUL TRAIN
    # SPLIT (episode-based) per evitare qualsiasi leakage val→train. Lo stesso
    # seed=42 è usato in train_wm.py per garantire consistency.
    print("\nComputing z normalization stats (train split only)...")
    N_seq_total = len(dataset["sequences"])
    ep_ids_seqs = dataset["episode_ids"]
    unique_eps_seq = np.unique(ep_ids_seqs)
    rng_split = np.random.default_rng(args.split_seed)
    shuffled_eps = unique_eps_seq.copy()
    rng_split.shuffle(shuffled_eps)
    n_val_eps = max(1, int(len(shuffled_eps) * args.val_frac))
    val_eps = set(shuffled_eps[:n_val_eps])
    train_mask = np.array([ep not in val_eps for ep in ep_ids_seqs])

    # Stats da sequences del train split: (N_seq_train, seq_len+1, d_latent)
    z_train = dataset["sequences"][train_mask]   # (N_train, N+1, d)
    z_mean = z_train.reshape(-1, z_train.shape[-1]).mean(axis=0)  # (d,)
    z_std  = z_train.reshape(-1, z_train.shape[-1]).std(axis=0)   # (d,)
    z_std  = np.maximum(z_std, 1e-6)  # safety

    print(f"  train sequences: {train_mask.sum():,}  val: {(~train_mask).sum():,}")
    print(f"  z_mean   : [{z_mean.min():+.4f}, {z_mean.max():+.4f}]  "
          f"mean={z_mean.mean():+.4f}")
    print(f"  z_std    : [{z_std.min():.4f}, {z_std.max():.4f}]  "
          f"mean={z_std.mean():.4f}")

    # Applica normalizzazione a TUTTE le sequenze (train + val)
    dataset["sequences"] = (
        (dataset["sequences"] - z_mean) / z_std
    ).astype(np.float32)

    # Verifica stats post-normalization sul train
    z_after = dataset["sequences"][train_mask].reshape(-1, z_train.shape[-1])
    print(f"  post-norm mean: {z_after.mean():+.4f}  std: {z_after.std():.4f}  "
          f"(expected 0, 1)")

    # Salva stats nel dataset per uso downstream (inference)
    dataset["z_mean"] = z_mean.astype(np.float32)
    dataset["z_std"]  = z_std.astype(np.float32)
    dataset["val_frac"] = np.array([args.val_frac], dtype=np.float32)
    dataset["split_seed"] = np.array([args.split_seed], dtype=np.int32)

    # --- Stats ---
    N_seq = len(dataset["sequences"])
    d_lat = dataset["sequences"].shape[-1]
    print(f"\nWorld model dataset:")
    print(f"  sequences shape : {dataset['sequences'].shape}")
    print(f"  actions shape   : {dataset['actions'].shape}")
    print(f"  rewards shape   : {dataset['rewards'].shape}")
    print(f"  inventories shape: {dataset['inventories'].shape}")
    print(f"  time_left shape : {dataset['time_left'].shape}")
    print(f"  switch_mask shape: {dataset['switch_mask'].shape}")

    counts      = np.bincount(dataset["regimes"].flatten().astype(int), minlength=3)
    total_steps = dataset["regimes"].size
    for i, name in enumerate(["low_vol", "mid_vol", "high_vol"]):
        print(f"  regime {i} ({name:10s}): {counts[i]:7d} steps ({100*counts[i]/total_steps:.1f}%)")

    n_mixed = sum(
        len(np.unique(dataset["regimes"][j])) > 1
        for j in range(N_seq)
    )
    n_switch = dataset["switch_mask"].sum()
    print(f"  mixed sequences  : {n_mixed:,} / {N_seq:,} ({100*n_mixed/N_seq:.1f}%)")
    print(f"  switch steps     : {n_switch:,} / {dataset['switch_mask'].size:,} "
          f"({100*n_switch/dataset['switch_mask'].size:.2f}%)")

    inv_min = dataset["inventories"].min()
    inv_max = dataset["inventories"].max()
    print(f"  inventory range  : [{inv_min:.2f}, {inv_max:.2f}]  "
          f"→ suggerito --inv_max={max(abs(inv_min), abs(inv_max)):.1f}")

    # --- Shuffle sequences prima di salvare ---
    # Proteggiamo le chiavi globali (stats, metadata) che hanno shape fissa
    # indipendente da N_seq.
    GLOBAL_KEYS = {"z_mean", "z_std", "val_frac", "split_seed"}
    rng  = np.random.default_rng(42)
    perm = rng.permutation(N_seq)
    dataset = {
        k: (v if k in GLOBAL_KEYS else v[perm])
        for k, v in dataset.items()
    }

    out_path = args.out
    np.savez_compressed(out_path, **dataset)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"\nSaved to {out_path}  ({size_mb:.1f} MB)")
    print(f"Keys: {list(dataset.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build world model latent sequence dataset")
    parser.add_argument("--dataset",    type=str,   default="data/dataset_ordered.npz")
    parser.add_argument("--ckpt",       type=str,   default="checkpoints/encoder_best.pt")
    parser.add_argument("--seq_len",    type=int,   default=20)
    parser.add_argument("--stride",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=2048)
    parser.add_argument("--val_frac",   type=float, default=0.1,
                        help="Val fraction for computing z normalization stats on train split (must match train_wm.py)")
    parser.add_argument("--split_seed", type=int,   default=42,
                        help="Seed for episode split (must match train_wm.py)")
    parser.add_argument("--out",        type=str,   default="data/wm_dataset.npz")
    args = parser.parse_args()
    main(args)
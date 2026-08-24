#!/usr/bin/env python3
"""
compare_supervised_jepa.py — side-by-side R² of supervised-grid vs JEPA-grid-readout.

Both arms share:
  - the same backbone class (HorizonJEPAEncoder, 4 region tokens, non-causal)
  - the same readout class (AttnPoolReadout, no bottleneck)
  - the same target set (22 observables)
  - the same grouped stock-day split
  - the same per-stock normalization (stock_stats inherited from the JEPA ckpt)

What differs:
  - F0 (supervised-grid): encoder trained end-to-end by MSE on the 22 targets.
  - F1 (jepa-grid-readout): encoder frozen (JEPA-horizon SSL); only readout trained.

Output: per-target R² table on the val split, plus summary numbers.

Usage
-----
python -m scripts.evaluation.compare_supervised_jepa \\
  --dataset data/lobench_processed.npz \\
  --supervised_ckpt checkpoints/supervised_grid/v1/best.pt \\
  --jepa_readout_ckpt validation/jepa_grid_readout/v1/best.pt \\
  --out_dir validation/compare_supervised_jepa/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent, _THIS.parent.parent.parent,
           _THIS.parent.parent.parent.parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.historical.train_jepa_horizon import (  # type: ignore
    HorizonJEPAEncoder,
    HorizonJEPAEncoderConfig,
)
from training.train_tokenizer_t import (  # type: ignore
    compute_valid_endpoints, normalize_book_window, grouped_split_by_stock_day,
    derive_raw_features_array,
)
from training.historical.train_supervised_grid import (  # type: ignore
    AttnPoolReadout, ReadoutConfig, SupervisedGrid,
    build_targets, standardize_targets, r2_per_target, summarize_r2,
    FUTURE_HORIZONS, VOL_HORIZONS,
)


def robust_load(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_supervised(ckpt_path: str, device: torch.device) -> SupervisedGrid:
    ckpt = robust_load(ckpt_path, device)
    enc_cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    rd_cfg = ReadoutConfig(**ckpt["readout_cfg"])
    model = SupervisedGrid(enc_cfg, rd_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_jepa_readout(ckpt_path: str, device: torch.device):
    ckpt = robust_load(ckpt_path, device)
    enc_cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    enc = HorizonJEPAEncoder(enc_cfg).to(device)
    # We don't have the encoder state_dict in the readout checkpoint; the user
    # must pass --jepa_ckpt with the encoder. Done in main.
    rd_cfg = ReadoutConfig(**ckpt["readout_cfg"])
    rd = AttnPoolReadout(rd_cfg).to(device)
    rd.load_state_dict(ckpt["readout_state_dict"])
    rd.eval()
    return enc, rd, ckpt


@torch.no_grad()
def evaluate_model_on_val(model_fn, book, mid_z, stock_ids, t_val, stock_stats, K,
                          y_val_z, device, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Run a forward (book → predicted standardized targets) over the val endpoints.
    `model_fn(book, stock_ids)` returns the (B, 22) prediction."""
    n = len(t_val)
    preds = np.empty((n, y_val_z.shape[1]), dtype=np.float32)
    idx = 0
    t0 = time.time()
    Wbuf = np.empty((batch_size, K, 2, book.shape[2], 2), dtype=np.float32)
    Sbuf = np.empty(batch_size, dtype=np.int64)
    while idx < n:
        end = min(idx + batch_size, n)
        B = end - idx
        for j in range(B):
            t = int(t_val[idx + j]); s = int(stock_ids[t])
            Wbuf[j] = normalize_book_window(book[t-K+1:t+1], mid_z[t-K+1:t+1], s, stock_stats)
            Sbuf[j] = s
        W = torch.from_numpy(Wbuf[:B]).to(device)
        S = torch.from_numpy(Sbuf[:B]).to(device)
        y_pred = model_fn(W, S).cpu().numpy()
        preds[idx:end] = y_pred
        idx = end
        if (idx % (batch_size * 20)) == 0:
            print(f"    {idx}/{n}  [{time.time()-t0:.1f}s]")
    return preds, y_val_z


def main():
    p = argparse.ArgumentParser(description="Side-by-side comparison supervised vs JEPA")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--supervised_ckpt", type=str, required=True)
    p.add_argument("--jepa_readout_ckpt", type=str, required=True)
    p.add_argument("--jepa_ckpt", type=str, required=True,
                   help="Original JEPA encoder checkpoint (provides the frozen encoder weights "
                        "for the JEPA arm)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92); print("SIDE-BY-SIDE COMPARISON"); print("=" * 92)

    print("[1/5] Loading models...")
    sup, sup_ckpt = load_supervised(args.supervised_ckpt, device)
    enc_j, rd_j, jr_ckpt = load_jepa_readout(args.jepa_readout_ckpt, device)
    jepa_ckpt = robust_load(args.jepa_ckpt, device)
    state_key = "online_state_dict" if "online_state_dict" in jepa_ckpt else "encoder_state_dict"
    enc_j.load_state_dict(jepa_ckpt[state_key])
    enc_j.eval()
    for p_ in enc_j.parameters(): p_.requires_grad = False
    print(f"  supervised : {args.supervised_ckpt}")
    print(f"  jepa enc   : {args.jepa_ckpt}")
    print(f"  jepa rd    : {args.jepa_readout_ckpt}")

    # Sanity: target_names and standardization should match between the two checkpoints.
    sup_names = sup_ckpt["target_names"]; jr_names = jr_ckpt["target_names"]
    if sup_names != jr_names:
        raise ValueError(f"target_names mismatch:\nsup={sup_names}\njr={jr_names}")
    target_names = sup_names

    print("[2/5] Rebuilding val split (same protocol as the two trainers)...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    min_spread_per_stock = raw["min_spread_z_per_stock"].astype(np.float32)
    raw_feat, _ = derive_raw_features_array(book, mid_z, stock_ids, min_spread_per_stock)

    bid_v, ask_v = book[:, 0, :, 1], book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & \
               (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    K = sup.encoder.cfg.K
    max_h = max(max(FUTURE_HORIZONS), max(VOL_HORIZONS))
    valid_t = compute_valid_endpoints(stock_ids, day_ids, K, max_h, vol_mask)
    _, val_pos = grouped_split_by_stock_day(stock_ids, day_ids, valid_t, args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    if args.max_val_samples > 0 and len(val_pos) > args.max_val_samples:
        val_pos = np.sort(rng.choice(val_pos, args.max_val_samples, replace=False))
    t_val = valid_t[val_pos]
    print(f"  val endpoints: {len(t_val):,}")

    # Reconstruct train-set targets just to recover (mean, std) for standardization.
    # We use the stored target_mean/std from the supervised checkpoint to avoid recomputing.
    print("[3/5] Building val target vectors with the trained standardization...")
    y_val_raw, _ = build_targets(book, mid_z, stock_ids, t_val, raw_feat, min_spread_per_stock)
    target_mean = np.asarray(sup_ckpt["target_mean"], dtype=np.float32)
    target_std = np.asarray(sup_ckpt["target_std"], dtype=np.float32)
    y_val_z = (y_val_raw - target_mean) / np.maximum(target_std, 1e-8)
    print(f"  y_val_z: {y_val_z.shape}")

    stock_stats = {k: np.asarray(v, dtype=np.float32) if not isinstance(v, (int, float)) else v
                   for k, v in sup_ckpt["stock_stats"].items()}

    print("[4/5] Forward passes on val...")
    print("  supervised-grid forward...")
    sup_pred, _ = evaluate_model_on_val(lambda b, s: sup(b, s),
                                        book, mid_z, stock_ids, t_val, stock_stats, K,
                                        y_val_z, device, args.batch_size)
    print("  jepa-grid-readout forward...")
    def jepa_fwd(b, s):
        g = enc_j(b, s); return rd_j(g)
    jepa_pred, _ = evaluate_model_on_val(jepa_fwd,
                                         book, mid_z, stock_ids, t_val, stock_stats, K,
                                         y_val_z, device, args.batch_size)

    print("[5/5] Metrics...")
    r2_sup = r2_per_target(y_val_z, sup_pred)
    r2_jep = r2_per_target(y_val_z, jepa_pred)
    s_sup = summarize_r2(target_names, r2_sup)
    s_jep = summarize_r2(target_names, r2_jep)

    print()
    print("Per-target R² (val):")
    print(f"  {'target':<28s} {'supervised':>12s} {'jepa':>12s} {'Δ (sup-jepa)':>14s}")
    print("  " + "-" * 70)
    for i, name in enumerate(target_names):
        d = r2_sup[i] - r2_jep[i]
        print(f"  {name:<28s} {r2_sup[i]:>+12.4f} {r2_jep[i]:>+12.4f} {d:>+14.4f}")
    print("  " + "-" * 70)
    print(f"  {'mean_all':<28s} {s_sup['mean_all']:>+12.4f} {s_jep['mean_all']:>+12.4f} "
          f"{s_sup['mean_all']-s_jep['mean_all']:>+14.4f}")
    print(f"  {'mean_future':<28s} {s_sup['mean_future']:>+12.4f} {s_jep['mean_future']:>+12.4f} "
          f"{s_sup['mean_future']-s_jep['mean_future']:>+14.4f}")
    print(f"  {'mean_vol':<28s} {s_sup['mean_vol']:>+12.4f} {s_jep['mean_vol']:>+12.4f} "
          f"{s_sup['mean_vol']-s_jep['mean_vol']:>+14.4f}")

    result = {
        "target_names": target_names,
        "r2_supervised": r2_sup.tolist(),
        "r2_jepa": r2_jep.tolist(),
        "summary_supervised": s_sup,
        "summary_jepa": s_jep,
        "n_val": int(len(t_val)),
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_dir/'comparison.json'}")


if __name__ == "__main__":
    main()

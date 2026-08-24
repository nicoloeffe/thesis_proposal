#!/usr/bin/env python3
"""
extract_readouts_multiseed.py — STAGE 1 of the accessibility-ladder battery.

Purpose
=======
Extract fixed (non-learned) 512-dim readouts from every frozen encoder produced
by the multiseed run, on ONE shared train/val subsample, and dump them to disk.
The rank-ladder analysis (PCA + min-norm OLS + MLP/ridge references) is STAGE 2
and consumes these dumps — so re-running the ladder (new m-schedule, extra
target) never re-touches the encoders, which are ~90% of the cost.

What it does NOT do (by design, kept out on purpose):
  - no PCA, no OLS, no R^2, no MLP, no ridge, no SVD harmonization, no plotting.
  Those live in the STAGE-2 ladder script. This file only produces the readouts
  and the shared raw targets, aligned row-for-row.

Two poolings, BOTH fixed / non-learned / 512-dim:
  - last_concat512 : grid[:, -1].reshape(B, S*D)     (last timestep, task-aligned)
  - tmean_concat512: grid.mean(dim=1).reshape(B, S*D) (temporal mean, robustness)
Attention pooling is deliberately excluded: it is a *learned* reader and would
reintroduce adaptation into a probe meant to measure variance geometry.

Contract / alignment guarantees
===============================
  * valid_t is built with K=20, max_h=20, and the SAME vol_mask (vol_clip=5.0)
    as the training arms -> identical endpoint set (bit-identical split).
  * grouped_split_by_stock_day(..., val_frac, split_seed=0) -> same held-out
    stock-days the encoders never trained on (no leakage in the probe val set).
  * grouped_split_by_stock_day returns POSITIONS inside valid_t.  These are
    mapped exactly once to raw endpoints (train_t=valid_t[train_pos], likewise
    for validation) before any target/window access.
  * the train/val subsample (n_train / n_val) is drawn ONCE with a fixed rng and
    REUSED for every checkpoint, so every readout dump and the shared targets
    share one row ordering. DataLoader uses shuffle=False to preserve it.
  * stock_stats used to normalize the encoder input is the encoder's OWN (what
    it was trained with); we assert all checkpoints agree with a reference,
    since the fixed split makes train-only stats identical across arms/seeds.

Outputs (out_dir/)
==================
  split.npz                      canonical raw endpoints + position audit + hashes
  targets_shared.npz             y_train_raw, y_val_raw (N,22 RAW), target_names
  readouts/{arm}_seed{N}_ep{E}.npz
                                 last_concat512_{train,val}, tmean_concat512_{train,val}
  analysis_manifest.json         provenance, fingerprints and validated inventory

Expected checkpoint layout (matches the agent multiseed brief):
  {ckpt_root}/{arm}/seed{N}/epoch_XXX.pt      arm in {jepa_horizon, jepa_masked, supervised}
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from analysis_artifacts import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    READOUT_SCHEMA,
    READOUT_VERSION,
    TARGET_SCHEMA,
    TARGET_VERSION,
    atomic_savez,
    atomic_write_json,
    build_endpoint_split,
    canonical_sha256,
    endpoint_sha256,
    load_split,
    save_split,
    sha256_array,
    sha256_file,
)

# --- Project imports: mirror the sys.path robustness of the existing probes ---
_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, *_THIS.parents, Path.cwd(), *Path.cwd().parents]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _imp(modname: str):
    """Import training.<mod> or flat <mod>, whichever the layout exposes."""
    import importlib
    for cand in (f"training.{modname}", modname):
        try:
            return importlib.import_module(cand)
        except Exception:
            continue
    raise SystemExit(f"Cannot import {modname!r} (tried training.{modname} and {modname}). "
                     "Run from the thesis project root.")


_tok = _imp("train_tokenizer_t")
compute_valid_endpoints = _tok.compute_valid_endpoints
normalize_book_window = _tok.normalize_book_window
grouped_split_by_stock_day = _tok.grouped_split_by_stock_day
derive_raw_features_array = _tok.derive_raw_features_array
compute_future_feature_targets = _tok.compute_future_feature_targets
compute_vol_targets = _tok.compute_vol_targets

_hor = _imp("train_jepa_horizon")
HorizonJEPAEncoder = _hor.HorizonJEPAEncoder
HorizonJEPAEncoderConfig = _hor.HorizonJEPAEncoderConfig

_msk = _imp("train_jepa_masked")
MaskedJEPAEncoder = _msk.MaskedJEPAEncoder
MaskedJEPAEncoderConfig = _msk.MaskedJEPAEncoderConfig

_sup = _imp("train_supervised_grid")
SupervisedGrid = _sup.SupervisedGrid
ReadoutConfig = _sup.ReadoutConfig

# Target definitions — the SAME 22 raw columns the supervised arm trains on.
FUTURE_FEATURES = ["d_spread_z", "d_microprice_rel", "d_best_bid_rel",
                   "d_best_ask_rel", "d_top_imbalance"]
FUTURE_HORIZONS = [1, 5, 10, 20]
VOL_HORIZONS = [5, 20]
VOL_CLIP = 5.0
ARMS = ("jepa_horizon", "jepa_masked", "supervised")


def robust_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def to_numpy_stats(stock_stats: Dict) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stock_stats.items()}


def stock_stats_fingerprint(stock_stats: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            key: sha256_array(np.asarray(value, dtype=np.float32))
            for key, value in sorted(stock_stats.items())
        }
    )


def checkpoint_stock_stats_fingerprint(path: Path) -> str:
    checkpoint = robust_torch_load(path, torch.device("cpu"))
    if "stock_stats" not in checkpoint:
        raise ValueError(f"{path}: no stock_stats in checkpoint")
    return stock_stats_fingerprint(to_numpy_stats(checkpoint["stock_stats"]))


class RawWindowDataset(Dataset):
    """Copy of the probe's dataset: yields normalized (K, ...) windows for endpoint t."""
    def __init__(self, book, mid_z, stock_ids, valid_t, stock_stats, K):
        self.book = book; self.mid_z = mid_z; self.stock_ids = stock_ids
        self.valid_t = valid_t; self.stock_stats = stock_stats; self.K = K

    def __len__(self):
        return len(self.valid_t)

    def __getitem__(self, idx):
        t = int(self.valid_t[idx]); s = int(self.stock_ids[t]); K = self.K
        book_win = self.book[t - K + 1: t + 1]
        mid_win = self.mid_z[t - K + 1: t + 1]
        book_norm = normalize_book_window(book_win, mid_win, s, self.stock_stats)
        return torch.from_numpy(book_norm).float(), torch.tensor(s, dtype=torch.long)


# ------------------------------------------------------------------------------
# Per-arm loading: return a uniform encode(book, stock_ids) -> (B,K,S,D) callable
# ------------------------------------------------------------------------------

def _encode_fn(module) -> Callable:
    """Wrap an encoder module so it can be called with or without mask=None."""
    def encode(book, stock_ids):
        try:
            return module(book, stock_ids, mask=None)
        except TypeError:
            return module(book, stock_ids)
    return encode


def load_encoder(arm: str, ckpt_path: str, device: torch.device):
    """Return (encode_callable, stock_stats_dict). Discards any supervised readout."""
    ckpt = robust_torch_load(ckpt_path, device)
    if "stock_stats" not in ckpt:
        raise ValueError(f"{ckpt_path}: no stock_stats in checkpoint")
    stock_stats = to_numpy_stats(ckpt["stock_stats"])

    if arm == "jepa_horizon":
        enc = HorizonJEPAEncoder(HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])).to(device)
        state = ckpt.get("online_state_dict", ckpt.get("encoder_state_dict"))
        if state is None:
            raise ValueError(f"{ckpt_path}: missing online_state_dict")
        enc.load_state_dict(state); enc.eval()
        return _encode_fn(enc), stock_stats

    if arm == "jepa_masked":
        enc = MaskedJEPAEncoder(MaskedJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])).to(device)
        state = ckpt.get("online_state_dict", ckpt.get("encoder_state_dict"))
        if state is None:
            raise ValueError(f"{ckpt_path}: missing online_state_dict")
        enc.load_state_dict(state); enc.eval()
        return _encode_fn(enc), stock_stats

    if arm == "supervised":
        model = SupervisedGrid(
            HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"]),
            ReadoutConfig(**ckpt["readout_cfg"]),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return _encode_fn(model.encoder), stock_stats  # readout discarded

    raise ValueError(f"unknown arm {arm!r}")


@torch.no_grad()
def extract_poolings(encode: Callable, ds: Dataset, batch_size: int,
                     num_workers: int, device: torch.device) -> Dict[str, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=num_workers > 0, drop_last=False)
    last_c, tmean_c = [], []
    for book, sid in loader:
        book = book.to(device, non_blocking=True)
        sid = sid.to(device, non_blocking=True)
        grid = encode(book, sid)                      # (B, K, S, D)
        B = grid.shape[0]
        last = grid[:, -1, :, :]                      # (B, S, D)  last timestep
        tmean = grid.mean(dim=1)                      # (B, S, D)  mean over K
        last_c.append(last.reshape(B, -1).float().cpu().numpy().astype(np.float32))
        tmean_c.append(tmean.reshape(B, -1).float().cpu().numpy().astype(np.float32))
    return {
        "last_concat512": np.concatenate(last_c, axis=0),
        "tmean_concat512": np.concatenate(tmean_c, axis=0),
    }


def build_raw_targets(book, mid_z, stock_ids, valid_t_sub, min_spread_per_stock) -> Tuple[np.ndarray, List[str]]:
    n_stocks = int(stock_ids.max()) + 1
    raw_feat, _ = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)
    fut = compute_future_feature_targets(raw_feat, valid_t_sub, FUTURE_FEATURES, FUTURE_HORIZONS)
    vol = compute_vol_targets(mid_z, valid_t_sub, VOL_HORIZONS, min_spread_per_stock, stock_ids)
    y = np.concatenate([fut, vol], axis=1).astype(np.float32)   # RAW, unstandardized
    names = [f"{f}@{h}" for f in FUTURE_FEATURES for h in FUTURE_HORIZONS]
    names += [f"realized_vol@{h}" for h in VOL_HORIZONS]
    return y, names


def parse_epoch(p: Path) -> Optional[int]:
    m = re.search(r"epoch_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else None


READOUT_ARRAYS = (
    "last_concat512_train",
    "last_concat512_val",
    "tmean_concat512_train",
    "tmean_concat512_val",
)
SOURCE_FILES = (
    "experiment01/reference/analysis_artifacts.py",
    "experiment01/reference/extract_readouts_multiseed.py",
    "models/model_tokenizer_t.py",
    "training/train_tokenizer_t.py",
    "training/train_jepa_horizon.py",
    "training/train_jepa_masked.py",
    "training/train_supervised_grid.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_info(repo_root: Path) -> Dict[str, Any]:
    def run(*argv: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *argv], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD") or None,
        # This is provenance only. Compatibility is decided by source hashes.
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def environment_info(device: torch.device) -> Dict[str, Any]:
    device_name = platform.processor() or "cpu"
    if device.type == "cuda":
        try:
            device_name = torch.cuda.get_device_name(device)
        except Exception:
            device_name = "cuda"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "rocm": getattr(torch.version, "hip", None),
        "actual_device": str(device),
        "device_name": device_name,
    }


def source_inventory(repo_root: Path) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for rel in SOURCE_FILES:
        path = repo_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"required extraction source is missing: {path}")
        files[rel] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    fingerprint = canonical_sha256(
        {rel: meta["sha256"] for rel, meta in sorted(files.items())}
    )
    return {"fingerprint": fingerprint, "files": files}


def discover_checkpoints(
    ckpt_root: Path, arms: List[str], seeds: List[int], epochs: str
) -> Dict[str, Dict[str, Any]]:
    requested: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        for seed in seeds:
            seed_dir = ckpt_root / arm / f"seed{seed}"
            if not seed_dir.is_dir():
                raise FileNotFoundError(f"checkpoint directory is missing: {seed_dir}")
            all_ckpts = sorted(
                (p for p in seed_dir.glob("epoch_*.pt") if parse_epoch(p) is not None),
                key=lambda p: int(parse_epoch(p)),
            )
            if epochs == "all":
                chosen = all_ckpts
            elif epochs == "last":
                chosen = all_ckpts[-1:]
            else:
                wanted = {int(item.strip()) for item in epochs.split(",") if item.strip()}
                chosen = [p for p in all_ckpts if parse_epoch(p) in wanted]
                found = {int(parse_epoch(p)) for p in chosen}
                missing = sorted(wanted - found)
                if missing:
                    raise FileNotFoundError(
                        f"{seed_dir}: missing requested checkpoint epochs {missing}"
                    )
            if not chosen:
                raise FileNotFoundError(f"no checkpoint selected in {seed_dir}")
            for path in chosen:
                epoch = int(parse_epoch(path))
                tag = f"{arm}_seed{seed}_ep{epoch:03d}"
                requested[tag] = {
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "stock_stats_sha256": checkpoint_stock_stats_fingerprint(path),
                }
    return dict(sorted(requested.items()))


def _npz_scalar(npz: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(npz[key])
    if value.ndim != 0:
        raise ValueError(f"{key} must be a scalar, got shape {value.shape}")
    return value.item()


def _matching_file_record(path: Path, record: Mapping[str, Any]) -> Tuple[bool, str]:
    if not path.is_file():
        return False, "file is missing"
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        return False, "file size differs from manifest"
    if record.get("file_sha256") != sha256_file(path):
        return False, "file SHA-256 differs from manifest"
    return True, "ok"


def validate_targets_for_resume(
    path: Path,
    record: Optional[Mapping[str, Any]],
    expected: Mapping[str, Any],
    n_train: int,
    n_val: int,
    n_targets: int,
) -> Tuple[bool, str]:
    if record is None:
        return False, "manifest entry is missing"
    ok, why = _matching_file_record(path, record)
    if not ok:
        return ok, why
    for key in (
        "artifact_fingerprint",
        "train_endpoint_sha256",
        "val_endpoint_sha256",
    ):
        if record.get(key) != expected[key]:
            return False, f"target manifest metadata mismatch: {key}"
    if record.get("shape_train") != [n_train, n_targets]:
        return False, "target manifest train shape mismatch"
    if record.get("shape_val") != [n_val, n_targets]:
        return False, "target manifest validation shape mismatch"
    required = {
        "schema_name", "schema_version", "artifact_fingerprint",
        "protocol_fingerprint", "split_fingerprint", "dataset_sha256",
        "train_endpoint_sha256", "val_endpoint_sha256", "n_train", "n_val",
        "y_train_raw", "y_val_raw", "target_names",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != required:
                return False, "target NPZ key set differs from schema"
            for key, value in expected.items():
                if _npz_scalar(data, key) != value:
                    return False, f"target metadata mismatch: {key}"
            if data["y_train_raw"].shape != (n_train, n_targets):
                return False, "target train shape mismatch"
            if data["y_val_raw"].shape != (n_val, n_targets):
                return False, "target validation shape mismatch"
            if data["y_train_raw"].dtype != np.float32 or data["y_val_raw"].dtype != np.float32:
                return False, "target dtype mismatch"
            if data["target_names"].shape != (n_targets,):
                return False, "target name shape mismatch"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"cannot validate target NPZ: {exc}"
    return True, "ok"


def validate_readout_for_resume(
    path: Path,
    record: Optional[Mapping[str, Any]],
    expected: Mapping[str, Any],
    n_train: int,
    n_val: int,
) -> Tuple[bool, str]:
    """Fail-closed validation used before a pre-existing dump may be skipped."""
    if record is None:
        return False, "manifest entry is missing"
    ok, why = _matching_file_record(path, record)
    if not ok:
        return ok, why
    for key in (
        "artifact_fingerprint",
        "checkpoint_sha256",
        "stock_stats_sha256",
        "train_endpoint_sha256",
        "val_endpoint_sha256",
    ):
        if record.get(key) != expected[key]:
            return False, f"readout manifest metadata mismatch: {key}"
    expected_arrays = {
        "last_concat512_train": {"shape": [n_train, 512], "dtype": "float32"},
        "last_concat512_val": {"shape": [n_val, 512], "dtype": "float32"},
        "tmean_concat512_train": {"shape": [n_train, 512], "dtype": "float32"},
        "tmean_concat512_val": {"shape": [n_val, 512], "dtype": "float32"},
    }
    if record.get("arrays") != expected_arrays:
        return False, "readout manifest array schema mismatch"
    metadata_keys = {
        "schema_name", "schema_version", "artifact_fingerprint",
        "protocol_fingerprint", "split_fingerprint", "dataset_sha256",
        "train_endpoint_sha256", "val_endpoint_sha256", "checkpoint_sha256",
        "stock_stats_sha256", "arm", "seed", "epoch", "n_train", "n_val",
    }
    required = metadata_keys | set(READOUT_ARRAYS)
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != required:
                return False, "readout NPZ key set differs from schema"
            for key, value in expected.items():
                if _npz_scalar(data, key) != value:
                    return False, f"readout metadata mismatch: {key}"
            shapes = {
                "last_concat512_train": (n_train, 512),
                "last_concat512_val": (n_val, 512),
                "tmean_concat512_train": (n_train, 512),
                "tmean_concat512_val": (n_val, 512),
            }
            for key, shape in shapes.items():
                if data[key].shape != shape:
                    return False, f"{key} shape mismatch"
                if data[key].dtype != np.float32:
                    return False, f"{key} dtype mismatch"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"cannot validate readout NPZ: {exc}"
    return True, "ok"


def _manifest_compatible(existing: Mapping[str, Any], planned: Mapping[str, Any]) -> None:
    schema = existing.get("schema", {})
    if schema != {"name": MANIFEST_SCHEMA, "version": MANIFEST_VERSION}:
        raise RuntimeError("output manifest has an unsupported schema")
    def checkpoint_identity(manifest: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            tag: {
                key: metadata.get(key)
                for key in (
                    "arm", "seed", "epoch", "sha256", "stock_stats_sha256"
                )
            }
            for tag, metadata in manifest.get("requested_checkpoints", {}).items()
        }

    comparisons = (
        ("dataset SHA-256", existing.get("dataset", {}).get("sha256"),
         planned["dataset"]["sha256"]),
        ("source fingerprint", existing.get("source", {}).get("fingerprint"),
         planned["source"]["fingerprint"]),
        ("protocol fingerprint", existing.get("protocol", {}).get("fingerprint"),
         planned["protocol"]["fingerprint"]),
        ("split fingerprint", existing.get("split", {}).get("fingerprint"),
         planned["split"]["fingerprint"]),
        ("requested checkpoints", checkpoint_identity(existing),
         checkpoint_identity(planned)),
    )
    mismatches = [label for label, old, new in comparisons if old != new]
    if mismatches:
        raise RuntimeError(
            "output directory is incompatible with this run ("
            + ", ".join(mismatches)
            + "); use a new --out_dir"
        )


def _write_manifest(path: Path, manifest: Dict[str, Any], status: str) -> None:
    manifest["status"] = status
    manifest["updated_at_utc"] = utc_now()
    manifest["summary"]["complete"] = len(manifest["readouts"])
    expected = set(manifest["requested_checkpoints"])
    manifest["summary"]["missing"] = sorted(expected - set(manifest["readouts"]))
    atomic_write_json(path, manifest)


def main():
    ap = argparse.ArgumentParser(description="Stage 1: extract fixed readouts from frozen multiseed encoders")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt_root", required=True, help="root with {arm}/seed{N}/epoch_XXX.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--max_h", type=int, default=20)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=100_000)
    ap.add_argument("--n_val", type=int, default=50_000)
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=str, default="all",
                    help="'all', 'last', or comma list e.g. '8,10,12,20'")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but PyTorch cannot access a CUDA/ROCm "
            "device; refusing a silent CPU fallback"
        )
    device = torch.device(args.device)
    out = Path(args.out_dir)
    manifest_path = out / "analysis_manifest.json"
    split_path = out / "split.npz"
    targets_path = out / "targets_shared.npz"
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    repo_root = _THIS.parents[2]
    dataset_path = Path(args.dataset)
    ckpt_root = Path(args.ckpt_root)

    print("[1/4] Loading dataset...")
    dataset_sha256 = sha256_file(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as raw:
        book = raw["book"]
        mid_z = raw["mid_z"]
        stock_ids = raw["stock_ids"].astype(np.int64)
        day_ids = raw["day_ids"].astype(np.int64)
        min_spread = raw["min_spread_z_per_stock"]

    print("[2/4] Building shared valid_t + split + subsample (once, fixed)...")
    bid_v, ask_v = book[:, 0, :, 1], book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= VOL_CLIP) & (np.abs(ask_v).max(axis=1) <= VOL_CLIP)
    split = build_endpoint_split(
        stock_ids,
        day_ids,
        vol_mask,
        K=args.K,
        max_horizon=args.max_h,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        n_train=args.n_train,
        n_val=args.n_val,
        subsample_seed=args.subsample_seed,
        compute_valid_endpoints_fn=compute_valid_endpoints,
        grouped_split_fn=grouped_split_by_stock_day,
    )
    train_t, val_t = split.train_t, split.val_t
    print(
        f"      valid_t={len(split.valid_t):,}  "
        f"train_sub={len(train_t):,}  val_sub={len(val_t):,}"
    )

    requested_checkpoints = discover_checkpoints(ckpt_root, arms, seeds, args.epochs)
    stock_stats_hashes = {
        metadata["stock_stats_sha256"]
        for metadata in requested_checkpoints.values()
    }
    if len(stock_stats_hashes) > 1:
        raise RuntimeError(
            "requested checkpoints do not share identical train-only stock_stats"
        )
    source = source_inventory(repo_root)
    env = environment_info(device)
    split_config = {
        "K": args.K,
        "max_horizon": args.max_h,
        "vol_clip": VOL_CLIP,
        "val_frac": args.val_frac,
        "split_seed": args.split_seed,
        "subsample_seed": args.subsample_seed,
        "requested_n_train": args.n_train,
        "requested_n_val": args.n_val,
        "grouping": "stock_id+day_id",
        "split_algorithm": "grouped_split_by_stock_day.v1",
    }
    split.bind(dataset_sha256=dataset_sha256, config=split_config)
    numeric_environment = {
        "numpy": env["numpy"],
        "torch": env["torch"],
        "rocm": env["rocm"],
        "actual_device": env["actual_device"],
    }
    protocol_payload = {
        "schema_versions": {
            "manifest": MANIFEST_VERSION,
            "readout": READOUT_VERSION,
            "target": TARGET_VERSION,
        },
        "dataset_sha256": dataset_sha256,
        "source_fingerprint": source["fingerprint"],
        "split_fingerprint": split.split_fingerprint,
        "checkpoint_sha256": {
            tag: meta["sha256"] for tag, meta in requested_checkpoints.items()
        },
        "stock_stats_sha256": sorted(stock_stats_hashes),
        "poolings": ["last_concat512", "tmean_concat512"],
        "readout_definition": {
            "last_concat512": "grid[:, -1].reshape(B, S*D)",
            "tmean_concat512": "grid.mean(axis=1).reshape(B, S*D)",
        },
        "numeric_environment": numeric_environment,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    protocol_fingerprint = canonical_sha256(protocol_payload)
    now = utc_now()
    planned_manifest: Dict[str, Any] = {
        "schema": {"name": MANIFEST_SCHEMA, "version": MANIFEST_VERSION},
        "status": "initializing",
        "created_at_utc": now,
        "updated_at_utc": now,
        "git": git_info(repo_root),
        "environment": env,
        "source": source,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": dataset_sha256,
            "size_bytes": dataset_path.stat().st_size,
        },
        "protocol": {
            **split_config,
            "fingerprint": protocol_fingerprint,
            "poolings": ["last_concat512", "tmean_concat512"],
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "split": {
            "path": split_path.name,
            "file_sha256": None,
            "fingerprint": split.split_fingerprint,
            "n_valid_total": len(split.valid_t),
            "n_train": len(train_t),
            "n_val": len(val_t),
            "train_endpoint_sha256": split.train_endpoint_sha256,
            "val_endpoint_sha256": split.val_endpoint_sha256,
        },
        "targets": None,
        "requested_checkpoints": requested_checkpoints,
        "readouts": {},
        "summary": {
            "expected": len(requested_checkpoints),
            "complete": 0,
            "missing": sorted(requested_checkpoints),
        },
    }

    if out.exists() and any(out.iterdir()):
        if not manifest_path.is_file():
            raise RuntimeError(
                f"{out} is non-empty but has no compatible analysis_manifest.json; "
                "use a new --out_dir"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        _manifest_compatible(existing_manifest, planned_manifest)
        manifest = existing_manifest
        manifest["git"] = planned_manifest["git"]
        manifest["environment"] = env
        manifest["source"] = source
        manifest["protocol"] = planned_manifest["protocol"]
        manifest["requested_checkpoints"] = requested_checkpoints
    else:
        out.mkdir(parents=True, exist_ok=True)
        manifest = planned_manifest
    (out / "readouts").mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, manifest, "initializing")

    split_is_valid = False
    if split_path.is_file() and manifest["split"].get("file_sha256"):
        try:
            loaded_split = load_split(
                split_path, expected_dataset_sha256=dataset_sha256
            )
            split_is_valid = (
                loaded_split.split_fingerprint == split.split_fingerprint
                and manifest["split"]["file_sha256"] == sha256_file(split_path)
            )
        except (OSError, ValueError, KeyError):
            split_is_valid = False
    if not split_is_valid:
        save_split(
            split_path,
            split,
            dataset_sha256=dataset_sha256,
            split_config=split_config,
        )
    manifest["split"].update(
        file_sha256=sha256_file(split_path),
        size_bytes=split_path.stat().st_size,
    )
    _write_manifest(manifest_path, manifest, "extracting")

    print("[3/4] Building shared RAW targets (aligned to subsample)...")
    y_train, names = build_raw_targets(book, mid_z, stock_ids, train_t, min_spread)
    y_val, names_val = build_raw_targets(book, mid_z, stock_ids, val_t, min_spread)
    if names_val != names:
        raise RuntimeError("train and validation target schemas differ")
    target_artifact_fingerprint = canonical_sha256(
        {
            "schema": TARGET_SCHEMA,
            "version": TARGET_VERSION,
            "protocol_fingerprint": protocol_fingerprint,
            "split_fingerprint": split.split_fingerprint,
            "target_names": names,
        }
    )
    target_metadata = {
        "schema_name": TARGET_SCHEMA,
        "schema_version": TARGET_VERSION,
        "artifact_fingerprint": target_artifact_fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "split_fingerprint": split.split_fingerprint,
        "dataset_sha256": dataset_sha256,
        "train_endpoint_sha256": split.train_endpoint_sha256,
        "val_endpoint_sha256": split.val_endpoint_sha256,
        "n_train": len(train_t),
        "n_val": len(val_t),
    }
    targets_record = manifest.get("targets")
    targets_ok, targets_why = validate_targets_for_resume(
        targets_path,
        targets_record,
        target_metadata,
        len(train_t),
        len(val_t),
        len(names),
    )
    if targets_ok:
        print("      [reuse] validated targets_shared.npz")
    else:
        if targets_path.exists():
            print(f"      [regen] targets_shared.npz: {targets_why}")
        atomic_savez(
            targets_path,
            **target_metadata,
            y_train_raw=y_train.astype(np.float32, copy=False),
            y_val_raw=y_val.astype(np.float32, copy=False),
            target_names=np.asarray(names, dtype=str),
        )
    manifest["targets"] = {
        "path": targets_path.name,
        "file_sha256": sha256_file(targets_path),
        "size_bytes": targets_path.stat().st_size,
        "artifact_fingerprint": target_artifact_fingerprint,
        "shape_train": list(y_train.shape),
        "shape_val": list(y_val.shape),
        "train_endpoint_sha256": split.train_endpoint_sha256,
        "val_endpoint_sha256": split.val_endpoint_sha256,
    }
    _write_manifest(manifest_path, manifest, "extracting")
    print(f"      targets y_train={y_train.shape} y_val={y_val.shape}  ({len(names)} cols)")

    print("[4/4] Extracting readouts per checkpoint...")
    ref_stats = None
    expected_paths = {
        str(Path("readouts") / f"{tag}.npz") for tag in requested_checkpoints
    }
    actual_paths = {
        str(path.relative_to(out)) for path in (out / "readouts").glob("*.npz")
    }
    unexpected = sorted(actual_paths - expected_paths)
    if unexpected:
        raise RuntimeError(
            f"unexpected readout NPZ files in compatible output directory: {unexpected}"
        )

    for tag, checkpoint in requested_checkpoints.items():
        arm = str(checkpoint["arm"])
        seed = int(checkpoint["seed"])
        epoch = int(checkpoint["epoch"])
        checkpoint_path = Path(str(checkpoint["path"]))
        dst = out / "readouts" / f"{tag}.npz"
        artifact_fingerprint = canonical_sha256(
            {
                "schema": READOUT_SCHEMA,
                "version": READOUT_VERSION,
                "protocol_fingerprint": protocol_fingerprint,
                "split_fingerprint": split.split_fingerprint,
                "checkpoint_sha256": checkpoint["sha256"],
                "stock_stats_sha256": checkpoint["stock_stats_sha256"],
                "arm": arm,
                "seed": seed,
                "epoch": epoch,
                "arrays": list(READOUT_ARRAYS),
            }
        )
        metadata = {
            "schema_name": READOUT_SCHEMA,
            "schema_version": READOUT_VERSION,
            "artifact_fingerprint": artifact_fingerprint,
            "protocol_fingerprint": protocol_fingerprint,
            "split_fingerprint": split.split_fingerprint,
            "dataset_sha256": dataset_sha256,
            "train_endpoint_sha256": split.train_endpoint_sha256,
            "val_endpoint_sha256": split.val_endpoint_sha256,
            "checkpoint_sha256": checkpoint["sha256"],
            "stock_stats_sha256": checkpoint["stock_stats_sha256"],
            "arm": arm,
            "seed": seed,
            "epoch": epoch,
            "n_train": len(train_t),
            "n_val": len(val_t),
        }
        old_record = manifest["readouts"].get(tag)
        valid, why = validate_readout_for_resume(
            dst, old_record, metadata, len(train_t), len(val_t)
        )
        if valid:
            print(f"  [reuse] {tag} (schema + metadata + SHA-256 valid)")
            continue
        if dst.exists():
            print(f"  [regen] {tag}: {why}")
        t0 = time.time()
        encode, stats = load_encoder(arm, str(checkpoint_path), device)
        if stock_stats_fingerprint(stats) != checkpoint["stock_stats_sha256"]:
            raise RuntimeError(f"{tag}: loaded stock_stats fingerprint changed")
        if ref_stats is None:
            ref_stats = stats
        else:
            for key in ref_stats:
                if not np.allclose(
                    np.asarray(stats[key]), np.asarray(ref_stats[key]), atol=1e-5
                ):
                    raise RuntimeError(
                        f"{tag}: stock_stats[{key!r}] differs from reference"
                    )
        ds_tr = RawWindowDataset(book, mid_z, stock_ids, train_t, stats, args.K)
        ds_va = RawWindowDataset(book, mid_z, stock_ids, val_t, stats, args.K)
        rtr = extract_poolings(
            encode, ds_tr, args.batch_size, args.num_workers, device
        )
        rva = extract_poolings(
            encode, ds_va, args.batch_size, args.num_workers, device
        )
        if rtr["last_concat512"].shape[1] != 512:
            raise RuntimeError(
                f"{tag}: expected a 512-dimensional readout, got "
                f"{rtr['last_concat512'].shape[1]}"
            )
        atomic_savez(
            dst,
            **metadata,
            last_concat512_train=rtr["last_concat512"],
            last_concat512_val=rva["last_concat512"],
            tmean_concat512_train=rtr["tmean_concat512"],
            tmean_concat512_val=rva["tmean_concat512"],
        )
        arrays = {
            key: {
                "shape": list(
                    rtr[key.removesuffix("_train")].shape
                    if key.endswith("_train")
                    else rva[key.removesuffix("_val")].shape
                ),
                "dtype": "float32",
            }
            for key in READOUT_ARRAYS
        }
        manifest["readouts"][tag] = {
            "path": str(Path("readouts") / dst.name),
            "checkpoint_sha256": checkpoint["sha256"],
            "stock_stats_sha256": checkpoint["stock_stats_sha256"],
            "artifact_fingerprint": artifact_fingerprint,
            "file_sha256": sha256_file(dst),
            "size_bytes": dst.stat().st_size,
            "train_endpoint_sha256": split.train_endpoint_sha256,
            "val_endpoint_sha256": split.val_endpoint_sha256,
            "arrays": arrays,
        }
        _write_manifest(manifest_path, manifest, "extracting")
        print(
            f"  [done] {tag}  D={rtr['last_concat512'].shape[1]}  "
            f"{time.time() - t0:.1f}s"
        )
        del encode, rtr, rva
        if device.type == "cuda":
            torch.cuda.empty_cache()

    disk_paths = {
        str(path.relative_to(out)) for path in (out / "readouts").glob("*.npz")
    }
    manifest_paths = {
        str(record["path"]) for record in manifest["readouts"].values()
    }
    if disk_paths != expected_paths or manifest_paths != expected_paths:
        raise RuntimeError(
            "final readout inventory mismatch: "
            f"disk={len(disk_paths)}, manifest={len(manifest_paths)}, "
            f"expected={len(expected_paths)}"
        )
    _write_manifest(manifest_path, manifest, "complete")
    print(
        f"\nExtraction complete: {len(manifest['readouts'])} checkpoints "
        f"-> {out / 'readouts'}"
    )


if __name__ == "__main__":
    main()

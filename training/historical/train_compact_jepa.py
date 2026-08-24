#!/usr/bin/env python3
"""
train_compact_jepa.py — Compact Horizon JEPA: state-based multi-horizon JEPA.

Why this file exists
====================
The current Horizon JEPA (train_jepa_horizon.py) is token-preserving: the encoder
emits a grid H_t in R^{20x4x128}, and "the state" is only a post-hoc readout
(last_concat512). That is a legitimate token-level JEPA, but it is NOT a state
encoder directly comparable to the supervised tokenizer A1-T, which natively
emits z_t in R^32.

This file is the STATE-BASED reformulation. The training objective is unchanged:
online encoder sees W_t, EMA target encoder sees W_{t+H}, a horizon-conditioned
predictor maps the current state to the future state, loss is the LayerNorm-
normalized distance between predicted and target latent. The ONLY change is the
backbone: the encoder is A1-T-like and emits z_t in R^32 directly, and the
predictor operates state-to-state (z -> z) instead of grid-to-grid.

    grid Horizon JEPA :  W_t -> H_t (B,20,4,128)   ; predictor: grid -> grid
    Compact Horizon JEPA: W_t -> z_t (B,32)        ; predictor: z   -> z

The grid trainer is left untouched and remains usable as the token-level
ablation. Components reused (not duplicated):
  - LOBSpatialEncoder, LOBTemporalEncoder   from model_tokenizer_t  (A1-T backbone)
  - HorizonJEPADataset, parse_horizons      from train_jepa_horizon (data is identical)
  - data pipeline utilities                 from train_tokenizer_t

Methodological notes (agreed before writing this)
=================================================
  - The main run is PURE JEPA: no VICReg / variance / covariance term in the
    active loss. The variance hinge is pre-wired behind --lambda_var (default
    0.0, OFF) so that, if z collapses, we do not need to rewrite the trainer.
    With --lambda_var 0.0 the experiment is the pure compact JEPA.
  - Collapse is monitored from epoch 1: z_std_mean, z_std_min, z_eff_rank,
    z_norm, z_cov_offdiag, plus per-horizon loss/cosine and anchor diagnostics.
  - Checkpoints carry model_type="compact_horizon_jepa" so loaders dispatch
    correctly and the grid checkpoints stay loadable by their own loader.

After training: run the probe gate on z32 (same targets/protocol as A1-T) BEFORE
building the world-model latent dataset. Those are separate scripts.

Example
-------
python -m training.train_compact_jepa \
  --dataset data/lobench_processed.npz \
  --ckpt_dir checkpoints/compact_jepa/v1 \
  --epochs 20 --batch_size 256 --horizons 0,1,5,10,20

Smoke test (no dataset needed):
python -m training.train_compact_jepa --smoke_test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# =============================================================================
#  Project imports
# =============================================================================

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent, _THIS.parent.parent.parent,
           _THIS.parent.parent.parent.parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from models.model_tokenizer_t import LOBSpatialEncoder, LOBTemporalEncoder  # type: ignore
except Exception:
    LOBSpatialEncoder = None
    LOBTemporalEncoder = None

try:
    from training.historical.train_jepa_horizon import (  # type: ignore
        HorizonJEPADataset,
        parse_horizons,
    )
except Exception:
    HorizonJEPADataset = None
    parse_horizons = None

try:
    from training.train_tokenizer_t import (  # type: ignore
        compute_valid_endpoints,
        grouped_split_by_stock_day,
        compute_stock_stats_train_only,
    )
except Exception:
    compute_valid_endpoints = None
    grouped_split_by_stock_day = None
    compute_stock_stats_train_only = None


def _check_project_imports_or_exit() -> None:
    missing = []
    if LOBSpatialEncoder is None or LOBTemporalEncoder is None:
        missing.append("model_tokenizer_t (LOBSpatialEncoder/LOBTemporalEncoder)")
    if HorizonJEPADataset is None or parse_horizons is None:
        missing.append("train_jepa_horizon (HorizonJEPADataset/parse_horizons)")
    if compute_valid_endpoints is None or grouped_split_by_stock_day is None \
            or compute_stock_stats_train_only is None:
        missing.append("train_tokenizer_t (data utilities)")
    if missing:
        raise SystemExit(
            "Cannot import project modules: " + "; ".join(missing) + ". "
            "Run from the project root (so that `training` is importable)."
        )


# =============================================================================
#  Configs
# =============================================================================

@dataclass
class CompactHorizonJEPAEncoderConfig:
    """A1-T-like state encoder. Field names match TokenizerConfigT exactly so
    that this object can be passed directly to LOBSpatialEncoder/
    LOBTemporalEncoder (they only read attributes)."""

    L: int = 10
    n_stocks: int = 7
    K: int = 20

    d_model: int = 128
    d_latent: int = 32

    spatial_n_layers: int = 2
    spatial_n_heads: int = 4
    spatial_d_ffn: int = 256

    temporal_n_layers: int = 2
    temporal_n_heads: int = 4
    temporal_d_ffn: int = 256

    dropout: float = 0.1
    stock_emb_init_scale: float = 0.02

    @classmethod
    def from_dict(cls, d: Dict) -> "CompactHorizonJEPAEncoderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompactHorizonJEPAPredictorConfig:
    """Horizon-conditioned state predictor: (z_t, H) -> z_hat_{t+H}."""

    d_latent: int = 32
    n_horizons: int = 5
    d_horizon: int = 32          # horizon embedding dim
    d_hidden: int = 256
    n_layers: int = 2            # number of hidden layers in the MLP
    dropout: float = 0.0
    residual: bool = True        # z_hat = z + MLP([z, h]); identity-friendly for H=0/H=1

    @classmethod
    def from_dict(cls, d: Dict) -> "CompactHorizonJEPAPredictorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
#  Encoder — A1-T-like, emits z_t in R^{d_latent}
# =============================================================================

class CompactHorizonJEPAEncoder(nn.Module):
    """W_t -> z_t.

    Forward path (identical to A1-T's encode()):
        book:      (B, K, 2, L, 2)
        stock_ids: (B,)
        --> stock_embed                       (B, d_model)  broadcast over K
        --> LOBSpatialEncoder (per snapshot)   (B, K, d_model)
        --> LOBTemporalEncoder (causal)        (B, d_model)   [last timestep]
        --> Linear(d_model -> d_latent)        (B, d_latent)
    """

    def __init__(self, cfg: CompactHorizonJEPAEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.stock_embed = nn.Embedding(cfg.n_stocks, cfg.d_model)
        nn.init.trunc_normal_(self.stock_embed.weight, std=cfg.stock_emb_init_scale)
        # cfg duck-types as TokenizerConfigT for these two A1-T modules.
        self.spatial_encoder = LOBSpatialEncoder(cfg)
        self.temporal_encoder = LOBTemporalEncoder(cfg)
        self.proj = nn.Linear(cfg.d_model, cfg.d_latent)

    def forward(self, book: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, K = book.shape[0], book.shape[1]
        s_emb = self.stock_embed(stock_ids)                          # (B, d_model)
        s_emb_K = s_emb.unsqueeze(1).expand(B, K, cfg.d_model)        # (B, K, d_model)
        h_seq = self.spatial_encoder(book, s_emb_K)                  # (B, K, d_model)
        c_K = self.temporal_encoder(h_seq)                           # (B, d_model)
        z = self.proj(c_K)                                           # (B, d_latent)
        return z


# =============================================================================
#  Predictor — horizon-conditioned, state-to-state
# =============================================================================

class CompactHorizonJEPAPredictor(nn.Module):
    """(z_t, horizon_idx) -> z_hat_{t+H}.

    Residual MLP: z_hat = z + MLP([z, horizon_embed(H)]). Residual so that the
    near-identity regime (H=0 anchor, H=1) is reachable with a small delta.
    """

    def __init__(self, cfg: CompactHorizonJEPAPredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.horizon_embed = nn.Embedding(cfg.n_horizons, cfg.d_horizon)
        nn.init.trunc_normal_(self.horizon_embed.weight, std=0.02)

        d_in = cfg.d_latent + cfg.d_horizon
        layers: List[nn.Module] = []
        d_prev = d_in
        for _ in range(max(cfg.n_layers, 1)):
            layers += [nn.Linear(d_prev, cfg.d_hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            d_prev = cfg.d_hidden
        layers += [nn.Linear(d_prev, cfg.d_latent)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, horizon_idx: torch.Tensor) -> torch.Tensor:
        # z: (B, d_latent)   horizon_idx: (B,) long
        h = self.horizon_embed(horizon_idx)                          # (B, d_horizon)
        x = torch.cat([z, h], dim=-1)                                # (B, d_latent + d_horizon)
        out = self.mlp(x)                                            # (B, d_latent)
        if self.cfg.residual:
            out = z + out
        return out


# =============================================================================
#  EMA, loss, anti-collapse, diagnostics
# =============================================================================

@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, tau: float) -> None:
    """target <- tau * target + (1 - tau) * online (params); buffers copied."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.data.copy_(b_o.data)


def cosine_tau_schedule(epoch: int, total: int,
                        tau_start: float = 0.996, tau_end: float = 0.9995) -> float:
    if total <= 1:
        return tau_end
    e = max(1, min(epoch, total))
    progress = (e - 1) / (total - 1)
    return tau_start + (tau_end - tau_start) * 0.5 * (1.0 - math.cos(math.pi * progress))


def jepa_loss_vec(predicted: torch.Tensor, target: torch.Tensor,
                  loss_type: str = "l1") -> torch.Tensor:
    """LayerNorm-normalized distance between predicted and (detached) target
    latent vectors. Same convention as the grid trainer's jepa_loss_full,
    here over the last (d_latent) dimension."""
    pred_ln = F.layer_norm(predicted, predicted.shape[-1:])
    target_ln = F.layer_norm(target, target.shape[-1:]).detach()
    diff = pred_ln - target_ln
    if loss_type == "l1":
        return diff.abs().mean()
    elif loss_type == "l2":
        return diff.pow(2).mean()
    raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type}")


def variance_hinge(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style per-dimension variance hinge: mean relu(gamma - std(z_dim)).
    Computed but only ADDED to the loss when --lambda_var > 0."""
    std = torch.sqrt(z.var(dim=0) + eps)                             # (d,)
    return F.relu(gamma - std).mean()


@torch.no_grad()
def latent_diagnostics(z: torch.Tensor) -> Dict[str, float]:
    """Collapse diagnostics for a batch of latent states z: (B, d)."""
    std = z.std(dim=0)                                               # (d,)
    zc = z - z.mean(dim=0, keepdim=True)
    B, d = z.shape
    cov = (zc.T @ zc) / max(B - 1, 1)                                # (d, d)
    offdiag_abs = cov.abs().sum() - cov.diagonal().abs().sum()
    cov_offdiag = (offdiag_abs / max(d * d - d, 1)).item()
    try:
        s = torch.linalg.svdvals(zc)
        p = s / (s.sum() + 1e-12)
        p = p[p > 1e-12]
        eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()
    except Exception:
        eff_rank = float("nan")
    return {
        "z_std_mean": std.mean().item(),
        "z_std_min": std.min().item(),
        "z_eff_rank": eff_rank,
        "z_norm": z.norm(dim=-1).mean().item(),
        "z_cov_offdiag": cov_offdiag,
    }


# =============================================================================
#  Epoch loop
# =============================================================================

def run_epoch(
    online: CompactHorizonJEPAEncoder,
    target: CompactHorizonJEPAEncoder,
    predictor: CompactHorizonJEPAPredictor,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    tau: float,
    loss_type: str,
    grad_clip: float,
    horizons: List[int],
    lambda_var: float,
    train: bool,
) -> Dict[str, float]:
    if train:
        online.train()
        predictor.train()
    else:
        online.eval()
        predictor.eval()
    target.eval()

    nH = len(horizons)
    has_anchor = 0 in horizons
    anchor_idx = horizons.index(0) if has_anchor else -1

    sums: Dict[str, float] = {
        "L_total": 0.0, "L_jepa": 0.0, "L_var": 0.0,
        "online_norm": 0.0, "target_norm_H0": 0.0, "pred_norm_mean": 0.0,
        "cos_online_target_H0": 0.0, "gap_norm_H0": 0.0,
        "z_std_mean": 0.0, "z_std_min": 0.0, "z_eff_rank": 0.0,
        "z_norm": 0.0, "z_cov_offdiag": 0.0,
    }
    for H in horizons:
        sums[f"L_H{H}"] = 0.0
        sums[f"cos_pred_target_H{H}"] = 0.0
    n_total = 0

    for (W_t, target_windows, stock_ids) in loader:
        # W_t:            (B, K, 2, L, 2)       endpoint window (online input)
        # target_windows: (B, nH, K, 2, L, 2)   one window per target horizon
        W_t = W_t.to(device, non_blocking=True)
        target_windows = target_windows.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        B = W_t.shape[0]

        # ----- Target encoder over all horizons in one batched no-grad forward -----
        with torch.no_grad():
            W_flat = target_windows.reshape(B * nH, *target_windows.shape[2:])  # (B*nH,K,2,L,2)
            stock_flat = stock_ids.unsqueeze(1).expand(B, nH).reshape(-1)        # (B*nH,)
            target_z_flat = target(W_flat, stock_flat)                          # (B*nH, d_latent)
            target_z_all = target_z_flat.reshape(B, nH, -1)                      # (B, nH, d_latent)

        # ----- Online encoder + per-horizon predictor + loss -----
        with torch.set_grad_enabled(train):
            online_z = online(W_t, stock_ids)                                   # (B, d_latent)

            per_h_losses: List[torch.Tensor] = []
            cos_per_h: Dict[int, float] = {}
            pred_norm_sum = 0.0
            for h_idx, H in enumerate(horizons):
                horizon_idx = torch.full((B,), h_idx, device=device, dtype=torch.long)
                pred_H = predictor(online_z, horizon_idx)                       # (B, d_latent)
                target_H = target_z_all[:, h_idx]                               # (B, d_latent)
                per_h_losses.append(jepa_loss_vec(pred_H, target_H, loss_type))
                with torch.no_grad():
                    cos_per_h[H] = F.cosine_similarity(pred_H, target_H, dim=-1).mean().item()
                    pred_norm_sum += pred_H.norm(dim=-1).mean().item()

            L_jepa = torch.stack(per_h_losses).mean()
            L_var = variance_hinge(online_z)                # always computed (monitoring)
            L_total = L_jepa + (lambda_var * L_var if lambda_var > 0 else 0.0 * L_var)

        if train:
            optimizer.zero_grad(set_to_none=True)
            L_total.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(online.parameters()) + list(predictor.parameters()), grad_clip,
                )
            optimizer.step()
            update_ema(target, online, tau=tau)

        # ----- Diagnostics -----
        with torch.no_grad():
            if has_anchor:
                t_z_H0 = target_z_all[:, anchor_idx]                            # (B, d_latent)
                cos_OT_H0 = F.cosine_similarity(online_z, t_z_H0, dim=-1).mean().item()
                gap_norm_H0 = (online_z - t_z_H0).norm(dim=-1).mean().item()
                target_norm_H0 = t_z_H0.norm(dim=-1).mean().item()
            else:
                cos_OT_H0 = gap_norm_H0 = target_norm_H0 = float("nan")
            online_norm = online_z.norm(dim=-1).mean().item()
            ld = latent_diagnostics(online_z)

        # ----- Accumulate (B-weighted) -----
        sums["L_total"] += float(L_total.item()) * B
        sums["L_jepa"] += float(L_jepa.item()) * B
        sums["L_var"] += float(L_var.item()) * B
        sums["online_norm"] += online_norm * B
        sums["pred_norm_mean"] += (pred_norm_sum / nH) * B
        for k, v in ld.items():
            sums[k] += v * B
        if has_anchor:
            sums["target_norm_H0"] += target_norm_H0 * B
            sums["cos_online_target_H0"] += cos_OT_H0 * B
            sums["gap_norm_H0"] += gap_norm_H0 * B
        for h_idx, H in enumerate(horizons):
            sums[f"L_H{H}"] += float(per_h_losses[h_idx].item()) * B
            sums[f"cos_pred_target_H{H}"] += cos_per_h[H] * B
        n_total += B

    out = {k: v / max(n_total, 1) for k, v in sums.items()}
    out["n"] = n_total
    if not has_anchor:
        out["target_norm_H0"] = out["cos_online_target_H0"] = out["gap_norm_H0"] = float("nan")
    return out


# =============================================================================
#  Checkpointing / loading
# =============================================================================

MODEL_TYPE = "compact_horizon_jepa"


def save_checkpoint(
    path: Path,
    epoch: int,
    online: CompactHorizonJEPAEncoder,
    target: CompactHorizonJEPAEncoder,
    predictor: CompactHorizonJEPAPredictor,
    optimizer: torch.optim.Optimizer,
    enc_cfg: CompactHorizonJEPAEncoderConfig,
    pred_cfg: CompactHorizonJEPAPredictorConfig,
    train_args: dict,
    stock_stats: dict,
    horizons: List[int],
    val_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_type": MODEL_TYPE,
        "format_version": "compact_horizon_jepa_v1",
        "epoch": epoch,
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "enc_cfg": enc_cfg.to_dict(),
        "pred_cfg": pred_cfg.to_dict(),
        "train_args": train_args,
        "stock_stats": {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in stock_stats.items()},
        "horizons": list(horizons),
        "val_metrics": val_metrics,
    }
    torch.save(state, path)


def load_compact_horizon_jepa_encoder(ckpt_path: str, device: torch.device,
                                      which: str = "online"):
    """Load a CompactHorizonJEPAEncoder from a checkpoint. Dispatches on
    model_type so grid-JEPA checkpoints are rejected with a clear message
    (use the grid loader for those). `which` in {"online", "target"}."""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    mt = ckpt.get("model_type", None)
    if mt != MODEL_TYPE:
        raise ValueError(
            f"Checkpoint model_type={mt!r}; expected {MODEL_TYPE!r}. "
            f"This is not a Compact Horizon JEPA checkpoint."
        )
    enc_cfg = CompactHorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    enc = CompactHorizonJEPAEncoder(enc_cfg).to(device)
    key = "online_state_dict" if which == "online" else "target_state_dict"
    enc.load_state_dict(ckpt[key])
    enc.eval()
    return enc, ckpt


# =============================================================================
#  Smoke test (no dataset required)
# =============================================================================

def smoke_test() -> None:
    print("=" * 80)
    print("Compact Horizon JEPA — SMOKE TEST")
    print("=" * 80)
    if LOBSpatialEncoder is None:
        raise SystemExit("Run from the project root: model_tokenizer_t not importable.")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [0, 1, 5, 10, 20]
    nH = len(horizons)
    B, K, L = 8, 20, 10

    enc_cfg = CompactHorizonJEPAEncoderConfig(L=L, n_stocks=7, K=K, d_model=32, d_latent=16)
    pred_cfg = CompactHorizonJEPAPredictorConfig(d_latent=16, n_horizons=nH,
                                                 d_horizon=16, d_hidden=64, n_layers=2)

    online = CompactHorizonJEPAEncoder(enc_cfg).to(device)
    target = CompactHorizonJEPAEncoder(enc_cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad = False
    predictor = CompactHorizonJEPAPredictor(pred_cfg).to(device)

    book = torch.randn(B, K, 2, L, 2, device=device)
    target_windows = torch.randn(B, nH, K, 2, L, 2, device=device)
    stock_ids = torch.randint(0, 7, (B,), device=device)

    z = online(book, stock_ids)
    assert z.shape == (B, 16), f"encoder output {z.shape}, expected {(B, 16)}"
    print(f"  encoder OK     : W_t {tuple(book.shape)} -> z {tuple(z.shape)}")

    W_flat = target_windows.reshape(B * nH, K, 2, L, 2)
    stock_flat = stock_ids.unsqueeze(1).expand(B, nH).reshape(-1)
    tz = target(W_flat, stock_flat).reshape(B, nH, -1)
    assert tz.shape == (B, nH, 16)
    print(f"  target OK      : batched over {nH} horizons -> {tuple(tz.shape)}")

    for h_idx, H in enumerate(horizons):
        hi = torch.full((B,), h_idx, device=device, dtype=torch.long)
        pred = predictor(z, hi)
        assert pred.shape == (B, 16), f"predictor H={H} -> {pred.shape}"
    print(f"  predictor OK   : (z, H) -> z_hat for H in {horizons}")

    loss = jepa_loss_vec(predictor(z, torch.zeros(B, dtype=torch.long, device=device)),
                         tz[:, 0])
    lvar = variance_hinge(z)
    diag = latent_diagnostics(z)
    print(f"  loss OK        : jepa_loss_vec={loss.item():.4f}  variance_hinge={lvar.item():.4f}")
    print(f"  diagnostics OK : {', '.join(f'{k}={v:.3f}' for k, v in diag.items())}")

    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=1e-3)
    m = run_epoch(online, target, predictor,
                  loader=[(book.cpu(), target_windows.cpu(), stock_ids.cpu())],
                  optimizer=opt, device=device, tau=0.99, loss_type="l1",
                  grad_clip=1.0, horizons=horizons, lambda_var=0.0, train=True)
    print(f"  run_epoch OK   : L_total={m['L_total']:.4f}  "
          f"z_eff_rank={m['z_eff_rank']:.2f}  z_std_min={m['z_std_min']:.3f}")
    n_online = sum(p.numel() for p in online.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"  params         : online={n_online:,}  predictor={n_pred:,}")
    print("\nSmoke test passed.")


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compact Horizon JEPA: state-based multi-horizon JEPA")

    p.add_argument("--smoke_test", action="store_true", help="Run smoke test and exit")

    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--ckpt_dir", type=str, default=None)

    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eta_min_frac", type=float, default=0.01)

    # Encoder (A1-T-like)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_latent", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--spatial_n_layers", type=int, default=2)
    p.add_argument("--spatial_n_heads", type=int, default=4)
    p.add_argument("--spatial_d_ffn", type=int, default=256)
    p.add_argument("--temporal_n_layers", type=int, default=2)
    p.add_argument("--temporal_n_heads", type=int, default=4)
    p.add_argument("--temporal_d_ffn", type=int, default=256)

    # Predictor (state-to-state)
    p.add_argument("--pred_d_hidden", type=int, default=256)
    p.add_argument("--pred_n_layers", type=int, default=2)
    p.add_argument("--pred_d_horizon", type=int, default=32)
    p.add_argument("--pred_dropout", type=float, default=0.0)
    p.add_argument("--pred_residual", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--horizons", type=str, default="0,1,5,10,20")
    p.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument("--tau_start", type=float, default=0.996)
    p.add_argument("--tau_end", type=float, default=0.9995)

    # Anti-collapse: pre-wired, OFF by default. The main run is pure JEPA.
    p.add_argument("--lambda_var", type=float, default=0.0,
                   help="Weight of the VICReg-style variance hinge. 0.0 = pure JEPA "
                        "(main experiment). >0 = anti-collapse contingency only.")

    p.add_argument("--save_every", type=int, default=5)
    args = p.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.dataset is None or args.ckpt_dir is None:
        raise SystemExit("--dataset and --ckpt_dir are required unless --smoke_test")

    _check_project_imports_or_exit()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    horizons = parse_horizons(args.horizons)
    max_horizon = max(horizons)
    nH = len(horizons)

    print("=" * 92)
    print("COMPACT HORIZON JEPA — state-based multi-horizon JEPA")
    print("=" * 92)
    print(f"device       : {device}")
    print(f"dataset      : {args.dataset}")
    print(f"horizons     : {horizons}  (nH={nH}, max_horizon={max_horizon}, "
          f"anchor H=0 {'present' if 0 in horizons else 'ABSENT'})")
    if args.lambda_var > 0:
        print("-" * 92)
        print(f"  WARNING: --lambda_var={args.lambda_var} > 0  -> variance hinge is ACTIVE.")
        print("  This run is NOT the pure compact JEPA. Use only as anti-collapse contingency.")
        print("-" * 92)
    else:
        print("loss         : pure JEPA (lambda_var=0.0; no VICReg/var-cov term)")

    print("\n[1/9] Loading raw LOBench dataset...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    n_stocks = int(raw["min_spread_z_per_stock"].shape[0]) if "min_spread_z_per_stock" in raw.files \
        else int(stock_ids.max() + 1)
    N, L = len(mid_z), book.shape[2]
    print(f"  N={N:,}  n_stocks={n_stocks}  L={L}")

    print("\n[2/9] vol_clip mask...")
    bid_v, ask_v = book[:, 0, :, 1], book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    print(f"  pass: {vol_mask.sum():,}/{N:,} ({100 * vol_mask.sum() / N:.2f}%)")

    enc_cfg = CompactHorizonJEPAEncoderConfig(
        L=L, n_stocks=n_stocks, K=args.K,
        d_model=args.d_model, d_latent=args.d_latent,
        spatial_n_layers=args.spatial_n_layers, spatial_n_heads=args.spatial_n_heads,
        spatial_d_ffn=args.spatial_d_ffn,
        temporal_n_layers=args.temporal_n_layers, temporal_n_heads=args.temporal_n_heads,
        temporal_d_ffn=args.temporal_d_ffn, dropout=args.dropout,
    )
    pred_cfg = CompactHorizonJEPAPredictorConfig(
        d_latent=args.d_latent, n_horizons=nH, d_horizon=args.pred_d_horizon,
        d_hidden=args.pred_d_hidden, n_layers=args.pred_n_layers,
        dropout=args.pred_dropout, residual=args.pred_residual,
    )
    assert enc_cfg.d_latent == pred_cfg.d_latent, "encoder/predictor d_latent mismatch"

    print("\n[3/9] Config:")
    print(f"  encoder  : d_model={enc_cfg.d_model} d_latent={enc_cfg.d_latent} "
          f"spatial(L{enc_cfg.spatial_n_layers},H{enc_cfg.spatial_n_heads}) "
          f"temporal(L{enc_cfg.temporal_n_layers},H{enc_cfg.temporal_n_heads}, causal)")
    print(f"  predictor: residual={pred_cfg.residual} d_hidden={pred_cfg.d_hidden} "
          f"n_layers={pred_cfg.n_layers} d_horizon={pred_cfg.d_horizon}")
    print(f"  loss={args.loss_type}  tau {args.tau_start}->{args.tau_end}  lambda_var={args.lambda_var}")

    print("\n[4/9] Valid endpoints...")
    valid_t = compute_valid_endpoints(stock_ids, day_ids, args.K, max_horizon, vol_mask)
    print(f"  valid_t: {len(valid_t):,}")

    print("\n[5/9] Grouped split by (stock, day)...")
    train_pos, val_pos = grouped_split_by_stock_day(stock_ids, day_ids, valid_t, args.val_frac, args.seed)
    if args.max_train_samples > 0 and len(train_pos) > args.max_train_samples:
        train_pos = np.sort(np.random.default_rng(args.seed).choice(
            train_pos, size=args.max_train_samples, replace=False))
    if args.max_val_samples > 0 and len(val_pos) > args.max_val_samples:
        val_pos = np.sort(np.random.default_rng(args.seed + 1).choice(
            val_pos, size=args.max_val_samples, replace=False))
    valid_t_train, valid_t_val = valid_t[train_pos], valid_t[val_pos]
    print(f"  train endpoints: {len(valid_t_train):,}   val endpoints: {len(valid_t_val):,}")

    print("\n[6/9] Per-stock normalization stats (TRAIN-only)...")
    stock_stats = compute_stock_stats_train_only(book, mid_z, stock_ids, day_ids, valid_t_train, n_stocks)

    print("\n[7/9] Datasets (HorizonJEPADataset, unchanged)...")
    ds_train = HorizonJEPADataset(book, mid_z, stock_ids, valid_t_train, stock_stats, args.K, horizons)
    ds_val = HorizonJEPADataset(book, mid_z, stock_ids, valid_t_val, stock_stats, args.K, horizons)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                          persistent_workers=args.num_workers > 0, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0, drop_last=False)

    print("\n[8/9] Models...")
    online = CompactHorizonJEPAEncoder(enc_cfg).to(device)
    target = CompactHorizonJEPAEncoder(enc_cfg).to(device)
    target.load_state_dict(online.state_dict())
    for prm in target.parameters():
        prm.requires_grad = False
    predictor = CompactHorizonJEPAPredictor(pred_cfg).to(device)
    n_online = sum(x.numel() for x in online.parameters())
    n_pred = sum(x.numel() for x in predictor.parameters())
    print(f"  online params={n_online:,}  predictor params={n_pred:,}  "
          f"total trainable={n_online + n_pred:,}")

    optimizer = torch.optim.AdamW(
        list(online.parameters()) + list(predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * args.eta_min_frac)

    print("\n[9/9] Training...")
    print("  Reading per-H losses: H=1 overlaps W_t in 19/20 timesteps (near-identity),")
    print("  H=20 has zero overlap (pure extrapolation). Watch L_H20 / cos_H20 trajectory.")
    print("  Collapse watch: z_std_min -> 0, z_eff_rank dropping, or z_cov_offdiag rising.")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_val = float("inf")

    def _per_h(m: Dict[str, float], tmpl: str) -> str:
        return "  ".join(f"H{H:>2d}={m[tmpl.format(H=H)]:.4f}" for H in horizons)

    for epoch in range(1, args.epochs + 1):
        tau = cosine_tau_schedule(epoch, args.epochs, args.tau_start, args.tau_end)
        t0 = time.time()
        tr = run_epoch(online, target, predictor, dl_train, optimizer, device,
                       tau=tau, loss_type=args.loss_type, grad_clip=args.grad_clip,
                       horizons=horizons, lambda_var=args.lambda_var, train=True)
        va = run_epoch(online, target, predictor, dl_val, None, device,
                       tau=tau, loss_type=args.loss_type, grad_clip=args.grad_clip,
                       horizons=horizons, lambda_var=args.lambda_var, train=False)
        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        for tag, m in (("train", tr), ("val", va)):
            print(
                f"epoch {epoch:3d}/{args.epochs}  [{tag}]  "
                f"tau={tau:.4f} lr={lr_now:.2e}  [{dt:.1f}s]\n"
                f"  L_total={m['L_total']:.4f}  L_jepa={m['L_jepa']:.4f}  "
                f"L_var={m['L_var']:.4f}\n"
                f"  L   : {_per_h(m, 'L_H{H}')}\n"
                f"  cos : {_per_h(m, 'cos_pred_target_H{H}')}\n"
                f"  anchor H0: cos_OT={m['cos_online_target_H0']:+.3f}  "
                f"gap_norm={m['gap_norm_H0']:.4f}\n"
                f"  latent: z_std_mean={m['z_std_mean']:.3f}  z_std_min={m['z_std_min']:.3f}  "
                f"z_eff_rank={m['z_eff_rank']:.2f}  z_norm={m['z_norm']:.2f}  "
                f"cov_offdiag={m['z_cov_offdiag']:.4f}  pred_norm={m['pred_norm_mean']:.2f}"
            )

        history.append({"epoch": epoch, "tau": tau, "lr": lr_now, "train": tr, "val": va})

        if va["L_total"] < best_val:
            best_val = va["L_total"]
            save_checkpoint(ckpt_dir / "best.pt", epoch, online, target, predictor,
                            optimizer, enc_cfg, pred_cfg, vars(args), stock_stats, horizons, va)
            print(f"  -> saved best.pt (val L_total={best_val:.4f})")
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", epoch, online, target, predictor,
                            optimizer, enc_cfg, pred_cfg, vars(args), stock_stats, horizons, va)
            print(f"  -> saved epoch_{epoch:03d}.pt")
        save_checkpoint(ckpt_dir / "last.pt", epoch, online, target, predictor,
                        optimizer, enc_cfg, pred_cfg, vars(args), stock_stats, horizons, va)

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. History: {ckpt_dir / 'history.json'}")


if __name__ == "__main__":
    main()

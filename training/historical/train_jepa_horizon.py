"""
train_jepa_horizon.py — Horizon JEPA: horizon-conditioned JEPA latent extrapolation.

Purpose
=======
Self-supervised pretraining of an LOB encoder via multi-horizon latent
extrapolation. Derived from Masked JEPA Stage 1, but reframes the task:

    Masked JEPA (predecessor):
        intra-window masked completion
        online encoder sees masked W_t, predicts target encoder's W_t at masked positions

    Horizon JEPA (this file):
        cross-window latent extrapolation
        online encoder sees full W_t, predicts target encoder's W_{t+H}
        H is conditioned via a learnable horizon embedding into the predictor

Horizon set matches F0 (supervised tokenizer A1-T) for direct comparison:
    H ∈ {0, 1, 5, 10, 20}    (configurable via --horizons)

H=0 is an explicit anchor: predictor(online(W_t), H=0) ≈ target(W_t).
This anchors the coordinate system between online and target encoders and
mitigates the coordinate-gap failure mode observed in F1-v1.

Architecture
============
- Encoder: identical to Masked JEPA (token-preserving dual-axis, S=4 spatial regions
  × K=20 timesteps). Stage 2 interface unchanged: z_t comes from online(W_t).
- Predictor: Masked JEPA predictor + learnable horizon embedding summed into the
  predictor input after down-projection. Shared weights across horizons.
- Loss: per-horizon L1/L2 on LayerNorm-normalized target embeddings, full grid.
  L_total = mean over horizons.

Implementation notes
====================
- Target encoder is run batched across horizons: stack (B, nH, K, 2, L, 2)
  into (B·nH, K, 2, L, 2) and forward once under no_grad.
- Per-horizon diagnostics: L_H, cos_pred_target_H for each H.
- Anchor diagnostics: cos(online(W_t), target(W_t)) and gap_norm. These reveal
  whether H=0 is doing its job as a coordinate anchor.
- The Masked JEPA masking infrastructure (mask_token in encoder, etc.) is kept but
  unused at training time — kept for ablation symmetry only.

Reading per-horizon losses
==========================
L_H1 is naturally lower than L_H20 not because H=1 is "easier" in some abstract
sense, but because the t+1 K-window overlaps with the t K-window in 19 of 20
timesteps (near-identity shift); H=20 has zero overlap and is pure extrapolation.
Watch the *trajectory* of L_H20 across epochs, and the per-horizon cos
similarity, rather than the absolute L_H values across horizons.
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Line-buffered stdout (visible in real time without stdbuf).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# =============================================================================
#  Project imports
# =============================================================================

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, _THIS.parent.parent, _THIS.parent.parent.parent, _THIS.parent.parent.parent.parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from training.train_tokenizer_t import (  # type: ignore
        compute_valid_endpoints,
        normalize_book_window,
        compute_stock_stats,
        grouped_split_by_stock_day,
        compute_stock_stats_train_only,
    )
except Exception:
    compute_valid_endpoints = None
    normalize_book_window = None
    compute_stock_stats = None
    grouped_split_by_stock_day = None
    compute_stock_stats_train_only = None


def _check_project_imports_or_exit() -> None:
    missing = []
    if compute_valid_endpoints is None or normalize_book_window is None or compute_stock_stats is None:
        missing.append("train_tokenizer_t")
    if grouped_split_by_stock_day is None or compute_stock_stats_train_only is None:
        missing.append("train_tokenizer_t")
    if missing:
        raise SystemExit(
            f"Cannot import project modules: {missing}. "
            f"Run from the project root or add it to PYTHONPATH."
        )


# =============================================================================
#  Config dataclasses
# =============================================================================

@dataclass
class HorizonJEPAEncoderConfig:
    """JEPA Stage 1 encoder config. Identical structure to Masked JEPA."""

    L: int = 10
    n_stocks: int = 7
    K: int = 20
    S: int = 4
    raw_per_token: int = 10

    d_model: int = 128
    d_latent: int = 32

    spatial_n_layers: int = 2
    spatial_n_heads: int = 4
    spatial_d_ffn: int = 256

    temporal_n_layers: int = 2
    temporal_n_heads: int = 4
    temporal_d_ffn: int = 256
    temporal_causal: bool = False

    dropout: float = 0.1
    stock_emb_init_scale: float = 0.02

    @classmethod
    def from_dict(cls, d: Dict) -> "HorizonJEPAEncoderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HorizonJEPAPredictorConfig:
    """Horizon-conditioned predictor. Adds n_horizons vs Masked JEPA."""

    K: int = 20
    S: int = 4
    d_in: int = 128
    d_pred: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_ffn: int = 128
    dropout: float = 0.0
    n_horizons: int = 5

    @classmethod
    def from_dict(cls, d: Dict) -> "HorizonJEPAPredictorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
#  Structured Tokenizer (unchanged from Masked JEPA)
# =============================================================================

TOKEN_NAMES = ("bid_top", "bid_deep", "ask_top", "ask_deep")
S_DEFAULT = 4


def split_window_into_regions(book: torch.Tensor) -> torch.Tensor:
    """Split LOB window into 4 spatio-region tokens per timestep.

    Args:
        book: (B, K, 2, L, 2)  side ∈ {0=bid, 1=ask}, feat ∈ {0=price, 1=volume}.

    Returns:
        regions: (B, K, 4, 10)  ordered (bid_top, bid_deep, ask_top, ask_deep).
    """
    B, K, side_dim, L, feat_dim = book.shape
    if side_dim != 2:
        raise ValueError(f"Expected side dim = 2, got {side_dim}")
    if feat_dim != 2:
        raise ValueError(f"Expected feature dim = 2, got {feat_dim}")
    half = L // 2
    if L != 2 * half:
        raise ValueError(f"L must be even to split into top/deep, got L={L}")

    bid_top = book[:, :, 0, :half, :].reshape(B, K, half * feat_dim)
    bid_deep = book[:, :, 0, half:, :].reshape(B, K, half * feat_dim)
    ask_top = book[:, :, 1, :half, :].reshape(B, K, half * feat_dim)
    ask_deep = book[:, :, 1, half:, :].reshape(B, K, half * feat_dim)
    return torch.stack([bid_top, bid_deep, ask_top, ask_deep], dim=2)


class StructuredTokenizer(nn.Module):
    """Raw LOB window -> token embeddings with time/spatial/stock bias."""

    def __init__(self, cfg: HorizonJEPAEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.raw_per_token, cfg.d_model)

        self.time_pos = nn.Parameter(torch.zeros(cfg.K, cfg.d_model))
        self.spatial_pos = nn.Parameter(torch.zeros(cfg.S, cfg.d_model))
        self.stock_emb = nn.Embedding(cfg.n_stocks, cfg.d_model)

        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.stock_emb.weight, std=cfg.stock_emb_init_scale)

    def content_and_bias(self, book: torch.Tensor, stock_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        regions = split_window_into_regions(book)            # (B, K, S, raw)
        content = self.proj(regions)                          # (B, K, S, d)
        bias = (
            self.time_pos.view(1, self.cfg.K, 1, self.cfg.d_model)
            + self.spatial_pos.view(1, 1, self.cfg.S, self.cfg.d_model)
            + self.stock_emb(stock_ids).view(-1, 1, 1, self.cfg.d_model)
        )
        return content, bias

    def forward(self, book: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        content, bias = self.content_and_bias(book, stock_ids)
        return content + bias


# =============================================================================
#  Encoder (masking path kept for symmetry but unused at Horizon JEPA training)
# =============================================================================

class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(h)
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


def _build_causal_2d_mask(K: int, S: int, device: torch.device) -> torch.Tensor:
    time_idx = torch.arange(K, device=device).unsqueeze(1).expand(K, S).reshape(-1)
    return time_idx.unsqueeze(1) < time_idx.unsqueeze(0)


class HorizonJEPAEncoder(nn.Module):
    """Token-preserving dual-axis encoder.

    Pipeline:
        raw window -> StructuredTokenizer -> (B, K, S, d)
        optional content masking (kept for ablation symmetry, unused in Horizon JEPA train)
        spatial attention per timestep
        temporal attention over K·S tokens (optionally causal)
        output (B, K, S, d)
    """

    def __init__(self, cfg: HorizonJEPAEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = StructuredTokenizer(cfg)

        # Mask token kept for Masked JEPA-style ablation; not used in Horizon JEPA training.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.spatial_blocks = nn.ModuleList([
            _TransformerBlock(cfg.d_model, cfg.spatial_n_heads, cfg.spatial_d_ffn, cfg.dropout)
            for _ in range(cfg.spatial_n_layers)
        ])
        self.temporal_blocks = nn.ModuleList([
            _TransformerBlock(cfg.d_model, cfg.temporal_n_heads, cfg.temporal_d_ffn, cfg.dropout)
            for _ in range(cfg.temporal_n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)

        if cfg.temporal_causal:
            self.register_buffer(
                "_causal_mask",
                _build_causal_2d_mask(cfg.K, cfg.S, torch.device("cpu")),
                persistent=False,
            )

    def forward(
        self,
        book: torch.Tensor,
        stock_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        B = book.shape[0]

        content, bias = self.tokenizer.content_and_bias(book, stock_ids)
        if mask is not None:
            # Masked JEPA masking path (kept for compatibility; not invoked in Horizon JEPA)
            x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(content), content)
        else:
            x = content
        x = x + bias

        # Spatial attention per timestep
        x = x.reshape(B * cfg.K, cfg.S, cfg.d_model)
        for blk in self.spatial_blocks:
            x = blk(x, attn_mask=None)
        x = x.reshape(B, cfg.K, cfg.S, cfg.d_model)

        # Temporal / spatio-temporal attention
        x = x.reshape(B, cfg.K * cfg.S, cfg.d_model)
        causal_mask = None
        if cfg.temporal_causal:
            causal_mask = self._causal_mask.to(x.device)
        for blk in self.temporal_blocks:
            x = blk(x, attn_mask=causal_mask)

        x = self.final_norm(x)
        return x.reshape(B, cfg.K, cfg.S, cfg.d_model)


@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, tau: float) -> None:
    """EMA update: target <- tau * target + (1 - tau) * online."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.data.copy_(b_o.data)


def cosine_tau_schedule(epoch: int, total: int, tau_start: float = 0.996, tau_end: float = 0.9995) -> float:
    if total <= 1:
        return tau_end
    e = max(1, min(epoch, total))
    progress = (e - 1) / (total - 1)
    return tau_start + (tau_end - tau_start) * 0.5 * (1.0 - math.cos(math.pi * progress))


# =============================================================================
#  Horizon-conditioned Predictor
# =============================================================================

class HorizonJEPAPredictor(nn.Module):
    """Narrow transformer predictor with horizon-conditioned input.

    Inputs:
        online_output: (B, K, S, d_in)
        horizon_idx:   (B,) long, indexing into self.horizon_embed (0 .. n_horizons-1)

    Output:
        predicted:     (B, K, S, d_in)

    The horizon embedding is summed (broadcast over K, S) into the post-down-proj
    representation, alongside time and spatial position embeddings.
    """

    def __init__(self, cfg: HorizonJEPAPredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.down = nn.Linear(cfg.d_in, cfg.d_pred)
        self.up = nn.Linear(cfg.d_pred, cfg.d_in)
        self.norm_in = nn.LayerNorm(cfg.d_pred)
        self.norm_out = nn.LayerNorm(cfg.d_in)

        self.pred_time_pos = nn.Parameter(torch.zeros(cfg.K, cfg.d_pred))
        self.pred_spatial_pos = nn.Parameter(torch.zeros(cfg.S, cfg.d_pred))
        self.horizon_embed = nn.Embedding(cfg.n_horizons, cfg.d_pred)

        nn.init.trunc_normal_(self.pred_time_pos, std=0.02)
        nn.init.trunc_normal_(self.pred_spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.horizon_embed.weight, std=0.02)

        self.blocks = nn.ModuleList([
            _TransformerBlock(cfg.d_pred, cfg.n_heads, cfg.d_ffn, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

    def forward(self, online_output: torch.Tensor, horizon_idx: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = online_output.shape[0]
        x = self.down(online_output)                                # (B, K, S, d_pred)

        pos = (
            self.pred_time_pos.view(1, cfg.K, 1, cfg.d_pred)
            + self.pred_spatial_pos.view(1, 1, cfg.S, cfg.d_pred)
        )
        h_emb = self.horizon_embed(horizon_idx).view(B, 1, 1, cfg.d_pred)  # broadcast over K, S
        x = self.norm_in(x + pos + h_emb)

        x = x.reshape(B, cfg.K * cfg.S, cfg.d_pred)
        for blk in self.blocks:
            x = blk(x, attn_mask=None)
        x = x.reshape(B, cfg.K, cfg.S, cfg.d_pred)

        return self.norm_out(self.up(x))


# =============================================================================
#  Dataset
# =============================================================================

class HorizonJEPADataset(Dataset):
    """Returns the endpoint window W_t plus multi-horizon target windows.

    For each valid endpoint t:
        W_t            = normalized K-window ending at t            (online input)
        target_windows = normalized K-windows ending at (t + h_i)   for h_i in horizons

    W_t is ALWAYS the endpoint window, decoupled from the horizon list. This
    allows the no-anchor ablation (horizons without 0): the online encoder
    still sees W_t, while the predictor is trained only on the requested
    horizons. When 0 is in horizons, the H=0 target window coincides with W_t
    (recomputed for code simplicity; the redundancy is one extra window).

    __getitem__ returns (W_t, target_windows, stock_id) with shapes
        W_t:            (K, 2, L, 2)
        target_windows: (nH, K, 2, L, 2)
    """

    def __init__(
        self,
        book: np.ndarray,
        mid_z: np.ndarray,
        stock_ids: np.ndarray,
        valid_t: np.ndarray,
        stock_stats: Dict[str, np.ndarray],
        K: int,
        horizons: List[int],
    ):
        if not horizons:
            raise ValueError("Horizons list is empty.")
        if any(h < 0 for h in horizons):
            raise ValueError(f"Horizons must be non-negative; got {horizons}")
        self.book = book
        self.mid_z = mid_z
        self.stock_ids = stock_ids
        self.valid_t = valid_t
        self.stock_stats = stock_stats
        self.K = K
        self.horizons = list(horizons)
        self.n_horizons = len(self.horizons)
        self.L = book.shape[2]

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, idx: int):
        t = int(self.valid_t[idx])
        s = int(self.stock_ids[t])
        K = self.K

        # W_t: endpoint window (offset 0). Always supplied for the online encoder,
        # independently of whether 0 is among the target horizons.
        w_t = normalize_book_window(
            self.book[t - K + 1 : t + 1],
            self.mid_z[t - K + 1 : t + 1],
            s, self.stock_stats,
        )

        # Target windows: one per requested horizon.
        target_windows = np.empty((self.n_horizons, K, 2, self.L, 2), dtype=np.float32)
        for h_idx, H in enumerate(self.horizons):
            start = t - K + 1 + H
            stop = t + 1 + H
            book_win = self.book[start:stop]
            mid_win = self.mid_z[start:stop]
            target_windows[h_idx] = normalize_book_window(book_win, mid_win, s, self.stock_stats)

        return (
            torch.from_numpy(w_t).float(),
            torch.from_numpy(target_windows).float(),
            torch.tensor(s, dtype=torch.long),
        )


# =============================================================================
#  Loss and diagnostics
# =============================================================================

def jepa_loss_full(predicted: torch.Tensor, target: torch.Tensor, loss_type: str = "l1") -> torch.Tensor:
    """JEPA loss over all (K, S) positions, LayerNorm-normalized."""
    pred_ln = F.layer_norm(predicted, predicted.shape[-1:])
    target_ln = F.layer_norm(target, target.shape[-1:]).detach()
    diff = pred_ln - target_ln
    if loss_type == "l1":
        return diff.abs().mean()
    elif loss_type == "l2":
        return diff.pow(2).mean()
    else:
        raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type}")


@torch.no_grad()
def cosine_token_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity over all per-token (B, K, S) positions."""
    return F.cosine_similarity(a, b, dim=-1).mean().item()


@torch.no_grad()
def pooled_diagnostics(online_out: torch.Tensor) -> Tuple[float, float]:
    """Spatial-mean-pool last timestep; return (per-dim std mean, effective rank)."""
    pooled = online_out[:, -1, :, :].mean(dim=1)              # (B, d)
    pooled_std = pooled.std(dim=0).mean().item()
    pooled_c = pooled - pooled.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(pooled_c)
        p = s / (s.sum() + 1e-12)
        p = p[p > 1e-12]
        eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()
    except Exception:
        eff_rank = float("nan")
    return pooled_std, eff_rank


# =============================================================================
#  Epoch loop
# =============================================================================

def run_epoch(
    online: HorizonJEPAEncoder,
    target: HorizonJEPAEncoder,
    predictor: HorizonJEPAPredictor,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    tau: float,
    loss_type: str,
    grad_clip: float,
    horizons: List[int],
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
        "L_total": 0.0,
        "online_norm": 0.0,
        "target_norm_H0": 0.0,
        "pred_norm_mean": 0.0,
        "cos_online_target_H0": 0.0,
        "gap_norm_H0": 0.0,
        "pooled_std": 0.0,
        "pooled_eff_rank": 0.0,
    }
    for H in horizons:
        sums[f"L_H{H}"] = 0.0
        sums[f"cos_pred_target_H{H}"] = 0.0
    n_total = 0

    for batch_idx, (W_t, target_windows, stock_ids) in enumerate(loader):
        # W_t:            (B, K, 2, L, 2)      endpoint window for the online encoder
        # target_windows: (B, nH, K, 2, L, 2)  one window per target horizon
        W_t = W_t.to(device, non_blocking=True)
        target_windows = target_windows.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        B = W_t.shape[0]

        # ----- Target encoder on all horizons in one batched no-grad forward -----
        with torch.no_grad():
            W_flat = target_windows.reshape(B * nH, *target_windows.shape[2:])  # (B·nH, K, 2, L, 2)
            stock_flat = stock_ids.unsqueeze(1).expand(B, nH).reshape(-1)        # (B·nH,)
            target_out_flat = target(W_flat, stock_flat, mask=None)              # (B·nH, K, S, d)
            target_out_all = target_out_flat.reshape(B, nH, *target_out_flat.shape[1:])  # (B, nH, K, S, d)

        # ----- Per-horizon predictor + loss -----
        with torch.set_grad_enabled(train):
            online_out = online(W_t, stock_ids, mask=None)          # (B, K, S, d)

            per_h_losses: List[torch.Tensor] = []
            cos_per_h: Dict[int, float] = {}
            pred_norm_sum = 0.0
            for h_idx, H in enumerate(horizons):
                horizon_idx = torch.full((B,), h_idx, device=device, dtype=torch.long)
                pred_H = predictor(online_out, horizon_idx)         # (B, K, S, d)
                target_H = target_out_all[:, h_idx]                  # (B, K, S, d)
                L_H = jepa_loss_full(pred_H, target_H, loss_type=loss_type)
                per_h_losses.append(L_H)
                with torch.no_grad():
                    cos_per_h[H] = cosine_token_mean(pred_H, target_H)
                    pred_norm_sum += pred_H.norm(dim=-1).mean().item()

            L_total = torch.stack(per_h_losses).mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            L_total.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(online.parameters()) + list(predictor.parameters()),
                    grad_clip,
                )
            optimizer.step()
            update_ema(target, online, tau=tau)

        # ----- Diagnostics (anchor + pooled rank) -----
        # Anchor diagnostics (cos/gap online-vs-target at H=0) only exist when
        # 0 is in the horizon list. In the no-anchor ablation they are NaN.
        with torch.no_grad():
            if has_anchor:
                target_out_H0 = target_out_all[:, anchor_idx]       # (B, K, S, d)
                cos_OT_H0 = cosine_token_mean(online_out, target_out_H0)
                gap_norm_H0 = (online_out - target_out_H0).norm(dim=-1).mean().item()
                target_norm_H0 = target_out_H0.norm(dim=-1).mean().item()
            else:
                cos_OT_H0 = float("nan")
                gap_norm_H0 = float("nan")
                target_norm_H0 = float("nan")
            online_norm = online_out.norm(dim=-1).mean().item()
            pooled_std, eff_rank = pooled_diagnostics(online_out)

        # ----- Accumulate (B-weighted) -----
        sums["L_total"] += float(L_total.item()) * B
        sums["online_norm"] += online_norm * B
        sums["pred_norm_mean"] += (pred_norm_sum / nH) * B
        sums["pooled_std"] += pooled_std * B
        sums["pooled_eff_rank"] += eff_rank * B
        # Anchor metrics: accumulate only when the anchor exists; otherwise the
        # keys stay at 0.0 and are forced to NaN after the loop.
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
    # No-anchor ablation: anchor diagnostics are undefined, report as NaN.
    if not has_anchor:
        out["target_norm_H0"] = float("nan")
        out["cos_online_target_H0"] = float("nan")
        out["gap_norm_H0"] = float("nan")
    return out


# =============================================================================
#  Checkpointing
# =============================================================================

def save_checkpoint(
    path: Path,
    epoch: int,
    online: HorizonJEPAEncoder,
    target: HorizonJEPAEncoder,
    predictor: HorizonJEPAPredictor,
    optimizer: torch.optim.Optimizer,
    enc_cfg: HorizonJEPAEncoderConfig,
    pred_cfg: HorizonJEPAPredictorConfig,
    train_args: dict,
    stock_stats: dict,
    horizons: List[int],
    val_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "enc_cfg": enc_cfg.to_dict(),
        "pred_cfg": pred_cfg.to_dict(),
        "train_args": train_args,
        "stock_stats": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stock_stats.items()},
        "horizons": list(horizons),
        "val_metrics": val_metrics,
        "format_version": "jepa_horizon",
    }
    torch.save(state, path)


# =============================================================================
#  Smoke test
# =============================================================================

def smoke_test() -> None:
    print("=" * 80)
    print("Horizon JEPA — SMOKE TEST")
    print("=" * 80)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [0, 1, 5, 10, 20]
    nH = len(horizons)

    enc_cfg = HorizonJEPAEncoderConfig(L=10, n_stocks=7, K=20, S=4)
    pred_cfg = HorizonJEPAPredictorConfig(K=20, S=4, d_in=enc_cfg.d_model, n_horizons=nH)

    online = HorizonJEPAEncoder(enc_cfg).to(device)
    target = HorizonJEPAEncoder(enc_cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad = False
    predictor = HorizonJEPAPredictor(pred_cfg).to(device)

    n_online = sum(p.numel() for p in online.parameters())
    n_target = sum(p.numel() for p in target.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"  online params    : {n_online:,}")
    print(f"  target params    : {n_target:,} (EMA copy)")
    print(f"  predictor params : {n_pred:,}  (incl. horizon_embed: {pred_cfg.n_horizons}×{pred_cfg.d_pred})")
    print(f"  horizons         : {horizons}")

    B = 4
    # New data path: W_t (endpoint window) is decoupled from target windows.
    W_t = torch.randn(B, enc_cfg.K, 2, enc_cfg.L, 2, device=device)
    target_windows = torch.randn(B, nH, enc_cfg.K, 2, enc_cfg.L, 2, device=device)
    stock_ids = torch.randint(0, enc_cfg.n_stocks, (B,), device=device)

    regions = split_window_into_regions(W_t)
    print(f"  region split shape : {tuple(regions.shape)}")
    assert regions.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.raw_per_token)

    online_out = online(W_t, stock_ids, mask=None)
    print(f"  online_out shape   : {tuple(online_out.shape)}")
    assert online_out.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)

    # Target batched forward across horizons
    with torch.no_grad():
        W_flat = target_windows.reshape(B * nH, *target_windows.shape[2:])
        stock_flat = stock_ids.unsqueeze(1).expand(B, nH).reshape(-1)
        target_out_flat = target(W_flat, stock_flat, mask=None)
        target_out_all = target_out_flat.reshape(B, nH, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)
    print(f"  target_out_all     : {tuple(target_out_all.shape)}")
    assert target_out_all.shape == (B, nH, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)

    # Per-horizon predictor + loss
    per_h_losses = []
    print("  per-horizon (random data; at init target=online so cos≈1):")
    for h_idx, H in enumerate(horizons):
        horizon_idx = torch.full((B,), h_idx, device=device, dtype=torch.long)
        pred_H = predictor(online_out, horizon_idx)
        target_H = target_out_all[:, h_idx]
        assert pred_H.shape == target_H.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)
        L_H = jepa_loss_full(pred_H, target_H, loss_type="l1")
        cos_H = cosine_token_mean(pred_H, target_H)
        per_h_losses.append(L_H)
        print(f"    H={H:2d}  L_jepa={L_H.item():.4f}  cos_pred_target={cos_H:+.3f}")

    L_total = torch.stack(per_h_losses).mean()
    print(f"  L_total            : {L_total.item():.4f}")

    L_total.backward()
    n_grad_online = sum(1 for p in online.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_grad_pred = sum(1 for p in predictor.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_grad_target = sum(1 for p in target.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  online params w/ grad   : {n_grad_online}/{sum(1 for p in online.parameters() if p.requires_grad)}")
    print(f"  predictor params w/ grad: {n_grad_pred}/{sum(1 for p in predictor.parameters() if p.requires_grad)}")
    print(f"  target params w/ grad   : {n_grad_target}/{sum(1 for _ in target.parameters())} (must be 0)")
    assert n_grad_target == 0

    he_grad = predictor.horizon_embed.weight.grad
    he_nonzero = he_grad is not None and he_grad.abs().sum().item() > 0
    print(f"  horizon_embed grad nz   : {he_nonzero}")
    assert he_nonzero, "horizon_embed must receive gradient"

    # EMA sanity
    target_sum_before = sum(p.data.abs().sum().item() for p in target.parameters())
    with torch.no_grad():
        next(online.parameters()).add_(1e-3)
    update_ema(target, online, tau=0.99)
    target_sum_after = sum(p.data.abs().sum().item() for p in target.parameters())
    ema_delta = abs(target_sum_after - target_sum_before)
    print(f"  EMA update delta        : {ema_delta:.6f} (>0 expected)")
    assert ema_delta > 0

    # Anchor diagnostics
    with torch.no_grad():
        target_out_H0 = target_out_all[:, 0]
        cos_OT_H0 = cosine_token_mean(online_out, target_out_H0)
        gap_norm_H0 = (online_out - target_out_H0).norm(dim=-1).mean().item()
    print(f"  cos(online, target_H0)  : {cos_OT_H0:+.3f}  (≈1 expected at init: target==online)")
    print(f"  gap_norm_H0             : {gap_norm_H0:.4f}  (≈0 expected at init)")

    print("\nAll smoke checks passed.")
    print("=" * 80)


# =============================================================================
#  Main
# =============================================================================

def parse_horizons(s: str) -> List[int]:
    """Parse comma-separated horizon string. Output is sorted, non-negative.

    H=0 is OPTIONAL. When present it acts as the BYOL-style anchor target
    (predictor(online(W_t), H=0) vs target(W_t)). When absent (no-anchor
    ablation) the predictor is trained only on strictly-future horizons.

    The online encoder ALWAYS sees W_t (the endpoint window) regardless of
    whether 0 is in the horizon list; W_t is supplied by the dataset as a
    dedicated tensor, decoupled from the target-horizon list.
    """
    horizons = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    if not horizons:
        raise ValueError("Horizons list is empty.")
    if any(h < 0 for h in horizons):
        raise ValueError(f"Horizons must be non-negative. Got {horizons}")
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Horizon JEPA: horizon-conditioned JEPA latent extrapolation"
    )

    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test and exit")

    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--ckpt_dir", type=str, required=False)

    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--vol_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eta_min_frac", type=float, default=0.01)

    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_latent", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spatial_n_layers", type=int, default=2)
    parser.add_argument("--spatial_n_heads", type=int, default=4)
    parser.add_argument("--temporal_n_layers", type=int, default=2)
    parser.add_argument("--temporal_n_heads", type=int, default=4)
    parser.add_argument(
        "--temporal_causal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Block-causal attention over the flattened (K, S) token grid.",
    )

    parser.add_argument("--pred_n_layers", type=int, default=4)
    parser.add_argument("--pred_d_model", type=int, default=64)
    parser.add_argument("--pred_d_ffn", type=int, default=128)
    parser.add_argument("--pred_n_heads", type=int, default=4)

    parser.add_argument(
        "--horizons", type=str, default="0,1,5,10,20",
        help="Comma-separated horizon offsets (must include 0 as anchor).",
    )

    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])

    parser.add_argument("--tau_start", type=float, default=0.996)
    parser.add_argument("--tau_end", type=float, default=0.9995)

    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.dataset is None or args.ckpt_dir is None:
        raise SystemExit("--dataset and --ckpt_dir are required when not running --smoke_test")

    _check_project_imports_or_exit()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    horizons = parse_horizons(args.horizons)
    max_horizon = max(horizons)
    nH = len(horizons)
    print(f"Horizons: {horizons}  (max_horizon={max_horizon}, nH={nH})")
    print(f"H=0 anchor: present (always required as first entry)")

    print("\n[1/10] Loading raw LOBench dataset...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    n_stocks = int(raw["min_spread_z_per_stock"].shape[0])
    N = len(mid_z)
    L = book.shape[2]
    print(f"  N={N:,}  n_stocks={n_stocks}  L={L}")

    print("\n[2/10] Computing vol_clip mask...")
    bid_v = book[:, 0, :, 1]
    ask_v = book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    print(f"  pass vol_clip: {vol_mask.sum():,}/{N:,} ({100 * vol_mask.sum() / N:.2f}%)")

    enc_cfg = HorizonJEPAEncoderConfig(
        L=L,
        n_stocks=n_stocks,
        K=args.K,
        S=S_DEFAULT,
        raw_per_token=10,
        d_model=args.d_model,
        d_latent=args.d_latent,
        spatial_n_layers=args.spatial_n_layers,
        spatial_n_heads=args.spatial_n_heads,
        temporal_n_layers=args.temporal_n_layers,
        temporal_n_heads=args.temporal_n_heads,
        temporal_causal=args.temporal_causal,
        dropout=args.dropout,
    )
    pred_cfg = HorizonJEPAPredictorConfig(
        K=args.K,
        S=S_DEFAULT,
        d_in=args.d_model,
        d_pred=args.pred_d_model,
        n_layers=args.pred_n_layers,
        n_heads=args.pred_n_heads,
        d_ffn=args.pred_d_ffn,
        dropout=0.0,
        n_horizons=nH,
    )

    print("\n[3/10] Config:")
    print(f"  K={args.K} S={S_DEFAULT} d_model={args.d_model} d_latent={args.d_latent}")
    print(
        f"  encoder: spatial_layers={args.spatial_n_layers} spatial_heads={args.spatial_n_heads} "
        f"temporal_layers={args.temporal_n_layers} temporal_heads={args.temporal_n_heads} "
        f"temporal_causal={args.temporal_causal}"
    )
    print(
        f"  predictor: layers={args.pred_n_layers} d_pred={args.pred_d_model} "
        f"d_ffn={args.pred_d_ffn} heads={args.pred_n_heads} horizon_embed={nH}×{args.pred_d_model}"
    )
    print(f"  loss: {args.loss_type}  tau: {args.tau_start} -> {args.tau_end}")

    print("\n[4/10] Computing valid endpoints (max_horizon for K-window space)...")
    t0 = time.time()
    valid_t = compute_valid_endpoints(stock_ids, day_ids, args.K, max_horizon, vol_mask)
    print(f"  valid_t: {len(valid_t):,}  ({time.time() - t0:.1f}s)")

    print("\n[5/10] Grouped split by (stock, day)...")
    train_pos, val_pos = grouped_split_by_stock_day(stock_ids, day_ids, valid_t, args.val_frac, args.seed)
    if args.max_train_samples > 0 and len(train_pos) > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_pos = np.sort(rng.choice(train_pos, size=args.max_train_samples, replace=False))
    if args.max_val_samples > 0 and len(val_pos) > args.max_val_samples:
        rng = np.random.default_rng(args.seed + 1)
        val_pos = np.sort(rng.choice(val_pos, size=args.max_val_samples, replace=False))

    valid_t_train = valid_t[train_pos]
    valid_t_val = valid_t[val_pos]
    print(f"  train endpoints: {len(valid_t_train):,}")
    print(f"  val   endpoints: {len(valid_t_val):,}")

    print("\n[6/10] Computing per-stock normalization stats (TRAIN-only)...")
    stock_stats = compute_stock_stats_train_only(book, mid_z, stock_ids, day_ids, valid_t_train, n_stocks)

    print("\n[7/10] Building datasets...")
    ds_train = HorizonJEPADataset(book, mid_z, stock_ids, valid_t_train, stock_stats, args.K, horizons)
    ds_val = HorizonJEPADataset(book, mid_z, stock_ids, valid_t_val, stock_stats, args.K, horizons)
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    print("  datasets and dataloaders built")

    print("\n[8/10] Building models...")
    online = HorizonJEPAEncoder(enc_cfg).to(device)
    target = HorizonJEPAEncoder(enc_cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad = False
    predictor = HorizonJEPAPredictor(pred_cfg).to(device)

    n_online = sum(p.numel() for p in online.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"  online params   : {n_online:,}")
    print(f"  predictor params: {n_pred:,}")
    print(f"  total trainable : {n_online + n_pred:,}")

    optimizer = torch.optim.AdamW(
        list(online.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * args.eta_min_frac,
    )

    print("\n[9/10] Training...")
    print("  Note on reading per-H losses: H=1 has 19/20 timestep overlap with W_t")
    print("        (near-identity shift); H=20 has 0 overlap (pure extrapolation).")
    print("        L_H1 << L_H20 expected by task composition, not capacity.")
    print("        Watch the *trajectory* of L_H20 and cos_pred_target_H20 across epochs.")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_val = float("inf")

    def _per_h_str(m: Dict[str, float], key_template: str) -> str:
        return "  ".join([f"H{H:>2d}={m[key_template.format(H=H)]:.4f}" for H in horizons])

    for epoch in range(1, args.epochs + 1):
        tau = cosine_tau_schedule(epoch, args.epochs, args.tau_start, args.tau_end)
        t0 = time.time()

        train_m = run_epoch(
            online, target, predictor, dl_train, optimizer, device,
            tau=tau, loss_type=args.loss_type, grad_clip=args.grad_clip,
            horizons=horizons, train=True,
        )
        val_m = run_epoch(
            online, target, predictor, dl_val, optimizer=None, device=device,
            tau=tau, loss_type=args.loss_type, grad_clip=args.grad_clip,
            horizons=horizons, train=False,
        )
        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:3d}/{args.epochs}  tau={tau:.4f}  lr={lr_now:.2e}  [{dt:.1f}s]\n"
            f"  train  L_total={train_m['L_total']:.4f}  "
            f"online_norm={train_m['online_norm']:.2f}  target_norm={train_m['target_norm_H0']:.2f}  "
            f"pred_norm={train_m['pred_norm_mean']:.2f}\n"
            f"         L   : {_per_h_str(train_m, 'L_H{H}')}\n"
            f"         cos : {_per_h_str(train_m, 'cos_pred_target_H{H}')}\n"
            f"         anchor H0: cos_OT={train_m['cos_online_target_H0']:+.3f}  "
            f"gap_norm={train_m['gap_norm_H0']:.4f}\n"
            f"         pooled  : std={train_m['pooled_std']:.3f}  "
            f"eff_rank={train_m['pooled_eff_rank']:.2f}\n"
            f"  val    L_total={val_m['L_total']:.4f}  "
            f"online_norm={val_m['online_norm']:.2f}  target_norm={val_m['target_norm_H0']:.2f}  "
            f"pred_norm={val_m['pred_norm_mean']:.2f}\n"
            f"         L   : {_per_h_str(val_m, 'L_H{H}')}\n"
            f"         cos : {_per_h_str(val_m, 'cos_pred_target_H{H}')}\n"
            f"         anchor H0: cos_OT={val_m['cos_online_target_H0']:+.3f}  "
            f"gap_norm={val_m['gap_norm_H0']:.4f}\n"
            f"         pooled  : std={val_m['pooled_std']:.3f}  "
            f"eff_rank={val_m['pooled_eff_rank']:.2f}"
        )

        history.append({
            "epoch": epoch,
            "tau": tau,
            "lr": lr_now,
            "train": train_m,
            "val": val_m,
        })

        if val_m["L_total"] < best_val:
            best_val = val_m["L_total"]
            save_checkpoint(
                ckpt_dir / "best.pt",
                epoch, online, target, predictor, optimizer,
                enc_cfg, pred_cfg, vars(args), stock_stats, horizons, val_m,
            )
            print(f"  -> saved best.pt (val L_total={best_val:.4f})")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                epoch, online, target, predictor, optimizer,
                enc_cfg, pred_cfg, vars(args), stock_stats, horizons, val_m,
            )
            print(f"  -> saved epoch_{epoch:03d}.pt")

        save_checkpoint(
            ckpt_dir / "last.pt",
            epoch, online, target, predictor, optimizer,
            enc_cfg, pred_cfg, vars(args), stock_stats, horizons, val_m,
        )

    print("\n[10/10] Saving history...")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  history: {ckpt_dir / 'history.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

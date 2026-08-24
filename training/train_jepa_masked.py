"""
train_jepa_masked.py — Masked JEPA Stage 1: spatio-temporal JEPA on LOB tokens.

Purpose
=======
Pretrains a JEPA-style encoder with TOKEN-PRESERVING spatio-temporal masked
prediction on LOB windows. This is Stage 1 of the two-stage Masked JEPA pipeline.

Stage 2, in a separate script, trains an F0-style one-step Energy Score WM on
frozen latents z_t produced by this encoder through spatial mean pooling at the
last timestep.

Key design
==========
- Backbone: dual-axis spatial+temporal, but TOKEN-PRESERVING. Spatial pooling
  is moved to the end and only applied when extracting z_t for downstream tasks.
- Tokenization: 4 spatial regions per timestep:
      bid_top, bid_deep, ask_top, ask_deep
  each carrying 5 levels × 2 features = 10 raw values.
  With K=20 this yields 80 spatio-temporal tokens per window.
- Masking: block masking on token grid (K, S), 50-65% ratio, with semantically
  defined spatial blocks.
- Target encoder: EMA of online encoder, sees full unmasked window.
- Predictor: narrow transformer with separate learnable position embeddings;
  predicts target embeddings at masked positions.
- Loss: L1/L2 on LayerNorm-normalized target embeddings, only at masked positions.
- No VICReg in v1. Anti-collapse relies on masked prediction, EMA target,
  target normalization and predictor bottleneck.
- Causal temporal attention: exposed via CLI, default no-causal, which is more
  JEPA-like for masked representation prediction.

Important masking detail
========================
Masked online tokens are:

    mask_token + time_pos + spatial_pos + stock_emb

not just mask_token. This preserves target position and stock identity while
removing only the raw content.
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
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Make terminal output visible immediately even without stdbuf.
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
    """Called from main(). Smoke test does not need project data utilities."""
    missing = []
    if compute_valid_endpoints is None or normalize_book_window is None or compute_stock_stats is None:
        missing.append("training.train_tokenizer_t")
    if grouped_split_by_stock_day is None or compute_stock_stats_train_only is None:
        missing.append("training.train_tokenizer_t")
    if missing:
        raise SystemExit(
            f"Cannot import project modules: {missing}. "
            f"Run from the project root or add it to PYTHONPATH."
        )


# =============================================================================
#  Config dataclasses
# =============================================================================

@dataclass
class MaskedJEPAEncoderConfig:
    """Configuration for the JEPA Stage 1 encoder, used by online and target."""

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
    def from_dict(cls, d: Dict) -> "MaskedJEPAEncoderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MaskedJEPAPredictorConfig:
    """Configuration for the narrow JEPA predictor."""

    K: int = 20
    S: int = 4
    d_in: int = 128
    d_pred: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_ffn: int = 128
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict) -> "MaskedJEPAPredictorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
#  Structured Tokenizer
# =============================================================================

# Token order convention:
#   0: bid_top   = bid side, levels 0-4
#   1: bid_deep  = bid side, levels 5-9
#   2: ask_top   = ask side, levels 0-4
#   3: ask_deep  = ask side, levels 5-9
TOKEN_NAMES = ("bid_top", "bid_deep", "ask_top", "ask_deep")
S_DEFAULT = 4


def split_window_into_regions(book: torch.Tensor) -> torch.Tensor:
    """Split LOB window into 4 spatio-region tokens per timestep.

    Args:
        book: (B, K, 2, L, 2), sides {0=bid, 1=ask}, features {0=price, 1=volume}.

    Returns:
        regions: (B, K, 4, 10), where 10 = 5 levels × 2 features.
    """
    B, K, side_dim, L, feat_dim = book.shape
    if side_dim != 2:
        raise ValueError(f"Expected side dim = 2, got {side_dim}")
    if feat_dim != 2:
        raise ValueError(f"Expected feature dim = 2, got {feat_dim}")
    half = L // 2
    if L != 2 * half:
        raise ValueError(f"L must be even to split into top/deep evenly, got L={L}")

    bid_top = book[:, :, 0, :half, :].reshape(B, K, half * feat_dim)
    bid_deep = book[:, :, 0, half:, :].reshape(B, K, half * feat_dim)
    ask_top = book[:, :, 1, :half, :].reshape(B, K, half * feat_dim)
    ask_deep = book[:, :, 1, half:, :].reshape(B, K, half * feat_dim)
    return torch.stack([bid_top, bid_deep, ask_top, ask_deep], dim=2)


class StructuredTokenizer(nn.Module):
    """Raw LOB window -> token embeddings with time/spatial/stock bias."""

    def __init__(self, cfg: MaskedJEPAEncoderConfig):
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
        """Return raw content projection and positional/stock bias separately.

        This separation is crucial for JEPA masking. Masked tokens should lose
        raw content but keep time/spatial/stock identity:

            masked_x = mask_token + bias
        """
        regions = split_window_into_regions(book)                              # (B, K, S, 10)
        content = self.proj(regions)                                            # (B, K, S, d)

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
#  Encoder
# =============================================================================

class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

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
    """Build a block-causal mask over a flattened (K, S) token grid.

    Returned shape is (K*S, K*S), where rows are query positions and columns are
    key positions. True means attention is disallowed.

    Query at time i may attend to all keys with time j <= i, across all spatial
    positions. Same-timestep spatial tokens mutually attend.
    """
    time_idx = torch.arange(K, device=device).unsqueeze(1).expand(K, S).reshape(-1)
    return time_idx.unsqueeze(1) < time_idx.unsqueeze(0)


class MaskedJEPAEncoder(nn.Module):
    """Token-preserving dual-axis encoder.

    Pipeline:
        raw window -> StructuredTokenizer -> (B, K, S, d)
        optional content masking, keeping positional/stock bias
        spatial attention per timestep, no pooling
        temporal/spatio-temporal attention over K*S tokens
        output (B, K, S, d)
    """

    def __init__(self, cfg: MaskedJEPAEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = StructuredTokenizer(cfg)

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

    def forward(self, book: torch.Tensor, stock_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            book: (B, K, 2, L, 2)
            stock_ids: (B,)
            mask: optional (B, K, S) bool, True = content-masked for online branch.

        Returns:
            token_grid: (B, K, S, d_model)
        """
        cfg = self.cfg
        B = book.shape[0]

        content, bias = self.tokenizer.content_and_bias(book, stock_ids)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(content), content)
        else:
            x = content
        x = x + bias

        # Spatial attention per timestep: (B,K,S,d) -> (B*K,S,d) -> (B,K,S,d)
        x = x.reshape(B * cfg.K, cfg.S, cfg.d_model)
        for blk in self.spatial_blocks:
            x = blk(x, attn_mask=None)
        x = x.reshape(B, cfg.K, cfg.S, cfg.d_model)

        # Temporal/spatio-temporal attention: (B,K,S,d) -> (B,K*S,d)
        x = x.reshape(B, cfg.K * cfg.S, cfg.d_model)
        causal_mask = None
        if cfg.temporal_causal:
            causal_mask = self._causal_mask.to(x.device)
        for blk in self.temporal_blocks:
            x = blk(x, attn_mask=causal_mask)

        x = self.final_norm(x)
        return x.reshape(B, cfg.K, cfg.S, cfg.d_model)

    @torch.no_grad()
    def pool_z(self, token_grid: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """Pool last timestep spatial tokens and project to z_t."""
        last = token_grid[:, -1, :, :]                                           # (B, S, d)
        pooled = last.mean(dim=1)                                                # (B, d)
        return proj(pooled)


@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, tau: float) -> None:
    """EMA update: target <- tau target + (1-tau) online."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.data.copy_(b_o.data)


def cosine_tau_schedule(epoch: int, total: int, tau_start: float = 0.996, tau_end: float = 0.9995) -> float:
    """Cosine ramp from tau_start at epoch 1 to tau_end at final epoch."""
    if total <= 1:
        return tau_end
    e = max(1, min(epoch, total))
    progress = (e - 1) / (total - 1)
    return tau_start + (tau_end - tau_start) * 0.5 * (1.0 - math.cos(math.pi * progress))


# =============================================================================
#  Predictor
# =============================================================================

class MaskedJEPAPredictor(nn.Module):
    """Narrow transformer predictor with separate position embeddings."""

    def __init__(self, cfg: MaskedJEPAPredictorConfig):
        super().__init__()
        self.cfg = cfg
        self.down = nn.Linear(cfg.d_in, cfg.d_pred)
        self.up = nn.Linear(cfg.d_pred, cfg.d_in)
        self.norm_in = nn.LayerNorm(cfg.d_pred)
        self.norm_out = nn.LayerNorm(cfg.d_in)

        self.pred_mask_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.d_pred))
        self.pred_time_pos = nn.Parameter(torch.zeros(cfg.K, cfg.d_pred))
        self.pred_spatial_pos = nn.Parameter(torch.zeros(cfg.S, cfg.d_pred))
        nn.init.trunc_normal_(self.pred_mask_token, std=0.02)
        nn.init.trunc_normal_(self.pred_time_pos, std=0.02)
        nn.init.trunc_normal_(self.pred_spatial_pos, std=0.02)

        self.blocks = nn.ModuleList([
            _TransformerBlock(cfg.d_pred, cfg.n_heads, cfg.d_ffn, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

    def forward(self, online_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict target embeddings for masked positions.

        Args:
            online_output: (B, K, S, d_in)
            mask: (B, K, S) bool

        Returns:
            predicted: (B, K, S, d_in)
        """
        cfg = self.cfg
        B = online_output.shape[0]
        x = self.down(online_output)                                             # (B, K, S, d_pred)

        pos = (
            self.pred_time_pos.view(1, cfg.K, 1, cfg.d_pred)
            + self.pred_spatial_pos.view(1, 1, cfg.S, cfg.d_pred)
        )
        x = torch.where(mask.unsqueeze(-1), self.pred_mask_token.expand_as(x), x)
        x = self.norm_in(x + pos)

        x = x.reshape(B, cfg.K * cfg.S, cfg.d_pred)
        for blk in self.blocks:
            x = blk(x, attn_mask=None)
        x = x.reshape(B, cfg.K, cfg.S, cfg.d_pred)

        return self.norm_out(self.up(x))


# =============================================================================
#  Block masking
# =============================================================================

SEMANTIC_SPATIAL_BLOCKS = {
    "single_region": [(0,), (1,), (2,), (3,)],
    "side_block": [(0, 1), (2, 3)],
    "depth_block": [(0, 2), (1, 3)],
    "all_spatial": [(0, 1, 2, 3)],
}

DEFAULT_SPATIAL_FAMILY_PROBS = {
    "single_region": 0.45,
    "side_block": 0.30,
    "depth_block": 0.20,
    "all_spatial": 0.05,
}


def sample_spatial_block(rng: random.Random, family_probs: Dict[str, float]) -> Tuple[int, ...]:
    families = list(family_probs.keys())
    weights = [family_probs[f] for f in families]
    family = rng.choices(families, weights=weights, k=1)[0]
    return rng.choice(SEMANTIC_SPATIAL_BLOCKS[family])


def sample_one_window_mask(
    K: int,
    S: int,
    rng: random.Random,
    family_probs: Dict[str, float],
    mask_ratio_low: float,
    mask_ratio_high: float,
    n_blocks_min: int,
    n_blocks_max: int,
    tau_t_min: int,
    tau_t_max: int,
    keep_last_at_least_one: bool,
) -> np.ndarray:
    """Sample a single (K, S) bool mask. True = masked."""
    total = K * S
    mask = np.zeros((K, S), dtype=bool)

    n_blocks = rng.randint(n_blocks_min, n_blocks_max)
    for _ in range(n_blocks):
        spatial = sample_spatial_block(rng, family_probs)
        tau_t = min(rng.randint(tau_t_min, tau_t_max), K)
        t_start = rng.randint(0, K - tau_t)
        for t in range(t_start, t_start + tau_t):
            for s in spatial:
                mask[t, s] = True

    while mask.sum() / total < mask_ratio_low:
        spatial = sample_spatial_block(rng, family_probs)
        tau_t = min(rng.randint(tau_t_min, tau_t_max), K)
        t_start = rng.randint(0, K - tau_t)
        for t in range(t_start, t_start + tau_t):
            for s in spatial:
                mask[t, s] = True
        if mask.sum() / total >= 0.95:
            break

    while mask.sum() / total > mask_ratio_high:
        masked_idx = np.argwhere(mask)
        if len(masked_idx) == 0:
            break
        i = rng.randrange(len(masked_idx))
        t, s = masked_idx[i]
        mask[t, s] = False

    if keep_last_at_least_one and mask[K - 1, :].all():
        mask[K - 1, rng.randrange(S)] = False

    return mask


def sample_batch_masks(
    B: int,
    K: int,
    S: int,
    seed_offset: int,
    family_probs: Dict[str, float],
    mask_ratio_low: float,
    mask_ratio_high: float,
    n_blocks_min: int,
    n_blocks_max: int,
    tau_t_min: int,
    tau_t_max: int,
    keep_last_at_least_one: bool,
) -> torch.Tensor:
    masks = np.empty((B, K, S), dtype=bool)
    for b in range(B):
        rng = random.Random(seed_offset + b)
        masks[b] = sample_one_window_mask(
            K=K,
            S=S,
            rng=rng,
            family_probs=family_probs,
            mask_ratio_low=mask_ratio_low,
            mask_ratio_high=mask_ratio_high,
            n_blocks_min=n_blocks_min,
            n_blocks_max=n_blocks_max,
            tau_t_min=tau_t_min,
            tau_t_max=tau_t_max,
            keep_last_at_least_one=keep_last_at_least_one,
        )
    return torch.from_numpy(masks)


# =============================================================================
#  Dataset
# =============================================================================

class MaskedJEPADataset(Dataset):
    """Returns normalized K-window ending at valid endpoint t."""

    def __init__(
        self,
        book: np.ndarray,
        mid_z: np.ndarray,
        stock_ids: np.ndarray,
        valid_t: np.ndarray,
        stock_stats: Dict[str, np.ndarray],
        K: int,
    ):
        self.book = book
        self.mid_z = mid_z
        self.stock_ids = stock_ids
        self.valid_t = valid_t
        self.stock_stats = stock_stats
        self.K = K

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, idx: int):
        t = int(self.valid_t[idx])
        s = int(self.stock_ids[t])
        book_win = self.book[t - self.K + 1 : t + 1]
        mid_win = self.mid_z[t - self.K + 1 : t + 1]
        book_norm = normalize_book_window(book_win, mid_win, s, self.stock_stats)
        return torch.from_numpy(book_norm).float(), torch.tensor(s, dtype=torch.long)


# =============================================================================
#  Loss and diagnostics
# =============================================================================

def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str = "l1") -> torch.Tensor:
    """JEPA loss on masked positions after per-token LayerNorm."""
    pred_ln = F.layer_norm(predicted, predicted.shape[-1:])
    target_ln = F.layer_norm(target, target.shape[-1:]).detach()

    diff = pred_ln - target_ln
    if loss_type == "l1":
        per_position = diff.abs().mean(dim=-1)
    elif loss_type == "l2":
        per_position = diff.pow(2).mean(dim=-1)
    else:
        raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type}")

    mask_f = mask.float()
    return (per_position * mask_f).sum() / mask_f.sum().clamp_min(1.0)


@torch.no_grad()
def diagnostic_metrics(
    online_out: torch.Tensor,
    target_out: torch.Tensor,
    predicted: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute scalar diagnostics for logging."""
    online_norm = online_out.norm(dim=-1).mean().item()
    target_norm = target_out.norm(dim=-1).mean().item()
    pred_norm = predicted.norm(dim=-1).mean().item()
    mask_ratio = mask.float().mean().item()

    unmasked = (~mask).float()
    if unmasked.sum() > 0:
        cos = F.cosine_similarity(online_out, target_out, dim=-1)
        cos_unmasked = (cos * unmasked).sum() / unmasked.sum()
    else:
        cos_unmasked = online_out.new_tensor(0.0)

    pooled = online_out[:, -1, :, :].mean(dim=1)
    pooled_std = pooled.std(dim=0).mean().item()

    pooled_c = pooled - pooled.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(pooled_c)
        p = s / (s.sum() + 1e-12)
        p = p[p > 1e-12]
        eff_rank = torch.exp(-(p * torch.log(p)).sum()).item()
    except Exception:
        eff_rank = float("nan")

    return {
        "online_norm": float(online_norm),
        "target_norm": float(target_norm),
        "pred_norm": float(pred_norm),
        "mask_ratio": float(mask_ratio),
        "cos_unmasked": float(cos_unmasked.item()),
        "pooled_std": float(pooled_std),
        "pooled_eff_rank": float(eff_rank),
    }


# =============================================================================
#  Epoch loop
# =============================================================================

def run_epoch(
    online: MaskedJEPAEncoder,
    target: MaskedJEPAEncoder,
    predictor: MaskedJEPAPredictor,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    tau: float,
    loss_type: str,
    family_probs: Dict[str, float],
    mask_ratio_low: float,
    mask_ratio_high: float,
    n_blocks_min: int,
    n_blocks_max: int,
    tau_t_min: int,
    tau_t_max: int,
    keep_last_at_least_one: bool,
    grad_clip: float,
    mask_seed_offset_base: int,
    train: bool,
) -> Dict[str, float]:
    if train:
        online.train()
        predictor.train()
    else:
        online.eval()
        predictor.eval()
    target.eval()

    sums = {
        "L_jepa": 0.0,
        "online_norm": 0.0,
        "target_norm": 0.0,
        "pred_norm": 0.0,
        "mask_ratio": 0.0,
        "cos_unmasked": 0.0,
        "pooled_std": 0.0,
        "pooled_eff_rank": 0.0,
    }
    n_total = 0

    for batch_idx, (book, stock_ids) in enumerate(loader):
        book = book.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        B = book.shape[0]
        K = online.cfg.K
        S = online.cfg.S

        mask = sample_batch_masks(
            B=B,
            K=K,
            S=S,
            seed_offset=mask_seed_offset_base + batch_idx * 1000,
            family_probs=family_probs,
            mask_ratio_low=mask_ratio_low,
            mask_ratio_high=mask_ratio_high,
            n_blocks_min=n_blocks_min,
            n_blocks_max=n_blocks_max,
            tau_t_min=tau_t_min,
            tau_t_max=tau_t_max,
            keep_last_at_least_one=keep_last_at_least_one,
        ).to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            online_out = online(book, stock_ids, mask=mask)
            with torch.no_grad():
                target_out = target(book, stock_ids, mask=None)
            predicted = predictor(online_out, mask)
            loss = jepa_loss(predicted, target_out, mask, loss_type=loss_type)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(online.parameters()) + list(predictor.parameters()),
                    grad_clip,
                )
            optimizer.step()
            update_ema(target, online, tau=tau)

        diag = diagnostic_metrics(online_out, target_out, predicted, mask)
        sums["L_jepa"] += float(loss.item()) * B
        for k, v in diag.items():
            sums[k] += float(v) * B
        n_total += B

    out = {k: v / max(n_total, 1) for k, v in sums.items()}
    out["n"] = n_total
    return out


# =============================================================================
#  Checkpointing
# =============================================================================

def save_checkpoint(
    path: Path,
    epoch: int,
    online: MaskedJEPAEncoder,
    target: MaskedJEPAEncoder,
    predictor: MaskedJEPAPredictor,
    optimizer: torch.optim.Optimizer,
    enc_cfg: MaskedJEPAEncoderConfig,
    pred_cfg: MaskedJEPAPredictorConfig,
    train_args: dict,
    stock_stats: dict,
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
        "val_metrics": val_metrics,
        "format_version": "jepa_masked",
    }
    torch.save(state, path)


# =============================================================================
#  Smoke test
# =============================================================================

def smoke_test() -> None:
    print("=" * 80)
    print("Masked JEPA Stage 1 — SMOKE TEST")
    print("=" * 80)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_cfg = MaskedJEPAEncoderConfig(L=10, n_stocks=7, K=20, S=4)
    pred_cfg = MaskedJEPAPredictorConfig(K=20, S=4, d_in=enc_cfg.d_model)

    online = MaskedJEPAEncoder(enc_cfg).to(device)
    target = MaskedJEPAEncoder(enc_cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad = False
    predictor = MaskedJEPAPredictor(pred_cfg).to(device)

    n_online = sum(p.numel() for p in online.parameters())
    n_target = sum(p.numel() for p in target.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"  online params    : {n_online:,}")
    print(f"  target params    : {n_target:,} (EMA copy)")
    print(f"  predictor params : {n_pred:,}")

    B = 4
    book = torch.randn(B, enc_cfg.K, 2, enc_cfg.L, 2, device=device)
    stock_ids = torch.randint(0, enc_cfg.n_stocks, (B,), device=device)

    regions = split_window_into_regions(book)
    print(f"  region split shape : {tuple(regions.shape)}")
    assert regions.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.raw_per_token)

    content, bias = online.tokenizer.content_and_bias(book, stock_ids)
    print(f"  content shape      : {tuple(content.shape)}")
    print(f"  bias shape         : {tuple(bias.shape)}")
    assert content.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)
    assert bias.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)

    mask = sample_batch_masks(
        B=B,
        K=enc_cfg.K,
        S=enc_cfg.S,
        seed_offset=42,
        family_probs=DEFAULT_SPATIAL_FAMILY_PROBS.copy(),
        mask_ratio_low=0.5,
        mask_ratio_high=0.65,
        n_blocks_min=2,
        n_blocks_max=3,
        tau_t_min=2,
        tau_t_max=4,
        keep_last_at_least_one=True,
    ).to(device)
    print(f"  mask shape         : {tuple(mask.shape)}")
    print(f"  mask ratios        : {[round(m.float().mean().item(), 3) for m in mask]}")
    assert mask.shape == (B, enc_cfg.K, enc_cfg.S)
    assert all(mask[b, -1, :].sum().item() < enc_cfg.S for b in range(B))

    online_out = online(book, stock_ids, mask=mask)
    with torch.no_grad():
        target_out = target(book, stock_ids, mask=None)
    predicted = predictor(online_out, mask)

    print(f"  online out shape   : {tuple(online_out.shape)}")
    print(f"  target out shape   : {tuple(target_out.shape)}")
    print(f"  predicted shape    : {tuple(predicted.shape)}")
    assert online_out.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)
    assert target_out.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)
    assert predicted.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)

    loss_l1 = jepa_loss(predicted, target_out, mask, loss_type="l1")
    loss_l2 = jepa_loss(predicted, target_out, mask, loss_type="l2")
    print(f"  L_jepa l1          : {loss_l1.item():.4f}")
    print(f"  L_jepa l2          : {loss_l2.item():.4f}")

    loss_l1.backward()
    n_grad_online = sum(1 for p in online.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_grad_pred = sum(1 for p in predictor.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_grad_target = sum(1 for p in target.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  online params with grad   : {n_grad_online}/{sum(1 for p in online.parameters() if p.requires_grad)}")
    print(f"  predictor params with grad: {n_grad_pred}/{sum(1 for p in predictor.parameters() if p.requires_grad)}")
    print(f"  target params with grad   : {n_grad_target}/{sum(1 for _ in target.parameters())} (must be 0)")
    assert n_grad_target == 0

    # EMA sanity: perturb online slightly, then target must move.
    target_sum_before = sum(p.data.abs().sum().item() for p in target.parameters())
    with torch.no_grad():
        first_param = next(online.parameters())
        first_param.add_(1e-3)
    update_ema(target, online, tau=0.99)
    target_sum_after = sum(p.data.abs().sum().item() for p in target.parameters())
    ema_delta = abs(target_sum_after - target_sum_before)
    print(f"  EMA update delta   : {ema_delta:.6f} (>0 expected)")
    assert ema_delta > 0

    diag = diagnostic_metrics(online_out, target_out, predicted, mask)
    print(f"  diagnostics        : {diag}")

    enc_cfg_causal = MaskedJEPAEncoderConfig(L=10, n_stocks=7, K=20, S=4, temporal_causal=True)
    enc_causal = MaskedJEPAEncoder(enc_cfg_causal).to(device)
    causal_out = enc_causal(book, stock_ids, mask=None)
    print(f"  causal forward OK  : {tuple(causal_out.shape)}")
    assert causal_out.shape == (B, enc_cfg.K, enc_cfg.S, enc_cfg.d_model)

    print("\nAll smoke checks passed.")
    print("=" * 80)


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Masked JEPA Stage 1: spatio-temporal JEPA pretraining on LOB tokens"
    )

    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test and exit")

    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--ckpt_dir", type=str, required=False)

    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--vol_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    # split_seed is DECOUPLED from --seed: it controls ONLY the grouped
    # train/val split, held FIXED across seeds and arms so multi-seed bars
    # measure optimization variance, not split variance. --seed varies.
    parser.add_argument("--split_seed", type=int, default=0)
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
        help="Use block-causal attention over the flattened (K,S) token grid.",
    )

    parser.add_argument("--pred_n_layers", type=int, default=4)
    parser.add_argument("--pred_d_model", type=int, default=64)
    parser.add_argument("--pred_d_ffn", type=int, default=128)
    parser.add_argument("--pred_n_heads", type=int, default=4)

    parser.add_argument("--mask_ratio_low", type=float, default=0.50)
    parser.add_argument("--mask_ratio_high", type=float, default=0.65)
    parser.add_argument("--n_blocks_min", type=int, default=2)
    parser.add_argument("--n_blocks_max", type=int, default=3)
    parser.add_argument("--tau_t_min", type=int, default=2)
    parser.add_argument("--tau_t_max", type=int, default=4)
    parser.add_argument("--keep_last_at_least_one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--single_region_prob", type=float, default=0.45)
    parser.add_argument("--side_block_prob", type=float, default=0.30)
    parser.add_argument("--depth_block_prob", type=float, default=0.20)
    parser.add_argument("--all_spatial_prob", type=float, default=0.05)

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

    enc_cfg = MaskedJEPAEncoderConfig(
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
    pred_cfg = MaskedJEPAPredictorConfig(
        K=args.K,
        S=S_DEFAULT,
        d_in=args.d_model,
        d_pred=args.pred_d_model,
        n_layers=args.pred_n_layers,
        n_heads=args.pred_n_heads,
        d_ffn=args.pred_d_ffn,
        dropout=0.0,
    )

    # NOTE: completion does NOT look ahead, so max_horizon=1 would suffice for
    # this arm's objective. We deliberately set it to 20 to match the horizon
    # and supervised arms, so all three compute an IDENTICAL valid_t (same
    # endpoint set, same grouped split). Endpoint parity for the controlled
    # comparison, not a requirement of the masked objective.
    max_horizon = 20

    print("\n[3/10] Config:")
    print(f"  K={args.K} S={S_DEFAULT} d_model={args.d_model} d_latent={args.d_latent}")
    print(
        f"  encoder: spatial_layers={args.spatial_n_layers} spatial_heads={args.spatial_n_heads} "
        f"temporal_layers={args.temporal_n_layers} temporal_heads={args.temporal_n_heads} "
        f"temporal_causal={args.temporal_causal}"
    )
    print(
        f"  predictor: layers={args.pred_n_layers} d_pred={args.pred_d_model} "
        f"d_ffn={args.pred_d_ffn} heads={args.pred_n_heads}"
    )
    print(
        f"  masking: ratio [{args.mask_ratio_low}, {args.mask_ratio_high}], "
        f"blocks {args.n_blocks_min}-{args.n_blocks_max}, "
        f"tau_t {args.tau_t_min}-{args.tau_t_max}, keep_last={args.keep_last_at_least_one}"
    )
    print(f"  loss: {args.loss_type}  tau: {args.tau_start} -> {args.tau_end}")

    print("\n[4/10] Computing valid endpoints...")
    t0 = time.time()
    valid_t = compute_valid_endpoints(stock_ids, day_ids, args.K, max_horizon, vol_mask)
    print(f"  valid_t: {len(valid_t):,}  ({time.time() - t0:.1f}s)")

    print("\n[5/10] Grouped split by (stock, day)...")
    train_pos, val_pos = grouped_split_by_stock_day(stock_ids, day_ids, valid_t, args.val_frac, args.split_seed)
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
    ds_train = MaskedJEPADataset(book, mid_z, stock_ids, valid_t_train, stock_stats, args.K)
    ds_val = MaskedJEPADataset(book, mid_z, stock_ids, valid_t_val, stock_stats, args.K)
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
    online = MaskedJEPAEncoder(enc_cfg).to(device)
    target = MaskedJEPAEncoder(enc_cfg).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad = False
    predictor = MaskedJEPAPredictor(pred_cfg).to(device)

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

    family_probs = {
        "single_region": args.single_region_prob,
        "side_block": args.side_block_prob,
        "depth_block": args.depth_block_prob,
        "all_spatial": args.all_spatial_prob,
    }
    total_prob = sum(family_probs.values())
    if total_prob <= 0:
        raise ValueError("Spatial family probabilities must sum to a positive value")
    family_probs = {k: v / total_prob for k, v in family_probs.items()}

    print("\n[9/10] Training...")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tau = cosine_tau_schedule(epoch, args.epochs, args.tau_start, args.tau_end)
        t0 = time.time()

        train_m = run_epoch(
            online=online,
            target=target,
            predictor=predictor,
            loader=dl_train,
            optimizer=optimizer,
            device=device,
            tau=tau,
            loss_type=args.loss_type,
            family_probs=family_probs,
            mask_ratio_low=args.mask_ratio_low,
            mask_ratio_high=args.mask_ratio_high,
            n_blocks_min=args.n_blocks_min,
            n_blocks_max=args.n_blocks_max,
            tau_t_min=args.tau_t_min,
            tau_t_max=args.tau_t_max,
            keep_last_at_least_one=args.keep_last_at_least_one,
            grad_clip=args.grad_clip,
            mask_seed_offset_base=args.seed + epoch * 10**6,
            train=True,
        )
        val_m = run_epoch(
            online=online,
            target=target,
            predictor=predictor,
            loader=dl_val,
            optimizer=None,
            device=device,
            tau=tau,
            loss_type=args.loss_type,
            family_probs=family_probs,
            mask_ratio_low=args.mask_ratio_low,
            mask_ratio_high=args.mask_ratio_high,
            n_blocks_min=args.n_blocks_min,
            n_blocks_max=args.n_blocks_max,
            tau_t_min=args.tau_t_min,
            tau_t_max=args.tau_t_max,
            keep_last_at_least_one=args.keep_last_at_least_one,
            grad_clip=args.grad_clip,
            mask_seed_offset_base=args.seed + epoch * 10**6 + 5 * 10**5,
            train=False,
        )
        scheduler.step()
        dt = time.time() - t0

        print(
            f"epoch {epoch:3d}/{args.epochs}  tau={tau:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  [{dt:.1f}s]\n"
            f"  train  L={train_m['L_jepa']:.4f}  "
            f"online_norm={train_m['online_norm']:.2f}  target_norm={train_m['target_norm']:.2f}  "
            f"pred_norm={train_m['pred_norm']:.2f}\n"
            f"         mask_ratio={train_m['mask_ratio']:.3f}  "
            f"cos_unmasked={train_m['cos_unmasked']:.3f}  "
            f"pooled_std={train_m['pooled_std']:.3f}  "
            f"pooled_eff_rank={train_m['pooled_eff_rank']:.2f}\n"
            f"  val    L={val_m['L_jepa']:.4f}  "
            f"online_norm={val_m['online_norm']:.2f}  target_norm={val_m['target_norm']:.2f}  "
            f"pred_norm={val_m['pred_norm']:.2f}\n"
            f"         mask_ratio={val_m['mask_ratio']:.3f}  "
            f"cos_unmasked={val_m['cos_unmasked']:.3f}  "
            f"pooled_std={val_m['pooled_std']:.3f}  "
            f"pooled_eff_rank={val_m['pooled_eff_rank']:.2f}"
        )

        history.append({
            "epoch": epoch,
            "tau": tau,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_m,
            "val": val_m,
        })

        if val_m["L_jepa"] < best_val:
            best_val = val_m["L_jepa"]
            save_checkpoint(
                ckpt_dir / "best.pt",
                epoch,
                online,
                target,
                predictor,
                optimizer,
                enc_cfg,
                pred_cfg,
                vars(args),
                stock_stats,
                val_m,
            )
            print(f"  -> saved best.pt (val L={best_val:.4f})")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                epoch,
                online,
                target,
                predictor,
                optimizer,
                enc_cfg,
                pred_cfg,
                vars(args),
                stock_stats,
                val_m,
            )
            print(f"  -> saved epoch_{epoch:03d}.pt")

        save_checkpoint(
            ckpt_dir / "last.pt",
            epoch,
            online,
            target,
            predictor,
            optimizer,
            enc_cfg,
            pred_cfg,
            vars(args),
            stock_stats,
            val_m,
        )

    print("\n[10/10] Saving history...")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  history: {ckpt_dir / 'history.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

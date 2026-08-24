"""
model_tokenizer_t.py — Temporal LOB tokenizer (A1-T) for sequence-aware
state representation.

Design rationale
================
Module A1 v1 / v2 produced *snapshot* tokens: each token represented the LOB at a
single instant. Empirically (PACF + latent->raw probe) this captured ~88% of the
single-snapshot predictive content but missed 30% of dynamic information that
lives across consecutive snapshots in long-memory features (spread, microprice,
best bid/ask).

A1-T (this module) builds *temporal state* tokens:

    z_t = TokenizerT(o_{t-K+1}, ..., o_t, stock_id) ∈ R^{d_latent}

with K=20 covering the empirical raw PACF range (lag 11-13). The architecture is
a dual-axis Transformer:

    spatial encoder (per snapshot k):
        2L LOB-level tokens -> attention -> h_k ∈ R^{d_model}

    temporal encoder (causal):
        h_{1..K} -> causal attention -> c_K ∈ R^{d_model}

    projection:
        z_t = Linear(c_K) ∈ R^{d_latent}

Loss design
===========
Seven losses, each with a precise role:

    1. L_recon   — reconstruct snapshot t (preserve LOB geometry)
    2. L_struct  — book ladder constraints (no crossed spread, monotonicity)
    3. L_stats   — predict immediate microstructure scalars from z_t
    4. L_future  — predict future raw feature deltas from z_t (multi-horizon)
    5. L_vol     — predict realized volatility from z_t
    6. L_cov     — soft off-diagonal covariance regularization on z
    7. L_dyn     — dynamics-aware metric alignment: distances in z-space
                   should match distances in future-target space

The forward() returns ALL losses; the trainer applies the curriculum schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class TokenizerConfigT:
    """A1-T temporal tokenizer config."""

    # ----- LOB structure (matches LOBench dataset) -----
    L: int = 10                 # number of price levels per side
    n_stocks: int = 7
    stock_emb_init_scale: float = 0.02

    # ----- Temporal context -----
    K: int = 20                 # number of consecutive snapshots in input window

    # ----- Architecture -----
    d_model: int = 128
    d_latent: int = 32
    spatial_n_layers: int = 2
    spatial_n_heads: int = 4
    spatial_d_ffn: int = 256
    temporal_n_layers: int = 2
    temporal_n_heads: int = 4
    temporal_d_ffn: int = 256
    dropout: float = 0.1

    # ----- Auxiliary heads: future raw feature targets -----
    # These match the long-memory features identified in raw_lob_pacf_analysis.
    # Each (feature, horizon) pair becomes one scalar target.
    future_features: List[str] = field(default_factory=lambda: [
        "d_spread_z",
        "d_microprice_rel",
        "d_best_bid_rel",
        "d_best_ask_rel",
        "d_top_imbalance",
    ])
    future_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # ----- Realized volatility horizons -----
    vol_horizons: List[int] = field(default_factory=lambda: [5, 20])

    # ----- Stats predictor (immediate features at snapshot t) -----
    # Same scalars as v1: spread_norm, imbalance, bid_conc, ask_conc.
    n_stats: int = 4

    # ----- Loss weights (defaults; trainer can override per-epoch) -----
    w_recon: float = 1.0
    w_struct: float = 0.1
    w_stats: float = 0.3
    w_future: float = 0.5
    w_vol: float = 0.3
    w_cov: float = 0.05
    w_dyn: float = 0.1

    # ----- Numerical stabilizers -----
    price_eps: float = 1e-6
    vol_eps: float = 1e-6
    dyn_dist_eps: float = 1e-8

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TokenizerConfigT":
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @property
    def n_future_targets(self) -> int:
        return len(self.future_features) * len(self.future_horizons)

    @property
    def n_vol_targets(self) -> int:
        return len(self.vol_horizons)

    @property
    def n_dyn_targets(self) -> int:
        """Concatenated future + vol target dim, used for L_dyn."""
        return self.n_future_targets + self.n_vol_targets


# --------------------------------------------------------------------------- #
# Building blocks                                                             #
# --------------------------------------------------------------------------- #

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block. Causal mask is supplied externally if needed."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Spatial encoder: per-snapshot LOB attention                                 #
# --------------------------------------------------------------------------- #

class LOBSpatialEncoder(nn.Module):
    """
    Encodes a single LOB snapshot (2L levels × 2 features) into a d_model vector.

    Input:  book shape (..., 2, L, 2)
            stock_emb shape (..., d_model)  [broadcast bias to all spatial tokens]
    Output: h shape (..., d_model)
    """

    def __init__(self, cfg: TokenizerConfigT):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Linear(2, cfg.d_model)               # (price, volume) -> d_model
        self.pos_embed = nn.Embedding(2 * cfg.L, cfg.d_model)      # spatial position
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.spatial_n_heads,
                             cfg.spatial_d_ffn, cfg.dropout)
            for _ in range(cfg.spatial_n_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.d_model)

    def forward(self, book: torch.Tensor, stock_emb: torch.Tensor) -> torch.Tensor:
        """
        book:      (..., 2, L, 2)  -- can be (B, 2, L, 2) or (B, K, 2, L, 2)
        stock_emb: (..., d_model)  -- matching leading dims
        returns h: (..., d_model)
        """
        leading = book.shape[:-3]                  # all dims before (2, L, 2)
        L = self.cfg.L
        # Flatten leading dims for processing.
        flat = book.reshape(-1, 2, L, 2)           # (B*, 2, L, 2)
        s_flat = stock_emb.reshape(-1, self.cfg.d_model)  # (B*, d_model)

        Bf = flat.shape[0]
        tokens = flat.reshape(Bf, 2 * L, 2)        # (B*, 2L, 2)
        x = self.token_embed(tokens)               # (B*, 2L, d_model)
        pos_ids = torch.arange(2 * L, device=x.device)
        x = x + self.pos_embed(pos_ids).unsqueeze(0)
        x = x + s_flat.unsqueeze(1)                # broadcast stock bias

        for block in self.blocks:
            x = block(x, attn_mask=None)           # spatial attention is full
        x = self.norm_out(x)
        h = x.mean(dim=1)                          # mean-pool 2L tokens -> (B*, d_model)
        return h.reshape(*leading, self.cfg.d_model)


# --------------------------------------------------------------------------- #
# Temporal encoder: causal attention over K snapshot embeddings               #
# --------------------------------------------------------------------------- #

class LOBTemporalEncoder(nn.Module):
    """
    Causal Transformer over K snapshot embeddings. Output is the last token
    representation (the present, after attending causally to history).

    Input:  h_seq shape (B, K, d_model)
    Output: c_K   shape (B, d_model)
    """

    def __init__(self, cfg: TokenizerConfigT):
        super().__init__()
        self.cfg = cfg
        self.time_embed = nn.Embedding(cfg.K, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.temporal_n_heads,
                             cfg.temporal_d_ffn, cfg.dropout)
            for _ in range(cfg.temporal_n_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.d_model)

    def _causal_mask(self, K: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        B, K, D = h_seq.shape
        if K != self.cfg.K:
            raise ValueError(f"Expected temporal length {self.cfg.K}, got {K}")
        time_ids = torch.arange(K, device=h_seq.device)
        x = h_seq + self.time_embed(time_ids).unsqueeze(0)
        mask = self._causal_mask(K, x.device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.norm_out(x)
        return x[:, -1, :]                         # last token = present state


# --------------------------------------------------------------------------- #
# Decoder: reconstruct snapshot t from z_t (no temporal axis on output)       #
# --------------------------------------------------------------------------- #

class LOBSnapshotDecoder(nn.Module):
    """
    Reconstructs a single snapshot (price ladder + volumes) from z_t.

    Input:  z shape (B, d_latent)
    Output: book_pred shape (B, 2, L, 2)
    """

    def __init__(self, cfg: TokenizerConfigT):
        super().__init__()
        self.cfg = cfg
        hidden = max(cfg.d_model, 128)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_latent, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 2 * cfg.L * 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.mlp(z)
        return out.reshape(-1, 2, self.cfg.L, 2)


# --------------------------------------------------------------------------- #
# Auxiliary heads                                                             #
# --------------------------------------------------------------------------- #

class FutureFeatureHead(nn.Module):
    """Predicts (n_features × n_horizons) future raw feature delta scalars from z_t."""

    def __init__(self, cfg: TokenizerConfigT):
        super().__init__()
        self.out_dim = cfg.n_future_targets
        hidden = max(cfg.d_model, 128)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_latent, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class RealizedVolHead(nn.Module):
    """Predicts realized volatility scalars at vol_horizons from z_t."""

    def __init__(self, cfg: TokenizerConfigT):
        super().__init__()
        self.out_dim = cfg.n_vol_targets
        hidden = max(cfg.d_model, 64)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_latent, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class StatsHead(nn.Module):
    """Predicts immediate microstructure scalars (spread, imbalance, etc.) at snapshot t."""

    def __init__(self, cfg: TokenizerConfigT):
        super().__init__()
        self.out_dim = cfg.n_stats
        hidden = max(cfg.d_model // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_latent, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


# --------------------------------------------------------------------------- #
# Full tokenizer model                                                        #
# --------------------------------------------------------------------------- #

class LOBAutoTokenizerT(nn.Module):
    """
    Temporal LOB tokenizer (A1-T).

    Forward expects:
        books_window:  (B, K, 2, L, 2)   K consecutive normalized LOB snapshots
        stock_ids:     (B,)              long, per-window stock id
        targets: dict containing:
            stats_t:        (B, n_stats)               immediate scalars at t (ground truth)
            future_targets: (B, n_future_targets)      future raw deltas (ground truth, standardized)
            vol_targets:    (B, n_vol_targets)         realized vol at vol_horizons (ground truth, standardized)

    Returns dict with:
        z:               (B, d_latent)
        book_pred:       (B, 2, L, 2)
        future_pred:     (B, n_future_targets)
        vol_pred:        (B, n_vol_targets)
        stats_pred:      (B, n_stats)
        losses: dict of named scalar losses (each is a Tensor scalar):
            L_recon, L_struct, L_stats, L_future, L_vol, L_cov, L_dyn,
            L_total_default (sum with cfg default weights, for logging only)
    """

    def __init__(self, cfg: Optional[TokenizerConfigT] = None):
        super().__init__()
        self.cfg = cfg or TokenizerConfigT()
        C = self.cfg

        # Stock embedding shared across snapshots in window.
        self.stock_embed = nn.Embedding(C.n_stocks, C.d_model)
        nn.init.normal_(self.stock_embed.weight, std=C.stock_emb_init_scale)

        self.spatial_encoder = LOBSpatialEncoder(C)
        self.temporal_encoder = LOBTemporalEncoder(C)
        self.proj = nn.Linear(C.d_model, C.d_latent)

        self.decoder = LOBSnapshotDecoder(C)
        self.stats_head = StatsHead(C)
        self.future_head = FutureFeatureHead(C)
        self.vol_head = RealizedVolHead(C)

        # Per-level reconstruction weights, decreasing with depth (level 0 = top).
        # level_w shape (1, 1, L) so it broadcasts over (B, sides, L).
        level_w = torch.exp(-0.1 * torch.arange(C.L).float()).reshape(1, 1, C.L)
        self.register_buffer("level_w", level_w, persistent=False)

        self._init_linear_weights()

    def _init_linear_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------- encoding ------------------------- #

    def encode(self, books_window: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        """books_window: (B, K, 2, L, 2). Returns z: (B, d_latent)."""
        B, K, _, L, _ = books_window.shape
        if K != self.cfg.K:
            raise ValueError(f"Expected K={self.cfg.K}, got {K}")
        if L != self.cfg.L:
            raise ValueError(f"Expected L={self.cfg.L}, got {L}")
        s_emb = self.stock_embed(stock_ids)                  # (B, d_model)
        s_emb_K = s_emb.unsqueeze(1).expand(B, K, self.cfg.d_model)  # (B, K, d_model)

        h_seq = self.spatial_encoder(books_window, s_emb_K)  # (B, K, d_model)
        c_K = self.temporal_encoder(h_seq)                   # (B, d_model)
        z = self.proj(c_K)                                   # (B, d_latent)
        return z

    # ------------------------- loss components ------------------------- #

    def _recon_loss(self, book_pred: torch.Tensor, book_true: torch.Tensor) -> torch.Tensor:
        """Weighted price+volume reconstruction loss on snapshot t."""
        C = self.cfg
        level_w = self.level_w.squeeze(-1)                   # (1, 1, L)

        price_pred = book_pred[:, :, :, 0]
        price_true = book_true[:, :, :, 0]
        vol_pred = book_pred[:, :, :, 1]
        vol_true = book_true[:, :, :, 1]

        price_sq = (price_pred - price_true).pow(2)
        vol_sq = (vol_pred - vol_true).pow(2)

        price_scale = price_true.pow(2).mean(dim=(1, 2), keepdim=True) + C.price_eps
        vol_scale = vol_true.pow(2).mean(dim=(1, 2), keepdim=True) + C.vol_eps

        price_err = (price_sq / price_scale * level_w).mean()
        vol_err = (vol_sq / vol_scale * level_w).mean()

        # Equal weights between price and volume (match v1 default 0.25:1.0 was
        # an artifact; here we treat them as comparable since both are
        # per-sample normalized).
        return 0.5 * price_err + 1.0 * vol_err

    def _structure_loss(self, book_pred: torch.Tensor) -> torch.Tensor:
        """Penalize invalid relative price ladders."""
        bid = book_pred[:, 0, :, 0]
        ask = book_pred[:, 1, :, 0]

        bid_mono = F.relu(bid[:, 1:] - bid[:, :-1]).mean()
        ask_mono = F.relu(ask[:, :-1] - ask[:, 1:]).mean()

        bid_side = F.relu(bid).mean()
        ask_side = F.relu(-ask).mean()
        spread = F.relu(bid[:, 0] - ask[:, 0]).mean()

        return bid_mono + ask_mono + 0.5 * (bid_side + ask_side) + spread

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Soft off-diagonal covariance penalty on z, batch-level."""
        if z.shape[0] < 2:
            return z.new_zeros(())
        zc = z - z.mean(dim=0, keepdim=True)
        zc = zc / (zc.std(dim=0, keepdim=True) + 1e-4)
        c = (zc.T @ zc) / (zc.shape[0] - 1)
        off = c - torch.diag(torch.diag(c))
        return off.pow(2).mean()

    def _dyn_metric_alignment_loss(
        self,
        z: torch.Tensor,
        future_targets: torch.Tensor,
        vol_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dynamics-aware metric alignment.

        Builds two pairwise distance matrices on the batch:
            D_z(i,j) = ||normalize(z_i) - normalize(z_j)||²  (latent geometry)
            D_g(i,j) = ||g_i - g_j||²                         (future geometry)
        where g_i = concat(future_targets_i, vol_targets_i) (already standardized
        upstream by the data pipeline).

        Both distance matrices are standardized over the off-diagonal entries
        (zero mean, unit std) and the loss is the MSE between them on the
        off-diagonal.

        Aligns latent geometry to dynamic-target geometry, soft and symmetric.
        """
        C = self.cfg
        B = z.shape[0]
        if B < 4:
            # Need a non-trivial batch; otherwise skip.
            return z.new_zeros(())

        # Build target vector g and normalize z (batch-level).
        g = torch.cat([future_targets, vol_targets], dim=-1)    # (B, n_dyn_targets)
        z_n = z - z.mean(dim=0, keepdim=True)
        z_n = z_n / (z_n.std(dim=0, keepdim=True) + 1e-4)

        # Pairwise squared distances.
        D_z = torch.cdist(z_n, z_n, p=2).pow(2)                  # (B, B)
        D_g = torch.cdist(g, g, p=2).pow(2)                      # (B, B)

        # Mask off the diagonal (always zero).
        mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
        D_z_off = D_z[mask]
        D_g_off = D_g[mask]

        # Standardize each off-diagonal flattened distribution.
        def std_z(x):
            mu = x.mean()
            sd = x.std() + C.dyn_dist_eps
            return (x - mu) / sd

        return (std_z(D_z_off) - std_z(D_g_off)).pow(2).mean()

    # ------------------------- forward ------------------------- #

    def forward(
        self,
        books_window: torch.Tensor,
        stock_ids: torch.Tensor,
        stats_t: torch.Tensor,
        future_targets: torch.Tensor,
        vol_targets: torch.Tensor,
    ) -> Dict:
        """
        Single forward pass producing all losses.

        books_window:    (B, K, 2, L, 2)
        stock_ids:       (B,) long
        stats_t:         (B, n_stats)            immediate scalars (standardized) for snapshot t
        future_targets:  (B, n_future_targets)   future raw deltas (standardized)
        vol_targets:     (B, n_vol_targets)      realized vol (standardized)
        """
        C = self.cfg
        # Snapshot t = last in window.
        book_t = books_window[:, -1]                              # (B, 2, L, 2)

        z = self.encode(books_window, stock_ids)                  # (B, d_latent)
        book_pred = self.decoder(z)                               # (B, 2, L, 2)
        future_pred = self.future_head(z)                         # (B, n_future_targets)
        vol_pred = self.vol_head(z)                               # (B, n_vol_targets)
        stats_pred = self.stats_head(z)                           # (B, n_stats)

        L_recon = self._recon_loss(book_pred, book_t)
        L_struct = self._structure_loss(book_pred)
        L_stats = F.mse_loss(stats_pred, stats_t)
        L_future = F.mse_loss(future_pred, future_targets)
        L_vol = F.mse_loss(vol_pred, vol_targets)
        L_cov = self._covariance_loss(z)
        L_dyn = self._dyn_metric_alignment_loss(z, future_targets, vol_targets)

        # Default-weighted total (informational; trainer may override weights).
        L_total_default = (
            C.w_recon * L_recon
            + C.w_struct * L_struct
            + C.w_stats * L_stats
            + C.w_future * L_future
            + C.w_vol * L_vol
            + C.w_cov * L_cov
            + C.w_dyn * L_dyn
        )

        return {
            "z": z,
            "book_pred": book_pred,
            "future_pred": future_pred,
            "vol_pred": vol_pred,
            "stats_pred": stats_pred,
            "losses": {
                "recon": L_recon,
                "struct": L_struct,
                "stats": L_stats,
                "future": L_future,
                "vol": L_vol,
                "cov": L_cov,
                "dyn": L_dyn,
                "total_default": L_total_default,
            },
        }


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = TokenizerConfigT()
    model = LOBAutoTokenizerT(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params              : {n_params:,}")
    print(f"K                         : {cfg.K}")
    print(f"d_latent                  : {cfg.d_latent}")
    print(f"n_future_targets          : {cfg.n_future_targets}")
    print(f"n_vol_targets             : {cfg.n_vol_targets}")
    print(f"n_dyn_targets             : {cfg.n_dyn_targets}")

    B = 8
    books = torch.randn(B, cfg.K, 2, cfg.L, 2, device=device).abs()
    stock_ids = torch.randint(0, cfg.n_stocks, (B,), device=device)
    stats_t = torch.randn(B, cfg.n_stats, device=device)
    future_targets = torch.randn(B, cfg.n_future_targets, device=device)
    vol_targets = torch.randn(B, cfg.n_vol_targets, device=device)

    out = model(books, stock_ids, stats_t, future_targets, vol_targets)
    print(f"\nz shape           : {out['z'].shape}        expected (B, {cfg.d_latent})")
    print(f"book_pred shape   : {out['book_pred'].shape}  expected (B, 2, {cfg.L}, 2)")
    print(f"future_pred shape : {out['future_pred'].shape}")
    print(f"vol_pred shape    : {out['vol_pred'].shape}")
    print(f"stats_pred shape  : {out['stats_pred'].shape}")
    print(f"\nLosses (raw, unweighted):")
    for k, v in out["losses"].items():
        print(f"  {k:14s}: {v.item():+.4f}")

    out["losses"]["total_default"].backward()
    print("\nBackward                  : OK")
    assert model.stock_embed.weight.grad is not None, "stock_embed has no grad"
    print("stock_embed.grad          : OK")
    print("All smoke tests passed.")

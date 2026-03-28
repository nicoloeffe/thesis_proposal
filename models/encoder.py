"""
encoder.py — Modulo A: Encoder Transformer del Limit Order Book.

Architettura:
  - Input    : book (2, L, 2) → 2L token, ciascuno con 2 feature (price, volume)
  - Embedding: proiezione lineare 2 → d_model con spectral norm
  - Pos enc  : embedding appreso per (side × level), shape (2*L, d_model)
  - Encoder  : N layer Transformer Encoder con multi-head self-attention (pre-norm)
  - Pooling  : mean pooling sull'output + concat scalari (mid, spread, imbalance, inv)
  - Latente  : proiezione lineare → z ∈ R^d_latent con spectral norm

Training (autoencoder):
  - Decoder MLP ausiliario che ricostruisce i volumi del book da z
  - Loss di ricostruzione pesata (livelli vicini al best hanno peso maggiore)
  - Al termine del training il decoder viene scartato e l'encoder congelato

Reference: Technical Report — Modulo A, Sezione 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

class EncoderConfig:
    L:             int   = 10     # book depth (levels per side)
    d_model:       int   = 64     # transformer internal dimension
    n_heads:       int   = 4      # attention heads
    n_layers:      int   = 2      # transformer encoder layers
    d_latent:      int   = 32     # latent space dimension
    dropout:       float = 0.1    # dropout in transformer
    level_weights: tuple = (4.0, 2.0)  # weights for level 0, 1; rest = 1.0


# ---------------------------------------------------------------------------
# LOB Encoder
# ---------------------------------------------------------------------------

class LOBEncoder(nn.Module):
    """
    Transformer encoder for Limit Order Book snapshots.

    Input:
        book    : (B, 2, L, 2)  — bid/ask sides, L levels, [price, volume]
        scalars : (B, 4)        — [mid, spread, imbalance, inventory]

    Output:
        z : (B, d_latent)
    """

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        C = self.cfg

        n_tokens = 2 * C.L

        # --- Input embedding (spectral norm for Lipschitz stability) ---
        self.token_embed = spectral_norm(nn.Linear(2, C.d_model))

        # --- Learned positional encoding ---
        self.pos_embed = nn.Embedding(n_tokens, C.d_model)

        # --- Transformer Encoder ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=C.d_model,
            nhead=C.n_heads,
            dim_feedforward=C.d_model * 4,
            dropout=C.dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=C.n_layers)

        # --- Latent projection: pooled + scalars → z (spectral norm) ---
        self.latent_proj = spectral_norm(nn.Linear(C.d_model + 4, C.d_latent))

        self._init_weights()

    def _init_weights(self) -> None:
        # spectral_norm from parametrizations wraps weight as a parametrization
        # access original weight via .parametrizations.weight.original
        nn.init.xavier_uniform_(self.token_embed.parametrizations.weight.original)
        nn.init.zeros_(self.token_embed.bias)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.latent_proj.parametrizations.weight.original)
        nn.init.zeros_(self.latent_proj.bias)

    def forward(self, book: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            book    : (B, 2, L, 2)
            scalars : (B, 4)

        Returns:
            z : (B, d_latent)
        """
        B, sides, L, feat = book.shape

        # Flatten to token sequence: (B, 2*L, 2)
        tokens = book.reshape(B, 2 * L, 2)

        # Token embedding + positional encoding
        x = self.token_embed(tokens)                              # (B, T, d_model)
        pos_ids = torch.arange(2 * L, device=book.device)
        x = x + self.pos_embed(pos_ids)                          # broadcast over batch

        # Transformer self-attention
        x = self.transformer(x)                                   # (B, T, d_model)

        # Mean pooling + concat scalars → latent
        pooled = x.mean(dim=1)                                    # (B, d_model)
        z = self.latent_proj(torch.cat([pooled, scalars], dim=-1))  # (B, d_latent)
        return z


# ---------------------------------------------------------------------------
# Auxiliary Decoder (training only)
# ---------------------------------------------------------------------------

class LOBDecoder(nn.Module):
    """
    MLP decoder: reconstructs full book snapshot (prices + volumes) from z.
    Output: (B, 2, L, 2) — same shape as input book.

    Prices are relative offsets in ticks (can be negative), volumes are ≥ 0.
    Two separate output heads to apply appropriate activations.
    """

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        C = self.cfg

        hidden = C.d_model * 2

        self.trunk = nn.Sequential(
            nn.Linear(C.d_latent, C.d_model),
            nn.LayerNorm(C.d_model),
            nn.GELU(),
            nn.Linear(C.d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # Price head: relative tick offsets, unbounded
        self.price_head = nn.Linear(hidden, 2 * C.L)
        # Volume head: non-negative
        self.vol_head   = nn.Sequential(
            nn.Linear(hidden, 2 * C.L),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, d_latent)
        Returns:
            book_pred : (B, 2, L, 2) — [price_rel, volume]
        """
        B = z.shape[0]
        L = self.cfg.L
        h = self.trunk(z)
        prices  = self.price_head(h).reshape(B, 2, L)   # (B, 2, L)
        volumes = self.vol_head(h).reshape(B, 2, L)      # (B, 2, L)
        return torch.stack([prices, volumes], dim=-1)    # (B, 2, L, 2)


# ---------------------------------------------------------------------------
# Temporal predictor head (training only)
# ---------------------------------------------------------------------------

class TemporalPredictor(nn.Module):
    """
    Lightweight MLP that predicts z_{t+1} from z_t.
    Used as auxiliary training signal — discarded after pretraining.
    """

    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d_latent),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Autoencoder — Proposal C
# ---------------------------------------------------------------------------

class LOBAutoEncoder(nn.Module):
    """
    Encoder + Decoder for pretraining. Training objective (Proposal C):

      L = L_recon_full + λ_temp * L_temporal + λ_decorr * L_decorrelation

    1. L_recon_full  : MSE on full book (prices + volumes), level-weighted
    2. L_temporal    : MSE between predicted z_{t+1} and actual encoded z_{t+1}
    3. L_decorrelation: off-diagonal covariance penalty (VICReg-style)
                        prevents dimensional collapse

    After training, only the encoder is kept.
    """

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.encoder    = LOBEncoder(self.cfg)
        self.decoder    = LOBDecoder(self.cfg)
        self.temporal   = TemporalPredictor(self.cfg.d_latent)

        # Level weights: (1, 1, L, 1) for broadcasting over (B, 2, L, 2)
        C = self.cfg
        weights = [1.0] * C.L
        for i, w in enumerate(C.level_weights):
            if i < C.L:
                weights[i] = w
        self.register_buffer(
            "level_w",
            torch.tensor(weights, dtype=torch.float32).reshape(1, 1, C.L, 1)
        )

    def _recon_loss(
        self,
        book_pred: torch.Tensor,
        book_true: torch.Tensor,
        w_price: float = 0.1,
        w_vol:   float = 1.0,
    ) -> torch.Tensor:
        """
        Weighted MSE on full book.
        book_pred/true: (B, 2, L, 2) — last dim is [price_rel, volume]
        Level weights applied only to volumes (not prices — tick offsets
        at top-of-book are nearly constant, no need to overweight them).
        """
        sq_err = (book_pred - book_true) ** 2   # (B, 2, L, 2)
        # Prices: uniform weights across levels
        loss_price = sq_err[:, :, :, 0].mean()
        # Volumes: level-weighted (top-of-book more important)
        loss_vol = (sq_err[:, :, :, 1] * self.level_w.squeeze(-1)).mean()
        return w_price * loss_price + w_vol * loss_vol

    def _temporal_loss(
        self,
        z_t: torch.Tensor,
        z_t1_target: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted and actual next latent."""
        z_t1_pred = self.temporal(z_t)
        return F.mse_loss(z_t1_pred, z_t1_target.detach())

    def _decorrelation_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        VICReg-style off-diagonal covariance penalty.
        Penalises redundancy between latent dimensions.
        """
        B, D = z.shape
        z_c = z - z.mean(dim=0)              # centre
        cov = (z_c.T @ z_c) / (B - 1)       # (D, D)
        # Zero diagonal, penalise off-diagonal squared entries
        diag_mask = torch.eye(D, device=z.device, dtype=torch.bool)
        off_diag  = cov[~diag_mask]
        return (off_diag ** 2).mean()

    def forward(
        self,
        book: torch.Tensor,
        scalars: torch.Tensor,
        book_next: torch.Tensor | None = None,
        scalars_next: torch.Tensor | None = None,
        lambda_temp:   float = 0.3,
        lambda_decorr: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            book         : (B, 2, L, 2)
            scalars      : (B, 4)
            book_next    : (B, 2, L, 2)  optional — for temporal loss
            scalars_next : (B, 4)        optional — for temporal loss
            lambda_temp  : weight for temporal loss
            lambda_decorr: weight for decorrelation loss

        Returns:
            z          : (B, d_latent)
            book_pred  : (B, 2, L, 2)
            loss_dict  : {"total", "recon", "temporal", "decorr"}
        """
        z         = self.encoder(book, scalars)
        book_pred = self.decoder(z)

        loss_recon = self._recon_loss(book_pred, book)
        loss_temp  = torch.tensor(0.0, device=z.device)
        loss_decorr = self._decorrelation_loss(z)

        if book_next is not None and scalars_next is not None:
            with torch.no_grad():
                z_next = self.encoder(book_next, scalars_next)
            loss_temp = self._temporal_loss(z, z_next)

        total = loss_recon + lambda_temp * loss_temp + lambda_decorr * loss_decorr

        return z, book_pred, {
            "total":   total,
            "recon":   loss_recon,
            "temporal": loss_temp,
            "decorr":  loss_decorr,
        }

    def encode(self, book: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Encode only. Use after training."""
        return self.encoder(book, scalars)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = EncoderConfig()
    model = LOBAutoEncoder(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc    = sum(p.numel() for p in model.encoder.parameters())
    print(f"Parametri totali  : {n_params:,}")
    print(f"Parametri encoder : {n_enc:,}")

    B = 32
    book      = torch.randn(B, 2, cfg.L, 2).to(device)
    scalars   = torch.randn(B, 4).to(device)
    book_next = torch.randn(B, 2, cfg.L, 2).to(device)
    sc_next   = torch.randn(B, 4).to(device)

    z, book_pred, losses = model(book, scalars, book_next, sc_next)
    print(f"z shape        : {z.shape}            — atteso (B, {cfg.d_latent})")
    print(f"book_pred shape: {book_pred.shape}  — atteso (B, 2, {cfg.L}, 2)")
    print(f"loss total     : {losses['total'].item():.4f}")
    print(f"  recon        : {losses['recon'].item():.4f}")
    print(f"  temporal     : {losses['temporal'].item():.4f}")
    print(f"  decorr       : {losses['decorr'].item():.4f}")

    losses["total"].backward()
    print("Backward       : OK")

    model.eval()
    with torch.no_grad():
        z2 = model.encode(book, scalars)
    print(f"encode() shape : {z2.shape}  ✓")
    print("Tutto OK")
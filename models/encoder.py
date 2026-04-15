"""
encoder.py — Modulo A: Encoder Transformer del Limit Order Book (v3).

Architettura:
  - Input    : book (2, L, 2) → 2L token, ciascuno con 2 feature (price, volume)
  - Embedding: proiezione lineare 2 → d_model con spectral norm
  - Pos enc  : embedding appreso per (side × level), shape (2*L, d_model)
  - Encoder  : N layer Transformer Encoder con multi-head self-attention (pre-norm)
  - Pooling  : mean pooling sull'output
  - Latente  : proiezione lineare → z ∈ R^d_latent con spectral norm

Note architetturali (v3):
  - Nessun scalare in input: z codifica tutta la microstruttura dal book.
  - Testa ausiliaria BookStatsPredictor: predice statistiche aggregate
    (volume medio, concentrazione al best) da z. Questa testa risolve
    il problema della normalizzazione per-sample nella recon loss,
    che elimina il segnale di scala assoluta — il principale discriminante
    tra regimi. La testa viene scartata dopo il training.

Training (autoencoder):
  L = L_recon + λ_stats * L_stats + λ_temp * L_temporal
      + λ_decorr * L_decorr + λ_var * L_var

  1. L_recon   : MSE volumi, level-weighted, per-sample normalised (shape)
  2. L_stats   : MSE su aggregate book statistics (preserva scala assoluta)
  3. L_temporal: z_t → z_{t+1} predictor (smoothness)
  4. L_decorr  : VICReg off-diagonal covariance
  5. L_var     : VICReg variance term

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
    d_model:       int   = 128    # transformer internal dimension
    n_heads:       int   = 4      # attention heads
    n_layers:      int   = 4      # transformer encoder layers
    d_latent:      int   = 24     # latent space dimension (was 16 in v2)
    dropout:       float = 0.1    # dropout in transformer
    level_weights: tuple = (4.0, 2.0)  # weights for level 0, 1; rest = 1.0


# ---------------------------------------------------------------------------
# LOB Encoder
# ---------------------------------------------------------------------------

class LOBEncoder(nn.Module):
    """
    Transformer encoder for Limit Order Book snapshots.

    Input:
        book : (B, 2, L, 2) — bid/ask sides, L levels, [price_rel, volume_norm]

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

        # --- Latent projection: pooled → z (spectral norm) ---
        self.latent_proj = spectral_norm(nn.Linear(C.d_model, C.d_latent))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.token_embed.parametrizations.weight.original)
        nn.init.zeros_(self.token_embed.bias)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.latent_proj.parametrizations.weight.original)
        nn.init.zeros_(self.latent_proj.bias)

    def forward(self, book: torch.Tensor) -> torch.Tensor:
        B, sides, L, feat = book.shape
        tokens = book.reshape(B, 2 * L, 2)
        x = self.token_embed(tokens)
        pos_ids = torch.arange(2 * L, device=book.device)
        x = x + self.pos_embed(pos_ids)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        z = self.latent_proj(pooled)
        return z


# ---------------------------------------------------------------------------
# Auxiliary Decoder (training only)
# ---------------------------------------------------------------------------

class LOBDecoder(nn.Module):
    """
    MLP decoder: reconstructs book from z.
    Output: (B, 2, L, 2) — [price_rel, volume].
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
        self.price_head = nn.Linear(hidden, 2 * C.L)
        self.vol_head = nn.Sequential(
            nn.Linear(hidden, 2 * C.L),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        L = self.cfg.L
        h = self.trunk(z)
        prices  = self.price_head(h).reshape(B, 2, L)
        volumes = self.vol_head(h).reshape(B, 2, L)
        return torch.stack([prices, volumes], dim=-1)


# ---------------------------------------------------------------------------
# Auxiliary heads (training only — all discarded after pretraining)
# ---------------------------------------------------------------------------

class TemporalPredictor(nn.Module):
    """MLP: z_t → z_{t+1}. No action conditioning (no market impact)."""

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


class BookStatsPredictor(nn.Module):
    """
    Predicts aggregate book statistics from z.

    Targets (computed from the normalised book, NOT per-sample normalised):
      0: log(mean_bid_vol + eps)     — absolute scale, bid side
      1: log(mean_ask_vol + eps)     — absolute scale, ask side
      2: bid_l0_vol / (total_bid + eps)  — concentration at best bid
      3: ask_l0_vol / (total_ask + eps)  — concentration at best ask

    Why: the per-sample normalised recon loss removes absolute volume scale,
    which is the primary regime discriminator. This head forces z to encode
    scale information that the recon loss alone doesn't incentivise.
    """

    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @staticmethod
    def compute_targets(book: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregate stats from normalised book.
        Args:
            book: (B, 2, L, 2) — last dim is [price_rel, vol_normalised]
        Returns:
            targets: (B, 4)
        """
        eps = 1e-6
        bid_vols = book[:, 0, :, 1]    # (B, L)
        ask_vols = book[:, 1, :, 1]    # (B, L)

        mean_bid = bid_vols.mean(dim=1)   # (B,)
        mean_ask = ask_vols.mean(dim=1)   # (B,)
        total_bid = bid_vols.sum(dim=1)   # (B,)
        total_ask = ask_vols.sum(dim=1)   # (B,)

        log_mean_bid = torch.log(mean_bid + eps)
        log_mean_ask = torch.log(mean_ask + eps)
        bid_concentration = bid_vols[:, 0] / (total_bid + eps)
        ask_concentration = ask_vols[:, 0] / (total_ask + eps)

        return torch.stack([
            log_mean_bid, log_mean_ask,
            bid_concentration, ask_concentration
        ], dim=1)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class LOBAutoEncoder(nn.Module):
    """
    Encoder + auxiliary heads for pretraining.

    L = L_recon + λ_stats * L_stats + λ_temp * L_temporal
        + λ_decorr * L_decorr + λ_var * L_var

    After training, only the encoder is kept.
    """

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.encoder    = LOBEncoder(self.cfg)
        self.decoder    = LOBDecoder(self.cfg)
        self.temporal   = TemporalPredictor(self.cfg.d_latent)
        self.stats_head = BookStatsPredictor(self.cfg.d_latent)

        # Level weights for recon loss
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
    ) -> torch.Tensor:
        """Volume-normalised reconstruction loss (captures shape, not scale)."""
        vol_pred = book_pred[:, :, :, 1]
        vol_true = book_true[:, :, :, 1]
        sq_err = (vol_pred - vol_true) ** 2
        weighted_sq_err = sq_err * self.level_w.squeeze(-1)
        sample_vol_sq = (vol_true ** 2).mean(dim=(1, 2), keepdim=True) + 1e-6
        normalised_err = weighted_sq_err / sample_vol_sq
        return normalised_err.mean()

    def _stats_loss(
        self,
        z: torch.Tensor,
        book: torch.Tensor,
    ) -> torch.Tensor:
        """MSE on aggregate book statistics (captures scale)."""
        targets = BookStatsPredictor.compute_targets(book)  # (B, 4)
        preds = self.stats_head(z)                          # (B, 4)
        return F.mse_loss(preds, targets.detach())

    def _temporal_loss(
        self,
        z_t: torch.Tensor,
        z_t1_target: torch.Tensor,
    ) -> torch.Tensor:
        z_t1_pred = self.temporal(z_t)
        return F.mse_loss(z_t1_pred, z_t1_target.detach())

    def _decorrelation_loss(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / (B - 1)
        diag_mask = torch.eye(D, device=z.device, dtype=torch.bool)
        off_diag = cov[~diag_mask]
        return (off_diag ** 2).mean()

    def _variance_loss(self, z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        std = z.std(dim=0)
        return F.relu(gamma - std).mean()

    def forward(
        self,
        book: torch.Tensor,
        book_next: torch.Tensor | None = None,
        lambda_temp:   float = 0.3,
        lambda_stats:  float = 1.0,
        lambda_decorr: float = 0.05,
        lambda_var:    float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            book         : (B, 2, L, 2)
            book_next    : (B, 2, L, 2)  optional — for temporal loss
            lambda_temp  : weight for temporal loss
            lambda_stats : weight for aggregate stats loss
            lambda_decorr: weight for decorrelation loss
            lambda_var   : weight for variance loss (VICReg)

        Returns:
            z          : (B, d_latent)
            book_pred  : (B, 2, L, 2)
            loss_dict  : {"total", "recon", "stats", "temporal", "decorr", "var"}
        """
        z         = self.encoder(book)
        book_pred = self.decoder(z)

        loss_recon  = self._recon_loss(book_pred, book)
        loss_stats  = self._stats_loss(z, book)
        loss_temp   = torch.tensor(0.0, device=z.device)
        loss_decorr = self._decorrelation_loss(z)
        loss_var    = self._variance_loss(z)

        if book_next is not None and lambda_temp > 0:
            with torch.no_grad():
                z_next = self.encoder(book_next)
            loss_temp = self._temporal_loss(z, z_next)
            loss_temp = torch.clamp(loss_temp, max=10.0)

        total = (loss_recon
                 + lambda_stats * loss_stats
                 + lambda_temp * loss_temp
                 + lambda_decorr * loss_decorr
                 + lambda_var * loss_var)

        return z, book_pred, {
            "total":    total,
            "recon":    loss_recon,
            "stats":    loss_stats,
            "temporal": loss_temp,
            "decorr":   loss_decorr,
            "var":      loss_var,
        }

    def encode(self, book: torch.Tensor) -> torch.Tensor:
        """Encode only. Use after training."""
        return self.encoder(book)


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
    print(f"d_latent          : {cfg.d_latent}")

    B = 32
    book      = torch.randn(B, 2, cfg.L, 2).abs().to(device)  # abs for realistic vols
    book_next = torch.randn(B, 2, cfg.L, 2).abs().to(device)

    z, book_pred, losses = model(book, book_next)
    print(f"z shape        : {z.shape}            — atteso (B, {cfg.d_latent})")
    print(f"book_pred shape: {book_pred.shape}  — atteso (B, 2, {cfg.L}, 2)")
    for k, v in losses.items():
        print(f"  {k:12s}: {v.item():.4f}")

    losses["total"].backward()
    print("Backward       : OK")

    model.eval()
    with torch.no_grad():
        z2 = model.encode(book)
    print(f"encode() shape : {z2.shape}  ✓")
    print("Tutto OK")
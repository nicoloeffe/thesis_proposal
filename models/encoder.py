"""
encoder.py — Modulo A: Encoder Transformer del Limit Order Book (v5).

Architettura:
  - Input    : book (2, L, 2) → 2L token, ciascuno con 2 feature (price, volume)
  - Embedding: proiezione lineare 2 → d_model con spectral norm
  - Pos enc  : embedding appreso per (side × level), shape (2*L, d_model)
  - Encoder  : N layer Transformer Encoder con multi-head self-attention (pre-norm)
  - Pooling  : mean pooling sull'output
  - Latente  : proiezione lineare → z ∈ R^d_latent con spectral norm

Filosofia del Modulo A (v5):
  L'encoder apprende una STATIC REPRESENTATION dello snapshot del book.
  La struttura temporale è delegata interamente al Modulo B (Causal Transformer
  + MDN head GMM).

Cambiamenti v5 rispetto a v4:
  - Rimossi VICReg variance + decorrelation loss:
    * empiricamente non cambiavano i probe downstream (v4 vs v4.1 identici);
    * reconstruction e stats loss prevengono già il collasso del latente;
    * il whitening imposto da VICReg var era deleterio (distribuzione quasi-
      sferica → perdita di struttura gerarchica tra regimi).
  - Aggiunta On-Manifold Contractive Loss (L_contr):
    * forza smoothness locale tramite coppie vicine NEL BATCH (non perturbazioni
      sintetiche OOD);
    * loss = mean(||z_i - z_j||² / ||o_i - o_j||²) per le coppie al di sotto
      del tau-esimo percentile di pairwise distance nell'input;
    * fornisce bound empirica della costante di Lipschitz dell'encoder ristretto
      alla data manifold — condizione sufficiente per la stabilità del gradient
      descent interno del Modulo C (DRO Wasserstein);
    * generalizzabile a qualunque dataset (non sfrutta artefatti del data
      generation, a differenza della temporal loss v3).

Requisiti funzionali di z (v5):
  1. Ricostruire il book                  → L_recon
  2. Preservare scala e struttura aggreg. → L_stats
  3. Bi-Lipschitz empirica sulla manifold → L_contr

Training (autoencoder):
  L = L_recon + λ_stats * L_stats + λ_contr * L_contr

  1. L_recon : MSE volumi, level-weighted, per-sample normalised (shape)
  2. L_stats : MSE su 6 aggregate book statistics (scale, concentration,
               imbalance, spread)
  3. L_contr : on-manifold contractive loss su coppie vicine nel batch

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
    d_latent:      int   = 16     # latent space dimension (v4: ridotto da 24 a 16)
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
# Auxiliary head: BookStatsPredictor (training only — discarded after)
# ---------------------------------------------------------------------------

class BookStatsPredictor(nn.Module):
    """
    Predicts aggregate book statistics from z.

    Targets (6, computed from the normalised book):
      0: log(mean_bid_vol + eps)           — absolute scale, bid side
      1: log(mean_ask_vol + eps)           — absolute scale, ask side
      2: bid_l0_vol / (total_bid + eps)    — concentration at best bid
      3: ask_l0_vol / (total_ask + eps)    — concentration at best ask
      4: (total_bid - total_ask) / total   — order flow imbalance (directional)
      5: best_ask_price - best_bid_price   — spread, in normalised tick units

    Why these 6:
      - (0, 1): the per-sample-normalised recon loss removes absolute volume
        scale, which is the primary regime discriminator. These targets force
        z to encode scale info that recon alone doesn't incentivise.
      - (2, 3): concentration captures book shape (concentrated vs distributed
        liquidity), informative about regime and adverse selection.
      - (4): imbalance is the single-best predictor of short-term mid-move
        (Cartea-Jaimungal, Cont-de Larrard). Downstream (WM, critic) benefits
        enormously from z that already encodes it.
      - (5): spread is a direct proxy of liquidity and adverse selection cost,
        and it's in the observation anyway — we make sure z preserves it.

    The price features (bid/ask L0 prices) are already in normalised tick
    units in the input book (computed in LOBDataset.normalize_book), so the
    spread target is computed directly in those units.
    """

    def __init__(self, d_latent: int, n_targets: int = 6) -> None:
        super().__init__()
        self.n_targets = n_targets
        self.net = nn.Sequential(
            nn.Linear(d_latent, 64),
            nn.GELU(),
            nn.Linear(64, n_targets),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @staticmethod
    def compute_targets(book: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregate stats from normalised book.

        Args:
            book: (B, 2, L, 2) — last dim is [price_rel, vol_normalised]
                  price_rel is already in normalised tick units (see
                  LOBDataset.normalize_book in train_encoder.py)
        Returns:
            targets: (B, 6)
        """
        eps = 1e-6
        bid_vols   = book[:, 0, :, 1]   # (B, L)
        ask_vols   = book[:, 1, :, 1]   # (B, L)
        bid_prices = book[:, 0, :, 0]   # (B, L) — in normalised tick units
        ask_prices = book[:, 1, :, 0]   # (B, L)

        mean_bid  = bid_vols.mean(dim=1)
        mean_ask  = ask_vols.mean(dim=1)
        total_bid = bid_vols.sum(dim=1)
        total_ask = ask_vols.sum(dim=1)

        log_mean_bid      = torch.log(mean_bid + eps)
        log_mean_ask      = torch.log(mean_ask + eps)
        bid_concentration = bid_vols[:, 0] / (total_bid + eps)
        ask_concentration = ask_vols[:, 0] / (total_ask + eps)

        # NEW (v4): directional and liquidity features
        imbalance = (total_bid - total_ask) / (total_bid + total_ask + eps)
        spread    = ask_prices[:, 0] - bid_prices[:, 0]  # normalised tick units

        return torch.stack([
            log_mean_bid, log_mean_ask,
            bid_concentration, ask_concentration,
            imbalance, spread,
        ], dim=1)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class LOBAutoEncoder(nn.Module):
    """
    Encoder + auxiliary heads for pretraining.

    L = L_recon + λ_stats * L_stats + λ_decorr * L_decorr + λ_var * L_var

    After training, only the encoder is kept.
    """

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.encoder    = LOBEncoder(self.cfg)
        self.decoder    = LOBDecoder(self.cfg)
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
        """MSE on aggregate book statistics (scale + directional + liquidity)."""
        targets = BookStatsPredictor.compute_targets(book)   # (B, 6)
        preds   = self.stats_head(z)                         # (B, 6)
        return F.mse_loss(preds, targets.detach())

    def _contractive_loss(
        self,
        z: torch.Tensor,
        book: torch.Tensor,
        tau_percentile: float = 10.0,
    ) -> torch.Tensor:
        """
        On-manifold contractive loss.

        Per ogni batch: prende le coppie di sample più vicine in input-space
        (sotto il tau_percentile-esimo percentile delle pairwise distances)
        e forza i loro z a essere vicini proporzionalmente. Questo è
        contractive ristretto alla data manifold — non usa perturbazioni
        sintetiche OOD.

        Loss: mean over near-pairs of ||z_i - z_j||² / (||o_i - o_j||² + eps)

        Args:
            z:    (B, d_latent)
            book: (B, 2, L, 2)
            tau_percentile: percentile cutoff per "vicini" (default 10).
        """
        B = book.shape[0]
        eps = 1e-6
        o = book.reshape(B, -1)                 # (B, D_in)

        # Pairwise L2 distances (upper triangle only, no self-pairs)
        do = torch.cdist(o, o, p=2)             # (B, B)
        dz = torch.cdist(z, z, p=2)             # (B, B)
        mask_upper = torch.triu(torch.ones_like(do, dtype=torch.bool), diagonal=1)

        do_pairs = do[mask_upper]               # (B*(B-1)/2,)
        dz_pairs = dz[mask_upper]

        # Seleziona le pairs più vicine in input-space
        k = max(1, int(len(do_pairs) * tau_percentile / 100.0))
        _, near_idx = torch.topk(do_pairs, k=k, largest=False)
        do_near = do_pairs[near_idx]
        dz_near = dz_pairs[near_idx]

        # Contractive ratio squared: vogliamo dz piccolo quando do è piccolo
        ratio = (dz_near ** 2) / (do_near ** 2 + eps)
        return ratio.mean()

    def forward(
        self,
        book: torch.Tensor,
        lambda_stats: float = 3.0,
        lambda_contr: float = 0.1,
        contr_tau_percentile: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            book                 : (B, 2, L, 2)
            lambda_stats         : weight for aggregate stats loss
            lambda_contr         : weight for on-manifold contractive loss
            contr_tau_percentile : percentile cutoff for "near pairs"

        Returns:
            z          : (B, d_latent)
            book_pred  : (B, 2, L, 2)
            loss_dict  : {"total", "recon", "stats", "contr"}
        """
        z         = self.encoder(book)
        book_pred = self.decoder(z)

        loss_recon = self._recon_loss(book_pred, book)
        loss_stats = self._stats_loss(z, book)
        loss_contr = self._contractive_loss(z, book, tau_percentile=contr_tau_percentile)

        total = (loss_recon
                 + lambda_stats * loss_stats
                 + lambda_contr * loss_contr)

        return z, book_pred, {
            "total": total,
            "recon": loss_recon,
            "stats": loss_stats,
            "contr": loss_contr,
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
    book = torch.randn(B, 2, cfg.L, 2).abs().to(device)  # abs for realistic vols

    z, book_pred, losses = model(book)
    print(f"z shape        : {z.shape}            — atteso (B, {cfg.d_latent})")
    print(f"book_pred shape: {book_pred.shape}  — atteso (B, 2, {cfg.L}, 2)")
    for k, v in losses.items():
        print(f"  {k:8s}: {v.item():.4f}")

    losses["total"].backward()
    print("Backward       : OK")

    model.eval()
    with torch.no_grad():
        z2 = model.encode(book)
    print(f"encode() shape : {z2.shape}  ✓")
    print("Tutto OK")
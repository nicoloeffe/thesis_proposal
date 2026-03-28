"""
world_model.py — Modulo B: World Model neurale per la dinamica latente del LOB.

Architettura:
  - Input  : sequenza di coppie (z_t, a_t) con z_t ∈ R^d_latent, a_t ∈ R^d_action
  - Token  : proiezione lineare [z_t, a_t] → d_model + positional encoding appreso
  - Core   : Causal Transformer (4 layer, 4 heads, pre-norm, causal mask)
  - Output : GMM head — K componenti con parametri (π_k, μ_k, log_σ_k) per ogni timestep

Training:
  - Teacher forcing su tutta la sequenza
  - Loss: NLL della GMM (log-sum-exp trick per stabilità)
  - Logging: NLL totale, entropia dei pesi π (mode collapse check), NLL per regime

Reference: Technical Report — Modulo B.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class WorldModelConfig:
    d_latent:   int   = 32     # latent dim (must match EncoderConfig.d_latent)
    d_action:   int   = 3      # action dim (k_bid, k_ask, qty)
    d_model:    int   = 128    # transformer internal dim
    n_heads:    int   = 4      # attention heads
    n_layers:   int   = 4      # transformer layers
    d_ffn:      int   = 512    # feedforward dim
    dropout:    float = 0.1    # dropout
    n_gmm:      int   = 5      # GMM components
    max_seq:    int   = 200    # max sequence length for positional encoding


# ---------------------------------------------------------------------------
# Causal Transformer block
# ---------------------------------------------------------------------------

class CausalTransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn  = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads,
            dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ffn),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ffn, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# GMM Head
# ---------------------------------------------------------------------------

class GMMHead(nn.Module):
    """
    Projects hidden state h_t to GMM parameters:
      π_k     : mixture weights (softmax)
      μ_k     : means ∈ R^d_latent
      log_σ_k : log std ∈ R^d_latent (diagonal covariance)

    Output shapes (given input h: (B, T, d_model)):
      pi      : (B, T, K)
      mu      : (B, T, K, d_latent)
      log_sig : (B, T, K, d_latent)
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        K = cfg.n_gmm
        D = cfg.d_latent

        self.pi_head  = nn.Linear(cfg.d_model, K)
        self.mu_head  = nn.Linear(cfg.d_model, K * D)
        self.sig_head = nn.Linear(cfg.d_model, K * D)

        self.K = K
        self.D = D

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = h.shape
        K, D = self.K, self.D

        pi      = F.softmax(self.pi_head(h), dim=-1)          # (B, T, K)
        mu      = self.mu_head(h).reshape(B, T, K, D)         # (B, T, K, D)
        log_sig = self.sig_head(h).reshape(B, T, K, D)        # (B, T, K, D)
        log_sig = torch.clamp(log_sig, -6.0, 2.0)             # numerical stability

        return pi, mu, log_sig


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------

class LOBWorldModel(nn.Module):
    """
    Causal Transformer World Model for LOB latent dynamics.

    Given a sequence (z_0, a_0), ..., (z_{N-1}, a_{N-1}), predicts
    the distribution of z_1, ..., z_N as a GMM at each position.

    In training with teacher forcing:
      - Input tokens: [(z_0,a_0), (z_1,a_1), ..., (z_{N-1},a_{N-1})]
      - Targets:      [z_1, z_2, ..., z_N]
      - Loss: NLL averaged over all positions and batch

    At inference:
      - Pass context window, take output at last position as prediction.
    """

    def __init__(self, cfg: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WorldModelConfig()
        C = self.cfg

        # Input projection: [z_t, a_t] → d_model
        self.input_proj = nn.Linear(C.d_latent + C.d_action, C.d_model)

        # Learned positional encoding
        self.pos_emb = nn.Embedding(C.max_seq, C.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(C) for _ in range(C.n_layers)
        ])

        self.norm_out = nn.LayerNorm(C.d_model)
        self.gmm_head = GMMHead(C)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for causal attention. True = masked out."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        z_seq:  torch.Tensor,   # (B, N+1, d_latent)  — z_0 .. z_N
        a_seq:  torch.Tensor,   # (B, N,   d_action)   — a_0 .. a_{N-1}
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_seq : (B, N+1, d_latent)  — latent sequence including target z_N
            a_seq : (B, N,   d_action)  — action sequence

        Returns:
            pi      : (B, N, K)
            mu      : (B, N, K, d_latent)
            log_sig : (B, N, K, d_latent)

        where outputs at position t predict z_{t+1}.
        """
        B, N, _ = a_seq.shape

        # Input: (z_0..z_{N-1}, a_0..a_{N-1}) → (B, N, d_latent+d_action)
        z_in  = z_seq[:, :N, :]                           # (B, N, d_latent)
        token = torch.cat([z_in, a_seq], dim=-1)          # (B, N, d_latent+d_action)
        x = self.input_proj(token)                        # (B, N, d_model)

        # Positional encoding
        pos = torch.arange(N, device=x.device)
        x   = x + self.pos_emb(pos)                      # (B, N, d_model)

        # Causal transformer
        mask = self._causal_mask(N, x.device)
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm_out(x)                              # (B, N, d_model)

        # GMM head
        pi, mu, log_sig = self.gmm_head(x)               # (B,N,K), (B,N,K,D), (B,N,K,D)

        return pi, mu, log_sig

    def nll_loss(
        self,
        pi:      torch.Tensor,   # (B, N, K)
        mu:      torch.Tensor,   # (B, N, K, D)
        log_sig: torch.Tensor,   # (B, N, K, D)
        z_next:  torch.Tensor,   # (B, N, D) — targets z_1..z_N
    ) -> torch.Tensor:
        """
        Negative log-likelihood of GMM with log-sum-exp trick.

        log p(z) = log Σ_k π_k * N(z | μ_k, σ_k²)
                 = logsumexp_k [ log π_k + log N(z | μ_k, σ_k²) ]

        where log N(z | μ, σ²) = -0.5 * Σ_d [ log(2π) + 2*log_σ + (z-μ)²/σ² ]
        """
        B, N, K, D = mu.shape

        # Expand z_next for broadcasting: (B, N, 1, D)
        z = z_next.unsqueeze(2)

        # Log-likelihood of each component: (B, N, K)
        sig      = torch.exp(log_sig)                            # (B, N, K, D)
        log_norm = -0.5 * (
            D * math.log(2 * math.pi)
            + 2 * log_sig.sum(dim=-1)                           # (B, N, K)
            + ((z - mu) ** 2 / (sig ** 2 + 1e-8)).sum(dim=-1)  # (B, N, K)
        )

        # log π_k + log N(z | μ_k, σ_k²): (B, N, K)
        log_pi   = torch.log(pi + 1e-8)
        log_comp = log_pi + log_norm                            # (B, N, K)

        # log-sum-exp over components: (B, N)
        log_p = torch.logsumexp(log_comp, dim=-1)

        return -log_p.mean()

    @torch.no_grad()
    def diagnostics(
        self,
        pi:      torch.Tensor,
        mu:      torch.Tensor,
        log_sig: torch.Tensor,
        z_next:  torch.Tensor,
        regimes: torch.Tensor,   # (B,) — regime index per sequence
    ) -> dict[str, float]:
        """
        Compute diagnostic metrics:
          - nll_total   : overall NLL
          - entropy_pi  : mean entropy of mixture weights (mode collapse check)
          - nll_regime_* : NLL per regime
        """
        nll_total = self.nll_loss(pi, mu, log_sig, z_next).item()

        # Entropy of π: H(π) = -Σ π log π, averaged over B and N
        ent = -(pi * torch.log(pi + 1e-8)).sum(dim=-1).mean().item()

        diag = {"nll_total": nll_total, "entropy_pi": ent}

        # NLL per regime
        B, N, K, D = mu.shape
        for reg_id in [0, 1, 2]:
            mask = (regimes == reg_id)
            if mask.sum() == 0:
                continue
            nll_reg = self.nll_loss(
                pi[mask], mu[mask], log_sig[mask], z_next[mask]
            ).item()
            names = ["low_vol", "mid_vol", "high_vol"]
            diag[f"nll_{names[reg_id]}"] = nll_reg

        return diag

    def encode(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the hidden state at the last position — used at inference
        to get the context embedding for DRO stress testing.
        """
        pi, mu, log_sig = self.forward(z_seq, a_seq)
        return mu[:, -1, :, :]   # (B, K, D) — GMM means at last step


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = WorldModelConfig()
    model = LOBWorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"World model params: {n_params:,}")

    B, N = 16, 20
    z_seq = torch.randn(B, N + 1, cfg.d_latent).to(device)
    a_seq = torch.randn(B, N, cfg.d_action).to(device)

    pi, mu, log_sig = model(z_seq, a_seq)
    print(f"pi      shape: {pi.shape}       — expected (B, N, K) = ({B}, {N}, {cfg.n_gmm})")
    print(f"mu      shape: {mu.shape}  — expected (B, N, K, D) = ({B}, {N}, {cfg.n_gmm}, {cfg.d_latent})")
    print(f"log_sig shape: {log_sig.shape}  — expected (B, N, K, D)")

    z_next = z_seq[:, 1:, :]   # targets
    loss = model.nll_loss(pi, mu, log_sig, z_next)
    print(f"NLL loss: {loss.item():.4f}")

    loss.backward()
    print("Backward: OK")

    regimes = torch.randint(0, 3, (B,)).to(device)
    diag = model.diagnostics(pi, mu, log_sig, z_next, regimes)
    print(f"Diagnostics: {diag}")
    print("Tutto OK")
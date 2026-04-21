"""
world_model.py — Modulo B: World Model neurale per la dinamica latente del LOB (v2).

Architettura:
  - Input  : sequenza di latenti z_t ∈ R^d_latent (NO azioni, NO stato MM)
  - Token  : proiezione lineare z_t → d_model + positional encoding appreso
  - Core   : Causal Transformer (4 layer, 4 heads, pre-norm, causal mask)
  - Output : GMM head — K componenti con parametri (π_k, μ_k, log_σ_k) per ogni timestep

Scelte di design (giustificate nel report):

  1. NO action conditioning.
     Nel simulatore stilizzato (v1, senza market impact) la dinamica del LOB
     è indipendente dalle azioni del MM:
         P(z_{t+1} | z_t, a_t) = P(z_{t+1} | z_t)
     Le azioni influenzano solo l'inventario e il PnL, pertinenti al critico
     (Fase 3), non al world model.

  2. NO MM state (inventory, time_left).
     Lo stato interno del MM non influenza la marginale P(z_{t+1} | z_t):
     è informazione privata del MM, non osservabile dal mercato. Il dataset WM
     contiene comunque inventory e time_left per uso downstream (critic training),
     ma non vengono passati al WM.

  3. Dinamica genuinamente stocastica.
     Anche condizionatamente a z_t, la transizione contiene rumore irriducibile
     (shock gaussiano del mid, arrivi Poisson di MO/LO/cancel, pro-rata sampling).
     La mixture con K componenti cattura questa varianza — una singola gaussiana
     sarebbe sotto-parametrizzata nei regimi ad alta volatilità.

Training:
  - Teacher forcing su tutta la sequenza
  - Loss: NLL della GMM (log-sum-exp trick per stabilità numerica)
  - Auxiliary: regime classification head (regolarizzazione dello hidden state)

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
    d_latent:       int   = 16     # latent dim (must match EncoderConfig.d_latent, v5: 16)
    d_model:        int   = 128    # transformer internal dim
    n_heads:        int   = 4      # attention heads
    n_layers:       int   = 4      # transformer layers
    d_ffn:          int   = 512    # feedforward dim
    dropout:        float = 0.1    # dropout
    n_gmm:          int   = 5      # GMM components
    max_seq:        int   = 200    # max sequence length for positional encoding
    n_regimes:      int   = 3      # regime classes for auxiliary head
    lambda_regime:  float = 0.1    # weight for regime classification loss


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
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out
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
        # Clamp log_sig per z normalizzato (std=1 globale):
        # σ ∈ [e^-4.5, e^0.7] = [0.011, 2.01] — range adeguato alla scala N(0,1)
        log_sig = torch.clamp(log_sig, -4.5, 0.7)

        return pi, mu, log_sig


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------

class LOBWorldModel(nn.Module):
    """
    Causal Transformer World Model for LOB latent dynamics.

    Given a sequence z_0, ..., z_{N-1}, predicts
    the distribution of z_1, ..., z_N as a GMM at each position.

    No action conditioning: the LOB dynamics in the stylised simulator
    are independent of MM actions (no market impact).

    In training with teacher forcing:
      - Input tokens: [z_0, z_1, ..., z_{N-1}]
      - Targets:      [z_1, z_2, ..., z_N]
      - Loss: NLL averaged over all positions and batch
    """

    def __init__(self, cfg: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WorldModelConfig()
        C = self.cfg

        # Input projection: z_t → d_model (no actions)
        self.input_proj = nn.Linear(C.d_latent, C.d_model)

        # Learned positional encoding
        self.pos_emb = nn.Embedding(C.max_seq, C.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(C) for _ in range(C.n_layers)
        ])

        self.norm_out = nn.LayerNorm(C.d_model)
        self.gmm_head = GMMHead(C)

        # Auxiliary regime classification head
        self.regime_head = nn.Sequential(
            nn.Linear(C.d_model, 64),
            nn.GELU(),
            nn.Dropout(C.dropout),
            nn.Linear(64, C.n_regimes),
        )

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
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        z_seq: torch.Tensor,   # (B, N+1, d_latent) — z_0 .. z_N
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_seq : (B, N+1, d_latent) — latent sequence including target z_N

        Returns:
            pi      : (B, N, K)
            mu      : (B, N, K, d_latent)
            log_sig : (B, N, K, d_latent)

        where outputs at position t predict z_{t+1}.
        """
        B = z_seq.shape[0]
        N = z_seq.shape[1] - 1

        # Input: z_0..z_{N-1} → (B, N, d_latent)
        z_in = z_seq[:, :N, :]                                # (B, N, d_latent)
        x = self.input_proj(z_in)                             # (B, N, d_model)

        # Positional encoding
        pos = torch.arange(N, device=x.device)
        x   = x + self.pos_emb(pos)

        # Causal transformer
        mask = self._causal_mask(N, x.device)
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm_out(x)                                  # (B, N, d_model)

        # Store for regime head
        self._last_hidden = x

        # GMM head
        pi, mu, log_sig = self.gmm_head(x)

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
        """
        B, N, K, D = mu.shape

        z = z_next.unsqueeze(2)                                      # (B, N, 1, D)

        sig      = torch.exp(log_sig)
        log_norm = -0.5 * (
            D * math.log(2 * math.pi)
            + 2 * log_sig.sum(dim=-1)
            + ((z - mu) ** 2 / (sig ** 2 + 1e-8)).sum(dim=-1)
        )                                                            # (B, N, K)

        log_pi   = torch.log(pi + 1e-8)
        log_comp = log_pi + log_norm                                 # (B, N, K)

        log_p = torch.logsumexp(log_comp, dim=-1)                   # (B, N)

        return -log_p.mean()

    def regime_loss(
        self,
        regimes: torch.Tensor,   # (B, N) per-step regime labels
    ) -> tuple[torch.Tensor, float]:
        h = self._last_hidden
        logits = self.regime_head(h)
        B, N, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * N, C),
            regimes.reshape(B * N).long(),
        )
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == regimes).float().mean().item()
        return loss, acc

    @torch.no_grad()
    def diagnostics(
        self,
        pi:      torch.Tensor,
        mu:      torch.Tensor,
        log_sig: torch.Tensor,
        z_next:  torch.Tensor,
        regimes: torch.Tensor,
    ) -> dict[str, float]:
        nll_total = self.nll_loss(pi, mu, log_sig, z_next).item()
        ent = -(pi * torch.log(pi + 1e-8)).sum(dim=-1).mean().item()

        diag = {"nll_total": nll_total, "entropy_pi": ent}

        B, N, K, D = mu.shape
        reg_flat = regimes.reshape(-1)
        for reg_id in [0, 1, 2]:
            mask = (reg_flat == reg_id)
            if mask.sum() == 0:
                continue
            pi_f = pi.reshape(B*N, K)[mask].unsqueeze(1)
            mu_f = mu.reshape(B*N, K, D)[mask].unsqueeze(1)
            ls_f = log_sig.reshape(B*N, K, D)[mask].unsqueeze(1)
            zn_f = z_next.reshape(B*N, D)[mask].unsqueeze(1)
            nll_reg = self.nll_loss(pi_f, mu_f, ls_f, zn_f).item()
            names = ["low_vol", "mid_vol", "high_vol"]
            diag[f"nll_{names[reg_id]}"] = nll_reg

        return diag

    def predict(
        self,
        z_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return GMM means at the last position — used at inference
        for context embedding / DRO stress testing.
        """
        pi, mu, log_sig = self.forward(z_seq)
        return mu[:, -1, :, :]   # (B, K, D)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = WorldModelConfig()
    model = LOBWorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"World model params: {n_params:,}  d_latent={cfg.d_latent}")

    B, N = 16, 20
    z_seq = torch.randn(B, N + 1, cfg.d_latent).to(device)

    pi, mu, log_sig = model(z_seq)
    print(f"pi shape: {pi.shape}  — expected ({B}, {N}, {cfg.n_gmm})")

    z_next = z_seq[:, 1:, :]
    loss = model.nll_loss(pi, mu, log_sig, z_next)
    print(f"NLL loss: {loss.item():.4f}")

    # Regime head test
    regimes = torch.randint(0, 3, (B, N)).to(device)
    reg_loss, reg_acc = model.regime_loss(regimes)
    print(f"Regime loss: {reg_loss.item():.4f}  acc: {reg_acc:.3f}")

    total = loss + cfg.lambda_regime * reg_loss
    total.backward()
    print(f"Backward OK  (total={total.item():.4f})")

    # Diagnostics
    pi2, mu2, ls2 = model(z_seq)
    diag = model.diagnostics(pi2, mu2, ls2, z_next, regimes)
    print(f"Diagnostics: {diag}")

    # Predict
    pred = model.predict(z_seq)
    print(f"Predict shape: {pred.shape}  — expected ({B}, {cfg.n_gmm}, {cfg.d_latent})")
    print("Tutto OK")
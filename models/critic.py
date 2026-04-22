"""
critic.py — Modulo C (parte 1): Value Network V_θ(s).

v2 (redesigned):
  Stato aumentato:
    s_t = [z_t, inv_norm_t, tl_t]  ∈ R^(d_z + 2)

  Architettura a RAMI SEPARATI:
    z_t      → MLP_z     → z_embed    ∈ R^hidden_emb
    [inv,tl] → MLP_scalar → s_embed   ∈ R^hidden_emb
    concat(z_embed, s_embed) → head → V

  Motivazione: v1 (MLP unico) mostrava gradient ratio z/[inv,tl] ≈ 0.06,
  cioè il critico usava quasi esclusivamente le feature scalari. I rami
  separati forzano la rete a costruire un embedding dedicato per z,
  aumentando il leverage del DRO sulla dinamica di mercato.

  Regolarizzazione:
    - Spectral normalization su tutti i Linear layers (coerente con
      l'encoder, Lipschitz bounded per design, zero hyperparameter)
    - GELU activation (non satura, libera dynamic range)
    - Orthogonal init con gain=1.0 (default sano, non 0.01 v1)

  API backward-compatible:
    - gradient_penalty() e estimate_lipschitz() mantenuti per eval
    - forward() accetta (B, d_state) o (B, K, d_state)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snlinear(in_dim: int, out_dim: int, use_spectral: bool = True) -> nn.Module:
    """Linear con spectral_norm opzionale + orthogonal init."""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)
    if use_spectral:
        return spectral_norm(layer)
    return layer


# ---------------------------------------------------------------------------
# ValueNetwork v2 (two-branch)
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """
    V_θ(s) con due rami paralleli (z vs scalars) e head di fusione.

    Args:
        d_state      : dim totale dello stato = d_z + 2
        d_z          : dim di z (default 16)
        hidden       : dim hidden dei rami
        n_layers     : numero di hidden layer per ramo
        hidden_head  : dim del hidden della head finale (default hidden)
        use_spectral : se True, spectral_norm su tutti i Linear
    """

    def __init__(
        self,
        d_state:     int   = 18,
        d_z:         int   = 16,
        d_action:    int   = 0,                        # v3: dim actions (0 = v2 mode)
        hidden:      int   = 128,
        n_layers:    int   = 3,
        hidden_head: int | None = None,
        use_spectral: bool = True,
    ) -> None:
        super().__init__()

        # v3 supporta stato aumentato con actions:
        #   layout: [z (d_z), inv (1), tl (1), actions (d_action)]
        # v2 retrocompat: d_action=0 → stato [z, inv, tl] come prima
        expected_d_state = d_z + 2 + d_action
        assert d_state == expected_d_state, (
            f"d_state ({d_state}) must be d_z+2+d_action ({expected_d_state}). "
            f"Layout expected: [z ({d_z}), inv+tl (2), actions ({d_action})]."
        )
        self.d_state  = d_state
        self.d_z      = d_z
        self.d_action = d_action
        self.n_layers = n_layers
        hidden_head   = hidden_head if hidden_head is not None else hidden

        # -- Branch 1: z → z_embed ---------------------------------------
        z_layers = []
        in_dim = d_z
        for _ in range(n_layers):
            z_layers += [_snlinear(in_dim, hidden, use_spectral), nn.GELU()]
            in_dim = hidden
        self.branch_z = nn.Sequential(*z_layers)

        # -- Branch 2: [inv, tl] → scalar_embed ---------------------------
        hidden_scalar = max(32, hidden // 2)
        s_layers = []
        in_dim = 2
        for _ in range(n_layers):
            s_layers += [_snlinear(in_dim, hidden_scalar, use_spectral), nn.GELU()]
            in_dim = hidden_scalar
        self.branch_scalar = nn.Sequential(*s_layers)

        # -- Branch 3 (v3 only): actions → action_embed -------------------
        # Actions sono la quote A-S osservate: [k_bid_norm, k_ask_norm, q_bid_norm, q_ask_norm]
        # Sono determinate dalla policy (quindi funzione dello stato), ma le passiamo
        # come feature esplicite per evitare che il critico debba re-inferire A-S da z.
        # Questo è state augmentation standard per mitigare parziale osservabilità.
        if d_action > 0:
            hidden_action = max(32, hidden // 2)
            a_layers = []
            in_dim = d_action
            for _ in range(n_layers):
                a_layers += [_snlinear(in_dim, hidden_action, use_spectral), nn.GELU()]
                in_dim = hidden_action
            self.branch_action = nn.Sequential(*a_layers)
        else:
            self.branch_action = None
            hidden_action = 0

        # -- Fusion head --------------------------------------------------
        # spectral_norm solo sui feature extractor, NON sulla head finale.
        fuse_in = hidden + hidden_scalar + hidden_action
        self.fusion = nn.Sequential(
            _snlinear(fuse_in, hidden_head, use_spectral=use_spectral),
            nn.GELU(),
            _snlinear(hidden_head, 1, use_spectral=False),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_flat(self, s: torch.Tensor) -> torch.Tensor:
        """s : (B, d_state) → V : (B,)

        Layout input (v3): [z (d_z), inv (1), tl (1), actions (d_action)]
        Retrocompat (v2):  [z (d_z), inv (1), tl (1)] con d_action=0
        """
        z        = s[:, :self.d_z]                                  # (B, d_z)
        scalars  = s[:, self.d_z : self.d_z + 2]                    # (B, 2)
        z_emb    = self.branch_z(z)                                 # (B, hidden)
        s_emb    = self.branch_scalar(scalars)                      # (B, hidden_scalar)

        if self.branch_action is not None:
            actions = s[:, self.d_z + 2 : self.d_z + 2 + self.d_action]  # (B, d_action)
            a_emb   = self.branch_action(actions)                   # (B, hidden_action)
            fused   = torch.cat([z_emb, s_emb, a_emb], dim=-1)
        else:
            fused = torch.cat([z_emb, s_emb], dim=-1)

        v = self.fusion(fused).squeeze(-1)
        return v

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s : (B, d_state) or (B, K, d_state)
        Returns:
            V : (B,) or (B, K)
        """
        if s.dim() == 3:
            B, K, D = s.shape
            v = self._forward_flat(s.reshape(B * K, D))
            return v.reshape(B, K)
        return self._forward_flat(s)

    # ------------------------------------------------------------------
    # Diagnostics (API compat con v1)
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def estimate_lipschitz(self, s_samples: torch.Tensor, n_pairs: int = 5000) -> float:
        """Stima empirica di L = max ‖∇_s V(s)‖₂ su un campione."""
        was_training = self.training
        self.eval()
        s = s_samples[:min(len(s_samples), n_pairs)].detach().clone().requires_grad_(True)
        v = self.forward(s)
        grads = torch.autograd.grad(v.sum(), s, create_graph=False)[0]
        grad_norms = grads.norm(dim=-1)
        if was_training:
            self.train()
        return float(grad_norms.max().item())

    def gradient_penalty(self, s: torch.Tensor) -> torch.Tensor:
        """
        Gradient penalty: penalizza ‖∇_s V(s)‖² per mantenere Lipschitz bound soft.
        NB: con spectral_norm attiva, la Lipschitz è già bounded; il GP è
        ridondante ma mantenuto per eventuali fine-tune.
        """
        s_gp = s.detach().requires_grad_(True)
        v    = self.forward(s_gp)
        grads = torch.autograd.grad(v.sum(), s_gp, create_graph=True)[0]
        return (grads.norm(dim=-1) ** 2).mean()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    D_Z = 16

    # Test v2 (no actions)
    print("\n--- v2 (2 branches) ---")
    model = ValueNetwork(
        d_state=D_Z + 2, d_z=D_Z, d_action=0,
        hidden=128, n_layers=3, use_spectral=True,
    ).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_p:,}  d_state={model.d_state}  (2 branches)")

    z        = torch.randn(32, D_Z, device=device)
    inv_norm = torch.rand(32, 1, device=device) * 2 - 1
    tl       = torch.rand(32, 1, device=device)
    s        = torch.cat([z, inv_norm, tl], dim=-1)
    v = model(s)
    print(f"V(s) shape: {v.shape}  range: [{v.min().item():.3f}, {v.max().item():.3f}]")

    # Test v3 (with actions)
    print("\n--- v3 (3 branches with actions) ---")
    D_A = 4
    model_v3 = ValueNetwork(
        d_state=D_Z + 2 + D_A, d_z=D_Z, d_action=D_A,
        hidden=128, n_layers=3, use_spectral=True,
    ).to(device)
    n_p_v3 = sum(p.numel() for p in model_v3.parameters())
    print(f"Params: {n_p_v3:,}  d_state={model_v3.d_state}  (3 branches)")

    actions = torch.rand(32, D_A, device=device)       # normalized in [0,1]
    s3 = torch.cat([z, inv_norm, tl, actions], dim=-1)
    v3 = model_v3(s3)
    print(f"V(s) shape: {v3.shape}  range: [{v3.min().item():.3f}, {v3.max().item():.3f}]")

    s_test = torch.randn(1000, D_Z + 2 + D_A, device=device)
    L = model_v3.estimate_lipschitz(s_test)
    print(f"Empirical Lipschitz (init): {L:.4f}")

    # Test gradient flow across 3 branches
    s3.requires_grad_(True)
    v3 = model_v3(s3)
    grads = torch.autograd.grad(v3.sum(), s3)[0]
    g_z   = grads[:, :D_Z].abs().mean().item()
    g_inv = grads[:, D_Z].abs().mean().item()
    g_tl  = grads[:, D_Z + 1].abs().mean().item()
    g_act = grads[:, D_Z + 2:].abs().mean().item()
    print(f"Gradient means @ init: |∇z|={g_z:.4f}  |∇inv|={g_inv:.4f}  "
          f"|∇tl|={g_tl:.4f}  |∇a|={g_act:.4f}")

    # Test 3D input path
    s3_batch = torch.randn(8, 5, D_Z + 2 + D_A, device=device)
    v3_batch = model_v3(s3_batch)
    print(f"V(s) 3D shape: {v3_batch.shape}")
    print("OK")
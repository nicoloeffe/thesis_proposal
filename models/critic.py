"""
critic.py — Modulo C (parte 1): Value Network V_θ(z).

Rete neurale che stima il valore atteso cumulato della strategia A-S
a partire da uno stato latente z:

    V_θ(z) ≈ E[ Σ_t γ^t r_t | z_0 = z ]

Addestrata offline con TD learning su traiettorie latenti generate
dalla policy A-S nel simulatore nominale (wm_dataset.npz).

La differenziabilità di V_θ rispetto a z è il presupposto che permette
al Modulo C di risolvere l'inner problem del DRO via backpropagation:

    x* = argmin_x { V_θ(x) + λ||x - y||² }
    ∇_x { V_θ(x) + λ||x-y||² } = ∇_x V_θ(x) + 2λ(x-y)

Reference: Technical Report — Sezione 3, Fase 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Value Network
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """
    MLP che stima V(z) = E[Σ γ^t r_t | z_0 = z].

    Architettura: 3 layer hidden con LayerNorm e GELU.
    Input: z ∈ R^d_latent
    Output: scalare V(z)

    Nota: la rete deve essere differenziabile rispetto a z per il DRO.
    Non usare operazioni non differenziabili (argmax, round, etc.).
    """

    def __init__(
        self,
        d_latent: int = 32,
        hidden:   int = 256,
        n_layers: int = 3,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent

        layers = []
        in_dim = d_latent
        for i in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, d_latent) or (B, K, d_latent)
        Returns:
            V : (B,) or (B, K)
        """
        shape = z.shape
        if z.dim() == 3:
            B, K, D = shape
            v = self.net(z.reshape(B * K, D))
            return v.reshape(B, K)
        else:
            return self.net(z).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ValueNetwork(d_latent=32, hidden=256, n_layers=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    # Test forward + gradient w.r.t. z (required for DRO inner problem)
    z = torch.randn(16, 32, device=device, requires_grad=True)
    v = model(z)
    print(f"V(z) shape: {v.shape}  — atteso (16,)")
    print(f"V(z) range: [{v.min().item():.4f}, {v.max().item():.4f}]")

    # Gradient w.r.t. z — critical for DRO
    v.sum().backward()
    print(f"∇_z V shape: {z.grad.shape}  — atteso (16, 32)")
    print(f"∇_z V norm:  {z.grad.norm().item():.4f}  (must be > 0)")

    # Test with (B, K, D) input
    z3 = torch.randn(8, 5, 32, device=device)
    v3 = model(z3)
    print(f"V(z) 3D shape: {v3.shape}  — atteso (8, 5)")
    print("OK")
"""
critic.py — Modulo C (parte 1): Value Network V_θ(z).

Architettura: MLP con Tanh activation (Lipschitz per layer = 1 × σ_max(W)).
Nessun spectral_norm hard — il bound Lipschitz è soft (gradient penalty nel
training) e misurato empiricamente. Questo preserva la capacità della rete.

Il DRO inner solver richiede che ‖∇_z V‖ sia limitato ma non necessariamente
≤ 1: basta che sia finito e stimabile, così da calibrare il learning rate
dell'inner loop. Il bound empirico viene salvato nel checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    MLP con Tanh activation, orthogonal init.
    No LayerNorm, no Dropout, no spectral_norm.
    """

    def __init__(
        self,
        d_latent: int = 32,
        hidden:   int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.n_layers = n_layers

        layers = []
        in_dim = d_latent
        for _ in range(n_layers):
            linear = nn.Linear(in_dim, hidden)
            nn.init.orthogonal_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.Tanh()]
            in_dim = hidden

        head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.zeros_(head.bias)
        layers.append(head)

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, d_latent) or (B, K, d_latent)
        Returns:
            V : (B,) or (B, K)
        """
        if z.dim() == 3:
            B, K, D = z.shape
            v = self.net(z.reshape(B * K, D))
            return v.reshape(B, K)
        return self.net(z).squeeze(-1)

    @torch.enable_grad()
    def estimate_lipschitz(self, z_samples: torch.Tensor, n_pairs: int = 5000) -> float:
        """
        Stima empirica di L = max ‖∇_z V(z)‖₂ su un campione.
        Utile per calibrare il DRO inner solver.
        """
        was_training = self.training
        self.eval()
        z = z_samples[:min(len(z_samples), n_pairs)].detach().clone().requires_grad_(True)
        v = self.forward(z)
        grads = torch.autograd.grad(v.sum(), z, create_graph=False)[0]
        grad_norms = grads.norm(dim=-1)
        if was_training:
            self.train()
        return float(grad_norms.max().item())

    def gradient_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """
        Gradient penalty: penalizza ‖∇_z V(z)‖² > target.
        Chiamato nel training loop.
        """
        z_gp = z.detach().requires_grad_(True)
        v = self.forward(z_gp)
        grads = torch.autograd.grad(
            v.sum(), z_gp, create_graph=True
        )[0]
        return (grads.norm(dim=-1) ** 2).mean()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ValueNetwork(d_latent=24, hidden=256, n_layers=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    z = torch.randn(16, 24, device=device, requires_grad=True)
    v = model(z)
    print(f"V(z) shape: {v.shape}")
    print(f"V(z) range: [{v.min().item():.4f}, {v.max().item():.4f}]")

    v.sum().backward()
    grad_norm = z.grad.norm(dim=-1)
    print(f"‖∇_z V‖ mean: {grad_norm.mean().item():.4f}  "
          f"max: {grad_norm.max().item():.4f}")

    # Gradient penalty
    gp = model.gradient_penalty(z.detach())
    print(f"Gradient penalty: {gp.item():.4f}")

    # Empirical Lipschitz
    z_test = torch.randn(1000, 24, device=device)
    L = model.estimate_lipschitz(z_test)
    print(f"Empirical Lipschitz (init): {L:.4f}")

    # 3D input
    z3 = torch.randn(8, 5, 24, device=device)
    v3 = model(z3)
    print(f"V(z) 3D shape: {v3.shape}")
    print("OK")
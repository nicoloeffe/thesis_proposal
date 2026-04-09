"""
dro.py — Modulo C: Distributionally Robust Optimization su Wasserstein.

Implementa il duale Wasserstein one-step a tre livelli:

  inf_{Q ∈ U_ε} E_Q[V(z')] = sup_{λ≥0} { -λε + E_{y~P} [ inf_x { V(x) + λ c(x,y) } ] }

dove c(x,y) è il costo di trasporto (L2 o Mahalanobis).

Fase 1 — Campionamento nominale:
    y_1..y_m ~ GMM: per ogni componente k, campiona n_s punti da N(μ_k, σ_k²).
    Produce K×n_s atomi con pesi π_k/n_s.

Fase 2 — Inner solver (per ogni y_j):
    x*(y_j, λ) = argmin_x { V(x) + λ · c(x, y_j) }
    Con costo Mahalanobis: c(x,y) = Σ_d (x_d - y_d)² / σ²_kd
    Learning rate adattivo per dimensione: lr_d = min(base_lr, σ²_kd / (2λ))
    Trust radius in unità di σ: |x_d - y_d| ≤ R · σ_kd

Fase 3 — Bisection su λ:
    Trova λ* tale che transport(λ*) = ε.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DROConfig:
    # Inner optimization
    inner_steps:    int   = 100
    inner_lr:       float = 0.05

    # Outer optimization — bisection on λ
    outer_steps:    int   = 30
    lambda_init:    float = 50.0

    # Trust region in units of σ per dimension
    trust_radius_sigma: float = 3.0

    # Nominal sampling
    n_samples_per_component: int = 3

    # Cost type
    cost_type: str = "mahalanobis"   # "mahalanobis" or "l2"

    # General
    gamma: float = 0.95


# ---------------------------------------------------------------------------
# Inner problem solver
# ---------------------------------------------------------------------------

class InnerSolver:
    """
    Solves the inner problem for each nominal point y_j:

        x*(y_j, λ) = argmin_x { V(x) + λ · c(x, y_j) }

    Cost function:
      - L2:          c(x,y) = Σ_d (x_d - y_d)²
      - Mahalanobis: c(x,y) = Σ_d (x_d - y_d)² / σ²_d

    Adaptive lr (Mahalanobis):
        Per-dimension lr_d = min(base_lr, σ²_d / (2λ + ε))
        The Hessian of the penalty along dim d is 2λ/σ²_d, so the
        optimal GD step is σ²_d/(2λ). Capped at base_lr.

    Trust region:
        |x_d - y_d| ≤ R · σ_d   (Mahalanobis)
        |x_d - y_d| ≤ R          (L2)
    """

    def __init__(self, critic: nn.Module, cfg: DROConfig) -> None:
        self.critic = critic
        self.cfg = cfg

    def solve(
        self,
        y: torch.Tensor,          # (m, D) — nominal points
        lam: float,
        sigma: torch.Tensor,      # (m, D) — per-point σ (from GMM component)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        C = self.cfg
        R = C.trust_radius_sigma
        mahal = (C.cost_type == "mahalanobis")

        # Precompute inverse variance and trust bounds
        if mahal:
            sig2 = sigma ** 2 + 1e-8                    # (m, D)
            inv_sig2 = 1.0 / sig2                       # (m, D)
            # Per-dim adaptive lr: σ²_d / (2λ + 1)
            lr_per_dim = torch.clamp(sig2 / (2.0 * lam + 1.0), max=C.inner_lr)  # (m, D)
            trust_bound = R * sigma                     # (m, D)
        else:
            inv_sig2 = None
            lr_scalar = min(C.inner_lr, 0.5 / (2.0 * lam + 1.0))
            lr_per_dim = None
            trust_bound = R  # scalar

        x = y.clone().detach().requires_grad_(True)
        y_det = y.detach()

        for step in range(C.inner_steps):
            v = self.critic(x)                          # (m,)

            if mahal:
                penalty = lam * (((x - y_det) ** 2) * inv_sig2).sum(dim=-1)  # (m,)
            else:
                penalty = lam * ((x - y_det) ** 2).sum(dim=-1)               # (m,)

            obj = v + penalty

            grad = torch.autograd.grad(
                obj.sum(), x, create_graph=False
            )[0]                                        # (m, D)

            # Gradient descent with adaptive lr
            if mahal:
                x_new = x - lr_per_dim * grad
            else:
                x_new = x - lr_scalar * grad

            # Trust region clamp
            x_new = torch.clamp(x_new, y_det - trust_bound, y_det + trust_bound)

            x = x_new.detach().requires_grad_(True)

        with torch.no_grad():
            v_star = self.critic(x)

        return x.detach(), v_star


# ---------------------------------------------------------------------------
# DRO Module
# ---------------------------------------------------------------------------

class WassersteinDRO:
    """
    Full three-level Wasserstein DRO.

    Given GMM parameters (π, μ, log_σ), critic V_θ, and radius ε:
      1. Sample nominal points from GMM
      2. For each point, solve inner problem
      3. Bisect on λ to satisfy transport = ε
    """

    def __init__(
        self,
        critic: nn.Module,
        cfg: DROConfig | None = None,
    ) -> None:
        self.cfg = cfg or DROConfig()
        self.critic = critic
        self.inner_solver = InnerSolver(critic, self.cfg)

    def _sample_nominal(
        self,
        pi: torch.Tensor,        # (K,)
        mu: torch.Tensor,        # (K, D)
        log_sig: torch.Tensor,   # (K, D)
        seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample n_s points per component from the GMM.

        Uses a fixed seed so that the same GMM parameters always produce
        the same nominal points — critical for consistent V_nom across ε.

        Returns:
            y       : (K*n_s, D)  — sampled points
            weights : (K*n_s,)    — weights π_k/n_s per point
            sigma   : (K*n_s, D)  — σ of the parent component (for Mahalanobis)
        """
        K, D = mu.shape
        n_s = self.cfg.n_samples_per_component
        sig = torch.exp(log_sig)  # (K, D)

        if n_s == 0:
            # Deterministic: use centroids only
            return mu.clone(), pi.clone(), sig.clone()

        # Fixed seed for reproducibility across ε values
        gen = torch.Generator(device=mu.device).manual_seed(seed)

        # Sample from each component
        # y_kj = μ_k + σ_k * ε_kj,  ε_kj ~ N(0, I)
        eps = torch.randn(K, n_s, D, device=mu.device, generator=gen)  # (K, n_s, D)
        y = mu.unsqueeze(1) + sig.unsqueeze(1) * eps             # (K, n_s, D)
        y = y.reshape(K * n_s, D)

        # Weights: π_k / n_s for each sample from component k
        weights = (pi / n_s).unsqueeze(1).expand(K, n_s).reshape(K * n_s)

        # Each sample inherits σ from its parent component
        sigma = sig.unsqueeze(1).expand(K, n_s, D).reshape(K * n_s, D)

        return y, weights, sigma

    def _compute_transport(
        self,
        y: torch.Tensor,          # (m, D)
        weights: torch.Tensor,    # (m,)
        sigma: torch.Tensor,      # (m, D)
        lam: float,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Solve inner problem at given λ, return weighted transport."""
        x_star, v_star = self.inner_solver.solve(y, lam, sigma)

        if self.cfg.cost_type == "mahalanobis":
            sig2 = sigma ** 2 + 1e-8
            transport_per_point = (((x_star - y) ** 2) / sig2).sum(dim=-1)  # (m,)
        else:
            transport_per_point = ((x_star - y) ** 2).sum(dim=-1)           # (m,)

        w_transport = (weights * transport_per_point).sum().item()
        return w_transport, x_star, v_star

    def solve_one_step(
        self,
        pi: torch.Tensor,        # (K,)
        mu: torch.Tensor,        # (K, D)
        log_sig: torch.Tensor,   # (K, D)
        epsilon: float,
    ) -> dict:
        """
        Full three-level optimization for a single transition.

        Bisection on λ: transport(λ) is monotonically decreasing in λ,
        so we find λ* such that transport(λ*) = ε.
        """
        C = self.cfg

        # Phase 1: sample nominal points
        y, weights, sigma = self._sample_nominal(pi, mu, log_sig)

        # Nominal expected value
        with torch.no_grad():
            v_nominal = (weights * self.critic(y)).sum().item()

        # Special case: ε = 0
        if epsilon <= 1e-10:
            with torch.no_grad():
                v_star = self.critic(y)
            return {
                "x_star":       y.clone(),
                "v_star":       v_star,
                "v_robust":     v_nominal,
                "v_nominal":    v_nominal,
                "lambda_star":  float("inf"),
                "dual_value":   v_nominal,
                "transport":    0.0,
            }

        # Phase 3: bisection on λ
        lam_low  = 1e-4
        lam_high = C.lambda_init

        # Expand upper bound if needed
        t_high, _, _ = self._compute_transport(y, weights, sigma, lam_high)
        while t_high > epsilon and lam_high < 1e6:
            lam_high *= 2
            t_high, _, _ = self._compute_transport(y, weights, sigma, lam_high)

        # Expand lower bound if needed
        t_low, _, _ = self._compute_transport(y, weights, sigma, lam_low)
        if t_low < epsilon:
            # Adversary can't use full budget — V too flat locally
            x_star, v_star = self.inner_solver.solve(y, lam_low, sigma)
            if self.cfg.cost_type == "mahalanobis":
                transport_k = (((x_star - y) ** 2) / (sigma ** 2 + 1e-8)).sum(dim=-1)
            else:
                transport_k = ((x_star - y) ** 2).sum(dim=-1)
            w_transport = (weights * transport_k).sum().item()
            v_robust = (weights * v_star).sum().item()
            dual_value = -lam_low * epsilon + (weights * (v_star + lam_low * transport_k)).sum().item()
            return {
                "x_star":       x_star,
                "v_star":       v_star,
                "v_robust":     v_robust,
                "v_nominal":    v_nominal,
                "lambda_star":  lam_low,
                "dual_value":   dual_value,
                "transport":    w_transport,
            }

        # Bisection loop
        for step in range(C.outer_steps):
            lam_mid = (lam_low + lam_high) / 2
            t_mid, _, _ = self._compute_transport(y, weights, sigma, lam_mid)

            if t_mid > epsilon:
                lam_low = lam_mid
            else:
                lam_high = lam_mid

            if abs(t_mid - epsilon) / (epsilon + 1e-8) < 0.01:
                break

        # Final solve at converged λ*
        lam_star = (lam_low + lam_high) / 2
        x_star, v_star = self.inner_solver.solve(y, lam_star, sigma)
        if self.cfg.cost_type == "mahalanobis":
            transport_k = (((x_star - y) ** 2) / (sigma ** 2 + 1e-8)).sum(dim=-1)
        else:
            transport_k = ((x_star - y) ** 2).sum(dim=-1)
        w_transport = (weights * transport_k).sum().item()
        v_robust = (weights * v_star).sum().item()
        dual_value = -lam_star * epsilon + (weights * (v_star + lam_star * transport_k)).sum().item()

        return {
            "x_star":       x_star,
            "v_star":       v_star,
            "v_robust":     v_robust,
            "v_nominal":    v_nominal,
            "lambda_star":  lam_star,
            "dual_value":   dual_value,
            "transport":    w_transport,
        }

    def robust_bellman_backup(
        self,
        reward: float,
        pi: torch.Tensor,
        mu: torch.Tensor,
        log_sig: torch.Tensor,
        epsilon: float,
    ) -> dict:
        """
        Robust Bellman backup:
            y_rob = r + γ · E_Q*[V(x*)]
        """
        result = self.solve_one_step(pi, mu, log_sig, epsilon)
        y_rob  = reward + self.cfg.gamma * result["v_robust"]

        result["y_rob"]     = y_rob
        result["y_nominal"] = reward + self.cfg.gamma * result["v_nominal"]
        return result


# ---------------------------------------------------------------------------
# Stress test runner
# ---------------------------------------------------------------------------

class StressTestRunner:
    """
    Runs DRO stress testing over trajectories for multiple ε values.
    """

    def __init__(
        self,
        world_model: nn.Module,
        critic: nn.Module,
        cfg: DROConfig | None = None,
    ) -> None:
        self.world_model = world_model
        self.critic = critic
        self.cfg = cfg or DROConfig()
        self.dro = WassersteinDRO(critic, self.cfg)

    @torch.no_grad()
    def _get_gmm_at_t(
        self,
        z_seq: torch.Tensor,    # (1, N+1, D)
        a_seq: torch.Tensor,    # (1, N, 3)
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi, mu, log_sig = self.world_model(z_seq, a_seq)
        return pi[0, t, :], mu[0, t, :, :], log_sig[0, t, :, :]

    def run_trajectory(
        self,
        z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        rewards: torch.Tensor,
        epsilon: float,
    ) -> dict:
        N = a_seq.shape[1]
        gamma = self.cfg.gamma

        y_rob_list, y_nom_list = [], []
        v_robust_list, v_nominal_list = [], []
        lambda_list, transport_list = [], []

        for t in range(N):
            pi, mu, log_sig = self._get_gmm_at_t(z_seq, a_seq, t)
            r_t = rewards[0, t].item()

            with torch.enable_grad():
                result = self.dro.robust_bellman_backup(
                    r_t, pi, mu, log_sig, epsilon
                )

            y_rob_list.append(result["y_rob"])
            y_nom_list.append(result["y_nominal"])
            v_robust_list.append(result["v_robust"])
            v_nominal_list.append(result["v_nominal"])
            lambda_list.append(result["lambda_star"])
            transport_list.append(result["transport"])

        v_rob = sum(gamma ** t * y_rob_list[t] for t in range(N))
        v_nom = sum(gamma ** t * y_nom_list[t] for t in range(N))

        return {
            "v_rob": v_rob, "v_nominal": v_nom,
            "y_rob": y_rob_list, "y_nominal": y_nom_list,
            "v_robust": v_robust_list, "v_nominal_st": v_nominal_list,
            "lambdas": lambda_list, "transports": transport_list,
        }

    def run_stress_test(
        self,
        sequences: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        epsilons: list[float],
        n_trajectories: int = 50,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict:
        M = sequences.shape[0]
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(M, generator=rng)[:n_trajectories]

        results = {}
        for eps in epsilons:
            if verbose:
                print(f"\n  ε = {eps:.4f}  ", end="", flush=True)

            traj_v_rob, traj_v_nom = [], []
            traj_lambdas, traj_transports = [], []

            for i, traj_idx in enumerate(idx):
                z_seq = sequences[traj_idx:traj_idx+1]
                a_seq = actions[traj_idx:traj_idx+1]
                rew   = rewards[traj_idx:traj_idx+1]

                res = self.run_trajectory(z_seq, a_seq, rew, epsilon=eps)
                traj_v_rob.append(res["v_rob"])
                traj_v_nom.append(res["v_nominal"])
                traj_lambdas.extend(res["lambdas"])
                traj_transports.extend(res["transports"])

                if verbose and (i + 1) % 10 == 0:
                    print(f"[{i+1}/{n_trajectories}]", end=" ", flush=True)

            v_rob_arr = torch.tensor(traj_v_rob)
            v_nom_arr = torch.tensor(traj_v_nom)
            v_rob_mean = v_rob_arr.mean().item()
            v_nom_mean = v_nom_arr.mean().item()
            degradation = (v_nom_mean - v_rob_mean) / (abs(v_nom_mean) + 1e-8)

            results[eps] = {
                "v_rob_mean":     v_rob_mean,
                "v_rob_std":      v_rob_arr.std().item(),
                "v_nominal_mean": v_nom_mean,
                "degradation":    degradation,
                "mean_lambda":    float(torch.tensor(traj_lambdas).mean()),
                "mean_transport": float(torch.tensor(traj_transports).mean()),
                "per_traj_v_rob": traj_v_rob,
            }

            if verbose:
                print(f"  V_rob={v_rob_mean:.4f}  V_nom={v_nom_mean:.4f}  "
                      f"deg={degradation*100:.1f}%  λ*={results[eps]['mean_lambda']:.3f}")

        return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from models.critic import ValueNetwork
    critic = ValueNetwork(d_latent=24).to(device)
    critic.eval()

    K, D = 5, 24
    pi      = torch.softmax(torch.randn(K), dim=0).to(device)
    mu      = torch.randn(K, D).to(device) * 0.5
    log_sig = torch.full((K, D), -1.0).to(device)  # σ ≈ 0.37

    print("\n--- Test Mahalanobis ---")
    cfg_m = DROConfig(inner_steps=50, outer_steps=20, cost_type="mahalanobis",
                      n_samples_per_component=3, trust_radius_sigma=3.0)
    dro_m = WassersteinDRO(critic, cfg_m)
    res_m = dro_m.robust_bellman_backup(0.01, pi, mu, log_sig, epsilon=0.1)
    print(f"  y_rob={res_m['y_rob']:.6f}  y_nom={res_m['y_nominal']:.6f}")
    print(f"  λ*={res_m['lambda_star']:.4f}  transport={res_m['transport']:.6f}")
    print(f"  n_atoms={res_m['x_star'].shape[0]}  (K={K} × n_s={cfg_m.n_samples_per_component})")

    print("\n--- Test L2 ---")
    cfg_l = DROConfig(inner_steps=50, outer_steps=20, cost_type="l2",
                      n_samples_per_component=0, trust_radius_sigma=0.5)
    dro_l = WassersteinDRO(critic, cfg_l)
    res_l = dro_l.robust_bellman_backup(0.01, pi, mu, log_sig, epsilon=0.1)
    print(f"  y_rob={res_l['y_rob']:.6f}  y_nom={res_l['y_nominal']:.6f}")
    print(f"  λ*={res_l['lambda_star']:.4f}  transport={res_l['transport']:.6f}")
    print(f"  n_atoms={res_l['x_star'].shape[0]}  (centroids only)")

    print("\n--- Test ε=0 ---")
    res_0 = dro_m.robust_bellman_backup(0.01, pi, mu, log_sig, epsilon=0.0)
    print(f"  y_rob={res_0['y_rob']:.6f}  y_nom={res_0['y_nominal']:.6f}")
    assert abs(res_0['y_rob'] - res_0['y_nominal']) < 1e-6, "ε=0 should give nominal"

    print("\nOK")
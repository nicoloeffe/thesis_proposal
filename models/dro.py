"""
dro.py — Modulo C: Distributionally Robust Optimization su Wasserstein.

Implementa il duale Wasserstein one-step a tre livelli di ottimizzazione:

  inf_{Q ∈ U_ε} E_Q[V(z')] = sup_{λ≥0} { -λε + E_{y~P} [ inf_x { V(x) + λ‖x-y‖² } ] }

Fase 1 (middle) — Campionamento punti nominali:
    y_1..y_m ~ GMM del world model  (oppure centroidi pesati)

Fase 2 (inner) — Ottimizzazione avversariale per ogni y_j:
    x*(y_j, λ) = argmin_x { V(x) + λ‖x - y_j‖² }
    Risolto con M passi di gradient descent su x.

Fase 3 (outer) — Aggiornamento di λ:
    ∇_λ ĝ(λ) = -ε + (1/m) Σ_j ‖x*_j - y_j‖²
    λ ← max(0, λ + α · ∇_λ ĝ)

Output: backup robusto one-step e stati avversariali.

Reference: Technical Report — Sezione 3.
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
    # Inner optimization (Phase 2)
    inner_steps:    int   = 100      # M: gradient steps per centroid
    inner_lr:       float = 0.05     # η_base: base step size (adapted per λ)

    # Outer optimization (Phase 3) — bisection on λ
    outer_steps:    int   = 30       # bisection iterations
    lambda_init:    float = 50.0     # upper bound for bisection search

    # Trust region: max ‖x - y‖ per dimension.
    # Prevents inner solver from reaching OOD regions where the critic
    # extrapolates unreliably. Calibrated on training z std (~1.0 per dim).
    trust_radius:   float = 3.0      # max per-dim deviation from centroid

    # General
    gamma:          float = 0.95     # discount factor for Bellman backup


# ---------------------------------------------------------------------------
# Inner problem solver
# ---------------------------------------------------------------------------

class InnerSolver:
    """
    Solves the inner problem for each nominal point y_k:

        x*(y_k, λ) = argmin_x { V(x) + λ‖x - y_k‖² }

    via gradient descent on x, starting from x = y_k.

    Adaptive learning rate:
        lr = min(base_lr, 0.5 / (2λ + 1))
      The inner objective has curvature ≥ 2λ from the quadratic penalty.
      Optimal GD step for a function with Hessian H is ~1/‖H‖.
      With 2λI dominating, lr ≈ 1/(2λ) is near-optimal. Cap at base_lr.

    Trust region:
        After each step, clip |x_d - y_d| ≤ R per dimension.
        Prevents the critic from being evaluated in OOD regions where
        it extrapolates unreliably (arbitrary negative values).
    """

    def __init__(self, critic: nn.Module, cfg: DROConfig) -> None:
        self.critic = critic
        self.cfg = cfg

    def solve(
        self,
        y: torch.Tensor,          # (K, D) — nominal centroids
        lam: float,               # current λ
    ) -> tuple[torch.Tensor, torch.Tensor]:
        R = self.cfg.trust_radius

        # Adaptive lr: scale with curvature of the quadratic penalty
        lr = min(self.cfg.inner_lr, 0.5 / (2.0 * lam + 1.0))

        x = y.clone().detach().requires_grad_(True)
        y_det = y.detach()

        for step in range(self.cfg.inner_steps):
            v = self.critic(x)                                    # (K,)
            penalty = lam * ((x - y_det) ** 2).sum(dim=-1)       # (K,)
            obj = v + penalty

            grad = torch.autograd.grad(
                obj.sum(), x, create_graph=False
            )[0]                                                  # (K, D)

            # Gradient descent step
            x_new = x - lr * grad

            # Trust region: clip per-dimension deviation from y
            x_new = torch.clamp(x_new, y_det - R, y_det + R)

            x = x_new.detach().requires_grad_(True)

        with torch.no_grad():
            v_star = self.critic(x)

        return x.detach(), v_star


# ---------------------------------------------------------------------------
# DRO Module (full three-level optimization)
# ---------------------------------------------------------------------------

class WassersteinDRO:
    """
    Distributionally Robust Optimization module.

    Given:
      - World model GMM parameters (π, μ, log_σ) at a single timestep
      - Critic V_θ(z)
      - Ambiguity set radius ε

    Computes:
      - Adversarial next states x*
      - Robust Bellman backup: y_rob = r + γ * V(x*_worst)
      - Dual function value ĝ(λ*)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 1: Get nominal points and their weights.

        Uses GMM centroids μ_k with weights π_k.
        Deterministic, zero MC variance, and sufficient for K=5 in d=32
        because the GMM components already capture the distribution structure.

        Returns:
            y       : (K, D) — centroid positions
            weights : (K,)   — mixture weights π_k (sum to 1)
        """
        return mu.clone(), pi.clone()

    def solve_one_step(
        self,
        pi: torch.Tensor,        # (K,)
        mu: torch.Tensor,        # (K, D)
        log_sig: torch.Tensor,   # (K, D)
        epsilon: float,           # Wasserstein radius
    ) -> dict:
        """
        Full three-level optimization for a single transition.

        The dual says:
          inf_Q E_Q[V] = sup_λ { -λε + Σ_k π_k * inf_x { V(x) + λ‖x-μ_k‖² } }

        Outer loop: BISECTION on λ.
        At the optimum, complementary slackness gives:
            transport(λ*) = ε
        transport(λ) is monotonically decreasing in λ (higher penalty → less movement),
        so bisection finds λ* in O(log₂(range/tol)) steps. No learning rate to tune.

        Special case ε=0: λ→∞, x*=y, return nominal value directly.
        """
        C = self.cfg

        # Phase 1: get centroids and weights
        y, weights = self._sample_nominal(pi, mu, log_sig)  # (K, D), (K,)

        # Nominal expected value: Σ π_k V(μ_k)
        with torch.no_grad():
            v_nominal = (weights * self.critic(y)).sum().item()

        # Special case: ε = 0 → no perturbation allowed
        if epsilon <= 1e-10:
            return {
                "x_star":       y.clone(),
                "v_star":       self.critic(y).detach(),
                "v_robust":     v_nominal,
                "v_nominal":    v_nominal,
                "lambda_star":  float("inf"),
                "dual_value":   v_nominal,
                "transport":    0.0,
            }

        # Helper: solve inner problem for a given λ and return weighted transport
        def compute_transport(lam: float) -> tuple[float, torch.Tensor, torch.Tensor]:
            x_star, v_star = self.inner_solver.solve(y, lam)
            transport = ((x_star - y) ** 2).sum(dim=-1)       # (K,)
            w_transport = (weights * transport).sum().item()   # scalar
            return w_transport, x_star, v_star

        # Bisection: find λ* such that transport(λ*) = ε
        # λ_low → large transport (small penalty), λ_high → small transport
        lam_low  = 1e-4
        lam_high = C.lambda_init   # start with configured upper bound

        # Expand upper bound if needed (transport at lam_high should be < ε)
        t_high, _, _ = compute_transport(lam_high)
        while t_high > epsilon and lam_high < 1e6:
            lam_high *= 2
            t_high, _, _ = compute_transport(lam_high)

        # Expand lower bound if needed (transport at lam_low should be > ε)
        t_low, _, _ = compute_transport(lam_low)
        if t_low < epsilon:
            # Even with minimal penalty, transport < ε
            # This means the adversary can't use all the budget — V is too flat
            # Return the result at lam_low (maximum perturbation we can achieve)
            x_star, v_star = self.inner_solver.solve(y, lam_low)
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
        best_result = None
        for step in range(C.outer_steps):
            lam_mid = (lam_low + lam_high) / 2
            t_mid, x_star, v_star = compute_transport(lam_mid)

            if t_mid > epsilon:
                lam_low = lam_mid    # transport too large → increase λ
            else:
                lam_high = lam_mid   # transport too small → decrease λ

            # Check convergence
            if abs(t_mid - epsilon) / (epsilon + 1e-8) < 0.01:
                break

        # Final solve at converged λ
        lam_star = (lam_low + lam_high) / 2
        x_star, v_star = self.inner_solver.solve(y, lam_star)
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
        Compute the robust Bellman backup for a single transition:

            y_rob = r + γ * Σ_k π_k V(x*_k)

        where x*_k are the adversarial states obtained from the dual.
        The robust E[V] is the weighted mean, as per the duality derivation.

        Returns dict with y_rob plus all DRO diagnostics.
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
    Runs DRO stress testing over a set of trajectories for multiple ε values.

    For each trajectory and each timestep t:
      1. Get GMM from world model at position t
      2. Run DRO to get robust backup
      3. Accumulate V_rob(ε) = E_τ [ Σ_t γ^t y_rob_t(ε) ]
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
        t: int,                  # position in sequence
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get GMM parameters at position t from the world model.
        Returns: pi (K,), mu (K,D), log_sig (K,D)
        """
        pi, mu, log_sig = self.world_model(z_seq, a_seq)
        # Extract position t: (1, K) -> (K,), (1, K, D) -> (K, D)
        return (
            pi[0, t, :],
            mu[0, t, :, :],
            log_sig[0, t, :, :],
        )

    def run_trajectory(
        self,
        z_seq: torch.Tensor,     # (1, N+1, D)
        a_seq: torch.Tensor,     # (1, N, 3)
        rewards: torch.Tensor,   # (1, N)
        epsilon: float,
    ) -> dict:
        """
        Run DRO on every timestep of a single trajectory.
        Returns per-step results and trajectory-level V_rob.
        """
        N = a_seq.shape[1]
        gamma = self.cfg.gamma

        y_rob_list = []
        y_nom_list = []
        v_robust_list = []
        v_nominal_list = []
        lambda_list = []
        transport_list = []

        for t in range(N):
            pi, mu, log_sig = self._get_gmm_at_t(z_seq, a_seq, t)
            r_t = rewards[0, t].item()

            # Need gradients for inner optimization
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

        # Discounted sums
        v_rob = sum(gamma ** t * y_rob_list[t] for t in range(N))
        v_nom = sum(gamma ** t * y_nom_list[t] for t in range(N))

        return {
            "v_rob":        v_rob,
            "v_nominal":    v_nom,
            "y_rob":        y_rob_list,
            "y_nominal":    y_nom_list,
            "v_robust":     v_robust_list,
            "v_nominal_st": v_nominal_list,
            "lambdas":      lambda_list,
            "transports":   transport_list,
        }

    def run_stress_test(
        self,
        sequences: torch.Tensor,    # (M, N+1, D)
        actions: torch.Tensor,      # (M, N, 3)
        rewards: torch.Tensor,      # (M, N)
        epsilons: list[float],
        n_trajectories: int = 50,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict:
        """
        Run the full stress test: compute V_rob(ε) for each ε.

        Returns:
            results[ε] = {
                "v_rob_mean", "v_rob_std",
                "v_nominal_mean",
                "degradation",            # (v_nom - v_rob) / |v_nom|
                "mean_lambda", "mean_transport",
                "per_traj_v_rob",
            }
        """
        M = sequences.shape[0]
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(M, generator=rng)[:n_trajectories]

        results = {}

        for eps in epsilons:
            if verbose:
                print(f"\n  ε = {eps:.4f}  ", end="", flush=True)

            traj_v_rob = []
            traj_v_nom = []
            traj_lambdas = []
            traj_transports = []

            for i, traj_idx in enumerate(idx):
                z_seq = sequences[traj_idx:traj_idx+1]    # (1, N+1, D)
                a_seq = actions[traj_idx:traj_idx+1]      # (1, N, 3)
                rew   = rewards[traj_idx:traj_idx+1]      # (1, N)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dummy critic
    from models.critic import ValueNetwork
    critic = ValueNetwork(d_latent=32).to(device)
    critic.eval()

    cfg = DROConfig(inner_steps=10, outer_steps=5)
    dro = WassersteinDRO(critic, cfg)

    # Dummy GMM params
    K, D = 5, 32
    pi      = torch.softmax(torch.randn(K), dim=0).to(device)
    mu      = torch.randn(K, D).to(device) * 0.1
    log_sig = torch.full((K, D), -2.0).to(device)

    result = dro.robust_bellman_backup(
        reward=0.01, pi=pi, mu=mu, log_sig=log_sig, epsilon=0.1
    )
    print(f"y_rob     : {result['y_rob']:.6f}")
    print(f"y_nominal : {result['y_nominal']:.6f}")
    print(f"v_robust  : {result['v_robust']:.6f}")
    print(f"v_nominal : {result['v_nominal']:.6f}")
    print(f"λ*        : {result['lambda_star']:.4f}")
    print(f"transport : {result['transport']:.6f}")
    print("OK")
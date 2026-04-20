"""
Default configuration for the Market Making simulator.

Design principles (post-refactor):
  - Each parameter governs one distinct economic mechanism.
  - Depth emerges endogenously from the LO/cancellation balance (no external
    baseline target). At each level, the steady-state volume is approximately:
        V*_ℓ ≈ (λ_lo · E[LO size] · e^(-α·ℓ)) / (λ_cancel_per_share + sweep_ℓ)
  - The MM's risk aversion (γ) is a constant property of the agent, not of
    the regime. Regime differentiation of quotes is driven by σ and κ.
  - The inventory penalty (α_inventory) is a constant preference of the MM.

Empirical ratios between calm and stressed conditions (Cont-Stoikov-Talreja 2010,
Huang-Rosenbaum 2015):
  - Book depth varies by ~3-5x
  - MO arrival rate varies by ~1.5-2x
  - Cancellation rate per share varies by ~3-4x
  - LO arrival rate is relatively stable
"""
from dataclasses import dataclass


@dataclass
class EnvConfig:
    # --- Market structure ---
    tick_size: float = 0.01
    L: int = 10
    initial_mid: float = 100.0

    # --- Mid-price dynamics ---
    sigma_mid: float = 0.02

    # --- Market microstructure ---
    p_informed: float = 0.2
    lambda_mo_buy: float = 0.5
    lambda_mo_sell: float = 0.5
    mo_size_lambda: float = 3.0
    lambda_lo_bid: float = 1.0
    lambda_lo_ask: float = 1.0
    # Queue-reactive cancellation rate, per share, per step.
    # Replaces the former baseline_vol / bid_mult / ask_mult mechanism.
    # Expected cancellations at level ℓ this step ≈ Poisson(λ_cancel_per_share · V_ℓ_bg),
    # where V_ℓ_bg is background (non-MM) volume at that level.
    # Equilibrium depth emerges from the balance λ_lo vs λ_cancel_per_share.
    lambda_cancel_per_share: float = 0.16
    lo_alpha: float = 0.4
    # LO intensity profile — suppression factor at the best level (ℓ = 0).
    # Profile: shape[0] = lo_best_supply, shape[ℓ] = exp(-α·ℓ) for ℓ ≥ 1,
    # then normalised so that Σ shape = Σ exp(-α·ℓ) (aggregate arrival rate
    # preserved). When lo_best_supply < exp(-α) the profile has an internal
    # peak at ℓ = 1 (BMP02 hump).
    # Economic interpretation: f_0 < 1 encodes trader aversion to quoting at
    # the best (adverse-selection premium). Scales inversely with p_informed
    # — Glosten-Milgrom (1985), O'Hara (1995).
    # Parameter roles are CLEANLY SEPARATED:
    #   lo_alpha       → dispersion of liquidity in deeper levels (ℓ ≥ 1)
    #   lo_best_supply → strength of the L0 suppression (hump emergence)
    lo_best_supply: float = 0.6

    # --- Episode ---
    T_max: int = 1000
    # Silent warmup (no MM quotes) executed at the start of reset() so that the
    # book reaches its endogenous equilibrium with the current regime before
    # the episode starts. Avoids the regime-agnostic initial transient.
    warmup_steps: int = 150

    # --- Reward ---
    # Constant across regimes: represents the MM's internal risk preference
    # (not a market property). The MTM risk is already state-dependent and
    # scales with σ² endogenously.
    alpha_inventory: float = 1e-3

    # --- Dataset generation ---
    N_episodes: int = 300
    dataset_path: str = "dataset.npz"

    # --- Avellaneda-Stoikov policy ---
    # γ is a property of the MM's risk aversion, NOT of the market regime.
    # Kept constant across regimes. Regime-level quote differentiation is
    # produced by σ (volatility channel) and κ (microstructure channel).
    as_gamma: float = 5.0
    as_kappa: float = 50.0
    as_mix_ratio: float = 1.0   # A-S only in the dataset (no random policy)
    # Mild inventory tilt on quote size (intra-regime MM behaviour).
    # q_bid = q0 · exp(η · inv),  q_ask = q0 · exp(-η · inv), with q0 = 0.15 · depth_mean.
    # No upper clamp (only a floor at 1) so the size is governed by a single law.
    # Set to 0.0 to disable the tilt (pure book-proportional size).
    as_eta: float = -0.015

    # --- Mixed regime episodes ---
    mixed_regime_frac: float = 0.30
    max_switches: int = 2
    switch_warmup_frac: float = 0.15
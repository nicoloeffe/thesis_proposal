"""
Default configuration for the Market Making simulator.

The key ratio that determines book depth is:
    Q_equilibrium ~ λ_lo / (λ_mo * sweep_fraction + λ_cancel)

Between calm and stressed conditions, empirical LOB studies
(Cont-Stoikov-Talreja 2010, Huang-Rosenbaum 2015) show:
  - Book depth varies by ~3-5x, not 100x
  - MO arrival rate varies by ~1.5-2x
  - Cancellation rate varies by ~1.5-2x
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
    lambda_cancel_bid: float = 0.5
    lambda_cancel_ask: float = 0.5
    lo_alpha: float = 0.4       

    # --- Episode ---
    T_max: int = 1000

    # --- Reward ---
    alpha_inventory: float = 1e-3   # penalità inventory nel reward del MM

    # --- Dataset generation ---
    N_episodes: int = 300
    dataset_path: str = "dataset.npz"

    # --- Avellaneda-Stoikov policy ---
    as_gamma: float = 5.0
    as_kappa: float = 50.0         #
    as_mix_ratio: float = 1.0      # solo A-S (niente random policy nel dataset)
    as_eta: float = -0.035         # shape parametro per dynamic order size (inventory assoluta)

    # --- Mixed regime episodes ---
    mixed_regime_frac: float = 0.30
    max_switches: int = 2
    switch_warmup_frac: float = 0.15



"""
Default configuration for the Market Making simulator.

Regime calibration rationale (see CALIBRATION_NOTES below).

The key ratio that determines book depth is:
    Q_equilibrium ~ λ_lo / (λ_mo * sweep_fraction + λ_cancel)

Between calm and stressed conditions, empirical LOB studies
(Cont-Stoikov-Talreja 2010, Huang-Rosenbaum 2015) show:
  - Book depth varies by ~3-5x, not 100x
  - MO arrival rate varies by ~1.5-2x
  - Cancellation rate varies by ~1.5-2x
  - LO arrival rate is relatively stable

We calibrate to produce:
  - low_vol:  ~40-60 volume at best levels
  - mid_vol:  ~15-25 volume at best levels
  - high_vol: ~5-12 volume at best levels

This gives a realistic ~5x ratio between extremes.
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

    # --- Episode ---
    T_max: int = 1000

    # --- Reward ---
    alpha_inventory: float = 1e-3   # penalità inventory nel reward del MM

    # --- Dataset generation ---
    N_episodes: int = 300
    dataset_path: str = "dataset.npz"

    # --- Avellaneda-Stoikov policy ---
    as_gamma: float = 5.0
    as_kappa: float = 50.0
    as_mix_ratio: float = 1.0      # solo A-S (niente random policy nel dataset)
    as_eta: float = -0.035         # shape parametro per dynamic order size (inventory assoluta)

    # --- Mixed regime episodes ---
    mixed_regime_frac: float = 0.30
    max_switches: int = 2
    switch_warmup_frac: float = 0.15


# ─────────────────────────────────────────────────────────
# CALIBRATION_NOTES (v2)
# ─────────────────────────────────────────────────────────
#
# Volume at level 0 depends on:
#   + LO arrival: λ_lo * P(level=0) * E[vol]
#     with geometric P(0)≈0.2 and E[vol]=5.5 → +1.1*λ_lo per step
#   - MO consumption: λ_mo * E[mo_size+1], split across levels
#   - Cancellations: λ_cancel * E[cancel_vol]
#
# Key constraint: E[MO_size] / L0_vol should be < ~1.5
# to prevent book collapse and degenerate imbalance.
#
# Regime-defining params (set regime identity):
#   σ_mid:     0.008 / 0.020 / 0.050
#   p_informed: 0.08 / 0.20  / 0.35
#   λ_mo:      0.35 / 0.50  / 0.75
#
# Structural params (tuned for book stability):
#   mo_size:   2.1  / 3.0  / 3.0
#   λ_lo:      0.95 / 1.00 / 0.90
#   λ_cancel:  0.40 / 0.50 / 0.55
#
# Measured steady-state (20 eps × 1000 step):
#   L0 vol:      12 / 6 / 3       (ratio ~4x)
#   L0 vuoto:    3% / 14% / 28%
#   |imb|>0.8:  <0.1% / <0.2% / ~4%
#   vol/side:    110 / 86 / 53    (ratio ~2x)
# ─────────────────────────────────────────────────────────
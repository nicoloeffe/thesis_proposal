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
    alpha_inventory: float = 1e-4

    # --- Dataset generation ---
    N_episodes: int = 300
    dataset_path: str = "dataset.npz"

    # --- Random policy ---
    rand_delta_low: int = 1
    rand_delta_high: int = 5
    rand_qty_low: int = 1
    rand_qty_high: int = 10

    # --- Avellaneda-Stoikov policy ---
    as_gamma: float = 5.0
    as_kappa: float = 50.0
    as_mix_ratio: float = 1.0

    # --- Mixed regime episodes ---
    mixed_regime_frac: float = 0.30
    max_switches: int = 2
    switch_warmup_frac: float = 0.15


# ─────────────────────────────────────────────────────────
# CALIBRATION_NOTES
# ─────────────────────────────────────────────────────────
#
# Volume at level 0 depends on:
#   + LO arrival: λ_lo * P(level=0) * E[vol]
#     with geometric P(0)≈0.2 and E[vol]=5.5 → +1.1*λ_lo per step
#   - MO consumption: λ_mo * E[mo_size+1], split across levels
#   - Cancellations: λ_cancel * E[cancel_vol]
#
# For the book NOT to explode/collapse, we need the
# in-flow ≈ out-flow at steady state. The parameters below
# are chosen so that:
#   low_vol:  in-flow > out-flow → book accumulates (thick)
#   mid_vol:  in-flow ≈ out-flow → moderate depth
#   high_vol: in-flow < out-flow → book is thin
#
# σ_mid: 0.008 / 0.020 / 0.045  (ratio 1:2.5:5.6)
#   Realistic for intraday: calm morning → volatile news event
#
# p_informed: 0.08 / 0.20 / 0.35  (ratio 1:2.5:4.4)
#   Calm: few informed traders. Stressed: toxic flow.
#
# λ_mo: 0.4 / 0.5 / 0.7  (ratio 1:1.25:1.75)
#   MO frequency increases moderately in stress, not dramatically.
#
# mo_size: 2.0 / 3.0 / 4.5  (ratio 1:1.5:2.25)
#   Stressed MOs are more aggressive, sweeping deeper.
#
# λ_lo: 1.0 / 1.0 / 0.8  (ratio 1:1:0.8)
#   LO supply is relatively stable. Slight decrease in stress
#   as market makers widen or withdraw.
#
# λ_cancel: 0.4 / 0.5 / 0.7  (ratio 1:1.25:1.75)
#   Faster cancellation in stress (protective pulling).
# ─────────────────────────────────────────────────────────

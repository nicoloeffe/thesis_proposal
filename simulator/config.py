"""Default configuration for the Market Making simulator."""
from dataclasses import dataclass


@dataclass
class EnvConfig:
    # --- Market structure ---
    tick_size: float = 0.01          # minimum price increment
    L: int = 10                      # number of levels per side in the book
    initial_mid: float = 100.0       # starting mid-price

    # --- Mid-price dynamics ---
    sigma_mid: float = 0.02          # std-dev of mid-price Gaussian random walk

    # --- Market microstructure ---
    p_informed: float = 0.2          # fraction of informed (adverse selection) MOs
    lambda_mo_buy: float = 0.5       # market buy orders
    lambda_mo_sell: float = 0.5      # market sell orders
    mo_size_lambda: float = 3.0      # mean MO size (Poisson) — controls how many levels get swept
    lambda_lo_bid: float = 1.0       # limit bid orders
    lambda_lo_ask: float = 1.0       # limit ask orders
    lambda_cancel_bid: float = 0.5   # bid cancellations
    lambda_cancel_ask: float = 0.5   # ask cancellations

    # --- Episode ---
    T_max: int = 1000                # max steps per episode

    # --- Reward ---
    alpha_inventory: float = 1e-4    # inventory penalty coefficient

    # --- Dataset generation ---
    N_episodes: int = 100            # number of episodes to simulate
    dataset_path: str = "dataset.npz"

    # --- Random policy (used in simulate.py) ---
    rand_delta_low: int = 1          # min offset in ticks
    rand_delta_high: int = 5         # max offset in ticks
    rand_qty_low: int = 1            # min quote quantity
    rand_qty_high: int = 10          # max quote quantity

    # --- Avellaneda-Stoikov policy ---
    as_gamma: float = 5.0            # risk-aversion coefficient (deve essere O(1/sigma^2) per skew efficace)
    as_kappa: float = 50.0           # order arrival rate sensitivity (high → narrow spread floor)
    as_mix_ratio: float = 1.0        # fraction of episodes using A-S policy (1.0 = only A-S)

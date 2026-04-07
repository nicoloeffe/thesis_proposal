"""
MarketMakingEnv — Fase 1: Simulatore LOB.

Observation dict:
    book        : np.ndarray (2, L, 2)  — [side, level, (price, volume)]
    mid         : float
    spread      : float
    imbalance   : float
    inventory   : float

Action tuple: (delta_bid, delta_ask, qty)
    delta_bid/ask : real-valued tick offsets from mid (snapped to grid, min 1)
    qty           : quote size
"""

from __future__ import annotations

import numpy as np
from typing import Any

from config import EnvConfig


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Observation = dict[str, Any]
Action = tuple[float, float, float]


class MarketMakingEnv:
    """Stylised Limit Order Book environment for market-making research."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng()

        # internal state (populated by reset)
        self.mid: float = 0.0
        self.book: np.ndarray = np.zeros((2, self.cfg.L, 2))  # (side, level, [price, vol])
        self.inventory: float = 0.0
        self.t: int = 0

        # last MM quotes (set during step)
        self._mm_bid_price: float = 0.0
        self._mm_ask_price: float = 0.0
        self._mm_qty: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the environment and return the initial observation o_0."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.mid = self._snap(self.cfg.initial_mid)
        self.inventory = 0.0
        self.t = 0
        self._mm_bid_price = 0.0
        self._mm_ask_price = 0.0
        self._mm_qty = 0.0

        self._rebuild_book()

        # Baseline volume per side: used to scale cancellation rate (CST-style).
        # Cancel rate ~ θ * Q: more volume → more cancellations → natural equilibrium.
        self._baseline_vol_per_side = max(
            1.0, 0.5 * (self.book[0, :, 1].sum() + self.book[1, :, 1].sum())
        )

        return self._make_obs()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one environment step.

        Steps (as per spec):
          1. Snap action to tick grid, derive MM quote prices.
          2. Insert MM quotes into the book.
          3. Update mid-price with Gaussian random walk.
          4. Sample MO / LO / cancellation counts (Poisson).
          5. Apply events to the book; compute MM executions.
          6. Update inventory, compute reward.
          7. Build and return o_{t+1}.
        """
        delta_bid, delta_ask, qty = action

        # --- 1. Snap to grid ---
        k_bid = max(1, int(round(delta_bid)))
        k_ask = max(1, int(round(delta_ask)))
        bid_price = self._snap(self.mid - k_bid * self.cfg.tick_size)
        ask_price = self._snap(self.mid + k_ask * self.cfg.tick_size)
        self._mm_bid_price = bid_price
        self._mm_ask_price = ask_price
        self._mm_qty = float(qty)

        # --- 2. Insert MM quotes into book ---
        self._insert_mm_quotes(bid_price, ask_price, qty)

        # --- 3. Mid-price random walk ---
        # Sample shock first — needed for adverse selection below
        shock = self.rng.normal(0.0, self.cfg.sigma_mid)
        old_mid = self.mid
        self.mid = self._snap(self.mid + shock)
        self._shift_book_prices(old_mid, self.mid)

        # --- 4. Sample Poisson event counts ---
        # Adverse selection: p_informed fraction of MOs are "informed" —
        # they arrive in the direction of the realized shock, hitting the MM
        # just before the price moves against them.
        n_mo_buy  = self.rng.poisson(self.cfg.lambda_mo_buy)
        n_mo_sell = self.rng.poisson(self.cfg.lambda_mo_sell)

        n_informed = self.rng.poisson(self.cfg.lambda_mo_buy * self.cfg.p_informed)
        if shock > 0:
            # price went up → informed sellers hit bid (MM buys at a soon-to-be-bad price)
            n_mo_sell += n_informed
        else:
            # price went down → informed buyers hit ask (MM sells at a soon-to-be-bad price)
            n_mo_buy += n_informed

        n_lo_bid   = self.rng.poisson(self.cfg.lambda_lo_bid)
        n_lo_ask   = self.rng.poisson(self.cfg.lambda_lo_ask)

        # Cancellation rate scales with book depth (CST: θ * Q).
        # When the book is thicker than baseline, more cancellations occur;
        # when thinner, fewer. This creates a natural depth equilibrium.
        bid_vol = self.book[0, :, 1].sum()
        ask_vol = self.book[1, :, 1].sum()
        base = self._baseline_vol_per_side
        n_can_bid  = self.rng.poisson(self.cfg.lambda_cancel_bid * max(0.1, bid_vol / base))
        n_can_ask  = self.rng.poisson(self.cfg.lambda_cancel_ask * max(0.1, ask_vol / base))

        # --- 5. Apply events; track MM executions ---
        q_exec_ask = self._apply_mo_buy(n_mo_buy)    # MO buys hit the ask side
        q_exec_bid = self._apply_mo_sell(n_mo_sell)  # MO sells hit the bid side
        self._apply_lo(n_lo_bid, side=0)
        self._apply_lo(n_lo_ask, side=1)
        self._apply_cancellations(n_can_bid, side=0)
        self._apply_cancellations(n_can_ask, side=1)

        # --- 6. Inventory and reward ---
        # Capture inventory before this step's executions for MTM calculation
        inventory_prev = self.inventory
        self.inventory += q_exec_bid - q_exec_ask

        # PnL has two components:
        # 1. Spread PnL: half-spread earned on each execution
        # 2. Mark-to-market: change in value of inventory due to mid-price move
        #    inventory_prev * shock captures the adverse selection cost —
        #    if MM bought and mid fell, or sold and mid rose, this is negative
        realized_mid_move = self.mid - old_mid
        spread_pnl = (ask_price - old_mid) * q_exec_ask + (old_mid - bid_price) * q_exec_bid
        mtm_pnl = inventory_prev * realized_mid_move
        pnl = spread_pnl + mtm_pnl
        reward = pnl - self.cfg.alpha_inventory * self.inventory ** 2

        # --- 7. Build next observation ---
        self.t += 1
        done = self.t >= self.cfg.T_max
        obs_next = self._make_obs()

        info = {
            "q_exec_bid": q_exec_bid,
            "q_exec_ask": q_exec_ask,
            "pnl": pnl,
            "bid_price": bid_price,
            "ask_price": ask_price,
        }
        return obs_next, float(reward), done, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _snap(self, price: float) -> float:
        """Round price to the nearest tick."""
        ts = self.cfg.tick_size
        return round(round(price / ts) * ts, 10)

    def _rebuild_book(self) -> None:
        """Initialise book levels around the current mid-price."""
        L = self.cfg.L
        ts = self.cfg.tick_size
        self.book = np.zeros((2, L, 2), dtype=np.float64)

        for lvl in range(L):
            # Bid side: levels below mid
            bid_p = self._snap(self.mid - (lvl + 1) * ts)
            bid_v = float(self.rng.integers(1, 20))
            self.book[0, lvl] = [bid_p, bid_v]

            # Ask side: levels above mid
            ask_p = self._snap(self.mid + (lvl + 1) * ts)
            ask_v = float(self.rng.integers(1, 20))
            self.book[1, lvl] = [ask_p, ask_v]

    def _shift_book_prices(self, old_mid: float, new_mid: float) -> None:
        """Shift all book price levels by the same delta as the mid-price move.
        Volumes are preserved; prices are snapped to the tick grid after the shift.
        MM quote references are updated accordingly so execution detection stays correct.
        """
        delta = new_mid - old_mid
        if abs(delta) < 1e-12:
            return
        for side in range(2):
            for lvl in range(self.cfg.L):
                self.book[side, lvl, 0] = self._snap(self.book[side, lvl, 0] + delta)
        # keep MM quote prices in sync with the shifted book
        self._mm_bid_price = self._snap(self._mm_bid_price + delta)
        self._mm_ask_price = self._snap(self._mm_ask_price + delta)

    def _insert_mm_quotes(self, bid_price: float, ask_price: float, qty: float) -> None:
        """Add MM's quote volume to the corresponding book levels."""
        self._add_volume_at_price(side=0, price=bid_price, volume=qty)
        self._add_volume_at_price(side=1, price=ask_price, volume=qty)

    def _add_volume_at_price(self, side: int, price: float, volume: float) -> None:
        """
        Find the level whose price matches `price` and add `volume`.
        If the price is outside the tracked L levels, the quote is ignored
        (it would be too deep to affect execution in this simplified model).
        """
        for lvl in range(self.cfg.L):
            if abs(self.book[side, lvl, 0] - price) < 1e-9:
                self.book[side, lvl, 1] += volume
                return
        # Price not in the current grid — insert at the closest boundary level
        # (simplified: we skip since MO sweep handles best-level only)

    def _apply_mo_buy(self, n: int) -> float:
        """
        Apply `n` market buy orders sweeping the ask side.
        Each MO has a random size sampled from Poisson(mo_size_lambda) + 1,
        and sweeps levels from best ask inward until its size is exhausted.
        Returns the volume executed against the MM's ask quote.
        """
        q_exec_mm = 0.0
        for _ in range(n):
            mo_size = int(self.rng.poisson(self.cfg.mo_size_lambda)) + 1
            remaining = float(mo_size)
            for lvl in range(self.cfg.L):
                if remaining <= 0:
                    break
                avail = self.book[1, lvl, 1]
                if avail <= 0:
                    continue
                consumed = min(remaining, avail)
                self.book[1, lvl, 1] -= consumed
                if abs(self.book[1, lvl, 0] - self._mm_ask_price) < 1e-9:
                    q_exec_mm += consumed
                remaining -= consumed
        return q_exec_mm

    def _apply_mo_sell(self, n: int) -> float:
        """
        Apply `n` market sell orders sweeping the bid side.
        Each MO has a random size sampled from Poisson(mo_size_lambda) + 1,
        and sweeps levels from best bid inward until its size is exhausted.
        Returns the volume executed against the MM's bid quote.
        """
        q_exec_mm = 0.0
        for _ in range(n):
            mo_size = int(self.rng.poisson(self.cfg.mo_size_lambda)) + 1
            remaining = float(mo_size)
            for lvl in range(self.cfg.L):
                if remaining <= 0:
                    break
                avail = self.book[0, lvl, 1]
                if avail <= 0:
                    continue
                consumed = min(remaining, avail)
                self.book[0, lvl, 1] -= consumed
                if abs(self.book[0, lvl, 0] - self._mm_bid_price) < 1e-9:
                    q_exec_mm += consumed
                remaining -= consumed
        return q_exec_mm

    def _apply_lo(self, n: int, side: int) -> None:
        """
        Add `n` limit orders on `side` (0=bid, 1=ask).
        Each LO picks a level with geometric probability (best = most likely)
        and adds a random volume sampled from Uniform[1, 10].
        """
        # Geometric distribution with p=0.2 — flatter decay than p=0.5,
        # ensures deeper levels receive meaningful volume (ratio level0/level9 ≈ 8x
        # vs 512x with p=0.5). More realistic LOB depth structure.
        probs = np.array([(0.8 ** lvl) * 0.2 for lvl in range(self.cfg.L)], dtype=np.float64)
        probs /= probs.sum()
        for _ in range(n):
            lvl = self.rng.choice(self.cfg.L, p=probs)
            vol = float(self.rng.integers(1, 11))
            self.book[side, lvl, 1] += vol

    def _apply_cancellations(self, n: int, side: int) -> None:
        """
        Cancel volume from `n` randomly selected levels on `side`.
        Only levels with volume > 0 are eligible. Volume is clamped to ≥ 0.
        """
        eligible = [lvl for lvl in range(self.cfg.L) if self.book[side, lvl, 1] > 0]
        for _ in range(n):
            if not eligible:
                break
            lvl = self.rng.choice(eligible)
            cancel_vol = float(self.rng.integers(1, max(2, int(self.book[side, lvl, 1]) + 1)))
            self.book[side, lvl, 1] = max(0.0, self.book[side, lvl, 1] - cancel_vol)
            if self.book[side, lvl, 1] == 0:
                eligible.remove(lvl)

    def _make_obs(self) -> Observation:
        """Build the observation dictionary from current state."""
        total_bid_vol = self.book[0, :, 1].sum()
        total_ask_vol = self.book[1, :, 1].sum()
        denom = total_bid_vol + total_ask_vol
        imbalance = (total_bid_vol - total_ask_vol) / denom if denom > 0 else 0.0

        # True best bid/ask: first level with volume > 0.
        # If a side is empty, fall back to the grid boundary.
        best_bid = self.book[0, 0, 0]
        for lvl in range(self.cfg.L):
            if self.book[0, lvl, 1] > 0:
                best_bid = self.book[0, lvl, 0]
                break

        best_ask = self.book[1, 0, 0]
        for lvl in range(self.cfg.L):
            if self.book[1, lvl, 1] > 0:
                best_ask = self.book[1, lvl, 0]
                break

        spread = best_ask - best_bid

        return {
            "book": self.book.copy(),         # (2, L, 2)
            "mid": self.mid,
            "spread": spread,
            "imbalance": imbalance,
            "inventory": self.inventory,
        }
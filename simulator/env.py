"""
MarketMakingEnv — Fase 1: Simulatore LOB (v2).

Modifiche rispetto a v1:
  - Cancel-and-replace: le quote MM vengono rimosse a inizio step e reinserite.
  - Pro-rata fills: il volume MM è tracciato separatamente; l'attribuzione
    delle esecuzioni è proporzionale alla frazione MM sul livello.
  - Protezione da cancellazioni: solo il volume background viene cancellato.
  - Clamp k ∈ [1, L]: le quote MM non possono finire fuori dal book.
  - Nessun fallback silenzioso in _place_mm_volume.

Observation dict:
  book      : np.ndarray (2, L, 2) — [side, level, (price, volume)]
  mid       : float
  spread    : float
  imbalance : float
  inventory : float

Action tuple: (delta_bid, delta_ask, q_bid, q_ask)
  delta_bid/ask : tick offsets from mid (snapped to grid, clamped to [1, L])
  q_bid/q_ask   : quote size per lato
"""

from __future__ import annotations

import numpy as np
from typing import Any

from config import EnvConfig

Observation = dict[str, Any]
Action = tuple[float, float, float, float]


class MarketMakingEnv:
    """Stylised Limit Order Book environment for market-making research."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng()

        self.mid: float = 0.0
        self.book: np.ndarray = np.zeros((2, self.cfg.L, 2))
        self.inventory: float = 0.0
        self.t: int = 0

        # MM quote state
        self._mm_bid_price: float = 0.0
        self._mm_ask_price: float = 0.0
        self._mm_bid_qty: float = 0.0
        self._mm_ask_qty: float = 0.0

        # Track MM volume per level for pro-rata fills
        # Shape: (2, L) — [side, level]
        self._mm_volume: np.ndarray = np.zeros((2, self.cfg.L))

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
        self._mm_bid_qty = 0.0
        self._mm_ask_qty = 0.0
        self._mm_volume = np.zeros((2, self.cfg.L))

        self._rebuild_book()

        self._baseline_vol_per_side = max(
            1.0, 0.5 * (self.book[0, :, 1].sum() + self.book[1, :, 1].sum())
        )

        return self._make_obs()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one environment step.

        Timing convention (v3):
          1. Remove old MM quotes, insert new ones
          2. Sample shock and event counts (shock determines informed flow)
          3. Execute all MOs, LOs, cancellations on current book
          4. Compute PnL at execution prices (pre-shock)
          5. Apply mid-price shock and shift book for next step
        """
        delta_bid, delta_ask, q_bid, q_ask = action

        # --- 0. Remove previous MM quotes (cancel-and-replace) ---
        self._remove_mm_quotes()

        # --- 1. Snap to grid, clamp to [1, L] ---
        k_bid = max(1, min(self.cfg.L, int(round(delta_bid))))
        k_ask = max(1, min(self.cfg.L, int(round(delta_ask))))
        bid_price = self._snap(self.mid - k_bid * self.cfg.tick_size)
        ask_price = self._snap(self.mid + k_ask * self.cfg.tick_size)

        self._mm_bid_price = bid_price
        self._mm_ask_price = ask_price
        self._mm_bid_qty = float(q_bid)
        self._mm_ask_qty = float(q_ask)

        # --- 2. Insert MM quotes into book ---
        self._insert_mm_quotes(bid_price, ask_price, q_bid, q_ask)

        # --- 3. Sample shock and event counts ---
        # The shock is sampled now but APPLIED AFTER executions.
        # Informed traders observe the shock direction and trade accordingly.
        shock = self.rng.normal(0.0, self.cfg.sigma_mid)

        n_mo_buy = self.rng.poisson(self.cfg.lambda_mo_buy)
        n_mo_sell = self.rng.poisson(self.cfg.lambda_mo_sell)

        n_informed = self.rng.poisson(
            0.5 * (self.cfg.lambda_mo_buy + self.cfg.lambda_mo_sell) * self.cfg.p_informed
        )
        # Adverse selection: informed traders trade WITH the shock direction.
        # shock > 0 → price will rise → informed BUY (sweep ask, hurt MM ask)
        # shock < 0 → price will fall → informed SELL (sweep bid, hurt MM bid)
        if shock > 0:
            n_mo_buy += n_informed
        else:
            n_mo_sell += n_informed

        n_lo_bid = self.rng.poisson(self.cfg.lambda_lo_bid)
        n_lo_ask = self.rng.poisson(self.cfg.lambda_lo_ask)

        bid_vol = self.book[0, :, 1].sum()
        ask_vol = self.book[1, :, 1].sum()
        base = self._baseline_vol_per_side
        n_can_bid = self.rng.poisson(self.cfg.lambda_cancel_bid * max(0.1, bid_vol / base))
        n_can_ask = self.rng.poisson(self.cfg.lambda_cancel_ask * max(0.1, ask_vol / base))

        # --- 4. Apply events on CURRENT book (pre-shock prices) ---
        q_exec_ask = self._apply_mo_buy(n_mo_buy)
        q_exec_bid = self._apply_mo_sell(n_mo_sell)
        self._apply_lo(n_lo_bid, side=0)
        self._apply_lo(n_lo_ask, side=1)
        self._apply_cancellations(n_can_bid, side=0)
        self._apply_cancellations(n_can_ask, side=1)

        # --- 5. PnL at execution prices (pre-shock, consistent) ---
        self.inventory += q_exec_bid - q_exec_ask

        spread_pnl = (ask_price - self.mid) * q_exec_ask + (self.mid - bid_price) * q_exec_bid

        # --- 6. Apply mid-price shock AFTER executions ---
        old_mid = self.mid
        self.mid = self._snap(self.mid + shock)
        self._shift_book_prices(old_mid, self.mid)

        # MTM on full post-execution inventory: contracts just acquired
        # also experience the shock within this timestep.
        mtm_pnl = self.inventory * (self.mid - old_mid)
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
        ts = self.cfg.tick_size
        return round(round(price / ts) * ts, 10)

    def _rebuild_book(self) -> None:
        L = self.cfg.L
        ts = self.cfg.tick_size
        self.book = np.zeros((2, L, 2), dtype=np.float64)
        self._mm_volume = np.zeros((2, L))
        for lvl in range(L):
            bid_p = self._snap(self.mid - (lvl + 1) * ts)
            bid_v = float(self.rng.integers(1, 20))
            self.book[0, lvl] = [bid_p, bid_v]

            ask_p = self._snap(self.mid + (lvl + 1) * ts)
            ask_v = float(self.rng.integers(1, 20))
            self.book[1, lvl] = [ask_p, ask_v]

    def _shift_book_prices(self, old_mid: float, new_mid: float) -> None:
        delta = new_mid - old_mid
        if abs(delta) < 1e-12:
            return
        for side in range(2):
            for lvl in range(self.cfg.L):
                self.book[side, lvl, 0] = self._snap(self.book[side, lvl, 0] + delta)
        self._mm_bid_price = self._snap(self._mm_bid_price + delta)
        self._mm_ask_price = self._snap(self._mm_ask_price + delta)

    # --- MM quote management ---

    def _remove_mm_quotes(self) -> None:
        """Remove previous MM volume from the book (cancel-and-replace)."""
        for side in range(2):
            for lvl in range(self.cfg.L):
                mm_vol = self._mm_volume[side, lvl]
                if mm_vol > 0:
                    self.book[side, lvl, 1] = max(0.0, self.book[side, lvl, 1] - mm_vol)
        self._mm_volume[:] = 0.0

    def _insert_mm_quotes(
        self,
        bid_price: float,
        ask_price: float,
        q_bid: float,
        q_ask: float,
    ) -> None:
        self._place_mm_volume(side=0, price=bid_price, volume=q_bid)
        self._place_mm_volume(side=1, price=ask_price, volume=q_ask)

    def _place_mm_volume(self, side: int, price: float, volume: float) -> None:
        """
        Place MM volume at the exact price level.
        No silent fallback — if the price doesn't match any level
        (shouldn't happen with clamped k), the quote is dropped.
        """
        for lvl in range(self.cfg.L):
            if abs(self.book[side, lvl, 0] - price) < 1e-9:
                self.book[side, lvl, 1] += volume
                self._mm_volume[side, lvl] += volume
                return
        # Safety net: price doesn't match any level — skip silently.
        # With k clamped to [1, L] this should not happen.

    # --- Market order processing (pro-rata MM attribution) ---

    def _apply_mo_buy(self, n: int) -> float:
        """Buy MOs sweep the ask side. MM attribution is pro-rata."""
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

                # Pro-rata: MM gets filled proportionally to its share
                mm_at_lvl = self._mm_volume[1, lvl]
                if mm_at_lvl > 0:
                    mm_frac = min(1.0, mm_at_lvl / avail)
                    mm_filled = consumed * mm_frac
                    q_exec_mm += mm_filled
                    self._mm_volume[1, lvl] = max(0.0, mm_at_lvl - mm_filled)

                self.book[1, lvl, 1] -= consumed
                remaining -= consumed
        return q_exec_mm

    def _apply_mo_sell(self, n: int) -> float:
        """Sell MOs sweep the bid side. MM attribution is pro-rata."""
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

                # Pro-rata: MM gets filled proportionally to its share
                mm_at_lvl = self._mm_volume[0, lvl]
                if mm_at_lvl > 0:
                    mm_frac = min(1.0, mm_at_lvl / avail)
                    mm_filled = consumed * mm_frac
                    q_exec_mm += mm_filled
                    self._mm_volume[0, lvl] = max(0.0, mm_at_lvl - mm_filled)

                self.book[0, lvl, 1] -= consumed
                remaining -= consumed
        return q_exec_mm

    # --- Limit orders and cancellations ---

    def _apply_lo(self, n: int, side: int) -> None:
        probs = np.array([(0.8 ** lvl) * 0.2 for lvl in range(self.cfg.L)], dtype=np.float64)
        probs /= probs.sum()
        for _ in range(n):
            lvl = self.rng.choice(self.cfg.L, p=probs)
            vol = float(self.rng.integers(1, 11))
            self.book[side, lvl, 1] += vol

    def _apply_cancellations(self, n: int, side: int) -> None:
        """
        Cancel only background (non-MM) volume.
        MM quotes persist until filled by MOs or explicitly removed
        at the start of the next step via cancel-and-replace.
        """
        for _ in range(n):
            # Build eligible list: levels with background volume > 0
            eligible = []
            for lvl in range(self.cfg.L):
                bg_vol = self.book[side, lvl, 1] - self._mm_volume[side, lvl]
                if bg_vol > 0.5:
                    eligible.append(lvl)
            if not eligible:
                break
            lvl = self.rng.choice(eligible)
            bg_vol = self.book[side, lvl, 1] - self._mm_volume[side, lvl]
            cancel_vol = float(self.rng.integers(1, max(2, int(bg_vol) + 1)))
            cancel_vol = min(cancel_vol, bg_vol)
            # Floor at MM volume: never eat into MM's share
            self.book[side, lvl, 1] = max(
                self._mm_volume[side, lvl],
                self.book[side, lvl, 1] - cancel_vol,
            )

    # --- Observation ---

    def _make_obs(self) -> Observation:
        total_bid_vol = self.book[0, :, 1].sum()
        total_ask_vol = self.book[1, :, 1].sum()
        denom = total_bid_vol + total_ask_vol
        imbalance = (total_bid_vol - total_ask_vol) / denom if denom > 0 else 0.0

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
            "book": self.book.copy(),
            "mid": self.mid,
            "spread": spread,
            "imbalance": imbalance,
            "inventory": self.inventory,
        }
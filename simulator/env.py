"""
MarketMakingEnv — Stylised LOB simulator for market-making research.

Modelling choices (refactored):
  - Queue-reactive cancellations: cancellation rate at each level is
    proportional to the background volume present at that level.
    This yields an endogenous steady-state depth; no external baseline target
    is imposed on the book.
  - LO arrival profile is BMP02-compliant with cleanly separated parameters:
        λ_lo(0) = lo_best_supply        (L0 suppression: adverse-selection premium)
        λ_lo(ℓ) = exp(-lo_alpha · ℓ)    (dispersion in deep levels, ℓ ≥ 1)
    then normalised to preserve Σ λ_lo = Σ exp(-α·ℓ). When
    lo_best_supply < exp(-lo_alpha) the profile peaks at L1 (BMP02 hump);
    otherwise it is monotone decreasing (classical CST).
    References: Bouchaud-Mézard-Potters (2002); Gould-Porter-Williams (2013);
    Glosten-Milgrom (1985); O'Hara (1995).
  - Initial book is regime-aware: volumes are initialised close to the
    analytical steady state V*_ℓ ≈ λ_lo · E[LO size] · profile(ℓ) / λ_cancel,
    and a silent warmup (no MM) is run inside reset() to let the book converge
    to its true equilibrium (which also accounts for MO sweep at L0).
  - Regime switches are declared as STRUCTURAL SHOCKS: parameters change
    instantaneously while the book state is preserved. The microstructure
    adapts gradually through the endogenous LO/cancellation/MO processes,
    with characteristic time ~ 1 / λ_cancel_per_share (10–15 steps typically).
  - MM fills are pro-rata at the level where the MM quotes. This is a
    deliberate simplification: no queue position modelling. Documented as a
    limitation in the thesis perimeter.
  - MM quotes are cancel-and-replace every step (no persistence).

Observation dict:
  book      : np.ndarray (2, L, 2) — [side, level, (price, volume)]
  mid       : float
  spread    : float
  imbalance : float
  inventory : float

Action tuple: (delta_bid, delta_ask, q_bid, q_ask)
  delta_bid/ask : tick offsets from mid (snapped to grid, clamped to [1, L])
  q_bid/q_ask   : quote size per side
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
        """Reset the environment and return the initial observation o_0.

        Steps:
          1. Zero the agent state (inventory, MM quotes, counters).
          2. Rebuild the book with regime-aware initial volumes (analytical V*).
          3. Run a silent warmup (no MM) so that the book converges to its
             true endogenous equilibrium, including MO-sweep effects at L0.
        """
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

        # Silent warmup: run the market dynamics (without MM quoting) for
        # warmup_steps so the book reaches its endogenous equilibrium under
        # the current regime parameters.
        for _ in range(int(self.cfg.warmup_steps)):
            self._market_step_no_mm()

        return self._make_obs()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one environment step.

        Timing convention (v3):
          1. Remove old MM quotes, insert new ones.
          2. Sample shock and event counts (shock determines informed flow).
          3. Execute all MOs, LOs, cancellations on current book.
          4. Compute PnL at execution prices (pre-shock).
          5. Apply mid-price shock and shift book for next step.
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
        if shock > 0:
            n_mo_buy += n_informed
        else:
            n_mo_sell += n_informed

        # --- 4. Apply events on CURRENT book ---
        q_exec_ask = self._apply_mo_buy(n_mo_buy)
        q_exec_bid = self._apply_mo_sell(n_mo_sell)

        self._apply_lo(self.cfg.lambda_lo_bid, side=0)
        self._apply_lo(self.cfg.lambda_lo_ask, side=1)

        # Queue-reactive cancellations: rate per share on background volume.
        # No external baseline target; equilibrium emerges from LO vs cancel balance.
        self._apply_cancellations(self.cfg.lambda_cancel_per_share, side=0)
        self._apply_cancellations(self.cfg.lambda_cancel_per_share, side=1)

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
        """Initialise the book with regime-aware volumes close to the
        analytical LO/cancel equilibrium for the hump-shaped profile:

            V*_ℓ ≈ λ_lo · E[LO size] · profile(ℓ) / λ_cancel_per_share

        where profile(ℓ) is the BMP02-compliant hump intensity (normalised
        so its sum equals the sum of the pure exponential profile).

        This is only a good approximation for ℓ > 0 (at ℓ = 0 MO sweep pulls
        the equilibrium below this value). The subsequent warmup in reset()
        corrects for that, converging to the true equilibrium.
        """
        L = self.cfg.L
        ts = self.cfg.tick_size
        self.book = np.zeros((2, L, 2), dtype=np.float64)
        self._mm_volume = np.zeros((2, L))

        # E[LO size] per arrival: rng.integers(1, 11) → mean ≈ 5.5; use 5.0 as a
        # slightly conservative estimate.
        mean_lo_size = 5.0
        lambda_lo_avg = 0.5 * (self.cfg.lambda_lo_bid + self.cfg.lambda_lo_ask)
        lambda_cancel = max(1e-6, self.cfg.lambda_cancel_per_share)
        profile = self._lo_intensity_profile()  # hump-shaped, normalised

        for lvl in range(L):
            v_target = lambda_lo_avg * mean_lo_size * profile[lvl] / lambda_cancel
            # Poisson noise around the target for initial variety; floor at 1
            # to ensure non-empty book.
            bid_v = float(max(1, self.rng.poisson(max(0.5, v_target))))
            ask_v = float(max(1, self.rng.poisson(max(0.5, v_target))))

            bid_p = self._snap(self.mid - (lvl + 1) * ts)
            ask_p = self._snap(self.mid + (lvl + 1) * ts)

            self.book[0, lvl] = [bid_p, bid_v]
            self.book[1, lvl] = [ask_p, ask_v]

    def _market_step_no_mm(self) -> None:
        """Simulate one step of market dynamics WITHOUT the MM quoting.

        Used for the silent warmup inside reset(). Everything is identical to
        step() except the MM does not place any volume; MO attribution still
        runs (returns 0 since _mm_volume is all zero), so the bookkeeping is
        consistent.
        """
        shock = self.rng.normal(0.0, self.cfg.sigma_mid)

        n_mo_buy = self.rng.poisson(self.cfg.lambda_mo_buy)
        n_mo_sell = self.rng.poisson(self.cfg.lambda_mo_sell)
        n_informed = self.rng.poisson(
            0.5 * (self.cfg.lambda_mo_buy + self.cfg.lambda_mo_sell) * self.cfg.p_informed
        )
        if shock > 0:
            n_mo_buy += n_informed
        else:
            n_mo_sell += n_informed

        self._apply_mo_buy(n_mo_buy)
        self._apply_mo_sell(n_mo_sell)
        self._apply_lo(self.cfg.lambda_lo_bid, side=0)
        self._apply_lo(self.cfg.lambda_lo_ask, side=1)
        self._apply_cancellations(self.cfg.lambda_cancel_per_share, side=0)
        self._apply_cancellations(self.cfg.lambda_cancel_per_share, side=1)

        old_mid = self.mid
        self.mid = self._snap(self.mid + shock)
        self._shift_book_prices(old_mid, self.mid)

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

    def _lo_intensity_profile(self) -> np.ndarray:
        """BMP02-compliant LO arrival profile with cleanly separated parameters.

        Shape:
            λ_lo(0) = f_0              ← suppression at the best (adverse-
                                         selection aversion, Glosten-Milgrom
                                         1985, O'Hara 1995)
            λ_lo(ℓ) = exp(-α · ℓ)      for ℓ ≥ 1  ← exponential decay in
                                                    deep levels (CST 2010)

        Normalised so that Σ shape = Σ exp(-α · ℓ), i.e. the aggregate arrival
        rate matches that of the pure exponential profile. This preserves the
        equilibrium depth governed by λ_lo / λ_cancel and does not require
        re-tuning of λ_cancel_per_share.

        When f_0 < exp(-α) the profile has an internal peak at ℓ = 1 (BMP02
        hump). When f_0 ≥ exp(-α) the profile is monotone decreasing (pure
        CST-style; applicable to calm regimes where adverse selection is low).

        Parameter roles are CLEANLY SEPARATED:
          lo_alpha       → dispersion of liquidity in deep levels (ℓ ≥ 1)
          lo_best_supply → L0 suppression strength (peak emergence)

        References: Bouchaud-Mézard-Potters (2002); Gould-Porter-Williams (2013);
        Glosten-Milgrom (1985); O'Hara (1995).
        """
        L = self.cfg.L
        levels = np.arange(L)
        alpha = self.cfg.lo_alpha
        f0 = self.cfg.lo_best_supply

        # Raw shape: exponential everywhere, with L0 overridden by f_0
        exp_profile = np.exp(-alpha * levels)
        shape = exp_profile.copy()
        shape[0] = f0

        # Normalise to preserve aggregate intensity = Σ exp(-α·ℓ)
        target_total = float(exp_profile.sum())
        shape_sum = float(shape.sum())
        if shape_sum <= 0:
            return exp_profile
        return shape * (target_total / shape_sum)

    def _apply_lo(self, n_base: float, side: int) -> None:
        """Apply limit order arrivals with BMP02-compliant profile.

        The per-level intensity is λ(ℓ) = n_base · profile(ℓ), where profile is
        the normalised shape with L0 suppression (see _lo_intensity_profile).
        When lo_best_supply < exp(-lo_alpha) the profile peaks internally at
        L1 (BMP02 hump); otherwise it is monotone decreasing (classical CST
        style — appropriate for calm regimes with low adverse selection).

        Aggregate arrival intensity per step is unchanged relative to a pure
        exponential profile, so this modification does NOT alter the
        equilibrium depth (still governed by λ_lo / λ_cancel_per_share).
        """
        lambda_vec = n_base * self._lo_intensity_profile()
        new_orders = self.rng.poisson(lambda_vec)
        for lvl, n in enumerate(new_orders):
            for _ in range(int(n)):
                self.book[side, lvl, 1] += float(self.rng.integers(1, 11))

    def _apply_cancellations(self, lambda_per_share: float, side: int) -> None:
        """Queue-reactive cancellations.

        The cancellation intensity at each level is proportional to the
        background (non-MM) volume at that level:
            N_cancel_ℓ ~ Poisson(λ_per_share · V_ℓ_bg)

        This produces a depth equilibrium V*_ℓ that emerges endogenously from
        the LO vs cancellation balance, with no external baseline target. The
        characteristic relaxation time after a regime switch is ≈ 1 / λ_per_share.

        Cancellations only act on background volume (MM volume is protected;
        MM handles its own cancel-and-replace separately).
        """
        if lambda_per_share <= 0.0:
            return
        for lvl in range(self.cfg.L):
            bg_vol = self.book[side, lvl, 1] - self._mm_volume[side, lvl]
            if bg_vol <= 0.5:
                continue
            n_cancel = int(self.rng.poisson(lambda_per_share * bg_vol))
            if n_cancel <= 0:
                continue
            # Cap at available background volume
            n_cancel = min(n_cancel, int(bg_vol))
            self.book[side, lvl, 1] = max(
                self._mm_volume[side, lvl],
                self.book[side, lvl, 1] - float(n_cancel),
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
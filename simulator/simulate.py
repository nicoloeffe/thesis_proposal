"""
simulate.py — Offline multi-regime LOB dataset generation with A-S + deterministic size.

Regime design (post-refactor):
  Each regime tunes four economically distinct channels:
    - σ, p_informed, λ_mo, mo_size_λ      (price dynamics & toxicity)
    - λ_lo, λ_cancel_per_share           (endogenous book depth)
    - lo_alpha                            (volume profile shape)
    - κ                                   (microstructure half-spread)
  γ (MM risk aversion) is constant across regimes. α_inventory is constant across
  regimes. baseline_vol has been removed — depth emerges from LO/cancel balance.

Saved dataset fields:
  observations      : (N_total, obs_dim)
  actions           : (N_total, 4)  — [k_bid, k_ask, q_bid, q_ask]
  rewards           : (N_total,)
  next_observations : (N_total, obs_dim)
  regimes           : (N_total,)
  episode_ids       : (N_total,)
  switch_mask       : (N_total,)
  timesteps         : (N_total,)
  inventories       : (N_total,)
  time_left         : (N_total,)
"""

from __future__ import annotations

import numpy as np
import argparse
from dataclasses import replace

from config import EnvConfig
from env import MarketMakingEnv, Observation

# ---------------------------------------------------------------------------
# Regime definitions
#
# Calibrated to produce (at steady state):
#   - depth ratio low:mid:high ≈ 3:1.5:1 (driven by λ_cancel_per_share)
#   - spread ratio low:mid:high ≈ 1:1.7:2.6  (driven by σ and κ)
#   - k_bid/k_ask ≈ 1.2 / 2.0 / 3.3 ticks (emerges from γ=5 + regime σ, κ)
# γ is constant (set in EnvConfig); not regime-specific.
# ---------------------------------------------------------------------------

REGIMES = [
    {
        "name": "low_vol",
        "sigma_mid": 0.008,
        "p_informed": 0.02,
        "lambda_mo_buy": 0.35,
        "lambda_mo_sell": 0.35,
        "mo_size_lambda": 3.0,
        "lambda_lo_bid": 0.95,
        "lambda_lo_ask": 0.95,
        "lambda_cancel_per_share": 0.08,  # slow cancellations → deep book
        "lo_alpha": 0.40,                 # volumes concentrated near the best
        "lo_best_supply": 0.85,           # mild L0 suppression (low adverse selection)
        "as_kappa": 80.0,                 # tighter microstructure spread
    },
    {
        "name": "mid_vol",
        "sigma_mid": 0.015,
        "p_informed": 0.10,
        "lambda_mo_buy": 0.50,
        "lambda_mo_sell": 0.50,
        "mo_size_lambda": 2.5,
        "lambda_lo_bid": 0.70,
        "lambda_lo_ask": 0.70,
        "lambda_cancel_per_share": 0.16,
        "lo_alpha": 0.25,
        "lo_best_supply": 0.70,           # moderate L0 suppression (hump at L1)
        "as_kappa": 50.0,
    },
    {
        "name": "high_vol",
        "sigma_mid": 0.028,
        "p_informed": 0.18,
        "lambda_mo_buy": 0.75,
        "lambda_mo_sell": 0.75,
        "mo_size_lambda": 2.0,
        "lambda_lo_bid": 0.65,
        "lambda_lo_ask": 0.65,
        "lambda_cancel_per_share": 0.28,  # aggressive cancellations → thin book
        "lo_alpha": 0.15,                 # more disperse volumes across levels
        "lo_best_supply": 0.65,           # strong-ish L0 suppression (high adverse selection, balanced against MO sweep)
        "as_kappa": 30.0,                 # wider microstructure spread
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def obs_to_vector(obs: Observation, L: int) -> np.ndarray:
    book_flat = obs["book"].flatten()
    scalars   = np.array([obs["mid"], obs["spread"], obs["imbalance"], obs["inventory"]])
    return np.concatenate([book_flat, scalars])


def apply_regime(cfg: EnvConfig, regime_idx: int) -> EnvConfig:
    """Return a copy of cfg with the regime-specific fields overwritten.

    γ (as_gamma) and α_inventory are NOT regime-specific: they remain at
    the values set in cfg itself.
    """
    regime = REGIMES[regime_idx]
    return replace(
        cfg,
        sigma_mid=regime["sigma_mid"],
        p_informed=regime["p_informed"],
        lambda_mo_buy=regime["lambda_mo_buy"],
        lambda_mo_sell=regime["lambda_mo_sell"],
        mo_size_lambda=regime["mo_size_lambda"],
        lambda_lo_bid=regime["lambda_lo_bid"],
        lambda_lo_ask=regime["lambda_lo_ask"],
        lambda_cancel_per_share=regime["lambda_cancel_per_share"],
        lo_alpha=regime["lo_alpha"],
        lo_best_supply=regime["lo_best_supply"],
        as_kappa=regime["as_kappa"],
    )


def apply_regime_to_env(env: MarketMakingEnv, regime_idx: int) -> None:
    """In-place regime switch on a live environment.

    The book state is DELIBERATELY NOT reset: a regime switch is declared as
    a structural shock on the parameter landscape, while the microstructure
    state persists. The book then adapts gradually via the endogenous
    LO/cancellation/MO processes (characteristic relaxation time
    ≈ 1 / λ_cancel_per_share).
    """
    regime = REGIMES[regime_idx]
    env.cfg.sigma_mid               = regime["sigma_mid"]
    env.cfg.p_informed              = regime["p_informed"]
    env.cfg.lambda_mo_buy           = regime["lambda_mo_buy"]
    env.cfg.lambda_mo_sell          = regime["lambda_mo_sell"]
    env.cfg.mo_size_lambda          = regime["mo_size_lambda"]
    env.cfg.lambda_lo_bid           = regime["lambda_lo_bid"]
    env.cfg.lambda_lo_ask           = regime["lambda_lo_ask"]
    env.cfg.lambda_cancel_per_share = regime["lambda_cancel_per_share"]
    env.cfg.lo_alpha                = regime["lo_alpha"]
    env.cfg.lo_best_supply          = regime["lo_best_supply"]
    env.cfg.as_kappa                = regime["as_kappa"]


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov policy with deterministic size (book + inventory)
# ---------------------------------------------------------------------------

def sample_as_action(
    obs: Observation,
    cfg: EnvConfig,
    t: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Avellaneda-Stoikov quoting with a mild inventory tilt on size.

    Quote prices:
        r           = mid - γ σ² (T-t)/T · inv
        half_spread = 0.5 γ σ² (T-t)/T + log(1 + γ/κ) / γ
    γ is constant across regimes (MM's risk preference).
    Regime-level differentiation of half_spread comes from σ (volatility
    channel) and κ (microstructure channel).

    Size:
        q0    = max(1, 0.15 · depth_mean_top3)
        q_bid = max(1, q0 · exp(η · inv))
        q_ask = max(1, q0 · exp(-η · inv))
    A single law governs size; only a floor at 1, no upper clamp.
    """
    mid = obs["mid"]
    inv = obs["inventory"]
    L = cfg.L
    time_remaining = max(1e-3, (cfg.T_max - t) / cfg.T_max)

    # A-S quote prices
    r = mid - cfg.as_gamma * (cfg.sigma_mid ** 2) * time_remaining * inv
    half_spread = (
        0.5 * cfg.as_gamma * (cfg.sigma_mid ** 2) * time_remaining
        + np.log(1.0 + cfg.as_gamma / cfg.as_kappa) / cfg.as_gamma
    )

    k_bid = float(max(1, min(L, round((mid - (r - half_spread)) / cfg.tick_size))))
    k_ask = float(max(1, min(L, round(((r + half_spread) - mid) / cfg.tick_size))))

    # Robust depth estimate: average of top-3 levels
    top_k = min(3, L)
    bid_top = obs["book"][0, :top_k, 1].mean()
    ask_top = obs["book"][1, :top_k, 1].mean()
    depth_mean = max(1.0, 0.5 * (bid_top + ask_top))

    # Natural size scale: ~15% of the top-3 depth
    q0 = max(1.0, 0.15 * depth_mean)

    # Mild inventory tilt (no upper clamp; only a floor at 1)
    eta = cfg.as_eta
    q_bid = float(max(1.0, q0 * np.exp(eta * inv)))
    q_ask = float(max(1.0, q0 * np.exp(-eta * inv)))

    return k_bid, k_ask, q_bid, q_ask

# ---------------------------------------------------------------------------
# Switch schedule
# ---------------------------------------------------------------------------

def generate_switch_schedule(
    start_regime: int,
    T_max: int,
    rng: np.random.Generator,
    max_switches: int = 2,
    warmup_frac: float = 0.15,
) -> list[tuple[int, int]]:
    n_switches = rng.integers(1, max_switches + 1)
    t_min = int(T_max * warmup_frac)
    t_max = int(T_max * (1 - warmup_frac))
    if t_max <= t_min:
        return []
    switch_times = sorted(rng.choice(range(t_min, t_max), size=n_switches, replace=False))
    schedule = []
    current = start_regime
    n_regimes = len(REGIMES)
    for t in switch_times:
        candidates = [r for r in range(n_regimes) if r != current]
        new_regime = rng.choice(candidates)
        schedule.append((int(t), int(new_regime)))
        current = new_regime
    return schedule

# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    cfg: EnvConfig,
    start_regime: int,
    rng: np.random.Generator,
    switch_schedule: list[tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    cfg_r = apply_regime(cfg, start_regime)
    env = MarketMakingEnv(cfg_r)
    # reset() runs a silent warmup (without MM) so the book starts at the
    # endogenous equilibrium of the current regime. No external baseline is set.
    obs = env.reset(seed=int(rng.integers(0, 2**31)))

    switch_schedule = switch_schedule or []
    switch_dict = {t: r for t, r in switch_schedule}

    obs_list, act_list, rew_list = [], [], []
    nobs_list, reg_list, sw_list = [], [], []
    ts_list, inv_list, tleft_list = [], [], []

    current_regime = start_regime

    for t in range(cfg.T_max):
        is_switch = 0
        if t in switch_dict:
            new_regime = switch_dict[t]
            # Structural shock: parameters change, book state persists.
            apply_regime_to_env(env, new_regime)
            current_regime = new_regime
            is_switch = 1

        inventory_t = obs["inventory"]
        time_left_t = (cfg.T_max - t) / cfg.T_max
        obs_vec = obs_to_vector(obs, cfg.L)

        action = sample_as_action(obs, env.cfg, t, rng)

        next_obs, reward, done, _ = env.step(action)
        next_obs_vec = obs_to_vector(next_obs, cfg.L)

        obs_list.append(obs_vec)
        act_list.append(np.array(action, dtype=np.float32))
        rew_list.append(reward)
        nobs_list.append(next_obs_vec)
        reg_list.append(current_regime)
        sw_list.append(is_switch)
        ts_list.append(t)
        inv_list.append(inventory_t)
        tleft_list.append(time_left_t)

        obs = next_obs
        if done:
            break

    return {
        "observations": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.float32),
        "rewards": np.array(rew_list, dtype=np.float32),
        "next_observations": np.array(nobs_list, dtype=np.float32),
        "regimes": np.array(reg_list, dtype=np.int8),
        "switch_mask": np.array(sw_list, dtype=np.int8),
        "timesteps": np.array(ts_list, dtype=np.int32),
        "inventories": np.array(inv_list, dtype=np.float32),
        "time_left": np.array(tleft_list, dtype=np.float32),
    }

# ---------------------------------------------------------------------------
# Multi-regime dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    cfg: EnvConfig | None = None,
    seed: int = 42,
    verbose: bool = True,
    shuffle: bool = True,
) -> dict[str, np.ndarray]:
    cfg = cfg or EnvConfig()
    rng = np.random.default_rng(seed)

    n_regimes = len(REGIMES)
    n_mixed = int(cfg.N_episodes * cfg.mixed_regime_frac)
    n_pure = cfg.N_episodes - n_mixed
    eps_per_regime = n_pure // n_regimes
    remainder = n_pure % n_regimes

    all_parts: list[dict] = []
    ep_counter = 0

    if verbose:
        print("Policy: Avellaneda-Stoikov only (as_mix_ratio=1.0)")
        print(f"Shared MM params: γ={cfg.as_gamma}, α_inventory={cfg.alpha_inventory}, "
              f"η={cfg.as_eta}, warmup_steps={cfg.warmup_steps}")

    # Pure episodes
    for i, regime in enumerate(REGIMES):
        n_eps = eps_per_regime + (1 if i < remainder else 0)
        if verbose:
            print(
                f"\nRegime {i} — {regime['name']:10s} "
                f"σ={regime['sigma_mid']:.3f} "
                f"κ={regime['as_kappa']:.0f} "
                f"p_inf={regime['p_informed']:.2f} "
                f"λ_mo={regime['lambda_mo_buy']:.2f} "
                f"mo_size={regime['mo_size_lambda']:.1f} "
                f"λ_cancel/share={regime['lambda_cancel_per_share']:.2f} "
                f"pure episodes={n_eps}"
            )

        for ep in range(n_eps):
            part = run_episode(cfg, start_regime=i, rng=rng)
            N_steps = len(part["rewards"])
            part["episode_ids"] = np.full(N_steps, ep_counter, dtype=np.int32)
            all_parts.append(part)
            ep_counter += 1

            if verbose and (ep + 1) % max(1, n_eps // 3) == 0:
                print(
                    f"  ep {ep+1:4d}/{n_eps} policy=as     "
                    f"steps={N_steps} regime={i}"
                )

    # Mixed regime episodes
    if verbose and n_mixed > 0:
        print(f"\nMixed regime episodes: {n_mixed}")

    for ep in range(n_mixed):
        start_regime = int(rng.integers(0, n_regimes))
        schedule = generate_switch_schedule(
            start_regime, cfg.T_max, rng,
            max_switches=cfg.max_switches,
            warmup_frac=cfg.switch_warmup_frac,
        )

        part = run_episode(cfg, start_regime=start_regime, rng=rng, switch_schedule=schedule)
        N_steps = len(part["rewards"])
        part["episode_ids"] = np.full(N_steps, ep_counter, dtype=np.int32)
        all_parts.append(part)
        ep_counter += 1

        if verbose and (ep + 1) % max(1, n_mixed // 3) == 0:
            print(
                f"  mixed ep {ep+1:4d}/{n_mixed} policy=as     "
                f"steps={N_steps} switches={len(schedule)} "
                f"regimes={[start_regime] + [r for _, r in schedule]}"
            )

    dataset = {
        key: np.concatenate([p[key] for p in all_parts], axis=0)
        for key in all_parts[0].keys()
    }
    N = len(dataset["rewards"])

    if shuffle:
        perm = rng.permutation(N)
        dataset = {k: v[perm] for k, v in dataset.items()}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {N:,} transitions da {ep_counter} episodi")
        print(f"  observations shape  : {dataset['observations'].shape}")
        print(f"  actions shape       : {dataset['actions'].shape}")
        print(f"  timesteps shape     : {dataset['timesteps'].shape}")
        print(f"  inventories shape   : {dataset['inventories'].shape}")
        print(f"  time_left shape     : {dataset['time_left'].shape}")
        print(
            f"  rewards      — mean={dataset['rewards'].mean():.4f} "
            f"std={dataset['rewards'].std():.4f}"
        )
        print(
            f"  inventories  — mean={dataset['inventories'].mean():.3f} "
            f"std={dataset['inventories'].std():.3f} "
            f"max_abs={np.abs(dataset['inventories']).max():.1f}"
        )
        counts = np.bincount(dataset["regimes"], minlength=n_regimes)
        for i, regime in enumerate(REGIMES):
            frac = counts[i] / N * 100
            print(
                f"  regime {i} ({regime['name']:10s}): "
                f"{counts[i]:7d} transitions ({frac:.1f}%)"
            )
        n_switches = dataset["switch_mask"].sum()
        n_mixed_actual = len(set(
            dataset["episode_ids"][dataset["switch_mask"] == 1].tolist()
        ))
        print(f"  regime switches     : {n_switches:,} totali")
        print(f"  episodi misti       : {n_mixed_actual}")

    return dataset


def save_dataset(dataset: dict[str, np.ndarray], path: str) -> None:
    np.savez_compressed(path, **dataset)
    print(f"Dataset salvato in {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera dataset offline multi-regime LOB (A-S only)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--T_max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--mixed_frac", type=float, default=None)
    parser.add_argument("--no_shuffle", action="store_true")
    args = parser.parse_args()

    cfg = EnvConfig()
    if args.episodes is not None:
        cfg.N_episodes = args.episodes
    if args.T_max is not None:
        cfg.T_max = args.T_max
    if args.out is not None:
        cfg.dataset_path = args.out
    if args.mixed_frac is not None:
        cfg.mixed_regime_frac = args.mixed_frac

    print(f"Generazione {cfg.N_episodes} episodi x {cfg.T_max} step "
          f"su {len(REGIMES)} regimi (policy=A-S only)...")
    dataset = generate_dataset(cfg, seed=args.seed, shuffle=not args.no_shuffle)
    save_dataset(dataset, cfg.dataset_path)
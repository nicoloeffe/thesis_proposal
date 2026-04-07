"""
simulate.py — Generazione dataset offline multi-regime.

Regimi diversificati: non solo σ_mid e p_informed, ma tutta la microstruttura
del book (frequenza MO, aggressività, liquidità LO, cancellazioni).
Questo rende i regimi distinguibili anche da un singolo snapshot.

Sequenze miste: una frazione degli episodi ha 1-2 regime switch intra-episodio.
Il regime label è per-step, non per-episodio.

Struttura del dataset salvato:
    observations      : (N_total, obs_dim)
    actions           : (N_total, 3)
    rewards           : (N_total,)
    next_observations : (N_total, obs_dim)
    regimes           : (N_total,)   — 0/1/2, per-step
    episode_ids       : (N_total,)   — identifies episode boundaries
    switch_mask       : (N_total,)   — 1 at switch points, 0 otherwise
"""

from __future__ import annotations

import numpy as np
import argparse
from dataclasses import replace

from config import EnvConfig
from env import MarketMakingEnv, Observation


# ---------------------------------------------------------------------------
# Regime definitions — diversified microstructure
# ---------------------------------------------------------------------------

def _as_gamma(sigma_mid: float, tick_size: float = 0.01, inv_typical: float = 5.0) -> float:
    """Optimal A-S risk-aversion: gamma = tick_size / (sigma^2 * inv_typical)."""
    return tick_size / (sigma_mid ** 2 * inv_typical)


REGIMES = [
    {
        # low_vol: calm market. Factors vs baseline (mid_vol):
        # σ 0.4x, λ_mo 0.7x, mo_size 0.7x, λ_lo 1.1x, λ_cancel 0.6x, p_inf 0.4x
        "name": "low_vol",
        "sigma_mid":         0.008,
        "p_informed":        0.08,
        "lambda_mo_buy":     0.35,
        "lambda_mo_sell":    0.35,
        "mo_size_lambda":    2.1,
        "lambda_lo_bid":     1.1,
        "lambda_lo_ask":     1.1,
        "lambda_cancel_bid": 0.30,
        "lambda_cancel_ask": 0.30,
    },
    {
        # mid_vol: baseline. Calibrated so λ_lo ≈ λ_mo + λ_cancel
        # (near-balanced order flow, Cont & de Larrard 2013).
        "name": "mid_vol",
        "sigma_mid":         0.020,
        "p_informed":        0.20,
        "lambda_mo_buy":     0.50,
        "lambda_mo_sell":    0.50,
        "mo_size_lambda":    3.0,
        "lambda_lo_bid":     1.00,
        "lambda_lo_ask":     1.00,
        "lambda_cancel_bid": 0.50,
        "lambda_cancel_ask": 0.50,
    },
    {
        # high_vol: stressed market. Factors vs baseline:
        # σ 2.5x, λ_mo 1.5x, mo_size 1.5x, λ_lo 0.7x, λ_cancel 1.5x, p_inf 1.75x
        "name": "high_vol",
        "sigma_mid":         0.050,
        "p_informed":        0.35,
        "lambda_mo_buy":     0.75,
        "lambda_mo_sell":    0.75,
        "mo_size_lambda":    4.5,
        "lambda_lo_bid":     0.70,
        "lambda_lo_ask":     0.70,
        "lambda_cancel_bid": 0.75,
        "lambda_cancel_ask": 0.75,
    },
]
# Compute A-S gamma from formula for each regime
for _r in REGIMES:
    _r["as_gamma"] = _as_gamma(_r["sigma_mid"])


# ---------------------------------------------------------------------------
# Observation helper
# ---------------------------------------------------------------------------

def obs_to_vector(obs: Observation, L: int) -> np.ndarray:
    """Flatten an observation dict into a 1-D numpy array."""
    book_flat = obs["book"].flatten()
    scalars = np.array([obs["mid"], obs["spread"], obs["imbalance"], obs["inventory"]])
    return np.concatenate([book_flat, scalars])


# ---------------------------------------------------------------------------
# Apply regime to env config
# ---------------------------------------------------------------------------

def apply_regime(cfg: EnvConfig, regime_idx: int) -> EnvConfig:
    """Create a copy of cfg with all regime-specific parameters applied."""
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
        lambda_cancel_bid=regime["lambda_cancel_bid"],
        lambda_cancel_ask=regime["lambda_cancel_ask"],
        as_gamma=regime["as_gamma"],
    )


def apply_regime_to_env(env: MarketMakingEnv, regime_idx: int) -> None:
    """Switch the env's config to a new regime in-place (for mid-episode switch)."""
    regime = REGIMES[regime_idx]
    env.cfg.sigma_mid       = regime["sigma_mid"]
    env.cfg.p_informed      = regime["p_informed"]
    env.cfg.lambda_mo_buy   = regime["lambda_mo_buy"]
    env.cfg.lambda_mo_sell  = regime["lambda_mo_sell"]
    env.cfg.mo_size_lambda  = regime["mo_size_lambda"]
    env.cfg.lambda_lo_bid   = regime["lambda_lo_bid"]
    env.cfg.lambda_lo_ask   = regime["lambda_lo_ask"]
    env.cfg.lambda_cancel_bid = regime["lambda_cancel_bid"]
    env.cfg.lambda_cancel_ask = regime["lambda_cancel_ask"]
    env.cfg.as_gamma        = regime["as_gamma"]


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------

def sample_random_action(cfg: EnvConfig, rng: np.random.Generator) -> tuple[float, float, float]:
    delta_bid = float(rng.integers(cfg.rand_delta_low, cfg.rand_delta_high + 1))
    delta_ask = float(rng.integers(cfg.rand_delta_low, cfg.rand_delta_high + 1))
    qty = float(rng.integers(cfg.rand_qty_low, cfg.rand_qty_high + 1))
    return delta_bid, delta_ask, qty


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov policy
# ---------------------------------------------------------------------------

def sample_as_action(
    obs: Observation,
    cfg: EnvConfig,
    t: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    mid = obs["mid"]
    inv = obs["inventory"]
    time_remaining = max(1e-3, (cfg.T_max - t) / cfg.T_max)

    r = mid - cfg.as_gamma * (cfg.sigma_mid ** 2) * time_remaining * inv
    half_spread = (
        0.5 * cfg.as_gamma * (cfg.sigma_mid ** 2) * time_remaining
        + np.log(1.0 + cfg.as_gamma / cfg.as_kappa) / cfg.as_gamma
    )

    k_bid_raw = (mid - (r - half_spread)) / cfg.tick_size
    k_ask_raw = ((r + half_spread) - mid) / cfg.tick_size

    noise = rng.normal(0.0, 0.5)
    k_bid = float(np.clip(round(k_bid_raw + noise), 1, cfg.rand_delta_high))
    k_ask = float(np.clip(round(k_ask_raw + noise), 1, cfg.rand_delta_high))
    qty = float(rng.integers(cfg.rand_qty_low, cfg.rand_qty_high + 1))
    return k_bid, k_ask, qty


# ---------------------------------------------------------------------------
# Generate switch schedule for a mixed episode
# ---------------------------------------------------------------------------

def generate_switch_schedule(
    start_regime: int,
    T_max: int,
    rng: np.random.Generator,
    max_switches: int = 2,
    warmup_frac: float = 0.15,
) -> list[tuple[int, int]]:
    """
    Generate a list of (timestep, new_regime) switch events.

    Switches happen only in the middle portion of the episode
    (between warmup_frac and 1-warmup_frac) to avoid transients.
    The new regime is always different from the current one.

    Returns:
        List of (t_switch, regime_idx) — sorted by time.
    """
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
        # Pick a different regime
        candidates = [r for r in range(n_regimes) if r != current]
        new_regime = rng.choice(candidates)
        schedule.append((int(t), int(new_regime)))
        current = new_regime

    return schedule


# ---------------------------------------------------------------------------
# Single episode rollout (supports mid-episode regime switches)
# ---------------------------------------------------------------------------

def run_episode(
    cfg: EnvConfig,
    start_regime: int,
    policy: str,
    rng: np.random.Generator,
    switch_schedule: list[tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """
    Run a single episode, optionally with intra-episode regime switches.

    Returns dict with per-step arrays:
        observations, actions, rewards, next_observations,
        regimes (per-step!), switch_mask
    """
    cfg_r = apply_regime(cfg, start_regime)
    env = MarketMakingEnv(cfg_r)
    ep_seed = int(rng.integers(0, 2**31))
    obs = env.reset(seed=ep_seed)

    switch_schedule = switch_schedule or []
    switch_dict = {t: r for t, r in switch_schedule}

    obs_list = []
    act_list = []
    rew_list = []
    nobs_list = []
    reg_list = []
    sw_list = []

    current_regime = start_regime

    for t in range(cfg.T_max):
        # Check for regime switch
        is_switch = 0
        if t in switch_dict:
            new_regime = switch_dict[t]
            apply_regime_to_env(env, new_regime)
            current_regime = new_regime
            is_switch = 1

        obs_vec = obs_to_vector(obs, cfg.L)

        if policy == "as":
            action = sample_as_action(obs, env.cfg, t, rng)
        else:
            action = sample_random_action(env.cfg, rng)

        next_obs, reward, done, _ = env.step(action)
        next_obs_vec = obs_to_vector(next_obs, cfg.L)

        obs_list.append(obs_vec)
        act_list.append(np.array(action, dtype=np.float32))
        rew_list.append(reward)
        nobs_list.append(next_obs_vec)
        reg_list.append(current_regime)
        sw_list.append(is_switch)

        obs = next_obs
        if done:
            break

    return {
        "observations":      np.array(obs_list, dtype=np.float32),
        "actions":           np.array(act_list, dtype=np.float32),
        "rewards":           np.array(rew_list, dtype=np.float32),
        "next_observations": np.array(nobs_list, dtype=np.float32),
        "regimes":           np.array(reg_list, dtype=np.int8),
        "switch_mask":       np.array(sw_list, dtype=np.int8),
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
    """
    Generate a multi-regime dataset with:
    - Pure single-regime episodes (equally split across regimes)
    - Mixed episodes with intra-episode regime switches

    Per-step regime labels and switch_mask are always included.
    """
    cfg = cfg or EnvConfig()
    rng = np.random.default_rng(seed)

    n_regimes = len(REGIMES)
    n_mixed = int(cfg.N_episodes * cfg.mixed_regime_frac)
    n_pure = cfg.N_episodes - n_mixed
    eps_per_regime = n_pure // n_regimes
    remainder = n_pure % n_regimes

    all_parts: list[dict] = []
    ep_counter = 0

    # Decide policy per episode
    n_as = int(cfg.N_episodes * cfg.as_mix_ratio)
    n_random = cfg.N_episodes - n_as
    policies = ["as"] * n_as + ["random"] * n_random
    rng.shuffle(policies)
    policy_idx = 0

    # --- Pure single-regime episodes ---
    for i, regime in enumerate(REGIMES):
        n_eps = eps_per_regime + (1 if i < remainder else 0)
        if verbose:
            print(f"\nRegime {i} — {regime['name']:10s}  "
                  f"sigma={regime['sigma_mid']:.3f}  "
                  f"p_informed={regime['p_informed']:.2f}  "
                  f"lambda_mo={regime['lambda_mo_buy']:.1f}  "
                  f"pure episodes={n_eps}")

        for ep in range(n_eps):
            policy = policies[policy_idx % len(policies)]
            policy_idx += 1

            part = run_episode(cfg, start_regime=i, policy=policy, rng=rng)
            N_steps = len(part["rewards"])
            part["episode_ids"] = np.full(N_steps, ep_counter, dtype=np.int32)
            all_parts.append(part)
            ep_counter += 1

            if verbose and (ep + 1) % max(1, n_eps // 3) == 0:
                print(f"    ep {ep+1:4d}/{n_eps}  policy={policy:6s}  "
                      f"steps={N_steps}  regime={i}")

    # --- Mixed regime episodes ---
    if verbose and n_mixed > 0:
        print(f"\nMixed regime episodes: {n_mixed}")

    for ep in range(n_mixed):
        start_regime = int(rng.integers(0, n_regimes))
        policy = policies[policy_idx % len(policies)]
        policy_idx += 1

        schedule = generate_switch_schedule(
            start_regime, cfg.T_max, rng,
            max_switches=cfg.max_switches,
            warmup_frac=cfg.switch_warmup_frac,
        )

        part = run_episode(
            cfg, start_regime=start_regime, policy=policy,
            rng=rng, switch_schedule=schedule,
        )
        N_steps = len(part["rewards"])
        part["episode_ids"] = np.full(N_steps, ep_counter, dtype=np.int32)
        all_parts.append(part)
        ep_counter += 1

        if verbose and (ep + 1) % max(1, n_mixed // 3) == 0:
            n_sw = len(schedule)
            regimes_hit = [start_regime] + [r for _, r in schedule]
            print(f"    mixed ep {ep+1:4d}/{n_mixed}  policy={policy:6s}  "
                  f"steps={N_steps}  switches={n_sw}  "
                  f"regimes={regimes_hit}")

    # --- Concatenate ---
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
        print(f"Dataset: {N:,} transitions from {ep_counter} episodes")
        print(f"  observations shape : {dataset['observations'].shape}")
        print(f"  shuffle            : {shuffle}")
        print(f"  rewards — mean={dataset['rewards'].mean():.4f}  "
              f"std={dataset['rewards'].std():.4f}")
        counts = np.bincount(dataset["regimes"], minlength=n_regimes)
        for i, regime in enumerate(REGIMES):
            frac = counts[i] / N * 100
            print(f"  regime {i} ({regime['name']:10s}): "
                  f"{counts[i]:7d} transitions ({frac:.1f}%)")
        n_switches = dataset["switch_mask"].sum()
        print(f"  regime switches    : {n_switches:,} total")
        n_mixed_actual = len(set(
            dataset["episode_ids"][dataset["switch_mask"] == 1]
        ))
        print(f"  mixed episodes     : {n_mixed_actual}")


    return dataset


def save_dataset(dataset: dict[str, np.ndarray], path: str) -> None:
    np.savez_compressed(path, **dataset)
    print(f"Dataset saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate offline multi-regime LOB dataset")
    parser.add_argument("--episodes",     type=int,   default=None)
    parser.add_argument("--T_max",        type=int,   default=None)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--out",          type=str,   default=None)
    parser.add_argument("--as_ratio",     type=float, default=None)
    parser.add_argument("--mixed_frac",   type=float, default=None,
                        help="Fraction of episodes with regime switches (default: 0.30)")
    parser.add_argument("--no_shuffle",   action="store_true",
                        help="Keep episodes in order, add episode_ids (for world model)")
    args = parser.parse_args()

    cfg = EnvConfig()
    if args.episodes   is not None: cfg.N_episodes        = args.episodes
    if args.T_max      is not None: cfg.T_max             = args.T_max
    if args.out        is not None: cfg.dataset_path      = args.out
    if args.as_ratio   is not None: cfg.as_mix_ratio      = args.as_ratio
    if args.mixed_frac is not None: cfg.mixed_regime_frac = args.mixed_frac

    print(f"Generating {cfg.N_episodes} episodes x {cfg.T_max} steps "
          f"across {len(REGIMES)} regimes ...")
    print(f"  Pure: {int(cfg.N_episodes * (1 - cfg.mixed_regime_frac))}  "
          f"Mixed: {int(cfg.N_episodes * cfg.mixed_regime_frac)}")
    dataset = generate_dataset(cfg, seed=args.seed, shuffle=not args.no_shuffle)
    save_dataset(dataset, cfg.dataset_path)
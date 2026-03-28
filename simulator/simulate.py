"""
simulate.py — Generazione dataset offline per la Fase 1.

Genera un dataset multi-regime: ogni regime ha una sigma_mid diversa,
modellando condizioni di mercato calmo, normale e volatile.
All'interno di ogni regime, gli episodi usano policy mista A-S / random.

Struttura del dataset salvato:
    observations      : (N_total, obs_dim)
    actions           : (N_total, 3)
    rewards           : (N_total,)
    next_observations : (N_total, obs_dim)
    regimes           : (N_total,)   — 0=low, 1=mid, 2=high volatility

dove obs_dim = 2*L*2 + 4  (book flattened + mid, spread, imbalance, inventory).
"""

from __future__ import annotations

import numpy as np
import argparse
from dataclasses import replace

from config import EnvConfig
from env import MarketMakingEnv, Observation


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

def _as_gamma(sigma_mid: float, tick_size: float = 0.01, inv_typical: float = 5.0) -> float:
    """Optimal A-S risk-aversion: gamma = tick_size / (sigma^2 * inv_typical)."""
    return tick_size / (sigma_mid ** 2 * inv_typical)


REGIMES = [
    {"name": "low_vol",  "sigma_mid": 0.01, "p_informed": 0.1},
    {"name": "mid_vol",  "sigma_mid": 0.02, "p_informed": 0.2},
    {"name": "high_vol", "sigma_mid": 0.04, "p_informed": 0.3},
]
# Compute as_gamma from formula for each regime
for _r in REGIMES:
    _r["as_gamma"] = _as_gamma(_r["sigma_mid"])


# ---------------------------------------------------------------------------
# Observation helper
# ---------------------------------------------------------------------------

def obs_to_vector(obs: Observation, L: int) -> np.ndarray:
    """Flatten an observation dict into a 1-D numpy array."""
    book_flat = obs["book"].flatten()
    scalars = np.array([obs["mid"], obs["spread"], obs["imbalance"], obs["inventory"]])
    return np.concatenate([book_flat, scalars])            # (2*L*2 + 4,)


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------

def sample_random_action(cfg: EnvConfig, rng: np.random.Generator) -> tuple[float, float, float]:
    """Sample a random action: (delta_bid, delta_ask, qty) ~ Uniform."""
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
    """
    Avellaneda-Stoikov reservation price policy.

        r_t         = mid_t - gamma * sigma^2 * (T-t)/T * inv_t
        half_spread = 0.5 * gamma * sigma^2 * (T-t)/T
                    + ln(1 + gamma/kappa) / gamma

    Offsets in ticks from mid, with small Gaussian noise for diversity.
    """
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
# Single-regime rollout
# ---------------------------------------------------------------------------

def run_regime(
    cfg: EnvConfig,
    n_episodes: int,
    regime_idx: int,
    rng: np.random.Generator,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Run n_episodes rollouts for a single regime.
    Returns dict with observations, actions, rewards, next_observations, regimes.
    """
    regime = REGIMES[regime_idx]
    # Override regime-specific params on a copy of cfg
    cfg_r = replace(cfg,
                    sigma_mid=regime["sigma_mid"],
                    p_informed=regime["p_informed"],
                    as_gamma=regime["as_gamma"])

    env = MarketMakingEnv(cfg_r)

    all_obs:      list[np.ndarray] = []
    all_actions:  list[np.ndarray] = []
    all_rewards:  list[float]      = []
    all_next_obs: list[np.ndarray] = []

    n_as     = int(n_episodes * cfg_r.as_mix_ratio)
    n_random = n_episodes - n_as
    episode_policies = ["as"] * n_as + ["random"] * n_random
    rng.shuffle(episode_policies)

    for ep, policy in enumerate(episode_policies):
        ep_seed = int(rng.integers(0, 2**31))
        obs = env.reset(seed=ep_seed)
        obs_vec = obs_to_vector(obs, cfg_r.L)

        ep_steps = 0
        for t in range(cfg_r.T_max):
            if policy == "as":
                action = sample_as_action(obs, cfg_r, t, rng)
            else:
                action = sample_random_action(cfg_r, rng)

            next_obs, reward, done, _ = env.step(action)
            next_obs_vec = obs_to_vector(next_obs, cfg_r.L)

            all_obs.append(obs_vec)
            all_actions.append(np.array(action, dtype=np.float32))
            all_rewards.append(reward)
            all_next_obs.append(next_obs_vec)

            obs_vec = next_obs_vec
            obs = next_obs
            ep_steps += 1
            if done:
                break

        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"    ep {ep+1:4d}/{n_episodes}  policy={policy:6s}  steps={ep_steps}")

    N = len(all_rewards)
    return {
        "observations":      np.array(all_obs,      dtype=np.float32),
        "actions":           np.array(all_actions,  dtype=np.float32),
        "rewards":           np.array(all_rewards,  dtype=np.float32),
        "next_observations": np.array(all_next_obs, dtype=np.float32),
        "regimes":           np.full(N, regime_idx, dtype=np.int8),
    }


# ---------------------------------------------------------------------------
# Multi-regime dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    cfg: EnvConfig | None = None,
    seed: int | None = 42,
    verbose: bool = True,
    shuffle: bool = True,
) -> dict[str, np.ndarray]:
    """
    Generate a multi-regime dataset.
    If shuffle=False, keeps episodes in order and adds episode_ids array.
    """
    cfg = cfg or EnvConfig()
    rng = np.random.default_rng(seed)

    n_regimes = len(REGIMES)
    eps_per_regime = cfg.N_episodes // n_regimes
    remainder = cfg.N_episodes % n_regimes

    all_parts: list[dict] = []
    ep_counter = 0

    for i, regime in enumerate(REGIMES):
        n_eps = eps_per_regime + (1 if i < remainder else 0)
        if verbose:
            print(f"\nRegime {i} — {regime['name']:10s}  "
                  f"sigma={regime['sigma_mid']:.3f}  "
                  f"p_informed={regime['p_informed']:.1f}  "
                  f"episodes={n_eps}")
        part = run_regime(cfg, n_eps, regime_idx=i, rng=rng, verbose=verbose)

        # Add episode_ids
        N_part = len(part["rewards"])
        ep_ids = np.repeat(
            np.arange(ep_counter, ep_counter + n_eps, dtype=np.int32),
            cfg.T_max
        )[:N_part]
        part["episode_ids"] = ep_ids
        ep_counter += n_eps
        all_parts.append(part)

    dataset = {
        key: np.concatenate([p[key] for p in all_parts], axis=0)
        for key in all_parts[0].keys()
    }

    N = len(dataset["rewards"])

    if shuffle:
        perm = rng.permutation(N)
        dataset = {k: v[perm] for k, v in dataset.items()}

    if verbose:
        print(f"\nDataset: {N} transitions  "
              f"({eps_per_regime} eps/regime × {n_regimes} regimes)")
        print(f"  observations shape : {dataset['observations'].shape}")
        print(f"  actions shape      : {dataset['actions'].shape}")
        print(f"  shuffle            : {shuffle}")
        print(f"  rewards — mean={dataset['rewards'].mean():.4f}  "
              f"std={dataset['rewards'].std():.4f}")
        counts = np.bincount(dataset["regimes"], minlength=n_regimes)
        for i, regime in enumerate(REGIMES):
            print(f"  regime {i} ({regime['name']:10s}): {counts[i]:7d} transitions")

    return dataset


def save_dataset(dataset: dict[str, np.ndarray], path: str) -> None:
    np.savez_compressed(path, **dataset)
    print(f"Dataset saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate offline multi-regime LOB dataset")
    parser.add_argument("--episodes",   type=int,   default=None)
    parser.add_argument("--T_max",      type=int,   default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out",        type=str,   default=None)
    parser.add_argument("--as_ratio",   type=float, default=None)
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Keep episodes in order, add episode_ids (for world model)")
    args = parser.parse_args()

    cfg = EnvConfig()
    if args.episodes is not None: cfg.N_episodes   = args.episodes
    if args.T_max    is not None: cfg.T_max        = args.T_max
    if args.out      is not None: cfg.dataset_path = args.out
    if args.as_ratio is not None: cfg.as_mix_ratio = args.as_ratio

    print(f"Generating {cfg.N_episodes} episodes x {cfg.T_max} steps "
          f"across {len(REGIMES)} regimes ...")
    dataset = generate_dataset(cfg, seed=args.seed, shuffle=not args.no_shuffle)
    save_dataset(dataset, cfg.dataset_path)

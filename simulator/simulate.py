"""
simulate.py — Generazione dataset offline multi-regime con A-S + size deterministica

Dataset salvato:
  observations      : (N_total, obs_dim)
  actions           : (N_total, 4)  — [delta_bid, delta_ask, q_bid, q_ask]
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
# ---------------------------------------------------------------------------

def _as_gamma(sigma_mid: float, tick_size: float = 0.01, inv_typical: float = 5.0) -> float:
    """Optimal A-S risk-aversion: gamma = tick_size / (sigma^2 * inv_typical)."""
    return tick_size / (sigma_mid ** 2 * inv_typical)



REGIMES = [
    {
        "name": "low_vol",
        "sigma_mid": 0.008,
        "p_informed": 0.02,
        "lambda_mo_buy": 0.35,
        "lambda_mo_sell": 0.35,
        "mo_size_lambda": 3.0,     # CORREZIONE: Book denso = trader piazzano ordini grandi senza paura
        "lambda_lo_bid": 0.95,
        "lambda_lo_ask": 0.95,
        "lambda_cancel_bid": 0.35,
        "lambda_cancel_ask": 0.35,
        "lo_alpha": 0.40,          # Pendenza decisa, liquidità centrata al best
        "baseline_vol": 60.0,
        "as_kappa": 80.0,
    },
    {
        "name": "mid_vol",
        "sigma_mid": 0.015,
        "p_informed": 0.10,
        "lambda_mo_buy": 0.50,
        "lambda_mo_sell": 0.50,
        "mo_size_lambda": 2.5,     # Transizione logica
        "lambda_lo_bid": 0.70,
        "lambda_lo_ask": 0.70,
        "lambda_cancel_bid": 0.45,
        "lambda_cancel_ask": 0.45,
        "lo_alpha": 0.25,          # Dispersione moderata
        "baseline_vol": 35.0,
        "as_kappa": 50.0,
    },
    {
        "name": "high_vol",
        "sigma_mid": 0.028,
        "p_informed": 0.18,
        "lambda_mo_buy": 0.75,
        "lambda_mo_sell": 0.75,
        "mo_size_lambda": 2.0,     # CORREZIONE: Book sottile = frammentazione aggressiva (Order Splitting)
        "lambda_lo_bid": 0.65,     # Aumentato per non farsi mangiare vivo dai cancel
        "lambda_lo_ask": 0.65,
        "lambda_cancel_bid": 0.50,
        "lambda_cancel_ask": 0.50,
        "lo_alpha": 0.15,          # CORREZIONE: Abbastanza piatto da fare la "gobba", ma non un rettangolo piatto.
        "baseline_vol": 20.0,      # Rapporto esatto di 3x rispetto a low_vol (60/20)
        "as_kappa": 30.0,
    },
]
# (Il calcolo di as_gamma rimane uguale)
for _r in REGIMES:
    _r["as_gamma"] = _as_gamma(_r["sigma_mid"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def obs_to_vector(obs: Observation, L: int) -> np.ndarray:
    book_flat = obs["book"].flatten()
    scalars   = np.array([obs["mid"], obs["spread"], obs["imbalance"], obs["inventory"]])
    return np.concatenate([book_flat, scalars])


def apply_regime(cfg: EnvConfig, regime_idx: int) -> EnvConfig:
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
        lo_alpha=regime["lo_alpha"],
        as_gamma=regime["as_gamma"],
        as_kappa=regime["as_kappa"], # FIX: Ora applichiamo la kappa specifica del regime
    )

def apply_regime_to_env(env: MarketMakingEnv, regime_idx: int) -> None:
    regime = REGIMES[regime_idx]
    env.cfg.sigma_mid         = regime["sigma_mid"]
    env.cfg.p_informed        = regime["p_informed"]
    env.cfg.lambda_mo_buy     = regime["lambda_mo_buy"]
    env.cfg.lambda_mo_sell    = regime["lambda_mo_sell"]
    env.cfg.mo_size_lambda    = regime["mo_size_lambda"]
    env.cfg.lambda_lo_bid     = regime["lambda_lo_bid"]
    env.cfg.lambda_lo_ask     = regime["lambda_lo_ask"]
    env.cfg.lambda_cancel_bid = regime["lambda_cancel_bid"]
    env.cfg.lambda_cancel_ask = regime["lambda_cancel_ask"]
    env.cfg.lo_alpha          = regime["lo_alpha"]
    env.cfg.as_gamma          = regime["as_gamma"]
    env.cfg.as_kappa          = regime["as_kappa"] 
    env._baseline_vol_per_side = regime["baseline_vol"]


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov policy con size deterministica (book + inventory)
# ---------------------------------------------------------------------------

def sample_as_action(
    obs: Observation,
    cfg: EnvConfig,
    t: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    mid = obs["mid"]
    inv = obs["inventory"]
    L = cfg.L
    time_remaining = max(1e-3, (cfg.T_max - t) / cfg.T_max)

    # Prezzi A-S classici
    r = mid - cfg.as_gamma * (cfg.sigma_mid ** 2) * time_remaining * inv
    half_spread = (
        0.5 * cfg.as_gamma * (cfg.sigma_mid ** 2) * time_remaining
        + np.log(1.0 + cfg.as_gamma / cfg.as_kappa) / cfg.as_gamma
    )

    noise = 0.0
    k_bid = float(max(1, min(L, round((mid - (r - half_spread)) / cfg.tick_size + noise))))
    k_ask = float(max(1, min(L, round(((r + half_spread) - mid) / cfg.tick_size + noise))))

    # --- Robust depth estimate: average of top-3 levels ---
    top_k = min(3, L)
    bid_top = obs["book"][0, :top_k, 1].mean()
    ask_top = obs["book"][1, :top_k, 1].mean()
    depth_mean = max(1.0, 0.5 * (bid_top + ask_top))

    # Scala naturale della size: ~15% profondità media
    q0 = max(1.0, 0.15 * depth_mean)

    # Distorsione asimmetrica basata su inventory (alla Stanford)
    eta   = cfg.as_eta
    q_bid = q0 * np.exp( eta * inv)   
    q_ask = q0 * np.exp(-eta * inv)   
    # Clamp soft: non superare la depth media
    q_bid = float(max(1.0, min(q_bid, depth_mean)))
    q_ask = float(max(1.0, min(q_ask, depth_mean)))

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
    obs = env.reset(seed=int(rng.integers(0, 2**31)))

    # Set regime-specific baseline after reset
    env._baseline_vol_per_side = REGIMES[start_regime]["baseline_vol"]

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

    # solo A-S
    if verbose:
        print("Policy: Avellaneda-Stoikov only (as_mix_ratio=1.0)")

    # Pure episodes
    for i, regime in enumerate(REGIMES):
        n_eps = eps_per_regime + (1 if i < remainder else 0)
        if verbose:
            print(
                f"\nRegime {i} — {regime['name']:10s} "
                f"sigma={regime['sigma_mid']:.3f} "
                f"p_informed={regime['p_informed']:.2f} "
                f"lambda_mo={regime['lambda_mo_buy']:.2f} "
                f"mo_size={regime['mo_size_lambda']:.1f} "
                f"baseline_vol={regime['baseline_vol']:.0f} "
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
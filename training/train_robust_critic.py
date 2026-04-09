"""
train_robust_critic.py — Backward Robust Dynamic Programming (frozen backbone).

    t = T:   V_rob(z, T) = V_nom(z_T)
    t < T:   V_rob(z, t) = inf_Q E_Q[ r̃_t + γ V_rob(z', t+1) ]

Key fix: the last layer is initialized from NOMINAL weights (not random).
The optimizer only needs to learn δ = V_rob - V_nom, which is small and
systematic. With random init, the optimizer wastes capacity reconstructing
V_nom and the δ signal is lost in the noise.

Uso:
  python training/train_robust_critic.py --epsilon 0.05
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.critic import ValueNetwork
from models.world_model import LOBWorldModel, WorldModelConfig
from models.dro import DROConfig


# ---------------------------------------------------------------------------
# Frozen-φ critic
# ---------------------------------------------------------------------------

class FrozenPhiCritic(nn.Module):
    def __init__(self, critic: ValueNetwork):
        super().__init__()
        self.critic = critic
        self.last_layer_name = None
        for name, module in critic.named_modules():
            if isinstance(module, nn.Linear):
                self.last_layer_name = name
        for name, param in critic.named_parameters():
            if not name.startswith(self.last_layer_name):
                param.requires_grad_(False)
        parts = self.last_layer_name.split(".")
        obj = critic
        for p in parts:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        self.last_layer = obj
        self.n_frozen = sum(p.numel() for n, p in critic.named_parameters()
                           if not n.startswith(self.last_layer_name))
        self.n_trainable = sum(p.numel() for n, p in critic.named_parameters()
                              if n.startswith(self.last_layer_name))

    def forward(self, x): return self.critic(x)

    def get_last_layer_state(self):
        return {"weight": self.last_layer.weight.data.clone(),
                "bias": self.last_layer.bias.data.clone()}

    def set_last_layer_state(self, state):
        self.last_layer.weight.data.copy_(state["weight"])
        self.last_layer.bias.data.copy_(state["bias"])

    def trainable_params(self):
        return [self.last_layer.weight, self.last_layer.bias]


# ---------------------------------------------------------------------------
# Batched inner solver (Mahalanobis)
# ---------------------------------------------------------------------------

def batched_inner_solve(critic, y, lam, sigma, cfg):
    mahal = (cfg.cost_type == "mahalanobis")
    R = cfg.trust_radius_sigma
    y_det = y.detach()
    if mahal:
        sig2 = sigma ** 2 + 1e-8
        inv_sig2 = 1.0 / sig2
        lr_per_dim = torch.clamp(sig2 / (2.0 * lam.unsqueeze(-1) + 1.0), max=cfg.inner_lr)
        trust_bound = R * sigma
    else:
        lr_scalar = torch.clamp(0.5 / (2.0 * lam + 1.0), max=cfg.inner_lr).unsqueeze(-1)
        trust_bound = R
    x = y.clone().detach().requires_grad_(True)
    for step in range(cfg.inner_steps):
        v = critic(x)
        if mahal:
            penalty = lam * (((x - y_det) ** 2) * inv_sig2).sum(dim=-1)
        else:
            penalty = lam * ((x - y_det) ** 2).sum(dim=-1)
        obj = (v + penalty).sum()
        grad = torch.autograd.grad(obj, x, create_graph=False)[0]
        if mahal:
            x_new = x - lr_per_dim * grad
        else:
            x_new = x - lr_scalar * grad
        x_new = torch.clamp(x_new, y_det - trust_bound, y_det + trust_bound)
        x = x_new.detach().requires_grad_(True)
    return x.detach()


# ---------------------------------------------------------------------------
# Batched robust targets
# ---------------------------------------------------------------------------

def batched_robust_targets(critic, pi_t, mu_t, log_sig_t, r_t, epsilon, cfg,
                           device, chunk_size=5000, verbose=True):
    M, K, D = mu_t.shape
    gamma = cfg.gamma
    n_s = cfg.n_samples_per_component
    mahal = (cfg.cost_type == "mahalanobis")
    targets = torch.zeros(M)
    gen = torch.Generator(device=device).manual_seed(0)
    t0 = time.time()
    n_chunks = (M + chunk_size - 1) // chunk_size

    for c in range(n_chunks):
        i0, i1 = c * chunk_size, min((c + 1) * chunk_size, M)
        nc = i1 - i0
        pi_c = pi_t[i0:i1].to(device)
        mu_c = mu_t[i0:i1].to(device)
        sig_c = torch.exp(log_sig_t[i0:i1].to(device))
        r_c = r_t[i0:i1]

        if n_s > 0:
            eps_noise = torch.randn(nc, K, n_s, D, device=device, generator=gen)
            y_flat = (mu_c.unsqueeze(2) + sig_c.unsqueeze(2) * eps_noise).reshape(nc, K*n_s, D)
            w_flat = (pi_c / n_s).unsqueeze(2).expand(nc, K, n_s).reshape(nc, K*n_s)
            sig_flat = sig_c.unsqueeze(2).expand(nc, K, n_s, D).reshape(nc, K*n_s, D)
            m = K * n_s
        else:
            y_flat, w_flat, sig_flat = mu_c, pi_c, sig_c
            m = K

        y_all = y_flat.reshape(nc * m, D)
        sig_all = sig_flat.reshape(nc * m, D)
        lam_low = torch.full((nc,), 1e-4, device=device)
        lam_high = torch.full((nc,), cfg.lambda_init, device=device)

        for _ in range(cfg.outer_steps):
            lam_mid = (lam_low + lam_high) / 2
            x_star = batched_inner_solve(critic, y_all, lam_mid.repeat_interleave(m), sig_all, cfg)
            x_resh = x_star.reshape(nc, m, D)
            if mahal:
                tp = (((x_resh - y_flat)**2) / (sig_flat.reshape(nc, m, D)**2 + 1e-8)).sum(-1)
            else:
                tp = ((x_resh - y_flat)**2).sum(-1)
            wt = (w_flat * tp).sum(-1)
            too_high = wt > epsilon
            lam_low = torch.where(too_high, lam_mid, lam_low)
            lam_high = torch.where(too_high, lam_high, lam_mid)
            if ((lam_high - lam_low) / (lam_high + 1e-8) < 0.01).all():
                break

        lam_final = ((lam_low + lam_high) / 2).repeat_interleave(m)
        x_star = batched_inner_solve(critic, y_all, lam_final, sig_all, cfg)
        with torch.no_grad():
            v_star = critic(x_star)
        v_robust = (w_flat * v_star.reshape(nc, m)).sum(-1)
        targets[i0:i1] = r_c + gamma * v_robust.cpu()

        if verbose and (c + 1) % max(1, n_chunks // 5) == 0:
            print(f"   chunk {c+1}/{n_chunks}  W={wt.mean():.4f}  "
                  f"λ*={(lam_low+lam_high).mean()/2:.2f}  t={time.time()-t0:.1f}s")

    if verbose:
        print(f"   Done: {M} seqs  mean={targets.mean():.4f}  "
              f"std={targets.std():.4f}  ({time.time()-t0:.1f}s)")
    return targets


# ---------------------------------------------------------------------------
# Pre-compute GMMs
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_gmm_by_timestep(world_model, sequences, actions, device, batch_size=64):
    M = sequences.shape[0]
    pi_l, mu_l, ls_l = [], [], []
    for i in range(0, M, batch_size):
        pi, mu, ls = world_model(sequences[i:i+batch_size].to(device),
                                  actions[i:i+batch_size].to(device))
        pi_l.append(pi.cpu()); mu_l.append(mu.cpu()); ls_l.append(ls.cpu())
    return torch.cat(pi_l), torch.cat(mu_l), torch.cat(ls_l)


# ---------------------------------------------------------------------------
# Fit last layer (initialized from nominal, learning δ)
# ---------------------------------------------------------------------------

def fit_last_layer(critic, z, targets, train_mask, val_mask, device,
                   epochs=10, batch_size=2048, lr=1e-3):
    train_ds = TensorDataset(z[train_mask], targets[train_mask])
    val_ds = TensorDataset(z[val_mask], targets[val_mask])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.Adam(critic.trainable_params(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = critic.get_last_layer_state()

    for epoch in range(1, epochs + 1):
        critic.critic.eval()
        for zb, yb in train_loader:
            zb, yb = zb.to(device), yb.to(device)
            loss = loss_fn(critic(zb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        vl = 0.0
        with torch.no_grad():
            for zb, yb in val_loader:
                zb, yb = zb.to(device), yb.to(device)
                vl += loss_fn(critic(zb), yb).item()
        vl /= len(val_loader)
        if vl < best_val:
            best_val = vl
            best_state = critic.get_last_layer_state()

    # Restore best
    critic.set_last_layer_state(best_state)
    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # World model
    wm_ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    wm_cfg = WorldModelConfig()
    for k, v in wm_ckpt["cfg"].items(): setattr(wm_cfg, k, v)
    world_model = LOBWorldModel(wm_cfg).to(device)
    world_model.load_state_dict(wm_ckpt["model"]); world_model.eval()
    for p in world_model.parameters(): p.requires_grad_(False)

    # Critic
    cr_ckpt = torch.load(args.critic_ckpt, map_location=device, weights_only=False)
    cr_cfg = cr_ckpt["cfg"]
    reward_stats = cr_ckpt["reward_stats"]
    print(f"Reward stats: μ={reward_stats['mean']:.6f}  σ={reward_stats['std']:.6f}")

    base_critic = ValueNetwork(d_latent=cr_cfg["d_latent"], hidden=cr_cfg["hidden"],
                               n_layers=cr_cfg["n_layers"]).to(device)
    base_critic.load_state_dict(cr_ckpt["model"])

    nom_critic = ValueNetwork(d_latent=cr_cfg["d_latent"], hidden=cr_cfg["hidden"],
                              n_layers=cr_cfg["n_layers"]).to(device)
    nom_critic.load_state_dict(cr_ckpt["model"]); nom_critic.eval()
    for p in nom_critic.parameters(): p.requires_grad_(False)

    critic = FrozenPhiCritic(base_critic)
    print(f"Critic: {critic.n_frozen:,} frozen + {critic.n_trainable:,} trainable")

    # Save nominal last-layer weights — used as init for every timestep
    nominal_last_layer = critic.get_last_layer_state()
    print(f"Nominal last layer saved (init for each timestep)")

    # Dataset
    data = np.load(args.dataset)
    sequences = torch.from_numpy(data["sequences"])
    actions = torch.from_numpy(data["actions"])
    rewards = torch.from_numpy(data["rewards"])
    M, Np1, D = sequences.shape; N = Np1 - 1

    rewards_norm = (rewards - reward_stats["mean"]) / (reward_stats["std"] + 1e-8)

    if args.n_sequences < M:
        rng = np.random.default_rng(42)
        idx = rng.choice(M, size=args.n_sequences, replace=False)
        sequences, actions, rewards_norm = sequences[idx], actions[idx], rewards_norm[idx]
        M = args.n_sequences
    print(f"Using {M:,} sequences, N={N}")

    # Split
    if "episode_ids" in data:
        eids = data["episode_ids"]
        if args.n_sequences < len(eids): eids = eids[idx]
        ue = np.unique(eids); np.random.default_rng(123).shuffle(ue)
        nv = max(1, int(len(ue) * 0.1)); ve = set(ue[-nv:])
        is_val = np.array([e in ve for e in eids])
    else:
        is_val = np.random.default_rng(123).random(M) < 0.1
    train_mask, val_mask = torch.from_numpy(~is_val), torch.from_numpy(is_val)
    print(f"Split: {train_mask.sum():,} train, {val_mask.sum():,} val")

    # GMMs
    print("\nPre-computing GMMs...")
    pi_all, mu_all, log_sig_all = precompute_gmm_by_timestep(world_model, sequences, actions, device)

    # DRO config
    dro_cfg = DROConfig(inner_steps=args.inner_steps, inner_lr=args.inner_lr,
                        outer_steps=args.outer_steps, lambda_init=args.lambda_init,
                        trust_radius_sigma=args.trust_radius, cost_type=args.cost_type,
                        n_samples_per_component=args.n_samples_per_comp, gamma=args.gamma)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"BACKWARD ROBUST DP (frozen backbone, nominal init)")
    print(f"  ε={args.epsilon}  γ={args.gamma}  cost={dro_cfg.cost_type}")
    print(f"  N={N}  inner={dro_cfg.inner_steps}  R={dro_cfg.trust_radius_sigma}σ")
    print(f"  Last layer init: NOMINAL (learning δ = V_rob - V_nom)")
    print(f"{'='*70}")

    layer_weights = {}
    diagnostics = []
    t_total = time.time()

    for t in range(N - 1, -1, -1):
        print(f"\n{'─'*50}\n  t={t}  ({N-1-t+1}/{N})")
        z_t = sequences[:, t, :]
        r_t = rewards_norm[:, t]

        if t == N - 1:
            # Terminal: V_rob(z, N) = V_nom(z_N)
            z_N = sequences[:, N, :].to(device)
            with torch.no_grad(): v_term = nom_critic(z_N).cpu()
            targets = r_t + args.gamma * v_term
            print(f"  Terminal bootstrap: V_nom mean={v_term.mean():.4f}")
        else:
            # Load V_rob(·, t+1) — fitted at previous backward step
            critic.set_last_layer_state(layer_weights[t + 1])
            critic.critic.eval()

            print(f"  Computing robust targets with V_rob(·, {t+1})...")
            targets = batched_robust_targets(
                critic, pi_all[:, t], mu_all[:, t], log_sig_all[:, t], r_t,
                epsilon=args.epsilon, cfg=dro_cfg, device=device,
                chunk_size=args.chunk_size)

        # Initialize last layer from NOMINAL (not random!)
        # Optimizer only needs to learn δ = V_rob - V_nom
        critic.set_last_layer_state(nominal_last_layer)

        print(f"  Fitting V_rob(·, {t})...")
        val_loss = fit_last_layer(critic, z_t, targets, train_mask, val_mask, device,
                                  epochs=args.fit_epochs, batch_size=args.fit_batch_size,
                                  lr=args.fit_lr)
        layer_weights[t] = critic.get_last_layer_state()

        # Diagnostics
        critic.critic.eval()
        with torch.no_grad():
            v_pred = critic(z_t.to(device)).cpu()
            v_nom = nom_critic(z_t.to(device)).cpu()
        residual = (v_pred - targets).abs().mean().item()
        delta_mean = (v_nom - v_pred).mean().item()  # how much worse than nominal

        diag = {"t": t, "target_mean": targets.mean().item(),
                "target_std": targets.std().item(), "val_loss": val_loss,
                "residual": residual, "v_rob_mean": v_pred.mean().item(),
                "v_nom_mean": v_nom.mean().item(), "delta": delta_mean}
        diagnostics.append(diag)
        print(f"  → target={diag['target_mean']:.4f}  V_rob={diag['v_rob_mean']:.4f}  "
              f"V_nom={diag['v_nom_mean']:.4f}  δ={delta_mean:.4f}  "
              f"val={val_loss:.4f}  res={residual:.4f}")

    print(f"\n{'='*70}")
    print(f"BACKWARD ROBUST DP COMPLETE  ({time.time()-t_total:.1f}s)")
    print(f"{'─'*60}")
    print(f"{'t':>3s} {'target':>8s} {'V_rob':>8s} {'V_nom':>8s} "
          f"{'δ':>8s} {'res':>8s} {'val':>8s}")
    print(f"{'─'*60}")
    for d in sorted(diagnostics, key=lambda x: x["t"]):
        print(f"{d['t']:3d} {d['target_mean']:8.4f} {d['v_rob_mean']:8.4f} "
              f"{d['v_nom_mean']:8.4f} {d['delta']:8.4f} "
              f"{d['residual']:8.4f} {d['val_loss']:8.4f}")

    ckpt = {"layer_weights": layer_weights, "diagnostics": diagnostics,
            "cfg": cr_cfg, "reward_stats": reward_stats, "epsilon": args.epsilon,
            "gamma": args.gamma, "N": N, "cost_type": args.cost_type,
            "critic_base": base_critic.state_dict(),
            "nominal_last_layer": nominal_last_layer}
    torch.save(ckpt, ckpt_dir / "robust_critic_backward.pt")
    print(f"\nSaved: {ckpt_dir / 'robust_critic_backward.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Backward Robust DP")
    p.add_argument("--critic_ckpt", type=str, default="checkpoints/critic_best.pt")
    p.add_argument("--wm_ckpt", type=str, default="checkpoints/wm_best.pt")
    p.add_argument("--dataset", type=str, default="data/wm_dataset.npz")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--n_sequences", type=int, default=50000)
    p.add_argument("--inner_steps", type=int, default=50)
    p.add_argument("--inner_lr", type=float, default=0.05)
    p.add_argument("--outer_steps", type=int, default=20)
    p.add_argument("--lambda_init", type=float, default=50.0)
    p.add_argument("--trust_radius", type=float, default=3.0)
    p.add_argument("--cost_type", type=str, default="mahalanobis")
    p.add_argument("--n_samples_per_comp", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--chunk_size", type=int, default=5000)
    p.add_argument("--fit_epochs", type=int, default=20)
    p.add_argument("--fit_batch_size", type=int, default=2048)
    p.add_argument("--fit_lr", type=float, default=1e-3)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = p.parse_args()
    main(args)
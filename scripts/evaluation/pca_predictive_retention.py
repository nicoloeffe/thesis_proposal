#!/usr/bin/env python3
"""
pca_predictive_retention.py

Variance-vs-prediction retention curves for the Horizon JEPA state representation.

Context
-------
Follow-up to a prior project finding: PCA at 32 dims on the last-timestep concat
(concat512) destroyed the `future_delta` probe targets. That was a single
operating point. This script turns it into two *curves* vs the number of retained
components m, for two unsupervised reduction candidates:

  * flat PCA on the 512-d last-timestep concat
  * per-token PCA on the 4 semantic tokens (128-d each), structure preserving

For each m it reports, on the held-out split:
  (1) cumulative explained variance retained             -- a *variance* metric
  (2) predictive R^2 retained toward observable targets  -- a *prediction* metric
      (closed-form ridge from the m PCA scores; optional MLP curve behind a flag)

The gap between (1) and (2) is exactly the "predictive information living in
low-variance directions" concern. If the curves track each other, PCA-by-variance
is a safe bottleneck for the world model; if they diverge, it is not, and the
point at which (2) recovers tells the honest *predictive* dimensionality.

This is a diagnostic/eval script. It does NOT train or modify any model. It reuses
the data pipeline of probe_jepa_horizon_readouts.py (encoder loading, valid
endpoints / grouped split, observable targets, readout extraction). Place it next
to probe_jepa_horizon_readouts.py in the project tree and run from the project
root.

Example
-------
python -m scripts.evaluation.pca_predictive_retention \
  --dataset data/lobench_processed.npz \
  --horizon_ckpt checkpoints/jepa_horizon/v1/best.pt \
  --out_dir validation/pca_retention/jepa_horizon_best \
  --max_train_samples 100000 --max_val_samples 50000 \
  --batch_size 512 --num_workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve()
for p in [HERE.parent, *HERE.parents, Path.cwd(), *Path.cwd().parents]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# ---- Project imports (same as probe_jepa_horizon_readouts.py) ----
try:
    from training.historical.train_jepa_horizon import (
        HorizonJEPAEncoder,
        HorizonJEPAEncoderConfig,
    )
except Exception as e:
    raise SystemExit(
        "Cannot import Horizon JEPA classes from training/train_jepa_horizon.py. "
        "Run this script from the thesis project root. Original error: " + repr(e)
    )

try:
    from training.train_tokenizer_t import (  # type: ignore
        compute_valid_endpoints,
        normalize_book_window,
        derive_raw_features_array,
        compute_future_feature_targets,
        compute_vol_targets,
        fit_target_standardizer,
        apply_standardizer,
        grouped_split_by_stock_day,
        compute_stock_stats_train_only,
    )
except Exception as e:
    raise SystemExit(
        "Cannot import tokenizer data utilities from training/train_tokenizer_t.py. "
        "Run this script from the thesis project root. Original error: " + repr(e)
    )


# =============================================================================
# Utilities (copied verbatim from probe_jepa_horizon_readouts.py)
# =============================================================================

def robust_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def local_grouped_split_by_stock_day(
    stock_ids: np.ndarray,
    day_ids: np.ndarray,
    valid_t: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_day = int(day_ids.max()) + 1
    composite = stock_ids[valid_t].astype(np.int64) * max_day + day_ids[valid_t].astype(np.int64)
    groups = np.unique(composite)
    rng.shuffle(groups)
    n_val = max(1, int(round(val_frac * len(groups))))
    val_groups = set(groups[:n_val].tolist())
    val_mask = np.array([g in val_groups for g in composite])
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def maybe_subsample(pos: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n is None or max_n <= 0 or len(pos) <= max_n:
        return pos
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pos, size=max_n, replace=False))


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    yc = y_true - y_true.mean(axis=0, keepdims=True)
    ss_tot = (yc ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def split_target_indices(names: List[str]) -> Tuple[List[int], List[int]]:
    """future_delta targets vs realized_vol targets (matches probe convention)."""
    future_idx = [i for i, n in enumerate(names) if not n.startswith("realized_vol")]
    vol_idx = [i for i, n in enumerate(names) if n.startswith("realized_vol")]
    return future_idx, vol_idx


def to_numpy_stats(stock_stats: Dict) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stock_stats.items()}


# =============================================================================
# Dataset / encoder extraction (copied verbatim from the probe)
# =============================================================================

class RawWindowDataset(Dataset):
    def __init__(
        self,
        book: np.ndarray,
        mid_z: np.ndarray,
        stock_ids: np.ndarray,
        valid_t: np.ndarray,
        stock_stats: Dict[str, np.ndarray],
        K: int,
    ):
        self.book = book
        self.mid_z = mid_z
        self.stock_ids = stock_ids
        self.valid_t = valid_t
        self.stock_stats = stock_stats
        self.K = K

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, idx: int):
        t = int(self.valid_t[idx])
        s = int(self.stock_ids[t])
        K = self.K
        book_win = self.book[t - K + 1: t + 1]
        mid_win = self.mid_z[t - K + 1: t + 1]
        book_norm = normalize_book_window(book_win, mid_win, s, self.stock_stats)
        return torch.from_numpy(book_norm).float(), torch.tensor(s, dtype=torch.long)


def load_horizon_jepa_encoder(ckpt_path: str, device: torch.device) -> Tuple[HorizonJEPAEncoder, Dict]:
    ckpt = robust_torch_load(ckpt_path, device)
    enc_cfg = HorizonJEPAEncoderConfig.from_dict(ckpt["enc_cfg"])
    enc = HorizonJEPAEncoder(enc_cfg).to(device)
    state = ckpt.get("online_state_dict", ckpt.get("encoder_state_dict", None))
    if state is None:
        raise ValueError("Horizon JEPA checkpoint must contain online_state_dict")
    enc.load_state_dict(state)
    enc.eval()
    return enc, ckpt


@torch.no_grad()
def extract_token_grids(
    encoder: HorizonJEPAEncoder,
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label: str,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    last_tokens, last_concat = [], []
    t0 = time.time()
    n = 0
    for book, stock_ids in loader:
        book = book.to(device, non_blocking=True)
        stock_ids = stock_ids.to(device, non_blocking=True)
        grid = encoder(book, stock_ids, mask=None)                  # (B,K,S,D)
        last = grid[:, -1, :, :]                                    # (B,4,128)
        last_tokens.append(last.detach().cpu().numpy().astype(np.float32))
        last_concat.append(last.reshape(last.shape[0], -1).detach().cpu().numpy().astype(np.float32))
        n += book.shape[0]
    dt = time.time() - t0
    print(f"  Horizon JEPA {label}: extracted {n:,} token readouts in {dt:.1f}s ({n/max(dt,1e-9):.0f}/s)")
    out = {
        "last_tokens": np.concatenate(last_tokens, axis=0),     # (N,4,128)
        "last_concat512": np.concatenate(last_concat, axis=0),  # (N,512)
    }
    for k, v in out.items():
        print(f"    {k:18s} shape={v.shape}")
    return out


# =============================================================================
# PCA  (covariance form: centered only, NO per-dimension scaling)
# -----------------------------------------------------------------------------
# Centered-only PCA is an orthogonal map: distances on the retained subspace are
# preserved (up to truncation). This keeps the reduction isometric, consistent
# with an energy-score world model that lives on Euclidean distances.
# =============================================================================

def pca_fit(x_train: np.ndarray) -> Dict[str, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True).astype(np.float64)
    xc = x_train.astype(np.float64) - mu
    n = xc.shape[0]
    cov = (xc.T @ xc) / max(n - 1, 1)
    evals, evecs = np.linalg.eigh(cov)            # ascending
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    return {"mean": mu, "components": evecs, "explained_var": evals}


def pca_scores(pca: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xc = x.astype(np.float64) - pca["mean"]
    return xc @ pca["components"]                  # (n, d), columns ordered by variance


# =============================================================================
# Closed-form ridge: predictive R^2 from a block of PCA scores
# =============================================================================

def standardize_block(train: np.ndarray, val: np.ndarray, eps: float = 1e-8):
    mu = train.mean(axis=0, keepdims=True)
    sd = np.maximum(train.std(axis=0, keepdims=True), eps)
    return (train - mu) / sd, (val - mu) / sd


def ridge_r2(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: List[float],
) -> Tuple[np.ndarray, float, float]:
    """Fit ridge for each lambda, pick best mean-R^2 on val. Features must be
    standardized; targets are assumed (near) zero-mean (probe standardizer)."""
    s_train = s_train.astype(np.float64)
    s_val = s_val.astype(np.float64)
    y_train = y_train.astype(np.float64)
    m = s_train.shape[1]
    A = s_train.T @ s_train
    B = s_train.T @ y_train
    eye = np.eye(m)
    best_r2 = None
    best_mean = -np.inf
    best_lam = lambdas[0]
    for lam in lambdas:
        W = np.linalg.solve(A + lam * eye, B)
        pred = s_val @ W
        r2 = r2_per_target(y_val, pred)
        mean_r2 = float(np.mean(r2))
        if mean_r2 > best_mean:
            best_mean, best_lam, best_r2 = mean_r2, lam, r2
    return best_r2.astype(np.float64), best_mean, best_lam


# =============================================================================
# Optional nonlinear (MLP) retention curve
# =============================================================================

def mlp_r2(
    s_train: np.ndarray,
    y_train: np.ndarray,
    s_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    hidden: int,
    epochs: int,
    patience: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 1024,
) -> np.ndarray:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset

    in_dim = s_train.shape[1]
    out_dim = y_train.shape[1]
    model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden, out_dim),
    ).to(device)
    tds = TensorDataset(torch.from_numpy(s_train).float(), torch.from_numpy(y_train).float())
    loader = DataLoader(tds, batch_size=batch_size, shuffle=True, drop_last=False)
    xv = torch.from_numpy(s_val).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.03)
    best_mse, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_mse = float(F.mse_loss(model(xv), yv).item())
        if val_mse < best_mse - 1e-7:
            best_mse, bad = val_mse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xv).detach().cpu().numpy()
    return r2_per_target(y_val, pred).astype(np.float64)


# =============================================================================
# Retention curves
# =============================================================================

def parse_int_grid(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def flat_pca_curve(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    m_grid: List[int],
    lambdas: List[float],
    target_names: List[str],
    run_mlp: bool,
    mlp_kwargs: Optional[dict],
    device: torch.device,
) -> Dict:
    """PCA on the flat 512-d concat. Curves of explained variance and predictive
    R^2 retained vs number of components m."""
    d = x_train.shape[1]
    m_grid = sorted({min(m, d) for m in m_grid if m >= 1})
    pca = pca_fit(x_train)
    sc_train_full = pca_scores(pca, x_train)
    sc_val_full = pca_scores(pca, x_val)
    sc_train_std, sc_val_std = standardize_block(sc_train_full, sc_val_full)

    ev = pca["explained_var"]
    ev_cum = np.cumsum(ev) / np.maximum(ev.sum(), 1e-12)
    fut_idx, vol_idx = split_target_indices(target_names)

    # Ceiling: ridge on the full representation.
    r2_full, _, lam_full = ridge_r2(sc_train_std, y_train, sc_val_std, y_val, lambdas)
    ceil = {
        "all": float(np.mean(r2_full)),
        "future_delta": float(np.mean(r2_full[fut_idx])) if fut_idx else float("nan"),
        "realized_vol": float(np.mean(r2_full[vol_idx])) if vol_idx else float("nan"),
    }
    return _run_curve(
        "flat_pca_concat512", m_grid, ev_cum, sc_train_std, sc_val_std,
        y_train, y_val, lambdas, fut_idx, vol_idx, ceil,
        component_selector=lambda m: np.arange(m),
        run_mlp=run_mlp, mlp_kwargs=mlp_kwargs, device=device,
    )


def per_token_pca_curve(
    tok_train: np.ndarray,
    tok_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    c_grid: List[int],
    lambdas: List[float],
    target_names: List[str],
    run_mlp: bool,
    mlp_kwargs: Optional[dict],
    device: torch.device,
) -> Dict:
    """Structure-preserving PCA: one PCA per semantic token (128-d each). Budget
    is c components per token -> total m = 4*c. Score blocks are concatenated."""
    n_tok, d_tok = tok_train.shape[1], tok_train.shape[2]
    c_grid = sorted({min(c, d_tok) for c in c_grid if c >= 1})

    per_tok = []
    for t in range(n_tok):
        pca = pca_fit(tok_train[:, t, :])
        st = pca_scores(pca, tok_train[:, t, :])
        sv = pca_scores(pca, tok_val[:, t, :])
        per_tok.append({"pca": pca, "sc_train": st, "sc_val": sv})

    # Full concatenated score matrix, token-blocked, then standardized once.
    sc_train_full = np.concatenate([p["sc_train"] for p in per_tok], axis=1)
    sc_val_full = np.concatenate([p["sc_val"] for p in per_tok], axis=1)
    sc_train_std, sc_val_std = standardize_block(sc_train_full, sc_val_full)

    total_ev = sum(float(p["pca"]["explained_var"].sum()) for p in per_tok)

    def selector(c: int) -> np.ndarray:
        # first c columns of each token block
        return np.concatenate([np.arange(t * d_tok, t * d_tok + c) for t in range(n_tok)])

    def ev_at(c: int) -> float:
        kept = sum(float(p["pca"]["explained_var"][:c].sum()) for p in per_tok)
        return kept / max(total_ev, 1e-12)

    ev_cum_map = {4 * c: ev_at(c) for c in c_grid}
    fut_idx, vol_idx = split_target_indices(target_names)

    r2_full, _, _ = ridge_r2(sc_train_std, y_train, sc_val_std, y_val, lambdas)
    ceil = {
        "all": float(np.mean(r2_full)),
        "future_delta": float(np.mean(r2_full[fut_idx])) if fut_idx else float("nan"),
        "realized_vol": float(np.mean(r2_full[vol_idx])) if vol_idx else float("nan"),
    }

    m_grid = [4 * c for c in c_grid]
    return _run_curve(
        "per_token_pca", m_grid, None, sc_train_std, sc_val_std,
        y_train, y_val, lambdas, fut_idx, vol_idx, ceil,
        component_selector=lambda m: selector(m // 4),
        run_mlp=run_mlp, mlp_kwargs=mlp_kwargs, device=device,
        ev_cum_map=ev_cum_map,
    )


def _run_curve(
    name, m_grid, ev_cum, sc_train_std, sc_val_std, y_train, y_val,
    lambdas, fut_idx, vol_idx, ceil, component_selector,
    run_mlp, mlp_kwargs, device, ev_cum_map=None,
) -> Dict:
    rows = []
    print(f"\n  [{name}] ceiling R^2  all={ceil['all']:.4f}  "
          f"future_delta={ceil['future_delta']:.4f}  realized_vol={ceil['realized_vol']:.4f}")
    print(f"  {'m':>5s} {'expl_var':>9s} {'R2_all':>8s} {'R2_fut':>8s} {'R2_vol':>8s} "
          f"{'ret_all':>8s} {'ret_fut':>8s}" + ("  R2_all_mlp" if run_mlp else ""))
    for m in m_grid:
        cols = component_selector(m)
        st = np.ascontiguousarray(sc_train_std[:, cols])
        sv = np.ascontiguousarray(sc_val_std[:, cols])
        r2, _, lam = ridge_r2(st, y_train, sv, y_val, lambdas)
        r2_all = float(np.mean(r2))
        r2_fut = float(np.mean(r2[fut_idx])) if fut_idx else float("nan")
        r2_vol = float(np.mean(r2[vol_idx])) if vol_idx else float("nan")
        if ev_cum_map is not None:
            ev_m = float(ev_cum_map[m])
        else:
            ev_m = float(ev_cum[m - 1])
        row = {
            "m": int(m),
            "explained_var_retained": ev_m,
            "r2_all": r2_all,
            "r2_future_delta": r2_fut,
            "r2_realized_vol": r2_vol,
            "r2_retained_all": r2_all / ceil["all"] if ceil["all"] > 1e-9 else float("nan"),
            "r2_retained_future_delta": (r2_fut / ceil["future_delta"]
                                         if ceil["future_delta"] > 1e-9 else float("nan")),
            "ridge_lambda": float(lam),
        }
        line = (f"  {m:5d} {ev_m:9.4f} {r2_all:8.4f} {r2_fut:8.4f} {r2_vol:8.4f} "
                f"{row['r2_retained_all']:8.4f} {row['r2_retained_future_delta']:8.4f}")
        if run_mlp:
            r2_mlp = mlp_r2(st.astype(np.float32), y_train.astype(np.float32),
                            sv.astype(np.float32), y_val.astype(np.float32),
                            device=device, **mlp_kwargs)
            row["r2_all_mlp"] = float(np.mean(r2_mlp))
            row["r2_future_delta_mlp"] = float(np.mean(r2_mlp[fut_idx])) if fut_idx else float("nan")
            line += f"  {row['r2_all_mlp']:9.4f}"
        print(line)
        rows.append(row)
    return {"name": name, "ceiling": ceil, "curve": rows}


# =============================================================================
# Plotting (optional, skipped gracefully if matplotlib is unavailable)
# =============================================================================

def save_plots(results: List[Dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable, skipping plots ({e!r})")
        return
    for res in results:
        rows = res["curve"]
        m = [r["m"] for r in rows]
        ev = [r["explained_var_retained"] for r in rows]
        ret_all = [r["r2_retained_all"] for r in rows]
        ret_fut = [r["r2_retained_future_delta"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(m, ev, "o-", label="explained variance retained", color="#888888")
        ax.plot(m, ret_all, "s-", label="predictive R² retained (all)", color="#1f77b4")
        ax.plot(m, ret_fut, "^-", label="predictive R² retained (future_delta)", color="#d62728")
        ax.set_xlabel("retained components m")
        ax.set_ylabel("fraction retained")
        ax.set_title(f"{res['name']} — variance vs prediction")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"curve_{res['name']}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  [plot] saved {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon_ckpt", required=True)
    p.add_argument("--out_dir", default="validation/pca_retention/jepa_horizon")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--max_train_samples", type=int, default=100000)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Curve controls
    p.add_argument("--m_grid_flat", type=str,
                   default="1,2,4,8,12,16,24,32,48,64,96,128,160,192,256,320,384,448,512",
                   help="Retained-component grid for flat PCA on concat512.")
    p.add_argument("--c_grid_per_token", type=str,
                   default="1,2,3,4,6,8,12,16,24,32,48,64,96,128",
                   help="Components-per-token grid for per-token PCA (total m = 4*c).")
    p.add_argument("--ridge_lambdas", type=str, default="0.1,1.0,10.0,100.0",
                   help="Ridge lambda grid; best mean-R^2 on val is reported per m.")
    p.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)

    # Optional nonlinear curve
    p.add_argument("--run_mlp_curve", action=argparse.BooleanOptionalAction, default=False,
                   help="Also fit a small MLP per m (slower; nonlinear retention).")
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--mlp_epochs", type=int, default=60)
    p.add_argument("--mlp_patience", type=int, default=12)

    # Targets (same defaults as probe_jepa_horizon_readouts.py)
    p.add_argument("--future_features", type=str,
                   default="d_spread_z,d_microprice_rel,d_best_bid_rel,d_best_ask_rel,d_top_imbalance")
    p.add_argument("--future_horizons", type=str, default="1,5,10,20")
    p.add_argument("--vol_horizons", type=str, default="5,20")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    m_grid_flat = parse_int_grid(args.m_grid_flat)
    c_grid_per_token = parse_int_grid(args.c_grid_per_token)
    lambdas = [float(x) for x in args.ridge_lambdas.split(",") if x.strip()]
    future_features = [x.strip() for x in args.future_features.split(",") if x.strip()]
    future_horizons = [int(x) for x in args.future_horizons.split(",") if x.strip()]
    vol_horizons = [int(x) for x in args.vol_horizons.split(",") if x.strip()]
    max_h = max(max(future_horizons), max(vol_horizons))

    print("=" * 92)
    print("PCA PREDICTIVE-RETENTION CURVES — variance vs predictive R^2 (Horizon JEPA state)")
    print("=" * 92)
    print(f"dataset      : {args.dataset}")
    print(f"horizon_ckpt : {args.horizon_ckpt}")
    print(f"device       : {device}")
    print(f"ridge lambdas: {lambdas}")

    print("\n[1/7] Loading Horizon JEPA checkpoint...")
    encoder, ckpt = load_horizon_jepa_encoder(args.horizon_ckpt, device)
    K = int(encoder.cfg.K)
    print(f"  epoch={ckpt.get('epoch', 'N/A')}  K={K} S={encoder.cfg.S} d_model={encoder.cfg.d_model}")

    print("\n[2/7] Loading raw LOBench...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    n_stocks = int(raw["min_spread_z_per_stock"].shape[0]) if "min_spread_z_per_stock" in raw.files else int(stock_ids.max() + 1)
    print(f"  N={len(mid_z):,} n_stocks={n_stocks} L={book.shape[2]}")

    print("\n[3/7] Valid endpoints and grouped split...")
    bid_v = book[:, 0, :, 1]
    ask_v = book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    valid_t = compute_valid_endpoints(stock_ids, day_ids, K, max_h, vol_mask)
    splitter = grouped_split_by_stock_day if grouped_split_by_stock_day is not None else local_grouped_split_by_stock_day
    train_pos, val_pos = splitter(stock_ids, day_ids, valid_t, args.val_frac, args.split_seed)
    train_pos = maybe_subsample(train_pos, args.max_train_samples, args.seed + 11)
    val_pos = maybe_subsample(val_pos, args.max_val_samples, args.seed + 17)
    train_t = valid_t[train_pos]
    val_t = valid_t[val_pos]
    print(f"  valid_t={len(valid_t):,} train={len(train_t):,} val={len(val_t):,} max_h={max_h}")

    print("\n[4/7] Building observable targets...")
    t0 = time.time()
    raw_feat, raw_names = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)
    fut_train_raw = compute_future_feature_targets(raw_feat, train_t, future_features, future_horizons)
    fut_val_raw = compute_future_feature_targets(raw_feat, val_t, future_features, future_horizons)
    vol_train_raw = compute_vol_targets(mid_z, train_t, vol_horizons, raw["min_spread_z_per_stock"], stock_ids)
    vol_val_raw = compute_vol_targets(mid_z, val_t, vol_horizons, raw["min_spread_z_per_stock"], stock_ids)
    y_train_raw = np.concatenate([fut_train_raw, vol_train_raw], axis=1).astype(np.float32)
    y_val_raw = np.concatenate([fut_val_raw, vol_val_raw], axis=1).astype(np.float32)
    y_mu, y_sd = fit_target_standardizer(y_train_raw)
    y_train = apply_standardizer(y_train_raw, y_mu, y_sd).astype(np.float32)
    y_val = apply_standardizer(y_val_raw, y_mu, y_sd).astype(np.float32)
    target_names = []
    for f in future_features:
        for h in future_horizons:
            target_names.append(f"{f}@{h}")
    for h in vol_horizons:
        target_names.append(f"realized_vol@{h}")
    print(f"  targets train={y_train.shape} val={y_val.shape} built in {time.time()-t0:.1f}s")

    print("\n[5/7] Normalization stats and datasets...")
    if "stock_stats" in ckpt:
        stock_stats = to_numpy_stats(ckpt["stock_stats"])
        print("  using stock_stats from Horizon JEPA checkpoint")
    elif compute_stock_stats_train_only is not None:
        stock_stats = compute_stock_stats_train_only(book, mid_z, stock_ids, day_ids, train_t, n_stocks)
        print("  computed train-only stock_stats")
    else:
        raise RuntimeError("No stock_stats in checkpoint and compute_stock_stats_train_only unavailable")

    ds_train = RawWindowDataset(book, mid_z, stock_ids, train_t, stock_stats, K)
    ds_val = RawWindowDataset(book, mid_z, stock_ids, val_t, stock_stats, K)

    print("\n[6/7] Extracting Horizon JEPA readouts...")
    r_train = extract_token_grids(encoder, ds_train, args.batch_size, args.num_workers, device, "train")
    r_val = extract_token_grids(encoder, ds_val, args.batch_size, args.num_workers, device, "val")

    print("\n[7/7] PCA retention curves...")
    mlp_kwargs = None
    if args.run_mlp_curve:
        mlp_kwargs = dict(hidden=args.mlp_hidden, epochs=args.mlp_epochs, patience=args.mlp_patience)
        print("  MLP curve ENABLED (slower).")

    results = []
    results.append(flat_pca_curve(
        r_train["last_concat512"], r_val["last_concat512"], y_train, y_val,
        m_grid_flat, lambdas, target_names, args.run_mlp_curve, mlp_kwargs, device,
    ))
    results.append(per_token_pca_curve(
        r_train["last_tokens"], r_val["last_tokens"], y_train, y_val,
        c_grid_per_token, lambdas, target_names, args.run_mlp_curve, mlp_kwargs, device,
    ))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args),
        "checkpoint": {"path": args.horizon_ckpt, "epoch": ckpt.get("epoch", None)},
        "target_names": target_names,
        "n_train": int(len(train_t)),
        "n_val": int(len(val_t)),
        "note": "Follow-up to prior finding: PCA32 on concat512 destroyed future_delta. "
                "Curves quantify at which m predictive R^2 recovers and whether "
                "per-token PCA preserves it better than flat PCA.",
    }
    with open(out_dir / "pca_retention_curves.json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    npz_payload = {}
    for res in results:
        rows = res["curve"]
        for key in ["m", "explained_var_retained", "r2_all", "r2_future_delta",
                    "r2_realized_vol", "r2_retained_all", "r2_retained_future_delta"]:
            npz_payload[f"{res['name']}__{key}"] = np.array([r[key] for r in rows], dtype=np.float64)
    np.savez_compressed(out_dir / "pca_retention_curves.npz", **npz_payload)

    if args.save_plots:
        save_plots(results, out_dir)

    print(f"\nSaved outputs to: {out_dir}")
    print("Read: gap between 'explained variance retained' and 'predictive R^2 retained'")
    print("      large gap  -> predictive signal in low-variance dirs (PCA-by-variance unsafe)")
    print("      small gap  -> PCA is a safe bottleneck; pick m where R^2 retained saturates")


if __name__ == "__main__":
    main()

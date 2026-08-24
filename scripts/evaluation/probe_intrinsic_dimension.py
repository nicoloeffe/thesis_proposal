#!/usr/bin/env python3
"""
probe_intrinsic_dimension.py — first non-linear probe of representation geometry.

Consumes the STAGE-1 readout dumps written by
experiment01.historical.extract_readouts_multiseed and,
per (arm, seed), measures TWO numbers on the fixed 512-dim readouts:

  * effective rank  (participation ratio  PR = (Σλ)² / Σλ²  of the covariance
    spectrum) — a LINEAR count of how many directions carry variance. This is a
    non-saturating effective-rank (unlike a cumulative-variance V90 threshold).

  * intrinsic dimension  (TwoNN, Facco et al. 2017) — a NON-LINEAR estimate of
    the dimension of the manifold the points actually lie on.

The read of interest is the GAP  PR − ID:
    PR ≫ ID   -> low-dim manifold folded through many variance directions
                 -> curved / organized structure  (encouraging).
    PR ≈ ID   -> the cloud genuinely fills its variance directions
                 -> diffuse / closer to isotropic noise  (claim weakens).

SCOPE CAVEAT — do not over-read:
    This measures the geometry of the WHOLE cloud, not of the directional signal
    specifically. A "structured" reading is a GREEN light for the target-
    conditioned test (level-sets / persistence, to do with Vaccarino). A
    "diffuse" reading is a YELLOW flag, NOT a verdict — the signal is a sub-part
    of the cloud and could be structured even if the whole cloud is not.

ESTIMATOR CAVEAT — read the contrast, not the absolute:
    ALL intrinsic-dimension estimators under-estimate high ID at finite sample
    size; TwoNN saturates when true ID is large relative to log(N). So the
    robust read is the CONTRAST BETWEEN ARMS (same N, same estimator, same
    ambient D), not the absolute number. Use --sweep to check whether ID keeps
    climbing with N (not saturated) or plateaus (saturated / trustworthy).

Does not touch any existing file. Reads dumps, writes two CSVs + a manifest.

Inputs (produced by the historical extraction module with --out_dir READOUT_DIR):
    READOUT_DIR/readouts/{arm}_seed{N}_ep{E:03d}.npz
        keys: last_concat512_train, last_concat512_val,
              tmean_concat512_train, tmean_concat512_val

Outputs (--out_dir):
    id_long.csv     one row per (arm, seed, epoch, n, repeat, metric values)
    id_agg.csv      per-arm mean±std of eff_rank and ID at the headline n
    id_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ARMS = ("supervised", "jepa_horizon", "jepa_masked")


def _need_sklearn():
    try:
        from sklearn.neighbors import NearestNeighbors  # noqa: F401
    except Exception:
        raise SystemExit(
            "scikit-learn is required for the nearest-neighbor search.\n"
            "  pip install scikit-learn"
        )
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors


# ------------------------------------------------------------------------------
# Geometry primitives
# ------------------------------------------------------------------------------

def effective_rank(X: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum: (Σλ)² / Σλ².

    A linear, non-saturating effective-rank. X is (n, D)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    C = (Xc.T @ Xc) / max(n - 1, 1)                 # (D, D)
    lam = np.linalg.eigvalsh(C.astype(np.float64))
    lam = np.clip(lam, 0.0, None)
    s = lam.sum()
    if s <= 0:
        return 0.0
    return float(s * s / np.sum(lam * lam))


def _knn_dists(X: np.ndarray, k: int, NearestNeighbors) -> np.ndarray:
    """Return distances to the k nearest neighbors (excluding self), shape (n, k)."""
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute",
                          metric="euclidean", n_jobs=-1).fit(X)
    d, _ = nn.kneighbors(X)                          # (n, k+1); col 0 is self (0)
    return d[:, 1:]                                  # drop self


def twonn(X: np.ndarray, NearestNeighbors, discard_frac: float = 0.10) -> float:
    """TwoNN intrinsic-dimension estimator (Facco et al. 2017).

    For each point: mu = r2 / r1  (ratio of 2nd- to 1st-NN distance). The mu_i
    are Pareto(d); d is the slope of  -log(1 - F(mu))  vs  log(mu)  through the
    origin, discarding the top `discard_frac` of the empirical CDF for robustness.
    """
    d = _knn_dists(X, k=2, NearestNeighbors=NearestNeighbors)
    r1, r2 = d[:, 0], d[:, 1]
    keep = r1 > 0
    mu = r2[keep] / r1[keep]
    mu = mu[mu > 1.0 + 1e-12]                        # mu must be >= 1
    if mu.size < 100:
        return float("nan")
    mu = np.sort(mu)
    m = mu.size
    F = np.arange(1, m + 1) / m
    cut = max(int((1.0 - discard_frac) * m), 10)     # avoid F -> 1 (log blow-up)
    x = np.log(mu[:cut])
    y = -np.log(1.0 - F[:cut])
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)              # slope through origin


def mle_id(X: np.ndarray, NearestNeighbors, ks=(5, 10, 20)) -> Dict[int, float]:
    """Levina–Bickel MLE intrinsic dimension, for each k in `ks` (cross-check)."""
    kmax = max(ks)
    d = _knn_dists(X, k=kmax, NearestNeighbors=NearestNeighbors)   # (n, kmax)
    out: Dict[int, float] = {}
    for k in ks:
        rk = d[:, k - 1][:, None]                    # k-th neighbor distance
        rj = d[:, : k - 1]                           # neighbors 1..k-1
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log(rk / rj)
        s = np.sum(logs, axis=1)
        valid = np.isfinite(s) & (s > 0)
        d_inv = s[valid] / (k - 1)
        out[k] = float(1.0 / np.mean(d_inv)) if d_inv.size else float("nan")
    return out


def whiten(X: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """PCA-whiten with an eigenvalue floor so near-null directions do not blow up.

    eps floors each eigenvalue at eps * lambda_max before inverting the sqrt.
    NOTE: whitening amplifies noise in the smallest directions — read the
    whitened ID as a secondary stress-test, not a primary number."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    n = Xc.shape[0]
    C = (Xc.T @ Xc) / max(n - 1, 1)
    lam, V = np.linalg.eigh(C.astype(np.float64))
    lam = np.clip(lam, 0.0, None)
    floor = eps * lam.max() if lam.max() > 0 else 1.0
    scale = 1.0 / np.sqrt(lam + floor)
    return (Xc @ (V * scale)).astype(np.float32)


# ------------------------------------------------------------------------------
# Dump loading
# ------------------------------------------------------------------------------

_EP_RE = re.compile(r"_ep(\d+)\.npz$")


def find_epochs(readout_dir: Path, arm: str, seed: int) -> List[int]:
    eps = []
    for p in (readout_dir / "readouts").glob(f"{arm}_seed{seed}_ep*.npz"):
        m = _EP_RE.search(p.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def load_pool(readout_dir: Path, arm: str, seed: int, epoch: int,
              pooling: str, split: str) -> np.ndarray:
    f = readout_dir / "readouts" / f"{arm}_seed{seed}_ep{epoch:03d}.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    with np.load(f) as z:
        key = f"{pooling}_{split}"
        if key not in z:
            raise KeyError(f"{f.name}: key {key!r} not found (has {list(z.keys())})")
        return z[key].astype(np.float32)


def subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if X.shape[0] <= n:
        return X
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx]


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def resolve_epochs(all_eps: List[int], spec: str) -> List[int]:
    if not all_eps:
        return []
    if spec == "all":
        return all_eps
    if spec == "last":
        return [all_eps[-1]]
    want = {int(x) for x in spec.split(",") if x.strip()}
    return [e for e in all_eps if e in want]


def mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser(
        description="Effective rank vs TwoNN intrinsic dimension on frozen readouts")
    ap.add_argument("--readout_dir", required=True,
                    help="STAGE-1 out_dir (contains readouts/, subsample.npz, ...)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", default="last", help="'last', 'all', or comma list")
    ap.add_argument("--pooling", default="last_concat512",
                    choices=["last_concat512", "tmean_concat512"])
    ap.add_argument("--split", default="train", choices=["train", "val"],
                    help="pool to sample from; train is larger and matches the "
                         "ladder's PCA-fit set")
    ap.add_argument("--n_sub", type=int, default=30_000,
                    help="subsample size for each ID estimate")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent subsamples per (arm,seed) for stability bars")
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--discard_frac", type=float, default=0.10,
                    help="TwoNN CDF tail fraction discarded for robustness")
    # optional extras (off by default — keep the first taste clean)
    ap.add_argument("--sweep", action="store_true",
                    help="also estimate ID at n in {10k,20k,n_sub} to detect "
                         "high-ID saturation")
    ap.add_argument("--whiten", action="store_true",
                    help="also estimate ID on variance-equalized (whitened) "
                         "readouts — the direct bridge to the accessibility "
                         "scissor (secondary, noise-sensitive)")
    ap.add_argument("--whiten_eps", type=float, default=1e-2)
    ap.add_argument("--mle", action="store_true",
                    help="also compute Levina-Bickel MLE (k=5,10,20) cross-check")
    args = ap.parse_args()

    NearestNeighbors = _need_sklearn()

    readout_dir = Path(args.readout_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # n-grid for the sweep: distinct, sorted, capped at n_sub
    if args.sweep:
        n_grid = sorted({n for n in (10_000, 20_000, args.n_sub) if n <= args.n_sub})
    else:
        n_grid = [args.n_sub]
    headline_n = max(n_grid)

    rows: List[dict] = []
    print(f"[cfg] arms={arms} seeds={seeds} epochs={args.epochs} "
          f"pooling={args.pooling} split={args.split}")
    print(f"[cfg] n_grid={n_grid} repeats={args.repeats} "
          f"whiten={args.whiten} mle={args.mle}\n")

    for arm in arms:
        for sd in seeds:
            all_eps = find_epochs(readout_dir, arm, sd)
            eps = resolve_epochs(all_eps, args.epochs)
            if not eps:
                print(f"  [skip] {arm} seed{sd}: no matching dumps"); continue
            for ep in eps:
                try:
                    pool = load_pool(readout_dir, arm, sd, ep, args.pooling, args.split)
                except (FileNotFoundError, KeyError) as e:
                    print(f"  [skip] {arm} seed{sd} ep{ep}: {e}"); continue

                er = effective_rank(pool)              # once per (arm,seed,epoch)
                base = f"{arm} seed{sd} ep{ep:03d}"
                print(f"  {base}  pool={pool.shape}  eff_rank={er:6.2f}")

                for n in n_grid:
                    for r in range(args.repeats):
                        rng = np.random.default_rng(
                            args.subsample_seed + 1000 * r + n)
                        Xs = subsample(pool, n, rng)
                        t0 = time.time()
                        idv = twonn(Xs, NearestNeighbors, args.discard_frac)
                        rec = {"arm": arm, "seed": sd, "epoch": ep, "n": Xs.shape[0],
                               "repeat": r, "eff_rank": er, "id_twonn": idv}
                        if args.whiten:
                            rec["id_twonn_white"] = twonn(
                                whiten(Xs, args.whiten_eps),
                                NearestNeighbors, args.discard_frac)
                        if args.mle:
                            for k, v in mle_id(Xs, NearestNeighbors).items():
                                rec[f"id_mle_k{k}"] = v
                        rows.append(rec)
                        extra = ""
                        if args.whiten:
                            extra += f"  id_white={rec['id_twonn_white']:5.2f}"
                        print(f"      n={Xs.shape[0]:>6} r{r}  "
                              f"id_twonn={idv:5.2f}{extra}  ({time.time()-t0:.1f}s)")

    if not rows:
        raise SystemExit("No results produced — check --readout_dir / dump names.")

    # ---- long CSV -----------------------------------------------------------
    fields = sorted({k for row in rows for k in row})
    lead = ["arm", "seed", "epoch", "n", "repeat", "eff_rank", "id_twonn"]
    fields = lead + [f for f in fields if f not in lead]
    with open(out / "id_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # ---- aggregate at the headline n ---------------------------------------
    # eff_rank: mean/std over seeds (one value per seed). ID: over seed×repeat.
    er_by_arm: Dict[str, List[float]] = defaultdict(list)
    id_by_arm: Dict[str, List[float]] = defaultdict(list)
    idw_by_arm: Dict[str, List[float]] = defaultdict(list)
    seen_er = set()
    for row in rows:
        if row["n"] != headline_n:
            continue
        a = row["arm"]
        key = (a, row["seed"], row["epoch"])
        if key not in seen_er:
            er_by_arm[a].append(row["eff_rank"]); seen_er.add(key)
        id_by_arm[a].append(row["id_twonn"])
        if "id_twonn_white" in row:
            idw_by_arm[a].append(row["id_twonn_white"])

    agg_rows = []
    print(f"\n=== summary @ n={headline_n} "
          f"(mean ± std; ID over seed×repeat, eff_rank over seeds) ===")
    hdr = f"  {'arm':<14}{'eff_rank':>16}{'id_twonn':>16}{'gap(PR-ID)':>14}"
    if idw_by_arm:
        hdr += f"{'id_white':>14}"
    print(hdr)
    for a in arms:
        if a not in id_by_arm:
            continue
        er_m, er_s = mean_std(er_by_arm[a])
        id_m, id_s = mean_std(id_by_arm[a])
        gap = er_m - id_m
        line = (f"  {a:<14}{er_m:8.2f}±{er_s:5.2f}{id_m:8.2f}±{id_s:5.2f}"
                f"{gap:14.2f}")
        rec = {"arm": a, "n": headline_n,
               "eff_rank_mean": er_m, "eff_rank_std": er_s,
               "id_twonn_mean": id_m, "id_twonn_std": id_s, "gap_pr_minus_id": gap}
        if idw_by_arm:
            idw_m, idw_s = mean_std(idw_by_arm[a])
            line += f"{idw_m:8.2f}±{idw_s:5.2f}"
            rec["id_white_mean"] = idw_m; rec["id_white_std"] = idw_s
        print(line)
        agg_rows.append(rec)

    with open(out / "id_agg.csv", "w", newline="") as f:
        af = sorted({k for row in agg_rows for k in row})
        lead2 = ["arm", "n", "eff_rank_mean", "eff_rank_std",
                 "id_twonn_mean", "id_twonn_std", "gap_pr_minus_id"]
        af = lead2 + [c for c in af if c not in lead2]
        w = csv.DictWriter(f, fieldnames=af)
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)

    with open(out / "id_manifest.json", "w") as f:
        json.dump({"config": vars(args), "n_grid": n_grid,
                   "headline_n": headline_n, "n_records": len(rows)}, f, indent=2)

    print(f"\nWrote: {out/'id_long.csv'}, {out/'id_agg.csv'}, {out/'id_manifest.json'}")
    if args.sweep:
        print("\n[sweep] compare id_twonn across n in id_long.csv: rising with n "
              "=> high true ID (not saturated); flat => estimate is trustworthy.")


if __name__ == "__main__":
    main()

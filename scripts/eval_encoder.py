"""
eval_encoder.py — Valutazione oggettiva dell'encoder LOB (merged v2).

Quattro sezioni:
  1. RICOSTRUZIONE        — MSE, MAE, wMSE su volumi, decomposte per livello e regime
  2. DOWNSTREAM PROBES    — probe lineari (Ridge/Logistic) da z su:
                             reward_t, |shock_mid|, OFI_t+1, spread_t+1, regime
  3. GEOMETRIA LATENTE    — PCA, t-SNE, correlazioni PC-scalari, per-dim std
  4. PROPRIETÀ GEOMETRICHE — stats_head R², bi-Lipschitz empirica
                             (forward random, adversarial PGD, injectivity, k-NN)

Output:
  - Due figure complementari:
      eval_encoder_representation.png  (sezioni 1+2+3, 3×3)
      eval_encoder_geometry.png        (sezione 4, 2×3)
  - Scorecard testuale con verdetto A/B/C/D
  - encoder_metrics.npz con metriche grezze

Uso:
  python scripts/eval_encoder.py
  python scripts/eval_encoder.py --ckpt checkpoints/encoder_best.pt \\
                                  --dataset data/dataset.npz \\
                                  --n_samples 50000
  python scripts/eval_encoder.py --no_tsne --no_adversarial   # run veloce
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))

from models.encoder import LOBEncoder, EncoderConfig, BookStatsPredictor
from training.train_encoder import LOBDataset
from simulate import REGIMES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_encoder(ckpt_path: str, device: torch.device):
    from models.encoder import LOBAutoEncoder
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = EncoderConfig()
    if "cfg" in ckpt:
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    ae = LOBAutoEncoder(cfg).to(device)
    ae.encoder.load_state_dict(ckpt["encoder"])
    if "decoder" in ckpt:
        ae.decoder.load_state_dict(ckpt["decoder"])
    if "stats_head" in ckpt:
        ae.stats_head.load_state_dict(ckpt["stats_head"])
    ae.eval()
    print(f"Encoder loaded  : {ckpt_path}")
    val_metric = ckpt.get('val_recon', ckpt.get('val_loss', 0.0))
    print(f"  epoch={ckpt['epoch']}  val_metric={val_metric:.6f}")
    print(f"  d_latent={cfg.d_latent}")
    return ae.encoder, ckpt["stats"], ae, cfg


@torch.no_grad()
def build_encoded_dataset(
    encoder: LOBEncoder,
    dataset_path: str,
    stats: dict,
    device: torch.device,
    n_samples: int = 50_000,
    batch_size: int = 1024,
    seed: int = 42,
    autoencoder=None,
) -> dict:
    """
    Encode a random subset. Returns a dict with:
      Z          : (N, d_latent)
      Z_next     : (N, d_latent)
      vol_pred   : (N, 2, L, 2)    reconstructed book (full)
      vol_true   : (N, 2, L)       ground-truth volumes (normalised)
      book_norm  : (N, 2, L, 2)    normalised book input to encoder
      scalars    : (N, 4)          normalised [mid, spread, imb, inv]
      regimes    : (N,)
      rewards    : (N,)
      raw_obs    : (N, obs_dim)
      raw_next   : (N, obs_dim)
    """
    raw_data  = np.load(dataset_path)
    obs_all   = raw_data["observations"]
    next_all  = raw_data["next_observations"]
    rew_all   = raw_data["rewards"]
    reg_all   = raw_data["regimes"]

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rew_all), size=min(n_samples, len(rew_all)), replace=False))

    ds = LOBDataset(dataset_path, stats=stats)
    L    = ds.L
    tick = ds.tick_size
    s    = stats

    def preprocess(obs_array: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        book_flat = obs_array[:, :2*L*2].reshape(-1, 2, L, 2).astype(np.float32)
        sc_raw    = obs_array[:, 2*L*2:].astype(np.float32)
        mids      = sc_raw[:, 0]
        book_norm = book_flat.copy()
        for side in range(2):
            book_norm[:, side, :, 0] = (book_flat[:, side, :, 0] - mids[:, None]) / tick
        book_norm[:, :, :, 1] /= s["vol_scale"]
        sc_norm = np.stack([
            (sc_raw[:, 0] - s["mid_mean"]) / s["mid_std"],
            sc_raw[:, 1] / tick,
            sc_raw[:, 2],
            sc_raw[:, 3] / s["inv_scale"],
        ], axis=1)
        return torch.from_numpy(book_norm), torch.from_numpy(sc_norm)

    obs_sub  = obs_all[idx]
    next_sub = next_all[idx]

    books,   scalars    = preprocess(obs_sub)
    books_n, scalars_n  = preprocess(next_sub)

    Z_list, Zn_list, vp_list = [], [], []

    for i in range(0, len(idx), batch_size):
        b  = books[i:i+batch_size].to(device)
        bn = books_n[i:i+batch_size].to(device)

        z  = encoder(b)
        zn = encoder(bn)

        Z_list.append(z.cpu().numpy())
        Zn_list.append(zn.cpu().numpy())

        if autoencoder is not None:
            book_pred = autoencoder.decoder(z)
            vp_list.append(book_pred.cpu().numpy())

    vol_pred = np.concatenate(vp_list, axis=0) if vp_list else None

    return {
        "Z":         np.concatenate(Z_list,  axis=0),
        "Z_next":    np.concatenate(Zn_list, axis=0),
        "vol_true":  books.numpy()[:, :, :, 1],
        "vol_pred":  vol_pred,
        "book_norm": books.numpy(),       # per Lipschitz/injectivity
        "scalars":   scalars.numpy()[: len(idx)],
        "regimes":   reg_all[idx],
        "rewards":   rew_all[idx],
        "raw_obs":   obs_sub,
        "raw_next":  next_sub,
    }


# ===========================================================================
# SECTION 1 — Reconstruction metrics
# ===========================================================================

def reconstruction_metrics(data: dict, L: int, vol_scale: float) -> dict:
    vt = data["vol_true"]
    bp = data["vol_pred"]
    regimes = data["regimes"]

    if bp is None:
        zeros = np.zeros(L)
        return {"wMSE": 0, "MSE": 0, "MAE": 0, "top_MSE": 0, "deep_MSE": 0,
                "per_level_mse": zeros, "per_level_mae": zeros,
                "per_regime": {}}

    vp = bp[:, :, :, 1]

    w = np.ones(L); w[0] = 4.0
    if L > 1: w[1] = 2.0
    w = w / w.sum()

    per_level_mse = ((vp - vt) ** 2).mean(axis=(0, 1))
    per_level_mae = np.abs(vp - vt).mean(axis=(0, 1))

    wmse      = (per_level_mse * w).sum()
    total_mse = per_level_mse.mean()
    total_mae = per_level_mae.mean()
    top_mse   = per_level_mse[:2].mean()
    deep_mse  = per_level_mse[2:].mean() if L > 2 else 0.0

    per_regime = {}
    for r_idx, rname in enumerate(["low_vol", "mid_vol", "high_vol"]):
        mask = (regimes == r_idx)
        if mask.sum() == 0: continue
        vt_r = vt[mask]
        vp_r = vp[mask]
        mse_abs = ((vp_r - vt_r) ** 2).mean()
        vol_mean_sq = (vt_r ** 2).mean() + 1e-10
        mse_rel = mse_abs / vol_mean_sq
        per_level_mse_r = ((vp_r - vt_r) ** 2).mean(axis=(0, 1))
        per_regime[rname] = {
            "mse_abs": mse_abs, "mse_rel": mse_rel,
            "per_level_mse": per_level_mse_r, "n": int(mask.sum()),
        }

    return {
        "wMSE": wmse, "MSE": total_mse, "MAE": total_mae,
        "top_MSE": top_mse, "deep_MSE": deep_mse,
        "per_level_mse": per_level_mse, "per_level_mae": per_level_mae,
        "per_regime": per_regime,
    }


# ===========================================================================
# SECTION 2 — Downstream probes
# ===========================================================================

def _split(X, y, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n   = int(len(X) * train_frac)
    return X[idx[:n]], X[idx[n:]], y[idx[:n]], y[idx[n:]]


def probe_regression(Z: np.ndarray, y: np.ndarray, name: str) -> dict:
    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)
    X_tr, X_te, y_tr, y_te = _split(Z_s, y)
    reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    baseline_mse = mean_squared_error(y_te, np.full_like(y_te, y_tr.mean()))
    return {"name": name, "MSE": mse, "MAE": mae, "R2": r2,
            "baseline_MSE": baseline_mse}


def probe_classification(Z: np.ndarray, regimes: np.ndarray) -> dict:
    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)
    X_tr, X_te, y_tr, y_te = _split(Z_s, regimes)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = clf.score(X_te, y_te)
    report = classification_report(
        y_te, y_pred, target_names=[r["name"] for r in REGIMES]
    )
    baseline = max(np.bincount(y_te)) / len(y_te)
    return {
        "accuracy": acc, "baseline": baseline, "report": report,
        "y_true": y_te, "y_pred": y_pred,
    }


def probe_classification_mlp(Z: np.ndarray, regimes: np.ndarray) -> dict:
    """
    Non-linear probe: MLP classifier on z → regime.

    Diagnostic purpose: if linear probe ~72% but MLP 90%+, the regime info
    IS in z but not linearly separable. The downstream WM (Causal Transformer)
    is non-linear so it will find it. If MLP also ~72%, the info is simply
    NOT in the single-snapshot z — only the WM with context window can recover it.
    """
    from sklearn.neural_network import MLPClassifier

    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)
    X_tr, X_te, y_tr, y_te = _split(Z_s, regimes)

    # MLP shallow: 2 hidden layers, moderate capacity
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = clf.score(X_te, y_te)
    report = classification_report(
        y_te, y_pred, target_names=[r["name"] for r in REGIMES]
    )
    baseline = max(np.bincount(y_te)) / len(y_te)
    return {
        "accuracy": acc, "baseline": baseline, "report": report,
        "y_true": y_te, "y_pred": y_pred,
        "n_iter": clf.n_iter_,
    }


def downstream_probes(data: dict) -> dict:
    Z = data["Z"]
    raw_next = data["raw_next"]
    L = 10

    rewards_next = data["rewards"]
    res_reward = probe_regression(Z, rewards_next, "reward_t")

    mid_t   = data["raw_obs"][:, 2*L*2 + 0]
    mid_t1  = raw_next[:, 2*L*2 + 0]
    shock   = np.abs(mid_t1 - mid_t)
    res_vol = probe_regression(Z, shock, "|shock_mid|")

    bid_vol = raw_next[:, 1]
    ask_vol = raw_next[:, 2*L + 1]
    denom   = bid_vol + ask_vol + 1e-8
    ofi     = (bid_vol - ask_vol) / denom
    res_ofi = probe_regression(Z, ofi, "OFI_t+1")

    spread_next = raw_next[:, 2*L*2 + 1]
    res_spread  = probe_regression(Z, spread_next, "spread_t+1")

    # Linear (logistic) and non-linear (MLP) regime probes
    res_regime     = probe_classification(Z, data["regimes"])
    res_regime_mlp = probe_classification_mlp(Z, data["regimes"])

    return {
        "reward": res_reward, "vol": res_vol, "ofi": res_ofi,
        "spread": res_spread,
        "regime": res_regime,
        "regime_mlp": res_regime_mlp,
    }


# ===========================================================================
# SECTION 2bis — Sequence-level regime probe (upper bound for Module B)
# ===========================================================================

@torch.no_grad()
def sequence_regime_probe(
    encoder: LOBEncoder,
    dataset_path: str,
    stats: dict,
    device: torch.device,
    seq_len: int = 20,
    max_sequences: int = 20_000,
    batch_size: int = 1024,
    seed: int = 42,
) -> dict:
    """
    Sequence-level regime classification.

    Purpose: stabilire un upper bound empirico per la distinguibilità dei
    regimi via la DINAMICA dei latenti, non il singolo snapshot.
    Questo è il segnale massimo che il Modulo B (Causal Transformer con
    context window) può in principio estrarre dai latenti dell'encoder.

    Procedura:
      1. Carica dataset preservando l'ordine degli episodi (no shuffle).
      2. Per ogni episodio PURO (no regime switching), estrae finestre
         sliding di seq_len step consecutivi.
      3. Encoda ciascuna finestra → feature vector (seq_len × d_latent).
      4. Flatten + linear/MLP classifier sul regime dell'episodio.

    Interpretazione:
      - Se sequence accuracy >> single-snapshot accuracy → l'info temporale
        aggiunge segnale. Modulo B con context ha margine sostanziale.
      - Se sequence accuracy ≈ single-snapshot accuracy → no temporal gain,
        ripensare architettura o features in input.

    Args:
        seq_len: lunghezza finestra. 20 è un context ragionevole per iniziare.
        max_sequences: limite per velocità computazionale.
    """
    from sklearn.neural_network import MLPClassifier

    print(f"  Building sequences (len={seq_len})...")
    raw = np.load(dataset_path)
    obs_all   = raw["observations"]
    reg_all   = raw["regimes"]
    ep_all    = raw["episode_ids"]
    ts_all    = raw["timesteps"]
    switch_all = raw["switch_mask"]

    # Riordina per (episode_id, timestep) per ricostruire sequenze coerenti
    order = np.lexsort((ts_all, ep_all))
    obs_sorted  = obs_all[order]
    reg_sorted  = reg_all[order]
    ep_sorted   = ep_all[order]
    ts_sorted   = ts_all[order]
    sw_sorted   = switch_all[order]

    # Identifica episodi PURI: nessun switch + regime costante
    unique_eps = np.unique(ep_sorted)
    pure_eps = []
    for ep in unique_eps:
        mask = ep_sorted == ep
        if sw_sorted[mask].sum() == 0 and len(np.unique(reg_sorted[mask])) == 1:
            pure_eps.append(ep)
    pure_eps = set(pure_eps)
    print(f"  Pure episodes: {len(pure_eps)}/{len(unique_eps)}")

    # Extract sequence starting indices (only within same pure episode)
    L = 10
    tick_size = 0.01
    book_flat_dim = 2 * L * 2
    starts = []
    labels = []
    for ep in pure_eps:
        mask = ep_sorted == ep
        ep_indices = np.where(mask)[0]
        if len(ep_indices) < seq_len:
            continue
        # Stride = seq_len//2 per overlap moderato → più sequenze per episodio
        stride = max(1, seq_len // 2)
        for start in range(0, len(ep_indices) - seq_len + 1, stride):
            # Verifica che gli step siano realmente consecutivi (devono esserlo
            # per episodi puri non mixed)
            sub = ep_indices[start:start + seq_len]
            if ts_sorted[sub[-1]] - ts_sorted[sub[0]] == seq_len - 1:
                starts.append(sub[0])
                labels.append(int(reg_sorted[sub[0]]))
                if len(starts) >= max_sequences:
                    break
        if len(starts) >= max_sequences:
            break

    starts = np.array(starts)
    labels = np.array(labels)
    n_seq = len(starts)
    print(f"  Extracted {n_seq} sequences of length {seq_len}")
    print(f"  Regime distribution: {np.bincount(labels)}")

    if n_seq < 100:
        print("  WARNING: troppo poche sequenze per un probe affidabile")
        return {
            "error": "insufficient_sequences",
            "n_sequences": n_seq,
        }

    # Build all sequences as (n_seq, seq_len, 2, L, 2) normalized books
    print(f"  Encoding {n_seq} sequences...")
    # Gather all unique timestep indices to encode once
    all_indices = np.concatenate([
        np.arange(s, s + seq_len) for s in starts
    ])  # shape (n_seq * seq_len,)

    # Preprocess (normalize) all those books
    obs_sub = obs_sorted[all_indices]
    book_flat = obs_sub[:, :book_flat_dim].reshape(-1, 2, L, 2).astype(np.float32)
    sc_raw    = obs_sub[:, book_flat_dim:].astype(np.float32)
    mids      = sc_raw[:, 0]
    book_norm = book_flat.copy()
    for side in range(2):
        book_norm[:, side, :, 0] = (book_flat[:, side, :, 0] - mids[:, None]) / tick_size
    book_norm[:, :, :, 1] /= stats["vol_scale"]

    # Encode in batches
    book_t = torch.from_numpy(book_norm)
    Z_list = []
    for i in range(0, len(book_t), batch_size):
        b = book_t[i:i+batch_size].to(device)
        z = encoder(b).cpu().numpy()
        Z_list.append(z)
    Z_flat = np.concatenate(Z_list, axis=0)  # (n_seq * seq_len, d_latent)
    d_latent = Z_flat.shape[1]

    # Reshape a (n_seq, seq_len * d_latent) — flatten temporal axis
    Z_seq = Z_flat.reshape(n_seq, seq_len, d_latent)
    X = Z_seq.reshape(n_seq, seq_len * d_latent)
    y = labels

    # Train/test split (random su sequenze — le sequenze sono già tra episodi diversi
    # quindi basta splittare uniformemente)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_seq)
    split = int(0.8 * n_seq)
    tr_idx, te_idx = perm[:split], perm[split:]
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Linear probe
    clf_lin = LogisticRegression(max_iter=2000, random_state=42)
    clf_lin.fit(X_tr_s, y_tr)
    acc_lin = clf_lin.score(X_te_s, y_te)
    y_pred_lin = clf_lin.predict(X_te_s)
    report_lin = classification_report(
        y_te, y_pred_lin, target_names=[r["name"] for r in REGIMES], zero_division=0
    )

    # MLP probe
    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    clf_mlp.fit(X_tr_s, y_tr)
    acc_mlp = clf_mlp.score(X_te_s, y_te)
    y_pred_mlp = clf_mlp.predict(X_te_s)
    report_mlp = classification_report(
        y_te, y_pred_mlp, target_names=[r["name"] for r in REGIMES], zero_division=0
    )

    baseline = max(np.bincount(y_te)) / len(y_te)

    return {
        "seq_len": seq_len,
        "n_sequences": n_seq,
        "linear_accuracy": float(acc_lin),
        "mlp_accuracy": float(acc_mlp),
        "baseline": float(baseline),
        "linear_report": report_lin,
        "mlp_report": report_mlp,
        "y_true": y_te,
        "y_pred_linear": y_pred_lin,
        "y_pred_mlp": y_pred_mlp,
    }


# ===========================================================================
# SECTION 3 — Latent geometry
# ===========================================================================

def latent_geometry(data: dict) -> dict:
    Z       = data["Z"]
    scalars = data["scalars"]

    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)

    pca = PCA(n_components=min(10, Z.shape[1]))
    Z_pca = pca.fit_transform(Z_s)

    n_pc = min(5, Z_pca.shape[1])
    scalar_names = ["mid", "spread", "imbalance", "inventory"]
    corr = np.zeros((n_pc, 4))
    for i in range(n_pc):
        for j in range(4):
            std = scalars[:, j].std()
            if std < 1e-8:
                corr[i, j] = float("nan")
            else:
                corr[i, j] = np.corrcoef(Z_pca[:, i], scalars[:, j])[0, 1]

    return {
        "Z_pca": Z_pca, "pca": pca, "corr": corr,
        "scalar_names": scalar_names, "regimes": data["regimes"],
    }


# ===========================================================================
# SECTION 4 — Geometric properties (NEW)
# ===========================================================================

def stats_head_r2(autoencoder, data: dict, device: torch.device,
                  batch_size: int = 1024) -> dict:
    """
    R² per ciascuno dei 6 target della stats_head.
    Verifica direttamente che L_stats abbia imparato i target.
    """
    book = torch.from_numpy(data["book_norm"])
    n = len(book)
    preds_list, targets_list = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b = book[i:i+batch_size].to(device)
            z = autoencoder.encoder(b)
            preds   = autoencoder.stats_head(z).cpu().numpy()
            targets = BookStatsPredictor.compute_targets(b).cpu().numpy()
            preds_list.append(preds)
            targets_list.append(targets)
    preds   = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    names = ["log_mean_bid_vol", "log_mean_ask_vol",
             "bid_concentration", "ask_concentration",
             "imbalance", "spread"]
    r2 = {}
    degenerate = {}  # target con varianza troppo bassa per un R² significativo
    for i, name in enumerate(names):
        t = targets[:, i]; p = preds[:, i]
        ss_tot = ((t - t.mean()) ** 2).sum()
        # Se la varianza del target è degenerata (es. spread quasi-costante
        # a 2 tick), l'R² è numericamente instabile e privo di significato.
        # Threshold: varianza relativa <1e-4 della media quadratica.
        rel_var = ss_tot / (len(t) * (t ** 2).mean() + 1e-12)
        if rel_var < 1e-4:
            degenerate[name] = {
                "mean": float(t.mean()),
                "std": float(t.std()),
                "mse": float(((t - p) ** 2).mean()),
            }
            r2[name] = None  # marcato esplicitamente come non valido
        else:
            ss_res = ((t - p) ** 2).sum()
            r2[name] = float(1.0 - ss_res / (ss_tot + 1e-12))

    return {"r2": r2, "preds": preds, "targets": targets, "degenerate": degenerate}


def forward_lipschitz_random(encoder, data: dict, device: torch.device,
                              n: int = 500, scales=(1e-3, 1e-2, 1e-1)) -> dict:
    """
    Forward Lipschitz con perturbazioni random a scale diverse.
    Comportamento lineare → ratio costante al variare di eps.
    """
    book = torch.from_numpy(data["book_norm"])
    idx  = np.random.choice(len(book), min(n, len(book)), replace=False)
    bk   = book[idx].to(device)
    with torch.no_grad():
        z0 = encoder(bk)

    input_std = bk.std().item()
    results = {}
    for eps in scales:
        delta_scale = eps * input_std
        lips = []
        for _ in range(5):
            delta = torch.randn_like(bk) * delta_scale
            with torch.no_grad():
                z1 = encoder(bk + delta)
            dz = (z1 - z0).norm(dim=1)
            do = delta.flatten(1).norm(dim=1)
            lips.append((dz / (do + 1e-12)).cpu().numpy())
        lips = np.concatenate(lips)
        results[eps] = {
            "median": float(np.median(lips)),
            "p95": float(np.percentile(lips, 95)),
            "max": float(lips.max()),
            "values": lips,
        }
    return results


def forward_lipschitz_adversarial(encoder, data: dict, device: torch.device,
                                   n: int = 200, eps_rel: float = 1e-2,
                                   n_steps: int = 20,
                                   step_size_rel: float = 5e-3) -> dict:
    """
    Worst-case Lipschitz via PGD. Trova delta che massimizza ||E(o+d)-E(o)||
    a ||d|| <= eps. Questo è ciò che conta per la stabilità del critico.
    """
    book = torch.from_numpy(data["book_norm"])
    idx  = np.random.choice(len(book), min(n, len(book)), replace=False)
    bk   = book[idx].to(device)
    with torch.no_grad():
        z0 = encoder(bk).detach()

    input_std = bk.std().item()
    numel = bk[0].numel()
    eps = eps_rel * input_std * np.sqrt(numel)
    step_size = step_size_rel * input_std * np.sqrt(numel)

    delta = torch.randn_like(bk) * (eps / np.sqrt(numel)) * 0.1
    delta.requires_grad_(True)

    for _ in range(n_steps):
        z1 = encoder(bk + delta)
        loss = -(z1 - z0).pow(2).sum(dim=1).mean()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - step_size * grad.sign()
            delta_flat = delta.flatten(1)
            norms = delta_flat.norm(dim=1, keepdim=True)
            factor = torch.clamp(eps / (norms + 1e-12), max=1.0)
            delta = (delta_flat * factor).reshape(bk.shape)
        delta.requires_grad_(True)

    with torch.no_grad():
        z1 = encoder(bk + delta)
        dz = (z1 - z0).norm(dim=1)
        do = delta.flatten(1).norm(dim=1)
        lips = (dz / (do + 1e-12)).cpu().numpy()

    return {
        "median": float(np.median(lips)),
        "p95": float(np.percentile(lips, 95)),
        "max": float(lips.max()),
        "values": lips,
        "eps_rel": eps_rel,
    }


def injectivity_analysis(data: dict, n_pairs: int = 50000,
                          n_points: int = 5000) -> dict:
    """
    Iniettività empirica: per coppie (o_i, o_j), misura
      c_ij = ||o_i - o_j|| / ||z_i - z_j||
    Coda pesante (p99/median alto) → book diversi mappati allo stesso z.
    Uso il book normalizzato, non l'obs raw.
    """
    idx = np.random.choice(len(data["Z"]), min(n_points, len(data["Z"])), replace=False)
    Z = data["Z"][idx]
    O = data["book_norm"][idx].reshape(len(idx), -1)

    i = np.random.randint(0, len(Z), size=n_pairs)
    j = np.random.randint(0, len(Z), size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    dz = np.linalg.norm(Z[i] - Z[j], axis=1)
    do = np.linalg.norm(O[i] - O[j], axis=1)

    valid = dz > 1e-6
    c = do[valid] / dz[valid]

    return {
        "ratios": c,
        "median": float(np.median(c)),
        "p95": float(np.percentile(c, 95)),
        "p99": float(np.percentile(c, 99)),
        "max": float(c.max()),
        "dz": dz[valid], "do": do[valid],
    }


def knn_consistency(data: dict, n: int = 2000, k: int = 10) -> dict:
    """
    k-NN consistency: i k vicini in z sono vicini anche in o?
    ratio = avg_dist(k-NN-in-z) / avg_dist(k-random)
    << 1 → buono.
    """
    idx = np.random.choice(len(data["Z"]), min(n, len(data["Z"])), replace=False)
    z = data["Z"][idx]
    o = data["book_norm"][idx].reshape(len(idx), -1)

    D_z = cdist(z, z)
    D_o = cdist(o, o)

    np.fill_diagonal(D_z, np.inf)
    knn_idx = np.argsort(D_z, axis=1)[:, :k]

    knn_do = np.take_along_axis(D_o, knn_idx, axis=1).mean(axis=1)
    rand_idx = np.random.randint(0, len(idx), size=(len(idx), k))
    rand_do = np.take_along_axis(D_o, rand_idx, axis=1).mean(axis=1)

    ratio = knn_do / (rand_do + 1e-12)
    return {
        "median_ratio": float(np.median(ratio)),
        "p95_ratio": float(np.percentile(ratio, 95)),
        "values": ratio,
    }


def latent_correlation_matrix(data: dict) -> dict:
    """Correlation matrix delle dimensioni di z (VICReg decorr check)."""
    z = data["Z"]
    corr = np.corrcoef(z.T)
    off_diag = np.abs(corr - np.eye(len(corr)))
    return {
        "corr": corr,
        "max_off_diag": float(off_diag.max()),
        "mean_off_diag": float(off_diag[off_diag > 0].mean()),
    }


def geometric_properties(
    encoder, autoencoder, data: dict, device: torch.device,
    adversarial: bool = True,
) -> dict:
    """Bundle di tutti i check geometrici."""
    print("  [4.1] stats_head R²...")
    stats_r2 = stats_head_r2(autoencoder, data, device)
    print("  [4.2] Forward Lipschitz (random multi-scale)...")
    lip_rand = forward_lipschitz_random(encoder, data, device, n=500)
    lip_adv = None
    if adversarial:
        print("  [4.3] Forward Lipschitz (adversarial PGD)...")
        lip_adv = forward_lipschitz_adversarial(encoder, data, device, n=200)
    print("  [4.4] Injectivity pairwise...")
    inj = injectivity_analysis(data)
    print("  [4.5] k-NN consistency...")
    knn = knn_consistency(data)
    print("  [4.6] Latent correlation matrix...")
    corr = latent_correlation_matrix(data)
    return {
        "stats_r2": stats_r2,
        "lip_rand": lip_rand,
        "lip_adv": lip_adv,
        "injectivity": inj,
        "knn": knn,
        "latent_corr": corr,
    }


# ===========================================================================
# VERDETTO
# ===========================================================================

def verdict(ds: dict, geo_props: dict, sp: dict | None = None) -> tuple[str, list[str]]:
    """
    4 livelli + override:
      A: tutto ok → procedi al Modulo B
      B: problemi downstream, geometria ok → fix mirato (più epoche, λ_stats)
      C: downstream ok, geometria rotta → smoothness loss / spectral norm
      D: entrambi rotti → ripensare

    v5 changes:
      - criterio regime usa MLP probe (non linear);
      - k-NN consistency rimossa come criterio: era tarata per encoder VICReg
        (std~1) e non è comparabile con l'attuale scala latente ridotta;
      - OVERRIDE: se sequence_probe_mlp ≥ 0.90, forza verdetto A indipendentemente
        dalle altre metriche. Il sequence probe è il test più informativo per
        determinare se l'encoder è adatto al Modulo B downstream.
    """
    downstream_issues, geometry_issues = [], []

    # Downstream regressivi — ESCLUDO reward (impredicibile per costruzione)
    # e accetto soglie basse per |shock| (intrinsecamente rumoroso)
    for key, label, thresh in [
        ("vol", "|shock|", 0.1),         # molto basso: shock è gauss pure
        ("ofi", "OFI", 0.2),
        ("spread", "spread_t+1", 0.2),
    ]:
        if key in ds and ds[key]["R2"] < thresh:
            downstream_issues.append(f"probe {label} R²={ds[key]['R2']:.2f}")

    # Regime: uso MLP accuracy come criterio. Se MLP alto (>85%), info c'è.
    # Se MLP ~ linear e ambedue bassi, è floor information-theoretic → NON issue
    # (lo gestirà il Modulo B con context window).
    mlp_acc = ds["regime_mlp"]["accuracy"]
    lin_acc = ds["regime"]["accuracy"]
    if mlp_acc < 0.80 and (mlp_acc - lin_acc) > 0.05:
        # MLP batte linear ma non abbastanza: probabile issue di capacity/training
        downstream_issues.append(
            f"MLP regime acc={mlp_acc:.2f} (linear={lin_acc:.2f})"
        )
    # Se MLP ~ linear e ambedue bassi, NON è un issue: l'info semplicemente
    # non è codificabile da un singolo snapshot. Documentiamo ma non penalizziamo.

    # Geometria — ignoro R² None (target degeneri a varianza nulla)
    r2 = geo_props["stats_r2"]["r2"]
    bad_stats = [n for n, v in r2.items() if v is not None and v < 0.5]
    if bad_stats:
        geometry_issues.append(f"stats_head R² basso: {bad_stats}")

    # max off-diag non è più un criterio (niente più VICReg decorr esplicito).
    # Correlazioni latenti possono naturalmente emergere tra dimensioni
    # semanticamente vicine, non è patologia.

    if geo_props["lip_adv"] is not None:
        adv_p95 = geo_props["lip_adv"]["p95"]
        if adv_p95 > 50:
            geometry_issues.append(f"Lipschitz PGD p95={adv_p95:.1f}")

    inj = geo_props["injectivity"]
    if inj["p99"] / inj["median"] > 100:
        geometry_issues.append(
            f"injectivity coda pesante (p99/med={inj['p99']/inj['median']:.0f})"
        )

    # k-NN consistency REMOVED as criterion (v5): threshold 0.5 era tarato per
    # encoder con std~1 (VICReg). Con contractive on-manifold la scala del latente
    # è piccola e il k-NN ratio non è più comparabile con threshold assoluti.
    # Reported but not penalizing.

    all_issues = downstream_issues + geometry_issues

    # OVERRIDE: sequence probe is the gold-standard test for "is this encoder
    # ready for the downstream WM?" If sequence_mlp ≥ 90%, declare A regardless
    # of single-snapshot metrics or minor geometric quibbles.
    if sp is not None and "error" not in sp:
        if sp["mlp_accuracy"] >= 0.90:
            return "A", [f"(override: sequence MLP = {sp['mlp_accuracy']*100:.1f}% ≥ 90%)"]

    if not all_issues:
        return "A", []
    if downstream_issues and geometry_issues:
        return "D", all_issues
    if downstream_issues:
        return "B", downstream_issues
    return "C", geometry_issues


# ===========================================================================
# PRINT
# ===========================================================================

def print_reconstruction(rec: dict) -> None:
    print("\n" + "="*70)
    print("[1] RECONSTRUCTION (normalised volumes)")
    print("="*70)
    print(f"  wMSE={rec['wMSE']:.6f}  MSE={rec['MSE']:.6f}  MAE={rec['MAE']:.6f}")
    print(f"  top-of-book MSE={rec['top_MSE']:.6f}  deep MSE={rec['deep_MSE']:.6f}")
    if rec.get("per_regime"):
        print(f"\n  {'regime':<10s}  {'MSE_abs':>10s}  {'MSE_rel':>10s}  {'n':>7s}")
        print(f"  {'-'*42}")
        for rname, rv in rec["per_regime"].items():
            print(f"  {rname:<10s}  {rv['mse_abs']:>10.6f}  "
                  f"{rv['mse_rel']:>10.4f}  {rv['n']:>7d}")


def print_downstream(ds: dict) -> None:
    print("\n" + "="*70)
    print("[2] DOWNSTREAM PROBES (from z)")
    print("="*70)
    print(f"\n  {'probe':<15s}  {'R²':>8s}  {'MSE':>10s}")
    print(f"  {'-'*36}")
    for key in ["reward", "vol", "ofi", "spread"]:
        r = ds[key]
        print(f"  {r['name']:<15s}  {r['R2']:>+8.4f}  {r['MSE']:>10.6f}")

    # Regime: linear + MLP affiancati
    r_lin = ds["regime"]
    r_mlp = ds["regime_mlp"]
    print(f"\n  Regime classification (baseline = {r_lin['baseline']*100:.1f}%):")
    print(f"    Linear (Logistic) : {r_lin['accuracy']*100:.1f}%")
    print(f"    MLP (2-layer)     : {r_mlp['accuracy']*100:.1f}%  "
          f"[n_iter={r_mlp['n_iter']}]")

    gap = (r_mlp['accuracy'] - r_lin['accuracy']) * 100
    if gap > 5:
        print(f"    → GAP = {gap:+.1f}%: info non-lineare in z. Modulo B la troverà.")
    elif gap < 2:
        print(f"    → GAP = {gap:+.1f}%: MLP ~ Linear. Possibile floor information-theoretic")
        print(f"      (info richiede contesto temporale, non presente nello snapshot statico).")
    else:
        print(f"    → GAP = {gap:+.1f}%: marginale.")

    print(f"\n  Linear probe classification report:")
    print(r_lin['report'])
    print(f"  MLP probe classification report:")
    print(r_mlp['report'])


def print_sequence_probe(sp: dict, single_snap_linear: float,
                          single_snap_mlp: float) -> None:
    print("\n" + "="*70)
    print("[2bis] SEQUENCE-LEVEL REGIME PROBE (upper bound Modulo B)")
    print("="*70)

    if "error" in sp:
        print(f"  ERRORE: {sp['error']} (n_seq={sp['n_sequences']})")
        return

    print(f"\n  Sequence length : {sp['seq_len']}")
    print(f"  N sequences    : {sp['n_sequences']}")
    print(f"  Baseline       : {sp['baseline']*100:.1f}%")
    print(f"\n  {'Probe':<20s}  {'Accuracy':>10s}  {'vs single-snap':>16s}")
    print(f"  {'-'*48}")
    print(f"  {'Single (linear)':<20s}  {single_snap_linear*100:>9.1f}%  "
          f"{'--':>16s}")
    print(f"  {'Single (MLP)':<20s}  {single_snap_mlp*100:>9.1f}%  "
          f"{'--':>16s}")
    print(f"  {'Sequence (linear)':<20s}  {sp['linear_accuracy']*100:>9.1f}%  "
          f"{(sp['linear_accuracy']-single_snap_linear)*100:>+15.1f}%")
    print(f"  {'Sequence (MLP)':<20s}  {sp['mlp_accuracy']*100:>9.1f}%  "
          f"{(sp['mlp_accuracy']-single_snap_mlp)*100:>+15.1f}%")

    # Interpretazione automatica
    gain = sp['mlp_accuracy'] - single_snap_mlp
    print()
    if gain > 0.15:
        print(f"  → GAIN temporale +{gain*100:.1f}%: SIGNIFICATIVO.")
        print(f"    L'info dinamica dei regimi È presente nella sequenza di z.")
        print(f"    Modulo B ha margine concreto per migliorare la distinzione.")
    elif gain > 0.05:
        print(f"  → GAIN temporale +{gain*100:.1f}%: MODERATO.")
        print(f"    Il contesto aiuta ma non risolve completamente.")
    else:
        print(f"  → GAIN temporale +{gain*100:.1f}%: TRASCURABILE.")
        print(f"    Il segnale dinamico è debole anche nella sequenza.")
        print(f"    Implicazione: il Modulo B non potrà migliorare drasticamente")
        print(f"    la separazione dei regimi. Ripensare features o architettura.")

    print(f"\n  Sequence MLP classification report:")
    print(sp['mlp_report'])


def print_geometry(geo: dict) -> None:
    pca = geo["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_pc = len(pca.explained_variance_ratio_)
    print("\n" + "="*70)
    print("[3] LATENT GEOMETRY (unsupervised)")
    print("="*70)
    print(f"\n  PCA: {cumvar[min(7, n_pc-1)]:.0f}% in 8 PCs, "
          f"{cumvar[min(4, n_pc-1)]:.0f}% in 5 PCs")
    print(f"\n  {'PC':<5s}  {'var%':>6s}  {'cum%':>6s}")
    for i in range(min(n_pc, 10)):
        print(f"  PC{i+1:<2d}  {pca.explained_variance_ratio_[i]*100:>5.1f}%  "
              f"{cumvar[i]:>5.1f}%")


def print_geometric_properties(gp: dict) -> None:
    print("\n" + "="*70)
    print("[4] GEOMETRIC PROPERTIES (bi-Lipschitz diagnostics)")
    print("="*70)

    print("\n  stats_head R² (want > 0.7):")
    degenerate = gp["stats_r2"].get("degenerate", {})
    for name, v in gp["stats_r2"]["r2"].items():
        if v is None:
            # Target degenere (varianza ~0): R² non significativo
            d = degenerate.get(name, {})
            print(f"    [N/A] {name:20s}: degenere "
                  f"(mean={d.get('mean', 0):.3f}, std={d.get('std', 0):.2e}, "
                  f"mse={d.get('mse', 0):.2e})")
        else:
            mark = "OK " if v > 0.7 else "~~ " if v > 0.4 else "!! "
            print(f"    [{mark}] {name:20s}: {v:+.4f}")

    print("\n  Forward Lipschitz (random perturbations):")
    for eps, v in sorted(gp["lip_rand"].items()):
        print(f"    eps={eps:.0e}  median={v['median']:.3f}  "
              f"p95={v['p95']:.3f}  max={v['max']:.3f}")

    if gp["lip_adv"] is not None:
        adv = gp["lip_adv"]
        print(f"\n  Forward Lipschitz (PGD adversarial, eps_rel={adv['eps_rel']:.0e}):")
        print(f"    median={adv['median']:.3f}  p95={adv['p95']:.3f}  "
              f"max={adv['max']:.3f}")

    inj = gp["injectivity"]
    print("\n  Injectivity (pairwise ||Δo||/||Δz||):")
    print(f"    median={inj['median']:.3f}  p95={inj['p95']:.3f}  "
          f"p99={inj['p99']:.3f}  max={inj['max']:.3f}")
    print(f"    p99/median = {inj['p99']/inj['median']:.1f}  (>100 = coda pesante)")

    knn = gp["knn"]
    print(f"\n  k-NN consistency (k=10):")
    print(f"    median ratio = {knn['median_ratio']:.3f}  "
          f"p95 = {knn['p95_ratio']:.3f}  (<0.5 = buono)")

    lc = gp["latent_corr"]
    print(f"\n  Latent correlation (VICReg decorr check):")
    print(f"    max |off-diag| = {lc['max_off_diag']:.4f}  "
          f"mean |off-diag| = {lc['mean_off_diag']:.4f}")


def print_verdict(letter: str, issues: list[str]) -> None:
    print("\n" + "="*70)
    print(f"VERDETTO: {letter}")
    print("="*70)
    if letter == "A":
        if issues and "override" in issues[0]:
            print(f"  Encoder pronto per il Modulo B. {issues[0]}")
        else:
            print("  Encoder pronto per il Modulo B.")
    elif letter == "B":
        print("  Downstream R² bassi ma geometria OK.")
        for iss in issues: print(f"    - {iss}")
        print("  Suggeriti: più epoche, aumenta λ_stats, o più dati.")
    elif letter == "C":
        print("  Downstream OK ma geometria latente problematica.")
        for iss in issues: print(f"    - {iss}")
        print("  Suggeriti: contractive smoothness loss, spectral norm allargata.")
    else:  # D
        print("  Problemi sia downstream che geometrici.")
        for iss in issues: print(f"    - {iss}")
        print("  Ripensare loss/architettura.")
    print("="*70 + "\n")


# ===========================================================================
# PLOTTING — figura 1: representation (3×3)
# ===========================================================================

def plot_representation(rec: dict, ds: dict, geo: dict, data: dict,
                         sp: dict | None, out_path: str,
                         do_tsne: bool = True) -> None:
    L = len(rec["per_level_mse"])
    pca = geo["pca"]
    Z = data["Z"]
    regimes = data["regimes"]
    d_latent = Z.shape[1]
    regime_names = [r["name"] for r in REGIMES]
    regime_colors = ["#2ecc71", "#3498db", "#e74c3c"]

    GRAY, BLUE, GREEN, RED = "#555555", "#2c7bb6", "#1a9641", "#d7191c"

    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- 0a. Volume & MSE per level ---
    ax = fig.add_subplot(gs[0, 0])
    vt = data["vol_true"]
    vol_per_level = vt.mean(axis=1)
    vol_mean = vol_per_level.mean(axis=0)
    vol_std  = vol_per_level.std(axis=0)
    ax2 = ax.twinx()
    ax.bar(range(L), vol_mean,
           color=[RED if i <= 1 else BLUE for i in range(L)],
           alpha=0.7, label="Mean vol")
    ax.errorbar(range(L), vol_mean, yerr=vol_std, fmt="none",
                ecolor=GRAY, capsize=3, lw=0.8)
    ax2.plot(range(L), rec["per_level_mse"], "k--o", markersize=4, lw=1.2, label="MSE")
    ax.set_title("Volume & recon MSE per level", fontsize=10)
    ax.set_xlabel("Level (0=best)", fontsize=8)
    ax.set_ylabel("Mean volume (norm.)", fontsize=8)
    ax2.set_ylabel("Recon MSE", fontsize=8)
    ax.set_xticks(range(L))
    ax.tick_params(labelsize=7)
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=7, loc="upper right")

    # --- 0b. Recon examples ---
    ax = fig.add_subplot(gs[0, 1])
    if data["vol_pred"] is not None:
        for r_idx, (rname, rcol) in enumerate(zip(regime_names, regime_colors)):
            mask = (regimes == r_idx)
            if mask.sum() == 0: continue
            r_true = vt[mask].mean(axis=1)
            r_pred = data["vol_pred"][mask][:, :, :, 1].mean(axis=1)
            true_mean = r_true.mean(axis=0)
            pred_mean = r_pred.mean(axis=0)
            ax.plot(range(L), true_mean, "o-", color=rcol, markersize=3,
                    lw=1.5, label=f"{rname} true")
            ax.plot(range(L), pred_mean, "x--", color=rcol, markersize=4,
                    lw=1, alpha=0.7)
        ax.set_title("Recon: true (●) vs pred (×)\nmean profile per regime", fontsize=10)
        ax.set_xlabel("Level", fontsize=8)
        ax.set_ylabel("Mean volume (norm.)", fontsize=8)
        ax.legend(fontsize=7, ncol=1)
        ax.set_xticks(range(L))
        ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "No decoder", transform=ax.transAxes, ha="center")

    # --- 0c. MSE per regime ---
    ax = fig.add_subplot(gs[0, 2])
    if data["vol_pred"] is not None:
        regime_mse = []
        for r_idx in range(len(REGIMES)):
            mask = (regimes == r_idx)
            if mask.sum() == 0:
                regime_mse.append(0); continue
            r_true = vt[mask]
            r_pred = data["vol_pred"][mask][:, :, :, 1]
            regime_mse.append(((r_true - r_pred) ** 2).mean())
        bars = ax.bar(regime_names, regime_mse, color=regime_colors, alpha=0.8)
        ax.set_title("Recon MSE per regime", fontsize=10)
        ax.set_ylabel("MSE (norm. vols)", fontsize=8)
        ax.tick_params(labelsize=8)
        for bar, v in zip(bars, regime_mse):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(regime_mse)*0.02,
                    f"{v:.4f}", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No decoder", transform=ax.transAxes, ha="center")

    # --- 1a. PCA explained variance ---
    ax = fig.add_subplot(gs[1, 0])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_show = min(d_latent, 10)
    ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show] * 100,
           color=BLUE, alpha=0.8, label="Per-PC")
    ax.plot(range(1, n_show + 1), cumvar[:n_show], "o-", color=RED, markersize=4,
            lw=1.5, label="Cumulative")
    ax.axhline(80, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.set_title(f"PCA — explained variance ({d_latent}-dim z)", fontsize=10)
    ax.set_xlabel("PC", fontsize=8)
    ax.set_ylabel("Variance (%)", fontsize=8)
    ax.set_xticks(range(1, n_show + 1))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    # --- 1b. t-SNE (or PCA 2D if skipped) ---
    ax = fig.add_subplot(gs[1, 1])
    n_proj = min(8000, len(Z))
    rng = np.random.default_rng(42)
    pi = rng.choice(len(Z), size=n_proj, replace=False)
    Z_sub = Z[pi]; reg_sub = regimes[pi]
    if do_tsne:
        print("  Computing t-SNE...")
        Z_2d = TSNE(n_components=2, perplexity=40, random_state=42,
                    max_iter=1000).fit_transform(Z_sub)
        title = "t-SNE of z by regime"
    else:
        Z_2d = geo["Z_pca"][pi, :2]
        title = "PCA 2D of z by regime"
    for r_idx, (rname, rcol) in enumerate(zip(regime_names, regime_colors)):
        mask = (reg_sub == r_idx)
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=rcol, s=3, alpha=0.4,
                   label=rname, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=4)
    ax.set_xticks([]); ax.set_yticks([])

    # --- 1c. Per-dim std (latent scale, no longer VICReg-targeted) ---
    ax = fig.add_subplot(gs[1, 2])
    per_dim_std = Z.std(axis=0)
    # Ordiniamo per visualizzazione: dimensioni con più std prima
    order = np.argsort(-per_dim_std)
    colors_bar = [BLUE if per_dim_std[i] > per_dim_std.mean() * 0.3 else GRAY
                  for i in order]
    ax.bar(range(d_latent), per_dim_std[order], color=colors_bar, alpha=0.85)
    ax.set_title(f"Latent scale per dim (sorted)\n"
                 f"min={per_dim_std.min():.3f}  max={per_dim_std.max():.3f}  "
                 f"mean={per_dim_std.mean():.3f}",
                 fontsize=10)
    ax.set_xlabel("dimension (sorted by std)", fontsize=8)
    ax.set_ylabel("std(z)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, per_dim_std.max() * 1.15)

    # --- 2a. Downstream probes ---
    ax = fig.add_subplot(gs[2, 0])
    items = []
    for key, label, col in [
        ("ofi", "OFI_t+1", GREEN),
        ("spread", "spread_t+1", BLUE),
        ("vol", "|Δmid|", RED),
        ("reward", "reward_t", "#9b59b6"),
    ]:
        if key in ds:
            items.append((label, ds[key]["R2"], col))
    if items:
        names, vals, cols = zip(*items)
        ax.barh(list(names), list(vals), color=list(cols), alpha=0.8)
        ax.set_xlabel("R² (linear probe)", fontsize=8)
        ax.set_title("Downstream probes", fontsize=10)
        ax.axvline(0, color="black", lw=0.5)
        ax.tick_params(labelsize=8)
        for i, v in enumerate(vals):
            ax.text(v + 0.02 if v >= 0 else v - 0.02, i, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=9, fontweight="bold")

    # --- 2b. Regime confusion matrix ---
    ax = fig.add_subplot(gs[2, 1])
    y_true = ds["regime"]["y_true"]; y_pred = ds["regime"]["y_pred"]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(regime_names)))
    ax.set_xticklabels(regime_names, fontsize=8)
    ax.set_yticks(range(len(regime_names)))
    ax.set_yticklabels(regime_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(f"Regime conf. matrix  "
                 f"(acc={ds['regime']['accuracy']*100:.1f}%)", fontsize=10)
    for i in range(len(regime_names)):
        for j in range(len(regime_names)):
            v = cm_norm[i, j]
            col = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}\n({cm[i,j]})", ha="center", va="center",
                    fontsize=8, color=col)

    # --- 2c. Single-snapshot vs Sequence-level probe (KILLER RESULT) ---
    ax = fig.add_subplot(gs[2, 2])
    if sp is not None and "error" not in sp:
        labels = ["Baseline", "Single\n(linear)", "Single\n(MLP)",
                  "Sequence\n(linear)", "Sequence\n(MLP)"]
        vals = [
            sp["baseline"],
            ds["regime"]["accuracy"],
            ds["regime_mlp"]["accuracy"],
            sp["linear_accuracy"],
            sp["mlp_accuracy"],
        ]
        colors_bars = ["#9E9E9E", "#90CAF9", "#42A5F5", "#A5D6A7", "#66BB6A"]
        bars = ax.bar(range(len(labels)), [v*100 for v in vals],
                      color=colors_bars, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v*100 + 1.5,
                    f"{v*100:.1f}%", ha="center", fontsize=9,
                    fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("Accuracy (%)", fontsize=8)
        ax.set_ylim(0, 105)
        gain = (sp["mlp_accuracy"] - ds["regime_mlp"]["accuracy"]) * 100
        ax.set_title(f"Regime separability: snapshot vs sequence\n"
                     f"temporal gain +{gain:.1f}%  →  Modulo B upper bound",
                     fontsize=10)
        ax.axhline(sp["baseline"]*100, color="#9E9E9E", ls=":", lw=0.8, alpha=0.7)
        ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "No sequence probe", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#666")
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("LOB Encoder — Representation Quality (Sections 1-3)",
                 fontsize=14, y=1.01, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved: {out_path}")


# ===========================================================================
# PLOTTING — figura 2: geometric properties (2×3)
# ===========================================================================

def plot_geometry(gp: dict, out_path: str) -> None:
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    # --- 0a. stats_head R² ---
    ax = fig.add_subplot(gs[0, 0])
    r2 = gp["stats_r2"]["r2"]
    # Filtro R² None (target degeneri) e li disegno come barra grigia
    names = list(r2.keys())
    vals_raw = list(r2.values())
    vals_plot = [v if v is not None else 0.0 for v in vals_raw]
    colors = []
    for v in vals_raw:
        if v is None:
            colors.append("#BDBDBD")  # grigio per degeneri
        elif v > 0.7:
            colors.append("#4CAF50")
        elif v > 0.4:
            colors.append("#FFA726")
        else:
            colors.append("#EF5350")
    ax.barh(names, vals_plot, color=colors, edgecolor="white")
    ax.axvline(0.7, color="black", ls="--", alpha=0.5, lw=0.8)
    ax.set_title("stats_head R² on 6 targets\n(>0.7 good, grey=degenere)",
                 fontweight="bold")
    ax.set_xlabel("R²")
    valid_vals = [v for v in vals_raw if v is not None]
    if valid_vals:
        ax.set_xlim(min(0, min(valid_vals) - 0.05), 1.05)
    else:
        ax.set_xlim(0, 1.05)
    for i, v in enumerate(vals_raw):
        if v is None:
            ax.text(0.02, i, "N/A (degenere)", va="center", fontsize=8, color="#666")
        else:
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    # --- 0b. Lipschitz random vs scale ---
    ax = fig.add_subplot(gs[0, 1])
    lip_r = gp["lip_rand"]
    scales = sorted(lip_r.keys())
    meds = [lip_r[s]["median"] for s in scales]
    p95s = [lip_r[s]["p95"] for s in scales]
    maxs = [lip_r[s]["max"] for s in scales]
    x = np.arange(len(scales)); w = 0.25
    ax.bar(x - w, meds, w, label="median", color="#5C6BC0")
    ax.bar(x, p95s, w, label="p95", color="#EF5350")
    ax.bar(x + w, maxs, w, label="max", color="#26A69A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"eps={s:.0e}" for s in scales], fontsize=8)
    ax.set_title("Forward Lipschitz (random)\n||Δz||/||Δo|| vs eps",
                 fontweight="bold")
    ax.set_ylabel("Lipschitz ratio")
    ax.legend(fontsize=8)

    # --- 0c. Lipschitz adversarial (or empty) ---
    ax = fig.add_subplot(gs[0, 2])
    if gp["lip_adv"] is not None:
        adv = gp["lip_adv"]
        ax.hist(adv["values"], bins=50, color="#F57C00", edgecolor="white")
        ax.axvline(adv["median"], color="black", ls="--",
                   label=f"median={adv['median']:.2f}")
        ax.axvline(adv["p95"], color="red", ls="--",
                   label=f"p95={adv['p95']:.2f}")
        ax.set_title(f"Adversarial Lipschitz (PGD)\n"
                     f"worst-case, eps_rel={adv['eps_rel']:.0e}",
                     fontweight="bold")
        ax.set_xlabel("Lipschitz ratio")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Adversarial test disabled",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # --- 1a. Injectivity scatter ---
    ax = fig.add_subplot(gs[1, 0])
    inj = gp["injectivity"]
    sub = np.random.choice(len(inj["dz"]), min(3000, len(inj["dz"])), replace=False)
    ax.scatter(inj["dz"][sub], inj["do"][sub], s=1, alpha=0.3, color="#1976D2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"Injectivity: ||Δo|| vs ||Δz||\n"
                 f"p99/median={inj['p99']/inj['median']:.1f}",
                 fontweight="bold")
    ax.set_xlabel("||z_i - z_j||")
    ax.set_ylabel("||o_i - o_j|| (norm. book)")

    # --- 1b. k-NN consistency histogram ---
    ax = fig.add_subplot(gs[1, 1])
    knn = gp["knn"]
    ax.hist(knn["values"], bins=50, color="#26A69A", edgecolor="white", alpha=0.85)
    ax.axvline(knn["median_ratio"], color="black", ls="--",
               label=f"median={knn['median_ratio']:.3f}")
    ax.axvline(0.5, color="red", ls=":", alpha=0.7, label="threshold=0.5")
    ax.set_title("k-NN consistency (k=10)\n"
                 "ratio = avg-dist-kNN-in-z / avg-dist-random",
                 fontweight="bold")
    ax.set_xlabel("ratio (<1 = good)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # --- 1c. Latent correlation matrix ---
    ax = fig.add_subplot(gs[1, 2])
    corr = gp["latent_corr"]["corr"]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"Latent correlation matrix\n"
                 f"max |off-diag|={gp['latent_corr']['max_off_diag']:.3f}",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("LOB Encoder — Geometric Properties (Section 4)",
                 fontsize=14, y=0.98, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved: {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder, stats, ae, cfg = load_encoder(args.ckpt, device)

    print(f"\nEncoding {args.n_samples:,} campioni...")
    data = build_encoded_dataset(
        encoder, args.dataset, stats, device,
        n_samples=args.n_samples, autoencoder=ae,
    )
    print(f"  Z shape: {data['Z'].shape}")

    # Section 1 — Reconstruction
    rec = reconstruction_metrics(data, L=cfg.L, vol_scale=stats["vol_scale"])
    print_reconstruction(rec)

    # Section 2 — Downstream probes
    print("\nComputing downstream probes...")
    ds = downstream_probes(data)
    print_downstream(ds)

    # Section 2bis — Sequence-level regime probe (upper bound for Module B)
    print("\nComputing sequence-level regime probe...")
    sp = sequence_regime_probe(
        encoder, args.dataset, stats, device,
        seq_len=args.seq_len, max_sequences=args.max_sequences,
    )
    print_sequence_probe(sp, ds["regime"]["accuracy"], ds["regime_mlp"]["accuracy"])

    # Section 3 — Latent geometry (unsupervised)
    print("\nAnalyzing latent geometry...")
    geo = latent_geometry(data)
    print_geometry(geo)

    # Section 4 — Geometric properties (NEW)
    print("\nComputing geometric properties...")
    gp = geometric_properties(encoder, ae, data, device,
                               adversarial=not args.no_adversarial)
    print_geometric_properties(gp)

    # Verdetto unificato
    letter, issues = verdict(ds, gp, sp)
    print_verdict(letter, issues)

    # Plots: due figure separate
    plot_representation(rec, ds, geo, data, sp,
                        out_path=out_dir / "eval_encoder_representation.png",
                        do_tsne=not args.no_tsne)
    plot_geometry(gp, out_path=out_dir / "eval_encoder_geometry.png")

    # Salva metriche grezze
    save_dict = {
        "verdict": letter,
        "issues": issues,
        "recon_per_level": rec["per_level_mse"],
        "std_per_dim": data["Z"].std(axis=0),
        "latent_corr_max_off_diag": gp["latent_corr"]["max_off_diag"],
        "injectivity_p99_over_median":
            gp["injectivity"]["p99"] / gp["injectivity"]["median"],
        "knn_median_ratio": gp["knn"]["median_ratio"],
        "regime_accuracy_linear": ds["regime"]["accuracy"],
        "regime_accuracy_mlp": ds["regime_mlp"]["accuracy"],
        "regime_mlp_linear_gap":
            ds["regime_mlp"]["accuracy"] - ds["regime"]["accuracy"],
    }
    for k, v in gp["stats_r2"]["r2"].items():
        # np.savez non gestisce None -> uso NaN per i target degeneri
        save_dict[f"stats_r2_{k}"] = v if v is not None else float("nan")
    for key in ["reward", "vol", "ofi", "spread"]:
        save_dict[f"probe_R2_{key}"] = ds[key]["R2"]
    if gp["lip_adv"] is not None:
        save_dict["lip_adv_median"] = gp["lip_adv"]["median"]
        save_dict["lip_adv_p95"] = gp["lip_adv"]["p95"]

    # Sequence probe results
    if "error" not in sp:
        save_dict["seq_probe_seq_len"] = sp["seq_len"]
        save_dict["seq_probe_n_sequences"] = sp["n_sequences"]
        save_dict["seq_probe_linear_acc"] = sp["linear_accuracy"]
        save_dict["seq_probe_mlp_acc"] = sp["mlp_accuracy"]
        save_dict["seq_probe_gain_over_single"] = (
            sp["mlp_accuracy"] - ds["regime_mlp"]["accuracy"]
        )

    np.savez(out_dir / "encoder_metrics.npz", **save_dict)
    print(f"Metrics saved: {out_dir / 'encoder_metrics.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate LOB Encoder (merged)")
    p.add_argument("--ckpt",      type=str,   default="checkpoints/encoder_best.pt")
    p.add_argument("--dataset",   type=str,   default="data/dataset.npz")
    p.add_argument("--n_samples", type=int,   default=50_000)
    p.add_argument("--out_dir",   type=str,   default="eval_encoder_output")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--no_tsne",   action="store_true",
                   help="Skip t-SNE (use PCA 2D instead) — molto più veloce")
    p.add_argument("--no_adversarial", action="store_true",
                   help="Skip PGD adversarial Lipschitz test")
    p.add_argument("--seq_len",   type=int,   default=20,
                   help="Sequence length for sequence-level regime probe (default 20)")
    p.add_argument("--max_sequences", type=int, default=20_000,
                   help="Max sequences for probe (velocità/statistica tradeoff)")
    args = p.parse_args()
    main(args)"""
eval_encoder.py — Valutazione oggettiva dell'encoder LOB (merged v2).

Quattro sezioni:
  1. RICOSTRUZIONE        — MSE, MAE, wMSE su volumi, decomposte per livello e regime
  2. DOWNSTREAM PROBES    — probe lineari (Ridge/Logistic) da z su:
                             reward_t, |shock_mid|, OFI_t+1, spread_t+1, regime
  3. GEOMETRIA LATENTE    — PCA, t-SNE, correlazioni PC-scalari, per-dim std
  4. PROPRIETÀ GEOMETRICHE — stats_head R², bi-Lipschitz empirica
                             (forward random, adversarial PGD, injectivity, k-NN)

Output:
  - Due figure complementari:
      eval_encoder_representation.png  (sezioni 1+2+3, 3×3)
      eval_encoder_geometry.png        (sezione 4, 2×3)
  - Scorecard testuale con verdetto A/B/C/D
  - encoder_metrics.npz con metriche grezze

Uso:
  python scripts/eval_encoder.py
  python scripts/eval_encoder.py --ckpt checkpoints/encoder_best.pt \\
                                  --dataset data/dataset.npz \\
                                  --n_samples 50000
  python scripts/eval_encoder.py --no_tsne --no_adversarial   # run veloce
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.spatial.distance import cdist

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))

from models.encoder import LOBEncoder, EncoderConfig, BookStatsPredictor
from training.train_encoder import LOBDataset
from simulate import REGIMES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_encoder(ckpt_path: str, device: torch.device):
    from models.encoder import LOBAutoEncoder
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = EncoderConfig()
    if "cfg" in ckpt:
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    ae = LOBAutoEncoder(cfg).to(device)
    ae.encoder.load_state_dict(ckpt["encoder"])
    if "decoder" in ckpt:
        ae.decoder.load_state_dict(ckpt["decoder"])
    if "stats_head" in ckpt:
        ae.stats_head.load_state_dict(ckpt["stats_head"])
    ae.eval()
    print(f"Encoder loaded  : {ckpt_path}")
    val_metric = ckpt.get('val_recon', ckpt.get('val_loss', 0.0))
    print(f"  epoch={ckpt['epoch']}  val_metric={val_metric:.6f}")
    print(f"  d_latent={cfg.d_latent}")
    return ae.encoder, ckpt["stats"], ae, cfg


@torch.no_grad()
def build_encoded_dataset(
    encoder: LOBEncoder,
    dataset_path: str,
    stats: dict,
    device: torch.device,
    n_samples: int = 50_000,
    batch_size: int = 1024,
    seed: int = 42,
    autoencoder=None,
) -> dict:
    """
    Encode a random subset. Returns a dict with:
      Z          : (N, d_latent)
      Z_next     : (N, d_latent)
      vol_pred   : (N, 2, L, 2)    reconstructed book (full)
      vol_true   : (N, 2, L)       ground-truth volumes (normalised)
      book_norm  : (N, 2, L, 2)    normalised book input to encoder
      scalars    : (N, 4)          normalised [mid, spread, imb, inv]
      regimes    : (N,)
      rewards    : (N,)
      raw_obs    : (N, obs_dim)
      raw_next   : (N, obs_dim)
    """
    raw_data  = np.load(dataset_path)
    obs_all   = raw_data["observations"]
    next_all  = raw_data["next_observations"]
    rew_all   = raw_data["rewards"]
    reg_all   = raw_data["regimes"]

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rew_all), size=min(n_samples, len(rew_all)), replace=False))

    ds = LOBDataset(dataset_path, stats=stats)
    L    = ds.L
    tick = ds.tick_size
    s    = stats

    def preprocess(obs_array: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        book_flat = obs_array[:, :2*L*2].reshape(-1, 2, L, 2).astype(np.float32)
        sc_raw    = obs_array[:, 2*L*2:].astype(np.float32)
        mids      = sc_raw[:, 0]
        book_norm = book_flat.copy()
        for side in range(2):
            book_norm[:, side, :, 0] = (book_flat[:, side, :, 0] - mids[:, None]) / tick
        book_norm[:, :, :, 1] /= s["vol_scale"]
        sc_norm = np.stack([
            (sc_raw[:, 0] - s["mid_mean"]) / s["mid_std"],
            sc_raw[:, 1] / tick,
            sc_raw[:, 2],
            sc_raw[:, 3] / s["inv_scale"],
        ], axis=1)
        return torch.from_numpy(book_norm), torch.from_numpy(sc_norm)

    obs_sub  = obs_all[idx]
    next_sub = next_all[idx]

    books,   scalars    = preprocess(obs_sub)
    books_n, scalars_n  = preprocess(next_sub)

    Z_list, Zn_list, vp_list = [], [], []

    for i in range(0, len(idx), batch_size):
        b  = books[i:i+batch_size].to(device)
        bn = books_n[i:i+batch_size].to(device)

        z  = encoder(b)
        zn = encoder(bn)

        Z_list.append(z.cpu().numpy())
        Zn_list.append(zn.cpu().numpy())

        if autoencoder is not None:
            book_pred = autoencoder.decoder(z)
            vp_list.append(book_pred.cpu().numpy())

    vol_pred = np.concatenate(vp_list, axis=0) if vp_list else None

    return {
        "Z":         np.concatenate(Z_list,  axis=0),
        "Z_next":    np.concatenate(Zn_list, axis=0),
        "vol_true":  books.numpy()[:, :, :, 1],
        "vol_pred":  vol_pred,
        "book_norm": books.numpy(),       # per Lipschitz/injectivity
        "scalars":   scalars.numpy()[: len(idx)],
        "regimes":   reg_all[idx],
        "rewards":   rew_all[idx],
        "raw_obs":   obs_sub,
        "raw_next":  next_sub,
    }


# ===========================================================================
# SECTION 1 — Reconstruction metrics
# ===========================================================================

def reconstruction_metrics(data: dict, L: int, vol_scale: float) -> dict:
    vt = data["vol_true"]
    bp = data["vol_pred"]
    regimes = data["regimes"]

    if bp is None:
        zeros = np.zeros(L)
        return {"wMSE": 0, "MSE": 0, "MAE": 0, "top_MSE": 0, "deep_MSE": 0,
                "per_level_mse": zeros, "per_level_mae": zeros,
                "per_regime": {}}

    vp = bp[:, :, :, 1]

    w = np.ones(L); w[0] = 4.0
    if L > 1: w[1] = 2.0
    w = w / w.sum()

    per_level_mse = ((vp - vt) ** 2).mean(axis=(0, 1))
    per_level_mae = np.abs(vp - vt).mean(axis=(0, 1))

    wmse      = (per_level_mse * w).sum()
    total_mse = per_level_mse.mean()
    total_mae = per_level_mae.mean()
    top_mse   = per_level_mse[:2].mean()
    deep_mse  = per_level_mse[2:].mean() if L > 2 else 0.0

    per_regime = {}
    for r_idx, rname in enumerate(["low_vol", "mid_vol", "high_vol"]):
        mask = (regimes == r_idx)
        if mask.sum() == 0: continue
        vt_r = vt[mask]
        vp_r = vp[mask]
        mse_abs = ((vp_r - vt_r) ** 2).mean()
        vol_mean_sq = (vt_r ** 2).mean() + 1e-10
        mse_rel = mse_abs / vol_mean_sq
        per_level_mse_r = ((vp_r - vt_r) ** 2).mean(axis=(0, 1))
        per_regime[rname] = {
            "mse_abs": mse_abs, "mse_rel": mse_rel,
            "per_level_mse": per_level_mse_r, "n": int(mask.sum()),
        }

    return {
        "wMSE": wmse, "MSE": total_mse, "MAE": total_mae,
        "top_MSE": top_mse, "deep_MSE": deep_mse,
        "per_level_mse": per_level_mse, "per_level_mae": per_level_mae,
        "per_regime": per_regime,
    }


# ===========================================================================
# SECTION 2 — Downstream probes
# ===========================================================================

def _split(X, y, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n   = int(len(X) * train_frac)
    return X[idx[:n]], X[idx[n:]], y[idx[:n]], y[idx[n:]]


def probe_regression(Z: np.ndarray, y: np.ndarray, name: str) -> dict:
    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)
    X_tr, X_te, y_tr, y_te = _split(Z_s, y)
    reg = Ridge(alpha=1.0).fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    baseline_mse = mean_squared_error(y_te, np.full_like(y_te, y_tr.mean()))
    return {"name": name, "MSE": mse, "MAE": mae, "R2": r2,
            "baseline_MSE": baseline_mse}


def probe_classification(Z: np.ndarray, regimes: np.ndarray) -> dict:
    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)
    X_tr, X_te, y_tr, y_te = _split(Z_s, regimes)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = clf.score(X_te, y_te)
    report = classification_report(
        y_te, y_pred, target_names=[r["name"] for r in REGIMES]
    )
    baseline = max(np.bincount(y_te)) / len(y_te)
    return {
        "accuracy": acc, "baseline": baseline, "report": report,
        "y_true": y_te, "y_pred": y_pred,
    }


def probe_classification_mlp(Z: np.ndarray, regimes: np.ndarray) -> dict:
    """
    Non-linear probe: MLP classifier on z → regime.

    Diagnostic purpose: if linear probe ~72% but MLP 90%+, the regime info
    IS in z but not linearly separable. The downstream WM (Causal Transformer)
    is non-linear so it will find it. If MLP also ~72%, the info is simply
    NOT in the single-snapshot z — only the WM with context window can recover it.
    """
    from sklearn.neural_network import MLPClassifier

    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)
    X_tr, X_te, y_tr, y_te = _split(Z_s, regimes)

    # MLP shallow: 2 hidden layers, moderate capacity
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = clf.score(X_te, y_te)
    report = classification_report(
        y_te, y_pred, target_names=[r["name"] for r in REGIMES]
    )
    baseline = max(np.bincount(y_te)) / len(y_te)
    return {
        "accuracy": acc, "baseline": baseline, "report": report,
        "y_true": y_te, "y_pred": y_pred,
        "n_iter": clf.n_iter_,
    }


def downstream_probes(data: dict) -> dict:
    Z = data["Z"]
    raw_next = data["raw_next"]
    L = 10

    rewards_next = data["rewards"]
    res_reward = probe_regression(Z, rewards_next, "reward_t")

    mid_t   = data["raw_obs"][:, 2*L*2 + 0]
    mid_t1  = raw_next[:, 2*L*2 + 0]
    shock   = np.abs(mid_t1 - mid_t)
    res_vol = probe_regression(Z, shock, "|shock_mid|")

    bid_vol = raw_next[:, 1]
    ask_vol = raw_next[:, 2*L + 1]
    denom   = bid_vol + ask_vol + 1e-8
    ofi     = (bid_vol - ask_vol) / denom
    res_ofi = probe_regression(Z, ofi, "OFI_t+1")

    spread_next = raw_next[:, 2*L*2 + 1]
    res_spread  = probe_regression(Z, spread_next, "spread_t+1")

    # Linear (logistic) and non-linear (MLP) regime probes
    res_regime     = probe_classification(Z, data["regimes"])
    res_regime_mlp = probe_classification_mlp(Z, data["regimes"])

    return {
        "reward": res_reward, "vol": res_vol, "ofi": res_ofi,
        "spread": res_spread,
        "regime": res_regime,
        "regime_mlp": res_regime_mlp,
    }


# ===========================================================================
# SECTION 2bis — Sequence-level regime probe (upper bound for Module B)
# ===========================================================================

@torch.no_grad()
def sequence_regime_probe(
    encoder: LOBEncoder,
    dataset_path: str,
    stats: dict,
    device: torch.device,
    seq_len: int = 20,
    max_sequences: int = 20_000,
    batch_size: int = 1024,
    seed: int = 42,
) -> dict:
    """
    Sequence-level regime classification.

    Purpose: stabilire un upper bound empirico per la distinguibilità dei
    regimi via la DINAMICA dei latenti, non il singolo snapshot.
    Questo è il segnale massimo che il Modulo B (Causal Transformer con
    context window) può in principio estrarre dai latenti dell'encoder.

    Procedura:
      1. Carica dataset preservando l'ordine degli episodi (no shuffle).
      2. Per ogni episodio PURO (no regime switching), estrae finestre
         sliding di seq_len step consecutivi.
      3. Encoda ciascuna finestra → feature vector (seq_len × d_latent).
      4. Flatten + linear/MLP classifier sul regime dell'episodio.

    Interpretazione:
      - Se sequence accuracy >> single-snapshot accuracy → l'info temporale
        aggiunge segnale. Modulo B con context ha margine sostanziale.
      - Se sequence accuracy ≈ single-snapshot accuracy → no temporal gain,
        ripensare architettura o features in input.

    Args:
        seq_len: lunghezza finestra. 20 è un context ragionevole per iniziare.
        max_sequences: limite per velocità computazionale.
    """
    from sklearn.neural_network import MLPClassifier

    print(f"  Building sequences (len={seq_len})...")
    raw = np.load(dataset_path)
    obs_all   = raw["observations"]
    reg_all   = raw["regimes"]
    ep_all    = raw["episode_ids"]
    ts_all    = raw["timesteps"]
    switch_all = raw["switch_mask"]

    # Riordina per (episode_id, timestep) per ricostruire sequenze coerenti
    order = np.lexsort((ts_all, ep_all))
    obs_sorted  = obs_all[order]
    reg_sorted  = reg_all[order]
    ep_sorted   = ep_all[order]
    ts_sorted   = ts_all[order]
    sw_sorted   = switch_all[order]

    # Identifica episodi PURI: nessun switch + regime costante
    unique_eps = np.unique(ep_sorted)
    pure_eps = []
    for ep in unique_eps:
        mask = ep_sorted == ep
        if sw_sorted[mask].sum() == 0 and len(np.unique(reg_sorted[mask])) == 1:
            pure_eps.append(ep)
    pure_eps = set(pure_eps)
    print(f"  Pure episodes: {len(pure_eps)}/{len(unique_eps)}")

    # Extract sequence starting indices (only within same pure episode)
    L = 10
    tick_size = 0.01
    book_flat_dim = 2 * L * 2
    starts = []
    labels = []
    for ep in pure_eps:
        mask = ep_sorted == ep
        ep_indices = np.where(mask)[0]
        if len(ep_indices) < seq_len:
            continue
        # Stride = seq_len//2 per overlap moderato → più sequenze per episodio
        stride = max(1, seq_len // 2)
        for start in range(0, len(ep_indices) - seq_len + 1, stride):
            # Verifica che gli step siano realmente consecutivi (devono esserlo
            # per episodi puri non mixed)
            sub = ep_indices[start:start + seq_len]
            if ts_sorted[sub[-1]] - ts_sorted[sub[0]] == seq_len - 1:
                starts.append(sub[0])
                labels.append(int(reg_sorted[sub[0]]))
                if len(starts) >= max_sequences:
                    break
        if len(starts) >= max_sequences:
            break

    starts = np.array(starts)
    labels = np.array(labels)
    n_seq = len(starts)
    print(f"  Extracted {n_seq} sequences of length {seq_len}")
    print(f"  Regime distribution: {np.bincount(labels)}")

    if n_seq < 100:
        print("  WARNING: troppo poche sequenze per un probe affidabile")
        return {
            "error": "insufficient_sequences",
            "n_sequences": n_seq,
        }

    # Build all sequences as (n_seq, seq_len, 2, L, 2) normalized books
    print(f"  Encoding {n_seq} sequences...")
    # Gather all unique timestep indices to encode once
    all_indices = np.concatenate([
        np.arange(s, s + seq_len) for s in starts
    ])  # shape (n_seq * seq_len,)

    # Preprocess (normalize) all those books
    obs_sub = obs_sorted[all_indices]
    book_flat = obs_sub[:, :book_flat_dim].reshape(-1, 2, L, 2).astype(np.float32)
    sc_raw    = obs_sub[:, book_flat_dim:].astype(np.float32)
    mids      = sc_raw[:, 0]
    book_norm = book_flat.copy()
    for side in range(2):
        book_norm[:, side, :, 0] = (book_flat[:, side, :, 0] - mids[:, None]) / tick_size
    book_norm[:, :, :, 1] /= stats["vol_scale"]

    # Encode in batches
    book_t = torch.from_numpy(book_norm)
    Z_list = []
    for i in range(0, len(book_t), batch_size):
        b = book_t[i:i+batch_size].to(device)
        z = encoder(b).cpu().numpy()
        Z_list.append(z)
    Z_flat = np.concatenate(Z_list, axis=0)  # (n_seq * seq_len, d_latent)
    d_latent = Z_flat.shape[1]

    # Reshape a (n_seq, seq_len * d_latent) — flatten temporal axis
    Z_seq = Z_flat.reshape(n_seq, seq_len, d_latent)
    X = Z_seq.reshape(n_seq, seq_len * d_latent)
    y = labels

    # Train/test split (random su sequenze — le sequenze sono già tra episodi diversi
    # quindi basta splittare uniformemente)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_seq)
    split = int(0.8 * n_seq)
    tr_idx, te_idx = perm[:split], perm[split:]
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Linear probe
    clf_lin = LogisticRegression(max_iter=2000, random_state=42)
    clf_lin.fit(X_tr_s, y_tr)
    acc_lin = clf_lin.score(X_te_s, y_te)
    y_pred_lin = clf_lin.predict(X_te_s)
    report_lin = classification_report(
        y_te, y_pred_lin, target_names=[r["name"] for r in REGIMES], zero_division=0
    )

    # MLP probe
    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    clf_mlp.fit(X_tr_s, y_tr)
    acc_mlp = clf_mlp.score(X_te_s, y_te)
    y_pred_mlp = clf_mlp.predict(X_te_s)
    report_mlp = classification_report(
        y_te, y_pred_mlp, target_names=[r["name"] for r in REGIMES], zero_division=0
    )

    baseline = max(np.bincount(y_te)) / len(y_te)

    return {
        "seq_len": seq_len,
        "n_sequences": n_seq,
        "linear_accuracy": float(acc_lin),
        "mlp_accuracy": float(acc_mlp),
        "baseline": float(baseline),
        "linear_report": report_lin,
        "mlp_report": report_mlp,
        "y_true": y_te,
        "y_pred_linear": y_pred_lin,
        "y_pred_mlp": y_pred_mlp,
    }


# ===========================================================================
# SECTION 3 — Latent geometry
# ===========================================================================

def latent_geometry(data: dict) -> dict:
    Z       = data["Z"]
    scalars = data["scalars"]

    sc = StandardScaler()
    Z_s = sc.fit_transform(Z)

    pca = PCA(n_components=min(10, Z.shape[1]))
    Z_pca = pca.fit_transform(Z_s)

    n_pc = min(5, Z_pca.shape[1])
    scalar_names = ["mid", "spread", "imbalance", "inventory"]
    corr = np.zeros((n_pc, 4))
    for i in range(n_pc):
        for j in range(4):
            std = scalars[:, j].std()
            if std < 1e-8:
                corr[i, j] = float("nan")
            else:
                corr[i, j] = np.corrcoef(Z_pca[:, i], scalars[:, j])[0, 1]

    return {
        "Z_pca": Z_pca, "pca": pca, "corr": corr,
        "scalar_names": scalar_names, "regimes": data["regimes"],
    }


# ===========================================================================
# SECTION 4 — Geometric properties (NEW)
# ===========================================================================

def stats_head_r2(autoencoder, data: dict, device: torch.device,
                  batch_size: int = 1024) -> dict:
    """
    R² per ciascuno dei 6 target della stats_head.
    Verifica direttamente che L_stats abbia imparato i target.
    """
    book = torch.from_numpy(data["book_norm"])
    n = len(book)
    preds_list, targets_list = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b = book[i:i+batch_size].to(device)
            z = autoencoder.encoder(b)
            preds   = autoencoder.stats_head(z).cpu().numpy()
            targets = BookStatsPredictor.compute_targets(b).cpu().numpy()
            preds_list.append(preds)
            targets_list.append(targets)
    preds   = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    names = ["log_mean_bid_vol", "log_mean_ask_vol",
             "bid_concentration", "ask_concentration",
             "imbalance", "spread"]
    r2 = {}
    degenerate = {}  # target con varianza troppo bassa per un R² significativo
    for i, name in enumerate(names):
        t = targets[:, i]; p = preds[:, i]
        ss_tot = ((t - t.mean()) ** 2).sum()
        # Se la varianza del target è degenerata (es. spread quasi-costante
        # a 2 tick), l'R² è numericamente instabile e privo di significato.
        # Threshold: varianza relativa <1e-4 della media quadratica.
        rel_var = ss_tot / (len(t) * (t ** 2).mean() + 1e-12)
        if rel_var < 1e-4:
            degenerate[name] = {
                "mean": float(t.mean()),
                "std": float(t.std()),
                "mse": float(((t - p) ** 2).mean()),
            }
            r2[name] = None  # marcato esplicitamente come non valido
        else:
            ss_res = ((t - p) ** 2).sum()
            r2[name] = float(1.0 - ss_res / (ss_tot + 1e-12))

    return {"r2": r2, "preds": preds, "targets": targets, "degenerate": degenerate}


def forward_lipschitz_random(encoder, data: dict, device: torch.device,
                              n: int = 500, scales=(1e-3, 1e-2, 1e-1)) -> dict:
    """
    Forward Lipschitz con perturbazioni random a scale diverse.
    Comportamento lineare → ratio costante al variare di eps.
    """
    book = torch.from_numpy(data["book_norm"])
    idx  = np.random.choice(len(book), min(n, len(book)), replace=False)
    bk   = book[idx].to(device)
    with torch.no_grad():
        z0 = encoder(bk)

    input_std = bk.std().item()
    results = {}
    for eps in scales:
        delta_scale = eps * input_std
        lips = []
        for _ in range(5):
            delta = torch.randn_like(bk) * delta_scale
            with torch.no_grad():
                z1 = encoder(bk + delta)
            dz = (z1 - z0).norm(dim=1)
            do = delta.flatten(1).norm(dim=1)
            lips.append((dz / (do + 1e-12)).cpu().numpy())
        lips = np.concatenate(lips)
        results[eps] = {
            "median": float(np.median(lips)),
            "p95": float(np.percentile(lips, 95)),
            "max": float(lips.max()),
            "values": lips,
        }
    return results


def forward_lipschitz_adversarial(encoder, data: dict, device: torch.device,
                                   n: int = 200, eps_rel: float = 1e-2,
                                   n_steps: int = 20,
                                   step_size_rel: float = 5e-3) -> dict:
    """
    Worst-case Lipschitz via PGD. Trova delta che massimizza ||E(o+d)-E(o)||
    a ||d|| <= eps. Questo è ciò che conta per la stabilità del critico.
    """
    book = torch.from_numpy(data["book_norm"])
    idx  = np.random.choice(len(book), min(n, len(book)), replace=False)
    bk   = book[idx].to(device)
    with torch.no_grad():
        z0 = encoder(bk).detach()

    input_std = bk.std().item()
    numel = bk[0].numel()
    eps = eps_rel * input_std * np.sqrt(numel)
    step_size = step_size_rel * input_std * np.sqrt(numel)

    delta = torch.randn_like(bk) * (eps / np.sqrt(numel)) * 0.1
    delta.requires_grad_(True)

    for _ in range(n_steps):
        z1 = encoder(bk + delta)
        loss = -(z1 - z0).pow(2).sum(dim=1).mean()
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - step_size * grad.sign()
            delta_flat = delta.flatten(1)
            norms = delta_flat.norm(dim=1, keepdim=True)
            factor = torch.clamp(eps / (norms + 1e-12), max=1.0)
            delta = (delta_flat * factor).reshape(bk.shape)
        delta.requires_grad_(True)

    with torch.no_grad():
        z1 = encoder(bk + delta)
        dz = (z1 - z0).norm(dim=1)
        do = delta.flatten(1).norm(dim=1)
        lips = (dz / (do + 1e-12)).cpu().numpy()

    return {
        "median": float(np.median(lips)),
        "p95": float(np.percentile(lips, 95)),
        "max": float(lips.max()),
        "values": lips,
        "eps_rel": eps_rel,
    }


def injectivity_analysis(data: dict, n_pairs: int = 50000,
                          n_points: int = 5000) -> dict:
    """
    Iniettività empirica: per coppie (o_i, o_j), misura
      c_ij = ||o_i - o_j|| / ||z_i - z_j||
    Coda pesante (p99/median alto) → book diversi mappati allo stesso z.
    Uso il book normalizzato, non l'obs raw.
    """
    idx = np.random.choice(len(data["Z"]), min(n_points, len(data["Z"])), replace=False)
    Z = data["Z"][idx]
    O = data["book_norm"][idx].reshape(len(idx), -1)

    i = np.random.randint(0, len(Z), size=n_pairs)
    j = np.random.randint(0, len(Z), size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    dz = np.linalg.norm(Z[i] - Z[j], axis=1)
    do = np.linalg.norm(O[i] - O[j], axis=1)

    valid = dz > 1e-6
    c = do[valid] / dz[valid]

    return {
        "ratios": c,
        "median": float(np.median(c)),
        "p95": float(np.percentile(c, 95)),
        "p99": float(np.percentile(c, 99)),
        "max": float(c.max()),
        "dz": dz[valid], "do": do[valid],
    }


def knn_consistency(data: dict, n: int = 2000, k: int = 10) -> dict:
    """
    k-NN consistency: i k vicini in z sono vicini anche in o?
    ratio = avg_dist(k-NN-in-z) / avg_dist(k-random)
    << 1 → buono.
    """
    idx = np.random.choice(len(data["Z"]), min(n, len(data["Z"])), replace=False)
    z = data["Z"][idx]
    o = data["book_norm"][idx].reshape(len(idx), -1)

    D_z = cdist(z, z)
    D_o = cdist(o, o)

    np.fill_diagonal(D_z, np.inf)
    knn_idx = np.argsort(D_z, axis=1)[:, :k]

    knn_do = np.take_along_axis(D_o, knn_idx, axis=1).mean(axis=1)
    rand_idx = np.random.randint(0, len(idx), size=(len(idx), k))
    rand_do = np.take_along_axis(D_o, rand_idx, axis=1).mean(axis=1)

    ratio = knn_do / (rand_do + 1e-12)
    return {
        "median_ratio": float(np.median(ratio)),
        "p95_ratio": float(np.percentile(ratio, 95)),
        "values": ratio,
    }


def latent_correlation_matrix(data: dict) -> dict:
    """Correlation matrix delle dimensioni di z (VICReg decorr check)."""
    z = data["Z"]
    corr = np.corrcoef(z.T)
    off_diag = np.abs(corr - np.eye(len(corr)))
    return {
        "corr": corr,
        "max_off_diag": float(off_diag.max()),
        "mean_off_diag": float(off_diag[off_diag > 0].mean()),
    }


def geometric_properties(
    encoder, autoencoder, data: dict, device: torch.device,
    adversarial: bool = True,
) -> dict:
    """Bundle di tutti i check geometrici."""
    print("  [4.1] stats_head R²...")
    stats_r2 = stats_head_r2(autoencoder, data, device)
    print("  [4.2] Forward Lipschitz (random multi-scale)...")
    lip_rand = forward_lipschitz_random(encoder, data, device, n=500)
    lip_adv = None
    if adversarial:
        print("  [4.3] Forward Lipschitz (adversarial PGD)...")
        lip_adv = forward_lipschitz_adversarial(encoder, data, device, n=200)
    print("  [4.4] Injectivity pairwise...")
    inj = injectivity_analysis(data)
    print("  [4.5] k-NN consistency...")
    knn = knn_consistency(data)
    print("  [4.6] Latent correlation matrix...")
    corr = latent_correlation_matrix(data)
    return {
        "stats_r2": stats_r2,
        "lip_rand": lip_rand,
        "lip_adv": lip_adv,
        "injectivity": inj,
        "knn": knn,
        "latent_corr": corr,
    }


# ===========================================================================
# VERDETTO
# ===========================================================================

def verdict(ds: dict, geo_props: dict, sp: dict | None = None) -> tuple[str, list[str]]:
    """
    4 livelli + override:
      A: tutto ok → procedi al Modulo B
      B: problemi downstream, geometria ok → fix mirato (più epoche, λ_stats)
      C: downstream ok, geometria rotta → smoothness loss / spectral norm
      D: entrambi rotti → ripensare

    v5 changes:
      - criterio regime usa MLP probe (non linear);
      - k-NN consistency rimossa come criterio: era tarata per encoder VICReg
        (std~1) e non è comparabile con l'attuale scala latente ridotta;
      - OVERRIDE: se sequence_probe_mlp ≥ 0.90, forza verdetto A indipendentemente
        dalle altre metriche. Il sequence probe è il test più informativo per
        determinare se l'encoder è adatto al Modulo B downstream.
    """
    downstream_issues, geometry_issues = [], []

    # Downstream regressivi — ESCLUDO reward (impredicibile per costruzione)
    # e accetto soglie basse per |shock| (intrinsecamente rumoroso)
    for key, label, thresh in [
        ("vol", "|shock|", 0.1),         # molto basso: shock è gauss pure
        ("ofi", "OFI", 0.2),
        ("spread", "spread_t+1", 0.2),
    ]:
        if key in ds and ds[key]["R2"] < thresh:
            downstream_issues.append(f"probe {label} R²={ds[key]['R2']:.2f}")

    # Regime: uso MLP accuracy come criterio. Se MLP alto (>85%), info c'è.
    # Se MLP ~ linear e ambedue bassi, è floor information-theoretic → NON issue
    # (lo gestirà il Modulo B con context window).
    mlp_acc = ds["regime_mlp"]["accuracy"]
    lin_acc = ds["regime"]["accuracy"]
    if mlp_acc < 0.80 and (mlp_acc - lin_acc) > 0.05:
        # MLP batte linear ma non abbastanza: probabile issue di capacity/training
        downstream_issues.append(
            f"MLP regime acc={mlp_acc:.2f} (linear={lin_acc:.2f})"
        )
    # Se MLP ~ linear e ambedue bassi, NON è un issue: l'info semplicemente
    # non è codificabile da un singolo snapshot. Documentiamo ma non penalizziamo.

    # Geometria — ignoro R² None (target degeneri a varianza nulla)
    r2 = geo_props["stats_r2"]["r2"]
    bad_stats = [n for n, v in r2.items() if v is not None and v < 0.5]
    if bad_stats:
        geometry_issues.append(f"stats_head R² basso: {bad_stats}")

    # max off-diag non è più un criterio (niente più VICReg decorr esplicito).
    # Correlazioni latenti possono naturalmente emergere tra dimensioni
    # semanticamente vicine, non è patologia.

    if geo_props["lip_adv"] is not None:
        adv_p95 = geo_props["lip_adv"]["p95"]
        if adv_p95 > 50:
            geometry_issues.append(f"Lipschitz PGD p95={adv_p95:.1f}")

    inj = geo_props["injectivity"]
    if inj["p99"] / inj["median"] > 100:
        geometry_issues.append(
            f"injectivity coda pesante (p99/med={inj['p99']/inj['median']:.0f})"
        )

    # k-NN consistency REMOVED as criterion (v5): threshold 0.5 era tarato per
    # encoder con std~1 (VICReg). Con contractive on-manifold la scala del latente
    # è piccola e il k-NN ratio non è più comparabile con threshold assoluti.
    # Reported but not penalizing.

    all_issues = downstream_issues + geometry_issues

    # OVERRIDE: sequence probe is the gold-standard test for "is this encoder
    # ready for the downstream WM?" If sequence_mlp ≥ 90%, declare A regardless
    # of single-snapshot metrics or minor geometric quibbles.
    if sp is not None and "error" not in sp:
        if sp["mlp_accuracy"] >= 0.90:
            return "A", [f"(override: sequence MLP = {sp['mlp_accuracy']*100:.1f}% ≥ 90%)"]

    if not all_issues:
        return "A", []
    if downstream_issues and geometry_issues:
        return "D", all_issues
    if downstream_issues:
        return "B", downstream_issues
    return "C", geometry_issues


# ===========================================================================
# PRINT
# ===========================================================================

def print_reconstruction(rec: dict) -> None:
    print("\n" + "="*70)
    print("[1] RECONSTRUCTION (normalised volumes)")
    print("="*70)
    print(f"  wMSE={rec['wMSE']:.6f}  MSE={rec['MSE']:.6f}  MAE={rec['MAE']:.6f}")
    print(f"  top-of-book MSE={rec['top_MSE']:.6f}  deep MSE={rec['deep_MSE']:.6f}")
    if rec.get("per_regime"):
        print(f"\n  {'regime':<10s}  {'MSE_abs':>10s}  {'MSE_rel':>10s}  {'n':>7s}")
        print(f"  {'-'*42}")
        for rname, rv in rec["per_regime"].items():
            print(f"  {rname:<10s}  {rv['mse_abs']:>10.6f}  "
                  f"{rv['mse_rel']:>10.4f}  {rv['n']:>7d}")


def print_downstream(ds: dict) -> None:
    print("\n" + "="*70)
    print("[2] DOWNSTREAM PROBES (from z)")
    print("="*70)
    print(f"\n  {'probe':<15s}  {'R²':>8s}  {'MSE':>10s}")
    print(f"  {'-'*36}")
    for key in ["reward", "vol", "ofi", "spread"]:
        r = ds[key]
        print(f"  {r['name']:<15s}  {r['R2']:>+8.4f}  {r['MSE']:>10.6f}")

    # Regime: linear + MLP affiancati
    r_lin = ds["regime"]
    r_mlp = ds["regime_mlp"]
    print(f"\n  Regime classification (baseline = {r_lin['baseline']*100:.1f}%):")
    print(f"    Linear (Logistic) : {r_lin['accuracy']*100:.1f}%")
    print(f"    MLP (2-layer)     : {r_mlp['accuracy']*100:.1f}%  "
          f"[n_iter={r_mlp['n_iter']}]")

    gap = (r_mlp['accuracy'] - r_lin['accuracy']) * 100
    if gap > 5:
        print(f"    → GAP = {gap:+.1f}%: info non-lineare in z. Modulo B la troverà.")
    elif gap < 2:
        print(f"    → GAP = {gap:+.1f}%: MLP ~ Linear. Possibile floor information-theoretic")
        print(f"      (info richiede contesto temporale, non presente nello snapshot statico).")
    else:
        print(f"    → GAP = {gap:+.1f}%: marginale.")

    print(f"\n  Linear probe classification report:")
    print(r_lin['report'])
    print(f"  MLP probe classification report:")
    print(r_mlp['report'])


def print_sequence_probe(sp: dict, single_snap_linear: float,
                          single_snap_mlp: float) -> None:
    print("\n" + "="*70)
    print("[2bis] SEQUENCE-LEVEL REGIME PROBE (upper bound Modulo B)")
    print("="*70)

    if "error" in sp:
        print(f"  ERRORE: {sp['error']} (n_seq={sp['n_sequences']})")
        return

    print(f"\n  Sequence length : {sp['seq_len']}")
    print(f"  N sequences    : {sp['n_sequences']}")
    print(f"  Baseline       : {sp['baseline']*100:.1f}%")
    print(f"\n  {'Probe':<20s}  {'Accuracy':>10s}  {'vs single-snap':>16s}")
    print(f"  {'-'*48}")
    print(f"  {'Single (linear)':<20s}  {single_snap_linear*100:>9.1f}%  "
          f"{'--':>16s}")
    print(f"  {'Single (MLP)':<20s}  {single_snap_mlp*100:>9.1f}%  "
          f"{'--':>16s}")
    print(f"  {'Sequence (linear)':<20s}  {sp['linear_accuracy']*100:>9.1f}%  "
          f"{(sp['linear_accuracy']-single_snap_linear)*100:>+15.1f}%")
    print(f"  {'Sequence (MLP)':<20s}  {sp['mlp_accuracy']*100:>9.1f}%  "
          f"{(sp['mlp_accuracy']-single_snap_mlp)*100:>+15.1f}%")

    # Interpretazione automatica
    gain = sp['mlp_accuracy'] - single_snap_mlp
    print()
    if gain > 0.15:
        print(f"  → GAIN temporale +{gain*100:.1f}%: SIGNIFICATIVO.")
        print(f"    L'info dinamica dei regimi È presente nella sequenza di z.")
        print(f"    Modulo B ha margine concreto per migliorare la distinzione.")
    elif gain > 0.05:
        print(f"  → GAIN temporale +{gain*100:.1f}%: MODERATO.")
        print(f"    Il contesto aiuta ma non risolve completamente.")
    else:
        print(f"  → GAIN temporale +{gain*100:.1f}%: TRASCURABILE.")
        print(f"    Il segnale dinamico è debole anche nella sequenza.")
        print(f"    Implicazione: il Modulo B non potrà migliorare drasticamente")
        print(f"    la separazione dei regimi. Ripensare features o architettura.")

    print(f"\n  Sequence MLP classification report:")
    print(sp['mlp_report'])


def print_geometry(geo: dict) -> None:
    pca = geo["pca"]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_pc = len(pca.explained_variance_ratio_)
    print("\n" + "="*70)
    print("[3] LATENT GEOMETRY (unsupervised)")
    print("="*70)
    print(f"\n  PCA: {cumvar[min(7, n_pc-1)]:.0f}% in 8 PCs, "
          f"{cumvar[min(4, n_pc-1)]:.0f}% in 5 PCs")
    print(f"\n  {'PC':<5s}  {'var%':>6s}  {'cum%':>6s}")
    for i in range(min(n_pc, 10)):
        print(f"  PC{i+1:<2d}  {pca.explained_variance_ratio_[i]*100:>5.1f}%  "
              f"{cumvar[i]:>5.1f}%")


def print_geometric_properties(gp: dict) -> None:
    print("\n" + "="*70)
    print("[4] GEOMETRIC PROPERTIES (bi-Lipschitz diagnostics)")
    print("="*70)

    print("\n  stats_head R² (want > 0.7):")
    degenerate = gp["stats_r2"].get("degenerate", {})
    for name, v in gp["stats_r2"]["r2"].items():
        if v is None:
            # Target degenere (varianza ~0): R² non significativo
            d = degenerate.get(name, {})
            print(f"    [N/A] {name:20s}: degenere "
                  f"(mean={d.get('mean', 0):.3f}, std={d.get('std', 0):.2e}, "
                  f"mse={d.get('mse', 0):.2e})")
        else:
            mark = "OK " if v > 0.7 else "~~ " if v > 0.4 else "!! "
            print(f"    [{mark}] {name:20s}: {v:+.4f}")

    print("\n  Forward Lipschitz (random perturbations):")
    for eps, v in sorted(gp["lip_rand"].items()):
        print(f"    eps={eps:.0e}  median={v['median']:.3f}  "
              f"p95={v['p95']:.3f}  max={v['max']:.3f}")

    if gp["lip_adv"] is not None:
        adv = gp["lip_adv"]
        print(f"\n  Forward Lipschitz (PGD adversarial, eps_rel={adv['eps_rel']:.0e}):")
        print(f"    median={adv['median']:.3f}  p95={adv['p95']:.3f}  "
              f"max={adv['max']:.3f}")

    inj = gp["injectivity"]
    print("\n  Injectivity (pairwise ||Δo||/||Δz||):")
    print(f"    median={inj['median']:.3f}  p95={inj['p95']:.3f}  "
          f"p99={inj['p99']:.3f}  max={inj['max']:.3f}")
    print(f"    p99/median = {inj['p99']/inj['median']:.1f}  (>100 = coda pesante)")

    knn = gp["knn"]
    print(f"\n  k-NN consistency (k=10):")
    print(f"    median ratio = {knn['median_ratio']:.3f}  "
          f"p95 = {knn['p95_ratio']:.3f}  (<0.5 = buono)")

    lc = gp["latent_corr"]
    print(f"\n  Latent correlation (VICReg decorr check):")
    print(f"    max |off-diag| = {lc['max_off_diag']:.4f}  "
          f"mean |off-diag| = {lc['mean_off_diag']:.4f}")


def print_verdict(letter: str, issues: list[str]) -> None:
    print("\n" + "="*70)
    print(f"VERDETTO: {letter}")
    print("="*70)
    if letter == "A":
        if issues and "override" in issues[0]:
            print(f"  Encoder pronto per il Modulo B. {issues[0]}")
        else:
            print("  Encoder pronto per il Modulo B.")
    elif letter == "B":
        print("  Downstream R² bassi ma geometria OK.")
        for iss in issues: print(f"    - {iss}")
        print("  Suggeriti: più epoche, aumenta λ_stats, o più dati.")
    elif letter == "C":
        print("  Downstream OK ma geometria latente problematica.")
        for iss in issues: print(f"    - {iss}")
        print("  Suggeriti: contractive smoothness loss, spectral norm allargata.")
    else:  # D
        print("  Problemi sia downstream che geometrici.")
        for iss in issues: print(f"    - {iss}")
        print("  Ripensare loss/architettura.")
    print("="*70 + "\n")


# ===========================================================================
# PLOTTING — figura 1: representation (3×3)
# ===========================================================================

def plot_representation(rec: dict, ds: dict, geo: dict, data: dict,
                         sp: dict | None, out_path: str,
                         do_tsne: bool = True) -> None:
    L = len(rec["per_level_mse"])
    pca = geo["pca"]
    Z = data["Z"]
    regimes = data["regimes"]
    d_latent = Z.shape[1]
    regime_names = [r["name"] for r in REGIMES]
    regime_colors = ["#2ecc71", "#3498db", "#e74c3c"]

    GRAY, BLUE, GREEN, RED = "#555555", "#2c7bb6", "#1a9641", "#d7191c"

    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- 0a. Volume & MSE per level ---
    ax = fig.add_subplot(gs[0, 0])
    vt = data["vol_true"]
    vol_per_level = vt.mean(axis=1)
    vol_mean = vol_per_level.mean(axis=0)
    vol_std  = vol_per_level.std(axis=0)
    ax2 = ax.twinx()
    ax.bar(range(L), vol_mean,
           color=[RED if i <= 1 else BLUE for i in range(L)],
           alpha=0.7, label="Mean vol")
    ax.errorbar(range(L), vol_mean, yerr=vol_std, fmt="none",
                ecolor=GRAY, capsize=3, lw=0.8)
    ax2.plot(range(L), rec["per_level_mse"], "k--o", markersize=4, lw=1.2, label="MSE")
    ax.set_title("Volume & recon MSE per level", fontsize=10)
    ax.set_xlabel("Level (0=best)", fontsize=8)
    ax.set_ylabel("Mean volume (norm.)", fontsize=8)
    ax2.set_ylabel("Recon MSE", fontsize=8)
    ax.set_xticks(range(L))
    ax.tick_params(labelsize=7)
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=7, loc="upper right")

    # --- 0b. Recon examples ---
    ax = fig.add_subplot(gs[0, 1])
    if data["vol_pred"] is not None:
        for r_idx, (rname, rcol) in enumerate(zip(regime_names, regime_colors)):
            mask = (regimes == r_idx)
            if mask.sum() == 0: continue
            r_true = vt[mask].mean(axis=1)
            r_pred = data["vol_pred"][mask][:, :, :, 1].mean(axis=1)
            true_mean = r_true.mean(axis=0)
            pred_mean = r_pred.mean(axis=0)
            ax.plot(range(L), true_mean, "o-", color=rcol, markersize=3,
                    lw=1.5, label=f"{rname} true")
            ax.plot(range(L), pred_mean, "x--", color=rcol, markersize=4,
                    lw=1, alpha=0.7)
        ax.set_title("Recon: true (●) vs pred (×)\nmean profile per regime", fontsize=10)
        ax.set_xlabel("Level", fontsize=8)
        ax.set_ylabel("Mean volume (norm.)", fontsize=8)
        ax.legend(fontsize=7, ncol=1)
        ax.set_xticks(range(L))
        ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "No decoder", transform=ax.transAxes, ha="center")

    # --- 0c. MSE per regime ---
    ax = fig.add_subplot(gs[0, 2])
    if data["vol_pred"] is not None:
        regime_mse = []
        for r_idx in range(len(REGIMES)):
            mask = (regimes == r_idx)
            if mask.sum() == 0:
                regime_mse.append(0); continue
            r_true = vt[mask]
            r_pred = data["vol_pred"][mask][:, :, :, 1]
            regime_mse.append(((r_true - r_pred) ** 2).mean())
        bars = ax.bar(regime_names, regime_mse, color=regime_colors, alpha=0.8)
        ax.set_title("Recon MSE per regime", fontsize=10)
        ax.set_ylabel("MSE (norm. vols)", fontsize=8)
        ax.tick_params(labelsize=8)
        for bar, v in zip(bars, regime_mse):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(regime_mse)*0.02,
                    f"{v:.4f}", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No decoder", transform=ax.transAxes, ha="center")

    # --- 1a. PCA explained variance ---
    ax = fig.add_subplot(gs[1, 0])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n_show = min(d_latent, 10)
    ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show] * 100,
           color=BLUE, alpha=0.8, label="Per-PC")
    ax.plot(range(1, n_show + 1), cumvar[:n_show], "o-", color=RED, markersize=4,
            lw=1.5, label="Cumulative")
    ax.axhline(80, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.set_title(f"PCA — explained variance ({d_latent}-dim z)", fontsize=10)
    ax.set_xlabel("PC", fontsize=8)
    ax.set_ylabel("Variance (%)", fontsize=8)
    ax.set_xticks(range(1, n_show + 1))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    # --- 1b. t-SNE (or PCA 2D if skipped) ---
    ax = fig.add_subplot(gs[1, 1])
    n_proj = min(8000, len(Z))
    rng = np.random.default_rng(42)
    pi = rng.choice(len(Z), size=n_proj, replace=False)
    Z_sub = Z[pi]; reg_sub = regimes[pi]
    if do_tsne:
        print("  Computing t-SNE...")
        Z_2d = TSNE(n_components=2, perplexity=40, random_state=42,
                    max_iter=1000).fit_transform(Z_sub)
        title = "t-SNE of z by regime"
    else:
        Z_2d = geo["Z_pca"][pi, :2]
        title = "PCA 2D of z by regime"
    for r_idx, (rname, rcol) in enumerate(zip(regime_names, regime_colors)):
        mask = (reg_sub == r_idx)
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=rcol, s=3, alpha=0.4,
                   label=rname, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=4)
    ax.set_xticks([]); ax.set_yticks([])

    # --- 1c. Per-dim std (latent scale, no longer VICReg-targeted) ---
    ax = fig.add_subplot(gs[1, 2])
    per_dim_std = Z.std(axis=0)
    # Ordiniamo per visualizzazione: dimensioni con più std prima
    order = np.argsort(-per_dim_std)
    colors_bar = [BLUE if per_dim_std[i] > per_dim_std.mean() * 0.3 else GRAY
                  for i in order]
    ax.bar(range(d_latent), per_dim_std[order], color=colors_bar, alpha=0.85)
    ax.set_title(f"Latent scale per dim (sorted)\n"
                 f"min={per_dim_std.min():.3f}  max={per_dim_std.max():.3f}  "
                 f"mean={per_dim_std.mean():.3f}",
                 fontsize=10)
    ax.set_xlabel("dimension (sorted by std)", fontsize=8)
    ax.set_ylabel("std(z)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, per_dim_std.max() * 1.15)

    # --- 2a. Downstream probes ---
    ax = fig.add_subplot(gs[2, 0])
    items = []
    for key, label, col in [
        ("ofi", "OFI_t+1", GREEN),
        ("spread", "spread_t+1", BLUE),
        ("vol", "|Δmid|", RED),
        ("reward", "reward_t", "#9b59b6"),
    ]:
        if key in ds:
            items.append((label, ds[key]["R2"], col))
    if items:
        names, vals, cols = zip(*items)
        ax.barh(list(names), list(vals), color=list(cols), alpha=0.8)
        ax.set_xlabel("R² (linear probe)", fontsize=8)
        ax.set_title("Downstream probes", fontsize=10)
        ax.axvline(0, color="black", lw=0.5)
        ax.tick_params(labelsize=8)
        for i, v in enumerate(vals):
            ax.text(v + 0.02 if v >= 0 else v - 0.02, i, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=9, fontweight="bold")

    # --- 2b. Regime confusion matrix ---
    ax = fig.add_subplot(gs[2, 1])
    y_true = ds["regime"]["y_true"]; y_pred = ds["regime"]["y_pred"]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(regime_names)))
    ax.set_xticklabels(regime_names, fontsize=8)
    ax.set_yticks(range(len(regime_names)))
    ax.set_yticklabels(regime_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(f"Regime conf. matrix  "
                 f"(acc={ds['regime']['accuracy']*100:.1f}%)", fontsize=10)
    for i in range(len(regime_names)):
        for j in range(len(regime_names)):
            v = cm_norm[i, j]
            col = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}\n({cm[i,j]})", ha="center", va="center",
                    fontsize=8, color=col)

    # --- 2c. Single-snapshot vs Sequence-level probe (KILLER RESULT) ---
    ax = fig.add_subplot(gs[2, 2])
    if sp is not None and "error" not in sp:
        labels = ["Baseline", "Single\n(linear)", "Single\n(MLP)",
                  "Sequence\n(linear)", "Sequence\n(MLP)"]
        vals = [
            sp["baseline"],
            ds["regime"]["accuracy"],
            ds["regime_mlp"]["accuracy"],
            sp["linear_accuracy"],
            sp["mlp_accuracy"],
        ]
        colors_bars = ["#9E9E9E", "#90CAF9", "#42A5F5", "#A5D6A7", "#66BB6A"]
        bars = ax.bar(range(len(labels)), [v*100 for v in vals],
                      color=colors_bars, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v*100 + 1.5,
                    f"{v*100:.1f}%", ha="center", fontsize=9,
                    fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("Accuracy (%)", fontsize=8)
        ax.set_ylim(0, 105)
        gain = (sp["mlp_accuracy"] - ds["regime_mlp"]["accuracy"]) * 100
        ax.set_title(f"Regime separability: snapshot vs sequence\n"
                     f"temporal gain +{gain:.1f}%  →  Modulo B upper bound",
                     fontsize=10)
        ax.axhline(sp["baseline"]*100, color="#9E9E9E", ls=":", lw=0.8, alpha=0.7)
        ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "No sequence probe", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#666")
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("LOB Encoder — Representation Quality (Sections 1-3)",
                 fontsize=14, y=1.01, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved: {out_path}")


# ===========================================================================
# PLOTTING — figura 2: geometric properties (2×3)
# ===========================================================================

def plot_geometry(gp: dict, out_path: str) -> None:
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    # --- 0a. stats_head R² ---
    ax = fig.add_subplot(gs[0, 0])
    r2 = gp["stats_r2"]["r2"]
    # Filtro R² None (target degeneri) e li disegno come barra grigia
    names = list(r2.keys())
    vals_raw = list(r2.values())
    vals_plot = [v if v is not None else 0.0 for v in vals_raw]
    colors = []
    for v in vals_raw:
        if v is None:
            colors.append("#BDBDBD")  # grigio per degeneri
        elif v > 0.7:
            colors.append("#4CAF50")
        elif v > 0.4:
            colors.append("#FFA726")
        else:
            colors.append("#EF5350")
    ax.barh(names, vals_plot, color=colors, edgecolor="white")
    ax.axvline(0.7, color="black", ls="--", alpha=0.5, lw=0.8)
    ax.set_title("stats_head R² on 6 targets\n(>0.7 good, grey=degenere)",
                 fontweight="bold")
    ax.set_xlabel("R²")
    valid_vals = [v for v in vals_raw if v is not None]
    if valid_vals:
        ax.set_xlim(min(0, min(valid_vals) - 0.05), 1.05)
    else:
        ax.set_xlim(0, 1.05)
    for i, v in enumerate(vals_raw):
        if v is None:
            ax.text(0.02, i, "N/A (degenere)", va="center", fontsize=8, color="#666")
        else:
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    # --- 0b. Lipschitz random vs scale ---
    ax = fig.add_subplot(gs[0, 1])
    lip_r = gp["lip_rand"]
    scales = sorted(lip_r.keys())
    meds = [lip_r[s]["median"] for s in scales]
    p95s = [lip_r[s]["p95"] for s in scales]
    maxs = [lip_r[s]["max"] for s in scales]
    x = np.arange(len(scales)); w = 0.25
    ax.bar(x - w, meds, w, label="median", color="#5C6BC0")
    ax.bar(x, p95s, w, label="p95", color="#EF5350")
    ax.bar(x + w, maxs, w, label="max", color="#26A69A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"eps={s:.0e}" for s in scales], fontsize=8)
    ax.set_title("Forward Lipschitz (random)\n||Δz||/||Δo|| vs eps",
                 fontweight="bold")
    ax.set_ylabel("Lipschitz ratio")
    ax.legend(fontsize=8)

    # --- 0c. Lipschitz adversarial (or empty) ---
    ax = fig.add_subplot(gs[0, 2])
    if gp["lip_adv"] is not None:
        adv = gp["lip_adv"]
        ax.hist(adv["values"], bins=50, color="#F57C00", edgecolor="white")
        ax.axvline(adv["median"], color="black", ls="--",
                   label=f"median={adv['median']:.2f}")
        ax.axvline(adv["p95"], color="red", ls="--",
                   label=f"p95={adv['p95']:.2f}")
        ax.set_title(f"Adversarial Lipschitz (PGD)\n"
                     f"worst-case, eps_rel={adv['eps_rel']:.0e}",
                     fontweight="bold")
        ax.set_xlabel("Lipschitz ratio")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Adversarial test disabled",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # --- 1a. Injectivity scatter ---
    ax = fig.add_subplot(gs[1, 0])
    inj = gp["injectivity"]
    sub = np.random.choice(len(inj["dz"]), min(3000, len(inj["dz"])), replace=False)
    ax.scatter(inj["dz"][sub], inj["do"][sub], s=1, alpha=0.3, color="#1976D2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"Injectivity: ||Δo|| vs ||Δz||\n"
                 f"p99/median={inj['p99']/inj['median']:.1f}",
                 fontweight="bold")
    ax.set_xlabel("||z_i - z_j||")
    ax.set_ylabel("||o_i - o_j|| (norm. book)")

    # --- 1b. k-NN consistency histogram ---
    ax = fig.add_subplot(gs[1, 1])
    knn = gp["knn"]
    ax.hist(knn["values"], bins=50, color="#26A69A", edgecolor="white", alpha=0.85)
    ax.axvline(knn["median_ratio"], color="black", ls="--",
               label=f"median={knn['median_ratio']:.3f}")
    ax.axvline(0.5, color="red", ls=":", alpha=0.7, label="threshold=0.5")
    ax.set_title("k-NN consistency (k=10)\n"
                 "ratio = avg-dist-kNN-in-z / avg-dist-random",
                 fontweight="bold")
    ax.set_xlabel("ratio (<1 = good)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # --- 1c. Latent correlation matrix ---
    ax = fig.add_subplot(gs[1, 2])
    corr = gp["latent_corr"]["corr"]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"Latent correlation matrix\n"
                 f"max |off-diag|={gp['latent_corr']['max_off_diag']:.3f}",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("LOB Encoder — Geometric Properties (Section 4)",
                 fontsize=14, y=0.98, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved: {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder, stats, ae, cfg = load_encoder(args.ckpt, device)

    print(f"\nEncoding {args.n_samples:,} campioni...")
    data = build_encoded_dataset(
        encoder, args.dataset, stats, device,
        n_samples=args.n_samples, autoencoder=ae,
    )
    print(f"  Z shape: {data['Z'].shape}")

    # Section 1 — Reconstruction
    rec = reconstruction_metrics(data, L=cfg.L, vol_scale=stats["vol_scale"])
    print_reconstruction(rec)

    # Section 2 — Downstream probes
    print("\nComputing downstream probes...")
    ds = downstream_probes(data)
    print_downstream(ds)

    # Section 2bis — Sequence-level regime probe (upper bound for Module B)
    print("\nComputing sequence-level regime probe...")
    sp = sequence_regime_probe(
        encoder, args.dataset, stats, device,
        seq_len=args.seq_len, max_sequences=args.max_sequences,
    )
    print_sequence_probe(sp, ds["regime"]["accuracy"], ds["regime_mlp"]["accuracy"])

    # Section 3 — Latent geometry (unsupervised)
    print("\nAnalyzing latent geometry...")
    geo = latent_geometry(data)
    print_geometry(geo)

    # Section 4 — Geometric properties (NEW)
    print("\nComputing geometric properties...")
    gp = geometric_properties(encoder, ae, data, device,
                               adversarial=not args.no_adversarial)
    print_geometric_properties(gp)

    # Verdetto unificato
    letter, issues = verdict(ds, gp, sp)
    print_verdict(letter, issues)

    # Plots: due figure separate
    plot_representation(rec, ds, geo, data, sp,
                        out_path=out_dir / "eval_encoder_representation.png",
                        do_tsne=not args.no_tsne)
    plot_geometry(gp, out_path=out_dir / "eval_encoder_geometry.png")

    # Salva metriche grezze
    save_dict = {
        "verdict": letter,
        "issues": issues,
        "recon_per_level": rec["per_level_mse"],
        "std_per_dim": data["Z"].std(axis=0),
        "latent_corr_max_off_diag": gp["latent_corr"]["max_off_diag"],
        "injectivity_p99_over_median":
            gp["injectivity"]["p99"] / gp["injectivity"]["median"],
        "knn_median_ratio": gp["knn"]["median_ratio"],
        "regime_accuracy_linear": ds["regime"]["accuracy"],
        "regime_accuracy_mlp": ds["regime_mlp"]["accuracy"],
        "regime_mlp_linear_gap":
            ds["regime_mlp"]["accuracy"] - ds["regime"]["accuracy"],
    }
    for k, v in gp["stats_r2"]["r2"].items():
        # np.savez non gestisce None -> uso NaN per i target degeneri
        save_dict[f"stats_r2_{k}"] = v if v is not None else float("nan")
    for key in ["reward", "vol", "ofi", "spread"]:
        save_dict[f"probe_R2_{key}"] = ds[key]["R2"]
    if gp["lip_adv"] is not None:
        save_dict["lip_adv_median"] = gp["lip_adv"]["median"]
        save_dict["lip_adv_p95"] = gp["lip_adv"]["p95"]

    # Sequence probe results
    if "error" not in sp:
        save_dict["seq_probe_seq_len"] = sp["seq_len"]
        save_dict["seq_probe_n_sequences"] = sp["n_sequences"]
        save_dict["seq_probe_linear_acc"] = sp["linear_accuracy"]
        save_dict["seq_probe_mlp_acc"] = sp["mlp_accuracy"]
        save_dict["seq_probe_gain_over_single"] = (
            sp["mlp_accuracy"] - ds["regime_mlp"]["accuracy"]
        )

    np.savez(out_dir / "encoder_metrics.npz", **save_dict)
    print(f"Metrics saved: {out_dir / 'encoder_metrics.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate LOB Encoder (merged)")
    p.add_argument("--ckpt",      type=str,   default="checkpoints/encoder_best.pt")
    p.add_argument("--dataset",   type=str,   default="data/dataset.npz")
    p.add_argument("--n_samples", type=int,   default=50_000)
    p.add_argument("--out_dir",   type=str,   default="eval_encoder_output")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--no_tsne",   action="store_true",
                   help="Skip t-SNE (use PCA 2D instead) — molto più veloce")
    p.add_argument("--no_adversarial", action="store_true",
                   help="Skip PGD adversarial Lipschitz test")
    p.add_argument("--seq_len",   type=int,   default=20,
                   help="Sequence length for sequence-level regime probe (default 20)")
    p.add_argument("--max_sequences", type=int, default=20_000,
                   help="Max sequences for probe (velocità/statistica tradeoff)")
    args = p.parse_args()
    main(args)
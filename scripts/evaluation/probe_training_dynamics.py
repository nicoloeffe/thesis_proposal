#!/usr/bin/env python3
"""
probe_training_dynamics.py — Probe versione 3: dinamica di training F1-v4.

Scopo
=====
Caratterizza il fenomeno "JEPA loss scende ma probe predittivo degrada" osservato
durante il training di F1-v4 (Horizon JEPA). Gira su TUTTI i checkpoint salvati
per epoca (epoch_001.pt ... epoch_NNN.pt) e, per ciascuno:

  1. carica l'encoder online
  2. estrae il readout `last_concat512` su un set fisso train/val
  3. addestra due probe sui 22 target (feature x horizon + realized vol):
        - LinearBottleneckProbe  z_dim=32   (compressione supervised, come baseline)
        - MLPProbe               hidden=256 (probe non lineare)
  4. registra R^2 per OGNI target separato (versione 3 = granularita per
     feature x horizon, non solo aggregato)
  5. legge dalle val_metrics salvate dentro il checkpoint le curve di:
        - JEPA loss totale e per-horizon (L_total, L_H{H})
        - geometria del latente (pooled_std, pooled_eff_rank)

Coerenza con i numeri di riferimento
====================================
Il readout e lo z_dim sono scelti per essere confrontabili con la metrica
principale gia usata nella tesi (concat512 -> linear bottleneck z32 -> targets).
Lo standardizer dei target e lo split train/val sono fittati UNA volta sola e
riusati identici per tutti i checkpoint: senza questo le curve R^2 non sono
confrontabili tra epoche.

Output
======
out_dir/
    probe_dynamics_long.csv     riga = (epoch, probe_type, target_name, feature,
                                        horizon, r2)
    training_dynamics.csv       riga = (epoch, val_L_total, val_L_H*, pooled_std,
                                        pooled_eff_rank, ...)
    plot_loss_vs_probe.png      R^2 aggregato vs epoche con JEPA loss sovrapposta
    plot_per_horizon.png        R^2 per horizon vs epoche
    plot_per_feature.png        R^2 per feature vs epoche
    plot_geometry.png           pooled_std / eff_rank vs epoche, con R^2 sovrapp.
    summary.json                meta + epoca di picco R^2 + diagnosi sintetica

Uso
===
    python probe_training_dynamics.py \
        --dataset data/lobench_processed.npz \
        --ckpt_dir checkpoints/jepa_horizon/v1 \
        --out_dir validation/training_dynamics/v1 \
        --max_train_samples 100000 \
        --max_val_samples 50000 \
        --batch_size 512 \
        --num_workers 2

Nota: questo script NON modifica il trainer ne il probe esistente. Importa e
riusa le funzioni di probe_jepa_horizon_readouts.py e train_tokenizer_t.py.
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

# Headless backend: lo script puo girare via SSH / senza display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Robust import path: trova i moduli di progetto sia da root sia da subdir ---
HERE = Path(__file__).resolve()
for _p in [HERE.parent, *HERE.parents, Path.cwd(), *Path.cwd().parents]:
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# --- Riuso del probe esistente: niente riscrittura di logica gia testata. ---
try:
    from probe_jepa_horizon_readouts import (
        load_horizon_jepa_encoder,
        extract_token_grids,
        RawWindowDataset,
        LinearBottleneckProbe,
        MLPProbe,
        train_torch_probe,
        r2_per_target,
        standardize_x,
        to_numpy_stats,
        local_grouped_split_by_stock_day,
        maybe_subsample,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Impossibile importare da probe_jepa_horizon_readouts.py. "
        "Esegui dalla root del progetto. Errore: " + repr(e)
    )

try:
    from train_tokenizer_t import (
        compute_valid_endpoints,
        derive_raw_features_array,
        compute_future_feature_targets,
        compute_vol_targets,
        fit_target_standardizer,
        apply_standardizer,
        compute_stock_stats_train_only,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Impossibile importare da train_tokenizer_t.py. "
        "Esegui dalla root del progetto. Errore: " + repr(e)
    )

try:
    from train_tokenizer_t import grouped_split_by_stock_day  # type: ignore
except Exception:
    grouped_split_by_stock_day = None


# =============================================================================
#  Checkpoint discovery
# =============================================================================

def discover_checkpoints(ckpt_dir: Path) -> List[Tuple[int, Path]]:
    """Trova epoch_XXX.pt e li ordina per numero di epoca crescente.

    Ritorna lista di (epoch, path). best.pt e last.pt sono ignorati: questo
    script lavora sulla SEQUENZA per-epoca.
    """
    found: List[Tuple[int, Path]] = []
    for p in sorted(ckpt_dir.glob("epoch_*.pt")):
        stem = p.stem  # "epoch_007"
        try:
            ep = int(stem.split("_")[1])
        except (IndexError, ValueError):
            print(f"  [skip] nome non parsabile: {p.name}")
            continue
        found.append((ep, p))
    found.sort(key=lambda x: x[0])
    return found


def read_val_metrics_from_ckpt(path: Path, device: torch.device) -> Dict:
    """Legge val_metrics salvato dentro il checkpoint senza costruire l'encoder.

    val_metrics e' il dict ritornato da run_epoch in fase di validazione:
    contiene L_total, L_H{H}, pooled_std, pooled_eff_rank, cos/gap, ecc.
    Robusto al fatto che history.json potrebbe non esistere (training interrotto).
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    vm = ckpt.get("val_metrics", {}) or {}
    vm = dict(vm)
    vm["_epoch"] = int(ckpt.get("epoch", -1))
    vm["_horizons"] = list(ckpt.get("horizons", []))
    return vm


# =============================================================================
#  One-time setup: dataset, split, target observabili, standardizer
# =============================================================================

def build_fixed_probe_data(args, device: torch.device):
    """Costruisce UNA volta sola: valid_t, split train/val, target standardizzati,
    stock_stats. Tutto cio' che NON dipende dai pesi dell'encoder.

    Ritorna un dict con le strutture riusate a ogni checkpoint.
    """
    print("\n[setup] Caricamento dataset LOBench...")
    raw = np.load(args.dataset)
    book = raw["book"].astype(np.float32, copy=False)
    mid_z = raw["mid_z"].astype(np.float32, copy=False)
    stock_ids = raw["stock_ids"].astype(np.int64, copy=False)
    day_ids = raw["day_ids"].astype(np.int64, copy=False)
    if "min_spread_z_per_stock" in raw.files:
        min_spread = raw["min_spread_z_per_stock"].astype(np.float32, copy=False)
        n_stocks = int(min_spread.shape[0])
    else:
        raise SystemExit("Il dataset NPZ deve contenere min_spread_z_per_stock")
    print(f"  N={len(mid_z):,}  n_stocks={n_stocks}  L={book.shape[2]}")

    future_features = [x.strip() for x in args.future_features.split(",") if x.strip()]
    future_horizons = [int(x) for x in args.future_horizons.split(",") if x.strip()]
    vol_horizons = [int(x) for x in args.vol_horizons.split(",") if x.strip()]
    max_h = max(max(future_horizons), max(vol_horizons))

    print("\n[setup] Valid endpoints + split grouped (FISSO per tutti i checkpoint)...")
    bid_v = book[:, 0, :, 1]
    ask_v = book[:, 1, :, 1]
    vol_mask = (np.abs(bid_v).max(axis=1) <= args.vol_clip) & \
               (np.abs(ask_v).max(axis=1) <= args.vol_clip)
    valid_t = compute_valid_endpoints(stock_ids, day_ids, args.K, max_h, vol_mask)

    splitter = grouped_split_by_stock_day if grouped_split_by_stock_day is not None \
        else local_grouped_split_by_stock_day
    train_pos, val_pos = splitter(stock_ids, day_ids, valid_t, args.val_frac, args.split_seed)
    train_pos = maybe_subsample(train_pos, args.max_train_samples, args.seed + 11)
    val_pos = maybe_subsample(val_pos, args.max_val_samples, args.seed + 17)
    train_t = valid_t[train_pos]
    val_t = valid_t[val_pos]
    print(f"  valid_t={len(valid_t):,}  train={len(train_t):,}  val={len(val_t):,}  max_h={max_h}")

    print("\n[setup] Costruzione target osservabili (UNA volta, indipendenti dall'encoder)...")
    t0 = time.time()
    raw_feat, _ = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)
    fut_train_raw = compute_future_feature_targets(raw_feat, train_t, future_features, future_horizons)
    fut_val_raw = compute_future_feature_targets(raw_feat, val_t, future_features, future_horizons)
    vol_train_raw = compute_vol_targets(mid_z, train_t, vol_horizons, min_spread, stock_ids)
    vol_val_raw = compute_vol_targets(mid_z, val_t, vol_horizons, min_spread, stock_ids)
    y_train_raw = np.concatenate([fut_train_raw, vol_train_raw], axis=1).astype(np.float32)
    y_val_raw = np.concatenate([fut_val_raw, vol_val_raw], axis=1).astype(np.float32)

    # Standardizer fittato SOLO su train, riusato identico ovunque.
    y_mu, y_sd = fit_target_standardizer(y_train_raw)
    y_train = apply_standardizer(y_train_raw, y_mu, y_sd).astype(np.float32)
    y_val = apply_standardizer(y_val_raw, y_mu, y_sd).astype(np.float32)

    target_names: List[str] = []
    target_feature: List[str] = []
    target_horizon: List[int] = []
    for f in future_features:
        for h in future_horizons:
            target_names.append(f"{f}@{h}")
            target_feature.append(f)
            target_horizon.append(h)
    for h in vol_horizons:
        target_names.append(f"realized_vol@{h}")
        target_feature.append("realized_vol")
        target_horizon.append(h)
    print(f"  target: {len(target_names)}  shape train={y_train.shape} val={y_val.shape}  "
          f"({time.time()-t0:.1f}s)")

    return {
        "book": book, "mid_z": mid_z, "stock_ids": stock_ids, "day_ids": day_ids,
        "n_stocks": n_stocks, "train_t": train_t, "val_t": val_t,
        "y_train": y_train, "y_val": y_val,
        "target_names": target_names, "target_feature": target_feature,
        "target_horizon": target_horizon,
    }


def resolve_stock_stats(ckpt_path: Path, data: Dict, args, device: torch.device) -> Dict[str, np.ndarray]:
    """stock_stats e' identico per tutti i checkpoint dello stesso run (train-only).
    Lo prendiamo dal primo checkpoint se presente, altrimenti lo ricalcoliamo.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if "stock_stats" in ckpt and ckpt["stock_stats"]:
        print("  stock_stats: dal checkpoint")
        return to_numpy_stats(ckpt["stock_stats"])
    if compute_stock_stats_train_only is not None:
        print("  stock_stats: ricalcolato train-only")
        return compute_stock_stats_train_only(
            data["book"], data["mid_z"], data["stock_ids"], data["day_ids"],
            data["train_t"], data["n_stocks"],
        )
    raise SystemExit("stock_stats assente nel checkpoint e ricalcolo non disponibile")


# =============================================================================
#  Probe di un singolo checkpoint
# =============================================================================

def probe_one_checkpoint(
    ckpt_path: Path,
    epoch: int,
    data: Dict,
    stock_stats: Dict[str, np.ndarray],
    args,
    device: torch.device,
) -> Tuple[List[Dict], Dict]:
    """Estrae readout + addestra linear & MLP probe per un checkpoint.

    Ritorna:
        rows:      lista di dict (una riga per (probe_type, target))
        diag:      dict con info diagnostiche del checkpoint
    """
    encoder, ckpt = load_horizon_jepa_encoder(str(ckpt_path), device)
    K = int(encoder.cfg.K)

    ds_train = RawWindowDataset(
        data["book"], data["mid_z"], data["stock_ids"], data["train_t"], stock_stats, K)
    ds_val = RawWindowDataset(
        data["book"], data["mid_z"], data["stock_ids"], data["val_t"], stock_stats, K)

    r_train = extract_token_grids(encoder, ds_train, args.batch_size, args.num_workers, device, "train")
    r_val = extract_token_grids(encoder, ds_val, args.batch_size, args.num_workers, device, "val")

    # Readout: concat512 (concat dei 4 token semantici all'ultimo timestep).
    xtr, xva, _, _ = standardize_x(r_train["last_concat512"], r_val["last_concat512"])
    y_train = data["y_train"]
    y_val = data["y_val"]
    target_names = data["target_names"]

    rows: List[Dict] = []

    # --- Probe A: linear bottleneck z32 ---
    lin = LinearBottleneckProbe(in_dim=xtr.shape[1], z_dim=args.z_dim, out_dim=y_train.shape[1])
    yhat_lin, info_lin = train_torch_probe(
        lin, xtr, y_train, xva, y_val, device,
        batch_size=args.probe_batch_size, epochs=args.probe_epochs,
        lr=args.probe_lr, weight_decay=args.probe_weight_decay,
        patience=args.probe_patience, label=f"ep{epoch:03d}_linz{args.z_dim}",
    )
    r2_lin = r2_per_target(y_val, yhat_lin)

    # --- Probe B: MLP hidden=256 ---
    mlp = MLPProbe(in_dim=xtr.shape[1], hidden=args.mlp_hidden, out_dim=y_train.shape[1], dropout=0.1)
    yhat_mlp, info_mlp = train_torch_probe(
        mlp, xtr, y_train, xva, y_val, device,
        batch_size=args.probe_batch_size, epochs=args.probe_epochs,
        lr=args.probe_lr, weight_decay=args.probe_weight_decay,
        patience=args.probe_patience, label=f"ep{epoch:03d}_mlp{args.mlp_hidden}",
    )
    r2_mlp = r2_per_target(y_val, yhat_mlp)

    for i, name in enumerate(target_names):
        rows.append({
            "epoch": epoch, "probe_type": "linear",
            "target_name": name, "feature": data["target_feature"][i],
            "horizon": data["target_horizon"][i], "r2": float(r2_lin[i]),
        })
        rows.append({
            "epoch": epoch, "probe_type": "mlp",
            "target_name": name, "feature": data["target_feature"][i],
            "horizon": data["target_horizon"][i], "r2": float(r2_mlp[i]),
        })

    diag = {
        "epoch": epoch,
        "linear_best_val_mse": info_lin["best_val_mse"],
        "mlp_best_val_mse": info_mlp["best_val_mse"],
        "r2_linear_mean": float(np.mean(r2_lin)),
        "r2_mlp_mean": float(np.mean(r2_mlp)),
    }
    return rows, diag


# =============================================================================
#  Plotting
# =============================================================================

def _aggregate_r2(long_rows: List[Dict], probe_type: str) -> Dict[int, float]:
    """R^2 medio su tutti i target, per epoca, per un dato probe_type."""
    by_ep: Dict[int, List[float]] = {}
    for r in long_rows:
        if r["probe_type"] != probe_type:
            continue
        by_ep.setdefault(r["epoch"], []).append(r["r2"])
    return {ep: float(np.mean(v)) for ep, v in sorted(by_ep.items())}


def plot_loss_vs_probe(long_rows, train_dyn, out_path: Path) -> None:
    """R^2 aggregato (linear + mlp) vs epoche, con JEPA loss sovrapposta."""
    lin = _aggregate_r2(long_rows, "linear")
    mlp = _aggregate_r2(long_rows, "mlp")
    eps = sorted(lin.keys())
    loss_eps = sorted(train_dyn.keys())
    loss = [train_dyn[e].get("L_total", np.nan) for e in loss_eps]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(eps, [lin[e] for e in eps], "o-", color="#1f77b4", label="probe lineare R² (mean)")
    ax1.plot(eps, [mlp[e] for e in eps], "s-", color="#2ca02c", label="probe MLP R² (mean)")
    ax1.set_xlabel("epoca")
    ax1.set_ylabel("R² medio sui 22 target", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(loss_eps, loss, "x--", color="#d62728", label="JEPA val loss (L_total)")
    ax2.set_ylabel("JEPA val loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Marca l'epoca di picco R^2 lineare.
    if lin:
        best_ep = max(lin, key=lin.get)
        ax1.axvline(best_ep, color="grey", ls=":", alpha=0.7)
        ax1.annotate(f"picco R² lin @ ep{best_ep}", xy=(best_ep, lin[best_ep]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    ax1.set_title("F1-v4: JEPA loss vs accessibilità predittiva del latente")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_horizon(long_rows, out_path: Path) -> None:
    """R^2 lineare per horizon (media sulle feature), vs epoche."""
    horizons = sorted({r["horizon"] for r in long_rows})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for H in horizons:
        by_ep: Dict[int, List[float]] = {}
        for r in long_rows:
            if r["probe_type"] != "linear" or r["horizon"] != H:
                continue
            by_ep.setdefault(r["epoch"], []).append(r["r2"])
        eps = sorted(by_ep.keys())
        ax.plot(eps, [np.mean(by_ep[e]) for e in eps], "o-", label=f"H={H}")
    ax.set_xlabel("epoca")
    ax.set_ylabel("R² lineare (media sulle feature)")
    ax.set_title("F1-v4: R² per horizon lungo il training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_feature(long_rows, out_path: Path) -> None:
    """R^2 lineare per feature (media sugli horizon), vs epoche."""
    features = sorted({r["feature"] for r in long_rows})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for f in features:
        by_ep: Dict[int, List[float]] = {}
        for r in long_rows:
            if r["probe_type"] != "linear" or r["feature"] != f:
                continue
            by_ep.setdefault(r["epoch"], []).append(r["r2"])
        eps = sorted(by_ep.keys())
        ax.plot(eps, [np.mean(by_ep[e]) for e in eps], "o-", label=f)
    ax.set_xlabel("epoca")
    ax.set_ylabel("R² lineare (media sugli horizon)")
    ax.set_title("F1-v4: R² per feature lungo il training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_geometry(long_rows, train_dyn, out_path: Path) -> None:
    """Geometria del latente (pooled_std, eff_rank) vs epoche, con R^2 sovrapp."""
    lin = _aggregate_r2(long_rows, "linear")
    eps_g = sorted(train_dyn.keys())
    pooled_std = [train_dyn[e].get("pooled_std", np.nan) for e in eps_g]
    eff_rank = [train_dyn[e].get("pooled_eff_rank", np.nan) for e in eps_g]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(eps_g, eff_rank, "o-", color="#9467bd", label="effective rank (pooled)")
    ax1.set_xlabel("epoca")
    ax1.set_ylabel("effective rank del latente", color="#9467bd")
    ax1.tick_params(axis="y", labelcolor="#9467bd")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(eps_g, pooled_std, "^--", color="#8c564b", label="pooled std")
    ax1b.set_ylabel("pooled std", color="#8c564b")
    ax1b.tick_params(axis="y", labelcolor="#8c564b")

    # R^2 lineare come riferimento (scala 0..1 sovrapposta a destra spostata).
    if lin:
        eps = sorted(lin.keys())
        ax1b.plot(eps, [lin[e] for e in eps], "x:", color="#1f77b4",
                  label="probe lineare R² (mean)", alpha=0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    ax1.set_title("F1-v4: collasso geometrico del latente vs R² predittivo")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Probe v3: dinamica di training F1-v4 (loss JEPA vs R² predittivo)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory con epoch_XXX.pt (es. checkpoints/jepa_horizon/v1)")
    p.add_argument("--out_dir", default="validation/training_dynamics/v1")

    # Split / target (devono essere COERENTI tra checkpoint).
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--vol_clip", type=float, default=5.0)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--max_train_samples", type=int, default=100000)
    p.add_argument("--max_val_samples", type=int, default=50000)
    p.add_argument("--future_features", type=str,
                   default="d_spread_z,d_microprice_rel,d_best_bid_rel,d_best_ask_rel,d_top_imbalance")
    p.add_argument("--future_horizons", type=str, default="1,5,10,20")
    p.add_argument("--vol_horizons", type=str, default="5,20")

    # Estrazione encoder.
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Probe.
    p.add_argument("--z_dim", type=int, default=32, help="Bottleneck del probe lineare")
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--probe_batch_size", type=int, default=1024)
    p.add_argument("--probe_epochs", type=int, default=80)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_weight_decay", type=float, default=1e-3)
    p.add_argument("--probe_patience", type=int, default=15)

    # Limita il numero di checkpoint processati (debug).
    p.add_argument("--max_checkpoints", type=int, default=0,
                   help="0 = tutti; altrimenti processa solo i primi N")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("PROBE v3 — DINAMICA DI TRAINING F1-v4")
    print("=" * 92)
    print(f"dataset   : {args.dataset}")
    print(f"ckpt_dir  : {ckpt_dir}")
    print(f"out_dir   : {out_dir}")
    print(f"device    : {device}")

    checkpoints = discover_checkpoints(ckpt_dir)
    if not checkpoints:
        raise SystemExit(f"Nessun epoch_*.pt trovato in {ckpt_dir}")
    if args.max_checkpoints and args.max_checkpoints > 0:
        checkpoints = checkpoints[:args.max_checkpoints]
    print(f"checkpoint trovati: {len(checkpoints)} "
          f"(epoche {checkpoints[0][0]}..{checkpoints[-1][0]})")

    # ----- Setup fisso (una volta) -----
    data = build_fixed_probe_data(args, device)
    stock_stats = resolve_stock_stats(checkpoints[0][1], data, args, device)

    # ----- Loop sui checkpoint -----
    long_rows: List[Dict] = []
    train_dyn: Dict[int, Dict] = {}
    diags: List[Dict] = []

    t_start = time.time()
    for i, (epoch, path) in enumerate(checkpoints):
        print("\n" + "-" * 92)
        print(f"[{i+1}/{len(checkpoints)}] checkpoint epoch {epoch}  ({path.name})")
        print("-" * 92)
        t0 = time.time()

        # Curve di loss / geometria: lette da val_metrics dentro il checkpoint.
        # Robusto anche se history.json non esiste (training interrotto).
        train_dyn[epoch] = read_val_metrics_from_ckpt(path, device)

        rows, diag = probe_one_checkpoint(path, epoch, data, stock_stats, args, device)
        long_rows.extend(rows)
        diags.append(diag)
        print(f"  epoch {epoch}: R²_lin(mean)={diag['r2_linear_mean']:+.4f}  "
              f"R²_mlp(mean)={diag['r2_mlp_mean']:+.4f}  ({time.time()-t0:.1f}s)")

    total_dt = time.time() - t_start
    print(f"\nProbe completato su {len(checkpoints)} checkpoint in {total_dt/60:.1f} min")

    # ----- Scrittura CSV -----
    long_csv = out_dir / "probe_dynamics_long.csv"
    with open(long_csv, "w") as f:
        f.write("epoch,probe_type,target_name,feature,horizon,r2\n")
        for r in long_rows:
            f.write(f"{r['epoch']},{r['probe_type']},{r['target_name']},"
                    f"{r['feature']},{r['horizon']},{r['r2']:.6f}\n")
    print(f"  scritto {long_csv}")

    # training_dynamics.csv: colonne dinamiche per le L_H{H} presenti.
    all_loss_keys = set()
    for vm in train_dyn.values():
        all_loss_keys.update(k for k in vm.keys()
                             if k.startswith("L_H") or k in ("L_total",))
    loss_cols = ["L_total"] + sorted(
        [k for k in all_loss_keys if k.startswith("L_H")],
        key=lambda k: int(k[3:]) if k[3:].isdigit() else 1_000_000,
    )
    geom_cols = ["pooled_std", "pooled_eff_rank", "online_norm",
                 "target_norm_H0", "cos_online_target_H0", "gap_norm_H0"]
    train_csv = out_dir / "training_dynamics.csv"
    with open(train_csv, "w") as f:
        header = ["epoch"] + loss_cols + geom_cols
        f.write(",".join(header) + "\n")
        for ep in sorted(train_dyn.keys()):
            vm = train_dyn[ep]
            vals = [str(ep)]
            for c in loss_cols + geom_cols:
                v = vm.get(c, "")
                vals.append(f"{v:.6f}" if isinstance(v, (int, float)) else "")
            f.write(",".join(vals) + "\n")
    print(f"  scritto {train_csv}")

    # ----- Plot -----
    plot_loss_vs_probe(long_rows, train_dyn, out_dir / "plot_loss_vs_probe.png")
    plot_per_horizon(long_rows, out_dir / "plot_per_horizon.png")
    plot_per_feature(long_rows, out_dir / "plot_per_feature.png")
    plot_geometry(long_rows, train_dyn, out_dir / "plot_geometry.png")
    print(f"  scritti 4 plot in {out_dir}")

    # ----- Summary diagnostico -----
    lin_agg = _aggregate_r2(long_rows, "linear")
    mlp_agg = _aggregate_r2(long_rows, "mlp")
    best_lin_ep = max(lin_agg, key=lin_agg.get) if lin_agg else None
    best_mlp_ep = max(mlp_agg, key=mlp_agg.get) if mlp_agg else None
    last_ep = max(lin_agg.keys()) if lin_agg else None

    summary = {
        "n_checkpoints": len(checkpoints),
        "epoch_range": [checkpoints[0][0], checkpoints[-1][0]],
        "best_linear_epoch": best_lin_ep,
        "best_linear_r2": lin_agg.get(best_lin_ep) if best_lin_ep else None,
        "best_mlp_epoch": best_mlp_ep,
        "best_mlp_r2": mlp_agg.get(best_mlp_ep) if best_mlp_ep else None,
        "final_linear_r2": lin_agg.get(last_ep) if last_ep else None,
        "final_mlp_r2": mlp_agg.get(last_ep) if last_ep else None,
        "linear_r2_drop_peak_to_final": (
            lin_agg[best_lin_ep] - lin_agg[last_ep]
            if (best_lin_ep is not None and last_ep is not None) else None
        ),
        "args": vars(args),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    if best_lin_ep is not None:
        print(f"  picco R² lineare : {lin_agg[best_lin_ep]:+.4f} @ epoca {best_lin_ep}")
        print(f"  R² lineare finale: {lin_agg[last_ep]:+.4f} @ epoca {last_ep}")
        drop = summary["linear_r2_drop_peak_to_final"]
        print(f"  degrado picco->fine: {drop:+.4f}")
        if drop is not None and drop > 0.01:
            print("  --> fenomeno loss-vs-probe CONFERMATO: R² degrada dopo il picco.")
        else:
            print("  --> nessun degrado netto del probe rilevato.")
    print(f"\nOutput in: {out_dir}")


if __name__ == "__main__":
    main()

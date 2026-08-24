#!/usr/bin/env python3
"""
screen_heldout_gate1.py — Gate 1 of the held-out control (dataset-only, no encoder).

Purpose
=======
The scissor result is descriptive: JEPA buries the DIRECTIONAL signal in
low-variance / nonlinear directions; supervised exposes it. It does NOT yet say
WHY supervised is accessible -- genuine geometry (A) or co-adaptation to the 14
trained targets (B). Only held-out targets (which NEITHER arm optimized) can
separate A from B: on held-out, JEPA is unchanged (never co-adapted to anything),
and supervised's accessibility either holds (A) or collapses toward JEPA (B).

But held-out targets must first be ADMISSIBLE. Gate 1 (this script) tests the
independence condition, in TARGET space only -- no representations, no checkpoints:

    a candidate h is admissible iff R^2(h | 14 trained targets) is LOW.

If a candidate is well-predicted by a linear combination of the trained targets,
it is "already seen" (like d_spread_z = d_best_ask_rel - d_best_bid_rel) and a
linear probe reads it for free -> a fake held-out. We keep only low-R^2 candidates.
(Condition 2, non-trivial MLP ceiling in BOTH representations, is Gate 2 and needs
the extracted readouts.)

Candidates (all derivable from the same snapshots, no trade prints needed):
  depth/shape deltas the trained targets ignore (top-of-book only):
    d_imbalance_top5, d_imbalance_all, d_log_depth_top5, d_log_depth_all,
    d_depth_width_z         (over horizons {1,5,10,20})
  a timing target of a different KIND:
    time_to_next_mid_move   (steps until mid_z changes; 'any movement')

Trained targets regressed against (the 14 independent):
  directional: d_spread_z, d_microprice_rel, d_top_imbalance  x {1,5,10,20}  (12)
  (d_best_bid_rel/d_best_ask_rel are +-1/2 * d_spread_z -> excluded, redundant)
  volatility : realized_vol x {5,20}                                          (2)

Output: a table  candidate@h -> R^2(.|14 trained)  + PASS/FAIL, plus the
mid-move distribution pre-check (flags a degenerate timing target).
"""

from __future__ import annotations
import argparse, importlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

from analysis_artifacts import (
    SPLIT_SCHEMA_NAME,
    SPLIT_SCHEMA_VERSION,
    TARGET_SCHEMA_NAME,
    TARGET_VERSION,
    atomic_savez,
    atomic_write_json,
    build_endpoint_split,
    canonical_sha256,
    endpoint_sha256,
    load_split,
    sha256_file,
)

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, *_THIS.parents, Path.cwd(), *Path.cwd().parents]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _imp(mod):
    for cand in (f"training.{mod}", mod):
        try:
            return importlib.import_module(cand)
        except Exception:
            continue
    raise SystemExit(f"Cannot import {mod!r}; run from the project root.")


_tok = _imp("train_tokenizer_t")
derive_raw_features_array = _tok.derive_raw_features_array
compute_future_feature_targets = _tok.compute_future_feature_targets
compute_vol_targets = _tok.compute_vol_targets
compute_valid_endpoints = _tok.compute_valid_endpoints
grouped_split_by_stock_day = _tok.grouped_split_by_stock_day

# trained set (independent 14): 3 dir features x 4 horizons + 2 vol
TRAINED_DIR = ["d_spread_z", "d_microprice_rel", "d_top_imbalance"]
TRAINED_DIR_FEATS = ["spread_z", "microprice_rel", "top_imbalance"]
HORIZONS = [1, 5, 10, 20]
VOL_HORIZONS = [5, 20]
VOL_CLIP = 5.0
TIMING_MAX_LOOK = 600
TIMING_TARGET_POLICY = "log1p_observed_or_capped_all_rows"
# held-out candidates (feature names in RAW_FEATURE_NAMES; deltas taken here)
CAND_FEATS = ["imbalance_top5", "imbalance_all", "log_depth_top5",
              "log_depth_all", "depth_width_z"]


def r2_oos(y_true, y_pred):
    resid = ((y_true - y_pred) ** 2).sum(0)
    tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return 1.0 - resid / np.maximum(tot, 1e-12)


def ols_r2(Xtr, Ytr, Xva, Yva):
    """R^2 on val of Y ~ [1, X], fit on train. Y can be (n,) or (n,k)."""
    Xtr1 = np.concatenate([np.ones((Xtr.shape[0], 1)), Xtr], 1)
    Xva1 = np.concatenate([np.ones((Xva.shape[0], 1)), Xva], 1)
    W, *_ = np.linalg.lstsq(Xtr1, Ytr, rcond=None)
    return r2_oos(Yva, Xva1 @ W)


def time_to_next_mid_move(
    mid_z, stock_ids, day_ids, valid_t, max_look=TIMING_MAX_LOOK
):
    """Steps forward until mid_z changes, within the same stock-day.
    'any movement' = exact change of the tick-grid mid_z. Right-censored at day
    or stock boundary, or at max_look. Returns (durations, censored_mask)
    aligned to valid_t."""
    dur = np.full(len(valid_t), -1, dtype=np.int32)
    cens = np.zeros(len(valid_t), dtype=bool)
    N = len(mid_z)
    for i, t in enumerate(valid_t):
        t = int(t); s = stock_ids[t]; d = day_ids[t]; base = mid_z[t]
        k = 1
        while (t + k < N and stock_ids[t + k] == s
               and day_ids[t + k] == d and k <= max_look):
            if mid_z[t + k] != base:
                dur[i] = k
                break
            k += 1
        if dur[i] == -1:
            cens[i] = True
            # ``k`` becomes max_look + 1 when the search exhausts its horizon.
            # The stored target is an observed-or-capped duration, so never
            # leak that loop sentinel into the target.
            dur[i] = min(k, max_look)
    return dur, cens


def timing_target(durations):
    """Canonical timing target used by both Gate 1 and the saved artifact."""
    return np.log1p(np.asarray(durations, dtype=np.float64))


def main():
    ap = argparse.ArgumentParser(description="Gate 1: held-out independence screening")
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--split", "--subsample", dest="split_path", default=None,
        help="path to the canonical stage-1 split artifact (the legacy "
             "--subsample spelling remains an alias). If omitted, rebuilds "
             "valid_t + grouped split + subsample.",
    )
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--max_h", type=int, default=20)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=100_000)
    ap.add_argument("--n_val", type=int, default=50_000)
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--pass_threshold", type=float, default=0.30,
                    help="candidate PASSES Gate 1 if R^2(.|14 trained) < this")
    ap.add_argument("--save_heldout", default=None,
                    help="if set, write PASSING held-out targets (excluding the "
                         "numerically fragile depth_width_z) to this .npz, aligned "
                         "to the subsample, for Gate 2.")
    ap.add_argument("--exclude", default="depth_width_z",
                    help="comma list of feature stems to exclude from the saved set")
    args = ap.parse_args()

    dataset_sha256 = sha256_file(args.dataset)
    raw = np.load(args.dataset)
    book = raw["book"]; mid_z = raw["mid_z"]
    stock_ids = raw["stock_ids"].astype(np.int64); day_ids = raw["day_ids"].astype(np.int64)
    min_spread = raw["min_spread_z_per_stock"]
    n_stocks = int(stock_ids.max()) + 1

    if args.split_path:
        split = load_split(
            args.split_path,
            expected_dataset_sha256=dataset_sha256,
        )
        # A canonical split stores raw endpoints explicitly. load_split rejects
        # legacy position-only files and validates their endpoint hashes.
        train_t = np.asarray(split.train_t, dtype=np.int64)
        val_t = np.asarray(split.val_t, dtype=np.int64)
        print(f"[reuse] canonical split from {args.split_path}: "
              f"train={len(train_t):,} val={len(val_t):,}")
    else:
        bid_v, ask_v = book[:, 0, :, 1], book[:, 1, :, 1]
        vmask = (np.abs(bid_v).max(1) <= VOL_CLIP) & (np.abs(ask_v).max(1) <= VOL_CLIP)
        split = build_endpoint_split(
            stock_ids,
            day_ids,
            vmask,
            K=args.K,
            max_horizon=args.max_h,
            val_frac=args.val_frac,
            split_seed=args.split_seed,
            n_train=args.n_train,
            n_val=args.n_val,
            subsample_seed=args.subsample_seed,
            compute_valid_endpoints_fn=compute_valid_endpoints,
            grouped_split_fn=grouped_split_by_stock_day,
        )
        # grouped_split_by_stock_day returns positions in valid_t. Make the
        # position -> raw endpoint conversion explicit at this boundary.
        valid_t = np.asarray(split.valid_t, dtype=np.int64)
        train_pos = np.asarray(split.train_pos, dtype=np.int64)
        val_pos = np.asarray(split.val_pos, dtype=np.int64)
        train_t = valid_t[train_pos]
        val_t = valid_t[val_pos]
        if (not np.array_equal(train_t, split.train_t)
                or not np.array_equal(val_t, split.val_t)):
            raise RuntimeError("split artifact endpoint mapping is inconsistent")
        split.bind(dataset_sha256=dataset_sha256, config=split.config)
        print(f"[build] train={len(train_t):,} val={len(val_t):,}")

    raw_feat, _ = derive_raw_features_array(book, mid_z, stock_ids, n_stocks)

    def trained_matrix(endpoints):
        dirm = compute_future_feature_targets(
            raw_feat, endpoints, TRAINED_DIR_FEATS, HORIZONS
        )
        vol = compute_vol_targets(
            mid_z, endpoints, VOL_HORIZONS, min_spread, stock_ids
        )
        return np.concatenate([dirm, vol], 1).astype(np.float64)   # (n, 14)

    Xtr, Xva = trained_matrix(train_t), trained_matrix(val_t)
    print(f"[trained] regressor matrix: {Xtr.shape[1]} independent targets")

    # candidate depth/shape deltas
    cand_names, cand_tr, cand_va = [], [], []
    for f in CAND_FEATS:
        ctr = compute_future_feature_targets(raw_feat, train_t, [f], HORIZONS)
        cva = compute_future_feature_targets(raw_feat, val_t, [f], HORIZONS)
        for j, h in enumerate(HORIZONS):
            cand_names.append(f"d_{f}@{h}"); cand_tr.append(ctr[:, j]); cand_va.append(cva[:, j])

    # timing candidate (its own pre-check below)
    dtr, ctr_c = time_to_next_mid_move(
        mid_z, stock_ids, day_ids, train_t, max_look=TIMING_MAX_LOOK
    )
    dva, cva_c = time_to_next_mid_move(
        mid_z, stock_ids, day_ids, val_t, max_look=TIMING_MAX_LOOK
    )
    # The estimand is the observed-or-capped log duration. Censored rows are
    # therefore part of both Gate 1 and the persisted target, with identical
    # row support and semantics.
    ytr_time = timing_target(dtr)
    yva_time = timing_target(dva)

    print("\n" + "=" * 66)
    print("GATE 1 - independence of candidates from the 14 trained targets")
    print(f"(PASS if R^2(candidate | 14 trained) < {args.pass_threshold})")
    print("=" * 66)
    print(f'{"candidate":<24}{"R2(.|14)":>12}   verdict')
    print("-" * 66)
    results = []
    for nm, ytr, yva in zip(cand_names, cand_tr, cand_va):
        r2 = float(ols_r2(Xtr, ytr.astype(np.float64), Xva, yva.astype(np.float64)))
        verdict = "PASS" if r2 < args.pass_threshold else "fail (redundant)"
        results.append((nm, r2, verdict))
        print(f'{nm:<24}{r2:>12.4f}   {verdict}')

    # timing candidate
    r2t = float(ols_r2(Xtr, ytr_time, Xva, yva_time))
    tv = "PASS" if r2t < args.pass_threshold else "fail (redundant)"
    print(f'{"time_to_next_mid_move":<24}{r2t:>12.4f}   {tv}')

    print("\n" + "-" * 66)
    print("mid-move distribution pre-check (degeneracy guard for the timing target)")
    alld = np.concatenate([dtr, dva]); allc = np.concatenate([ctr_c, cva_c])
    frac_move1 = float((alld[~allc] == 1).mean()) if (~allc).any() else float("nan")
    print(f"  censored (no move within day/max_look): {allc.mean()*100:.1f}%")
    print(f"  P(move at next step, i.e. duration==1):  {frac_move1*100:.1f}%")
    print(f"  duration percentiles (non-censored): "
          f"p50={np.percentile(alld[~allc],50):.0f}  "
          f"p90={np.percentile(alld[~allc],90):.0f}  "
          f"p99={np.percentile(alld[~allc],99):.0f}")
    if frac_move1 > 0.85:
        print("  [WARN] mid moves almost every step -> target near-degenerate (drop it).")
    print("=" * 66)
    passed = [nm for nm, r2, v in results if v == "PASS"] + (["time_to_next_mid_move"] if tv == "PASS" else [])
    print(f"\nPASS Gate 1 ({len(passed)}): {passed}")
    print("-> these go to Gate 2 (MLP-ceiling in BOTH reps) on the extracted readouts.")

    if args.save_heldout:
        excl = set(x.strip() for x in args.exclude.split(",") if x.strip())
        pass_set = {nm for nm, r2, v in results if v == "PASS"}
        # rebuild passing depth/imbalance columns for ALL rows (aligned to subsample),
        # and the timing target for ALL rows (censored included, capped) so it aligns
        # row-for-row with the extracted readouts (which have no dropped rows).
        keep_names, ytr_cols, yva_cols = [], [], []
        for f in CAND_FEATS:
            if f in excl:
                continue
            ctr = compute_future_feature_targets(raw_feat, train_t, [f], HORIZONS)
            cva = compute_future_feature_targets(raw_feat, val_t, [f], HORIZONS)
            for j, h in enumerate(HORIZONS):
                nm = f"d_{f}@{h}"
                if nm in pass_set:
                    keep_names.append(nm); ytr_cols.append(ctr[:, j]); yva_cols.append(cva[:, j])
        if tv == "PASS":
            keep_names.append("time_to_next_mid_move")
            ytr_cols.append(timing_target(dtr))
            yva_cols.append(timing_target(dva))
        y_train_ho = (
            np.stack(ytr_cols, 1).astype(np.float32)
            if ytr_cols else np.empty((len(train_t), 0), dtype=np.float32)
        )
        y_val_ho = (
            np.stack(yva_cols, 1).astype(np.float32)
            if yva_cols else np.empty((len(val_t), 0), dtype=np.float32)
        )
        train_hash = endpoint_sha256(train_t)
        val_hash = endpoint_sha256(val_t)
        gate1_source_sha256 = sha256_file(_THIS)
        artifact_fingerprint = canonical_sha256(
            {
                "schema": TARGET_SCHEMA_NAME,
                "version": TARGET_VERSION,
                "artifact_kind": "heldout_targets",
                "source_sha256": gate1_source_sha256,
                "dataset_sha256": dataset_sha256,
                "split_fingerprint": split.split_fingerprint,
                "train_endpoint_sha256": train_hash,
                "val_endpoint_sha256": val_hash,
                "pass_threshold": args.pass_threshold,
                "excluded_feature_stems": sorted(excl),
                "heldout_names": keep_names,
                "timing_target_policy": TIMING_TARGET_POLICY,
                "timing_max_look": TIMING_MAX_LOOK,
            }
        )
        heldout_path = Path(args.save_heldout)
        manifest_path = None
        manifest = None
        if args.split_path:
            candidate = (
                Path(args.split_path).resolve().parent / "analysis_manifest.json"
            )
            if candidate.is_file():
                with candidate.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                if (
                    manifest.get("dataset", {}).get("sha256") != dataset_sha256
                    or manifest.get("split", {}).get("fingerprint")
                    != split.split_fingerprint
                ):
                    raise RuntimeError(
                        "refusing to overwrite held-out targets for an "
                        "incompatible manifest"
                    )
                manifest_path = candidate
        atomic_savez(
            heldout_path,
            schema_name=np.asarray(TARGET_SCHEMA_NAME),
            schema_version=np.asarray(TARGET_VERSION, dtype=np.int64),
            artifact_kind=np.asarray("heldout_targets"),
            artifact_fingerprint=np.asarray(artifact_fingerprint),
            source_sha256=np.asarray(gate1_source_sha256),
            split_schema_name=np.asarray(SPLIT_SCHEMA_NAME),
            split_schema_version=np.asarray(
                SPLIT_SCHEMA_VERSION, dtype=np.int64
            ),
            split_fingerprint=np.asarray(split.split_fingerprint),
            dataset_sha256=np.asarray(dataset_sha256),
            train_endpoint_sha256=np.asarray(train_hash),
            val_endpoint_sha256=np.asarray(val_hash),
            n_train=np.asarray(len(train_t), dtype=np.int64),
            n_val=np.asarray(len(val_t), dtype=np.int64),
            y_train_heldout=y_train_ho,
            y_val_heldout=y_val_ho,
            heldout_names=np.asarray(keep_names),
        )
        if manifest_path is not None and manifest is not None:
            try:
                recorded_path = str(
                    heldout_path.resolve().relative_to(manifest_path.parent)
                )
            except ValueError:
                recorded_path = str(heldout_path.resolve())
            manifest["heldout_targets"] = {
                "path": recorded_path,
                "file_sha256": sha256_file(heldout_path),
                "size_bytes": heldout_path.stat().st_size,
                "artifact_fingerprint": artifact_fingerprint,
                "source": {
                    "path": str(_THIS.relative_to(_THIS.parent)),
                    "sha256": gate1_source_sha256,
                    "size_bytes": _THIS.stat().st_size,
                },
                "config": {
                    "pass_threshold": args.pass_threshold,
                    "excluded_feature_stems": sorted(excl),
                    "timing_target_policy": TIMING_TARGET_POLICY,
                    "timing_max_look": TIMING_MAX_LOOK,
                },
                "shape_train": list(y_train_ho.shape),
                "shape_val": list(y_val_ho.shape),
                "train_endpoint_sha256": train_hash,
                "val_endpoint_sha256": val_hash,
            }
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(manifest_path, manifest)
        print(f"\n[saved] {len(keep_names)} held-out targets -> {args.save_heldout}")
        print(f"        {keep_names}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ladder_accessibility.py — STAGE 2 of the accessibility-ladder battery.

Consumes the STAGE-1 dumps (extract_readouts_multiseed.py) and produces the
accessibility curves R^2(m): how much target signal a linear reader recovers
from the top-m PCA directions (variance-ordered) of a frozen representation.

Core idea — the claim is literally "signal lives in low-variance directions",
so we order directions by VARIANCE (PCA) and read the top-m with a min-norm OLS
that has ZERO shrinkage. No weight decay (which would penalize low-variance
reads), no learned bottleneck, linearity held fixed. Nonlinearity enters only
as a horizontal reference (MLP ceiling); ridge is a soft-truncation cross-check.

Pipeline (all on the fixed 100k/50k subsample, split_seed=0):
  1. Harmonize targets via SVD of the RAW directional block. The algebraic
     redundancy  d_spread_z@h = d_best_ask_rel@h - d_best_bid_rel@h  shows up as
     a near-zero singular-value cliff; we report the spectrum and cut on the gap,
     and drop the derived d_spread_z columns for the AGGREGATE only (per-target
     always keeps all). Directional and vol are reported SEPARATELY.
  2. For each dump and each pooling (last_concat512, tmean_concat512):
       PCA on train (centered, NOT per-dim standardized -> preserves variance
       ordering), project train/val to top-m, min-norm OLS, R^2 per target on val.
  3. Ridge sweep (numpy) and MLP ceiling (torch/GPU) as horizontal references.
  4. Aggregate over seeds -> mean/std bands. Write tidy CSVs + one headline plot.

Outputs (out_dir/analysis/):
  harmonization.json      spectrum, gap-rank, dropped columns
  ladder_long.csv         arm,seed,epoch,pooling,target,m,r2
  refs_long.csv           arm,seed,epoch,pooling,ref(mlp|ridge),lambda,target,r2
  ladder_agg.csv          arm,epoch,pooling,block(dir|vol),m,r2_mean,r2_std,n_seeds
  plot_ladder_headline.png
"""

from __future__ import annotations
import argparse, json, re, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from analysis_artifacts import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    READOUT_SCHEMA,
    READOUT_VERSION,
    SPLIT_SCHEMA,
    SPLIT_VERSION,
    TARGET_SCHEMA,
    TARGET_VERSION,
    atomic_write_json,
    load_split,
    sha256_file,
)
from consolidation_geometry import (
    DEFAULT_POOLINGS,
    derive_pooling,
    hadamard_mean_basis,
    haar_bases,
    ladder_from_stats,
    linear_stats,
    pca_from_stats,
    principal_angle_curve,
    random_subspace_null,
    schedule_for_dimension,
    spectral_diagnostics,
)

# Target layout — MUST match STAGE 1.
FUTURE_FEATURES = ["d_spread_z", "d_microprice_rel", "d_best_bid_rel",
                   "d_best_ask_rel", "d_top_imbalance"]
FUTURE_HORIZONS = [1, 5, 10, 20]
VOL_HORIZONS = [5, 20]
N_DIR = len(FUTURE_FEATURES) * len(FUTURE_HORIZONS)   # 20
SCHEDULE_DEFAULT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
POOLINGS = DEFAULT_POOLINGS


# ------------------------------------------------------------------ utilities
def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Out-of-sample R^2 per column (val mean as baseline). Scale-free."""
    resid = ((y_true - y_pred) ** 2).sum(axis=0)
    tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - resid / np.maximum(tot, 1e-12)


def block_of(name: str) -> str:
    """Map a target name to its reporting block. Held-out families are separate."""
    if name.startswith("realized_vol@"):
        return "vol"
    if name.startswith("d_imbalance_"):
        return "ho_imbalance"
    if name.startswith("d_log_depth_"):
        return "ho_depth"
    if name == "time_to_next_mid_move":
        return "ho_timing"
    if name.startswith("d_best_bid_rel@") or name.startswith("d_best_ask_rel@"):
        return "drop"                       # redundant +-1/2 spread copies
    if name.startswith("d_"):
        return "dir"                        # trained directional (spread, micro, top_imb)
    return "drop"


def dir_indices() -> Dict[str, List[int]]:
    """Column indices in the 22-target vector."""
    names = [f"{f}@{h}" for f in FUTURE_FEATURES for h in FUTURE_HORIZONS]
    names += [f"realized_vol@{h}" for h in VOL_HORIZONS]
    # mid = (best_bid + best_ask)/2  =>  best_bid_rel = -spread/2, best_ask_rel = +spread/2.
    # So {spread, bid_rel, ask_rel} are ONE quantity (rank 1). For the AGGREGATE we
    # keep d_spread_z and drop the two _rel copies (else spread is triple-counted).
    # d_microprice_rel is KEPT (independent, low-variance); harmonize() reports the
    # spectrum so we can see whether it is truly redundant or merely near-collinear.
    redundant_cols = [i for i, n in enumerate(names)
                      if n.startswith("d_best_bid_rel@") or n.startswith("d_best_ask_rel@")]
    dir_all = list(range(N_DIR))
    dir_indep = [i for i in dir_all if i not in redundant_cols]  # 12 independent
    vol_cols = list(range(N_DIR, N_DIR + len(VOL_HORIZONS)))
    return {"names": names, "dir_all": dir_all, "dir_indep": dir_indep,
            "redundant_cols": redundant_cols, "vol_cols": vol_cols}


# --------------------------------------------------------- stage-1 input preflight
def _fail(message: str) -> None:
    raise RuntimeError(f"stage-1 input integrity failure: {message}")


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(data[key])
    if value.ndim != 0:
        _fail(f"{key} must be scalar, got {value.shape}")
    return value.item()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"manifest field {label} must be an object")
    return value


def _validate_recorded_file(
    path: Path, record: Mapping[str, Any], label: str
) -> None:
    if not path.is_file():
        _fail(f"{label} file is missing: {path}")
    if record.get("size_bytes") != path.stat().st_size:
        _fail(f"{label} size differs from manifest")
    expected_sha = record.get("file_sha256")
    if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
        _fail(f"{label} SHA-256 differs from manifest")


def _npz_specs(path: Path, keys: List[str]) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
    """Read NPY headers inside an NPZ without materializing the large arrays."""
    specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            for key in keys:
                member = f"{key}.npy"
                if member not in members:
                    _fail(f"{path.name} is missing array {key}")
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        _fail(
                            f"{path.name}:{key} uses unsupported NPY header "
                            f"version {version}"
                        )
                specs[key] = (tuple(shape), np.dtype(dtype))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        _fail(f"cannot inspect {path}: {exc}")
    return specs


def _manifest_path(in_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"manifest {label}.path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"manifest {label}.path must be relative to --in_dir")
    return in_dir / relative


def validate_stage1_inputs(
    in_dir: Path, heldout_arg: Optional[str] = None
) -> Dict[str, Any]:
    """Fail closed before Stage 2 consumes any Stage-1 rows."""
    manifest_path = in_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        _fail(f"missing manifest: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        _fail(f"cannot read manifest: {exc}")
    if not isinstance(manifest, dict):
        _fail("manifest root must be an object")
    if manifest.get("schema") != {
        "name": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
    }:
        _fail("manifest schema is not the supported v2 schema")
    if manifest.get("status") != "complete":
        _fail(f"manifest status is {manifest.get('status')!r}, not 'complete'")

    dataset = _require_mapping(manifest.get("dataset"), "dataset")
    dataset_sha = dataset.get("sha256")
    if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        _fail("manifest dataset SHA-256 is missing or malformed")
    protocol = _require_mapping(manifest.get("protocol"), "protocol")
    protocol_fingerprint = protocol.get("fingerprint")
    if not isinstance(protocol_fingerprint, str) or not protocol_fingerprint:
        _fail("manifest protocol fingerprint is missing")

    split_record = _require_mapping(manifest.get("split"), "split")
    split_path = _manifest_path(in_dir, split_record.get("path"), "split")
    if split_path != in_dir / "split.npz":
        _fail("canonical split path must be split.npz")
    _validate_recorded_file(split_path, split_record, "split")
    try:
        split = load_split(split_path, expected_dataset_sha256=dataset_sha)
    except (OSError, ValueError, KeyError) as exc:
        _fail(f"invalid canonical split: {exc}")
    if split_record.get("fingerprint") != split.split_fingerprint:
        _fail("split fingerprint differs from manifest")
    split_expectations = {
        "n_valid_total": len(split.valid_t),
        "n_train": len(split.train_t),
        "n_val": len(split.val_t),
        "train_endpoint_sha256": split.train_endpoint_sha256,
        "val_endpoint_sha256": split.val_endpoint_sha256,
    }
    for key, expected in split_expectations.items():
        if split_record.get(key) != expected:
            _fail(f"manifest split field {key} is inconsistent")

    target_record = _require_mapping(manifest.get("targets"), "targets")
    target_path = _manifest_path(in_dir, target_record.get("path"), "targets")
    if target_path != in_dir / "targets_shared.npz":
        _fail("canonical trained-target path must be targets_shared.npz")
    _validate_recorded_file(target_path, target_record, "trained targets")
    target_required = {
        "schema_name", "schema_version", "artifact_fingerprint",
        "protocol_fingerprint", "split_fingerprint", "dataset_sha256",
        "train_endpoint_sha256", "val_endpoint_sha256", "n_train", "n_val",
        "y_train_raw", "y_val_raw", "target_names",
    }
    try:
        with np.load(target_path, allow_pickle=False) as targets:
            if set(targets.files) != target_required:
                _fail("trained-target NPZ key set differs from schema")
            target_metadata = {
                "schema_name": TARGET_SCHEMA,
                "schema_version": TARGET_VERSION,
                "protocol_fingerprint": protocol_fingerprint,
                "split_fingerprint": split.split_fingerprint,
                "dataset_sha256": dataset_sha,
                "train_endpoint_sha256": split.train_endpoint_sha256,
                "val_endpoint_sha256": split.val_endpoint_sha256,
                "n_train": len(split.train_t),
                "n_val": len(split.val_t),
            }
            for key, expected in target_metadata.items():
                if _scalar(targets, key) != expected:
                    _fail(f"trained-target metadata mismatch: {key}")
            trained_names = [str(value) for value in targets["target_names"].tolist()]
            expected_names = dir_indices()["names"]
            if trained_names != expected_names:
                _fail("trained target_names do not match the canonical order")
            target_artifact_fingerprint = str(
                _scalar(targets, "artifact_fingerprint")
            )
    except (OSError, ValueError, KeyError) as exc:
        _fail(f"cannot validate trained targets: {exc}")
    target_specs = _npz_specs(
        target_path, ["y_train_raw", "y_val_raw", "target_names"]
    )
    if target_specs["y_train_raw"] != (
        (len(split.train_t), len(expected_names)),
        np.dtype(np.float32),
    ):
        _fail("trained-target train array shape/dtype is invalid")
    if target_specs["y_val_raw"] != (
        (len(split.val_t), len(expected_names)),
        np.dtype(np.float32),
    ):
        _fail("trained-target validation array shape/dtype is invalid")
    if target_specs["target_names"][0] != (len(expected_names),):
        _fail("trained target_names shape is invalid")
    target_record_expectations = {
        "artifact_fingerprint": target_artifact_fingerprint,
        "shape_train": [len(split.train_t), len(expected_names)],
        "shape_val": [len(split.val_t), len(expected_names)],
        "train_endpoint_sha256": split.train_endpoint_sha256,
        "val_endpoint_sha256": split.val_endpoint_sha256,
    }
    for key, expected in target_record_expectations.items():
        if target_record.get(key) != expected:
            _fail(f"manifest trained-target field {key} is inconsistent")

    requested = _require_mapping(
        manifest.get("requested_checkpoints"), "requested_checkpoints"
    )
    readout_records = _require_mapping(manifest.get("readouts"), "readouts")
    if not requested:
        _fail("manifest requests no checkpoints")
    if set(requested) != set(readout_records):
        _fail("requested checkpoint tags and readout manifest tags differ")
    summary = _require_mapping(manifest.get("summary"), "summary")
    if (
        summary.get("expected") != len(requested)
        or summary.get("complete") != len(requested)
        or summary.get("missing") != []
    ):
        _fail("manifest summary does not describe an exact complete inventory")

    readout_dir = in_dir / "readouts"
    if not readout_dir.is_dir():
        _fail("readouts directory is missing")
    expected_relative_paths = {
        str(Path("readouts") / f"{tag}.npz") for tag in requested
    }
    disk_relative_paths = {
        str(path.relative_to(in_dir))
        for path in readout_dir.iterdir()
        if path.is_file()
    }
    if disk_relative_paths != expected_relative_paths:
        _fail("readout files on disk do not match the manifest inventory exactly")
    if any(path.is_dir() for path in readout_dir.iterdir()):
        _fail("unexpected subdirectory in readouts inventory")

    readout_paths: List[Path] = []
    readout_arrays = (
        "last_concat512_train", "last_concat512_val",
        "tmean_concat512_train", "tmean_concat512_val",
    )
    readout_metadata_keys = {
        "schema_name", "schema_version", "artifact_fingerprint",
        "protocol_fingerprint", "split_fingerprint", "dataset_sha256",
        "train_endpoint_sha256", "val_endpoint_sha256", "checkpoint_sha256",
        "stock_stats_sha256", "arm", "seed", "epoch", "n_train", "n_val",
    }
    expected_array_record = {
        "last_concat512_train": {
            "shape": [len(split.train_t), 512], "dtype": "float32"
        },
        "last_concat512_val": {
            "shape": [len(split.val_t), 512], "dtype": "float32"
        },
        "tmean_concat512_train": {
            "shape": [len(split.train_t), 512], "dtype": "float32"
        },
        "tmean_concat512_val": {
            "shape": [len(split.val_t), 512], "dtype": "float32"
        },
    }
    for tag in sorted(requested):
        checkpoint = _require_mapping(requested[tag], f"requested_checkpoints.{tag}")
        record = _require_mapping(readout_records[tag], f"readouts.{tag}")
        parsed_tag = parse_tag(f"{tag}.npz")
        expected_tag = (
            checkpoint.get("arm"),
            checkpoint.get("seed"),
            checkpoint.get("epoch"),
        )
        if parsed_tag != expected_tag:
            _fail(f"{tag}: tag does not match checkpoint arm/seed/epoch")
        expected_relative = str(Path("readouts") / f"{tag}.npz")
        if record.get("path") != expected_relative:
            _fail(f"{tag}: non-canonical readout path")
        path = in_dir / expected_relative
        _validate_recorded_file(path, record, f"readout {tag}")
        if record.get("arrays") != expected_array_record:
            _fail(f"{tag}: manifest array schema is inconsistent")
        expected_checkpoint_sha = checkpoint.get("sha256")
        expected_stock_stats_sha = checkpoint.get("stock_stats_sha256")
        if (
            not isinstance(expected_stock_stats_sha, str)
            or not expected_stock_stats_sha
        ):
            _fail(f"{tag}: checkpoint stock_stats SHA-256 is missing")
        record_expectations = {
            "checkpoint_sha256": expected_checkpoint_sha,
            "stock_stats_sha256": expected_stock_stats_sha,
            "train_endpoint_sha256": split.train_endpoint_sha256,
            "val_endpoint_sha256": split.val_endpoint_sha256,
        }
        for key, expected in record_expectations.items():
            if record.get(key) != expected:
                _fail(f"{tag}: manifest field {key} is inconsistent")
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != readout_metadata_keys | set(readout_arrays):
                    _fail(f"{tag}: readout NPZ key set differs from schema")
                metadata = {
                    "schema_name": READOUT_SCHEMA,
                    "schema_version": READOUT_VERSION,
                    "protocol_fingerprint": protocol_fingerprint,
                    "split_fingerprint": split.split_fingerprint,
                    "dataset_sha256": dataset_sha,
                    "train_endpoint_sha256": split.train_endpoint_sha256,
                    "val_endpoint_sha256": split.val_endpoint_sha256,
                    "checkpoint_sha256": expected_checkpoint_sha,
                    "stock_stats_sha256": expected_stock_stats_sha,
                    "arm": checkpoint.get("arm"),
                    "seed": checkpoint.get("seed"),
                    "epoch": checkpoint.get("epoch"),
                    "n_train": len(split.train_t),
                    "n_val": len(split.val_t),
                }
                for key, expected in metadata.items():
                    if _scalar(data, key) != expected:
                        _fail(f"{tag}: readout metadata mismatch: {key}")
                artifact_fingerprint = str(
                    _scalar(data, "artifact_fingerprint")
                )
        except (OSError, ValueError, KeyError) as exc:
            _fail(f"{tag}: cannot validate readout: {exc}")
        if record.get("artifact_fingerprint") != artifact_fingerprint:
            _fail(f"{tag}: artifact fingerprint differs from manifest")
        specs = _npz_specs(path, list(readout_arrays))
        for key in readout_arrays:
            n_rows = len(split.train_t) if key.endswith("_train") else len(split.val_t)
            if specs[key] != ((n_rows, 512), np.dtype(np.float32)):
                _fail(f"{tag}:{key} shape/dtype is invalid")
        readout_paths.append(path)

    heldout_path: Optional[Path] = None
    if heldout_arg is not None:
        heldout_path = Path(heldout_arg).resolve()
        heldout_record = _require_mapping(
            manifest.get("heldout_targets"), "heldout_targets"
        )
        recorded = heldout_record.get("path")
        if not isinstance(recorded, str) or not recorded:
            _fail("held-out manifest path is missing")
        recorded_path = Path(recorded)
        if not recorded_path.is_absolute():
            recorded_path = in_dir / recorded_path
        if recorded_path.resolve() != heldout_path:
            _fail("--heldout does not match the artifact recorded in the manifest")
        _validate_recorded_file(heldout_path, heldout_record, "held-out targets")
        heldout_required = {
            "schema_name", "schema_version", "artifact_kind",
            "artifact_fingerprint", "source_sha256", "split_schema_name",
            "split_schema_version", "split_fingerprint", "dataset_sha256",
            "train_endpoint_sha256", "val_endpoint_sha256", "n_train", "n_val",
            "y_train_heldout", "y_val_heldout", "heldout_names",
        }
        try:
            with np.load(heldout_path, allow_pickle=False) as heldout:
                if set(heldout.files) != heldout_required:
                    _fail("held-out NPZ key set differs from schema")
                heldout_metadata = {
                    "schema_name": TARGET_SCHEMA,
                    "schema_version": TARGET_VERSION,
                    "artifact_kind": "heldout_targets",
                    "split_schema_name": SPLIT_SCHEMA,
                    "split_schema_version": SPLIT_VERSION,
                    "split_fingerprint": split.split_fingerprint,
                    "dataset_sha256": dataset_sha,
                    "train_endpoint_sha256": split.train_endpoint_sha256,
                    "val_endpoint_sha256": split.val_endpoint_sha256,
                    "n_train": len(split.train_t),
                    "n_val": len(split.val_t),
                }
                for key, expected in heldout_metadata.items():
                    if _scalar(heldout, key) != expected:
                        _fail(f"held-out metadata mismatch: {key}")
                heldout_names = [
                    str(value) for value in heldout["heldout_names"].tolist()
                ]
                heldout_artifact_fingerprint = str(
                    _scalar(heldout, "artifact_fingerprint")
                )
        except (OSError, ValueError, KeyError) as exc:
            _fail(f"cannot validate held-out targets: {exc}")
        if len(set(heldout_names)) != len(heldout_names):
            _fail("held-out target names are not unique")
        if set(heldout_names) & set(expected_names):
            _fail("held-out target names overlap trained target names")
        heldout_specs = _npz_specs(
            heldout_path,
            ["y_train_heldout", "y_val_heldout", "heldout_names"],
        )
        n_heldout = len(heldout_names)
        if heldout_specs["y_train_heldout"] != (
            (len(split.train_t), n_heldout), np.dtype(np.float32)
        ):
            _fail("held-out train array shape/dtype is invalid")
        if heldout_specs["y_val_heldout"] != (
            (len(split.val_t), n_heldout), np.dtype(np.float32)
        ):
            _fail("held-out validation array shape/dtype is invalid")
        if heldout_specs["heldout_names"][0] != (n_heldout,):
            _fail("held-out names shape is invalid")
        heldout_record_expectations = {
            "artifact_fingerprint": heldout_artifact_fingerprint,
            "shape_train": [len(split.train_t), n_heldout],
            "shape_val": [len(split.val_t), n_heldout],
            "train_endpoint_sha256": split.train_endpoint_sha256,
            "val_endpoint_sha256": split.val_endpoint_sha256,
        }
        for key, expected in heldout_record_expectations.items():
            if heldout_record.get(key) != expected:
                _fail(f"manifest held-out field {key} is inconsistent")

    return {
        "manifest": manifest,
        "split": split,
        "targets_path": target_path,
        "readout_paths": readout_paths,
        "heldout_path": heldout_path,
    }


# ------------------------------------------------------------- harmonization
def harmonize(y_train_raw: np.ndarray, idx: Dict) -> Dict:
    """SVD of the RAW centered directional block; gap-based effective rank."""
    D = y_train_raw[:, idx["dir_all"]]
    Dc = D - D.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Dc, compute_uv=False)
    sv_norm = sv / sv[0]
    # gap = largest ratio drop between consecutive singular values (log gap).
    logs = np.log10(np.maximum(sv_norm, 1e-30))
    gaps = logs[:-1] - logs[1:]
    cut = int(np.argmax(gaps)) + 1               # keep components up to the cliff
    return {
        "singular_values": sv.tolist(),
        "singular_values_normalized": sv_norm.tolist(),
        "effective_rank_gap": cut,
        "n_directional_columns": len(idx["dir_all"]),
        "dropped_for_aggregate": [idx["names"][i] for i in idx["redundant_cols"]],
        "note": ("spread/bid_rel/ask_rel are ONE quantity (rank 1) because "
                 "mid=(bid+ask)/2, so 3 features x 4 horizons collapse to rank 12. "
                 "effective_rank_gap==12 => only that exact redundancy; <12 => "
                 "d_microprice_rel (=spread*imbalance/2) is also near-collinear."),
    }


# --------------------------------------------------------------------- ladder
def pca_fit(Xtr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_mean, V) where V columns are variance-ordered PC directions.
    Centered, NOT per-dim standardized (standardizing destroys variance order)."""
    mu = Xtr.mean(axis=0, keepdims=True).astype(np.float64)
    Xc = Xtr.astype(np.float64) - mu
    # economy SVD: Xc = U S Vt ; rows of Vt are PCs, ordered by singular value.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu.astype(np.float32), Vt.T.astype(np.float32)   # V: (Dfeat, r)


def ladder_curve(Xtr, Xva, Ytr, Yva, mu, V, schedule) -> Dict[int, np.ndarray]:
    """R^2 per target on val, for each m in schedule. Min-norm OLS, zero shrink."""
    Ztr_full = (Xtr - mu) @ V           # (Ntr, r)
    Zva_full = (Xva - mu) @ V
    Ytr_c = Ytr - Ytr.mean(axis=0, keepdims=True)   # center targets on train mean
    ytr_mean = Ytr.mean(axis=0, keepdims=True)
    out = {}
    rmax = V.shape[1]
    for m in schedule:
        mm = min(m, rmax)
        Ztr = Ztr_full[:, :mm]
        Zva = Zva_full[:, :mm]
        W, *_ = np.linalg.lstsq(Ztr, Ytr_c, rcond=None)   # min-norm least squares
        pred = Zva @ W + ytr_mean                          # add back train mean
        out[m] = r2_per_target(Yva, pred)
    return out


def ridge_curve(Xtr, Xva, Ytr, Yva, mu, lambdas) -> Dict[float, np.ndarray]:
    """Ridge on the full centered representation: soft-truncation cross-check."""
    Xc = (Xtr - mu).astype(np.float64)
    Xv = (Xva - mu).astype(np.float64)
    Ytr_c = (Ytr - Ytr.mean(axis=0, keepdims=True)).astype(np.float64)
    ytr_mean = Ytr.mean(axis=0, keepdims=True)
    G = Xc.T @ Xc
    XtY = Xc.T @ Ytr_c
    I = np.eye(G.shape[0])
    out = {}
    for lam in lambdas:
        W = np.linalg.solve(G + lam * I, XtY)
        pred = Xv @ W + ytr_mean
        out[float(lam)] = r2_per_target(Yva, pred)
    return out


def ridge_curve_from_stats(stats, lambdas) -> Dict[float, np.ndarray]:
    """Ridge reference evaluated from the same centered sufficient statistics."""
    identity = np.eye(stats.dimension, dtype=np.float64)
    out = {}
    for lam in lambdas:
        weights = np.linalg.solve(
            stats.gram_train + float(lam) * identity,
            stats.cross_train,
        )
        cross_term = np.einsum("dt,dt->t", weights, stats.cross_val)
        quadratic = np.einsum(
            "dt,de,et->t", weights, stats.gram_val, weights
        )
        sse = (
            stats.val_y_train_centered_ss
            - 2.0 * cross_term
            + quadratic
        )
        out[float(lam)] = 1.0 - sse / np.maximum(
            stats.val_total_ss, 1e-12
        )
    return out


def mlp_ceiling(
    Xtr,
    Xva,
    Ytr,
    Yva,
    device,
    hidden=256,
    epochs=80,
    lr=1e-3,
    wd=1e-4,
    mlp_seeds=5,
    patience=10,
    split_seed=0,
    internal_val_fraction=0.1,
    batch_size=4096,
):
    """Deterministic multiseed MLP ceiling with fixed internal validation.

    Both X and Y standardization are fit on the reader-training subset.  The
    internal split is shared across reader seeds, so the reported standard
    deviation isolates reader initialization/optimization variance.
    """
    import torch
    import torch.nn as nn

    if not 0.0 < internal_val_fraction < 1.0:
        raise ValueError("internal_val_fraction must be in (0, 1)")
    if mlp_seeds < 1 or patience < 1 or epochs < 1:
        raise ValueError("mlp_seeds, patience and epochs must be positive")
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"MLP device {requested!r} requested but CUDA/ROCm is unavailable"
        )
    dev = torch.device(requested)

    x_train = np.asarray(Xtr, dtype=np.float32)
    x_outer_val = np.asarray(Xva, dtype=np.float32)
    y_train = np.asarray(Ytr, dtype=np.float32)
    y_outer_val = np.asarray(Yva, dtype=np.float32)
    split_rng = np.random.default_rng(split_seed)
    order = split_rng.permutation(len(x_train))
    n_internal_val = max(1, int(round(len(order) * internal_val_fraction)))
    internal_val_idx = order[:n_internal_val]
    reader_train_idx = order[n_internal_val:]
    if len(reader_train_idx) == 0:
        raise ValueError("internal reader split leaves no training rows")

    xmu = x_train[reader_train_idx].mean(axis=0, keepdims=True)
    xsd = x_train[reader_train_idx].std(axis=0, keepdims=True)
    xsd = np.where(xsd > 1e-6, xsd, 1.0)
    ymu = y_train[reader_train_idx].mean(axis=0, keepdims=True)
    ysd = y_train[reader_train_idx].std(axis=0, keepdims=True)
    ysd = np.where(ysd > 1e-6, ysd, 1.0)
    x_standard = (x_train - xmu) / xsd
    xv_standard = (x_outer_val - xmu) / xsd
    y_standard = (y_train - ymu) / ysd
    yv_standard = (y_outer_val - ymu) / ysd

    x_tensor = torch.as_tensor(x_standard, dtype=torch.float32, device=dev)
    xv_tensor = torch.as_tensor(xv_standard, dtype=torch.float32, device=dev)
    y_tensor = torch.as_tensor(y_standard, dtype=torch.float32, device=dev)
    train_idx_tensor = torch.as_tensor(
        reader_train_idx, dtype=torch.long, device=dev
    )
    internal_idx_tensor = torch.as_tensor(
        internal_val_idx, dtype=torch.long, device=dev
    )
    loss_function = nn.MSELoss()
    run_r2 = []
    epochs_used = []

    for reader_seed in range(mlp_seeds):
        seed = split_seed + reader_seed
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        net = nn.Sequential(
            nn.Linear(x_train.shape[1], hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, y_train.shape[1]),
        ).to(dev)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=wd
        )
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0
        best_epoch = 0
        epoch_rng = np.random.default_rng(seed)

        for epoch in range(1, epochs + 1):
            net.train()
            permutation = epoch_rng.permutation(len(reader_train_idx))
            for start in range(0, len(permutation), batch_size):
                positions = torch.as_tensor(
                    permutation[start : start + batch_size],
                    dtype=torch.long,
                    device=dev,
                )
                rows = train_idx_tensor[positions]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(net(x_tensor[rows]), y_tensor[rows])
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                internal_loss = float(
                    loss_function(
                        net(x_tensor[internal_idx_tensor]),
                        y_tensor[internal_idx_tensor],
                    ).item()
                )
            if internal_loss < best_loss:
                best_loss = internal_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in net.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is None:
            raise RuntimeError("MLP early stopping did not record a model")
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            prediction = net(xv_tensor).cpu().numpy()
        run_r2.append(r2_per_target(yv_standard, prediction))
        epochs_used.append(best_epoch)

    runs = np.stack(run_r2, axis=0)
    return {
        "mean": runs.mean(axis=0),
        "std": runs.std(axis=0),
        "n_seeds": int(mlp_seeds),
        "epochs_used": np.asarray(epochs_used, dtype=np.int64),
        "runs": runs,
        "reader_seeds": np.arange(split_seed, split_seed + mlp_seeds),
        "internal_train_indices": reader_train_idx,
        "internal_val_indices": internal_val_idx,
    }


# ----------------------------------------------------------------------- main
def parse_tag(fn: str):
    m = re.match(r"(?P<arm>.+)_seed(?P<seed>\d+)_ep(?P<ep>\d+)\.npz$", fn)
    if not m:
        return None
    return m.group("arm"), int(m.group("seed")), int(m.group("ep"))


def main():
    ap = argparse.ArgumentParser(description="Stage 2: accessibility rank-ladder")
    ap.add_argument("--in_dir", required=True, help="STAGE-1 out_dir (has readouts/, targets_shared.npz)")
    ap.add_argument("--out_dir", default=None, help="default: <in_dir>/analysis")
    ap.add_argument("--schedule", default=",".join(map(str, SCHEDULE_DEFAULT)))
    ap.add_argument(
        "--poolings",
        default=",".join(POOLINGS),
        help="comma-separated fixed poolings, including offline Hadamard variants",
    )
    ap.add_argument("--ridge_lambdas", default="0.1,1,10,100,1000")
    ap.add_argument("--run_mlp", action="store_true", help="also compute MLP ceiling (torch)")
    ap.add_argument(
        "--mlp_poolings",
        default="meanK_concatS",
        help="poolings to run MLP on",
    )
    ap.add_argument("--mlp_seeds", type=int, default=5)
    ap.add_argument("--mlp_patience", type=int, default=10)
    ap.add_argument("--mlp_max_epochs", type=int, default=80)
    ap.add_argument("--mlp_internal_val_fraction", type=float, default=0.10)
    ap.add_argument("--mlp_epochs_only", default="", help="restrict MLP to these encoder epochs, e.g. '10,11,12'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--null_draws", type=int, default=20)
    ap.add_argument("--null_seed", type=int, default=0)
    ap.add_argument(
        "--skip_null",
        action="store_true",
        help="skip the random-subspace reference (enabled by default)",
    )
    ap.add_argument("--angle_gap_threshold", type=float, default=1e-3)
    ap.add_argument("--reaggregate", action="store_true",
                    help="skip extraction/ladder; recompute aggregate+verdict from an "
                         "existing ladder_long.csv (e.g. after fixing the drop-list)")
    ap.add_argument("--heldout", default=None,
                    help="path to targets_heldout.npz (from screen_heldout_gate1 "
                         "--save_heldout). Adds held-out targets as separate blocks "
                         "(ho_imbalance, ho_depth, ho_timing) for the A-vs-B test.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out_dir) if args.out_dir else in_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    if args.reaggregate:
        idx = dir_indices()
        import csv as _csv
        lad_rows = []
        with open(out / "ladder_long.csv") as f:
            for r in _csv.DictReader(f):
                lad_rows.append(dict(arm=r["arm"], seed=r["seed"], epoch=int(r["epoch"]),
                                     pooling=r["pooling"], target=r["target"],
                                     m=int(r["m"]), r2=float(r["r2"])))
        print(f"[reaggregate] loaded {len(lad_rows):,} rows from ladder_long.csv")
        agg_rows = _aggregate(lad_rows, idx)
        _write_csv(out / "ladder_agg.csv", agg_rows,
                   ["arm", "epoch", "pooling", "block", "m", "r2_mean", "r2_std", "n_seeds"])
        try:
            _headline_plot(agg_rows, out / "plot_ladder_headline.png")
        except Exception as e:
            print(f"      (plot skipped: {e})")
        _print_verdict(agg_rows)
        print(f"Done (reaggregate) -> {out}")
        return
    inventory = validate_stage1_inputs(in_dir, args.heldout)
    print(
        f"[preflight] validated manifest v{MANIFEST_VERSION}, split, targets and "
        f"{len(inventory['readout_paths'])} readout dumps"
    )
    schedule = [int(x) for x in args.schedule.split(",")]
    poolings = [p for p in args.poolings.split(",") if p]
    ridge_lambdas = [float(x) for x in args.ridge_lambdas.split(",")]
    mlp_pools = set(p for p in args.mlp_poolings.split(",") if p)
    mlp_eps = set(int(x) for x in args.mlp_epochs_only.split(",")) if args.mlp_epochs_only else None
    split_seed = int(inventory["manifest"]["protocol"]["split_seed"])
    if split_seed != 0:
        raise RuntimeError(
            f"consolidation protocol requires split_seed=0, got {split_seed}"
        )

    idx = dir_indices()
    with np.load(inventory["targets_path"], allow_pickle=False) as tgt:
        y_train = tgt["y_train_raw"].astype(np.float64)
        y_val = tgt["y_val_raw"].astype(np.float64)
    names = list(idx["names"])

    if args.heldout:
        with np.load(inventory["heldout_path"], allow_pickle=False) as ho:
            ho_names = [str(x) for x in ho["heldout_names"]]
            y_train = np.concatenate(
                [y_train, ho["y_train_heldout"].astype(np.float64)], axis=1
            )
            y_val = np.concatenate(
                [y_val, ho["y_val_heldout"].astype(np.float64)], axis=1
            )
        names = names + ho_names
        print(f"[heldout] loaded {len(ho_names)} held-out targets from {args.heldout}")

    print("[1/3] Harmonization (SVD, gap rank)...")
    harm = harmonize(y_train, idx)
    with open(out / "harmonization.json", "w") as f:
        json.dump(harm, f, indent=2)
    print(f"      directional effective rank (gap) = {harm['effective_rank_gap']} "
          f"(expected {len(idx['dir_indep'])}: spread/bid/ask are one quantity; "
          f"<{len(idx['dir_indep'])} => microprice also near-collinear)")

    dumps = inventory["readout_paths"]
    print(
        f"[2/3] Geometry over {len(dumps)} dumps x "
        f"{len(poolings)} fixed poolings..."
    )
    lad_rows: List[Dict] = []
    ref_rows: List[Dict] = []
    mlp_run_rows: List[Dict] = []
    null_rows: List[Dict] = []
    angle_rows: List[Dict] = []
    spectral_rows: List[Dict] = []
    basis_cache = {}
    t0 = time.time()
    for di, p in enumerate(dumps):
        meta = parse_tag(p.name)
        if meta is None:
            continue
        arm, seed, ep = meta
        with np.load(p, allow_pickle=False) as dump:
            for pool in poolings:
                Xtr = derive_pooling(dump, pool, "train")
                Xva = derive_pooling(dump, pool, "val")
                dimension = int(Xtr.shape[1])
                # Preserve the legacy m=16 point for D=512 and add its exact
                # normalized counterpart (m/D=1/32) for every other D.
                normalized_anchor = max(1, int(round(dimension / 32)))
                grid = schedule_for_dimension(
                    [*schedule, normalized_anchor], dimension
                )
                stats = linear_stats(Xtr, y_train, Xva, y_val)
                eigenvalues, eigenvectors = pca_from_stats(stats)
                curve = ladder_from_stats(stats, eigenvectors, grid)
                full_r2 = curve[dimension]
                for m, r2 in curve.items():
                    fraction = np.divide(
                        r2,
                        full_r2,
                        out=np.full_like(r2, np.nan),
                        where=np.abs(full_r2) > 1e-12,
                    )
                    for ti, nm in enumerate(names):
                        lad_rows.append(
                            dict(
                                arm=arm,
                                seed=seed,
                                epoch=ep,
                                pooling=pool,
                                dimension=dimension,
                                target=nm,
                                m=m,
                                m_fraction=m / dimension,
                                r2=float(r2[ti]),
                                r2_full=float(full_r2[ti]),
                                r2_fraction=float(fraction[ti]),
                            )
                        )

                rc = ridge_curve_from_stats(stats, ridge_lambdas)
                for lam, r2 in rc.items():
                    for ti, nm in enumerate(names):
                        ref_rows.append(
                            dict(
                                arm=arm,
                                seed=seed,
                                epoch=ep,
                                pooling=pool,
                                ref="ridge",
                                lam=lam,
                                target=nm,
                                r2=float(r2[ti]),
                                mlp_mean=np.nan,
                                mlp_std=np.nan,
                                mlp_n_seeds=0,
                                mlp_epochs_used=np.nan,
                            )
                        )

                spectrum = spectral_diagnostics(eigenvalues)
                spectral_rows.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        epoch=ep,
                        pooling=pool,
                        dimension=dimension,
                        participation_ratio=spectrum[
                            "participation_ratio"
                        ],
                        effective_rank=spectrum["effective_rank"],
                    )
                )

                angle_blocks = {
                    "dir": idx["dir_indep"],
                    "vol": idx["vol_cols"],
                }
                if "time_to_next_mid_move" in names:
                    angle_blocks["timing"] = [
                        names.index("time_to_next_mid_move")
                    ]
                mean_basis = (
                    hadamard_mean_basis(dimension)
                    if pool in {"last_concat512", "meanK_concatS"}
                    else None
                )
                for block, target_indices in angle_blocks.items():
                    rows = principal_angle_curve(
                        stats,
                        eigenvalues,
                        eigenvectors,
                        target_indices,
                        grid,
                        reliability_threshold=args.angle_gap_threshold,
                        mean_basis=mean_basis,
                    )
                    for row in rows:
                        angle_rows.append(
                            dict(
                                arm=arm,
                                seed=seed,
                                epoch=ep,
                                pooling=pool,
                                dimension=dimension,
                                block=block,
                                m_fraction=row["m"] / dimension,
                                **row,
                            )
                        )

                if not args.skip_null:
                    cache_key = (
                        dimension,
                        args.null_draws,
                        args.null_seed,
                    )
                    if cache_key not in basis_cache:
                        basis_cache[cache_key] = haar_bases(
                            dimension,
                            n_draws=args.null_draws,
                            seed=args.null_seed,
                        )
                    null = random_subspace_null(
                        Xtr,
                        y_train,
                        grid,
                        n_draws=args.null_draws,
                        seed=args.null_seed,
                        X_val=Xva,
                        Y_val=y_val,
                        stats=stats,
                        bases=basis_cache[cache_key],
                    )
                    for m in grid:
                        observed = np.divide(
                            curve[m],
                            full_r2,
                            out=np.full_like(full_r2, np.nan),
                            where=np.abs(full_r2) > 1e-12,
                        )
                        for ti, nm in enumerate(names):
                            null_mean = float(null["mean"][m][ti])
                            null_rows.append(
                                dict(
                                    arm=arm,
                                    seed=seed,
                                    epoch=ep,
                                    pooling=pool,
                                    dimension=dimension,
                                    target=nm,
                                    m=m,
                                    m_fraction=m / dimension,
                                    r2_full=float(full_r2[ti]),
                                    observed_r2=float(curve[m][ti]),
                                    observed_fraction=float(observed[ti]),
                                    null_fraction_mean=null_mean,
                                    null_fraction_std=float(
                                        null["std"][m][ti]
                                    ),
                                    analytic_fraction=m / dimension,
                                    alignment=(
                                        "below_null"
                                        if observed[ti] < null_mean
                                        else "above_null"
                                    ),
                                    n_draws=args.null_draws,
                                    null_seed=args.null_seed,
                                )
                            )

                if (
                    args.run_mlp
                    and pool in mlp_pools
                    and (mlp_eps is None or ep in mlp_eps)
                ):
                    mlp = mlp_ceiling(
                        Xtr,
                        Xva,
                        y_train,
                        y_val,
                        args.device,
                        epochs=args.mlp_max_epochs,
                        mlp_seeds=args.mlp_seeds,
                        patience=args.mlp_patience,
                        split_seed=split_seed,
                        internal_val_fraction=args.mlp_internal_val_fraction,
                    )
                    mean_epochs = float(np.mean(mlp["epochs_used"]))
                    for ti, nm in enumerate(names):
                        ref_rows.append(
                            dict(
                                arm=arm,
                                seed=seed,
                                epoch=ep,
                                pooling=pool,
                                ref="mlp",
                                lam=np.nan,
                                target=nm,
                                r2=float(mlp["mean"][ti]),
                                mlp_mean=float(mlp["mean"][ti]),
                                mlp_std=float(mlp["std"][ti]),
                                mlp_n_seeds=mlp["n_seeds"],
                                mlp_epochs_used=mean_epochs,
                            )
                        )
                        for run_index, reader_seed in enumerate(
                            mlp["reader_seeds"]
                        ):
                            mlp_run_rows.append(
                                dict(
                                    arm=arm,
                                    encoder_seed=seed,
                                    epoch=ep,
                                    pooling=pool,
                                    reader_seed=int(reader_seed),
                                    target=nm,
                                    r2=float(mlp["runs"][run_index, ti]),
                                    epochs_used=int(
                                        mlp["epochs_used"][run_index]
                                    ),
                                )
                            )
                del stats, Xtr, Xva
        print(
            f"      {di + 1}/{len(dumps)} dumps  "
            f"({time.time() - t0:.0f}s)"
        )

    # write long tables
    _write_csv(
        out / "ladder_long.csv",
        lad_rows,
        [
            "arm", "seed", "epoch", "pooling", "dimension", "target", "m",
            "m_fraction", "r2", "r2_full", "r2_fraction",
        ],
    )
    _write_csv(
        out / "refs_long.csv",
        ref_rows,
        [
            "arm", "seed", "epoch", "pooling", "ref", "lam", "target", "r2",
            "mlp_mean", "mlp_std", "mlp_n_seeds", "mlp_epochs_used",
        ],
    )
    _write_csv(
        out / "mlp_reader_runs.csv",
        mlp_run_rows,
        [
            "arm", "encoder_seed", "epoch", "pooling", "reader_seed",
            "target", "r2", "epochs_used",
        ],
    )
    _write_csv(
        out / "random_subspace_null.csv",
        null_rows,
        [
            "arm", "seed", "epoch", "pooling", "dimension", "target", "m",
            "m_fraction", "r2_full", "observed_r2", "observed_fraction",
            "null_fraction_mean", "null_fraction_std", "analytic_fraction",
            "alignment", "n_draws", "null_seed",
        ],
    )
    _write_csv(
        out / "principal_angles.csv",
        angle_rows,
        [
            "arm", "seed", "epoch", "pooling", "dimension", "block", "m",
            "m_fraction", "k", "cos2", "aligned_energy", "signal_rank",
            "relative_eigen_gap", "reliable", "mean_subspace_energy",
        ],
    )
    _write_csv(
        out / "spectral_diagnostics.csv",
        spectral_rows,
        [
            "arm", "seed", "epoch", "pooling", "dimension",
            "participation_ratio", "effective_rank",
        ],
    )

    print("[3/3] Aggregating over seeds + headline plot...")
    agg_rows = _aggregate(lad_rows, idx)
    _write_csv(out / "ladder_agg.csv", agg_rows,
               ["arm", "epoch", "pooling", "block", "m", "r2_mean", "r2_std", "n_seeds"])
    fraction_agg = _aggregate_fraction(lad_rows)
    _write_csv(
        out / "ladder_fraction_agg.csv",
        fraction_agg,
        [
            "arm", "epoch", "pooling", "block", "m", "r2_mean", "r2_std",
            "n_seeds", "r2_full_mean", "r2_full_min",
            "denominator_reliable",
        ],
    )
    mlp_agg = _aggregate_mlp(ref_rows)
    _write_csv(
        out / "mlp_agg.csv",
        mlp_agg,
        [
            "arm", "epoch", "pooling", "block", "r2_mean", "encoder_std",
            "reader_std", "total_std", "n_encoder_seeds", "n_reader_seeds",
            "epochs_used_mean",
        ],
    )
    try:
        _headline_plot(agg_rows, out / "plot_ladder_headline.png")
    except Exception as e:
        print(f"      (legacy headline plot skipped: {e})")
    for pool in poolings:
        try:
            _normalized_plot(
                fraction_agg,
                null_rows,
                pool,
                out / f"plot_ladder_normalized_{pool}.png",
            )
        except Exception as e:
            print(f"      ({pool} normalized plot skipped: {e})")
    _print_verdict(agg_rows, pool="meanK_concatS")
    output_inventory = {}
    for output_path in sorted(out.iterdir()):
        if output_path.is_file() and output_path.name != "analysis_manifest.json":
            output_inventory[output_path.name] = {
                "sha256": sha256_file(output_path),
                "size_bytes": output_path.stat().st_size,
            }
    atomic_write_json(
        out / "analysis_manifest.json",
        {
            "schema": {
                "name": "consolidation_analysis_manifest",
                "version": 1,
            },
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage1_manifest_sha256": sha256_file(
                in_dir / "analysis_manifest.json"
            ),
            "heldout_sha256": (
                sha256_file(inventory["heldout_path"])
                if inventory["heldout_path"] is not None
                else None
            ),
            "sources": {
                "ladder_accessibility.py": sha256_file(Path(__file__)),
                "consolidation_geometry.py": sha256_file(
                    Path(__file__).with_name("consolidation_geometry.py")
                ),
            },
            "config": {
                "split_seed": split_seed,
                "schedule": schedule,
                "poolings": poolings,
                "ridge_lambdas": ridge_lambdas,
                "null_draws": 0 if args.skip_null else args.null_draws,
                "null_seed": args.null_seed,
                "angle_gap_threshold": args.angle_gap_threshold,
                "run_mlp": args.run_mlp,
                "mlp_poolings": sorted(mlp_pools),
                "mlp_seeds": args.mlp_seeds,
                "mlp_patience": args.mlp_patience,
                "mlp_max_epochs": args.mlp_max_epochs,
                "mlp_internal_val_fraction": args.mlp_internal_val_fraction,
                "device": args.device,
            },
            "outputs": output_inventory,
        },
    )
    print(f"Done -> {out}")


def _print_verdict(agg_rows, pool="last_concat512"):
    """Terminal verdict: R^2(m) table per arm, the scissor vs jepa_horizon, vol control.
    Canonical epoch per arm = epoch maximizing the CONTENT ceiling (directional R^2
    at m=512). This picks the converged state for every arm; selecting on m=32
    instead would reward near-init epochs for JEPA (whose low-rank accessibility is
    highest before training organizes signal into low-variance directions)."""
    def get(arm, ep, block, m):
        for r in agg_rows:
            if (r["arm"], r["epoch"], r["pooling"], r["block"], r["m"]) == (arm, ep, pool, block, m):
                return r
        return None
    arms = sorted({r["arm"] for r in agg_rows if r["pooling"] == pool and r["block"] == "dir"})
    ms = sorted({r["m"] for r in agg_rows if r["pooling"] == pool and r["block"] == "dir"})
    show = [m for m in (8, 16, 32, 64, 128, 512) if m in ms] or ms
    m_conv = max(ms)  # content ceiling = largest m available
    peak = {}
    for arm in arms:
        cand = {r["epoch"]: r["r2_mean"] for r in agg_rows
                if r["arm"] == arm and r["pooling"] == pool and r["block"] == "dir" and r["m"] == m_conv}
        if cand:
            peak[arm] = max(cand, key=cand.get)
    print("\n" + "=" * 78)
    print(f"DIRECTIONAL block (12 independent), pooling={pool}")
    print(f'{"arm":<16}{"ep*":>5} |' + "".join(f'{f"m={m}":>13}' for m in show))
    print("-" * 78)
    for arm in arms:
        ep = peak.get(arm)
        line = f'{arm:<16}{ep:>5} |'
        for m in show:
            r = get(arm, ep, "dir", m)
            line += f'{r["r2_mean"]:>8.3f}±{r["r2_std"]:<4.3f}' if r else f'{"--":>13}'
        print(line)
    if "supervised" in peak and "jepa_horizon" in peak:
        print("\nSCISSOR  supervised - jepa_horizon (each at its own ep*):")
        for m in show:
            rs = get("supervised", peak["supervised"], "dir", m)
            rj = get("jepa_horizon", peak["jepa_horizon"], "dir", m)
            if rs and rj:
                gap = rs["r2_mean"] - rj["r2_mean"]
                band = rs["r2_std"] + rj["r2_std"]
                # require a meaningful gap (>0.005) AND separation from the band;
                # at m=512 the gap CLOSES by design (content equivalence), so a
                # "within noise" there is the expected content result, not a failure.
                flag = ("SURVIVES" if (gap > 0.005 and gap > 2 * band) else
                        "marginal" if (gap > 0.005 and gap > band) else "within noise")
                print(f"  m={m:>3}: gap={gap:>+6.3f}  band~{band:.3f}  -> {flag}")
    print("\nVOL control (negative control: scissor expected ~0):")
    for arm in arms:
        ep = peak.get(arm)
        for m in (16, 512):
            r = get(arm, ep, "vol", m)
            if r:
                print(f"  {arm:<16} m={m:>3}: R2_vol={r['r2_mean']:>+6.3f}")

    # ---- HELD-OUT: the A-vs-B test, per family ----
    ho_blocks = [b for b in ("ho_imbalance", "ho_depth", "ho_timing")
                 if any(r["block"] == b for r in agg_rows)]
    if ho_blocks and "supervised" in peak and "jepa_horizon" in peak:
        print("\n" + "-" * 78)
        print("HELD-OUT (A vs B): does supervised's LOW-RANK accessibility COLLAPSE")
        print("toward jepa on targets NEITHER arm was trained on?")
        print("  collapse -> B (co-adaptation);  holds -> A (genuine geometry).")
        print("  read m=16 (accessibility) vs m=512 (content). sup uses its dir ep*;")
        print("  jepa its own. Reference: on TRAINED dir, the m=16 gap was ~0.31.")
        for b in ho_blocks:
            print(f"\n  [{b}]  supervised           jepa_horizon         gap(sup-jepa)")
            for m in (16, 512):
                rs = get("supervised", peak["supervised"], b, m)
                rj = get("jepa_horizon", peak["jepa_horizon"], b, m)
                if rs and rj:
                    gap = rs["r2_mean"] - rj["r2_mean"]
                    print(f"    m={m:>3}: {rs['r2_mean']:>7.3f}±{rs['r2_std']:.3f}     "
                          f"{rj['r2_mean']:>7.3f}±{rj['r2_std']:.3f}     {gap:>+7.3f}")
    print("=" * 78)


def _aggregate(lad_rows, idx=None):
    """Mean/std over seeds of the block-averaged R^2, per (arm,epoch,pooling,block,m).
    Blocks: dir, vol, ho_imbalance, ho_depth, ho_timing (via block_of)."""
    perseed: Dict[tuple, List[float]] = {}
    for r in lad_rows:
        block = block_of(r["target"])
        if block == "drop":
            continue
        key = (r["arm"], r["seed"], r["epoch"], r["pooling"], block, r["m"])
        perseed.setdefault(key, []).append(r["r2"])
    perseed_mean = {k: float(np.mean(v)) for k, v in perseed.items()}
    bykey: Dict[tuple, List[float]] = {}
    for (arm, seed, ep, pool, block, m), val in perseed_mean.items():
        bykey.setdefault((arm, ep, pool, block, m), []).append(val)
    rows = []
    for (arm, ep, pool, block, m), vals in sorted(bykey.items()):
        rows.append(dict(arm=arm, epoch=ep, pooling=pool, block=block, m=m,
                         r2_mean=float(np.mean(vals)), r2_std=float(np.std(vals)),
                         n_seeds=len(vals)))
    return rows


def _aggregate_fraction(lad_rows, denominator_threshold=0.01):
    """Block ratio of means, rather than the unstable mean of target ratios."""
    block_means = {}
    dimensions = {}
    for row in lad_rows:
        block = block_of(row["target"])
        if block == "drop":
            continue
        key = (
            row["arm"],
            row["seed"],
            row["epoch"],
            row["pooling"],
            block,
            row["m"],
        )
        block_means.setdefault(key, []).append(float(row["r2"]))
        dimensions[(row["arm"], row["seed"], row["epoch"], row["pooling"])] = int(
            row["dimension"]
        )
    block_means = {
        key: float(np.mean(values)) for key, values in block_means.items()
    }
    per_seed = {}
    for (arm, seed, epoch, pooling, block, m), value in block_means.items():
        dimension = dimensions[(arm, seed, epoch, pooling)]
        full = block_means[(arm, seed, epoch, pooling, block, dimension)]
        per_seed[(arm, seed, epoch, pooling, block, m)] = {
            "fraction": value / full if abs(full) > 1e-12 else float("nan"),
            "full": full,
        }
    grouped = {}
    for (arm, seed, epoch, pooling, block, m), value in per_seed.items():
        grouped.setdefault((arm, epoch, pooling, block, m), []).append(value)
    rows = []
    for (arm, epoch, pooling, block, m), values in sorted(grouped.items()):
        fractions = np.asarray([value["fraction"] for value in values])
        full = np.asarray([value["full"] for value in values])
        rows.append(
            {
                "arm": arm,
                "epoch": epoch,
                "pooling": pooling,
                "block": block,
                "m": m,
                "r2_mean": float(np.nanmean(fractions)),
                "r2_std": float(np.nanstd(fractions)),
                "n_seeds": len(values),
                "r2_full_mean": float(full.mean()),
                "r2_full_min": float(full.min()),
                "denominator_reliable": bool(
                    np.all(full >= denominator_threshold)
                ),
            }
        )
    return rows


def _aggregate_mlp(ref_rows):
    """Compose encoder variation and reader variation for MLP block summaries."""
    per_encoder = {}
    for row in ref_rows:
        if row["ref"] != "mlp":
            continue
        block = block_of(row["target"])
        if block == "drop":
            continue
        key = (
            row["arm"],
            row["seed"],
            row["epoch"],
            row["pooling"],
            block,
        )
        bucket = per_encoder.setdefault(
            key, {"means": [], "variances": [], "epochs": [], "n_reader": []}
        )
        bucket["means"].append(float(row["mlp_mean"]))
        bucket["variances"].append(float(row["mlp_std"]) ** 2)
        bucket["epochs"].append(float(row["mlp_epochs_used"]))
        bucket["n_reader"].append(int(row["mlp_n_seeds"]))

    grouped = {}
    for (arm, seed, epoch, pooling, block), values in per_encoder.items():
        key = (arm, epoch, pooling, block)
        grouped.setdefault(key, []).append(
            {
                "mean": float(np.mean(values["means"])),
                "reader_var": float(np.mean(values["variances"])),
                "epochs": float(np.mean(values["epochs"])),
                "n_reader": max(values["n_reader"]),
            }
        )

    rows = []
    for (arm, epoch, pooling, block), encoders in sorted(grouped.items()):
        means = np.asarray([item["mean"] for item in encoders])
        reader_std = float(
            np.sqrt(np.mean([item["reader_var"] for item in encoders]))
        )
        encoder_std = float(means.std())
        rows.append(
            {
                "arm": arm,
                "epoch": epoch,
                "pooling": pooling,
                "block": block,
                "r2_mean": float(means.mean()),
                "encoder_std": encoder_std,
                "reader_std": reader_std,
                "total_std": float(
                    np.sqrt(encoder_std**2 + reader_std**2)
                ),
                "n_encoder_seeds": len(encoders),
                "n_reader_seeds": max(
                    item["n_reader"] for item in encoders
                ),
                "epochs_used_mean": float(
                    np.mean([item["epochs"] for item in encoders])
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict], cols: List[str]):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def _headline_plot(agg_rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Directional block, last_concat512, at each arm's max epoch present.
    sub = [r for r in agg_rows if r["block"] == "dir" and r["pooling"] == "last_concat512"]
    if not sub:
        raise RuntimeError("no directional/last_concat512 rows")
    arms = sorted(set(r["arm"] for r in sub))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arm in arms:
        ep = max(r["epoch"] for r in sub if r["arm"] == arm)
        rows = sorted([r for r in sub if r["arm"] == arm and r["epoch"] == ep], key=lambda r: r["m"])
        ms = [r["m"] for r in rows]
        mean = np.array([r["r2_mean"] for r in rows])
        std = np.array([r["r2_std"] for r in rows])
        ax.plot(ms, mean, "o-", label=f"{arm} (ep{ep})")
        ax.fill_between(ms, mean - std, mean + std, alpha=0.2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m  (top-m PCA directions, variance-ordered)")
    ax.set_ylabel("R² (val, directional aggregate)")
    ax.set_title("Accessibility ladder — directional signal vs PCA rank (multiseed band)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _normalized_plot(fraction_agg, null_rows, pool: str, path: Path):
    """Normalized directional ladder with numerical and analytic null curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = [
        row
        for row in fraction_agg
        if row["block"] == "dir" and row["pooling"] == pool
    ]
    if not sub:
        raise RuntimeError(f"no normalized directional rows for {pool}")
    max_m = max(row["m"] for row in sub)
    fraction_lookup = {
        row["m"]: row["m_fraction"]
        for row in null_rows
        if row["pooling"] == pool
    }
    arms = sorted({row["arm"] for row in sub})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arm in arms:
        epochs = sorted(
            {row["epoch"] for row in sub if row["arm"] == arm}
        )
        epoch = epochs[-1]
        rows = sorted(
            [
                row
                for row in sub
                if row["arm"] == arm and row["epoch"] == epoch
            ],
            key=lambda row: row["m"],
        )
        xs = [
            fraction_lookup.get(row["m"], row["m"] / max_m)
            for row in rows
        ]
        line = ax.plot(
            xs,
            [row["r2_mean"] for row in rows],
            "o-",
            label=f"{arm} observed",
        )[0]
        ax.fill_between(
            xs,
            [row["r2_mean"] - row["r2_std"] for row in rows],
            [row["r2_mean"] + row["r2_std"] for row in rows],
            alpha=0.16,
            color=line.get_color(),
        )
        numeric_per_seed = {}
        for row in null_rows:
            if (
                row["pooling"] == pool
                and row["arm"] == arm
                and block_of(row["target"]) == "dir"
            ):
                numeric_per_seed.setdefault(
                    (row["seed"], row["m"]), []
                ).append(row)
        numeric = {}
        for (_, m), target_rows in numeric_per_seed.items():
            denominator = float(
                np.sum([row["r2_full"] for row in target_rows])
            )
            numerator = float(
                np.sum(
                    [
                        row["null_fraction_mean"] * row["r2_full"]
                        for row in target_rows
                    ]
                )
            )
            numeric.setdefault(m, []).append(
                numerator / denominator
                if abs(denominator) > 1e-12
                else float("nan")
            )
        if numeric:
            ms = sorted(numeric)
            ax.plot(
                [fraction_lookup.get(m, m / max_m) for m in ms],
                [float(np.nanmean(numeric[m])) for m in ms],
                ":",
                color=line.get_color(),
                label=f"{arm} random null",
            )
    diagonal = np.linspace(0.0, 1.0, 101)
    ax.plot(diagonal, diagonal, "k--", linewidth=1.2, label="analytic m/D")
    ax.set_xlabel("m / D")
    ax.set_ylabel("R²(m) / R²(D)")
    ax.set_title(f"Normalized accessibility — {pool}")
    ax.set_xlim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Preregistered T2 token-role matched-null diagnostic for Experiment 01.

The implementation deliberately operates on the historical post-P0 readout
dumps.  It first reproduces the historical OLS cells, then caches one set of
full 512-dimensional sufficient statistics per arm/seed/readout.  Every
observed or random projection is evaluated from those statistics; feature
matrices are never rescanned per draw.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .errors import ExperimentIntegrityError
from .io import (
    atomic_savez,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    canonical_json_sha256,
    sha256_file,
)
from .reference.consolidation_geometry import (
    HADAMARD4,
    LinearStats,
    linear_stats,
    pca_from_stats,
    r2_from_basis,
    signal_basis,
)
from .reference.ladder_accessibility import block_of


SPEC_RELATIVE_PATH = (
    "docs/experiment01/SPEC_EXPERIMENT_01_TOKEN_ROLE_MATCHED_NULL_20260826.md"
)
SPEC_SHA256 = "51db7780a12fefe41e3191e5fef77ce4e87deee5becf861a0000fc8c30864a68"
INPUT_HASHES = {
    "analysis_manifest.json": (
        "96eaf6b2e6829779697c224d6919364ce159e016c0caf1975fc0ee752ccb2e91"
    ),
    "targets_shared.npz": (
        "f2ab87577875e8c535d9e7ebdd4b60df991f20c2564cb9c8d57aeaa9ac9e9ac9"
    ),
    "split.npz": (
        "0c5149c1260c153c8bdbe3ac8a453750816b4ef62eaa6b54ac03ffb396245cc3"
    ),
    "analysis_consolidation_20260728/ladder_long.csv": (
        "62dfeb18c0d9a8c792e7c788b803e3666f8ecf423956119f15076573e1d05785"
    ),
    "analysis_consolidation_20260728/ladder_agg.csv": (
        "fc32ec77807a6cff4a04ea1358a2ea8a6a977a332a711f7960ef724cd51086db"
    ),
}
ARMS = ("jepa_horizon", "jepa_masked", "supervised")
ENCODER_SEEDS = (0, 1, 2)
READOUTS = {
    "last_concat512": {
        "dump_key": "last_concat512",
        "historical_full": "last_concat512",
        "historical_common": "last_hadamard_mean128",
        "historical_complement": "last_hadamard_contrast384",
        "index": 0,
    },
    "meanK_concatS": {
        "dump_key": "tmean_concat512",
        "historical_full": "meanK_concatS",
        "historical_common": "meanK_hadamard_mean128",
        "historical_complement": "meanK_hadamard_contrast384",
        "index": 1,
    },
}
BASE_SEED = 20260826
N_DRAWS = 100
REPRODUCTION_TOLERANCE = 5e-10
BASIS_TOLERANCE = 1e-10
CEILING_THRESHOLD = 0.01
TARGET_DIMENSION = 22
FEATURE_DIMENSION = 512
CHANNEL_DIMENSION = 128


@dataclass(frozen=True)
class TokenRoleConfig:
    n_draws: int = N_DRAWS
    base_seed: int = BASE_SEED
    reproduction_tolerance: float = REPRODUCTION_TOLERANCE
    basis_tolerance: float = BASIS_TOLERANCE
    ceiling_threshold: float = CEILING_THRESHOLD

    def validate(self) -> None:
        if self.n_draws != N_DRAWS:
            raise ExperimentIntegrityError("T2 fixes exactly 100 null draws")
        if self.base_seed != BASE_SEED:
            raise ExperimentIntegrityError("T2 base seed changed")
        if self.reproduction_tolerance != REPRODUCTION_TOLERANCE:
            raise ExperimentIntegrityError("T2 reproduction tolerance changed")
        if self.basis_tolerance != BASIS_TOLERANCE:
            raise ExperimentIntegrityError("T2 basis tolerance changed")
        if self.ceiling_threshold != CEILING_THRESHOLD:
            raise ExperimentIntegrityError("T2 ceiling threshold changed")


@dataclass
class CachedRoleStats:
    """Historical linear statistics plus deterministic shuffled-target cross terms."""

    stats: LinearStats
    cross_train_shuffled: np.ndarray
    cross_val_shuffled: np.ndarray
    source_sha256: str
    arm: str
    encoder_seed: int
    readout: str
    n_train: int
    n_val: int

    def shuffled_stats(self) -> LinearStats:
        return LinearStats(
            x_mean=self.stats.x_mean,
            y_mean=self.stats.y_mean,
            gram_train=self.stats.gram_train,
            cross_train=self.cross_train_shuffled,
            gram_val=self.stats.gram_val,
            cross_val=self.cross_val_shuffled,
            val_y_train_centered_ss=self.stats.val_y_train_centered_ss,
            val_total_ss=self.stats.val_total_ss,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_file_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentIntegrityError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ExperimentIntegrityError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return {
        "path": path.name,
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def _readout_key(arm: str, seed: int) -> str:
    return f"{arm}_seed{seed}_ep020"


def _readout_path(input_dir: Path, arm: str, seed: int) -> Path:
    return input_dir / "readouts" / f"{_readout_key(arm, seed)}.npz"


def validate_historical_inputs(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Hash and structurally validate every frozen historical T2 input."""
    root = Path(input_dir)
    out = Path(output_dir)
    records: dict[str, Any] = {}
    for relative, expected in INPUT_HASHES.items():
        path = root / relative
        record = _require_file_hash(path, expected, relative)
        record["relative_path"] = relative
        records[relative] = record

    manifest = json.loads((root / "analysis_manifest.json").read_text())
    protocol = manifest.get("protocol", {})
    if (
        protocol.get("requested_n_train") != 100000
        or protocol.get("requested_n_val") != 50000
        or protocol.get("K") != 20
    ):
        raise ExperimentIntegrityError("historical readout protocol dimensions changed")
    expected_keys = {_readout_key(arm, seed) for arm in ARMS for seed in ENCODER_SEEDS}
    inventory = manifest.get("readouts", {})
    if set(inventory) != expected_keys:
        raise ExperimentIntegrityError("historical readout inventory is not exactly 3x3")
    readout_records: dict[str, Any] = {}
    endpoint_pairs = set()
    for key in sorted(expected_keys):
        entry = inventory[key]
        path = root / entry["path"]
        actual = sha256_file(path)
        if actual != entry["file_sha256"]:
            raise ExperimentIntegrityError(f"readout hash mismatch: {key}")
        expected_arrays = {
            f"{base}_{split}": shape
            for base in ("last_concat512", "tmean_concat512")
            for split, shape in (("train", [100000, 512]), ("val", [50000, 512]))
        }
        arrays = entry.get("arrays", {})
        if set(arrays) != set(expected_arrays):
            raise ExperimentIntegrityError(f"readout array inventory differs: {key}")
        for array_key, shape in expected_arrays.items():
            if arrays[array_key].get("shape") != shape or arrays[array_key].get(
                "dtype"
            ) != "float32":
                raise ExperimentIntegrityError(f"readout array contract differs: {key}")
        endpoint_pairs.add(
            (entry["train_endpoint_sha256"], entry["val_endpoint_sha256"])
        )
        readout_records[key] = {
            "relative_path": entry["path"],
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "train_endpoint_sha256": entry["train_endpoint_sha256"],
            "val_endpoint_sha256": entry["val_endpoint_sha256"],
        }
    if len(endpoint_pairs) != 1:
        raise ExperimentIntegrityError("readout endpoints are not globally identical")

    targets_path = root / "targets_shared.npz"
    with np.load(targets_path, allow_pickle=False) as targets:
        required = {"y_train_raw", "y_val_raw", "target_names"}
        if not required.issubset(targets.files):
            raise ExperimentIntegrityError("trained-target artifact is incomplete")
        if targets["y_train_raw"].shape != (100000, TARGET_DIMENSION):
            raise ExperimentIntegrityError("historical train target shape differs")
        if targets["y_val_raw"].shape != (50000, TARGET_DIMENSION):
            raise ExperimentIntegrityError("historical validation target shape differs")
        names = [str(value) for value in targets["target_names"].tolist()]
        independent = [name for name in names if block_of(name) == "dir"]
        if len(independent) != 12:
            raise ExperimentIntegrityError("directional independent target count is not 12")
        endpoint = next(iter(endpoint_pairs))
        if str(targets["train_endpoint_sha256"].item()) != endpoint[0]:
            raise ExperimentIntegrityError("train target/readout endpoint hash mismatch")
        if str(targets["val_endpoint_sha256"].item()) != endpoint[1]:
            raise ExperimentIntegrityError("validation target/readout endpoint hash mismatch")

    payload = {
        "schema_name": "thesis.experiment01.token_role.input_gate",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "passed",
        "input_dir_name": root.name,
        "fixed_files": records,
        "readouts": readout_records,
        "n_readout_dumps": len(readout_records),
        "n_train": 100000,
        "n_validation": 50000,
        "n_targets": len(names),
        "n_independent_directional_targets": len(independent),
        "target_names": names,
        "independent_directional_targets": independent,
        "train_endpoint_sha256": endpoint[0],
        "validation_endpoint_sha256": endpoint[1],
    }
    payload["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "created_at_utc"}
    )
    if write:
        atomic_write_json(out / "input_gate.json", payload)
    return payload


def observed_role_bases() -> dict[str, np.ndarray]:
    eye = np.eye(CHANNEL_DIMENSION, dtype=np.float64)
    bases = {
        "common": np.kron(HADAMARD4[:, 0:1], eye),
        "complement": np.kron(HADAMARD4[:, 1:4], eye),
    }
    validate_complementary_bases(bases["common"], bases["complement"])
    return bases


def _haar_q(dimension: int, seed_components: Sequence[int]) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_components)))
    raw = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(raw)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


def structured_role_bases(draw_id: int, base_seed: int = BASE_SEED) -> dict[str, np.ndarray]:
    if not 0 <= int(draw_id) < N_DRAWS:
        raise ValueError("structured draw_id is outside 0..99")
    q = _haar_q(4, [base_seed, int(draw_id), 4])
    eye = np.eye(CHANNEL_DIMENSION, dtype=np.float64)
    bases = {
        "common": np.kron(q[:, 0:1], eye),
        "complement": np.kron(q[:, 1:4], eye),
    }
    validate_complementary_bases(bases["common"], bases["complement"])
    return bases


def generic_feature_basis(
    draw_id: int, dimension: int, base_seed: int = BASE_SEED
) -> np.ndarray:
    if dimension not in (128, 384):
        raise ValueError("generic T2 subspace dimension must be 128 or 384")
    if not 0 <= int(draw_id) < N_DRAWS:
        raise ValueError("generic draw_id is outside 0..99")
    rng = np.random.default_rng(
        np.random.SeedSequence([base_seed, int(draw_id), int(dimension), 512])
    )
    q, r = np.linalg.qr(
        rng.standard_normal((FEATURE_DIMENSION, dimension)), mode="reduced"
    )
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    _validate_basis(q)
    return q


def _validate_basis(basis: np.ndarray, tolerance: float = BASIS_TOLERANCE) -> None:
    q = np.asarray(basis, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != FEATURE_DIMENSION:
        raise ExperimentIntegrityError(f"invalid T2 basis shape {q.shape}")
    error = float(np.max(np.abs(q.T @ q - np.eye(q.shape[1]))))
    if not np.isfinite(error) or error > tolerance:
        raise ExperimentIntegrityError(f"basis orthogonality error {error:.3e}")


def validate_complementary_bases(common: np.ndarray, complement: np.ndarray) -> None:
    _validate_basis(common)
    _validate_basis(complement)
    if common.shape != (512, 128) or complement.shape != (512, 384):
        raise ExperimentIntegrityError("role basis dimensions differ")
    cross_error = float(np.max(np.abs(common.T @ complement)))
    complete_error = float(
        np.max(
            np.abs(
                common @ common.T
                + complement @ complement.T
                - np.eye(FEATURE_DIMENSION)
            )
        )
    )
    if cross_error > BASIS_TOLERANCE or complete_error > BASIS_TOLERANCE:
        raise ExperimentIntegrityError("role bases are not exact complements")


def _cache_path(output_dir: Path, arm: str, seed: int, readout: str) -> Path:
    return output_dir / "sufficient_statistics" / f"{arm}_seed{seed}_{readout}.npz"


def _load_targets(input_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(input_dir / "targets_shared.npz", allow_pickle=False) as payload:
        y_train = np.asarray(payload["y_train_raw"])
        y_val = np.asarray(payload["y_val_raw"])
        names = [str(value) for value in payload["target_names"].tolist()]
    if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
        raise ExperimentIntegrityError("non-finite historical targets")
    return y_train, y_val, names


def _shuffled_cross_terms(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    *,
    encoder_seed: int,
    readout_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = np.asarray(x_train, dtype=np.float64).mean(axis=0, keepdims=True)
    y_mean = np.asarray(y_train, dtype=np.float64).mean(axis=0, keepdims=True)
    train_rng = np.random.default_rng(
        np.random.SeedSequence(
            [BASE_SEED, int(encoder_seed), int(readout_index), 0, 991]
        )
    )
    val_rng = np.random.default_rng(
        np.random.SeedSequence(
            [BASE_SEED, int(encoder_seed), int(readout_index), 1, 991]
        )
    )
    train_order = train_rng.permutation(len(y_train))
    val_order = val_rng.permutation(len(y_val))
    cross_train = (np.asarray(x_train, dtype=np.float64) - x_mean).T @ (
        np.asarray(y_train[train_order], dtype=np.float64) - y_mean
    )
    cross_val = (np.asarray(x_val, dtype=np.float64) - x_mean).T @ (
        np.asarray(y_val[val_order], dtype=np.float64) - y_mean
    )
    return cross_train, cross_val


def build_or_load_cached_stats(
    input_dir: str | Path,
    output_dir: str | Path,
    arm: str,
    seed: int,
    readout: str,
) -> CachedRoleStats:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    if arm not in ARMS or seed not in ENCODER_SEEDS or readout not in READOUTS:
        raise ValueError("unknown T2 feature set")
    source = _readout_path(input_root, arm, seed)
    source_hash = sha256_file(source)
    cache = _cache_path(output_root, arm, seed, readout)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as payload:
            if str(payload["source_sha256"].item()) != source_hash:
                raise ExperimentIntegrityError("T2 sufficient-statistics source changed")
            if str(payload["arm"].item()) != arm:
                raise ExperimentIntegrityError("T2 cached arm differs")
            if int(payload["encoder_seed"].item()) != seed:
                raise ExperimentIntegrityError("T2 cached encoder seed differs")
            if str(payload["readout"].item()) != readout:
                raise ExperimentIntegrityError("T2 cached readout differs")
            stats = LinearStats(
                x_mean=payload["x_mean"],
                y_mean=payload["y_mean"],
                gram_train=payload["gram_train"],
                cross_train=payload["cross_train"],
                gram_val=payload["gram_val"],
                cross_val=payload["cross_val"],
                val_y_train_centered_ss=payload["val_y_train_centered_ss"],
                val_total_ss=payload["val_total_ss"],
            )
            result = CachedRoleStats(
                stats=stats,
                cross_train_shuffled=payload["cross_train_shuffled"],
                cross_val_shuffled=payload["cross_val_shuffled"],
                source_sha256=source_hash,
                arm=arm,
                encoder_seed=seed,
                readout=readout,
                n_train=int(payload["n_train"].item()),
                n_val=int(payload["n_val"].item()),
            )
        _validate_cached_stats(result)
        return result

    y_train, y_val, _ = _load_targets(input_root)
    base = READOUTS[readout]["dump_key"]
    with np.load(source, allow_pickle=False) as dump:
        x_train = np.asarray(dump[f"{base}_train"])
        x_val = np.asarray(dump[f"{base}_val"])
        if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
            raise ExperimentIntegrityError("non-finite historical readout")
        stats = linear_stats(x_train, y_train, x_val, y_val)
        shuffled_train, shuffled_val = _shuffled_cross_terms(
            x_train,
            x_val,
            y_train,
            y_val,
            encoder_seed=seed,
            readout_index=int(READOUTS[readout]["index"]),
        )
    result = CachedRoleStats(
        stats=stats,
        cross_train_shuffled=shuffled_train,
        cross_val_shuffled=shuffled_val,
        source_sha256=source_hash,
        arm=arm,
        encoder_seed=seed,
        readout=readout,
        n_train=len(y_train),
        n_val=len(y_val),
    )
    _validate_cached_stats(result)
    atomic_savez(
        cache,
        schema_name=np.asarray("thesis.experiment01.token_role.stats"),
        schema_version=np.asarray(1, dtype=np.int64),
        source_sha256=np.asarray(source_hash),
        arm=np.asarray(arm),
        encoder_seed=np.asarray(seed, dtype=np.int64),
        readout=np.asarray(readout),
        n_train=np.asarray(result.n_train, dtype=np.int64),
        n_val=np.asarray(result.n_val, dtype=np.int64),
        x_mean=stats.x_mean,
        y_mean=stats.y_mean,
        gram_train=stats.gram_train,
        cross_train=stats.cross_train,
        gram_val=stats.gram_val,
        cross_val=stats.cross_val,
        val_y_train_centered_ss=stats.val_y_train_centered_ss,
        val_total_ss=stats.val_total_ss,
        cross_train_shuffled=shuffled_train,
        cross_val_shuffled=shuffled_val,
    )
    return result


def _validate_cached_stats(cached: CachedRoleStats) -> None:
    stats = cached.stats
    expected = {
        "gram_train": (512, 512),
        "cross_train": (512, 22),
        "gram_val": (512, 512),
        "cross_val": (512, 22),
        "cross_train_shuffled": (512, 22),
        "cross_val_shuffled": (512, 22),
    }
    values = {
        "gram_train": stats.gram_train,
        "cross_train": stats.cross_train,
        "gram_val": stats.gram_val,
        "cross_val": stats.cross_val,
        "cross_train_shuffled": cached.cross_train_shuffled,
        "cross_val_shuffled": cached.cross_val_shuffled,
    }
    for name, shape in expected.items():
        value = np.asarray(values[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ExperimentIntegrityError(f"invalid cached statistic {name}")
    if cached.n_train != 100000 or cached.n_val != 50000:
        raise ExperimentIntegrityError("cached T2 row count differs")


def freeze_protocol(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TokenRoleConfig = TokenRoleConfig(),
) -> dict[str, Any]:
    """Freeze T2 source/input identities before null production."""
    config.validate()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    repo = _repo_root()
    spec = repo / SPEC_RELATIVE_PATH
    if sha256_file(spec) != SPEC_SHA256:
        raise ExperimentIntegrityError("T2 preregistration specification changed")
    gate = validate_historical_inputs(input_dir, output_root, write=True)
    sources = [
        repo / "experiment01/token_role.py",
        repo / "scripts/experiment01/run_experiment_01_token_role.py",
        repo / "tests/test_experiment01_token_role.py",
    ]
    for path in sources:
        if not path.is_file():
            raise ExperimentIntegrityError(f"T2 implementation source missing: {path}")
    payload: dict[str, Any] = {
        "schema_name": "thesis.experiment01.token_role.protocol",
        "schema_version": 1,
        "status": "frozen_pre_null",
        "created_at_utc": _utc_now(),
        "specification": {
            "relative_path": SPEC_RELATIVE_PATH,
            "sha256": SPEC_SHA256,
        },
        "config": asdict(config),
        "input_gate_payload_sha256": gate["payload_sha256"],
        "sources": {
            str(path.relative_to(repo)): sha256_file(path) for path in sources
        },
        "inventory": {
            "arms": list(ARMS),
            "encoder_seeds": list(ENCODER_SEEDS),
            "readouts": list(READOUTS),
            "null_draws": N_DRAWS,
            "structured_subspace_dimensions": [128, 384],
            "generic_subspace_dimensions": [128, 384],
            "primary_directional_targets": 12,
        },
        "scientific_status": (
            "post-hoc corrective diagnostic; Phase-I A1 and Phase-III R3 unchanged"
        ),
    }
    payload["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "created_at_utc"}
    )
    path = output_root / "protocol_frozen.json"
    if path.exists():
        existing = json.loads(path.read_text())
        comparable_existing = dict(existing)
        comparable_payload = dict(payload)
        comparable_existing.pop("created_at_utc", None)
        comparable_payload.pop("created_at_utc", None)
        if comparable_existing != comparable_payload:
            raise ExperimentIntegrityError("existing T2 frozen protocol differs")
        return existing
    forbidden = list(output_root.glob("token_role_*.parquet"))
    if forbidden:
        raise ExperimentIntegrityError("refusing to freeze protocol after null output")
    atomic_write_json(path, payload)
    return payload


def _require_frozen_protocol(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "protocol_frozen.json"
    if not path.is_file():
        raise ExperimentIntegrityError("T2 protocol is not frozen")
    payload = json.loads(path.read_text())
    if payload.get("status") != "frozen_pre_null":
        raise ExperimentIntegrityError("T2 protocol status differs")
    if payload.get("specification", {}).get("sha256") != SPEC_SHA256:
        raise ExperimentIntegrityError("T2 frozen specification hash differs")
    repo = _repo_root()
    for relative, expected in payload.get("sources", {}).items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ExperimentIntegrityError(f"T2 frozen source changed: {relative}")
    return payload


def reproduce_historical_token_role(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> dict[str, Any]:
    """Reproduce every full/common/complement historical per-target OLS cell."""
    if tolerance != REPRODUCTION_TOLERANCE:
        raise ExperimentIntegrityError("T2 reproduction tolerance changed")
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    _require_frozen_protocol(output_root)
    validate_historical_inputs(input_root, output_root, write=True)
    ladder = pd.read_csv(
        input_root / "analysis_consolidation_20260728/ladder_long.csv"
    )
    aggregate_reference = pd.read_csv(
        input_root / "analysis_consolidation_20260728/ladder_agg.csv"
    )
    _, _, names = _load_targets(input_root)
    bases = observed_role_bases()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for arm in ARMS:
        for seed in ENCODER_SEEDS:
            for readout, readout_meta in READOUTS.items():
                cached = build_or_load_cached_stats(
                    input_root, output_root, arm, seed, readout
                )
                calculations = {
                    "full": r2_from_basis(cached.stats, np.eye(512)),
                    "common": r2_from_basis(cached.stats, bases["common"]),
                    "complement": r2_from_basis(
                        cached.stats, bases["complement"]
                    ),
                }
                historical_poolings = {
                    "full": readout_meta["historical_full"],
                    "common": readout_meta["historical_common"],
                    "complement": readout_meta["historical_complement"],
                }
                for subspace, values in calculations.items():
                    historical_pool = historical_poolings[subspace]
                    dimension = {"full": 512, "common": 128, "complement": 384}[
                        subspace
                    ]
                    reference = ladder.loc[
                        (ladder["arm"] == arm)
                        & (ladder["seed"] == seed)
                        & (ladder["epoch"] == 20)
                        & (ladder["pooling"] == historical_pool)
                        & (ladder["m"] == dimension)
                    ]
                    if not set(names).issubset(set(reference["target"])):
                        raise ExperimentIntegrityError(
                            f"historical reproduction cell missing: {arm}/{seed}/{historical_pool}"
                        )
                    reference = reference.set_index("target").loc[names]
                    if len(reference) != len(names) or not reference.index.is_unique:
                        raise ExperimentIntegrityError(
                            f"historical reproduction cell duplicated: {arm}/{seed}/{historical_pool}"
                        )
                    for target_index, target in enumerate(names):
                        observed = float(values[target_index])
                        expected = float(reference.iloc[target_index]["r2"])
                        difference = abs(observed - expected)
                        rows.append(
                            {
                                "arm": arm,
                                "encoder_seed": seed,
                                "epoch": 20,
                                "readout": readout,
                                "historical_pooling": historical_pool,
                                "subspace": subspace,
                                "dimension": dimension,
                                "target_index": target_index,
                                "target": target,
                                "target_block": block_of(target),
                                "recomputed_r2": observed,
                                "historical_r2": expected,
                                "absolute_error": difference,
                                "tolerance": tolerance,
                                "passed": bool(difference <= tolerance),
                            }
                        )
    frame = pd.DataFrame(rows)
    if len(frame) != 3 * 3 * 2 * 3 * 22:
        raise ExperimentIntegrityError("T2 reproduction inventory differs")
    summary_rows: list[dict[str, Any]] = []
    independent = frame["target_block"] == "dir"
    per_seed = (
        frame.loc[independent]
        .groupby(["arm", "encoder_seed", "readout", "subspace"], as_index=False)[
            "recomputed_r2"
        ]
        .mean()
    )
    aggregate = (
        per_seed.groupby(["arm", "readout", "subspace"], as_index=False)[
            "recomputed_r2"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    historical_mapping = {
        (readout, subspace): READOUTS[readout][
            {
                "full": "historical_full",
                "common": "historical_common",
                "complement": "historical_complement",
            }[subspace]
        ]
        for readout in READOUTS
        for subspace in ("full", "common", "complement")
    }
    for record in aggregate.to_dict("records"):
        historical_pool = historical_mapping[(record["readout"], record["subspace"])]
        dimension = {"full": 512, "common": 128, "complement": 384}[
            record["subspace"]
        ]
        ref = aggregate_reference.loc[
            (aggregate_reference["arm"] == record["arm"])
            & (aggregate_reference["epoch"] == 20)
            & (aggregate_reference["pooling"] == historical_pool)
            & (aggregate_reference["block"] == "dir")
            & (aggregate_reference["m"] == dimension)
        ]
        if len(ref) != 1:
            raise ExperimentIntegrityError("historical aggregate reproduction cell missing")
        expected = float(ref.iloc[0]["r2_mean"])
        error = abs(float(record["mean"]) - expected)
        summary_rows.append(
            {
                "arm": record["arm"],
                "readout": record["readout"],
                "subspace": record["subspace"],
                "dimension": dimension,
                "recomputed_r2_mean": float(record["mean"]),
                "historical_r2_mean": expected,
                "absolute_error": error,
                "n_encoder_seeds": int(record["count"]),
                "passed": bool(error <= tolerance),
            }
        )
    summary = pd.DataFrame(summary_rows)
    passed = bool(frame["passed"].all() and summary["passed"].all())
    atomic_write_parquet(frame, output_root / "reproduction_per_target.parquet")
    atomic_write_parquet(summary, output_root / "reproduction_summary.parquet")
    payload = {
        "schema_name": "thesis.experiment01.token_role.reproduction_gate",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "passed" if passed else "failed",
        "tolerance": tolerance,
        "n_per_target_cells": len(frame),
        "n_aggregate_cells": len(summary),
        "max_per_target_absolute_error": float(frame["absolute_error"].max()),
        "max_aggregate_absolute_error": float(summary["absolute_error"].max()),
        "runtime_seconds": time.perf_counter() - started,
        "reproduction_per_target_sha256": sha256_file(
            output_root / "reproduction_per_target.parquet"
        ),
        "reproduction_summary_sha256": sha256_file(
            output_root / "reproduction_summary.parquet"
        ),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(output_root / "reproduction_gate.json", payload)
    if not passed:
        raise ExperimentIntegrityError(
            f"T2 reproduction failed; maximum per-target error "
            f"{payload['max_per_target_absolute_error']:.3e}"
        )
    return payload


def _require_reproduction(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "reproduction_gate.json"
    if not path.is_file():
        raise ExperimentIntegrityError("T2 reproduction gate is missing")
    payload = json.loads(path.read_text())
    if payload.get("status") != "passed":
        raise ExperimentIntegrityError("T2 reproduction gate did not pass")
    for name, field in (
        ("reproduction_per_target.parquet", "reproduction_per_target_sha256"),
        ("reproduction_summary.parquet", "reproduction_summary_sha256"),
    ):
        if sha256_file(output_dir / name) != payload[field]:
            raise ExperimentIntegrityError("T2 reproduction artifact changed")
    return payload


def _rank_from_gram(gram: np.ndarray, n_rows: int) -> tuple[int, float]:
    values = np.linalg.eigvalsh((gram + gram.T) * 0.5)
    largest = max(float(values[-1]), 0.0) if values.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(n_rows, len(values)) * largest
    return int(np.sum(values > tolerance)), float(tolerance)


def _basis_geometry(
    stats: LinearStats,
    basis: np.ndarray,
    eigenvectors: np.ndarray,
    signal: np.ndarray,
    *,
    n_train: int,
) -> dict[str, Any]:
    _validate_basis(basis)
    projected_gram = basis.T @ stats.gram_train @ basis
    rank, rank_tolerance = _rank_from_gram(projected_gram, n_train)
    total_trace = float(np.trace(stats.gram_train))
    captured_trace = float(np.trace(projected_gram))
    trace_fraction = (
        captured_trace / total_trace if total_trace > np.finfo(float).tiny else np.nan
    )
    dimension = basis.shape[1]
    top = eigenvectors[:, :dimension]
    pca_overlap = float(np.square(top.T @ basis).sum() / dimension)
    signal_rank = signal.shape[1]
    signal_energy = (
        float(np.square(basis.T @ signal).sum() / signal_rank)
        if signal_rank
        else np.nan
    )
    return {
        "dimension": dimension,
        "numerical_rank": rank,
        "rank_tolerance": rank_tolerance,
        "trace_fraction": trace_fraction,
        "top_pca_overlap": pca_overlap,
        "directional_signal_rank": signal_rank,
        "directional_signal_energy": signal_energy,
    }


def _evaluate_basis_rows(
    stats: LinearStats,
    basis: np.ndarray,
    *,
    full_r2: np.ndarray,
    names: Sequence[str],
    arm: str,
    encoder_seed: int,
    readout: str,
    null_family: str,
    draw_id: int | None,
    subspace: str,
    eigenvectors: np.ndarray,
    signal: np.ndarray,
    n_train: int,
    shuffled: bool = False,
) -> list[dict[str, Any]]:
    geometry = _basis_geometry(
        stats, basis, eigenvectors, signal, n_train=n_train
    )
    values = r2_from_basis(stats, basis)
    if not np.isfinite(values).all():
        raise ExperimentIntegrityError("non-finite projected T2 R2")
    rows = []
    status = (
        "valid"
        if geometry["numerical_rank"] == geometry["dimension"]
        else "valid_rank_deficient"
    )
    rank_note = "" if status == "valid" else "numerical_rank_below_dimension"
    for index, (name, value, full) in enumerate(zip(names, values, full_r2)):
        retention = value / full if full > CEILING_THRESHOLD else np.nan
        trace_fraction = geometry["trace_fraction"]
        rows.append(
            {
                "arm": arm,
                "encoder_seed": int(encoder_seed),
                "epoch": 20,
                "readout": readout,
                "null_family": null_family,
                "draw_id": draw_id,
                "subspace": subspace,
                "target_index": index,
                "target": name,
                "target_block": block_of(name),
                "r2": float(value),
                "full_r2": float(full),
                "retention": float(retention) if np.isfinite(retention) else np.nan,
                "r2_per_trace_fraction": (
                    float(value / trace_fraction)
                    if np.isfinite(trace_fraction) and abs(trace_fraction) > 1e-15
                    else np.nan
                ),
                "shuffled_target": bool(shuffled),
                "status": status,
                "failure_reason": rank_note,
                **geometry,
            }
        )
    return rows


def benchmark_token_role(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    n_benchmark_draws: int = 5,
) -> dict[str, Any]:
    """Benchmark the preregistered five-draw structured-null unit."""
    if n_benchmark_draws != 5:
        raise ExperimentIntegrityError("T2 benchmark fixes five structured draws")
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    _require_frozen_protocol(output_root)
    _require_reproduction(output_root)
    _, _, names = _load_targets(input_root)
    direction = [index for index, name in enumerate(names) if block_of(name) == "dir"]
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cached = build_or_load_cached_stats(
        input_root, output_root, "jepa_horizon", 0, "last_concat512"
    )
    cache_seconds = time.perf_counter() - started
    values, vectors = pca_from_stats(cached.stats)
    del values
    signal = signal_basis(cached.stats, direction)
    full = r2_from_basis(cached.stats, np.eye(512))
    draw_started = time.perf_counter()
    checksum = 0.0
    for draw_id in range(n_benchmark_draws):
        bases = structured_role_bases(draw_id)
        for subspace in ("common", "complement"):
            result = _evaluate_basis_rows(
                cached.stats,
                bases[subspace],
                full_r2=full,
                names=names,
                arm="jepa_horizon",
                encoder_seed=0,
                readout="last_concat512",
                null_family="structured_role",
                draw_id=draw_id,
                subspace=subspace,
                eigenvectors=vectors,
                signal=signal,
                n_train=cached.n_train,
            )
            checksum += sum(row["r2"] for row in result)
    draw_seconds = time.perf_counter() - draw_started
    repeated = structured_role_bases(0)
    deterministic = all(
        np.array_equal(repeated[key], structured_role_bases(0)[key])
        for key in repeated
    )
    if not deterministic:
        raise ExperimentIntegrityError("T2 structured draw is not deterministic")
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    seconds_per_pair = draw_seconds / n_benchmark_draws
    n_feature_sets = len(ARMS) * len(ENCODER_SEEDS) * len(READOUTS)
    projected_pair_count = n_feature_sets * N_DRAWS * 3
    projected_seconds = seconds_per_pair * projected_pair_count
    row_counts = {
        "token_role_observed": n_feature_sets * 2 * TARGET_DIMENSION,
        "token_role_structured_null": n_feature_sets
        * N_DRAWS
        * 2
        * TARGET_DIMENSION,
        "token_role_generic_null": n_feature_sets
        * N_DRAWS
        * 2
        * TARGET_DIMENSION,
        "token_role_shuffled_control": n_feature_sets
        * (N_DRAWS + 1)
        * 2
        * TARGET_DIMENSION,
        "token_role_commonality": n_feature_sets * TARGET_DIMENSION,
    }
    total_rows = int(sum(row_counts.values()))
    estimated_storage = int(total_rows * 360)
    payload = {
        "schema_name": "thesis.experiment01.token_role.benchmark",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "passed",
        "feature_set": "jepa_horizon/seed0/last_concat512",
        "n_structured_draws": n_benchmark_draws,
        "cache_load_seconds": cache_seconds,
        "draw_seconds": draw_seconds,
        "seconds_per_structured_common_complement_pair": seconds_per_pair,
        "projected_full_grid_seconds_conservative": projected_seconds,
        "projected_full_grid_hours_conservative": projected_seconds / 3600.0,
        "peak_rss_before_bytes": int(rss_before * 1024),
        "peak_rss_after_bytes": int(rss_after * 1024),
        "peak_rss_bytes": int(max(rss_before, rss_after) * 1024),
        "checksum": checksum,
        "deterministic_resume_basis_check": deterministic,
        "estimated_row_counts": row_counts,
        "estimated_total_rows": total_rows,
        "estimated_output_storage_bytes": estimated_storage,
        "note": (
            "Runtime projection counts observed, structured, generic and shuffled "
            "projection pairs conservatively; generic QR overhead is not benchmarked."
        ),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(output_root / "benchmark.json", payload)
    return payload


def _commonality_rows(
    stats: LinearStats,
    *,
    full: np.ndarray,
    common: np.ndarray,
    complement: np.ndarray,
    names: Sequence[str],
    arm: str,
    encoder_seed: int,
    readout: str,
) -> list[dict[str, Any]]:
    intercept = 1.0 - stats.val_y_train_centered_ss / np.maximum(
        stats.val_total_ss, 1e-12
    )
    rows = []
    for index, name in enumerate(names):
        shared = common[index] + complement[index] - full[index] - intercept[index]
        phi_common = 0.5 * (
            (common[index] - intercept[index]) + (full[index] - complement[index])
        )
        phi_complement = 0.5 * (
            (complement[index] - intercept[index]) + (full[index] - common[index])
        )
        rows.append(
            {
                "arm": arm,
                "encoder_seed": encoder_seed,
                "epoch": 20,
                "readout": readout,
                "target_index": index,
                "target": name,
                "target_block": block_of(name),
                "intercept_only_r2": float(intercept[index]),
                "full_r2": float(full[index]),
                "common_r2": float(common[index]),
                "complement_r2": float(complement[index]),
                "full_minus_common": float(full[index] - common[index]),
                "full_minus_complement": float(full[index] - complement[index]),
                "shared_commonality": float(shared),
                "phi_common": float(phi_common),
                "phi_complement": float(phi_complement),
                "shapley_sum": float(phi_common + phi_complement),
                "full_increment_over_intercept": float(full[index] - intercept[index]),
            }
        )
    return rows


def _write_feature_shards(
    output_root: Path,
    feature_key: str,
    frames: Mapping[str, pd.DataFrame],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for label, frame in frames.items():
        path = shard_root / f"{feature_key}__{label}.parquet"
        atomic_write_parquet(frame, path)
        records[label] = {
            "path": str(path.relative_to(output_root)),
            "sha256": sha256_file(path),
            "rows": len(frame),
        }
    complete = {
        "schema_name": "thesis.experiment01.token_role.feature_shard",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "feature_key": feature_key,
        "source_sha256": source_sha256,
        "artifacts": records,
    }
    complete["payload_sha256"] = canonical_json_sha256(complete)
    atomic_write_json(shard_root / f"{feature_key}__complete.json", complete)
    return complete


def _load_complete_feature_shard(
    output_root: Path, feature_key: str, source_sha256: str
) -> dict[str, Any] | None:
    path = output_root / "shards" / f"{feature_key}__complete.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("source_sha256") != source_sha256:
        raise ExperimentIntegrityError("T2 completed feature source differs")
    for record in payload.get("artifacts", {}).values():
        artifact = output_root / record["path"]
        if (
            not artifact.is_file()
            or sha256_file(artifact) != record["sha256"]
            or len(pd.read_parquet(artifact, columns=["arm"])) != int(record["rows"])
        ):
            raise ExperimentIntegrityError("T2 completed feature shard changed")
    return payload


def _empty_failure_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "arm",
            "encoder_seed",
            "readout",
            "null_family",
            "draw_id",
            "subspace",
            "target",
            "failure_reason",
        ]
    )


def run_token_role_nulls(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TokenRoleConfig = TokenRoleConfig(),
) -> dict[str, Any]:
    """Run the observed, structured, generic and shuffled T2 projections."""
    config.validate()
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    protocol = _require_frozen_protocol(output_root)
    _require_reproduction(output_root)
    benchmark_path = output_root / "benchmark.json"
    if not benchmark_path.is_file():
        raise ExperimentIntegrityError("T2 benchmark is missing")
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark.get("status") != "passed" or not benchmark.get(
        "deterministic_resume_basis_check"
    ):
        raise ExperimentIntegrityError("T2 benchmark did not pass")
    _, _, names = _load_targets(input_root)
    directional = [index for index, name in enumerate(names) if block_of(name) == "dir"]
    observed_bases = observed_role_bases()
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    computed = 0
    resumed = 0
    generic_basis_cache: dict[tuple[int, int], np.ndarray] = {}

    for arm in ARMS:
        for encoder_seed in ENCODER_SEEDS:
            source_hash = sha256_file(_readout_path(input_root, arm, encoder_seed))
            for readout in READOUTS:
                feature_key = f"{arm}_seed{encoder_seed}_{readout}"
                existing = _load_complete_feature_shard(
                    output_root, feature_key, source_hash
                )
                if existing is not None:
                    completed.append(existing)
                    resumed += 1
                    continue
                cached = build_or_load_cached_stats(
                    input_root, output_root, arm, encoder_seed, readout
                )
                stats = cached.stats
                shuffled_stats = cached.shuffled_stats()
                _, eigenvectors = pca_from_stats(stats)
                signal = signal_basis(stats, directional)
                full = r2_from_basis(stats, np.eye(512))
                shuffled_full = r2_from_basis(shuffled_stats, np.eye(512))

                observed_rows: list[dict[str, Any]] = []
                structured_rows: list[dict[str, Any]] = []
                generic_rows: list[dict[str, Any]] = []
                shuffled_rows: list[dict[str, Any]] = []
                observed_values: dict[str, np.ndarray] = {}
                try:
                    for subspace, basis in observed_bases.items():
                        normal = _evaluate_basis_rows(
                            stats,
                            basis,
                            full_r2=full,
                            names=names,
                            arm=arm,
                            encoder_seed=encoder_seed,
                            readout=readout,
                            null_family="observed_hadamard",
                            draw_id=None,
                            subspace=subspace,
                            eigenvectors=eigenvectors,
                            signal=signal,
                            n_train=cached.n_train,
                        )
                        observed_rows.extend(normal)
                        observed_values[subspace] = np.asarray(
                            [row["r2"] for row in normal]
                        )
                        shuffled_rows.extend(
                            _evaluate_basis_rows(
                                shuffled_stats,
                                basis,
                                full_r2=shuffled_full,
                                names=names,
                                arm=arm,
                                encoder_seed=encoder_seed,
                                readout=readout,
                                null_family="observed_hadamard",
                                draw_id=None,
                                subspace=subspace,
                                eigenvectors=eigenvectors,
                                signal=signal,
                                n_train=cached.n_train,
                                shuffled=True,
                            )
                        )
                    for draw_id in range(config.n_draws):
                        structured = structured_role_bases(
                            draw_id, base_seed=config.base_seed
                        )
                        for subspace, basis in structured.items():
                            structured_rows.extend(
                                _evaluate_basis_rows(
                                    stats,
                                    basis,
                                    full_r2=full,
                                    names=names,
                                    arm=arm,
                                    encoder_seed=encoder_seed,
                                    readout=readout,
                                    null_family="structured_role",
                                    draw_id=draw_id,
                                    subspace=subspace,
                                    eigenvectors=eigenvectors,
                                    signal=signal,
                                    n_train=cached.n_train,
                                )
                            )
                            shuffled_rows.extend(
                                _evaluate_basis_rows(
                                    shuffled_stats,
                                    basis,
                                    full_r2=shuffled_full,
                                    names=names,
                                    arm=arm,
                                    encoder_seed=encoder_seed,
                                    readout=readout,
                                    null_family="structured_role",
                                    draw_id=draw_id,
                                    subspace=subspace,
                                    eigenvectors=eigenvectors,
                                    signal=signal,
                                    n_train=cached.n_train,
                                    shuffled=True,
                                )
                            )
                        for subspace, dimension in (
                            ("common", 128),
                            ("complement", 384),
                        ):
                            generic_key = (draw_id, dimension)
                            if generic_key not in generic_basis_cache:
                                generic_basis_cache[generic_key] = generic_feature_basis(
                                    draw_id,
                                    dimension,
                                    base_seed=config.base_seed,
                                )
                            generic_rows.extend(
                                _evaluate_basis_rows(
                                    stats,
                                    generic_basis_cache[generic_key],
                                    full_r2=full,
                                    names=names,
                                    arm=arm,
                                    encoder_seed=encoder_seed,
                                    readout=readout,
                                    null_family="generic_feature",
                                    draw_id=draw_id,
                                    subspace=subspace,
                                    eigenvectors=eigenvectors,
                                    signal=signal,
                                    n_train=cached.n_train,
                                )
                            )
                except Exception as exc:
                    failures.append(
                        {
                            "arm": arm,
                            "encoder_seed": encoder_seed,
                            "readout": readout,
                            "null_family": "feature_set",
                            "draw_id": None,
                            "subspace": "",
                            "target": "",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise
                commonality = _commonality_rows(
                    stats,
                    full=full,
                    common=observed_values["common"],
                    complement=observed_values["complement"],
                    names=names,
                    arm=arm,
                    encoder_seed=encoder_seed,
                    readout=readout,
                )
                frames = {
                    "observed": pd.DataFrame(observed_rows),
                    "structured": pd.DataFrame(structured_rows),
                    "generic": pd.DataFrame(generic_rows),
                    "shuffled": pd.DataFrame(shuffled_rows),
                    "commonality": pd.DataFrame(commonality),
                }
                feature_complete = _write_feature_shards(
                    output_root,
                    feature_key,
                    frames,
                    source_sha256=source_hash,
                )
                completed.append(feature_complete)
                computed += 1

    if failures:
        failure_frame = pd.DataFrame(failures)
    else:
        failure_frame = _empty_failure_frame()
    atomic_write_parquet(failure_frame, output_root / "token_role_failures.parquet")
    if len(completed) != 18:
        raise ExperimentIntegrityError("T2 feature-set completion count differs")

    labels = {
        "observed": "token_role_observed.parquet",
        "structured": "token_role_structured_null.parquet",
        "generic": "token_role_generic_null.parquet",
        "shuffled": "token_role_shuffled_control.parquet",
        "commonality": "token_role_commonality.parquet",
    }
    consolidated: dict[str, Any] = {}
    for label, filename in labels.items():
        parts = [
            pd.read_parquet(output_root / item["artifacts"][label]["path"])
            for item in completed
        ]
        frame = pd.concat(parts, ignore_index=True)
        atomic_write_parquet(frame, output_root / filename)
        consolidated[label] = {
            "path": filename,
            "rows": len(frame),
            "sha256": sha256_file(output_root / filename),
        }
    expected_rows = {
        "observed": 18 * 2 * 22,
        "structured": 18 * 100 * 2 * 22,
        "generic": 18 * 100 * 2 * 22,
        "shuffled": 18 * 101 * 2 * 22,
        "commonality": 18 * 22,
    }
    for label, count in expected_rows.items():
        if consolidated[label]["rows"] != count:
            raise ExperimentIntegrityError(f"T2 {label} row inventory differs")
    payload = {
        "schema_name": "thesis.experiment01.token_role.run",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "passed" if failure_frame.empty else "failed",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "runtime_seconds": time.perf_counter() - started,
        "feature_sets_computed": computed,
        "feature_sets_resumed": resumed,
        "feature_sets_complete": len(completed),
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "artifacts": consolidated,
        "failures": len(failure_frame),
        "failure_table_sha256": sha256_file(
            output_root / "token_role_failures.parquet"
        ),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(output_root / "run_metadata.json", payload)
    if not failure_frame.empty:
        raise ExperimentIntegrityError("T2 null grid contains failures")
    return payload


def plus_one_probability(
    observed: float, null_values: Sequence[float], *, tail: str
) -> tuple[float, float, int]:
    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or values.size != N_DRAWS or not np.isfinite(values).all():
        raise ExperimentIntegrityError("T2 null distribution is incomplete")
    if tail == "lower":
        count = int(np.sum(values <= observed))
    elif tail == "upper":
        count = int(np.sum(values >= observed))
    else:
        raise ValueError("tail must be lower or upper")
    return (1.0 + count) / (1.0 + len(values)), count / len(values), count


def _null_trend_rows(
    observed: pd.DataFrame, null: pd.DataFrame, null_family: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouping = ["arm", "encoder_seed", "readout", "subspace", "target_index", "target"]
    for keys, group in null.groupby(grouping, sort=True):
        x = group["trace_fraction"].to_numpy(dtype=np.float64)
        y = group["r2"].to_numpy(dtype=np.float64)
        if len(group) != N_DRAWS:
            raise ExperimentIntegrityError("T2 target-level null draw count differs")
        design = np.column_stack([np.ones(len(x)), x])
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        selected = observed.loc[
            (observed["arm"] == keys[0])
            & (observed["encoder_seed"] == keys[1])
            & (observed["readout"] == keys[2])
            & (observed["subspace"] == keys[3])
            & (observed["target_index"] == keys[4])
        ]
        if len(selected) != 1:
            raise ExperimentIntegrityError("T2 observed/null target pairing differs")
        item = selected.iloc[0]
        fitted = float(coefficients[0] + coefficients[1] * item["trace_fraction"])
        rows.append(
            {
                "level": "target_trace_trend",
                "null_family": null_family,
                "arm": keys[0],
                "encoder_seed": int(keys[1]),
                "readout": keys[2],
                "subspace": keys[3],
                "target_index": int(keys[4]),
                "target": keys[5],
                "target_block": item["target_block"],
                "observed_r2": float(item["r2"]),
                "observed_trace_fraction": float(item["trace_fraction"]),
                "null_trace_intercept": float(coefficients[0]),
                "null_trace_slope": float(coefficients[1]),
                "observed_trace_fitted_r2": fitted,
                "observed_trace_residual": float(item["r2"] - fitted),
                "null_draws": len(group),
                "tail": "",
                "plus_one_p": np.nan,
                "raw_percentile": np.nan,
                "tail_count": np.nan,
                "classification": "descriptive",
            }
        )
    return rows


def summarize_token_role(output_dir: str | Path) -> dict[str, Any]:
    """Create null probabilities, decision rules and the frozen T2 summary."""
    root = Path(output_dir)
    protocol = _require_frozen_protocol(root)
    reproduction = _require_reproduction(root)
    run_path = root / "run_metadata.json"
    if not run_path.is_file():
        raise ExperimentIntegrityError("T2 run metadata is missing")
    run = json.loads(run_path.read_text())
    if run.get("status") != "passed" or run.get("failures") != 0:
        raise ExperimentIntegrityError("T2 run did not pass")
    for record in run["artifacts"].values():
        if sha256_file(root / record["path"]) != record["sha256"]:
            raise ExperimentIntegrityError("T2 null artifact changed before summary")
    observed = pd.read_parquet(root / "token_role_observed.parquet")
    structured = pd.read_parquet(root / "token_role_structured_null.parquet")
    generic = pd.read_parquet(root / "token_role_generic_null.parquet")
    shuffled = pd.read_parquet(root / "token_role_shuffled_control.parquet")
    independent_observed = observed.loc[observed["target_block"] == "dir"].copy()
    independent_structured = structured.loc[structured["target_block"] == "dir"].copy()
    independent_generic = generic.loc[generic["target_block"] == "dir"].copy()
    group_keys = ["arm", "encoder_seed", "readout", "subspace"]
    observed_block = independent_observed.groupby(group_keys, as_index=False).agg(
        observed_r2=("r2", "mean"),
        observed_trace_fraction=("trace_fraction", "first"),
        observed_signal_energy=("directional_signal_energy", "first"),
    )
    rows: list[dict[str, Any]] = []
    for family, null_frame in (
        ("structured_role", independent_structured),
        ("generic_feature", independent_generic),
    ):
        block_null = (
            null_frame.groupby(group_keys + ["draw_id"], as_index=False)
            .agg(
                null_r2=("r2", "mean"),
                null_trace_fraction=("trace_fraction", "first"),
                null_signal_energy=("directional_signal_energy", "first"),
            )
        )
        for observed_item in observed_block.itertuples(index=False):
            selected = block_null.loc[
                (block_null["arm"] == observed_item.arm)
                & (block_null["encoder_seed"] == observed_item.encoder_seed)
                & (block_null["readout"] == observed_item.readout)
                & (block_null["subspace"] == observed_item.subspace)
            ].sort_values("draw_id")
            tail = "lower" if observed_item.subspace == "common" else "upper"
            p_value, percentile, count = plus_one_probability(
                observed_item.observed_r2,
                selected["null_r2"].to_numpy(),
                tail=tail,
            )
            trend_design = np.column_stack(
                [np.ones(len(selected)), selected["null_trace_fraction"].to_numpy()]
            )
            trend = np.linalg.lstsq(
                trend_design, selected["null_r2"].to_numpy(), rcond=None
            )[0]
            fitted = float(
                trend[0] + trend[1] * observed_item.observed_trace_fraction
            )
            rows.append(
                {
                    "level": "directional_block",
                    "null_family": family,
                    "arm": observed_item.arm,
                    "encoder_seed": int(observed_item.encoder_seed),
                    "readout": observed_item.readout,
                    "subspace": observed_item.subspace,
                    "target_index": np.nan,
                    "target": "directional_independent_mean",
                    "target_block": "dir",
                    "observed_r2": float(observed_item.observed_r2),
                    "observed_trace_fraction": float(
                        observed_item.observed_trace_fraction
                    ),
                    "observed_signal_energy": float(
                        observed_item.observed_signal_energy
                    ),
                    "null_r2_mean": float(selected["null_r2"].mean()),
                    "null_r2_std": float(selected["null_r2"].std(ddof=0)),
                    "null_r2_min": float(selected["null_r2"].min()),
                    "null_r2_max": float(selected["null_r2"].max()),
                    "null_trace_intercept": float(trend[0]),
                    "null_trace_slope": float(trend[1]),
                    "observed_trace_fitted_r2": fitted,
                    "observed_trace_residual": float(
                        observed_item.observed_r2 - fitted
                    ),
                    "null_draws": len(selected),
                    "tail": tail,
                    "plus_one_p": p_value,
                    "raw_percentile": percentile,
                    "tail_count": count,
                    "classification": (
                        "unusually_weak"
                        if tail == "lower" and p_value <= 0.05
                        else "unusually_strong"
                        if tail == "upper" and p_value <= 0.05
                        else "typical_or_mixed"
                    ),
                }
            )
        rows.extend(_null_trend_rows(observed, null_frame, family))
    summary_frame = pd.DataFrame(rows)
    atomic_write_parquet(summary_frame, root / "token_role_null_summary.parquet")

    block_structured = summary_frame.loc[
        (summary_frame["level"] == "directional_block")
        & (summary_frame["null_family"] == "structured_role")
    ]
    decisions = []
    for arm in ARMS:
        for readout in READOUTS:
            group = block_structured.loc[
                (block_structured["arm"] == arm)
                & (block_structured["readout"] == readout)
            ]
            common = group.loc[group["subspace"] == "common"]
            complement = group.loc[group["subspace"] == "complement"]
            if len(common) != 3 or len(complement) != 3:
                raise ExperimentIntegrityError("T2 seed-level decision inventory differs")
            weak = bool((common["plus_one_p"] <= 0.05).all())
            strong = bool((complement["plus_one_p"] <= 0.05).all())
            decisions.append(
                {
                    "arm": arm,
                    "readout": readout,
                    "common_unusually_weak_all_seeds": weak,
                    "complement_unusually_strong_all_seeds": strong,
                    "joint_pattern_all_seeds": bool(weak and strong),
                    "common_p_by_seed": {
                        str(int(row.encoder_seed)): float(row.plus_one_p)
                        for row in common.itertuples(index=False)
                    },
                    "complement_p_by_seed": {
                        str(int(row.encoder_seed)): float(row.plus_one_p)
                        for row in complement.itertuples(index=False)
                    },
                }
            )
    shuffled_directional = shuffled.loc[shuffled["target_block"] == "dir"]
    shuffled_summary = (
        shuffled_directional.groupby(
            ["arm", "encoder_seed", "readout", "null_family", "subspace"],
            dropna=False,
        )["r2"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )
    atomic_write_parquet(
        shuffled_summary, root / "token_role_shuffled_summary.parquet"
    )
    payload: dict[str, Any] = {
        "schema_name": "thesis.experiment01.token_role.summary",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "complete",
        "scientific_status": (
            "post-hoc corrective diagnostic; no Phase-I or Phase-III reclassification"
        ),
        "phase1_outcome": "A1 unchanged",
        "phase3_outcome": "R3 unchanged",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "reproduction_payload_sha256": reproduction["payload_sha256"],
        "decisions": decisions,
        "n_failures": int(len(pd.read_parquet(root / "token_role_failures.parquet"))),
        "artifacts": {
            "token_role_null_summary.parquet": sha256_file(
                root / "token_role_null_summary.parquet"
            ),
            "token_role_shuffled_summary.parquet": sha256_file(
                root / "token_role_shuffled_summary.parquet"
            ),
        },
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(root / "token_role_summary.json", payload)
    return payload


def _format_p(value: float) -> str:
    return f"{value:.4f}"


def write_token_role_report(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    summary_path = root / "token_role_summary.json"
    if not summary_path.is_file():
        raise ExperimentIntegrityError("T2 summary is missing")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        raise ExperimentIntegrityError("T2 summary is incomplete")
    null_summary = pd.read_parquet(root / "token_role_null_summary.parquet")
    block = null_summary.loc[
        (null_summary["level"] == "directional_block")
        & (null_summary["null_family"] == "structured_role")
    ].sort_values(["readout", "arm", "encoder_seed", "subspace"])
    table_lines = [
        "| readout | arm | seed | block | observed R² | null mean±sd | p | trace fraction | trace residual |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in block.itertuples(index=False):
        table_lines.append(
            f"| {row.readout} | {row.arm} | {int(row.encoder_seed)} | "
            f"{row.subspace} | {row.observed_r2:.6f} | "
            f"{row.null_r2_mean:.6f}±{row.null_r2_std:.6f} | "
            f"{_format_p(row.plus_one_p)} | {row.observed_trace_fraction:.4f} | "
            f"{row.observed_trace_residual:.6f} |"
        )
    decision_lines = []
    for decision in summary["decisions"]:
        decision_lines.append(
            f"- `{decision['arm']}/{decision['readout']}`: common weak in all seeds = "
            f"`{str(decision['common_unusually_weak_all_seeds']).lower()}`; "
            f"complement strong in all seeds = "
            f"`{str(decision['complement_unusually_strong_all_seeds']).lower()}`."
        )
    report = f"""# Experiment 01 — T2 token-role matched-null diagnostic

## Status and scope

T2 is complete as a preregistered corrective, post-hoc diagnostic. It uses the
historical 100,000/50,000 train/validation endpoint sample, not the production
test split. Phase-I technical outcome `A1` and Phase-III-R outcome `R3` remain
unchanged.

The operator under test acts on the four contextual role tokens after choosing
either `last_concat512` or `meanK_concatS`. It is distinct from temporal
averaging and from the PCA decomposition.

## Historical reproduction gate

The mandatory full/common/complement gate passed with maximum per-target error
`{json.loads((root / 'reproduction_gate.json').read_text())['max_per_target_absolute_error']:.3e}`
at tolerance `{REPRODUCTION_TOLERANCE:.1e}`. This reproduces all 1,188 historical
per-target cells before evaluating any random subspace.

## Primary structured role-space null

The table reports the unweighted mean across the 12 independent directional
targets within each encoder seed. `p` is the preregistered plus-one lower-tail
probability for the 128D common block and upper-tail probability for the 384D
complement, each against 100 matched role-Haar draws.

{chr(10).join(table_lines)}

## Frozen decision rule

{chr(10).join(decision_lines)}

Only the structured role-space null controls the primary conclusion. The
generic 512D null, trace-conditioned residuals, PCA overlap, signal-span energy,
commonality/Shapley rows and shuffled-target results are serialized as
secondary diagnostics. The independently fitted common and complement R²
values are not additive information components.

## Interpretation boundary

The diagnostic can establish whether the fixed all-ones role direction and its
zero-sum complement are exceptional among matched role directions. It cannot
attribute the pattern causally to JEPA versus supervised objectives, because
the supervised pretraining arm saw target-aligned labels. It also cannot merge
role projection, temporal pooling and PCA anti-alignment into one mechanism.

## Artifacts

- `token_role_observed.parquet`
- `token_role_structured_null.parquet`
- `token_role_generic_null.parquet`
- `token_role_commonality.parquet`
- `token_role_null_summary.parquet`
- `token_role_shuffled_control.parquet`
- `token_role_failures.parquet`
- `token_role_summary.json`
- `manifest.json`
"""
    atomic_write_text(root / "REPORT_EXPERIMENT_01_TOKEN_ROLE.md", report)
    return {
        "path": "REPORT_EXPERIMENT_01_TOKEN_ROLE.md",
        "sha256": sha256_file(root / "REPORT_EXPERIMENT_01_TOKEN_ROLE.md"),
    }


def _generate_token_role_figures(output_dir: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    summary = pd.read_parquet(output_dir / "token_role_null_summary.parquet")
    block = summary.loc[
        (summary["level"] == "directional_block")
        & (summary["null_family"] == "structured_role")
    ]
    figure_root = output_dir / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    records = []
    for readout in READOUTS:
        selected = block.loc[block["readout"] == readout]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for axis, subspace in zip(axes, ("common", "complement")):
            values = selected.loc[selected["subspace"] == subspace]
            x = np.arange(len(values))
            axis.errorbar(
                x,
                values["null_r2_mean"],
                yerr=values["null_r2_std"],
                fmt="o",
                color="0.5",
                label="role-Haar null mean±sd",
            )
            axis.scatter(x, values["observed_r2"], color="tab:red", label="Hadamard")
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_title(subspace)
            axis.set_xticks(x)
            axis.set_xticklabels(
                [f"{arm}\ns{seed}" for arm, seed in zip(values["arm"], values["encoder_seed"])],
                rotation=45,
                ha="right",
            )
            axis.set_ylabel("directional block validation R²")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"T2 structured role null — {readout}")
        fig.tight_layout()
        path = figure_root / f"01_structured_null_{readout}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        records.append(
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
        )
    return records


def finalize_token_role(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    summary = summarize_token_role(root)
    figures = _generate_token_role_figures(root)
    report = write_token_role_report(root)
    required = [
        "protocol_frozen.json",
        "input_gate.json",
        "reproduction_per_target.parquet",
        "reproduction_summary.parquet",
        "reproduction_gate.json",
        "benchmark.json",
        "run_metadata.json",
        "token_role_observed.parquet",
        "token_role_structured_null.parquet",
        "token_role_generic_null.parquet",
        "token_role_commonality.parquet",
        "token_role_null_summary.parquet",
        "token_role_shuffled_control.parquet",
        "token_role_shuffled_summary.parquet",
        "token_role_failures.parquet",
        "token_role_summary.json",
        "REPORT_EXPERIMENT_01_TOKEN_ROLE.md",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ExperimentIntegrityError(f"T2 final artifact inventory missing: {missing}")
    failures = pd.read_parquet(root / "token_role_failures.parquet")
    if len(failures):
        raise ExperimentIntegrityError("T2 failure table is non-empty")
    artifacts = {
        name: {
            "sha256": sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in required
    }
    for cache_root in (root / "sufficient_statistics", root / "shards"):
        for path in sorted(cache_root.glob("*")):
            if path.is_file():
                relative = str(path.relative_to(root))
                artifacts[relative] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
    for figure in figures:
        path = root / figure["path"]
        artifacts[figure["path"]] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    repo = _repo_root()
    sources = [
        "experiment01/token_role.py",
        "scripts/experiment01/run_experiment_01_token_role.py",
        "tests/test_experiment01_token_role.py",
        SPEC_RELATIVE_PATH,
    ]
    manifest: dict[str, Any] = {
        "schema_name": "thesis.experiment01.token_role.manifest",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "complete",
        "scientific_status": summary["scientific_status"],
        "phase1_outcome": "A1 unchanged",
        "phase3_outcome": "R3 unchanged",
        "specification_sha256": SPEC_SHA256,
        "protocol_payload_sha256": summary["protocol_payload_sha256"],
        "summary_payload_sha256": summary["payload_sha256"],
        "artifacts": artifacts,
        "sources": {name: sha256_file(repo / name) for name in sources},
        "report": report,
        "figures": figures,
    }
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(root / "manifest.json", manifest)
    digest = sha256_file(root / "manifest.json")
    atomic_write_text(root / "manifest.sha256", f"{digest}  manifest.json\n")
    return {
        "status": "complete",
        "manifest_file_sha256": digest,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "summary": summary,
    }

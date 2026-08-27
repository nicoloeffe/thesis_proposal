"""One-time fixed-test evaluation and grouped uncertainty for F16."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader

from experiment01.f16 import BUDGETS, F16IntegrityError, _relative
from experiment01.f16_convergence import (
    _array_from_record,
    _feature_array,
    projection_stats,
    role_projections,
)
from experiment01.f16_evaluation import (
    ANCHOR_ARMS,
    READOUTS,
    _load_f16_encoder,
    _poolings,
    _read_json,
    _source_inventory as evaluation_source_inventory,
    _stats_arrays,
    _stats_from_npz,
    _target_blocks,
    _verify_checkpoint_manifest,
    _verify_parquet,
    load_feature_stats,
)
from experiment01.io import (
    atomic_savez,
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_sha256,
    sha256_array,
    sha256_file,
)
from experiment01.linear import (
    SufficientStats,
    eigensystem,
    evaluate_stats,
    fit_alpha,
    select_targets,
    transformed_design,
)
from experiment01.reference.extract_readouts_multiseed import RawWindowDataset


TEST_SCHEMA_VERSION = 1
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20260827
TEST_SOURCE_FILES = (
    "experiment01/f16_test.py",
    "experiment01/f16_reporting.py",
    "experiment01/f16_evaluation.py",
    "experiment01/f16_convergence.py",
    "experiment01/linear.py",
    "training/train_supervised_grid.py",
    "training/train_jepa_horizon.py",
    "training/train_tokenizer_t.py",
)


def _test_source_inventory(repo_root: Path) -> dict[str, Any]:
    files = {}
    for relative in TEST_SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise F16IntegrityError(f"missing F16 test source: {relative}")
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {
        "files": files,
        "fingerprint": canonical_json_sha256(
            {key: value["sha256"] for key, value in sorted(files.items())}
        ),
    }


def _verify_selection_manifest(repo_root: Path, output_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    path = output_root / "f16_validation_selection_manifest.json"
    manifest = _read_json(path)
    unsigned = dict(manifest)
    fingerprint = unsigned.pop("manifest_fingerprint", None)
    if fingerprint != canonical_json_sha256(unsigned):
        raise F16IntegrityError("F16 validation selection manifest fingerprint drift")
    if (
        manifest.get("status") != "validation_selections_frozen_test_locked"
        or manifest.get("test_barrier") != "locked"
        or manifest.get("test_accessed") is not False
        or manifest.get("failures")
    ):
        raise F16IntegrityError("F16 validation selections are not frozen before test")
    for record in manifest["artifacts"].values():
        artifact = repo_root / record["path"]
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise F16IntegrityError(f"F16 validation selection artifact drift: {artifact}")
    selection_record = manifest["artifacts"]["selection_table"]
    selections = pd.read_parquet(repo_root / selection_record["path"])
    if len(selections) != int(selection_record["rows"]):
        raise F16IntegrityError("F16 validation selection table row count drift")
    return manifest, selections


def unlock_f16_test(repo_root: Path, output_root: Path, bundle_root: Path) -> dict[str, Any]:
    """Create the single immutable unlock record after every selection is frozen."""
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    bundle_root = bundle_root.resolve()
    unlock_path = output_root / "f16_test_unlock.json"
    if unlock_path.exists():
        existing = _read_json(unlock_path)
        unsigned = dict(existing)
        fingerprint = unsigned.pop("unlock_fingerprint", None)
        if fingerprint != canonical_json_sha256(unsigned):
            raise F16IntegrityError("existing F16 test unlock fingerprint drift")
        if existing.get("test_source") != _test_source_inventory(repo_root):
            raise F16IntegrityError("F16 test source changed after one-time unlock")
        return existing
    checkpoints, checkpoint_manifest, cohort = _verify_checkpoint_manifest(repo_root, output_root)
    del checkpoints
    selections, selection_table = _verify_selection_manifest(repo_root, output_root)
    del selection_table
    extraction_path = output_root / "f16_validation_extraction_state.json"
    extraction = _read_json(extraction_path)
    unsigned_extraction = {
        key: value for key, value in extraction.items() if key != "manifest_fingerprint"
    }
    if (
        extraction.get("status") != "complete"
        or extraction.get("test_accessed") is not False
        or extraction.get("manifest_fingerprint") != canonical_json_sha256(unsigned_extraction)
    ):
        raise F16IntegrityError("F16 validation extraction is not frozen before unlock")
    if extraction["base_fingerprint"]["evaluation_source_fingerprint"] != evaluation_source_inventory(repo_root)["fingerprint"]:
        raise F16IntegrityError("F16 evaluation source drift before test unlock")
    failures_path = output_root / "f16_failures.parquet"
    if not pd.read_parquet(failures_path).empty:
        raise F16IntegrityError("F16 failure table is non-empty before test unlock")
    test_record = cohort["cohorts"]["test"]
    test_rows_path = repo_root / test_record["path"]
    if sha256_file(test_rows_path) != test_record["sha256"]:
        raise F16IntegrityError("F16 sealed test row manifest drift")
    if test_record["outcome_arrays_accessed_during_selection"] is not False:
        raise F16IntegrityError("F16 test cohort selection used outcomes")
    bundle_manifest_path = bundle_root / "manifest.json"
    protocol = _read_json(output_root / "f16_manifest.json")
    if sha256_file(bundle_manifest_path) != protocol["bundle_manifest_sha256"]:
        raise F16IntegrityError("F16 bundle drift before test unlock")
    source = _test_source_inventory(repo_root)
    payload: dict[str, Any] = {
        "schema_name": "thesis.experiment01.f16_test_unlock",
        "schema_version": TEST_SCHEMA_VERSION,
        "status": "unlocked_once_for_fixed_evaluation",
        "unlocked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "fixed test evaluation, grouped uncertainty, summary, report and integrity audit only",
        "checkpoint_manifest_sha256": sha256_file(output_root / "f16_checkpoint_manifest.json"),
        "validation_extraction_state_sha256": sha256_file(extraction_path),
        "validation_selection_manifest_sha256": sha256_file(
            output_root / "f16_validation_selection_manifest.json"
        ),
        "failure_table_sha256": sha256_file(failures_path),
        "bundle_manifest_sha256": sha256_file(bundle_manifest_path),
        "test_row_manifest": {
            "path": test_record["path"],
            "sha256": test_record["sha256"],
            "rows": test_record["rows"],
            "stock_days": test_record["stock_days"],
            "row_key_sequence_sha256": test_record["row_key_sequence_sha256"],
            "endpoint_index_sha256": test_record["endpoint_index_sha256"],
        },
        "test_source": source,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "hierarchy": "resample stocks, then stock-days within sampled stocks",
        },
        "selection_changes_permitted": False,
        "second_unlock_permitted": False,
        "failures": [],
    }
    payload["unlock_fingerprint"] = canonical_json_sha256(payload)
    atomic_write_json(unlock_path, payload)
    return payload


def _verify_unlock(repo_root: Path, output_root: Path) -> dict[str, Any]:
    unlock = _read_json(output_root / "f16_test_unlock.json")
    unsigned = dict(unlock)
    fingerprint = unsigned.pop("unlock_fingerprint", None)
    if fingerprint != canonical_json_sha256(unsigned):
        raise F16IntegrityError("F16 test unlock fingerprint drift")
    if unlock.get("status") != "unlocked_once_for_fixed_evaluation":
        raise F16IntegrityError("F16 test is not unlocked for fixed evaluation")
    if unlock.get("selection_changes_permitted") is not False:
        raise F16IntegrityError("F16 unlock improperly permits selection changes")
    if unlock.get("test_source") != _test_source_inventory(repo_root):
        raise F16IntegrityError("F16 test source drift after unlock")
    return unlock


def _load_test_stats(repo_root: Path, record: Mapping[str, Any]) -> dict[str, SufficientStats]:
    path = repo_root / record["path"]
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise F16IntegrityError(f"F16 test sufficient-statistics drift: {path}")
    with np.load(path, allow_pickle=False) as data:
        if str(data["source_fingerprint"].item()) != canonical_json_sha256(record["source_fingerprint"]):
            raise F16IntegrityError(f"F16 test sufficient-statistics fingerprint drift: {path}")
        return {
            readout: _stats_from_npz(
                data, f"{'last' if readout == 'last_concat512' else 'meanK'}_test"
            )
            for readout in READOUTS
        }


def _save_test_stats(
    path: Path,
    stats: Mapping[str, SufficientStats],
    fingerprint: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "source_fingerprint": np.asarray(canonical_json_sha256(fingerprint))
    }
    for readout, value in stats.items():
        prefix = "last" if readout == "last_concat512" else "meanK"
        arrays.update(_stats_arrays(value, f"{prefix}_test"))
    atomic_savez(path, **arrays)
    return {
        "path": _relative(path, repo_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "source_fingerprint": fingerprint,
    }


def _raw_group_models(
    feature: Mapping[str, Any],
    cached,
    selections: pd.DataFrame,
    blocks: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, list[dict[str, Any]]]:
    if feature["checkpoint_kind"] not in {"best", "canonical_epoch20"}:
        return {readout: [] for readout in READOUTS}
    if feature["encoder_family"] not in {"supervised_f16", "jepa_horizon"}:
        return {readout: [] for readout in READOUTS}
    result: dict[str, list[dict[str, Any]]] = {readout: [] for readout in READOUTS}
    selected = selections.loc[
        selections["feature_key"].eq(feature["feature_key"])
        & selections["feature_view"].eq("full")
        & selections["reader_family"].eq("ridge_trace_normalized")
    ]
    for row in selected.itertuples(index=False):
        indices, _independent = blocks[row.target_block]
        train = select_targets(cached[row.readout].budgets[row.analysis_budget], indices)
        model = fit_alpha(transformed_design(train), float(row.alpha))
        identity = {
            "feature_key": feature["feature_key"],
            "encoder_family": feature["encoder_family"],
            "trained_budget": feature["trained_budget"],
            "encoder_seed": int(feature["encoder_seed"]),
            "checkpoint_kind": feature["checkpoint_kind"],
            "readout": row.readout,
            "axis": row.axis,
            "analysis_budget": row.analysis_budget,
            "target_block": row.target_block,
            "alpha": float(row.alpha),
        }
        result[row.readout].append(
            {
                "model_key": canonical_json_sha256(identity),
                "identity": identity,
                "target_indices": indices,
                "beta": model.beta_raw,
                "intercept": model.intercept,
            }
        )
    return result


def _empty_group_accumulators(
    models: Mapping[str, Sequence[Mapping[str, Any]]], n_groups: int
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for readout_models in models.values():
        for model in readout_models:
            target_count = len(model["target_indices"])
            result[model["model_key"]] = {
                "n": np.zeros(n_groups, dtype=np.int64),
                "y_sum": np.zeros((n_groups, target_count), dtype=np.float64),
                "yty": np.zeros((n_groups, target_count), dtype=np.float64),
                "residual_ss": np.zeros((n_groups, target_count), dtype=np.float64),
            }
    return result


def _add_group_batch(
    accumulators: Mapping[str, dict[str, np.ndarray]],
    models: Sequence[Mapping[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> None:
    unique = np.unique(groups)
    for model in models:
        indices = model["target_indices"]
        truth = y[:, indices].astype(np.float64, copy=False)
        prediction = x.astype(np.float64, copy=False) @ model["beta"] + model["intercept"]
        residual = np.square(truth - prediction)
        accumulator = accumulators[model["model_key"]]
        for group in unique:
            mask = groups == group
            accumulator["n"][group] += int(mask.sum())
            accumulator["y_sum"][group] += truth[mask].sum(axis=0)
            accumulator["yty"][group] += np.square(truth[mask]).sum(axis=0)
            accumulator["residual_ss"][group] += residual[mask].sum(axis=0)


def _group_frame(
    models: Mapping[str, Sequence[Mapping[str, Any]]],
    accumulators: Mapping[str, Mapping[str, np.ndarray]],
    groups: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    columns = (
        "model_key",
        "feature_key",
        "encoder_family",
        "trained_budget",
        "encoder_seed",
        "checkpoint_kind",
        "readout",
        "axis",
        "analysis_budget",
        "target_block",
        "alpha",
        "stock_id",
        "stock_symbol",
        "trading_date",
        "stock_day_id",
        "target_index",
        "target_name",
        "target_independent",
        "n_rows",
        "y_sum",
        "yty",
        "residual_ss",
    )
    rows = []
    for readout_models in models.values():
        for model in readout_models:
            accumulator = accumulators[model["model_key"]]
            for group_index, group in groups.iterrows():
                for local, target_index in enumerate(model["target_indices"]):
                    rows.append(
                        {
                            "model_key": model["model_key"],
                            **model["identity"],
                            "stock_id": int(group.stock_id),
                            "stock_symbol": str(group.stock_symbol),
                            "trading_date": str(group.trading_date),
                            "stock_day_id": int(group.stock_day_id),
                            "target_index": int(target_index),
                            "target_name": definitions[int(target_index)]["name"],
                            "target_independent": bool(definitions[int(target_index)]["independent"]),
                            "n_rows": int(accumulator["n"][group_index]),
                            "y_sum": float(accumulator["y_sum"][group_index, local]),
                            "yty": float(accumulator["yty"][group_index, local]),
                            "residual_ss": float(accumulator["residual_ss"][group_index, local]),
                        }
                    )
    return pd.DataFrame(rows, columns=columns)


@torch.inference_mode()
def _extract_new_test(
    checkpoint: Mapping[str, Any],
    *,
    dataset_path: Path,
    test_rows: pd.DataFrame,
    y_test: np.ndarray,
    group_codes: np.ndarray,
    group_table: pd.DataFrame,
    models: Mapping[str, Sequence[Mapping[str, Any]]],
    definitions: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, SufficientStats], pd.DataFrame]:
    stats = {readout: SufficientStats.zeros(512, 23) for readout in READOUTS}
    group_accumulators = _empty_group_accumulators(models, len(group_table))
    with np.load(dataset_path, allow_pickle=False) as raw:
        encoder, stock_stats = _load_f16_encoder(
            Path(checkpoint["absolute_path"]), checkpoint, device
        )
        loader = DataLoader(
            RawWindowDataset(
                raw["book"],
                raw["mid_z"],
                raw["stock_ids"].astype(np.int64, copy=False),
                test_rows["endpoint_index"].to_numpy(dtype=np.int64),
                stock_stats,
                20,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
        cursor = 0
        for book, stock in loader:
            n = len(book)
            features = _poolings(
                encoder(book.to(device, non_blocking=True), stock.to(device, non_blocking=True))
            )
            target = y_test[cursor : cursor + n]
            groups = group_codes[cursor : cursor + n]
            for readout, x in features.items():
                stats[readout].add_rows(x, target)
                _add_group_batch(group_accumulators, models[readout], x, target, groups)
            cursor += n
        if cursor != len(test_rows):
            raise F16IntegrityError("F16 new-encoder test extraction cursor mismatch")
        del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats, _group_frame(models, group_accumulators, group_table, definitions)


def _extract_anchor_test(
    bundle_root: Path,
    bundle_manifest: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    test_rows: pd.DataFrame,
    y_test: np.ndarray,
    group_codes: np.ndarray,
    group_table: pd.DataFrame,
    models: Mapping[str, Sequence[Mapping[str, Any]]],
    definitions: Sequence[Mapping[str, Any]],
    chunk_rows: int,
) -> tuple[dict[str, SufficientStats], pd.DataFrame]:
    stats = {readout: SufficientStats.zeros(512, 23) for readout in READOUTS}
    group_accumulators = _empty_group_accumulators(models, len(group_table))
    positions = test_rows["source_row_position"].to_numpy(dtype=np.int64)
    for readout in READOUTS:
        array = _feature_array(
            bundle_root,
            bundle_manifest,
            feature["encoder_family"],
            int(feature["encoder_seed"]),
            readout,
            "test",
        )
        for start in range(0, len(test_rows), chunk_rows):
            stop = min(start + chunk_rows, len(test_rows))
            x = np.asarray(array[positions[start:stop]], dtype=np.float32)
            target = y_test[start:stop]
            groups = group_codes[start:stop]
            stats[readout].add_rows(x, target)
            _add_group_batch(group_accumulators, models[readout], x, target, groups)
    return stats, _group_frame(models, group_accumulators, group_table, definitions)


def extract_f16_test_statistics(
    repo_root: Path,
    output_root: Path,
    bundle_root: Path,
    dataset_path: Path,
    *,
    device_name: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 2,
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    bundle_root = bundle_root.resolve()
    dataset_path = dataset_path.resolve()
    unlock = _verify_unlock(repo_root, output_root)
    _checkpoints, _checkpoint_manifest, cohort = _verify_checkpoint_manifest(repo_root, output_root)
    selection_manifest, selections = _verify_selection_manifest(repo_root, output_root)
    extraction = _read_json(output_root / "f16_validation_extraction_state.json")
    bundle_manifest = _read_json(bundle_root / "manifest.json")
    if sha256_file(bundle_root / "manifest.json") != unlock["bundle_manifest_sha256"]:
        raise F16IntegrityError("F16 bundle changed after test unlock")
    if sha256_file(dataset_path) != bundle_manifest["provenance"]["dataset_sha256"]:
        raise F16IntegrityError("F16 dataset drift after test unlock")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise F16IntegrityError("F16 test extraction requested unavailable GPU")
    device = torch.device(device_name)
    definitions = bundle_manifest["targets"]["definitions"]
    blocks = _target_blocks(definitions)
    test_rows = _verify_parquet(repo_root, cohort["cohorts"]["test"])
    targets = _array_from_record(bundle_root, bundle_manifest["targets"]["arrays"]["test"])
    y_test = np.asarray(
        targets[test_rows["source_row_position"].to_numpy(dtype=np.int64)], dtype=np.float32
    )
    if y_test.shape != (len(test_rows), 23) or not np.isfinite(y_test).all():
        raise F16IntegrityError("F16 fixed test targets are invalid")
    group_columns = ["stock_id", "stock_symbol", "trading_date", "stock_day_id"]
    group_table = test_rows[group_columns].drop_duplicates().sort_values(
        ["stock_id", "trading_date"], kind="stable"
    ).reset_index(drop=True)
    group_keys = list(zip(group_table.stock_id.astype(int), group_table.trading_date.astype(str)))
    group_lookup = {key: index for index, key in enumerate(group_keys)}
    group_codes = np.asarray(
        [group_lookup[(int(row.stock_id), str(row.trading_date))] for row in test_rows.itertuples()],
        dtype=np.int64,
    )
    if len(group_table) != 87 or group_table["stock_id"].nunique() != 7:
        raise F16IntegrityError("F16 fixed test grouping is not 87 days across 7 stocks")
    base = {
        "algorithm": "f16_fixed_test_sufficient_and_group_residuals.v1",
        "unlock_sha256": sha256_file(output_root / "f16_test_unlock.json"),
        "selection_manifest_sha256": sha256_file(
            output_root / "f16_validation_selection_manifest.json"
        ),
        "validation_extraction_state_sha256": sha256_file(
            output_root / "f16_validation_extraction_state.json"
        ),
        "test_rows_sha256": cohort["cohorts"]["test"]["sha256"],
        "test_targets_sha256": sha256_array(y_test),
        "test_source_fingerprint": unlock["test_source"]["fingerprint"],
    }
    state_path = output_root / "f16_test_extraction_state.json"
    state = (
        _read_json(state_path)
        if state_path.exists()
        else {
            "schema_name": "thesis.experiment01.f16_test_extraction_state",
            "schema_version": TEST_SCHEMA_VERSION,
            "status": "extracting",
            "base_fingerprint": base,
            "feature_sets": {},
            "test_accessed": True,
        }
    )
    if state.get("base_fingerprint") != base or state.get("test_accessed") is not True:
        raise F16IntegrityError("stale F16 test extraction state")
    cache_root = output_root / "test_sufficient_statistics"
    group_root = output_root / "test_group_residuals"
    cache_root.mkdir(parents=True, exist_ok=True)
    group_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for key, raw_record in sorted(extraction["feature_sets"].items()):
        feature = {"feature_key": key, **raw_record}
        fingerprint = {**base, "feature_key": key, "checkpoint_sha256": feature["checkpoint_sha256"]}
        existing = state["feature_sets"].get(key)
        if isinstance(existing, Mapping):
            stats_path = repo_root / existing["test_stats"]["path"]
            group_path = repo_root / existing["group_residuals"]["path"]
            if (
                existing.get("source_fingerprint") == fingerprint
                and stats_path.is_file()
                and sha256_file(stats_path) == existing["test_stats"]["sha256"]
                and group_path.is_file()
                and sha256_file(group_path) == existing["group_residuals"]["sha256"]
            ):
                print(f"F16 test extraction: skip verified {key}", flush=True)
                continue
        cached = load_feature_stats(repo_root, feature)
        models = _raw_group_models(feature, cached, selections, blocks)
        print(f"F16 test extraction: start {key}", flush=True)
        if feature["encoder_family"] == "supervised_f16":
            checkpoint = dict(feature)
            # The authoritative path comes from the frozen checkpoint inventory.
            inventory = pd.read_parquet(
                repo_root / _read_json(output_root / "f16_checkpoint_manifest.json")["checkpoint_inventory"]["path"]
            )
            matches = inventory.loc[
                inventory["trained_budget"].eq(feature["trained_budget"])
                & inventory["encoder_seed"].eq(int(feature["encoder_seed"]))
                & inventory["checkpoint_kind"].eq(feature["checkpoint_kind"])
            ]
            if len(matches) != 1 or matches.iloc[0]["sha256"] != feature["checkpoint_sha256"]:
                raise F16IntegrityError(f"F16 test checkpoint inventory mismatch: {key}")
            checkpoint["absolute_path"] = str(repo_root / matches.iloc[0]["path"])
            checkpoint["sha256"] = feature["checkpoint_sha256"]
            stats, groups = _extract_new_test(
                checkpoint,
                dataset_path=dataset_path,
                test_rows=test_rows,
                y_test=y_test,
                group_codes=group_codes,
                group_table=group_table,
                models=models,
                definitions=definitions,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            stats, groups = _extract_anchor_test(
                bundle_root,
                bundle_manifest,
                feature,
                test_rows=test_rows,
                y_test=y_test,
                group_codes=group_codes,
                group_table=group_table,
                models=models,
                definitions=definitions,
                chunk_rows=chunk_rows,
            )
        stats_path = cache_root / f"{key}.npz"
        group_path = group_root / f"{key}.parquet"
        stats_record = _save_test_stats(stats_path, stats, fingerprint, repo_root)
        atomic_write_parquet(groups, group_path)
        state["feature_sets"][key] = {
            "encoder_family": feature["encoder_family"],
            "trained_budget": feature["trained_budget"],
            "encoder_seed": int(feature["encoder_seed"]),
            "checkpoint_kind": feature["checkpoint_kind"],
            "checkpoint_sha256": feature["checkpoint_sha256"],
            "source_fingerprint": fingerprint,
            "test_stats": stats_record,
            "group_residuals": {
                "path": _relative(group_path, repo_root),
                "sha256": sha256_file(group_path),
                "size_bytes": group_path.stat().st_size,
                "rows": len(groups),
            },
        }
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(state_path, state)
        peak_rss = max(peak_rss, process.memory_info().rss)
    if len(state["feature_sets"]) != 33:
        raise F16IntegrityError("F16 test extraction did not produce exactly 33 feature sets")
    state["status"] = "complete"
    state["runtime"] = {
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "peak_ram_bytes_this_invocation": max(
            peak_rss, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
        "peak_vram_bytes_this_invocation": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    state["manifest_fingerprint"] = canonical_json_sha256(
        {key: value for key, value in state.items() if key != "manifest_fingerprint"}
    )
    atomic_write_json(state_path, state)
    return state


def _projection(
    feature_view: str,
    whitening_k: float | int | None,
    cached,
):
    if feature_view == "full":
        return None
    if feature_view == "role_common":
        return role_projections()[0]
    if feature_view == "role_contrast":
        return role_projections()[1]
    if feature_view == "whiten_topk":
        if cached.covariance is None:
            raise F16IntegrityError("F16 whitening requested without covariance stats")
        k = int(whitening_k)
        spectrum = eigensystem(cached.covariance.covariance, cached.covariance.n)
        if k > spectrum.diagnostics.numerical_rank:
            raise F16IntegrityError("F16 frozen whitening depth exceeds test feature rank")
        scales = np.ones(512, dtype=np.float64)
        if k:
            scales[:k] = 1.0 / np.sqrt(spectrum.eigenvalues[:k])
        return (spectrum.eigenvectors * scales[None, :]) @ spectrum.eigenvectors.T
    raise F16IntegrityError(f"unknown frozen F16 feature view {feature_view}")


def evaluate_f16_test(repo_root: Path, output_root: Path, bundle_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    bundle_root = bundle_root.resolve()
    _verify_unlock(repo_root, output_root)
    selection_manifest, selections = _verify_selection_manifest(repo_root, output_root)
    del selections
    extraction = _read_json(output_root / "f16_validation_extraction_state.json")
    test_state = _read_json(output_root / "f16_test_extraction_state.json")
    if test_state.get("status") != "complete" or test_state.get("test_accessed") is not True:
        raise F16IntegrityError("F16 test extraction is not complete")
    unsigned = {key: value for key, value in test_state.items() if key != "manifest_fingerprint"}
    if test_state.get("manifest_fingerprint") != canonical_json_sha256(unsigned):
        raise F16IntegrityError("F16 test extraction state fingerprint drift")
    validation_results = pd.read_parquet(
        repo_root / selection_manifest["artifacts"]["validation_results"]["path"]
    )
    bundle_manifest = _read_json(bundle_root / "manifest.json")
    definitions = bundle_manifest["targets"]["definitions"]
    blocks = _target_blocks(definitions)
    output_parts = []
    config_columns = [
        "feature_key",
        "readout",
        "axis",
        "analysis_budget",
        "target_block",
        "feature_view",
        "whitening_k",
        "reader_family",
        "alpha",
    ]
    for feature_key, feature_rows in validation_results.groupby("feature_key", sort=False):
        feature_record = {"feature_key": feature_key, **extraction["feature_sets"][feature_key]}
        cached = load_feature_stats(repo_root, feature_record)
        test_record = test_state["feature_sets"][feature_key]["test_stats"]
        test_stats = _load_test_stats(repo_root, test_record)
        feature_output = feature_rows.copy()
        feature_output["test_r2"] = np.nan
        configs = feature_rows[config_columns].drop_duplicates()
        for config in configs.itertuples(index=False):
            indices, _independent = blocks[config.target_block]
            train = cached[config.readout].budgets[config.analysis_budget]
            test = test_stats[config.readout]
            transform = _projection(config.feature_view, config.whitening_k, cached[config.readout])
            if transform is not None:
                train = projection_stats(train, transform)
                test = projection_stats(test, transform)
            train_block = select_targets(train, indices)
            test_block = select_targets(test, indices)
            model = fit_alpha(transformed_design(train_block), float(config.alpha))
            scores = evaluate_stats(model, test_block)
            mask = np.ones(len(feature_output), dtype=bool)
            for column in config_columns:
                value = getattr(config, column)
                if column == "whitening_k" and pd.isna(value):
                    mask &= feature_output[column].isna().to_numpy()
                else:
                    mask &= feature_output[column].eq(value).to_numpy()
            selected_indices = feature_output.index[mask]
            local_by_global = {int(global_index): local for local, global_index in enumerate(indices)}
            for row_index in selected_indices:
                global_index = int(feature_output.at[row_index, "target_index"])
                feature_output.at[row_index, "test_r2"] = float(scores.values[local_by_global[global_index]])
        if feature_output["test_r2"].isna().any():
            raise F16IntegrityError(f"F16 fixed test results incomplete: {feature_key}")
        feature_output["test_accessed"] = True
        output_parts.append(feature_output)
    results = pd.concat(output_parts, ignore_index=True)
    results_path = output_root / "f16_results.parquet"
    atomic_write_parquet(results, results_path)
    geometry = pd.read_parquet(
        repo_root / selection_manifest["artifacts"]["validation_geometry"]["path"]
    )
    geometry["test_r2"] = np.nan
    geometry["test_retention"] = np.nan
    aggregate = (
        results.loc[
            results["target_independent"].astype(bool)
            & results["reader_family"].eq("ridge_trace_normalized")
            & results["axis"].eq("B_fixed_b16")
        ]
        .groupby(
            ["feature_key", "readout", "target_block", "feature_view", "whitening_k"],
            dropna=False,
            as_index=False,
        )["test_r2"]
        .mean()
    )
    lookup = {
        (
            row.feature_key,
            row.readout,
            row.target_block,
            row.feature_view,
            None if pd.isna(row.whitening_k) else int(row.whitening_k),
        ): float(row.test_r2)
        for row in aggregate.itertuples(index=False)
    }
    for index, row in geometry.iterrows():
        family = row["metric_family"]
        if family == "role_retention":
            subspace = lookup.get((row.feature_key, row.readout, row.target_block, row.feature_view, None))
            full = lookup.get((row.feature_key, row.readout, row.target_block, "full", None))
            if subspace is not None and full is not None:
                geometry.at[index, "test_r2"] = subspace
                geometry.at[index, "test_retention"] = subspace / full if abs(full) >= 1e-12 else np.nan
        elif family == "pooling_loss":
            last = lookup.get((row.feature_key, "last_concat512", row.target_block, "full", None))
            mean = lookup.get((row.feature_key, "meanK_concatS", row.target_block, "full", None))
            if last is not None and mean is not None:
                geometry.at[index, "test_r2"] = last - mean
                geometry.at[index, "test_retention"] = mean / last if abs(last) >= 1e-12 else np.nan
        elif family == "whitening_bridge":
            value = lookup.get((row.feature_key, row.readout, row.target_block, "whiten_topk", int(row.whitening_k)))
            if value is not None:
                geometry.at[index, "test_r2"] = value
    geometry["test_accessed"] = True
    geometry_path = output_root / "f16_geometry.parquet"
    atomic_write_parquet(geometry, geometry_path)
    return results, geometry


def _bootstrap_weights(groups: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stocks = np.sort(groups["stock_id"].unique())
    by_stock = {
        int(stock): groups.index[groups["stock_id"].eq(stock)].to_numpy(dtype=np.int64)
        for stock in stocks
    }
    weights = np.zeros((BOOTSTRAP_DRAWS, len(groups)), dtype=np.int16)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_stocks = rng.choice(stocks, size=len(stocks), replace=True)
        for stock in sampled_stocks:
            days = by_stock[int(stock)]
            sampled_days = rng.choice(days, size=len(days), replace=True)
            np.add.at(weights[draw], sampled_days, 1)
    return weights


def _weighted_block_r2(
    frame: pd.DataFrame,
    weights: np.ndarray,
    group_keys: pd.DataFrame,
) -> np.ndarray:
    targets = np.sort(frame.loc[frame["target_independent"].astype(bool), "target_index"].unique())
    values = []
    for target in targets:
        selected = frame.loc[frame["target_index"].eq(target)].set_index(
            ["stock_id", "trading_date"]
        )
        ordered = selected.reindex(
            pd.MultiIndex.from_frame(group_keys[["stock_id", "trading_date"]])
        )
        if ordered[["n_rows", "y_sum", "yty", "residual_ss"]].isna().any().any():
            raise F16IntegrityError("F16 grouped residual table does not cover every test stock-day")
        n = weights @ ordered["n_rows"].to_numpy(dtype=np.float64)
        y_sum = weights @ ordered["y_sum"].to_numpy(dtype=np.float64)
        yty = weights @ ordered["yty"].to_numpy(dtype=np.float64)
        residual = weights @ ordered["residual_ss"].to_numpy(dtype=np.float64)
        total = yty - np.square(y_sum) / n
        if np.any(total <= 0):
            raise F16IntegrityError("F16 grouped bootstrap produced a constant target")
        values.append(1.0 - residual / total)
    return np.mean(np.column_stack(values), axis=1)


def grouped_f16_uncertainty(repo_root: Path, output_root: Path) -> pd.DataFrame:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    _verify_unlock(repo_root, output_root)
    test_state = _read_json(output_root / "f16_test_extraction_state.json")
    frames = []
    for record in test_state["feature_sets"].values():
        group_record = record["group_residuals"]
        path = repo_root / group_record["path"]
        if sha256_file(path) != group_record["sha256"]:
            raise F16IntegrityError("F16 grouped residual cache drift")
        frame = pd.read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    residuals = pd.concat(frames, ignore_index=True)
    group_keys = residuals[["stock_id", "stock_symbol", "trading_date", "stock_day_id"]].drop_duplicates().sort_values(
        ["stock_id", "trading_date"], kind="stable"
    ).reset_index(drop=True)
    if len(group_keys) != 87:
        raise F16IntegrityError("F16 grouped uncertainty does not cover 87 stock-days")
    bootstrap = _bootstrap_weights(group_keys)
    point_weights = np.ones((1, len(group_keys)), dtype=np.float64)
    rows = []
    for budget in BUDGETS:
        for seed in (0, 1, 2):
            new_feature = f"supervised_f16_{budget}_seed{seed}_best"
            horizon_feature = f"jepa_horizon_seed{seed}_canonical"
            for axis in ("A_label_matched", "B_fixed_b16"):
                analysis_budget = budget if axis == "A_label_matched" else "b_16"
                for readout in READOUTS:
                    for block in ("directional", "volatility", "timing"):
                        common = (
                            residuals["readout"].eq(readout)
                            & residuals["axis"].eq(axis)
                            & residuals["analysis_budget"].eq(analysis_budget)
                            & residuals["target_block"].eq(block)
                        )
                        new = residuals.loc[common & residuals["feature_key"].eq(new_feature)]
                        horizon = residuals.loc[common & residuals["feature_key"].eq(horizon_feature)]
                        if new.empty or horizon.empty:
                            raise F16IntegrityError(
                                f"missing F16 grouped pair {budget}/{seed}/{axis}/{readout}/{block}"
                            )
                        new_point = _weighted_block_r2(new, point_weights, group_keys)[0]
                        horizon_point = _weighted_block_r2(horizon, point_weights, group_keys)[0]
                        draws = _weighted_block_r2(new, bootstrap, group_keys) - _weighted_block_r2(
                            horizon, bootstrap, group_keys
                        )
                        identity = {
                            "trained_budget": budget,
                            "encoder_seed": seed,
                            "axis": axis,
                            "analysis_budget": analysis_budget,
                            "readout": readout,
                            "target_block": block,
                        }
                        rows.append(
                            {
                                **identity,
                                "estimate_type": "hierarchical_bootstrap",
                                "omitted_stock_id": np.nan,
                                "supervised_f16_r2": new_point,
                                "jepa_horizon_r2": horizon_point,
                                "paired_gap": new_point - horizon_point,
                                "lower_95": float(np.quantile(draws, 0.025)),
                                "upper_95": float(np.quantile(draws, 0.975)),
                                "bootstrap_draws": BOOTSTRAP_DRAWS,
                                "bootstrap_seed": BOOTSTRAP_SEED,
                            }
                        )
                        for stock in sorted(group_keys["stock_id"].unique()):
                            weights = np.ones((1, len(group_keys)), dtype=np.float64)
                            weights[0, group_keys["stock_id"].eq(stock).to_numpy()] = 0.0
                            new_loo = _weighted_block_r2(new, weights, group_keys)[0]
                            horizon_loo = _weighted_block_r2(horizon, weights, group_keys)[0]
                            rows.append(
                                {
                                    **identity,
                                    "estimate_type": "leave_one_stock_out",
                                    "omitted_stock_id": int(stock),
                                    "supervised_f16_r2": new_loo,
                                    "jepa_horizon_r2": horizon_loo,
                                    "paired_gap": new_loo - horizon_loo,
                                    "lower_95": np.nan,
                                    "upper_95": np.nan,
                                    "bootstrap_draws": 0,
                                    "bootstrap_seed": BOOTSTRAP_SEED,
                                }
                            )
    output = pd.DataFrame(rows)
    if len(output) != 4 * 3 * 2 * 2 * 3 * 8:
        raise F16IntegrityError("F16 grouped uncertainty table has unexpected size")
    path = output_root / "f16_grouped_uncertainty.parquet"
    atomic_write_parquet(output, path)
    return output


def run_f16_fixed_test(
    repo_root: Path,
    output_root: Path,
    bundle_root: Path,
    dataset_path: Path,
    *,
    device_name: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 2,
    chunk_rows: int = 8192,
) -> dict[str, Any]:
    started = time.perf_counter()
    state = extract_f16_test_statistics(
        repo_root,
        output_root,
        bundle_root,
        dataset_path,
        device_name=device_name,
        batch_size=batch_size,
        num_workers=num_workers,
        chunk_rows=chunk_rows,
    )
    results, geometry = evaluate_f16_test(repo_root, output_root, bundle_root)
    grouped = grouped_f16_uncertainty(repo_root, output_root)
    payload = {
        "schema_name": "thesis.experiment01.f16_fixed_test_complete",
        "schema_version": TEST_SCHEMA_VERSION,
        "status": "fixed_test_complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "unlock_sha256": sha256_file(Path(output_root) / "f16_test_unlock.json"),
        "test_extraction_state_sha256": sha256_file(
            Path(output_root) / "f16_test_extraction_state.json"
        ),
        "artifacts": {
            "results": {
                "path": _relative(Path(output_root) / "f16_results.parquet", Path(repo_root)),
                "sha256": sha256_file(Path(output_root) / "f16_results.parquet"),
                "rows": len(results),
            },
            "geometry": {
                "path": _relative(Path(output_root) / "f16_geometry.parquet", Path(repo_root)),
                "sha256": sha256_file(Path(output_root) / "f16_geometry.parquet"),
                "rows": len(geometry),
            },
            "grouped_uncertainty": {
                "path": _relative(
                    Path(output_root) / "f16_grouped_uncertainty.parquet", Path(repo_root)
                ),
                "sha256": sha256_file(Path(output_root) / "f16_grouped_uncertainty.parquet"),
                "rows": len(grouped),
            },
        },
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "test_accessed": True,
        "selection_changes_after_unlock": False,
        "failures": [],
    }
    payload["manifest_fingerprint"] = canonical_json_sha256(payload)
    atomic_write_json(Path(output_root) / "f16_fixed_test_complete.json", payload)
    return payload

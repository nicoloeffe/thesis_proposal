from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiment01.errors import ExperimentIntegrityError
from experiment01.io import sha256_array, sha256_file
from experiment01.phase3 import (
    LOW_BUDGETS,
    MLP_MIN_ROWS,
    PRIMARY_BRANCHES,
    READER_SEEDS,
    SPECTRAL_BANDS,
    VALID_RANK,
    FrozenFeatureTransform,
    Phase3MLPConfig,
    adaptive_reader_seeds,
    add_targetwise_normalized_recovery,
    assert_test_access_allowed,
    audit_historical_mlp_semantics,
    build_phase1_branch_whitening_effects,
    build_phase3_job_inventory,
    canonical_r2_from_sums,
    fit_target_standardizer,
    freeze_selection_manifest,
    load_frozen_feature_transform,
    load_subset_positions,
    make_primary_mlp,
    paired_metric_difference,
    run_synthetic_conditioning_gate,
    select_weight_decay,
    target_indices_for_block,
    train_validation_only_mlp,
    variance_components,
    verify_completed_cell,
    verify_full_budget_linear_parity,
    verify_spectral_band_identity,
)
from experiment01.schema import FeatureSet, TargetDefinition
from experiment01.sharded import ArrayShard, ShardedArray


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "validation/experiment01/execution_20260730/phase1"


def _target_definitions() -> tuple[TargetDefinition, ...]:
    values = []
    for prefix in ("d_spread_z", "d_microprice_rel", "d_top_imbalance"):
        for horizon in (1, 5, 10, 20):
            values.append(TargetDefinition(f"{prefix}@{horizon}", "directional", True, (), None))
    values.extend(
        [
            TargetDefinition("realized_vol@5", "volatility", True, (), None),
            TargetDefinition("realized_vol@20", "volatility", True, (), None),
            TargetDefinition("time_to_next_mid_move", "timing", True, (), None),
        ]
    )
    return tuple(values)


def _native_transform(dimension: int, mean: np.ndarray | None = None):
    center = np.zeros(dimension, dtype=np.float32) if mean is None else mean.astype(np.float32)
    return FrozenFeatureTransform(
        kind="native",
        mean=center,
        basis=None,
        scales=None,
        input_dimension=dimension,
        output_dimension=dimension,
        transform_hash="synthetic-native",
        source_transform_sha256="synthetic",
    )


def _selection_record(job_key: str = "cell") -> dict[str, object]:
    return {
        "job_key": job_key,
        "selected_weight_decay": 0.001,
        "selected_checkpoint_step": 1000,
        "validation_r2": 0.2,
        "training_seed": 7919,
        "input_transform_hash": "a" * 64,
        "subset_hash": "b" * 64,
        "model_definition_hash": "c" * 64,
    }


def test_01_exact_phase1_subset_reuse():
    payload = json.loads((PHASE1 / "subset_manifest.json").read_text())
    record = next(
        value
        for value in payload["subsets"]
        if value["budget_label"] == "b_1_4" and value["subsample_seed"] == 0
    )
    positions = load_subset_positions(PHASE1, record)
    assert len(positions) == record["n_rows"] == 7116
    subset = pd.read_parquet(PHASE1 / record["path"], columns=["row_key"])
    assert sha256_array(subset["row_key"].astype(str).to_numpy(dtype="U")) == record["row_key_sha256"]


def test_02_mlp_budget_eligibility_and_inventory_cardinality():
    payload = json.loads((PHASE1 / "subset_manifest.json").read_text())
    selection, evaluation = build_phase3_job_inventory(payload)
    assert selection["n_rows"].min() >= MLP_MIN_ROWS
    assert len(selection) == 8388
    assert len(evaluation) == 13068
    assert selection.job_key.is_unique and evaluation.job_key.is_unique


def test_03_target_standardization_uses_only_labelled_subset():
    y = np.arange(40, dtype=np.float32).reshape(20, 2)
    positions = np.array([1, 3, 7, 9], dtype=np.int64)
    scaler = fit_target_standardizer(y, positions, (0, 1), subset_hash="subset")
    assert np.allclose(scaler.mean, y[positions].mean(axis=0))
    changed = y.copy()
    changed[np.setdiff1d(np.arange(len(y)), positions)] += 10000
    second = fit_target_standardizer(changed, positions, (0, 1), subset_hash="subset")
    assert np.array_equal(scaler.mean, second.mean)
    assert np.array_equal(scaler.scale, second.scale)


def test_04_primary_mlp_has_no_coordinate_standardization_or_norm_layers():
    import torch.nn as nn

    model = make_primary_mlp(512, 12)
    assert [type(layer) for layer in model] == [nn.Linear, nn.GELU, nn.Dropout, nn.Linear]
    assert not any(isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm)) for layer in model)
    assert model[0].in_features == 512 and model[0].out_features == 256
    assert model[2].p == 0.10 and model[3].out_features == 12


def test_05_train_only_centering_pca_and_whitening_provenance():
    feature = FeatureSet("jepa_horizon", 0, "last_concat512", 512, np.dtype("float32"), {})
    native = load_frozen_feature_transform(PHASE1, feature, kind="native")
    white = load_frozen_feature_transform(PHASE1, feature, kind="full_whitened")
    top = load_frozen_feature_transform(
        PHASE1, feature, kind="pca_coordinates", spectral_arm="top_128"
    )
    assert native.source_transform_sha256 == white.source_transform_sha256 == top.source_transform_sha256
    assert np.array_equal(native.mean, white.mean) and np.array_equal(native.mean, top.mean)
    assert native.basis is None and white.basis.shape == (512, 508) and top.basis.shape == (512, 128)


def test_06_valid_rank_dimension_is_exactly_508():
    feature = FeatureSet("supervised", 2, "last_concat512", 512, np.dtype("float32"), {})
    white = load_frozen_feature_transform(PHASE1, feature, kind="full_whitened")
    assert VALID_RANK == 508 == white.output_dimension
    assert white.scales.shape == (508,) and np.all(white.scales > 0)


def test_07_equal_spectral_bands_are_disjoint_and_complete():
    result = verify_spectral_band_identity()
    assert result["status"] == "pass"
    assert all(stop - start == 127 for _, start, stop in SPECTRAL_BANDS)
    assert SPECTRAL_BANDS[0][1] == 0 and SPECTRAL_BANDS[-1][2] == 508


def test_08_selection_manifest_is_deterministic(tmp_path):
    hashes = []
    for name in ("a", "b"):
        path = tmp_path / name / "selection_manifest.json"
        hashes.append(
            freeze_selection_manifest(
                [_selection_record()], path, selection_inventory_sha256="d" * 64
            )
        )
    assert hashes[0] == hashes[1]


def test_09_test_access_blocked_until_selection_freeze(tmp_path):
    path = tmp_path / "selection_manifest.json"
    with pytest.raises(ExperimentIntegrityError):
        assert_test_access_allowed(path)
    digest = freeze_selection_manifest(
        [_selection_record()], path, selection_inventory_sha256="d" * 64
    )
    assert assert_test_access_allowed(path) == digest
    path.write_text(path.read_text() + " ")
    with pytest.raises(ExperimentIntegrityError):
        assert_test_access_allowed(path)


def test_10_adaptive_reader_seed_replication_is_nested():
    for budget in LOW_BUDGETS:
        assert adaptive_reader_seeds(budget) == READER_SEEDS
    assert adaptive_reader_seeds("b_8") == READER_SEEDS[:3]
    assert adaptive_reader_seeds("full_train") == READER_SEEDS[:3]


def test_11_step_based_training_and_early_stopping():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(256, 4)).astype(np.float32)
    y = (x[:, :1] + 0.05 * rng.normal(size=(256, 1))).astype(np.float32)
    positions = np.arange(len(x), dtype=np.int64)
    scaler = fit_target_standardizer(y, positions, (0,), subset_hash="small")
    config = Phase3MLPConfig(
        width=32,
        max_steps=40,
        min_steps=20,
        validation_interval=10,
        patience_evaluations=2,
        evaluation_chunk_rows=256,
    )
    result = train_validation_only_mlp(
        x, y, positions, x, y, _native_transform(4, x.mean(axis=0)), (0,), scaler,
        reader_seed=7, weight_decay=0.0, device="cpu", config=config,
        primary_width=False, enforce_preregistered_schedule=False,
    )
    assert result["best_step"] % 10 == 0
    assert result["last_step"] >= 20
    assert result["batch_size"] == 256


def test_12_weight_decay_exact_tie_selects_larger_value():
    candidates = [
        {"weight_decay": value, "validation_r2": 0.4}
        for value in (0.0, 1e-5, 1e-3)
    ]
    assert select_weight_decay(candidates)["weight_decay"] == 1e-3


def test_13_targetwise_ceiling_normalization_and_threshold():
    base = {
        "branch": "jepa_horizon", "encoder_seed": 0, "readout": "last_concat512",
        "target_block": "directional", "transform": "native", "spectral_arm": "none",
        "width": 256, "reader_seed": 0,
    }
    rows = [
        {**base, "target_name": "a", "budget_label": "full_train", "test_r2": 0.02},
        {**base, "target_name": "a", "budget_label": "b_1_4", "test_r2": -0.01},
        {**base, "target_name": "b", "budget_label": "full_train", "test_r2": 0.009},
        {**base, "target_name": "b", "budget_label": "b_1_4", "test_r2": 0.003},
    ]
    out = add_targetwise_normalized_recovery(pd.DataFrame(rows))
    a_low = out[(out.target_name == "a") & (out.budget_label == "b_1_4")].iloc[0]
    b_low = out[(out.target_name == "b") & (out.budget_label == "b_1_4")].iloc[0]
    assert a_low.ceiling_eligible and a_low.normalized_recovery == pytest.approx(-0.5)
    assert not b_low.ceiling_eligible and np.isnan(b_low.normalized_recovery)


def test_14_canonical_r2_uses_evaluation_mean():
    truth = np.array([1.0, 2.0, 4.0])
    prediction = np.array([1.5, 2.5, 3.5])
    r2 = canonical_r2_from_sums(
        np.array([np.sum((truth - prediction) ** 2)]),
        np.array([truth.sum()]),
        np.array([np.sum(truth**2)]),
        len(truth),
    )
    expected = 1 - np.sum((truth - prediction) ** 2) / np.sum((truth - truth.mean()) ** 2)
    assert r2[0] == pytest.approx(expected)


def test_15_hierarchical_variance_components_are_separate():
    rows = []
    for encoder in (0, 1, 2):
        for subset in (0, 1):
            for reader in (0, 1, 2):
                rows.append(
                    {
                        "job_family": "primary_directional", "readout": "last_concat512",
                        "target_block": "directional", "target_name": "a", "transform": "native",
                        "spectral_arm": "none", "budget_label": "b_1", "branch": "supervised",
                        "width": 256, "encoder_seed": encoder, "subsample_seed": subset,
                        "reader_seed": reader, "normalized_recovery": encoder + subset / 10 + reader / 100,
                    }
                )
    out = variance_components(pd.DataFrame(rows), "normalized_recovery").iloc[0]
    assert out.sd_reader_within_subset_encoder > 0
    assert out.sd_subsample_within_encoder > out.sd_reader_within_subset_encoder
    assert out.sd_encoder_between_means > out.sd_subsample_within_encoder


def test_16_paired_branch_and_transform_comparison_requires_exact_pairs():
    rows = []
    for branch, value in (("jepa_horizon", 0.2), ("supervised", 0.5)):
        rows.append(
            {
                "encoder_seed": 0, "readout": "last_concat512", "target_block": "directional",
                "target_name": "a", "budget_label": "b_1", "subsample_seed": 0,
                "reader_seed": 0, "width": 256, "branch": branch, "test_r2": value,
            }
        )
    out = paired_metric_difference(
        pd.DataFrame(rows), metric="test_r2", comparison_column="branch",
        left_value="supervised", right_value="jepa_horizon",
    )
    assert out.difference.iloc[0] == pytest.approx(0.3)


def test_17_conditioning_gate_equal_information_and_whitening_relief():
    result = run_synthetic_conditioning_gate()
    assert result["status"] == "pass"
    assert result["whitening_r2_improvement"] > 0.25
    assert result["oracle_prediction_max_abs_difference"] < 1e-10


def test_18_small_nonlinear_pipeline_detects_quadratic_signal():
    rng = np.random.default_rng(4)
    x_train = rng.normal(size=(1024, 4)).astype(np.float32)
    x_val = rng.normal(size=(512, 4)).astype(np.float32)
    y_train = (x_train[:, :1] ** 2).astype(np.float32)
    y_val = (x_val[:, :1] ** 2).astype(np.float32)
    positions = np.arange(len(x_train), dtype=np.int64)
    scaler = fit_target_standardizer(y_train, positions, (0,), subset_hash="quadratic")
    config = Phase3MLPConfig(
        width=64, max_steps=400, min_steps=100, validation_interval=50,
        patience_evaluations=6, evaluation_chunk_rows=512,
    )
    result = train_validation_only_mlp(
        x_train, y_train, positions, x_val, y_val,
        _native_transform(4, x_train.mean(axis=0)), (0,), scaler,
        reader_seed=4, weight_decay=0.0, device="cpu", config=config,
        primary_width=False, enforce_preregistered_schedule=False,
    )
    assert result["validation_r2"] > 0.5


def test_19_historical_mlp_static_gate_and_semantics():
    audit = audit_historical_mlp_semantics(ROOT)
    assert audit["status"] == "pass"
    assert audit["coordinatewise_input_standardization"] is True
    assert audit["hidden_layers"] == [256, 256]
    assert audit["references"]["jepa_horizon"] == pytest.approx(0.3191358981)


def test_20_full_budget_linear_parity_reads_frozen_rows():
    result = verify_full_budget_linear_parity(PHASE1, _target_definitions())
    assert result["status"] == "pass"
    assert result["row_count_per_reader"] == 90
    assert result["joined_row_count"] == 90


def test_21_target_block_inventory_is_exact():
    definitions = _target_definitions()
    assert len(target_indices_for_block(definitions, "directional")) == 12
    assert len(target_indices_for_block(definitions, "volatility")) == 2
    assert len(target_indices_for_block(definitions, "timing")) == 1


def test_22_streaming_smoke_native_and_whitened(tmp_path):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(256, 4)).astype(np.float32)
    y = (x[:, :1] - x[:, 1:2]).astype(np.float32)
    x_paths, y_paths = [], []
    for index, (start, stop) in enumerate(((0, 128), (128, 256))):
        xp, yp = tmp_path / f"x{index}.npy", tmp_path / f"y{index}.npy"
        np.save(xp, x[start:stop])
        np.save(yp, y[start:stop])
        x_paths.append(ArrayShard(xp, start, stop))
        y_paths.append(ArrayShard(yp, start, stop))
    xs = ShardedArray(x_paths, x.shape, x.dtype)
    ys = ShardedArray(y_paths, y.shape, y.dtype)
    positions = np.arange(len(x), dtype=np.int64)
    scaler = fit_target_standardizer(ys, positions, (0,), subset_hash="stream")
    covariance = np.cov(x.astype(np.float64), rowvar=False, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    transforms = [
        _native_transform(4, x.mean(axis=0)),
        FrozenFeatureTransform(
            "full_whitened", x.mean(axis=0), vectors.astype(np.float32),
            (1 / np.sqrt(values)).astype(np.float32), 4, 4, "white", "synthetic",
        ),
    ]
    config = Phase3MLPConfig(
        width=16, max_steps=20, min_steps=10, validation_interval=10,
        patience_evaluations=2, evaluation_chunk_rows=128,
    )
    for transform in transforms:
        result = train_validation_only_mlp(
            xs, ys, positions, xs, ys, transform, (0,), scaler,
            reader_seed=1, weight_decay=0.0, device="cpu", config=config,
            primary_width=False, enforce_preregistered_schedule=False,
        )
        assert np.isfinite(result["validation_r2"])


def test_23_restartability_verifies_completed_cell_hashes(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_text('{"ok": true}\n')
    fingerprint = {"job_key": "abc", "settings": "frozen"}
    state = {
        "status": "complete",
        "fingerprint": fingerprint,
        "artifacts": [
            {
                "path": artifact.name,
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        ],
    }
    state_path = tmp_path / "complete.json"
    state_path.write_text(json.dumps(state))
    assert verify_completed_cell(state_path, fingerprint)["status"] == "complete"
    artifact.write_text("corrupt")
    with pytest.raises(ExperimentIntegrityError):
        verify_completed_cell(state_path, fingerprint)

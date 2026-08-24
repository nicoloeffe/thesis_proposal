from __future__ import annotations

import json

import numpy as np

from experiment01.linear import SufficientStats, sufficient_stats
from experiment01.predictability_allocation import (
    EXPECTED_TARGET_NAMES,
    AllocationProtocol,
    allocation_tables,
    default_protocol_payload,
    deterministic_haar_basis,
    evaluate_preregistered_decision,
    freeze_protocol_payload,
    load_protocol,
    normalized_raw_windows,
    predictive_mass_fraction,
    raw_ridge_predictability,
    relationship_tables,
    sample_coverage_diagnostics,
    validate_sample_contract,
)
from training.train_tokenizer_t import normalize_book_window


def test_frozen_fractional_sample_contract_is_neither_tiny_nor_full():
    observed = validate_sample_contract(100_000, 50_000, 7_323_510)
    assert observed["full_dataset_fit"] is False
    assert 0.01 <= observed["fraction_of_valid_endpoints"] <= 0.05
    assert 0.01 <= observed["fraction_of_dataset_rows"] <= 0.05


def test_fractional_sample_coverage_requires_broad_disjoint_stock_days():
    n_train = 100_000
    n_validation = 50_000
    n = n_train + n_validation
    endpoint = np.arange(n, dtype=np.int64)
    stock = endpoint % 7
    day = np.empty(n, dtype=np.int64)
    day[:n_train] = np.arange(n_train) % 1200
    day[n_train:] = 10_000 + np.arange(n_validation) % 140
    observed = sample_coverage_diagnostics(
        stock,
        day,
        endpoint,
        endpoint[:n_train],
        endpoint[n_train:],
    )
    assert observed["passed"] is True
    assert observed["all_stocks_in_each_split"] is True
    assert observed["stock_day_disjoint"] is True
    assert observed["all_valid_stock_days_represented"] is True
    assert observed["raw_train_rows_over_dimension"] == 125.0


def test_vectorized_raw_windows_match_canonical_normalization():
    rng = np.random.default_rng(17)
    n = 24
    midpoint = rng.normal(10.0, 0.2, size=n).astype(np.float32)
    book = np.empty((n, 2, 10, 2), dtype=np.float32)
    offsets = np.linspace(0.01, 0.10, 10, dtype=np.float32)
    book[:, 0, :, 0] = midpoint[:, None] - offsets[None, :]
    book[:, 1, :, 0] = midpoint[:, None] + offsets[None, :]
    book[:, :, :, 1] = rng.normal(4.0, 1.0, size=(n, 2, 10))
    stock = np.zeros(n, dtype=np.int64)
    day = np.zeros(n, dtype=np.int64)
    stats = {
        "depth_scale_per_stock": np.asarray([0.04], dtype=np.float32),
        "vol_min_per_stock": np.asarray([1.25], dtype=np.float32),
        "vol_scale_per_stock": np.asarray([2.5], dtype=np.float32),
    }
    endpoints = np.asarray([19, 20, 23], dtype=np.int64)
    observed = normalized_raw_windows(
        book, midpoint, stock, day, endpoints, stats
    )
    expected = np.stack(
        [
            normalize_book_window(
                book[t - 19 : t + 1], midpoint[t - 19 : t + 1], 0, stats
            ).reshape(-1)
            for t in endpoints
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_fractional_predictive_mass_is_scale_invariant_and_localizes_signal():
    rng = np.random.default_rng(23)
    scales = np.asarray([5.0, 3.0, 2.0, 1.0, 0.7, 0.4])
    x = rng.normal(size=(4000, 6)) * scales
    y = np.column_stack(
        [
            x[:, 0] + 0.05 * rng.normal(size=len(x)),
            x[:, -1] + 0.05 * rng.normal(size=len(x)),
        ]
    )
    first = predictive_mass_fraction(
        sufficient_stats(x, y), top_k=2, full_mass_floor=1e-12
    )
    second = predictive_mass_fraction(
        sufficient_stats(x, y * np.asarray([11.0, 0.3])),
        top_k=2,
        full_mass_floor=1e-12,
    )
    np.testing.assert_allclose(first.top_fraction, second.top_fraction, atol=2e-12)
    assert first.top_fraction[0] > 0.95
    assert first.top_fraction[1] < 0.05


def test_matched_haar_allocation_is_deterministic_and_bounded():
    first = deterministic_haar_basis(
        9, 3, seed=41, branch_index=1, encoder_seed=2, draw=7
    )
    second = deterministic_haar_basis(
        9, 3, seed=41, branch_index=1, encoder_seed=2, draw=7
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(3), atol=2e-12)

    rng = np.random.default_rng(31)
    x = rng.normal(size=(900, 9))
    y = np.column_stack([x[:, 0], x[:, -1]]) + 0.1 * rng.normal(size=(900, 2))
    protocol = AllocationProtocol(
        status="frozen", top_k=3, haar_draws=99, full_mass_floor=1e-12
    )
    allocation, null = allocation_tables(
        x,
        y,
        ("d_imbalance_top5@1", "time_to_next_mid_move"),
        branch="jepa_horizon",
        encoder_seed=0,
        protocol=protocol,
    )
    assert allocation["F_top_k"].between(0.0, 1.0).all()
    assert null["haar_fraction"].between(0.0, 1.0).all()
    assert len(null) == 99 * 2


def test_raw_ridge_predictability_is_out_of_fold_and_target_specific():
    rng = np.random.default_rng(43)
    folds: list[SufficientStats] = []
    beta = np.asarray([1.2, -0.8, 0.4, 0.0])
    for _ in range(5):
        x = rng.normal(size=(500, 4))
        y = np.column_stack(
            [x @ beta + 0.1 * rng.normal(size=len(x)), rng.normal(size=len(x))]
        )
        folds.append(sufficient_stats(x, y))
    x_validation = rng.normal(size=(1000, 4))
    y_validation = np.column_stack(
        [
            x_validation @ beta + 0.1 * rng.normal(size=len(x_validation)),
            rng.normal(size=len(x_validation)),
        ]
    )
    result = raw_ridge_predictability(
        folds,
        sufficient_stats(x_validation, y_validation),
        ("d_imbalance_top5@1", "time_to_next_mid_move"),
        alpha_grid=(0.0, 1e-4, 1e-2, 1.0),
    )
    assert result.loc[0, "P_raw_linear"] > 0.98
    assert result.loc[1, "P_raw_linear"] < 0.10


def test_protocol_freeze_is_approval_and_input_bound(tmp_path):
    draft = default_protocol_payload()
    audit = {
        "passed": True,
        "outcomes_read": False,
        "inventory_sha256": "a" * 64,
    }
    frozen = freeze_protocol_payload(
        draft,
        audit,
        scientific_approver="Sol",
        approved_at_utc="2026-08-19T00:00:00+00:00",
    )
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    protocol, loaded = load_protocol(path, require_frozen=True)
    assert protocol.status == "frozen"
    assert loaded["freeze"]["input_inventory_sha256"] == "a" * 64
    assert loaded["freeze"]["outcomes_read_before_freeze"] is False


def test_preregistered_decision_uses_horizon_minus_supervised_contrast():
    names = list(EXPECTED_TARGET_NAMES)
    predictability = {
        "target_name": names,
        "P_raw_linear": np.linspace(0.0, 1.0, len(names)),
    }
    rows = []
    low = set(names[:6])
    for branch in ("supervised", "jepa_horizon", "jepa_masked"):
        for seed in (0, 1, 2):
            for index, name in enumerate(names):
                if branch == "jepa_horizon":
                    fraction = float(index)
                elif branch == "supervised":
                    fraction = float(-index)
                else:
                    fraction = 0.0
                rows.append(
                    {
                        "branch": branch,
                        "encoder_seed": seed,
                        "target_name": name,
                        "target_family": (
                            "imbalance"
                            if name.startswith("d_imbalance")
                            else "depth"
                            if name.startswith("d_log_depth")
                            else "timing"
                        ),
                        "F_top_k": fraction,
                        "haar_percentile": fraction,
                        "below_null_q05": branch == "jepa_horizon" and name in low,
                    }
                )
    import pandas as pd

    protocol = AllocationProtocol(status="frozen")
    _, correlations, summary, _ = relationship_tables(
        pd.DataFrame(predictability), pd.DataFrame(rows), protocol=protocol
    )
    decision = evaluate_preregistered_decision(
        correlations, summary, protocol=protocol
    )
    assert decision["outcome"] == "pass"
    assert all(decision["gates"].values())

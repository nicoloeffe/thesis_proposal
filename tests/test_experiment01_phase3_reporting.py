from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from experiment01.linear import SufficientStats
from experiment01.phase3_reporting import (
    _crossfit_predictor_r2,
    assign_phase3_outcome,
    build_phase3_reader_gap,
    hierarchical_bootstrap_gap,
)


def _raw_results() -> pd.DataFrame:
    rows = []
    for branch, recovery in (("supervised", 0.8), ("jepa_horizon", 0.5)):
        for reader_seed in (0, 1):
            rows.append(
                {
                    "job_family": "primary_directional",
                    "branch": branch,
                    "encoder_seed": 0,
                    "readout": "last_concat512",
                    "target_block": "directional",
                    "target_name": "target",
                    "transform": "native",
                    "spectral_arm": "none",
                    "budget_label": "b_1",
                    "n_stock_days": 7,
                    "n_rows": 5000,
                    "subsample_seed": 0,
                    "reader_seed": reader_seed,
                    "width": 256,
                    "test_r2": recovery * 0.2,
                    "full_budget_ceiling": 0.2,
                    "ceiling_eligible": True,
                    "normalized_recovery": recovery,
                }
            )
    return pd.DataFrame(rows)


def test_reader_gap_is_strictly_paired():
    gap = build_phase3_reader_gap(_raw_results())
    assert len(gap) == 2
    assert np.allclose(gap.normalized_recovery_gap, 0.3)
    assert np.allclose(gap.raw_r2_gap, 0.06)


def test_hierarchical_bootstrap_is_seed_deterministic():
    rows = []
    for encoder in (0, 1, 2):
        for subset in (0, 1):
            for reader in (0, 1, 2):
                rows.append(
                    {
                        "job_family": "primary_directional",
                        "readout": "last_concat512",
                        "target_block": "directional",
                        "target_name": "target",
                        "transform": "native",
                        "budget_label": "b_1",
                        "width": 256,
                        "encoder_seed": encoder,
                        "subsample_seed": subset,
                        "reader_seed": reader,
                        "both_ceiling_eligible": True,
                        "normalized_recovery_gap": 0.2 + encoder * 0.01,
                    }
                )
    frame = pd.DataFrame(rows)
    first = hierarchical_bootstrap_gap(frame, draws=200, seed=17)
    second = hierarchical_bootstrap_gap(frame, draws=200, seed=17)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0].robust_delta_010


def test_outcome_r3_requires_persistent_whitened_gap_and_seed_stability():
    gap_rows = []
    paired_rows = []
    for transform, gap in (("native", 0.25), ("full_whitened", 0.15)):
        for budget in ("b_1_4", "b_1_2", "b_1", "b_2", "b_4"):
            gap_rows.append(
                {
                    "job_family": "primary_directional",
                    "readout": "last_concat512",
                    "target_block": "directional",
                    "transform": transform,
                    "budget_label": budget,
                    "width": 256,
                    "mean": gap,
                    "robust_delta_010": True,
                }
            )
            for encoder in (0, 1, 2):
                paired_rows.append(
                    {
                        "target_block": "directional",
                        "transform": transform,
                        "budget_label": budget,
                        "encoder_seed": encoder,
                        "subsample_seed": 0,
                        "reader_seed": 0,
                        "both_ceiling_eligible": True,
                        "normalized_recovery_gap": gap,
                    }
                )
    ridge = pd.DataFrame(
        [
            {
                "target_block": "directional",
                "transform": "native",
                "budget_label": budget,
                "mean": 0.5,
                "robust_delta_010": True,
            }
            for budget in ("b_1_4", "b_1_2", "b_1", "b_2", "b_4")
        ]
    )
    results = pd.DataFrame(
        [
            {
                "job_family": "primary_directional",
                "budget_label": "full_train",
                "width": 256,
                "branch": branch,
                "transform": transform,
                "target_name": target,
                "ceiling_eligible": True,
            }
            for branch in ("jepa_horizon", "supervised")
            for transform in ("native", "full_whitened")
            for target in ("a", "b")
        ]
    )
    outcome = assign_phase3_outcome(
        results, pd.DataFrame(paired_rows), pd.DataFrame(gap_rows), ridge
    )
    assert outcome["outcome"] == "R3"


def test_reduced_outcome_uses_only_the_two_observed_adjacent_low_budgets():
    observed = ("b_1_4", "b_1_2")
    gap_rows = []
    paired_rows = []
    for transform, gap in (("native", 0.25), ("full_whitened", 0.15)):
        for budget in observed:
            gap_rows.append(
                {
                    "job_family": "primary_directional",
                    "readout": "last_concat512",
                    "target_block": "directional",
                    "transform": transform,
                    "budget_label": budget,
                    "width": 256,
                    "mean": gap,
                    "robust_delta_010": True,
                }
            )
            for encoder in (0, 1, 2):
                paired_rows.append(
                    {
                        "target_block": "directional",
                        "transform": transform,
                        "budget_label": budget,
                        "encoder_seed": encoder,
                        "subsample_seed": 0,
                        "reader_seed": 0,
                        "both_ceiling_eligible": True,
                        "normalized_recovery_gap": gap,
                    }
                )
    ridge = pd.DataFrame(
        [
            {
                "target_block": "directional",
                "transform": "native",
                "budget_label": budget,
                "mean": mean,
                "robust_delta_010": True,
            }
            for budget, mean in (
                ("b_1_4", 0.5),
                ("b_1_2", 0.5),
                ("b_1", 99.0),
                ("b_2", 99.0),
                ("b_4", 99.0),
            )
        ]
    )
    results = pd.DataFrame(
        [
            {
                "job_family": "primary_directional",
                "budget_label": "full_train",
                "width": 256,
                "branch": branch,
                "transform": transform,
                "target_name": target,
                "ceiling_eligible": True,
            }
            for branch in ("jepa_horizon", "supervised")
            for transform in ("native", "full_whitened")
            for target in ("a", "b")
        ]
    )
    outcome = assign_phase3_outcome(
        results, pd.DataFrame(paired_rows), pd.DataFrame(gap_rows), ridge
    )
    assert outcome["requirements"]["analyzed_low_budget_labels"] == list(observed)
    assert outcome["requirements"]["ridge_native_low_budget_mean_gap"] == 0.5


def test_crossfit_predictor_uses_opposite_fold_statistics():
    rng = np.random.default_rng(3)
    x_a = rng.normal(size=(500, 3))
    x_b = rng.normal(size=(500, 3))
    beta = np.array([[1.0], [0.0], [0.0]])
    y_a = x_a @ beta
    y_b = x_b @ beta
    a = SufficientStats.zeros(3, 1)
    b = SufficientStats.zeros(3, 1)
    a.add_rows(x_a, y_a)
    b.add_rows(x_b, y_b)
    score = _crossfit_predictor_r2(a, b, np.ones(3), np.arange(3))
    assert score[0] > 0.9


def test_phase3_report_keeps_r3_secondary_when_low_budget_fits_fail():
    source = (
        Path(__file__).resolve().parents[1]
        / "experiment01"
        / "phase3_reporting.py"
    ).read_text(encoding="utf-8")
    assert "Primary scientific result and identifiability" in source
    assert "Frozen preregistered technical classification" in source
    assert "not identified in the executed regime" in source
    assert "difference between failed fits" in source

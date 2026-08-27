from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment01.linear import (
    evaluate,
    evaluate_stats,
    fit_alpha,
    sufficient_stats,
    transformed_design,
)
from experiment01.phase2 import (
    FeatureStatistics,
    deterministic_haar_basis,
    phase2_schedule,
    phase2_ladder_table,
    predictive_mass_table,
    project_stats,
    random_subspace_table,
    spectral_bands,
)
from experiment01.schema import FeatureSet, InputBundle, TargetDefinition
from experiment01.phase2_reporting import _nonmonotonic_status


def _minimal_bundle(tmp_path: Path, feature: FeatureSet) -> InputBundle:
    definitions = tuple(
        TargetDefinition(
            name=name,
            block=block,
            independent=True,
            redundant_with=(),
            semantics=None,
        )
        for name, block in (
            ("direction", "directional"),
            ("volatility", "volatility"),
            ("timing", "timing"),
        )
    )
    return InputBundle(
        root=tmp_path,
        manifest={},
        rows={},
        target_paths={},
        target_definitions=definitions,
        feature_sets=(feature,),
    )


def test_phase2_schedule_and_bands_cover_valid_spectrum_exactly():
    assert phase2_schedule(508) == (1, 2, 4, 8, 16, 32, 64, 128, 256, 508)
    bands = spectral_bands(508)
    assert bands[-1] == ("257:D_valid", 256, 508)
    covered = np.concatenate([np.arange(start, stop) for _, start, stop in bands])
    np.testing.assert_array_equal(covered, np.arange(508))


def test_deterministic_haar_basis_is_repeatable_and_orthonormal():
    first = deterministic_haar_basis(
        17,
        6,
        branch_index=1,
        encoder_seed=2,
        readout_index=0,
        draw=19,
    )
    second = deterministic_haar_basis(
        17,
        6,
        branch_index=1,
        encoder_seed=2,
        readout_index=0,
        draw=19,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(6), atol=2e-12)


def test_projected_sufficient_statistics_match_direct_projection():
    rng = np.random.default_rng(61)
    x_train = rng.normal(size=(211, 9))
    y_train = rng.normal(size=(211, 3))
    x_test = rng.normal(size=(103, 9))
    y_test = rng.normal(size=(103, 3))
    basis = deterministic_haar_basis(
        9,
        4,
        branch_index=0,
        encoder_seed=0,
        readout_index=1,
        draw=3,
    )
    train_stats = project_stats(sufficient_stats(x_train, y_train), basis)
    test_stats = project_stats(sufficient_stats(x_test, y_test), basis)
    model = fit_alpha(transformed_design(train_stats), 0.0)
    observed = evaluate_stats(model, test_stats).values

    direct_model = fit_alpha(
        transformed_design(sufficient_stats(x_train @ basis, y_train)), 0.0
    )
    expected = evaluate(direct_model, x_test @ basis, y_test).values
    np.testing.assert_allclose(observed, expected, rtol=2e-11, atol=2e-11)


def test_predictive_mass_uses_declared_formula_and_marks_invalid_tail(tmp_path):
    rng = np.random.default_rng(87)
    base = rng.normal(size=(401, 4))
    x = base - base.mean(axis=0)
    covariance = x.T @ x / len(x)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    x_pc = x @ vectors[:, order]
    y = np.column_stack(
        [x_pc[:, 0] + 0.1 * rng.normal(size=len(x)), rng.normal(size=len(x)), x_pc[:, 2]]
    )
    stats = sufficient_stats(x_pc, y)
    feature = FeatureSet(
        branch="supervised",
        encoder_seed=0,
        readout="last_concat512",
        dimension=4,
        dtype=np.dtype("float32"),
        paths={},
    )
    bundle = _minimal_bundle(tmp_path, feature)
    statistics = FeatureStatistics(
        eigenvalues=values,
        numerical_rank=3,
        numerical_tolerance=float(values[3] + 1e-12),
        budgets={"full_train": stats},
        validation=stats,
        test=stats,
    )
    table = predictive_mass_table(bundle, feature, statistics)
    direction = table[
        table["target_name"].eq("direction")
        & table["direction_index"].eq(1)
    ].iloc[0]
    variance_y = stats.target_centered_ss[0] / stats.n
    expected = stats.cross[0, 0] ** 2 / values[0] / variance_y
    np.testing.assert_allclose(direction["predictive_mass"], expected)
    invalid = table[table["direction_index"].eq(4)]
    assert invalid["fit_status"].eq("invalid").all()
    assert invalid["predictive_mass"].isna().all()
    assert invalid["failure_reason"].eq(
        "numerically_invalid_covariance_direction"
    ).all()


def test_full_rank_haar_cell_is_treated_as_exact_subspace_tie(tmp_path):
    rng = np.random.default_rng(104)
    x_train = rng.normal(size=(151, 4))
    y_train = rng.normal(size=(151, 3))
    x_validation = rng.normal(size=(79, 4))
    y_validation = rng.normal(size=(79, 3))
    x_test = rng.normal(size=(83, 4))
    y_test = rng.normal(size=(83, 3))
    feature = FeatureSet(
        branch="supervised",
        encoder_seed=0,
        readout="last_concat512",
        dimension=4,
        dtype=np.dtype("float32"),
        paths={},
    )
    bundle = _minimal_bundle(tmp_path, feature)
    train = sufficient_stats(x_train, y_train)
    statistics = FeatureStatistics(
        eigenvalues=np.linalg.eigvalsh(train.gram)[::-1],
        numerical_rank=4,
        numerical_tolerance=0.0,
        budgets={
            label: train
            for label in ("b_1_8", "b_1_4", "b_4", "b_16", "full_train")
        },
        validation=sufficient_stats(x_validation, y_validation),
        test=sufficient_stats(x_test, y_test),
    )
    ladder = phase2_ladder_table(bundle, feature, statistics, commit="test")
    null = random_subspace_table(
        bundle, feature, statistics, ladder, n_draws=100
    )
    full = null[null["subspace_dimension"].eq(4)]
    assert full["top_pca_percentile"].eq(100.0).all()
    assert full["empirical_p_random_exceeds_top"].eq(0.0).all()


def test_unequal_width_band_difference_is_never_promoted_to_matched_evidence():
    misleadingly_positive = pd.DataFrame(
        {
            "readout": ["last_concat512", "last_concat512"],
            "target_block": ["directional", "directional"],
            "metric": [
                "predictive_mass_fraction_33_64_minus_17_32",
                "predictive_mass_fraction_33_64_minus_17_32",
            ],
            "branch": ["supervised", "jepa_horizon"],
            "supports_33_64_more_informative": [True, True],
        }
    )
    status = _nonmonotonic_status(misleadingly_positive)
    assert status.startswith("not_dimension_matched")
    assert "16 directions" in status
    assert "32" in status

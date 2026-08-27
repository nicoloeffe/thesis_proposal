from __future__ import annotations

import numpy as np

from experiment01.training_audit import _subsample_positions


def test_historical_same_seed_train_rows_match_across_arms() -> None:
    train = np.arange(1_000, dtype=np.int64)
    validation = np.arange(2_000, 2_400, dtype=np.int64)
    supervised_train, _ = _subsample_positions(
        train, validation, seed=2, arm="supervised", n_train=200, n_validation=100
    )
    horizon_train, _ = _subsample_positions(
        train, validation, seed=2, arm="jepa_horizon", n_train=200, n_validation=100
    )
    masked_train, _ = _subsample_positions(
        train, validation, seed=2, arm="jepa_masked", n_train=200, n_validation=100
    )
    np.testing.assert_array_equal(supervised_train, horizon_train)
    np.testing.assert_array_equal(supervised_train, masked_train)


def test_historical_validation_rng_differs_between_supervised_and_jepa() -> None:
    train = np.arange(1_000, dtype=np.int64)
    validation = np.arange(2_000, 2_400, dtype=np.int64)
    _, supervised_validation = _subsample_positions(
        train, validation, seed=1, arm="supervised", n_train=200, n_validation=100
    )
    _, horizon_validation = _subsample_positions(
        train, validation, seed=1, arm="jepa_horizon", n_train=200, n_validation=100
    )
    assert not np.array_equal(supervised_validation, horizon_validation)


def test_encoder_seed_changes_historical_train_rows() -> None:
    train = np.arange(1_000, dtype=np.int64)
    validation = np.arange(2_000, 2_400, dtype=np.int64)
    seed0, _ = _subsample_positions(
        train, validation, seed=0, arm="supervised", n_train=200, n_validation=100
    )
    seed1, _ = _subsample_positions(
        train, validation, seed=1, arm="supervised", n_train=200, n_validation=100
    )
    assert not np.array_equal(seed0, seed1)

from __future__ import annotations

import numpy as np
import pytest

from experiment01.reference.consolidation_geometry import linear_stats, r2_from_basis
from experiment01.token_role import (
    generic_feature_basis,
    observed_role_bases,
    plus_one_probability,
    structured_role_bases,
    validate_complementary_bases,
)


def test_observed_role_bases_are_orthonormal_complements() -> None:
    bases = observed_role_bases()
    assert bases["common"].shape == (512, 128)
    assert bases["complement"].shape == (512, 384)
    validate_complementary_bases(bases["common"], bases["complement"])


def test_structured_role_bases_are_deterministic_and_distinct() -> None:
    first = structured_role_bases(7)
    repeated = structured_role_bases(7)
    other = structured_role_bases(8)
    assert np.array_equal(first["common"], repeated["common"])
    assert np.array_equal(first["complement"], repeated["complement"])
    assert not np.array_equal(first["common"], other["common"])
    validate_complementary_bases(first["common"], first["complement"])


@pytest.mark.parametrize("dimension", [128, 384])
def test_generic_feature_basis_is_deterministic_and_orthonormal(
    dimension: int,
) -> None:
    first = generic_feature_basis(2, dimension)
    second = generic_feature_basis(2, dimension)
    assert np.array_equal(first, second)
    np.testing.assert_allclose(
        first.T @ first, np.eye(dimension), atol=1e-12, rtol=0.0
    )


def test_projected_sufficient_statistics_match_direct_projection() -> None:
    rng = np.random.default_rng(44)
    x_train = rng.normal(size=(800, 16))
    x_val = rng.normal(size=(300, 16))
    y_train = rng.normal(size=(800, 4))
    y_val = rng.normal(size=(300, 4))
    q, _ = np.linalg.qr(rng.normal(size=(16, 7)), mode="reduced")
    full_stats = linear_stats(x_train, y_train, x_val, y_val)
    projected_stats = linear_stats(x_train @ q, y_train, x_val @ q, y_val)
    from_full = r2_from_basis(full_stats, q)
    direct = r2_from_basis(projected_stats, np.eye(7))
    np.testing.assert_allclose(from_full, direct, atol=1e-12, rtol=0.0)


def test_commonality_shapley_identity() -> None:
    intercept = -0.02
    full = 0.21
    common = 0.04
    complement = 0.20
    phi_common = 0.5 * ((common - intercept) + (full - complement))
    phi_complement = 0.5 * ((complement - intercept) + (full - common))
    assert phi_common + phi_complement == pytest.approx(full - intercept)
    assert common + complement - full - intercept == pytest.approx(0.05)


def test_plus_one_probability_uses_registered_tail_and_denominator() -> None:
    null = np.arange(100, dtype=np.float64)
    lower, lower_percentile, lower_count = plus_one_probability(2.5, null, tail="lower")
    upper, upper_percentile, upper_count = plus_one_probability(97.5, null, tail="upper")
    assert (lower, lower_percentile, lower_count) == pytest.approx((4 / 101, 0.03, 3))
    assert (upper, upper_percentile, upper_count) == pytest.approx((3 / 101, 0.02, 2))


def test_plus_one_probability_fails_closed_on_incomplete_null() -> None:
    with pytest.raises(Exception, match="incomplete"):
        plus_one_probability(0.0, np.zeros(99), tail="lower")

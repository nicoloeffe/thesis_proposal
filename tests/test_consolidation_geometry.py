from __future__ import annotations

import numpy as np

from experiment01.historical.consolidation_geometry import (
    HADAMARD4,
    hadamard_mean_basis,
    hadamard_transform,
    ladder_from_stats,
    linear_stats,
    pca_from_stats,
    principal_angle_curve,
    random_subspace_null,
)
from experiment01.historical.ladder_accessibility import mlp_ceiling


def _ladder(xtr, ytr, xva, yva, schedule):
    stats = linear_stats(xtr, ytr, xva, yva)
    _, vectors = pca_from_stats(stats)
    return ladder_from_stats(stats, vectors, schedule)


def test_hadamard_is_orthogonal():
    np.testing.assert_allclose(
        HADAMARD4.T @ HADAMARD4, np.eye(4), rtol=0.0, atol=1e-12
    )


def test_ladder_is_invariant_to_full_hadamard_rotation():
    rng = np.random.default_rng(11)
    xtr = rng.normal(size=(700, 512))
    xtr *= np.linspace(0.5, 2.0, 512)
    xva = rng.normal(size=(300, 512))
    xva *= np.linspace(0.5, 2.0, 512)
    beta = rng.normal(size=(512, 3))
    ytr = xtr @ beta
    yva = xva @ beta
    schedule = [8, 32, 128, 512]

    original = _ladder(xtr, ytr, xva, yva, schedule)
    rotated = _ladder(
        hadamard_transform(xtr),
        ytr,
        hadamard_transform(xva),
        yva,
        schedule,
    )

    for m in schedule:
        np.testing.assert_allclose(original[m], rotated[m], atol=1e-9, rtol=1e-9)


def test_random_subspace_null_matches_isotropic_fraction():
    rng = np.random.default_rng(23)
    dimension = 24
    x = rng.normal(size=(5000, dimension))
    # A complete isotropic target basis makes the average recovered fraction
    # exactly dimension-counting in the population.
    y = x.copy()
    result = random_subspace_null(
        x,
        y,
        [6, 12, 24],
        n_draws=30,
        seed=5,
    )

    for m in (6, 12, 24):
        observed = float(np.nanmean(result["mean"][m]))
        assert abs(observed - m / dimension) < 0.035


def test_principal_angle_energy_is_one_at_full_rank():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(600, 16))
    beta = rng.normal(size=(16, 4))
    y = x @ beta
    stats = linear_stats(x, y)
    values, vectors = pca_from_stats(stats)

    rows = principal_angle_curve(
        stats,
        values,
        vectors,
        target_indices=[0, 1, 2, 3],
        schedule=[4, 16],
    )
    full = [row for row in rows if row["m"] == 16]

    assert full
    np.testing.assert_allclose(full[0]["aligned_energy"], 1.0, atol=1e-12)


def test_hadamard_mean_basis_is_orthonormal():
    basis = hadamard_mean_basis(512)
    assert basis is not None
    np.testing.assert_allclose(basis.T @ basis, np.eye(128), atol=1e-12)


def test_mlp_ceiling_is_reproducible_with_fixed_reader_seeds():
    rng = np.random.default_rng(41)
    xtr = rng.normal(size=(180, 10)).astype(np.float32)
    xva = rng.normal(size=(70, 10)).astype(np.float32)
    beta = rng.normal(size=(10, 2)).astype(np.float32)
    ytr = xtr @ beta
    yva = xva @ beta
    kwargs = dict(
        device="cpu",
        hidden=16,
        epochs=6,
        mlp_seeds=2,
        patience=2,
        split_seed=0,
        batch_size=64,
    )

    first = mlp_ceiling(xtr, xva, ytr, yva, **kwargs)
    second = mlp_ceiling(xtr, xva, ytr, yva, **kwargs)

    np.testing.assert_array_equal(first["runs"], second["runs"])
    np.testing.assert_array_equal(first["epochs_used"], second["epochs_used"])
    np.testing.assert_array_equal(
        first["internal_val_indices"], second["internal_val_indices"]
    )

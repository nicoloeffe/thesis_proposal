"""Target-blind readout transforms and geometry for consolidation analyses.

The functions in this module are deliberately independent of checkpoints and
training code.  They operate only on frozen readout matrices and targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


HADAMARD4 = 0.5 * np.asarray(
    [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)

POOLING_ALIASES = {
    "last_concat512": ("last_concat512", "identity"),
    "meanK_concatS": ("tmean_concat512", "identity"),
    "tmean_concat512": ("tmean_concat512", "identity"),
    "mean_all128": ("tmean_concat512", "mean_all"),
    "last_hadamard_mean128": ("last_concat512", "hadamard_mean"),
    "last_hadamard_contrast384": (
        "last_concat512",
        "hadamard_contrast",
    ),
    "meanK_hadamard_mean128": ("tmean_concat512", "hadamard_mean"),
    "meanK_hadamard_contrast384": (
        "tmean_concat512",
        "hadamard_contrast",
    ),
}

DEFAULT_POOLINGS = [
    "last_concat512",
    "meanK_concatS",
    "mean_all128",
    "last_hadamard_mean128",
    "last_hadamard_contrast384",
    "meanK_hadamard_mean128",
    "meanK_hadamard_contrast384",
]


def hadamard_transform(concat512: np.ndarray) -> np.ndarray:
    """Rotate four concatenated 128-d tokens with the normalized Hadamard."""
    x = np.asarray(concat512)
    if x.ndim != 2 or x.shape[1] != 512:
        raise ValueError(f"expected (N, 512), got {x.shape}")
    tokens = x.reshape(x.shape[0], 4, 128)
    rotated = np.einsum("ab,nbd->nad", HADAMARD4, tokens, optimize=True)
    return rotated.reshape(x.shape[0], 512)


def derive_pooling(dump, pooling: str, split: str) -> np.ndarray:
    """Load or derive a fixed, target-blind pooling from a stage-1 dump."""
    if pooling not in POOLING_ALIASES:
        raise KeyError(
            f"unknown pooling {pooling!r}; choices={sorted(POOLING_ALIASES)}"
        )
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    base, transform = POOLING_ALIASES[pooling]
    x = np.asarray(dump[f"{base}_{split}"])
    if x.ndim != 2 or x.shape[1] != 512:
        raise ValueError(f"{base}_{split}: expected (N, 512), got {x.shape}")
    if transform == "identity":
        return x
    rotated = hadamard_transform(x)
    if transform == "mean_all":
        # The normalized Hadamard mean is twice the arithmetic mean.  The
        # constant scale is immaterial to PCA/OLS, but mean_all keeps its
        # literal definition here.
        return x.reshape(x.shape[0], 4, 128).mean(axis=1)
    if transform == "hadamard_mean":
        return rotated[:, :128]
    if transform == "hadamard_contrast":
        return rotated[:, 128:]
    raise AssertionError(transform)


def schedule_for_dimension(schedule: Iterable[int], dimension: int) -> list[int]:
    """Clamp a ladder grid to D, remove duplicates, and always include D."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    values = {min(int(m), dimension) for m in schedule if int(m) > 0}
    values.add(dimension)
    return sorted(values)


@dataclass
class LinearStats:
    """Sufficient statistics for train-fit / validation-evaluated linear probes."""

    x_mean: np.ndarray
    y_mean: np.ndarray
    gram_train: np.ndarray
    cross_train: np.ndarray
    gram_val: np.ndarray
    cross_val: np.ndarray
    val_y_train_centered_ss: np.ndarray
    val_total_ss: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.gram_train.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.cross_train.shape[1])


def linear_stats(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> LinearStats:
    """Build float64 sufficient statistics with a train-fitted intercept."""
    xtr = np.asarray(x_train, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    if ytr.ndim == 1:
        ytr = ytr[:, None]
    if xtr.ndim != 2 or ytr.ndim != 2 or len(xtr) != len(ytr):
        raise ValueError("x_train/y_train must be aligned 2-D matrices")
    xva = xtr if x_val is None else np.asarray(x_val, dtype=np.float64)
    yva = ytr if y_val is None else np.asarray(y_val, dtype=np.float64)
    if yva.ndim == 1:
        yva = yva[:, None]
    if (
        xva.ndim != 2
        or yva.ndim != 2
        or len(xva) != len(yva)
        or xva.shape[1] != xtr.shape[1]
        or yva.shape[1] != ytr.shape[1]
    ):
        raise ValueError("validation matrices are not aligned with train")

    x_mean = xtr.mean(axis=0, keepdims=True)
    y_mean = ytr.mean(axis=0, keepdims=True)
    xc = xtr - x_mean
    yc = ytr - y_mean
    xv = xva - x_mean
    yv = yva - y_mean
    yv_baseline = yva - yva.mean(axis=0, keepdims=True)
    return LinearStats(
        x_mean=x_mean,
        y_mean=y_mean,
        gram_train=xc.T @ xc,
        cross_train=xc.T @ yc,
        gram_val=xv.T @ xv,
        cross_val=xv.T @ yv,
        val_y_train_centered_ss=np.einsum("nt,nt->t", yv, yv),
        val_total_ss=np.einsum("nt,nt->t", yv_baseline, yv_baseline),
    )


def pca_from_stats(stats: LinearStats) -> tuple[np.ndarray, np.ndarray]:
    """Return descending covariance eigenvalues and eigenvectors."""
    values, vectors = np.linalg.eigh(stats.gram_train)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    return values, vectors[:, order]


def r2_from_basis(stats: LinearStats, basis: np.ndarray) -> np.ndarray:
    """Validation R² for min-norm OLS on the columns of an orthonormal basis."""
    q = np.asarray(basis, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != stats.dimension:
        raise ValueError("basis has incompatible shape")
    a = q.T @ stats.gram_train @ q
    c = q.T @ stats.cross_train
    weights = np.linalg.lstsq(a, c, rcond=None)[0]
    av = q.T @ stats.gram_val @ q
    cv = q.T @ stats.cross_val
    cross_term = np.einsum("mt,mt->t", weights, cv)
    quadratic = np.einsum("mt,mn,nt->t", weights, av, weights)
    sse = stats.val_y_train_centered_ss - 2.0 * cross_term + quadratic
    return 1.0 - sse / np.maximum(stats.val_total_ss, 1e-12)


def ladder_from_stats(
    stats: LinearStats,
    eigenvectors: np.ndarray,
    schedule: Sequence[int],
) -> Dict[int, np.ndarray]:
    """PCA-ordered min-norm OLS ladder from precomputed statistics."""
    return {
        int(m): r2_from_basis(stats, eigenvectors[:, : int(m)])
        for m in schedule_for_dimension(schedule, stats.dimension)
    }


def haar_bases(
    dimension: int, n_draws: int = 20, seed: int = 0
) -> list[np.ndarray]:
    """Deterministic Haar-distributed orthogonal bases."""
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(n_draws):
        q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
        signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
        bases.append(q * signs[None, :])
    return bases


def random_subspace_null(
    X: np.ndarray,
    Y: np.ndarray,
    m_grid: Sequence[int],
    n_draws: int = 20,
    seed: int = 0,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    *,
    stats: Optional[LinearStats] = None,
    bases: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, object]:
    """Fraction of full-rank R² recovered by random m-dimensional subspaces.

    The optional sufficient statistics avoid repeatedly materializing projected
    100k-row matrices in the production analysis.  The public X/Y interface
    remains the one specified by the consolidation protocol.
    """
    st = stats or linear_stats(X, Y, X_val, Y_val)
    grid = schedule_for_dimension(m_grid, st.dimension)
    random_bases = list(bases) if bases is not None else haar_bases(
        st.dimension, n_draws=n_draws, seed=seed
    )
    if len(random_bases) != n_draws:
        raise ValueError("number of supplied bases does not match n_draws")
    full_r2 = r2_from_basis(st, np.eye(st.dimension))
    draws = {
        m: np.empty((n_draws, st.n_targets), dtype=np.float64) for m in grid
    }
    for draw_index, q in enumerate(random_bases):
        if q.shape != (st.dimension, st.dimension):
            raise ValueError("random basis has incompatible shape")
        for m in grid:
            r2 = full_r2 if m == st.dimension else r2_from_basis(st, q[:, :m])
            draws[m][draw_index] = np.divide(
                r2,
                full_r2,
                out=np.full_like(r2, np.nan),
                where=np.abs(full_r2) > 1e-12,
            )
    return {
        "mean": {m: np.nanmean(values, axis=0) for m, values in draws.items()},
        "std": {m: np.nanstd(values, axis=0) for m, values in draws.items()},
        "draws": draws,
        "full_r2": full_r2,
        "n_draws": n_draws,
        "seed": seed,
    }


def signal_basis(stats: LinearStats, target_indices: Sequence[int]) -> np.ndarray:
    """Orthonormal basis for the full-rank min-norm OLS coefficient span."""
    idx = np.asarray(target_indices, dtype=np.int64)
    coefficients = np.linalg.pinv(stats.gram_train) @ stats.cross_train[:, idx]
    u, singular_values, _ = np.linalg.svd(coefficients, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((stats.dimension, 0), dtype=np.float64)
    tolerance = (
        np.finfo(np.float64).eps
        * max(coefficients.shape)
        * singular_values[0]
    )
    rank = int(np.sum(singular_values > tolerance))
    return u[:, :rank]


def hadamard_mean_basis(dimension: int) -> Optional[np.ndarray]:
    """Fixed common-token subspace for a four-token concatenated readout."""
    if dimension != 512:
        return None
    eye = np.eye(128, dtype=np.float64)
    return np.concatenate([0.5 * eye] * 4, axis=0)


def principal_angle_curve(
    stats: LinearStats,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    target_indices: Sequence[int],
    schedule: Sequence[int],
    reliability_threshold: float = 1e-3,
    mean_basis: Optional[np.ndarray] = None,
) -> list[dict]:
    """Principal-angle rows between a signal span and top-m variance spans."""
    qs = signal_basis(stats, target_indices)
    rank = qs.shape[1]
    if rank == 0:
        return []
    mean_energy = np.nan
    if mean_basis is not None:
        qm = np.asarray(mean_basis, dtype=np.float64)
        mean_energy = float(np.linalg.norm(qm.T @ qs, ord="fro") ** 2 / rank)
    rows = []
    for m in schedule_for_dimension(schedule, stats.dimension):
        singular_values = np.linalg.svd(
            qs.T @ eigenvectors[:, :m], compute_uv=False
        )
        cos2 = np.square(np.clip(singular_values, 0.0, 1.0))
        energy = float(cos2.sum() / rank)
        if m < stats.dimension and eigenvalues[m - 1] > 0.0:
            relative_gap = float(
                (eigenvalues[m - 1] - eigenvalues[m])
                / eigenvalues[m - 1]
            )
            reliable = relative_gap >= reliability_threshold
        else:
            relative_gap = float("nan")
            reliable = True
        for k in range(rank):
            rows.append(
                {
                    "m": m,
                    "k": k + 1,
                    "cos2": float(cos2[k]) if k < len(cos2) else 0.0,
                    "aligned_energy": energy,
                    "signal_rank": rank,
                    "relative_eigen_gap": relative_gap,
                    "reliable": reliable,
                    "mean_subspace_energy": mean_energy,
                }
            )
    return rows


def spectral_diagnostics(eigenvalues: np.ndarray) -> dict:
    """Participation ratio and entropy effective rank."""
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return {"participation_ratio": 0.0, "effective_rank": 0.0}
    probabilities = values / total
    nonzero = probabilities > 0.0
    return {
        "participation_ratio": float(
            total * total / np.maximum(np.square(values).sum(), 1e-30)
        ),
        "effective_rank": float(
            np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
        ),
    }

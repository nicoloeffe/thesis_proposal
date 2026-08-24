"""Sufficient-statistic linear readers and progressive top-k whitening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .constants import ALPHA_GRID, WHITEN_K_BASE
from .errors import ExperimentIntegrityError


@dataclass
class SufficientStats:
    """Additive uncentered sufficient statistics.

    Keeping sums rather than only centered matrices makes nested-budget updates
    exact: newly added stock-day blocks are accumulated without rescanning
    earlier labelled rows.
    """

    n: int
    x_sum: np.ndarray
    y_sum: np.ndarray
    xtx: np.ndarray
    xty: np.ndarray
    yty: np.ndarray

    @classmethod
    def zeros(cls, dimension: int, n_targets: int) -> "SufficientStats":
        return cls(
            n=0,
            x_sum=np.zeros(dimension, dtype=np.float64),
            y_sum=np.zeros(n_targets, dtype=np.float64),
            xtx=np.zeros((dimension, dimension), dtype=np.float64),
            xty=np.zeros((dimension, n_targets), dtype=np.float64),
            yty=np.zeros(n_targets, dtype=np.float64),
        )

    @property
    def dimension(self) -> int:
        return int(self.x_sum.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.y_sum.shape[0])

    @property
    def x_mean(self) -> np.ndarray:
        self._require_nonempty()
        return self.x_sum / self.n

    @property
    def y_mean(self) -> np.ndarray:
        self._require_nonempty()
        return self.y_sum / self.n

    @property
    def gram(self) -> np.ndarray:
        """Labelled-design covariance ``Xc.T @ Xc / n``."""
        self._require_nonempty()
        value = self.xtx / self.n - np.outer(self.x_mean, self.x_mean)
        return (value + value.T) * 0.5

    @property
    def cross(self) -> np.ndarray:
        self._require_nonempty()
        return self.xty / self.n - np.outer(self.x_mean, self.y_mean)

    @property
    def target_centered_ss(self) -> np.ndarray:
        self._require_nonempty()
        return self.yty - self.n * np.square(self.y_mean)

    def _require_nonempty(self) -> None:
        if self.n <= 0:
            raise ValueError("sufficient statistics are empty")

    def add_rows(self, x: np.ndarray, y: np.ndarray) -> None:
        x_value, y_value = _aligned_xy(x, y)
        if x_value.shape[1] != self.dimension:
            raise ValueError("feature dimension differs from accumulator")
        if y_value.shape[1] != self.n_targets:
            raise ValueError("target dimension differs from accumulator")
        if not np.isfinite(x_value).all() or not np.isfinite(y_value).all():
            raise ExperimentIntegrityError("NaN or infinity in labelled rows")
        self.n += len(x_value)
        self.x_sum += x_value.sum(axis=0, dtype=np.float64)
        self.y_sum += y_value.sum(axis=0, dtype=np.float64)
        self.xtx += x_value.T @ x_value
        self.xty += x_value.T @ y_value
        self.yty += np.einsum("nt,nt->t", y_value, y_value)

    def add(self, other: "SufficientStats") -> None:
        if (
            other.dimension != self.dimension
            or other.n_targets != self.n_targets
        ):
            raise ValueError("incompatible sufficient statistics")
        self.n += other.n
        self.x_sum += other.x_sum
        self.y_sum += other.y_sum
        self.xtx += other.xtx
        self.xty += other.xty
        self.yty += other.yty

    def copy(self) -> "SufficientStats":
        return SufficientStats(
            n=self.n,
            x_sum=self.x_sum.copy(),
            y_sum=self.y_sum.copy(),
            xtx=self.xtx.copy(),
            xty=self.xty.copy(),
            yty=self.yty.copy(),
        )


def _aligned_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_value = np.asarray(x, dtype=np.float64)
    y_value = np.asarray(y, dtype=np.float64)
    if y_value.ndim == 1:
        y_value = y_value[:, None]
    if x_value.ndim != 2 or y_value.ndim != 2 or len(x_value) != len(y_value):
        raise ValueError("X and Y must be aligned two-dimensional arrays")
    return x_value, y_value


def sufficient_stats(
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_rows: int = 65536,
) -> SufficientStats:
    x_shape_value = getattr(x, "shape", None)
    y_shape_value = getattr(y, "shape", None)
    x_shape = tuple(
        np.asarray(x).shape if x_shape_value is None else x_shape_value
    )
    y_shape = tuple(
        np.asarray(y).shape if y_shape_value is None else y_shape_value
    )
    y_was_vector = len(y_shape) == 1
    if y_was_vector:
        y_shape = (y_shape[0], 1)
    if (
        len(x_shape) != 2
        or len(y_shape) != 2
        or x_shape[0] != y_shape[0]
    ):
        raise ValueError("X and Y must be aligned two-dimensional arrays")
    result = SufficientStats.zeros(x_shape[1], y_shape[1])
    for start in range(0, x_shape[0], chunk_rows):
        x_chunk = x[start : start + chunk_rows]
        y_chunk = y[start : start + chunk_rows]
        if y_was_vector:
            y_chunk = np.asarray(y_chunk)[:, None]
        result.add_rows(
            x_chunk,
            y_chunk,
        )
    return result


def select_targets(
    stats: SufficientStats, target_indices: Sequence[int]
) -> SufficientStats:
    """Return a target-column view while reusing the feature Gram matrix."""
    indices = np.asarray(target_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= stats.n_targets):
        raise ValueError("target indices are invalid")
    return SufficientStats(
        n=stats.n,
        x_sum=stats.x_sum,
        y_sum=stats.y_sum[indices],
        xtx=stats.xtx,
        xty=stats.xty[:, indices],
        yty=stats.yty[indices],
    )


@dataclass(frozen=True)
class SpectrumDiagnostics:
    trace_cov: float
    trace_cov_over_dim: float
    lambda_max_cov: float
    lambda_min_valid_cov: float
    condition_number: float
    numerical_rank: int
    numerical_tolerance: float


@dataclass(frozen=True)
class Eigensystem:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    diagnostics: SpectrumDiagnostics


def numerical_tolerance(
    lambda_max: float, n_rows: int, dimension: int, dtype=np.float64
) -> float:
    """Machine-scale symmetric-eigensolver tolerance, with no scientific floor."""
    if lambda_max <= 0.0:
        return 0.0
    return float(
        np.finfo(np.dtype(dtype)).eps
        * max(int(n_rows), int(dimension))
        * float(lambda_max)
    )


def eigensystem(gram: np.ndarray, n_rows: int) -> Eigensystem:
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gram/covariance matrix must be square")
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    largest = max(float(values[0]), 0.0) if len(values) else 0.0
    tolerance = numerical_tolerance(largest, n_rows, len(values))
    valid = values > tolerance
    rank = int(valid.sum())
    smallest = float(values[valid][-1]) if rank else float("nan")
    condition = largest / smallest if rank and smallest > 0.0 else float("inf")
    trace = float(np.trace(matrix))
    diagnostics = SpectrumDiagnostics(
        trace_cov=trace,
        trace_cov_over_dim=trace / len(values),
        lambda_max_cov=largest,
        lambda_min_valid_cov=smallest,
        condition_number=float(condition),
        numerical_rank=rank,
        numerical_tolerance=tolerance,
    )
    return Eigensystem(values, vectors, diagnostics)


@dataclass(frozen=True)
class WhiteningFit:
    mean: np.ndarray
    covariance: np.ndarray
    eigensystem: Eigensystem
    n_rows: int

    @property
    def valid_dimension(self) -> int:
        return self.eigensystem.diagnostics.numerical_rank


def fit_unlabelled_covariance(
    x_train: np.ndarray, *, chunk_rows: int = 65536
) -> WhiteningFit:
    shape_value = getattr(x_train, "shape", None)
    shape = tuple(
        np.asarray(x_train).shape if shape_value is None else shape_value
    )
    if len(shape) != 2 or shape[0] == 0:
        raise ValueError("unlabelled training features must be a non-empty matrix")
    dimension = shape[1]
    count = 0
    total = np.zeros(dimension, dtype=np.float64)
    second = np.zeros((dimension, dimension), dtype=np.float64)
    for start in range(0, shape[0], chunk_rows):
        chunk = np.asarray(
            x_train[start : start + chunk_rows], dtype=np.float64
        )
        if not np.isfinite(chunk).all():
            raise ExperimentIntegrityError("NaN or infinity in unlabelled train features")
        count += len(chunk)
        total += chunk.sum(axis=0)
        second += chunk.T @ chunk
    mean = total / count
    covariance = second / count - np.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5
    return WhiteningFit(
        mean=mean,
        covariance=covariance,
        eigensystem=eigensystem(covariance, count),
        n_rows=count,
    )


def whitening_k_grid(fit: WhiteningFit) -> tuple[int, ...]:
    values = {
        int(value)
        for value in WHITEN_K_BASE
        if value <= fit.covariance.shape[0]
    }
    values.add(int(fit.valid_dimension))
    return tuple(sorted(values))


@dataclass(frozen=True)
class WhiteningTransform:
    requested_k: int
    effective_k: int | None
    matrix: np.ndarray | None
    valid: bool
    failure_reason: str
    numerical_tolerance: float
    smallest_inverted_eigenvalue: float
    condition_number: float


def whitening_transform(fit: WhiteningFit, requested_k: int) -> WhiteningTransform:
    requested = int(requested_k)
    dimension = fit.covariance.shape[0]
    rank = fit.valid_dimension
    tolerance = fit.eigensystem.diagnostics.numerical_tolerance
    if requested < 0 or requested > dimension:
        return WhiteningTransform(
            requested, None, None, False, "requested_k_out_of_bounds",
            tolerance, float("nan"), float("nan")
        )
    if requested > rank:
        return WhiteningTransform(
            requested,
            None,
            None,
            False,
            "requested_k_exceeds_numerical_rank",
            tolerance,
            float("nan"),
            float("nan"),
        )
    values = fit.eigensystem.eigenvalues
    vectors = fit.eigensystem.eigenvectors
    scales = np.ones(dimension, dtype=np.float64)
    if requested:
        scales[:requested] = 1.0 / np.sqrt(values[:requested])
        smallest = float(values[requested - 1])
    else:
        smallest = float("nan")
    matrix = (vectors * scales[None, :]) @ vectors.T
    positive_scales = scales[scales > 0.0]
    transform_condition = float(positive_scales.max() / positive_scales.min())
    return WhiteningTransform(
        requested_k=requested,
        effective_k=requested,
        matrix=(matrix + matrix.T) * 0.5,
        valid=True,
        failure_reason="",
        numerical_tolerance=tolerance,
        smallest_inverted_eigenvalue=smallest,
        condition_number=transform_condition,
    )


@dataclass(frozen=True)
class Design:
    gram: np.ndarray
    cross: np.ndarray
    x_mean_raw: np.ndarray
    y_mean: np.ndarray
    raw_transform: np.ndarray
    n_rows: int
    eigensystem: Eigensystem


def transformed_design(
    stats: SufficientStats, transform: np.ndarray | None = None
) -> Design:
    dimension = stats.dimension
    matrix = (
        np.eye(dimension, dtype=np.float64)
        if transform is None
        else np.asarray(transform, dtype=np.float64)
    )
    if matrix.shape != (dimension, dimension):
        raise ValueError("transform has incompatible shape")
    gram = matrix.T @ stats.gram @ matrix
    gram = (gram + gram.T) * 0.5
    cross = matrix.T @ stats.cross
    return Design(
        gram=gram,
        cross=cross,
        x_mean_raw=stats.x_mean,
        y_mean=stats.y_mean,
        raw_transform=matrix,
        n_rows=stats.n,
        eigensystem=eigensystem(gram, stats.n),
    )


@dataclass(frozen=True)
class LinearModel:
    alpha: float
    lambda_absolute: float
    weights_transformed: np.ndarray
    beta_raw: np.ndarray
    intercept: np.ndarray
    numerical_rank: int
    numerical_tolerance: float


def fit_alpha(design: Design, alpha: float) -> LinearModel:
    value = float(alpha)
    if value < 0.0 or not np.isfinite(value):
        raise ValueError("alpha must be finite and non-negative")
    spectrum = design.eigensystem
    trace_scale = spectrum.diagnostics.trace_cov_over_dim
    lambda_absolute = value * trace_scale
    eigenvalues = spectrum.eigenvalues
    projected = spectrum.eigenvectors.T @ design.cross
    if value == 0.0:
        valid = eigenvalues > spectrum.diagnostics.numerical_tolerance
        inverse = np.zeros_like(eigenvalues)
        inverse[valid] = 1.0 / eigenvalues[valid]
    else:
        inverse = 1.0 / (eigenvalues + lambda_absolute)
    weights = spectrum.eigenvectors @ (inverse[:, None] * projected)
    beta_raw = design.raw_transform @ weights
    intercept = design.y_mean - design.x_mean_raw @ beta_raw
    return LinearModel(
        alpha=value,
        lambda_absolute=float(lambda_absolute),
        weights_transformed=weights,
        beta_raw=beta_raw,
        intercept=intercept,
        numerical_rank=spectrum.diagnostics.numerical_rank,
        numerical_tolerance=spectrum.diagnostics.numerical_tolerance,
    )


@dataclass(frozen=True)
class R2Scores:
    values: np.ndarray
    valid: np.ndarray
    reasons: tuple[str, ...]


def r2_per_target(y_true: np.ndarray, prediction: np.ndarray) -> R2Scores:
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if truth.ndim == 1:
        truth = truth[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if truth.shape != predicted.shape:
        raise ValueError("truth and prediction shapes differ")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ExperimentIntegrityError("NaN or infinity while evaluating R2")
    residual = np.einsum("nt,nt->t", truth - predicted, truth - predicted)
    centered = truth - truth.mean(axis=0, keepdims=True)
    total = np.einsum("nt,nt->t", centered, centered)
    scale = np.maximum(
        np.einsum("nt,nt->t", truth, truth),
        1.0,
    )
    tolerance = np.finfo(np.float64).eps * len(truth) * scale
    valid = total > tolerance
    scores = np.full(truth.shape[1], np.nan, dtype=np.float64)
    scores[valid] = 1.0 - residual[valid] / total[valid]
    reasons = tuple("" if flag else "constant_target" for flag in valid)
    return R2Scores(scores, valid, reasons)


def predict(model: LinearModel, x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != model.beta_raw.shape[0]:
        raise ValueError("feature matrix has incompatible shape")
    return value @ model.beta_raw + model.intercept


def evaluate(model: LinearModel, x: np.ndarray, y: np.ndarray) -> R2Scores:
    return r2_per_target(y, predict(model, x))


def evaluate_stats(model: LinearModel, stats: SufficientStats) -> R2Scores:
    """Evaluate R² without rescanning rows, using uncentered sufficient stats."""
    if model.beta_raw.shape != (stats.dimension, stats.n_targets):
        raise ValueError("model and evaluation statistics are incompatible")
    beta = model.beta_raw
    intercept = model.intercept
    linear_cross = np.einsum("dt,dt->t", beta, stats.xty)
    quadratic = np.einsum("dt,de,et->t", beta, stats.xtx, beta)
    beta_x_sum = beta.T @ stats.x_sum
    residual_ss = (
        stats.yty
        - 2.0 * (linear_cross + intercept * stats.y_sum)
        + quadratic
        + 2.0 * intercept * beta_x_sum
        + stats.n * np.square(intercept)
    )
    total = stats.yty - np.square(stats.y_sum) / stats.n
    scale = np.maximum(stats.yty, 1.0)
    tolerance = np.finfo(np.float64).eps * stats.n * scale
    valid = total > tolerance
    scores = np.full(stats.n_targets, np.nan, dtype=np.float64)
    # Roundoff can make an in-sample residual a tiny negative number.
    residual_ss = np.maximum(residual_ss, 0.0)
    scores[valid] = 1.0 - residual_ss[valid] / total[valid]
    reasons = tuple("" if flag else "constant_target" for flag in valid)
    return R2Scores(scores, valid, reasons)


@dataclass(frozen=True)
class TunedAlpha:
    alpha: float
    index: int
    validation_score: float
    scores_by_alpha: np.ndarray


def tune_alpha(
    design: Design,
    x_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    independent_target_indices: Sequence[int],
    alpha_grid: Iterable[float] = ALPHA_GRID,
    *,
    validation_stats: SufficientStats | None = None,
) -> TunedAlpha:
    """Select alpha on validation aggregate R²; exact ties prefer larger alpha."""
    indices = np.asarray(independent_target_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("at least one independent validation target is required")
    grid = np.asarray(list(alpha_grid), dtype=np.float64)
    if not np.array_equal(grid, ALPHA_GRID) and (
        len(grid) == 0 or np.any(grid < 0.0)
    ):
        raise ValueError("invalid alpha grid")
    values = np.full(len(grid), -np.inf, dtype=np.float64)
    for index, alpha in enumerate(grid):
        model = fit_alpha(design, float(alpha))
        if validation_stats is not None:
            scores = evaluate_stats(model, validation_stats)
        else:
            if x_validation is None or y_validation is None:
                raise ValueError(
                    "validation arrays or validation_stats must be supplied"
                )
            scores = evaluate(model, x_validation, y_validation)
        eligible = indices[scores.valid[indices]]
        if len(eligible):
            values[index] = float(np.mean(scores.values[eligible]))
    if not np.isfinite(values).any():
        raise ExperimentIntegrityError(
            "validation contains no non-constant independent target"
        )
    best = float(np.max(values))
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(best)) * 16.0
    tied = np.flatnonzero(values >= best - tolerance)
    chosen = int(tied[-1])
    return TunedAlpha(float(grid[chosen]), chosen, float(values[chosen]), values)


def direct_ridge_solution(
    x: np.ndarray, y: np.ndarray, alpha: float
) -> LinearModel:
    """Independent direct-solver reference used only by correctness tests."""
    stats = sufficient_stats(x, y)
    design = transformed_design(stats)
    scale = design.eigensystem.diagnostics.trace_cov_over_dim
    absolute = float(alpha) * scale
    if alpha == 0.0:
        largest = max(float(design.eigensystem.eigenvalues[0]), 0.0)
        tolerance = design.eigensystem.diagnostics.numerical_tolerance
        # Match the declared Gram-eigensystem rank rule exactly.  NumPy's
        # default lstsq rcond applies a different SVD rank threshold, so an
        # otherwise independent direct solve can represent a different
        # min-norm estimator on ill-conditioned real features.
        rcond = 0.0 if largest == 0.0 else float(np.sqrt(tolerance / largest))
        weights = np.linalg.lstsq(
            np.asarray(x, dtype=np.float64) - stats.x_mean,
            np.asarray(y, dtype=np.float64) - stats.y_mean,
            rcond=rcond,
        )[0]
    else:
        weights = np.linalg.solve(
            design.gram + absolute * np.eye(stats.dimension),
            design.cross,
        )
    beta = weights
    return LinearModel(
        alpha=float(alpha),
        lambda_absolute=absolute,
        weights_transformed=weights,
        beta_raw=beta,
        intercept=stats.y_mean - stats.x_mean @ beta,
        numerical_rank=design.eigensystem.diagnostics.numerical_rank,
        numerical_tolerance=design.eigensystem.diagnostics.numerical_tolerance,
    )

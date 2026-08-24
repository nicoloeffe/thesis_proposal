"""Preregistered Experiment 01: finite-sample accessibility.

The package is intentionally independent from encoder training.  It consumes a
provenance-complete bundle of frozen features, stable row identities and target
arrays, and refuses superseded two-way or capped bundles.
"""

from .constants import (
    ALPHA_GRID,
    BRANCHES,
    EXPERIMENT_VERSION,
    READOUTS,
    RESULT_COLUMNS,
)
from .errors import ExperimentIntegrityError

__all__ = [
    "ALPHA_GRID",
    "BRANCHES",
    "EXPERIMENT_VERSION",
    "ExperimentIntegrityError",
    "READOUTS",
    "RESULT_COLUMNS",
]

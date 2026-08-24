"""Experiment-specific exceptions."""


class ExperimentIntegrityError(RuntimeError):
    """Raised when a preregistered identity or leakage invariant is violated."""


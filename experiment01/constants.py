"""Immutable preregistered constants for Experiment 01."""

from __future__ import annotations

import numpy as np


EXPERIMENT_VERSION = "2.0"
INPUT_SCHEMA = "thesis.experiment01.input"
INPUT_SCHEMA_VERSION = 1
SUBSET_SCHEMA = "thesis.experiment01.subsets"
SUBSET_SCHEMA_VERSION = 1

BRANCHES = ("supervised", "jepa_horizon", "jepa_masked")
READOUTS = ("last_concat512", "meanK_concatS")
SPLITS = ("train", "validation", "test")

ALPHA_GRID = np.concatenate(
    [np.asarray([0.0], dtype=np.float64), np.logspace(-8, 4, 31, dtype=np.float64)]
)
WHITEN_K_BASE = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
CEILING_THRESHOLD = 0.01
PRIMARY_GAP_DELTA = 0.10
GAP_SENSITIVITIES = (0.05, 0.10, 0.15)
LOW_BUDGETS = (1 / 8, 1 / 4, 1 / 2, 1, 2, 4)
FRACTIONAL_BUDGETS = (1 / 8, 1 / 4, 1 / 2)

READER_FAMILIES = (
    "min_norm_ols_raw",
    "ridge_raw_common_alpha",
    "ridge_raw_tuned_alpha",
    "ridge_whiten_topk_common_alpha",
    "ridge_whiten_topk_tuned_alpha",
)

RESULT_COLUMNS = (
    "experiment_version",
    "commit_hash",
    "branch",
    "encoder_seed",
    "readout",
    "target_block",
    "target_name",
    "target_independent",
    "budget_kind",
    "budget_days_per_stock",
    "budget_stock_day_equivalents",
    "n_stock_days",
    "n_rows",
    "n_rows_over_dim",
    "subsample_seed",
    "block_anchor_quantile",
    "feature_view",
    "feature_dim",
    "whiten_k_requested",
    "whiten_k_effective",
    "pca_fraction",
    "subspace_seed",
    "reader_family",
    "alpha",
    "lambda_absolute",
    "alpha_selected",
    "trace_cov",
    "trace_cov_over_dim",
    "lambda_max_cov",
    "lambda_min_valid_cov",
    "condition_number",
    "numerical_rank",
    "numerical_tolerance",
    "train_r2",
    "val_r2",
    "test_r2",
    "full_budget_test_r2",
    "normalized_recovery",
    "ceiling_eligible",
    "fit_status",
    "failure_reason",
    "runtime_seconds",
)

EXPERIMENTAL_KEY_COLUMNS = (
    "branch",
    "encoder_seed",
    "readout",
    "target_name",
    "budget_kind",
    "budget_days_per_stock",
    "subsample_seed",
    "block_anchor_quantile",
    "feature_view",
    "whiten_k_requested",
    "pca_fraction",
    "subspace_seed",
    "reader_family",
    "alpha",
)

"""Nested stock-day label-budget generation and manifest serialization."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .constants import (
    FRACTIONAL_BUDGETS,
    SUBSET_SCHEMA,
    SUBSET_SCHEMA_VERSION,
)
from .errors import ExperimentIntegrityError
from .io import atomic_write_json, atomic_write_parquet, sha256_array, sha256_file


@dataclass(frozen=True)
class Budget:
    kind: str
    days_per_stock: float | None
    label: str
    minimum_seeds: int

    @property
    def is_fractional(self) -> bool:
        return self.kind == "fractional"

    @property
    def is_full_train(self) -> bool:
        return self.kind == "full_train"


@dataclass(frozen=True)
class SubsetSelection:
    budget: Budget
    subsample_seed: int
    row_indices: np.ndarray
    row_keys: np.ndarray
    n_stock_days: int
    stock_day_equivalents: float
    anchor_quantile: float | None
    per_stock_anchors: Mapping[int, float]

    @property
    def n_rows(self) -> int:
        return int(len(self.row_indices))

    @property
    def row_key_sha256(self) -> str:
        return sha256_array(self.row_keys.astype("U", copy=False))


def minimum_seed_count(days_per_stock: float | None, kind: str) -> int:
    if kind == "full_train":
        return 1
    if days_per_stock is None:
        raise ValueError("non-full budget requires days_per_stock")
    if days_per_stock <= 2:
        return 10
    if days_per_stock <= 16:
        return 5
    return 3


def budget_schedule(rows: pd.DataFrame) -> tuple[Budget, ...]:
    """Construct the preregistered powers-of-two schedule and terminal levels."""
    counts = (
        rows[["stock_id", "stock_day_id"]]
        .drop_duplicates()
        .groupby("stock_id", observed=True)["stock_day_id"]
        .count()
    )
    if len(counts) == 0:
        raise ExperimentIntegrityError("training row manifest has no stock-days")
    balanced_max = int(counts.min())
    if balanced_max < 1:
        raise ExperimentIntegrityError("at least one stock has no training day")
    values: list[Budget] = [
        Budget(
            "fractional",
            value,
            f"b_{Fraction(value).numerator}_{Fraction(value).denominator}",
            minimum_seed_count(value, "fractional"),
        )
        for value in FRACTIONAL_BUDGETS
    ]
    integer = 1
    while integer <= balanced_max:
        values.append(
            Budget(
                "integer_days",
                float(integer),
                f"b_{integer}",
                minimum_seed_count(float(integer), "integer_days"),
            )
        )
        integer *= 2
    if not any(
        value.days_per_stock == float(balanced_max)
        and value.kind == "integer_days"
        for value in values
    ):
        values.append(
            Budget(
                "balanced_max",
                float(balanced_max),
                "balanced_max",
                minimum_seed_count(float(balanced_max), "balanced_max"),
            )
        )
    if not bool((counts == balanced_max).all()):
        values.append(Budget("full_train", None, "full_train", 1))
    else:
        # The exact balanced maximum is already full_train.  Keep the required
        # terminal name and remove the duplicate balanced cell.
        values = [
            value
            for value in values
            if not (
                value.days_per_stock == float(balanced_max)
                and value.kind in {"integer_days", "balanced_max"}
            )
        ]
        values.append(Budget("full_train", None, "full_train", 1))
    return tuple(values)


def _seed_for_stock(seed: int, stock_id: int) -> np.random.SeedSequence:
    if seed < 0 or stock_id < 0:
        raise ValueError("subsampling seed and stock_id must be non-negative")
    return np.random.SeedSequence(
        [
            int(seed) & 0xFFFFFFFF,
            int(stock_id) & 0xFFFFFFFF,
            (int(stock_id) >> 32) & 0xFFFFFFFF,
            0x45585031,
        ]
    )


def _fraction_length(n_rows: int, fraction: float) -> int:
    rational = Fraction(fraction).limit_denominator()
    # Ceiling is deterministic and is recorded as part of the subset schema.
    return max(1, (n_rows * rational.numerator + rational.denominator - 1) // rational.denominator)


def nested_interval(n_rows: int, length: int, anchor: int) -> tuple[int, int]:
    """Smallest clipped interval of ``length`` centered on one fixed anchor."""
    if not (1 <= length <= n_rows):
        raise ValueError("length must be in [1, n_rows]")
    if not (0 <= anchor < n_rows):
        raise ValueError("anchor is out of bounds")
    start = anchor - (length - 1) // 2
    start = max(0, min(start, n_rows - length))
    return start, start + length


def _group_positions(rows: pd.DataFrame) -> dict[int, dict[int, np.ndarray]]:
    result: dict[int, dict[int, np.ndarray]] = {}
    for (stock, day), group in rows.groupby(
        ["stock_id", "stock_day_id"], sort=False, observed=True
    ):
        positions = group.index.to_numpy(dtype=np.int64)
        order = group["endpoint_order"].to_numpy(dtype=np.int64)
        if not np.array_equal(order, np.arange(len(order), dtype=np.int64)):
            raise ExperimentIntegrityError(
                f"stock-day ({stock}, {day}) is not in canonical endpoint order"
            )
        result.setdefault(int(stock), {})[int(day)] = positions
    return result


def selections_for_seed(
    rows: pd.DataFrame,
    seed: int,
    budgets: Iterable[Budget] | None = None,
    *,
    fixed_anchor_quantile: float | None = None,
) -> tuple[SubsetSelection, ...]:
    """Generate all nested subsets applicable to one subsampling seed."""
    if not isinstance(rows.index, pd.RangeIndex) or rows.index.start != 0:
        rows = rows.reset_index(drop=True)
    schedule = tuple(budgets) if budgets is not None else budget_schedule(rows)
    groups = _group_positions(rows)
    stocks = sorted(groups)
    permutations: dict[int, list[int]] = {}
    anchors: dict[int, int] = {}
    anchor_quantiles: dict[int, float] = {}
    for stock in stocks:
        rng = np.random.default_rng(_seed_for_stock(seed, stock))
        days = np.asarray(sorted(groups[stock]), dtype=np.int64)
        permutation = rng.permutation(days).tolist()
        permutations[stock] = [int(value) for value in permutation]
        first_positions = groups[stock][permutations[stock][0]]
        if fixed_anchor_quantile is None:
            anchor = int(rng.integers(0, len(first_positions)))
        else:
            if not 0.0 <= fixed_anchor_quantile <= 1.0:
                raise ValueError("fixed_anchor_quantile must lie in [0, 1]")
            anchor = int(round(fixed_anchor_quantile * (len(first_positions) - 1)))
        anchors[stock] = anchor
        anchor_quantiles[stock] = (
            0.5 if len(first_positions) == 1 else anchor / (len(first_positions) - 1)
        )

    selections: list[SubsetSelection] = []
    for budget in schedule:
        if budget.is_full_train:
            if seed != 0:
                continue
            indices = np.arange(len(rows), dtype=np.int64)
            n_groups = sum(len(days) for days in groups.values())
            selection_seed = -1
            per_stock_anchor: dict[int, float] = {}
            mean_anchor = None
            equivalents = float(n_groups)
        elif seed >= budget.minimum_seeds:
            continue
        elif budget.is_fractional:
            chosen: list[np.ndarray] = []
            for stock in stocks:
                day = permutations[stock][0]
                positions = groups[stock][day]
                length = _fraction_length(len(positions), float(budget.days_per_stock))
                start, end = nested_interval(len(positions), length, anchors[stock])
                chosen.append(positions[start:end])
            indices = np.sort(np.concatenate(chosen))
            n_groups = len(stocks)
            selection_seed = seed
            per_stock_anchor = dict(anchor_quantiles)
            mean_anchor = float(np.mean(list(anchor_quantiles.values())))
            equivalents = float(budget.days_per_stock) * len(stocks)
        else:
            n_days = int(budget.days_per_stock)
            chosen = []
            for stock in stocks:
                days = permutations[stock][:n_days]
                if len(days) != n_days:
                    raise ExperimentIntegrityError(
                        f"stock {stock} has fewer than {n_days} training days"
                    )
                chosen.extend(groups[stock][day] for day in days)
            indices = np.sort(np.concatenate(chosen))
            n_groups = n_days * len(stocks)
            selection_seed = seed
            per_stock_anchor = {}
            mean_anchor = None
            equivalents = float(n_groups)
        keys = rows.iloc[indices]["row_key"].astype(str).to_numpy(dtype="U")
        selections.append(
            SubsetSelection(
                budget=budget,
                subsample_seed=selection_seed,
                row_indices=indices,
                row_keys=keys,
                n_stock_days=n_groups,
                stock_day_equivalents=equivalents,
                anchor_quantile=mean_anchor,
                per_stock_anchors=per_stock_anchor,
            )
        )
    _validate_nested(selections)
    return tuple(selections)


def _validate_nested(selections: Iterable[SubsetSelection]) -> None:
    by_seed: dict[int, list[SubsetSelection]] = {}
    for selection in selections:
        if selection.subsample_seed < 0:
            continue
        by_seed.setdefault(selection.subsample_seed, []).append(selection)
    for seed, values in by_seed.items():
        ordered = sorted(
            values,
            key=lambda item: (
                float("inf")
                if item.budget.days_per_stock is None
                else item.budget.days_per_stock
            ),
        )
        previous: set[int] = set()
        for selection in ordered:
            current = set(selection.row_indices.tolist())
            if previous and not previous.issubset(current):
                raise ExperimentIntegrityError(
                    f"label subsets are not nested for subsampling seed {seed}"
                )
            previous = current


def generate_all_selections(rows: pd.DataFrame) -> tuple[SubsetSelection, ...]:
    schedule = budget_schedule(rows)
    maximum = max(
        budget.minimum_seeds for budget in schedule if not budget.is_full_train
    )
    values: list[SubsetSelection] = []
    for seed in range(maximum):
        values.extend(selections_for_seed(rows, seed, schedule))
    keys = [
        (
            value.budget.label,
            value.subsample_seed,
            value.anchor_quantile,
        )
        for value in values
    ]
    if len(keys) != len(set(keys)):
        raise ExperimentIntegrityError("duplicate subset experimental keys")
    full = [value for value in values if value.budget.is_full_train]
    if len(full) != 1 or full[0].subsample_seed != -1:
        raise ExperimentIntegrityError("full_train must have one deterministic realization")
    return tuple(values)


def anchor_sensitivity(
    rows: pd.DataFrame, seeds: Iterable[int] = range(10)
) -> tuple[SubsetSelection, ...]:
    fractional = [
        budget for budget in budget_schedule(rows) if budget.is_fractional
    ]
    values: list[SubsetSelection] = []
    for quantile in (0.0, 0.5, 1.0):
        for seed in seeds:
            values.extend(
                selections_for_seed(
                    rows,
                    int(seed),
                    fractional,
                    fixed_anchor_quantile=quantile,
                )
            )
    return tuple(values)


def write_subset_manifests(
    rows: pd.DataFrame,
    selections: Iterable[SubsetSelection],
    out_dir: str | Path,
    *,
    source_row_key_sha256: str,
) -> dict[str, object]:
    destination = Path(out_dir)
    subset_dir = destination / "subset_manifests"
    subset_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for selection in selections:
        seed_label = (
            "deterministic"
            if selection.subsample_seed < 0
            else f"seed_{selection.subsample_seed:03d}"
        )
        relative = Path("subset_manifests") / selection.budget.label / f"{seed_label}.parquet"
        path = destination / relative
        selected = rows.iloc[selection.row_indices].copy()
        selected.insert(0, "source_row_position", selection.row_indices)
        selected["budget_kind"] = selection.budget.kind
        selected["budget_days_per_stock"] = selection.budget.days_per_stock
        selected["subsample_seed"] = selection.subsample_seed
        atomic_write_parquet(selected, path)
        records.append(
            {
                "budget_kind": selection.budget.kind,
                "budget_label": selection.budget.label,
                "budget_days_per_stock": selection.budget.days_per_stock,
                "budget_stock_day_equivalents": selection.stock_day_equivalents,
                "subsample_seed": selection.subsample_seed,
                "n_rows": selection.n_rows,
                "n_stock_days": selection.n_stock_days,
                "block_anchor_quantile": selection.anchor_quantile,
                "per_stock_anchor_quantiles": {
                    str(key): value
                    for key, value in sorted(selection.per_stock_anchors.items())
                },
                "row_key_sha256": selection.row_key_sha256,
                "path": str(relative),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_name": SUBSET_SCHEMA,
        "schema_version": SUBSET_SCHEMA_VERSION,
        "fractional_length_rounding": "ceil(n_valid_endpoints * fraction)",
        "anchor_algorithm": "uniform_endpoint_anchor_then_clipped_nested_center_interval.v1",
        "day_permutation_algorithm": "numpy.default_rng.SeedSequence(seed,stock).permutation.v1",
        "source_train_row_key_sha256": source_row_key_sha256,
        "subsets": records,
    }
    atomic_write_json(destination / "subset_manifest.json", payload)
    return payload

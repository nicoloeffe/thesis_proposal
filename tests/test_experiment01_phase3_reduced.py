from __future__ import annotations

import json
from pathlib import Path

from experiment01.phase3_reduced import (
    EXPECTED_EVALUATION_MODELS,
    EXPECTED_LOGICAL_CELLS,
    EXPECTED_SELECTION_MODELS,
    PHASE3_REDUCED_CONTROL_BUDGETS,
    PHASE3_REDUCED_PRIMARY_BUDGETS,
    PHASE3_REDUCED_READER_SEEDS,
    PHASE3_REDUCED_SPECTRAL_ARMS,
    build_phase3_reduced_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "validation/experiment01/execution_20260730/phase1"


def _inventory():
    payload = json.loads((PHASE1 / "subset_manifest.json").read_text())
    return build_phase3_reduced_inventory(payload)


def test_reduced_inventory_has_confirmed_cardinality_and_no_capacity_sweep():
    selection, evaluation = _inventory()
    assert len(selection) == EXPECTED_SELECTION_MODELS == 648
    assert len(evaluation) == EXPECTED_EVALUATION_MODELS == 648
    assert selection.logical_job_key.nunique() == EXPECTED_LOGICAL_CELLS == 216
    assert evaluation.logical_job_key.nunique() == EXPECTED_LOGICAL_CELLS
    assert "capacity_sensitivity" not in set(selection.job_family)
    assert set(selection.width) == {256}


def test_reduced_primary_keeps_two_adjacent_low_budgets_and_full():
    selection, _ = _inventory()
    primary = selection.loc[selection.job_family == "primary_directional"]
    assert set(primary.budget_label) == set(PHASE3_REDUCED_PRIMARY_BUDGETS)
    assert PHASE3_REDUCED_PRIMARY_BUDGETS == ("b_1_4", "b_1_2", "full_train")
    low = primary.loc[primary.budget_label != "full_train"]
    assert set(low.subsample_seed) == {0, 1, 2}
    assert set(primary.branch) == {"jepa_horizon", "supervised"}
    assert set(primary["transform"]) == {"native", "full_whitened"}
    assert set(primary.encoder_seed) == {0, 1, 2}


def test_reduced_controls_and_spectral_scope_are_exact():
    selection, _ = _inventory()
    controls = selection.loc[selection.job_family == "specificity_control"]
    assert set(controls.target_block) == {"volatility", "timing"}
    assert set(controls.budget_label) == set(PHASE3_REDUCED_CONTROL_BUDGETS)
    spectral = selection.loc[selection.job_family == "spectral_diagnostic"]
    assert set(spectral.branch) == {"jepa_horizon"}
    assert set(spectral.target_block) == {"directional"}
    assert set(spectral.spectral_arm) == set(PHASE3_REDUCED_SPECTRAL_ARMS)


def test_reduced_replication_is_paired_and_exactly_three():
    selection, evaluation = _inventory()
    assert selection.groupby("logical_job_key").size().eq(3).all()
    assert evaluation.groupby("logical_job_key").size().eq(3).all()
    assert set(evaluation.reader_seed) == set(PHASE3_REDUCED_READER_SEEDS)
    paired = selection.loc[
        selection.job_family.isin(("primary_directional", "specificity_control"))
    ]
    key = [
        "job_family",
        "encoder_seed",
        "target_block",
        "transform",
        "budget_label",
        "subsample_seed",
        "weight_decay",
    ]
    assert paired.groupby(key).branch.nunique().eq(2).all()

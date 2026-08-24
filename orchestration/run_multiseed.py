#!/usr/bin/env python3
"""
Orchestration-only launcher for the 9-run multiseed training sweep
(3 arms x 3 seeds), executed strictly in series on a single GPU.

This script does NOT contain any training logic. It only builds argv lists
for the three existing, validated training scripts and runs them as
subprocesses, one at a time:

    training.train_jepa_horizon    (--ckpt_dir)
    training.train_jepa_masked     (--ckpt_dir)
    training.train_supervised_grid (--out_dir, requires --jepa_ckpt)

DAG:
    1) jepa_horizon: seed 0, 1, 2
    2) jepa_masked:  seed 0, 1, 2
    3) supervised:   seed 0, 1, 2  (all three use jepa_horizon/seed0/last.pt
                                    as --jepa_ckpt, so stock_stats normalization
                                    is identical across every arm/seed)

Resumability: a run is considered done and is skipped if its output
directory already contains last.pt.

Out of scope (deliberately not done here): probing, R^2/accessibility,
aggregation, plotting. Training + checkpoints + logs only.

Usage:
    ../rocm_env/bin/python orchestration/run_multiseed.py
(run from the repo root, or paths below are resolved relative to this
file's location regardless of cwd)
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# The venv name/location was described loosely ("../rocm_env or something
# like that, one level up from the repo"). Auto-detect among the likely
# names instead of hardcoding a guess.
_VENV_CANDIDATES = ["rocm_env", "rocmvenv", "rocm_venv", "venv_rocm", "venv"]


def _find_venv_python() -> Path:
    for name in _VENV_CANDIDATES:
        candidate = REPO_ROOT.parent / name / "bin" / "python"
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Could not auto-detect the ROCm venv next to the repo root "
        f"({REPO_ROOT.parent}). Looked for: {_VENV_CANDIDATES}. "
        "Edit VENV_PYTHON in this script to point at the right "
        "<venv>/bin/python explicitly."
    )


VENV_PYTHON = _find_venv_python()

DATASET = "data/lobench_processed.npz"
CKPT_ROOT = Path("checkpoints/multiseed")
LOG_ROOT = Path("logs")

SEEDS = [0, 1, 2]
SPLIT_SEED = 0  # fixed across every run: keeps train/val split identical

# Fixed hyperparameters, identical across every run (only --seed varies).
# max_train/val_samples caps the (huge) full dataset down to the subset size
# used in past training runs, so a single epoch finishes in a reasonable time.
COMMON_ARGS = [
    "--dataset", DATASET,
    "--split_seed", str(SPLIT_SEED),
    "--save_every", "1",
    "--d_model", "128",
    "--epochs", "20",
    "--K", "20",
    "--max_train_samples", "500000",
    "--max_val_samples", "50000",
]

# Reference checkpoint that all 3 supervised runs inherit stock_stats from.
JEPA_HORIZON_SEED0_CKPT = CKPT_ROOT / "jepa_horizon" / "seed0" / "last.pt"


# --------------------------------------------------------------------------
# Run definitions
# --------------------------------------------------------------------------

class Run:
    def __init__(self, arm: str, seed: int, module: str, extra_args: list[str],
                 out_dir: Path, depends_on: Path | None = None):
        self.arm = arm
        self.seed = seed
        self.module = module
        self.extra_args = extra_args
        self.out_dir = out_dir
        self.depends_on = depends_on  # a checkpoint file that must exist first
        self.status = "PENDING"
        self.elapsed_s = 0.0

    @property
    def name(self) -> str:
        return f"{self.arm}_seed{self.seed}"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.name}.log"

    def build_argv(self) -> list[str]:
        return [str(VENV_PYTHON), "-u", "-m", self.module, *COMMON_ARGS,
                "--seed", str(self.seed), *self.extra_args]


def build_runs() -> list[Run]:
    runs: list[Run] = []

    for seed in SEEDS:
        out_dir = CKPT_ROOT / "jepa_horizon" / f"seed{seed}"
        runs.append(Run(
            arm="jepa_horizon", seed=seed,
            module="training.train_jepa_horizon",
            extra_args=["--ckpt_dir", str(out_dir)],
            out_dir=out_dir,
        ))

    for seed in SEEDS:
        out_dir = CKPT_ROOT / "jepa_masked" / f"seed{seed}"
        runs.append(Run(
            arm="jepa_masked", seed=seed,
            module="training.train_jepa_masked",
            extra_args=["--ckpt_dir", str(out_dir)],
            out_dir=out_dir,
        ))

    for seed in SEEDS:
        out_dir = CKPT_ROOT / "supervised" / f"seed{seed}"
        runs.append(Run(
            arm="supervised", seed=seed,
            module="training.train_supervised_grid",
            extra_args=["--out_dir", str(out_dir),
                        "--jepa_ckpt", str(JEPA_HORIZON_SEED0_CKPT)],
            out_dir=out_dir,
            depends_on=JEPA_HORIZON_SEED0_CKPT,
        ))

    return runs


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_one(run: Run) -> None:
    if (run.out_dir / "last.pt").exists():
        run.status = "SKIPPED (already has last.pt)"
        print(f"[{_now()}] SKIP  {run.name}: last.pt already present in {run.out_dir}")
        return

    if run.depends_on is not None and not run.depends_on.exists():
        run.status = f"BLOCKED (missing dependency {run.depends_on})"
        print(f"[{_now()}] BLOCK {run.name}: dependency {run.depends_on} not found")
        return

    run.out_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    argv = run.build_argv()

    print(f"[{_now()}] START {run.name}")
    print(f"           cmd: {' '.join(argv)}")
    print(f"           log: {run.log_path}")

    t0 = dt.datetime.now()
    with open(run.log_path, "a") as log_file:
        log_file.write(f"\n=== {_now()} START {run.name} ===\n")
        log_file.write(f"cmd: {' '.join(argv)}\n")
        log_file.flush()
        proc = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        elapsed = (dt.datetime.now() - t0).total_seconds()
        log_file.write(f"=== {_now()} END {run.name} (returncode={proc.returncode}, "
                        f"elapsed={elapsed/3600:.2f}h) ===\n")

    run.elapsed_s = elapsed
    if proc.returncode == 0:
        run.status = "COMPLETED"
        print(f"[{_now()}] DONE  {run.name} ({elapsed/3600:.2f}h)")
    else:
        run.status = f"FAILED (returncode={proc.returncode})"
        print(f"[{_now()}] FAIL  {run.name} (returncode={proc.returncode}), see {run.log_path}")


def print_summary(runs: list[Run]) -> None:
    print("\n" + "=" * 72)
    print("MULTISEED RUN SUMMARY")
    print("=" * 72)
    for run in runs:
        hrs = f"{run.elapsed_s/3600:.2f}h" if run.elapsed_s else "-"
        print(f"  {run.name:<22} {run.status:<45} {hrs}")

    n_completed = sum(1 for r in runs if r.status == "COMPLETED")
    n_skipped = sum(1 for r in runs if r.status.startswith("SKIPPED"))
    n_failed = sum(1 for r in runs if r.status.startswith("FAILED"))
    n_blocked = sum(1 for r in runs if r.status.startswith("BLOCKED"))
    print("-" * 72)
    print(f"  completed={n_completed} skipped={n_skipped} "
          f"failed={n_failed} blocked={n_blocked} total={len(runs)}")
    print("=" * 72)


def main() -> None:
    print(f"Repo root   : {REPO_ROOT}")
    print(f"Venv python : {VENV_PYTHON}")
    print(f"Dataset     : {DATASET}")
    print(f"Checkpoints : {CKPT_ROOT}")
    print(f"Logs        : {LOG_ROOT}")
    print()

    runs = build_runs()
    try:
        for run in runs:
            run_one(run)
    except KeyboardInterrupt:
        print(f"\n[{_now()}] Interrupted by user. Partial summary below.")
    finally:
        print_summary(runs)


if __name__ == "__main__":
    main()

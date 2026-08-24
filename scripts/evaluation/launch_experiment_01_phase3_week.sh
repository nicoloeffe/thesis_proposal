#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../..")"
PYTHON="${PROJECT_ROOT}/../rocm_env/bin/python"
RUNNER="${PROJECT_ROOT}/scripts/evaluation/run_experiment_01_phase3_week.py"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ROCm Python not found or not executable: ${PYTHON}" >&2
  exit 1
fi
if ! systemd-inhibit --list >/dev/null 2>&1; then
  echo "Cannot acquire or inspect a systemd sleep inhibitor; refusing unattended run." >&2
  exit 1
fi

exec systemd-inhibit \
  --what=sleep:idle \
  --who=experiment01-phase3 \
  --why="Experiment 01 Phase III production run" \
  --mode=block \
  "${PYTHON}" "${RUNNER}" "$@"

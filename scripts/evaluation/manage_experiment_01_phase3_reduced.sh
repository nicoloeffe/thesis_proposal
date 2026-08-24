#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../..")"
SESSION_NAME="experiment01-phase3-r"
LAUNCHER="${PROJECT_ROOT}/scripts/evaluation/launch_experiment_01_phase3_reduced.sh"
RUNNER_DIR="${PROJECT_ROOT}/validation/experiment01/execution_20260730/phase3_reduced_runner"

usage() {
  echo "Usage: $0 {start|status|logs|attach|stop}"
}

case "${1:-}" in
  start)
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
      echo "Session ${SESSION_NAME} is already running." >&2
      exit 1
    fi
    mkdir -p "${RUNNER_DIR}"
    tmux new-session -d -s "${SESSION_NAME}" "${LAUNCHER}"
    echo "Started ${SESSION_NAME}."
    ;;
  status)
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
      echo "tmux: running"
    else
      echo "tmux: not running"
    fi
    if [[ -f "${RUNNER_DIR}/status.json" ]]; then
      "${PROJECT_ROOT}/../rocm_env/bin/python" -m json.tool "${RUNNER_DIR}/status.json"
    else
      echo "No runner status has been written yet."
    fi
    ;;
  logs)
    if [[ -d "${RUNNER_DIR}/logs" ]]; then
      latest_log="$(find "${RUNNER_DIR}/logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
      if [[ -n "${latest_log}" ]]; then
        tail -n 100 "${latest_log}"
      else
        echo "No stage log has been written yet."
      fi
    else
      echo "No log directory has been written yet."
    fi
    ;;
  attach)
    exec tmux attach-session -t "${SESSION_NAME}"
    ;;
  stop)
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
      echo "Session ${SESSION_NAME} is not running." >&2
      exit 1
    fi
    tmux send-keys -t "${SESSION_NAME}" C-c
    echo "Sent a graceful interrupt to ${SESSION_NAME}."
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

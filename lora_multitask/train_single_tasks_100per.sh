#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_100per.sh"
SAVE_ROOT="${SAVE_ROOT:-runs/single_tasks_100per}"
AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"
LOG_APPEND="${LOG_APPEND:-0}"

run_task() {
  local task="$1"
  local loss_weights="$2"
  local save_dir="${SAVE_ROOT}/${task}"
  echo "================================================"
  echo "[single-task][100per] task=${task} save_dir=${save_dir}"
  echo "================================================"
  TASKS="${task}" \
  LOSS_WEIGHTS="${loss_weights}" \
  SAVE_DIR="${save_dir}" \
  AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN}" \
  LOG_APPEND="${LOG_APPEND}" \
  bash "${BASE_SCRIPT}"
}


run_task cnt "0,0,1"


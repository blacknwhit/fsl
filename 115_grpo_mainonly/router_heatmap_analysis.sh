#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/115_grpo_mainonly/router_heatmap_analysis.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"
SCRIPT_PATH="${PROJECT_ROOT}/115_grpo_mainonly/router_heatmap_analysis.py"

CHECKPOINT_DEFAULT="${PROJECT_ROOT}/runs/115_mainonly_1_10per/best_combo.pt"
CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_DEFAULT}}"
OUTPUT_DIR_DEFAULT="$(dirname "${CHECKPOINT}")/router_vis"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_DEFAULT}}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE}"
fi

# Keep import behavior aligned with train/test launchers.
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"

# Prefer host driver libs and avoid CUDA 803 caused by incompatible compat stubs.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: script not found: ${SCRIPT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[run] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[run] PYTHONPATH=${PYTHONPATH}"
echo "[run] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[run] cmd: ${cmd[*]}"

"${cmd[@]}"

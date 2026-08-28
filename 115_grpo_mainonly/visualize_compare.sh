#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/115_grpo_mainonly/visualize_compare.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"
SCRIPT_PATH="${PROJECT_ROOT}/115_grpo_mainonly/visualize_compare.py"

LORA_CHECKPOINT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/lora_20per/best_combo.pt"
OURS_CHECKPOINT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/415_20per/best_combo.pt"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${LORA_CHECKPOINT_DEFAULT}}"
OURS_CHECKPOINT="${OURS_CHECKPOINT:-${OURS_CHECKPOINT_DEFAULT}}"

OUTPUT_DIR_DEFAULT="$(dirname "${OURS_CHECKPOINT}")/vis_compare"
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
if [[ ! -f "${LORA_CHECKPOINT}" ]]; then
  echo "ERROR: LoRA checkpoint not found: ${LORA_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${OURS_CHECKPOINT}" ]]; then
  echo "ERROR: Ours checkpoint not found: ${OURS_CHECKPOINT}" >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --lora-ckpt "${LORA_CHECKPOINT}"
  --ours-ckpt "${OURS_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[run] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[run] PYTHONPATH=${PYTHONPATH}"
echo "[run] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[run] LORA_CHECKPOINT=${LORA_CHECKPOINT}"
echo "[run] OURS_CHECKPOINT=${OURS_CHECKPOINT}"
echo "[run] cmd: ${cmd[*]}"

"${cmd[@]}"
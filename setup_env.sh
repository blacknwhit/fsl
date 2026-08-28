#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"

CUDA_VERSION="${CUDA_VERSION:-}"
if [[ -z "${CUDA_VERSION}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
fi

TORCH_BUILD="cpu"
if [[ -n "${CUDA_VERSION}" ]]; then
  CUDA_MAJOR="${CUDA_VERSION%%.*}"
  CUDA_MINOR="${CUDA_VERSION#*.}"
  CUDA_MINOR="${CUDA_MINOR%%.*}"

  if [[ "${CUDA_MAJOR}" -eq 11 && "${CUDA_MINOR}" -ge 8 ]]; then
    TORCH_BUILD="cu118"
  elif [[ "${CUDA_MAJOR}" -ge 12 ]]; then
    # CUDA 12.x drivers are backward compatible with cu121 wheels.
    TORCH_BUILD="cu121"
  fi
fi

if [[ "${TORCH_BUILD}" == "cpu" ]]; then
  "${PYTHON_BIN}" -m pip install --upgrade \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
else
  "${PYTHON_BIN}" -m pip install --upgrade \
    "torch==${TORCH_VERSION}+${TORCH_BUILD}" \
    "torchvision==${TORCHVISION_VERSION}+${TORCH_BUILD}" \
    "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_BUILD}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_BUILD}"
fi

"${PYTHON_BIN}" -m pip install --upgrade -r requirements-fsl.txt

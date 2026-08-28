#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PLAIN_LORA="${PLAIN_LORA:-1}"
export LORA_MOE="${LORA_MOE:-1}"
export USE_PRIVATE_EXPERTS="${USE_PRIVATE_EXPERTS:-1}"
export USE_SHARED_EXPERTS="${USE_SHARED_EXPERTS:-0}"
export SAVE_DIR="${SAVE_DIR:-runs/415_10per_lora_private_experts}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$(basename "${SAVE_DIR}")}"

exec bash "${SCRIPT_DIR}/train_100per.sh" "$@"

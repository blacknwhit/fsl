#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_100per_from_scratch_base.sh"

STAGE2_INIT_CHECKPOINT="" \
STAGE1_EPOCHS="${STAGE1_EPOCHS:-200}" \
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}" \
NUM_EXPERTS_PRIVATE="${NUM_EXPERTS_PRIVATE:-3}" \
NUM_EXPERTS_SHARED="${NUM_EXPERTS_SHARED:-6}" \
LORA_RANK="${LORA_RANK:-16}" \
SAVE_DIR="${SAVE_DIR:-runs/415_100per_s1e200_s2e50_p3_s6_r16}" \
bash "${BASE_SCRIPT}"

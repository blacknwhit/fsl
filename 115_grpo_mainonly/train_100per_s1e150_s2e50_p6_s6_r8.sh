#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_100per_from_scratch_base.sh"

STAGE2_INIT_CHECKPOINT="/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/415_100per_s1e150_s2e50_p6_s6_r8/stage1_best.pt" \
STAGE1_EPOCHS="${STAGE1_EPOCHS:-150}" \
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}" \
NUM_EXPERTS_PRIVATE="${NUM_EXPERTS_PRIVATE:-6}" \
NUM_EXPERTS_SHARED="${NUM_EXPERTS_SHARED:-6}" \
LORA_RANK="${LORA_RANK:-8}" \
SAVE_DIR="${SAVE_DIR:-runs/415_100per_s1e150_s2e50_p6_s6_r8}" \
bash "${BASE_SCRIPT}"

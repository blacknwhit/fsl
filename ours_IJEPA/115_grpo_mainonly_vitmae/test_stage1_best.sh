#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_TEST_SCRIPT="${SCRIPT_DIR}/test.sh"

CKPT_DEFAULT="${REPO_ROOT}/runs/115_grpo_mainonly_vitmae/stage1_best.pt"
CKPT="${1:-${CKPT_DEFAULT}}"

if [[ "${CKPT}" != /* ]]; then
  CKPT="${REPO_ROOT}/${CKPT}"
fi

if [[ ! -f "${BASE_TEST_SCRIPT}" ]]; then
  echo "Base test script not found: ${BASE_TEST_SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[run] base_test: ${BASE_TEST_SCRIPT}"
  echo "[run] ckpt: ${CKPT}"
  echo "[dry-run] command prepared, not executing eval."
  exit 0
fi

exec bash "${BASE_TEST_SCRIPT}" "${CKPT}"

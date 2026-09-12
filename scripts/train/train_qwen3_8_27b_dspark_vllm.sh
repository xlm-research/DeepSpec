#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export CONFIG_PATH="${REPO_ROOT}/config/dspark/dspark_qwen3_8_27b_vllm.py"
export OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/output/dspark_qwen3_8_27b_vllm}
export VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-${PYTHON_BIN:-python}}
export VLLM_SOURCE_DIR=${VLLM_SOURCE_DIR:-${REPO_ROOT}/vllm}
if [[ "${BOUNDED_OFFLINE:-true}" != "true" ]]; then
    echo "The Qwen vLLM launcher requires BOUNDED_OFFLINE=true." >&2
    exit 1
fi
exec bash "${SCRIPT_DIR}/train_qwen3_8_27b_dspark_128gpu.sh"

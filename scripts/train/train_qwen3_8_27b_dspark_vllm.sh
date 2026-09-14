#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export CONFIG_PATH="${REPO_ROOT}/config/dspark/dspark_qwen3_8_27b_vllm.py"
export OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/output/dspark_qwen3_8_27b_vllm}
export VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-$(command -v python)}
export VLLM_SOURCE_DIR=${VLLM_SOURCE_DIR:-${REPO_ROOT}/vllm}
# Preserve the established eight-GPU-node producer layout across draft changes.
export TARGET_CONTEXT_PARALLEL_SIZE=${TARGET_CONTEXT_PARALLEL_SIZE:-1}
export TARGET_TENSOR_PARALLEL_SIZE=${TARGET_TENSOR_PARALLEL_SIZE:-4}
export TARGET_FSDP_SIZE=${TARGET_FSDP_SIZE:-2}
echo "vLLM interpreter=${VLLM_PYTHON_BIN}, source=${VLLM_SOURCE_DIR}"
if [[ "${BOUNDED_OFFLINE:-true}" != "true" ]]; then
    echo "The Qwen vLLM launcher requires BOUNDED_OFFLINE=true." >&2
    exit 1
fi
exec bash "${SCRIPT_DIR}/train_qwen3_8_27b_dspark_128gpu.sh"

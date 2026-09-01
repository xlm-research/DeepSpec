#!/usr/bin/env bash
set -euo pipefail

# Single-node, one-optimizer-step Qwen3.8 DSpark smoke with real 128K records.
# The script uses every visible GPU, groups them with CP=2, and selects exactly
# one packed record per effective data-parallel replica.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BASE_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
PACKED_SOURCE_PATH=${PACKED_SOURCE_PATH:-${BASE_DIR}/train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl}
SMOKE_ROOT=${SMOKE_ROOT:-${BASE_DIR}/output/qwen3_8_27b_dspark_cp2_128k_smoke}
MASTER_PORT=${MASTER_PORT:-29621}
DRY_RUN=${DRY_RUN:-false}

RESOLVED_NNODES=${SENSECORE_PYTORCH_NNODES:-${NNODES:-${WORLD_SIZE:-1}}}
if [[ "${RESOLVED_NNODES}" != "1" ]]; then
    echo "This smoke launcher is single-node only; resolved NNODES=${RESOLVED_NNODES}." >&2
    exit 1
fi
if [[ -n "${LOCAL_RANK:-}" ]]; then
    echo "Run this smoke launcher directly, not from inside a torchrun worker." >&2
    exit 1
fi
command -v "${PYTHON_BIN}" >/dev/null || {
    echo "PYTHON_BIN is not available on PATH: ${PYTHON_BIN}" >&2
    exit 1
}
if [[ ! -f "${PACKED_SOURCE_PATH}" ]]; then
    echo "Packed 256K source does not exist: ${PACKED_SOURCE_PATH}" >&2
    exit 1
fi

NPROC_PER_NODE=$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "No CUDA GPU is visible to PyTorch; detected ${NPROC_PER_NODE}." >&2
    exit 1
fi
if ((NPROC_PER_NODE < 2 || NPROC_PER_NODE % 2 != 0)); then
    echo "CP2 smoke requires an even number of at least 2 visible GPUs; got ${NPROC_PER_NODE}." >&2
    exit 1
fi

CONTEXT_PARALLEL_SIZE=2
FSDP_SIZE=$((NPROC_PER_NODE / CONTEXT_PARALLEL_SIZE))
DATA_PARALLEL_SIZE=$((NPROC_PER_NODE / CONTEXT_PARALLEL_SIZE))
SMOKE_DATA_PATH=${SMOKE_DATA_PATH:-${SMOKE_ROOT}/data/packed_128k_first${DATA_PARALLEL_SIZE}.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SMOKE_ROOT}/run_${NPROC_PER_NODE}gpu}
DATA_BATCH_CACHE_DIR=${DATA_BATCH_CACHE_DIR:-${OUTPUT_ROOT}/target_data_batch_cache}

"${PYTHON_BIN}" scripts/data/subset_jsonl.py \
    --input-path "${PACKED_SOURCE_PATH}" \
    --output-path "${SMOKE_DATA_PATH}" \
    --num-records "${DATA_PARALLEL_SIZE}" \
    --minimum-packed-tokens 131072

echo "Prepared real-128K smoke input:"
echo "  visible GPUs=${NPROC_PER_NODE}, CP=2, DP=${DATA_PARALLEL_SIZE}, DP_SHARD=${FSDP_SIZE}"
echo "  packed records=${DATA_PARALLEL_SIZE}, max length=131072, optimizer steps=1"
echo "  source=${SMOKE_DATA_PATH}"
echo "  transient target cache=${DATA_BATCH_CACHE_DIR}"
echo "  output=${OUTPUT_ROOT}"

exec env \
    NNODES=1 \
    NODE_RANK=0 \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${MASTER_PORT}" \
    CONTEXT_PARALLEL_SIZE=2 \
    FSDP_SIZE="${FSDP_SIZE}" \
    MAX_LENGTH=131072 \
    SOURCE_JSONL_PATH="${SMOKE_DATA_PATH}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    ONLINE_TARGET=true \
    DATA_BATCH_SIZE=1 \
    DATA_BATCH_CACHE_DIR="${DATA_BATCH_CACHE_DIR}" \
    LOCAL_BATCH_SIZE=1 \
    GLOBAL_BATCH_SIZE="${DATA_PARALLEL_SIZE}" \
    NUM_TRAIN_EPOCHS=1 \
    MAX_TRAIN_STEPS=1 \
    LOGGING_STEPS=1 \
    SAVE_STEPS=1 \
    SAVE_CHECKPOINTS=false \
    TORCH_COMPILE=false \
    TARGET_CACHE_FSDP=true \
    PRODUCTION_RUN=false \
    DRY_RUN="${DRY_RUN}" \
    bash scripts/train/train_qwen3_8_27b_dspark_128gpu.sh

#!/usr/bin/env bash
set -euo pipefail

# Qwen3.6 target-cache generation followed by DSpark training.
# Run once per node. WORLD_SIZE/RANK are node count/rank; train.py and the
# cache builder spawn one process per visible local GPU.

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-${ROOT_DIR}/config/dspark/dspark_qwen3_6_27b.py}
MODEL_PATH=${MODEL_PATH:-/mnt/afs_agents/share_models/Qwen/Qwen3.6-27B}
DATA_PATH=${DATA_PATH:-}
RUN_DIR=${RUN_DIR:-${ROOT_DIR}/output/qwen3_6_dspark}
CACHE_DIR=${CACHE_DIR:-${RUN_DIR}/target_cache}

CP_SIZE=${CP_SIZE:-8}
FSDP_SIZE=${FSDP_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-262144}
NUM_ANCHORS=${NUM_ANCHORS:-64}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MIN_LOSS_TOKENS=${MIN_LOSS_TOKENS:-1}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-50}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}
MEDIA_ROOT=${MEDIA_ROOT:-}
MEDIA_URI_MAP=${MEDIA_URI_MAP:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_NODES=${NUM_NODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export WORLD_SIZE=${NUM_NODES}
export RANK=${NODE_RANK}
unset LOCAL_RANK

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_IDS[@]}
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))
MODEL_PARALLEL_SIZE=$((CP_SIZE * FSDP_SIZE))
EFFECTIVE_DATA_REPLICAS=$((TOTAL_GPUS / CP_SIZE))

if [[ -z "${DATA_PATH}" ]]; then
    echo "Set DATA_PATH to one OpenAI/conversations JSONL dataset." >&2
    exit 1
fi
if (( TOTAL_GPUS % MODEL_PARALLEL_SIZE != 0 )); then
    echo "TOTAL_GPUS=${TOTAL_GPUS} must be divisible by CP*FSDP=${MODEL_PARALLEL_SIZE}." >&2
    exit 1
fi
if (( NUM_ANCHORS < CP_SIZE || NUM_ANCHORS % CP_SIZE != 0 )); then
    echo "NUM_ANCHORS must be a positive multiple of CP_SIZE." >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % EFFECTIVE_DATA_REPLICAS != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by ${EFFECTIVE_DATA_REPLICAS} data replicas." >&2
    exit 1
fi
if (( NUM_NODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" ]]; then
    echo "Multi-node execution requires a reachable rank-0 MASTER_ADDR." >&2
    exit 1
fi
for required_path in "${CONFIG}" "${MODEL_PATH}" "${DATA_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
COMPILE_CACHE_ROOT=${COMPILE_CACHE_ROOT:-${RUN_DIR}/compile_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${COMPILE_CACHE_ROOT}/triton/node_rank_${NODE_RANK}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${COMPILE_CACHE_ROOT}/torchinductor/node_rank_${NODE_RANK}}
export TORCHINDUCTOR_FX_GRAPH_CACHE=${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}
export DEEPSPEC_OUTPUT_ROOT=${DEEPSPEC_OUTPUT_ROOT:-${RUN_DIR}/home}
export PYTHONPATH=${ROOT_DIR}:${PYTHONPATH:-}
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=$(readlink -f "${ROOT_DIR}")

mkdir -p \
    "${RUN_DIR}/logs" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}"
cd "${ROOT_DIR}"

COMMON_OPTS=(
    --opts "model.target_model_name_or_path=${MODEL_PATH}"
    --opts "model.num_anchors=${NUM_ANCHORS}"
    --opts "data.max_length=${MAX_LENGTH}"
    --opts "data.multimodal=true"
)
MEDIA_ARGS=()
if [[ -n "${MEDIA_ROOT}" ]]; then
    MEDIA_ARGS+=(--media-root "${MEDIA_ROOT}")
fi
if [[ -n "${MEDIA_URI_MAP}" ]]; then
    MEDIA_ARGS+=(--media-uri-map "${MEDIA_URI_MAP}")
fi

if (( NODE_RANK == 0 )); then
    echo "Qwen3.6 layout: ${NUM_NODES} nodes x ${NUM_GPUS} GPUs = ${TOTAL_GPUS}"
    echo "CP=${CP_SIZE}, FSDP=${FSDP_SIZE}, effective data replicas=${EFFECTIVE_DATA_REPLICAS}"
fi

if [[ ! -f "${CACHE_DIR}/manifest.json" ]]; then
    if [[ -d "${CACHE_DIR}" ]] && [[ -n "$(ls -A "${CACHE_DIR}")" ]]; then
        echo "Incomplete non-empty cache directory: ${CACHE_DIR}" >&2
        exit 1
    fi
    echo "[1/2] Preparing Qwen3.6 multimodal target cache"
    MASTER_PORT=29600 "${PYTHON}" scripts/data/prepare_target_cache.py \
        --config "${CONFIG}" \
        "${COMMON_OPTS[@]}" \
        --train-data-path "${DATA_PATH}" \
        --output-dir "${CACHE_DIR}" \
        --min-loss-tokens "${MIN_LOSS_TOKENS}" \
        --local-batch-size 1 \
        --num-workers 0 \
        --fsdp \
        --fsdp-size "${FSDP_SIZE}" \
        --context-parallel-size "${CP_SIZE}" \
        "${MEDIA_ARGS[@]}" \
        2>&1 | tee -a "${RUN_DIR}/logs/cache.node${NODE_RANK}.log"
else
    echo "[1/2] Reusing complete target cache: ${CACHE_DIR}"
fi

TRAIN_STEP_OPTS=()
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    TRAIN_STEP_OPTS+=(--opts "train.max_train_steps=${MAX_TRAIN_STEPS}")
fi

echo "[2/2] Training Qwen3.6 DSpark"
MASTER_PORT=29610 "${PYTHON}" train.py \
    --config "${CONFIG}" \
    "${COMMON_OPTS[@]}" \
    --opts "data.target_cache_path=${CACHE_DIR}" \
    --opts "data.num_workers=1" \
    --opts "data.prefetch_factor=1" \
    --opts "train.sharding_strategy=full_shard" \
    --opts "train.fsdp_layerwise=true" \
    --opts "train.fsdp_size=${FSDP_SIZE}" \
    --opts "train.context_parallel_size=${CP_SIZE}" \
    --opts "train.expert_parallel_size=1" \
    --opts "train.tensor_parallel_size=1" \
    --opts "train.local_batch_size=1" \
    --opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --opts "train.num_train_epochs=${NUM_TRAIN_EPOCHS}" \
    --opts "train.torch_compile=false" \
    --opts "logging.logging_steps=1" \
    --opts "logging.checkpointing_steps=${CHECKPOINTING_STEPS}" \
    "${TRAIN_STEP_OPTS[@]}" \
    2>&1 | tee -a "${RUN_DIR}/logs/train.node${NODE_RANK}.log"

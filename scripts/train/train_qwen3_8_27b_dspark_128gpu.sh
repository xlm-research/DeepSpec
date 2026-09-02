#!/usr/bin/env bash
set -euo pipefail

# Production launcher for Qwen3.8-27B DSpark on homogeneous multi-GPU nodes.
# Run this script once on every node with the same rendezvous address and a
# shared filesystem. Bounded offline mode generates, trains, and deletes one
# transient target-feature partition at a time. Set BOUNDED_OFFLINE=false only
# to reuse the legacy full-cache workflow.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BASE_DIR}"

# No scheduler topology means a direct single-node launch. SenseCore reports
# node topology explicitly; generic launchers may report WORLD_SIZE/RANK in
# either nodes or GPU processes.
SCHEDULER_WORLD_SIZE=${WORLD_SIZE:-}
SCHEDULER_LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-}
SCHEDULER_RANK=${RANK:-}
NNODES=${SENSECORE_PYTORCH_NNODES:-${NNODES:-}}
NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-}}
REQUESTED_NPROC_PER_NODE=${NPROC_PER_NODE:-}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29501}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ -n "${LOCAL_RANK:-}" ]]; then
    echo "Run this launcher once per node, not from inside an existing torchrun worker." >&2
    exit 1
fi
command -v "${PYTHON_BIN}" >/dev/null || {
    echo "PYTHON_BIN is not available on PATH: ${PYTHON_BIN}" >&2
    exit 1
}
if ! DETECTED_NPROC_PER_NODE=$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())'); then
    echo "Unable to detect visible GPUs with torch.cuda.device_count()." >&2
    exit 1
fi
if [[ ! "${DETECTED_NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "No CUDA GPU is visible to PyTorch; detected ${DETECTED_NPROC_PER_NODE}." >&2
    exit 1
fi
if [[ -n "${REQUESTED_NPROC_PER_NODE}" ]]; then
    if [[ ! "${REQUESTED_NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NPROC_PER_NODE must be a positive integer when supplied; got ${REQUESTED_NPROC_PER_NODE}." >&2
        exit 1
    fi
    if ((REQUESTED_NPROC_PER_NODE != DETECTED_NPROC_PER_NODE)); then
        echo "NPROC_PER_NODE=${REQUESTED_NPROC_PER_NODE} does not match the ${DETECTED_NPROC_PER_NODE} GPUs visible to PyTorch." >&2
        exit 1
    fi
fi
NPROC_PER_NODE=${DETECTED_NPROC_PER_NODE}

WORLD_SIZE_IS_GPU_PROCESSES=false
if [[ -z "${NNODES}" ]]; then
    if [[ -n "${SCHEDULER_WORLD_SIZE}" ]]; then
        NNODES=${SCHEDULER_WORLD_SIZE}
        if [[ -n "${SCHEDULER_LOCAL_WORLD_SIZE}" ]]; then
            if [[ ! "${SCHEDULER_WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${SCHEDULER_LOCAL_WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
                echo "WORLD_SIZE and LOCAL_WORLD_SIZE must be positive integers." >&2
                exit 1
            fi
            if ((SCHEDULER_LOCAL_WORLD_SIZE != NPROC_PER_NODE)); then
                echo "LOCAL_WORLD_SIZE=${SCHEDULER_LOCAL_WORLD_SIZE} does not match the ${NPROC_PER_NODE} GPUs visible to PyTorch." >&2
                exit 1
            fi
            if ((SCHEDULER_WORLD_SIZE % SCHEDULER_LOCAL_WORLD_SIZE != 0)); then
                echo "WORLD_SIZE=${SCHEDULER_WORLD_SIZE} must be divisible by LOCAL_WORLD_SIZE=${SCHEDULER_LOCAL_WORLD_SIZE}." >&2
                exit 1
            fi
            NNODES=$((SCHEDULER_WORLD_SIZE / SCHEDULER_LOCAL_WORLD_SIZE))
            WORLD_SIZE_IS_GPU_PROCESSES=true
        fi
    else
        NNODES=1
    fi
fi
if [[ ! "${NNODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NNODES must be a positive integer; got ${NNODES}." >&2
    exit 1
fi
if [[ -z "${NODE_RANK}" && -n "${SCHEDULER_RANK}" ]]; then
    NODE_RANK=${SCHEDULER_RANK}
    if [[ "${WORLD_SIZE_IS_GPU_PROCESSES}" == "true" ]]; then
        if [[ ! "${SCHEDULER_RANK}" =~ ^(0|[1-9][0-9]*)$ ]]; then
            echo "RANK must be a canonical non-negative integer; got ${SCHEDULER_RANK}." >&2
            exit 1
        fi
        NODE_RANK=$((SCHEDULER_RANK / SCHEDULER_LOCAL_WORLD_SIZE))
    fi
fi
if [[ -z "${NODE_RANK}" ]]; then
    if ((NNODES == 1)); then
        NODE_RANK=0
    else
        echo "NODE_RANK (or SENSECORE_PYTORCH_NODE_RANK/RANK) is required." >&2
        exit 1
    fi
fi

for topology_var in NODE_RANK NPROC_PER_NODE MASTER_PORT; do
    topology_value=${!topology_var}
    if [[ ! "${topology_value}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "${topology_var} must be a canonical non-negative integer; got ${topology_value}." >&2
        exit 1
    fi
done
if ((NODE_RANK >= NNODES)); then
    echo "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}." >&2
    exit 1
fi
if ((MASTER_PORT < 1 || MASTER_PORT > 65535)); then
    echo "MASTER_PORT must be in [1, 65535]; got ${MASTER_PORT}." >&2
    exit 1
fi
if [[ -z "${MASTER_ADDR}" ]]; then
    if ((NNODES == 1)); then
        MASTER_ADDR=127.0.0.1
    else
        echo "A cross-node reachable MASTER_ADDR is required for NNODES=${NNODES}." >&2
        exit 1
    fi
fi
if ((NNODES > 1)) && [[ "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "::1" || "${MASTER_ADDR}" == "[::1]" || "${MASTER_ADDR}" == "0.0.0.0" || "${MASTER_ADDR}" == "::" ]]; then
    echo "A cross-node reachable MASTER_ADDR is required for NNODES=${NNODES}." >&2
    exit 1
fi

MAX_LENGTH=${MAX_LENGTH:-131072}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-${CP:-1}}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${TP:-4}}
if [[ ! "${CONTEXT_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONTEXT_PARALLEL_SIZE must be a positive integer; got ${CONTEXT_PARALLEL_SIZE}." >&2
    exit 1
fi
if [[ ! "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TENSOR_PARALLEL_SIZE must be a positive integer; got ${TENSOR_PARALLEL_SIZE}." >&2
    exit 1
fi
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B}
SOURCE_JSONL_PATH=${SOURCE_JSONL_PATH:-${BASE_DIR}/train_data/spec_o3_coldstartsft.repeat60.deepspec.jsonl}
TARGET_CACHE_PATH=${TARGET_CACHE_PATH:-${BASE_DIR}/output/dspark_qwen3_8_27b_target_cache/cp${CONTEXT_PARALLEL_SIZE}_maxlen${MAX_LENGTH}}
DEFAULT_OUTPUT_ROOT=${BASE_DIR}/output/dspark_qwen3_8_27b_multinode_production
if ((CONTEXT_PARALLEL_SIZE > 1 || TENSOR_PARALLEL_SIZE > 1)); then
    DEFAULT_OUTPUT_ROOT=${BASE_DIR}/output/dspark_qwen3_8_27b_cp${CONTEXT_PARALLEL_SIZE}_tp${TENSOR_PARALLEL_SIZE}_maxlen${MAX_LENGTH}
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}

LEARNING_RATE=${LEARNING_RATE:-0.0006}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-10}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}
BOUNDED_OFFLINE=${BOUNDED_OFFLINE:-true}
DATA_PARTITIONS=${DATA_PARTITIONS:-512}
JSONL_INDEX_CACHE_DIR=${JSONL_INDEX_CACHE_DIR:-${OUTPUT_ROOT}/jsonl_index_cache}
if ((NPROC_PER_NODE % (CONTEXT_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE) != 0)); then
    echo "Visible GPUs per node ${NPROC_PER_NODE} must be divisible by CONTEXT_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE=$((CONTEXT_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE))." >&2
    exit 1
fi
if ((512 % CONTEXT_PARALLEL_SIZE != 0)); then
    echo "Qwen3.8 DSpark model.num_anchors=512 must be divisible by CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE}." >&2
    exit 1
fi
FSDP_SIZE=${FSDP_SIZE:-$((NPROC_PER_NODE / (CONTEXT_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE)))}
TARGET_CACHE_FSDP_SIZE=${TARGET_CACHE_FSDP_SIZE:-$((NPROC_PER_NODE / CONTEXT_PARALLEL_SIZE))}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-3000}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-true}
DEFAULT_TORCH_COMPILE=false
DEFAULT_TARGET_CACHE_FSDP=false
if ((CONTEXT_PARALLEL_SIZE > 1)); then
    DEFAULT_TORCH_COMPILE=false
    DEFAULT_TARGET_CACHE_FSDP=true
fi
TORCH_COMPILE=${TORCH_COMPILE:-${DEFAULT_TORCH_COMPILE}}
TORCHRUN_PER_RANK_LOGS=${TORCHRUN_PER_RANK_LOGS:-true}
AUTO_PREPARE_CACHE=${AUTO_PREPARE_CACHE:-true}
TARGET_CACHE_FSDP=${TARGET_CACHE_FSDP:-${DEFAULT_TARGET_CACHE_FSDP}}
PREPARE_NUM_WORKERS=${PREPARE_NUM_WORKERS:-1}
DRY_RUN=${DRY_RUN:-false}
PRODUCTION_RUN=${PRODUCTION_RUN:-true}

for integer_var in MAX_LENGTH CONTEXT_PARALLEL_SIZE TENSOR_PARALLEL_SIZE LOCAL_BATCH_SIZE GLOBAL_BATCH_SIZE NUM_TRAIN_EPOCHS FSDP_SIZE TARGET_CACHE_FSDP_SIZE LOGGING_STEPS SAVE_STEPS PREPARE_NUM_WORKERS DATA_PARTITIONS; do
    integer_value=${!integer_var}
    if [[ ! "${integer_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${integer_var} must be a positive integer; got ${integer_value}." >&2
        exit 1
    fi
done
if [[ -n "${MAX_TRAIN_STEPS}" ]] && [[ ! "${MAX_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TRAIN_STEPS must be empty or a positive integer; got ${MAX_TRAIN_STEPS}." >&2
    exit 1
fi
for boolean_var in BOUNDED_OFFLINE SAVE_CHECKPOINTS TORCH_COMPILE TORCHRUN_PER_RANK_LOGS AUTO_PREPARE_CACHE TARGET_CACHE_FSDP DRY_RUN PRODUCTION_RUN; do
    boolean_value=${!boolean_var}
    if [[ "${boolean_value}" != "true" && "${boolean_value}" != "false" ]]; then
        echo "${boolean_var} must be true or false; got ${boolean_value}." >&2
        exit 1
    fi
done
if ((CONTEXT_PARALLEL_SIZE > 1)); then
    if ((LOCAL_BATCH_SIZE != 1)); then
        echo "CONTEXT_PARALLEL_SIZE > 1 requires LOCAL_BATCH_SIZE=1." >&2
        exit 1
    fi
    if ((MAX_LENGTH % (2 * CONTEXT_PARALLEL_SIZE) != 0)); then
        echo "MAX_LENGTH=${MAX_LENGTH} must be divisible by 2 * CONTEXT_PARALLEL_SIZE=$((2 * CONTEXT_PARALLEL_SIZE))." >&2
        exit 1
    fi
    if [[ "${TORCH_COMPILE}" != "false" ]]; then
        echo "CONTEXT_PARALLEL_SIZE > 1 requires TORCH_COMPILE=false." >&2
        exit 1
    fi
    if [[ "${BOUNDED_OFFLINE}" == "false" && "${TARGET_CACHE_FSDP}" != "true" ]]; then
        echo "Full-cache CONTEXT_PARALLEL_SIZE > 1 requires TARGET_CACHE_FSDP=true." >&2
        exit 1
    fi
fi
if [[ "${PRODUCTION_RUN}" == "true" ]]; then
    if [[ "${DRY_RUN}" != "false" ]]; then
        echo "PRODUCTION_RUN=true requires DRY_RUN=false." >&2
        exit 1
    fi
    if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
        echo "PRODUCTION_RUN=true requires MAX_TRAIN_STEPS to be unset." >&2
        exit 1
    fi
    if [[ "${SAVE_CHECKPOINTS}" != "true" ]]; then
        echo "PRODUCTION_RUN=true requires SAVE_CHECKPOINTS=true." >&2
        exit 1
    fi
fi

TRAIN_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
MODEL_PARALLEL_SIZE=$((FSDP_SIZE * CONTEXT_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE))
if ((TRAIN_WORLD_SIZE % MODEL_PARALLEL_SIZE != 0)); then
    echo "World size ${TRAIN_WORLD_SIZE} must be divisible by FSDP_SIZE * CONTEXT_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE=${MODEL_PARALLEL_SIZE}." >&2
    exit 1
fi
CACHE_MODEL_PARALLEL_SIZE=$((TARGET_CACHE_FSDP_SIZE * CONTEXT_PARALLEL_SIZE))
if ((TRAIN_WORLD_SIZE % CACHE_MODEL_PARALLEL_SIZE != 0)); then
    echo "World size ${TRAIN_WORLD_SIZE} must be divisible by TARGET_CACHE_FSDP_SIZE * CONTEXT_PARALLEL_SIZE=${CACHE_MODEL_PARALLEL_SIZE}." >&2
    exit 1
fi
DP_REPLICATE=$((TRAIN_WORLD_SIZE / MODEL_PARALLEL_SIZE))
DATA_PARALLEL_SIZE=$((TRAIN_WORLD_SIZE / (CONTEXT_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE)))
EFFECTIVE_FSDP_SHARD_SIZE=$((FSDP_SIZE * CONTEXT_PARALLEL_SIZE))
MICRO_GLOBAL_BATCH_SIZE=$((DATA_PARALLEL_SIZE * LOCAL_BATCH_SIZE))
if ((GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH_SIZE != 0)); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by data parallel size * LOCAL_BATCH_SIZE=${MICRO_GLOBAL_BATCH_SIZE}." >&2
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH_SIZE))

if [[ -z "${DATA_BATCH_CACHE_DIR:-}" ]]; then
    DATA_BATCH_CACHE_TIMESTAMP=${DATA_BATCH_CACHE_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}
    if [[ ! "${DATA_BATCH_CACHE_TIMESTAMP}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
        echo "DATA_BATCH_CACHE_TIMESTAMP must use YYYYMMDD_HHMMSS; got ${DATA_BATCH_CACHE_TIMESTAMP}." >&2
        exit 1
    fi
    TARGET_MODEL_NAME=$(basename "${TARGET_MODEL_PATH%/}")
    DATA_BATCH_CACHE_DIR=${OUTPUT_ROOT}/${TARGET_MODEL_NAME}_dp${DP_REPLICATE}_fsdp${FSDP_SIZE}_cp${CONTEXT_PARALLEL_SIZE}_tp${TENSOR_PARALLEL_SIZE}_${DATA_BATCH_CACHE_TIMESTAMP}
fi

if [[ ! -d "${TARGET_MODEL_PATH}" ]]; then
    echo "TARGET_MODEL_PATH does not exist: ${TARGET_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${TARGET_MODEL_PATH}/config.json" || ! -f "${TARGET_MODEL_PATH}/model.safetensors.index.json" ]]; then
    echo "TARGET_MODEL_PATH is not a complete sharded Qwen3.8 checkpoint: ${TARGET_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${SOURCE_JSONL_PATH}" ]]; then
    echo "SOURCE_JSONL_PATH does not exist: ${SOURCE_JSONL_PATH}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
NODE_LOG=${LOG_DIR}/node_rank_${NODE_RANK}.log
TORCHRUN_LOG_DIR=${LOG_DIR}/torchrun_node_rank_${NODE_RANK}
mkdir -p "${TORCHRUN_LOG_DIR}"
exec > >(tee -a "${NODE_LOG}") 2>&1

START_TIME=$(date '+%Y-%m-%d %H:%M:%S %z')
diagnose_exit() {
    status=$?
    set +e
    echo "[deepspec-launch-exit] time=$(date '+%Y-%m-%d %H:%M:%S %z') host=$(hostname) node_rank=${NODE_RANK} exit_code=${status}"
    if ((status == 0)) && [[ "${DRY_RUN}" == "true" ]]; then
        echo "[deepspec-launch-diagnosis] dry run completed; training was not started"
    elif ((status != 0)); then
        echo "[deepspec-launch-diagnosis] inspect ${NODE_LOG} and ${TORCHRUN_LOG_DIR} for the first failing rank"
    fi
    trap - EXIT
    exit "${status}"
}
trap diagnose_exit EXIT

PREPARE_FSDP_ARGS=()
if [[ "${TARGET_CACHE_FSDP}" == "true" ]]; then
    PREPARE_FSDP_ARGS=(--fsdp)
fi
PREPARE_CACHE_COMMAND=(
    "${PYTHON_BIN}"
    scripts/data/prepare_target_cache.py
    --config config/dspark/dspark_qwen3_8_27b.py
    --train-data-path "${SOURCE_JSONL_PATH}"
    --output-dir "${TARGET_CACHE_PATH}"
    --local-batch-size 1
    --num-workers "${PREPARE_NUM_WORKERS}"
    "${PREPARE_FSDP_ARGS[@]}"
    --fsdp-size "${TARGET_CACHE_FSDP_SIZE}"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --target-micro-chunk-size 0
    --target-cache-cpu-offload false
    --opts "model.target_model_name_or_path=${TARGET_MODEL_PATH}"
    --opts "data.max_length=${MAX_LENGTH}"
    --opts "data.multimodal=false"
    --opts "data.store_target_last_hidden_states=true"
)
PREPARE_CACHE_ENV=(
    env
    "WORLD_SIZE=${NNODES}"
    "RANK=${NODE_RANK}"
    "MASTER_ADDR=${MASTER_ADDR}"
    "MASTER_PORT=${MASTER_PORT}"
)

if [[ "${BOUNDED_OFFLINE}" == "false" ]]; then
    if [[ ! -f "${TARGET_CACHE_PATH}/manifest.json" ]]; then
        if [[ -d "${TARGET_CACHE_PATH}" ]] && [[ -n "$(find "${TARGET_CACHE_PATH}" -mindepth 1 -print -quit)" ]]; then
            echo "TARGET_CACHE_PATH contains an incomplete cache without manifest.json: ${TARGET_CACHE_PATH}" >&2
            echo "Use a new TARGET_CACHE_PATH or explicitly remove the partial cache after confirming it is disposable." >&2
            exit 1
        fi
        if [[ "${AUTO_PREPARE_CACHE}" != "true" ]]; then
            echo "Target cache is missing and AUTO_PREPARE_CACHE=false: ${TARGET_CACHE_PATH}" >&2
            exit 1
        fi
        echo "Qwen3.8 DSpark target cache is missing; preparing it before training."
        echo "  Cache storage scales with the number of valid tokens; verify shared-disk capacity before a full run."
        echo "  cache output=${TARGET_CACHE_PATH}"
        echo "  target-cache FSDP=${TARGET_CACHE_FSDP}, FSDP_SIZE=${TARGET_CACHE_FSDP_SIZE}, CP=${CONTEXT_PARALLEL_SIZE}"
        if [[ "${DRY_RUN}" == "true" ]]; then
            printf '  prepare command:'
            printf ' %q' "${PREPARE_CACHE_ENV[@]}" "${PREPARE_CACHE_COMMAND[@]}"
            printf '\n'
        else
            mkdir -p "$(dirname "${TARGET_CACHE_PATH}")"
            "${PREPARE_CACHE_ENV[@]}" "${PREPARE_CACHE_COMMAND[@]}"
            if [[ ! -f "${TARGET_CACHE_PATH}/manifest.json" ]]; then
                echo "Target-cache preparation finished without manifest.json: ${TARGET_CACHE_PATH}" >&2
                exit 1
            fi
        fi
    else
        echo "Using completed Qwen3.8 DSpark target cache: ${TARGET_CACHE_PATH}"
    fi
else
    echo "Using bounded offline Qwen3.8 target partitions; no full cache is required."
fi

TRAIN_SCHEDULE_ARGS=(
    --opts "train.num_train_epochs=${NUM_TRAIN_EPOCHS}"
    --opts "train.max_train_steps=null"
)
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    TRAIN_SCHEDULE_ARGS=(--opts "train.max_train_steps=${MAX_TRAIN_STEPS}")
fi

TARGET_DATA_ARGS=(
    --opts "data.online_target=false"
    --opts "data.offline_target_data_batches=true"
    --opts "data.train_data_path=${SOURCE_JSONL_PATH}"
    --opts "data.jsonl_index_cache_dir=${JSONL_INDEX_CACHE_DIR}"
    --opts "data.data_batch_cache_dir=${DATA_BATCH_CACHE_DIR}"
    --opts "data.target_cache_path=null"
    --opts "train.data_partitions=${DATA_PARTITIONS}"
    --opts "train.offline_target_parallel.dp_replicate=${DP_REPLICATE}"
    --opts "train.offline_target_parallel.dp_shard=${FSDP_SIZE}"
    --opts "train.offline_target_parallel.cp=${CONTEXT_PARALLEL_SIZE}"
    --opts "train.offline_target_parallel.tp=${TENSOR_PARALLEL_SIZE}"
    --opts "train.offline_target_parallel.ep=1"
    --opts "train.offline_target_parallel.expert_tp=1"
    --opts "train.offline_target_parallel.use_fsdp=true"
    --opts "train.offline_target_parallel.context_parallel_backend=model_native"
)
if [[ "${BOUNDED_OFFLINE}" == "false" ]]; then
    TARGET_DATA_ARGS=(
        --opts "data.online_target=false"
        --opts "data.offline_target_data_batches=false"
        --opts "data.source_jsonl_path=${SOURCE_JSONL_PATH}"
        --opts "data.target_cache_path=${TARGET_CACHE_PATH}"
        --opts "train.data_partitions=null"
    )
fi

echo "Launching Qwen3.8-27B DSpark FSDP2 on ${TRAIN_WORLD_SIZE} GPUs:"
echo "  start=${START_TIME}, host=$(hostname), pid=$$"
echo "  node=${NODE_RANK}/${NNODES}, rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
echo "  visible GPUs per node=${NPROC_PER_NODE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  homogeneous-node requirement=every node must detect ${NPROC_PER_NODE} visible GPUs"
echo "  topology=DP_REPLICATE=${DP_REPLICATE}, DP_SHARD=${FSDP_SIZE}, CP=${CONTEXT_PARALLEL_SIZE}, TP=${TENSOR_PARALLEL_SIZE}, effective FSDP shard=${EFFECTIVE_FSDP_SHARD_SIZE}"
echo "  data parallel size=${DATA_PARALLEL_SIZE}"
echo "  local batch=${LOCAL_BATCH_SIZE}, global batch=${GLOBAL_BATCH_SIZE}, gradient accumulation=${GRADIENT_ACCUMULATION_STEPS}"
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    echo "  schedule=diagnostic override, max steps=${MAX_TRAIN_STEPS}"
else
    echo "  schedule=dataset-derived, epochs=${NUM_TRAIN_EPOCHS}"
fi
echo "  target model=${TARGET_MODEL_PATH}"
if [[ "${BOUNDED_OFFLINE}" == "true" ]]; then
    echo "  target supervision=bounded offline, data partitions per epoch=${DATA_PARTITIONS}"
    echo "  transient target cache=${DATA_BATCH_CACHE_DIR}"
    echo "  JSONL index cache=${JSONL_INDEX_CACHE_DIR}"
    echo "  target FSDP shard includes draft TP ranks=${TENSOR_PARALLEL_SIZE}"
else
    echo "  target supervision=full offline cache, target cache=${TARGET_CACHE_PATH}"
fi
echo "  source JSONL=${SOURCE_JSONL_PATH:-<not supplied>}"
echo "  checkpoint dir=${CHECKPOINT_DIR}"
echo "  tensorboard dir=${TENSORBOARD_DIR}"
echo "  torch.compile=${TORCH_COMPILE}, save checkpoints=${SAVE_CHECKPOINTS}"
echo "  production safety=${PRODUCTION_RUN}, dry run=${DRY_RUN}"
echo "  node log=${NODE_LOG}"
echo "  launcher=${PYTHON_BIN} -m torch.distributed.run"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE:-20000}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}

LAUNCHER=("${PYTHON_BIN}" -m torch.distributed.run)
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  WARNING: DRY_RUN=true; only the torchrun command will be printed"
    LAUNCHER=(echo "${PYTHON_BIN}" -m torch.distributed.run)
fi

TORCHRUN_LOG_ARGS=()
if [[ "${TORCHRUN_PER_RANK_LOGS}" == "true" && "${DRY_RUN}" != "true" ]]; then
    TORCHRUN_LOG_ARGS=(
        --log-dir "${TORCHRUN_LOG_DIR}"
        --redirects 3
        --tee 3
    )
fi

set -x
"${LAUNCHER[@]}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TORCHRUN_LOG_ARGS[@]}" \
    train.py \
    --config config/dspark/dspark_qwen3_8_27b.py \
    --opts "model.target_model_name_or_path=${TARGET_MODEL_PATH}" \
    "${TARGET_DATA_ARGS[@]}" \
    --opts "data.max_length=${MAX_LENGTH}" \
    --opts "data.store_target_last_hidden_states=true" \
    --opts "train.lr=${LEARNING_RATE}" \
    --opts "train.local_batch_size=${LOCAL_BATCH_SIZE}" \
    --opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --opts "train.parallel.dp_replicate=${DP_REPLICATE}" \
    --opts "train.parallel.dp_shard=${FSDP_SIZE}" \
    --opts "train.parallel.cp=${CONTEXT_PARALLEL_SIZE}" \
    --opts "train.parallel.tp=${TENSOR_PARALLEL_SIZE}" \
    --opts "train.parallel.use_compile=${TORCH_COMPILE}" \
    --opts "train.torch_compile=${TORCH_COMPILE}" \
    "${TRAIN_SCHEDULE_ARGS[@]}" \
    --opts "logging.logging_steps=${LOGGING_STEPS}" \
    --opts "logging.checkpointing_steps=${SAVE_STEPS}" \
    --opts "logging.save_checkpoints=${SAVE_CHECKPOINTS}" \
    --opts "logging.checkpoint_dir=${CHECKPOINT_DIR}" \
    --opts "logging.tensorboard_dir=${TENSORBOARD_DIR}"

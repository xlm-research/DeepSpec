#!/usr/bin/env bash
set -eo pipefail

env | grep -E '^(LOCAL_RANK|RANK|WORLD_SIZE|LOCAL_WORLD_SIZE|NPROC_PER_NODE|MASTER_ADDR|MASTER_PORT|SENSECORE_PYTORCH_NNODES|SENSECORE_PYTORCH_NODE_RANK)=' || true

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
cd "${repo_root}"

# SenseCore launches this same script on every node and injects these values.
NNODES=${SENSECORE_PYTORCH_NNODES:-${NNODES:-1}}
NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29501}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NUM_GPUS=$((NNODES * NPROC_PER_NODE))

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-${repo_root}/train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-${repo_root}/output/deepseek_v4_flash_dspark_fsdp2_multinode}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}

MAX_LENGTH=${MAX_LENGTH:-131072}
NUM_ANCHORS=${NUM_ANCHORS:-512}
LEARNING_RATE=${LEARNING_RATE:-0.00001}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-5}
DRY_RUN=${DRY_RUN:-false}

# Keep one copy of the validated 8-GPU topology per node. With contiguous
# global ranks, CP/FSDP and target EP8 remain node-local; dp_replicate scales
# training across nodes.
DP_REPLICATE=${DP_REPLICATE:-${NNODES}}
DP_SHARD=${DP_SHARD:-4}
CP=${CP:-2}
TP=${TP:-1}
DRAFT_EP=${DRAFT_EP:-1}
TARGET_EP=${TARGET_EP:-8}

DENSE_PARALLEL_SIZE=$((DP_REPLICATE * DP_SHARD * CP * TP))
SPARSE_DOMAIN=$((DP_SHARD * CP * TP))
DATA_PARALLEL_SIZE=$((DP_REPLICATE * DP_SHARD))
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((DATA_PARALLEL_SIZE * LOCAL_BATCH_SIZE))}

if ((NUM_GPUS != DENSE_PARALLEL_SIZE)); then
    echo "NUM_GPUS=${NUM_GPUS} must equal DP_REPLICATE*DP_SHARD*CP*TP=${DENSE_PARALLEL_SIZE}." >&2
    exit 1
fi
if ((SPARSE_DOMAIN % TARGET_EP != 0)); then
    echo "DP_SHARD*CP*TP=${SPARSE_DOMAIN} must be divisible by TARGET_EP=${TARGET_EP}." >&2
    exit 1
fi
if ((GLOBAL_BATCH_SIZE % (DATA_PARALLEL_SIZE * LOCAL_BATCH_SIZE) != 0)); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by DATA_PARALLEL_SIZE*LOCAL_BATCH_SIZE=$((DATA_PARALLEL_SIZE * LOCAL_BATCH_SIZE))." >&2
    exit 1
fi
if ((NNODES > 1)) && [[ "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "127.0.0.1" ]]; then
    echo "SenseCore must provide a cross-node reachable MASTER_ADDR when NNODES=${NNODES}." >&2
    exit 1
fi
if [[ ! -d "${TARGET_MODEL_PATH}" ]]; then
    echo "TARGET_MODEL_PATH does not exist: ${TARGET_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
    echo "TRAIN_DATA_PATH does not exist: ${TRAIN_DATA_PATH}" >&2
    exit 1
fi
if ((MAX_LENGTH < 1 || MAX_LENGTH > 131072)); then
    echo "MAX_LENGTH must be in [1, 131072]; got ${MAX_LENGTH}." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Launching DeepSeek-V4-Flash DSpark FSDP2:"
echo "  node=${NODE_RANK}/${NNODES}, rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
echo "  GPUs=${NUM_GPUS}, GPUs/node=${NPROC_PER_NODE}"
echo "  DP_REPLICATE=${DP_REPLICATE}, DP_SHARD=${DP_SHARD}, CP=${CP}, TP=${TP}"
echo "  draft EP=${DRAFT_EP}, target EP=${TARGET_EP}, GBS=${GLOBAL_BATCH_SIZE}"

export CUDA_VISIBLE_DEVICES
export DEEPSPEC_OUTPUT_ROOT=${OUTPUT_ROOT}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

set -x
LAUNCHER=(torchrun)
if [[ "${DRY_RUN}" == "true" ]]; then
    LAUNCHER=(echo torchrun)
fi

"${LAUNCHER[@]}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --config config/dspark/dspark_deepseek_v4.py \
    --opts "model.target_model_name_or_path=${TARGET_MODEL_PATH}" \
    --opts "data.train_data_path=${TRAIN_DATA_PATH}" \
    --opts "data.max_length=${MAX_LENGTH}" \
    --opts "model.num_anchors=${NUM_ANCHORS}" \
    --opts "train.lr=${LEARNING_RATE}" \
    --opts "train.local_batch_size=${LOCAL_BATCH_SIZE}" \
    --opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --opts "train.parallel.dp_replicate=${DP_REPLICATE}" \
    --opts "train.parallel.dp_shard=${DP_SHARD}" \
    --opts "train.parallel.cp=${CP}" \
    --opts "train.parallel.tp=${TP}" \
    --opts "train.parallel.ep=${DRAFT_EP}" \
    --opts "train.target_parallel.ep=${TARGET_EP}" \
    --opts "train.max_train_steps=${MAX_TRAIN_STEPS}" \
    --opts "logging.logging_steps=10000" \
    2>&1 | tee "${LOG_DIR}/node_rank_${NODE_RANK}.log"

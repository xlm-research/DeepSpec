#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Flash 128K online DSpark distillation.
# Every micro-batch runs one frozen-target forward and immediately feeds the
# rank-local hidden-state shard to the draft model; no target cache is written.
# Run this command once on every node. RANK/WORLD_SIZE are node rank/count;
# train.py spawns one process per visible local GPU.

ROOT_DIR=${ROOT_DIR:-/mnt/afs_share/wzj/deepspec1}
PYTHON=${PYTHON:-/mnt/afs_share/wzj/DeepSpec/.envs/deepspec-qwen36/bin/python}
CONFIG=${CONFIG:-${ROOT_DIR}/config/dspark/dspark_deepseek_v4.py}
MODEL_PATH=${MODEL_PATH:-/mnt/afs_agents/share_models/deepseek-ai/DeepSeek-V4-Flash}
DATA_PATH=${DATA_PATH:-${ROOT_DIR}/train_data/deepseek_v4_128k.jsonl}

PROJECT_NAME=${PROJECT_NAME:-deepspec_128k}
EXP_NAME=${EXP_NAME:-dspark_deepseek_v4_flash_128k}
RUN_DIR=${RUN_DIR:-${ROOT_DIR}/output/deepseek_v4_128k/full_cp_ep_tp}

CP_SIZE=${CP_SIZE:-8}
TARGET_EP_SIZE=${TARGET_EP_SIZE:-2}
TARGET_TP_SIZE=${TARGET_TP_SIZE:-2}
TARGET_FSDP_SIZE=${TARGET_FSDP_SIZE:-9}
DRAFT_EP_SIZE=${DRAFT_EP_SIZE:-2}
DRAFT_TP_SIZE=${DRAFT_TP_SIZE:-2}
DRAFT_FSDP_SIZE=${DRAFT_FSDP_SIZE:-9}
NUM_ANCHORS=${NUM_ANCHORS:-512}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-504}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-}
MIN_LOSS_TOKENS=${MIN_LOSS_TOKENS:-14}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-50}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_NODES=${NUM_NODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export WORLD_SIZE=${NUM_NODES}
export RANK=${NODE_RANK}
unset LOCAL_RANK

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DEEPSPEC_V4_CP_QUERY_CHUNK=${DEEPSPEC_V4_CP_QUERY_CHUNK:-16}
export DEEPSPEC_V4_CP_INDEX_KEY_CHUNK=${DEEPSPEC_V4_CP_INDEX_KEY_CHUNK:-1024}
export DEEPSPEC_V4_EP_TOKEN_CHUNK=${DEEPSPEC_V4_EP_TOKEN_CHUNK:-4096}

COMPILE_CACHE_ROOT=${COMPILE_CACHE_ROOT:-/mnt/afs_rl/wzj/.cache/deepspec/deepseek_v4_cp${CP_SIZE}_ep${DRAFT_EP_SIZE}_tp${DRAFT_TP_SIZE}}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${COMPILE_CACHE_ROOT}/triton/node_rank_${NODE_RANK}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${COMPILE_CACHE_ROOT}/torchinductor/node_rank_${NODE_RANK}}
export TORCHINDUCTOR_FX_GRAPH_CACHE=${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_IDS[@]}
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))
TARGET_MODEL_PARALLEL_SIZE=$((CP_SIZE * TARGET_EP_SIZE * TARGET_TP_SIZE * TARGET_FSDP_SIZE))
DRAFT_MODEL_PARALLEL_SIZE=$((CP_SIZE * DRAFT_EP_SIZE * DRAFT_TP_SIZE * DRAFT_FSDP_SIZE))

if (( NUM_GPUS != 8 )); then
    echo "This launcher requires exactly 8 visible GPUs per node; got ${NUM_GPUS}." >&2
    exit 1
fi
if [[ -n "${EXPECTED_NUM_NODES}" ]] && (( NUM_NODES != EXPECTED_NUM_NODES )); then
    echo "Expected ${EXPECTED_NUM_NODES} nodes, got ${NUM_NODES}." >&2
    exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NUM_NODES )); then
    echo "Invalid NODE_RANK=${NODE_RANK} for NUM_NODES=${NUM_NODES}." >&2
    exit 1
fi
if (( TARGET_EP_SIZE != DRAFT_EP_SIZE || TARGET_TP_SIZE != DRAFT_TP_SIZE || TARGET_FSDP_SIZE != DRAFT_FSDP_SIZE )); then
    echo "Online rank-local handoff requires identical target/draft EP, TP, and FSDP sizes." >&2
    exit 1
fi
if (( NUM_NODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "Multi-node execution requires a rank-0 MASTER_ADDR reachable by every node." >&2
    exit 1
fi
if (( TOTAL_GPUS % TARGET_MODEL_PARALLEL_SIZE != 0 )); then
    echo "TOTAL_GPUS=${TOTAL_GPUS} must be divisible by target CP*EP*TP*FSDP=${TARGET_MODEL_PARALLEL_SIZE}." >&2
    exit 1
fi
if (( TOTAL_GPUS % DRAFT_MODEL_PARALLEL_SIZE != 0 )); then
    echo "TOTAL_GPUS=${TOTAL_GPUS} must be divisible by draft CP*EP*TP*FSDP=${DRAFT_MODEL_PARALLEL_SIZE}." >&2
    exit 1
fi
if (( NUM_ANCHORS < CP_SIZE || NUM_ANCHORS % CP_SIZE != 0 )); then
    echo "NUM_ANCHORS=${NUM_ANCHORS} must be a positive multiple of CP_SIZE=${CP_SIZE}." >&2
    exit 1
fi

if (( 256 % TARGET_EP_SIZE != 0 || 256 % DRAFT_EP_SIZE != 0 )); then
    echo "DeepSeek-V4 has 256 routed experts; TARGET_EP_SIZE and DRAFT_EP_SIZE must divide 256." >&2
    exit 1
fi
if (( 64 % TARGET_TP_SIZE != 0 || 64 % DRAFT_TP_SIZE != 0 || 8 % TARGET_TP_SIZE != 0 || 8 % DRAFT_TP_SIZE != 0 )); then
    echo "TP size must divide both 64 attention heads and 8 output groups." >&2
    exit 1
fi
if (( 2048 % TARGET_TP_SIZE != 0 || 2048 % DRAFT_TP_SIZE != 0 || 129280 % TARGET_TP_SIZE != 0 || 129280 % DRAFT_TP_SIZE != 0 )); then
    echo "TP size must divide the 2048 expert width and 129280 vocabulary." >&2
    exit 1
fi

EFFECTIVE_DATA_REPLICAS=$((TOTAL_GPUS / (CP_SIZE * DRAFT_EP_SIZE * DRAFT_TP_SIZE)))
TARGET_DP_SIZE=$((TOTAL_GPUS / TARGET_MODEL_PARALLEL_SIZE))
DRAFT_DP_SIZE=$((TOTAL_GPUS / DRAFT_MODEL_PARALLEL_SIZE))
if (( GLOBAL_BATCH_SIZE % EFFECTIVE_DATA_REPLICAS != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by effective data replicas=${EFFECTIVE_DATA_REPLICAS}." >&2
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / EFFECTIVE_DATA_REPLICAS))

for required_path in "${PYTHON}" "${CONFIG}" "${MODEL_PATH}" "${DATA_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

CUDA_COUNT=$("${PYTHON}" -c "import torch; print(torch.cuda.device_count())")
if (( CUDA_COUNT != NUM_GPUS )); then
    echo "Expected ${NUM_GPUS} CUDA devices, but PyTorch sees ${CUDA_COUNT}." >&2
    exit 1
fi

DATA_PATH="${DATA_PATH}" "${PYTHON}" - <<'PY'
import json
import os

path = os.environ["DATA_PATH"]
with open(path, "r", encoding="utf-8") as handle:
    record = next((json.loads(line) for line in handle if line.strip()), None)
if record is None:
    raise RuntimeError(f"Dataset is empty: {path}")
conversation = record.get("conversations")
if not isinstance(conversation, list) or not conversation:
    raise RuntimeError("Each JSONL record must contain a non-empty `conversations` list.")
if not any(item.get("role") == "assistant" for item in conversation):
    raise RuntimeError("Each training sample must contain at least one assistant turn.")
PY

mkdir -p \
    "${RUN_DIR}/home" \
    "${RUN_DIR}/logs" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}"

export DEEPSPEC_OUTPUT_ROOT=${DEEPSPEC_OUTPUT_ROOT:-${RUN_DIR}/home}
export PYTHONPATH=${ROOT_DIR}:${PYTHONPATH:-}
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=$(readlink -f "${ROOT_DIR}")
cd "${ROOT_DIR}"

COMMON_OPTS=(
    --opts "project_name=${PROJECT_NAME}"
    --opts "exp_name=${EXP_NAME}"
    --opts "model.target_model_name_or_path=${MODEL_PATH}"
    --opts "model.num_anchors=${NUM_ANCHORS}"
    --opts "data.max_length=131072"
)

TRAIN_STEP_OPTS=()
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    TRAIN_STEP_OPTS+=(--opts "train.max_train_steps=${MAX_TRAIN_STEPS}")
fi

if (( NODE_RANK == 0 )); then
    echo "Distributed layout: ${NUM_NODES} nodes x ${NUM_GPUS} GPUs = ${TOTAL_GPUS} GPUs"
    echo "Target: CP=${CP_SIZE}, EP=${TARGET_EP_SIZE}, TP=${TARGET_TP_SIZE}, FSDP=${TARGET_FSDP_SIZE}, DP=${TARGET_DP_SIZE}"
    echo "Draft:  CP=${CP_SIZE}, EP=${DRAFT_EP_SIZE}, TP=${DRAFT_TP_SIZE}, FSDP=${DRAFT_FSDP_SIZE}, DP=${DRAFT_DP_SIZE}"
    echo "EP All-to-All token chunk=${DEEPSPEC_V4_EP_TOKEN_CHUNK}"
    echo "Target supervision=online (no disk cache), epochs=${NUM_TRAIN_EPOCHS}"
    echo "Effective data replicas=${EFFECTIVE_DATA_REPLICAS}"
    echo "Global batch=${GLOBAL_BATCH_SIZE}, gradient accumulation=${GRADIENT_ACCUMULATION_STEPS}"
    echo "Dataset=${DATA_PATH}"
fi

echo "[1/1] Online DeepSeek-V4 target forward + DSpark training on ${TOTAL_GPUS} GPUs"
MASTER_PORT=29610 "${PYTHON}" train.py \
    --config "${CONFIG}" \
    "${COMMON_OPTS[@]}" \
    --opts "data.online_target=true" \
    --opts "data.train_data_path=${DATA_PATH}" \
    --opts "data.target_cache_path=null" \
    --opts "data.min_loss_tokens=${MIN_LOSS_TOKENS}" \
    --opts "data.num_workers=1" \
    --opts "train.sharding_strategy=full_shard" \
    --opts "train.fsdp_layerwise=true" \
    --opts "train.fsdp_size=${DRAFT_FSDP_SIZE}" \
    --opts "train.context_parallel_size=${CP_SIZE}" \
    --opts "train.expert_parallel_size=${DRAFT_EP_SIZE}" \
    --opts "train.tensor_parallel_size=${DRAFT_TP_SIZE}" \
    --opts "train.local_batch_size=1" \
    --opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --opts "train.num_train_epochs=${NUM_TRAIN_EPOCHS}" \
    --opts "train.torch_compile=false" \
    --opts "logging.logging_steps=1" \
    --opts "logging.checkpointing_steps=${CHECKPOINTING_STEPS}" \
    "${TRAIN_STEP_OPTS[@]}" \
    2>&1 | tee -a "${RUN_DIR}/logs/online_train.node${NODE_RANK}.log"

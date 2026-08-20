#!/usr/bin/env bash
# Some schedulers invoke scripts as `sh script.sh`, which ignores the shebang.
# Keep this guard POSIX-sh-compatible so it can re-exec bash before any bashisms
# such as arrays, [[ ... ]], or (( ... )) are parsed/executed.
if [ -z "${BASH_VERSION:-}" ]; then
    exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

# DeepSeek-V4-Flash 128K online DSpark distillation.
#
# Run this launcher once on every node. RANK/WORLD_SIZE are the node rank/count;
# DeepSpec's train.py spawns one worker per visible local GPU. The frozen target
# and trainable draft share one DP x CP x EP x TP x FSDP topology, so EP/TP are
# applied to both models and online hidden-state handoff remains rank-local.

# SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${ROOT_DIR:-/mnt/afs-agentpro/hongjiawei/code/DeepSpec}
ENV_SCRIPT=${ENV_SCRIPT:-/mnt/afs-agentpro/yangbo1/ms-swift/env.sh}

if [[ ! -f "${ENV_SCRIPT}" ]]; then
    echo "Environment script not found: ${ENV_SCRIPT}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${ENV_SCRIPT}"

PYTHON=${PYTHON:-python3}
CONFIG=${CONFIG:-${ROOT_DIR}/config/dspark/dspark_deepseek_v4.py}
MODEL_PATH=${MODEL_PATH:-/mnt/afs-agentpro/hongjiawei/code/ms-swift/megatron_output/sensenova-flash-284b-v1-1-fp8-20260807-step800-opd-fp8-step160-dpo-v1-DPO/TP2_PP4_VPP4_CP4_EP8_ETP1_GPUS128_GBS64_DPO_FULL43/v3-20260814-174015/checkpoint-148}

# DeepSpec consumes JSONL records with conversations(role/content). This
# validation launcher converts only a small prefix and reuses it on later runs.
# Set PREPARE_DATASET=false when DATA_PATH already points to compatible JSONL.
SOURCE_DATA_PATH=${SOURCE_DATA_PATH:-/mnt/afs-agentpro/yangbo1/Maxwell-Jia/Spec-o3-ColdStartSFT/train_sharegpt.json}
THINKING_TOOLCALL_DATA_PATH=${THINKING_TOOLCALL_DATA_PATH:-${ROOT_DIR}/scripts/data/sensenova_flash_v15_use_data_json.thinking_toolcall.readable_full.flash_conversations.jsonl}
DATA_REPEAT=${DATA_REPEAT:-1}
DATA_MAX_RECORDS=${DATA_MAX_RECORDS:-8}
DATA_PATH=${DATA_PATH:-${THINKING_TOOLCALL_DATA_PATH}}
PREPARE_DATASET=${PREPARE_DATASET:-false}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-deepseek_v4_flash_body}

PROJECT_NAME=${PROJECT_NAME:-deepspec_deepseek_v4_128k}
EXP_NAME=${EXP_NAME:-dspark_deepseek_v4_thinking_toolcall_online}
RUN_DIR=${RUN_DIR:-${ROOT_DIR}/output/deepseek_v4_dspark_thinking_toolcall}

# Single-node validation layout. Pure EP overlays the complete base topology
# and does not multiply world size.
CP_SIZE=${CP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-8}
EP_SIZE=${EP_SIZE:-8}
TP_SIZE=${TP_SIZE:-1}
PURE_EXPERT_PARALLEL=${PURE_EXPERT_PARALLEL:-true}

NUM_ANCHORS=${NUM_ANCHORS:-504}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-11142}
MAX_LENGTH=${MAX_LENGTH:-131072}
MIN_LOSS_TOKENS=${MIN_LOSS_TOKENS:-14}
LENGTH_BUCKET_SIZE=${LENGTH_BUCKET_SIZE:-512}
DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-1}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-1}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-1}
SHARDING_STRATEGY=${SHARDING_STRATEGY:-full_shard}
FSDP_LAYERWISE=${FSDP_LAYERWISE:-true}
TORCH_COMPILE=${TORCH_COMPILE:-false}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-}
DIST_TIMEOUT_MINUTES=${DIST_TIMEOUT_MINUTES:-${DEEPSPEC_DIST_TIMEOUT_MINUTES:-30}}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-2}}}
NODE_RANK=${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-0}}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29610}
export WORLD_SIZE=${NUM_NODES}
export RANK=${NODE_RANK}
unset LOCAL_RANK

die() {
    echo "$*" >&2
    exit 1
}

require_positive_int() {
    local name=$1
    local value=${!name}
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
        die "${name} must be a positive integer; got '${value}'."
}

require_nonnegative_int() {
    local name=$1
    local value=${!name}
    [[ "${value}" =~ ^[0-9]+$ ]] || \
        die "${name} must be a non-negative integer; got '${value}'."
}

MISSING_PARALLEL_SIZES=()
for parallel_name in CP_SIZE FSDP_SIZE EP_SIZE TP_SIZE; do
    if [[ -z "${!parallel_name}" ]]; then
        MISSING_PARALLEL_SIZES+=("${parallel_name}")
    fi
done
if (( ${#MISSING_PARALLEL_SIZES[@]} > 0 )); then
    die "Set the parallel sizes before launch: ${MISSING_PARALLEL_SIZES[*]}. Example: CP_SIZE=... FSDP_SIZE=... EP_SIZE=... TP_SIZE=... bash ${BASH_SOURCE[0]}"
fi

for positive_name in \
    CP_SIZE FSDP_SIZE EP_SIZE TP_SIZE NUM_ANCHORS GLOBAL_BATCH_SIZE \
    LOCAL_BATCH_SIZE NUM_TRAIN_EPOCHS MAX_LENGTH MIN_LOSS_TOKENS \
    DATA_NUM_WORKERS PREFETCH_FACTOR CHECKPOINTING_STEPS LOGGING_STEPS \
    DATA_REPEAT DATA_MAX_RECORDS NPROC_PER_NODE NUM_NODES DIST_TIMEOUT_MINUTES; do
    require_positive_int "${positive_name}"
done
require_nonnegative_int NODE_RANK
require_nonnegative_int LENGTH_BUCKET_SIZE
if [[ -n "${EXPECTED_NUM_NODES}" ]]; then
    require_positive_int EXPECTED_NUM_NODES
fi
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    require_positive_int MAX_TRAIN_STEPS
fi
if (( LOCAL_BATCH_SIZE != 1 )); then
    die "Online DeepSeek-V4 training requires LOCAL_BATCH_SIZE=1."
fi
if [[ "${PREPARE_DATASET}" != "true" && "${PREPARE_DATASET}" != "false" ]]; then
    die "PREPARE_DATASET must be true or false; got '${PREPARE_DATASET}'."
fi
if [[ "${PURE_EXPERT_PARALLEL}" != "true" && "${PURE_EXPERT_PARALLEL}" != "false" ]]; then
    die "PURE_EXPERT_PARALLEL must be true or false; got '${PURE_EXPERT_PARALLEL}'."
fi
if [[ -n "${EXPECTED_NUM_NODES}" ]] && (( NUM_NODES != EXPECTED_NUM_NODES )); then
    die "Expected ${EXPECTED_NUM_NODES} nodes, got ${NUM_NODES}."
fi
if (( NODE_RANK >= NUM_NODES )); then
    die "Invalid NODE_RANK=${NODE_RANK} for NUM_NODES=${NUM_NODES}."
fi
if (( NUM_NODES > 1 )) && \
   [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    die "Multi-node execution requires a rank-0 MASTER_ADDR reachable by every node."
fi

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    die "CUDA_VISIBLE_DEVICES must list the local training GPUs."
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_IDS[@]}
if (( NUM_GPUS != NPROC_PER_NODE )); then
    die "NPROC_PER_NODE=${NPROC_PER_NODE}, but CUDA_VISIBLE_DEVICES contains ${NUM_GPUS} GPUs."
fi

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))
if [[ "${PURE_EXPERT_PARALLEL}" == "true" ]]; then
    MODEL_PARALLEL_SIZE=$((CP_SIZE * TP_SIZE * FSDP_SIZE))
    MODEL_PARALLEL_LABEL="CP*TP*FSDP"
    EFFECTIVE_DATA_REPLICAS=$((TOTAL_GPUS / (CP_SIZE * TP_SIZE)))
else
    MODEL_PARALLEL_SIZE=$((CP_SIZE * EP_SIZE * TP_SIZE * FSDP_SIZE))
    MODEL_PARALLEL_LABEL="CP*EP*TP*FSDP"
    EFFECTIVE_DATA_REPLICAS=$((TOTAL_GPUS / (CP_SIZE * EP_SIZE * TP_SIZE)))
fi
if (( TOTAL_GPUS % MODEL_PARALLEL_SIZE != 0 )); then
    die "TOTAL_GPUS=${TOTAL_GPUS} must be divisible by ${MODEL_PARALLEL_LABEL}=${MODEL_PARALLEL_SIZE}."
fi
if [[ "${PURE_EXPERT_PARALLEL}" == "true" ]] && (( TOTAL_GPUS % EP_SIZE != 0 )); then
    die "Pure EP requires TOTAL_GPUS=${TOTAL_GPUS} to be divisible by EP_SIZE=${EP_SIZE}."
fi
DP_SIZE=$((TOTAL_GPUS / MODEL_PARALLEL_SIZE))
if (( GLOBAL_BATCH_SIZE % EFFECTIVE_DATA_REPLICAS != 0 )); then
    die "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by effective data replicas=${EFFECTIVE_DATA_REPLICAS}."
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / EFFECTIVE_DATA_REPLICAS))

if (( LENGTH_BUCKET_SIZE > 0 )) && \
   (( LENGTH_BUCKET_SIZE < EFFECTIVE_DATA_REPLICAS || LENGTH_BUCKET_SIZE % EFFECTIVE_DATA_REPLICAS != 0 )); then
    die "LENGTH_BUCKET_SIZE=${LENGTH_BUCKET_SIZE} must be 0 or a multiple of effective data replicas=${EFFECTIVE_DATA_REPLICAS}."
fi

if (( NUM_ANCHORS < CP_SIZE || NUM_ANCHORS % CP_SIZE != 0 )); then
    die "NUM_ANCHORS=${NUM_ANCHORS} must be a positive multiple of CP_SIZE=${CP_SIZE}."
fi
if (( MAX_LENGTH < CP_SIZE )); then
    die "MAX_LENGTH=${MAX_LENGTH} must provide at least one token per CP rank (${CP_SIZE})."
fi
if (( 256 % EP_SIZE != 0 )); then
    die "DeepSeek-V4 has 256 routed experts; EP_SIZE=${EP_SIZE} must divide 256."
fi
if (( 64 % TP_SIZE != 0 || 8 % TP_SIZE != 0 )); then
    die "TP_SIZE=${TP_SIZE} must divide both 64 attention heads and 8 output groups."
fi
if (( 2048 % TP_SIZE != 0 || 129280 % TP_SIZE != 0 )); then
    die "TP_SIZE=${TP_SIZE} must divide the 2048 expert width and 129280 vocabulary."
fi

if [[ "${PYTHON}" == */* ]]; then
    [[ -x "${PYTHON}" ]] || die "Python executable not found: ${PYTHON}"
else
    command -v "${PYTHON}" >/dev/null 2>&1 || \
        die "Python command not found after sourcing ${ENV_SCRIPT}: ${PYTHON}"
fi
for required_path in "${CONFIG}" "${MODEL_PATH}"; do
    [[ -e "${required_path}" ]] || die "Required path not found: ${required_path}"
done

# Match the reference launcher's CUDA/cuDNN environment without copying its
# WandB credential or Megatron-only settings.
if [[ -z "${CUDNN_HOME:-}" && -z "${CUDNN_PATH:-}" ]]; then
    PYTHON_CUDNN_HOME=$(
        "${PYTHON}" -c \
            'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_path("purelib")) / "nvidia" / "cudnn")'
    )
    if [[ -f "${PYTHON_CUDNN_HOME}/lib/libcudnn.so.9" ]]; then
        export CUDNN_HOME=${PYTHON_CUDNN_HOME}
    fi
    unset PYTHON_CUDNN_HOME
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NVTE_NORM_FWD_USE_CUDNN=${NVTE_NORM_FWD_USE_CUDNN:-1}
export NVTE_NORM_BWD_USE_CUDNN=${NVTE_NORM_BWD_USE_CUDNN:-1}
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export V4_INDEXER_IMPL=${V4_INDEXER_IMPL:-streamindex}
export DEEPSPEC_V4_CP_QUERY_CHUNK=${DEEPSPEC_V4_CP_QUERY_CHUNK:-64}
export DEEPSPEC_V4_CP_INDEX_KEY_CHUNK=${DEEPSPEC_V4_CP_INDEX_KEY_CHUNK:-2048}
export DEEPSPEC_V4_EP_TOKEN_CHUNK=${DEEPSPEC_V4_EP_TOKEN_CHUNK:-4096}
export DEEPSPEC_DIST_TIMEOUT_MINUTES=${DIST_TIMEOUT_MINUTES}
# Per-stage, per-rank progress messages flush synchronously and are intended for
# hang diagnosis, not steady-state throughput.  Set true for a diagnostic run.
export DEEPSPEC_DEBUG_PROGRESS=${DEEPSPEC_DEBUG_PROGRESS:-false}

CACHE_ROOT=${CACHE_ROOT:-/mnt/afs-agentpro/wzj/.cache/deepspec}
COMPILE_CACHE_ROOT=${COMPILE_CACHE_ROOT:-${CACHE_ROOT}/deepseek_v4_cp${CP_SIZE}_ep${EP_SIZE}_tp${TP_SIZE}}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${COMPILE_CACHE_ROOT}/triton/node_rank_${NODE_RANK}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${COMPILE_CACHE_ROOT}/torchinductor/node_rank_${NODE_RANK}}
export TORCHINDUCTOR_FX_GRAPH_CACHE=${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}

RUN_HOME=${RUN_HOME:-${RUN_DIR}/home}
mkdir -p \
    "${RUN_HOME}" \
    "${RUN_DIR}/logs" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "$(dirname -- "${DATA_PATH}")"
export HOME=${RUN_HOME}
export DEEPSPEC_OUTPUT_ROOT=${DEEPSPEC_OUTPUT_ROOT:-${RUN_HOME}}
export PYTHONPATH=${ROOT_DIR}:${PYTHONPATH:-}
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=$(readlink -f "${ROOT_DIR}")

prepare_dataset() {
    if [[ "${PREPARE_DATASET}" == "false" ]]; then
        [[ -s "${DATA_PATH}" ]] || \
            die "DATA_PATH is missing or empty with PREPARE_DATASET=false: ${DATA_PATH}"
        return
    fi
    [[ -f "${SOURCE_DATA_PATH}" ]] || \
        die "Source dataset not found: ${SOURCE_DATA_PATH}"
    [[ "$(readlink -f "${SOURCE_DATA_PATH}")" != "$(readlink -m "${DATA_PATH}")" ]] || \
        die "SOURCE_DATA_PATH and DATA_PATH must be different files."
    command -v flock >/dev/null 2>&1 || \
        die "The 'flock' command is required for safe multi-node dataset conversion."

    # Every node enters the same file lock; only the first stale/missing check
    # performs conversion, and os.replace publishes the completed file atomically.
    (
        flock -x 9
        if [[ ! -s "${DATA_PATH}" || "${SOURCE_DATA_PATH}" -nt "${DATA_PATH}" ]]; then
            echo "Converting ${SOURCE_DATA_PATH} -> ${DATA_PATH} (repeat=${DATA_REPEAT})"
            SOURCE_DATA_PATH="${SOURCE_DATA_PATH}" \
            DATA_PATH="${DATA_PATH}" \
            DATA_REPEAT="${DATA_REPEAT}" \
            DATA_MAX_RECORDS="${DATA_MAX_RECORDS}" \
            "${PYTHON}" - <<'PY'
import json
import os

source_path = os.environ["SOURCE_DATA_PATH"]
data_path = os.environ["DATA_PATH"]
repeat = int(os.environ["DATA_REPEAT"])
max_records = int(os.environ["DATA_MAX_RECORDS"])
role_map = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}

with open(source_path, "r", encoding="utf-8") as handle:
    records = json.load(handle)
if not isinstance(records, list) or not records:
    raise RuntimeError(f"Expected a non-empty JSON array: {source_path}")
records = records[:max_records]

converted = []
for record_index, record in enumerate(records):
    raw_conversation = record.get("conversations")
    if not isinstance(raw_conversation, list) or not raw_conversation:
        raise RuntimeError(
            f"Record {record_index} has no non-empty conversations list."
        )
    conversation = []
    for message_index, message in enumerate(raw_conversation):
        raw_role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        role = role_map.get(raw_role)
        if role is None:
            raise RuntimeError(
                f"Unsupported role at record {record_index}, message "
                f"{message_index}: {raw_role!r}"
            )
        if not isinstance(content, str):
            raise RuntimeError(
                f"Non-string content at record {record_index}, message "
                f"{message_index}: {type(content).__name__}"
            )
        conversation.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in conversation):
        raise RuntimeError(f"Record {record_index} has no assistant turn.")
    converted.append({"conversations": conversation})

tmp_path = f"{data_path}.tmp.{os.getpid()}"
try:
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for _ in range(repeat):
            for record in converted:
                json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, data_path)
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

print(f"Wrote {len(converted) * repeat} DeepSpec JSONL records to {data_path}")
PY
        fi
    ) 9>"${DATA_PATH}.lock"
}

prepare_dataset

DATASET_SIZE=$(wc -l < "${DATA_PATH}")
DATASET_SIZE=${DATASET_SIZE//[[:space:]]/}
require_positive_int DATASET_SIZE
if (( DATASET_SIZE < GLOBAL_BATCH_SIZE )); then
    die "Dataset has ${DATASET_SIZE} records, fewer than GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}."
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
    raise RuntimeError("Each JSONL record must contain a non-empty conversations list.")
if not any(item.get("role") == "assistant" for item in conversation):
    raise RuntimeError("Each training sample must contain at least one assistant turn.")
PY

CUDA_COUNT=$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
if (( CUDA_COUNT != NUM_GPUS )); then
    die "Expected ${NUM_GPUS} CUDA devices, but PyTorch sees ${CUDA_COUNT}."
fi

cd "${ROOT_DIR}"

COMMON_OPTS=(
    --opts "project_name=${PROJECT_NAME}"
    --opts "exp_name=${EXP_NAME}"
    --opts "model.target_model_name_or_path=${MODEL_PATH}"
    --opts "model.num_anchors=${NUM_ANCHORS}"
    --opts "data.chat_template=${CHAT_TEMPLATE}"
    --opts "data.max_length=${MAX_LENGTH}"
    --opts "data.length_bucket_size=${LENGTH_BUCKET_SIZE}"
)

TRAIN_STEP_OPTS=()
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    TRAIN_STEP_OPTS+=(--opts "train.max_train_steps=${MAX_TRAIN_STEPS}")
fi

if (( NODE_RANK == 0 )); then
    echo "Distributed layout: ${NUM_NODES} nodes x ${NUM_GPUS} GPUs = ${TOTAL_GPUS} GPUs"
    echo "Topology: CP=${CP_SIZE}, EP=${EP_SIZE}, TP=${TP_SIZE}, FSDP=${FSDP_SIZE}, DP=${DP_SIZE}"
    echo "CP query chunk=${DEEPSPEC_V4_CP_QUERY_CHUNK}, CP index-key chunk=${DEEPSPEC_V4_CP_INDEX_KEY_CHUNK}"
    echo "EP All-to-All token chunk=${DEEPSPEC_V4_EP_TOKEN_CHUNK}"
    echo "Routed experts=pure EP across the base topology (EP does not multiply world size), distributed timeout=${DEEPSPEC_DIST_TIMEOUT_MINUTES} minutes"
    echo "Target supervision=online (no disk cache), epochs=${NUM_TRAIN_EPOCHS}"
    echo "Effective data replicas=${EFFECTIVE_DATA_REPLICAS}"
    echo "Global batch=${GLOBAL_BATCH_SIZE}, gradient accumulation=${GRADIENT_ACCUMULATION_STEPS}"
    echo "Length-grouped sampling window=${LENGTH_BUCKET_SIZE} global samples (0 disables)"
    echo "Model=${MODEL_PATH}"
    echo "Dataset=${DATA_PATH} (${DATASET_SIZE} records)"
    echo "Output=${RUN_DIR}"
fi

echo "[1/1] Online DeepSeek-V4 target forward + DSpark training on ${TOTAL_GPUS} GPUs"
"${PYTHON}" train.py \
    --config "${CONFIG}" \
    "${COMMON_OPTS[@]}" \
    --opts "data.online_target=true" \
    --opts "data.train_data_path=${DATA_PATH}" \
    --opts "data.target_cache_path=null" \
    --opts "data.min_loss_tokens=${MIN_LOSS_TOKENS}" \
    --opts "data.num_workers=${DATA_NUM_WORKERS}" \
    --opts "data.prefetch_factor=${PREFETCH_FACTOR}" \
    --opts "train.sharding_strategy=${SHARDING_STRATEGY}" \
    --opts "train.fsdp_layerwise=${FSDP_LAYERWISE}" \
    --opts "train.fsdp_size=${FSDP_SIZE}" \
    --opts "train.context_parallel_size=${CP_SIZE}" \
    --opts "train.expert_parallel_size=${EP_SIZE}" \
    --opts "train.tensor_parallel_size=${TP_SIZE}" \
    --opts "train.pure_expert_parallel=${PURE_EXPERT_PARALLEL}" \
    --opts "train.local_batch_size=${LOCAL_BATCH_SIZE}" \
    --opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --opts "train.num_train_epochs=${NUM_TRAIN_EPOCHS}" \
    --opts "train.torch_compile=${TORCH_COMPILE}" \
    --opts "logging.logging_steps=${LOGGING_STEPS}" \
    --opts "logging.checkpointing_steps=${CHECKPOINTING_STEPS}" \
    "${TRAIN_STEP_OPTS[@]}" \
    2>&1 | tee -a "${RUN_DIR}/logs/online_train.node${NODE_RANK}.log"

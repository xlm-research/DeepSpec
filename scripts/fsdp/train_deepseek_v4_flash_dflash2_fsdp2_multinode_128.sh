#!/usr/bin/env bash
set -eo pipefail

env | grep -E '^(LOCAL_RANK|RANK|WORLD_SIZE|LOCAL_WORLD_SIZE|NPROC_PER_NODE|MASTER_ADDR|MASTER_PORT|SENSECORE_PYTORCH_NNODES|SENSECORE_PYTORCH_NODE_RANK)=' || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BASE_DIR}"

# SenseCore invokes this script once on every node. Its outer WORLD_SIZE/RANK
# describe nodes, while torchrun creates the per-GPU process world below.
SCHEDULER_WORLD_SIZE=${WORLD_SIZE:-}
SCHEDULER_RANK=${RANK:-}
NNODES=${SENSECORE_PYTORCH_NNODES:-${SCHEDULER_WORLD_SIZE:-1}}
NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-${SCHEDULER_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29501}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-${BASE_DIR}/train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/output/deepseek_v4_flash_dflash2_fsdp2_multinode_128}
TARGET_CACHE_DIR=${TARGET_CACHE_DIR:-${OUTPUT_ROOT}/target_cache}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}
mkdir -p "${LOG_DIR}"
NODE_LOG=${LOG_DIR}/node_rank_${NODE_RANK}.log
TORCHRUN_LOG_DIR=${LOG_DIR}/torchrun_node_rank_${NODE_RANK}
mkdir -p "${TORCHRUN_LOG_DIR}"

# Preserve the raw shell/Python output and append a short failure diagnosis.
exec > >(tee -a "${NODE_LOG}") 2>&1
START_TIME=$(date '+%Y-%m-%d %H:%M:%S %z')
diagnose_exit() {
    status=$?
    set +e
    echo "[deepspec-launch-exit] time=$(date '+%Y-%m-%d %H:%M:%S %z') host=$(hostname) node_rank=${NODE_RANK} exit_code=${status}"
    if ((status == 0)); then
        echo "[deepspec-launch-diagnosis] completed successfully"
    elif [[ -f "${NODE_LOG}" ]] && grep -qE 'waiting for clients|assigned_ranks|DistStoreError|Rendezvous' "${NODE_LOG}"; then
        echo "[deepspec-launch-diagnosis] rendezvous failure: compare NNODES, NODE_RANK, MASTER_ADDR and MASTER_PORT on every node"
    elif [[ -f "${NODE_LOG}" ]] && grep -qE 'OutOfMemoryError|CUDA out of memory' "${NODE_LOG}"; then
        echo "[deepspec-launch-diagnosis] CUDA OOM: inspect the earliest [deepspec-fatal] rank and its [deepspec-cuda] peak-memory line"
    elif [[ -f "${NODE_LOG}" ]] && grep -qE 'collective operation timeout|ALLTOALL_BASE|Watchdog caught collective' "${NODE_LOG}"; then
        echo "[deepspec-launch-diagnosis] NCCL collective timeout: find the earliest failing rank in the per-rank stderr logs"
    elif [[ -f "${NODE_LOG}" ]] && grep -q '\[deepspec-fatal\]' "${NODE_LOG}"; then
        echo "[deepspec-launch-diagnosis] Python training failure: the first [deepspec-fatal] block contains the original traceback"
    elif [[ -f "${NODE_LOG}" ]] && grep -qE 'must (equal|be|provide)|does not (exist|match)|requires [0-9]+ nodes' "${NODE_LOG}"; then
        echo "[deepspec-launch-diagnosis] preflight configuration failure: fix the validation message immediately above"
    else
        echo "[deepspec-launch-diagnosis] unrecognized launcher failure: inspect ${TORCHRUN_LOG_DIR} and the container termination reason"
    fi
    trap - EXIT
    exit "${status}"
}
trap diagnose_exit EXIT

for topology_var in NNODES NODE_RANK NPROC_PER_NODE; do
    topology_value=${!topology_var}
    if [[ ! "${topology_value}" =~ ^[0-9]+$ ]]; then
        echo "${topology_var} must be a non-negative integer; got ${topology_value}." >&2
        exit 1
    fi
done
if ((NNODES < 1 || NPROC_PER_NODE < 1)); then
    echo "NNODES and NPROC_PER_NODE must be positive; got NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
    exit 1
fi
if ((NODE_RANK >= NNODES)); then
    echo "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}." >&2
    exit 1
fi

TRAIN_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if [[ -n "${SCHEDULER_WORLD_SIZE}" ]]; then
    if [[ ! "${SCHEDULER_WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Scheduler WORLD_SIZE must be a positive integer; got ${SCHEDULER_WORLD_SIZE}." >&2
        exit 1
    fi
    if ((SCHEDULER_WORLD_SIZE == NNODES)); then
        SCHEDULER_WORLD_SIZE_UNIT=nodes
    elif ((SCHEDULER_WORLD_SIZE == TRAIN_WORLD_SIZE)); then
        SCHEDULER_WORLD_SIZE_UNIT=gpu_processes
    else
        echo "Scheduler WORLD_SIZE=${SCHEDULER_WORLD_SIZE} matches neither NNODES=${NNODES} nor NNODES*NPROC_PER_NODE=${TRAIN_WORLD_SIZE}." >&2
        exit 1
    fi
else
    SCHEDULER_WORLD_SIZE_UNIT=not_injected
fi

MAX_LENGTH=${MAX_LENGTH:-131072}
NUM_ANCHORS=${NUM_ANCHORS:-512}
VERIFICATION_BLOCK_SIZE=${VERIFICATION_BLOCK_SIZE:-8}
BLOCK_SIZE=${BLOCK_SIZE:-7}
PROPOSAL_HIDDEN_OFFSET=${PROPOSAL_HIDDEN_OFFSET:-1}
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS:-3}
TARGET_LAYER_IDS=${TARGET_LAYER_IDS:-[0,1,2]}
CONV_KERNEL_SIZE=${CONV_KERNEL_SIZE:-2}
CONV_GROUP_SIZE=${CONV_GROUP_SIZE:-16}
SELECTOR_RANK=${SELECTOR_RANK:-256}
SELECTOR_TOP_K=${SELECTOR_TOP_K:-16}
LEARNING_RATE=${LEARNING_RATE:-0.00001}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
# Let the trainer derive max_train_steps from the usable cached dataset size.
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}
SAVE_STEPS=${SAVE_STEPS:-200}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-true}
DRY_RUN=${DRY_RUN:-false}
TORCHRUN_PER_RANK_LOGS=${TORCHRUN_PER_RANK_LOGS:-true}
PROFILE_ENABLED=${PROFILE_ENABLED:-false}

# One 8-GPU node is one model domain. The trainable DFlash2 draft keeps EP1;
# the frozen MoE target overlays an independent EP8 sparse view on the same
# CP/TP ranks. Additional nodes become data replicas automatically.
DP_SHARD=${DP_SHARD:-1}
CP=${CP:-8}
TP=${TP:-1}
DRAFT_EP=${DRAFT_EP:-1}
TARGET_EP=${TARGET_EP:-8}

for positive_var in DP_SHARD CP TP DRAFT_EP TARGET_EP MAX_LENGTH NUM_ANCHORS VERIFICATION_BLOCK_SIZE BLOCK_SIZE PROPOSAL_HIDDEN_OFFSET NUM_DRAFT_LAYERS CONV_KERNEL_SIZE CONV_GROUP_SIZE SELECTOR_RANK SELECTOR_TOP_K LOCAL_BATCH_SIZE GLOBAL_BATCH_SIZE NUM_TRAIN_EPOCHS SAVE_STEPS; do
    positive_value=${!positive_var}
    if [[ ! "${positive_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${positive_var} must be a positive integer; got ${positive_value}." >&2
        exit 1
    fi
done
if [[ -n "${MAX_TRAIN_STEPS}" ]] && [[ ! "${MAX_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TRAIN_STEPS must be empty or a positive integer; got ${MAX_TRAIN_STEPS}." >&2
    exit 1
fi
if ((MAX_LENGTH > 131072)); then
    echo "MAX_LENGTH must not exceed 131072; got ${MAX_LENGTH}." >&2
    exit 1
fi
if ((PROPOSAL_HIDDEN_OFFSET != 1)); then
    echo "PROPOSAL_HIDDEN_OFFSET must be 1 because verification position 0 is the clean anchor." >&2
    exit 1
fi
if ((BLOCK_SIZE != VERIFICATION_BLOCK_SIZE - 1)); then
    echo "BLOCK_SIZE must equal VERIFICATION_BLOCK_SIZE-1; got ${BLOCK_SIZE} and ${VERIFICATION_BLOCK_SIZE}." >&2
    exit 1
fi
if ((NUM_ANCHORS < CP)); then
    echo "NUM_ANCHORS=${NUM_ANCHORS} must be at least CP=${CP}." >&2
    exit 1
fi
if ((DRAFT_EP != 1)); then
    echo "DeepSeek-V4 DFlash2 uses an isolated draft EP1 mesh; got DRAFT_EP=${DRAFT_EP}." >&2
    exit 1
fi

DENSE_REPLICA_DOMAIN=$((DP_SHARD * CP * TP))
if ((TRAIN_WORLD_SIZE % DENSE_REPLICA_DOMAIN != 0)); then
    echo "TRAIN_WORLD_SIZE=${TRAIN_WORLD_SIZE} must be divisible by DP_SHARD*CP*TP=${DENSE_REPLICA_DOMAIN}." >&2
    exit 1
fi
DP_REPLICATE=$((TRAIN_WORLD_SIZE / DENSE_REPLICA_DOMAIN))
DENSE_PARALLEL_SIZE=$((DP_REPLICATE * DP_SHARD * CP * TP))
FSDP_SHARD_SIZE=$((DP_SHARD * CP))
SPARSE_DOMAIN=$((DP_SHARD * CP * TP))
DATA_PARALLEL_SIZE=$((DP_REPLICATE * DP_SHARD))
MICRO_GLOBAL_BATCH_SIZE=$((DATA_PARALLEL_SIZE * LOCAL_BATCH_SIZE))

if ((TRAIN_WORLD_SIZE != DENSE_PARALLEL_SIZE)); then
    echo "TRAIN_WORLD_SIZE=${TRAIN_WORLD_SIZE} must equal DP_REPLICATE*DP_SHARD*CP*TP=${DENSE_PARALLEL_SIZE}." >&2
    exit 1
fi
if ((SPARSE_DOMAIN % TARGET_EP != 0)); then
    echo "DP_SHARD*CP*TP=${SPARSE_DOMAIN} must be divisible by TARGET_EP=${TARGET_EP}." >&2
    exit 1
fi
TARGET_EFSDP=$((SPARSE_DOMAIN / TARGET_EP))
if ((GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH_SIZE != 0)); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by DATA_PARALLEL_SIZE*LOCAL_BATCH_SIZE=${MICRO_GLOBAL_BATCH_SIZE}." >&2
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
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH_SIZE))

TRAIN_SCHEDULE_ARGS=(
    --opts "train.num_train_epochs=${NUM_TRAIN_EPOCHS}"
    --opts "train.max_train_steps=null"
)
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    TRAIN_SCHEDULE_ARGS=(--opts "train.max_train_steps=${MAX_TRAIN_STEPS}")
fi

PROFILE_ARGS=()
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
    PROFILE_TRACE_DIR=${PROFILE_TRACE_DIR:-${OUTPUT_ROOT}/torch_profile}
    PROFILE_RANKS=${PROFILE_RANKS:-[0]}
    PROFILE_SKIP_FIRST_STEPS=${PROFILE_SKIP_FIRST_STEPS:-$((GRADIENT_ACCUMULATION_STEPS - 1))}
    PROFILE_WAIT_STEPS=${PROFILE_WAIT_STEPS:-0}
    PROFILE_WARMUP_STEPS=${PROFILE_WARMUP_STEPS:-1}
    PROFILE_ACTIVE_STEPS=${PROFILE_ACTIVE_STEPS:-${GRADIENT_ACCUMULATION_STEPS}}
    PROFILE_REPEAT=${PROFILE_REPEAT:-1}
    PROFILE_ROW_LIMIT=${PROFILE_ROW_LIMIT:-100}
    PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-true}
    PROFILE_MEMORY=${PROFILE_MEMORY:-true}
    PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-true}
    PROFILE_WITH_FLOPS=${PROFILE_WITH_FLOPS:-true}
    PROFILE_USE_GZIP=${PROFILE_USE_GZIP:-true}
    for profile_var in PROFILE_SKIP_FIRST_STEPS PROFILE_WAIT_STEPS PROFILE_WARMUP_STEPS; do
        profile_value=${!profile_var}
        if [[ ! "${profile_value}" =~ ^[0-9]+$ ]]; then
            echo "${profile_var} must be a non-negative integer; got ${profile_value}." >&2
            exit 1
        fi
    done
    for profile_var in PROFILE_ACTIVE_STEPS PROFILE_REPEAT PROFILE_ROW_LIMIT; do
        profile_value=${!profile_var}
        if [[ ! "${profile_value}" =~ ^[1-9][0-9]*$ ]]; then
            echo "${profile_var} must be a positive integer; got ${profile_value}." >&2
            exit 1
        fi
    done
    PROFILE_REQUIRED_MICRO_STEPS=$((
        PROFILE_SKIP_FIRST_STEPS
        + PROFILE_REPEAT * (PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS + PROFILE_ACTIVE_STEPS)
    ))
    if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
        PROFILE_AVAILABLE_MICRO_STEPS=$((MAX_TRAIN_STEPS * GRADIENT_ACCUMULATION_STEPS))
        if ((PROFILE_REQUIRED_MICRO_STEPS > PROFILE_AVAILABLE_MICRO_STEPS)); then
            echo "Profiler needs ${PROFILE_REQUIRED_MICRO_STEPS} micro-steps, but MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS} provides only ${PROFILE_AVAILABLE_MICRO_STEPS}." >&2
            exit 1
        fi
    fi
    PROFILE_ARGS=(
        --opts "profiling.enabled=true"
        --opts "profiling.trace_dir=${PROFILE_TRACE_DIR}"
        --opts "profiling.ranks=${PROFILE_RANKS}"
        --opts "profiling.skip_first_steps=${PROFILE_SKIP_FIRST_STEPS}"
        --opts "profiling.wait_steps=${PROFILE_WAIT_STEPS}"
        --opts "profiling.warmup_steps=${PROFILE_WARMUP_STEPS}"
        --opts "profiling.active_steps=${PROFILE_ACTIVE_STEPS}"
        --opts "profiling.repeat=${PROFILE_REPEAT}"
        --opts "profiling.record_shapes=${PROFILE_RECORD_SHAPES}"
        --opts "profiling.profile_memory=${PROFILE_MEMORY}"
        --opts "profiling.with_stack=${PROFILE_WITH_STACK}"
        --opts "profiling.with_flops=${PROFILE_WITH_FLOPS}"
        --opts "profiling.use_gzip=${PROFILE_USE_GZIP}"
        --opts "profiling.row_limit=${PROFILE_ROW_LIMIT}"
    )
fi

echo "Launching DeepSeek-V4-Flash DFlash2 FSDP2 on ${TRAIN_WORLD_SIZE} GPUs:"
echo "  start=${START_TIME}, host=$(hostname), pid=$$"
echo "  node=${NODE_RANK}/${NNODES}, rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
echo "  scheduler world size=${SCHEDULER_WORLD_SIZE:-<unset>} (${SCHEDULER_WORLD_SIZE_UNIT})"
echo "  training world size=${TRAIN_WORLD_SIZE} (=NNODES*NPROC_PER_NODE)"
echo "  dense mesh: DP_REPLICATE=${DP_REPLICATE}, DP_SHARD=${DP_SHARD}, CP=${CP}, TP=${TP}"
echo "  FSDP shard=${FSDP_SHARD_SIZE}, draft EP=${DRAFT_EP}, target EP=${TARGET_EP}, target EFSDP=${TARGET_EFSDP}"
echo "  DFlash2: verification=${VERIFICATION_BLOCK_SIZE}, proposals=${BLOCK_SIZE}, hidden offset=${PROPOSAL_HIDDEN_OFFSET}"
echo "  DFlash2: layers=${NUM_DRAFT_LAYERS}, target layers=${TARGET_LAYER_IDS}, conv=${CONV_KERNEL_SIZE}x/group${CONV_GROUP_SIZE}, selector rank=${SELECTOR_RANK}/top-k=${SELECTOR_TOP_K}"
echo "  local batch=${LOCAL_BATCH_SIZE}, global batch=${GLOBAL_BATCH_SIZE}, gradient accumulation=${GRADIENT_ACCUMULATION_STEPS}"
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    echo "  schedule=fixed diagnostic override, max steps=${MAX_TRAIN_STEPS}"
else
    echo "  schedule=dataset-derived, epochs=${NUM_TRAIN_EPOCHS}, max steps computed after loading the dataset"
fi
echo "  max length=${MAX_LENGTH}, anchors=${NUM_ANCHORS}, logging every step, checkpoint every ${SAVE_STEPS} steps"
echo "  profiler=${PROFILE_ENABLED}, save checkpoints=${SAVE_CHECKPOINTS}"
echo "  offline target cache=${TARGET_CACHE_DIR}"
if [[ "${PROFILE_ENABLED}" == "true" ]]; then
    echo "  profile dir=${PROFILE_TRACE_DIR}, ranks=${PROFILE_RANKS}, schedule(micro-steps)=skip:${PROFILE_SKIP_FIRST_STEPS}/wait:${PROFILE_WAIT_STEPS}/warmup:${PROFILE_WARMUP_STEPS}/active:${PROFILE_ACTIVE_STEPS}/repeat:${PROFILE_REPEAT}"
fi
echo "  node log=${NODE_LOG}"
echo "  per-rank logs=${TORCHRUN_LOG_DIR}"
echo "  master resolution=$(getent hosts "${MASTER_ADDR}" 2>&1 | head -1 || echo '<unresolved>')"
echo "  python=$(command -v python3), torchrun=$(command -v torchrun)"
python3 -c 'import socket, torch; print(f"  runtime: host={socket.gethostname()}, python_torch={torch.__version__}, cuda={torch.version.cuda}, cuda_devices={torch.cuda.device_count()}")'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>&1 || true

export CUDA_VISIBLE_DEVICES
export DEEPSPEC_OUTPUT_ROOT=${OUTPUT_ROOT}
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

LAUNCHER=(torchrun)
if [[ "${DRY_RUN}" == "true" ]]; then
    LAUNCHER=(echo torchrun)
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
    scripts/data/prepare_deepseek_v4_target_cache.py \
    --config config/dflash2/dflash2_deepseek_v4.py \
    --opts "model.target_model_name_or_path=${TARGET_MODEL_PATH}" \
    --opts "model.target_layer_ids=${TARGET_LAYER_IDS}" \
    --opts "data.max_length=${MAX_LENGTH}" \
    --opts "train.parallel.dp_replicate=${DP_REPLICATE}" \
    --opts "train.parallel.dp_shard=${DP_SHARD}" \
    --opts "train.parallel.cp=${CP}" \
    --opts "train.parallel.tp=${TP}" \
    --opts "train.parallel.ep=${DRAFT_EP}" \
    --opts "train.target_parallel.ep=${TARGET_EP}" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir "${TARGET_CACHE_DIR}"

"${LAUNCHER[@]}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TORCHRUN_LOG_ARGS[@]}" \
    train.py \
    --config config/dflash2/dflash2_deepseek_v4.py \
    --opts "model.target_model_name_or_path=${TARGET_MODEL_PATH}" \
    --opts "model.verification_block_size=${VERIFICATION_BLOCK_SIZE}" \
    --opts "model.block_size=${BLOCK_SIZE}" \
    --opts "model.proposal_hidden_offset=${PROPOSAL_HIDDEN_OFFSET}" \
    --opts "model.num_draft_layers=${NUM_DRAFT_LAYERS}" \
    --opts "model.target_layer_ids=${TARGET_LAYER_IDS}" \
    --opts "model.num_anchors=${NUM_ANCHORS}" \
    --opts "model.conv_kernel_size=${CONV_KERNEL_SIZE}" \
    --opts "model.conv_group_size=${CONV_GROUP_SIZE}" \
    --opts "model.selector_rank=${SELECTOR_RANK}" \
    --opts "model.selector_top_k=${SELECTOR_TOP_K}" \
    --opts "data.target_cache_path=${TARGET_CACHE_DIR}" \
    --opts "data.source_jsonl_path=${TRAIN_DATA_PATH}" \
    --opts "data.max_length=${MAX_LENGTH}" \
    --opts "train.lr=${LEARNING_RATE}" \
    --opts "train.local_batch_size=${LOCAL_BATCH_SIZE}" \
    --opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --opts "train.parallel.dp_replicate=${DP_REPLICATE}" \
    --opts "train.parallel.dp_shard=${DP_SHARD}" \
    --opts "train.parallel.cp=${CP}" \
    --opts "train.parallel.tp=${TP}" \
    --opts "train.parallel.ep=${DRAFT_EP}" \
    --opts "train.target_parallel.ep=${TARGET_EP}" \
    "${TRAIN_SCHEDULE_ARGS[@]}" \
    --opts "logging.logging_steps=1" \
    --opts "logging.checkpointing_steps=${SAVE_STEPS}" \
    --opts "logging.save_checkpoints=${SAVE_CHECKPOINTS}" \
    "${PROFILE_ARGS[@]}"

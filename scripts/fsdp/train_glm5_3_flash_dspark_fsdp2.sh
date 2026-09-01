#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
# The cluster prepends /shared/bin, whose git requires a newer glibc than this
# image. Prefer the system git so train.py can record the source revision.
export PATH="/usr/bin:/bin:${PATH}"

if [[ -n "${LOCAL_RANK:-}" ]]; then
    echo "Launch this script once per node, not from inside a torchrun worker." >&2
    exit 2
fi

scheduler_world_size="${WORLD_SIZE:-}"
scheduler_rank="${RANK:-}"
dry_run="${DRY_RUN:-false}"
gpu_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${gpu_devices}" ]]; then
    IFS=',' read -r -a visible_gpus <<< "${gpu_devices}"
    visible_gpu_count=${#visible_gpus[@]}
else
    visible_gpu_count=$(
        python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null \
        || echo 0
    )
fi
if [[ ! "${visible_gpu_count}" =~ ^[0-9]+$ ]]; then
    echo "Unable to determine the number of visible GPUs." >&2
    exit 2
fi
if ((visible_gpu_count == 0)); then
    if [[ "${dry_run}" == "true" ]]; then
        visible_gpu_count="${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-8}}"
    else
        echo "No CUDA GPUs are visible to PyTorch." >&2
        exit 2
    fi
fi
nproc_per_node="${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-${visible_gpu_count}}}"
if [[ ! "${nproc_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer; got ${nproc_per_node}." >&2
    exit 2
fi

# Resolve both launchers that report WORLD_SIZE in nodes (for example the
# SenseCore outer launcher) and launchers that report it in GPU processes.
scheduler_world_size_unit="not_injected"
if [[ -n "${NNODES:-}" ]]; then
    nnodes="${NNODES}"
    topology_source="explicit"
elif [[ -n "${SENSECORE_PYTORCH_NNODES:-}" ]]; then
    nnodes="${SENSECORE_PYTORCH_NNODES}"
    topology_source="SenseCore"
elif [[ -n "${SLURM_NNODES:-}" ]]; then
    nnodes="${SLURM_NNODES}"
    topology_source="Slurm"
elif [[ -n "${scheduler_world_size}" ]]; then
    topology_source="standard WORLD_SIZE/RANK"
    if [[ -n "${LOCAL_WORLD_SIZE:-}" ]] \
        && [[ "${scheduler_world_size}" =~ ^[1-9][0-9]*$ ]] \
        && (( scheduler_world_size % nproc_per_node == 0 )); then
        nnodes=$((scheduler_world_size / nproc_per_node))
        scheduler_world_size_unit="gpu_processes"
    else
        nnodes="${scheduler_world_size}"
        scheduler_world_size_unit="nodes"
    fi
else
    nnodes=1
    topology_source="local"
fi

if [[ -n "${NODE_RANK:-}" ]]; then
    node_rank="${NODE_RANK}"
elif [[ -n "${SENSECORE_PYTORCH_NODE_RANK:-}" ]]; then
    node_rank="${SENSECORE_PYTORCH_NODE_RANK}"
elif [[ -n "${SLURM_NODEID:-}" ]]; then
    node_rank="${SLURM_NODEID}"
elif [[ -n "${scheduler_rank}" ]]; then
    if [[ "${scheduler_world_size_unit}" == "gpu_processes" ]] \
        || { [[ "${scheduler_world_size}" =~ ^[1-9][0-9]*$ ]] \
            && ((scheduler_world_size == nnodes * nproc_per_node)); }; then
        if [[ ! "${scheduler_rank}" =~ ^[0-9]+$ ]]; then
            echo "Scheduler RANK must be a non-negative integer." >&2
            exit 2
        fi
        node_rank=$((scheduler_rank / nproc_per_node))
    else
        node_rank="${scheduler_rank}"
    fi
else
    node_rank=0
fi

master_addr="${MASTER_ADDR:-}"
if [[ -z "${master_addr}" && -n "${SLURM_JOB_NODELIST:-}" ]] \
    && command -v scontrol >/dev/null 2>&1; then
    master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    master_addr=${master_addr%%$'\n'*}
fi
if [[ -z "${master_addr}" ]]; then
    if [[ "${nnodes}" == "1" ]]; then
        master_addr="127.0.0.1"
    else
        echo "The scheduler must provide MASTER_ADDR for a multi-node job." >&2
        exit 2
    fi
fi
master_port="${MASTER_PORT:-29501}"
target_model_path="${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash}"
train_data_path="${TRAIN_DATA_PATH:-}"
output_root="${OUTPUT_ROOT:-${repo_root}/output/glm5_3_flash_dspark_fsdp2}"
jsonl_index_cache_dir="${JSONL_INDEX_CACHE_DIR:-${output_root}/jsonl_index_cache}"
data_batch_cache_dir="${DATA_BATCH_CACHE_DIR:-${TMPDIR:-/tmp}/deepspec_glm5_target_cache_${master_port}}"
max_length="${MAX_LENGTH:-131072}"
num_anchors="${NUM_ANCHORS:-512}"
learning_rate="${LEARNING_RATE:-0.00001}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-1}"
max_train_steps="${MAX_TRAIN_STEPS:-}"
data_batch_size="${DATA_BATCH_SIZE:-256}"
local_batch_size="${LOCAL_BATCH_SIZE:-1}"
save_steps="${SAVE_STEPS:-3000}"
save_checkpoints="${SAVE_CHECKPOINTS:-true}"

for topology_var in nnodes node_rank nproc_per_node visible_gpu_count; do
    topology_value=${!topology_var}
    if [[ ! "${topology_value}" =~ ^[0-9]+$ ]]; then
        echo "${topology_var} must be a non-negative integer; got ${topology_value}." >&2
        exit 2
    fi
done
if ((nnodes < 1 || nproc_per_node < 1)); then
    echo "NNODES and NPROC_PER_NODE must be positive." >&2
    exit 2
fi
if ((node_rank >= nnodes)); then
    echo "NODE_RANK=${node_rank} must be smaller than NNODES=${nnodes}." >&2
    exit 2
fi
if ((visible_gpu_count < nproc_per_node)); then
    echo "NPROC_PER_NODE=${nproc_per_node} exceeds ${visible_gpu_count} visible GPUs." >&2
    exit 2
fi

train_world_size=$((nnodes * nproc_per_node))
if [[ -z "${train_data_path}" ]]; then
    if ((nnodes > 1 || train_world_size > 8)); then
        train_data_path="${repo_root}/train_data/spec_o3_coldstartsft.repeat60.deepspec.jsonl"
    else
        train_data_path="${repo_root}/train_data/spec_o3_coldstartsft.first8.repeat1.deepspec.jsonl"
    fi
fi
if [[ -n "${scheduler_world_size}" ]]; then
    if [[ ! "${scheduler_world_size}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Scheduler WORLD_SIZE must be a positive integer." >&2
        exit 2
    fi
    if ((scheduler_world_size == nnodes)); then
        scheduler_world_size_unit="nodes"
    elif ((scheduler_world_size == train_world_size)); then
        scheduler_world_size_unit="gpu_processes"
    else
        echo "Scheduler WORLD_SIZE=${scheduler_world_size} matches neither NNODES=${nnodes} nor training world size=${train_world_size}." >&2
        exit 2
    fi
fi
if ((train_world_size % 4 != 0)); then
    echo "GLM target TP=4 requires NNODES*NPROC_PER_NODE to be divisible by 4; got ${train_world_size}." >&2
    exit 2
fi
if ((nnodes > 1)) \
    && [[ "${master_addr}" == "127.0.0.1" \
        || "${master_addr}" == "localhost" \
        || "${master_addr}" == "::1" \
        || "${master_addr}" == "[::1]" \
        || "${master_addr}" == "0.0.0.0" \
        || "${master_addr}" == "::" ]]; then
    echo "NNODES=${nnodes} requires a cross-node reachable MASTER_ADDR." >&2
    exit 2
fi

# Default to HSDP: shard within each physical node and replicate across nodes.
# Either dimension can be overridden; the other is then derived from WORLD_SIZE.
dp_replicate="${DP_REPLICATE:-}"
dp_shard="${DP_SHARD:-}"
if [[ -z "${dp_replicate}" && -z "${dp_shard}" ]]; then
    dp_replicate=${nnodes}
    dp_shard=${nproc_per_node}
elif [[ -z "${dp_replicate}" ]]; then
    if [[ ! "${dp_shard}" =~ ^[1-9][0-9]*$ ]] \
        || (( train_world_size % dp_shard != 0 )); then
        echo "DP_SHARD must be a positive divisor of TRAIN_WORLD_SIZE=${train_world_size}." >&2
        exit 2
    fi
    dp_replicate=$((train_world_size / dp_shard))
elif [[ -z "${dp_shard}" ]]; then
    if [[ ! "${dp_replicate}" =~ ^[1-9][0-9]*$ ]] \
        || (( train_world_size % dp_replicate != 0 )); then
        echo "DP_REPLICATE must be a positive divisor of TRAIN_WORLD_SIZE=${train_world_size}." >&2
        exit 2
    fi
    dp_shard=$((train_world_size / dp_replicate))
fi
if [[ ! "${dp_replicate}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${dp_shard}" =~ ^[1-9][0-9]*$ ]] \
    || (( dp_replicate * dp_shard != train_world_size )); then
    echo "Draft DP_REPLICATE*DP_SHARD must equal TRAIN_WORLD_SIZE=${train_world_size}." >&2
    exit 2
fi

gcd() {
    local left=$1
    local right=$2
    local remainder
    while ((right != 0)); do
        remainder=$((left % right))
        left=${right}
        right=${remainder}
    done
    echo "${left}"
}

draft_ep="${DRAFT_EP:-auto}"
if [[ "${draft_ep}" == "auto" ]]; then
    draft_ep=$(gcd "${dp_shard}" 288)
fi
if [[ ! "${draft_ep}" =~ ^[1-9][0-9]*$ ]] \
    || (( dp_shard % draft_ep != 0 )) \
    || (( 288 % draft_ep != 0 )); then
    echo "DRAFT_EP must divide both DP_SHARD=${dp_shard} and 288 experts; got ${draft_ep}." >&2
    exit 2
fi

# Keep TP4 groups and target FSDP shards node-local whenever the physical node
# shape permits it. Otherwise form one valid global mesh; no node count is fixed.
target_dp_replicate="${TARGET_DP_REPLICATE:-}"
target_dp_shard="${TARGET_DP_SHARD:-}"
if [[ -z "${target_dp_replicate}" && -z "${target_dp_shard}" ]]; then
    if ((nproc_per_node % 4 == 0)); then
        target_dp_replicate=${nnodes}
        target_dp_shard=$((nproc_per_node / 4))
    else
        target_dp_replicate=1
        target_dp_shard=$((train_world_size / 4))
    fi
elif [[ -z "${target_dp_replicate}" ]]; then
    target_domain=$((train_world_size / 4))
    if [[ ! "${target_dp_shard}" =~ ^[1-9][0-9]*$ ]] \
        || (( target_domain % target_dp_shard != 0 )); then
        echo "TARGET_DP_SHARD must divide TRAIN_WORLD_SIZE/TP=${target_domain}." >&2
        exit 2
    fi
    target_dp_replicate=$((target_domain / target_dp_shard))
elif [[ -z "${target_dp_shard}" ]]; then
    target_domain=$((train_world_size / 4))
    if [[ ! "${target_dp_replicate}" =~ ^[1-9][0-9]*$ ]] \
        || (( target_domain % target_dp_replicate != 0 )); then
        echo "TARGET_DP_REPLICATE must divide TRAIN_WORLD_SIZE/TP=${target_domain}." >&2
        exit 2
    fi
    target_dp_shard=$((target_domain / target_dp_replicate))
fi
if [[ ! "${target_dp_replicate}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${target_dp_shard}" =~ ^[1-9][0-9]*$ ]] \
    || (( target_dp_replicate * target_dp_shard * 4 != train_world_size )); then
    echo "Target DP_REPLICATE*DP_SHARD*TP4 must equal TRAIN_WORLD_SIZE=${train_world_size}." >&2
    exit 2
fi

global_batch_size="${GLOBAL_BATCH_SIZE:-${train_world_size}}"
for positive_var in max_length num_train_epochs local_batch_size global_batch_size save_steps; do
    positive_value=${!positive_var}
    if [[ ! "${positive_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${positive_var} must be a positive integer; got ${positive_value}." >&2
        exit 2
    fi
done
if [[ "${data_batch_size}" != "auto" ]] \
    && [[ ! "${data_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DATA_BATCH_SIZE must be 'auto' or a positive integer." >&2
    exit 2
fi
if [[ -n "${max_train_steps}" ]] \
    && [[ ! "${max_train_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TRAIN_STEPS must be empty or a positive integer." >&2
    exit 2
fi
if [[ ! "${master_port}" =~ ^[1-9][0-9]*$ ]] \
    || ((master_port > 65535)); then
    echo "MASTER_PORT must be an integer in [1, 65535]." >&2
    exit 2
fi
if ((max_length > 1048576)); then
    echo "MAX_LENGTH must not exceed 1048576." >&2
    exit 2
fi
if ((local_batch_size != 1)); then
    echo "GLM runtime target inference currently requires LOCAL_BATCH_SIZE=1." >&2
    exit 2
fi
if ((global_batch_size % (train_world_size * local_batch_size) != 0)); then
    echo "GLOBAL_BATCH_SIZE must be divisible by TRAIN_WORLD_SIZE*LOCAL_BATCH_SIZE=$((train_world_size * local_batch_size))." >&2
    exit 2
fi
for boolean_var in dry_run save_checkpoints; do
    boolean_value=${!boolean_var}
    if [[ "${boolean_value}" != "true" && "${boolean_value}" != "false" ]]; then
        echo "${boolean_var} must be true or false." >&2
        exit 2
    fi
done
if [[ ! -d "${target_model_path}" ]]; then
    echo "Target model directory does not exist: ${target_model_path}" >&2
    exit 2
fi
if [[ ! -f "${train_data_path}" ]]; then
    echo "Training data does not exist: ${train_data_path}" >&2
    exit 2
fi
if [[ "${dry_run}" != "true" ]]; then
    shard_count=$(find "${target_model_path}" -maxdepth 1 -name 'model-*-of-00062.safetensors' | wc -l)
    if ((shard_count != 62)); then
        echo "GLM-5.3 checkpoint is incomplete: found ${shard_count}/62 safetensor shards." >&2
        exit 2
    fi
    if [[ ! -f "${target_model_path}/model.safetensors.index.json" ]]; then
        echo "Missing ${target_model_path}/model.safetensors.index.json." >&2
        exit 2
    fi
fi

echo "Launching GLM-5.3-Flash bounded offline training on ${train_world_size} GPUs:"
echo "  node=${node_rank}/${nnodes}, local GPUs=${nproc_per_node}, rendezvous=${master_addr}:${master_port}"
echo "  scheduler WORLD_SIZE=${scheduler_world_size:-<unset>} (${scheduler_world_size_unit})"
echo "  topology source=${topology_source}"
echo "  draft HSDP: DP_REPLICATE=${dp_replicate}, DP_SHARD=${dp_shard}, EP=${draft_ep}"
echo "  target HSDP: DP_REPLICATE=${target_dp_replicate}, DP_SHARD=${target_dp_shard}, TP=4, EP=1"
echo "  batch: local=${local_batch_size}, global=${global_batch_size}, data partitions=${data_batch_size}"
if [[ -n "${max_train_steps}" ]]; then
    echo "  schedule: diagnostic max steps=${max_train_steps}"
else
    echo "  schedule: dataset-derived, epochs=${num_train_epochs}"
fi
echo "  shared checkpoint/JSONL index root=${output_root}, ${jsonl_index_cache_dir}"
echo "  node-local or shared transient cache=${data_batch_cache_dir}"
echo "  training data=${train_data_path}"

if [[ -n "${gpu_devices}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu_devices}"
fi
export DEEPSPEC_OUTPUT_ROOT="${output_root}"
export DEEPSPEC_DATA_BATCH_CACHE_DIR="${data_batch_cache_dir}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launcher=(torchrun)
if [[ "${dry_run}" == "true" ]]; then
    launcher=(echo torchrun)
fi

train_schedule_args=(
    --opts "train.num_train_epochs=${num_train_epochs}"
    --opts "train.max_train_steps=null"
)
if [[ -n "${max_train_steps}" ]]; then
    train_schedule_args=(--opts "train.max_train_steps=${max_train_steps}")
fi

"${launcher[@]}" \
    --nproc_per_node "${nproc_per_node}" \
    --nnodes "${nnodes}" \
    --node_rank "${node_rank}" \
    --master_addr "${master_addr}" \
    --master_port "${master_port}" \
    train.py \
    --config config/dspark/dspark_glm5_3_flash.py \
    --opts "model.target_model_name_or_path=${target_model_path}" \
    --opts "data.train_data_path=${train_data_path}" \
    --opts "data.source_jsonl_path=${train_data_path}" \
    --opts "data.jsonl_index_cache_dir=${jsonl_index_cache_dir}" \
    --opts "data.data_batch_cache_dir=${data_batch_cache_dir}" \
    --opts "data.store_target_last_hidden_states=true" \
    --opts "data.max_length=${max_length}" \
    --opts "model.num_anchors=${num_anchors}" \
    --opts "train.lr=${learning_rate}" \
    --opts "train.local_batch_size=${local_batch_size}" \
    --opts "train.global_batch_size=${global_batch_size}" \
    --opts "train.data_batch_size=${data_batch_size}" \
    "${train_schedule_args[@]}" \
    --opts "train.parallel.dp_replicate=${dp_replicate}" \
    --opts "train.parallel.dp_shard=${dp_shard}" \
    --opts "train.parallel.cp=1" \
    --opts "train.parallel.tp=1" \
    --opts "train.parallel.ep=${draft_ep}" \
    --opts "train.offline_target_parallel.dp_replicate=${target_dp_replicate}" \
    --opts "train.offline_target_parallel.dp_shard=${target_dp_shard}" \
    --opts "train.offline_target_parallel.cp=1" \
    --opts "train.offline_target_parallel.tp=4" \
    --opts "train.offline_target_parallel.ep=1" \
    --opts "logging.logging_steps=1" \
    --opts "logging.checkpointing_steps=${save_steps}" \
    --opts "logging.save_checkpoints=${save_checkpoints}"

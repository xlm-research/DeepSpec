#!/usr/bin/env bash
set -euo pipefail

# Production launcher for scheduler-managed homogeneous multi-GPU nodes.
# Configure this command once in SenseCore/Slurm; the scheduler invokes it once
# per node and injects the shared rendezvous topology.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_command="${PYTHON_BIN:-python}"
if ! python_bin=$(command -v "${python_command}"); then
    echo "PYTHON_BIN is not available on PATH: ${python_command}" >&2
    exit 2
fi
# The cluster prepends /shared/bin, whose helper binaries may require a newer
# glibc than this image. Resolve Python first, then prefer the system tools
# without changing the interpreter used for torchrun.
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
        "${python_bin}" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null \
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
if [[ -n "${SENSECORE_PYTORCH_NNODES:-}" ]]; then
    nnodes="${SENSECORE_PYTORCH_NNODES}"
    topology_source="SenseCore"
elif [[ -n "${SLURM_NNODES:-}" ]]; then
    nnodes="${SLURM_NNODES}"
    topology_source="Slurm"
elif [[ -n "${NNODES:-}" ]]; then
    nnodes="${NNODES}"
    topology_source="explicit"
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

if [[ -n "${SENSECORE_PYTORCH_NODE_RANK:-}" ]]; then
    node_rank="${SENSECORE_PYTORCH_NODE_RANK}"
elif [[ -n "${SLURM_NODEID:-}" ]]; then
    node_rank="${SLURM_NODEID}"
elif [[ -n "${NODE_RANK:-}" ]]; then
    node_rank="${NODE_RANK}"
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

output_root="${OUTPUT_ROOT:-${repo_root}/output/glm5_3_flash_dspark_fsdp2}"
log_dir="${LOG_DIR:-${output_root}/logs}"
torchrun_per_rank_logs="${TORCHRUN_PER_RANK_LOGS:-true}"
logging_steps="${LOGGING_STEPS:-1}"
launch_start_time="$(date '+%Y-%m-%d %H:%M:%S %z')"
launch_id="$(date '+%Y%m%d_%H%M%S')_pid$$"
launch_host="$(hostname)"
# Do not interpolate an unvalidated scheduler value into a filesystem path.
node_log_rank="${node_rank}"
if [[ ! "${node_log_rank}" =~ ^[0-9]+$ ]]; then
    node_log_rank="invalid"
fi
node_log="${log_dir}/node_rank_${node_log_rank}.log"
torchrun_log_root="${log_dir}/torchrun_node_rank_${node_log_rank}"
torchrun_log_dir="${torchrun_log_root}/${launch_id}"
torchrun_log_args=()

diagnose_exit() {
    status=$?
    set +e
    echo "[deepspec-launch-exit] time=$(date '+%Y-%m-%d %H:%M:%S %z') host=${launch_host} node_rank=${node_rank} exit_code=${status}"
    if ((status == 0)); then
        if [[ "${dry_run}" == "true" ]]; then
            echo "[deepspec-launch-diagnosis] dry run completed; training was not started"
        else
            echo "[deepspec-launch-diagnosis] training command completed successfully"
        fi
    elif [[ "${torchrun_per_rank_logs}" == "true" && -d "${torchrun_log_dir}" ]]; then
        worker_error=$(
            find "${torchrun_log_dir}" -type f -name error.json -size +0c \
                -printf '%T@ %p\n' 2>/dev/null \
                | sort -n \
                | sed -n '1s/^[^ ]* //p'
        )
        if [[ -z "${worker_error}" ]]; then
            worker_error=$(
                grep -IlR \
                    --include='stderr.log' \
                    -E '\[deepspec-fatal\]|OutOfMemoryError|CUDA out of memory|Watchdog caught collective|collective operation timeout|Traceback \(most recent call last\)' \
                    "${torchrun_log_dir}" 2>/dev/null \
                    | head -n 1
            )
        fi
        if [[ -n "${worker_error}" ]]; then
            echo "[deepspec-launch-diagnosis] worker failure record=${worker_error}"
        else
            echo "[deepspec-launch-diagnosis] no worker failure record was found; inspect ${node_log} and ${torchrun_log_dir}"
        fi
    elif [[ "${torchrun_per_rank_logs}" == "false" ]]; then
        echo "[deepspec-launch-diagnosis] per-rank logs are disabled; all local worker output is in ${node_log}"
    elif [[ ! -f "${node_log}" ]]; then
        echo "[deepspec-launch-diagnosis] node log could not be created at ${node_log}; inspect scheduler output"
    else
        echo "[deepspec-launch-diagnosis] launcher failed before worker logs were created; inspect ${node_log}"
    fi
    trap - EXIT
    exit "${status}"
}
trap diagnose_exit EXIT

if [[ "${dry_run}" != "true" ]]; then
    mkdir -p "${log_dir}"
    # Replace the node summary for compatibility, while keeping each torchrun
    # launch isolated below its own timestamped per-rank directory.
    exec > >(tee "${node_log}") 2>&1
fi

# This is deliberately the first emitted record in a real node log.
echo "[deepspec-launch-start] time=${launch_start_time} host=${launch_host} pid=$$ node_rank=${node_rank} launch_id=${launch_id}"

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
target_model_source_path="${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash}"
target_model_path="${target_model_source_path}"
if [[ -z "${TARGET_MODEL_CACHE_DIR+x}" ]]; then
    target_model_cache_dir="${TMPDIR:-/tmp}/deepspec-model-cache"
    target_model_cache_mode="auto"
else
    target_model_cache_dir="${TARGET_MODEL_CACHE_DIR}"
    target_model_cache_mode="explicit"
fi
target_model_cache_copy_workers="${TARGET_MODEL_CACHE_COPY_WORKERS:-8}"
dcp_load_threads="${DEEPSPEC_DCP_LOAD_THREADS:-8}"
target_model_cache_status="disabled"
if [[ -n "${target_model_cache_dir}" && "${target_model_cache_dir}" != "off" ]]; then
    if [[ ! "${target_model_cache_copy_workers}" =~ ^[1-9][0-9]*$ ]]; then
        echo "TARGET_MODEL_CACHE_COPY_WORKERS must be a positive integer; got ${target_model_cache_copy_workers}." >&2
        exit 2
    fi
    target_model_cache_status="pending"
fi
if [[ ! "${dcp_load_threads}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPSPEC_DCP_LOAD_THREADS must be a positive integer; got ${dcp_load_threads}." >&2
    exit 2
fi
train_data_path="${TRAIN_DATA_PATH:-}"
jsonl_index_cache_dir="${JSONL_INDEX_CACHE_DIR:-${output_root}/jsonl_index_cache}"
data_batch_cache_dir="${DATA_BATCH_CACHE_DIR:-}"
max_length="${MAX_LENGTH:-131072}"
num_anchors="${NUM_ANCHORS:-512}"
learning_rate="${LEARNING_RATE:-0.00001}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-1}"
max_train_steps="${MAX_TRAIN_STEPS:-}"
data_batch_size="${DATA_BATCH_SIZE:-256}"
partitioned_model_swap="${PARTITIONED_MODEL_SWAP:-false}"
partition_max_samples="${PARTITION_MAX_SAMPLES:-512}"
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
target_sparse_domain=$((target_dp_shard * 4))
target_ep="${TARGET_EP:-auto}"
if [[ "${target_ep}" == "auto" ]]; then
    target_ep=$(gcd "${target_sparse_domain}" 288)
fi
if [[ ! "${target_ep}" =~ ^[1-9][0-9]*$ ]] \
    || (( target_sparse_domain % target_ep != 0 )) \
    || (( 288 % target_ep != 0 )); then
    echo "TARGET_EP must divide both target DP_SHARD*TP=${target_sparse_domain} and 288 experts; got ${target_ep}." >&2
    exit 2
fi

if [[ -z "${data_batch_cache_dir}" ]]; then
    cache_topology_name="target_dp${target_dp_replicate}_fsdp${target_dp_shard}_cp1_tp4_ep${target_ep}_etp1__draft_dp${dp_replicate}_fsdp${dp_shard}_cp1_tp1_ep${draft_ep}_etp1"
    data_batch_cache_dir="${output_root}/data_batch_cache/${cache_topology_name}"
fi

global_batch_size="${GLOBAL_BATCH_SIZE:-${train_world_size}}"
for positive_var in max_length num_train_epochs local_batch_size global_batch_size save_steps partition_max_samples logging_steps; do
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
for boolean_var in dry_run save_checkpoints torchrun_per_rank_logs partitioned_model_swap; do
    boolean_value=${!boolean_var}
    if [[ "${boolean_value}" != "true" && "${boolean_value}" != "false" ]]; then
        echo "${boolean_var} must be true or false." >&2
        exit 2
    fi
done
if [[ "${partitioned_model_swap}" == "true" ]]; then
    if [[ "${save_checkpoints}" != "true" ]]; then
        echo "PARTITIONED_MODEL_SWAP=true requires SAVE_CHECKPOINTS=true." >&2
        exit 2
    fi
    if ((partition_max_samples < global_batch_size)); then
        echo "PARTITION_MAX_SAMPLES must be at least GLOBAL_BATCH_SIZE=${global_batch_size}." >&2
        exit 2
    fi
    configured_data_batch_size="null"
else
    configured_data_batch_size="${data_batch_size}"
fi
if [[ ! -d "${target_model_source_path}" ]]; then
    echo "Target model directory does not exist: ${target_model_source_path}" >&2
    exit 2
fi
if [[ ! -f "${train_data_path}" ]]; then
    echo "Training data does not exist: ${train_data_path}" >&2
    exit 2
fi
if [[ "${dry_run}" != "true" ]]; then
    shard_count=$(find "${target_model_source_path}" -maxdepth 1 -name 'model-*-of-00062.safetensors' | wc -l)
    if ((shard_count != 62)); then
        echo "GLM-5.3 checkpoint is incomplete: found ${shard_count}/62 safetensor shards." >&2
        exit 2
    fi
    if [[ ! -f "${target_model_source_path}/model.safetensors.index.json" ]]; then
        echo "Missing ${target_model_source_path}/model.safetensors.index.json." >&2
        exit 2
    fi
fi

checkpoint_fingerprint() {
    local checkpoint_dir=$1
    (
        cd "${checkpoint_dir}"
        sha256sum config.json model.safetensors.index.json
        find . -maxdepth 1 -type f -name 'model-*-of-00062.safetensors' \
            -printf '%f %s\n' | LC_ALL=C sort
    ) | sha256sum | cut -d' ' -f1
}

stage_target_model_checkpoint() {
    local fingerprint cache_path marker lock_path lock_fd
    local cached_fingerprint source_bytes available_bytes required_bytes
    local stage_dir stage_started copied_fingerprint cached_shard_count
    local source_real_path cache_real_path

    if [[ ! -f "${target_model_source_path}/config.json" ]]; then
        echo "[deepspec-model-cache] missing ${target_model_source_path}/config.json" >&2
        return 1
    fi
    if ! mkdir -p "${target_model_cache_dir}"; then
        echo "[deepspec-model-cache] cannot create cache root ${target_model_cache_dir}" >&2
        return 1
    fi
    if ! source_real_path=$(readlink -f "${target_model_source_path}") \
        || ! cache_real_path=$(readlink -f "${target_model_cache_dir}"); then
        echo "[deepspec-model-cache] cannot resolve source or cache path" >&2
        return 1
    fi
    if [[ "${cache_real_path}" == "${source_real_path}"/* ]]; then
        echo "[deepspec-model-cache] cache root must not be inside the source checkpoint: ${cache_real_path}" >&2
        return 1
    fi

    if ! fingerprint=$(checkpoint_fingerprint "${target_model_source_path}"); then
        echo "[deepspec-model-cache] cannot fingerprint ${target_model_source_path}" >&2
        return 1
    fi
    cache_path="${target_model_cache_dir}/glm5-${fingerprint}"
    marker="${cache_path}/.deepspec-cache-ready"
    lock_path="${target_model_cache_dir}/.glm5-${fingerprint}.lock"
    if [[ "$(readlink -m "${cache_path}")" == "${source_real_path}" ]]; then
        target_model_cache_status="already-local"
        return 0
    fi
    if ! exec {lock_fd}>"${lock_path}"; then
        echo "[deepspec-model-cache] cannot open lock ${lock_path}" >&2
        return 1
    fi
    if ! flock -x "${lock_fd}"; then
        echo "[deepspec-model-cache] cannot lock ${lock_path}" >&2
        exec {lock_fd}>&-
        return 1
    fi

    cached_fingerprint=""
    cached_shard_count=0
    if [[ -f "${marker}" ]]; then
        cached_fingerprint=$(sed -n '1p' "${marker}")
        cached_shard_count=$(
            find "${cache_path}" -maxdepth 1 \
                -name 'model-*-of-00062.safetensors' | wc -l
        )
    fi
    if [[ "${cached_fingerprint}" == "${fingerprint}" ]] \
        && ((cached_shard_count == 62)) \
        && [[ -f "${cache_path}/config.json" ]] \
        && [[ -f "${cache_path}/model.safetensors.index.json" ]] \
        && [[ "$(checkpoint_fingerprint "${cache_path}")" == "${fingerprint}" ]]; then
        target_model_path="${cache_path}"
        target_model_cache_status="hit"
        echo "[deepspec-model-cache] time=$(date '+%Y-%m-%d %H:%M:%S %z') cache hit: ${cache_path}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 0
    fi

    if [[ -e "${cache_path}" ]]; then
        echo "[deepspec-model-cache] removing incomplete cache owned by this launcher: ${cache_path}"
        if ! rm -rf -- "${cache_path}"; then
            flock -u "${lock_fd}"
            exec {lock_fd}>&-
            return 1
        fi
    fi
    stage_dir="${target_model_cache_dir}/.glm5-${fingerprint}.partial"
    if [[ -e "${stage_dir}" ]]; then
        echo "[deepspec-model-cache] removing interrupted staging directory: ${stage_dir}"
        if ! rm -rf -- "${stage_dir}"; then
            flock -u "${lock_fd}"
            exec {lock_fd}>&-
            return 1
        fi
    fi

    if ! source_bytes=$(du -sb "${target_model_source_path}" | cut -f1) \
        || ! available_bytes=$(df -PB1 "${target_model_cache_dir}" | awk 'NR == 2 {print $4}') \
        || [[ ! "${source_bytes}" =~ ^[0-9]+$ ]] \
        || [[ ! "${available_bytes}" =~ ^[0-9]+$ ]]; then
        echo "[deepspec-model-cache] cannot determine source size or available space" >&2
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    required_bytes=$((source_bytes + source_bytes / 20))
    if ((available_bytes < required_bytes)); then
        echo "[deepspec-model-cache] insufficient space in ${target_model_cache_dir}: required=${required_bytes} bytes, available=${available_bytes} bytes" >&2
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi

    stage_started=$(date +%s)
    echo "[deepspec-model-cache] time=$(date '+%Y-%m-%d %H:%M:%S %z') staging ${source_bytes} bytes once per node with ${target_model_cache_copy_workers} shard workers: ${target_model_source_path} -> ${cache_path}"
    if ! mkdir -p "${stage_dir}"; then
        echo "[deepspec-model-cache] cannot create staging directory ${stage_dir}" >&2
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! find "${target_model_source_path}" -mindepth 1 -maxdepth 1 \
        ! -name 'model-*-of-00062.safetensors' -print0 \
        | xargs -0 -r -n 1 cp -a --reflink=auto -t "${stage_dir}"; then
        echo "[deepspec-model-cache] checkpoint metadata copy failed; deleting ${stage_dir}" >&2
        rm -rf -- "${stage_dir}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! find "${target_model_source_path}" -maxdepth 1 \
        -name 'model-*-of-00062.safetensors' -print0 \
        | xargs -0 -r -n 1 -P "${target_model_cache_copy_workers}" \
            cp -a --reflink=auto -t "${stage_dir}"; then
        echo "[deepspec-model-cache] checkpoint copy failed; deleting ${stage_dir}" >&2
        rm -rf -- "${stage_dir}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! copied_fingerprint=$(checkpoint_fingerprint "${stage_dir}"); then
        echo "[deepspec-model-cache] cannot validate ${stage_dir}; deleting it" >&2
        rm -rf -- "${stage_dir}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if [[ "${copied_fingerprint}" != "${fingerprint}" ]]; then
        echo "[deepspec-model-cache] source changed or staged checkpoint failed validation; deleting ${stage_dir}" >&2
        rm -rf -- "${stage_dir}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! printf '%s\n' "${fingerprint}" > "${stage_dir}/.deepspec-cache-ready"; then
        echo "[deepspec-model-cache] cannot write ready marker; deleting ${stage_dir}" >&2
        rm -rf -- "${stage_dir}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! mv "${stage_dir}" "${cache_path}"; then
        echo "[deepspec-model-cache] cannot publish staged checkpoint ${cache_path}" >&2
        rm -rf -- "${stage_dir}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi

    target_model_path="${cache_path}"
    target_model_cache_status="staged"
    echo "[deepspec-model-cache] time=$(date '+%Y-%m-%d %H:%M:%S %z') staging complete in $(( $(date +%s) - stage_started ))s: ${cache_path}"
    flock -u "${lock_fd}"
    exec {lock_fd}>&-
}

if [[ "${target_model_cache_status}" == "pending" ]]; then
    if [[ "${dry_run}" == "true" ]]; then
        target_model_cache_status="dry-run"
    elif ! stage_target_model_checkpoint; then
        target_model_path="${target_model_source_path}"
        target_model_cache_status="fallback-to-source"
        if [[ "${target_model_cache_mode}" == "explicit" ]]; then
            echo "Explicit TARGET_MODEL_CACHE_DIR could not be prepared: ${target_model_cache_dir}" >&2
            exit 2
        fi
        echo "[deepspec-model-cache] automatic local caching unavailable; loading directly from ${target_model_source_path}" >&2
    fi
fi

if [[ "${dry_run}" != "true" && "${torchrun_per_rank_logs}" == "true" ]]; then
    mkdir -p "${torchrun_log_dir}"
    torchrun_log_args=(
        --log-dir "${torchrun_log_dir}"
        --redirects 3
        --tee "0:3"
    )
fi

echo "Launching GLM-5.3-Flash bounded offline training on ${train_world_size} GPUs:"
echo "  host=$(hostname), pid=$$"
echo "  node=${node_rank}/${nnodes}, local GPUs=${nproc_per_node}, rendezvous=${master_addr}:${master_port}"
echo "  homogeneous-node requirement=every node must expose at least ${nproc_per_node} visible GPUs"
echo "  scheduler WORLD_SIZE=${scheduler_world_size:-<unset>} (${scheduler_world_size_unit})"
echo "  topology source=${topology_source}"
echo "  draft HSDP: DP_REPLICATE=${dp_replicate}, DP_SHARD=${dp_shard}, EP=${draft_ep}"
echo "  target HSDP: DP_REPLICATE=${target_dp_replicate}, DP_SHARD=${target_dp_shard}, TP=4, EP=${target_ep}"
echo "  batch: local=${local_batch_size}, global=${global_batch_size}, data partitions=${data_batch_size}"
echo "  partitioned model swap=${partitioned_model_swap}, max global samples=${partition_max_samples}"
if [[ -n "${max_train_steps}" ]]; then
    echo "  schedule: diagnostic max steps=${max_train_steps}"
else
    echo "  schedule: dataset-derived, epochs=${num_train_epochs}"
fi
echo "  shared checkpoint/JSONL index root=${output_root}, ${jsonl_index_cache_dir}"
echo "  node-local or shared transient cache=${data_batch_cache_dir}"
echo "  target model source=${target_model_source_path}"
echo "  target model effective path=${target_model_path}"
echo "  target model cache=${target_model_cache_status} (${target_model_cache_dir:-off}, copy workers=${target_model_cache_copy_workers})"
echo "  target DCP reader threads=${dcp_load_threads}"
echo "  training data=${train_data_path}"
echo "  launcher=${python_bin} -m torch.distributed.run"
if [[ "${dry_run}" != "true" ]]; then
    echo "  node log=${node_log}"
    if [[ "${torchrun_per_rank_logs}" == "true" ]]; then
        echo "  per-rank worker logs=${torchrun_log_dir}; console tee=local rank 0 stdout/stderr"
    else
        echo "  per-rank worker logs=disabled; node log receives every local worker"
    fi
fi
echo "  logging steps=${logging_steps}"

if [[ -n "${gpu_devices}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu_devices}"
fi
export DEEPSPEC_OUTPUT_ROOT="${output_root}"
export DEEPSPEC_DATA_BATCH_CACHE_DIR="${data_batch_cache_dir}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DEEPSPEC_DCP_LOAD_THREADS="${dcp_load_threads}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-20000}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"

launcher=("${python_bin}" -m torch.distributed.run)
if [[ "${dry_run}" == "true" ]]; then
    launcher=(echo "${python_bin}" -m torch.distributed.run)
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
    "${torchrun_log_args[@]}" \
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
    --opts "train.data_batch_size=${configured_data_batch_size}" \
    --opts "train.partitioned_model_swap.enabled=${partitioned_model_swap}" \
    --opts "train.partitioned_model_swap.max_samples=${partition_max_samples}" \
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
    --opts "train.offline_target_parallel.ep=${target_ep}" \
    --opts "logging.logging_steps=${logging_steps}" \
    --opts "logging.checkpointing_steps=${save_steps}" \
    --opts "logging.save_checkpoints=${save_checkpoints}"

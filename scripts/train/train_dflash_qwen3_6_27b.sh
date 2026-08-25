#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
context_parallel_size="${CONTEXT_PARALLEL_SIZE:-2}"
train_data_path="${repo_root}/train_data/spec_o3_coldstartsft.repeat60.deepspec.jsonl"
cache_parent="${repo_root}/output/dflash_qwen3_6_27b_target_cache"
target_cache_dir="${cache_parent}/cp${context_parallel_size}"

if [[ ! "${context_parallel_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONTEXT_PARALLEL_SIZE must be a positive integer." >&2
    exit 2
fi

IFS=',' read -r -a visible_gpus <<< "${gpu_devices}"
world_size="${#visible_gpus[@]}"
if (( world_size % context_parallel_size != 0 )); then
    echo "GPU count (${world_size}) must be divisible by CP (${context_parallel_size})." >&2
    exit 2
fi
fsdp_size="$((world_size / context_parallel_size))"

if [[ ! -f "${train_data_path}" ]]; then
    echo "Training data does not exist: ${train_data_path}" >&2
    exit 2
fi
if [[ -L "${target_cache_dir}" ]]; then
    echo "Refusing to use a symlink as the disposable cache directory: ${target_cache_dir}" >&2
    exit 2
fi

# An interrupted cache build has no usable manifest and cannot be resumed.
if [[ -d "${target_cache_dir}" && ! -f "${target_cache_dir}/manifest.json" ]]; then
    echo "Removing incomplete target cache: ${target_cache_dir}"
    rm -rf -- "${target_cache_dir}"
fi

if [[ ! -f "${target_cache_dir}/manifest.json" ]]; then
    mkdir -p "${cache_parent}"
    echo "Preparing offline Qwen3.6 target cache (CP=${context_parallel_size}, FSDP=${fsdp_size})..."
    CUDA_VISIBLE_DEVICES="${gpu_devices}" python scripts/data/prepare_target_cache.py \
        --config config/dflash/dflash_qwen3_6_27b.py \
        --train-data-path "${train_data_path}" \
        --output-dir "${target_cache_dir}" \
        --local-batch-size 1 \
        --num-workers 1 \
        --fsdp \
        --fsdp-size "${fsdp_size}" \
        --context-parallel-size "${context_parallel_size}"
else
    echo "Reusing completed target cache: ${target_cache_dir}"
fi

echo "Training DFlash from the offline target cache..."
CUDA_VISIBLE_DEVICES="${gpu_devices}" python train.py \
    --config config/dflash/dflash_qwen3_6_27b.py \
    --opts "data.target_cache_path=${target_cache_dir}" \
    --opts "train.context_parallel_size=${context_parallel_size}" \
    --opts "train.fsdp_size=${fsdp_size}"

# Only a successful training exit consumes the offline cache.  Failures keep a
# complete cache for retry; incomplete builds are cleaned on the next launch.
if [[ "${target_cache_dir}" != "${repo_root}/output/dflash_qwen3_6_27b_target_cache/cp${context_parallel_size}" ]]; then
    echo "Refusing to delete an unexpected cache path: ${target_cache_dir}" >&2
    exit 2
fi
echo "Training completed; deleting consumed target hidden-state cache: ${target_cache_dir}"
rm -rf -- "${target_cache_dir}"
rmdir "${cache_parent}" 2>/dev/null || true

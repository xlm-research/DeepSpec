#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
# The cluster prepends /shared/bin, whose git requires a newer glibc than this
# image. Prefer the system git so train.py can record the source revision.
export PATH="/usr/bin:/bin:${PATH}"

gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
target_model_path="${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash}"
train_data_path="${TRAIN_DATA_PATH:-${repo_root}/train_data/spec_o3_coldstartsft.first8.repeat1.deepspec.jsonl}"
output_root="${OUTPUT_ROOT:-${repo_root}/output/glm5_3_flash_dspark_fsdp2}"
max_length="${MAX_LENGTH:-131072}"
num_anchors="${NUM_ANCHORS:-512}"
learning_rate="${LEARNING_RATE:-0.00001}"
global_batch_size="${GLOBAL_BATCH_SIZE:-8}"
data_batch_size="${DATA_BATCH_SIZE:-1}"
max_train_steps="${MAX_TRAIN_STEPS:-1}"
dry_run="${DRY_RUN:-false}"

IFS=',' read -r -a visible_gpus <<< "${gpu_devices}"
if (( ${#visible_gpus[@]} != 8 )); then
    echo "GLM-5.3 DSpark currently requires exactly 8 visible GPUs; got ${#visible_gpus[@]}." >&2
    exit 2
fi
if [[ ! "${max_length}" =~ ^[1-9][0-9]*$ ]] || (( max_length > 1048576 )); then
    echo "MAX_LENGTH must be an integer in [1, 1048576]." >&2
    exit 2
fi
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
    if (( shard_count != 62 )); then
        echo "GLM-5.3 checkpoint is incomplete: found ${shard_count}/62 safetensor shards." >&2
        exit 2
    fi
    if [[ ! -f "${target_model_path}/model.safetensors.index.json" ]]; then
        echo "Missing ${target_model_path}/model.safetensors.index.json." >&2
        exit 2
    fi
fi

echo "Training GLM-5.3-Flash DSpark on 8 GPUs (FSDP2 + draft EP8)..."
export CUDA_VISIBLE_DEVICES="${gpu_devices}"
export DEEPSPEC_OUTPUT_ROOT="${output_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launcher=(torchrun)
if [[ "${dry_run}" == "true" ]]; then
    launcher=(echo torchrun)
fi

"${launcher[@]}" --standalone --nproc-per-node=8 train.py \
    --config config/dspark/dspark_glm5_3_flash.py \
    --opts "model.target_model_name_or_path=${target_model_path}" \
    --opts "data.train_data_path=${train_data_path}" \
    --opts "data.max_length=${max_length}" \
    --opts "model.num_anchors=${num_anchors}" \
    --opts "train.lr=${learning_rate}" \
    --opts "train.global_batch_size=${global_batch_size}" \
    --opts "train.data_batch_size=${data_batch_size}" \
    --opts "train.max_train_steps=${max_train_steps}" \
    --opts "logging.logging_steps=1"

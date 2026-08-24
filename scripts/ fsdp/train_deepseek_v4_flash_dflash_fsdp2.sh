#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
target_model_path="${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731}"
train_data_path="${TRAIN_DATA_PATH:-${repo_root}/train_data/spec_o3_coldstartsft.first8.repeat1.deepspec.jsonl}"
output_root="${OUTPUT_ROOT:-${repo_root}/output/deepseek_v4_flash_dflash_fsdp2}"
max_length="${MAX_LENGTH:-131072}"
num_anchors="${NUM_ANCHORS:-512}"
learning_rate="${LEARNING_RATE:-0.00001}"
global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
max_train_steps="${MAX_TRAIN_STEPS:-5}"

IFS=',' read -r -a visible_gpus <<< "${gpu_devices}"
if (( ${#visible_gpus[@]} != 8 )); then
    echo "This config requires exactly 8 visible GPUs; got ${#visible_gpus[@]}." >&2
    exit 2
fi
if [[ ! "${max_length}" =~ ^[1-9][0-9]*$ ]] || (( max_length > 131072 )); then
    echo "MAX_LENGTH must be an integer in [1, 131072]." >&2
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

echo "Training DeepSeek-V4-Flash DFlash (draft EP1, target EP8)..."
CUDA_VISIBLE_DEVICES="${gpu_devices}" \
DEEPSPEC_OUTPUT_ROOT="${output_root}" \
torchrun --standalone --nproc-per-node=8 train.py \
    --config config/dflash/dflash_deepseek_v4.py \
    --opts "model.target_model_name_or_path=${target_model_path}" \
    --opts "data.train_data_path=${train_data_path}" \
    --opts "data.max_length=${max_length}" \
    --opts "model.num_anchors=${num_anchors}" \
    --opts "train.lr=${learning_rate}" \
    --opts "train.global_batch_size=${global_batch_size}" \
    --opts "train.parallel.ep=1" \
    --opts "train.target_parallel.ep=8" \
    --opts "train.max_train_steps=${max_train_steps}" \
    --opts "logging.logging_steps=1"

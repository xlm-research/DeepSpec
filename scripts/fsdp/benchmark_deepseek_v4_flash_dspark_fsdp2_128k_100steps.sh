#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
target_model_path="${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731}"
train_data_path="${TRAIN_DATA_PATH:-${repo_root}/train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl}"
output_root="${OUTPUT_ROOT:-${repo_root}/output/deepseek_v4_flash_dspark_fsdp2_128k_100step_benchmark}"

IFS=',' read -r -a visible_gpus <<< "${gpu_devices}"
if (( ${#visible_gpus[@]} != 8 )); then
    echo "This benchmark requires exactly 8 visible GPUs; got ${#visible_gpus[@]}." >&2
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

echo "Benchmarking 100 DSpark optimizer steps at 128K with per-micro-batch target inference..."
mkdir -p "${output_root}/benchmark"
CUDA_VISIBLE_DEVICES="${gpu_devices}" \
DEEPSPEC_OUTPUT_ROOT="${output_root}" \
torchrun --standalone --nproc-per-node=8 scripts/benchmark_train_no_checkpoint.py \
    --config config/dspark/dspark_deepseek_v4.py \
    --opts "model.target_model_name_or_path=${target_model_path}" \
    --opts "data.train_data_path=${train_data_path}" \
    --opts "data.max_length=131072" \
    --opts "model.num_anchors=512" \
    --opts "train.lr=0.00001" \
    --opts "train.global_batch_size=4" \
    --opts "train.parallel.ep=1" \
    --opts "train.target_parallel.ep=8" \
    --opts "train.max_train_steps=100" \
    --opts "logging.logging_steps=1" \
    --opts "logging.checkpointing_steps=1000000000" \
    2>&1 | tee "${output_root}/benchmark/train.log"

echo "Benchmark complete. No model or optimizer weights were saved."

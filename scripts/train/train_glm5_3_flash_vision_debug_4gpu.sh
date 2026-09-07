#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Four B300 GPUs are sufficient for the full GLM-5.3 visual target with TP=4.
# Override any value on the command line, for example MAX_TRAIN_STEPS=10.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${repo_root}/train_data/glm5_vision_debug.jsonl}"
export MEDIA_ROOT="${MEDIA_ROOT:-${repo_root}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${repo_root}/output/glm5_vision_train}"

export MULTIMODAL=true
export TARGET_BACKEND="${TARGET_BACKEND:-native}"
export MAX_LENGTH="${MAX_LENGTH:-131072}"
export LOCAL_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS-1}"

export PARTITIONED_MODEL_SWAP=true
export PARTITION_MAX_SAMPLES="${PARTITION_MAX_SAMPLES:-512}"
export SAVE_CHECKPOINTS=true
export SAVE_STEPS="${SAVE_STEPS:-1}"

exec bash "${repo_root}/scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh"

#!/usr/bin/env bash
set -euo pipefail

# Production launcher for DeepSeek-V4-Flash DSpark on SenseCore 16 x 8 GPUs.
# SenseCore must invoke this script once on every node and provide the shared
# rendezvous variables. Change OUTPUT_ROOT to start a separate training run;
# reusing it intentionally enables checkpoint auto-resume.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BASE_DIR}"

SCHEDULER_NNODES=${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-}}
if [[ -z "${SCHEDULER_NNODES}" ]]; then
    echo "A SenseCore 16-node job is required; no scheduler node count was injected." >&2
    exit 1
fi
if [[ ! "${SCHEDULER_NNODES}" =~ ^[0-9]+$ ]] || ((SCHEDULER_NNODES != 16)); then
    echo "This production launcher requires 16 nodes x 8 GPUs; got node count ${SCHEDULER_NNODES}." >&2
    exit 1
fi

export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731}
export TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/mnt/afs-agentpro/hongjiawei/code/DeepSpec-old/scripts/data/sensenova_flash_v15_use_data_json.thinking_toolcall.readable_full.flash_conversations.jsonl}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/output/dspark_128gpu_production}
export JSONL_INDEX_CACHE_DIR=${JSONL_INDEX_CACHE_DIR:-${BASE_DIR}/output/jsonl_index_cache}
export DATA_BATCH_CACHE_DIR=${DATA_BATCH_CACHE_DIR:-${OUTPUT_ROOT}/target_data_batch_cache}

export MAX_LENGTH=131072
export NUM_ANCHORS=512
export BLOCK_SIZE=7
export NUM_DRAFT_LAYERS=3
export TARGET_LAYER_IDS='[0,1,2]'
export LEARNING_RATE=${LEARNING_RATE:-0.00001}
export LOCAL_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=64
export DATA_BATCH_SIZE=128
export NUM_TRAIN_EPOCHS=1

export DP_SHARD=1
export CP=8
export TP=1
# Pure draft EP8 can execute, but its sharded-expert checkpoint export is not
# production-safe yet. Keep draft EP1 until save/resume is EP-aware and tested.
export DRAFT_EP=1
export TARGET_EP=8

export RESHARD_AFTER_FORWARD=false
export FSDP_FORWARD_PREFETCH=true
export FSDP_BACKWARD_PREFETCH=true
export FSDP_PREFETCH_DEPTH=2
export FSDP_REDUCE_DTYPE=bf16
export FSDP_WRAP_GRANULARITY=block

export SAVE_STEPS=200
export SAVE_CHECKPOINTS=true
export PROFILE_ENABLED=false
export TORCHRUN_PER_RANK_LOGS=true
export DRY_RUN=false
export PRODUCTION_RUN=true

# A production run must use the dataset-derived one-epoch schedule.
unset MAX_TRAIN_STEPS

exec bash scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh

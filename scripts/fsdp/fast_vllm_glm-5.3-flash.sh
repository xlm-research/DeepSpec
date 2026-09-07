#!/usr/bin/env bash
set -euo pipefail

# Quick start: vLLM target features -> draft training, one dataset partition
# at a time. The main launcher handles both single-node and multi-node jobs;
# invoke this script once per node with the scheduler's rendezvous environment.
#
# Train your full dataset:
#   TRAIN_DATA_PATH=/path/to/train.jsonl bash scripts/fsdp/fast_vllm_glm-5.3-flash.sh
# Preview the launch command without loading models or creating caches:
#   DRY_RUN=true CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash scripts/fsdp/fast_vllm_glm-5.3-flash.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-${repo_root}/vllm/.venv/bin/python}"
export VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-${PYTHON_BIN}}"
export TARGET_BACKEND=vllm
export PARTITIONED_MODEL_SWAP=true
export DATA_BATCH_SIZE="${DATA_BATCH_SIZE:-8}"
export MAX_LENGTH="${MAX_LENGTH:-131072}"
export VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-instanttensor}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"

# Keep weight pre-caching enabled. Reuse the main launcher's fingerprint check,
# copy lock and atomic publication, with an AFS cache instead of its /tmp default.
# Checkpoints, logs, JSONL indices and partition features also default under
# OUTPUT_ROOT in the main launcher. Raw vLLM exports are temporary feature files.
export OUTPUT_ROOT="${OUTPUT_ROOT:-${repo_root}/output/glm5_3_flash_dspark_fsdp2}"
export TARGET_MODEL_CACHE_DIR="${TARGET_MODEL_CACHE_DIR-${OUTPUT_ROOT}/model_cache}"
export VLLM_RAW_CACHE_DIR="${VLLM_RAW_CACHE_DIR:-${OUTPUT_ROOT}/vllm_raw_cache}"

exec bash "${repo_root}/scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh"

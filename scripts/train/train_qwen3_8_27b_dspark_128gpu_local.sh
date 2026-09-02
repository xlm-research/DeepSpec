#!/usr/bin/env bash
set -euo pipefail

# Local single-node entry point using TP4 without context parallelism.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

unset SENSECORE_PYTORCH_NNODES SENSECORE_PYTORCH_NODE_RANK
unset WORLD_SIZE LOCAL_WORLD_SIZE RANK LOCAL_RANK NPROC_PER_NODE
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
export TORCH_COMPILE=${TORCH_COMPILE:-false}

exec "${SCRIPT_DIR}/train_qwen3_8_27b_dspark_128gpu.sh"

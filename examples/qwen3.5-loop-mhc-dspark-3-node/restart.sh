#!/usr/bin/env bash
# Resume the existing three-node Loop/mHC run with accelerated target inference.
#
# Usage: bash examples/qwen3.5-loop-mhc-dspark-3-node/restart.sh [--dry-run|--print-config-only] [key=value ...]
# Run on the existing Ray head after stopping the previous training process.
#
# Defaults restore the latest checkpoint from the 20261012_165500 run, reuse
# its dataset cache and node placement, and write to a new output directory.
# Model, optimizer, scheduler, RNG and step state are restored; the draft
# model configuration is unchanged. Target mHC defaults to triton/all.
#
# Optional environment variables:
#   LOAD_PATH           Checkpoints directory; defaults to the original run
#   CKPT_STEP           Specific iteration; unset selects the latest saved one
#   OUTPUT_DIR          New output directory; defaults to a timestamped path
#   CACHE_DIR           Dataset cache; defaults to the original run's cache
#   TRAIN_NODE_IPS      Defaults to 10.120.0.211
#   INFERENCE_NODE_IPS  Defaults to 10.120.0.218,10.120.0.187
#   RAY_ADDRESS         Existing Ray address (run.sh defaults to auto)
#   WANDB_ENTITY        Defaults to the original run's W&B team
#                       (2718328742-central-south-university)
#   WANDB_PROJECT       Defaults to torchspec-qwen-loop
#   VLLM_QWEN3_5_MHC_BACKEND  triton (default), or torch
#   VLLM_QWEN3_5_MHC_FUSION   all (default), or none for the reference path
#   TORCHSPEC_TRITON_CACHE_DIR  Node-local cache (default: /tmp/torchspec-triton-$UID)
# Other environment variables and CLI overrides are forwarded to run.sh.
# --dry-run validates checkpoint components and prints the command without
# loading the environment, contacting Ray, or starting training.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,/^$/p' "${BASH_SOURCE[0]}"
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

export LOAD_PATH="${LOAD_PATH:-$ROOT_DIR/outputs/qwen3.5-loop-dspark-128k_20261012_165500/checkpoints}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3.5-loop-dspark-128k_resume_$(date +%Y%m%d_%H%M%S)}"
export CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/cache/qwen3.5-loop-dspark-128k-vllm-decode-3node}"
export TRAIN_NODE_IPS="${TRAIN_NODE_IPS:-10.120.0.211}"
export INFERENCE_NODE_IPS="${INFERENCE_NODE_IPS:-10.120.0.218,10.120.0.187}"
export VLLM_QWEN3_5_MHC_BACKEND="${VLLM_QWEN3_5_MHC_BACKEND:-triton}"
export VLLM_QWEN3_5_MHC_FUSION="${VLLM_QWEN3_5_MHC_FUSION:-all}"
# Keep resumed runs in the original W&B team and project.
export WANDB_ENTITY="${WANDB_ENTITY:-2718328742-central-south-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-torchspec-qwen-loop}"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'WANDB_ENTITY=%q WANDB_PROJECT=%q VLLM_QWEN3_5_MHC_BACKEND=%q VLLM_QWEN3_5_MHC_FUSION=%q ' \
    "$WANDB_ENTITY" "$WANDB_PROJECT" \
    "$VLLM_QWEN3_5_MHC_BACKEND" "$VLLM_QWEN3_5_MHC_FUSION"
fi

exec bash "$SCRIPT_DIR/run.sh" "$@" training.continual_training=false

#!/usr/bin/env bash
# Launch Qwen3.8-27B DSpark train-with-decode on one 8-GPU node.
#
# Default topology: 4 GPUs for FSDP draft training, 4 GPUs for one vLLM
# DSpark engine (TP=4, PP=1). TorchSpec starts a local Ray instance.
#
# Usage:
#   bash examples/train-with-decode/run_vllm_qwen3_8_27b_dspark_1node.sh \
#     [config.yaml] [OmegaConf overrides ...]
#
# Useful environment variables:
#   CUDA_VISIBLE_DEVICES  GPUs to use (default: 0,1,2,3,4,5,6,7)
#   TARGET_MODEL_PATH     Qwen3.8-27B checkpoint path (set by h800conda.sh)
#   TRAIN_DATA_PATH       Training dataset path (overrides the YAML config)
#   EVAL_DATA_PATH        Optional evaluation dataset override
#   TRAIN_GPUS            Training GPUs (default: 4)
#   INFERENCE_GPUS        vLLM GPUs / TP size (default: 4; total must be 8)
#   DSPARK_BLOCK_SIZE     DSpark block/draft length (default: 7)
#   MOONCAKE_PROTOCOL     tcp or rdma (default: tcp)
#   CONFIG_FILE           Config path (default: the bundled single-node config)
#   OUTPUT_DIR            Optional checkpoint/output directory override
#   CACHE_DIR             Optional cache directory override
#   LOG_DIR               Log directory (default: running_logs)
#   RAY_ADDRESS           Ray address (default: local)
#   WANDB_PROJECT         W&B project (default: torchspec)
#   WANDB_ENTITY          W&B account/team (default: the logged-in account)
#   WANDB_NAME            W&B run-group prefix (default: qwen3_8_27b-dspark-1node)
# W&B uses existing login credentials or WANDB_API_KEY from the environment.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,/^$/p' "${BASH_SOURCE[0]}"
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/train_with_decode/vllm_qwen3_8_27b_dspark_1node.yaml}"

if [[ $# -gt 0 && "$1" != *=* && "$1" != --* ]]; then
  CONFIG_FILE="$1"
  shift
fi
if [[ -f "$CONFIG_FILE" ]]; then
  CONFIG_FILE="$(cd -- "$(dirname -- "$CONFIG_FILE")" && pwd)/$(basename -- "$CONFIG_FILE")"
elif [[ -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
  CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
else
  echo "ERROR: config file does not exist: $CONFIG_FILE" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"

TRAIN_GPUS="${TRAIN_GPUS:-4}"
INFERENCE_GPUS="${INFERENCE_GPUS:-4}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/sensenova-flash-lite-v42-text-all.jsonl}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/running_logs}"

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $name must be a positive integer, got '$value'" >&2
    exit 1
  fi
}

require_positive_int TRAIN_GPUS "$TRAIN_GPUS"
require_positive_int INFERENCE_GPUS "$INFERENCE_GPUS"
require_positive_int DSPARK_BLOCK_SIZE "$DSPARK_BLOCK_SIZE"

if (( TRAIN_GPUS + INFERENCE_GPUS != 8 )); then
  echo "ERROR: TRAIN_GPUS + INFERENCE_GPUS must equal 8" >&2
  exit 1
fi
IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#GPU_ARRAY[@]} < 8 )); then
  echo "ERROR: CUDA_VISIBLE_DEVICES must expose at least 8 GPUs" >&2
  exit 1
fi

cd "$ROOT_DIR"
# Activate the same environment and local vLLM checkout as the 3-node recipe.
source "$ROOT_DIR/h800conda.sh"

topology_overrides=(
  "training.training_num_nodes=1"
  "training.training_num_gpus_per_node=$TRAIN_GPUS"
  "training.train_with_decode=true"
  "training.dflash_block_size=$DSPARK_BLOCK_SIZE"
  "dataset.min_loss_tokens=$((2 * DSPARK_BLOCK_SIZE))"
  "inference.inference_engine_type=vllm"
  "inference.inference_num_gpus=$INFERENCE_GPUS"
  "inference.inference_num_gpus_per_engine=$INFERENCE_GPUS"
  "inference.inference_num_gpus_per_node=8"
  "inference.vllm.nnodes=1"
  "inference.vllm.tp_size=$INFERENCE_GPUS"
  "inference.vllm.pp_size=1"
  "decode.speculative_algorithm=dspark"
  "decode.speculative_num_draft_tokens=$DSPARK_BLOCK_SIZE"
  "mooncake.protocol=$MOONCAKE_PROTOCOL"
)

if [[ -n "${TRAIN_DATA_PATH:-}" ]]; then
  topology_overrides+=("dataset.train_data_path=$TRAIN_DATA_PATH")
fi
if [[ -n "${EVAL_DATA_PATH:-}" ]]; then
  topology_overrides+=("dataset.eval_data_path=$EVAL_DATA_PATH")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  topology_overrides+=("output_dir=$OUTPUT_DIR")
fi
if [[ -n "${CACHE_DIR:-}" ]]; then
  topology_overrides+=("cache_dir=$CACHE_DIR")
fi
if [[ -n "${TRAIN_ENV_VARS:-}" ]]; then
  topology_overrides+=("training.train_env_vars=$TRAIN_ENV_VARS")
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/qwen3_8_27b_dspark_vllm_1node_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "Qwen3.8-27B DSpark vLLM train-with-decode (single node, 8 GPUs)"
echo "Config: $CONFIG_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES ($TRAIN_GPUS training + $INFERENCE_GPUS inference, TP=$INFERENCE_GPUS)"
echo "DSpark block size: $DSPARK_BLOCK_SIZE; Mooncake: $MOONCAKE_PROTOCOL"
echo "Ray: $RAY_ADDRESS; Log: $log_file"

python -m torchspec.train_entry \
  --config "$CONFIG_FILE" \
  "${topology_overrides[@]}" \
  "$@"

echo "Training completed"

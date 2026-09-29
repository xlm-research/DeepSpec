#!/usr/bin/env bash
# Launch Qwen3.8-27B DSpark with 128 GPUs and a 131072-token context.
#
# Topology: 2 x 8 GPUs for FSDP training, 14 x 8 GPUs for independent vLLM
# engines (TP=8, PP=1). Global batch size stays 32: 16 ranks x 1 sample x 2.
# Start a Ray cluster with sixteen 8-GPU nodes before launching this script.
#
# Usage:
#   bash examples/train-with-decode/run_vllm_qwen3_8_27b_dspark_128gpu.sh \
#     [--dry-run] [config.yaml] [OmegaConf overrides ...]
#
# By default this starts a NEW run, overriding the old checkpoint in the
# 3-node YAML. To resume, set LOAD_PATH to a complete checkpoints directory;
# CKPT_STEP optionally selects an iteration (otherwise use the latest).
#
# Environment variables:
#   LOAD_PATH           Checkpoints directory containing model + optimizer
#   CKPT_STEP           Optional checkpoint iteration, e.g. 1000
#   TRAIN_NODE_IPS      Optional comma-separated list of 2 training node IPs
#   INFERENCE_NODE_IPS  Optional comma-separated list of 14 inference node IPs
#   TRAIN_DATA_PATH     Defaults to mydata/sensenova-flash-lite-v42-text-1000000.jsonl
#   CONFIG_FILE         Defaults to the existing DSpark 3-node YAML
#   OUTPUT_DIR          Defaults to a new timestamped 128-GPU output directory
#   CACHE_DIR           Defaults to the existing 3-node tokenization cache
#   LOG_DIR             Defaults to running_logs
#   MOONCAKE_PROTOCOL   tcp (default) or rdma, if configured on all nodes
#   TARGET_MODEL_PATH   Target checkpoint; default supplied by h800conda.sh
#   RAY_ADDRESS         Existing Ray cluster address, if not discovered locally
#
# --dry-run prints the command without activating Conda, contacting Ray, or
# starting training. Explicit OmegaConf overrides take precedence over defaults.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,/^$/p' "${BASH_SOURCE[0]}"
  exit 0
fi

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/train_with_decode/vllm_qwen3_8_27b_dspark_3node.yaml}"
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
cd "$ROOT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3_8_27b-dspark-128k-128gpu_$timestamp}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/cache/qwen3_8_27b-dspark-vllm-decode-3node}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/running_logs}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$ROOT_DIR/mydata/sensenova-flash-lite-v42-text-1000000.jsonl}"
export WANDB_NAME="${WANDB_NAME:-qwen3_8_27b-dspark-128gpu_$timestamp}"

# The trainer silently skips a missing checkpoint. Check explicit resume
# requests before allocating the cluster so a typo cannot start a fresh run.
resume_step=""
if [[ -n "${LOAD_PATH:-}" ]]; then
  resume_step="${CKPT_STEP:-}"
  if [[ -z "$resume_step" ]]; then
    tracker="$LOAD_PATH/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker" ]]; then
      echo "ERROR: missing checkpoint tracker: $tracker; set CKPT_STEP explicitly if needed." >&2
      exit 1
    fi
    resume_step="$(<"$tracker")"
  fi
  if [[ ! "$resume_step" =~ ^[0-9]+$ ]]; then
    echo "ERROR: checkpoint iteration must be a non-negative integer: $resume_step" >&2
    exit 1
  fi
  resume_step="$((10#$resume_step))"
  printf -v checkpoint_name 'iter_%07d' "$resume_step"
  for component in model optimizer; do
    if [[ ! -d "$LOAD_PATH/$checkpoint_name/$component" ]]; then
      echo "ERROR: incomplete checkpoint: missing $LOAD_PATH/$checkpoint_name/$component" >&2
      exit 1
    fi
  done
elif [[ -n "${CKPT_STEP:-}" ]]; then
  echo "ERROR: CKPT_STEP requires LOAD_PATH." >&2
  exit 1
fi

overrides=(
  "dataset.train_data_path=$TRAIN_DATA_PATH"
  "dataset.min_loss_tokens=14"
  "training.training_num_nodes=2"
  "training.training_num_gpus_per_node=8"
  "training.micro_batch_size=1"
  "training.draft_accumulation_steps=2"
  "training.prefetch_depth=4"
  "training.max_seq_length=131072"
  "training.num_epochs=1"
  "training.train_with_decode=true"
  "training.dflash_block_size=7"
  "training.load_path=${LOAD_PATH:-null}"
  "training.ckpt_step=${resume_step:-null}"
  "training.continual_training=false"
  "inference.inference_engine_type=vllm"
  "inference.inference_num_gpus=112"
  "inference.inference_num_gpus_per_engine=8"
  "inference.inference_num_gpus_per_node=8"
  "inference.inference_batch_size=4"
  "inference.inference_buffer_threshold=128"
  "inference.inference_fetch_batch=128"
  "inference.max_sample_pool_size=128"
  "inference.vllm.nnodes=1"
  "inference.vllm.tp_size=8"
  "inference.vllm.pp_size=1"
  "inference.vllm.extra_args.max_model_len=131072"
  "inference.vllm.extra_args.max_num_batched_tokens=8192"
  "decode.speculative_algorithm=dspark"
  "decode.speculative_num_draft_tokens=7"
  "decode.weight_sync_enabled=true"
  "decode.weight_sync_interval=100"
  "mooncake.protocol=${MOONCAKE_PROTOCOL:-tcp}"
  "output_dir=$OUTPUT_DIR"
  "cache_dir=$CACHE_DIR"
)

format_ip_list() {
  local raw="$1" expected="$2" name="$3" ip result="["
  local -a ips
  IFS=',' read -r -a ips <<< "$raw"
  if (( ${#ips[@]} != expected )) || [[ "$raw" == *, ]]; then
    echo "ERROR: $name must contain $expected comma-separated node IPs." >&2
    return 1
  fi
  for ip in "${ips[@]}"; do
    if [[ -z "$ip" || "$ip" == *[[:space:]\"\\]* ]]; then
      echo "ERROR: invalid entry in $name: '$ip'" >&2
      return 1
    fi
    [[ "$result" == "[" ]] || result+=","
    result+="\"$ip\""
  done
  printf '%s]' "$result"
}

if [[ -n "${TRAIN_NODE_IPS:-}" || -n "${INFERENCE_NODE_IPS:-}" ]]; then
  train_ips="$(format_ip_list "${TRAIN_NODE_IPS:-}" 2 TRAIN_NODE_IPS)"
  inference_ips="$(format_ip_list "${INFERENCE_NODE_IPS:-}" 14 INFERENCE_NODE_IPS)"
  overrides+=(
    "training.placement_strategy=custom"
    "training.training_node_ips=$train_ips"
    "inference.inference_node_ips=$inference_ips"
  )
fi

if [[ -n "${TRAIN_ENV_VARS:-}" ]]; then
  overrides+=("training.train_env_vars=$TRAIN_ENV_VARS")
fi

command=(python -m torchspec.train_entry --config "$CONFIG_FILE" "${overrides[@]}" "$@")
if [[ "$dry_run" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

# Use the tested H800 environment and locally patched vLLM checkout.
source "$ROOT_DIR/h800conda.sh"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"

mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/qwen3_8_27b_dspark_vllm_128gpu_$timestamp.log"
exec > >(tee -a "$log_file") 2>&1

echo "Qwen3.8-27B DSpark: default 128 GPUs (16 training + 112 inference), 128K context"
echo "Global batch: 32; draft sync: every 100 steps (before CLI overrides)"
echo "Output: $OUTPUT_DIR; log: $log_file"
echo "Resume: ${LOAD_PATH:-none (new run)}; iteration: ${resume_step:-none}"

# Config inspection does not need a running Ray cluster.
print_config_only=false
for arg in "$@"; do
  if [[ "$arg" == "--print-config-only" ]]; then
    print_config_only=true
  fi
done
if [[ "$print_config_only" == false ]] && ! ray status >/dev/null 2>&1; then
  echo "ERROR: cannot connect to Ray. Start a cluster with sixteen 8-GPU nodes." >&2
  exit 1
fi

"${command[@]}"

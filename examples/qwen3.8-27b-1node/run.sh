#!/usr/bin/env bash
# Train DSpark on existing samples with vLLM hidden-state extraction and 128K context.
#
# Usage: bash examples/qwen3.8-27b-1node/run.sh [--dry-run] [key=value ...]
# Run on one 8-GPU node. TorchSpec starts a local Ray instance by default.
#
# Optional environment variables:
#   CUDA_VISIBLE_DEVICES  GPUs to use (default: 0,1,2,3,4,5,6,7)
#   TRAIN_DATA_PATH    JSONL dataset (default: the existing 1M-sample dataset)
#   LOAD_PATH          Complete checkpoints directory; unset starts a new run
#   CKPT_STEP          Checkpoint iteration; unset selects latest
#   OUTPUT_DIR         Defaults to a new timestamped output directory
#   CACHE_DIR          Defaults to cache/qwen3.8-27b-1node
#   LOG_DIR            Defaults to running_logs
#   TARGET_MODEL_PATH  Target checkpoint (default supplied by h800conda.sh)
#   MOONCAKE_PROTOCOL  tcp (default), or rdma when configured
#   RAY_ADDRESS        Ray address (default: local)
#
# --dry-run prints the command without contacting Ray or starting training.
# --print-config-only loads and prints the resolved config without training.

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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3.8-27b-1node_$timestamp}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/cache/qwen3.8-27b-1node}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/running_logs}"
export WANDB_NAME="${WANDB_NAME:-qwen3.8-27b-1node_$timestamp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

overrides=("output_dir=$OUTPUT_DIR" "cache_dir=$CACHE_DIR")
if [[ -n "${TRAIN_DATA_PATH:-}" ]]; then
  overrides+=("dataset.train_data_path=$TRAIN_DATA_PATH")
fi
if [[ -n "${MOONCAKE_PROTOCOL:-}" ]]; then
  overrides+=("mooncake.protocol=$MOONCAKE_PROTOCOL")
fi
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
    echo "ERROR: checkpoint iteration must be a non-negative integer." >&2
    exit 1
  fi
  resume_step="$((10#$resume_step))"
  printf -v checkpoint_name 'iter_%07d' "$resume_step"
  for component in model optimizer; do
    if [[ ! -d "$LOAD_PATH/$checkpoint_name/$component" ]]; then
      echo "ERROR: incomplete checkpoint: $LOAD_PATH/$checkpoint_name/$component" >&2
      exit 1
    fi
  done
  overrides+=("training.load_path=$LOAD_PATH" "training.ckpt_step=$resume_step")
elif [[ -n "${CKPT_STEP:-}" ]]; then
  echo "ERROR: CKPT_STEP requires LOAD_PATH." >&2
  exit 1
fi

# Keep this recipe in prefill training mode, including with CLI overrides.
launch=(
  python -m torchspec.train_entry --config "$SCRIPT_DIR/config.yaml"
  "${overrides[@]}" "$@"
  training.train_with_decode=false
  decode.target_only=false
  decode.speculative_algorithm=null
  decode.speculative_draft_model_path=null
  decode.weight_sync_enabled=false
  decode.weight_sync_interval=0
)
if [[ "$dry_run" == true ]]; then
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi

source "$ROOT_DIR/h800conda.sh"
# The local extract_hidden_states implementation is supported by Model Runner V1.
export VLLM_USE_V2_MODEL_RUNNER=0
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"

mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/qwen3.8-27b-1node_$timestamp.log"
exec > >(tee -a "$log_file") 2>&1
echo "Qwen3.8-27B: target hidden-state extraction; DSpark training on existing answers; 128K context"
echo "Default GPUs: 4 training + 4 inference (TP=4); global batch: 16"
echo "Ray: $RAY_ADDRESS"
echo "Output: $OUTPUT_DIR; log: $log_file"

"${launch[@]}"

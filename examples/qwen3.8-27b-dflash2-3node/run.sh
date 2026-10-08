#!/usr/bin/env bash
# Train DFlash2 on existing samples with vLLM hidden-state extraction and 128K context.
#
# Usage: bash examples/qwen3.8-27b-dflash2-3node/run.sh [--dry-run] [key=value ...]
# Run on the head of an existing three-node Ray cluster.
#
# Optional environment variables:
#   TRAIN_NODE_IPS      One training node IP
#   INFERENCE_NODE_IPS  Two comma-separated inference node IPs
#   TRAIN_DATA_PATH    JSONL dataset (default: the existing 1M-sample dataset)
#   LOAD_PATH          Complete checkpoints directory; unset starts a new run
#   CKPT_STEP          Checkpoint iteration; unset selects latest
#   OUTPUT_DIR         Defaults to a new timestamped output directory
#   CACHE_DIR          Defaults to cache/qwen3_8_27b-dflash2-vllm-decode-3node
#   LOG_DIR            Defaults to running_logs
#   TARGET_MODEL_PATH  Target checkpoint (default supplied by h800conda.sh)
#   MOONCAKE_PROTOCOL  rdma (default, mlx5_0 on the validated cluster), or tcp
#   RAY_ADDRESS        Existing Ray address, if not discovered locally
#
# --dry-run prints the command without contacting Ray or starting training.
# --print-config-only loads and prints the resolved config without training.
# Training logs go directly to LOG_DIR; use tail -f to follow them.

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
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3.8-27b-dflash2-3node_$timestamp}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/cache/qwen3_8_27b-dflash2-vllm-decode-3node}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/running_logs}"
export WANDB_NAME="${WANDB_NAME:-qwen3.8-27b-dflash2-3node_$timestamp}"

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

format_ip_list() {
  local raw="$1" expected="$2" name="$3" ip result="["
  local -a ips
  IFS=',' read -r -a ips <<< "$raw"
  if (( ${#ips[@]} != expected )) || [[ "$raw" == *, ]]; then
    echo "ERROR: $name must contain $expected comma-separated node IP(s)." >&2
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
  train_ips="$(format_ip_list "${TRAIN_NODE_IPS:-}" 1 TRAIN_NODE_IPS)"
  inference_ips="$(format_ip_list "${INFERENCE_NODE_IPS:-}" 2 INFERENCE_NODE_IPS)"
  overrides+=(
    "training.placement_strategy=custom"
    "training.training_node_ips=$train_ips"
    "inference.inference_node_ips=$inference_ips"
  )
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
log_file="$LOG_DIR/qwen3.8-27b-dflash2-3node_$timestamp.log"
print_config_only=false
for arg in "$@"; do
  if [[ "$arg" == "--print-config-only" ]]; then
    print_config_only=true
  fi
done
if [[ "$print_config_only" == false ]]; then
  printf 'Training logs: %s\n' "$log_file"
  printf 'Follow logs from another terminal: tail -f %q\n' "$log_file"
  # Keep a stalled terminal reader from blocking the training driver's logging.
  exec >> "$log_file" 2>&1
fi
echo "Qwen3.8-27B: target hidden-state extraction; DFlash2 training on existing answers; 128K context"
echo "Default GPUs: 8 training + 2 x 8 inference; global batch: 32"
echo "Output: $OUTPUT_DIR; log: $log_file"

if [[ "$print_config_only" == false ]] && ! ray status >/dev/null 2>&1; then
  echo "ERROR: start or join a Ray cluster with three 8-GPU nodes before launching." >&2
  exit 1
fi
"${launch[@]}"

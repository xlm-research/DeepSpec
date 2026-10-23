#!/usr/bin/env bash
# Train Loop/mHC DSpark on three H800 nodes (8 FSDP + two TP=8 targets).
#
# Usage: bash examples/qwen3.5-loop-mhc-dspark-3-node/run.sh [--dry-run|--print-config-only] [key=value ...]
# Run on the head of an existing three-node Ray cluster.
#
# Optional environment variables:
#   TRAIN_NODE_IPS      One training node IP; set together with INFERENCE_NODE_IPS
#   INFERENCE_NODE_IPS  Two comma-separated inference node IPs
#   TRAIN_DATA_PATH     JSONL dataset (default: the existing 1M-sample dataset)
#   LOOP_TARGET_MODEL_PATH  Loop/mHC checkpoint (default: ckpt-225-bf16)
#   LOAD_PATH           Complete checkpoints directory; unset starts a new run
#   CKPT_STEP           Checkpoint iteration; unset selects latest
#   OUTPUT_DIR          Defaults to a new timestamped output directory
#   CACHE_DIR           Defaults to cache/qwen3.5-loop-mhc-dspark-3-node
#   LOG_DIR             Defaults to running_logs
#   MOONCAKE_PROTOCOL   tcp (default), or rdma when configured on all nodes
#   TORCHSPEC_TRITON_CACHE_DIR  Node-local cache (default: /tmp/torchspec-triton-$UID)
#   RAY_ADDRESS         Existing Ray address (default: auto)
#   WANDB_MODE          online (default), offline, or disabled
#
# --dry-run prints the command without sourcing the environment or contacting Ray.
# --print-config-only prints the resolved config without starting training.
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
print_config_only=false
for arg in "$@"; do
  if [[ "$arg" == "--print-config-only" ]]; then
    print_config_only=true
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
# Use a recipe-specific variable: the shared shell may already select Qwen3.8.
export LOOP_TARGET_MODEL_PATH="${LOOP_TARGET_MODEL_PATH:-/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16}"
export TARGET_MODEL_PATH="$LOOP_TARGET_MODEL_PATH"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3.5-loop-mhc-dspark-3-node_$timestamp}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/cache/qwen3.5-loop-mhc-dspark-3-node}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/running_logs}"
triton_cache_dir="${TORCHSPEC_TRITON_CACHE_DIR:-/tmp/torchspec-triton-${UID}}"
export WANDB_NAME="${WANDB_NAME:-qwen3.5-loop-mhc-dspark-3-node_$timestamp}"

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
  for component in model optimizer lr_scheduler rng.pt meta.json; do
    if [[ ! -e "$LOAD_PATH/$checkpoint_name/$component" ]]; then
      echo "ERROR: incomplete checkpoint: $LOAD_PATH/$checkpoint_name/$component" >&2
      exit 1
    fi
  done
  if [[ "$(realpath -m "$OUTPUT_DIR/checkpoints")" == "$(realpath -m "$LOAD_PATH")" ]]; then
    echo "ERROR: restore into a new OUTPUT_DIR to preserve the original run." >&2
    exit 1
  fi
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

# Keep the Loop capture contract even when generic CLI overrides are supplied.
launch=(
  python -m torchspec.train_entry --config "$SCRIPT_DIR/config.yaml"
  "${overrides[@]}" "$@"
  training.train_with_decode=false
  inference.vllm.direct_hidden_state_capture=true
  inference.store_last_hidden_states=true
  inference.last_hidden_states_prenorm=false
  decode.target_only=false
  decode.speculative_algorithm=null
  decode.speculative_draft_model_path=null
  decode.weight_sync_enabled=false
  decode.weight_sync_interval=0
)
if [[ "$dry_run" == true ]]; then
  printf 'LOOP_TARGET_MODEL_PATH=%q RAY_ADDRESS=%q TORCHSPEC_TRITON_CACHE_DIR=%q TRITON_CACHE_DIR=%q VLLM_USE_V2_MODEL_RUNNER=1 ' "$LOOP_TARGET_MODEL_PATH" "$RAY_ADDRESS" "$triton_cache_dir" "$triton_cache_dir"
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi

source "$ROOT_DIR/h800conda.sh"
# Override the shared cache inherited from Conda and existing Ray nodes.
# get_torchspec_env_vars forwards this path to each actor before Triton imports.
export TORCHSPEC_TRITON_CACHE_DIR="$triton_cache_dir"
export TRITON_CACHE_DIR="$triton_cache_dir"
export VLLM_USE_V2_MODEL_RUNNER=1
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"

if [[ "$print_config_only" == false ]]; then
  if [[ "$RAY_ADDRESS" == local ]] || ! ray status --address "$RAY_ADDRESS" >/dev/null 2>&1; then
    echo "ERROR: start or join an existing Ray cluster with three 8-GPU nodes before launching." >&2
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  log_file="$LOG_DIR/qwen3.5-loop-mhc-dspark-3-node_$timestamp.log"
  printf 'Training logs: %s\n' "$log_file"
  printf 'Follow logs from another terminal: tail -f %q\n' "$log_file"
  exec >> "$log_file" 2>&1
fi
echo "Loop/mHC: V2 direct target capture; DSpark training on existing answers"
echo "Defaults: 8 FSDP + 2 x TP=8 target GPUs; 128K; global batch 32; last-turn supervision"
echo "Output: $OUTPUT_DIR"

"${launch[@]}"

#!/usr/bin/env bash
# Launch Qwen3.8-27B DSpark train-with-decode on three 8-GPU nodes.
#
# Topology (default):
#   node 0: 8 GPUs for FSDP draft training
#   nodes 1-2: one independent 8-GPU vLLM DSpark engine each (TP=8, PP=1)
# TP=8 divides both the target's 24 attention heads and the draft's 40 heads.
#
# Start a Ray head on node 0 and two Ray workers before running this script on
# node 0.  The launcher uses the existing cluster; it does not start or stop
# Ray so that a failed training process does not tear down other jobs.
#
# Usage:
#   bash examples/train-with-decode/run_vllm_qwen3_8_27b_dspark_3node.sh \
#     [config.yaml] [OmegaConf overrides ...]
#
# Useful environment variables:
#   CONFIG_FILE             Config path (default: the bundled 3-node config)
#   TARGET_MODEL_PATH       Qwen3.8-27B checkpoint path (set by h800conda.sh)
#   TRAIN_NODE_IPS          Optional comma-separated training node IPs
#   INFERENCE_NODE_IPS      Optional comma-separated inference node IPs
#   TRAIN_GPUS              Training GPUs per node (default: 8)
#   TRAIN_NODES             Training nodes (default: 1)
#   INFERENCE_NODES         Single-node vLLM replicas (default: 2)
#   INFERENCE_GPUS_PER_NODE GPUs per vLLM node (default: 8)
#   INFERENCE_GPUS          Total vLLM GPUs (default: 16)
#   DSPARK_BLOCK_SIZE       DSpark block/draft length (default: 7)
#   MOONCAKE_PROTOCOL       tcp or rdma (default: tcp)
#   LOG_DIR                 Log directory (default: running_logs)
#
# The optional node-IP variables enable deterministic placement.  If omitted,
# TorchSpec's training_first placement assigns the first eight sorted Ray
# bundles to training.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# h800conda.sh activates the tested environment and prepends the local vLLM
# checkout to PYTHONPATH.
source "$ROOT_DIR/h800conda.sh"

export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"

TRAIN_GPUS="${TRAIN_GPUS:-8}"
TRAIN_NODES="${TRAIN_NODES:-1}"
INFERENCE_NODES="${INFERENCE_NODES:-2}"
INFERENCE_GPUS_PER_NODE="${INFERENCE_GPUS_PER_NODE:-8}"
INFERENCE_GPUS="${INFERENCE_GPUS:-$((INFERENCE_NODES * INFERENCE_GPUS_PER_NODE))}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/train_with_decode/vllm_qwen3_8_27b_dspark_3node.yaml}"
TRAIN_DATA_PATH="/mnt/afs_share/lezewei/TorchSpec/mydata/sensenova-flash-lite-v42-text-1000000.jsonl"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/running_logs}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,55p' "$0"
  exit 0
fi

# Accept a config as the first positional argument while keeping ordinary
# key=value arguments available as direct OmegaConf overrides.
if [[ $# -gt 0 && "$1" != *=* ]]; then
  candidate="$1"
  if [[ -f "$candidate" ]]; then
    CONFIG_FILE="$(cd -- "$(dirname -- "$candidate")" && pwd)/$(basename -- "$candidate")"
    shift
  elif [[ -f "$ROOT_DIR/$candidate" ]]; then
    CONFIG_FILE="$ROOT_DIR/$candidate"
    shift
  fi
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: config file does not exist: $CONFIG_FILE" >&2
  exit 1
fi

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $name must be a positive integer, got '$value'" >&2
    exit 1
  fi
}

require_positive_int TRAIN_GPUS "$TRAIN_GPUS"
require_positive_int TRAIN_NODES "$TRAIN_NODES"
require_positive_int INFERENCE_NODES "$INFERENCE_NODES"
require_positive_int INFERENCE_GPUS_PER_NODE "$INFERENCE_GPUS_PER_NODE"
require_positive_int INFERENCE_GPUS "$INFERENCE_GPUS"
require_positive_int DSPARK_BLOCK_SIZE "$DSPARK_BLOCK_SIZE"

expected_inference_gpus=$((INFERENCE_NODES * INFERENCE_GPUS_PER_NODE))
if (( INFERENCE_GPUS != expected_inference_gpus )); then
  echo "ERROR: INFERENCE_GPUS=$INFERENCE_GPUS must equal " \
    "INFERENCE_NODES*INFERENCE_GPUS_PER_NODE=$expected_inference_gpus" >&2
  exit 1
fi

format_ip_list() {
  local raw="$1"
  local -a ips
  local ip
  local result="["
  IFS=',' read -r -a ips <<< "$raw"
  if (( ${#ips[@]} == 0 )); then
    echo "ERROR: empty node IP list" >&2
    exit 1
  fi
  for ip in "${ips[@]}"; do
    if [[ -z "$ip" ]]; then
      echo "ERROR: node IP list contains an empty entry: '$raw'" >&2
      exit 1
    fi
    [[ "$result" == "[" ]] || result+=","
    result+="\"$ip\""
  done
  result+="]"
  printf '%s' "$result"
}

topology_overrides=(
  "training.training_num_nodes=$TRAIN_NODES"
  "training.training_num_gpus_per_node=$TRAIN_GPUS"
  "training.train_with_decode=true"
  "training.dflash_block_size=$DSPARK_BLOCK_SIZE"
  "dataset.min_loss_tokens=$((2 * DSPARK_BLOCK_SIZE))"
  "inference.inference_engine_type=vllm"
  "inference.inference_num_gpus=$INFERENCE_GPUS"
  "inference.inference_num_gpus_per_engine=$INFERENCE_GPUS_PER_NODE"
  "inference.inference_num_gpus_per_node=$INFERENCE_GPUS_PER_NODE"
  "inference.vllm.nnodes=1"
  "inference.vllm.tp_size=$INFERENCE_GPUS_PER_NODE"
  "inference.vllm.pp_size=1"
  "decode.speculative_algorithm=dspark"
  "decode.speculative_num_draft_tokens=$DSPARK_BLOCK_SIZE"
  "mooncake.protocol=$MOONCAKE_PROTOCOL"
)

if [[ -n "${TRAIN_NODE_IPS:-}" || -n "${INFERENCE_NODE_IPS:-}" ]]; then
  if [[ -z "${TRAIN_NODE_IPS:-}" || -z "${INFERENCE_NODE_IPS:-}" ]]; then
    echo "ERROR: set both TRAIN_NODE_IPS and INFERENCE_NODE_IPS for custom placement" >&2
    exit 1
  fi
  topology_overrides+=(
    "training.placement_strategy=custom"
    "training.training_node_ips=$(format_ip_list "$TRAIN_NODE_IPS")"
    "inference.inference_node_ips=$(format_ip_list "$INFERENCE_NODE_IPS")"
  )
fi

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
log_file="$LOG_DIR/qwen3_8_27b_dspark_vllm_3node_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

echo "============================================================"
echo "Qwen3.8-27B DSpark vLLM train-with-decode"
echo "============================================================"
echo "Config:             $CONFIG_FILE"
echo "Training data:      $TRAIN_DATA_PATH"
echo "Training:           $TRAIN_NODES node(s) x $TRAIN_GPUS GPU(s)"
echo "Inference:          $INFERENCE_NODES replica(s) x $INFERENCE_GPUS_PER_NODE GPU(s), TP=$INFERENCE_GPUS_PER_NODE each, PP=1"
echo "DSpark block size:  $DSPARK_BLOCK_SIZE"
echo "Mooncake protocol:  $MOONCAKE_PROTOCOL"
echo "Log:                $log_file"
echo "============================================================"

if ! ray status >/dev/null 2>&1; then
  echo "ERROR: cannot connect to a Ray cluster from this node." >&2
  echo "Start one Ray head on the training node and two 8-GPU workers, then rerun." >&2
  echo "See docs/ray.md and examples/kimi-k25-3node-h100/setup_ray_cluster.sh." >&2
  exit 1
fi

python -m torchspec.train_entry \
  --config "$CONFIG_FILE" \
  "${topology_overrides[@]}" \
  "$@"

echo "============================================================"
echo "Training completed"
echo "============================================================"

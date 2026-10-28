#!/usr/bin/env bash
# Train Loop/mHC DSpark on one H800 node (4 target + 4 training GPUs).
# Usage: bash examples/qwen3.5-loop-mhc-dspark/run.sh [--dry-run|--print-config-only] [key=value ...]
# Environment: OUTPUT_DIR, CACHE_DIR, TRAIN_DATA_PATH, TARGET_MODEL_PATH,
# LOAD_PATH (checkpoints directory), CKPT_STEP (default: latest), LOG_DIR.

set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  sed -n '2,/^$/p' "${BASH_SOURCE[0]}"
  exit 0
fi
dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
  dry_run=true
  shift
fi
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$example_dir/../.." && pwd)
cd "$repo_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-$repo_dir/outputs/qwen3.5-loop-mhc-dspark_$timestamp}"
overrides=("output_dir=$OUTPUT_DIR" "cache_dir=${CACHE_DIR:-$repo_dir/cache/qwen3.5-loop-mhc-dspark}")
if [[ -n "${TRAIN_DATA_PATH:-}" ]]; then
  overrides+=("dataset.train_data_path=$TRAIN_DATA_PATH")
fi
if [[ -n "${LOAD_PATH:-}" ]]; then
  resume_step="${CKPT_STEP:-$(cat "$LOAD_PATH/latest_checkpointed_iteration.txt")}"
  if [[ ! "$resume_step" =~ ^[0-9]+$ ]]; then
    echo "Invalid checkpoint iteration: $resume_step" >&2
    exit 1
  fi
  resume_step=$((10#$resume_step))
  printf -v checkpoint_name 'iter_%07d' "$resume_step"
  for component in model optimizer lr_scheduler rng.pt meta.json; do
    if [[ ! -e "$LOAD_PATH/$checkpoint_name/$component" ]]; then
      echo "Incomplete checkpoint: $LOAD_PATH/$checkpoint_name/$component" >&2
      exit 1
    fi
  done
  if [[ "$(realpath -m "$OUTPUT_DIR/checkpoints")" == "$(realpath -m "$LOAD_PATH")" ]]; then
    echo "Restore into a new OUTPUT_DIR to preserve the original run." >&2
    exit 1
  fi
  overrides+=("training.load_path=$LOAD_PATH" "training.ckpt_step=$resume_step")
elif [[ -n "${CKPT_STEP:-}" ]]; then
  echo 'CKPT_STEP requires LOAD_PATH.' >&2
  exit 1
fi
launch=(
  python -m torchspec.train_entry --config "$example_dir/config.yaml"
  "${overrides[@]}" "$@"
)
if [[ "$dry_run" == true ]]; then
  printf 'VLLM_USE_V2_MODEL_RUNNER=1 '
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi
source "$repo_dir/h800conda-tmp.sh"
export VLLM_USE_V2_MODEL_RUNNER=1
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"
log_dir="${LOG_DIR:-$repo_dir/running_logs}"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/qwen3.5-loop-mhc-dspark_$timestamp.log") 2>&1
echo "Loop/mHC: TP=4 target + 4 FSDP ranks; five DSpark layers; 128K; last-turn supervision"
echo "Output: $OUTPUT_DIR"
"${launch[@]}"

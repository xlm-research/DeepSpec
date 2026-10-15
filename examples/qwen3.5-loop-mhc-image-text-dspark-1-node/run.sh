#!/usr/bin/env bash
# Fresh Loop/mHC image-text DSpark training: all 327 sources, one complete pass.
# Usage: run.sh [--dry-run|--prepare-only|--preflight-only] [key=value ...]
# Environment: OUTPUT_DIR, CACHE_DIR, LOG_DIR, TARGET_MODEL_PATH, CUDA_VISIBLE_DEVICES.

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
if [[ -n "${LOAD_PATH:-}" || -n "${CKPT_STEP:-}" ]]; then
  echo 'This full-data recipe starts fresh; checkpoint restore is not supported.' >&2
  exit 1
fi
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$example_dir/../.." && pwd)
cd "$repo_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
output_dir="${OUTPUT_DIR:-$repo_dir/outputs/qwen3.5-loop-mhc-image-text-dspark-1-node_$timestamp}"
launch=(
  uv run --no-project --python "$repo_dir/.venv/bin/python" "$repo_dir/.venv/bin/python"
  "$repo_dir/examples/qwen3.8-27b-image-text-1node/full_train.py"
  --config "$example_dir/config.yaml"
  --legacy-text-config "$repo_dir/examples/qwen3.5-loop-mhc-dspark/config.yaml"
  --output-dir "$output_dir"
  --cache-dir "${CACHE_DIR:-$repo_dir/cache/qwen3.5-loop-mhc-image-text-dspark-1-node}"
  "$@"
)
if [[ "$dry_run" == true ]]; then
  printf 'VLLM_USE_V2_MODEL_RUNNER=1 '
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi
source "$repo_dir/h800conda.sh"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_QWEN3_5_MHC_BACKEND="${VLLM_QWEN3_5_MHC_BACKEND:-triton}"
log_dir="${LOG_DIR:-$output_dir/launch_logs}"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/launch_$timestamp.log") 2>&1
"${launch[@]}"

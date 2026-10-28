#!/usr/bin/env bash
# Fresh Loop/mHC image-text DSpark training: all 327 sources, one complete pass.
# Usage: run.sh [--dry-run|--prepare-only|--preflight-only|--benchmark-config PATH] [key=value ...]
# Environment: OUTPUT_DIR, CACHE_DIR, LOG_DIR, TARGET_MODEL_PATH, CUDA_VISIBLE_DEVICES,
# TORCHSPEC_CONDA_ENV (default: /tmp/deepspec_vllm_torchtitan_envs).
# ENV_MODE: tmp (default) or afs. --benchmark-config runs training directly.

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
benchmark_config=""
if [[ "${1:-}" == --benchmark-config ]]; then
  benchmark_config="${2:?--benchmark-config requires a config path}"
  shift 2
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
case "${ENV_MODE:-tmp}" in
  tmp)
    environment_script="$repo_dir/h800conda-tmp.sh"
    pipeline_python="${TORCHSPEC_CONDA_ENV:-/tmp/deepspec_vllm_torchtitan_envs}/bin/python"
    ;;
  afs)
    environment_script="$repo_dir/h800conda.sh"
    pipeline_python=/mnt/afs_share/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python
    ;;
  *) echo 'ENV_MODE must be tmp or afs' >&2; exit 1 ;;
esac
if [[ "$dry_run" != true ]]; then
  environment_started=$EPOCHREALTIME
  source "$environment_script"
  printf 'ENVIRONMENT_READY mode=%s start=%s end=%s python=%s\n' "${ENV_MODE:-tmp}" "$environment_started" "$EPOCHREALTIME" "$PIPELINE_PYTHON"
  pipeline_python="$PIPELINE_PYTHON"
fi
launch=(
  "$pipeline_python"
  "$repo_dir/examples/qwen3.8-27b-image-text-1node/full_train.py"
  --config "$example_dir/config.yaml"
  --legacy-text-config "$repo_dir/examples/qwen3.5-loop-mhc-dspark/config.yaml"
  --output-dir "$output_dir"
  --cache-dir "${CACHE_DIR:-$repo_dir/cache/qwen3.5-loop-mhc-image-text-dspark-1-node}"
  "$@"
)
if [[ -n "$benchmark_config" ]]; then
  # Conda activation can override these; isolate compilation caches per trial.
  export TRITON_CACHE_DIR="$output_dir/triton"
  export TORCHINDUCTOR_CACHE_DIR="$output_dir/inductor"
  launch=("$pipeline_python" -m torchspec.train_entry --config "$benchmark_config" "$@")
fi
if [[ "$dry_run" == true ]]; then
  printf 'VLLM_USE_V2_MODEL_RUNNER=1 VLLM_QWEN3_5_MHC_BACKEND=%q VLLM_QWEN3_5_MHC_FUSION=%q ' \
    "${VLLM_QWEN3_5_MHC_BACKEND:-triton}" "${VLLM_QWEN3_5_MHC_FUSION:-all}"
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_QWEN3_5_MHC_BACKEND="${VLLM_QWEN3_5_MHC_BACKEND:-triton}"
export VLLM_QWEN3_5_MHC_FUSION="${VLLM_QWEN3_5_MHC_FUSION:-all}"
log_dir="${LOG_DIR:-$output_dir/launch_logs}"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/launch_$timestamp.log") 2>&1
"${launch[@]}"

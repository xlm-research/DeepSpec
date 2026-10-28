#!/usr/bin/env bash
# Recover the DFlash2 prefill-extraction run from 2026-10-01 using host RDMA.
# Three-node CPU PUT/GET checks passed on mlx5_0 for .218, .187, and .211.
# Run on the training/head node (10.120.0.211), with the existing Ray cluster.
#
# First validate 10 optimizer steps:
#   bash examples/run_qwen3_8_27b_dflash2_rdma_recovery.sh \
#     training.num_train_steps=10 training.save_interval=10
# Full run from initial weights (new output directory):
#   bash examples/run_qwen3_8_27b_dflash2_rdma_recovery.sh
# To resume, pass training.load_path=<output>/checkpoints and
# training.continual_training=false (restore optimizer). Leave ckpt_step unset
# to read the checkpoint tracker: this checkout saves step 10 as iter_0000011.
# Set DRY_RUN=1 to validate and print the command without launching training.
# BASE_CONFIG, OUTPUT_DIR and LOG_DIR may be overridden explicitly.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/h800conda-tmp.sh"
# The prefill extractor is implemented by the V1 runner in this vLLM checkout.
# Override h800conda-tmp.sh's V2 default before TorchSpec forwards env vars to Ray.
export VLLM_USE_V2_MODEL_RUNNER=0

BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/outputs/qwen3_8_27b-dflash2-128k_20261001_215811/config.yaml}"
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Missing source run config: $BASE_CONFIG" >&2
    exit 1
fi

run_stamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3_8_27b-dflash2-128k-rdma_$run_stamp}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/running_logs}"

# 128K x 5120 x bf16 = 1.25 GiB per layer. A 2 GiB buffer fits a
# single layer; the connector splits larger groups across buffer submissions.
# Keep the sample pool >= 8: each trainer dispatch needs one sample per DP rank.
# 256 GiB per inference node provides 512 GiB total store capacity; do not
# shrink this to a small smoke-test segment while retaining 128K samples.
overrides=(
    "mooncake.protocol=rdma"
    "mooncake.device_name=mlx5_0"
    "mooncake.enable_gpu_direct=false"
    "mooncake.global_segment_size=256GB"
    "mooncake.local_buffer_size=16GB"
    "mooncake.host_buffer_size=2GB"
    "mooncake.async_put_pool_size=1"
    "inference.inference_batch_size=1"
    "inference.inference_buffer_threshold=8"
    "inference.inference_fetch_batch=2"
    "inference.max_sample_pool_size=8"
    "training.prefetch_depth=1"
    # Keep the original 971449/32-step LR schedule in the short acceptance run.
    "training.lr_total_steps=30357"
    "output_dir=$OUTPUT_DIR"
)

# Validate with the same loader as train_entry, without writing a snapshot.
python - "$BASE_CONFIG" "${overrides[@]}" "$@" <<'PY'
import sys
from torchspec.config import load_config
config = load_config(sys.argv[1], sys.argv[2:])
print(f"Validated config: protocol={config.mooncake.protocol}, "
      f"device={config.mooncake.device_name}, "
      f"max_seq_length={config.training.max_seq_length}, "
      f"output_dir={config.output_dir}")
PY

command=(python -m torchspec.train_entry --config "$BASE_CONFIG" "${overrides[@]}" "$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/qwen3_8_27b-dflash2-rdma_${run_stamp}.log"
echo "Log: $log_file"
echo "Output: $OUTPUT_DIR"
"${command[@]}" 2>&1 | tee "$log_file"

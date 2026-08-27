#!/usr/bin/env bash
set -euo pipefail

# PyTorch Profiler launcher for DeepSeek-V4 DSpark/DFlash2 training. DSpark
# performs target inference immediately before each micro-batch; DFlash2 uses
# its offline target cache. The selected validated multi-node launcher owns the
# corresponding target lifecycle.
#
# Default behavior:
#   1. Run two optimizer steps without writing checkpoints.
#   2. Use the last micro-step of step 1 as profiler warmup.
#   3. Capture every micro-step plus the optimizer boundary of step 2.
#   4. Trace global rank 0 only.  Set PROFILE_RANKS=all for a communication
#      trace from every rank (this can generate a very large amount of data).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

PROFILE_METHOD=${PROFILE_METHOD:-dspark}
case "${PROFILE_METHOD}" in
    dspark)
        TRAIN_SCRIPT=${SCRIPT_DIR}/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh
        ;;
    dflash2)
        TRAIN_SCRIPT=${SCRIPT_DIR}/train_deepseek_v4_flash_dflash2_fsdp2_multinode_128.sh
        ;;
    *)
        echo "PROFILE_METHOD must be dspark or dflash2; got ${PROFILE_METHOD}." >&2
        exit 1
        ;;
esac

export PROFILE_ENABLED=true
export PROFILE_RANKS=${PROFILE_RANKS:-[0]}
export MAX_TRAIN_STEPS=${PROFILE_TRAIN_STEPS:-2}
export SAVE_CHECKPOINTS=false
export TORCHRUN_PER_RANK_LOGS=${TORCHRUN_PER_RANK_LOGS:-true}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/output/deepseek_v4_flash_${PROFILE_METHOD}_torch_profile}
export PROFILE_TRACE_DIR=${PROFILE_TRACE_DIR:-${OUTPUT_ROOT}/torch_profile}

echo "Launching DeepSeek-V4 ${PROFILE_METHOD} PyTorch profile:"
echo "  training script=${TRAIN_SCRIPT}"
echo "  optimizer steps=${MAX_TRAIN_STEPS}"
echo "  profiled ranks=${PROFILE_RANKS}"
echo "  trace directory=${PROFILE_TRACE_DIR}"
echo "  checkpoints disabled"

exec "${TRAIN_SCRIPT}"

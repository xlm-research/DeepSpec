#!/usr/bin/env bash
set -euo pipefail

# Online DeepSeek-V4 smoke test using 18 long (<128K) coding samples.
# Each sample runs target forward and immediate draft training without a disk
# cache. It inherits the full launcher's CP/EP/TP/FSDP sizes.

ROOT_DIR=${ROOT_DIR:-/mnt/afs_share/wzj/deepspec1}

export ROOT_DIR
export CP_SIZE=${CP_SIZE:-8}
export TARGET_EP_SIZE=${TARGET_EP_SIZE:-2}
export TARGET_TP_SIZE=${TARGET_TP_SIZE:-2}
export TARGET_FSDP_SIZE=${TARGET_FSDP_SIZE:-9}
export DATA_PATH=${DATA_PATH:-${ROOT_DIR}/train_data/deepseek_v4_long_under_128k.jsonl}
export PROJECT_NAME=${PROJECT_NAME:-deepspec_128k_smoke}
export EXP_NAME=${EXP_NAME:-dspark_deepseek_v4_long_smoke}
export RUN_DIR=${RUN_DIR:-${ROOT_DIR}/output/deepseek_v4_128k/long_smoke_cp_ep_tp}
export NUM_TRAIN_EPOCHS=1

# The selected dataset has 18 samples. The full launcher verifies that this
# batch is divisible by the effective data-replica count of the chosen layout.
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-18}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1}
export CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-1000000}

if [[ ! -s "${DATA_PATH}" ]]; then
    echo "Smoke dataset not found or empty: ${DATA_PATH}" >&2
    echo "Generate it with scripts/data/select_long_openai_samples.py first." >&2
    exit 1
fi
if [[ ! -f "${ROOT_DIR}/deepseek_v4_full.sh" ]]; then
    echo "Base launcher not found: ${ROOT_DIR}/deepseek_v4_full.sh" >&2
    exit 1
fi

exec bash "${ROOT_DIR}/deepseek_v4_full.sh"

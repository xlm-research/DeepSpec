#!/usr/bin/env bash
# B300: conda and real 4K TP4 + TP4 training; override --stage when needed.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../../../env.sh"
exec bash "${SCRIPT_DIR}/debug_single_node.sh" \
    --stage 4k \
    --model "${TARGET_MODEL_PATH}" \
    --pool-gib 64 --window 8 --producer-batch-size 4 --writer-inflight 2 \
    --steps 3 --timeout-seconds 3600 --allocation-timeout-seconds 600 "$@"

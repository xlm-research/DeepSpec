#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
source "${REPO_ROOT}/h800conda.sh"
set -u
exec bash "${SCRIPT_DIR}/debug_single_node.sh" \
    --model "${TARGET_MODEL_PATH}" \
    --pool-gib 64 --window 8 --producer-batch-size 4 --steps 3 \
    "$@"

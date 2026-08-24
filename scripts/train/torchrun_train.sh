#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

nproc_per_node="${NPROC_PER_NODE:-8}"
config_path="${CONFIG_PATH:-config/distributed/fsdp2_8gpu.py}"
target_cache_dir="${TARGET_CACHE_DIR:?Set TARGET_CACHE_DIR to a completed target cache}"

torchrun \
    --standalone \
    --nproc-per-node="${nproc_per_node}" \
    train.py \
    --config "${config_path}" \
    --opts "data.target_cache_path=${target_cache_dir}"

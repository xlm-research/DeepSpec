#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
if [[ -z "${DEBUG_PYTHON:-}" ]]; then
    if [[ -x /tmp/deepspec_vllm_torchtitan_envs/bin/python ]]; then
        DEBUG_PYTHON=/tmp/deepspec_vllm_torchtitan_envs/bin/python
    else
        DEBUG_PYTHON=/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python
    fi
fi
if [[ ! -x "${DEBUG_PYTHON}" ]]; then
    printf 'Python not found: %s. Set DEBUG_PYTHON to the training environment.\n' "${DEBUG_PYTHON}" >&2
    exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/torchtitan:${REPO_ROOT}/vllm"
CUDA_RUNTIME_LIB=$("${DEBUG_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
debug_args=()
if [[ "${DRY_RUN:-false}" == true ]]; then
    debug_args+=(--dry-run)
fi
exec "${DEBUG_PYTHON}" -u "${SCRIPT_DIR}/debug_single_node.py" "${debug_args[@]}" "$@"

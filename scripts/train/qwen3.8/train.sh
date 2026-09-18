#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PIPELINE_PYTHON=/tmp/deepspec_vllm_torchtitan_envs/bin/python

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/torchtitan:${REPO_ROOT}/vllm"

# Use a fresh directory for every invocation; --output can override it.
DEFAULT_OUTPUT="${REPO_ROOT}/outputs/qwen3.8_ray_mooncake_vllm_torchtitan_$(date +%Y%m%d_%H%M%S)_$$"
launch_command=(
    "${PIPELINE_PYTHON}" -u -m deepspec.pipeline.run
    --model /mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B
    --source "${REPO_ROOT}/outputs/dspark_torchtitan_orchestration_20260914/128k-source.jsonl"
    --output "${DEFAULT_OUTPUT}"
    --context-length 131072
    --steps 3
    --window 8
    --pool-gib 64
    --protocol tcp
    --receive-device cpu
    --timeout-seconds 1800
    # argparse uses the last value when a scalar option appears more than once.
    "$@"
)

printf 'Working directory: %s\n' "${REPO_ROOT}"
printf 'Command:'
printf ' %q' "${launch_command[@]}"
printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exit 0
fi

if [[ ! -x "${PIPELINE_PYTHON}" ]]; then
    printf 'Python executable not found: %s\n' "${PIPELINE_PYTHON}" >&2
    exit 1
fi

# Mooncake links against CUDA 12 even when PyTorch uses CUDA 13.
CUDA_RUNTIME_LIB=$("${PIPELINE_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${PIPELINE_PYTHON}" -c 'from mooncake.store import MooncakeDistributedStore, ReplicateConfig'

exec "${launch_command[@]}"

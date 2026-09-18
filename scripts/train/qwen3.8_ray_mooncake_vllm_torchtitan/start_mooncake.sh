#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash start_mooncake.sh [MOONCAKE_MASTER_FLAGS...]

Start a standalone Mooncake Master in foreground; Ctrl-C stops it.
Environment:
  MOONCAKE_RPC_ADDRESS    RPC bind address (default: 0.0.0.0)
  MOONCAKE_RPC_PORT       RPC port (default: 50051)
  MOONCAKE_METRICS_PORT   HTTP metrics port (default: 9003)
  MOONCAKE_KV_LEASE_TTL   Feature lease duration (default: 300s)
  DRY_RUN                true prints the command without starting a process

Additional flags are passed directly to the installed mooncake_master.
Defaults match training: disk offload and disk eviction are disabled.
Python is fixed at /tmp/deepspec_vllm_torchtitan_envs/bin/python.

This starts the metadata/control service, not the feature memory pool.
Training entry points automatically manage their own Master and FeatureBuffer
pool. For two-node runs, the pool is on consumer node B. They do not reuse this
standalone Master; no separate Mooncake startup is needed for training.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi
if [[ "${DRY_RUN:-false}" != true && "${DRY_RUN:-false}" != false ]]; then
    printf 'DRY_RUN must be true or false.\n' >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PIPELINE_PYTHON=/tmp/deepspec_vllm_torchtitan_envs/bin/python
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/torchtitan:${REPO_ROOT}/vllm"

if [[ ! -x "${PIPELINE_PYTHON}" ]]; then
    printf 'Required Python environment is missing: %s\n' "${PIPELINE_PYTHON}" >&2
    exit 1
fi
# Use the Mooncake binary and CUDA 12 runtime from the same fixed environment.
CUDA_RUNTIME_LIB=$("${PIPELINE_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
MOONCAKE_MASTER=$("${PIPELINE_PYTHON}" -c 'from pathlib import Path; import mooncake; print(Path(mooncake.__file__).parent / "mooncake_master")')
if [[ ! -x "${MOONCAKE_MASTER}" ]]; then
    printf 'Mooncake Master executable is missing: %s\n' "${MOONCAKE_MASTER}" >&2
    exit 1
fi

launch_command=(
    "${MOONCAKE_MASTER}"
    "--rpc_address=${MOONCAKE_RPC_ADDRESS:-0.0.0.0}"
    "--rpc_port=${MOONCAKE_RPC_PORT:-50051}"
    "--metrics_port=${MOONCAKE_METRICS_PORT:-9003}"
    "--default_kv_lease_ttl=${MOONCAKE_KV_LEASE_TTL:-300s}"
    --enable_offload=false
    --enable_disk_eviction=false
    "$@"
)
printf 'Standalone Mooncake Master (foreground; no feature pool)\nCommand:'
printf ' %q' "${launch_command[@]}"
printf '\n'
if [[ "${DRY_RUN:-false}" == true ]]; then
    exit 0
fi
exec "${launch_command[@]}"

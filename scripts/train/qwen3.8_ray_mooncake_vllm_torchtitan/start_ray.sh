#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash start_ray.sh head
       bash start_ray.sh worker HEAD_IP:PORT

Run head on producer node A; run worker on consumer node B.
GPU placement is enforced by train_multinode.sh with distinct
PRODUCER_NODE and CONSUMER_NODE, not by the Ray head/worker role.

Environment:
  RAY_NODE_IP                 This node's Ray IP (default: auto-detect)
  RAY_HEAD_PORT               Head port (default: 26379)
  RAY_NUM_GPUS                GPUs advertised on this node (default: 8)
  RAY_NUM_CPUS                CPUs advertised on this node (default: 24)
  RAY_OBJECT_STORE_MEMORY     Ray object store bytes (default: 134217728)
  PIPELINE_RAY_TEMP_DIR       Head temp directory (default: /tmp/dsray-...)
  PIPELINE_RAY_BLOCK          true keeps either role in foreground (default: false)
  DRY_RUN                    true prints the command without starting Ray

Python is fixed at /tmp/deepspec_vllm_torchtitan_envs/bin/python.
Ray's object store is separate from the Mooncake feature pool on node B.
EOF
}

ROLE=${1:-}
case "${ROLE}" in
    -h|--help) usage; exit 0 ;;
    head) [[ $# -eq 1 ]] || { usage >&2; exit 2; } ;;
    worker) [[ $# -eq 2 && -n "${2}" ]] || { usage >&2; exit 2; } ;;
    *) usage >&2; exit 2 ;;
esac
for value in "${PIPELINE_RAY_BLOCK:-false}" "${DRY_RUN:-false}"; do
    if [[ "${value}" != true && "${value}" != false ]]; then
        printf 'PIPELINE_RAY_BLOCK and DRY_RUN must be true or false.\n' >&2
        exit 2
    fi
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PIPELINE_PYTHON=/tmp/deepspec_vllm_torchtitan_envs/bin/python
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/torchtitan:${REPO_ROOT}/vllm"

if [[ ! -x "${PIPELINE_PYTHON}" ]]; then
    printf 'Required Python environment is missing: %s\n' "${PIPELINE_PYTHON}" >&2
    exit 1
fi
# Mooncake needs CUDA 12's runtime even when PyTorch uses CUDA 13.
CUDA_RUNTIME_LIB=$("${PIPELINE_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cuda_runtime/lib")')
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

launch_command=(
    "${PIPELINE_PYTHON}" -m ray.scripts.scripts start
    --num-gpus "${RAY_NUM_GPUS:-8}" --num-cpus "${RAY_NUM_CPUS:-24}"
    --object-store-memory "${RAY_OBJECT_STORE_MEMORY:-134217728}"
    --disable-usage-stats
)
if [[ "${ROLE}" == head ]]; then
    NODE_IP=${RAY_NODE_IP:-$("${PIPELINE_PYTHON}" -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')}
    HEAD_ADDRESS="${NODE_IP}:${RAY_HEAD_PORT:-26379}"
    # Keep UNIX socket paths below Ray's length limit.
    RAY_TEMP_DIR=${PIPELINE_RAY_TEMP_DIR:-/tmp/dsray-$(date +%m%d_%H%M%S)-$$}
    launch_command+=(
        --head --node-ip-address "${NODE_IP}" --port "${RAY_HEAD_PORT:-26379}"
        --include-dashboard=false --temp-dir "${RAY_TEMP_DIR}"
    )
    printf 'Head address: %s\nRay logs: %s/session_latest/logs\n' "${HEAD_ADDRESS}" "${RAY_TEMP_DIR}"
    printf 'On node B: bash %q worker %q\n' "${SCRIPT_DIR}/start_ray.sh" "${HEAD_ADDRESS}"
else
    HEAD_ADDRESS=${2}
    NODE_IP=${RAY_NODE_IP:-$("${PIPELINE_PYTHON}" - "${HEAD_ADDRESS}" <<'PY'
import socket
import sys

host, port = sys.argv[1].rsplit(':', 1)
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.connect((host, int(port)))
    print(sock.getsockname()[0])
PY
)}
    launch_command+=(--address "${HEAD_ADDRESS}" --node-ip-address "${NODE_IP}")
    printf 'Worker node: %s\nHead address: %s\n' "${NODE_IP}" "${HEAD_ADDRESS}"
fi
if [[ "${PIPELINE_RAY_BLOCK:-false}" == true ]]; then
    launch_command+=(--block)
fi
printf 'Command:'
printf ' %q' "${launch_command[@]}"
printf '\n'
if [[ "${DRY_RUN:-false}" == true ]]; then
    exit 0
fi

"${PIPELINE_PYTHON}" - <<'PY'
import importlib.metadata
import sys
import torch
import ray
import mooncake.store

print('Python:', sys.executable, flush=True)
for package in ('torch', 'vllm', 'ray', 'mooncake-transfer-engine', 'transformers'):
    print(f'{package}: {importlib.metadata.version(package)}', flush=True)
PY

exec "${launch_command[@]}"

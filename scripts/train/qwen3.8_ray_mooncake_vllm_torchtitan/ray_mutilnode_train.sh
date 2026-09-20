#!/usr/bin/env bash
# Start Ray on each container, then run the existing two-node pipeline.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
RAY_SCRIPT="${SCRIPT_DIR}/start_ray.sh"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_multinode.sh"
PIPELINE_PYTHON=${PIPELINE_PYTHON:-/tmp/deepspec_vllm_torchtitan_envs/bin/python}

RAY_HEAD_PORT=${RAY_HEAD_PORT:-26379}
RAY_NUM_GPUS=${RAY_NUM_GPUS:-8}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-24}
RAY_EXPECTED_NODES=${RAY_EXPECTED_NODES:-2}
RAY_EXPECTED_GPUS=${RAY_EXPECTED_GPUS:-$((10#$RAY_EXPECTED_NODES * 10#$RAY_NUM_GPUS))}
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-900}
RAY_CONNECT_TIMEOUT=${RAY_CONNECT_TIMEOUT:-10}
RAY_POLL_INTERVAL=${RAY_POLL_INTERVAL:-5}
PIPELINE_RAY_BLOCK=${PIPELINE_RAY_BLOCK:-true}
DRY_RUN=${DRY_RUN:-false}

usage() {
    cat <<'EOF'
Usage:
  ray_mutilnode_train.sh head
  ray_mutilnode_train.sh worker [HEAD_IP[:PORT]]
  ray_mutilnode_train.sh wait
  ray_mutilnode_train.sh status
  ray_mutilnode_train.sh train [train_multinode.sh arguments ...]

Required for worker: RAY_NODE_IP and a head address (environment or argument).
Required for train: RAY_HEAD_ADDRESS, PRODUCER_NODE, CONSUMER_NODE.

Useful settings:
  RAY_NUM_GPUS/RAY_NUM_CPUS       Resources per node (8/24).
  RAY_EXPECTED_NODES/RAY_EXPECTED_GPUS
                                  Readiness target (2/16).
  RAY_WAIT_TIMEOUT                 Total wait time in seconds (900).
  RAY_CONNECT_TIMEOUT              One Ray query timeout in seconds (10).
  RAY_POLL_INTERVAL                Poll interval in seconds (5).
  PRODUCER_DP/CONSUMER_DP          Training DP defaults (2/2; each may be 1 or 2).
  PIPELINE_RAY_BLOCK               Keep head/worker in foreground (true).
  DRY_RUN=true                     Print commands without starting anything.

The current pipeline uses one producer and one consumer node. Set
RAY_EXPECTED_NODES=6 (and RAY_EXPECTED_GPUS=48) when all six nodes must be
present before training.
EOF
}

die() {
    printf 'ray_mutilnode_train.sh: %s\n' "$*" >&2
    exit 2
}

is_bool() { [[ "$1" == true || "$1" == false ]]; }
is_positive_int() { [[ "$1" =~ ^[0-9]+$ ]] && ((10#$1 > 0)); }

is_bool "${DRY_RUN}" || die 'DRY_RUN must be true or false'
is_bool "${PIPELINE_RAY_BLOCK}" || die 'PIPELINE_RAY_BLOCK must be true or false'
for name in RAY_HEAD_PORT RAY_NUM_GPUS RAY_NUM_CPUS RAY_EXPECTED_NODES \
    RAY_EXPECTED_GPUS RAY_WAIT_TIMEOUT RAY_CONNECT_TIMEOUT RAY_POLL_INTERVAL; do
    is_positive_int "${!name}" || die "${name} must be a positive integer"
done

MODE=${1:-}
if [[ -z "${MODE}" || "${MODE}" == -h || "${MODE}" == --help ]]; then
    usage
    exit 0
fi
shift

print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

check_runtime() {
    [[ "${DRY_RUN}" == true ]] && return 0
    [[ -f "${RAY_SCRIPT}" ]] || die "Ray launcher not found: ${RAY_SCRIPT}"
    [[ -f "${TRAIN_SCRIPT}" ]] || die "Training launcher not found: ${TRAIN_SCRIPT}"
    command -v timeout >/dev/null 2>&1 || die 'timeout is required'
    [[ -x "${PIPELINE_PYTHON}" ]] || die "Python executable not found: ${PIPELINE_PYTHON}"
}

local_ip() {
    [[ -n "${RAY_NODE_IP:-}" ]] && { printf '%s\n' "${RAY_NODE_IP}"; return; }
    [[ "${DRY_RUN}" == true ]] && { printf '<auto-detected-node-ip>\n'; return; }
    local ip
    ip=$("${PIPELINE_PYTHON}" -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
    [[ -n "${ip}" ]] || die 'set RAY_NODE_IP; no local IP was detected'
    printf '%s\n' "${ip}"
}

with_port() {
    local address=$1
    [[ -n "${address}" ]] || die 'Ray head address is empty'
    case "${address}" in
        *://*|*:* ) printf '%s\n' "${address}" ;;
        * ) printf '%s:%s\n' "${address}" "${RAY_HEAD_PORT}" ;;
    esac
}

head_address() {
    with_port "${RAY_HEAD_ADDRESS:-${RAY_HEAD_IP:-}}"
}

ray_probe() {
    local address=$1
    local seconds=${2:-${RAY_CONNECT_TIMEOUT}}
    timeout --signal=TERM --kill-after=1s "${seconds}s" \
        "${PIPELINE_PYTHON}" - "${address}" "${RAY_EXPECTED_NODES}" \
        "${RAY_EXPECTED_GPUS}" "${RAY_NUM_GPUS}" <<'PY'
import json
import sys

import ray

address, expected_nodes, expected_gpus, gpus_per_node = sys.argv[1:]
expected_nodes = int(expected_nodes)
expected_gpus = int(expected_gpus)
gpus_per_node = float(gpus_per_node)
ray.init(address=address, logging_level="ERROR")
try:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    gpus = [float(node.get("Resources", {}).get("GPU", 0)) for node in nodes]
    result = {
        "nodes": len(nodes),
        "gpus": sum(gpus),
        "expected_nodes": expected_nodes,
        "expected_gpus": expected_gpus,
        "gpu_ready_nodes": sum(value >= gpus_per_node for value in gpus),
        "members": [
            {
                "ip": node.get("NodeManagerAddress"),
                "gpus": node.get("Resources", {}).get("GPU", 0),
                "node_id": node.get("NodeID", "")[:12],
            }
            for node in nodes
        ],
    }
    result["ready"] = (
        result["nodes"] >= expected_nodes
        and result["gpus"] >= expected_gpus
        and result["gpu_ready_nodes"] >= expected_nodes
    )
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["ready"] else 1)
finally:
    ray.shutdown()
PY
}

wait_for_ray() {
    local address=$1
    local deadline=$((SECONDS + RAY_WAIT_TIMEOUT))
    local last=unavailable
    while ((SECONDS < deadline)); do
        local remaining=$((deadline - SECONDS))
        local probe_timeout=${RAY_CONNECT_TIMEOUT}
        ((probe_timeout > remaining)) && probe_timeout=${remaining}
        if last=$(ray_probe "${address}" "${probe_timeout}" 2>&1); then
            printf 'Ray ready: %s\n' "${last}"
            return 0
        fi
        printf 'Waiting for Ray at %s (%s nodes / %s GPUs): %s\n' \
            "${address}" "${RAY_EXPECTED_NODES}" "${RAY_EXPECTED_GPUS}" "${last}"
        remaining=$((deadline - SECONDS))
        ((remaining <= 0)) && break
        sleep "$((RAY_POLL_INTERVAL < remaining ? RAY_POLL_INTERVAL : remaining))"
    done
    printf 'Timed out after %ss waiting for Ray at %s. Last status: %s\n' \
        "${RAY_WAIT_TIMEOUT}" "${address}" "${last}" >&2
    return 1
}

start_ray() {
    local role=$1
    local node_ip=$2
    shift 2
    local -a command=(bash "${RAY_SCRIPT}" "${role}" "$@")
    print_command env RAY_NODE_IP="${node_ip}" RAY_NUM_GPUS="${RAY_NUM_GPUS}" \
        RAY_NUM_CPUS="${RAY_NUM_CPUS}" RAY_HEAD_PORT="${RAY_HEAD_PORT}" \
        PIPELINE_RAY_BLOCK="${PIPELINE_RAY_BLOCK}" "${command[@]}"
    [[ "${DRY_RUN}" == true ]] && return 0
    exec env RAY_NODE_IP="${node_ip}" RAY_NUM_GPUS="${RAY_NUM_GPUS}" \
        RAY_NUM_CPUS="${RAY_NUM_CPUS}" RAY_HEAD_PORT="${RAY_HEAD_PORT}" \
        PIPELINE_RAY_BLOCK="${PIPELINE_RAY_BLOCK}" bash "${RAY_SCRIPT}" "${role}" "$@"
}

mode_head() {
    [[ $# -eq 0 ]] || die 'head takes no arguments'
    check_runtime
    local node_ip
    node_ip=$(local_ip)
    printf 'Starting Ray head at %s:%s (%s GPUs, %s CPUs).\n' \
        "${node_ip}" "${RAY_HEAD_PORT}" "${RAY_NUM_GPUS}" "${RAY_NUM_CPUS}"
    start_ray head "${node_ip}"
}

mode_worker() {
    [[ $# -le 1 ]] || die 'worker accepts at most one head address'
    check_runtime
    [[ -n "${RAY_NODE_IP:-}" || "${DRY_RUN}" == true ]] || die 'worker requires RAY_NODE_IP'
    local address
    if [[ $# -eq 1 ]]; then
        [[ "$1" != -* ]] || die 'invalid head address'
        address=$(with_port "$1")
    else
        address=$(head_address)
    fi
    local node_ip
    node_ip=$(local_ip)
    printf 'Joining Ray head %s as worker %s (%s GPUs, %s CPUs).\n' \
        "${address}" "${node_ip}" "${RAY_NUM_GPUS}" "${RAY_NUM_CPUS}"
    start_ray worker "${node_ip}" "${address}"
}

mode_wait() {
    [[ $# -eq 0 ]] || die 'wait takes no arguments'
    check_runtime
    local address
    address=$(head_address)
    if [[ "${DRY_RUN}" == true ]]; then
        printf 'DRY_RUN: would wait for %s nodes / %s GPUs at %s\n' \
            "${RAY_EXPECTED_NODES}" "${RAY_EXPECTED_GPUS}" "${address}"
        return 0
    fi
    wait_for_ray "${address}"
}

mode_status() {
    [[ $# -eq 0 ]] || die 'status takes no arguments'
    check_runtime
    local address
    address=$(head_address)
    if [[ "${DRY_RUN}" == true ]]; then
        printf 'DRY_RUN: would query Ray at %s\n' "${address}"
        return 0
    fi
    printf 'Ray membership:\n'
    ray_probe "${address}" || true
    printf '\nRay status:\n'
    timeout --signal=TERM --kill-after=1s "${RAY_CONNECT_TIMEOUT}s" \
        "${PIPELINE_PYTHON}" -m ray.scripts.scripts status --address "${address}"
}

mode_train() {
    check_runtime
    local address=${RAY_HEAD_ADDRESS:-}
    [[ -n "${address}" ]] || die 'train requires RAY_HEAD_ADDRESS'
    address=$(with_port "${address}")
    local producer=${PRODUCER_NODE:-}
    local consumer=${CONSUMER_NODE:-}
    [[ -n "${producer}" ]] || die 'train requires PRODUCER_NODE'
    [[ -n "${consumer}" ]] || die 'train requires CONSUMER_NODE'
    [[ "${producer}" != "${consumer}" ]] || die 'producer and consumer nodes must differ'

    local output="${REPO_ROOT}/outputs/dspark_ray_multinode_$(date +%Y%m%d_%H%M%S)_$$"
    local producer_dp=${PRODUCER_DP:-2}
    local consumer_dp=${CONSUMER_DP:-2}
    local -a passthrough=()
    while (($#)); do
        case "$1" in
            --output)
                (($# >= 2)) || die '--output requires a path'
                output=$2
                shift 2
                ;;
            --output=*)
                output=${1#--output=}
                [[ -n "${output}" ]] || die '--output requires a path'
                shift
                ;;
            --producer-dp)
                (($# >= 2)) || die '--producer-dp requires a value'
                producer_dp=$2
                shift 2
                ;;
            --producer-dp=*)
                producer_dp=${1#--producer-dp=}
                shift
                ;;
            --consumer-dp)
                (($# >= 2)) || die '--consumer-dp requires a value'
                consumer_dp=$2
                shift 2
                ;;
            --consumer-dp=*)
                consumer_dp=${1#--consumer-dp=}
                shift
                ;;
            *)
                passthrough+=("$1")
                shift
                ;;
        esac
    done
    [[ "${producer_dp}" == 1 || "${producer_dp}" == 2 ]] || die 'PRODUCER_DP must be 1 or 2'
    [[ "${consumer_dp}" == 1 || "${consumer_dp}" == 2 ]] || die 'CONSUMER_DP must be 1 or 2'
    if [[ "${output}" != /* ]]; then
        output="${REPO_ROOT}/${output}"
    fi
    output=${output%/}
    [[ -n "${output}" && "${output}" != / ]] || die 'output path must not be /'

    local run_name=${output##*/}
    local log_path="${REPO_ROOT}/outputs/launch_logs/${run_name}.log"
    local -a command=(bash "${TRAIN_SCRIPT}"
        --producer-dp "${producer_dp}"
        --consumer-dp "${consumer_dp}"
        "${passthrough[@]}"
        --output "${output}"
    )
    printf 'Training: head=%s producer=%s consumer=%s.\n' \
        "${address}" "${producer}" "${consumer}"
    print_command env RAY_HEAD_ADDRESS="${address}" \
        PRODUCER_NODE="${producer}" CONSUMER_NODE="${consumer}" "${command[@]}"
    printf 'Launch log: %s\n' "${log_path}"
    if [[ "${DRY_RUN}" == true ]]; then
        printf 'DRY_RUN: would wait for %s nodes / %s GPUs at %s\n' \
            "${RAY_EXPECTED_NODES}" "${RAY_EXPECTED_GPUS}" "${address}"
        return 0
    fi
    wait_for_ray "${address}"
    mkdir -p "${REPO_ROOT}/outputs/launch_logs"
    env RAY_HEAD_ADDRESS="${address}" PRODUCER_NODE="${producer}" \
        CONSUMER_NODE="${consumer}" "${command[@]}" 2>&1 | tee "${log_path}"
}

case "${MODE}" in
    head) mode_head "$@" ;;
    worker) mode_worker "$@" ;;
    wait) mode_wait "$@" ;;
    status) mode_status "$@" ;;
    train) mode_train "$@" ;;
    *) usage >&2; die "unknown mode: ${MODE}" ;;
esac

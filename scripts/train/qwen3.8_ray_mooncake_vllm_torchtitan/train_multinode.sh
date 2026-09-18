#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${RAY_HEAD_ADDRESS:?Set RAY_HEAD_ADDRESS to the existing Ray Head IP:port}"
: "${PRODUCER_NODE:?Set PRODUCER_NODE to the producer Ray node IP or ID}"
: "${CONSUMER_NODE:?Set CONSUMER_NODE to the consumer Ray node IP or ID}"

exec bash "${SCRIPT_DIR}/train.sh" \
    --model /mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B \
    --ray-address "${RAY_HEAD_ADDRESS}" \
    --producer-node "${PRODUCER_NODE}" \
    --consumer-node "${CONSUMER_NODE}" \
    --context-length 131072 --pool-gib 64 --timeout-seconds 3600 \
    "$@"

#!/usr/bin/env bash
# Start or join Ray on this node, then return without --block.
#
# Usage: bash examples/qwen3.8-27b-dflash2-3node/ray_build.sh [--dry-run]
# Run the same command on every node; h800conda.sh activates the Ray environment.
#
# Environment variables:
#   MASTER_ADDR       Required: the same head IPv4 address or hostname on all nodes
#                     (hostnames must resolve to exactly one IPv4 address).
#   RAY_PORT          Head port (default: 6379; independent of MASTER_PORT)
#   RAY_WAIT_TIMEOUT  Maximum seconds a worker waits for the head (default: 120)
#
# Every node advertises 8 GPUs and disables usage statistics.
# --dry-run only prints the role and command; it does not contact or start Ray.

set -euo pipefail

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

dry_run=false
case "${1:-}" in
  -h|--help)
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}"
    exit 0
    ;;
  --dry-run)
    dry_run=true
    shift
    ;;
  "") ;;
  *) fail "Unknown argument: $1" ;;
esac
(( $# == 0 )) || fail "Unexpected arguments: $*"

MASTER_ADDR="${MASTER_ADDR:-}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_WAIT_TIMEOUT="${RAY_WAIT_TIMEOUT:-120}"
[[ -n "$MASTER_ADDR" ]] || fail "MASTER_ADDR must point to the head on every node."
[[ "$RAY_PORT" =~ ^[0-9]{1,5}$ ]] || fail "RAY_PORT must be an integer from 1 to 65535."
RAY_PORT="$((10#$RAY_PORT))"
(( RAY_PORT >= 1 && RAY_PORT <= 65535 )) || fail "RAY_PORT must be from 1 to 65535."
[[ "$RAY_WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "RAY_WAIT_TIMEOUT must be a positive integer."

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/h800conda.sh"

for command in ray getent ip awk timeout; do
  command -v "$command" >/dev/null 2>&1 || fail "Required command not found: $command"
done

if ! master_ip="$(getent ahostsv4 "$MASTER_ADDR" | awk '!seen[$1]++ {print $1}')"; then
  fail "Cannot resolve MASTER_ADDR=$MASTER_ADDR to IPv4."
fi
[[ -n "$master_ip" ]] || fail "MASTER_ADDR=$MASTER_ADDR has no IPv4 address."
[[ "$master_ip" != *$'\n'* ]] || fail "MASTER_ADDR resolves to multiple IPv4 addresses; use the head IP explicitly."
[[ "$master_ip" != 127.* && "$master_ip" != 0.0.0.0 ]] || fail "MASTER_ADDR must be reachable from other nodes."

# Compare every interface address, including secondary addresses on the head.
local_ips="$(ip -o -4 addr show | awk '{sub(/\/.*/, "", $4); print $4}')"
[[ -n "$local_ips" ]] || fail "Cannot find a local IPv4 address."
role=worker
for local_ip in $local_ips; do
  if [[ "$local_ip" == "$master_ip" ]]; then
    role=head
    break
  fi
done

ray_address="$master_ip:$RAY_PORT"
export RAY_USAGE_STATS_ENABLED=0
launch=(ray start --num-gpus=8 --disable-usage-stats)
if [[ "$role" == head ]]; then
  launch+=(--head "--node-ip-address=$master_ip" "--port=$RAY_PORT")
else
  # Ray selects the worker IP using the route to this head address.
  launch+=("--address=$ray_address")
fi

printf 'Ray role: %s; head: %s\n' "$role" "$ray_address"
if [[ "$dry_run" == true ]]; then
  printf 'RAY_USAGE_STATS_ENABLED=0 '
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi

if [[ "$role" == worker ]]; then
  printf 'Waiting up to %s seconds for Ray head at %s...\n' "$RAY_WAIT_TIMEOUT" "$ray_address"
  # Bound the entire readiness loop, including a stalled Ray status request.
  if ! timeout --kill-after=5s "${RAY_WAIT_TIMEOUT}s" bash -c '
    until ray status --address="$1" >/dev/null 2>&1; do
      sleep 2
    done
  ' _ "$ray_address"; then
    fail "Ray head at $ray_address did not become ready within $RAY_WAIT_TIMEOUT seconds."
  fi
fi

exec "${launch[@]}"

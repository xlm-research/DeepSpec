#!/usr/bin/env bash
# One-command launcher for the Qwen3.8-27B three-node recipes.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$ROOT_DIR/tools/launch_3node.py" "$@"

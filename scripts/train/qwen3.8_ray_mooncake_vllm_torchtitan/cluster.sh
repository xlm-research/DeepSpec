#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Compatibility entry point for existing commands and job definitions.
exec bash "${SCRIPT_DIR}/start_ray.sh" "$@"

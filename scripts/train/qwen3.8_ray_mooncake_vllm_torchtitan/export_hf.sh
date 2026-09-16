#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PIPELINE_PYTHON=/tmp/deepspec_vllm_torchtitan_envs/bin/python

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/torchtitan:${REPO_ROOT}/vllm"
export CUDA_VISIBLE_DEVICES=''

# Arguments: checkpoint directory, HF output directory; --help is also supported.
# The native exporter reads model configuration and precision from commit.json.
export_command=(
    "${PIPELINE_PYTHON}" -u -m torchtitan.models.dspark_draft.export
    "$@"
)

printf 'Working directory: %s\n' "${REPO_ROOT}"
printf 'Command: CUDA_VISIBLE_DEVICES=%q' "${CUDA_VISIBLE_DEVICES}"
printf ' %q' "${export_command[@]}"
printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    exit 0
fi

if [[ ! -x "${PIPELINE_PYTHON}" ]]; then
    printf 'Python executable not found: %s\n' "${PIPELINE_PYTHON}" >&2
    exit 1
fi

exec "${export_command[@]}"

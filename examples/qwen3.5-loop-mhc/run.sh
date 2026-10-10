#!/usr/bin/env bash
set -euo pipefail

loop_example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
loop_repo_dir=$(cd -- "${loop_example_dir}/../.." && pwd)
source "${loop_repo_dir}/h800conda.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
exec uv run --no-project --python "${loop_repo_dir}/.venv/bin/python" \
    "${loop_repo_dir}/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --config "${loop_example_dir}/config.yaml" "$@"

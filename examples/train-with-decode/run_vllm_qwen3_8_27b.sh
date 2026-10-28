#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

# h800conda-tmp.sh also puts the locally compiled vLLM checkout on PYTHONPATH.
source "$repo_root/h800conda-tmp.sh"

python -m torchspec.train_entry \
  --config "$repo_root/configs/train_with_decode/vllm_qwen3_8_27b.yaml" \
  "$@"

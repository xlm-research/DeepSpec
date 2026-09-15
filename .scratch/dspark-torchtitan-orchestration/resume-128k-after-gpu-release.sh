#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source ./env.sh

run_dir="$PWD/outputs/dspark_torchtitan_orchestration_20260914"
request_path="$run_dir/128k-h800-request.json"
data_path="${DEEPSPEC_SCALE_DATA:-$run_dir/128k-source.jsonl}"
for required in "$request_path" "$data_path" "$TARGET_MODEL_PATH/config.json"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required run input: $required" >&2
        exit 1
    fi
done

# The H800 run regenerates missing features and resumes its own saved progress.
# The orchestrator requires all GPUs idle.
exec env -u TORCHINDUCTOR_COMPILE_THREADS \
    OMP_NUM_THREADS=1 \
    PYTHONPATH="$PWD:$PWD/torchtitan:$PWD/vllm" \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    DEEPSPEC_SCALE_DATA="$data_path" \
    DEEPSPEC_SCALE_OUTPUT="$run_dir/qwen38-128k-h800" \
    DEEPSPEC_SCALE_CAPTURE="$run_dir/scale-initialization-h800" \
    DEEPSPEC_SCALE_REFERENCE= \
    "$VIRTUAL_ENV/bin/python" \
    -m deepspec.orchestration.run \
    "$request_path" \
    >> "$run_dir/qwen38-128k-h800.log" 2>&1

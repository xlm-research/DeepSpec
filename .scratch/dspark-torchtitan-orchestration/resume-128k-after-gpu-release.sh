#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs-agentpro/lezewei/DeepSpec

# Resume the retained target partition. The orchestrator requires all GPUs idle.
exec env -u TORCHINDUCTOR_COMPILE_THREADS \
    OMP_NUM_THREADS=1 \
    PYTHONPATH="$PWD:$PWD/torchtitan:$PWD/vllm" \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    DEEPSPEC_SCALE_DATA="$PWD/output/dspark_torchtitan_orchestration_20260914/128k-source.jsonl" \
    DEEPSPEC_SCALE_OUTPUT="$PWD/output/dspark_torchtitan_orchestration_20260914/qwen38-128k-v2" \
    DEEPSPEC_SCALE_CAPTURE="$PWD/output/dspark_torchtitan_orchestration_20260914/scale-initialization" \
    DEEPSPEC_SCALE_REFERENCE= \
    /mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python \
    -m deepspec.orchestration.run \
    output/dspark_torchtitan_orchestration_20260914/128k-v2-request.json \
    >> output/dspark_torchtitan_orchestration_20260914/qwen38-128k-v2.log 2>&1

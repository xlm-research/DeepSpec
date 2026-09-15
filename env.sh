#!/usr/bin/env bash
# Source this file to use the local TorchTitan/vLLM environment on this machine.
_deepspec_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$_deepspec_repo/.envs/orchestration/bin/activate"
export CUDA_HOME=/tmp/deepspec_vllm_torchtitan_envs/lib/python3.12/site-packages/nvidia/cu13
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$_deepspec_repo:$_deepspec_repo/torchtitan:$_deepspec_repo/vllm${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PYTHON_BIN="$VIRTUAL_ENV/bin/python"
export VLLM_SOURCE_DIR="$_deepspec_repo/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDA_ARCH_LIST=9.0
export TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B}
unset _deepspec_repo

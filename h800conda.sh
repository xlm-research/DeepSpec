#!/usr/bin/env bash
# Source this file before running the H800 pipeline.
source /mnt/afs_share/miniconda3/etc/profile.d/conda.sh
conda activate deepspec_vllm_torchtitan_envs || return

_deepspec_h800_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
_deepspec_h800_site=$("${CONDA_PREFIX}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export PIPELINE_PYTHON="${CONDA_PREFIX}/bin/python"
export DEBUG_PYTHON="${PIPELINE_PYTHON}"
export PYTHONPATH="${_deepspec_h800_root}:${_deepspec_h800_root}/torchtitan:${_deepspec_h800_root}/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${_deepspec_h800_site}/nvidia/cu13"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${_deepspec_h800_site}/nvidia/cuda_runtime/lib:${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLLM_PYTHON_BIN="${PIPELINE_PYTHON}"
export VLLM_SOURCE_DIR="${_deepspec_h800_root}/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_CUDA_ARCH_LIST=9.0
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B}
unset _deepspec_h800_root _deepspec_h800_site

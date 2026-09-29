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
# Mooncake's libcurl must reach the node-local metadata service directly.
_deepspec_h800_no_proxy="${no_proxy:-${NO_PROXY:-}}"
for _deepspec_h800_host in localhost 127.0.0.1 ::1 $(hostname -I); do
    case ",${_deepspec_h800_no_proxy}," in
        *,"${_deepspec_h800_host}",*) ;;
        *) _deepspec_h800_no_proxy="${_deepspec_h800_no_proxy:+${_deepspec_h800_no_proxy},}${_deepspec_h800_host}" ;;
    esac
done
export no_proxy="${_deepspec_h800_no_proxy}"
export NO_PROXY="${no_proxy}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=1
export TORCH_CUDA_ARCH_LIST=9.0
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
if [[ -z "${TARGET_MODEL_PATH:-}" ]]; then
    TARGET_MODEL_PATH=/mnt/afs_agents/hongjiawei/delivery/sensenova-flash-lite-v42-20260830-dpo-mtp
fi
export TARGET_MODEL_PATH
unset _deepspec_h800_root _deepspec_h800_site _deepspec_h800_no_proxy _deepspec_h800_host

#!/usr/bin/env bash
# Source this file on B300 before launching Ray, vLLM or TorchTitan.
# Override DEEPSPEC_CONDA_SH / DEEPSPEC_CONDA_ENV on nodes with another install.
_deepspec_activate() {
    local repo conda_sh site arch
    repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return
    conda_sh=${DEEPSPEC_CONDA_SH:-/mnt/afs-agentpro/share/env/miniconda3/etc/profile.d/conda.sh}
    if [[ ! -f "$conda_sh" ]]; then
        printf 'Conda initialization file not found: %s. Set DEEPSPEC_CONDA_SH.\n' "$conda_sh" >&2
        return 1
    fi
    source "$conda_sh" || return
    conda activate "${DEEPSPEC_CONDA_ENV:-deepspec_vllm_torchtitan_envs}" || return
    site=$("${CONDA_PREFIX}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))') || return
    if [[ ! -x "$site/nvidia/cu13/bin/nvcc" ]]; then
        printf 'CUDA 13 compiler missing from conda environment: %s\n' "$CONDA_PREFIX" >&2
        return 1
    fi
    export PIPELINE_PYTHON="${CONDA_PREFIX}/bin/python"
    export DEBUG_PYTHON="$PIPELINE_PYTHON"
    export CUDA_HOME="$site/nvidia/cu13"
    export CUDA_PATH="$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    # Mooncake's wheel needs CUDA 12 runtime alongside PyTorch's CUDA 13.
    export LD_LIBRARY_PATH="$site/nvidia/cuda_runtime/lib:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$repo:$repo/torchtitan:$repo/vllm${PYTHONPATH:+:$PYTHONPATH}"
    export VLLM_PYTHON_BIN="$PIPELINE_PYTHON"
    export VLLM_SOURCE_DIR="$repo/vllm"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -) || return
        [[ -n "$arch" ]] || { printf 'No CUDA GPU architecture detected.\n' >&2; return 1; }
        export TORCH_CUDA_ARCH_LIST="$arch"
    fi
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export TOKENIZERS_PARALLELISM=false
    export WANDB_MODE=online
    export WANDB_PROJECT=deepspec
    export WANDB_NAME=dspark-run
    export TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B}
}
if _deepspec_activate; then
    unset -f _deepspec_activate
else
    unset -f _deepspec_activate
    return 1 2>/dev/null || exit 1
fi

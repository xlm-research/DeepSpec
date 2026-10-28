#!/usr/bin/env bash
# Source this file before running the H800 pipeline.
export TORCHSPEC_CONDA_ENV="${TORCHSPEC_CONDA_ENV:-/tmp/deepspec_vllm_torchtitan_envs}"
case "$(realpath -m -- "$TORCHSPEC_CONDA_ENV")" in
    /tmp/*) ;;
    *) echo 'TORCHSPEC_CONDA_ENV must point to an environment under /tmp.' >&2; return 1 ;;
esac
if [[ ! -x "$TORCHSPEC_CONDA_ENV/bin/python" ]]; then
    echo "Missing local environment: $TORCHSPEC_CONDA_ENV. Extract it under /tmp first." >&2
    return 1
fi
# Load the extracted environment directly, without invoking the AFS Conda base.
for _deepspec_h800_var in PATH LD_LIBRARY_PATH PYTHONPATH; do
    IFS=: read -r -a _deepspec_h800_entries <<< "${!_deepspec_h800_var-}"
    _deepspec_h800_clean=""
    for _deepspec_h800_entry in "${_deepspec_h800_entries[@]}"; do
        case "$_deepspec_h800_entry" in
            /mnt/afs_share/miniconda3|/mnt/afs_share/miniconda3/*|'') continue ;;
        esac
        _deepspec_h800_clean="${_deepspec_h800_clean:+$_deepspec_h800_clean:}$_deepspec_h800_entry"
    done
    export "$_deepspec_h800_var=$_deepspec_h800_clean"
done
export CONDA_PREFIX="$TORCHSPEC_CONDA_ENV"
export CONDA_DEFAULT_ENV="$CONDA_PREFIX"
unset PYTHONHOME
if [[ -f "$CONDA_PREFIX/conda-meta/state" ]]; then
    while IFS=$'\t' read -r _deepspec_h800_key _deepspec_h800_value; do
        export "$_deepspec_h800_key=$_deepspec_h800_value"
    done < <("$CONDA_PREFIX/bin/python" -c '
import json, os, re
from pathlib import Path
prefix = os.environ["CONDA_PREFIX"]
state = json.loads((Path(prefix) / "conda-meta/state").read_text())
for key, value in state.get("env_vars", {}).items():
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise ValueError(f"Invalid environment variable name: {key}")
    value = str(value).replace("/mnt/afs_share/miniconda3/envs/deepspec_vllm_torchtitan_envs", prefix)
    if "\n" in value or "\t" in value:
        raise ValueError(f"Unsupported multiline environment variable: {key}")
    print(key + "\t" + value)
')
fi
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/cargo/bin:$PATH"
hash -r

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
unset _deepspec_h800_var _deepspec_h800_entries _deepspec_h800_entry _deepspec_h800_clean
unset _deepspec_h800_key _deepspec_h800_value

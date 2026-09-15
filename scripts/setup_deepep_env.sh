#!/usr/bin/env bash
# Build a draft-only DeepEP v2 environment without changing the target environment.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
deepep_commit=01dc3aaac82068020353dce2c302e38153c0bfaa
env_dir=$(realpath -m -- "${1:-${repo_dir}/../.venvs/deepspec-deepep-v2-01dc3aaa}")
base_python=${DEEPSPEC_DEEPEP_BASE_PYTHON:-python}
cuda_dir=${DEEPSPEC_DEEPEP_CUDA_HOME:-/usr/local/cuda}
source_dir=${DEEPSPEC_DEEPEP_SOURCE_DIR:-${env_dir}/src/DeepEP}
marker=${env_dir}/.deepspec-deepep-environment

if [[ -e "${env_dir}" && ! -f "${marker}" ]]; then
    echo "Refusing to install into an existing environment not created for this task: ${env_dir}" >&2
    exit 1
fi
if [[ ! -x "${cuda_dir}/bin/nvcc" ]]; then
    echo "CUDA nvcc is missing: ${cuda_dir}/bin/nvcc" >&2
    exit 1
fi
if [[ ! -f "${marker}" ]]; then
    "${base_python}" -m venv --system-site-packages "${env_dir}"
    echo "Isolated DSpark draft DeepEP environment." > "${marker}"
fi
env_python=${env_dir}/bin/python
site_dir=$("${env_python}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
install_options=()
if [[ ! -f "${site_dir}/nvidia/nccl/lib/libnccl.so.2" ]]; then
    # Install a private copy even if the base interpreter already has these versions.
    install_options+=(--ignore-installed)
fi
"${env_python}" -m pip install --no-deps "${install_options[@]}" -r "${repo_dir}/requirements-deepep.txt"

# PyTorch wheels use DT_RPATH, which can select the base environment's older NCCL.
# Load our NCCL before torch using a venv-private .pth. No LD_PRELOAD or
# LD_LIBRARY_PATH is exported, so vLLM's original Python children stay unchanged.
"${env_python}" - <<'PY'
import sysconfig
from pathlib import Path

site = Path(sysconfig.get_path("purelib"))
library = site / "nvidia/nccl/lib/libnccl.so.2"
if not library.is_file():
    raise RuntimeError(f"NCCL must be installed inside the isolated venv: {library}")
(site / "00_deepspec_nccl.pth").write_text(
    f"import ctypes; ctypes.CDLL({str(library)!r}, mode=ctypes.RTLD_GLOBAL)\n"
)
PY
nccl_lib_dir=$("${env_python}" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl/lib")')

if [[ ! -d "${source_dir}/.git" ]]; then
    git clone --no-checkout https://github.com/deepseek-ai/DeepEP.git "${source_dir}"
elif [[ -n "$(git -C "${source_dir}" status --porcelain)" ]]; then
    echo "DeepEP source checkout has changes; refusing to overwrite them: ${source_dir}" >&2
    exit 1
fi
git -C "${source_dir}" checkout --detach "${deepep_commit}"
git -C "${source_dir}" submodule update --init --recursive

# Hide GPUs during installation. CUDA sources compile for the explicit architecture;
# DeepEP's v2 communication kernels will JIT during the first real GPU invocation.
CUDA_VISIBLE_DEVICES='' CUDA_HOME="${cuda_dir}" \
    TORCH_CUDA_ARCH_LIST="${DEEPSPEC_DEEPEP_CUDA_ARCH:-9.0}" \
    MAX_JOBS="${DEEPSPEC_DEEPEP_BUILD_JOBS:-8}" \
    LIBRARY_PATH="${nccl_lib_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
    "${env_python}" -m pip install --no-build-isolation --no-deps "${source_dir}"

CUDA_VISIBLE_DEVICES='' "${env_python}" - <<'PY'
import ctypes
import importlib.metadata
import sys
from pathlib import Path

import torch
import deep_ep

loaded = {
    line.split()[-1]
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "libnccl.so" in line
}
if len(loaded) != 1 or not Path(next(iter(loaded))).is_relative_to(sys.prefix):
    raise RuntimeError(f"NCCL was not isolated to the draft environment: {loaded}")
version = ctypes.c_int()
status = ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(version))
if status != 0 or version.value < 23004:
    raise RuntimeError(f"NCCL runtime is incompatible: status={status}, version={version.value}")
if torch.cuda.is_initialized():
    raise RuntimeError("CPU import verification unexpectedly initialized CUDA")
assert hasattr(deep_ep, "ElasticBuffer")
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; DeepEP: {importlib.metadata.version('deep_ep')}")
print(f"NCCL runtime: {version.value}; loaded: {next(iter(loaded))}")
print("CPU imports passed; no GPU kernel or distributed collective has run.")
PY

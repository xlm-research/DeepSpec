source /mnt/afs-agentpro/share/env/miniconda3/etc/profile.d/conda.sh
conda activate deepspec_vllm_torchtitan_envs
# Bind the existing source-built vLLM runtime independently of draft launch options.
export VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN:-${CONDA_PREFIX}/bin/python}
export VLLM_SOURCE_DIR=${VLLM_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vllm}

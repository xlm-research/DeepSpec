#!/usr/bin/env bash
# Run with: source ./cp_envs.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' '请使用 source ./cp_envs.sh，以便在当前终端激活 Conda 环境。' >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    printf '%s\n' '未找到 conda，请先加载 Conda，再 source 此脚本。' >&2
    return 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh" || return 1

# Reuse an environment that has already been extracted.
if [[ -x /tmp/deepspec_vllm_torchtitan_envs/bin/python && -d /tmp/deepspec_vllm_torchtitan_envs/conda-meta ]]; then
    conda activate /tmp/deepspec_vllm_torchtitan_envs || return 1
    return 0
fi

mkdir -p /tmp/deepspec_vllm_torchtitan_envs || return 1
if [[ ! -f /tmp/deepspec_vllm_torchtitan_envs/deepspec_vllm_torchtitan_envs.tar ]]; then
    cp -- /mnt/afs_share/lezewei/TorchSpec/envs/deepspec_vllm_torchtitan_envs.tar /tmp/deepspec_vllm_torchtitan_envs/ || return 1
fi
# The archive already contains a deepspec_vllm_torchtitan_envs/ directory.
tar -xf /tmp/deepspec_vllm_torchtitan_envs/deepspec_vllm_torchtitan_envs.tar -C /tmp/deepspec_vllm_torchtitan_envs --strip-components=1 || return 1

conda activate /tmp/deepspec_vllm_torchtitan_envs || return 1
if [[ -f /tmp/deepspec_vllm_torchtitan_envs/bin/conda-unpack ]]; then
    /tmp/deepspec_vllm_torchtitan_envs/bin/python /tmp/deepspec_vllm_torchtitan_envs/bin/conda-unpack || return 1
fi

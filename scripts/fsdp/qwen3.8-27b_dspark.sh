#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

SRC="${REPO_ROOT}/envs/deepspec_vllm_torchtitan_envs.tar"

export VLLM_WORKER_MULTIPROC_METHOD=fork

CACHE_DIR=$(mktemp -d /tmp/deepspec-env-tar-cache.XXXXXX) &&
LOCAL_TAR="$CACHE_DIR/deepspec_vllm_torchtitan_envs.tar" &&
DST="$CACHE_DIR/deepspec_vllm_torchtitan_envs" &&
echo "正在拷贝：$SRC → $LOCAL_TAR" &&
cp -a -- "$SRC" "$LOCAL_TAR" &&
echo "正在解包：$LOCAL_TAR → $CACHE_DIR" &&
tar -xpf "$LOCAL_TAR" -C "$CACHE_DIR" &&
echo "环境准备完成：$DST" &&
DEEPSPEC_FAST_PY="$DST/bin/python" &&
PYTHON_BIN="$DEEPSPEC_FAST_PY" \
VLLM_PYTHON_BIN="$DEEPSPEC_FAST_PY" \
CONTEXT_PARALLEL_SIZE=1 \
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/dspark_qwen3_8_27b_vllm}" \
bash scripts/train/train_qwen3_8_27b_dspark_vllm.sh

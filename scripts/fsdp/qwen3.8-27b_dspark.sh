#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export VLLM_WORKER_MULTIPROC_METHOD=fork
if [[ -n "${DEEPSPEC_ENV_DIR:-}" ]]; then
    DST="${DEEPSPEC_ENV_DIR}"
else
    SRC="${DEEPSPEC_ENV_TAR:-${REPO_ROOT}/envs/deepspec_vllm_torchtitan_env.tar}"
    if [[ ! -f "${SRC}" ]]; then
        echo "Environment archive not found: ${SRC}; set DEEPSPEC_ENV_TAR or DEEPSPEC_ENV_DIR." >&2
        exit 1
    fi
    CACHE_DIR=$(mktemp -d /tmp/deepspec-env-tar-cache.XXXXXX)
    LOCAL_TAR="$CACHE_DIR/deepspec_vllm_torchtitan_env.tar"
    DST="$CACHE_DIR/deepspec_vllm_torchtitan_envs"
    echo "正在拷贝：$SRC → $LOCAL_TAR"
    cp -a -- "$SRC" "$LOCAL_TAR"
    echo "正在解包：$LOCAL_TAR → $CACHE_DIR"
    tar -xpf "$LOCAL_TAR" -C "$CACHE_DIR"
fi
DEEPSPEC_FAST_PY="$DST/bin/python"
if [[ ! -x "${DEEPSPEC_FAST_PY}" ]]; then
    echo "Environment Python not found: ${DEEPSPEC_FAST_PY}" >&2
    exit 1
fi
echo "环境准备完成：$DST"
PYTHON_BIN="$DEEPSPEC_FAST_PY" \
VLLM_PYTHON_BIN="$DEEPSPEC_FAST_PY" \
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP:-1}}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/dspark_qwen3_8_27b_vllm}" \
exec bash scripts/train/train_qwen3_8_27b_dspark_vllm.sh

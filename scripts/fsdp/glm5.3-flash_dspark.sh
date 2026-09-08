#!/usr/bin/env bash

SRC=/mnt/afs-agentpro/lezewei/DeepSpec/envs/deepspec_vllm_torchtitan_envs.tar

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
OUTPUT_ROOT="$PWD/output/glm5_3_flash_dspark_fsdp2_output_$(date +%Y%m%d)" \
bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh

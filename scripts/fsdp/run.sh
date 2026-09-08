source env.sh
DEEPSPEC_FAST_PY="${DEEPSPEC_FAST_PY:-/tmp/deepspec-env-cache/deepspec_vllm_torchtitan_envs-20260907/bin/python}"
if [[ ! -x "$DEEPSPEC_FAST_PY" ]]; then
    DEEPSPEC_FAST_PY="$(command -v python)"
fi

PYTHON_BIN="$DEEPSPEC_FAST_PY" VLLM_PYTHON_BIN="$DEEPSPEC_FAST_PY" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/output/glm5_3_flash_dspark_fsdp2_localenv4}" \
bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh

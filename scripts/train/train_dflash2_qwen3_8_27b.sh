#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
source /mnt/afs-agentpro/yangbo1/ms-swift/env.sh
gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
context_parallel_size="${CONTEXT_PARALLEL_SIZE:-2}"
max_context_length="${MAX_CONTEXT_LENGTH:-262144}"
global_batch_size="${GLOBAL_BATCH_SIZE:-512}"
run_variant="${RUN_VARIANT:-}"
run_post_train_eval="${RUN_POST_TRAIN_EVAL:-1}"
eval_max_samples="${EVAL_MAX_SAMPLES:-64}"
eval_max_new_tokens="${EVAL_MAX_NEW_TOKENS:-512}"
min_acceptance_length="${MIN_ACCEPTANCE_LENGTH:-2.0}"
default_train_data_path="${repo_root}/train_data/spec_o3_coldstartsft.repeat60.deepspec.jsonl"
train_data_path="${TRAIN_DATA_PATH:-${default_train_data_path}}"
cache_parent="${repo_root}/output/dflash2_qwen3_8_27b_target_cache"

if [[ ! "${context_parallel_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONTEXT_PARALLEL_SIZE must be a positive integer." >&2
    exit 2
fi
if [[ ! "${max_context_length}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONTEXT_LENGTH must be a positive integer." >&2
    exit 2
fi
if [[ ! "${global_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GLOBAL_BATCH_SIZE must be a positive integer." >&2
    exit 2
fi
if [[ "${run_post_train_eval}" != "0" && "${run_post_train_eval}" != "1" ]]; then
    echo "RUN_POST_TRAIN_EVAL must be 0 or 1." >&2
    exit 2
fi
if [[ ! "${eval_max_samples}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_MAX_SAMPLES must be a positive integer." >&2
    exit 2
fi
if [[ ! "${eval_max_new_tokens}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_MAX_NEW_TOKENS must be a positive integer." >&2
    exit 2
fi
if [[ -n "${run_variant}" && ! "${run_variant}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_VARIANT contains unsupported path characters." >&2
    exit 2
fi
if (( max_context_length > 262144 )); then
    echo "MAX_CONTEXT_LENGTH cannot exceed the model limit of 262144." >&2
    exit 2
fi

IFS=',' read -r -a visible_gpus <<< "${gpu_devices}"
world_size="${#visible_gpus[@]}"
if (( world_size % context_parallel_size != 0 )); then
    echo "GPU count (${world_size}) must be divisible by CP (${context_parallel_size})." >&2
    exit 2
fi
fsdp_size="$((world_size / context_parallel_size))"

if [[ ! -f "${train_data_path}" ]]; then
    echo "Training data does not exist: ${train_data_path}" >&2
    exit 2
fi

train_data_sha256="$(sha256sum -- "${train_data_path}" | awk '{print $1}')"
if [[ ! "${train_data_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Unable to compute a stable SHA-256 for: ${train_data_path}" >&2
    exit 2
fi
data_identity="${train_data_sha256:0:12}"
artifact_name="ctx${max_context_length}_${data_identity}"
if [[ -n "${run_variant}" ]]; then
    artifact_name="${run_variant}_${artifact_name}"
fi
target_cache_dir="${cache_parent}/${artifact_name}_cp${context_parallel_size}"
checkpoint_dir="${repo_root}/output/dflash2_qwen3_8_27b_${artifact_name}_checkpoints"
tensorboard_dir="${checkpoint_dir}/tensorboard"

if [[ -L "${target_cache_dir}" ]]; then
    echo "Refusing to use a symlink as the disposable cache directory: ${target_cache_dir}" >&2
    exit 2
fi

if [[ -f "${target_cache_dir}/manifest.json" ]]; then
    if ! python scripts/data/validate_target_cache.py \
        --cache-dir "${target_cache_dir}" \
        --source-jsonl-path "${train_data_path}" \
        --target-model-name-or-path "/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B" \
        --target-layer-ids "5,19,33,47,61" \
        --chat-template qwen \
        --max-length "${max_context_length}" \
        --context-parallel-size "${context_parallel_size}" \
        --stores-target-last-hidden-states false; then
        echo "Removing stale or incompatible disposable target cache: ${target_cache_dir}"
        rm -rf -- "${target_cache_dir}"
    fi
fi

if [[ -d "${target_cache_dir}" && ! -f "${target_cache_dir}/manifest.json" ]]; then
    echo "Removing incomplete target cache: ${target_cache_dir}"
    rm -rf -- "${target_cache_dir}"
fi

if [[ ! -f "${target_cache_dir}/manifest.json" ]]; then
    mkdir -p "${cache_parent}"
    echo "Preparing offline Qwen3.8 target cache " \
        "(max_length=${max_context_length}, CP=${context_parallel_size}, " \
        "FSDP=${fsdp_size})..."
    CUDA_VISIBLE_DEVICES="${gpu_devices}" python scripts/data/prepare_target_cache.py \
        --config config/dflash2/dflash2_qwen3_8_27b.py \
        --opts "data.max_length=${max_context_length}" \
        --train-data-path "${train_data_path}" \
        --output-dir "${target_cache_dir}" \
        --local-batch-size 1 \
        --num-workers 1 \
        --fsdp \
        --fsdp-size "${fsdp_size}" \
        --context-parallel-size "${context_parallel_size}"
else
    echo "Reusing completed target cache: ${target_cache_dir}"
fi

echo "Training Qwen3.8-27B DFlash2 from the offline target cache..."
echo "Checkpoints will be saved to: ${checkpoint_dir}"
CUDA_VISIBLE_DEVICES="${gpu_devices}" python train.py \
    --config config/dflash2/dflash2_qwen3_8_27b.py \
    --opts "data.target_cache_path=${target_cache_dir}" \
    --opts "data.max_length=${max_context_length}" \
    --opts "train.context_parallel_size=${context_parallel_size}" \
    --opts "train.fsdp_size=${fsdp_size}" \
    --opts "train.global_batch_size=${global_batch_size}" \
    --opts "data.source_jsonl_path=${train_data_path}" \
    --opts "logging.checkpoint_dir=${checkpoint_dir}" \
    --opts "logging.tensorboard_dir=${tensorboard_dir}"

# Consume the large hidden-state cache only after a clean training exit.
expected_cache_dir="${repo_root}/output/dflash2_qwen3_8_27b_target_cache"
expected_cache_dir="${expected_cache_dir}/${artifact_name}_cp${context_parallel_size}"
if [[ "${target_cache_dir}" != "${expected_cache_dir}" ]]; then
    echo "Refusing to delete an unexpected cache path: ${target_cache_dir}" >&2
    exit 2
fi
echo "Training completed; deleting consumed target cache: ${target_cache_dir}"
rm -rf -- "${target_cache_dir}"
rmdir "${cache_parent}" 2>/dev/null || true

if [[ "${run_post_train_eval}" == "1" ]]; then
    latest_checkpoint="${checkpoint_dir}/step_latest"
    if [[ ! -e "${latest_checkpoint}" ]]; then
        echo "Missing final checkpoint for acceptance evaluation: ${latest_checkpoint}" >&2
        exit 2
    fi
    eval_dir="${checkpoint_dir}/acceptance_eval"
    mkdir -p "${eval_dir}"
    for temperature in 0 1; do
        results_json="${eval_dir}/temperature_${temperature}.json"
        CUDA_VISIBLE_DEVICES="${gpu_devices}" python eval.py \
            --target_name_or_path "/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B" \
            --draft_name_or_path "${latest_checkpoint}" \
            --temperature "${temperature}" \
            --max-new-tokens "${eval_max_new_tokens}" \
            --results-json "${results_json}" \
            --task "gsm8k:${eval_max_samples}" \
            --task "math500:${eval_max_samples}" \
            --task "mbpp:${eval_max_samples}" \
            --task "mt-bench:${eval_max_samples}"
    done
    python scripts/eval/check_acceptance_results.py \
        --results-json "${eval_dir}/temperature_0.json" \
        --results-json "${eval_dir}/temperature_1.json" \
        --min-average-acceptance-length "${min_acceptance_length}"
fi
echo "DFlash2 checkpoints are available at: ${checkpoint_dir}"

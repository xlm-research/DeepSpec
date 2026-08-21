#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
source /mnt/afs-agentpro/yangbo1/ms-swift/env.sh

source_data_path="${SOURCE_TRAIN_DATA_PATH:-${repo_root}/train_data/spec_o3_coldstartsft.repeat60.deepspec.jsonl}"
packed_data_path="${PACKED_TRAIN_DATA_PATH:-${repo_root}/train_data/spec_o3_coldstartsft.repeat60.deepspec.packed_256k.jsonl}"
target_model_path="/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B"
allow_low_diversity="${ALLOW_LOW_DIVERSITY_DATA:-0}"
min_unique_records="${MIN_UNIQUE_TRAIN_RECORDS:-10000}"
min_unique_ratio="${MIN_UNIQUE_TRAIN_RATIO:-0.25}"
packed_global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
force_repack="${FORCE_REPACK_PACKED_DATA:-0}"

if [[ ! -f "${source_data_path}" ]]; then
    echo "Source training data does not exist: ${source_data_path}" >&2
    exit 2
fi

if [[ "${allow_low_diversity}" != "0" && "${allow_low_diversity}" != "1" ]]; then
    echo "ALLOW_LOW_DIVERSITY_DATA must be 0 or 1." >&2
    exit 2
fi
if [[ ! "${packed_global_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GLOBAL_BATCH_SIZE must be a positive integer." >&2
    exit 2
fi
if [[ "${force_repack}" != "0" && "${force_repack}" != "1" ]]; then
    echo "FORCE_REPACK_PACKED_DATA must be 0 or 1." >&2
    exit 2
fi
quality_args=(
    --input-path "${source_data_path}"
    --min-unique-records "${min_unique_records}"
    --min-unique-ratio "${min_unique_ratio}"
)
if [[ "${allow_low_diversity}" == "1" ]]; then
    quality_args+=(--allow-low-diversity)
fi
python scripts/data/check_dataset_quality.py "${quality_args[@]}"

needs_pack=0
if [[ ! -f "${packed_data_path}" || "${force_repack}" == "1" ]]; then
    needs_pack=1
elif ! python scripts/data/validate_packed_dataset.py \
    --source-path "${source_data_path}" \
    --packed-path "${packed_data_path}" \
    --model-name-or-path "${target_model_path}" \
    --chat-template qwen \
    --target-length 262144; then
    echo "Refusing to reuse packed data with missing or stale provenance." >&2
    echo "Set FORCE_REPACK_PACKED_DATA=1 to rebuild this derived artifact." >&2
    exit 2
fi

if [[ "${needs_pack}" == "1" ]]; then
    echo "Building continuously packed 256K training data: ${packed_data_path}"
    pack_args=(
        --input-path "${source_data_path}"
        --output-path "${packed_data_path}"
        --model-name-or-path "${target_model_path}"
        --chat-template qwen
        --target-length 262144
    )
    if [[ -f "${packed_data_path}" ]]; then
        pack_args+=(--force)
    fi
    python scripts/data/pack_conversations.py "${pack_args[@]}"
else
    echo "Reusing packed 256K training data: ${packed_data_path}"
fi

packed_records="$(wc -l < "${packed_data_path}")"
if (( packed_records < packed_global_batch_size )); then
    echo "Packed data is too small for GLOBAL_BATCH_SIZE=${packed_global_batch_size}: ${packed_records} records." >&2
    exit 2
fi
echo "Packed training schedule: $((packed_records / packed_global_batch_size)) optimizer steps per epoch."

TRAIN_DATA_PATH="${packed_data_path}" \
MAX_CONTEXT_LENGTH=262144 \
GLOBAL_BATCH_SIZE="${packed_global_batch_size}" \
RUN_VARIANT=packed256k \
bash scripts/train/train_dflash2_qwen3_8_27b.sh

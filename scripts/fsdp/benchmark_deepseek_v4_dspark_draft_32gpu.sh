#!/usr/bin/env bash
set -eo pipefail

# Benchmark only the trainable DSpark draft model on 4 nodes x 8 GPUs. The
# frozen target model is not constructed and no target-feature cache is used.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BASE_DIR}"

SCHEDULER_WORLD_SIZE=${WORLD_SIZE:-}
SCHEDULER_RANK=${RANK:-}
NNODES=${NNODES:-${SENSECORE_PYTORCH_NNODES:-${SCHEDULER_WORLD_SIZE:-4}}}
NODE_RANK=${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${SCHEDULER_RANK:-0}}}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29661}

TARGET_CONFIG=${TARGET_CONFIG:-/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731}
BENCH_OUTPUT_ROOT=${BENCH_OUTPUT_ROOT:-${BASE_DIR}/output/dspark_32gpu_draft_speed_test}
RESULT_JSON=${RESULT_JSON:-${BENCH_OUTPUT_ROOT}/result.json}
LOG_DIR=${LOG_DIR:-${BENCH_OUTPUT_ROOT}/logs}

SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-131072}
NUM_ANCHORS=${NUM_ANCHORS:-512}
BLOCK_SIZE=${BLOCK_SIZE:-7}
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS:-3}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
LOCAL_BATCH_SIZE=1
DATASET_SIZE=${DATASET_SIZE:-713144}
WARMUP_STEPS=${WARMUP_STEPS:-2}
MEASURE_STEPS=${MEASURE_STEPS:-7}

DP_SHARD=1
CP=8
TP=1

for integer_var in \
    NNODES NODE_RANK NPROC_PER_NODE SEQUENCE_LENGTH NUM_ANCHORS BLOCK_SIZE \
    NUM_DRAFT_LAYERS GLOBAL_BATCH_SIZE DATASET_SIZE WARMUP_STEPS MEASURE_STEPS; do
    integer_value=${!integer_var}
    if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
        echo "${integer_var} must be a non-negative integer; got ${integer_value}." >&2
        exit 1
    fi
done

if ((NNODES < 1 || NPROC_PER_NODE < 1)); then
    echo "NNODES and NPROC_PER_NODE must be positive." >&2
    exit 1
fi
if ((NODE_RANK >= NNODES)); then
    echo "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}." >&2
    exit 1
fi

TRAIN_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if ((TRAIN_WORLD_SIZE != 32)); then
    echo "This benchmark requires 32 GPUs; got NNODES*NPROC_PER_NODE=${TRAIN_WORLD_SIZE}." >&2
    echo "For SenseCore, request 4 nodes x 8 GPUs or set NNODES/NODE_RANK explicitly." >&2
    exit 1
fi
if ((NNODES > 1)) && [[ "${MASTER_ADDR}" == "localhost" || "${MASTER_ADDR}" == "127.0.0.1" ]]; then
    echo "A cross-node reachable MASTER_ADDR is required for NNODES=${NNODES}." >&2
    exit 1
fi
if [[ ! -e "${TARGET_CONFIG}" ]]; then
    echo "Target config path does not exist: ${TARGET_CONFIG}" >&2
    exit 1
fi
if ((SEQUENCE_LENGTH % CP != 0)); then
    echo "SEQUENCE_LENGTH=${SEQUENCE_LENGTH} must be divisible by CP=${CP}." >&2
    exit 1
fi

DENSE_REPLICA_DOMAIN=$((DP_SHARD * CP * TP))
DP_REPLICATE=$((TRAIN_WORLD_SIZE / DENSE_REPLICA_DOMAIN))
EFFECTIVE_DATA_BATCH=$((DP_REPLICATE * DP_SHARD * LOCAL_BATCH_SIZE))
if ((GLOBAL_BATCH_SIZE % EFFECTIVE_DATA_BATCH != 0)); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by ${EFFECTIVE_DATA_BATCH}." >&2
    exit 1
fi
ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / EFFECTIVE_DATA_BATCH))
USABLE_SAMPLES=$((DATASET_SIZE / GLOBAL_BATCH_SIZE * GLOBAL_BATCH_SIZE))
FULL_DATASET_STEPS=$((USABLE_SAMPLES / GLOBAL_BATCH_SIZE))

mkdir -p "${LOG_DIR}"
NODE_LOG=${LOG_DIR}/node_rank_${NODE_RANK}.log

echo "Launching 32-GPU DeepSeek-V4 DSpark draft-only benchmark:"
echo "  node=${NODE_RANK}/${NNODES}, rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
echo "  topology=DP_REPLICATE ${DP_REPLICATE}, DP_SHARD ${DP_SHARD}, CP ${CP}, TP ${TP}"
echo "  global batch=${GLOBAL_BATCH_SIZE}, accumulation=${ACCUMULATION_STEPS}"
echo "  sequence length=${SEQUENCE_LENGTH}, anchors=${NUM_ANCHORS}, block size=${BLOCK_SIZE}"
echo "  warmup=${WARMUP_STEPS}, measured steps=${MEASURE_STEPS}"
echo "  dataset=${DATASET_SIZE}, usable samples=${USABLE_SAMPLES}, optimizer steps=${FULL_DATASET_STEPS}"
echo "  result=${RESULT_JSON}"
echo "  target model is not constructed; target inference and disk caching are excluded"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/fsdp/benchmark_deepseek_v4_dspark_draft_overlap.py \
    --target-config "${TARGET_CONFIG}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --num-anchors "${NUM_ANCHORS}" \
    --block-size "${BLOCK_SIZE}" \
    --num-draft-layers "${NUM_DRAFT_LAYERS}" \
    --accumulation-steps "${ACCUMULATION_STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --measure-steps "${MEASURE_STEPS}" \
    --dp-replicate "${DP_REPLICATE}" \
    --dp-shard "${DP_SHARD}" \
    --cp "${CP}" \
    --tp "${TP}" \
    --no-reshard-after-forward \
    --forward-prefetch \
    --backward-prefetch \
    --prefetch-depth 2 \
    --reduce-dtype bf16 \
    --wrap-granularity block \
    --no-last-backward-hint \
    --output-json "${RESULT_JSON}" \
    2>&1 | tee "${NODE_LOG}"

if ((NODE_RANK == 0)); then
    RESULT_JSON="${RESULT_JSON}" \
    DATASET_SIZE="${DATASET_SIZE}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    python3 - <<'PY'
import json
import os

result_path = os.environ["RESULT_JSON"]
dataset_size = int(os.environ["DATASET_SIZE"])
global_batch_size = int(os.environ["GLOBAL_BATCH_SIZE"])

with open(result_path, encoding="utf-8") as handle:
    result = json.load(handle)

usable_samples = dataset_size // global_batch_size * global_batch_size
optimizer_steps = usable_samples // global_batch_size
median_step_s = float(result["median_step_time_ms_max_rank"]) / 1000.0
p95_step_s = float(result["p95_step_time_ms_max_rank"]) / 1000.0
median_hours = optimizer_steps * median_step_s / 3600.0
p95_hours = optimizer_steps * p95_step_s / 3600.0
allocated_gib = int(result["peak_memory_allocated_bytes_max_rank"]) / 2**30
reserved_gib = int(result["peak_memory_reserved_bytes_max_rank"]) / 2**30

print("\n===== Full-dataset draft-only estimate =====")
print(f"Usable samples: {usable_samples:,}")
print(f"Optimizer steps: {optimizer_steps:,}")
print(f"Median step: {median_step_s:.3f} s")
print(f"P95 step: {p95_step_s:.3f} s")
print(f"Median draft throughput: {result['draft_tokens_per_second_median']:,.1f} token/s")
print(f"Peak allocated/reserved: {allocated_gib:.2f}/{reserved_gib:.2f} GiB")
print(f"Full draft training ETA (median): {median_hours:.2f} h")
print(f"Full draft training ETA (P95): {p95_hours:.2f} h")
print("Scope: draft forward/backward/FSDP/optimizer only; excludes target inference, cache I/O, and checkpoints.")
PY
fi

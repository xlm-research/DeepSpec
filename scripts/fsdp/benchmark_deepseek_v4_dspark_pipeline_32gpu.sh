#!/usr/bin/env bash
set -eo pipefail

# Measure the real serialized pipeline on 4 nodes x 8 GPUs:
# target inference -> disk cache -> draft training -> cache deletion.
# A bounded number of optimizer steps is used, and rank 0 projects the measured
# phase times onto one full epoch of the configured dataset.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BASE_DIR}"

SCHEDULER_WORLD_SIZE=${WORLD_SIZE:-}
SCHEDULER_RANK=${RANK:-}
BENCH_NNODES=${NNODES:-${SENSECORE_PYTORCH_NNODES:-${SCHEDULER_WORLD_SIZE:-4}}}
BENCH_NODE_RANK=${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${SCHEDULER_RANK:-0}}}
BENCH_NPROC_PER_NODE=${NPROC_PER_NODE:-8}
BENCH_MASTER_ADDR=${MASTER_ADDR:-localhost}
BENCH_MASTER_PORT=${MASTER_PORT:-29662}

BENCH_OUTPUT_ROOT=${BENCH_OUTPUT_ROOT:-${BASE_DIR}/output/dspark_32gpu_pipeline_speed_test}
ESTIMATE_JSON=${ESTIMATE_JSON:-${BENCH_OUTPUT_ROOT}/pipeline_estimate.json}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/mnt/afs-agentpro/hongjiawei/code/DeepSpec-old/scripts/data/sensenova_flash_v15_use_data_json.thinking_toolcall.readable_full.flash_conversations.jsonl}
JSONL_INDEX_CACHE_DIR=${JSONL_INDEX_CACHE_DIR:-${BASE_DIR}/output/jsonl_index_cache}

MAX_LENGTH=${MAX_LENGTH:-131072}
NUM_ANCHORS=${NUM_ANCHORS:-512}
BLOCK_SIZE=${BLOCK_SIZE:-7}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
BENCHMARK_STEPS=${BENCHMARK_STEPS:-16}
BENCHMARK_DATA_BATCHES=${BENCHMARK_DATA_BATCHES:-4}

for integer_var in \
    BENCH_NNODES BENCH_NODE_RANK BENCH_NPROC_PER_NODE MAX_LENGTH NUM_ANCHORS BLOCK_SIZE \
    GLOBAL_BATCH_SIZE BENCHMARK_STEPS BENCHMARK_DATA_BATCHES; do
    integer_value=${!integer_var}
    if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
        echo "${integer_var} must be a non-negative integer; got ${integer_value}." >&2
        exit 1
    fi
done

if ((BENCH_NNODES < 1 || BENCH_NPROC_PER_NODE < 1)); then
    echo "BENCH_NNODES and BENCH_NPROC_PER_NODE must be positive." >&2
    exit 1
fi
if ((BENCH_NODE_RANK >= BENCH_NNODES)); then
    echo "BENCH_NODE_RANK=${BENCH_NODE_RANK} must be smaller than BENCH_NNODES=${BENCH_NNODES}." >&2
    exit 1
fi

TRAIN_WORLD_SIZE=$((BENCH_NNODES * BENCH_NPROC_PER_NODE))
if ((TRAIN_WORLD_SIZE != 32)); then
    echo "This benchmark requires 32 GPUs; got ${TRAIN_WORLD_SIZE}." >&2
    echo "For SenseCore, request 4 nodes x 8 GPUs or set NNODES/NODE_RANK explicitly." >&2
    exit 1
fi
if ((BENCH_NNODES > 1)) && [[ "${BENCH_MASTER_ADDR}" == "localhost" || "${BENCH_MASTER_ADDR}" == "127.0.0.1" ]]; then
    echo "A cross-node reachable MASTER_ADDR is required." >&2
    exit 1
fi
if ((BENCHMARK_STEPS < 1 || BENCHMARK_DATA_BATCHES < 1)); then
    echo "BENCHMARK_STEPS and BENCHMARK_DATA_BATCHES must be positive." >&2
    exit 1
fi
if ((BENCHMARK_DATA_BATCHES > BENCHMARK_STEPS)); then
    echo "BENCHMARK_DATA_BATCHES cannot exceed BENCHMARK_STEPS." >&2
    exit 1
fi
if [[ ! -e "${TARGET_MODEL_PATH}" ]]; then
    echo "Target model path does not exist: ${TARGET_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_DATA_PATH}" ]]; then
    echo "Training JSONL does not exist: ${TRAIN_DATA_PATH}" >&2
    exit 1
fi

BENCHMARK_GLOBAL_SAMPLES=$((BENCHMARK_STEPS * GLOBAL_BATCH_SIZE))
echo "Launching real 32-GPU target+draft pipeline benchmark:"
echo "  node=${BENCH_NODE_RANK}/${BENCH_NNODES}, rendezvous=${BENCH_MASTER_ADDR}:${BENCH_MASTER_PORT}"
echo "  benchmark optimizer steps=${BENCHMARK_STEPS}, global samples=${BENCHMARK_GLOBAL_SAMPLES}"
echo "  data batches=${BENCHMARK_DATA_BATCHES}, global batch=${GLOBAL_BATCH_SIZE}"
echo "  max length=${MAX_LENGTH}, anchors=${NUM_ANCHORS}, block size=${BLOCK_SIZE}"
echo "  output=${BENCH_OUTPUT_ROOT}"
echo "  checkpoints are disabled; successful blocks delete their transient cache"

SENSECORE_PYTORCH_NNODES="${BENCH_NNODES}" \
SENSECORE_PYTORCH_NODE_RANK="${BENCH_NODE_RANK}" \
NPROC_PER_NODE="${BENCH_NPROC_PER_NODE}" \
MASTER_ADDR="${BENCH_MASTER_ADDR}" \
MASTER_PORT="${BENCH_MASTER_PORT}" \
OUTPUT_ROOT="${BENCH_OUTPUT_ROOT}" \
TARGET_MODEL_PATH="${TARGET_MODEL_PATH}" \
TRAIN_DATA_PATH="${TRAIN_DATA_PATH}" \
JSONL_INDEX_CACHE_DIR="${JSONL_INDEX_CACHE_DIR}" \
DATA_BATCH_CACHE_DIR="${BENCH_OUTPUT_ROOT}/target_data_batch_cache" \
MAX_LENGTH="${MAX_LENGTH}" \
NUM_ANCHORS="${NUM_ANCHORS}" \
BLOCK_SIZE="${BLOCK_SIZE}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
DATA_BATCH_SIZE="${BENCHMARK_DATA_BATCHES}" \
MAX_TRAIN_STEPS="${BENCHMARK_STEPS}" \
SAVE_CHECKPOINTS=false \
PROFILE_ENABLED=false \
PRODUCTION_RUN=false \
TORCHRUN_PER_RANK_LOGS=true \
RESHARD_AFTER_FORWARD=false \
FSDP_FORWARD_PREFETCH=true \
FSDP_BACKWARD_PREFETCH=true \
FSDP_PREFETCH_DEPTH=2 \
FSDP_REDUCE_DTYPE=bf16 \
FSDP_WRAP_GRANULARITY=block \
bash scripts/fsdp/train_deepseek_v4_flash_dspark_fsdp2_multinode_128.sh

if ((BENCH_NODE_RANK == 0)); then
    NODE_LOG=${BENCH_OUTPUT_ROOT}/logs/node_rank_0.log
    NODE_LOG="${NODE_LOG}" \
    ESTIMATE_JSON="${ESTIMATE_JSON}" \
    EXPECTED_DATA_BATCHES="${BENCHMARK_DATA_BATCHES}" \
    EXPECTED_BENCHMARK_STEPS="${BENCHMARK_STEPS}" \
    NUM_ANCHORS="${NUM_ANCHORS}" \
    BLOCK_SIZE="${BLOCK_SIZE}" \
    python3 - <<'PY'
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

node_log = Path(os.environ["NODE_LOG"])
estimate_path = Path(os.environ["ESTIMATE_JSON"])
expected_batches = int(os.environ["EXPECTED_DATA_BATCHES"])
expected_benchmark_steps = int(os.environ["EXPECTED_BENCHMARK_STEPS"])
num_anchors = int(os.environ["NUM_ANCHORS"])
block_size = int(os.environ["BLOCK_SIZE"])

lines = node_log.read_text(encoding="utf-8", errors="replace").splitlines()
timestamp_pattern = r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
start_pattern = re.compile(
    timestamp_pattern
    + rf" Data batch (?P<index>\d+)/{expected_batches}: starting isolated target inference"
)
ready_pattern = re.compile(
    timestamp_pattern
    + r" Target data batch (?P<index>\d+) ready: global_samples=(?P<samples>\d+);"
)
finish_pattern = re.compile(
    timestamp_pattern
    + r" Data batch (?P<index>\d+) draft training finished;"
)

last_run_start = None
for line_index, line in enumerate(lines):
    match = start_pattern.search(line)
    if match and int(match.group("index")) == 1:
        last_run_start = line_index
if last_run_start is None:
    raise RuntimeError(f"Could not find the latest data-batch start in {node_log}")

events: dict[str, dict[int, tuple[datetime, int | None]]] = {
    "start": {},
    "ready": {},
    "finish": {},
}
for line in lines[last_run_start:]:
    for event_name, pattern in (
        ("start", start_pattern),
        ("ready", ready_pattern),
        ("finish", finish_pattern),
    ):
        match = pattern.search(line)
        if not match:
            continue
        index = int(match.group("index"))
        if index not in events[event_name]:
            timestamp = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
            samples = int(match.group("samples")) if event_name == "ready" else None
            events[event_name][index] = (timestamp, samples)
        break

expected_indices = set(range(1, expected_batches + 1))
for event_name, event_values in events.items():
    if set(event_values) != expected_indices:
        raise RuntimeError(
            f"Incomplete {event_name} events: expected={sorted(expected_indices)}, "
            f"actual={sorted(event_values)}"
        )

target_seconds_by_batch = []
draft_seconds_by_batch = []
samples_by_batch = []
for index in sorted(expected_indices):
    start_time = events["start"][index][0]
    ready_time, global_samples = events["ready"][index]
    finish_time = events["finish"][index][0]
    target_seconds_by_batch.append((ready_time - start_time).total_seconds())
    draft_seconds_by_batch.append((finish_time - ready_time).total_seconds())
    assert global_samples is not None
    samples_by_batch.append(global_samples)

info_lines = lines[:last_run_start]
def last_integer(pattern: str) -> int:
    matcher = re.compile(pattern)
    values = [int(match.group(1)) for line in info_lines if (match := matcher.search(line))]
    if not values:
        raise RuntimeError(f"Missing log value for pattern: {pattern}")
    return values[-1]

samples_per_epoch = last_integer(r"Samples per epoch = (\d+)")
global_batch_size = last_integer(r"Global batch size = (\d+)")
max_train_steps = last_integer(r"Max train steps = (\d+)")
if max_train_steps != expected_benchmark_steps:
    raise RuntimeError(
        f"Benchmark step mismatch: log={max_train_steps}, expected={expected_benchmark_steps}"
    )

benchmark_samples = sum(samples_by_batch)
expected_samples = expected_benchmark_steps * global_batch_size
if benchmark_samples != expected_samples:
    raise RuntimeError(
        f"Benchmark sample mismatch: log={benchmark_samples}, expected={expected_samples}"
    )

target_seconds = sum(target_seconds_by_batch)
draft_seconds = sum(draft_seconds_by_batch)
full_optimizer_steps = samples_per_epoch // global_batch_size
target_eta_seconds = target_seconds * samples_per_epoch / benchmark_samples
draft_eta_seconds = draft_seconds * full_optimizer_steps / expected_benchmark_steps
total_eta_seconds = target_eta_seconds + draft_eta_seconds
draft_tokens_per_step = num_anchors * block_size * global_batch_size

result = {
    "scope": "real target inference/cache plus real draft training/cache cleanup",
    "benchmark": {
        "data_batches": expected_batches,
        "optimizer_steps": expected_benchmark_steps,
        "global_samples": benchmark_samples,
        "target_seconds": target_seconds,
        "draft_seconds": draft_seconds,
        "pipeline_seconds": target_seconds + draft_seconds,
        "target_seconds_by_batch": target_seconds_by_batch,
        "draft_seconds_by_batch": draft_seconds_by_batch,
        "global_samples_by_batch": samples_by_batch,
        "target_samples_per_second": benchmark_samples / target_seconds,
        "draft_steps_per_second": expected_benchmark_steps / draft_seconds,
        "draft_tokens_per_second": (
            expected_benchmark_steps * draft_tokens_per_step / draft_seconds
        ),
    },
    "full_dataset_estimate": {
        "samples": samples_per_epoch,
        "optimizer_steps": full_optimizer_steps,
        "target_hours": target_eta_seconds / 3600.0,
        "draft_hours": draft_eta_seconds / 3600.0,
        "total_hours": total_eta_seconds / 3600.0,
        "total_days": total_eta_seconds / 86400.0,
        "excludes": [
            "one-time model initialization",
            "checkpoint saving",
            "scheduler startup and shutdown",
        ],
    },
}

estimate_path.parent.mkdir(parents=True, exist_ok=True)
estimate_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("\n===== 32-GPU real pipeline measurement =====")
print(f"Measured samples/steps: {benchmark_samples:,}/{expected_benchmark_steps}")
print(f"Target phase: {target_seconds:.1f} s, {benchmark_samples / target_seconds:.3f} sample/s")
print(f"Draft phase: {draft_seconds:.1f} s, {expected_benchmark_steps / draft_seconds:.3f} step/s")
print(
    "Draft throughput: "
    f"{expected_benchmark_steps * draft_tokens_per_step / draft_seconds:,.1f} token/s"
)
print(f"Full target ETA: {target_eta_seconds / 3600.0:.2f} h")
print(f"Full draft ETA: {draft_eta_seconds / 3600.0:.2f} h")
print(f"Full current-method ETA: {total_eta_seconds / 3600.0:.2f} h ({total_eta_seconds / 86400.0:.2f} days)")
print("The total excludes one-time initialization, checkpoint writes, and scheduler overhead.")
print(f"Result JSON: {estimate_path}")
PY

    CACHE_ROOT=${BENCH_OUTPUT_ROOT}/target_data_batch_cache
    if [[ -d "${CACHE_ROOT}" ]]; then
        CACHE_FILE_COUNT=$(find "${CACHE_ROOT}" -type f | wc -l)
        if ((CACHE_FILE_COUNT != 0)); then
            echo "Unexpected cache residue after a successful benchmark: ${CACHE_FILE_COUNT} files" >&2
            exit 1
        fi
    fi
fi

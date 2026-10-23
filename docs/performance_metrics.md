# Performance Metrics

TorchSpec includes a performance metrics system designed to identify
pipeline bottlenecks between the asynchronous inference and synchronous training
stages.

## Enabling

Set `debug.enable_perf_metrics: true` in your YAML config, or pass the CLI
override:

```bash
python train.py --config configs/default.yaml debug.enable_perf_metrics=true
```

Performance metrics are enabled by default. Set `debug.enable_perf_metrics: false`
to disable the timing and throughput measurements.

## Metrics Reference

All metrics live under the `perf/` namespace in W&B and TensorBoard (tied to
`train/step`). Full-data runs also retain them in `metrics.jsonl`.

### Training Metrics (per optimizer step)

| Metric | Unit | Source | Description |
|---|---|---|---|
| `perf/draft_train_time` | seconds | `trainer.py` + `loop.py` | Controller dispatch wait plus the maximum trainer wall-clock duration across training ranks, from starting to wait for the first batch to GPU completion of the optimizer update, across all gradient-accumulation micro-batches. |
| `perf/train_tokens` | tokens | `data/utils.py` + `trainer.py` + `loop.py` | Total input tokens across all training ranks and accumulation micro-batches, excluding padding, dummy samples, and USP lookahead overlap. Includes context tokens regardless of the loss mask. |
| `perf/train_gpus` | GPUs | `loop.py` | Number of draft training actors/GPUs; target inference GPUs are excluded. |
| `perf/train_tgs` | tokens/GPU/s | `loop.py` | `train_tokens / (draft_train_time * train_gpus)`, including dispatch and data waits. Omitted if the measured duration is zero. |
| `perf/step_time` | seconds | `trainer.py` | Trainer step duration, covering data fetch + compute + optimizer step. Immediate metrics use wall-clock time after aggregation synchronizes CUDA; deferred metrics use CUDA events spanning the trainer step. |
| `perf/data_time` | seconds | `trainer.py` | Wall-clock time spent in the data iterator (Ray queue get + Mooncake RDMA fetch + collation + H2D copy). Measured with `time.time()` since this is CPU/network-bound work. |
| `perf/wait_time` | seconds | `loop.py` | `dispatch_wait + data_time`: controller dispatch waiting plus the trainer's data-fetch time for the same optimizer step. |
| `perf/compute_time` | seconds | `trainer.py` | CUDA-event duration of forward + backward + optimizer, including communication waits on that execution path. Read after the existing immediate or deferred metric synchronization; no additional synchronization is introduced for the totals. |
| `perf/total_wait_time` | seconds | `loop.py` | Cumulative `wait_time` over reported optimizer steps in this invocation. |
| `perf/total_compute_time` | seconds | `loop.py` | Cumulative `compute_time` over reported optimizer steps in this invocation. |
| `perf/train_capacity` | samples/s | `loop.py` | `global_batch_size / step_time`. The effective training throughput if data were always available. |

Use `draft_train_time` to compare draft training speed at the same batch size,
sequence lengths, accumulation count, and hardware, after warmup. The trainer
starts its clock **before requesting the first batch**, with or without CPU
prefetch. Its interval includes initial and later batch waits, data loading,
H2D, CPU preparation, forward/loss, backward, gradient communication, and the
optimizer update (including copying updated parameters back to the model).
The controller adds that optimizer step's `dispatch_wait` before reporting the
metric, including when trainer metrics are deferred or flushed at shutdown.
Do not add `data_time` or `wait_time` again: the trainer interval already includes
data fetching, which may overlap GPU work.

Metric aggregation, logging, evaluation, and checkpoint saving are outside the
measured interval. This is one duration per optimizer update, not a cumulative
total or pure GPU compute time. Keep the prefetch configuration fixed when
comparing runs. The controller uses the longest interval across training ranks.

`train_tgs` measures input-token training throughput, not decoded output or sampled
DSpark prediction positions. Token counts are captured before batch padding using
CPU metadata, accumulated per optimizer step, and summed from the existing Ray
actor results. USP ranks count only their own sequence chunk, excluding overlapping
lookahead tokens. Counts and durations stay paired when metrics are deferred or
flushed at shutdown. This adds no GPU collective or per-micro-batch host sync.
The denominator includes data waits, so TGS reflects the training pipeline's
observed throughput. W&B, TensorBoard, and full-data `metrics.jsonl` receive the
same values as the `TIMING` log and progress bar.

An independent timing thread waits for the CUDA event immediately after the
optimizer and records completion with `time.perf_counter()`. This avoids adding
a GPU wait to the training thread before metric aggregation, and preserves the
original duration when metrics are reported one step later or flushed at the
end. The wall-clock observation can include a small thread scheduling delay;
compare averages over warmed-up steps rather than individual sub-millisecond
differences. Disabling performance metrics also disables this timing worker.

`compute_time` already includes the optimizer; do not add `optimizer_time`
again. Timings other than `draft_train_time` use rank 0, not a sum across GPUs. They cover all
micro-batches in the optimizer step. Data fetching can overlap queued GPU work,
so `wait_time` measures time spent obtaining data, not GPU idle time, and
`wait_time + compute_time` is not end-to-end wall time.

The cumulative metrics reset when the training process restarts, including on
checkpoint resume. They exclude evaluation and checkpoint saving. Deferred
metrics are paired with the dispatch wait from their own step; the final pending
step is included when metrics are flushed at shutdown. The `TIMING` log shows
per-step and cumulative durations, and completion logs report the final totals.

### Inference Metrics (aggregated between training steps)

| Metric | Unit | Source | Description |
|---|---|---|---|
| `perf/infer_capacity` | samples/s | `inference_manager.py` | System-level inference capacity: `per_slot_rate * max_concurrent_slots`. Per-slot rate is computed from recent `engine.generate()` batch times. |
| `perf/infer_batch_time` | seconds | `inference_manager.py` | Average wall-clock time of a single `engine.generate.remote()` call. Measured inside the semaphore (excludes engine slot wait time). |
| `perf/infer_samples_per_sec` | samples/s | `inference_manager.py` | Observed throughput of valid inference results across all engines. |
| `perf/infer_prompt_tokens_per_sec` | tokens/s | `inference_manager.py` | Prompt tokens in valid results divided by elapsed wall time. Available when the engine reports token counts. |
| `perf/infer_completion_tokens_per_sec` | tokens/s | `inference_manager.py` | Generated tokens in valid results divided by elapsed wall time, including prefill time. Zero for vLLM prefill-only extraction. |
| `perf/infer_tokens_per_sec` | tokens/s | `inference_manager.py` | Combined prompt and completion token throughput across all engines. |

The vLLM prefill and decode paths report token counts automatically. These metrics
are logged to the existing W&B run at training steps; no additional W&B settings
are needed. Prefill-only extraction counts prompt tokens and excludes the dummy
output token. Decode counts the full completion, including its final token even
though the training loss mask excludes that position. Dropped or invalid results
are excluded, so these rates describe usable training data production rather than
all work performed inside vLLM.

For fixed-length prefill comparisons, set
`inference.vllm.extra_args.disable_log_stats=false` to log one JSON
`PREFILL_METRICS` record per valid request. It includes vLLM's `ttft_seconds`,
`scheduled_prefill_seconds` (first scheduling to first token), queue time,
batch `generate` wall time, input/internal output token counts, cache hits and
completion status. The internal output token remains excluded from training
supervision and `completion_tokens`.

These request times include the enabled capture path; they are neither pure
GPU kernel time nor a guarantee that all Mooncake keys are already readable.
The `generate` time is shared by every request in that batch, so it is a
per-request latency only at batch size 1. Compare a fixed set of requests after
warmup instead of averaging the moving-window throughput snapshots.

### Pipeline Health Metrics

| Metric | Unit | Source | Description |
|---|---|---|---|
| `perf/dispatch_wait` | seconds | `loop.py` | Wall-clock time the main loop spent waiting for the sample pool to accumulate enough samples for dispatch. High values indicate inference is the bottleneck. |

## tqdm Progress Bar

When perf metrics are enabled, the progress bar shows additional fields:

```
Training: 10%|== | loss=0.350, acc=0.830, thru=12.1, I=18.5, T=13.2, draft=1.25s, TGS=32000.0, wait=0.10s, compute=1.20s, pool=24, epoch=1/10
```

| Key | Meaning |
|---|---|
| `thru` | Actual pipeline throughput (samples/s). This is the realized end-to-end speed of the system — how many samples are actually flowing through the pipeline per second. In steady state, `thru ≈ min(I, T)`. Derived from the controller's 10-second sliding window on sample pool inflow. |
| `I` | Inference capacity (samples/s) — how fast inference *could* produce if training were infinitely fast. |
| `T` | Training capacity (samples/s) — how fast training *could* consume if inference were infinitely fast. |
| `draft` | Dispatch wait plus trainer duration from waiting for the first batch through GPU completion of the optimizer update (seconds) |
| `TGS` | Input tokens per draft training GPU per second, including data waits |
| `wait` | Dispatch wait plus data-fetch time for the reported optimizer step (seconds) |
| `compute` | Forward + backward + optimizer time for that same step (seconds) |
| `pool` | Current sample pool size |

When perf metrics are disabled, the bar shows only `thru` (pipeline throughput).

## Bottleneck Identification Guide

### Quick diagnosis

```
if dispatch_wait >> 0:
    bottleneck = "inference"    # training is starved for data
elif pool grows over time:
    bottleneck = "training"     # inference outpaces training
```

### Drill-down within training

```
if data_time >> compute_time:
    bottleneck = "mooncake transfer / data loading"
elif compute_time >> data_time:
    bottleneck = "GPU compute (forward/backward/optimizer)"
```

### Interpreting capacity metrics

- **`I > T`**: Inference can produce data faster than training can consume.
  Training is the bottleneck. The pool will tend to grow.
- **`T > I`**: Training can consume faster than inference produces.
  Inference is the bottleneck. You'll see high `dispatch_wait`.
- **`I ~ T`**: System is balanced. Pipeline throughput (`thru`) approaches
  both capacity values.

### Example scenarios

**Scenario 1: Inference bottleneck**
```
thru=8.2, I=8.5, T=15.3, wait=2.1s, pool=0
```
Inference capacity (8.5) is much lower than training capacity (15.3).
Dispatch wait is high (2.1s per step). Pool is empty. Consider adding more
inference engines or increasing `inference_batch_size`.

**Scenario 2: Training bottleneck**
```
thru=12.1, I=25.0, T=12.3, wait=0.0s, pool=48
```
Training capacity (12.3) is much lower than inference capacity (25.0).
Dispatch wait is near zero. Pool is large and growing. Check whether
`data_time` or `compute_time` dominates to decide between optimizing Mooncake
transfer or GPU computation.

**Scenario 3: Mooncake transfer bottleneck**
```
step_time=1.2s, data_time=0.9s, compute_time=0.25s
```
Data loading takes 75% of step time. Mooncake RDMA transfer or Ray queue
latency is the sub-bottleneck. Consider increasing Mooncake buffer sizes or
reducing sequence length.

## Implementation Notes

### Why CUDA events instead of `torch.cuda.synchronize()`?

`torch.cuda.synchronize()` forces the CPU to wait for all GPU work to
complete, which **stalls the CUDA execution pipeline** and can measurably
degrade training throughput. CUDA events (`torch.cuda.Event(enable_timing=True)`)
insert markers into the GPU stream with low overhead. GPU-stage timing is
extracted later at an existing immediate or deferred metric sync point.
`draft_train_time` instead observes the optimizer's completion event in a
dedicated thread to include CPU work and all micro-batch waits in its
wall-clock interval.

### Why `time.time()` for data loading?

Data loading involves CPU/network operations (Ray queue get, Mooncake RDMA
fetch, tensor collation, H2D memcpy). The GPU stream may be idle during these
operations, so CUDA events would report near-zero time. Wall-clock time
correctly captures the actual wait.

### Concurrency model for inference capacity

Multiple inference engines run in parallel, controlled by an asyncio semaphore
(`max_concurrent_per_engine * num_engines` slots). Each `engine.generate.remote()`
call is timed individually **after** the semaphore is acquired (excluding slot
wait time). The system capacity is:

```
infer_capacity = (total_samples / total_engine_time) * max_concurrent_slots
```

This represents the theoretical maximum throughput if all engine slots are
fully utilized.

The `*_per_sec` metrics instead measure observed throughput. They use up to the
last 50 batches that completed within the last 10 seconds. Their denominator is
the time from the earliest start of those batches to the metric collection time,
measured with `time.perf_counter()`. This includes each selected batch's full
duration and idle gaps, counts overlapping engine time once, and does not multiply
by the configured concurrency. Timing includes the Ray call, engine processing,
and Mooncake publication; it is not isolated GPU compute time. No rates are emitted
when there are no recent completed batches.

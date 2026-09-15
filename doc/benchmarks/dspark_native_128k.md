# Native Qwen 128K phase acceptance

Current-machine progress is recorded in
[Qwen 128K acceptance on H800](dspark_native_128k_h800.md). The sections below
retain the workload contract and September 14 B300 prerequisite evidence.

Ticket 10 passed on H800 on 2026-09-15. The run completed real 128K training in two
five-update phases and an uninterrupted ten-update replay. Their final DCP states,
supervision and RNG records compare exactly; both runs released all workers.
See the [H800 comparison report](dspark_native_128k_h800_comparison.md) for measured
phase costs and evidence. The [resident comparison](dspark_native_128k_h800_resident.md)
and export/retention/failure checks below complete the acceptance. Historical B300
prerequisites remain separate from the H800 full-scale results.

## H800 closing evidence

All artifacts below are under `outputs/dspark_torchtitan_orchestration_20260914`.
The following tests finished without skips:

- `h800-retention-float32.log` and `h800-retention-bfloat16.log`: real two-GPU
  training, exports in the requested precision, rolling retention plus a retained
  milestone, and exact trajectory comparison after restoring either recovery
  point. Durations were 83.091 and 86.431 seconds. These use short numerical
  fixtures; full-size training and restart evidence comes from the 128K runs.
- `h800-retention-failure-green.log`: a real fourth checkpoint payload is written,
  then its commit rename fails. All three existing committed recovery points
  remain unchanged, no phase success is published, and workers release the GPUs
  (17.497 seconds).
- `h800-full-hf-export.log`: the full-size native checkpoint exports on CPU and
  the existing Qwen consumer reloads all 64 parameters with the requested
  precision; source state is unchanged and CUDA stays uninitialized (328.275
  seconds). The resulting artifact is `h800-full-hf-export/hf/`.

The resident reference completed the same ten-update, 128K workload. Its complete
cost was 1517.764 seconds, versus 3561.700 seconds for native two-phase execution;
this is a baseline, not a speedup. The report explicitly records physical GPU
count and allocator differences. Each configuration has one successful timing
run, so these measurements do not establish repeat-run variability.

## Fixed workload

The production recipe is `torchtitan.models.dspark_draft.config_registry.qwen38_27b_tp4`.
`tests.torchtitan_scale_fixtures.qwen38_128k_acceptance` bounds it to ten updates
and global batch four while retaining the production 1000-update scheduler
horizon and 40-update warmup. Comparisons use this same bounded workload.

| Item | Value |
| --- | --- |
| GPUs / topology | Eight B300, DP shard2 x TP4, CP1 / PP1 |
| Draft | Five layers, hidden 5120, FFN 17408, vocabulary 248320 |
| Attention | 24 Q / 4 KV heads, head width 256 |
| Context | 131072 tokens for all 40 actual samples |
| DSpark | 512 anchors, block seven, Markov rank 256, confidence enabled |
| Supervision | Teacher layers 1/16/31/46/61 and actual final normalized hidden |
| Updates | Local batch one, global batch four, GAS two, ten updates |
| Precision | BF16 parameters, FP32 reductions and master/Adam state |
| Optimizer | AdamW, LR 6e-4, no weight decay, clip norm one |
| Transforms | Native SelectiveAC; SP and outer model compile disabled |
| Checkpoints | Synchronous full DCP at each phase boundary, keep latest two |
| Partitions | Initial validation uses five updates followed by five updates |

The original Qwen3.8-27B teacher assets and source-built vLLM environment remain
unchanged. Two TP4 replicas use BF16, memory fraction 0.45, maximum batched
tokens 8192 and the original eager inference path. Target production and waits
are excluded from reported draft elapsed time.

The native CPU preparation entry parses and tokenizes the actual first 40
records from the existing packed source, preserves seed-42 sample order and
full-update grouping, and emits `cuda_initialized=false`. The archived plan is
`output/dspark_torchtitan_orchestration_20260914/qwen38-128k-v2/inputs/input-plan.json`,
SHA256 `99eee5f9ef2b748ccbb0c944f6cc410730dc1057bd560d5c2650ceb51f55f9a3`.
Every sample is 131072 tokens; its input identity matches the earlier preparation.

## Measurement and reference contract

The native phase timer synchronizes GPU work and preserves each rank's event
timeline. Parent elapsed time splits into launch, the native span from earliest
rank initialization through latest rank completion, and process exit. It records
initialization, restore, training including feature I/O, save and close separately.
Concurrent rank times are never added. Compiler records describe nested work
inside training, including actual local-cache hits/misses and compiler thread
count; these durations are not added again to phase totals.

Native metrics reset CUDA peak counters after logging each update. Therefore,
the phase-end timing counters do not represent whole-phase memory peaks. The
summary separately reads per-update TensorBoard active/reserved peaks, verifies
the worker PID and complete update coverage, and states the recorded ranks.
Fixed-feature replays enable native all-rank metric recording. This metrics-only
configuration difference from the initial run must remain visible in reports.

The complete draft cost additionally includes orchestration-side feature
validation and consumption-manifest preparation, plus the post-exit GPU release
check. These are reported as `draft_preparation_seconds` and
`draft_release_seconds`; `draft_total_seconds` adds both to the worker lifecycle
(`draft_elapsed_seconds`). Fixed-feature replays include their shared producer
validation/merge cost once as `shared_preparation_seconds`. Waiting to acquire
an externally occupied GPU pool is excluded. No full-scale timing pass is claimed
for these new fields until the resumed real run completes.

The first validation can capture complete initial weights and each rank's
CPU/CUDA/Python/NumPy RNG for subsequent fixed-state comparisons. That optional
`validation_capture` interval is recorded separately. Performance replays load
the common initialization and do not repeat capture.

`tests/run_torchtitan_scale_replay.py` reuses verified real feature bytes and the
same whole-run plan. It changes only complete-update phase boundaries, invokes
the DeepSpec draft entry for every phase, and waits for worker release before
the next phase. `tests/compare_torchtitan_checkpoints.py` compares complete native
DCP contents bitwise on CPU, allowing only partition-local manifest identity and
cursor representation to differ; global position, optimizer, scheduler and all
rank RNG/buffer state must match. Small supervision observations additionally
preserve selected target IDs, masks and RNG after every forward.

`tests/run_torchtitan_resident_benchmark.py` launches and times the retained
reference, including interpreter startup and complete worker exit. It verifies
the entire GPU pool is idle before and after execution.
`tests/benchmark_torchtitan_resident.py` uses the retained actual training loop
and matching two-rank FSDP2 logical batch/GAS, with the same initialization,
features, SelectiveAC and optimizer schedule. It uses two physical GPUs; six
GPUs remain unused. Its comparison with eight-rank native TP4 therefore includes
a topology difference, which must be reported separately from the cost of
checkpoint handoff. A native uninterrupted replay isolates phase-boundary cost
on the same eight-rank topology. No speedup is assumed.

`tests/summarize_torchtitan_scale.py` validates full geometry, ordered update
boundaries, committed metadata, resource observations and synchronized timing.
It compares each forward's supervision/RNG against another native run or the
matching resident DP owners. Its event accounting was checked on three actual
eight-rank SAC phase logs (`scale-timing-summary-preflight.log`); this CPU
preflight does not establish full-scale training acceptance.

Tiny numerical tests currently set `TORCHINDUCTOR_COMPILE_THREADS=1` to limit
test startup overhead. Full-scale baseline measurements must record and use
the established production setting, rather than silently inherit that override.

## Prerequisite evidence

The first actual target partition completed on September 14 at approximately
22:59 Asia/Shanghai. `128k-first-target-acceptance.json` records twenty samples,
each 131072 tokens, 161103279860 feature bytes, two actual TP4 replicas, matching
sample order and fully released target workers. Target elapsed time was
3135.927 seconds. This is target-only evidence, with zero native 128K updates.

An unrelated ms-swift Qwen3.5-2B job acquired all eight GPUs before draft launch.
Only this task's orchestration parent was terminated, leaving its immutable
plan, verified producer manifest and feature files intact. The other job was
untouched. `128k-resource-conflict.json` records the boundary; progress remains
zero with no checkpoint and no captured initialization. The resume command is
`.scratch/dspark-torchtitan-orchestration/resume-128k-after-gpu-release.sh`.
It reuses the first target partition and still enforces an idle GPU pool.

All paths below are under `output/dspark_torchtitan_orchestration_20260914`.

- Ticket 09's actual eight-GPU FP32/BF16 training, DCP restart and commit-failure
  protection are recorded in `dspark_native_tp.md`.
- `gqa-tp4-sac-fp32-artifact-acceptance.json` verifies the real eight-GPU FP32
  SelectiveAC continuous/first/resumed phases. Both updates and the full observed
  trajectory are bitwise equal to no AC, with valid DCP metadata and worker exit.
  The original combined test completed all FP32 assertions, then encountered a
  test variable-name collision before starting BF16. The collision is fixed;
  BF16 runs separately. The failed combined runner is not counted as a pass.
- `gqa-tp4-selective-ac-bf16-test.log` passes the complete eight-GPU BF16
  SelectiveAC continuous/first/resumed suite in 610.399 seconds without skips.
  Both updates, all gradients and optimizer state, scheduler and RNG are bitwise
  equal across restart and against no AC. Synchronized timing checks also pass.
- `gqa-tp4-cpu-hf-export-test.log` reconstructs a real TP4 BF16 DCP on CPU and
  loads the resulting HF artifact using the existing DeepSpec Qwen consumer.
  All 31 parameters match the actual native update bitwise; CUDA stays
  uninitialized. The 0.477-second conversion time excludes interpreter startup
  and is a small-model correctness observation, not full-scale performance.

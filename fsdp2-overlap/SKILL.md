---
name: fsdp2-overlap
description: Diagnose, implement, and verify computation/communication overlap improvements for a user-selected draft-model training path using PyTorch FSDP2 (`fully_shard`). Use for AllGather/ReduceScatter scheduling, FSDP2 communication grouping, prefetching, gradient accumulation, or exposed-communication analysis. Do not use for FSDP1/DDP-only code, inference-only tuning, or changes to an out-of-scope target/verifier model.
---

# FSDP2 Draft-Model Overlap

Optimize the selected draft-model training path from profiling evidence while preserving synchronous training semantics. Treat overlap as a scheduling problem: launch communication when its input is ready, run independent computation concurrently, and wait only at the real dependency boundary.

## Required grounding

Before changing code:

1. Read [references/tuning-playbook.md](references/tuning-playbook.md) for the diagnostic and tuning workflow.
2. If the draft model is MoE, uses `grouped_mm`, expert parallelism, conditional experts, or a separate expert mesh, also read [references/draft-moe.md](references/draft-moe.md).
3. Inspect the installed PyTorch version and local FSDP2 API/source. FSDP2 changes quickly; do not assume an API from `main` exists in the repository's pinned version.

## Scope and semantic invariants

- Modify only the draft-model path named by the user. If draft and target/verifier models share code, isolate changes behind an existing draft-specific abstraction or the narrowest compatible condition. Prove the other path is unchanged.
- Preserve synchronous optimization: the final required gradient reduction must finish before its optimizer update, and the next optimizer-step forward must observe the updated parameters.
- Preserve the configured global batch, microbatch semantics, accumulation count, loss normalization, sharding mesh, and numerical policy unless the user authorizes a change.
- Parameters may be prefetched with AllGather before use. Gradients cannot be prefetched before they exist; they may only be reduced as soon as the relevant FSDP group is ready.
- Do not equate communication duration with exposed communication. Optimize the portion that extends the critical path or stalls compute.
- Do not claim a speedup without comparable measurements. If the target hardware is unavailable, add or document reproducible instrumentation and clearly mark performance as unmeasured.

## Workflow

### 1. Establish the execution graph

Locate the draft model, FSDP2 application code, actual forward order, backward order, optimizer boundary, gradient-accumulation logic, mixed-precision policy, activation checkpointing, device mesh, and any TP/PP/EP collectives. Confirm that calls use `torch.distributed.fsdp.fully_shard`, not FSDP1's wrapper API.

Map each `fully_shard()` call to its parameter group and expected AllGather/ReduceScatter. Record shared code that must remain behaviorally unchanged.

### 2. Establish a controlled baseline

Use a representative, fixed workload and exclude warmup. Measure step time or tokens/s, peak memory, collective timing, and exposed communication. Use PyTorch Profiler for module/operator attribution and Nsight Systems for CUDA-stream, NCCL, CPU-launch, and synchronization evidence. Use Nsight Compute only after a specific compute kernel is shown to be the bottleneck.

### 3. Change the smallest evidenced bottleneck

Choose changes from the playbook; do not enable every optimization knob. Start with FSDP2 communication-group boundaries, then prefetch timing, then communication/memory tradeoffs, and only then experimental APIs. Change one meaningful variable at a time when benchmarking permits.

Prefer feature detection or version-gated code for optional APIs. Do not silently fall back to an option with different training semantics.

### 4. Validate correctness and performance

Run relevant tests and a multi-step training check. Compare loss, gradient norms or selected gradients, parameter updates, collective order, peak memory, and performance under the same workload. Account for expected BF16 tolerance if the authorized change modifies reduction dtype.

Remove an introduced optimization if it regresses the representative workload, breaks correctness, causes deadlock/OOM, or only improves a synthetic trace while reducing end-to-end throughput. Preserve unrelated user changes.

## Completion report

Lead with the measured outcome. Include:

- draft-model files and configurations changed;
- the diagnosed critical-path problem and trace evidence;
- before/after workload, throughput or step time, exposed communication, and peak memory;
- correctness checks performed;
- exact reproduction commands;
- remaining unavoidable or unmeasured communication tail;
- confirmation that the target/verifier path was not changed, when applicable.

Label unavailable measurements as unmeasured; never substitute estimates for results.

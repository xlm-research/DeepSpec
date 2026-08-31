# FSDP2 Overlap Tuning Playbook

Use this reference to diagnose and modify a draft-model training path implemented with `torch.distributed.fsdp.fully_shard` (FSDP2).

## 1. Mental model

FSDP2 shards parameters, gradients, and optimizer state. For each FSDP parameter group:

- AllGather reconstructs unsharded parameters before forward computation.
- With `reshard_after_forward=True`, FSDP reshards after forward and AllGathers again before that group's backward computation.
- Once the group's unsharded gradients are ready, ReduceScatter produces the local gradient shard.
- The optimizer updates local DTensor parameter shards.

The next optimizer-step forward depends on the completed optimizer update, but layerwise communication inside the current step can overlap with independent layer computation. The practical objective is to shorten exposed communication on the critical path, not to eliminate synchronization semantics.

Use these terms precisely:

- **Input prefetch:** prepare a future microbatch through storage/CPU/H2D work.
- **Parameter prefetch:** issue a future FSDP group's AllGather before its parameters are needed.
- **Gradient overlap:** issue ReduceScatter after the group's gradients become ready while earlier-layer backward computation continues. A gradient that does not exist cannot be prefetched.

Typical pipeline:

```text
forward compute:   FWD L0 ------- FWD L1 ------- FWD L2
all-gather stream:       AG L1 -------- AG L2

backward compute:  BWD L2 ------- BWD L1 ------- BWD L0
all-gather stream:       AG L1 -------- AG L0
reduce-scatter:          RS L2 -------- RS L1 -------- RS L0
```

Expect some boundary exposure. The first parameter materialization and final gradient-reduction tail may have no independent work available to hide them.

## 2. Verify the local implementation

Before proposing APIs, inspect the repository and installed runtime:

```python
import inspect
import torch
from torch.distributed.fsdp import FSDPModule, fully_shard

print(torch.__version__)
print(inspect.signature(fully_shard))
for name in (
    "set_modules_to_forward_prefetch",
    "set_modules_to_backward_prefetch",
    "set_requires_gradient_sync",
    "set_reshard_after_backward",
    "set_post_optim_event",
    "set_separate_reduce_scatter_group",
    "set_reduce_scatter_max_input_buffers",
):
    print(name, hasattr(FSDPModule, name))
```

Also inspect the pinned PyTorch source or matching version documentation. APIs present only in current PyTorch `main` must not be added unconditionally to an older runtime.

Confirm these implementation details:

- the optimizer is constructed after FSDP2 has converted parameters to DTensors;
- the training path invokes the module call (`model(...)`) so FSDP hooks run, rather than bypassing hooks through a direct `model.forward(...)` call;
- all ranks construct groups and call collective setup in a consistent order;
- TP, PP, CP, EP, or custom collectives are distinguished from FSDP AllGather/ReduceScatter.

## 3. Establish a useful baseline

Hold constant:

- model and target draft-model path;
- world size, node count, topology, and process placement;
- global batch, microbatch, sequence length, and accumulation count;
- precision, checkpointing, compilation, and data source;
- warmup count and measured iteration window.

Collect at least:

- median and a tail percentile of steady-state step time;
- tokens/s or examples/s;
- peak allocated and reserved GPU memory;
- AllGather and ReduceScatter duration and placement;
- compute-stream stalls caused by communication dependencies;
- exposed communication after useful compute ends;
- CPU launch gaps and accidental synchronization.

Use PyTorch Profiler to relate framework modules/operators to CUDA and NCCL work. Use Nsight Systems to decide whether collectives actually overlap with compute and with each other. Inspect the target multi-node topology because a single-node trace may hide network limitations.

Do not rely on GPU utilization alone: queued NCCL work, memory-bound kernels, or synchronization can all produce misleading utilization.

## 4. Tune communication groups first

Each `fully_shard()` call creates a communication group from parameters not already assigned to a child group. FSDP2 does not expose DDP's `bucket_cap_mb`; module grouping defines collective boundaries.

Apply `fully_shard()` bottom-up. A Transformer-like starting point is:

```python
for block in draft_model.layers:
    fully_shard(block, mesh=mesh, mp_policy=mp_policy)
fully_shard(draft_model, mesh=mesh, mp_policy=mp_policy)
```

This is a starting hypothesis, not a universal optimum.

| Trace symptom | Likely action |
|---|---|
| One root-sized AllGather/ReduceScatter with little overlap | Create child groups before sharding the root |
| A layer's collective is longer than adjacent compute | Split that layer at a meaningful execution boundary |
| Many tiny NCCL calls dominated by launch latency | Group small adjacent modules that execute together |
| Root group produces a large boundary tail | Inspect embeddings, norm, and output head left in the root; regroup only if execution order permits |
| Group is invoked multiple times per iteration | Verify grouped-module semantics and avoid accidental repeated collectives |

Do not choose a fixed MB target without measurement. Group boundaries must also respect actual execution order, shared parameters, conditional branches, and collective consistency across ranks.

## 5. Tune parameter prefetch

FSDP2 normally uses separate CUDA streams and automatic backward scheduling. Explicit prefetch is useful when the trace shows that CPU launch timing or a nonstandard execution order leaves AllGather exposed.

For a static sequential path, a one-module forward look-ahead is the safe starting point:

```python
for current, following in zip(blocks, blocks[1:]):
    current.set_modules_to_forward_prefetch([following])
```

For backward, first verify the default reverse post-forward order. Override it only when the real backward order differs or earlier issuance is measurably useful:

```python
for index in range(1, len(blocks)):
    blocks[index].set_modules_to_backward_prefetch([blocks[index - 1]])
```

Rules:

- Start with look-ahead depth 1.
- A deeper list reserves more unsharded parameter memory; accept it only with measured benefit and memory headroom.
- Do not encode a static prefetch order for genuinely dynamic control flow unless the selected path is proven stable.
- For dynamic but predictable local code, consider version-supported `unshard(async_op=True)` and wait immediately before first use. Preserve dependency correctness.

## 6. Tune gradient reduction and accumulation

ReduceScatter can begin only after an FSDP group's gradients are ready. Diagnose late readiness separately from slow communication.

For gradient accumulation, intermediate microbatches intentionally use the same parameters. When supported by the local FSDP2 version, disable gradient synchronization only for non-final microbatches and restore it for the final backward:

```python
is_final = microbatch_index == accumulation_steps - 1
draft_model.set_requires_gradient_sync(is_final)
loss.backward()
```

Verify loss scaling and gradient normalization; do not assume the training loop already normalizes by accumulation steps. Use exception-safe state restoration if the loop can fail mid-window.

Additional version-dependent options:

- `set_reshard_after_backward(False)` may avoid re-AllGathering before the next accumulation microbatch, trading memory for communication.
- `set_is_last_backward(...)` may improve microbatch scheduling in versions that expose it.
- HSDP can sometimes skip only the replication AllReduce on intermediate microbatches while retaining ReduceScatter; do not substitute this for full gradient-sync disabling without understanding the intended algorithm.

The optimizer update still waits for the final required reductions. Do not start the next optimizer-step forward with stale parameters.

## 7. Communication/memory tradeoffs

Evaluate these only after the trace identifies the corresponding bottleneck:

### `reshard_after_forward`

- `True`: lower parameter memory, but requires a backward AllGather.
- `False`: retains unsharded parameters and removes that backward AllGather at higher memory cost.
- Some versions accept an integer partial reshard size as an intermediate tradeoff.

Use selective changes rather than disabling resharding everywhere. Confirm the root module's version-specific default before overriding it.

### Reduction dtype

`MixedPrecisionPolicy.reduce_dtype` controls gradient-reduction dtype. If it is `None` while `param_dtype` is set, current FSDP2 uses the compute dtype. BF16 reduction reduces bytes relative to FP32, but it is a numerical-policy change. Compare loss, gradients, and convergence-sensitive signals before accepting it.

### Separate ReduceScatter process group

If Nsight shows AllGather and ReduceScatter serialized through one communicator despite being on separate streams, and the local version supports it, test `set_separate_reduce_scatter_group()`. Call setup consistently across ranks. More concurrency can increase network or GPU contention, so compare end-to-end throughput rather than assuming improvement.

### ReduceScatter input-buffer depth

If the compute stream stalls while waiting to reuse a still-in-flight ReduceScatter input buffer, and the local version exposes `set_reduce_scatter_max_input_buffers`, test a small increase such as 2. This is an experimental memory/overlap dial, not a general default.

### Post-optimizer event

Use `set_post_optim_event()` only when a trace proves that FSDP's default optimizer dependency includes unrelated work and delays a later AllGather. The event must be recorded after the actual optimizer update; it must not weaken that dependency.

## 8. Remove false dependencies

Inspect the hot path for:

- `.item()`, synchronous `.cpu()`, explicit device synchronization, or printing CUDA tensors;
- Python hooks or control flow that prevent the CPU from issuing future collectives early;
- DataLoader or H2D stalls mislabeled as FSDP communication stalls;
- recompilation, dynamic-shape guards, or allocator pressure between groups;
- process-group reuse that serializes collectives expected to overlap;
- activation-checkpointing order that invalidates assumed prefetch order.

Do not remove a synchronization operation until its correctness purpose is understood.

## 9. Validation and stopping rule

After each accepted change:

1. Run correctness tests and several optimizer steps.
2. Check loss, selected gradients or gradient norms, parameter updates, and cross-rank consistency.
3. Check for deadlock, OOM, collective-order mismatch, and memory growth.
4. Re-profile the same steady-state workload.
5. Report both throughput and peak memory; calculate no speedup from incomparable runs.

Stop when the remaining exposed communication is a true dependency boundary, the communication is longer than available independent compute, further overlap exceeds the memory budget, or measured changes no longer improve the representative workload. State which condition applies.

## Authoritative reference

Prefer the documentation matching the installed PyTorch version. Current-main reference: [PyTorch FSDP2 `fully_shard` documentation](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html).

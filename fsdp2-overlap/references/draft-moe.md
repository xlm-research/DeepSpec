# Draft MoE and `grouped_mm` Considerations

Read this reference only when the selected draft model contains MoE experts, `grouped_mm`, expert parallelism, conditional expert execution, or expert-specific FSDP meshes.

## Separate the pipelines

A typical draft MoE block may include:

```text
router
  -> token permutation / packing
  -> expert dispatch All-to-All (when expert-parallel)
  -> grouped_mm expert FFN
  -> expert combine All-to-All
  -> unpermute and weighted combine
```

FSDP2 adds parameter AllGather and gradient ReduceScatter around managed parameter groups. These are distinct from expert-parallel All-to-All. Label or identify each collective in the trace before changing FSDP grouping.

`grouped_mm` packages independent expert GEMMs, often with different token counts, into grouped GPU work. It reduces per-expert launch overhead but does not guarantee high utilization when token counts are small or imbalanced. Do not blame FSDP overlap for time caused by routing, permutation, expert imbalance, or tiny GEMMs.

## Preserve draft-only scope

- Identify the exact draft-model construction and training call path.
- If the target/verifier shares a block class or parallelization helper, keep behavior unchanged by default and activate new scheduling only for the draft instance through the narrowest existing configuration boundary.
- Add a regression check that exercises the target/verifier path when shared code changes are unavoidable.
- Do not change speculative-decoding acceptance semantics, target-model weights, or inference behavior while optimizing draft training.

## Build the real execution order

MoE execution may not match a simple sequential Transformer assumption. Record:

- dense and MoE block order in forward;
- actual backward order under activation checkpointing;
- expert groups executed on every rank versus conditional groups;
- which parameters use the DP/FSDP mesh and which use an expert mesh;
- when router, permutation, All-to-All, `grouped_mm`, AllGather, and ReduceScatter occur.

Configure explicit prefetch only from this observed order. A static next-block rule can prefetch the wrong expert group, reserve unnecessary memory, or create inconsistent collective ordering if ranks take different branches.

## Diagnose by timeline symptom

| Symptom | Investigation or action |
|---|---|
| `grouped_mm` waits for FSDP AllGather | Prefetch the exact expert/block parameter group earlier if the execution order is rank-consistent |
| ReduceScatter starts long after expert gradients are computed | Check FSDP group composition; unrelated late-ready parameters may share the group |
| All-to-All and FSDP collectives serialize | Identify communicator/process-group reuse and network saturation before adding concurrency |
| NCCL overlaps `grouped_mm` but step time worsens | Check SM, copy-engine, and HBM/network contention; overlap is not automatically beneficial |
| Many experts receive few tokens | Tune routing/batching only if authorized; FSDP scheduling cannot repair poor grouped-GEMM shapes |
| Different ranks produce different unused expert gradients | Use a version-supported correctness mechanism for unused parameters or redesign grouping; never force mismatched collectives |
| Expert parameters dominate a root FSDP group | Consider expert-aware child grouping or mesh placement only if it preserves the existing parallel algorithm |

## MoE-specific measurement

In addition to the general playbook, report:

- tokens routed per expert and imbalance statistics for the measured window;
- `grouped_mm` duration and representative shapes;
- token permutation/unpermutation time;
- expert dispatch/combine All-to-All time;
- FSDP AllGather and ReduceScatter time for dense versus expert groups;
- whether observed overlap improves draft-model tokens/s and step time;
- peak memory added by prefetched expert weights or deeper ReduceScatter buffering.

Keep the workload and router behavior comparable. A change in expert routing distribution can invalidate a before/after overlap comparison.

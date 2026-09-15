#!/usr/bin/env python3
"""Microbenchmark the production GLM expert EP forward/backward, without a target.

Run with the isolated draft Python (not a system ``torchrun`` executable):
    CUDA_VISIBLE_DEVICES=3,6 /path/to/draft/python -m torch.distributed.run \
        --standalone --nproc-per-node=2 scripts/benchmark_glm5_draft_ep.py

Optional full GLM expert geometry; substantially larger than the default:
    ... scripts/benchmark_glm5_draft_ep.py --num-experts 288 --hidden 4096 \
        --intermediate 2048 --top-k 8 --tokens 256

This measures dispatch, real GLM grouped GEMM, combine, and backward. It excludes
router projection, shared experts, attention, loss, FSDP/replica gradient sync,
optimizer, target generation, model initialization, and warmup/JIT. Numerical
equivalence is checked separately by the DeepEP runtime and GLM integration tests.
"""

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "deepep", "both"), default="both")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=256, help="Input tokens per EP rank")
    parser.add_argument("--chunk", type=int, default=4096,
                        help="Production token chunk and fixed DeepEP per-rank capacity (at least EP world size)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path, help="Also write rank-zero JSON to this path")
    args = parser.parse_args()
    for name in ("hidden", "intermediate", "num_experts", "top_k", "tokens", "chunk", "warmup", "steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.top_k > args.num_experts:
        parser.error("--top-k must not exceed --num-experts")
    return args


def benchmark_backend(args, backend, torch, dist, device):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts

    from deepspec.modeling.deepseek_v4_parallel import _parallelize_moe

    rank, world = dist.get_rank(), dist.get_world_size()
    config = Glm5NextTextConfig(
        hidden_size=args.hidden,
        moe_intermediate_size=args.intermediate,
        n_routed_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        num_hidden_layers=1,
        mlp_layer_types=["sparse"],
    )
    config._experts_implementation = "grouped_mm"
    # Shard on meta first: full-geometry runs never allocate all experts per GPU.
    moe = torch.nn.Module()
    with torch.device("meta"):
        moe.experts = Glm5NextTextExperts(config).to(dtype=torch.bfloat16)
    moe.shared_experts = torch.nn.Identity()
    topology = SimpleNamespace(
        expert_parallel_size=world,
        expert_parallel_rank=rank,
        expert_parallel_group=dist.group.WORLD,
        pure_expert_parallel=True,
        tensor_parallel_size=1,
    )
    effective_chunk = max(args.chunk, world)
    dispatcher = None
    if backend == "deepep":
        from deepspec.distributed.deepep_dispatch import DeepEPDispatcher, require_deepep

        require_deepep()
        dispatcher = DeepEPDispatcher(
            dist.group.WORLD,
            num_experts=args.num_experts,
            hidden_size=args.hidden,
            top_k=args.top_k,
            max_tokens_per_rank=effective_chunk,
        )
    _parallelize_moe(moe, topology=topology, expert_dispatcher=dispatcher)
    moe.to_empty(device=device)
    weights_rng = torch.Generator(device=device).manual_seed(args.seed + rank)
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(mean=0.0, std=0.02, generator=weights_rng)
    # Reset independent RNGs for each backend: identical local weights, inputs,
    # global routing IDs, selected scores, and output cotangents on every run.
    input_rng = torch.Generator(device=device).manual_seed(args.seed + 100000 + rank)
    hidden = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16,
                         generator=input_rng, requires_grad=True)
    routing_logits = torch.randn(args.tokens, args.num_experts, device=device, generator=input_rng)
    selected_logits, indices = routing_logits.topk(args.top_k, dim=-1)
    scores = selected_logits.softmax(dim=-1).detach().requires_grad_(True)
    cotangent = torch.randn(hidden.shape, device=device, dtype=hidden.dtype, generator=input_rng)
    del routing_logits, selected_logits

    def step():
        output = moe.experts(hidden, indices, scores)
        output.backward(cotangent)

    def clear_gradients():
        moe.zero_grad(set_to_none=True)
        hidden.grad = None
        scores.grad = None

    for _ in range(args.warmup):
        clear_gradients()
        step()
    torch.cuda.synchronize(device)
    dist.barrier()
    clear_gradients()
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(args.steps):
        clear_gradients()
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - start))
    local_result = {
        "rank": rank,
        "forward_backward_ms": samples,
        "peak_pytorch_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_pytorch_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "deepep_buffer_bytes": int(dispatcher._buffer.num_bytes) if dispatcher is not None else 0,
    }
    records = [None] * world
    dist.all_gather_object(records, local_result)
    if dispatcher is not None:
        dispatcher.close()
    del step, clear_gradients, moe, dispatcher, hidden, indices, scores, cotangent
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    max_rank_samples = [max(record["forward_backward_ms"][i] for record in records)
                        for i in range(args.steps)]
    ordered = sorted(max_rank_samples)
    return {
        "backend": backend,
        "effective_chunk": effective_chunk,
        "deepep_max_tokens_per_rank": effective_chunk if backend == "deepep" else None,
        "max_rank_forward_backward_ms": {
            "mean": statistics.mean(max_rank_samples),
            "median": statistics.median(max_rank_samples),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "min": min(ordered),
            "max": max(ordered),
        },
        "max_rank_peak_pytorch_allocated_bytes": max(r["peak_pytorch_allocated_bytes"] for r in records),
        "max_rank_peak_pytorch_reserved_bytes": max(r["peak_pytorch_reserved_bytes"] for r in records),
        "max_rank_deepep_buffer_bytes": max(r["deepep_buffer_bytes"] for r in records),
        "ranks": records,
    }


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2 or args.num_experts % world:
        raise ValueError("Use torchrun with EP world >= 2 dividing --num-experts")
    if args.backend != "native" and args.hidden % 256:
        raise ValueError("DeepEP requires --hidden divisible by 256")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    os.environ["DEEPSPEC_V4_EP_TOKEN_CHUNK"] = str(args.chunk)
    try:
        backends = ("native", "deepep") if args.backend == "both" else (args.backend,)
        results = [benchmark_backend(args, backend, torch, dist, device) for backend in backends]
        if dist.get_rank() == 0:
            result = {
                "benchmark": "glm5_draft_expert_ep_microbenchmark",
                "microbenchmark": True,
                "scope": "Production GLM grouped_mm experts and EP dispatch/combine forward+backward; fixed precomputed routing",
                "excludes": "target, router projection, shared experts, attention, loss, FSDP/replica gradient sync, optimizer, initialization, warmup/JIT",
                "memory_scope": "PyTorch allocator peaks; separately reported DeepEP communication-buffer capacity is not included in allocator peaks",
                "dtype": "bfloat16",
                "world_size": world,
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "EP_DISABLE_GIN": os.environ.get("EP_DISABLE_GIN", "0"),
                "results": results,
            }
            if "deepep" in backends:
                result["deep_ep"] = importlib.metadata.version("deep_ep")
            if len(results) == 2:
                result["native_over_deepep_mean_time_ratio"] = (
                    results[0]["max_rank_forward_backward_ms"]["mean"]
                    / results[1]["max_rank_forward_backward_ms"]["mean"]
                )
            serialized = json.dumps(result, indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized)
            print(serialized, end="", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

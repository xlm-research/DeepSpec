#!/usr/bin/env python3
"""Profile only the trainable DeepSeek-V4 DSpark draft under FSDP2.

The online target is never constructed. Synthetic target features have the
same tensor contract as the real online target, so the draft backbone,
grouped-MoE, loss, backward, FSDP collectives, clipping, and optimizer all run.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.profiler import record_function
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepspec.distributed import ParallelConfig, ParallelContext, apply_parallelism
from deepspec.distributed.fsdp import clip_grad_norm_, gradient_sync_context
from deepspec.modeling.dspark.deepseek_v4 import (
    DeepseekV4DSparkModel,
    build_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.training import BF16Optimizer
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.utils.config import ConfigNode
from deepspec.utils.distributed import init_dist
from deepspec.utils.metrics import reset as reset_metrics
from deepspec.utils.torch_profiler import build_torch_profiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-config",
        default="/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    parser.add_argument("--sequence-length", type=int, default=131072)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--num-draft-layers", type=int, default=3)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--dp-replicate", type=int, default=1)
    parser.add_argument("--dp-shard", type=int, default=1)
    parser.add_argument("--cp", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--reshard-after-forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--forward-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--backward-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--reduce-dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument(
        "--wrap-granularity",
        choices=("block", "block_components"),
        default="block",
    )
    parser.add_argument("--profile-dir")
    parser.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help=(
            "Bracket the first measured optimizer step with cudaProfilerStart/Stop "
            "for a tightly scoped Nsight Systems capture."
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--last-backward-hint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _draft_model_args(args: argparse.Namespace) -> ConfigNode:
    return ConfigNode(
        block_size=args.block_size,
        num_draft_layers=args.num_draft_layers,
        target_layer_ids=[0, 1, 2],
        mask_token_id=129279,
        num_anchors=args.num_anchors,
        sliding_window=128,
        markov_rank=256,
        markov_head_type="vanilla",
        confidence_head_alpha=0.0,
        confidence_head_with_markov=False,
    )


def _build_batch(
    *,
    args: argparse.Namespace,
    model: DeepseekV4DSparkModel,
    context: ParallelContext,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sequence_length = int(args.sequence_length)
    local_length = sequence_length // int(args.cp)
    if sequence_length % int(args.cp):
        raise ValueError("sequence length must be divisible by CP")
    input_ids = (
        torch.arange(sequence_length, device=device, dtype=torch.long)
        .remainder_(int(model.config.vocab_size) - 1)
        .unsqueeze(0)
    )
    loss_mask = torch.ones((1, sequence_length), device=device, dtype=torch.float32)
    feature_width = len(model.target_layer_ids) * int(model.config.hidden_size)
    generator = torch.Generator(device=device)
    generator.manual_seed(1000 + int(context.context_parallel_rank))
    target_hidden_states = torch.randn(
        (1, local_length, feature_width),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    metadata = torch.tensor([0], device=device, dtype=torch.int64)
    return {
        "input_ids": input_ids,
        "target_hidden_states": target_hidden_states,
        "loss_mask": loss_mask,
        "context_start": metadata,
        "context_len": metadata,
        "seq_len": metadata,
    }


def _trainable_parameter_checksum(model: torch.nn.Module) -> torch.Tensor:
    checksum = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float64)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        checksum.add_(value.detach().double().sum())
    dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
    return checksum


def _distributed_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu())


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sample.")
    position = (len(ordered) - 1) * float(percentile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _run_optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: BF16Optimizer,
    batch: dict[str, torch.Tensor],
    accumulation_steps: int,
    use_last_backward_hint: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    final_loss = None
    for micro_step in range(accumulation_steps):
        should_sync = micro_step + 1 == accumulation_steps
        with record_function("deepspec::draft_benchmark_micro_step"):
            with gradient_sync_context(
                model,
                should_sync=should_sync,
                use_last_backward_hint=use_last_backward_hint,
            ):
                with record_function("deepspec::draft_forward"):
                    outputs = model(**batch)
                with record_function("deepspec::draft_loss"):
                    loss = compute_dspark_loss(
                        outputs=outputs,
                        loss_decay_gamma=4.0,
                        ce_loss_alpha=1.0,
                        l1_loss_alpha=0.0,
                        confidence_head_alpha=0.0,
                    ) / accumulation_steps
                final_loss = loss.detach()
                with record_function("deepspec::draft_backward"):
                    loss.backward()
            reset_metrics()
    with record_function("deepspec::draft_gradient_norm"):
        grad_norm = clip_grad_norm_(model, 1.0)
    with record_function("deepspec::draft_optimizer"):
        optimizer.step()
    assert final_loss is not None
    return final_loss, grad_norm.detach()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device, rank, world_size = init_dist(local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    parallel_config = ParallelConfig(
        dp_replicate=args.dp_replicate,
        dp_shard=args.dp_shard,
        cp=args.cp,
        tp=args.tp,
        ep=1,
        use_fsdp=True,
        context_parallel_backend="model_native",
        reshard_after_forward=args.reshard_after_forward,
        forward_prefetch=args.forward_prefetch,
        backward_prefetch=args.backward_prefetch,
        prefetch_depth=args.prefetch_depth,
        reduce_dtype=args.reduce_dtype,
        fsdp_wrap_granularity=args.wrap_granularity,
    )
    parallel_config.validate_world_size(world_size)
    context = ParallelContext.build(parallel_config, device_type=device.type)
    configure_loss_reduction_group(context.loss_mesh.get_group())

    target_config = AutoConfig.from_pretrained(args.target_config)
    draft_config = build_draft_config(target_config, _draft_model_args(args))
    model = DeepseekV4DSparkModel(draft_config).to(device=device, dtype=torch.bfloat16)
    model.set_embedding_head_trainable(False)
    model = apply_parallelism(
        model,
        context,
        parallel_config,
        param_dtype=torch.bfloat16,
        sequence_length=args.sequence_length,
    )
    model.train()
    optimizer = BF16Optimizer(model, 1.0e-5, 100, 0.0, 0.0)
    batch = _build_batch(args=args, model=model, context=context, device=device)
    initial_checksum = _trainable_parameter_checksum(model)

    for _ in range(args.warmup_steps):
        _run_optimizer_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            accumulation_steps=args.accumulation_steps,
            use_last_backward_hint=args.last_backward_hint,
        )
    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    step_times_ms = []
    losses = []
    grad_norms = []
    for measure_index in range(args.measure_steps):
        capture_with_cuda_profiler = (
            args.cuda_profiler_capture and measure_index == 0
        )
        if capture_with_cuda_profiler:
            dist.barrier()
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStart()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss, grad_norm = _run_optimizer_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            accumulation_steps=args.accumulation_steps,
            use_last_backward_hint=args.last_backward_hint,
        )
        end.record()
        end.synchronize()
        if capture_with_cuda_profiler:
            # All CUDA work is complete before the first rank closes the
            # process-tree capture, so every rank contributes a full step.
            dist.barrier()
            torch.cuda.cudart().cudaProfilerStop()
        step_times_ms.append(_distributed_max(start.elapsed_time(end), device))
        losses.append(float(loss.float().cpu()))
        grad_norms.append(float(grad_norm.float().cpu()))

    # Keep the performance/correctness accounting independent from the extra
    # profiled optimizer step and from profiler memory allocations.
    peak_allocated = _distributed_max(torch.cuda.max_memory_allocated(device), device)
    peak_reserved = _distributed_max(torch.cuda.max_memory_reserved(device), device)
    final_checksum = _trainable_parameter_checksum(model)

    profile_config = None
    if args.profile_dir:
        profile_config = ConfigNode(
            enabled=True,
            trace_dir=os.path.abspath(args.profile_dir),
            ranks="all",
            skip_first_steps=0,
            wait_steps=0,
            warmup_steps=0,
            active_steps=1,
            repeat=1,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            use_gzip=True,
            row_limit=200,
        )
    profiler = (
        build_torch_profiler(profile_config, global_rank=rank, world_size=world_size)
        if profile_config is not None
        else nullcontext()
    )
    if profile_config is not None:
        with profiler:
            with record_function("deepspec::draft_benchmark_profile_step"):
                _run_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    accumulation_steps=args.accumulation_steps,
                    use_last_backward_hint=args.last_backward_hint,
                )
            profiler.step()

    mean_step_seconds = sum(step_times_ms) / len(step_times_ms) / 1000.0
    median_step_seconds = statistics.median(step_times_ms) / 1000.0
    draft_tokens_per_step = (
        args.num_anchors * args.block_size * args.accumulation_steps
        * args.dp_replicate * args.dp_shard
    )
    result = {
        "host": socket.gethostname(),
        "world_size": world_size,
        "torch_version": torch.__version__,
        "workload": {
            "sequence_length": args.sequence_length,
            "num_anchors": args.num_anchors,
            "block_size": args.block_size,
            "num_draft_layers": args.num_draft_layers,
            "accumulation_steps": args.accumulation_steps,
            "draft_tokens_per_optimizer_step": draft_tokens_per_step,
            "target_model_constructed": False,
            "last_backward_hint": args.last_backward_hint,
            "cuda_profiler_capture": args.cuda_profiler_capture,
        },
        "parallel": parallel_config.to_dict(),
        "step_times_ms_max_rank": step_times_ms,
        "mean_step_time_ms_max_rank": mean_step_seconds * 1000.0,
        "median_step_time_ms_max_rank": median_step_seconds * 1000.0,
        "p95_step_time_ms_max_rank": _percentile(step_times_ms, 0.95),
        "step_time_stdev_ms_max_rank": (
            statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0
        ),
        "draft_tokens_per_second": draft_tokens_per_step / mean_step_seconds,
        "draft_tokens_per_second_median": (
            draft_tokens_per_step / median_step_seconds
        ),
        "peak_memory_allocated_bytes_max_rank": int(peak_allocated),
        "peak_memory_reserved_bytes_max_rank": int(peak_reserved),
        "losses": losses,
        "grad_norms": grad_norms,
        "initial_parameter_checksum": float(initial_checksum.cpu()),
        "final_parameter_checksum": float(final_checksum.cpu()),
        "profile_dir": os.path.abspath(args.profile_dir) if args.profile_dir else None,
    }
    if rank == 0:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

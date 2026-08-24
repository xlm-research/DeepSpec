from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from .config import ParallelConfig
from .mesh import ParallelContext


@dataclass(frozen=True)
class TensorParallelPlan:
    styles: dict[str, object]
    attention_fqns: tuple[str, ...]


def build_transformer_tp_plan(model: nn.Module) -> TensorParallelPlan:
    """Build a plan from real module FQNs instead of architecture guesses."""

    styles: dict[str, object] = {}
    attention_fqns = []
    for fqn, module in model.named_modules():
        if not fqn:
            continue
        if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            attention_fqns.append(fqn)
            styles.update(
                {
                    f"{fqn}.q_proj": ColwiseParallel(),
                    f"{fqn}.k_proj": ColwiseParallel(),
                    f"{fqn}.v_proj": ColwiseParallel(),
                    f"{fqn}.o_proj": RowwiseParallel(),
                }
            )
        if all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")):
            styles.update(
                {
                    f"{fqn}.gate_proj": ColwiseParallel(),
                    f"{fqn}.up_proj": ColwiseParallel(),
                    f"{fqn}.down_proj": RowwiseParallel(),
                }
            )
    if not styles:
        raise ValueError(
            f"No supported Attention/MLP projection FQNs found on {type(model).__name__}."
        )
    return TensorParallelPlan(styles, tuple(attention_fqns))


def apply_tensor_parallelism(
    model: nn.Module,
    context: ParallelContext,
    config: ParallelConfig,
) -> nn.Module:
    if config.tp == 1:
        return model
    if config.use_sequence_parallel:
        raise NotImplementedError(
            "The current project models have multi-input decoder blocks. Their "
            "sequence layouts require a model-specific PrepareModuleInput plan; "
            "this combination is rejected instead of silently mis-sharding inputs."
        )
    plan = build_transformer_tp_plan(model)
    parallelize_module(model, context.tp_mesh, plan.styles)

    # The projection outputs are local tensors, so view/reshape code must use
    # local Q/KV head counts. The Q:KV grouping ratio remains unchanged.
    modules = dict(model.named_modules())
    for fqn in plan.attention_fqns:
        attention = modules[fqn]
        for attr in ("num_attention_heads", "num_heads"):
            if hasattr(attention, attr):
                setattr(attention, attr, int(getattr(attention, attr)) // config.tp)
        for attr in ("num_key_value_heads", "num_kv_heads"):
            if hasattr(attention, attr):
                setattr(attention, attr, int(getattr(attention, attr)) // config.tp)
        if hasattr(attention, "num_key_value_groups"):
            q_attr = "num_attention_heads" if hasattr(attention, "num_attention_heads") else "num_heads"
            kv_attr = "num_key_value_heads" if hasattr(attention, "num_key_value_heads") else "num_kv_heads"
            q_heads = int(getattr(attention, q_attr))
            kv_heads = int(getattr(attention, kv_attr))
            attention.num_key_value_groups = q_heads // kv_heads
    return model


__all__ = [
    "TensorParallelPlan",
    "apply_tensor_parallelism",
    "build_transformer_tp_plan",
]

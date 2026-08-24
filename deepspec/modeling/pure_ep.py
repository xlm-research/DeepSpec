"""Utilities for routed experts that are partitioned only by expert parallelism."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def get_pure_expert_modules(model: nn.Module) -> list[nn.Module]:
    """Return routed-expert containers marked by the DeepSeek-V4 EP adapter."""

    return [
        module
        for module in model.modules()
        if bool(getattr(module, "_deepspec_pure_expert_parallel", False))
    ]


def _source_rank(process_group) -> int:
    if hasattr(dist, "get_global_rank"):
        return int(dist.get_global_rank(process_group, 0))
    return int(dist.get_process_group_ranks(process_group)[0])


def distribute_expert_parameter(
    module: nn.Module,
    parameter_name: str,
    *,
    dim: int,
    rank: int,
    size: int,
    process_group,
) -> None:
    """Keep this EP rank's expert slice without startup-time broadcasts."""

    del process_group
    parameter = module._parameters[parameter_name]
    if parameter is None:
        raise RuntimeError(f"Missing expert parameter {parameter_name!r}.")
    dim = int(dim) % parameter.ndim
    size = int(size)
    rank = int(rank)
    if int(parameter.shape[dim]) % size != 0:
        raise ValueError(
            f"{parameter_name} dimension {dim}={parameter.shape[dim]} "
            f"is not divisible by EP size {size}."
        )
    shard_width = int(parameter.shape[dim]) // size
    local_shard = parameter.detach().narrow(
        dim, rank * shard_width, shard_width
    ).contiguous()
    if not parameter.is_meta:
        local_shard = local_shard.clone().to(
            device=(
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else parameter.device
            )
        )
    module._parameters[parameter_name] = nn.Parameter(
        local_shard,
        requires_grad=bool(parameter.requires_grad),
    )


def materialize_modules_locally(modules: list[nn.Module], *, device) -> None:
    """Materialize rank-local expert modules without parameter collectives."""

    if not modules:
        return
    seen_parameters: set[int] = set()
    seen_buffers: set[int] = set()
    for root in modules:
        for module in root.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is None or id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                tensor = (
                    torch.empty_like(parameter, device=device)
                    if parameter.is_meta
                    else parameter.to(device=device)
                )
                module._parameters[name] = nn.Parameter(
                    tensor, requires_grad=bool(parameter.requires_grad)
                )
            for name, buffer in list(module._buffers.items()):
                if buffer is None or id(buffer) in seen_buffers:
                    continue
                seen_buffers.add(id(buffer))
                module._buffers[name] = (
                    torch.empty_like(buffer, device=device)
                    if buffer.is_meta
                    else buffer.to(device=device)
                )


def materialize_and_broadcast_modules(
    modules: list[nn.Module], *, device, process_group
) -> None:
    """Replicate EP-owned modules without letting FSDP shard their parameters."""

    if not modules:
        return
    source_rank = _source_rank(process_group)
    seen_parameters: set[int] = set()
    seen_buffers: set[int] = set()
    for root in modules:
        for module in root.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is None or id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                tensor = (
                    torch.empty_like(parameter, device=device)
                    if parameter.is_meta
                    else parameter.to(device=device)
                )
                materialized = nn.Parameter(
                    tensor,
                    requires_grad=bool(parameter.requires_grad),
                )
                module._parameters[name] = materialized
                dist.broadcast(
                    materialized.data,
                    src=source_rank,
                    group=process_group,
                )
            for name, buffer in list(module._buffers.items()):
                if buffer is None or id(buffer) in seen_buffers:
                    continue
                seen_buffers.add(id(buffer))
                materialized = (
                    torch.empty_like(buffer, device=device)
                    if buffer.is_meta
                    else buffer.to(device=device)
                )
                module._buffers[name] = materialized
                dist.broadcast(materialized, src=source_rank, group=process_group)


def synchronize_module_gradients(
    modules: list[nn.Module], *, process_groups: list[tuple[object, int]]
) -> None:
    """Average replicated pure-EP gradients over orthogonal replica axes."""

    seen_parameters: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            # Routing is data dependent, so an expert may be unused on one
            # replica while the corresponding parameter has a gradient on
            # another. Every rank must still issue the exact same collectives
            # in the exact same order; a zero gradient is the correct local
            # contribution for an unused expert.
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            for process_group, size in process_groups:
                size = int(size)
                if size <= 1:
                    continue
                dist.all_reduce(parameter.grad, group=process_group)
                parameter.grad.div_(size)


def synchronize_pure_expert_gradients(modules: list[nn.Module], *, sparse_mesh) -> None:
    """Average EP-owned parameters over their orthogonal replica dimensions."""

    if not modules or sparse_mesh is None:
        return
    process_groups = []
    for dimension in ("dp_replicate", "expert_fsdp"):
        mesh = sparse_mesh[dimension]
        size = int(mesh.size())
        if size > 1:
            process_groups.append((mesh.get_group(), size))
    synchronize_module_gradients(modules, process_groups=process_groups)


__all__ = [
    "get_pure_expert_modules",
    "materialize_and_broadcast_modules",
    "materialize_modules_locally",
    "synchronize_module_gradients",
    "synchronize_pure_expert_gradients",
]

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from .config import ParallelConfig
from .mesh import ParallelContext
from deepspec.modeling.pure_ep import get_pure_expert_modules


def transformer_blocks(model: nn.Module) -> list[nn.Module]:
    layers = getattr(model, "layers", None)
    if layers is None:
        nested = getattr(model, "model", None)
        layers = getattr(nested, "layers", None) if nested is not None else None
    return list(layers or [])


def apply_fsdp2(
    model: nn.Module,
    context: ParallelContext,
    config: ParallelConfig,
    *,
    param_dtype: torch.dtype,
) -> nn.Module:
    if not config.use_fsdp:
        return model
    blocks = transformer_blocks(model)
    if not blocks:
        raise ValueError(
            "FSDP2 bottom-up wrapping requires the model to expose Transformer "
            "blocks as `.layers` or `.model.layers`."
        )
    mesh = (
        context.dense_mesh["dp_shard_cp"]
        if config.dp_replicate == 1
        else context.fsdp_mesh
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=param_dtype,
    )
    ignored_params = {
        parameter
        for expert_root in get_pure_expert_modules(model)
        for parameter in expert_root.parameters()
    }
    for block in blocks:
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=config.reshard_after_forward,
            ignored_params=ignored_params,
        )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        # Match TorchTitan's root policy: root-owned heads/selectors may be
        # used after the final child block and must stay materialized until
        # their backward graph has run. Child blocks still reshard eagerly.
        reshard_after_forward=False,
        ignored_params=ignored_params,
    )
    return model


@contextmanager
def gradient_sync_context(model: nn.Module, *, should_sync: bool):
    if should_sync:
        yield
        return
    if isinstance(model, FSDPModule):
        model.set_requires_gradient_sync(False, recurse=True)
        model.set_reshard_after_backward(False, recurse=True)
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True, recurse=True)
            model.set_reshard_after_backward(True, recurse=True)
        return
    no_sync = getattr(model, "no_sync", None)
    with (no_sync() if no_sync is not None else nullcontext()):
        yield


@torch.no_grad()
def clip_grad_norm_(
    model: nn.Module,
    max_norm: float,
    *,
    pure_expert_modules: list[nn.Module] | None = None,
    expert_parallel_group=None,
) -> torch.Tensor:
    """Clip a model containing both FSDP2 DTensors and pure-EP tensors.

    PyTorch's generic clipper batches gradients by device and dtype, which
    makes its foreach kernel reject a bucket containing both Tensor and
    DTensor gradients.  Compute the two contributions independently and use
    one common coefficient so this remains a true global L2-norm clip.
    """

    expert_parameter_ids = {
        id(parameter)
        for module in (pure_expert_modules or [])
        for parameter in module.parameters()
    }
    distributed_parameters = []
    expert_parameters = []
    other_parameters = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if id(parameter) in expert_parameter_ids:
            expert_parameters.append(parameter)
        elif isinstance(parameter.grad, torch.distributed.tensor.DTensor):
            distributed_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    device = next(
        (parameter.grad.device for parameter in model.parameters() if parameter.grad is not None),
        torch.device("cpu"),
    )
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    if distributed_parameters:
        dtensor_norm = torch.nn.utils.clip_grad_norm_(
            distributed_parameters, float("inf")
        )
        if isinstance(dtensor_norm, torch.distributed.tensor.DTensor):
            dtensor_norm = dtensor_norm.to_local()
        total_sq.add_(dtensor_norm.float().square())

    expert_sq = torch.zeros_like(total_sq)
    for parameter in expert_parameters:
        expert_sq.add_(parameter.grad.float().square().sum())
    if expert_parallel_group is not None:
        torch.distributed.all_reduce(expert_sq, group=expert_parallel_group)
    total_sq.add_(expert_sq)

    # This bucket is normally empty: all dense parameters are managed by
    # FSDP2 and all ignored parameters are the pure experts. Keep it for
    # small/test models and non-EP configurations.
    for parameter in other_parameters:
        total_sq.add_(parameter.grad.float().square().sum())

    total_norm = total_sq.sqrt()
    coefficient = min(float(max_norm) / (float(total_norm.item()) + 1e-6), 1.0)
    if coefficient < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return total_norm


__all__ = [
    "apply_fsdp2",
    "clip_grad_norm_",
    "gradient_sync_context",
    "transformer_blocks",
]

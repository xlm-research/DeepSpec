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


def fsdp_reduce_dtype(
    param_dtype: torch.dtype,
    policy: str = "auto",
) -> torch.dtype:
    """Resolve the draft FSDP gradient-reduction dtype."""

    if policy == "fp32":
        return torch.float32
    if policy == "bf16":
        if param_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("BF16 FSDP reduction requires BF16/FP16 parameters.")
        return torch.bfloat16
    if policy == "auto":
        return (
            torch.bfloat16
            if param_dtype in (torch.bfloat16, torch.float16)
            else torch.float32
        )
    raise ValueError(f"Unsupported FSDP reduce dtype policy {policy!r}.")


def configure_fsdp2_prefetch(
    forward_order: list[nn.Module],
    *,
    backward_order: list[nn.Module] | None = None,
    forward_prefetch: bool,
    backward_prefetch: bool,
    prefetch_depth: int = 1,
) -> None:
    """Install a static one-block-ahead FSDP2 prefetch schedule.

    FSDP2 can infer a backward order after the first eager iteration, but that
    leaves the first iteration exposed and the inferred order can be disturbed
    by model-specific wrappers. The DSpark draft is a static decoder stack, so
    make both directions explicit.
    """

    forward_modules = [
        module for module in forward_order if isinstance(module, FSDPModule)
    ]
    if backward_order is None:
        backward_modules = list(reversed(forward_modules))
    else:
        backward_modules = [
            module for module in backward_order if isinstance(module, FSDPModule)
        ]
    depth = max(int(prefetch_depth), 1)
    if forward_prefetch:
        for index, module in enumerate(forward_modules[:-1]):
            module.set_modules_to_forward_prefetch(
                forward_modules[index + 1 : index + 1 + depth]
            )
    if backward_prefetch:
        for index, module in enumerate(backward_modules[:-1]):
            module.set_modules_to_backward_prefetch(
                backward_modules[index + 1 : index + 1 + depth]
            )


def _fully_shard_draft_blocks(
    blocks: list[nn.Module],
    *,
    mesh,
    mp_policy: MixedPrecisionPolicy,
    ignored_params: set[nn.Parameter],
    reshard_after_forward: bool,
    granularity: str,
) -> tuple[list[nn.Module], list[nn.Module]]:
    """Shard draft modules bottom-up and return their real execution orders."""

    forward_order: list[nn.Module] = []
    backward_by_block: list[list[nn.Module]] = []
    for block in blocks:
        components: list[nn.Module] = []
        if granularity == "block_components":
            for name in ("self_attn", "mlp"):
                component = getattr(block, name, None)
                if component is None:
                    continue
                fully_shard(
                    component,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                    ignored_params=ignored_params,
                )
                components.append(component)
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            ignored_params=ignored_params,
        )
        # Nested FSDP hooks enter forward at block -> attention -> MLP, while
        # their output hooks enter backward at block -> MLP -> attention.
        forward_order.extend([block, *components])
        backward_by_block.append([block, *reversed(components)])
    backward_order = [
        module
        for block_order in reversed(backward_by_block)
        for module in block_order
    ]
    return forward_order, backward_order


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
        # Production training uses BF16 parameters, so both gradient
        # ReduceScatter and replicated-gradient reduction communicate BF16.
        # Master weights and Adam moments are independently kept in FP32 by
        # MasterWeightAdamW after the reduced gradients reach the optimizer.
        reduce_dtype=fsdp_reduce_dtype(param_dtype, config.reduce_dtype),
        output_dtype=param_dtype,
    )
    ignored_params = {
        parameter
        for expert_root in get_pure_expert_modules(model)
        for parameter in expert_root.parameters()
    }
    forward_order, backward_order = _fully_shard_draft_blocks(
        blocks,
        mesh=mesh,
        mp_policy=mp_policy,
        ignored_params=ignored_params,
        reshard_after_forward=config.reshard_after_forward,
        granularity=config.fsdp_wrap_granularity,
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
    configure_fsdp2_prefetch(
        forward_order,
        backward_order=backward_order,
        forward_prefetch=config.forward_prefetch,
        backward_prefetch=config.backward_prefetch,
        prefetch_depth=config.prefetch_depth,
    )
    return model


@contextmanager
def gradient_sync_context(
    model: nn.Module,
    *,
    should_sync: bool,
    use_last_backward_hint: bool = False,
):
    set_is_last_backward = (
        getattr(model, "set_is_last_backward", None)
        if isinstance(model, FSDPModule)
        else None
    )
    if use_last_backward_hint and callable(set_is_last_backward):
        # PyTorch 2.11 otherwise treats every accumulated micro-batch as the
        # last backward, waiting for pending reductions and clearing backward
        # prefetch state each time. Only the synchronized micro-batch closes
        # this optimizer-step backward window.
        set_is_last_backward(should_sync)
    if should_sync:
        try:
            yield
        finally:
            if use_last_backward_hint and callable(set_is_last_backward):
                set_is_last_backward(True)
        return
    if isinstance(model, FSDPModule):
        model.set_requires_gradient_sync(False, recurse=True)
        model.set_reshard_after_backward(False, recurse=True)
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True, recurse=True)
            model.set_reshard_after_backward(True, recurse=True)
            if use_last_backward_hint and callable(set_is_last_backward):
                set_is_last_backward(True)
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
    # Keep the clipping coefficient on device.  Calling total_norm.item() here
    # serialized the final HSDP gradient collective with every FP32 optimizer
    # kernel, which is especially costly with a cross-node dp_replicate mesh.
    coefficient = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if isinstance(gradient, torch.distributed.tensor.DTensor):
            gradient.to_local().mul_(coefficient)
        else:
            gradient.mul_(coefficient)
    return total_norm


__all__ = [
    "apply_fsdp2",
    "clip_grad_norm_",
    "configure_fsdp2_prefetch",
    "fsdp_reduce_dtype",
    "gradient_sync_context",
    "transformer_blocks",
]

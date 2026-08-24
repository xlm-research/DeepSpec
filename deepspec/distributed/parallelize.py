from __future__ import annotations

import torch
from torch import nn

from .config import ParallelConfig
from .expert_parallel import apply_expert_parallelism
from .fsdp import apply_fsdp2, transformer_blocks
from .mesh import ParallelContext
from .tensor_parallel import apply_tensor_parallelism


def apply_activation_checkpoint(model: nn.Module) -> nn.Module:
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable):
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )

    blocks = transformer_blocks(model)
    if not blocks:
        raise ValueError("Activation checkpointing requires discoverable Transformer blocks.")
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = model.model.layers
    for index, block in enumerate(list(layers)):
        layers[index] = checkpoint_wrapper(
            block,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
    return model


def apply_compile(model: nn.Module) -> nn.Module:
    blocks = transformer_blocks(model)
    if not blocks:
        raise ValueError("torch.compile requires discoverable Transformer blocks.")
    # Module.compile preserves module/state-dict FQNs and lets FSDP2 wrap the
    # block after compilation, matching the ordering validated for torch 2.11.
    for block in blocks:
        block.compile(dynamic=True)
    return model


def apply_parallelism(
    model: nn.Module,
    context: ParallelContext,
    config: ParallelConfig,
    *,
    param_dtype: torch.dtype,
    sequence_length: int | None = None,
) -> nn.Module:
    config.validate_model(model, sequence_length=sequence_length)
    model_parallel_hook = getattr(model, "apply_model_parallelism", None)
    if callable(model_parallel_hook):
        model_parallel_hook(context=context, config=config)
    else:
        model = apply_tensor_parallelism(model, context, config)
        model = apply_expert_parallelism(model, context, config)
    if config.use_activation_checkpoint:
        model = apply_activation_checkpoint(model)
    if config.use_compile:
        model = apply_compile(model)
    model = apply_fsdp2(
        model,
        context,
        config,
        param_dtype=param_dtype,
    )
    return model


__all__ = ["apply_activation_checkpoint", "apply_compile", "apply_parallelism"]

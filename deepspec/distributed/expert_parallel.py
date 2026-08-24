from __future__ import annotations

from torch import nn

from .config import ParallelConfig
from .expert_dispatch import NativeExpertDispatcher, resolve_dispatch_backend
from .mesh import ParallelContext


def apply_expert_parallelism(
    model: nn.Module,
    context: ParallelContext,
    config: ParallelConfig,
) -> nn.Module:
    if config.ep == 1:
        return model
    backend = resolve_dispatch_backend(config.expert_dispatch_backend)
    if backend != "native":
        raise RuntimeError("DeepEP is not certified in the current environment.")
    assert context.sparse_mesh is not None
    ep_group = context.sparse_mesh["ep"].get_group()
    adapters = []
    for module in model.modules():
        adapter = getattr(module, "apply_expert_parallel", None)
        if callable(adapter):
            adapters.append(adapter)
    if not adapters:
        raise NotImplementedError(
            "The model advertises MoE experts but exposes no "
            "apply_expert_parallel(dispatcher, expert_mesh) adapter. The stable "
            "native dispatcher is available, but model routing semantics cannot "
            "be inferred safely."
        )
    num_experts = int(
        getattr(model.config, "num_experts", 0)
        or getattr(model.config, "num_local_experts", 0)
        or getattr(model.config, "n_routed_experts", 0)
    )
    dispatcher = NativeExpertDispatcher(group=ep_group, num_experts=num_experts)
    for adapter in adapters:
        adapter(dispatcher=dispatcher, expert_mesh=context.sparse_mesh)
    return model


__all__ = ["apply_expert_parallelism"]

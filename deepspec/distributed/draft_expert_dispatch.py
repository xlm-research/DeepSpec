"""Draft-only selection and ownership of optional expert communication buffers."""

from __future__ import annotations

import warnings

import torch
import torch.distributed as dist


def build_draft_expert_dispatcher(model, *, topology):
    requested = getattr(topology, "expert_dispatch_backend", "native")
    if requested == "native":
        return None
    if requested not in {"deepep", "auto"}:
        raise ValueError(f"Unknown draft expert dispatch backend: {requested!r}.")
    unsupported = (
        topology.expert_parallel_size <= 1
        or topology.tensor_parallel_size != 1
        or topology.context_parallel_size != 1
        or getattr(topology, "use_compile", False)
        or getattr(topology, "use_activation_checkpoint", False)
    )
    if unsupported:
        reason = "DeepEP draft training requires EP>1, TP=CP=1, and eager execution without AC."
        if requested == "deepep":
            raise NotImplementedError(reason)
        warnings.warn(f"Draft EP auto selection uses native: {reason}", stacklevel=2)
        return None

    from .deepep_dispatch import DeepEPDispatcher, require_deepep

    dependency_error = None
    try:
        require_deepep()
    except (ImportError, RuntimeError) as exc:
        dependency_error = exc
    available = dependency_error is None
    if dist.is_initialized():
        # All EP peers must choose the same collectives. In particular, auto
        # cannot silently mix native and DeepEP on heterogeneous node images.
        group = topology.expert_parallel_group
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend(group) == "nccl"
            else torch.device("cpu")
        )
        supported = torch.tensor(int(available), device=device, dtype=torch.int32)
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=group)
        available = bool(supported.item())
    if not available:
        error = dependency_error or RuntimeError(
            "DeepEP is unavailable on another rank in this draft EP group."
        )
        if requested == "deepep":
            raise error
        warnings.warn(f"Draft EP auto selection uses native: {error}", stacklevel=2)
        return None

    return DeepEPDispatcher(
        topology.expert_parallel_group,
        num_experts=int(model.config.n_routed_experts),
        hidden_size=int(model.config.hidden_size),
        top_k=int(model.config.num_experts_per_tok),
        max_tokens_per_rank=int(
            getattr(topology, "expert_dispatch_max_tokens_per_rank", 4096)
        ),
    )


def close_draft_expert_dispatchers(model) -> None:
    """Collectively release each model-owned buffer once, before handing GPUs back."""
    if model is None:
        return
    seen = set()
    for module in model.modules():
        dispatcher = getattr(module, "_deepspec_deepep_dispatcher", None)
        if dispatcher is not None and id(dispatcher) not in seen:
            dispatcher.close()
            seen.add(id(dispatcher))

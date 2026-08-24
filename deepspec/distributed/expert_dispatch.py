from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import warnings

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


@dataclass
class DispatchedTokens:
    tokens: torch.Tensor
    local_expert_indices: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]
    restore_token_indices: torch.Tensor
    restore_routing_weights: torch.Tensor
    num_input_tokens: int


class NativeExpertDispatcher:
    """Autograd-capable top-k token dispatch/combine using all_to_all_single."""

    def __init__(self, *, group, num_experts: int):
        self.group = group
        self.ep_size = dist.get_world_size(group)
        self.num_experts = int(num_experts)
        if self.num_experts < 1 or self.num_experts % self.ep_size:
            raise ValueError(
                f"num_experts={num_experts} must be positive and divisible by EP={self.ep_size}."
            )
        self.local_experts = self.num_experts // self.ep_size

    def _exchange_counts(self, input_splits: list[int], device) -> list[int]:
        send = torch.tensor(input_splits, dtype=torch.int64, device=device)
        receive = torch.empty_like(send)
        dist.all_to_all_single(receive, send, group=self.group)
        return [int(value) for value in receive.cpu().tolist()]

    def dispatch(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> DispatchedTokens:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [tokens, hidden], got {tuple(tokens.shape)}.")
        if expert_indices.shape != routing_weights.shape or expert_indices.ndim != 2:
            raise ValueError("expert_indices and routing_weights must both be [tokens, top_k].")
        if expert_indices.shape[0] != tokens.shape[0]:
            raise ValueError("routing tensors and tokens disagree on token count.")
        if expert_indices.numel() and (
            int(expert_indices.min()) < 0 or int(expert_indices.max()) >= self.num_experts
        ):
            raise ValueError("router emitted an expert index outside [0, num_experts).")

        token_indices = torch.arange(tokens.shape[0], device=tokens.device)
        token_indices = token_indices[:, None].expand_as(expert_indices).reshape(-1)
        flat_experts = expert_indices.reshape(-1).to(torch.long)
        flat_weights = routing_weights.reshape(-1)
        destinations = torch.div(flat_experts, self.local_experts, rounding_mode="floor")
        order = torch.argsort(destinations, stable=True)
        destinations = destinations[order]
        send_tokens = tokens[token_indices[order]].contiguous()
        send_local_experts = (flat_experts[order] % self.local_experts).contiguous()
        input_splits = torch.bincount(
            destinations,
            minlength=self.ep_size,
        ).cpu().tolist()
        input_splits = [int(value) for value in input_splits]
        output_splits = self._exchange_counts(input_splits, tokens.device)
        receive_tokens = tokens.new_empty((sum(output_splits), tokens.shape[-1]))
        receive_experts = torch.empty(
            sum(output_splits), dtype=torch.long, device=tokens.device
        )
        receive_tokens = dist_nn.all_to_all_single(
            receive_tokens,
            send_tokens,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.group,
        )
        dist.all_to_all_single(
            receive_experts,
            send_local_experts,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.group,
        )
        return DispatchedTokens(
            tokens=receive_tokens,
            local_expert_indices=receive_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            restore_token_indices=token_indices[order],
            restore_routing_weights=flat_weights[order],
            num_input_tokens=tokens.shape[0],
        )

    def combine(
        self,
        local_expert_outputs: torch.Tensor,
        dispatched: DispatchedTokens,
    ) -> torch.Tensor:
        if local_expert_outputs.shape != dispatched.tokens.shape:
            raise ValueError(
                "local expert output shape must equal dispatched token shape: "
                f"{tuple(local_expert_outputs.shape)} != {tuple(dispatched.tokens.shape)}."
            )
        returned = local_expert_outputs.new_empty(
            (sum(dispatched.input_splits), local_expert_outputs.shape[-1])
        )
        returned = dist_nn.all_to_all_single(
            returned,
            local_expert_outputs.contiguous(),
            output_split_sizes=dispatched.input_splits,
            input_split_sizes=dispatched.output_splits,
            group=self.group,
        )
        weighted = returned * dispatched.restore_routing_weights.to(returned.dtype).unsqueeze(-1)
        combined = returned.new_zeros(
            dispatched.num_input_tokens,
            returned.shape[-1],
        )
        combined.index_add_(0, dispatched.restore_token_indices, weighted)
        return combined


def deepep_compatibility() -> tuple[bool, str]:
    if importlib.util.find_spec("deep_ep") is None and importlib.util.find_spec("deepep") is None:
        return False, "DeepEP Python package is not installed"
    if not torch.cuda.is_available():
        return False, "DeepEP requires CUDA"
    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:
        return False, f"GPU compute capability {capability} is below the supported Hopper class"
    return False, (
        "DeepEP was found, but this project has no certified DeepEP/CUDA/NCCL "
        "version tuple; explicit adapter certification is required"
    )


def resolve_dispatch_backend(requested: str) -> str:
    if requested == "native":
        return "native"
    compatible, reason = deepep_compatibility()
    if requested == "deepep":
        if not compatible:
            raise RuntimeError(f"DeepEP backend is unavailable: {reason}.")
        return "deepep"
    if requested == "auto":
        if compatible:
            return "deepep"
        warnings.warn(f"DeepEP auto-selection fell back to native: {reason}.", stacklevel=2)
        return "native"
    raise ValueError(f"Unknown expert dispatch backend {requested!r}.")


__all__ = [
    "DispatchedTokens",
    "NativeExpertDispatcher",
    "deepep_compatibility",
    "resolve_dispatch_backend",
]

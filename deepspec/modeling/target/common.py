from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TargetForwardResult:
    """The contiguous target-model token shard owned by one CP rank."""

    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor
    context_start: int = 0


__all__ = ["TargetForwardResult"]


from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TargetForwardResult:
    """The target-model token shard owned by one CP rank.

    ``context_start`` identifies contiguous layouts. Model-native head/tail
    layouts store zero and describe their ordering in the cache manifest.
    """

    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor
    context_start: int = 0


__all__ = ["TargetForwardResult"]

from __future__ import annotations

import torch.distributed as dist


_loss_group = None


def configure_loss_reduction_group(group) -> None:
    """Set the one pre-created group used for token normalization."""

    global _loss_group
    _loss_group = group


def loss_reduction_group():
    return _loss_group


def loss_reduction_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(_loss_group)


__all__ = [
    "configure_loss_reduction_group",
    "loss_reduction_group",
    "loss_reduction_size",
]

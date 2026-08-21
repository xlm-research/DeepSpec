from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.utils.metrics import add_metric


def _selector_position_weights(
    *,
    block_size: int,
    device,
    loss_decay_gamma: Optional[float],
) -> torch.Tensor:
    weights = torch.ones(block_size, device=device, dtype=torch.float32)
    if loss_decay_gamma is not None and float(loss_decay_gamma) > 0.0:
        positions = torch.arange(block_size, device=device, dtype=torch.float32)
        weights = torch.exp(-positions / float(loss_decay_gamma))
    return weights.view(1, 1, block_size)


def compute_dflash2_loss(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: Optional[float],
    ce_loss_alpha: float,
    selector_loss_alpha: float,
    selector_loss_decay_gamma: Optional[float] = None,
) -> torch.Tensor:
    base_loss = compute_dspark_loss(
        outputs=outputs,
        loss_decay_gamma=loss_decay_gamma,
        ce_loss_alpha=float(ce_loss_alpha),
        l1_loss_alpha=0.0,
        confidence_head_alpha=0.0,
    )
    if any(
        value is None
        for value in (
            outputs.selector_scores,
            outputs.selector_target_indices,
            outputs.selector_loss_mask,
            outputs.selector_recall_mask,
        )
    ):
        raise ValueError("DFlash2 forward output is missing candidate-selector data.")

    scores = outputs.selector_scores
    target_indices = outputs.selector_target_indices
    loss_mask = outputs.selector_loss_mask.bool()
    recall_mask = outputs.selector_recall_mask.bool()
    block_size = int(scores.shape[-2])
    top_k = int(scores.shape[-1])
    position_weights = _selector_position_weights(
        block_size=block_size,
        device=scores.device,
        loss_decay_gamma=selector_loss_decay_gamma,
    )
    weights = loss_mask.to(torch.float32) * position_weights
    per_token_loss = F.cross_entropy(
        scores.float().reshape(-1, top_k),
        target_indices.reshape(-1),
        reduction="none",
    ).reshape_as(target_indices)
    selector_loss_num = (per_token_loss * weights).sum()
    selector_loss_den = weights.sum()

    world_size = dist.get_world_size()
    global_selector_den = selector_loss_den.detach().clone()
    if world_size > 1:
        dist.all_reduce(global_selector_den, op=dist.ReduceOp.SUM)
    selector_backward_loss = (
        selector_loss_num / (global_selector_den + 1e-6)
    ) * world_size
    total_loss = base_loss + float(selector_loss_alpha) * selector_backward_loss

    predicted_indices = scores.detach().argmax(dim=-1)
    selected_correct = predicted_indices.eq(target_indices) & loss_mask
    eval_mask = outputs.eval_mask.bool()
    for position in range(block_size):
        add_metric(
            f"selector_recall@{position}",
            recall_mask[..., position].to(torch.float32).sum(),
            den=eval_mask[..., position].to(torch.float32).sum(),
            tag="train",
        )
        add_metric(
            f"selector_accuracy@{position}",
            selected_correct[..., position].to(torch.float32).sum(),
            den=loss_mask[..., position].to(torch.float32).sum(),
            tag="train",
        )
    add_metric(
        "selector_loss",
        selector_loss_num,
        den=selector_loss_den,
        tag="train",
    )
    add_metric(
        "dflash2_loss",
        total_loss.detach(),
        reduction="mean",
        tag="train",
    )
    return total_loss


__all__ = ["compute_dflash2_loss"]

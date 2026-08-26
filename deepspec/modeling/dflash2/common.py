"""DFlash2 dynamic convolution and candidate-path selector.

The module layout and checkpoint keys follow the first-party DFlash2 inference
implementation.  Training-specific helpers are kept here so the same selector
parameters are used for teacher-forced path loss and sequential proposal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _reshape_independent_blocks(hidden: torch.Tensor, block_size: int):
    batch_size, total_length, hidden_size = hidden.shape
    block_size = int(block_size)
    if block_size <= 0 or total_length % block_size != 0:
        raise ValueError(
            "DFlash2 convolution requires a positive block size that divides "
            f"the flattened draft length: length={total_length}, "
            f"block_size={block_size}."
        )
    num_blocks = total_length // block_size
    return (
        hidden.reshape(batch_size * num_blocks, block_size, hidden_size),
        batch_size,
        num_blocks,
    )


def grouped_dynamic_causal_conv(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    batch_size, length, hidden_size = hidden.shape
    group_size = int(group_size)
    if hidden_size % group_size != 0:
        raise ValueError(
            "DFlash2 conv_group_size must divide hidden_size: "
            f"{hidden_size} % {group_size} != 0."
        )
    groups = hidden_size // group_size
    blocks = hidden.reshape(batch_size, length, groups, group_size)
    dynamic = dynamic.reshape(batch_size, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = (
            blocks
            if offset == 0
            else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        )
        kernel = base[offset].reshape(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.reshape_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    """Two-sided wrapper around an Attention or MLP sublayer.

    Each wrapper predicts one dynamic kernel before the sublayer and reuses its
    second half after the sublayer.  Flattened sampled anchors are reshaped into
    independent verification blocks so no convolution crosses anchor borders.
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        if self.kernel_size < 1:
            raise ValueError("conv_kernel_size must be positive.")
        if hidden_size % self.group_size != 0:
            raise ValueError(
                "conv_group_size must divide hidden_size: "
                f"{hidden_size} % {self.group_size} != 0."
            )
        groups = hidden_size // self.group_size
        # Shape and names intentionally match the official DFlash2 checkpoint.
        self.base_kernel = nn.Parameter(
            torch.zeros(2, self.kernel_size, hidden_size)
        )
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * self.kernel_size * groups,
            bias=False,
        )
        self.reset_to_identity()

    @torch.no_grad()
    def reset_to_identity(self):
        self.base_kernel.zero_()
        self.base_kernel[:, 0, :].fill_(1.0)
        self.kernel_projection.weight.zero_()

    def prepare(self, hidden: torch.Tensor, *, block_size: int):
        blocked, batch_size, num_blocks = _reshape_independent_blocks(
            hidden,
            block_size,
        )
        groups = blocked.shape[-1] // self.group_size
        dynamic = self.kernel_projection(blocked).reshape(
            *blocked.shape[:-1],
            2,
            self.kernel_size,
            groups,
        )
        prepared = grouped_dynamic_causal_conv(
            blocked,
            dynamic[..., 0, :, :],
            self.base_kernel[0],
            self.group_size,
        )
        return (
            prepared.reshape(batch_size, num_blocks * int(block_size), -1),
            dynamic[..., 1, :, :],
        )

    def finish(
        self,
        hidden: torch.Tensor,
        dynamic: torch.Tensor,
        *,
        block_size: int,
    ) -> torch.Tensor:
        blocked, batch_size, num_blocks = _reshape_independent_blocks(
            hidden,
            block_size,
        )
        output = grouped_dynamic_causal_conv(
            blocked,
            dynamic,
            self.base_kernel[1],
            self.group_size,
        )
        return output.reshape(batch_size, num_blocks * int(block_size), -1)


class CandidateSelector(nn.Module):
    """Low-rank pairwise selector over each position's top-k candidates."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
        initializer_range: float,
    ):
        super().__init__()
        self.rank = int(rank)
        self.top_k = int(top_k)
        self.initializer_range = float(initializer_range)
        if self.rank < 1:
            raise ValueError("selector_rank must be positive.")
        if not 1 <= self.top_k <= int(vocab_size):
            raise ValueError(
                "selector_top_k must be in [1, vocab_size]: "
                f"top_k={self.top_k}, vocab_size={int(vocab_size)}."
            )
        # These are Parameters rather than Embedding modules because the public
        # DFlash2 safetensors keys omit a trailing `.weight`.
        self.predecessor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), self.rank)
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(int(vocab_size), self.rank)
        )
        self.hidden_projection = nn.Linear(
            int(hidden_size),
            self.rank,
            bias=False,
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Start as an exact no-op over the base DFlash logits."""

        self.predecessor_codebook.normal_(
            mean=0.0, std=self.initializer_range
        )
        # The selector is bilinear. Zeroing one codebook collapses its score
        # to the base logit while still giving this tensor a first-step
        # gradient; the other factors start learning after it moves off zero.
        self.successor_codebook.zero_()
        self.hidden_projection.weight.normal_(
            mean=0.0, std=self.initializer_range
        )

    def pair_scores(
        self,
        *,
        hidden: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the same pairwise transition rule in training and decode."""

        hidden_code = self.hidden_projection(hidden)
        predecessor_code = F.embedding(
            predecessor_ids,
            self.predecessor_codebook,
        )
        successor_code = F.embedding(
            candidate_ids,
            self.successor_codebook,
        )
        pairwise = (
            (predecessor_code * hidden_code).unsqueeze(-2) * successor_code
        ).sum(dim=-1)
        return unary + pairwise

    def _candidate_scores(
        self,
        *,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = min(self.top_k, int(logits.shape[-1]))
        unary, candidates = torch.topk(
            logits,
            top_k,
            dim=-1,
            sorted=False,
        )
        return (
            self.pair_scores(
                hidden=hidden,
                unary=unary,
                candidate_ids=candidates,
                predecessor_ids=predecessor_ids,
            ),
            candidates,
        )

    def training_outputs(
        self,
        *,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        predecessor_ids: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scores, candidates = self._candidate_scores(
            hidden=hidden,
            logits=logits,
            predecessor_ids=predecessor_ids,
        )
        matches = candidates.eq(target_ids.unsqueeze(-1))
        recalled = matches.any(dim=-1) & eval_mask.bool()
        target_indices = matches.to(torch.int64).argmax(dim=-1)
        return {
            "selector_scores": scores,
            "selector_target_indices": target_indices,
            "selector_loss_mask": recalled,
            "selector_recall_mask": recalled,
        }

    def select(
        self,
        *,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        unary, candidates = torch.topk(
            logits,
            min(self.top_k, int(logits.shape[-1])),
            dim=-1,
            sorted=False,
        )
        predecessor = anchor_ids
        path = []
        probability_rows = []
        for position in range(hidden.shape[-2]):
            scores = self.pair_scores(
                hidden=hidden[..., position, :],
                unary=unary[..., position, :],
                candidate_ids=candidates[..., position, :],
                predecessor_ids=predecessor,
            )
            if float(temperature) > 0.0:
                probs = torch.softmax(scores.float() / float(temperature), dim=-1)
                selected = torch.multinomial(
                    probs.reshape(-1, probs.shape[-1]),
                    1,
                ).reshape(probs.shape[:-1])
                probability_rows.append(probs)
            else:
                selected = scores.argmax(dim=-1)
            predecessor = candidates[..., position, :].gather(
                -1,
                selected.unsqueeze(-1),
            ).squeeze(-1)
            path.append(predecessor)
        return (
            torch.stack(path, dim=-1),
            candidates,
            (
                torch.stack(probability_rows, dim=-2)
                if probability_rows
                else None
            ),
        )


__all__ = [
    "CandidateSelector",
    "GroupedDynamicCausalConv",
    "grouped_dynamic_causal_conv",
]

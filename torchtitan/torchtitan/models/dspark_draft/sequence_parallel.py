"""Sequence-local activations with complete, nonduplicated TP gradients."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _gather(value, group):
    size = dist.get_world_size(group)
    length = torch.tensor([value.shape[1]], device=value.device, dtype=torch.int64)
    lengths = [torch.empty_like(length) for _ in range(size)]
    dist.all_gather(lengths, length, group=group)
    lengths = [int(item.item()) for item in lengths]
    width = max(lengths)
    padded = F.pad(value, (0, 0, 0, width - value.shape[1])).contiguous()
    chunks = [torch.empty_like(padded) for _ in range(size)]
    dist.all_gather(chunks, padded, group=group)
    return torch.cat([chunk[:, :n] for chunk, n in zip(chunks, lengths)], dim=1)


class _Scatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value.tensor_split(dist.get_world_size(group), dim=1)[
            dist.get_rank(group)
        ].contiguous()

    @staticmethod
    def backward(ctx, grad):
        return _gather(grad, ctx.group), None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return _gather(value, group)

    @staticmethod
    def backward(ctx, grad):
        # Column projections already reconstruct the complete hidden gradient.
        # SUM here would count the same gradient once per TP peer.
        return grad.tensor_split(dist.get_world_size(ctx.group), dim=1)[
            dist.get_rank(ctx.group)
        ].contiguous(), None


class SequenceLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias, group):
        ctx.save_for_backward(value, weight)
        ctx.group = group
        ctx.has_bias = bias is not None
        return F.linear(value, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        value, weight = ctx.saved_tensors
        grad_input = grad @ weight if ctx.needs_input_grad[0] else None
        # Contract the full token dimension once. Summing rounded BF16 local
        # parameter gradients changes the update compared with the full GEMM.
        full_grad = _gather(grad, ctx.group).reshape(-1, grad.shape[-1])
        full_input = _gather(value, ctx.group).reshape(-1, value.shape[-1])
        grad_weight = full_grad.T @ full_input if ctx.needs_input_grad[1] else None
        return grad_input, grad_weight, full_grad.sum(0) if ctx.has_bias else None, None


class SequenceScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized, weight, group):
        ctx.save_for_backward(normalized, weight)
        ctx.group = group
        return normalized * weight

    @staticmethod
    def backward(ctx, grad):
        normalized, weight = ctx.saved_tensors
        products = _gather(grad * normalized, ctx.group)
        return grad * weight, products.reshape(-1, weight.numel()).sum(0), None


scatter_sequence = _Scatter.apply
gather_sequence = _Gather.apply

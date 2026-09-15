"""DSpark vocabulary reductions that never materialize complete logits."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


class VocabLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, group):
        ctx.save_for_backward(inputs, weight)
        ctx.group = group
        ctx.has_bias = bias is not None
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        inputs, weight = ctx.saved_tensors
        flat = grad.reshape(-1, grad.shape[-1])
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = flat.float() @ weight.float()
            dist.all_reduce(grad_input, group=ctx.group)
            grad_input = grad_input.to(inputs.dtype).reshape_as(inputs)
        grad_weight = (
            flat.T @ inputs.reshape(-1, inputs.shape[-1])
            if ctx.needs_input_grad[1]
            else None
        )
        return grad_input, grad_weight, flat.sum(0) if ctx.has_bias else None, None


def _probabilities(logits, group):
    values = logits.float()
    maximum = values.amax(-1, keepdim=True)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    numerator = (values - maximum).exp()
    denominator = numerator.sum(-1, keepdim=True)
    dist.all_reduce(denominator, group=group)
    return numerator / denominator, maximum + denominator.log()


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, group):
        probabilities, _ = _probabilities(logits, group)
        ctx.save_for_backward(probabilities)
        ctx.group = group
        ctx.dtype = logits.dtype
        return probabilities

    @staticmethod
    def backward(ctx, grad):
        (probabilities,) = ctx.saved_tensors
        contraction = (grad * probabilities).sum(-1, keepdim=True)
        dist.all_reduce(contraction, group=ctx.group)
        return (probabilities * (grad - contraction)).to(ctx.dtype), None


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, group):
        probabilities, normalizer = _probabilities(logits, group)
        start = dist.get_rank(group) * logits.shape[-1]
        local_targets = targets - start
        owns = (local_targets >= 0) & (local_targets < logits.shape[-1])
        indices = local_targets.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)
        selected = logits.float().gather(-1, indices).squeeze(-1) * owns
        dist.all_reduce(selected, group=group)
        ctx.save_for_backward(probabilities, indices, owns)
        ctx.dtype = logits.dtype
        return normalizer.squeeze(-1) - selected

    @staticmethod
    def backward(ctx, grad):
        probabilities, indices, owns = ctx.saved_tensors
        result = probabilities.clone()
        result.scatter_add_(-1, indices, -owns.unsqueeze(-1).to(result.dtype))
        return (result * grad.unsqueeze(-1)).to(ctx.dtype), None, None


class _VocabSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        result = value.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        # The reduced scalar is replicated, so its adjoint is one local copy.
        return grad, None


softmax = _Softmax.apply
cross_entropy = _CrossEntropy.apply
vocab_sum = _VocabSum.apply

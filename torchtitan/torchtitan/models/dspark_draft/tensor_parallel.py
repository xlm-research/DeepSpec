"""Local SPMD computation for the draft's separate context and query streams."""

# X: leading batch/token/head axes, D: hidden width, E: projected width.

import spmd_types as spmd
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.nn.functional as F

from torchtitan.distributed.spmd_types import (
    plain_tensor_to_dtensor_state_dict,
    spmd_distribute_tensor,
)
from torchtitan.models.common.decoder_sharding import dense_param_placement

from .model import Qwen3RMSNorm


def local_parameter(parameter):
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def redistribute(value, group, src, dst):
    return spmd.redistribute(
        value,
        group,
        src=src,
        dst=dst,
        backward_options={"op_dtype": torch.float32},
    )


class _ColumnLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_XD, weight_ED, bias_E, group):
        ctx.save_for_backward(input_XD, weight_ED)
        ctx.group = group
        ctx.has_bias = bias_E is not None
        return F.linear(input_XD, weight_ED, bias_E)

    @staticmethod
    def backward(ctx, grad_XE):
        input_XD, weight_ED = ctx.saved_tensors
        input_TD = input_XD.reshape(-1, input_XD.shape[-1])
        grad_TE = grad_XE.reshape(-1, grad_XE.shape[-1])
        grad_full_TE = None
        if ctx.needs_input_grad[0] or weight_ED.dtype == torch.float32:
            grad_full_TE = redistribute(grad_TE, ctx.group, spmd.S(1), spmd.I)
        grad_input_XD = None
        if ctx.needs_input_grad[0]:
            # Preserve the complete reduction dimension of the retained
            # GEMM. An all-to-all changes which weight axis is local without
            # replicating weights or optimizer state across the TP group.
            weight_ED_local = redistribute(weight_ED, ctx.group, spmd.S(0), spmd.S(1))
            grad_input_TD = grad_full_TE @ weight_ED_local
            grad_input_XD = redistribute(
                grad_input_TD, ctx.group, spmd.S(1), spmd.I
            ).reshape_as(input_XD)
        grad_weight_ED = None
        if ctx.needs_input_grad[1] and weight_ED.dtype == torch.float32:
            input_TD_local = input_TD.chunk(dist.get_world_size(ctx.group), dim=1)[
                dist.get_rank(ctx.group)
            ]
            grad_weight_ED = redistribute(
                grad_full_TE.T @ input_TD_local,
                ctx.group,
                spmd.S(1),
                spmd.S(0),
            )
        elif ctx.needs_input_grad[1]:
            grad_weight_ED = grad_TE.T @ input_TD
        grad_bias_E = grad_TE.sum(0) if ctx.has_bias else None
        return grad_input_XD, grad_weight_ED, grad_bias_E, None


class _RowLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_XD, weight_ED, group):
        ctx.save_for_backward(input_XD, weight_ED)
        if input_XD.dtype == torch.float32:
            input_XD = redistribute(input_XD, group, spmd.S(input_XD.ndim - 1), spmd.I)
            weight_ED = redistribute(weight_ED, group, spmd.S(1), spmd.S(0))
            output_XE = F.linear(input_XD, weight_ED)
            return redistribute(output_XE, group, spmd.S(output_XE.ndim - 1), spmd.I)
        output_XE = F.linear(input_XD.float(), weight_ED.float())
        return redistribute(output_XE, group, spmd.P, spmd.I).to(input_XD.dtype)

    @staticmethod
    def backward(ctx, grad_XE):
        input_XD, weight_ED = ctx.saved_tensors
        input_TD = input_XD.reshape(-1, input_XD.shape[-1])
        grad_TE = grad_XE.reshape(-1, grad_XE.shape[-1])
        # Both contractions already have their complete reduction dimension.
        # Use the retained BF16 GEMM instead of an FP32 GEMM plus a later cast.
        grad_input_XD = (grad_TE @ weight_ED).reshape_as(input_XD)
        grad_weight_ED = grad_TE.T @ input_TD
        return grad_input_XD, grad_weight_ED, None


class _HeadScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized_XD, weight_D, group):
        ctx.save_for_backward(normalized_XD, weight_D)
        ctx.group = group
        return normalized_XD * weight_D

    @staticmethod
    def backward(ctx, grad_XD):
        normalized_XD, weight_D = ctx.saved_tensors
        grad_normalized_XD = grad_XD * weight_D
        product_XD = grad_XD * normalized_XD
        if weight_D.dtype == torch.float32:
            product_XD = redistribute(
                product_XD, ctx.group, spmd.S(product_XD.ndim - 2), spmd.I
            )
        grad_weight_D = product_XD.reshape(-1, weight_D.numel()).sum(
            0, dtype=torch.float32
        )
        if weight_D.dtype == torch.bfloat16:
            dist.all_reduce(grad_weight_D, group=ctx.group)
        return grad_normalized_XD, grad_weight_D.to(weight_D.dtype), None


class DraftLinear(nn.Linear):
    def __init__(self, source, *, group, kind):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device="meta",
            dtype=source.weight.dtype,
        )
        self.weight = source.weight
        self.bias = source.bias
        self.group = group
        self.kind = kind

    def forward(self, input_XD):
        weight_ED = local_parameter(self.weight)
        bias_E = local_parameter(self.bias) if self.bias is not None else None
        if self.kind == "column":
            return _ColumnLinear.apply(input_XD, weight_ED, bias_E, self.group)
        if self.kind == "row":
            return _RowLinear.apply(input_XD, weight_ED, self.group)
        return F.linear(input_XD, weight_ED, bias_E)


class DraftEmbedding(nn.Embedding):
    def __init__(self, source):
        super().__init__(
            source.num_embeddings,
            source.embedding_dim,
            padding_idx=source.padding_idx,
            device="meta",
            dtype=source.weight.dtype,
        )
        self.weight = source.weight

    def forward(self, input_X):
        return F.embedding(input_X, local_parameter(self.weight), self.padding_idx)


class DraftNorm(Qwen3RMSNorm):
    def __init__(self, source, *, group, partial):
        nn.Module.__init__(self)
        self.weight = source.weight
        self.variance_epsilon = source.variance_epsilon
        self.group = group
        self.partial = partial

    def forward(self, input_XD):
        input_dtype = input_XD.dtype
        hidden_XD = input_XD.float()
        variance_X1 = hidden_XD.pow(2).mean(-1, keepdim=True)
        hidden_XD = hidden_XD * torch.rsqrt(variance_X1 + self.variance_epsilon)
        weight_D = local_parameter(self.weight)
        if self.partial:
            return _HeadScale.apply(hidden_XD.to(input_dtype), weight_D, self.group)
        return weight_D * hidden_XD.to(input_dtype)


def apply_tensor_parallel(model, parallel_dims):
    size = parallel_dims.tp
    config = model.config
    if any(
        value % size
        for value in (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.intermediate_size,
            config.hidden_size,
        )
    ):
        raise ValueError(
            "TP must divide query heads, KV heads, hidden and feed-forward widths"
        )
    group = parallel_dims.get_mesh("tp").get_group()
    layouts = {}
    for name, module in list(model.named_modules()):
        if not name:
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        if isinstance(module, nn.Linear):
            kind = "replicated"
            if name.startswith("layers."):
                if attribute in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                    kind = "column"
                elif attribute in ("o_proj", "down_proj"):
                    kind = "row"
            if kind == "row" and module.bias is not None:
                raise ValueError(
                    "The draft TP row projection requires bias-free weights"
                )
            setattr(parent, attribute, DraftLinear(module, group=group, kind=kind))
            placement = (
                spmd.S(0)
                if kind == "column"
                else spmd.S(1)
                if kind == "row"
                else spmd.I
            )
            layouts[f"{name}.weight"] = dense_param_placement(tp=placement)
            if module.bias is not None:
                layouts[f"{name}.bias"] = dense_param_placement(
                    tp=spmd.S(0) if kind == "column" else spmd.I
                )
        elif isinstance(module, nn.Embedding):
            setattr(parent, attribute, DraftEmbedding(module))
            layouts[f"{name}.weight"] = dense_param_placement(tp=spmd.I)
        elif isinstance(module, Qwen3RMSNorm):
            setattr(
                parent,
                attribute,
                DraftNorm(
                    module, group=group, partial=attribute in ("q_norm", "k_norm")
                ),
            )
            layouts[f"{name}.weight"] = dense_param_placement(tp=spmd.I)
    parameters = dict(model.named_parameters())
    if parameters.keys() != layouts.keys():
        raise ValueError("Every draft parameter requires an explicit TP storage layout")
    mesh = parallel_dims.spmd_dense_mesh()
    shards = {
        name: spmd_distribute_tensor(parameter.detach(), mesh, layouts[name])
        for name, parameter in parameters.items()
    }
    # The pinned FSDP API requires DTensor storage for an n-D mesh. Computation
    # remains local SPMD; this bridge supplies global shapes to FSDP and DCP.
    storage = plain_tensor_to_dtensor_state_dict(
        shards, state_dict_layouts=layouts, parallel_dims=parallel_dims
    )
    for name, parameter in parameters.items():
        parent_name, _, attribute = name.rpartition(".")
        model.get_submodule(parent_name).register_parameter(
            attribute,
            nn.Parameter(storage[name], requires_grad=parameter.requires_grad),
        )
    for layer in model.layers:
        layer.self_attn.tensor_parallel_size = size
        layer.self_attn.num_attention_heads //= size
        layer.self_attn.num_key_value_heads //= size

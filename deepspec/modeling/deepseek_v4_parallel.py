"""DeepSeek-V4 tensor/expert parallel parameter and execution adapters.

The adapter is intentionally applied before FSDP.  EP first slices the expert
axis, TP then slices attention heads, expert intermediate channels, shared MLP
channels, and vocabulary projections.  FSDP consequently shards only the
rank-local ``(ep, tp)`` parameter partition.

DeepSeek-V4 uses one shared KV head, so the KV/compressor branch stays
replicated while query heads and grouped output projections are tensor-sharded.
The small replicated branch receives the complete TP gradient through the
autograd-aware collectives below.
"""

from __future__ import annotations

import copy
import os
from types import MethodType

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import nn


class _AllReduceForward(torch.autograd.Function):
    """Sum in forward and leave the rank-local gradient unchanged."""

    @staticmethod
    def forward(ctx, tensor, group, size):
        ctx.group = group
        ctx.size = int(size)
        if ctx.size == 1:
            return tensor
        output = tensor.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _AllReduceBackward(torch.autograd.Function):
    """Identity in forward and sum gradient contributions in backward."""

    @staticmethod
    def forward(ctx, tensor, group, size):
        ctx.group = group
        ctx.size = int(size)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.size == 1:
            return grad_output, None, None
        gradient = grad_output.contiguous().clone()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=ctx.group)
        return gradient, None, None


class _AllGatherLastDim(torch.autograd.Function):
    """Gather vocabulary shards in forward and select the local grad in backward."""

    @staticmethod
    def forward(ctx, tensor, group, rank, size):
        ctx.rank = int(rank)
        ctx.size = int(size)
        if ctx.size == 1:
            return tensor
        pieces = [torch.empty_like(tensor) for _ in range(ctx.size)]
        dist.all_gather(pieces, tensor.contiguous(), group=group)
        return torch.cat(pieces, dim=-1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.size == 1:
            return grad_output, None, None, None
        pieces = grad_output.chunk(ctx.size, dim=-1)
        return pieces[ctx.rank].contiguous(), None, None, None


def _balanced_split_sizes(total: int, size: int) -> list[int]:
    """Split a leading dimension into deterministic near-equal chunks."""

    base, remainder = divmod(int(total), int(size))
    return [base + int(rank < remainder) for rank in range(int(size))]


class _ReplicatedFirstDimShard(torch.autograd.Function):
    """Select one EP token shard and restore the full input gradient.

    Activations entering an MoE block are replicated across EP ranks.  Each
    rank dispatches only its deterministic token shard.  In backward the
    non-overlapping token gradients are gathered so every replicated upstream
    branch receives the complete gradient exactly once.
    """

    @staticmethod
    def forward(ctx, tensor, group, rank, size):
        ctx.group = group
        ctx.rank = int(rank)
        ctx.size = int(size)
        ctx.total = int(tensor.shape[0])
        ctx.trailing_shape = tuple(tensor.shape[1:])
        if ctx.size == 1:
            return tensor
        ctx.split_sizes = _balanced_split_sizes(ctx.total, ctx.size)
        start = sum(ctx.split_sizes[: ctx.rank])
        return tensor.narrow(0, start, ctx.split_sizes[ctx.rank]).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.size == 1:
            return grad_output, None, None, None
        max_size = max(ctx.split_sizes)
        padded_shape = (max_size, *ctx.trailing_shape)
        padded = grad_output.new_zeros(padded_shape)
        local_size = ctx.split_sizes[ctx.rank]
        if local_size:
            padded[:local_size].copy_(grad_output)
        gathered = [torch.empty_like(padded) for _ in range(ctx.size)]
        dist.all_gather(gathered, padded, group=ctx.group)
        full_gradient = torch.cat(
            [piece[:length] for piece, length in zip(gathered, ctx.split_sizes)],
            dim=0,
        )
        return full_gradient.contiguous(), None, None, None


class _GatherFirstDimNoReduce(torch.autograd.Function):
    """Gather EP token shards, selecting rather than summing in backward.

    The gathered MoE output is replicated across EP ranks.  Those ranks run
    identical downstream computation, so one local slice of the replicated
    output gradient is the exact gradient for the source token shard.  Summing
    all copies here would incorrectly multiply gradients by EP size.
    """

    @staticmethod
    def forward(ctx, tensor, group, split_sizes, rank, size):
        ctx.group = group
        ctx.rank = int(rank)
        ctx.size = int(size)
        ctx.split_sizes = tuple(int(length) for length in split_sizes)
        if ctx.size == 1:
            return tensor
        max_size = max(ctx.split_sizes)
        padded = tensor.new_zeros((max_size, *tensor.shape[1:]))
        if tensor.shape[0]:
            padded[: tensor.shape[0]].copy_(tensor)
        gathered = [torch.empty_like(padded) for _ in range(ctx.size)]
        dist.all_gather(gathered, padded, group=group)
        return torch.cat(
            [piece[:length] for piece, length in zip(gathered, ctx.split_sizes)],
            dim=0,
        ).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.size == 1:
            return grad_output, None, None, None, None
        start = sum(ctx.split_sizes[: ctx.rank])
        local = grad_output.narrow(
            0, start, ctx.split_sizes[ctx.rank]
        ).contiguous()
        return local, None, None, None, None


def _all_to_all_variable(
    tensor: torch.Tensor,
    *,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group,
    size: int,
) -> torch.Tensor:
    """Autograd-aware variable-split All-to-All along dimension zero."""

    if int(size) == 1:
        return tensor
    output = tensor.new_empty((sum(output_split_sizes), *tensor.shape[1:]))
    return dist_nn.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )


def _all_to_all_indices(
    tensor: torch.Tensor,
    *,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group,
    size: int,
) -> torch.Tensor:
    """Variable-split All-to-All for non-differentiable routing metadata."""

    if int(size) == 1:
        return tensor
    output = tensor.new_empty((sum(output_split_sizes), *tensor.shape[1:]))
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def _all_reduce_forward(tensor, *, group, size):
    return _AllReduceForward.apply(tensor, group, int(size))


def _all_reduce_backward(tensor, *, group, size):
    return _AllReduceBackward.apply(tensor, group, int(size))


def _require_divisible(value: int, size: int, name: str) -> int:
    value = int(value)
    size = int(size)
    if value % size != 0:
        raise ValueError(f"{name}={value} must be divisible by parallel size {size}.")
    return value // size


def _replace_parameter(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    original = module._parameters[name]
    module._parameters[name] = nn.Parameter(
        tensor.contiguous(), requires_grad=bool(original.requires_grad)
    )


def _slice_parameter(
    module: nn.Module,
    name: str,
    *,
    dim: int,
    rank: int,
    size: int,
) -> None:
    if int(size) == 1:
        return
    parameter = module._parameters[name]
    local_size = _require_divisible(
        parameter.shape[dim], size, f"{type(module).__name__}.{name}.shape[{dim}]"
    )
    start = int(rank) * local_size
    _replace_parameter(module, name, parameter.narrow(dim, start, local_size))


def _slice_packed_gate_up(
    module: nn.Module,
    name: str,
    *,
    dim: int,
    rank: int,
    size: int,
) -> None:
    if int(size) == 1:
        return
    parameter = module._parameters[name]
    packed_size = int(parameter.shape[dim])
    if packed_size % 2 != 0:
        raise ValueError(f"Packed gate/up dimension must be even, got {packed_size}.")
    half = packed_size // 2
    local_half = _require_divisible(half, size, "MoE intermediate size")
    start = int(rank) * local_half
    gate = parameter.narrow(dim, start, local_half)
    up = parameter.narrow(dim, half + start, local_half)
    _replace_parameter(module, name, torch.cat([gate, up], dim=dim))


def _register_column_parallel_input(module, *, group, size) -> None:
    if int(size) == 1:
        return

    def prepare_input(_module, inputs):
        if not inputs:
            return inputs
        return (
            _all_reduce_backward(inputs[0], group=group, size=size),
            *inputs[1:],
        )

    module.register_forward_pre_hook(prepare_input)


def _register_row_parallel_output(module, *, group, size) -> None:
    if int(size) == 1:
        return

    def reduce_output(_module, _inputs, output):
        return _all_reduce_forward(output, group=group, size=size)

    module.register_forward_hook(reduce_output)


def _register_replicated_output_gradient(module, *, group, size) -> None:
    """Sum gradients entering a replicated branch such as the shared KV head."""

    if int(size) == 1:
        return

    def prepare_backward(_module, _inputs, output):
        return _all_reduce_backward(output, group=group, size=size)

    module.register_forward_hook(prepare_backward)


def _column_parallel_linear(linear: nn.Linear, *, group, rank: int, size: int):
    if int(size) == 1:
        return
    _slice_parameter(linear, "weight", dim=0, rank=rank, size=size)
    if linear.bias is not None:
        _slice_parameter(linear, "bias", dim=0, rank=rank, size=size)
    linear.out_features = int(linear.weight.shape[0])
    _register_column_parallel_input(linear, group=group, size=size)
    linear._deepspec_tensor_parallel = "column"


def _row_parallel_linear(
    linear: nn.Linear,
    *,
    group,
    rank: int,
    size: int,
    split_input: bool = False,
):
    if int(size) == 1:
        return
    _slice_parameter(linear, "weight", dim=1, rank=rank, size=size)
    linear.in_features = int(linear.weight.shape[1])
    if linear.bias is not None:
        # A row-parallel bias must only be added after the partial outputs have
        # been reduced. DeepSeek-V4 uses bias-free projections on these paths.
        raise NotImplementedError("Row-parallel linear with bias is not supported.")
    if split_input:
        def split_last_dim(_module, inputs):
            tensor = inputs[0]
            local_width = _require_divisible(
                tensor.shape[-1], size, "row-parallel input width"
            )
            local = tensor.narrow(-1, int(rank) * local_width, local_width)
            return (local.contiguous(), *inputs[1:])

        linear.register_forward_pre_hook(split_last_dim)
    _register_row_parallel_output(linear, group=group, size=size)
    linear._deepspec_tensor_parallel = "row_split" if split_input else "row"


def _parallelize_vocab_embedding(embedding: nn.Embedding, *, topology) -> None:
    size = int(topology.tensor_parallel_size)
    if size == 1:
        return
    rank = int(topology.tensor_parallel_rank)
    global_vocab = int(embedding.weight.shape[0])
    local_vocab = _require_divisible(global_vocab, size, "vocab_size")
    vocab_start = rank * local_vocab
    _slice_parameter(embedding, "weight", dim=0, rank=rank, size=size)
    embedding.num_embeddings = local_vocab
    original_padding_idx = embedding.padding_idx
    embedding.padding_idx = (
        int(original_padding_idx) - vocab_start
        if original_padding_idx is not None
        and vocab_start <= int(original_padding_idx) < vocab_start + local_vocab
        else None
    )

    def vocab_forward(self, input_ids):
        owned = (input_ids >= vocab_start) & (
            input_ids < vocab_start + local_vocab
        )
        local_ids = (input_ids - vocab_start).masked_fill(~owned, 0)
        output = F.embedding(
            local_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output = output * owned.unsqueeze(-1).to(output.dtype)
        return _all_reduce_forward(
            output,
            group=topology.tensor_parallel_group,
            size=size,
        )

    embedding.forward = MethodType(vocab_forward, embedding)
    embedding._deepspec_global_vocab_size = global_vocab
    embedding._deepspec_vocab_start = vocab_start
    embedding._deepspec_tensor_parallel = "vocab_embedding"


def _parallelize_vocab_projection(linear: nn.Linear, *, topology) -> None:
    size = int(topology.tensor_parallel_size)
    if size == 1:
        return
    rank = int(topology.tensor_parallel_rank)
    _slice_parameter(linear, "weight", dim=0, rank=rank, size=size)
    if linear.bias is not None:
        _slice_parameter(linear, "bias", dim=0, rank=rank, size=size)
    linear.out_features = int(linear.weight.shape[0])

    def vocab_projection_forward(self, hidden_states):
        hidden_states = _all_reduce_backward(
            hidden_states,
            group=topology.tensor_parallel_group,
            size=size,
        )
        local_logits = F.linear(hidden_states, self.weight, self.bias)
        return _AllGatherLastDim.apply(
            local_logits,
            topology.tensor_parallel_group,
            rank,
            size,
        )

    linear.forward = MethodType(vocab_projection_forward, linear)
    linear._deepspec_tensor_parallel = "vocab_projection"


def _parallelize_attention(attention: nn.Module, *, topology) -> None:
    size = int(topology.tensor_parallel_size)
    if size == 1:
        attention._deepspec_local_o_groups = int(attention.config.o_groups)
        return
    rank = int(topology.tensor_parallel_rank)
    group = topology.tensor_parallel_group
    global_heads = int(attention.num_heads)
    local_heads = _require_divisible(global_heads, size, "num_attention_heads")
    global_groups = int(attention.config.o_groups)
    local_groups = _require_divisible(global_groups, size, "o_groups")

    _column_parallel_linear(attention.q_b_proj, group=group, rank=rank, size=size)
    _slice_parameter(attention, "sinks", dim=0, rank=rank, size=size)

    # Each TP rank owns contiguous output groups. The per-group input width is
    # unchanged because num_heads / o_groups is invariant under this split.
    o_rank = int(attention.config.o_lora_rank)
    group_start = rank * local_groups
    weight = attention.o_a_proj.weight
    _replace_parameter(
        attention.o_a_proj,
        "weight",
        weight.narrow(0, group_start * o_rank, local_groups * o_rank),
    )
    attention.o_a_proj.n_groups = local_groups
    attention.o_a_proj.out_features = local_groups * o_rank
    attention.o_a_proj._deepspec_tensor_parallel = "group_column"

    _row_parallel_linear(
        attention.o_b_proj,
        group=group,
        rank=rank,
        size=size,
        split_input=False,
    )
    attention.num_heads = local_heads
    attention.num_key_value_groups = local_heads
    attention._deepspec_local_o_groups = local_groups
    # The upstream eager forward reads ``self.config.o_groups`` directly.
    # Give each attention module a private config view so CP=1 also reshapes
    # the local grouped projection correctly without mutating the model config.
    attention.config = copy.copy(attention.config)
    attention.config.o_groups = local_groups

    # V4's single MQA KV head remains replicated. Its gradient is the sum of
    # the contributions made by every local query-head partition.
    _register_replicated_output_gradient(
        attention.kv_norm, group=group, size=size
    )

    compressor = getattr(attention, "compressor", None)
    indexer = None if compressor is None else getattr(compressor, "indexer", None)
    if indexer is not None:
        local_index_heads = _require_divisible(
            int(indexer.num_heads), size, "index_n_heads"
        )
        _column_parallel_linear(
            indexer.q_b_proj, group=group, rank=rank, size=size
        )
        _column_parallel_linear(
            indexer.scorer.weights_proj, group=group, rank=rank, size=size
        )
        indexer.num_heads = local_index_heads
        _register_row_parallel_output(indexer.scorer, group=group, size=size)
        indexer._deepspec_tensor_parallel_group = group
        indexer._deepspec_tensor_parallel_size = size


def _parallelize_shared_mlp(mlp: nn.Module, *, topology) -> None:
    size = int(topology.tensor_parallel_size)
    if size == 1:
        return
    rank = int(topology.tensor_parallel_rank)
    group = topology.tensor_parallel_group
    _column_parallel_linear(mlp.gate_proj, group=group, rank=rank, size=size)
    _column_parallel_linear(mlp.up_proj, group=group, rank=rank, size=size)
    _row_parallel_linear(
        mlp.down_proj,
        group=group,
        rank=rank,
        size=size,
        split_input=False,
    )
    mlp.intermediate_size = int(mlp.gate_proj.out_features)


def _parallelize_moe(moe: nn.Module, *, topology) -> None:
    ep_size = int(topology.expert_parallel_size)
    ep_rank = int(topology.expert_parallel_rank)
    tp_size = int(topology.tensor_parallel_size)
    tp_rank = int(topology.tensor_parallel_rank)
    ep_group = topology.expert_parallel_group
    tp_group = topology.tensor_parallel_group
    experts = moe.experts
    global_experts = int(experts.num_experts)
    local_experts = _require_divisible(
        global_experts, ep_size, "n_routed_experts"
    )

    if ep_size > 1:
        original_experts_forward = experts.forward
        _slice_parameter(
            experts, "gate_up_proj", dim=0, rank=ep_rank, size=ep_size
        )
        _slice_parameter(
            experts, "down_proj", dim=0, rank=ep_rank, size=ep_size
        )
        experts.num_experts = local_experts

    if tp_size > 1:
        _slice_packed_gate_up(
            experts,
            "gate_up_proj",
            dim=1,
            rank=tp_rank,
            size=tp_size,
        )
        _slice_parameter(
            experts, "down_proj", dim=2, rank=tp_rank, size=tp_size
        )
        experts.intermediate_dim = _require_divisible(
            int(experts.intermediate_dim), tp_size, "moe_intermediate_size"
        )

    if ep_size > 1:
        configured_chunk = int(
            os.environ.get("DEEPSPEC_V4_EP_TOKEN_CHUNK", "4096")
        )
        if configured_chunk < 1:
            raise ValueError("DEEPSPEC_V4_EP_TOKEN_CHUNK must be positive.")
        token_chunk_size = max(configured_chunk, ep_size)

        def all_to_all_experts_forward(
            self, hidden_states, top_k_index, top_k_weights
        ):
            total_tokens = int(hidden_states.shape[0])
            if int(top_k_index.shape[0]) != total_tokens:
                raise ValueError("MoE routing indices must align with tokens.")

            # Expert TP ranks own partial intermediate channels.  Their input
            # gradient contributions must be summed back into the replicated
            # hidden state.  Router weights are already multiplied after the
            # TP partial expert outputs have been summed, so they need only the
            # EP reduction that joins disjoint source-token ranges.
            hidden_states = _all_reduce_backward(
                hidden_states, group=tp_group, size=tp_size
            )
            top_k_weights = _all_reduce_backward(
                top_k_weights, group=ep_group, size=ep_size
            )

            output_chunks = []
            for chunk_start in range(0, total_tokens, token_chunk_size):
                chunk_end = min(chunk_start + token_chunk_size, total_tokens)
                full_chunk_length = chunk_end - chunk_start
                source_splits = _balanced_split_sizes(
                    full_chunk_length, ep_size
                )
                source_start = sum(source_splits[:ep_rank])
                source_length = source_splits[ep_rank]

                full_hidden_chunk = hidden_states[chunk_start:chunk_end]
                local_hidden = _ReplicatedFirstDimShard.apply(
                    full_hidden_chunk, ep_group, ep_rank, ep_size
                )
                local_indices = top_k_index[
                    chunk_start + source_start :
                    chunk_start + source_start + source_length
                ]
                local_weights = top_k_weights[
                    chunk_start + source_start :
                    chunk_start + source_start + source_length
                ]

                top_k = int(local_indices.shape[-1])
                pair_tokens = torch.arange(
                    source_length,
                    device=hidden_states.device,
                    dtype=torch.long,
                ).repeat_interleave(top_k)
                global_expert = local_indices.reshape(-1)
                destinations = torch.div(
                    global_expert,
                    local_experts,
                    rounding_mode="floor",
                )
                local_expert = torch.remainder(
                    global_expert, local_experts
                )
                order = torch.argsort(destinations, stable=True)
                destinations = destinations[order]
                send_tokens = pair_tokens[order]
                send_experts = local_expert[order].contiguous()
                send_weights = local_weights.reshape(-1)[order]
                send_hidden = local_hidden[send_tokens].contiguous()

                send_counts_tensor = torch.bincount(
                    destinations, minlength=ep_size
                ).to(dtype=torch.int64)
                recv_counts_tensor = torch.empty_like(send_counts_tensor)
                dist.all_to_all_single(
                    recv_counts_tensor,
                    send_counts_tensor,
                    group=ep_group,
                )
                send_counts = [
                    int(value) for value in send_counts_tensor.tolist()
                ]
                recv_counts = [
                    int(value) for value in recv_counts_tensor.tolist()
                ]

                received_hidden = _all_to_all_variable(
                    send_hidden,
                    output_split_sizes=recv_counts,
                    input_split_sizes=send_counts,
                    group=ep_group,
                    size=ep_size,
                )
                received_experts = _all_to_all_indices(
                    send_experts,
                    output_split_sizes=recv_counts,
                    input_split_sizes=send_counts,
                    group=ep_group,
                    size=ep_size,
                )

                # Each received row is one token/expert pair.  Expert routing
                # weights stay on the source rank and are applied after the
                # reverse All-to-All, avoiding another large dispatch tensor.
                unit_weights = received_hidden.new_ones(
                    (received_hidden.shape[0], 1)
                )
                received_output = original_experts_forward(
                    received_hidden,
                    received_experts.view(-1, 1),
                    unit_weights,
                )
                received_output = _all_reduce_forward(
                    received_output, group=tp_group, size=tp_size
                )
                returned_output = _all_to_all_variable(
                    received_output,
                    output_split_sizes=send_counts,
                    input_split_sizes=recv_counts,
                    group=ep_group,
                    size=ep_size,
                )

                local_output = local_hidden.new_zeros(local_hidden.shape)
                local_output.index_add_(
                    0,
                    send_tokens,
                    returned_output
                    * send_weights.to(returned_output.dtype).unsqueeze(-1),
                )
                output_chunks.append(
                    _GatherFirstDimNoReduce.apply(
                        local_output,
                        ep_group,
                        source_splits,
                        ep_rank,
                        ep_size,
                    )
                )

            if not output_chunks:
                return hidden_states.new_empty(hidden_states.shape)
            return torch.cat(output_chunks, dim=0)

        experts.forward = MethodType(all_to_all_experts_forward, experts)
        experts._deepspec_expert_dispatch = "all_to_all"

    elif tp_size > 1:
        def prepare_expert_inputs(_module, inputs):
            hidden_states, top_k_index, top_k_weights = inputs
            hidden_states = _all_reduce_backward(
                hidden_states,
                group=tp_group,
                size=tp_size,
            )
            top_k_weights = _all_reduce_backward(
                top_k_weights,
                group=tp_group,
                size=tp_size,
            )
            return hidden_states, top_k_index, top_k_weights

        def combine_expert_outputs(_module, _inputs, output):
            return _all_reduce_forward(
                output, group=tp_group, size=tp_size
            )

        experts.register_forward_pre_hook(prepare_expert_inputs)
        experts.register_forward_hook(combine_expert_outputs)

    _parallelize_shared_mlp(moe.shared_experts, topology=topology)
    experts._deepspec_expert_parallel_size = ep_size
    experts._deepspec_expert_parallel_rank = ep_rank
    experts._deepspec_tensor_parallel_size = tp_size
    experts._deepspec_tensor_parallel_rank = tp_rank


def _get_backbone(model):
    candidate = getattr(model, "model", model)
    if hasattr(candidate, "language_model"):
        candidate = candidate.language_model
    if not hasattr(candidate, "layers"):
        raise TypeError("DeepSeek-V4 model does not expose decoder layers.")
    return candidate


def parallelize_deepseek_v4_model(model, *, topology, draft: bool = False):
    """Apply DeepSeek-V4 EP/TP slicing and autograd-aware collectives in-place."""

    if getattr(model, "_deepspec_ep_tp_installed", False):
        return model
    if str(model.config.model_type) != "deepseek_v4":
        raise ValueError(
            "DeepSeek-V4 model parallel adapter received model_type="
            f"{model.config.model_type!r}."
        )
    backbone = _get_backbone(model)
    for layer in backbone.layers:
        _parallelize_attention(layer.self_attn, topology=topology)
        _parallelize_moe(layer.mlp, topology=topology)

    embed_tokens = getattr(backbone, "embed_tokens", None)
    if embed_tokens is not None:
        _parallelize_vocab_embedding(embed_tokens, topology=topology)

    if draft:
        if hasattr(backbone, "fc"):
            _row_parallel_linear(
                backbone.fc,
                group=topology.tensor_parallel_group,
                rank=topology.tensor_parallel_rank,
                size=topology.tensor_parallel_size,
                split_input=True,
            )
        if hasattr(backbone, "lm_head"):
            _parallelize_vocab_projection(backbone.lm_head, topology=topology)
        markov_head = getattr(backbone, "markov_head", None)
        if markov_head is not None:
            _parallelize_vocab_embedding(markov_head.markov_w1, topology=topology)
            _parallelize_vocab_projection(markov_head.markov_w2, topology=topology)

    model._deepspec_ep_tp_installed = True
    model._deepspec_parallel_topology = topology
    backbone._deepspec_parallel_topology = topology
    return model


__all__ = ["parallelize_deepseek_v4_model"]

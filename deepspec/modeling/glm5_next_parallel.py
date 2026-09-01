"""Tensor/expert parallel adapters for GLM-5.3 target and DSpark draft models."""

import torch

from deepspec.modeling.deepseek_v4_parallel import (
    _column_parallel_linear,
    _parallelize_moe,
    _parallelize_shared_mlp,
    _parallelize_vocab_embedding,
    _parallelize_vocab_projection,
    _register_replicated_output_gradient,
    _replace_parameter,
    _require_divisible,
    _row_parallel_linear,
    _slice_parameter,
)


def _slice_packed_heads(
    module, name: str, *, segments: int, rank: int, size: int
) -> None:
    parameter = module._parameters[name]
    segment_width = _require_divisible(
        int(parameter.shape[0]), segments, f"{type(module).__name__}.{name}"
    )
    local_width = _require_divisible(segment_width, size, name)
    local_segments = [
        parameter.narrow(0, segment * segment_width + rank * local_width, local_width)
        for segment in range(segments)
    ]
    _replace_parameter(module, name, torch.cat(local_segments, dim=0))


def _parallelize_linear_attention(attention, *, topology) -> None:
    size = int(topology.tensor_parallel_size)
    if size == 1:
        return
    rank = int(topology.tensor_parallel_rank)
    group = topology.tensor_parallel_group
    local_heads = _require_divisible(attention.num_heads, size, "linear_num_heads")

    for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
        _column_parallel_linear(projection, group=group, rank=rank, size=size)
    _slice_packed_heads(
        attention.conv1d,
        "weight",
        segments=3,
        rank=rank,
        size=size,
    )
    attention.conv1d.in_channels //= size
    attention.conv1d.out_channels //= size
    attention.conv1d.groups //= size

    forget_gate = attention.forget_gate
    _column_parallel_linear(
        forget_gate.f_b_proj, group=group, rank=rank, size=size
    )
    _slice_parameter(forget_gate, "dt_bias", dim=0, rank=rank, size=size)
    _slice_parameter(forget_gate, "A_log", dim=0, rank=rank, size=size)
    forget_gate.num_heads = local_heads
    forget_gate.qkv_dim //= size

    _column_parallel_linear(attention.b_proj, group=group, rank=rank, size=size)
    _column_parallel_linear(attention.g_b_proj, group=group, rank=rank, size=size)
    _row_parallel_linear(
        attention.o_proj,
        group=group,
        rank=rank,
        size=size,
        split_input=False,
    )
    attention.num_heads = local_heads
    attention.qkv_dim //= size
    attention.conv_dim //= size


def _parallelize_sparse_attention(attention, *, topology) -> None:
    size = int(topology.tensor_parallel_size)
    if size == 1:
        return
    rank = int(topology.tensor_parallel_rank)
    group = topology.tensor_parallel_group
    local_heads = _require_divisible(
        int(attention.num_heads), size, "num_attention_heads"
    )
    _column_parallel_linear(
        attention.q_b_proj, group=group, rank=rank, size=size
    )
    _slice_parameter(attention, "sinks", dim=0, rank=rank, size=size)
    _row_parallel_linear(
        attention.o_proj,
        group=group,
        rank=rank,
        size=size,
        split_input=False,
    )
    attention.num_heads = local_heads
    _register_replicated_output_gradient(
        attention.kv_norm, group=group, size=size
    )


def _parallelize_attention(attention, *, topology) -> None:
    if hasattr(attention, "q_proj"):
        _parallelize_linear_attention(attention, topology=topology)
    else:
        _parallelize_sparse_attention(attention, topology=topology)


def parallelize_glm5_next_model(model, *, topology, draft: bool = False):
    """Apply GLM-5.3 TP and pure expert parallelism in-place."""

    if getattr(model, "_deepspec_ep_tp_installed", False):
        return model
    if str(model.config.model_type) != "glm5_next_text":
        raise ValueError(
            "GLM-5.3 model parallel adapter received model_type="
            f"{model.config.model_type!r}."
        )
    for layer in model.layers:
        _parallelize_attention(layer.self_attn, topology=topology)
        if hasattr(layer.mlp, "experts"):
            _parallelize_moe(layer.mlp, topology=topology)
        else:
            _parallelize_shared_mlp(layer.mlp, topology=topology)

    _parallelize_vocab_embedding(model.embed_tokens, topology=topology)
    if draft:
        _row_parallel_linear(
            model.fc,
            group=topology.tensor_parallel_group,
            rank=topology.tensor_parallel_rank,
            size=topology.tensor_parallel_size,
            split_input=True,
        )
        _parallelize_vocab_projection(model.lm_head, topology=topology)
        if model.markov_head is not None:
            _parallelize_vocab_embedding(
                model.markov_head.markov_w1, topology=topology
            )
            _parallelize_vocab_projection(
                model.markov_head.markov_w2, topology=topology
            )

    model._deepspec_ep_tp_installed = True
    model._deepspec_parallel_topology = topology
    return model


__all__ = ["parallelize_glm5_next_model"]

"""Tensor/expert parallel adapter for the GLM-5.3 DSpark draft."""

from deepspec.modeling.deepseek_v4_parallel import (
    _column_parallel_linear,
    _parallelize_moe,
    _parallelize_vocab_embedding,
    _parallelize_vocab_projection,
    _register_replicated_output_gradient,
    _require_divisible,
    _row_parallel_linear,
    _slice_parameter,
)


def _parallelize_attention(attention, *, topology) -> None:
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


def parallelize_glm5_next_model(model, *, topology, draft: bool = False):
    """Apply GLM-5.3 draft TP and pure expert parallelism in-place."""

    if getattr(model, "_deepspec_ep_tp_installed", False):
        return model
    if str(model.config.model_type) != "glm5_next_text":
        raise ValueError(
            "GLM-5.3 model parallel adapter received model_type="
            f"{model.config.model_type!r}."
        )
    for layer in model.layers:
        _parallelize_attention(layer.self_attn, topology=topology)
        _parallelize_moe(layer.mlp, topology=topology)

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

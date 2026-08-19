"""CPU correctness smoke for orthogonal DeepSeek-V4 EP + TP."""

import copy
import os
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _tiny_config(*, num_hidden_layers: int, layer_types: list[str]):
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )

    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        q_lora_rank=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        partial_rotary_factor=0.5,
        layer_types=layer_types,
        compress_rates={
            "heavily_compressed_attention": 4,
            "compressed_sparse_attention": 2,
        },
        mlp_layer_types=["moe"] * num_hidden_layers,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=4,
        index_head_dim=2,
        index_topk=2,
        hc_mult=1,
        hc_sinkhorn_iters=2,
        attention_dropout=0.0,
    )
    config.quantization_config = None
    config._attn_implementation = "eager"
    return config


def _test_target_forward(topology):
    from deepspec.modeling.deepseek_v4_parallel import (
        parallelize_deepseek_v4_model,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Model,
    )

    torch.manual_seed(1234)
    config = _tiny_config(
        num_hidden_layers=3,
        layer_types=[
            "sliding_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
    )
    reference = DeepseekV4Model(config).eval()
    parallel = copy.deepcopy(reference).eval()
    input_ids = torch.arange(12).view(1, -1) % config.vocab_size
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
    parallelize_deepseek_v4_model(
        parallel, topology=topology, draft=False
    )
    with torch.no_grad():
        actual = parallel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=2.0e-4, atol=2.0e-4)

    # FSDP padding steps can contain one token per CP rank, which leaves some
    # EP source shards empty. Variable-split All-to-All must still make every
    # rank enter matching collectives and return the exact output.
    tiny_ids = input_ids[:, :1]
    tiny_mask = attention_mask[:, :1]
    with torch.no_grad():
        tiny_expected = reference(
            input_ids=tiny_ids,
            attention_mask=tiny_mask,
            use_cache=False,
        ).last_hidden_state
        tiny_actual = parallel(
            input_ids=tiny_ids,
            attention_mask=tiny_mask,
            use_cache=False,
        ).last_hidden_state
    torch.testing.assert_close(
        tiny_actual, tiny_expected, rtol=2.0e-4, atol=2.0e-4
    )


def _test_draft_forward_backward(topology):
    from deepspec.modeling.dspark.deepseek_v4.modeling import (
        DeepseekV4DSparkModel,
    )

    torch.manual_seed(5678)
    config = _tiny_config(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
    )
    config.target_layer_ids = [0]
    config.mask_token_id = 31
    config.num_anchors = 2
    config.block_size = 2
    config.enable_confidence_head = False
    config.confidence_head_with_markov = False
    config.markov_rank = 0

    reference = DeepseekV4DSparkModel(config)
    parallel = copy.deepcopy(reference)
    input_ids = torch.arange(8).view(1, -1) % (config.vocab_size - 1)
    loss_mask = torch.ones_like(input_ids)
    torch.manual_seed(9012)
    target_hidden = torch.randn(1, 8, config.hidden_size)

    torch.manual_seed(3456)
    expected = reference(
        input_ids=input_ids,
        target_hidden_states=target_hidden,
        target_last_hidden_states=target_hidden,
        loss_mask=loss_mask,
    )
    expected_loss = expected.draft_logits.float().square().mean()
    expected_loss.backward()

    parallel.configure_parallelism(topology)
    # Deliberately desynchronize process-local RNG. The model must broadcast
    # anchor choices from EP0/TP0 so model-parallel ranks still execute the
    # same logical sample and match the single-rank reference.
    torch.manual_seed(3456 + topology.global_rank)
    actual = parallel(
        input_ids=input_ids,
        target_hidden_states=target_hidden,
        target_last_hidden_states=target_hidden,
        loss_mask=loss_mask,
    )
    actual_loss = actual.draft_logits.float().square().mean()
    torch.testing.assert_close(
        actual.draft_logits,
        expected.draft_logits,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    actual_loss.backward()

    # These two tensors are replicated across TP/EP. Their autograd inputs
    # must therefore receive the sum of all local head/expert contributions.
    torch.testing.assert_close(
        parallel.layers[0].self_attn.q_a_proj.weight.grad,
        reference.layers[0].self_attn.q_a_proj.weight.grad,
        rtol=4.0e-4,
        atol=4.0e-4,
    )
    torch.testing.assert_close(
        parallel.layers[0].mlp.gate.weight.grad,
        reference.layers[0].mlp.gate.weight.grad,
        rtol=4.0e-4,
        atol=4.0e-4,
    )

    local_q = parallel.layers[0].self_attn.q_b_proj.weight.grad
    expected_q = reference.layers[0].self_attn.q_b_proj.weight.grad.chunk(
        topology.tensor_parallel_size, dim=0
    )[topology.tensor_parallel_rank]
    torch.testing.assert_close(
        local_q, expected_q, rtol=4.0e-4, atol=4.0e-4
    )

    parallel_experts = parallel.layers[0].mlp.experts
    reference_experts = reference.layers[0].mlp.experts
    assert parallel_experts._deepspec_expert_dispatch == "all_to_all"
    assert parallel_experts._deepspec_pure_expert_parallel
    expected_gate_up = reference_experts.gate_up_proj.grad.chunk(
        topology.expert_parallel_size, dim=0
    )[topology.expert_parallel_rank]
    torch.testing.assert_close(
        parallel_experts.gate_up_proj.grad,
        expected_gate_up,
        rtol=4.0e-4,
        atol=4.0e-4,
    )
    expected_down = reference_experts.down_proj.grad.chunk(
        topology.expert_parallel_size, dim=0
    )[topology.expert_parallel_rank]
    torch.testing.assert_close(
        parallel_experts.down_proj.grad,
        expected_down,
        rtol=4.0e-4,
        atol=4.0e-4,
    )


def _test_context_and_tensor_parallel(topology):
    from deepspec.modeling.deepseek_v4_parallel import (
        parallelize_deepseek_v4_model,
    )
    from deepspec.modeling.dspark.deepseek_v4.modeling import (
        DeepseekV4DSparkModel,
    )
    from deepspec.modeling.target import (
        install_deepseek_v4_ring_context_parallel,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Model,
    )

    torch.manual_seed(2468)
    target_config = _tiny_config(
        num_hidden_layers=3,
        layer_types=[
            "sliding_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
    )
    target_reference = DeepseekV4Model(target_config).eval()
    target_parallel = copy.deepcopy(target_reference).eval()
    input_ids = torch.arange(12).view(1, -1) % target_config.vocab_size
    attention_mask = torch.ones_like(input_ids)
    captured = {}
    handles = []
    for layer_idx, layer in enumerate(target_reference.layers):
        def capture(_module, _inputs, output, *, index=layer_idx):
            captured[index] = target_reference.hc_head(output).detach()

        handles.append(layer.register_forward_hook(capture))
    with torch.no_grad():
        expected_last = target_reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
    for handle in handles:
        handle.remove()
    expected_hidden = torch.cat(
        [captured[index] for index in range(3)], dim=-1
    )

    parallelize_deepseek_v4_model(
        target_parallel, topology=topology, draft=False
    )
    install_deepseek_v4_ring_context_parallel(target_parallel)
    with torch.no_grad():
        result = target_parallel.forward_context_parallel(
            model_inputs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            target_layer_ids=[0, 1, 2],
            context_parallel_group=topology.context_parallel_group,
            context_parallel_rank=topology.context_parallel_rank,
            context_parallel_size=topology.context_parallel_size,
            tensor_parallel_group=topology.tensor_parallel_group,
            tensor_parallel_rank=topology.tensor_parallel_rank,
            tensor_parallel_size=topology.tensor_parallel_size,
            device=torch.device("cpu"),
        )
    start = int(result.context_start)
    local_length = int(result.target_last_hidden_states.shape[1])
    torch.testing.assert_close(
        result.target_last_hidden_states,
        expected_last[:, start : start + local_length],
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    torch.testing.assert_close(
        result.target_hidden_states,
        expected_hidden[:, start : start + local_length],
        rtol=2.0e-4,
        atol=2.0e-4,
    )

    torch.manual_seed(1357)
    draft_config = _tiny_config(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
    )
    draft_config.target_layer_ids = [0]
    draft_config.mask_token_id = 31
    draft_config.num_anchors = 2
    draft_config.block_size = 2
    draft_config.enable_confidence_head = False
    draft_config.confidence_head_with_markov = False
    draft_config.markov_rank = 0
    draft_reference = DeepseekV4DSparkModel(draft_config)
    draft_parallel = copy.deepcopy(draft_reference)
    draft_reference.configure_context_parallel(
        size=topology.context_parallel_size,
        rank=topology.context_parallel_rank,
        group=topology.context_parallel_group,
    )
    draft_parallel.configure_parallelism(topology)

    draft_ids = input_ids[:, :8]
    torch.manual_seed(9753)
    full_target = torch.randn(1, 8, draft_config.hidden_size)
    local_start = topology.context_parallel_rank * 4
    local_target = full_target[:, local_start : local_start + 4]
    forward_kwargs = dict(
        input_ids=draft_ids,
        target_hidden_states=local_target,
        target_last_hidden_states=local_target,
        loss_mask=torch.ones_like(draft_ids),
        context_start=torch.tensor([local_start]),
        context_len=torch.tensor([4]),
        seq_len=torch.tensor([8]),
    )
    torch.manual_seed(8642)
    expected_draft = draft_reference(**forward_kwargs)
    torch.manual_seed(8642)
    actual_draft = draft_parallel(**forward_kwargs)
    torch.testing.assert_close(
        actual_draft.draft_logits,
        expected_draft.draft_logits,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    actual_draft.draft_logits.float().square().mean().backward()


def _worker(rank: int, world_size: int, rendezvous: str):
    from deepspec.utils.parallel import build_parallel_topology

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        topology = build_parallel_topology(
            context_parallel_size=1,
            expert_parallel_size=2,
            tensor_parallel_size=2,
            fsdp_size=1,
            create_fsdp_groups=False,
        )
        assert topology.sample_parallel_size == 1
        assert topology.sample_parallel_rank == 0
        assert topology.expert_parallel_rank == rank // 2
        assert topology.tensor_parallel_rank == rank % 2
        _test_target_forward(topology)
        _test_draft_forward_backward(topology)
        cp_tp_topology = build_parallel_topology(
            context_parallel_size=2,
            expert_parallel_size=1,
            tensor_parallel_size=2,
            fsdp_size=1,
            create_fsdp_groups=False,
        )
        assert cp_tp_topology.context_parallel_rank == rank // 2
        assert cp_tp_topology.tensor_parallel_rank == rank % 2
        _test_context_and_tensor_parallel(cp_tp_topology)
        fsdp_topology = build_parallel_topology(
            context_parallel_size=1,
            expert_parallel_size=1,
            tensor_parallel_size=1,
            fsdp_size=2,
            create_fsdp_groups=True,
        )
        assert fsdp_topology.fsdp_rank == rank % 2
        assert fsdp_topology.data_parallel_rank == rank // 2
        assert fsdp_topology.fsdp_replica_size == 2
        shard_sum = torch.tensor(float(rank))
        dist.all_reduce(shard_sum, group=fsdp_topology.fsdp_group)
        expected_shard_sum = 1.0 if rank < 2 else 5.0
        assert shard_sum.item() == expected_shard_sum
        replica_sum = torch.tensor(1.0)
        dist.all_reduce(
            replica_sum, group=fsdp_topology.fsdp_replica_group
        )
        assert replica_sum.item() == 2.0

        ep_fsdp_topology = build_parallel_topology(
            context_parallel_size=1,
            expert_parallel_size=2,
            tensor_parallel_size=1,
            fsdp_size=2,
            create_fsdp_groups=True,
        )
        assert ep_fsdp_topology.expert_parallel_rank == rank // 2
        assert ep_fsdp_topology.fsdp_rank == rank % 2
        assert ep_fsdp_topology.sample_parallel_size == 2
        assert ep_fsdp_topology.sample_parallel_rank == rank % 2
        fsdp_axis_sum = torch.tensor(float(rank))
        dist.all_reduce(
            fsdp_axis_sum, group=ep_fsdp_topology.fsdp_group
        )
        assert fsdp_axis_sum.item() == expected_shard_sum
        expert_axis_sum = torch.tensor(float(rank))
        dist.all_reduce(
            expert_axis_sum, group=ep_fsdp_topology.expert_parallel_group
        )
        expected_expert_sum = 2.0 if rank % 2 == 0 else 4.0
        assert expert_axis_sum.item() == expected_expert_sum
    finally:
        dist.destroy_process_group()


def test_deepseek_v4_ep_tp_slices_meta_parameters():
    from accelerate import init_empty_weights
    from deepspec.modeling.dspark.deepseek_v4.modeling import (
        DeepseekV4DSparkModel,
    )

    config = _tiny_config(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
    )
    config.target_layer_ids = [0]
    config.mask_token_id = 31
    config.num_anchors = 2
    config.block_size = 2
    config.enable_confidence_head = False
    config.confidence_head_with_markov = False
    config.markov_rank = 0
    with init_empty_weights(include_buffers=True):
        model = DeepseekV4DSparkModel(config)
    topology = SimpleNamespace(
        context_parallel_size=1,
        context_parallel_rank=0,
        context_parallel_group=None,
        expert_parallel_size=2,
        expert_parallel_rank=0,
        expert_parallel_group=None,
        tensor_parallel_size=2,
        tensor_parallel_rank=0,
        tensor_parallel_group=None,
        model_parallel_src_rank=0,
    )
    model.configure_parallelism(topology)
    assert all(parameter.is_meta for parameter in model.parameters())
    assert model.layers[0].mlp.experts.num_experts == 2
    assert model.layers[0].self_attn.num_heads == 2
    assert model.embed_tokens.weight.shape == (16, config.hidden_size)


def test_deepseek_v4_ep_tp_matches_single_rank():
    if not dist.is_available():
        return
    fd, rendezvous = tempfile.mkstemp(prefix="deepspec-v4-ep-tp-")
    os.close(fd)
    os.unlink(rendezvous)
    try:
        mp.spawn(_worker, args=(4, rendezvous), nprocs=4, join=True)
    finally:
        if os.path.exists(rendezvous):
            os.unlink(rendezvous)


if __name__ == "__main__":
    test_deepseek_v4_ep_tp_matches_single_rank()

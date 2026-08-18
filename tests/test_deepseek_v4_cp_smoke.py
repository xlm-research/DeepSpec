"""Small CPU correctness smoke for the DeepSeek-V4 Ring CP adapter."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, rendezvous: str):
    from deepspec.modeling.target import (
        install_deepseek_v4_ring_context_parallel,
    )
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Model,
    )

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(1234)
    config = DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        partial_rotary_factor=0.5,
        layer_types=[
            "sliding_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        compress_rates={
            "heavily_compressed_attention": 4,
            "compressed_sparse_attention": 2,
        },
        mlp_layer_types=["moe", "moe", "moe"],
        sliding_window=4,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=4,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    model = DeepseekV4Model(config).eval()
    input_ids = torch.arange(17).view(1, -1) % config.vocab_size
    attention_mask = torch.ones_like(input_ids)
    captured = {}
    handles = []
    for layer_idx, layer in enumerate(model.layers):
        def capture(_module, _inputs, output, *, index=layer_idx):
            captured[index] = model.hc_head(output).detach()

        handles.append(layer.register_forward_hook(capture))
    with torch.no_grad():
        reference = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
    for handle in handles:
        handle.remove()
    reference_hidden = torch.cat(
        [captured[layer_idx] for layer_idx in range(config.num_hidden_layers)],
        dim=-1,
    )
    install_deepseek_v4_ring_context_parallel(model)
    with torch.no_grad():
        result = model.forward_context_parallel(
            model_inputs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            target_layer_ids=[0, 1, 2],
            context_parallel_group=dist.group.WORLD,
            context_parallel_rank=rank,
            context_parallel_size=world_size,
            device=torch.device("cpu"),
        )
    start = result.context_start
    expected = reference[:, start : start + result.target_last_hidden_states.shape[1]]
    torch.testing.assert_close(
        result.target_last_hidden_states,
        expected,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    local_length = 9 if rank == 0 else 8
    assert result.target_hidden_states.shape == (
        1,
        local_length,
        3 * config.hidden_size,
    )
    torch.testing.assert_close(
        result.target_hidden_states,
        reference_hidden[:, start : start + local_length],
        rtol=2.0e-4,
        atol=2.0e-4,
    )

    from deepspec.modeling.dspark.deepseek_v4.modeling import (
        DeepseekV4DSparkModel,
    )

    draft_config = DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        partial_rotary_factor=0.5,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        sliding_window=4,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=4,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    draft_config.target_layer_ids = [0]
    draft_config.mask_token_id = 63
    draft_config.num_anchors = 4
    draft_config.block_size = 3
    draft_config.enable_confidence_head = False
    draft_config.markov_rank = 0
    draft_config.quantization_config = None
    draft = DeepseekV4DSparkModel(draft_config)
    draft.configure_context_parallel(
        size=world_size,
        rank=rank,
        group=dist.group.WORLD,
        model_parallel_group=dist.group.WORLD,
    )
    torch.manual_seed(5678)
    input_ids = input_ids[:, :16]
    full_target = torch.randn(1, 16, draft_config.hidden_size)
    local_target = full_target[:, rank * 8 : (rank + 1) * 8]
    draft_output = draft(
        input_ids=input_ids,
        target_hidden_states=local_target,
        target_last_hidden_states=local_target,
        loss_mask=torch.ones_like(input_ids),
        context_start=torch.tensor([rank * 8]),
        context_len=torch.tensor([8]),
        seq_len=torch.tensor([16]),
    )
    assert draft_output.draft_logits.shape == (1, 2, 3, 64)
    draft_output.draft_logits.float().mean().backward()
    dist.destroy_process_group()


def test_deepseek_v4_ring_cp_matches_single_rank():
    if not dist.is_available():
        return
    fd, rendezvous = tempfile.mkstemp(prefix="deepspec-v4-cp-")
    os.close(fd)
    os.unlink(rendezvous)
    try:
        mp.spawn(_worker, args=(2, rendezvous), nprocs=2, join=True)
    finally:
        if os.path.exists(rendezvous):
            os.unlink(rendezvous)


if __name__ == "__main__":
    test_deepseek_v4_ring_cp_matches_single_rank()

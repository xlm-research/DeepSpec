import copy
import os
import unittest

import torch
import torch.distributed as dist
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4SparseMoeBlock,
)

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.mesh import ParallelContext
from deepspec.modeling.deepseek_v4_parallel import _parallelize_moe
from tests.distributed_test_utils import require_torchrun


class DeepseekV4ExpertDispatchTest(unittest.TestCase):
    def test_unequal_ep_token_counts_across_multiple_chunks(self):
        runtime = require_torchrun(self, world_size=2)
        topology = ParallelContext.build(
            ParallelConfig(dp_shard=2, ep=2),
            device_type=runtime.device.type,
        )
        config = DeepseekV4Config(
            hidden_size=8,
            moe_intermediate_size=4,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            mlp_layer_types=["moe"],
        )
        torch.manual_seed(2026)
        reference = DeepseekV4SparseMoeBlock(config, layer_idx=0).to(
            runtime.device, dtype=torch.float32
        )
        for parameter in reference.parameters():
            torch.nn.init.uniform_(parameter, a=-0.1, b=0.1)
        distributed_moe = copy.deepcopy(reference)

        previous_chunk_size = os.environ.get("DEEPSPEC_V4_EP_TOKEN_CHUNK")
        os.environ["DEEPSPEC_V4_EP_TOKEN_CHUNK"] = "4"
        try:
            _parallelize_moe(distributed_moe, topology=topology)
        finally:
            if previous_chunk_size is None:
                os.environ.pop("DEEPSPEC_V4_EP_TOKEN_CHUNK", None)
            else:
                os.environ["DEEPSPEC_V4_EP_TOKEN_CHUNK"] = previous_chunk_size

        # Rank 0 owns fewer tokens than one chunk; rank 1 owns enough tokens
        # for three chunks. All EP ranks must still issue three identical
        # count/data/combine collective sequences.
        num_tokens = 3 if runtime.global_rank == 0 else 9
        generator = torch.Generator(device=runtime.device).manual_seed(
            1000 + runtime.global_rank
        )
        hidden = torch.randn(
            num_tokens,
            config.hidden_size,
            generator=generator,
            device=runtime.device,
            requires_grad=True,
        )
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        expert_indices = (
            torch.arange(
                num_tokens * config.num_experts_per_tok,
                device=runtime.device,
                dtype=torch.long,
            ).reshape(num_tokens, config.num_experts_per_tok)
            % config.n_routed_experts
        )
        routing_weights = torch.rand(
            num_tokens,
            config.num_experts_per_tok,
            generator=generator,
            device=runtime.device,
        )

        expected = reference.experts(
            reference_hidden, expert_indices, routing_weights
        )
        actual = distributed_moe.experts(
            hidden, expert_indices, routing_weights
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(
            hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6
        )
        dist.barrier()


if __name__ == "__main__":
    unittest.main()

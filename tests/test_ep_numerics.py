import unittest

import torch

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.expert_dispatch import NativeExpertDispatcher
from deepspec.distributed.mesh import ParallelContext
from tests.distributed_test_utils import require_torchrun


class ExpertParallelNumericsTest(unittest.TestCase):
    def test_topk_dispatch_combine_and_backward(self):
        runtime = require_torchrun(self, world_size=2)
        config = ParallelConfig(dp_shard=2, ep=2)
        context = ParallelContext.build(config, device_type=runtime.device.type)
        ep_group = context.sparse_mesh["ep"].get_group()
        dispatcher = NativeExpertDispatcher(group=ep_group, num_experts=4)
        tokens = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=runtime.device)
            + runtime.global_rank
        ).requires_grad_()
        expert_indices = torch.tensor(
            [[runtime.global_rank, 3], [2, 1]], device=runtime.device
        )
        routing_weights = torch.tensor(
            [[0.75, 0.25], [0.4, 0.6]], device=runtime.device
        )
        dispatched = dispatcher.dispatch(tokens, expert_indices, routing_weights)
        ep_rank = context.sparse_mesh["ep"].get_local_rank()
        global_experts = ep_rank * dispatcher.local_experts + dispatched.local_expert_indices
        expert_outputs = dispatched.tokens * (global_experts + 1).to(tokens.dtype).unsqueeze(-1)
        actual = dispatcher.combine(expert_outputs, dispatched)
        expected_scale = (
            routing_weights * (expert_indices + 1).to(routing_weights.dtype)
        ).sum(dim=-1, keepdim=True)
        expected = tokens * expected_scale
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(tokens.grad, expected_scale.expand_as(tokens))


if __name__ == "__main__":
    unittest.main()

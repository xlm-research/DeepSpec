import copy
import unittest

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.context_parallel import FixedContextParallel
from deepspec.distributed.mesh import ParallelContext
from tests.distributed_test_utils import require_torchrun


class _SDPAAttention(nn.Module):
    def __init__(self, hidden=256, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        batch, seq, hidden = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(tensor):
            return tensor.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            heads(q), heads(k), heads(v), is_causal=True
        )
        return self.out(output.transpose(1, 2).reshape(batch, seq, hidden))


class ContextParallelNumericsTest(unittest.TestCase):
    def test_native_cp_loss_and_gradients_match(self):
        runtime = require_torchrun(self, world_size=2)
        torch.manual_seed(17)
        baseline = _SDPAAttention().to(runtime.device)
        parallel_model = copy.deepcopy(baseline)
        config = ParallelConfig(cp=2)
        context = ParallelContext.build(config, device_type=runtime.device.type)
        backend = FixedContextParallel(context, backend="pytorch")
        # Efficient-attention kernels pad LSE tiles to 32 positions, so use a
        # realistic tile-aligned local sequence/head dimension in this test.
        x = torch.randn(2, 128, 256, device=runtime.device)
        expected = baseline(x)
        expected_loss = expected.float().square().sum()
        expected_loss.backward()
        cp_input = x.clone()
        with backend.forward_context(buffers=[cp_input], sequence_dims=[1]):
            actual = parallel_model(cp_input)
            actual_loss = actual.float().square().sum()
            actual_loss.backward()
        reduced_loss = actual_loss.detach().clone()
        dist.all_reduce(reduced_loss, group=context.cp_mesh.get_group())
        torch.testing.assert_close(reduced_loss, expected_loss, rtol=2e-4, atol=2e-4)
        for parameter, expected_parameter in zip(
            parallel_model.parameters(), baseline.parameters()
        ):
            dist.all_reduce(parameter.grad, group=context.cp_mesh.get_group())
            torch.testing.assert_close(
                parameter.grad,
                expected_parameter.grad,
                rtol=3e-4,
                atol=3e-4,
            )


if __name__ == "__main__":
    unittest.main()

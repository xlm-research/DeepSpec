import copy
import unittest

import torch
from torch import nn

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.fsdp import apply_fsdp2, clip_grad_norm_
from deepspec.distributed.mesh import ParallelContext
from deepspec.training.optimizer import BF16Optimizer
from tests.distributed_test_utils import require_torchrun


class _Block(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.linear1 = nn.Linear(hidden, hidden * 2)
        self.linear2 = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        return x + self.linear2(torch.nn.functional.silu(self.linear1(x)))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FSDP2NumericsTest(unittest.TestCase):
    def test_forward_backward_and_update_match(self):
        runtime = require_torchrun(self, world_size=2)
        torch.manual_seed(2026)
        baseline = _Model().to(runtime.device)
        sharded = copy.deepcopy(baseline)
        config = ParallelConfig(dp_shard=2)
        context = ParallelContext.build(config, device_type=runtime.device.type)
        apply_fsdp2(sharded, context, config, param_dtype=torch.float32)
        base_optim = BF16Optimizer(baseline, 1e-3, 2, 0, 0)
        shard_optim = BF16Optimizer(sharded, 1e-3, 2, 0, 0)
        x = torch.randn(3, 4, 8, device=runtime.device)
        expected = baseline(x)
        actual = sharded(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        base_norm = torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
        shard_norm = clip_grad_norm_(sharded, 1.0)
        torch.testing.assert_close(shard_norm, base_norm, rtol=1e-5, atol=1e-5)
        base_optim.step()
        shard_optim.step()
        torch.testing.assert_close(sharded(x), baseline(x), rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()

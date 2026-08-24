import copy
import os
import unittest

import torch
from torch import nn

from deepspec.distributed.parallelize import apply_compile


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16))]
        )

    def forward(self, x):
        return self.layers[0](x)


@unittest.skipUnless("LOCAL_RANK" in os.environ and torch.cuda.is_available(), "run with torchrun")
class CompileTest(unittest.TestCase):
    def test_block_compile_forward_backward(self):
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        torch.manual_seed(5)
        baseline = _Model().to(device)
        compiled = apply_compile(copy.deepcopy(baseline))
        x = torch.randn(4, 16, device=device)
        expected = baseline(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in compiled.parameters()))


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from torch import nn

from deepspec.distributed.parallelize import apply_activation_checkpoint


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))]
        )

    def forward(self, x):
        return self.layers[0](x)


class ActivationCheckpointTest(unittest.TestCase):
    def test_non_reentrant_checkpoint_backward(self):
        model = apply_activation_checkpoint(_Model())
        x = torch.randn(2, 8, requires_grad=True)
        model(x).square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIn("CheckpointWrapper", type(model.layers[0]).__name__)


if __name__ == "__main__":
    unittest.main()

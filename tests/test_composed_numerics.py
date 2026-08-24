import copy
import unittest

import torch

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.fsdp import apply_fsdp2
from deepspec.distributed.mesh import ParallelContext
from deepspec.distributed.tensor_parallel import apply_tensor_parallelism
from deepspec.training.optimizer import BF16Optimizer
from tests.distributed_test_utils import require_torchrun
from tests.test_tp_numerics import _Model


class FSDP2TensorParallelNumericsTest(unittest.TestCase):
    def test_fsdp2_tp_update_matches(self):
        runtime = require_torchrun(self, world_size=4)
        torch.manual_seed(808)
        baseline = _Model().to(runtime.device)
        composed = copy.deepcopy(baseline)
        config = ParallelConfig(dp_shard=2, tp=2)
        context = ParallelContext.build(config, device_type=runtime.device.type)
        apply_tensor_parallelism(composed, context, config)
        apply_fsdp2(composed, context, config, param_dtype=torch.float32)
        baseline_optimizer = BF16Optimizer(baseline, 1e-3, 2, 0, 0)
        composed_optimizer = BF16Optimizer(composed, 1e-3, 2, 0, 0)
        x = torch.randn(2, 4, 8, device=runtime.device)
        expected = baseline(x)
        actual = composed(x)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        baseline_optimizer.step()
        composed_optimizer.step()
        torch.testing.assert_close(composed(x), baseline(x), rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()

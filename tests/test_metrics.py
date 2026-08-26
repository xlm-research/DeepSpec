import unittest

import torch
import torch.distributed as dist

from deepspec.utils.metrics import (
    add_metric,
    configure_reduction_group,
    flush_async,
)
from tests.distributed_test_utils import require_torchrun


class PackedMetricReductionTest(unittest.TestCase):
    def test_async_packed_reductions_preserve_metric_semantics(self):
        runtime = require_torchrun(self, world_size=2)
        configure_reduction_group(None)
        rank_value = torch.tensor(
            float(dist.get_rank() + 1), device=runtime.device
        )
        add_metric("ratio", rank_value, den=rank_value.new_tensor(2.0))
        add_metric("mean", rank_value, reduction="dp_mean")
        add_metric("local", rank_value, reduction="last")

        pending = flush_async()
        # The returned work is genuinely asynchronous: independent CUDA work
        # may be submitted before the host asks for the reduced summary.
        independent = (rank_value + 1.0).square()
        summary = pending.wait()

        self.assertAlmostEqual(summary["train/ratio"], 0.75)
        self.assertAlmostEqual(summary["train/mean"], 1.5)
        self.assertAlmostEqual(
            summary["train/local"], float(dist.get_rank() + 1)
        )
        self.assertTrue(torch.isfinite(independent))


if __name__ == "__main__":
    unittest.main()

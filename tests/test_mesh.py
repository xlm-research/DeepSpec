import unittest

import torch.distributed as dist

from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.mesh import ParallelContext
from tests.distributed_test_utils import require_torchrun


class MeshTest(unittest.TestCase):
    def test_named_dense_and_sparse_views(self):
        runtime = require_torchrun(self, world_size=2)
        config = ParallelConfig(tp=2, ep=2, use_fsdp=False)
        context = ParallelContext.build(config, device_type=runtime.device.type)
        self.assertEqual(tuple(context.dense_mesh.mesh.shape), (1, 1, 1, 2))
        self.assertEqual(context.tp_mesh.size(), 2)
        self.assertEqual(context.dp_mesh.size(), 1)
        self.assertIsNotNone(context.sparse_mesh)
        self.assertEqual(tuple(context.sparse_mesh.mesh.shape), (1, 1, 2))
        self.assertEqual(context.config.dp_shard * context.config.cp * context.config.tp,
                         context.config.expert_fsdp * context.config.ep)
        dist.barrier()


if __name__ == "__main__":
    unittest.main()

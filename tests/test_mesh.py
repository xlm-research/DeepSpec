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

    def test_target_sparse_view_reuses_draft_dense_mesh(self):
        runtime = require_torchrun(self, world_size=2)
        draft_config = ParallelConfig(tp=2, ep=1, use_fsdp=False)
        target_config = ParallelConfig(tp=2, ep=2, use_fsdp=False)
        draft = ParallelContext.build(
            draft_config, device_type=runtime.device.type
        )
        target = draft.with_sparse_config(target_config)
        self.assertIs(target.dense_mesh, draft.dense_mesh)
        self.assertIs(target.cp_mesh, draft.cp_mesh)
        self.assertIs(target.tp_mesh, draft.tp_mesh)
        self.assertIs(target.fsdp_mesh, draft.fsdp_mesh)
        self.assertIsNone(draft.sparse_mesh)
        self.assertIsNotNone(target.sparse_mesh)
        self.assertEqual(tuple(target.sparse_mesh.mesh.shape), (1, 1, 2))
        self.assertEqual(
            target.local_group_dict()["ep"],
            tuple(range(runtime.world_size)),
        )
        dist.barrier()


if __name__ == "__main__":
    unittest.main()

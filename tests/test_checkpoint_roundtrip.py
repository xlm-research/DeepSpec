import os
import shutil
import tempfile
import unittest

import torch
from torch import nn

from deepspec.distributed.distributed_checkpoint import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.fsdp import apply_fsdp2
from deepspec.distributed.mesh import ParallelContext
from deepspec.training.optimizer import BF16Optimizer
from tests.distributed_test_utils import require_torchrun


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, x):
        return self.layers[0](x)


class DistributedCheckpointRoundTripTest(unittest.TestCase):
    @unittest.skipIf("LOCAL_RANK" in os.environ, "single-process no-dist test")
    def test_model_optimizer_scheduler_progress_and_rng(self):
        torch.manual_seed(99)
        model = _Model()
        optimizer = BF16Optimizer(model, 1e-3, 4, 0, 0)
        x = torch.randn(2, 4)
        model(x).square().mean().backward()
        optimizer.step()
        expected = model(x).detach().clone()
        expected_lr = optimizer.get_learning_rate()
        progress = TrainingProgress(
            next_micro_step=6,
            global_step=3,
            epoch=1,
            data_position=12,
            local_batch_size=2,
            saved_world_size=1,
            parallel_config={"dp_shard": 1},
            model_config={"hidden_size": 4},
        )
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer_bundle=optimizer,
                progress=progress,
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            restored = TrainingProgress(
                next_micro_step=0,
                global_step=0,
                epoch=0,
                data_position=0,
                local_batch_size=2,
                saved_world_size=1,
                parallel_config={},
                model_config={},
            )
            load_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer_bundle=optimizer,
                progress=restored,
            )
            torch.testing.assert_close(model(x), expected)
            self.assertEqual(optimizer.get_learning_rate(), expected_lr)
            self.assertEqual(restored.next_micro_step, 6)
            self.assertEqual(restored.data_position, 12)
            self.assertEqual(restored.parallel_config, {"dp_shard": 1})

    def test_fsdp2_same_world_size_roundtrip(self):
        runtime = require_torchrun(self, world_size=2)
        torch.manual_seed(101)
        model = _Model().to(runtime.device)
        config = ParallelConfig(dp_shard=2)
        context = ParallelContext.build(config, device_type=runtime.device.type)
        apply_fsdp2(model, context, config, param_dtype=torch.float32)
        optimizer = BF16Optimizer(model, 1e-3, 4, 0, 0)
        x = torch.randn(2, 4, device=runtime.device)
        model(x).square().mean().backward()
        optimizer.step()
        expected = model(x).detach().clone()
        progress = TrainingProgress(
            next_micro_step=8,
            global_step=4,
            epoch=2,
            data_position=16,
            local_batch_size=2,
            saved_world_size=2,
            parallel_config=config.to_dict(),
            model_config={"hidden_size": 4},
        )
        paths = [None]
        if runtime.global_rank == 0:
            paths[0] = tempfile.mkdtemp(prefix="deepspec-distributed-checkpoint-")
        torch.distributed.broadcast_object_list(paths, src=0)
        checkpoint_dir = paths[0]
        save_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer_bundle=optimizer,
            progress=progress,
        )
        model(x + 1).square().mean().backward()
        optimizer.step()
        restored = TrainingProgress(0, 0, 0, 0, 2, 2, {}, {})
        load_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer_bundle=optimizer,
            progress=restored,
        )
        torch.testing.assert_close(model(x), expected)
        self.assertEqual(restored.next_micro_step, 8)
        torch.distributed.barrier()
        if runtime.global_rank == 0:
            shutil.rmtree(checkpoint_dir)


if __name__ == "__main__":
    unittest.main()

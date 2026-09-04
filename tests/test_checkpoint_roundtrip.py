import copy
import os
import random
import shutil
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from deepspec.distributed.distributed_checkpoint import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
    write_checkpoint_metadata,
)
from deepspec.distributed.config import ParallelConfig
from deepspec.distributed.fsdp import apply_fsdp2
from deepspec.distributed.mesh import ParallelContext
from deepspec.training.optimizer import BF16Optimizer
from deepspec.trainer.ckpt_manager import validate_partition_checkpoint
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
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        random.seed(1234)
        np.random.seed(2345)
        torch.manual_seed(3456)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        expected_python_random = random.random()
        expected_numpy_random = float(np.random.random())
        expected_torch_random = torch.rand(4)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
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
            model(x + 1).square().mean().backward()
            optimizer.step()
            random.seed(9)
            np.random.seed(9)
            torch.manual_seed(9)
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
            self._assert_nested_state_equal(
                optimizer.state_dict(),
                expected_optimizer,
            )
            self.assertEqual(random.random(), expected_python_random)
            self.assertEqual(float(np.random.random()), expected_numpy_random)
            torch.testing.assert_close(torch.rand(4), expected_torch_random)
            self.assertEqual(restored.next_micro_step, 6)
            self.assertEqual(restored.global_step, 3)
            self.assertEqual(restored.epoch, 1)
            self.assertEqual(restored.data_position, 12)
            self.assertEqual(restored.parallel_config, {"dp_shard": 1})

    def _assert_nested_state_equal(self, actual, expected):
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected)
            return
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self._assert_nested_state_equal(actual[key], expected[key])
            return
        if isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self._assert_nested_state_equal(actual_item, expected_item)
            return
        self.assertEqual(actual, expected)

    @unittest.skipIf("LOCAL_RANK" in os.environ, "single-process no-dist test")
    def test_partition_metadata_roundtrips_with_full_training_state(self):
        model = _Model()
        optimizer = BF16Optimizer(model, 1e-3, 4, 0, 0)
        progress = TrainingProgress(
            next_micro_step=4,
            global_step=2,
            epoch=0,
            data_position=8,
            local_batch_size=2,
            saved_world_size=1,
            parallel_config={"dp_shard": 1},
            model_config={"hidden_size": 4},
            partition_id=3,
            partition_start_next_micro_step=2,
            partition_end_next_micro_step=4,
            checkpointed=True,
        )
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer_bundle=optimizer,
                progress=progress,
            )
            restored = TrainingProgress(
                0,
                0,
                0,
                0,
                2,
                1,
                {},
                {},
                partition_id=-1,
                partition_start_next_micro_step=-1,
                partition_end_next_micro_step=-1,
            )
            load_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer_bundle=optimizer,
                progress=restored,
            )
        self.assertEqual(restored.partition_id, 3)
        self.assertEqual(restored.partition_start_next_micro_step, 2)
        self.assertEqual(restored.partition_end_next_micro_step, 4)
        self.assertTrue(restored.checkpointed)

    @unittest.skipIf("LOCAL_RANK" in os.environ, "single-process filesystem test")
    def test_partition_checkpoint_validation_requires_committed_identity(self):
        progress = TrainingProgress(
            next_micro_step=4,
            global_step=2,
            epoch=0,
            data_position=4,
            local_batch_size=1,
            saved_world_size=1,
            parallel_config={},
            model_config={},
            partition_id=3,
            partition_start_next_micro_step=2,
            partition_end_next_micro_step=4,
            checkpointed=True,
        )
        partition = {
            "partition_id": 3,
            "epoch": 0,
            "start_next_micro_step": 2,
            "end_next_micro_step": 4,
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            os.makedirs(os.path.join(checkpoint_dir, "distributed_checkpoint"))
            for relative_path in (
                "train_config.py",
                "config.json",
                "model.safetensors",
                os.path.join("distributed_checkpoint", ".metadata"),
                os.path.join("distributed_checkpoint", "__0_0.distcp"),
            ):
                with open(os.path.join(checkpoint_dir, relative_path), "wb") as handle:
                    handle.write(b"complete")
            write_checkpoint_metadata(checkpoint_dir, progress=progress)
            metadata = validate_partition_checkpoint(
                checkpoint_dir,
                partition_metadata=partition,
                next_micro_step=4,
            )
            self.assertTrue(metadata["checkpointed"])
            wrong_partition = dict(partition, partition_id=4)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_partition_checkpoint(
                    checkpoint_dir,
                    partition_metadata=wrong_partition,
                    next_micro_step=4,
                )
            wrong_epoch = dict(partition, epoch=1)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_partition_checkpoint(
                    checkpoint_dir,
                    partition_metadata=wrong_epoch,
                    next_micro_step=4,
                )

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

import copy
import os
import random
import shutil
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.distributed.tensor import DTensor

from deepspec.distributed.distributed_checkpoint import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
    write_checkpoint_metadata,
    full_model_state_dict,
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
    def _ep_model(self, runtime):
        torch.manual_seed(11)
        model = _Model().to(runtime.device)
        model.experts = nn.Module()
        model.experts.weight = nn.Parameter(
            torch.full((2, 4), float(runtime.global_rank + 1), device=runtime.device)
        )
        model.experts._deepspec_pure_expert_parallel = True
        model.expert_parallel_group = torch.distributed.group.WORLD
        if runtime.device.type == "cuda":
            config = ParallelConfig(dp_shard=2, ep=2, use_fsdp=True)
            parallel = ParallelContext.build(config, device_type="cuda")
            model = apply_fsdp2(model, parallel, config, param_dtype=torch.float32)
        return model

    def test_rank_local_experts_optimizer_and_rng_roundtrip(self):
        runtime = require_torchrun(self, world_size=2)
        model = self._ep_model(runtime)
        optimizer = BF16Optimizer(model, 1e-3, 4, 0, 0)
        loss = (
            model(torch.ones(1, 4, device=runtime.device)).square().mean()
            + model.experts.weight.square().mean()
        )
        loss.backward()
        optimizer.step()
        expected_parameters = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        torch.manual_seed(100 + runtime.global_rank)
        random.seed(200 + runtime.global_rank)
        np.random.seed(300 + runtime.global_rank)
        rng = torch.get_rng_state().clone()
        expected_random = torch.rand(5)
        torch.set_rng_state(rng)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        expected_python, expected_numpy = random.random(), float(np.random.random())
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        progress = TrainingProgress(
            next_micro_step=1,
            global_step=1,
            epoch=0,
            data_position=1,
            local_batch_size=1,
            saved_world_size=2,
            parallel_config={"ep": 2},
            model_config={},
        )
        paths = [
            tempfile.mkdtemp(prefix="deepspec-ep-checkpoint-")
            if runtime.global_rank == 0
            else None
        ]
        torch.distributed.broadcast_object_list(paths, src=0)
        try:
            save_training_checkpoint(
                checkpoint_dir=paths[0],
                model=model,
                optimizer_bundle=optimizer,
                progress=progress,
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(7)
            load_training_checkpoint(
                checkpoint_dir=paths[0],
                model=model,
                optimizer_bundle=optimizer,
                progress=progress,
            )

            def local(value):
                return value.to_local() if isinstance(value, DTensor) else value

            errors = [
                float(
                    (local(p.detach()) - local(expected_parameters[name])).abs().max()
                )
                for name, p in model.named_parameters()
            ]
            actual_optimizer = optimizer.state_dict()
            expected_state = expected_optimizer["optimizer_state_dict"]["state"]
            actual_state = actual_optimizer["optimizer_state_dict"]["state"]
            for key in expected_state:
                for name, expected in expected_state[key].items():
                    if torch.is_tensor(expected):
                        errors.append(
                            float(
                                (local(actual_state[key][name]) - local(expected))
                                .abs()
                                .max()
                            )
                        )
            errors.append(float((torch.rand(5) - expected_random).abs().max()))
            errors.extend(
                [
                    abs(random.random() - expected_python),
                    abs(float(np.random.random()) - expected_numpy),
                ]
            )
            maximum = torch.tensor(max(errors), device=runtime.device)
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
            self.assertEqual(
                maximum.item(),
                0.0,
                "EP weights, optimizer moments, or rank-local RNG were lost",
            )
            legacy = os.path.join(paths[0], "legacy")
            torch.distributed.checkpoint.save(
                {"model": model.state_dict()},
                checkpoint_id=os.path.join(legacy, "distributed_checkpoint"),
            )
            with self.assertRaisesRegex(ValueError, "complete EP tensor"):
                load_training_checkpoint(
                    checkpoint_dir=legacy,
                    model=model,
                    optimizer_bundle=optimizer,
                    progress=progress,
                )
        finally:
            torch.distributed.barrier()
            if runtime.global_rank == 0:
                shutil.rmtree(paths[0])

    def test_full_model_export_gathers_experts_in_ep_rank_order(self):
        runtime = require_torchrun(self, world_size=2)
        model = self._ep_model(runtime)
        state = full_model_state_dict(model)
        error = torch.zeros((), device=runtime.device)
        if runtime.global_rank == 0:
            actual = state["experts.weight"]
            expected = torch.cat([torch.ones(2, 4), torch.full((2, 4), 2.0)])
            error.fill_(
                1
                if actual.shape != expected.shape
                else float((actual - expected).abs().max())
            )
        torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.MAX)
        self.assertEqual(error.item(), 0.0, "HF export must contain all EP experts")

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

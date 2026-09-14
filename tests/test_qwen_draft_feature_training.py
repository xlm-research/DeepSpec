"""Consume fixed producer files through the Qwen draft training entry point."""

import gc
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist

from deepspec.data.draft_feature_reader import DraftFeatureIndex
from deepspec.distributed import ParallelConfig, ParallelContext
from deepspec.trainer.qwen3_8_vllm_trainer import Qwen3_8VllmDSparkTrainer
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.utils.metrics import configure_reduction_group
from tests.distributed_test_utils import require_torchrun
from tests import test_dspark_training_baseline as baseline
from tests.test_dspark_training_baseline import FixedFeatureTrainer


class ReadyFeatureTrainer(FixedFeatureTrainer, Qwen3_8VllmDSparkTrainer):
    def __init__(self, runtime, topology, dtype, *, fixture, index, control_group):
        FixedFeatureTrainer.__init__(self, runtime, topology, dtype, fixture=fixture)
        self.ready_index = index
        self._vllm_control_group = control_group
        self.data_batch_optimizer_aligned = True

    def iter_training_batches(self, batches):
        return self.iter_ready_features(self.ready_index)


class QwenDraftFeatureTrainingTest(unittest.TestCase):
    def test_ready_cp2_features_preserve_real_fsdp_updates(self):
        runtime = require_torchrun(self, world_size=2)
        reference_directory = os.environ.get("DEEPSPEC_BASELINE_REFERENCE")
        if not reference_directory:
            self.skipTest(
                "set DEEPSPEC_BASELINE_REFERENCE to the captured training fixture"
            )
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_ready_features(runtime, reference_directory, dtype)

    def check_ready_features(self, runtime, reference_directory, dtype):
        topology = ParallelContext.build(
            ParallelConfig(dp_shard=2, reduce_dtype="fp32")
        )
        configure_loss_reduction_group(topology.loss_mesh.get_group())
        configure_reduction_group(topology.loss_mesh.get_group())
        control = dist.new_group(backend="gloo")
        local_fixture = torch.load(
            Path(reference_directory) / f"{dtype}_rank{runtime.global_rank}.pt",
            weights_only=True,
        )
        root: list[str | None] = [None]
        temporary = tempfile.TemporaryDirectory() if runtime.global_rank == 0 else None
        if temporary:
            root[0] = temporary.name
        dist.broadcast_object_list(root, src=0, group=control)
        assert root[0] is not None
        directory = Path(root[0])
        identity = {"teacher": "fixed-baseline", "layout": {"cp": 2}}
        if runtime.global_rank == 0:
            samples = []
            for dp_rank in range(2):
                fixture = torch.load(
                    Path(reference_directory) / f"{dtype}_rank{dp_rank}.pt",
                    weights_only=True,
                )
                for micro_step, batch in enumerate(fixture["features"]):
                    position = micro_step * 2 + dp_rank
                    shards = []
                    for cp_rank, positions in enumerate(
                        (list(range(4)) + list(range(12, 16)), list(range(4, 12)))
                    ):
                        shard = {name: tensor.clone() for name, tensor in batch.items()}
                        for name in (
                            "target_hidden_states",
                            "target_last_hidden_states",
                        ):
                            shard[name] = shard[name][:, positions].contiguous()
                        shard["context_chunk_len"] = torch.tensor([8])
                        path = directory / f"sample{position}-cp{cp_rank}.pt"
                        torch.save(shard, path)
                        shards.append(
                            {
                                "path": str(path),
                                "cp_rank": cp_rank,
                                "owner": 1 - dp_rank,
                            }
                        )
                    samples.append(
                        {
                            "position": position,
                            "sample_id": f"sample{position}",
                            "epoch": 0,
                            "shards": shards,
                        }
                    )
            DraftFeatureIndex.create(
                samples=samples,
                producer_identity=identity,
                partition_id=1,
                start_micro_step=0,
                data_parallel_size=2,
                gradient_accumulation_steps=2,
                samples_per_epoch=8,
            ).save(directory / "index.json")
        dist.barrier(group=control)
        index = DraftFeatureIndex.load(
            directory / "index.json",
            producer_identity=identity,
            next_micro_step=0,
            data_parallel_size=2,
            gradient_accumulation_steps=2,
        )
        trainer = ReadyFeatureTrainer(
            runtime,
            topology,
            dtype,
            fixture=local_fixture,
            index=index,
            control_group=control,
        )
        torch.set_rng_state(local_fixture["initial_cpu_rng"])
        torch.cuda.set_rng_state(local_fixture["initial_cuda_rng"], runtime.device)
        actual = trainer.train_and_observe()
        baseline.DSparkTrainingBaselineTest().assert_state_close(
            actual, local_fixture["result"]
        )
        del trainer
        gc.collect()
        # A missing shard on only one consumer must stop all ranks before the
        # first forward/backward or optimizer update can participate.
        if runtime.global_rank == 0:
            (directory / "sample1-cp1.pt").unlink()
        dist.barrier(group=control)
        trainer = ReadyFeatureTrainer(
            runtime,
            topology,
            dtype,
            fixture=local_fixture,
            index=index,
            control_group=control,
        )
        with self.assertRaisesRegex(RuntimeError, "FileNotFoundError"):
            trainer.train_and_observe()
        self.assertEqual(trainer.next_micro_step, 0)
        self.assertEqual(trainer.optimizer.updates, [])
        self.assertEqual(trainer.observations, [])
        trainer._active_prefetcher.close()
        trainer._active_prefetcher = None
        del trainer
        gc.collect()
        dist.barrier(group=control)
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()

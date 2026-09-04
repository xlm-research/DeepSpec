from __future__ import annotations

import math
import os
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist

from deepspec.distributed import ParallelConfig, ParallelContext
from deepspec.trainer import Glm5NextDSparkTrainer
from tests.distributed_test_utils import require_torchrun


class _FakeTarget:
    def forward_training_batch(self, batch):
        sample_id = int(batch["input_ids"][0, 0].item())
        sequence_length = int(batch["input_ids"].shape[1])

        def metadata(value):
            return torch.tensor(
                [value],
                dtype=torch.long,
                device=batch["input_ids"].device,
            )

        hidden = torch.full(
            (1, sequence_length, 2),
            sample_id,
            dtype=torch.float32,
            device=batch["input_ids"].device,
        )
        return {
            "input_ids": batch["input_ids"],
            "loss_mask": batch["loss_mask"],
            "target_hidden_states": hidden,
            "target_last_hidden_states": hidden.clone(),
            "context_start": metadata(0),
            "context_len": metadata(sequence_length),
            "seq_len": metadata(sequence_length),
        }


class HeterogeneousTargetDataBatchTest(unittest.TestCase):
    def test_target_tp4_routes_owner_local_cache_across_two_partitions(self):
        runtime = require_torchrun(self, world_size=None)
        if runtime.world_size % 4:
            self.skipTest("target TP4 requires world_size divisible by four")
        local_world_size = int(
            os.environ.get(
                "DEEPSPEC_TEST_LOCAL_WORLD_SIZE",
                os.environ.get("LOCAL_WORLD_SIZE", runtime.world_size),
            )
        )
        if runtime.world_size % local_world_size:
            self.fail(
                "world_size must be divisible by the effective local world size: "
                f"world_size={runtime.world_size}, "
                f"local_world_size={local_world_size}"
            )
        node_count = runtime.world_size // local_world_size
        draft_config = ParallelConfig(
            dp_replicate=node_count,
            dp_shard=local_world_size,
            ep=math.gcd(local_world_size, 288),
        )
        if local_world_size % 4 == 0:
            target_dp_shard = local_world_size // 4
            target_config = ParallelConfig(
                dp_replicate=node_count,
                dp_shard=target_dp_shard,
                tp=4,
                ep=math.gcd(target_dp_shard * 4, 288),
            )
        else:
            target_dp_shard = runtime.world_size // 4
            target_config = ParallelConfig(
                dp_shard=target_dp_shard,
                tp=4,
                ep=math.gcd(target_dp_shard * 4, 288),
            )
        draft = ParallelContext.build(
            draft_config,
            device_type=runtime.device.type,
        )
        target = ParallelContext.build(
            target_config,
            device_type=runtime.device.type,
        )

        cache_root = [None]
        if runtime.global_rank == 0:
            cache_root[0] = tempfile.mkdtemp(
                prefix=".deepspec-data-batch-test-",
                dir=os.getcwd(),
            )
        dist.broadcast_object_list(cache_root, src=0)

        trainer = object.__new__(Glm5NextDSparkTrainer)
        trainer.args = SimpleNamespace(
            train=SimpleNamespace(local_batch_size=1),
        )
        trainer.device = runtime.device
        trainer.global_rank = runtime.global_rank
        trainer.world_size = runtime.world_size
        trainer.parallel_config = draft_config
        trainer.parallel = draft
        trainer.target_parallel_config = target_config
        trainer.target_parallel = target
        trainer.data_parallel_size = draft.data_parallel_size
        trainer.data_parallel_rank = draft.data_parallel_rank
        trainer.online_target_enabled = False
        trainer.offline_target_data_batches_enabled = True
        trainer.heterogeneous_target_data_batches = True
        trainer.online_target = _FakeTarget()
        trainer.data_batch_size = 2
        trainer.data_batch_micro_batches = (1, 1)
        trainer.data_batch_cache_root = cache_root[0]
        trainer.data_batch_rank_cache_dir = None
        trainer._active_data_batch_cache = None
        trainer._data_batch_phase = None
        trainer._data_batch_end_after_current = False
        trainer._initialize_data_batch_cache()
        dist.barrier()

        write_calls = []
        write_cache_file = trainer._write_data_batch_cache_file

        def tracked_write_cache_file(batch, **kwargs):
            write_calls.append(kwargs)
            return write_cache_file(batch, **kwargs)

        trainer._write_data_batch_cache_file = tracked_write_cache_file

        sequence_length = runtime.global_rank + 1
        raw_batches = []
        for partition in range(2):
            raw_batches.append(
                {
                    "input_ids": torch.full(
                        (1, sequence_length),
                        partition * 100 + runtime.global_rank,
                        dtype=torch.long,
                        device=runtime.device,
                    ),
                    "attention_mask": torch.ones(
                        (1, sequence_length),
                        dtype=torch.long,
                        device=runtime.device,
                    ),
                    "loss_mask": torch.ones(
                        (1, sequence_length),
                        dtype=torch.long,
                        device=runtime.device,
                    ),
                }
            )
        batches = trainer.iter_training_batches(iter(raw_batches))
        prepared = next(batches)
        self.assertEqual(
            tuple(prepared["target_hidden_states"].shape),
            (1, sequence_length, 2),
        )
        self.assertEqual(
            int(prepared["target_hidden_states"][0, 0, 0].item()),
            runtime.global_rank,
        )
        first_partition_path = trainer._data_batch_cache_file_path(
            batch_index=1,
            sample_index=0,
        )
        self.assertTrue(os.path.isfile(first_partition_path))
        self.assertEqual(len(write_calls), 1)

        prepared = next(batches)
        self.assertFalse(os.path.exists(first_partition_path))
        self.assertEqual(
            int(prepared["target_hidden_states"][0, 0, 0].item()),
            100 + runtime.global_rank,
        )
        self.assertEqual(len(write_calls), 2)
        with self.assertRaises(StopIteration):
            next(batches)

        rank_entries = os.listdir(trainer.data_batch_rank_cache_dir)
        self.assertEqual(rank_entries, [".deepspec-transient-data-batch-cache"])
        os.unlink(
            os.path.join(
                trainer.data_batch_rank_cache_dir,
                ".deepspec-transient-data-batch-cache",
            )
        )
        os.rmdir(trainer.data_batch_rank_cache_dir)
        dist.barrier()
        if runtime.global_rank == 0:
            shutil.rmtree(cache_root[0])
        dist.barrier()


if __name__ == "__main__":
    unittest.main()

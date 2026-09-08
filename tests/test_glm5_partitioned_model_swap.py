from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
import json
import os
import tempfile
import unittest
import weakref
import sys
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from safetensors.torch import load_file, save_file
import torch

from deepspec.modeling.target import Glm5NextOnlineTarget
from deepspec.trainer.dspark_trainer import Glm5NextDSparkTrainer
from deepspec.trainer.glm5_partitioned_swap import (
    Glm5PartitionCache,
    Glm5TrainingPartition,
    build_journal_record,
    compute_glm5_training_partitions,
    validate_journal_record,
)
from deepspec.utils import StatelessResumableDistributedSampler, load_config, parse_opts_to_config
from deepspec.trainer.glm5_vllm import (
    VllmPartitionConfig,
    child_environment,
    convert_hidden_states,
    generate_job,
    live_process_group,
    load_hidden_states_with_retry,
    rank_group,
    run_worker_process,
    write_request,
)
from deepspec.trainer.glm5_partitioned_swap import atomic_write_json, load_json


def _cached_batch(sequence_length=3):
    def metadata(value):
        return torch.tensor([value], dtype=torch.long)

    return {
        "input_ids": torch.arange(sequence_length).unsqueeze(0),
        "loss_mask": torch.ones((1, sequence_length), dtype=torch.long),
        "target_hidden_states": torch.ones((1, sequence_length, 6)),
        "target_last_hidden_states": torch.ones((1, sequence_length, 2)),
        "context_start": metadata(0),
        "context_len": metadata(sequence_length),
        "seq_len": metadata(sequence_length),
    }


class _RangeDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class Glm5PartitionPlanTest(unittest.TestCase):
    def test_eight_partitions_balance_steps_and_cover_the_complete_epoch(self):
        partitions = compute_glm5_training_partitions(
            max_samples=None,
            data_batch_size=8,
            global_batch_size=128,
            gradient_accumulation_steps=2,
            micro_batches_per_epoch=636,
            max_train_steps=318,
        )
        self.assertEqual([item.optimizer_steps for item in partitions], [40] * 6 + [39] * 2)
        self.assertEqual(sum(item.global_sample_count for item in partitions), 318 * 128)
        cursor = 0
        for index, item in enumerate(partitions):
            self.assertEqual(item.partition_id, index)
            self.assertEqual(item.epoch, 0)
            self.assertEqual(item.start_next_micro_step, cursor)
            self.assertEqual(item.end_next_micro_step % 2, 0)
            cursor = item.end_next_micro_step
        self.assertEqual(cursor, 636)

    def test_count_based_partitions_repeat_the_dataset_boundaries_each_epoch(self):
        partitions = compute_glm5_training_partitions(
            max_samples=None,
            data_batch_size=2,
            global_batch_size=8,
            gradient_accumulation_steps=3,
            micro_batches_per_epoch=9,
            max_train_steps=8,
        )
        self.assertEqual([item.optimizer_steps for item in partitions], [2, 1, 2, 1, 2])
        self.assertEqual([item.epoch for item in partitions], [0, 0, 1, 1, 2])
        for item in partitions:
            self.assertEqual(item.start_next_micro_step // 9, (item.end_next_micro_step - 1) // 9)
        self.assertEqual(partitions[-1].end_next_micro_step, 24)

    def test_step_limit_truncates_execution_without_repartitioning_the_dataset(self):
        common = dict(
            max_samples=None,
            data_batch_size=8,
            global_batch_size=128,
            gradient_accumulation_steps=2,
            micro_batches_per_epoch=636,
        )
        full = compute_glm5_training_partitions(**common, max_train_steps=318)
        limited = compute_glm5_training_partitions(**common, max_train_steps=81)
        self.assertEqual(limited[:2], full[:2])
        self.assertEqual([item.optimizer_steps for item in limited], [40, 40, 1])
        self.assertEqual(limited[-1].start_next_micro_step, full[2].start_next_micro_step)
        self.assertEqual(limited[-1].end_next_micro_step, 162)

    def test_count_based_partitions_handle_short_and_empty_runs(self):
        for steps, count in ((0, 8), (3, 8), (3, "auto")):
            with self.subTest(steps=steps, count=count):
                partitions = compute_glm5_training_partitions(
                    max_samples=None,
                    data_batch_size=count,
                    global_batch_size=8,
                    gradient_accumulation_steps=1,
                    micro_batches_per_epoch=3,
                    max_train_steps=steps,
                )
                self.assertEqual([item.optimizer_steps for item in partitions], [1] * steps)

    def test_partition_count_and_sample_cap_are_mutually_exclusive(self):
        common = dict(
            global_batch_size=8,
            gradient_accumulation_steps=1,
            micro_batches_per_epoch=16,
            max_train_steps=16,
        )
        for cap, count in ((512, 8), (None, None)):
            with self.subTest(cap=cap, count=count):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    compute_glm5_training_partitions(max_samples=cap, data_batch_size=count, **common)
        with self.assertRaisesRegex(ValueError, "positive"):
            compute_glm5_training_partitions(max_samples=None, data_batch_size=0, **common)

    def test_partitions_are_optimizer_aligned_bounded_and_epoch_local(self):
        partitions = compute_glm5_training_partitions(
            max_samples=512,
            global_batch_size=96,
            gradient_accumulation_steps=3,
            micro_batches_per_epoch=21,
            max_train_steps=16,
        )
        self.assertEqual([item.optimizer_steps for item in partitions], [5, 2, 5, 2, 2])
        self.assertEqual([item.global_sample_count for item in partitions], [480, 192, 480, 192, 192])
        for item in partitions:
            self.assertLessEqual(item.global_sample_count, 512)
            self.assertEqual(item.start_next_micro_step % 3, 0)
            self.assertEqual(item.end_next_micro_step % 3, 0)
            self.assertEqual(
                item.start_next_micro_step // 21,
                (item.end_next_micro_step - 1) // 21,
            )

    def test_max_samples_smaller_than_global_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one global batch"):
            compute_glm5_training_partitions(
                max_samples=63,
                global_batch_size=64,
                gradient_accumulation_steps=1,
                micro_batches_per_epoch=4,
                max_train_steps=4,
            )

    def test_partitioned_sampler_stream_matches_unpartitioned_stream(self):
        dataset = _RangeDataset(200)
        partitions = compute_glm5_training_partitions(
            max_samples=None,
            data_batch_size=8,
            global_batch_size=8,
            gradient_accumulation_steps=4,
            micro_batches_per_epoch=100,
            max_train_steps=50,
        )
        samples_by_epoch = {0: [], 1: []}
        for rank in range(2):
            common = dict(dataset=dataset, num_replicas=2, rank=rank, total_size=200)
            complete = list(
                StatelessResumableDistributedSampler(**common, num_samples=200)
            )
            partitioned = []
            for partition in partitions:
                samples = list(
                    StatelessResumableDistributedSampler(
                        **common,
                        start_global_offset_samples=partition.start_next_micro_step,
                        num_samples=partition.end_next_micro_step - partition.start_next_micro_step,
                    )
                )
                partitioned.extend(samples)
                samples_by_epoch[partition.epoch].extend(samples)
            self.assertEqual(partitioned, complete)
        for epoch, samples in samples_by_epoch.items():
            self.assertEqual(sum(item.epoch == epoch for item in partitions), 8)
            self.assertEqual(sorted(samples), list(range(200)))

    def test_glm_config_keeps_new_mode_disabled_by_default(self):
        config = load_config("config/dspark/dspark_glm5_3_flash.py")
        self.assertFalse(config.train.partitioned_model_swap.enabled)
        self.assertEqual(config.train.partitioned_model_swap.max_samples, 512)
        self.assertEqual(config.train.data_batch_size, 8)


class Glm5PartitionCacheTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.cache = Glm5PartitionCache(root=self.tempdir.name, global_rank=3)
        self.partition = Glm5TrainingPartition(7, 2, 12, 14, 2, 16)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_sample_manifest_ready_and_checkpoint_gated_delete(self):
        incomplete = self.cache.prepare_incomplete(
            self.partition,
            replace_matching=False,
        )
        samples = []
        for offset in range(2):
            samples.append(
                self.cache.write_sample(
                    partition=self.partition,
                    batch=_cached_batch(3 + offset),
                    logical_sample_id=100 + offset,
                    dataset_index=20 + offset,
                    stream_micro_step=12 + offset,
                )
            )
        manifest = self.cache.write_local_manifest(
            partition=self.partition,
            samples=samples,
            target_shard_layout={"tp": 4, "ep": 8},
            state="LOCAL_COMPLETE",
        )
        self.assertEqual(manifest["writer_rank"], 3)
        self.assertEqual(manifest["local_sample_count"], 2)
        self.assertGreater(manifest["local_file_size"], 0)
        self.cache.validate_incomplete(self.partition)
        ready = self.cache.commit_ready(self.partition)
        self.assertFalse(os.path.exists(incomplete))
        self.assertTrue(ready.endswith("partition_000007.ready"))
        _, ready_manifest = self.cache.validate_ready(self.partition)
        self.assertEqual(ready_manifest["state"], "READY")
        self.assertEqual(
            ready_manifest["samples"][0]["tensors"]["input_ids"]["shape"],
            [1, 3],
        )
        self.cache.delete_ready(self.partition)
        self.assertFalse(os.path.exists(ready))

    def test_incomplete_failure_is_retained_and_matching_retry_is_scoped(self):
        incomplete = self.cache.prepare_incomplete(
            self.partition,
            replace_matching=False,
        )
        sentinel = os.path.join(incomplete, "failure.txt")
        with open(sentinel, "w", encoding="utf-8") as handle:
            handle.write("keep until an explicit GENERATING retry")
        self.assertTrue(os.path.exists(sentinel))
        with self.assertRaises(FileExistsError):
            self.cache.prepare_incomplete(
                self.partition,
                replace_matching=False,
            )
        replacement = self.cache.prepare_incomplete(
            self.partition,
            replace_matching=True,
        )
        self.assertEqual(replacement, incomplete)
        self.assertFalse(os.path.exists(sentinel))

    def test_ready_cache_is_never_overwritten_by_generation(self):
        self.cache.prepare_incomplete(self.partition, replace_matching=False)
        sample = self.cache.write_sample(
            partition=self.partition,
            batch=_cached_batch(),
            logical_sample_id=1,
            dataset_index=2,
            stream_micro_step=12,
        )
        self.cache.write_local_manifest(
            partition=self.partition,
            samples=[sample],
            target_shard_layout={},
            state="LOCAL_COMPLETE",
        )
        self.cache.commit_ready(self.partition)
        with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
            self.cache.prepare_incomplete(
                self.partition,
                replace_matching=True,
            )

    def test_manifest_tensor_validation_precedes_ready(self):
        self.cache.prepare_incomplete(self.partition, replace_matching=False)
        invalid = _cached_batch()
        invalid["context_len"] = torch.tensor([99])
        with self.assertRaisesRegex(ValueError, "context_len"):
            self.cache.write_sample(
                partition=self.partition,
                batch=invalid,
                logical_sample_id=1,
                dataset_index=2,
                stream_micro_step=12,
            )


class Glm5PartitionJournalTest(unittest.TestCase):
    def test_journal_identity_mismatch_is_not_guessed(self):
        partition = Glm5TrainingPartition(1, 0, 2, 4, 2, 16)
        identity = {"dataset": "a", "world_size": 8}
        record = build_journal_record(
            phase="READY",
            partition=partition,
            run_identity=identity,
        )
        validate_journal_record(
            json.loads(json.dumps(record)),
            partition=partition,
            run_identity=identity,
        )
        with self.assertRaisesRegex(ValueError, "run identity"):
            validate_journal_record(
                record,
                partition=partition,
                run_identity={"dataset": "b", "world_size": 8},
            )


class Glm5PartitionLifecycleTest(unittest.TestCase):
    def test_declared_lifecycle_has_the_required_order(self):
        self.assertEqual(
            Glm5NextDSparkTrainer.partitioned_model_swap_lifecycle,
            (
                "PREPARE_PARTITION",
                "TARGET_LOAD",
                "TARGET_GENERATE_FEATURES",
                "PARTITION_FEATURES_READY",
                "TARGET_UNLOAD",
                "DRAFT_LOAD",
                "DRAFT_TRAIN_PARTITION",
                "DRAFT_SAVE_CHECKPOINT",
                "DRAFT_UNLOAD",
                "PARTITION_CACHE_DELETE",
                "NEXT_PARTITION",
            ),
        )

    def _trainer(self):
        trainer = object.__new__(Glm5NextDSparkTrainer)
        partition = Glm5TrainingPartition(0, 0, 0, 1, 1, 8)
        trainer.partitioned_model_swap_enabled = True
        trainer.gradient_accumulation_steps = 1
        trainer.max_train_steps = 1
        trainer.next_micro_step = 0
        trainer._partitions_by_id = {0: partition}
        trainer._partitions_by_start = {0: partition}
        trainer._partition_run_identity = {"run": 1}
        trainer._data_batch_phase = None
        trainer.global_rank = 0
        trainer.world_size = 1
        trainer.events = []
        trainer._load_partition_journal = lambda: None

        def write_journal(phase, current, checkpoint_dir=None):
            trainer.events.append(phase)
            return build_journal_record(
                phase=phase,
                partition=current,
                run_identity=trainer._partition_run_identity,
                checkpoint_dir=checkpoint_dir,
            )

        trainer._write_partition_journal = write_journal
        trainer._generate_partition_features = lambda current, recovering: trainer.events.append(
            "GENERATE"
        )

        def train_partition(current, journal_is_training):
            trainer.events.append("TRAIN")
            trainer.next_micro_step = current.end_next_micro_step
            return "/checkpoint/step_1"

        trainer._train_ready_partition = train_partition

        def cleanup(current, checkpoint_dir):
            trainer.events.append("CLEANUP")
            trainer.next_micro_step = current.end_next_micro_step

        trainer._cleanup_checkpointed_partition = cleanup
        return trainer, partition

    @patch("deepspec.trainer.dspark_trainer.print_on_global_main")
    def test_complete_lifecycle_order(self, _print):
        trainer, _partition = self._trainer()
        trainer.train()
        self.assertEqual(
            trainer.events,
            ["GENERATING", "GENERATE", "TRAIN", "CLEANUP"],
        )

    @patch("deepspec.trainer.dspark_trainer.print_on_global_main")
    def test_generation_failure_never_starts_training(self, _print):
        trainer, _partition = self._trainer()

        def fail_generation(current, recovering):
            trainer.events.append("GENERATE")
            raise RuntimeError("target failed")

        trainer._generate_partition_features = fail_generation
        with self.assertRaisesRegex(RuntimeError, "target failed"):
            trainer.train()
        self.assertNotIn("TRAIN", trainer.events)
        self.assertNotIn("CLEANUP", trainer.events)

    @patch("deepspec.trainer.dspark_trainer.print_on_global_main")
    def test_training_failure_retains_ready_cache(self, _print):
        trainer, _partition = self._trainer()

        def fail_training(current, journal_is_training):
            trainer.events.append("TRAIN")
            raise RuntimeError("optimizer failed")

        trainer._train_ready_partition = fail_training
        with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
            trainer.train()
        self.assertNotIn("CLEANUP", trainer.events)

    @patch("deepspec.trainer.dspark_trainer.print_on_global_main")
    def test_nonzero_checkpoint_without_journal_is_rejected(self, _print):
        trainer, partition = self._trainer()
        trainer.next_micro_step = partition.end_next_micro_step
        with self.assertRaisesRegex(ValueError, "requires its matching"):
            trainer.train()
        self.assertNotIn("TRAIN", trainer.events)
        self.assertNotIn("CLEANUP", trainer.events)

    @patch("deepspec.trainer.dspark_trainer.print_on_global_main")
    def test_training_journal_with_completed_checkpoint_does_not_retrain(self, _print):
        trainer, partition = self._trainer()
        trainer._load_partition_journal = lambda: build_journal_record(
            phase="TRAINING",
            partition=partition,
            run_identity=trainer._partition_run_identity,
        )
        trainer._completed_partition_checkpoint = lambda current: "/checkpoint/step_1"
        trainer._validate_ready_partition = lambda current: None
        trainer.next_micro_step = partition.end_next_micro_step
        trainer.train()
        self.assertNotIn("TRAIN", trainer.events)
        self.assertEqual(trainer.events, ["CHECKPOINTED", "CLEANUP"])

    def test_target_and_draft_phase_guards_fail_before_forward(self):
        target = object.__new__(Glm5NextOnlineTarget)
        target.require_phase_guard = True
        target.execution_phase = "DRAFT_TRAIN_PARTITION"
        with self.assertRaisesRegex(RuntimeError, "TARGET_GENERATE_FEATURES"):
            target.forward_training_batch({})

        trainer, _partition = self._trainer()
        trainer._data_batch_phase = "TARGET_GENERATE_FEATURES"
        with self.assertRaisesRegex(RuntimeError, "DRAFT_TRAIN_PARTITION"):
            trainer.run_batch({})

    def test_active_model_guards_reject_overlap(self):
        trainer, _partition = self._trainer()
        trainer.draft_model = object()
        trainer.model = None
        trainer.optimizer = None
        with self.assertRaisesRegex(RuntimeError, "draft-absence guard"):
            trainer._assert_no_draft_state()
        trainer.draft_model = None
        trainer.online_target = object()
        with self.assertRaisesRegex(RuntimeError, "target-absence guard"):
            trainer._assert_no_target_state()

    @patch(
        "deepspec.modeling.target.online."
        "uninstall_glm5_next_bounded_target_prefill",
        new=lambda _model: None,
    )
    @patch("deepspec.modeling.target.online.dist.is_initialized", return_value=False)
    def test_target_teardown_releases_root_children_and_parameters(
        self,
        _is_initialized,
    ):
        target = object.__new__(Glm5NextOnlineTarget)
        target.device = torch.device("cpu")
        target.execution_phase = "TARGET_GENERATE_FEATURES"
        target.topology = object()
        target.feature_output_device = torch.device("cpu")

        root = torch.nn.Module()
        backbone = torch.nn.Module()
        backbone.layers = torch.nn.ModuleList()
        backbone.probe = torch.nn.Linear(2, 2)
        root.language_model = backbone
        root_reference = weakref.ref(root)
        backbone_reference = weakref.ref(backbone)
        parameter_reference = weakref.ref(next(root.parameters()))
        target.model = root
        del root
        del backbone

        target.close()
        self.assertIsNone(root_reference())
        self.assertIsNone(backbone_reference())
        self.assertIsNone(parameter_reference())
        self.assertIsNone(target._released_model_weakref())

    @patch("deepspec.trainer.dspark_trainer.dist.is_initialized", return_value=False)
    def test_partial_draft_teardown_does_not_require_optimizer_or_wrapper(
        self,
        _is_initialized,
    ):
        trainer, _partition = self._trainer()
        draft = torch.nn.Sequential(torch.nn.Linear(2, 2))
        draft_reference = weakref.ref(draft)
        parameter_reference = weakref.ref(next(draft.parameters()))
        trainer.device = torch.device("cpu")
        trainer.draft_model = draft
        trainer.model = None
        trainer.optimizer = None
        trainer._ready_cache_loader = None
        trainer._pure_expert_modules = ()
        trainer._set_swap_phase = lambda phase: trainer.events.append(phase)
        del draft

        trainer._unload_draft()
        self.assertIsNone(draft_reference())
        self.assertIsNone(parameter_reference())
        self.assertIsNone(trainer.draft_model)


class Glm5VllmPartitionTest(unittest.TestCase):
    @staticmethod
    def _load_with_nonblocking_lock(path):
        # Reproduce AFS's EAGAIN on contention even on local CI filesystems.
        with open(path + ".lock") as reader:
            fcntl.flock(reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return load_file(path)

    def test_feature_reader_waits_for_complete_async_write(self):
        expected = {
            "token_ids": torch.tensor([1, 2, 3]),
            "hidden_states": torch.arange(24, dtype=torch.bfloat16).reshape(3, 4, 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "features.safetensors")
            Path(path).write_bytes(b"incomplete asynchronous write")
            attempted = threading.Event()
            with open(path + ".lock", "w") as writer:
                fcntl.flock(writer, fcntl.LOCK_EX)

                def finish_write():
                    try:
                        self.assertTrue(attempted.wait(5))
                        save_file(expected, path)
                    finally:
                        fcntl.flock(writer, fcntl.LOCK_UN)

                def load(current):
                    try:
                        return self._load_with_nonblocking_lock(current)
                    except BlockingIOError:
                        attempted.set()
                        raise

                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(finish_write)
                    actual = load_hidden_states_with_retry(
                        path, loader=load, timeout_seconds=5,
                    )
                    future.result(timeout=5)
            self.assertTrue(attempted.is_set())
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_feature_reader_times_out_while_writer_keeps_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "features.safetensors")
            with open(path + ".lock", "w") as writer:
                fcntl.flock(writer, fcntl.LOCK_EX)
                with self.assertRaisesRegex(TimeoutError, "hidden-state writer"):
                    load_hidden_states_with_retry(
                        path,
                        loader=self._load_with_nonblocking_lock,
                        timeout_seconds=0.02,
                    )

    def test_feature_reader_propagates_storage_errors(self):
        error = OSError(errno.EIO, "storage read failed")
        loader = Mock(side_effect=error)
        with self.assertRaises(OSError) as raised:
            load_hidden_states_with_retry("features", loader=loader, timeout_seconds=5)
        self.assertIs(raised.exception, error)
        loader.assert_called_once_with("features")

    def test_launcher_worker_overrides_are_valid_config_keys(self):
        config = parse_opts_to_config(
            [
                "train.partitioned_model_swap.target_backend=vllm",
                "train.partitioned_model_swap.vllm.python_executable=/python",
                "train.partitioned_model_swap.vllm.source_dir=/vllm",
                "train.partitioned_model_swap.vllm.raw_cache_dir=/dev/shm",
            ],
            load_config("config/dspark/dspark_glm5_3_flash.py"),
        )
        worker = VllmPartitionConfig(**config.train.partitioned_model_swap.vllm)
        self.assertEqual(worker.python_executable, "/python")
        self.assertEqual(worker.raw_cache_dir, "/dev/shm")

    def _features(self, batch):
        length = batch["input_ids"].numel()
        hidden = torch.arange(length * 8, dtype=torch.bfloat16).reshape(length, 4, 2)
        return convert_hidden_states(
            {"token_ids": batch["input_ids"][0], "hidden_states": hidden},
            batch,
            norm_weight=torch.tensor([0.5, 1.5]),
            norm_eps=1e-6,
            num_layers=3,
        )

    def test_feature_conversion_preserves_masks_layers_and_final_norm(self):
        batch = {
            "input_ids": torch.tensor([[8, 3, 5]]),
            "loss_mask": torch.tensor([[0, 1, 0]]),
        }
        features = self._features(batch)
        raw = torch.arange(24, dtype=torch.bfloat16).reshape(3, 4, 2)
        torch.testing.assert_close(
            features["target_hidden_states"][0], raw[:, :3].flatten(1)
        )
        expected = torch.nn.functional.rms_norm(
            raw[:, -1].float(), [2], torch.tensor([0.5, 1.5]), 1e-6
        ).bfloat16()
        torch.testing.assert_close(
            features["target_last_hidden_states"][0], expected, rtol=0, atol=0
        )
        self.assertTrue(torch.equal(features["loss_mask"], batch["loss_mask"]))
        self.assertEqual(features["context_start"].item(), 0)
        self.assertEqual(features["seq_len"].item(), 3)

    def test_wrong_tokens_and_nonfinite_features_are_rejected(self):
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "loss_mask": torch.ones(1, 2, dtype=torch.long),
        }
        hidden = torch.ones(2, 4, 2, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "token IDs"):
            convert_hidden_states(
                {"token_ids": torch.tensor([2, 1]), "hidden_states": hidden},
                batch,
                norm_weight=torch.ones(2),
                norm_eps=1e-6,
                num_layers=3,
            )
        hidden[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            convert_hidden_states(
                {"token_ids": batch["input_ids"][0], "hidden_states": hidden},
                batch,
                norm_weight=torch.ones(2),
                norm_eps=1e-6,
                num_layers=3,
            )

    def test_node_local_tp_replicas_keep_physical_gpu_order(self):
        ranks, devices = rank_group(
            global_rank=14,
            local_rank=6,
            local_world_size=8,
            tp_size=4,
            devices=[f"GPU-{i}" for i in (7, 4, 6, 2, 3, 0, 5, 1)],
        )
        self.assertEqual(ranks, [12, 13, 14, 15])
        self.assertEqual(devices, ["GPU-3", "GPU-0", "GPU-5", "GPU-1"])
        with patch.dict(
            os.environ,
            {
                "RANK": "14",
                "MASTER_PORT": "29500",
                "TORCHELASTIC_ERROR_FILE": "/training/error",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "VLLM_WORKER_MULTIPROC_METHOD": "fork",
            },
        ):
            env = child_environment(devices)
        self.assertNotIn("RANK", env)
        self.assertNotIn("MASTER_PORT", env)
        self.assertNotIn("TORCHELASTIC_ERROR_FILE", env)
        self.assertNotIn("PYTORCH_ALLOC_CONF", env)
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)
        self.assertEqual(env["VLLM_WORKER_MULTIPROC_METHOD"], "fork")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], ",".join(devices))

    @patch("deepspec.trainer.dspark_trainer.print_on_global_main")
    @patch("deepspec.trainer.dspark_trainer.dist.is_initialized", return_value=False)
    def test_generating_recovery_keeps_partial_ready_and_original_sample_owners(
        self, *_
    ):
        # Exercise both a completed leader and a completed follower. Identical
        # prompts have different masks and dataset positions and must stay distinct.
        for ready_rank in (0, 1):
            with (
                self.subTest(ready_rank=ready_rank),
                tempfile.TemporaryDirectory() as root,
            ):
                partition = Glm5TrainingPartition(0, 0, 0, 1, 1, 2)
                trainer = object.__new__(Glm5NextDSparkTrainer)
                trainer.global_rank = trainer.data_parallel_rank = 0
                trainer.world_size = trainer.data_parallel_size = 2
                trainer.samples_per_epoch = 2
                trainer.target_backend = "vllm"
                trainer.draft_model = trainer.model = trainer.optimizer = (
                    trainer.online_target
                ) = None
                trainer._partition_control_group = None
                trainer._vllm_owner_ranks = [0, 1]
                trainer._vllm_devices = ["0", "1"]
                trainer._vllm_teacher_identity = {"backend": "test"}
                trainer._vllm_config = VllmPartitionConfig(tensor_parallel_size=2)
                trainer.data_batch_cache_root = root
                trainer._partition_cache = Glm5PartitionCache(root=root, global_rank=0)
                trainer.train_dataset = _RangeDataset(2)
                trainer.train_dataset.close = lambda: None
                trainer.args = SimpleNamespace(
                    train=SimpleNamespace(local_batch_size=1),
                    model=SimpleNamespace(target_model_name_or_path="unused"),
                    data=SimpleNamespace(max_length=3),
                )
                events = []
                trainer._write_partition_journal = lambda phase, p: events.append(phase)
                batches = []
                indices = []
                for rank in range(2):
                    (index,) = StatelessResumableDistributedSampler(
                        dataset=trainer.train_dataset,
                        num_replicas=2,
                        rank=rank,
                        total_size=2,
                        num_samples=1,
                    )
                    indices.append(index)
                    batch = {
                        "input_ids": torch.tensor([[8, 3, 5]]),
                        "loss_mask": torch.tensor([[rank, 1, 0]]),
                    }
                    batches.append(batch)
                    cache = Glm5PartitionCache(root=root, global_rank=rank)
                    directory = cache.prepare_incomplete(
                        partition, replace_matching=False
                    )
                    request = write_request(
                        directory,
                        batch,
                        logical_sample_id=rank,
                        dataset_index=index,
                        stream_micro_step=0,
                    )
                    layout = {**trainer._target_shard_layout(), "global_rank": rank}
                    atomic_write_json(
                        os.path.join(directory, "vllm_requests.json"),
                        {
                            "teacher": trainer._vllm_teacher_identity,
                            "requests": [request],
                            "target_shard_layout": layout,
                        },
                    )
                    if rank == ready_rank:
                        sample = cache.write_sample(
                            partition=partition,
                            batch=self._features(batch),
                            logical_sample_id=rank,
                            dataset_index=index,
                            stream_micro_step=0,
                        )
                        cache.write_local_manifest(
                            partition=partition,
                            samples=[sample],
                            target_shard_layout=layout,
                            state="LOCAL_COMPLETE",
                        )
                        ready_dir = cache.commit_ready(partition)
                        saved_path = Path(ready_dir) / sample["file"]
                        saved_bytes = saved_path.read_bytes()

                class Loader(list):
                    sampler = [indices[0]]

                def gather(output, value, **kwargs):
                    output[:] = [ready_rank == rank for rank in range(2)]

                def worker(*, job_path, config, devices):
                    job = load_json(job_path)
                    self.assertEqual(job["owner_ranks"], [1 - ready_rank])
                    generate_job(job, extract=self._features)
                    atomic_write_json(
                        job_path + ".complete",
                        {"partition": job["partition"], "teacher": job["teacher"]},
                    )

                with (
                    patch(
                        "deepspec.trainer.dspark_trainer.BaseTrainer._build_train_dataloader",
                        return_value=Loader([batches[0]]),
                    ),
                    patch(
                        "deepspec.trainer.dspark_trainer.dist.all_gather_object",
                        side_effect=gather,
                    ),
                    patch(
                        "deepspec.trainer.glm5_vllm.run_worker_process",
                        side_effect=worker,
                    ),
                ):
                    trainer._generate_vllm_partition_features(
                        partition, recovering=True
                    )
                self.assertEqual(events, ["READY"])
                self.assertEqual(saved_path.read_bytes(), saved_bytes)
                if ready_rank == 0:
                    # The real follower performs its own collective commit.
                    follower = Glm5PartitionCache(root=root, global_rank=1)
                    follower.validate_incomplete(partition)
                    follower.commit_ready(partition)
                for rank in range(2):
                    directory, manifest = Glm5PartitionCache(
                        root=root, global_rank=rank
                    ).validate_ready(partition)
                    (sample,) = manifest["samples"]
                    self.assertEqual(sample["logical_sample_id"], rank)
                    self.assertEqual(sample["dataset_index"], indices[rank])
                    cached = torch.load(
                        os.path.join(directory, sample["file"]), weights_only=True
                    )
                    self.assertTrue(
                        torch.equal(cached["loss_mask"], batches[rank]["loss_mask"])
                    )

    def test_worker_exit_releases_orphan_processes_on_success_and_failure(self):
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as root:
                executable = Path(root) / "python"
                job = Path(root) / "job.json"
                executable.write_text(
                    f"#!{sys.executable}\n"
                    "import os, subprocess, sys, time\n"
                    "from pathlib import Path\n"
                    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                    "Path(sys.argv[-1]).write_text(str(os.getpgrp()))\n"
                    f"sys.exit({code})\n"
                )
                executable.chmod(0o755)
                config = VllmPartitionConfig(python_executable=str(executable))
                if code:
                    with self.assertRaisesRegex(RuntimeError, "status 7"):
                        run_worker_process(
                            job_path=str(job), config=config, devices=["0"]
                        )
                else:
                    run_worker_process(job_path=str(job), config=config, devices=["0"])
                self.assertFalse(live_process_group(int(job.read_text())))

    def test_spawn_entrypoint_does_not_import_training_or_initialize_cuda(self):
        entrypoint = (
            Path(__file__).resolve().parents[1]
            / "scripts/data/generate_glm5_vllm_partition.py"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__mp_main__'); "
                "assert not any(name in sys.modules for name in ('torch', 'vllm', 'deepspec.trainer'))",
                str(entrypoint),
            ],
            check=True,
            timeout=30,
        )

    def test_partition_checkpoint_resume_matches_continuous_optimizer_and_rng(self):
        from deepspec.distributed.distributed_checkpoint import (
            TrainingProgress,
            load_training_checkpoint,
            save_training_checkpoint,
        )
        from deepspec.training import BF16Optimizer

        def build():
            model = torch.nn.Sequential(torch.nn.Linear(6, 2), torch.nn.Dropout(0.3))
            return model, BF16Optimizer(
                model, lr=1e-3, total_steps=4, warmup_ratio=0, weight_decay=0
            )

        features = self._features(
            {
                "input_ids": torch.tensor([[8, 3, 5]]),
                "loss_mask": torch.tensor([[0, 1, 1]]),
            }
        )

        def step(model, optimizer):
            loss = (
                (
                    model(features["target_hidden_states"].float())
                    - features["target_last_hidden_states"].float()
                )
                .square()
                .mean()
            )
            loss.backward()
            optimizer.step()
            return loss.detach()

        torch.manual_seed(91)
        model, optimizer = build()
        step(model, optimizer)
        progress = TrainingProgress(
            next_micro_step=1,
            global_step=1,
            epoch=0,
            data_position=1,
            local_batch_size=1,
            saved_world_size=1,
            parallel_config={},
            model_config={},
        )
        with tempfile.TemporaryDirectory() as root:
            save_training_checkpoint(
                checkpoint_dir=root,
                model=model,
                optimizer_bundle=optimizer,
                progress=progress,
            )
            expected_loss = step(model, optimizer)
            resumed_model, resumed_optimizer = build()
            load_training_checkpoint(
                checkpoint_dir=root,
                model=resumed_model,
                optimizer_bundle=resumed_optimizer,
                progress=progress,
            )
            actual_loss = step(resumed_model, resumed_optimizer)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            self.assertEqual(
                optimizer.get_learning_rate(), resumed_optimizer.get_learning_rate()
            )
            for actual, expected in zip(resumed_model.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

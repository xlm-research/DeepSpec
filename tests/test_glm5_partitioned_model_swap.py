from __future__ import annotations

import json
import os
import tempfile
import unittest
import weakref
from unittest.mock import patch

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
from deepspec.utils import StatelessResumableDistributedSampler, load_config


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
        dataset = _RangeDataset(29)
        common = dict(
            dataset=dataset,
            num_replicas=2,
            rank=1,
            total_size=24,
        )
        complete = list(
            StatelessResumableDistributedSampler(
                **common,
                start_global_offset_samples=0,
                num_samples=24,
            )
        )
        partitioned = []
        for start, count in ((0, 5), (5, 7), (12, 4), (16, 8)):
            partitioned.extend(
                StatelessResumableDistributedSampler(
                    **common,
                    start_global_offset_samples=start,
                    num_samples=count,
                )
            )
        self.assertEqual(partitioned, complete)

    def test_glm_config_keeps_new_mode_disabled_by_default(self):
        config = load_config("config/dspark/dspark_glm5_3_flash.py")
        self.assertFalse(config.train.partitioned_model_swap.enabled)
        self.assertEqual(config.train.partitioned_model_swap.max_samples, 512)
        self.assertEqual(config.train.data_batch_size, 256)


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


if __name__ == "__main__":
    unittest.main()

import copy
import gc
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from transformers import AutoConfig

from deepspec.modeling.dflash2.deepseek_v4 import (
    DeepseekV4DFlash2Model,
    build_draft_config as build_dflash2_config,
)
from deepspec.modeling.dspark.deepseek_v4 import (
    DeepseekV4DSparkModel,
    build_draft_config as build_dspark_config,
)
from deepspec.trainer.base_trainer import (
    _compute_data_batch_schedule,
    _release_target_features,
    _resolve_data_batch_partition_count,
)
from deepspec.trainer.dflash2_trainer import DeepseekV4DFlash2Trainer
from deepspec.trainer.dspark_trainer import DeepseekV4DSparkTrainer
from deepspec.utils import load_config


TARGET = "/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731"


def tiny_target_config():
    config = AutoConfig.from_pretrained(TARGET)
    config.hidden_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 1
    config.head_dim = 16
    config.q_lora_rank = 32
    config.o_groups = 4
    config.o_lora_rank = 16
    config.hc_mult = 2
    config.moe_intermediate_size = 32
    config.n_routed_experts = 8
    config.n_shared_experts = 1
    config.num_experts_per_tok = 2
    config.vocab_size = 128
    config.expert_dtype = "bf16"
    config.num_hidden_layers = 3
    return config


def model_args(path):
    args = load_config(path).model
    args.mask_token_id = 127
    args.num_anchors = 2
    args.target_layer_ids = [0, 1, 2]
    return args


class DeepseekV4DraftModelTest(unittest.TestCase):
    def test_data_batch_size_is_capped_by_remaining_optimizer_steps(self):
        self.assertEqual(
            _resolve_data_batch_partition_count(
                "auto",
                remaining_optimizer_steps=125,
            ),
            125,
        )
        self.assertEqual(
            _resolve_data_batch_partition_count(
                "AUTO",
                remaining_optimizer_steps=7,
            ),
            7,
        )
        self.assertEqual(
            _resolve_data_batch_partition_count(
                3,
                remaining_optimizer_steps=7,
            ),
            3,
        )
        self.assertEqual(
            _resolve_data_batch_partition_count(
                256,
                remaining_optimizer_steps=1,
            ),
            1,
        )
        self.assertEqual(
            _resolve_data_batch_partition_count(
                256,
                remaining_optimizer_steps=0,
            ),
            0,
        )
        with self.assertRaisesRegex(ValueError, "positive or 'auto'"):
            _resolve_data_batch_partition_count(
                0,
                remaining_optimizer_steps=7,
            )

    def test_data_batch_schedule_splits_total_samples_by_ratio(self):
        micro_batches, optimizer_steps = _compute_data_batch_schedule(
            data_batch_size=3,
            total_samples=15000,
            global_batch_size=8,
            data_parallel_size=8,
            local_batch_size=1,
        )
        self.assertEqual(micro_batches, (625, 625, 625))
        self.assertEqual(optimizer_steps, (625, 625, 625))
        self.assertEqual(tuple(count * 8 for count in micro_batches), (5000,) * 3)
        self.assertEqual(
            _compute_data_batch_schedule(
                data_batch_size=3,
                total_samples=64,
                global_batch_size=8,
                data_parallel_size=1,
                local_batch_size=1,
            ),
            ((24, 24, 16), (3, 3, 2)),
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _compute_data_batch_schedule(
                data_batch_size=3,
                total_samples=16,
                global_batch_size=8,
                data_parallel_size=1,
                local_batch_size=1,
            )

    def _exercise(self, model):
        seq = 32
        output = model(
            input_ids=torch.randint(0, 127, (1, seq)),
            target_hidden_states=torch.randn(1, seq, 192),
            loss_mask=torch.ones(1, seq, dtype=torch.bool),
            target_last_hidden_states=None,
            context_start=torch.tensor([0]),
            context_len=torch.tensor([seq]),
            seq_len=torch.tensor([seq]),
        )
        self.assertEqual(tuple(output.draft_logits.shape), (1, 2, 7, 128))
        loss = output.draft_logits.float().square().mean()
        if output.selector_scores is not None:
            loss = loss + output.selector_scores.float().square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        return output

    def test_dspark_and_dflash_backbone(self):
        config = build_dspark_config(
            copy.deepcopy(tiny_target_config()),
            model_args("config/dspark/dspark_deepseek_v4.py"),
        )
        self.assertEqual(config._experts_implementation, "grouped_mm")
        output = self._exercise(DeepseekV4DSparkModel(config).float())
        self.assertIsNone(output.selector_scores)

    def test_dflash2_dynamic_conv_and_selector(self):
        config = build_dflash2_config(
            copy.deepcopy(tiny_target_config()),
            model_args("config/dflash2/dflash2_deepseek_v4.py"),
        )
        model = DeepseekV4DFlash2Model(config).float()
        output = self._exercise(model)
        self.assertIsNotNone(output.selector_scores)
        self.assertTrue(
            any(layer.attention_conv is not None for layer in model.layers)
        )
        self.assertEqual(
            torch.count_nonzero(
                model.candidate_selector.successor_codebook
            ).item(),
            0,
        )

    def test_dflash2_config_and_checkpoint_are_isolated_from_dspark(self):
        target = tiny_target_config()
        dflash_config = build_dflash2_config(
            copy.deepcopy(target),
            model_args("config/dflash2/dflash2_deepseek_v4.py"),
        )
        dspark_config = build_dspark_config(
            copy.deepcopy(target),
            model_args("config/dspark/dspark_deepseek_v4.py"),
        )
        self.assertEqual(
            dflash_config.architectures,
            ["DeepseekV4DFlash2Model"],
        )
        self.assertEqual(dflash_config.verification_block_size, 8)
        self.assertEqual(dflash_config.proposal_hidden_offset, 1)
        self.assertEqual(dflash_config.block_size, 7)
        self.assertFalse(dflash_config.is_causal)
        self.assertFalse(dflash_config.sample_from_anchor)
        self.assertEqual(dflash_config.dflash_config["block_size"], 8)
        self.assertEqual(dflash_config.dflash_config["selector_top_k"], 16)
        self.assertEqual(
            dspark_config.architectures,
            ["DeepseekV4DSparkModel"],
        )
        self.assertEqual(dspark_config._experts_implementation, "grouped_mm")
        self.assertEqual(dflash_config._experts_implementation, "grouped_mm")
        self.assertFalse(hasattr(dspark_config, "dflash_config"))

        model = DeepseekV4DFlash2Model(dflash_config).float()
        filtered = model.filter_checkpoint_state_dict(
            {
                "embed_tokens.weight": torch.empty(1),
                "lm_head.weight": torch.empty(1),
                "candidate_selector.successor_codebook": torch.empty(1),
            }
        )
        self.assertEqual(
            set(filtered),
            {"candidate_selector.successor_codebook"},
        )

    def test_dflash2_rejects_non_anchor_block_layout(self):
        args = model_args("config/dflash2/dflash2_deepseek_v4.py")
        args.block_size = 6
        with self.assertRaisesRegex(ValueError, "verification_block_size - 1"):
            build_dflash2_config(copy.deepcopy(tiny_target_config()), args)

    def test_online_target_features_are_owned_and_released_by_outer_batch(self):
        class FakeOnlineTarget:
            def forward_training_batch(self, _batch):
                return {
                    "input_ids": torch.ones(1, 4, dtype=torch.long),
                    "loss_mask": torch.ones(1, 4, dtype=torch.bool),
                    "target_hidden_states": torch.randn(1, 4, 6),
                    "target_last_hidden_states": torch.randn(1, 4, 2),
                    "context_start": torch.tensor([0]),
                    "context_len": torch.tensor([4]),
                    "seq_len": torch.tensor([4]),
                }

        trainer = object.__new__(DeepseekV4DSparkTrainer)
        trainer.online_target_enabled = True
        trainer.online_target = FakeOnlineTarget()
        trainer.data_batch_micro_batches = None
        trainer._data_batch_phase = None
        trainer.args = SimpleNamespace(
            model=SimpleNamespace(
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
            )
        )
        trainer.forward_model = lambda **_kwargs: object()
        batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.bool),
            "loss_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        with patch(
            "deepspec.trainer.dspark_trainer.compute_dspark_loss",
            return_value=torch.ones((), requires_grad=True),
        ):
            loss = trainer.run_batch(batch)

        self.assertIn("target_hidden_states", batch)
        self.assertIn("target_last_hidden_states", batch)
        self.assertNotIn("attention_mask", batch)
        hidden_ref = weakref.ref(batch["target_hidden_states"])
        last_hidden_ref = weakref.ref(batch["target_last_hidden_states"])
        loss.backward()
        del loss
        _release_target_features(batch)
        gc.collect()
        self.assertIsNone(hidden_ref())
        self.assertIsNone(last_hidden_ref())

    def test_online_target_disk_cache_is_deleted_after_each_data_batch(self):
        prepared_sample_ids = []
        testcase = self

        class FakeOnlineTarget:
            def forward_training_batch(self, batch):
                testcase.assertEqual(
                    trainer._data_batch_phase,
                    "target_inference",
                )
                sample_id = int(batch["input_ids"].item())
                prepared_sample_ids.append(sample_id)
                return {
                    "input_ids": batch["input_ids"],
                    "loss_mask": batch["loss_mask"],
                    "target_hidden_states": torch.full((1, 1, 2), sample_id),
                    "target_last_hidden_states": torch.full((1, 1, 2), sample_id),
                    "context_start": torch.tensor([0]),
                    "context_len": torch.tensor([1]),
                    "seq_len": torch.tensor([1]),
                }

        trainer = object.__new__(DeepseekV4DSparkTrainer)
        trainer.online_target_enabled = True
        trainer.online_target = FakeOnlineTarget()
        trainer.data_batch_size = 2
        trainer.data_batch_micro_batches = (4, 2)
        trainer.data_parallel_size = 1
        trainer.global_rank = 0
        trainer.device = torch.device("cpu")
        raw_batches = [
            {
                "input_ids": torch.tensor([[sample_id]]),
                "attention_mask": torch.ones(1, 1, dtype=torch.bool),
                "loss_mask": torch.ones(1, 1, dtype=torch.bool),
            }
            for sample_id in range(6)
        ]

        with tempfile.TemporaryDirectory() as cache_root:
            trainer.data_batch_cache_root = cache_root
            trainer.data_batch_rank_cache_dir = None
            trainer._active_data_batch_cache = None
            trainer._initialize_data_batch_cache()

            training_batches = trainer.iter_training_batches(iter(raw_batches))
            first = next(training_batches)
            first_block_paths = list(trainer._active_data_batch_cache)
            self.assertEqual(trainer._data_batch_phase, "draft_training")
            self.assertEqual(prepared_sample_ids, [0, 1, 2, 3])
            self.assertEqual(len(first_block_paths), 4)
            self.assertTrue(all(os.path.isfile(path) for path in first_block_paths))
            self.assertEqual(int(first["target_hidden_states"][0, 0, 0]), 0)
            for expected_id in (1, 2, 3):
                batch = next(training_batches)
                self.assertEqual(
                    int(batch["target_hidden_states"][0, 0, 0]), expected_id
                )

            fifth = next(training_batches)
            self.assertTrue(all(not os.path.exists(path) for path in first_block_paths))
            self.assertEqual(prepared_sample_ids, [0, 1, 2, 3, 4, 5])
            self.assertEqual(int(fifth["target_hidden_states"][0, 0, 0]), 4)
            self.assertEqual(
                int(next(training_batches)["target_hidden_states"][0, 0, 0]), 5
            )
            second_block_paths = list(trainer._active_data_batch_cache)
            with self.assertRaises(StopIteration):
                next(training_batches)
            self.assertTrue(
                all(not os.path.exists(path) for path in second_block_paths)
            )
            self.assertIsNone(trainer._active_data_batch_cache)
            self.assertIsNone(trainer._data_batch_phase)

    def test_target_inference_progress_synchronizes_each_optimizer_window(self):
        trainer = object.__new__(DeepseekV4DSparkTrainer)
        trainer.gradient_accumulation_steps = 2
        trainer.data_batch_micro_batches = (5,)
        trainer.device = torch.device("cpu")

        with (
            patch(
                "deepspec.trainer.base_trainer.dist.is_initialized",
                return_value=True,
            ),
            patch("deepspec.trainer.base_trainer.dist.barrier") as barrier,
            patch("deepspec.trainer.base_trainer.print_on_global_main") as log,
        ):
            for processed_samples in range(1, 6):
                trainer._synchronize_target_inference_progress(
                    data_batch_index=1,
                    processed_samples=processed_samples,
                    total_samples=5,
                )

        self.assertEqual(barrier.call_count, 2)
        self.assertEqual(log.call_count, 2)
        self.assertIn("2/5 local samples", log.call_args_list[0].args[0])
        self.assertIn("4/5 local samples", log.call_args_list[1].args[0])

    def test_isolated_draft_phase_refuses_inline_target_inference(self):
        trainer = object.__new__(DeepseekV4DSparkTrainer)
        trainer.online_target_enabled = True
        trainer.data_batch_micro_batches = (1,)
        trainer._data_batch_phase = "draft_training"
        with self.assertRaisesRegex(RuntimeError, "precomputed target hidden"):
            trainer.run_batch({"input_ids": torch.ones(1, 1, dtype=torch.long)})

        trainer._data_batch_phase = "draft_training"
        with self.assertRaisesRegex(RuntimeError, "only allowed"):
            trainer.prepare_online_target_batch({})

    def test_dflash2_honors_isolated_target_and_draft_phases(self):
        class FakeOnlineTarget:
            def forward_training_batch(self, _batch):
                raise AssertionError("target forward must not run in draft phase")

        trainer = object.__new__(DeepseekV4DFlash2Trainer)
        trainer.online_target_enabled = True
        trainer.online_target = FakeOnlineTarget()
        trainer.data_batch_micro_batches = (1,)
        trainer._data_batch_phase = "draft_training"
        trainer.args = SimpleNamespace(
            model=load_config("config/dflash2/dflash2_deepseek_v4.py").model
        )
        trainer.forward_model = lambda **_kwargs: object()

        with self.assertRaisesRegex(RuntimeError, "precomputed target hidden"):
            trainer.run_batch({"input_ids": torch.ones(1, 1, dtype=torch.long)})

        batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4, dtype=torch.bool),
            "target_hidden_states": torch.randn(1, 4, 6),
            "target_last_hidden_states": torch.randn(1, 4, 2),
            "context_start": torch.tensor([0]),
            "context_len": torch.tensor([4]),
            "seq_len": torch.tensor([4]),
        }
        with patch(
            "deepspec.trainer.dflash2_trainer.compute_dflash2_loss",
            return_value=torch.ones((), requires_grad=True),
        ):
            trainer.run_batch(batch)
        self.assertNotIn("target_last_hidden_states", batch)

    def test_interrupted_data_batch_keeps_its_disk_cache(self):
        class FakeOnlineTarget:
            def forward_training_batch(self, batch):
                return {
                    "input_ids": batch["input_ids"],
                    "loss_mask": batch["loss_mask"],
                    "target_hidden_states": torch.ones(1, 1, 2),
                    "target_last_hidden_states": torch.ones(1, 1, 2),
                    "context_start": torch.tensor([0]),
                    "context_len": torch.tensor([1]),
                    "seq_len": torch.tensor([1]),
                }

        trainer = object.__new__(DeepseekV4DSparkTrainer)
        trainer.online_target_enabled = True
        trainer.online_target = FakeOnlineTarget()
        trainer.data_batch_size = 1
        trainer.data_batch_micro_batches = (2,)
        trainer.data_parallel_size = 1
        trainer.global_rank = 0
        trainer.device = torch.device("cpu")
        raw_batches = [
            {
                "input_ids": torch.tensor([[sample_id]]),
                "attention_mask": torch.ones(1, 1, dtype=torch.bool),
                "loss_mask": torch.ones(1, 1, dtype=torch.bool),
            }
            for sample_id in range(2)
        ]

        with tempfile.TemporaryDirectory() as cache_root:
            trainer.data_batch_cache_root = cache_root
            trainer.data_batch_rank_cache_dir = None
            trainer._active_data_batch_cache = None
            trainer._initialize_data_batch_cache()
            training_batches = trainer.iter_training_batches(iter(raw_batches))
            next(training_batches)
            cached_paths = list(trainer._active_data_batch_cache)
            training_batches.close()
            self.assertTrue(all(os.path.isfile(path) for path in cached_paths))
            trainer._initialize_data_batch_cache()
            self.assertTrue(all(not os.path.exists(path) for path in cached_paths))

    def test_data_batch_cache_refuses_to_delete_unowned_directory(self):
        trainer = object.__new__(DeepseekV4DSparkTrainer)
        trainer.global_rank = 0
        with tempfile.TemporaryDirectory() as cache_root:
            trainer.data_batch_cache_root = cache_root
            rank_dir = os.path.join(cache_root, "rank_00000")
            os.makedirs(rank_dir)
            sentinel = os.path.join(rank_dir, "user-file")
            with open(sentinel, "w", encoding="utf-8") as handle:
                handle.write("keep")
            with self.assertRaisesRegex(ValueError, "unowned"):
                trainer._initialize_data_batch_cache()
            self.assertTrue(os.path.isfile(sentinel))


if __name__ == "__main__":
    unittest.main()

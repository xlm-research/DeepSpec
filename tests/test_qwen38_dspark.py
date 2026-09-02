import copy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from deepspec.distributed import (
    ParallelConfig,
    ParallelContext,
    apply_parallelism,
)
from deepspec.eval.dspark import Qwen3_8DSparkEvaluator
from deepspec.modeling.dspark.qwen3_8 import (
    Qwen3_8DSparkModel,
    build_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.target import online
from deepspec.modeling.target import qwen3_6_cp
from deepspec.modeling.target.common import TargetForwardResult
from deepspec.trainer import Qwen3_8DSparkTrainer
from deepspec.utils.config import ConfigNode, load_config
from tests.distributed_test_utils import require_torchrun


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGURED_TARGET = "/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B"
REAL_TARGET = os.environ.get("DEEPSPEC_QWEN38_MODEL_PATH", CONFIGURED_TARGET)
HAS_REAL_TARGET = Path(REAL_TARGET).is_dir()


def _model_args(**overrides):
    values = {
        "block_size": 3,
        "num_draft_layers": 1,
        "target_layer_ids": [1, 3],
        "mask_token_id": 127,
        "num_anchors": 2,
        "markov_rank": 0,
        "confidence_head_alpha": 0.0,
    }
    values.update(overrides)
    return ConfigNode(values)


def _tiny_target_config():
    text_config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 4,
    )
    return ConfigNode({
        "model_type": "qwen3_5",
        "text_config": text_config,
    })


class Qwen3_8DSparkConfigTest(unittest.TestCase):
    @unittest.skipUnless(
        HAS_REAL_TARGET,
        "Set DEEPSPEC_QWEN38_MODEL_PATH to run the real-config test.",
    )
    def test_real_target_builds_dedicated_dspark_config(self):
        target_config = AutoConfig.from_pretrained(REAL_TARGET)
        args = load_config(
            REPO_ROOT / "config/dspark/dspark_qwen3_8_27b.py"
        ).model
        config = build_draft_config(target_config, args)

        self.assertEqual(config.architectures, ["Qwen3_8DSparkModel"])
        self.assertEqual(config.deepspec_target_family, "qwen3_8")
        self.assertEqual(config.model_type, "qwen3_5_text")
        self.assertEqual(config.hidden_size, 5120)
        self.assertEqual(config.num_hidden_layers, 5)
        self.assertEqual(config.target_layer_ids, [1, 16, 31, 46, 61])
        self.assertEqual(config._attn_implementation, "flex_attention")
        self.assertEqual(config.target_context_layout, "native_head_tail")
        self.assertEqual(config.deepspec_draft_architecture, "qwen3_full_attention")
        self.assertEqual(config.deepspec_draft_rope, "full_head")
        self.assertEqual(config.partial_rotary_factor, 1.0)
        self.assertEqual(
            config.rope_parameters,
            {"rope_type": "default", "rope_theta": 10_000_000.0},
        )

    def test_training_config_selects_qwen38_dspark_trainer(self):
        config = load_config(
            REPO_ROOT / "config/dspark/dspark_qwen3_8_27b.py"
        )
        self.assertIs(config.train.trainer_cls, Qwen3_8DSparkTrainer)
        parallel = ParallelConfig.from_mapping(config.train, world_size=8)
        self.assertEqual(
            (parallel.dp_replicate, parallel.dp_shard, parallel.cp, parallel.tp),
            (1, 1, 2, 4),
        )
        self.assertEqual(config.data.max_length, 131072)
        self.assertEqual(config.model.target_model_name_or_path, CONFIGURED_TARGET)
        self.assertFalse(config.data.multimodal)
        self.assertTrue(config.data.store_target_last_hidden_states)
        self.assertFalse(config.data.online_target)
        self.assertTrue(config.data.offline_target_data_batches)
        self.assertEqual(config.train.data_partitions, 512)
        self.assertIsNone(config.data.target_cache_path)
        target_parallel = ParallelConfig.from_mapping(
            {"parallel": dict(config.train.offline_target_parallel)},
            world_size=8,
        )
        self.assertEqual(
            (
                target_parallel.dp_replicate,
                target_parallel.dp_shard,
                target_parallel.cp,
                target_parallel.tp,
            ),
            (1, 1, 2, 4),
        )
        self.assertEqual(config.train.num_train_epochs, 10)

    def test_checkpoint_architecture_selects_qwen38_evaluator(self):
        from eval import EVALUATORS

        self.assertIs(
            EVALUATORS["Qwen3_8DSparkModel"],
            Qwen3_8DSparkEvaluator,
        )

    def test_rejects_a_different_target_shape(self):
        target_config = _tiny_target_config()
        with self.assertRaisesRegex(ValueError, "Qwen3.8-27B DSpark expects"):
            build_draft_config(target_config, _model_args())

    @unittest.skipUnless(
        HAS_REAL_TARGET,
        "Set DEEPSPEC_QWEN38_MODEL_PATH to run the real-config test.",
    )
    def test_rejects_same_shape_with_incompatible_target_architecture(self):
        target_config = AutoConfig.from_pretrained(REAL_TARGET)
        args = load_config(
            REPO_ROOT / "config/dspark/dspark_qwen3_8_27b.py"
        ).model
        incompatible_fields = {
            "intermediate_size": 16384,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "linear_num_value_heads": 32,
        }
        for field, incompatible_value in incompatible_fields.items():
            with self.subTest(field=field):
                incompatible = copy.deepcopy(target_config)
                setattr(incompatible.text_config, field, incompatible_value)
                with self.assertRaisesRegex(ValueError, field):
                    build_draft_config(incompatible, args)

    @unittest.skipUnless(
        HAS_REAL_TARGET,
        "Set DEEPSPEC_QWEN38_MODEL_PATH to run the real-config test.",
    )
    def test_rejects_incompatible_target_rope(self):
        target_config = AutoConfig.from_pretrained(REAL_TARGET)
        target_config.text_config.rope_parameters = dict(
            target_config.text_config.rope_parameters
        )
        target_config.text_config.rope_parameters["rope_theta"] = 1_000_000
        args = load_config(
            REPO_ROOT / "config/dspark/dspark_qwen3_8_27b.py"
        ).model

        with self.assertRaisesRegex(ValueError, "rope_parameters.rope_theta"):
            build_draft_config(target_config, args)


class Qwen3_8DSparkModelTest(unittest.TestCase):
    def _build_tiny_model(
        self,
        device="cpu",
        *,
        production_heads=False,
        tp4_compatible=False,
    ):
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=8 if tp4_compatible else 4,
            num_key_value_heads=4 if tp4_compatible else 2,
            head_dim=16,
            max_position_embeddings=128,
            layer_types=["full_attention"],
        )
        config.target_layer_ids = [1, 3]
        config.num_target_layers = 4
        config.block_size = 3
        config.mask_token_id = 127
        config.num_anchors = 2
        config.enable_confidence_head = bool(production_heads)
        config.markov_rank = 8 if production_heads else 0
        if production_heads:
            config.markov_head_type = "vanilla"
            config.confidence_head_with_markov = True
        config._attn_implementation = "flex_attention"
        return Qwen3_8DSparkModel(config).float().to(device)

    def test_cp2_tp4_fsdp_forward_and_backward(self):
        runtime = require_torchrun(self, world_size=8)
        torch.manual_seed(20260902)
        parallel_config = ParallelConfig(
            cp=2,
            tp=4,
            use_fsdp=True,
            context_parallel_backend="model_native",
        )
        topology = ParallelContext.build(
            parallel_config,
            device_type=runtime.device.type,
        )
        model = self._build_tiny_model(
            runtime.device,
            tp4_compatible=True,
        ).to(torch.bfloat16)
        model.configure_context_parallel(
            size=2,
            rank=topology.context_parallel_rank,
            group=topology.context_parallel_group,
            model_parallel_group=topology.model_mesh.get_group(),
            model_parallel_src_rank=topology.model_parallel_src_rank,
        )
        model = apply_parallelism(
            model,
            topology,
            parallel_config,
            param_dtype=torch.bfloat16,
            sequence_length=16,
        )
        output = model(
            input_ids=(
                torch.arange(16, device=runtime.device)
                .remainder(120)
                .unsqueeze(0)
            ),
            target_hidden_states=torch.randn(
                1,
                8,
                128,
                device=runtime.device,
                dtype=torch.bfloat16,
            ),
            loss_mask=torch.ones(
                1,
                16,
                dtype=torch.bool,
                device=runtime.device,
            ),
            context_chunk_len=torch.tensor([8], device=runtime.device),
            seq_len=torch.tensor([16], device=runtime.device),
        )
        self.assertEqual(tuple(output.draft_logits.shape), (1, 1, 3, 128))
        output.draft_logits.float().square().mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )
        torch.distributed.barrier()

    def test_initializes_embedding_and_head_from_checkpoint_tensors(self):
        model = self._build_tiny_model()
        embed_weight = torch.randn_like(model.embed_tokens.weight)
        head_weight = torch.randn_like(model.lm_head.weight)
        model.initialize_embedding_and_head_weights(
            embed_weight=embed_weight,
            lm_head_weight=head_weight,
            freeze=True,
        )
        torch.testing.assert_close(model.embed_tokens.weight, embed_weight)
        torch.testing.assert_close(model.lm_head.weight, head_weight)
        self.assertFalse(model.embed_tokens.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "FlexAttention requires CUDA")
    def test_tiny_forward_and_backward(self):
        model = self._build_tiny_model("cuda")
        output = model(
            input_ids=torch.randint(0, 127, (1, 16), device="cuda"),
            target_hidden_states=torch.randn(1, 16, 128, device="cuda"),
            loss_mask=torch.ones(1, 16, dtype=torch.bool, device="cuda"),
            target_last_hidden_states=None,
        )
        self.assertEqual(tuple(output.draft_logits.shape), (1, 2, 3, 128))
        output.draft_logits.square().mean().backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))

    def test_checkpoint_roundtrip_preserves_production_heads(self):
        model = self._build_tiny_model(production_heads=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = Qwen3_8DSparkModel.from_pretrained(
                tmpdir,
                attn_implementation="eager",
            )

        self.assertIsNotNone(loaded.markov_head)
        self.assertIsNotNone(loaded.confidence_head)
        self.assertTrue(loaded.confidence_head_with_markov)
        self.assertEqual(model.state_dict().keys(), loaded.state_dict().keys())
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)

    @unittest.skipUnless(torch.cuda.is_available(), "FlexAttention requires CUDA")
    def test_production_heads_bf16_loss_and_backward(self):
        torch.manual_seed(17)
        model = self._build_tiny_model(
            "cuda",
            production_heads=True,
        ).to(torch.bfloat16)
        model.set_embedding_head_trainable(False)
        output = model(
            input_ids=torch.randint(0, 127, (1, 16), device="cuda"),
            target_hidden_states=torch.randn(
                1,
                16,
                128,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            target_last_hidden_states=torch.randn(
                1,
                16,
                64,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            loss_mask=torch.ones(1, 16, dtype=torch.bool, device="cuda"),
        )
        self.assertEqual(tuple(output.confidence_pred.shape), (1, 2, 3))
        self.assertIsNotNone(output.aligned_target_logits)
        with patch("deepspec.modeling.dspark.loss.add_metric") as add_metric:
            loss = compute_dspark_loss(
                outputs=output,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
            )
        metric_names = [call.args[0] for call in add_metric.call_args_list]
        self.assertIn("step_accept_rate@0", metric_names)
        self.assertNotIn("accept_rate@0", metric_names)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.markov_head.markov_w1.weight.grad)
        self.assertIsNotNone(model.confidence_head.proj.weight.grad)


class Qwen3_8DSparkTrainerTest(unittest.TestCase):
    def test_tp_ranks_share_one_transient_cache_owner(self):
        trainer = object.__new__(Qwen3_8DSparkTrainer)
        trainer.global_rank = 6
        trainer.online_target = SimpleNamespace(cache_replicated_across_tp=True)
        trainer.target_parallel = SimpleNamespace(
            tensor_parallel_size=4,
            tensor_parallel_group=object(),
        )
        with patch(
            "deepspec.trainer.base_trainer.dist.get_process_group_ranks",
            return_value=[4, 5, 6, 7],
        ):
            self.assertEqual(trainer._target_cache_owner_rank(), 4)

    def test_builds_qwen38_partition_target(self):
        trainer = object.__new__(Qwen3_8DSparkTrainer)
        trainer.args = SimpleNamespace(
            model=SimpleNamespace(
                target_model_name_or_path="target",
                target_layer_ids=[1, 3],
            )
        )
        trainer.target_parallel = object()
        trainer.device = torch.device("cpu")
        trainer.checkpoint_dir_root = "/tmp/checkpoint"

        with patch("deepspec.modeling.target.Qwen3_8OnlineTarget") as target_cls:
            target = trainer.build_online_target()

        self.assertIs(target, target_cls.return_value)
        target_cls.assert_called_once_with(
            model_name_or_path="target",
            target_layer_ids=[1, 3],
            topology=trainer.target_parallel,
            device=trainer.device,
            rank_local_cache_dir="/tmp/checkpoint/target_rank_local",
        )

    def test_online_target_batch_is_passed_to_qwen_draft(self):
        hidden = torch.randn(1, 3, 8)

        class FakeOnlineTarget:
            def forward_training_batch(self, batch):
                return {
                    "input_ids": batch["input_ids"][:, :3],
                    "loss_mask": batch["loss_mask"][:, :3],
                    "target_hidden_states": hidden,
                    "target_last_hidden_states": torch.randn(1, 3, 4),
                    "context_chunk_len": torch.tensor([3]),
                    "seq_len": torch.tensor([3]),
                }

        trainer = object.__new__(Qwen3_8DSparkTrainer)
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
        forwarded = {}

        def forward_model(**kwargs):
            forwarded.update(kwargs)
            return object()

        trainer.forward_model = forward_model
        batch = {
            "input_ids": torch.tensor([[10, 11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "loss_mask": torch.tensor([[0, 1, 1, 0]]),
        }
        with patch(
            "deepspec.trainer.dspark_trainer.compute_dspark_loss",
            return_value=torch.ones((), requires_grad=True),
        ):
            trainer.run_batch(batch)

        self.assertIs(forwarded["target_hidden_states"], hidden)
        self.assertEqual(forwarded["context_chunk_len"].item(), 3)
        self.assertNotIn("attention_mask", batch)

    def test_offline_qwen_cache_context_len_remains_supported(self):
        trainer = object.__new__(Qwen3_8DSparkTrainer)
        trainer.online_target_enabled = False
        trainer.data_batch_micro_batches = None
        trainer.args = SimpleNamespace(
            model=SimpleNamespace(
                l1_loss_alpha=0.0,
                confidence_head_alpha=0.0,
                loss_decay_gamma=4.0,
                ce_loss_alpha=1.0,
            )
        )
        forwarded = {}
        trainer.forward_model = lambda **kwargs: forwarded.update(kwargs) or object()
        batch = {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "loss_mask": torch.ones(1, 3, dtype=torch.bool),
            "target_hidden_states": torch.randn(1, 3, 8),
            "context_start": torch.tensor([0]),
            "context_len": torch.tensor([3]),
            "seq_len": torch.tensor([3]),
        }
        with patch(
            "deepspec.trainer.dspark_trainer.compute_dspark_loss",
            return_value=torch.ones((), requires_grad=True),
        ):
            trainer.run_batch(batch)

        self.assertIs(forwarded["context_chunk_len"], batch["context_len"])

    def test_isolated_draft_phase_refuses_inline_target_inference(self):
        trainer = object.__new__(Qwen3_8DSparkTrainer)
        trainer.online_target_enabled = True
        trainer.data_batch_micro_batches = (1,)
        trainer._data_batch_phase = "draft_training"
        with self.assertRaisesRegex(RuntimeError, "precomputed target hidden"):
            trainer.run_batch({"input_ids": torch.ones(1, 1, dtype=torch.long)})
        with self.assertRaisesRegex(RuntimeError, "only allowed"):
            trainer.prepare_online_target_batch({})

    def test_local_checkpoint_initialization_does_not_load_full_target(self):
        class FakeDraft:
            def __init__(self):
                self.initialized_with = None

            def to(self, *, device, dtype):
                self.device = device
                self.dtype = dtype
                return self

            def initialize_embedding_and_head_weights(self, **kwargs):
                self.initialized_with = kwargs

        draft = FakeDraft()
        trainer = object.__new__(Qwen3_8DSparkTrainer)
        trainer.args = SimpleNamespace(
            model=SimpleNamespace(target_model_name_or_path=CONFIGURED_TARGET)
        )
        trainer.device = torch.device("cpu")
        trainer.precision_dtype = torch.bfloat16
        trainer._build_draft_model = lambda **_kwargs: draft
        weights = [torch.randn(4, 2), torch.randn(4, 2)]

        with (
            patch(
                "deepspec.trainer.base_trainer.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(model_type="qwen3_5"),
            ),
            patch(
                "deepspec.trainer.base_trainer.is_multimodal_config",
                return_value=False,
            ),
            patch(
                "deepspec.trainer.base_trainer.AutoTokenizer.from_pretrained",
                return_value=object(),
            ),
            patch(
                "deepspec.trainer.base_trainer._load_checkpoint_tensor",
                side_effect=weights,
            ) as load_tensor,
            patch(
                "deepspec.trainer.base_trainer.load_target_model_with_head"
            ) as load_full_target,
        ):
            built_draft, _tokenizer = trainer.build_models()

        self.assertIs(built_draft, draft)
        self.assertEqual(load_tensor.call_count, 2)
        load_full_target.assert_not_called()
        torch.testing.assert_close(
            draft.initialized_with["embed_weight"],
            weights[0].to(torch.bfloat16),
        )
        torch.testing.assert_close(
            draft.initialized_with["lm_head_weight"],
            weights[1].to(torch.bfloat16),
        )
        self.assertTrue(draft.initialized_with["freeze"])


class Qwen3_8TargetTest(unittest.TestCase):
    def test_deltanet_cp_transfers_transformers_dict_cache_state(self):
        text_config = Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["linear_attention"],
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        source_cache = DynamicCache(config=text_config)
        source_layer = source_cache.layers[0]
        conv_state = torch.randn(1, 12, 4)
        recurrent_state = torch.randn(1, 2, 4, 4)
        source_layer.lazy_initialization(
            conv_states=conv_state,
            recurrent_states=recurrent_state,
            state_idx=0,
        )
        source_layer.conv_states[0].copy_(conv_state)
        source_layer.recurrent_states[0].copy_(recurrent_state)

        with patch(
            "deepspec.modeling.target.qwen3_6_cp._send_optional_tensor"
        ) as send_tensor:
            qwen3_6_cp._send_linear_state(
                cache=source_cache,
                layer_idx=0,
                dst=1,
                group=object(),
                device=torch.device("cpu"),
            )
        self.assertEqual(send_tensor.call_count, 2)
        torch.testing.assert_close(
            send_tensor.call_args_list[0].kwargs["tensor"],
            conv_state,
        )
        torch.testing.assert_close(
            send_tensor.call_args_list[1].kwargs["tensor"],
            recurrent_state,
        )

        destination_cache = DynamicCache(config=text_config)
        with patch(
            "deepspec.modeling.target.qwen3_6_cp._recv_optional_tensor",
            side_effect=[conv_state.clone(), recurrent_state.clone()],
        ):
            qwen3_6_cp._recv_linear_state(
                cache=destination_cache,
                layer_idx=0,
                src=0,
                group=object(),
                device=torch.device("cpu"),
            )
        destination_layer = destination_cache.layers[0]
        torch.testing.assert_close(destination_layer.conv_states[0], conv_state)
        torch.testing.assert_close(
            destination_layer.recurrent_states[0],
            recurrent_state,
        )
        self.assertTrue(destination_layer.has_previous_state[0])

    def test_target_reuses_tp_ranks_as_fsdp_shards(self):
        class FakeModel:
            def eval(self):
                return self

            def requires_grad_(self, value):
                self.requires_grad = value
                return self

        target_config = SimpleNamespace(
            model_type="qwen3_5",
            text_config=SimpleNamespace(num_hidden_layers=64),
            vision_config=SimpleNamespace(depth=27),
        )
        topology = SimpleNamespace(
            expert_parallel_size=1,
            tensor_parallel_size=4,
            context_parallel_size=1,
        )
        model = FakeModel()
        with (
            patch(
                "deepspec.modeling.target.online.AutoConfig.from_pretrained",
                return_value=target_config,
            ),
            patch(
                "deepspec.modeling.target.online._build_rank_local_ep_target_model",
                return_value=model,
            ) as build_model,
            patch(
                "deepspec.modeling.target.online._wrap_target_model_with_fsdp",
                return_value=model,
            ) as wrap_model,
        ):
            target = online.Qwen3_8OnlineTarget(
                model_name_or_path="target",
                target_layer_ids=[1, 16, 31, 46, 61],
                topology=topology,
                device=torch.device("cpu"),
                rank_local_cache_dir="/tmp/cache",
            )

        self.assertIs(target.model, model)
        self.assertEqual(target_config.text_config.num_hidden_layers, 62)
        self.assertEqual(target_config.vision_config.depth, 0)
        self.assertFalse(model.requires_grad)
        self.assertEqual(
            build_model.call_args.kwargs["attn_implementation"], "sdpa"
        )
        self.assertTrue(
            wrap_model.call_args.kwargs["shard_tensor_parallel_dimension"]
        )

    def test_text_batch_is_trimmed_and_returns_qwen_cp_metadata(self):
        teacher = online.Qwen3_8OnlineTarget.__new__(online.Qwen3_8OnlineTarget)
        teacher.model = object()
        teacher.target_layer_ids = [1, 3]
        teacher.device = torch.device("cpu")
        teacher.topology = SimpleNamespace(context_parallel_size=1)
        hidden = torch.randn(1, 3, 8)
        last_hidden = torch.randn(1, 3, 4)

        with patch(
            "deepspec.modeling.target.online._run_target_forward_with_hooks",
            return_value=TargetForwardResult(
                target_hidden_states=hidden,
                target_last_hidden_states=last_hidden,
                context_start=0,
            ),
        ) as target_forward:
            result = teacher.forward_training_batch({
                "input_ids": torch.tensor([[10, 11, 12, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "loss_mask": torch.tensor([[0, 1, 1, 0]]),
            })

        self.assertEqual(tuple(result["input_ids"].shape), (1, 3))
        self.assertEqual(result["context_chunk_len"].item(), 3)
        self.assertEqual(result["seq_len"].item(), 3)
        self.assertIs(result["target_hidden_states"], hidden)
        self.assertIs(result["target_last_hidden_states"], last_hidden)
        target_forward.assert_called_once()

    def test_native_head_tail_cp_shape_is_validated(self):
        class FakeTarget:
            _deepspec_context_layout = "native_head_tail"

            def forward_context_parallel(self, **_kwargs):
                return TargetForwardResult(
                    target_hidden_states=torch.randn(1, 2, 8),
                    target_last_hidden_states=torch.randn(1, 2, 4),
                    context_start=0,
                )

        result = online._run_target_forward_context_parallel(
            target_model=FakeTarget(),
            model_inputs={
                "input_ids": torch.tensor([[10, 11, 12]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            },
            target_layer_ids=[1, 3],
            topology=SimpleNamespace(
                context_parallel_group=None,
                context_parallel_rank=0,
                context_parallel_size=2,
                tensor_parallel_group=None,
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
            ),
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(result.target_hidden_states.shape), (1, 2, 8))

if __name__ == "__main__":
    unittest.main()

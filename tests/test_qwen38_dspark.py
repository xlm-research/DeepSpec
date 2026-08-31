import copy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from deepspec.eval.dspark import Qwen3_8DSparkEvaluator
from deepspec.modeling.dspark.qwen3_8 import (
    Qwen3_8DSparkModel,
    build_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.trainer import Qwen3_8DSparkTrainer
from deepspec.utils.config import ConfigNode, load_config


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
        self.assertEqual(config.model.target_model_name_or_path, CONFIGURED_TARGET)
        self.assertFalse(config.data.multimodal)
        self.assertTrue(config.data.store_target_last_hidden_states)
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
    def _build_tiny_model(self, device="cpu", *, production_heads=False):
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
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


if __name__ == "__main__":
    unittest.main()

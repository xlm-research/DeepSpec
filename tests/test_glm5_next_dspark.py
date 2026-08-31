import copy
from types import SimpleNamespace
import unittest

import torch
from transformers import AutoConfig, AutoTokenizer

from deepspec.modeling.dspark.glm5_next import (
    Glm5NextDSparkModel,
    build_draft_config,
)
from deepspec.modeling.target import Glm5NextOnlineTarget
from deepspec.trainer import Glm5NextDSparkTrainer
from deepspec.utils import load_config
from deepspec.data.parser import preprocess_record


TARGET = "/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash"


def tiny_target_config():
    config = AutoConfig.from_pretrained(TARGET)
    text = config.text_config
    text.hidden_size = 64
    text.num_attention_heads = 4
    text.num_key_value_heads = 4
    text.q_lora_rank = 32
    text.kv_lora_rank = 16
    text.qk_nope_head_dim = 16
    text.qk_rope_head_dim = 0
    text.qk_head_dim = 16
    text.v_head_dim = 16
    text.intermediate_size = 128
    text.moe_intermediate_size = 32
    text.n_routed_experts = 8
    text.n_shared_experts = 1
    text.num_experts_per_tok = 2
    text.hc_mult = 2
    text.vocab_size = 128
    text.pad_token_id = 127
    text.num_hidden_layers = 3
    return config


def model_args():
    args = copy.deepcopy(load_config("config/dspark/dspark_glm5_3_flash.py").model)
    args.mask_token_id = 127
    args.num_anchors = 2
    args.sliding_window = 16
    args.markov_rank = 16
    return args


class Glm5NextDSparkTest(unittest.TestCase):
    def test_config_selects_glm_trainer_and_parallel_layout(self):
        config = load_config("config/dspark/dspark_glm5_3_flash.py")
        self.assertIs(config.train.trainer_cls, Glm5NextDSparkTrainer)
        self.assertEqual(config.model.target_model_name_or_path, TARGET)
        self.assertEqual(config.train.parallel.ep, 8)
        self.assertEqual(config.train.target_parallel.ep, 1)
        self.assertEqual(config.data.chat_template, "glm5_next")

    def test_draft_forward_and_backward(self):
        config = build_draft_config(tiny_target_config(), model_args())
        self.assertEqual(config.architectures, ["Glm5NextDSparkModel"])
        self.assertEqual(config.model_type, "glm5_next_text")
        self.assertEqual(config.mlp_layer_types, ["sparse"] * 3)
        self.assertEqual(config._experts_implementation, "grouped_mm")

        model = Glm5NextDSparkModel(config).float()
        seq_len = 32
        output = model(
            input_ids=torch.randint(0, 127, (1, seq_len)),
            target_hidden_states=torch.randn(1, seq_len, 3 * 64),
            loss_mask=torch.ones(1, seq_len, dtype=torch.bool),
            target_last_hidden_states=torch.randn(1, seq_len, 64),
            context_start=torch.tensor([0]),
            context_len=torch.tensor([seq_len]),
            seq_len=torch.tensor([seq_len]),
        )
        self.assertEqual(tuple(output.draft_logits.shape), (1, 2, 7, 128))
        self.assertEqual(tuple(output.target_ids.shape), (1, 2, 7))
        self.assertEqual(tuple(output.aligned_target_logits.shape), (1, 2, 7, 128))
        self.assertEqual(tuple(output.confidence_pred.shape), (1, 2, 7))

        loss = output.draft_logits.float().square().mean()
        loss = loss + output.confidence_pred.float().square().mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_glm_chat_template_marks_only_assistant_content(self):
        tokenizer = AutoTokenizer.from_pretrained(TARGET)
        processed = preprocess_record(
            record={
                "conversations": [
                    {"role": "user", "content": "first-user"},
                    {"role": "assistant", "content": "first-answer"},
                    {"role": "user", "content": "second-user"},
                    {"role": "assistant", "content": "second-answer"},
                ]
            },
            tokenizer=tokenizer,
            chat_template="glm5_next",
            max_length=256,
        )
        selected = tokenizer.decode(
            processed["input_ids"][processed["loss_mask"].bool()]
        )
        self.assertIn("first-answer", selected)
        self.assertIn("second-answer", selected)
        self.assertNotIn("first-user", selected)
        self.assertNotIn("second-user", selected)
        self.assertNotIn("<|user|>", selected)

    def test_online_target_rejects_unimplemented_parallel_axes_before_loading(self):
        base = dict(
            expert_parallel_size=1,
            tensor_parallel_size=1,
            context_parallel_size=1,
        )
        for field, message in (
            ("expert_parallel_size", "target_parallel.ep=1"),
            ("tensor_parallel_size", "parallel.tp=1"),
            ("context_parallel_size", "parallel.cp=1"),
        ):
            values = dict(base)
            values[field] = 2
            with self.subTest(field=field), self.assertRaisesRegex(
                NotImplementedError, message
            ):
                Glm5NextOnlineTarget(
                    model_name_or_path=TARGET,
                    target_layer_ids=[0, 1, 2],
                    topology=SimpleNamespace(**values),
                    device=torch.device("cpu"),
                    rank_local_cache_dir="unused",
                )


if __name__ == "__main__":
    unittest.main()

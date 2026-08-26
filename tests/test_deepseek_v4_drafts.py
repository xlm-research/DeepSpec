import copy
import unittest

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


if __name__ == "__main__":
    unittest.main()

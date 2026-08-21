import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from deepspec.modeling.dflash2.common import (
    CandidateSelector,
    GroupedDynamicCausalConv,
)
from deepspec.modeling.dflash2.qwen3_8.config import build_draft_config
from deepspec.utils.config import load_config, parse_opts_to_config


class DFlash2DynamicConvTest(unittest.TestCase):
    def test_identity_initialization(self):
        conv = GroupedDynamicCausalConv(
            hidden_size=4,
            kernel_size=2,
            group_size=2,
        )
        conv.reset_to_identity()
        hidden = torch.randn(2, 8, 4)
        prepared, kernel = conv.prepare(hidden, block_size=4)
        output = conv.finish(prepared, kernel, block_size=4)
        torch.testing.assert_close(output, hidden)

    def test_convolution_does_not_cross_sampled_anchor_blocks(self):
        conv = GroupedDynamicCausalConv(
            hidden_size=4,
            kernel_size=2,
            group_size=2,
        )
        with torch.no_grad():
            conv.kernel_projection.weight.zero_()
            conv.base_kernel.zero_()
            conv.base_kernel[0, 1].fill_(1.0)
            conv.base_kernel[1, 0].fill_(1.0)
        hidden = torch.tensor(
            [[[1.0] * 4, [2.0] * 4, [100.0] * 4, [200.0] * 4]]
        )
        prepared, kernel = conv.prepare(hidden, block_size=2)
        output = conv.finish(prepared, kernel, block_size=2)
        expected = torch.tensor(
            [[[0.0] * 4, [1.0] * 4, [0.0] * 4, [100.0] * 4]]
        )
        torch.testing.assert_close(output, expected)


class DFlash2CandidateSelectorTest(unittest.TestCase):
    def test_teacher_forced_topk_targets(self):
        selector = CandidateSelector(
            vocab_size=6,
            hidden_size=4,
            rank=2,
            top_k=2,
            initializer_range=0.02,
        )
        with torch.no_grad():
            selector.predecessor_codebook.zero_()
            selector.successor_codebook.zero_()
            selector.hidden_projection.weight.zero_()
        logits = torch.tensor(
            [[[[0.0, 5.0, 4.0, 1.0, 0.0, -1.0],
               [0.0, 1.0, 2.0, 5.0, 4.0, -1.0]]]]
        )
        outputs = selector.training_outputs(
            hidden=torch.zeros(1, 1, 2, 4),
            logits=logits,
            predecessor_ids=torch.tensor([[[0, 1]]]),
            target_ids=torch.tensor([[[2, 5]]]),
            eval_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        )
        self.assertEqual(tuple(outputs["selector_scores"].shape), (1, 1, 2, 2))
        self.assertEqual(outputs["selector_loss_mask"].tolist(), [[[True, False]]])
        self.assertEqual(
            outputs["selector_recall_mask"].tolist(),
            [[[True, False]]],
        )

    def test_public_checkpoint_codebook_keys_have_no_weight_suffix(self):
        selector = CandidateSelector(
            vocab_size=8,
            hidden_size=4,
            rank=2,
            top_k=2,
            initializer_range=0.02,
        )
        keys = set(selector.state_dict())
        self.assertIn("predecessor_codebook", keys)
        self.assertIn("successor_codebook", keys)
        self.assertNotIn("predecessor_codebook.weight", keys)

    def test_selector_follows_the_selected_predecessor(self):
        selector = CandidateSelector(
            vocab_size=5,
            hidden_size=1,
            rank=1,
            top_k=2,
            initializer_range=0.02,
        )
        with torch.no_grad():
            selector.predecessor_codebook.zero_()
            selector.successor_codebook.zero_()
            selector.hidden_projection.weight.fill_(1.0)
            selector.predecessor_codebook[0].fill_(1.0)
            selector.predecessor_codebook[2].fill_(1.0)
            selector.successor_codebook[2].fill_(10.0)
            selector.successor_codebook[3].fill_(10.0)
        logits = torch.tensor(
            [[[0.0, 5.0, 4.0, -10.0, -10.0],
              [0.0, 4.0, -10.0, 5.0, -10.0]]]
        )
        tokens, candidates, probs = selector.select(
            hidden=torch.ones(1, 2, 1),
            logits=logits,
            anchor_ids=torch.tensor([0]),
            temperature=0.0,
        )
        self.assertEqual(tokens.tolist(), [[2, 3]])
        self.assertEqual(tuple(candidates.shape), (1, 2, 2))
        self.assertIsNone(probs)


class Qwen3_8DFlash2ConfigTest(unittest.TestCase):
    def test_documented_qwen38_layout(self):
        target_config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=5120,
                vocab_size=248320,
                num_hidden_layers=64,
            )
        )
        model_args = SimpleNamespace(
            verification_block_size=8,
            num_draft_layers=5,
            target_layer_ids=[5, 19, 33, 47, 61],
            conv_group_size=16,
            conv_kernel_size=2,
            mask_token_id=248070,
            selector_rank=256,
            selector_top_k=16,
            num_anchors=512,
        )
        config = build_draft_config(target_config, model_args)
        self.assertEqual(config.architectures, ["DFlash2DraftModel"])
        self.assertEqual(config.verification_block_size, 8)
        self.assertEqual(config.block_size, 7)
        self.assertEqual(config.proposal_hidden_offset, 1)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.dflash_config["block_size"], 8)
        self.assertEqual(config.dflash_config["selector_top_k"], 16)
        self.assertTrue(config.use_sliding_window)
        self.assertEqual(config.sliding_window, 2048)

    def test_checkpoint_directory_can_be_overridden(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = parse_opts_to_config(
            ["logging.checkpoint_dir=/tmp/dflash2-test-checkpoints"],
            load_config(repo_root / "config/dflash2/dflash2_qwen3_8_27b.py"),
        )
        self.assertEqual(
            config.logging.checkpoint_dir,
            "/tmp/dflash2-test-checkpoints",
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from deepspec.modeling.dflash2.common import (
    CandidateSelector,
    GroupedDynamicCausalConv,
)
from deepspec.modeling.dflash2.loss import compute_dflash2_loss
from deepspec.modeling.dflash2.qwen3_8.config import build_draft_config
from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.utils.config import load_config, parse_opts_to_config
from deepspec.utils.metrics import configure_reduction_group
from deepspec.utils.metrics import flush as flush_metrics
from deepspec.utils.metrics import reset as reset_metrics
from deepspec.utils import training_logger
from tests.distributed_test_utils import require_torchrun


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
    def test_selector_starts_as_noop_and_can_learn_on_first_step(self):
        selector = CandidateSelector(
            vocab_size=8,
            hidden_size=4,
            rank=2,
            top_k=3,
            initializer_range=0.02,
        )
        hidden = torch.randn(2, 4)
        unary = torch.randn(2, 3)
        candidates = torch.tensor([[1, 3, 5], [0, 2, 7]])
        predecessors = torch.tensor([4, 6])
        scores = selector.pair_scores(
            hidden=hidden,
            unary=unary,
            candidate_ids=candidates,
            predecessor_ids=predecessors,
        )
        torch.testing.assert_close(scores, unary)
        self.assertEqual(
            torch.count_nonzero(selector.successor_codebook).item(), 0
        )

        scores.sum().backward()
        self.assertIsNotNone(selector.successor_codebook.grad)
        self.assertGreater(
            selector.successor_codebook.grad.abs().sum().item(), 0.0
        )

    def test_selector_rejects_candidate_pool_larger_than_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "selector_top_k"):
            CandidateSelector(
                vocab_size=4,
                hidden_size=2,
                rank=1,
                top_k=5,
                initializer_range=0.02,
            )

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


class DFlash2LossTest(unittest.TestCase):
    def test_base_and_selector_cross_entropy_are_both_trainable(self):
        self.addCleanup(reset_metrics)
        reset_metrics()
        draft_logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
        selector_scores = torch.zeros(1, 1, 2, 2, requires_grad=True)
        outputs = DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=torch.tensor([[[0, 1]]]),
            eval_mask=torch.ones(1, 1, 2, dtype=torch.bool),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
            selector_scores=selector_scores,
            selector_target_indices=torch.tensor([[[0, 1]]]),
            selector_loss_mask=torch.ones(1, 1, 2, dtype=torch.bool),
            selector_recall_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        )
        with patch("deepspec.modeling.dflash2.loss.add_metric") as add_metric:
            loss = compute_dflash2_loss(
                outputs=outputs,
                loss_decay_gamma=None,
                ce_loss_alpha=1.0,
                selector_loss_alpha=1.0,
                selector_loss_decay_gamma=None,
            )
        dflash2_metric = next(
            call
            for call in add_metric.call_args_list
            if call.args[0] == "dflash2_loss"
        )
        self.assertEqual(dflash2_metric.kwargs["reduction"], "dp_mean")
        torch.testing.assert_close(
            loss.detach(),
            torch.log(torch.tensor(3.0)) + torch.log(torch.tensor(2.0)),
        )
        loss.backward()
        self.assertGreater(draft_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(selector_scores.grad.abs().sum().item(), 0.0)

    def test_console_prefers_global_dflash2_loss(self):
        with patch.object(training_logger, "print_on_global_main") as printer:
            training_logger._print_summary(
                summary={"train/loss": 0.0, "train/dflash2_loss": 12.5},
                global_step=1,
                next_micro_step=1,
                micro_batches_per_epoch=1,
                max_train_steps=1,
            )
        self.assertIn("loss=12.5000", printer.call_args.args[0])


class DFlash2DistributedLossTest(unittest.TestCase):
    def test_global_loss_when_rank_zero_has_no_valid_anchor(self):
        runtime = require_torchrun(self, world_size=2)
        configure_loss_reduction_group(dist.group.WORLD)
        configure_reduction_group(dist.group.WORLD)
        reset_metrics()

        has_tokens = runtime.global_rank == 1
        eval_mask = torch.full(
            (1, 1, 2),
            has_tokens,
            dtype=torch.bool,
            device=runtime.device,
        )
        draft_logits = torch.zeros(
            1, 1, 2, 3, device=runtime.device, requires_grad=True
        )
        selector_scores = torch.zeros(
            1, 1, 2, 2, device=runtime.device, requires_grad=True
        )
        outputs = DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=torch.tensor([[[0, 1]]], device=runtime.device),
            eval_mask=eval_mask,
            block_keep_mask=torch.full(
                (1, 1), has_tokens, dtype=torch.bool, device=runtime.device
            ),
            selector_scores=selector_scores,
            selector_target_indices=torch.tensor(
                [[[0, 1]]], device=runtime.device
            ),
            selector_loss_mask=eval_mask,
            selector_recall_mask=eval_mask,
        )
        loss = compute_dflash2_loss(
            outputs=outputs,
            loss_decay_gamma=None,
            ce_loss_alpha=1.0,
            selector_loss_alpha=1.0,
            selector_loss_decay_gamma=None,
        )
        loss.backward()
        metrics = flush_metrics()
        expected = torch.log(torch.tensor(3.0)) + torch.log(torch.tensor(2.0))
        self.assertAlmostEqual(
            metrics["train/dflash2_loss"], expected.item(), places=5
        )
        dist.barrier()


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
        self.assertEqual(config.target_context_layout, "native_head_tail")
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

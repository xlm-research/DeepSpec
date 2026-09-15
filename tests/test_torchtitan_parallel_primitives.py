"""CPU distributed derivative checks; real phase acceptance runs separately."""

import unittest
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchtitan.models.dspark_draft.common import DSparkForwardOutput
from torchtitan.models.dspark_draft.loss import DSparkLoss
from torchtitan.models.dspark_draft.sequence_parallel import (
    gather_sequence,
    scatter_sequence,
    SequenceLinear,
    SequenceScale,
)
from torchtitan.models.dspark_draft.vocabulary_parallel import VocabLinear


class ParallelPrimitiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "RANK" not in os.environ:
            raise unittest.SkipTest("Run under torchrun with four Gloo ranks")
        if not dist.is_initialized():
            dist.init_process_group("gloo")

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_uneven_sequence_projection_and_norm_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(1219)
            source = torch.randn(2, 6, 8).to(dtype).requires_grad_()
            weight = torch.randn(12, 8).to(dtype).requires_grad_()
            scale = torch.randn(12).to(dtype).requires_grad_()
            expected = F.linear(source, weight) * scale
            grad = torch.randn_like(expected)
            expected.backward(grad)
            x, w, s = [
                value.detach().clone().requires_grad_()
                for value in (source, weight, scale)
            ]
            local = scatter_sequence(x, dist.group.WORLD)
            local = SequenceLinear.apply(local, w, None, dist.group.WORLD)
            local = SequenceScale.apply(local, s, dist.group.WORLD)
            actual = gather_sequence(local, dist.group.WORLD)
            actual.backward(grad)
            for observed, reference in (
                (actual, expected),
                (x.grad, source.grad),
                (w.grad, weight.grad),
                (s.grad, scale.grad),
            ):
                torch.testing.assert_close(observed, reference, rtol=1e-5, atol=1e-6)

    def test_full_dspark_vocab_loss_and_frozen_head_hidden_gradient(self):
        group = dist.group.WORLD
        rank, size = dist.get_rank(), dist.get_world_size()
        for zero_mask in (False, True):
            torch.manual_seed(1415)
            hidden = torch.randn(1, 2, 3, 8, requires_grad=True)
            head = torch.randn(32, 8)
            markov = torch.randn(1, 2, 3, 32, requires_grad=True)
            confidence = torch.randn(1, 2, 3, requires_grad=True)
            teacher = torch.randn(1, 2, 3, 32)
            targets = torch.randint(32, (1, 2, 3))
            mask = (
                torch.zeros_like(targets, dtype=torch.bool)
                if zero_mask
                else torch.tensor([[[True, True, False], [True, False, False]]])
            )
            keep = mask.any(-1)
            logits = F.linear(hidden, head) + markov
            reference_fn = DSparkLoss(DSparkLoss.Config())
            reference_fn.gas = 2
            prediction = DSparkForwardOutput(
                logits, targets, mask, keep, confidence, teacher
            )
            expected, expected_terms = reference_fn(prediction, targets)
            expected.backward()
            x = hidden.detach().clone().requires_grad_()
            c = confidence.detach().clone().requires_grad_()
            local_bias = markov.detach().chunk(size, -1)[rank].clone().requires_grad_()
            local_logits = (
                VocabLinear.apply(x, head.chunk(size, 0)[rank], None, group)
                + local_bias
            )
            local_prediction = DSparkForwardOutput(
                local_logits, targets, mask, keep, c, teacher.chunk(size, -1)[rank]
            )
            loss_fn = DSparkLoss(DSparkLoss.Config(enable_vocab_parallel=True))
            loss_fn.gas = 2
            loss_fn.vocab_group = group
            actual, actual_terms = loss_fn(local_prediction, targets)
            actual.backward()
            for observed, reference in (
                (actual, expected),
                (x.grad, hidden.grad),
                (c.grad, confidence.grad),
                (local_bias.grad, markov.grad.chunk(size, -1)[rank]),
            ):
                torch.testing.assert_close(observed, reference, rtol=1e-4, atol=1e-6)
            for key in expected_terms:
                torch.testing.assert_close(
                    actual_terms[key], expected_terms[key], rtol=1e-5, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()

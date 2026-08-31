import unittest
from unittest.mock import patch

import torch

from deepspec.eval.dspark.draft_ops import build_dspark_proposal
from deepspec.utils.sampling import (
    logits_to_probs,
    sample_residual,
    sample_tokens,
)


class SamplingTest(unittest.TestCase):
    def test_top_k_filters_and_renormalizes_in_float32(self):
        logits = torch.tensor([[[1.0, 4.0, 3.0, 2.0]]], dtype=torch.bfloat16)
        probs = logits_to_probs(logits, temperature=1.0, top_k=2)

        expected = torch.zeros_like(probs)
        expected[..., 1:3] = torch.softmax(
            torch.tensor([4.0, 3.0]),
            dim=-1,
        )
        self.assertEqual(probs.dtype, torch.float32)
        torch.testing.assert_close(probs, expected, rtol=0.0, atol=0.0)

    def test_top_p_keeps_crossing_token_and_at_least_one(self):
        logits = torch.log(
            torch.tensor([[[0.4, 0.3, 0.2, 0.1]]], dtype=torch.float32)
        )
        probs = logits_to_probs(logits, temperature=1.0, top_p=0.6)
        expected = torch.tensor([[[4.0 / 7.0, 3.0 / 7.0, 0.0, 0.0]]])
        torch.testing.assert_close(probs, expected)

        tiny_nucleus = logits_to_probs(logits, temperature=1.0, top_p=1e-12)
        torch.testing.assert_close(
            tiny_nucleus,
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_top_k_and_top_p_compose_without_non_finite_probabilities(self):
        logits = torch.tensor(
            [[[12.0, 11.0, 10.0, -100.0, -200.0]]],
            dtype=torch.bfloat16,
        )
        probs = logits_to_probs(
            logits,
            temperature=0.01,
            top_k=3,
            top_p=0.9,
        )
        self.assertTrue(bool(torch.isfinite(probs).all().item()))
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, 1))
        torch.testing.assert_close(
            probs,
            torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_sampling_filter_validation_and_greedy_behavior(self):
        logits = torch.tensor([[[1.0, 3.0, 2.0]]], dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "top_k"):
            logits_to_probs(logits, temperature=1.0, top_k=-1)
        with self.assertRaisesRegex(ValueError, "top_p"):
            logits_to_probs(logits, temperature=1.0, top_p=0.0)
        with self.assertRaisesRegex(ValueError, "top_p"):
            logits_to_probs(logits, temperature=1.0, top_p=1.1)

        greedy = logits_to_probs(
            logits,
            temperature=0.0,
            top_k=-1,
            top_p=0.0,
        )
        torch.testing.assert_close(greedy, torch.tensor([[[0.0, 1.0, 0.0]]]))

    def test_bfloat16_proposal_uses_verification_probabilities(self):
        logits = torch.tensor(
            [[[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]]],
            dtype=torch.bfloat16,
        )

        class DraftModel:
            proposal_hidden_offset = 0
            confidence_head = None

            def compute_logits(self, _hidden_states):
                return logits

            def sample_draft_tokens(
                self,
                base_logits,
                *,
                first_prev_token_ids,
                temperature,
                hidden_states,
            ):
                del first_prev_token_ids, hidden_states
                return sample_tokens(base_logits, temperature), base_logits

        sampled_probabilities = []

        def capture_multinomial(probabilities, num_samples):
            self.assertEqual(num_samples, 1)
            sampled_probabilities.append(probabilities.detach().clone())
            return probabilities.argmax(dim=-1, keepdim=True)

        with patch("torch.multinomial", side_effect=capture_multinomial):
            proposal = build_dspark_proposal(
                DraftModel(),
                draft_input_ids=torch.tensor([[7]]),
                block_hidden=torch.zeros(1, 2, 4),
                block_size=2,
                temperature=0.7,
                confidence_threshold=0.0,
            )

        expected = logits_to_probs(logits, temperature=0.7)
        self.assertEqual(sampled_probabilities[0].dtype, torch.float32)
        torch.testing.assert_close(
            sampled_probabilities[0],
            expected.reshape(-1, expected.size(-1)),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            proposal.draft_probs,
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_greedy_sampling_does_not_call_multinomial(self):
        logits = torch.tensor([[[1.0, 3.0, 2.0]]], dtype=torch.bfloat16)
        with patch("torch.multinomial") as multinomial:
            sampled = sample_tokens(logits, temperature=0.0)
        multinomial.assert_not_called()
        torch.testing.assert_close(sampled, torch.tensor([[1]]))

    def test_tiny_positive_residual_is_not_replaced_by_target(self):
        target_probs = torch.tensor([[1e-9, 1.0]], dtype=torch.float32)
        draft_probs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        sampled_probabilities = []

        def capture_multinomial(probabilities, num_samples):
            self.assertEqual(num_samples, 1)
            sampled_probabilities.append(probabilities.detach().clone())
            return probabilities.argmax(dim=-1, keepdim=True)

        with patch("torch.multinomial", side_effect=capture_multinomial):
            token = sample_residual(target_probs, draft_probs)

        torch.testing.assert_close(token, torch.tensor([0]))
        torch.testing.assert_close(
            sampled_probabilities[0],
            torch.tensor([[1.0, 0.0]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_stochastic_rejection_recovers_target_distribution(self):
        torch.manual_seed(20260831)
        num_samples = 200_000
        target_logits = torch.tensor(
            [[[1.25, -0.75, 0.0, 0.5]]], dtype=torch.bfloat16
        )
        draft_logits = torch.tensor(
            [[[-0.5, 1.0, 0.25, 0.75]]], dtype=torch.bfloat16
        )
        target_probs = logits_to_probs(target_logits, temperature=0.8)[0, 0]
        draft_probs = logits_to_probs(draft_logits, temperature=0.8)[0, 0]

        proposals = sample_tokens(
            draft_logits.expand(num_samples, -1, -1),
            temperature=0.8,
        ).squeeze(1)
        selected_target = target_probs[proposals]
        selected_draft = draft_probs[proposals]
        accepted = torch.rand(num_samples) < torch.minimum(
            selected_target / selected_draft,
            torch.ones_like(selected_target),
        )
        output = proposals.clone()
        num_rejected = int((~accepted).sum().item())
        if num_rejected:
            output[~accepted] = sample_residual(
                target_probs.expand(num_rejected, -1),
                draft_probs.expand(num_rejected, -1),
            )

        empirical = torch.bincount(output, minlength=4).float() / num_samples
        torch.testing.assert_close(empirical, target_probs, rtol=0.0, atol=0.004)


if __name__ == "__main__":
    unittest.main()

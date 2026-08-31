import argparse
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

from deepspec.eval.base_evaluator import (
    BaseEvaluator,
    DraftProposal,
    generate_decoding_sample,
    verify_draft_tokens,
)
from tests.distributed_test_utils import require_torchrun


class _FakeTargetAdapter:
    def __init__(self):
        self.reconcile_count = 0

    def create_generation_cache(self, target_model):
        del target_model
        return object()

    def build_prefill_inputs(self, *, target_model, model_inputs, cache):
        del target_model, cache
        return {"input_ids": model_inputs["input_ids"], "is_prefill": True}

    def build_verify_inputs(
        self,
        *,
        verify_input_ids,
        position_ids,
        start,
        cache,
    ):
        del position_ids, start, cache
        return {"input_ids": verify_input_ids, "is_prefill": False}

    def reconcile_generation_cache(self, **kwargs):
        self.reconcile_count += 1
        return kwargs["cache"]


class _FakeTargetModel:
    def __init__(self, *, initial_token=1, bonus_token=2, vocab_size=8):
        self.initial_token = int(initial_token)
        self.bonus_token = int(bonus_token)
        self.vocab_size = int(vocab_size)
        self.call_count = 0
        self.verify_input_lengths = []

    def __call__(self, *, input_ids, is_prefill):
        self.call_count += 1
        if is_prefill:
            logits = torch.full(
                (input_ids.shape[0], 1, self.vocab_size),
                -100.0,
                device=input_ids.device,
            )
            logits[..., self.initial_token] = 0.0
        else:
            logits = torch.full(
                (*input_ids.shape, self.vocab_size),
                -100.0,
                device=input_ids.device,
            )
            self.verify_input_lengths.append(int(input_ids.shape[1]))
            for pos_idx in range(input_ids.shape[1] - 1):
                next_token = int(input_ids[0, pos_idx + 1].item())
                logits[:, pos_idx, next_token] = 0.0
            logits[:, -1, self.bonus_token] = 0.0
        return SimpleNamespace(logits=logits)


def _generate(
    *,
    max_new_tokens,
    proposal_tokens=(2, 2, 2),
    initial_token=1,
    bonus_token=2,
    stop_token_ids=None,
):
    adapter = _FakeTargetAdapter()
    model = _FakeTargetModel(
        initial_token=initial_token,
        bonus_token=bonus_token,
    )
    counters = {"init": 0, "propose": 0, "update": 0}

    def init_context(**kwargs):
        del kwargs
        counters["init"] += 1
        return SimpleNamespace()

    def propose(*, output_ids, start, **kwargs):
        del kwargs
        counters["propose"] += 1
        tokens = torch.tensor(
            [proposal_tokens],
            dtype=torch.long,
            device=output_ids.device,
        )
        verify_input_ids = torch.cat(
            [output_ids[:, start : start + 1], tokens],
            dim=1,
        )
        draft_probs = torch.zeros(
            (1, len(proposal_tokens), model.vocab_size),
            dtype=torch.float32,
            device=output_ids.device,
        )
        draft_probs.scatter_(-1, tokens.unsqueeze(-1), 1.0)
        return DraftProposal(
            draft_token_count=len(proposal_tokens),
            verify_input_ids=verify_input_ids,
            draft_probs=draft_probs,
        )

    def update(context, verification):
        del context, verification
        counters["update"] += 1

    response = generate_decoding_sample(
        target_model=model,
        input_ids=torch.tensor([[6, 7]], dtype=torch.long),
        max_new_tokens=max_new_tokens,
        max_proposal_tokens=3,
        temperature=0.0,
        stop_token_ids=stop_token_ids,
        init_context=init_context,
        propose=propose,
        update=update,
        target_adapter=adapter,
    )
    return response, model, adapter, counters


class GenerationBoundaryTest(unittest.TestCase):
    def test_zero_budget_returns_prompt_without_model_call(self):
        response, model, _, counters = _generate(max_new_tokens=0)

        self.assertEqual(response.output_ids.tolist(), [[6, 7]])
        self.assertEqual(response.num_output_tokens, 0)
        self.assertEqual(response.verify_count, 0)
        self.assertEqual(model.call_count, 0)
        self.assertEqual(counters, {"init": 0, "propose": 0, "update": 0})

    def test_one_token_budget_does_not_start_a_proposal(self):
        response, model, _, counters = _generate(max_new_tokens=1)

        self.assertEqual(response.num_output_tokens, 1)
        self.assertEqual(response.verify_count, 0)
        self.assertEqual(response.acceptance_lengths, [])
        self.assertEqual(model.call_count, 1)
        self.assertEqual(counters, {"init": 0, "propose": 0, "update": 0})

    def test_final_proposal_is_limited_to_output_budget(self):
        response, model, _, counters = _generate(max_new_tokens=4)

        self.assertEqual(response.num_output_tokens, 4)
        self.assertEqual(response.acceptance_lengths, [3])
        self.assertEqual(response.proposal_lengths, [2])
        self.assertEqual(response.accepted_draft_lengths, [2])
        self.assertEqual(model.verify_input_lengths, [3])
        self.assertEqual(counters["propose"], 1)

    def test_single_remaining_slot_uses_target_only_verification(self):
        response, model, _, _ = _generate(max_new_tokens=2)

        self.assertEqual(response.num_output_tokens, 2)
        self.assertEqual(response.acceptance_lengths, [1])
        self.assertEqual(response.proposal_lengths, [0])
        self.assertEqual(response.accepted_draft_lengths, [0])
        self.assertEqual(model.verify_input_lengths, [1])

    def test_initial_eos_stops_before_context_initialization(self):
        response, _, _, counters = _generate(
            max_new_tokens=8,
            initial_token=4,
            stop_token_ids=[4],
        )

        self.assertEqual(response.num_output_tokens, 1)
        self.assertEqual(response.output_ids[0, -1].item(), 4)
        self.assertEqual(response.verify_count, 0)
        self.assertEqual(counters, {"init": 0, "propose": 0, "update": 0})

    def test_accepted_draft_eos_truncates_metrics_at_eos(self):
        response, _, adapter, counters = _generate(
            max_new_tokens=8,
            proposal_tokens=(2, 4, 3),
            stop_token_ids=[4],
        )

        self.assertEqual(response.num_output_tokens, 3)
        self.assertEqual(response.output_ids[0, -1].item(), 4)
        self.assertEqual(response.acceptance_lengths, [2])
        self.assertEqual(response.proposal_lengths, [2])
        self.assertEqual(response.accepted_draft_lengths, [2])
        self.assertEqual(adapter.reconcile_count, 0)
        self.assertEqual(counters["update"], 0)

    def test_target_eos_skips_unused_cache_reconciliation(self):
        response, _, adapter, counters = _generate(
            max_new_tokens=8,
            proposal_tokens=(2,),
            bonus_token=4,
            stop_token_ids=[4],
        )

        self.assertEqual(response.num_output_tokens, 3)
        self.assertEqual(response.output_ids[0, -1].item(), 4)
        self.assertEqual(response.acceptance_lengths, [2])
        self.assertEqual(adapter.reconcile_count, 0)
        self.assertEqual(counters["update"], 0)

    def test_negative_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _generate(max_new_tokens=-1)


class _MetricsEvaluator(BaseEvaluator):
    @property
    def max_proposal_tokens(self):
        return self._max_proposal_tokens


def _make_metrics_evaluator(max_proposal_tokens=3):
    evaluator = object.__new__(_MetricsEvaluator)
    evaluator._max_proposal_tokens = max_proposal_tokens
    evaluator.device = torch.device("cpu")
    return evaluator


def _metric_responses():
    return [
        SimpleNamespace(
            acceptance_lengths=[4, 2, 3, 1, 1],
            proposal_lengths=[3, 3, 2, 1, 0],
            accepted_draft_lengths=[3, 1, 2, 0, 0],
        )
    ]


class AcceptanceMetricTest(unittest.TestCase):
    def test_conditional_prefix_and_block_rates_have_distinct_denominators(self):
        evaluator = _make_metrics_evaluator()
        with mock.patch(
            "deepspec.eval.base_evaluator.dist.all_reduce",
            side_effect=lambda tensor, op: None,
        ):
            summary = evaluator.allreduce_response_metrics(_metric_responses())
        row = evaluator.build_metrics_row(dataset_name="toy", metric_summary=summary)

        self.assertEqual(summary["proposals_at_pos"], [4, 3, 2])
        self.assertEqual(summary["accepted_at_pos"], [3, 2, 1])
        self.assertEqual(summary["conditional_opportunities_at_pos"], [4, 3, 1])
        self.assertEqual(row["accept_rates_by_position"], [0.75, 2 / 3, 0.5])
        self.assertEqual(
            row["prefix_accept_rates_by_position"],
            row["accept_rates_by_position"],
        )
        self.assertEqual(
            row["conditional_accept_rates_by_position"],
            [0.75, 2 / 3, 1.0],
        )
        self.assertEqual(row["scheduled_block_acceptance_rate"], 0.5)
        self.assertEqual(row["full_block_acceptance_rate"], 0.5)
        self.assertEqual(row["full_block_schedule_rate"], 0.4)
        self.assertEqual(row["full_block_completion_rate"], 0.2)
        self.assertEqual(row["raw_counts"]["proposal_count"], 5)

    def test_all_new_counts_are_aggregated_across_ranks(self):
        evaluator = _make_metrics_evaluator()

        def double_for_second_identical_rank(tensor, op):
            del op
            tensor.mul_(2)

        with mock.patch(
            "deepspec.eval.base_evaluator.dist.all_reduce",
            side_effect=double_for_second_identical_rank,
        ):
            summary = evaluator.allreduce_response_metrics(_metric_responses())

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["proposal_count"], 10)
        self.assertEqual(summary["scheduled_block_count"], 8)
        self.assertEqual(summary["scheduled_block_accepted_count"], 4)
        self.assertEqual(summary["full_block_proposal_count"], 4)
        self.assertEqual(summary["full_block_accepted_count"], 2)
        self.assertEqual(summary["proposals_at_pos"], [8, 6, 4])
        self.assertEqual(summary["accepted_at_pos"], [6, 4, 2])
        self.assertEqual(summary["conditional_opportunities_at_pos"], [8, 6, 2])

    def test_real_ddp_reduction_preserves_all_denominators(self):
        runtime = require_torchrun(self, world_size=2)
        evaluator = _make_metrics_evaluator()
        evaluator.device = runtime.device
        if dist.get_rank() == 0:
            responses = [
                SimpleNamespace(
                    acceptance_lengths=[4],
                    proposal_lengths=[3],
                    accepted_draft_lengths=[3],
                )
            ]
        else:
            responses = [
                SimpleNamespace(
                    acceptance_lengths=[2],
                    proposal_lengths=[3],
                    accepted_draft_lengths=[1],
                )
            ]

        summary = evaluator.allreduce_response_metrics(responses)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["proposal_count"], 2)
        self.assertEqual(summary["acceptance_length_sum"], 6)
        self.assertEqual(summary["proposal_length_sum"], 6)
        self.assertEqual(summary["scheduled_block_count"], 2)
        self.assertEqual(summary["scheduled_block_accepted_count"], 1)
        self.assertEqual(summary["full_block_proposal_count"], 2)
        self.assertEqual(summary["full_block_accepted_count"], 1)
        self.assertEqual(summary["proposals_at_pos"], [2, 2, 2])
        self.assertEqual(summary["accepted_at_pos"], [2, 1, 1])
        self.assertEqual(summary["conditional_opportunities_at_pos"], [2, 2, 1])

    def test_results_json_contains_reproduction_metadata_and_raw_counts(self):
        evaluator = _make_metrics_evaluator()
        evaluator.tasks = [("toy", 12)]
        with mock.patch(
            "deepspec.eval.base_evaluator.dist.all_reduce",
            side_effect=lambda tensor, op: None,
        ):
            summary = evaluator.allreduce_response_metrics(_metric_responses())
        evaluator.metrics_rows = [
            evaluator.build_metrics_row(dataset_name="toy", metric_summary=summary)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            evaluator.args = SimpleNamespace(
                results_json=str(output_path),
                target_name_or_path="target",
                draft_name_or_path="draft",
                temperature=1.0,
                confidence_threshold=0.25,
                max_new_tokens=128,
                seed=17,
            )
            with mock.patch(
                "deepspec.eval.base_evaluator.dist.get_rank",
                return_value=0,
            ):
                evaluator.write_results_json()
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["confidence_threshold"], 0.25)
        self.assertEqual(payload["max_new_tokens"], 128)
        self.assertEqual(payload["max_proposal_tokens"], 3)
        self.assertEqual(payload["seed"], 17)
        self.assertEqual(payload["top_k"], 0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["confidence_scheduler"]["mode"], "static")
        self.assertIsNone(
            payload["confidence_scheduler"]["sequential_temperatures"]
        )
        self.assertEqual(payload["tasks"], [{"dataset": "toy", "max_samples": 12}])
        self.assertEqual(payload["metrics"][0]["raw_counts"]["proposal_count"], 5)
        self.assertIn("conditional_accept_rates_by_position", payload["metric_definitions"])

    def test_selected_tiny_draft_probability_is_not_clamped(self):
        tiny = 1e-12

        class TinyProbabilityModel:
            def __call__(self, *, input_ids, is_prefill):
                del input_ids, is_prefill
                logits = torch.tensor(
                    [[[torch.log(torch.tensor(tiny)), 0.0], [0.0, -100.0]]]
                )
                return SimpleNamespace(logits=logits)

        proposal = DraftProposal(
            draft_token_count=1,
            verify_input_ids=torch.tensor([[1, 0]]),
            draft_probs=torch.tensor([[[tiny, 1.0 - tiny]]]),
        )
        result = verify_draft_tokens(
            target_model=TinyProbabilityModel(),
            proposal=proposal,
            position_ids=torch.arange(4).unsqueeze(0),
            start=1,
            past_key_values_target=object(),
            temperature=1.0,
            max_proposal_tokens=1,
            current_token_ids=torch.tensor([[1]]),
            target_adapter=_FakeTargetAdapter(),
        )

        self.assertEqual(result.accepted_draft_tokens, 1)

    def test_target_sampling_filter_is_used_by_rejection_sampling(self):
        class FilteredTargetModel:
            def __call__(self, *, input_ids, is_prefill):
                del input_ids, is_prefill
                return SimpleNamespace(
                    logits=torch.tensor([[[10.0, 0.0], [10.0, 0.0]]])
                )

        proposal = DraftProposal(
            draft_token_count=1,
            verify_input_ids=torch.tensor([[0, 1]]),
            draft_probs=torch.tensor([[[0.5, 0.5]]]),
        )
        result = verify_draft_tokens(
            target_model=FilteredTargetModel(),
            proposal=proposal,
            position_ids=torch.arange(4).unsqueeze(0),
            start=1,
            past_key_values_target=object(),
            temperature=1.0,
            max_proposal_tokens=1,
            current_token_ids=torch.tensor([[0]]),
            target_adapter=_FakeTargetAdapter(),
            top_k=1,
            top_p=1.0,
        )

        self.assertEqual(result.accepted_draft_tokens, 0)
        self.assertEqual(result.next_token.item(), 0)
        torch.testing.assert_close(
            result.target_probs,
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            rtol=0.0,
            atol=0.0,
        )


class EvalCliTest(unittest.TestCase):
    def test_max_new_tokens_parser_accepts_zero_and_rejects_negative(self):
        eval_cli = importlib.import_module("eval")
        self.assertEqual(eval_cli._non_negative_int("0"), 0)
        with self.assertRaises(argparse.ArgumentTypeError):
            eval_cli._non_negative_int("-1")

    def test_sampling_argument_validation(self):
        eval_cli = importlib.import_module("eval")
        self.assertEqual(eval_cli._non_negative_float("0"), 0.0)
        self.assertEqual(eval_cli._top_p_float("0.95"), 0.95)
        self.assertEqual(eval_cli._closed_unit_float("0"), 0.0)
        self.assertEqual(eval_cli._closed_unit_float("1"), 1.0)
        with self.assertRaises(argparse.ArgumentTypeError):
            eval_cli._non_negative_float("-0.1")
        with self.assertRaises(argparse.ArgumentTypeError):
            eval_cli._top_p_float("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            eval_cli._top_p_float("1.1")
        with self.assertRaises(argparse.ArgumentTypeError):
            eval_cli._closed_unit_float("-0.1")
        with self.assertRaises(argparse.ArgumentTypeError):
            eval_cli._closed_unit_float("1.1")


if __name__ == "__main__":
    unittest.main()

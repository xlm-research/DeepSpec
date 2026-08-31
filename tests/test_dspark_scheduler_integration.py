import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from deepspec.eval.dspark.draft_ops import build_dspark_proposal
from deepspec.eval.dspark.evaluator import (
    Qwen3DSparkEvaluator,
    _load_confidence_scaler,
    _load_sps_profile,
)
from deepspec.eval.dspark.scheduler import (
    HardwareAwarePrefixScheduler,
    SPSProfile,
    SequentialTemperatureScaler,
)


def _write_json(directory: str, filename: str, payload: dict) -> str:
    path = Path(directory) / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _calibration_payload(*, temperature=1.0, top_k=0, top_p=1.0):
    return {
        "schema_version": 1,
        "method": "sequential_temperature_scaling",
        "target_model": "target",
        "draft_model": "draft",
        "sampling": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
        "temperatures": [1.0, 1.5, 2.0],
        "num_bins": 15,
    }


class SchedulerArtifactTest(unittest.TestCase):
    def test_loads_matching_calibration_and_complete_sps_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration_path = _write_json(
                tmpdir,
                "calibration.json",
                _calibration_payload(),
            )
            profile_path = _write_json(
                tmpdir,
                "sps.json",
                {
                    "schema_version": 1,
                    "steps_per_second": {"1": 10, "2": 9, "3": 8, "4": 7},
                },
            )
            scaler = _load_confidence_scaler(
                calibration_path,
                block_size=3,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                target_model="target",
                draft_model="draft",
            )
            profile = _load_sps_profile(
                profile_path,
                minimum_batch_size=1,
                maximum_batch_size=4,
            )

        self.assertEqual(scaler.temperatures.tolist(), [1.0, 1.5, 2.0])
        self.assertEqual(profile.lookup(4), 7.0)

    def test_rejects_calibration_for_a_different_sampling_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_json(
                tmpdir,
                "calibration.json",
                _calibration_payload(temperature=1.0, top_k=20, top_p=0.95),
            )
            with self.assertRaisesRegex(ValueError, "sampling mismatch"):
                _load_confidence_scaler(
                    path,
                    block_size=3,
                    temperature=0.8,
                    top_k=20,
                    top_p=0.95,
                    target_model="target",
                    draft_model="draft",
                )

    def test_static_nondefault_sampling_requires_calibration(self):
        evaluator = object.__new__(Qwen3DSparkEvaluator)
        evaluator.draft_model = SimpleNamespace(
            block_size=3,
            confidence_head=object(),
        )
        evaluator.args = SimpleNamespace(
            scheduler_mode="static",
            confidence_threshold=0.5,
            temperature=0.8,
            top_k=0,
            top_p=1.0,
            confidence_calibration_json=None,
            sps_profile_json=None,
        )

        with self.assertRaisesRegex(ValueError, "requires a matching"):
            evaluator._configure_confidence_scheduling()

    def test_hardware_scheduler_requires_calibration_and_profile(self):
        evaluator = object.__new__(Qwen3DSparkEvaluator)
        evaluator.draft_model = SimpleNamespace(
            block_size=3,
            confidence_head=object(),
        )
        evaluator.args = SimpleNamespace(
            scheduler_mode="hardware-aware",
            confidence_threshold=0.0,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            confidence_calibration_json=None,
            sps_profile_json=None,
        )

        with self.assertRaisesRegex(ValueError, "calibration-json"):
            evaluator._configure_confidence_scheduling()

    def test_evaluator_configures_calibrated_hardware_scheduler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calibration_path = _write_json(
                tmpdir,
                "calibration.json",
                _calibration_payload(),
            )
            profile_path = _write_json(
                tmpdir,
                "sps.json",
                {
                    "schema_version": 1,
                    "steps_per_second": {"1": 1, "2": 1, "3": 0.8, "4": 0.5},
                },
            )
            evaluator = object.__new__(Qwen3DSparkEvaluator)
            evaluator.draft_model = SimpleNamespace(
                block_size=3,
                confidence_head=object(),
            )
            evaluator.args = SimpleNamespace(
                target_name_or_path="target",
                draft_name_or_path="draft",
                scheduler_mode="hardware-aware",
                confidence_threshold=0.0,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                confidence_calibration_json=calibration_path,
                sps_profile_json=profile_path,
                confidence_observations_jsonl=None,
            )
            evaluator._configure_confidence_scheduling()

        self.assertIsNotNone(evaluator.confidence_scaler)
        self.assertIsNotNone(evaluator.prefix_scheduler)
        self.assertEqual(evaluator.args.confidence_temperatures, [1.0, 1.5, 2.0])
        self.assertEqual(evaluator.args.sps_profile["4"], 0.5)


class DraftSchedulerIntegrationTest(unittest.TestCase):
    def test_calibrated_hardware_scheduler_controls_verified_prefix(self):
        confidence_logits = torch.logit(torch.tensor([[0.9, 0.9, 0.1]]))

        class DraftModel:
            proposal_hidden_offset = 0
            confidence_head = object()

            def compute_logits(self, hidden_states):
                return torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    5,
                )

            def sample_draft_tokens(
                self,
                base_logits,
                *,
                first_prev_token_ids,
                temperature,
                hidden_states,
            ):
                del first_prev_token_ids, temperature, hidden_states
                return torch.zeros((1, 3), dtype=torch.long), base_logits

            def predict_confidence_step(
                self,
                hidden_states,
                prev_token_ids,
            ):
                del hidden_states, prev_token_ids
                return confidence_logits

        scaler = SequentialTemperatureScaler(torch.ones(3))
        scheduler = HardwareAwarePrefixScheduler(
            SPSProfile.from_mapping({1: 1.0, 2: 1.0, 3: 0.8, 4: 0.5})
        )
        proposal = build_dspark_proposal(
            DraftModel(),
            draft_input_ids=torch.tensor([[4]]),
            block_hidden=torch.zeros(1, 3, 8),
            block_size=3,
            temperature=0.0,
            confidence_threshold=0.0,
            confidence_scaler=scaler,
            prefix_scheduler=scheduler,
        )

        self.assertEqual(proposal.draft_token_count, 2)
        self.assertEqual(tuple(proposal.verify_input_ids.shape), (1, 3))
        torch.testing.assert_close(
            proposal.confidence_probs,
            torch.tensor([[0.9, 0.9]]),
        )


if __name__ == "__main__":
    unittest.main()

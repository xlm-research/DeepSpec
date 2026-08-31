import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from deepspec.eval.dspark.confidence_head import ConfidenceHeadRecorder
from deepspec.eval.dspark.draft_ops import DSparkDraftProposal
from deepspec.eval.dspark.evaluator import _load_confidence_scaler
from deepspec.eval.dspark.scheduler import SequentialTemperatureScaler
from scripts.eval.fit_dspark_sts import (
    build_calibration_artifact,
    load_observations,
    write_json_atomic,
)


class ConfidenceObservationTest(unittest.TestCase):
    def test_recorder_writes_raw_logits_and_prefix_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.jsonl"
            recorder = ConfidenceHeadRecorder(
                device=torch.device("cpu"),
                max_proposal_tokens=3,
                num_bins=4,
                num_fine_bins=8,
                draft_name_or_path="draft",
                tensorboard_dir=None,
                step=None,
                artifact_root=None,
                observation_path=str(path),
                observation_metadata={
                    "target_model": "target",
                    "draft_model": "draft",
                    "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
                    "block_size": 3,
                },
            )
            recorder.start("validation")
            recorder.observe(
                proposal=DSparkDraftProposal(
                    draft_token_count=2,
                    verify_input_ids=torch.tensor([[3, 4, 5]]),
                    draft_probs=torch.ones(1, 2, 8) / 8,
                    confidence_logits=torch.tensor([[1.0, -2.0]]),
                    confidence_probs=torch.sigmoid(torch.tensor([[1.0, -2.0]])),
                ),
                verification=SimpleNamespace(
                    effective_proposal_length=2,
                    accept_prefix_mask=torch.tensor([[1, 0]]),
                ),
            )
            recorder.close()
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(records[0]["record_type"], "metadata")
        self.assertEqual(records[1]["dataset"], "validation")
        self.assertEqual(records[1]["confidence_logits"], [1.0, -2.0])
        self.assertEqual(records[1]["prefix_targets"], [1, 0])


class FitSequentialTemperatureScalingTest(unittest.TestCase):
    def test_variable_length_observations_fit_and_reload(self):
        rows = [
            {"confidence_logits": [2.0, 1.0, -1.0], "prefix_targets": [1, 1, 0]},
            {"confidence_logits": [1.0, -1.0], "prefix_targets": [1, 0]},
            {"confidence_logits": [-2.0, 0.5, 0.2], "prefix_targets": [0, 0, 0]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            observations_path = Path(tmpdir) / "observations.jsonl"
            observations_path.write_text(
                json.dumps({
                    "record_type": "metadata",
                    "schema_version": 1,
                    "target_model": "target",
                    "draft_model": "draft",
                    "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
                    "block_size": 3,
                }) + "\n" + "".join(
                    json.dumps({"record_type": "observation", **row}) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            logits, targets, valid_mask, sources, metadata = load_observations(
                [observations_path]
            )
            scaler = SequentialTemperatureScaler.fit(
                logits,
                targets,
                temperature_grid=[0.5, 1.0, 2.0],
                num_bins=3,
                valid_mask=valid_mask,
            )
            payload = build_calibration_artifact(
                scaler=scaler,
                confidence_logits=logits,
                prefix_targets=targets,
                valid_mask=valid_mask,
                target_model="target",
                draft_model="draft",
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                sources=sources,
            )
            output_path = write_json_atomic(
                Path(tmpdir) / "calibration.json",
                payload,
            )
            loaded = _load_confidence_scaler(
                str(output_path),
                block_size=3,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                target_model="target",
                draft_model="draft",
            )

        torch.testing.assert_close(loaded.temperatures, scaler.temperatures)
        self.assertEqual(payload["num_observations"], 3)
        self.assertEqual(len(payload["calibrated_prefix_ece"]), 3)
        self.assertEqual(metadata["sampling"]["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()

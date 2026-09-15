"""Retain two complete native recovery points and export loadable HF weights."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from deepspec.modeling.dspark.qwen3_8 import Qwen3_8DSparkModel
from torchtitan.models.dspark_draft.planning import make_input_plan

from tests import test_torchtitan_phase_checkpoint as checkpoint_tests
from tests.test_torchtitan_hf_export import assert_hf_weights


class PhaseRetentionTest(unittest.TestCase):
    assert_state_equal = checkpoint_tests.NativePhaseCheckpointTest.assert_state_equal

    def test_real_retention_export_and_both_recovery_points(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        reference = Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"])
        base = Path(os.environ["DEEPSPEC_NATIVE_TEST_OUTPUT"]).resolve()
        base.mkdir(parents=True, exist_ok=False)
        fixtures = [
            torch.load(reference / f"torch.bfloat16_rank{rank}.pt", weights_only=True)
            for rank in range(2)
        ]
        plan_path = base / "input-plan.json"
        plan_path.write_text(
            json.dumps(
                make_input_plan(
                    run_id="native-dspark-checkpoint-reference",
                    ordered_batches=[
                        (
                            f"batch-{index}-rank{rank}.pt",
                            fixtures[rank]["features"][index % 4],
                        )
                        for index in range(8)
                        for rank in range(2)
                    ],
                ),
                sort_keys=True,
            )
        )
        roots = {}
        for name, start, stop in (
            ("continuous", 0, 4),
            ("first", 0, 3),
            ("from-two", 2, 4),
            ("from-three", 3, 4),
        ):
            root = base / name
            root.mkdir()
            roots[name] = root
            if not start:
                torch.save(fixtures[0]["initial_weights"], root / "initial-weights.pt")
            entries = []
            for index in range(start * 2, stop * 2):
                for rank in range(2):
                    filename = f"batch-{index}-rank{rank}.pt"
                    torch.save(fixtures[rank]["features"][index % 4], root / filename)
                    entries.append({"id": filename, "path": filename})
            (root / "features.json").write_text(json.dumps({"batches": entries}))
            environment = dict(
                os.environ,
                DEEPSPEC_PHASE_TEST_ROOT=str(root),
                DEEPSPEC_PHASE_TEST_DTYPE="bfloat16",
            )
            checkpoint_folder = (
                roots["first"] / "checkpoints"
                if name == "from-three"
                else root / "checkpoints"
            )
            request = {
                "draft_python": sys.executable,
                "draft_source": str(Path("torchtitan").resolve()),
                "workers": 2,
                "result_path": str(root / "phase-result.json"),
                "recipe_args": [
                    "--module",
                    "tests.torchtitan_phase_fixtures",
                    "--config",
                    "retention_features",
                    "--checkpoint.export-dtype",
                    os.environ.get("DEEPSPEC_HF_EXPORT_DTYPE", "float32"),
                ],
                "phase": {
                    "run_id": "native-dspark-checkpoint-reference",
                    "stop_update": stop,
                    "feature_manifest": str(root / "features.json"),
                    "microbatch_start": start * 2,
                    "plan_path": str(plan_path),
                    "checkpoint_folder": str(checkpoint_folder),
                    "resume_checkpoint": str(
                        roots["first"] / f"checkpoints/step-{start}"
                    )
                    if start
                    else None,
                },
            }
            if start:
                environment["DEEPSPEC_PHASE_PERTURB_INITIALIZATION"] = "1"
            (root / "request.json").write_text(json.dumps(request))
            with (root / "native.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "deepspec.orchestration.draft",
                        str(root / "request.json"),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            self.assertEqual(
                result.returncode, 0, (root / "native.log").read_text()[-14000:]
            )
            if name == "first":
                self.assertEqual(
                    sorted(p.name for p in checkpoint_folder.glob("step-*")),
                    ["step-1", "step-2", "step-3"],
                )
                exported = Qwen3_8DSparkModel.from_pretrained(
                    checkpoint_folder / "hf/step-3", dtype=torch.bfloat16
                )
                observed = torch.load(root / "native-rank0.pt", weights_only=True)
                self.assert_state_equal(
                    exported.state_dict(), observed["updates"][-1]["parameters"]
                )
                assert_hf_weights(
                    self,
                    checkpoint_folder / "hf/step-3",
                    observed["updates"][-1]["parameters"],
                    getattr(
                        torch, os.environ.get("DEEPSPEC_HF_EXPORT_DTYPE", "float32")
                    ),
                )
                self.assertFalse((checkpoint_folder / "step-3/config.json").exists())
            if name.startswith("from-"):
                for rank in range(2):
                    expected = torch.load(
                        roots["continuous"] / f"native-rank{rank}.pt", weights_only=True
                    )
                    actual = torch.load(
                        root / f"native-rank{rank}.pt", weights_only=True
                    )
                    expected["updates"] = expected["updates"][start:]
                    expected["microbatches"] = expected["microbatches"][start * 2 :]
                    expected["metrics"]["grad_norm"] = expected["metrics"]["grad_norm"][
                        start:
                    ]
                    self.assert_state_equal(actual, expected)
        self.assertEqual(
            sorted(p.name for p in (roots["first"] / "checkpoints").glob("step-*")),
            ["step-1", "step-3", "step-4"],
        )


if __name__ == "__main__":
    unittest.main()

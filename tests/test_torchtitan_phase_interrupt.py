"""Interrupt real native phases at input and process boundaries, then resume."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

import torch

from deepspec.orchestration.io import require_idle
from deepspec.orchestration.journal import recover_commit
from torchtitan.models.dspark_draft.planning import make_input_plan

from tests import test_torchtitan_phase_checkpoint as checkpoint_tests


class PhaseInterruptionTest(unittest.TestCase):
    assert_state_equal = checkpoint_tests.NativePhaseCheckpointTest.assert_state_equal

    def test_real_process_and_input_failures_preserve_recovery(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        reference = Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"])
        base = Path(os.environ["DEEPSPEC_NATIVE_TEST_OUTPUT"]).resolve()
        base.mkdir(parents=True, exist_ok=False)
        fixtures = [
            torch.load(reference / f"torch.float32_rank{rank}.pt", weights_only=True)
            for rank in range(2)
        ]
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        for mode in os.environ.get(
            "DEEPSPEC_FAILURE_MODES", "before-first,bad-input,parent,worker"
        ).split(","):
            with self.subTest(mode=mode):
                root = base / mode
                root.mkdir()
                plan_path = root / "input-plan.json"
                plan = make_input_plan(
                    run_id="native-dspark-checkpoint-reference",
                    ordered_batches=[
                        (
                            f"batch-{index}-rank{rank}.pt",
                            fixtures[rank]["features"][index],
                        )
                        for index in range(4)
                        for rank in range(2)
                    ],
                )
                plan.update(training_steps=2, gradient_accumulation_steps=2)
                plan_path.write_text(json.dumps(plan, sort_keys=True))
                attempt = root / "failed"
                attempt.mkdir()
                self.write_features(attempt, fixtures, 0)
                blocked_index = 0 if mode == "before-first" else 2
                blocked_path = attempt / f"batch-{blocked_index}-rank0.pt"
                if mode in ("parent", "worker"):
                    blocked_path.unlink()
                    os.mkfifo(blocked_path)
                else:
                    batch = dict(fixtures[0]["features"][blocked_index])
                    batch["input_ids"] = batch["input_ids"] + 1
                    torch.save(batch, blocked_path)
                request = self.request(root, attempt, plan_path, 0, None)
                environment = dict(
                    os.environ,
                    DEEPSPEC_PHASE_TEST_ROOT=str(attempt),
                    DEEPSPEC_PHASE_TEST_DTYPE="float32",
                )
                with (attempt / "native.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "deepspec.orchestration.draft",
                            str(request),
                        ],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    if mode in ("parent", "worker"):
                        deadline = time.monotonic() + 600
                        marker = root / "checkpoints/step-1/commit.json"
                        while not marker.is_file():
                            self.assertIsNone(
                                process.poll(),
                                (attempt / "native.log").read_text()[-8000:],
                            )
                            self.assertLess(time.monotonic(), deadline)
                            time.sleep(0.2)
                        if mode == "parent":
                            os.kill(process.pid, signal.SIGKILL)
                        else:
                            pids = subprocess.run(
                                [
                                    "nvidia-smi",
                                    "-i",
                                    ",".join(devices),
                                    "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits",
                                ],
                                check=True,
                                text=True,
                                capture_output=True,
                            ).stdout.split()
                            self.assertTrue(pids)
                            os.kill(int(pids[0]), signal.SIGKILL)
                    code = process.wait(timeout=600)
                self.assertNotEqual(code, 0)
                if mode in ("before-first", "bad-input"):
                    self.assertIn(
                        "Feature tokens or loss mask differ from the input plan",
                        (attempt / "native.log").read_text(),
                    )
                self.assertFalse((attempt / "phase-result.json").exists())
                require_idle(devices)
                commit = recover_commit(root, plan_path=plan_path, workers=2)
                start = 0 if mode == "before-first" else 1
                self.assertEqual(commit["completed_updates"] if commit else 0, start)
                resumed = root / "resumed"
                resumed.mkdir()
                self.write_features(resumed, fixtures, start)
                request = self.request(
                    root,
                    resumed,
                    plan_path,
                    start,
                    commit["checkpoint"] if commit else None,
                )
                environment.update(DEEPSPEC_PHASE_TEST_ROOT=str(resumed))
                if start:
                    environment["DEEPSPEC_PHASE_PERTURB_INITIALIZATION"] = "1"
                with (resumed / "native.log").open("w") as log:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "deepspec.orchestration.draft",
                            str(request),
                        ],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                self.assertEqual(
                    result.returncode, 0, (resumed / "native.log").read_text()[-12000:]
                )
                require_idle(devices)
                for rank in range(2):
                    expected = torch.load(
                        Path(os.environ["DEEPSPEC_CONTINUOUS_REFERENCE"])
                        / f"native-rank{rank}.pt",
                        weights_only=True,
                    )
                    actual = torch.load(
                        resumed / f"native-rank{rank}.pt", weights_only=True
                    )
                    expected["updates"] = expected["updates"][start:]
                    expected["microbatches"] = expected["microbatches"][start * 2 :]
                    expected["metrics"]["grad_norm"] = expected["metrics"]["grad_norm"][
                        start:
                    ]
                    self.assert_state_equal(actual, expected)

    def write_features(self, root, fixtures, start):
        if not start:
            torch.save(fixtures[0]["initial_weights"], root / "initial-weights.pt")
        entries = []
        for index in range(start * 2, 4):
            for rank in range(2):
                filename = f"batch-{index}-rank{rank}.pt"
                torch.save(fixtures[rank]["features"][index], root / filename)
                entries.append({"id": filename, "path": filename})
        (root / "features.json").write_text(json.dumps({"batches": entries}))

    def request(self, root, attempt, plan, start, resume):
        request = {
            "draft_python": sys.executable,
            "draft_source": str(Path("torchtitan").resolve()),
            "workers": 2,
            "result_path": str(attempt / "phase-result.json"),
            "recipe_args": [
                "--module",
                "tests.torchtitan_phase_fixtures",
                "--config",
                "interruption_features",
            ],
            "phase": {
                "run_id": "native-dspark-checkpoint-reference",
                "stop_update": 2,
                "feature_manifest": str(attempt / "features.json"),
                "microbatch_start": start * 2,
                "plan_path": str(plan),
                "checkpoint_folder": str(root / "checkpoints"),
                "resume_checkpoint": resume,
            },
        }
        path = attempt / "request.json"
        path.write_text(json.dumps(request))
        return path


if __name__ == "__main__":
    unittest.main()

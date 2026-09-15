"""Compare real native updates across complete worker teardown and DCP restore."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


class NativePhaseCheckpointTest(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected, strict=True):
                self.assert_state_equal(a, b)
        elif torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            self.assertEqual(actual, expected)

    def test_process_restart_preserves_the_complete_training_trajectory(self):
        tp = int(os.environ.get("DEEPSPEC_PHASE_TP", "1"))
        workers = int(os.environ.get("DEEPSPEC_PHASE_WORKERS", "2"))
        if torch.cuda.device_count() < workers * tp:
            self.skipTest(f"requires {workers * tp} actual CUDA GPUs")
        reference = Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"])
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="dspark-native-checkpoint-"
        ) as temporary:
            base = Path(os.environ.get("DEEPSPEC_NATIVE_TEST_OUTPUT", temporary))
            for dtype in os.environ.get(
                "DEEPSPEC_CHECKPOINT_DTYPES", "float32,bfloat16"
            ).split(","):
                with self.subTest(dtype=dtype):
                    fixtures = [
                        torch.load(
                            reference / f"torch.{dtype}_rank{rank}.pt",
                            weights_only=True,
                        )
                        for rank in range(workers)
                    ]
                    from torchtitan.models.dspark_draft.planning import make_input_plan

                    plan_path = (base / dtype / "input-plan.json").resolve()
                    plan_path.parent.mkdir(parents=True, exist_ok=True)
                    plan_path.write_text(
                        json.dumps(
                            make_input_plan(
                                run_id="native-dspark-checkpoint-reference",
                                ordered_batches=[
                                    (
                                        f"batch-{index}-rank{rank}.pt",
                                        fixtures[rank]["features"][index],
                                    )
                                    for index in range(4)
                                    for rank in range(workers)
                                ],
                            ),
                            sort_keys=True,
                        )
                    )
                    roots = {}
                    for name, start, stop in (
                        ("continuous", 0, 2),
                        ("first", 0, 1),
                        ("resumed", 1, 2),
                    ):
                        archived = os.environ.get(
                            "DEEPSPEC_CHECKPOINT_CONTINUOUS_REFERENCE"
                        )
                        if name == "continuous" and archived:
                            root = Path(archived).resolve() / dtype
                            for rank in range(workers * tp):
                                self.assertTrue(
                                    (root / f"native-rank{rank}.pt").is_file()
                                )
                            roots[name] = root
                            continue
                        root = (base / dtype / name).resolve()
                        root.mkdir(parents=True, exist_ok=False)
                        roots[name] = root
                        torch.save(
                            fixtures[0]["initial_weights"], root / "initial-weights.pt"
                        )
                        entries = []
                        for index in range(start * 2, stop * 2):
                            for rank in range(workers):
                                filename = f"batch-{index}-rank{rank}.pt"
                                torch.save(
                                    fixtures[rank]["features"][index], root / filename
                                )
                                entries.append({"id": filename, "path": filename})
                        (root / "features.json").write_text(
                            json.dumps({"batches": entries})
                        )
                        environment = os.environ.copy()
                        environment.update(
                            {
                                "DEEPSPEC_PHASE_TEST_ROOT": str(root),
                                "DEEPSPEC_PHASE_TEST_DTYPE": dtype,
                            }
                        )
                        if name == "resumed":
                            environment["DEEPSPEC_PHASE_RESTORE"] = str(
                                roots["first"] / "checkpoints/step-1"
                            )
                            environment["DEEPSPEC_PHASE_PERTURB_INITIALIZATION"] = "1"
                        request = {
                            "draft_python": sys.executable,
                            "draft_source": str(repository / "torchtitan"),
                            "workers": workers * tp,
                            "result_path": str(root / "phase-result.json"),
                            "recipe_args": [
                                "--module",
                                "tests.torchtitan_phase_fixtures",
                                "--config",
                                "checkpoint_features",
                            ],
                            "phase": {
                                "run_id": "native-dspark-checkpoint-reference",
                                "stop_update": stop,
                                "feature_manifest": str(root / "features.json"),
                                "microbatch_start": start * 2,
                                "plan_path": str(plan_path),
                                "checkpoint_folder": str(root / "checkpoints"),
                                "resume_checkpoint": str(
                                    roots["first"] / "checkpoints/step-1"
                                )
                                if name == "resumed"
                                else None,
                            },
                        }
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
                            result.returncode,
                            0,
                            (root / "native.log").read_text()[-14000:],
                        )
                        phase = json.loads((root / "phase-result.json").read_text())
                        self.assertEqual(phase["completed_updates"], stop)
                        if os.environ.get("DEEPSPEC_PHASE_MEASURE"):
                            returned = json.loads(
                                (root / "native.log").read_text().splitlines()[-1]
                            )
                            timing = returned["timing"]
                            self.assertGreaterEqual(timing["launch_seconds"], 0)
                            self.assertGreaterEqual(timing["exit_seconds"], 0)
                            self.assertAlmostEqual(
                                timing["launch_seconds"]
                                + timing["native_seconds"]
                                + timing["exit_seconds"],
                                returned["draft_elapsed_seconds"],
                            )
                            for rank_timing in timing["ranks"]:
                                end = 0.0
                                for event in rank_timing["events"]:
                                    self.assertGreaterEqual(event["start_seconds"], end)
                                    end = event["start_seconds"] + event["seconds"]
                                self.assertLessEqual(end, rank_timing["native_seconds"])
                        processes = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-compute-apps=pid",
                                "--format=csv,noheader,nounits",
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        gpu_pids = {int(pid) for pid in processes.stdout.split()}
                        self.assertFalse(gpu_pids.intersection(phase["worker_pids"]))
                        self.assertEqual(phase["consumed_range"], [start * 2, stop * 2])
                        self.assertTrue(
                            (root / f"checkpoints/step-{stop}/.metadata").is_file()
                        )
                        self.assertFalse(
                            list((root / "checkpoints").rglob("*.safetensors"))
                        )
                    for rank in range(workers * tp):
                        observed = {
                            name: torch.load(
                                root / f"native-rank{rank}.pt", weights_only=True
                            )
                            for name, root in roots.items()
                        }
                        combined = observed["resumed"]
                        for key in ("microbatches", "updates"):
                            combined[key] = observed["first"][key] + combined[key]
                        combined["metrics"]["grad_norm"] = (
                            observed["first"]["metrics"]["grad_norm"]
                            + combined["metrics"]["grad_norm"]
                        )
                        self.assert_state_equal(combined, observed["continuous"])
                        numerical = os.environ.get(
                            "DEEPSPEC_CHECKPOINT_NUMERICAL_REFERENCE"
                        )
                        if numerical:
                            numerical_reference = torch.load(
                                Path(numerical) / dtype / f"native-rank{rank}.pt",
                                weights_only=True,
                            )
                            self.assert_state_equal(
                                observed["continuous"], numerical_reference
                            )


if __name__ == "__main__":
    unittest.main()

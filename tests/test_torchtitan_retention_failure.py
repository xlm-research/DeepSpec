"""A failed fourth commit must retain both previous native recovery points."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from deepspec.orchestration.io import require_idle


class RetentionFailureTest(unittest.TestCase):
    def test_failed_commit_does_not_purge_previous_points(self):
        reference = Path(os.environ["DEEPSPEC_RETENTION_REFERENCE"])
        root = Path(os.environ["DEEPSPEC_NATIVE_TEST_OUTPUT"]).resolve()
        root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(reference / "checkpoints", root / "checkpoints")
        plan = root / "input-plan.json"
        shutil.copyfile(reference.parent / "input-plan.json", plan)
        entries = []
        for index in (6, 7):
            for rank in range(2):
                name = f"batch-{index}-rank{rank}.pt"
                shutil.copyfile(reference / name, root / name)
                entries.append({"id": name, "path": name})
        (root / "features.json").write_text(json.dumps({"batches": entries}))
        previous = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "checkpoints").glob("step-*/commit.json")
        }
        self.assertEqual(len(previous), 3)
        marker = root / "checkpoints/step-4/commit.json"
        marker.mkdir(parents=True)
        request = json.loads((reference / "request.json").read_text())
        request["result_path"] = str(root / "phase-result.json")
        request["phase"].update(
            stop_update=4,
            feature_manifest=str(root / "features.json"),
            microbatch_start=6,
            plan_path=str(plan),
            checkpoint_folder=str(root / "checkpoints"),
            resume_checkpoint=str(root / "checkpoints/step-3"),
        )
        path = root / "request.json"
        path.write_text(json.dumps(request))
        environment = dict(
            os.environ,
            DEEPSPEC_PHASE_TEST_ROOT=str(root),
            DEEPSPEC_PHASE_TEST_DTYPE="bfloat16",
            DEEPSPEC_PHASE_PERTURB_INITIALIZATION="1",
        )
        with (root / "native.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "-m", "deepspec.orchestration.draft", str(path)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("IsADirectoryError", (root / "native.log").read_text())
        self.assertFalse((root / "phase-result.json").exists())
        self.assertTrue((marker.parent / ".metadata").is_file())
        for path, expected in previous.items():
            self.assertEqual(
                hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected
            )
        require_idle(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


if __name__ == "__main__":
    unittest.main()

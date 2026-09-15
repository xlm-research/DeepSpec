"""A real filesystem commit failure must prevent a successful phase handoff."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch


class NativePhaseFailureTest(unittest.TestCase):
    def test_failed_commit_exits_without_publishing_success(self):
        reference = Path(os.environ["DEEPSPEC_PHASE_CHECKPOINT_REFERENCE"])
        original_request = json.loads((reference / "request.json").read_text())
        if torch.cuda.device_count() < original_request["workers"]:
            self.skipTest(f"requires {original_request['workers']} actual CUDA GPUs")
        previous = reference / "checkpoints/step-1/commit.json"
        digest = hashlib.sha256(previous.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="dspark-commit-failure-") as temporary:
            root = Path(
                os.environ.get("DEEPSPEC_NATIVE_TEST_OUTPUT", temporary)
            ).resolve()
            root.mkdir(parents=True, exist_ok=True)
            for source in reference.iterdir():
                if source.suffix == ".pt" and not source.name.startswith("native-rank"):
                    shutil.copyfile(source, root / source.name)
            shutil.copyfile(reference / "features.json", root / "features.json")
            request = original_request
            request["result_path"] = str(root / "phase-result.json")
            request["phase"]["feature_manifest"] = str(root / "features.json")
            request["phase"]["checkpoint_folder"] = str(root / "checkpoints")
            (root / "request.json").write_text(json.dumps(request))
            # DCP can write real tensor shards and metadata, but cannot replace
            # this directory with its final commit file.
            marker = root / "checkpoints/step-1/commit.json"
            marker.mkdir(parents=True, exist_ok=False)
            environment = os.environ.copy()
            environment.update(
                {
                    "DEEPSPEC_PHASE_TEST_ROOT": str(root),
                    "DEEPSPEC_PHASE_TEST_DTYPE": "float32",
                }
            )
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
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("IsADirectoryError", (root / "native.log").read_text())
            self.assertTrue((marker.parent / ".metadata").is_file())
            self.assertFalse(marker.is_file())
            self.assertFalse((root / "phase-result.json").exists())
            self.assertEqual(hashlib.sha256(previous.read_bytes()).hexdigest(), digest)
            processes = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    os.environ["CUDA_VISIBLE_DEVICES"],
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(processes.stdout.strip(), processes.stdout)


if __name__ == "__main__":
    unittest.main()

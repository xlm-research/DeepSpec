"""Repair a damaged GPU phase HF export through the DeepSpec recovery entry."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch

from deepspec.orchestration.journal import finish_pending_exports
from tests.test_torchtitan_hf_export import assert_hf_weights
from torchtitan.models.dspark_draft.export import weights_complete


def checkpoint_hashes(folder):
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in folder.glob("step-*/*")
        if path.is_file()
    }


class PendingHFExportTest(unittest.TestCase):
    def test_missing_distributed_shard_is_repaired_and_loadable(self):
        reference = os.environ.get("DEEPSPEC_RETENTION_REFERENCE")
        if not reference:
            raise unittest.SkipTest("requires completed native retention checkpoints")
        source = Path(reference)
        with tempfile.TemporaryDirectory(prefix="dspark-export-repair-") as temporary:
            root = Path(os.environ.get("DEEPSPEC_NATIVE_TEST_OUTPUT", temporary))
            folder = root / "checkpoints"
            shutil.copytree(source / "checkpoints", folder)
            before = checkpoint_hashes(folder)
            output = folder / "hf/step-3"
            self.assertTrue(weights_complete(output))
            index = output / "model.safetensors.index.json"
            self.assertTrue(index.is_file())
            name = next(iter(json.loads(index.read_text())["weight_map"].values()))
            (output / name).rename(root / f"retained-{name}")
            self.assertFalse(weights_complete(output))
            commit = json.loads((folder / "step-4/commit.json").read_text())
            # Relocate the caller's copy, preserving all original checkpoint files.
            commit["checkpoint"] = str(folder / "step-4")
            request = {
                "draft_python": sys.executable,
                "draft_source": str(Path("torchtitan").resolve()),
            }
            finish_pending_exports(request, commit)
            self.assertTrue(weights_complete(output))
            expected = torch.load(source / "native-rank0.pt", weights_only=True)[
                "updates"
            ][-1]["parameters"]
            dtype = commit["resolved_recipe"]["checkpoint"]["export_dtype"]
            assert_hf_weights(self, output, expected, getattr(torch, dtype))
            timestamps = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
            finish_pending_exports(request, commit)
            self.assertEqual(
                timestamps,
                {path.name: path.stat().st_mtime_ns for path in output.iterdir()},
            )
            self.assertEqual(before, checkpoint_hashes(folder))
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()

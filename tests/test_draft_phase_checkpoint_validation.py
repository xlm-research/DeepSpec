"""Validate real saved DCP artifacts through the public checkpoint boundary."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import torch.distributed as dist

from deepspec.trainer.draft_phase_checkpoint import (
    discover_draft_phase_checkpoint,
    validate_draft_phase_checkpoint,
)


class DraftPhaseCheckpointValidationTest(unittest.TestCase):
    rendezvous: tempfile.TemporaryDirectory

    @classmethod
    def setUpClass(cls):
        cls.rendezvous = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "gloo",
            init_method=f"file://{cls.rendezvous.name}/rendezvous",
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.rendezvous.cleanup()

    def setUp(self):
        source = os.environ.get("DEEPSPEC_CHECKPOINT_REFERENCE")
        if not source:
            self.skipTest("requires a real completed Qwen phase checkpoint")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "step_1"
        shutil.copytree(source, self.checkpoint)

    def rewrite(self, name, mutate):
        path = self.checkpoint / name
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))
        commit_path = self.checkpoint / "draft_phase_commit.json"
        commit = json.loads(commit_path.read_text())
        commit["files"][name] = {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        commit_path.write_text(json.dumps(commit))

    def test_missing_storage_cannot_be_hidden_by_omitting_manifest_entry(self):
        commit_path = self.checkpoint / "draft_phase_commit.json"
        commit = json.loads(commit_path.read_text())
        name = next(name for name in commit["files"] if name.endswith(".distcp"))
        del commit["files"][name]
        commit_path.write_text(json.dumps(commit))
        (self.checkpoint / name).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            validate_draft_phase_checkpoint(self.checkpoint)

    def test_phase_progress_must_agree_with_the_saved_input_index(self):
        self.rewrite(
            "distributed_checkpoint_metadata.json",
            lambda value: value.update(next_micro_step=4),
        )
        with self.assertRaisesRegex(ValueError, "progress"):
            validate_draft_phase_checkpoint(self.checkpoint)

    def test_frozen_weights_file_corruption_is_rejected(self):
        storage = next((self.checkpoint / "distributed_checkpoint").glob("*.distcp"))
        with storage.open("r+b") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_draft_phase_checkpoint(self.checkpoint)

    def test_discovery_uses_previous_complete_state_after_a_damaged_write(self):
        root = self.checkpoint.parent
        damaged = root / "step_2"
        shutil.copytree(self.checkpoint, damaged)
        next((damaged / "distributed_checkpoint").glob("*.distcp")).unlink()
        (root / "step_latest").symlink_to("step_2", target_is_directory=True)
        (root / ".step_3.incomplete-test").mkdir()
        (root / "step_3").mkdir()
        self.assertEqual(discover_draft_phase_checkpoint(root), str(self.checkpoint))

    def test_first_incomplete_attempt_is_not_a_resume_point(self):
        (self.checkpoint / "draft_phase_commit.json").unlink()
        (self.checkpoint.parent / "step_latest").symlink_to(
            "step_1", target_is_directory=True
        )
        self.assertIsNone(discover_draft_phase_checkpoint(self.checkpoint.parent))

    def test_damaged_committed_state_is_not_silently_restarted_from_scratch(self):
        next((self.checkpoint / "distributed_checkpoint").glob("*.distcp")).unlink()
        with self.assertRaisesRegex(RuntimeError, "No valid committed"):
            discover_draft_phase_checkpoint(self.checkpoint.parent)


if __name__ == "__main__":
    unittest.main()

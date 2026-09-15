"""Validate the prefetched cursor recorded by the real scale forward hook."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest

import torch

from tests.summarize_torchtitan_scale import supervision_at, training_memory_at


class ScaleSupervisionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def phase(self, start, stop, cursors=None) -> dict[str, Any]:
        root = self.root / f"phase-{start}-{stop}"
        root.mkdir()
        workers = list(range(100, 108))
        (root / "draft-result.json").write_text(json.dumps({"worker_pids": workers}))
        if cursors is None:
            cursors = [2 * (index // 2 + 1) for index in range(start, stop)]
        for rank in range(8):
            torch.save(
                [
                    {
                        "next_microbatch": cursor,
                        "target_ids": torch.tensor([rank, start + index]),
                        "cpu_rng": torch.tensor([index], dtype=torch.uint8),
                    }
                    for index, cursor in enumerate(cursors)
                ],
                root / f"supervision-rank{rank}.pt",
            )
        return {
            "draft": {
                "consumed_range": [start, stop],
                "worker_pids": workers,
                "commit": {
                    "resolved_recipe": {
                        "dataloader": {"manifest": str(root / "features.json")}
                    }
                },
            }
        }

    def test_phased_and_continuous_have_the_same_forward_order(self):
        phased = supervision_at(self.root, [self.phase(0, 10), self.phase(10, 20)])
        continuous = supervision_at(self.root, [self.phase(0, 20)])
        for rank in range(8):
            self.assertEqual(
                [v["next_microbatch"] for v in phased[rank]], list(range(1, 21))
            )
            self.assertEqual(
                [v["next_microbatch"] for v in continuous[rank]], list(range(1, 21))
            )
            self.assertEqual(
                [v["target_ids"].tolist() for v in phased[rank]],
                [[rank, i] for i in range(20)],
            )
        # Normalization must not rewrite the original evidence.
        original = torch.load(
            self.root / "phase-0-10/supervision-rank0.pt", weights_only=True
        )
        self.assertEqual(
            [v["next_microbatch"] for v in original], [2, 2, 4, 4, 6, 6, 8, 8, 10, 10]
        )
        self.assertTrue(torch.equal(phased[0][0]["cpu_rng"], original[0]["cpu_rng"]))

    def test_incorrect_prefetch_position_is_rejected(self):
        phase = self.phase(
            0, 20, cursors=[2, 4] + [2 * (i // 2 + 1) for i in range(2, 20)]
        )
        with self.assertRaisesRegex(ValueError, "cursor"):
            supervision_at(self.root, [phase])

    def test_missing_forward_is_rejected(self):
        phase = self.phase(0, 20, cursors=[2 * (i // 2 + 1) for i in range(19)])
        with self.assertRaises(ValueError):
            supervision_at(self.root, [phase])

    def test_repeated_forward_is_rejected(self):
        phase = self.phase(0, 20, cursors=[2] + [2 * (i // 2 + 1) for i in range(20)])
        with self.assertRaises(ValueError):
            supervision_at(self.root, [phase])

    def test_phase_gap_is_rejected(self):
        with self.assertRaises(ValueError):
            supervision_at(self.root, [self.phase(0, 10), self.phase(12, 20)])

    def test_wrong_worker_evidence_is_rejected(self):
        phase = self.phase(0, 20)
        phase["draft"]["worker_pids"][0] = 999
        with self.assertRaisesRegex(ValueError, "different phase"):
            supervision_at(self.root, [phase])

    def memory_phase(self, missing_last_step=False):
        from torch.utils.tensorboard import SummaryWriter

        phase = self.phase(0, 20)
        recipe = phase["draft"]["commit"]["resolved_recipe"]
        recipe.update(
            {
                "dump_folder": str(self.root),
                "metrics": {
                    "enable_tensorboard": True,
                    "log_freq": 1,
                    "save_tb_folder": "tb",
                    "save_for_all_ranks": True,
                },
            }
        )
        for rank, pid in enumerate(phase["draft"]["worker_pids"]):
            directory = self.root / "tb" / f"rank_{rank}"
            with SummaryWriter(str(directory)) as writer:
                for step in range(1, 10 if missing_last_step and rank == 7 else 11):
                    writer.add_scalar("memory/max_active(GiB)", 20 + rank + step, step)
                    writer.add_scalar("memory/max_reserved(GiB)", 40 + rank, step)
                    writer.add_scalar("memory/num_alloc_retries", 0, step)
                    writer.add_scalar("memory/num_ooms", 0, step)
            path = next(directory.glob("events.out*"))
            path.rename(
                path.with_name(path.name.replace(f".{os.getpid()}.", f".{pid}."))
            )
        return phase

    def test_memory_uses_every_update_and_actual_worker_file(self):
        report = training_memory_at(self.memory_phase())
        self.assertTrue(report["all_ranks_recorded"])
        self.assertEqual([rank["rank"] for rank in report["ranks"]], list(range(8)))
        self.assertEqual(report["peak_active_gib"], 37)
        self.assertEqual(report["peak_reserved_gib"], 47)

    def test_incomplete_memory_intervals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "every update"):
            training_memory_at(self.memory_phase(missing_last_step=True))


if __name__ == "__main__":
    unittest.main()

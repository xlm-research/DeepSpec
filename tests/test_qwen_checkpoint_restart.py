"""A fresh process restores a Qwen phase using its committed DCP alone."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from tests import test_dspark_training_baseline as baseline


class QwenCheckpointRestartTest(unittest.TestCase):
    @unittest.skipIf(
        "LOCAL_RANK" in os.environ, "launch this orchestration test outside torchrun"
    )
    def test_fresh_process_matches_continuous_training(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA devices")
        if not os.environ.get("DEEPSPEC_BASELINE_REFERENCE"):
            self.skipTest("requires the captured immutable Qwen fixture")
        output = os.environ.get("DEEPSPEC_PHASE_CHECKPOINT_OUTPUT")
        temporary = tempfile.TemporaryDirectory() if output is None else None
        if output is None:
            assert temporary is not None
            root = Path(temporary.name)
        else:
            root = Path(output)
            root.mkdir(parents=True, exist_ok=False)
        try:
            for mode in ("continuous", "save", "resume"):
                directory = root / ("baseline" if mode == "continuous" else "restart")
                environment = dict(
                    os.environ,
                    DEEPSPEC_PHASE_TEST_MODE=mode,
                    DEEPSPEC_PHASE_TEST_ROOT=str(directory),
                    OMP_NUM_THREADS="1",
                )
                if mode == "resume":
                    # Move only the complete checkpoint to a fresh storage root.
                    # No producer sidecars or objects from the prior job survive.
                    isolated = root / "isolated_checkpoints"
                    shutil.copytree(
                        directory / "checkpoints" / "step_1", isolated / "step_1"
                    )
                    (isolated / "step_latest").symlink_to(
                        "step_1", target_is_directory=True
                    )
                    # An interrupted later save must not prevent a recovered
                    # phase from committing its recomputed update.
                    shutil.copytree(isolated / "step_1", isolated / "step_2")
                    storage = next(
                        (isolated / "step_2" / "distributed_checkpoint").glob(
                            "*.distcp"
                        )
                    )
                    storage.write_bytes(b"incomplete later write")
                    environment["DEEPSPEC_PHASE_CHECKPOINT_DIR"] = str(isolated)
                log = root / f"{mode}.log"
                print(f"Qwen checkpoint fresh process: {mode}; log={log}", flush=True)
                with log.open("w") as stream:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "torch.distributed.run",
                            "--standalone",
                            "--nproc-per-node=2",
                            "-m",
                            "unittest",
                            "tests.test_qwen_phase_checkpoint.QwenDraftPhaseCheckpointTest",
                        ],
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=900,
                    )
                self.assertEqual(result.returncode, 0, log.read_text()[-16000:])
            comparison = baseline.DSparkTrainingBaselineTest()
            for rank in range(2):
                expected = torch.load(
                    root / "baseline" / f"continuous_rank{rank}.pt", weights_only=True
                )
                before = torch.load(
                    root / "restart" / f"save_rank{rank}.pt", weights_only=True
                )
                after = torch.load(
                    root / "restart" / f"resume_rank{rank}.pt", weights_only=True
                )
                for name in ("observations", "gradients_after_clip"):
                    comparison.assert_state_close(
                        before[name] + after[name], expected[name]
                    )
                for name in (
                    "model",
                    "optimizer",
                    "next_micro_step",
                    "cpu_rng",
                    "cuda_rng",
                    "python_rng",
                    "numpy_rng",
                ):
                    comparison.assert_state_close(after[name], expected[name])
        finally:
            if temporary:
                temporary.cleanup()


if __name__ == "__main__":
    unittest.main()

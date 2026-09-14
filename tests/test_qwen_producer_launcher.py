"""Changing draft degrees keeps the existing vLLM producer binding."""

import unittest

from tests import test_qwen38_multinode_launcher as launchers


class QwenProducerLauncherTest(unittest.TestCase):
    def test_draft_layout_does_not_change_producer_layout(self):
        launcher = (
            launchers.REPO_ROOT / "scripts/train/train_qwen3_8_27b_dspark_vllm.sh"
        )
        for cp, tp, fsdp in ((1, 1, 8), (2, 2, 2)):
            with self.subTest(cp=cp, tp=tp):
                result = launchers.Qwen38MultiNodeLauncherTest()._run_launcher(
                    launcher=launcher,
                    CONTEXT_PARALLEL_SIZE=cp,
                    TENSOR_PARALLEL_SIZE=tp,
                    FSDP_SIZE=fsdp,
                    VLLM_PYTHON_BIN="/existing/producer/bin/python",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for name, value in (
                    ("dp_replicate", 2),
                    ("dp_shard", 2),
                    ("cp", 1),
                    ("tp", 4),
                ):
                    self.assertIn(
                        f"train.offline_target_parallel.{name}={value}", result.stdout
                    )
                self.assertIn(f"train.parallel.cp={cp}", result.stdout)
                self.assertIn(f"train.parallel.tp={tp}", result.stdout)
                self.assertIn(
                    "vLLM interpreter=/existing/producer/bin/python", result.stdout
                )


if __name__ == "__main__":
    unittest.main()

"""Check requested HF precision against an actual committed native update."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import load_file
import torch

from deepspec.modeling.dspark.qwen3_8 import Qwen3_8DSparkModel
from torchtitan.models.dspark_draft.export import export_checkpoint


def assert_hf_weights(test, output, expected, dtype):
    config = json.loads((output / "config.json").read_text())
    test.assertEqual(config["dtype"], str(dtype).removeprefix("torch."))
    serialized = {}
    for path in output.glob("*.safetensors"):
        serialized.update(load_file(path))
    test.assertEqual(serialized.keys(), expected.keys())
    for name, value in serialized.items():
        test.assertEqual(value.dtype, dtype)
        torch.testing.assert_close(value, expected[name].to(dtype), rtol=0, atol=0)
    consumer = Qwen3_8DSparkModel.from_pretrained(output, dtype=dtype)
    for name, value in consumer.state_dict().items():
        torch.testing.assert_close(value, expected[name].to(dtype), rtol=0, atol=0)


class NativeHFExportTest(unittest.TestCase):
    def test_cpu_export_preserves_weights_and_requested_precision(self):
        reference = Path(os.environ["DEEPSPEC_PHASE_CHECKPOINT_REFERENCE"])
        phase = json.loads((reference / "phase-result.json").read_text())
        checkpoint = Path(phase["commit"]["checkpoint"])
        dtype_name = phase["commit"]["resolved_recipe"]["checkpoint"]["export_dtype"]
        expected = torch.load(reference / "native-rank0.pt", weights_only=True)[
            "updates"
        ][-1]["parameters"]
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in checkpoint.iterdir()
            if path.is_file()
        }
        with tempfile.TemporaryDirectory(prefix="dspark-hf-export-") as temporary:
            output = Path(os.environ.get("DEEPSPEC_HF_EXPORT_OUTPUT", temporary)) / "hf"
            self.assertFalse(output.exists())
            result = export_checkpoint(str(checkpoint), str(output))
            self.assertEqual(result["export_dtype"], dtype_name)
            assert_hf_weights(self, output, expected, getattr(torch, dtype_name))
            timestamps = {
                path.name: path.stat().st_mtime_ns for path in output.iterdir()
            }
            self.assertEqual(export_checkpoint(str(checkpoint), str(output)), result)
            self.assertEqual(
                timestamps,
                {path.name: path.stat().st_mtime_ns for path in output.iterdir()},
            )
        self.assertEqual(
            before,
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in checkpoint.iterdir()
                if path.is_file()
            },
        )
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()

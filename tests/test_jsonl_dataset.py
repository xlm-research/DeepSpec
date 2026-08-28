import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepspec.data.jsonl_dataset import JsonLineDataset


class JsonLineDatasetTest(unittest.TestCase):
    def test_explicit_cache_dir_is_reused_without_rescanning_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "data.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps({"conversations": [{"role": "user", "content": value}]})
                    for value in ("one", "two")
                )
                + "\n",
                encoding="utf-8",
            )
            cache_dir = root / "shared-index"

            dataset = JsonLineDataset([str(source)], cache_dir=str(cache_dir))
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["conversations"][0]["content"], "two")
            dataset.close()
            self.assertEqual(len(list(cache_dir.glob("jsonlindex-*.pkl"))), 1)

            with patch(
                "deepspec.data.jsonl_dataset.mmap.mmap",
                side_effect=AssertionError("source was rescanned"),
            ):
                reused = JsonLineDataset([str(source)], cache_dir=str(cache_dir))
            self.assertEqual(len(reused), 2)
            reused.close()


if __name__ == "__main__":
    unittest.main()

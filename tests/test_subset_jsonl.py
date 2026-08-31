from pathlib import Path
import tempfile
import unittest

from scripts.data.subset_jsonl import build_jsonl_subset


class SubsetJsonlTest(unittest.TestCase):
    def test_copies_first_non_empty_records_atomically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.jsonl"
            output = root / "subset.jsonl"
            source.write_bytes(b'{"id":1}\n\n{"id":2}\r\n{"id":3}\n')

            copied = build_jsonl_subset(
                input_path=source,
                output_path=output,
                num_records=2,
            )

            self.assertEqual(copied, 2)
            self.assertEqual(output.read_bytes(), b'{"id":1}\n{"id":2}\n')

    def test_short_input_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.jsonl"
            output = root / "subset.jsonl"
            source.write_text('{"id":1}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "only contains 1"):
                build_jsonl_subset(
                    input_path=source,
                    output_path=output,
                    num_records=2,
                )

            self.assertFalse(output.exists())

    def test_rejects_short_packed_record(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.jsonl"
            output = root / "subset.jsonl"
            source.write_text(
                '{"packed_conversations":[[]],"packed_untruncated_tokens":7}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "at least 8"):
                build_jsonl_subset(
                    input_path=source,
                    output_path=output,
                    num_records=1,
                    minimum_packed_tokens=8,
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

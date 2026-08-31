#!/usr/bin/env python3
"""Copy the first N non-empty records from a JSONL file atomically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_jsonl_subset(
    *,
    input_path: Path,
    output_path: Path,
    num_records: int,
    minimum_packed_tokens: int | None = None,
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    num_records = int(num_records)
    if num_records < 1:
        raise ValueError("num_records must be positive.")
    if minimum_packed_tokens is not None and int(minimum_packed_tokens) < 1:
        raise ValueError("minimum_packed_tokens must be positive when supplied.")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    copied = 0
    try:
        with input_path.open("rb") as source, tmp_path.open("wb") as target:
            for raw_line in source:
                raw_line = raw_line.rstrip(b"\r\n")
                if not raw_line:
                    continue
                if minimum_packed_tokens is not None:
                    record = json.loads(raw_line)
                    conversations = record.get("packed_conversations")
                    packed_tokens = int(record.get("packed_untruncated_tokens", 0))
                    if not isinstance(conversations, list) or not conversations:
                        raise ValueError(
                            "minimum_packed_tokens requires non-empty "
                            "packed_conversations records."
                        )
                    if packed_tokens < int(minimum_packed_tokens):
                        raise ValueError(
                            f"Packed record {copied + 1} has {packed_tokens} tokens; "
                            f"at least {minimum_packed_tokens} are required."
                        )
                target.write(raw_line)
                target.write(b"\n")
                copied += 1
                if copied == num_records:
                    break
            if copied != num_records:
                raise ValueError(
                    f"Requested {num_records} records, but {input_path} only "
                    f"contains {copied} non-empty records."
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(tmp_path, output_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return copied


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-records", type=int, required=True)
    parser.add_argument("--minimum-packed-tokens", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    copied = build_jsonl_subset(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        num_records=args.num_records,
        minimum_packed_tokens=args.minimum_packed_tokens,
    )
    print(f"Wrote {copied} JSONL records to {Path(args.output_path).resolve()}.")


if __name__ == "__main__":
    main()

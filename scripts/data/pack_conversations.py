#!/usr/bin/env python3
"""Pack independently rendered conversations into fixed-length records."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from deepspec.data.parser import preprocess_record
from deepspec.data.target_cache_dataset import (
    atomic_json_dump,
    compute_file_fingerprint,
)


warnings.filterwarnings(
    "ignore",
    message="System prompt from the sample overrides the registered template.*",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--target-length", type=int, default=262144)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing packed dataset.",
    )
    return parser.parse_args()


def _load_record(raw_line: bytes, *, input_path: Path, line_number: int):
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at {input_path}:{line_number}."
        ) from exc
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(
            f"Missing non-empty conversations at {input_path}:{line_number}."
        )
    return record


def _measure_unique_records(
    *,
    input_path: Path,
    tokenizer,
    chat_template: str,
    target_length: int,
):
    length_by_line = {}
    record_count = 0
    started_at = time.monotonic()
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.rstrip(b"\r\n")
            if not raw_line:
                continue
            record_count += 1
            if raw_line in length_by_line:
                continue
            record = _load_record(
                raw_line,
                input_path=input_path,
                line_number=line_number,
            )
            processed = preprocess_record(
                record=record,
                tokenizer=tokenizer,
                chat_template=chat_template,
                max_length=target_length + 1,
            )
            token_length = int(processed["input_ids"].shape[0])
            if token_length > target_length:
                raise ValueError(
                    "An individual source record exceeds the packing target: "
                    f"line={line_number}, tokens={token_length}, "
                    f"target={target_length}."
                )
            length_by_line[raw_line] = token_length
            if len(length_by_line) % 100 == 0:
                elapsed = time.monotonic() - started_at
                print(
                    "Measured "
                    f"{len(length_by_line)} unique records in {elapsed:.1f}s.",
                    flush=True,
                )
    if record_count == 0:
        raise ValueError(f"Input dataset is empty: {input_path}")
    return length_by_line, record_count


def _iter_source_records(input_path: Path, length_by_line):
    with input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.rstrip(b"\r\n")
            if not raw_line:
                continue
            record = _load_record(
                raw_line,
                input_path=input_path,
                line_number=line_number,
            )
            yield record["conversations"], int(length_by_line[raw_line])


def _write_packed_record(
    handle,
    *,
    conversations,
    untruncated_tokens: int,
    target_length: int,
):
    payload = {
        "packed_conversations": conversations,
        "packed_source_count": len(conversations),
        "packed_untruncated_tokens": int(untruncated_tokens),
        "packed_target_length": int(target_length),
    }
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def build_packed_dataset(
    *,
    input_path: Path,
    output_path: Path,
    tokenizer,
    model_name_or_path: str,
    chat_template: str,
    target_length: int,
):
    source_fingerprint_before = compute_file_fingerprint(str(input_path))
    length_by_line, source_record_count = _measure_unique_records(
        input_path=input_path,
        tokenizer=tokenizer,
        chat_template=chat_template,
        target_length=target_length,
    )
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packed_count = 0
    consumed_source_records = 0
    packed_conversations = []
    packed_tokens = 0

    try:
        with tmp_path.open("w", encoding="utf-8") as output_handle:
            for conversations, token_length in _iter_source_records(
                input_path,
                length_by_line,
            ):
                packed_conversations.append(conversations)
                packed_tokens += token_length
                consumed_source_records += 1
                if packed_tokens < target_length:
                    continue
                _write_packed_record(
                    output_handle,
                    conversations=packed_conversations,
                    untruncated_tokens=packed_tokens,
                    target_length=target_length,
                )
                packed_count += 1
                packed_conversations = []
                packed_tokens = 0

            # Repeat records from the start so the final output record also
            # reaches target_length instead of leaving a short tail.
            if packed_conversations:
                while packed_tokens < target_length:
                    for conversations, token_length in _iter_source_records(
                        input_path,
                        length_by_line,
                    ):
                        packed_conversations.append(conversations)
                        packed_tokens += token_length
                        if packed_tokens >= target_length:
                            break
                _write_packed_record(
                    output_handle,
                    conversations=packed_conversations,
                    untruncated_tokens=packed_tokens,
                    target_length=target_length,
                )
                packed_count += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(tmp_path, output_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    source_fingerprint_after = compute_file_fingerprint(str(input_path))
    if source_fingerprint_after != source_fingerprint_before:
        raise RuntimeError(
            "Source data changed while it was being packed; discard the output "
            "and retry from a stable input file."
        )
    summary = {
        "format_version": 1,
        "input_path": str(input_path),
        "input_fingerprint": source_fingerprint_after,
        "source_records": source_record_count,
        "unique_source_records": len(length_by_line),
        "consumed_source_records": consumed_source_records,
        "packed_records": packed_count,
        "target_length": target_length,
        "model_name_or_path": str(model_name_or_path),
        "chat_template": str(chat_template),
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_fingerprint": compute_file_fingerprint(str(output_path)),
    }
    atomic_json_dump(summary, f"{output_path}.manifest.json")
    return summary


def main():
    args = parse_args()
    if int(args.target_length) <= 0:
        raise ValueError("--target-length must be positive.")
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --force to replace it."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    summary = build_packed_dataset(
        input_path=input_path,
        output_path=output_path,
        tokenizer=tokenizer,
        model_name_or_path=args.model_name_or_path,
        chat_template=args.chat_template,
        target_length=int(args.target_length),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

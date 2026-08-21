#!/usr/bin/env python3
"""Validate provenance before reusing an existing packed JSONL artifact."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepspec.data.target_cache_dataset import compute_file_fingerprint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--packed-path", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--target-length", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    packed_path = Path(args.packed_path).expanduser().resolve()
    manifest_path = Path(f"{packed_path}.manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Packed provenance manifest is missing: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert int(manifest.get("format_version", -1)) == 1
    assert str(manifest.get("model_name_or_path")) == str(args.model_name_or_path)
    assert str(manifest.get("chat_template")) == str(args.chat_template)
    assert int(manifest.get("target_length", -1)) == int(args.target_length)
    assert manifest.get("input_fingerprint") == compute_file_fingerprint(
        args.source_path
    ), "Packed source content has changed."
    assert manifest.get("output_fingerprint") == compute_file_fingerprint(
        str(packed_path)
    ), "Packed JSONL content has changed."
    print(f"Packed dataset provenance is valid: {packed_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Packed dataset validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

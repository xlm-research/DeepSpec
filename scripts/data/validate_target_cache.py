#!/usr/bin/env python3
"""Validate that a reusable target cache belongs to the current training run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepspec.data.target_cache_dataset import validate_target_cache_identity


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--source-jsonl-path", action="append", required=True)
    parser.add_argument("--target-model-name-or-path", required=True)
    parser.add_argument("--target-layer-ids", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--context-parallel-size", type=int, required=True)
    parser.add_argument(
        "--stores-target-last-hidden-states",
        type=_parse_bool,
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    layer_ids = [
        int(value) for value in str(args.target_layer_ids).split(",") if value
    ]
    validate_target_cache_identity(
        cache_dir=args.cache_dir,
        source_jsonl_paths=args.source_jsonl_path,
        target_model_name_or_path=args.target_model_name_or_path,
        target_layer_ids=layer_ids,
        chat_template=args.chat_template,
        max_length=args.max_length,
        context_parallel_size=args.context_parallel_size,
        stores_target_last_hidden_states=(
            args.stores_target_last_hidden_states
        ),
    )
    print(f"Target cache identity is valid: {args.cache_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Target cache validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

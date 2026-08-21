#!/usr/bin/env python3
"""Reject mechanically repeated data unless a narrow experiment opts in."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepspec.data.dataset_quality import (
    analyze_conversation_jsonl,
    dataset_quality_failures,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--min-unique-records", type=int, default=10_000)
    parser.add_argument("--min-unique-ratio", type=float, default=0.25)
    parser.add_argument("--allow-low-diversity", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_unique_records < 1:
        raise ValueError("--min-unique-records must be positive.")
    if not 0.0 < args.min_unique_ratio <= 1.0:
        raise ValueError("--min-unique-ratio must be in (0, 1].")
    summary = analyze_conversation_jsonl(args.input_path)
    failures = dataset_quality_failures(
        summary,
        min_unique_records=args.min_unique_records,
        min_unique_ratio=args.min_unique_ratio,
    )
    summary["quality_failures"] = failures
    print(json.dumps(summary, indent=2), flush=True)
    if failures and not args.allow_low_diversity:
        raise SystemExit(
            "Training data failed the production diversity gate. Supply a "
            "broader SOURCE_TRAIN_DATA_PATH, or set "
            "ALLOW_LOW_DIVERSITY_DATA=1 only for an intentional narrow/smoke run."
        )
    if failures:
        print(
            "WARNING: low-diversity training was explicitly allowed; this run "
            "must not be treated as a general-purpose draft model.",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()

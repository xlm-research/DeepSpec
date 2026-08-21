#!/usr/bin/env python3
"""Fail a training run when acceptance-length evaluation is not useful."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", action="append", required=True)
    parser.add_argument("--min-average-acceptance-length", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_average_acceptance_length <= 1.0:
        raise ValueError(
            "The acceptance gate must be greater than 1.0; one target token "
            "per verification provides no speculative benefit."
        )
    failures = []
    for raw_path in args.results_json:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("metrics")
        if not isinstance(rows, list) or not rows:
            failures.append(f"{path}: no evaluation metrics")
            continue
        average = sum(float(row["acceptance_length"]) for row in rows) / len(rows)
        temperature = float(payload["temperature"])
        print(
            f"T={temperature:g}: mean acceptance length={average:.4f} "
            f"across {len(rows)} datasets",
            flush=True,
        )
        if average < args.min_average_acceptance_length:
            failures.append(
                f"T={temperature:g}: {average:.4f} < "
                f"{args.min_average_acceptance_length:.4f}"
            )
    if failures:
        raise SystemExit("Acceptance gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()

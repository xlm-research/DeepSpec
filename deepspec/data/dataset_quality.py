"""Streaming quality checks for conversation JSONL training sources."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


def _record_identity(record, *, path: Path, line_number: int) -> bytes:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(
            f"{path}:{line_number} has no non-empty conversations field."
        )
    canonical = json.dumps(
        conversations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).digest()


def analyze_conversation_jsonl(path: str | Path):
    path = Path(path).expanduser().resolve()
    identities = Counter()
    total_records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
            identities[
                _record_identity(record, path=path, line_number=line_number)
            ] += 1
            total_records += 1
    if total_records == 0:
        raise ValueError(f"Training dataset is empty: {path}")
    unique_records = len(identities)
    return {
        "path": str(path),
        "total_records": total_records,
        "unique_records": unique_records,
        "unique_ratio": unique_records / total_records,
        "duplicate_records": total_records - unique_records,
        "max_repeat_count": max(identities.values()),
        "mean_repeat_count": total_records / unique_records,
    }


def dataset_quality_failures(
    summary,
    *,
    min_unique_records: int,
    min_unique_ratio: float,
):
    failures = []
    if int(summary["unique_records"]) < int(min_unique_records):
        failures.append(
            "unique_records="
            f"{summary['unique_records']} < required {int(min_unique_records)}"
        )
    if float(summary["unique_ratio"]) < float(min_unique_ratio):
        failures.append(
            f"unique_ratio={summary['unique_ratio']:.4f} < required "
            f"{float(min_unique_ratio):.4f}"
        )
    return failures


__all__ = ["analyze_conversation_jsonl", "dataset_quality_failures"]

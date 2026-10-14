"""Identity accounting for one complete pass over DSpark supervision windows."""

import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any

from torchspec.data.utils import deserialize_packed_loss_mask


class CompleteDataPass:
    def __init__(self, manifest_path: str, dispatch_size: int, accumulation_steps: int) -> None:
        manifest = json.loads(Path(manifest_path).read_text())
        self.windows: dict[str, dict[str, Any]] = {
            window["window_id"]: window for window in manifest["windows"]
        }
        self.sources: dict[str, dict[str, Any]] = {
            source["source_id"]: source for source in manifest["sources"]
        }
        if not self.windows or len(self.windows) != manifest["window_count"]:
            raise ValueError("Full-data manifest has missing or duplicate windows")
        if len(self.sources) != manifest["source_count"]:
            raise ValueError("Full-data manifest has missing or duplicate sources")
        assignments: Counter[tuple[str, int]] = Counter()
        declared: set[str] = set()
        for source in self.sources.values():
            declared.update(source["window_ids"])
        for window_id, window in self.windows.items():
            owner = self.sources.get(window["source_id"])
            if owner is None or window_id not in owner["window_ids"]:
                raise ValueError(f"Invalid source provenance for window {window_id}")
            assignments.update(
                (window["source_id"], i) for i in window["supervised_message_indices"]
            )
        expected = Counter(
            (source_id, index)
            for source_id, source in self.sources.items()
            for index in source["assistant_message_indices"]
        )
        if assignments != expected or any(count != 1 for count in assignments.values()):
            raise ValueError(
                "Full-data manifest must supervise every original segment exactly once"
            )
        if declared != set(self.windows):
            raise ValueError("Source/window manifest identities do not reconcile")
        self.dispatch_size = dispatch_size
        self.accumulation_steps = accumulation_steps
        self.optimizer_steps = math.ceil(len(self.windows) / (dispatch_size * accumulation_steps))
        self.dispatch_slots = self.optimizer_steps * accumulation_steps * dispatch_size
        self.enqueued: set[str] = set()
        self.submitted: set[str] = set()
        self.received: set[str] = set()
        self.dispatched: set[str] = set()
        self.committed: set[str] = set()
        self.duplicates: set[str] = set()
        self.failed: dict[str, str] = {}
        self.pending_batches: deque[list[str]] = deque()
        self.dispatched_slots = 0
        self.padding_count = 0
        self.completed_steps = 0
        self.status = "running"
        self.error: str | None = None

    def validate_dataset(self, samples: list[dict[str, Any]]) -> None:
        ids = [sample["data_id"] for sample in samples]
        if len(ids) != len(set(ids)) or set(ids) != set(self.windows):
            raise ValueError("Loaded dataset differs from expected full-data window IDs")
        for sample in samples:
            selection = (sample.get("metadata") or {}).get("supervision_window")
            if selection != self.windows[sample["data_id"]]:
                raise ValueError(f"Loaded supervision differs for window {sample['data_id']}")

    def enqueue(self, data_id: str) -> None:
        if data_id not in self.windows or data_id in self.enqueued:
            raise RuntimeError(f"Unknown or repeated full-data submission: {data_id}")
        self.enqueued.add(data_id)

    def submit(self, data_id: str) -> None:
        if data_id not in self.enqueued or data_id in self.submitted:
            raise RuntimeError(f"Unknown or repeated inference request: {data_id}")
        self.submitted.add(data_id)

    def receive(self, result: Any) -> None:
        data_id = result.data_id
        if data_id in self.received:
            self.duplicates.add(data_id)
            raise RuntimeError(f"Duplicate full-data inference result: {data_id}")
        if data_id not in self.submitted:
            raise RuntimeError(f"Unrequested full-data inference result: {data_id}")
        window = self.windows[data_id]
        if result.packed_loss_mask is None:
            raise RuntimeError(f"Missing expanded supervision mask for window {data_id}")
        runs = deserialize_packed_loss_mask(result.packed_loss_mask)
        if (
            sum(runs) != window["expanded_tokens"]
            or sum(runs[1::2]) != window["supervised_tokens"]
            or window["supervised_tokens"] <= 0
        ):
            raise RuntimeError(f"Invalid expanded supervision mask for window {data_id}")
        self.received.add(data_id)

    @property
    def all_received(self) -> bool:
        return len(self.received) == len(self.windows)

    def dispatch(self, real_ids: list[str], slots: int) -> None:
        if self.dispatched_slots + slots > self.dispatch_slots:
            raise RuntimeError("Full-data dispatch exceeded the complete-pass endpoint")
        for data_id in real_ids:
            if data_id not in self.received or data_id in self.dispatched:
                raise RuntimeError(f"Unknown or repeated training dispatch: {data_id}")
        self.dispatched.update(real_ids)
        self.pending_batches.append(real_ids)
        self.dispatched_slots += slots
        self.padding_count += slots - len(real_ids)

    def commit(self, step: int, num_batches: int) -> None:
        if step != self.completed_steps + 1 or num_batches != self.accumulation_steps:
            raise RuntimeError("Optimizer commit does not match the complete-pass endpoint")
        if len(self.pending_batches) < num_batches:
            raise RuntimeError("Optimizer commit has insufficient dispatched batches")
        for _ in range(num_batches):
            self.committed.update(self.pending_batches.popleft())
        self.completed_steps = step

    def fail(self, message: str, data_ids: list[str] | None = None) -> None:
        self.status = "incomplete"
        self.error = message
        for data_id in data_ids or []:
            if data_id in self.windows:
                self.failed[data_id] = message

    def finalize(self) -> None:
        if (
            self.error
            or self.committed != set(self.windows)
            or self.completed_steps != self.optimizer_steps
            or self.pending_batches
        ):
            raise RuntimeError("Full-data pass is incomplete; inspect coverage.json")
        self.status = "complete"

    def report(self) -> dict[str, Any]:
        committed_sources = [
            source_id
            for source_id, source in self.sources.items()
            if set(source["window_ids"]) <= self.committed
        ]
        return {
            "status": self.status,
            "error": self.error,
            "expected_source_count": len(self.sources),
            "expected_window_count": len(self.windows),
            "expected_segment_count": sum(
                len(source["assistant_message_indices"]) for source in self.sources.values()
            ),
            "enqueued_window_ids": sorted(self.enqueued),
            "submitted_window_ids": sorted(self.submitted),
            "received_window_ids": sorted(self.received),
            "in_flight_window_ids": sorted(self.submitted - self.received - self.failed.keys()),
            "dispatched_window_ids": sorted(self.dispatched),
            "committed_window_ids": sorted(self.committed),
            "committed_window_count": len(self.committed),
            "committed_source_ids": sorted(committed_sources),
            "committed_source_count": len(committed_sources),
            "committed_segment_count": sum(
                len(self.windows[window_id]["supervised_message_indices"])
                for window_id in self.committed
            ),
            "missing_window_ids": sorted(set(self.windows) - self.committed),
            "duplicate_window_ids": sorted(self.duplicates),
            "failed_windows": self.failed,
            "dummy_count": self.padding_count,
            "optimizer_steps": self.completed_steps,
            "expected_optimizer_steps": self.optimizer_steps,
        }

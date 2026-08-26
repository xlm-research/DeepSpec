#!/usr/bin/env python3
"""Summarize FSDP communication overlap from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path


Interval = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def duration_ns(intervals: list[Interval]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_ns(left: list[Interval], right: list[Interval]) -> int:
    left = merge_intervals(left)
    right = merge_intervals(right)
    left_index = right_index = total = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end > start:
            total += end - start
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def main() -> None:
    args = parse_args()
    connection = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
    rows = connection.execute(
        """
        SELECT kernel.globalPid, kernel.deviceId, kernel.start, kernel.end, strings.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
        JOIN StringIds AS strings ON strings.id = kernel.demangledName
        ORDER BY kernel.globalPid, kernel.start
        """
    )
    groups: dict[tuple[int, int], dict[str, list[Interval]]] = {}
    for global_pid, device_id, start, end, name in rows:
        group = groups.setdefault(
            (global_pid, device_id),
            {"all_gather": [], "reduce_scatter": [], "compute": [], "grouped_mm": []},
        )
        interval = (int(start), int(end))
        if "ncclDevKernel_AllGather" in name:
            group["all_gather"].append(interval)
        elif "ncclDevKernel_ReduceScatter" in name:
            group["reduce_scatter"].append(interval)
        elif "ncclDevKernel" not in name:
            group["compute"].append(interval)
            if "GroupProblemShape" in name:
                group["grouped_mm"].append(interval)

    ranks = []
    for (global_pid, device_id), group in sorted(groups.items()):
        communication = group["all_gather"] + group["reduce_scatter"]
        communication_ns = duration_ns(communication)
        overlap_ns = intersection_ns(communication, group["compute"])
        ranks.append(
            {
                "global_pid": global_pid,
                "device_id": device_id,
                "all_gather_count": len(group["all_gather"]),
                "all_gather_union_ms": duration_ns(group["all_gather"]) / 1e6,
                "reduce_scatter_count": len(group["reduce_scatter"]),
                "reduce_scatter_union_ms": duration_ns(group["reduce_scatter"]) / 1e6,
                "fsdp_comm_union_ms": communication_ns / 1e6,
                "fsdp_compute_overlap_ms": overlap_ns / 1e6,
                "fsdp_hidden_percent": (
                    100.0 * overlap_ns / communication_ns if communication_ns else 0.0
                ),
                "fsdp_exposed_ms": (communication_ns - overlap_ns) / 1e6,
                "grouped_mm_union_ms": duration_ns(group["grouped_mm"]) / 1e6,
                "grouped_mm_fsdp_overlap_ms": intersection_ns(
                    communication, group["grouped_mm"]
                )
                / 1e6,
            }
        )

    metric_names = [
        name
        for name in ranks[0]
        if name not in {"global_pid", "device_id"}
    ]
    summary = {
        "source": str(args.sqlite),
        "rank_count": len(ranks),
        "ranks": ranks,
        "medians": {
            name: statistics.median(float(rank[name]) for rank in ranks)
            for name in metric_names
        },
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

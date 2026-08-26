#!/usr/bin/env python3
"""Summarize FSDP/grouped-mm overlap from draft-only Kineto traces."""

from __future__ import annotations

import argparse
import bisect
import glob
import gzip
import json
import os
from pathlib import Path
from statistics import mean, median


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals):
    return sum(end - start for start, end in _merge(intervals))


def _intersection(left, right):
    left = _merge(left)
    right = _merge(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        total += max(end - start, 0.0)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _clip(intervals, windows):
    windows = _merge(windows)
    window_ends = [end for _, end in windows]
    clipped = []
    for start, end in intervals:
        # Skip every window ending at/before this interval. Trace kernels and
        # profiler phase windows may each number in the millions/hundreds, so
        # rescanning all windows for every kernel is prohibitively expensive.
        index = bisect.bisect_right(window_ends, start)
        while index < len(windows):
            window_start, window_end = windows[index]
            if window_start >= end:
                break
            overlap_start = max(start, window_start)
            overlap_end = min(end, window_end)
            if overlap_end > overlap_start:
                clipped.append((overlap_start, overlap_end))
            index += 1
    return clipped


def _is_nccl(name: str) -> bool:
    return "nccl" in name.lower()


def _is_all_gather(name: str) -> bool:
    lowered = name.lower()
    return _is_nccl(name) and ("allgather" in lowered or "all_gather" in lowered)


def _is_reduce_scatter(name: str) -> bool:
    lowered = name.lower()
    return _is_nccl(name) and (
        "reducescatter" in lowered or "reduce_scatter" in lowered
    )


def _is_grouped_mm(name: str) -> bool:
    return "groupproblemshape" in name.lower()


def _phase_metrics(kernels, windows):
    selected = [
        event
        for event in kernels
        if _clip([(event[0], event[1])], windows)
    ]
    compute = _clip(
        [(start, end) for start, end, name in selected if not _is_nccl(name)],
        windows,
    )
    nccl = _clip(
        [(start, end) for start, end, name in selected if _is_nccl(name)],
        windows,
    )
    all_gather = _clip(
        [(start, end) for start, end, name in selected if _is_all_gather(name)],
        windows,
    )
    reduce_scatter = _clip(
        [(start, end) for start, end, name in selected if _is_reduce_scatter(name)],
        windows,
    )
    grouped = _clip(
        [(start, end) for start, end, name in selected if _is_grouped_mm(name)],
        windows,
    )
    fsdp = [*all_gather, *reduce_scatter]
    fsdp_duration = _duration(fsdp)
    fsdp_overlap = _intersection(fsdp, compute)
    rs_duration = _duration(reduce_scatter)
    rs_overlap = _intersection(reduce_scatter, compute)
    return {
        "compute_union_ms": _duration(compute) / 1000.0,
        "nccl_union_ms": _duration(nccl) / 1000.0,
        "all_gather_count": sum(_is_all_gather(name) for _, _, name in selected),
        "all_gather_union_ms": _duration(all_gather) / 1000.0,
        "reduce_scatter_count": sum(
            _is_reduce_scatter(name) for _, _, name in selected
        ),
        "reduce_scatter_union_ms": rs_duration / 1000.0,
        "fsdp_comm_union_ms": fsdp_duration / 1000.0,
        "fsdp_compute_overlap_ms": fsdp_overlap / 1000.0,
        "fsdp_hidden_percent": (
            100.0 * fsdp_overlap / fsdp_duration if fsdp_duration else 0.0
        ),
        "fsdp_exposed_ms": (fsdp_duration - fsdp_overlap) / 1000.0,
        "reduce_scatter_exposed_ms": (rs_duration - rs_overlap) / 1000.0,
        "grouped_mm_union_ms": _duration(grouped) / 1000.0,
        "grouped_mm_nccl_overlap_ms": _intersection(grouped, nccl) / 1000.0,
        "grouped_mm_fsdp_overlap_ms": _intersection(grouped, fsdp) / 1000.0,
    }


def analyze_trace(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    annotations = {}
    kernels = []
    host_sync_calls = 0
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        start = float(event["ts"])
        end = start + float(event["dur"])
        name = str(event.get("name", ""))
        category = event.get("cat")
        if category == "user_annotation" and name.startswith("deepspec::"):
            annotations.setdefault(name, []).append((start, end))
        elif category == "kernel":
            kernels.append((start, end, name))
        elif category in {"cuda_runtime", "cuda_driver"} and name in {
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaEventBlockingSync",
        }:
            host_sync_calls += 1
    step_windows = annotations.get("deepspec::draft_benchmark_profile_step", [])
    if len(step_windows) != 1:
        raise RuntimeError(f"Expected one profile step in {path}, got {step_windows}")
    forward_windows = annotations.get("deepspec::draft_forward", [])
    backward_windows = annotations.get("deepspec::draft_backward", [])
    return {
        "trace": os.path.basename(path),
        "step_cpu_ms": _duration(step_windows) / 1000.0,
        "host_sync_calls": host_sync_calls,
        "full_step": _phase_metrics(kernels, step_windows),
        "draft_forward": _phase_metrics(kernels, forward_windows),
        "draft_backward": _phase_metrics(kernels, backward_windows),
        "final_micro_backward": _phase_metrics(
            kernels,
            backward_windows[-1:] if backward_windows else [],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.trace_dir, "*.pt.trace.json.gz")))
    if not paths:
        raise FileNotFoundError(f"No Kineto traces in {args.trace_dir}")
    ranks = [analyze_trace(path) for path in paths]
    aggregate = {
        "rank_count": len(ranks),
        "ranks": ranks,
        "mean": {},
        "median": {},
    }
    aggregate["mean"]["step_cpu_ms"] = mean(row["step_cpu_ms"] for row in ranks)
    aggregate["mean"]["host_sync_calls"] = mean(
        row["host_sync_calls"] for row in ranks
    )
    aggregate["median"]["step_cpu_ms"] = median(
        row["step_cpu_ms"] for row in ranks
    )
    aggregate["median"]["host_sync_calls"] = median(
        row["host_sync_calls"] for row in ranks
    )
    for phase in ("full_step", "draft_forward", "draft_backward", "final_micro_backward"):
        aggregate["mean"][phase] = {
            key: mean(row[phase][key] for row in ranks)
            for key in ranks[0][phase]
        }
        aggregate["median"][phase] = {
            key: median(row[phase][key] for row in ranks)
            for key in ranks[0][phase]
        }
    rendered = json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n"
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

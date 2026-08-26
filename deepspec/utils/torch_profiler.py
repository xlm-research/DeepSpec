from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any

import torch


class NullProfiler:
    """Profiler-compatible no-op used on ranks that are not being traced."""

    enabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def step(self) -> None:
        return None


def _rank_is_selected(ranks: Any, global_rank: int) -> bool:
    if isinstance(ranks, str):
        value = ranks.strip().lower()
        if value == "all":
            return True
        if not value:
            return False
        ranks = [part.strip() for part in value.split(",")]
    elif isinstance(ranks, int):
        ranks = [ranks]
    elif ranks is None:
        ranks = [0]
    try:
        return int(global_rank) in {int(rank) for rank in ranks}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "profiling.ranks must be 'all', an integer, a list of integers, "
            f"or a comma-separated string; got {ranks!r}."
        ) from exc


def _positive_int(config, name: str, *, allow_zero: bool = False) -> int:
    value = int(config.get(name, 0))
    minimum = 0 if allow_zero else 1
    if value < minimum:
        operator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"profiling.{name} must be {operator}; got {value}.")
    return value


@dataclass
class _TraceWriter:
    trace_dir: str
    worker_name: str
    row_limit: int
    record_shapes: bool
    use_gzip: bool

    def __post_init__(self) -> None:
        self._cycle = 0
        self._trace_handler = torch.profiler.tensorboard_trace_handler(
            self.trace_dir,
            worker_name=self.worker_name,
            use_gzip=self.use_gzip,
        )

    def __call__(self, profiler) -> None:
        self._trace_handler(profiler)
        averages = profiler.key_averages(
            group_by_input_shape=self.record_shapes,
        )
        self_sort_by = (
            "self_device_time_total"
            if torch.cuda.is_available()
            else "self_cpu_time_total"
        )
        inclusive_sort_by = (
            "device_time_total" if torch.cuda.is_available() else "cpu_time_total"
        )
        self_table = averages.table(
            sort_by=self_sort_by,
            row_limit=self.row_limit,
        )
        inclusive_table = averages.table(
            sort_by=inclusive_sort_by,
            row_limit=self.row_limit,
        )
        summary_path = os.path.join(
            self.trace_dir,
            f"{self.worker_name}.cycle_{self._cycle}.key_averages.txt",
        )
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(f"Inclusive operators/stages sorted by {inclusive_sort_by}\n")
            handle.write(inclusive_table)
            handle.write("\n\n")
            handle.write(f"Self time sorted by {self_sort_by}\n")
            handle.write(self_table)
            handle.write("\n")
        # Kineto may expose both the CPU record_function event and its GPU user
        # annotation under the same name.  Keep the enclosing CPU event (the
        # one with the longest CPU duration) so each logical stage appears once
        # and its duration remains a wall-clock-like host view.
        stage_events = {}
        for event in averages:
            if not event.key.startswith("deepspec::"):
                continue
            previous = stage_events.get(event.key)
            if previous is None or event.cpu_time_total > previous.cpu_time_total:
                stage_events[event.key] = event
        stage_rows = []
        for event in stage_events.values():
            stage_rows.append(
                {
                    "name": event.key,
                    "count": int(event.count),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "self_cpu_time_total_us": float(event.self_cpu_time_total),
                    "device_time_total_us": float(event.device_time_total),
                    "self_device_time_total_us": float(event.self_device_time_total),
                    "cpu_memory_usage_bytes": int(event.cpu_memory_usage),
                    "device_memory_usage_bytes": int(event.device_memory_usage),
                }
            )
        stage_rows.sort(
            key=lambda row: (
                row["device_time_total_us"],
                row["cpu_time_total_us"],
            ),
            reverse=True,
        )
        stages_path = os.path.join(
            self.trace_dir,
            f"{self.worker_name}.cycle_{self._cycle}.stages.json",
        )
        with open(stages_path, "w", encoding="utf-8") as handle:
            json.dump(stage_rows, handle, indent=2, ensure_ascii=False)
        print(
            "[deepspec-profiler] "
            f"rank trace cycle {self._cycle} written to {self.trace_dir}; "
            f"operator summary={summary_path}; stage summary={stages_path}",
            flush=True,
        )
        self._cycle += 1


def build_torch_profiler(config, *, global_rank: int, world_size: int):
    """Build a scheduled PyTorch profiler for one training rank.

    Schedule units are *micro-steps*, because ``BaseTrainer`` calls ``step``
    after every forward/backward pair.  This makes gradient accumulation and
    the optimizer boundary visible in the same trace.
    """

    if not config or not bool(config.get("enabled", False)):
        return NullProfiler()
    if not _rank_is_selected(config.get("ranks", [0]), global_rank):
        return NullProfiler()

    trace_dir = os.path.abspath(os.fspath(config.get("trace_dir", "torch_profile")))
    os.makedirs(trace_dir, exist_ok=True)
    wait_steps = _positive_int(config, "wait_steps", allow_zero=True)
    warmup_steps = _positive_int(config, "warmup_steps", allow_zero=True)
    active_steps = _positive_int(config, "active_steps")
    repeat = _positive_int(config, "repeat")
    skip_first_steps = _positive_int(
        config,
        "skip_first_steps",
        allow_zero=True,
    )
    row_limit = _positive_int(config, "row_limit")
    record_shapes = bool(config.get("record_shapes", True))
    profile_memory = bool(config.get("profile_memory", True))
    with_stack = bool(config.get("with_stack", True))
    with_flops = bool(config.get("with_flops", True))
    use_gzip = bool(config.get("use_gzip", True))

    worker_name = (
        f"{socket.gethostname().replace('.', '_')}.rank_{int(global_rank):05d}"
    )
    metadata = {
        "worker_name": worker_name,
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "schedule_unit": "training_micro_step",
        "schedule": {
            "skip_first_steps": skip_first_steps,
            "wait_steps": wait_steps,
            "warmup_steps": warmup_steps,
            "active_steps": active_steps,
            "repeat": repeat,
        },
        "record_shapes": record_shapes,
        "profile_memory": profile_memory,
        "with_stack": with_stack,
        "with_flops": with_flops,
    }
    metadata_path = os.path.join(trace_dir, f"{worker_name}.metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_writer = _TraceWriter(
        trace_dir=trace_dir,
        worker_name=worker_name,
        row_limit=row_limit,
        record_shapes=record_shapes,
        use_gzip=use_gzip,
    )
    print(
        "[deepspec-profiler] "
        f"enabled on global_rank={global_rank}; trace_dir={trace_dir}; "
        f"skip_first={skip_first_steps}, wait={wait_steps}, "
        f"warmup={warmup_steps}, active={active_steps}, repeat={repeat}",
        flush=True,
    )
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=repeat,
            skip_first=skip_first_steps,
        ),
        on_trace_ready=trace_writer,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
        acc_events=True,
    )


__all__ = ["NullProfiler", "build_torch_profiler"]

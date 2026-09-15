"""Optional synchronized phase measurements, independent of training state."""

from contextlib import contextmanager
import resource
import socket
import time

import torch
import torch.distributed as dist


_COMPILE_FIELDS = (
    "compile_id",
    "co_name",
    "is_forward",
    "is_runtime",
    "start_time_us",
    "end_time_us",
    "duration_us",
    "dynamo_cumulative_compile_time_us",
    "inductor_cumulative_compile_time_us",
    "backward_cumulative_compile_time_us",
    "triton_compile_time_us",
    "runtime_triton_autotune_time_us",
    "inductor_fx_local_cache_hit_count",
    "inductor_fx_local_cache_miss_count",
    "aotautograd_local_cache_hit_count",
    "aotautograd_local_cache_miss_count",
    "recompile_reason",
    "fail_type",
)


class PhaseTiming:
    def __init__(self, enabled):
        self.enabled = enabled
        self.started = time.monotonic()
        self.events = []

    def record(self, name, started, **metadata):
        if self.enabled:
            torch.cuda.synchronize()
            self.events.append(
                {
                    "name": name,
                    "start_seconds": started - self.started,
                    "seconds": time.monotonic() - started,
                    **metadata,
                }
            )

    @contextmanager
    def measure(self, name, **metadata):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        started = time.monotonic()
        yield
        self.record(name, started, **metadata)

    def report(self):
        if not self.enabled:
            return None
        from torch._dynamo.utils import get_compilation_metrics
        from torch._inductor import config as inductor_config

        torch.cuda.synchronize()
        finished = time.monotonic()
        rank = {
            "rank": dist.get_rank(),
            "hostname": socket.gethostname(),
            "started_monotonic": self.started,
            "finished_monotonic": finished,
            "native_seconds": finished - self.started,
            "events": self.events,
            # Compiler durations are nested observations within training;
            # they must not be added to training or phase wall times.
            "compilation": [
                {name: getattr(metric, name) for name in _COMPILE_FIELDS}
                for metric in get_compilation_metrics()
            ],
            "compile_threads": inductor_config.compile_threads,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, rank)
        # Preserve each rank's nonoverlapping timeline. Concurrent rank times
        # must never be added together to estimate the phase wall time.
        result = {
            "max_rank_seconds": max(value["native_seconds"] for value in ranks),
            "peak_allocated_bytes": max(
                value["peak_allocated_bytes"] for value in ranks
            ),
            "ranks": ranks,
        }
        if len({value["hostname"] for value in ranks}) == 1:
            result["started_monotonic"] = min(
                value["started_monotonic"] for value in ranks
            )
            result["finished_monotonic"] = max(
                value["finished_monotonic"] for value in ranks
            )
            result["native_seconds"] = (
                result["finished_monotonic"] - result["started_monotonic"]
            )
        return result

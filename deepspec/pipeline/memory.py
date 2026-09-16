"""Single-node feature budget shared by the Store and all model ranks."""

from pathlib import Path

GIB = 1024**3


def node_memory():
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) * 1024
    # Resolve the job's cgroup rather than assuming the hierarchy root is its limit.
    entry = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    current = Path("/sys/fs/cgroup") / entry.lstrip("/")
    if not current.exists():
        current = Path("/sys/fs/cgroup")
    limits, remaining = [info["MemTotal"]], [info["MemAvailable"]]
    while str(current).startswith("/sys/fs/cgroup"):
        limit_file, used_file = current / "memory.max", current / "memory.current"
        if limit_file.exists() and used_file.exists():
            raw = limit_file.read_text().strip()
            if raw != "max":
                limit = int(raw)
                limits.append(limit)
                remaining.append(max(0, limit - int(used_file.read_text())))
        if current == Path("/sys/fs/cgroup"):
            break
        current = current.parent
    return {
        "physical_bytes": info["MemTotal"],
        "available_bytes": info["MemAvailable"],
        "limit_bytes": min(limits),
        "headroom_bytes": min(remaining),
    }


def feature_budget(pool_bytes, max_sample_bytes, window, readers, gas):
    snapshot = node_memory()
    # The pool includes its replicas (one here); freed objects remain in this pool.
    # Writer raw/gather/converted buffers, every rank's CPU/GPU-verification
    # staging, and each Store client's preallocated local buffer are outside it.
    reserve = 64 * GIB
    budget = max(
        0,
        min(
            int(snapshot["physical_bytes"] * 0.8),
            int(snapshot["limit_bytes"] * 0.8),
            snapshot["headroom_bytes"] - reserve,
        ),
    )
    scratch = (3 * window + 2 * readers * gas) * max_sample_bytes
    # Each training rank has separate metadata and background feature clients.
    local_buffers = (2 * readers + 2) * 16 * 1024**2
    estimate = pool_bytes + scratch + local_buffers + GIB
    if estimate > budget:
        raise ValueError(
            f"Feature allocation bound {estimate} exceeds node budget {budget}"
        )
    return {
        "feature_memory_budget": budget,
        "feature_memory_bound": estimate,
        "memory_reserve_bytes": reserve,
        "memory_snapshot": snapshot,
        "pool_bytes": pool_bytes,
        "scratch_bound_bytes": scratch,
        "client_buffers_bytes": local_buffers,
    }

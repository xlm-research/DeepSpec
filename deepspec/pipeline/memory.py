"""Single-node feature budget shared by the Store and all model ranks."""

import math
import time
import uuid
from pathlib import Path

GIB = 1024**3


def node_feature_budget(
    *,
    pool_bytes,
    max_sample_bytes,
    writers,
    writer_inflight,
    readers,
    gas,
    prefetch_depth,
    prefetch_bytes,
    snapshot,
    feature_cap=None,
    client_buffer_bytes=None,
):
    """Total host-feature bound before allocation; GPU memory is separate."""
    values = (pool_bytes, writers, readers)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("Pool/writer/reader counts must be nonnegative integers")
    if any(
        type(v) is not int or v <= 0
        for v in (max_sample_bytes, writer_inflight, gas, prefetch_depth)
    ):
        raise ValueError("Sample bytes, inflight, GAS and prefetch must be positive")
    prefetch = (
        prefetch_depth * max_sample_bytes if prefetch_bytes is None else prefetch_bytes
    )
    if prefetch < max_sample_bytes:
        raise ValueError(
            "Per-reader prefetch byte limit cannot hold the largest sample"
        )
    clients = writers + 2 * readers + int(pool_bytes > 0)
    buffers = (
        clients * 16 * 1024**2 if client_buffer_bytes is None else client_buffer_bytes
    )
    if type(buffers) is not int or buffers < 0:
        raise ValueError(
            "Actual client buffer allocation must be nonnegative integer bytes"
        )
    writer_bound = 3 * writers * writer_inflight * max_sample_bytes
    transport_bound = writers * writer_inflight * max_sample_bytes
    reader_bound = readers * (2 * gas * max_sample_bytes + prefetch)
    bound = pool_bytes + writer_bound + transport_bound + reader_bound + buffers + GIB
    static = min(snapshot["physical_bytes"] * 4 // 5, snapshot["limit_bytes"] * 4 // 5)
    if feature_cap is not None:
        static = min(static, feature_cap)
    startup = max(0, min(static, snapshot["headroom_bytes"] - 64 * GIB))
    if bound > startup:
        raise ValueError(
            f"Feature allocation bound {bound} exceeds startup budget {startup}"
        )
    return {
        "pool_bytes": pool_bytes,
        "writers": writers,
        "readers": readers,
        "writer_bound": writer_bound,
        "transport_bound": transport_bound,
        "reader_bound": reader_bound,
        "client_count": clients,
        "client_bound": buffers,
        "prefetch_bytes_per_reader": prefetch,
        "reserve_bytes": GIB,
        "headroom_reserve_bytes": 64 * GIB,
        "feature_bound": bound,
        "static_cap": static,
        "startup_budget": startup,
        "retained_charge_lower_bound": 0,
        "remaining_bound": bound,
        "memory_snapshot": dict(snapshot),
    }


def check_runtime_budget(approved, snapshot, *, run_id, update_index, charges=()):
    """Only proven, retained, non-overlapping charges reduce future allocation."""
    retained, seen = 0, set()
    for charge in charges:
        identity = charge.get("allocation_id")
        if not identity or identity in seen:
            raise ValueError("Duplicate or missing retained allocation identity")
        seen.add(identity)
        if (
            charge.get("run_id") == run_id
            and charge.get("resident_locked") is True
            and set(charge.get("charged_domains", ())) >= {"physical", "cgroup"}
            and charge.get("valid_through_update", -1) >= update_index
        ):
            nbytes = charge["nbytes"]
            if type(nbytes) is not int or nbytes < 0:
                raise ValueError("Retained charge must be nonnegative integer bytes")
            retained += nbytes
    bound = approved["feature_bound"]
    if retained > bound:
        raise ValueError("Retained charge exceeds approved component bound")
    current_cap = min(
        approved["static_cap"],
        snapshot["physical_bytes"] * 4 // 5,
        snapshot["limit_bytes"] * 4 // 5,
    )
    remaining = bound - retained
    pressure = (
        bound > min(approved["startup_budget"], current_cap)
        or snapshot["headroom_bytes"] < 64 * GIB + remaining
    )
    return {
        "pressure": pressure,
        "feature_bound": bound,
        "retained_charge_lower_bound": retained,
        "remaining_bound": remaining,
        "current_static_cap": current_cap,
        "headroom_bytes": snapshot["headroom_bytes"],
        "headroom_required": 64 * GIB + remaining,
    }


class BudgetAdmission:
    """Request/epoch/freshness fencing on the controller's monotonic clock."""

    def __init__(
        self,
        budgets,
        *,
        run_id,
        plan_hash,
        freshness,
        transfer_timeout,
        run_deadline,
        clock=time.monotonic,
    ):
        if any(
            not math.isfinite(v) or v <= 0
            for v in (freshness, transfer_timeout, run_deadline)
        ):
            raise ValueError("Budget deadlines must be finite and positive")
        self.budgets = budgets
        self.run_id, self.plan_hash = run_id, plan_hash
        self.freshness, self.transfer_timeout = freshness, transfer_timeout
        self.run_deadline, self.clock = run_deadline, clock
        self.requests, self.started, self.epochs, self.sequences = {}, {}, {}, {}
        self.last_request_ids = {}
        self.approved_requests = {}
        self.admitted = set()
        self.last_observations = {}
        self.reason = None

    def request(self, update_index):
        now = self.clock()
        self.started.setdefault(update_index, now)
        if now >= min(
            self.run_deadline, self.started[update_index] + self.transfer_timeout
        ):
            raise TimeoutError("Update-group budget wait exceeded shared deadline")
        request = {
            "schema_version": 3,
            "run_id": self.run_id,
            "plan_hash": self.plan_hash,
            "sender_identity": {"component": "admission"},
            "event_id": uuid.uuid4().hex,
            "update_index": update_index,
            "request_id": uuid.uuid4().hex,
        }
        self.requests[update_index] = (request, now)
        self.approved_requests.pop(update_index, None)
        return dict(request)

    def check(self, update_index, responses):
        from .runtime import validate_message

        request, sent_at = self.requests[update_index]
        self.approved_requests.pop(update_index, None)
        now = self.clock()
        if now >= min(
            self.run_deadline, self.started[update_index] + self.transfer_timeout
        ):
            raise TimeoutError("Update-group budget wait exceeded shared deadline")
        age = now - sent_at
        self.reason = "budget_snapshot_stale"
        if not 0 <= age < self.freshness:
            return False
        self.reason = "budget_node_missing"
        node_ids = [r["node_id"] for r in responses]
        if len(set(node_ids)) != len(node_ids) or set(node_ids) != set(self.budgets):
            return False
        observations, epochs, sequences = {}, {}, {}
        for response in responses:
            validate_message(response, run_id=self.run_id, plan_hash=self.plan_hash)
            node_id = response["node_id"]
            epoch = (response["boot_id"], response["agent_epoch"])
            sequence = response["sample_seq"]
            self.reason = "budget_identity_stale"
            if (
                response["request_id"] != request["request_id"]
                or response["update_index"] != update_index
                or (node_id in self.epochs and epoch != self.epochs[node_id])
                or sequence < self.sequences.get(node_id, -1)
                or (
                    sequence == self.sequences.get(node_id, -1)
                    and self.last_request_ids.get(node_id) != request["request_id"]
                )
            ):
                return False
            observations[node_id] = check_runtime_budget(
                self.budgets[node_id],
                response["memory"],
                run_id=self.run_id,
                update_index=update_index,
                charges=response.get("retained_charges", ()),
            )
            observations[node_id]["age_upper_bound"] = age
            epochs[node_id], sequences[node_id] = epoch, sequence
        self.last_observations = observations
        self.epochs.update(epochs)
        self.sequences.update(sequences)
        self.last_request_ids.update({node: request["request_id"] for node in node_ids})
        self.reason = (
            "node_headroom"
            if any(r["pressure"] for r in observations.values())
            else None
        )
        if self.reason is None:
            self.approved_requests[update_index] = request["request_id"]
        return self.reason is None

    def commit(self, update_index):
        if update_index not in self.approved_requests:
            raise ValueError("Cannot commit an unapproved update group")
        request, sent_at = self.requests[update_index]
        now = self.clock()
        if (
            self.approved_requests[update_index] != request["request_id"]
            or not 0 <= now - sent_at < self.freshness
            or now
            >= min(
                self.run_deadline, self.started[update_index] + self.transfer_timeout
            )
        ):
            self.approved_requests.pop(update_index, None)
            raise ValueError("Budget approval is no longer fresh at commit")
        self.admitted.add(update_index)

    def is_admitted(self, update_index):
        return update_index in self.admitted


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


def feature_budget(
    pool_bytes,
    max_sample_bytes,
    window,
    readers,
    gas,
    *,
    writer=True,
    snapshot=None,
    writer_inflight=None,
):
    snapshot = node_memory() if snapshot is None else snapshot
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
    staging = window if writer_inflight is None else writer_inflight
    if staging < 1:
        raise ValueError("Writer staging must allow at least one sample")
    scratch = (3 * staging * int(writer) + 2 * readers * gas) * max_sample_bytes
    # The async writer copies converted features into one registered host
    # buffer per in-flight slot.  Keep this separate from model scratch so
    # diagnostics can distinguish transport pressure from tensor pressure.
    transport_staging = staging * int(writer) * max_sample_bytes
    # Each training rank has separate metadata and background feature clients.
    local_buffers = (2 * readers + int(writer) + int(pool_bytes > 0)) * 16 * 1024**2
    estimate = pool_bytes + scratch + transport_staging + local_buffers + GIB
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
        "transport_staging_bound_bytes": transport_staging,
        "client_buffers_bytes": local_buffers,
    }

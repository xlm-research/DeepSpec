"""Lifecycle helpers for a run-owned Mooncake master process."""

from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class PipelineError(ValueError):
    def __init__(
        self,
        code,
        message,
        *,
        run_id=None,
        phase=None,
        node_id=None,
        field_path=None,
        retryable=False,
        exit_code=2,
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.details = {
            "code": code,
            "message": message,
            "run_id": run_id,
            "phase": phase,
            "node_id": node_id,
            "field_path": field_path,
            "retryable": retryable,
        }

    def to_dict(self):
        return dict(self.details)

    def __reduce__(self):
        # BaseException otherwise reconstructs from args=(message,), losing the
        # required code argument when Ray transports the original failure.
        return (
            type(self),
            (self.details["code"], self.details["message"]),
            self.__dict__,
        )


def atomic_json(path, value):
    """Validate before touching the old file, then atomically publish and sync."""
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_message(message, *, run_id, plan_hash):
    required = ("schema_version", "run_id", "plan_hash", "sender_identity", "event_id")
    if any(not message.get(k) for k in required) or message["schema_version"] != 3:
        raise PipelineError("MESSAGE_INVALID", "Message envelope is incomplete")
    if message["run_id"] != run_id or message["plan_hash"] != plan_hash:
        raise PipelineError(
            "MESSAGE_IDENTITY_MISMATCH", "Message belongs to another run or plan"
        )
    json.dumps(message, allow_nan=False)
    return message


def atomic_json_once(path, value):
    """Publish a fully written request exactly once without replacing evidence."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    atomic_json(temporary, value)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Deadline:
    expires_at: float

    @classmethod
    def after(cls, seconds, *, clock=time.monotonic):
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Deadline duration must be positive and finite")
        return cls(clock() + seconds)

    def remaining(self, *, clock=time.monotonic):
        remaining = self.expires_at - clock()
        if remaining <= 0:
            raise TimeoutError("Shared deadline expired")
        return remaining


@contextmanager
def bounded_lock(lock, deadline):
    if not lock.acquire(timeout=deadline.remaining()):
        raise TimeoutError("Lock acquisition exceeded shared deadline")
    try:
        yield
    finally:
        lock.release()


def message_envelope(run_id, plan_hash, sender_identity, **payload):
    return {
        **payload,
        "schema_version": 3,
        "run_id": run_id,
        "plan_hash": plan_hash,
        "sender_identity": sender_identity,
        "event_id": uuid.uuid4().hex,
    }


def component_deadline(config, *, clock=time.monotonic):
    duration = config.get("timeouts_seconds", {}).get(
        "run", config.get("timeout_seconds", 1800)
    )
    remaining = float(os.environ.get("DEEPSPEC_RUN_REMAINING_SECONDS", duration))
    return Deadline.after(min(duration, remaining), clock=clock)


def actor_identity(config):
    """Capture a live actor's Linux identity before any native initialization."""
    import ray

    from deepspec.orchestration.process import capture_process

    context = ray.get_runtime_context()
    return message_envelope(
        config["run_id"],
        config["plan_hash"],
        {"component": "actor", "actor_id": context.get_actor_id()},
        node_id=context.get_node_id(),
        actor_id=context.get_actor_id(),
        process=capture_process(os.getpid(), config["run_id"]),
    )


def get_with_deadline(
    ref, config, *, deadline=None, timeout=None, clock=time.monotonic
):
    import ray

    duration = config.get("timeouts_seconds", {}).get(
        "transfer", config.get("timeout_seconds", 1800)
    )
    if timeout is not None:
        duration = min(duration, timeout)
    if deadline is not None:
        duration = min(duration, deadline.remaining(clock=clock))
    if not math.isfinite(duration) or duration <= 0:
        raise TimeoutError("RPC deadline expired")
    return ray.get(ref, timeout=duration)


def notify_buffer_failure(buffer, config, error, *, message):
    cleanup = config.get("timeouts_seconds", {}).get("cleanup", 35)
    try:
        get_with_deadline(
            buffer.fail.remote(message),
            config,
            deadline=Deadline.after(cleanup),
            timeout=cleanup,
        )
    except Exception as secondary:  # noqa: BLE001 -- notification failure must not replace the first cause
        error.add_note(f"Failure notification failed: {secondary}")


# Required measurements may be null only with a field-specific missing reason.
EVENT_FIELDS = {
    "phase_duration": ("phase", "duration_seconds"),
    "resource_sample": ("memory", "processes", "gpu_processes"),
    "gpu_occupancy_observed": (
        "node_id",
        "gpu_uuids",
        "gpu_sharing",
        "external_processes",
    ),
    "node_environment": ("identities", "placement"),
    "feature_produced": ("position", "nbytes", "tokens", "duration_seconds"),
    "feature_read": (
        "position",
        "reader_rank",
        "nbytes",
        "verified",
        "duration_seconds",
    ),
    "reader_copy_retired": ("position", "reader_rank", "nbytes"),
    "rank_update_completed": (
        "optimizer_step",
        "native_cursor",
        "sample_cursor",
        "loss",
    ),
    "budget_sample": (
        "approved_bytes",
        "observed_bytes",
        "headroom_bytes",
        "request_id",
    ),
    "wait": ("reason", "duration_seconds"),
    "checkpoint_committed": (
        "path",
        "commit_identity",
        "native_cursor",
        "sample_cursor",
    ),
    "checkpoint_expectation": ("path", "sha256"),
    "cleanup": ("resource_id", "release_state"),
    "orphan_failure": ("reason", "cleanup_complete"),
}


class EventWriter:
    """One immutable identity per file, with an exclusive lifetime writer lock."""

    def __init__(self, path, *, run_id, plan_hash, sender_identity):
        self.identity = json.loads(json.dumps(sender_identity, allow_nan=False))
        self.run_id, self.plan_hash = run_id, plan_hash
        if not run_id or not plan_hash or not self.identity:
            raise PipelineError(
                "EVENT_IDENTITY_MISSING",
                "Events require run, plan and sender identities",
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.stream.seek(0)
            first = self.stream.readline()
            if first:
                old = json.loads(first)
                if (old["run_id"], old["plan_hash"], old["sender_identity"]) != (
                    run_id,
                    plan_hash,
                    self.identity,
                ):
                    raise PipelineError(
                        "EVENT_IDENTITY_MISMATCH",
                        "An event file cannot change its writer identity",
                    )
            self.stream.seek(0, os.SEEK_END)
        except BaseException as error:
            self.stream.close()
            if isinstance(error, BlockingIOError):
                raise PipelineError(
                    "EVENT_WRITER_BUSY", "Event file already has a writer"
                ) from error
            raise

    def emit(self, kind, data, *, basis, missing=None, causes=()):
        if (
            basis not in ("declared", "observed", "verified")
            or kind not in EVENT_FIELDS
        ):
            raise PipelineError("EVENT_INVALID", "Unknown event type or evidence basis")
        missing = missing or {}
        for field in EVENT_FIELDS[kind]:
            if field not in data or (data[field] is None and not missing.get(field)):
                raise PipelineError(
                    "EVENT_FIELD_MISSING",
                    f"{kind} needs {field} or a missing reason",
                    field_path=field,
                )
        event = {
            "schema_version": 3,
            "run_id": self.run_id,
            "plan_hash": self.plan_hash,
            "sender_identity": self.identity,
            "event_id": uuid.uuid4().hex,
            "event": kind,
            "basis": basis,
            "data": data,
            "missing": missing,
            "causes": list(causes),
            "local_monotonic": time.monotonic(),
        }
        encoded = json.dumps(event, allow_nan=False) + "\n"
        self.stream.write(encoded)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return event

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def endpoint_parts(endpoint):
    try:
        host, port = endpoint.rsplit(":", 1)
        return host, int(port)
    except (AttributeError, ValueError) as error:
        raise ValueError(f"Invalid Mooncake endpoint: {endpoint!r}") from error


def wait_for_endpoint(endpoint, *, process=None, timeout=30, poll_interval=0.1):
    host, port = endpoint_parts(endpoint)
    deadline = time.monotonic() + float(timeout)
    while True:
        try:
            with socket.create_connection(
                (host, port), timeout=min(1.0, poll_interval + 0.5)
            ):
                return
        except OSError:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"Mooncake master exited before becoming reachable: {process.returncode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Mooncake master did not become reachable at {endpoint}"
                )
            time.sleep(max(float(poll_interval), 0.01))


class MooncakeMaster:
    """Start and stop a master under the existing orphan-process supervisor."""

    def __init__(
        self,
        endpoint,
        log_path,
        *,
        metrics_port,
        ttl_seconds=300,
        env=None,
        run_timeout=86400,
        cleanup_timeout=35,
    ):
        self.endpoint = endpoint
        self.log_path = Path(log_path)
        self.metrics_port = int(metrics_port)
        self.ttl_seconds = int(ttl_seconds)
        self.env = dict(os.environ if env is None else env)
        self.process = None
        self.handle = None
        self.run_timeout, self.cleanup_timeout = run_timeout, cleanup_timeout
        self.log = None

    def start(self, *, timeout=30):
        if self.process is not None:
            return self
        import mooncake

        from deepspec.orchestration.process import start_owned

        _host, port = endpoint_parts(self.endpoint)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        command = [
            str(Path(mooncake.__file__).parent / "mooncake_master"),
            f"--rpc_port={port}",
            f"--metrics_port={self.metrics_port}",
            f"--default_kv_lease_ttl={self.ttl_seconds}s",
            "--enable_offload=false",
            "--enable_disk_eviction=false",
        ]
        self.env.setdefault("DEEPSPEC_ORCHESTRATOR_PID", str(os.getpid()))
        try:
            self.handle = start_owned(
                command,
                env=self.env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                timeout=self.run_timeout,
                cleanup_timeout=self.cleanup_timeout,
                report_path=self.log_path.with_suffix(
                    self.log_path.suffix + ".cleanup.json"
                ),
            )
            self.process = self.handle.process
        except Exception:
            self.log.close()
            self.log = None
            raise
        try:
            wait_for_endpoint(self.endpoint, process=self.process, timeout=timeout)
        except BaseException as error:
            try:
                self.stop()
            except Exception as cleanup_error:  # noqa: BLE001 -- keep startup as the first failure
                error.add_note(f"Master cleanup failed: {cleanup_error}")
            raise
        return self

    def poll(self):
        return None if self.process is None else self.process.poll()

    def stop(self, *, timeout=None):
        if self.handle is None:
            return {"cleanup_complete": True}
        deadline = Deadline.after(self.cleanup_timeout if timeout is None else timeout)
        try:
            report = self.handle.stop(timeout=deadline.remaining())
            if not report["cleanup_complete"]:
                raise RuntimeError(f"Master cleanup could not be confirmed: {report}")
            self.handle = self.process = None
            return report
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None

    def __enter__(self):
        return self.start()

    def __exit__(self, _type, _value, _traceback):
        self.stop()

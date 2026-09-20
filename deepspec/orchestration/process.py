"""Bounded, run-owned supervision with PID/start-time identity checks."""

import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path


def _stat(pid, proc_root=Path("/proc")):
    fields = (Path(proc_root) / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": int(pid),
        "state": fields[0],
        "parent_pid": int(fields[1]),
        "group_id": int(fields[2]),
        "start_ticks": int(fields[19]),
    }


def capture_process(pid, run_id, *, proc_root=Path("/proc")):
    before = _stat(pid, proc_root)
    marker = f"DEEPSPEC_PIPELINE_RUN_ID={run_id}".encode()
    environment = (Path(proc_root) / str(pid) / "environ").read_bytes().split(b"\0")
    after = _stat(pid, proc_root)
    if before["start_ticks"] != after["start_ticks"] or marker not in environment:
        raise ValueError("Process run marker or PID/start-time does not match")
    return {**after, "run_id": run_id}


def signal_process(identity, sig, *, proc_root=Path("/proc")):
    """Never signal a reused PID or a process whose ownership cannot be checked."""
    fd = None
    try:
        state = _stat(identity["pid"], proc_root)
        if state["start_ticks"] != identity["start_ticks"]:
            return "unknown"
        if state["state"] == "Z":
            return "released"
        current = capture_process(
            identity["pid"], identity["run_id"], proc_root=proc_root
        )
        if current["start_ticks"] != identity["start_ticks"]:
            return "unknown"
        if current["state"] == "Z":
            return "released"
        # Prefer pidfd when this Python build exposes it. Older build headers
        # omit the API even on Linux; psutil also fences its cached PID identity.
        use_pidfd = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
        if use_pidfd:
            fd = os.pidfd_open(identity["pid"])
        else:
            import psutil

            try:
                process = psutil.Process(identity["pid"])
            except psutil.NoSuchProcess:
                return "released"
        current = capture_process(
            identity["pid"], identity["run_id"], proc_root=proc_root
        )
        if current["start_ticks"] != identity["start_ticks"]:
            return "unknown"
        if use_pidfd:
            signal.pidfd_send_signal(fd, sig)
        else:
            try:
                process.send_signal(sig)
            except psutil.NoSuchProcess:
                return "released"
            except psutil.AccessDenied:
                return "unknown"
        return "signalled"
    except (FileNotFoundError, ProcessLookupError):
        return "released"
    except (PermissionError, ValueError, OSError):
        return "unknown"
    finally:
        if fd is not None:
            os.close(fd)


def live_group(group):
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            item = _stat(path.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if item["group_id"] == group and item["state"] != "Z":
            return True
    return False


def descendants(parent):
    table = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            item = _stat(path.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        table[item["pid"]] = (item["parent_pid"], item["state"])
    owned = {parent}
    while True:
        children = {pid for pid, (owner, _) in table.items() if owner in owned}
        if children.issubset(owned):
            break
        owned.update(children)
    return {pid for pid in owned if pid != parent and table[pid][1] != "Z"}


def unverifiable_processes(pids, *, proc_root=Path("/proc")):
    """Retain uncertainty until /proc proves exit; never signal these PIDs."""
    remaining = set()
    for pid in pids:
        try:
            if _stat(pid, proc_root)["state"] != "Z":
                remaining.add(pid)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (PermissionError, ValueError):
            remaining.add(pid)
    return remaining


class NodeLease:
    """An expired token cannot restart work, even before the watchdog wakes."""

    def __init__(self, token, *, timeout, on_expire, clock=time.monotonic):
        if not token or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Lease token and finite positive timeout are required")
        self.token, self.timeout, self.clock = token, timeout, clock
        self.on_expire = on_expire
        self.last_seen, self.sequence = clock(), -1
        self.expired, self.notified = False, False
        self.lock = threading.Lock()

    def heartbeat(self, token, sequence):
        with self.lock:
            if self.clock() - self.last_seen >= self.timeout:
                self.expired = True
            if (
                self.expired
                or token != self.token
                or type(sequence) is not int
                or sequence <= self.sequence
            ):
                return False
            self.last_seen, self.sequence = self.clock(), sequence
            return True

    def is_active(self):
        with self.lock:
            if self.clock() - self.last_seen >= self.timeout:
                self.expired = True
            return not self.expired

    def check(self):
        notify = False
        with self.lock:
            if self.clock() - self.last_seen >= self.timeout:
                self.expired = True
            if self.expired and not self.notified:
                self.notified, notify = True, True
        if notify:
            self.on_expire()
        return not self.expired


def _duration(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Process deadlines must be finite and positive")
    return value


def supervised_actor_runtime_env(runtime_env=None):
    """Let the actor's independent supervisor finish after Ray worker exit.

    Ray's CoreWorker kills direct children even in separate process sessions.
    Opt out only for actors whose native children all use ``start_owned``; that
    subreaper provides bounded parent-loss cleanup and persists process proof.
    This actor-local setting does not alter Raylet or other actors' policies.
    """
    result = dict(runtime_env or {})
    environment = dict(result.get("env_vars", {}))
    key = "RAY_kill_child_processes_on_worker_exit"
    if environment.get(key, "0") != "0":
        raise ValueError("Ray child killing conflicts with independent supervision")
    environment[key] = "0"
    result["env_vars"] = environment
    return result


def supervise(command):
    from deepspec.pipeline.runtime import atomic_json

    parent = int(os.environ["DEEPSPEC_ORCHESTRATOR_PID"])
    run_id = os.environ.setdefault("DEEPSPEC_PIPELINE_RUN_ID", uuid.uuid4().hex)
    own_identity = _stat(os.getpid())
    report_identity = {
        "supervisor_pid": own_identity["pid"],
        "supervisor_start_ticks": own_identity["start_ticks"],
    }
    if parent != os.getppid():
        # The driver may die between Popen and supervisor initialization. No
        # workload has started yet; persist that fact before exiting.
        path = os.environ.get("DEEPSPEC_SUPERVISOR_REPORT_PATH")
        if path:
            atomic_json(
                path,
                {
                    **report_identity,
                    "run_id": run_id,
                    "reason": "local_parent_lost",
                    "processes": [],
                    "unknown": [],
                    "unverifiable_pids": [],
                    "cleanup_complete": True,
                },
            )
        return 1
    parent_ticks = os.environ.get("DEEPSPEC_ORCHESTRATOR_START_TICKS")
    if parent_ticks is None:
        parent_ticks = _stat(parent)["start_ticks"]
    parent_ticks = int(parent_ticks)
    timeout = _duration(os.environ.get("DEEPSPEC_SUPERVISOR_TIMEOUT", "86400"))
    cleanup_timeout = _duration(
        os.environ.get("DEEPSPEC_SUPERVISOR_CLEANUP_TIMEOUT", "35")
    )
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Could not establish the process subreaper")
    process = subprocess.Popen(command, start_new_session=True)
    tracked, unverifiable = {}, set()
    interrupted = [None]
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, signal.SIGINT))
    deadline = time.monotonic() + timeout
    reason, code = "completed", None

    def observe():
        for pid in descendants(os.getpid()):
            try:
                identity = capture_process(pid, run_id)
                tracked[(pid, identity["start_ticks"])] = identity
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (ValueError, PermissionError):
                unverifiable.add(pid)

    try:
        while code is None:
            observe()
            code = process.poll()
            if code is not None:
                break
            try:
                parent_alive = (
                    os.getppid() == parent
                    and _stat(parent)["start_ticks"] == parent_ticks
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                parent_alive = False
            if not parent_alive:
                reason, code = "local_parent_lost", 1
            elif interrupted[0] is not None:
                reason, code = "cancelled", 128 + interrupted[0]
            elif time.monotonic() >= deadline:
                reason, code = "run_deadline", 124
            else:
                time.sleep(0.05)
    finally:
        cleanup_deadline = time.monotonic() + cleanup_timeout
        term_deadline = time.monotonic() + min(1, cleanup_timeout / 3)
        remaining = []
        while True:
            observe()
            remaining = []
            for identity in tracked.values():
                state = signal_process(
                    identity,
                    signal.SIGTERM
                    if time.monotonic() < term_deadline
                    else signal.SIGKILL,
                )
                if state != "released":
                    remaining.append({**identity, "release_state": state})
            # Reap adopted zombies, including grandchildren with new sessions.
            process.poll()
            while True:
                try:
                    if os.waitpid(-1, os.WNOHANG)[0] == 0:
                        break
                except ChildProcessError:
                    break
            live_unverifiable = unverifiable_processes(unverifiable)
            if (
                not remaining and not live_unverifiable
            ) or time.monotonic() >= cleanup_deadline:
                break
            time.sleep(min(0.05, max(0, cleanup_deadline - time.monotonic())))
        report = {
            **report_identity,
            "run_id": run_id,
            "reason": reason,
            "processes": list(tracked.values()),
            "unknown": remaining,
            "identity_observation_failures": sorted(unverifiable),
            "unverifiable_pids": sorted(live_unverifiable),
            "cleanup_complete": not remaining and not live_unverifiable,
        }
        path = os.environ.get("DEEPSPEC_SUPERVISOR_REPORT_PATH")
        if path:
            atomic_json(path, report)
        if not report["cleanup_complete"]:
            code = code or 1
    return code


class OwnedProcessHandle:
    def __init__(
        self, process, command, run_id, report_path, *, timeout, cleanup_timeout
    ):
        self.process, self.command = process, command
        self.report_path = Path(report_path)
        self.expires_at = time.monotonic() + timeout
        self.cleanup_timeout = cleanup_timeout
        self.identity = capture_process(process.pid, run_id)

    def poll(self):
        return self.process.poll()

    def _report(self):
        if self.report_path.exists():
            try:
                report = json.loads(self.report_path.read_text())
                if (
                    report.get("run_id") == self.identity["run_id"]
                    and report.get("supervisor_pid") == self.identity["pid"]
                    and report.get("supervisor_start_ticks")
                    == self.identity["start_ticks"]
                ):
                    return report
            except (OSError, ValueError):
                pass
        return {
            "cleanup_complete": False,
            "unknown": [self.identity],
            "reason": "missing_or_mismatched_supervisor_report",
        }

    def result(self):
        try:
            code = self.process.wait(
                timeout=max(0.001, self.expires_at - time.monotonic())
                + self.cleanup_timeout
            )
        except subprocess.TimeoutExpired as error:
            self.stop(timeout=self.cleanup_timeout)
            raise TimeoutError(
                "Owned process exceeded its run and cleanup deadlines"
            ) from error
        if code:
            raise subprocess.CalledProcessError(code, self.command)
        if not self._report()["cleanup_complete"]:
            raise RuntimeError("Owned process cleanup could not be confirmed")
        return {"exit_code": code, "cleanup": self._report()}

    def stop(self, *, timeout=None):
        timeout = self.cleanup_timeout if timeout is None else _duration(timeout)
        deadline = time.monotonic() + timeout
        if self.process.poll() is None:
            state = signal_process(self.identity, signal.SIGTERM)
            if state == "unknown":
                return {"cleanup_complete": False, "unknown": [self.identity]}
            try:
                self.process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                # Killing a supervisor cannot prove that native descendants stopped.
                signal_process(self.identity, signal.SIGKILL)
                return {
                    "cleanup_complete": False,
                    "unknown": [self.identity],
                    "reason": "cleanup_deadline",
                }
        return self._report()


def start_owned(
    command, *, env=None, timeout=86400, cleanup_timeout=35, report_path=None, **kwargs
):
    if kwargs.get("start_new_session", True) is not True:
        raise ValueError("An owned supervisor requires a separate process session")
    # Raylet kills a dead worker's process group. The parent-death watcher must
    # survive that cleanup long enough to reap native children and persist proof.
    kwargs["start_new_session"] = True
    timeout, cleanup_timeout = _duration(timeout), _duration(cleanup_timeout)
    environment = dict(os.environ if env is None else env)
    run_id = environment.setdefault("DEEPSPEC_PIPELINE_RUN_ID", uuid.uuid4().hex)
    if report_path is None:
        report_path = (
            Path(tempfile.gettempdir()) / f"deepspec-supervisor-{uuid.uuid4().hex}.json"
        )
    environment.update(
        DEEPSPEC_ORCHESTRATOR_PID=str(os.getpid()),
        DEEPSPEC_ORCHESTRATOR_START_TICKS=str(_stat(os.getpid())["start_ticks"]),
        DEEPSPEC_SUPERVISOR_TIMEOUT=str(timeout),
        DEEPSPEC_SUPERVISOR_CLEANUP_TIMEOUT=str(cleanup_timeout),
        DEEPSPEC_SUPERVISOR_REPORT_PATH=str(report_path),
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "deepspec.orchestration.process", *command],
        env=environment,
        **kwargs,
    )
    return OwnedProcessHandle(
        process,
        command,
        run_id,
        report_path,
        timeout=timeout,
        cleanup_timeout=cleanup_timeout,
    )


def run_owned(command, *, env=None, timeout=86400, cleanup_timeout=35, **kwargs):
    handle = start_owned(
        command, env=env, timeout=timeout, cleanup_timeout=cleanup_timeout, **kwargs
    )
    try:
        handle.result()
    except BaseException:
        handle.stop()
        raise


if __name__ == "__main__":
    sys.exit(supervise(sys.argv[1:]))

"""Supervise owned descendants, including loss of the orchestrator."""

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def live_group(group):
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group and fields[0] != "Z":
            return True
    return False


def descendants(parent):
    table = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        table[int(path.name)] = (int(fields[1]), fields[0])
    owned = {parent}
    while True:
        children = {pid for pid, (owner, _) in table.items() if owner in owned}
        if children.issubset(owned):
            break
        owned.update(children)
    return {pid for pid in owned if pid != parent and table[pid][1] != "Z"}


def supervise(command):
    parent = int(os.environ["DEEPSPEC_ORCHESTRATOR_PID"])
    # Elastic workers create separate sessions. A Linux subreaper keeps orphaned
    # descendants attached to this supervisor until they are stopped and reaped.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Could not establish the process subreaper")
    process = subprocess.Popen(command, start_new_session=True)
    stopped = threading.Event()
    stop_deadline = None

    def terminate(sig=signal.SIGTERM):
        for pid in descendants(os.getpid()):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def watch():
        while not stopped.wait(0.2):
            if os.getppid() != parent or (
                stop_deadline is not None and time.monotonic() >= stop_deadline
            ):
                terminate(signal.SIGKILL)
                return

    def interrupted(signum, frame):
        nonlocal stop_deadline
        stop_deadline = time.monotonic() + 5
        terminate(signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        return process.wait()
    finally:
        terminate()
        deadline = time.monotonic() + 5
        while descendants(os.getpid()) and time.monotonic() < deadline:
            time.sleep(0.1)
        terminate(signal.SIGKILL)
        process.wait()
        deadline = time.monotonic() + 30
        while descendants(os.getpid()):
            if time.monotonic() >= deadline:
                raise RuntimeError("Owned descendant processes have not exited")
            terminate(signal.SIGKILL)
            time.sleep(0.1)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        stopped.set()
        watcher.join()


def run_owned(command, *, env=None, **kwargs):
    environment = dict(os.environ if env is None else env)
    environment["DEEPSPEC_ORCHESTRATOR_PID"] = str(os.getpid())
    supervisor = subprocess.Popen(
        [sys.executable, "-m", "deepspec.orchestration.process", *command],
        env=environment,
        **kwargs,
    )
    try:
        code = supervisor.wait()
    except BaseException:
        supervisor.terminate()
        supervisor.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


if __name__ == "__main__":
    sys.exit(supervise(sys.argv[1:]))

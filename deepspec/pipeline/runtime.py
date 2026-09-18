"""Lifecycle helpers for a run-owned Mooncake master process."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


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
            with socket.create_connection((host, port), timeout=min(1.0, poll_interval + 0.5)):
                return
        except OSError:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"Mooncake master exited before becoming reachable: {process.returncode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Mooncake master did not become reachable at {endpoint}")
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
    ):
        self.endpoint = endpoint
        self.log_path = Path(log_path)
        self.metrics_port = int(metrics_port)
        self.ttl_seconds = int(ttl_seconds)
        self.env = dict(os.environ if env is None else env)
        self.process = None
        self.log = None

    def start(self, *, timeout=30):
        if self.process is not None:
            return self
        import mooncake

        _host, port = endpoint_parts(self.endpoint)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        command = [
            sys.executable,
            "-m",
            "deepspec.orchestration.process",
            str(Path(mooncake.__file__).parent / "mooncake_master"),
            f"--rpc_port={port}",
            f"--metrics_port={self.metrics_port}",
            f"--default_kv_lease_ttl={self.ttl_seconds}s",
            "--enable_offload=false",
            "--enable_disk_eviction=false",
        ]
        self.env.setdefault("DEEPSPEC_ORCHESTRATOR_PID", str(os.getpid()))
        try:
            self.process = subprocess.Popen(
                command,
                env=self.env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            self.log.close()
            self.log = None
            raise
        try:
            wait_for_endpoint(self.endpoint, process=self.process, timeout=timeout)
        except BaseException:
            self.stop()
            raise
        return self

    def poll(self):
        return None if self.process is None else self.process.poll()

    def stop(self):
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None
            self.process = None

    def __enter__(self):
        return self.start()

    def __exit__(self, _type, _value, _traceback):
        self.stop()

"""Durable metadata and GPU resource observations for phase orchestration."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_idle(devices):
    deadline = time.monotonic() + 30
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                ",".join(map(str, devices)),
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        if not result.stdout.strip():
            return {
                "devices": devices,
                "compute_processes": [],
                "observed_at": time.time(),
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU pool is still occupied: {result.stdout.strip()}")
        time.sleep(0.2)

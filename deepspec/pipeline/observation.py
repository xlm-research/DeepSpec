"""Independent node sampling, outside model/Store RPC executors."""

import subprocess
import threading
from pathlib import Path

from .runtime import EventWriter


class ResourceSampler:
    def __init__(self, plan, node_id, output):
        self.plan, self.node_id = plan, node_id
        self.events = EventWriter(
            Path(output) / f"events/observer-{node_id}.jsonl",
            run_id=plan["run_id"],
            plan_hash=plan["plan_hash"],
            sender_identity={
                "component": "independent_node_sampler",
                "node_id": node_id,
            },
        )
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="node-resource-observer"
        )
        self.thread.start()

    def _sample(self):
        from .cluster import gpu_processes, process_memory
        from .memory import node_memory

        data, missing = {}, {}
        for field, sample in (
            ("memory", node_memory),
            ("processes", lambda: process_memory(self.plan["run_id"])),
            ("gpu_processes", lambda: gpu_processes(self.plan["run_id"])),
        ):
            for attempt in range(3):
                try:
                    data[field] = sample()
                    break
                except Exception as error:  # noqa: BLE001 -- failed observations remain explicit missing evidence
                    # GPU teardown can briefly block nvidia-smi. Retry only its
                    # bounded 10-second timeout, retaining every failed attempt.
                    # A persistent failure still creates missing evidence.
                    if (
                        field == "gpu_processes"
                        and isinstance(error, subprocess.TimeoutExpired)
                        and attempt < 2
                    ):
                        data.setdefault("sample_retries", {}).setdefault(
                            field, []
                        ).append(repr(error))
                        continue
                    data[field], missing[field] = None, repr(error)
                    break
        self.events.emit("resource_sample", data, basis="observed", missing=missing)

    def _run(self):
        try:
            while True:
                self._sample()
                if self.stopped.wait(1):
                    self._sample()
                    return
        finally:
            self.events.close()

    def stop(self, *, timeout):
        self.stopped.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError("Independent node sampler did not stop")

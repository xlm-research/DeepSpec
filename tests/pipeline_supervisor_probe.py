"""Minimal real Ray worker-exit regression; CPU only, exact-owned cleanup.

The interactive mode retains one owned zero-GPU Ray cluster for fast diagnosis.
Feed JSON requests on stdin; an empty line ends the session and stops that cluster.
"""

import argparse
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path


class SupervisorHost:
    owns_supervised_processes = True

    def launch(self, directory, run_id):
        from deepspec.orchestration.process import capture_process, start_owned

        ready = Path(directory) / "child.json"
        script = (
            "import os,json,time; from pathlib import Path; "
            "from deepspec.orchestration.process import capture_process; "
            f"Path({str(ready)!r}).write_text(json.dumps(capture_process(os.getpid(), {run_id!r}))); "
            "time.sleep(60)"
        )
        self.handle = start_owned(
            [sys.executable, "-c", script],
            timeout=30,
            cleanup_timeout=3,
            report_path=Path(directory) / "cleanup.json",
        )
        return {
            "supervisor": self.handle.identity,
            "actor": capture_process(os.getpid(), run_id),
        }


def run_case(ray, root, *, kill_children=None, use_allocator=True):
    from deepspec.orchestration.process import signal_process
    from deepspec.pipeline.runtime import atomic_json

    run_id = uuid.uuid4().hex
    directory = Path(root).resolve() / run_id
    directory.mkdir(parents=True)
    environment = {"DEEPSPEC_PIPELINE_RUN_ID": run_id, "CUDA_VISIBLE_DEVICES": ""}
    if kill_children is not None:
        environment["RAY_kill_child_processes_on_worker_exit"] = str(int(kill_children))
    options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "max_restarts": 0,
        "runtime_env": {"env_vars": environment},
    }
    if use_allocator:
        from deepspec.pipeline.controller import ActorAllocator, ResourceRegistry
        from deepspec.pipeline.runtime import Deadline

        plan = {
            "run_id": run_id,
            "plan_hash": "cpu-supervisor-probe",
            "config": {"output_dir": str(directory)},
        }
        allocator = ActorAllocator(
            plan, registry=ResourceRegistry(run_id, plan["plan_hash"])
        )
        actor = allocator.create(
            "supervisor-host",
            SupervisorHost,
            node_id=ray.get_runtime_context().get_node_id(),
            role="process-probe",
            deadline=Deadline.after(30),
            options=options,
        )
    else:
        actor = ray.remote(SupervisorHost).options(**options).remote()
    identities, child, result = (
        {},
        None,
        {
            "run_id": run_id,
            "kill_children": kill_children,
            "use_allocator": use_allocator,
        },
    )
    try:
        identities = ray.get(actor.launch.remote(str(directory), run_id), timeout=30)
        until = time.monotonic() + 5
        while not (directory / "child.json").exists() and time.monotonic() < until:
            time.sleep(0.01)
        child = json.loads((directory / "child.json").read_text())
        started = time.monotonic()
        ray.kill(actor, no_restart=True)
        until = time.monotonic() + 5
        while not (directory / "cleanup.json").exists() and time.monotonic() < until:
            time.sleep(0.01)
        report = (
            json.loads((directory / "cleanup.json").read_text())
            if (directory / "cleanup.json").exists()
            else {}
        )
        result.update(
            passed=report.get("cleanup_complete") is True
            and signal_process(child, 0) == "released",
            report=report,
            identities=identities,
            child=child,
            seconds=time.monotonic() - started,
        )
    finally:
        ray.kill(actor, no_restart=True)
        for identity in (identities.get("supervisor"), child):
            if identity is not None:
                signal_process(identity, signal.SIGTERM)
        until = time.monotonic() + 4
        while (
            child is not None
            and signal_process(child, 0) != "released"
            and time.monotonic() < until
        ):
            time.sleep(0.01)
        if child is not None:
            result["rescue"] = signal_process(child, signal.SIGKILL)
        atomic_json(directory / "probe.json", result)
    return result


def main():
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        namespace="supervisor-diagnosis-" + uuid.uuid4().hex,
        log_to_driver=False,
        object_store_memory=100 * 1024**2,
    )
    try:
        if args.interactive:
            print("supervisor probe ready", flush=True)
            for line in sys.stdin:
                if not line.strip():
                    break
                print(
                    json.dumps(run_case(ray, args.output, **json.loads(line))),
                    flush=True,
                )
        else:
            result = run_case(ray, args.output)
            print(json.dumps(result))
            if not result["passed"]:
                raise SystemExit(1)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

"""Single-node TP4 + TP4 debugging, with a separately verified large CPU pool.

Use debug_single_node.sh to select Python and configure CUDA runtime libraries.
The default 1024 GiB means 1 TiB of Store capacity, not a process RSS limit.
The probe writes one 4K synthetic sample; it does not test a full pool.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
GIB = 1024**3


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def snapshot():
    from deepspec.pipeline.memory import node_memory

    return {"time": time.time(), **node_memory()}


def marked_processes(marker):
    import psutil

    result = []
    for process in psutil.process_iter():
        try:
            if (
                process.pid != os.getpid()
                and process.status() != psutil.STATUS_ZOMBIE
                and process.environ().get("DEEPSPEC_DEBUG_SESSION") == marker
            ):
                result.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def spawn(command, log, env):
    # The existing subreaper supervises private Ray/torchrun descendants too.
    child_env = dict(env, DEEPSPEC_ORCHESTRATOR_PID=str(os.getpid()))
    print("Launch:", shlex.join(command), flush=True)
    with log.open("w") as stream:
        return subprocess.Popen(
            [sys.executable, "-m", "deepspec.orchestration.process", *command],
            cwd=ROOT,
            env=child_env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def wait(process, timeout, directory, label, ready=None, companions=()):
    deadline = time.monotonic() + timeout
    next_report = 0
    while True:
        for companion in companions:
            require(companion.poll() is None, f"{label}: a service exited")
        code = process.poll()
        if code is not None:
            require(code == 0, f"{label} exited {code}; see {directory}")
            require(ready is None, f"{label} exited before publishing readiness")
            return
        if ready is not None and ready.exists():
            return
        require(time.monotonic() < deadline, f"{label} timed out after {timeout}s")
        memory = snapshot()
        with (directory / "memory.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": label, **memory}) + "\n")
        if time.monotonic() >= next_report:
            print(
                f"{label}: running; headroom={memory['headroom_bytes'] / GIB:.1f} GiB",
                flush=True,
            )
            next_report = time.monotonic() + 30
        time.sleep(2)


def probe_worker(role, directory):
    import torch

    from deepspec.pipeline.store import (
        FIELDS,
        TensorStore,
        describe_tensors,
        object_keys,
    )

    config = read(directory / "config.json")
    started = time.monotonic()
    before = snapshot()
    store = TensorStore(
        config["store"], pool_bytes=config["pool_bytes"] if role == "owner" else 0
    )
    try:
        if role == "owner":
            write(
                directory / "owner-ready.json",
                {
                    "pool_bytes": config["pool_bytes"],
                    "endpoint": store.endpoint,
                    "setup_seconds": time.monotonic() - started,
                    "before": before,
                    "after": snapshot(),
                },
            )
            while not (directory / "stop-owner").exists():
                time.sleep(0.2)
        elif role == "writer":
            torch.manual_seed(0)
            length = 4096
            tensors = {
                "input_ids": torch.arange(length).reshape(1, -1),
                "loss_mask": torch.ones(1, length, dtype=torch.bool),
                "seq_len": torch.tensor([length]),
                "context_chunk_len": torch.tensor([length]),
                "target_hidden_states": torch.randn(
                    1, length, 25600, dtype=torch.bfloat16
                ),
                "target_last_hidden_states": torch.randn(
                    1, length, 5120, dtype=torch.bfloat16
                ),
            }
            fields = describe_tensors("large-pool-probe/sample-0", tensors)
            store.put(fields, tensors)
            write(directory / "descriptor.json", fields)
            write(directory / "writer-result.json", store.last_write)
        else:
            fields = read(directory / "descriptor.json")
            tensors = store.get(fields, FIELDS, device="cpu", verify=True)
            require(set(tensors) == set(FIELDS), "Probe fields are incomplete")
            store.remove(fields)
            require(
                all(store.client.is_exist(key) == 0 for key in object_keys(fields)),
                "Probe objects survived explicit deletion",
            )
            write(
                directory / "reader-result.json",
                {**store.last_read, "deleted_chunks": len(object_keys(fields))},
            )
    finally:
        store.close()


def pool_probe(args, output, env):
    import mooncake

    from deepspec.pipeline.store import free_port

    directory = output / "pool-probe"
    directory.mkdir()
    port, metrics = free_port(), free_port()
    while metrics == port:
        metrics = free_port()
    config = {
        "pool_bytes": args.pool_gib * GIB,
        "store": {
            "host": "127.0.0.1",
            "master": f"127.0.0.1:{port}",
            "protocol": "tcp",
        },
    }
    write(directory / "config.json", config)
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES="")
    processes = []
    try:
        master = spawn(
            [
                str(Path(mooncake.__file__).parent / "mooncake_master"),
                "--rpc_address=127.0.0.1",
                f"--rpc_port={port}",
                f"--metrics_port={metrics}",
                "--default_kv_lease_ttl=300s",
                "--enable_offload=false",
                "--enable_disk_eviction=false",
            ],
            directory / "master.log",
            cpu_env,
        )
        processes.append(master)
        deadline = time.monotonic() + 30
        while True:
            require(master.poll() is None, "Probe master exited")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                require(time.monotonic() < deadline, "Probe master startup timed out")
                time.sleep(0.2)
        for role in ("owner", "writer", "reader"):
            process = spawn(
                [
                    sys.executable,
                    "-u",
                    str(SCRIPT),
                    "--probe-worker",
                    role,
                    "--output",
                    str(directory),
                ],
                directory / f"{role}.log",
                cpu_env,
            )
            processes.append(process)
            wait(
                process,
                args.probe_timeout,
                directory,
                f"pool/{role}",
                ready=directory / "owner-ready.json" if role == "owner" else None,
                companions=tuple(processes[:-1]),
            )
            if role != "owner":
                processes.remove(process)
        result = {
            "owner": read(directory / "owner-ready.json"),
            "writer": read(directory / "writer-result.json"),
            "reader": read(directory / "reader-result.json"),
            "full_capacity_stress_test": False,
        }
        require(result["reader"]["verified"], "Probe did not verify SHA256")
        require(
            result["writer"]["nbytes"] == result["reader"]["nbytes"],
            "Probe byte counts differ",
        )
        (directory / "stop-owner").touch()
        wait(processes[1], 60, directory, "pool/close", companions=(master,))
        # Training currently waits only 60 seconds for FeatureBuffer construction.
        require(
            result["owner"]["setup_seconds"] < 45,
            "Pool setup needs more startup time; review run.py's 60s actor timeout",
        )
        write(directory / "result.json", result)
        return result
    finally:
        for process in reversed(processes):
            stop(process)


def training_command(args, output, context):
    return [
        sys.executable,
        "-u",
        "-m",
        "deepspec.pipeline.run",
        "--model",
        str(args.model),
        "--source",
        str(args.source),
        "--output",
        str(output),
        "--context-length",
        str(context),
        "--steps",
        str(args.steps),
        "--pool-gib",
        str(args.pool_gib),
        "--window",
        str(args.window),
        "--producer-batch-size",
        str(args.producer_batch_size),
        "--writer-inflight",
        str(args.writer_inflight),
        "--epochs",
        str(args.epochs),
        "--pool-utilization",
        str(0.99 if args.stage == "peak" else 0.75),
        *(["--retain-for-peak"] if args.stage == "peak" else []),
        "--producer-dp",
        "1",
        "--consumer-dp",
        "1",
        "--protocol",
        "tcp",
        "--receive-device",
        "cpu",
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]


def verify_training(directory):
    import torch
    import torch.distributed.checkpoint as dcp

    from deepspec.pipeline.run import summarize_events

    config, result = read(directory / "pipeline.json"), read(directory / "result.json")
    plan = read(directory / "inputs/input-plan.json")
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]
    samples, steps = config["samples"], config["steps"]
    require(not (directory / "failure.json").exists(), "Launcher recorded a failure")
    require(len(samples) == steps * 4, "Wrong sample count")
    require(len({s["sample_id"] for s in samples}) == len(samples), "Repeated samples")
    require(
        {s["epoch"] for s in samples}.issubset(set(range(config.get("epochs", 1)))),
        "Unexpected input epoch",
    )
    require(
        {s["length"] for s in samples} == {config["context_length"]}, "Wrong lengths"
    )
    require(
        plan["data_parallel_size"] == 1 and plan["gradient_accumulation_steps"] == 4,
        "Consumer layout changed",
    )
    summary = summarize_events(directory / "events.jsonl", config)
    require(
        json.loads(json.dumps(summary)) == result["events"], "Event summary differs"
    )
    require(summary["verified_rank_receives"] == len(samples) * 4, "Wrong read count")
    require(
        result["buffer"]["produced"] == result["buffer"]["released"] == len(samples)
        and result["buffer"]["remaining"] == 0,
        "Pool did not drain",
    )
    batches = [e["batch_size"] for e in events if e["event"] == "inference_batch"]
    require(sum(batches) == len(samples), "Missing batched inference requests")
    require(
        max(batches) == min(config["producer_batch_size"], len(samples)),
        "Requested producer batch size was never exercised",
    )
    resident = peak = 0
    for event in events:
        if event["event"] == "ready":
            resident += samples[event["position"]]["nbytes"]
            peak = max(peak, resident)
        elif event["event"] == "released":
            resident -= samples[event["position"]]["nbytes"]
        if "resident_bytes" in event:
            require(
                event["resident_bytes"] == resident, "Resident byte accounting differs"
            )
    require(
        resident == 0 and peak == result["buffer"]["peak_resident_bytes"],
        "Resident peak does not match READY/delete events",
    )
    if config.get("retain_for_peak"):
        require(
            peak >= config["retain_until_bytes"] >= config["pool_bytes"] * 0.95,
            "Requested CPU pool peak was not reached",
        )
        require(
            sum(e["event"] == "pool_peak_reached" for e in events) == 1,
            "Missing one-time peak release",
        )
    for rank in range(4):
        for kind in (
            "claimed",
            "received",
            "gpu_ready",
            "compute_start",
            "compute_end",
        ):
            require(
                [
                    e["position"]
                    for e in events
                    if e["event"] == kind and e["reader"] == rank
                ]
                == list(range(len(samples))),
                f"Rank {rank} has wrong {kind} order",
            )
    for position in range(len(samples)):
        acks = [
            e for e in events if e["event"] == "received" and e["position"] == position
        ]
        releases = [
            e for e in events if e["event"] == "released" and e["position"] == position
        ]
        require(
            len(releases) == 1
            and releases[0]["monotonic"] >= max(e["monotonic"] for e in acks),
            "Object released before all readers acknowledged",
        )
    gradients = [e for e in events if e["event"] == "context_gradient_verified"]
    require(
        all(math.isfinite(v) and v > 0 for e in gradients for v in e["norms"].values()),
        "Invalid context gradient",
    )
    reads = [e for e in events if e["event"] == "transfer_end"]
    require(
        len(reads) == len(samples) * 4 and all(e["store"]["verified"] for e in reads),
        "Missing transfer checksums",
    )
    log = re.sub(r"\x1b\[[0-9;]*m", "", (directory / "consumer.log").read_text())
    losses = re.findall(r"step:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)", log)
    require(
        Counter(int(s) for s, _, _ in losses)
        == Counter({s: 4 for s in range(1, steps + 1)}),
        "Missing rank update logs",
    )
    require(
        all(math.isfinite(float(v)) for _, loss, norm in losses for v in (loss, norm)),
        "Nonfinite training metrics",
    )
    commit = result["consumer"]["commit"]
    checkpoint = Path(commit["checkpoint"])
    require(
        read(checkpoint / "commit.json") == commit
        and commit["completed_updates"] == steps
        and commit["next_global_microbatch"] == len(samples),
        "Wrong checkpoint cursor",
    )
    require(
        hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest()
        == commit["metadata_sha256"],
        "Checkpoint metadata hash mismatch",
    )
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    shards = set()
    for storage in metadata.storage_data.values():
        shard = checkpoint / storage.relative_path
        require(
            0
            <= storage.offset
            <= storage.offset + storage.length
            <= shard.stat().st_size,
            "Checkpoint shard is truncated",
        )
        shards.add(shard.name)
    require(len(shards) == 4, "Expected four checkpoint shards")
    state = {"optimizer": {"state": {"fc.weight": {"step": torch.zeros(())}}}}
    dcp.load(state, checkpoint_id=checkpoint)
    optimizer_step = state["optimizer"]["state"]["fc.weight"]["step"].item()
    require(optimizer_step == steps, "Optimizer step mismatch")
    for name, digest in read(directory / "environment.json")["source_sha256"].items():
        require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
            f"Source changed during training: {name}",
        )
    report = {
        "verified": True,
        "samples": len(samples),
        "steps": steps,
        "fc_optimizer_step": optimizer_step,
        "events": summary,
        "pool_bytes": config["pool_bytes"],
        "window": config["window"],
        "peak_reserved_bytes": result["buffer"]["peak_reserved_bytes"],
        "peak_resident_bytes": peak,
        "peak_pool_utilization": peak / config["pool_bytes"],
        "inference_batch_sizes": batches,
        "epochs_used": sorted({s["epoch"] for s in samples}),
        "retained_for_capacity_stress": config.get("retain_for_peak", False),
        "metadata_sha256": commit["metadata_sha256"],
    }
    write(directory / "verification.json", report)
    return report


def preflight(args):
    from mooncake.store import MooncakeDistributedStore  # noqa: F401

    from deepspec.pipeline.memory import feature_budget
    from deepspec.pipeline.run import require_idle_gpus
    from deepspec.trainer.qwen3_8_vllm import teacher_identity

    require(args.source.is_file(), f"Missing source: {args.source}")
    teacher = teacher_identity(str(args.model), [1, 16, 31, 46, 61])
    contexts = [4096] if args.stage == "4k" else [4096, 131072]
    memory = snapshot()
    budgets = {}
    for length in contexts:
        # Two int64 token/mask input arrays plus the two int64 scalar fields.
        sample_bytes = length * (5120 * 6 * 2 + 16) + 16
        budgets[str(length)] = feature_budget(
            args.pool_gib * GIB,
            sample_bytes,
            args.window,
            4,
            4,
            snapshot=memory,
            writer_inflight=args.writer_inflight,
        )
    return {
        "python": sys.executable,
        "hostname": socket.gethostname(),
        "gpu_inventory": require_idle_gpus(),
        "teacher": teacher,
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "vllm",
                "ray",
                "mooncake-transfer-engine",
                "transformers",
                "numpy",
            )
        },
        "budgets": budgets,
        "memory": memory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("check", "pool", "4k", "128k", "peak", "all"), default="all"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "outputs/dspark_torchtitan_orchestration_20260914/128k-source.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pool-gib", type=int, default=1024, help="Store pool only; 1024 GiB = 1 TiB"
    )
    parser.add_argument("--window", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--producer-batch-size", type=int, default=20)
    parser.add_argument("--writer-inflight", type=int, default=2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--probe-timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--probe-worker", choices=("owner", "writer", "reader"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.probe_worker:
        probe_worker(args.probe_worker, args.output)
        return
    args.window = (
        args.window
        if args.window is not None
        else (140 if args.stage == "peak" else 40)
    )
    args.steps = (
        args.steps if args.steps is not None else (35 if args.stage == "peak" else 5)
    )
    args.epochs = (
        args.epochs if args.epochs is not None else (2 if args.stage == "peak" else 1)
    )
    args.timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else (10800 if args.stage == "peak" else 3600)
    )
    if min(
        args.pool_gib,
        args.steps,
        args.timeout_seconds,
        args.probe_timeout,
        args.producer_batch_size,
        args.writer_inflight,
        args.epochs,
    ) <= 0 or args.window < max(4, args.producer_batch_size):
        parser.error(
            "Sizes/timeouts/steps must be positive; window must hold a producer batch and 4 consumers"
        )
    args.model, args.source = args.model.resolve(), args.source.resolve()
    output = (
        args.output
        or ROOT
        / "outputs"
        / f"dspark_single_debug_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    ).resolve()
    stages = ("pool", "4k", "128k") if args.stage == "all" else (args.stage,)
    print(
        f"Output: {output}\nPool: {args.pool_gib} GiB; window: {args.window}; stages: {stages}",
        flush=True,
    )
    if args.dry_run:
        for stage in stages:
            if stage in ("4k", "128k", "peak"):
                print(
                    shlex.join(
                        training_command(
                            args, output / stage, 4096 if stage == "4k" else 131072
                        )
                    )
                )
            else:
                print(
                    f"{stage}: preflight"
                    + (
                        " + separate CPU owner/writer/reader probe"
                        if stage == "pool"
                        else ""
                    )
                )
        return
    output.mkdir(parents=True, exist_ok=False)
    marker = "dspark-debug-" + uuid.uuid4().hex[:12]
    env = dict(os.environ, DEEPSPEC_DEBUG_SESSION=marker)
    report = {"session": marker, "status": "running", "stages": {}}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        write(output / "preflight.json", preflight(args))
        print("Preflight passed", flush=True)
        for stage in stages:
            if stage == "check":
                report["stages"][stage] = {"verified": True}
            elif stage == "pool":
                report["stages"][stage] = pool_probe(args, output, env)
            else:
                from deepspec.pipeline.run import require_idle_gpus

                require_idle_gpus()
                command = training_command(
                    args, output / stage, 4096 if stage == "4k" else 131072
                )
                process = spawn(command, output / f"{stage}-launcher.log", env)
                try:
                    wait(process, args.timeout_seconds + 1200, output, stage)
                finally:
                    stop(process)
                report["stages"][stage] = verify_training(output / stage)
                require_idle_gpus()
            print(f"PASS: {stage}", flush=True)
            write(output / "debug-result.json", report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        import psutil

        residual = marked_processes(marker)
        for process in residual:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(residual, timeout=5)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
        remaining = [p.pid for p in marked_processes(marker)]
        report["cleanup"] = {
            "terminated_residual_pids": [p.pid for p in residual],
            "remaining_pids": remaining,
        }
        if remaining:
            report["status"] = "failed"
        write(output / "debug-result.json", report)
        print(f"Debug {report['status']}: {output / 'debug-result.json'}", flush=True)
        require(not remaining, f"Debug processes survived cleanup: {remaining}")


if __name__ == "__main__":
    main()

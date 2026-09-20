"""Append-only acceptance runs, with separate evidence levels and explicit gaps.

The initial CLI inventories cases only. Later probes register executors through
the Python API; an absent executor always produces not_run, never passed.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

NORMAL_CASES = (
    "m0-4k",
    "m0-128k",
    "m1-11-4k",
    "m1-11-128k",
    "m1-12-4k",
    "m1-21-4k",
    "m1-22-4k",
    "m1-22-128k",
    "m2-dp1-4k",
    "m2-dp2-4k",
    "m2-dp2-128k",
    "m3-4k",
    "m3-128k",
)
FAULT_CASES = (
    "m3-rank-failure",
    "m3-node-unreachable",
    "m2-128k-slow-reader",
    "m3-128k-slow-reader",
    "m2-128k-delete-failure",
    "m3-128k-headroom",
    "partial-allocation",
    "initialization-failure",
    "cancel-producing",
    "cancel-reading",
    "cancel-draining",
    "driver-sigkill",
)
PROBE_CASES = (
    "collective-probe",
    "allocation-probe",
    "execution-contracts",
    "capacity-contracts",
    "verification-contracts",
    "local-store-probe",
    "lifecycle-probe",
)
STATUSES = {"planned", "not_run", "blocked", "failed", "passed"}
LEVELS = {
    "inventory",
    "cpu_contract",
    "cpu_process",
    "ray_allocation",
    "transport",
    "training",
}


def _atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def record_result(root, case, *, status, evidence_level, reason=None, evidence=None):
    if case not in NORMAL_CASES + FAULT_CASES + PROBE_CASES:
        raise ValueError(f"Unknown acceptance case: {case}")
    if status not in STATUSES or evidence_level not in LEVELS:
        raise ValueError("Unknown status or evidence level")
    evidence = {} if evidence is None else evidence
    if status == "passed":
        if evidence_level == "inventory" or evidence.get("executed") is not True:
            raise ValueError("Unexecuted cases cannot pass")
        required = [
            "run_id",
            "started_at",
            "ended_at",
            "checks",
            "artifacts",
        ]
        identity = evidence.get("plan_hash") or (
            evidence.get("test_suite_hash")
            if case in PROBE_CASES and evidence_level != "training"
            else None
        )
        if (
            not identity
            or any(not evidence.get(k) for k in required)
            or any(v is not True for v in evidence["checks"].values())
        ):
            raise ValueError(
                "Passing evidence must include identities and successful checks"
            )
        if evidence_level == "training":
            required = [
                "actual_lengths",
                "layout",
                "environment",
                "verification",
                "cleanup",
                "rank_events",
            ]
            if any(not evidence.get(k) for k in required):
                raise ValueError(
                    "Training acceptance needs lengths, layout, rank, verification and cleanup evidence"
                )
        if any(not Path(p).is_file() for p in evidence["artifacts"]):
            raise ValueError("Evidence artifacts must exist")
    elif not reason:
        raise ValueError("Non-passing results require an explicit reason")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    record_id = uuid.uuid4().hex
    directory = root / case / record_id
    directory.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "record_id": record_id,
        "case": case,
        "status": status,
        "evidence_level": evidence_level,
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": evidence.get("run_id"),
        "plan_hash": evidence.get("plan_hash"),
        "test_suite_hash": evidence.get("test_suite_hash"),
        "actual_lengths": evidence.get("actual_lengths"),
        "layout": evidence.get("layout"),
        "started_at": evidence.get("started_at"),
        "ended_at": evidence.get("ended_at"),
        "verification": evidence.get("verification"),
        "cleanup": evidence.get("cleanup"),
        "evidence": evidence,
    }
    _atomic_json(directory / "result.json", result)
    with (root / ".index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = root / "results.json"
        index = (
            json.loads(index_path.read_text())
            if index_path.exists()
            else {"schema_version": 1, "runs": []}
        )
        index["runs"].append({**result, "result_path": str(directory / "result.json")})
        _atomic_json(index_path, index)
    return result


def execute_capacity_contracts(root):
    """CPU test evidence, including failure timing; no models or Ray cluster."""
    return execute_contract_tests(
        root,
        "capacity-contracts",
        [
            "tests/test_pipeline_memory.py",
            "tests/test_pipeline_buffer.py",
            "tests/test_pipeline_writer.py",
            "tests/test_pipeline_store.py",
            "tests/test_mooncake_transport.py",
        ],
        selection="cpu_contract or not test_pipeline_store",
    )


def execute_verification_contracts(root):
    return execute_contract_tests(
        root,
        "verification-contracts",
        [
            "tests/test_pipeline_verification.py",
            "tests/test_pipeline_training.py",
            "tests/test_pipeline_foundations.py",
            "tests/test_pipeline_acceptance.py",
        ],
    )


def execute_execution_contracts(root):
    return execute_contract_tests(
        root,
        "execution-contracts",
        [
            *map(str, sorted(Path("tests").glob("test_pipeline_*.py"))),
            "tests/test_metrics.py",
        ],
        selection="not async_packed",
    )


def execute_training_plan(root, case, plan_path):
    """Record only newly executed native runs; historical results are not reused."""
    from deepspec.pipeline.cli import load_run
    from deepspec.pipeline.execution import run_plan

    output, plan, _ = load_run(Path(plan_path).parent)
    layout = case.split("-")[0].upper()
    context = 131072 if case.endswith("128k") else 4096
    expected_dp = (1, 1)
    if layout == "M1":
        expected_dp = tuple(int(d) for d in case.split("-")[1])
    elif layout == "M2":
        expected_dp = (int(case.split("-")[1].removeprefix("dp")), 2)
    elif layout == "M3":
        expected_dp = (1, 2)
    actual_dp = (plan["config"]["inference"]["dp"], plan["config"]["training"]["dp"])
    if (
        case not in NORMAL_CASES
        or actual_dp != expected_dp
        or plan["layout"] != layout
        or plan["config"]["data"]["context_length"] != context
    ):
        raise ValueError("Acceptance case differs from the frozen topology/context")
    if plan["counts"]["optimizer_steps"] < 3 or any(
        s["length"] != context for s in plan["samples"]
    ):
        raise ValueError(
            "Normal acceptance requires at least three updates and full-length actual inputs"
        )
    index = json.loads((Path(root) / "results.json").read_text())["runs"]
    if not any(
        r["case"] == "execution-contracts" and r["status"] == "passed" for r in index
    ):
        raise ValueError("Native execution/evidence CPU contracts must pass first")
    probes = [json.loads(p.read_text()) for p in output.glob("probes/*/result.json")]
    if not any(
        p.get("evidence_level") == "ray_allocation"
        and p.get("status") == "passed"
        and p.get("plan_hash") == plan["plan_hash"]
        and not p.get("injected_partial")
        for p in probes
    ):
        raise ValueError(
            "This plan requires a successful real allocation probe before training"
        )
    started, reason = datetime.now(timezone.utc).isoformat(), None
    try:
        status = run_plan(plan_path)
    except Exception as error:  # noqa: BLE001 -- preserve failed attempts in the append-only index
        reason = repr(error)
        status = json.loads((output / "status.json").read_text())
    verification = (
        json.loads((output / "verification.json").read_text())
        if (output / "verification.json").exists()
        else {}
    )
    passed = (
        status.get("state") == "succeeded"
        and verification.get("verified") is True
        and verification.get("independent") is True
    )
    events = sorted((output / "events").glob("training-*.jsonl"))
    artifacts = [
        output / name
        for name in (
            "plan.json",
            "status.json",
            "environment.json",
            "verification.json",
            "actual-placement.json",
            "source-release.json",
            "resource-cleanup.json",
        )
    ]
    return record_result(
        root,
        case,
        status="passed" if passed else "failed",
        evidence_level="training",
        reason=None
        if passed
        else reason or str(status.get("reason") or "Independent verification failed"),
        evidence={
            "executed": True,
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "native_run": passed,
                "cleanup": status.get("cleanup_complete") is True,
            },
            "actual_lengths": [s["length"] for s in plan["samples"]],
            "layout": plan["layout"],
            "environment": json.loads((output / "environment.json").read_text()),
            "verification": verification,
            "cleanup": status.get("cleanup"),
            "rank_events": [str(p) for p in events],
            "artifacts": [str(p) for p in artifacts + events if p.is_file()],
        },
    )


def execute_contract_tests(root, case, tests, *, selection=None):
    run_id = uuid.uuid4().hex
    directory = Path(root).resolve() / "cpu-contracts" / case / run_id
    directory.mkdir(parents=True, exist_ok=False)
    paths = sorted(
        set(Path("deepspec/pipeline").rglob("*.py"))
        | {Path(p) for p in tests}
        | set(Path("tests").glob("pipeline_*probe.py"))
        | set(Path("torchtitan/torchtitan/models/dspark_draft").rglob("*.py"))
        | set(Path("torchtitan/torchtitan/components/checkpointer").rglob("*.py"))
        | set(Path("torchtitan/torchtitan/components/optimizer").rglob("*.py"))
        | {
            Path(__file__).resolve(),
            Path("deepspec/orchestration/process.py"),
            Path("deepspec/trainer/qwen3_8_vllm.py"),
            Path("torchtitan/torchtitan/trainer.py"),
            Path("vllm/vllm/config/parallel.py"),
            Path("vllm/vllm/v1/engine/utils.py"),
            Path("vllm/vllm/v1/engine/core.py"),
            Path("vllm/vllm/v1/executor/ray_executor_v2.py"),
            Path(
                "scripts/train/qwen3.8_ray_mooncake_vllm_torchtitan/debug_single_node.py"
            ),
        }
    )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    suite_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    junit, log = directory / "junit.xml", directory / "pytest.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *tests,
        f"--junitxml={junit}",
    ]
    if selection:
        command.extend(("-k", selection))
    started = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        directory / "request.json",
        {
            "command": command,
            "started_at": started,
            "test_suite_hash": suite_hash,
            "source_hashes": hashes,
        },
    )
    # The subprocess supervisor owns only this test process tree. Its parent
    # remains alive while waiting, including when pytest imports native modules.
    code, reason = 0, None
    with log.open("w") as stream:
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            DEEPSPEC_ORCHESTRATOR_PID=str(os.getpid()),
        )
        supervisor = subprocess.Popen(
            [sys.executable, "-m", "deepspec.orchestration.process", *command],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            code = supervisor.wait(timeout=600)
            if code:
                reason = f"CPU {case} suite exited {code}"
        except subprocess.TimeoutExpired:
            supervisor.terminate()
            try:
                supervisor.wait(timeout=45)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                supervisor.wait(timeout=10)
            code, reason = (
                124,
                f"CPU {case} suite exceeded 600-second execution deadline",
            )
    stable = all(
        p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
        for p in paths
    )
    if not stable:
        code, reason = 1, "Source files changed during the CPU contract run"
    evidence = {
        "executed": True,
        "run_id": run_id,
        "test_suite_hash": suite_hash,
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "checks": {case: code == 0, "source_unchanged": stable},
        "artifacts": [str(log), str(junit), str(directory / "request.json")],
        "scope": "CPU contracts; verification uses tiny native DCP/optimizer fixtures and explicit topology doubles. No Ray GPU placement, model training or 128K acceptance.",
    }
    return record_result(
        root,
        case,
        status="passed" if code == 0 else "failed",
        evidence_level="cpu_contract",
        reason=reason,
        evidence=evidence,
    )


def execute_cpu_collective(root, *, fail_rank=None):
    """Eight local CPU ranks, explicitly distinct from planned multi-node evidence."""
    from deepspec.orchestration.process import start_owned

    if fail_rank is not None and (type(fail_rank) is not int or not 0 <= fail_rank < 8):
        raise ValueError("Injected rank must belong to this eight-rank CPU probe")
    run_id = uuid.uuid4().hex
    output = Path(root).resolve() / "collective-probe" / run_id
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path("tests/pipeline_collective_probe.py"),
        Path(__file__),
        Path("deepspec/orchestration/process.py"),
        Path("deepspec/pipeline/runtime.py"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    suite_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        "--max-restarts=0",
        "-m",
        "tests.pipeline_collective_probe",
        "--backend",
        "gloo",
        "--timeout",
        "15",
        "--output",
        str(output),
    ]
    if fail_rank is not None:
        command += ["--fail-rank", str(fail_rank)]
    started = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        output / "request.json",
        {
            "command": command,
            "source_hashes": hashes,
            "started_at": started,
            "fail_rank": fail_rank,
        },
    )
    failure, handle, cleanup = None, None, {}
    with (output / "probe.log").open("w") as stream:
        try:
            handle = start_owned(
                command,
                env=dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS="1",
                    DEEPSPEC_PIPELINE_RUN_ID=run_id,
                ),
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=240,
                cleanup_timeout=45,
                report_path=output / "supervisor.json",
            )
            handle.result()
        except Exception as error:  # noqa: BLE001 -- retain failed probe and cleanup evidence
            failure = repr(error)
        finally:
            if handle is not None:
                cleanup = handle.stop(timeout=45)
    ranks = [json.loads(p.read_text()) for p in output.glob("rank-*.json")]
    identities = [json.loads(p.read_text()) for p in output.glob("identity-*.json")]
    if fail_rank is None:
        outcome = (
            failure is None
            and len(ranks) == 8
            and all(r["status"] == "passed" for r in ranks)
        )
    else:
        outcome = failure is not None and any(
            r["rank"] == fail_rank
            and "Injected owned probe rank failure" in r.get("error", "")
            for r in ranks
        )
    checks = {
        "all_ranks_joined": {r["rank"] for r in identities} == set(range(8)),
        "expected_outcome": outcome,
        "supervisor_cleanup": cleanup.get("cleanup_complete") is True,
        "cpu_only": bool(identities)
        and all(
            r["backend"] == "gloo" and r["cuda_visible_devices"] == ""
            for r in identities
        ),
        "source_unchanged": all(
            hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in paths
        ),
    }
    evidence = {
        "executed": True,
        "run_id": run_id,
        "test_suite_hash": suite_hash,
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "injected_rank": fail_rank,
        "native_error": failure,
        "scope": "One physical node, eight CPU Gloo ranks; no model/GPU. Does not establish M3 cross-node behavior.",
        "artifacts": [str(p) for p in sorted(output.iterdir()) if p.is_file()],
    }
    return record_result(
        root,
        "collective-probe",
        status="passed" if all(checks.values()) else "failed",
        evidence_level="cpu_process",
        reason=None
        if all(checks.values())
        else failure or "Incomplete collective evidence",
        evidence=evidence,
    )


def execute_lifecycle_probe(root):
    """Real CPU services only, with independent supervisor cleanup evidence."""
    from deepspec.orchestration.process import start_owned

    run_id = uuid.uuid4().hex
    directory = Path(root).resolve() / "lifecycle-probe" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    paths = [
        *Path("deepspec/pipeline").glob("*.py"),
        Path("deepspec/orchestration/process.py"),
        Path("tests/pipeline_lifecycle_probe.py"),
        Path(__file__),
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    suite_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    command = [
        sys.executable,
        "-m",
        "tests.pipeline_lifecycle_probe",
        "--output",
        str(directory),
    ]
    started = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        directory / "request.json",
        {
            "command": command,
            "started_at": started,
            "source_hashes": hashes,
            "test_suite_hash": suite_hash,
        },
    )
    reason, handle, probe, cleanup = None, None, {}, {}
    with (directory / "probe.log").open("w") as stream:
        try:
            handle = start_owned(
                command,
                env=dict(
                    os.environ, CUDA_VISIBLE_DEVICES="", DEEPSPEC_PIPELINE_RUN_ID=run_id
                ),
                timeout=600,
                cleanup_timeout=45,
                report_path=directory / "supervisor.json",
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            handle.result()
            probe = json.loads((directory / "probe.json").read_text())
        except Exception as error:  # noqa: BLE001 -- preserve failed attempts in the append-only evidence index
            reason = repr(error)
        finally:
            if handle is not None:
                cleanup = handle.stop(timeout=45)
    if (directory / "probe.json").exists():
        try:
            probe = json.loads((directory / "probe.json").read_text())
        except ValueError as error:
            reason = reason or repr(error)
    final_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    _atomic_json(directory / "source-final.json", final_hashes)
    checks = {
        "service_probe": reason is None and probe.get("cleanup_complete") is True,
        "bystander_survived": probe.get("bystander_survived") is True,
        "supervisor_cleanup": cleanup.get("cleanup_complete") is True,
        "transport_check": probe.get("transport", {}).get("status") == "passed",
        "driver_loss": probe.get("driver_loss", {}).get("passed") is True,
        "all_phase_cancellations": (
            {r.get("phase") for r in probe.get("phase_cancellation", [])}
            == {"allocate", "initialize", "ready", "run", "drain", "verify"}
            and all(
                r.get("passed") is True for r in probe.get("phase_cancellation", [])
            )
        ),
        "source_unchanged_during_probe": final_hashes == hashes,
    }
    evidence = {
        "executed": True,
        "run_id": run_id,
        "test_suite_hash": suite_hash,
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "artifacts": [
            str(directory / name)
            for name in (
                "request.json",
                "probe.log",
                "probe.json",
                "supervisor.json",
                "source-final.json",
            )
            if (directory / name).exists()
        ],
        "scope": "Real local CPU service, transport, driver-SIGKILL/blocked-control cleanup and cancellation in six controller phases; synthetic topology metadata, 64 MiB pool, no GPU/model.",
    }
    return record_result(
        root,
        "lifecycle-probe",
        status="passed" if all(checks.values()) else "failed",
        evidence_level="cpu_process",
        reason=reason,
        evidence=evidence,
    )


def execute_cpu_preview(root, case):
    """Run explicit synthetic preview contracts; never claim training evidence."""
    from unittest.mock import patch

    from deepspec.pipeline import cli
    from deepspec.pipeline.planning import build_plan
    from deepspec.pipeline.runtime import PipelineError
    from deepspec.pipeline.schema import normalize_task_config
    from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config

    layout = case.rsplit("-", 1)[0].upper()
    evidence_root = Path(root).resolve() / "cpu-contracts" / case / uuid.uuid4().hex
    evidence_root.mkdir(parents=True, exist_ok=False)
    config = task_config(layout, output_dir=evidence_root / "preview")
    config["data"]["context_length"] = 131072 if case.endswith("128k") else 4096
    config_path = evidence_root / "config.json"
    _atomic_json(config_path, config)
    calls = {"cpu_inspections": 0, "cpu_preparations": 0, "large_pool_allocations": 0}

    def inspect(config, run):
        calls["cpu_inspections"] += 1
        return node_facts(config, now=time.monotonic())

    def prepare(config, run):
        calls["cpu_preparations"] += 1
        return input_plan(config), {"run_id": run.run_id, "schema_version": 2}

    def forbidden(*args, **kwargs):
        calls["large_pool_allocations"] += 1
        raise AssertionError("A CPU preview attempted to create a Store")

    started = datetime.now(timezone.utc).isoformat()
    with (
        patch.object(cli, "inspect_nodes", inspect),
        patch.object(cli, "prepare_input", prepare),
        patch("deepspec.pipeline.store.TensorStore", forbidden),
    ):
        result = cli.preview(config_path)
    normalized = normalize_task_config(config).to_dict()
    invalid = node_facts(normalized, now=time.monotonic())
    invalid[0]["gpu_available"] = 0
    try:
        build_plan(
            normalized,
            invalid,
            input_plan(normalized),
            run_id="negative-fixture",
            now=time.monotonic(),
        )
    except PipelineError as error:
        rejected_capacity = error.details["code"] == "INSUFFICIENT_GPU"
    else:
        rejected_capacity = False
    _atomic_json(evidence_root / "calls.json", calls)
    evidence = {
        "executed": True,
        "run_id": result["run_id"],
        "plan_hash": result["plan_hash"],
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "preview_complete": result["phase_detail"] == "preview_complete",
            "capacity_negative_rejected": rejected_capacity,
            "no_pool": calls["large_pool_allocations"] == 0,
        },
        "artifacts": [result["plan_path"], str(evidence_root / "calls.json")],
        "scope": "Synthetic node/input backend, CPU contract only; no Ray/GPU/model/transport execution.",
        "synthetic_context_length": config["data"]["context_length"],
        "calls": calls,
    }
    return record_result(
        root, case, status="passed", evidence_level="cpu_contract", evidence=evidence
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", choices=(*NORMAL_CASES, *FAULT_CASES, "all")
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--lifecycle-probe",
        action="store_true",
        help="Execute the real local CPU service lifecycle probe",
    )
    parser.add_argument(
        "--capacity-contracts",
        action="store_true",
        help="Execute bounded feature-budget, writer, reader and deletion CPU tests",
    )
    parser.add_argument(
        "--verification-contracts",
        action="store_true",
        help="Execute independent native DCP, progress and cleanup CPU contracts",
    )
    parser.add_argument(
        "--execution-contracts",
        action="store_true",
        help="Execute native controller, CLI, evidence and backend CPU contracts",
    )
    parser.add_argument(
        "--execute-plan", help="Execute the frozen plan for exactly one normal --case"
    )
    parser.add_argument(
        "--cpu-contract",
        action="store_true",
        help="Execute synthetic preview cases only",
    )
    parser.add_argument(
        "--cpu-collective-probe",
        action="store_true",
        help="Run eight local CPU Gloo ranks under independent supervision",
    )
    parser.add_argument(
        "--collective-fail-rank",
        type=int,
        help="Inject failure in one owned CPU probe rank",
    )
    parser.add_argument("--output-root", default="outputs/ray-topology-acceptance")
    args = parser.parse_args(argv)
    if args.collective_fail_rank is not None and not args.cpu_collective_probe:
        parser.error("--collective-fail-rank requires --cpu-collective-probe")
    if args.cpu_collective_probe:
        result = execute_cpu_collective(
            args.output_root, fail_rank=args.collective_fail_rank
        )
        print(json.dumps(result))
        return 0 if result["status"] == "passed" else 1
    if args.execute_plan:
        if not args.case or len(args.case) != 1 or args.case[0] not in NORMAL_CASES:
            parser.error("--execute-plan requires exactly one normal --case")
        result = execute_training_plan(
            args.output_root, args.case[0], args.execute_plan
        )
        print(json.dumps(result))
        return 0 if result["status"] == "passed" else 1
    if args.list:
        print(
            json.dumps(
                {"normal": NORMAL_CASES, "faults": FAULT_CASES, "probes": PROBE_CASES},
                indent=2,
            )
        )
        return 0
    if args.lifecycle_probe:
        result = execute_lifecycle_probe(args.output_root)
        print(json.dumps(result))
        return 0 if result["status"] == "passed" else 1
    if args.capacity_contracts:
        result = execute_capacity_contracts(args.output_root)
        print(json.dumps(result))
        return 0 if result["status"] == "passed" else 1
    if args.verification_contracts:
        result = execute_verification_contracts(args.output_root)
        print(json.dumps(result))
        return 0 if result["status"] == "passed" else 1
    if args.execution_contracts:
        result = execute_execution_contracts(args.output_root)
        print(json.dumps(result))
        return 0 if result["status"] == "passed" else 1
    cases = args.case or ["all"]
    if "all" in cases:
        cases = NORMAL_CASES if args.cpu_contract else NORMAL_CASES + FAULT_CASES
    for case in dict.fromkeys(cases):
        if args.cpu_contract:
            if case not in NORMAL_CASES:
                parser.error("CPU preview contracts accept only normal matrix cases")
            result = execute_cpu_preview(args.output_root, case)
            print(json.dumps(result))
            continue
        result = record_result(
            args.output_root,
            case,
            status="not_run",
            evidence_level="inventory",
            reason="Inventory only; use --execute-plan with a prepared plan or an explicit probe option. No training was run.",
        )
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

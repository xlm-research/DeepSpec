"""Frozen run entry points and independently supervised CPU verification."""

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .planning import Run, TopologyPlan, resolve_node_facts
from .runtime import (
    PipelineError,
    atomic_json,
    atomic_json_once,
    message_envelope,
    validate_message,
)
from .schema import content_hash


def _artifact_hash(path):
    # The compatibility snapshot refers back to this plan. Excluding only that
    # reference avoids a cycle; its value is checked separately on every read.
    if path.name in ("pipeline.json", "environment.json"):
        value = json.loads(path.read_text())
        value.pop("plan_hash", None)
        return content_hash(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_artifacts(plan):
    output = Path(plan["config"]["output_dir"]).resolve()
    paths = {
        output / n
        for n in (
            "pipeline.json",
            "environment.json",
            "inputs/input-plan.json",
        )
    }
    paths.update(
        Path(s["input_path"]).resolve()
        for s in plan["input_plan"]["batches"]
        if "input_path" in s
    )
    compatibility = json.loads((output / "pipeline.json").read_text())
    if compatibility.get("manifest_path"):
        paths.add(Path(compatibility["manifest_path"]).resolve())
    manifest = {}
    for path in sorted(paths):
        if not path.is_relative_to(output):
            raise PipelineError(
                "INPUT_PATH_INVALID", "Prepared inputs must belong to the run"
            )
        manifest[str(path.relative_to(output))] = _artifact_hash(path)
    return TopologyPlan.freeze(
        {**plan, "frozen_artifacts": manifest, "evidence_capture": True}
    ).to_dict()


def validate_artifacts(plan):
    output = Path(plan["config"]["output_dir"]).resolve()
    manifest = plan.get("frozen_artifacts", {})
    if (
        not {"pipeline.json", "environment.json", "inputs/input-plan.json"}
        <= manifest.keys()
    ):
        raise PipelineError(
            "PLAN_NOT_SEALED", "Run requires a freshly previewed, sealed plan"
        )
    for name, expected in manifest.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or _artifact_hash(path) != expected:
            raise PipelineError(
                "FROZEN_ARTIFACT_CHANGED",
                f"Frozen artifact changed: {name}",
                field_path=name,
            )
        if path.name in ("pipeline.json", "environment.json"):
            value = json.loads(path.read_text())
            if value.get("plan_hash", plan["plan_hash"]) != plan["plan_hash"]:
                raise PipelineError(
                    "RUN_IDENTITY_MISMATCH", f"Artifact identity differs: {name}"
                )


def verification_budget(plan, *, snapshot=None):
    """A separate post-model CPU working set, charged alongside the live pool."""
    node = plan["services"]["pool_node_id"]
    approved = plan["node_budgets"][node]
    required = plan["input_plan"].get("verification_memory_bytes")
    if type(required) is not int or required <= 0:
        raise PipelineError(
            "VERIFIER_BUDGET_MISSING",
            "CPU verification memory was not measured during preparation",
            exit_code=4,
        )
    ceiling = min(approved["startup_budget"], approved["static_cap"])
    live = approved["pool_bytes"] + approved["client_bound"] + approved["reserve_bytes"]
    if snapshot is not None:
        ceiling = min(
            ceiling,
            snapshot["physical_bytes"] * 4 // 5,
            snapshot["limit_bytes"] * 4 // 5,
            snapshot["headroom_bytes"] - approved["headroom_reserve_bytes"],
        )
    if required + live > ceiling:
        raise PipelineError(
            "VERIFIER_MEMORY_INSUFFICIENT",
            "CPU verification working set exceeds the approved node budget",
            node_id=node,
            exit_code=4,
        )
    return required


def revalidate_environment(plan):
    from .cluster import inspect_task_nodes

    run = Run(
        plan["run_id"], f"deepspec-{plan['run_id']}", plan["config"]["output_dir"], ""
    )
    facts = inspect_task_nodes(plan["config"], run)
    selected = resolve_node_facts(plan["config"], facts, now=time.monotonic())
    for alias, old in plan["nodes"].items():
        fresh = selected[alias]
        if any(old[k] != fresh[k] for k in ("node_id", "ip", "boot_id", "identities")):
            raise PipelineError(
                "ENVIRONMENT_CHANGED",
                "Node, model, input, source or dependency identity changed after preview",
                node_id=old["node_id"],
            )
    verification_budget(
        plan, snapshot=selected[plan["config"]["store"]["node"]]["memory"]
    )
    validate_artifacts(plan)
    return facts


def read_events(output):
    # Preserve each sender's order. Cross-node monotonic timestamps are never
    # used to impose a fabricated global timeline.
    return [
        json.loads(line)
        for path in sorted((Path(output) / "events").glob("*.jsonl"))
        for line in path.read_text().splitlines()
        if line
    ]


def checkpoint_path(plan, events):
    paths = {e["data"]["path"] for e in events if e["event"] == "checkpoint_committed"}
    if len(paths) != 1:
        raise ValueError("All ranks must identify one committed checkpoint")
    path = Path(paths.pop()).resolve()
    if not path.is_relative_to(
        Path(plan["config"]["output_dir"]).resolve() / "checkpoints"
    ):
        raise ValueError("Checkpoint is outside the run directory")
    return path


def verify_saved(plan, *, full, memory_budget_bytes):
    from .verification import (
        load_checkpoint_expectations,
        verify_checkpoint,
        verify_commits,
        verify_execution,
        verify_progress,
        verify_placement,
    )

    output = Path(plan["config"]["output_dir"])
    validate_artifacts(plan)
    events = read_events(output)
    checkpoint = checkpoint_path(plan, events)
    if full:
        result = verify_execution(
            plan,
            checkpoint,
            events=events,
            source=json.loads((output / "source-release.json").read_text()),
            registries=[
                json.loads((output / n).read_text())
                for n in ("allocation.json", "native-allocation.json")
            ],
            cleanup=json.loads((output / "resource-cleanup.json").read_text())[
                "observations"
            ],
            memory_budget_bytes=memory_budget_bytes,
        )
    else:
        counts = verify_progress(plan, events)
        saved = verify_checkpoint(
            plan,
            checkpoint,
            load_checkpoint_expectations(plan, events),
            memory_budget_bytes=memory_budget_bytes,
        )
        verify_commits(plan, events, checkpoint, saved)
        result = message_envelope(
            plan["run_id"],
            plan["plan_hash"],
            {"component": "cpu_verifier"},
            verified=True,
            counts=counts,
            checkpoint=saved,
        )
    result["placement"] = verify_placement(
        plan, json.loads((output / "actual-placement.json").read_text())
    )
    if full:
        from .metrics import require_metrics

        result["metrics"] = require_metrics(plan, events)
    result.update(
        independent=True,
        verifier_pid=os.getpid(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    return result


class CPUVerifier:
    """One planned CPU slot; a blocked DCP RPC is terminated by its owner."""

    def __init__(self, plan):
        self.plan = plan

    def identity(self):
        from .runtime import actor_identity

        return actor_identity(self.plan)

    def verify(self, *, full=False):
        from .memory import node_memory

        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Verifier was assigned GPUs")
        budget = verification_budget(self.plan, snapshot=node_memory())
        return verify_saved(self.plan, full=full, memory_budget_bytes=budget)


def _supervise(plan, mode, request_path, result_path, *, duration):
    from deepspec.orchestration.process import start_owned

    policy = plan["timeouts_seconds"]
    atomic_json_once(
        request_path, {"plan": plan, "mode": mode, "result_path": str(result_path)}
    )
    with request_path.with_suffix(".log").open("x") as log:
        handle = start_owned(
            [sys.executable, "-m", "deepspec.pipeline.execution", str(request_path)],
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                DEEPSPEC_PIPELINE_RUN_ID=plan["run_id"],
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=duration,
            cleanup_timeout=policy["cleanup"],
            report_path=request_path.with_suffix(".cleanup.json"),
        )
        error = None
        try:
            handle.result()
        except Exception as failure:  # noqa: BLE001 -- prefer a persisted structured failure
            error = failure
        finally:
            cleanup = handle.stop(timeout=policy["cleanup"])
    if not result_path.exists() or not cleanup["cleanup_complete"]:
        raise PipelineError(
            "DRIVER_FAILED",
            str(error or "Driver cleanup not confirmed"),
            run_id=plan["run_id"],
            exit_code=3,
        )
    result = json.loads(result_path.read_text())
    if error and result.get("verified", result.get("state") == "succeeded"):
        raise PipelineError(
            "DRIVER_FAILED", str(error), run_id=plan["run_id"], exit_code=3
        )
    return result


def supervise_run(plan):
    output = Path(plan["config"]["output_dir"])
    return _supervise(
        plan,
        "run",
        output / "driver-request.json",
        output / "status.json",
        duration=plan["timeouts_seconds"]["run"]
        + 2 * plan["timeouts_seconds"]["cleanup"],
    )


def run_plan(plan_path):
    from .cli import load_run

    path = Path(plan_path).resolve()
    if path.name != "plan.json":
        raise PipelineError(
            "PLAN_PATH_INVALID", "Use the frozen run plan.json", field_path="plan"
        )
    output, plan, status = load_run(path.parent)
    if (
        status["state"] in {"succeeded", "failed", "cancelled"}
        or (output / "execution.json").exists()
    ):
        raise PipelineError(
            "RUN_ALREADY_EXECUTED", "Plan was already executed or terminated"
        )
    validate_artifacts(plan)
    verification_budget(plan)
    return supervise_run(plan)


def verify_run(run_dir):
    from .cli import load_run

    output, plan, status = load_run(run_dir)
    if status["state"] not in {"succeeded", "failed", "cancelled"} or not status.get(
        "cleanup_finished"
    ):
        raise PipelineError(
            "RUN_NOT_FINISHED",
            "Independent verification requires a completed cleanup attempt",
        )
    validate_artifacts(plan)
    verification_budget(plan)
    attempt = output / "verification" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    result = _supervise(
        plan,
        "verify",
        attempt / "request.json",
        attempt / "result.json",
        duration=plan["timeouts_seconds"]["run"],
    )
    atomic_json(output / "verification.json", result)
    return result


def status_run(run_dir):
    from .cli import load_run

    output, plan, status = load_run(run_dir)
    result = dict(status)
    started = (output / "execution.json").exists()
    terminal = status["state"] in {"succeeded", "failed", "cancelled"}
    result["health"] = (
        "terminal" if terminal else "unknown" if started else "not_started"
    )
    result["orphan_reports"] = [
        json.loads(p.read_text()) for p in sorted(output.glob("orphan-*.json"))
    ]
    heartbeat = output / "controller-heartbeat.json"
    if started and not terminal and heartbeat.exists():
        beat = json.loads(heartbeat.read_text())
        execution = json.loads((output / "execution.json").read_text())
        if (beat.get("run_id"), beat.get("plan_hash"), beat.get("execution_id")) != (
            plan["run_id"],
            plan["plan_hash"],
            execution["execution_id"],
        ):
            raise PipelineError(
                "HEARTBEAT_IDENTITY_MISMATCH",
                "Controller heartbeat belongs to another execution",
            )
        if (
            beat.get("boot_id")
            == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        ):
            age = time.monotonic() - beat["local_monotonic"]
            result["health"] = (
                "healthy"
                if 0 <= age < plan["timeouts_seconds"]["lease"]
                else "lease_expired"
            )
        result["controller_heartbeat"] = beat
    if result["orphan_reports"] and not terminal:
        result["health"] = "driver_lost"
    events = read_events(output)
    result["progress"] = {
        "produced": sum(e["event"] == "feature_produced" for e in events),
        "complete_reads": sum(
            e["event"] == "feature_read" and e["data"].get("verified") is True
            for e in events
        ),
        "rank_updates": {
            str(r["global_rank"]): max(
                (
                    e["data"]["optimizer_step"]
                    for e in events
                    if e["event"] == "rank_update_completed"
                    and e["sender_identity"].get("global_rank") == r["global_rank"]
                ),
                default=0,
            )
            for r in plan["training_ranks"]
        },
    }
    result["progress"]["committed_ranks"] = sorted(
        {
            e["sender_identity"].get("global_rank")
            for e in events
            if e["event"] == "checkpoint_committed" and e["basis"] == "verified"
        }
    )
    result["expected"] = plan["counts"]
    result["declared"] = {
        "layout": plan["layout"],
        "replicas": plan["replicas"],
        "training_ranks": plan["training_ranks"],
        "node_budgets": plan["node_budgets"],
    }
    for name, filename in (
        ("actual", "actual-placement.json"),
        ("verification_result", "verification.json"),
        ("resource_cleanup", "resource-cleanup.json"),
    ):
        path = output / filename
        result[name] = json.loads(path.read_text()) if path.exists() else None
    result["node_cleanup"] = [
        json.loads(p.read_text()) for p in sorted(output.glob("cleanup-*.json"))
    ] + result["orphan_reports"]
    expected_nodes = {n["node_id"] for n in plan["nodes"].values()}
    for report in result["node_cleanup"]:
        validate_message(report, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
        if report.get("node_id") not in expected_nodes:
            raise PipelineError(
                "CLEANUP_IDENTITY_MISMATCH", "Cleanup node is outside the plan"
            )
    if result["health"] in {"driver_lost", "lease_expired"} and not terminal:
        result["controller_state"] = result["state"]
        result["state"] = "failed"
        result["reason"] = {
            "code": "DRIVER_LEASE_EXPIRED",
            "message": "Controller lease is no longer live; cleanup remains confirmed only where node reports prove it",
        }
        result["cleanup_complete"] = (
            bool(result["node_cleanup"])
            and {n["node_id"] for n in result["node_cleanup"]} == expected_nodes
            and all(n["cleanup_complete"] for n in result["node_cleanup"])
        )
    return result


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    plan = TopologyPlan.from_dict(request["plan"]).to_dict()
    try:
        if request["mode"] == "run":
            from .controller import RunController
            from .operations import NativeRunOperations

            result = RunController(plan, NativeRunOperations(plan)).run()
        else:
            from .operations import verify_completed_run

            result = verify_completed_run(plan)
    except BaseException as error:  # noqa: BLE001 -- independent worker records failures without rewriting status
        result = message_envelope(
            plan["run_id"],
            plan["plan_hash"],
            {"component": "driver"},
            verified=False,
            error=repr(error),
        )
        if request["mode"] == "run":
            path = Path(request["result_path"])
            existing = json.loads(path.read_text()) if path.exists() else {}
            if existing.get("state") in {"succeeded", "failed", "cancelled"}:
                return 3
            result.update(
                state="failed",
                phase_detail="driver_setup",
                cleanup_finished=True,
                cleanup_complete=False,
                reason={"code": "DRIVER_SETUP_FAILED", "message": repr(error)},
            )
    atomic_json(request["result_path"], result)
    return 0 if result.get("verified", result.get("state") == "succeeded") else 3


if __name__ == "__main__":
    raise SystemExit(main())

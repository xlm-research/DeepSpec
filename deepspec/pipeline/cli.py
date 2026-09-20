"""Versioned DeepSpec task interface. Preview performs CPU preparation only."""

import argparse
import json
import sys
import time
from pathlib import Path

from .cluster import inspect_task_nodes as inspect_nodes
from .planning import Run, TopologyPlan, build_plan, prepare_input, resolve_node_facts
from .runtime import (
    Deadline,
    PipelineError,
    atomic_json,
    atomic_json_once,
    message_envelope,
    validate_message,
)
from .schema import content_hash, upgrade_task_config


def preview(config_path):
    try:
        value = json.loads(Path(config_path).read_text())
    except (OSError, ValueError) as error:
        raise PipelineError(
            "CONFIG_READ_ERROR", str(error), field_path="config"
        ) from error
    normalized = upgrade_task_config(value)
    config = normalized.to_dict()
    run = Run.create(config["output_dir"])
    output = Path(run.output_dir)
    status = {
        "schema_version": 3,
        **run.to_dict(),
        "state": "preparing",
        "phase_detail": "preflight",
        "plan_hash": None,
    }
    atomic_json(output / "config.normalized.json", config)
    atomic_json(output / "status.json", status)
    try:
        facts = inspect_nodes(config, run)
        if config["ray_address"] == "auto":
            addresses = {n.get("ray_address") for n in facts}
            if len(addresses) != 1 or not next(iter(addresses)) or "auto" in addresses:
                raise PipelineError(
                    "RAY_ADDRESS_UNRESOLVED",
                    "Inspection must report one resolved Ray address",
                    field_path="ray_address",
                    exit_code=4,
                )
            config["ray_address"] = addresses.pop()
            normalized = upgrade_task_config(config)
            config = normalized.to_dict()
            atomic_json(output / "config.normalized.json", config)
        resolve_node_facts(config, facts, now=time.monotonic())
        inputs, legacy = prepare_input(config, run)
        if any(
            time.monotonic() - n["request_sent_at"]
            >= config["timeouts_seconds"]["budget_snapshot"]
            for n in facts
        ):
            facts = inspect_nodes(config, run)
        plan = build_plan(
            config, facts, inputs, run_id=run.run_id, now=time.monotonic()
        )
        atomic_json(
            output / "environment.json",
            {
                "schema_version": 3,
                "run_id": run.run_id,
                "plan_hash": plan.plan_hash,
                "nodes": facts,
            },
        )
        native_path = output / "inputs/input-plan.json"
        if not native_path.exists():
            atomic_json(native_path, inputs)
        legacy.update(
            plan_hash=plan.plan_hash,
            config_hash=normalized.config_hash,
            topology_plan_path=str(output / "plan.json"),
            node_budgets=plan.to_dict()["node_budgets"],
            timeouts_seconds=config["timeouts_seconds"],
            approved_inference_dp=config["inference"]["dp"],
        )
        atomic_json(output / "pipeline.json", legacy)
        from .execution import freeze_artifacts

        plan = TopologyPlan.from_dict(freeze_artifacts(plan.to_dict()))
        legacy["plan_hash"] = plan.plan_hash
        atomic_json(output / "pipeline.json", legacy)
        environment = json.loads((output / "environment.json").read_text())
        environment["plan_hash"] = plan.plan_hash
        atomic_json(output / "environment.json", environment)
        atomic_json(output / "plan.json", plan.to_dict())
        status.update(phase_detail="preview_complete", plan_hash=plan.plan_hash)
        atomic_json(output / "status.json", status)
        return {
            "run_id": run.run_id,
            "plan_hash": plan.plan_hash,
            "plan_path": str(output / "plan.json"),
            "state": "preparing",
            "phase_detail": "preview_complete",
        }
    except BaseException as error:
        details = (
            error.to_dict()
            if isinstance(error, PipelineError)
            else {"code": "PREPARATION_FAILED", "message": str(error)}
        )
        details.update(run_id=run.run_id, phase="preparing")
        if isinstance(error, PipelineError):
            error.details.update(run_id=run.run_id, phase="preparing")
        status.update(state="failed", reason=details)
        atomic_json(output / "status.json", status)
        if isinstance(error, Exception) and not isinstance(error, PipelineError):
            raise PipelineError(
                "PREPARATION_FAILED",
                str(error),
                run_id=run.run_id,
                phase="preparing",
                field_path="data.source_path",
                exit_code=3,
            ) from error
        raise


def load_run(run_dir):
    output = Path(run_dir).resolve()
    try:
        plan = TopologyPlan.from_dict(
            json.loads((output / "plan.json").read_text())
        ).to_dict()
        config = json.loads((output / "config.normalized.json").read_text())
        status = json.loads((output / "status.json").read_text())
    except (OSError, ValueError) as error:
        raise PipelineError(
            "RUN_READ_ERROR", str(error), field_path="run_dir"
        ) from error
    if (
        Path(plan["config"]["output_dir"]).resolve() != output
        or content_hash(config) != plan["config_hash"]
        or (status.get("run_id"), status.get("plan_hash"))
        != (plan["run_id"], plan["plan_hash"])
    ):
        raise PipelineError(
            "RUN_IDENTITY_MISMATCH",
            "Directory, configuration and status must belong to the same frozen plan",
            field_path="run_dir",
        )
    return output, plan, status


def cancel(run_dir):
    output, plan, status = load_run(run_dir)
    terminal = {"succeeded", "failed", "cancelled"}
    if status["state"] in terminal and status.get("cleanup_finished", True):
        return status
    try:
        execution = json.loads((output / "execution.json").read_text())
    except (OSError, ValueError) as error:
        raise PipelineError(
            "RUN_NOT_STARTED",
            "No active controller owns this plan",
            field_path="run_dir",
        ) from error
    validate_message(execution, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
    request = message_envelope(
        plan["run_id"],
        plan["plan_hash"],
        {"component": "cancel_cli"},
        execution_id=execution["execution_id"],
        requested_at=time.time(),
    )
    path = output / "control/cancel.json"
    atomic_json_once(path, request)
    saved = json.loads(path.read_text())
    validate_message(saved, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
    if saved.get("execution_id") != execution["execution_id"]:
        raise PipelineError(
            "CANCEL_IDENTITY_MISMATCH",
            "Existing request belongs to another execution",
            field_path="control/cancel.json",
        )
    policy = plan["timeouts_seconds"]
    deadline = Deadline.after(policy["transfer"] + policy["cleanup"])
    while True:
        _, _, status = load_run(output)
        if status["state"] in terminal and status.get("cleanup_finished", True):
            return status
        try:
            time.sleep(min(0.05, deadline.remaining()))
        except TimeoutError as error:
            raise PipelineError(
                "CANCEL_CLEANUP_UNKNOWN",
                "Controller did not confirm cleanup before the cancellation deadline",
                run_id=plan["run_id"],
                phase="cleanup",
                exit_code=3,
            ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview_parser = commands.add_parser("preview")
    preview_parser.add_argument("--config", required=True)
    cancel_parser = commands.add_parser("cancel")
    cancel_parser.add_argument("--run-dir", required=True)
    transport_parser = commands.add_parser("transport-check")
    transport_parser.add_argument("--plan", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--run-dir", required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-dir", required=True)
    status_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "preview":
            result = preview(args.config)
        elif args.command == "cancel":
            result = cancel(args.run_dir)
        elif args.command == "run":
            from .execution import run_plan

            result = run_plan(args.plan)
        elif args.command == "verify":
            from .execution import verify_run

            result = verify_run(args.run_dir)
        elif args.command == "status":
            from .execution import status_run

            result = status_run(args.run_dir)
        else:
            from .transport import transport_check

            result = transport_check(args.plan)
    except PipelineError as error:
        print(json.dumps(error.to_dict(), allow_nan=False), file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError, TimeoutError) as error:
        wrapped = PipelineError(
            "PREPARATION_FAILED" if args.command == "preview" else "CONTROL_FAILED",
            str(error),
            phase="preparing" if args.command == "preview" else "cancel",
            field_path="config" if args.command == "preview" else "run_dir",
            exit_code=3,
        )
        print(json.dumps(wrapped.to_dict()), file=sys.stderr)
        return 3
    if args.command == "status" and not args.json:
        print(f"Run: {result['run_id']}\nState: {result['state']} ({result['health']})")
        print(
            f"Produced: {result['progress']['produced']} / {result['expected']['samples']}"
        )
        print(
            f"Complete reads: {result['progress']['complete_reads']} / {result['expected']['reader_count']}"
        )
        print(f"Rank updates: {json.dumps(result['progress']['rank_updates'])}")
        print(f"Checkpoint committed ranks: {result['progress']['committed_ranks']}")
        print(
            f"Cleanup: {'confirmed' if result.get('cleanup_complete') else 'unknown or incomplete'}"
        )
    else:
        print(json.dumps(result, allow_nan=False))
    if args.command == "cancel" and result["state"] == "cancelled":
        return 130
    if args.command == "transport-check" and result["status"] != "passed":
        return 3
    if args.command == "run" and result["state"] != "succeeded":
        return 130 if result["state"] == "cancelled" else 3
    if args.command == "verify" and not result["verified"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Execution entry contracts: frozen inputs, independent verification and status."""

import json

import pytest

from deepspec.pipeline import cli
from deepspec.pipeline.planning import TopologyPlan, build_plan
from deepspec.pipeline.runtime import PipelineError, atomic_json
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


def frozen_run(tmp_path):
    from deepspec.pipeline.execution import freeze_artifacts

    config = task_config("M0", output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="execution", now=100
    ).to_dict()
    atomic_json(tmp_path / "inputs/input-plan.json", plan["input_plan"])
    atomic_json(tmp_path / "pipeline.json", {"run_id": plan["run_id"]})
    atomic_json(tmp_path / "environment.json", {"nodes": list(plan["nodes"].values())})
    plan = freeze_artifacts(plan)
    atomic_json(tmp_path / "plan.json", plan)
    atomic_json(tmp_path / "config.normalized.json", plan["config"])
    atomic_json(
        tmp_path / "status.json",
        {
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "state": "preparing",
            "phase_detail": "preview_complete",
        },
    )
    return plan


@pytest.mark.parametrize(
    "name", ["pipeline.json", "environment.json", "inputs/input-plan.json"]
)
def test_run_rejects_frozen_artifact_changes_before_driver_start(
    tmp_path, monkeypatch, name
):
    from deepspec.pipeline import execution

    frozen_run(tmp_path)
    (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        execution, "supervise_run", lambda *a, **k: pytest.fail("driver started")
    )
    with pytest.raises(PipelineError, match="changed"):
        execution.run_plan(tmp_path / "plan.json")
    assert not (tmp_path / "execution.json").exists()


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled"])
def test_run_never_reuses_a_terminal_directory(tmp_path, monkeypatch, state):
    from deepspec.pipeline import execution

    frozen_run(tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    status["state"] = state
    atomic_json(tmp_path / "status.json", status)
    monkeypatch.setattr(
        execution, "supervise_run", lambda *a, **k: pytest.fail("driver started")
    )
    with pytest.raises(PipelineError, match="already"):
        execution.run_plan(tmp_path / "plan.json")


def test_frozen_compatibility_plan_hash_is_validated_without_hash_cycle(tmp_path):
    from deepspec.pipeline.execution import validate_artifacts

    plan = frozen_run(tmp_path)
    path = tmp_path / "pipeline.json"
    config = json.loads(path.read_text())
    config["plan_hash"] = plan["plan_hash"]
    atomic_json(path, config)
    validate_artifacts(plan)
    config["plan_hash"] = "another-plan"
    atomic_json(path, config)
    with pytest.raises(PipelineError, match="identity"):
        validate_artifacts(plan)


def test_status_preview_is_read_only_and_does_not_require_a_lease(tmp_path, capsys):
    frozen_run(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert cli.main(["status", "--run-dir", str(tmp_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "preparing"
    assert result["health"] == "not_started"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_status_human_output_keeps_reads_separate_from_checkpoint(tmp_path, capsys):
    from deepspec.pipeline.runtime import EventWriter

    plan = frozen_run(tmp_path)
    with EventWriter(
        tmp_path / "events/reader.jsonl",
        run_id=plan["run_id"],
        plan_hash=plan["plan_hash"],
        sender_identity={"component": "reader"},
    ) as writer:
        writer.emit(
            "feature_read",
            {
                "position": 0,
                "reader_rank": 0,
                "nbytes": 1,
                "verified": True,
                "duration_seconds": 0.1,
            },
            basis="verified",
        )
    assert cli.main(["status", "--run-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Complete reads: 1 / 48" in output
    assert "Checkpoint committed ranks: []" in output


def test_verification_budget_rejects_before_native_load(tmp_path):
    from deepspec.pipeline.execution import verification_budget

    plan = frozen_run(tmp_path)
    plan["input_plan"]["verification_memory_bytes"] = 1024**5
    plan = TopologyPlan.freeze(plan).to_dict()
    with pytest.raises(PipelineError, match="CPU verification"):
        verification_budget(plan)


def test_new_cli_run_and_verify_dispatch_and_return_failure(
    tmp_path, monkeypatch, capsys
):
    from deepspec.pipeline import execution

    monkeypatch.setattr(execution, "run_plan", lambda path: {"state": "failed"})
    monkeypatch.setattr(execution, "verify_run", lambda path: {"verified": False})
    assert cli.main(["run", "--plan", str(tmp_path / "plan.json")]) == 3
    assert cli.main(["verify", "--run-dir", str(tmp_path)]) == 3
    assert len(capsys.readouterr().out.splitlines()) == 2


@pytest.mark.parametrize("failed_release", ["models", "gpus", None])
def test_cpu_verifier_cannot_start_until_models_and_gpu_allocations_are_gone(
    tmp_path, monkeypatch, failed_release
):
    import threading
    from types import SimpleNamespace
    from deepspec.pipeline import operations
    from deepspec.pipeline.runtime import Deadline, message_envelope

    plan = frozen_run(tmp_path)
    operation = operations.NativeRunOperations(plan)
    calls = []

    def release(label):
        calls.append(label)
        return {"cleanup_complete": failed_release != label}

    monkeypatch.setattr(operation, "stop_groups", lambda **kw: release("models"))
    monkeypatch.setattr(operation, "release_allocations", lambda **kw: release("gpus"))
    monkeypatch.setattr(operations, "verification_budget", lambda plan: 1024)
    actor = SimpleNamespace(
        identity=SimpleNamespace(remote=lambda: "identity"),
        verify=SimpleNamespace(remote=lambda: "verify"),
    )

    def create(*args, **kwargs):
        calls.append("verifier")
        assert kwargs["options"]["num_gpus"] == 0
        assert (
            kwargs["options"]["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == ""
        )
        return actor

    monkeypatch.setattr(operation.actors, "create", create)
    monkeypatch.setattr(operation.actors, "resolve", lambda *a, **kw: {})
    monkeypatch.setattr(operation.agents, "register_process", lambda *a, **kw: None)
    monkeypatch.setattr(
        operation.actors, "stop", lambda **kw: {"cleanup_complete": True}
    )
    monkeypatch.setattr(
        operation,
        "_get",
        lambda *a: message_envelope(
            plan["run_id"],
            plan["plan_hash"],
            {"component": "cpu_verifier"},
            verified=True,
            independent=True,
        ),
    )
    try:
        if failed_release:
            with pytest.raises(RuntimeError):
                operation.verify(deadline=Deadline.after(5), stop=threading.Event())
            assert "verifier" not in calls
        else:
            assert operation.verify(deadline=Deadline.after(5), stop=threading.Event())[
                "independent"
            ]
            assert calls == ["models", "gpus", "verifier"]
    finally:
        for group in (operation.store, operation.training, operation.inference):
            group.executor.shutdown(wait=False)


def test_status_rejects_a_stale_heartbeat_without_mutating_running_state(tmp_path):
    import time
    from deepspec.pipeline.execution import status_run

    plan = frozen_run(tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    status["state"] = "running"
    atomic_json(tmp_path / "status.json", status)
    atomic_json(tmp_path / "execution.json", {"execution_id": "one"})
    atomic_json(
        tmp_path / "controller-heartbeat.json",
        {
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "execution_id": "one",
            "boot_id": __import__("pathlib")
            .Path("/proc/sys/kernel/random/boot_id")
            .read_text()
            .strip(),
            "local_monotonic": time.monotonic() - 2 * plan["timeouts_seconds"]["lease"],
        },
    )
    before = (tmp_path / "status.json").read_bytes()
    assert status_run(tmp_path)["health"] == "lease_expired"
    assert (tmp_path / "status.json").read_bytes() == before


def test_orphan_report_overrides_recent_controller_heartbeat(tmp_path):
    import time
    from pathlib import Path
    from deepspec.pipeline.execution import status_run
    from deepspec.pipeline.runtime import message_envelope

    plan = frozen_run(tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    status["state"] = "running"
    atomic_json(tmp_path / "status.json", status)
    atomic_json(tmp_path / "execution.json", {"execution_id": "one"})
    atomic_json(
        tmp_path / "controller-heartbeat.json",
        {
            "run_id": plan["run_id"],
            "plan_hash": plan["plan_hash"],
            "execution_id": "one",
            "local_monotonic": time.monotonic(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        },
    )
    node = next(iter(plan["nodes"].values()))["node_id"]
    atomic_json(
        tmp_path / f"orphan-{node}.json",
        message_envelope(
            plan["run_id"],
            plan["plan_hash"],
            {"component": "node_agent"},
            node_id=node,
            cleanup_complete=True,
            orphan=True,
        ),
    )
    before = (tmp_path / "status.json").read_bytes()
    result = status_run(tmp_path)
    assert result["state"] == "failed"
    assert result["health"] == "driver_lost"
    assert result["cleanup_complete"] is True
    assert (tmp_path / "status.json").read_bytes() == before


def test_pipeline_error_preserves_structured_fields_across_ray_serialization():
    import ray.cloudpickle as pickle

    original = PipelineError(
        "NODE_GPU_BUSY",
        "Occupied device",
        run_id="run",
        phase="allocating",
        node_id="node-a",
        field_path="inference.nodes",
        retryable=True,
        exit_code=4,
    )
    restored = pickle.loads(pickle.dumps(original))
    assert restored.to_dict() == original.to_dict()
    assert restored.exit_code == 4


def test_pipeline_error_survives_the_native_ray_exception_envelope():
    from ray.exceptions import RayError, RayTaskError

    original = PipelineError(
        "NODE_GPU_BUSY", "Occupied device", node_id="node-a", exit_code=4
    )
    restored = RayError.from_bytes(
        RayTaskError("probe", "probe traceback", original).to_bytes()
    )
    assert restored.cause.to_dict() == original.to_dict()
    assert restored.cause.exit_code == 4


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_late_checkpoint_event_cannot_revive_terminal_status(tmp_path, state):
    from deepspec.pipeline.execution import status_run
    from deepspec.pipeline.runtime import EventWriter

    plan = frozen_run(tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    status["state"] = state
    atomic_json(tmp_path / "status.json", status)
    with EventWriter(
        tmp_path / "events/late-rank.jsonl",
        run_id=plan["run_id"],
        plan_hash=plan["plan_hash"],
        sender_identity={"component": "trainer", "global_rank": 0},
    ) as writer:
        writer.emit(
            "checkpoint_committed",
            {
                "path": "/late",
                "commit_identity": "late",
                "native_cursor": 12,
                "sample_cursor": 12,
            },
            basis="verified",
        )
    result = status_run(tmp_path)
    assert result["state"] == state and result["health"] == "terminal"
    assert result["progress"]["committed_ranks"] == [0]

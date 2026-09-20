import json

import pytest

from deepspec.pipeline.controller import Allocation, ResourceRegistry, RunState
from deepspec.pipeline.planning import TopologyPlan
from deepspec.pipeline.runtime import EventWriter, PipelineError, atomic_json
from deepspec.pipeline.schema import TaskConfig
from deepspec.pipeline.topology import (
    expected_counts,
    planned_producer,
    planned_readers,
    training_rank_table,
)
from tests.pipeline_topology_fixtures import task_config


def test_config_and_plan_are_deeply_immutable_and_identity_checked():
    raw = task_config()
    config = TaskConfig.from_dict(raw)
    raw["training"]["steps"] = 999
    assert config.to_dict()["training"]["steps"] == 3
    exported = config.to_dict()
    exported["training"]["steps"] = 999
    assert config.to_dict()["training"]["steps"] == 3
    plan = TopologyPlan.freeze(
        {"run_id": "run-a", "config": config.to_dict(), "input_plan": {"id": "a"}}
    )
    assert TopologyPlan.from_dict(plan.to_dict()) == plan
    tampered = plan.to_dict()
    tampered["config"]["timeouts_seconds"]["cleanup"] += 1
    with pytest.raises(PipelineError, match="hash"):
        TopologyPlan.from_dict(tampered)
    with pytest.raises((TypeError, ValueError)):
        TopologyPlan.freeze({"actor": object()})


@pytest.mark.parametrize("steps", [1, 3, 5])
@pytest.mark.parametrize("dp", [1, 2])
def test_counts_are_derived_from_plan_not_acceptance_constants(steps, dp):
    config = task_config(f"M1-2{dp}", steps=steps)
    counts = expected_counts(config, sample_count=4 * steps)
    assert counts == {
        "samples": 4 * steps,
        "optimizer_steps": steps,
        "gas": 4 // dp,
        "native_cursor": 4 * steps // dp,
        "sample_cursor": 4 * steps,
        "reader_count": 16 * steps,
    }
    assert planned_producer(config, 1) == 1
    assert planned_readers(config, 1) == tuple(range((1 % dp) * 4, (1 % dp) * 4 + 4))
    with pytest.raises(ValueError):
        expected_counts(config, sample_count=4 * steps - 1)


def test_node_rank_and_dp_rank_are_distinct_for_single_training_node():
    table = training_rank_table([("node-c", 8)], tp=4, dp=2)
    assert table[4]["node_rank"] == 0
    assert table[4]["local_rank"] == 4
    assert table[4]["dp_rank"] == 1
    multi = training_rank_table([("node-b", 4), ("node-c", 4)], tp=4, dp=2)
    assert multi[4]["node_rank"] == 1
    assert multi[4]["local_rank"] == 0
    assert multi[4]["gpu_uuid"] is None  # Actual allocation is a separate fact.


def test_registry_binds_observed_process_and_devices_without_changing_ownership():
    from deepspec.pipeline.controller import ProcessIdentity

    ledger = ResourceRegistry("r", "h")
    for name in ("first", "second"):
        ledger.register(
            Allocation(
                name,
                "r",
                "h",
                "DeepSpec",
                "actor",
                "inference",
                f"actor-{name}",
                "node-a",
                borrower="vLLM",
            )
        )
        ledger.transition(name, "acquiring")
    process = ProcessIdentity("node-a", 100, 123, "r", 90, 100)
    ledger.observe("first", process=process, gpu_uuids=("GPU-a",))
    ledger.observe("first", process=process, gpu_uuids=("GPU-a",))
    observed = ledger.owned_resources()[0]
    assert observed.process == process and observed.gpu_uuids == ("GPU-a",)
    assert observed.borrower == "vLLM" and observed.owner == "DeepSpec"
    with pytest.raises(PipelineError):
        ledger.observe("second", gpu_uuids=("GPU-a",))
    with pytest.raises(PipelineError):
        ledger.observe(
            "first", process=ProcessIdentity("node-a", 100, 456, "r", 90, 100)
        )
    ledger.transition("first", "releasing")
    ledger.transition("first", "released")
    with pytest.raises(PipelineError):
        ledger.observe("first", process=process)


def test_registry_rejects_shared_gpu_cross_run_and_terminal_revival():
    registry = ResourceRegistry("r", "h")
    entry = Allocation(
        "a",
        "r",
        "h",
        "DeepSpec",
        "worker",
        "inference",
        "ray-a",
        "node-a",
        gpu_uuids=("GPU-a",),
    )
    registry.register(entry)
    with pytest.raises(PipelineError):
        registry.register(
            Allocation(
                "b",
                "r",
                "h",
                "DeepSpec",
                "worker",
                "training",
                "ray-b",
                "node-a",
                gpu_uuids=("GPU-a",),
            )
        )
    with pytest.raises(PipelineError):
        registry.register(
            Allocation(
                "c", "other", "h", "DeepSpec", "actor", "training", "ray-c", "node-a"
            )
        )
    for state in ("acquiring", "acquired", "releasing", "unknown"):
        registry.transition("a", state)
    assert not registry.cleanup_complete
    with pytest.raises(PipelineError):
        registry.transition("a", "acquired")


def test_run_keeps_first_terminal_reason():
    state = RunState()
    state.transition("failed", reason="initialization")
    with pytest.raises(PipelineError):
        state.transition("succeeded")
    assert state.reason == "initialization"


def test_atomic_output_and_single_writer_events(tmp_path):
    target = tmp_path / "status.json"
    atomic_json(target, {"ok": True})
    with pytest.raises(ValueError):
        atomic_json(target, {"nan": float("nan")})
    assert json.loads(target.read_text()) == {"ok": True}
    path = tmp_path / "events" / "trainer-node-a-0.jsonl"
    with EventWriter(
        path,
        run_id="r",
        plan_hash="h",
        sender_identity={"component": "trainer", "node_id": "node-a", "rank": 0},
    ) as writer:
        with pytest.raises(PipelineError):
            EventWriter(
                path, run_id="r", plan_hash="h", sender_identity={"component": "other"}
            )
        event = writer.emit(
            "rank_update_completed",
            {"optimizer_step": 1, "native_cursor": 2, "sample_cursor": 4, "loss": 0.5},
            basis="observed",
        )
    assert event["schema_version"] == 3 and event["run_id"] == "r"
    assert event["plan_hash"] == "h" and event["event_id"]
    assert json.loads(path.read_text())["sender_identity"]["rank"] == 0

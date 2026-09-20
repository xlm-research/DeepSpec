"""All-role gates and partial-allocation rollback without model startup."""

import asyncio

import pytest

from deepspec.pipeline.controller import (
    AllocationGate,
    PlacementAllocator,
    ResourceRegistry,
)
from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import Deadline, PipelineError
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


def plan():
    config = task_config("M0")
    return build_plan(
        config, node_facts(config), input_plan(config), run_id="gate-run", now=100
    ).to_dict()


def envelope(p, **payload):
    return {
        "schema_version": 3,
        "run_id": p["run_id"],
        "plan_hash": p["plan_hash"],
        "sender_identity": {"component": "test_shell"},
        "event_id": "test-event",
        **payload,
    }


def reports(p):
    uuids = [gpu["uuid"] for gpu in next(iter(p["nodes"].values()))["gpus"]]
    node = next(iter(p["nodes"].values()))["node_id"]
    workers = [
        envelope(
            p,
            participant=f"inference/0/{rank}",
            node_id=node,
            pg_id="actual-inference-0",
            bundle_index=rank,
            gpu_uuids=[uuids[rank]],
            actor_id=f"worker-{rank}",
            pid=100 + rank,
            start_ticks=10 + rank,
        )
        for rank in range(4)
    ]
    workers.append(
        envelope(
            p,
            participant="training/0",
            node_id=node,
            pg_id="actual-training-0",
            bundle_index=0,
            gpu_uuids=uuids[4:8],
            actor_id="launcher",
            pid=200,
            start_ticks=20,
        )
    )
    return workers


def gate(p):
    return AllocationGate(
        p, pg_ids={item["id"]: "actual-" + item["id"] for item in p["placement_groups"]}
    )


def test_allocation_gate_waits_for_all_roles_and_duplicate_is_idempotent():
    p = plan()
    coordination = gate(p)

    async def scenario():
        waiter = asyncio.create_task(coordination.wait_allocation(Deadline.after(2)))
        for report in reports(p)[:-1]:
            await coordination.report_allocation(report)
        await coordination.report_allocation(reports(p)[0])
        await asyncio.sleep(0)
        assert not waiter.done()
        await coordination.report_allocation(reports(p)[-1])
        assert (await waiter)["ready"]
        assert len(coordination.allocations) == 5

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_id", "other"),
        ("pg_id", "foreign"),
        ("bundle_index", 8),
        ("gpu_uuids", ["unknown"]),
    ],
)
def test_allocation_conflict_wakes_every_waiter_and_never_reopens(field, value):
    p = plan()
    coordination = gate(p)

    async def scenario():
        waiters = [
            asyncio.create_task(coordination.wait_allocation(Deadline.after(2)))
            for _ in range(2)
        ]
        with pytest.raises(PipelineError):
            await coordination.report_allocation({**reports(p)[0], field: value})
        for waiter in waiters:
            with pytest.raises(PipelineError):
                await waiter
        with pytest.raises(PipelineError):
            await coordination.report_allocation(reports(p)[0])

    asyncio.run(scenario())


def test_gate_rejects_duplicate_physical_device_and_missing_report_timeout():
    p = plan()

    async def scenario():
        coordination = gate(p)
        items = reports(p)
        await coordination.report_allocation(items[0])
        with pytest.raises(PipelineError, match="GPU"):
            await coordination.report_allocation(
                {**items[1], "gpu_uuids": items[0]["gpu_uuids"]}
            )
        missing = gate(p)
        with pytest.raises(TimeoutError):
            await missing.wait_allocation(Deadline.after(0.01))
        with pytest.raises(PipelineError):
            await missing.report_allocation(items[0])

    asyncio.run(scenario())


def initialization_reports(p):
    result = []
    for allocation in reports(p)[:-1]:
        _, dp, tp = allocation["participant"].split("/")
        result.append(
            envelope(
                p,
                participant=allocation["participant"],
                node_id=allocation["node_id"],
                actor_id=allocation["actor_id"],
                dp_rank=int(dp),
                tp_rank=int(tp),
            )
        )
    for rank in p["training_ranks"]:
        node = next(n for n in p["nodes"].values() if n["node_id"] == rank["node_id"])
        result.append(
            envelope(
                p,
                participant=f"rank/{rank['global_rank']}",
                **rank,
                world=len(p["training_ranks"]),
                gas=p["counts"]["gas"],
                input_plan_hash=p["input_plan_hash"],
                model_identity=node["identities"]["model"],
            )
        )
    result.append(
        envelope(
            p,
            participant="store",
            node_id=p["services"]["pool_node_id"],
            transport_verified=True,
        )
    )
    return result


def test_initialization_cannot_start_early_or_finish_without_every_rank():
    p = plan()

    async def scenario():
        early = gate(p)
        with pytest.raises(PipelineError, match="before GPU initialization"):
            await early.report_initialized(initialization_reports(p)[0])
        coordination = gate(p)
        for allocation in reports(p):
            await coordination.report_allocation(allocation)
        waiter = asyncio.create_task(coordination.wait_initialized(Deadline.after(2)))
        for initialized in initialization_reports(p)[:-1]:
            await coordination.report_initialized(initialized)
        await asyncio.sleep(0)
        assert not waiter.done()
        await coordination.report_initialized(initialization_reports(p)[-1])
        assert (await waiter)["ready"]

    asyncio.run(scenario())


def test_late_success_cannot_revive_a_cancelled_gate():
    p = plan()

    async def scenario():
        coordination = gate(p)
        await coordination.abort("user cancelled")
        await coordination.abort("cleanup error")
        with pytest.raises(PipelineError, match="user cancelled"):
            await coordination.report_allocation(reports(p)[0])

    asyncio.run(scenario())


def test_rpc_timeout_is_clamped_to_the_remaining_run(monkeypatch):
    import sys
    from types import SimpleNamespace

    from deepspec.pipeline.runtime import component_deadline, get_with_deadline

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(get=lambda ref, **kw: calls.append((ref, kw))),
    )
    config = {"timeout_seconds": 5, "timeouts_seconds": {"run": 10, "transfer": 5}}
    monkeypatch.setenv("DEEPSPEC_RUN_REMAINING_SECONDS", "3")
    deadline = component_deadline(config, clock=lambda: 100)
    assert deadline.expires_at == 103
    get_with_deadline("ref", config, deadline=deadline, clock=lambda: 102)
    assert calls == [("ref", {"timeout": 1})]
    with pytest.raises(TimeoutError):
        get_with_deadline("late", config, deadline=deadline, clock=lambda: 103)
    assert len(calls) == 1


@pytest.mark.parametrize("confirmed", [False, True])
def test_owned_master_requires_supervisor_cleanup_confirmation(
    tmp_path, monkeypatch, confirmed
):
    from types import SimpleNamespace

    from deepspec.pipeline.runtime import MooncakeMaster

    calls = []
    process = SimpleNamespace(poll=lambda: None)
    handle = SimpleNamespace(
        process=process,
        stop=lambda **kw: calls.append(kw) or {"cleanup_complete": confirmed},
    )
    monkeypatch.setattr(
        "deepspec.orchestration.process.start_owned", lambda *a, **kw: handle
    )
    monkeypatch.setattr(
        "deepspec.pipeline.runtime.subprocess.Popen",
        lambda *a, **kw: pytest.fail("unsupervised process path"),
    )
    monkeypatch.setattr(
        "deepspec.pipeline.runtime.wait_for_endpoint", lambda *a, **kw: None
    )
    master = MooncakeMaster(
        "127.0.0.1:23456", tmp_path / "master.log", metrics_port=23457
    )
    master.start()
    if confirmed:
        master.stop(timeout=1)
        master.stop(timeout=1)
        assert len(calls) == 1
    else:
        with pytest.raises(RuntimeError, match="cleanup"):
            master.stop(timeout=1)
    assert calls[0]["timeout"] <= 1


class PlacementBackend:
    def __init__(self, fail_at=None, unreachable=False):
        self.created, self.removed, self.timeouts = [], [], []
        self.fail_at, self.unreachable = fail_at, unreachable

    def create(self, spec, *, name):
        handle = "actual-" + spec["id"]
        self.created.append(handle)
        return handle

    def identity(self, handle):
        return handle

    def ready(self, handle, *, timeout):
        self.timeouts.append(timeout)
        if len(self.timeouts) == self.fail_at:
            raise TimeoutError("partial allocation")

    def remove(self, handle):
        self.removed.append(handle)

    def removed_state(self, handle, *, timeout):
        return not self.unreachable


@pytest.mark.parametrize("unreachable", [False, True])
def test_partial_placement_creation_rolls_back_only_recorded_ids(unreachable):
    p = plan()
    registry = ResourceRegistry(p["run_id"], p["plan_hash"])
    backend = PlacementBackend(fail_at=2, unreachable=unreachable)
    allocator = PlacementAllocator(p, registry=registry, backend=backend)
    with pytest.raises(TimeoutError, match="partial allocation"):
        allocator.allocate(deadline=Deadline.after(2), cleanup_timeout=1)
    assert set(backend.removed) == set(backend.created)
    assert registry.cleanup_complete is not unreachable
    before = list(backend.removed)
    allocator.stop(deadline=Deadline.after(1))
    assert backend.removed == before
    assert backend.timeouts[1] <= backend.timeouts[0]


def test_actor_registered_before_readiness_and_partial_failure_rolls_back():
    from deepspec.pipeline.controller import ActorAllocator

    p = plan()
    registry = ResourceRegistry(p["run_id"], p["plan_hash"])
    removed = []

    class Backend:
        def create(self, cls, *, args, kwargs, options, node_id):
            assert options["max_restarts"] == options["max_task_retries"] == 0
            return options["name"]

        def identity(self, actor):
            return actor

        def get(self, ref, *, timeout):
            assert len(registry.owned_resources()) == 1
            raise TimeoutError("constructor failed")

        def kill(self, actor):
            removed.append(actor)

        def dead(self, actor, *, timeout):
            return True

    actors = ActorAllocator(p, registry=registry, backend=Backend())
    handle = actors.create(
        "service", object, node_id="node-a", role="store", deadline=Deadline.after(1)
    )
    with pytest.raises(TimeoutError, match="constructor"):
        actors.resolve("service", "ready-ref", deadline=Deadline.after(1))
    assert actors.stop(deadline=Deadline.after(1))["cleanup_complete"]
    actors.stop(deadline=Deadline.after(1))
    assert removed == [handle]


def test_blocking_native_start_keeps_control_responsive_and_cannot_revive(
    tmp_path, monkeypatch
):
    import json
    import threading
    from types import SimpleNamespace

    from deepspec.pipeline.actors import Producer

    entered, release = threading.Event(), threading.Event()

    class BlockingProducer(Producer):
        def run(self):
            entered.set()
            release.wait(2)
            return {"samples": 4}

    config = {
        "run_id": "control-probe",
        "plan_hash": "h",
        "timeout_seconds": 2,
        "buffer_name": "synthetic",
        "namespace": "test",
    }
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(config))
    buffer = SimpleNamespace(fail=SimpleNamespace(remote=lambda reason: None))
    monkeypatch.setattr(
        "deepspec.pipeline.actors.ray.get_actor", lambda *a, **kw: buffer
    )
    monkeypatch.setattr(
        "deepspec.pipeline.actors.get_with_deadline", lambda *a, **kw: None
    )
    actor = BlockingProducer(path)
    try:
        assert actor.start()["started"]
        assert entered.wait(1)
        assert actor.status()["state"] == "running"
        assert not actor.ready()["ready"]
        assert not actor.stop("cancelled", timeout=0.01)["cleanup_complete"]
        assert actor.status()["state"] == "stopped"
        with pytest.raises(RuntimeError, match="stopped"):
            actor.start()
        release.set()
        assert actor.stop("cancelled", timeout=1)["cleanup_complete"]
        assert actor.status()["state"] == "stopped"
    finally:
        release.set()
        if hasattr(actor, "stop"):
            actor.stop("test cleanup", timeout=2)


@pytest.mark.parametrize(
    "failed_phase", [None, "allocate", "initialize", "ready", "run", "drain", "verify"]
)
def test_controller_preserves_first_failure_and_verifies_before_success(
    tmp_path, failed_phase
):
    from deepspec.pipeline.controller import RunController

    config = task_config("M0", output_dir=tmp_path)
    p = build_plan(
        config, node_facts(config), input_plan(config), run_id="controller-run", now=100
    ).to_dict()
    calls, cleanup_deadlines = [], []

    class Operations:
        def check(self):
            pass

        def __getattr__(self, name):
            def call(*, deadline, **kwargs):
                calls.append(name)
                if name in RunController.CLEANUP_STEPS:
                    cleanup_deadlines.append(deadline.expires_at)
                if name == failed_phase:
                    raise RuntimeError("first failure: " + name)
                if failed_phase and name == "close_objects":
                    raise RuntimeError("secondary cleanup failure")
                return {
                    "ready": True,
                    "verified": True,
                    "independent": True,
                    "run_id": p["run_id"],
                    "plan_hash": p["plan_hash"],
                    "cleanup_complete": True,
                }

            return call

    result = RunController(p, Operations()).run()
    if failed_phase:
        assert result["state"] == "failed"
        assert "first failure: " + failed_phase in result["reason"]["message"]
        assert "secondary cleanup failure" in str(result["cleanup"])
        assert not result["cleanup_complete"]
    else:
        assert result["state"] == "succeeded" and result["cleanup_complete"]
        assert calls.index("verify") < calls.index("verify_cleanup")
    assert len(set(cleanup_deadlines)) == 1
    with pytest.raises(PipelineError, match="executed"):
        RunController(p, Operations()).run()


def test_controller_run_deadline_does_not_reset_cleanup_or_accept_late_success(
    tmp_path,
):
    import threading

    from deepspec.pipeline.controller import RunController

    config = task_config("M0", output_dir=tmp_path)
    config["timeouts_seconds"].update(run=1, cleanup=2)
    p = build_plan(
        config, node_facts(config), input_plan(config), run_id="timeout-run", now=100
    ).to_dict()
    release = threading.Event()

    class Operations:
        def check(self):
            pass

        def run(self, **kwargs):
            release.wait(5)
            return {"ready": True}

        def __getattr__(self, name):
            def call(**kwargs):
                if name == "stop_groups":
                    release.set()
                return {"ready": True, "cleanup_complete": True}

            return call

    try:
        result = RunController(p, Operations()).run()
        assert result["state"] == "failed" and result["cleanup_complete"]
        assert result["reason"]["code"] == "RUN_DEADLINE"
    finally:
        release.set()

"""Training ownership, native rank membership and readiness on CPU doubles."""

import asyncio
from types import SimpleNamespace

import pytest

from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import message_envelope
from deepspec.pipeline.vllm_adapter import NativeCoordination
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


@pytest.fixture(params=["M1-12", "M3"])
def coordination(tmp_path, request):
    config = task_config(request.param, output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="training-test", now=100
    ).to_dict()

    def agent(node):
        async def register(request):
            assert request["fencing_token"] == "fence-" + node
            return message_envelope(
                plan["run_id"],
                plan["plan_hash"],
                {"component": "agent"},
                process=request["process"],
            )

        async def devices(request):
            return message_envelope(
                plan["run_id"],
                plan["plan_hash"],
                {"component": "agent"},
                node_id=node,
                gpu_uuids=request["gpu_uuids"],
                external_processes=[],
            )

        return SimpleNamespace(
            register_process=SimpleNamespace(remote=register),
            check_allocated_devices=SimpleNamespace(remote=devices),
        )

    nodes = [n["node_id"] for n in plan["nodes"].values()]
    result = NativeCoordination(
        plan,
        pg_ids={p["id"]: "actual-" + p["id"] for p in plan["placement_groups"]},
        node_agents={node: agent(node) for node in nodes},
        tokens={node: "fence-" + node for node in nodes},
    )
    yield result
    result.close_events()


def message(gate, **payload):
    return message_envelope(
        gate.plan["run_id"], gate.plan["plan_hash"], {"component": "test"}, **payload
    )


def process(gate, pid):
    return {
        "pid": pid,
        "start_ticks": pid + 100,
        "run_id": gate.plan["run_id"],
        "parent_pid": 90,
        "group_id": 100,
    }


async def allocated(gate):
    for index, (key, expected) in enumerate(gate.expected.items()):
        rank = int(key.split("/")[-1])
        available = [g["uuid"] for g in gate.nodes[expected["node_id"]]["gpus"]]
        devices = (
            available[rank : rank + 1]
            if key.startswith("inference/")
            else available[: expected["gpu_count"]]
        )
        identity = process(gate, 200 + index)
        report = message(
            gate,
            **expected,
            participant=key,
            actor_id=f"actor-{index}",
            gpu_uuids=devices,
            process=identity,
            pid=identity["pid"],
            start_ticks=identity["start_ticks"],
        )
        if key.startswith("training/"):
            await gate.observe_launcher(report, timeout=1)
        else:
            await gate.report_allocation(report)


def rank_started(gate, index):
    rank = gate.plan["training_ranks"][index]
    return message(
        gate,
        **rank,
        participant=f"rank/{index}",
        world=len(gate.plan["training_ranks"]),
        process=process(gate, 300 + index),
    )


def rank_initialized(gate, index):
    started = rank_started(gate, index)
    rank = gate.plan["training_ranks"][index]
    return {
        **started,
        "pid": started["process"]["pid"],
        "start_ticks": started["process"]["start_ticks"],
        "gas": gate.plan["counts"]["gas"],
        "input_plan_hash": gate.plan["input_plan_hash"],
        "model_identity": gate.nodes[rank["node_id"]]["identities"]["model"],
        "gpu_uuid": gate.allocations[f"training/{rank['node_rank']}"]["gpu_uuids"][
            rank["local_rank"]
        ],
        "tp_members": [
            r["global_rank"]
            for r in gate.plan["training_ranks"]
            if r["dp_rank"] == rank["dp_rank"]
        ],
        "dp_members": [
            r["global_rank"]
            for r in gate.plan["training_ranks"]
            if r["tp_rank"] == rank["tp_rank"]
        ],
    }


def connector_report(gate, key):
    allocation = gate.allocations[key]
    _, replica, tp = key.split("/")
    return message(
        gate,
        participant=key,
        **{
            field: allocation[field]
            for field in ("node_id", "actor_id", "gpu_uuids", "pid", "start_ticks")
        },
        dp_rank=int(replica),
        tp_rank=int(tp),
        tp_world_size=gate.plan["config"]["inference"]["tp"],
        writer=int(tp) == 0,
    )


@pytest.mark.parametrize("connector_first", [False, True])
def test_inference_requires_native_and_connector_readiness(
    coordination, connector_first
):
    async def scenario():
        gate = coordination
        await allocated(gate)
        key = "inference/0/0"
        connector = connector_report(gate, key)
        native = message(
            gate,
            **{
                field: connector[field]
                for field in (
                    "participant",
                    "node_id",
                    "actor_id",
                    "tp_rank",
                    "dp_rank",
                )
            },
        )
        await gate.report_connector(connector, ready=False)
        assert not gate.connectors and not gate.initialized
        first, second = (
            (
                lambda: gate.report_connector(connector),
                lambda: gate.report_initialized(native),
            )
            if connector_first
            else (
                lambda: gate.report_initialized(native),
                lambda: gate.report_connector(connector),
            )
        )
        await first()
        assert key not in gate.initialized
        await second()
        assert key in gate.initialized
        # Duplicate delivery must not add a participant or change its identity.
        await first()
        await second()
        assert list(gate.initialized) == [key]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("writer", False),
        ("tp_rank", 1),
        ("dp_rank", 1),
        ("tp_world_size", 8),
        ("gpu_uuids", ["GPU-b-0"]),
        ("start_ticks", 99999),
        ("node_id", "node-b"),
    ],
)
def test_wrong_connector_identity_aborts_before_store_creation(
    coordination, field, value
):
    async def scenario():
        gate = coordination
        await allocated(gate)
        report = connector_report(gate, "inference/0/0")
        report[field] = value
        with pytest.raises(ValueError, match="identity differs"):
            await gate.report_connector(report, ready=False)
        assert not gate.connectors and not gate.initialized
        with pytest.raises(ValueError):
            await gate.wait_for_initialization(timeout=1)

    asyncio.run(scenario())


def test_all_native_ranks_must_match_registered_processes_before_ready(coordination):
    async def scenario():
        gate = coordination
        await allocated(gate)
        for key, value in gate.allocations.items():
            if key.startswith("inference/"):
                _, replica, rank = key.split("/")
                await gate.report_initialized(
                    message(
                        gate,
                        participant=key,
                        node_id=value["node_id"],
                        actor_id=value["actor_id"],
                        dp_rank=int(replica),
                        tp_rank=int(rank),
                    )
                )
                await gate.report_connector(connector_report(gate, key))
        await gate.report_initialized(
            message(
                gate,
                participant="store",
                node_id=gate.plan["services"]["pool_node_id"],
                transport_verified=True,
            )
        )
        for rank in range(8):
            await gate.register_training_process(rank_started(gate, rank), timeout=1)
        waiter = asyncio.create_task(gate.wait_for_initialization(timeout=2))
        for rank in range(7):
            await gate.report_initialized(rank_initialized(gate, rank))
        await asyncio.sleep(0)
        assert not waiter.done()
        await gate.report_initialized(rank_initialized(gate, 7))
        assert (await waiter)["ready"]
        assert len(gate.rank_processes) == 8

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault",
    [
        "unregistered",
        "reused_pid",
        "wrong_gpu",
        "wrong_dp_group",
        "wrong_gas",
        "wrong_world",
    ],
)
def test_native_rank_mismatch_aborts_shared_initialization_gate(coordination, fault):
    async def scenario():
        gate = coordination
        await allocated(gate)
        if fault != "unregistered":
            await gate.register_training_process(rank_started(gate, 0), timeout=1)
        report = rank_initialized(gate, 0)
        field, value = {
            "unregistered": ("pid", 300),
            "reused_pid": ("start_ticks", 99999),
            "wrong_gpu": ("gpu_uuid", "GPU-b-7"),
            "wrong_dp_group": ("dp_members", [0, 1]),
            "wrong_gas": ("gas", 4),
            "wrong_world": ("world", 4),
        }[fault]
        report[field] = value
        with pytest.raises(ValueError):
            await gate.report_initialized(report)
        with pytest.raises(ValueError):
            await gate.wait_for_initialization(timeout=1)
        assert not gate.initialized

    asyncio.run(scenario())


def test_rank_cannot_start_before_all_inference_and_training_allocations(coordination):
    async def scenario():
        with pytest.raises(ValueError, match="all-role allocation"):
            await coordination.register_training_process(
                rank_started(coordination, 0), timeout=1
            )
        assert not coordination.rank_processes

    asyncio.run(scenario())


def test_consumer_does_not_launch_torchrun_while_all_role_gate_is_closed(
    tmp_path, monkeypatch
):
    import json

    from deepspec.orchestration import process as process_module
    from deepspec.pipeline.actors import Consumer

    config = task_config("M0", output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="before-gpu", now=100
    ).to_dict()
    path = tmp_path / "pipeline.json"
    path.write_text(
        json.dumps(
            {
                "run_id": plan["run_id"],
                "plan_hash": plan["plan_hash"],
                "timeout_seconds": 2,
                "timeouts_seconds": plan["timeouts_seconds"],
                "namespace": "test",
                "buffer_name": "test",
                "consumer_world_size": 4,
                "consumer_dp": 1,
                "consumer_nodes": 1,
            }
        )
    )
    gate = SimpleNamespace(
        wait_for_allocation=SimpleNamespace(remote=lambda **kwargs: "allocation-gate")
    )
    consumer = Consumer(path, native_plan=plan, gate=gate)
    monkeypatch.setattr(
        "deepspec.pipeline.actors.ray.get_actor", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "deepspec.pipeline.actors.ray.get_runtime_context",
        lambda: SimpleNamespace(
            get_accelerator_ids=lambda: {"GPU": ["1", "3", "5", "7"]}
        ),
    )

    def closed(ref, **kwargs):
        assert ref == "allocation-gate"
        raise TimeoutError("gate closed")

    monkeypatch.setattr(consumer, "_get", closed)
    monkeypatch.setattr(
        process_module,
        "start_owned",
        lambda *args, **kwargs: pytest.fail(
            "torchrun cannot start before allocation gate"
        ),
    )
    try:
        with pytest.raises(TimeoutError, match="gate closed"):
            consumer.run()
    finally:
        consumer._executor.shutdown(wait=False)

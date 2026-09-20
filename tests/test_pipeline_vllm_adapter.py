"""Native callbacks, identity and ownership contracts with explicit CPU doubles."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import message_envelope
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


@pytest.mark.parametrize("wrong_device", [False, True])
def test_native_actor_observation_is_owned_before_gate_and_creator_can_arrive_late(
    tmp_path, wrong_device
):
    from deepspec.pipeline.vllm_adapter import NativeCoordination

    config = task_config("M0", output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="adapter-run", now=100
    ).to_dict()
    acknowledgements = []

    async def register_process(request):
        assert request["fencing_token"] == "fence"
        acknowledgements.append(request["process"])
        return message_envelope(
            plan["run_id"],
            plan["plan_hash"],
            {"component": "node_agent"},
            process=request["process"],
        )

    async def check_devices(request):
        return message_envelope(
            plan["run_id"],
            plan["plan_hash"],
            {"component": "node_agent"},
            node_id="node-a",
            gpu_uuids=request["gpu_uuids"],
            external_processes=[],
        )

    agent = SimpleNamespace(
        register_process=SimpleNamespace(remote=register_process),
        check_allocated_devices=SimpleNamespace(remote=check_devices),
    )
    pg_ids = {p["id"]: "actual-" + p["id"] for p in plan["placement_groups"]}
    coordination = NativeCoordination(
        plan, pg_ids=pg_ids, node_agents={"node-a": agent}, tokens={"node-a": "fence"}
    )
    process = {
        "pid": 100,
        "start_ticks": 123,
        "run_id": plan["run_id"],
        "parent_pid": 90,
        "group_id": 100,
        "state": "S",
    }
    message = message_envelope(
        plan["run_id"],
        plan["plan_hash"],
        {"component": "native_test"},
        participant="inference/0/0",
        actor_id="worker-0",
        node_id="node-a",
        pg_id=pg_ids["inference-0"],
        bundle_index=0,
        process=process,
        gpu_uuids=["foreign-device" if wrong_device else "GPU-a-0"],
        pid=100,
        start_ticks=123,
    )
    actor = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "worker-0"))

    async def scenario():
        if wrong_device:
            with pytest.raises(ValueError):
                await coordination.observe_native(message, timeout=1)
            assert coordination.error is not None
            assert not coordination.allocations
        else:
            await coordination.observe_native(message, timeout=1)
            repeated = {
                **message,
                "event_id": "redelivered",
                "process": {**process, "state": "R"},
            }
            await coordination.observe_native(repeated, timeout=1)
            await coordination.register_native(message, actor)
            await coordination.register_native(message, actor)
            assert coordination.allocations["inference/0/0"]["actor_id"] == "worker-0"
            assert len(coordination.native_registry.owned_resources()) == 1
            resource = coordination.native_registry.owned_resources()[0]
            assert resource.borrower == "vLLM" and resource.process.pid == 100
            assert resource.gpu_uuids == ("GPU-a-0",)
        assert len(acknowledgements) == 1
        stored = json.loads((tmp_path / "native-allocation.json").read_text())
        assert stored["run_id"] == plan["run_id"]
        assert stored["resources"][0]["ray_id"] == "worker-0"
        coordination.close_events()

    asyncio.run(scenario())


def test_actual_bundle_index_is_derived_from_ray_resource_identity():
    from deepspec.pipeline.vllm_adapter import observed_bundle

    assert (
        observed_bundle({"GPU_group_3_abc": [(7, 1)], "GPU_group_abc": [(7, 1)]}, "abc")
        == 3
    )
    for resources in (
        {"GPU": [(7, 1)]},
        {"GPU_group_3_foreign": [(7, 1)]},
        {"GPU_group_3_abc": [(7, 1)], "GPU_group_4_abc": [(8, 1)]},
    ):
        with pytest.raises(ValueError):
            observed_bundle(resources, "abc")


@pytest.mark.parametrize(
    "key,value",
    [
        ("VLLM_DP_SIZE", "2"),
        ("VLLM_DP_RANK", "1"),
        ("VLLM_DP_RANK_LOCAL", "0"),
        ("VLLM_DP_MASTER_IP", "foreign"),
        ("VLLM_RAY_BUNDLE_INDICES", "4,5,6,7"),
        ("VLLM_RAY_PER_WORKER_GPUS", "0.5"),
        ("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0"),
        ("VLLM_RAY_DP_PACK_STRATEGY", "span"),
        ("VLLM_RAY_DP_PLACEMENT_NODE_IPS", "foreign"),
    ],
)
def test_env_cannot_override_immutable_native_placement(monkeypatch, key, value):
    from deepspec.pipeline.vllm_adapter import validate_native_environment

    config = task_config("M0")
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="native-test", now=100
    ).to_dict()
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="conflicts"):
        validate_native_environment(plan)


def test_native_hook_passes_handles_outside_json_and_uses_actual_physical_gpu(
    monkeypatch,
):
    import ray

    from deepspec.orchestration import process as process_module
    from deepspec.pipeline import cluster
    from deepspec.pipeline.vllm_adapter import NativePlacementHooks

    config = task_config("M0")
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="native-test", now=100
    ).to_dict()
    calls, waits = [], []

    def call(method):
        def remote(*args, **kwargs):
            json.dumps(args[0])
            calls.append((method, args, kwargs))
            return message_envelope(
                plan["run_id"], plan["plan_hash"], {"component": "gate"}, accepted=True
            )

        return SimpleNamespace(remote=remote)

    def get(value, *, timeout):
        waits.append(timeout)
        return value

    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(
        cluster,
        "gpu_inventory",
        lambda **kwargs: [
            {"index": "7", "uuid": "GPU-a-3"},
            {"index": "0", "uuid": "GPU-unassigned"},
        ],
    )
    monkeypatch.setattr(
        process_module,
        "capture_process",
        lambda *args: {
            "pid": 100,
            "start_ticks": 123,
            "run_id": plan["run_id"],
            "parent_pid": 90,
            "group_id": 100,
        },
    )
    gate = SimpleNamespace(
        register_native=call("created"), observe_native=call("observed")
    )
    hook = NativePlacementHooks(plan, gate)
    actor = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "worker-3"))
    common = {"replica": 0, "rank": 3, "node_id": "node-a", "pg_id": "abc"}
    hook("worker_created", {**common, "actor": actor, "bundle_index": 3}, timeout=2)
    hook(
        "worker_allocated",
        {
            **common,
            "actor_id": "worker-3",
            "physical_gpu_ids": [7],
            "resource_ids": {"GPU_group_3_abc": [(7, 1)]},
        },
        timeout=2,
    )
    assert calls[0][1][1] is actor
    observed = calls[1][1][0]
    assert observed["gpu_uuids"] == ["GPU-a-3"]
    assert observed["bundle_index"] == 3 and observed["participant"] == "inference/0/3"
    assert observed["pid"] == 100 and all(0 < wait <= 2 for wait in waits)


def test_native_cleanup_releases_only_registered_actors_and_preserves_unknown(
    tmp_path, monkeypatch
):
    from deepspec.pipeline.controller import RayActorBackend
    from deepspec.pipeline.vllm_adapter import NativeCoordination

    config = task_config("M0", output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="native-test", now=100
    ).to_dict()
    pg_ids = {p["id"]: "actual-" + p["id"] for p in plan["placement_groups"]}
    coordination = NativeCoordination(
        plan,
        pg_ids=pg_ids,
        node_agents={"node-a": object()},
        tokens={"node-a": "fence"},
    )
    killed = []
    monkeypatch.setattr(
        RayActorBackend,
        "kill",
        lambda self, actor: killed.append(actor._actor_id.hex()),
    )
    monkeypatch.setattr(
        RayActorBackend,
        "dead",
        lambda self, actor, **kwargs: actor._actor_id.hex() == "worker-0",
    )

    async def scenario():
        for rank in range(2):
            identity = f"worker-{rank}"
            actor = SimpleNamespace(
                _actor_id=SimpleNamespace(hex=lambda identity=identity: identity)
            )
            message = message_envelope(
                plan["run_id"],
                plan["plan_hash"],
                {"component": "creator"},
                participant=f"inference/0/{rank}",
                actor_id=identity,
                node_id="node-a",
                pg_id=pg_ids["inference-0"],
                bundle_index=rank,
            )
            await coordination.register_native(message, actor)
        assert not (await coordination.stop_native("cancelled", timeout=2))[
            "cleanup_complete"
        ]
        assert not (await coordination.stop_native("cancelled", timeout=2))[
            "cleanup_complete"
        ]
        assert sorted(killed) == ["worker-0", "worker-1"]
        assert [
            r.release_state for r in coordination.native_registry.owned_resources()
        ] == ["released", "unknown"]
        coordination.close_events()

    asyncio.run(scenario())

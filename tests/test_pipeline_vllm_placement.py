"""Native vLLM placement seams with explicit CPU doubles; no GPU allocations."""

import copy
import json
import os
from types import SimpleNamespace

import pytest


class PlacementGroup:
    def __init__(self, index, tp):
        self.id = SimpleNamespace(hex=lambda: f"pg-{index}")
        self.bundle_specs = [{"GPU": 1.0} for _ in range(tp)] + [{"CPU": 1.0}]


def placement(tp=4, dp=2):
    from vllm.config.parallel import ParallelConfig

    parallel = ParallelConfig(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        data_parallel_size_local=dp,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
    )
    groups = [PlacementGroup(i, tp) for i in range(dp)]
    plan = {
        "version": 1,
        "run_id": "cpu-seam",
        "plan_hash": "frozen-hash",
        "timeout_seconds": 3,
        "replicas": [
            {
                "placement_group_id": f"pg-{i}",
                "worker_bundle_indices": list(range(tp)),
                "bundle_node_ids": ["node-a"] * (tp + 1),
                "core_bundle_index": tp,
            }
            for i in range(dp)
        ],
    }
    return parallel, groups, plan


def test_runtime_handles_do_not_change_graph_hash_or_enter_json():
    from pydantic import TypeAdapter

    parallel, groups, plan = placement()
    before = parallel.compute_hash()
    parallel.bind_ray_placement(groups, [0, 1], plan, lambda *a, **kw: None)
    assert parallel.compute_hash() == before
    exported = json.loads(TypeAdapter(type(parallel)).dump_json(parallel))
    assert "ray_placement_groups" not in exported
    assert "ray_placement_callback" not in exported
    assert exported["ray_placement_plan"] == plan
    cloned = copy.deepcopy(parallel)
    assert [pg.id.hex() for pg in cloned.ray_placement_groups] == ["pg-0", "pg-1"]


@pytest.mark.parametrize("layout", ["M0", "M1-21"])
def test_adapter_dp1_and_dp2_enter_from_vllm_config_and_missing_capability_stops_start(
    layout, monkeypatch
):
    import vllm.engine.arg_utils as args_module
    import vllm.v1.engine.core as core_module
    from vllm.v1.engine.async_llm import AsyncLLM

    from deepspec.pipeline.planning import build_plan
    from deepspec.pipeline.vllm_adapter import NativeInferenceAdapter
    from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config

    config = task_config(layout)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="native-test", now=100
    ).to_dict()
    parallel, groups, _ = placement(dp=config["inference"]["dp"])
    parallel.data_parallel_master_ip = plan["nodes"]["a"]["ip"]
    calls, shutdowns = [], []

    class Args:
        def __init__(self, **values):
            calls.append(values)

        def create_engine_config(self):
            return SimpleNamespace(parallel_config=parallel)

    engine = SimpleNamespace(shutdown=lambda **kwargs: shutdowns.append(kwargs))

    def create(cls, value):
        assert value.parallel_config.ray_placement_groups == groups
        assert value.parallel_config.ray_placement_callback is not None
        return engine

    monkeypatch.setattr(args_module, "AsyncEngineArgs", Args)
    monkeypatch.setattr(AsyncLLM, "from_vllm_config", classmethod(create))
    monkeypatch.setattr(
        AsyncLLM,
        "from_engine_args",
        lambda *args, **kwargs: pytest.fail(
            "Adapter must bind native config before launch"
        ),
    )
    adapter = NativeInferenceAdapter(plan, groups, object(), runtime_env={})
    assert adapter.start({}, timeout=2) is engine
    assert calls[0]["data_parallel_size"] == config["inference"]["dp"]
    assert adapter.status() == {"state": "initialized", "error": None}
    adapter.stop(timeout=2)
    adapter.stop(timeout=2)
    assert len(shutdowns) == 1 and shutdowns[0]["timeout"] > 0
    monkeypatch.setattr(core_module, "RAY_PLACEMENT_API_VERSION", 0)
    unsupported = NativeInferenceAdapter(plan, groups, object(), runtime_env={})
    with pytest.raises(ValueError, match="Missing native placement"):
        unsupported.start({}, timeout=2)
    assert len(calls) == 1 and unsupported.status()["state"] == "failed"


@pytest.mark.parametrize("layout", ["M0", "M1-21", "M2-DP1", "M2-DP2"])
def test_adapter_binds_effective_native_config_without_overriding_ray_gpu_visibility(
    layout, monkeypatch
):
    from deepspec.pipeline.planning import build_plan
    from deepspec.pipeline.vllm_adapter import bind_native_config
    from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config

    config = task_config(layout)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="native-test", now=100
    ).to_dict()
    tp, dp = config["inference"]["tp"], config["inference"]["dp"]
    monkeypatch.delenv("VLLM_RAY_BUNDLE_INDICES", raising=False)
    parallel, groups, _ = placement(tp=tp, dp=dp)
    ip = plan["nodes"]["a"]["ip"]
    parallel.data_parallel_master_ip = "127.0.0.1" if dp == 1 else ip
    bound = bind_native_config(
        SimpleNamespace(parallel_config=parallel),
        plan,
        groups,
        object(),
        timeout=2,
        runtime_env={"env_vars": {"PYTHONPATH": "/fixture"}},
    )
    assert bound.parallel_config is parallel
    assert parallel.data_parallel_master_ip == ip
    assert parallel.ray_placement_local_dp_ranks == list(range(dp))
    assert parallel.ray_runtime_env["env_vars"]["VLLM_RAY_BUNDLE_INDICES"] == ",".join(
        map(str, range(tp))
    )
    assert "CUDA_VISIBLE_DEVICES" not in parallel.ray_runtime_env["env_vars"]
    assert parallel.ray_placement_plan["replicas"][0]["bundle_node_ids"] == [
        *plan["replicas"][0]["worker_nodes"],
        "node-a",
    ]
    parallel.data_parallel_size = dp + 1
    with pytest.raises(ValueError, match="data_parallel_size"):
        bind_native_config(bound, plan, groups, object(), timeout=2, runtime_env={})


@pytest.mark.parametrize(
    "fault",
    [
        "missing_pg",
        "missing_local_rank",
        "fractional_gpu",
        "gpu_core",
        "missing_node",
        "wrong_pg",
    ],
)
def test_borrowed_placement_rejects_incomplete_or_conflicting_declarations(fault):
    parallel, groups, plan = placement()
    local_ranks = [0, 1]
    if fault == "missing_pg":
        groups.pop()
    elif fault == "missing_local_rank":
        local_ranks.pop()
    elif fault == "fractional_gpu":
        groups[0].bundle_specs[0]["GPU"] = 0.5
    elif fault == "gpu_core":
        groups[1].bundle_specs[-1]["GPU"] = 1
    elif fault == "missing_node":
        plan["replicas"][0]["bundle_node_ids"].pop()
    else:
        plan["replicas"][0]["placement_group_id"] = "foreign-pg"
    with pytest.raises(ValueError):
        parallel.bind_ray_placement(groups, local_ranks, plan, lambda *a, **kw: None)


def test_borrowed_cpu_core_keeps_dp_identity_without_local_gpu_range(monkeypatch):
    from vllm.v1.engine.core import EngineCoreActorMixin

    parallel, groups, plan = placement(tp=8)
    parallel.bind_ray_placement(groups, [0, 1], plan, lambda *a, **kw: None)
    parallel.placement_group = groups[1]
    parallel.data_parallel_rank_local = 1
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    core = object.__new__(EngineCoreActorMixin)
    monkeypatch.setattr(
        core,
        "_set_assigned_physical_gpu_ids",
        lambda *a: pytest.fail("CPU core must not derive local GPU 8..15"),
    )
    core._set_visible_devices(SimpleNamespace(parallel_config=parallel), 1)
    assert parallel.data_parallel_rank_local == 1
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original


def test_deferred_native_worker_reports_raw_pg_resources_without_cuda_initialization(
    monkeypatch,
):
    import ray
    import ray._private.worker as ray_worker
    from vllm.config import VllmConfig
    from vllm.v1.executor.ray_executor_v2 import RayWorkerProc

    parallel, groups, plan = placement()
    events = []
    parallel.bind_ray_placement(
        groups,
        [0, 1],
        plan,
        lambda event, payload, **kwargs: events.append((event, payload)),
    )
    config = object.__new__(VllmConfig)
    config.parallel_config = parallel
    worker = object.__new__(RayWorkerProc)
    worker._init_kwargs = {"vllm_config": config, "rank": 3}
    monkeypatch.setattr(
        worker, "get_node_and_physical_gpu_ids", lambda: ("node-a", [7])
    )
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_actor_id=lambda: "worker-3"),
    )
    monkeypatch.setattr(ray.util, "get_current_placement_group", lambda: groups[0])
    raw = {"GPU_group_3_pg-0": [(7, 1)], "GPU_group_pg-0": [(7, 1)]}
    monkeypatch.setattr(ray_worker, "get_resource_ids", lambda: raw)
    report = worker.get_placement_report()
    assert report["resource_ids"] == raw and report["physical_gpu_ids"] == [7]
    assert report["rank"] == 3 and report["actor_id"] == "worker-3"
    assert events == [("worker_allocated", report)]


@pytest.mark.parametrize("fail_second", [False, True])
def test_native_manager_borrows_groups_and_cleans_partial_core_creation(
    monkeypatch, fail_second
):
    import ray
    from vllm.v1.engine.utils import CoreEngineActorManager, EngineZmqAddresses

    parallel, groups, plan = placement()
    events, actors, killed, removed, timeouts = [], [], [], [], []
    parallel.bind_ray_placement(
        groups, [0, 1], plan, lambda event, payload, **kw: events.append(event)
    )

    class Remote:
        def options(self, **options):
            assert options["num_cpus"] == 1 and options["num_gpus"] == 0
            assert options["max_restarts"] == options["max_task_retries"] == 0
            return self

        def remote(self, **kwargs):
            if fail_second and actors:
                raise RuntimeError("second core allocation failed")

            class Actor:
                pass

            actor = Actor()
            actor._actor_id = SimpleNamespace(hex=lambda: f"actor-{len(actors)}")
            actor.wait_for_init = SimpleNamespace(remote=lambda: "init")
            actor.run = SimpleNamespace(remote=lambda: "run")
            actors.append(actor)
            return actor

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "remote", lambda cls: Remote())
    monkeypatch.setattr(ray, "get", lambda refs, *, timeout: timeouts.append(timeout))
    monkeypatch.setattr(ray, "kill", lambda actor, **kw: killed.append(actor))
    monkeypatch.setattr(ray.util, "remove_placement_group", removed.append)
    monkeypatch.setattr(
        CoreEngineActorManager,
        "create_dp_placement_groups",
        lambda *a: pytest.fail("Borrowed groups must bypass automatic placement"),
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        model_config=SimpleNamespace(is_moe=False),
        instance_id="probe",
        kv_transfer_config=None,
    )
    kwargs = {
        "vllm_config": config,
        "addresses": EngineZmqAddresses(inputs=[], outputs=[]),
        "executor_class": object,
        "log_stats": False,
    }
    if fail_second:
        with pytest.raises(RuntimeError, match="second core"):
            CoreEngineActorManager(**kwargs)
    else:
        manager = CoreEngineActorManager(**kwargs)
        assert manager.created_placement_groups == []
        manager.shutdown()
        assert timeouts and 0 < timeouts[0] <= 3
    assert killed == actors
    assert not removed


@pytest.mark.parametrize("reject_gate", [False, True])
def test_v2_collects_all_device_reports_and_waits_before_any_worker_initializes(
    monkeypatch, reject_gate
):
    import time

    import vllm.v1.executor.ray_executor_v2 as native

    parallel, groups, plan = placement(dp=1)
    calls = []

    def coordinate(event, payload, **kwargs):
        calls.append(event)
        if event == "allocation_ready":
            assert calls.count("placement_report") == 4
            assert "initialize" not in calls
            if reject_gate:
                raise TimeoutError("Training launcher allocation is missing")

    parallel.bind_ray_placement(groups, [0], plan, coordinate)
    executor = object.__new__(native.RayExecutorV2)
    executor.parallel_config = parallel
    executor.vllm_config = SimpleNamespace(parallel_config=parallel)
    executor.driver_env_vars = {}
    executor._placement_deadline = time.monotonic() + 3
    executor.ray_worker_handles = []
    for rank in range(4):

        def identify(rank=rank):
            calls.append("device_report")
            return ("node-a", [rank * 2])

        def report():
            calls.append("placement_report")
            return {"observed": True}

        def initialize(*args, **kwargs):
            assert "allocation_ready" in calls
            calls.append("initialize")

        actor = SimpleNamespace(
            get_node_and_physical_gpu_ids=SimpleNamespace(remote=identify),
            get_placement_report=SimpleNamespace(remote=report),
            initialize_worker=SimpleNamespace(remote=initialize),
            wait_for_init=SimpleNamespace(remote=lambda: {"status": "READY"}),
        )
        executor.ray_worker_handles.append(
            native.RayWorkerHandle(
                actor=actor,
                rank=rank,
                local_rank=-1,
                node_id="node-a",
                bundle_id_idx=rank,
            )
        )
    waits = []

    def get(refs, *, timeout):
        waits.append(timeout)
        return refs

    monkeypatch.setattr(native, "ray", SimpleNamespace(get=get))
    if reject_gate:
        with pytest.raises(TimeoutError, match="launcher"):
            executor._initialize_workers()
        assert "initialize" not in calls
    else:
        assert len(executor._initialize_workers()) == 4
        assert calls.count("initialize") == 4
        assert parallel.assigned_physical_gpu_ids == [0, 2, 4, 6]
    assert calls.count("device_report") == 4
    assert all(0 < timeout <= 3 for timeout in waits)

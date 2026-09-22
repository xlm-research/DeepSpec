"""Service ownership and rollback contracts; no native service is launched."""

from types import SimpleNamespace

import pytest

from deepspec.pipeline.controller import ActorAllocator, ResourceRegistry
from deepspec.pipeline.groups import StoreService
from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import Deadline, message_envelope
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


@pytest.mark.parametrize("layout", ["M0", "M1-12"])
@pytest.mark.parametrize("wandb_mode", [None, "offline"])
def test_training_group_owns_whole_local_bundle_and_registers_before_torchrun(
    tmp_path, layout, wandb_mode, monkeypatch
):
    from deepspec.pipeline.groups import TrainingGroup

    wandb_env = {
        "WANDB_PROJECT": "dspark-test",
        "WANDB_NAME": "training-test",
        "WANDB_API_KEY": "test-only-key",
    }
    monkeypatch.delenv("WANDB_MODE", raising=False)
    if wandb_mode is not None:
        wandb_env["WANDB_MODE"] = wandb_mode
    for key, value in wandb_env.items():
        monkeypatch.setenv(key, value)
    config = task_config(layout, output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="train-group", now=100
    ).to_dict()
    rank = plan["training_ranks"][0]
    world = rank["local_world_size"]
    registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
    pg = SimpleNamespace(id=SimpleNamespace(hex=lambda: "owned-training-pg"))
    registered, calls = [], []

    def method(name):
        return SimpleNamespace(remote=lambda *args, **kwargs: (name, args, kwargs))

    gate = SimpleNamespace(
        **{
            name: method(name)
            for name in ("observe_launcher", "wait_for_initialization", "native_failed")
        }
    )

    class Backend:
        def create(self, cls, *, args, kwargs, options, node_id):
            assert cls.__name__ == "Consumer" and cls.owns_supervised_processes
            assert node_id == rank["node_id"] and kwargs["node_rank"] == 0
            assert options["num_gpus"] == world and options["num_cpus"] == 2 * world
            strategy = options["scheduling_strategy"]
            assert (
                strategy.placement_group is pg
                and strategy.placement_group_bundle_index == 0
            )
            assert strategy.placement_group_capture_child_tasks is False
            assert kwargs["gate"] is gate and kwargs["native_plan"] == plan
            env = options["runtime_env"]["env_vars"]
            assert {key: env[key] for key in wandb_env} == wandb_env
            assert env["WANDB_MODE"] == (wandb_mode or "disabled")
            return SimpleNamespace(
                **{
                    name: method(name)
                    for name in ("allocation", "start", "status", "stop")
                }
            )

        def identity(self, actor):
            return "launcher-id"

        def get(self, ref, *, timeout):
            name, _args, _kwargs = ref
            calls.append(name)
            assert timeout > 0
            if name == "allocation":
                return message_envelope(
                    plan["run_id"],
                    plan["plan_hash"],
                    {"component": "launcher"},
                    actor_id="launcher-id",
                    node_id=rank["node_id"],
                    participant="training/0",
                    pg_id="owned-training-pg",
                    bundle_index=0,
                    process={
                        "pid": 100,
                        "start_ticks": 200,
                        "parent_pid": 90,
                        "group_id": 100,
                        "run_id": plan["run_id"],
                    },
                    gpu_uuids=[f"allocated-device-{index}" for index in range(world)],
                )
            if name in ("observe_launcher", "start"):
                assert registered and registry.owned_resources()[0].process.pid == 100
            if name == "start":
                assert "observe_launcher" in calls
            return {
                "ready": True,
                "started": True,
                "cleanup_complete": True,
                "errors": [],
            }

        def kill(self, actor):
            calls.append("kill")

        def dead(self, actor, *, timeout):
            return True

    actors = ActorAllocator(plan, registry=registry, backend=Backend())
    group = TrainingGroup()
    try:
        resources = {
            "actors": actors,
            "gate": gate,
            "placement_groups": {"training-0": pg},
            "config_path": tmp_path / "pipeline.json",
            "register_process": lambda identity, **kwargs: registered.append(identity),
        }
        group.allocate(plan, resources=resources, deadline=Deadline.after(2)).result(
            deadline=Deadline.after(2)
        )
        group.start(gate=gate, deadline=Deadline.after(2)).result(
            deadline=Deadline.after(2)
        )
        assert group.ready(deadline=Deadline.after(2))["ready"]
        assert not group.stop("finished", deadline=Deadline.after(2)).unknown
        assert len(registry.owned_resources()) == 1 and calls.count("kill") == 1
    finally:
        group.stop("test cleanup", deadline=Deadline.after(2))


def test_native_inference_group_uses_separate_cpu_frontend_and_borrows_exact_pgs(
    tmp_path,
):
    from deepspec.pipeline.controller import Allocation
    from deepspec.pipeline.groups import InferenceGroup

    config = task_config("M1-21", output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="group-run", now=100
    ).to_dict()
    registry = ResourceRegistry(plan["run_id"], plan["plan_hash"])
    groups = {}
    for replica in plan["replicas"]:
        name = f"inference-{replica['replica_id']}"
        groups[name] = SimpleNamespace(id=SimpleNamespace(hex=lambda name=name: name))
        registry.register(
            Allocation(
                name,
                plan["run_id"],
                plan["plan_hash"],
                "DeepSpec",
                "placement_group",
                "inference",
                name,
                "node-a",
            )
        )
        registry.transition(name, "acquiring")
        registry.transition(name, "acquired")
    calls, registered = [], []

    def remote(method):
        return SimpleNamespace(remote=lambda *args, **kwargs: (method, args, kwargs))

    gate = SimpleNamespace(
        **{
            method: remote(method)
            for method in ("wait_for_initialization", "native_failed", "stop_native")
        }
    )

    class Backend:
        def create(self, cls, *, args, kwargs, options, node_id):
            assert cls.__name__ == "Producer" and node_id == "node-a"
            assert options["num_cpus"] == 1 and options["num_gpus"] == 0
            assert "scheduling_strategy" not in options
            assert kwargs["placement_groups"] == list(groups.values())
            assert kwargs["gate"] is gate
            assert (
                options["runtime_env"]["env_vars"]["VLLM_RAY_BUNDLE_INDICES"]
                == "0,1,2,3"
            )
            calls.append("create_frontend")
            return SimpleNamespace(
                **{
                    method: remote(method)
                    for method in ("identity", "start", "status", "stop")
                }
            )

        def identity(self, actor):
            return "frontend-id"

        def get(self, ref, *, timeout):
            method, _args, kwargs = ref
            assert timeout > 0
            if method == "identity":
                return message_envelope(
                    plan["run_id"],
                    plan["plan_hash"],
                    {"component": "frontend"},
                    actor_id="frontend-id",
                    node_id="node-a",
                    process={"pid": 100},
                )
            if method == "start":
                assert registered
                assert kwargs["initialization_timeout"] > 0
            return {
                "started": True,
                "ready": True,
                "cleanup_complete": True,
                "error": None,
                "resources": [],
                "errors": [],
            }

        def kill(self, actor):
            calls.append("kill_frontend")

        def dead(self, actor, *, timeout):
            return True

    actors = ActorAllocator(plan, registry=registry, backend=Backend())
    group = InferenceGroup()
    resources = {
        "actors": actors,
        "gate": gate,
        "config_path": tmp_path / "config.json",
        "placement_groups": groups,
        "register_process": lambda identity, **kwargs: registered.append(identity),
    }
    try:
        group.allocate(plan, resources=resources, deadline=Deadline.after(2)).result(
            deadline=Deadline.after(2)
        )
        group.start(gate=gate, deadline=Deadline.after(2)).result(
            deadline=Deadline.after(2)
        )
        assert group.ready(deadline=Deadline.after(2))["ready"]
        result = group.stop("finished", deadline=Deadline.after(2))
        assert not result.unknown
        assert group.stop("again", deadline=Deadline.after(2)) == result
        assert calls == ["create_frontend", "kill_frontend"]
        assert all(
            r.release_state == "acquired"
            for r in registry.owned_resources()
            if r.kind == "placement_group"
        )
    finally:
        group.stop("test cleanup", deadline=Deadline.after(2))


@pytest.mark.parametrize("close_failure", [False, True])
def test_external_master_is_only_borrowed_even_when_pool_cleanup_fails(
    tmp_path, close_failure
):
    config = task_config("M0", output_dir=tmp_path / "service")
    config["store"]["master"] = {"mode": "external", "endpoint": "10.123.0.1:12345"}
    p = build_plan(
        config, node_facts(config), input_plan(config), run_id="service-run", now=100
    ).to_dict()
    registry = ResourceRegistry(p["run_id"], p["plan_hash"])
    calls, registrations = [], []

    class Backend:
        def create(self, cls, *, args, kwargs, options, node_id):
            assert (
                cls.__name__ == "FeatureBuffer"
            )  # No master actor/native process may be created.
            assert kwargs["defer_store"] is True
            assert options["num_gpus"] == 0
            calls.append(("create", options["name"]))
            actor = SimpleNamespace(name=options["name"])
            for method in ("identity", "start", "service_status", "close"):
                setattr(
                    actor,
                    method,
                    SimpleNamespace(
                        remote=lambda *a, _method=method, **kw: (_method, a, kw)
                    ),
                )
            return actor

        def identity(self, actor):
            return actor.name

        def get(self, ref, *, timeout):
            assert timeout > 0
            method, _args, _kwargs = ref
            calls.append((method, timeout))
            if method == "identity":
                return message_envelope(
                    p["run_id"],
                    p["plan_hash"],
                    {"component": "fixture"},
                    actor_id="service-run-feature-buffer",
                    node_id="node-a",
                    process={"pid": 100},
                )
            if method == "start":
                assert registrations  # Native construction cannot precede NodeAgent ownership acknowledgement.
                return {"started": True}
            if method == "close" and close_failure:
                raise TimeoutError("native close blocked")
            return {"ready": True, "pool_bytes": 64, "cleanup_complete": True}

        def kill(self, actor):
            calls.append(("kill", actor.name))

        def dead(self, actor, *, timeout):
            return True

    actors = ActorAllocator(p, registry=registry, backend=Backend())
    service = StoreService()
    resources = {
        "actors": actors,
        "config": {},
        "register_process": lambda identity, **kw: registrations.append(identity),
    }
    try:
        service.allocate(p, resources=resources, deadline=Deadline.after(2)).result(
            deadline=Deadline.after(2)
        )
        service.start(gate=None, deadline=Deadline.after(2)).result(
            deadline=Deadline.after(2)
        )
        assert service.ready(deadline=Deadline.after(1))["ready"]
        result = service.stop("finished", deadline=Deadline.after(1))
        assert bool(result.unknown) == close_failure
        assert service.stop("again", deadline=Deadline.after(1)) == result
        assert [c[1] for c in calls if c[0] == "kill"] == ["service-run-feature-buffer"]
        external = next(
            r for r in registry.to_dict()["resources"] if r["owner"] == "external"
        )
        assert external["ray_id"] == "endpoint:10.123.0.1:12345"
        assert external["release_state"] == ("unknown" if close_failure else "released")
        assert not service.status(deadline=Deadline.after(1))["ready"]
        with pytest.raises(RuntimeError, match="stopped"):
            service.start(gate=None, deadline=Deadline.after(1))
    finally:
        service.stop("test cleanup", deadline=Deadline.after(2))

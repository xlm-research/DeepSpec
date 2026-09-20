"""Two node-local launchers must join one frozen native training world."""

import json
from types import SimpleNamespace

import pytest

from deepspec.pipeline.controller import ActorAllocator, ResourceRegistry
from deepspec.pipeline.groups import TrainingGroup
from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import Deadline, message_envelope
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


@pytest.mark.parametrize("failed_node", [None, 1])
def test_m3_launchers_share_endpoint_and_any_node_failure_fails_group(
    tmp_path, failed_node
):
    config = task_config("M3", output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="m3-group", now=100
    ).to_dict()
    path = tmp_path / "pipeline.runtime.json"
    path.write_text(
        json.dumps({"run_id": plan["run_id"], "plan_hash": plan["plan_hash"]})
    )
    groups = {
        f"training-{i}": SimpleNamespace(id=SimpleNamespace(hex=lambda i=i: f"pg-{i}"))
        for i in range(2)
    }
    registered, reports, starts, endpoints, kills = [], [], [], {}, []

    def method(rank, name):
        return SimpleNamespace(remote=lambda *a, **kw: (rank, name, a, kw))

    gate = SimpleNamespace(
        **{
            n: method(-1, n)
            for n in ("observe_launcher", "wait_for_initialization", "native_failed")
        }
    )

    class Backend:
        def create(self, cls, *, args, kwargs, options, node_id):
            rank = kwargs["node_rank"]
            assert (
                cls.__name__ == "Consumer"
                and options["num_gpus"] == 4
                and options["num_cpus"] == 8
            )
            assert (
                options["scheduling_strategy"].placement_group
                is groups[f"training-{rank}"]
            )
            assert node_id == f"node-{'bc'[rank]}"
            return SimpleNamespace(
                rank=rank,
                **{
                    n: method(rank, n)
                    for n in (
                        "allocation",
                        "reserve_rendezvous",
                        "configure_rendezvous",
                        "start",
                        "status",
                        "stop",
                    )
                },
            )

        def identity(self, actor):
            return f"launcher-{actor.rank}"

        def get(self, ref, *, timeout):
            rank, name, args, kwargs = ref
            if name == "allocation":
                return message_envelope(
                    plan["run_id"],
                    plan["plan_hash"],
                    {"component": "launcher"},
                    actor_id=f"launcher-{rank}",
                    node_id=f"node-{'bc'[rank]}",
                    participant=f"training/{rank}",
                    pg_id=f"pg-{rank}",
                    bundle_index=0,
                    process={
                        "pid": 100 + rank,
                        "start_ticks": 200,
                        "parent_pid": 90,
                        "group_id": 100 + rank,
                        "run_id": plan["run_id"],
                    },
                    gpu_uuids=[f"gpu-{rank}-{i}" for i in range(4)],
                )
            if name == "observe_launcher":
                reports.append(args[0])
            if name == "reserve_rendezvous":
                assert rank == 0
                return message_envelope(
                    plan["run_id"],
                    plan["plan_hash"],
                    {"component": "rendezvous"},
                    host="10.123.0.2",
                    port=23456,
                    node_id="node-b",
                    actor_id="launcher-0",
                )
            if name == "configure_rendezvous":
                endpoints[rank] = args[0]
            if name == "start":
                assert len(registered) == len(reports) == 2
                assert endpoints[0] == endpoints[1]
                starts.append(rank)
            if name == "status":
                return {
                    "state": "failed" if rank == failed_node else "finished",
                    "error": "injected rank failure" if rank == failed_node else None,
                }
            return {"ready": True, "cleanup_complete": True, "errors": []}

        def kill(self, actor):
            kills.append(actor.rank)

        def dead(self, actor, *, timeout):
            return True

    actors = ActorAllocator(
        plan,
        registry=ResourceRegistry(plan["run_id"], plan["plan_hash"]),
        backend=Backend(),
    )
    group = TrainingGroup()
    try:
        group.allocate(
            plan,
            resources={
                "actors": actors,
                "gate": gate,
                "placement_groups": groups,
                "config_path": path,
                "register_process": lambda report, **kw: registered.append(report),
            },
            deadline=Deadline.after(5),
        ).result(deadline=Deadline.after(5))
        group.start(gate=gate, deadline=Deadline.after(5)).result(
            deadline=Deadline.after(5)
        )
        assert starts == [0, 1]
        assert json.loads(path.read_text())["consumer_rendezvous"] == {
            "host": "10.123.0.2",
            "port": 23456,
        }
        status = group.status(deadline=Deadline.after(5))
        assert status["state"] == ("failed" if failed_node is not None else "finished")
        assert not group.stop("done", deadline=Deadline.after(5)).unknown
        assert sorted(kills) == [0, 1]
    finally:
        group.executor.shutdown(wait=False)


def test_rendezvous_reservation_is_exclusive_frozen_and_released(monkeypatch):
    import socket
    from deepspec.pipeline.actors import Consumer

    launcher = Consumer.__new__(Consumer)
    launcher.node_rank = 0
    launcher.native_plan = {
        "training_ranks": [{"node_id": "node-b"}],
        "nodes": {"b": {"node_id": "node-b", "ip": "127.0.0.1"}},
    }
    launcher.config = {"consumer_nodes": 2, "consumer_dp": 2, "consumer_world_size": 8}
    launcher._rendezvous_reservation = None
    launcher._work = None
    monkeypatch.setattr(launcher, "identity", lambda: {"node_id": "node-b"})
    monkeypatch.setattr(launcher, "_reply", lambda **kw: kw)
    report = launcher.reserve_rendezvous()
    endpoint = {k: report[k] for k in ("host", "port")}
    try:
        with socket.socket() as contender:
            with pytest.raises(OSError):
                contender.bind((endpoint["host"], endpoint["port"]))
        assert launcher.reserve_rendezvous() == report
        launcher.configure_rendezvous(endpoint)
        with pytest.raises(ValueError, match="frozen"):
            launcher.configure_rendezvous(
                {**endpoint, "port": endpoint["port"] % 65535 + 1}
            )
    finally:
        launcher._close_native(Deadline.after(1))
    with socket.socket() as contender:
        contender.bind((endpoint["host"], endpoint["port"]))

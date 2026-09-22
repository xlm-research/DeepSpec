import sys
import types
from pathlib import Path

import pytest

from deepspec.pipeline.cluster import (
    _inspect_task_nodes_worker,
    native_placement_capabilities,
)
from deepspec.pipeline.planning import Run
from tests.pipeline_topology_fixtures import node_facts, task_config


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("slow_inspection", [False, True])
@pytest.mark.parametrize(
    "sharing,ray_free", [("exclusive", 8), ("shared", 8), ("shared", 2)]
)
def test_inspection_reserves_only_cpu_and_releases_all_actors(
    tmp_path, monkeypatch, failure, slow_inspection, sharing, ray_free
):
    from deepspec.pipeline import cluster

    now, samples = [100.0], []
    monkeypatch.setattr(
        cluster, "time", types.SimpleNamespace(monotonic=lambda: now[0])
    )
    config = task_config(output_dir=tmp_path / "run")
    config["gpu_sharing"] = sharing
    run = Run.create(config["output_dir"])
    facts = node_facts(config)
    actors, killed, requirements = [], [], []

    class Factory:
        def options(self, **kwargs):
            return self

        def remote(self, config, run, token):
            i = len(actors)

            def inspect():
                if failure and i == 1:
                    raise RuntimeError("injected inspection failure")
                node = facts[i].copy()
                if slow_inspection:
                    now[0] += 10
                node["agent_epoch"] = token
                witness = Path(run["output_dir"]) / f"node-{i}"
                witness.write_text(token)
                node["shared_paths"] = {
                    "readable": True,
                    "writable": True,
                    "witness_path": str(witness),
                }
                node["free_gpu_uuids"] = [gpu["uuid"] for gpu in node["gpus"][:2]]
                return node

            def sample(request_id):
                samples.append(i)
                return {
                    "node_id": facts[i]["node_id"],
                    "agent_epoch": token,
                    "request_id": request_id,
                    "sample_seq": 2,
                    "memory": facts[i]["memory"],
                    "observed_at": now[0],
                }

            actor = types.SimpleNamespace(
                ready=types.SimpleNamespace(remote=lambda: True),
                inspect=types.SimpleNamespace(remote=inspect),
                sample_startup_memory=types.SimpleNamespace(remote=sample),
            )
            actors.append(actor)
            return actor

    def remote(**kwargs):
        requirements.append(kwargs)
        return lambda cls: Factory()

    ray = types.ModuleType("ray")
    ray.init = lambda **kwargs: None
    ray.shutdown = lambda: None
    ray.nodes = lambda: [
        {"NodeID": n["node_id"], "NodeManagerAddress": n["ip"], "Alive": True}
        for n in facts
    ]
    ray.remote = remote
    ray.get = lambda values, **kwargs: values
    ray.kill = lambda actor, **kwargs: killed.append(actor)
    state = types.ModuleType("ray._private.state")
    state.available_resources_per_node = lambda: {
        n["node_id"]: {"CPU": 63, "GPU": ray_free} for n in facts
    }
    strategies = types.ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = lambda *args, **kwargs: None
    for name, module in (
        ("ray", ray),
        ("ray._private.state", state),
        ("ray.util.scheduling_strategies", strategies),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            _inspect_task_nodes_worker(config, run.to_dict())
    else:
        result = _inspect_task_nodes_worker(config, run.to_dict())
        assert len(result) == 3
        assert all(n["cpu_available"] == 64 for n in result)
        assert all(
            n["gpu_available"] == min(ray_free, 8 if sharing == "shared" else 2)
            for n in result
        )
        assert all(n["ray_gpu_available"] == ray_free for n in result)
        assert all(len(n["free_gpu_uuids"]) == 2 for n in result)
        assert len(samples) == len(result)
        assert all(now[0] - n["request_sent_at"] < 5 for n in result)
    assert len(killed) == len(actors) == 3
    assert all(r["num_gpus"] == 0 and r["num_cpus"] == 1 for r in requirements)


def test_installed_version_does_not_substitute_for_native_capability(tmp_path):
    for file in (
        "config/parallel.py",
        "v1/engine/core.py",
        "v1/executor/ray_executor_v2.py",
    ):
        path = tmp_path / "vllm/vllm" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("__version__ = '999.0'\n")
    assert not any(native_placement_capabilities(tmp_path).values())


@pytest.mark.parametrize(
    "changed",
    [
        "vllm/vllm/config/parallel.py",
        "deepspec/pipeline/mooncake/buffers.py",
        "deepspec/orchestration/process.py",
    ],
)
def test_native_transport_and_supervisor_changes_invalidate_source_identity(
    monkeypatch, changed
):
    from deepspec.pipeline.cluster import source_hashes
    from deepspec.pipeline.run import ROOT
    from deepspec.pipeline.schema import content_hash

    before = content_hash(source_hashes(ROOT))
    original = Path.read_bytes

    def modified(path):
        data = original(path)
        return (
            data + b"\n# changed implementation\n" if path == ROOT / changed else data
        )

    monkeypatch.setattr(Path, "read_bytes", modified)
    assert content_hash(source_hashes(ROOT)) != before

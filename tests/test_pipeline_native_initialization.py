"""Native backend API and fail-fast readiness regression from the M0 run."""

import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("backend", ["dtensor", "spmd_types"])
@pytest.mark.parametrize("dp", [1, 2])
def test_handshake_uses_pinned_native_backend_mesh_names(monkeypatch, backend, dp):
    import torch.distributed as dist
    from torchtitan.distributed.parallel_dims import ParallelDims
    from deepspec.pipeline.training import native_rank_groups

    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=dp,
        cp=1,
        tp=4,
        pp=1,
        ep=1,
        world_size=4 * dp,
        spmd_backend=backend,
    )

    def mesh(members):
        return SimpleNamespace(
            size=lambda: len(members),
            get_local_rank=lambda: 0,
            get_group=lambda: members,
        )

    dims._single_axis_meshes = {
        "tp": mesh([0, 1, 2, 3]),
        "dp_shard" if backend == "spmd_types" else "fsdp": mesh(
            list(range(0, 4 * dp, 4))
        ),
    }
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: group)
    assert native_rank_groups(dims) == {
        "tp_rank": 0,
        "dp_rank": 0,
        "tp_members": [0, 1, 2, 3],
        "dp_members": list(range(0, 4 * dp, 4)),
    }


@pytest.mark.parametrize("failed_role", ["training", "inference"])
@pytest.mark.parametrize("phase", ["initialize", "ready"])
def test_ready_observes_launcher_failure_before_waiting_for_gate(
    tmp_path, failed_role, phase, monkeypatch
):
    from deepspec.pipeline.operations import NativeRunOperations
    from deepspec.pipeline.runtime import Deadline
    from tests.test_pipeline_execution import frozen_run

    import ray

    monkeypatch.setattr(
        ray, "wait", lambda *a, **kw: pytest.fail("Waited after launcher failure")
    )
    operation = NativeRunOperations(frozen_run(tmp_path))
    for group in (operation.training, operation.inference, operation.store):
        group.executor.shutdown(wait=False)
    started = SimpleNamespace(result=lambda **kw: None)
    operation.store = SimpleNamespace(start=lambda **kw: started)
    for role in ("training", "inference"):
        setattr(
            operation,
            role,
            SimpleNamespace(
                start=lambda **kw: started,
                ready=lambda **kw: pytest.fail(
                    "Blocked on gate despite a dead launcher"
                ),
                status=lambda role=role, **kw: {
                    "state": "failed" if role == failed_role else "running",
                    "error": "native initialization failed"
                    if role == failed_role
                    else None,
                },
            ),
        )
    operation.gate = SimpleNamespace(
        wait_for_initialization=SimpleNamespace(remote=lambda **kw: "pending"),
        wait_for_allocation=SimpleNamespace(remote=lambda **kw: "pending"),
    )
    with pytest.raises(RuntimeError, match="native initialization failed"):
        getattr(operation, phase)(deadline=Deadline.after(5), stop=threading.Event())


def test_trainer_construction_failure_reaches_native_gate(monkeypatch):
    from deepspec.pipeline.trainer import StreamingDSparkTrainer, DSparkTrainer
    from deepspec.pipeline.training import TrainingHandshake

    errors = []
    handshake = SimpleNamespace(failed=errors.append)
    monkeypatch.setattr(TrainingHandshake, "begin", lambda _: handshake)
    original = ValueError("native model initialization failed")

    def fail(self, config):
        raise original

    monkeypatch.setattr(DSparkTrainer, "__init__", fail)
    with pytest.raises(ValueError) as caught:
        StreamingDSparkTrainer(
            SimpleNamespace(dataloader=SimpleNamespace(pipeline_config="fixture"))
        )
    assert caught.value is original and errors == [original]

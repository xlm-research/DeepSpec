"""Bound the extraction queue before any pinned host allocation occurs."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
)

from deepspec.pipeline.connector import MooncakeHiddenStatesConnector


@pytest.mark.parametrize("rank,rejected", [(0, False), (0, True), (3, False)])
def test_connector_validates_writer_before_store_and_reports_readiness_after(
    monkeypatch, rank, rejected
):
    from vllm.distributed import parallel_state

    import deepspec.pipeline.connector as module
    from deepspec.pipeline.runtime import Deadline
    from vllm import distributed

    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._is_tp_rank_zero = rank == 0
    connector.store = None
    connector.producer_rank = 1
    connector.pipeline = {
        "buffer_name": "buffer",
        "namespace": "test",
        "transport": {"async_put_pool_size": 1},
        "store": {},
        "timeouts_seconds": {"initialization": 2},
    }
    connector.run_deadline = Deadline.after(2)
    calls = []

    def callback(event, payload, *, timeout):
        assert 0 < timeout <= 2
        assert payload == {
            "replica": 1,
            "tp_rank": rank,
            "tp_world_size": 4,
            "writer": rank == 0,
            "node_id": "node-a",
            "actor_id": "actor",
            "physical_gpu_ids": ["7"],
        }
        calls.append(event)
        if rejected:
            raise ValueError("identity rejected")

    connector.native_parallel = SimpleNamespace(
        ray_placement_plan={}, ray_placement_event=callback
    )
    monkeypatch.setattr(
        ExampleHiddenStatesConnector, "register_kv_caches", lambda *a: None
    )
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=4)
    )
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_node_id=lambda: "node-a",
            get_actor_id=lambda: "actor",
            get_accelerator_ids=lambda: {"GPU": ["7"]},
        ),
    )
    monkeypatch.setattr(
        module.ray,
        "get_actor",
        lambda *a, **kw: SimpleNamespace(
            event=SimpleNamespace(remote=lambda *a, **kw: None)
        ),
    )
    monkeypatch.setattr(connector, "_get", lambda *a, **kw: None)
    monkeypatch.setattr(
        module, "TensorStore", lambda *a, **kw: calls.append("store") or object()
    )
    if rejected:
        with pytest.raises(ValueError, match="identity rejected"):
            connector.register_kv_caches({})
        assert calls == ["connector_check"] and connector.store is None
    else:
        connector.register_kv_caches({})
        assert calls == [
            "connector_check",
            *(["store"] if rank == 0 else []),
            "connector_initialized",
        ]


def test_writer_credit_blocks_allocation_until_a_write_finishes(monkeypatch):
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._is_tp_rank_zero = True
    connector.pipeline = {"timeout_seconds": 2}
    connector.write_slots = threading.BoundedSemaphore(2)
    connector._req_futures = {}
    allocated = []
    release = threading.Event()
    attempted = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as executor:

        def submit(self, pending):
            allocated.append(pending.req_id)
            future = executor.submit(release.wait, 2)
            self._req_futures[pending.req_id] = future
            future.add_done_callback(
                lambda completed: self._on_write_done(pending.req_id, completed)
            )

        monkeypatch.setattr(ExampleHiddenStatesConnector, "_submit_async_write", submit)
        connector._submit_async_write(SimpleNamespace(req_id="0"))
        connector._submit_async_write(SimpleNamespace(req_id="1"))

        def third():
            attempted.set()
            connector._submit_async_write(SimpleNamespace(req_id="2"))

        thread = threading.Thread(target=third)
        thread.start()
        try:
            assert attempted.wait(1)
            thread.join(0.05)
            assert thread.is_alive() and allocated == ["0", "1"]
        finally:
            release.set()
            thread.join(2)
        assert not thread.is_alive() and allocated == ["0", "1", "2"]
    assert connector.write_slots.acquire(blocking=False)
    assert connector.write_slots.acquire(blocking=False)
    assert not connector.write_slots.acquire(blocking=False)


def test_failed_writer_submission_returns_its_slot(monkeypatch):
    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._is_tp_rank_zero = True
    connector.pipeline = {"timeout_seconds": 1}
    connector.write_slots = threading.BoundedSemaphore(1)

    def fail(*args):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(ExampleHiddenStatesConnector, "_submit_async_write", fail)
    with pytest.raises(RuntimeError, match="allocation failed"):
        connector._submit_async_write(SimpleNamespace(req_id="0"))
    assert connector.write_slots.acquire(blocking=False)
    assert not connector.write_slots.acquire(blocking=False)


@pytest.mark.parametrize("write_failed", [False, True])
def test_publish_requires_completed_store_write(monkeypatch, write_failed):
    import deepspec.pipeline.connector as module

    connector = MooncakeHiddenStatesConnector.__new__(MooncakeHiddenStatesConnector)
    connector._is_tp_rank_zero = True
    connector.producer_rank, connector.node_id = 0, "writer-node"
    sample = {
        "position": 0,
        "sample_id": "s",
        "input_identity": "input",
        "length": 1,
        "input_path": "/synthetic/batch.pt",
    }
    connector.pipeline = {
        "producer_dp": 1,
        "samples": [sample],
        "run_id": "test",
        "teacher": {"hidden_size": 8, "target_layer_ids": [0]},
        "transport": {"chunk_bytes": 1024},
        "timeout_seconds": 2,
    }
    published, failures = [], []
    connector.buffer = SimpleNamespace(
        begin_write=SimpleNamespace(remote=lambda *args: None),
        publish=SimpleNamespace(remote=lambda *args: published.append(args)),
        fail=SimpleNamespace(remote=lambda error: failures.append(error)),
    )
    started, release = threading.Event(), threading.Event()

    def wait(**kwargs):
        started.set()
        assert release.wait(2)
        if write_failed:
            raise RuntimeError("write failed")
        return {"nbytes": 16}

    connector.store = SimpleNamespace(
        put_async=lambda *a, **kw: SimpleNamespace(wait=wait)
    )
    monkeypatch.setattr(module.ray, "get", lambda result, **kwargs: result)
    monkeypatch.setattr(module.torch, "load", lambda *a, **kw: {})
    monkeypatch.setattr(module, "convert_hidden_states", lambda *a, **kw: {})
    monkeypatch.setattr(module, "describe_tensors", lambda *a, **kw: {})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            connector._write_tensors,
            {},
            SimpleNamespace(synchronize=lambda: None),
            "0",
            None,
        )
        try:
            assert started.wait(1)
            assert published == []
        finally:
            release.set()
        if write_failed:
            with pytest.raises(RuntimeError, match="write failed"):
                future.result(timeout=2)
            assert failures and published == []
        else:
            future.result(timeout=2)
            assert len(published) == 1 and not failures

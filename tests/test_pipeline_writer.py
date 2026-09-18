"""Bound the extraction queue before any pinned host allocation occurs."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
)

from deepspec.pipeline.connector import MooncakeHiddenStatesConnector


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

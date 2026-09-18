"""CPU-only checks for the bounded Mooncake transport primitives."""

import socket
import threading

import pytest

from deepspec.pipeline.mooncake import DeleteManager, HostBufferPool
from deepspec.pipeline.runtime import endpoint_parts, wait_for_endpoint
from deepspec.pipeline.schema import normalize_pipeline_config


def test_host_pool_replaces_small_free_buffer_and_keeps_registration_balanced():
    registered, unregistered = [], []
    pool = HostBufferPool(
        max_buffers=1,
        pin_memory=False,
        register=lambda pointer, size: registered.append((pointer, size)),
        unregister=lambda pointer: unregistered.append(pointer),
    )
    small = pool.acquire(32)
    pool.release(small)
    large = pool.acquire(1024)
    assert large.size >= 1024 and len(registered) == 2
    pool.release(large)
    pool.close()
    assert len(unregistered) == 2


def test_host_pool_blocks_when_all_slots_are_owned():
    pool = HostBufferPool(max_buffers=1, pin_memory=False)
    owned = pool.acquire(32)
    completed = threading.Event()

    def acquire():
        pool.acquire(32, timeout=1)
        completed.set()

    thread = threading.Thread(target=acquire)
    thread.start()
    assert not completed.wait(0.05)
    pool.release(owned)
    thread.join(1)
    assert completed.is_set()
    # The second caller owns the slot until it explicitly releases it.
    pool.release(pool.buffers[0])
    pool.close()


def test_delete_manager_retries_and_reports_terminal_failure():
    calls = []

    def flaky(_fields):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("metadata unavailable")

    manager = DeleteManager(flaky, max_attempts=3, retry_delay=0)
    manager.delete({"key": "value"})
    assert len(calls) == 3 and manager.successes == 1
    manager.close()

    failed = DeleteManager(lambda _: (_ for _ in ()).throw(RuntimeError("no")), max_attempts=2, retry_delay=0)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        failed.delete({})
    failed.close()


def test_schema_upgrade_preserves_v1_keys_and_adds_transport_defaults():
    config = {"verify_transfers": True, "store": {}}
    normalize_pipeline_config(config)
    assert config["schema_version"] == 2
    assert config["transport"]["chunk_bytes"] == 8 * 1024**2
    assert config["store"]["verify_mode"] == "full"


def test_wait_for_endpoint_reports_a_live_tcp_listener():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    thread = threading.Thread(target=lambda: listener.accept()[0].close())
    thread.start()
    assert endpoint_parts(endpoint)[0] == "127.0.0.1"
    wait_for_endpoint(endpoint, timeout=1, poll_interval=0.01)
    thread.join(1)
    listener.close()

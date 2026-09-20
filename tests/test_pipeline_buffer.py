"""Keep admission bounded and prevent premature or duplicate consumption."""

import asyncio
from builtins import ExceptionGroup
from types import SimpleNamespace

import pytest

from deepspec.pipeline.buffer import BufferLedger, FeatureBuffer


def samples(count=8):
    return [
        {
            "position": i,
            "sample_id": f"sample-{i}",
            "input_identity": str(i),
            "length": 10,
            "nbytes": 100,
        }
        for i in range(count)
    ]


def descriptor(sample):
    return {**sample, "fields": {"feature": {"nbytes": 100}}}


def test_complete_update_capacity_is_required_to_avoid_accumulation_deadlock():
    with pytest.raises(ValueError, match="complete optimizer update"):
        BufferLedger(
            samples(), capacity=800, window=3, readers=[0, 1], samples_per_update=4
        )
    with pytest.raises(ValueError, match="complete one optimizer update"):
        BufferLedger(
            samples(), capacity=399, window=4, readers=[0, 1], samples_per_update=4
        )


def test_credit_returns_only_after_every_reader_and_successful_delete():
    plan = samples()
    ledger = BufferLedger(
        plan, capacity=400, window=4, readers=[0, 1], samples_per_update=4
    )
    for i in range(4):
        assert ledger.reserve(i)
    assert not ledger.reserve(4)
    assert ledger.claim(0, 0) is None
    ledger.start_write(0)
    ledger.ready(0, descriptor(plan[0]))
    for reader in (0, 1):
        assert ledger.claim(0, reader)["position"] == 0
    assert ledger.acknowledge(0, 0) is None
    with pytest.raises(ValueError, match="all readers"):
        ledger.deleted(0)
    assert ledger.acknowledge(0, 1) is not None
    assert ledger.reserved_bytes == 400
    ledger.deleted(0)
    assert ledger.reserved_bytes == 300
    assert not ledger.reserve(4)  # Hysteresis keeps admission paused above 60%.
    ledger.start_write(1)
    ledger.ready(1, descriptor(plan[1]))
    for reader in (0, 1):
        ledger.claim(1, reader)
        ledger.acknowledge(1, reader)
    ledger.deleted(1)
    assert ledger.reserve(4)


def test_identity_duplicates_and_ack_without_read_are_rejected():
    plan = samples()
    ledger = BufferLedger(
        plan, capacity=800, window=4, readers=[0, 1], samples_per_update=4
    )
    ledger.reserve(0)
    ledger.start_write(0)
    with pytest.raises(ValueError, match="input_identity"):
        ledger.ready(0, {**descriptor(plan[0]), "input_identity": "wrong"})
    ledger.ready(0, descriptor(plan[0]))
    with pytest.raises(ValueError, match="unclaimed"):
        ledger.acknowledge(0, 0)
    ledger.claim(0, 0)
    with pytest.raises(ValueError, match="Duplicate feature claim"):
        ledger.claim(0, 0)
    ledger.acknowledge(0, 0)
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.acknowledge(0, 0)


def test_backpressure_can_resume_inside_an_accumulation_group():
    plan = samples()
    ledger = BufferLedger(
        plan, capacity=400, window=4, readers=[0], samples_per_update=4
    )
    for position in range(4):
        ledger.reserve(position)
        ledger.start_write(position)
        ledger.ready(position, descriptor(plan[position]))
        ledger.claim(position, 0)
    assert not ledger.reserve(4)
    for position in (0, 1):
        ledger.acknowledge(position, 0)
        ledger.deleted(position)
    assert ledger.reserve(4)
    assert ledger.reserve(5)
    assert not ledger.reserve(6)
    # Once the prior update releases its last samples, a 75%-full pool
    # must still admit the final sample required by the next update.
    for position in (2, 3):
        ledger.acknowledge(position, 0)
        ledger.deleted(position)
    assert ledger.reserve(6)
    assert ledger.reserve(7)


def test_dp_samples_release_after_their_tp_group_and_reject_other_groups():
    plan = samples()
    groups = [range(4) if i % 2 == 0 else range(4, 8) for i in range(8)]
    ledger = BufferLedger(
        plan,
        capacity=400,
        window=4,
        readers=range(8),
        samples_per_update=4,
        readers_by_position=groups,
    )
    for position in range(4):
        assert ledger.reserve(position)
        ledger.start_write(position)
        ledger.ready(position, descriptor(plan[position]))
    for position in (1, 0, 3, 2):
        group = list(groups[position])
        with pytest.raises(ValueError, match="Unexpected feature reader"):
            ledger.claim(position, (group[0] + 4) % 8)
        for reader in group:
            ledger.claim(position, reader)
            fields = ledger.acknowledge(position, reader)
            assert (fields is not None) == (reader == group[-1])
        ledger.deleted(position)
    assert ledger.reserved_bytes == 0 and ledger.released == 4


def test_invalid_per_sample_reader_sets_are_rejected():
    for groups in ([[]] * 8, [[8]] * 8, [[0]] * 7):
        with pytest.raises(ValueError, match="valid group"):
            BufferLedger(
                samples(),
                capacity=800,
                window=4,
                readers=range(8),
                samples_per_update=4,
                readers_by_position=groups,
            )


def test_two_writers_publish_out_of_order_with_unique_ownership():
    plan = samples(4)
    ledger = BufferLedger(
        plan, capacity=400, window=4, readers=[0], samples_per_update=4, producer_dp=2
    )
    for position in range(4):
        assert ledger.reserve(position)
        with pytest.raises(ValueError, match="Unexpected feature producer"):
            ledger.start_write(position, 1 - position % 2)
        ledger.start_write(position, position % 2)
        assert ledger.claim(position, 0) is None
        with pytest.raises(ValueError, match="Duplicate"):
            ledger.start_write(position, position % 2)
    for position in (1, 3, 0, 2):
        with pytest.raises(ValueError, match="Unexpected feature producer"):
            ledger.ready(position, descriptor(plan[position]), 1 - position % 2)
        ledger.ready(position, descriptor(plan[position]), position % 2)
        with pytest.raises(ValueError, match="Duplicate"):
            ledger.ready(position, descriptor(plan[position]), position % 2)
        ledger.claim(position, 0)
        ledger.acknowledge(position, 0)
        ledger.deleted(position)
    assert ledger.released == 4 and ledger.reserved_bytes == 0


def test_resident_peak_excludes_reservations_and_waits_for_deletion():
    plan = samples(4)
    ledger = BufferLedger(
        plan, capacity=400, window=4, readers=[0], samples_per_update=4
    )
    for position in range(4):
        assert ledger.reserve(position)
    assert ledger.peak_bytes == 400 and ledger.peak_resident_bytes == 0
    ledger.start_write(0)
    ledger.ready(0, descriptor(plan[0]))
    ledger.claim(0, 0)
    ledger.acknowledge(0, 0)
    assert ledger.resident_bytes == ledger.peak_resident_bytes == 100
    ledger.deleted(0)
    assert ledger.resident_bytes == 0 and ledger.peak_resident_bytes == 100


def test_partial_batch_returns_before_waiting_and_retention_drains_at_peak(
    async_buffer,
):
    config, buffer, _ = async_buffer
    config["retain_until_bytes"] = 400
    buffer.retention_released = False
    deleted = []
    buffer.store.remove = lambda fields: deleted.append(fields)

    async def run():
        # A batch larger than available capacity must return a runnable prefix.
        assert await asyncio.wait_for(buffer.reserve_batch(0, 20), 0.5) == [0, 1, 2, 3]
        for position in range(3):
            await buffer.begin_write(position, position % 2)
            await buffer.publish(
                position, descriptor(config["samples"][position]), position % 2
            )
            await buffer.claim(position, 0)
            await buffer.acknowledge(position, 0)
        assert not deleted and buffer.ledger.resident_bytes == 300
        await buffer.begin_write(3, 1)
        await buffer.publish(3, descriptor(config["samples"][3]), 1)
        assert buffer.retention_released and len(deleted) == 3
        assert buffer.ledger.peak_resident_bytes == 400
        assert buffer.ledger.resident_bytes == 100
        await buffer.claim(3, 0)
        await buffer.acknowledge(3, 0)
        assert buffer.ledger.resident_bytes == 0
        assert await asyncio.wait_for(buffer.reserve_batch(4, 20), 0.5) == [4, 5, 6, 7]

    asyncio.run(run())


def test_unreached_retention_target_fails_instead_of_claiming_success(async_buffer):
    config, buffer, _ = async_buffer
    buffer.retention_released = False
    buffer.ledger.next_position = len(config["samples"])
    with pytest.raises(RuntimeError, match="resident peak"):
        asyncio.run(buffer.finish_production())


def test_deferred_pool_initialization_keeps_stop_responsive(async_buffer, monkeypatch):
    import threading

    config, _, _ = async_buffer
    entered, release = threading.Event(), threading.Event()
    closed = []

    def slow_store(*args, **kwargs):
        entered.set()
        release.wait(2)
        return SimpleNamespace(
            endpoint="owned-pool", close=lambda **kw: closed.append(True)
        )

    monkeypatch.setattr("deepspec.pipeline.store.TensorStore", slow_store)

    async def scenario():
        buffer = FeatureBuffer(config, defer_store=True)
        try:
            assert not entered.is_set()
            assert (await buffer.start())["started"]
            await asyncio.to_thread(entered.wait, 1)
            assert not buffer.ready()
            with pytest.raises(TimeoutError):
                await buffer.close(timeout=0.01)
            release.set()
            await buffer.close(timeout=1)
            assert not buffer.ready()
            assert closed == [True]
            with pytest.raises(RuntimeError, match="closed"):
                await buffer.start()
        finally:
            release.set()
            await buffer.close(timeout=2)

    asyncio.run(scenario())


@pytest.fixture
def async_buffer(tmp_path, monkeypatch):
    from deepspec.pipeline import store

    monkeypatch.setattr(
        store, "TensorStore", lambda *args, **kwargs: SimpleNamespace(endpoint="test")
    )
    config = {
        "samples": samples(),
        "producer_dp": 2,
        "capacity_bytes": 400,
        "window": 4,
        "consumer_world_size": 1,
        "samples_per_update": 4,
        "store": {},
        "pool_bytes": 400,
        "feature_memory_budget": 400,
        "memory_reserve_bytes": 0,
        "scratch_bound_bytes": 0,
        "timeout_seconds": 2,
        "events_path": str(tmp_path / "events.jsonl"),
    }
    buffer = FeatureBuffer(config)
    proxy = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(buffer, name))
            for name in (
                "reserve",
                "reserve_batch",
                "finish_production",
                "wait_for_failure",
                "fail",
            )
        }
    )
    yield config, buffer, proxy
    buffer.delete_manager.close(timeout=2)
    buffer.events.close()


def test_native_pool_start_does_not_require_legacy_single_node_budget(async_buffer):
    config, _, _ = async_buffer
    config = {k: v for k, v in config.items() if k != "feature_memory_budget"}
    buffer = FeatureBuffer(config, defer_store=True)
    try:
        buffer._open_store()
        assert buffer.ready()
    finally:
        buffer.delete_manager.close(timeout=2)
        buffer.events.close()


def test_pool_close_continues_after_initialization_failed_with_live_store(async_buffer):
    config, _, _ = async_buffer
    closed = []
    buffer = FeatureBuffer(config, defer_store=True)

    def partial_start():
        buffer.store = SimpleNamespace(close=lambda **kw: closed.append(True))
        raise RuntimeError("failure after Store allocation")

    buffer._open_store = partial_start

    async def scenario():
        await buffer.start()
        with pytest.raises(RuntimeError, match="after Store"):
            await buffer._initialization
        result = await buffer.close(timeout=2)
        assert result["cleanup_complete"] and closed == [True]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_source", ["request", "writer"])
def test_async_production_cancels_requests_and_blocked_admission(
    async_buffer, failure_source
):
    from deepspec.pipeline.actors import run_async_production

    config, buffer, proxy = async_buffer

    async def scenario():
        waiting = asyncio.Event()
        cancelled = []
        active = [0, 0]

        async def generate(sample, rank):
            position = sample["position"]
            assert rank == position % 2
            active[rank] += 1
            assert active[rank] == 1
            try:
                if position < 2:
                    return
                if position == 3:
                    waiting.set()
                if position == 2 and failure_source == "request":
                    await waiting.wait()
                    raise RuntimeError("injected request failure")
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.append(position)
                    raise
            finally:
                active[rank] -= 1

        async def fail_writer():
            await waiting.wait()
            await buffer.fail("injected writer failure")

        writer = (
            asyncio.create_task(fail_writer()) if failure_source == "writer" else None
        )
        with pytest.raises(ExceptionGroup, match="TaskGroup"):
            await asyncio.wait_for(run_async_production(config, proxy, generate), 3)
        if writer is not None:
            await asyncio.wait_for(writer, 3)
        assert 3 in cancelled and active == [0, 0]
        assert buffer.error and not buffer.producer_finished
        assert buffer.ledger.next_position == 4

    asyncio.run(scenario())


def test_production_finish_requires_entire_plan_all_writes_and_no_failure(async_buffer):
    _, buffer, _ = async_buffer

    async def scenario():
        with pytest.raises(ValueError, match="full plan"):
            await buffer.finish_production()
        # Finish must still reject a failed stream even if no writes remain.
        buffer.ledger.next_position = len(buffer.ledger.samples)
        await buffer.fail("writer failed after publication")
        with pytest.raises(RuntimeError, match="writer failed"):
            await buffer.finish_production()
        assert not buffer.producer_finished

    asyncio.run(scenario())


def test_batch_rechecks_memory_at_every_update_boundary(async_buffer, monkeypatch):
    config, buffer, _ = async_buffer
    buffer.ledger.capacity = 800
    buffer.ledger.window = 8
    config["scratch_bound_bytes"] = 100
    snapshots = iter([1000, 0, 1000])
    monkeypatch.setattr(
        "deepspec.pipeline.memory.node_memory",
        lambda: {"headroom_bytes": next(snapshots)},
    )

    async def scenario():
        assert await buffer.reserve_batch(0, 8) == [0, 1, 2, 3]
        assert buffer.ledger.reserved_bytes == 400
        assert await buffer.reserve_batch(4, 4) == [4, 5, 6, 7]

    asyncio.run(scenario())


def test_released_feature_retains_audit_and_invalid_shape_never_publishes():
    plan = samples(4)
    plan[0]["fields"] = {"feature": {"shape": [50], "dtype": "bfloat16"}}
    ledger = BufferLedger(
        plan, capacity=400, window=4, readers=[0], samples_per_update=4
    )
    ledger.reserve(0)
    ledger.start_write(0)
    bad = {
        **descriptor(plan[0]),
        "fields": {"feature": {"nbytes": 100, "shape": [25], "dtype": "float32"}},
    }
    with pytest.raises(ValueError, match="shape|dtype"):
        ledger.ready(0, bad)
    good = {
        **descriptor(plan[0]),
        "fields": {"feature": {"nbytes": 100, "shape": [50], "dtype": "bfloat16"}},
    }
    ledger.ready(0, good)
    ledger.claim(0, 0)
    ledger.acknowledge(0, 0)
    ledger.deleted(0)
    assert ledger.history[0]["state"] == "released"
    assert ledger.history[0]["acked"] == {0}


def test_reader_copy_remains_accounted_after_source_deletion(async_buffer):
    config, buffer, _ = async_buffer
    buffer.store.remove = lambda fields: None

    async def scenario():
        await buffer.reserve(0)
        await buffer.begin_write(0)
        await buffer.publish(0, descriptor(config["samples"][0]))
        await buffer.claim(0, 0)
        await buffer.acknowledge(0, 0)
        assert 0 not in buffer.ledger.records
        assert buffer.reader_copies[0, 0] == {
            "state": "materialized_and_acked",
            "nbytes": 100,
        }
        await buffer.reader_copy_state(0, 0, "active")
        await buffer.reader_copy_state(0, 0, "retired")
        assert buffer.reader_copies[0, 0]["state"] == "retired"
        with pytest.raises(ValueError, match="lifetime"):
            await buffer.reader_copy_state(0, 0, "active")

    asyncio.run(scenario())


def test_failed_deletion_keeps_credit_and_wakes_waiting_readers(async_buffer):
    import threading

    config, buffer, _ = async_buffer
    buffer.delete_manager.max_attempts = 2
    buffer.delete_manager.retry_delay = 0
    attempts = []
    deleting, release = threading.Event(), threading.Event()

    def remove(fields):
        attempts.append(fields)
        deleting.set()
        assert release.wait(2)
        raise RuntimeError("injected delete failure")

    buffer.store.remove = remove

    async def scenario():
        await buffer.reserve(0)
        await buffer.begin_write(0)
        await buffer.publish(0, descriptor(config["samples"][0]))
        await buffer.claim(0, 0)
        waiter = asyncio.create_task(buffer.claim(1, 0))
        ack = asyncio.create_task(buffer.acknowledge(0, 0))
        try:
            assert await asyncio.to_thread(deleting.wait, 1)
            assert buffer.ledger.reserved_bytes == 100
            assert buffer.ledger.records[0]["state"] == "deleting"
            assert buffer.reader_copies[0, 0]["state"] == "materialized_and_acked"
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="2 attempts"):
            await asyncio.wait_for(ack, 2)
        with pytest.raises(RuntimeError, match="deletion"):
            await asyncio.wait_for(waiter, 2)
        assert len(attempts) == 2
        assert buffer.ledger.reserved_bytes == 100 and buffer.ledger.released == 0

    asyncio.run(scenario())


def test_new_plan_ack_requires_independent_verified_full_copy(async_buffer):
    config, buffer, _ = async_buffer
    config["plan_hash"] = "frozen"
    buffer.store.remove = lambda fields: None

    async def scenario():
        await buffer.reserve(0)
        await buffer.begin_write(0)
        await buffer.publish(0, descriptor(config["samples"][0]))
        await buffer.claim(0, 0)
        for verified, nbytes in ((False, 100), (True, 99), (None, None)):
            with pytest.raises(ValueError, match="verified independent copy"):
                await buffer.acknowledge(0, 0, verified=verified, nbytes=nbytes)
            assert not buffer.ledger.records[0]["acked"]
        await buffer.acknowledge(0, 0, verified=True, nbytes=100)
        assert buffer.ledger.released == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["normal", "backpressure", "failure"])
def test_structured_buffer_evidence_covers_reads_release_and_waits(
    async_buffer, tmp_path, outcome
):
    import json
    from deepspec.pipeline.runtime import EventWriter

    config, buffer, _ = async_buffer
    config.update(run_id="evidence", plan_hash="frozen")
    path = tmp_path / "structured.jsonl"
    buffer.structured_events = EventWriter(
        path,
        run_id="evidence",
        plan_hash="frozen",
        sender_identity={"component": "feature_buffer"},
    )
    buffer.store.remove = lambda fields: None

    async def scenario():
        await buffer.event("inference_start", position=0)
        await buffer.reserve_batch(0, 4)
        waiter = None
        if outcome == "backpressure":
            waiter = asyncio.create_task(buffer.reserve_batch(4, 1))
            await asyncio.sleep(0.03)
        if outcome == "failure":
            await buffer.fail("injected write failure")
            snapshot = await buffer.source_snapshot()
            assert snapshot["reserved_bytes"] == 400 and all(
                not r["delete_confirmed"] for r in snapshot["records"]
            )
            return
        for position in range(4):
            await buffer.begin_write(position, position % 2)
            await buffer.publish(
                position,
                descriptor(config["samples"][position]),
                position % 2,
                {"seconds": 0.01},
            )
            await buffer.claim(position, 0)
            await buffer.acknowledge(
                position, 0, verified=True, nbytes=100, duration_seconds=0.02
            )
        if waiter is not None:
            assert await asyncio.wait_for(waiter, 1) == [4]
        snapshot = await buffer.source_snapshot()
        released = [r for r in snapshot["records"] if r["state"] == "released"]
        assert len(released) == 4 and all(
            r["delete_confirmed"] and r["acked"] == [0] for r in released
        )

    try:
        asyncio.run(scenario())
    finally:
        buffer.structured_events.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if outcome != "failure":
        assert sum(e["event"] == "feature_produced" for e in events) == 4
        assert (
            sum(
                e["event"] == "feature_read" and e["basis"] == "verified"
                for e in events
            )
            == 4
        )
    if outcome == "backpressure":
        assert any(
            e["event"] == "wait" and e["data"]["duration_seconds"] > 0 for e in events
        )


def test_node_budget_sampling_does_not_block_failure_notification(async_buffer):
    from deepspec.pipeline.memory import GIB, BudgetAdmission

    _config, buffer, _ = async_buffer
    buffer.runtime_budget = BudgetAdmission(
        {"n": {"feature_bound": GIB}},
        run_id="r",
        plan_hash="h",
        freshness=5,
        transfer_timeout=10,
        run_deadline=1e30,
    )

    async def scenario():
        sampling, release = asyncio.Event(), asyncio.Event()

        async def sample(request):
            sampling.set()
            await release.wait()
            return {}

        buffer.node_monitors = [
            SimpleNamespace(sample_budget=SimpleNamespace(remote=sample))
        ]
        task = asyncio.create_task(buffer.reserve_batch(0, 4))
        await sampling.wait()
        await asyncio.wait_for(buffer.fail("cancel during node sampling"), 0.5)
        release.set()
        with pytest.raises(RuntimeError, match="cancel during node sampling"):
            await task
        assert buffer.ledger.reserved_bytes == 0

    asyncio.run(scenario())


def test_fresh_node_budgets_are_checked_for_each_group_in_a_batch(async_buffer):
    import time

    from deepspec.pipeline.memory import GIB, BudgetAdmission

    _config, buffer, _ = async_buffer
    buffer.ledger.capacity = 800
    buffer.ledger.window = 8
    budget = {
        "feature_bound": 200 * GIB,
        "startup_budget": 236 * GIB,
        "static_cap": 400 * GIB,
    }
    buffer.runtime_budget = BudgetAdmission(
        {"n": budget},
        run_id="r",
        plan_hash="h",
        freshness=5,
        transfer_timeout=10,
        run_deadline=time.monotonic() + 100,
    )
    requests = []

    async def sample(request):
        requests.append(request)
        return {
            **request,
            "node_id": "n",
            "boot_id": "b",
            "agent_epoch": "e",
            "sample_seq": len(requests),
            "memory": {
                "physical_bytes": 500 * GIB,
                "limit_bytes": 500 * GIB,
                "headroom_bytes": (300 if request["update_index"] == 0 else 100) * GIB,
            },
        }

    buffer.node_monitors = [
        SimpleNamespace(sample_budget=SimpleNamespace(remote=sample))
    ]
    assert asyncio.run(buffer.reserve_batch(0, 8)) == [0, 1, 2, 3]
    assert [request["update_index"] for request in requests] == [0, 1]
    assert buffer.runtime_budget.is_admitted(0)
    assert not buffer.runtime_budget.is_admitted(1)


@pytest.mark.parametrize("dp", [1, 2])
def test_async_production_preserves_batch_parallelism_and_frozen_positions(dp):
    from deepspec.pipeline.actors import run_async_production

    async def scenario():
        count, batch = 12, 3
        active, peak = [0] * dp, [0] * dp
        full = [asyncio.Event() for _ in range(dp)]
        requests, reservations, finished = [], [], []

        async def reserve_batch(position, limit):
            reservations.append((position, limit))
            return list(range(position, min(position + limit, count)))

        async def finish():
            finished.append(True)

        async def failure():
            await asyncio.Event().wait()

        async def fail(reason):
            raise AssertionError(reason)

        buffer = SimpleNamespace(
            **{
                name: SimpleNamespace(remote=method)
                for name, method in (
                    ("reserve_batch", reserve_batch),
                    ("finish_production", finish),
                    ("wait_for_failure", failure),
                    ("fail", fail),
                )
            }
        )

        async def generate(sample, rank):
            assert rank == sample["position"] % dp
            requests.append(sample["position"])
            active[rank] += 1
            peak[rank] = max(peak[rank], active[rank])
            if active[rank] == batch:
                full[rank].set()
            await full[rank].wait()
            await asyncio.sleep((count - sample["position"]) % 3 * 0.001)
            active[rank] -= 1

        config = {
            "samples": [{"position": p} for p in range(count)],
            "producer_dp": dp,
            "producer_batch_size": batch,
            "timeout_seconds": 2,
        }
        await asyncio.wait_for(run_async_production(config, buffer, generate), 2)
        assert peak == [batch] * dp and active == [0] * dp
        assert sorted(requests) == list(range(count)) and finished == [True]
        assert reservations == [(p, batch * dp) for p in range(0, count, batch * dp)]

    asyncio.run(scenario())

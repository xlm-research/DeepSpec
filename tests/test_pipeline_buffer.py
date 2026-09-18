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
            for name in ("reserve", "finish_production", "wait_for_failure", "fail")
        }
    )
    yield config, buffer, proxy
    buffer.events.close()


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
            await writer
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

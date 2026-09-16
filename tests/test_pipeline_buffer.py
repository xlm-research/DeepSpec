"""Keep admission bounded and prevent premature or duplicate consumption."""

import pytest

from deepspec.pipeline.buffer import BufferLedger


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

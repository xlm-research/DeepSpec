import pytest

from deepspec.pipeline.memory import (
    GIB,
    BudgetAdmission,
    check_runtime_budget,
    node_feature_budget,
)


def approved():
    return {
        "feature_bound": 200 * GIB,
        "startup_budget": 236 * GIB,
        "static_cap": 400 * GIB,
        "headroom_reserve_bytes": 64 * GIB,
    }


def memory(headroom=236):
    return {
        "physical_bytes": 500 * GIB,
        "limit_bytes": 500 * GIB,
        "headroom_bytes": headroom * GIB,
    }


def charge(**overrides):
    return {
        "allocation_id": "pinned-pool",
        "run_id": "r",
        "nbytes": 64 * GIB,
        "resident_locked": True,
        "charged_domains": ["physical", "cgroup"],
        "valid_through_update": 2,
        **overrides,
    }


def test_resident_pool_is_not_double_charged_but_declared_pool_is_never_credit():
    assert not check_runtime_budget(
        approved(), memory(300), run_id="r", update_index=1
    )["pressure"]
    result = check_runtime_budget(
        approved(), memory(), run_id="r", update_index=1, charges=[charge()]
    )
    assert result["remaining_bound"] == 136 * GIB and not result["pressure"]
    assert check_runtime_budget(
        approved(), memory(196), run_id="r", update_index=1, charges=[charge()]
    )["pressure"]
    for invalid in (
        charge(resident_locked=False),
        charge(run_id="other"),
        charge(valid_through_update=0),
        charge(charged_domains=["physical"]),
    ):
        result = check_runtime_budget(
            approved(), memory(), run_id="r", update_index=1, charges=[invalid]
        )
        assert result["retained_charge_lower_bound"] == 0 and result["pressure"]
    assert check_runtime_budget(
        approved(), memory(), run_id="r", update_index=1, charges=[]
    )["pressure"]


def test_charge_overlap_is_rejected_and_static_cap_still_applies():
    with pytest.raises(ValueError, match="Duplicate"):
        check_runtime_budget(
            approved(),
            memory(),
            run_id="r",
            update_index=1,
            charges=[charge(), charge()],
        )
    snapshot = memory(300)
    snapshot["limit_bytes"] = 100 * GIB
    assert check_runtime_budget(
        approved(), snapshot, run_id="r", update_index=1, charges=[charge()]
    )["pressure"]


def test_node_bound_includes_reader_prefetch_and_actual_client_buffers():
    result = node_feature_budget(
        pool_bytes=64 * GIB,
        max_sample_bytes=GIB,
        writers=2,
        writer_inflight=2,
        readers=4,
        gas=2,
        prefetch_depth=2,
        prefetch_bytes=3 * GIB,
        snapshot=memory(300),
        client_buffer_bytes=123,
    )
    assert result["writer_bound"] == 12 * GIB
    assert result["transport_bound"] == 4 * GIB
    assert result["reader_bound"] == 28 * GIB
    assert result["feature_bound"] == 109 * GIB + 123


@pytest.mark.parametrize(
    "layout,writers,readers",
    [
        ("M0", [1], [4]),
        ("M2-DP2", [2, 0, 0], [0, 0, 8]),
        ("M3", [1, 0, 0], [0, 4, 4]),
    ],
)
def test_node_role_bounds_follow_the_frozen_layout(layout, writers, readers):
    from deepspec.pipeline.planning import build_plan
    from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config

    config = task_config(layout)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="fixture", now=100
    ).to_dict()
    bounds = [plan["node_budgets"][node] for node in sorted(plan["node_budgets"])]
    assert [b["writers"] for b in bounds] == writers
    assert [b["readers"] for b in bounds] == readers
    assert sum(b["pool_bytes"] > 0 for b in bounds) == 1
    for bound in bounds:
        assert bound["retained_charge_lower_bound"] == 0
        assert bound["remaining_bound"] == bound["feature_bound"]


def response(request, *, seq=1, epoch="e", headroom=300):
    return {
        **request,
        "node_id": "n",
        "boot_id": "boot",
        "agent_epoch": epoch,
        "sample_seq": seq,
        "memory": memory(headroom),
        "retained_charges": [],
    }


def test_freshness_is_measured_on_controller_clock_and_exact_boundary_is_stale():
    clock = [0.0]
    gate = BudgetAdmission(
        {"n": approved()},
        run_id="r",
        plan_hash="h",
        freshness=5,
        transfer_timeout=10,
        run_deadline=100,
        clock=lambda: clock[0],
    )
    request = gate.request(0)
    reply = response(request)
    reply["sampled_at"] = -999999  # Foreign monotonic clocks are irrelevant.
    clock[0] = 4.99
    assert gate.check(0, [reply])
    clock[0] = 5
    assert not gate.check(0, [reply])
    retry = gate.request(0)
    assert retry["request_id"] != request["request_id"]
    clock[0] = 9.99
    assert gate.check(0, [response(retry, seq=2)])
    clock[0] = 10
    with pytest.raises(TimeoutError):
        gate.request(0)


def test_epoch_change_missing_node_late_reply_and_pressure_block_new_group():
    gate = BudgetAdmission(
        {"n": approved()},
        run_id="r",
        plan_hash="h",
        freshness=5,
        transfer_timeout=10,
        run_deadline=100,
        clock=lambda: 0,
    )
    request = gate.request(0)
    assert not gate.check(0, [])
    assert gate.check(0, [response(request)])
    gate.commit(0)
    request2 = gate.request(1)
    assert not gate.check(1, [response(request)])
    assert not gate.check(1, [response(request2, seq=2, epoch="new")])
    assert not gate.check(1, [response(request2, seq=2, headroom=100)])
    assert gate.is_admitted(0)


def test_commit_requires_its_own_fresh_successful_request():
    clock = [0.0]
    gate = BudgetAdmission(
        {"n": approved()},
        run_id="r",
        plan_hash="h",
        freshness=5,
        transfer_timeout=10,
        run_deadline=100,
        clock=lambda: clock[0],
    )
    first = gate.request(0)
    with pytest.raises(ValueError, match="approved"):
        gate.commit(0)
    assert gate.check(0, [response(first)])
    gate.request(1)
    with pytest.raises(ValueError, match="approved"):
        gate.commit(1)
    clock[0] = 5
    with pytest.raises(ValueError, match="fresh"):
        gate.commit(0)
    assert not gate.is_admitted(0)


def test_refresh_invalidates_old_approval_without_extending_deadline():
    clock = [0.0]
    gate = BudgetAdmission(
        {"n": approved()},
        run_id="r",
        plan_hash="h",
        freshness=5,
        transfer_timeout=10,
        run_deadline=100,
        clock=lambda: clock[0],
    )
    request = gate.request(0)
    assert gate.check(0, [response(request)])
    refreshed = gate.request(0)
    with pytest.raises(ValueError, match="approved"):
        gate.commit(0)
    assert not gate.check(
        0, [response(refreshed)]
    )  # New request needs a new sample sequence.
    clock[0] = 10
    with pytest.raises(TimeoutError, match="shared deadline"):
        gate.check(0, [response(refreshed, seq=2)])

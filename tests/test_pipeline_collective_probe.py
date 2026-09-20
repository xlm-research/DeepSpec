"""The communication probe cannot substitute one node for a planned M3 world."""

import pytest

from deepspec.pipeline.planning import build_plan
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


@pytest.mark.parametrize("layout", ["M1-12", "M3"])
def test_collective_probe_uses_frozen_rank_groups_and_cursors(layout):
    from tests.pipeline_collective_probe import rank_contract

    config = task_config(layout, steps=5)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="probe", now=100
    ).to_dict()
    for expected in plan["training_ranks"]:
        rank = expected["global_rank"]
        report = rank_contract(
            plan,
            rank=rank,
            local_rank=expected["local_rank"],
            world=8,
            node_id=expected["node_id"],
        )
        assert report["tp_group"] == list(range(rank // 4 * 4, rank // 4 * 4 + 4))
        assert report["dp_group"] == [rank % 4, rank % 4 + 4]
        assert report["positions"] == list(range(rank // 4, 20, 2))
        assert report["native_cursor"] == 10 and report["sample_cursor"] == 20
        with pytest.raises(ValueError, match="identity"):
            rank_contract(
                plan,
                rank=rank,
                local_rank=expected["local_rank"],
                world=8,
                node_id="wrong-node",
            )


@pytest.mark.parametrize("steps,dp", [(1, 1), (3, 2), (5, 2)])
def test_cpu_data_probe_counts_are_not_hard_coded(steps, dp):
    from tests.pipeline_rank_probe import data_contract

    config = {
        "steps": steps,
        "consumer_dp": dp,
        "consumer_world_size": 4 * dp,
        "samples_per_update": 4,
        "samples": [{"position": p, "length": 16} for p in range(4 * steps)],
    }
    result = data_contract(config, rank=4 * (dp - 1))
    assert result["steps"] == steps and result["gas"] == 4 // dp
    assert result["positions"] == list(range(dp - 1, 4 * steps, dp))
    assert result["native_cursor"] == 4 * steps // dp
    config["samples"].pop()
    with pytest.raises(ValueError, match="complete updates"):
        data_contract(config, rank=0)

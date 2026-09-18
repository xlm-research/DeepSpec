"""Guard node placement and accounting when local GPU IDs repeat across nodes."""

import json
import os
import subprocess
import sys

import pytest

from deepspec.pipeline.cluster import gpu_processes, select_nodes
from deepspec.pipeline.memory import GIB, feature_budget
from deepspec.pipeline.run import summarize_events


def test_cleanup_stops_only_exact_run_processes():
    from deepspec.pipeline.cluster import stop_run_processes

    run_id = f"cleanup-test-{os.getpid()}"
    children = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=dict(os.environ, DEEPSPEC_PIPELINE_RUN_ID=marker),
        )
        for marker in (run_id, run_id + "-other")
    ]
    try:
        signals = stop_run_processes(run_id)
        assert {row["pid"] for row in signals} == {children[0].pid}
        assert children[0].wait(timeout=5) < 0
        assert children[1].poll() is None
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_exiting_gpu_worker_stays_owned_but_reused_pid_does_not(tmp_path, monkeypatch):
    from deepspec.pipeline.run import check_gpu_ownership

    monkeypatch.setattr(
        "deepspec.pipeline.cluster.subprocess.check_output",
        lambda *args, **kwargs: "77, GPU-test, 53174\n",
    )
    process = tmp_path / "77"
    process.mkdir()

    def status(state, start):
        fields = [state, "1", *(["0"] * 17), str(start)]
        (process / "stat").write_text("77 (worker) " + " ".join(fields))

    status("S", 12345)
    (process / "environ").write_bytes(b"DEEPSPEC_PIPELINE_RUN_ID=test-run\0")
    known = set()
    local_known = set()
    monkeypatch.setattr(
        "deepspec.pipeline.cluster.gpu_processes",
        lambda run_id, known_owned: gpu_processes(
            run_id, known_owned, proc_root=tmp_path
        ),
    )
    assert gpu_processes("test-run", known, proc_root=tmp_path)[0]["owned"]
    check_gpu_ownership(tmp_path, "test-run", local_known)
    (process / "environ").write_bytes(b"")
    assert gpu_processes("test-run", known, proc_root=tmp_path)[0]["owned"]
    check_gpu_ownership(tmp_path, "test-run", local_known)
    status("Z", 12345)
    assert gpu_processes("test-run", known, proc_root=tmp_path) == []
    check_gpu_ownership(tmp_path, "test-run", local_known)
    status("S", 99999)
    assert not gpu_processes("test-run", known, proc_root=tmp_path)[0]["owned"]
    with pytest.raises(RuntimeError, match="Another job"):
        check_gpu_ownership(tmp_path, "test-run", local_known)
    # An owned startup helper may initialize CUDA before inheriting a run marker.
    monkeypatch.setattr("deepspec.orchestration.process.descendants", lambda pid: {77})
    check_gpu_ownership(tmp_path, "test-run", local_known)
    monkeypatch.setattr("deepspec.orchestration.process.descendants", lambda pid: set())
    check_gpu_ownership(tmp_path, "test-run", local_known)
    status("S", 100000)
    with pytest.raises(RuntimeError, match="Another job"):
        check_gpu_ownership(tmp_path, "test-run", local_known)


def test_node_selection_rejects_aliases_for_the_same_machine():
    nodes = [
        {
            "Alive": True,
            "NodeID": "a",
            "NodeManagerAddress": "10.0.0.1",
            "Resources": {"GPU": 8},
        }
    ]
    with pytest.raises(ValueError, match="distinct"):
        select_nodes(nodes, "a", "10.0.0.1")
    with pytest.raises(ValueError, match="live Ray node"):
        select_nodes(nodes, "a", "missing")


def test_memory_limits_apply_to_each_node_and_include_production_staging():
    snapshot = {
        "physical_bytes": 1024 * GIB,
        "limit_bytes": 1024 * GIB,
        "headroom_bytes": 1024 * GIB,
    }
    producer = feature_budget(0, 8 * GIB, 8, 0, 4, writer=True, snapshot=snapshot)
    consumer = feature_budget(
        64 * GIB, 8 * GIB, 8, 4, 4, writer=False, snapshot=snapshot
    )
    combined = feature_budget(64 * GIB, 8 * GIB, 8, 4, 4, snapshot=snapshot)
    assert (
        producer["scratch_bound_bytes"] + consumer["scratch_bound_bytes"]
        == combined["scratch_bound_bytes"]
    )
    assert producer["pool_bytes"] == 0
    small_consumer = {**snapshot, "limit_bytes": 128 * GIB}
    with pytest.raises(ValueError, match="exceeds node budget"):
        feature_budget(
            64 * GIB, 8 * GIB, 8, 4, 4, writer=False, snapshot=small_consumer
        )


def test_bounded_writer_staging_is_independent_of_retained_pool_window():
    snapshot = {
        key: 2048 * GIB for key in ("physical_bytes", "limit_bytes", "headroom_bytes")
    }
    bounded = feature_budget(
        1024 * GIB, 8 * GIB, 140, 4, 4, writer_inflight=2, snapshot=snapshot
    )
    assert bounded["scratch_bound_bytes"] == (3 * 2 + 2 * 4 * 4) * 8 * GIB
    with pytest.raises(ValueError, match="exceeds node budget"):
        feature_budget(1024 * GIB, 8 * GIB, 140, 4, 4, snapshot=snapshot)


def test_repeated_local_gpu_ids_are_valid_only_on_the_assigned_distinct_nodes(tmp_path):
    config = {
        "samples": [{"position": 0}],
        "consumer_world_size": 4,
        "steps": 1,
        "cluster_address": "head:1",
        "producer_node_id": "a",
        "consumer_node_id": "b",
    }
    rows = [
        {"event": "producer_worker", "node_id": "a", "ray_gpu_ids": [str(i)]}
        for i in range(4)
    ]
    rows.append(
        {
            "event": "consumer_launcher",
            "node_id": "b",
            "ray_gpu_ids": [str(i) for i in range(4)],
        }
    )
    for rank in range(4):
        for event in (
            "claimed",
            "received",
            "gpu_ready",
            "compute_start",
            "compute_end",
        ):
            rows.append(
                {"event": event, "position": 0, "reader": rank, "monotonic": 1.0}
            )
        rows.append({"event": "optimizer_update_complete", "reader": rank, "step": 1})
        rows.append({"event": "context_gradient_verified", "reader": rank})
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = summarize_events(path, config)
    assert result["verified_rank_receives"] == 4
    assert result["producer_gpu_assignments"] == [("a", str(i)) for i in range(4)]
    with pytest.raises(RuntimeError, match="assigned nodes"):
        summarize_events(path, {**config, "consumer_node_id": "wrong"})
    rows[4]["node_id"] = "a"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(RuntimeError, match="GPU assignments"):
        summarize_events(path, config)


def test_dp_launchers_share_rendezvous_but_have_distinct_node_ranks():
    from deepspec.pipeline.actors import consumer_command
    from deepspec.pipeline.topology import consumer_microbatches

    config = {
        "run_id": "dp-test",
        "consumer_dp": 2,
        "consumer_world_size": 8,
        "consumer_rendezvous": {"host": "10.0.0.2", "port": 23456},
        "samples": [None] * 12,
    }
    commands = [consumer_command(config, rank) for rank in (0, 1)]
    for rank, command in enumerate(commands):
        assert f"--node-rank={rank}" in command
        assert "--nnodes=2" in command and "--nproc-per-node=4" in command
        assert "--master-addr=10.0.0.2" in command
        assert "--master-port=23456" in command and "--standalone" not in command
    assert consumer_microbatches(config) == 6
    with pytest.raises(ValueError, match="DP microstep"):
        consumer_microbatches({**config, "samples": [None] * 3})


def test_separated_consumer_dp2_uses_one_eight_rank_launcher():
    from deepspec.pipeline.actors import consumer_command

    config = {"consumer_dp": 2, "consumer_world_size": 8, "consumer_nodes": 1}
    command = consumer_command(config, 0)
    assert "--standalone" in command and "--nproc-per-node=8" in command
    assert not any(arg.startswith("--nnodes") for arg in command)
    with pytest.raises(ValueError, match="node rank"):
        consumer_command(config, 1)


def test_dp_groups_require_capacity_on_their_separate_role_nodes():
    nodes = [
        {
            "Alive": True,
            "NodeID": name,
            "NodeManagerAddress": name,
            "Resources": {"GPU": 4},
        }
        for name in ("a", "b")
    ]
    with pytest.raises(ValueError, match="eight GPUs"):
        select_nodes(nodes, "a", "b", consumer_dp=2)
    nodes[1]["Resources"]["GPU"] = 8
    assert select_nodes(nodes, "a", "b", consumer_dp=2) == nodes
    with pytest.raises(ValueError, match="eight GPUs"):
        select_nodes(nodes, "a", "b", consumer_dp=2, producer_dp=2)
    nodes[0]["Resources"]["GPU"] = 8
    assert select_nodes(nodes, "a", "b", consumer_dp=2, producer_dp=2) == nodes


@pytest.mark.parametrize("producer_dp", [1, 2])
def test_dp_events_validate_sample_ownership_native_cursor_and_rank_placement(
    tmp_path, producer_dp
):
    from deepspec.pipeline.topology import sample_readers

    config = {
        "samples": [{"position": p} for p in range(4)],
        "consumer_world_size": 8,
        "consumer_dp": 2,
        "producer_dp": producer_dp,
        "samples_per_update": 4,
        "steps": 1,
        "cluster_address": "head:1",
        "producer_node_id": "a",
        "producer_node_ids": ["a", "a"] if producer_dp == 2 else ["a"],
        "consumer_node_id": "b",
        "consumer_node_ids": ["b"] if producer_dp == 2 else ["b", "a"],
        "consumer_nodes": 1 if producer_dp == 2 else 2,
        "consumer_hostnames": ["host-b", "host-b"]
        if producer_dp == 2
        else ["host-b", "host-a"],
        "role_separation": producer_dp == 2,
    }
    rows = (
        [
            {
                "event": "producer_worker",
                "node_id": node,
                "ray_gpu_ids": [str(i + 4 * rank) if producer_dp == 2 else str(i + 4)],
                "producer_rank": rank,
                "tp_rank": i,
            }
            for rank, node in enumerate(config["producer_node_ids"])
            for i in range(4)
        ]
        + [
            {
                "event": "consumer_launcher",
                "node_id": "b",
                "ray_gpu_ids": [str(i) for i in range(8 if producer_dp == 2 else 4)],
            },
        ]
        + (
            []
            if producer_dp == 2
            else [
                {
                    "event": "consumer_launcher",
                    "node_id": "a",
                    "ray_gpu_ids": [str(i) for i in range(4)],
                }
            ]
        )
    )
    for sample in config["samples"]:
        if producer_dp == 2:
            for event in (
                "reserved",
                "inference_start",
                "inference_end",
                "write_started",
                "ready",
                "write_complete",
            ):
                rows.append(
                    {
                        "event": event,
                        "position": sample["position"],
                        "producer_rank": sample["position"] % 2,
                        "monotonic": 1.0,
                    }
                )
        for rank in sample_readers(config, sample["position"]):
            for event in (
                "claimed",
                "received",
                "gpu_ready",
                "compute_start",
                "compute_end",
            ):
                rows.append(
                    {
                        "event": event,
                        "position": sample["position"],
                        "reader": rank,
                        "monotonic": 1.0,
                    }
                )
    for rank in range(8):
        rows.extend(
            [
                {
                    "event": "optimizer_update_complete",
                    "reader": rank,
                    "step": 1,
                    "next_global_microbatch": 2,
                    "next_global_sample": 4,
                },
                {"event": "context_gradient_verified", "reader": rank},
                {
                    "event": "consumer_rank_initialized",
                    "reader": rank,
                    "dp_rank": rank // 4,
                    "tp_rank": rank % 4,
                    "world_size": 8,
                    "gradient_accumulation_steps": 2,
                    "hostname": config["consumer_hostnames"][rank // 4],
                },
            ]
        )
    path = tmp_path / "events.jsonl"

    def write():
        path.write_text("\n".join(json.dumps(row) for row in rows))

    write()
    assert summarize_events(path, config)["verified_rank_receives"] == 16
    row = next(r for r in rows if r["event"] == "received")
    row["reader"] = 4
    write()
    with pytest.raises(RuntimeError, match="rank events"):
        summarize_events(path, config)
    row["reader"] = 0
    row = next(r for r in rows if r["event"] == "optimizer_update_complete")
    row["next_global_microbatch"] = 4
    write()
    with pytest.raises(RuntimeError, match="checkpoint cursor"):
        summarize_events(path, config)
    row["next_global_microbatch"] = 2
    row = next(r for r in rows if r["event"] == "consumer_rank_initialized")
    row["hostname"] = "host-a"
    write()
    with pytest.raises(RuntimeError, match="DP/TP topology"):
        summarize_events(path, config)
    row["hostname"] = "host-b"
    if producer_dp == 2:
        row = next(r for r in rows if r["event"] == "ready")
        row["producer_rank"] = 1
        write()
        with pytest.raises(RuntimeError, match="wrongly routed"):
            summarize_events(path, config)
        row["producer_rank"] = 0
        launcher = next(r for r in rows if r["event"] == "consumer_launcher")
        launcher.update(node_id="a", ray_gpu_ids=[str(i) for i in range(8, 16)])
        write()
        with pytest.raises(RuntimeError, match="separate physical nodes"):
            summarize_events(path, config)


def test_overlapping_producer_intervals_are_counted_once():
    from deepspec.pipeline.run import merge_intervals

    assert merge_intervals([(4, 8), (1, 5), (2, 3), (10, 12)]) == [(1, 8), (10, 12)]

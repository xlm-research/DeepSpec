"""Exercise independent sampling through transient and persistent GPU query faults."""

import subprocess

import pytest

from deepspec.pipeline.execution import read_events
from deepspec.pipeline.metrics import require_metrics, summarize_metrics
from deepspec.pipeline.observation import ResourceSampler
from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import EventWriter
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


@pytest.mark.parametrize("timeouts", [1, 2, None, "command_failure"])
def test_gpu_sampling_retries_are_bounded_and_preserve_failures(
    tmp_path, monkeypatch, timeouts
):
    config = task_config("M0")
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="sampling", now=100
    ).to_dict()
    node = next(iter(plan["node_budgets"]))
    monkeypatch.setattr(
        "deepspec.pipeline.memory.node_memory", lambda: {"headroom_bytes": 300}
    )
    monkeypatch.setattr(
        "deepspec.pipeline.cluster.process_memory", lambda run_id: {"rss_bytes": 100}
    )
    calls = []
    observed = [{"pid": 123, "gpu_uuid": "gpu-0", "used_mib": "42", "owned": True}]

    def query(run_id):
        calls.append(run_id)
        if timeouts == "command_failure":
            raise subprocess.CalledProcessError(1, ["nvidia-smi"])
        if timeouts is None or len(calls) <= timeouts:
            raise subprocess.TimeoutExpired(["nvidia-smi"], 10)
        return observed

    monkeypatch.setattr("deepspec.pipeline.cluster.gpu_processes", query)
    sampler = ResourceSampler(plan, node, tmp_path)
    sampler.stop(timeout=5)
    with EventWriter(
        tmp_path / "events/production.jsonl",
        run_id=plan["run_id"],
        plan_hash=plan["plan_hash"],
        sender_identity={"component": "producer"},
    ) as writer:
        writer.emit(
            "feature_produced",
            {
                "position": 0,
                "nbytes": 20,
                "tokens": sum(s["length"] for s in plan["samples"]),
                "duration_seconds": 1,
            },
            basis="observed",
        )
        writer.emit(
            "phase_duration",
            {"phase": "production", "duration_seconds": 1},
            basis="observed",
        )
    events = read_events(tmp_path)
    samples = [event for event in events if event["event"] == "resource_sample"]
    assert len(samples) >= 2
    report = summarize_metrics(plan, events)
    if timeouts in (1, 2):
        assert require_metrics(plan, events)["missing"] == []
        assert len(calls) == len(samples) + timeouts
        assert all(event["data"]["gpu_processes"] == observed for event in samples)
        retries = [
            error
            for event in samples
            for error in event["data"]
            .get("sample_retries", {})
            .get("gpu_processes", [])
        ]
        assert len(retries) == timeouts
        assert all("TimeoutExpired" in error for error in retries)
    else:
        attempts = 3 if timeouts is None else 1
        assert len(calls) == attempts * len(samples)
        assert len(report["missing"]) == len(samples)
        assert all(item["metric"] == "gpu_processes" for item in report["missing"])
        assert all(event["data"]["gpu_processes"] is None for event in samples)
        for event in samples:
            retries = event["data"].get("sample_retries", {}).get("gpu_processes", [])
            assert len(retries) == attempts - 1
        with pytest.raises(ValueError, match="Required runtime metrics"):
            require_metrics(plan, events)

"""Explicit GPU sharing preserves occupancy evidence and unrelated processes."""

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import Deadline, PipelineError, message_envelope
from deepspec.pipeline.schema import normalize_task_config, upgrade_task_config
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


def test_shared_policy_is_explicit_validated_and_bound_to_config_hash():
    config = task_config("M1-22")
    exclusive = normalize_task_config(config)
    assert exclusive.to_dict()["gpu_sharing"] == "exclusive"
    config["gpu_sharing"] = "shared"
    shared = normalize_task_config(config)
    assert shared.config_hash != exclusive.config_hash
    config["gpu_sharing"] = "ignore"
    with pytest.raises(PipelineError, match="shared"):
        normalize_task_config(config)


def test_cli_preserves_explicit_sharing_through_legacy_upgrade(tmp_path, monkeypatch):
    from deepspec.pipeline import legacy, run

    captured = []

    def capture(config, **kwargs):
        captured.append(upgrade_task_config(config).to_dict())
        return {"state": "succeeded"}

    monkeypatch.setattr(legacy, "run_config", capture)
    assert (
        run.main(
            [
                "--source",
                str(tmp_path / "input.jsonl"),
                "--output",
                str(tmp_path / "run"),
                "--gpu-sharing",
                "shared",
            ]
        )
        == 0
    )
    assert captured[0]["gpu_sharing"] == "shared"


@pytest.mark.parametrize("sharing", ["exclusive", "shared"])
def test_gpu_admission_and_cleanup_preserve_other_job(tmp_path, monkeypatch, sharing):
    from deepspec.pipeline.cluster import NodeAgent

    config = task_config("M0", output_dir=tmp_path)
    config["gpu_sharing"] = sharing
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="shared-run", now=100
    ).to_dict()
    other_job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    external = [
        {
            "pid": other_job.pid,
            "gpu_uuid": "GPU-a-0",
            "owned": False,
            "used_mib": "1024",
        }
    ]
    monkeypatch.setattr(
        "deepspec.pipeline.cluster.gpu_processes", lambda *args: external
    )
    agent = NodeAgent(plan, "node-a", "fence")
    try:
        request = message_envelope(
            plan["run_id"],
            plan["plan_hash"],
            {"component": "test"},
            gpu_uuids=["GPU-a-0"],
        )
        if sharing == "shared":
            reply = agent.check_allocated_devices(request)
            assert reply["external_processes"] == external
            assert reply["gpu_sharing"] == "shared"
        else:
            with pytest.raises(RuntimeError, match="external processes"):
                agent.check_allocated_devices(request)
        assert agent.close()["cleanup_complete"]
        assert other_job.poll() is None
    finally:
        agent.close()
        other_job.terminate()
        other_job.wait(timeout=5)


@pytest.mark.parametrize(
    "sharing,reply_policy,allowed",
    [
        ("exclusive", "exclusive", False),
        ("exclusive", "shared", False),
        ("shared", "exclusive", False),
        ("shared", "shared", True),
    ],
)
def test_native_gate_enforces_frozen_sharing_policy_and_records_occupancy(
    tmp_path, sharing, reply_policy, allowed
):
    from deepspec.pipeline.vllm_adapter import NativeCoordination

    config = task_config("M0", output_dir=tmp_path)
    config["gpu_sharing"] = sharing
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="gate-run", now=100
    ).to_dict()
    external = [{"pid": 123, "gpu_uuid": "GPU-a-0", "owned": False, "used_mib": "1024"}]

    async def check(request):
        return message_envelope(
            plan["run_id"],
            plan["plan_hash"],
            {"component": "node_agent"},
            node_id="node-a",
            gpu_uuids=request["gpu_uuids"],
            gpu_sharing=reply_policy,
            external_processes=external,
        )

    gate = NativeCoordination(
        plan,
        pg_ids={p["id"]: "actual-" + p["id"] for p in plan["placement_groups"]},
        node_agents={
            "node-a": SimpleNamespace(
                check_allocated_devices=SimpleNamespace(remote=check)
            )
        },
        tokens={"node-a": "fence"},
    )
    try:
        call = gate._check_devices(
            {"node_id": "node-a", "gpu_uuids": ["GPU-a-0"]}, Deadline.after(5)
        )
        if allowed:
            asyncio.run(call)
        else:
            with pytest.raises(ValueError, match="sharing policy"):
                asyncio.run(call)
    finally:
        gate.events.close()
    if allowed:
        events = [
            json.loads(line)
            for line in (tmp_path / "events/native-gate.jsonl").read_text().splitlines()
        ]
        assert events[-1]["event"] == "gpu_occupancy_observed"
        assert events[-1]["data"]["external_processes"] == external

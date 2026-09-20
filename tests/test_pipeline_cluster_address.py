"""An auto-discovered Ray cluster must not change during preparation."""

import json
import time

import pytest

from deepspec.pipeline import cli
from deepspec.pipeline.runtime import PipelineError
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


@pytest.mark.parametrize("resolved", ["10.123.0.1:26379", None])
def test_preview_pins_auto_address_before_a_later_inspection(
    tmp_path, monkeypatch, resolved
):
    config = task_config("M0", output_dir=tmp_path / "run")
    config["ray_address"] = "auto"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    now = [time.monotonic()]
    monkeypatch.setattr(cli.time, "monotonic", lambda: now[0])
    addresses = []

    def inspect(current, run):
        addresses.append(current["ray_address"])
        facts = node_facts(current, now=now[0])
        for fact in facts:
            if resolved:
                fact["ray_address"] = resolved
        return facts

    def prepare(current, run):
        assert current["ray_address"] == resolved
        now[0] += 10
        return input_plan(current), {"run_id": run.run_id}

    monkeypatch.setattr(cli, "inspect_nodes", inspect)
    monkeypatch.setattr(cli, "prepare_input", prepare)
    if resolved is None:
        with pytest.raises(PipelineError, match="resolved Ray address"):
            cli.preview(path)
    else:
        result = cli.preview(path)
        assert addresses == ["auto", resolved]
        output, plan, _ = cli.load_run(tmp_path / "run")
        assert plan["config"]["ray_address"] == resolved
        assert result["plan_hash"] == plan["plan_hash"]

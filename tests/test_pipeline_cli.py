import json
import sys
import types

import pytest

from deepspec.pipeline import cli
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


def test_legacy_cli_allocation_deadline_survives_v3_upgrade(tmp_path, monkeypatch):
    from deepspec.pipeline import legacy, run
    from deepspec.pipeline.schema import upgrade_task_config

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
                "--allocation-timeout-seconds",
                "600",
                "--timeout-seconds",
                "3600",
            ]
        )
        == 0
    )
    policy = captured[0]["timeouts_seconds"]
    assert policy["allocation"] == 600
    assert policy["run"] == policy["initialization"] == policy["transfer"] == 3600
    assert policy["cleanup"] == 120


def install_cpu_backend(monkeypatch):
    calls = {"inspect": 0, "prepare": 0}

    def inspect(config, run):
        calls["inspect"] += 1
        return node_facts(config, now=cli.time.monotonic())

    def prepare(config, run):
        calls["prepare"] += 1
        return input_plan(config), {"schema_version": 2, "run_id": run.run_id}

    monkeypatch.setattr(cli, "inspect_nodes", inspect)
    monkeypatch.setattr(cli, "prepare_input", prepare)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preview attempted a GPU/model/pool side effect")

    monkeypatch.setitem(
        sys.modules, "ray", types.SimpleNamespace(init=forbidden, remote=forbidden)
    )
    monkeypatch.setattr("deepspec.pipeline.store.TensorStore", forbidden)
    return calls


def test_preview_writes_consistent_frozen_files(tmp_path, monkeypatch, capsys):
    calls = install_cpu_backend(monkeypatch)
    config = task_config(output_dir=tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert cli.main(["preview", "--config", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    run = tmp_path / "run"
    for file in (
        "config.normalized.json",
        "plan.json",
        "environment.json",
        "inputs/input-plan.json",
        "pipeline.json",
        "status.json",
    ):
        assert (run / file).is_file()
    plan = json.loads((run / "plan.json").read_text())
    status = json.loads((run / "status.json").read_text())
    assert status["run_id"] == result["run_id"] == plan["run_id"]
    assert (
        status["state"] == "preparing" and status["phase_detail"] == "preview_complete"
    )
    assert calls == {"inspect": 1, "prepare": 1}


def test_preview_upgrades_legacy_dp2_dp1_before_any_gpu_side_effect(
    tmp_path, monkeypatch, capsys
):
    from tests.test_pipeline_cluster import legacy_task

    calls = install_cpu_backend(monkeypatch)
    old = legacy_task(inference_dp=2, training_dp=1)
    old["output_dir"] = str(tmp_path / "run")
    old["prefetch_bytes"] = 1024**3
    old["store"]["rdma_devices"] = "mlx5_0,mlx5_2"
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(old))
    assert cli.main(["preview", "--config", str(path)]) == 0
    capsys.readouterr()
    plan = json.loads((tmp_path / "run/plan.json").read_text())
    assert plan["config"]["inference"]["dp"] == 2
    assert plan["config"]["training"]["dp"] == 1
    assert plan["config"]["transport"]["rdma_devices"] == "mlx5_0,mlx5_2"
    assert plan["config"]["transport"]["protocol"] == "tcp"
    assert calls == {"inspect": 1, "prepare": 1}


@pytest.mark.parametrize(
    "mutation", ["output_exists", "placeholder", "unknown", "unsupported_transport"]
)
def test_invalid_config_has_no_inspection_or_preparation(
    tmp_path, monkeypatch, capsys, mutation
):
    calls = install_cpu_backend(monkeypatch)
    config = task_config(output_dir=tmp_path / "run")
    if mutation == "output_exists":
        (tmp_path / "run").mkdir()
    elif mutation == "placeholder":
        config["model_path"] = "${MODEL}"
    elif mutation == "unknown":
        config["typo"] = 1
    elif mutation == "unsupported_transport":
        config["transport"]["protocol"] = "rdma"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert cli.main(["preview", "--config", str(path)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert {
        "code",
        "message",
        "run_id",
        "phase",
        "node_id",
        "field_path",
        "retryable",
    } <= error.keys()
    assert error["field_path"]
    assert calls == {"inspect": 0, "prepare": 0}


def test_missing_native_seam_is_explicit_and_cannot_preview_successfully(
    tmp_path, monkeypatch, capsys
):
    calls = install_cpu_backend(monkeypatch)

    def missing(config, run):
        facts = node_facts(config, now=cli.time.monotonic())
        facts[0]["capabilities"]["borrowed_pg"] = False
        return facts

    monkeypatch.setattr(cli, "inspect_nodes", missing)
    config = task_config(output_dir=tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert cli.main(["preview", "--config", str(path)]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "BACKEND_CAPABILITY_MISSING"
    assert calls["prepare"] == 0


def test_preparation_exception_preserves_identity_in_error_and_status(
    tmp_path, monkeypatch, capsys
):
    install_cpu_backend(monkeypatch)

    def fail(config, run):
        raise RuntimeError("native input preparation failed")

    monkeypatch.setattr(cli, "prepare_input", fail)
    config = task_config(output_dir=tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert cli.main(["preview", "--config", str(path)]) == 3
    error = json.loads(capsys.readouterr().err)
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert error["run_id"] == status["run_id"]
    assert status["state"] == "failed"


def test_cancel_fences_identity_and_keeps_first_request_and_reason(tmp_path, capsys):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from deepspec.pipeline.controller import RunController
    from deepspec.pipeline.planning import build_plan
    from deepspec.pipeline.runtime import atomic_json

    config = task_config("M0", output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="cancel-run", now=100
    ).to_dict()
    atomic_json(tmp_path / "plan.json", plan)
    atomic_json(tmp_path / "config.normalized.json", plan["config"])
    active = threading.Event()

    class Operations:
        def check(self):
            pass

        def run(self, *, stop, **kw):
            active.set()
            stop.wait(5)
            return {"ready": True}

        def __getattr__(self, name):
            def call(**kw):
                if name == "close_objects":
                    raise RuntimeError("cleanup noise")
                return {"ready": True, "cleanup_complete": True}

            return call

    controller = RunController(plan, Operations())
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(controller.run)
        assert active.wait(2)
        assert cli.main(["cancel", "--run-dir", str(tmp_path)]) == 130
        first = (tmp_path / "control/cancel.json").read_bytes()
        result = future.result(timeout=2)
        assert result["state"] == "cancelled"
        assert result["reason"]["code"] == "CANCELLED"
        assert not result["cleanup_complete"]
        assert "cleanup noise" in str(result["cleanup"])
    assert cli.main(["cancel", "--run-dir", str(tmp_path)]) == 130
    assert first == (tmp_path / "control/cancel.json").read_bytes()
    status = json.loads((tmp_path / "status.json").read_text())
    status["run_id"] = "someone-else"
    atomic_json(tmp_path / "status.json", status)
    assert cli.main(["cancel", "--run-dir", str(tmp_path)]) == 2
    assert first == (tmp_path / "control/cancel.json").read_bytes()
    assert "RUN_IDENTITY_MISMATCH" in capsys.readouterr().err

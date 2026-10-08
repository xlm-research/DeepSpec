"""Exercise cluster reuse/startup and SSH argument forwarding without real Ray jobs."""

import json
import os
import subprocess
import sys
import threading
from unittest.mock import Mock

import pytest

from tools import launch_3node


@pytest.fixture
def launcher(monkeypatch):
    for key in launch_3node.FORWARD_KEYS | {
        "TRAIN_NODE_IPS",
        "INFERENCE_NODE_IPS",
        "RAY_ADDRESS",
        "RAY_PORT",
    }:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(launch_3node.subprocess, "check_output", lambda *a, **kw: "10.120.0.211\n")
    # A context manager is only needed when the head is already listening.
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(launch_3node.socket, "create_connection", lambda *a, **kw: connection)
    monkeypatch.setattr(launch_3node.time, "sleep", lambda _: None)
    instance = launch_3node.Launcher()
    instance.run_on = Mock(return_value=subprocess.CompletedProcess([], 0))
    return instance


def healthy_nodes(launcher):
    return dict.fromkeys([launcher.head, *launcher.workers], 8)


def test_healthy_cluster_is_reused_without_starting_or_stopping_nodes(launcher):
    launcher.nodes = Mock(return_value=healthy_nodes(launcher))
    launcher.ensure_cluster()
    launcher.run_on.assert_not_called()


def test_cold_start_launches_head_then_both_workers_in_parallel(launcher, monkeypatch):
    monkeypatch.setattr(
        launch_3node.socket,
        "create_connection",
        Mock(side_effect=ConnectionRefusedError),
    )
    launcher.nodes = Mock(side_effect=[{launcher.head: 8}, healthy_nodes(launcher)])
    barrier = threading.Barrier(2)
    started = []

    def start(node, command, **kwargs):
        if node != launcher.head:
            assert started[0] == launcher.head
            barrier.wait(timeout=5)  # Fails if workers are started serially.
        started.append(node)
        assert command[:2] == ["ray", "start"]
        assert "--node-ip-address" in command

    launcher.run_on.side_effect = start
    launcher.ensure_cluster()
    assert started[0] == launcher.head
    assert set(started) == set(healthy_nodes(launcher))


def test_only_missing_worker_is_started(launcher):
    launcher.nodes = Mock(
        side_effect=[
            {launcher.head: 8, launcher.workers[0]: 8},
            healthy_nodes(launcher),
        ]
    )
    launcher.ensure_cluster()
    assert launcher.run_on.call_count == 1
    assert launcher.run_on.call_args.args[0] == launcher.workers[1]


def test_insufficient_gpus_fail_before_starting_more_nodes(launcher):
    launcher.nodes = Mock(return_value={launcher.head: 4})
    with pytest.raises(RuntimeError, match="needs 8"):
        launcher.ensure_cluster()
    launcher.run_on.assert_not_called()


def test_worker_failure_prevents_training(launcher, monkeypatch):
    launcher.nodes = Mock(return_value={launcher.head: 8})
    launcher.run_on.side_effect = subprocess.CalledProcessError(255, "ssh")
    monkeypatch.setattr(launch_3node, "Launcher", lambda: launcher)
    monkeypatch.setattr(launch_3node.subprocess, "run", Mock())
    with pytest.raises(subprocess.CalledProcessError):
        launch_3node.main([])
    assert all(call.args[1][:2] == ["ray", "start"] for call in launcher.run_on.call_args_list)


def test_missing_nodes_time_out(launcher, monkeypatch):
    launcher.nodes = Mock(return_value={launcher.head: 8})
    monkeypatch.setattr(launch_3node.time, "monotonic", Mock(side_effect=[0, 121]))
    with pytest.raises(RuntimeError, match="Timed out waiting"):
        launcher.ensure_cluster()


def test_launch_from_worker_forwards_recipe_to_head(launcher, monkeypatch):
    launcher.local_ips = [launcher.workers[0]]
    launcher.ensure_cluster = Mock()
    monkeypatch.setattr(launch_3node, "Launcher", lambda: launcher)
    assert launch_3node.main(["dspark", "training.num_train_steps=10"]) == 0
    launcher.ensure_cluster.assert_not_called()
    assert launcher.run_on.call_args.args == (
        launcher.head,
        ["bash", str(launch_3node.ROOT / "run.sh"), "dspark", "training.num_train_steps=10"],
    )


@pytest.mark.parametrize("remote", [False, True])
def test_shell_forwarding_preserves_literal_arguments_and_environment(
    launcher,
    monkeypatch,
    tmp_path,
    remote,
):
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    (root / "h800conda.sh").write_text(":\n")
    monkeypatch.setattr(launch_3node, "ROOT", root)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh = fake_bin / "ssh"
    ssh.write_text("#!/bin/sh\nexec bash -s\n")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    literal = "spaces ' quotes ; $(touch SHOULD_NOT_EXIST) `touch SHOULD_NOT_EXIST`"
    launcher.env["OUTPUT_DIR"] = literal
    # Use the real run_on implementation with a fake SSH transport.
    result = launch_3node.Launcher.run_on(
        launcher,
        launcher.workers[0] if remote else launcher.head,
        [
            sys.executable,
            "-c",
            "import json, os, sys; print(json.dumps([sys.argv[1], os.environ['OUTPUT_DIR']]))",
            literal,
        ],
        capture_output=True,
    )
    assert json.loads(result.stdout) == [literal, literal]
    assert not (root / "SHOULD_NOT_EXIST").exists()


def test_dry_run_has_no_network_or_cluster_side_effects(launcher, monkeypatch, capsys):
    monkeypatch.setattr(launch_3node, "Launcher", lambda: launcher)
    launcher.ensure_cluster = Mock()
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(launch_3node.subprocess, "run", run)
    assert launch_3node.main(["dspark", "--dry-run", "training.num_train_steps=10"]) == 0
    launcher.run_on.assert_not_called()
    launcher.ensure_cluster.assert_not_called()
    assert "No SSH/Ray calls made" in capsys.readouterr().out
    assert run.call_args.args[0][-2:] == ["--dry-run", "training.num_train_steps=10"]


def test_print_config_does_not_start_ray_or_contact_head(launcher, monkeypatch):
    launcher.local_ips = [launcher.workers[0]]
    launcher.ensure_cluster = Mock()
    monkeypatch.setattr(launch_3node, "Launcher", lambda: launcher)
    assert launch_3node.main(["--print-config-only"]) == 0
    launcher.ensure_cluster.assert_not_called()
    assert launcher.run_on.call_args.args[0] == launcher.workers[0]


@pytest.mark.parametrize(
    "key,value",
    [
        ("INFERENCE_NODE_IPS", "10.0.0.2"),
        ("INFERENCE_NODE_IPS", "10.120.0.211,10.0.0.2"),
        ("TRAIN_NODE_IPS", "10.0.0.1,10.0.0.2"),
        ("RAY_ADDRESS", "10.0.0.99:6379"),
        ("RAY_PORT", "0"),
    ],
)
def test_invalid_cluster_settings_are_rejected(launcher, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        launch_3node.Launcher()

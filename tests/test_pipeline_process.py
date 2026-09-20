"""Run-local process identity and independent supervision contracts."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from deepspec.orchestration.process import (
    NodeLease,
    capture_process,
    signal_process,
    start_owned,
)


def fake_process(root, *, ticks=10, run_id="owned"):
    path = root / "77"
    path.mkdir(exist_ok=True)
    fields = ["S", "1", "77", *("0" for _ in range(16)), str(ticks)]
    (path / "stat").write_text("77 (worker with spaces) " + " ".join(fields))
    (path / "environ").write_bytes(f"DEEPSPEC_PIPELINE_RUN_ID={run_id}\0".encode())


def test_signal_requires_marker_and_unchanged_pid_start_time(tmp_path, monkeypatch):
    fake_process(tmp_path)
    identity = capture_process(77, "owned", proc_root=tmp_path)
    assert identity["start_ticks"] == 10
    monkeypatch.setattr(
        os,
        "pidfd_open",
        lambda *a: pytest.fail("must not signal reused PID"),
        raising=False,
    )
    fake_process(tmp_path, ticks=11)
    assert signal_process(identity, signal.SIGTERM, proc_root=tmp_path) == "unknown"
    fake_process(tmp_path, run_id="bystander")
    assert signal_process(identity, signal.SIGTERM, proc_root=tmp_path) == "unknown"


def test_unreadable_identity_is_unknown_and_never_signalled(tmp_path, monkeypatch):
    fake_process(tmp_path)
    identity = capture_process(77, "owned", proc_root=tmp_path)
    original = Path.read_bytes

    def denied(path):
        if path.name == "environ":
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", denied)
    assert signal_process(identity, signal.SIGKILL, proc_root=tmp_path) == "unknown"


def test_unverifiable_observation_resolves_only_after_process_exit(
    tmp_path, monkeypatch
):
    from deepspec.orchestration.process import unverifiable_processes

    fake_process(tmp_path)
    assert unverifiable_processes({77}, proc_root=tmp_path) == {77}

    stat = tmp_path / "77/stat"
    stat.write_text(stat.read_text().replace(") S ", ") Z "))
    assert unverifiable_processes({77}, proc_root=tmp_path) == set()
    stat.unlink()
    assert unverifiable_processes({77}, proc_root=tmp_path) == set()
    fake_process(tmp_path)
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path == stat:
            raise PermissionError("cannot prove process exit")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    assert unverifiable_processes({77}, proc_root=tmp_path) == {77}


def test_stale_cleanup_file_cannot_confirm_a_new_supervisor(tmp_path):
    from deepspec.orchestration.process import OwnedProcessHandle

    handle = OwnedProcessHandle.__new__(OwnedProcessHandle)
    handle.report_path = tmp_path / "cleanup.json"
    handle.identity = {"pid": 77, "start_ticks": 12, "run_id": "this-run"}
    for stale in (
        {"run_id": "another-run", "supervisor_pid": 77, "supervisor_start_ticks": 12},
        {"run_id": "this-run", "supervisor_pid": 77, "supervisor_start_ticks": 11},
    ):
        handle.report_path.write_text(json.dumps({**stale, "cleanup_complete": True}))
        assert not handle._report()["cleanup_complete"]


def test_lease_fences_old_sequences_tokens_and_expired_driver():
    now, stopped = [0.0], []
    lease = NodeLease(
        "token", timeout=5, on_expire=lambda: stopped.append(True), clock=lambda: now[0]
    )
    assert lease.heartbeat("token", 1)
    now[0] = 4
    assert not lease.heartbeat("token", 1)
    assert not lease.heartbeat("old", 2)
    now[0] = 5
    assert not lease.heartbeat("token", 2)
    lease.check()
    lease.check()
    assert stopped == [True]
    assert not lease.heartbeat("token", 3)


def test_node_watchdog_cleans_registered_process_while_control_thread_is_busy(tmp_path):
    from deepspec.pipeline.cluster import NodeAgent
    from deepspec.pipeline.planning import build_plan
    from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config

    config = task_config("M0", output_dir=tmp_path)
    config["timeouts_seconds"].update(lease=3, heartbeat=1, cleanup=2)
    p = build_plan(
        config,
        node_facts(config),
        input_plan(config),
        run_id="node-lease-probe",
        now=100,
    ).to_dict()
    agent = NodeAgent(p, "node-a", "fence", exit_on_orphan=False)
    owned = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env=dict(os.environ, DEEPSPEC_PIPELINE_RUN_ID=p["run_id"]),
    )
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    identity = capture_process(owned.pid, p["run_id"])
    request = {
        "schema_version": 3,
        "run_id": p["run_id"],
        "plan_hash": p["plan_hash"],
        "sender_identity": {"component": "controller"},
        "event_id": "register",
        "fencing_token": "fence",
        "process": identity,
    }
    try:
        agent.register_process(request)
        # No agent RPC is serviced here: the watchdog must make progress alone.
        time.sleep(3.5)
        owned.wait(timeout=3)
        deadline = time.monotonic() + 3
        while agent.cleanup_result is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert agent.cleanup_result["cleanup_complete"]
        assert agent.cleanup_result["orphan"]
        assert bystander.poll() is None
        assert not agent.heartbeat({**request, "sequence": 2})["accepted"]
        assert not (tmp_path / "status.json").exists()
        assert (tmp_path / "orphan-node-a.json").is_file()
    finally:
        agent.close()
        for process in (owned, bystander):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)


def test_nonblocking_handle_times_out_and_preserves_bystander(tmp_path):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    handle = start_owned(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=0.5,
        cleanup_timeout=3,
        env=dict(os.environ, DEEPSPEC_PIPELINE_RUN_ID="process-probe"),
    )
    try:
        assert handle.poll() is None
        started = time.monotonic()
        with pytest.raises((TimeoutError, subprocess.CalledProcessError)):
            handle.result()
        assert time.monotonic() - started < 5
        assert handle.stop(timeout=3)["cleanup_complete"]
        assert handle.stop(timeout=3)["cleanup_complete"]
        assert bystander.poll() is None
    finally:
        handle.stop(timeout=3)
        bystander.terminate()
        bystander.wait(timeout=3)


def test_supervisor_uses_a_separate_session_from_its_ray_actor_parent():
    handle = start_owned(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=5,
        cleanup_timeout=2,
    )
    try:
        assert handle.identity["group_id"] == handle.identity["pid"]
        assert handle.identity["group_id"] != os.getpgrp()
    finally:
        handle.stop(timeout=3)


def test_node_cleanup_leaves_supervisor_alive_to_reap_term_resistant_child(tmp_path):
    from deepspec.pipeline.cluster import NodeAgent
    from deepspec.pipeline.planning import build_plan
    from deepspec.pipeline.runtime import message_envelope
    from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config

    config = task_config("M0", output_dir=tmp_path)
    config["timeouts_seconds"].update(lease=10, heartbeat=1, cleanup=3)
    p = build_plan(
        config, node_facts(config), input_plan(config), run_id="stubborn-child", now=100
    ).to_dict()
    ready = tmp_path / "ready.json"
    script = (
        "import os,sys,time,signal,json; from pathlib import Path; "
        "from deepspec.orchestration.process import capture_process; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(json.dumps(capture_process(os.getpid(), os.environ['DEEPSPEC_PIPELINE_RUN_ID']))); "
        "time.sleep(30)"
    )
    handle = start_owned(
        [sys.executable, "-c", script, str(ready)],
        timeout=10,
        cleanup_timeout=3,
        report_path=tmp_path / "supervisor.json",
        env=dict(os.environ, DEEPSPEC_PIPELINE_RUN_ID=p["run_id"]),
    )
    agent = NodeAgent(p, "node-a", "fence")
    try:
        until = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < until:
            time.sleep(0.02)
        assert ready.exists()
        agent.register_process(
            message_envelope(
                p["run_id"],
                p["plan_hash"],
                {"component": "test"},
                fencing_token="fence",
                process=handle.identity,
                supervisor_report_path=str(handle.report_path),
            )
        )
        result = agent.cleanup(timeout=3)
        assert result["cleanup_complete"], result
        assert handle.stop(timeout=1)["cleanup_complete"]
    finally:
        if ready.exists():
            signal_process(json.loads(ready.read_text()), signal.SIGKILL)
        handle.stop(timeout=1)
        agent.close()


def test_subreaper_stops_new_session_grandchild_after_parent_exits(tmp_path):
    marker = tmp_path / "grandchild.json"
    child = "import time; time.sleep(30)"
    script = (
        "import json,subprocess,sys,time; from pathlib import Path; "
        "from deepspec.orchestration.process import capture_process; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True); "
        f"Path({str(marker)!r}).write_text(json.dumps(capture_process(p.pid,'descendant-probe'))); "
        "time.sleep(0.3)"
    )
    handle = start_owned(
        [sys.executable, "-c", script],
        timeout=5,
        cleanup_timeout=3,
        env=dict(os.environ, DEEPSPEC_PIPELINE_RUN_ID="descendant-probe"),
    )
    try:
        handle.result()
        import json

        identity = json.loads(marker.read_text())
        assert signal_process(identity, 0) == "released"
    finally:
        handle.stop(timeout=3)


@pytest.mark.parametrize("wait_for_child", [False, True])
def test_supervisor_survives_driver_sigkill_long_enough_to_clean_children(
    tmp_path, wait_for_child
):
    identity_path, report_path = tmp_path / "identity.json", tmp_path / "cleanup.json"
    ready_path = tmp_path / "child-ready"
    child = f"import time; from pathlib import Path; Path({str(ready_path)!r}).touch(); time.sleep(30)"
    script = (
        "import json,os,sys,time; from pathlib import Path; "
        "from deepspec.orchestration.process import start_owned; "
        f"h=start_owned([sys.executable,'-c',{child!r}],timeout=20,cleanup_timeout=3,"
        f"report_path={str(report_path)!r},env=dict(os.environ,DEEPSPEC_PIPELINE_RUN_ID='driver-kill-probe')); "
        f"Path({str(identity_path)!r}).write_text(json.dumps(h.identity)); time.sleep(30)"
    )
    driver = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 5
        while (
            not identity_path.exists() or (wait_for_child and not ready_path.exists())
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert identity_path.exists()
        if wait_for_child:
            assert ready_path.exists()
        driver.kill()
        driver.wait(timeout=3)
        while not report_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["reason"] == "local_parent_lost" and report["cleanup_complete"]
        if wait_for_child:
            assert report["processes"]
        assert all(signal_process(p, 0) == "released" for p in report["processes"])
    finally:
        if driver.poll() is None:
            driver.terminate()
            driver.wait(timeout=3)
        if identity_path.exists():
            signal_process(json.loads(identity_path.read_text()), signal.SIGTERM)

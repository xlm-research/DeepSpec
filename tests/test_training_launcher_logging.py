"""A stalled console reader must not stall the training process."""

import os
import pty
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path):
    root = Path(__file__).resolve().parents[1]
    relative_script = Path("examples/qwen3.8-27b-3node/run.sh")
    script = tmp_path / relative_script
    script.parent.mkdir(parents=True)
    shutil.copyfile(root / relative_script, script)
    (tmp_path / "h800conda.sh").write_text(":\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ray").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "python").write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if '--print-config-only' in sys.argv:\n"
        "    print('resolved training config')\n"
        "else:\n"
        "    sys.stdout.write('training log line\\n' * 131072)\n"
        "    print('TRAINING_COMPLETED')\n"
    )
    for executable in bin_dir.iterdir():
        executable.chmod(0o755)
    env = dict(os.environ)
    for name in ("LOAD_PATH", "CKPT_STEP", "TRAIN_NODE_IPS", "INFERENCE_NODE_IPS"):
        env.pop(name, None)
    env.update(
        PATH=f"{bin_dir}:/usr/bin:/bin",
        LOG_DIR=str(tmp_path / "logs"),
        OUTPUT_DIR=str(tmp_path / "output"),
        CACHE_DIR=str(tmp_path / "cache"),
    )
    return script, env


def test_training_finishes_when_terminal_is_not_reading(launcher):
    script, env = launcher
    master_fd, slave_fd = pty.openpty()
    process = None
    try:
        process = subprocess.Popen(
            ["bash", str(script)],
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            start_new_session=True,
        )
        # Leave the terminal open without reading it, as with a stalled console.
        try:
            returncode = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("Training stalled while the attached terminal was not reading logs")
        assert returncode == 0
        logs = list(Path(env["LOG_DIR"]).glob("*.log"))
        assert len(logs) == 1
        assert "TRAINING_COMPLETED" in logs[0].read_text()
    finally:
        if process is not None:
            # Include any process-substitution children created by the launcher.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        os.close(master_fd)
        os.close(slave_fd)


def test_print_config_only_remains_visible(launcher):
    script, env = launcher
    result = subprocess.run(
        ["bash", str(script), "--print-config-only"],
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
        check=True,
    )
    assert "resolved training config" in result.stdout

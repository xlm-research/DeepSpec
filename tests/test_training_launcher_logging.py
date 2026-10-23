"""A stalled console reader must not stall the training process."""

import json
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


@pytest.fixture
def loop_launcher(tmp_path):
    root = Path(__file__).resolve().parents[1]
    relative_dir = Path("examples/qwen3.5-loop-mhc-dspark-3-node")
    recipe = tmp_path / relative_dir
    recipe.mkdir(parents=True)
    for name in ("run.sh", "restart.sh"):
        shutil.copyfile(root / relative_dir / name, recipe / name)
    # Conda currently selects a shared AFS cache, even for node-local launches.
    (tmp_path / "h800conda.sh").write_text("export TRITON_CACHE_DIR=/afs/conda/cache\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os\n"
        "keys = ['TRITON_CACHE_DIR', 'TORCHSPEC_TRITON_CACHE_DIR', 'VLLM_QWEN3_5_MHC_BACKEND', "
        "'VLLM_QWEN3_5_MHC_FUSION']\n"
        "print(json.dumps({k: os.environ.get(k) for k in keys}))\n"
    )
    python.chmod(0o755)
    env = dict(os.environ)
    for key in (
        "LOAD_PATH",
        "CKPT_STEP",
        "TRAIN_NODE_IPS",
        "INFERENCE_NODE_IPS",
        "TORCHSPEC_TRITON_CACHE_DIR",
        "VLLM_QWEN3_5_MHC_BACKEND",
        "VLLM_QWEN3_5_MHC_FUSION",
    ):
        env.pop(key, None)
    env.update(
        PATH=f"{bin_dir}:/usr/bin:/bin",
        OUTPUT_DIR=str(tmp_path / "output"),
        CACHE_DIR=str(tmp_path / "cache"),
        TRITON_CACHE_DIR="/afs/inherited/cache",
    )
    return recipe, env


@pytest.mark.parametrize("cache_override", [None, "/tmp/custom triton cache"])
def test_loop_launcher_sets_node_local_cache_after_conda(loop_launcher, cache_override):
    recipe, env = loop_launcher
    if cache_override is not None:
        env["TORCHSPEC_TRITON_CACHE_DIR"] = cache_override
    result = subprocess.run(
        ["bash", str(recipe / "run.sh"), "--print-config-only"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    actual = json.loads(result.stdout.splitlines()[-1])
    assert actual["TRITON_CACHE_DIR"] == (cache_override or f"/tmp/torchspec-triton-{os.getuid()}")
    assert actual["TORCHSPEC_TRITON_CACHE_DIR"] == actual["TRITON_CACHE_DIR"]


@pytest.mark.parametrize("override", [False, True])
def test_loop_restart_preserves_explicit_mhc_selection(loop_launcher, tmp_path, override):
    recipe, env = loop_launcher
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "iter_0005000"
    checkpoint.mkdir(parents=True)
    for name in ("model", "optimizer", "lr_scheduler", "rng.pt", "meta.json"):
        (checkpoint / name).touch()
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("5000\n")
    env["LOAD_PATH"] = str(checkpoint_root)
    if override:
        env.update(VLLM_QWEN3_5_MHC_BACKEND="torch", VLLM_QWEN3_5_MHC_FUSION="none")
    result = subprocess.run(
        ["bash", str(recipe / "restart.sh"), "--print-config-only"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    actual = json.loads(result.stdout.splitlines()[-1])
    assert actual["VLLM_QWEN3_5_MHC_BACKEND"] == ("torch" if override else "triton")
    assert actual["VLLM_QWEN3_5_MHC_FUSION"] == ("none" if override else "all")
    assert actual["TRITON_CACHE_DIR"] == f"/tmp/torchspec-triton-{os.getuid()}"
    assert actual["TORCHSPEC_TRITON_CACHE_DIR"] == actual["TRITON_CACHE_DIR"]

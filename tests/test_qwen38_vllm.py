from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from deepspec.trainer.glm5_partitioned_swap import atomic_write_json, load_json
from deepspec.trainer.qwen3_8_vllm import (
    QwenVllmConfig,
    convert_hidden_states,
    live_process_group,
    run_worker_process,
)
from deepspec.trainer.qwen3_8_vllm_trainer import (
    Qwen3_8VllmDSparkTrainer,
    replica_layout,
)
from tests import test_qwen38_multinode_launcher as launcher_tests

REPO_ROOT = launcher_tests.REPO_ROOT


def teacher_sample(length=7):
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn(length, 3, 8, generator=generator).bfloat16()
    ids = torch.arange(length).long().unsqueeze(0)
    batch = {"input_ids": ids, "loss_mask": ids.remainder(2).bool()}
    tensors = {"token_ids": ids[0], "hidden_states": hidden}
    return tensors, batch


def convert(tensors, batch, **kwargs):
    return convert_hidden_states(
        tensors,
        batch,
        hidden_size=tensors["hidden_states"].shape[-1],
        num_layers=2,
        **kwargs,
    )


def test_features_preserve_decoder_outputs_and_exact_final_norm_output():
    tensors, batch = teacher_sample()
    features = convert(tensors, batch)
    torch.testing.assert_close(
        features["target_hidden_states"][0], tensors["hidden_states"][:, :2].flatten(1)
    )
    torch.testing.assert_close(
        features["target_last_hidden_states"][0],
        tensors["hidden_states"][:, -1],
        rtol=0,
        atol=0,
    )
    assert torch.equal(features["input_ids"], batch["input_ids"])
    assert torch.equal(features["loss_mask"], batch["loss_mask"])
    assert features["seq_len"].item() == 7


def test_cp2_shards_reconstruct_token_order_and_zero_only_padding():
    tensors, batch = teacher_sample()
    shards = [convert(tensors, batch, cp_size=2, cp_rank=i) for i in range(2)]
    # Seven tokens pad to eight; ranks own [0,1,6,pad] and [2,3,4,5].
    expected_positions = ([0, 1, 6], [2, 3, 4, 5])
    reference = convert(tensors, batch)
    for shard, positions in zip(shards, expected_positions):
        assert shard["context_chunk_len"].item() == 4
        assert torch.equal(shard["input_ids"], batch["input_ids"])
        for name in ("target_hidden_states", "target_last_hidden_states"):
            torch.testing.assert_close(
                shard[name][0, : len(positions)],
                reference[name][0, positions],
                rtol=0,
                atol=0,
            )
    assert not shards[0]["target_hidden_states"][0, -1].any()
    assert not shards[0]["target_last_hidden_states"][0, -1].any()


@pytest.mark.parametrize("failure", ["tokens", "shape", "nan"])
def test_bad_teacher_outputs_are_rejected(failure):
    tensors, batch = teacher_sample()
    if failure == "tokens":
        tensors["token_ids"] = tensors["token_ids"].flip(0)
    elif failure == "shape":
        tensors["hidden_states"] = tensors["hidden_states"][:, :2]
    else:
        tensors["hidden_states"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        convert(tensors, batch)


def test_tp4_cp2_model_group_has_one_teacher_and_one_cache_owner_per_cp_rank():
    layout = replica_layout(
        global_rank=13,
        local_rank=5,
        local_size=8,
        cp_size=2,
        tp_size=4,
        vllm_tp=4,
        devices=["7", "5", "3", "1", "6", "4", "2", "0"],
    )
    assert layout == (8, [8, 12], ["7", "5", "3", "1"])
    with pytest.raises(ValueError):
        replica_layout(
            global_rank=0,
            local_rank=0,
            local_size=4,
            cp_size=2,
            tp_size=4,
            vllm_tp=4,
            devices=["0", "1", "2", "3"],
        )


def test_two_feature_partitions_preserve_inflight_gradients(tmp_path):
    # Exercise the real partition iterator, using a CPU extractor in place of
    # the external teacher. The optimizer window deliberately spans partitions.
    trainer = Qwen3_8VllmDSparkTrainer.__new__(Qwen3_8VllmDSparkTrainer)
    trainer.global_rank = trainer._vllm_owner = 0
    trainer.world_size = 1
    trainer._cp_cache_owners = [0]
    trainer._vllm_devices = ["0"]
    trainer._vllm_control_group = None
    trainer.data_batch_micro_batches = (1, 1)
    trainer.parallel = SimpleNamespace(context_parallel_rank=0)
    trainer.device = torch.device("cpu")
    trainer.qwen_vllm_config = QwenVllmConfig(tensor_parallel_size=1)
    trainer._teacher_identity = {"full_layers": 64}
    trainer.args = SimpleNamespace(
        model=SimpleNamespace(target_model_name_or_path="unused"),
        data=SimpleNamespace(max_length=7),
    )
    trainer.checkpoint_dir_root = str(tmp_path / "checkpoints")
    trainer.data_batch_cache_root = str(tmp_path / "cache")
    trainer.data_batch_rank_cache_dir = str(tmp_path / "cache/rank_00000")
    trainer._initialize_data_batch_cache()
    tensors, batch = teacher_sample()
    batches = [
        dict(batch, attention_mask=torch.ones_like(batch["input_ids"]))
        for _ in range(2)
    ]

    def fake_worker(job_path, config, devices):
        job = load_json(job_path)
        for request in job["requests"]:
            data = torch.load(request["input_path"], weights_only=True)
            output_path = Path(request["output_paths"][0])
            output_path.parent.mkdir(parents=True)
            torch.save(convert(tensors, data), output_path)
        atomic_write_json(
            str(job_path) + ".complete",
            {"teacher": trainer._teacher_identity, "samples": [{}]},
        )

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with (
        patch(
            "deepspec.trainer.qwen3_8_vllm_trainer.run_worker_process",
            side_effect=fake_worker,
        ),
        patch("torch.cuda.synchronize"),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
        patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda out, obj, **kw: out.__setitem__(0, obj),
        ),
    ):
        for features in trainer.iter_training_batches(batches):
            assert trainer._data_batch_phase == "draft_training"
            assert trainer._data_batch_end_after_current
            (parameter * features["target_hidden_states"].float().sum() / 2).backward()
        expected_gradient = (
            convert(tensors, batch)["target_hidden_states"].float().sum()
        )
        torch.testing.assert_close(parameter.grad, expected_gradient)
        optimizer.step()
        torch.testing.assert_close(parameter, 1 - 0.1 * expected_gradient)
    assert trainer._active_data_batch_cache is None
    assert not list(Path(trainer.data_batch_rank_cache_dir).glob("data_batch_*"))


def test_new_launcher_uses_separate_config_and_leaves_native_default():
    helper = launcher_tests.Qwen38MultiNodeLauncherTest()
    result = helper._run_launcher(
        launcher=REPO_ROOT / "scripts/train/train_qwen3_8_27b_dspark_vllm.sh"
    )
    assert result.returncode == 0, result.stderr
    assert "config/dspark/dspark_qwen3_8_27b_vllm.py" in result.stdout
    assert "train.parallel.tp=4" in result.stdout
    assert "train.data_partitions=512" in result.stdout
    result = helper._run_launcher()
    assert result.returncode == 0, result.stderr
    assert "--config config/dspark/dspark_qwen3_8_27b.py" in result.stdout


def test_worker_entrypoint_is_inert_under_multiprocessing_spawn():
    import runpy

    with patch("deepspec.trainer.qwen3_8_vllm.worker_main") as worker:
        runpy.run_path(
            str(REPO_ROOT / "scripts/data/generate_qwen3_8_vllm_partition.py"),
            run_name="__mp_main__",
        )
    worker.assert_not_called()


@pytest.mark.parametrize("exit_code", [0, 7, None])
def test_worker_releases_descendants_after_success_failure_or_timeout(
    tmp_path, exit_code
):
    executable = tmp_path / "python"
    job = tmp_path / "job.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path(sys.argv[-1]).write_text(str(os.getpgrp()))\n"
        + ("time.sleep(60)\n" if exit_code is None else f"sys.exit({exit_code})\n")
    )
    executable.chmod(0o755)
    config = QwenVllmConfig(python_executable=str(executable), timeout_seconds=1)
    if exit_code is None:
        with pytest.raises(subprocess.TimeoutExpired):
            run_worker_process(job, config, ["0"])
    elif exit_code:
        with pytest.raises(RuntimeError, match="status 7"):
            run_worker_process(job, config, ["0"])
    else:
        run_worker_process(job, config, ["0"])
    assert not live_process_group(int(job.read_text()))

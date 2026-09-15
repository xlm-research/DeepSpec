"""Real Gloo/NCCL check of a rank-local cache error after a successful batch."""

import argparse
from contextlib import ExitStack
from datetime import timedelta
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from deepspec.trainer.glm5_partitioned_swap import atomic_write_json, load_json
from tests.test_qwen38_vllm import convert, partition_trainer, teacher_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    compute_group = (
        dist.new_group(backend="nccl", timeout=timedelta(seconds=20))
        if device.type == "cuda" else dist.group.WORLD
    )
    trainer = partition_trainer(
        args.root, counts=(2,), rank=rank, world_size=dist.get_world_size()
    )
    trainer.device = device
    trainer._vllm_control_group = dist.group.WORLD
    tensors, batch = teacher_sample()
    batches = [
        dict(batch, attention_mask=torch.ones_like(batch["input_ids"]))
        for _ in range(2)
    ]

    def fake_worker(job_path, config, devices):
        job = load_json(job_path)
        for request in job["requests"]:
            path = Path(request["output_paths"][0])
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(convert(tensors, batch), path)
        atomic_write_json(
            str(job_path) + ".complete",
            {"teacher": trainer._teacher_identity, "samples": [{}, {}]},
        )

    real_load = torch.load

    def load(path, *args, **kwargs):
        if rank == 1 and Path(path).name == "sample_00000001.pt":
            raise FileNotFoundError(f"Injected consumer-local missing file: {path}")
        return real_load(path, *args, **kwargs)

    iterator = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch(
                "deepspec.trainer.qwen3_8_vllm_trainer.run_worker_process",
                side_effect=fake_worker,
            ))
            if device.type == "cpu":
                stack.enter_context(patch("torch.cuda.synchronize"))
                stack.enter_context(patch("torch.cuda.empty_cache"))
            iterator = trainer.iter_training_batches(batches)
            features = next(iterator)
            parameter = torch.nn.Parameter(torch.ones((), device=device))
            (parameter * features["target_hidden_states"].float().sum()).backward()
            dist.all_reduce(parameter.grad, group=compute_group)
            # Rank 1 loses access only after all ranks passed preflight and a
            # real first-batch backward/collective. Rank 0 can still read it.
            with patch("torch.load", side_effect=load):
                try:
                    next(iterator)
                except RuntimeError as exc:
                    assert "rank 1: FileNotFoundError" in str(exc), str(exc)
                    assert "sample_00000001.pt" in str(exc), str(exc)
                else:
                    raise AssertionError(
                        "A rank yielded a batch after a peer's read failure"
                    )
            print(f"CACHE_FAILURE_PROPAGATED rank={rank} device={device}", flush=True)
    finally:
        if iterator is not None:
            iterator.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Time the real partition training loop with CUDA completion on every rank.

Use TRAIN_ENTRYPOINT=scripts/benchmark_glm5_partition_step.py with the GLM
launcher. Results are written below DEEPSPEC_OUTPUT_ROOT/benchmark. The first
step includes any compilation triggered by the measured training shapes.
"""

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_timers():
    import torch
    import torch.distributed as dist

    from deepspec.trainer.base_trainer import BaseTrainer
    from deepspec.trainer.dspark_trainer import Glm5NextDSparkTrainer

    destination = Path(os.environ["DEEPSPEC_OUTPUT_ROOT"]) / "benchmark"
    destination.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    phases = []
    original_phase = Glm5NextDSparkTrainer._set_swap_phase
    original_batches = BaseTrainer.iter_training_batches
    original_cleanup = Glm5NextDSparkTrainer.clean_up

    def record_phase(trainer, phase):
        now = time.perf_counter()
        if phases:
            phases[-1]["seconds"] = now - phases[-1]["start_monotonic"]
        phases.append(
            {"phase": phase, "start_monotonic": now, "start_unix": time.time()}
        )
        (destination / f"phases_rank_{rank}.json").write_text(
            json.dumps(phases, indent=2) + "\n"
        )
        return original_phase(trainer, phase)

    def timed_batches(trainer, batches):
        if trainer.gradient_accumulation_steps != 1:
            raise ValueError(
                "This benchmark requires one micro-batch per optimizer step."
            )
        for batch in original_batches(trainer, batches):
            expected = int(trainer.args.data.max_length)
            if tuple(batch["input_ids"].shape) != (1, expected):
                raise ValueError(
                    "Benchmark inputs must use the full configured context."
                )
            record = {
                "rank": rank,
                "tokens": batch["input_ids"].numel(),
                "supervised_tokens": int(batch["loss_mask"].sum()),
                "num_anchors": int(trainer.args.model.num_anchors),
                "block_size": int(trainer.args.model.block_size),
                "global_batch_size": int(trainer.args.train.global_batch_size),
                "target_hidden_shape": list(batch["target_hidden_states"].shape),
                "target_last_hidden_shape": list(
                    batch["target_last_hidden_states"].shape
                ),
            }
            torch.cuda.synchronize(trainer.device)
            dist.barrier(group=trainer._partition_control_group)
            torch.cuda.reset_peak_memory_stats(trainer.device)
            started = time.perf_counter()
            yield batch
            torch.cuda.synchronize(trainer.device)
            record.update(
                seconds=time.perf_counter() - started,
                global_step=trainer.global_step,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(trainer.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(trainer.device),
            )
            records = [None] * trainer.world_size
            dist.all_gather_object(
                records, record, group=trainer._partition_control_group
            )
            if rank == 0:
                result = {
                    "global_step": trainer.global_step,
                    "max_rank_seconds": max(item["seconds"] for item in records),
                    "global_context_tokens": sum(item["tokens"] for item in records),
                    "includes": "forward, loss, backward, gradient synchronization/clipping, optimizer, metrics; first-step compilation",
                    "excludes": "target generation, model loading, cache read/H2D, checkpoint writing",
                    "ranks": records,
                }
                (destination / f"step_{trainer.global_step}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print("[deepspec-step-benchmark] " + json.dumps(result), flush=True)

    def clean_up(trainer):
        try:
            return original_cleanup(trainer)
        finally:
            if phases:
                phases[-1]["seconds"] = (
                    time.perf_counter() - phases[-1]["start_monotonic"]
                )
                (destination / f"phases_rank_{rank}.json").write_text(
                    json.dumps(phases, indent=2) + "\n"
                )

    Glm5NextDSparkTrainer._set_swap_phase = record_phase
    BaseTrainer.iter_training_batches = timed_batches
    Glm5NextDSparkTrainer.clean_up = clean_up


if __name__ == "__main__":
    import train

    install_timers()
    train.main(int(os.environ["LOCAL_RANK"]))

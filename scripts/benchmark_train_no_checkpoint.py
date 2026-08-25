"""Run the normal trainer while disabling checkpoint writes for benchmarks."""

import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

# Executing a file below scripts/ otherwise resolves ``train`` to the
# scripts/train namespace package instead of the repository entry point.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import parse_args
from deepspec.utils import CustomJSONEncoder, seed_all


os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def main(local_rank: int) -> None:
    args = parse_args()
    seed_all(int(args.seed))
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)

    trainer = args.train.trainer_cls(local_rank, args)

    # BaseTrainer.train() performs both periodic and unconditional final saves
    # through this method. Replacing it on the benchmark instance keeps the
    # production trainer and launch script unchanged and writes no weights.
    trainer.save_and_eval_checkpoint = lambda: None

    dist.barrier()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started_at

    elapsed = torch.tensor(elapsed_seconds, dtype=torch.float64, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        steps = int(trainer.max_train_steps)
        result = {
            "optimizer_steps": steps,
            "training_seconds": elapsed.item(),
            "seconds_per_step": elapsed.item() / steps,
            "checkpoint_writes": 0,
        }
        print("BENCHMARK_RESULT=" + json.dumps(result), flush=True)

    trainer.clean_up()


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch this benchmark with torchrun.")
    main(int(os.environ["LOCAL_RANK"]))

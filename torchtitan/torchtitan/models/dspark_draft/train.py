"""Run the configured DSpark phase with TorchTitan's own Trainer loop."""

import json
import os
from pathlib import Path
import sys
import traceback

import torch.distributed as dist

from torchtitan.config.manager import ConfigManager
from torchtitan.tools.logging import init_logger

from .trainer import DSparkTrainer


def run(config):
    if not isinstance(config, DSparkTrainer.Config):
        raise ValueError("A DSpark Trainer recipe is required")
    trainer = None
    succeeded = False
    try:
        trainer = config.build()
        trainer.train()
        if trainer.completed_updates != trainer.phase_stop_update:
            raise RuntimeError("Draft training stopped before the requested update")
        result = {
            "completed_updates": trainer.step,
            "consumed_microbatches": trainer.dataloader.cursor,
        }
        if config.checkpoint.enable:
            commit = trainer.checkpointer.last_commit
            if commit is None or commit["completed_updates"] != trainer.step:
                raise RuntimeError("Draft phase has no committed complete checkpoint")
            result.update(
                {
                    "commit": commit,
                    "next_global_microbatch": trainer.dataloader.next_global_microbatch,
                    "consumed_range": [
                        trainer.dataloader.global_microbatch_start,
                        trainer.dataloader.next_global_microbatch,
                    ],
                }
            )
            workers = [None] * dist.get_world_size()
            dist.all_gather_object(workers, os.getpid())
            result["worker_pids"] = workers
            if trainer.checkpointer.exports:
                result["hf_exports"] = trainer.checkpointer.exports
        with trainer.phase_timing.measure("close"):
            trainer.close()
        timing = trainer.phase_timing.report()
        if timing is not None:
            result["timing"] = timing
        if dist.get_rank() == 0 and os.environ.get("DEEPSPEC_PHASE_RESULT"):
            path = Path(os.environ["DEEPSPEC_PHASE_RESULT"])
            temporary = path.with_suffix(".incomplete")
            temporary.write_text(json.dumps(result) + "\n")
            temporary.replace(path)
        succeeded = True
        return result
    finally:
        # A failed rank must report failure to Elastic before peers can block
        # forever in collectives. The process supervisor releases those peers.
        if succeeded:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    init_logger()
    try:
        run(ConfigManager().parse_args())
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)

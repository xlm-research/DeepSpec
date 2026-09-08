import argparse
import faulthandler
import json
import os
import socket
import sys
import traceback
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from deepspec.utils import (
    CustomJSONEncoder,
    load_config,
    parse_opts_to_config,
    seed_all,
)

os.environ['USE_TORCH']='true'
os.environ['WANDB_DISABLED']='true'
os.environ['TOKENIZERS_PARALLELISM']='false'
torch.set_float32_matmul_precision("high")
faulthandler.enable(all_threads=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    args = parser.parse_args()
    config = parse_opts_to_config(args.opts, load_config(args.config))
    config._origin_config_path = os.path.abspath(args.config)
    config._origin_opts = list(args.opts)
    return config


@record
def main(local_rank):
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(torch.cuda.device_count())
    torch.cuda.set_device(local_rank)

    args = parse_args()
    seed_all(int(args.seed))
    global_rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(
        "[deepspec-process] "
        f"host={socket.gethostname()} pid={os.getpid()} "
        f"global_rank={global_rank}/{world_size} local_rank={local_rank} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    if global_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)

    trainer = None
    completed = False
    try:
        trainer = args.train.trainer_cls(local_rank, args)
        trainer.train()
        completed = True
    except BaseException as exc:
        print(
            "[deepspec-fatal] "
            f"host={socket.gethostname()} pid={os.getpid()} "
            f"global_rank={global_rank} local_rank={local_rank} "
            f"exception={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                print(
                    "[deepspec-cuda] "
                    f"device={device} allocated={torch.cuda.memory_allocated(device)} "
                    f"reserved={torch.cuda.memory_reserved(device)} "
                    f"max_allocated={torch.cuda.max_memory_allocated(device)} "
                    f"max_reserved={torch.cuda.max_memory_reserved(device)}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as memory_exc:
                print(
                    f"[deepspec-cuda] unable to read memory stats: {memory_exc}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        active_failure = sys.exc_info()[0] is not None
        try:
            if trainer is not None:
                trainer.clean_up(synchronize=completed)
        except BaseException as cleanup_exc:
            print(
                "[deepspec-cleanup-failure] "
                f"global_rank={global_rank} local_rank={local_rank} "
                f"exception={type(cleanup_exc).__name__}: {cleanup_exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            if not active_failure:
                raise
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        # Standard torchrun contract: one Python process per worker.
        main(int(os.environ["LOCAL_RANK"]))
    else:
        # Backward-compatible single-node entry. New launches should use
        # torchrun so rank/world-size semantics also work across nodes.
        torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())

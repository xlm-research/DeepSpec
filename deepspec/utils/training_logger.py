import time
import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from deepspec.utils import ensure_dir, is_global_main_process, print_on_global_main
from deepspec.utils.metrics import add_metric, flush_async, reset


_writer: Optional[SummaryWriter] = None
_wandb_run = None
_logging_steps: int = 1
_session_start_wall: Optional[float] = None
_session_start_step: int = 0


def init(*, logging_steps: int, tensorboard_dir: Optional[str] = None) -> None:
    global _writer, _wandb_run, _logging_steps
    _logging_steps = int(logging_steps)
    if not is_global_main_process():
        return
    if tensorboard_dir is not None:
        ensure_dir(tensorboard_dir)
        _writer = SummaryWriter(tensorboard_dir)
    wandb_enabled = os.environ.get("WANDB_ENABLE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if wandb_enabled:
        try:
            import wandb

            wandb_dir = os.environ.get("WANDB_DIR")
            if wandb_dir:
                ensure_dir(wandb_dir)
            _wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "deepspec"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=os.environ.get("WANDB_NAME") or None,
                group=os.environ.get("WANDB_GROUP") or None,
                job_type=os.environ.get("WANDB_JOB_TYPE", "train"),
                dir=wandb_dir,
                resume=os.environ.get("WANDB_RESUME", "allow"),
                config={
                    "tensorboard_dir": tensorboard_dir,
                    "logging_steps": _logging_steps,
                    "root_dir": os.environ.get("ROOT_DIR"),
                    "source_jsonl_path": os.environ.get("SOURCE_JSONL_PATH"),
                    "target_model_path": os.environ.get("TARGET_MODEL_PATH"),
                    "max_length": os.environ.get("MAX_LENGTH"),
                    "context_parallel_size": os.environ.get("CONTEXT_PARALLEL_SIZE"),
                    "fsdp_size": os.environ.get("FSDP_SIZE"),
                    "global_batch_size": os.environ.get("GLOBAL_BATCH_SIZE"),
                    "data_batch_size": os.environ.get("DATA_BATCH_SIZE"),
                    "online_target": os.environ.get("ONLINE_TARGET"),
                },
            )
            print_on_global_main(f"W&B logging enabled: {_wandb_run.url}")
        except Exception as exc:
            _wandb_run = None
            print_on_global_main(f"W&B logging disabled after init failure: {exc}")


def start_session(*, global_step: int) -> None:
    global _session_start_wall, _session_start_step
    reset()
    _session_start_wall = time.time()
    _session_start_step = int(global_step)


def begin_optimizer_step(
    *,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
    learning_rate: float,
    grad_norm,
):
    add_metric("lr", learning_rate, reduction="last", tag="train")
    add_metric("grad_norm", grad_norm, reduction="last", tag="train")

    if global_step % _logging_steps != 0:
        return None

    return (
        flush_async(),
        dict(
            global_step=global_step,
            next_micro_step=next_micro_step,
            micro_batches_per_epoch=micro_batches_per_epoch,
            max_train_steps=max_train_steps,
        ),
    )


def finish_optimizer_step(pending):
    if pending is None:
        return None
    reduction, metadata = pending
    global_main = is_global_main_process()
    # Every rank waits for collective completion, but only rank 0 needs to
    # synchronize reduced CUDA scalars back to the CPU for printing/writing.
    summary = reduction.wait(materialize=global_main)
    if global_main:
        _write_scalars(summary, global_step=metadata["global_step"])
        _print_summary(
            summary=summary,
            **metadata,
        )
    return summary


def on_optimizer_step(**kwargs):
    """Synchronous compatibility wrapper for external callers."""

    return finish_optimizer_step(begin_optimizer_step(**kwargs))


def close() -> None:
    global _writer, _wandb_run
    if _writer is not None:
        _writer.close()
        _writer = None
    if _wandb_run is not None:
        try:
            _wandb_run.finish()
        finally:
            _wandb_run = None


def _write_scalars(summary, *, global_step: int) -> None:
    if _writer is not None:
        for key, value in summary.items():
            _writer.add_scalar(key, value, global_step)
    if _wandb_run is not None:
        _wandb_run.log(dict(summary), step=global_step)


def _print_summary(
    *,
    summary,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
) -> None:
    session_start_wall = _session_start_wall
    if session_start_wall is None:
        session_start_wall = time.time()
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    session_elapsed = time.time() - session_start_wall
    completed_session_steps = global_step - _session_start_step
    remaining_steps = max(max_train_steps - global_step, 0)
    remaining_min = (
        session_elapsed * remaining_steps / max(completed_session_steps, 1)
    ) / 60
    loss_text = ""
    # DFlash2 emits a correctly token-normalized cross-rank objective. Prefer
    # it over the legacy rank-local DSpark scalar when both are present. Runs
    # without DFlash2 retain the existing logging path unchanged.
    loss_key = (
        "train/dflash2_loss"
        if "train/dflash2_loss" in summary
        else "train/loss"
    )
    if loss_key in summary:
        loss_text = f" loss={summary[loss_key]:.4f}"
    print_on_global_main(
        f"epoch={current_epoch} "
        f"step={global_step}/{max_train_steps}"
        f"{loss_text} "
        f"| elapsed={session_elapsed / 60:.1f}min"
        f" | remaining={remaining_min:.1f}min"
    )

import json
import os
import time
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from deepspec.utils import (
    CustomJSONEncoder,
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
)
from deepspec.utils.metrics import add_metric, flush_async, reset


_writer: Optional[SummaryWriter] = None
_wandb_run = None
_logging_steps: int = 1
_session_start_wall: Optional[float] = None
_session_start_step: int = 0


def init(
    *,
    logging_steps: int,
    tensorboard_dir: Optional[str] = None,
    project_name: str = "deepspec",
    exp_name: Optional[str] = None,
    config=None,
) -> None:
    global _writer, _wandb_run, _logging_steps
    close()
    _logging_steps = int(logging_steps)
    if not is_global_main_process():
        return
    if tensorboard_dir is not None:
        ensure_dir(tensorboard_dir)
        _writer = SummaryWriter(tensorboard_dir)
    if os.environ.get("WANDB_MODE") in ("online", "offline"):
        # Other ranks and runs with W&B disabled need no SDK or credentials.
        import wandb

        log_dir = os.environ.get("WANDB_DIR") or tensorboard_dir or "."
        ensure_dir(log_dir)
        _wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", project_name),
            name=os.environ.get(
                "WANDB_NAME", os.environ.get("WANDB_RUN_NAME", exp_name)
            ),
            entity=os.environ.get("WANDB_ENTITY", os.environ.get("WANDB_TEAM")),
            dir=log_dir,
            config=json.loads(json.dumps(config, cls=CustomJSONEncoder)),
        )
        _wandb_run.define_metric("global_step")
        _wandb_run.define_metric("*", step_metric="global_step")
        if os.environ.get("WANDB_MODE") == "online" and _wandb_run.url:
            print_on_global_main(f"W&B loss curves: {_wandb_run.url}")


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
    writer, run = _writer, _wandb_run
    _writer = _wandb_run = None
    try:
        if writer is not None:
            writer.close()
    finally:
        if run is not None:
            run.finish()


def _write_scalars(summary, *, global_step: int) -> None:
    if _writer is not None:
        for key, value in summary.items():
            _writer.add_scalar(key, value, global_step)
    if _wandb_run is not None:
        _wandb_run.log(
            {**summary, "global_step": global_step}, step=global_step, commit=True
        )


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

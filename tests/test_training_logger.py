"""W&B receives reduced metrics once per optimizer logging interval."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from deepspec.utils import training_logger


@pytest.fixture(autouse=True)
def logger_state(monkeypatch):
    training_logger.close()
    for key in list(training_logger.os.environ):
        if key.startswith("WANDB_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(training_logger, "is_global_main_process", lambda: True)
    monkeypatch.setattr(training_logger, "_print_summary", Mock())
    monkeypatch.setattr(training_logger, "print_on_global_main", Mock())
    yield
    training_logger.close()


@pytest.mark.parametrize("mode", ["online", "offline"])
@pytest.mark.parametrize("primary", [True, False])
def test_only_primary_rank_logs_reduced_metrics(monkeypatch, mode, primary, tmp_path):
    monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setattr(training_logger, "is_global_main_process", lambda: primary)
    run = Mock(url=None)
    sdk = SimpleNamespace(init=Mock(return_value=run))
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    training_logger.init(
        logging_steps=1,
        project_name="deepspec",
        exp_name="dspark-test",
        config={"dtype": torch.bfloat16, "path": Path("checkpoints")},
    )
    summary = {
        "train/loss": 1.5,
        "train/ce_loss": 2.0,
        "train/l1_loss": 1.0,
        "train/confidence_loss": 0.4,
        "train/lr": 6e-4,
    }
    reduction = Mock()
    reduction.wait.return_value = summary if primary else None
    training_logger.finish_optimizer_step((reduction, {"global_step": 12}))
    reduction.wait.assert_called_once_with(materialize=primary)
    if primary:
        assert sdk.init.call_args.kwargs["project"] == "deepspec"
        assert sdk.init.call_args.kwargs["name"] == "dspark-test"
        assert sdk.init.call_args.kwargs["config"] == {
            "dtype": "torch.bfloat16",
            "path": "checkpoints",
        }
        run.log.assert_called_once_with(
            {**summary, "global_step": 12}, step=12, commit=True
        )
    else:
        sdk.init.assert_not_called()
        run.log.assert_not_called()
    training_logger.close()
    training_logger.close()
    assert run.finish.call_count == int(primary)


@pytest.mark.parametrize("mode", [None, "disabled"])
def test_disabled_logging_needs_no_wandb_sdk(monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setitem(sys.modules, "wandb", None)
    training_logger.init(logging_steps=1)
    assert training_logger._wandb_run is None


def test_logging_frequency_uses_optimizer_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    run = Mock(url=None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kw: run))
    monkeypatch.setattr(training_logger, "add_metric", Mock())
    reduction = Mock()
    reduction.wait.return_value = {"train/loss": 0.75}
    flush = Mock(return_value=reduction)
    monkeypatch.setattr(training_logger, "flush_async", flush)
    training_logger.init(logging_steps=2)
    metadata = dict(
        next_micro_step=16,
        micro_batches_per_epoch=32,
        max_train_steps=10,
        learning_rate=6e-4,
        grad_norm=0.2,
    )
    training_logger.on_optimizer_step(global_step=3, **metadata)
    run.log.assert_not_called()
    training_logger.on_optimizer_step(global_step=4, **metadata)
    flush.assert_called_once()
    run.log.assert_called_once_with(
        {"train/loss": 0.75, "global_step": 4}, step=4, commit=True
    )


def test_close_finishes_wandb_if_tensorboard_close_fails(monkeypatch):
    writer = Mock()
    writer.close.side_effect = OSError("disk unavailable")
    run = Mock()
    monkeypatch.setattr(training_logger, "_writer", writer)
    monkeypatch.setattr(training_logger, "_wandb_run", run)
    with pytest.raises(OSError, match="disk unavailable"):
        training_logger.close()
    run.finish.assert_called_once()
    training_logger.close()
    run.finish.assert_called_once()

"""Startup failures must reach the driver before queued collective exports."""

from argparse import Namespace
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
import torch

from torchspec import train_entry


def _ready(value=None, error=None):
    ref = Future()
    if error is None:
        ref.set_result(value)
    else:
        ref.set_exception(error)
    return ref


def _startup(monkeypatch, tmp_path, *, train_with_decode=True, init_error=None):
    args = Namespace(
        train_with_decode=train_with_decode,
        inference_engine_type="vllm",
        dp_size=2,
        training_num_nodes=1,
        training_num_gpus_per_node=2,
        decode_speculative_algorithm="EAGLE3" if train_with_decode else None,
        decode_speculative_draft_model_path=None,
        output_dir=str(tmp_path),
    )
    controller = MagicMock()
    controller.load_dataset.remote.return_value = _ready(4)
    controller.load_eval_dataset.remote.return_value = _ready(0)
    controller_class = MagicMock()
    controller_class.options.return_value.remote.return_value = controller
    monkeypatch.setattr(train_entry, "AsyncTrainingController", controller_class)
    runtime_context = MagicMock()
    runtime_context.get_node_id.return_value = "01" * 28
    monkeypatch.setattr(train_entry.ray, "get_runtime_context", lambda: runtime_context)

    events = []
    refs = [_ready(0, error=init_error), _ready(0)]

    def get(refs_to_wait):
        if refs_to_wait == refs:
            events.append("wait_train")
        return [ref.result(timeout=0) for ref in refs_to_wait]

    monkeypatch.setattr(train_entry.ray, "get", get)
    group = MagicMock()
    group.async_init.return_value = refs

    def save(_path):
        events.append("save")
        if init_error is not None:
            # Ray runs a queued method even after an earlier method failed.
            # Its peers are still in init collectives, so export times out.
            raise TimeoutError("peer still waiting in TargetLMHead all_reduce")

    group.save_draft_model_for_serving.side_effect = save
    monkeypatch.setattr(train_entry, "allocate_train_group", lambda **kwargs: group)
    monkeypatch.setattr(train_entry, "init_tracking", lambda _args: None)
    monkeypatch.setattr(
        train_entry, "_get_draft_model_config", lambda _args: Namespace(vocab_size=16)
    )
    monkeypatch.setattr(
        train_entry, "create_placement_groups", lambda *a, **kw: {"training": (), "inference": ()}
    )
    monkeypatch.setattr(train_entry, "launch_mooncake_master", lambda _args: None)
    monkeypatch.setattr(train_entry, "build_mooncake_config", lambda _args: None)
    monkeypatch.setattr(train_entry, "auto_calculate_training_steps", lambda *a: None)

    def prepare(*_args):
        events.append("prepare_inference")
        return [object()], [_ready(None)]

    monkeypatch.setattr(train_entry, "prepare_inference_engines", prepare)
    monkeypatch.setattr(
        train_entry, "setup_async_training_with_engines", lambda *a, **kw: (controller, object())
    )
    monkeypatch.setattr(train_entry, "run_training_loop", lambda *a, **kw: events.append("train"))
    return args, group, events


@pytest.mark.parametrize("algorithm", ["EAGLE3", "dspark"])
def test_decode_surfaces_trainer_oom_before_collective_export(monkeypatch, tmp_path, algorithm):
    error = torch.OutOfMemoryError("rank 0 optimizer allocation failed")
    args, group, events = _startup(monkeypatch, tmp_path, init_error=error)
    args.decode_speculative_algorithm = algorithm

    with pytest.raises(torch.OutOfMemoryError, match="rank 0 optimizer"):
        train_entry.train_async_no_generation(args)

    group.save_draft_model_for_serving.assert_not_called()
    assert "prepare_inference" not in events


@pytest.mark.parametrize("algorithm", ["EAGLE3", "dspark"])
def test_decode_waits_for_trainers_before_export_and_inference(monkeypatch, tmp_path, algorithm):
    args, group, events = _startup(monkeypatch, tmp_path)
    args.decode_speculative_algorithm = algorithm

    train_entry.train_async_no_generation(args)

    group.save_draft_model_for_serving.assert_called_once()
    assert events.index("wait_train") < events.index("save") < events.index("prepare_inference")
    assert events[-1] == "train"


def test_target_only_decode_never_exports_a_serving_draft(monkeypatch, tmp_path):
    args, group, events = _startup(monkeypatch, tmp_path)
    args.decode_speculative_algorithm = None
    args.decode_target_only = True

    train_entry.train_async_no_generation(args)

    group.save_draft_model_for_serving.assert_not_called()
    assert "prepare_inference" in events
    assert events[0] == "prepare_inference"
    assert events[-1] == "train"


@pytest.mark.parametrize("algorithm", ["EAGLE3", "dspark"])
def test_existing_draft_keeps_training_and_inference_initialization_parallel(
    monkeypatch, tmp_path, algorithm
):
    args, group, events = _startup(monkeypatch, tmp_path)
    args.decode_speculative_algorithm = algorithm
    args.decode_speculative_draft_model_path = "/existing/draft"

    train_entry.train_async_no_generation(args)

    group.save_draft_model_for_serving.assert_not_called()
    assert args.decode_speculative_draft_model_path == "/existing/draft"
    assert events[0] == "prepare_inference"
    assert events[-1] == "train"


def test_prefill_still_initializes_inference_before_waiting_for_trainers(monkeypatch, tmp_path):
    args, group, events = _startup(monkeypatch, tmp_path, train_with_decode=False)

    train_entry.train_async_no_generation(args)

    group.save_draft_model_for_serving.assert_not_called()
    assert events[0] == "prepare_inference"
    assert events[-1] == "train"

"""Complete-pass behavior at the controller's public queue boundary."""

import asyncio
import json
import os
import queue
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torchspec.controller import training_controller as controller_module
from torchspec.utils.types import InferenceOutput


def full_controller(tmp_path, monkeypatch, count=17, *, submit=True):
    monkeypatch.setattr(controller_module, "Queue", queue.Queue)
    windows = [
        {
            "source_id": f"source_{i}",
            "source_index": i,
            "window_id": f"window_{i}",
            "supervised_message_indices": [1],
            "supervised_assistant_ordinals": [0],
            "message_indices": [0, 1],
            "assistant_count": 1,
            "expanded_tokens": 16,
            "supervised_tokens": 14,
            "image_count": 0,
            "context_reductions": [],
        }
        for i in range(count)
    ]
    manifest = {
        "source_count": count,
        "window_count": count,
        "windows": windows,
        "sources": [
            {
                "source_id": f"source_{i}",
                "source_index": i,
                "assistant_message_indices": [1],
                "window_ids": [f"window_{i}"],
            }
            for i in range(count)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(
        complete_data_pass=True,
        supervision_manifest_path=str(path),
        per_dp_rank_batch_size=1,
        draft_accumulation_steps=4,
        global_batch_size=16,
        dynamic_loss_mask=True,
        output_dir=str(tmp_path),
        ttt_length=7,
        distributed_timeout_minutes=1,
    )
    cls = controller_module.AsyncTrainingController.__ray_metadata__.modified_class
    samples = [
        {
            "data_id": window["window_id"],
            "input_ids": torch.arange(16),
            "metadata": {"supervision_window": window},
        }
        for window in windows
    ]
    if submit:
        controller = cls(args, dp_size=4)
        controller.add_dataset(samples)
    else:
        model = os.environ.get("TARGET_MODEL_PATH")
        if not model or not (Path(model) / "chat_template.jinja").exists():
            pytest.skip("Set TARGET_MODEL_PATH for the real full-data workflow fixture")
        source = tmp_path / "source.jsonl"
        source.write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"Question for training source {i}."},
                            {
                                "role": "assistant",
                                "content": "A complete answer preserves source coverage.",
                                "reasoning_content": "Track every training source and preserve every target.",
                            },
                        ]
                    }
                )
                + "\n"
                for i in range(count)
            )
        )
        prepared = tmp_path / "prepared"
        script = Path(__file__).parents[1] / "examples/qwen3.8-27b-image-text-1node/prepare_data.py"
        runpy.run_path(str(script))["prepare"](
            SimpleNamespace(
                source=str(source),
                output=str(prepared),
                media_root=str(tmp_path),
                model=model,
                full_data=True,
                max_length=4096,
                min_loss_tokens=14,
                expected_sources=count,
            )
        )
        args.supervision_manifest_path = str(prepared / "manifest.json")
        args.train_data_path = str(prepared / "train.jsonl")
        args.max_seq_length = 4096
        args.target_model_path = model
        args.prompt_key = "messages"
        args.renderer = "qwen3.5-full-data"
        args.num_proc = 1
        args.min_loss_tokens = 14
        args.last_turn_loss_only = False
        args.cache_dir = str(tmp_path / "cache")
        args.inference_engine_type = "vllm"
        args.train_with_decode = False
        args.defer_tokenization = False
        args.shuffle_dataset = False
        controller = cls(args, dp_size=4)
        controller.load_dataset(args)
    return controller


def output(data_id):
    return InferenceOutput(
        data_id=data_id,
        mooncake_key=f"capture_{data_id}",
        tensor_shapes={"input_ids": (16,), "hidden_states": (16, 8), "last_hidden_states": (16, 4)},
        tensor_dtypes={
            "input_ids": torch.int64,
            "hidden_states": torch.bfloat16,
            "last_hidden_states": torch.bfloat16,
        },
        packed_loss_mask="2,14",
    )


def consume_queues(controller):
    samples = []
    for rank_queue in controller.get_train_queues():
        while not rank_queue.empty():
            samples.append(rank_queue.get_nowait())
    return samples


def test_complete_pass_waits_for_inflight_and_commits_all_real_tail_windows(tmp_path, monkeypatch):
    controller = full_controller(tmp_path, monkeypatch)
    prompts = controller.get_prompts(17)
    controller.push_inference_results([output(prompt.data_id) for prompt in prompts[:-1]])
    for _ in range(4):
        assert controller.try_dispatch_batch()
    first = consume_queues(controller)
    assert {sample.data_id for sample in first} == {f"window_{i}" for i in range(16)}
    controller.commit_training_step(step=1, num_batches=4)
    assert not controller.try_dispatch_batch()
    assert controller.get_coverage_report()["in_flight_window_ids"] == ["window_16"]

    controller.push_inference_results([output("window_16")])
    for _ in range(4):
        assert controller.try_dispatch_batch()
    tail = consume_queues(controller)
    real = [sample for sample in tail if not (sample.metadata or {}).get("zero_loss_dummy")]
    padding = [sample for sample in tail if (sample.metadata or {}).get("zero_loss_dummy")]
    assert [sample.data_id for sample in real] == ["window_16"]
    assert len(padding) == 15
    controller.commit_training_step(step=2, num_batches=4)
    report = controller.get_coverage_report()
    assert report["committed_window_count"] == 17
    assert report["committed_source_count"] == 17
    assert report["committed_segment_count"] == 17
    assert report["optimizer_steps"] == 2
    assert report["missing_window_ids"] == []
    assert report["duplicate_window_ids"] == []
    assert not controller.try_dispatch_batch()


def test_padding_only_rank_produces_finite_zero_loss_without_reading_mooncake(
    tmp_path, monkeypatch
):
    from torchspec.training.data_fetcher import MooncakeDataset

    controller = full_controller(tmp_path, monkeypatch, count=1)
    prompt = controller.get_prompts(1)[0]
    controller.push_inference_results([output(prompt.data_id)])
    for _ in range(4):
        assert controller.try_dispatch_batch()
    rank_queue = controller.get_train_queues()[3]
    rank_queue.put(None)

    class UnusedStore:
        def get(self, *args, **kwargs):
            raise AssertionError("Padding cannot fetch a real capture")

        def remove_eagle3_tensors(self, *args, **kwargs):
            raise AssertionError("Padding cannot delete a real capture")

    batches = list(
        MooncakeDataset(
            rank_queue,
            UnusedStore(),
            device=torch.device("cpu"),
            min_loss_tokens=14,
        )
    )
    assert len(batches) == 4
    for batch in batches:
        assert not batch["loss_mask"].any()
        assert batch["hidden_states"].shape == (1, 16, 8)
        assert batch["last_hidden_states"].shape == (1, 16, 4)
        assert torch.isfinite(batch["hidden_states"]).all()
    assert controller.get_speeds()["training_total"] == 1


def test_full_data_inference_failure_reports_window_and_stops_without_reloading(
    tmp_path, monkeypatch
):
    from torchspec.controller import inference_manager as inference_module
    from torchspec.controller.inference_manager import AsyncInferenceManager

    controller = full_controller(tmp_path, monkeypatch, count=1)
    monkeypatch.setattr(inference_module.ray, "put", lambda value: value)

    class ControllerActor:
        def __getattr__(self, name):
            async def call(*args, **kwargs):
                return getattr(controller, name)(*args, **kwargs)

            return SimpleNamespace(remote=call)

    async def invalid_capture(**kwargs):
        return [None]

    args = SimpleNamespace(
        complete_data_pass=True,
        inference_engine_type="vllm",
        inference_batch_size=1,
        inference_buffer_threshold=1,
        inference_fetch_batch=1,
        train_with_decode=False,
        defer_tokenization=False,
        enable_perf_metrics=False,
    )
    cls = AsyncInferenceManager.__ray_metadata__.modified_class
    manager = cls(
        args,
        ControllerActor(),
        inference_engines=[SimpleNamespace(generate=SimpleNamespace(remote=invalid_capture))],
    )

    async def run():
        with pytest.raises(RuntimeError, match="window_0"):
            await asyncio.wait_for(manager.run(), timeout=1)

    asyncio.run(run())
    report = controller.get_coverage_report()
    assert report["status"] == "incomplete"
    assert "window_0" in report["failed_windows"]
    assert report["committed_window_count"] == 0


@pytest.mark.parametrize("window_count, expected_steps", [(1, 1), (16, 1), (17, 2), (32, 2)])
def test_complete_pass_derives_optimizer_and_scheduler_endpoint_from_all_windows(
    window_count, expected_steps
):
    from torchspec.controller.setup import auto_calculate_training_steps

    args = SimpleNamespace(
        complete_data_pass=True,
        global_batch_size=16,
        per_dp_rank_batch_size=1,
        draft_accumulation_steps=4,
        num_epochs=1,
        num_train_steps=None,
        lr_total_steps=None,
    )
    auto_calculate_training_steps(args, window_count)
    assert args.steps_per_epoch == expected_steps
    assert args.num_train_steps == expected_steps
    assert args.lr_total_steps == expected_steps
    assert args.num_epochs == 1


def test_duplicate_capture_fails_before_extra_supervision_can_be_committed(tmp_path, monkeypatch):
    controller = full_controller(tmp_path, monkeypatch, count=1)
    prompt = controller.get_prompts(1)[0]
    controller.push_inference_results([output(prompt.data_id)])
    with pytest.raises(RuntimeError, match="Duplicate"):
        controller.push_inference_results([output(prompt.data_id)])
    report = controller.get_coverage_report()
    assert report["status"] == "incomplete"
    assert report["duplicate_window_ids"] == ["window_0"]
    assert report["committed_window_count"] == 0


@pytest.mark.parametrize("key,value", [("num_train_steps", 10), ("lr_total_steps", 11)])
def test_complete_pass_rejects_incompatible_smoke_horizon(key, value):
    from torchspec.controller.setup import auto_calculate_training_steps

    args = SimpleNamespace(
        complete_data_pass=True,
        global_batch_size=16,
        per_dp_rank_batch_size=1,
        draft_accumulation_steps=4,
        num_epochs=1,
        num_train_steps=None,
        lr_total_steps=None,
    )
    setattr(args, key, value)
    with pytest.raises(ValueError, match="complete-pass endpoint 2"):
        auto_calculate_training_steps(args, 17)


@pytest.mark.parametrize("delayed_metrics", [False, True])
def test_training_workflow_waits_for_delayed_tail_and_saves_complete_coverage(
    tmp_path, monkeypatch, delayed_metrics
):
    from torchspec.controller import loop

    controller = full_controller(tmp_path, monkeypatch, submit=False)
    clock = SimpleNamespace(now=0.0)
    pending = []

    def capture(prompt):
        length = len(prompt.input_ids)
        return InferenceOutput(
            data_id=prompt.data_id,
            mooncake_key=f"capture_{prompt.data_id}",
            tensor_shapes={
                "input_ids": (length,),
                "hidden_states": (length, 8),
                "last_hidden_states": (length, 4),
            },
            tensor_dtypes={
                "input_ids": torch.int64,
                "hidden_states": torch.bfloat16,
                "last_hidden_states": torch.bfloat16,
            },
            packed_loss_mask=prompt.packed_loss_mask,
        )

    class ControllerActor:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                if name == "try_dispatch_batch" and pending and clock.now >= 6.0:
                    controller.push_inference_results([capture(pending.pop())])
                result = getattr(controller, name)(*args, **kwargs)
                if name == "submit_training_dataset":
                    prompts = controller.get_prompts(17)
                    controller.push_inference_results([capture(p) for p in prompts[:-1]])
                    pending.append(prompts[-1])
                return result

            return SimpleNamespace(remote=call)

    class TrainingGroup:
        def __init__(self):
            self._actor_handlers = []
            self.pending_metrics = None
            for rank, rank_queue in enumerate(controller.get_train_queues()):

                def train(*, step, num_batches, rank_queue=rank_queue, rank=rank):
                    for _ in range(num_batches):
                        rank_queue.get_nowait()
                    metrics = {
                        "train/avg_loss": 1.0,
                        "train/avg_acc": 0.5,
                        "train/ce_loss": 1.5 if step == 0 else 2.0,
                        "train/l1_loss": 0.25 if step == 0 else 0.5,
                        "train/confidence_loss": 0.5 if step == 0 else 0.75,
                    }
                    if delayed_metrics and rank == 0:
                        previous = self.pending_metrics
                        self.pending_metrics = {**metrics, "_metrics_step": step + 1}
                        return previous or {}
                    return metrics

                self._actor_handlers.append(
                    SimpleNamespace(
                        get_global_step=SimpleNamespace(remote=lambda: 0),
                        train_from_queue=SimpleNamespace(remote=train),
                    )
                )

        def flush_pending_train_metrics(self):
            return [self.pending_metrics] if delayed_metrics else []

        def save_model(self, step, **kwargs):
            path = tmp_path / "checkpoints" / "latest_checkpointed_iteration.txt"
            path.parent.mkdir(exist_ok=True)
            path.write_text(str(step + 1))

    def advance(seconds):
        clock.now += seconds

    monkeypatch.setattr(loop.ray, "get", lambda value, **kwargs: value)
    monkeypatch.setattr(loop.time, "sleep", advance)
    monkeypatch.setattr(loop.time, "monotonic", lambda: clock.now)
    monkeypatch.setenv("TORCHSPEC_CI_STEP_LOGS", "1")
    args = SimpleNamespace(
        complete_data_pass=True,
        num_train_steps=2,
        steps_per_epoch=2,
        num_epochs=1,
        global_batch_size=16,
        per_dp_rank_batch_size=1,
        draft_accumulation_steps=4,
        training_num_nodes=1,
        training_num_gpus_per_node=4,
        prefetch_depth=4,
        dp_size=4,
        eval_interval=0,
        enable_perf_metrics=False,
        save_interval=0,
        save_per_epoch=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        output_dir=str(tmp_path),
        train_with_decode=False,
        distributed_timeout_minutes=1,
    )
    inference = SimpleNamespace(flush_metrics=SimpleNamespace(remote=lambda: {}))
    loop.training_loop(
        args,
        ControllerActor(),
        inference,
        TrainingGroup(),
        dataset_size=17,
        eval_dataset_size=0,
    )
    report = json.loads((tmp_path / "coverage.json").read_text())
    assert report["status"] == "complete"
    assert report["committed_window_count"] == 17
    assert report["optimizer_steps"] == 2
    assert report["dummy_count"] == 15
    assert (tmp_path / "checkpoints/latest_checkpointed_iteration.txt").read_text() == "2"
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in metrics] == [1, 2]
    assert [row["total_loss"] for row in metrics] == pytest.approx([0.875, 1.4])
    from PIL import Image

    with Image.open(tmp_path / "loss.png") as plot:
        assert plot.format == "PNG"
        assert plot.width >= 800

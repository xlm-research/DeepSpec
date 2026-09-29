import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ray.exceptions import RayTaskError

from torchspec.controller.inference_manager import AsyncInferenceManager
from torchspec.utils.types import InferenceInput


def _manager(engines, *, enable_perf_metrics=False):
    controller = MagicMock()
    controller.get_prompts.remote = AsyncMock(return_value=[])
    controller.set_inference_error.remote = AsyncMock()
    args = SimpleNamespace(
        inference_engine_type="vllm",
        inference_batch_size=1,
        train_with_decode=True,
        defer_tokenization=True,
        enable_perf_metrics=enable_perf_metrics,
    )
    manager_class = AsyncInferenceManager.__ray_metadata__.modified_class
    return manager_class(args, controller, inference_engines=engines), controller


@pytest.mark.parametrize("wrapper", ["direct", "ray", "ray_cause"])
@pytest.mark.parametrize("num_engines", [1, 2])
def test_dead_engine_stops_dispatch_and_reports_fatal_error(wrapper, num_engines):
    from vllm.v1.engine.exceptions import EngineDeadError

    error = EngineDeadError()
    if wrapper != "direct":
        error = RayTaskError("generate_with_decode", "engine traceback", error)
        if wrapper == "ray_cause":
            error = error.as_instanceof_cause()

    async def run():
        engines = [MagicMock() for _ in range(num_engines)]

        async def fail(**kwargs):
            await asyncio.sleep(0)
            raise error

        for engine in engines:
            engine.generate_with_decode.remote = AsyncMock(side_effect=fail)
        manager, controller = _manager(engines)
        manager._prompt_buffer.extend(
            InferenceInput(data_id=str(i), formatted_prompt="prompt")
            for i in range(num_engines + 2)
        )

        with pytest.raises(type(error)):
            await asyncio.wait_for(manager.run(), timeout=1)

        assert not manager._running
        controller.set_inference_error.remote.assert_awaited_once()
        assert "EngineCore" in controller.set_inference_error.remote.call_args.args[0]
        for engine in engines:
            engine.generate_with_decode.remote.assert_awaited_once()
        assert not manager._pending_tasks

    asyncio.run(run())


def test_bad_sample_does_not_stop_inference():
    async def run():
        error = ValueError("bad prompt")
        engine = MagicMock()
        engine.generate_with_decode.remote = AsyncMock(side_effect=error)
        manager, controller = _manager([engine])
        entry = InferenceInput(data_id="bad", formatted_prompt="prompt")

        assert await manager._dispatch_batch([entry]) == [(entry, error)]
        assert manager._running
        controller.set_inference_error.remote.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("enable_perf_metrics", [False, True])
def test_inference_throughput_counts_only_valid_outputs(monkeypatch, decode, enable_perf_metrics):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "torchspec.controller.inference_manager.time.perf_counter", lambda: clock.now
    )

    async def run():
        completion_tokens = 400 if decode else 0

        async def generate(**kwargs):
            clock.now += 2.0
            return [
                {
                    "mooncake_key": "valid",
                    "prompt_tokens": 1000,
                    "completion_tokens": completion_tokens,
                },
                None,
                {"prompt_tokens": 9000, "completion_tokens": 9000},
            ]

        engine = MagicMock()
        engine.generate.remote = AsyncMock(side_effect=generate)
        engine.generate_with_decode.remote = AsyncMock(side_effect=generate)
        manager, controller = _manager([engine], enable_perf_metrics=enable_perf_metrics)
        manager._train_with_decode = decode
        controller.push_inference_results.remote = AsyncMock()
        entries = [InferenceInput(data_id=str(i), formatted_prompt="prompt") for i in range(3)]

        results = await manager._dispatch_batch(entries)
        assert await manager._forward_results(results) == 1
        metrics = manager.flush_metrics()

        assert metrics["inference/prompt_tokens_avg"] == 1000
        assert metrics["inference/completion_tokens_avg"] == completion_tokens
        if enable_perf_metrics:
            assert metrics["perf/infer_batch_time"] == 2.0
            assert metrics["perf/infer_samples_per_sec"] == 0.5
            assert metrics["perf/infer_prompt_tokens_per_sec"] == 500
            assert metrics["perf/infer_completion_tokens_per_sec"] == completion_tokens / 2
            assert metrics["perf/infer_tokens_per_sec"] == (1000 + completion_tokens) / 2
        else:
            assert not any(key.startswith("perf/") for key in metrics)

    asyncio.run(run())


def test_inference_throughput_uses_wall_time_across_concurrent_engines(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "torchspec.controller.inference_manager.time.perf_counter", lambda: clock.now
    )

    async def run():
        second_started = asyncio.Event()

        async def first_batch(**kwargs):
            clock.now = 102.0
            await second_started.wait()
            clock.now = 104.0
            return [
                {"mooncake_key": "a", "prompt_tokens": 200, "completion_tokens": 60},
                {"mooncake_key": "b", "prompt_tokens": 300, "completion_tokens": 40},
            ]

        async def second_batch(**kwargs):
            second_started.set()
            await asyncio.sleep(0)
            clock.now = 106.0
            return [{"mooncake_key": "c", "prompt_tokens": 900, "completion_tokens": 300}]

        engines = [MagicMock(), MagicMock()]
        engines[0].generate_with_decode.remote = AsyncMock(side_effect=first_batch)
        engines[1].generate_with_decode.remote = AsyncMock(side_effect=second_batch)
        manager, _ = _manager(engines, enable_perf_metrics=True)
        entries = [InferenceInput(data_id=str(i), formatted_prompt="prompt") for i in range(3)]
        assert manager.flush_metrics() == {}

        await asyncio.gather(
            manager._dispatch_batch(entries[:2]), manager._dispatch_batch(entries[2:])
        )
        metrics = manager.flush_metrics()
        assert metrics["perf/infer_capacity"] == 0.75
        assert metrics["perf/infer_batch_time"] == 4
        assert metrics["perf/infer_samples_per_sec"] == 0.5
        assert metrics["perf/infer_prompt_tokens_per_sec"] == pytest.approx(1400 / 6)
        assert metrics["perf/infer_completion_tokens_per_sec"] == pytest.approx(400 / 6)
        assert metrics["perf/infer_tokens_per_sec"] == 300
        assert manager.flush_metrics() == metrics

        clock.now = 108.0
        assert manager.flush_metrics()["perf/infer_tokens_per_sec"] == 225
        clock.now = 115.0
        assert manager.flush_metrics()["perf/infer_tokens_per_sec"] == pytest.approx(1200 / 13)
        clock.now = 117.0
        assert manager.flush_metrics() == {}

    asyncio.run(run())


@pytest.mark.parametrize("elapsed", [0.0, 2.0])
@pytest.mark.parametrize("outputs", [[], [None], [{"mooncake_key": "no-token-counts"}]])
def test_inference_throughput_without_token_counts(monkeypatch, elapsed, outputs):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "torchspec.controller.inference_manager.time.perf_counter", lambda: clock.now
    )

    async def run():
        async def generate(**kwargs):
            clock.now += elapsed
            return outputs

        engine = MagicMock()
        engine.generate_with_decode.remote = AsyncMock(side_effect=generate)
        manager, _ = _manager([engine], enable_perf_metrics=True)
        await manager._dispatch_batch([InferenceInput(data_id="row", formatted_prompt="prompt")])
        metrics = manager.flush_metrics()
        if not outputs or elapsed == 0:
            assert metrics == {}
        else:
            assert metrics["perf/infer_samples_per_sec"] == (0.5 if outputs[0] else 0)
            assert not any("tokens_per_sec" in key for key in metrics)

    asyncio.run(run())

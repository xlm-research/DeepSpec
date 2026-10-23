"""Timing boundaries across data transfer, accumulation, and deferred reporting."""

import concurrent.futures
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchspec.data.utils import DataCollatorWithPadding
from torchspec.training import trainer as trainer_module
from torchspec.training.data_fetcher import PrefetchedDataFetcher
from torchspec.training.trainer import DeferredTrainMetrics, Trainer, _measure_draft_train_time


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("prefetched", [False, True])
@pytest.mark.parametrize("enable_perf", [False, True])
@pytest.mark.parametrize("initial_wait", [0.0, 100.0])
def test_draft_timing_includes_initial_wait_and_accumulation_but_excludes_aggregation(
    monkeypatch, deferred, prefetched, enable_perf, initial_wait
):
    clock = SimpleNamespace(now=0.0)
    timer = SimpleNamespace(time=lambda: clock.now, perf_counter=lambda: clock.now)
    monkeypatch.setattr(trainer_module, "time", timer)

    class Event:
        def __init__(self, *, enable_timing=False, blocking=False):
            self.blocking = blocking
            self.complete = False

        def record(self):
            self.recorded_at = clock.now

        def synchronize(self):
            if self.blocking and not self.complete:
                # The optimizer's GPU work finishes after its host call returns.
                clock.now += 5.0
            self.complete = True

        def elapsed_time(self, end):
            return (end.recorded_at - self.recorded_at) * 1000

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(Trainer, "_setup_device_mesh", lambda self: None)

    class TimingTrainer(Trainer):
        def init_model(self):
            pass

        def _train_step(self, *, step, **kwargs):
            clock.now += 3.0 + step
            return {}

        def _aggregate_metrics(self, all_step_metrics, step, *, grad_norm=None):
            clock.now += 1000.0  # Must never be charged to draft training time.
            if deferred:
                return DeferredTrainMetrics(
                    cpu_values=torch.tensor([1.0]),
                    source_values=torch.tensor([1.0]),
                    ready_event=Event(),
                    metric_step=step + 1,
                    finalize_values=lambda values: {"train/avg_loss": values[0]},
                )
            return {"train/avg_loss": 1.0, "_metrics_step": step + 1}

    trainer = TimingTrainer(
        Namespace(
            enable_perf_metrics=enable_perf,
            use_pytorch_profiler=False,
            record_memory_history=False,
        )
    )
    trainer.model = torch.nn.Identity()

    def optimizer_step():
        clock.now += 7.0
        return None

    trainer.optimizer = SimpleNamespace(step=optimizer_step)

    if enable_perf:
        # Deterministic completion observation; the separate test below exercises
        # real asynchronous waiting. This keeps exact boundary checks sleep-free.
        def submit(fn, *args):
            future = concurrent.futures.Future()
            future.set_result(fn(*args))
            return future

        monkeypatch.setattr(trainer._timing_executor, "submit", submit)
    else:
        assert trainer._timing_executor is None

    batch_indices = iter(range(8))
    collator = DataCollatorWithPadding()

    def receive_cpu_batch():
        batch_idx = next(batch_indices)
        clock.now += initial_wait if batch_idx % 4 == 0 else 11.0
        return collator(
            [
                {
                    "input_ids": torch.zeros(1, length, dtype=torch.long),
                    "loss_mask": torch.zeros(1, length, dtype=torch.long),
                }
                for length in (batch_idx + 1, 2)
            ]
        )

    def transfer(batch):
        clock.now += 2.0
        return batch

    if prefetched:
        fetcher = PrefetchedDataFetcher([])
        fetcher._started = True
        monkeypatch.setattr(fetcher._queue, "get", receive_cpu_batch)
        monkeypatch.setattr(fetcher, "_to_device", transfer)
        trainer.data_fetcher = fetcher
    else:

        def direct_batches():
            for _ in range(8):
                yield transfer(receive_cpu_batch())

        trainer.data_fetcher = direct_batches()

    try:
        first = trainer.train_from_queue(step=0, num_batches=4)
        second = trainer.train_from_queue(step=1, num_batches=4)
        tail = trainer.flush_pending_metrics()
        assert trainer.flush_pending_metrics() == {}
        if deferred:
            assert first == {}
            metrics = [second, tail]
        else:
            assert tail == {}
            metrics = [first, second]
        assert [m["_metrics_step"] for m in metrics] == [1, 2]
        if enable_perf:
            # Initial wait + all transfers/compute/later waits + optimizer/GPU completion.
            expected = initial_wait + 4 * 2 + 4 * 3 + 3 * 11 + 7 + 5
            assert [m["perf/draft_train_time"] for m in metrics] == [expected, expected + 4]
            assert [m["_num_input_tokens"] for m in metrics] == [18, 34]
        else:
            assert all("perf/draft_train_time" not in m for m in metrics)
            assert all("_num_input_tokens" not in m for m in metrics)
    finally:
        trainer.close()


def test_completion_observer_waits_for_gpu_and_preserves_time_until_deferred_finalize(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(trainer_module, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    waiting = threading.Event()
    gpu_finished = threading.Event()

    class GPUEvent:
        def synchronize(self):
            waiting.set()
            assert gpu_finished.wait(timeout=5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_measure_draft_train_time, clock.now, GPUEvent())
        try:
            assert waiting.wait(timeout=5)
            assert not future.done()
            clock.now = 25.0
        finally:
            gpu_finished.set()
        assert future.result(timeout=5) == 15.0

    clock.now = 1000.0  # Next step and metric reporting cannot extend this step.
    metrics = DeferredTrainMetrics(
        cpu_values=torch.tensor([1.0]),
        source_values=torch.tensor([1.0]),
        ready_event=Mock(),
        metric_step=7,
        finalize_values=lambda values: {"train/avg_loss": values[0]},
        draft_train_time_future=future,
    ).finalize()
    assert metrics["_metrics_step"] == 7
    assert metrics["perf/draft_train_time"] == 15.0

"""Bounded CPU feature reads while the current microbatch uses the GPU."""

import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .store import FEATURE_FIELDS, TensorStore


class FeaturePrefetch:
    def __init__(self, store_config, *, depth=2, timeout=1800, device=None, event=None):
        if depth < 1:
            raise ValueError("Feature prefetch depth must be positive")
        self.store = TensorStore(store_config)
        self.depth = depth
        self.timeout = timeout
        self.event = event
        self.pending = {}
        self.peak_pending = 0
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dspark-prefetch",
            initializer=self._initialize,
            initargs=(device,),
        )

    @staticmethod
    def _initialize(device):
        # CUDA's current device is thread-local. Pinned allocations must follow
        # this Titan rank's assigned device, including in the prefetch thread.
        if device is not None:
            torch.cuda.set_device(device)

    def _fetch(self, descriptor):
        started = time.monotonic()
        position = descriptor["position"]
        if self.event:
            self.event("transfer_start", position=position)
        result = self.store.get(descriptor["fields"], FEATURE_FIELDS, device="cpu")
        if self.event:
            self.event(
                "transfer_end", position=position, seconds=time.monotonic() - started
            )
        return result

    def submit(self, descriptor):
        position = descriptor["position"]
        if position in self.pending:
            return
        if len(self.pending) >= self.depth:
            raise RuntimeError("The feature prefetch window is full")
        self.pending[position] = self.executor.submit(self._fetch, descriptor)
        self.peak_pending = max(self.peak_pending, len(self.pending))

    def take(self, position):
        future = self.pending[position]
        result = future.result(timeout=self.timeout)
        del self.pending[position]
        return result

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.store.close()

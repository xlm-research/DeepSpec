"""Bounded CPU feature reads while the current microbatch uses the GPU."""

import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .store import FEATURE_FIELDS, TensorStore


class FeaturePrefetch:
    def __init__(
        self,
        store_config,
        *,
        depth=2,
        max_bytes=None,
        timeout=1800,
        device=None,
        event=None,
    ):
        if depth < 1:
            raise ValueError("Feature prefetch depth must be positive")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("Feature prefetch byte budget must be positive")
        self.store = TensorStore(
            store_config,
            get_workers=int(store_config.get("prefetch_workers", depth)),
        )
        self.depth = depth
        self.max_bytes = int(max_bytes) if max_bytes is not None else None
        self.timeout = timeout
        self.event = event
        self.pending = {}
        self.pending_bytes = 0
        self.peak_pending = 0
        self.peak_pending_bytes = 0
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, min(depth, int(store_config.get("prefetch_workers", depth)))),
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
                "transfer_end",
                position=position,
                seconds=time.monotonic() - started,
                store=self.store.last_read,
            )
        return result

    def submit(self, descriptor):
        position = descriptor["position"]
        if position in self.pending:
            return
        if len(self.pending) >= self.depth:
            raise RuntimeError("The feature prefetch window is full")
        nbytes = sum(
            int(descriptor["fields"][name]["nbytes"]) for name in FEATURE_FIELDS
        )
        if self.max_bytes is not None and self.pending_bytes + nbytes > self.max_bytes:
            raise RuntimeError("The feature prefetch byte budget is full")
        self.pending[position] = self.executor.submit(self._fetch, descriptor)
        self.pending[position]._deepspec_nbytes = nbytes
        self.pending_bytes += nbytes
        self.peak_pending = max(self.peak_pending, len(self.pending))
        self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)

    def take(self, position):
        future = self.pending[position]
        try:
            return future.result(timeout=self.timeout)
        finally:
            del self.pending[position]
            self.pending_bytes -= getattr(future, "_deepspec_nbytes", 0)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.pending_bytes = 0
        self.store.close()

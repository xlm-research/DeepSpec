"""Bounded reusable buffers and asynchronous transfer handles.

Mooncake's Python client accepts integer pointers to caller-owned memory.  A
transfer therefore owns both the registered allocation and the source tensor
references until the native call has completed.  These classes make that
ownership explicit and keep the amount of staging memory bounded.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch


def _aligned(value: int, alignment: int = 256) -> int:
    if value <= 0:
        raise ValueError("A Mooncake buffer must have positive size")
    return ((int(value) + alignment - 1) // alignment) * alignment


@dataclass
class HostBuffer:
    """One registered host allocation owned by :class:`HostBufferPool`."""

    tensor: torch.Tensor
    registered: bool = False
    in_use: bool = False

    @property
    def size(self) -> int:
        return int(self.tensor.numel())

    @property
    def ptr(self) -> int:
        return int(self.tensor.data_ptr())

    def copy_tensors(
        self,
        tensors: Iterable[torch.Tensor],
        *,
        pin_memory: bool,
    ) -> tuple[list[int], list[int], torch.cuda.Event | None, list[torch.Tensor]]:
        """Copy tensors into this buffer and return pointers and sizes.

        CUDA copies are issued on the caller's current stream.  The returned
        event is synchronized by the worker thread before it calls Mooncake;
        source tensors are retained in the returned list until that happens.
        """

        pointers: list[int] = []
        sizes: list[int] = []
        keepalive: list[torch.Tensor] = []
        offset = 0
        has_cuda = False
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Mooncake staging expects torch.Tensor values")
            source = tensor.contiguous()
            nbytes = int(source.numel() * source.element_size())
            if nbytes <= 0:
                raise ValueError("Mooncake cannot transfer an empty tensor")
            if offset + nbytes > self.size:
                raise ValueError(
                    f"Host staging buffer is too small: need {offset + nbytes}, "
                    f"have {self.size}"
                )
            destination = self.tensor[offset : offset + nbytes]
            source_bytes = source.view(torch.uint8).view(-1)
            # non_blocking is useful for a pinned destination, but the event
            # below is still required because Mooncake runs outside CUDA stream
            # ordering.
            destination.copy_(source_bytes, non_blocking=pin_memory)
            pointers.append(self.ptr + offset)
            sizes.append(nbytes)
            offset += nbytes
            has_cuda = has_cuda or source.is_cuda
            keepalive.append(source)

        event = None
        if has_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        return pointers, sizes, event, keepalive

    def close(self) -> None:
        self.tensor = torch.empty(0, dtype=torch.uint8)
        self.registered = False
        self.in_use = False


class HostBufferPool:
    """A byte-bounded pool of reusable, optionally pinned host buffers.

    Buffers are allocated lazily because the feature shape is only known after
    the first prepared sample.  At most ``max_buffers`` allocations can be in
    flight; callers block when all suitable buffers are owned by transfers.
    """

    def __init__(
        self,
        *,
        max_buffers: int = 1,
        register: Callable[[int, int], None] | None = None,
        unregister: Callable[[int], None] | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        if max_buffers < 1:
            raise ValueError("max_buffers must be positive")
        self.max_buffers = int(max_buffers)
        self._register = register
        self._unregister = unregister
        self._pin_memory = (
            torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        )
        self._buffers: list[HostBuffer] = []
        self._condition = threading.Condition()
        self._closed = False

    @property
    def buffers(self) -> tuple[HostBuffer, ...]:
        with self._condition:
            return tuple(self._buffers)

    def _allocate(self, size: int) -> HostBuffer:
        nbytes = _aligned(size)
        try:
            tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=self._pin_memory)
        except (RuntimeError, TypeError):
            # CPU-only test environments often have no pinned allocator.  The
            # transport remains correct over TCP with ordinary host memory.
            tensor = torch.empty(nbytes, dtype=torch.uint8)
        buffer = HostBuffer(tensor=tensor)
        if self._register is not None:
            self._register(buffer.ptr, buffer.size)
            buffer.registered = True
        return buffer

    def acquire(self, size: int, timeout: float | None = None) -> HostBuffer:
        if size <= 0:
            raise ValueError("Cannot acquire a zero-sized Mooncake buffer")
        started = time.monotonic()
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("Mooncake host buffer pool is closed")
                suitable = next(
                    (item for item in self._buffers if not item.in_use and item.size >= size),
                    None,
                )
                if suitable is not None:
                    suitable.in_use = True
                    return suitable
                if len(self._buffers) < self.max_buffers:
                    buffer = self._allocate(size)
                    buffer.in_use = True
                    self._buffers.append(buffer)
                    return buffer
                # Samples can have different sequence lengths.  If every
                # slot is free but none is large enough, replace the smallest
                # free allocation instead of waiting forever for a slot that
                # can never satisfy this request.
                replace = next(
                    (item for item in self._buffers if not item.in_use), None
                )
                if replace is not None:
                    self._buffers.remove(replace)
                    if replace.registered and self._unregister is not None:
                        self._unregister(replace.ptr)
                    replace.close()
                    buffer = self._allocate(size)
                    buffer.in_use = True
                    self._buffers.append(buffer)
                    return buffer
                remaining = None
                if timeout is not None:
                    remaining = float(timeout) - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for Mooncake host buffer")
                self._condition.wait(remaining)

    def release(self, buffer: HostBuffer) -> None:
        with self._condition:
            if buffer not in self._buffers:
                return
            buffer.in_use = False
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if any(item.in_use for item in self._buffers):
                raise RuntimeError("Cannot close Mooncake host buffers with transfers in flight")
            self._closed = True
            buffers = list(self._buffers)
            self._buffers.clear()
            self._condition.notify_all()
        for buffer in buffers:
            if buffer.registered and self._unregister is not None:
                self._unregister(buffer.ptr)
            buffer.close()


class TransferHandle:
    """A Future with explicit transfer completion semantics."""

    def __init__(
        self,
        future: Future,
        *,
        keepalive: object | None = None,
    ) -> None:
        self.future = future
        # Keep source tensors and any CUDA event reachable until the native
        # transfer has completed.  The callback below releases the buffer, but
        # this object is intentionally retained by callers until wait().
        self.keepalive = keepalive

    def done(self) -> bool:
        return self.future.done()

    def wait(self, timeout: float | None = None):
        return self.future.result(timeout=timeout)

    result = wait

    def exception(self, timeout: float | None = None):
        return self.future.exception(timeout=timeout)


class AsyncPutManager:
    """Run bounded put tasks and surface asynchronous failures."""

    def __init__(self, *, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="deepspec-mooncake-put"
        )
        self._condition = threading.Condition()
        self._inflight: set[Future] = set()
        self._errors: list[BaseException] = []
        self._closed = False

    def submit(
        self,
        operation: Callable[[], object],
        *,
        release: Callable[[], None],
        keepalive: object | None = None,
    ) -> TransferHandle:
        with self._condition:
            if self._closed:
                raise RuntimeError("Mooncake async put manager is closed")
            future = self._executor.submit(operation)
            self._inflight.add(future)

        def completed(done: Future) -> None:
            try:
                done.result()
            except Exception as error:  # surfaced by check_errors/drain  # noqa: BLE001
                with self._condition:
                    self._errors.append(error)
            finally:
                try:
                    release()
                finally:
                    with self._condition:
                        self._inflight.discard(done)
                        self._condition.notify_all()

        future.add_done_callback(completed)
        return TransferHandle(future, keepalive=keepalive)

    def check_errors(self) -> None:
        with self._condition:
            if self._errors:
                raise self._errors.pop(0)

    def drain(self, timeout: float | None = None) -> None:
        started = time.monotonic()
        while True:
            with self._condition:
                if not self._inflight:
                    break
                remaining = None
                if timeout is not None:
                    remaining = float(timeout) - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Timed out draining Mooncake transfers")
                self._condition.wait(remaining)
        self.check_errors()

    def shutdown(self, timeout: float | None = None) -> None:
        error = None
        try:
            self.drain(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- preserve worker failure for shutdown
            error = exc
        with self._condition:
            self._closed = True
            pending = bool(self._inflight)
        # A failed drain cannot synchronously join a blocked native put. The
        # operation and its callback keep the registered buffers alive until
        # completion or termination of the owning process.
        self._executor.shutdown(wait=not pending, cancel_futures=True)
        if error is not None:
            raise error

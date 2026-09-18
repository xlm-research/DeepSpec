"""Bounded, retrying deletion for Mooncake object groups.

Deletion is part of the ledger's correctness boundary: capacity is returned
only after Mooncake confirms the remove.  The manager runs retries in worker
threads so a Ray actor's asyncio loop is never blocked by a slow metadata RPC.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor


class DeleteManager:
    def __init__(
        self,
        delete_fn,
        *,
        max_workers=1,
        max_attempts=5,
        retry_delay=0.1,
        retry_backoff=2.0,
    ):
        if max_workers < 1 or max_attempts < 1:
            raise ValueError("Delete worker and attempt counts must be positive")
        if retry_delay < 0 or retry_backoff < 1:
            raise ValueError("Delete retry settings are invalid")
        self.delete_fn = delete_fn
        self.max_attempts = int(max_attempts)
        self.retry_delay = float(retry_delay)
        self.retry_backoff = float(retry_backoff)
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers), thread_name_prefix="deepspec-mooncake-delete"
        )
        self._lock = threading.Lock()
        self._futures = set()
        self._closed = False
        self.attempts = 0
        self.successes = 0
        self.failures = 0

    def _delete_with_retry(self, fields):
        delay = self.retry_delay
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            with self._lock:
                self.attempts += 1
            try:
                self.delete_fn(fields)
                with self._lock:
                    self.successes += 1
                return
            except Exception as error:  # noqa: BLE001 -- retry all worker failures
                last_error = error
                if attempt == self.max_attempts:
                    break
                if delay:
                    time.sleep(delay)
                delay *= self.retry_backoff
        with self._lock:
            self.failures += 1
        raise RuntimeError(
            f"Mooncake deletion failed after {self.max_attempts} attempts"
        ) from last_error

    def submit(self, fields) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("Mooncake delete manager is closed")
            future = self._executor.submit(self._delete_with_retry, fields)
            self._futures.add(future)

        def finished(done):
            with self._lock:
                self._futures.discard(done)

        future.add_done_callback(finished)
        return future

    def delete(self, fields, timeout=None):
        return self.submit(fields).result(timeout=timeout)

    def drain(self, timeout=None):
        start = time.monotonic()
        while True:
            with self._lock:
                futures = tuple(self._futures)
            if not futures:
                return
            remaining = None
            if timeout is not None:
                remaining = float(timeout) - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("Timed out draining Mooncake deletions")
            for future in futures:
                future.result(timeout=remaining)

    def close(self, timeout=None):
        error = None
        try:
            self.drain(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- close must always stop workers
            error = exc
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)
        if error is not None:
            raise error

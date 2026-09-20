"""Registered tensor buffers and bounded Mooncake transfer ownership.

The pipeline deliberately keeps the descriptor/ledger protocol outside this
module. ``TensorStore`` owns the native client and makes the lifetime of
registered memory explicit: a synchronous operation unregisters its source
after completion, while an asynchronous operation retains a reusable pinned
buffer until Mooncake has finished reading it.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from .mooncake import AsyncPutManager, HostBufferPool, TransferHandle

FIELDS = (
    "input_ids",
    "loss_mask",
    "seq_len",
    "context_chunk_len",
    "target_hidden_states",
    "target_last_hidden_states",
)
FEATURE_FIELDS = FIELDS[-2:]
DTYPES = {
    name: getattr(torch, name) for name in ("int64", "bool", "float32", "bfloat16")
}
CHUNK_BYTES = 8 * 1024**2
_MISSING_OBJECT = -704


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def tensor_digest(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return hashlib.sha256(
        memoryview(tensor.contiguous().view(torch.uint8).numpy())
    ).hexdigest()


def describe_tensors(prefix, tensors, *, chunk_bytes=CHUNK_BYTES):
    """Build an immutable descriptor and chunk every tensor at ``chunk_bytes``."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    result = {}
    for name in FIELDS:
        tensor = tensors[name]
        if not tensor.is_contiguous() or tensor.is_cuda:
            raise ValueError("Producer tensors must be contiguous CPU tensors")
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported feature dtype: {dtype}")
        size = tensor.numel() * tensor.element_size()
        if size <= 0:
            raise ValueError(f"Tensor {name} must not be empty")
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": dtype,
            "nbytes": size,
            "sha256": tensor_digest(tensor),
            "chunks": [
                {
                    "key": f"{prefix}/{name}/{offset // chunk_bytes}",
                    "offset": offset,
                    "nbytes": min(chunk_bytes, size - offset),
                }
                for offset in range(0, size, chunk_bytes)
            ],
        }
    return result


def object_keys(fields):
    return [chunk["key"] for spec in fields.values() for chunk in spec["chunks"]]


@dataclass(frozen=True)
class MooncakeCapabilities:
    """Features detected on the installed Mooncake Python client."""

    force_delete: bool
    batch_is_exist: bool
    multi_buffer_put: bool
    registered_buffers: bool


class TensorStore:
    """Thread-safe Mooncake client with synchronous and staged async paths."""

    def __init__(
        self,
        config,
        *,
        pool_bytes=0,
        async_put_pool_size=None,
        host_buffer_size=None,
        get_workers=None,
    ):
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        self.store_config = dict(config)
        self.client = MooncakeDistributedStore()
        self.quarantined = []
        self._io_lock = threading.RLock()
        self._registered_buffers = {}
        self._last_write_lock = threading.Lock()
        self._closed = False
        self._close_lock = threading.Lock()
        self._close_finished = threading.Event()
        self._close_thread = None
        self._close_error = None
        self.last_write = None
        self.last_read = None
        self.verify_mode = self.store_config.get("verify_mode", "full")
        if self.verify_mode not in ("full", "none"):
            raise ValueError("Mooncake verify_mode must be 'full' or 'none'")
        configured_pool = self.store_config.get("async_put_pool_size", 1)
        self.async_put_pool_size = int(
            configured_pool if async_put_pool_size is None else async_put_pool_size
        )
        if self.async_put_pool_size < 0:
            raise ValueError("async_put_pool_size must be non-negative")
        self.host_buffer_size = int(
            host_buffer_size
            if host_buffer_size is not None
            else self.store_config.get("host_buffer_size", 0)
        )
        self.get_workers = max(
            1,
            int(
                get_workers
                if get_workers is not None
                else self.store_config.get("get_workers", 1)
            ),
        )
        self.wait_for_visibility = bool(
            self.store_config.get("wait_for_visibility", False)
        )
        self.visibility_timeout = float(
            self.store_config.get("visibility_timeout", 30.0)
        )
        self.visibility_poll_interval = float(
            self.store_config.get("visibility_poll_interval", 0.002)
        )
        self._host_buffer_pool = None
        self._put_manager = None
        self._get_executor = None
        self.config = ReplicateConfig()
        self.config.replica_num = 1
        self.config.with_hard_pin = True
        self.config.dfs_replica_num = 0
        self.config.nof_replica_num = 0
        host = self.store_config["host"]
        if self.store_config.get("per_node_hosts"):
            host = self.store_config.get("hosts_by_hostname", {}).get(
                socket.gethostname()
            )
            if host is None:
                host = os.environ["DEEPSPEC_STORE_HOST"]
        status = self.client.setup(
            local_hostname=f"{host}:{free_port()}",
            metadata_server="P2PHANDSHAKE",
            global_segment_size=pool_bytes,
            local_buffer_size=16 * 1024**2,
            protocol=self.store_config["protocol"],
            rdma_devices=self.store_config.get("rdma_devices", ""),
            master_server_addr=self.store_config["master"],
            enable_ssd_offload=False,
        )
        if status not in (None, 0):
            raise RuntimeError(f"Mooncake setup failed: {status}")
        self.endpoint = self.client.get_hostname()
        self.capabilities = self._detect_capabilities()

    def _detect_capabilities(self):
        batch_remove = getattr(self.client, "batch_remove", None)
        force_delete = callable(batch_remove)
        try:
            signature = inspect.signature(batch_remove)
            force_delete = force_delete and (
                "force" in signature.parameters or not signature.parameters
            )
        except (TypeError, ValueError):
            pass
        return MooncakeCapabilities(
            force_delete=force_delete,
            batch_is_exist=callable(getattr(self.client, "batch_is_exist", None)),
            multi_buffer_put=callable(
                getattr(self.client, "batch_put_from_multi_buffers", None)
            ),
            registered_buffers=callable(getattr(self.client, "register_buffer", None))
            and callable(getattr(self.client, "unregister_buffer", None)),
        )

    @staticmethod
    def _status_ok(status):
        return status is None or status == 0

    @staticmethod
    def _result_list(results, length=0):
        if results is None:
            return [0] * length
        return list(results)

    def _register_ptr(self, pointer, size):
        with self._io_lock:
            if pointer in self._registered_buffers:
                return
            register = getattr(self.client, "register_buffer", None)
            if not callable(register):
                raise RuntimeError(  # noqa: TRY004 -- a missing native capability is a runtime failure
                    "Mooncake client cannot register transfer buffers"
                )
            status = register(pointer, size)
            if not self._status_ok(status):
                raise RuntimeError(f"Mooncake buffer registration failed: {status}")
            self._registered_buffers[pointer] = size

    def _unregister_ptr(self, pointer):
        with self._io_lock:
            if pointer not in self._registered_buffers:
                return
            unregister = getattr(self.client, "unregister_buffer", None)
            if not callable(unregister):
                self.quarantined.append(pointer)
                raise RuntimeError(  # noqa: TRY004 -- a missing native capability is a runtime failure
                    "Mooncake client cannot unregister transfer buffers"
                )
            status = unregister(pointer)
            if not self._status_ok(status):
                self.quarantined.append(pointer)
                raise RuntimeError(f"Mooncake unregister failed: {status}")
            self._registered_buffers.pop(pointer, None)

    def _register(self, tensors):
        registered = []
        try:
            for tensor in tensors:
                self._register_ptr(
                    tensor.data_ptr(), tensor.numel() * tensor.element_size()
                )
                registered.append(tensor)
        except BaseException:
            self.quarantined.extend(registered)
            raise
        return registered

    def _unregister(self, tensors):
        for tensor in tensors:
            self._unregister_ptr(tensor.data_ptr())

    @property
    def supports_multi_buffer_put(self):
        """Whether the installed client supports scatter-source puts."""

        return self.capabilities.multi_buffer_put

    def register_external_buffer(self, pointer, size):
        """Register caller-owned memory until an explicit unregister."""

        if self._closed:
            raise RuntimeError("Mooncake tensor store is closed")
        self._register_ptr(int(pointer), int(size))
        return True

    def unregister_external_buffer(self, pointer):
        if self._closed:
            raise RuntimeError("Mooncake tensor store is closed")
        self._unregister_ptr(int(pointer))
        return True

    def check_async_errors(self):
        if self._put_manager is not None:
            self._put_manager.check_errors()

    def put_registered_multi_buffers(self, keys, pointers, sizes):
        """Publish objects assembled from already registered fragments."""

        if self._closed:
            raise RuntimeError("Mooncake tensor store is closed")
        keys, pointers, sizes = list(keys), list(pointers), list(sizes)
        if not keys or len(keys) != len(pointers) or len(keys) != len(sizes):
            raise ValueError(
                "Registered Mooncake put lists must have equal non-zero length"
            )
        method = getattr(self.client, "batch_put_from_multi_buffers", None)
        if not callable(method):
            raise RuntimeError(  # noqa: TRY004 -- a missing native capability is a runtime failure
                "Mooncake scatter-source put is unavailable"
            )
        with self._io_lock:
            try:
                results = method(keys, pointers, sizes, config=self.config)
            except TypeError as keyword_error:
                try:
                    results = method(keys, pointers, sizes, self.config)
                except TypeError:
                    raise keyword_error
        results = self._result_list(results, len(keys))
        if len(results) != len(keys) or any(status != 0 for status in results):
            self._cleanup_keys(keys)
            raise RuntimeError(f"Mooncake multi-buffer put failed: {results}")

    @staticmethod
    def _validate_chunks(name, spec, tensor_nbytes):
        if int(spec.get("nbytes", -1)) != tensor_nbytes:
            raise ValueError(f"Invalid tensor descriptor: {name}")
        expected = 0
        chunks = spec.get("chunks") or []
        for chunk in chunks:
            if (
                int(chunk.get("offset", -1)) != expected
                or int(chunk.get("nbytes", 0)) <= 0
                or expected + int(chunk["nbytes"]) > tensor_nbytes
                or not chunk.get("key")
            ):
                raise ValueError(f"Invalid tensor chunks: {name}")
            expected += int(chunk["nbytes"])
        if expected != tensor_nbytes:
            raise ValueError(f"Incomplete tensor chunks: {name}")

    def _plan_put(self, fields, tensors):
        names = list(fields)
        if not names or any(name not in tensors for name in names):
            raise ValueError("Tensor values do not match the Mooncake descriptor")
        keys, offsets, sizes = [], [], []
        total = 0
        for name in names:
            tensor = tensors[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Mooncake value {name} is not a tensor")
            nbytes = int(tensor.numel() * tensor.element_size())
            if fields[name]["shape"] != list(tensor.shape) or fields[name][
                "dtype"
            ] != str(tensor.dtype).removeprefix("torch."):
                raise ValueError(f"Tensor shape/dtype differs from descriptor: {name}")
            self._validate_chunks(name, fields[name], nbytes)
            for chunk in fields[name]["chunks"]:
                keys.append(chunk["key"])
                offsets.append((name, int(chunk["offset"])))
                sizes.append(int(chunk["nbytes"]))
            total += nbytes
        return names, keys, offsets, sizes, total

    def _batch_put(self, keys, pointers, sizes):
        with self._io_lock:
            method = self.client.batch_put_from
            try:
                results = method(keys, pointers, sizes, config=self.config)
            except TypeError as keyword_error:
                try:
                    results = method(keys, pointers, sizes, self.config)
                except TypeError:
                    raise keyword_error
        results = self._result_list(results, len(keys))
        if len(results) != len(keys) or any(value != 0 for value in results):
            raise RuntimeError(f"Mooncake put failed: {results}")
        return results

    def _cleanup_keys(self, keys):
        if not keys:
            return
        try:
            self._remove_keys(keys)
        except Exception:  # noqa: BLE001 -- cleanup must not mask the put failure
            self.quarantined.append(tuple(keys))

    def _perform_put(self, keys, pointers, sizes, started, prepared):
        transfer_started = time.monotonic()
        try:
            self._batch_put(keys, pointers, sizes)
            if self.wait_for_visibility:
                self.wait_for_keys(
                    keys,
                    timeout=self.visibility_timeout,
                    poll_interval=self.visibility_poll_interval,
                )
        except BaseException:
            self._cleanup_keys(keys)
            raise
        finished = time.monotonic()
        metrics = {
            "nbytes": sum(sizes),
            "prepare_seconds": prepared - started,
            "transfer_seconds": finished - transfer_started,
            "unregister_seconds": 0.0,
        }
        with self._last_write_lock:
            self.last_write = metrics
        return metrics

    def put(self, fields, tensors):
        """Synchronously publish tensors from their registered source buffers."""

        if self._closed:
            raise RuntimeError("Mooncake tensor store is closed")
        started = time.monotonic()
        names, keys, offsets, sizes, _total = self._plan_put(fields, tensors)
        source_tensors = [tensors[name] for name in names]
        if any(
            not tensor.is_contiguous() or tensor.is_cuda for tensor in source_tensors
        ):
            raise ValueError("Synchronous Mooncake puts require contiguous CPU tensors")
        pointers = [tensors[name].data_ptr() + offset for name, offset in offsets]
        transfer_started = time.monotonic()
        with self._io_lock:
            buffers = self._register(source_tensors)
            try:
                self._batch_put(keys, pointers, sizes)
                if self.wait_for_visibility:
                    self.wait_for_keys(
                        keys,
                        timeout=self.visibility_timeout,
                        poll_interval=self.visibility_poll_interval,
                    )
            except Exception:
                self.quarantined.extend(buffers)
                self._cleanup_keys(keys)
                raise
            transfer_finished = time.monotonic()
            self._unregister(buffers)
        metrics = {
            "nbytes": sum(sizes),
            "prepare_seconds": transfer_started - started,
            "transfer_seconds": transfer_finished - transfer_started,
            "unregister_seconds": time.monotonic() - transfer_finished,
        }
        with self._last_write_lock:
            self.last_write = metrics

    def _ensure_async_put(self):
        if self.async_put_pool_size < 1:
            raise RuntimeError(
                "Async Mooncake puts are disabled; set async_put_pool_size >= 1"
            )
        if self._host_buffer_pool is None:
            self._host_buffer_pool = HostBufferPool(
                max_buffers=self.async_put_pool_size,
                register=self._register_ptr,
                unregister=self._unregister_ptr,
                pin_memory=self.store_config.get("pin_memory"),
            )
            self._put_manager = AsyncPutManager(max_workers=self.async_put_pool_size)
        return self._host_buffer_pool, self._put_manager

    def put_async(self, fields, tensors, *, timeout=None):
        """Stage a tensor group and publish it on a bounded worker pool."""

        if self._closed:
            raise RuntimeError("Mooncake tensor store is closed")
        started = time.monotonic()
        names, keys, offsets, sizes, total = self._plan_put(fields, tensors)
        if self.host_buffer_size and total > self.host_buffer_size:
            raise ValueError("Feature exceeds the planned writer staging byte bound")
        pool, manager = self._ensure_async_put()
        buffer = pool.acquire(max(total, self.host_buffer_size), timeout=timeout)
        try:
            source_tensors = [tensors[name] for name in names]
            base_pointers, _copied_sizes, event, keepalive = buffer.copy_tensors(
                source_tensors,
                pin_memory=buffer.tensor.is_pinned(),
            )
            index = {name: i for i, name in enumerate(names)}
            pointers = [base_pointers[index[name]] + offset for name, offset in offsets]

            # Capture source views and the CUDA event in the operation itself,
            # so dropping the public handle cannot free memory early.
            def operation():
                if event is not None:
                    event.synchronize()
                _ = keepalive
                return self._perform_put(
                    keys, pointers, sizes, started, time.monotonic()
                )

            return manager.submit(
                operation,
                release=lambda: pool.release(buffer),
                keepalive=(buffer, keepalive, event),
            )
        except BaseException:
            pool.release(buffer)
            raise

    def get(self, fields, names, *, device="cpu", verify=True):
        if self._closed:
            raise RuntimeError("Mooncake tensor store is closed")
        started = time.monotonic()
        tensors = {}
        keys, pointers, sizes = [], [], []
        for name in names:
            spec = fields[name]
            tensor = torch.empty(
                spec["shape"],
                dtype=DTYPES[spec["dtype"]],
                device=device,
                pin_memory=str(device) == "cpu" and torch.cuda.is_available(),
            )
            if math.prod(spec["shape"]) * tensor.element_size() != spec["nbytes"]:
                raise ValueError(f"Invalid tensor descriptor: {name}")
            self._validate_chunks(name, spec, int(spec["nbytes"]))
            for chunk in spec["chunks"]:
                keys.append(chunk["key"])
                pointers.append(tensor.data_ptr() + int(chunk["offset"]))
                sizes.append(int(chunk["nbytes"]))
            tensors[name] = tensor
        if self.wait_for_visibility:
            self.wait_for_keys(
                keys,
                timeout=self.visibility_timeout,
                poll_interval=self.visibility_poll_interval,
            )
        transfer_started = time.monotonic()
        with self._io_lock:
            buffers = self._register(list(tensors.values()))
            try:
                results = self.client.batch_get_into(keys, pointers, sizes)
                if list(results) != sizes:
                    raise RuntimeError(
                        f"Mooncake get returned {results}, expected {sizes}"
                    )
            except Exception:
                self.quarantined.extend(buffers)
                raise
            transfer_finished = time.monotonic()
            self._unregister(buffers)
        verify_started = time.monotonic()
        should_verify = bool(verify and self.verify_mode != "none")
        if should_verify:
            for name, tensor in tensors.items():
                if tensor_digest(tensor) != fields[name]["sha256"]:
                    raise ValueError(f"Transferred feature checksum mismatch: {name}")
        self.last_read = {
            "nbytes": sum(sizes),
            "prepare_seconds": transfer_started - started,
            "transfer_seconds": transfer_finished - transfer_started,
            "unregister_seconds": verify_started - transfer_finished,
            "verify_seconds": time.monotonic() - verify_started,
            "verified": should_verify,
        }
        return tensors

    def get_async(self, fields, names, *, device="cpu", verify=True):
        """Schedule a receive without exposing the executor directly."""

        if self._closed:
            raise RuntimeError("Mooncake tensor store is closed")
        if self._get_executor is None:
            self._get_executor = ThreadPoolExecutor(
                max_workers=self.get_workers, thread_name_prefix="deepspec-mooncake-get"
            )
        future = self._get_executor.submit(
            self.get, fields, names, device=device, verify=verify
        )
        return TransferHandle(future)

    def batch_exists(self, keys):
        keys = list(keys)
        if not keys:
            return {}
        with self._io_lock:
            method = getattr(self.client, "batch_is_exist", None)
            if callable(method):
                result = list(method(keys))
            else:
                single = getattr(self.client, "is_exist", None)
                if not callable(single):
                    raise RuntimeError(  # noqa: TRY004 -- a missing native capability is a runtime failure
                        "Mooncake client cannot query object visibility"
                    )
                result = [single(key) for key in keys]
        if len(result) != len(keys):
            raise RuntimeError(
                f"Mooncake existence query returned {len(result)} results for {len(keys)} keys"
            )
        if any(value not in (0, 1, False, True) for value in result):
            raise RuntimeError(f"Mooncake existence query failed: {result}")
        return {key: value == 1 or value is True for key, value in zip(keys, result)}

    def exists(self, key):
        return self.batch_exists([key]).get(key, False)

    def wait_for_keys(self, keys, *, timeout=None, poll_interval=None):
        keys = list(keys)
        if not keys:
            return
        timeout = self.visibility_timeout if timeout is None else float(timeout)
        interval = (
            self.visibility_poll_interval
            if poll_interval is None
            else max(float(poll_interval), 0.001)
        )
        start = time.monotonic()
        deadline = None if timeout <= 0 else start + timeout
        while True:
            census = self.batch_exists(keys)
            missing = [key for key in keys if not census[key]]
            if not missing:
                return
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Mooncake keys; missing {missing[:8]}"
                )
            sleep_for = interval if deadline is None else min(interval, deadline - now)
            time.sleep(max(sleep_for, 0.0))

    def _remove_keys(self, keys):
        keys = list(keys)
        if not keys:
            return
        with self._io_lock:
            method = self.client.batch_remove
            try:
                results = method(keys, force=True)
            except TypeError:
                results = method(keys)
        results = self._result_list(results, len(keys))
        if len(results) != len(keys):
            raise RuntimeError(f"Mooncake remove returned {results}")
        failures = [
            (key, status)
            for key, status in zip(keys, results)
            if status not in (0, None, _MISSING_OBJECT)
        ]
        if failures:
            raise RuntimeError(f"Mooncake remove failed: {failures}")
        return results

    def remove(self, fields):
        if self._closed:
            raise RuntimeError("Mooncake tensor store is closed")
        keys = object_keys(fields)
        self._remove_keys(keys)
        if any(self.batch_exists(keys).values()):
            raise RuntimeError("Mooncake objects are still visible after removal")

    def close(self, *, timeout=35):
        """Bound the caller's wait while retaining native buffers until completion.

        A timeout is not cleanup success. The actor/process supervisor must then
        stop this client's owning process; no Python cancellation can stop C++.
        """
        from .runtime import Deadline, bounded_lock

        deadline = Deadline.after(timeout)
        with bounded_lock(self._close_lock, deadline):
            if self._close_thread is None:

                def close_native():
                    try:
                        self._close_blocking()
                    except BaseException as error:  # noqa: BLE001 -- deliver native cleanup failure to the waiting caller
                        self._close_error = error
                    finally:
                        self._close_finished.set()

                self._close_thread = threading.Thread(
                    target=close_native, name="deepspec-store-close", daemon=True
                )
                self._close_thread.start()
        if not self._close_finished.wait(deadline.remaining()):
            raise TimeoutError(
                "Mooncake close exceeded cleanup deadline; native buffers remain owned"
            )
        if self._close_error is not None:
            raise self._close_error

    def _close_blocking(self):
        if self._closed:
            return
        self._closed = True
        error = None
        if self._put_manager is not None:
            try:
                self._put_manager.shutdown()
            except Exception as exc:  # noqa: BLE001 -- continue deterministic cleanup
                error = exc
            self._put_manager = None
        if self._host_buffer_pool is not None:
            try:
                self._host_buffer_pool.close()
            except Exception as exc:  # noqa: BLE001 -- continue deterministic cleanup
                error = error or exc
            self._host_buffer_pool = None
        if self._get_executor is not None:
            self._get_executor.shutdown(wait=True, cancel_futures=False)
            self._get_executor = None
        with self._io_lock:
            status = self.client.close()
        if status not in (None, 0):
            error = error or RuntimeError(f"Mooncake close failed: {status}")
        if error is None:
            self.quarantined.clear()
            self._registered_buffers.clear()
        if error is not None:
            raise error

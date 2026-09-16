"""Registered tensor buffers and immutable, hard-pinned Mooncake objects."""

import hashlib
import math
import socket

import torch

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


def describe_tensors(prefix, tensors):
    result = {}
    for name in FIELDS:
        tensor = tensors[name]
        if not tensor.is_contiguous() or tensor.is_cuda:
            raise ValueError("Producer tensors must be contiguous CPU tensors")
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported feature dtype: {dtype}")
        size = tensor.numel() * tensor.element_size()
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": dtype,
            "nbytes": size,
            "sha256": tensor_digest(tensor),
            "chunks": [
                {
                    "key": f"{prefix}/{name}/{offset // CHUNK_BYTES}",
                    "offset": offset,
                    "nbytes": min(CHUNK_BYTES, size - offset),
                }
                for offset in range(0, size, CHUNK_BYTES)
            ],
        }
    return result


def object_keys(fields):
    return [chunk["key"] for spec in fields.values() for chunk in spec["chunks"]]


class TensorStore:
    def __init__(self, config, *, pool_bytes=0):
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        self.client = MooncakeDistributedStore()
        self.quarantined = []
        self.config = ReplicateConfig()
        self.config.replica_num = 1
        self.config.with_hard_pin = True
        self.config.dfs_replica_num = 0
        self.config.nof_replica_num = 0
        status = self.client.setup(
            local_hostname=f"{config['host']}:{free_port()}",
            metadata_server="P2PHANDSHAKE",
            global_segment_size=pool_bytes,
            local_buffer_size=16 * 1024**2,
            protocol=config["protocol"],
            rdma_devices=config.get("rdma_devices", ""),
            master_server_addr=config["master"],
            enable_ssd_offload=False,
        )
        if status != 0:
            raise RuntimeError(f"Mooncake setup failed: {status}")

    def _register(self, tensors):
        registered = []
        try:
            for tensor in tensors:
                status = self.client.register_buffer(
                    tensor.data_ptr(), tensor.numel() * tensor.element_size()
                )
                if status != 0:
                    raise RuntimeError(f"Mooncake buffer registration failed: {status}")
                registered.append(tensor)
        except BaseException:
            self.quarantined.extend(registered)
            raise
        return registered

    def _unregister(self, tensors):
        for tensor in tensors:
            status = self.client.unregister_buffer(tensor.data_ptr())
            if status != 0:
                self.quarantined.extend(tensors)
                raise RuntimeError(f"Mooncake unregister failed: {status}")

    def put(self, fields, tensors):
        buffers = self._register(list(tensors.values()))
        keys, pointers, sizes = [], [], []
        for name, spec in fields.items():
            for chunk in spec["chunks"]:
                keys.append(chunk["key"])
                pointers.append(tensors[name].data_ptr() + chunk["offset"])
                sizes.append(chunk["nbytes"])
        try:
            results = self.client.batch_put_from(keys, pointers, sizes, self.config)
            if len(results) != len(keys) or any(value != 0 for value in results):
                raise RuntimeError(f"Mooncake put failed: {results}")
        except BaseException:
            # A failed transfer does not authorize reuse of registered storage.
            self.quarantined.extend(buffers)
            raise
        self._unregister(buffers)

    def get(self, fields, names, *, device="cpu", verify=True):
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
            offset = 0
            for chunk in spec["chunks"]:
                if chunk["offset"] != offset or chunk["nbytes"] <= 0:
                    raise ValueError(f"Invalid tensor chunks: {name}")
                keys.append(chunk["key"])
                pointers.append(tensor.data_ptr() + offset)
                sizes.append(chunk["nbytes"])
                offset += chunk["nbytes"]
            if offset != spec["nbytes"]:
                raise ValueError(f"Incomplete tensor chunks: {name}")
            tensors[name] = tensor
        buffers = self._register(list(tensors.values()))
        try:
            results = self.client.batch_get_into(keys, pointers, sizes)
            if list(results) != sizes:
                raise RuntimeError(f"Mooncake get returned {results}, expected {sizes}")
        except BaseException:
            self.quarantined.extend(buffers)
            raise
        self._unregister(buffers)
        if verify:
            for name, tensor in tensors.items():
                if tensor_digest(tensor) != fields[name]["sha256"]:
                    raise ValueError(f"Transferred feature checksum mismatch: {name}")
        return tensors

    def remove(self, fields):
        keys = object_keys(fields)
        results = self.client.batch_remove(keys, force=True)
        if len(results) != len(keys) or any(value != 0 for value in results):
            raise RuntimeError(f"Mooncake remove failed: {results}")

    def close(self):
        status = self.client.close()
        if status != 0:
            raise RuntimeError(f"Mooncake close failed: {status}")
        self.quarantined.clear()

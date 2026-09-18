"""Reusable Mooncake transport primitives for the DeepSpec pipeline.

The package deliberately keeps the native Mooncake import lazy.  CPU-only
unit tests can exercise buffer ownership, backpressure and failure handling
without importing the transfer-engine extension.
"""

from .buffers import AsyncPutManager, HostBuffer, HostBufferPool, TransferHandle
from .deletion import DeleteManager

__all__ = [
    "AsyncPutManager",
    "DeleteManager",
    "HostBuffer",
    "HostBufferPool",
    "TransferHandle",
]

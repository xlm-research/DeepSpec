"""Versioned transport settings with a compatibility upgrade path.

Existing preparation files predate the explicit transport section.  Readers
accept those files as schema version 1 and materialize the version 2 defaults
without changing the descriptor or ledger fields used by the native trainer.
"""

from __future__ import annotations

from .store import CHUNK_BYTES

CURRENT_SCHEMA_VERSION = 2


def normalize_pipeline_config(config):
    """Upgrade a pipeline dictionary in place and return it.

    Keeping the old top-level keys is intentional: old launchers and manifests
    can be resumed while new components consume the grouped ``transport``
    settings.  A future schema can reject incompatible values here before any
    Ray actor or Mooncake client is started.
    """

    version = int(config.get("schema_version", 1))
    if version < 1 or version > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported DeepSpec pipeline schema {version}; "
            f"expected 1..{CURRENT_SCHEMA_VERSION}"
        )
    transport = config.setdefault("transport", {})
    if not isinstance(transport, dict):
        raise TypeError("pipeline transport settings must be an object")
    transport.setdefault("chunk_bytes", CHUNK_BYTES)
    transport.setdefault("verify_mode", "full" if config.get("verify_transfers", True) else "none")
    transport.setdefault("prefetch_depth", int(config.get("prefetch_depth", 2)))
    transport.setdefault("prefetch_bytes", config.get("prefetch_bytes"))
    transport.setdefault("async_put_pool_size", int(config.get("writer_inflight") or 1))
    transport.setdefault("wait_for_visibility", True)
    transport["chunk_bytes"] = int(transport["chunk_bytes"])
    transport["prefetch_depth"] = int(transport["prefetch_depth"])
    if transport["prefetch_bytes"] is not None:
        transport["prefetch_bytes"] = int(transport["prefetch_bytes"])
    transport["async_put_pool_size"] = int(transport["async_put_pool_size"])
    if transport["prefetch_depth"] < 1:
        raise ValueError("transport.prefetch_depth must be positive")
    if transport["verify_mode"] not in ("full", "none"):
        raise ValueError("transport.verify_mode must be 'full' or 'none'")
    if transport["prefetch_bytes"] is not None and transport["prefetch_bytes"] <= 0:
        raise ValueError("transport.prefetch_bytes must be positive")
    if transport["chunk_bytes"] <= 0:
        raise ValueError("transport.chunk_bytes must be positive")
    store = config.setdefault("store", {})
    if not isinstance(store, dict):
        raise TypeError("pipeline store settings must be an object")
    store.setdefault("async_put_pool_size", transport["async_put_pool_size"])
    store.setdefault("verify_mode", transport["verify_mode"])
    store.setdefault("wait_for_visibility", transport["wait_for_visibility"])
    config["prefetch_depth"] = transport["prefetch_depth"]
    if transport["prefetch_bytes"] is not None:
        config["prefetch_bytes"] = transport["prefetch_bytes"]
    config["schema_version"] = CURRENT_SCHEMA_VERSION
    return config

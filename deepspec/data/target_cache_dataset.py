"""Target-cache storage protocol, writers, dataset, collator, and validation."""

import hashlib
import json
import mmap
import os
import queue
import shutil
import struct
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from deepspec.data.parser import (
    MultimodalTruncationError,
    normalize_media_uri_map,
    preprocess_multimodal_record,
    preprocess_record,
)


def _debug_progress(message):
    if os.environ.get("DEEPSPEC_DEBUG_PROGRESS", "false").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        f"[rank={rank} local_rank={local_rank}] {message}",
        flush=True,
    )


LEGACY_TARGET_CACHE_VERSION = 3
TARGET_CACHE_VERSION = 4
SUPPORTED_TARGET_CACHE_VERSIONS = (
    LEGACY_TARGET_CACHE_VERSION,
    TARGET_CACHE_VERSION,
)
# sample id, shard id, full sequence length, local context start/length, offsets.
INDEX_RECORD_STRUCT = struct.Struct("<QIIIIQQQQQ")
INDEX_RECORD_SIZE = INDEX_RECORD_STRUCT.size

TARGET_CACHE_HIDDEN_DTYPE = "bfloat16"
TARGET_CACHE_TOKEN_DTYPE  = "int32"
TARGET_CACHE_MASK_DTYPE   = "uint8"


def atomic_json_dump(payload, path: str):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def compute_file_fingerprint(path: str, *, chunk_size: int = 8 * 1024 * 1024):
    """Return a stable content identity for a source JSONL file."""

    resolved_path = os.path.abspath(os.path.expanduser(str(path)))
    digest = hashlib.sha256()
    with open(resolved_path, "rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return {
        "path": resolved_path,
        "size": os.path.getsize(resolved_path),
        "sha256": digest.hexdigest(),
    }


def build_source_jsonl_fingerprints(paths):
    return [compute_file_fingerprint(path) for path in paths]


def build_target_cache_shard_path(cache_dir: str, file_name: str) -> str:
    return os.path.join(cache_dir, file_name)


def expected_target_cache_tensor_numel(
    *,
    seq_len: int,
    context_len: int | None = None,
    hidden_size: int,
    num_target_layers: int,
    stores_target_last_hidden_states: bool = True,
):
    context_len = int(seq_len) if context_len is None else int(context_len)
    return {
        "input_ids": int(seq_len),
        "attention_mask": int(seq_len),
        "loss_mask": int(seq_len),
        "target_hidden_states": context_len * int(num_target_layers) * int(hidden_size),
        "target_last_hidden_states": (
            context_len * int(hidden_size)
            if bool(stores_target_last_hidden_states)
            else 0
        ),
    }


def expected_target_cache_tensor_nbytes(
    *,
    seq_len: int,
    context_len: int | None = None,
    hidden_size: int,
    num_target_layers: int,
    stores_target_last_hidden_states: bool = True,
):
    numel = expected_target_cache_tensor_numel(
        seq_len=seq_len,
        context_len=context_len,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
        stores_target_last_hidden_states=stores_target_last_hidden_states,
    )
    return {
        "input_ids": numel["input_ids"] * 4,
        "attention_mask": numel["attention_mask"],
        "loss_mask": numel["loss_mask"],
        "target_hidden_states": numel["target_hidden_states"] * 2,
        "target_last_hidden_states": numel["target_last_hidden_states"] * 2,
    }


def pack_index_record(
    *,
    sample_id: int,
    shard_id: int,
    seq_len: int,
    context_start: int,
    context_len: int,
    input_ids_offset: int,
    attention_mask_offset: int,
    loss_mask_offset: int,
    target_hidden_states_offset: int,
    target_last_hidden_states_offset: int,
):
    return INDEX_RECORD_STRUCT.pack(
        int(sample_id),
        int(shard_id),
        int(seq_len),
        int(context_start),
        int(context_len),
        int(input_ids_offset),
        int(attention_mask_offset),
        int(loss_mask_offset),
        int(target_hidden_states_offset),
        int(target_last_hidden_states_offset),
    )


def unpack_index_record(buffer, offset: int = 0):
    (
        sample_id,
        shard_id,
        seq_len,
        context_start,
        context_len,
        input_ids_offset,
        attention_mask_offset,
        loss_mask_offset,
        target_hidden_states_offset,
        target_last_hidden_states_offset,
    ) = INDEX_RECORD_STRUCT.unpack_from(buffer, offset)
    return {
        "sample_id": sample_id,
        "shard_id": shard_id,
        "seq_len": seq_len,
        "context_start": context_start,
        "context_len": context_len,
        "input_ids_offset": input_ids_offset,
        "attention_mask_offset": attention_mask_offset,
        "loss_mask_offset": loss_mask_offset,
        "target_hidden_states_offset": target_hidden_states_offset,
        "target_last_hidden_states_offset": target_last_hidden_states_offset,
    }


def load_target_cache_manifest(cache_dir: str):
    manifest_path = os.path.join(cache_dir, "manifest.json")
    assert os.path.exists(manifest_path), f"Missing target cache manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_target_cache_manifest(cache_dir=cache_dir, manifest=manifest)
    return manifest


def validate_target_cache_manifest(*, cache_dir: str, manifest):
    required_fields = {
        "version",
        "num_samples",
        "num_shards",
        "target_layer_ids",
        "hidden_dtype",
        "token_dtype",
        "mask_dtype",
        "index_record_size",
        "hidden_size",
        "cache_context_parallel_size",
        "context_layout",
        "index_files",
        "shards",
    }
    missing = sorted(required_fields - set(manifest))
    assert not missing, (
        f"Target cache manifest is missing required fields {missing}: "
        f"{os.path.join(cache_dir, 'manifest.json')}"
    )
    version = int(manifest["version"])
    assert version in SUPPORTED_TARGET_CACHE_VERSIONS, (
        "Unsupported target cache manifest version: "
        f"{manifest['version']} not in {SUPPORTED_TARGET_CACHE_VERSIONS}"
    )
    if version >= TARGET_CACHE_VERSION:
        required_v4_fields = {
            "source_jsonl_fingerprints",
            "stores_target_last_hidden_states",
        }
        missing = sorted(required_v4_fields - set(manifest))
        assert not missing, (
            f"Target cache v{version} manifest is missing required fields "
            f"{missing}: {os.path.join(cache_dir, 'manifest.json')}"
        )
        assert isinstance(manifest["stores_target_last_hidden_states"], bool), (
            "stores_target_last_hidden_states must be a JSON boolean."
        )
        fingerprints = manifest["source_jsonl_fingerprints"]
        assert isinstance(fingerprints, list) and fingerprints, (
            "source_jsonl_fingerprints must be a non-empty list."
        )
        for fingerprint in fingerprints:
            assert {"path", "size", "sha256"}.issubset(fingerprint), (
                f"Invalid source fingerprint: {fingerprint!r}"
            )
            assert int(fingerprint["size"]) >= 0
            assert len(str(fingerprint["sha256"])) == 64
    assert manifest["hidden_dtype"] == TARGET_CACHE_HIDDEN_DTYPE, (
        "Unsupported hidden_dtype in target cache manifest: "
        f"{manifest['hidden_dtype']}"
    )
    assert manifest["token_dtype"] == TARGET_CACHE_TOKEN_DTYPE, (
        "Unsupported token_dtype in target cache manifest: "
        f"{manifest['token_dtype']}"
    )
    assert manifest["mask_dtype"] == TARGET_CACHE_MASK_DTYPE, (
        "Unsupported mask_dtype in target cache manifest: "
        f"{manifest['mask_dtype']}"
    )
    assert int(manifest["index_record_size"]) == INDEX_RECORD_SIZE, (
        "index_record_size does not match canonical target cache protocol: "
        f"{manifest['index_record_size']} != {INDEX_RECORD_SIZE}"
    )
    hidden_size = int(manifest["hidden_size"])
    assert hidden_size > 0, f"hidden_size must be positive, got {hidden_size}"
    target_layer_ids = [int(layer_id) for layer_id in manifest["target_layer_ids"]]
    assert target_layer_ids, "target_layer_ids must not be empty."
    assert target_layer_ids == sorted(target_layer_ids), (
        "target_layer_ids must be sorted in ascending order."
    )
    num_shards = int(manifest["num_shards"])
    assert num_shards == len(manifest["shards"]), (
        "num_shards does not match shard metadata count: "
        f"{num_shards} != {len(manifest['shards'])}"
    )
    for expected_shard_id, shard in enumerate(manifest["shards"]):
        assert int(shard["shard_id"]) == expected_shard_id, (
            "Target cache shard ids must be contiguous starting from 0: "
            f"expected {expected_shard_id}, got {shard['shard_id']}"
        )
        shard_path = build_target_cache_shard_path(cache_dir, shard["file_name"])
        assert os.path.exists(shard_path), f"Missing target cache shard file: {shard_path}"
    cache_cp_size = int(manifest["cache_context_parallel_size"])
    assert cache_cp_size > 0, (
        "cache_context_parallel_size must be positive, got "
        f"{cache_cp_size}."
    )
    context_layout = str(manifest["context_layout"])
    assert context_layout in ("contiguous", "native_head_tail"), (
        "Unsupported target-cache context layout: "
        f"{context_layout!r}."
    )
    index_files = [str(file_name) for file_name in manifest["index_files"]]
    assert len(index_files) == cache_cp_size, (
        "Target cache must contain one index per cached CP rank: "
        f"{len(index_files)} != {cache_cp_size}."
    )
    num_samples = int(manifest["num_samples"])
    for file_name in index_files:
        index_path = os.path.join(cache_dir, file_name)
        assert os.path.exists(index_path), (
            f"Missing target cache index file: {index_path}"
        )
        index_size = os.path.getsize(index_path)
        assert index_size % INDEX_RECORD_SIZE == 0, (
            f"{file_name} size is not a multiple of the canonical record size: "
            f"{index_size} % {INDEX_RECORD_SIZE} != 0"
        )
        assert index_size == num_samples * INDEX_RECORD_SIZE, (
            f"{file_name} size does not match manifest.num_samples: "
            f"{index_size} != {num_samples} * {INDEX_RECORD_SIZE}"
        )


def validate_train_cache(*, train_dataset, draft_model, target_model_name_or_path):
    manifest = train_dataset.manifest
    expected_layer_ids = [int(layer_id) for layer_id in draft_model.target_layer_ids]
    assert [int(layer_id) for layer_id in manifest["target_layer_ids"]] == expected_layer_ids, (
        "Target cache target_layer_ids do not match draft model configuration: "
        f"{manifest['target_layer_ids']} != {expected_layer_ids}"
    )
    assert int(manifest["hidden_size"]) == int(draft_model.config.hidden_size), (
        "Target cache hidden_size does not match draft model hidden size: "
        f"{manifest['hidden_size']} != {draft_model.config.hidden_size}"
    )
    cache_target_model_name = manifest["target_model_name_or_path"]
    assert str(cache_target_model_name) == str(target_model_name_or_path), (
        "Target cache target_model_name_or_path does not match training config: "
        f"{cache_target_model_name} != {target_model_name_or_path}"
    )
    expected_layout = getattr(
        draft_model.config,
        "target_context_layout",
        None,
    )
    if expected_layout is not None and train_dataset.context_parallel_size > 1:
        assert str(manifest["context_layout"]) == str(expected_layout), (
            "Target cache context_layout does not match the draft model: "
            f"{manifest['context_layout']!r} != {expected_layout!r}."
        )


def validate_target_cache_identity(
    *,
    cache_dir: str,
    source_jsonl_paths,
    target_model_name_or_path: str,
    target_layer_ids,
    chat_template: str,
    max_length: int,
    context_parallel_size: int,
    stores_target_last_hidden_states: bool,
):
    """Validate every input that makes a prepared cache reusable."""

    manifest = load_target_cache_manifest(cache_dir)
    assert str(manifest.get("target_model_name_or_path")) == str(
        target_model_name_or_path
    ), "Target model does not match cache identity."
    assert [int(value) for value in manifest["target_layer_ids"]] == [
        int(value) for value in target_layer_ids
    ], "Target layer ids do not match cache identity."
    assert str(manifest.get("chat_template")) == str(chat_template), (
        "Chat template does not match cache identity."
    )
    assert int(manifest.get("max_length", -1)) == int(max_length), (
        "Maximum sequence length does not match cache identity."
    )
    assert int(manifest["cache_context_parallel_size"]) == int(
        context_parallel_size
    ), "Context-parallel size does not match cache identity."
    assert bool(manifest.get("stores_target_last_hidden_states", True)) == bool(
        stores_target_last_hidden_states
    ), "Final-hidden-state storage mode does not match cache identity."
    cached_fingerprints = manifest.get("source_jsonl_fingerprints")
    assert cached_fingerprints, (
        "Cache predates source fingerprints and must be regenerated."
    )
    current_fingerprints = build_source_jsonl_fingerprints(source_jsonl_paths)
    assert len(cached_fingerprints) == len(current_fingerprints), (
        "Source file count does not match cache identity."
    )
    for cached, current in zip(cached_fingerprints, current_fingerprints):
        assert int(cached["size"]) == int(current["size"]) and str(
            cached["sha256"]
        ) == str(current["sha256"]), (
            "Source JSONL content does not match cache identity: "
            f"cached={cached}, current={current}"
        )
    return manifest


def _tensor_to_bytes(tensor: torch.Tensor, dtype: torch.dtype):
    cpu_tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    return cpu_tensor.numpy().tobytes()


def _tensor_to_bfloat16_bytes(tensor: torch.Tensor):
    cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return cpu_tensor.view(torch.uint16).numpy().tobytes()


def compute_local_sample_range(*, num_samples: int, rank: int, world_size: int):
    base = int(num_samples) // int(world_size)
    remainder = int(num_samples) % int(world_size)
    start = rank * base + min(rank, remainder)
    local_count = base + (1 if rank < remainder else 0)
    return start, start + local_count


def prepare_target_cache_output_dir(output_dir: str):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        existing = sorted(os.listdir(output_dir))
        if existing:
            raise FileExistsError(
                f"Target cache output dir is not empty: {output_dir}. "
                "Use a new output directory."
            )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "_tmp"), exist_ok=True)


@dataclass
class LocalCacheWriteSummary:
    global_rank: int
    context_parallel_rank: int
    source_sample_start: int
    source_sample_end: int
    num_local_samples: int
    num_local_shards: int
    local_shard_files: list[str]

    def to_json(self):
        return {
            "global_rank": self.global_rank,
            "context_parallel_rank": self.context_parallel_rank,
            "source_sample_start": self.source_sample_start,
            "source_sample_end": self.source_sample_end,
            "num_local_samples": self.num_local_samples,
            "num_local_shards": self.num_local_shards,
            "local_shard_files": list(self.local_shard_files),
        }


@dataclass(frozen=True)
class TargetCacheSampleBytes:
    sample_id: int
    seq_len: int
    context_start: int
    context_len: int
    input_ids: bytes
    attention_mask: bytes
    loss_mask: bytes
    target_hidden_states: bytes
    target_last_hidden_states: bytes


def build_target_cache_sample_bytes(
    *,
    sample_id: int,
    context_start: int,
    context_layout: str = "contiguous",
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    target_hidden_states: torch.Tensor,
    target_last_hidden_states: torch.Tensor | None,
):
    seq_len = int(input_ids.shape[0])
    context_start = int(context_start)
    context_len = int(target_hidden_states.shape[0])
    context_layout = str(context_layout)
    if context_layout not in ("contiguous", "native_head_tail"):
        raise ValueError(f"Unsupported context layout: {context_layout!r}.")
    if (
        target_last_hidden_states is not None
        and int(target_last_hidden_states.shape[0]) != context_len
    ):
        raise ValueError(
            "Target hidden tensors must have the same local context length: "
            f"{target_hidden_states.shape} versus "
            f"{target_last_hidden_states.shape}."
        )
    if context_start < 0 or context_len <= 0:
        raise ValueError(
            "A cache context shard must have a non-negative start and positive "
            f"length, got start={context_start}, length={context_len}."
        )
    if context_layout == "contiguous" and context_start + context_len > seq_len:
        raise ValueError(
            "A cache context shard extends beyond the source sequence: "
            f"{context_start} + {context_len} > {seq_len}."
        )
    if context_layout == "native_head_tail" and (
        context_start != 0 or context_len % 2 != 0
    ):
        raise ValueError(
            "Native head/tail cache shards use context_start=0 and an even "
            f"local length, got start={context_start}, length={context_len}."
        )
    return TargetCacheSampleBytes(
        sample_id=int(sample_id),
        seq_len=seq_len,
        context_start=context_start,
        context_len=context_len,
        input_ids=_tensor_to_bytes(input_ids, torch.int32),
        attention_mask=_tensor_to_bytes(attention_mask, torch.uint8),
        loss_mask=_tensor_to_bytes(loss_mask, torch.uint8),
        target_hidden_states=_tensor_to_bfloat16_bytes(target_hidden_states),
        target_last_hidden_states=(
            _tensor_to_bfloat16_bytes(target_last_hidden_states)
            if target_last_hidden_states is not None
            else b""
        ),
    )


class LocalTargetCacheWriter:
    def __init__(
        self,
        *,
        rank_dir: str,
        max_shard_bytes: int,
        context_layout: str = "contiguous",
    ):
        self.rank_dir = rank_dir
        self.max_shard_bytes = int(max_shard_bytes)
        self.local_index_path = os.path.join(rank_dir, "samples.local.idx")
        self.index_handle = open(self.local_index_path, "wb")
        self.current_shard_id = -1
        self.current_shard_handle = None
        self.current_shard_size = 0
        self.local_shard_files = []
        self.num_local_samples = 0
        self.context_layout = str(context_layout)

    def close(self):
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
            self.current_shard_handle = None
        if getattr(self, "index_handle", None) is not None:
            self.index_handle.flush()
            os.fsync(self.index_handle.fileno())
            self.index_handle.close()
            self.index_handle = None

    def _open_new_shard(self):
        if self.current_shard_handle is not None:
            self.current_shard_handle.flush()
            os.fsync(self.current_shard_handle.fileno())
            self.current_shard_handle.close()
        self.current_shard_id += 1
        file_name = f"shard-local-{self.current_shard_id:05d}.bin"
        shard_path = os.path.join(self.rank_dir, file_name)
        self.current_shard_handle = open(shard_path, "wb")
        self.current_shard_size = 0
        self.local_shard_files.append(file_name)

    def _ensure_shard(self, sample_nbytes: int):
        if self.current_shard_handle is None:
            self._open_new_shard()
            return
        if (
            self.current_shard_size > 0
            and self.current_shard_size + int(sample_nbytes) > self.max_shard_bytes
        ):
            self._open_new_shard()

    def write_sample_bytes(self, sample: TargetCacheSampleBytes):
        sample_nbytes = (
            len(sample.input_ids)
            + len(sample.attention_mask)
            + len(sample.loss_mask)
            + len(sample.target_hidden_states)
            + len(sample.target_last_hidden_states)
        )
        self._ensure_shard(sample_nbytes)
        input_ids_offset = self.current_shard_size
        self.current_shard_handle.write(sample.input_ids)
        self.current_shard_size += len(sample.input_ids)
        attention_mask_offset = self.current_shard_size
        self.current_shard_handle.write(sample.attention_mask)
        self.current_shard_size += len(sample.attention_mask)
        loss_mask_offset = self.current_shard_size
        self.current_shard_handle.write(sample.loss_mask)
        self.current_shard_size += len(sample.loss_mask)
        target_hidden_states_offset = self.current_shard_size
        self.current_shard_handle.write(sample.target_hidden_states)
        self.current_shard_size += len(sample.target_hidden_states)
        target_last_hidden_states_offset = self.current_shard_size
        self.current_shard_handle.write(sample.target_last_hidden_states)
        self.current_shard_size += len(sample.target_last_hidden_states)
        self.index_handle.write(
            pack_index_record(
                sample_id=sample.sample_id,
                shard_id=self.current_shard_id,
                seq_len=sample.seq_len,
                context_start=sample.context_start,
                context_len=sample.context_len,
                input_ids_offset=input_ids_offset,
                attention_mask_offset=attention_mask_offset,
                loss_mask_offset=loss_mask_offset,
                target_hidden_states_offset=target_hidden_states_offset,
                target_last_hidden_states_offset=target_last_hidden_states_offset,
            )
        )
        self.num_local_samples += 1

    def write_sample(
        self,
        *,
        sample_id: int,
        context_start: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: torch.Tensor | None,
    ):
        sample = build_target_cache_sample_bytes(
            sample_id=sample_id,
            context_start=context_start,
            context_layout=self.context_layout,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
        )
        self.write_sample_bytes(sample)


class AsyncTargetCacheWriter:
    def __init__(
        self,
        *,
        rank_dir: str,
        max_shard_bytes: int,
        max_queue_size: int = 128,
        context_layout: str = "contiguous",
    ):
        self.writer = LocalTargetCacheWriter(
            rank_dir=rank_dir,
            max_shard_bytes=max_shard_bytes,
            context_layout=context_layout,
        )
        # Queue CPU byte records only; never hold CUDA tensor references here.
        self.queue = queue.Queue(maxsize=int(max_queue_size))
        self.sentinel = object()
        self.num_local_samples = 0
        self._closed = False
        self._exception = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"target-cache-writer-{os.path.basename(rank_dir)}",
        )
        self.thread.start()

    @property
    def local_shard_files(self):
        return self.writer.local_shard_files

    def _run(self):
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self.sentinel:
                        break
                    self.writer.write_sample_bytes(item)
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self._exception = exc
        finally:
            try:
                self.writer.close()
            except BaseException as exc:
                if self._exception is None:
                    self._exception = exc

    def _raise_if_failed(self):
        if self._exception is not None:
            raise RuntimeError("Async target cache writer failed.") from self._exception

    def _put(self, item):
        while True:
            self._raise_if_failed()
            try:
                self.queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    def write_sample(
        self,
        *,
        context_start: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: torch.Tensor | None,
    ):
        sample = build_target_cache_sample_bytes(
            sample_id=self.num_local_samples,
            context_start=context_start,
            context_layout=self.writer.context_layout,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
        )
        self._put(sample)
        self.num_local_samples += 1

    def close(self):
        if self._closed:
            self._raise_if_failed()
            return
        if self._exception is None:
            self._put(self.sentinel)
        self.thread.join()
        self._closed = True
        self._raise_if_failed()
        assert self.writer.num_local_samples == self.num_local_samples, (
            "Async target cache writer lost samples: "
            f"{self.writer.num_local_samples} != {self.num_local_samples}"
        )


def load_local_cache_write_summary(rank_dir: str):
    with open(os.path.join(rank_dir, "summary.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_global_target_cache_shard_map(summaries):
    shard_map = {}
    shards = []
    next_shard_id = 0
    for summary in sorted(
        summaries,
        key=lambda item: (
            int(item["source_sample_start"]),
            int(item["context_parallel_rank"]),
            int(item["global_rank"]),
        ),
    ):
        local_map = []
        for _local_shard_id, _file_name in enumerate(summary["local_shard_files"]):
            local_map.append(next_shard_id)
            shards.append(
                {
                    "shard_id": next_shard_id,
                    "file_name": f"shard-{next_shard_id:05d}.bin",
                }
            )
            next_shard_id += 1
        shard_map[int(summary["global_rank"])] = local_map
    return shard_map, shards


def rename_local_target_cache_shards(*, output_dir: str, rank_dir: str, summary, shard_map):
    local_map = shard_map[int(summary["global_rank"])]
    for local_shard_id, file_name in enumerate(summary["local_shard_files"]):
        source = os.path.join(rank_dir, file_name)
        target = os.path.join(output_dir, f"shard-{local_map[local_shard_id]:05d}.bin")
        os.replace(source, target)


def finalize_target_cache_indices(
    *,
    output_dir: str,
    summaries,
    shard_map,
    context_parallel_size: int,
    context_layout: str = "contiguous",
):
    """Build one dense logical-sample index for every cached CP rank."""

    index_files = []
    per_cp_records = []
    for context_parallel_rank in range(int(context_parallel_size)):
        file_name = f"samples.cp{context_parallel_rank:03d}.idx"
        index_files.append(file_name)
        index_tmp_path = os.path.join(output_dir, f"{file_name}.tmp")
        records_for_cp = []
        next_expected_sample_id = 0
        cp_summaries = [
            summary
            for summary in summaries
            if int(summary["context_parallel_rank"]) == context_parallel_rank
        ]
        with open(index_tmp_path, "wb") as output_handle:
            for summary in sorted(
                cp_summaries,
                key=lambda item: int(item["source_sample_start"]),
            ):
                rank_dir = os.path.join(
                    output_dir,
                    "_tmp",
                    f"rank_{int(summary['global_rank'])}",
                )
                local_index_path = os.path.join(rank_dir, "samples.local.idx")
                with open(local_index_path, "rb") as local_handle:
                    local_bytes = local_handle.read()
                assert len(local_bytes) % INDEX_RECORD_SIZE == 0, (
                    "Local target cache index has invalid size: "
                    f"{local_index_path}"
                )
                next_local_sample_id = 0
                for offset in range(0, len(local_bytes), INDEX_RECORD_SIZE):
                    record = unpack_index_record(local_bytes, offset)
                    assert int(record["sample_id"]) == next_local_sample_id, (
                        "Local target cache index is not ordered by local sample_id: "
                        f"got {record['sample_id']}, expected "
                        f"{next_local_sample_id}"
                    )
                    record["sample_id"] = next_expected_sample_id
                    record["shard_id"] = shard_map[int(summary["global_rank"])][
                        int(record["shard_id"])
                    ]
                    output_handle.write(pack_index_record(**record))
                    records_for_cp.append(
                        (
                            int(record["seq_len"]),
                            int(record["context_start"]),
                            int(record["context_len"]),
                        )
                    )
                    next_local_sample_id += 1
                    next_expected_sample_id += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(index_tmp_path, os.path.join(output_dir, file_name))
        per_cp_records.append(records_for_cp)

    sample_counts = [len(records) for records in per_cp_records]
    if len(set(sample_counts)) != 1:
        raise RuntimeError(
            "CP ranks wrote different logical sample counts: "
            f"{sample_counts}."
        )
    num_samples = sample_counts[0]
    context_layout = str(context_layout)
    if context_layout not in ("contiguous", "native_head_tail"):
        raise ValueError(f"Unsupported context layout: {context_layout!r}.")
    for sample_id in range(num_samples):
        seq_lens = [records[sample_id][0] for records in per_cp_records]
        if len(set(seq_lens)) != 1:
            raise RuntimeError(
                f"CP cache sample {sample_id} has inconsistent sequence "
                f"lengths: {seq_lens}."
            )
        if context_layout == "contiguous":
            intervals = sorted(
                (records[sample_id][1], records[sample_id][2])
                for records in per_cp_records
            )
            next_start = 0
            for context_start, context_len in intervals:
                if context_start != next_start:
                    raise RuntimeError(
                        f"CP cache sample {sample_id} has a gap or overlap at "
                        f"token {next_start}: intervals={intervals}."
                    )
                next_start += context_len
            if next_start != seq_lens[0]:
                raise RuntimeError(
                    f"CP cache sample {sample_id} covers {next_start} tokens but "
                    f"the sequence contains {seq_lens[0]}."
                )
            continue

        starts = [records[sample_id][1] for records in per_cp_records]
        local_lengths = [records[sample_id][2] for records in per_cp_records]
        if any(start != 0 for start in starts):
            raise RuntimeError(
                f"Native head/tail sample {sample_id} must store zero context "
                f"starts, got {starts}."
            )
        if len(set(local_lengths)) != 1 or local_lengths[0] % 2 != 0:
            raise RuntimeError(
                f"Native head/tail sample {sample_id} requires equal even local "
                f"lengths, got {local_lengths}."
            )
        padded_length = local_lengths[0] * int(context_parallel_size)
        padding = padded_length - seq_lens[0]
        if padding < 0 or padding >= 2 * int(context_parallel_size):
            raise RuntimeError(
                f"Native head/tail sample {sample_id} has invalid padded "
                f"length {padded_length} for sequence length {seq_lens[0]}."
            )
    return num_samples, index_files


def build_target_cache_manifest(
    *,
    num_samples: int,
    shards,
    target_layer_ids,
    hidden_size: int,
    extra_fields=None,
):
    extra_fields = dict(extra_fields or {})
    # ponytail-lite: legacy callers stay on v3; explicit storage metadata
    # opts newly prepared caches into v4 without changing old callers.
    version = (
        TARGET_CACHE_VERSION
        if "stores_target_last_hidden_states" in extra_fields
        else LEGACY_TARGET_CACHE_VERSION
    )
    manifest = {
        "version": version,
        "num_samples": int(num_samples),
        "num_shards": len(shards),
        "target_layer_ids": [int(layer_id) for layer_id in target_layer_ids],
        "hidden_dtype": TARGET_CACHE_HIDDEN_DTYPE,
        "token_dtype": TARGET_CACHE_TOKEN_DTYPE,
        "mask_dtype": TARGET_CACHE_MASK_DTYPE,
        "index_record_size": INDEX_RECORD_SIZE,
        "hidden_size": int(hidden_size),
        "shards": shards,
    }
    manifest.update(extra_fields)
    return manifest


def write_target_cache_manifest(*, output_dir: str, manifest):
    atomic_json_dump(manifest, os.path.join(output_dir, "manifest.json"))


def cleanup_target_cache_tmp_dir(output_dir: str):
    tmp_dir = os.path.join(output_dir, "_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


class CacheDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: str,
        max_open_shards: int = 4,
        context_parallel_size: int = 1,
        context_parallel_rank: int = 0,
        expected_context_layout: str | None = None,
    ):
        super().__init__()
        self.cache_dir = os.path.abspath(cache_dir)
        self.manifest = load_target_cache_manifest(self.cache_dir)
        self.num_samples = int(self.manifest["num_samples"])
        self.hidden_size = int(self.manifest["hidden_size"])
        self.target_layer_ids = [int(layer_id) for layer_id in self.manifest["target_layer_ids"]]
        self.num_target_layers = len(self.target_layer_ids)
        # Version 3 caches always stored this tensor. Version 4 can omit
        # it for CE-only methods such as DFlash2.
        self.stores_target_last_hidden_states = bool(
            self.manifest.get("stores_target_last_hidden_states", True)
        )
        self.context_parallel_size = int(context_parallel_size)
        self.context_parallel_rank = int(context_parallel_rank)
        if self.context_parallel_size < 1:
            raise ValueError("context_parallel_size must be positive.")
        if not 0 <= self.context_parallel_rank < self.context_parallel_size:
            raise ValueError(
                "context_parallel_rank must be in [0, context_parallel_size), "
                f"got {self.context_parallel_rank} and "
                f"{self.context_parallel_size}."
            )
        cache_cp_size = int(self.manifest["cache_context_parallel_size"])
        if self.context_parallel_size != cache_cp_size:
            raise ValueError(
                "Training CP size must match the rank-sharded target cache: "
                f"{self.context_parallel_size} != {cache_cp_size}. Regenerate "
                "the target cache for the requested CP size."
            )
        self.context_layout = str(self.manifest["context_layout"])
        if self.context_layout not in ("contiguous", "native_head_tail"):
            raise ValueError(
                "Training received an unsupported target-cache layout: "
                f"{self.context_layout!r}."
            )
        if (
            expected_context_layout is not None
            and self.context_layout != str(expected_context_layout)
        ):
            raise ValueError(
                "Target cache context layout is incompatible with training CP: "
                f"{self.context_layout!r} != {expected_context_layout!r}."
            )
        self.index_path = os.path.join(
            self.cache_dir,
            self.manifest["index_files"][self.context_parallel_rank],
        )
        self.index_file = None
        self.index_mmap = None
        self.max_open_shards = max_open_shards
        self.shard_handles = OrderedDict()
        self.shard_mmaps = OrderedDict()
        self.shard_paths = {
            int(shard["shard_id"]): build_target_cache_shard_path(
                self.cache_dir,
                shard["file_name"],
            )
            for shard in self.manifest["shards"]
        }

    def __len__(self):
        return self.num_samples

    def get_length_hint(self, index: int):
        """Return the cached sequence length without reading tensor payloads."""

        if not (0 <= int(index) < self.num_samples):
            raise IndexError(index)
        return int(self._read_record(int(index))["seq_len"])

    def close(self):
        for shard_mmap in getattr(self, "shard_mmaps", {}).values():
            shard_mmap.close()
        for handle in getattr(self, "shard_handles", {}).values():
            handle.close()
        if hasattr(self, "shard_mmaps"):
            self.shard_mmaps.clear()
        if hasattr(self, "shard_handles"):
            self.shard_handles.clear()
        if getattr(self, "index_mmap", None) is not None:
            self.index_mmap.close()
            self.index_mmap = None
        if getattr(self, "index_file", None) is not None:
            self.index_file.close()
            self.index_file = None

    def __del__(self):  # pragma: no cover
        self.close()

    def __getstate__(self):  # pragma: no cover
        state = dict(self.__dict__)
        state["index_file"] = None
        state["index_mmap"] = None
        state["shard_handles"] = OrderedDict()
        state["shard_mmaps"] = OrderedDict()
        return state

    def _ensure_index_mmap(self):
        if self.index_mmap is None:
            self.index_file = open(self.index_path, "rb")
            self.index_mmap = mmap.mmap(
                self.index_file.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )

    def _get_shard_mmap(self, shard_id: int):
        shard_id = int(shard_id)
        if shard_id in self.shard_mmaps:
            self.shard_mmaps.move_to_end(shard_id)
            self.shard_handles.move_to_end(shard_id)
            return self.shard_mmaps[shard_id]
        shard_path = self.shard_paths[shard_id]
        handle = open(shard_path, "rb")
        self.shard_handles[shard_id] = handle
        self.shard_mmaps[shard_id] = mmap.mmap(
            handle.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )
        while len(self.shard_mmaps) > self.max_open_shards:
            evicted_id, evicted_mmap = self.shard_mmaps.popitem(last=False)
            evicted_mmap.close()
            self.shard_handles.pop(evicted_id).close()
        return self.shard_mmaps[shard_id]

    def _read_record(self, index: int):
        self._ensure_index_mmap()
        offset = int(index) * INDEX_RECORD_SIZE
        record = unpack_index_record(self.index_mmap, offset)
        assert int(record["sample_id"]) == int(index), (
            "Target cache index is not sorted by sample_id or sample ids are not dense: "
            f"record sample_id={record['sample_id']}, expected {index}"
        )
        return record

    def _read_tensor_from_shard(
        self,
        *,
        shard_mmap,
        offset: int,
        shape,
        np_dtype,
        torch_dtype,
        nbytes: int,
    ):
        assert int(offset) + int(nbytes) <= shard_mmap.size(), (
            "Target cache tensor extends beyond shard size: "
            f"offset={offset}, nbytes={nbytes}, shard_size={shard_mmap.size()}"
        )
        array = np.frombuffer(
            shard_mmap,
            dtype=np_dtype,
            count=int(np.prod(shape)),
            offset=int(offset),
        ).copy()
        tensor = torch.from_numpy(array).view(*shape)
        if tensor.dtype != torch_dtype:
            tensor = tensor.to(dtype=torch_dtype)
        return tensor

    def _read_bfloat16_tensor_from_shard(
        self,
        *,
        shard_mmap,
        offset: int,
        shape,
        nbytes: int,
    ):
        assert int(offset) + int(nbytes) <= shard_mmap.size(), (
            "Target cache tensor extends beyond shard size: "
            f"offset={offset}, nbytes={nbytes}, shard_size={shard_mmap.size()}"
        )
        array = np.frombuffer(
            shard_mmap,
            dtype=np.uint16,
            count=int(np.prod(shape)),
            offset=int(offset),
        ).copy()
        tensor = torch.from_numpy(array).view(torch.bfloat16)
        return tensor.view(*shape)

    def __getitem__(self, index: int):
        if not (0 <= int(index) < self.num_samples):
            raise IndexError(index)
        record = self._read_record(int(index))
        seq_len = int(record["seq_len"])
        context_start = int(record["context_start"])
        context_len = int(record["context_len"])
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        assert context_start >= 0 and context_len > 0, (
            "Invalid context shard: "
            f"start={context_start}, length={context_len}."
        )
        if self.context_layout == "contiguous":
            assert context_start + context_len <= seq_len
        else:
            assert context_start == 0 and context_len % 2 == 0
            assert context_len * self.context_parallel_size >= seq_len
        shard_mmap = self._get_shard_mmap(int(record["shard_id"]))
        nbytes = expected_target_cache_tensor_nbytes(
            seq_len=seq_len,
            context_len=context_len,
            hidden_size=self.hidden_size,
            num_target_layers=self.num_target_layers,
            stores_target_last_hidden_states=(
                self.stores_target_last_hidden_states
            ),
        )
        input_ids = self._read_tensor_from_shard(
            shard_mmap=shard_mmap,
            offset=record["input_ids_offset"],
            shape=(seq_len,),
            np_dtype=np.int32,
            torch_dtype=torch.int32,
            nbytes=nbytes["input_ids"],
        )
        loss_mask = self._read_tensor_from_shard(
            shard_mmap=shard_mmap,
            offset=record["loss_mask_offset"],
            shape=(seq_len,),
            np_dtype=np.uint8,
            torch_dtype=torch.uint8,
            nbytes=nbytes["loss_mask"],
        )
        target_hidden_states = self._read_bfloat16_tensor_from_shard(
            shard_mmap=shard_mmap,
            offset=record["target_hidden_states_offset"],
            shape=(context_len, self.num_target_layers * self.hidden_size),
            nbytes=nbytes["target_hidden_states"],
        )
        target_last_hidden_states = None
        if self.stores_target_last_hidden_states:
            target_last_hidden_states = self._read_bfloat16_tensor_from_shard(
                shard_mmap=shard_mmap,
                offset=record["target_last_hidden_states_offset"],
                shape=(context_len, self.hidden_size),
                nbytes=nbytes["target_last_hidden_states"],
            )
        sample = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": target_hidden_states,
            "context_start": context_start,
            "context_len": context_len,
            "seq_len": seq_len,
        }
        if self.context_layout == "contiguous":
            base, remainder = divmod(seq_len, self.context_parallel_size)
            expected_context_start = (
                self.context_parallel_rank * base
                + min(self.context_parallel_rank, remainder)
            )
            expected_context_len = base + int(
                self.context_parallel_rank < remainder
            )
            if (
                context_start != expected_context_start
                or context_len != expected_context_len
            ):
                raise RuntimeError(
                    "Contiguous target-cache shard does not match its CP "
                    "partition: "
                    f"cached=[{context_start}, {context_start + context_len}), "
                    f"expected=[{expected_context_start}, "
                    f"{expected_context_start + expected_context_len})."
                )
        if target_last_hidden_states is not None:
            sample["target_last_hidden_states"] = target_last_hidden_states
        return sample


def _pad_1d_batch(features: List[Dict], key: str):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    dtype = features[0][key].dtype
    out = torch.zeros((batch_size, max_length), dtype=dtype)
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]
    return out


def _pad_hidden_batch(features: List[Dict], key: str):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    hidden_dim = features[0][key].shape[1]
    dtype = features[0][key].dtype
    out = torch.zeros((batch_size, max_length, hidden_dim), dtype=dtype)
    for i, item in enumerate(features):
        seq_len = item[key].shape[0]
        out[i, :seq_len] = item[key]
    return out


class ConversationCollator:
    def __init__(
        self,
        tokenizer,
        chat_template,
        max_length,
        min_loss_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = int(max_length)
        self.min_loss_tokens = int(min_loss_tokens)

    def _process_packed_feature(self, item):
        conversations = item.get("packed_conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(
                "packed_conversations must be a non-empty list of "
                "conversation message lists."
            )
        pieces = []
        remaining_length = self.max_length
        for conversation in conversations:
            if not isinstance(conversation, list) or not conversation:
                raise ValueError(
                    "Each packed_conversations entry must be a non-empty "
                    "message list."
                )
            piece = preprocess_record(
                record={"conversations": conversation},
                tokenizer=self.tokenizer,
                chat_template=self.chat_template,
                max_length=remaining_length,
            )
            pieces.append(piece)
            remaining_length -= int(piece["input_ids"].shape[0])
            if remaining_length <= 0:
                break
        return {
            key: torch.cat([piece[key] for piece in pieces], dim=0)
            for key in ("input_ids", "attention_mask", "loss_mask")
        }

    def _process_feature(self, item):
        _debug_progress("collator: preprocess_record start")
        if "packed_conversations" in item:
            processed = self._process_packed_feature(item)
        else:
            processed = preprocess_record(
                record=item,
                tokenizer=self.tokenizer,
                chat_template=self.chat_template,
                max_length=self.max_length,
            )
        loss_tokens = int(processed["loss_mask"].sum().item())
        _debug_progress(
            "collator: preprocess_record done "
            f"seq_len={processed['input_ids'].shape[0]} loss_tokens={loss_tokens}"
        )
        if loss_tokens < self.min_loss_tokens:
            _debug_progress(
                "collator: dropping sample with too few loss tokens "
                f"loss_tokens={loss_tokens} min_loss_tokens={self.min_loss_tokens}"
            )
            return None
        return processed

    def __call__(self, features: List[Dict]):
        _debug_progress(f"collator: received {len(features)} raw feature(s)")
        features = [self._process_feature(item) for item in features]
        features = [item for item in features if item is not None]
        _debug_progress(f"collator: kept {len(features)} feature(s) after filtering")
        if not features:
            _debug_progress("collator: returning None because all features were filtered")
            return None
        batch = {}
        for key in ("input_ids", "attention_mask", "loss_mask"):
            batch[key] = _pad_1d_batch(features, key)
        _debug_progress(
            "collator: built CPU batch "
            + ", ".join(
                f"{key}=shape{tuple(value.shape)} dtype={value.dtype}"
                for key, value in batch.items()
            )
        )
        return batch


def _pad_multimodal_sequence_batch(
    features: List[Dict],
    key: str,
    *,
    padding_side: str,
    padding_value: int,
):
    max_length = max(item[key].shape[0] for item in features)
    batch_size = len(features)
    dtype = features[0][key].dtype
    trailing_shape = tuple(features[0][key].shape[1:])
    if not all(tuple(item[key].shape[1:]) == trailing_shape for item in features):
        raise ValueError(
            f"Processor sequence field {key!r} has inconsistent trailing shapes."
        )
    out = torch.full(
        (batch_size, max_length, *trailing_shape),
        fill_value=padding_value,
        dtype=dtype,
    )
    for index, item in enumerate(features):
        seq_len = item[key].shape[0]
        if padding_side == "left":
            out[index, max_length - seq_len :] = item[key]
        else:
            out[index, :seq_len] = item[key]
    return out


class MultimodalConversationCollator:
    """Collate processor tensors without flattening visual inputs."""

    def __init__(
        self,
        processor,
        chat_template,
        max_length,
        min_loss_tokens: int,
        media_root=None,
        media_uri_map=None,
    ):
        self.processor = processor
        self.chat_template = chat_template
        self.max_length = int(max_length)
        self.min_loss_tokens = int(min_loss_tokens)
        self.media_root = media_root
        self.media_uri_map = normalize_media_uri_map(media_uri_map)
        tokenizer = processor.tokenizer
        self.padding_side = str(getattr(tokenizer, "padding_side", "right"))
        if self.padding_side not in ("left", "right"):
            raise ValueError(
                f"Unsupported tokenizer padding_side: {self.padding_side}"
            )
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def _process_feature(self, item):
        try:
            processed = preprocess_multimodal_record(
                record=item,
                processor=self.processor,
                chat_template=self.chat_template,
                max_length=self.max_length,
                media_root=self.media_root,
                media_uri_map=self.media_uri_map,
            )
        except MultimodalTruncationError as exc:
            warnings.warn(
                f"Skipping truncated multimodal sample: {exc}",
                stacklevel=2,
            )
            return None
        if int(processed["loss_mask"].sum().item()) < self.min_loss_tokens:
            return None
        return processed

    def __call__(self, features: List[Dict]):
        features = [self._process_feature(item) for item in features]
        features = [item for item in features if item is not None]
        if not features:
            return None

        batch = {}
        all_keys = set().union(*(item.keys() for item in features))
        for key in sorted(all_keys):
            values = [item[key] for item in features if key in item]
            if not values or not all(
                isinstance(value, torch.Tensor) for value in values
            ):
                continue
            is_sequence_field = len(values) == len(features) and all(
                value.ndim >= 1
                and value.shape[0] == item["input_ids"].shape[0]
                for item, value in zip(features, values)
            )
            if is_sequence_field:
                padding_value = self.pad_token_id if key == "input_ids" else 0
                batch[key] = _pad_multimodal_sequence_batch(
                    features,
                    key,
                    padding_side=self.padding_side,
                    padding_value=padding_value,
                )
                continue
            if any(value.ndim == 0 for value in values):
                batch[key] = torch.stack(values)
                continue
            trailing_shape = tuple(values[0].shape[1:])
            if not all(
                tuple(value.shape[1:]) == trailing_shape for value in values
            ):
                raise ValueError(
                    f"Cannot collate processor field {key!r} with shapes "
                    f"{[tuple(value.shape) for value in values]}. Add a thin "
                    "TargetModelAdapter for this model family."
                )
            batch[key] = torch.cat(values, dim=0)
        return batch


class CacheCollator:
    def __call__(self, features: List[Dict]):
        batch = {}
        for key in ("input_ids", "loss_mask"):
            batch[key] = _pad_1d_batch(features, key)
        attention_mask = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        for i, item in enumerate(features):
            attention_mask[i, : item["input_ids"].shape[0]] = 1
        batch["attention_mask"] = attention_mask
        batch["target_hidden_states"] = _pad_hidden_batch(
            features, "target_hidden_states"
        )
        has_target_last = [
            "target_last_hidden_states" in feature for feature in features
        ]
        if any(has_target_last) and not all(has_target_last):
            raise ValueError(
                "A cache batch cannot mix samples with and without target "
                "final hidden states."
            )
        if all(has_target_last):
            batch["target_last_hidden_states"] = _pad_hidden_batch(
                features, "target_last_hidden_states"
            )
        for key in ("context_start", "context_len", "seq_len"):
            batch[key] = torch.tensor([int(item[key]) for item in features])
        return batch

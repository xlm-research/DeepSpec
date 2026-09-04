"""FSDP2-aware DCP loading for native GLM-5 Hugging Face checkpoints."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, ReadItem
from torch.distributed.tensor import DTensor


_GLM5_RENAMES = (
    (".self_attn.forget_gate.f_a_proj.", ".self_attn.f_a_proj."),
    (".self_attn.forget_gate.f_b_proj.", ".self_attn.f_b_proj."),
    (".self_attn.forget_gate.dt_bias", ".self_attn.dt_bias"),
    (".self_attn.forget_gate.A_log", ".self_attn.A_log"),
    (".attn_hc.fn", ".hc_attn_fn"),
    (".attn_hc.base", ".hc_attn_base"),
    (".attn_hc.scale", ".hc_attn_scale"),
    (".ffn_hc.fn", ".hc_ffn_fn"),
    (".ffn_hc.base", ".hc_ffn_base"),
    (".ffn_hc.scale", ".hc_ffn_scale"),
)


def _checkpoint_fqn(model_fqn: str) -> str:
    checkpoint_fqn = f"model.{model_fqn}"
    for model_name, checkpoint_name in _GLM5_RENAMES:
        checkpoint_fqn = checkpoint_fqn.replace(model_name, checkpoint_name)
    return checkpoint_fqn


def _local_chunks(
    tensor: torch.Tensor,
) -> list[tuple[ChunkStorageMetadata, torch.Tensor]]:
    if isinstance(tensor, DTensor):
        if tensor.device_mesh.get_coordinate() is None:
            return []
        chunks = tensor.__create_chunk_list__()
        if len(chunks) != 1:
            raise RuntimeError(
                "GLM-5 DCP loading expects one local DTensor chunk per rank, got "
                f"{len(chunks)}."
            )
        local_tensor = tensor.to_local()
    else:
        chunks = [
            ChunkStorageMetadata(
                offsets=torch.Size([0] * tensor.ndim),
                sizes=tensor.size(),
            )
        ]
        local_tensor = tensor

    result = []
    for chunk in chunks:
        view = local_tensor
        for dim, size in enumerate(chunk.sizes):
            if int(view.shape[dim]) < int(size):
                raise RuntimeError(
                    "DCP destination shard is smaller than its metadata: "
                    f"tensor={tuple(view.shape)}, chunk={tuple(chunk.sizes)}."
                )
            if int(view.shape[dim]) != int(size):
                view = view.narrow(dim, 0, int(size))
        result.append((chunk, view))
    return result


@dataclass(frozen=True)
class Glm5CheckpointTopology:
    tensor_parallel_rank: int
    tensor_parallel_size: int
    expert_parallel_rank: int
    expert_parallel_size: int


@dataclass(frozen=True)
class _Glm5HuggingFaceStorageInfo:
    relative_path: str
    shape: torch.Size
    dtype: torch.dtype


class Glm5HuggingFaceLoadPlanner(DefaultLoadPlanner):
    """Map native GLM-5 tensors directly into post-TP/EP FSDP2 shards."""

    def __init__(self, topology: Glm5CheckpointTopology):
        super().__init__(
            flatten_state_dict=False,
            flatten_sharded_tensors=False,
            allow_partial_load=False,
        )
        self.topology = topology
        self._destination_views: dict[str, torch.Tensor] = {}
        self._read_item_index = 0

    def _tensor_metadata(self, checkpoint_fqn: str) -> TensorStorageMetadata:
        if self.metadata is None:
            raise RuntimeError("DCP checkpoint metadata is not initialized.")
        metadata = self.metadata.state_dict_metadata.get(checkpoint_fqn)
        if not isinstance(metadata, TensorStorageMetadata):
            raise RuntimeError(
                f"Missing tensor {checkpoint_fqn!r} in the GLM-5 checkpoint."
            )
        return metadata

    def _add_read(
        self,
        *,
        checkpoint_fqn: str,
        checkpoint_offsets,
        destination: torch.Tensor,
    ) -> ReadItem:
        metadata = self._tensor_metadata(checkpoint_fqn)
        offsets = torch.Size(checkpoint_offsets)
        lengths = destination.size()
        if len(offsets) != len(metadata.size) or len(lengths) != len(metadata.size):
            raise RuntimeError(
                f"Invalid DCP slice rank for {checkpoint_fqn}: "
                f"checkpoint={tuple(metadata.size)}, offsets={tuple(offsets)}, "
                f"destination={tuple(lengths)}."
            )
        for offset, length, size in zip(offsets, lengths, metadata.size):
            if int(offset) < 0 or int(offset) + int(length) > int(size):
                raise RuntimeError(
                    f"DCP slice exceeds {checkpoint_fqn}: "
                    f"checkpoint={tuple(metadata.size)}, offsets={tuple(offsets)}, "
                    f"lengths={tuple(lengths)}."
                )

        destination_fqn = f"__deepspec_glm5_dcp_item_{self._read_item_index}"
        self._read_item_index += 1
        self._destination_views[destination_fqn] = destination
        return ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(destination_fqn),
            dest_offsets=torch.Size([0] * destination.ndim),
            storage_index=MetadataIndex(
                checkpoint_fqn,
                torch.Size([0] * len(metadata.size)),
            ),
            storage_offsets=offsets,
            lengths=lengths,
        )

    def _add_full_tensor_read(
        self, *, checkpoint_fqn: str, destination: torch.Tensor
    ) -> ReadItem:
        checkpoint_shape = self._tensor_metadata(checkpoint_fqn).size
        if checkpoint_shape != destination.size():
            raise RuntimeError(
                f"Invalid source shape for {checkpoint_fqn}: "
                f"checkpoint={tuple(checkpoint_shape)}, "
                f"destination={tuple(destination.size())}."
            )
        return self._add_read(
            checkpoint_fqn=checkpoint_fqn,
            checkpoint_offsets=[0] * destination.ndim,
            destination=destination,
        )

    def _tp_offsets(
        self,
        *,
        model_fqn: str,
        checkpoint_shape: torch.Size,
        model_shape: torch.Size,
    ) -> torch.Size:
        if len(checkpoint_shape) != len(model_shape):
            raise RuntimeError(
                f"Checkpoint rank mismatch for {model_fqn}: "
                f"checkpoint={tuple(checkpoint_shape)}, model={tuple(model_shape)}."
            )
        sharded_dimensions = []
        offsets = [0] * len(model_shape)
        tp_size = int(self.topology.tensor_parallel_size)
        tp_rank = int(self.topology.tensor_parallel_rank)
        for dim, (checkpoint_size, model_size) in enumerate(
            zip(checkpoint_shape, model_shape)
        ):
            if int(checkpoint_size) == int(model_size):
                continue
            if int(checkpoint_size) == int(model_size) * tp_size:
                sharded_dimensions.append(dim)
                offsets[dim] = tp_rank * int(model_size)
                continue
            raise RuntimeError(
                f"Unsupported TP layout for {model_fqn}: "
                f"checkpoint={tuple(checkpoint_shape)}, model={tuple(model_shape)}, "
                f"tp={tp_size}."
            )
        if len(sharded_dimensions) > 1:
            raise RuntimeError(
                f"GLM-5 tensor {model_fqn} is sharded across multiple TP "
                f"dimensions: {sharded_dimensions}."
            )
        return torch.Size(offsets)

    def _ordinary_reads(self, model_fqn: str, tensor: torch.Tensor) -> list[ReadItem]:
        checkpoint_fqn = _checkpoint_fqn(model_fqn)
        metadata = self._tensor_metadata(checkpoint_fqn)
        tp_offsets = self._tp_offsets(
            model_fqn=model_fqn,
            checkpoint_shape=metadata.size,
            model_shape=tensor.size(),
        )
        reads = []
        for chunk, destination in _local_chunks(tensor):
            if destination.numel() == 0:
                continue
            checkpoint_offsets = [
                int(tp_offset) + int(chunk_offset)
                for tp_offset, chunk_offset in zip(tp_offsets, chunk.offsets)
            ]
            reads.append(
                self._add_read(
                    checkpoint_fqn=checkpoint_fqn,
                    checkpoint_offsets=checkpoint_offsets,
                    destination=destination,
                )
            )
        return reads

    def _conv1d_reads(self, model_fqn: str, tensor: torch.Tensor) -> list[ReadItem]:
        if tensor.ndim != 3 or int(tensor.shape[0]) % 3:
            raise RuntimeError(
                f"Invalid fused GLM-5 conv1d tensor {model_fqn}: {tuple(tensor.shape)}."
            )
        local_segment_width = int(tensor.shape[0]) // 3
        tp_size = int(self.topology.tensor_parallel_size)
        tp_rank = int(self.topology.tensor_parallel_rank)
        reads = []
        for chunk, local_chunk in _local_chunks(tensor):
            chunk_start = int(chunk.offsets[0])
            chunk_end = chunk_start + int(chunk.sizes[0])
            for segment, projection in enumerate(("q", "k", "v")):
                segment_start = segment * local_segment_width
                segment_end = segment_start + local_segment_width
                read_start = max(chunk_start, segment_start)
                read_end = min(chunk_end, segment_end)
                if read_start >= read_end:
                    continue

                checkpoint_fqn = _checkpoint_fqn(model_fqn).replace(
                    "conv1d.weight", f"{projection}_conv1d.weight"
                )
                metadata = self._tensor_metadata(checkpoint_fqn)
                expected_shape = torch.Size(
                    [local_segment_width * tp_size, *tensor.shape[1:]]
                )
                if metadata.size != expected_shape:
                    raise RuntimeError(
                        f"Invalid source shape for {checkpoint_fqn}: "
                        f"expected={tuple(expected_shape)}, got={tuple(metadata.size)}."
                    )

                destination = local_chunk.narrow(
                    0,
                    read_start - chunk_start,
                    read_end - read_start,
                )
                checkpoint_offsets = [
                    tp_rank * local_segment_width + read_start - segment_start,
                    *[int(offset) for offset in chunk.offsets[1:]],
                ]
                reads.append(
                    self._add_read(
                        checkpoint_fqn=checkpoint_fqn,
                        checkpoint_offsets=checkpoint_offsets,
                        destination=destination,
                    )
                )
        return reads

    def _expert_reads(self, model_fqn: str, tensor: torch.Tensor) -> list[ReadItem]:
        if tensor.ndim != 3:
            raise RuntimeError(
                f"Invalid fused GLM-5 expert tensor {model_fqn}: {tuple(tensor.shape)}."
            )
        is_gate_up = model_fqn.endswith(".experts.gate_up_proj")
        if not is_gate_up and not model_fqn.endswith(".experts.down_proj"):
            raise AssertionError(f"Unexpected expert tensor: {model_fqn}.")
        if is_gate_up and int(tensor.shape[1]) % 2:
            raise RuntimeError(f"Packed gate/up width must be even for {model_fqn}.")

        ep_rank = int(self.topology.expert_parallel_rank)
        ep_size = int(self.topology.expert_parallel_size)
        local_expert_count = int(tensor.shape[0])
        expert_base = ep_rank * local_expert_count if ep_size > 1 else 0
        reads = []
        for chunk, local_chunk in _local_chunks(tensor):
            if any(int(offset) for offset in chunk.offsets[1:]) or tuple(
                int(size) for size in chunk.sizes[1:]
            ) != tuple(int(size) for size in tensor.shape[1:]):
                raise RuntimeError(
                    f"GLM-5 expert FSDP shards must use dimension 0: {model_fqn}."
                )
            first_expert = int(chunk.offsets[0])
            for local_index in range(int(chunk.sizes[0])):
                model_expert = first_expert + local_index
                checkpoint_expert = expert_base + model_expert
                expert_destination = local_chunk.select(0, local_index)
                if is_gate_up:
                    intermediate_size = int(tensor.shape[1]) // 2
                    for projection, offset in (
                        ("gate", 0),
                        ("up", intermediate_size),
                    ):
                        checkpoint_fqn = _checkpoint_fqn(model_fqn).replace(
                            "experts.gate_up_proj",
                            f"experts.{checkpoint_expert}.{projection}_proj.weight",
                        )
                        destination = expert_destination.narrow(
                            0, offset, intermediate_size
                        )
                        reads.append(
                            self._add_full_tensor_read(
                                checkpoint_fqn=checkpoint_fqn,
                                destination=destination,
                            )
                        )
                else:
                    checkpoint_fqn = _checkpoint_fqn(model_fqn).replace(
                        "experts.down_proj",
                        f"experts.{checkpoint_expert}.down_proj.weight",
                    )
                    reads.append(
                        self._add_full_tensor_read(
                            checkpoint_fqn=checkpoint_fqn,
                            destination=expert_destination,
                        )
                    )
        return reads

    def create_local_plan(self) -> LoadPlan:
        self._destination_views.clear()
        self._read_item_index = 0
        reads = []
        for model_fqn, tensor in self.state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"GLM-5 model state {model_fqn!r} is not a tensor: "
                    f"{type(tensor)!r}."
                )
            if model_fqn.endswith(".self_attn.conv1d.weight"):
                reads.extend(self._conv1d_reads(model_fqn, tensor))
            elif model_fqn.endswith(
                (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
            ):
                reads.extend(self._expert_reads(model_fqn, tensor))
            else:
                reads.extend(self._ordinary_reads(model_fqn, tensor))
        return LoadPlan(reads)

    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        try:
            return self._destination_views[read_item.dest_index.fqn]
        except KeyError as error:
            raise RuntimeError(
                f"Unknown GLM-5 DCP destination {read_item.dest_index.fqn!r}."
            ) from error

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        del read_item, tensor


class Glm5QuantizedHuggingFaceStorageReader(
    dcp.QuantizedHuggingFaceStorageReader
):
    """Vectorized FP8 block dequantization for GLM-5 checkpoints.

    PyTorch's generic reader walks every 128x128 block in Python.  A GLM-5
    expert matrix contains hundreds of those blocks, and each rank loads
    thousands of expert matrices.  Broadcasting the scale grid performs the
    same FP32 multiply in a small number of tensor operations instead.
    """

    def read_metadata(self) -> Metadata:
        """Build DCP metadata directly from each safetensors JSON header.

        The generic reader calls ``get_slice`` twice for all 76k checkpoint
        entries.  GLM checkpoints are ordinary, unsharded Hugging Face files,
        so their header already contains every shape and dtype we need.
        """

        from safetensors.torch import _getdtype

        metadata_started = time.perf_counter()
        checkpoint_path = Path(self.path)
        self._weight_scale_mapping.clear()
        self._tensor_full_shapes.clear()
        self._load_quantization_metadata()
        scale_fqns = set(self._weight_scale_mapping.values())
        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _Glm5HuggingFaceStorageInfo] = {}
        shard_paths = sorted(
            {checkpoint_path / filename for filename in self._weight_map.values()}
        )

        for shard_path in shard_paths:
            with shard_path.open("rb") as handle:
                length_bytes = handle.read(8)
                if len(length_bytes) != 8:
                    raise RuntimeError(f"Invalid safetensors header: {shard_path}.")
                header_length = int.from_bytes(length_bytes, byteorder="little")
                header_bytes = handle.read(header_length)
                if len(header_bytes) != header_length:
                    raise RuntimeError(f"Truncated safetensors header: {shard_path}.")
            header = json.loads(header_bytes)
            extra_metadata = header.pop("__metadata__", None)
            if isinstance(extra_metadata, dict) and extra_metadata.get(
                "DCP_SHARDING_INFO"
            ):
                # Not expected for native GLM, but retain the generic reader's
                # behavior if a DCP-sharded Hugging Face file is supplied.
                metadata = super().read_metadata()
                self.metadata_seconds = time.perf_counter() - metadata_started
                return metadata

            for fqn, tensor_header in header.items():
                # Scale tensors are read directly alongside their FP8 weights;
                # the load planner never addresses them as destinations.
                if fqn in scale_fqns:
                    continue
                if fqn in state_dict_metadata:
                    raise RuntimeError(
                        f"Duplicate unsharded GLM checkpoint tensor {fqn!r}."
                    )
                shape = torch.Size(tensor_header["shape"])
                dtype = _getdtype(tensor_header["dtype"])
                offsets = torch.Size([0] * len(shape))
                state_dict_metadata[fqn] = TensorStorageMetadata(
                    properties=TensorProperties(dtype=dtype),
                    size=shape,
                    chunks=[ChunkStorageMetadata(offsets=offsets, sizes=shape)],
                )
                storage_data[MetadataIndex(fqn=fqn, offset=offsets)] = (
                    _Glm5HuggingFaceStorageInfo(
                        relative_path=os.fspath(shard_path),
                        shape=shape,
                        dtype=dtype,
                    )
                )
                self._tensor_full_shapes[fqn] = shape

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,
            storage_data=storage_data,
            storage_meta=StorageMeta(load_id=self.load_id),
        )
        self.metadata_seconds = time.perf_counter() - metadata_started
        return metadata

    def _dequantize_tensor(
        self,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        full_tensor_shape: torch.Size,
        slice_info: tuple[tuple[int, int], tuple[int, int], slice, slice],
    ) -> torch.Tensor:
        del full_tensor_shape
        (row_block_range, col_block_range, row_slice, col_slice) = slice_info
        row_block_start, row_block_end = row_block_range
        col_block_start, col_block_end = col_block_range
        block_rows = row_block_end - row_block_start
        block_cols = col_block_end - col_block_start

        upcasted_weight = weight.to(torch.float32)
        scales = scale_inv[
            row_block_start:row_block_end,
            col_block_start:col_block_end,
        ].to(device=upcasted_weight.device, dtype=torch.float32)

        # GLM-5's expert matrices and their EP/TP slices are block-aligned.
        # Keep that overwhelmingly common path allocation-light.
        if tuple(upcasted_weight.shape) == (
            block_rows * self.block_size,
            block_cols * self.block_size,
        ):
            blocks = upcasted_weight.reshape(
                block_rows,
                self.block_size,
                block_cols,
                self.block_size,
            )
            blocks.mul_(scales[:, None, :, None])
            return blocks.reshape_as(upcasted_weight).to(self.target_dtype)

        # TP slices and final edge blocks are not necessarily aligned. Expand
        # only their small scale grid, then crop it to the requested slice.
        row_offset = int(row_slice.start) - row_block_start * self.block_size
        col_offset = int(col_slice.start) - col_block_start * self.block_size
        element_scales = scales.repeat_interleave(
            self.block_size, dim=0
        ).repeat_interleave(self.block_size, dim=1)
        element_scales = element_scales[
            row_offset : row_offset + int(weight.shape[0]),
            col_offset : col_offset + int(weight.shape[1]),
        ]
        upcasted_weight.mul_(element_scales)
        return upcasted_weight.to(self.target_dtype)


def _quantization_mapping(config) -> Mapping:
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None:
        return {}
    if hasattr(quantization_config, "to_dict"):
        return quantization_config.to_dict()
    return dict(quantization_config)


def load_glm5_huggingface_checkpoint(
    *,
    model,
    checkpoint_dir: str,
    config,
    topology,
) -> None:
    """Load native GLM-5 safetensors into an already-sharded FSDP2 model."""

    planner_topology = Glm5CheckpointTopology(
        tensor_parallel_rank=int(topology.tensor_parallel_rank),
        tensor_parallel_size=int(topology.tensor_parallel_size),
        expert_parallel_rank=int(topology.expert_parallel_rank),
        expert_parallel_size=int(topology.expert_parallel_size),
    )
    quantization_config = _quantization_mapping(config)
    quant_method = str(quantization_config.get("quant_method", "")).lower()
    reader_threads = int(os.environ.get("DEEPSPEC_DCP_LOAD_THREADS", "8"))
    if reader_threads < 1:
        raise ValueError("DEEPSPEC_DCP_LOAD_THREADS must be positive.")
    if quant_method in ("fp8", "quantizationmethod.fp8"):
        block_shape = tuple(
            int(size)
            for size in quantization_config.get("weight_block_size", (128, 128))
        )
        if block_shape != (128, 128):
            raise ValueError(
                "The DCP FP8 reader requires GLM-5 weight_block_size=[128, 128], "
                f"got {block_shape}."
            )
        storage_reader = Glm5QuantizedHuggingFaceStorageReader(
            checkpoint_dir,
            thread_count=reader_threads,
            target_dtype=torch.bfloat16,
            block_size=128,
        )
        reader_kind = "glm5-fast-fp8"
    else:
        storage_reader = dcp.HuggingFaceStorageReader(
            checkpoint_dir,
            thread_count=reader_threads,
        )
        reader_kind = "pytorch-huggingface"

    load_started = time.perf_counter()
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            "[deepspec-target-load] loading full-depth GLM-5 text target with FSDP2 DCP "
            f"(TP={planner_topology.tensor_parallel_size}, "
            f"EP={planner_topology.expert_parallel_size}, "
            f"reader={reader_kind}, reader_threads={reader_threads}; "
            "visual tower skipped, "
            "lm_head handled by draft initializer)",
            flush=True,
        )

    state_dict_started = time.perf_counter()
    model_state = model.state_dict()
    state_dict_seconds = time.perf_counter() - state_dict_started
    dcp_started = time.perf_counter()
    dcp.load(
        model_state,
        storage_reader=storage_reader,
        planner=Glm5HuggingFaceLoadPlanner(planner_topology),
    )
    dcp_seconds = time.perf_counter() - dcp_started
    assign_started = time.perf_counter()
    incompatible = model.load_state_dict(model_state, strict=True, assign=True)
    assign_seconds = time.perf_counter() - assign_started
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FSDP2 DCP produced an incompatible GLM-5 state dict: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if meta_parameters:
        raise RuntimeError(
            f"FSDP2 DCP left GLM-5 parameters on meta: {meta_parameters[:8]}."
        )
    if not dist.is_initialized() or dist.get_rank() == 0:
        metadata_seconds = getattr(storage_reader, "metadata_seconds", None)
        metadata_detail = (
            f", metadata_scan={metadata_seconds:.1f}s"
            if metadata_seconds is not None
            else ""
        )
        print(
            "[deepspec-target-load] GLM-5 FSDP2 DCP load complete "
            f"in {time.perf_counter() - load_started:.1f}s "
            f"(state_dict={state_dict_seconds:.1f}s, "
            f"read_and_dequantize={dcp_seconds:.1f}s, "
            f"assign={assign_seconds:.1f}s{metadata_detail})",
            flush=True,
        )


__all__ = [
    "Glm5CheckpointTopology",
    "Glm5HuggingFaceLoadPlanner",
    "Glm5QuantizedHuggingFaceStorageReader",
    "load_glm5_huggingface_checkpoint",
]

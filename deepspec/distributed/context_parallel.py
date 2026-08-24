from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch.distributed.tensor.experimental import context_parallel

from .mesh import ParallelContext


@dataclass(frozen=True)
class ContextParallelAssignment:
    degree: int
    mesh_name: str
    backend: str


class ContextParallelScheduler(ABC):
    """Scheduling boundary for a future real micro-batch Dynamic-CP system."""

    @abstractmethod
    def assignment_for_micro_batch(self, metadata: Mapping[str, object]) -> ContextParallelAssignment:
        raise NotImplementedError


class FixedContextParallelScheduler(ContextParallelScheduler):
    def __init__(self, *, degree: int, backend: str = "pytorch"):
        self.assignment = ContextParallelAssignment(degree, "cp", backend)

    def assignment_for_micro_batch(self, metadata: Mapping[str, object]) -> ContextParallelAssignment:
        del metadata
        return self.assignment


class DynamicContextParallelScheduler(ContextParallelScheduler):
    def assignment_for_micro_batch(self, metadata: Mapping[str, object]) -> ContextParallelAssignment:
        del metadata
        raise NotImplementedError(
            "Dynamic Context Parallel needs micro-batch packing, GPU-domain "
            "allocation, loss normalization and communication-group scheduling; "
            "no variable-only pseudo scheduler is provided."
        )


class FixedContextParallel:
    def __init__(self, context: ParallelContext, *, backend: str):
        self.context = context
        self.backend = backend
        self.scheduler = FixedContextParallelScheduler(
            degree=context.config.cp,
            backend=backend,
        )

    @contextmanager
    def forward_context(
        self,
        *,
        buffers: Iterable[torch.Tensor] = (),
        sequence_dims: Iterable[int] = (),
        no_restore_buffers: Iterable[torch.Tensor] = (),
    ):
        buffers = list(buffers)
        sequence_dims = list(sequence_dims)
        if self.context.config.cp == 1 or self.backend == "model_native":
            with nullcontext():
                yield
            return
        if self.backend != "pytorch":
            raise ValueError(f"Unsupported fixed CP backend {self.backend!r}.")
        if not buffers:
            raise ValueError(
                "PyTorch fixed CP requires every sequence-dependent input/buffer "
                "and its explicit sequence dimension."
            )
        if len(buffers) != len(sequence_dims):
            raise ValueError("buffers and sequence_dims must have identical lengths.")
        with context_parallel(
            self.context.cp_mesh,
            buffers=buffers,
            buffer_seq_dims=sequence_dims,
            no_restore_buffers=set(no_restore_buffers),
        ):
            yield


__all__ = [
    "ContextParallelAssignment",
    "ContextParallelScheduler",
    "DynamicContextParallelScheduler",
    "FixedContextParallel",
    "FixedContextParallelScheduler",
]

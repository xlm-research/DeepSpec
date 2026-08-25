from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .config import ParallelConfig


@dataclass
class ParallelContext:
    """Named mesh registry; no process group is created in a training step."""

    config: ParallelConfig
    dense_mesh: DeviceMesh
    sparse_mesh: DeviceMesh | None
    dp_mesh: DeviceMesh
    loss_mesh: DeviceMesh
    fsdp_mesh: DeviceMesh
    cp_mesh: DeviceMesh
    tp_mesh: DeviceMesh
    model_mesh: DeviceMesh

    @classmethod
    def build(
        cls,
        config: ParallelConfig,
        *,
        device_type: str | None = None,
    ) -> "ParallelContext":
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before building meshes.")
        config.validate_world_size(dist.get_world_size())
        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dense = init_device_mesh(
            device_type,
            (config.dp_replicate, config.dp_shard, config.cp, config.tp),
            mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"),
        )
        # Flattened views are created once here. They become named slices on
        # the root mesh and centralize every composite collective domain.
        dp = dense[("dp_replicate", "dp_shard")]._flatten("dp")
        loss = dense[("dp_replicate", "dp_shard", "cp")]._flatten("loss")
        dense[("dp_shard", "cp")]._flatten("dp_shard_cp")
        fsdp = dense[("dp_replicate", "dp_shard_cp")]
        model = dense[("cp", "tp")]._flatten("model")
        cp = dense["cp"]
        tp = dense["tp"]

        sparse = None
        if config.ep > 1:
            # Alternative view over the exact same ordered rank tensor.
            sparse = DeviceMesh(
                device_type,
                dense.mesh.reshape(
                    config.dp_replicate,
                    config.expert_fsdp,
                    config.ep,
                ),
                mesh_dim_names=("dp_replicate", "expert_fsdp", "ep"),
            )
        return cls(config, dense, sparse, dp, loss, fsdp, cp, tp, model)

    def with_sparse_config(self, config: ParallelConfig) -> "ParallelContext":
        """Create a model-specific sparse view while reusing the dense mesh.

        TorchTitan initializes the world/dense mesh once and derives its
        sparse ``(dp_replicate, efsdp, ep)`` view from the same ordered rank
        tensor.  Reusing the dense registry is important when an online target
        and a trainable draft need different EP degrees: it avoids creating a
        second set of CP/TP/FSDP process groups while keeping the target EP
        communicator independent from the draft model.
        """

        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before building meshes.")
        config.validate_world_size(dist.get_world_size())
        dense_dimensions = ("dp_replicate", "dp_shard", "cp", "tp", "pp")
        changed = [
            name
            for name in dense_dimensions
            if getattr(config, name) != getattr(self.config, name)
        ]
        if changed:
            raise ValueError(
                "A sparse model view must reuse the existing dense layout; "
                f"changed={changed}."
            )

        sparse = None
        if config.ep > 1:
            sparse = DeviceMesh(
                self.dense_mesh.device_type,
                self.dense_mesh.mesh.reshape(
                    config.dp_replicate,
                    config.expert_fsdp,
                    config.ep,
                ),
                mesh_dim_names=("dp_replicate", "expert_fsdp", "ep"),
            )
        return type(self)(
            config=config,
            dense_mesh=self.dense_mesh,
            sparse_mesh=sparse,
            dp_mesh=self.dp_mesh,
            loss_mesh=self.loss_mesh,
            fsdp_mesh=self.fsdp_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            model_mesh=self.model_mesh,
        )

    @staticmethod
    def _group_ranks(group) -> tuple[int, ...]:
        if group is None:
            return ()
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))

    def local_group_dict(self) -> dict[str, object]:
        """Return the rank-local communicator mapping for diagnostics."""

        groups = {
            "dense_coordinate": tuple(self.dense_mesh.get_coordinate() or ()),
            "dp_replicate": self._group_ranks(
                self.dense_mesh["dp_replicate"].get_group()
            ),
            "dp_shard": self._group_ranks(
                self.dense_mesh["dp_shard"].get_group()
            ),
            "fsdp_shard": self._group_ranks(
                self.fsdp_mesh["dp_shard_cp"].get_group()
            ),
            "cp": self._group_ranks(self.context_parallel_group),
            "tp": self._group_ranks(self.tensor_parallel_group),
            "ep": self._group_ranks(self.expert_parallel_group),
        }
        if self.sparse_mesh is not None:
            groups["expert_fsdp"] = self._group_ranks(
                self.sparse_mesh["expert_fsdp"].get_group()
            )
        else:
            groups["expert_fsdp"] = ()
        return groups

    def mesh(self, name: str) -> DeviceMesh:
        meshes = {
            "dense": self.dense_mesh,
            "sparse": self.sparse_mesh,
            "dp": self.dp_mesh,
            "loss": self.loss_mesh,
            "fsdp": self.fsdp_mesh,
            "cp": self.cp_mesh,
            "tp": self.tp_mesh,
            "model": self.model_mesh,
        }
        if name not in meshes or meshes[name] is None:
            raise KeyError(f"Mesh {name!r} is not available for {self.config}.")
        return meshes[name]

    def group(self, name: str):
        mesh = self.mesh(name)
        if mesh.ndim != 1:
            raise ValueError(f"Mesh {name!r} is {mesh.ndim}-D; request a named dimension.")
        return mesh.get_group()

    @property
    def context_parallel_rank(self) -> int:
        return self.cp_mesh.get_local_rank()

    @property
    def context_parallel_size(self) -> int:
        return self.config.cp

    @property
    def context_parallel_group(self):
        return self.cp_mesh.get_group()

    @property
    def tensor_parallel_rank(self) -> int:
        return self.tp_mesh.get_local_rank()

    @property
    def tensor_parallel_size(self) -> int:
        return self.config.tp

    @property
    def tensor_parallel_group(self):
        return self.tp_mesh.get_group()

    @property
    def expert_parallel_size(self) -> int:
        return self.config.ep

    @property
    def expert_parallel_rank(self) -> int:
        return 0 if self.sparse_mesh is None else self.sparse_mesh["ep"].get_local_rank()

    @property
    def expert_parallel_group(self):
        return None if self.sparse_mesh is None else self.sparse_mesh["ep"].get_group()

    @property
    def pure_expert_parallel(self) -> bool:
        return self.config.ep > 1

    @property
    def fsdp_size(self) -> int:
        return self.config.fsdp_shard_size

    @property
    def fsdp_group(self):
        return self.fsdp_mesh.get_group(("dp_replicate", "dp_shard_cp")) if self.fsdp_mesh.ndim > 1 else self.fsdp_mesh.get_group()

    @property
    def data_parallel_rank(self) -> int:
        coordinate = self.dense_mesh.get_coordinate()
        assert coordinate is not None
        return coordinate[0] * self.config.dp_shard + coordinate[1]

    @property
    def data_parallel_size(self) -> int:
        return self.config.data_parallel_size

    @property
    def model_parallel_src_rank(self) -> int:
        ranks = self.model_mesh.mesh.reshape(-1)
        return int(ranks[0].item())

    def topology_dict(self) -> dict[str, object]:
        return {
            "dense_shape": tuple(self.dense_mesh.mesh.shape),
            "dense_names": tuple(self.dense_mesh.mesh_dim_names or ()),
            "sparse_shape": (
                tuple(self.sparse_mesh.mesh.shape) if self.sparse_mesh is not None else None
            ),
            "data_parallel_size": self.data_parallel_size,
            "fsdp_shard_size": self.config.fsdp_shard_size,
        }

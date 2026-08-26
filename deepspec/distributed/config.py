from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class ParallelConfig:
    """All parallel degrees and transforms used by the training engine.

    ``dp_shard`` and ``cp`` share the dense FSDP shard domain.  TP is an
    orthogonal parameter layout.  In a future MoE model, the same ranks can be
    viewed as ``expert_fsdp x ep`` rather than multiplying EP into world size.
    """

    dp_replicate: int = 1
    dp_shard: int = 1
    tp: int = 1
    cp: int = 1
    ep: int = 1
    expert_tp: int = 1
    pp: int = 1

    use_fsdp: bool = True
    use_sequence_parallel: bool = False
    use_loss_parallel: bool = False
    use_activation_checkpoint: bool = False
    use_compile: bool = False

    expert_dispatch_backend: str = "native"
    context_parallel_backend: str = "pytorch"
    dynamic_context_parallel: bool = False
    reshard_after_forward: bool = True
    forward_prefetch: bool = False
    backward_prefetch: bool = False
    prefetch_depth: int = 1
    # Preserve the historical shared-path policy. Individual draft configs may
    # opt into BF16 reduction after a numerical/performance comparison.
    reduce_dtype: str = "fp32"
    fsdp_wrap_granularity: str = "block"

    @property
    def dense_world_size(self) -> int:
        return (
            self.dp_replicate
            * self.dp_shard
            * self.cp
            * self.tp
            * self.pp
        )

    @property
    def data_parallel_size(self) -> int:
        return self.dp_replicate * self.dp_shard

    @property
    def fsdp_shard_size(self) -> int:
        # CP ranks contribute different sequence-token gradients, so they are
        # part of the dense shard/reduction domain rather than replicas.
        return self.dp_shard * self.cp

    @property
    def expert_fsdp(self) -> int:
        sparse_domain = self.dp_shard * self.cp * self.tp
        if sparse_domain % self.ep:
            raise ValueError(
                "The sparse rank domain must be divisible by ep: "
                f"dp_shard*cp*tp={sparse_domain}, ep={self.ep}."
            )
        return sparse_domain // self.ep

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        train: Mapping[str, Any],
        *,
        world_size: int,
    ) -> "ParallelConfig":
        """Read the nested config, with a lossless legacy-config bridge."""

        nested = train.get("parallel")
        if nested is not None:
            known = {field.name for field in fields(cls)}
            unknown = sorted(set(nested) - known)
            if unknown:
                raise ValueError(f"Unknown train.parallel keys: {unknown}.")
            values = dict(nested)
            config = cls(**values)
            config.validate_world_size(world_size)
            return config

        cp = int(train.get("context_parallel_size", 1))
        configured_fsdp = train.get("fsdp_size")
        if configured_fsdp is None:
            if world_size % cp:
                raise ValueError(
                    "world_size must be divisible by legacy "
                    f"context_parallel_size: {world_size} % {cp} != 0."
                )
            dp_shard = world_size // cp
        else:
            dp_shard = int(configured_fsdp)
        base = dp_shard * cp
        if world_size % base:
            raise ValueError(
                "Legacy fsdp_size*context_parallel_size must divide world_size: "
                f"{dp_shard}*{cp} does not divide {world_size}."
            )
        # Existing DeepSpec CP is an exact ring-FlexAttention implementation
        # for its mixed target-context/draft-query attention.  Keep that
        # backend for old configs; new configs opt into PyTorch SDPA CP.
        config = cls(
            dp_replicate=world_size // base,
            dp_shard=dp_shard,
            cp=cp,
            use_fsdp=bool(train.get("use_fsdp", True)),
            use_activation_checkpoint=bool(
                train.get("use_activation_checkpoint", False)
            ),
            use_compile=bool(train.get("torch_compile", False)),
            context_parallel_backend=("model_native" if cp > 1 else "pytorch"),
            reshard_after_forward=(
                str(train.get("sharding_strategy", "full_shard"))
                not in {"shard_grad_op", "hybrid_shard_zero2", "_hybrid_shard_zero2"}
            ),
        )
        config.validate_world_size(world_size)
        return config

    def validate_world_size(self, world_size: int) -> None:
        degree_names = (
            "dp_replicate",
            "dp_shard",
            "tp",
            "cp",
            "ep",
            "expert_tp",
            "pp",
        )
        for name in degree_names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"parallel degree {name} must be a positive integer, got {value!r}.")
        if self.pp != 1:
            raise NotImplementedError(
                "Pipeline parallelism is reserved in the mesh schema but is not "
                "implemented by the current trainer. Set pp=1."
            )
        if self.dense_world_size != int(world_size):
            raise ValueError(
                "world_size does not match the dense mesh: "
                f"world_size={world_size}, expected "
                "dp_replicate*dp_shard*cp*tp*pp="
                f"{self.dense_world_size}."
            )
        if not self.use_fsdp and self.dp_replicate * self.dp_shard * self.cp > 1:
            raise ValueError(
                "use_fsdp=false is only valid when dp_replicate=dp_shard=cp=1; "
                "otherwise replicated parameters would receive unsynchronized "
                "gradients. TP-only execution remains supported."
            )
        if self.expert_tp > self.tp or self.tp % self.expert_tp:
            raise ValueError(
                "expert_tp must divide tp and cannot exceed it: "
                f"expert_tp={self.expert_tp}, tp={self.tp}."
            )
        # Evaluates and validates the alternative sparse mesh equation.
        expert_fsdp = self.expert_fsdp
        if self.dp_shard * self.cp * self.tp != expert_fsdp * self.ep:
            raise AssertionError("dense/sparse mesh views do not cover identical ranks")
        if self.use_sequence_parallel and self.tp == 1:
            raise ValueError("Sequence Parallel requires tp > 1.")
        if self.use_loss_parallel and self.tp == 1:
            raise ValueError("Loss Parallel requires tp > 1.")
        if self.dynamic_context_parallel:
            raise NotImplementedError(
                "dynamic_context_parallel is intentionally not implemented. "
                "Use fixed cp or provide a real micro-batch scheduler/backend."
            )
        if self.context_parallel_backend not in {"pytorch", "model_native"}:
            raise ValueError(
                "context_parallel_backend must be 'pytorch' or 'model_native', "
                f"got {self.context_parallel_backend!r}."
            )
        if self.expert_dispatch_backend not in {"native", "deepep", "auto"}:
            raise ValueError(
                "expert_dispatch_backend must be native, deepep, or auto, got "
                f"{self.expert_dispatch_backend!r}."
            )
        if (
            not isinstance(self.prefetch_depth, int)
            or isinstance(self.prefetch_depth, bool)
            or self.prefetch_depth < 1
        ):
            raise ValueError(
                "prefetch_depth must be a positive integer, got "
                f"{self.prefetch_depth!r}."
            )
        if self.reduce_dtype not in {"auto", "bf16", "fp32"}:
            raise ValueError(
                "reduce_dtype must be auto, bf16, or fp32, got "
                f"{self.reduce_dtype!r}."
            )
        if self.fsdp_wrap_granularity not in {"block", "block_components"}:
            raise ValueError(
                "fsdp_wrap_granularity must be block or block_components, got "
                f"{self.fsdp_wrap_granularity!r}."
            )

    def validate_model(
        self,
        model_or_config,
        *,
        sequence_length: int | None = None,
        has_moe: bool | None = None,
    ) -> None:
        config = getattr(model_or_config, "config", model_or_config)
        hidden_size = getattr(config, "hidden_size", None)
        attention_heads = getattr(config, "num_attention_heads", None)
        kv_heads = getattr(config, "num_key_value_heads", None)
        model_type = str(getattr(config, "model_type", ""))
        for name, value in (
            ("hidden_size", hidden_size),
            ("num_attention_heads", attention_heads),
            ("num_key_value_heads", kv_heads),
        ):
            if name == "num_key_value_heads" and model_type == "deepseek_v4":
                # V4 uses one replicated MQA KV head while query/output heads
                # are tensor-parallel. Its model adapter synchronizes the
                # replicated branch's gradient explicitly.
                continue
            if self.tp > 1 and value is not None and int(value) % self.tp:
                raise ValueError(
                    f"TP={self.tp} requires config.{name}={value} to be divisible by TP."
                )
        if sequence_length is not None and self.cp > 1:
            divisor = 2 * self.cp if self.context_parallel_backend in {
                "pytorch", "model_native"
            } else self.cp
            if int(sequence_length) % divisor:
                raise ValueError(
                    f"CP={self.cp} with {self.context_parallel_backend} head/tail "
                    f"layout requires sequence_length={sequence_length} to be "
                    f"divisible by {divisor}."
                )
        if has_moe is None:
            has_moe = bool(
                getattr(config, "num_experts", 0)
                or getattr(config, "num_local_experts", 0)
                or getattr(config, "n_routed_experts", 0)
            )
        num_experts = int(
            getattr(config, "num_experts", 0)
            or getattr(config, "num_local_experts", 0)
            or getattr(config, "n_routed_experts", 0)
        )
        if self.ep > 1 and not has_moe:
            raise ValueError(
                "EP > 1 was requested, but the selected draft model has no MoE "
                "router/experts. Set ep=1."
            )
        if self.ep > 1 and num_experts and num_experts % self.ep:
            raise ValueError(
                f"EP={self.ep} requires num_experts={num_experts} to be divisible by EP."
            )
        model_name = type(model_or_config).__name__
        is_dspark = "DSpark" in model_name or "DFlash2" in model_name
        if is_dspark and self.use_sequence_parallel:
            raise NotImplementedError(
                "Sequence Parallel is not valid for DSpark/DFlash2's replicated "
                "draft-query plus sharded-target-context input contract."
            )
        if is_dspark and self.use_loss_parallel:
            raise NotImplementedError(
                "Loss Parallel cannot shard DSpark/DFlash2 lm_head because the "
                "candidate selector consumes full-vocabulary logits."
            )
        if (
            is_dspark
            and self.cp > 1
            and self.context_parallel_backend == "pytorch"
        ):
            raise NotImplementedError(
                "PyTorch context_parallel patches SDPA, while DSpark/DFlash2 "
                "uses mixed-context FlexAttention. Select backend='model_native' "
                "for its fixed-degree ring CP."
            )
        if self.cp > 1 and self.use_compile and self.context_parallel_backend == "model_native":
            raise NotImplementedError(
                "torch.compile with DSpark ring CP is not enabled; its four "
                "FlexAttention kernels are already compiled independently."
            )

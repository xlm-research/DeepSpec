import json
import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import deepspec.utils.training_logger as training_logger
from deepspec.data import (
    CacheCollator,
    CacheDataset,
    ConversationCollator,
    validate_train_cache,
)
from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.modeling.pure_ep import (
    get_pure_expert_modules,
    materialize_modules_locally,
    synchronize_module_gradients,
)
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    is_model_parallel_checkpoint,
    is_rank_local_checkpoint,
    load_rank_local_draft_model,
    load_resume_draft_model,
    load_training_state,
    save_checkpoint,
)
from deepspec.utils import (
    BF16Optimizer,
    StatelessResumableDistributedSampler,
    build_parallel_topology,
    ensure_dir,
    init_dist,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
from deepspec.utils.hfai_suspend import SuspendController

_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_SHARDING_STRATEGIES = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}

_HYBRID_STRATEGIES = (
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
)


def _load_safetensors_weight(
    checkpoint_dir: str,
    *,
    candidate_names: tuple[str, ...],
    shard_dim: int | None = None,
    shard_rank: int = 0,
    shard_size: int = 1,
) -> torch.Tensor:
    """Load one named tensor without constructing a multi-hundred-B model."""

    from safetensors import safe_open

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        for name in candidate_names:
            file_name = weight_map.get(name)
            if file_name is None:
                continue
            with safe_open(
                os.path.join(checkpoint_dir, file_name),
                framework="pt",
                device="cpu",
            ) as handle:
                if int(shard_size) == 1:
                    if int(shard_size) == 1:
                        return handle.get_tensor(name)
                    tensor_slice = handle.get_slice(name)
                    shape = tensor_slice.get_shape()
                    dim = int(shard_dim)
                    if int(shape[dim]) % int(shard_size) != 0:
                        raise ValueError(
                            f"Cannot shard {name} shape={shape} on dim={dim} "
                            f"across {shard_size} TP ranks."
                        )
                    width = int(shape[dim]) // int(shard_size)
                    slices = [slice(None)] * len(shape)
                    slices[dim] = slice(
                        int(shard_rank) * width,
                        (int(shard_rank) + 1) * width,
                    )
                    return tensor_slice[tuple(slices)]
                tensor_slice = handle.get_slice(name)
                shape = tensor_slice.get_shape()
                dim = int(shard_dim)
                if int(shape[dim]) % int(shard_size) != 0:
                    raise ValueError(
                        f"Cannot shard {name} shape={shape} on dim={dim} "
                        f"across {shard_size} TP ranks."
                    )
                width = int(shape[dim]) // int(shard_size)
                slices = [slice(None)] * len(shape)
                slices[dim] = slice(
                    int(shard_rank) * width,
                    (int(shard_rank) + 1) * width,
                )
                return tensor_slice[tuple(slices)]
    else:
        tensor_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.isfile(tensor_path):
            with safe_open(tensor_path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for name in candidate_names:
                    if name in available:
                        return handle.get_tensor(name)
    raise KeyError(
        "None of the requested checkpoint tensors exists: "
        f"{candidate_names}."
    )


def _load_deepseek_v4_embedding_and_head(
    checkpoint_dir: str, *, tensor_parallel_rank: int, tensor_parallel_size: int
):
    embed_weight = _load_safetensors_weight(
        checkpoint_dir,
        candidate_names=(
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "embed.weight",
        ),
        shard_dim=0,
        shard_rank=tensor_parallel_rank,
        shard_size=tensor_parallel_size,
    )
    lm_head_weight = _load_safetensors_weight(
        checkpoint_dir,
        candidate_names=(
            "lm_head.weight",
            "model.lm_head.weight",
            "head.weight",
        ),
        shard_dim=0,
        shard_rank=tensor_parallel_rank,
        shard_size=tensor_parallel_size,
    )
    return embed_weight, lm_head_weight


def _build_fsdp_kwargs(
    *,
    sharding_strategy_name: str,
    precision_dtype,
    parallel,
) -> dict:
    sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    replicate_size = int(parallel.fsdp_replica_size)
    if replicate_size > 1:
        if sharding_strategy == ShardingStrategy.FULL_SHARD:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        elif sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
            sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif sharding_strategy not in _HYBRID_STRATEGIES:
            raise ValueError(
                "train.fsdp_size < world_size requires full_shard, "
                "shard_grad_op, or a hybrid sharding strategy."
            )
    fsdp_kwargs = dict(
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=precision_dtype,
            buffer_dtype=precision_dtype,
        ),
        sharding_strategy=sharding_strategy,
        process_group=parallel.fsdp_group,
    )
    if sharding_strategy in _HYBRID_STRATEGIES:
        if parallel.fsdp_replica_group is None:
            raise RuntimeError(
                "Hybrid FSDP requires an orthogonal replica process group."
            )
        fsdp_kwargs["process_group"] = (
            parallel.fsdp_group,
            parallel.fsdp_replica_group,
        )
    return fsdp_kwargs


def _compute_gradient_accumulation_steps(
    *, world_size: int, local_batch_size: int, global_batch_size: int
) -> int:
    denom = world_size * local_batch_size
    assert global_batch_size % denom == 0, (
        "global_batch_size must be divisible by world_size * local_batch_size: "
        f"global_batch_size={global_batch_size}, world_size={world_size}, "
        f"local_batch_size={local_batch_size}"
    )
    return global_batch_size // denom


def _compute_samples_per_epoch(*, dataset_size: int, global_batch_size: int) -> int:
    samples_per_epoch = (dataset_size // global_batch_size) * global_batch_size
    assert samples_per_epoch > 0, (
        "train dataset is too small to form one full global batch: "
        f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
    )
    return samples_per_epoch


def _compute_training_schedule(
    *,
    world_size: int,
    dataset_size: int,
    local_batch_size: int,
    global_batch_size: int,
    num_train_epochs: int,
    max_train_steps=None,
) -> tuple[int, int, int, int, int, int, int]:
    gradient_accumulation_steps = _compute_gradient_accumulation_steps(
        world_size=world_size,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
    )
    samples_per_epoch = _compute_samples_per_epoch(
        dataset_size=dataset_size,
        global_batch_size=global_batch_size,
    )
    per_rank_samples_per_epoch = samples_per_epoch // world_size
    micro_batches_per_epoch = per_rank_samples_per_epoch // local_batch_size
    steps_per_epoch = micro_batches_per_epoch // gradient_accumulation_steps
    if max_train_steps is None:
        resolved_max_train_steps = int(num_train_epochs) * steps_per_epoch
        resolved_num_train_epochs = int(num_train_epochs)
    else:
        resolved_max_train_steps = int(max_train_steps)
        resolved_num_train_epochs = math.ceil(
            resolved_max_train_steps / steps_per_epoch
        )
    return (
        gradient_accumulation_steps,
        samples_per_epoch,
        per_rank_samples_per_epoch,
        micro_batches_per_epoch,
        steps_per_epoch,
        resolved_max_train_steps,
        resolved_num_train_epochs,
    )


def _launch_eval(
    *,
    target_model_name_or_path: str,
    checkpoint_dir: str,
    step: int,
    tensorboard_dir: str,
    exp_name: str,
) -> None:
    from deepspec.utils.constant import auto_eval_command
    if auto_eval_command is not None:
        command = auto_eval_command(
            target_model_name_or_path,
            checkpoint_dir,
            step,
            tensorboard_dir,
            exp_name,
        )
        print_on_global_main(f"Submitting auto eval for {checkpoint_dir}")
        print_on_global_main(command)
        os.system(command)
    else:
        print("You can use this function to launch your auto eval script!")

class BaseTrainer:
    data_collator_cls = None

    def __init__(self, local_rank, args):
        self.args = args
        self.device, self.global_rank, self.world_size = init_dist(local_rank)
        self.context_parallel_size = int(
            self.args.train.get("context_parallel_size", 1)
        )
        self.expert_parallel_size = int(
            self.args.train.get("expert_parallel_size", 1)
        )
        self.tensor_parallel_size = int(
            self.args.train.get("tensor_parallel_size", 1)
        )
        self.pure_expert_parallel = bool(
            self.args.train.get("pure_expert_parallel", False)
        )
        if self.context_parallel_size < 1:
            raise ValueError("train.context_parallel_size must be positive.")
        if self.expert_parallel_size < 1 or self.tensor_parallel_size < 1:
            raise ValueError(
                "train.expert_parallel_size and train.tensor_parallel_size "
                "must be positive."
            )
        if self.context_parallel_size > 1 and int(
            self.args.train.local_batch_size
        ) != 1:
            raise ValueError(
                "Context-parallel training currently requires "
                "train.local_batch_size=1."
            )
        configured_fsdp_size = self.args.train.get("fsdp_size")
        self.fsdp_size = (
            self.world_size
            // (
                self.context_parallel_size
                * self.tensor_parallel_size
                * (1 if self.pure_expert_parallel else self.expert_parallel_size)
            )
            if configured_fsdp_size is None
            else int(configured_fsdp_size)
        )
        self.parallel = build_parallel_topology(
            context_parallel_size=self.context_parallel_size,
            expert_parallel_size=self.expert_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            fsdp_size=self.fsdp_size,
            create_fsdp_groups=True,
            pure_expert_parallel=self.pure_expert_parallel,
        )
        self.data_parallel_size = self.parallel.sample_parallel_size
        self.data_parallel_rank = self.parallel.sample_parallel_rank
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir
        self.resume_checkpoint_dir = discover_latest_checkpoint(
            self.checkpoint_dir_root
        )
        self.resume_is_model_parallel = bool(
            self.resume_checkpoint_dir is not None
            and is_model_parallel_checkpoint(self.resume_checkpoint_dir)
        )
        self.resume_is_rank_local = bool(
            self.resume_checkpoint_dir is not None
            and is_rank_local_checkpoint(self.resume_checkpoint_dir)
        )
        if self.resume_is_rank_local and not is_rank_local_checkpoint(
            self.resume_checkpoint_dir,
            global_rank=self.global_rank,
        ):
            raise FileNotFoundError(
                "Rank-local checkpoint is incomplete for global rank "
                f"{self.global_rank}: {self.resume_checkpoint_dir}."
            )
        if self.resume_is_rank_local:
            self.resume_is_model_parallel = False
        self.suspend_controller = SuspendController(device=self.device)
        self.next_micro_step = 0
        self.online_target_enabled = bool(
            self.args.data.get("online_target", False)
        )
        self.online_target = None
        if self.online_target_enabled and int(
            self.args.train.local_batch_size
        ) != 1:
            raise ValueError(
                "Online target training currently requires "
                "train.local_batch_size=1."
            )
        if self.online_target_enabled and self.args.data.get(
            "target_cache_path"
        ) not in (None, ""):
            raise ValueError(
                "Online target training cannot also use data.target_cache_path."
            )

        if is_global_main_process():
            ensure_dir(self.checkpoint_dir_root)
        training_logger.init(
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
        )

        self.draft_model, self.tokenizer = self.build_models()
        if (
            self.resume_checkpoint_dir is not None
            and not self.resume_is_rank_local
            and not self.resume_is_model_parallel
        ):
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
                parallel=self.parallel,
            )
        configure_parallelism = getattr(
            self.draft_model, "configure_parallelism", None
        )
        configure_cp = getattr(self.draft_model, "configure_context_parallel", None)
        if (
            self.expert_parallel_size > 1
            or self.tensor_parallel_size > 1
        ) and configure_parallelism is None:
            raise NotImplementedError(
                f"{type(self.draft_model).__name__} does not implement "
                "expert/tensor-parallel training."
            )
        if self.context_parallel_size > 1 and (
            configure_parallelism is None and configure_cp is None
        ):
            raise NotImplementedError(
                f"{type(self.draft_model).__name__} does not implement "
                "context-parallel training."
            )
        if configure_parallelism is not None:
            configure_parallelism(self.parallel)
        elif self.context_parallel_size > 1:
            configure_cp(
                size=self.context_parallel_size,
                rank=self.parallel.context_parallel_rank,
                group=self.parallel.context_parallel_group,
                model_parallel_group=self.parallel.model_parallel_group,
                model_parallel_src_rank=self.parallel.model_parallel_src_rank,
            )
        if (
            getattr(self, "_pending_deepseek_v4_embedding_head", None)
            is not None
            and self.resume_checkpoint_dir is None
        ):
            embed_weight, lm_head_weight = _load_deepseek_v4_embedding_and_head(
                self._pending_deepseek_v4_embedding_head,
                tensor_parallel_rank=self.parallel.tensor_parallel_rank,
                tensor_parallel_size=self.parallel.tensor_parallel_size,
            )
            self.draft_model.initialize_embedding_and_head_weights(
                embed_weight=embed_weight.to(dtype=self.precision_dtype),
                lm_head_weight=lm_head_weight.to(dtype=self.precision_dtype),
                freeze=True,
            )
            del embed_weight, lm_head_weight
        if self.resume_is_model_parallel:
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
                parallel=self.parallel,
            )
        self.model = self.draft_model
        if (
            self.context_parallel_size > 1
            or self.expert_parallel_size > 1
            or self.tensor_parallel_size > 1
        ) and bool(
            self.args.train.torch_compile
        ):
            print_on_global_main(
                "Disabling whole-model torch.compile for distributed CP/EP/TP."
            )
            self.args.train.torch_compile = False
        if self.args.train.torch_compile:
            print_on_local_main("Compiling training model with torch.compile...")
            self.model = torch.compile(self.model, dynamic=True)
        self.model = self._wrap_with_fsdp(self.model)
        if self.resume_is_rank_local:
            load_rank_local_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                model=self.model,
                draft_model=self.draft_model,
                global_rank=self.global_rank,
                parallel=self.parallel,
            )

        if self.online_target_enabled:
            train_data_paths = self.args.data.get("train_data_path")
            if isinstance(train_data_paths, (str, os.PathLike)):
                train_data_paths = [os.fspath(train_data_paths)]
            elif train_data_paths is not None:
                train_data_paths = [os.fspath(path) for path in train_data_paths]
            if not train_data_paths:
                raise ValueError(
                    "Online target training requires data.train_data_path."
                )
            self.train_data_paths = train_data_paths
            self.train_dataset = JsonLineDataset(data_paths=train_data_paths)
            self.data_collator = ConversationCollator(
                tokenizer=self.tokenizer,
                chat_template=self.args.data.chat_template,
                max_length=int(self.args.data.max_length),
                min_loss_tokens=int(
                    self.args.data.get("min_loss_tokens", 1)
                ),
            )
        else:
            self.train_data_paths = None
            self.train_dataset = CacheDataset(
                cache_dir=self.args.data.target_cache_path,
                context_parallel_size=self.context_parallel_size,
                context_parallel_rank=self.parallel.context_parallel_rank,
            )
            validate_train_cache(
                train_dataset=self.train_dataset,
                draft_model=self.draft_model,
                target_model_name_or_path=(
                    self.args.model.target_model_name_or_path
                ),
            )
            collator_cls = self.data_collator_cls or CacheCollator
            self.data_collator = collator_cls()

        (
            self.gradient_accumulation_steps,
            self.samples_per_epoch,
            self.per_rank_samples_per_epoch,
            self.micro_batches_per_epoch,
            self.steps_per_epoch,
            self.max_train_steps,
            self.args.train.num_train_epochs,
        ) = _compute_training_schedule(
            world_size=self.data_parallel_size,
            dataset_size=len(self.train_dataset),
            local_batch_size=int(self.args.train.local_batch_size),
            global_batch_size=int(self.args.train.global_batch_size),
            num_train_epochs=int(self.args.train.num_train_epochs),
            max_train_steps=self.args.train.max_train_steps,
        )

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=float(self.args.train.lr),
            total_steps=self.max_train_steps,
            warmup_ratio=float(self.args.train.warmup_ratio),
            weight_decay=float(self.args.train.weight_decay),
        )
        if self.resume_checkpoint_dir is not None:
            resume_state = load_training_state(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                optimizer=self.optimizer,
                global_rank=self.global_rank,
                world_size=self.world_size,
                local_batch_size=int(self.args.train.local_batch_size),
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                micro_batches_per_epoch=self.micro_batches_per_epoch,
            )
            self.next_micro_step = resume_state.next_micro_step
        else:
            print_on_local_main("Training from scratch.")
        if self.online_target_enabled:
            self.online_target = self.build_online_target()
        self.info_board()

    @property
    def global_step(self):
        return self.next_micro_step // self.gradient_accumulation_steps

    def info_board(self):
        print_on_local_main("***** Running training *****")
        print_on_local_main(f"  Train dataset size = {len(self.train_dataset)}")
        print_on_local_main(
            "  Parallel topology = "
            f"CP {self.context_parallel_size} x "
            f"EP {self.expert_parallel_size} x "
            f"TP {self.tensor_parallel_size} x FSDP {self.fsdp_size} "
            f"x DP {self.parallel.data_parallel_size} "
            f"(effective data replicas {self.data_parallel_size})"
        )
        if self.expert_parallel_size > 1:
            print_on_local_main("  EP token dispatch = variable-split All-to-All")
        print_on_local_main(
            "  Target supervision = "
            + (
                "online frozen-target forward (no disk cache)"
                if self.online_target_enabled
                else "offline target cache"
            )
        )
        print_on_local_main(
            "  Startup parameter loading = rank-local (no parameter collectives)"
        )
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(
            f"  Global batch size = {self.args.train.global_batch_size}"
        )
        print_on_local_main(
            "  Gradient accumulation steps = "
            f"{self.gradient_accumulation_steps}"
        )
        print_on_local_main(f"  Steps per epoch = {self.steps_per_epoch}")
        print_on_local_main(f"  Max train steps = {self.max_train_steps}")

    def build_models(self):
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )

        self._pending_deepseek_v4_embedding_head = None
        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        # Every rank initializes the same deterministic parameters, then keeps
        # only its own EP/TP/FSDP slice. No source rank is involved.
        draft_model = draft_model.to(device="cpu", dtype=self.precision_dtype)

        # Training only needs two target tensors.  Constructing the complete
        # 149-GB V4-Flash model once per local process just to read them would
        # exhaust host memory, so V4 reads the two safetensors entries directly.
        if str(target_config.model_type) == "deepseek_v4":
            self._pending_deepseek_v4_embedding_head = str(
                model_args.target_model_name_or_path
            )
        else:
            target_model = AutoModelForCausalLM.from_pretrained(
                model_args.target_model_name_or_path,
                dtype=self.precision_dtype,
            ).to(device="cpu").eval()
            target_embed_tokens = target_model.get_input_embeddings()
            target_lm_head = target_model.get_output_embeddings()
            assert (target_lm_head is not None) and (target_embed_tokens is not None)
            draft_model.initialize_embeddings_and_head(
                embed_tokens=target_embed_tokens,
                lm_head=target_lm_head,
                freeze=True,
            )
            del target_model
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        raise NotImplementedError

    def build_online_target(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement online target training."
        )

    def _wrap_with_fsdp(self, model):
        fsdp_kwargs = _build_fsdp_kwargs(
            sharding_strategy_name=self.args.train.sharding_strategy,
            precision_dtype=self.precision_dtype,
            parallel=self.parallel,
        )
        fsdp_kwargs["device_id"] = self.device
        self._pure_expert_modules = get_pure_expert_modules(model)
        if self._pure_expert_modules:
            materialize_modules_locally(
                self._pure_expert_modules,
                device=self.device,
            )
            fsdp_kwargs["ignored_modules"] = self._pure_expert_modules
        fsdp_kwargs["sync_module_states"] = False
        if bool(self.args.train.get("fsdp_layerwise", False)):
            uncompiled_model = getattr(model, "_orig_mod", model)
            decoder_layers = list(getattr(uncompiled_model, "layers", []))
            if not decoder_layers:
                raise ValueError(
                    "train.fsdp_layerwise=true but the draft model does not "
                    "expose decoder layers."
                )
            fsdp_kwargs["auto_wrap_policy"] = ModuleWrapPolicy(
                {type(layer) for layer in decoder_layers}
            )
        return FSDP(model, **fsdp_kwargs)

    def _synchronize_pure_expert_gradients(self):
        modules = getattr(self, "_pure_expert_modules", None)
        if not modules:
            return
        process_groups = [
            (
                self.parallel.tensor_parallel_group,
                self.tensor_parallel_size,
            ),
        ]
        if self.pure_expert_parallel:
            process_groups.append(
                (
                    self.parallel.expert_replica_group,
                    self.parallel.expert_replica_size,
                )
            )
        else:
            process_groups.extend(
                [
                    (self.parallel.fsdp_group, self.fsdp_size),
                    (
                        self.parallel.fsdp_replica_group,
                        self.parallel.fsdp_replica_size,
                    ),
                ]
            )
        synchronize_module_gradients(modules, process_groups=process_groups)

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.data_parallel_size,
            rank=self.data_parallel_rank,
            total_size=self.samples_per_epoch,
            start_global_offset_samples=start_offset_samples,
            num_samples=num_samples,
        )
        num_workers = int(self.args.data.num_workers)
        loader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(
                self.args.data.get("prefetch_factor", 1)
            )
        return DataLoader(
            **loader_kwargs,
        )

    def run_batch(self, batch):
        raise NotImplementedError

    def _checkpoint_kwargs(self):
        return dict(
            model=self.model,
            draft_model=self.draft_model,
            optimizer=self.optimizer,
            checkpoint_dir_root=self.checkpoint_dir_root,
            train_config=self.args,
            next_micro_step=self.next_micro_step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            global_rank=self.global_rank,
            world_size=self.world_size,
            local_batch_size=int(self.args.train.local_batch_size),
            parallel=self.parallel,
        )

    def save_and_eval_checkpoint(self):
        checkpoint_dir = save_checkpoint(**self._checkpoint_kwargs())
        can_run_hf_eval = (
            self.expert_parallel_size == 1
            and self.tensor_parallel_size == 1
        )
        if is_global_main_process() and can_run_hf_eval:
            _launch_eval(
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                checkpoint_dir=checkpoint_dir,
                step=self.global_step,
                tensorboard_dir=self.args.logging.tensorboard_dir,
                exp_name=self.args.exp_name,
            )
        elif is_global_main_process():
            print_on_global_main(
                "Skipping automatic HF evaluation for an EP/TP-sharded "
                "checkpoint. Resume training directly from step_latest."
            )
        dist.barrier()
        return checkpoint_dir

    def _save_and_suspend(self):
        print_on_global_main("Saving checkpoint before suspending...")
        save_checkpoint(**self._checkpoint_kwargs())
        dist.barrier()
        if is_global_main_process():
            print_on_global_main("Going to suspend...")
            self.suspend_controller.go_suspend()
        dist.barrier()

    def train(self):
        self.model.train()
        if self.global_step >= self.max_train_steps:
            return

        local_batch_size = int(self.args.train.local_batch_size)
        total_micro_steps = self.max_train_steps * self.gradient_accumulation_steps
        remaining_micro_steps = total_micro_steps - self.next_micro_step
        remaining_samples = remaining_micro_steps * local_batch_size

        dataloader = self._build_train_dataloader(
            start_offset_samples=self.next_micro_step * local_batch_size,
            num_samples=remaining_samples,
        )
        prefetcher = CUDAPrefetcher(dataloader, self.device)
        training_logger.start_session(global_step=self.global_step)

        with self.suspend_controller.monitoring():
            for batch in prefetcher:
                should_sync = (
                    (self.next_micro_step + 1) % self.gradient_accumulation_steps == 0
                )
                sync_context = nullcontext() if should_sync else self.model.no_sync()
                with sync_context:
                    loss = self.run_batch(batch) / self.gradient_accumulation_steps
                    loss.backward()
                self.next_micro_step += 1

                if not should_sync:
                    continue

                self._synchronize_pure_expert_gradients()
                grad_norm = FSDP.clip_grad_norm_(
                    self.model,
                    float(self.args.train.max_grad_norm),
                )
                self.optimizer.step()
                training_logger.on_optimizer_step(
                    global_step=self.global_step,
                    next_micro_step=self.next_micro_step,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    max_train_steps=self.max_train_steps,
                    learning_rate=self.optimizer.get_learning_rate(),
                    grad_norm=grad_norm.item(),
                )

                if self.global_step % int(self.args.logging.checkpointing_steps) == 0:
                    self.save_and_eval_checkpoint()

                if self.suspend_controller.requested():
                    self._save_and_suspend()
                    return

        self.save_and_eval_checkpoint()

    def clean_up(self):
        if self.online_target is not None:
            self.online_target.close()
            self.online_target = None
        close_dataset = getattr(self.train_dataset, "close", None)
        if close_dataset is not None:
            close_dataset()
        training_logger.close()
        dist.barrier()
        dist.destroy_process_group()

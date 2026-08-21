from contextlib import nullcontext
import math
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from deepspec.data import CacheDataset, validate_train_cache
from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.modeling.target_adapter import (
    get_target_embeddings,
    is_multimodal_config,
    load_target_model_with_head,
)
from deepspec.utils import (
    BF16Optimizer,
    StatelessResumableDistributedSampler,
    ensure_dir,
    init_dist,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    load_resume_draft_model,
    load_training_state,
    save_checkpoint,
)
import deepspec.utils.training_logger as training_logger
from deepspec.utils.hfai_suspend import SuspendController
from deepspec.utils.parallel import build_parallel_topology


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


def _build_fsdp_kwargs(
    *,
    sharding_strategy_name: str,
    precision_dtype,
    world_size: int,
    fsdp_size: int,
) -> dict:
    sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    replicate_size = world_size // int(fsdp_size)
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
    )
    if sharding_strategy in _HYBRID_STRATEGIES:
        fsdp_kwargs["device_mesh"] = init_device_mesh(
            "cuda",
            (replicate_size, int(fsdp_size)),
            mesh_dim_names=("replicate", "shard"),
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
        command = auto_eval_command(target_model_name_or_path,checkpoint_dir,step,tensorboard_dir,exp_name)
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
        configured_fsdp_size = self.args.train.get("fsdp_size")
        self.fsdp_size = (
            self.world_size // self.context_parallel_size
            if configured_fsdp_size is None
            else int(configured_fsdp_size)
        )
        self.parallel = build_parallel_topology(
            context_parallel_size=self.context_parallel_size,
            fsdp_size=self.fsdp_size,
            create_fsdp_groups=False,
        )
        self.data_parallel_size = self.parallel.sample_parallel_size
        self.data_parallel_rank = self.parallel.sample_parallel_rank
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir
        self.resume_checkpoint_dir = discover_latest_checkpoint(
            self.checkpoint_dir_root
        )
        self.suspend_controller = SuspendController(device=self.device)
        self.next_micro_step = 0

        if is_global_main_process():
            ensure_dir(self.checkpoint_dir_root)
        training_logger.init(
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
        )

        self.draft_model, self.tokenizer = self.build_models()
        if self.resume_checkpoint_dir is not None:
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                device=self.device,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
            )
        configure_cp = getattr(
            self.draft_model, "configure_context_parallel", None
        )
        if configure_cp is None:
            if self.context_parallel_size > 1:
                raise NotImplementedError(
                    f"{type(self.draft_model).__name__} does not implement CP."
                )
        else:
            configure_cp(
                size=self.context_parallel_size,
                rank=self.parallel.context_parallel_rank,
                group=self.parallel.context_parallel_group,
                model_parallel_group=self.parallel.model_parallel_group,
                model_parallel_src_rank=self.parallel.model_parallel_src_rank,
            )
        self.model = self.draft_model
        if self.context_parallel_size > 1 and bool(
            self.args.train.torch_compile
        ):
            print_on_global_main(
                "Disabling torch.compile because the CP path contains "
                "autograd collectives and dynamic FlexAttention masks."
            )
            self.args.train.torch_compile = False
        if self.args.train.torch_compile:
            print_on_local_main("Compiling training model with torch.compile...")
            self.model = torch.compile(self.model, dynamic=True)
        self.model = self._wrap_with_fsdp(self.model)

        self.train_dataset = CacheDataset(
            cache_dir=self.args.data.target_cache_path,
            context_parallel_size=self.context_parallel_size,
            context_parallel_rank=self.parallel.context_parallel_rank,
        )
        # Hashing a 256K packed source on every worker creates needless shared
        # filesystem traffic. Validate the complete cache identity once, then
        # broadcast any error so all ranks fail together before training.
        cache_validation_error = [None]
        if is_global_main_process():
            source_jsonl_path = self.args.data.get("source_jsonl_path")
            try:
                validate_train_cache(
                    train_dataset=self.train_dataset,
                    draft_model=self.draft_model,
                    target_model_name_or_path=(
                        self.args.model.target_model_name_or_path
                    ),
                    source_jsonl_paths=(
                        [source_jsonl_path] if source_jsonl_path else None
                    ),
                    chat_template=self.args.data.get("chat_template"),
                    max_length=self.args.data.get("max_length"),
                    stores_target_last_hidden_states=self.args.data.get(
                        "store_target_last_hidden_states"
                    ),
                )
            except (AssertionError, OSError, ValueError) as exc:
                cache_validation_error[0] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(cache_validation_error, src=0)
        if cache_validation_error[0] is not None:
            raise ValueError(
                "Target cache identity validation failed: "
                f"{cache_validation_error[0]}"
            )

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
        self.info_board()

    @property
    def global_step(self):
        return self.next_micro_step // self.gradient_accumulation_steps

    def info_board(self):
        print_on_local_main("***** Running training *****")
        print_on_local_main(f"  Train dataset size = {len(self.train_dataset)}")
        print_on_local_main(
            "  Parallel topology = "
            f"CP {self.context_parallel_size} x FSDP {self.fsdp_size} "
            f"(effective data replicas {self.data_parallel_size})"
        )
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(f"  Global batch size = {self.args.train.global_batch_size}")
        print_on_local_main(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        print_on_local_main(f"  Steps per epoch = {self.steps_per_epoch}")
        print_on_local_main(f"  Max train steps = {self.max_train_steps}")

    def build_models(self):
        model_args = self.args.model

        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )
        if is_multimodal_config(target_config):
            processor = AutoProcessor.from_pretrained(
                model_args.target_model_name_or_path,
            )
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.target_model_name_or_path,
            )

        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        # Training only uses the target checkpoint to initialize frozen draft
        # embeddings and lm_head weights.
        target_model = load_target_model_with_head(
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens, target_lm_head = get_target_embeddings(target_model)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        del target_model
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        raise NotImplementedError

    def _wrap_with_fsdp(self, model):
        fsdp_kwargs = _build_fsdp_kwargs(
            sharding_strategy_name=self.args.train.sharding_strategy,
            precision_dtype=self.precision_dtype,
            world_size=self.world_size,
            fsdp_size=self.fsdp_size,
        )
        fsdp_kwargs["device_id"] = self.device
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

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.data_parallel_size,
            rank=self.data_parallel_rank,
            total_size=self.samples_per_epoch,
            start_global_offset_samples=start_offset_samples,
            num_samples=num_samples,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator_cls(),
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=1,
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
        )

    def save_and_eval_checkpoint(self):
        checkpoint_dir = save_checkpoint(**self._checkpoint_kwargs())
        if is_global_main_process():
            _launch_eval(
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                checkpoint_dir=checkpoint_dir,
                step=self.global_step,
                tensorboard_dir=self.args.logging.tensorboard_dir,
                exp_name=self.args.exp_name,
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
                # Target activations are immutable offline supervision.  Once
                # backward has consumed them, drop the last Python references
                # immediately so their CUDA storage can be reused by the next
                # micro-batch instead of keeping it alive through optimizer and
                # checkpoint work.
                del loss
                batch.pop("target_hidden_states", None)
                batch.pop("target_last_hidden_states", None)
                self.next_micro_step += 1

                if not should_sync:
                    continue

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
        close_dataset = getattr(self.train_dataset, "close", None)
        if close_dataset is not None:
            close_dataset()
        training_logger.close()
        dist.barrier()
        dist.destroy_process_group()

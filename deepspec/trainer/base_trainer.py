import math
import os
import json

import torch
import torch.distributed as dist
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from deepspec.data import (
    CacheCollator,
    CacheDataset,
    ConversationCollator,
    validate_train_cache,
)
from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.modeling.target_adapter import (
    get_target_embeddings,
    is_multimodal_config,
    load_target_model_with_head,
)
from deepspec.utils import (
    StatelessResumableDistributedSampler,
    ensure_dir,
    init_dist,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
from deepspec.training import BF16Optimizer
from deepspec.training.loss import configure_loss_reduction_group
from deepspec.distributed import ParallelConfig, ParallelContext, apply_parallelism
from deepspec.distributed.context_parallel import FixedContextParallel
from deepspec.distributed.distributed_checkpoint import (
    TrainingProgress as DistributedTrainingProgress,
    has_distributed_checkpoint,
    load_training_checkpoint as load_distributed_training_checkpoint,
)
from deepspec.distributed.fsdp import clip_grad_norm_, gradient_sync_context
from deepspec.modeling.pure_ep import (
    get_pure_expert_modules,
    synchronize_pure_expert_gradients,
)
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    load_resume_draft_model,
    load_training_state,
    save_checkpoint,
)
import deepspec.utils.training_logger as training_logger
from deepspec.utils.hfai_suspend import SuspendController
from deepspec.utils.metrics import configure_reduction_group
from deepspec.utils.torch_profiler import build_torch_profiler


_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _load_checkpoint_tensor(checkpoint_dir: str, names: tuple[str, ...]):
    """Load one safetensors entry without constructing the full target model."""

    from safetensors import safe_open

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        for name in names:
            shard = weight_map.get(name)
            if shard is not None:
                with safe_open(
                    os.path.join(checkpoint_dir, shard),
                    framework="pt",
                    device="cpu",
                ) as handle:
                    return handle.get_tensor(name)
    tensor_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(tensor_path):
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in names:
                if name in available:
                    return handle.get_tensor(name)
    raise KeyError(f"None of {names} exists in {checkpoint_dir}.")


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
            target_model_name_or_path, checkpoint_dir, step, tensorboard_dir, exp_name
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
        self.parallel_config = ParallelConfig.from_mapping(
            self.args.train,
            world_size=self.world_size,
        )
        self.parallel = ParallelContext.build(
            self.parallel_config,
            device_type=self.device.type,
        )
        # The frozen online teacher may use a different sparse expert view
        # from the trainable draft while retaining the exact same dense rank
        # layout and CP partition. This lets a DeepSeek teacher route its 128K
        # token volume with EP without forcing sparse all-to-all into the
        # anchor-sized draft model.
        self.target_parallel_config = self.parallel_config
        self.target_parallel = self.parallel
        target_parallel_overrides = self.args.train.get("target_parallel")
        if target_parallel_overrides is not None:
            merged_target_parallel = self.parallel_config.to_dict()
            merged_target_parallel.update(dict(target_parallel_overrides))
            self.target_parallel_config = ParallelConfig.from_mapping(
                {"parallel": merged_target_parallel},
                world_size=self.world_size,
            )
            dense_dimensions = (
                "dp_replicate",
                "dp_shard",
                "cp",
                "tp",
                "pp",
            )
            changed_dense = [
                name
                for name in dense_dimensions
                if getattr(self.target_parallel_config, name)
                != getattr(self.parallel_config, name)
            ]
            if changed_dense:
                raise ValueError(
                    "train.target_parallel may override sparse target settings "
                    f"but not the shared dense layout; changed={changed_dense}."
                )
            self.target_parallel = self.parallel.with_sparse_config(
                self.target_parallel_config
            )
        print(
            "[deepspec-mesh] "
            f"global_rank={self.global_rank} "
            f"draft={self.parallel.local_group_dict()} "
            f"target={self.target_parallel.local_group_dict()}",
            flush=True,
        )
        self.context_parallel_size = self.parallel_config.cp
        self.fsdp_size = self.parallel_config.fsdp_shard_size
        self.data_parallel_size = self.parallel.data_parallel_size
        self.data_parallel_rank = self.parallel.data_parallel_rank
        reduction_group = self.parallel.loss_mesh.get_group()
        configure_loss_reduction_group(reduction_group)
        configure_reduction_group(reduction_group)
        self.fixed_context_parallel = FixedContextParallel(
            self.parallel,
            backend=self.parallel_config.context_parallel_backend,
        )
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir
        self.resume_checkpoint_dir = discover_latest_checkpoint(
            self.checkpoint_dir_root
        )
        self.suspend_controller = SuspendController(device=self.device)
        self.next_micro_step = 0
        self.online_target_enabled = bool(self.args.data.get("online_target", False))
        self.online_target = None
        if self.online_target_enabled and int(self.args.train.local_batch_size) != 1:
            raise ValueError("Online target training requires local_batch_size=1.")

        if is_global_main_process():
            ensure_dir(self.checkpoint_dir_root)
        training_logger.init(
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
        )

        self.draft_model, self.tokenizer = self.build_models()
        resume_is_distributed = bool(
            self.resume_checkpoint_dir is not None
            and has_distributed_checkpoint(self.resume_checkpoint_dir)
        )
        if self.resume_checkpoint_dir is not None and not resume_is_distributed:
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                device=self.device,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
            )
        configure_cp = getattr(self.draft_model, "configure_context_parallel", None)
        if configure_cp is None:
            if self.context_parallel_size > 1:
                raise NotImplementedError(
                    f"{type(self.draft_model).__name__} does not implement CP."
                )
        else:
            configure_cp(
                size=self.context_parallel_size,
                rank=self.parallel.context_parallel_rank,
                group=self.parallel.cp_mesh.get_group(),
                model_parallel_group=self.parallel.model_mesh.get_group(),
                model_parallel_src_rank=self.parallel.model_parallel_src_rank,
            )
        self.model = apply_parallelism(
            self.draft_model,
            self.parallel,
            self.parallel_config,
            param_dtype=self.precision_dtype,
            sequence_length=self.args.data.get("max_length"),
        )
        self._pure_expert_modules = get_pure_expert_modules(self.draft_model)

        if self.online_target_enabled:
            paths = self.args.data.get("train_data_path")
            paths = (
                [os.fspath(paths)]
                if isinstance(paths, (str, os.PathLike))
                else list(paths or [])
            )
            if not paths:
                raise ValueError(
                    "Online target training requires data.train_data_path."
                )
            self.train_dataset = JsonLineDataset(paths)
            self.data_collator = ConversationCollator(
                tokenizer=self.tokenizer,
                chat_template=self.args.data.chat_template,
                max_length=int(self.args.data.max_length),
                min_loss_tokens=int(self.args.data.get("min_loss_tokens", 1)),
            )
        else:
            self.train_dataset = CacheDataset(
                cache_dir=self.args.data.target_cache_path,
                context_parallel_size=self.context_parallel_size,
                context_parallel_rank=self.parallel.context_parallel_rank,
            )
            self.data_collator = (self.data_collator_cls or CacheCollator)()
        # Hashing a 256K packed source on every worker creates needless shared
        # filesystem traffic. Validate the complete cache identity once, then
        # broadcast any error so all ranks fail together before training.
        cache_validation_error = [None]
        if not self.online_target_enabled and is_global_main_process():
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
                f"Target cache identity validation failed: {cache_validation_error[0]}"
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
        if resume_is_distributed:
            model_config = getattr(self.draft_model.config, "to_dict", lambda: {})()
            progress = DistributedTrainingProgress(
                next_micro_step=0,
                global_step=0,
                epoch=0,
                data_position=0,
                local_batch_size=int(self.args.train.local_batch_size),
                saved_world_size=self.world_size,
                parallel_config=self.parallel_config.to_dict(),
                model_config=model_config,
            )
            progress = load_distributed_training_checkpoint(
                checkpoint_dir=self.resume_checkpoint_dir,
                model=self.model,
                optimizer_bundle=self.optimizer,
                progress=progress,
            )
            if int(progress.local_batch_size) != int(self.args.train.local_batch_size):
                raise ValueError(
                    "Resume local_batch_size mismatch: "
                    f"{progress.local_batch_size} != {self.args.train.local_batch_size}."
                )
            self.next_micro_step = int(progress.next_micro_step)
            print_on_global_main(
                f"AUTO-RESUME distributed_checkpoint from {self.resume_checkpoint_dir}, "
                f"next_micro_step={self.next_micro_step}, saved_world_size={progress.saved_world_size}."
            )
        elif self.resume_checkpoint_dir is not None:
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
            f"CP {self.context_parallel_size} x FSDP {self.fsdp_size} "
            f"(effective data replicas {self.data_parallel_size}), "
            f"draft EP {self.parallel_config.ep}, "
            f"target EP {self.target_parallel_config.ep}"
        )
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(
            f"  Global batch size = {self.args.train.global_batch_size}"
        )
        print_on_local_main(
            f"  Gradient accumulation steps = {self.gradient_accumulation_steps}"
        )
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
        if str(target_config.model_type) == "deepseek_v4":
            embed_weight = _load_checkpoint_tensor(
                model_args.target_model_name_or_path,
                (
                    "model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight",
                    "embed.weight",
                ),
            )
            head_weight = _load_checkpoint_tensor(
                model_args.target_model_name_or_path,
                ("lm_head.weight", "model.lm_head.weight", "head.weight"),
            )
            draft_model.initialize_embedding_and_head_weights(
                embed_weight=embed_weight.to(self.precision_dtype),
                lm_head_weight=head_weight.to(self.precision_dtype),
                freeze=True,
            )
        else:
            target_model = (
                load_target_model_with_head(
                    model_args.target_model_name_or_path,
                    dtype=self.precision_dtype,
                )
                .to(device="cpu")
                .eval()
            )
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

    def build_online_target(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement online target training."
        )

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
            collate_fn=self.data_collator,
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def run_batch(self, batch):
        raise NotImplementedError

    def forward_model(self, **kwargs):
        buffer_names = self.args.train.get(
            "context_parallel_buffers",
            ["input_ids", "labels", "attention_mask", "position_ids"],
        )
        sequence_dims = self.args.train.get("context_parallel_sequence_dims", {})
        buffers = []
        dims = []
        for name in buffer_names:
            value = kwargs.get(name)
            if isinstance(value, torch.Tensor):
                buffers.append(value)
                dims.append(int(sequence_dims.get(name, 1)))
        with self.fixed_context_parallel.forward_context(
            buffers=buffers,
            sequence_dims=dims,
        ):
            # Always call the module to preserve compile/FSDP pre-forward hooks.
            return self.model(**kwargs)

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
            parallel_config=self.parallel_config,
            model_config=getattr(self.draft_model.config, "to_dict", lambda: {})(),
            micro_batches_per_epoch=self.micro_batches_per_epoch,
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
        profiler = build_torch_profiler(
            self.args.get("profiling"),
            global_rank=self.global_rank,
            world_size=self.world_size,
        )
        save_checkpoints = bool(self.args.logging.get("save_checkpoints", True))
        pending_log = None

        with profiler, self.suspend_controller.monitoring():
            for batch in prefetcher:
                should_sync = (
                    self.next_micro_step + 1
                ) % self.gradient_accumulation_steps == 0
                with record_function("deepspec::training_micro_step"):
                    with gradient_sync_context(self.model, should_sync=should_sync):
                        with record_function("deepspec::forward_and_loss"):
                            loss = (
                                self.run_batch(batch) / self.gradient_accumulation_steps
                            )
                        with record_function("deepspec::backward"):
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

                    if should_sync:
                        with record_function("deepspec::expert_gradient_sync"):
                            synchronize_pure_expert_gradients(
                                self._pure_expert_modules,
                                sparse_mesh=self.parallel.sparse_mesh,
                            )

                        with record_function("deepspec::gradient_norm"):
                            grad_norm = clip_grad_norm_(
                                self.model,
                                float(self.args.train.max_grad_norm),
                                pure_expert_modules=self._pure_expert_modules,
                                expert_parallel_group=(
                                    self.parallel.expert_parallel_group
                                    if self.parallel.pure_expert_parallel
                                    else None
                                ),
                            )
                        # A packed metric reduction from the previous optimizer
                        # step has had an entire target/draft forward+backward
                        # window to complete. Drain it now, then launch this
                        # step's metrics before Adam so its collective can run
                        # concurrently with optimizer kernels and the next
                        # training step instead of extending this step's tail.
                        with record_function("deepspec::training_log_previous_wait"):
                            training_logger.finish_optimizer_step(pending_log)
                        with record_function("deepspec::training_log"):
                            pending_log = training_logger.begin_optimizer_step(
                                global_step=self.global_step,
                                next_micro_step=self.next_micro_step,
                                micro_batches_per_epoch=self.micro_batches_per_epoch,
                                max_train_steps=self.max_train_steps,
                                learning_rate=self.optimizer.get_learning_rate(),
                                grad_norm=grad_norm,
                            )
                        with record_function("deepspec::optimizer_step"):
                            self.optimizer.step()

                        if (
                            save_checkpoints
                            and self.global_step
                            % int(self.args.logging.checkpointing_steps)
                            == 0
                        ):
                            training_logger.finish_optimizer_step(pending_log)
                            pending_log = None
                            with record_function("deepspec::checkpoint"):
                                self.save_and_eval_checkpoint()

                        if self.suspend_controller.requested():
                            training_logger.finish_optimizer_step(pending_log)
                            pending_log = None
                            self._save_and_suspend()
                            profiler.step()
                            return
                profiler.step()

        # Preserve the final log record. This wait is outside the profiled
        # training-step region and, in normal runs, the final checkpoint/exit
        # is not part of throughput timing either.
        training_logger.finish_optimizer_step(pending_log)

        if save_checkpoints:
            self.save_and_eval_checkpoint()
        else:
            print_on_global_main("Checkpoint saving is disabled for this training run.")

    def clean_up(self, *, synchronize: bool = True):
        if self.online_target is not None:
            self.online_target.close()
            self.online_target = None
        close_dataset = getattr(self.train_dataset, "close", None)
        if close_dataset is not None:
            close_dataset()
        training_logger.close()
        if synchronize and dist.is_initialized():
            dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

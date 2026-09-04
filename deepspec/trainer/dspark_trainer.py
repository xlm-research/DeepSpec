import gc
import json
import os
import random
import weakref

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import record_function

from deepspec.data import CacheCollator, ConversationCollator
from deepspec.data.cuda_prefetcher import move_batch_to_device
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.distributed import apply_parallelism
from deepspec.distributed.distributed_checkpoint import (
    TrainingProgress as DistributedTrainingProgress,
    has_distributed_checkpoint,
    load_training_checkpoint as load_distributed_training_checkpoint,
    read_checkpoint_metadata,
)
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.modeling.dspark.qwen3_6 import Qwen3_6DSparkModel
from deepspec.modeling.dspark.qwen3_6.config import (
    build_draft_config as build_qwen3_6_draft_config,
)
from deepspec.modeling.dspark.qwen3_8 import Qwen3_8DSparkModel
from deepspec.modeling.dspark.qwen3_8.config import (
    build_draft_config as build_qwen3_8_draft_config,
)
from deepspec.modeling.pure_ep import get_pure_expert_modules
from deepspec.trainer.base_trainer import BaseTrainer, _compute_training_schedule
from deepspec.trainer.ckpt_manager import (
    save_checkpoint,
    validate_partition_checkpoint,
)
from deepspec.trainer.glm5_partitioned_swap import (
    Glm5PartitionCache,
    Glm5ReadyCacheLoader,
    atomic_write_json,
    build_journal_record,
    compute_glm5_training_partitions,
    journal_path,
    load_json,
    validate_journal_record,
)
from deepspec.training import BF16Optimizer
from deepspec.utils import (
    StatelessResumableDistributedSampler,
    main_process_first,
    print_on_global_main,
    print_on_local_main,
)
import deepspec.utils.training_logger as training_logger


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    def prepare_online_target_batch(self, batch):
        if (
            self.data_batch_micro_batches is not None
            and self._data_batch_phase != "target_inference"
        ):
            raise RuntimeError(
                "Target inference is only allowed during the isolated "
                "target_inference phase."
            )
        with record_function("deepspec::target_forward"):
            target_batch = self.online_target.forward_training_batch(batch)
        batch.clear()
        batch.update(target_batch)
        return batch

    # Training step.
    def run_batch(self, batch):
        if self.target_runtime_enabled and self.data_batch_micro_batches is not None:
            if self._data_batch_phase != "draft_training":
                raise RuntimeError(
                    "Draft forward is only allowed during the isolated "
                    "draft_training phase."
                )
            if "target_hidden_states" not in batch:
                raise RuntimeError(
                    "Isolated draft training requires precomputed target hidden "
                    "states; refusing to run target inference from run_batch."
                )
        elif self.target_runtime_enabled and "target_hidden_states" not in batch:
            self.prepare_online_target_batch(batch)
        needs_target_logits = (
            float(self.args.model.l1_loss_alpha) > 0.0
            or float(self.args.model.confidence_head_alpha) > 0.0
        )
        if needs_target_logits:
            target_last_hidden_states = batch["target_last_hidden_states"]
        else:
            # DFlash is CE-only, so the target model's final-layer feature is
            # unused.  Release it before the draft forward starts.
            batch.pop("target_last_hidden_states", None)
            target_last_hidden_states = None
        context_chunk_len = batch.get("context_chunk_len")
        if context_chunk_len is None:
            # Reusable full-cache records use the canonical context_len field.
            context_chunk_len = batch.get("context_len")
        with record_function("deepspec::draft_forward"):
            outputs = self.forward_model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=target_last_hidden_states,
                context_chunk_len=context_chunk_len,
                seq_len=batch["seq_len"],
            )
        with record_function("deepspec::loss"):
            loss = compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=self.args.model.loss_decay_gamma,
                ce_loss_alpha=float(self.args.model.ce_loss_alpha),
                l1_loss_alpha=float(self.args.model.l1_loss_alpha),
                confidence_head_alpha=float(self.args.model.confidence_head_alpha),
            )
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)


class Qwen3_6DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_6_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3_6DSparkModel(draft_config)


class Qwen3_8DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_8_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3_8DSparkModel(draft_config)

    def build_online_target(self):
        from deepspec.modeling.target import Qwen3_8OnlineTarget

        return Qwen3_8OnlineTarget(
            model_name_or_path=self.args.model.target_model_name_or_path,
            target_layer_ids=self.args.model.target_layer_ids,
            topology=self.target_parallel,
            device=self.device,
            rank_local_cache_dir=os.path.join(
                self.checkpoint_dir_root, "target_rank_local"
            ),
        )


class DeepseekV4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        from deepspec.modeling.dspark.deepseek_v4 import (
            DeepseekV4DSparkModel,
            build_draft_config,
        )

        return DeepseekV4DSparkModel(
            build_draft_config(target_config=target_config, model_args=model_args)
        )

    def build_online_target(self):
        from deepspec.modeling.target import DeepseekV4OnlineTarget

        return DeepseekV4OnlineTarget(
            model_name_or_path=self.args.model.target_model_name_or_path,
            target_layer_ids=self.args.model.target_layer_ids,
            topology=self.target_parallel,
            device=self.device,
            rank_local_cache_dir=os.path.join(
                self.checkpoint_dir_root, "target_rank_local"
            ),
        )

    def prepare_online_target_batch(self, batch):
        if (
            self.data_batch_micro_batches is not None
            and self._data_batch_phase != "target_inference"
        ):
            raise RuntimeError(
                "Target inference is only allowed during the isolated "
                "target_inference phase."
            )
        with record_function("deepspec::target_forward"):
            target_batch = self.online_target.forward_training_batch(batch)
        # Keep generated supervision on the outer training-loop batch so its
        # final GPU references can be dropped immediately after draft backward.
        batch.clear()
        batch.update(target_batch)
        return batch

    def run_batch(self, batch):
        if self.target_runtime_enabled and self.data_batch_micro_batches is not None:
            if self._data_batch_phase != "draft_training":
                raise RuntimeError(
                    "Draft forward is only allowed during the isolated "
                    "draft_training phase."
                )
            if "target_hidden_states" not in batch:
                raise RuntimeError(
                    "Isolated draft training requires precomputed target hidden "
                    "states; refusing to run target inference from run_batch."
                )
        elif self.target_runtime_enabled and "target_hidden_states" not in batch:
            self.prepare_online_target_batch(batch)
        needs_target_logits = (
            float(self.args.model.l1_loss_alpha) > 0.0
            or float(self.args.model.confidence_head_alpha) > 0.0
        )
        if not needs_target_logits:
            batch.pop("target_last_hidden_states", None)
        with record_function("deepspec::draft_forward"):
            outputs = self.forward_model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=batch.get("target_last_hidden_states"),
                context_start=batch["context_start"],
                context_len=batch["context_len"],
                seq_len=batch["seq_len"],
            )
        with record_function("deepspec::loss"):
            return compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=self.args.model.loss_decay_gamma,
                ce_loss_alpha=float(self.args.model.ce_loss_alpha),
                l1_loss_alpha=float(self.args.model.l1_loss_alpha),
                confidence_head_alpha=float(self.args.model.confidence_head_alpha),
            )


class Glm5NextDSparkTrainer(DeepseekV4DSparkTrainer):
    supports_partitioned_model_swap = True
    partitioned_model_swap_lifecycle = (
        "PREPARE_PARTITION",
        "TARGET_LOAD",
        "TARGET_GENERATE_FEATURES",
        "PARTITION_FEATURES_READY",
        "TARGET_UNLOAD",
        "DRAFT_LOAD",
        "DRAFT_TRAIN_PARTITION",
        "DRAFT_SAVE_CHECKPOINT",
        "DRAFT_UNLOAD",
        "PARTITION_CACHE_DELETE",
        "NEXT_PARTITION",
    )

    def validate_target_tokenizer(self, *, target_config, tokenizer) -> None:
        from deepspec.modeling.dspark.glm5_next.config import (
            validate_glm5_next_target_config,
            validate_glm5_next_tokenizer,
        )

        text_config = validate_glm5_next_target_config(target_config)
        validate_glm5_next_tokenizer(
            tokenizer,
            mask_token_id=self.args.model.mask_token_id,
        )
        if int(self.args.model.mask_token_id) >= int(text_config.vocab_size):
            raise ValueError("GLM-5.3 DSpark mask token is outside the vocabulary.")

    def _build_draft_model(self, *, target_config, model_args):
        from deepspec.modeling.dspark.glm5_next import (
            Glm5NextDSparkModel,
            build_draft_config,
        )

        return Glm5NextDSparkModel(
            build_draft_config(target_config=target_config, model_args=model_args),
            expert_parallel_size=self.parallel.expert_parallel_size,
            expert_parallel_rank=self.parallel.expert_parallel_rank,
        )

    def build_online_target(self):
        from deepspec.modeling.target import Glm5NextOnlineTarget

        return Glm5NextOnlineTarget(
            model_name_or_path=self.args.model.target_model_name_or_path,
            target_layer_ids=self.args.model.target_layer_ids,
            topology=self.target_parallel,
            device=self.device,
            rank_local_cache_dir=os.path.join(
                self.checkpoint_dir_root, "target_rank_local"
            ),
            require_phase_guard=bool(self.partitioned_model_swap_enabled),
        )

    def _initialize_partitioned_model_swap(self) -> None:
        self.draft_model = None
        self.model = None
        self.optimizer = None
        self._pure_expert_modules = ()
        self._draft_residency_verified = False
        self._active_prefetcher = None
        self._active_partition = None
        self._active_train_end_step = self.max_train_steps if hasattr(
            self, "max_train_steps"
        ) else 0
        self._ready_cache_dir = None
        self._ready_cache_manifest = None
        self._last_partition_checkpoint = None

        self.target_config, self.tokenizer = self.load_target_config_and_tokenizer()
        self._initial_draft_rng_state = self._capture_rng_state()

        paths = self.args.data.get("train_data_path")
        paths = (
            [os.fspath(paths)]
            if isinstance(paths, (str, os.PathLike))
            else list(paths or [])
        )
        if not paths:
            raise ValueError(
                "GLM partitioned model swap requires data.train_data_path."
            )
        with main_process_first():
            self.train_dataset = JsonLineDataset(
                paths,
                cache_dir=self.args.data.get("jsonl_index_cache_dir"),
            )
        self.data_collator = ConversationCollator(
            tokenizer=self.tokenizer,
            chat_template=self.args.data.chat_template,
            max_length=int(self.args.data.max_length),
            min_loss_tokens=int(self.args.data.get("min_loss_tokens", 1)),
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
        self._partitions = compute_glm5_training_partitions(
            max_samples=self.partitioned_model_swap_max_samples,
            global_batch_size=int(self.args.train.global_batch_size),
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            micro_batches_per_epoch=self.micro_batches_per_epoch,
            max_train_steps=self.max_train_steps,
        )
        self._partitions_by_id = {
            partition.partition_id: partition for partition in self._partitions
        }
        self._partitions_by_start = {
            partition.start_next_micro_step: partition
            for partition in self._partitions
        }
        self._partition_cache = Glm5PartitionCache(
            root=self.data_batch_cache_root,
            global_rank=self.global_rank,
        )
        self.data_batch_rank_cache_dir = self._partition_cache.rank_root
        self._journal_path = journal_path(self.checkpoint_dir_root)
        train_data_files = []
        for path in paths:
            absolute_path = os.path.abspath(os.fspath(path))
            file_stat = os.stat(absolute_path)
            train_data_files.append(
                {
                    "path": absolute_path,
                    "size": int(file_stat.st_size),
                    "mtime_ns": int(file_stat.st_mtime_ns),
                }
            )
        self._partition_run_identity = {
            "version": 2,
            "target_model_name_or_path": os.path.abspath(
                os.fspath(self.args.model.target_model_name_or_path)
            ),
            "train_data_files": train_data_files,
            "dataset_size": len(self.train_dataset),
            "samples_per_epoch": self.samples_per_epoch,
            "global_batch_size": int(self.args.train.global_batch_size),
            "local_batch_size": int(self.args.train.local_batch_size),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "micro_batches_per_epoch": self.micro_batches_per_epoch,
            "max_train_steps": self.max_train_steps,
            "max_samples": self.partitioned_model_swap_max_samples,
            "world_size": self.world_size,
            "seed": int(self.args.seed),
            "model": dict(self.args.model),
            "data_contract": {
                "chat_template": str(self.args.data.chat_template),
                "max_length": int(self.args.data.max_length),
                "min_loss_tokens": int(self.args.data.get("min_loss_tokens", 1)),
                "store_target_last_hidden_states": bool(
                    self.args.data.get("store_target_last_hidden_states", True)
                ),
            },
            "draft_training_contract": {
                "lr": float(self.args.train.lr),
                "warmup_ratio": float(self.args.train.warmup_ratio),
                "weight_decay": float(self.args.train.get("weight_decay", 0.0)),
                "max_grad_norm": float(self.args.train.max_grad_norm),
                "precision": str(self.args.train.precision),
                "parallel": self.parallel_config.to_dict(),
            },
            "target_parallel": self.target_parallel_config.to_dict(),
        }

        if self.target_parallel_config.tp > 1 and not self.heterogeneous_target_data_batches:
            raise ValueError(
                "GLM target TP requires the existing heterogeneous target-data-batch "
                "writer path in partitioned model-swap mode."
            )
        self.next_micro_step = 0
        if self.resume_checkpoint_dir is not None:
            metadata = self._read_checkpoint_metadata(
                self.resume_checkpoint_dir,
                require_distributed=True,
            )
            self.next_micro_step = int(metadata["next_micro_step"])
        boundaries = set(self._partitions_by_start)
        boundaries.add(self.max_train_steps * self.gradient_accumulation_steps)
        if self.next_micro_step not in boundaries:
            raise ValueError(
                "GLM partitioned model swap can only resume at a partition "
                f"boundary, got next_micro_step={self.next_micro_step}."
            )
        self._active_train_end_step = self.max_train_steps
        if self.resume_checkpoint_dir is None:
            print_on_local_main("GLM partitioned model swap starting from scratch.")
        else:
            print_on_global_main(
                "GLM partitioned model swap resume metadata: "
                f"checkpoint={self.resume_checkpoint_dir}, "
                f"next_micro_step={self.next_micro_step}."
            )

    def _capture_rng_state(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": (
                torch.cuda.get_rng_state(self.device).clone()
                if self.device.type == "cuda"
                else None
            ),
        }

    def _restore_rng_state(self, state) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if self.device.type == "cuda" and state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"], self.device)

    def _rank_zero_value(self, description, callback):
        if not dist.is_initialized():
            return callback()
        payload = [None, None]
        if self.global_rank == 0:
            try:
                payload[0] = callback()
            except Exception as exc:
                payload[1] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(payload, src=0)
        if payload[1] is not None:
            raise RuntimeError(f"{description} failed: {payload[1]}")
        return payload[0]

    def _collective_action(self, description, callback):
        result = None
        local_error = None
        try:
            result = callback()
        except Exception as exc:
            local_error = f"rank {self.global_rank}: {type(exc).__name__}: {exc}"
        if dist.is_initialized():
            errors = [None] * self.world_size
            dist.all_gather_object(errors, local_error)
        else:
            errors = [local_error]
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError(f"{description} failed collectively: {'; '.join(failures)}")
        return result

    def _set_swap_phase(self, phase: str) -> None:
        self._data_batch_phase = phase
        target = getattr(self, "online_target", None)
        set_phase = getattr(target, "set_execution_phase", None)
        if callable(set_phase):
            set_phase(phase if phase == "TARGET_GENERATE_FEATURES" else None)
        print_on_global_main(f"[deepspec-glm-partition] phase={phase}")

    def _assert_no_draft_state(self) -> None:
        def check():
            resident = [
                name
                for name in ("draft_model", "model", "optimizer")
                if getattr(self, name, None) is not None
            ]
            if resident:
                raise RuntimeError(
                    "Target load requires all draft training state to be absent; "
                    f"resident={resident}."
                )

        self._collective_action("draft-absence guard", check)

    def _assert_no_target_state(self) -> None:
        def check():
            target = getattr(self, "online_target", None)
            if target is not None:
                raise RuntimeError("Draft load requires the GLM target object to be absent.")
            released = getattr(self, "_last_target_model_weakref", None)
            if released is not None and released() is not None:
                raise RuntimeError("Released GLM target weights remain reachable.")

        self._collective_action("target-absence guard", check)

    def _write_partition_journal(
        self,
        phase: str,
        partition,
        *,
        checkpoint_dir: str | None = None,
    ) -> dict:
        record = build_journal_record(
            phase=phase,
            partition=partition,
            run_identity=self._partition_run_identity,
            checkpoint_dir=checkpoint_dir,
        )
        self._rank_zero_value(
            f"write partition journal phase {phase}",
            lambda: atomic_write_json(self._journal_path, record),
        )
        if dist.is_initialized():
            dist.barrier()
        return record

    def _load_partition_journal(self):
        return self._rank_zero_value(
            "read partition journal",
            lambda: load_json(self._journal_path),
        )

    def _read_checkpoint_metadata(self, checkpoint_dir, *, require_distributed):
        checkpoint_dir = os.path.abspath(os.fspath(checkpoint_dir))

        def read():
            if require_distributed and not has_distributed_checkpoint(checkpoint_dir):
                raise FileNotFoundError(
                    f"No complete distributed checkpoint under {checkpoint_dir}."
                )
            return read_checkpoint_metadata(checkpoint_dir)

        metadata = self._rank_zero_value("read checkpoint metadata", read)
        next_micro_step = int(metadata["next_micro_step"])
        if int(metadata["global_step"]) != (
            next_micro_step // self.gradient_accumulation_steps
        ):
            raise ValueError("Checkpoint global_step disagrees with next_micro_step.")
        if int(metadata["saved_world_size"]) != self.world_size:
            raise ValueError(
                "Partitioned model swap currently requires the checkpoint world "
                f"size to remain {self.world_size}."
            )
        return metadata

    def _completed_partition_checkpoint(self, partition):
        checkpoint_dir = os.path.join(
            self.checkpoint_dir_root,
            f"step_{partition.end_next_micro_step // self.gradient_accumulation_steps}",
        )

        def read_if_present():
            if not os.path.isdir(checkpoint_dir):
                return None
            return validate_partition_checkpoint(
                checkpoint_dir,
                partition_metadata=partition.identity(),
                next_micro_step=partition.end_next_micro_step,
            )

        metadata = self._rank_zero_value(
            "inspect completed partition checkpoint", read_if_present
        )
        if metadata is None:
            return None
        expected = {
            "partition_id": partition.partition_id,
            "partition_start_next_micro_step": partition.start_next_micro_step,
            "partition_end_next_micro_step": partition.end_next_micro_step,
            "next_micro_step": partition.end_next_micro_step,
            "checkpointed": True,
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "Completed checkpoint identity does not match the partition; "
                f"checkpoint={checkpoint_dir}."
            )
        return checkpoint_dir

    def _target_shard_layout(self) -> dict:
        return {
            "global_rank": self.global_rank,
            "world_size": self.world_size,
            "parallel_config": self.target_parallel_config.to_dict(),
            "groups": self.target_parallel.local_group_dict(),
            "tensor_parallel_rank": self.target_parallel.tensor_parallel_rank,
            "expert_parallel_rank": self.target_parallel.expert_parallel_rank,
        }

    def _generate_partition_features(self, partition, *, recovering: bool) -> dict:
        self._set_swap_phase("PREPARE_PARTITION")
        self._collective_action(
            "prepare incomplete partition cache",
            lambda: self._partition_cache.prepare_incomplete(
                partition,
                replace_matching=True,
            ),
        )
        self._assert_no_draft_state()
        self._set_swap_phase("TARGET_LOAD")
        self.online_target = self.build_online_target()
        self._collective_action(
            "target-load guard",
            lambda: (
                None
                if getattr(self.online_target, "model", None) is not None
                else (_ for _ in ()).throw(RuntimeError("GLM target did not load."))
            ),
        )
        self._set_swap_phase("TARGET_GENERATE_FEATURES")

        micro_batch_count = (
            partition.end_next_micro_step - partition.start_next_micro_step
        )
        dataloader = BaseTrainer._build_train_dataloader(
            self,
            start_offset_samples=(
                partition.start_next_micro_step
                * int(self.args.train.local_batch_size)
            ),
            num_samples=micro_batch_count * int(self.args.train.local_batch_size),
            persistent_workers=False,
        )
        dataset_indices = iter(dataloader.sampler)
        data_iterator = iter(dataloader)
        samples = []
        try:
            for local_index in range(micro_batch_count):
                try:
                    cpu_batch = next(data_iterator)
                    dataset_index = int(next(dataset_indices))
                except StopIteration as exc:
                    raise RuntimeError(
                        "Partition DataLoader ended before its planned optimizer "
                        "boundary."
                    ) from exc
                if cpu_batch is None:
                    raise ValueError(
                        f"Dataset index {dataset_index} produced no training sample."
                    )
                local_batch = move_batch_to_device(cpu_batch, self.device)
                stream_micro_step = partition.start_next_micro_step + local_index
                logical_sample_id = (
                    stream_micro_step * self.data_parallel_size
                    + self.data_parallel_rank
                )
                if self.heterogeneous_target_data_batches:
                    for owner_index in range(
                        self.target_parallel.tensor_parallel_size
                    ):
                        owner_global_rank, target_input = (
                            self._broadcast_target_input_from_tp_owner(
                                local_batch,
                                owner_index,
                            )
                        )
                        prepared = self.prepare_online_target_batch(target_input)
                        if self.global_rank == owner_global_rank:
                            samples.append(
                                self._partition_cache.write_sample(
                                    partition=partition,
                                    batch=prepared,
                                    logical_sample_id=logical_sample_id,
                                    dataset_index=dataset_index,
                                    stream_micro_step=stream_micro_step,
                                )
                            )
                        else:
                            prepared.clear()
                else:
                    prepared = self.prepare_online_target_batch(local_batch)
                    samples.append(
                        self._partition_cache.write_sample(
                            partition=partition,
                            batch=prepared,
                            logical_sample_id=logical_sample_id,
                            dataset_index=dataset_index,
                            stream_micro_step=stream_micro_step,
                        )
                    )
                local_batch.clear()
            try:
                next(data_iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("Partition DataLoader exceeded its planned range.")
        finally:
            shutdown_workers = getattr(data_iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                shutdown_workers()
            close_dataset = getattr(self.train_dataset, "close", None)
            if callable(close_dataset):
                close_dataset()
            del data_iterator
            del dataloader

        if len(samples) != micro_batch_count:
            raise RuntimeError(
                "Each draft rank must write exactly one cache record per local "
                f"micro-batch: {len(samples)} != {micro_batch_count}."
            )
        self._partition_cache.write_local_manifest(
            partition=partition,
            samples=samples,
            target_shard_layout=self._target_shard_layout(),
            state="LOCAL_COMPLETE",
        )
        self._collective_action(
            "validate local completed feature cache",
            lambda: self._partition_cache.validate_incomplete(partition),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if dist.is_initialized():
            dist.barrier()
        self._collective_action(
            "commit local READY feature cache",
            lambda: self._partition_cache.commit_ready(partition),
        )
        self._set_swap_phase("PARTITION_FEATURES_READY")
        ready_dir, manifest = self._validate_ready_partition(partition)
        self._write_partition_journal("READY", partition)
        self._ready_cache_dir = ready_dir
        self._ready_cache_manifest = manifest
        return manifest

    def _validate_ready_partition(self, partition):
        result = self._collective_action(
            "validate READY partition cache",
            lambda: self._partition_cache.validate_ready(partition),
        )
        ready_dir, manifest = result
        expected_local = (
            partition.end_next_micro_step - partition.start_next_micro_step
        ) * int(self.args.train.local_batch_size)
        if int(manifest["local_sample_count"]) != expected_local:
            raise ValueError(
                "READY manifest has the wrong local sample count: "
                f"{manifest['local_sample_count']} != {expected_local}."
            )
        samples = manifest["samples"]
        expected_stream = list(
            range(
                partition.start_next_micro_step,
                partition.end_next_micro_step,
            )
        )
        actual_stream = [int(item["stream_micro_step"]) for item in samples]
        if actual_stream != expected_stream:
            raise ValueError("READY cache training-stream positions changed.")
        expected_ids = [
            stream_micro_step * self.data_parallel_size + self.data_parallel_rank
            for stream_micro_step in expected_stream
        ]
        local_ids = [int(item["logical_sample_id"]) for item in samples]
        if local_ids != expected_ids:
            raise ValueError("READY cache logical sample identities changed.")
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.data_parallel_size,
            rank=self.data_parallel_rank,
            total_size=self.samples_per_epoch,
            start_global_offset_samples=partition.start_next_micro_step,
            num_samples=len(expected_stream),
        )
        expected_dataset_indices = list(sampler)
        actual_dataset_indices = [int(item["dataset_index"]) for item in samples]
        if actual_dataset_indices != expected_dataset_indices:
            raise ValueError("READY cache dataset indices changed.")
        expected_layout = json.loads(json.dumps(self._target_shard_layout()))
        if manifest.get("target_shard_layout") != expected_layout:
            raise ValueError("READY cache target shard layout changed.")
        if dist.is_initialized():
            ids_by_rank = [None] * self.world_size
            dist.all_gather_object(ids_by_rank, local_ids)
            all_ids = [sample_id for rank_ids in ids_by_rank for sample_id in rank_ids]
            if len(all_ids) != len(set(all_ids)):
                raise RuntimeError(
                    "TP/EP target replicas produced duplicate cache writers."
                )
        self._ready_cache_dir = ready_dir
        self._ready_cache_manifest = manifest
        return ready_dir, manifest

    def _unload_target(self) -> None:
        self._set_swap_phase("TARGET_UNLOAD")
        target = getattr(self, "online_target", None)
        if target is not None:
            target.close()
            self._last_target_model_weakref = target._released_model_weakref
            self.online_target = None
            del target
            gc.collect()
        self._assert_no_target_state()

    def _partition_start_checkpoint(self, partition):
        self._draft_resume_metadata = None
        if partition.start_next_micro_step == 0:
            return None
        checkpoint_dir = os.path.join(
            self.checkpoint_dir_root,
            f"step_{partition.start_next_micro_step // self.gradient_accumulation_steps}",
        )
        metadata = self._read_checkpoint_metadata(
            checkpoint_dir,
            require_distributed=True,
        )
        if int(metadata["next_micro_step"]) != partition.start_next_micro_step:
            raise ValueError(
                "Base draft checkpoint does not match the partition start."
            )
        self._draft_resume_metadata = metadata
        return checkpoint_dir

    def _load_draft(self, partition) -> None:
        self._assert_no_target_state()
        self._assert_no_draft_state()
        self._set_swap_phase("DRAFT_LOAD")
        checkpoint_dir = self._partition_start_checkpoint(partition)
        if checkpoint_dir is None:
            self._restore_rng_state(self._initial_draft_rng_state)

        draft_model, _tokenizer = self.build_models()
        self.draft_model = draft_model
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
        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=float(self.args.train.lr),
            total_steps=self.max_train_steps,
            warmup_ratio=float(self.args.train.warmup_ratio),
            weight_decay=float(self.args.train.weight_decay),
        )
        self._draft_residency_verified = False
        if checkpoint_dir is not None:
            progress_kwargs = {}
            if "partition_id" in self._draft_resume_metadata:
                progress_kwargs = {
                    "partition_id": -1,
                    "partition_start_next_micro_step": -1,
                    "partition_end_next_micro_step": -1,
                }
            progress = DistributedTrainingProgress(
                next_micro_step=0,
                global_step=0,
                epoch=0,
                data_position=0,
                local_batch_size=int(self.args.train.local_batch_size),
                saved_world_size=self.world_size,
                parallel_config=self.parallel_config.to_dict(),
                model_config=getattr(
                    self.draft_model.config, "to_dict", lambda: {}
                )(),
                **progress_kwargs,
            )
            progress = load_distributed_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=self.model,
                optimizer_bundle=self.optimizer,
                progress=progress,
            )
            self.next_micro_step = int(progress.next_micro_step)
            if progress.partition_id is not None and (
                not progress.checkpointed
                or progress.partition_end_next_micro_step
                != partition.start_next_micro_step
            ):
                raise ValueError(
                    "Restored partition checkpoint metadata does not end at the "
                    "next partition start."
                )
        else:
            self.next_micro_step = 0
        if self.next_micro_step != partition.start_next_micro_step:
            raise RuntimeError(
                "Loaded draft progress does not match partition start: "
                f"{self.next_micro_step} != {partition.start_next_micro_step}."
            )

    def _unload_draft(self) -> None:
        if getattr(self, "draft_model", None) is None:
            self.model = None
            self.optimizer = None
            self._pure_expert_modules = ()
            return
        self._set_swap_phase("DRAFT_UNLOAD")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if dist.is_initialized():
            dist.barrier()
        before = self._cuda_memory_snapshot()

        draft = self.draft_model
        model = getattr(self, "model", None)
        model_reference = weakref.ref(model) if model is not None else None
        draft_reference = weakref.ref(draft)
        module_references = [weakref.ref(module) for module in draft.modules()]
        parameter_references = [
            weakref.ref(parameter) for parameter in draft.parameters()
        ]
        optimizer = getattr(self, "optimizer", None)
        parameter = None
        for parameter in draft.parameters():
            parameter.grad = None
        if optimizer is not None:
            optimizer.optimizer.state.clear()
            optimizer.optimizer.param_groups.clear()
            if hasattr(optimizer.scheduler, "optimizer"):
                optimizer.scheduler.optimizer = None
        self.optimizer = None
        self._pure_expert_modules = ()
        self.model = None
        self.draft_model = None
        self._ready_cache_loader = None
        parameter = None
        optimizer = None
        model = None
        del draft
        gc.collect()
        reachable_modules = sum(
            reference() is not None for reference in module_references
        )
        reachable_parameters = sum(
            reference() is not None for reference in parameter_references
        )
        if (
            (model_reference is not None and model_reference() is not None)
            or draft_reference() is not None
            or reachable_modules
            or reachable_parameters
        ):
            raise RuntimeError(
                "Draft teardown left model/FSDP training state reachable: "
                f"modules={reachable_modules}, parameters={reachable_parameters}."
            )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
            torch.cuda.synchronize(self.device)
        if dist.is_initialized():
            dist.barrier()
        after = self._cuda_memory_snapshot()
        print(
            "[deepspec-glm-draft-unload] "
            f"rank={self.global_rank} allocated_before={before['allocated']} "
            f"reserved_before={before['reserved']} "
            f"allocated_after={after['allocated']} "
            f"reserved_after={after['reserved']} model_reachable=false",
            flush=True,
        )

    def _cuda_memory_snapshot(self):
        if self.device.type != "cuda":
            return {"allocated": 0, "reserved": 0}
        return {
            "allocated": int(torch.cuda.memory_allocated(self.device)),
            "reserved": int(torch.cuda.memory_reserved(self.device)),
        }

    def _build_train_dataloader(self, *args, **kwargs):
        if getattr(self, "partitioned_model_swap_enabled", False) and (
            self._data_batch_phase in (
                "DRAFT_TRAIN_PARTITION",
                "DRAFT_SAVE_CHECKPOINT",
            )
        ):
            if self._ready_cache_dir is None or self._ready_cache_manifest is None:
                raise RuntimeError("Draft training requires a validated READY cache.")
            loader = Glm5ReadyCacheLoader(
                ready_dir=self._ready_cache_dir,
                manifest=self._ready_cache_manifest,
            )
            expected = self._active_partition.end_next_micro_step - (
                self._active_partition.start_next_micro_step
            )
            if len(loader) != expected:
                raise RuntimeError(
                    f"READY cache length changed: {len(loader)} != {expected}."
                )
            self._ready_cache_loader = loader
            return loader
        return super()._build_train_dataloader(*args, **kwargs)

    def prepare_online_target_batch(self, batch):
        if (
            getattr(self, "partitioned_model_swap_enabled", False)
            and self._data_batch_phase != "TARGET_GENERATE_FEATURES"
        ):
            raise RuntimeError(
                "GLM target forward is only allowed during "
                "TARGET_GENERATE_FEATURES."
            )
        return super().prepare_online_target_batch(batch)

    def forward_model(self, **kwargs):
        if (
            getattr(self, "partitioned_model_swap_enabled", False)
            and self._data_batch_phase != "DRAFT_TRAIN_PARTITION"
        ):
            raise RuntimeError(
                "GLM draft forward is only allowed during "
                "DRAFT_TRAIN_PARTITION."
            )
        return super().forward_model(**kwargs)

    def run_batch(self, batch):
        if (
            getattr(self, "partitioned_model_swap_enabled", False)
            and self._data_batch_phase != "DRAFT_TRAIN_PARTITION"
        ):
            raise RuntimeError(
                "GLM draft forward/backward is only allowed during "
                "DRAFT_TRAIN_PARTITION."
            )
        return super().run_batch(batch)

    def save_and_eval_checkpoint(self):
        if not getattr(self, "partitioned_model_swap_enabled", False):
            return super().save_and_eval_checkpoint()
        partition = self._active_partition
        if partition is None or self.next_micro_step != partition.end_next_micro_step:
            raise RuntimeError(
                "A partition checkpoint may only be saved at its planned end."
            )
        self._set_swap_phase("DRAFT_SAVE_CHECKPOINT")
        checkpoint_dir = save_checkpoint(
            **self._checkpoint_kwargs(),
            partition_metadata=partition.identity(),
            atomic_commit=True,
        )
        self._last_partition_checkpoint = checkpoint_dir
        return checkpoint_dir

    def _train_ready_partition(self, partition, *, journal_is_training: bool) -> str:
        self._validate_ready_partition(partition)
        self._unload_target()
        self._load_draft(partition)
        self._active_partition = partition
        self._active_train_end_step = (
            partition.end_next_micro_step // self.gradient_accumulation_steps
        )
        self._last_partition_checkpoint = None
        self._set_swap_phase("DRAFT_TRAIN_PARTITION")
        if not journal_is_training:
            self._write_partition_journal("TRAINING", partition)
        BaseTrainer.train(self)
        if self.next_micro_step != partition.end_next_micro_step:
            raise RuntimeError("Draft training stopped before the partition boundary.")
        checkpoint_dir = self._last_partition_checkpoint
        if checkpoint_dir is None:
            raise RuntimeError("Partition training finished without a checkpoint.")
        self._write_partition_journal(
            "CHECKPOINTED",
            partition,
            checkpoint_dir=checkpoint_dir,
        )
        return checkpoint_dir

    def _cleanup_checkpointed_partition(self, partition, checkpoint_dir: str) -> None:
        completed = self._completed_partition_checkpoint(partition)
        if completed is None or os.path.realpath(completed) != os.path.realpath(
            checkpoint_dir
        ):
            raise ValueError(
                "CHECKPOINTED journal does not match a complete partition checkpoint."
            )
        self.next_micro_step = partition.end_next_micro_step
        self._unload_draft()
        self._set_swap_phase("PARTITION_CACHE_DELETE")
        self._collective_action(
            "atomically mark checkpointed partition cache for deletion",
            lambda: self._partition_cache.begin_delete(partition),
        )
        self._collective_action(
            "delete checkpointed partition cache",
            lambda: self._partition_cache.finish_delete(partition),
        )
        self._ready_cache_dir = None
        self._ready_cache_manifest = None
        self._active_partition = None
        self._write_partition_journal(
            "CLEANED",
            partition,
            checkpoint_dir=completed,
        )
        self._set_swap_phase("NEXT_PARTITION")

    def train(self):
        if not getattr(self, "partitioned_model_swap_enabled", False):
            return super().train()
        final_micro_step = self.max_train_steps * self.gradient_accumulation_steps
        journal = self._load_partition_journal()
        if self.next_micro_step > 0 and journal is None:
            raise ValueError(
                "A nonzero GLM partition checkpoint requires its matching "
                "partition_journal.json; refusing to guess recovery state."
            )
        if self.next_micro_step > 0:
            completed_partitions = [
                partition
                for partition in self._partitions_by_id.values()
                if partition.end_next_micro_step == self.next_micro_step
            ]
            if len(completed_partitions) != 1:
                raise ValueError(
                    "Resume progress does not identify exactly one completed "
                    "GLM partition."
                )
            completed_checkpoint = self._completed_partition_checkpoint(
                completed_partitions[0]
            )
            if completed_checkpoint is None:
                raise ValueError(
                    "step_latest does not contain a complete checkpoint for its "
                    "GLM partition."
                )
            resume_checkpoint = getattr(self, "resume_checkpoint_dir", None)
            if resume_checkpoint is not None and os.path.realpath(
                resume_checkpoint
            ) != os.path.realpath(completed_checkpoint):
                raise ValueError(
                    "step_latest points to a checkpoint with the wrong GLM "
                    "partition identity."
                )
        while True:
            if journal is not None:
                partition_value = journal.get("partition", {})
                partition_id = int(partition_value.get("partition_id", -1))
                partition = self._partitions_by_id.get(partition_id)
                if partition is None:
                    raise ValueError(
                        f"Partition journal refers to unknown partition {partition_id}."
                    )
                validate_journal_record(
                    journal,
                    partition=partition,
                    run_identity=self._partition_run_identity,
                )
                phase = journal.get("phase")
                if phase == "CLEANED":
                    if self.next_micro_step != partition.end_next_micro_step:
                        raise ValueError(
                            "CLEANED journal disagrees with checkpoint progress."
                        )
                    journal = None
                    continue
                if phase == "TRAINING":
                    if self.next_micro_step not in (
                        partition.start_next_micro_step,
                        partition.end_next_micro_step,
                    ):
                        raise ValueError(
                            "TRAINING journal disagrees with checkpoint progress."
                        )
                    completed = self._completed_partition_checkpoint(partition)
                    if completed is not None:
                        self.next_micro_step = partition.end_next_micro_step
                        journal = self._write_partition_journal(
                            "CHECKPOINTED",
                            partition,
                            checkpoint_dir=completed,
                        )
                        phase = "CHECKPOINTED"
                    elif self.next_micro_step != partition.start_next_micro_step:
                        raise ValueError(
                            "TRAINING progress reached the partition end without "
                            "a matching complete checkpoint."
                        )
                if phase == "CHECKPOINTED":
                    if self.next_micro_step != partition.end_next_micro_step:
                        raise ValueError(
                            "CHECKPOINTED journal disagrees with checkpoint progress."
                        )
                    checkpoint_dir = journal.get("checkpoint_dir")
                    if not checkpoint_dir:
                        raise ValueError(
                            "CHECKPOINTED journal is missing checkpoint_dir."
                        )
                    self._cleanup_checkpointed_partition(
                        partition,
                        checkpoint_dir,
                    )
                    journal = None
                    continue
                if self.next_micro_step != partition.start_next_micro_step:
                    raise ValueError(
                        f"Journal phase {phase} requires progress at partition start."
                    )
                if phase == "GENERATING":
                    self._generate_partition_features(partition, recovering=True)
                    journal = self._load_partition_journal()
                    phase = "READY"
                elif phase == "READY":
                    self._validate_ready_partition(partition)
                elif phase == "TRAINING":
                    self._validate_ready_partition(partition)
                else:
                    raise ValueError(f"Unsupported partition journal phase: {phase!r}")
                checkpoint_dir = self._train_ready_partition(
                    partition,
                    journal_is_training=phase == "TRAINING",
                )
                self._cleanup_checkpointed_partition(partition, checkpoint_dir)
                journal = None
                continue

            if self.next_micro_step >= final_micro_step:
                break
            partition = self._partitions_by_start.get(self.next_micro_step)
            if partition is None:
                raise ValueError(
                    "Training progress is not at a planned GLM partition boundary: "
                    f"{self.next_micro_step}."
                )
            journal = self._write_partition_journal("GENERATING", partition)
            self._generate_partition_features(partition, recovering=False)
            checkpoint_dir = self._train_ready_partition(
                partition,
                journal_is_training=False,
            )
            self._cleanup_checkpointed_partition(partition, checkpoint_dir)
            journal = None

        self._data_batch_phase = None
        print_on_global_main(
            "GLM partitioned model swap training complete: "
            f"next_micro_step={self.next_micro_step}, global_step={self.global_step}."
        )

    def _clean_up_partitioned_model_swap(
        self, *, synchronize: bool = True
    ) -> None:
        target = getattr(self, "online_target", None)
        if target is not None:
            target.close()
            self.online_target = None
        if getattr(self, "draft_model", None) is not None:
            self._unload_draft()
        close_dataset = getattr(getattr(self, "train_dataset", None), "close", None)
        if callable(close_dataset):
            close_dataset()
        training_logger.close()
        if synchronize and dist.is_initialized():
            dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

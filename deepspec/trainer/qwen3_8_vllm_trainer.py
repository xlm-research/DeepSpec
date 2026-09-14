"""Qwen-only vLLM teacher backend, preserving the existing DSpark update loop."""

from dataclasses import asdict, replace
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import record_function

from deepspec.utils.distributed import StatelessResumableDistributedSampler
from deepspec.data.cuda_prefetcher import move_batch_to_device
from deepspec.data.draft_feature_reader import DraftFeatureIndex, feature_input_identity
from deepspec.trainer.dspark_trainer import Qwen3_8DSparkTrainer
from deepspec.distributed.distributed_checkpoint import TrainingProgress
from deepspec.trainer.draft_phase_checkpoint import (
    discover_draft_phase_checkpoint,
    save_draft_phase_checkpoint,
    validate_draft_phase_resume,
)
from deepspec.trainer.glm5_partitioned_swap import atomic_write_json, load_json
from deepspec.trainer.qwen3_8_vllm import (
    QwenVllmConfig,
    run_worker_process,
    teacher_identity,
)


def replica_layout(
    *, global_rank, local_rank, local_size, cp_size, tp_size, vllm_tp, devices
):
    """One vLLM replica serves one producer CP x TP group on the same node."""
    group_size = cp_size * tp_size
    if local_size % group_size or vllm_tp > group_size or group_size % vllm_tp:
        raise ValueError(
            "Producer CP x TP groups must fit on one node and be divisible by vLLM TP."
        )
    if len(devices) != local_size or not 0 <= local_rank < local_size:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain one device per local rank.")
    start = local_rank // group_size * group_size
    owner = global_rank - local_rank + start
    return (
        owner,
        [owner + rank * tp_size for rank in range(cp_size)],
        devices[start : start + vllm_tp],
    )


class Qwen3_8VllmDSparkTrainer(Qwen3_8DSparkTrainer):
    """Extract a partition in a subprocess, then train with its immutable features.

    The producer keeps its fixed layout while draft consumers read indexed
    features. Every normal phase ends after a complete optimizer update; the
    existing per-epoch sample stream and accumulation windows are preserved.
    """

    optimizer_aligned_data_partitions = True

    _data_batch_phase: str | None
    _active_data_batch_cache: list[str] | tuple[str, ...] | None

    def __init__(self, local_rank, args):
        if (
            args.data.get("online_target", False)
            or not args.data.get("offline_target_data_batches", False)
            or args.data.get("multimodal", False)
            or (args.train.get("partitioned_model_swap") or {}).get("enabled", False)
        ):
            raise ValueError(
                "Qwen vLLM requires text-only bounded offline target batches."
            )
        self.qwen_vllm_config = QwenVllmConfig(**args.train.qwen_vllm)
        super().__init__(local_rank, args)
        self._vllm_control_group = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=self.qwen_vllm_config.timeout_seconds + 300),
        )
        local_size = int(os.environ["LOCAL_WORLD_SIZE"])
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = (
            visible_devices.split(",")
            if visible_devices
            else [str(i) for i in range(local_size)]
        )
        self._vllm_owner, self._cp_cache_owners, self._vllm_devices = replica_layout(
            global_rank=self.global_rank,
            local_rank=local_rank,
            local_size=local_size,
            cp_size=self.target_parallel_config.cp,
            tp_size=self.target_parallel_config.tp,
            vllm_tp=self.qwen_vllm_config.tensor_parallel_size,
            devices=devices,
        )
        if self._vllm_owner != self.target_parallel.model_parallel_src_rank:
            raise ValueError(
                "Qwen vLLM owner does not match the producer model-parallel mesh."
            )
        self._bind_producer_identity()
        self._active_draft_feature_index = None
        self._last_draft_phase_checkpoint = None
        print(
            f"[qwen38-vllm] rank={self.global_rank} owner={self._vllm_owner} "
            f"teacher_tp={self.qwen_vllm_config.tensor_parallel_size} "
            f"draft_cp={self.context_parallel_size} draft_tp={self.parallel_config.tp} "
            "teacher_layers=64 final_state=full_model_final_norm_output",
            flush=True,
        )

    def _validate_heterogeneous_target_data_batch_layout(self):
        # Both meshes already validate against the world size. Indexed files
        # support independent consumers without the native teacher's TP scatter.
        return

    def discover_resume_checkpoint(self):
        return discover_draft_phase_checkpoint(self.checkpoint_dir_root)

    def load_resume_checkpoint(self, progress):
        metadata = validate_draft_phase_resume(
            self.resume_checkpoint_dir, train_config=self.args, progress=progress
        )
        # DCP requests only keys present in the Stateful load template.
        progress.partition_id = -1
        progress.partition_start_next_micro_step = -1
        progress.partition_end_next_micro_step = -1
        progress = super().load_resume_checkpoint(progress)

        def validate_progress():
            for name in (
                "next_micro_step",
                "global_step",
                "epoch",
                "data_position",
                "partition_id",
                "partition_start_next_micro_step",
                "partition_end_next_micro_step",
                "checkpointed",
                "saved_world_size",
                "parallel_config",
                "model_config",
            ):
                if json.loads(json.dumps(getattr(progress, name))) != metadata[name]:
                    raise ValueError(f"Draft resume DCP {name} progress mismatch.")

        self._collective_stage(validate_progress)
        return progress

    def _build_train_dataloader(self, *args, **kwargs):
        # Producer owners collate the fixed global sample stream on CPU. Draft
        # consumers receive ready features, so no draft input prefetch is needed.
        return ()

    def _bind_producer_identity(self):
        executable = shutil.which(self.qwen_vllm_config.python_executable)
        if executable is None:
            raise ValueError("The fixed vLLM producer interpreter does not exist.")
        source = self.qwen_vllm_config.source_dir
        if source is None:
            source = str(Path(__file__).resolve().parents[2] / "vllm")
        source = str(Path(source).resolve())
        if not (Path(source) / "vllm").is_dir():
            raise ValueError("The fixed vLLM producer source checkout does not exist.")
        self.qwen_vllm_config = replace(
            self.qwen_vllm_config,
            python_executable=os.path.abspath(executable),
            source_dir=source,
        )
        mapping = [None] * self.world_size
        dist.all_gather_object(
            mapping,
            {
                "rank": self.global_rank,
                "owner": self._vllm_owner,
                "cache_owners": self._cp_cache_owners,
                "devices": self._vllm_devices,
            },
            group=self._vllm_control_group,
        )
        self._producer_identity = {
            "teacher": self._teacher_identity,
            "inference": asdict(self.qwen_vllm_config),
            "layout": {
                name: getattr(self.target_parallel_config, name)
                for name in ("dp_replicate", "dp_shard", "cp", "tp", "pp")
            },
            "mapping": mapping,
            "sample_plan": {
                "seed": 42,
                "samples_per_epoch": self.samples_per_epoch,
                "global_batch_size": int(self.args.train.global_batch_size),
                "max_length": int(self.args.data.max_length),
                "chat_template": self.args.data.get("chat_template"),
                "min_loss_tokens": int(self.args.data.get("min_loss_tokens", 1)),
                "sources": [
                    {
                        "path": str(Path(path).resolve()),
                        "size": Path(path).stat().st_size,
                        "mtime_ns": Path(path).stat().st_mtime_ns,
                    }
                    for path in self.train_dataset.data_paths
                ],
            },
        }

        def persist():
            if self.global_rank != 0:
                return
            path = Path(self.checkpoint_dir_root) / "qwen38_vllm_producer.json"
            previous = load_json(path)
            checkpoint_identity = (
                self._checkpoint_producer_identity()
                if self.resume_checkpoint_dir is not None
                else None
            )
            if any(
                identity is not None and identity != self._producer_identity
                for identity in (previous, checkpoint_identity)
            ):
                raise ValueError("The fixed Qwen producer configuration changed.")
            atomic_write_json(path, self._producer_identity)

        self._collective_stage(persist)

    def _checkpoint_producer_identity(self):
        checkpoint = Path(self.resume_checkpoint_dir)
        return load_json(checkpoint / "draft_feature_index.json")["producer_identity"]

    def _build_draft_model(self, *, target_config, model_args):
        self._teacher_identity = teacher_identity(
            model_args.target_model_name_or_path, list(model_args.target_layer_ids)
        )
        identity_path = Path(self.checkpoint_dir_root) / "qwen38_vllm_teacher.json"
        error = None
        if self.global_rank == 0:
            try:
                previous = load_json(identity_path)
                checkpoint_identity = (
                    self._checkpoint_producer_identity()["teacher"]
                    if self.resume_checkpoint_dir is not None
                    else None
                )
                if any(
                    identity is not None and identity != self._teacher_identity
                    for identity in (previous, checkpoint_identity)
                ):
                    raise ValueError(
                        "The Qwen vLLM checkpoint directory belongs to a different teacher."
                    )
                atomic_write_json(identity_path, self._teacher_identity)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise ValueError(errors[0])
        draft = super()._build_draft_model(
            target_config=target_config, model_args=model_args
        )
        draft.config.deepspec_target_backend = "vllm"
        draft.config.deepspec_target_execution_num_hidden_layers = 64
        draft.config.deepspec_target_final_hidden_source = self._teacher_identity[
            "target_final_hidden_source"
        ]
        return draft

    def build_online_target(self):
        # The subprocess owns the entire teacher; no native teacher is allocated.
        return None

    def prepare_online_target_batch(self, batch):
        raise RuntimeError(
            "Qwen vLLM extraction runs only at a target partition boundary."
        )

    def _collective_stage(self, action):
        error = None
        result = None
        try:
            result = action()
        except Exception as exc:
            error = f"rank {self.global_rank}: {type(exc).__name__}: {exc}"
        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(
            errors, error, group=getattr(self, "_vllm_control_group", None)
        )
        if any(errors):
            raise RuntimeError(
                "Qwen vLLM partition failed: " + "; ".join(e for e in errors if e)
            )
        return result

    def iter_training_batches(self, batches):
        if self.data_batch_micro_batches is None:
            raise RuntimeError("Qwen vLLM requires a bounded data partition schedule.")
        end_step = getattr(self, "_active_train_end_step", self.max_train_steps)
        if end_step is None:
            raise RuntimeError("Qwen draft training requires a bounded update count.")
        end_micro_step = int(end_step) * self.gradient_accumulation_steps
        for count in self.data_batch_micro_batches:
            if self.next_micro_step >= end_micro_step:
                break
            count = min(count, end_micro_step - self.next_micro_step)
            self._data_batch_phase = "target_inference"
            start_micro_step = self.next_micro_step
            # A phase keeps its identity when a fresh process starts at its cursor.
            partition_index = start_micro_step // self.gradient_accumulation_steps + 1
            start_position = start_micro_step * self.data_parallel_size
            end_position = start_position + count * self.data_parallel_size
            job_dir = None
            samples = []
            self._active_data_batch_cache = []

            def prepare():
                nonlocal job_dir
                if self.global_rank != self._vllm_owner:
                    return
                sampler = StatelessResumableDistributedSampler(
                    self.train_dataset,
                    num_replicas=1,
                    rank=0,
                    total_size=self.samples_per_epoch,
                    start_global_offset_samples=start_position,
                    num_samples=end_position - start_position,
                )
                requests: list[dict[str, Any]] = []
                for position, dataset_index in enumerate(sampler, start=start_position):
                    if (
                        position % self.target_parallel.data_parallel_size
                        != self.target_parallel.data_parallel_rank
                    ):
                        continue
                    if job_dir is None:
                        job_dir = Path(
                            tempfile.mkdtemp(
                                prefix="qwen38-job-", dir=self.data_batch_rank_cache_dir
                            )
                        )
                    batch = self.data_collator([self.train_dataset[dataset_index]])
                    if batch["input_ids"].shape[0] != 1 or not bool(
                        batch["attention_mask"].bool().all()
                    ):
                        raise ValueError(
                            "Qwen vLLM requires one unpadded text sequence per request."
                        )
                    input_path = job_dir / f"input_{len(requests):08d}.pt"
                    torch.save(
                        {
                            "input_ids": batch["input_ids"].long(),
                            "loss_mask": batch["loss_mask"],
                        },
                        input_path,
                    )
                    paths = [
                        self._data_batch_cache_file_path(
                            batch_index=partition_index,
                            sample_index=len(requests),
                            cache_rank=rank,
                        )
                        for rank in self._cp_cache_owners
                    ]
                    requests.append(
                        {"input_path": str(input_path), "output_paths": paths}
                    )
                    epoch = position // self.samples_per_epoch
                    samples.append(
                        {
                            "position": position,
                            "sample_id": f"{epoch}:{dataset_index}",
                            "input_identity": feature_input_identity(batch),
                            "epoch": epoch,
                            "shards": [
                                {"path": path, "owner": owner, "cp_rank": cp_rank}
                                for cp_rank, (path, owner) in enumerate(
                                    zip(paths, self._cp_cache_owners)
                                )
                            ],
                        }
                    )
                if job_dir is not None:
                    atomic_write_json(
                        job_dir / "job.json",
                        {
                            "model_path": self.args.model.target_model_name_or_path,
                            "teacher": self._teacher_identity,
                            "config": asdict(self.qwen_vllm_config),
                            "max_length": int(self.args.data.max_length),
                            "requests": requests,
                        },
                    )

            self._collective_stage(prepare)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

            def generate():
                if job_dir is None:
                    return
                job_path = job_dir / "job.json"
                print(
                    f"[qwen38-vllm] partition={partition_index} samples={len(samples)} job={job_path}",
                    flush=True,
                )
                run_worker_process(job_path, self.qwen_vllm_config, self._vllm_devices)
                complete = load_json(str(job_path) + ".complete")
                if (
                    complete is None
                    or complete.get("teacher") != self._teacher_identity
                    or len(complete.get("samples", [])) != len(samples)
                ):
                    raise RuntimeError(
                        "Qwen vLLM partition completion record is missing or inconsistent."
                    )
                atomic_write_json(
                    Path(self.checkpoint_dir_root)
                    / f"vllm_rank{self._vllm_owner}_partition{partition_index}.json",
                    complete,
                )
                shutil.rmtree(job_dir)

            self._collective_stage(generate)
            self._data_batch_phase = "draft_training"
            with record_function("deepspec::draft_feature_index"):
                gathered: list[list[dict[str, Any]]] = [
                    [] for _ in range(self.world_size)
                ]
                dist.all_gather_object(
                    gathered, samples, group=self._vllm_control_group
                )
                index_path = (
                    Path(self.checkpoint_dir_root)
                    / "draft_feature_indexes"
                    / f"micro_{start_micro_step:012d}.json"
                )

                def write_index():
                    if self.global_rank != 0:
                        return
                    index = DraftFeatureIndex.create(
                        samples=[sample for records in gathered for sample in records],
                        producer_identity=self._producer_identity,
                        partition_id=partition_index,
                        start_micro_step=start_micro_step,
                        data_parallel_size=self.data_parallel_size,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        samples_per_epoch=self.samples_per_epoch,
                    )
                    index_path.parent.mkdir(parents=True, exist_ok=True)
                    index.save(index_path)

                self._collective_stage(write_index)
                index = self._collective_stage(
                    lambda: DraftFeatureIndex.load(
                        index_path,
                        producer_identity=self._producer_identity,
                        next_micro_step=self.next_micro_step,
                        data_parallel_size=self.data_parallel_size,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                    )
                )
            self._active_draft_feature_index = index
            owned_paths = index.owned_paths(self.global_rank)
            self._active_data_batch_cache = owned_paths
            yield from self.iter_ready_features(index)
            if bool(self.args.logging.get("save_checkpoints", True)):
                self.save_and_eval_checkpoint()
            # iter_ready_features waits for every consumer after the last update.
            self._collective_stage(lambda: self._delete_data_batch_cache(owned_paths))
            self._active_data_batch_cache = None
            self._active_draft_feature_index = None
            self._data_batch_phase = None

    def save_and_eval_checkpoint(self):
        # Ordinary Qwen saves follow complete phases. BaseTrainer's periodic
        # and final calls share this entry without forcing HF exports.
        index = self._active_draft_feature_index
        if index is None or self.next_micro_step != index.end_micro_step:
            return self._last_draft_phase_checkpoint
        if (
            self._last_draft_phase_checkpoint is not None
            and Path(self._last_draft_phase_checkpoint).name
            == f"step_{self.global_step}"
        ):
            return self._last_draft_phase_checkpoint
        progress = TrainingProgress(
            next_micro_step=self.next_micro_step,
            global_step=self.global_step,
            epoch=self.next_micro_step // self.micro_batches_per_epoch,
            data_position=self.next_micro_step * int(self.args.train.local_batch_size),
            local_batch_size=int(self.args.train.local_batch_size),
            saved_world_size=self.world_size,
            parallel_config=self.parallel_config.to_dict(),
            model_config=self.draft_model.config.to_dict(),
            partition_id=index.partition_id,
            partition_start_next_micro_step=index.start_micro_step,
            partition_end_next_micro_step=index.end_micro_step,
            checkpointed=True,
        )
        with record_function("deepspec::draft_phase_checkpoint"):
            checkpoint = save_draft_phase_checkpoint(
                checkpoint_dir_root=self.checkpoint_dir_root,
                model=self.model,
                optimizer_bundle=self.optimizer,
                progress=progress,
                train_config=self.args,
                feature_index=index,
                control_group=self._vllm_control_group,
            )
        self._last_draft_phase_checkpoint = checkpoint
        return checkpoint

    def _save_and_suspend(self):
        self.save_and_eval_checkpoint()
        dist.barrier(group=self._vllm_control_group)
        if self.global_rank == 0:
            self.suspend_controller.go_suspend()
        dist.barrier(group=self._vllm_control_group)

    def iter_ready_features(self, index: DraftFeatureIndex):
        """Feed an indexed producer phase to the retained Qwen update loop."""
        if index.data_parallel_size != self.parallel.data_parallel_size:
            raise ValueError(
                "Draft feature index has a different data-parallel layout."
            )
        if index.gradient_accumulation_steps != self.gradient_accumulation_steps:
            raise ValueError(
                "Draft feature index has a different accumulation schedule."
            )
        if not index.start_micro_step <= self.next_micro_step <= index.end_micro_step:
            raise ValueError("Draft progress is outside the ready feature index.")
        self._data_batch_phase = "draft_training"
        for micro_step in range(self.next_micro_step, index.end_micro_step):
            if self.next_micro_step != micro_step:
                raise RuntimeError(
                    "Draft progress no longer matches its feature index."
                )

            def read():
                cpu_batch = index.read(
                    micro_step=micro_step,
                    data_parallel_rank=self.parallel.data_parallel_rank,
                    context_parallel_size=self.parallel_config.cp,
                    context_parallel_rank=self.parallel.context_parallel_rank,
                )
                return move_batch_to_device(cpu_batch, self.device)

            # Coordinate file/identity failures before any peer enters the next
            # forward/backward collective with a different or missing sample.
            with record_function("deepspec::draft_feature_read"):
                batch = self._collective_stage(read)
            self._data_batch_end_after_current = micro_step + 1 == index.end_micro_step
            yield batch
            self._data_batch_end_after_current = False
            del batch
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self._vllm_control_group)
        self._data_batch_phase = None
